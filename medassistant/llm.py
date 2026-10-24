"""OpenAI-compatible client: chat, SSE token streaming, embeddings.

Transport-level retries only (HTTP 429/5xx, timeouts, connection errors — up to
`attempts` tries with linear backoff). Semantic failures (empty/invalid model
output) are raised to the caller and never retried here.
"""

import json
import threading
import time

import httpx

from . import config
from .loopguard import LoopGuard

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# JSON-string GBNF fragments shared by schema_to_grammar below (llama.cpp
# grammar mode, owner decision 2026-10-18: llama-server's response_format
# json_schema/json_object crashes on Gemma-style vocabs — the autoparser
# grammar cannot accept the " model" leading-space piece of the generation
# header — while a user-supplied "grammar" field is exempt from the buggy
# prompt prefill and constrains generation identically).
_GBNF_CHAR = ('( [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" '
              '(["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] '
              '[0-9a-fA-F] [0-9a-fA-F]) )')
_GBNF_STRING = f'string ::= "\\"" {_GBNF_CHAR}* "\\""'
_GBNF_WS = 'ws ::= [ \\t\\n]*'
# This pinned llama.cpp build refuses a GBNF {0,N} repeat with N > 1000
# ("Failed to initialize samplers: failed to parse grammar"; live
# 2026-10-20: maxLength 700 OK, 1001 FAIL, bisected max accepted = 1000).
# Bounded strings longer than this are emitted as adjacent repeats that
# sum to the cap (see schema_to_grammar).
_GBNF_MAX_REPEAT = 1000


def schema_to_grammar(schema, root="root"):
    """Minimal JSON-Schema -> GBNF for llama.cpp's request-level "grammar"
    field. Supports exactly the constructs the labeler schemas use — flat
    objects of required array-of-string properties (pipeline.EXTRACT_SCHEMA /
    SELECT_SCHEMA), plus scalars for future variants. Raises ValueError on
    anything else; the caller then falls back to response_format.
    """
    def rule(s):
        if "enum" in s:
            # enum of strings only (the labeler's candidate ids); the GBNF
            # literal must contain the JSON quotes themselves (\"id\")
            if not all(isinstance(v, str) for v in s["enum"]):
                raise ValueError("grammar mode supports string enums only")
            alts = [v.replace("\\", "\\\\").replace('"', '\\"')
                    for v in s["enum"]]
            return "(" + " | ".join(f'"\\"{a}\\""' for a in alts) + ")"
        t = s.get("type")
        if t == "string":
            max_len = s.get("maxLength")
            if isinstance(max_len, int) and max_len >= 0:
                # bounded string (owner decision 2026-10-20): an unbounded
                # GBNF string can absorb the whole token budget — a
                # degenerate reasoning field repeated itself to
                # truncation and the object never closed. The grammar
                # itself caps the field, so the JSON always parses. This
                # pinned llama.cpp build refuses a {0,N} repeat with
                # N > 1000 ("Failed to initialize samplers: failed to
                # parse grammar"; live 2026-10-20: maxLength 700 OK,
                # 1001 FAIL, bisected max accepted = 1000), so a bigger
                # cap is emitted as two adjacent bounded repeats summing
                # to the cap — char{0,1000} char{0,rest} bounds the
                # string identically (SELECT_REASONING_MAXLEN 1400
                # fits); the fall-through return below serves caps
                # within one repeat
                head = min(max_len, _GBNF_MAX_REPEAT)
                tail = max_len - head
                if tail > 0:
                    return (f'"\\"" {_GBNF_CHAR}{{0,{head}}} {_GBNF_CHAR}'
                            f'{{0,{tail}}} "\\""')
                return (f'"\\"" {_GBNF_CHAR}{{0,{max_len}}} "\\""')
            return "string"
        if t == "integer":
            return '"-"? [0-9]+'
        if t == "number":
            return '"-"? [0-9]+ ("." [0-9]+)? (("e"|"E") ("+"|"-")? [0-9]+)?'
        if t == "boolean":
            return '("true"|"false")'
        if t == "array":
            item = rule(s.get("items") or {"type": "string"})
            max_items = s.get("maxItems")
            if isinstance(max_items, int) and max_items >= 1:
                # bounded repetition: a greedy-degenerate labeler cannot
                # append ids forever — the grammar itself caps the array
                # (2026-10-18: an unbounded picks array looped until
                # truncation and the unclosed object failed to parse)
                return f'"[" ws ({item} (ws "," ws {item}){{0,{max_items - 1}}})? ws "]"'
            return f'"[" ws ({item} (ws "," ws {item})*)? ws "]"'
        if t == "object":
            props = s.get("properties") or {}
            req = s.get("required") or []
            if not props or any(p not in req for p in props):
                raise ValueError("grammar mode needs all-required properties")
            parts = []
            for k, v in props.items():
                parts.append(f'ws "\\"{k}\\"" ws ":" ws {rule(v)}')
            return '"{" ' + ' ws "," '.join(parts) + ' ws "}"'
        raise ValueError(f"unsupported schema type: {t!r}")

    body = rule(schema)
    return f"{root} ::= {body}\n{_GBNF_STRING}\n{_GBNF_WS}"


class LLMError(RuntimeError):
    pass


class LLMLoopError(LLMError):
    """Degenerate repetition in pre-content reasoning — the watchdog
    aborted the stream (owner decision 2026-10-06). Raising out of the
    httpx.stream context closes the connection and cancels generation.
    `tail` carries the last LOOP_TAIL_CHARS chars of the looped reasoning
    (2026-10-24) so the audit rows keep the content; the raise site in
    chat_stream is its only producer."""

    def __init__(self, message, tail=None):
        super().__init__(message)
        self.tail = tail


class LLMCancelled(LLMError):
    """User-requested stop of a streaming completion (owner decision
    2026-10-15): the StreamCancel flag fired or the connection was dropped
    by the stopper thread. Never retried."""


class StreamCancel:
    """Cancellation handle for chat_stream: a flag plus one attached close
    callback (the live httpx response).

    The stopper thread calls set() — it both flags the stop and drops the
    connection,     so llama-server cancels generation at once even while the reader
    is blocked in prefill, before the first streamed token. The reader
    thread attaches/detaches the callback around the response lifetime; a
    close racing in from set() is harmless (closing twice is a no-op).
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._close = None

    def is_set(self):
        return self._event.is_set()

    def set(self):
        self._event.set()
        with self._lock:
            close, self._close = self._close, None
        if close is not None:
            try:
                close()
            except Exception:
                pass  # the flag still stops the reader between events

    def attach(self, close):
        """Register the close callback; fires immediately when already set."""
        fire = None
        with self._lock:
            if self._event.is_set():
                fire = close
            else:
                self._close = close
        if fire is not None:
            try:
                fire()
            except Exception:
                pass

    def detach(self):
        with self._lock:
            self._close = None


def _headers(endpoint):
    h = {"Content-Type": "application/json"}
    if endpoint.get("api_key"):
        h["Authorization"] = f"Bearer {endpoint['api_key']}"
    return h


def _payload(endpoint, **kw):
    payload = {"model": endpoint["model"], **kw}
    if endpoint.get("extra_body"):
        payload.update(endpoint["extra_body"])
    return payload


def _sleep(i):
    time.sleep(min(2 * (i + 1), 8))


def chat(endpoint, messages, temperature=0.0, max_tokens=2048,
          json_schema=None, json_mode=False, timeout=300, attempts=4):
    """Non-streaming completion -> content string."""
    kw = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_schema is not None:
        if config.STRUCTURED_OUTPUT == "grammar":
            # llama.cpp grammar mode: schema -> GBNF in the request-level
            # "grammar" field (user grammars skip the buggy autoparser
            # prefill; see schema_to_grammar). Falls back to response_format
            # when the schema uses constructs the converter does not cover.
            try:
                kw["grammar"] = schema_to_grammar(json_schema)
            except ValueError:
                kw["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "out", "schema": json_schema,
                                    "strict": True},
                }
        elif config.STRUCTURED_OUTPUT == "response_format":
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": json_schema,
                                "strict": True},
            }
        # STRUCTURED_OUTPUT=none: no response_format, no grammar — the
        # labeler thinks and llm.parse_json_object extracts the object
    elif json_mode:
        kw["response_format"] = {"type": "json_object"}
    url = f"{endpoint['base_url']}/chat/completions"
    body = json.dumps(_payload(endpoint, **kw), ensure_ascii=False)
    last = None
    for i in range(attempts):
        try:
            r = httpx.post(url, content=body.encode("utf-8"),
                           headers=_headers(endpoint), timeout=timeout)
            if r.status_code in RETRYABLE_STATUS:
                last = LLMError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
                _sleep(i)
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
            data = r.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise LLMError(
                    f"malformed chat response: {json.dumps(data, ensure_ascii=False)[:300]}"
                ) from e
        except LLMError:
            raise
        except (httpx.HTTPError, OSError) as e:
            last = LLMError(f"{type(e).__name__} from {url}: {e}")
        _sleep(i)
    raise last


def chat_stream(endpoint, messages, temperature=0.0, max_tokens=2048,
                timeout=600, attempts=4, freq_penalty=None, cancel=None):
    """Streaming completion -> generator of typed events:
    {"type": "reasoning", "text": ...} (thinking models) and
    {"type": "content", "text": ...}.

    Retried only until the first event has been yielded; after that, a transport
    error surfaces as LLMError (a partial answer cannot be replayed).

    While no content token has arrived, a LoopGuard watches the reasoning
    stream; the same unit repeated back-to-back (consecutive run) aborts
    the request with LLMLoopError (retrying mid-content would duplicate
    streamed text — only the reasoning phase is guarded). `freq_penalty`
    adds an anti-repetition nudge for the retry attempt (answerer only).

    `cancel` (StreamCancel, owner decision 2026-10-15) aborts the request
    with LLMCancelled: set() drops the connection (so a reader blocked in
    prefill wakes at once — llama-server cancels generation), and the flag is
    re-checked between events. A cancelled request is never retried.
    """
    kw = {"messages": messages, "temperature": temperature,
          "max_tokens": max_tokens, "stream": True}
    url = f"{endpoint['base_url']}/chat/completions"
    payload = _payload(endpoint, **kw)
    if freq_penalty is not None:  # retry nudge overrides extra_body
        payload["frequency_penalty"] = freq_penalty
    body = json.dumps(payload, ensure_ascii=False)
    guard = LoopGuard(min_unit=config.LOOP_MIN_UNIT,
                      repeats=config.LOOP_REPEATS,
                      window_chars=config.LOOP_WINDOW_CHARS,
                      tail_chars=config.LOOP_TAIL_CHARS) \
        if config.LOOP_GUARD else None
    last = None
    saw_content = False
    for i in range(attempts):
        if cancel is not None and cancel.is_set():
            raise LLMCancelled("cancelled by user")
        got_event = False
        try:
            with httpx.stream("POST", url, content=body.encode("utf-8"),
                              headers=_headers(endpoint), timeout=timeout) as r:
                if r.status_code in RETRYABLE_STATUS:
                    r.read()
                    last = LLMError(f"HTTP {r.status_code} from {url}")
                else:
                    r.raise_for_status()
                    if cancel is not None:
                        cancel.attach(r.close)
                    try:
                        for line in r.iter_lines():
                            if cancel is not None and cancel.is_set():
                                raise LLMCancelled("cancelled by user")
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                return
                            try:
                                delta = json.loads(data)["choices"][0]["delta"]
                            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                                continue
                            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                            if reasoning:
                                if guard is not None and not saw_content:
                                    reason = guard.feed(reasoning)
                                    if reason:
                                        raise LLMLoopError(
                                            f"degenerate reasoning loop: {reason}",
                                            tail=guard.tail)
                                got_event = True
                                yield {"type": "reasoning", "text": reasoning}
                            content = delta.get("content")
                            if content:
                                saw_content = True
                                got_event = True
                                yield {"type": "content", "text": content}
                    finally:
                        if cancel is not None:
                            cancel.detach()
                    return
        except LLMError as e:
            if cancel is not None and cancel.is_set():
                raise LLMCancelled("cancelled by user") from e
            if got_event or isinstance(e, LLMLoopError):
                raise
            last = e
        except (httpx.HTTPError, OSError) as e:
            if cancel is not None and cancel.is_set():
                raise LLMCancelled("cancelled by user") from e
            if got_event:
                raise LLMError(f"{type(e).__name__} mid-stream from {url}: {e}")
            last = LLMError(f"{type(e).__name__} from {url}: {e}")
        except Exception as e:
            # a cross-thread close can surface as an unexpected internal
            # error in the reader — a set flag means it was the stop
            if cancel is not None and cancel.is_set():
                raise LLMCancelled("cancelled by user") from e
            raise
        _sleep(i)
    raise last


def embed(endpoint, texts, batch=64, timeout=120, attempts=4):
    """Embeddings -> list of float lists."""
    out = []
    url = f"{endpoint['base_url']}/embeddings"
    for start in range(0, len(texts), batch):
        chunk_texts = texts[start:start + batch]
        body = json.dumps(
            _payload(endpoint, input=chunk_texts), ensure_ascii=False)
        last = None
        for i in range(attempts):
            try:
                r = httpx.post(url, content=body.encode("utf-8"),
                               headers=_headers(endpoint), timeout=timeout)
                if r.status_code in RETRYABLE_STATUS:
                    last = LLMError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
                    _sleep(i)
                    continue
                if r.status_code >= 400:
                    raise LLMError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
                data = r.json()
                try:
                    rows = sorted(data["data"], key=lambda d: d["index"])
                    out.extend(d["embedding"] for d in rows)
                except (KeyError, TypeError) as e:
                    raise LLMError(
                        f"malformed embeddings response: "
                        f"{json.dumps(data, ensure_ascii=False)[:300]}"
                    ) from e
                break
            except LLMError:
                raise
            except (httpx.HTTPError, OSError) as e:
                last = LLMError(f"{type(e).__name__} from {url}: {e}")
            _sleep(i)
        else:
            raise last
    return out


def parse_json_object(text):
    """Extract the first JSON object from a model reply -> dict.

    Tolerates code fences and surrounding prose; raises LLMError when no object.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"no JSON object in response: {text[:300]}")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"invalid JSON in response: {text[:300]}") from e

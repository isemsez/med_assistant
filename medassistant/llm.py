"""OpenAI-compatible client: chat, SSE token streaming, embeddings.

Transport-level retries only (HTTP 429/5xx, timeouts, connection errors — up to
`attempts` tries with linear backoff). Semantic failures (empty/invalid model
output) are raised to the caller and never retried here.
"""

import json
import time

import httpx

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


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
        kw["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": json_schema, "strict": True},
        }
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
                timeout=600, attempts=4):
    """Streaming completion -> generator of typed events:
    {"type": "reasoning", "text": ...} (thinking models) and
    {"type": "content", "text": ...}.

    Retried only until the first event has been yielded; after that, a transport
    error surfaces as LLMError (a partial answer cannot be replayed).
    """
    url = f"{endpoint['base_url']}/chat/completions"
    body = json.dumps(
        _payload(endpoint, messages=messages, temperature=temperature,
                 max_tokens=max_tokens, stream=True),
        ensure_ascii=False,
    )
    last = None
    for i in range(attempts):
        got_event = False
        try:
            with httpx.stream("POST", url, content=body.encode("utf-8"),
                              headers=_headers(endpoint), timeout=timeout) as r:
                if r.status_code in RETRYABLE_STATUS:
                    r.read()
                    last = LLMError(f"HTTP {r.status_code} from {url}")
                else:
                    r.raise_for_status()
                    for line in r.iter_lines():
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
                            got_event = True
                            yield {"type": "reasoning", "text": reasoning}
                        content = delta.get("content")
                        if content:
                            got_event = True
                            yield {"type": "content", "text": content}
                    return
        except LLMError as e:
            if got_event:
                raise
            last = e
        except (httpx.HTTPError, OSError) as e:
            if got_event:
                raise LLMError(f"{type(e).__name__} mid-stream from {url}: {e}")
            last = LLMError(f"{type(e).__name__} from {url}: {e}")
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

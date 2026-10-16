import json
import socket
import threading
import time
import unittest

import httpx

from medassistant import llm


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


def ok(content):
    return FakeResp(payload={"choices": [{"message": {"content": content}}]})


class TestParseJsonObject(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(llm.parse_json_object('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(llm.parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_prose_wrapped(self):
        self.assertEqual(llm.parse_json_object('Вот: {"a": 1} — конец.'), {"a": 1})

    def test_none_and_empty(self):
        with self.assertRaises(llm.LLMError):
            llm.parse_json_object(None)
        with self.assertRaises(llm.LLMError):
            llm.parse_json_object("no json here")

    def test_invalid_json(self):
        with self.assertRaises(llm.LLMError):
            llm.parse_json_object("{broken")


class TestChat(unittest.TestCase):
    EP = {"base_url": "http://x", "api_key": "k", "model": "m", "extra_body": {}}

    def setUp(self):
        self._orig_sleep = llm._sleep
        llm._sleep = lambda i: None
        self._orig_post = httpx.post
        self.requests = []

    def tearDown(self):
        llm._sleep = self._orig_sleep
        httpx.post = self._orig_post

    def _patch(self, responses):
        seq = list(responses)

        def fake_post(url, **kw):
            self.requests.append((url, kw))
            return seq.pop(0)

        httpx.post = fake_post

    def test_content(self):
        self._patch([ok("привет")])
        self.assertEqual(llm.chat(self.EP, [{"role": "user", "content": "q"}]), "привет")

    def test_payload_fields(self):
        self._patch([ok("x")])
        llm.chat(self.EP, [{"role": "user", "content": "q"}],
                 max_tokens=7, json_mode=True)
        url, kw = self.requests[0]
        self.assertEqual(url, "http://x/chat/completions")
        body = json.loads(kw["content"])
        self.assertEqual(body["model"], "m")
        self.assertEqual(body["max_tokens"], 7)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(kw["headers"]["Authorization"], "Bearer k")

    def test_retry_on_429_then_ok(self):
        self._patch([FakeResp(429), ok("after retry")])
        self.assertEqual(llm.chat(self.EP, []), "after retry")
        self.assertEqual(len(self.requests), 2)

    def test_non_retryable_status_raises(self):
        self._patch([FakeResp(400, text="bad format")])
        with self.assertRaises(llm.LLMError):
            llm.chat(self.EP, [])

    def test_malformed_raises(self):
        self._patch([FakeResp(payload={"no": "choices"})])
        with self.assertRaises(llm.LLMError):
            llm.chat(self.EP, [])

    def test_exhausted_retries(self):
        self._patch([FakeResp(500)] * 4)
        with self.assertRaises(llm.LLMError):
            llm.chat(self.EP, [], attempts=4)


class TestChatStream(unittest.TestCase):
    EP = {"base_url": "http://x", "api_key": "k", "model": "m", "extra_body": {}}

    def setUp(self):
        self._orig_sleep = llm._sleep
        llm._sleep = lambda i: None
        self._orig_stream = httpx.stream
        self.last_kw = {}

    def tearDown(self):
        llm._sleep = self._orig_sleep
        httpx.stream = self._orig_stream

    def _patch(self, lines):
        class FakeStreamResp:
            status_code = 200

            def __init__(self):
                self._lines = list(lines)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def raise_for_status(self):
                pass

            def read(self):
                return b""

            def iter_lines(self):
                return iter(self._lines)

        def fake(method, url, **kw):
            self.last_kw = kw
            return FakeStreamResp()

        httpx.stream = fake

    def test_reasoning_and_content_events(self):
        self._patch([
            'data: {"choices":[{"delta":{"reasoning":"думаю "}}]}',
            ': keep-alive comment',
            'data: {"choices":[{"delta":{"reasoning":"ещё"}}]}',
            'data: {"choices":[{"delta":{"content":"Привет"}}]}',
            'data: {"choices":[{"delta":{}}]}',
            'data: [DONE]',
        ])
        events = list(llm.chat_stream(self.EP, []))
        self.assertEqual(events, [
            {"type": "reasoning", "text": "думаю "},
            {"type": "reasoning", "text": "ещё"},
            {"type": "content", "text": "Привет"},
        ])

    def test_reasoning_content_fallback_key(self):
        self._patch([
            'data: {"choices":[{"delta":{"reasoning_content":"rc"}}]}',
            'data: [DONE]',
        ])
        self.assertEqual(list(llm.chat_stream(self.EP, [])),
                         [{"type": "reasoning", "text": "rc"}])

    # -- thinking-loop watchdog (owner decision 2026-10-06) ----------------

    def _loop_lines(self):
        phrase = "негиперпигментационные, негиперкератотические, "
        lines = ['data: {"choices":[{"delta":{"reasoning":"мысль. "}}]}']
        for _ in range(8):
            lines.append("data: " + json.dumps(
                {"choices": [{"delta": {"reasoning": phrase}}]},
                ensure_ascii=False))
        lines.append("data: [DONE]")
        return lines

    def test_loop_aborts_pre_content(self):
        self._patch(self._loop_lines())
        events = []
        with self.assertRaises(llm.LLMLoopError):
            for ev in llm.chat_stream(self.EP, []):
                events.append(ev)
        self.assertTrue(events)  # some reasoning streamed before the trip

    def test_no_abort_after_content(self):
        # a loop after the first answer token must not abort the stream
        lines = ['data: {"choices":[{"delta":{"content":"Ответ"}}]}']
        lines += self._loop_lines()[1:]
        self._patch(lines)
        events = list(llm.chat_stream(self.EP, []))
        self.assertEqual(events[0], {"type": "content", "text": "Ответ"})
        self.assertEqual(len([e for e in events
                              if e["type"] == "reasoning"]), 8)

    def test_loop_guard_disabled(self):
        from medassistant import config
        orig = config.LOOP_GUARD
        config.LOOP_GUARD = False
        try:
            self._patch(self._loop_lines())
            events = list(llm.chat_stream(self.EP, []))
        finally:
            config.LOOP_GUARD = orig
        self.assertEqual(len(events), 9)

    def test_freq_penalty_in_payload_and_overrides_extra_body(self):
        self._patch(["data: [DONE]"])
        list(llm.chat_stream(self.EP, [], freq_penalty=0.5))
        self.assertEqual(json.loads(self.last_kw["content"])
                         ["frequency_penalty"], 0.5)
        ep = {**self.EP, "extra_body": {"frequency_penalty": 0.1}}
        self._patch(["data: [DONE]"])
        list(llm.chat_stream(ep, [], freq_penalty=0.5))
        self.assertEqual(json.loads(self.last_kw["content"])
                         ["frequency_penalty"], 0.5)

    def test_no_freq_penalty_by_default(self):
        self._patch(["data: [DONE]"])
        list(llm.chat_stream(self.EP, []))
        self.assertNotIn("frequency_penalty",
                         json.loads(self.last_kw["content"]))


class _SSEServer:
    """Minimal real HTTP/SSE server on 127.0.0.1 for the cancel tests: sends
    response headers, then either stays silent (the prefill window) or
    streams `data:` lines on a chunked body. Sets .client_gone when the peer
    (httpx) drops the connection; counts accepted requests (no-retry check).
    """

    LINE = 'data: {"choices":[{"delta":{"content":"ток"}}]}\n\n'.encode()

    def __init__(self, silent_s=0.0, line_interval_s=0.03, lines=400):
        self.silent_s = silent_s
        self.line_interval_s = line_interval_s
        self.n_lines = lines
        self.requests = 0
        self.client_gone = threading.Event()
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        self.port = self._sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def close(self):
        self._sock.close()

    def _serve(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return  # listener closed — test over
            self.requests += 1
            try:
                self._handle(conn)
            except OSError:  # reset while idle — the drop we watch for
                self.client_gone.set()
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle(self, conn):
        conn.settimeout(20)
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = conn.recv(4096)
            if not chunk:
                return
            head += chunk
        head, _, rest = head.partition(b"\r\n\r\n")
        # drain the request body: leftover body bytes must not leak into the
        # silence phase's EOF watch below
        clen = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                clen = int(line.split(b":", 1)[1])
        while len(rest) < clen:
            chunk = conn.recv(4096)
            if not chunk:
                return
            rest += chunk
        conn.sendall(b"HTTP/1.1 200 OK\r\n"
                     b"Content-Type: text/event-stream\r\n"
                     b"Transfer-Encoding: chunked\r\n\r\n")
        if self.silent_s:
            # prefill window: nothing flows; a FIN from the client shows up
            # as an empty recv, a reset as OSError — either is the drop
            conn.settimeout(self.silent_s + 15)
            if conn.recv(1) == b"":
                self.client_gone.set()
            return
        for _ in range(self.n_lines):
            if self.client_gone.is_set():
                return
            try:
                conn.sendall(b"%x\r\n" % len(self.LINE) + self.LINE + b"\r\n")
            except OSError:
                self.client_gone.set()
                return
            time.sleep(self.line_interval_s)
        conn.sendall(b"0\r\n\r\n")  # end of the chunked body


class TestStreamCancel(unittest.TestCase):
    """StreamCancel + LLMCancelled (owner decision 2026-10-15): set() drops
    the live connection so a reader blocked in prefill wakes at once, and a
    cancelled request is never retried."""

    def _endpoint(self, port):
        return {"base_url": f"http://127.0.0.1:{port}", "api_key": "",
                "model": "m", "extra_body": {}}

    def _stream_in_thread(self, port, cancel):
        out = {}

        def run():
            try:
                out["events"] = list(llm.chat_stream(
                    self._endpoint(port), [{"role": "user", "content": "q"}],
                    attempts=4, cancel=cancel))
            except Exception as e:  # the expected LLMCancelled lands here
                out["error"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t, out

    def test_cancel_wakes_blocked_reader_and_drops_connection(self):
        # prefill window: headers, then silence; cancel must raise
        # LLMCancelled long before the silence ends, and the server-side
        # peer must see the connection drop (Ollama then cancels generation)
        srv = _SSEServer(silent_s=8.0)
        try:
            cancel = llm.StreamCancel()
            t, out = self._stream_in_thread(srv.port, cancel)
            time.sleep(0.5)  # the request reached the silent read
            t0 = time.monotonic()
            cancel.set()
            t.join(timeout=5)
            self.assertFalse(t.is_alive(), "cancel must wake the blocked reader")
            self.assertLess(time.monotonic() - t0, 4)
            self.assertIsInstance(out.get("error"), llm.LLMCancelled)
            self.assertTrue(srv.client_gone.wait(timeout=10),
                            "the generation-side peer must see the drop")
        finally:
            srv.close()

    def test_cancel_between_lines_not_retried(self):
        # lines flow; a cancel between two events stops at once and the
        # transport makes exactly one request — no retry after a stop
        srv = _SSEServer()
        try:
            cancel = llm.StreamCancel()
            t, out = self._stream_in_thread(srv.port, cancel)
            time.sleep(0.3)  # a few events yielded already
            cancel.set()
            t.join(timeout=5)
            self.assertFalse(t.is_alive())
            self.assertIsInstance(out.get("error"), llm.LLMCancelled)
            self.assertEqual(srv.requests, 1)
            self.assertTrue(srv.client_gone.wait(timeout=10))
        finally:
            srv.close()

    def test_set_before_attach_fires_on_attach(self):
        cancel = llm.StreamCancel()
        cancel.set()
        fired = []
        cancel.attach(lambda: fired.append(1))
        self.assertEqual(fired, [1])

    def test_detach_removes_close(self):
        cancel = llm.StreamCancel()
        fired = []
        cancel.attach(lambda: fired.append(1))
        cancel.detach()
        cancel.set()
        self.assertEqual(fired, [])
        self.assertTrue(cancel.is_set())


class TestPayload(unittest.TestCase):
    EP = {"model": "m", "extra_body": {}}

    def test_extra_body_merge(self):
        p = llm._payload({"model": "m", "extra_body": {"thinking": {"type": "disabled"}}},
                         messages=[], temperature=0.0, max_tokens=5)
        self.assertEqual(p["thinking"], {"type": "disabled"})
        self.assertEqual(p["max_tokens"], 5)

    def test_json_schema_kw(self):
        schema = {"type": "object"}
        p = llm._payload(
            self.EP, messages=[],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "out", "schema": schema}})
        self.assertEqual(p["response_format"]["json_schema"]["schema"], schema)


if __name__ == "__main__":
    unittest.main()


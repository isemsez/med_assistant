import json
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


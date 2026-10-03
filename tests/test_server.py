"""Server API/SSE integration tests: fully offline.

config caches paths at import time, so this module reloads config+server with
env overrides pointing SESSIONS_DB/AUDIT_LOG/CLINREC_DIR at a temp dir, and
patches load_pipeline with a fake — no real data assets or endpoints loaded.
"""

import importlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

from fastapi.testclient import TestClient

from medassistant import llm

PDF_FILE = "777_1_Тестовая рекомендация.pdf"
TMP = None
srv = None


def setUpModule():
    global TMP, srv
    TMP = Path(tempfile.mkdtemp())
    (TMP / "pdf").mkdir()
    (TMP / "pdf" / PDF_FILE).write_bytes(b"%PDF-1.4 test")
    # docs_meta + index fixtures so /api/health stays fully offline
    (TMP / "docs_meta.json").write_text(
        json.dumps([{"code_version": "777_1", "name": "Тестовая рекомендация",
                     "url": "https://x/1", "file": PDF_FILE}], ensure_ascii=False),
        encoding="utf-8")
    index_dir = TMP / "index"
    index_dir.mkdir()
    docs = {"777_1": {"code_version": "777_1", "name": "Тестовая рекомендация",
                      "url": "https://x/1", "file": PDF_FILE}}
    from medassistant import gindex
    (index_dir / "meta.json").write_text(
        json.dumps({"dim": 4, "rows": [{"code_version": "777_1",
                                        "name": "Тестовая рекомендация",
                                        "text": "тест"}],
                    "docs_meta_sha1": gindex.docs_hash(docs),
                    "docs_meta_count": 1}, ensure_ascii=False),
        encoding="utf-8")
    os.environ["SESSIONS_DB"] = str(TMP / "sessions.sqlite")
    os.environ["AUDIT_LOG"] = str(TMP / "audit.jsonl")
    os.environ["CLINREC_DIR"] = str(TMP / "pdf")
    os.environ["DOCS_META"] = str(TMP / "docs_meta.json")
    os.environ["INDEX_DIR"] = str(index_dir)
    import medassistant.config
    import medassistant.server
    importlib.reload(medassistant.config)
    importlib.reload(medassistant.server)
    srv = medassistant.server


def tearDownModule():
    for k in ("SESSIONS_DB", "AUDIT_LOG", "CLINREC_DIR", "DOCS_META", "INDEX_DIR"):
        os.environ.pop(k, None)
    import medassistant.config
    import medassistant.server
    importlib.reload(medassistant.config)
    importlib.reload(medassistant.server)
    shutil.rmtree(TMP, ignore_errors=True)


def parse_sse(text):
    """'event: x\\ndata: {...}\\n\\n...' -> [(event, data_dict), ...]."""
    out = []
    for block in text.split("\n\n"):
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event is not None:
            out.append((event, data))
    return out


class FakePipe:
    """Stands in for the pipeline: two recordable generators."""

    def __init__(self, answer_events=None, appendix_events=None,
                 appendix_error=None):
        self.answer_events = answer_events or [
            {"type": "reasoning", "text": "размышление"},
            {"type": "content", "text": "Ответ про ХСН."},
        ]
        self.appendix_events = appendix_events
        self.appendix_error = appendix_error
        self.calls = []

    def stream_answer(self, question, history=None):
        self.calls.append(("stream", question, list(history or [])))
        if isinstance(self.answer_events, Exception):
            raise self.answer_events
        yield from self.answer_events

    def iter_appendix(self, question, raw_answer, session_id=None):
        from medassistant.pipeline import AppendixResult
        self.calls.append(("appendix", question, raw_answer))
        if self.appendix_error is not None:
            raise self.appendix_error
        result = AppendixResult(
            "ok", diseases=["ХСН"],
            by_disease=[{"disease": "ХСН",
                         "picks": [{"code_version": "777_1", "name": "Тестовая рекомендация",
                                    "url": "/data/pdf/" + quote(PDF_FILE)}],
                         "auto_pick": False}],
            appendix_text="Клинические рекомендации Минздрава России (для проверки):")
        yield {"stage": "extract"}
        yield {"stage": "search", "disease": "ХСН"}
        yield {"stage": "done", "result": result}


def make_client(pipe):
    """Entered TestClient (startup ran) + its context manager for teardown."""
    orig = srv.load_pipeline
    srv.load_pipeline = lambda audit=None, with_index=True: pipe
    try:
        cm = TestClient(srv.app)
        client = cm.__enter__()
    finally:
        srv.load_pipeline = orig
    return client, cm


class Base(unittest.TestCase):
    def setUp(self):
        self.pipe = FakePipe()
        self.client, self._cm = make_client(self.pipe)

    def tearDown(self):
        self._cm.__exit__(None, None, None)

    def chat(self, question="Вопрос про ХСН?", session_id=None):
        if session_id is None:
            session_id = self.client.post("/api/sessions",
                                          json={"title": ""}).json()["id"]
        r = self.client.post("/api/chat", json={"session_id": session_id,
                                                "question": question})
        self.assertEqual(r.status_code, 200)
        return session_id, parse_sse(r.text)


class TestSessions(Base):
    def test_create_list_delete(self):
        sid = self.client.post("/api/sessions", json={"title": ""}).json()["id"]
        sessions = self.client.get("/api/sessions").json()
        self.assertIn(sid, [s["id"] for s in sessions])
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/messages").json(), [])
        self.assertEqual(self.client.delete(f"/api/sessions/{sid}").json(), {"ok": True})
        self.assertEqual(self.client.delete(f"/api/sessions/{sid}").status_code, 404)

    def test_unknown_session_404(self):
        self.assertEqual(
            self.client.get("/api/sessions/999/messages").status_code, 404)
        self.assertEqual(
            self.client.post("/api/chat",
                             json={"session_id": 999, "question": "q"}).status_code, 404)


class TestChat(Base):
    def test_happy_path_event_order(self):
        sid, events = self.chat()
        names = [e for e, _ in events]
        self.assertEqual(names[0], "reasoning")
        self.assertIn("token", names)
        i_done = names.index("done")
        self.assertGreater(names.index("stage"), i_done)  # composer unblocks first
        self.assertGreater(names.index("appendix"), names.index("stage"))
        done = dict(events)[ "done"]
        self.assertIsNotNone(done["message_id"])
        self.assertIsNotNone(done["user_message_id"])
        appendix = dict(events)["appendix"]
        self.assertEqual(appendix["status"], "ok")

    def test_thought_prefix_dropped_across_deltas(self):
        self.pipe.answer_events = [
            {"type": "reasoning", "text": "thou"},
            {"type": "reasoning", "text": "ght\nмысль"},
            {"type": "content", "text": "Ответ."},
        ]
        sid, events = self.chat()
        reasoning = "".join(d["t"] for e, d in events if e == "reasoning")
        self.assertEqual(reasoning, "мысль")
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual(msgs[1]["reasoning"], "мысль")

    def test_history_passed_to_answerer(self):
        sid, _ = self.chat("Первый вопрос?")
        self.chat("Второй вопрос?", session_id=sid)
        stream_calls = [c for c in self.pipe.calls if c[0] == "stream"]
        self.assertEqual(len(stream_calls), 2)
        second_history = stream_calls[1][2]
        self.assertEqual([m["role"] for m in second_history], ["user", "assistant"])
        self.assertEqual(second_history[0]["content"], "Первый вопрос?")

    def test_messages_and_appendix_persisted(self):
        sid, _ = self.chat()
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[0]["content"], "Вопрос про ХСН?")
        self.assertEqual(msgs[1]["appendix"]["status"], "ok")
        self.assertEqual(msgs[1]["appendix"]["by_disease"][0]["picks"][0]
                         ["code_version"], "777_1")

    def test_answerer_error(self):
        self.pipe.answer_events = llm.LLMError("HTTP 500 from http://x")
        sid, events = self.chat()
        names = [e for e, _ in events]
        self.assertIn("error", names)
        done = dict(events)["done"]
        self.assertIsNone(done["message_id"])
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual([m["role"] for m in msgs], ["user"])  # no assistant row

    def test_empty_answer(self):
        self.pipe.answer_events = [{"type": "reasoning", "text": "только мысль"}]
        sid, events = self.chat()
        done = dict(events)["done"]
        self.assertIsNone(done["message_id"])
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual(len(msgs), 1)

    def test_appendix_crash_streams_error(self):
        self.pipe.appendix_error = RuntimeError("machinery bug")
        sid, events = self.chat()
        names = [e for e, _ in events]
        self.assertIn("error", names)
        self.assertNotIn("appendix", names)
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual(msgs[1]["appendix"], None)  # answer itself survived

    def test_empty_question_400(self):
        sid = self.client.post("/api/sessions", json={"title": ""}).json()["id"]
        r = self.client.post("/api/chat", json={"session_id": sid,
                                                "question": "   "})
        self.assertEqual(r.status_code, 400)

    def test_rewind(self):
        sid, _ = self.chat()
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        assistant_id = msgs[1]["id"]
        r = self.client.delete(f"/api/sessions/{sid}/messages/{assistant_id}")
        self.assertEqual(r.json(), {"deleted": 1})
        self.assertEqual(
            len(self.client.get(f"/api/sessions/{sid}/messages").json()), 1)
        self.assertEqual(self.client.delete(
            f"/api/sessions/{sid}/messages/{assistant_id}").status_code, 404)


class TestStatic(Base):
    def test_index_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Мед.ассистент", r.text)

    def test_pdf_served(self):
        r = self.client.get("/data/pdf/" + quote(PDF_FILE))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content[:8], b"%PDF-1.4")


class TestHealthRoute(Base):
    def test_health_shape_and_local_checks(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(sorted(body["checks"]),
                         ["answerer", "corpus", "embed", "index", "labeler"])
        self.assertTrue(body["checks"]["index"]["ok"])
        self.assertTrue(body["checks"]["index"]["fresh"])
        self.assertTrue(body["checks"]["corpus"]["ok"])
        self.assertEqual(body["checks"]["corpus"]["pdfs"], 1)


class TestStatsAndAdmin(Base):
    def test_stats_reflects_chats(self):
        self.chat()
        r = self.client.get("/api/stats?days=7")
        self.assertEqual(r.status_code, 200)
        s = r.json()
        self.assertGreaterEqual(s["turns"], 1)
        self.assertGreaterEqual(s["answers"], 1)
        self.assertGreaterEqual(s["latency_s"]["answerer"]["n"], 1)

    def test_admin_page_served(self):
        r = self.client.get("/admin")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Состояние системы", r.text)


if __name__ == "__main__":
    unittest.main()

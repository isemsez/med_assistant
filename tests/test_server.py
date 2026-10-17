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
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
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
    # no drugs.sqlite in TMP: the МНН lookup starts disabled (None) — tests
    # that want it set app.state.drugmap explicitly
    os.environ["DRUGS_DB"] = str(TMP / "no-drugs.sqlite")
    cards_dir = TMP / "cards"
    cards_dir.mkdir()
    (cards_dir / "Тестов_ТТ.амбулаторная_карта.md").write_text(
        "## Диагнозы\n- тест", encoding="utf-8")
    os.environ["CARD_DIR"] = str(cards_dir)
    import medassistant.config
    import medassistant.server
    importlib.reload(medassistant.config)
    importlib.reload(medassistant.server)
    srv = medassistant.server


def tearDownModule():
    for k in ("SESSIONS_DB", "AUDIT_LOG", "CLINREC_DIR", "DOCS_META",
              "INDEX_DIR", "CARD_DIR", "DRUGS_DB"):
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
        # "events" (default) | "endless" (streams forever; the server's stop
        # flag breaks the loop) | "cancel_sim" (waits for the flag, then
        # raises LLMCancelled — the instant-drop path)
        self.answer_mode = "events"
        self.answer_yields = 0  # events yielded by the endless stream so far

    def stream_answer(self, question, history=None, card=None, drug=None,
                      cancel=None):
        self.calls.append(("stream", question, list(history or []), card, drug))
        if isinstance(self.answer_events, Exception):
            raise self.answer_events
        if self.answer_mode == "endless":
            while True:
                self.answer_yields += 1
                yield {"type": "reasoning", "text": "мысль "}
                time.sleep(0.005)
        if self.answer_mode == "cancel_sim":
            self.answer_yields += 1
            yield {"type": "reasoning", "text": "мысль"}
            while cancel is None or not cancel.is_set():
                time.sleep(0.005)
            raise llm.LLMCancelled("cancelled by user")
        yield from self.answer_events

    def iter_appendix(self, question, raw_answer, session_id=None, manual=False):
        from medassistant.pipeline import AppendixResult
        self.calls.append(("appendix", question, raw_answer, {"manual": manual}))
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

    def chat(self, question="Вопрос про ХСН?", session_id=None, card=None):
        if session_id is None:
            session_id = self.client.post("/api/sessions",
                                          json={"title": ""}).json()["id"]
        payload = {"session_id": session_id, "question": question}
        if card is not None:
            payload["card"] = card
        r = self.client.post("/api/chat", json=payload)
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

    def test_reasoning_reset_clears_thinking(self):
        # watchdog aborted a looped stream upstream; the retry restarts
        # reasoning (stray "thought\n" prefix dropped again) and only the
        # fresh reasoning is streamed and stored (owner decision 2026-10-06;
        # the reset carries a visible note since 2026-10-14)
        self.pipe.answer_events = [
            {"type": "reasoning", "text": "loop"},
            {"type": "reasoning_reset"},
            {"type": "reasoning", "text": "thought\nзаново"},
            {"type": "content", "text": "Ответ."},
        ]
        sid, events = self.chat()
        self.assertIn(("reasoning_reset",
                       {"note": "Зацикливание размышлений — перезапускаю."}),
                      events)
        reasoning = "".join(d["t"] for e, d in events if e == "reasoning")
        self.assertEqual(reasoning, "заново")
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual(msgs[1]["reasoning"], "заново")

    def test_double_loop_fails_honestly(self):
        self.pipe.answer_events = llm.LLMLoopError(
            "degenerate reasoning loop: unit x3")
        sid, events = self.chat()
        names = [e for e, _ in events]
        # the fatal loop's garbage is dropped with its own note before the
        # honest failure (owner report 2026-10-14)
        self.assertIn(("reasoning_reset",
                       {"note": "Зацикливание размышлений — генерация "
                                "прервана."}), events)
        self.assertLess(events.index(("reasoning_reset",
                                      {"note": "Зацикливание размышлений — "
                                               "генерация прервана."})),
                        names.index("error"))
        self.assertIn("error", names)
        done = dict(events)["done"]
        self.assertIsNone(done["message_id"])

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


class TestStop(Base):
    """User stop of the answerer (owner decision 2026-10-15): POST
    /api/sessions/{id}/stop drops the Ollama connection, the stream ends
    with a `stopped` event, and the half-finished turn's rows are deleted
    (undo send). Answerer phase only — completed turns are untouchable."""

    def _turns(self, sid):
        with srv.app.state.turns_lock:
            return list(srv.app.state.turns.get(sid, []))

    def _chat_in_thread(self, sid, question="Вопрос про ХСН?"):
        out = {}

        def run():
            out["r"] = self.client.post(
                "/api/chat", json={"session_id": sid, "question": question})

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t, out

    def _wait_registered(self, sid, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._turns(sid):
                return
            time.sleep(0.01)
        self.fail("the in-flight turn never registered")

    def _wait_answer_events(self, n, timeout=5):
        # deterministic stop point: the turn must actually be streaming (the
        # stop may legitimately land before the first event otherwise — the
        # first two deltas die in the "thought\n" prefix drop)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.pipe.answer_yields >= n:
                return
            time.sleep(0.01)
        self.fail("the answerer never streamed")

    def _stop(self, sid, n=5):
        t, out = self._chat_in_thread(sid)
        self._wait_registered(sid)
        self._wait_answer_events(n)
        r = self.client.post(f"/api/sessions/{sid}/stop")
        t.join(timeout=10)
        self.assertFalse(t.is_alive(), "the stream must end after a stop")
        return r, out

    def test_stop_mid_answerer_undoes_the_turn(self):
        sid = self.client.post("/api/sessions", json={"title": ""}).json()["id"]
        self.pipe.answer_mode = "endless"
        r, out = self._stop(sid)
        self.assertEqual(r.json(), {"ok": True, "stopped": 1})
        names = [e for e, _ in parse_sse(out["r"].text)]
        self.assertIn("reasoning", names)
        self.assertEqual(names[-1], "stopped")
        self.assertNotIn("done", names, "an undone turn is never completed")
        self.assertEqual(
            self.client.get(f"/api/sessions/{sid}/messages").json(), [])
        self.assertEqual(self._turns(sid), [], "registry cleaned up")

    def test_stop_via_llm_cancelled_exception(self):
        # the instant path: the dropped connection surfaces in the streaming
        # thread as llm.LLMCancelled — same undo-send semantics
        sid = self.client.post("/api/sessions", json={"title": ""}).json()["id"]
        self.pipe.answer_mode = "cancel_sim"
        r, out = self._stop(sid, n=1)  # one event, then the flag wait
        self.assertEqual(r.json(), {"ok": True, "stopped": 1})
        names = [e for e, _ in parse_sse(out["r"].text)]
        self.assertEqual(names[-1], "stopped")
        self.assertEqual(
            self.client.get(f"/api/sessions/{sid}/messages").json(), [])

    def test_stop_audited(self):
        sid = self.client.post("/api/sessions", json={"title": ""}).json()["id"]
        self.pipe.answer_mode = "endless"
        r, _ = self._stop(sid)
        self.assertEqual(r.json()["stopped"], 1)
        recs = [json.loads(l) for l in (TMP / "audit.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        stops = [r for r in recs if r["stage"] == "user_stop"
                 and r.get("session_id") == sid]
        self.assertEqual(len(stops), 1)

    def test_stop_idle_session_is_noop(self):
        sid = self.client.post("/api/sessions", json={"title": ""}).json()["id"]
        self.assertEqual(self.client.post(f"/api/sessions/{sid}/stop").json(),
                         {"ok": True, "stopped": 0})

    def test_stop_after_done_keeps_turn(self):
        # stop is answerer-phase only (owner decision 2026-10-15): a
        # completed turn and its appendix are untouchable
        sid, _ = self.chat()
        self.assertEqual(self.client.post(f"/api/sessions/{sid}/stop").json(),
                         {"ok": True, "stopped": 0})
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])

    def test_stop_unknown_session_404(self):
        self.assertEqual(
            self.client.post("/api/sessions/999/stop").status_code, 404)


class TestPdfTurn(Base):
    def test_event_order_kind_done_stage_appendix(self):
        sid, events = self.chat("pdf ХСН")
        names = [e for e, _ in events]
        self.assertEqual(names[0], "kind")
        self.assertEqual(names[1], "done")
        self.assertLess(names.index("done"), names.index("stage"),
                        "composer must unblock before the machinery streams")
        self.assertEqual(names[-1], "appendix")
        kind = dict(events)["kind"]
        self.assertEqual(kind, {"turn": "pdf", "payload": "ХСН"})
        self.assertEqual(dict(events)["appendix"]["status"], "ok")
        done = dict(events)["done"]
        self.assertIsNotNone(done["message_id"])
        self.assertIsNotNone(done["user_message_id"])

    def test_no_answerer_call_and_manual_flag(self):
        sid, _ = self.chat("pdf ХСН")
        self.assertEqual([c[0] for c in self.pipe.calls], ["appendix"])
        question, raw_answer, kw = self.pipe.calls[0][1:]
        self.assertEqual(question, "ХСН")
        self.assertEqual(raw_answer, "")
        self.assertTrue(kw["manual"])

    def test_pdf_turn_is_input_only(self):
        sid, _ = self.chat("Первый вопрос?")
        sid, _ = self.chat("pdf ХСН", session_id=sid)
        appendix_calls = [c for c in self.pipe.calls if c[0] == "appendix"]
        self.assertEqual(appendix_calls[1][2], "",
                         "manual pdf turns must not use earlier answers")

    def test_messages_persisted_with_kind_and_caption(self):
        sid, _ = self.chat("pdf ХСН")
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual([m["kind"] for m in msgs], ["pdf", "pdf"])
        self.assertEqual(msgs[0]["content"], "pdf ХСН")
        self.assertEqual(msgs[1]["content"], "Подбор pdf: «ХСН»")
        self.assertEqual(msgs[1]["appendix"]["status"], "ok")

    def test_pdf_turn_excluded_from_answerer_history(self):
        sid, _ = self.chat("Вопрос 1?")
        self.chat("pdf ХСН", session_id=sid)
        self.chat("Вопрос 2?", session_id=sid)
        stream_calls = [c for c in self.pipe.calls if c[0] == "stream"]
        self.assertEqual(len(stream_calls), 2)
        self.assertEqual([m["content"] for m in stream_calls[1][2]],
                         ["Вопрос 1?", "Ответ про ХСН."])

    def test_bare_pdf_400(self):
        sid = self.client.post("/api/sessions", json={"title": ""}).json()["id"]
        for q in ("pdf", "пдф   "):
            r = self.client.post("/api/chat", json={"session_id": sid,
                                                    "question": q})
            self.assertEqual(r.status_code, 400)

    def test_cyrillic_prefix_routed(self):
        sid, events = self.chat("ПДФ ХСН")
        self.assertIn("kind", [e for e, _ in events])
        self.assertEqual(dict(events)["kind"]["payload"], "ХСН")

    def test_no_space_prefix_goes_to_answerer(self):
        sid, events = self.chat("pdfхсн — что это?")
        self.assertNotIn("kind", [e for e, _ in events])
        self.assertEqual(self.pipe.calls[0][0], "stream")

    def test_appendix_crash_streams_error_keeps_assistant_row(self):
        self.pipe.appendix_error = RuntimeError("machinery bug")
        sid, events = self.chat("pdf ХСН")
        names = [e for e, _ in events]
        self.assertIn("error", names)
        self.assertNotIn("appendix", names)
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["kind"], "pdf")
        self.assertIsNone(msgs[1]["appendix"])


CARD_FILE = "Тестов_ТТ.амбулаторная_карта.md"
CARD_TEXT = "## Диагнозы\n- тест"


class TestCards(Base):
    def _post_card(self, card, question="Вопрос про ХСН?", session_id=None):
        if session_id is None:
            session_id = self.client.post("/api/sessions",
                                          json={"title": ""}).json()["id"]
        return session_id, self.client.post(
            "/api/chat", json={"session_id": session_id,
                               "question": question, "card": card})

    def test_cards_listed(self):
        body = self.client.get("/api/cards").json()
        self.assertEqual([c["file"] for c in body], [CARD_FILE])
        self.assertEqual(body[0]["name"], "Тестов ТТ")
        self.assertGreater(body[0]["size"], 0)

    def test_cards_missing_dir_empty(self):
        with patch.object(srv.cards.config, "CARD_DIR", Path("nowhere")):
            self.assertEqual(self.client.get("/api/cards").json(), [])

    def test_card_reaches_answerer_question_stored_plain(self):
        sid, _ = self.chat(card=CARD_FILE)
        stream = [c for c in self.pipe.calls if c[0] == "stream"][0]
        self.assertEqual(stream[1], "Вопрос про ХСН?")
        self.assertEqual(stream[3], ("Тестов ТТ", CARD_TEXT))
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual(msgs[0]["content"], "Вопрос про ХСН?")

    def test_card_never_leaks_into_history(self):
        sid, _ = self.chat(card=CARD_FILE)
        self.chat("Второй вопрос?", session_id=sid)
        second = [c for c in self.pipe.calls if c[0] == "stream"][1]
        self.assertEqual([m["content"] for m in second[2]],
                         ["Вопрос про ХСН?", "Ответ про ХСН."])
        self.assertIsNone(second[3], "the next question carries no card")

    def test_invalid_card_400_question_not_sent(self):
        for bad in ("Нет_такого.амбулаторная_карта.md", "..\\evil.амбулаторная_карта.md",
                    "../evil.амбулаторная_карта.md", "Тестов_ТТ.md", ".."):
            sid, r = self._post_card(bad)
            self.assertEqual(r.status_code, 400, bad)
            self.assertEqual(self.pipe.calls, [], bad)

    def test_oversize_card_400(self):
        with patch.object(srv.cards.config, "CARD_MAX_CHARS", 5):
            sid, r = self._post_card(CARD_FILE)
        self.assertEqual(r.status_code, 400)

    def test_pdf_turn_ignores_card_entirely(self):
        # an INVALID card must not even be validated on the pdf path
        sid, r = self._post_card("ghost.амбулаторная_карта.md", question="pdf ХСН")
        self.assertEqual(r.status_code, 200)
        self.assertEqual([c[0] for c in self.pipe.calls], ["appendix"])

    def test_card_used_audit_no_content(self):
        sid, _ = self.chat(card=CARD_FILE)
        sid, r = self._post_card("ghost.амбулаторная_карта.md", session_id=sid)
        self.assertEqual(r.status_code, 400)
        recs = [json.loads(l) for l in (TMP / "audit.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        used = [r for r in recs if r["stage"] == "card_used"]
        self.assertGreaterEqual(len(used), 1)  # the file is shared across tests
        self.assertEqual(used[-1]["file"], CARD_FILE)
        self.assertGreater(used[-1]["chars"], 0)
        rejected = [r for r in recs if r["stage"] == "card_rejected"]
        self.assertEqual(rejected[-1]["file"], "ghost.амбулаторная_карта.md")
        # audit carries filename + char count only, never the card content
        self.assertNotIn(CARD_TEXT.splitlines()[-1],
                         json.dumps(recs, ensure_ascii=False))

    def test_pin_saved_on_send_and_returned(self):
        sid, _ = self.chat(card=CARD_FILE)
        r = self.client.get(f"/api/sessions/{sid}/card")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"file": CARD_FILE, "name": "Тестов ТТ"})

    def test_pin_absent_without_card_use(self):
        sid = self.client.post("/api/sessions", json={"title": ""}).json()["id"]
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/card").json(),
                         {"file": None})
        self.assertEqual(self.client.get("/api/sessions/999/card").status_code,
                         404)

    def test_pin_survives_cardless_and_pdf_turns(self):
        sid, _ = self.chat(card=CARD_FILE)
        self.chat("Без карты?", session_id=sid)
        self.chat("pdf ХСН", session_id=sid)
        r = self.client.get(f"/api/sessions/{sid}/card").json()
        self.assertEqual(r["file"], CARD_FILE, "pin = last used card, kept")

    def test_pin_on_pick_post(self):
        sid = self.client.post("/api/sessions", json={"title": ""}).json()["id"]
        r = self.client.post(f"/api/sessions/{sid}/card",
                             json={"file": CARD_FILE})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            self.client.get(f"/api/sessions/{sid}/card").json(),
            {"file": CARD_FILE, "name": "Тестов ТТ"})

    def test_pin_post_rejects_unknown_file(self):
        sid = self.client.post("/api/sessions", json={"title": ""}).json()["id"]
        r = self.client.post(f"/api/sessions/{sid}/card",
                             json={"file": "ghost.амбулаторная_карта.md"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(
            self.client.get(f"/api/sessions/{sid}/card").json(), {"file": None})

    def test_pin_post_clears_with_empty_file(self):
        sid, _ = self.chat(card=CARD_FILE)
        r = self.client.post(f"/api/sessions/{sid}/card", json={"file": ""})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            self.client.get(f"/api/sessions/{sid}/card").json(), {"file": None})

    def test_pin_post_unknown_session_404(self):
        self.assertEqual(
            self.client.post("/api/sessions/999/card",
                             json={"file": CARD_FILE}).status_code, 404)

    def test_pin_with_missing_file_dropped(self):
        sid, _ = self.chat(card=CARD_FILE)
        srv.app.state.store.set_session_card(sid, "ghost.амбулаторная_карта.md")
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/card").json(),
                         {"file": None})
        self.assertIsNone(srv.app.state.store.get_session_card(sid))


class TestStatic(Base):
    def test_index_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Мед. ассистент", r.text)

    def test_js_css_served_no_cache(self):
        # stale-asset fix (owner decision 2026-10-13; .html added
        # 2026-10-17 after a cached old index.html + fresh app.js wedged
        # the composer — session 84): every reload must revalidate
        for path in ("/", "/admin", "/app.js", "/util.js", "/style.css"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 200, path)
            self.assertEqual(r.headers.get("cache-control"), "no-cache", path)

    def test_pdf_served(self):
        r = self.client.get("/data/pdf/" + quote(PDF_FILE))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content[:8], b"%PDF-1.4")


class TestDrugMapHint(Base):
    """Register-grounded МНН lookup v2 (owner decision 2026-10-13): the SSE
    event fires before the first token, the P.S. reaches the answerer, the
    stored question stays plain, usable matches persist on the user row;
    placeholder-МНН rows are not keyed, so such names never match."""

    def setUp(self):
        super().setUp()
        from medassistant.drugmap import DrugMap
        self.assertIsNone(srv.app.state.drugmap, "TMP has no drugs.sqlite")
        srv.app.state.drugmap = DrugMap([
            ("Сорбифер Дурулес", "Железа сульфат+[Аскорбиновая кислота]"),
            ("Кагоцел®", "~"),
        ])

    def tearDown(self):
        srv.app.state.drugmap = None
        super().tearDown()

    def test_event_before_tokens_and_ps_passed(self):
        sid, events = self.chat("Что содержится в сорбифер дурулес")
        names = [e for e, _ in events]
        self.assertIn("drug_map", names)
        self.assertLess(names.index("drug_map"), names.index("reasoning"))
        payload = dict(events)["drug_map"]
        self.assertEqual(payload["matches"],
                         [{"trade": "Сорбифер Дурулес",
                           "inn": "железа сульфат + аскорбиновая кислота",
                           "token": "сорбифер"}])
        # the answerer got the register matches; the stored question stays
        # plain while the matches persist on the user row (history payload)
        stream = [c for c in self.pipe.calls if c[0] == "stream"][-1]
        self.assertEqual(stream[4][0]["trade"], "Сорбифер Дурулес")
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertEqual(msgs[0]["content"], "Что содержится в сорбифер дурулес")
        self.assertEqual(msgs[0]["drug_map"],
                         [{"trade": "Сорбифер Дурулес",
                           "inn": "железа сульфат + аскорбиновая кислота",
                           "token": "сорбифер"}])
        self.assertIsNone(msgs[1]["drug_map"])

    def test_no_event_without_match(self):
        sid, events = self.chat("Просто вопрос про ХСН?")
        self.assertNotIn("drug_map", [e for e, _ in events])
        stream = [c for c in self.pipe.calls if c[0] == "stream"][-1]
        self.assertEqual(stream[4], [])
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertIsNone(msgs[0]["drug_map"])

    def test_placeholder_rows_never_match(self):
        # «Кагоцел®» carries placeholder МНН ~ → not keyed → no match, no
        # event, no P.S., no stored drug_map
        sid, events = self.chat("про кагоцел безопасен")
        self.assertNotIn("drug_map", [e for e, _ in events])
        stream = [c for c in self.pipe.calls if c[0] == "stream"][-1]
        self.assertEqual(stream[4], [])
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertIsNone(msgs[0]["drug_map"])

    def test_pdf_turn_skips_lookup(self):
        sid, events = self.chat("pdf сорбифер дурулес")
        self.assertNotIn("drug_map", [e for e, _ in events])
        self.assertEqual([c[0] for c in self.pipe.calls], ["appendix"])
        msgs = self.client.get(f"/api/sessions/{sid}/messages").json()
        self.assertIsNone(msgs[0]["drug_map"])

    def test_disabled_map_changes_nothing(self):
        srv.app.state.drugmap = None
        sid, events = self.chat("Что содержится в сорбифер дурулес")
        self.assertNotIn("drug_map", [e for e, _ in events])
        stream = [c for c in self.pipe.calls if c[0] == "stream"][-1]
        self.assertEqual(stream[4], [])


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
    def test_startup_creates_sessions_backup(self):
        backups = list((TMP / "backups").glob("sessions-*.sqlite"))
        self.assertGreaterEqual(len(backups), 1,
                                "server startup must checkpoint+copy the DB")
        audit_lines = (TMP / "audit.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("backup_failed", audit_lines)

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

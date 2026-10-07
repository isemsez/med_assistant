"""Chat service: SSE streaming answers + background appendix (architecture.md §4).

Run: uvicorn medassistant.server:app --host 127.0.0.1 --port 8642
"""

import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import audit as audit_mod
from .. import store as store_mod
from .. import appendix, config, health, llm, stats
from ..pipeline import load_pipeline, split_pdf_prefix
from ..store import load_store

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="med_assistant", docs_url=None, redoc_url=None)


class ChatRequest(BaseModel):
    session_id: int
    question: str


class SessionRequest(BaseModel):
    title: str = ""


@app.on_event("startup")
def _wire():
    app.state.audit = audit_mod.Audit(config.AUDIT_LOG_PATH)
    app.state.store = load_store()
    try:
        store_mod.backup()
    except Exception as e:  # a failed backup must never block the app
        app.state.audit.log("backup_failed", error=str(e))
    app.state.pipe = load_pipeline(audit=app.state.audit)


@app.get("/api/health")
def get_health():
    """Liveness/readiness: endpoints, index freshness, corpus (cached 30s)."""
    return health.run_health()


@app.get("/api/stats")
def get_stats(days: int = 7):
    """Audit-log aggregates (admin page data source)."""
    days = min(max(days, 1), 90)
    return stats.collect(config.AUDIT_LOG_PATH, days=days)


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")


@app.post("/api/sessions")
def create_session(req: SessionRequest):
    sid = app.state.store.create_session(req.title)
    return {"id": sid}


@app.get("/api/sessions")
def list_sessions():
    return app.state.store.list_sessions()


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: int):
    if not app.state.store.session_exists(session_id):
        raise HTTPException(404, "session not found")
    return app.state.store.get_messages(session_id)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: int):
    if not app.state.store.delete_session(session_id):
        raise HTTPException(404, "session not found")
    app.state.audit.log("delete_session", session_id=session_id)
    return {"ok": True}


@app.delete("/api/sessions/{session_id}/messages/{message_id}")
def rewind(session_id: int, message_id: int):
    deleted = app.state.store.truncate_from(session_id, message_id)
    if deleted is None:
        raise HTTPException(404, "session or message not found")
    app.state.audit.log("rewind", session_id=session_id,
                        from_message_id=message_id, deleted=deleted)
    return {"deleted": deleted}


STAGE_TEXT = {
    "extract": "Поиск болезней в тексте…",
    "search": "Ищу соответствующие pdf.",
    "auto_pick": "Соответствие найдено.",
    "select": "Фильтрую по: «{disease}»",
}

# manual «pdf » turns: same stages, a wording that fits an explicit request
STAGE_TEXT_PDF = {**STAGE_TEXT, "extract": "Разбираю запрос pdf…"}


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_machinery(sid, message_id, events, stage_text):
    """Shared appendix-machinery SSE loop: stage events, honest error note on
    a crash, final appendix payload persisted onto the assistant message."""
    store, aud = app.state.store, app.state.audit
    result = None
    try:
        for ev in events:
            if ev.get("stage") == "done":
                result = ev["result"]
                continue
            text = stage_text.get(ev["stage"], ev["stage"]).format(
                disease=ev.get("disease", ""))
            yield _sse("stage", {"stage": ev["stage"],
                                 "disease": ev.get("disease"),
                                 "code_version": ev.get("code_version"),
                                 "cards": ev.get("cards"),
                                 "text": text})
    except Exception as e:  # machinery bug — surface, never crash the stream
        aud.log("appendix_crash", session_id=sid, error=repr(e))
        yield _sse("error", {"text": appendix.NOTE_SEARCH_ERROR})
        result = None
    if result is not None:
        payload = result.to_payload()
        store.set_appendix(message_id, payload)
        yield _sse("appendix", payload)


@app.post("/api/chat")
def chat(req: ChatRequest):
    store, pipe, aud = app.state.store, app.state.pipe, app.state.audit
    if not store.session_exists(req.session_id):
        raise HTTPException(404, "session not found")
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "empty question")
    pdf_payload = split_pdf_prefix(question)
    if pdf_payload == "":
        raise HTTPException(400, "укажите заболевание после «pdf»")
    sid = req.session_id

    if pdf_payload is not None:
        def gen():
            # manual pdf turn (owner decision 2026-10-04): labeler 1 in direct
            # mode parses the payload; the turn is input-only — no context
            # from the earlier conversation
            mid_user = store.add_message(sid, "user", question, kind="pdf")
            store.touch(sid, title=question[:60])
            aud.log("pdf_query", session_id=sid, payload=pdf_payload)
            message_id = store.add_message(
                sid, "assistant", f"Подбор pdf: «{pdf_payload}»", kind="pdf")
            yield _sse("kind", {"turn": "pdf", "payload": pdf_payload})
            yield _sse("done", {"session_id": sid, "message_id": message_id,
                                "user_message_id": mid_user, "failed": False})
            yield from _run_machinery(
                sid, message_id,
                pipe.iter_appendix(pdf_payload, "",
                                   session_id=sid, manual=True),
                STAGE_TEXT_PDF)
            store.touch(sid)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    def gen():
        history = store.history_window(sid, config.HISTORY_WINDOW)
        mid_user = store.add_message(sid, "user", question)
        store.touch(sid, title=question[:60])
        aud.log("question", session_id=sid, question=question)

        parts, reasoning_parts, failed = [], [], False
        t_answer = time.monotonic()
        # medgemma's reasoning always starts with a stray "thought\n"; it
        # arrives split across stream deltas, so it is dropped char-wise
        drop = len("thought\n")
        try:
            for event in pipe.stream_answer(question, history):
                if event["type"] == "reasoning":
                    t = event["text"]
                    if drop:
                        d = min(drop, len(t))
                        t, drop = t[d:], drop - d
                        if not t:
                            continue
                    reasoning_parts.append(t)
                    yield _sse("reasoning", {"t": t})
                elif event["type"] == "reasoning_reset":
                    # a degenerate thinking loop was aborted upstream; the
                    # retry restarts reasoning from scratch (2026-10-06)
                    reasoning_parts, drop = [], len("thought\n")
                    yield _sse("reasoning_reset", {})
                else:
                    parts.append(event["text"])
                    yield _sse("token", {"t": event["text"]})
        except llm.LLMError as e:
            failed = True
            aud.log("stage_stop", stop_stage="answerer", session_id=sid,
                    error=str(e))
            yield _sse("error", {"text": "Сбой генерации ответа."})

        raw_answer = "".join(parts)
        if not raw_answer:
            yield _sse("done", {"session_id": sid, "message_id": None,
                                "user_message_id": mid_user})
            return

        message_id = store.add_message(
            sid, "assistant", raw_answer,
            reasoning="".join(reasoning_parts) or None)
        aud.log("raw_answer", session_id=sid, answer=raw_answer,
                dur_s=round(time.monotonic() - t_answer, 2))
        # the answer is complete — the physician may ask the next question
        # while the appendix machinery runs below
        yield _sse("done", {"session_id": sid, "message_id": message_id,
                            "user_message_id": mid_user, "failed": failed})

        yield from _run_machinery(
            sid, message_id,
            pipe.iter_appendix(question, raw_answer, session_id=sid),
            STAGE_TEXT)
        store.touch(sid)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# local guideline PDFs for the appendix links; mounted before the "/" catch-all.
# StaticFiles(check_dir=True) makes a missing corpus a clear startup failure.
app.mount("/data/pdf", StaticFiles(directory=config.CLINREC_DIR), name="pdf")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

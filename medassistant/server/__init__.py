"""Chat service: SSE streaming answers + background appendix (architecture.md §4).

Run: uvicorn medassistant.server:app --host 127.0.0.1 --port 8642
"""

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import audit as audit_mod
from .. import store as store_mod
from .. import appendix, cards, config, drugmap, health, llm, stats
from ..pipeline import load_pipeline, split_pdf_prefix
from ..store import load_store

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="med_assistant", docs_url=None, redoc_url=None)


@app.middleware("http")
async def _static_revalidate(request, call_next):
    """Stale-JS fix (owner decision 2026-10-13): .js/.css responses carry
    `Cache-Control: no-cache` so an open tab revalidates app.js/style.css
    on the next reload instead of running yesterday's bundle."""
    resp = await call_next(request)
    if request.url.path.endswith((".js", ".css")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


class ChatRequest(BaseModel):
    session_id: int
    question: str
    card: str | None = None  # patient-card filename; ignored on pdf turns


class SessionRequest(BaseModel):
    title: str = ""


class CardPinRequest(BaseModel):
    file: str = ""  # empty clears the pin


class Turn:
    """One in-flight answerer turn (stop-registry entry, owner decision
    2026-10-15). `done` flips the moment the answerer stream finishes
    normally — stop requests skip done turns: stop is answerer-phase only,
    the appendix machinery always runs to completion."""

    def __init__(self, message_id):
        self.cancel = llm.StreamCancel()
        self.message_id = message_id  # the turn's user row
        self.done = False


def _register_turn(app, sid, turn):
    with app.state.turns_lock:
        app.state.turns.setdefault(sid, []).append(turn)


def _unregister_turn(app, sid, turn):
    with app.state.turns_lock:
        turns = app.state.turns.get(sid)
        if turns and turn in turns:
            turns.remove(turn)
            if not turns:
                app.state.turns.pop(sid, None)


@app.on_event("startup")
def _wire():
    app.state.audit = audit_mod.Audit(config.AUDIT_LOG_PATH)
    app.state.store = load_store()
    app.state.turns = {}  # session_id -> [Turn] (in-flight answerer turns)
    app.state.turns_lock = threading.Lock()
    try:
        store_mod.backup()
    except Exception as e:  # a failed backup must never block the app
        app.state.audit.log("backup_failed", error=str(e))
    app.state.pipe = load_pipeline(audit=app.state.audit)
    # register-grounded МНН lookup (owner decision 2026-10-12): a missing
    # drugs.sqlite silently disables the feature
    app.state.drugmap = drugmap.load()


@app.get("/api/health")
def get_health():
    """Liveness/readiness: endpoints, index freshness, corpus (cached 30s)."""
    return health.run_health()


@app.get("/api/stats")
def get_stats(days: int = 7):
    """Audit-log aggregates (admin page data source)."""
    days = min(max(days, 1), 90)
    return stats.collect(config.AUDIT_LOG_PATH, days=days)


@app.get("/api/cards")
def get_cards():
    """Patient cards for the composer picker (architecture.md §9)."""
    return cards.list_cards()


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


@app.get("/api/sessions/{session_id}/card")
def get_session_card(session_id: int):
    """The session's pinned patient card (the last one picked/used there),
    for picker preselection — invisible in the chat itself (architecture.md
    §9). A pin whose file no longer exists is dropped."""
    if not app.state.store.session_exists(session_id):
        raise HTTPException(404, "session not found")
    file = app.state.store.get_session_card(session_id)
    if not file:
        return {"file": None}
    card = next((c for c in cards.list_cards() if c["file"] == file), None)
    if card is None:
        app.state.store.set_session_card(session_id, None)
        return {"file": None}
    return {"file": card["file"], "name": card["name"]}


@app.post("/api/sessions/{session_id}/card")
def pin_session_card(session_id: int, req: CardPinRequest):
    """Pin a card to the session the moment the physician picks it in the
    UI (owner decision 2026-10-07) — so the preselection survives session
    switches even when no question was sent. The send path re-pins on
    card_used as a safety net."""
    if not app.state.store.session_exists(session_id):
        raise HTTPException(404, "session not found")
    if req.file and req.file not in {c["file"] for c in cards.list_cards()}:
        raise HTTPException(400, "unknown card file")
    app.state.store.set_session_card(session_id, req.file or None)
    return {"ok": True}


@app.post("/api/sessions/{session_id}/stop")
def stop_turn(session_id: int):
    """Stop the session's in-flight answerer turn(s) — undo the send (owner
    decision 2026-10-15): set() drops the Ollama connection, so generation
    cancels at once even mid-prefill, and the stream thread deletes the
    turn's rows and emits `stopped`. Turn-phase turns are skipped: once the
    answer text completes, the appendix machinery runs to completion."""
    if not app.state.store.session_exists(session_id):
        raise HTTPException(404, "session not found")
    with app.state.turns_lock:
        turns = list(app.state.turns.get(session_id, []))
    stopped = 0
    for turn in turns:
        if not turn.done:
            turn.cancel.set()
            stopped += 1
    if stopped:
        app.state.audit.log("user_stop", session_id=session_id,
                            phase="answerer")
    return {"ok": True, "stopped": stopped}


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

    # patient card (owner decision 2026-10-06): pdf turns ignore it entirely;
    # otherwise validate and re-read fresh — the doctor may have edited it.
    # Any violation 400s before the question is sent.
    card_ctx = None
    if pdf_payload is None and req.card:
        try:
            card_ctx = cards.load_card(req.card)
        except cards.CardError as e:
            aud.log("card_rejected", session_id=sid, file=req.card,
                    error=str(e))
            raise HTTPException(400, str(e))

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
        # register-grounded trade-name -> МНН matches (owner decision
        # 2026-10-12): computed BEFORE the user row is stored — the usable
        # matches persist on it (2026-10-13) and the SSE event fires before
        # the first token so the block is instant
        drug = (app.state.drugmap.lookup(question, cap=config.DRUG_PS_MAX)
                if app.state.drugmap is not None else [])
        usable = [m for m in drug if m["inn"]]
        mid_user = store.add_message(sid, "user", question,
                                     drug_map=usable or None)
        store.touch(sid, title=question[:60])
        aud.log("question", session_id=sid, question=question)
        if drug:
            aud.log("drug_map", session_id=sid, matches=drug)
            if usable:
                yield _sse("drug_map", {"matches": usable})
        if card_ctx is not None:
            # filename + char count only — card content is never logged
            aud.log("card_used", session_id=sid, file=req.card,
                    chars=len(card_ctx[1]))
            # pin the card to the session for later picker preselection
            store.set_session_card(sid, req.card)

        # user stop (owner decision 2026-10-15): the turn registers until the
        # answerer stream completes; POST /api/sessions/{sid}/stop drops the
        # Ollama connection (instant even mid-prefill). This thread alone
        # deletes the turn's rows — the stop request only sets the flag.
        turn = Turn(mid_user)
        _register_turn(app, sid, turn)
        stopped = False
        try:
            parts, reasoning_parts, failed = [], [], False
            t_answer = time.monotonic()
            # medgemma's reasoning always starts with a stray "thought\n"; it
            # arrives split across stream deltas, so it is dropped char-wise
            drop = len("thought\n")
            it = pipe.stream_answer(question, history, card=card_ctx,
                                    drug=drug, cancel=turn.cancel)
            try:
                for event in it:
                    if turn.cancel.is_set():
                        stopped = True
                        break
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
                        # retry restarts reasoning from scratch (2026-10-06).
                        # The note replaces the old silent wipe so the restart
                        # is visible (owner report 2026-10-14)
                        reasoning_parts, drop = [], len("thought\n")
                        yield _sse("reasoning_reset", {
                            "note": "Зацикливание размышлений — перезапускаю."})
                    else:
                        parts.append(event["text"])
                        yield _sse("token", {"t": event["text"]})
            except llm.LLMCancelled:
                stopped = True  # the stopper thread dropped the connection
            except llm.LLMError as e:
                failed = True
                aud.log("stage_stop", stop_stage="answerer", session_id=sid,
                        error=str(e))
                if isinstance(e, llm.LLMLoopError):
                    # the retry looped again — drop the looped garbage and say
                    # why the thinking block is empty before the honest failure
                    yield _sse("reasoning_reset", {
                        "note": "Зацикливание размышлений — генерация прервана."})
                yield _sse("error", {"text": "Сбой генерации ответа."})
            finally:
                # a cooperative stop breaks mid-stream: closing the generator
                # drops the Ollama connection (no-op on normal completion)
                it.close()
            if stopped or turn.cancel.is_set():
                # undo the send: nothing is stored; the finally below deletes
                # the question row and the client restores it into the composer
                yield _sse("stopped", {"user_message_id": mid_user})
                return
            turn.done = True  # answerer finished — stop no longer applies

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
        finally:
            _unregister_turn(app, sid, turn)
            if not turn.done and (stopped or turn.cancel.is_set()):
                # the stopped turn's rows go now — even when the client already
                # disconnected (finally still runs on GeneratorExit, and the
                # dropped Ollama connection guarantees this thread woke up)
                try:
                    store.truncate_from(sid, mid_user)
                except Exception as e:
                    aud.log("stop_cleanup_failed", session_id=sid,
                            error=repr(e))

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# local guideline PDFs for the appendix links; mounted before the "/" catch-all.
# StaticFiles(check_dir=True) makes a missing corpus a clear startup failure.
app.mount("/data/pdf", StaticFiles(directory=config.CLINREC_DIR), name="pdf")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

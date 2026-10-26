"""Thin CLI over the pipeline (offline-testable, architecture.md §4).

Commands:
  build-index                  embed all guideline names -> data/index/
  ask --question "…"           live: stream answer, then run the appendix machinery
                               (--card FILE appends a patient card to the prompt)
  ask --answer-file FILE       appendix machinery over pre-generated answers
                               (JSONL rows: {question, raw_answer[, diseases[, picks]]};
                               `diseases` stubs labeler 1, `picks` stubs labeler 2)
  search --name "…"            debug: guideline search tool alone (needs embed)
  canary                       health checks + one E2E question through the
                               real pipeline; exit 1 on failure
  regression [--goldens F]     golden set against live endpoints (pre-ship)
"""

import argparse
import json
import sys
import time
from pathlib import Path

from . import audit as audit_mod
from . import cards, config, drugmap, gindex, llm
from .pipeline import split_pdf_prefix


def _print_result(result):
    print("\n--- Приложение ------------------------------------------------")
    if result.appendix_text:
        print(result.appendix_text)
    for note in result.notes:
        print(f"[!] {note}")
    if not result.appendix_text and not result.notes:
        print("(нет приложений и сообщений)")


def cmd_build_index(args):
    config.require(config.EMBED, "EMBED")
    docs = config.load_docs_meta(args.docs)
    n_docs, n_rows = gindex.build_index(docs, config.EMBED, args.out)
    print(f"index built: {n_docs} guidelines, {n_rows} vectors -> {args.out}")


def cmd_search(args):
    config.require(config.EMBED, "EMBED")
    from .abbr import Abbr, load_abbr
    abbr_map = (Abbr(load_abbr(config.ABBR_PATH))
                if config.ABBR_PATH.exists() else None)
    index = gindex.GuidelineIndex.load(args.index)
    variants = abbr_map.variants(args.name) if abbr_map else [args.name]
    print(f"variants: {variants}")
    for card in index.search(variants, config.EMBED, top_k=args.top_k):
        print(f"  {card['score']:<8.4f} {card['code_version']:<10} {card['name']}")


def _run_row(pipe, row, session_id=None):
    question = row.get("question") or ""
    raw_answer = row.get("raw_answer") or ""
    if not raw_answer:
        print("[skip] row without raw_answer", file=sys.stderr)
        return
    diseases, picks = row.get("diseases"), row.get("picks")
    kw = {}
    if diseases is not None:
        kw["diseases"] = diseases
        if picks is not None:
            # fully offline stub: dummy search cards, picks come from the row
            kw["search_fn"] = lambda v: [{"code_version": "—", "name": "—",
                                          "score": 1.0}]
            kw["select_fn"] = lambda q, a, d, c: list(picks)
    elif picks is not None:
        print("[skip] picks without diseases", file=sys.stderr)
        return
    result = pipe.run_appendix(question, raw_answer, session_id=session_id, **kw)
    _print_result(result)


def _load_card_spec(spec):
    """`--card` accepts a filename inside CARD_DIR or a direct path; both go
    through the same validation (architecture.md §9)."""
    p = Path(spec)
    return (cards.load_card_path(p) if p.exists() else cards.load_card(spec))


def cmd_ask(args):
    aud = audit_mod.Audit(config.AUDIT_LOG_PATH)
    pipe = _load_pipeline(aud)
    store = None
    session_id = args.session
    if args.session or args.save:
        from .store import load_store
        store = load_store()
        if session_id is None:
            session_id = store.create_session()
        print(f"[session {session_id}]")

    if args.answer_file:
        for line in Path(args.answer_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                _run_row(pipe, json.loads(line), session_id=session_id)
        return

    if not args.question:
        print("either --question or --answer-file is required", file=sys.stderr)
        raise SystemExit(2)

    pdf_payload = split_pdf_prefix(args.question)
    if pdf_payload == "":
        print("укажите заболевание после «pdf»", file=sys.stderr)
        raise SystemExit(2)
    if pdf_payload is not None:
        # manual pdf turn (owner decision 2026-10-04): labeler 1 in direct
        # mode over the payload, no answerer call; input-only — no context
        # from the earlier conversation. The card never applies here.
        if args.card:
            print("[карта] для pdf-запроса карта пациента не используется",
                  file=sys.stderr)
        if store:
            store.add_message(session_id, "user", args.question, kind="pdf")
        aud.log("pdf_query", session_id=session_id, payload=pdf_payload)
        result = None
        for ev in pipe.iter_appendix(pdf_payload, "",
                                      session_id=session_id, manual=True):
            if ev.get("stage") == "done":
                result = ev["result"]
            else:
                print(f"[подбор pdf] {ev['stage']}"
                      + (f": {ev['disease']}" if ev.get("disease") else ""),
                      file=sys.stderr)
        _print_result(result)
        if store:
            store.add_message(session_id, "assistant",
                              f"Подбор pdf: «{pdf_payload}»",
                              appendix_payload=result.to_payload(), kind="pdf")
            store.touch(session_id)
        return

    card = None
    if args.card:
        try:
            card = _load_card_spec(args.card)
        except cards.CardError as e:
            print(f"[карта] {e}", file=sys.stderr)
            raise SystemExit(2)

    history = store.history_window(session_id, config.HISTORY_WINDOW) if store else []
    # register-grounded МНН lookup (owner decision 2026-10-12): same P.S.
    # the web app injects; a note line per usable match, persisted on the
    # saved user row like the server does
    drug = None
    dmap = drugmap.load()
    if dmap is not None:
        drug = dmap.lookup(args.question, cap=config.DRUG_PS_MAX)
    usable = [m for m in (drug or []) if m["inn"]]
    if store:
        store.add_message(session_id, "user", args.question,
                          drug_map=usable or None)
    aud.log("question", session_id=session_id, question=args.question)
    if card is not None:
        aud.log("card_used", session_id=session_id, file=args.card,
                chars=len(card[1]))
    if drug:
        aud.log("drug_map", session_id=session_id, matches=drug)
        for m in drug:
            if m["inn"]:
                print(f"[МНН] {m['trade']}: {m['inn']}", file=sys.stderr)

    parts = []
    t_answer = time.monotonic()
    try:
        shown_thinking = False
        for event in pipe.stream_answer(args.question, history, card=card,
                                        drug=drug):
            if event["type"] == "reasoning":
                if not shown_thinking:
                    print("(модель размышляет — скрыто, пишется в stderr)",
                          file=sys.stderr)
                    shown_thinking = True
                sys.stderr.write(event["text"])
                sys.stderr.flush()
            elif event["type"] == "reasoning_reset":
                print("(прерываю размышления — перезапуск)", file=sys.stderr)
            else:
                parts.append(event["text"])
                print(event["text"], end="", flush=True)
        print()
    except llm.LLMError as e:
        aud.log("stage_stop", stop_stage="answerer", session_id=session_id,
                error=str(e), tail=getattr(e, "tail", None))
        print(f"\n[сбой ответа] {e}", file=sys.stderr)
        raise SystemExit(1)
    raw_answer = "".join(parts)
    aud.log("raw_answer", session_id=session_id, answer=raw_answer,
            dur_s=round(time.monotonic() - t_answer, 2))

    result = None
    for ev in pipe.iter_appendix(args.question, raw_answer, session_id=session_id):
        if ev.get("stage") == "done":
            result = ev["result"]
        else:
            print(f"[подбор] {ev['stage']}"
                  + (f": {ev['disease']}" if ev.get("disease") else ""),
                  file=sys.stderr)
    _print_result(result)
    if store:
        store.add_message(session_id, "assistant", raw_answer,
                          appendix_payload=result.to_payload())
        store.touch(session_id)


def _load_pipeline(aud):
    from .pipeline import load_pipeline
    return load_pipeline(audit=aud)


def cmd_canary(args):
    from . import canary
    ok = canary.main(e2e=not args.no_e2e, max_tokens=args.max_tokens)
    if not ok:
        raise SystemExit(1)


def cmd_regression(args):
    from . import regression
    ok = regression.run(args.goldens, max_tokens=args.max_tokens, tier=args.tier)
    if not ok:
        raise SystemExit(1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="medassistant",
                                description="Answer-first medical assistant (architecture.md)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="embed guideline names -> index")
    b.add_argument("--docs", default=str(config.DOCS_META_PATH))
    b.add_argument("--out", default=str(config.INDEX_DIR))
    b.set_defaults(func=cmd_build_index)

    s = sub.add_parser("search", help="guideline search tool debug")
    s.add_argument("--name", required=True)
    s.add_argument("--index", default=str(config.INDEX_DIR))
    s.add_argument("--top-k", type=int, default=config.TOP_K)
    s.set_defaults(func=cmd_search)

    a = sub.add_parser("ask", help="ask a question / re-run appendix on answers")
    a.add_argument("--question")
    a.add_argument("--answer-file")
    a.add_argument("--card", default=None,
                   help="patient card (filename in data/patient_cards or a path), "
                        "appended to the answerer prompt for this question")
    a.add_argument("--session", type=int, default=None)
    a.add_argument("--save", action="store_true",
                   help="store the exchange in the sessions DB")
    a.set_defaults(func=cmd_ask)

    c = sub.add_parser("canary", help="health checks + one E2E question")
    c.add_argument("--no-e2e", action="store_true",
                   help="health checks only (fast, no model generation)")
    c.add_argument("--max-tokens", type=int, default=3072,
                   help="answerer token budget for the E2E question "
                        "(thinking needs headroom; production chats may differ)")
    c.set_defaults(func=cmd_canary)

    rg = sub.add_parser("regression", help="golden set against live endpoints")
    rg.add_argument("--goldens", default=str(config.PROJECT_ROOT / "tests" / "goldens.jsonl"))
    rg.add_argument("--max-tokens", type=int, default=3072,
                    help="answerer token budget per golden question")
    rg.add_argument("--tier", choices=("all", "search", "full"), default="all",
                    help="run only one tier (search = deterministic + fast)")
    rg.set_defaults(func=cmd_regression)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

"""Audit-log aggregates for /api/stats and the /admin page.

Reads logs/audit.jsonl line by line (tolerant to bad lines), keeps records
inside a trailing window, and returns counters, latency and the last errors.
Pure Python over the existing audit trail — no new dependencies (owner
decision 2026-10-02: laptop-only, built-in monitoring).
"""

import datetime
import json
from collections import Counter
from pathlib import Path

ERROR_STAGES = {"stage_stop", "labeler_extract_invalid", "labeler_select_invalid",
                "labeler_schema_fallback", "appendix_crash"}
LATENCY_BY_STAGE = {  # record stage -> latency field source
    "answerer": "raw_answer",
    "extract": "labeler_extract",
    "search": "search",
    "select": "labeler_select",
}


def _parse_ts(rec):
    try:
        return datetime.datetime.fromisoformat(rec["ts"])
    except (KeyError, ValueError, TypeError):
        return None


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def _audit_files(p):
    """Active file plus rotated generations, oldest first (newest last)."""
    files = []
    rotated = []
    for f in p.parent.glob(p.name + ".*"):
        suffix = f.name[len(p.name) + 1:]
        if suffix.isdigit():
            rotated.append((int(suffix), f))
    rotated.sort(reverse=True)  # .N (oldest) .. .1
    files.extend(f for _, f in rotated)
    files.append(p)
    return files


def read_records(path, days=None, now=None):
    """Records inside the trailing window across ALL generations.

    Returns (records, skipped): records are newest-last; skipped counts lines
    that were valid enough to open but unusable (bad JSON, unparseable ts
    inside the file). days=None keeps all.
    """
    p = Path(path)
    if not p.is_file():
        return [], 0
    if days is not None:
        now = now or datetime.datetime.now(datetime.timezone.utc)
        cutoff = now - datetime.timedelta(days=days)
    out, skipped = [], 0
    for f in _audit_files(p):
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if days is not None:
                ts = _parse_ts(rec)
                if ts is None or ts < cutoff:
                    if ts is None:
                        skipped += 1
                    continue
            out.append(rec)
    return out, skipped


def collect(path, days=7, now=None):
    records, skipped = read_records(path, days=days, now=now)
    stages = Counter(r.get("stage") for r in records)

    status = Counter(r.get("status") for r in records
                     if r.get("stage") == "appendix_done")
    stops = Counter((r.get("stop_stage") or "?")
                    for r in records if r.get("stage") == "stage_stop")
    picks_recs = [r for r in records if r.get("stage") == "picks"]
    auto = sum(1 for r in picks_recs if r.get("auto"))

    latency = {}
    for stage, rec_stage in LATENCY_BY_STAGE.items():
        durs = [r["dur_s"] for r in records
                if r.get("stage") == rec_stage and isinstance(r.get("dur_s"), (int, float))]
        latency[stage] = ({"n": len(durs), "mean": _mean(durs),
                           "max": max(durs) if durs else None})

    last_errors = []
    for r in reversed(records):
        if r.get("stage") in ERROR_STAGES:
            last_errors.append({
                "ts": r.get("ts"), "stage": r.get("stage"),
                "stop_stage": r.get("stop_stage"),
                "disease": r.get("disease"),
                "error": (r.get("error") or "")[:200]})
        if len(last_errors) >= 5:
            break

    return {
        "window_days": days,
        "skipped_records": skipped,
        "turns": stages.get("question", 0),
        "pdf_turns": stages.get("pdf_query", 0),
        "answers": stages.get("raw_answer", 0),
        "user_stops": stages.get("user_stop", 0),
        "appendix_status": dict(status),
        "stage_stop": dict(stops),
        "invalid": {"extract": stages.get("labeler_extract_invalid", 0),
                    "select": stages.get("labeler_select_invalid", 0)},
        "schema_fallback": stages.get("labeler_schema_fallback", 0),
        "appendix_crash": stages.get("appendix_crash", 0),
        "picks": {"calls": len(picks_recs), "auto": auto},
        "latency_s": latency,
        "last_errors": last_errors,
    }

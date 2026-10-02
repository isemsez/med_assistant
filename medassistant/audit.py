"""JSONL audit trail: one record per pipeline stage per turn
(architecture.md §3[5] — question, raw answer, labeler outputs, tool results,
picks — for auditability)."""

import datetime
import json
import os
from pathlib import Path


def _ts():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Audit:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, stage, session_id=None, **fields):
        rec = {"ts": _ts(), "stage": stage}
        if session_id is not None:
            rec["session_id"] = session_id
        rec.update(fields)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

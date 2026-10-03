"""JSONL audit trail: one record per pipeline stage per turn
(architecture.md §3[5] — question, raw answer, labeler outputs, tool results,
picks — for auditability).

Size-based rotation: when the active file crosses AUDIT_ROTATE_MB it shifts to
``audit.jsonl.1`` (up to AUDIT_KEEP generations) — the file must never eat the
laptop disk. Stats read all generations.
"""

import datetime
import json
import os
from pathlib import Path

from . import config


def _ts():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Audit:
    def __init__(self, path, max_bytes=None, keep=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = (max_bytes if max_bytes is not None
                          else config.AUDIT_ROTATE_MB * 1024 * 1024)
        self.keep = keep if keep is not None else config.AUDIT_KEEP

    def _rotated(self, i):
        return self.path.parent / f"{self.path.name}.{i}"

    def _rotate(self):
        for i in range(self.keep - 1, 0, -1):
            src, dst = self._rotated(i), self._rotated(i + 1)
            if src.exists():
                src.replace(dst)
        self.path.replace(self._rotated(1))

    def log(self, stage, session_id=None, **fields):
        if (self.max_bytes and self.keep > 0 and self.path.exists()
                and self.path.stat().st_size > self.max_bytes):
            self._rotate()
        rec = {"ts": _ts(), "stage": stage}
        if session_id is not None:
            rec["session_id"] = session_id
        rec.update(fields)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

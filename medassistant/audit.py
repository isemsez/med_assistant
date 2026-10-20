"""JSONL audit trail: one record per pipeline stage per turn
(architecture.md §3[5] — question, raw answer, labeler outputs, tool results,
picks — for auditability).

Size-based rotation: when the active file crosses AUDIT_ROTATE_MB it shifts to
``audit.jsonl.1`` (up to AUDIT_KEEP generations) — the file must never eat the
laptop disk. Stats read all generations. Writes are serialized by a lock (the
Audit instance is shared across request threads) and a rotation that fails
(e.g. a concurrent stats reader holds the file open on Windows) is skipped,
not raised — the next write retries it.
"""

import datetime
import json
import os
import threading
from pathlib import Path

from . import config


TS_FMT = "%Y/%m/%d-%H:%M:%S"


def utc_ts():
    """One timestamp format across audit/store/health — Moscow local wall
    clock ("2026/10/17-14:33:45", no offset suffix; host TZ is Moscow),
    owner decision 2026-10-20. stats._parse_ts round-trips exactly this
    (legacy ISO-UTC lines still in rotation windows convert there), so it
    must never diverge per copy."""
    return datetime.datetime.now().strftime(TS_FMT)


# back-compat alias for older imports
_ts = utc_ts


class Audit:
    def __init__(self, path, max_bytes=None, keep=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = (max_bytes if max_bytes is not None
                          else config.AUDIT_ROTATE_MB * 1024 * 1024)
        self.keep = keep if keep is not None else config.AUDIT_KEEP
        self._lock = threading.Lock()

    def _rotated(self, i):
        return self.path.parent / f"{self.path.name}.{i}"

    def _rotate(self):
        for i in range(self.keep - 1, 0, -1):
            src, dst = self._rotated(i), self._rotated(i + 1)
            if src.exists():
                src.replace(dst)
        self.path.replace(self._rotated(1))

    def log(self, stage, session_id=None, **fields):
        rec = {"ts": utc_ts(), "stage": stage}
        if session_id is not None:
            rec["session_id"] = session_id
        rec.update(fields)
        with self._lock:
            if (self.max_bytes and self.keep > 0 and self.path.exists()
                    and self.path.stat().st_size > self.max_bytes):
                try:
                    self._rotate()
                except OSError:
                    pass  # rotation is best-effort; appending must never fail
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

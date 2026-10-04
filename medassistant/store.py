"""Persistent sessions (SQLite, WAL). Conversations survive restarts.

Tables: sessions(id, title, created_at, updated_at),
messages(id, session_id, role, content, appendix_json, reasoning, kind,
created_at). `appendix_json` on assistant rows holds the full AppendixResult
payload; `reasoning` holds the answerer's thinking text (restored collapsed on
reload); `kind` separates chat turns from manual «pdf » turns ('pdf' rows
never enter the answerer history).
"""

import datetime
import json
import shutil
import sqlite3
import threading
from pathlib import Path

from .audit import utc_ts

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content TEXT NOT NULL,
    appendix_json TEXT,
    reasoning TEXT,
    kind TEXT NOT NULL DEFAULT 'chat',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""

TITLE_MAX = 60


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        """Add columns introduced after a database was first created."""
        cols = {r["name"] for r in
                self.conn.execute("PRAGMA table_info(messages)")}
        if "reasoning" not in cols:
            self.conn.execute("ALTER TABLE messages ADD COLUMN reasoning TEXT")
        if "kind" not in cols:
            self.conn.execute(
                "ALTER TABLE messages ADD COLUMN kind TEXT NOT NULL DEFAULT 'chat'")

    def close(self):
        with self._lock:
            self.conn.close()

    # -- sessions ---------------------------------------------------------

    def create_session(self, title=""):
        now = utc_ts()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO sessions (title, created_at, updated_at) VALUES (?,?,?)",
                (title, now, now))
            self.conn.commit()
            return cur.lastrowid

    def touch(self, session_id, title=None):
        """Update updated_at; set title only when it is currently empty."""
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET updated_at=?, "
                "title=CASE WHEN title='' AND ? IS NOT NULL THEN ? ELSE title END "
                "WHERE id=?",
                (utc_ts(), title, title, session_id))
            self.conn.commit()

    def list_sessions(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT s.id, s.title, s.updated_at, "
                "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n_messages "
                "FROM sessions s ORDER BY s.updated_at DESC, s.id DESC").fetchall()
            return [dict(r) for r in rows]

    def session_exists(self, session_id):
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
            return row is not None

    def delete_session(self, session_id):
        """Remove a session with all its messages (one transaction)."""
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                return False
            self.conn.execute("DELETE FROM messages WHERE session_id=?",
                              (session_id,))
            self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self.conn.commit()
            return True

    # -- messages ---------------------------------------------------------

    def add_message(self, session_id, role, content, appendix_payload=None,
                    reasoning=None, kind="chat"):
        now = utc_ts()
        appendix_json = (json.dumps(appendix_payload, ensure_ascii=False)
                         if appendix_payload is not None else None)
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO messages (session_id, role, content, appendix_json, "
                "reasoning, kind, created_at) VALUES (?,?,?,?,?,?,?)",
                (session_id, role, content, appendix_json, reasoning, kind, now))
            self.conn.commit()
            return cur.lastrowid

    def set_appendix(self, message_id, appendix_payload):
        with self._lock:
            self.conn.execute(
                "UPDATE messages SET appendix_json=? WHERE id=?",
                (json.dumps(appendix_payload, ensure_ascii=False), message_id))
            self.conn.commit()

    def get_messages(self, session_id):
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, role, content, appendix_json, reasoning, kind, "
                "created_at FROM messages WHERE session_id=? ORDER BY id",
                (session_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["appendix"] = (json.loads(d.pop("appendix_json"))
                             if d["appendix_json"] else None)
            out.append(d)
        return out

    def history_window(self, session_id, window):
        """Last `window` chat messages as OpenAI chat messages (for the
        answerer). Manual pdf turns never enter the answerer context."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? "
                "AND kind='chat' ORDER BY id DESC LIMIT ?",
                (session_id, window)).fetchall()
            return [{"role": r["role"], "content": r["content"]}
                    for r in reversed(rows)]

    def last_message(self, session_id, role, kind="chat"):
        """Content of the latest message of `role`/`kind`, or None."""
        with self._lock:
            row = self.conn.execute(
                "SELECT content FROM messages WHERE session_id=? AND role=? "
                "AND kind=? ORDER BY id DESC LIMIT 1",
                (session_id, role, kind)).fetchone()
            return row["content"] if row else None

    def truncate_from(self, session_id, message_id):
        """Delete `message_id` and every later message in the session.

        Rewind-to-question: the message itself goes too (it returns to the
        composer). Returns the number of deleted rows, or None when the
        message does not belong to the session.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM messages WHERE id=? AND session_id=?",
                (message_id, session_id)).fetchone()
            if row is None:
                return None
            cur = self.conn.execute(
                "DELETE FROM messages WHERE session_id=? AND id>=?",
                (session_id, message_id))
            self.conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (utc_ts(), session_id))
            self.conn.commit()
            return cur.rowcount

    # -- convenience ------------------------------------------------------

    def new_session_titled(self, first_question):
        title = first_question.strip().replace("\n", " ")
        if len(title) > TITLE_MAX:
            title = title[:TITLE_MAX - 1] + "…"
        return self.create_session(title)


def load_store(path=None):
    from . import config
    return Store(path or config.SESSIONS_DB)


def backup(path=None, keep=None):
    """Checkpoint WAL and copy the sessions DB to data/backups/ (keep newest N).

    Called at server startup — physician conversations are the most valuable
    local state and the cheapest to protect. Backups stay on this machine
    (laptop-only deployment). Returns the backup path or None when the DB
    does not exist yet.
    """
    from . import config
    db_path = Path(path or config.SESSIONS_DB)
    if not db_path.exists():
        return None
    keep = config.SESSIONS_BACKUP_KEEP if keep is None else keep
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    backups = db_path.parent / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    dest = backups / f"sessions-{stamp}.sqlite"
    shutil.copy2(db_path, dest)
    olds = sorted(backups.glob("sessions-*.sqlite"), reverse=True)  # by stamp
    for old in olds[keep:]:
        old.unlink()
    return dest

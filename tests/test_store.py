import tempfile
import unittest
from pathlib import Path

from medassistant.store import Store


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "s.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_create_and_list(self):
        sid = self.store.create_session("Вопрос о ХСН")
        sessions = self.store.list_sessions()
        self.assertEqual(sessions[0]["id"], sid)
        self.assertEqual(sessions[0]["title"], "Вопрос о ХСН")
        self.assertEqual(sessions[0]["n_messages"], 0)
        self.assertTrue(self.store.session_exists(sid))
        self.assertFalse(self.store.session_exists(999))

    def test_messages_roundtrip_with_appendix(self):
        sid = self.store.create_session()
        self.store.add_message(sid, "user", "вопрос")
        mid = self.store.add_message(sid, "assistant", "ответ",
                                     appendix_payload={"status": "ok",
                                                       "appendix_text": "блок"})
        msgs = self.store.get_messages(sid)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["appendix"]["status"], "ok")

        self.store.set_appendix(mid, {"status": "no_disease"})
        self.assertEqual(self.store.get_messages(sid)[1]["appendix"]["status"],
                         "no_disease")

    def test_reasoning_roundtrip(self):
        sid = self.store.create_session()
        self.store.add_message(sid, "user", "вопрос")
        self.store.add_message(sid, "assistant", "ответ", reasoning="думал…")
        self.store.add_message(sid, "assistant", "без размышлений")
        msgs = self.store.get_messages(sid)
        self.assertEqual(msgs[1]["reasoning"], "думал…")
        self.assertIsNone(msgs[2]["reasoning"])

    def test_migrates_legacy_db_without_reasoning_column(self):
        import sqlite3
        p = Path(self.tmp.name) / "legacy.db"
        conn = sqlite3.connect(p)
        conn.executescript("""
            CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL);
            CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL, content TEXT NOT NULL, appendix_json TEXT,
                created_at TEXT NOT NULL);
        """)
        conn.commit()
        conn.close()
        store = Store(p)  # must add the missing column, not crash
        sid = store.create_session()
        store.add_message(sid, "assistant", "ответ", reasoning="r")
        self.assertEqual(store.get_messages(sid)[0]["reasoning"], "r")
        store.close()

    def test_history_window_order_and_limit(self):
        sid = self.store.create_session()
        for i in range(5):
            self.store.add_message(sid, "user", f"q{i}")
            self.store.add_message(sid, "assistant", f"a{i}")
        h = self.store.history_window(sid, 4)
        self.assertEqual([m["content"] for m in h], ["q3", "a3", "q4", "a4"])
        self.assertEqual(h[0]["role"], "user")

    def test_touch_title_only_when_empty(self):
        sid = self.store.create_session("")
        self.store.touch(sid, title="первый вопрос")
        self.assertEqual(self.store.list_sessions()[0]["title"], "первый вопрос")
        self.store.touch(sid, title="второй не должен попасть")
        self.assertEqual(self.store.list_sessions()[0]["title"], "первый вопрос")

    def test_new_session_titled_truncates(self):
        sid = self.store.new_session_titled("д" * 100)
        title = self.store.list_sessions()[0]["title"]
        self.assertEqual(sid, self.store.list_sessions()[0]["id"])
        self.assertLessEqual(len(title), 60)
        self.assertTrue(title.endswith("…"))

    def test_truncate_from_deletes_suffix_only(self):
        sid = self.store.create_session()
        other = self.store.create_session()
        ids = []
        for i in range(3):
            ids.append(self.store.add_message(sid, "user", f"q{i}"))
            ids.append(self.store.add_message(sid, "assistant", f"a{i}"))
        other_mid = self.store.add_message(other, "user", "other")
        self.assertEqual(self.store.truncate_from(sid, ids[2]), 4)  # q1..a2
        self.assertEqual([m["content"] for m in self.store.get_messages(sid)],
                         ["q0", "a0"])
        self.assertEqual(len(self.store.get_messages(other)), 1)
        self.assertGreater(other_mid, 0)

    def test_truncate_from_rejects_foreign_or_unknown_message(self):
        s1 = self.store.create_session()
        s2 = self.store.create_session()
        mid = self.store.add_message(s1, "user", "q")
        self.assertIsNone(self.store.truncate_from(s2, mid))
        self.assertIsNone(self.store.truncate_from(s1, 999))
        self.assertEqual(len(self.store.get_messages(s1)), 1)

    def test_truncate_all_leaves_empty_titled_session(self):
        sid = self.store.create_session("t")
        mid = self.store.add_message(sid, "user", "q")
        self.assertEqual(self.store.truncate_from(sid, mid), 1)
        self.assertEqual(self.store.get_messages(sid), [])
        row = self.store.list_sessions()[0]
        self.assertEqual(row["n_messages"], 0)
        self.assertEqual(row["title"], "t")

    def test_delete_session_removes_messages_only_own(self):
        sid = self.store.create_session("t")
        self.store.add_message(sid, "user", "q")
        other = self.store.create_session("o")
        self.store.add_message(other, "user", "keep")
        self.assertTrue(self.store.delete_session(sid))
        self.assertFalse(self.store.session_exists(sid))
        self.assertEqual([s["id"] for s in self.store.list_sessions()], [other])
        self.assertEqual(len(self.store.get_messages(other)), 1)
        self.assertFalse(self.store.delete_session(999))

    def test_persistence_across_reopen(self):
        sid = self.store.create_session("t")
        self.store.add_message(sid, "user", "вопрос")
        self.store.close()
        store2 = Store(Path(self.tmp.name) / "s.db")
        self.assertEqual(store2.list_sessions()[0]["title"], "t")
        self.assertEqual(store2.get_messages(sid)[0]["content"], "вопрос")
        store2.close()


class TestDocsMeta(unittest.TestCase):
    def test_real_docs_meta(self):
        from medassistant import config
        if not config.DOCS_META_PATH.exists():
            self.skipTest("docs_meta.json not present")
        docs = config.load_docs_meta()
        self.assertEqual(len(docs), 757)
        for cv, meta in list(docs.items())[:20]:
            self.assertTrue(meta["name"].strip())
            self.assertTrue(meta["url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()

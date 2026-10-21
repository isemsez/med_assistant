import datetime
import json
import tempfile
import unittest
from pathlib import Path

from medassistant.audit import Audit, TS_FMT, utc_ts


class TestTsFormat(unittest.TestCase):
    def test_utc_ts_moscow_local_format(self):
        # owner decision 2026-10-20: "2026/10/17-14:33:45", Moscow local,
        # no offset suffix — and stats._parse_ts round-trips it exactly
        ts = utc_ts()
        self.assertRegex(ts, r"^\d{4}/\d{2}/\d{2}-\d{2}:\d{2}:\d{2}$")
        dt = datetime.datetime.strptime(ts, TS_FMT)
        self.assertAlmostEqual((dt - datetime.datetime.now()).total_seconds(),
                               0, delta=2)
        self.assertEqual(
            __import__("medassistant.stats", fromlist=["_parse_ts"])._parse_ts(
                {"ts": ts}),
            dt)


class TestRotation(unittest.TestCase):
    def test_rotate_on_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            a = Audit(p, max_bytes=200, keep=2)
            for i in range(3):  # each wave of records exceeds 200 bytes
                for j in range(5):
                    a.log("question", session_id=i, question=f"q{i}-{j}" * 5)
                self.assertTrue(p.exists())
            self.assertTrue((Path(tmp) / "audit.jsonl.1").exists())
            self.assertTrue((Path(tmp) / "audit.jsonl.2").exists())
            self.assertFalse((Path(tmp) / "audit.jsonl.3").exists(),
                             "keep=2 caps generations at .2")
            cur = p.read_text(encoding="utf-8")
            self.assertIn("q2-", cur)  # newest wave stays in the active file

    def test_no_rotation_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            a = Audit(p, max_bytes=1_000_000, keep=3)
            a.log("question", question="q")
            self.assertFalse((Path(tmp) / "audit.jsonl.1").exists())

    def test_keep_zero_disables(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            a = Audit(p, max_bytes=50, keep=0)
            a.log("question", question="x" * 200)
            self.assertTrue(p.exists())
            self.assertFalse((Path(tmp) / "audit.jsonl.1").exists())

    def test_failed_rotation_never_blocks_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            a = Audit(p, max_bytes=10, keep=3)
            a.log("question", question="seed")  # file now above threshold

            def boom():
                raise OSError("sharing violation")

            a._rotate = boom
            a.log("answer", answer="still written")  # rotation skipped
            text = p.read_text(encoding="utf-8")
            self.assertIn("still written", text)
            self.assertIn("answer", text.splitlines()[-1])

    def test_concurrent_writes_all_landed(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            a = Audit(p, max_bytes=10_000_000, keep=3)
            errs = []

            def writer(tag):
                try:
                    for i in range(50):
                        a.log("question", q=f"{tag}-{i}")
                except Exception as e:  # noqa: BLE001
                    errs.append(e)

            threads = [threading.Thread(target=writer, args=(t,)) for t in "abcd"]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errs, [])
            lines = p.read_text(encoding="utf-8").splitlines()
            # 200 records + a blank separator before each «question» record
            self.assertEqual(len(lines), 400)
            import json as _json
            for ln in lines:
                if ln.strip():
                    self.assertTrue(_json.loads(ln)["q"])


class TestTurnSeparation(unittest.TestCase):
    def test_question_record_preceded_by_blank_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            a = Audit(p, max_bytes=1_000_000, keep=3)
            a.log("question", question="q1")
            a.log("raw_answer", answer="a1")
            a.log("question", question="q2")
            lines = p.read_text(encoding="utf-8").splitlines()
            recs = [(i, json.loads(ln)) for i, ln in enumerate(lines)
                    if ln.strip()]
            self.assertEqual([r["stage"] for _, r in recs],
                             ["question", "raw_answer", "question"])
            for i, r in recs:
                if r["stage"] == "question":
                    self.assertEqual(lines[i - 1], "",
                                     "blank line must precede «question»")
                else:
                    self.assertNotEqual(lines[i - 1], "",
                                        "only «question» gets a separator")


if __name__ == "__main__":
    unittest.main()

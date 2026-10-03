import datetime
import json
import tempfile
import unittest
from pathlib import Path

from medassistant import stats

NOW = datetime.datetime(2026, 10, 2, 12, 0, tzinfo=datetime.timezone.utc)


def ts(minutes_ago):
    return (NOW - datetime.timedelta(minutes=minutes_ago)).isoformat()


class TestCollect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "audit.jsonl"

    def write(self, recs):
        self.path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
            encoding="utf-8")

    def rec(self, minutes_ago, **kw):
        return {"ts": ts(minutes_ago), **kw}

    def test_counters_and_window(self):
        self.write([
            self.rec(60, stage="question", question="q1"),
            self.rec(59, stage="raw_answer", answer="a1", dur_s=41.0),
            self.rec(58, stage="labeler_extract", raw="{}", dur_s=2.5),
            self.rec(57, stage="diseases", diseases=["ХСН"]),
            self.rec(56, stage="search", disease="ХСН", cards=["156_2"], dur_s=1.0),
            self.rec(55, stage="picks", disease="ХСН", picks=["156_2"], auto=True),
            self.rec(54, stage="appendix_done", status="ok"),
            self.rec(10, stage="question", question="q2"),
            self.rec(9, stage="stage_stop", stop_stage="labeler_select",
                     error="no JSON"),
            self.rec(8, stage="appendix_crash", error="boom"),
            # outside a 1-day window by 8 days
            self.rec(60 * 24 * 8, stage="question", question="old"),
        ])
        s = stats.collect(self.path, days=1, now=NOW)
        self.assertEqual(s["turns"], 2)
        self.assertEqual(s["answers"], 1)
        self.assertEqual(s["appendix_status"], {"ok": 1})
        self.assertEqual(s["stage_stop"], {"labeler_select": 1})
        self.assertEqual(s["appendix_crash"], 1)
        self.assertEqual(s["picks"], {"calls": 1, "auto": 1})
        self.assertEqual(s["latency_s"]["answerer"], {"n": 1, "mean": 41.0, "max": 41.0})
        self.assertEqual(s["latency_s"]["extract"], {"n": 1, "mean": 2.5, "max": 2.5})
        self.assertEqual(s["latency_s"]["select"], {"n": 0, "mean": None, "max": None})
        self.assertEqual(len(s["last_errors"]), 2)
        self.assertEqual(s["last_errors"][0]["stage"], "appendix_crash")
        self.assertEqual(s["last_errors"][1]["stop_stage"], "labeler_select")

    def test_legacy_stage_stop_without_stop_stage(self):
        # pre-fix stage_stop records were never actually written (the old
        # duplicate-kwarg call raised TypeError), but if one ever appears it
        # must not crash aggregation
        self.write([self.rec(5, stage="stage_stop", error="x")])
        s = stats.collect(self.path, days=1, now=NOW)
        self.assertEqual(s["stage_stop"], {"?": 1})

    def test_reads_rotated_generations(self):
        old = [self.rec(60 * 24 * 3, stage="question", question="old-day3"),
               self.rec(60 * 24 * 2, stage="stage_stop", stop_stage="search",
                        error="e1")]
        new = [self.rec(30, stage="question", question="recent")]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        (self.path.parent / "audit.jsonl.2").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in old) + "\n",
            encoding="utf-8")
        (self.path.parent / "audit.jsonl.1").write_text("", encoding="utf-8")
        self.write(new)
        s = stats.collect(self.path, days=7, now=NOW)
        self.assertEqual(s["turns"], 2)
        self.assertEqual(s["stage_stop"], {"search": 1})
        self.assertEqual(s["last_errors"][0]["stop_stage"], "search")

    def test_skipped_records_counted(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        (self.path.parent / "audit.jsonl.1").write_text(
            "{broken\n" + json.dumps({"stage": "question", "ts": "not-a-date"})
            + "\n", encoding="utf-8")
        self.write([self.rec(5, stage="question", question="ok")])
        s = stats.collect(self.path, days=7, now=NOW)
        self.assertEqual(s["turns"], 1)
        self.assertEqual(s["skipped_records"], 2)

    def test_bad_lines_and_missing_file(self):
        self.path.write_text("{broken\n\n" + json.dumps(
            {"ts": ts(1), "stage": "question"}) + "\n", encoding="utf-8")
        s = stats.collect(self.path, days=1, now=NOW)
        self.assertEqual(s["turns"], 1)
        empty = stats.collect(Path(self.tmp) / "nope.jsonl", days=1, now=NOW)
        self.assertEqual(empty["turns"], 0)
        self.assertEqual(empty["last_errors"], [])


if __name__ == "__main__":
    unittest.main()

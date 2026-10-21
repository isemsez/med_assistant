import json
import tempfile
import unittest
from pathlib import Path

from medassistant import llm, regression
from medassistant.audit import Audit


class FakeIndex:
    def __init__(self, cards):
        self.cards = cards

    def search(self, variants, embed, top_k=5):
        self.variants = variants
        return self.cards


class FakePipe:
    def __init__(self, cards=None, answer="Ответ про ХСН.", result=None,
                 audit=None, loop=False):
        self.abbr_map = None
        self.index = FakeIndex(cards or [])
        self._answer = answer
        self.result = result
        self.audit = audit
        self.loop = loop

    def stream_answer(self, question, history=None, max_tokens=None):
        if isinstance(self._answer, Exception):
            raise self._answer
        yield {"type": "content", "text": self._answer}

    def run_appendix(self, question, raw_answer, session_id=None):
        # a turn that hit the select loop-guard writes the audit row the
        # full-tier runner must treat as a FAILED row
        if self.audit is not None and self.loop:
            self.audit.log("labeler_select_loop", disease="X",
                           reason="item x4: …echo", dropped=["2_1"])
        return self.result


def make_result(status="ok", picks=()):
    from medassistant.pipeline import AppendixResult
    groups = [{"disease": "X", "auto_pick": False,
               "picks": [{"code_version": cv, "name": name, "url": "u"}
                         for cv, name in picks]}] if picks else []
    return AppendixResult(status, by_disease=groups)


class TestLoadGoldens(unittest.TestCase):
    def test_tiers_comments_and_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "g.jsonl"
            p.write_text(
                "# comment\n\n"
                '{"disease": "ХСН", "expect_name_contains": "недостаточн"}\n'
                '{"question": "q", "expect_status": "no_disease"}\n',
                encoding="utf-8")
            rows = regression.load_goldens(p)
            self.assertEqual([r["_tier"] for r in rows], ["search", "full"])

    def test_invalid_row_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "g.jsonl"
            p.write_text('{"question": "q", "disease": "d"}\n', encoding="utf-8")
            with self.assertRaises(SystemExit):
                regression.load_goldens(p)
            p.write_text("{broken json\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                regression.load_goldens(p)

    def test_missing_file(self):
        with self.assertRaises(SystemExit):
            regression.load_goldens(Path("nope.jsonl"))


class TestSearchRow(unittest.TestCase):
    def test_pass_and_fail(self):
        pipe = FakePipe(cards=[{"code_version": "156_2",
                                "name": "Хроническая сердечная недостаточность",
                                "score": 0.91}])
        row = {"disease": "ХСН", "expect_name_contains": "недостаточн"}
        ok, _ = regression.run_search_row(pipe, row)
        self.assertTrue(ok)
        pipe = FakePipe(cards=[{"code_version": "1_1", "name": "Грипп",
                                "score": 0.5}])
        ok, detail = regression.run_search_row(pipe, row)
        self.assertFalse(ok)
        self.assertIn("недостаточн", detail)

    def test_no_candidates(self):
        ok, detail = regression.run_search_row(
            FakePipe(), {"disease": "X", "expect_name_contains": "y"})
        self.assertFalse(ok)


class TestFullRow(unittest.TestCase):
    ROW = {"question": "q?", "expect_name_contains": "недостаточн"}

    def test_ok_with_expected_pick(self):
        pipe = FakePipe(result=make_result(
            picks=[("156_2", "Хроническая сердечная недостаточность")]))
        ok, detail, res = regression.run_full_row(pipe, self.ROW, 50)
        self.assertTrue(ok)
        self.assertIn("ok", detail)

    def test_status_mismatch(self):
        pipe = FakePipe(result=make_result("no_disease"))
        row = {"question": "q?", "expect_status": "ok"}
        ok, detail, _ = regression.run_full_row(pipe, row, 50)
        self.assertFalse(ok)
        self.assertIn("status", detail)

    def test_expected_status_no_disease(self):
        pipe = FakePipe(result=make_result("no_disease"))
        row = {"question": "q?", "expect_status": "no_disease"}
        ok, _, _ = regression.run_full_row(pipe, row, 50)
        self.assertTrue(ok)

    def test_pick_missing(self):
        pipe = FakePipe(result=make_result(picks=[("1_1", "Грипп")]))
        ok, detail, _ = regression.run_full_row(pipe, self.ROW, 50)
        self.assertFalse(ok)
        self.assertIn("Грипп", detail)

    def test_answerer_error_and_empty(self):
        pipe = FakePipe(answer=llm.LLMError("HTTP 500"))
        ok, detail, _ = regression.run_full_row(pipe, self.ROW, 50)
        self.assertFalse(ok)
        pipe = FakePipe(answer="   ")
        ok, detail, _ = regression.run_full_row(pipe, self.ROW, 50)
        self.assertFalse(ok)
        self.assertIn("empty", detail)

    def test_select_loop_fails_row_even_with_correct_picks(self):
        # 2026-10-21 (owner): a select thinking loop is a FAILED full-tier
        # row even when the title guard or auto-pick shipped correct picks
        # afterwards — the runtime degrades gracefully, the golden set
        # demands a healthy selector, nothing is "greenified"
        with tempfile.TemporaryDirectory() as tmp:
            audit = Audit(Path(tmp) / "audit.jsonl")
            pipe = FakePipe(audit=audit, loop=True, result=make_result(
                picks=[("156_2", "Хроническая сердечная недостаточность")]))
            ok, detail, _ = regression.run_full_row(pipe, self.ROW, 50)
        self.assertFalse(ok)
        self.assertIn("select reasoning loop", detail)
        self.assertIn("item x4", detail)

    def test_no_loop_records_row_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = Audit(Path(tmp) / "audit.jsonl")
            pipe = FakePipe(audit=audit, result=make_result(
                picks=[("156_2", "Хроническая сердечная недостаточность")]))
            ok, detail, _ = regression.run_full_row(pipe, self.ROW, 50)
        self.assertTrue(ok)

    def test_loop_only_this_row_not_earlier_ones(self):
        # the shared throwaway audit accumulates rows — a loop written by
        # an EARLIER row must not fail later clean rows (slice by line)
        with tempfile.TemporaryDirectory() as tmp:
            audit = Audit(Path(tmp) / "audit.jsonl")
            audit.log("labeler_select_loop", disease="old",
                      reason="item x5: …", dropped=[])
            pipe = FakePipe(audit=audit, result=make_result(
                picks=[("156_2", "Хроническая сердечная недостаточность")]))
            ok, detail, _ = regression.run_full_row(pipe, self.ROW, 50)
        self.assertTrue(ok)

    def test_over_pick_fails_row(self):
        # 2026-10-21: the per-group 4B selector rubber-stamps (live: all
        # candidates picked) — expect_picks_max fails the row even when
        # the expected guideline is somewhere in the pile
        row = {"question": "q?", "expect_name_contains": "недостаточн",
               "expect_picks_max": 2}
        pipe = FakePipe(result=make_result(picks=[
            ("156_2", "Хроническая сердечная недостаточность"),
            ("1_1", "Грипп"), ("2_2", "Другое")]))
        ok, detail, _ = regression.run_full_row(pipe, row, 50)
        self.assertFalse(ok)
        self.assertIn("expect_picks_max", detail)
        self.assertIn("rubber-stamp", detail)
        # at the cap the row passes
        pipe = FakePipe(result=make_result(picks=[
            ("156_2", "Хроническая сердечная недостаточность")]))
        ok, _, _ = regression.run_full_row(pipe, row, 50)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from medassistant import llm, regression


class FakeIndex:
    def __init__(self, cards):
        self.cards = cards

    def search(self, variants, embed, top_k=5):
        self.variants = variants
        return self.cards


class FakePipe:
    def __init__(self, cards=None, answer="Ответ про ХСН.", result=None):
        self.abbr_map = None
        self.index = FakeIndex(cards or [])
        self._answer = answer
        self.result = result

    def stream_answer(self, question, history=None, max_tokens=None):
        if isinstance(self._answer, Exception):
            raise self._answer
        yield {"type": "content", "text": self._answer}

    def run_appendix(self, question, raw_answer, session_id=None):
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


if __name__ == "__main__":
    unittest.main()

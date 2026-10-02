import unittest

from medassistant.abbr import Abbr, parse_abbr2term

SAMPLE = """ААБК: Американская ассоциация банков крови
ХСН: хроническая сердечная недостаточность
ИБС: ишемическая болезнь сердца
lowercase-no-match: irrelevant
"""


class TestAbbr(unittest.TestCase):
    def test_parse(self):
        m = parse_abbr2term(SAMPLE)
        self.assertEqual(m["ХСН"], "хроническая сердечная недостаточность")
        self.assertNotIn("LOWER", {k[:5] for k in m})

    def test_variants_original_first(self):
        a = Abbr(parse_abbr2term(SAMPLE))
        self.assertEqual(a.variants("ХСН"), ["ХСН", "хроническая сердечная недостаточность"])

    def test_lowercase_tokens_skipped(self):
        a = Abbr(parse_abbr2term(SAMPLE))
        self.assertEqual(a.variants("хроническая усталость"), ["хроническая усталость"])

    def test_expansion_never_replaces(self):
        a = Abbr(parse_abbr2term(SAMPLE))
        v = a.variants("декомпенсация ХСН")
        self.assertEqual(v[0], "декомпенсация ХСН")
        self.assertIn("хроническая сердечная недостаточность", v)

    def test_multiple_and_cap(self):
        a = Abbr(parse_abbr2term(SAMPLE))
        v = a.variants("ХСН и ИБС")
        self.assertEqual(v, ["ХСН и ИБС",
                             "хроническая сердечная недостаточность",
                             "ишемическая болезнь сердца"])
        self.assertLessEqual(len(a.variants("ХСН и ИБС", cap=2)), 2)

    def test_empty(self):
        a = Abbr({})
        self.assertEqual(a.variants("ХСН"), ["ХСН"])


if __name__ == "__main__":
    unittest.main()

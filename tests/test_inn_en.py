"""Offline tests for scripts/inn_to_en.py (RU МНН -> EN INN converter)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inn_to_en import InnConverter, build, parse_pairs  # noqa: E402

PAIRS = [
    ("ibuprofen", "ibuprofenum"),
    ("levomenthol", "levomentholum"),
    ("dexamethasone", "dexamethasonum"),
    ("paracetamol", "paracetamolum"),
    ("levothyroxine sodium", "levothyroxinum natricum"),
    ("ascorbic acid", "acidum ascorbicum"),
    ("azilsartan medoxomil", "azilsartanum medoxomilum"),
    ("alteplase", "alteplasum"),
    ("meglumine", "megluminum"),
    ("hydrochlorothiazide", "hydrochlorothiazidum"),
    ("clotrimazole", "clotrimazolum"),
    ("thiamine", "thiaminum"),
    ("tryptophan", "tryptophanum"),
]


class TestParsePairs(unittest.TestCase):
    def test_tab_pairs_only(self):
        text = "ibuprofen\tibuprofenum\n\nnoise line\nx\ty\n"
        self.assertEqual(parse_pairs(text),
                         [("ibuprofen", "ibuprofenum"), ("x", "y")])


class TestConvert(unittest.TestCase):
    def setUp(self):
        self.conv = InnConverter(PAIRS)

    def test_plain(self):
        self.assertEqual(self.conv.convert("ибупрофен"), "ibuprofen")
        self.assertEqual(self.conv.convert("дексаметазон"), "dexamethasone")

    def test_th_ru_drops_h(self):
        self.assertEqual(self.conv.convert("левоментол"), "levomenthol")
        self.assertEqual(self.conv.convert("тиамин"), "thiamine")

    def test_hydro_chlor_stems(self):
        self.assertEqual(self.conv.convert("гидрохлоротиазид"),
                         "hydrochlorothiazide")

    def test_c_k_unification(self):
        self.assertEqual(self.conv.convert("клотримазол"), "clotrimazole")

    def test_salt_multi_token_per_word_um(self):
        self.assertEqual(self.conv.convert("левотироксин натрия"),
                         "levothyroxine sodium")

    def test_feminine_a_endings(self):
        self.assertEqual(self.conv.convert("алтеплаза"), "alteplase")
        self.assertEqual(self.conv.convert("азилсартана медоксомил"),
                         "azilsartan medoxomil")

    def test_iu_u_variant(self):
        self.assertEqual(self.conv.convert("меглюмин"), "meglumine")

    def test_acid_stem_prefix(self):
        self.assertEqual(self.conv.convert("аскорбиновая кислота"),
                         "ascorbic acid")

    def test_composite_all_or_nothing(self):
        self.assertIsNone(self.conv.convert("ибупрофен + загадка"))
        self.assertEqual(self.conv.convert("ибупрофен + парацетамол"),
                         "ibuprofen + paracetamol")

    def test_no_partial_word_fallback(self):
        # a descriptive phrase must never gloss via one of its words
        self.assertIsNone(self.conv.convert("триптофана натриевая соль"))

    def test_unknown_returns_none(self):
        self.assertIsNone(self.conv.convert("загадочное вещество"))


class TestBuild(unittest.TestCase):
    def test_stats_and_entries(self):
        entries, stats, _fuzzy = build(
            PAIRS, ["ибупрофен", "загадка", "ибупрофен + загадка"])
        self.assertEqual(entries, {"ибупрофен": "ibuprofen"})
        self.assertEqual(stats["register_inns"], 3)
        self.assertEqual(stats["converted"], 1)


if __name__ == "__main__":
    unittest.main()

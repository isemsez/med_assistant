"""Offline tests for scripts/inn_to_en.py (RU МНН -> EN INN converter and
its drugs.sqlite inn_en/inn_lat columns)."""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inn_to_en import (InnConverter, apply_db, build,  # noqa: E402
                       parse_pairs)

PAIRS = [
    ("ibuprofen", "ibuprofenum"),
    ("paracetamol", "paracetamolum"),
    ("levomenthol", "levomentholum"),
    ("dexamethasone", "dexamethasonum"),
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

    def test_plain_returns_en_and_lat(self):
        self.assertEqual(self.conv.convert("ибупрофен"),
                         ("ibuprofen", "ibuprofenum"))
        self.assertEqual(self.conv.convert("дексаметазон"),
                         ("dexamethasone", "dexamethasonum"))

    def test_th_ru_drops_h(self):
        self.assertEqual(self.conv.convert("левоментол"),
                         ("levomenthol", "levomentholum"))
        self.assertEqual(self.conv.convert("тиамин"), ("thiamine", "thiaminum"))

    def test_hydro_chlor_stems(self):
        self.assertEqual(self.conv.convert("гидрохлоротиазид"),
                         ("hydrochlorothiazide", "hydrochlorothiazidum"))

    def test_c_k_unification(self):
        self.assertEqual(self.conv.convert("клотримазол"),
                         ("clotrimazole", "clotrimazolum"))

    def test_salt_multi_token_per_word_um(self):
        self.assertEqual(self.conv.convert("левотироксин натрия"),
                         ("levothyroxine sodium",
                          "levothyroxinum natricum"))

    def test_feminine_a_endings(self):
        self.assertEqual(self.conv.convert("алтеплаза"),
                         ("alteplase", "alteplasum"))
        self.assertEqual(self.conv.convert("азилсартана медоксомил"),
                         ("azilsartan medoxomil",
                          "azilsartanum medoxomilum"))

    def test_iu_u_variant(self):
        self.assertEqual(self.conv.convert("меглюмин"),
                         ("meglumine", "megluminum"))

    def test_acid_stem_prefix(self):
        self.assertEqual(self.conv.convert("аскорбиновая кислота"),
                         ("ascorbic acid", "acidum ascorbicum"))

    def test_composite_per_component_empty_slots(self):
        # парацетамол translates, загадка does not: empty slot, not NULL
        en, lat = self.conv.convert("загадка + парацетамол")
        self.assertEqual(en, " + paracetamol")
        self.assertEqual(lat, " + paracetamolum")

    def test_composite_nothing_translates_is_null(self):
        self.assertEqual(self.conv.convert("загадка"), (None, None))
        self.assertEqual(self.conv.convert("загадка + тайна"), (None, None))

    def test_no_partial_word_fallback(self):
        # a descriptive phrase must never gloss via one of its words
        self.assertEqual(self.conv.convert("триптофана натриевая соль"),
                         (None, None))

    def test_unknown_returns_none(self):
        self.assertIsNone(self.conv.convert_component("загадочное вещество"))


class TestBuild(unittest.TestCase):
    def test_stats_and_mapping(self):
        mapping, stats = build(
            PAIRS, ["ибупрофен", "загадка", "загадка + парацетамол"])
        self.assertEqual(mapping["ибупрофен"], ("ibuprofen", "ibuprofenum"))
        self.assertEqual(stats["register_inns"], 3)
        self.assertEqual(stats["fully_translated"], 1)
        self.assertEqual(stats["partially_translated"], 1)
        self.assertEqual(stats["untranslated"], 1)


class TestApplyDb(unittest.TestCase):
    def test_columns_written_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "drugs.sqlite"
            con = sqlite3.connect(str(db))
            con.execute("create table drugs (trade_name text, inn text)")
            con.executemany("insert into drugs values (?, ?)", [
                ("Ибустин", "Ибупрофен"),
                ("Сорбифер Дурулес", "Железа сульфат+[Аскорбиновая кислота]"),
                ("Загадин", "Загадка"),
            ])
            con.commit()
            con.close()
            apply_db(PAIRS, db)
            con = sqlite3.connect(str(db))
            try:
                rows = con.execute(
                    "select inn, inn_en, inn_lat from drugs order by rowid"
                ).fetchall()
            finally:
                con.close()
            self.assertEqual(rows, [
                ("Ибупрофен", "ibuprofen", "ibuprofenum"),
                ("Железа сульфат+[Аскорбиновая кислота]",
                 " + ascorbic acid", " + acidum ascorbicum"),
                ("Загадка", None, None),
            ])
            apply_db(PAIRS, db)  # second run: no error, same state
            con = sqlite3.connect(str(db))
            try:
                rows2 = con.execute(
                    "select inn_en from drugs order by rowid").fetchall()
            finally:
                con.close()
            self.assertEqual(rows2, [("ibuprofen",),
                                     (" + ascorbic acid",), (None,)])


if __name__ == "__main__":
    unittest.main()

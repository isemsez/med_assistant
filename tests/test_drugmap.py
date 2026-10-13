import sqlite3
import tempfile
import unittest
from pathlib import Path

from medassistant.drugmap import DrugMap, clean_inn, load

ROWS = [
    ("Сорбифер Дурулес", "Железа сульфат+[Аскорбиновая кислота]"),
    ("Сорбифер", "Железа сульфат"),
    ("Нурофен", "Ибупрофен"),
    ("Рузам концентрат", "~"),
    ("Кагоцел", "Кагоцел"),
]


def dmap():
    return DrugMap(ROWS)


class TestCleanInn(unittest.TestCase):
    def test_brackets_plus_and_case(self):
        self.assertEqual(clean_inn("Железа сульфат+[Аскорбиновая кислота]"),
                         "железа сульфат + аскорбиновая кислота")

    def test_placeholder_stays_placeholder(self):
        self.assertEqual(clean_inn("~"), "~")


class TestLookup(unittest.TestCase):
    def test_nominal_multiword(self):
        m = dmap().lookup("Что содержится в сорбифер дурулес")
        self.assertEqual([x["trade"] for x in m], ["Сорбифер Дурулес"])
        self.assertEqual(m[0]["inn"], "железа сульфат + аскорбиновая кислота")

    def test_inflected_genitive(self):
        m = dmap().lookup("Состав сорбифера дурулеса?")
        self.assertEqual(m[0]["trade"], "Сорбифер Дурулес")

    def test_hyphenated_name(self):
        m = dmap().lookup("совместимость нурофен-кагоцел")  # one hyphenated blob
        self.assertIn("Нурофен", [x["trade"] for x in m])

    def test_subsumed_short_name_dropped(self):
        m = dmap().lookup("в сорбифер дурулес")
        self.assertEqual([x["trade"] for x in m], ["Сорбифер Дурулес"])

    def test_short_name_alone_matches(self):
        m = dmap().lookup("дозировка сорбифера")
        self.assertEqual([x["trade"] for x in m], ["Сорбифер"])

    def test_latin_case_insensitive(self):
        m = DrugMap([("Nurofen", "Ибупрофен")]).lookup("дозировка NUROFEN")
        self.assertEqual([x["trade"] for x in m], ["Nurofen"])
        self.assertEqual(m[0]["inn"], "ибупрофен")

    def test_no_match_plain_words(self):
        self.assertEqual(dmap().lookup("какие симптомы у гриппа"), [])

    def test_no_prefix_poison(self):
        # a longer word starting like the trade name must not match
        self.assertEqual(dmap().lookup("сорбиферация препарата"), [])

    def test_placeholder_inn_returns_none(self):
        m = dmap().lookup("про рузам концентрат")
        self.assertEqual(m, [{"trade": "Рузам концентрат", "inn": None}])

    def test_trademark_sign_stripped_from_display(self):
        m = DrugMap([("Аспирин®", "Ацетилсалициловая кислота")])
        self.assertEqual(m.lookup("дозировка аспирина"),
                         [{"trade": "Аспирин",
                           "inn": "ацетилсалициловая кислота"}])

    def test_cap_keeps_first_occurrence(self):
        m = dmap().lookup("нурофен и кагоцел", cap=1)
        self.assertEqual([x["trade"] for x in m], ["Нурофен"])

    def test_multiple_matches_sorted_by_position(self):
        m = dmap().lookup("кагоцел после нурофен")
        self.assertEqual([x["trade"] for x in m], ["Кагоцел", "Нурофен"])

    def test_empty_question_and_empty_map(self):
        self.assertEqual(dmap().lookup(""), [])
        self.assertEqual(len(DrugMap([])), 0)


class TestLoad(unittest.TestCase):
    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(load(path=Path(td) / "nope.sqlite"))

    def test_load_reads_sqlite(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "drugs.sqlite"
            con = sqlite3.connect(str(p))
            con.execute("create table drugs (trade_name text, inn text,"
                        " pharm_group text)")
            con.execute(
                "insert into drugs values (?, ?, ?)",
                ("Сорбифер Дурулес", "Железа сульфат+[Аскорбиновая кислота]",
                 "антианемические"))
            con.commit()
            con.close()
            m = load(path=p)
            self.assertEqual(len(m), 1)
            hit = m.lookup("что содержится в сорбифер дурулес")
            self.assertEqual(hit[0]["trade"], "Сорбифер Дурулес")
            self.assertEqual(hit[0]["inn"],
                             "железа сульфат + аскорбиновая кислота")


if __name__ == "__main__":
    unittest.main()

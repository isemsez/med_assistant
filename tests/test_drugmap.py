import sqlite3
import tempfile
import unittest
from pathlib import Path

from medassistant.drugmap import DrugMap, clean_inn, load

ROWS = [
    ("Сорбифер Дурулес", "Железа сульфат+[Аскорбиновая кислота]"),
    ("Нурофен®", "Ибупрофен"),
    ("Нурофен® Форте", "Ибупрофен"),
    ("Нурофен® Экспресс", "Ибупрофен"),
    ("Нурофен® для детей", "Ибупрофен"),
    ("Нурофен® Плюс", "Ибупрофен+Кодеин"),
    ("Нурофен® Интенсив", "Ибупрофен+Парацетамол"),
    ("L-Тироксин 50", "Левотироксин натрия"),
    ("Д-Пантенол", "Декспантенол"),
    ("Д-Пантенол плюс антисептик", "Декспантенол+Хлоргексидин"),
    ("1,3-Диэтилбензол", "Диэтилбензол"),          # digit-led: skipped
    ("9 Месяцев Фолиевая кислота", "Фолиевая кислота"),  # digit-led: skipped
    ("Азитромицин", "Азитромицин"),
    ("Азитромицин Форте", "Азитромицин"),
    ("Валерианы настойка", "Валерианы лекарственной корневищ с корнями настойка"),
    ("Валерианы настойка Реневал", "Валерианы лекарственной корневищ с корнями настойка"),
    ("Валерианы экстракт", "Валерианы лекарственной корневищ с корнями экстракт"),
    ("Железо Комплекс", "Железа фумарат"),
    ("Кагоцел®", "~"),                              # placeholder МНН: not keyed
]

ALIASES = {"валерьянка": "Валерианы настойка",  # value = register trade name
           "кагоцел": "меглюмина акридонацетат"}  # value = direct МНН


def dmap(aliases=None, glosses=None):
    return DrugMap(ROWS, aliases=aliases, glosses=glosses)


class TestCleanInn(unittest.TestCase):
    def test_brackets_plus_and_case(self):
        self.assertEqual(clean_inn("Железа сульфат+[Аскорбиновая кислота]"),
                         "железа сульфат + аскорбиновая кислота")

    def test_placeholder_stays_placeholder(self):
        self.assertEqual(clean_inn("~"), "~")


class TestPrefixMatching(unittest.TestCase):
    def test_prepositional(self):
        m = dmap().lookup("что есть в сорбифере")
        self.assertEqual(m, [{"trade": "Сорбифер Дурулес",
                              "inn": "железа сульфат + аскорбиновая кислота",
                              "token": "сорбифере"}])

    def test_instrumental(self):
        m = dmap().lookup("что делать с сорбифером")
        self.assertEqual([x["trade"] for x in m], ["Сорбифер Дурулес"])

    def test_genitive_both_words(self):
        m = dmap().lookup("состав сорбифера дурулеса")
        self.assertEqual([x["trade"] for x in m], ["Сорбифер Дурулес"])

    def test_inflected_inn_token(self):
        m = dmap().lookup("дозировка азитромицина")
        self.assertEqual(m, [{"trade": "Азитромицин", "inn": "азитромицин",
                              "token": "азитромицина"}])

    def test_plain_words_no_match(self):
        self.assertEqual(dmap().lookup("какие симптомы у гриппа"), [])
        # degenerate keys never exist; tier 3 must stay silent too
        self.assertEqual(dmap().lookup("дозировка и лекарство"), [])

    def test_stoplisted_word_never_matches(self):
        # «железо» would prefix-match the key from «Железо Комплекс» —
        # the stoplist must suppress it
        self.assertEqual(dmap().lookup("можно ли железо каждый день"), [])

    def test_empty_question_and_empty_map(self):
        self.assertEqual(dmap().lookup(""), [])
        self.assertEqual(len(DrugMap([])), 0)


class TestLeadingSegments(unittest.TestCase):
    def test_stereo_prefix_keys_next_word(self):
        m = dmap().lookup("l-тироксин 100 дозировка")
        self.assertEqual(m, [{"trade": "L-Тироксин 50",
                              "inn": "левотироксин натрия",
                              "token": "тироксин"}])

    def test_cyrillic_stereo_prefix(self):
        m = dmap().lookup("крем д-пантенол плюс антисептик")
        # adjacent «плюс антисептик» restricts the ambiguous пантенол group
        self.assertEqual(m, [{"trade": "Д-Пантенол плюс антисептик",
                              "inn": "декспантенол + хлоргексидин",
                              "token": "пантенол"}])

    def test_bare_stereo_prefix_modal_picks_base(self):
        m = dmap().lookup("д-пантенол детям")
        self.assertEqual(m[0]["trade"], "Д-Пантенол")
        self.assertEqual(m[0]["inn"], "декспантенол")

    def test_digit_led_names_skipped(self):
        self.assertEqual(dmap().lookup("про 1,3-диэтилбензол"), [])
        self.assertEqual(dmap().lookup("фолиевая кислота при беременности"), [])


class TestAmbiguity(unittest.TestCase):
    def test_bare_token_modal_most_rows(self):
        m = dmap().lookup("дозировка нурофена")
        self.assertEqual(m, [{"trade": "Нурофен", "inn": "ибупрофен",
                              "token": "нурофена"}])

    def test_adjacent_second_word_restricts_plus(self):
        m = dmap().lookup("нурофен плюс от чего")
        self.assertEqual(m, [{"trade": "Нурофен Плюс",
                              "inn": "ибупрофен + кодеин",
                              "token": "нурофен"}])

    def test_adjacent_second_word_restricts_intensive(self):
        m = dmap().lookup("нурофен интенсив состав")
        self.assertEqual(m, [{"trade": "Нурофен Интенсив",
                              "inn": "ибупрофен + парацетамол",
                              "token": "нурофен"}])

    def test_non_adjacent_plus_does_not_restrict(self):
        m = dmap().lookup("плюс к нурофену витамин")
        self.assertEqual(m, [{"trade": "Нурофен", "inn": "ибупрофен",
                              "token": "нурофену"}])

    def test_or_word_between_does_not_restrict(self):
        m = dmap().lookup("нурофен или кагоцел")
        self.assertEqual(m, [{"trade": "Нурофен", "inn": "ибупрофен",
                              "token": "нурофен"}])

    def test_repeated_token_deduped(self):
        m = dmap().lookup("нурофен после нурофена")
        self.assertEqual([x["trade"] for x in m], ["Нурофен"])

    def test_cap_keeps_first_occurrence(self):
        m = dmap().lookup("нурофен и сорбифер", cap=1)
        self.assertEqual([x["trade"] for x in m], ["Нурофен"])


class TestAliases(unittest.TestCase):
    def test_alias_to_register_name(self):
        m = dmap(aliases=ALIASES).lookup("валерьянка — это лекарство?")
        self.assertEqual(m, [{"trade": "Валерианы настойка",
                              "inn": "валерианы лекарственной корневищ"
                                     " с корнями настойка",
                              "via": "валерьянка"}])

    def test_alias_direct_inn_for_register_hole(self):
        m = dmap(aliases=ALIASES).lookup("кагоцел безопасен?")
        self.assertEqual(m, [{"trade": "Кагоцел",
                              "inn": "меглюмина акридонацетат"}])

    def test_alias_tier_skipped_when_dict_matched(self):
        m = dmap(aliases=ALIASES).lookup("нурофен или кагоцел")
        self.assertEqual([x["trade"] for x in m], ["Нурофен"])

    def test_no_aliases_no_tier3(self):
        self.assertEqual(dmap().lookup("валерьянка — это лекарство?"), [])


class TestFuzzy(unittest.TestCase):
    def test_tight_typo_matches_unambiguous_key(self):
        m = dmap().lookup("дозировка азитромицна")  # missing и
        self.assertEqual(m, [{"trade": "Азитромицин", "inn": "азитромицин",
                              "token": "азитромицна"}])

    def test_short_token_never_fuzzy(self):
        self.assertEqual(dmap().lookup("нурофн дозировка"), [])

    def test_ambiguous_key_never_fuzzy(self):
        # «нурофкн» is within Levenshtein 2 of the ambiguous key «нурофе» —
        # the typo tier only accepts unambiguous keys, so nothing matches
        self.assertEqual(dmap().lookup("нурофкн дозировка"), [])


class TestGlosses(unittest.TestCase):
    GLOSSES = {"железа сульфат + аскорбиновая кислота":
               "ferrous sulfate + ascorbic acid"}

    def test_gloss_attached_to_matching_inn(self):
        m = dmap(glosses=self.GLOSSES).lookup("что есть в сорбифере")
        self.assertEqual(m[0]["en"], "ferrous sulfate + ascorbic acid")

    def test_no_gloss_no_field(self):
        m = dmap(glosses=self.GLOSSES).lookup("дозировка нурофена")
        self.assertNotIn("en", m[0])

    def test_gloss_covers_alias_matches_too(self):
        glosses = {**self.GLOSSES,
                   "валерианы лекарственной корневищ с корнями настойка":
                       "valerian tincture"}
        m = dmap(aliases=ALIASES, glosses=glosses).lookup(
            "валерьянка помогает?")
        self.assertEqual(m[0]["en"], "valerian tincture")

    def test_load_reads_gloss_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "drugs.sqlite"
            con = sqlite3.connect(str(p))
            con.execute("create table drugs (trade_name text, inn text)")
            con.execute("insert into drugs values (?, ?)",
                        ("Валидол", "Левоментола раствор в ментилизовалерате"))
            con.commit()
            con.close()
            g = Path(td) / "gloss.yaml"
            g.write_text(
                "# gloss\nлевоментола раствор в ментилизовалерате:"
                " levomenthol in menthyl isovalerate\n", encoding="utf-8")
            m = load(path=p, aliases_path=Path(td) / "no-aliases.yaml",
                     gloss_path=g, inn_en_path=Path(td) / "no-inn-en.json")
            self.assertEqual(
                m.lookup("как действует валидол")[0]["en"],
                "levomenthol in menthyl isovalerate")


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

    def test_load_reads_alias_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "drugs.sqlite"
            con = sqlite3.connect(str(p))
            con.execute("create table drugs (trade_name text, inn text)")
            con.execute("insert into drugs values (?, ?)",
                        ("Валерианы настойка",
                         "Валерианы лекарственной корневищ с корнями настойка"))
            con.commit()
            con.close()
            a = Path(td) / "aliases.yaml"
            a.write_text("# comment\nвалерьянка: Валерианы настойка\n",
                         encoding="utf-8")
            m = load(path=p, aliases_path=a,
                     gloss_path=Path(td) / "no-gloss.yaml",
                     inn_en_path=Path(td) / "no-inn-en.json")
            hit = m.lookup("валерьянка помогает?")
            self.assertEqual(hit[0]["trade"], "Валерианы настойка")

    def test_load_missing_alias_file_ok(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "drugs.sqlite"
            con = sqlite3.connect(str(p))
            con.execute("create table drugs (trade_name text, inn text)")
            con.execute("insert into drugs values (?, ?)",
                        ("Кагоцел", "меглюмина акридонацетат"))
            con.commit()
            con.close()
            m = load(path=p, aliases_path=Path(td) / "nope.yaml",
                     gloss_path=Path(td) / "no-gloss.yaml",
                     inn_en_path=Path(td) / "no-inn-en.json")
            self.assertEqual(m.lookup("кагоцел?"),
                             [{"trade": "Кагоцел",
                               "inn": "меглюмина акридонацетат",
                               "token": "кагоцел"}])

    def test_generated_map_and_curated_gloss_merge(self):
        # the generated inn_en.json glosses the convertible mass; the
        # curated gloss file covers the true WHO holes and OVERRIDES it
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "drugs.sqlite"
            con = sqlite3.connect(str(p))
            con.execute("create table drugs (trade_name text, inn text)")
            con.executemany("insert into drugs values (?, ?)", [
                ("Дексазон", "Дексаметазон"),
                ("Загадин", "Загадочный комплекс"),
            ])
            con.commit()
            con.close()
            gen = Path(td) / "inn_en.json"
            gen.write_text('{"meta": {}, "entries": {'
                           '"дексаметазон": "dexamethasone", '
                           '"загадочный комплекс": "GENERATED"}}',
                           encoding="utf-8")
            cur = Path(td) / "gloss.yaml"
            cur.write_text("загадочный комплекс: CURATED\n", encoding="utf-8")
            m = load(path=p, aliases_path=Path(td) / "no.yaml",
                     gloss_path=cur, inn_en_path=gen)
            self.assertEqual(m.lookup("дозировка дексазона")[0]["en"],
                             "dexamethasone")
            self.assertEqual(m.lookup("про загадин")[0]["en"], "CURATED")

    def test_load_broken_generated_map_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "drugs.sqlite"
            con = sqlite3.connect(str(p))
            con.execute("create table drugs (trade_name text, inn text)")
            con.execute("insert into drugs values (?, ?)",
                        ("Дексазон", "Дексаметазон"))
            con.commit()
            con.close()
            gen = Path(td) / "inn_en.json"
            gen.write_text("{not json", encoding="utf-8")
            m = load(path=p, aliases_path=Path(td) / "no.yaml",
                     gloss_path=Path(td) / "no-gloss.yaml", inn_en_path=gen)
            self.assertNotIn("en", m.lookup("дозировка дексазона")[0])


if __name__ == "__main__":
    unittest.main()

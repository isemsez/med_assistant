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
    ("Истуриса®", "Осилодростат"),  # key «истурис»: Levenshtein-2 of «историю»
    ("Пертуссин", "Тимьяна ползучего травы экстракт+Калия бромид"),
    # disease-stem brand names (2026-10-15 stoplist work): keys collide
    # with inflected disease words in the dict tier and/or at dist 1
    ("Артрозан", "Мелоксикам"),                     # key «артроза»
    ("Артра", "Глюкозамин+Хондроитина сульфат"),    # key «артр»
    ("Витамин D3", "Колекальциферол"),              # key «витами»
    ("Бронхо-мунал", "Лизаты бактерий"),            # key «бронх»
    ("Диабетон МВ", "Гликлазид"),                   # key «диабето»
    ("ПОДАГРУМ", "Фебуксостат"),                    # key «подагру»
    ("Подагрель", "Фебуксостат"),                   # key «подагрел»
]

ALIASES = {"валерьянка": "Валерианы настойка",  # value = register trade name
           "кагоцел": "меглюмина акридонацетат"}  # value = direct МНН


def dmap(aliases=None):
    return DrugMap(ROWS, aliases=aliases)


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
        m = dmap().lookup("плюс к нурофену аспирин")
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


class TestTypoTier(unittest.TestCase):
    def test_dist1_typo_matches_unambiguous_key(self):
        m = dmap().lookup("дозировка азитромица")  # и→а, one edit
        self.assertEqual(m, [{"trade": "Азитромицин", "inn": "азитромицин",
                              "token": "азитромица"}])

    def test_dist2_typo_no_longer_matches(self):
        # «азитромицна» is Levenshtein-2 of the key «азитромици» — the
        # tier was tightened to dist 1 (owner decision 2026-10-15)
        self.assertEqual(dmap().lookup("дозировка азитромицна"), [])

    def test_percussion_incident_regression(self):
        # «перкуссии» is Levenshtein-2 of the key «пертусси» (Пертуссин) —
        # the 2026-10-15 incident: the P.S. grounded Пертуссин in a
        # cardiology question and the 4B wove its bromide into the answer
        self.assertEqual(
            dmap().lookup("как тона могут меняться при перкуссии"), [])

    def test_short_token_never_fuzzy(self):
        self.assertEqual(dmap().lookup("нурофн дозировка"), [])

    def test_ambiguous_key_never_fuzzy(self):
        # «нурофкн» is within Levenshtein 2 of the ambiguous key «нурофе» —
        # the typo tier only accepts unambiguous keys, so nothing matches
        self.assertEqual(dmap().lookup("нурофкн дозировка"), [])

    def test_common_word_history_not_fuzzy(self):
        # «историю» is Levenshtein-2 of the unambiguous key «истурис»
        # (Истуриса — осилодростат): stoplisted, and dist 2 no longer
        # matches anyway
        self.assertEqual(
            dmap().lookup("в кратце перескажи его историю"), [])

    def test_real_drug_still_matches_despite_history_stopword(self):
        m = dmap().lookup("дозировка истурисы")
        self.assertEqual(m, [{"trade": "Истуриса", "inn": "осилодростат",
                              "token": "истурисы"}])


class TestStemStoplist(unittest.TestCase):
    """Disease-stem collisions (owner decision 2026-10-15): brand names
    built on disease stems collide with inflected disease words in the
    dict tier and at dist 1. Exact stoplist tokens suppress the disease
    forms; drug inflections are distinct tokens and still ground."""

    def test_disease_forms_never_ground(self):
        for q in ("лечение артроза", "что делать при артрозе",
                  "какие витамины при беременности",
                  "обострение бронхита", "жар при бронхите",
                  "диета при сахарном диабете", "проблемы с диабетом",
                  "подагра чем лечить", "что принять при подагре"):
            self.assertEqual(dmap().lookup(q), [], q)

    def test_drug_forms_still_ground(self):
        cases = {
            "дозировка артрозана": "Артрозан",
            "витамин д3 норма": "Витамин D3",
            "бронхо мунал состав": "Бронхо-мунал",
            "лечение диабетоном": "Диабетон МВ",
            "подагрум цена": "ПОДАГРУМ",
            "подагрель дозировка": "Подагрель",
        }
        for q, trade in cases.items():
            m = dmap().lookup(q)
            self.assertEqual([x["trade"] for x in m], [trade], q)


class TestPairColumn(unittest.TestCase):
    """Bilingual МНН strings come ready-made from the inn_pair column
    (scripts/inn_to_en.py); the runtime only reads them."""

    def test_pair_column_renders_per_component(self):
        m = DrugMap([("Сорбифер Дурулес", "Железа сульфат+[Аскорбиновая"
                     " кислота]",
                     "железа сульфат (ferrous sulfate) + аскорбиновая"
                     " кислота (ascorbic acid)")]
                    ).lookup("что есть в сорбифере")
        self.assertEqual(m[0]["en"],
                         "железа сульфат (ferrous sulfate) + аскорбиновая"
                         " кислота (ascorbic acid)")

    def test_mono_pair(self):
        m = DrugMap([("Валидол", "Левоментола раствор в ментилизовалерате",
                      "левоментола раствор в ментилизовалерате (levomenthol"
                      " in menthyl isovalerate)")]).lookup(
            "как действует валидол")
        self.assertEqual(m[0]["en"],
                         "левоментола раствор в ментилизовалерате"
                         " (levomenthol in menthyl isovalerate)")

    def test_no_pair_column_no_en(self):
        m = dmap().lookup("дозировка нурофена")
        self.assertNotIn("en", m[0])

    def test_alias_matches_carry_pair_too(self):
        m = DrugMap([("Валерианы настойка",
                      "Валерианы лекарственной корневищ с корнями настойка",
                      "валерианы лекарственной корневищ с корнями настойка"
                      " (valerian tincture)")],
                    aliases=ALIASES).lookup("валерьянка помогает?")
        self.assertEqual(m[0]["en"],
                         "валерианы лекарственной корневищ с корнями"
                         " настойка (valerian tincture)")

    def test_load_reads_pair_column(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "drugs.sqlite"
            con = sqlite3.connect(str(p))
            con.execute("create table drugs (trade_name text, inn text,"
                        " inn_pair text)")
            con.execute("insert into drugs values (?, ?, ?)",
                        ("Валидол", "Левоментола раствор в ментилизовалерате",
                         "левоментола раствор в ментилизовалерате"
                         " (levomenthol in menthyl isovalerate)"))
            con.commit()
            con.close()
            m = load(path=p, aliases_path=Path(td) / "no-aliases.yaml")
            self.assertEqual(
                m.lookup("как действует валидол")[0]["en"],
                "левоментола раствор в ментилизовалерате (levomenthol"
                " in menthyl isovalerate)")


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
            m = load(path=p, aliases_path=a)
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
            m = load(path=p, aliases_path=Path(td) / "nope.yaml")
            self.assertEqual(m.lookup("кагоцел?"),
                             [{"trade": "Кагоцел",
                               "inn": "меглюмина акридонацетат",
                               "token": "кагоцел"}])


if __name__ == "__main__":
    unittest.main()

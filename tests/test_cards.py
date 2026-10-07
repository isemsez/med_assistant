"""Patient-card unit tests (offline): listing, the validation matrix, the
shipped examples, and the prompt block. Server/CLI integration lives in
test_server.py / test_cli.py (architecture.md §9)."""

import tempfile
import unittest
from pathlib import Path

from medassistant import cards, config


def write_card(d, name, text="текст выписки"):
    p = Path(d) / name
    p.write_text(text, encoding="utf-8")
    return p


class TestListCards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sorted_by_display_name(self):
        write_card(self.dir, "Бб_Бб.выписка.md")
        write_card(self.dir, "Аа_Аа.выписка.md")
        write_card(self.dir, "непрочитанный.md")  # wrong suffix -> skipped
        (self.dir / "sub").mkdir()
        write_card(self.dir / "sub", "Вв_Вв.выписка.md")  # subdirs not scanned
        listed = cards.list_cards(self.dir)
        self.assertEqual([c["file"] for c in listed],
                         ["Аа_Аа.выписка.md", "Бб_Бб.выписка.md"])
        self.assertEqual([c["name"] for c in listed], ["Аа Аа", "Бб Бб"])
        self.assertGreater(listed[0]["size"], 0)
        self.assertIsInstance(listed[0]["mtime"], int)

    def test_missing_dir_empty(self):
        self.assertEqual(cards.list_cards(Path("nowhere")), [])


class TestLoadCard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_card(self.dir, "Тестов_ТТ.выписка.md", "текст выписки")

    def tearDown(self):
        self.tmp.cleanup()

    def test_happy_path(self):
        name, text = cards.load_card("Тестов_ТТ.выписка.md", card_dir=self.dir)
        self.assertEqual(name, "Тестов ТТ")
        self.assertEqual(text, "текст выписки")

    def test_reread_fresh_every_call(self):
        cards.load_card("Тестов_ТТ.выписка.md", card_dir=self.dir)
        write_card(self.dir, "Тестов_ТТ.выписка.md", "новый текст")
        _, text = cards.load_card("Тестов_ТТ.выписка.md", card_dir=self.dir)
        self.assertEqual(text, "новый текст")

    def test_traversal_rejected(self):
        for bad in ("..\\evil.выписка.md", "../evil.выписка.md",
                    "sub/evil.выписка.md", "..", "."):
            with self.assertRaises(cards.CardError, msg=bad):
                cards.load_card(bad, card_dir=self.dir)

    def test_wrong_suffix_rejected(self):
        with self.assertRaises(cards.CardError):
            cards.load_card("Тестов_ТТ.md", card_dir=self.dir)

    def test_missing_file_rejected(self):
        with self.assertRaises(cards.CardError):
            cards.load_card("Нет_такого.выписка.md", card_dir=self.dir)

    def test_oversize_rejected(self):
        with self.assertRaises(cards.CardError):
            cards.load_card("Тестов_ТТ.выписка.md", card_dir=self.dir,
                            max_chars=5)

    def test_non_utf8_rejected(self):
        (self.dir / "Бинарная_ББ.выписка.md").write_bytes(b"\xff\xfe\x00bin")
        with self.assertRaises(cards.CardError):
            cards.load_card("Бинарная_ББ.выписка.md", card_dir=self.dir)

    def test_load_card_path_same_rules(self):
        name, text = cards.load_card_path(self.dir / "Тестов_ТТ.выписка.md")
        self.assertEqual((name, text), ("Тестов ТТ", "текст выписки"))
        with self.assertRaises(cards.CardError):
            cards.load_card_path(self.dir / "нет.md")


SHIPPED = ("Иванов_ИИ.выписка.md", "Петрова_АП.выписка.md")
SECTIONS = ("## Диагнозы", "## Препараты (МНН)", "## Аллергии",
            "## Ключевые показатели", "## Жалобы")


class TestShippedExamples(unittest.TestCase):
    def test_exist_within_limit_and_matched_by_glob(self):
        listed = {c["file"] for c in cards.list_cards(config.CARD_DIR)}
        for f in SHIPPED:
            self.assertIn(f, listed)
            text = (config.CARD_DIR / f).read_text(encoding="utf-8")
            self.assertGreater(len(text.strip()), 0)
            self.assertLessEqual(len(text), config.CARD_MAX_CHARS)
            self.assertLessEqual(len(text), 4000, "demo cards stay compact")
            for section in SECTIONS:
                self.assertIn(section, text)


class TestCardBlock(unittest.TestCase):
    def test_header_names_patient_and_wraps_verbatim(self):
        from medassistant import prompts
        block = prompts.card_block("Иванов ИИ", "строка 1\nстрока 2")
        self.assertTrue(block.startswith(
            "Выписка из амбулаторной карты пациента Иванов ИИ "
            "(приложена врачом; учитывай её при ответе):"))
        self.assertIn("\n\nстрока 1\nстрока 2\n\n", block)
        self.assertTrue(block.endswith("(конец выписки)"))
        self.assertNotIn(".выписка.md", block, "filenames never reach the model")


if __name__ == "__main__":
    unittest.main()

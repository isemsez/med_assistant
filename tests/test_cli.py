"""CLI `ask --card` wiring (offline): mocked pipeline, temp card dir."""

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from medassistant import cards, cli


class FakePipe:
    def __init__(self):
        self.calls = []

    def stream_answer(self, question, history=None, max_tokens=None, card=None,
                      drug=None):
        self.calls.append(("stream", question, history, card, drug))
        yield {"type": "content", "text": "Ответ."}

    def iter_appendix(self, question, raw_answer, session_id=None, manual=False):
        from medassistant.pipeline import AppendixResult
        self.calls.append(("appendix", question, manual))
        yield {"stage": "done", "result": AppendixResult("no_disease")}


class TestAskCard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "Тестов_ТТ.выписка.md").write_text(
            "текст выписки", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _ask(self, *extra):
        pipe = FakePipe()
        with mock.patch.object(cli, "_load_pipeline", return_value=pipe), \
             mock.patch.object(cli.audit_mod, "Audit"), \
             mock.patch.object(cli.drugmap, "load", return_value=None), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cli.main(["ask", "--question", "Вопрос?", *extra])
        return pipe

    def test_filename_in_card_dir(self):
        with mock.patch.object(cards.config, "CARD_DIR", self.dir):
            pipe = self._ask("--card", "Тестов_ТТ.выписка.md")
        self.assertEqual(pipe.calls[0][3], ("Тестов ТТ", "текст выписки"))

    def test_direct_path(self):
        pipe = self._ask("--card", str(self.dir / "Тестов_ТТ.выписка.md"))
        self.assertEqual(pipe.calls[0][3], ("Тестов ТТ", "текст выписки"))

    def test_no_card_by_default(self):
        pipe = self._ask()
        self.assertIsNone(pipe.calls[0][3])

    def test_bad_card_exits_2(self):
        with mock.patch.object(cards.config, "CARD_DIR", self.dir):
            with self.assertRaises(SystemExit):
                self._ask("--card", "ghost.выписка.md")

    def test_pdf_turn_ignores_card(self):
        pipe = FakePipe()
        err = io.StringIO()
        with mock.patch.object(cli, "_load_pipeline", return_value=pipe), \
             mock.patch.object(cli.audit_mod, "Audit"), \
             mock.patch.object(cli.drugmap, "load", return_value=None), \
             redirect_stdout(io.StringIO()), redirect_stderr(err):
            cli.main(["ask", "--question", "pdf ХСН",
                      "--card", "ghost.выписка.md"])
        # no stream call, no validation failure — the card was dropped silently
        self.assertEqual([c[0] for c in pipe.calls], ["appendix"])
        self.assertIn("не используется", err.getvalue())

    def test_drug_map_note_and_forwarding(self):
        from medassistant.drugmap import DrugMap
        pipe = FakePipe()
        err = io.StringIO()
        dmap = DrugMap([("Сорбифер Дурулес", "Железа сульфат+[Аскорбиновая кислота]")])
        with mock.patch.object(cli, "_load_pipeline", return_value=pipe), \
             mock.patch.object(cli.audit_mod, "Audit"), \
             mock.patch.object(cli.drugmap, "load", return_value=dmap), \
             redirect_stdout(io.StringIO()), redirect_stderr(err):
            cli.main(["ask", "--question", "Что содержится в сорбифер дурулес"])
        self.assertEqual(pipe.calls[0][4][0]["trade"], "Сорбифер Дурулес")
        self.assertIn("[МНН] Сорбифер Дурулес: железа сульфат + аскорбиновая кислота",
                      err.getvalue())

    def test_drug_map_absent_note_silent(self):
        pipe = self._ask()
        self.assertEqual(pipe.calls[0][4], None)


if __name__ == "__main__":
    unittest.main()

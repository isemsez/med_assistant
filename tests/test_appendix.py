import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

from medassistant import appendix

DOCS = {
    "1_1": {"name": "Грипп", "url": "https://x/1", "file": "1_1_Грипп.pdf"},
    "2_1": {"name": "ХСН", "url": "https://x/2", "file": "2_1_ХСН.pdf"},
    "3_1": {"name": "Сахарный диабет: ретинопатия", "url": "https://x/3",
            "file": "3_1_Сахарный диабет ретинопатия.pdf"},
    "4_1": {"name": "Старая запись", "url": "https://x/4"},
    "5_1": {"name": "Головная боль напряжения", "url": "https://x/5",
            "file": "5_1_Головная боль напряжения.pdf", "age": "Дети"},
    "6_1": {"name": "Мигрень", "url": "https://x/6",
            "file": "6_1_Мигрень.pdf", "age": "Взрослые"},
}


class TestFileTitle(unittest.TestCase):
    def test_strips_prefix_and_extension(self):
        self.assertEqual(
            appendix.file_title("3_1_Сахарный диабет ретинопатия.pdf", "3_1"),
            "Сахарный диабет ретинопатия")

    def test_stem_without_known_prefix(self):
        self.assertEqual(appendix.file_title("имя.pdf", "7_1"), "имя")


class TestResolvePicks(unittest.TestCase):
    def test_resolves_in_order_without_dir(self):
        picks = appendix.resolve_picks(["2_1", "1_1"], DOCS)
        self.assertEqual([p["code_version"] for p in picks], ["2_1", "1_1"])
        self.assertEqual(picks[0]["name"], "ХСН")
        self.assertEqual(picks[0]["url"],
                         "/data/pdf/" + quote("2_1_ХСН.pdf"))
        self.assertNotIn("path", picks[0])

    def test_name_comes_from_file_not_meta(self):
        picks = appendix.resolve_picks(["3_1"], DOCS)
        self.assertEqual(picks[0]["name"], "Сахарный диабет ретинопатия")

    def test_row_without_file_falls_back_to_meta(self):
        self.assertEqual(appendix.resolve_picks(["4_1"], DOCS),
                         [{"code_version": "4_1", "name": "Старая запись",
                           "url": "https://x/4"}])

    def test_unknown_dropped(self):
        self.assertEqual(appendix.resolve_picks(["nope"], DOCS), [])

    def test_empty(self):
        self.assertEqual(appendix.resolve_picks([], DOCS), [])

    def test_dir_adds_path_and_drops_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "2_1_ХСН.pdf").write_bytes(b"")
            picks = appendix.resolve_picks(["2_1", "1_1"], DOCS, root)
            self.assertEqual([p["code_version"] for p in picks], ["2_1"])
            self.assertEqual(picks[0]["path"], str(root / "2_1_ХСН.pdf"))

    def test_age_rides_on_the_pick(self):
        # owner catch 2026-10-21: same-NAME adult/pediatric register
        # pairs — the audience must travel with the pick so the renderer
        # can mark it; rows without age (old payloads/fakes) stay bare
        kids = appendix.resolve_picks(["5_1"], DOCS)[0]
        self.assertEqual(kids["age"], "Дети")
        self.assertNotIn("age", appendix.resolve_picks(["2_1"], DOCS)[0])


class TestRenderAppendix(unittest.TestCase):
    def test_pediatric_pick_marked(self):
        text = appendix.render_appendix([{
            "disease": "головная боль",
            "picks": [{"code_version": "5_1", "name": "Головная боль "
                      "напряжения", "url": "u", "age": "Дети"}],
        }])
        self.assertIn("«Головная боль напряжения» (5_1, дети): u", text)

    def test_adult_and_mixed_picks_unmarked(self):
        text = appendix.render_appendix([{
            "disease": "мигрень",
            "picks": [{"code_version": "6_1", "name": "Мигрень", "url": "u",
                       "age": "Взрослые"},
                      {"code_version": "7_1", "name": "Астма", "url": "u",
                       "age": "Взрослые, дети"}],
        }])
        self.assertIn("«Мигрень» (6_1): u", text)
        self.assertIn("«Астма» (7_1): u", text)
        self.assertNotIn("дети", text.replace("Астма", ""))

    def test_prefers_local_path_over_url(self):
        text = appendix.render_appendix([{
            "disease": "хроническая сердечная недостаточность",
            "picks": [{"code_version": "2_1", "name": "ХСН",
                       "url": "/data/pdf/2_1_%D0%A5.pdf",
                       "path": r"C:\pdf\2_1_ХСН.pdf"}],
        }])
        self.assertIn("«ХСН» (2_1): C:\\pdf\\2_1_ХСН.pdf", text)
        self.assertNotIn("/data/pdf", text)
        self.assertNotIn("«хроническая", text.replace("Клинические", ""))

    def test_falls_back_to_url_without_path(self):
        text = appendix.render_appendix([{
            "disease": "А",
            "picks": [{"code_version": "2_1", "name": "ХСН",
                       "url": "/data/pdf/2_1_%D0%A5.pdf"}],
        }])
        self.assertIn("«ХСН» (2_1): /data/pdf/2_1_%D0%A5.pdf", text)

    def test_multi_disease_headers(self):
        text = appendix.render_appendix([
            {"disease": "А", "picks": [{"code_version": "1_1", "name": "Грипп",
                                        "url": "u1"}]},
            {"disease": "Б", "picks": [{"code_version": "2_1", "name": "ХСН",
                                        "url": "u2"}]},
        ])
        self.assertIn("«А»:", text)
        self.assertIn("«Б»:", text)

    def test_none_when_no_picks(self):
        self.assertIsNone(appendix.render_appendix([{"disease": "А", "picks": []}]))
        self.assertIsNone(appendix.render_appendix([]))


class TestFilteredOut(unittest.TestCase):
    @staticmethod
    def _c(cv):
        return {"code_version": cv, "name": cv, "url": f"/data/pdf/{cv}.pdf"}

    def test_picked_anywhere_is_verified(self):
        by_disease = [{"disease": "А", "picks": [self._c("2_1")]}]
        cands = [("А", [self._c("2_1"), self._c("1_1")]),
                 ("Б", [self._c("2_1"), self._c("3_1")])]
        out = appendix.filtered_out(by_disease, cands)
        self.assertEqual([(c["code_version"], c["disease"]) for c in out],
                         [("1_1", "А"), ("3_1", "Б")])

    def test_dedup_across_diseases(self):
        out = appendix.filtered_out(
            [], [("А", [self._c("1_1")]), ("Б", [self._c("1_1")])])
        self.assertEqual([(c["code_version"], c["disease"]) for c in out],
                         [("1_1", "А")])

    def test_empty(self):
        self.assertEqual(appendix.filtered_out([], []), [])


class TestNotes(unittest.TestCase):
    def test_note_constants(self):
        for note in (appendix.NOTE_EXTRACT_ERROR, appendix.NOTE_NO_DISEASE,
                     appendix.NOTE_SEARCH_ERROR, appendix.NOTE_SELECT_ERROR):
            self.assertTrue(note.strip().endswith((".", "!")) or note.strip())


if __name__ == "__main__":
    unittest.main()

"""Pair-verdict select cache (owner decision 2026-10-26): per
(disease, audience bucket) -> guideline id verdicts, frozen in
data/select_cache.yaml — right attaches, wrong shows grayed; a machine
write never overwrites a human revision; 👎 toggles, 👍 confirms once.
"""

import tempfile
import unittest
from pathlib import Path

from medassistant.select_cache import SelectCache


class TestSelectCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "select_cache.yaml"
        self.sha = "stamp-1"

    def test_roundtrip_and_key_normalization(self):
        c = SelectCache(self.path, self.sha)
        c.put_machine("Инфаркт Миокарда", "none",
                      {"157_5": "right", "155_2": "wrong"})
        # the file stays human-readable (allow_unicode — like drug_aliases)
        self.assertIn("инфаркт миокарда",
                      self.path.read_text(encoding="utf-8"))
        c2 = SelectCache(self.path, self.sha)
        self.assertEqual(c2.get("инфаркт миокарда", "none", "157_5"),
                         {"verdict": "right"})
        self.assertEqual(c2.get("ИНФАРКТ МИОКАРДА", "none", "155_2"),
                         {"verdict": "wrong"})
        self.assertIsNone(c2.get("инфаркт миокарда", "child", "157_5"),
                          "audience bucket is part of the key")
        self.assertIsNone(c2.get("инфаркт миокарда", "none", "ghost_1"))

    def test_stamp_mismatch_resets(self):
        c = SelectCache(self.path, self.sha)
        c.put_machine("ХСН", "none", {"2_1": "right"})
        self.assertFalse(SelectCache(self.path, self.sha).was_reset())
        fresh = SelectCache(self.path, "stamp-2")
        self.assertTrue(fresh.was_reset())
        self.assertFalse(fresh.was_reset(), "the flag is one-shot")
        self.assertIsNone(fresh.get("ХСН", "none", "2_1"))
        # a write under the new stamp re-creates the file with it
        fresh.put_machine("ХСН", "none", {"2_1": "wrong"})
        self.assertFalse(SelectCache(self.path, "stamp-2").was_reset())

    def test_machine_never_overwrites_human(self):
        c = SelectCache(self.path, self.sha)
        c.negate("ХСН", "none", "2_1", "right")  # human: wrong
        c.put_machine("ХСН", "none", {"2_1": "right"})  # a later turn re-judged
        self.assertEqual(c.get("ХСН", "none", "2_1"),
                         {"verdict": "wrong", "human": True})

    def test_negate_toggles_and_handles_legacy_pairs(self):
        c = SelectCache(self.path, self.sha)
        # a pair never cached (legacy appendix row): negate works off `current`
        self.assertEqual(c.negate("ХСН", "none", "2_1", "right"),
                         {"verdict": "wrong", "human": True})
        # the second negate flips the STORED verdict, not the payload zone
        self.assertEqual(c.negate("ХСН", "none", "2_1", "ignored"),
                         {"verdict": "right", "human": True})

    def test_confirm_marks_human(self):
        c = SelectCache(self.path, self.sha)
        c.put_machine("ХСН", "none", {"2_1": "right"})
        self.assertEqual(c.confirm("ХСН", "none", "2_1", "right"),
                         {"verdict": "right", "human": True})
        self.assertEqual(c.get("ХСН", "none", "2_1"),
                         {"verdict": "right", "human": True})

    def test_forget_removes_one_pair_only(self):
        # ↻ (owner decision 2026-10-26 evening): ONE pair becomes unknown —
        # the next turn's selector re-judges exactly it — while sibling
        # pairs, other audience buckets and other diseases stay intact
        c = SelectCache(self.path, self.sha)
        c.put_machine("ХСН", "none", {"2_1": "right", "1_1": "wrong"})
        c.negate("ХСН", "none", "3_1", "right")  # human — goes the same way
        c.put_machine("ХСН", "child", {"2_1": "right"})
        c.put_machine("Грипп", "none", {"9_1": "right"})
        self.assertTrue(c.forget("ХСН", "none", "2_1"))
        self.assertIsNone(c.get("ХСН", "none", "2_1"))
        self.assertEqual(c.get("ХСН", "none", "1_1"), {"verdict": "wrong"})
        self.assertEqual(c.get("ХСН", "none", "3_1"),
                         {"verdict": "wrong", "human": True})
        self.assertEqual(c.get("ХСН", "child", "2_1"), {"verdict": "right"})
        self.assertEqual(c.get("Грипп", "none", "9_1"), {"verdict": "right"})
        # persisted: a fresh load sees the same state
        self.assertIsNone(SelectCache(self.path, self.sha)
                          .get("ХСН", "none", "2_1"))
        self.assertFalse(c.forget("ХСН", "none", "2_1"),
                         "forgetting an unknown pair is a no-op")

    def test_forget_last_pair_prunes_empty_buckets(self):
        c = SelectCache(self.path, self.sha)
        c.put_machine("ХСН", "none", {"2_1": "right"})
        self.assertTrue(c.forget("ХСН", "none", "2_1"))
        self.assertEqual(c.all_verdicts(), {},
                         "no empty disease/bucket skeletons linger")

    def test_all_verdicts_is_a_copy(self):
        c = SelectCache(self.path, self.sha)
        c.put_machine("ХСН", "none", {"2_1": "right"})
        dump = c.all_verdicts()
        dump["хсн"]["none"]["2_1"]["verdict"] = "wrong"
        self.assertEqual(c.get("ХСН", "none", "2_1"), {"verdict": "right"})

    def test_unreadable_file_is_empty(self):
        self.path.write_text("!!!: [not, yaml", encoding="utf-8")
        c = SelectCache(self.path, self.sha)
        self.assertIsNone(c.get("ХСН", "none", "2_1"))
        c.put_machine("ХСН", "none", {"2_1": "right"})  # re-creates the file
        self.assertEqual(SelectCache(self.path, self.sha)
                         .get("ХСН", "none", "2_1"), {"verdict": "right"})


if __name__ == "__main__":
    unittest.main()

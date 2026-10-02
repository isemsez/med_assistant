import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from medassistant import gindex


def e(dim, *vals):
    v = np.zeros(dim, dtype=np.float32)
    for i, x in enumerate(vals):
        v[i] = x
    return v


class TestDocTexts(unittest.TestCase):
    def test_name_and_variants(self):
        self.assertEqual(gindex.doc_texts({"name": "Рак носоглотки"}),
                         ["Рак носоглотки"])
        self.assertEqual(
            gindex.doc_texts({"name": "А", "mkb_names": "Б; В ;; А"}),
            ["А", "Б", "В"])
        self.assertEqual(gindex.doc_texts({"name": "", "mkb_names": ""}), [])


class TestSearchVectors(unittest.TestCase):
    def setUp(self):
        dim = 4
        rows = [
            {"code_version": "1_1", "name": "Грипп", "text": "Грипп"},
            {"code_version": "1_1", "name": "Грипп", "text": " Grip"},
            {"code_version": "2_1", "name": "ХСН", "text": "ХСН"},
            {"code_version": "3_1", "name": "Рак", "text": "Рак"},
        ]
        mat = np.stack([
            e(dim, 1, 0), e(dim, 0.9, 0.1), e(dim, 0, 1), e(dim, 0, 0, 1),
        ])
        self.idx = gindex.GuidelineIndex(mat, rows)

    def test_top1_exact(self):
        cards = self.idx.search_vectors([e(4, 1, 0)], top_k=5)
        self.assertEqual(cards[0]["code_version"], "1_1")
        self.assertGreater(cards[0]["score"], 0.99)

    def test_best_score_per_code_version(self):
        cards = self.idx.search_vectors([e(4, 1, 0)], top_k=5)
        cvs = [c["code_version"] for c in cards]
        self.assertEqual(len(cvs), len(set(cvs)), "one card per guideline")

    def test_top_k_and_order(self):
        cards = self.idx.search_vectors([e(4, 1, 0), e(4, 0, 1)], top_k=2)
        self.assertEqual([c["code_version"] for c in cards], ["1_1", "2_1"])
        self.assertGreaterEqual(cards[0]["score"], cards[1]["score"])

    def test_cards_have_no_url(self):
        for c in self.idx.search_vectors([e(4, 1, 0)]):
            self.assertNotIn("url", c)

    def test_empty(self):
        self.assertEqual(self.idx.search_vectors([], top_k=5), [])


class TestBuildIndex(unittest.TestCase):
    def test_build_and_roundtrip(self):
        docs = {
            "1_1": {"name": "Грипп", "mkb_names": "Grip", "url": "u1"},
            "2_1": {"name": "ХСН", "mkb_names": "", "url": "u2"},
        }
        calls = []

        def fake_embed(endpoint, texts, batch=64, **kw):
            calls.append(list(texts))
            return [list(e(4, 1, 0)) if "Grip" in t or "Грипп" in t
                    else list(e(4, 0, 1)) for t in texts]

        orig = gindex.llm.embed
        gindex.llm.embed = fake_embed
        try:
            with tempfile.TemporaryDirectory() as tmp:
                n_docs, n_rows = gindex.build_index(docs, {}, tmp)
                self.assertEqual((n_docs, n_rows), (2, 3))
                idx = gindex.GuidelineIndex.load(tmp)
                self.assertEqual(len(idx), 3)
                cards = idx.search_vectors([e(4, 1, 0)], top_k=1)
                self.assertEqual(cards[0]["code_version"], "1_1")
                meta = json.loads((Path(tmp) / "meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["dim"], 4)
        finally:
            gindex.llm.embed = orig
        self.assertEqual(calls[0], ["Грипп", "Grip", "ХСН"])


if __name__ == "__main__":
    unittest.main()

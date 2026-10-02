"""Guideline embedding index (architecture.md §3[3]).

Offline build: every active guideline's `name` plus every `mkb_names` variant
(semicolon-separated) is embedded as its own vector — all 757 guidelines, no
age/specialty filter. Search: embed the query variants (original name +
abbreviation expansions), cosine top-K cards [{code_version, name, score}].
Cards never contain URLs — the model picks ids, Python renders URLs.
"""

import json
from pathlib import Path

import numpy as np

from . import llm


def doc_texts(meta):
    """Embeddable texts for one guideline: name + mkb_names variants."""
    name = (meta.get("name") or "").strip()
    texts = [name] if name else []
    for part in (meta.get("mkb_names") or "").split(";"):
        p = part.strip()
        if p and p not in texts:
            texts.append(p)
    return texts


def build_index(docs, embed_endpoint, out_dir, batch=64):
    """docs: {code_version: meta} -> writes vectors.npy + meta.json in out_dir.

    Returns (n_docs, n_rows). Vectors are L2-normalized float32.
    """
    rows, texts = [], []
    for cv, meta in docs.items():
        for t in doc_texts(meta):
            rows.append({"code_version": cv, "name": (meta.get("name") or "").strip(),
                         "text": t})
            texts.append(t)
    vectors = llm.embed(embed_endpoint, texts, batch=batch)
    mat = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", mat)
    (out / "meta.json").write_text(
        json.dumps({"dim": int(mat.shape[1]), "rows": rows}, ensure_ascii=False),
        encoding="utf-8")
    return len(docs), len(rows)


class GuidelineIndex:
    def __init__(self, matrix, rows):
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.rows = rows
        self._by_cv = {}
        for i, r in enumerate(rows):
            self._by_cv.setdefault(r["code_version"], []).append(i)

    def __len__(self):
        return len(self.rows)

    @classmethod
    def load(cls, dir_path):
        d = Path(dir_path)
        matrix = np.load(d / "vectors.npy")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return cls(matrix, meta["rows"])

    def search_vectors(self, query_vecs, top_k=5):
        """Normalized query vectors -> top-K cards, best score per code_version."""
        if not len(self.rows) or not query_vecs:
            return []
        q = np.asarray(query_vecs, dtype=np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q = q / norms
        sims = self.matrix @ q.T  # (n_rows, n_queries)
        best_per_row = sims.max(axis=1)
        best = {}
        for i, r in enumerate(self.rows):
            cv = r["code_version"]
            s = float(best_per_row[i])
            if s > best.get(cv, (-1.0, None))[0]:
                best[cv] = (s, r)
        ranked = sorted(best.values(), key=lambda t: (-t[0], t[1]["code_version"]))
        return [{"code_version": r["code_version"], "name": r["name"],
                 "score": round(s, 4)} for s, r in ranked[:top_k]]

    def search(self, query_variants, embed_endpoint, top_k=5, batch=64):
        """Embed query variants (list[str]) -> top-K cards."""
        vecs = llm.embed(embed_endpoint, list(query_variants), batch=batch)
        return self.search_vectors(vecs, top_k=top_k)


def load_index(dir_path=None):
    from . import config
    return GuidelineIndex.load(dir_path or config.INDEX_DIR)

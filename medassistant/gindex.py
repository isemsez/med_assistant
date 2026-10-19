"""Guideline embedding index (architecture.md §3[3]).

Offline build: every active guideline's `name` plus every `mkb_names` variant
(semicolon-separated) is embedded as its own vector — all 757 guidelines, no
age/specialty filter. Search: embed the query variants (original name +
abbreviation expansions), cosine top-K cards [{code_version, name, score}].
Cards never contain URLs — the model picks ids, Python renders URLs.
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from . import llm


def norm_text(s):
    """NFC-normalize: the register mixes composed and decomposed Cyrillic
    (й as U+0439 vs и+U+0306), and the same name/MKБ string under two
    guidelines must embed to ONE vector — an NFD/NFC pair scored apart by
    ~1e-3 was enough to flip a golden top-1 (2026-10-18). Queries are
    normalized through the same helper in GuidelineIndex.search."""
    return unicodedata.normalize("NFC", s)

# trailing parenthetical that is pure ICD-code annotation, e.g.
# «Диабетическая полинейропатия (E10-E14+ с общим четвертым знаком .4)» ->
# «Диабетическая полиневропатия». The code noise dragged the exact-match
# variant's similarity for «диабетическая полинейропатия» from 0.78 down to
# 0.61 and kept the right guideline out of the top-5 (2026-10-16).
_ICD_PARENTH = re.compile(r"\s*\([^()]*[A-Z]\d{2}[^()]*\)\s*$")


def docs_hash(docs):
    """Canonical content hash of a docs_meta dict (build/health must agree).

    Order-independent (sort_keys) so a reordered register is not 'stale'.
    """
    blob = json.dumps(docs, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def doc_texts(meta):
    """Embeddable texts for one guideline: name + mkb_names variants.

    МКБ variants are stripped of trailing ICD-code parentheticals (pure
    annotation noise for the embedder) and deduped after stripping.
    """
    name = norm_text((meta.get("name") or "").strip())
    texts = [name] if name else []
    for part in (meta.get("mkb_names") or "").split(";"):
        p = _ICD_PARENTH.sub("", part.strip()).strip()
        p = norm_text(p)
        if p and p not in texts:
            texts.append(p)
    return texts


def build_index(docs, embed_endpoint, out_dir, batch=64):
    """docs: {code_version: meta} -> writes vectors.npy + meta.json in out_dir.

    Returns (n_docs, n_rows). Vectors are L2-normalized float32.

    Identical texts (the same name/MKБ variant can belong to several
    guidelines, e.g. «Фибрилляция и трепетание предсердий» under both the
    arrhythmia and the SVT recommendation) are embedded ONCE and the vector
    is shared by every row with that text: batched embedding is not
    bit-stable across batch positions, and a 3e-4 float jitter on a
    duplicated string was enough to flip a golden top-1 (2026-10-18,
    llama.cpp port). Exact ties then fall to the deterministic
    code_version tiebreak in search_vectors.
    """
    rows, texts = [], []
    for cv, meta in docs.items():
        for t in doc_texts(meta):
            rows.append({"code_version": cv, "name": (meta.get("name") or "").strip(),
                         "text": t})
            texts.append(t)
    uniq = list(dict.fromkeys(texts))
    vecs = llm.embed(embed_endpoint, uniq, batch=batch)
    by_text = dict(zip(uniq, vecs))
    mat = np.asarray([by_text[t] for t in texts], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", mat)
    # freshness stamp: health compares it against the current docs_meta
    # content — a refreshed register without build-index shows up as stale
    (out / "meta.json").write_text(
        json.dumps({"dim": int(mat.shape[1]), "rows": rows,
                    "docs_meta_sha1": docs_hash(docs),
                    "docs_meta_count": len(docs)}, ensure_ascii=False),
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
        """Normalized query vectors -> top-K cards, best score per code_version.

        Each card carries `matched` — the indexed text (name or МКБ variant)
        that produced the score. The labeler needs it to judge WHY a candidate
        surfaced: «Преждевременное половое развитие» reaching the top-5 via its
        МКБ variant «Полигландулярная гиперфункция» is obviously droppable only
        when the matched text is visible (2026-10-16, полинейропатия turn).
        `matched` never reaches the UI — the select event projects
        code_version/name only (pipeline.py).
        """
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
                 "score": round(s, 4), "matched": r["text"]}
                for s, r in ranked[:top_k]]

    def search(self, query_variants, embed_endpoint, top_k=5, batch=64):
        """Embed query variants (list[str]) -> top-K cards."""
        vecs = llm.embed(embed_endpoint,
                         [norm_text(v) for v in query_variants], batch=batch)
        return self.search_vectors(vecs, top_k=top_k)


def load_index(dir_path=None):
    from . import config
    return GuidelineIndex.load(dir_path or config.INDEX_DIR)

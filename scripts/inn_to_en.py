"""Build data/inn_en.json: register МНН -> English INN (WHO list).

Source of truth: data/inn_list_en-lat.txt — the owner-prepared pairing
"english<TAB>latin" of the WHO recommended INN list (through list 95,
mirrored by FIMEA). Russian МНН are Cyrillic renderings of the LATIN
forms, so conversion is mechanical: transliterate back to Latin,
canonicalize both sides (Russian drops h, writes кс/ц/и where Latin has
x/c/y; salt and acid grammar forms differ), and match token sequences.

Trust policy (owner decision 2026-10-14): only EXACT sequence matches
(after canonicalization) enter the file — the distance-1 tier is
reported but never written (a missing gloss costs nothing; a wrong gloss
is the hallucination we are fighting). Composites («x + y») convert
all-or-nothing. Entries from data/inn_gloss.yaml (true WHO holes like
ментилизовалерат) override the generated map at load time.

Rebuild after dropping in a refreshed WHO list:
    python scripts/inn_to_en.py            # writes data/inn_en.json
    python scripts/inn_to_en.py --report   # stats + fuzzy-tier preview
"""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medassistant import config  # noqa: E402
from medassistant.drugmap import PLACEHOLDER_INNS, clean_inn  # noqa: E402

RULES_VERSION = "1"

TRANS = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "iu", "я": "ia",
}

# salt/acid grammar forms -> one canonical token on both sides
SALT = {
    "natricum": "natri", "natrii": "natri", "натрия": "natri",
    "натри": "natri", "sodicum": "natri",
    "kalicum": "kali", "kalii": "kali", "калия": "kali", "кали": "kali",
    "calcicum": "calci", "calcii": "calci", "кальция": "calci",
    "magnicum": "magni", "magnii": "magni", "магния": "magni",
    "hydrochloridum": "hcl", "hydrochlorid": "hcl", "гидрохлорид": "hcl",
    "acidum": "acid", "acid": "acid", "кислота": "acid", "кислоты": "acid",
}

ACID_WORDS = {"acid"}
ACID_ADJ_TRANSFORM = ("ovaia", "ic"), ("evaia", "ic")


def parse_pairs(text):
    """'english<TAB>latin' lines -> [(en, latin)] (lowercased)."""
    out = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            out.append((parts[0].strip().lower(), parts[1].strip().lower()))
    return out


def _tokens(s):
    return [t for t in re.split(r"[^a-zа-яё0-9]+", s.lower()) if t]


def _strip_latin_endings(tok):
    """Per-token -um/-us/-a stripping (levothyroxinum natricum ->
    levotiroxin natric): Russian carries no Latin endings."""
    for suf in ("um", "us", "a"):
        if tok.endswith(suf) and len(tok) > 4:
            return tok[: -len(suf)]
    return tok


def _canon(joined):
    s = joined.replace("ph", "f").replace("th", "t").replace("ch", "kh")
    s = s.replace("ks", "x").replace("ts", "c").replace("h", "g")
    s = re.sub(r"(.)\1+", r"\1", s)      # collapse doubled letters
    s = s.replace("y", "i").replace("z", "s").replace("w", "v")
    s = s.replace("c", "k")
    return re.sub(r"[^a-z0-9]", "", s)


def canon_lat(name):
    toks = [SALT.get(t, _strip_latin_endings(t)) for t in _tokens(name)]
    return _canon("".join(toks))


def canon_ru(name):
    toks = []
    for t in _tokens(name):
        t = SALT.get(t) or "".join(TRANS.get(c, c) for c in t)
        toks.append(SALT.get(t, t))
    return _canon("".join(toks))


def canon_ru_variants(name):
    """Canon sequences with/without RU trailing а/я per token (feminine
    nouns keep the ending the Latin lacks: алтеплаза↔alteplas(um),
    азилсартана↔azilsartanum) and with ю→u (меглюмин↔meglumin). Exact
    tier — each variant still needs a unique index hit."""
    toks = []
    for t in _tokens(name):
        t = SALT.get(t) or "".join(TRANS.get(c, c) for c in t)
        t = SALT.get(t, t)
        variants = {t}
        if len(t) > 5:
            if t.endswith(("a", "ia")):
                variants.add(t[:-1])
            if "iu" in t:
                variants.add(t.replace("iu", "u"))
        toks.append(variants)
    seqs = [""]
    for vs in toks:
        seqs = [s + v for s in seqs for v in vs]
    return {_canon(s) for s in seqs}


class InnConverter:
    """WHO paired list -> RU МНН to EN INN converter (exact tier)."""

    def __init__(self, pairs):
        self.pairs = pairs
        # canonical token sequence -> {en}
        self.index = defaultdict(set)
        # acids: adjective stem candidates for the RU «<adj> кислота» form
        self.acids = []  # (latin_adj_canon, en)
        for en, latin in pairs:
            la, en_toks = _tokens(latin), _tokens(en)
            self.index[canon_lat(latin)].add(en)
            self.index[_canon("".join(
                SALT.get(t, t) for t in en_toks))].add(en)
            if "acidum" in la or "acid" in en_toks:
                adj = [t for t in la if t not in ("acidum",)]
                if len(la) == 2 and len(adj) == 1:
                    stem = canon_lat(adj[0])
                    if len(stem) >= 4:
                        self.acids.append((stem, en))
        self.acid_stems = sorted(self.acids, key=lambda x: -len(x[0]))

    # -- one RU component (no «+»), returns en or None -------------------

    def _exact(self, canon_seq):
        hit = self.index.get(canon_seq)
        if hit and len(hit) == 1:
            return next(iter(hit))
        return None

    def _match_acid(self, ru):
        """«<adj> кислота» ↔ «acidum <adj>um»: RU and Latin adjective stems
        differ by calquing (аскорбиновая vs ascorbic), so match by longest
        common prefix: LCP >= 5 and LCP >= 70% of the shorter stem, unique
        winner only."""
        toks = _tokens(ru)
        if len(toks) != 2 or toks[-1] not in ("кислота", "кислоты"):
            return None
        stem = toks[0]
        for suf in ("ая", "ой", "ые", "яя"):
            if stem.endswith(suf) and len(stem) > len(suf) + 3:
                stem = stem[: -len(suf)]
                break
        c = canon_ru(stem)
        best = (0, [])  # (lcp, [en])
        for cand, en in self.acid_stems:
            lcp = 0
            for a, b in zip(c, cand):
                if a != b:
                    break
                lcp += 1
            if lcp >= 5 and lcp >= 0.7 * min(len(c), len(cand)):
                if lcp > best[0]:
                    best = (lcp, [en])
                elif lcp == best[0] and en not in best[1]:
                    best[1].append(en)
        if best[0] and len(best[1]) == 1:
            return best[1][0]
        return None

    def convert_component(self, ru):
        """One «+»-free МНН component -> English INN or None. Full-name
        match only — no partial-word fallback (a partial gloss of a
        descriptive phrase is a wrong gloss)."""
        ru = ru.strip()
        if not ru:
            return None
        for seq in sorted(canon_ru_variants(ru), key=len):
            hit = self._exact(seq)
            if hit:
                return hit
        return self._match_acid(ru)

    def convert(self, ru_inn):
        """Full register МНН (may be «x + y») -> English or None
        (all-or-nothing: no half-glossed composites)."""
        parts = [p.strip() for p in ru_inn.split("+") if p.strip()]
        if not parts:
            return None
        ens = [self.convert_component(p) for p in parts]
        return " + ".join(ens) if all(ens) else None

    # -- diagnostics ------------------------------------------------------

    def near_misses(self, ru_inn):
        """Distance-1 candidates for the report (never written to the
        product file)."""
        parts = [p.strip() for p in ru_inn.split("+") if p.strip()]
        out = []
        for p in parts:
            c = canon_ru(p)
            for k in self.index:
                if abs(len(k) - len(c)) == 1 and _lev1(c, k):
                    out.append((p, sorted(self.index[k])))
        return out


def _lev1(a, b):
    if abs(len(a) - len(b)) != 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


def register_inns(db_path):
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("select inn from drugs").fetchall()
    finally:
        con.close()
    inns = {clean_inn(r[0]) for r in rows}
    return sorted(i for i in inns if i and i not in PLACEHOLDER_INNS)


def build(pairs, inns):
    """(pairs, register МНН list) -> (entries, stats)."""
    conv = InnConverter(pairs)
    entries, fuzzy = {}, {}
    for inn in inns:
        en = conv.convert(inn)
        if en:
            entries[inn] = en
        else:
            near = conv.near_misses(inn)
            if near:
                fuzzy[inn] = near
    stats = {
        "register_inns": len(inns),
        "converted": len(entries),
        "fuzzy_only": len(fuzzy),
        "unmatched": len(inns) - len(entries) - len(fuzzy),
        "rules_version": RULES_VERSION,
    }
    return entries, stats, fuzzy


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default=str(config.PROJECT_ROOT
                                           / "data" / "inn_list_en-lat.txt"))
    ap.add_argument("--drugs", default=str(config.DRUGS_DB))
    ap.add_argument("--out", default=str(config.PROJECT_ROOT
                                         / "data" / "inn_en.json"))
    ap.add_argument("--report", action="store_true",
                    help="print stats and fuzzy-tier preview, write nothing")
    args = ap.parse_args(argv)

    pairs = parse_pairs(Path(args.pairs).read_text(encoding="utf-8"))
    inns = register_inns(args.drugs)
    entries, stats, fuzzy = build(pairs, inns)

    src = Path(args.pairs)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": src.name,
        "source_sha1": hashlib.sha1(src.read_bytes()).hexdigest(),
        "pairs": len(pairs),
        **stats,
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if args.report:
        print("fuzzy-tier preview (NOT written):")
        for inn, near in list(fuzzy.items())[:15]:
            print(f"  {inn!r} ~ {near[:2]}")
        return

    Path(args.out).write_text(
        json.dumps({"meta": meta, "entries": entries}, ensure_ascii=False,
                   indent=1, sort_keys=True), encoding="utf-8")
    print(f"-> {args.out} ({len(entries)} entries)")
    spot = ["ибупрофен", "дексаметазон", "левотироксин натрия",
            "аскорбиновая кислота"]
    for s in spot:
        print(f"  spot {s!r}: {entries.get(s)}")


if __name__ == "__main__":
    main()

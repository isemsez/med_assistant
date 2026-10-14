"""Translate register МНН to English INNs and store them IN drugs.sqlite.

Source of truth: data/inn_list_en-lat.txt — the owner-prepared pairing
"english<TAB>latin" of the WHO recommended INN list (through list 95,
mirrored by FIMEA). Russian МНН are Cyrillic renderings of the LATIN
forms, so conversion is mechanical: transliterate back to Latin,
canonicalize both sides (Russian drops h, writes кс/ц/и where Latin has
x/c/y; salt and acid grammar forms differ), and match token sequences.

Output: columns `inn_en` / `inn_lat` in the drugs table (per row, same
value for all rows sharing an МНН). Compound МНН («x + y») translate
PER COMPONENT: each part is translated independently, parts that fail
stay untranslated (empty slot in the composite), a row where nothing
translated gets NULL — the runtime renderer puts the English in
parentheses after each substance name it has one for.

Trust policy (owner decision 2026-10-14): only EXACT rule matches are
written; the distance-1 tier is reported (--report) but never applied
(a missing gloss costs nothing; a wrong gloss is the hallucination we
are fighting). True WHO holes (ментилизовалерат, железа сульфат) belong
to the curated data/inn_gloss.yaml, applied per component at runtime.

    python scripts/inn_to_en.py             # alter + update drugs.sqlite
    python scripts/inn_to_en.py --report    # stats only, no writes
"""

import argparse
import hashlib
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medassistant import config  # noqa: E402
from medassistant.drugmap import PLACEHOLDER_INNS, clean_inn  # noqa: E402

RULES_VERSION = "2"

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
    """WHO paired list -> (en, latin) for RU МНН components (exact tier)."""

    def __init__(self, pairs):
        self.pairs = pairs
        # canonical token sequence -> {en: latin}
        self.index = defaultdict(dict)
        # acids: adjective stem candidates for the RU «<adj> кислота» form
        self.acids = []  # (latin_adj_canon, en, latin)
        for en, latin in pairs:
            la_toks, en_toks = _tokens(latin), _tokens(en)
            self.index[canon_lat(latin)][en] = latin
            self.index[_canon("".join(
                SALT.get(t, t) for t in en_toks))][en] = latin
            if "acidum" in la_toks or "acid" in en_toks:
                adj = [t for t in la_toks if t != "acidum"]
                if len(la_toks) == 2 and len(adj) == 1:
                    stem = canon_lat(adj[0])
                    if len(stem) >= 4:
                        self.acids.append((stem, en, latin))
        self.acid_stems = sorted(self.acids, key=lambda x: -len(x[0]))

    def _exact(self, canon_seq):
        hit = self.index.get(canon_seq)
        if hit and len(hit) == 1:
            en, latin = next(iter(hit.items()))
            return en, latin
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
        best = (0, [])  # (lcp, [(en, latin)])
        for cand, en, latin in self.acid_stems:
            lcp = 0
            for a, b in zip(c, cand):
                if a != b:
                    break
                lcp += 1
            if lcp >= 5 and lcp >= 0.7 * min(len(c), len(cand)):
                if lcp > best[0]:
                    best = (lcp, [(en, latin)])
                elif lcp == best[0] and (en, latin) not in best[1]:
                    best[1].append((en, latin))
        if best[0] and len(best[1]) == 1:
            return best[1][0]
        return None

    def convert_component(self, ru):
        """One «+»-free МНН component -> (en, latin) or None. Full-name
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
        """Full register МНН (may be «x + y») -> (en_composite,
        lat_composite) or (None, None). Per component; untranslated parts
        leave an empty slot; NULL when nothing translated."""
        parts = [p.strip() for p in ru_inn.split("+") if p.strip()]
        if not parts:
            return None, None
        hits = [self.convert_component(p) for p in parts]
        ens = [h[0] if h else "" for h in hits]
        lats = [h[1] if h else "" for h in hits]
        en = " + ".join(ens) if any(ens) else None
        lat = " + ".join(lats) if any(lats) else None
        return en, lat


def canon_lat(name):
    toks = [SALT.get(t, _strip_latin_endings(t)) for t in _tokens(name)]
    return _canon("".join(toks))


def register_inns(db_path):
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("select inn from drugs").fetchall()
    finally:
        con.close()
    inns = {clean_inn(r[0]) for r in rows}
    return sorted(i for i in inns if i and i not in PLACEHOLDER_INNS)


def build(pairs, inns):
    """(pairs, register МНН list) -> {inn: (en, lat)}, stats."""
    conv = InnConverter(pairs)
    out = {}
    partial = full = 0
    for inn in inns:
        en, lat = conv.convert(inn)
        if en:
            out[inn] = (en, lat)
            if "" in en.split(" + "):
                partial += 1
            else:
                full += 1
    stats = {
        "register_inns": len(inns),
        "fully_translated": full,
        "partially_translated": partial,
        "untranslated": len(inns) - full - partial,
        "rules_version": RULES_VERSION,
    }
    return out, stats


def apply_db(pairs, db_path):
    """Add inn_en/inn_lat to the drugs table (idempotent) and fill them
    per register МНН. Returns (mapping keyed by cleaned МНН, stats)."""
    conv = InnConverter(pairs)
    con = sqlite3.connect(str(db_path))
    updates = []
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(drugs)")}
        for col in ("inn_en", "inn_lat"):
            if col not in cols:
                con.execute(f"ALTER TABLE drugs ADD COLUMN {col} TEXT")
        # UPDATE matches the RAW register inn; conversion uses cleaned
        for (raw,) in con.execute("select distinct inn from drugs"):
            inn = clean_inn(raw)
            if not inn or inn in PLACEHOLDER_INNS:
                continue
            en, lat = conv.convert(inn)
            if en:
                updates.append((en, lat, raw))
        con.executemany(
            "UPDATE drugs SET inn_en=?, inn_lat=? WHERE inn=?", updates)
        con.commit()
    finally:
        con.close()
    mapping = {clean_inn(raw): (en, lat) for en, lat, raw in updates}
    full = sum(1 for en, _ in mapping.values()
               if "" not in en.split(" + "))
    partial = len(mapping) - full
    stats = {
        "register_inns": len(register_inns(db_path)),
        "fully_translated": full,
        "partially_translated": partial,
        "untranslated": len(register_inns(db_path)) - len(mapping),
        "rules_version": RULES_VERSION,
    }
    return mapping, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default=str(config.PROJECT_ROOT
                                           / "data" / "inn_list_en-lat.txt"))
    ap.add_argument("--drugs", default=str(config.DRUGS_DB))
    ap.add_argument("--report", action="store_true",
                    help="print stats only, write nothing")
    args = ap.parse_args(argv)

    pairs = parse_pairs(Path(args.pairs).read_text(encoding="utf-8"))
    src = Path(args.pairs)
    digest = hashlib.sha1(src.read_bytes()).hexdigest()[:12]
    print(f"source: {src.name} sha1={digest} pairs={len(pairs)}")

    if args.report:
        mapping, stats = build(pairs, register_inns(args.drugs))
    else:
        mapping, stats = apply_db(pairs, args.drugs)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    spot = ["ибупрофен", "дексаметазон", "левотироксин натрия",
            "аскорбиновая кислота",
            "железа сульфат + аскорбиновая кислота"]
    for s in spot:
        en, lat = mapping.get(s, (None, None))
        print(f"  spot {s!r}: en={en!r} lat={lat!r}")
    if not args.report:
        print(f"-> updated {args.drugs} (columns inn_en, inn_lat)")


if __name__ == "__main__":
    main()

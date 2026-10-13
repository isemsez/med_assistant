"""Register-grounded trade-name -> МНН lookup (architecture.md §3[1]).

`data/drugs.sqlite` (built by `scripts/grls_to_sqlite.py` from a ГРЛС
register export) maps «Торговое наименование» -> МНН. The doctor's question
is scanned for trade names; matches ground the answerer (a «справочно» P.S.
block after the question) and persist as a `.sources.drug` block under the
question bubble. A missing DB silently disables the whole feature.

Matching v2 — the owner's dictionary scheme (owner decision 2026-10-13,
replacing the v1 stem+endings regex scan; probe-validated against the real
register, 92.1% unambiguous keys). Build: every trade name (≤64 chars,
1–5 words) with a real МНН is keyed by its first word with only the last
letter stripped (never the first). Leading 1–2 letter segments followed by
`-`/`,` (stereo prefixes: «L-Тироксин», «Д-Пантенол», «D,L-Хлорамфеникол»)
are dropped so the key lands on the next word; digit-led names are skipped
entirely; keys <3 chars are skipped. Placeholder МНН (`~`) rows are not
keyed. Lookup: question tokens ≥5 chars, minus a stoplist, match the
LONGEST key they start with. A key with one distinct МНН resolves directly;
an ambiguous key restricts on the adjacent bigram — the first-word token
immediately followed by a token prefix-matching a later word of a name
(«нурофен плюс» — adjacency required, «плюс» alone is a common word) — and
otherwise falls back to the modal МНН (most register rows; tie → the МНН
matching the names' first word; then stable). When nothing matched at all:
the curated alias table `data/drug_aliases.yaml` (value = register trade
name resolved through this dict, or a direct МНН for register holes — the
file is owner-curated truth, never auto-extended), then a tight typo tier
(Levenshtein ≤2, same first 3 chars, token ≥7 chars, key must be
unambiguous). Results are deduped and capped.
"""

import re
import sqlite3
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[а-яa-z0-9]+")
# leading 1–2 letter segments followed by - or , (L-, Д-, D,L-)
LEAD_SEG_RE = re.compile(r"^(?:[а-яa-z]{1,2}[-,])+")
MAX_NAME_CHARS = 64
MAX_TOKENS = 5                  # longer register names are descriptive noise
KEY_MIN = 3                     # keys shorter than this never exist
TOKEN_MIN = 5                   # question tokens shorter never match a key
FUZZY_TOKEN_MIN = 7             # the typo tier needs a longer token
FUZZY_MAX_DIST = 2              # Levenshtein
FUZZY_HEAD = 3                  # token and key must share the first N chars
PLACEHOLDER_INNS = {"", "~", "-", "—", "?"}
TRADE_MARKS = "®™"              # stripped from display names
DEFAULT_CAP = 5

# Common question words that must never match a key (empirical; extend per
# validation — a false positive costs only a справочно line).
STOP_WORDS = {"таблетки", "таблеток", "препарата", "препараты", "железо"}


def _norm(text):
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def clean_inn(inn):
    """Register МНН -> readable form: «Железа сульфат+[Аскорбиновая
    кислота]» -> «железа сульфат + аскорбиновая кислота»."""
    s = re.sub(r"[\[\]]", "", (inn or "")).strip()
    s = re.sub(r"\s*\+\s*", " + ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _word_key(word):
    """тироксин -> тирокси: only the last letter is stripped, never more."""
    return word[:-1] if len(word) >= 5 else word


def _display_name(trade):
    """Register name as shown to the physician: ®/™ removed everywhere,
    whitespace collapsed («Нурофен® Плюс» -> «Нурофен Плюс»)."""
    s = trade.replace("®", "").replace("™", "")
    return re.sub(r"\s{2,}", " ", s).strip() or trade


def _levenshtein(a, b, cap):
    """DP distance, early-exit once every entry of a row exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,            # deletion
                           cur[j - 1] + 1,         # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


class DrugMap:
    def __init__(self, rows, aliases=None):
        """rows: (trade_name, inn) register rows (every row counts toward
        the modal МНН). `aliases`: curated {colloquial: value} map — the
        value either names a register trade name (resolved through this
        dict) or is a direct МНН for register holes."""
        groups = {}
        for trade, inn in rows:
            if not trade or len(trade) > MAX_NAME_CHARS:
                continue
            inn = clean_inn(inn)
            if not inn or inn in PLACEHOLDER_INNS:
                continue  # placeholder-МНН rows are not keyed
            norm = _norm(trade)
            if not norm or norm[0].isdigit():
                continue  # «1,3-Диэтил…», «9 месяцев Фолиевая кислота»
            words = WORD_RE.findall(LEAD_SEG_RE.sub("", norm))
            if not words or len(words) > MAX_TOKENS:
                continue
            key = _word_key(words[0])
            if len(key) < KEY_MIN:
                continue
            groups.setdefault(key, []).append(
                (_display_name(trade), inn, words))
        self._groups = {k: (m, Counter(inn for _, inn, _ in m))
                        for k, m in groups.items()}
        # alias: stemmed match key -> (normalized alias for display, value)
        self._aliases = {}
        for alias, value in (aliases or {}).items():
            a = _norm(alias)
            if len(_word_key(a)) >= KEY_MIN and _norm(value):
                self._aliases[_word_key(a)] = (a, _norm(value))

    def __len__(self):
        return sum(len(m) for m, _ in self._groups.values())

    # -- resolution ---------------------------------------------------------

    @staticmethod
    def _modal(members):
        """Most register rows; tie -> the МНН equal to / starting with the
        names' first word; then shorter (base drugs before combos), stable."""
        counter = Counter(inn for _, inn, _ in members)

        def rank(inn):
            self_named = any(
                inn == w[0] or inn.startswith(_word_key(w[0]))
                for _, i, w in members if i == inn)
            return (counter[inn], self_named, -len(inn), inn)

        return max(counter, key=rank)

    @staticmethod
    def _display(members, inn):
        resolved = [m for m in members if m[1] == inn]
        # prefer an exact first-word-only name («Нурофен», «Аспирин»),
        # then the shortest member of the resolved set
        return min(resolved, key=lambda m: (len(m[2]) > 1, len(m[0])))[0]

    def _resolve(self, key, next_token):
        """One key -> {"trade", "inn"}: unambiguous keys resolve directly;
        ambiguous keys restrict on the adjacent bigram first (the token
        right after the matched one prefix-matching a later word of a
        name), then fall back to the modal МНН."""
        members, counter = self._groups[key]
        pool = members
        if len(counter) > 1 and next_token:
            hit = [m for m in pool
                   if any(next_token.startswith(_word_key(w))
                          for w in m[2][1:])]
            if hit:
                pool = hit
        inns = {inn for _, inn, _ in pool}
        inn = next(iter(inns)) if len(inns) == 1 else self._modal(pool)
        return {"trade": self._display(pool, inn), "inn": inn}

    def _match_key(self, token):
        """The LONGEST dict key that is a prefix of the token, or None."""
        for n in range(len(token), KEY_MIN - 1, -1):
            if token[:n] in self._groups:
                return token[:n]
        return None

    # -- lookup tiers -------------------------------------------------------

    def _dict_matches(self, text, cap):
        """Dict tier (also resolves alias values): tokens ≥5 chars, not
        stoplisted; the raw token right after a match may restrict an
        ambiguous group (adjacency, no length filter — «плюс», «для»)."""
        raw = WORD_RE.findall(_norm(text))
        out, seen = [], set()
        for i, tok in enumerate(raw):
            if len(tok) < TOKEN_MIN or tok in STOP_WORDS:
                continue
            key = self._match_key(tok)
            if key is None:
                continue
            m = self._resolve(key, raw[i + 1] if i + 1 < len(raw) else None)
            m["token"] = tok  # the doctor's word, anchored to the register row
            sig = (_norm(m["trade"]), m["inn"])
            if sig not in seen:
                seen.add(sig)
                out.append(m)
                if len(out) >= cap:
                    break
        return out

    def _names_trade(self, value):
        """True when the alias value actually names a register trade name
        (its word sequence prefixes a keyed member's words) — then it
        resolves through the dict; otherwise it is a direct МНН for a
        register hole. Guards against a value like «меглюмина
        акридонацетат» hijacking the unrelated name «Меглюмина натрия
        сукцинат» via a shared first word."""
        words = WORD_RE.findall(value)
        if not words:
            return False
        key = self._match_key(words[0])
        if key is None:
            return False
        members, _ = self._groups[key]
        n = len(words)
        return any(m[2][:n] == words for m in members)

    def _alias_matches(self, text, cap):
        raw = WORD_RE.findall(_norm(text))
        out, seen = [], set()
        for tok in raw:
            if len(tok) < TOKEN_MIN or tok in STOP_WORDS:
                continue
            hit = None
            for a_key in self._aliases:
                if tok.startswith(a_key) and (hit is None
                                              or len(a_key) > len(hit)):
                    hit = a_key
            if hit is None:
                continue
            alias, value = self._aliases[hit]
            # the value names a register trade name -> resolve through the
            # dict (carrying the doctor's word so the P.S. can connect it);
            # otherwise it is a direct МНН (register hole: row carries `~`
            # or the substance has no row at all) and the displayed trade
            # name already IS the doctor's word
            if self._names_trade(value):
                matches = [{**{k: v for k, v in m.items() if k != "token"},
                            "via": tok}
                           for m in self._dict_matches(value, cap=1)]
            else:
                matches = [{"trade": alias.capitalize(),
                            "inn": clean_inn(value)}]
            for m in matches:
                sig = (_norm(m["trade"]), m["inn"])
                if sig not in seen:
                    seen.add(sig)
                    out.append(m)
            if len(out) >= cap:
                break
        return out

    def _fuzzy_matches(self, text, cap):
        raw = WORD_RE.findall(_norm(text))
        out, seen = [], set()
        for tok in raw:
            if len(tok) < FUZZY_TOKEN_MIN or tok in STOP_WORDS:
                continue
            best = None  # (dist, key)
            for key, (members, counter) in self._groups.items():
                if len(counter) != 1 or not key.startswith(tok[:FUZZY_HEAD]):
                    continue
                d = _levenshtein(tok, key, FUZZY_MAX_DIST)
                if d <= FUZZY_MAX_DIST and (best is None or (d, key) < best):
                    best = (d, key)
            if best is not None:
                m = self._resolve(best[1], None)
                m["token"] = tok  # the doctor's (typo'd) word
                sig = (_norm(m["trade"]), m["inn"])
                if sig not in seen:
                    seen.add(sig)
                    out.append(m)
                    if len(out) >= cap:
                        break
        return out

    def lookup(self, question, cap=DEFAULT_CAP):
        """[{"trade", "inn"}, ...] for trade names found in the question,
        by first occurrence. Dict matching first; only when nothing matched,
        the curated aliases, then the tight typo tier."""
        q = _norm(question)
        if not q:
            return []
        out = self._dict_matches(q, cap)
        if not out:
            out = self._alias_matches(q, cap)
        if not out:
            out = self._fuzzy_matches(q, cap)
        return out


ALIAS_LINE_RE = re.compile(r"^([^:#]+?)\s*:\s*(.+)$")


def _load_aliases(path):
    """data/drug_aliases.yaml — flat `colloquial: value`, owner-curated
    truth (never auto-extended). Missing/broken file = no aliases."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = ALIAS_LINE_RE.match(line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def load(path=None, aliases_path=None):
    """drugs.sqlite (+ optional alias file) -> DrugMap; None when the DB is
    missing (feature silently off)."""
    if path is None or aliases_path is None:
        from . import config
        path = path or config.DRUGS_DB
        aliases_path = aliases_path or config.DRUG_ALIASES_PATH
    p = Path(path)
    if not p.exists():
        return None
    con = sqlite3.connect(str(p))
    try:
        rows = con.execute("select trade_name, inn from drugs").fetchall()
    finally:
        con.close()
    return DrugMap(rows, aliases=_load_aliases(Path(aliases_path)))

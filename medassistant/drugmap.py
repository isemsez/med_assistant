"""Register-grounded trade-name -> МНН lookup (architecture.md §3[1]).

`data/drugs.sqlite` (built by `scripts/grls_to_sqlite.py` from a ГРЛС
register export) maps «Торговое наименование» -> МНН. The doctor's question
is scanned for trade names; matches ground the answerer (a «справочно» P.S.
block after the question) and surface as a UI hint. A missing DB silently
disables the whole feature.

Matching is inflection-tolerant (owner decision 2026-10-12): text is
normalized (lowercase, ё→е), every trade-name token is matched as a stem
plus up to 3 case-ending chars, tokens must appear in order. A cheap
first-stem substring prefilter keeps the scan over ~12k names in the
milliseconds; names subsumed by a longer match (Сорбифер vs Сорбифер
Дурулес) are dropped.
"""

import re
import sqlite3
from pathlib import Path

WORD_RE = re.compile(r"[а-яa-z0-9]+")
SEP = r"[^а-яa-z0-9]+"          # any non-word run between name tokens
BARRIER = r"(?<![а-яa-z0-9])"   # no word chars immediately before/after
STEM_MIN = 4                    # shorter tokens match verbatim only
STEM_CUT = 1                    # chars dropped from a token to form the stem
MAX_ENDINGS = 3                 # inflection chars allowed after the stem
MAX_TOKENS = 5                  # longer register names are descriptive noise
MAX_NAME_CHARS = 64
PLACEHOLDER_INNS = {"", "~", "-", "—", "?"}
TRADE_MARKS = "®™"  # stripped from the display name, matching ignores them
DEFAULT_CAP = 5


def _norm(text):
    return re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()


def clean_inn(inn):
    """Register МНН -> readable form: «Железа сульфат+[Аскорбиновая
    кислота]» -> «железа сульфат + аскорбиновая кислота»."""
    s = re.sub(r"[\[\]]", "", (inn or "")).strip()
    s = re.sub(r"\s*\+\s*", " + ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _token_pattern(token):
    if len(token) <= STEM_MIN:
        return re.escape(token)
    stem = re.escape(token[:-STEM_CUT])
    return f"{stem}[а-яa-z]{{0,{MAX_ENDINGS}}}"


class _Name:
    __slots__ = ("trade", "inn", "pattern_src", "prefilter")

    def __init__(self, trade, inn):
        self.trade = trade
        self.inn = inn
        tokens = WORD_RE.findall(_norm(trade))
        self.pattern_src = (BARRIER + (SEP + BARRIER).join(
            _token_pattern(t) for t in tokens) + r"(?![а-яa-z0-9])")
        first = tokens[0]
        self.prefilter = first[:-STEM_CUT] if len(first) > STEM_MIN else first


class DrugMap:
    def __init__(self, rows):
        """rows: iterable of (trade_name, inn) from the register; duplicate
        trade names keep the first real МНН seen."""
        by_norm = {}
        for trade, inn in rows:
            if not trade or len(trade) > MAX_NAME_CHARS:
                continue
            tokens = WORD_RE.findall(_norm(trade))
            if not tokens or len(tokens) > MAX_TOKENS:
                continue
            key = _norm(trade)
            if key not in by_norm:
                display = trade.strip(TRADE_MARKS).strip() or trade
                by_norm[key] = [display, None]
            clean = clean_inn(inn)
            if (clean and clean not in PLACEHOLDER_INNS
                    and by_norm[key][1] is None):
                by_norm[key][1] = clean
        self._names = [_Name(trade, inn) for trade, inn in by_norm.values()]

    def __len__(self):
        return len(self._names)

    def lookup(self, question, cap=DEFAULT_CAP):
        """[{"trade", "inn"}, ...] for trade names found in the question,
        by first occurrence; `inn` is None when the register carries no real
        МНН for the name (no P.S. line, audit-only)."""
        q = _norm(question)
        if not q:
            return []
        found = []
        for entry in self._names:
            if entry.prefilter not in q:
                continue
            m = re.search(entry.pattern_src, q)
            if m:
                found.append((m.start(), len(_norm(entry.trade)), entry))
        found.sort(key=lambda t: (t[0], -t[1]))
        out, token_sets = [], []
        for _, _, entry in found:
            tokens = set(WORD_RE.findall(_norm(entry.trade)))
            # subsumed: every token already belongs to a longer match
            if any(tokens <= seen for seen in token_sets):
                continue
            token_sets.append(tokens)
            out.append({"trade": entry.trade, "inn": entry.inn})
            if len(out) >= cap:
                break
        return out


def load(path=None):
    """drugs.sqlite -> DrugMap; None when the file is missing (feature off)."""
    if path is None:
        from . import config
        path = config.DRUGS_DB
    p = Path(path)
    if not p.exists():
        return None
    con = sqlite3.connect(str(p))
    try:
        rows = con.execute("select trade_name, inn from drugs").fetchall()
    finally:
        con.close()
    return DrugMap(rows)

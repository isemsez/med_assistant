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
LONGEST key they start with, provided the token is a real inflected
form of the key's original first word — the word itself, word+ending,
or (a vowel/ь/й-final word) the word minus that letter + ending, with
endings from a closed set. The raw word+≤3 length cap (2026-10-16) let
boundary words through («анестетик» = «анеста»+3 grounded АНЕСТА-А,
«азотемия» = «азота»+3); the endings whitelist (owner decision
2026-10-17) kills the whole class — non-ending tails are different
words — while real forms survive («нурофенами», «омеза», «грудного»).
A key with one distinct МНН resolves directly;
an ambiguous key restricts on the adjacent bigram — the first-word token
immediately followed by a token prefix-matching a later word of a name
(«нурофен плюс» — adjacency required, «плюс» alone is a common word) — and
otherwise falls back to the modal МНН (most register rows; tie → the МНН
matching the names' first word; then stable). When nothing matched at all:
the curated alias table `data/drug_aliases.yaml` (value = register trade
name resolved through this dict, or a direct МНН for register holes — the
file is owner-curated truth, never auto-extended; a recurring doctor
typo gets a curated alias the same way). The Levenshtein typo tier
(2026-10-13..17) was removed 2026-10-17: across the whole live history
it caught zero real typos and produced one documented false positive
(«перкуссии»→Пертуссин at the old dist 2), while dist-1 kept admitting
common words («панкреатит»→Панкреатин). A shared stoplist suppresses
disease-stem collisions in both the dict and alias tiers (owner decision
2026-10-15: brand names are built on disease stems — «артроза»→Артрозан,
«витамины»→Витамин D3, «диабетом»→Диабетон МВ — so inflected disease
words collide at any distance; disease and drug inflections are distinct
tokens, exact stoplist entries never block the drug forms). Results are
deduped and capped.
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
# Closed set of Russian nominal/adjectival case endings (owner decision
# 2026-10-17, replacing the raw word+≤3 length cap of 2026-10-16 that
# boundary words slipped through: «анестетик» = «анеста»+3, «азотемия» =
# «азота»+3). A token grounds a key only as слово+окончание or — for
# words ending in a vowel, ь or й — слово-minus-that-letter+окончание;
# any other tail («етик», «емия», «тый», digits like «бета2») is a
# different word, not an inflection. Real forms survive: «нурофенами»,
# «омеза», «истурисы», «грудного» (adjectival й-stem).
ENDINGS = frozenset({
    "а", "я", "у", "ю", "е", "и", "ы",           # sg gen/dat/acc/prep
    "ой", "ою", "ей", "ею", "ов", "ев", "ью",    # instr sg, gen pl
    "ом", "ем", "ам", "ям", "ах", "ях",          # instr sg, dat/prep pl
    "ок", "ек",                                  # fleeting-vowel gen pl
    "ым", "им", "ого", "его", "ому", "ему",      # adjective sg
    "ая", "яя", "ое", "ее", "ые", "ие", "ых", "их", "ую", "юю",  # adj
    "ами", "ями", "ыми", "ими",                  # instrumental pl
})
# word finals whose replacement is declension: vowels + ь + adjectival й
STEM_FINALS = "аяоиеыуюьй"
# adjective nominative finals — fully replaced by case endings
# («грудной» -> «грудн»+«ого», «седативный» -> «седативн»+«ого»)
ADJ_FINALS = ("ый", "ой", "ий")
PLACEHOLDER_INNS = {"", "~", "-", "—", "?"}
TRADE_MARKS = "®™"              # stripped from display names
DEFAULT_CAP = 5

# Common question words that must never match a key (empirical; extend per
# validation — a false positive injects a wrong drug into the answerer's
# context, and the 4B weaves it into the answer). Exact tokens only, NEVER
# prefixes: disease and drug inflections are distinct tokens («диабетом»
# болезнь vs «диабетоном» лекарство, «витамины» vs «витамин» в «витамин
# д3»), so a disease-form entry leaves the drug form intact — but a
# «бронхит*» prefix would block the real drug Бронхитол. The
# disease-stem families below (owner decision 2026-10-15) collided live
# in the dict tier: brand names are built on disease stems (Артрозан←артроз,
# Диабетон←диабет, Витамин D3←витамин, Бронхо-мунал←бронх-,
# ПОДАГРУМ←подагра).
STOP_WORDS = {
    "таблетки", "таблеток", "препарата", "препараты", "железо",
    # артрит*: «ревматоидного артрита» grounded Артра (key «артр») within
    # the inflation cap — the артрит family joins артроз below (2026-10-16);
    # the drug token «артра» stays
    "артрит", "артрита", "артрите", "артриту", "артритом",
    "артритов", "артритам", "артритах", "артритами",
    # артроз*: «лечение артроза» grounded Артрозан/Артра
    "артроз", "артроза", "артрозе", "артрозу", "артрозом",
    "артрозов", "артрозам", "артрозах",
    # бронхит*: «при бронхите» grounded Бронхо-мунал (key «бронх»)
    "бронхит", "бронхита", "бронхите", "бронхиту", "бронхитом",
    "бронхитов", "бронхитам", "бронхитах",
    # витамин* plural/adjective: «какие витамины» grounded Витамин D3;
    # singular forms stay — real «витамин д3» questions
    "витамины", "витаминов", "витаминам", "витаминах", "витаминами",
    "витаминные", "витаминных",
    # диабет*: «при диабете» grounded Диабетон МВ (key «диабето»)
    "диабета", "диабете", "диабету", "диабетом",
    # подагр*: «подагра» grounded ПОДАГРУМ / Подагрель
    "подагра", "подагре", "подагры", "подагру",
    # эффект*: «как долго оценивать эффект» grounded ЭФФЕКС Силденафил
    # in a neuropathic-pain question (2026-10-16). The endings rule now
    # kills «эффект» structurally — «эффекс» ends in a consonant, so the
    # same-length substitution «эффек»+«т» is never an inflection — the
    # entries stay as belt-and-braces; the real token «эффекс» stays
    "эффект", "эффекта", "эффекту", "эффектом", "эффекте",
    "эффекты", "эффектов", "эффектам", "эффектами", "эффектах",
    # азотеми*: «азота закись» is keyed by its own genitive «азота», and
    # «азотемия» = «азот»+«емия» was a live false positive (2026-10-16);
    # «емия» is not an ending, so the endings rule kills the family —
    # belt-and-braces
    "азотемия", "азотемии", "азотемию", "азотемий",
    "азотемией", "азотемиях", "азотемиям",
    # анестетик: «если он местный анестетик» grounded АНЕСТА-А (key word
    # «анеста», 2026-10-16, session 70); «етик» is not an ending, so the
    # endings rule kills it — belt-and-braces; «анеста» stays
    "анестетик",
    # Known accepted residual (2026-10-16): «азота» (5) still grounds
    # Азота закись («уровень азота мочевины в крови») — the drug's own
    # genitive IS «азота», an exact word the endings rule must admit.
}


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


class DrugMap:
    def __init__(self, rows, aliases=None):
        """rows: register rows (trade_name, inn[, inn_pair]) — inn_pair is
        the pre-rendered bilingual МНН written by scripts/inn_to_en.py
        into drugs.sqlite («железа сульфат (ferrous sulfate) +
        аскорбиновая кислота (ascorbic acid)», owner decision 2026-10-14).
        `aliases`: curated {colloquial: value} map — the value either
        names a register trade name (resolved through this dict) or is a
        direct МНН for register holes."""
        groups = {}
        self._db_gloss = {}
        for row in rows:
            trade, inn = row[0], row[1]
            inn_pair = row[2] if len(row) > 2 else None
            if not trade or len(trade) > MAX_NAME_CHARS:
                continue
            inn = clean_inn(inn)
            if not inn or inn in PLACEHOLDER_INNS:
                continue  # placeholder-МНН rows are not keyed
            if inn_pair:
                self._db_gloss[inn] = inn_pair.strip()
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
        # original first words per key: the endings whitelist checks the
        # token against them (keys of ≥5-letter words lost their last
        # letter, and one key may mix words of different lengths)
        self._key_words = {k: {m[2][0] for m in ms}
                           for k, ms in groups.items()}
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

    @staticmethod
    def _inflection_ok(token, words):
        """True when the token is a real inflected form of one of the
        words: the exact word, word+ending, or the word minus its final
        vowel/ь/й (or minus an adjective ый/ой/ий pair) + ending. Only
        the closed ENDINGS set qualifies: tails like «анеста»+«етик» or
        «эффек»+«т» (consonant-final substitution) are different words,
        not inflections (owner decision 2026-10-17)."""
        for w in words:
            if token == w:
                return True
            if token.startswith(w) and token[len(w):] in ENDINGS:
                return True
            stems = [w[:-1]] if w[-1] in STEM_FINALS else []
            if w[-2:] in ADJ_FINALS:
                stems.append(w[:-2])
            for stem in stems:
                if token.startswith(stem) and token[len(stem):] in ENDINGS:
                    return True
        return False

    def _match_key(self, token):
        """The LONGEST dict key that is a prefix of the token and whose
        token is a real inflected form of the key's original first word
        (see _inflection_ok), or None. The endings whitelist replaces the
        2026-10-16 length cap: «максимальной», «анестетик», «азотемия»,
        «бета2» are not forms of any keyed word."""
        for n in range(len(token), KEY_MIN - 1, -1):
            key = token[:n]
            if key in self._groups and self._inflection_ok(
                    token, self._key_words[key]):
                return key
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
            # the same inflection whitelist as the dict tier
            if not self._inflection_ok(tok, {alias}):
                continue
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

    def _glossed(self, inn):
        """Pre-rendered bilingual МНН from the register's inn_pair column
        (scripts/inn_to_en.py builds it: Russian components with English
        in parentheses after each substance that has one). None when the
        row has no translation — the plain Russian МНН is used."""
        return self._db_gloss.get(inn)

    def lookup(self, question, cap=DEFAULT_CAP):
        """[{"trade", "inn", ...}, ...] for trade names found in the
        question, by first occurrence; `token`/`via` anchor the doctor's
        word, `en` carries the rendered per-component English gloss when
        any component translates. Dict matching first; only when nothing
        matched, the curated aliases (the typo tier was removed
        2026-10-17 — see the module docstring)."""
        q = _norm(question)
        if not q:
            return []
        out = self._dict_matches(q, cap)
        if not out:
            out = self._alias_matches(q, cap)
        for m in out:
            en = self._glossed(m["inn"])
            if en:
                m["en"] = en
        return out


ALIAS_LINE_RE = re.compile(r"^([^:#]+?)\s*:\s*(.+)$")


def _load_flat(path):
    """Flat `key: value` yaml with # comments (drug_aliases.yaml,
    inn_gloss.yaml) — owner-curated truth, never auto-extended. Missing
    or broken file = empty map."""
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
    """drugs.sqlite (+ curated alias file) -> DrugMap; None when the DB is
    missing (feature silently off). Bilingual МНН strings come from the
    inn_pair column written by scripts/inn_to_en.py (WHO list + curated
    glosses folded at build time)."""
    if path is None or aliases_path is None:
        from . import config
        path = path or config.DRUGS_DB
        aliases_path = aliases_path or config.DRUG_ALIASES_PATH
    p = Path(path)
    if not p.exists():
        return None
    con = sqlite3.connect(str(p))
    try:
        try:
            rows = con.execute(
                "select trade_name, inn, inn_pair from drugs").fetchall()
        except sqlite3.OperationalError:  # pre-migration DB, no column yet
            rows = con.execute("select trade_name, inn from drugs").fetchall()
    finally:
        con.close()
    return DrugMap(rows, aliases=_load_flat(Path(aliases_path)))

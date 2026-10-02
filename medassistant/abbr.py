"""Abbreviation expansion for guideline-search queries (architecture.md §3[3]).

`data/abbr2term.yaml` (copied from the RuCCoD project — NO LICENSE there:
internal on-premise use only, do not redistribute) is a flat `АББР: термин`
map. Expansions ADD query variants, never replace the original name.
"""

import re
from pathlib import Path

ABBR_LINE_RE = re.compile(r"^([\u0410-\u042f\u0401][\u0410-\u042f\u0401A-Z0-9-]*)\s*:\s*(.+)$")
WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z]+")

MAX_VARIANTS = 5


def parse_abbr2term(text):
    """Flat YAML text -> {UPPER_ABBR: term}."""
    out = {}
    for line in text.splitlines():
        m = ABBR_LINE_RE.match(line.strip())
        if m:
            out[m.group(1).upper()] = m.group(2).strip()
    return out


def load_abbr(path):
    return parse_abbr2term(Path(path).read_text(encoding="utf-8"))


class Abbr:
    def __init__(self, mapping):
        self.mapping = {str(k).upper(): v for k, v in (mapping or {}).items()}

    def __len__(self):
        return len(self.mapping)

    def variants(self, disease, cap=MAX_VARIANTS):
        """[original name, expansion of each uppercase token, …] — dedup, capped.

        Entirely lowercase tokens are skipped: abbreviations are written
        uppercase (ХСН), and lowercase collisions with common words are real.
        """
        out = [disease]
        for token in WORD_RE.findall(disease or ""):
            if token.islower():
                continue
            expansion = self.mapping.get(token.upper())
            if expansion and expansion not in out:
                out.append(expansion)
            if len(out) >= cap:
                break
        return out

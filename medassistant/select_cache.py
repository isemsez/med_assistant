"""Persistent pair-verdict select cache (owner decision 2026-10-26).

The selector's judgment per (disease, audience bucket) -> guideline id is
frozen as a RIGHT/WRONG pair verdict in data/select_cache.yaml and reused
on later turns with the same disease: the live log showed 81% of select
group runs (87% of select seconds, ~77 of ~88 minutes across 29 real web
sessions) were exact (session, disease) repeats, and the 4B selector is
unstable on them — the same disease attached different pdf sets on
consecutive turns («инфаркт миокарда»: [153_2,154_4,155_2] then
[155_2,157_5,408_2] then [157_5]) and every repeat re-rolled the
select-loop coin flip. A cached verdict freezes the first healthy outcome
and makes follow-ups instant and consistent.

Verdict semantics (owner model): `right` — the pair's guideline attaches
normally in the appendix; `wrong` — it does not attach and renders grayed
inside the existing «отфильтрованные» collapsible (still visible, still
clickable, still revisable). Human revision (the row buttons): 👎 negates
the pair's CURRENT verdict (toggle right<->wrong), 👍 — shown only while a
pair was never revised — confirms the machine's verdict; any click marks
the pair human-revised (only 👎 remains afterwards), and a human revision
is never overwritten by machine writes.

The file is human-readable/hand-editable yaml like drug_aliases.yaml but,
unlike it, machine-written; it is local runtime state (never shipped in
the deploy zip) stamped with docs_meta_sha1 — a refreshed register
invalidates every verdict (ids may have shifted meaning) and the cache
rebuilds through use.
"""

import threading
from pathlib import Path

import yaml


def norm_disease(name):
    """Cache key form of an extracted disease: strip, lower, ё->е."""
    return (name or "").strip().lower().replace("ё", "е")


def _norm_bucket(bucket):
    return (bucket or "").strip().lower() or "none"


class SelectCache:
    def __init__(self, path, docs_sha1):
        self.path = Path(path)
        self.docs_sha1 = docs_sha1
        self._lock = threading.Lock()
        self._entries = {}
        self._reset = False
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return  # unreadable cache = empty cache; the first write re-creates it
        if data.get("docs_meta_sha1") != self.docs_sha1:
            self._reset = True
            return
        entries = data.get("entries")
        if isinstance(entries, dict):
            self._entries = entries

    def was_reset(self):
        """True exactly once when the loaded file's docs stamp mismatched —
        the caller logs `select_cache_reset`, then the flag clears."""
        with self._lock:
            r, self._reset = self._reset, False
            return r

    # -- read ---------------------------------------------------------------

    def get(self, disease, bucket, code_version):
        """{verdict: "right"|"wrong", human?: True} or None when unknown."""
        with self._lock:
            pairs = self._entries.get(norm_disease(disease), {})
            e = pairs.get(_norm_bucket(bucket), {}).get(code_version)
            return dict(e) if e else None

    def all_verdicts(self):
        """Deep copy for GET /api/verdicts."""
        with self._lock:
            return {d: {b: {cv: dict(e) for cv, e in pairs.items()}
                        for b, pairs in buckets.items()}
                    for d, buckets in self._entries.items()}

    # -- write --------------------------------------------------------------

    def _pairs(self, disease, bucket):
        return (self._entries.setdefault(norm_disease(disease), {})
                .setdefault(_norm_bucket(bucket), {}))

    def put_machine(self, disease, bucket, verdicts):
        """{code_version: "right"|"wrong"} from one settled turn. A machine
        write never touches a human-revised pair — the doctor's click
        outranks the selector's re-judgment."""
        if not verdicts:
            return
        with self._lock:
            pairs = self._pairs(disease, bucket)
            for cv, verdict in verdicts.items():
                if pairs.get(cv, {}).get("human"):
                    continue
                pairs[cv] = {"verdict": verdict}
            self._save()

    def negate(self, disease, bucket, code_version, current):
        """👎: flip the verdict — of the stored entry when one exists, else
        of `current` (the row's displayed zone; legacy appendix rows may
        name a pair never cached) — and mark the pair human-revised."""
        with self._lock:
            pairs = self._pairs(disease, bucket)
            stored = pairs.get(code_version, {}).get("verdict") or current
            entry = {"verdict": "wrong" if stored == "right" else "right",
                     "human": True}
            pairs[code_version] = entry
            self._save()
            return dict(entry)

    def confirm(self, disease, bucket, code_version, current):
        """👍 (offered only before the first revision): confirm the current
        verdict as human-revised."""
        with self._lock:
            pairs = self._pairs(disease, bucket)
            stored = pairs.get(code_version, {}).get("verdict") or current
            entry = {"verdict": stored, "human": True}
            pairs[code_version] = entry
            self._save()
            return dict(entry)

    def forget(self, disease, bucket, code_version):
        """↻ (owner decision 2026-10-26 evening, replacing the whole-bucket
        drop): remove ONE pair's verdict — the pair becomes unknown again
        and the next turn's live selector re-judges exactly it (partial
        coverage: cached pairs keep their verdicts, only the unknowns are
        judged). Machine or human, it goes the same way — the owner asked
        for the re-judge. Returns True when an entry was removed."""
        with self._lock:
            key = norm_disease(disease)
            buckets = self._entries.get(key)
            if not buckets:
                return False
            bkey = _norm_bucket(bucket)
            pairs = buckets.get(bkey)
            if not pairs or code_version not in pairs:
                return False
            del pairs[code_version]
            if not pairs:
                buckets.pop(bkey, None)
                if not buckets:
                    self._entries.pop(key, None)
            self._save()
            return True

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"docs_meta_sha1": self.docs_sha1, "entries": self._entries}
        self.path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8")

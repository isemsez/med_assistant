"""Golden-set semantic regression against live endpoints (manual pre-ship run:
python -m medassistant regression).

Goldens live in tests/goldens.jsonl (lines starting with '#' are skipped).
Two tiers per row:

  search tier  {"disease", "expect_name_contains"}
      deterministic: real index + embed endpoint only (fast). The expected
      guideline must be the top-1 search hit for the disease (abbreviation
      expansion included). Substring match on the guideline display name —
      tolerant across register refreshes that shift version codes and
      adult/pediatric variants.

  full tier    {"question", "expect_status"?, "expect_name_contains"?}
      the whole pipeline incl. answerer generation and both labeler calls
      (minutes per row on CPU). expect_status defaults to "ok".

Full-tier rows run with a throwaway audit file and are RED when the turn
produced a select thinking loop (`labeler_select_loop`, 2026-10-21): a
graceful runtime degradation (dropped picks, title-guard rescue) still
means the selector degenerated — the golden set demands it NOT, so a loop
fails the row even when the shipped picks are correct.

A row may carry "skip": "<reason>" (owner decision 2026-10-25): it prints as
[SKIP] with its reason and is excluded from the pass/fail count — the exit
code ignores skipped rows (temp-off lever for a known-red row under
investigation, the reason ships inside the row).

Exit code 0 = all non-skipped rows pass. The temp audit is deleted after
the run.
"""

import json
import time
from pathlib import Path

from . import config, llm
from .canary import answer_with_retry


def load_goldens(path):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"goldens file not found: {p}")
    rows = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{p}:{i}: invalid JSON row: {e}")
        if "question" in row and "disease" not in row:
            row["_tier"] = "full"
        elif ("disease" in row and "question" not in row
              and "expect_name_contains" in row):
            row["_tier"] = "search"
        else:
            raise SystemExit(f"{p}:{i}: row needs exactly one of "
                             "(disease + expect_name_contains) / question")
        rows.append(row)
    return rows


def _variants(pipe, disease):
    return (pipe.abbr_map.variants(disease)
            if pipe.abbr_map is not None else [disease])


def run_search_row(pipe, row):
    cards = pipe.index.search(_variants(pipe, row["disease"]), config.EMBED,
                              top_k=config.TOP_K)
    if not cards:
        return False, "no candidates"
    top = cards[0]
    stem = row["expect_name_contains"].lower()
    if stem not in (top["name"] or "").lower():
        return False, (f"top-1 «{top['name']}» ({top['code_version']}, "
                       f"{top['score']}) lacks «{row['expect_name_contains']}»")
    return True, f"{top['code_version']} «{top['name']}» {top['score']}"


def _count_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _select_loops_since(audit, start_line):
    """labeler_select_loop records written after line `start_line` of
    audit.path — the current row's slice of the shared throwaway audit."""
    if audit is None:
        return []
    loops = []
    try:
        with open(audit.path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if i <= start_line or not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("stage") == "labeler_select_loop":
                    loops.append(r.get("reason", ""))
    except OSError:
        pass
    return loops


def run_full_row(pipe, row, max_tokens):
    audit = getattr(pipe, "audit", None)
    start_line = _count_lines(audit.path) if audit is not None else 0
    t0 = time.monotonic()
    try:
        answer = answer_with_retry(pipe, row["question"], max_tokens)
    except llm.LLMError as e:
        return False, f"answerer failed: {e}", None
    if not answer.strip():
        return False, "empty answer twice", None
    result = pipe.run_appendix(row["question"], answer)
    # a select thinking loop is a FAILED row even when the title guard or
    # auto-pick shipped correct picks afterwards — the runtime degrades
    # gracefully, the golden set demands a healthy selector (2026-10-21)
    loops = _select_loops_since(audit, start_line)
    if loops:
        return False, f"select reasoning loop: {loops[0]}", result
    want_status = row.get("expect_status", "ok")
    if result.status != want_status:
        return (False,
                f"status {result.status} != {want_status} "
                f"(notes: {result.notes})", result)
    stem = row.get("expect_name_contains")
    if stem:
        picks = [p for g in result.by_disease for p in g["picks"]]
        if not any(stem.lower() in (p["name"] or "").lower() for p in picks):
            names = "; ".join(f"{p['name']} ({p['code_version']})"
                              for p in picks) or "(none)"
            return False, f"no pick contains «{stem}»: {names}", result
    # over-pick guard (2026-10-21): the per-group 4B selector rubber-
    # stamps — attaching nearly every candidate (live canary: 6 picks,
    # expected one among them); a row with expect_picks_max fails when
    # the selector stopped selecting
    max_picks = row.get("expect_picks_max")
    if max_picks is not None:
        n = sum(len(g["picks"]) for g in result.by_disease)
        if n > max_picks:
            names = "; ".join(f"{p['code_version']}"
                              for g in result.by_disease for p in g["picks"])
            return False, (f"{n} picks > expect_picks_max {max_picks} "
                           f"(rubber-stamp?): {names}"), result
    return True, (f"{result.status}, {len(answer)} chars, "
                  f"{time.monotonic() - t0:.0f}s"), result


def run(goldens_path, max_tokens=3072, tier="all"):
    import tempfile

    from .audit import Audit
    from .pipeline import load_pipeline
    rows = load_goldens(goldens_path)
    if tier != "all":
        rows = [r for r in rows if r["_tier"] == tier]
    n_search = sum(1 for r in rows if r["_tier"] == "search")
    print(f"goldens: {len(rows)} rows ({n_search} search-tier, "
          f"{len(rows) - n_search} full-tier) — {goldens_path}")
    with tempfile.TemporaryDirectory() as tmp:
        # throwaway audit: full-tier rows are checked for select loops
        # (labeler_select_loop) and deleted afterwards
        pipe = load_pipeline(audit=Audit(Path(tmp) / "audit.jsonl"))

        passed, skipped = _run_rows(pipe, rows, max_tokens)
    total = len(rows) - skipped
    ok = passed == total
    tail = f", {skipped} skipped" if skipped else ""
    print(f"\nREGRESSION {'OK' if ok else 'FAIL'}: "
          f"{passed}/{total} rows pass{tail}")
    return ok


def _run_rows(pipe, rows, max_tokens, out=print):
    """The row loop of run(): prints one [PASS]/[FAIL]/[SKIP] block per
    row, returns (passed, skipped). A truthy "skip" field prints the row as
    [SKIP] with its reason and keeps it out of the counts entirely — the
    exit code ignores skips (owner decision 2026-10-25)."""
    passed, skipped, auto_groups, total_groups = 0, 0, 0, 0
    for i, row in enumerate(rows, 1):
        tier, label = row["_tier"], row.get("disease") or row["question"]
        reason = row.get("skip")
        if reason:
            skipped += 1
            out(f"  [SKIP]  {tier:>6} #{i:<2} {label[:52]}\n"
                f"         — {reason}")
            continue
        try:
            if tier == "search":
                ok, detail = run_search_row(pipe, row)
            else:
                ok, detail, result = run_full_row(pipe, row, max_tokens)
                if result is not None:
                    groups = result.by_disease
                    total_groups += len(groups)
                    auto_groups += sum(1 for g in groups
                                       if g.get("auto_pick"))
        except Exception as e:  # one bad row must not kill the run
            ok, detail = False, f"{type(e).__name__}: {e}"
        passed += ok
        out(f"  [{'PASS' if ok else 'FAIL'}] {tier:>6} #{i:<2} "
            f"{label[:52]}\n"
            f"         {detail}")
    if total_groups:
        out(f"auto-pick rate on full-tier: {auto_groups}/{total_groups}")
    return passed, skipped


if __name__ == "__main__":
    raise SystemExit(0 if run(config.PROJECT_ROOT / "tests" / "goldens.jsonl") else 1)

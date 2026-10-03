"""End-to-end canary: health checks + one synthetic question through the real
pipeline (answerer -> extract -> search -> select/auto-pick -> renderer).

Run: python -m medassistant canary [--no-e2e] [--max-tokens N]
Exit code 0 = all green. No audit records are written (canary turns are not
physician usage); diagnostics go to stdout only.
"""

import time

from . import config, llm

# The synthetic case: a canonical abbreviated-cardiology question whose
# expected guideline is derivable deterministically from the index.
QUESTION = "Какие критерии диагностики хронической сердечной недостаточности?"
DISEASE = "хроническая сердечная недостаточность"
NAME_STEM = "недостаточност"  # covers -ость/-ости morphology


def _fmt(check):
    return "ok" if check.get("ok") else f"FAIL: {check.get('error', '?')}"


def health_phase():
    from . import health
    print("== Health checks ==")
    result = health.run_health(force=True)
    for name, check in result["checks"].items():
        extra = ""
        if not check.get("ok"):
            extra = f"  <- {_fmt(check)}"
        elif "fresh" in check:
            extra = (f" (fresh={check['fresh']}, docs={check.get('docs_meta')}, "
                     f"vectors={check.get('vectors')})")
        elif "pdfs" in check:
            extra = f" ({check['pdfs']} pdfs)"
        print(f"  {name:<10} {_fmt(check)}{extra}")
    return result["ok"]


def answer_with_retry(pipe, question, max_tokens, progress=False):
    """Stream an answer like production does (chunk timeouts, no read-timeout
    cliff on slow CPU generation); retry once with a doubled budget if
    truncation left it empty (thinking can eat a small budget). Test-harness
    helper — the live pipeline never retries at the semantic level."""
    budget, answer = max_tokens, ""
    for _ in (1, 2):
        parts, seen = [], 0
        for ev in pipe.stream_answer(question, max_tokens=budget):
            if ev["type"] == "content":
                parts.append(ev["text"])
                seen += len(ev["text"])
                if progress and seen // 400 > (seen - len(ev["text"])) // 400:
                    print(".", end="", flush=True)
        if progress:
            print(flush=True)
        answer = "".join(parts)
        if answer.strip():
            return answer
        budget *= 2
    return answer


def e2e_phase(max_tokens):
    """One real question end-to-end; returns (ok, detail)."""
    from .pipeline import load_pipeline
    print(f"\n== E2E question (max {max_tokens} tokens) ==")
    print(f"  {QUESTION}")
    pipe = load_pipeline(audit=None)

    t0 = time.monotonic()
    variants = (pipe.abbr_map.variants(DISEASE)
                if pipe.abbr_map is not None else [DISEASE])
    cards = pipe.index.search(variants, config.EMBED, top_k=config.TOP_K)
    if not cards:
        return False, "index search returned no candidates"
    expected = cards[0]
    print(f"  expected guideline (top-1, deterministic search): "
          f"{expected['code_version']} «{expected['name']}» "
          f"score={expected['score']}")
    if NAME_STEM not in (expected["name"] or "").lower():
        return False, (f"top-1 search hit for «{DISEASE}» is unexpected: "
                       f"«{expected['name']}» — index or embed model regressed")

    try:
        answer = answer_with_retry(pipe, QUESTION, max_tokens, progress=True)
    except llm.LLMError as e:
        return False, f"answerer failed: {e}"
    if not answer.strip():
        return False, "answerer returned an empty answer twice"
    print(f"  answer: {len(answer)} chars in {time.monotonic() - t0:.0f}s")

    t1 = time.monotonic()
    result = pipe.run_appendix(QUESTION, answer)
    print(f"  appendix: status={result.status} in {time.monotonic() - t1:.0f}s")
    picks = [p for g in result.by_disease for p in g["picks"]]
    picked = {p["code_version"] for p in picks}
    if result.status != "ok":
        return False, f"appendix status={result.status}, notes={result.notes}"
    if expected["code_version"] not in picked:
        got = ", ".join(sorted(picked)) or "(none)"
        return False, (f"expected guideline {expected['code_version']} "
                       f"not in picks: {got}")
    print(f"  picks: {sorted(picked)}")
    return True, "ok"


def main(e2e=True, max_tokens=3072):
    ok = health_phase()
    if not ok:
        print("\nCANARY FAIL — fix the failing checks above "
              "(see logs/ollama-*.log for model server errors).")
        return False
    if e2e:
        try:
            e2e_ok, detail = e2e_phase(max_tokens)
        except Exception as e:  # canary must never die with a bare traceback
            e2e_ok, detail = False, f"{type(e).__name__}: {e}"
        print(f"  E2E: {'ok' if e2e_ok else detail}")
        ok = ok and e2e_ok
    print("\nCANARY OK" if ok else "\nCANARY FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

# Workflow architecture — corrected draft (2026-09-27)

Status: **draft for owner review**. Where this document conflicts with README §3,
this document wins. Evaluation/exam logic (question generation, judge metrics,
decision rules) is intentionally **not part of this architecture**; goldens and
data assets are retained but out of scope here.

---

## 1. What the assistant is

A medical reference assistant for **Russian physicians in polyclinics**.

**Audience: all physicians — all specialties, pediatrics included.**
The general physician (терапевт / ВОП) is the *primary* user, because a GP
cannot hold deep knowledge of every narrow field and therefore benefits most
from a broad generalist assistant. Narrow specialists are also served, with the
caveat that a small generalist model serves their fields shallower.

**Two tiers, one pipeline:**

| Tier | Model | Where |
|---|---|---|
| Baseline (today) | `medgemma1.5:4b` Q4_K_M | local Ollama on this laptop |
| Target (future) | corporate `medgemma:27b-text` | internet API, corporate-hosted |

All endpoints are OpenAI-compatible; switching tiers is a `.env` change. The
pipeline below is identical for both tiers.

## 2. Core principle: answer first, anchor after

1. **The answerer is a plain medical model.** It answers the physician's
   question from general medical knowledge, in Russian, in clinical register —
   exactly as such a model always answers. It has **no knowledge that Russian
   clinical guidelines exist**; no guideline text is ever given to it, and its
   answer is **never edited or rewritten** afterwards.
2. **All Russia-specific anchoring is post-hoc.** After the answer exists, a
   second model (the *labeler*) reads it, extracts what disease(s) the answer
   is about, and a retrieval step connects them to official guideline PDFs.
   The result is appended as a **verification appendix** — links for the
   physician to check, not a source the answer came from.
3. **No unsolicited treatment.** The answerer must not volunteer drug names,
   doses, regimens or treatment tactics unless the physician explicitly asks
   for treatment. Treatment is the most country-specific and fastest-aging
   part of medicine; this assistant's value is the general medical knowledge
   around it. When treatment *is* explicitly requested, the answerer uses
   international nonproprietary names (МНН) only.
4. **Failures are surfaced, never retried, never hidden.** Any step that fails
   leaves an honest note; the answer still ships.
5. **The physician is the final check.** Every automated addition is clearly
   separated from what the model said.

## 3. The pipeline

```
physician question
   │
   ▼
[1] ANSWERER (plain medical model, no guideline knowledge)
   │   system prompt: clinical register, Russian, INN-only,
   │   no unsolicited treatment, no invented normative references
   ▼
raw answer  ───────────────────────────────────────┐ (never modified)
   │                                               │
   ▼                                               │
[2] LABELER call 1: "which disease(s) is this      │
    answer about?"  -> JSON list of names           │
   │   (empty list -> «заболевание не найдено» note)│
   ▼                                               │
[3] TOOL  search_guidelines(name)                  │
    pure Python: embed name -> cosine top-5        │
    over the guideline-name index  -> candidates   │
   │   (ids + names only, never URLs)              │
   ▼                                               │
[4] LABELER call 2: given the answer context       │
    + candidates -> pick ids or "none"             │
   │                                               │
   ▼                                               │
[5] RENDERER (pure Python)                         │
    ids -> docs_meta.json -> register PDF URL      │
   │                                               │
   ▼                                               ▼
final answer = raw answer + appendix block (register PDF links only)
```

### [1] Answerer

- Prompt = physician assistant persona + rules: answer in Russian in a
  clinical style (профессиональный медицинский тон); МНН only;   **no treatment content unless explicitly asked**; no citing or summarizing
  specific normative documents/orders/guidelines — if asked about them
  directly, say you have no access to them (never invent them).
- No footnote rule, no self-labelling, no few-shot examples that could leak.
  (The old footnote mode where the answerer tagged itself is dropped: it makes
  the answerer aware of the appendix machinery and contaminates the answer.
  The prompt stays **silent** about footnotes — no negative instruction
  either: the model has no such instinct to suppress, and naming the behavior
  in a negation would only plant it.)
- Today: main Ollama instance, CPU, thinking on, 2048-token budget.

### [2] Labeler — extraction call

- A **separate model instance** (today: same 4B, thinking off, on a second
  Ollama instance on the iGPU; tomorrow: whatever labeler model the tier
  provides).
- Input: question + raw answer. Output: **JSON list of disease names**
  (nominative case), `[]` when the answer is organizational/procedural or
  names no disease.
- Output is schema-constrained (structured decoding), temperature 0, small
  token budget.
- Several diseases may be extracted; each is processed independently
  (multi-disease answers get multi-disease appendices).

### [3] Tool — guideline search

- Pure Python, deterministic, no LLM inside.
- **Index (built offline):** for every active guideline in the register
  metadata (`analysis/docs_meta.json`), embed its `name` **and** every
  `mkb_names` variant (semicolon-separated) as separate vectors → one small
  vector index (~1–2k vectors), bge-m3 via the local embed endpoint.
  **All 757 guidelines are indexed — no age/GP filter** (the assistant serves
  all physicians, pediatrics included).
- **Query:** the extracted name **plus its abbreviation-expansion variants** —
  uppercase tokens looked up in `abbr2term` (4,508 Cyrillic medical
  abbreviations, e.g. «ХСН» → «хроническая сердечная недостаточность»);
  expansions ADD query variants, never replace the original. Every variant is
  embedded, results merged. Cosine similarity, **top-5 candidates** overall.
  Rationale: embeddings tolerate morphology and word order, but not the
  abbreviation↔full-form gap — and Russian medical prose is
  abbreviation-dense (ХСН, ИБС, ОКС, ХОБЛ, СД…).
- Returns compact cards: `code_version`, guideline `name`. **Never URLs, never
  statements** — the model picks from a menu, it cannot fabricate links.

### [4] Labeler — selection call

- Input: the answer context + the candidate cards. The model judges relevance
  ("is this candidate what the answer actually discussed?") and picks
  `code_version` ids, or `none` if no candidate fits.
- This is the step that repairs the classic fuzzy-matcher failure: a
  plausible-but-wrong link. The threshold can't read the answer; this call can.
- Bounded: exactly one selection call, no agentic loop. Per-call no-retry.

### [5] Renderer

- Pure Python: picked ids → `docs_meta.json` → guideline name, version code,
  **official register PDF URL** (`apicr.minzdrav.gov.ru/...GetClinrecPdf&id=…`).
- Appends a Russian appendix block under the untouched raw answer: the
  matched guideline names, each with its version code and register PDF URL.
- Every step of every call is logged to JSONL (question, raw answer, labeler
  outputs, tool results, picks) for auditability.

## 4. Delivery — the chat application

The assistant is delivered as a **chat GUI**: the physician talks to it like
to any chat assistant — types a question, gets an answer. The two models run
in the background as services; the physician never sees Ollama, endpoints,
or pipeline steps.

- **Frontend:** chat interface (web app in the browser) with conversation
  history. The answer streams token-by-token as it is generated; the
  appendix (or its notes) appears beneath the answer once the background
  passes finish — visually separated as a "sources" block, clearly not part
  of the model's reply.
- **Backend:** the orchestrator as a long-running service: session and
  history management, answerer call, then the appendix machinery
  (labeler 1 → tool → labeler 2 → renderer) per answer.
- **Background model services:** answerer (main Ollama instance, CPU),
  labeler (second instance, iGPU), embed endpoint — always-on services,
  nothing started per question.
- **Multi-turn:** each question carries the conversation history to the
  answerer (doctors ask follow-ups). The appendix is computed per reply
  from its **question + answer** — the question carries the disease name
  when the follow-up answer is terse; the selection step filters out
  diseases the answer did not actually discuss. Every assistant message
  gets its own links block.
- **On-prem / scalable:** everything inside the corporate network; the
  future 27B tier swaps the answerer endpoint via config — the GUI and the
  rest of the pipeline are unchanged.

## 5. Failure handling

| Failure | Behavior |
|---|---|
| Answerer call fails / empty | no answer; recorded |
**Any stage that is empty or erroneous stops the pipeline** — no
continuation with partial or faked results; the answer ships with the
honest note.

| Stage | Outcome | Physician-visible result |
|---|---|---|
| Labeler 1 | error (invalid / no output) | error note «сбой распознавания заболевания»; stop |
| Labeler 1 | valid: no disease in the answer | note «Заболевание в ответе не найдено»; stop |
| Tool | error (embed endpoint down) | error note «сбой подбора рекомендаций»; stop |
| Tool + Labeler 2 | valid: no match for the disease | per-disease note «Действующая клиническая рекомендация по заболеванию «X» не найдена» |
| Labeler 2 | error (invalid output) | error note «сбой подбора рекомендаций»; stop |

## 6. Corrections applied in this draft (vs. current implementation)

1. **Answerer is guideline-blind.** Footnote mode / self-labelling dropped;
   the answerer prompt no longer contains any appendix-related rule.
2. **Detection moved fully to the second model** (labeler), as a bounded
   two-call protocol with a retrieval tool in between — replacing
   "labeler names disease → Python decides alone". The selection call adds
   the relevance judgment that pure matching lacks.
3. **Embedding index over `name` + `mkb_names`** replaces the two-stage
   ICD-10 path (mkb2descr fuzzy matching → code proximity ladder). One hop,
   morphology/synonym tolerant, no ICD-10 intermediate.
4. **Model picks ids, Python renders URLs.** The model can never emit a link;
   fabrication of URLs is structurally impossible.
5. **No-unsolicited-treatment rule** added to the answerer prompt (decided
   2026-09-26, previously not in prompts).
6. **ГРЛС drug validator removed** from the flow: appending registered trade
   names is unsolicited treatment content.
7. **Scope: all physicians.** Index covers all 757 guidelines; the adult-GP
   558 filter was a corpus limitation, not a product boundary.
8. **Eval/exam logic removed** from the runtime project (goldens and data
   kept).
9. **Labeler fine-tuning deferred:** first ship prompt-only with
   schema-constrained decoding, measure, then QLoRA only if numbers demand it.
10. **27B migration path:** the "tool" is a JSON roundtrip at 4B and a native
    function call at 27B; index and rendering unchanged.
11. **No statement excerpts in the appendix** — guideline links only. A
    matched physician reads the whole PDF; excerpt-grade extraction also
    failed its quality gate.
12. **Chat GUI delivery (new feature)** — the assistant ships as a real chat
    application for physicians; the two models run in the background as
    services.

## 7. Removed / dormant components

| Component | Disposition |
|---|---|
| Footnote rule (`FOOTNOTE_RULE`, `SYSTEM_GENERAL` variant) | removed from answerer prompt |
| ГРЛС validator (`gpeval/drugs.py`, `phase7_grls.py`) | removed from the flow |
| Appendix statement excerpts (`--with-statements`) | removed — links only; extracted statements stay as data |
| ICD-10 two-stage matcher (`mkb2descr` path, `mkb.proximity`) | superseded by the embedding index |
| BM25 statement index, `gpeval/chunks.py` | dormant (served RAG/eval; rebuild only if fine-tuning returns) |
| Question generation, judge metrics, decision memo, regression bench | out of scope here; goldens kept |
| `abbr2term` | standard part of the tool: abbreviation expansions added as query variants |

## 8. Decisions (owner, 2026-09-27)

1. **Outcome notes per stage; errors never masquerade as results.** Labeler 1:
   valid «no disease» → «Заболевание в ответе не найдено»; error → its own
   error note. Tool + Labeler 2: valid «no match» → per-disease
   «Действующая клиническая рекомендация по заболеванию «X» не найдена»;
   error → its own error note. Any empty or errored stage stops the
   pipeline; the answer ships with the honest note.
2. **Dual-instance layout kept.** Answerer on the main Ollama instance (CPU),
   labeler on the second instance (iGPU) — as today.
3. **Treatment lookup tool deferred.** First get the existing intentions
   working (same gate as labeler fine-tuning); the websearch tool is a later
   add-on, not part of the current build.

# Workflow architecture

This is the authoritative design of med_assistant; `AGENTS.md` is the runbook.
Evaluation/exam logic (question generation, judge metrics, decision rules) is
intentionally **not part of this architecture**; goldens and data assets are
retained but out of scope here.

---

## 1. What the assistant is

A medical reference assistant for **Russian physicians in polyclinics**.

**Audience: all physicians — all specialties, pediatrics included.**
The general physician (терапевт / ВОП) is the *primary* user, because a GP
cannot hold deep knowledge of every narrow field and therefore benefits most
from a broad generalist assistant. Narrow specialists are also served, with
the caveat that a small generalist model serves their fields shallower.

**Two tiers, one pipeline:**

| Tier | Model | Where |
|---|---|---|
| Baseline (today) | `medgemma1.5-4b-q4km-text-only` Q4_K_M | local llama-server (llama.cpp Vulkan build, pinned in `deploy/llamacpp-pin.txt`) |
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
   ▼                                               ▼
[5] RENDERER (pure Python)
     ids -> docs_meta.json -> local PDF file
     (data/clinrec_minzdrav/, served at /data/pdf)
    │                                               │
    ▼                                               ▼
final answer = raw answer + appendix block (local PDF links only)
```

### [1] Answerer

- Prompt = physician assistant persona + rules: answer in Russian in a
  clinical style (профессиональный медицинский тон); МНН only; **no treatment
  content unless explicitly asked**; no citing or summarizing specific
  normative documents/orders/guidelines — if asked about them directly, say
  you have no access to them (never invent them).
  Do NOT add an "answer in МНН when the physician names a trade name"
  clause: at 4B it becomes a format target the model satisfies by
  fabricating a plausible-looking INN («Сорбифер Дурулес» → «железо(II)
  глюконат», 2026-10-11). Anti-fabrication rules don't fix it either:
  «do not invent figures/substance names» wording (tested 2026-10-11..12)
  still yielded fabricated INNs («Феррум гликонат», «Метилурацилат») and
  was reverted. The plain МНН-only rule answers such questions correctly
  in most runs but still fabricates occasionally (2026-10-11) — residual
  4B flakiness, documented lever: 27B tier via `CANDIDATE_*`.
  **Register-grounded МНН lookup** (owner decision 2026-10-12; matching v2
  2026-10-13): the question is scanned for register trade names
  (`data/drugs.sqlite`, built by `scripts/grls_to_sqlite.py` from a ГРЛС
   export); each usable match gets a «справочно» note in the user
   message (`prompts.drug_block` — BEFORE the question, then the card
   block; owner decision 2026-10-17, superseding the 2026-10-12
   after-question position: the 4B's thinking paraphrases the question
   into English on its first lines and phonetically re-identifies
   «валидол» as valium there, so diazepam pharmacology leaked into the
   answer under a correct МНН label; with the anchor in front of the
   question the binding wins — 3/3 clean live runs vs 0/2 after)
  and a `.sources.drug` block under the question bubble in the UI
  (`drug_map` SSE event live; persisted on the user row as
  `messages.drug_map`, restored on reload). This coexists with the
  МНН-only rule: it removes the knowledge gap that caused the fabrications,
  not the rule. A missing DB silently disables the feature. Matching v2
  (owner's dictionary scheme, replaces the v1 stem+endings scan):
  names keyed by first word (last letter stripped, ≥3 chars; leading
  stereo segments «L-», «Д-», «D,L-» dropped, digit-led names skipped;
   placeholder-МНН `~` rows not keyed); question tokens ≥5 chars (minus a
   stoplist) match the longest key they start with, provided the token is
   a real inflected form of the key's original first word — the word
   itself, word+ending, or a vowel/ь/й-final word (or an adjective
   ый/ой/ий stem) minus that tail + ending, endings from a closed set in
   `drugmap.py` (owner decision 2026-10-17, tightening the word+≤3 length
   cap of 2026-10-16: unbounded prefixes had grounded МАК-ПАС via
   «максимальной» and ЭФФЕКС Силденафил via «эффект» in one
   neuropathic-pain question, then «анестетик» = «анеста»+3 and
   «азотемия» = «азота»+3 slipped through exactly at the cap boundary —
   non-ending tails are different words, while real forms «нурофенами»,
   «омеза», «грудного» survive); an ambiguous key is
   restricted by the adjacent bigram («нурофен плюс» → ибупрофен+кодеин)
   or falls back to the modal МНН (most rows, tie → the МНН matching the
   names' first word; «нурофен» → ибупрофен). When nothing matched:
   curated aliases (`data/drug_aliases.yaml` — owner-curated truth, value =
   register trade name or a direct МНН for register holes like кагоцел →
   меглюмина акридонацетат; a recurring doctor typo gets a curated alias
   the same way). The Levenshtein typo tier (2026-10-13..17) was removed
   (owner decision 2026-10-17): across the whole live history it caught
   zero real typos and produced one documented false positive
   («перкуссии»→Пертуссин, 2026-10-15), while dist-1 kept admitting
   common words («панкреатит»→Панкреатин, «артерии»→Артерис-веро,
   «тиреоидит»→Тиреоидин). A shared stoplist suppresses disease-stem
   collisions in the dict and alias tiers (owner decision 2026-10-15:
   brand names are built on disease stems — «артроза»→Артрозан,
   «витамины»→Витамин D3, «бронхите»→Бронхо-мунал, «диабетом»→Диабетон
   МВ, «подагра»→ПОДАГРУМ all fired live; exact tokens only, never
   prefixes — disease and drug inflections are distinct tokens, a
   «бронхит*» prefix would block the real drug Бронхитол; the
   эффект*/артрит*/азотеми* families and the token «анестетик» joined it
   2026-10-16 — all later killed structurally by the 2026-10-17 endings
   rule, the entries stay as belt-and-braces; «азота» itself remains an
   accepted residual: it IS the keyed word of Азота закись). The remaining
   post-rule residue for owner curation is morphological homonyms in the
   dict tier («кроме», «проток», «статин», «фактор*», «жидкость*»,
   «колика»…). Matches are
  deduped and capped (`DRUG_PS_MAX`). Every P.S. line anchors the
  doctor's own word to the register row (`token`/`via`) — without the
  anchor the 4B re-identifies the drug from its own prior (валидол →
  "Valium" → диазепам, A/B-verified 2026-10-13); МНН lines are bilingual:
  each substance in Russian with its English in parentheses right after
  it when one exists («железа сульфат (ferrous sulfate) + аскорбиновая
  кислота (ascorbic acid)», owner decision 2026-10-14) — the model
  reasons in English but must answer in Russian, so it needs both. The
  rendered string is PRE-TRANSLATED in the register itself: the
  `inn_pair` column of `data/drugs.sqlite` (plus `inn_en`/`inn_lat`),
  written by `scripts/inn_to_en.py` from the WHO recommended INN list
  (`data/inn_list_en-lat.txt`, paired en↔latin; RU МНН convert as
  Cyrillic renderings of the Latin forms — exact whole-name rules, acid
  stem matching, and word-level composition from WHO-derived word
  alignments; the distance-1 tier is never written: a missing gloss
  costs nothing, a wrong gloss is the hallucination itself). The curated
  `data/inn_gloss.yaml` (owner truth for WHO holes like ментилизовалерат
  or фармакопейные соли) folds in at build time; re-run the script after
  editing it or after a WHO list refresh.
- No footnote rule, no self-labelling, no few-shot examples that could leak.
  The prompt stays **silent** about footnotes — no negative instruction
  either: the model has no such instinct to suppress, and naming the behavior
  in a negation would only plant it.
- Today: main Ollama instance, CPU, thinking on, 3072-token budget
  (raised from 2048 on 2026-10-16: num_predict — not num_ctx — caps output,
  and a verbose thinking phase ate the budget, truncating the answer
  mid-word at exactly 2048 generated tokens).
- **Thinking-loop watchdog** (owner decision 2026-10-06; detector
  redefined 2026-10-14): the reasoning stream occasionally collapses
  into a tail-spin — the same unit repeated back-to-back — that burns
  the whole budget (minutes of CPU, blocked composer, empty turn).
  While no answer token has arrived yet, `llm.chat_stream` feeds every
  reasoning delta into a `LoopGuard`; when the rolling
  `LOOP_WINDOW_CHARS` buffer *ends* with a unit of ≥ `LOOP_MIN_UNIT`
  chars repeated `LOOP_REPEATS` times **consecutively**, the request
  is aborted (`LLMLoopError` raised out of the stream context —
  closing the connection cancels Ollama generation and frees the CPU).
  Scattered recurring phrases never trip: the first detector counted
  occurrences of the tail anywhere in the window and aborted a healthy
  stream that was merely enumerating the card's example answers
  (different questions sharing one long suffix — owner report
  2026-10-14). The pipeline then retries once with an anti-repetition
  nudge (`LOOP_RETRY_FREQ_PENALTY`), emitting `reasoning_reset` so the
  UI drops the looped garbage and shows a visible
  «Зацикливание размышлений — …» note; a second loop or an empty retry
  fails with the honest «Сбой генерации ответа» note (its looped
  garbage is dropped with a note too). Not a semantic retry: the prompt
  and content are untouched, only the sampler is nudged on a turn that
  was already lost. Loops after the first answer token are not guarded
  (aborting there would duplicate streamed text); `num_predict` bounds
  them as before.

### [2] Labeler — extraction call

- A **separate model instance** (today: same 4B, thinking off, on its own
  llama-server process with the think-suppressing `labeler.jinja`; tomorrow:
  whatever labeler model the tier provides).
- Input: question + raw answer. Output: **JSON list of disease names**
  (nominative case), `[]` when neither names a disease.
- Sources: question + answer, both (owner decision 2026-10-20, widening the
  MVP answer-only anchor): answer-discussed diseases come FIRST (their
  extraction must not lose `DISEASE_CAP=3` slots), question-named extras are
  appended after. A «Да.» answer to «А при инсульте?» anchors the stroke
  guideline (bare confirmations extracted `[]` before this change and the
  guideline was missed); a drifted answer can no longer silently drop a
  disease the physician asked about. Question-only noise stays bounded by
  the symptom/complaint filter and the select stage (which still judges
  candidates against the answer).
- Earlier the same day the extraction was strictly answer-anchored with a
  bare-answer fallback — superseded by the question+answer widening above.
- Output is schema-constrained (structured decoding), temperature 0, small
  token budget.
- Several diseases may be extracted; each is processed independently
  (multi-disease answers get multi-disease appendices).

### [3] Tool — guideline search

- Pure Python, deterministic, no LLM inside.
- **Index (built offline):** for every active guideline in the register
  metadata (`data/docs_meta.json`), embed its `name` **and** every
  `mkb_names` variant (semicolon-separated) as separate vectors → one small
  vector index (~1–2k vectors), bge-m3 via the local embed endpoint.
  **All 757 guidelines are indexed — no age/GP filter** (the assistant serves
  all physicians, pediatrics included).
- **Query:** the extracted name **plus its abbreviation-expansion variants** —
   uppercase tokens looked up in `abbr2term` (4,506 Cyrillic medical
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
- This is the step that adds the relevance judgment a similarity threshold
  cannot make: the threshold can't read the answer; this call can.
- Bounded: exactly one selection call, no agentic loop. Per-call no-retry.
- Reasoning inside the grammar JSON (owner decision 2026-10-20, root-cause
  fix for empty picks): llama.cpp grammar mode constrains generation from the
  first token, so the selector cannot re-open `think` before the JSON like
  the old Ollama transport did — without inference steps it collapsed to
  `{"picks": []}` on ambiguous candidates. The per-call schema now requires
  a `"reasoning"` string property emitted BEFORE `"picks"` (GBNF emits
  required properties in dict order): the model writes a brief per-candidate
  verdict, then decides. Chain-of-thought and guaranteed-valid JSON at once.
  The reasoning string is grammar-bounded (`maxLength` 1400 via the
  `SELECT_REASONING_MAXLEN` constant — an unbounded string can absorb the
  whole token budget: a degenerate run repeated itself to truncation and
  left the object unclosed, live probe 2026-10-20; and the first cap, 700,
  could not mathematically fit a 5-candidate enumeration (~140–173 chars
  per verdict item, five ≈ 730+): the clamp slammed shut mid-list and
  shipped picks contradicting the lost verdicts, live 2026-10-20. At the
  cap the grammar closes the string and the JSON still parses; `len ==`
  the cap is greppable in audit as `reasoning_truncated`. This pinned
  llama.cpp parser refuses GBNF repeats over 1000 (700 OK, 1001 FAIL,
  bisected max accepted 1000, live 2026-10-20), so the cap is emitted as
  two adjacent bounded repeats `{0,1000}` + `{0,400}` summing to it
  (`schema_to_grammar`, `_GBNF_MAX_REPEAT`). Verdicts are
  terse by design — «подходит или не подходит», no explanations — so a
  real enumeration lands ~500 chars, far under the cap. The remaining
  echo fuel was the prompt itself: a `matched` term shared by several
  candidates is non-discriminative and the 4B echoes it verbatim
  (2026-10-21, «Синдром «гистаминовой» головной боли» on 3 of 5 lines →
  5× echo in the reasoning; live probes: 23× without the grammar — and
  unparseable, ~50× as a think-loop to the token cap under the answerer
  template with genuinely good per-candidate reasoning before the spiral,
  204 s empty — which killed the "select via the thinking answerer" idea:
  the repetition is prompt-fueled, not transport-fueled). `_select_user`
  therefore prints the «совпавший термин» parenthetical only when the
  term annotates a single candidate — unique matches keep the 2026-10-16
  sound-alike protection, shared ones disappear from every line. The
  register's audience structure became visible the same day (owner catch
  2026-10-21): same-NAME pairs are ADULT/PEDIATRIC versions of the same
  recommendation (810_1 «Головная боль напряжения» = Дети, 162_3 =
  Взрослые; 24 such pairs, zero same-name-same-audience docs) and their
  titles carry no age cue — the select prompt's rule 4 was inapplicable,
  identical-looking lines fed the echo, and the blind echo shipped the
  CHILDREN'S guideline to adult questions unnoticed. iter_appendix now
  enriches each card with docs_meta `age`; the candidate lines render
  «(дети)» / «(взрослые)» / «(взрослые и дети)» markers, the
  true-duplicate collapse keys on (title, audience) — merging nothing in
  the current register, armed for future same-audience refreshes — and
  the appendix marks pediatric picks «, дети» (resolve_picks carries
  `age` on the pick). The surviving degeneration is caught post-hoc by
  the thinking-loop watchdog family (2026-10-21):
  `_select_reasoning_loop` reuses `loopguard.LoopGuard` on the parsed
  reasoning (verbatim tail-spins) and adds a number-blind item-block
  cycle check — the same bodies under «1.», «2.»… break LoopGuard's
  back-to-back check, and the live shapes were a plain echo (same item
  x5), a cap-clamped truncated echo, and a full cyclic re-enumeration
  ([A,B,C]x4); the item tail must not end with >= 3 back-to-back copies
  of the same p-item block (p up to 8, a partial chopped tail block is
  discarded; 3 blocks, not 2 — an honest terse double-pick repeats one
  body twice). A looped reasoning judged nothing: its picks are dropped
  (`labeler_select_loop` audit row, no retry), the title guard is the
  deterministic net.
  `frequency_penalty` on the select call was probed live and REJECTED
  (2026-10-21): it did not prevent loops deterministically and degraded
  healthy runs — a hallucinated candidate id at 0.3, medically wrong
  synonym claims at 0.5. Selection itself became ONE CALL PER TITLE
  GROUP (owner decision 2026-10-21 evening, 27B pivoted away from —
  4B-only on an ordinary laptop): the 4B cannot enumerate a 5-candidate
  mixed list without degenerating, but judges 1-2 items reliably, so
  each recommendation title gets its own short call and the
  same-NAME adult/pediatric pair shares ONE call (owner constraint —
  the audience choice is comparative, splitting the pair would hide
  the twin); picks are unioned, per-call picks filtered to the group,
  a looped group kills only its own picks (`group=[ids]` on the audit
  rows). Honest open state (live 3/3, sharpened rules 3/4): per-group
  calls eliminated the loops but the 4B RUBBER-STAMPS — every
  candidate «подходит», детские included right after quoting rule 4,
  unrelated docs with self-contradicting reasoning (live canary:
  6 picks). Owner decisions 2026-10-21 evening: over-attachment is
  tolerated when RARE, a symptom query must attach its guideline pdf,
  and an UNCLEAR audience attaches BOTH versions of a pair — the
  doctor chooses himself (`_audience_complete` completes a half-picked
  pair when question+answer carry zero audience wording, best-scored
  twin, `select_audience_complete` audit row; any «ребён/детск/
  взросл…» marker stands the guard down — rule 4 territory). With that
  the loop-canary row («ГБН» present, no select loop) is GREEN live.
  Retrieval scores are NOT sent to the labeler (owner asked,
  rejected: the live pair's TOP score belonged to the детские doc —
  score-following would amplify wrong-audience picks; rank is already
  implicit in candidate order; the failures were judgment failures
  scores cannot encode) and
  `LABELER_SELECT_MAX_TOKENS` is 128 → 480 → 768 → 1024 (768 assumed
  prose density ~2.3 chars/token; a live degenerate short-item echo
  «5. 154_4 — подходит.»×N runs 1.75 chars/token — 1400 chars ≈ 800+
  tokens and the 768 cap chopped the string at 1347 chars before the
  grammar clamp could close it, unclosed object, needless select_error
  (2026-10-21 canary); 1024 keeps the char clamp authoritative; only
  degenerate runs pay the full cap, healthy enumerations stop early.
  Select latency grows to ~40 s worst case on CPU (~90 s for a full-cap
  echo, dropped by the loop guard) — accepted, laptop-only stack). The numbered-skeleton
  candidate tail in the
  user prompt is load-bearing (probed live: instruction-only tails reasoned
  right but still emitted `picks: []`; prose example values echoed verbatim).
  The parser stays lenient — picks-only output still parses
  (test fakes, CLI answer files).
- Title guard (owner decision 2026-10-19, kept as a safety net): after
  selection, a candidate whose guideline title contains the extracted disease
  verbatim is restored if the selector dropped it — but only when exactly one
  candidate matches (a generic «инфаркт» keeps «Острый инфаркт миокарда…»;
  adult/pediatric duplicates stay with the selector). With the
  reasoning-first fix the guard mainly nets grammar-legal empty-string
  reasoning on verbatim-title cases.

### [5] Renderer

- Pure Python: picked ids → `docs_meta.json` → the local guideline PDF
  (`data/clinrec_minzdrav/`, one file per register entry). The chat server
  serves the folder at `/data/pdf`; the appendix links there and opens the
  PDF in a browser tab.
- Guideline display names come from the PDF file name (textual part after the
  version-code prefix) — the on-disk document title, which differs from the
  register metadata name for ~114/757 entries. The CLI/audit appendix text
  prints the absolute file path instead of the URL. Picks whose PDF is missing
  on disk are dropped (per-disease «не найдено» note), never rendered as dead
  links; a missing corpus folder is a clear server startup failure.
- Appends a Russian appendix block under the untouched raw answer: the
  matched guideline names, each with its version code and local PDF link.
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
- **Background model services:** answerer (llama-server :8081, `num_ctx 12288`,
  thinking ON), labeler (llama-server :8082, same GGUF with the think-suppressing
  `labeler.jinja`), embed endpoint (llama-server :8083, bge-m3 F16,
  `--embeddings --pooling cls`) — always-on services started by
  `python -m medassistant.launcher` (or `Start.bat` on the target machine),
  nothing started per question. GPU policy: dGPU-only with system-default layer
  offload; iGPU-only or no GPU → `-ngl 0` pure CPU (`--force-cpu`/`--force-gpu`
  overrides).
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

**Any stage that is empty or erroneous stops the pipeline** — no continuation
with partial or faked results; the answer ships with the honest note.

| Stage | Outcome | Physician-visible result |
|---|---|---|
| Answerer | call fails / empty | no answer; recorded |
| Labeler 1 | error (invalid / no output) | error note «сбой распознавания заболевания»; stop |
| Labeler 1 | valid: no disease in the answer | note «Заболеваний в ответе не найдено»; stop |
| Tool | error (embed endpoint down) | error note «сбой подбора рекомендаций»; stop |
| Tool + Labeler 2 | valid: no match for the disease | per-disease note «Действующая клиническая рекомендация по заболеванию «X» не найдена» |
| Labeler 2 | error (invalid output) | error note «сбой подбора рекомендаций»; stop |

## 6. UX and the auto-pick fast path

1. **Thinking is visible.** The answerer's reasoning streams into a
   collapsible «Размышления модели» block above the answer; it is open while
   thinking, auto-collapses when the answer text starts, and stays open if
   the physician reopened it manually. Not persisted with the message.
2. **Appendix progress is visible and non-blocking.** Each machinery stage
   (extract → search → auto-pick/select) streams a status line under the
   answer. The composer unblocks as soon as the answer text is complete —
   the next question may be typed while the appendix finishes.
3. **Provisional candidate cards carry no retrieval score.** On labeler-2
   turns the streamed cards are name+link only — embedding cosine is a name
   match strength, not a probability of correctness, and showing it as a %
   read as misleading (owner decision 2026-10-04). Scores are internal to
   auto-pick: stored appendix payloads, the labeler-2 prompt, and the SSE
   candidate cards all stay score-free.
4. **Auto-pick fast path.** When the top search candidate scores ≥ 0.80 and
   leads the runner-up by ≥ 0.05 (or is the only candidate), labeler call 2
   is skipped and that candidate is taken (marked «автоподбор» in the
   appendix, `auto: true` in the audit log). Tunable: `AUTO_PICK`,
   `AUTO_PICK_MIN`, `AUTO_PICK_GAP` in `.env`. Known tradeoff: near-duplicate
   guideline versions (adult vs pediatric) may be auto-picked without the
   relevance check; the physician remains the final check — and since §7.1
   the runner-ups are one click away in the appendix.
5. **The answerer can be stopped — undo send** (owner decision 2026-10-15).
   While the answerer generates, the send button is a Stop control
   («Стоп»; Esc works too); under the label a small elapsed-time line ticks
   from the send (seconds, minutes after the first minute) and lingers 5 s
   over «Отправить» fading out (owner request 2026-10-17). A stop drops the
   HTTP connection to Ollama —
   generation is cancelled at once, even in the silent prefill window right
   after the send, which is exactly when a wrong question is noticed — and
   undoes the turn: the question row and the partial thinking/answer are
   deleted, the question returns to the composer for editing (a live
   analogue of the ↩ rewind). `POST /api/sessions/{id}/stop` cancels the
   session's in-flight answerer turns (audit `user_stop`; SSE `stopped`);
   the streaming thread alone deletes the rows, so the cleanup is safe even
   when the tab closes right after the click. Stop is answerer-phase only:
   once the answer text completes, the appendix machinery always runs to
   completion, and pdf turns (whose answerer phase is empty) are not
   stoppable. A cancelled generation is never retried (`llm.LLMCancelled`
   bypasses the loop-retry and the transport retry).

## 7. Filtered-out candidates — selection is not destructive

1. **Flat filtered-out list in the payload.** Labeler 2's selection is
   fallible (a wrong "none" verdict used to make all candidates vanish).
   `by_disease` groups carry only the verified picks; every resolved
   candidate that no disease verified lands in a flat top-level
   `filtered_out` list (deduped by id across diseases, each entry keeping
   the disease whose search produced it) — including auto-pick runner-ups
   (the §6.4 recovery path). Persisted in the sessions DB with the payload.
2. **UI: verified list + one collapsible.** The sources block shows the
   verified links (all diseases) by default; ONE «Показать отфильтрованные
   (N)» toggle for the whole block reveals the filtered-out candidates
   (muted styling, disease attribution kept). A single filtered-out entry
   is shown directly, without a toggle. The plain-text appendix
   (CLI/audit) remains picks-only.
3. **No-match notes dropped.** «Pdf по «X» не найдено» no longer fires when
   candidates exist but none were verified — the filtered-out list is the
   honest presentation. Legacy stored notes are filtered out in the UI.
   True error notes (extract/search/select failures) are unchanged.

## 8. Manual pdf queries

Labeler 1 sometimes extracts an incomplete disease list, so a discussed
guideline never appears. The physician can run the appendix machinery by
hand from the same composer:

1. **Prefix routing.** A message whose first whitespace-separated token is
   `pdf` or `пдф` (case-insensitive) is a *pdf turn*: the answerer is not
   called. The prefix must be a separate token («pdfхсн» is a normal
   question); a bare «pdf» is rejected with a validation error.
2. **Labeler 1 in direct mode parses the payload:** the doctor's free text
   goes through a dedicated extract prompt that normalizes names to the
   nominative case and decides whether one or several diseases are named —
   the same schema-constrained call, fallible like any labeler call, with
   honest notes on failure («Заболеваний в запросе не найдено» when it
   names none).
3. **Then the standard machinery unchanged:** search per disease (with
   abbreviation expansion), auto-pick fast path or labeler 2 selection,
   renderer with the filtered-out list. A pdf turn is input-only (owner
   decision 2026-10-04, superseding the earlier last-answer context): the
   selection call sees the payload alone, with a dedicated select prompt
   that has no «ответ ассистента» section.
4. **A pdf turn is a new chat turn, rendered as ONE row.** The user bubble
   is retitled «Врач (подбор pdf)» and shows the stripped payload
   («а что про инфаркт»); the sources block (status line while running)
   attaches directly below it. Under the hood it is still stored as a
   user row + an assistant caption row (`kind='pdf'`); the UI merges the
   pair. Previous answers are never mutated.
 5. **Pdf turns never enter the answerer history** (`messages.kind='pdf'`,
    excluded from the history window) — the command syntax must not leak
    into medical answers.
 6. **CLI parity:** `python -m medassistant ask --question "pdf ХСН"` runs
    the same manual path.

## 9. Patient card context (example-only)

Demo feature (owner decision 2026-10-06): the physician can attach a
hand-made patient chart («амбулаторная карта») so the answerer answers
*this* question in the context of *this* patient.

1. **Cards are hand-made text files** in `data/patient_cards/`
   (`CARD_DIR`): the attachable chart `{ФИО}.амбулаторная_карта.md`
   itself (Cyrillic OK; display name = filename minus the
   `.амбулаторная_карта.md` suffix, underscores → spaces) — owner
   decision 2026-10-14, superseding the 2026-10-09 two-file convention
   (attachable `{ФИО}.выписка.md` extract + non-attachable source chart):
   the `*.выписка.md` extracts are deleted, the full chart IS the card.
   The shipped charts are 4.8–6.6k chars — inside `CARD_MAX_CHARS`, so no
   summarization tier is needed on the 4B model (a textual extract is the
   only "preencoding" a chat-completions decoder accepts; no vectors
   involved). Five fake charts ship (Волков_ДС, Иванова_МП, Петрова_АС,
   Смирнова_ЕВ, Соколов_АВ). `GET /api/cards` lists them (a missing dir
   lists nothing). There is still **no card-adding mechanism and no
   summarizer**.
2. **Per-question opt-in, invisible session pin.** The composer has ONE
   toggle button (owner decision 2026-10-07, superseding the earlier
   chip + checkbox and +/× sub-button variants): OFF — «Карта пациента»,
   click opens the picker; ON — shows the patient name, click detaches
   it. While ON, every non-pdf question carries the card. The *visible*
   state is per chat session: switching chats (or reloading) resets the
   button to OFF — but the session invisibly remembers its card
   (`sessions.card_file`, filename only; pinned the moment a card is
   PICKED in the UI — `POST /api/sessions/{id}/card` — and re-pinned when
   a question is sent with it; a pin whose file vanished is dropped).
   `GET /api/sessions/{id}/card` returns it for the picker preselection:
   highlighted and placed at the BOTTOM of the list — nearest the click
   point, since the menu opens upward. **No per-message markers**: the
   stored user message stays the plain question; the card text exists
   only inside the answerer LLM call and never enters history, SSE
   events, or the appendix inputs (`iter_appendix` sees the plain
   question; labeler calls are unchanged).
3. **Prompt injection is one place:** `Pipeline._messages` appends
   `prompts.card_block(ФИО, text)` to the user message — a header line
   («Амбулаторная карта пациента {ФИО} (приложена врачом; учитывай её при
   ответе):»), the file text verbatim, and a closing separator
   («конец карты»). The model never sees the filename; `SYSTEM_ANSWERER`
   is untouched — the answerer stays a plain medical model, the card is
   user-supplied context.
4. **Validation (shared by server and CLI in `cards.py`), fresh read every
   request:** the doctor may edit the file mid-session. `POST /api/chat`
   gains optional `card` (filename): traversal-safe name inside `CARD_DIR`
    (`Path(...).name` + parent check), mandatory `.амбулаторная_карта.md`
     suffix,
    existence, size ≤ `CARD_MAX_CHARS` (default 8000 ≈ 2k tokens; the model
    context is 131k, runtime `num_ctx` is 12288 via the Modelfile — 3x the
    Ollama 4096 default, which silently truncated the prompt head (system
    persona + history) on card turns; owner decision 2026-10-09). Any
    violation → HTTP 400 with a Russian message **before** the question
    is sent.
5. **Pdf turns ignore the card entirely** (§8): the frontend never sends
   it for `pdf`/`пдф`-prefixed messages and the backend skips validation
   on that path.
6. **Audit:** `card_used` (file, chars) per question that used one,
   `card_rejected` on validation failure — filename and length only, card
   content is never logged (physician data stays out of the audit trail).
7. **CLI parity:** `python -m medassistant ask --question "…" --card
    Иванова_МП.амбулаторная_карта.md` (a filename in CARD_DIR or a direct
    path); the stored message (`--save`) is the plain question.

Out of scope (future): card upload/creation from the UI; original files
(pdf/doc/db) with automatic summary regeneration; per-message card markers
in sessions.sqlite; RAG over the card; OCR.

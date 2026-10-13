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
  export); each usable match gets a «справочно» P.S. appended to the user
  message (`prompts.drug_block`, after the question, before the card block)
  and a `.sources.drug` block under the question bubble in the UI
  (`drug_map` SSE event live; persisted on the user row as
  `messages.drug_map`, restored on reload). This coexists with the
  МНН-only rule: it removes the knowledge gap that caused the fabrications,
  not the rule. A missing DB silently disables the feature. Matching v2
  (owner's dictionary scheme, replaces the v1 stem+endings scan):
  names keyed by first word (last letter stripped, ≥3 chars; leading
  stereo segments «L-», «Д-», «D,L-» dropped, digit-led names skipped;
  placeholder-МНН `~` rows not keyed); question tokens ≥5 chars (minus a
  stoplist) match the longest key they start with; an ambiguous key is
  restricted by the adjacent bigram («нурофен плюс» → ибупрофен+кодеин)
  or falls back to the modal МНН (most rows, tie → the МНН matching the
  names' first word; «нурофен» → ибупрофен). When nothing matched:
  curated aliases (`data/drug_aliases.yaml` — owner-curated truth, value =
  register trade name or a direct МНН for register holes like кагоцел →
  меглюмина акридонацетат), then a tight typo tier (Levenshtein ≤2, same
  first 3 chars, token ≥7 chars, unambiguous keys only). Matches are
  deduped and capped (`DRUG_PS_MAX`). Every P.S. line anchors the
  doctor's own word to the register row (`token`/`via`) — without the
  anchor the 4B re-identifies the drug from its own prior (валидол →
  "Valium" → диазепам, A/B-verified 2026-10-13); composite/obscure МНН
  carry a curated English gloss (`data/inn_gloss.yaml`, owner truth) —
  the model reasons in English and substitutes a plausible compound when
  it cannot parse a rare token (ментилизовалерат → "menthylsalicylate",
  2026-10-13).
- No footnote rule, no self-labelling, no few-shot examples that could leak.
  The prompt stays **silent** about footnotes — no negative instruction
  either: the model has no such instinct to suppress, and naming the behavior
  in a negation would only plant it.
- Today: main Ollama instance, CPU, thinking on, 2048-token budget.
- **Thinking-loop watchdog** (owner decision 2026-10-06): the reasoning
  stream occasionally collapses into a verbatim repetition loop that burns
  the whole budget (minutes of CPU, blocked composer, empty turn). While
  no answer token has arrived yet, `llm.chat_stream` feeds every reasoning
  delta into a `LoopGuard`; when the last `LOOP_TAIL_CHARS` chars repeat
  `LOOP_REPEATS` times inside a `LOOP_WINDOW_CHARS` rolling buffer, the
  request is aborted (`LLMLoopError` raised out of the stream context —
  closing the connection cancels Ollama generation and frees the CPU).
  The pipeline then retries once with an anti-repetition nudge
  (`LOOP_RETRY_FREQ_PENALTY`), emitting `reasoning_reset` so the UI drops
  the looped garbage; a second loop or an empty retry fails with the
  honest «Сбой генерации ответа» note. Not a semantic retry: the prompt
  and content are untouched, only the sampler is nudged on a turn that
  was already lost. Loops after the first answer token are not guarded
  (aborting there would duplicate streamed text); `num_predict` bounds
  them as before.

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
pre-written patient-history summary («выписка из амбулаторной карты») so the
answerer answers *this* question in the context of *this* patient.

1. **Cards are hand-made text files** in `data/patient_cards/`
   (`CARD_DIR`), as two-file demo pairs per patient (owner decision
   2026-10-09): the attachable card `{ФИО}.выписка.md` (Cyrillic OK;
   display name = filename minus the `.выписка.md` suffix, underscores →
   spaces) plus its source chart `{ФИО}.амбулаторная_карта.md` — the
   "original" every extract fact must be derivable from, never attachable
   on the 4B tier: the summary file IS the artifact (a textual extract is
   the only "preencoding" a chat-completions decoder accepts; no vectors
   involved; full-chart attachment is deferred to the 27B tier — see the
   out-of-scope list). Five fake pairs ship (Волков_ДС, Иванова_МП,
   Петрова_АС, Смирнова_ЕВ, Соколов_АВ); the generator prompt
   `data/patient_cards/PROMPT_browser_agent_EMK.md` describes the pair
   convention. `GET /api/cards` lists the extracts only (a missing dir
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
   («Выписка из амбулаторной карты пациента {ФИО} (приложена врачом;
   учитывай её при ответе):»), the file text verbatim, and a closing
   separator. The model never sees the filename; `SYSTEM_ANSWERER` is
   untouched — the answerer stays a plain medical model, the card is
   user-supplied context.
4. **Validation (shared by server and CLI in `cards.py`), fresh read every
   request:** the doctor may edit the file mid-session. `POST /api/chat`
   gains optional `card` (filename): traversal-safe name inside `CARD_DIR`
   (`Path(...).name` + parent check), mandatory `.выписка.md` suffix,
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
   Иванова_МП.выписка.md` (a filename in CARD_DIR or a direct path); the
   stored message (`--save`) is the plain question.

Out of scope (future): attaching the full амбулаторная карта
(`*.амбулаторная_карта.md` as the card — planned for the 27B tier, owner
decision 2026-10-09); card upload/creation from the UI; original files
(pdf/doc/db) with automatic summary regeneration; per-message card markers
in sessions.sqlite; RAG over the card; OCR.

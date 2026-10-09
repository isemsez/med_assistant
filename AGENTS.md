# med_assistant

Medical reference assistant for Russian physicians — answer-first pipeline with a
post-hoc guideline verification appendix. The authoritative design is
[`.kilo/architecture.md`](.kilo/architecture.md); this README is the runbook.

Pipeline: ANSWERER (plain medical model, guideline-blind) → LABELER call 1
(diseases as JSON) → TOOL `search_guidelines` (embedding index, top-5 cards, no
URLs) → LABELER call 2 (pick ids or none) → RENDERER (register PDF links under
the untouched answer). Failures surface as honest notes; nothing is retried at
the semantic level. Delivered as a chat web app.

UX (architecture.md §6): the answerer's reasoning
streams into a collapsible «Размышления модели» block (auto-collapses when the
answer starts); the appendix machinery shows a live stage line («Поиск болезней
в тексте…» → «Ищу соответствующие pdf.» → …) and the composer unblocks
as soon as the answer text is done; when the labeler filter runs, the candidate
pdf cards stream to the page immediately after search (search is cheap, the
labeler call is not) and are filtered to the verified picks client-side when
the appendix payload arrives (cards are name+link only — retrieval scores
are display-free, consumed by auto-pick alone); when one search candidate clearly dominates
(`AUTO_PICK_MIN`/`AUTO_PICK_GAP`, default 0.80/0.05), labeler call 2 is skipped
and that candidate is auto-picked (marked «автоподбор»).

Manual pdf queries (architecture.md §8): a message
whose first whitespace token is `pdf`/`пдф` (case-insensitive) skips the
answerer — the stripped payload goes to labeler 1 in direct mode (it normalizes
the names and decides one vs several diseases), then the standard search →
auto-pick/select → renderer machinery; a pdf turn is input-only — the payload
is the sole context (owner decision 2026-10-04, no last-answer context).
Rendered as ONE chat row — the user bubble retitled «Врач (подбор
pdf)» with the stripped payload, sources block below (the stored assistant
caption row is merged away in the UI); stored with `messages.kind='pdf'`,
which never enters the answerer history. Bare «pdf» without a disease is a 400.

Patient card context (architecture.md §9, demo, owner decision 2026-10-06):
hand-made two-file demo pairs in `data/patient_cards/` — the attachable
`{ФИО}.выписка.md` extract plus its non-attachable source chart
`{ФИО}.амбулаторная_карта.md` (5 fake patients ship; the generator prompt
`data/patient_cards/PROMPT_browser_agent_EMK.md` describes the pair
convention); `GET /api/cards` lists the extracts only. The composer's «Карта пациента» is one
toggle button (owner decision 2026-10-07): OFF — click opens the picker;
ON — shows the patient name, click detaches it, and every non-pdf question
carries the card while ON. The visible state is per chat session —
switching chats or reloading resets the button to OFF — but the session
invisibly remembers its card filename (`sessions.card_file`, pinned on
pick via `POST /api/sessions/{id}/card`, re-pinned on send; `GET
/api/sessions/{id}/card` reads it): the picker preselects it,
highlighted at the bottom of the list (the menu opens upward). Message
content never stores card text; audit logs filename+chars only.
Validation (`cards.py`, shared with the CLI): traversal-safe filename,
`.выписка.md` suffix, size ≤ `CARD_MAX_CHARS` (default 8000) → 400 before
the question is sent; the file is re-read fresh on every request (the
doctor may edit it mid-session). Pdf turns ignore the card entirely.

## Layout

```
medassistant/            runtime package (config, llm, prompts, abbr, gindex,
                         pipeline, appendix, audit, store, health, stats,
                         canary, regression, cards, cli, server/)
scripts/                 one-time setup tools (strip_vision_gguf.py)
models/                  Modelfiles (medgemma-cpu, medgemma-labeler, bge-cpu)
                         + .backup.modelfile of the pre-text-only definitions;
                         medgemma1.5-4b-q4km-text-only.gguf source GGUF
data/docs_meta.json      register metadata, 757 guidelines (copied from gp-eval)
data/abbr2term.yaml      4,508 RU medical abbreviations (RuCCoD; NO LICENSE —
                         internal use only, do not redistribute)
data/index/              built embedding index (vectors.npy + meta.json with a
                         docs_meta freshness stamp)
data/patient_cards/      hand-made demo patient pairs: attachable
                         *.выписка.md + non-attachable source chart
                         *.амбулаторная_карта.md (5 fake patients,
                         generator prompt PROMPT_browser_agent_EMK.md) —
                         per-question answerer context; nothing is stored
                         (architecture.md §9)
data/sessions.sqlite     persistent chat sessions
data/backups/            startup backups of sessions.sqlite (newest 5 kept)
logs/audit.jsonl         one record per pipeline stage per turn (size-rotated,
                         3 generations)
logs/ollama-<inst>-<stamp>.*.log  per-boot Ollama logs (5 latest boots kept)
logs/web-<stamp>.*.log       per-boot web app logs (5 latest boots kept)
tests/                   offline unit tests + goldens.jsonl (live regression)
```

## Setup

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Models (owner decision 2026-10-06: text-only medgemma + CPU-pinned bge after
the iGPU `ErrorOutOfDeviceMemory` crash loop; base from the gp-eval runbook):
- main instance (`:11434`, auto-started): `medgemma-cpu` = medgemma1.5:4b
  **text-only** (vision projector stripped) with `PARAMETER num_gpu 0`
  (answerer, thinking on) **and** `bge-cpu` = bge-m3 with `PARAMETER num_gpu 0`
  (embeddings; stays off the iGPU so it never competes with the labeler for
  the Vulkan allocation pool — index vectors unchanged, same weights)
- second instance (`:11435`, iGPU): `medgemma-labeler` = medgemma1.5:4b
  **text-only** with a TEMPLATE override appending an empty `<think></think>`
  block (thinking OFF — otherwise reasoning eats the token budget and content
  comes back empty)
- the vision strip keeps all 444 text tensors byte-identical and only drops
  the 439 `v.*`/`mm.*` tensors (the ~1 GiB CLIP encoder this text-only
  pipeline never uses); rebuild after pulling a fresh medgemma1.5:4b:
  `pip install gguf`, then `python scripts/strip_vision_gguf.py --src <blob>
  --out models\medgemma1.5-4b-q4km-text-only.gguf` and `ollama create` each
  variant from `models\*.modelfile` (pre-strip definitions: `*.backup.modelfile`)
- env once: `setx OLLAMA_IGPU_ENABLE 1`, `setx OLLAMA_MAX_LOADED_MODELS 2`,
  then restart Ollama; after each reboot start everything with
  `powershell -File start_chat.ps1` (repo root; toggle: when everything is
  down it starts both Ollama instances and the web app; re-run while up
  STOPS it all; `-SkipWeb` = backends only, `-Foreground` = web app in the
  console)

## Build the index (once, needs the embed endpoint)

```powershell
python -m medassistant build-index        # ~1–2k vectors, minutes
python -m medassistant search --name "ХСН"   # tool debug: abbr expansion + top-5
```

Rebuild after dropping in a refreshed `data/docs_meta.json`.

## Run

```powershell
powershell -File start_chat.ps1                # toggle: all down -> starts
                                                # Ollama + web app
                                                # -> http://127.0.0.1:8866;
                                                # any up -> stops everything
powershell -File start_chat.ps1 -SkipCanary    # CLI full start: no offer
powershell -File start_chat.ps1 -SkipWeb       # Ollama backends only
powershell -File start_chat.ps1 -Foreground    # web app in this console
                                                # (Ctrl+C stops only it)
powershell -File start_chat.ps1 -Canary        # force the startup canary
```

Launch modes (owner decision 2026-10-04): BARE invocation (no arguments) =
hand-clicked (Explorer / Total Commander double-click) — after the backends
start, a 5 s countdown offers the canary (press any key to run it, timeout
skips); a smooth run closes the console on its own after a short linger
(10 s after a start, 2 s after a stop), while on any failure — canary
failure or an unexpected error — the window stays open until a key is
pressed: no auto-close on error (owner decision 2026-10-07). ANY argument =
deliberate CLI run: no offer, no linger (`-SkipCanary` is the
plain CLI full-start). The canary runs before the web app (~40 s: health
checks + one E2E question; it also warms the models, so the first real
question is faster afterwards). On canary failure the stack still comes up
and the script exits 1. Once the web port binds, the app URL opens in the
default browser automatically. The manual command stays available:
`python -m medassistant canary` (`--no-e2e` for the fast checks only).

If both Ollama instances are already up, the web app alone:

```powershell
uvicorn medassistant.server:app --host 127.0.0.1 --port 8866
```

(Port 8866 is chosen to avoid common Docker binds; the script runs everything
windowless with per-boot logs — `logs/ollama-<inst>-<stamp>.*.log` and, for
the background web app, `logs/web-<stamp>.*.log`; the 5 latest boots are
kept. The `uvicorn ...` line above is the manual equivalent.)

CLI (same pipeline, no GUI):

```powershell
python -m medassistant ask --question "Какие критерии диагностики ХСН?" --save
python -m medassistant ask --question "pdf ХСН"            # manual pdf query
                                                             # (no answerer call)
python -m medassistant ask --question "Что с одышкой?" `
    --card Иванов_ИИ.выписка.md                            # patient card context
                                                             # (filename or path)
python -m medassistant ask --answer-file answers.jsonl # appendix machinery over
                                                      # pre-generated answers
```

Answer-file rows: `{"question", "raw_answer"}` (live labeler + search) or with
`"diseases"` (skips labeler 1) or `"picks"` (renderer-only, fully offline).

## Observability (owner decision 2026-10-02: laptop-only, built-in)

- `GET /api/health` — answerer/labeler/embed endpoints, index freshness vs
  `docs_meta.json`, PDF corpus; results cached 30 s (`HEALTH_CACHE_S`).
- `http://127.0.0.1:8866/admin` — status page: health checks, usage/error
  counters and stage latency from `logs/audit.jsonl` (`GET /api/stats?days=N`),
  auto-refresh 30 s. No auth — localhost only.
- `python -m medassistant canary` — the same checks plus one E2E question
  through the real pipeline (`--no-e2e` for fast checks only); exit 1 on
  failure. Also the opt-in startup check of `start_chat.ps1` (`-Canary` /
  5 s keypress countdown on a bare hand-click launch; owner decision
  2026-10-04).
- `python -m medassistant regression` — `tests/goldens.jsonl` against live
  endpoints: deterministic search-tier rows (disease → expected top-1
  guideline) and full-tier rows (question → whole pipeline, tolerant name
  matching). Manual pre-ship step; full tier is minutes per question on CPU —
  `--tier search` runs the fast deterministic rows only. Known flake source:
  the answerer's thinking occasionally consumes the whole token budget and
  the row fails with "empty answer twice" — rerun, and consider raising
  `ANSWERER_MAX_TOKENS` if it repeats (thinking length varies run to run
  even at temperature 0).

Housekeeping: `audit.jsonl` size-rotates (`AUDIT_ROTATE_MB`/`AUDIT_KEEP`,
default 10 MB × 3); `sessions.sqlite` is WAL-checkpointed and copied to
`data/backups/` at every server startup (`SESSIONS_BACKUP_KEEP`, default 5 —
physician conversations are sensitive, backups stay local); Ollama and web
app logs are per-boot timestamped, 5 latest boots kept.

## Tests

```powershell
python -m unittest discover -s tests    # offline, no endpoints needed
```

Pre-ship checklist: the offline suite above, then `python -m medassistant
canary`, then `python -m medassistant regression` (both live, minutes on CPU).

## Notes

- Transport-level retries only (429/5xx/timeouts); invalid labeler output is
  never retried — the answer ships with an honest note.
- Thinking-loop watchdog (owner decision 2026-10-06): a degenerate
  pre-content reasoning loop (verbatim tail repetition) aborts the answerer
  stream — closing the connection cancels Ollama generation and frees the
  CPU — and retries once with `frequency_penalty`
  (`LOOP_RETRY_FREQ_PENALTY`, only knob family `LOOP_*` in `.env`; the UI
  gets `reasoning_reset` and drops the looped thinking). A second loop or
  an empty retry fails with «Сбой генерации ответа»; audit rows:
  `answerer_loop_abort`, `stage_stop stop_stage=answerer`.
- 27B tier: swap `CANDIDATE_*` in `.env`; everything else is unchanged.
- Default browser on this machine is Yandex Browser; `start <url>` (and any
  "open in browser" action) launches it.

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
as soon as the answer text is done; while the answerer generates, the send
button is a Stop control («Стоп», Esc too; owner decision 2026-10-15) — an
elapsed-time line ticks under the label and lingers 5 s over «Отправить»
fading out (owner request 2026-10-17) — `POST /api/sessions/{id}/stop` drops
the llama-server connection (generation cancels instantly, even mid-prefill),
deletes the half-finished turn and returns the question to the composer for
editing (answerer phase only; the appendix always runs to completion); when
the labeler filter runs, the candidate pdf cards stream to the page immediately
after search (search is cheap, the labeler call is not) and are filtered to the
verified picks client-side when the appendix payload arrives (cards are
name+link only — retrieval scores are display-free, consumed by auto-pick
alone); when one search candidate clearly dominates
(`AUTO_PICK_MIN`/`AUTO_PICK_GAP`, default 0.80/0.05), labeler call 2 is
skipped and that candidate is auto-picked (marked «автоподбор»).

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
hand-made demo charts in `data/patient_cards/` — the attachable
`{ФИО}.амбулаторная_карта.md` itself (owner decision 2026-10-14, superseding
the 2026-10-09 extract+chart pairs; the `*.выписка.md` extracts are gone);
5 fake patients ship; `GET /api/cards` lists the charts. The composer's «Карта
пациента» is one toggle button (owner decision 2026-10-07): OFF — click opens
the picker; ON — shows the patient name, click detaches it, and every non-pdf
question carries the card while ON. The visible state is per chat session —
switching chats or reloading resets the button to OFF — but the session
invisibly remembers its card filename (`sessions.card_file`, pinned on pick via
`POST /api/sessions/{id}/card`, re-pinned on send; `GET
/api/sessions/{id}/card` reads it): the picker preselects it,
highlighted at the bottom of the list (the menu opens upward). Message
content never stores card text; audit logs filename+chars only.
Validation (`cards.py`, shared with the CLI): traversal-safe filename,
`.амбулаторная_карта.md` suffix, size ≤ `CARD_MAX_CHARS` (default 8000) → 400 before
the question is sent; the file is re-read fresh on every request (the
doctor may edit it mid-session). Pdf turns ignore the card entirely.

## Layout

```
medassistant/            runtime package (config, llm, loopguard, prompts,
                         abbr, gindex, pipeline, appendix, audit, store,
                         health, stats, canary, regression, cards, cli, server/
                         + launcher.py)
scripts/                 one-time setup tools (strip_vision_gguf.py,
                         grls_to_sqlite.py)
models/                  medgemma1.5-4b-q4km-text-only.gguf (answerer +
                         labeler, same GGUF, different Jinja template),
                         bge-m3-f16.gguf (embeddings),
                         answerer.jinja, labeler.jinja
engine/llamacpp\         pinned llama.cpp Vulkan build (llama-server.exe +
                         dlls; see deploy/llamacpp-pin.txt)
deploy/                  build pins + scripts:
                         llamacpp-pin.txt, fetch_llamacpp.ps1,
                         bge-pin.txt, fetch_bge.ps1,
                         bootstrap_runtime.ps1, build_deploy_zip.ps1,
                         requirements.txt (freeze-pinned), hide_coding_dirs.bat
data/docs_meta.json      register metadata, 757 guidelines (copied from gp-eval)
data/drugs.sqlite        ГРЛС drugs register (built by scripts/grls_to_sqlite.py;
                         trade name → МНН grounding for the answerer, missing
                         file = feature silently off; inn_en/inn_lat/inn_pair
                         columns hold per-component English/Latin INNs and the
                         pre-rendered bilingual МНН string written by
                         scripts/inn_to_en.py — the runtime reads inn_pair,
                         never a side file)
data/drug_aliases.yaml   owner-curated colloquial aliases for the МНН lookup
                         (tier 3, fires only when the dict found nothing;
                         value = register trade name OR a direct МНН for
                         register holes — e.g. валерьянка, кагоцел; owner
                         truth, never auto-extended)
data/inn_gloss.yaml      owner-curated English glosses for МНН components the
                         WHO list cannot convert (true holes: ментилизовалерат,
                         фармакопейные соли железа, салицилаты); folded into
                         the inn_pair column AT BUILD TIME — re-run
                         scripts/inn_to_en.py after editing it
data/inn_list_en-lat.txt owner-prepared pairing "english<TAB>latin" of the
                         WHO recommended INN list (through list 95, FIMEA
                         mirror) — the conversion source for
                         scripts/inn_to_en.py, not loaded at runtime
data/abbr2term.yaml      4,506 RU medical abbreviations (RuCCoD; NO LICENSE —
                         internal use only, do not redistribute)
data/index/              built embedding index (vectors.npy + meta.json with a
                         docs_meta freshness stamp)
data/patient_cards/      hand-made demo patient charts: attachable
                          *.амбулаторная_карта.md (5 fake patients) —
                          per-question answerer context; nothing is stored
                          (architecture.md §9)
data/sessions.sqlite     persistent chat sessions
data/backups/            startup backups of sessions.sqlite (newest 5 kept)
logs/audit.jsonl         one record per pipeline stage per turn (size-rotated,
                          3 generations)
logs/llama-<role>-<stamp>.*.log  per-boot llama-server logs (answerer/labeler/
                          embed, 5 latest boots kept)
logs/web-<stamp>.*.log       per-boot web app logs (5 latest boots kept)
tests/                   offline unit tests + goldens.jsonl (live regression)
```

## Setup

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

The dev venv talks to llama-server (not Ollama). Build the runtime + zip:

```powershell
# 1. Pin + fetch llama.cpp (Vulkan x64) into engine\llamacpp\
pwsh deploy/fetch_llamacpp.ps1

# 2. Fetch the pinned bge-m3 F16 GGUF into models\
pwsh deploy/fetch_bge.ps1

# 3. Build the index through the llama-server bge endpoint (:8083)
python -m medassistant build-index
python -m medassistant search --name "ХСН"

# 4. Build the embeddable CPython runtime into python\
pwsh deploy/bootstrap_runtime.ps1

# 5. Assemble the deploy zip (whitelist build, dist-manifest pinned)
pwsh deploy/build_deploy_zip.ps1
```

Target machine: extract `medassistant.zip` anywhere (short path, e.g.
`C:\medassistant`), then double-click `Start.bat`. No Docker, no internet, no
prerequisites.

## Models (owner decision 2026-10-18)

`llama.cpp` (llama-server, Vulkan build pinned in `deploy/llamacpp-pin.txt`)
is the only supported backend. Three processes (ports 8081/8082/8083), started
by `python -m medassistant.launcher` or `Start.bat`:

- answerer (:8081): medgemma1.5-4b-q4km-text-only GGUF, `num_ctx 12288`
  (patient card + history + the 2048-token thinking/answer budget), thinking
  ON, `--temp 0`, alias `medgemma-answerer`, chat-template `answerer.jinja`.
- labeler (:8082): same GGUF, think-suppressing `labeler.jinja` (empty
  `think` block after the last turn), alias `medgemma-labeler`, `-c 8192`.
- embeddings (:8083): bge-m3-f16 GGUF, `--embeddings --pooling cls`,
  alias `bge-m3`, `-c 8192 -np 4`.

Web app stays on 8866.

Ollama is **dropped** as the supported backend. `llm.py` is pure
OpenAI-compatible, so the answerer/labeler/embed `*_BASE_URL`s in `.env` can
still point at Ollama and it will work, but that path is unsupported and
undocumented. No `num_gpu` pins, no `OLLAMA_*` env vars, no `setx` — the
launcher picks GPU (dGPU-only, system-default layer offload) vs CPU
(`-ngl 0`, iGPU-only/none) from display-adapter registry names, with
`--force-cpu`/`--force-gpu` overrides.

## Build the index (once, needs the embed endpoint)

```powershell
python -m medassistant build-index        # through EMBED_BASE_URL (:8083)
python -m medassistant search --name "ХСН"   # tool debug: abbr expansion + top-5
```

Rebuild after dropping in a refreshed `data/docs_meta.json`. The index's
`docs_meta_sha1` is stamped in `data/index/meta.json`; the deploy build fails
if it doesn't match the current `data/docs_meta.json`.

## Run

```powershell
# Dev: toggle launcher (start all / stop all) — no PowerShell in shipped path
python -m medassistant.launcher                 # start all / stop all
python -m medassistant.launcher --skip-canary   # CLI full start: no offer
python -m medassistant.launcher --skip-web      # llama-servers only
python -m medassistant.launcher --foreground    # web app in this console
                                                # (Ctrl+C stops only the web)
python -m medassistant.launcher --canary        # force the startup canary
python -m medassistant.launcher --force-cpu     # -ngl 0 regardless of GPU
python -m medassistant.launcher --force-gpu     # system-default layers

# Target machine (extract zip first, short path):
Start.bat
```

Launch modes (owner decision 2026-10-04): BARE invocation (no arguments) =
hand-clicked (Explorer / Total Commander double-click) — after the backends
start, a 4 s countdown offers the canary (press any key to run it, timeout
skips); a smooth run closes the console on its own after a short linger
(10 s after a start, 2 s after a stop), while on any failure — canary
failure or an unexpected error — the window stays open until a key is
pressed: no auto-close on error (owner decision 2026-10-07). ANY argument =
deliberate CLI run: no offer, no linger (`--skip-canary` is the
plain CLI full-start). The canary runs before the web app (~40 s: health
checks + one E2E question; it also warms the models, so the first real
question is faster afterwards). On canary failure the stack still comes up
and the launcher exits 1. Once the web port binds, the app URL opens in the
default browser automatically. The manual command stays available:
`python -m medassistant canary` (`--no-e2e` for the fast checks only).

If the backends are already up, the web app alone:

```powershell
uvicorn medassistant.server:app --host 127.0.0.1 --port 8866
```

(Port 8866 is chosen to avoid common Docker binds; the launcher runs
everything windowless with per-boot logs — `logs/llama-<role>-<stamp>.*.log`
and, for the background web app, `logs/web-<stamp>.*.log`; the 5 latest boots
are kept. The `uvicorn ...` line above is the manual equivalent.)

CLI (same pipeline, no GUI):

```powershell
python -m medassistant ask --question "Какие критерии диагностики ХСН?" --save
python -m medassistant ask --question "pdf ХСН"            # manual pdf query
                                                             # (no answerer call)
python -m medassistant ask --question "Что с одышкой?" `
    --card Иванова_МП.амбулаторная_карта.md            # patient card context
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
  failure. Also the opt-in startup check of the launcher (`--canary` / 4 s
  keypress countdown on a bare hand-click launch; owner decision 2026-10-04).
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
physician conversations are sensitive, backups stay local); llama-server and
web app logs are per-boot timestamped, 5 latest boots kept.

## Tests

```powershell
python -m unittest discover -s tests    # offline, no endpoints needed
```

Pre-ship checklist: the offline suite above, then `python -m medassistant
canary`, then `python -m medassistant regression` (both live, minutes on CPU).

## Notes

- Transport-level retries only (429/5xx/timeouts); invalid labeler output is
  never retried — the answer ships with an honest note.
- Thinking-loop watchdog (owner decision 2026-10-06; detector redefined
  2026-10-14): a degenerate pre-content reasoning loop — the same unit
  repeated back-to-back (consecutively; scattered recurring phrases are NOT
  loops — the first anywhere-in-window detector falsely aborted a healthy
  card-example enumeration, owner report 2026-10-14) — aborts the answerer
  stream — closing the connection cancels llama-server generation and frees
  the CPU — and retries once with `frequency_penalty`
  (`LOOP_RETRY_FREQ_PENALTY`, only knob family `LOOP_*` in `.env`; the UI
  gets `reasoning_reset`, drops the looped thinking and shows a
  «Зацикливание размышлений — …» note; a fatal second loop's garbage is
  dropped with a note too). A second loop or an empty retry fails with
  «Сбой генерации ответа»; audit rows: `answerer_loop_abort`,
  `stage_stop stop_stage=answerer`.
- 27B tier: swap `CANDIDATE_*` in `.env`; everything else is unchanged.
- Default browser on this machine is Yandex Browser; `start <url>` (and any
  "open in browser" action) launches it.

# med_assistant

Medical reference assistant for Russian physicians — answer-first pipeline with a
post-hoc guideline verification appendix. The authoritative design is
[`architecture.md`](architecture.md); this README is the runbook.

Pipeline: ANSWERER (plain medical model, guideline-blind) → LABELER call 1
(diseases as JSON) → TOOL `search_guidelines` (embedding index, top-5 cards, no
URLs) → LABELER call 2 (pick ids or none) → RENDERER (register PDF links under
the untouched answer). Failures surface as honest notes; nothing is retried at
the semantic level. Delivered as a chat web app.

UX (owner decisions 2026-09-29, architecture.md §9): the answerer's reasoning
streams into a collapsible «Размышления модели» block (auto-collapses when the
answer starts); the appendix machinery shows a live stage line («Поиск болезней
в тексте…» → «Ищу соответствующие pdf.» → …) and the composer unblocks
as soon as the answer text is done; when the labeler filter runs, the candidate
pdf cards stream to the page immediately after search (search is cheap, the
labeler call is not) and are filtered to the verified picks client-side when
the appendix payload arrives; when one search candidate clearly dominates
(`AUTO_PICK_MIN`/`AUTO_PICK_GAP`, default 0.80/0.05), labeler call 2 is skipped
and that candidate is auto-picked (marked «автоподбор»).

## Layout

```
medassistant/            runtime package (config, llm, prompts, abbr, gindex,
                         pipeline, appendix, audit, store, health, stats,
                         canary, regression, cli, server/)
data/docs_meta.json      register metadata, 757 guidelines (copied from gp-eval)
data/abbr2term.yaml      4,508 RU medical abbreviations (RuCCoD; NO LICENSE —
                         internal use only, do not redistribute)
data/index/              built embedding index (vectors.npy + meta.json with a
                         docs_meta freshness stamp)
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

Models (from the gp-eval runbook; per README §5 of gp-eval):
- main instance (`:11434`, auto-started): `medgemma-cpu` = medgemma1.5:4b with
  `PARAMETER num_gpu 0` (answerer, thinking on) **and** `bge-m3` (embeddings)
- second instance (`:11435`, iGPU): `medgemma-labeler` = medgemma1.5:4b with a
  TEMPLATE override appending an empty `<think></think>` block (thinking OFF —
  otherwise reasoning eats the token budget and content comes back empty)
- env once: `setx OLLAMA_IGPU_ENABLE 1`, `setx OLLAMA_MAX_LOADED_MODELS 2`,
  then restart Ollama; after each reboot start everything with
  `powershell -File scripts\start_assistant.ps1` (toggle: when everything is
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
powershell -File scripts\start_assistant.ps1    # toggle: all down -> starts
                                                 # Ollama + web app
                                                 # -> http://127.0.0.1:8866;
                                                 # any up -> stops everything
powershell -File scripts\start_assistant.ps1 -SkipWeb      # Ollama backends only
powershell -File scripts\start_assistant.ps1 -Foreground   # web app in this
                                              # console (Ctrl+C stops only it)
powershell -File scripts\start_assistant.ps1 -Canary       # force the startup
                                              # canary (any launch mode)
```

The startup canary is opt-in (owner decision 2026-10-04): launched by hand
(Explorer double-click / shortcut) the script shows a 4 s countdown — press
any key to run it, timeout skips; from a CLI it is skipped unless `-Canary`
is passed. It runs after the backends, before the web app (~40 s: health
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
python -m medassistant ask --answer-file mock.jsonl   # appendix machinery over
                                                      # pre-generated answers
```

`mock.jsonl` rows: `{"question", "raw_answer"}` (live labeler + search) or with
`"diseases"` (skips labeler 1) or `"picks"` (renderer-only, fully offline).

## Observability (owner decision 2026-10-02: laptop-only, built-in)

- `GET /api/health` — answerer/labeler/embed endpoints, index freshness vs
  `docs_meta.json`, PDF corpus; results cached 30 s (`HEALTH_CACHE_S`).
- `http://127.0.0.1:8866/admin` — status page: health checks, usage/error
  counters and stage latency from `logs/audit.jsonl` (`GET /api/stats?days=N`),
  auto-refresh 30 s. No auth — localhost only.
- `python -m medassistant canary` — the same checks plus one E2E question
  through the real pipeline (`--no-e2e` for fast checks only); exit 1 on
  failure. Also the opt-in startup check of `start_assistant.ps1`
  (`-Canary` / 4 s keypress countdown on a hand-click launch; owner
  decision 2026-10-04).
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
- 27B tier: swap `CANDIDATE_*` in `.env`; everything else is unchanged.
- Default browser on this machine is Yandex Browser; `start <url>` (and any
  "open in browser" action) launches it.

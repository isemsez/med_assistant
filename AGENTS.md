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
                         pipeline, appendix, audit, store, cli, server/)
data/docs_meta.json      register metadata, 757 guidelines (copied from gp-eval)
data/abbr2term.yaml      4,508 RU medical abbreviations (RuCCoD; NO LICENSE —
                         internal use only, do not redistribute)
data/index/              built embedding index (vectors.npy + meta.json)
data/sessions.sqlite     persistent chat sessions
logs/audit.jsonl         one record per pipeline stage per turn
tests/                   offline unit tests
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
  `powershell -File scripts\start_assistant.ps1` (starts both Ollama instances if
  down, then the web app)

## Build the index (once, needs the embed endpoint)

```powershell
python -m medassistant build-index        # ~1–2k vectors, minutes
python -m medassistant search --name "ХСН"   # tool debug: abbr expansion + top-5
```

Rebuild after dropping in a refreshed `data/docs_meta.json`.

## Run

```powershell
powershell -File scripts\start_assistant.ps1    # starts Ollama if down + web app
# -> http://127.0.0.1:8866
```

If both Ollama instances are already up, the web app alone:

```powershell
uvicorn medassistant.server:app --host 127.0.0.1 --port 8866
```

(Port 8866 is chosen to avoid common Docker binds; Ollama instances started
by the script run windowless, logging to `logs/ollama-*.log`.)

CLI (same pipeline, no GUI):

```powershell
python -m medassistant ask --question "Какие критерии диагностики ХСН?" --save
python -m medassistant ask --answer-file mock.jsonl   # appendix machinery over
                                                      # pre-generated answers
```

`mock.jsonl` rows: `{"question", "raw_answer"}` (live labeler + search) or with
`"diseases"` (skips labeler 1) or `"picks"` (renderer-only, fully offline).

## Tests

```powershell
python -m unittest discover -s tests    # offline, no endpoints needed
```

## Notes

- Transport-level retries only (429/5xx/timeouts); invalid labeler output is
  never retried — the answer ships with an honest note.
- 27B tier: swap `CANDIDATE_*` in `.env`; everything else is unchanged.
- Default browser on this machine is Yandex Browser; `start <url>` (and any
  "open in browser" action) launches it.

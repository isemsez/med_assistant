# med_assistant

Medical reference assistant for Russian physicians — answer-first pipeline with a
post-hoc guideline verification appendix. The authoritative design is
[`.kilo/architecture.md`](.kilo/architecture.md); this README is the runbook.

Pipeline: ANSWERER (plain medical model, guideline-blind) → LABELER call 1
(diseases as JSON; answer-anchored, the bare question analyzed only as a
FALLBACK when the answer names no disease — owner decision 2026-10-25
evening, back to the e77ef04 principle in code: the 2026-10-20 both-passes
widening cost a second extract call on every turn and near-form duplicates
ran double search+select («инфаркт миокарда» from the answer + «инфаркт»
from the question, session 98 16:47 — the ambiguous short form attached
the stroke pdf 814_1 live); the widening's drifted-answer cover is given
up — the doctor re-asks as a separate single-disease question; a «Да.»
answer to «А при инсульте?» still anchors the asked-about guideline via
the fallback; empty-after-denylist counts as empty) →
TOOL `search_guidelines` (embedding index, top-5 cards, no
URLs) → LABELER call 2 (pick ids or none) → RENDERER (register PDF links under
the untouched answer). Labeler call 2 reasons INSIDE the grammar JSON: the
per-call schema requires a per-candidate "reasoning" string before "picks"
(2026-10-20, root-cause fix for empty picks — llama.cpp grammar mode forces
the first generated token to "{" and cannot think before the JSON, unlike
the old Ollama transport; the reasoning string is grammar-bounded
(`maxLength` 1400 = `SELECT_REASONING_MAXLEN` — 700 could not fit a
5-candidate enumeration, ~140–173 chars per verdict item five ≈ 730+, so
the clamp slammed shut mid-list and shipped picks contradicting the lost
verdicts, live 2026-10-20; an unbounded one ran degenerate to the token cap
and left the object unclosed; this pinned llama.cpp refuses GBNF repeats
over 1000 (700 OK, 1001 FAIL, bisected max 1000), so the cap ships as two
adjacent bounded repeats `{0,1000}` + `{0,400}` (`llm.schema_to_grammar`,
`_GBNF_MAX_REPEAT`); verdicts are terse «подходит или не
подходит» and a string exactly at the cap is greppable in audit as
`reasoning_truncated`; and the echo fuel is in the prompt itself — a
`matched` term shared by several candidates is non-discriminative and got
echoed verbatim (2026-10-21: «Синдром «гистаминовой» головной боли» on 3 of
5 lines → 5× echo, 23× without the grammar, ~50× think-loop to the token
cap under the answerer template — every transport repeated it; live probes
also killed the "use the answerer model" idea: unbounded thinking starts
genuinely per-candidate and then spirals, 204 s, empty output), so
`_select_user` prints the parenthetical only when the term annotates a
single candidate) and `LABELER_SELECT_MAX_TOKENS` is 128 → 480 → 768 →
1024 (768 sized the budget at prose density ~2.3 chars/token; a live
degenerate short-item echo «5. 154_4 — подходит.»×N runs 1.75 — 1400
chars ≈ 800+ tokens, and the 768 cap chopped the string at 1347 chars
BEFORE the grammar clamp could close it, unclosed object, needless
select_error — 2026-10-21 canary) so the full enumeration always fits —
the wider cap costs latency (up to ~40 s per select on CPU; a full-cap
degenerate echo ~90 s and is dropped by the loop guard), accepted for a
laptop-only stack). What survives the prompt fix is caught post-hoc by
the thinking-loop watchdog family (2026-10-21): `_select_reasoning_loop`
reuses `loopguard.LoopGuard` on the parsed reasoning (verbatim
tail-spins) plus a number-blind item-block cycle check — the same bodies
under «1.», «2.»… break LoopGuard's back-to-back check, and the live
shapes were a plain echo (same item x5), a cap-clamped truncated echo,
and a full cyclic re-enumeration ([A,B,C]x4); the item tail must not end
with >= 3 back-to-back copies of the same p-item block (p up to 8, a
partial chopped tail block is discarded; 3 blocks, not 2 — an honest
terse double-pick repeats one body twice). A looped reasoning judged
nothing: its picks are dropped (`labeler_select_loop` audit row, no
retry — invalid labeler output is never retried), the title guard is
the deterministic net. The register's audience structure became visible
the same day (owner catch 2026-10-21): same-NAME pairs are
ADULT/PEDIATRIC versions (810_1 «Головная боль напряжения» = Дети,
162_3 = Взрослые; 24 pairs, zero same-name-same-audience docs), their
titles carry no age cue, and the blind echo had been shipping the
CHILDREN'S guideline to adult questions unnoticed. Cards are now
age-enriched from docs_meta `age`: the select prompt renders «(дети)» /
«(взрослые)» / «(взрослые и дети)» markers (rule 4 finally applicable),
the true-duplicate collapse keys on (title, audience) — merging nothing
today, armed for future same-audience register duplicates — and the
appendix marks pediatric picks «, дети». Selection itself became ONE
CALL PER TITLE GROUP (owner decision 2026-10-21 evening, 27B pivoted
away from — 4B-only on an ordinary laptop): the 4B cannot enumerate a
5-candidate mixed list without degenerating, but judges 1-2 items
reliably, so each recommendation title gets its own short call; the
same-NAME adult/pediatric pair shares ONE call (owner constraint — the
audience choice is comparative, splitting the pair would hide the
twin). Picks are unioned, each call's picks filtered to its own group,
and a looped group kills only its own picks (`group=[ids]` on the
labeler_select/_loop audit rows). `frequency_penalty` on the select
call was probed live and REJECTED (2026-10-21): it did not prevent
loops deterministically and degraded healthy runs — a hallucinated
candidate id in reasoning at 0.3, medically wrong synonym claims at
0.5. Honest open state (live 3/3 with sharpened rules 3/4): per-group
calls eliminated the loops but the 4B RUBBER-STAMPS — every candidate
marked «подходит», детские versions included right after quoting rule
4, unrelated docs picked with self-contradicting reasoning (live
canary: 6 picks). Owner decisions 2026-10-21 evening: over-attachment
is tolerated when RARE (no pick cap on the loop-canary row; the
`expect_picks_max` check exists for rows that need it), a symptom
query like «головная боль» MUST attach its guideline pdf, and an
UNCLEAR audience attaches BOTH versions of a pair — the doctor chooses
himself (`_audience_complete`: a half-picked pair with zero audience
wording in question+answer gets the missing twin added, best-scored,
`select_audience_complete` audit row; any «ребён/детск/взросл…»
marker stands the guard down, rule 4 territory). With that the
loop-canary row («А просто при головной боли?»: «ГБН» present, no
select loop) is GREEN live. Retrieval scores are NOT sent to the
labeler (owner asked 2026-10-21, rejected: in the live pair the TOP
score belonged to the детские doc — score-following would amplify
wrong-audience picks; rank is already implicit in candidate order, and
the failures were judgment failures scores cannot encode; scores stay
display-free, consumed by auto-pick alone). Also noted
live: the first labeler call after ~10 min idle sometimes fails
extract with «Сбой распознавания заболевания» (3× today, always the
first run, always fine on rerun) — cold-start flake, not yet
root-caused.
After selection, a unique candidate whose guideline title contains the
extracted disease verbatim is kept even if the selector dropped it (title
guard, 2026-10-19, kept as a safety net — e.g. for empty-string reasoning,
which stays grammar-legal).
Pair-verdict select cache (owner decision 2026-10-26): every settled
selection freezes per-pair verdicts — (normalized disease, audience
bucket none/child/adult from the same markers the audience guard uses) →
guideline id = right/wrong — into `data/select_cache.yaml`
(`SELECT_CACHE=0` disables); a later turn whose candidate pairs are all
covered skips labeler 2 AND auto-pick (stage «Ранее отобрано…», the
appendix rows carry the «из кэша» control; the motivation measured in the live log: 81% of select
group runs were exact (session, disease) repeats eating 87% of select
seconds, and the 4B attached different pdf sets for the same disease on
consecutive turns); partial coverage attaches cached-right pairs without
judging and selects over the unknowns only; a WRONG pair (machine or
human) never attaches — excluded before auto-pick and the guards — and
renders grayed among the filtered-out candidates, where every row carries
revision buttons: 👎 negates the pair's current verdict (toggle
attached/grayed), 👍 (never-revised pairs only) confirms it, any click
makes the pair human-revised (only 👎 remains) and is never overwritten
by machine writes (`POST /api/verdicts`, audit `select_verdict`; the UI
renders rows per CURRENT verdicts, stored payloads are never rewritten);
every cached row also carries a clickable «· из кэша» (owner decision
2026-10-26 evening, replacing a whole-bucket drop — and preferred over a
↻ glyph): forgetting ONE pair makes it unknown
again — the next question's live selector re-judges exactly it while
every other pair of the disease keeps its verdict (partial coverage);
the row falls back to its payload zone and shows «· переоценю заново»
until the re-judge freezes a fresh verdict;
a looped select group's pairs are never frozen. Only turns with a
session id read/write the cache — regression, canary and plain CLI runs
bypass it entirely (the loop-canary golden keeps exercising the live
selector); the file is stamped `docs_meta_sha1` (register refresh =
reset, audit `select_cache_reset`) and stays local — never in the deploy
zip. Failures surface as honest notes;
nothing is retried at the semantic level. Delivered as a chat web app.

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
alone); decided pairs stream in their DECIDED state (owner catch 2026-10-26
evening): a cached-wrong pdf — a 👎 or an earlier filter — grays out in the
provisional block at once instead of pretending to be reconsidered for the
whole select cycle, a cached-right pair is marked «уже отобрано», only the
unmarked rows are actually being filtered; when one search candidate clearly dominates
(`AUTO_PICK_MIN`/`AUTO_PICK_GAP`, default 0.80/0.05), labeler call 2 is
skipped and that candidate is auto-picked (marked «автоподбор»); the
chat auto-follows every stage update while following was on — the follow
decision is the view's position BEFORE the update (owner rule 2026-10-26:
checking after a >80px block like the candidates list broke the follow
exactly there, leaving «Кандидаты pdf…» and the status line below the
fold; manual scroll up still pauses the follow).

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
data/select_cache.yaml   pair-verdict select cache (disease+audience → pdf =
                          right/wrong; machine-written, 👎/👍 human-revisable;
                          docs_meta_sha1-stamped, local runtime state — never
                          in the deploy zip; SELECT_CACHE=0 disables)
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
`C:\med_assistant`; the archive's top folder is `med_assistant\`), then
double-click `Start.bat`. No Docker, no internet, no
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

The answerer and labeler are two llama-server processes not for a GPU/CPU
split (the launcher picks ONE backend for the whole stack) but because a chat
template binds per process — llama.cpp has no per-request template on
/chat/completions, and the roles need different byte-exact prompt tails
(think-open vs pre-closed think) — plus the background appendix of one turn
can overlap the next turn's answerer stream (two processes never queue behind
each other); on CPU both mmap the same GGUF, so the second instance costs ≈
KV cache only (owner decision 2026-10-20: keep both).

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
  E2E pick check (2026-10-20): labeler 1 follows the ANSWER, and the temp-0
  answerer sometimes answers about the question's other disease — the E2E
  then accepts a pick whose guideline title matches the extracted disease
  (the expected guideline check runs first, strictly).
- `python -m medassistant regression` — `tests/goldens.jsonl` against live
  endpoints: deterministic search-tier rows (disease → expected top-1
  guideline) and full-tier rows (question → whole pipeline, tolerant name
  matching). Full-tier rows run with a throwaway audit and are RED on any
  `labeler_select_loop` in the turn (owner decision 2026-10-21: a graceful
  degradation — dropped picks, title-guard rescue — still means the
  selector degenerated; the golden set demands it NOT, so a correct-picks
  turn with a looped reasoning fails the row); a row may carry a "skip"
  reason (2026-10-25): printed as [SKIP] with its reason, excluded from
  the count and the exit code — the general temp-off tool; #15 «Как
  оформить больничный лист?» (`no_disease`) is back ON since 2026-10-25:
  the extract-echo mitigation (register-absent examples + the
  `EXTRACT_DENYLIST` pre-retrieval filter) turned it green — it stays as
  the canary for the extract hallucination; the loop-canary row
  «А просто при головной боли?» expects the ADULT «ГБН» version (rule 4)
  and is the standing red demand for a healthy selector — it also fails
  on a wrong-audience pick, which is how the 4B's детские pick was
  caught. Manual pre-ship step; full tier is minutes per
  question on CPU — `--tier search` runs the fast deterministic rows
  only. Known flake sources: the answerer's thinking occasionally consumes
  the whole token budget and the row fails with "empty answer twice" —
  rerun, and consider raising `ANSWERER_MAX_TOKENS` if it repeats (thinking
  length varies run to run even at temperature 0); rare transient
  `select_error`/`extract_error` rows on an otherwise healthy stack are
  the same family — rerun first, investigate on repeat.

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
  `stage_stop stop_stage=answerer` — both also carry `tail`, the last
  `LOOP_TAIL_CHARS` (default 2000) chars of the looped thinking
  (2026-10-24: before, only the 80-char unit snippet inside `error`
  survived and the content was unrecoverable; `LoopGuard.tail` had been
  dead code). The retry now ALSO sends per-request `dry_*` sampler
  params via `chat_stream` (2026-10-24, knobs `LOOP_RETRY_DRY_*`,
  defaults 0.8/1.75/2/64/␞, `pipeline._retry_dry` reads config at call
  time, 0 = off): frequency_penalty alone still looped live
  2026-10-21 20:49 (fatal turn), while llama.cpp's DRY sampler
  (arXiv 2608.22761, live-verified per-request on the pinned b11724)
  penalizes a token for EXTENDING an already-seen sequence — the
  verbatim echo shape — as a logit adjustment, so it works at temp 0
  (determinism doctrine intact). The sequence breakers are overridden
  with never-emitted U+241E because llama.cpp's defaults (\n : " *)
  reset matching inside loop units containing those chars; the window
  stays shallow (`dry_penalty_last_n` 64, matching the engine's
  always-on repeat_last_n) so the penalty never reaches the patient
  card / prompt (verbatim quoting survives). The attempt-0 payload is
  byte-identical (test-pinned); the labeler select path is
  deliberately untouched — it is quoting-centric, a DRY window over
  the prompt would penalize title quoting, per-group calls already
  killed select loops, and the 2026-10-21 owner rejection stands
  there. The `answerer_loop_abort` row carries `retry_dry` (the
  multiplier actually armed for the retry).
- The same watchdog family guards labeler 2's in-JSON reasoning post-hoc
  (2026-10-21, after live echo loops shipped blind picks): a verbatim
  tail-spin is detected by reusing `loopguard.LoopGuard` on the complete
  reasoning string, and a numbered echo — the same item body under «1.»,
  «2.»… prefixes, which LoopGuard cannot see because the numbers break the
  back-to-back run, and which survives a cap-clamped tail (the clamp chops
  the final copy mid-word) — by a number-blind identical-item-body run
  (`pipeline._select_reasoning_loop`, ≥3 bodies — 2 would drop an honest
  terse double-pick). A looped reasoning judged nothing: its picks are
  dropped, the title guard stays the deterministic net, nothing is
  retried (invalid labeler output is never retried); audit row
  `labeler_select_loop` (reason + dropped picks).
- 27B tier: swap `CANDIDATE_*` in `.env`; everything else is unchanged.
- Default browser on this machine is Yandex Browser; `start <url>` (and any
  "open in browser" action) launches it.

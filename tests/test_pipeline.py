import shutil
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

from medassistant import appendix, llm
from medassistant.audit import Audit
from medassistant.pipeline import Pipeline

DOCS = {
    "1_1": {"name": "Грипп", "url": "https://x/1", "file": "1_1_Грипп.pdf"},
    "2_1": {"name": "ХСН", "url": "https://x/2", "file": "2_1_ХСН.pdf"},
    "3_1": {"name": "Рак носоглотки", "url": "https://x/3",
            "file": "3_1_Рак носоглотки.pdf"},
}
Q, A = "Вопрос врача?", "Ответ ассистента про ХСН."
# two close-scored candidates -> the auto-pick fast path must NOT trigger;
# the selection stage runs (this is the configuration the select tests need)
CARDS = [{"code_version": "2_1", "name": "ХСН", "score": 0.90},
         {"code_version": "1_1", "name": "Грипп", "score": 0.88}]

PDF_DIR = None


def setUpModule():
    global PDF_DIR
    PDF_DIR = Path(tempfile.mkdtemp())
    for meta in DOCS.values():
        (PDF_DIR / meta["file"]).write_bytes(b"")


def tearDownModule():
    shutil.rmtree(PDF_DIR, ignore_errors=True)


def pdf_url(code_version):
    return appendix.PDF_URL_PREFIX + quote(DOCS[code_version]["file"])


def make_pipe(tmp=None, **kw):
    audit = Audit(Path(tmp) / "audit.jsonl") if tmp else None
    defaults = dict(docs=DOCS, abbr_map=None, audit=audit,
                    search_fn=lambda variants: list(CARDS),
                    select_fn=lambda q, a, d, c: ["2_1"],
                    clinrec_dir=PDF_DIR)
    defaults.update(kw)
    return Pipeline(**defaults)


def pipe_with_real_docs(tmp, cards, **kw):
    # resolve_picks drops ids missing from docs_meta / disk — give the
    # real register ids real (empty) pdf files in the temp dir; the
    # fixture docs carry the card's age so enrichment applies
    docs = dict(DOCS)
    for c in cards:
        fname = f"{c['code_version']}_{c['name']}.pdf"
        (Path(tmp) / fname).write_bytes(b"")
        meta = {"name": c["name"], "url": "https://x/" + c["code_version"],
                "file": fname}
        if c.get("age"):
            meta["age"] = c["age"]
        docs[c["code_version"]] = meta
    return make_pipe(tmp, docs=docs, clinrec_dir=Path(tmp),
                     search_fn=lambda variants: list(cards), **kw)


class TestHappyPath(unittest.TestCase):
    def test_full_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = make_pipe(tmp)
            res = pipe.run_appendix(Q, A, extract_fn=lambda q, a: ["ХСН"])
            self.assertEqual(res.status, "ok")
            self.assertEqual(res.diseases, ["ХСН"])
            self.assertEqual(res.by_disease[0]["picks"][0]["url"], pdf_url("2_1"))
            self.assertEqual(res.by_disease[0]["picks"][0]["path"],
                             str(PDF_DIR / DOCS["2_1"]["file"]))
            self.assertIn(f"«ХСН» (2_1): {PDF_DIR / DOCS['2_1']['file']}",
                          res.appendix_text)
            self.assertEqual([(c["code_version"], c["disease"])
                              for c in res.filtered_out], [("1_1", "ХСН")])
            self.assertEqual(res.to_payload()["filtered_out"], res.filtered_out)
            self.assertEqual(res.notes, [])
            lines = (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8").splitlines()
            stages = [__import__("json").loads(l)["stage"] for l in lines]
            self.assertIn("diseases", stages)
            self.assertIn("search", stages)
            self.assertIn("appendix_done", stages)


class TestDiseasesParam(unittest.TestCase):
    def test_diseases_stub_skips_extract(self):
        pipe = make_pipe(extract_fn=lambda q, a: ["never called"])
        res = pipe.run_appendix(Q, A, diseases=["ХСН"])
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.diseases, ["ХСН"])


class TestFailureStops(unittest.TestCase):
    def test_extract_error(self):
        def boom(q, a):
            raise llm.LLMError("invalid json")
        pipe = make_pipe()
        res = pipe.run_appendix(Q, A, extract_fn=boom)
        self.assertEqual(res.status, "extract_error")
        self.assertEqual(res.notes, [appendix.NOTE_EXTRACT_ERROR])
        self.assertIsNone(res.appendix_text)

    def test_no_disease(self):
        pipe = make_pipe()
        res = pipe.run_appendix(Q, A, extract_fn=lambda q, a: [])
        self.assertEqual(res.status, "no_disease")
        self.assertEqual(res.notes, [appendix.NOTE_NO_DISEASE])

    def test_search_error_stops(self):
        searched = []

        def boom(variants):
            searched.append(variants)
            raise llm.LLMError("embed endpoint down")

        pipe = make_pipe(search_fn=boom)
        res = pipe.run_appendix(Q, A, diseases=["А", "Б"])
        self.assertEqual(res.status, "search_error")
        self.assertEqual(res.notes, [appendix.NOTE_SEARCH_ERROR])
        self.assertEqual(len(searched), 1, "pipeline stops at first disease")

    def test_select_error_stops(self):
        picked = []

        def boom(q, a, d, c):
            picked.append(d)
            raise llm.LLMError("no JSON object in response: …")

        pipe = make_pipe(select_fn=boom)
        res = pipe.run_appendix(Q, A, diseases=["А", "Б"])
        self.assertEqual(res.status, "select_error")
        self.assertEqual(res.notes, [appendix.NOTE_SELECT_ERROR])
        self.assertEqual(len(picked), 1)

    def test_no_match_silent(self):
        pipe = make_pipe(select_fn=lambda q, a, d, c: [])
        res = pipe.run_appendix(Q, A, diseases=["А", "Б"])
        self.assertEqual(res.status, "ok")
        self.assertIsNone(res.appendix_text)
        self.assertEqual(res.notes, [])
        self.assertEqual(res.by_disease, [])
        self.assertEqual([(c["code_version"], c["disease"])
                          for c in res.filtered_out],
                         [("2_1", "А"), ("1_1", "А")])

    def test_unknown_pick_ids_resolve_empty(self):
        pipe = make_pipe(select_fn=lambda q, a, d, c: ["ghost_9"])
        res = pipe.run_appendix(Q, A, diseases=["А"])
        self.assertEqual(res.notes, [])
        self.assertIsNone(res.appendix_text)
        self.assertEqual([c["code_version"] for c in res.filtered_out],
                         ["2_1", "1_1"])

    def test_missing_pdf_dropped_silent(self):
        pipe = make_pipe(clinrec_dir=Path("nowhere"))
        res = pipe.run_appendix(Q, A, extract_fn=lambda q, a: ["ХСН"])
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.notes, [])
        self.assertIsNone(res.appendix_text)
        self.assertEqual(res.by_disease, [])
        self.assertEqual(res.filtered_out, [])


class TestCap(unittest.TestCase):
    def test_cap_three(self):
        seen = []
        pipe = make_pipe(select_fn=lambda q, a, d, c: ["2_1"],
                         search_fn=lambda v: (seen.append(v), list(CARDS))[1])
        res = pipe.run_appendix(Q, A, diseases=["А", "Б", "В", "Г", "Д"])
        self.assertEqual(res.diseases, ["А", "Б", "В"])
        self.assertEqual(res.dropped, ["Г", "Д"])
        self.assertEqual(len(seen), 3)

    def test_dedup_and_strip(self):
        pipe = make_pipe()
        res = pipe.run_appendix(Q, A, diseases=[" ХСН ", "ХСН", ""])
        self.assertEqual(res.diseases, ["ХСН"])


class TestAbbrVariantsPassedToSearch(unittest.TestCase):
    def test_variants(self):
        from medassistant.abbr import Abbr
        got = []
        pipe = make_pipe(abbr_map=Abbr({"ХСН": "хроническая сердечная недостаточность"}))
        pipe.search_fn = lambda v: (got.append(v), list(CARDS))[1]
        pipe.run_appendix(Q, A, diseases=["ХСН"])
        self.assertEqual(got[0], ["ХСН", "хроническая сердечная недостаточность"])


class TestAutoPick(unittest.TestCase):
    def _run(self, cards, **kw):
        selected = []
        defaults = dict(docs=DOCS, clinrec_dir=PDF_DIR,
                        search_fn=lambda v: list(cards),
                        select_fn=lambda q, a, d, c: selected.append(d) or ["2_1"])
        defaults.update(kw)
        return Pipeline(**defaults).run_appendix(Q, A, diseases=["ХСН"]), selected

    def test_dominant_candidate_auto_picked(self):
        cards = [{"code_version": "2_1", "name": "ХСН", "score": 0.90},
                 {"code_version": "1_1", "name": "Грипп", "score": 0.80}]
        res, selected = self._run(cards)
        self.assertEqual(selected, [], "labeler 2 must not run")
        self.assertEqual(res.by_disease[0]["picks"][0]["code_version"], "2_1")
        self.assertTrue(res.by_disease[0]["auto_pick"])
        self.assertEqual([c["code_version"] for c in res.filtered_out], ["1_1"])

    def test_single_strong_candidate_auto_picked(self):
        res, selected = self._run([{"code_version": "2_1", "name": "ХСН",
                                    "score": 0.85}])
        self.assertEqual(selected, [])
        self.assertTrue(res.by_disease[0]["auto_pick"])
        self.assertEqual(res.filtered_out, [])

    def test_weak_top_not_auto_picked(self):
        cards = [{"code_version": "2_1", "name": "ХСН", "score": 0.79},
                 {"code_version": "1_1", "name": "Грипп", "score": 0.60}]
        res, selected = self._run(cards)
        self.assertEqual(selected, ["ХСН"], "labeler 2 runs")
        self.assertFalse(res.by_disease[0]["auto_pick"])

    def test_close_scores_not_auto_picked(self):
        cards = [{"code_version": "2_1", "name": "ХСН", "score": 0.90},
                 {"code_version": "1_1", "name": "Грипп", "score": 0.88}]
        res, selected = self._run(cards)
        self.assertEqual(selected, ["ХСН"], "labeler 2 runs")

    def test_kill_switch(self):
        from unittest.mock import patch
        from medassistant import config as cfg
        cards = [{"code_version": "2_1", "name": "ХСН", "score": 0.95}]
        with patch.object(cfg, "AUTO_PICK", False):
            res, selected = self._run(cards)
        self.assertEqual(selected, ["ХСН"], "disabled -> labeler 2 runs")
        self.assertFalse(res.by_disease[0]["auto_pick"])


class TestTitleGuard(unittest.TestCase):
    """A unique verbatim title match survives labeler over-filtering."""

    CARDS = [
        {"code_version": "1_1", "name": "Ишемический инсульт и транзиторная "
                                        "ишемическая атака", "score": 0.70},
        {"code_version": "2_1", "name": "Острый инфаркт миокарда с подъемом "
                                        "сегмента ST электрокардиограммы",
         "score": 0.66},
    ]

    def _run(self, disease, picks, cards=None):
        pipe = make_pipe(search_fn=lambda v: list(cards or self.CARDS),
                         select_fn=lambda q, a, d, c: list(picks))
        return pipe.run_appendix(Q, A, diseases=[disease])

    def test_added_when_labeler_returns_empty(self):
        res = self._run("инфаркт", [])
        self.assertEqual(res.status, "ok")
        self.assertEqual([p["code_version"]
                          for p in res.by_disease[0]["picks"]], ["2_1"])
        self.assertEqual(res.filtered_out[0]["code_version"], "1_1")

    def test_added_to_nonempty_picks(self):
        res = self._run("инфаркт миокарда", ["1_1"])
        self.assertEqual([p["code_version"]
                          for p in res.by_disease[0]["picks"]],
                         ["1_1", "2_1"])

    def test_ambiguous_titles_left_to_labeler(self):
        cards = self.CARDS + [
            {"code_version": "3_1", "name": "Инфаркт миокарда у детей",
             "score": 0.60}]
        res = self._run("инфаркт", [], cards=cards)
        self.assertIsNone(res.appendix_text)
        self.assertEqual(len(res.filtered_out), 3)


class TestIterAppendix(unittest.TestCase):
    def test_stage_sequence_happy_path(self):
        pipe = make_pipe()
        events = list(pipe.iter_appendix(Q, A, extract_fn=lambda q, a: ["ХСН"]))
        self.assertEqual([e["stage"] for e in events],
                         ["extract", "search", "select", "done"])
        self.assertEqual(events[-1]["result"].status, "ok")
        self.assertEqual(events[-1]["result"].by_disease[0]["picks"][0]["url"],
                         pdf_url("2_1"))

    def test_stage_sequence_auto_pick(self):
        pipe = make_pipe(search_fn=lambda v: [
            {"code_version": "2_1", "name": "ХСН", "score": 0.9}])
        events = list(pipe.iter_appendix(Q, A, extract_fn=lambda q, a: ["ХСН"]))
        self.assertEqual([e["stage"] for e in events],
                         ["extract", "search", "auto_pick", "done"])
        self.assertEqual(events[2]["code_version"], "2_1")
        self.assertNotIn("cards", events[2], "auto-pick needs no candidate list")

    def test_select_event_carries_file_enriched_cards(self):
        pipe = make_pipe()
        events = list(pipe.iter_appendix(Q, A, extract_fn=lambda q, a: ["ХСН"]))
        self.assertEqual(events[2]["stage"], "select")
        self.assertEqual(events[2]["cards"], [
            {"code_version": "2_1", "name": "ХСН", "url": pdf_url("2_1"),
             "path": str(PDF_DIR / DOCS["2_1"]["file"])},
            {"code_version": "1_1", "name": "Грипп", "url": pdf_url("1_1"),
             "path": str(PDF_DIR / DOCS["1_1"]["file"])},
        ])

    def test_stage_sequence_extract_error(self):
        def boom(q, a):
            raise llm.LLMError("invalid json")
        pipe = make_pipe()
        events = list(pipe.iter_appendix(Q, A, extract_fn=boom))
        self.assertEqual([e["stage"] for e in events], ["extract", "done"])
        self.assertEqual(events[-1]["result"].status, "extract_error")

    def test_stubbed_diseases_skip_extract_event(self):
        pipe = make_pipe()
        events = list(pipe.iter_appendix(Q, A, diseases=["ХСН"]))
        self.assertEqual(events[0]["stage"], "search")


class TestSplitPdfPrefix(unittest.TestCase):
    def test_prefix_variants(self):
        from medassistant.pipeline import split_pdf_prefix
        self.assertEqual(split_pdf_prefix("pdf ХСН"), "ХСН")
        self.assertEqual(split_pdf_prefix("ПДФ хсн"), "хсн")
        self.assertEqual(split_pdf_prefix("Pdf  ХСН и ХОБЛ"), "ХСН и ХОБЛ")
        self.assertEqual(split_pdf_prefix("  pdf ХСН "), "ХСН")

    def test_bare_prefix_is_empty_payload(self):
        from medassistant.pipeline import split_pdf_prefix
        self.assertEqual(split_pdf_prefix("pdf"), "")
        self.assertEqual(split_pdf_prefix("пдф   "), "")

    def test_non_pdf_text_is_none(self):
        from medassistant.pipeline import split_pdf_prefix
        self.assertIsNone(split_pdf_prefix("pdfхсн"))  # no space after keyword
        self.assertIsNone(split_pdf_prefix("Вопрос про ХСН?"))
        self.assertIsNone(split_pdf_prefix("какой pdf формат у файла?"))
        self.assertIsNone(split_pdf_prefix(""))


class TestManualExtract(unittest.TestCase):
    def _pipe(self, raw):
        from medassistant import prompts
        pipe = make_pipe()
        seen = {}

        def fake_labeler_chat(system, user, max_tokens, schema):
            seen["system"], seen["user"] = system, user
            return raw

        pipe._labeler_chat = fake_labeler_chat
        return pipe, seen

    def test_direct_prompt_parses_and_dedups(self):
        from medassistant import prompts
        pipe, seen = self._pipe('{"diseases": [" ХСН ", "ХСН", "ХОБЛ"]}')
        self.assertEqual(pipe.extract_manual("ХСН и ХОБЛ"), ["ХСН", "ХОБЛ"])
        self.assertIs(seen["system"], prompts.SYSTEM_LABELER_EXTRACT_QUESTION)
        self.assertEqual(seen["user"], "ХСН и ХОБЛ")

    def test_invalid_output_raises(self):
        pipe, _ = self._pipe("не JSON")
        with self.assertRaises(llm.LLMError):
            pipe.extract_manual("ХСН")


class TestManualAppendix(unittest.TestCase):
    def test_manual_flow_runs_extract_first(self):
        pipe = make_pipe()
        pipe.extract_manual = lambda payload: ["ХСН"]
        events = list(pipe.iter_appendix("ХСН", A, manual=True))
        self.assertEqual([e["stage"] for e in events],
                         ["extract", "search", "select", "done"])
        self.assertEqual(events[-1]["result"].status, "ok")

    def test_manual_no_disease_note(self):
        pipe = make_pipe()
        pipe.extract_manual = lambda payload: []
        res = pipe.run_appendix("что-то", A, manual=True)
        self.assertEqual(res.status, "no_disease")
        self.assertEqual(res.notes, [appendix.NOTE_NO_DISEASE_MANUAL])

    def test_injected_extract_fn_still_wins(self):
        pipe = make_pipe(extract_fn=lambda q, a: ["из_fn"])
        res = pipe.run_appendix("X", A, manual=True)
        self.assertEqual(res.diseases, ["из_fn"])

    def test_manual_select_gets_no_answer_context(self):
        seen = {}

        def sel(q, a, d, c):
            seen["answer"] = a
            return ["2_1"]

        pipe = make_pipe(select_fn=sel)
        pipe.extract_manual = lambda payload: ["ХСН"]
        res = pipe.run_appendix("ХСН", "контекст из прошлого вопроса", manual=True)
        self.assertEqual(res.status, "ok")
        self.assertEqual(seen["answer"], "")

    def test_manual_select_uses_direct_system_prompt(self):
        from medassistant import prompts
        seen = {}

        def fake_labeler_chat(system, user, max_tokens, schema):
            seen["system"], seen["user"] = system, user
            return '{"picks": ["2_1"]}'

        pipe = make_pipe(select_fn=None)
        pipe._labeler_chat = fake_labeler_chat
        pipe.extract_manual = lambda payload: ["ХСН"]
        res = pipe.run_appendix("ХСН", "старый ответ", manual=True)
        self.assertEqual(res.status, "ok")
        self.assertIs(seen["system"], prompts.SYSTEM_LABELER_SELECT_DIRECT)
        self.assertNotIn("Ответ ассистента", seen["user"])

    def test_select_user_prompt_omits_empty_answer(self):
        from medassistant.pipeline import _select_user
        without = _select_user("ХСН", "", "ХСН", CARDS)
        self.assertNotIn("Ответ ассистента", without)
        with_answer = _select_user(Q, A, "ХСН", CARDS)
        self.assertIn(f"Ответ ассистента:\n{A}", with_answer)

    def test_audit_records_manual_flag(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            pipe = make_pipe(tmp)
            pipe.extract_manual = lambda payload: ["ХСН"]
            pipe.run_appendix("ХСН", A, manual=True)
            recs = [json.loads(l) for l in
                    (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8")
                    .splitlines()]
            self.assertTrue(all(r.get("manual") for r in recs
                                if r["stage"] == "diseases"))


class TestSelectCoTSchema(unittest.TestCase):
    """Owner decision 2026-10-20 (root-cause fix for empty picks): the
    grammar forces the first generated token to be "{", so the selector
    cannot re-open think before the JSON — it reasons INSIDE the constrained
    JSON instead: a required "reasoning" string property emitted FIRST
    (GBNF property order = dict insertion order). The parser stays lenient:
    picks-only output (test fakes, CLI answer files) keeps parsing.
    Since 2026-10-21 selection is ONE CALL PER TITLE GROUP (owner
    decision: the 4B enumerates 1-2 items reliably, never 5 mixed; the
    adult/pediatric pair shares its call — the audience choice is
    comparative), so CARDS (two titles) means two calls, each schema
    scoped to its own group."""

    def _select(self, raw):
        calls = []

        def fake_labeler_chat(system, user, max_tokens, schema):
            calls.append({"schema": schema, "user": user,
                          "max_tokens": max_tokens})
            return raw

        pipe = make_pipe(select_fn=None)  # CARDS scores dodge auto-pick
        pipe._labeler_chat = fake_labeler_chat
        res = pipe.run_appendix(Q, A, extract_fn=lambda q, a: ["ХСН"])
        return res, calls

    def test_select_schema_reasoning_before_picks(self):
        res, calls = self._select('{"reasoning": "разбор", "picks": ["2_1"]}')
        self.assertEqual(res.status, "ok")
        # two title groups -> two calls, each schema scoped to its group
        self.assertEqual(len(calls), 2)
        from medassistant import config
        for call in calls:
            self.assertEqual(list(call["schema"]["properties"]),
                             ["reasoning", "picks"])
            self.assertEqual(call["schema"]["required"],
                             ["reasoning", "picks"])
            reasoning = call["schema"]["properties"]["reasoning"]
            self.assertEqual(reasoning["maxLength"], 1400,
                             "grammar-bounded CoT")
            picks = call["schema"]["properties"]["picks"]
            enum = picks["items"]["enum"]
            self.assertEqual(len(enum), 1)  # one candidate per group here
            self.assertIn(enum[0], [c["code_version"] for c in CARDS])
            self.assertIn(enum[0], call["user"])
            self.assertEqual(picks["maxItems"], 1)
            self.assertEqual(call["max_tokens"],
                             config.LABELER_SELECT_MAX_TOKENS)
        # the lenient raw returns ["2_1"] for both calls; the other
        # group's call must not keep it (per-group pick filter)
        self.assertEqual(res.by_disease[0]["picks"][0]["code_version"], "2_1")
        self.assertEqual(len(res.by_disease[0]["picks"]), 1)

    def test_picks_only_output_still_parses(self):
        res, _ = self._select('{"picks": ["2_1"]}')
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.by_disease[0]["picks"][0]["code_version"], "2_1")


class TestSelectReasoningBudget(unittest.TestCase):
    """2026-10-20: the 700-char grammar cap could not mathematically fit a
    5-candidate enumeration (~140-173 chars per verdict item, five ≈ 730+) —
    the clamp slammed shut mid-list and shipped picks contradicting the lost
    verdicts. Cap raised to SELECT_REASONING_MAXLEN = 1400 and the verdict
    skeleton made terse («подходит или не подходит», no «почему») so a real
    enumeration lands ~500 chars, far under the cap."""

    def test_verdict_skeletons_are_terse(self):
        from medassistant import prompts
        from medassistant.pipeline import _select_user
        self.assertNotIn("почему", _select_user(Q, A, "ХСН", CARDS))
        self.assertNotIn("почему", prompts.SYSTEM_LABELER_SELECT)
        self.assertNotIn("почему", prompts.SYSTEM_LABELER_SELECT_DIRECT)
        self.assertIn("<подходит или не подходит>.",
                      _select_user(Q, A, "ХСН", CARDS))
        self.assertIn("<подходит или не подходит>",
                      prompts.SYSTEM_LABELER_SELECT)
        self.assertIn("<подходит или не подходит>",
                      prompts.SYSTEM_LABELER_SELECT_DIRECT)

    def _select_audited(self, raw):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            def fake_labeler_chat(system, user, max_tokens, schema):
                return raw

            pipe = make_pipe(tmp, select_fn=None)
            pipe._labeler_chat = fake_labeler_chat
            pipe.run_appendix(Q, A, extract_fn=lambda q, a: ["ХСН"])
            with open(Path(tmp) / "audit.jsonl", "r", encoding="utf-8") as f:
                recs = [json.loads(l) for l in f if l.strip()]
        return recs

    def test_reasoning_truncation_flag_sets_at_cap(self):
        from medassistant.pipeline import SELECT_REASONING_MAXLEN
        raw = ('{"reasoning": "' + "р" * SELECT_REASONING_MAXLEN + '", '
               '"picks": ["2_1"]}')
        recs = self._select_audited(raw)
        sel = [r for r in recs if r["stage"] == "labeler_select"]
        self.assertEqual(len(sel), 2, "one call per title group")
        self.assertTrue(all(r["reasoning_truncated"] for r in sel),
                        "clamp landed — greppable in audit")

    def test_reasoning_truncation_flag_absent_under_cap(self):
        raw = ('{"reasoning": "1. 2_1 — подходит. 2. 1_1 — не подходит.", '
               '"picks": ["2_1"]}')
        recs = self._select_audited(raw)
        sel = [r for r in recs if r["stage"] == "labeler_select"]
        self.assertEqual(len(sel), 2, "one call per title group")
        self.assertNotIn("reasoning_truncated", sel[0])

    def test_token_cap_lets_char_clamp_close_the_json(self):
        # 2026-10-21 canary: a degenerate short-item echo runs 1.75
        # chars/token (prose measured ~2.3), so 1400 chars can need 800+
        # tokens — the 768 cap chopped the string at 1347 chars BEFORE the
        # grammar clamp could close it (unclosed object, select_error).
        # The default must keep the char clamp authoritative (+ picks).
        from medassistant import config
        from medassistant.pipeline import SELECT_REASONING_MAXLEN
        self.assertGreaterEqual(
            config.LABELER_SELECT_MAX_TOKENS,
            SELECT_REASONING_MAXLEN / 1.75 + 60,
            "token cap must let the 1400-char grammar clamp close the JSON")


class TestSelectMatchedSharedTermSuppressed(unittest.TestCase):
    """2026-10-21: a matched term annotating several candidates is
    non-discriminative and fueled the select echo loop («Синдром
    «гистаминовой» головной боли» on 3 of 5 lines -> 5x echo in the
    reasoning; 23x without grammar, ~50x think-loop under the answerer
    template). Shared terms are printed on NO line; unique ones keep the
    2026-10-16 sound-alike protection."""

    def _user(self, cards):
        from medassistant.pipeline import _select_user
        return _select_user(Q, A, "ХСН", cards)

    def test_shared_term_dropped_everywhere(self):
        cards = [{"code_version": "1_1", "name": "Мигрень",
                  "matched": "Синдром «гистаминовой» головной боли"},
                 {"code_version": "2_1", "name": "ХСН",
                  "matched": "Синдром «гистаминовой» головной боли"}]
        user = self._user(cards)
        self.assertNotIn("совпавший термин", user)

    def test_unique_term_kept(self):
        cards = [{"code_version": "1_1", "name": "Мигрень",
                  "matched": "Полигландулярная гиперфункция"},
                 {"code_version": "2_1", "name": "ХСН",
                  "matched": "Сердечная недостаточность"}]
        user = self._user(cards)
        self.assertIn("1. 1_1 — Мигрень (совпавший термин: "
                      "«Полигландулярная гиперфункция»)", user)
        self.assertIn("2. 2_1 — ХСН (совпавший термин: "
                      "«Сердечная недостаточность»)", user)

    def test_mixed_shared_and_unique(self):
        cards = [{"code_version": "1_1", "name": "Мигрень",
                  "matched": "Общий термин"},
                 {"code_version": "2_1", "name": "ХСН",
                  "matched": "Общий термин"},
                 {"code_version": "3_1", "name": "Рак носоглотки",
                  "matched": "Редкий термин"}]
        user = self._user(cards)
        self.assertNotIn("Общий термин", user)
        self.assertIn("3. 3_1 — Рак носоглотки (совпавший термин: "
                      "«Редкий термин»)", user)

    def test_term_equal_to_own_name_not_counted_as_shared(self):
        # m == name never annotates its own line (2026-10-16 rule) and its
        # term counts once for the OTHER candidate: the parenthetical text
        # appears a single time in the prompt, so it keeps the sound-alike
        # signal («совпавший термин: «ХСН»» says this candidate surfaced
        # via text identical to 1_1's name)
        cards = [{"code_version": "1_1", "name": "ХСН", "matched": "ХСН"},
                 {"code_version": "2_1", "name": "СН (новая)",
                  "matched": "ХСН"}]
        user = self._user(cards)
        self.assertIn("1. 1_1 — ХСН\n", user)
        self.assertIn("2. 2_1 — СН (новая) (совпавший термин: «ХСН»)", user)

    def test_cards_without_matched_untouched(self):
        user = self._user(CARDS)
        self.assertNotIn("совпавший термин", user)
        self.assertIn("1. 2_1 — ХСН", user)
        self.assertIn("2. 1_1 — Грипп", user)


class TestSelectReasoningLoopGuard(unittest.TestCase):
    """2026-10-21: both live select degenerations were thinking loops in
    the answerer-watchdog sense — a verbatim tail-spin (loopguard reuse)
    and a numbered echo (same item body under «1.», «2.»…, which LoopGuard
    cannot see because the numbers break the back-to-back run; it also
    survives a cap-clamped tail — the clamp chops the final copy
    mid-word). A looped reasoning judged nothing: its picks are dropped
    (`labeler_select_loop` audit row), the title guard stays the only
    net, and nothing is retried."""

    def _run(self, raw, disease="мигрень"):
        # «мигрень» appears in no CARDS title -> the title guard cannot
        # rescue, so dropped selector picks are observable as picks: []
        import json
        with tempfile.TemporaryDirectory() as tmp:
            pipe = make_pipe(tmp, select_fn=None)
            pipe._labeler_chat = (
                lambda system, user, max_tokens, schema: raw)
            res = pipe.run_appendix(Q, A, extract_fn=lambda q, a: [disease])
            recs = [json.loads(l) for l in (Path(tmp) / "audit.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
            return res, recs

    def _picks_row(self, recs):
        return [r for r in recs if r["stage"] == "picks"][0]

    def test_numbered_echo_dropped(self):
        # the live 09:34 shape: identical item bodies, distinct «1.», «2.»…
        raw = ('{"reasoning": "1. 2_1 — ХСН — подходит. '
               '2. 2_1 — ХСН — подходит. 3. 2_1 — ХСН — подходит.", '
               '"picks": ["2_1"]}')
        res, recs = self._run(raw)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.by_disease, [])
        self.assertEqual(self._picks_row(recs)["picks"], [])
        loop = [r for r in recs if r["stage"] == "labeler_select_loop"]
        self.assertEqual(len(loop), 2, "one per title group")
        self.assertTrue(loop[0]["reason"].startswith("cycle"))
        self.assertEqual(loop[0]["dropped"], ["2_1"])

    def test_clamped_tail_spin_dropped(self):
        # the live post-fix shape: clean items then «5.» repeated to the
        # cap — the final copy is chopped mid-word, so only the
        # number-blind body run catches it
        item = "5. 2_1 — ХСН — подходит. "
        raw = ('{"reasoning": "1. 2_1 — ХСН — подходит. 2. 2_1 — ХСН — '
               'подходит. ' + item * 5 + '3. 2_1 — ХСН — подх", '
               '"picks": ["2_1"]}')
        res, recs = self._run(raw)
        self.assertEqual(res.by_disease, [])
        self.assertEqual(self._picks_row(recs)["picks"], [])
        self.assertEqual(
            len([r for r in recs if r["stage"] == "labeler_select_loop"]), 2,
            "one per title group")

    def test_verbatim_spin_dropped_via_loopguard(self):
        # a non-itemized tail-spin is LoopGuard's own case
        raw = ('{"reasoning": "подходит для лечения ХСН. '
               'подходит для лечения ХСН. подходит для лечения ХСН. '
               'подходит для лечения ХСН.", "picks": ["2_1"]}')
        res, recs = self._run(raw)
        self.assertEqual(res.by_disease, [])
        loop = [r for r in recs if r["stage"] == "labeler_select_loop"]
        self.assertEqual(len(loop), 2, "one per title group")
        self.assertTrue(loop[0]["reason"].startswith("unit"))

    def test_cyclic_enumeration_dropped(self):
        # live 2026-10-21, deduped-candidate probe: the model enumerated
        # the 3 candidates CORRECTLY but cyclically 4x over — [A,B,C]x4 —
        # consecutive identity cannot see it, block periodicity can
        raw = ('{"reasoning": "1. 2_1 — ХСН — подходит. 2. 1_1 — Грипп — '
               'подходит. 3. 3_1 — Рак — не подходит. '
               '4. 2_1 — ХСН — подходит. 5. 1_1 — Грипп — подходит. '
               '6. 3_1 — Рак — не подходит. '
               '7. 2_1 — ХСН — подходит. 8. 1_1 — Грипп — подходит. '
               '9. 3_1 — Рак — не подходит. '
               '10. 2_1 — ХСН — подходит. 11. 1_1 — Грипп — подходит. '
               '12. 3_1 — Рак — не подходит.", "picks": ["2_1"]}')
        res, recs = self._run(raw)
        self.assertEqual(res.by_disease, [])
        loop = [r for r in recs if r["stage"] == "labeler_select_loop"]
        self.assertEqual(len(loop), 2, "one per title group")
        self.assertTrue(loop[0]["reason"].startswith("cycle x4 (p=3)"))
        self.assertEqual(loop[0]["dropped"], ["2_1"])

    def test_healthy_distinct_list_not_flagged(self):
        # 5 distinct bodies, no block repeats — the shapes the detector
        # must never touch
        raw = ('{"reasoning": "1. 2_1 — ХСН — подходит. 2. 1_1 — Грипп — '
               'не подходит. 3. 3_1 — Рак носоглотки — не подходит. '
               '4. 4_1 — Отит — не подходит. 5. 5_1 — Гайморит — не '
               'подходит.", "picks": ["2_1"]}')
        res, recs = self._run(raw)
        self.assertEqual(res.by_disease[0]["picks"][0]["code_version"], "2_1")
        self.assertNotIn("labeler_select_loop", [r["stage"] for r in recs])

    def test_healthy_enumeration_kept(self):
        raw = ('{"reasoning": "1. 2_1 — ХСН — подходит. '
               '2. 1_1 — Грипп — не подходит.", "picks": ["2_1"]}')
        res, recs = self._run(raw)
        self.assertEqual(res.by_disease[0]["picks"][0]["code_version"], "2_1")
        self.assertNotIn("labeler_select_loop", [r["stage"] for r in recs])

    def test_identical_verdict_pair_not_flagged(self):
        # two identical terse bodies < 3 — an honest double-pick keeps its
        # picks (threshold 3 protects this shape)
        raw = ('{"reasoning": "1. 2_1 — подходит. 2. 1_1 — подходит.", '
               '"picks": ["2_1", "1_1"]}')
        res, recs = self._run(raw)
        self.assertEqual([p["code_version"] for p in res.by_disease[0]["picks"]],
                         ["2_1", "1_1"])
        self.assertNotIn("labeler_select_loop", [r["stage"] for r in recs])

    def test_picks_only_output_not_flagged(self):
        # no reasoning key (lenient parse, test fakes/CLI answer files) —
        # nothing to judge a loop by
        res, recs = self._run('{"picks": ["2_1"]}')
        self.assertEqual(res.by_disease[0]["picks"][0]["code_version"], "2_1")
        self.assertNotIn("labeler_select_loop", [r["stage"] for r in recs])

    def test_title_guard_still_rescues_after_loop(self):
        # the degradation ladder: selector drops its looped picks, the
        # title guard re-attaches the unique title match deterministically
        raw = ('{"reasoning": "1. 2_1 — ХСН — подходит. '
               '2. 2_1 — ХСН — подходит. 3. 2_1 — ХСН — подходит.", '
               '"picks": ["2_1"]}')
        res, recs = self._run(raw, disease="ХСН")
        self.assertEqual(res.by_disease[0]["picks"][0]["code_version"], "2_1")
        stages = [r["stage"] for r in recs]
        self.assertIn("labeler_select_loop", stages)
        self.assertIn("select_title_guard", stages)

    def test_real_2026_10_21_1407_turn_replayed(self):
        # byte-exact replay of the live 14:07 session-90 looped raw (audit
        # row 2026/10/21-14:08:58: items 2-5 echo 162_3's annotated line
        # under 810_1's id) against the REAL retrieval shape — real five
        # candidates, real matched terms, real ages. Scores are flattened
        # a little so the turn still EXERCISES the selector (no auto-pick
        # either way: the real gap is 0.0448 < 0.05). Age-era ladder: no
        # collapse (adult/pediatric pairs never merge), the selector sees
        # five age-marked lines, the echo raw is dropped by the loop
        # guard, and the title guard stays silent — BOTH ГБН titles
        # contain «головная боль» (one дети, one взрослые) -> ambiguous.
        import json
        cards = [
            {"code_version": "810_1", "name": "Головная боль напряжения",
             "score": 0.74, "matched": "Головная боль напряжения",
             "age": "Дети"},
            {"code_version": "162_3",
             "name": "Головная боль напряжения (ГБН)", "score": 0.72,
             "matched": "Головная боль напряженного типа",
             "age": "Взрослые"},
            {"code_version": "169_2", "name": "Мигрень", "score": 0.70,
             "matched": "Синдром «гистаминовой» головной боли",
             "age": "Дети"},
            {"code_version": "295_4", "name": "Мигрень", "score": 0.69,
             "matched": "Синдром «гистаминовой» головной боли",
             "age": "Взрослые"},
            {"code_version": "895_1",
             "name": "Краниальные мононейропатии у взрослых",
             "score": 0.68,
             "matched": "Синдром «гистаминовой» головной боли",
             "age": "Взрослые"},
        ]
        raw = ('\n\n{"reasoning": "1. 810_1 — Головная боль напряжения — '
               'подходит. 2. 810_1 — Головная боль напряжения (ГБН) '
               '(совпавший термин: «Головная боль напряженного типа») — '
               'подходит. 3. 810_1 — Головная боль напряжения (ГБН) '
               '(совпавший термин: «Головная боль напряженного типа») — '
               'подходит. 4. 810_1 — Головная боль напряжения (ГБН) '
               '(совпавший термин: «Головная боль напряженного типа») — '
               'подходит. 5. 810_1 — Головная боль напряжения (ГБН) '
               '(совпавший термин: «Головная боль напряженного типа») — '
               'подходит.", "picks": ["810_1"]}')
        seen = {"users": []}

        def fake_labeler_chat(system, user, max_tokens, schema):
            seen["users"].append(user)
            return raw

        with tempfile.TemporaryDirectory() as tmp:
            pipe = pipe_with_real_docs(tmp, cards, select_fn=None)
            pipe._labeler_chat = fake_labeler_chat
            res = pipe.run_appendix(
                "А просто при головной боли?", "Да.",
                extract_fn=lambda q, a: ["головная боль"])
            recs = [json.loads(l) for l in
                    (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8")
                    .splitlines()]
        stages = [r["stage"] for r in recs]
        # age era: the adult/pediatric pairs reach the selector intact
        self.assertNotIn("search_dedup", stages)
        # per-group calls (owner decision 2026-10-21): 3 title groups ->
        # 3 prompts; the adult/pediatric PAIR shares ONE call (the
        # audience choice is comparative), and each prompt lists only its
        # own group with visible ages
        users = seen["users"]
        self.assertEqual(len(users), 3)
        pair_user = [u for u in users if "810_1" in u and "162_3" in u]
        self.assertEqual(len(pair_user), 1, "the pair is judged together")
        self.assertIn("1. 810_1 — Головная боль напряжения (дети)",
                      pair_user[0])
        self.assertIn("2. 162_3 — Головная боль напряжения (ГБН) "
                      "(взрослые)", pair_user[0])
        # 162_3's matched term is unique inside the pair -> shown there;
        # the Мигрень pair shares гистаминовой -> suppressed; 895_1's
        # single-candidate call re-sole-owns it -> shown
        self.assertIn("(совпавший термин: «Головная боль напряженного "
                      "типа»)", pair_user[0])
        migraine_user = [u for u in users if "169_2" in u][0]
        self.assertNotIn("совпавший термин", migraine_user)
        mono_user = [u for u in users if "895_1" in u][0]
        self.assertIn("(совпавший термин: «Синдром «гистаминовой» головной "
                      "боли»)", mono_user)
        # the guard dropped the blind picks in every group; the title
        # guard stayed silent (two ГБН titles match -> ambiguous)
        self.assertIn("labeler_select_loop", stages)
        loops = [r for r in recs if r["stage"] == "labeler_select_loop"]
        self.assertEqual(len(loops), 3, "one per title group")
        self.assertTrue(loops[0]["reason"].startswith("cycle x4 (p=1)"))
        self.assertEqual(loops[0]["dropped"], ["810_1"])
        self.assertNotIn("select_title_guard", stages)
        self.assertEqual(res.by_disease, [])
        self.assertEqual(self._picks_row(recs)["picks"], [])


class TestSearchDedup(unittest.TestCase):
    """2026-10-21, reworked same day after the owner catch: the register's
    same-NAME pairs are ADULT/PEDIATRIC versions (810_1 «Головная боль
    напряжения» = Дети, 162_3 = Взрослые; 24 such pairs, zero same-name-
    same-audience docs) — a title-only collapse merged them and shipped
    the CHILDREN'S doc to an adult question. The collapse now keys on
    (title, audience): it merges nothing today, stays armed for future
    same-audience register duplicates. The age is enriched onto the cards
    from docs_meta and rendered in the select prompt so rule 4 can work
    and the lines of a pair are no longer identical-looking."""

    REAL_CARDS = [
        {"code_version": "810_1", "name": "Головная боль напряжения",
         "score": 0.8204, "matched": "Головная боль напряжения",
         "age": "Дети"},
        {"code_version": "162_3", "name": "Головная боль напряжения (ГБН)",
         "score": 0.7756, "matched": "Головная боль напряженного типа",
         "age": "Взрослые"},
        {"code_version": "169_2", "name": "Мигрень", "score": 0.6897,
         "matched": "Синдром «гистаминовой» головной боли", "age": "Дети"},
        {"code_version": "295_4", "name": "Мигрень", "score": 0.6897,
         "matched": "Синдром «гистаминовой» головной боли",
         "age": "Взрослые"},
        {"code_version": "895_1",
         "name": "Краниальные мононейропатии у взрослых", "score": 0.6897,
         "matched": "Синдром «гистаминовой» головной боли",
         "age": "Взрослые"},
    ]

    def _pipe_with_real_docs(self, tmp, cards, **kw):
        return pipe_with_real_docs(tmp, cards, **kw)

    def test_dedup_keys_on_audience(self):
        from medassistant.pipeline import _dedup_key
        # same title, different audience: NEVER merges (owner catch)
        self.assertNotEqual(
            _dedup_key("Головная боль напряжения (ГБН)", "Взрослые"),
            _dedup_key("Головная боль напряжения", "Дети"))
        # same title same audience (parentheticals stripped): merges —
        # armed for future register refreshes
        self.assertEqual(
            _dedup_key("Головная боль напряжения (ГБН)", "Взрослые"),
            _dedup_key("Головная боль напряжения", "Взрослые"))

    def test_real_turn_pair_completes_to_both(self):
        # the live 2026-10-21 turn, end to end with real ages: no
        # collapse, no auto-pick (0.0448 gap) — three per-group calls on
        # age-marked candidates; under the owner doctrine (2026-10-21
        # evening: unclear audience -> BOTH versions, the doctor chooses)
        # even an adult-only pick is completed with the детские twin,
        # and the appendix marks which one is pediatric
        import json
        users = []

        def fake(system, user, max_tokens, schema):
            users.append(user)
            if "162_3" in user:  # the ГБН pair call
                return ('{"reasoning": "1. 162_3 — Головная боль '
                        'напряжения (ГБН) (взрослые) — подходит. '
                        '2. 810_1 — Головная боль напряжения (дети) — '
                        'подходит.", "picks": ["162_3"]}')
            return ('{"reasoning": "не подходит.", "picks": []}')

        with tempfile.TemporaryDirectory() as tmp:
            pipe = self._pipe_with_real_docs(tmp, self.REAL_CARDS,
                                             select_fn=None)
            pipe._labeler_chat = fake
            res = pipe.run_appendix(
                "А просто при головной боли?", "Да.",
                extract_fn=lambda q, a: ["головная боль"])
            recs = [json.loads(l) for l in
                    (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8")
                    .splitlines()]
        self.assertEqual(res.status, "ok")
        self.assertFalse(res.by_disease[0]["auto_pick"])
        self.assertEqual([p["code_version"] for p in res.by_disease[0]["picks"]],
                         ["162_3", "810_1"])
        # the doctor sees which of the two is the pediatric version
        self.assertIn("(810_1, дети)", res.appendix_text)
        stages = [r["stage"] for r in recs]
        self.assertNotIn("search_dedup", stages)
        self.assertNotIn("labeler_select_loop", stages)
        self.assertIn("select_audience_complete", stages)
        # the pair was judged in ONE call, ages visible — rule 4 works
        self.assertEqual(len(users), 3)
        pair_user = [u for u in users if "810_1" in u and "162_3" in u]
        self.assertEqual(len(pair_user), 1)
        self.assertIn("1. 810_1 — Головная боль напряжения (дети)",
                      pair_user[0])
        self.assertIn("2. 162_3 — Головная боль напряжения (ГБН) "
                      "(взрослые)", pair_user[0])

    def test_cross_age_pair_never_collapses(self):
        # end to end: both ГБН docs reach the selector — in the SAME
        # per-group call (owner constraint: the audience choice is
        # comparative, the pair must not be split)
        import json
        users = []

        def fake(system, user, max_tokens, schema):
            users.append(user)
            return '{"picks": []}'

        with tempfile.TemporaryDirectory() as tmp:
            pipe = self._pipe_with_real_docs(tmp, self.REAL_CARDS,
                                             select_fn=None)
            pipe._labeler_chat = fake
            pipe.run_appendix("А просто при головной боли?", "Да.",
                              extract_fn=lambda q, a: ["головная боль"])
            recs = [json.loads(l) for l in
                    (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8")
                    .splitlines()]
        self.assertNotIn("search_dedup", [r["stage"] for r in recs])
        self.assertEqual(len(users), 3)
        self.assertEqual(len([u for u in users
                              if "810_1" in u and "162_3" in u]), 1,
                         "the adult/pediatric pair shares one call")

    def test_distinct_titles_still_select(self):
        # CARDS are distinct recommendations with a sub-gap dodge — no
        # collapse, no age in the fixture docs, no markers in the prompt
        import json
        with tempfile.TemporaryDirectory() as tmp:
            pipe = make_pipe(tmp, select_fn=None)
            pipe._labeler_chat = (
                lambda system, user, max_tokens, schema:
                '{"reasoning": "1. 2_1 — подходит. 2. 1_1 — не подходит.", '
                '"picks": ["2_1"]}')
            res = pipe.run_appendix(Q, A, extract_fn=lambda q, a: ["ХСН"])
            recs = [json.loads(l) for l in
                    (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8")
                    .splitlines()]
        self.assertEqual(res.by_disease[0]["picks"][0]["code_version"], "2_1")
        self.assertNotIn("search_dedup", [r["stage"] for r in recs])

    def test_same_age_true_duplicates_still_collapse(self):
        # the collapse stays armed: same title AND same audience (a
        # hypothetical future register artifact) keeps the first seen
        import json
        cards = [
            {"code_version": "295_4", "name": "Мигрень", "score": 0.7,
             "age": "Взрослые"},
            {"code_version": "169_2", "name": "Мигрень", "score": 0.7,
             "age": "Взрослые"},
        ]
        seen = {}

        def fake(system, user, max_tokens, schema):
            seen["user"] = user
            return ('{"reasoning": "1. 295_4 — Мигрень (взрослые) — '
                    'подходит.", "picks": ["295_4"]}')

        with tempfile.TemporaryDirectory() as tmp:
            pipe = self._pipe_with_real_docs(tmp, cards, select_fn=None)
            pipe._labeler_chat = fake
            res = pipe.run_appendix(Q, A, extract_fn=lambda q, a: ["мигрень"])
            recs = [json.loads(l) for l in
                    (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8")
                    .splitlines()]
        self.assertIn("295_4", seen["user"])
        self.assertNotIn("169_2", seen["user"])
        dedup = [r for r in recs if r["stage"] == "search_dedup"][0]
        self.assertEqual(dedup["dropped"], ["169_2"])
        self.assertEqual(res.by_disease[0]["picks"][0]["code_version"],
                         "295_4")


class TestAudienceComplete(unittest.TestCase):
    """Owner doctrine 2026-10-21 evening: when the turn names NO audience,
    a same-NAME adult/pediatric pair attaches BOTH versions — the doctor
    chooses himself. The 4B stamps one side unpredictably (live: ГБН pair
    returned only 810_1 дети, dropping 162_3 взрослые), so a half-picked
    pair is completed deterministically; any audience wording in the turn
    stands the guard down (rule 4 then decides)."""

    REAL_CARDS = TestSearchDedup.REAL_CARDS

    def _run(self, question, raw_by_group):
        import json
        calls = []

        def fake(system, user, max_tokens, schema):
            calls.append(user)
            for key, raw in raw_by_group.items():
                if key in user:
                    return raw
            return '{"picks": []}'

        with tempfile.TemporaryDirectory() as tmp:
            pipe = pipe_with_real_docs(tmp, self.REAL_CARDS, select_fn=None)
            pipe._labeler_chat = fake
            res = pipe.run_appendix(question, "Да.",
                                    extract_fn=lambda q, a: ["головная боль"])
            recs = [json.loads(l) for l in
                    (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8")
                    .splitlines()]
        return res, recs

    def test_half_picked_pair_completed(self):
        # the live failure shape: the ГБН group returns only 810_1 дети —
        # the adult twin 162_3 is added by the guard, audited
        res, recs = self._run("А просто при головной боли?", {
            "162_3": '{"reasoning": "…", "picks": ["810_1"]}'})
        self.assertEqual([p["code_version"] for p in res.by_disease[0]["picks"]],
                         ["810_1", "162_3"])
        comp = [r for r in recs if r["stage"] == "select_audience_complete"]
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0]["added"], ["162_3"])

    def test_reverse_direction_completed(self):
        # adult-only pick of a pair — the детские twin is added
        res, recs = self._run("А просто при головной боли?", {
            "162_3": '{"reasoning": "…", "picks": ["162_3"]}'})
        self.assertEqual([p["code_version"] for p in res.by_disease[0]["picks"]],
                         ["162_3", "810_1"])
        comp = [r for r in recs if r["stage"] == "select_audience_complete"]
        self.assertEqual(comp[0]["added"], ["810_1"])

    def test_full_pair_untouched(self):
        res, recs = self._run("А просто при головной боли?", {
            "162_3": '{"reasoning": "…", "picks": ["810_1", "162_3"]}'})
        self.assertNotIn("select_audience_complete",
                         [r["stage"] for r in recs])
        self.assertEqual(len(res.by_disease[0]["picks"]), 2)

    def test_child_marker_stands_the_guard_down(self):
        # the turn names its audience — rule 4 territory, no completion
        res, recs = self._run("Головная боль у ребёнка 7 лет?", {
            "162_3": '{"reasoning": "…", "picks": ["810_1"]}'})
        self.assertEqual([p["code_version"] for p in res.by_disease[0]["picks"]],
                         ["810_1"])
        self.assertNotIn("select_audience_complete",
                         [r["stage"] for r in recs])

    def test_adult_marker_stands_the_guard_down(self):
        res, recs = self._run("Головная боль у взрослого пациента?", {
            "162_3": '{"reasoning": "…", "picks": ["810_1"]}'})
        self.assertNotIn("select_audience_complete",
                         [r["stage"] for r in recs])

    def test_rejected_whole_recommendation_not_completed(self):
        # neither side picked -> the model rejected the recommendation;
        # the guard adds nothing
        res, recs = self._run("А просто при головной боли?", {
            "895_1": '{"reasoning": "…", "picks": ["895_1"]}'})
        picks = [p["code_version"] for p in res.by_disease[0]["picks"]]
        self.assertNotIn("810_1", picks)
        self.assertNotIn("162_3", picks)
        self.assertNotIn("select_audience_complete",
                         [r["stage"] for r in recs])


class TestCardContext(unittest.TestCase):
    ENDPOINT = {"base_url": "http://x/v1", "api_key": "k", "model": "m",
                "extra_body": {}}

    def test_card_block_lands_in_user_message(self):
        pipe = make_pipe()
        msgs = pipe._messages(Q, None, card=("Иванов ИИ", "текст карты"))
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1]["role"], "user")
        content = msgs[-1]["content"]
        self.assertTrue(content.startswith(
            f"{Q}\n\nАмбулаторная карта пациента Иванов ИИ"))
        self.assertIn("текст карты", content)

    def test_no_card_keeps_plain_question(self):
        pipe = make_pipe()
        msgs = pipe._messages(Q, [{"role": "user", "content": "было"}])
        self.assertEqual(msgs[-1]["content"], Q)
        self.assertEqual(msgs[1]["content"], "было")  # history untouched

    def test_stream_answer_forwards_card(self):
        from unittest.mock import patch
        pipe = make_pipe()
        pipe.candidate = self.ENDPOINT
        seen = {}

        def fake_stream(endpoint, messages, **kw):
            seen["messages"] = messages
            yield {"type": "content", "text": "ok"}

        with patch.object(llm, "chat_stream", fake_stream):
            events = list(pipe.stream_answer(Q, card=("Иванов ИИ", "текст")))
        self.assertEqual(events[-1]["text"], "ok")
        self.assertTrue(seen["messages"][-1]["content"].startswith(
            f"{Q}\n\nАмбулаторная карта"))

    def test_answer_forwards_card(self):
        from unittest.mock import patch
        pipe = make_pipe()
        pipe.candidate = self.ENDPOINT
        seen = {}

        def fake_chat(endpoint, messages, **kw):
            seen["messages"] = messages
            return "ok"

        with patch.object(llm, "chat", fake_chat):
            out = pipe.answer(Q, card=("Иванов ИИ", "текст"))
        self.assertEqual(out, "ok")
        self.assertIn("текст", seen["messages"][-1]["content"])


class TestDrugContext(unittest.TestCase):
    MATCHES = [{"trade": "Сорбифер Дурулес",
                "inn": "железа сульфат + аскорбиновая кислота",
                "token": "сорбифер",
                "en": "железа сульфат (ferrous sulfate) + аскорбиновая"
                      " кислота (ascorbic acid)"}]

    def test_drug_ps_lands_before_question_and_card(self):
        # owner decision 2026-10-17: the register note precedes the question
        # so the thinking's English paraphrase of «валидол» cannot outrun
        # the МНН binding (валидол -> "valium" -> diazepam pharmacology)
        pipe = make_pipe()
        msgs = pipe._messages(Q, None, card=("Иванов ИИ", "текст карты"),
                              drug=self.MATCHES)
        content = msgs[-1]["content"]
        ps = ("(справочно, из государственного реестра лекарственных средств:\n"
              "«сорбифер» — торговое наименование «Сорбифер Дурулес», МНН:"
              " железа сульфат (ferrous sulfate) + аскорбиновая кислота"
              " (ascorbic acid))")
        self.assertEqual(
            content,
            f"{ps}\n\n{Q}\n\nАмбулаторная карта пациента Иванов ИИ"
            " (приложена врачом; учитывай её при ответе):\n\n"
            "текст карты\n\n(конец карты)")

    def test_drug_ps_one_line_per_match_and_alias_via(self):
        pipe = make_pipe()
        matches = self.MATCHES + [
            {"trade": "Валерианы настойка",
             "inn": "валерианы лекарственной корневищ с корнями настойка",
             "via": "валерьянки"},
            {"trade": "Кагоцел", "inn": "меглюмина акридонацетат"}]
        content = pipe._messages(Q, None, drug=matches)[-1]["content"]
        self.assertIn(
            "(справочно, из государственного реестра лекарственных средств:\n"
            "«сорбифер» — торговое наименование «Сорбифер Дурулес», МНН:"
            " железа сульфат (ferrous sulfate) + аскорбиновая кислота"
            " (ascorbic acid);\n"
            "«валерьянки» — разговорное название, «Валерианы настойка»"
            " — торговое наименование, МНН: валерианы лекарственной"
            " корневищ с корнями настойка;\n"
            "«Кагоцел» — торговое наименование, МНН: меглюмина"
            " акридонацетат)", content)

    def test_drug_ps_without_card(self):
        pipe = make_pipe()
        content = pipe._messages(Q, None, drug=self.MATCHES)[-1]["content"]
        self.assertTrue(content.startswith("(справочно"))
        self.assertTrue(content.endswith(f"{Q}"))
        self.assertIn(
            "МНН: железа сульфат (ferrous sulfate) + аскорбиновая кислота"
            " (ascorbic acid)", content)

    def test_placeholder_only_match_adds_nothing(self):
        pipe = make_pipe()
        content = pipe._messages(
            Q, None, drug=[{"trade": "Рузам концентрат", "inn": None}])[-1]["content"]
        self.assertEqual(content, Q)

    def test_empty_drug_keeps_plain_question(self):
        pipe = make_pipe()
        self.assertEqual(pipe._messages(Q, None, drug=[])[-1]["content"], Q)

    def test_stream_answer_forwards_drug(self):
        from unittest.mock import patch
        pipe = make_pipe()
        pipe.candidate = {"base_url": "http://x/v1", "api_key": "k",
                          "model": "m", "extra_body": {}}
        seen = {}

        def fake_stream(endpoint, messages, **kw):
            seen["messages"] = messages
            yield {"type": "content", "text": "ok"}

        with patch.object(llm, "chat_stream", fake_stream):
            events = list(pipe.stream_answer(Q, drug=self.MATCHES))
        self.assertEqual(events[-1]["text"], "ok")
        self.assertIn("справочно", seen["messages"][-1]["content"])



    def test_answerer_prompt_stays_slim(self):
        # 2026-10-25 aggressive slim: 1722 -> ~1000 chars; МНН 7 -> 3
        # mentions (rules-check echo fuel, MedGemma runs on one-liner
        # system prompts). Guards against re-fattening.
        from medassistant import prompts
        self.assertLessEqual(len(prompts.SYSTEM_ANSWERER), 1100)
        self.assertLessEqual(prompts.SYSTEM_ANSWERER.count("МНН"), 3)

class TestPromptSilence(unittest.TestCase):
    def test_answerer_prompt_is_silent_about_machinery(self):
        from medassistant import prompts
        for banned in ("Заболевание:", "сноск", "приложение", "appendix"):
            self.assertNotIn(banned, prompts.SYSTEM_ANSWERER)

    def test_labeler_prompts_demand_json(self):
        from medassistant import prompts
        for p in (prompts.SYSTEM_LABELER_EXTRACT,
                  prompts.SYSTEM_LABELER_EXTRACT_QUESTION,
                  prompts.SYSTEM_LABELER_SELECT):
            self.assertIn("JSON", p)

    def test_extract_examples_are_register_absent(self):
        # 2026-10-25 owner mitigation: extract prompt examples are echoed by
        # the 4B on administrative answers — an example present in the
        # register scores >= AUTO_PICK_MIN in search and the echoed example
        # ships a wrong pdf with no labeler-2 check (ХСН -> 156_2 0.867,
        # live 2026-10-25); the three register-present names must not creep
        # back; the specific replacement terms stay unpinned (cheap swaps)
        from medassistant import prompts
        for p in (prompts.SYSTEM_LABELER_EXTRACT,
                  prompts.SYSTEM_LABELER_EXTRACT_QUESTION):
            for name in ("железодефицитная анемия",
                         "хроническая сердечная недостаточность",
                         "диабетическая полинейропатия"):
                self.assertNotIn(name, p)

    def test_extract_denylist_matches_examples(self):
        # owner 2026-10-25 mitigation: the denylist must cover the RU terms
        # living in the extract prompts (examples <-> denylist sync); EN
        # equivalents ride along uncovered by this check — they cannot
        # appear in a Russian prompt
        from medassistant import prompts
        body = (prompts.SYSTEM_LABELER_EXTRACT
                + prompts.SYSTEM_LABELER_EXTRACT_QUESTION
                ).lower().replace("ё", "е")
        for term in prompts.EXTRACT_DENYLIST:
            if term.isascii():
                continue
            self.assertIn(term, body)

    def test_extract_answer_pass_short_circuits(self):
        # owner decision 2026-10-25 evening: the bare-question pass is a
        # FALLBACK for empty answers only — a non-empty answer-anchored
        # extraction is the whole result, the question pass never runs
        # (speed: one extract call on a normal turn; near-form duplicates
        # like «инфаркт миокарда»+«инфаркт» no longer double the machinery)
        from medassistant import prompts
        pipe = make_pipe()
        calls = []

        def fake_call(system, user, manual=False, fallback=False):
            calls.append(system)
            self.assertIs(system, prompts.SYSTEM_LABELER_EXTRACT)
            self.assertIn("Ответ ассистента", user)
            return ["артериальная гипертензия"]

        pipe._extract_call = fake_call
        self.assertEqual(
            pipe._real_extract("Вопрос про гипертонию и диабет?",
                               "Ответ про гипертензию."),
            ["артериальная гипертензия"])
        self.assertEqual(calls, [prompts.SYSTEM_LABELER_EXTRACT])

    def test_extract_question_fallback_on_empty_answer(self):
        # «Да.» to «А при инсульте?» — the answer names no disease, the
        # bare question is analyzed with the direct prompt (the original
        # 2026-10-20 case, e77ef04, now code-level); the fallback call is
        # marked so audit rows distinguish it
        from medassistant import prompts
        pipe = make_pipe()

        def fake_call(system, user, manual=False, fallback=False):
            if system is prompts.SYSTEM_LABELER_EXTRACT:
                self.assertFalse(fallback)
                return []
            self.assertIs(system, prompts.SYSTEM_LABELER_EXTRACT_QUESTION)
            self.assertNotIn("Ответ ассистента", user)
            self.assertEqual(user, "А при инсульте?")
            self.assertTrue(fallback)
            return ["инсульт"]

        pipe._extract_call = fake_call
        self.assertEqual(pipe._real_extract("А при инсульте?", "Да."),
                         ["инсульт"])

    def test_extract_passes_answer_anchor_prompt(self):
        # the answer pass stays the proven MVP answer-anchor — question-only
        # diseases belong to the separate direct pass
        from medassistant import prompts
        self.assertIn("фактически идет речь в ответе",
                      prompts.SYSTEM_LABELER_EXTRACT)
        self.assertNotIn("фактически идет речь в ответе",
                         prompts.SYSTEM_LABELER_EXTRACT_QUESTION)


class TestStreamAnswerLoopRetry(unittest.TestCase):
    """The thinking-loop watchdog retry (owner decision 2026-10-06):
    LLMLoopError -> one retry with the anti-repetition nudge, signalled to
    the consumer as a reasoning_reset event."""

    EP = {"base_url": "http://x", "api_key": "", "model": "m", "extra_body": {}}

    def setUp(self):
        from medassistant import config
        self.config = config
        self._orig = (config.LOOP_RETRIES, config.LOOP_RETRY_FREQ_PENALTY,
                      config.LOOP_RETRY_DRY_MULTIPLIER,
                      config.LOOP_RETRY_DRY_BASE,
                      config.LOOP_RETRY_DRY_ALLOWED_LENGTH,
                      config.LOOP_RETRY_DRY_PENALTY_LAST_N,
                      config.LOOP_RETRY_DRY_BREAKERS)
        (config.LOOP_RETRIES, config.LOOP_RETRY_FREQ_PENALTY,
         config.LOOP_RETRY_DRY_MULTIPLIER, config.LOOP_RETRY_DRY_BASE,
         config.LOOP_RETRY_DRY_ALLOWED_LENGTH,
         config.LOOP_RETRY_DRY_PENALTY_LAST_N,
         config.LOOP_RETRY_DRY_BREAKERS) = 1, 0.5, 0.8, 1.75, 2, 64, "\u241e"
        self._orig_stream = llm.chat_stream
        self.calls = []
        self.dry_calls = []

    def tearDown(self):
        (self.config.LOOP_RETRIES, self.config.LOOP_RETRY_FREQ_PENALTY,
         self.config.LOOP_RETRY_DRY_MULTIPLIER,
         self.config.LOOP_RETRY_DRY_BASE,
         self.config.LOOP_RETRY_DRY_ALLOWED_LENGTH,
         self.config.LOOP_RETRY_DRY_PENALTY_LAST_N,
         self.config.LOOP_RETRY_DRY_BREAKERS) = self._orig
        llm.chat_stream = self._orig_stream

    def _fake_chat_stream(self, plans):
        """plans: list of per-attempt behaviors — event list or exception."""
        seq = list(plans)

        def fake(endpoint, messages, max_tokens=None, freq_penalty=None,
                 dry=None, **kw):
            self.calls.append(freq_penalty)
            self.dry_calls.append(dry)
            plan = seq.pop(0)
            if isinstance(plan, Exception):
                raise plan
            yield from plan

        llm.chat_stream = fake

    def _pipe(self, audit=None):
        return Pipeline(candidate=self.EP, audit=audit)

    def test_loop_aborts_then_retries_with_nudge(self):
        self._fake_chat_stream([
            llm.LLMLoopError("degenerate reasoning loop: tail x3"),
            [{"type": "reasoning", "text": "заново"},
             {"type": "content", "text": "Ответ."}],
        ])
        events = list(self._pipe().stream_answer(Q))
        self.assertEqual(events, [
            {"type": "reasoning_reset"},
            {"type": "reasoning", "text": "заново"},
            {"type": "content", "text": "Ответ."},
        ])
        self.assertEqual(self.calls, [None, 0.5])

    def test_retry_sends_dry_from_config(self):
        # 2026-10-24: the loop-retry attempt carries per-request dry_* built
        # from config at call time (patched knobs honored); attempt-0
        # receives None — its payload stays byte-identical
        expected = {"dry_multiplier": 0.8, "dry_base": 1.75,
                    "dry_allowed_length": 2, "dry_penalty_last_n": 64,
                    "dry_sequence_breakers": ["\u241e"]}
        self._fake_chat_stream([
            llm.LLMLoopError("degenerate reasoning loop"),
            [{"type": "content", "text": "Ответ."}],
        ])
        list(self._pipe().stream_answer(Q))
        self.assertEqual(self.dry_calls, [None, expected])

    def test_retry_dry_disabled_by_zero_multiplier(self):
        self.config.LOOP_RETRY_DRY_MULTIPLIER = 0
        self._fake_chat_stream([
            llm.LLMLoopError("loop 1"),
            [{"type": "content", "text": "Ответ."}],
        ])
        list(self._pipe().stream_answer(Q))
        self.assertEqual(self.dry_calls, [None, None])

    def test_second_loop_raises(self):
        self._fake_chat_stream([
            llm.LLMLoopError("loop 1"),
            llm.LLMLoopError("loop 2"),
        ])
        with self.assertRaises(llm.LLMLoopError):
            list(self._pipe().stream_answer(Q))
        self.assertEqual(self.calls, [None, 0.5])

    def test_no_retry_when_disabled(self):
        self.config.LOOP_RETRIES = 0
        self._fake_chat_stream([llm.LLMLoopError("loop")])
        with self.assertRaises(llm.LLMLoopError):
            list(self._pipe().stream_answer(Q))
        self.assertEqual(self.calls, [None])

    def test_audit_logs_abort(self):
        import json
        self._fake_chat_stream([
            llm.LLMLoopError("degenerate reasoning loop"),
            [{"type": "content", "text": "Ответ."}],
        ])
        with tempfile.TemporaryDirectory() as tmp:
            audit = Audit(Path(tmp) / "audit.jsonl")
            list(self._pipe(audit=audit).stream_answer(Q))
            recs = [json.loads(l) for l in
                    (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8")
                    .splitlines()]
        self.assertEqual([r["stage"] for r in recs], ["answerer_loop_abort"])
        self.assertEqual(recs[0]["attempt"], 1)
        self.assertEqual(recs[0]["retry_dry"], 0.8)

    def test_audit_abort_row_carries_tail(self):
        # 2026-10-24: the looped thinking survives in the audit row — the
        # raise site (llm.chat_stream) attaches the guard's tail to the
        # exception; constructed here exactly as it does
        import json
        looped = "зацикленный фрагмент размышлений. " * 5
        self._fake_chat_stream([
            llm.LLMLoopError("degenerate reasoning loop: unit x3",
                             tail=looped),
            [{"type": "content", "text": "Ответ."}],
        ])
        with tempfile.TemporaryDirectory() as tmp:
            audit = Audit(Path(tmp) / "audit.jsonl")
            list(self._pipe(audit=audit).stream_answer(Q))
            recs = [json.loads(l) for l in
                    (Path(tmp) / "audit.jsonl").read_text(encoding="utf-8")
                    .splitlines()]
        self.assertEqual([r["stage"] for r in recs], ["answerer_loop_abort"])
        self.assertEqual(recs[0]["tail"], looped)
        self.assertIn("degenerate reasoning loop", recs[0]["error"])

    def test_clean_stream_has_no_reset(self):
        self._fake_chat_stream([
            [{"type": "reasoning", "text": "мысль"},
             {"type": "content", "text": "Ответ."}],
        ])
        events = list(self._pipe().stream_answer(Q))
        self.assertEqual(events, [
            {"type": "reasoning", "text": "мысль"},
            {"type": "content", "text": "Ответ."},
        ])
        self.assertEqual(self.calls, [None])


class TestStreamAnswerCancel(unittest.TestCase):
    """stream_answer(cancel=...) passthrough (owner decision 2026-10-15): a
    user stop reaches llm.chat_stream and raises LLMCancelled out of the
    generator — never caught by the thinking-loop retry."""

    EP = {"base_url": "http://x", "api_key": "", "model": "m", "extra_body": {}}

    def test_cancel_passthrough_raises_without_loop_retry(self):
        from unittest.mock import patch
        pipe = Pipeline(candidate=self.EP)
        cancel = llm.StreamCancel()
        calls = []

        def fake_stream(endpoint, messages, max_tokens=None, freq_penalty=None,
                        cancel=None, **kw):
            calls.append(freq_penalty)
            yield {"type": "content", "text": "частичный"}
            raise llm.LLMCancelled("cancelled by user")

        with patch.object(llm, "chat_stream", fake_stream):
            events = []
            with self.assertRaises(llm.LLMCancelled):
                for ev in pipe.stream_answer(Q, cancel=cancel):
                    events.append(ev)
        self.assertEqual(events, [{"type": "content", "text": "частичный"}])
        self.assertEqual(calls, [None], "a cancelled turn is never retried")


if __name__ == "__main__":
    unittest.main()


class TestExtractDenylist(unittest.TestCase):
    """Pre-retrieval denylist (owner decision 2026-10-25): the 4B extract
    dominantly echoes its own prompt examples on administrative answers —
    an echo must never reach search/auto-pick/select."""

    @staticmethod
    def _rows(tmp):
        import json
        return [json.loads(l) for l in
                Path(tmp, "audit.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]

    def test_denied_term_dropped_and_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = make_pipe(tmp=tmp)
            self.assertEqual(pipe._apply_extract_denylist(
                ["Синдром Жильбера", " Gilbert Syndrome "]), [])
            denied = [r for r in self._rows(tmp)
                      if r["stage"] == "extract_denied"]
            self.assertEqual(len(denied), 1)
            self.assertEqual(denied[0]["dropped"],
                             ["Синдром Жильбера", " Gilbert Syndrome "])

    def test_mixed_list_keeps_real_diseases(self):
        pipe = make_pipe()
        self.assertEqual(
            pipe._apply_extract_denylist(
                ["инфаркт миокарда", "синдром Марфана", "стенокардия"]),
            ["инфаркт миокарда", "стенокардия"])

    def test_all_denied_becomes_empty(self):
        pipe = make_pipe()
        self.assertEqual(pipe._apply_extract_denylist(
            ["наследственный гемохроматоз", "marfan syndrome"]), [])

    def test_every_extract_pass_filtered(self):
        # the filter lives in _extract_call, so BOTH _real_extract passes
        # go through it: the answer pass here is denied down to [] and the
        # empty-answer FALLBACK pass must be filtered too (2026-10-25
        # fallback semantics)
        from medassistant import prompts
        pipe = make_pipe()

        def fake_chat(system, user, max_tokens, schema):
            if system is prompts.SYSTEM_LABELER_EXTRACT:
                return '{"diseases": ["Marfan syndrome"]}'
            return ('{"diseases": ["инфаркт миокарда",'
                    ' "синдром Марфана"]}')
        pipe._labeler_chat = fake_chat
        self.assertEqual(pipe._real_extract("вопрос", "ответ"),
                         ["инфаркт миокарда"])

    def test_manual_pdf_path_filtered_to_no_disease(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = make_pipe(tmp=tmp)

            def fake_chat(system, user, max_tokens, schema):
                return '{"diseases": ["синдром Жильбера", "Гемохроматоз"]}'
            pipe._labeler_chat = fake_chat
            self.assertEqual(pipe.extract_manual("pdf синдром Жильбера"), [])
            events = list(pipe.iter_appendix("синдром Жильбера", "",
                                             manual=True))
            result = [e for e in events if e.get("stage") == "done"][0]["result"]
            self.assertEqual(result.status, "no_disease")
            stages = [r["stage"] for r in self._rows(tmp)]
            self.assertNotIn("search", stages)   # never reached retrieval


class TestSelectCacheWiring(unittest.TestCase):
    """Pair-verdict cache (owner decision 2026-10-26): a settled selection
    freezes per-pair verdicts keyed (disease, audience bucket); a fully
    covered disease skips labeler 2 entirely; a WRONG pair never attaches
    (it renders grayed among the filtered-out candidates); only
    session-carrying turns read or write (regression/canary/CLI bypass);
    looped groups are never frozen."""

    def _cache(self, tmp):
        from medassistant.select_cache import SelectCache
        return SelectCache(Path(tmp) / "select_cache.yaml", "stamp")

    def _run(self, pipe, sid=7, question=Q, answer=A, disease="ХСН"):
        events = list(pipe.iter_appendix(question, answer, session_id=sid,
                                         diseases=[disease]))
        res = [e for e in events if e.get("stage") == "done"][0]["result"]
        return res, [e.get("stage") for e in events if e.get("stage") != "done"]

    def test_sid_none_bypasses_read_and_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put_machine("ХСН", "none", {"2_1": "right", "1_1": "wrong"})
            called = []
            pipe = make_pipe(tmp, select_cache=cache,
                             select_fn=lambda q, a, d, c:
                             called.append([x["code_version"] for x in c])
                             or ["2_1"])
            res, _ = self._run(pipe, sid=None)
            self.assertEqual(called, [["2_1", "1_1"]],
                             "no sid -> full live select, cache ignored")
            self.assertEqual(cache.get("ХСН", "none", "2_1"),
                             {"verdict": "right"},
                             "no sid -> nothing rewritten")

    def test_full_coverage_hit_skips_labeler_and_autopick(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put_machine("ХСН", "none", {"2_1": "right", "1_1": "wrong"})
            called = []
            pipe = make_pipe(tmp, select_cache=cache,
                             select_fn=lambda q, a, d, c:
                             called.append(1) or ["2_1"])
            res, stages = self._run(pipe)
            self.assertEqual(called, [], "labeler 2 must not run")
            self.assertIn("select_cache", stages)
            self.assertNotIn("select", stages)
            self.assertNotIn("auto_pick", stages)
            g = res.by_disease[0]
            self.assertEqual([p["code_version"] for p in g["picks"]], ["2_1"])
            self.assertTrue(g["cached"])
            self.assertEqual(g["audience"], "none")
            self.assertEqual([(c["code_version"], c["audience"])
                              for c in res.filtered_out],
                             [("1_1", "none")], "wrong pair shows grayed")

    def test_partial_coverage_selects_unknowns_only_and_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards = [{"code_version": "2_1", "name": "ХСН", "score": 0.90},
                     {"code_version": "1_1", "name": "Грипп", "score": 0.84},
                     {"code_version": "3_1", "name": "Рак носоглотки",
                      "score": 0.83}]
            cache = self._cache(tmp)
            cache.put_machine("ХСН", "none", {"2_1": "right"})
            called = []
            pipe = make_pipe(tmp, select_cache=cache,
                             search_fn=lambda v: list(cards),
                             select_fn=lambda q, a, d, c:
                             called.append([x["code_version"] for x in c])
                             or ["1_1"])
            res, stages = self._run(pipe)
            self.assertEqual(called, [["1_1", "3_1"]],
                             "only unknown pairs are judged")
            self.assertIn("select", stages)
            self.assertEqual([p["code_version"] for p in
                              res.by_disease[0]["picks"]],
                             ["2_1", "1_1"],
                             "cached-right pair attaches without judging")
            self.assertEqual(cache.get("ХСН", "none", "1_1"),
                             {"verdict": "right"})
            self.assertEqual(cache.get("ХСН", "none", "3_1"),
                             {"verdict": "wrong"})

    def test_human_wrong_vetoes_dominant_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards = [{"code_version": "2_1", "name": "ХСН", "score": 0.95},
                     {"code_version": "1_1", "name": "Грипп", "score": 0.70}]
            cache = self._cache(tmp)
            cache.negate("ХСН", "none", "2_1", "right")  # human: wrong
            called = []
            pipe = make_pipe(tmp, select_cache=cache,
                             search_fn=lambda v: list(cards),
                             select_fn=lambda q, a, d, c:
                             called.append([x["code_version"] for x in c])
                             or [])
            res, stages = self._run(pipe)
            self.assertIn("select", stages)
            self.assertNotIn("auto_pick", stages,
                             "the vetoed top-1 must not auto-attach")
            self.assertEqual(called, [["1_1"]])
            self.assertEqual(res.by_disease, [])
            self.assertEqual([c["code_version"] for c in res.filtered_out],
                             ["2_1", "1_1"],
                             "the disliked pair stays visible, grayed")
            self.assertEqual(cache.get("ХСН", "none", "2_1"),
                             {"verdict": "wrong", "human": True},
                             "machine write must not overwrite the human")

    def test_looped_group_not_frozen(self):
        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            raw = ('{"reasoning": "1. 2_1 — ХСН — подходит. '
                   '2. 2_1 — ХСН — подходит. 3. 2_1 — ХСН — подходит.", '
                   '"picks": ["2_1"]}')
            pipe = make_pipe(tmp, select_cache=cache, select_fn=None)
            pipe._labeler_chat = (
                lambda system, user, max_tokens, schema: raw)
            res, _ = self._run(pipe)
            self.assertEqual(res.status, "ok")
            rows = [_json.loads(l) for l in
                    (Path(tmp) / "audit.jsonl").read_text(
                        encoding="utf-8").splitlines() if l.strip()]
            self.assertTrue([r for r in rows
                             if r["stage"] == "labeler_select_loop"])
            self.assertIsNone(cache.get("ХСН", "none", "2_1"),
                              "a degenerate judgment must not freeze")
            self.assertIsNone(cache.get("ХСН", "none", "1_1"))

    def test_audience_bucket_in_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put_machine("ХСН", "none", {"2_1": "right", "1_1": "wrong"})
            called = []
            pipe = make_pipe(tmp, select_cache=cache,
                             select_fn=lambda q, a, d, c:
                             called.append(1) or ["2_1"])
            res, stages = self._run(
                pipe, question="Что при ХСН у ребёнка?",
                answer="Ответ про ХСН.")
            self.assertEqual(called, [1],
                             "the none-bucket verdicts must NOT be consumed "
                             "under the child key — select runs live")
            self.assertNotIn("select_cache", stages)
            self.assertEqual(res.by_disease[0]["audience"], "child")
            self.assertEqual(cache.get("ХСН", "child", "2_1"),
                             {"verdict": "right"},
                             "the live turn froze verdicts under child")
            self.assertEqual(cache.get("ХСН", "none", "1_1"),
                             {"verdict": "wrong"},
                             "the none bucket is untouched")

    def test_provisional_cards_carry_known_verdicts(self):
        # partial coverage (owner catch 2026-10-26 evening): the candidates
        # block streamed BEFORE the select must show decided pairs in their
        # decided state — the 👎'd pair grays at once, the cached-right pair
        # is marked; only unknowns carry no verdict; the STORED payload
        # (filtered_out) stays unannotated
        with tempfile.TemporaryDirectory() as tmp:
            cards = [{"code_version": "2_1", "name": "ХСН", "score": 0.90},
                     {"code_version": "1_1", "name": "Грипп", "score": 0.88},
                     {"code_version": "3_1", "name": "Рак носоглотки",
                      "score": 0.70}]
            cache = self._cache(tmp)
            cache.put_machine("ХСН", "none", {"2_1": "right"})
            cache.negate("ХСН", "none", "1_1", "right")  # human: wrong
            called = []
            pipe = make_pipe(tmp, select_cache=cache,
                             search_fn=lambda v: list(cards),
                             select_fn=lambda q, a, d, c:
                             called.append([x["code_version"] for x in c])
                             or [])
            events = list(pipe.iter_appendix(Q, A, session_id=7,
                                             diseases=["ХСН"]))
            ev = [e for e in events if e.get("stage") == "select"][0]
            by_cv = {c["code_version"]: c for c in ev["cards"]}
            self.assertEqual(by_cv["2_1"].get("verdict"), "right")
            self.assertEqual(by_cv["1_1"].get("verdict"), "wrong")
            self.assertNotIn("verdict", by_cv["3_1"])
            self.assertEqual(called, [["3_1"]], "only the unknown is judged")
            res = [e for e in events if e.get("stage") == "done"][0]["result"]
            for f in res.filtered_out:
                self.assertNotIn("verdict", f, "stored payload stays clean")

    def test_sid_none_select_event_unannotated(self):
        # the sid bypass covers the streamed cards too — no verdict keys
        # leak into the event when the cache is not consulted at all
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            cache.put_machine("ХСН", "none", {"2_1": "right", "1_1": "wrong"})
            pipe = make_pipe(tmp, select_cache=cache, select_fn=None)
            pipe._labeler_chat = (
                lambda system, user, max_tokens, schema:
                '{"reasoning": "1. 2_1 — подходит.", "picks": ["2_1"]}')
            events = list(pipe.iter_appendix(Q, A, session_id=None,
                                             diseases=["ХСН"]))
            ev = [e for e in events if e.get("stage") == "select"][0]
            for c in ev["cards"]:
                self.assertNotIn("verdict", c)

    def test_machine_writes_after_live_select(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._cache(tmp)
            pipe = make_pipe(tmp, select_cache=cache,
                             select_fn=lambda q, a, d, c: ["2_1"])
            self._run(pipe)
            self.assertEqual(cache.get("ХСН", "none", "2_1"),
                             {"verdict": "right"})
            self.assertEqual(cache.get("ХСН", "none", "1_1"),
                             {"verdict": "wrong"})

    def test_auto_pick_turn_writes_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards = [{"code_version": "2_1", "name": "ХСН", "score": 0.90},
                     {"code_version": "1_1", "name": "Грипп", "score": 0.70}]
            cache = self._cache(tmp)
            called = []
            pipe = make_pipe(tmp, select_cache=cache,
                             search_fn=lambda v: list(cards),
                             select_fn=lambda q, a, d, c:
                             called.append(1) or [])
            res, stages = self._run(pipe)
            self.assertIn("auto_pick", stages)
            self.assertEqual(called, [])
            self.assertEqual(cache.get("ХСН", "none", "2_1"),
                             {"verdict": "right"})
            self.assertEqual(cache.get("ХСН", "none", "1_1"),
                             {"verdict": "wrong"})

    def test_cache_disabled_is_byte_identical_path(self):
        # cache=None (SELECT_CACHE=0) must leave the classic flow intact:
        # select sees the full card list, no select_cache stage, no write
        with tempfile.TemporaryDirectory() as tmp:
            called = []
            pipe = make_pipe(tmp,
                             select_fn=lambda q, a, d, c:
                             called.append([x["code_version"] for x in c])
                             or ["2_1"])
            res, stages = self._run(pipe)
            self.assertEqual(called, [["2_1", "1_1"]])
            self.assertNotIn("select_cache", stages)
            self.assertFalse(res.by_disease[0]["cached"])

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
        self.assertIs(seen["system"], prompts.SYSTEM_LABELER_EXTRACT_DIRECT)
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


class TestPromptSilence(unittest.TestCase):
    def test_answerer_prompt_is_silent_about_machinery(self):
        from medassistant import prompts
        for banned in ("Заболевание:", "сноск", "приложение", "appendix"):
            self.assertNotIn(banned, prompts.SYSTEM_ANSWERER)

    def test_labeler_prompts_demand_json(self):
        from medassistant import prompts
        for p in (prompts.SYSTEM_LABELER_EXTRACT,
                  prompts.SYSTEM_LABELER_EXTRACT_DIRECT,
                  prompts.SYSTEM_LABELER_SELECT):
            self.assertIn("JSON", p)


class TestStreamAnswerLoopRetry(unittest.TestCase):
    """The thinking-loop watchdog retry (owner decision 2026-10-06):
    LLMLoopError -> one retry with the anti-repetition nudge, signalled to
    the consumer as a reasoning_reset event."""

    EP = {"base_url": "http://x", "api_key": "", "model": "m", "extra_body": {}}

    def setUp(self):
        from medassistant import config
        self.config = config
        self._orig = (config.LOOP_RETRIES, config.LOOP_RETRY_FREQ_PENALTY)
        config.LOOP_RETRIES, config.LOOP_RETRY_FREQ_PENALTY = 1, 0.5
        self._orig_stream = llm.chat_stream
        self.calls = []

    def tearDown(self):
        self.config.LOOP_RETRIES, self.config.LOOP_RETRY_FREQ_PENALTY = self._orig
        llm.chat_stream = self._orig_stream

    def _fake_chat_stream(self, plans):
        """plans: list of per-attempt behaviors — event list or exception."""
        seq = list(plans)

        def fake(endpoint, messages, max_tokens=None, freq_penalty=None, **kw):
            self.calls.append(freq_penalty)
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


if __name__ == "__main__":
    unittest.main()

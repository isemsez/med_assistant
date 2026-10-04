"""Orchestrator: answerer -> labeler 1 (extract) -> tool -> labeler 2 (select)
-> renderer, with stage-stops and honest notes (architecture.md §3, §5).

No semantic retries: an invalid/empty labeler output or a dead embed endpoint
stops the pipeline with an error note; the raw answer is never modified.
Stage functions are injectable for offline tests and CLI mock runs.
"""

import json
import time

from . import appendix, config, llm, prompts

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {"diseases": {"type": "array", "items": {"type": "string"}}},
    "required": ["diseases"],
    "additionalProperties": False,
}
SELECT_SCHEMA = {
    "type": "object",
    "properties": {"picks": {"type": "array", "items": {"type": "string"}}},
    "required": ["picks"],
    "additionalProperties": False,
}


class AnswererError(RuntimeError):
    pass


PDF_PREFIX_TOKENS = {"pdf", "пдф"}


def split_pdf_prefix(text):
    """'pdf ХСН' / 'пдф ХСН' -> 'ХСН'; '' when the prefix stands alone;
    None when the text is not a pdf query (owner decision 2026-10-04)."""
    parts = text.strip().split(None, 1)
    if parts and parts[0].lower() in PDF_PREFIX_TOKENS:
        return parts[1] if len(parts) > 1 else ""
    return None


class AppendixResult:
    def __init__(self, status, diseases=None, dropped=None, by_disease=None,
                 notes=None, appendix_text=None, filtered_out=None):
        self.status = status  # ok | extract_error | no_disease | search_error | select_error
        self.diseases = diseases or []
        self.dropped = dropped or []
        self.by_disease = by_disease or []
        self.notes = notes or []
        self.appendix_text = appendix_text
        self.filtered_out = filtered_out or []

    def to_payload(self):
        return {
            "status": self.status,
            "diseases": self.diseases,
            "dropped": self.dropped,
            "by_disease": self.by_disease,
            "notes": self.notes,
            "appendix_text": self.appendix_text,
            "filtered_out": self.filtered_out,
        }


def _extract_user(question, answer):
    return f"Вопрос врача:\n{question}\n\nОтвет ассистента:\n{answer}"


def _select_user(question, answer, disease, cards):
    lines = [f"Вопрос врача:\n{question}", f"\nОтвет ассистента:\n{answer}",
             f"\nЗаболевание: «{disease}»",
             "\nКандидаты (идентификатор — название клинической рекомендации):"]
    for i, c in enumerate(cards, 1):
        lines.append(f"{i}. {c['code_version']} — {c['name']}")
    lines.append("\nВыведи JSON: {\"picks\": [идентификаторы подходящих кандидатов]}.")
    return "\n".join(lines)


class Pipeline:
    def __init__(self, docs=None, index=None, abbr_map=None, audit=None,
                 candidate=None, labeler=None, embed=None,
                 extract_fn=None, search_fn=None, select_fn=None,
                 clinrec_dir=None):
        self.docs = docs
        self.index = index
        self.abbr_map = abbr_map
        self.audit = audit
        self.candidate = candidate or config.CANDIDATE
        self.labeler = labeler or config.LABELER
        self.embed = embed or config.EMBED
        self.clinrec_dir = clinrec_dir or config.CLINREC_DIR
        self.extract_fn = extract_fn
        self.search_fn = search_fn
        self.select_fn = select_fn

    # -- answerer ---------------------------------------------------------

    def _messages(self, question, history):
        return ([{"role": "system", "content": prompts.SYSTEM_ANSWERER}]
                + list(history or [])
                + [{"role": "user", "content": question}])

    def stream_answer(self, question, history=None, max_tokens=None):
        """Yield answerer events: {"type": "reasoning"|"content", "text": ...}.
        Raises llm.LLMError on transport failure."""
        config.require(self.candidate, "CANDIDATE")
        yield from llm.chat_stream(
            self.candidate, self._messages(question, history),
            max_tokens=max_tokens or config.ANSWERER_MAX_TOKENS)

    def answer(self, question, history=None, max_tokens=None):
        """Non-streaming answer (whole string). Diagnostic probe for budget/
        timeout experiments — the product paths stream via stream_answer."""
        config.require(self.candidate, "CANDIDATE")
        return llm.chat(
            self.candidate, self._messages(question, history),
            max_tokens=max_tokens or config.ANSWERER_MAX_TOKENS)

    # -- labeler calls ----------------------------------------------------

    def _labeler_chat(self, system, user, max_tokens, schema):
        """One labeler call, json_schema with a json_object compatibility fallback."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            return llm.chat(self.labeler, messages, max_tokens=max_tokens,
                            json_schema=schema)
        except llm.LLMError as e:
            if "HTTP 4" not in str(e):
                raise
            # server rejected the json_schema response format — one fallback attempt
            if self.audit:
                self.audit.log("labeler_schema_fallback", error=str(e))
            return llm.chat(self.labeler, messages, max_tokens=max_tokens,
                            json_mode=True)

    @staticmethod
    def _parse_diseases(raw):
        """Labeler extract output -> clean deduped disease list."""
        try:
            data = llm.parse_json_object(raw)
            diseases = data["diseases"]
            if not isinstance(diseases, list) or not all(
                    isinstance(d, str) for d in diseases):
                raise llm.LLMError("diseases is not a list of strings")
        except (llm.LLMError, KeyError, TypeError) as e:
            raise llm.LLMError(str(e))
        seen, out = set(), []
        for d in diseases:
            d = d.strip()
            if d and d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def _extract_call(self, system, user, manual=False):
        t0 = time.monotonic()
        raw = self._labeler_chat(system, user,
                                 config.LABELER_EXTRACT_MAX_TOKENS,
                                 EXTRACT_SCHEMA)
        if self.audit:
            kw = {"manual": True} if manual else {}
            self.audit.log("labeler_extract", raw=raw,
                           dur_s=round(time.monotonic() - t0, 2), **kw)
        try:
            return self._parse_diseases(raw)
        except llm.LLMError as e:
            if self.audit:
                self.audit.log("labeler_extract_invalid", raw=raw, error=str(e))
            raise

    def _real_extract(self, question, answer):
        return self._extract_call(prompts.SYSTEM_LABELER_EXTRACT,
                                  _extract_user(question, answer))

    def extract_manual(self, payload):
        """Labeler 1 in direct mode: parse a doctor-typed «pdf » payload
        into a clean disease list (owner decision 2026-10-04 — labeler 1
        normalizes the manual input and decides one vs several diseases)."""
        return self._extract_call(prompts.SYSTEM_LABELER_EXTRACT_DIRECT,
                                  payload, manual=True)

    def _real_search(self, variants):
        config.require(self.embed, "EMBED")
        if self.index is None:
            from .gindex import load_index
            self.index = load_index()
        return self.index.search(variants, self.embed, top_k=config.TOP_K)

    def _real_select(self, question, answer, disease, cards):
        t0 = time.monotonic()
        raw = self._labeler_chat(prompts.SYSTEM_LABELER_SELECT,
                                 _select_user(question, answer, disease, cards),
                                 config.LABELER_SELECT_MAX_TOKENS,
                                 SELECT_SCHEMA)
        if self.audit:
            self.audit.log("labeler_select", disease=disease, raw=raw,
                           dur_s=round(time.monotonic() - t0, 2))
        try:
            data = llm.parse_json_object(raw)
            picks = data["picks"]
            if not isinstance(picks, list) or not all(
                    isinstance(p, str) for p in picks):
                raise llm.LLMError("picks is not a list of strings")
        except (llm.LLMError, KeyError, TypeError) as e:
            if self.audit:
                self.audit.log("labeler_select_invalid", disease=disease,
                               raw=raw, error=str(e))
            raise llm.LLMError(str(e))
        seen, out = set(), []
        for p in picks:
            p = p.strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    # -- appendix machinery -----------------------------------------------

    def _auto_pick(self, cards):
        """The clearly dominating candidate, or None (labeler 2 then runs).

        Owner decision 2026-09-29: a single strong candidate makes the
        selection call redundant. Top-1 must reach AUTO_PICK_MIN and beat the
        runner-up by AUTO_PICK_GAP (no runner-up -> dominating by definition).
        """
        if not config.AUTO_PICK or not cards:
            return None
        top = cards[0]
        if top["score"] < config.AUTO_PICK_MIN:
            return None
        if len(cards) == 1 or top["score"] - cards[1]["score"] >= config.AUTO_PICK_GAP:
            return top
        return None

    def iter_appendix(self, question, raw_answer, session_id=None, diseases=None,
                      extract_fn=None, search_fn=None, select_fn=None,
                      manual=False):
        """Generator over the appendix machinery (architecture.md §3[2]-[5]).

        Yields progress events {"stage": extract|search|auto_pick|select,
        "disease": ..., "code_version": ...}; the LAST event is
        {"stage": "done", "result": AppendixResult}. Stage fns are injectable
        for offline tests; `diseases` stubs labeler 1. `manual=True` runs
        labeler 1 in direct mode over `question` (the «pdf » payload, owner
        decision 2026-10-04) and rewords the no-disease note.
        """
        if (diseases is None and extract_fn is None and self.extract_fn is None
                and self.labeler and self.labeler.get("base_url")):
            config.require(self.labeler, "LABELER")
        notes, by_disease, cand_by_disease = [], [], []

        # [2] labeler 1 — extraction
        if diseases is None:
            yield {"stage": "extract"}
            if extract_fn is not None:
                extract = extract_fn
            elif self.extract_fn is not None:
                extract = self.extract_fn
            elif manual:
                def extract(q, a):
                    return self.extract_manual(q)
            else:
                extract = self._real_extract
            try:
                diseases = extract(question, raw_answer)
            except llm.LLMError as e:
                if self.audit:
                    self.audit.log("stage_stop", stop_stage="labeler_extract",
                                   session_id=session_id, error=str(e))
                yield {"stage": "done", "result": AppendixResult(
                    "extract_error", notes=[appendix.NOTE_EXTRACT_ERROR])}
                return
        seen, cleaned = set(), []
        for d in diseases or []:
            d = d.strip()
            if d and d not in seen:
                seen.add(d)
                cleaned.append(d)
        diseases = cleaned
        if self.audit:
            kw = {"manual": True} if manual else {}
            self.audit.log("diseases", session_id=session_id,
                           diseases=diseases, **kw)

        if not diseases:
            yield {"stage": "done", "result": AppendixResult(
                "no_disease",
                notes=[appendix.NOTE_NO_DISEASE_MANUAL if manual
                       else appendix.NOTE_NO_DISEASE])}
            return

        processed, dropped = diseases[:config.DISEASE_CAP], diseases[config.DISEASE_CAP:]
        if dropped and self.audit:
            self.audit.log("disease_cap", session_id=session_id,
                           dropped=dropped, cap=config.DISEASE_CAP)

        search = search_fn or self.search_fn or (lambda variants: self._real_search(variants))
        select = select_fn or self.select_fn or (lambda q, a, d, c: self._real_select(q, a, d, c))

        for disease in processed:
            # [3] tool — guideline search
            variants = (self.abbr_map.variants(disease)
                        if self.abbr_map is not None else [disease])
            yield {"stage": "search", "disease": disease}
            t_search = time.monotonic()
            try:
                cards = search(variants)
            except llm.LLMError as e:
                if self.audit:
                    self.audit.log("stage_stop", stop_stage="search", disease=disease,
                                   session_id=session_id, error=str(e))
                yield {"stage": "done", "result": AppendixResult(
                    "search_error", diseases=processed, dropped=dropped,
                    by_disease=by_disease, notes=notes + [appendix.NOTE_SEARCH_ERROR])}
                return
            if self.audit:
                self.audit.log("search", disease=disease, session_id=session_id,
                               variants=variants,
                               cards=[c["code_version"] for c in cards],
                               dur_s=round(time.monotonic() - t_search, 2))
            if not cards:
                continue

            # resolved candidates feed the flat filtered-out list: the UI
            # shows the verified picks and hides everything the selection
            # dropped behind one collapsible
            cand = (appendix.resolve_picks(
                        [c["code_version"] for c in cards], self.docs,
                        self.clinrec_dir)
                    if self.docs else
                    [{"code_version": c["code_version"],
                      "name": c["name"]} for c in cards])

            # [4] labeler 2 — selection, or auto-pick on a dominant candidate
            auto_card = self._auto_pick(cards)
            if auto_card is not None:
                yield {"stage": "auto_pick", "disease": disease,
                       "code_version": auto_card["code_version"]}
                picks, auto = [auto_card["code_version"]], True
            else:
                # search is cheap, the labeler call is not: hand the candidate
                # cards to the UI immediately; it filters them client-side
                # when the final appendix payload arrives. Retrieval score
                # rides along for the UI badge (2dp) — on copies only, the
                # stored filtered_out payload stays score-free; merge by id,
                # not zip — resolve_picks may drop entries
                score_by_cv = {c["code_version"]: c.get("score") for c in cards}
                scored = []
                for c in cand:
                    s = score_by_cv.get(c["code_version"])
                    scored.append({**c, "score": round(s, 2)}
                                  if s is not None else c)
                yield {"stage": "select", "disease": disease, "cards": scored}
                try:
                    picks = select(question, raw_answer, disease, cards)
                except llm.LLMError as e:
                    if self.audit:
                        self.audit.log("stage_stop", stop_stage="labeler_select",
                                       disease=disease, session_id=session_id,
                                       error=str(e))
                    yield {"stage": "done", "result": AppendixResult(
                        "select_error", diseases=processed, dropped=dropped,
                        by_disease=by_disease,
                        notes=notes + [appendix.NOTE_SELECT_ERROR])}
                    return
                auto = False
            if self.audit:
                self.audit.log("picks", disease=disease, session_id=session_id,
                               picks=picks, auto=auto)

            resolved = (appendix.resolve_picks(picks, self.docs, self.clinrec_dir)
                        if self.docs else [])
            if resolved:
                by_disease.append({"disease": disease, "picks": resolved,
                                   "auto_pick": auto})
            cand_by_disease.append((disease, cand))

        appendix_text = appendix.render_appendix(by_disease)
        result = AppendixResult("ok", diseases=processed, dropped=dropped,
                                by_disease=by_disease, notes=notes,
                                appendix_text=appendix_text,
                                filtered_out=appendix.filtered_out(
                                    by_disease, cand_by_disease))
        if self.audit:
            self.audit.log("appendix_done", session_id=session_id,
                           status=result.status,
                           appendix_text=appendix_text, notes=notes)
        yield {"stage": "done", "result": result}

    def run_appendix(self, question, raw_answer, session_id=None, diseases=None,
                     extract_fn=None, search_fn=None, select_fn=None,
                     manual=False):
        """Blocking wrapper over iter_appendix -> AppendixResult."""
        result = None
        for ev in self.iter_appendix(question, raw_answer, session_id=session_id,
                                     diseases=diseases, extract_fn=extract_fn,
                                     search_fn=search_fn, select_fn=select_fn,
                                     manual=manual):
            if ev.get("stage") == "done":
                result = ev["result"]
        return result


def load_pipeline(audit=None, with_index=True):
    """Fully wired Pipeline from config data assets (docs_meta, abbr, index)."""
    from .abbr import Abbr, load_abbr
    from .gindex import GuidelineIndex

    docs = config.load_docs_meta()
    abbr_map = (Abbr(load_abbr(config.ABBR_PATH))
                if config.ABBR_PATH.exists() else None)
    index = None
    if with_index:
        vec = config.INDEX_DIR / "vectors.npy"
        if vec.exists():
            index = GuidelineIndex.load(config.INDEX_DIR)
    return Pipeline(docs=docs, index=index, abbr_map=abbr_map, audit=audit)

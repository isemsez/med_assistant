"""Orchestrator: answerer -> labeler 1 (extract) -> tool -> labeler 2 (select)
-> renderer, with stage-stops and honest notes (architecture.md §3, §5).

No semantic retries: an invalid/empty labeler output or a dead embed endpoint
stops the pipeline with an error note; the raw answer is never modified.
Stage functions are injectable for offline tests and CLI mock runs.
"""

import json
import re
import time

from . import appendix, config, llm, loopguard, prompts

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
# Grammar cap on the per-candidate "reasoning" string in the per-call select
# schema (_real_select) — single source of truth for the schema dict and the
# audit reasoning_truncated check below. 1400 (2026-10-20): the first cap, 700,
# could not mathematically fit a 5-candidate enumeration — one verdict item
# measured ~140-173 chars, five ≈ 730+ — so the clamp slammed shut mid-list and
# shipped picks contradicting the lost verdicts (live 2026-10-20); the terse
# verdict skeleton keeps a real enumeration ~500 chars, far under the cap.
SELECT_REASONING_MAXLEN = 1400


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


def _title_key(text):
    """Lowercase, ё→е, punctuation collapsed — for title containment checks."""
    text = (text or "").lower().replace("ё", "е")
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def _dedup_key(name, age=None):
    """Identity for true-duplicate collapse: (title with parentheticals
    stripped, register audience). The register holds ZERO same-title-
    same-audience docs (surveyed 2026-10-21, 757 docs) — same-NAME pairs
    are always adult/pediatric versions of the recommendation (24 pairs:
    «Головная боль напряжения» 810_1 дети vs 162_3 взрослые, both
    «Мигрень», …) and must NEVER merge: the first title-collapse
    implementation merged them, kept the higher-scoring CHILDREN'S doc
    and auto-picked it for an adult question (owner catch 2026-10-21).
    The audience key keeps the collapse armed only for future register
    refreshes that introduce real same-audience duplicates."""
    return _title_key(re.sub(r"\([^)]*\)", " ", name or "")), (age or "").strip()


_AGE_MARKERS = {"Взрослые": "взрослые", "Дети": "дети",
                "Взрослые, дети": "взрослые и дети"}


def _extract_user(question, answer):
    return f"Вопрос врача:\n{question}\n\nОтвет ассистента:\n{answer}"


def _select_user(question, answer, disease, cards):
    lines = [f"Вопрос врача:\n{question}"]
    if answer:
        lines.append(f"\nОтвет ассистента:\n{answer}")
    lines.append(f"\nЗаболевание: «{disease}»\n"
                 "\nКандидаты (идентификатор — название клинической рекомендации):")
    # `matched` (the indexed МКБ variant that pulled the candidate in,
    # when different from its name) lets the labeler see through
    # sound-alike noise — a «поли-» prefix match on an unrelated doc —
    # instead of rubber-stamping every candidate (2026-10-16). A term
    # SHARED by several candidates is printed on none of them: identical
    # parentheticals say nothing that picks one candidate over another
    # and hand the 4B a repetitive hook it echoes verbatim (2026-10-21:
    # «Синдром «гистаминовой» головной боли» on 3 of 5 lines became a 5x
    # echo loop in the select reasoning — 23x without the grammar, a ~50x
    # think-loop to the token cap under the answerer template; the loop
    # is prompt-fueled, every transport repeated it).
    shown = [c.get("matched") for c in cards
             if c.get("matched") and c["matched"] != c["name"]]
    unique = {m for m in shown if shown.count(m) == 1}
    for i, c in enumerate(cards, 1):
        m = c.get("matched")
        extra = (f" (совпавший термин: «{m}»)"
                 if m and m != c["name"] and m in unique else "")
        # the audience marker (docs_meta `age`, enriched onto the card by
        # iter_appendix) is load-bearing (owner catch 2026-10-21): the
        # register holds same-NAME adult/pediatric pairs whose titles
        # carry no age cue, so without the marker the selector's rule 4
        # («педиатрические — только если речь о ребёнке») cannot fire and
        # identical-looking lines feed the echo loop — the blind echo
        # shipped the CHILDREN'S headache guideline to adult questions
        marker = _AGE_MARKERS.get((c.get("age") or "").strip())
        age_part = f" ({marker})" if marker else ""
        lines.append(f"{i}. {c['code_version']} — {c['name']}{age_part}{extra}")
    # the skeleton tail is load-bearing (probed live 2026-10-20): with it the
    # model enumerates every candidate with verdicts and picks correctly; an
    # instruction-only tail reasoned right but still emitted picks=[]; a
    # prose example value echoed verbatim into reasoning. Per-item verdicts
    # are terse by design («подходит или не подходит», no «почему»):
    # explained verdicts measured ~140-173 chars per item and clamped the
    # capped reasoning string mid-enumeration (2026-10-20)
    lines.append("\nВыведи JSON: {\"reasoning\": \"1. <идентификатор> — "
                 "<подходит или не подходит>. 2. …"
                 "\", "
                 "\"picks\": [идентификаторы подходящих кандидатов]}.")
    return "\n".join(lines)


def _select_reasoning_loop(reasoning):
    """Degenerate select reasoning -> short reason, else None (2026-10-21).

    Both live degenerations were thinking loops in the answerer-watchdog
    sense (owner decisions 2026-10-06/14: a unit repeated back-to-back):
    (a) a verbatim tail-spin — caught by reusing loopguard.LoopGuard on
    the complete string; (b) a numbered echo or CYCLE — the same item
    bodies under «1.», «2.»… prefixes (numbers break LoopGuard's verbatim
    run; a cap-clamped tail also chops the final copy mid-word), so item
    bodies are compared with the enumeration number stripped and the TAIL
    must not end with >= 3 back-to-back copies of the same p-item block
    (p=1 is the plain echo; live 2026-10-21 a deduped 3-candidate list
    was enumerated cyclically 4x over — [A,B,C]x4 — which consecutive
    identity alone cannot see). Blocks, not scattered recurrence: a
    healthy list of distinct candidates never repeats a whole block
    verbatim (each body carries its candidate id). A looped reasoning
    judged nothing — its picks are dropped by the caller, the title guard
    is the net."""
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None
    spin = loopguard.LoopGuard().feed(reasoning)
    if spin:
        return spin
    bodies = []
    for part in re.split(r"(?<=[.!?])\s+", reasoning):
        body = re.sub(r"^\s*\d+[.)]\s*", "", part.strip())
        if body:
            bodies.append(body)
    n = len(bodies)
    # largest period first (the maximal cycle reads cleanest in audit);
    # up to p-1 trailing bodies may be a partial block — a cap-clamped
    # tail chops the final copy mid-item and must not hide the cycle
    for p in range(min(8, n // 3), 0, -1):
        for drop in range(p):
            m = n - drop
            if m < 3 * p:
                break
            block = bodies[m - p:m]
            reps = 1
            while (reps * p + p <= m
                   and bodies[m - (reps + 1) * p:m - reps * p] == block):
                reps += 1
            if reps >= 3:
                sample = " | ".join(b[:40] for b in block)[:80]
                return f"cycle x{reps} (p={p}): …{sample}"
    return None


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

    def _messages(self, question, history, card=None, drug=None):
        """Answerer prompt; `card=(display_name, text)` appends the
        patient-card block to the user message (architecture.md §9), and
        `drug=[{trade, inn}, ...]` the register-grounded МНН note
        (architecture.md §3[1]). The system prompt, history and the stored
        question are unaffected.

        The МНН note goes BEFORE the question (owner decision 2026-10-17,
        superseding 2026-10-12's after-question position): the 4B's thinking
        paraphrases the question into English on its very first lines and
        phonetically re-identifies «валидол» as valium there — diazepam
        pharmacology then leaks into the answer under a correct МНН label.
        With the anchor already in front of the question the binding wins:
        3/3 clean live runs vs 0/2 with the note after the question.
        """
        drug_text = prompts.drug_block(drug)
        content = f"{drug_text}\n\n{question}" if drug_text else question
        if card is not None:
            display_name, text = card
            content = f"{content}\n\n{prompts.card_block(display_name, text)}"
        return ([{"role": "system", "content": prompts.SYSTEM_ANSWERER}]
                + list(history or [])
                + [{"role": "user", "content": content}])

    def stream_answer(self, question, history=None, max_tokens=None, card=None,
                      drug=None, cancel=None):
        """Yield answerer events: {"type": "reasoning"|"content", "text": ...},
        plus {"type": "reasoning_reset"} when a degenerate thinking loop was
        aborted and the stream restarted (owner decision 2026-10-06: one
        anti-repetition retry, then the error surfaces — see llm.chat_stream).
        `cancel` (llm.StreamCancel, owner decision 2026-10-15) aborts the
        stream with llm.LLMCancelled (user stop — dropped connection, never
        retried).
        Raises llm.LLMError on transport failure."""
        config.require(self.candidate, "CANDIDATE")
        for attempt in range(1 + config.LOOP_RETRIES):
            try:
                yield from llm.chat_stream(
                    self.candidate,
                    self._messages(question, history, card, drug),
                    max_tokens=max_tokens or config.ANSWERER_MAX_TOKENS,
                    freq_penalty=(config.LOOP_RETRY_FREQ_PENALTY
                                  if attempt else None),
                    cancel=cancel)
                return
            except llm.LLMLoopError as e:
                if attempt >= config.LOOP_RETRIES:
                    raise
                if self.audit:
                    self.audit.log("answerer_loop_abort",
                                   attempt=attempt + 1, error=str(e))
                yield {"type": "reasoning_reset"}

    def answer(self, question, history=None, max_tokens=None, card=None,
               drug=None):
        """Non-streaming answer (whole string). Diagnostic probe for budget/
        timeout experiments — the product paths stream via stream_answer."""
        config.require(self.candidate, "CANDIDATE")
        return llm.chat(
            self.candidate, self._messages(question, history, card, drug),
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
        # diseases from BOTH sources (owner decision 2026-10-20): the answer
        # pass is primary (its diseases must not lose DISEASE_CAP slots), the
        # question pass reuses the pdf-direct prompt over the bare question
        # — «включи каждое названное заболевание» sticks where questioning
        # the same pass's rules failed (probed live). A «Да.» answer to
        # «А при инсульте?» anchors the stroke guideline; a drifted answer
        # can no longer silently drop an asked-about disease. Merged
        # answer-first, exact strings deduped; near-forms (e.g.
        # гипертония/гипертензия) may stay as rare duplicate groups — select
        # resolves them to the same card.
        answer_diseases = self._extract_call(prompts.SYSTEM_LABELER_EXTRACT,
                                             _extract_user(question, answer))
        question_diseases = self._extract_call(
            prompts.SYSTEM_LABELER_EXTRACT_DIRECT, question)
        merged = list(answer_diseases)
        low = {d.lower() for d in merged}
        for d in question_diseases:
            if d.lower() not in low:
                merged.append(d)
                low.add(d.lower())
        return merged

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
        """One select call PER RECOMMENDATION TITLE GROUP (owner decision
        2026-10-21): the 4B cannot enumerate a 5-candidate mixed list
        without degenerating (live: echo loops 3/3, cyclic enumerations),
        but judges 1-2 items reliably. Same-NAME adult/pediatric versions
        share ONE call — the audience choice is comparative, splitting
        the pair would hide the twin (owner constraint); different
        recommendations are judged in separate short calls. Picks are
        unioned; a looped or invalid group fails only itself, and the
        title guard remains the net over the union."""
        # no answer -> a manual pdf turn: the direct prompt matches what the
        # user message actually contains (no «ответ ассистента» section)
        system = (prompts.SYSTEM_LABELER_SELECT_DIRECT if not answer
                  else prompts.SYSTEM_LABELER_SELECT)
        groups = {}
        for c in cards:
            groups.setdefault(_dedup_key(c["name"])[0], []).append(c)
        picked = []
        for group in groups.values():
            picked += self._select_group(system, question, answer, disease,
                                         group)
        return picked

    def _select_group(self, system, question, answer, disease, group):
        """One labeler call over one title group (see _real_select)."""
        t0 = time.monotonic()
        # per-call schema (owner decision 2026-10-18, llama.cpp port): the
        # pickable ids are exactly the group's code_versions and the array
        # is bounded by the group size — under grammar mode an unbounded
        # free-string array let a greedy-degenerate labeler loop ids until
        # truncation, and the unclosed object failed to parse.
        # "reasoning" comes FIRST (owner decision 2026-10-20): grammar mode
        # forces the first generated token to be "{", so the model cannot
        # re-open think like the old Ollama transport — the GBNF property
        # order makes it reason per candidate inside the JSON before picks,
        # the root-cause fix for empty picks (schema_to_grammar emits
        # required properties in dict order). SELECT_REASONING_MAXLEN bounds
        # the reasoning string in the grammar itself: an unbounded string ran
        # degenerate to the token cap and left the object unclosed
        # (2026-10-20), and the first cap, 700, was tighter than any
        # 5-candidate enumeration (~730+ chars measured) — the clamp slammed
        # shut mid-list and shipped picks contradicting the lost verdicts
        # (live 2026-10-20); at the cap the grammar slams the string shut and
        # the JSON still parses
        schema = {"type": "object",
                  "properties": {
                      "reasoning": {"type": "string",
                                    "maxLength": SELECT_REASONING_MAXLEN},
                      "picks": {
                          "type": "array",
                          "items": {"type": "string",
                                    "enum": [c["code_version"]
                                             for c in group]},
                          "maxItems": len(group)}},
                  "required": ["reasoning", "picks"],
                  "additionalProperties": False}
        ids = [c["code_version"] for c in group]
        raw = self._labeler_chat(system,
                                 _select_user(question, answer, disease,
                                              group),
                                 config.LABELER_SELECT_MAX_TOKENS,
                                 schema)
        try:
            data = llm.parse_json_object(raw)
        except (llm.LLMError, KeyError, TypeError) as e:
            if self.audit:
                self.audit.log("labeler_select", disease=disease, group=ids,
                               raw=raw,
                               dur_s=round(time.monotonic() - t0, 2))
                self.audit.log("labeler_select_invalid", disease=disease,
                               group=ids, raw=raw, error=str(e))
            raise llm.LLMError(str(e))
        # a reasoning string exactly at the grammar cap means the clamp
        # landed (observability, 2026-10-20): mid-enumeration that ships
        # unchecked/contradicting picks — the flag is purely observational,
        # both clamped shapes still parse the same
        r = data.get("reasoning") if isinstance(data, dict) else None
        if self.audit:
            kw = ({"reasoning_truncated": True}
                  if isinstance(r, str)
                  and len(r) == SELECT_REASONING_MAXLEN else {})
            self.audit.log("labeler_select", disease=disease, group=ids,
                           raw=raw, dur_s=round(time.monotonic() - t0, 2),
                           **kw)
        try:
            picks = data["picks"]
            if not isinstance(picks, list) or not all(
                    isinstance(p, str) for p in picks):
                raise llm.LLMError("picks is not a list of strings")
        except (llm.LLMError, KeyError, TypeError) as e:
            if self.audit:
                self.audit.log("labeler_select_invalid", disease=disease,
                               group=ids, raw=raw, error=str(e))
            raise llm.LLMError(str(e))
        # the lenient parse (picks-only fakes, CLI answer files) may emit
        # ids from other groups — a pick is kept only when it belongs to
        # THIS call's group, mirroring what the grammar would have forced
        seen, out = set(), []
        for p in picks:
            p = p.strip()
            if p and p in ids and p not in seen:
                seen.add(p)
                out.append(p)
        # thinking-loop watchdog, select edition (2026-10-21): a looped
        # reasoning judged nothing — its picks are a blind echo of the
        # first candidate line (live 09:34 and 14:07: [810_1] off echoes).
        # The picks are dropped, NOT retried (invalid labeler output is
        # never retried); only this group's picks die, the other groups
        # and the title guard still contribute.
        loop_reason = _select_reasoning_loop(r)
        if loop_reason:
            if self.audit:
                self.audit.log("labeler_select_loop", disease=disease,
                               group=ids, reason=loop_reason, dropped=out)
            return []
        return out

    # -- appendix machinery -----------------------------------------------

    # audience markers (owner doctrine 2026-10-21): when the turn names
    # NO audience, a same-NAME adult/pediatric pair must attach BOTH
    # versions — the doctor chooses himself; any marker (child OR adult
    # wording) means the turn states its audience and the guard stands
    # down, leaving the choice to the selector's rule 4
    _AUDIENCE_MARKERS = ("ребен", "ребён", "детск", "младен", "новорожд",
                         "подростк", "педиатр", "взросл")

    def _audience_complete(self, question, answer, cards, picks):
        """Complete a half-picked adult/pediatric pair (owner doctrine
        2026-10-21: unclear audience -> BOTH versions, the doctor
        chooses). The 4B stamps one side unpredictably (live: the ГБН
        pair returned only 810_1 дети on a bare adult-less question,
        dropping 162_3 взрослые); when one side of a pair is picked,
        the missing side's best-scored twin is added. Fires only with
        ZERO audience markers in question+answer; «Взрослые, дети» docs
        cover both audiences and never need completion."""
        text = f"{question} {answer or ''}".lower()
        if any(m in text for m in self._AUDIENCE_MARKERS):
            return picks, None
        by_title = {}
        for c in cards:
            by_title.setdefault(_dedup_key(c["name"])[0], []).append(c)
        picked = set(picks)
        added = []
        for group in by_title.values():
            sides = {"Дети": [c for c in group if c.get("age") == "Дети"],
                     "Взрослые": [c for c in group if c.get("age") == "Взрослые"]}
            for mine, other in (("Дети", "Взрослые"), ("Взрослые", "Дети")):
                if (sides[mine] and sides[other]
                        and any(c["code_version"] in picked
                                for c in sides[mine])
                        and not any(c["code_version"] in picked
                                    for c in sides[other])):
                    twin = sides[other][0]  # best-scored (retrieval order)
                    picked.add(twin["code_version"])
                    picks = picks + [twin["code_version"]]
                    added.append(twin["code_version"])
        return picks, (added or None)

    def _title_guard(self, disease, cards, picks):
        """Keep the one guideline whose title contains the disease verbatim.

        The think-suppressed labeler sometimes returns []/a subset even when a
        candidate's title is the disease itself (llama.cpp grammar mode cannot
        match the old Ollama think-then-JSON behavior; 2026-10-19). A unique
        verbatim title match is definitionally relevant, so it is appended
        when the selector dropped it. Ambiguous titles (two or more matches,
        e.g. adult vs pediatric forms) are left to the labeler.
        """
        key = _title_key(disease)
        if not key or not cards:
            return picks, None
        hits = [c for c in cards if key in _title_key(c.get("name", ""))]
        if len(hits) != 1:
            return picks, None
        cv = hits[0]["code_version"]
        if cv in picks:
            return picks, None
        return [*picks, cv], cv

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
        decision 2026-10-04) and rewords the no-disease note. A manual turn
        is input-only: `raw_answer` is forced empty so no earlier
        conversation leaks into the selection call.
        """
        if manual:
            raw_answer = ""
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

            # audience enrichment (owner catch 2026-10-21): the retrieval
            # card carries no age, but docs_meta does («Взрослые» / «Дети»
            # / «Взрослые, дети») and the register holds 24 same-NAME
            # adult/pediatric pairs whose titles give no cue. The age
            # rides on the card from here: the select prompt renders it
            # (rule 4 becomes applicable), dedup keys on it, the appendix
            # marks pediatric picks. Enriched copies — the search result
            # dicts are not ours to mutate (injected fakes share them).
            if self.docs:
                cards = [
                    dict(c, age=self.docs[c["code_version"]]["age"])
                    if self.docs.get(c["code_version"], {}).get("age")
                    else c
                    for c in cards]

            # true-duplicate collapse (2026-10-21, reworked same day after
            # the owner catch): keep ONE candidate per (title, audience) —
            # same-NAME pairs are adult/pediatric versions and NEVER merge
            # (the first title-only implementation kept the higher-scoring
            # CHILDREN'S doc and auto-picked it for an adult question).
            # The register currently holds zero same-(title, audience)
            # docs; the collapse stays armed for future refreshes.
            dropped_dups, seen_keys = [], set()
            distinct = []
            for c in cards:
                k = _dedup_key(c["name"], c.get("age"))
                if k in seen_keys:
                    dropped_dups.append(c["code_version"])
                    continue
                seen_keys.add(k)
                distinct.append(c)
            if dropped_dups and self.audit:
                self.audit.log("search_dedup", disease=disease,
                               session_id=session_id, dropped=dropped_dups)
            cards = distinct

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
                # when the final appendix payload arrives. Retrieval scores
                # are display-free: only auto-pick consumes them
                yield {"stage": "select", "disease": disease, "cards": cand}
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
                picks, guarded = self._title_guard(disease, cards, picks)
                if guarded and self.audit:
                    self.audit.log("select_title_guard", disease=disease,
                                   session_id=session_id, code_version=guarded,
                                   picks=picks)
                picks, completed = self._audience_complete(
                    question, raw_answer, cards, picks)
                if completed and self.audit:
                    self.audit.log("select_audience_complete",
                                   disease=disease, session_id=session_id,
                                   added=completed, picks=picks)
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

"""Answerer thinking-loop watchdog (owner decision 2026-10-06; redefined
2026-10-14 after a false abort).

medgemma's reasoning occasionally collapses into a tail-spin: the SAME
unit (word, phrase, sentence) is emitted back-to-back until the token
budget is exhausted — minutes of pegged CPU and a blocked composer. The
guard watches the pre-content reasoning stream and trips only on such
CONSECUTIVE repetition.

The first version counted occurrences of the recent tail anywhere inside
the window; that also fired on legitimately generated enumerations whose
items merely shared a long suffix — the card's own example answers
("…назначено лизиноприл 10 мг утром и изосорбида мононитрат 20 мг 2 раза
в сутки.") recurred across DIFFERENT questions the model was listing,
aborting a healthy stream (owner report 2026-10-14). Scattered repeats
are not loops; only a unit repeated verbatim back-to-back is.

Abort is done by the caller (llm.chat_stream): raising out of the
httpx.stream context closes the connection, which cancels Ollama
generation.
"""


class LoopGuard:
    """Trip when the buffer ends with a unit of >= `min_unit` chars
    repeated back-to-back >= `repeats` times (for ANY unit length that
    fits `repeats` whole copies inside a rolling `window_chars` buffer).
    Occurrences separated by other text never trip."""

    def __init__(self, min_unit=8, repeats=3, window_chars=4096):
        self.min_unit = min_unit
        self.repeats = repeats
        self.window_chars = window_chars
        self._buf = ""

    def feed(self, text):
        """Append reasoning text; return a short reason when a
        consecutive repetition run is detected, else None."""
        if not text:
            return None
        self._buf = (self._buf + text)[-self.window_chars:]
        run = self._tail_run()
        if run is None:
            return None
        unit, reps = run
        return f"unit x{reps}: …{unit[:80]}"

    def _tail_run(self):
        """(unit, reps) for the shortest unit length L >= min_unit such
        that the buffer ends with >= repeats back-to-back copies of that
        unit, else None. A period-L run forces buf[-1] == buf[-1-L], an
        O(1) quick reject before any slice is built."""
        buf, n = self._buf, len(self._buf)
        for length in range(self.min_unit, n // self.repeats + 1):
            if buf[n - 1] != buf[n - 1 - length]:
                continue
            unit = buf[n - length:]
            reps, pos = 1, n
            while (pos - 2 * length >= 0
                   and buf[pos - 2 * length:pos - length] == unit):
                reps += 1
                pos -= length
            if reps >= self.repeats:
                return unit, reps
        return None

    @property
    def tail(self):
        """Last chars seen (audit samples)."""
        return self._buf[-80:]

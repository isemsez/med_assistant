"""Answerer thinking-loop watchdog (owner decision 2026-10-06).

medgemma's reasoning occasionally collapses into a verbatim repetition
loop that runs until the token budget is exhausted — minutes of pegged CPU
and a blocked composer. The guard watches the pre-content reasoning stream
and trips when the recent tail repeats verbatim inside a rolling window.
Abort is done by the caller (llm.chat_stream): raising out of the
httpx.stream context closes the connection, which cancels Ollama
generation.
"""


class LoopGuard:
    """Trip when the last `tail_chars` chars occur >= `repeats` times
    inside a rolling `window_chars` buffer (overlapping occurrences —
    a loop period shorter than the tail shifts matches by less than the
    tail length, which str.count would skip)."""

    def __init__(self, tail_chars=64, repeats=3, window_chars=4096):
        self.tail_chars = tail_chars
        self.repeats = repeats
        self.window_chars = window_chars
        self._buf = ""

    def feed(self, text):
        """Append reasoning text; return a short reason when a loop is
        detected, else None."""
        if not text:
            return None
        self._buf = (self._buf + text)[-self.window_chars:]
        if len(self._buf) < self.tail_chars:
            return None
        tail = self._buf[-self.tail_chars:]
        count = 0
        pos = self._buf.find(tail)
        while pos != -1:
            count += 1
            pos = self._buf.find(tail, pos + 1)
        if count >= self.repeats:
            return f"tail x{count}: …{tail[:80]}"
        return None

    @property
    def tail(self):
        """Last chars seen (audit samples)."""
        return self._buf[-80:]

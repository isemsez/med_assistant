import unittest

from medassistant import config
from medassistant.loopguard import LoopGuard

# the observed medgemma failure (2026-10-06): verbatim phrase cycling,
# the same unit emitted back-to-back until the budget dies
LOOP_PHRASE = "негиперпигментационные, негиперкератотические, "

# the observed false abort (2026-10-14): DIFFERENT questions sharing one
# long answer suffix from the card's example section — recurring but
# never consecutive, so the old anywhere-in-window detector killed a
# healthy stream
SHARED_SUFFIX = ('м и изосорбида мононитрат 20 мг 2 раза в сутки."'
                 "\n*   **Doctor:**")
SCATTERED = "".join(
    f"Вопрос {i}: лизиноприл 10 мг утро{SHARED_SUFFIX} гипертензия {i}? "
    for i in range(8))

# flowing, non-repetitive medical reasoning
FLOWING = "".join(
    f"Шаг {i}: дифференциальный ряд зависит от локализации процесса. "
    for i in range(40))


class TestLoopGuard(unittest.TestCase):
    def setUp(self):
        self.guard = LoopGuard(min_unit=8, repeats=3, window_chars=4096)

    def test_fires_on_consecutive_run(self):
        self.assertIsNone(self.guard.feed("Разбираю заболевание. "))
        reason = None
        for _ in range(10):
            reason = self.guard.feed(LOOP_PHRASE)
            if reason:
                break
        self.assertIsNotNone(reason)
        self.assertIn("unit x", reason)

    def test_fires_on_short_unit_run(self):
        reason = self.guard.feed("пациента беспокоит " * 4)
        self.assertIsNotNone(reason)

    def test_silent_on_scattered_repeats(self):
        # regression (2026-10-14): shared suffixes across distinct items
        # must NOT trip — only back-to-back repetition is a loop
        self.assertIsNone(self.guard.feed(SCATTERED))

    def test_silent_on_flowing_text(self):
        self.assertIsNone(self.guard.feed(FLOWING))

    def test_silent_below_min_unit(self):
        # a run of a unit shorter than min_unit is not trip-worthy
        self.assertIsNone(self.guard.feed("короткая мысль"))
        self.assertIsNone(self.guard.feed("аб " * 6))

    def test_two_consecutive_copies_do_not_trip(self):
        # restating a sentence once verbatim is not a loop (repeats=3)
        sentence = "назначено: аторвастатин 40 мг вечером."
        self.assertIsNone(self.guard.feed(f"{sentence} {sentence} "))

    def test_empty_feed_is_noop(self):
        self.assertIsNone(self.guard.feed(""))

    def test_window_trim_still_fires(self):
        guard = LoopGuard(min_unit=8, repeats=3, window_chars=200)
        self.assertIsNone(guard.feed(FLOWING))  # trims old text on the way
        reason = None
        for _ in range(10):
            reason = guard.feed(LOOP_PHRASE)
            if reason:
                break
        self.assertIsNotNone(reason)

    def test_tail_sample(self):
        self.guard.feed("abc")
        self.assertEqual(self.guard.tail, "abc")

    def test_tail_keeps_last_n_chars_fed(self):
        # 2026-10-24: the audit `tail` field must survive the whole looped
        # thinking up to tail_chars — the last chars fed, not a snippet
        guard = LoopGuard(min_unit=8, repeats=3, window_chars=4096,
                          tail_chars=50)
        guard.feed("A" * 30)
        guard.feed("B" * 30)
        self.assertEqual(guard.tail, "A" * 20 + "B" * 30)

    def test_tail_resets_with_the_guard(self):
        # llm.chat_stream builds a fresh guard per attempt: a retry's tail
        # must not carry attempt 1's looped text
        self.guard.feed(LOOP_PHRASE * 8)
        self.assertEqual(self.guard.tail, (LOOP_PHRASE * 8)[-2000:])
        fresh = LoopGuard(min_unit=8, repeats=3, window_chars=4096)
        self.assertEqual(fresh.tail, "")


class TestConfigDefaults(unittest.TestCase):
    def test_defaults(self):
        self.assertTrue(config.LOOP_GUARD)
        self.assertEqual(config.LOOP_MIN_UNIT, 8)
        self.assertEqual(config.LOOP_REPEATS, 3)
        self.assertEqual(config.LOOP_WINDOW_CHARS, 4096)
        self.assertEqual(config.LOOP_RETRIES, 1)
        self.assertEqual(config.LOOP_RETRY_FREQ_PENALTY, 0.5)
        # loop-retry DRY sampler (2026-10-24): llama.cpp per-request dry_*,
        # breaker default U+241E (never emitted — the \n : " * defaults
        # reset matching inside loop units containing those chars)
        self.assertEqual(config.LOOP_RETRY_DRY_MULTIPLIER, 0.8)
        self.assertEqual(config.LOOP_RETRY_DRY_BASE, 1.75)
        self.assertEqual(config.LOOP_RETRY_DRY_ALLOWED_LENGTH, 2)
        self.assertEqual(config.LOOP_RETRY_DRY_PENALTY_LAST_N, 64)
        self.assertEqual(config.LOOP_RETRY_DRY_BREAKERS, "\u241e")
        self.assertEqual(config.LOOP_TAIL_CHARS, 2000)


if __name__ == "__main__":
    unittest.main()

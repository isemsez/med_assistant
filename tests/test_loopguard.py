import unittest

from medassistant import config
from medassistant.loopguard import LoopGuard

# the observed medgemma failure (2026-10-06): verbatim phrase cycling
LOOP_PHRASE = "негиперпигментационные, негиперкератотические, "

# flowing, non-repetitive medical reasoning
FLOWING = "".join(
    f"Шаг {i}: дифференциальный ряд зависит от локализации процесса. "
    for i in range(40))


class TestLoopGuard(unittest.TestCase):
    def setUp(self):
        self.guard = LoopGuard(tail_chars=64, repeats=3, window_chars=4096)

    def test_fires_on_verbatim_loop(self):
        self.assertIsNone(self.guard.feed("Разбираю заболевание. "))
        reason = None
        for _ in range(10):
            reason = self.guard.feed(LOOP_PHRASE)
            if reason:
                break
        self.assertIsNotNone(reason)
        self.assertIn("tail x", reason)

    def test_silent_on_flowing_text(self):
        self.assertIsNone(self.guard.feed(FLOWING))

    def test_silent_below_tail_chars(self):
        self.assertIsNone(self.guard.feed("короткая мысль"))

    def test_empty_feed_is_noop(self):
        self.assertIsNone(self.guard.feed(""))

    def test_window_trim_still_fires(self):
        guard = LoopGuard(tail_chars=64, repeats=3, window_chars=200)
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


class TestConfigDefaults(unittest.TestCase):
    def test_defaults(self):
        self.assertTrue(config.LOOP_GUARD)
        self.assertEqual(config.LOOP_TAIL_CHARS, 64)
        self.assertEqual(config.LOOP_REPEATS, 3)
        self.assertEqual(config.LOOP_WINDOW_CHARS, 4096)
        self.assertEqual(config.LOOP_RETRIES, 1)
        self.assertEqual(config.LOOP_RETRY_FREQ_PENALTY, 0.5)


if __name__ == "__main__":
    unittest.main()

import unittest

from dev.benchmarks.loop_cd import final_number, summarize


class LoopCDBenchmarkTest(unittest.TestCase):
    def test_final_number_requires_an_answer_marker(self):
        self.assertEqual(final_number("Worked 3 steps.\n#### $1,234.00."), 1234)
        self.assertEqual(final_number("#### 2\nAnswer: -5"), -5)
        self.assertIsNone(final_number("Calculated 20 but ran out of tokens"))
        self.assertIsNone(final_number("#### 1/2"))

    def test_summary_uses_paired_answers_and_excludes_short_rates(self):
        def speed(mode, tokens, rate):
            return dict(mode=mode, kind="speed", timings=dict(
                predicted_n=tokens, predicted_per_second=rate, prompt_ms=100,
                draft_n=20, draft_n_accepted=5,
            ))

        records = [speed("off", 256, 20), speed("on", 256, 15), speed("on", 5, 100)]
        for mode in ("off", "on"):
            good = mode == "on"
            records.append(dict(mode=mode, kind="gsm8k", id=42, answer="1" if good else None, correct=good))
        summary = summarize(records)
        self.assertEqual(summary["decode_change_pct"], -25)
        self.assertEqual(summary["on"]["draft_acceptance"], 0.25)
        self.assertEqual(summary["off"]["gsm8k_unparsed"], 1)
        self.assertEqual(summary["paired_gsm8k"], {"wrong_to_right": 1, "right_to_wrong": 0})

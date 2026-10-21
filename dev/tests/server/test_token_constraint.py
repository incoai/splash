import array
import unittest
from types import SimpleNamespace

from llguidance import LLExecutor, LLMatcher

from dev.tests.server_fixtures import grammar_tokenizers
from server import backend as backend_api
from server import constraints
from server import protocol as wire
from server.errors import ConstraintError


class TokenConstraintTest(unittest.TestCase):
    """Token masks and checks from a compiled grammar over a byte vocabulary."""

    @classmethod
    def setUpClass(cls):
        tokenizer, guidance = grammar_tokenizers()

        class Constraint(constraints.TokenConstraint):
            VOCABULARY = guidance.vocab_size
            EOS_TOKENS = (tokenizer.token_to_id("<eos>"),)

        cls.Constraint = Constraint
        cls.width = (Constraint.VOCABULARY + 31) // 32
        cls.matcher = LLMatcher(
            guidance, '%llguidance {}\nstart: "ab" | "ac"\n', log_level=0
        )
        cls.executor = LLExecutor()
        cls.a, cls.b, cls.c, cls.z = map(tokenizer.token_to_id, "abcz")
        cls.eos = Constraint.EOS_TOKENS[0]

    def constraint(self):
        return self.Constraint(self.matcher.deep_copy(), self.executor)

    def rows(self, payload):
        """The allowed token IDs of each mask row."""
        words = array.array("I", payload)
        return [
            [
                index * 32 + bit
                for index, word in enumerate(words[start : start + self.width])
                for bit in range(32)
                if word >> bit & 1
            ]
            for start in range(0, len(words), self.width)
        ]

    def test_draft_rows_stop_following_the_draft_at_its_first_invalid_token(self):
        a, b, c, z, eos = self.a, self.b, self.c, self.z, self.eos
        constraint = self.constraint()
        self.assertEqual(self.rows(constraint.masks(())), [[a]])
        self.assertEqual(self.rows(constraint.masks((a, b))), [[a], [b, c], [eos]])
        for draft in ((a, z, b), (a, self.Constraint.VOCABULARY, b), (a, -1, b)):
            with self.subTest(draft=draft):
                self.assertEqual(
                    self.rows(constraint.masks(draft)), [[a], [b, c], [b, c], [b, c]]
                )
        # Simulated drafts do not advance the grammar.
        self.assertEqual(self.rows(constraint.masks(())), [[a]])
        provide = backend_api.NativeBackend._mask_provider(
            SimpleNamespace(constraint=constraint)
        )
        self.assertEqual(
            provide(wire.MaskRequestEvent(1, 1, self.width, (a, z))),
            constraint.masks((a, z)),
        )

    def test_masks_validate_drafts_without_copying_or_advancing_the_matcher(self):
        class Uncopyable:
            """The matcher, which masks() must not copy."""

            def __init__(self, matcher):
                self.matcher = matcher

            def deep_copy(self):
                raise AssertionError("masks() copied the matcher")

            def __getattr__(self, name):
                return getattr(self.matcher, name)

        class Executor:
            """Hands LLGuidance the matchers inside Uncopyable."""

            def __init__(self, executor):
                self.executor = executor

            def __getattr__(self, name):
                compute = getattr(self.executor, name)
                return lambda matchers, *mask: compute(
                    [(matcher.matcher, *rest) for matcher, *rest in matchers], *mask
                )

        a, z = self.a, self.z
        constraint = self.Constraint(
            Uncopyable(self.matcher.deep_copy()), Executor(self.executor)
        )
        rows = self.rows(constraint.masks((a, z)))
        constraint.commit([a])
        self.assertEqual(self.rows(constraint.masks(())), rows[1:2])

    def test_generated_tokens_advance_the_grammar_and_end_with_eos(self):
        a, b, c, eos = self.a, self.b, self.c, self.eos
        constraint = self.constraint()
        # Committed tokens advance the grammar before its next mask.
        constraint.commit([a])
        self.assertEqual(self.rows(constraint.masks(())), [[b, c]])
        constraint.commit([b])
        self.assertEqual(self.rows(constraint.masks(())), [[eos]])
        # LLGuidance's bulk API rejects EOS once the grammar has stopped.
        self.assertFalse(constraint.matcher.deep_copy().consume_tokens([eos]))
        constraint.commit([eos])
        constraint.finish()
        self.assertFalse(constraint.matcher.is_error())
        for tokens in ([self.z], [self.Constraint.VOCABULARY], [-1]):
            with (
                self.subTest(tokens=tokens),
                self.assertRaises(ConstraintError) as caught,
            ):
                rejecting = self.constraint()
                rejecting.commit(tokens)
                rejecting.finish()
            # The parser's state dump, generated text included, stays out.
            self.assertNotIn("\n", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

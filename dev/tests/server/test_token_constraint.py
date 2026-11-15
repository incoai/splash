import array
import dataclasses
import random
import unittest
from types import SimpleNamespace

import numpy as np
from llguidance import LLMatcher
from llguidance.numpy import allocate_token_bitmask, fill_next_token_bitmask

from dev.tests.server_fixtures import grammar_tokenizers
from server import backend as backend_api
from server import constraints
from server import protocol as wire
from server.errors import ConstraintError

NO_PARENT = wire.NO_PARENT


def chain(tokens):
    """The parents of a chain of `tokens`: each follows the one before it."""
    return (NO_PARENT, *range(len(tokens) - 1)) if tokens else ()


class TokenConstraintTest(unittest.TestCase):
    """Token masks and checks from a compiled grammar over a byte vocabulary."""

    @classmethod
    def setUpClass(cls):
        tokenizer, guidance = grammar_tokenizers()
        cls.tokenizer = tokenizer

        # What an engine whose vocabulary is the tokenizer's, which stops at
        # <eos>, announces, with the mask rows of a draft of eight.
        cls.ready = wire.ReadyEvent(
            4, 4096, False, guidance.vocab_size, 9, (tokenizer.token_to_id("<eos>"),)
        )
        cls.width = (cls.ready.vocabulary_size + 31) // 32
        cls.matcher = LLMatcher(
            guidance, '%llguidance {}\nstart: "ab" | "ac"\n', log_level=0
        )
        # A grammar whose paths branch: after "x", "ab" then any "c"s, or
        # "b" then "d" or "e"; "!" ends it.
        cls.branching = LLMatcher(
            guidance,
            '%llguidance {}\nstart: "x" ("ab" "c"* | "b" ("d" | "e")) "!"\n',
            log_level=0,
        )
        cls.json = LLMatcher(
            guidance,
            LLMatcher.grammar_from_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "tag": {"enum": ["ab", "ac", "b"]},
                    },
                    "required": ["id", "tag"],
                    "additionalProperties": False,
                }
            ),
            log_level=0,
        )
        cls.a, cls.b, cls.c, cls.z = map(tokenizer.token_to_id, "abcz")
        cls.eos = cls.ready.stop_token_ids[0]

    def constraint(self, matcher=None, rows=None):
        ready = (
            self.ready
            if rows is None
            else dataclasses.replace(self.ready, mask_rows=rows)
        )
        return constraints.TokenConstraint((matcher or self.matcher).deep_copy(), ready)

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
        self.assertEqual(self.rows(constraint.masks((), ())), [[a]])
        self.assertEqual(
            self.rows(constraint.masks((a, b), chain((a, b)))), [[a], [b, c], [eos]]
        )
        for draft in ((a, z, b), (a, self.ready.vocabulary_size, b), (a, -1, b)):
            with self.subTest(draft=draft):
                self.assertEqual(
                    self.rows(constraint.masks(draft, chain(draft))),
                    [[a], [b, c], [b, c], [b, c]],
                )
        # Simulated drafts do not advance the grammar.
        self.assertEqual(self.rows(constraint.masks((), ())), [[a]])
        provide = backend_api.NativeBackend._mask_provider(
            SimpleNamespace(constraint=constraint)
        )
        self.assertEqual(
            provide(wire.MaskRequestEvent(1, 1, self.width, (a, z), chain((a, z)))),
            constraint.masks((a, z), chain((a, z))),
        )

    def test_each_tree_row_masks_its_own_path(self):
        """A row's mask follows its path, not the rows packed before it: the
        anchor's second child comes after the first one's subtree, a rejected
        token's subtree keeps the state before it, and a stop token ends its
        path."""
        x, a, b, c, d, e, z, q, bang = map(self.tokenizer.token_to_id, "xabcdezq!")
        eos = self.eos
        tokens = (x, a, b, c, z, b, e, bang, q, eos)
        parents = (NO_PARENT, 0, 1, 2, 2, 0, 5, 6, 4, 7)
        constraint = self.constraint(self.branching, rows=33)
        self.assertEqual(
            self.rows(constraint.masks(tokens, parents)),
            [
                [x],  # before the anchor
                [a, b],  # x
                [b],  # x a
                [bang, c],  # x a b
                [bang, c],  # x a b c
                [bang, c],  # z, rejected after x a b
                [d, e],  # x b, whose packed predecessor is z
                [bang],  # x b e
                [eos],  # x b e !
                [bang, c],  # q, below the rejected z
                [eos],  # <eos> ends x b e !
            ],
        )
        # The walk leaves the grammar where it found it.
        self.assertEqual(self.rows(constraint.masks((), ())), [[x]])

    def test_tree_masks_match_a_matcher_copy_per_path(self):
        """On random trees of up to 16 rows over a JSON grammar, each row's
        mask is the one a matcher copied for its path alone finds: the state
        after the path's tokens up to the first the grammar rejects."""
        generator = random.Random(1114)
        vocabulary = self.ready.vocabulary_size
        bitmask = allocate_token_bitmask(1, vocabulary)

        def reference(path):
            matcher = self.json.deep_copy()
            for token in path:
                if (
                    not 0 <= token < vocabulary
                    or token == self.eos
                    or not matcher.try_consume_tokens([token])
                ):
                    break
            fill_next_token_bitmask(matcher, bitmask, 0)
            return bitmask[0].copy()

        prefix = self.tokenizer.encode('{"id": 1').ids
        constraint = self.constraint(self.json, rows=33)
        constraint.commit(prefix)
        for trial in range(200):
            tokens, parents, paths = [], [], []
            for row in range(generator.randint(1, 16)):
                parent = generator.randrange(row) if row else NO_PARENT
                path = paths[parent] if row else prefix
                allowed = np.flatnonzero(
                    np.unpackbits(reference(path).view(np.uint8), bitorder="little")
                )
                token = (
                    int(generator.choice(allowed))
                    if allowed.size and generator.random() < 0.85
                    else generator.randrange(vocabulary)
                )
                tokens.append(token)
                parents.append(parent)
                paths.append([*path, token])
            # Children in any order after their parents, as well as in
            # preorder.
            with self.subTest(trial=trial):
                masks = np.frombuffer(
                    constraint.masks(tuple(tokens), tuple(parents)), dtype=np.int32
                ).reshape(-1, self.width)
                expected = [reference(prefix), *(reference(path) for path in paths)]
                self.assertTrue(
                    all(np.array_equal(row, want) for row, want in zip(masks, expected))
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

        a, b, c, z = self.a, self.b, self.c, self.z
        constraint = constraints.TokenConstraint(
            Uncopyable(self.matcher.deep_copy()), self.ready
        )
        rows = self.rows(constraint.masks((a, b, z, c), (NO_PARENT, 0, 0, 0)))
        constraint.commit([a])
        self.assertEqual(self.rows(constraint.masks((), ())), rows[1:2])

    def test_generated_tokens_advance_the_grammar_and_end_with_eos(self):
        a, b, c, eos = self.a, self.b, self.c, self.eos
        constraint = self.constraint()
        # Committed tokens advance the grammar before its next mask.
        constraint.commit([a])
        self.assertEqual(self.rows(constraint.masks((), ())), [[b, c]])
        constraint.commit([b])
        self.assertEqual(self.rows(constraint.masks((), ())), [[eos]])
        # LLGuidance's bulk API rejects EOS once the grammar has stopped.
        self.assertFalse(constraint.matcher.deep_copy().consume_tokens([eos]))
        constraint.commit([eos])
        constraint.finish()
        self.assertFalse(constraint.matcher.is_error())
        for tokens in ([self.z], [self.ready.vocabulary_size], [-1]):
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

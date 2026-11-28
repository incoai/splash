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
        self.assertEqual(self.rows(constraint.masks((), ())[0]), [[a]])
        self.assertEqual(
            self.rows(constraint.masks((a, b), chain((a, b)))[0]), [[a], [b, c], [eos]]
        )
        for draft in ((a, z, b), (a, self.ready.vocabulary_size, b), (a, -1, b)):
            with self.subTest(draft=draft):
                self.assertEqual(
                    self.rows(constraint.masks(draft, chain(draft))[0]),
                    [[a], [b, c], [b, c], [b, c]],
                )
        # Simulated drafts do not advance the grammar.
        self.assertEqual(self.rows(constraint.masks((), ())[0]), [[a]])
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
            self.rows(constraint.masks(tokens, parents)[0]),
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
        self.assertEqual(self.rows(constraint.masks((), ())[0]), [[x]])

    def test_tree_masks_match_a_matcher_copy_per_path(self):
        """On random trees of up to 16 rows over a JSON grammar, each row's
        mask is the one a matcher copied for its path alone finds: the state
        after the path's tokens up to the first the grammar rejects; and so
        are the tokens the grammar forces after it, none past a rejection
        and none before the anchor, whose row no cycle ends at."""
        generator = random.Random(1114)
        vocabulary = self.ready.vocabulary_size
        bitmask = allocate_token_bitmask(1, vocabulary)

        def reference(path):
            """The mask after the path's tokens up to the first the grammar
            rejects, and the tokens it forces after the whole path, none
            where it rejects one."""
            matcher = self.json.deep_copy()
            reached = True
            for token in path:
                if (
                    not 0 <= token < vocabulary
                    or token == self.eos
                    or not matcher.try_consume_tokens([token])
                ):
                    reached = False
                    break
            fill_next_token_bitmask(matcher, bitmask, 0)
            forced = tuple(matcher.compute_ff_tokens()) if reached else ()
            return bitmask[0].copy(), forced

        prefix = self.tokenizer.encode('{"id": 1').ids
        constraint = self.constraint(self.json, rows=33)
        constraint.commit(prefix)
        for trial in range(200):
            tokens, parents, paths = [], [], []
            for row in range(generator.randint(1, 16)):
                parent = generator.randrange(row) if row else NO_PARENT
                path = paths[parent] if row else prefix
                allowed = np.flatnonzero(
                    np.unpackbits(reference(path)[0].view(np.uint8), bitorder="little")
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
                masks, forced = constraint.masks(tuple(tokens), tuple(parents))
                masks = np.frombuffer(masks, dtype=np.int32).reshape(-1, self.width)
                expected = [reference(prefix), *(reference(path) for path in paths)]
                self.assertTrue(
                    all(
                        np.array_equal(row, want[0])
                        for row, want in zip(masks, expected)
                    )
                )
                self.assertEqual(
                    forced,
                    tuple(
                        value
                        for row, (_, tokens) in enumerate(expected[1:], 1)
                        for token in tokens[:32]
                        for value in (row, token)
                    ),
                )

    def test_rows_carry_the_tokens_their_grammar_forces(self):
        """Beside its masks, a request gets the tokens the grammar forces
        after each row's path, as pairs of a row and a token: the rest of
        "abc!" once "xa" begins it, the rest of "bd!" after "xb", nothing
        after the free choice that follows "x", and nothing below a token the
        grammar rejects. A first token's request gets its one row's: the
        anchor; a verify's, none before the anchor, which every cycle
        retains."""
        x, a, b, c, d, z, bang = map(self.tokenizer.token_to_id, "xabcdz!")
        guidance = grammar_tokenizers()[1]
        matcher = LLMatcher(
            guidance, '%llguidance {}\nstart: "x" ("abc" | "bd") "!"\n', log_level=0
        )
        constraint = self.constraint(matcher)
        self.assertEqual(constraint.masks((), ())[1], (0, x))
        _, forced = constraint.masks((x, a, b, z, b), (NO_PARENT, 0, 1, 1, 0))
        self.assertEqual(forced, (2, b, 2, c, 2, bang, 3, c, 3, bang, 5, d, 5, bang))

    def test_forced_tokens_fit_the_response_the_engine_takes(self):
        """The engine takes a row and a token for as many tokens as a
        request simulates (Ready's mask rows less one) after each row of a
        response, 144 words where it simulates 8. Under an 8-token chain
        along a 40-token forced run each of the anchor's and its 7 proposals'
        rows forces the next 8 tokens; without the bound, 9 rows of up to 32
        tokens each sent 576 words, past the engine's frame."""
        run = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN!"
        guidance = grammar_tokenizers()[1]
        matcher = LLMatcher(
            guidance, f'%llguidance {{}}\nstart: "x" "{run}"\n', log_level=0
        )
        tokens = tuple(map(self.tokenizer.token_to_id, "x" + run))
        constraint = self.constraint(matcher)
        rows = self.ready.mask_rows
        _, forced = constraint.masks(tokens[:8], chain(tokens[:8]))
        self.assertLessEqual(len(forced), 2 * rows * (rows - 1))
        self.assertEqual(
            forced,
            tuple(
                value
                for row in range(1, 9)
                for token in tokens[row : row + 8]
                for value in (row, token)
            ),
        )

    def test_masks_walk_forward_without_advancing_the_committed_matcher(self):
        """No matcher takes a token back (masks()): the walk goes forward on
        copies of the committed matcher, which moves only by commits."""

        class ForwardOnly:
            """A matcher, and every copy of it, refusing to roll back."""

            def __init__(self, matcher):
                self.matcher = matcher

            def rollback(self, _count):
                raise AssertionError("masks() rolled a matcher back")

            def deep_copy(self):
                return ForwardOnly(self.matcher.deep_copy())

            def __getattr__(self, name):
                return getattr(self.matcher, name)

        a, b, c, z = self.a, self.b, self.c, self.z
        constraint = constraints.TokenConstraint(
            ForwardOnly(self.matcher.deep_copy()), self.ready
        )
        rows = self.rows(constraint.masks((a, b, z, c), (NO_PARENT, 0, 0, 0))[0])
        constraint.commit([a])
        self.assertEqual(self.rows(constraint.masks((), ())[0]), rows[1:2])

    def test_the_next_row_0_is_the_committed_path_s_mask(self):
        """#790: LLGuidance 1.8.0 keeps the last mask it computed, keyed
        by its lexer state and parser row index alone; after a rollback
        another branch that reaches the same key gets the mask of the branch
        it left. Under an array of at most three arrays of at most two
        integers, after "[[1" the branches ",2],[3" and "],[2,3" each end
        inside an integer after as many lexemes, but only the first lets a
        "," follow. When the tree's walk rolled back, its last mask, the
        second branch's, stayed cached: the cycle retained the first branch
        and drew "," as its anchor under that branch's row, and the next
        request's row 0, after the commit, was the cached mask, which
        refused it: "verify mask is not synchronized to the pending anchor".
        Every row is a fresh matcher's mask after its path, and the next row
        0 admits the anchor."""
        guidance = grammar_tokenizers()[1]
        schema = {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "array", "maxItems": 2, "items": {"type": "integer"}},
        }
        grammar = LLMatcher.grammar_from_json_schema(schema)

        def tokens(text):
            return [self.tokenizer.token_to_id(char) for char in text]

        def fresh(text):
            matcher = LLMatcher(guidance, grammar, log_level=0)
            self.assertTrue(matcher.consume_tokens(tokens(text)))
            bitmask = allocate_token_bitmask(1, self.ready.vocabulary_size)
            fill_next_token_bitmask(matcher, bitmask, 0)
            return self.rows(bitmask[0].tobytes())[0]

        retained, other = ",2],[3", "],[2,3"
        constraint = self.constraint(LLMatcher(guidance, grammar, log_level=0), rows=17)
        constraint.commit(tokens("[["))
        # The anchor "1", then both branches below it, the retained one first.
        tree = tokens("1" + retained + other)
        parents = (NO_PARENT, *range(6), 0, *range(7, 12))
        rows = self.rows(constraint.masks(tuple(tree), parents)[0])
        paths = ["[[", *("[[" + "1" + retained[:depth] for depth in range(7))]
        paths += ["[[1" + other[: depth + 1] for depth in range(6)]
        self.assertEqual(rows, [fresh(path) for path in paths])
        # The retained branch's last row admits ",", which the target draws
        # as its anchor.
        comma = tokens(",")[0]
        self.assertIn(comma, rows[7])
        constraint.commit(tokens("1" + retained))
        row0 = self.rows(constraint.masks((comma,), (NO_PARENT,))[0])[0]
        self.assertEqual(row0, fresh("[[1" + retained))
        self.assertIn(comma, row0)

    def test_generated_tokens_advance_the_grammar_and_end_with_eos(self):
        a, b, c, eos = self.a, self.b, self.c, self.eos
        constraint = self.constraint()
        # Committed tokens advance the grammar before its next mask.
        constraint.commit([a])
        self.assertEqual(self.rows(constraint.masks((), ())[0]), [[b, c]])
        constraint.commit([b])
        self.assertEqual(self.rows(constraint.masks((), ())[0]), [[eos]])
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

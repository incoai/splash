"""Text marker matching follows BPE pieces, including merged newlines."""

import unittest

from dev.tests.server.test_chat_formats import ripple_tokenizer
from server.token_markers import compile_markers


def advance(program, state, token):
    if state >= program[0]:
        return state
    kind = next(
        (entry & 255 for entry in program[program[7] :] if entry >> 8 == token), 0
    )
    index = state * program[1] + kind
    return program[8 + index // 4] >> (8 * (index % 4)) & 255


class TokenMarkersTests(unittest.TestCase):
    def test_merged_markers_and_ordinary_punctuation(self):
        tokenizer = ripple_tokenizer()
        end = tokenizer.encode("</think>", add_special_tokens=False)
        program = compile_markers(tokenizer, "<think>", "</think>", end)
        self.assertLessEqual(len(program) * 4, 4096)
        for text, expected in (
            ("<think>\nWe need the number.\n</think>\n4", program[3]),
            ("<think>We need the number.</think>", program[3]),
            ("<think>number > number", program[2]),
            ("Answer <think>", program[4]),
        ):
            state = program[5]
            for token in tokenizer.encode(text, add_special_tokens=False):
                state = advance(program, state, token)
            self.assertEqual(state, expected, text)
        matrix = (program[0] * program[1] + 3) // 4
        for state in range(program[2], program[0]):
            offset = program[8 + matrix + state]
            reached = state
            for token in program[offset + 1 : offset + 1 + program[offset]]:
                reached = advance(program, reached, token)
            self.assertEqual(reached, program[3], state)

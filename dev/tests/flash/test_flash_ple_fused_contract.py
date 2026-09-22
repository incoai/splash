"""CPU contract for a fused PLE hash/gather implementation.

This does not execute Metal or qualify a GPU implementation.  It checks the
proposed reciprocal reduction and two-hash reuse against the independently
implemented, segment-based reference, including source-row address boundaries.
Only tiny integer fixtures are created; checkpoint tables are never read.
"""

import importlib.util
import random
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "flash_ple_reference.py"
)
SPEC = importlib.util.spec_from_file_location(
    "flash_ple_fused_reference", REFERENCE_PATH
)
REF = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REF
SPEC.loader.exec_module(REF)

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
SIGN64 = 1 << 63
INT64_MAX = SIGN64 - 1
VOCABULARY = 248320


def mulhi64_limbs(left, right):
    """High U64 product using four widened U32 products and explicit carry."""
    if not 0 <= left <= MASK64 or not 0 <= right <= MASK64:
        raise ValueError("mulhi operands must be U64")
    low_low = (left & MASK32) * (right & MASK32)
    low_high = (left & MASK32) * (right >> 32)
    high_low = (left >> 32) * (right & MASK32)
    high_high = (left >> 32) * (right >> 32)
    middle = (low_low >> 32) + (low_high & MASK32) + (high_low & MASK32)
    return high_high + (low_high >> 32) + (high_low >> 32) + (middle >> 32)


def reciprocal_remainder_u64(value, divisor):
    """Exact U64 remainder, floor reciprocal, and at most one correction.

    R=floor(2**64/d) gives q=high(value*R) <= floor(value/d), with quotient
    underestimation at most one.  Thus value-q*d is in [0,2*d).  The d=1
    case is separate because its reciprocal would need 65 bits.
    """
    if not 0 <= value <= MASK64 or not 1 <= divisor <= INT64_MAX:
        raise ValueError("invalid unsigned reduction")
    if divisor == 1:
        return 0
    reciprocal = (1 << 64) // divisor
    quotient = mulhi64_limbs(value, reciprocal)
    remainder = value - quotient * divisor
    return remainder - divisor if remainder >= divisor else remainder


def reciprocal_positive_signed_remainder(bits, divisor):
    """Nonnegative signed I64 modulo, including the INT64_MIN bit pattern."""
    if not 0 <= bits <= MASK64:
        raise ValueError("hash must be a U64 bit pattern")
    negative = bool(bits & SIGN64)
    magnitude = ((~bits + 1) & MASK64) if negative else bits
    remainder = reciprocal_remainder_u64(magnitude, divisor)
    return divisor - remainder if negative and remainder else remainder


def shared_hash_pair(current, previous, previous2, multipliers):
    """Wrap products in U64, then reuse one bigram and one trigram per row."""
    bigram = ((current * (multipliers[0] & MASK64)) & MASK64) ^ (
        (previous * (multipliers[1] & MASK64)) & MASK64
    )
    trigram = bigram ^ ((previous2 * (multipliers[2] & MASK64)) & MASK64)
    return bigram, trigram


def fused_gpu_contract(
    tokens,
    histories,
    multipliers=REF.MULTIPLIERS,
    sizes=REF.HEAD_SIZES,
    offsets=REF.HEAD_OFFSETS,
    *,
    eos=REF.EOS,
    vocabulary=VOCABULARY,
    table_rows=REF.TABLE_ROWS,
):
    """Model current shader diagnostics and separate ordered history update.

    Input layout is [lanes, real rows], history is [lanes, 2].  Diagnostics
    bit1 marks invalid used token IDs; bit2 marks invalid stored head params.
    A failed head emits -1, which a gather must turn into NaN before dereference.
    Unlike the stricter host reference, a masked older history entry is unused.
    """
    if (
        not tokens
        or not tokens[0]
        or len(histories) != len(tokens)
        or any(len(lane) != len(tokens[0]) for lane in tokens)
        or any(len(history) != 2 for history in histories)
        or len(multipliers) != 3
        or len(sizes) != 16
        or len(offsets) != 16
        or not 0 < table_rows <= INT64_MAX
        or not 0 <= eos < vocabulary
    ):
        raise ValueError("invalid fused PLE extents")
    source_history = tuple(tuple(history) for history in histories)
    result = []
    diagnostics = 0

    def valid_token(value):
        return 0 <= value < vocabulary

    for lane, lane_tokens in enumerate(tokens):
        lane_ids = []
        for row, current in enumerate(lane_tokens):
            previous = lane_tokens[row - 1] if row else source_history[lane][1]
            older = (
                lane_tokens[row - 2]
                if row > 1
                else source_history[lane][1 if row else 0]
            )
            previous2 = eos if previous == eos else older
            if not all(map(valid_token, (current, previous, previous2))):
                diagnostics |= 1
                lane_ids.append([-1] * 16)
                continue
            mixed = shared_hash_pair(current, previous, previous2, multipliers)
            row_ids = []
            for head, (size, offset) in enumerate(zip(sizes, offsets)):
                # Subtraction-based extent check avoids overflow at INT64_MAX.
                if (
                    size <= 0
                    or offset < 0
                    or offset >= table_rows
                    or size > table_rows - offset
                ):
                    diagnostics |= 2
                    row_ids.append(-1)
                else:
                    row_ids.append(
                        reciprocal_positive_signed_remainder(mixed[head // 8], size)
                        + offset
                    )
            lane_ids.append(row_ids)
        result.append(lane_ids)

    # All hash reads above used the immutable incoming history.  This models
    # the second dispatch; a bad incoming token leaves its whole lane unchanged.
    final_history = []
    for lane, lane_tokens in enumerate(tokens):
        if all(map(valid_token, lane_tokens)):
            final_history.append(list((list(source_history[lane]) + lane_tokens)[-2:]))
        else:
            diagnostics |= 1
            final_history.append(list(source_history[lane]))
    return np.asarray(result, dtype=np.int64), final_history, diagnostics


def source_address(
    source_id,
    column,
    *,
    rows_per_shard=2500012,
    shards=128,
    weight_stride=80,
    parameter_stride=10,
):
    """Global hashed row -> original shard, local row, and W/S/B byte offsets."""
    if (
        not 0 <= source_id < rows_per_shard * shards
        or not 0 <= column < 160
        or rows_per_shard <= 0
        or not 1 <= shards <= 128
        or weight_stride < 80
        or parameter_stride < 10
        or parameter_stride % 2
    ):
        raise ValueError("invalid PLE source address")
    shard, local_row = divmod(source_id, rows_per_shard)
    return (
        shard,
        local_row,
        local_row * weight_stride + column // 2,
        (column % 2) * 4,
        local_row * parameter_stride + (column // 32) * 2,
    )


class FlashPLEFusedContractTests(unittest.TestCase):
    def test_mulhi_carries_match_unbounded_product(self):
        corners = (0, 1, MASK32, 1 << 32, SIGN64 - 1, SIGN64, MASK64)
        pairs = [(a, b) for a in corners for b in corners]
        rng = random.Random(3816)
        pairs.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(5000))
        for left, right in pairs:
            self.assertEqual(mulhi64_limbs(left, right), (left * right) >> 64)

    def test_unsigned_reciprocal_requires_at_most_one_correction(self):
        divisors = (1, 2, 3, 17, MASK32, 1 << 32, INT64_MAX) + REF.HEAD_SIZES
        rng = random.Random(816)
        for divisor in divisors:
            values = (0, 1, divisor - 1, divisor, divisor + 1, SIGN64, MASK64)
            values += tuple(rng.getrandbits(64) for _ in range(1024))
            for value in values:
                self.assertEqual(
                    reciprocal_remainder_u64(value, divisor), value % divisor
                )
                if divisor > 1:
                    quotient = mulhi64_limbs(value, (1 << 64) // divisor)
                    self.assertIn(value // divisor - quotient, (0, 1))
                    self.assertLess(value - quotient * divisor, 2 * divisor)

    def test_signed_min_and_negative_hash_use_positive_source_remainder(self):
        signed_values = (-SIGN64, -INT64_MAX, -17, -1, 0, 1, 17, INT64_MAX)
        for divisor in (1, 2, 3, 17, INT64_MAX) + REF.HEAD_SIZES:
            for value in signed_values:
                bits = struct.unpack("<Q", struct.pack("<q", value))[0]
                self.assertEqual(
                    reciprocal_positive_signed_remainder(bits, divisor), value % divisor
                )
        # Unsigned % on the same bits would silently choose the wrong table row.
        self.assertNotEqual(MASK64 % 17, (-1) % 17)
        self.assertEqual(reciprocal_positive_signed_remainder(SIGN64, 1), 0)

    def test_real_checkpoint_products_never_set_sign_bit(self):
        for multiplier in REF.MULTIPLIERS:
            self.assertGreater(multiplier, 0)
            self.assertLess((VOCABULARY - 1) * multiplier, SIGN64)
        # This fact supports a qualified checkpoint-specific unsigned route;
        # it does not justify deleting signed handling for synthetic params.
        pair = shared_hash_pair(
            VOCABULARY - 1, REF.EOS, VOCABULARY - 1, REF.MULTIPLIERS
        )
        self.assertTrue(all(value < SIGN64 for value in pair))
        self.assertEqual(shared_hash_pair(1, 0, 0, (-SIGN64, 0, 0))[0], SIGN64)

    def test_two_hashes_keep_all_distinct_head_sizes_and_offsets(self):
        tokens, history = [[1, 19, REF.EOS, 4]], [[17, 18]]
        actual, final_history, diagnostics = fused_gpu_contract(tokens, history)
        expected, expected_history = REF.ngram_indices(tokens[0], history[0])
        np.testing.assert_array_equal(actual[0], expected)
        self.assertEqual(final_history, [expected_history])
        self.assertEqual(diagnostics, 0)
        self.assertEqual(
            len(set(actual[0, 0, :8] - np.asarray(REF.HEAD_OFFSETS[:8]))), 8
        )

    def test_random_wraparound_and_eos_match_independent_segment_reference(self):
        rng = random.Random(64032)
        for _ in range(128):
            rows, lanes = rng.choice((1, 2, 3, 4, 7, 8, 19, 32)), rng.randrange(1, 5)
            tokens = [
                [
                    REF.EOS if rng.randrange(5) == 0 else rng.randrange(VOCABULARY)
                    for _ in range(rows)
                ]
                for _ in range(lanes)
            ]
            histories = [
                [rng.choice((REF.EOS, rng.randrange(VOCABULARY))) for _ in range(2)]
                for _ in range(lanes)
            ]
            multipliers = tuple(rng.randrange(-SIGN64, SIGN64) for _ in range(3))
            actual, after, diagnostics = fused_gpu_contract(
                tokens, histories, multipliers
            )
            self.assertEqual(diagnostics, 0)
            for lane in range(lanes):
                expected, expected_after = REF.ngram_indices(
                    tokens[lane], histories[lane], multipliers=multipliers
                )
                np.testing.assert_array_equal(actual[lane], expected)
                self.assertEqual(after[lane], expected_after)

    def test_current_eos_sees_context_and_only_following_token_resets(self):
        tokens = [[7, 8, REF.EOS, 9, 10, REF.EOS, REF.EOS, 11]]
        actual, _, _ = fused_gpu_contract(tokens, [[3, 4]])
        reset, _, _ = fused_gpu_contract([[9, 10]], [[REF.EOS, REF.EOS]])
        np.testing.assert_array_equal(actual[0, 3:5], reset[0])
        reset_eos, _, _ = fused_gpu_contract([[REF.EOS]], [[REF.EOS, REF.EOS]])
        self.assertTrue(np.any(actual[0, 2] != reset_eos[0, 0]))
        np.testing.assert_array_equal(actual[0, 6], reset_eos[0, 0])

    def test_every_chunk_boundary_and_lane_permutation_preserve_output(self):
        tokens = [
            [2, REF.EOS, 3, 4, REF.EOS, 5, 6, 7],
            [REF.EOS, 8, 9, 10, 11, REF.EOS, REF.EOS, 12],
        ]
        histories = [[31, 32], [REF.EOS, REF.EOS]]
        complete, after, _ = fused_gpu_contract(tokens, histories)
        for split in range(1, len(tokens[0])):
            first, state, _ = fused_gpu_contract(
                [lane[:split] for lane in tokens], histories
            )
            second, state, _ = fused_gpu_contract(
                [lane[split:] for lane in tokens], state
            )
            np.testing.assert_array_equal(
                np.concatenate((first, second), axis=1), complete
            )
            self.assertEqual(state, after)
        permuted, permuted_after, _ = fused_gpu_contract(tokens[::-1], histories[::-1])
        np.testing.assert_array_equal(permuted[::-1], complete)
        self.assertEqual(permuted_after[::-1], after)
        self.assertEqual(histories, [[31, 32], [REF.EOS, REF.EOS]])

    def test_invalid_token_precedes_param_diagnostic_and_preserves_lane_history(self):
        for invalid in (-1, VOCABULARY, INT64_MAX):
            tokens, history = [[3, invalid, 4], [5, 6, 7]], [[1, 2], [8, 9]]
            actual, after, diagnostics = fused_gpu_contract(tokens, history)
            self.assertEqual(diagnostics, 1)
            self.assertTrue(np.all(actual[0, 1:] == -1))
            self.assertEqual(after, [[1, 2], [6, 7]])
            _, _, diagnostic = fused_gpu_contract(
                [[invalid]], [[1, 2]], sizes=(0,) * 16
            )
            self.assertEqual(diagnostic, 1)

    def test_unused_older_is_masked_after_eos_but_used_invalid_history_fails(self):
        actual, after, diagnostic = fused_gpu_contract([[13]], [[-1, REF.EOS]])
        cold, _, _ = fused_gpu_contract([[13]], [[REF.EOS, REF.EOS]])
        np.testing.assert_array_equal(actual, cold)
        self.assertEqual((after, diagnostic), ([[REF.EOS, 13]], 0))
        for history in ((-1, 12), (3, -1), (VOCABULARY, 12)):
            actual, after, diagnostic = fused_gpu_contract([[13]], [list(history)])
            self.assertTrue(np.all(actual == -1))
            self.assertEqual(diagnostic, 1)
            # The ordered updater checks incoming tokens, not initial history.
            self.assertEqual(after, [[history[1], 13]])

    def test_head_params_are_independent_and_do_not_suppress_history_update(self):
        sizes, offsets = list(REF.HEAD_SIZES), list(REF.HEAD_OFFSETS)
        for head, size, offset in (
            (0, 0, 0),
            (7, -1, 0),
            (8, 1, -1),
            (15, 1, REF.TABLE_ROWS),
            (9, INT64_MAX, 3),
        ):
            bad_sizes, bad_offsets = sizes.copy(), offsets.copy()
            bad_sizes[head], bad_offsets[head] = size, offset
            actual, after, diagnostic = fused_gpu_contract(
                [[13]], [[11, 12]], sizes=bad_sizes, offsets=bad_offsets
            )
            self.assertEqual(diagnostic, 2)
            self.assertEqual(actual[0, 0, head], -1)
            self.assertEqual(np.count_nonzero(actual == -1), 1)
            self.assertEqual(after, [[12, 13]])

    def test_offset_addition_stays_below_i64_limit(self):
        actual, _, diagnostic = fused_gpu_contract(
            [[13]],
            [[11, 12]],
            sizes=(INT64_MAX,) * 16,
            offsets=(0,) * 16,
            table_rows=INT64_MAX,
        )
        self.assertEqual(diagnostic, 0)
        self.assertTrue(np.all(actual >= 0))
        self.assertTrue(np.all(actual < INT64_MAX))
        actual, _, diagnostic = fused_gpu_contract(
            [[13]],
            [[11, 12]],
            sizes=(1,) * 16,
            offsets=(INT64_MAX - 1,) * 16,
            table_rows=INT64_MAX,
        )
        self.assertTrue(np.all(actual == INT64_MAX - 1))
        self.assertEqual(diagnostic, 0)

    def test_all_128_original_shards_first_and_last_addresses(self):
        for shard in range(128):
            for row in (0, 2500011):
                for column in (0, 1, 31, 32, 63, 64, 158, 159):
                    address = source_address(shard * 2500012 + row, column)
                    self.assertEqual(
                        address,
                        (
                            shard,
                            row,
                            row * 80 + column // 2,
                            (column % 2) * 4,
                            row * 10 + column // 32 * 2,
                        ),
                    )
        self.assertEqual(
            source_address(320001535, 159),
            (127, 2500011, 2500011 * 80 + 79, 4, 2500011 * 10 + 8),
        )

    def test_address_padding_and_bad_id_fail_before_source_access(self):
        self.assertEqual(
            source_address(
                23, 159, rows_per_shard=3, weight_stride=96, parameter_stride=16
            ),
            (7, 2, 271, 4, 40),
        )
        for source_id, column in ((-1, 0), (320001536, 0), (0, -1), (0, 160)):
            with self.assertRaises(ValueError):
                source_address(source_id, column)
        for kwargs in (
            {"shards": 129},
            {"weight_stride": 79},
            {"parameter_stride": 9},
            {"parameter_stride": 11},
        ):
            with self.assertRaises(ValueError):
                source_address(0, 0, **kwargs)

    def test_invalid_geometry_and_reciprocal_inputs_fail(self):
        for tokens, histories in (
            ([], []),
            ([[]], [[1, 2]]),
            ([[1], [2, 3]], [[1, 2], [3, 4]]),
            ([[1]], []),
            ([[1]], [[1]]),
        ):
            with self.assertRaises(ValueError):
                fused_gpu_contract(tokens, histories)
        for value, divisor in (
            (-1, 17),
            (MASK64 + 1, 17),
            (1, 0),
            (1, -1),
            (1, SIGN64),
        ):
            with self.assertRaises(ValueError):
                reciprocal_remainder_u64(value, divisor)


if __name__ == "__main__":
    unittest.main()

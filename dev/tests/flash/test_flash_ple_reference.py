"""CPU-only PLE fixtures: no table materialization or GPU commands."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "flash_ple_reference.py"
SPEC = importlib.util.spec_from_file_location("flash_ple_reference", PATH)
REF = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REF
SPEC.loader.exec_module(REF)


class FlashPleReferenceTests(unittest.TestCase):
    def test_bf16_rounds_ties_even_and_preserves_nan(self):
        raw = np.array([0x3F808000, 0x3F818000, 0x7F800001], dtype=np.uint32)
        rounded = REF.bf16(raw.view(np.float32)).view(np.uint32)
        np.testing.assert_array_equal(rounded[:2], [0x3F800000, 0x3F820000])
        self.assertTrue(np.isnan(rounded[2:].view(np.float32)[0]))

    def test_eos_token_sees_previous_context_then_resets(self):
        tokens = [11, 12, REF.EOS, 13, 14]
        self.assertEqual(
            REF.shifted_without_crossing_eos(tokens, 1), [REF.EOS, 11, 12, REF.EOS, 13]
        )
        self.assertEqual(
            REF.shifted_without_crossing_eos(tokens, 2),
            [REF.EOS, REF.EOS, 11, REF.EOS, REF.EOS],
        )

    def test_hash_streaming_matches_one_chunk_across_eos(self):
        tokens = [7, 8, REF.EOS, 9, 10, 11, REF.EOS, 12]
        all_rows, all_history = REF.ngram_indices(tokens)
        history, pieces = None, []
        for piece in (tokens[:2], tokens[2:3], tokens[3:6], tokens[6:]):
            rows, history = REF.ngram_indices(piece, history)
            pieces.append(rows)
        np.testing.assert_array_equal(all_rows, np.concatenate(pieces))
        self.assertEqual(history, all_history)
        reset_rows, _ = REF.ngram_indices([9])
        np.testing.assert_array_equal(all_rows[3], reset_rows[0])
        for head in range(16):
            self.assertTrue(np.all(all_rows[:, head] >= REF.HEAD_OFFSETS[head]))
            self.assertTrue(
                np.all(
                    all_rows[:, head] < REF.HEAD_OFFSETS[head] + REF.HEAD_SIZES[head]
                )
            )

    def test_hash_uses_signed_i64_overflow_and_positive_remainder(self):
        multiplier = 9223372036854775807
        rows, _ = REF.ngram_indices(
            [2, 3],
            previous_context=[0, 0],
            multipliers=(multiplier, 0, 0),
            sizes=(17,) * 16,
            offsets=(0,) * 16,
        )
        self.assertEqual(REF.signed_i64(2 * multiplier), -2)
        np.testing.assert_array_equal(rows[0], [15] * 16)
        self.assertEqual(REF.signed_i64(3 * multiplier), multiplier - 2)
        np.testing.assert_array_equal(rows[1], [(multiplier - 2) % 17] * 16)

    def test_checkpoint_geometry_and_shard_boundaries(self):
        offsets = REF.shard_offsets()
        self.assertEqual(offsets[-1], 320001536)
        self.assertEqual(set(np.diff(offsets)), {2500012})
        self.assertEqual(REF.shard_for_row(2500011), (0, 2500011))
        self.assertEqual(REF.shard_for_row(2500012), (1, 0))
        self.assertEqual(REF.shard_for_row(320001535), (127, 2500011))
        for row in (-1, 320001536):
            with self.assertRaises(IndexError):
                REF.shard_for_row(row)
        self.assertEqual(REF.shard_offsets(10, 3), (0, 4, 7, 10))

    def test_affine_low_nibble_first_and_group_scales(self):
        words = np.array([0x76543210, 0xFEDCBA98] * 10, dtype=np.uint32)
        scales = np.array([0.125, -0.25, 0.5, -1, 2], dtype=np.float32)
        biases = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        output = REF.decode_affine_row(words, scales, biases)
        expected = [scales[i // 32] * (i % 16) + biases[i // 32] for i in range(160)]
        np.testing.assert_array_equal(output, REF.bf16(expected))

    def test_affine_rounding_variants_keep_checkpoint_trap_visible(self):
        words = np.full(20, 0x77777777, dtype=np.uint32)
        scales = np.full(5, -14.25, dtype=np.float32)
        biases = np.full(5, 128, dtype=np.float32)
        final_only = REF.decode_affine_row(words, scales, biases)
        intermediate = REF.decode_affine_row(
            words, scales, biases, intermediate_bf16_product=True
        )
        np.testing.assert_array_equal(final_only, np.full(160, 28.25))
        np.testing.assert_array_equal(intermediate, np.full(160, 28.0))

    def test_gather_reads_selected_rows_and_scales_after_bf16_rounding(self):
        calls = []
        words = np.array([0x77777777] * 20, dtype=np.uint32)
        scales = REF.bf16([0.0101318359375] * 5)
        biases = REF.bf16([0.404296875] * 5)

        def reader(shard, row):
            calls.append((shard, row))
            return words, scales, biases

        indices = [[0, 2500012, 320001535]]
        result = REF.gather_embeddings(indices, reader)
        self.assertEqual(calls, [(0, 0), (1, 0), (127, 2500011)])
        self.assertEqual(result.shape, (1, 3, 160))
        decoded = REF.bf16(np.float32(7) * scales[0] + biases[0])
        expected = REF.bf16(decoded * REF.SHARED_WEIGHT_SCALE)
        np.testing.assert_array_equal(result, np.full(result.shape, expected))
        # This fixture detects incorrectly folding shared scaling into affine
        # coefficients and losing the intermediate BF16 boundary.
        folded = REF.bf16(
            (np.float32(7) * scales[0] + biases[0]) * REF.SHARED_WEIGHT_SCALE
        )
        self.assertNotEqual(float(expected), float(folded))

    def test_grouped_norm_keeps_groups_independent_and_one_plus_in_float32(self):
        x = REF.bf16([[[1, 2, 10, 20]]])
        weight = REF.bf16([0.00390625] * 4)
        output = REF.grouped_rms_norm(x, weight, 2)
        expected = REF.bf16(
            x
            / np.sqrt(np.array([[[2.5, 2.5, 250, 250]]], dtype=np.float32) + 1e-6)
            * np.float32(1.00390625)
        )
        np.testing.assert_array_equal(output, expected)
        first = REF.grouped_rms_norm(x[..., :2], weight[:2], 2)
        np.testing.assert_array_equal(output[..., :2], first)
        direct = REF.grouped_rms_norm(x, np.ones(4), 2, one_plus_weight=False)
        zero = REF.grouped_rms_norm(x, np.zeros(4), 2)
        np.testing.assert_array_equal(direct, zero)

    def test_cpu_sigmoid_retains_float_promotions(self):
        # Values independently evaluated in an explicit MLX CPU stream. The
        # float32 sigmoid and BF16-rounded denominator both fail these cases.
        x = REF.bf16([-5.9375, -2, -0.2, 0, 0.2, 2])
        expected = [
            0.0026397705078125,
            0.11962890625,
            0.451171875,
            0.5,
            0.55078125,
            0.87890625,
        ]
        np.testing.assert_array_equal(REF.sigmoid_cpu_bf16(x), expected)

    def test_metal_sigmoid_retains_integer_literal_bf16_rounding(self):
        x = REF.bf16([-5.9375, -2, -0.2, 0, 0.2, 2])
        expected = [
            0.00262451171875,
            0.11962890625,
            0.451171875,
            0.5,
            0.546875,
            0.87890625,
        ]
        np.testing.assert_array_equal(REF.sigmoid_metal_bf16(x), expected)
        np.testing.assert_array_equal(REF.sigmoid_bf16(x), expected)
        self.assertTrue(np.any(REF.sigmoid_cpu_bf16(x) != expected))

    def test_post_defaults_to_metal_and_cpu_mode_is_explicit(self):
        hidden, key, value, norms, conv = self.fixture(3)
        default = REF.ple_post(hidden, key, value, *norms, conv, hidden_size=2)
        metal = REF.ple_post(
            hidden, key, value, *norms, conv, hidden_size=2, sigmoid_mode="metal"
        )
        cpu = REF.ple_post(
            hidden, key, value, *norms, conv, hidden_size=2, sigmoid_mode="cpu"
        )
        np.testing.assert_array_equal(default.output, metal.output)
        self.assertTrue(np.any(cpu.output != metal.output))
        with self.assertRaises(ValueError):
            REF.ple_post(
                hidden, key, value, *norms, conv, hidden_size=2, sigmoid_mode="invalid"
            )

    def test_small_metal_row_sum_rounds_each_addition(self):
        products = np.zeros(32, dtype=np.float32)
        products[:4] = [1, 0.00390625, -1, 0]
        self.assertEqual(float(REF.mlx_metal_bf16_row_sum(products)), 0)
        self.assertEqual(
            float(REF.bf16(np.sum(products, dtype=np.float32))), 0.00390625
        )
        batched = np.stack([products, -products]).reshape(1, 2, 32)
        self.assertEqual(
            REF.mlx_metal_bf16_row_sum(batched, keepdims=True).shape, (1, 2, 1)
        )
        np.testing.assert_array_equal(REF.mlx_metal_bf16_row_sum(batched), [[0, 0]])

    def test_full_width_metal_row_sum_retains_four_contiguous_local_reads(self):
        # Each of 640 native threads receives the same four values. The first
        # addition is a tie at 1.0 and rounds away the small term before -1.
        products = np.tile([1, 0.00390625, -1, 0], 640).astype(np.float32)
        self.assertEqual(products.shape, (2560,))
        self.assertEqual(float(REF.mlx_metal_bf16_row_sum(products)), 0)
        self.assertEqual(float(REF.bf16(np.sum(products, dtype=np.float32))), 2.5)

    def test_metal_row_sum_handles_thread_bucket_tails_and_multiple_blocks(self):
        # Exact integer sums distinguish dropped tails, wrong shapes and a
        # missing second block without depending on a float32 reduction tree.
        for width in (
            1,
            32,
            64,
            65,
            127,
            128,
            129,
            511,
            512,
            513,
            1023,
            1024,
            1025,
            2560,
            4097,
            8193,
        ):
            products = np.zeros((2, width), dtype=np.float32)
            products[:, 0] = [1, -1]
            products[:, -1] += [2, -2]
            np.testing.assert_array_equal(
                REF.mlx_metal_bf16_row_sum(products), [3, -3], err_msg=f"width={width}"
            )
        for products in (np.float32(1), np.empty((1, 0), dtype=np.float32)):
            with self.assertRaises(ValueError):
                REF.mlx_metal_bf16_row_sum(products)

    def fixture(self, length=19):
        rng = np.random.default_rng(219)
        hidden = REF.bf16(rng.normal(size=(2, length, 8)))
        key = REF.bf16(rng.normal(size=hidden.shape))
        value = REF.bf16(rng.normal(size=(2, length, 2)))
        norms = [REF.bf16(rng.normal(0, 0.1, 8)) for _ in range(3)]
        conv = REF.bf16(rng.normal(0, 0.1, (8, 4, 1)))
        return hidden, key, value, norms, conv

    def test_convolution_dilation_three_streams_match_prefill(self):
        hidden, key, value, norms, conv = self.fixture()
        full = REF.ple_post(hidden, key, value, *norms, conv, hidden_size=2)
        state, outputs = None, []
        for begin, end in ((0, 1), (1, 5), (5, 9), (9, 19)):
            part = REF.ple_post(
                hidden[:, begin:end],
                key[:, begin:end],
                value[:, begin:end],
                *norms,
                conv,
                hidden_size=2,
                state=state,
            )
            state = part.conv_state
            outputs.append(part.output)
        np.testing.assert_array_equal(full.output, np.concatenate(outputs, axis=1))
        np.testing.assert_array_equal(full.conv_state, state)
        self.assertEqual(full.conv_state.shape, (2, 9, 8))

    def test_mask_zeros_incoming_conv_state_but_previous_conv_can_emit(self):
        hidden, key, value, norms, conv = self.fixture(2)
        state = REF.bf16(np.ones((2, 9, 8)))
        mask = np.array([[False, True], [False, False]])
        result = REF.ple_post(
            hidden, key, value, *norms, conv, hidden_size=2, state=state, mask=mask
        )
        np.testing.assert_array_equal(result.gated_values[:, 0], np.zeros((2, 8)))
        np.testing.assert_array_equal(result.normed_conv_inputs[:, 0], np.zeros((2, 8)))
        self.assertTrue(np.any(result.output[:, 0] != 0))
        np.testing.assert_array_equal(
            result.conv_state[:, -2:], result.normed_conv_inputs
        )

    def test_zero_gate_keeps_sign_zero_and_half_value(self):
        hidden = REF.bf16(np.ones((1, 1, 8)))
        key = np.zeros_like(hidden)
        value = REF.bf16([[[3, -5]]])
        norms = [np.zeros(8)] * 3
        conv = np.zeros((8, 4, 1))
        result = REF.ple_post(hidden, key, value, *norms, conv, hidden_size=2)
        np.testing.assert_array_equal(result.gate, np.zeros((1, 1, 4, 1)))
        expected = np.tile([1.5, -2.5], (1, 1, 4))
        np.testing.assert_array_equal(result.output, expected)
        np.testing.assert_array_equal(
            REF.inject(hidden, result.output), REF.bf16(hidden + expected)
        )

    def test_shapes_fail_instead_of_silently_broadcasting(self):
        with self.assertRaises(ValueError):
            REF.ngram_indices([1], [REF.EOS])
        with self.assertRaises(ValueError):
            REF.decode_affine_row(np.zeros(19), np.zeros(5), np.zeros(5))
        with self.assertRaises(ValueError):
            REF.inject(np.zeros((1, 2)), np.zeros((1, 1)))


if __name__ == "__main__":
    unittest.main()

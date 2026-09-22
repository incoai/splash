#!/usr/bin/env python3
"""Independent CPU audit of original-F32 cached HC-down candidate contracts.

The literal row-reuse audit compares every F32 lane partial before simd_sum:
if those bits agree, the unchanged hardware reduction receives identical data.
CPU arithmetic cannot predict Metal fast exp or MPP's hardware reduction order.
These are contract tests, not a GPU or end-to-end qualification.
"""

from __future__ import annotations

import json
import math
import struct
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1/manifest.json"


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bits32(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def number(word):
    return struct.unpack("<f", struct.pack("<I", int(word) << 16))[0]


def bf16(value):
    word = bits32(value)
    high, remainder = divmod(word, 65536)
    if word & 0x7F800000 == 0x7F800000:
        return high | (0x40 if word & 0x7FFFFF else 0)
    return (high + int(remainder > 32768 or (remainder == 32768 and high & 1))) & 65535


def sigmoid_trace(word, *, exponent_word=None):
    value = number(word)
    if exponent_word is None:
        try:
            exponential = f32(math.exp(abs(value)))
        except OverflowError:
            exponential = math.inf
        exponent_word = bf16(exponential)
    denominator = bf16(1 + number(exponent_word))
    tail = bf16(1 / number(denominator))
    sigmoid = tail if value < 0 else bf16(1 - number(tail))
    return exponent_word, denominator, tail, sigmoid


def hc_down_epilog(raw_f32, *, exponent_word=None):
    raw = bf16(raw_f32)
    divided = bf16(number(raw) / 4)
    sigmoid = sigmoid_trace(divided, exponent_word=exponent_word)[-1]
    activated = bf16(number(divided) * number(sigmoid))
    diagnostics = (
        4
        if any(
            not math.isfinite(v) for v in [f32(raw_f32), number(raw), number(activated)]
        )
        else 0
    )
    return raw, divided, sigmoid, activated, diagnostics


def injection_epilog(raw_f32, *, exponent_word=None):
    raw = bf16(raw_f32)
    divided = bf16(number(raw) / 4)
    sigmoid = sigmoid_trace(divided, exponent_word=exponent_word)[-1]
    gate = bf16(2 * number(sigmoid))
    diagnostics = (
        4
        if any(not math.isfinite(v) for v in [f32(raw_f32), number(raw), number(gate)])
        else 0
    )
    return raw, divided, sigmoid, gate, diagnostics


def pack_codes(codes, bits):
    # Bit-by-bit encoder independent of shader's byte/shift extraction.
    packed = bytearray((len(codes) * bits + 7) // 8)
    for k, code in enumerate(codes):
        for bit in range(bits):
            linear = k * bits + bit
            packed[linear // 8] |= ((int(code) >> bit) & 1) << (linear % 8)
    return bytes(packed)


def unpack_code(packed, bits, k):
    bit = k * bits
    byte, shift = divmod(bit, 8)
    word = packed[byte]
    if shift + bits > 8:
        word |= packed[byte + 1] << 8
    return (word >> shift) & ((1 << bits) - 1)


def cached_coefficients(codes, scales, biases):
    # Explicit float32 ufunc boundaries; weights stay F32, never BF16.
    group_indices = np.arange(codes.shape[-1]) // 64
    multiplied = np.multiply(
        codes.astype(np.float32), scales[:, group_indices], dtype=np.float32
    )
    return np.add(multiplied, biases[:, group_indices], dtype=np.float32)


def source_lane_partials(x, unpacked_codes, scales, biases):
    # Source traversal: each row/output/lane visits k=lane+32*j chronologically.
    rows, k_size = x.shape
    n_size = unpacked_codes.shape[0]
    result = np.zeros((rows, n_size, 32), dtype=np.float32)
    for r in range(rows):
        for n in range(n_size):
            for j in range(k_size // 32):
                coefficient = np.add(
                    np.multiply(
                        unpacked_codes[n, j * 32 : (j + 1) * 32].astype(np.float32),
                        scales[n, j // 2],
                        dtype=np.float32,
                    ),
                    biases[n, j // 2],
                    dtype=np.float32,
                )
                term = np.multiply(
                    x[r, j * 32 : (j + 1) * 32], coefficient, dtype=np.float32
                )
                result[r, n] = np.add(result[r, n], term, dtype=np.float32)
    return result


def reused_lane_partials(x, original_f32_cache):
    rows, k_size = x.shape
    n_size = original_f32_cache.shape[0]
    result = np.zeros((rows, n_size, 32), dtype=np.float32)
    for n in range(n_size):
        for j in range(k_size // 32):
            coefficient = original_f32_cache[n, j * 32 : (j + 1) * 32]
            # Only loop nesting changes: one loaded coefficient feeds all real
            # rows. Each row/lane's mul and add sequence remains literal.
            term = np.multiply(
                x[:, j * 32 : (j + 1) * 32], coefficient[None, :], dtype=np.float32
            )
            result[:, n] = np.add(result[:, n], term, dtype=np.float32)
    return result


def bf16_rank(word):
    return 32768 - (word & 32767) if word & 32768 else 32768 + word


def adjacent_midpoint_compatible(raw_word, cached_word, double_sum, absolute_bound):
    if (
        not all(
            math.isfinite(v)
            for v in [number(raw_word), number(cached_word), double_sum, absolute_bound]
        )
        or absolute_bound < 0
    ):
        return False
    if number(raw_word) == number(cached_word):
        return True
    if abs(bf16_rank(raw_word) - bf16_rank(cached_word)) != 1:
        return False
    shared_midpoint = (number(raw_word) + number(cached_word)) / 2
    return abs(double_sum - shared_midpoint) <= absolute_bound


class LiteralCoefficientAndTraversalTests(unittest.TestCase):
    def test_all_formats_unpack_cross_byte_codes_and_group_edges(self):
        for bits in [4, 5, 6, 8]:
            codes = [(k * 19 + 3) % (1 << bits) for k in range(10240)]
            packed = pack_codes(codes, bits)
            self.assertEqual(len(packed), 10240 * bits // 8)
            self.assertEqual(
                [unpack_code(packed, bits, k) for k in range(10240)], codes
            )
            for edge in [31, 32, 63, 64, 127, 128, 10239]:
                self.assertEqual(unpack_code(packed, bits, edge), codes[edge])

    def test_cache_original_f32_bits_for_every_code_and_signed_scale(self):
        scale_words = [
            0,
            0x8000,
            0x0001,
            0x8001,
            0x0080,
            0x8080,
            bf16(0.0101318359375),
            bf16(-0.0101318359375),
            bf16(1.0078125),
        ]
        bias_words = [0, 0x8000, 0x0001, 0x8001, bf16(0.404296875), bf16(-0.5), bf16(1)]
        for bits in [4, 5, 6, 8]:
            for sw in scale_words:
                for bw in bias_words:
                    codes = np.arange(1 << bits, dtype=np.uint16).reshape(1, -1)
                    # Extend to whole G64 groups for each actual packed format.
                    codes = np.tile(codes, (1, max(1, 64 // codes.shape[1])))
                    scales = np.full(
                        (1, codes.shape[1] // 64), number(sw), dtype=np.float32
                    )
                    biases = np.full_like(scales, number(bw))
                    actual = (
                        cached_coefficients(codes, scales, biases)
                        .reshape(-1)
                        .view(np.uint32)
                    )
                    expected = [
                        bits32(f32(f32(int(q) * number(sw)) + number(bw)))
                        for q in codes.flat
                    ]
                    np.testing.assert_array_equal(
                        actual, np.asarray(expected, dtype=np.uint32)
                    )

    def test_cache_is_not_allowed_to_round_coefficients_to_bf16(self):
        coefficient = f32(f32(3 * number(0x3F81)) + 0)
        self.assertEqual(bits32(coefficient), 0x40418000)
        self.assertNotEqual(bits32(coefficient), bits32(number(bf16(coefficient))))

    def test_literal_row_reuse_preserves_every_lane_partial_bit(self):
        rows, n_size, k_size = 16, 2, 10240
        input_values = [
            number(bf16(v)) for v in [0, -0.0, 0.03125, -0.625, 1.25, 255, -16]
        ]
        x = np.asarray(
            [
                input_values[(r * 11 + k * 3) % len(input_values)]
                for r in range(rows)
                for k in range(k_size)
            ],
            dtype=np.float32,
        ).reshape(rows, k_size)
        scale_values = [
            number(bf16(v)) for v in [0.0101318359375, -0.00390625, 0.0002, 5.125, -9.5]
        ]
        bias_values = [
            number(bf16(v)) for v in [0, 1, -1, 0.3, -0.5, 1 / 256, -1 / 2048]
        ]
        scales = np.asarray(
            [
                scale_values[(n * 7 + g) % len(scale_values)]
                for n in range(n_size)
                for g in range(k_size // 64)
            ],
            dtype=np.float32,
        ).reshape(n_size, -1)
        biases = np.asarray(
            [
                bias_values[(n * 3 + g) % len(bias_values)]
                for n in range(n_size)
                for g in range(k_size // 64)
            ],
            dtype=np.float32,
        ).reshape(n_size, -1)
        for bits in [4, 5, 6, 8]:
            codes = np.asarray(
                [
                    ((n * 13 + k * 19) % (1 << bits))
                    for n in range(n_size)
                    for k in range(k_size)
                ],
                dtype=np.uint16,
            ).reshape(n_size, -1)
            original_cache = cached_coefficients(codes, scales, biases)
            for r in [4, 8, 16]:
                source = source_lane_partials(x[:r], codes, scales, biases)
                reused = reused_lane_partials(x[:r], original_cache)
                np.testing.assert_array_equal(
                    source.view(np.uint32), reused.view(np.uint32)
                )

    def test_changed_reduction_is_an_explicit_alternative(self):
        large = np.float32(1 << 24)
        literal = np.add(
            np.add(large, np.float32(1), dtype=np.float32), -large, dtype=np.float32
        )
        reordered = np.add(
            np.add(large, -large, dtype=np.float32), np.float32(1), dtype=np.float32
        )
        self.assertEqual(float(literal), 0)
        self.assertEqual(float(reordered), 1)
        self.assertNotEqual(bits32(literal), bits32(reordered))

    def test_literal_row_reuse_retains_overflow_and_cancellation_classification(self):
        x = np.zeros((4, 10240), dtype=np.float32)
        x[:, 0] = number(0x7F7F)
        x[:, 32] = number(0x7F7F)
        codes = np.zeros((1, 10240), dtype=np.uint16)
        codes[0, 32] = 1
        scales = np.full((1, 160), -4, dtype=np.float32)
        biases = np.full((1, 160), 2, dtype=np.float32)
        cache = cached_coefficients(codes, scales, biases)
        with np.errstate(over="ignore", invalid="ignore"):
            source = source_lane_partials(x, codes, scales, biases)
            reused = reused_lane_partials(x, cache)
        self.assertTrue(np.all(np.isnan(source[:, 0, 0])))
        np.testing.assert_array_equal(np.isnan(source), np.isnan(reused))
        np.testing.assert_array_equal(source.view(np.uint32), reused.view(np.uint32))


class BF16EpilogContractTests(unittest.TestCase):
    def test_down_divides_after_raw_bf16_before_sigmoid_and_product(self):
        for raw, expected in [(2.0, 0.3125), (-2.0, -0.1884765625), (0, 0)]:
            result = hc_down_epilog(raw)
            self.assertEqual(number(result[1]), raw / 4)
            self.assertEqual(number(result[3]), expected)
            self.assertEqual(result[4], 0)

    def test_raw_projection_boundary_matters_at_subnormal_ties(self):
        dot = 2.5 * 2.0**-133
        canonical_raw = bf16(dot)
        canonical_divided = bf16(number(canonical_raw) / 4)
        bypassed_raw = bf16(dot / 4)
        self.assertEqual(canonical_raw, 2)
        self.assertEqual(canonical_divided, 0)
        self.assertEqual(bypassed_raw, 1)

    def test_divided_bf16_boundary_matters_before_silu(self):
        raw = number(5)
        canonical = hc_down_epilog(raw)
        direct_divided = raw / 4
        bypassed_divided = bf16(direct_divided * 0.5)
        self.assertEqual(canonical[1], 1)
        self.assertEqual(canonical[3], 0)
        self.assertEqual(bypassed_divided, 1)

    def test_injection_uses_divided_raw_dot_times_two_sigmoid(self):
        for raw, gate in [(0, 1), (2, 1.25), (-2, 0.75390625)]:
            result = injection_epilog(raw)
            self.assertEqual(number(result[3]), gate)
            self.assertEqual(result[4], 0)

    def test_precise_injection_threshold_differs_from_fast_down_sigmoid(self):
        precise = injection_epilog(-27.375)
        substituted_fast = injection_epilog(-27.375, exponent_word=0x446A)
        self.assertEqual(number(precise[1]), -6.84375)
        self.assertEqual(precise[2], 0x3A8B)
        self.assertEqual(precise[3], 0x3B0B)
        self.assertEqual(substituted_fast[3], 0x3B0C)

    def test_sigmoid_exp_overflow_is_valid_but_nonfinite_raw_dot_is_diagnosed(self):
        for raw in [number(0x7F7F), number(0xFF7F)]:
            self.assertEqual(hc_down_epilog(raw)[4], 0)
            self.assertEqual(injection_epilog(raw)[4], 0)
        for bad in [math.inf, -math.inf, math.nan]:
            self.assertEqual(hc_down_epilog(bad)[4], 4)
            self.assertEqual(injection_epilog(bad)[4], 4)

    def test_finite_f32_dot_bf16_overflow_must_set_diagnostic(self):
        dot = f32(number(0x7F7F) + 2**119)
        self.assertTrue(math.isfinite(dot))
        down = hc_down_epilog(dot)
        injection = injection_epilog(dot)
        self.assertEqual(down[0], 0x7F80)
        self.assertEqual(injection[0], 0x7F80)
        self.assertEqual(down[4], 4)
        self.assertEqual(
            number(injection[3]), 2
        )  # Finite gate does not erase dot overflow.
        self.assertEqual(injection[4], 4)


class GeometryAndEvidenceTests(unittest.TestCase):
    def test_low_output_count_explains_whole_k_underoccupancy(self):
        self.assertEqual(320 * 10240 * 4, 13107200)
        for rows in [4, 8, 16]:
            padded = (rows + 7) // 8 * 8
            self.assertEqual((padded // 8) * (320 // 64), 5 if rows <= 8 else 10)
            self.assertEqual((padded // 8) * (320 // 32), 10 if rows <= 8 else 20)
            self.assertGreaterEqual((320 + 4 + 3) // 4, 80)  # Literal SG4 row reuse.
            self.assertEqual(padded * 10240 * 2, 163840 if rows <= 8 else 327680)

    def test_row_major_cache_offsets_do_not_use_transposed_k_n(self):
        n_size, k_size = 320, 10240
        for n, k in [(0, 0), (1, 31), (31, 63), (319, 10239)]:
            offset = (n * k_size + k) * 4
            row_start, channel = divmod(offset // 4, k_size)
            self.assertEqual((row_start, channel), (n, k))
            self.assertLess(offset + 3, n_size * k_size * 4)

    def test_midpoint_evidence_rejects_nonadjacent_and_far_from_boundary(self):
        a, b = 0x3F80, 0x3F81
        midpoint = (number(a) + number(b)) / 2
        self.assertTrue(adjacent_midpoint_compatible(a, b, midpoint, 2**-24))
        self.assertFalse(adjacent_midpoint_compatible(a, b, 1.0, 2**-24))
        self.assertFalse(adjacent_midpoint_compatible(a, b + 1, midpoint, 1))
        self.assertFalse(adjacent_midpoint_compatible(a, b, midpoint, -1))
        self.assertFalse(adjacent_midpoint_compatible(0x7F80, b, midpoint, 1))
        self.assertTrue(adjacent_midpoint_compatible(0x8000, 0, 0, 0))

    def test_canonical_shader_reduction_and_operator_distinctions(self):
        shader = (
            ROOT / "runtime/metal/kernels/shared/flash_hc_fused.metal"
        ).read_text()
        self.assertIn("const uint k = g * group_size + block * 32 + lane;", shader)
        self.assertIn("const float coefficient = code * sf + bias;", shader)
        self.assertIn("sum += value * coefficient;", shader)
        self.assertIn("const float sum = simd_sum(partial);", shader)
        self.assertIn("const bfloat raw = bfloat(sum);", shader)
        self.assertIn("bfloat(float(raw) / float(p.streams))", shader)
        self.assertIn("hc_fused_sigmoid_fast(divided)", shader)
        self.assertIn("hc_fused_sigmoid_unary(divided)", shader)
        self.assertIn("bfloat(2.0f * float(sigmoid))", shader)


@unittest.skipUnless(MANIFEST.exists(), "local immutable model metadata is unavailable")
class ActualSourceFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())

    def test_representative_actual_source_prefixes_cover_all_bits_and_mixed_injection(
        self,
    ):
        quantization = self.manifest["quantization"]
        for layer, down_bits, injection_bits in [
            (1, 4, 4),
            (0, 5, 5),
            (15, 6, 6),
            (31, 8, 8),
            (19, 4, 5),
        ]:
            prefix = f"language_model.model.layers.{layer}.attn_hyper_connection"
            for role, bits, n_size in [
                ("input_mix_weight_down", down_bits, 320),
                ("block_inject_weight", injection_bits, 4),
            ]:
                p = prefix + "." + role
                q = quantization.get(p, quantization)
                self.assertEqual(
                    (q["bits"], q["group_size"], q["mode"]), (bits, 64, "affine")
                )
                self.assertEqual(
                    self.manifest["tensors"][p + ".scales"]["shape"], [n_size, 160]
                )
                self.assertEqual(
                    self.manifest["tensors"][p + ".weight"]["shape"],
                    [n_size, 10240 * bits // 32],
                )

    def test_main_hc_format_inventory_and_final_mixer(self):
        quantization = self.manifest["quantization"]
        found = Counter()
        for name in self.manifest["tensors"]:
            if name.startswith("language_model.model.") and name.endswith(
                ".input_mix_weight_down.weight"
            ):
                prefix = name[:-7]
                q = quantization.get(prefix, quantization)
                found[(q["bits"], q["group_size"])] += 1
        self.assertEqual(
            found, Counter({(4, 64): 64, (5, 64): 15, (6, 64): 14, (8, 64): 4})
        )
        final = "language_model.model.hyper_connection_mixer"
        self.assertNotIn(
            final + ".block_inject_weight.weight", self.manifest["tensors"]
        )
        self.assertEqual(
            self.manifest["tensors"][final + ".input_mix_weight_down.scales"]["shape"],
            [320, 160],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

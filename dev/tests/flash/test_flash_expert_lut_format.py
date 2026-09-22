#!/usr/bin/env python3
"""Independent CPU audit of Q4/G64 expert coefficient LUT arithmetic/layout.

This creates no Metal backend and never loads a model. Scalar references use
IEEE binary32 packing and an independent exact-rational nearest-even encoder.
Production converter checks become available when dev/tools/flash_expert_lut.py
is present. Passing these tests is not a GPU correctness or speed claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import struct
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CONVERTER_PATH = ROOT / "dev/tools/flash_expert_lut.py"
if CONVERTER_PATH.exists():
    SPEC = importlib.util.spec_from_file_location(
        "expert_lut_under_test", CONVERTER_PATH
    )
    CONVERTER = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = CONVERTER
    SPEC.loader.exec_module(CONVERTER)
else:
    CONVERTER = None


def f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def bf16_value(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(word) << 16))[0]


def independent_bf16_round_bits(bits: int) -> int:
    """Division/remainder implementation independent of add-0x7fff trick."""
    high, discarded = divmod(bits, 1 << 16)
    if bits & 0x7F800000 == 0x7F800000:
        return high | (0x40 if bits & 0x7FFFFF else 0)
    if discarded > 0x8000 or (discarded == 0x8000 and high & 1):
        high += 1
    return high & 0xFFFF


def scalar_source_coefficient(scale_word: int, bias_word: int, q: int) -> int:
    """Literal source boundary: F32(F32(q*s)+b), then BF16 nearest-even."""
    scale, bias = bf16_value(scale_word), bf16_value(bias_word)
    if not math.isfinite(scale) or not math.isfinite(bias):
        raise ValueError("nonfinite stored parameter")
    reconstructed = f32(f32(float(q) * scale) + bias)
    rounded = independent_bf16_round_bits(f32_bits(reconstructed))
    if not math.isfinite(reconstructed) or not math.isfinite(bf16_value(rounded)):
        raise ValueError("nonfinite reconstructed coefficient")
    return rounded


def _nearest_even_integer(value: Fraction) -> int:
    integer, remainder = divmod(value.numerator, value.denominator)
    twice = 2 * remainder
    if twice > value.denominator or (twice == value.denominator and integer & 1):
        integer += 1
    return integer


def rational_ieee_bits(value: Fraction, fraction_bits: int) -> int:
    """Encode finite exact rational into exponent8 IEEE storage, nearest-even.

    A zero Fraction is positive zero. Signed-zero arithmetic is audited against
    explicit handwritten sign cases separately, because Fraction drops its sign.
    """
    sign = int(value < 0)
    value = abs(value)
    if not value:
        return 0
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    power = Fraction(2) ** exponent
    if value < power:
        exponent -= 1
    sign_word = sign << (fraction_bits + 8)
    if exponent < -126:
        significant = _nearest_even_integer(
            value / (Fraction(2) ** (-126 - fraction_bits))
        )
        # Rounding the largest subnormal upward naturally emits minimum normal.
        return sign_word | significant
    significant = _nearest_even_integer(
        value / (Fraction(2) ** (exponent - fraction_bits))
    )
    if significant == 1 << (fraction_bits + 1):
        exponent += 1
        significant >>= 1
    if exponent > 127:
        return sign_word | (0xFF << fraction_bits)
    return (
        sign_word
        | ((exponent + 127) << fraction_bits)
        | (significant - (1 << fraction_bits))
    )


def rational_source_coefficient(scale_word: int, bias_word: int, q: int) -> int:
    """Independent exact arithmetic followed by each stated IEEE boundary."""
    scale, bias = bf16_value(scale_word), bf16_value(bias_word)
    product_bits = rational_ieee_bits(Fraction(scale) * q, 23)
    product = struct.unpack("<f", struct.pack("<I", product_bits))[0]
    if not math.isfinite(product):
        raise ValueError("FP32 product overflow")
    sum_bits = rational_ieee_bits(Fraction(product) + Fraction(bias), 23)
    summed = struct.unpack("<f", struct.pack("<I", sum_bits))[0]
    if not math.isfinite(summed):
        raise ValueError("FP32 sum overflow")
    result = rational_ieee_bits(Fraction(summed), 7)
    if not math.isfinite(bf16_value(result)):
        raise ValueError("BF16 coefficient overflow")
    return result


def independent_offset(e: int, n: int, k: int, q: int, n_size: int, k_size: int) -> int:
    # Tiled saved layout [E,Nblock64,Kgroup64,Nlane64,Q16], uint16 LE.
    block, n_lane = divmod(n, 64)
    group = k // 64
    return (
        ((e * (n_size // 64) + block) * (k_size // 64) + group) * 64 + n_lane
    ) * 32 + q * 2


class IndependentArithmeticTests(unittest.TestCase):
    def test_all_finite_bf16_values_round_trip_bit_exactly(self):
        for word in range(65536):
            if math.isfinite(bf16_value(word)):
                self.assertEqual(independent_bf16_round_bits(word << 16), word)

    def test_every_bf16_tie_rounds_to_even(self):
        for high in range(65536):
            if high & 0x7F80 == 0x7F80:
                continue
            tie = (high << 16) | 0x8000
            self.assertEqual(
                independent_bf16_round_bits(tie), (high + (high & 1)) & 0xFFFF
            )
            self.assertEqual(independent_bf16_round_bits(tie - 1), high)
            self.assertEqual(independent_bf16_round_bits(tie + 1), (high + 1) & 0xFFFF)

    def test_rational_reference_matches_ieee_for_edges_and_random_inputs(self):
        rng = random.Random(0xBF164064)
        words = [
            0,
            1,
            0x7F,
            0x80,
            0x81,
            0x3F80,
            0x3F81,
            0x7F7F,
            0x8001,
            0x807F,
            0x8080,
            0x8081,
            0xBF80,
            0xBF81,
            0xFF7F,
        ]
        words += [rng.randrange(65536) for _ in range(256)]
        checked = 0
        for scale in words:
            if not math.isfinite(bf16_value(scale)):
                continue
            for bias in [0, 0x0001, 0x8001, 0x0080, 0x8080, 0x3F80, 0xBF80]:
                for q in [0, 1, 3, 7, 15]:
                    try:
                        expected = scalar_source_coefficient(scale, bias, q)
                    except ValueError:
                        with self.assertRaises(ValueError):
                            rational_source_coefficient(scale, bias, q)
                        continue
                    actual = rational_source_coefficient(scale, bias, q)
                    # Fraction loses -0, so zero equality is tested explicitly below.
                    self.assertEqual(
                        actual & 0x7FFF if not actual & 0x7FFF else actual,
                        expected & 0x7FFF if not expected & 0x7FFF else expected,
                    )
                    checked += 1
        self.assertGreater(checked, 9000)

    def test_negative_scales_match_handwritten_golden(self):
        golden = [
            0x3F00,
            0x3E80,
            0x0000,
            0xBE80,
            0xBF00,
            0xBF40,
            0xBF80,
            0xBFA0,
            0xBFC0,
            0xBFE0,
            0xC000,
            0xC010,
            0xC020,
            0xC030,
            0xC040,
            0xC050,
        ]
        self.assertEqual(
            [scalar_source_coefficient(0xBE80, 0x3F00, q) for q in range(16)], golden
        )

    def test_signed_zero_rules_match_separate_multiply_add(self):
        # Under round-to-nearest, equal opposing nonzero terms cancel to +0.
        cases = [
            (0x0000, 0x8000, 1, 0x0000),
            (0x8000, 0x0000, 1, 0x0000),
            (0x8000, 0x8000, 1, 0x8000),
            (0xBF80, 0x8000, 0, 0x8000),
            (0xBF80, 0x0000, 0, 0x0000),
            (0xBF80, 0x3F80, 1, 0x0000),
        ]
        for scale, bias, q, expected in cases:
            self.assertEqual(scalar_source_coefficient(scale, bias, q), expected)

    def test_subnormal_coefficients_do_not_flush_to_zero(self):
        for scale in [0x0001, 0x007F, 0x0080, 0x8001, 0x807F, 0x8080]:
            for q in range(16):
                self.assertEqual(
                    scalar_source_coefficient(scale, 0, q),
                    rational_source_coefficient(scale, 0, q),
                )
        self.assertEqual(scalar_source_coefficient(0x0001, 0, 1), 0x0001)

    def test_real_expression_direct_bf16_rounding_is_not_the_contract(self):
        scale, bias, q = 0x3F81, 0xB080, 3
        exact = Fraction(bf16_value(scale)) * q + Fraction(bf16_value(bias))
        self.assertEqual(rational_ieee_bits(exact, 7), 0x4041)
        self.assertEqual(scalar_source_coefficient(scale, bias, q), 0x4042)

    def test_nan_inf_fp32_overflow_and_bf16_overflow_are_rejected(self):
        for bad in [0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81]:
            for q in [0, 15]:
                with self.assertRaises(ValueError):
                    scalar_source_coefficient(bad, 0, q)
                with self.assertRaises(ValueError):
                    scalar_source_coefficient(0, bad, q)
        with self.assertRaises(ValueError):
            scalar_source_coefficient(0x7F7F, 0, 15)  # FP32 multiply overflow.
        with self.assertRaises(ValueError):
            scalar_source_coefficient(0x7F7F, 0x7B00, 1)  # Finite FP32, BF16 overflow.


@unittest.skipUnless(CONVERTER is not None, "converter has not been created yet")
class ConverterArithmeticTests(unittest.TestCase):
    def test_full_finite_scale_domain_all_q4_codes_zero_bias(self):
        scales = np.array(
            [
                w
                for w in range(65536)
                if math.isfinite(bf16_value(w)) and abs(bf16_value(w)) <= 2.0**123
            ],
            dtype=np.uint16,
        )
        biases = np.zeros_like(scales)
        actual = CONVERTER.bf16_lut(scales, biases)
        self.assertEqual(actual.dtype, np.dtype(np.uint16))
        self.assertEqual(actual.shape, (len(scales), 16))
        for q in range(16):
            expected = np.array(
                [scalar_source_coefficient(s, 0, q) for s in scales], dtype=np.uint16
            )
            np.testing.assert_array_equal(actual[:, q], expected)

    def test_random_mixed_sign_biases_and_rounding_boundaries(self):
        rng = random.Random(0x164064)
        scales, biases = [], []
        while len(scales) < 4096:
            s, b = rng.randrange(65536), rng.randrange(65536)
            try:
                [scalar_source_coefficient(s, b, q) for q in range(16)]
            except ValueError:
                continue
            scales.append(s)
            biases.append(b)
        # Add cancellation, signed zeros, subnormals, and explicit halfway trap.
        scales += [0xBF80, 0x8000, 0x0000, 0x0001, 0x8001, 0x3F81]
        biases += [0x3F80, 0x8000, 0x8000, 0x8001, 0x0000, 0xB080]
        s = np.asarray(scales, dtype=np.uint16).reshape(2, -1)
        b = np.asarray(biases, dtype=np.uint16).reshape(2, -1)
        saved_s, saved_b = s.copy(), b.copy()
        expected = np.array(
            [
                [scalar_source_coefficient(s0, b0, q) for q in range(16)]
                for s0, b0 in zip(scales, biases)
            ],
            dtype=np.uint16,
        ).reshape(*s.shape, 16)
        np.testing.assert_array_equal(CONVERTER.bf16_lut(s, b), expected)
        np.testing.assert_array_equal(s, saved_s)
        np.testing.assert_array_equal(b, saved_b)

    def test_noncontiguous_parameter_views(self):
        scales = np.asarray(
            [[0x3F81, 0xBE80, 0x0080], [0x8001, 0xBF80, 0]], dtype=np.uint16
        ).T
        biases = np.asarray(
            [[0xB080, 0x3F00, 0], [0, 0x3F80, 0x8000]], dtype=np.uint16
        ).T
        self.assertFalse(scales.flags.c_contiguous)
        expected = np.array(
            [
                scalar_source_coefficient(s, b, q)
                for s, b in zip(scales.flat, biases.flat)
                for q in range(16)
            ],
            dtype=np.uint16,
        )
        np.testing.assert_array_equal(
            CONVERTER.bf16_lut(scales, biases).reshape(-1), expected
        )

    def test_converter_rejects_invalid_arithmetic(self):
        for scale, bias in [
            (0x7F80, 0),
            (0, 0x7FC0),
            (0xFF80, 0),
            (0x7F7F, 0),
            (0x7F7F, 0x7B00),
        ]:
            with self.assertRaises((ValueError, FloatingPointError)):
                CONVERTER.bf16_lut(
                    np.asarray([scale], dtype=np.uint16),
                    np.asarray([bias], dtype=np.uint16),
                )

    def test_converter_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            CONVERTER.bf16_lut(
                np.zeros((2, 3), dtype=np.uint16), np.zeros((1, 3), dtype=np.uint16)
            )

    def test_converter_rejects_empty_or_wrong_dtype_parameters(self):
        for dtype in [np.float32, np.int16, np.uint32, np.int64]:
            with self.assertRaises(ValueError):
                CONVERTER.bf16_lut(np.zeros(2, dtype=dtype), np.zeros(2, dtype=dtype))
        with self.assertRaises(ValueError):
            CONVERTER.bf16_lut(
                np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.uint16)
            )


class LayoutReferenceTests(unittest.TestCase):
    def test_saved_size_and_contiguous_n64_tile(self):
        for n, k in [(640, 2560), (2560, 640)]:
            self.assertEqual(512 * n * (k // 64) * 16 * 2, 419430400)
            offsets = {
                independent_offset(1, row, 64 * 2, q, n, k)
                for row in range(64, 128)
                for q in range(16)
            }
            start = min(offsets)
            self.assertEqual(offsets, set(range(start, start + 2048, 2)))
            self.assertEqual(start % 2048, 0)
        self.assertEqual(48 * 3 * 419430400, 60397977600)

    def test_little_endian_q4_words_preserve_original_codes(self):
        for word in [0x76543210, 0xFEDCBA98, 0, 0xFFFFFFFF, 0x102F384D]:
            storage = struct.pack("<I", word)
            for k in range(8):
                nibble = (storage[k // 2] >> (4 * (k & 1))) & 15
                self.assertEqual(nibble, (word >> (4 * k)) & 15)


@unittest.skipUnless(CONVERTER is not None, "converter has not been created yet")
class ConverterLayoutTests(unittest.TestCase):
    def test_tiled_storage_maps_every_element_to_independent_offset(self):
        e_size, n_size, k_size = 2, 128, 192
        logical = np.arange(
            e_size * n_size * (k_size // 64) * 16, dtype=np.uint16
        ).reshape(e_size, n_size, k_size // 64, 16)
        tiled = CONVERTER.tiled_lut(logical)
        self.assertEqual(tiled.shape, (e_size, n_size // 64, k_size // 64, 64, 16))
        self.assertTrue(tiled.flags.c_contiguous)
        stored = tiled.tobytes(order="C")
        for e in range(e_size):
            for n in range(n_size):
                for group in range(k_size // 64):
                    for q in range(16):
                        offset = independent_offset(e, n, group * 64, q, n_size, k_size)
                        self.assertEqual(
                            CONVERTER.lut_offset_bytes(
                                e, n, group * 64, q, n_size, k_size
                            ),
                            offset,
                        )
                        self.assertEqual(
                            struct.unpack_from("<H", stored, offset)[0],
                            int(logical[e, n, group, q]),
                        )

    def test_all_k_in_a_group_map_to_same_q_coefficient(self):
        for n_size, k_size in [(640, 2560), (2560, 640)]:
            for e, n, g, q in [
                (0, 0, 0, 0),
                (511, n_size - 1, k_size // 64 - 1, 15),
                (17, 63, 2, 7),
            ]:
                expected = independent_offset(e, n, g * 64, q, n_size, k_size)
                for k in range(g * 64, (g + 1) * 64):
                    self.assertEqual(
                        CONVERTER.lut_offset_bytes(e, n, k, q, n_size, k_size), expected
                    )

    def test_tiling_rejects_malformed_shape(self):
        for shape in [(2, 65, 3, 16), (2, 128, 3, 15), (128, 3, 16), (0, 128, 3, 16)]:
            with self.assertRaises(ValueError):
                CONVERTER.tiled_lut(np.zeros(shape, dtype=np.uint16))

    def test_offsets_reject_invalid_extent_indices_and_boolean_metadata(self):
        good = [0, 0, 0, 0, 640, 2560]
        for index, values in [
            (0, [-1, 512, True, 0.0]),
            (1, [-1, 640, False, 0.0]),
            (2, [-1, 2560, True, 0.0]),
            (3, [-1, 16, True, 0.0]),
            (4, [0, 65, True, 640.0]),
            (5, [0, 63, False, 2560.0]),
        ]:
            for value in values:
                arguments = good.copy()
                arguments[index] = value
                with self.assertRaises(ValueError):
                    CONVERTER.lut_offset_bytes(*arguments)


@unittest.skipUnless(CONVERTER is not None, "converter has not been created yet")
class ConverterMetadataTests(unittest.TestCase):
    def test_source_path_alignment_extent_and_types_are_strict(self):
        with tempfile.TemporaryDirectory(prefix="flash-lut-cpu-path-") as directory:
            top = Path(directory).resolve()
            package = top / "package"
            package.mkdir()
            (package / "weights.bin").write_bytes(bytes(16384))
            (top / "outside.bin").write_bytes(bytes(16384))
            good = {"shard": "weights.bin", "offset": 0, "length": 16}
            self.assertEqual(
                CONVERTER._safe_source(package, good), package / "weights.bin"
            )
            malformed = [
                {**good, "shard": "../outside.bin"},
                {**good, "shard": None},
                {**good, "offset": -16384},
                {**good, "offset": 1},
                {**good, "offset": False},
                {**good, "offset": 0.0},
                {**good, "length": True},
                {**good, "length": 0},
                {**good, "length": -1},
                {**good, "length": 16385},
                {**good, "offset": 16384, "length": 16},
            ]
            for metadata in malformed:
                with self.assertRaises(ValueError):
                    CONVERTER._safe_source(package, metadata)
            (package / "outside-link.bin").symlink_to(top / "outside.bin")
            with self.assertRaises(ValueError):
                CONVERTER._safe_source(package, {**good, "shard": "outside-link.bin"})

    def test_manifest_failure_leaves_source_and_output_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="flash-lut-cpu-manifest-") as directory:
            top = Path(directory).resolve()
            package = top / "package"
            package.mkdir()
            manifest = {
                "schema": "splash-local-qwen4-affine-v1",
                "alignment": 16384,
                "source_identity_sha256": "a" * 64,
                "tensors": {},
            }
            raw = json.dumps(manifest).encode()
            (package / "manifest.json").write_bytes(raw)
            (package / "manifest.sha256").write_text(
                hashlib.sha256(raw).hexdigest() + "\n"
            )
            source_before = {p.name: p.read_bytes() for p in package.iterdir()}
            output = top / "lut"
            # Missing tensor fails after private staging starts; staging must be removed.
            with self.assertRaises(ValueError):
                CONVERTER.convert_layer(package, output, 0)
            self.assertFalse(output.exists())
            self.assertFalse(list(top.glob(".lut.*")))
            self.assertEqual(
                {p.name: p.read_bytes() for p in package.iterdir()}, source_before
            )
            (package / "manifest.sha256").write_text("b" * 64 + "\n")
            with self.assertRaises(ValueError):
                CONVERTER.convert_layer(package, output, 0)
            self.assertFalse(output.exists())
            self.assertFalse(list(top.glob(".lut.*")))

    def test_existing_output_and_source_nested_output_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="flash-lut-cpu-output-") as directory:
            top = Path(directory).resolve()
            package = top / "package"
            package.mkdir()
            output = top / "existing"
            output.mkdir()
            (output / "preserve").write_bytes(b"untouched")
            with self.assertRaises(ValueError):
                CONVERTER.convert_layer(package, output, 0)
            self.assertEqual((output / "preserve").read_bytes(), b"untouched")
            with self.assertRaises(ValueError):
                CONVERTER.convert_layer(package, package / "nested-output", 0)
            self.assertFalse((package / "nested-output").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

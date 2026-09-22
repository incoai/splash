"""Independent CPU semantic checks; no Metal/MLX backend or model load."""

from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "dev/tests/flash"))
from test_flash_expert_lut_format import rational_source_coefficient  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "vocab_dictionary_under_test", ROOT / "dev/tools/flash_vocab_bf16_dictionary.py"
)
FORMAT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FORMAT)


class DictionarySemanticTests(unittest.TestCase):
    def test_q8_dictionary_matches_exact_rational_arithmetic(self):
        # Actual source-like signed normal ranges, asymmetric biases, and all
        # 256 codes. Reference applies both independent IEEE F32 boundaries.
        rng = random.Random(0xBF168064)
        pairs = [(0xB8A1, 0x3C22), (0x38A1, 0xBC22), (0x3B81, 0xBF01)]
        pairs += [
            (
                rng.randrange(0x3780, 0x3B00) | (rng.randrange(2) << 15),
                rng.randrange(0x3A00, 0x3E80) | (rng.randrange(2) << 15),
            )
            for _ in range(48)
        ]
        actual = FORMAT.make_dictionary(
            np.array([s | (b << 16) for s, b in pairs], dtype=np.uint32)
        )
        for row, (scale, bias) in enumerate(pairs):
            for code in range(256):
                expected = rational_source_coefficient(scale, bias, code)
                # Fraction loses zero sign; handwritten signed-zero test below.
                self.assertEqual(
                    int(actual[row, code]) & 0x7FFF
                    if expected & 0x7FFF == 0
                    else int(actual[row, code]),
                    expected,
                )

    def test_all_finite_bf16_values_round_trip_and_signed_zero(self):
        words = np.arange(65536, dtype=np.uint16)
        words = words[(words & 0x7F80) != 0x7F80]
        numeric = FORMAT.bf16_float(words)
        np.testing.assert_array_equal(FORMAT.bf16_words(numeric), words)
        np.testing.assert_array_equal(
            FORMAT.round_f64_to_bf16(numeric.astype(np.float64)), words
        )

    def test_halfway_boundaries_independent_even_reference(self):
        words = np.arange(0x0080, 0x7F00, dtype=np.uint16)
        low = FORMAT.bf16_float(words).astype(np.float64)
        high = FORMAT.bf16_float(words + 1).astype(np.float64)
        midpoint = (low + high) / 2
        expected = words + (words & 1)
        np.testing.assert_array_equal(FORMAT.round_f64_to_bf16(midpoint), expected)
        np.testing.assert_array_equal(
            FORMAT.round_f64_to_bf16(-midpoint), expected | 0x8000
        )
        np.testing.assert_array_equal(
            FORMAT.round_f64_to_bf16(np.nextafter(midpoint, -np.inf)), words
        )
        np.testing.assert_array_equal(
            FORMAT.round_f64_to_bf16(np.nextafter(midpoint, np.inf)), words + 1
        )

    def test_dictionary_product_rounding_traps_and_nonfinite_rejection(self):
        traps = FORMAT.cancellation_proofs()
        self.assertTrue(any(row["distinguishes_product_first"] for row in traps))
        for scale, bias in ((0x7F80, 0), (0, 0x7FC0), (0xFF80, 0), (0x7F7F, 0)):
            with self.assertRaises(ValueError):
                FORMAT.make_dictionary(
                    np.array([scale | (bias << 16)], dtype=np.uint32)
                )

    def test_u16_dictionary_indices_preserve_signed_scale_bias_pairs(self):
        pairs = np.array(
            [
                [0x3C22B8A1, 0xBC2238A1, 0x3C22B8A1],
                [0xBF013B81, 0x3C22B8A1, 0xBC2238A1],
            ],
            dtype=np.uint32,
        )
        unique, inverse = np.unique(pairs, return_inverse=True)
        indices = inverse.reshape(pairs.shape).astype(np.uint16)
        np.testing.assert_array_equal(unique[indices], pairs)
        dictionary = FORMAT.make_dictionary(unique)
        for row in range(2):
            for group in range(3):
                scale, bias = (
                    int(pairs[row, group] & 0xFFFF),
                    int(pairs[row, group] >> 16),
                )
                for code in (0, 25, 127, 128, 255):
                    self.assertEqual(
                        int(dictionary[indices[row, group], code]),
                        rational_source_coefficient(scale, bias, code),
                    )


if __name__ == "__main__":
    unittest.main()

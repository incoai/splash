#!/usr/bin/env python3
"""CPU contract audit for packed HC-down coefficient reuse across 2/4 rows.

The check is before simd_sum: bit-identical lane partials imply identical inputs
to the original hardware reduction.  This does not qualify Metal compiler IR,
fast exponentials, GPU timings, or the service lifecycle.
"""

from __future__ import annotations

import math
import struct
import unittest
from collections import Counter
from dataclasses import dataclass

import numpy as np


def number(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


@dataclass(frozen=True)
class PackedMatrix:
    bits: int
    group: int
    k_size: int
    n_size: int
    weight_stride: int
    parameter_stride: int
    weights: bytes
    scales: bytes
    biases: bytes


def fixture(
    bits: int, group: int, *, k_size: int = 10240, n_size: int = 3, padding: int = 0
) -> PackedMatrix:
    weight_stride = k_size * bits // 8 + padding
    parameter_stride = 2 * (k_size // group) + 2 * padding
    weights = bytearray([0xA5] * (weight_stride * n_size))
    scales = bytearray([0xA5] * (parameter_stride * n_size))
    biases = bytearray([0xA5] * (parameter_stride * n_size))
    scale_words = [0x3C26, 0xBB80, 0x0001, 0x8001, 0, 0x8000, 0x40A4]
    bias_words = [0, 0x8000, 0x3F80, 0xBF80, 0x3E9A, 0x0001, 0x8001]
    for n in range(n_size):
        # Independent bit-by-bit encoder; never reads the decoder under test.
        weights[n * weight_stride : n * weight_stride + k_size * bits // 8] = bytes(
            k_size * bits // 8
        )
        for k in range(k_size):
            code = (n * 13 + k * 19 + 3) % (1 << bits)
            for qbit in range(bits):
                linear = k * bits + qbit
                weights[n * weight_stride + linear // 8] |= ((code >> qbit) & 1) << (
                    linear % 8
                )
        for g in range(k_size // group):
            struct.pack_into(
                "<H",
                scales,
                n * parameter_stride + 2 * g,
                scale_words[(n * 7 + g) % len(scale_words)],
            )
            struct.pack_into(
                "<H",
                biases,
                n * parameter_stride + 2 * g,
                bias_words[(n * 3 + g) % len(bias_words)],
            )
    return PackedMatrix(
        bits,
        group,
        k_size,
        n_size,
        weight_stride,
        parameter_stride,
        bytes(weights),
        bytes(scales),
        bytes(biases),
    )


def unpack(m: PackedMatrix, n: int, k: int) -> int:
    byte, shift = divmod(k * m.bits, 8)
    byte += n * m.weight_stride
    code = m.weights[byte]
    if shift + m.bits > 8:
        code |= m.weights[byte + 1] << 8
    return (code >> shift) & ((1 << m.bits) - 1)


def parameter_words(m: PackedMatrix, data: bytes, n: int) -> np.ndarray:
    start = n * m.parameter_stride
    return np.frombuffer(data, dtype="<u2", count=m.k_size // m.group, offset=start)


def coefficient_block(m: PackedMatrix, n: int, g: int, block: int) -> np.ndarray:
    sf = np.float32(number(int(parameter_words(m, m.scales, n)[g])))
    bf = np.float32(number(int(parameter_words(m, m.biases, n)[g])))
    first = g * m.group + block * 32
    codes = np.asarray(
        [unpack(m, n, first + lane) for lane in range(32)], dtype=np.float32
    )
    # Explicit two operations match #pragma clang fp contract(off).
    return np.add(np.multiply(codes, sf, dtype=np.float32), bf, dtype=np.float32)


def source_partials(x: np.ndarray, m: PackedMatrix) -> np.ndarray:
    out = np.zeros((len(x), m.n_size, 32), dtype=np.float32)
    for r in range(len(x)):
        for n in range(m.n_size):
            for g in range(m.k_size // m.group):
                for block in range(m.group // 32):
                    first = g * m.group + block * 32
                    coefficient = coefficient_block(m, n, g, block)
                    product = np.multiply(
                        x[r, first : first + 32], coefficient, dtype=np.float32
                    )
                    out[r, n] = np.add(out[r, n], product, dtype=np.float32)
    return out


def reused_partials(x: np.ndarray, m: PackedMatrix, rows_per_simd: int) -> np.ndarray:
    out = np.full((len(x), m.n_size, 32), np.nan, dtype=np.float32)
    for base_row in range(0, len(x), rows_per_simd):
        real_rows = min(rows_per_simd, len(x) - base_row)
        for n in range(m.n_size):
            accumulators = np.zeros((rows_per_simd, 32), dtype=np.float32)
            for g in range(m.k_size // m.group):
                for block in range(m.group // 32):
                    first = g * m.group + block * 32
                    coefficient = coefficient_block(m, n, g, block)
                    for local_row in range(real_rows):
                        product = np.multiply(
                            x[base_row + local_row, first : first + 32],
                            coefficient,
                            dtype=np.float32,
                        )
                        accumulators[local_row] = np.add(
                            accumulators[local_row], product, dtype=np.float32
                        )
            out[base_row : base_row + real_rows, n] = accumulators[:real_rows]
    return out


def inputs(rows: int, k_size: int) -> np.ndarray:
    words = [0, 0x8000, 0x3D00, 0xBF20, 0x3FA0, 0x437F, 0xC180, 1, 0x8001]
    flat = [
        number(words[(r * 11 + k * 3) % len(words)])
        for r in range(rows)
        for k in range(k_size)
    ]
    return np.asarray(flat, dtype=np.float32).reshape(rows, k_size)


def grid_writes(rows: int, rows_per_simd: int, injection: bool, simdgroups: int = 4):
    n_size = 320 + (4 if injection else 0)
    writes = Counter()
    for grid_y in range((rows + rows_per_simd - 1) // rows_per_simd):
        for grid_x in range((n_size + simdgroups - 1) // simdgroups):
            for simd in range(simdgroups):
                n = grid_x * simdgroups + simd
                if n >= n_size:
                    continue
                for local_row in range(rows_per_simd):
                    row = grid_y * rows_per_simd + local_row
                    if row < rows:
                        writes[(row, n)] += 1
    return writes


class PackedAddressTests(unittest.TestCase):
    def test_all_codes_groups_padded_strides_and_last_byte(self):
        for bits in [4, 5, 6, 8]:
            for group in [32, 64, 128]:
                for padding in [0, 19]:
                    m = fixture(bits, group, padding=padding)
                    for n in range(m.n_size):
                        for k in range(m.k_size):
                            self.assertEqual(
                                unpack(m, n, k), (n * 13 + k * 19 + 3) % (1 << bits)
                            )
                        final_byte, shift = divmod((m.k_size - 1) * bits, 8)
                        read_size = 2 if shift + bits > 8 else 1
                        self.assertEqual(final_byte + read_size, m.k_size * bits // 8)
                    for g in range(m.k_size // group):
                        visited = [
                            g * group + block * 32 + lane
                            for block in range(group // 32)
                            for lane in range(32)
                        ]
                        self.assertEqual(
                            visited, list(range(g * group, (g + 1) * group))
                        )

    def test_grid_every_real_output_once_without_tail_writes(self):
        for rows in range(1, 33):
            for row_reuse in [2, 4]:
                for injection in [False, True]:
                    writes = grid_writes(rows, row_reuse, injection)
                    n_size = 324 if injection else 320
                    self.assertEqual(len(writes), rows * n_size)
                    self.assertTrue(all(count == 1 for count in writes.values()))
                    self.assertTrue(
                        all(0 <= row < rows and 0 <= n < n_size for row, n in writes)
                    )
        for rows in [4, 8, 16]:
            for row_reuse in [2, 4]:
                self.assertEqual(
                    (324 + 3) // 4 * ((rows + row_reuse - 1) // row_reuse),
                    81 * (rows // row_reuse),
                )


class LiteralArithmeticTests(unittest.TestCase):
    def assert_same_words(self, a: np.ndarray, b: np.ndarray):
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))

    def test_all_formats_groups_literal_full_10240_lane_partials(self):
        x = inputs(7, 10240)
        for bits in [4, 5, 6, 8]:
            for group in [32, 64, 128]:
                m = fixture(bits, group, n_size=2, padding=19)
                reference = source_partials(x, m)
                for row_reuse in [2, 4]:
                    self.assert_same_words(reference, reused_partials(x, m, row_reuse))

    def test_every_row_tail_up_to_32_independent_accumulators(self):
        x = inputs(32, 128)
        m = fixture(5, 32, k_size=128, n_size=2, padding=7)
        reference = source_partials(x, m)
        for rows in range(1, 33):
            for row_reuse in [2, 4]:
                self.assert_same_words(
                    reference[:rows], reused_partials(x[:rows], m, row_reuse)
                )

    def test_mixed_injection_uses_its_own_bits_group_and_strides(self):
        x = inputs(7, 128)
        for down_bits in [4, 5, 6, 8]:
            for down_group in [32, 64, 128]:
                down = fixture(down_bits, down_group, k_size=128, n_size=2, padding=3)
                down_reference = source_partials(x, down)
                for injection_bits in [4, 5, 6, 8]:
                    for injection_group in [32, 64, 128]:
                        injection = fixture(
                            injection_bits,
                            injection_group,
                            k_size=128,
                            n_size=4,
                            padding=11,
                        )
                        reference = source_partials(x, injection)
                        for row_reuse in [2, 4]:
                            self.assert_same_words(
                                down_reference, reused_partials(x, down, row_reuse)
                            )
                            self.assert_same_words(
                                reference, reused_partials(x, injection, row_reuse)
                            )

    def test_signed_zero_subnormal_infinity_and_nan_classification(self):
        x = inputs(4, 128)
        x[0, 0] = np.float32(math.inf)
        x[0, 32] = np.float32(-math.inf)
        x[1, 63] = np.float32(math.nan)
        x[2, 0] = np.float32(number(0x7F7F))
        x[2, 32] = np.float32(number(0x7F7F))
        for bits in [4, 5, 6, 8]:
            for group in [32, 64, 128]:
                m = fixture(bits, group, k_size=128, n_size=2)
                with np.errstate(over="ignore", invalid="ignore"):
                    reference = source_partials(x, m)
                    for row_reuse in [2, 4]:
                        candidate = reused_partials(x, m, row_reuse)
                        self.assert_same_words(reference, candidate)
                        np.testing.assert_array_equal(
                            np.isfinite(reference), np.isfinite(candidate)
                        )

    def test_accumulator_fusion_changes_valid_bfloat_input(self):
        # A BF16 activation and F32 affine coefficient can have an inexact
        # product: contraction preserves bits the canonical product rounds away.
        value = np.float32(number(0x3F7F))
        coefficient = np.add(
            np.multiply(np.float32(255), np.float32(number(0x3F81)), dtype=np.float32),
            np.float32(number(0x3E9A)),
            dtype=np.float32,
        )
        self.assertEqual(int(coefficient.view(np.uint32)), 0x4380A580)
        accumulator = -np.multiply(value, coefficient, dtype=np.float32)
        unfused = np.add(
            accumulator,
            np.multiply(value, coefficient, dtype=np.float32),
            dtype=np.float32,
        )
        fma_result = np.float32(float(value) * float(coefficient) + float(accumulator))
        self.assertEqual(int(unfused.view(np.uint32)), 0)
        self.assertEqual(int(fma_result.view(np.uint32)), 0x37800000)

    def test_serial_accumulation_and_coefficient_rounding_are_material(self):
        big = np.float32(1 << 24)
        serial = np.add(
            np.add(big, np.float32(1), dtype=np.float32), -big, dtype=np.float32
        )
        reordered = np.add(
            np.add(big, -big, dtype=np.float32), np.float32(1), dtype=np.float32
        )
        self.assertNotEqual(serial.view(np.uint32), reordered.view(np.uint32))
        coefficient = np.float32(3 * number(0x3F81))
        self.assertEqual(int(coefficient.view(np.uint32)), 0x40418000)
        self.assertNotEqual(int(coefficient.view(np.uint32)) & 0xFFFF, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

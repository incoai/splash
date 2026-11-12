"""MiniMax-H3's LoRA merge (install/h3.py): OpenVDN's merge_lora_state, which
folds each adapter's delta into the bf16 model in place."""

import unittest

import numpy as np

from install import h3


def bf16(x):
    return h3.to_f32("BF16", h3.bf16_bits(np.asarray(x, np.float32)))


class MergeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.w = bf16(rng.standard_normal((64, 48)).astype(np.float32))
        # A delta like the turbo adapter's: about 2e-4 of |W|, mostly below
        # half a bf16 ulp, and one like the default adapter's, ~4e-3 of |W|.
        self.small = (
            (rng.standard_normal((64, 4)) * 1e-3).astype(np.float32),
            (rng.standard_normal((4, 48)) * 0.1).astype(np.float32),
        )
        self.large = (
            (rng.standard_normal((64, 4)) * 2e-2).astype(np.float32),
            (rng.standard_normal((4, 48)) * 0.1).astype(np.float32),
        )

    def official(self, deltas):
        """merge_lora_state: w.add_((B.float() @ A.float() * scale).to(bf16))
        on a bf16 parameter, delta by delta."""
        w = self.w.copy()
        for b, a, sc in deltas:
            w = bf16(w + bf16(np.float32(sc) * (b @ a)))
        return w

    def test_rows_merge_as_openvdn_does_delta_by_delta(self):
        deltas = [(a, b, 1.0) for b, a in (self.large, self.small)]
        rows = h3.vdn_merged_rows(lambda r0, r1: self.w[r0:r1], deltas)
        want = self.official([(b, a, 1.0) for b, a in (self.large, self.small)])
        np.testing.assert_array_equal(rows(0, 64), want)
        # by rows, as the units convert: the same
        np.testing.assert_array_equal(
            np.concatenate([rows(r, min(r + 24, 64)) for r in range(0, 64, 24)]), want
        )
        # every merged value is a bf16 value
        np.testing.assert_array_equal(bf16(rows(0, 64)), rows(0, 64))

    def test_a_delta_below_half_an_ulp_leaves_the_weight_as_it_is(self):
        b, a = self.small
        merged = h3.vdn_merged_rows(lambda r0, r1: self.w[r0:r1], [(a, b, 1.0)])(0, 64)
        unrounded = self.w + b @ a
        kept = np.abs(b @ a) < np.abs(self.w) * 2.0**-9
        self.assertGreater(kept.mean(), 0.5)
        np.testing.assert_array_equal(merged[kept], self.w[kept])
        self.assertFalse(np.array_equal(merged, unrounded))


if __name__ == "__main__":
    unittest.main()

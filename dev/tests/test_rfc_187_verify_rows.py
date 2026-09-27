"""RFC 187 scaffolding test: verify-rows flag defaults to 8, off."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Rfc187VerifyRowsTest(unittest.TestCase):
    def test_header_defaults(self):
        header = ROOT / "runtime/ops/Rfc187VerifyRows.hpp"
        self.assertTrue(header.is_file())
        text = header.read_text()
        self.assertIn("kEnabledDefault = false", text)
        self.assertIn("SPLASH_VERIFY_ROWS", text)
        self.assertIn("kVerifyRowsDefault = 8", text)
        self.assertIn("kVerifyRowsCandidate = 16", text)

    def test_doc_exists(self):
        doc = ROOT / "docs/rfc/187-verify-16.md"
        self.assertTrue(doc.is_file())
        self.assertIn("issues/187", doc.read_text())

    def test_grid_and_warmup_roster(self):
        header = (ROOT / "runtime/ops/Rfc187VerifyRows.hpp").read_text()
        for symbol in ("rowsForLanes", "shapeSupported",
                       "kWarmVerifyPipelines", "static_assert"):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, header)
        # Every roster entry must name a kernel that ships in the metallib
        # source; the warm-up is a pre-build, not a new kernel.
        metal = (ROOT / "runtime/metal/kernels/decode/linear_q4.metal").read_text()
        for name in ("decode_linear_q4_n128_m16",
                     "decode_linear_q4_n256_m16",
                     "decode_linear_q4_n128_residual_m16",
                     "decode_linear_q4_n256_gate_up_m16"):
            with self.subTest(kernel=name):
                self.assertIn(name, metal)
                self.assertIn(name, header)


if __name__ == "__main__":
    unittest.main()

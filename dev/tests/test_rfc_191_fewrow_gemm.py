"""RFC 191 scaffolding test: few-row GEMM flag defaults off."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Rfc191FewRowGemmTest(unittest.TestCase):
    def test_header_defaults(self):
        header = ROOT / "runtime/ops/Rfc191FewRowGemm.hpp"
        self.assertTrue(header.is_file())
        text = header.read_text()
        self.assertIn("kEnabledDefault = false", text)
        self.assertIn("SPLASH_FEWROW_GEMM", text)
        self.assertIn("kBlockM = 32", text)

    def test_doc_exists(self):
        doc = ROOT / "docs/rfc/191-fewrow-gemm.md"
        self.assertTrue(doc.is_file())
        self.assertIn("issues/191", doc.read_text())

    def test_n64_kernels_present_in_metallib_source(self):
        src = ROOT / "runtime/metal/kernels/decode/linear_q4.metal"
        self.assertTrue(src.is_file())
        text = src.read_text()
        for name in (
            "decode_linear_q4_n64",
            "decode_linear_q4_n64_m16",
            "decode_linear_q4_n64_m32",
        ):
            with self.subTest(kernel=name):
                self.assertIn(name, text)
        header = ROOT / "runtime/ops/Rfc191FewRowGemm.hpp"
        header_text = header.read_text()
        for const in ("kN64Plain", "kN64M16", "kN64M32"):
            with self.subTest(const=const):
                self.assertIn(const, header_text)


if __name__ == "__main__":
    unittest.main()

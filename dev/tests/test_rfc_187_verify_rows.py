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


if __name__ == "__main__":
    unittest.main()

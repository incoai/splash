"""RFC 190 scaffolding test: deferred-GDN flag defaults off."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Rfc190DeferredGdnTest(unittest.TestCase):
    def test_header_defaults(self):
        header = ROOT / "runtime/ops/Rfc190DeferredGdn.hpp"
        self.assertTrue(header.is_file())
        text = header.read_text()
        self.assertIn("kEnabledDefault = false", text)
        self.assertIn("SPLASH_DEFERRED_GDN_REPLAY", text)

    def test_doc_exists(self):
        doc = ROOT / "docs/rfc/190-deferred-gdn.md"
        self.assertTrue(doc.is_file())
        self.assertIn("issues/190", doc.read_text())


if __name__ == "__main__":
    unittest.main()

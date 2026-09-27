"""RFC 192 scaffolding test: fused-attention flag defaults off."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Rfc192FusedAttentionTest(unittest.TestCase):
    def test_header_defaults(self):
        header = ROOT / "runtime/ops/Rfc192FusedAttention.hpp"
        self.assertTrue(header.is_file())
        text = header.read_text()
        self.assertIn("kEnabledDefault = false", text)
        self.assertIn("SPLASH_FUSED_PROMPT_ATTN", text)

    def test_doc_exists(self):
        doc = ROOT / "docs/rfc/192-fused-attn.md"
        self.assertTrue(doc.is_file())
        self.assertIn("issues/192", doc.read_text())


if __name__ == "__main__":
    unittest.main()

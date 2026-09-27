"""RFC 186 scaffolding test: prompt-lookup flag defaults off, doc present."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Rfc186PromptLookupTest(unittest.TestCase):
    def test_header_defaults_off(self):
        header = ROOT / "runtime/ops/Rfc186PromptLookup.hpp"
        self.assertTrue(header.is_file())
        text = header.read_text()
        self.assertIn("kEnabledDefault = false", text)
        self.assertIn("SPLASH_PROMPT_LOOKUP", text)
        self.assertIn("kMinSpanTokens = 16", text)

    def test_doc_exists(self):
        doc = ROOT / "docs/rfc/186-prompt-lookup.md"
        self.assertTrue(doc.is_file())
        self.assertIn("issues/186", doc.read_text())


if __name__ == "__main__":
    unittest.main()

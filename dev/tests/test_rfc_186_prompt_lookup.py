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

    def test_matcher_logic_mirrors_header_contract(self):
        # Python mirror of the header's constexpr matcher (no GPU needed).
        def match_at(haystack, off, needle):
            n = len(needle)
            if n == 0 or off + n > len(haystack):
                return False
            return haystack[off : off + n] == needle

        self.assertTrue(match_at([0, 1, 2, 3, 4, 5], 1, [1, 2, 3, 4]))
        self.assertFalse(match_at([0, 1, 2, 3, 4, 5], 2, [1, 2, 3, 4]))
        header = (ROOT / "runtime/ops/Rfc186PromptLookup.hpp").read_text()
        for symbol in (
            "PromptSpan",
            "meetsMinSpan",
            "hashTokens",
            "matchAt",
            "static_assert",
        ):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, header)


if __name__ == "__main__":
    unittest.main()

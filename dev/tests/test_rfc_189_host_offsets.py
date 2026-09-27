"""RFC 189 scaffolding test: host-offsets flag defaults off."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Rfc189HostOffsetsTest(unittest.TestCase):
    def test_header_defaults(self):
        header = ROOT / "runtime/ops/Rfc189HostOffsets.hpp"
        self.assertTrue(header.is_file())
        text = header.read_text()
        self.assertIn("kEnabledDefault = false", text)
        self.assertIn("SPLASH_HOST_POSITION_OFFSETS", text)

    def test_doc_exists(self):
        doc = ROOT / "docs/rfc/189-host-offsets.md"
        self.assertTrue(doc.is_file())
        self.assertIn("issues/189", doc.read_text())


if __name__ == "__main__":
    unittest.main()

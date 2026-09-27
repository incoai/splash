"""RFC 193 scaffolding test: work-interval flag defaults off."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Rfc193WorkIntervalTest(unittest.TestCase):
    def test_header_defaults(self):
        header = ROOT / "runtime/engine/Rfc193WorkInterval.hpp"
        self.assertTrue(header.is_file())
        text = header.read_text()
        self.assertIn("kEnabledDefault = false", text)
        self.assertIn("SPLASH_ENGINE_WORK_INTERVAL", text)

    def test_doc_exists(self):
        doc = ROOT / "docs/rfc/193-work-interval.md"
        self.assertTrue(doc.is_file())
        self.assertIn("issues/193", doc.read_text())

    def test_scope_contract(self):
        header = (ROOT / "runtime/engine/Rfc193WorkInterval.hpp").read_text()
        for symbol in ("IntervalScope", "WorkIntervalConfig",
                       "renewAllowed", "static_assert"):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, header)


if __name__ == "__main__":
    unittest.main()

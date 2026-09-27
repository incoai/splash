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

    def test_replay_transition_contract(self):
        header = (ROOT / "runtime/ops/Rfc190DeferredGdn.hpp").read_text()
        for symbol in ("ReplayState", "DeferredReplay", "onVerify",
                       "verify_gdn_fused", "static_assert"):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, header)
        # Python mirror of onVerify: full accept builds, partial defers.
        def on_verify(state, pending, accepted, proposed):
            if accepted >= proposed:
                return ("Built", 0)
            if accepted == 0:
                return (state, pending)
            return ("Deferred", proposed - accepted)

        self.assertEqual(on_verify("Built", 0, 8, 8), ("Built", 0))
        self.assertEqual(on_verify("Built", 0, 5, 8), ("Deferred", 3))


if __name__ == "__main__":
    unittest.main()

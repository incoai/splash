"""RFC 188 scaffolding test: in-place KV flag defaults off."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Rfc188InPlaceKvTest(unittest.TestCase):
    def test_header_defaults(self):
        header = ROOT / "runtime/ops/Rfc188InPlaceKv.hpp"
        self.assertTrue(header.is_file())
        text = header.read_text()
        self.assertIn("kEnabledDefault = false", text)
        self.assertIn("SPLASH_INPLACE_KV_APPEND", text)

    def test_doc_exists(self):
        doc = ROOT / "docs/rfc/188-inplace-kv.md"
        self.assertTrue(doc.is_file())
        self.assertIn("issues/188", doc.read_text())

    def test_squeezed_addressing_contract(self):
        header = (ROOT / "runtime/ops/Rfc188InPlaceKv.hpp").read_text()
        self.assertIn("chunkBaseForLane", header)
        self.assertIn("static_assert", header)
        device = (ROOT / "runtime/metal/kernels/common/paged_store_row.h").read_text()
        self.assertIn("splash_chunk_base_for_lane", device)
        # Host/device mirrors must agree on the squeezed base.
        self.assertIn("squeezed ? 0UL", device)
        self.assertIn("squeezed ? 0UL", header)


if __name__ == "__main__":
    unittest.main()

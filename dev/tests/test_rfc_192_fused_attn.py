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

    def test_fused_scale_and_warmup_roster(self):
        header = (ROOT / "runtime/ops/Rfc192FusedAttention.hpp").read_text()
        for symbol in ("fusedScaleAvailable", "kWarmPrefillPipelines",
                       "static_assert"):
            with self.subTest(symbol=symbol):
                self.assertIn(symbol, header)
        metal = (ROOT / "runtime/metal/kernels/prefill/attention_q8.metal").read_text()
        for name in ("prefill_attention_q8_split",
                     "prefill_attention_q8_reduce",
                     "prefill_attention_q8_split_kv2_g8",
                     "prefill_attention_q8_reduce_kv2_g8",
                     "prefill_attention_bf16_split",
                     "prefill_attention_bf16_split_kv2_g8"):
            with self.subTest(kernel=name):
                self.assertIn(name, metal)
                self.assertIn(name, header)


if __name__ == "__main__":
    unittest.main()

"""Inkling 2's GGUFs (general.architecture inkling2_loop): the configuration
and tokenizer the installer derives from a header like
unsloth/Inkling-2-Mini-GGUF's Mainline_Support/Inkling-2-Mini-UD-Q4_K_M
file (revision ae9327c5fbc7333116ba32eee78bb5d2bbefae4f), and from one in
Unsloth's own layout (inkling2), its root Inkling-2-Mini-UD-Q4_K_M.gguf
(revision 45fd08764f2b0bf16b03a5483a80e5c7aaec8772). Their architecture
metadata and tensor tables, read from the files' headers, are fixtures
(dev/tests/fixtures/model-configs/inkling-2-mini, gguf-*-inkling2.json
Unsloth's layout), and so are the config.json and the chat template of the
release they were made from, thinkingmachines/Inkling-2-Mini (revision
b85c3cfa615fc413cc9af3379f491b963f3b586a), which the header's
tokenizer.chat_template holds byte for byte."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from tokenizers import pre_tokenizers
from transformers import AutoTokenizer

from dev.tests.installer_fixtures import (
    CONFIGS,
    CONTROL,
    NORMAL,
    write_target,
)
from install import gguf, models

FIXTURES = CONFIGS / "inkling-2-mini"
TEMPLATE = (CONFIGS.parent / "chat_templates/inkling2.jinja").read_text()
LAYERS = 24
# Inkling 2's added tokens, IDs 200058 - 60 to 200057, every one a control
# token: those its template and its release name, and the unused ones.
BASE = 199998
NAMED = {
    199998: "<|unused|>",
    199999: "<|endoftext|>",
    200000: "<|message_user|>",
    200001: "<|message_model|>",
    200002: "<|message_system|>",
    200003: "<|message_tool|>",
    200004: "<|content_text|>",
    200005: "<|content_image|>",
    200006: "<|content_model_end_sampling|>",
    200008: "<|content_thinking|>",
    200010: "<|end_message|>",
    200022: "<|content_tool_error|>",
    200024: "<|content_xml|>",
    200028: "<|begin_of_text|>",
    200049: "<|content_invoke_tool_json|>",
    200057: "<|content_invoke_tool_text|>",
}
ADDED = [NAMED.get(index, f"<|unused_{index}|>") for index in range(BASE, 200058)]
# The split of thinkingmachines/Inkling-2-Mini's tokenizer.json before its
# byte-level BPE: o200k's pattern.
INKLING2_SPLIT = {
    "type": "Split",
    "pattern": {
        "Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    },
    "behavior": "Isolated",
    "invert": False,
}
# The keys model_config derives for every architecture that Inkling 2's
# release names otherwise, by the release's key: the engine's check reads the
# sizes of every family by these names.
GENERIC = {
    "max_position_embeddings": "model_max_length",
    "num_experts": "n_routed_experts",
}
LOOP_BOUNDS = "inkling2_loop.loop.start, inkling2_loop.loop.end"
# Unsloth's layout: its architecture metadata, and its loop's bounds.
UNSLOTH_METADATA = "gguf-metadata-inkling2.json"
UNSLOTH_LOOP_BOUNDS = "inkling2.loop.core_start, inkling2.loop.core_end"


def header_values(metadata="gguf-metadata.json"):
    """The header's metadata: the fixture's architecture metadata (of
    llama.cpp's converter's layout, or `metadata`) and a tokenizer of Inkling
    2's 200058 tokens, its added tokens at their IDs. The vocabulary holds
    "abc" whole, which no merge makes."""
    tokens = sorted(pre_tokenizers.ByteLevel.alphabet()) + ["ab", "abc"]
    tokens += [f"[base{index}]" for index in range(BASE - len(tokens))]
    return json.loads((FIXTURES / metadata).read_text()) | {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "inkling2",
        "tokenizer.ggml.tokens": tokens + ADDED,
        "tokenizer.ggml.token_type": [NORMAL] * BASE + [CONTROL] * len(ADDED),
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.eos_token_id": 200006,
        "tokenizer.ggml.eot_token_id": 200006,
        "tokenizer.ggml.padding_token_id": 199999,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_eos_token": False,
        "tokenizer.chat_template": TEMPLATE,
    }


def float32(value):
    """value as a float32 holds it, as the header stores a float."""
    if not isinstance(value, float):
        return value
    return struct.unpack("<f", struct.pack("<f", value))[0]


class Inkling2GgufTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def metadata(self, values=None):
        path = write_target(
            self.root / "inkling2.gguf",
            header_values() if values is None else values,
            (),
        )
        return gguf.Metadata(path)

    def test_the_derived_config_is_inkling_2_s_own(self):
        # The release's shape, its text model under text_config, with the
        # keys and values of that text_config the header states, each as the
        # float32 the header holds where it is a float; the sizes every
        # architecture's states, by their names; and the release's
        # eos_token_id, which it states beside text_config.
        config = gguf.model_config(self.metadata())
        release = json.loads((FIXTURES / "config.json").read_text())
        self.assertEqual(set(config), {"model_type", "text_config"})
        self.assertEqual(config["model_type"], release["model_type"])
        text, stated = config["text_config"], release["text_config"]
        self.assertEqual(
            {key: float32(value) for key, value in text.items()},
            {
                key: float32(
                    stated[GENERIC.get(key, key)]
                    if key != "eos_token_id"
                    else release["eos_token_id"]
                )
                for key in text
            },
        )
        self.assertEqual(set(text) - set(stated), {*GENERIC, "eos_token_id"})
        # What the header leaves out: the dtype, and the biases, norms and
        # shared experts its tensors show or lack.
        self.assertEqual(
            set(stated) - set(text),
            {
                "torch_dtype",
                "q_bias",
                "o_bias",
                "use_value_norm_no_gamma",
                "attn_res_normalize_values",
                "n_shared_experts",
                "use_gate_bias",
            },
        )
        # A float is the shortest decimal of the header's float32, as the
        # release writes it.
        self.assertEqual(
            (text["rms_norm_eps"], text["log_scaling_alpha"]), (1e-06, 0.1)
        )

    def test_each_derived_value_is_the_header_s(self):
        values = header_values()
        sliding = [layer % 4 != 3 for layer in range(LAYERS)]
        values |= {
            "inkling2_loop.attention.sliding_window_pattern": sliding,
            "inkling2_loop.attention.head_count": [16] * LAYERS,
            "inkling2_loop.attention.head_count_kv": [4] * LAYERS,
            "inkling2_loop.attention.key_length_swa": 64,
            "inkling2_loop.attention.sliding_window": 512,
            "inkling2_loop.attention.relative_length": 8,
            "inkling2_loop.attention.relative_extent": 512,
            "inkling2_loop.attention.qk_clip": 16.0,
            "inkling2_loop.attention.log_scale_alpha": 0.25,
            "inkling2_loop.vocab_size": 200064,
            "inkling2_loop.loop.start": 4,
            "inkling2_loop.loop.end": 20,
            "inkling2_loop.loop.count": 2,
            "inkling2_loop.loop.gate_rank": 32,
            "inkling2_loop.loop.varying": True,
            "inkling2_loop.expert_weights_scale": 2.5,
            "inkling2_loop.expert_feed_forward_length": 768,
            "inkling2_loop.shortconv.l_cache": 3,
            "inkling2_loop.attn_res.block_size": 4,
            "tokenizer.ggml.eos_token_id": 200010,
        }
        text = gguf.model_config(self.metadata(values))["text_config"]
        self.assertEqual(
            text["local_layer_ids"],
            [layer for layer, slides in enumerate(sliding) if slides],
        )
        self.assertEqual(
            {
                key: text[key]
                for key in (
                    "num_attention_heads",
                    "num_key_value_heads",
                    "swa_num_attention_heads",
                    "swa_num_key_value_heads",
                    "swa_head_dim",
                    "sliding_window_size",
                    "d_rel",
                    "rel_extent",
                    "qk_l2_clip_threshold",
                    "log_scaling_alpha",
                    "vocab_size",
                    "unpadded_vocab_size",
                    "loop_config",
                    "route_scale",
                    "intermediate_size",
                    "sconv_kernel_size",
                    "n_layers_per_attn_res_block",
                    "eos_token_id",
                )
            },
            {
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "swa_num_attention_heads": 16,
                "swa_num_key_value_heads": 4,
                "swa_head_dim": 64,
                "sliding_window_size": 512,
                "d_rel": 8,
                "rel_extent": 512,
                "qk_l2_clip_threshold": 16.0,
                "log_scaling_alpha": 0.25,
                "vocab_size": 200064,
                "unpadded_vocab_size": 200058,
                "loop_config": {
                    "core_start": 4,
                    "core_end": 20,
                    "num_passes": 2,
                    "gate_rank": 32,
                    "is_varloop": True,
                },
                "route_scale": 2.5,
                "intermediate_size": 768,
                "sconv_kernel_size": 3,
                "n_layers_per_attn_res_block": 4,
                "eos_token_id": 200010,
            },
        )
        # A head count stated once holds for every layer.
        values["inkling2_loop.attention.head_count"] = 12
        text = gguf.model_config(self.metadata(values))["text_config"]
        self.assertEqual(
            (text["num_attention_heads"], text["swa_num_attention_heads"]), (12, 12)
        )

    def test_metadata_the_config_cannot_state_is_refused(self):
        heads = [24] * LAYERS
        heads[5] = 16
        for key, value, reason in (
            (
                "inkling2_loop.vocab_size_unpadded",
                200057,
                "inkling2_loop.vocab_size_unpadded",
            ),
            ("inkling2_loop.vocab_size", 200000, "inkling2_loop.vocab_size"),
            ("inkling2_loop.attention.head_count", heads, "attention.head_count"),
            (
                "inkling2_loop.attention.head_count_kv",
                [8] * (LAYERS - 1),
                "attention.head_count_kv",
            ),
            (
                "inkling2_loop.attention.sliding_window_pattern",
                [True] * (LAYERS - 1),
                "inkling2_loop.attention.sliding_window_pattern",
            ),
            ("inkling2_loop.loop.start", 18, LOOP_BOUNDS),
            ("inkling2_loop.loop.end", 25, LOOP_BOUNDS),
            ("inkling2_loop.loop.varying", 0, "inkling2_loop.loop.varying"),
            ("inkling2_loop.loop.count", None, "inkling2_loop.loop.count"),
            ("inkling2_loop.attention.qk_clip", 22, "inkling2_loop.attention.qk_clip"),
            (
                "inkling2_loop.attention.relative_length",
                None,
                "inkling2_loop.attention.relative_length",
            ),
            (
                "inkling2_loop.activation.situ_beta",
                None,
                "inkling2_loop.activation.situ_beta",
            ),
            ("tokenizer.ggml.eos_token_id", None, "tokenizer.ggml.eos_token_id"),
        ):
            with self.subTest(key=key, value=value):
                values = header_values()
                if value is None:
                    del values[key]
                else:
                    values[key] = value
                with self.assertRaisesRegex(models.ModelError, reason):
                    gguf.model_config(self.metadata(values))

    def test_unsloth_s_layout_derives_the_same_config(self):
        # Unsloth's layout states its layer counts and arrays per layer
        # application (48: the loop's twelve layers three times), its loop by
        # other keys and no loop.varying, over the same tensors: the config is
        # the converter's layout's.
        values = header_values(UNSLOTH_METADATA)
        self.assertEqual(values["inkling2.block_count"], 48)
        self.assertEqual(
            gguf.model_config(self.metadata(values)),
            gguf.model_config(self.metadata()),
        )

    def test_unsloth_s_layout_maps_applications_to_weight_layers(self):
        # Two passes of layers 4-19 of 24: 40 applications. A weight layer past
        # the loop reads its application's value (layer 20's at 36), and the
        # loop's from its first pass.
        values = header_values(UNSLOTH_METADATA)
        weight = [layer % 4 != 3 for layer in range(LAYERS)]
        applications = weight[:20] + weight[4:]
        applications[36] = False
        values |= {
            "inkling2.block_count": 40,
            "inkling2.attention.sliding_window_pattern": applications,
            "inkling2.attention.head_count": [16] * 40,
            "inkling2.attention.head_count_kv": [4] * 40,
            "inkling2.loop.core_start": 4,
            "inkling2.loop.core_end": 20,
            "inkling2.num_loops": 2,
        }
        text = gguf.model_config(self.metadata(values))["text_config"]
        weight[20] = False
        self.assertEqual(
            text["local_layer_ids"],
            [layer for layer, slides in enumerate(weight) if slides],
        )
        self.assertEqual(
            (
                text["num_hidden_layers"],
                text["num_attention_heads"],
                text["swa_num_key_value_heads"],
                text["loop_config"],
            ),
            (
                LAYERS,
                16,
                4,
                {
                    "core_start": 4,
                    "core_end": 20,
                    "num_passes": 2,
                    "gate_rank": 64,
                    "is_varloop": False,
                },
            ),
        )
        for key, value, reason in (
            ("inkling2.loop.core_end", 30, UNSLOTH_LOOP_BOUNDS),
            ("inkling2.num_loops", None, "inkling2.num_loops"),
            (
                "inkling2.attention.sliding_window_pattern",
                applications[:-1],
                "inkling2.attention.sliding_window_pattern",
            ),
        ):
            with self.subTest(key=key, value=value):
                changed = dict(values)
                if value is None:
                    del changed[key]
                else:
                    changed[key] = value
                with self.assertRaisesRegex(models.ModelError, reason):
                    gguf.model_config(self.metadata(changed))

    def test_the_derived_tokenizer_is_inkling_2_s(self):
        files = gguf.tokenizer_files(self.metadata())
        for name, data in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        tokenizer = AutoTokenizer.from_pretrained(
            self.root / "tokenizer", local_files_only=True
        )
        backend = json.loads(files["tokenizer/tokenizer.json"])
        # o200k's split, then byte-level BPE, and no normalizer; a piece the
        # vocabulary holds whole is its token before any merge.
        self.assertIsNone(backend["normalizer"])
        self.assertEqual(backend["pre_tokenizer"]["pretokenizers"][0], INKLING2_SPLIT)
        self.assertEqual(
            [part["type"] for part in backend["pre_tokenizer"]["pretokenizers"]],
            ["Split", "ByteLevel"],
        )
        self.assertIs(backend["model"]["ignore_merges"], True)
        # Digits in runs of three, and a word with its contraction.
        pieces = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
        for text, spans in (("12345", [(0, 3), (3, 5)]), ("don't", [(0, 5)])):
            self.assertEqual([span for _, span in pieces(text)], spans)
        # "abc" is its whole token, though the merges make only "ab".
        self.assertEqual(tokenizer.encode("abc", add_special_tokens=False), [257])
        self.assertEqual(tokenizer.encode("ab", add_special_tokens=False), [256])
        # No NFC: a combining accent stays apart.
        self.assertNotEqual(tokenizer.encode("é"), tokenizer.encode("é"))
        # Every added token is a special token at its ID.
        self.assertEqual(
            tokenizer.encode("".join(ADDED), add_special_tokens=False),
            list(range(BASE, 200058)),
        )
        self.assertEqual(
            tokenizer.decode(list(range(BASE, 200058)), skip_special_tokens=True),
            "",
        )
        self.assertEqual(
            (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token),
            (None, "<|content_model_end_sampling|>", "<|endoftext|>"),
        )
        config = json.loads(files["tokenizer/tokenizer_config.json"])
        self.assertEqual(config["model_max_length"], 1048576)
        self.assertEqual(files["tokenizer/chat_template.jinja"].decode(), TEMPLATE)

    def test_the_fixtures_are_the_real_header_s(self):
        # The fixture tensor table is the real file's 744 tensors, and its
        # metadata the header's architecture keys.
        tensors = json.loads((FIXTURES / "gguf-tensors.json").read_text())
        self.assertEqual(len(tensors), 744)
        metadata = json.loads((FIXTURES / "gguf-metadata.json").read_text())
        self.assertEqual(metadata["general.architecture"], "inkling2_loop")
        self.assertTrue(
            all(
                key.startswith("inkling2_loop.")
                for key in metadata
                if key != "general.architecture"
            )
        )


if __name__ == "__main__":
    unittest.main()

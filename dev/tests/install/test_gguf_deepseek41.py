"""DeepSeek-V4.1's GGUFs (general.architecture deepseek41) in llama.cpp's
layout: the configuration, tokenizer and tensor screen the installer derives
from the headers of the IQ1_M files of
ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF (revision
65d53bc8be65123e255ff02f3ded958b6dbf54fd), the last part of six, the Engram
tables, which every variant shares as a file of its own name, and the
selection of their split files at the repository's root. Their scalar
metadata, arrays (but the token map's 129280 entries) and the tensor table
of all six parts, read from the files' headers, are fixtures
(dev/tests/fixtures/model-configs/deepseek-v4.1-flash), and so is the
config.json of the release they were made from,
deepseek-ai/DeepSeek-V4.1-Flash-1017 (revision
1bae0e1088993b9b82de08f59de5be5a34e42170)."""

import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokenizers import Tokenizer, pre_tokenizers

from dev.tests import fixture_files
from dev.tests.installer_fixtures import (
    CONFIGS,
    CONTROL,
    DEEPSEEK,
    GGML,
    NORMAL,
    USER_DEFINED,
)
from install import assembly, gguf, local, models, upstream

FIXTURES = CONFIGS / DEEPSEEK.name.lower()
TEMPLATE = (CONFIGS.parent / "chat_templates/deepseek41.jinja").read_text()
ENGRAM = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
SOURCE = "1bae0e1088993b9b82de08f59de5be5a34e42170"
VOCABULARY = 129280
# DeepSeek's added tokens the tests read, at their IDs and of the types the
# real files give them: its control tokens, its roles among them, special,
# and the tags its template writes in text, user-defined and not special.
ADDED = {
    0: ("<｜begin▁of▁sentence｜>", CONTROL),
    1: ("<｜end▁of▁sentence｜>", CONTROL),
    2: ("<｜▁pad▁｜>", CONTROL),
    128799: ("<｜System｜>", CONTROL),
    128803: ("<｜User｜>", CONTROL),
    128804: ("<｜Assistant｜>", CONTROL),
    128821: ("<think>", USER_DEFINED),
    128822: ("</think>", USER_DEFINED),
    128825: ("｜DSML｜", USER_DEFINED),
}
# The keys of DeepSeek's config.json no GGUF header states: training
# settings, the router's top-k method its correction bias implies, the
# vocabularies the Engram hash's primes and token map were made from, and
# the DSpark and MTP layers the GGUFs leave out.
UNSTATED = {
    "attention_bias",
    "attention_dropout",
    "hidden_act",
    "initializer_range",
    "tie_word_embeddings",
    "topk_method",
    "use_cache",
    "engram_vocab_size",
    "engram_compressed_vocab_size",
    "num_nextn_predict_layers",
    "dspark_block_size",
    "dspark_markov_rank",
    "dspark_n_routed_experts",
    "dspark_noise_token_id",
    "dspark_num_experts_per_tok",
    "dspark_target_layer_ids",
}
# The Engram tables' one tensor, the rows of every Engram layer's heads.
TABLE = ("per_layer_token_embd.weight", [256, 768022850], GGML["IQ4_NL"])


def header_values():
    """The first file's metadata: the scalars and arrays of the fixtures, its
    booleans, which scalar_metadata keeps as 1, a token map onto the Engram
    hash's compressed ids, and a tokenizer of DeepSeek's 129280 tokens, the
    added tokens the tests read at their IDs."""
    typed = json.loads((FIXTURES / "gguf-metadata.json").read_text())
    arrays = json.loads((FIXTURES / "gguf-arrays.json").read_text())
    tokens = [f"[base{index}]" for index in range(VOCABULARY)]
    types = [NORMAL] * VOCABULARY
    base = sorted(pre_tokenizers.ByteLevel.alphabet()) + ["12", "123"]
    tokens[3 : 3 + len(base)] = base
    for index, (text, kind) in ADDED.items():
        tokens[index], types[index] = text, kind
    return (
        typed["unsigned"]
        | typed["float"]
        | typed["string"]
        | arrays
        | {
            "deepseek41.expert_weights_norm": True,
            "deepseek41.engram.token_map": [
                token % 99092 for token in range(VOCABULARY)
            ],
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "joyai-llm",
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.merges": ["1 2", "12 3"],
            "tokenizer.ggml.bos_token_id": 0,
            "tokenizer.ggml.eos_token_id": 1,
            "tokenizer.ggml.padding_token_id": 1,
            "tokenizer.ggml.add_bos_token": False,
            "tokenizer.ggml.add_eos_token": False,
            "tokenizer.chat_template": TEMPLATE,
        }
    )


def tensor_table():
    """The real files' tensors: name, extents and type, in their order."""
    return [
        (name, extents, GGML[kind])
        for name, (kind, extents) in json.loads(
            (FIXTURES / "gguf-tensors.json").read_text()
        ).items()
    ]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


class DeepSeek41GgufTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def metadata(self, values=None, tensors=(), name="deepseek41.gguf"):
        path = fixture_files.write_gguf(
            self.root / name,
            header_values() if values is None else values,
            [(tensor, extents, kind, b"") for tensor, extents, kind in tensors],
        )
        return gguf.Metadata(path, tensors=True)

    def shared_part(self, values=None, name=ENGRAM):
        """The Engram tables' file, the last of six parts, which states no
        architecture of its own."""
        values = {
            "general.name": "DeepSeek-V4.1-Flash-1017 Engram tables",
            "deepseek41.engram.source": SOURCE,
            "split.no": 5,
            "split.count": 6,
            "split.tensors.count": len(tensor_table()),
        } | (values or {})
        return self.metadata(values, [TABLE], name=name)

    def test_the_derived_config_is_deepseek_s_own(self):
        config = gguf.model_config(self.metadata())
        release = json.loads((FIXTURES / "config.json").read_text())
        self.assertEqual(config["model_type"], release["model_type"])
        text, expected = config["text_config"], release["text_config"]
        # Every key of the release's text config but those no header states,
        # each of the header's value (its floats at f32), and the routed
        # experts again as num_experts, by which the engine plans a GGUF MoE.
        self.assertEqual(set(expected) - set(text), UNSTATED)
        self.assertEqual(set(text) - set(expected), {"num_experts", "eos_token_id"})
        for key in set(expected) & set(text):
            value = expected[key]
            if isinstance(value, float):
                self.assertEqual(text[key], f32(value), key)
            elif key == "rope_scaling":
                self.assertEqual(
                    text[key],
                    {
                        k: f32(v) if isinstance(v, float) else v
                        for k, v in value.items()
                    },
                )
            else:
                self.assertEqual(text[key], value, key)
        self.assertEqual(text["num_experts"], expected["n_routed_experts"])
        self.assertEqual(text["eos_token_id"], release["eos_token_id"])

    def test_metadata_the_config_cannot_state_is_refused(self):
        offsets = header_values()["deepseek41.ple.head_offsets"]
        for key, value, refusal in (
            ("deepseek41.expert_gating_func", 2, "expert gating function"),
            ("deepseek41.rope.scaling.type", "linear", "RoPE scaling"),
            ("deepseek41.swiglu_clamp_shexp", [7.0] * 40, "SwiGLU clamps"),
            ("deepseek41.attention.kv_source_layers", [8, 2], "kv_source_layers"),
            ("deepseek41.ple.layers", [1, 40], "ple.layers"),
            # Each head's rows follow the one before's in the one table.
            ("deepseek41.ple.head_offsets", [0, *offsets[2:], 0], "head_offsets"),
            ("deepseek41.ple.head_vocab_sizes", [16000057] * 47, "head_vocab_sizes"),
            # The layout ISTA-DASLab published before llama.cpp's.
            ("deepseek41.engram.file", ENGRAM, "until October 27, 2026"),
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(models.ModelError, refusal):
                    gguf.model_config(self.metadata(header_values() | {key: value}))

    def test_the_derived_tokenizer_splits_as_deepseek_s(self):
        files = gguf.tokenizer_files(self.metadata())
        backend = json.loads(files["tokenizer/tokenizer.json"])
        # DeepSeek's three splits, then byte-level BPE, without a normalizer.
        self.assertIsNone(backend["normalizer"])
        self.assertEqual(
            [part["type"] for part in backend["pre_tokenizer"]["pretokenizers"]],
            ["Split", "Split", "Split", "ByteLevel"],
        )
        tokenizer = Tokenizer.from_str(files["tokenizer/tokenizer.json"].decode())
        pieces = tokenizer.pre_tokenizer.pre_tokenize_str
        for text, spans in (
            # Digits in runs of three; a run of Han and katakana apart.
            ("12345", [(0, 3), (3, 5)]),
            ("中文テストabc", [(0, 5), (5, 8)]),
            # A punctuation mark opens the letters after it, and a run of
            # punctuation keeps its newlines.
            (".dotfile", [(0, 8)]),
            ("a, b!\n\n", [(0, 1), (1, 2), (2, 4), (4, 7)]),
        ):
            self.assertEqual([span for _, span in pieces(text)], spans, text)
        # Control tokens are special; the tags the template writes are text.
        self.assertEqual(
            tokenizer.encode("<｜User｜><think>｜DSML｜<｜end▁of▁sentence｜>").ids,
            [128803, 128821, 128825, 1],
        )
        self.assertEqual(files["tokenizer/chat_template.jinja"].decode(), TEMPLATE)

    def test_the_screen_reads_every_tensor_of_the_real_files(self):
        table = tensor_table()
        metadata = self.metadata(tensors=table)
        gguf.require_loadable(metadata)
        loaded = gguf.loaded_tensors(metadata)
        # Every tensor but the image router's bias, which text leaves alone.
        self.assertEqual(
            set(loaded),
            {name for name, _, _ in table if not name.endswith("exp_probs_b_vl.bias")},
        )
        self.assertEqual(len(table) - len(loaded), 40)
        # The KV sources compress, those of a ratio of two through a pooling
        # gate; the index sources score; the Engram layers project rows.
        for layer in (2, 8, 14):
            self.assertIn(f"blk.{layer}.attn_compressor_gate.weight", loaded)
        self.assertIn("blk.20.attn_compressor_kv.weight", loaded)
        self.assertNotIn("blk.20.attn_compressor_gate.weight", loaded)
        self.assertNotIn("blk.3.attn_compressor_kv.weight", loaded)
        self.assertIn("blk.24.indexer.attn_q_b.weight", loaded)
        self.assertNotIn("blk.24.indexer.attn_k.weight", loaded)
        self.assertEqual(
            {name for name in loaded if "engram_" in name},
            {
                f"blk.{layer}.engram_{part}.weight"
                for layer in (1, 14)
                for part in ("kv", "gate")
            },
        )
        self.assertEqual(loaded["blk.1.engram_gate.weight"], {"F32"})
        self.assertIn("IQ4_NL", loaded["per_layer_token_embd.weight"])

    def test_the_screen_refuses_what_the_loader_does_not_read(self):
        table = tensor_table()
        without_gate = [t for t in table if t[0] != "blk.2.attn_compressor_gate.weight"]
        sinks_q4 = [
            (name, extents, GGML["Q4_K"] if name == "blk.0.attn_sinks.weight" else kind)
            for name, extents, kind in table
        ]
        for tensors, refusal in (
            (without_gate, r"attn_compressor_gate\.weight missing \(1 tensor\)"),
            (sinks_q4, r"attn_sinks\.weight Q4_K \(1 tensor\)"),
        ):
            with self.subTest(refusal=refusal):
                with self.assertRaisesRegex(models.ModelError, refusal):
                    gguf.require_loadable(self.metadata(tensors=tensors))

    def test_the_engram_tables_are_the_last_part_by_their_header(self):
        first = self.metadata(
            header_values()
            | {
                "split.count": 6,
                "split.no": 0,
                "split.tensors.count": len(tensor_table()),
            }
        )
        self.assertTrue(gguf.last_part(first, self.shared_part()))
        for values in (
            {"split.no": 4},
            {"split.count": 7},
            {"split.tensors.count": 1042},
            {"general.architecture": "deepseek41"},
        ):
            with self.subTest(values=values):
                self.assertFalse(
                    gguf.last_part(first, self.shared_part(values, name="other.gguf"))
                )

    def test_a_later_part_is_no_target(self):
        with self.assertRaisesRegex(models.ModelError, "part 6 of a split GGUF"):
            gguf.model_config(self.shared_part())

    def test_split_files_at_the_root_select_their_variant(self):
        def parts(variant, count=6):
            return [
                f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-{variant}-{k:05d}-of-{count:05d}.gguf"
                for k in range(1, count)
            ]

        files = {
            *parts("IQ1_M"),
            *parts("IQ2_XXS"),
            *parts("IQ2_XS"),
            ENGRAM,
            "mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf",
            "README.md",
        }
        # Each variant's parts but the last, the Engram tables they share,
        # which the installer finds by the headers (upstream.shared_last_part).
        for variant in ("IQ1_M", "IQ2_XXS", "IQ2_XS"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    upstream.select_gguf(files, variant, shared_last=True),
                    (parts(variant), False),
                )
        with self.assertRaisesRegex(models.ModelError, "incomplete"):
            upstream.select_gguf(files, "IQ1_M")
        incomplete = files - {parts("IQ1_M")[2]}
        with self.assertRaisesRegex(models.ModelError, "incomplete"):
            upstream.select_gguf(incomplete, "IQ1_M", shared_last=True)
        # The Engram tables alone end in their own variant; the screen then
        # refuses them as no target (test_a_later_part_is_no_target).
        self.assertEqual(upstream.select_gguf(files, "IQ4_NL"), ([ENGRAM], True))
        self.assertEqual(
            upstream.last_part_name(parts("IQ1_M")[0]),
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00006-of-00006.gguf",
        )

    def test_local_files_prepare_with_their_shared_last_part_and_no_draft(self):
        # The engine knows the family's configuration once it loads the
        # family; until then the installer's check stands in for its own.
        directory = self.root / "DeepSeek-V4.1-Flash-GGUF"
        directory.mkdir()
        tensors = [tensor for tensor in tensor_table() if tensor[0] != TABLE[0]]
        names = [f"DeepSeek-V4.1-Flash-IQ1_M-{k:05d}-of-00003.gguf" for k in (1, 2)]
        split = {"split.count": 3, "split.tensors.count": len(tensors) + 1}
        for number, name in enumerate(names):
            self.metadata(
                (header_values() if number == 0 else {}) | split | {"split.no": number},
                tensors[number::2],
                name=f"{directory.name}/{name}",
            )
        # The Engram tables, the third part, under a name of their own.
        self.shared_part(split | {"split.no": 2}, name=f"{directory.name}/{ENGRAM}")
        last = "target/DeepSeek-V4.1-Flash-IQ1_M-00003-of-00003.gguf"
        with (
            mock.patch.object(upstream, "check_model", return_value=DEEPSEEK),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for target, linked in (
                (directory / names[1], False),
                (directory, False),
                # With llama.cpp's link to them beside the other parts.
                (directory, True),
            ):
                if linked:
                    (directory / Path(last).name).symlink_to(ENGRAM)
                with self.subTest(target=target.name, linked=linked):
                    record = assembly.verify(
                        local.prepare_local_model(target, models_dir=self.root / "m")
                    )
                    self.assertEqual(record["family"], "DeepSeek-V4.1-Flash")
                    self.assertEqual(
                        record["sources"]["draft"],
                        {"repo": models.NO_DRAFT, "revision": None},
                    )
                    self.assertLessEqual(
                        {*("target/" + name for name in names), last},
                        set(record["files"]),
                    )
                    self.assertFalse(
                        any(
                            n.startswith(("draft/", "engram/")) for n in record["files"]
                        )
                    )


if __name__ == "__main__":
    unittest.main()

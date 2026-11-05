"""Ripple's GGUFs (general.architecture ricursive): the configuration,
tokenizer and tensors the native loader reads, against the release's own
config.json and file.

The fixture holds the release file's scalars (gguf-metadata.json) and every
tensor of the real AD-Q4_K_M file with its type and extents
(gguf-tensors.json), so the screen is held to the release file itself."""

import json
import re
import tempfile
import unittest
from pathlib import Path

from tokenizers import pre_tokenizers

from dev.tests import fixture_files
from dev.tests.installer_fixtures import GGML
from install import gguf, models

FIXTURES = Path("dev/tests/fixtures/model-configs/ripple-65b")
LAYERS = 80

NORMAL = gguf.NORMAL_TOKEN
CONTROL = gguf.CONTROL_TOKEN

# The added tokens of the release's tokenizer.json, at their IDs; the ones
# the release marks special and the ones it does not.
ADDED = [
    (100256, "<|extra_id_0|>", False),
    (100257, "﻿", True),
    (100258, "<|fim_prefix|>", True),
    (100259, "<|fim_middle|>", True),
    (100260, "<|fim_suffix|>", True),
    (100261, "|||PHONE_NUMBER|||", False),
    (100262, "|||EMAIL_ADDRESS|||", False),
    (100263, "|||IP_ADDRESS|||", False),
    (100264, "<|im_start|>", True),
    (100265, "<|im_end|>", True),
    (100266, "<|extra_id_1|>", False),
    (100267, "<|extra_id_2|>", False),
    (100268, "<|extra_id_3|>", False),
    (100269, "<|extra_id_4|>", False),
    (100270, "<|extra_id_5|>", False),
    (100271, "<|extra_id_6|>", False),
    (100272, "<|extra_id_7|>", False),
    (100273, "<|extra_id_8|>", False),
    (100274, "<|extra_id_9|>", False),
    (100275, "<|extra_id_10|>", False),
    (100276, "<|endofprompt|>", True),
    (100277, "<|pad|>", True),
]

# The splits of ricursiveAI's tokenizer.json before its byte-level BPE: the
# DBRX converter's, which are llama-3's.
RIPPLE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


# The chat template the release GGUF embeds, which the derivation writes to
# chat_template.jinja (an excerpt is enough: the derivation only carries it).
CHAT_TEMPLATE = "{%- if tools %}{{- '<|im_start|>system\\n' }}{%- endif %}"


def scalar_metadata():
    """The release file header's scalars, as the fixture holds them."""
    typed = json.loads((FIXTURES / "gguf-metadata.json").read_text())
    return typed["unsigned"] | typed["float"] | typed["string"]


def header_values():
    """The header's metadata: its scalars, the layers its window bounds,
    which scalar_metadata leaves out, and a tokenizer of the release's
    100352 tokens, its added tokens at their IDs."""
    tokens = sorted(pre_tokenizers.ByteLevel.alphabet())
    # Two merges' results, so the BPE takes them: the release's first merges
    # are among the byte tokens' strings.
    tokens += ["Ġt", "in"]
    tokens += [f"[base{index}]" for index in range(len(tokens), 100256)]
    tokens += [text for _, text, _ in ADDED]
    tokens += [f"[pad{index}]" for index in range(100278, 100352)]
    added = {index: text for index, text, _ in ADDED}
    types = [CONTROL if index in added else NORMAL for index in range(len(tokens))]
    pattern = [layer % 5 != 4 for layer in range(LAYERS)]
    return (
        scalar_metadata()
        | {
            "tokenizer.chat_template": CHAT_TEMPLATE,
            "ricursive.attention.sliding_window_pattern": pattern,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "dbrx",
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.merges": ["Ġ t", "i n"],
            "tokenizer.ggml.bos_token_id": 100257,
            "tokenizer.ggml.eos_token_id": 100265,
            "tokenizer.ggml.unknown_token_id": 100257,
            "tokenizer.ggml.padding_token_id": 100277,
        }
    )


def tensor_table():
    """The real file's tensors: name, extents and type, in its order."""
    return [
        (name, extents, GGML[kind], b"")
        for name, (kind, extents) in json.loads(
            (FIXTURES / "gguf-tensors.json").read_text()
        ).items()
    ]


class RippleGgufTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def metadata(self, values=None, tensors=()):
        path = fixture_files.write_gguf(
            self.root / "ripple.gguf",
            header_values() if values is None else values,
            tensors,
        )
        return gguf.Metadata(path, tensors=True)

    def test_the_derived_config_is_the_release_s_own(self):
        # The keys and values of the release's config.json, flat, that the
        # header states: every size, the FFN's width, the window, the scale
        # and the clamp, the RoPE base, the layers that slide, and the tokens
        # that begin and pad a turn.
        config = gguf.model_config(self.metadata())
        release = json.loads((FIXTURES / "config.json").read_text())
        # The scale the header states is its F32 rounding of the release's;
        # every other value the derivation keeps is the release's own.
        self.assertEqual(
            config["embed_scale"], scalar_metadata()["ricursive.embedding_scale"]
        )
        headered = {key: value for key, value in config.items() if key != "embed_scale"}
        self.assertEqual(headered, {key: release[key] for key in headered})
        self.assertEqual(
            set(config),
            {
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "vocab_size",
                "max_position_embeddings",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "intermediate_size",
                "sliding_window",
                "embed_scale",
                "ffn_clamp_threshold",
                "rope_theta",
                "layer_types",
                "bos_token_id",
                "pad_token_id",
            },
        )

    def test_each_derived_value_is_the_header_s(self):
        sliding = [layer % 5 != 4 for layer in range(LAYERS)]
        values = header_values() | {
            "ricursive.attention.sliding_window_pattern": sliding,
            "ricursive.attention.sliding_window": 2048,
            "ricursive.embedding_scale": 32.0,
            "ricursive.swiglu_clamp": 12.0,
            "ricursive.rope.freq_base": 1000000.0,
        }
        config = gguf.model_config(self.metadata(values))
        self.assertEqual(
            config["layer_types"],
            ["sliding_attention" if slides else "full_attention" for slides in sliding],
        )
        self.assertEqual(
            {
                key: config[key]
                for key in (
                    "sliding_window",
                    "embed_scale",
                    "ffn_clamp_threshold",
                    "rope_theta",
                )
            },
            {
                "sliding_window": 2048,
                "embed_scale": 32.0,
                "ffn_clamp_threshold": 12.0,
                "rope_theta": 1000000.0,
            },
        )

    def test_a_sliding_pattern_of_other_layers_is_refused(self):
        for pattern in (
            [True] * (LAYERS - 1),
            ["sliding_attention"] * LAYERS,
        ):
            with (
                self.subTest(pattern=pattern[:3]),
                self.assertRaisesRegex(
                    models.ModelError, re.escape("attention.sliding_window_pattern")
                ),
            ):
                gguf.model_config(
                    self.metadata(
                        header_values()
                        | {"ricursive.attention.sliding_window_pattern": pattern}
                    )
                )

    def test_the_derived_tokenizer_splits_as_the_release_s_does(self):
        files = gguf.tokenizer_files(self.metadata())
        backend = json.loads(files["tokenizer/tokenizer.json"].decode())
        # The release's splits, then byte-level BPE, and no normalizer: the
        # release's tokenizer applies none.
        self.assertIsNone(backend["normalizer"])
        self.assertEqual(
            backend["pre_tokenizer"]["pretokenizers"],
            [
                {
                    "type": "Split",
                    "pattern": {"Regex": RIPPLE_REGEX},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "use_regex": False,
                    "trim_offsets": False,
                },
            ],
        )
        self.assertTrue(files["tokenizer/chat_template.jinja"].decode().strip())
        # Digits in runs of three, and a contraction's split whatever its
        # case: the release's splits, applied to a synthetic vocabulary whose
        # added tokens sit at their real IDs.
        from tokenizers import Tokenizer as Backend

        tokenizer = Backend.from_str(files["tokenizer/tokenizer.json"].decode())
        splits = tokenizer.pre_tokenizer.pre_tokenize_str
        # Digits in runs of three.
        self.assertEqual(
            [span for _, span in splits("12345")], [(0, 3), (3, 5)]
        )
        # A contraction's split whatever its case: the (?i:) of the release's
        # regex, which a lowercase-only transcription would lose.
        self.assertEqual(
            [span for _, span in splits("It's")],
            [span for _, span in splits("IT'S")],
            "the contraction split lost its case",
        )
        self.assertEqual(
            tokenizer.encode("<|im_start|>", add_special_tokens=False).ids, [100264]
        )
        self.assertEqual(
            tokenizer.encode("<|im_end|>", add_special_tokens=False).ids, [100265]
        )

    def test_the_screen_reads_every_tensor_of_the_real_file(self):
        table = tensor_table()
        metadata = self.metadata(tensors=table)
        gguf.require_loadable(metadata)
        self.assertEqual(
            set(gguf.loaded_tensors(metadata)), {name for name, _, _, _ in table}
        )
        loaded = gguf.loaded_tensors(metadata)
        # Every block: its four norms, its gate beside its queries, its
        # value-residual lambda, and its dense FFN.
        self.assertEqual(loaded["blk.4.attn_vres_lambda.weight"], {"F32"})
        self.assertIn("blk.4.attn_gate.weight", loaded)
        self.assertIn("blk.4.post_attention_norm.weight", loaded)
        self.assertIn("blk.4.post_ffw_norm.weight", loaded)
        self.assertIn("blk.79.ffn_down.weight", loaded)

    def test_the_screen_refuses_what_the_loader_does_not_read(self):
        def table(change):
            return [
                change(name, extents, kind, b"")
                for name, extents, kind, _ in tensor_table()
            ]

        for case, tensors, refusal in (
            (
                "a lambda of BF16",
                table(
                    lambda name, extents, kind, raw: (
                        name,
                        extents,
                        GGML["BF16"]
                        if name.endswith("attn_vres_lambda.weight")
                        else kind,
                        raw,
                    )
                ),
                "attn_vres_lambda.weight BF16 (80 tensors)",
            ),
            (
                "a block without its post-FFN norm",
                [
                    entry
                    for entry in tensor_table()
                    if entry[0] != "blk.0.post_ffw_norm.weight"
                ],
                "post_ffw_norm.weight missing (1 tensor)",
            ),
            (
                "a block without its gate",
                [
                    entry
                    for entry in tensor_table()
                    if entry[0] != "blk.0.attn_gate.weight"
                ],
                "attn_gate.weight missing (1 tensor)",
            ),
        ):
            with (
                self.subTest(case),
                self.assertRaisesRegex(models.ModelError, re.escape(refusal)),
            ):
                gguf.require_loadable(self.metadata(tensors=tensors))

    def test_the_fixture_metadata_is_the_header_s_scalars(self):
        # The engine's model check reads the scalars the fixture holds; this
        # holds them to the values the derivation and the engine's planner
        # hold the file to.
        metadata = self.metadata()
        fixture = json.loads((FIXTURES / "gguf-metadata.json").read_text())
        scalars = gguf.scalar_metadata(metadata)
        for kind in ("unsigned", "float", "string"):
            for key, value in fixture[kind].items():
                self.assertEqual(scalars[kind].get(key), value, key)
        self.assertEqual(gguf.model_config(metadata)["model_type"], "ricursive")


if __name__ == "__main__":
    unittest.main()

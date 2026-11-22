"""Inkling 2's GGUFs (general.architecture inkling2_loop): the configuration
and tokenizer the installer derives from a header like ggml-org's
Inkling-2-Mini-Q8_0.gguf, whose architecture metadata and tensor table are
fixtures (dev/tests/fixtures/model-configs/inkling-2-mini) beside the
release's config.json (thinkingmachines/Inkling-2-Mini, revision
b85c3cfa615f), and the quantized releases the screen takes.

The tokenizer parity test reads the real files where SPLASH_INKLING_GGUF
names a GGUF of the release and SPLASH_INKLING_TOKENIZER the release's own
tokenizer.json; it reads the GGUF's header alone."""

import json
import os
import re
import struct
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, decoders
from transformers import AutoTokenizer

from dev.tests import fixture_files
from dev.tests import inkling_fixtures as inkling
from dev.tests.installer_fixtures import GGML
from dev.tests.server_fixtures import NATIVE_CONTEXT
from install import families, gguf, hub, models, upstream
from server import constraints
from server.chat_format import INKLING
from server.chat_templates import NATIVE, ChatTemplates
from server.tokenization import PromptTokenizer

# The split of thinkingmachines/Inkling-2-Mini's tokenizer.json before its
# byte-level BPE, as that file holds it.
INKLING_SPLIT = {
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
RELEASE = json.loads((inkling.CONFIGS / "config.json").read_text())
# The keys the derived text_config holds beside the release's: the context
# and expert count every architecture's states, and the tokens that end a
# turn, which the release states at its top.
SPLASH_KEYS = {"max_position_embeddings", "num_experts", "eos_token_id"}
# The release's keys no header states: its dtype, and choices of the
# architecture llama.cpp's inkling2_loop makes for every model.
UNSTATED = {
    "torch_dtype",
    "q_bias",
    "o_bias",
    "use_value_norm_no_gamma",
    "attn_res_normalize_values",
    "use_gate_bias",
}
# A varied corpus: prose, Czech with diacritics, code, JSON, emoji, long
# numbers, whitespace runs, mixed scripts and contractions.
CORPUS = (
    "Hello world! The quick brown fox jumps over the lazy dog.",
    "Příliš žluťoučký kůň úpěl ďábelské ódy. ĚŠČŘŽÝÁÍÉ ůú ď ť ň",
    "def foo(x):\n    return {'a': [1, 2, 3]}  # comment\n\n\tif x >= 10:\r\n\t\tpass\n",
    '{"name":"weather","args":{"city":"Praha","days":3,"k":[true,false,null]}}',
    "emoji 😀👍🏽🇨🇿 👨‍👩‍👧‍👦 ✨ and more",
    "3141592653589793238462643383279502884197169399375105820974944592307816",
    "   leading      many    \n\n\n\t\t  trailing   ",
    "日本語のテキストと中文混合 한국어 текст на русском عربى हिन्दी",
    "I'm you're they'll WE'VE it's DON'T Hours Collins segundos",
    "https://example.com/path/to/file?q=1&b=2 /usr/local/bin//x a//b",
    "é vs é, K vs K, ﬁ ligature,  nbsp",
)


def single(value):
    """value as the float32 it is stored as."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


class InklingGgufTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def metadata(self, values=None, tensors=()):
        path = inkling.write_header(self.root / "inkling.gguf", values, tensors)
        return gguf.Metadata(path, tensors=True)

    def test_the_derived_config_is_inkling_s_own(self):
        # Nested as the release's config.json: its keys and values of the
        # text model that the header states, each float32 its own float.
        config = gguf.model_config(self.metadata())
        self.assertEqual(set(config), {"model_type", "text_config"})
        self.assertEqual(config["model_type"], RELEASE["model_type"])
        text, release = config["text_config"], RELEASE["text_config"]
        self.assertEqual(set(text), set(release) - UNSTATED | SPLASH_KEYS)
        for key in set(text) & set(release):
            with self.subTest(key=key):
                if isinstance(release[key], float):
                    self.assertEqual(single(text[key]), single(release[key]))
                else:
                    self.assertEqual(text[key], release[key])
        self.assertEqual(
            {key: text[key] for key in SPLASH_KEYS},
            {
                "max_position_embeddings": 1048576,
                "num_experts": 128,
                "eos_token_id": [RELEASE["eos_token_id"]],
            },
        )
        # Float32 values read as the configuration writes them.
        self.assertEqual(
            (text["rms_norm_eps"], text["log_scaling_alpha"]), (1e-06, 0.1)
        )

    def projector(self, name="mmproj-Inkling-2-Mini-Q8_0.gguf", **changed):
        """A header like ggml-org's inkling2v mmproj (its metadata, and its
        projector's two Q8_0 matrices among its tensors)."""
        values = {
            "general.architecture": "clip",
            "general.type": "mmproj",
            "clip.has_vision_encoder": True,
            "clip.vision.projection_dim": 2304,
            "clip.vision.image_size": 640,
            "clip.vision.patch_size": 20,
            "clip.vision.embedding_length": 1024,
            "clip.vision.feed_forward_length": 4096,
            "clip.vision.block_count": 23,
            "clip.vision.attention.head_count": 16,
            "clip.vision.image_mean": [0.5] * 3,
            "clip.vision.image_std": [0.5] * 3,
            "clip.vision.projector_type": "inkling2v",
            "clip.vision.attention.layer_norm_epsilon": 1e-5,
            "clip.vision.spatial_merge_size": 2,
        } | changed
        tensors = [
            ("v.patch_embd.weight", [20, 20, 3, 1024], GGML["F32"], b""),
            ("v.blk.0.attn_q.weight", [1024, 1024], GGML["Q8_0"], b""),
            ("mm.1.weight", [4096, 4096], GGML["Q8_0"], b""),
            ("mm.2.weight", [4096, 2304], GGML["Q8_0"], b""),
        ]
        path = fixture_files.write_gguf(self.root / name, values, tensors)
        return gguf.Metadata(path, tensors=True)

    def test_the_projector_derives_the_release_s_vision_config(self):
        # transformers' vision_config as the release's config.json states it,
        # but for the temporal patches its tower has none of.
        config = gguf.model_config(self.metadata(), self.projector())
        release = dict(RELEASE["vision_config"])
        del release["temporal_patch_size"]
        self.assertEqual(config["vision_config"], release)
        self.assertEqual(
            gguf.processor_config(self.projector()),
            {
                "patch_size": 20,
                "merge_size": 2,
                "image_mean": [0.5] * 3,
                "image_std": [0.5] * 3,
            },
        )
        with self.assertRaisesRegex(models.ModelError, "position grid"):
            gguf.vision_config(self.projector(**{"clip.vision.image_size": 650}))

    def test_the_family_takes_its_q8_0_projector_first(self):
        family = families.named("Inkling-2-Mini")
        self.assertTrue(family.vision)
        quantized = self.projector()
        self.projector("mmproj-Inkling-2-Mini-BF16.gguf")
        header, precision, kind = upstream.read_projector(
            hub.Repository.local_directory(self.root), "mmproj-Inkling-2-Mini-Q8_0.gguf"
        )
        self.assertEqual((precision, kind), ("Q8_0", "clip: F32, Q8_0"))
        self.assertEqual(header.values, quantized.values)
        # Inkling's processor is its own: 20-pixel patches of one frame.
        upstream._validate_processor(gguf.processor_config(quantized), family)
        with self.assertRaisesRegex(models.ModelError, "preprocessing"):
            upstream._validate_processor(
                gguf.processor_config(quantized), families.named("Qwen3.8-27B")
            )

    def test_each_derived_value_is_the_header_s(self):
        values = inkling.header_values()
        sliding = [layer % 3 != 0 for layer in range(24)]
        values |= {
            "inkling2_loop.attention.sliding_window_pattern": sliding,
            "inkling2_loop.attention.head_count": [16] * 24,
            "inkling2_loop.attention.head_count_kv": [4] * 24,
            "inkling2_loop.loop.start": 4,
            "inkling2_loop.loop.end": 12,
            "inkling2_loop.loop.count": 2,
            "inkling2_loop.loop.varying": True,
            "inkling2_loop.expert_weights_scale": 2.5,
            "inkling2_loop.expert_shared_count": 1,
            "inkling2_loop.vocab_size": 200064,
        }
        text = gguf.model_config(self.metadata(values))["text_config"]
        self.assertEqual(
            text["local_layer_ids"], [layer for layer in range(24) if layer % 3]
        )
        self.assertEqual(
            {
                key: text[key]
                for key in (
                    "num_attention_heads",
                    "swa_num_attention_heads",
                    "num_key_value_heads",
                    "swa_num_key_value_heads",
                    "route_scale",
                    "n_shared_experts",
                    "vocab_size",
                    "unpadded_vocab_size",
                    "loop_config",
                )
            },
            {
                "num_attention_heads": 16,
                "swa_num_attention_heads": 16,
                "num_key_value_heads": 4,
                "swa_num_key_value_heads": 4,
                "route_scale": 2.5,
                "n_shared_experts": 1,
                "vocab_size": 200064,
                "unpadded_vocab_size": 200058,
                "loop_config": {
                    "core_start": 4,
                    "core_end": 12,
                    "num_passes": 2,
                    "gate_rank": 64,
                    "is_varloop": True,
                },
            },
        )

    def test_the_tokens_ending_a_turn_are_stated_or_derived(self):
        stated = inkling.header_values() | {
            "tokenizer.ggml.eog_token_ids": [200006, 199999]
        }
        text = gguf.model_config(self.metadata(stated))["text_config"]
        self.assertEqual(text["eos_token_id"], [200006, 199999])
        values = inkling.header_values()
        values["tokenizer.ggml.eos_token_id"] = 199999
        with self.assertRaisesRegex(models.ModelError, "none of Inkling 2's stop"):
            gguf.model_config(self.metadata(values))

    def test_metadata_the_config_cannot_state_is_refused(self):
        for key, value, reason in (
            ("inkling2_loop.attention.head_count", [24] * 23 + [16], "one positive"),
            ("inkling2_loop.attention.head_count_kv", [8] * 23, "one positive"),
            ("inkling2_loop.vocab_size_unpadded", 200000, "vocab_size_unpadded"),
            ("inkling2_loop.attention.sliding_window_pattern", [True] * 23, "pattern"),
            ("inkling2_loop.loop.varying", None, "loop.varying"),
            ("inkling2_loop.attention.relative_length", None, "relative_length"),
            ("inkling2_loop.final_logit_softcapping", 30, "final_logit_softcapping"),
        ):
            with self.subTest(key=key):
                values = inkling.header_values()
                if value is None:
                    del values[key]
                else:
                    values[key] = value
                with self.assertRaisesRegex(models.ModelError, re.escape(reason)):
                    gguf.model_config(self.metadata(values))

    def test_unsloth_s_own_layout_is_refused_for_its_mainline_file(self):
        values = inkling.header_values()
        values["general.architecture"] = "inkling2"
        with self.assertRaisesRegex(models.ModelError, "Mainline_Support"):
            gguf.model_config(self.metadata(values))

    def test_each_quantized_release_passes_the_screen(self):
        # ggml-org's Q8_0 and IQ4_NL (experts of layers 0-19 IQ4_NL), and
        # Unsloth's Mainline_Support UD-Q8_K_XL (those experts NVFP4 with a
        # float scale each, everything else BF16), as their headers type
        # their tensors; a whole BF16 release is refused.
        ids = {name: kind for kind, name in gguf.TENSOR_TYPES.items()}
        table = inkling.tensor_table()

        def layer(name):
            return int(name.split(".")[1]) if name.startswith("blk.") else None

        def retyped(kind_of, extra=()):
            return [
                (name, extents, ids[kind_of(name, gguf.TENSOR_TYPES[kind])])
                for name, extents, kind in table
            ] + list(extra)

        def iq4(name, kind):
            expert = "_exps." in name and layer(name) < 20
            return "IQ4_NL" if expert else kind

        def unsloth(name, kind):
            if "_exps." in name and layer(name) < 20:
                return "NVFP4"
            return "BF16" if kind == "Q8_0" else kind

        scales = [
            (f"blk.{index}.ffn_{part}_exps.scale", [128], ids["F32"])
            for index in range(20)
            for part in ("gate_up", "down")
        ]
        for name, tensors in (
            ("Q8_0", table),
            ("IQ4_NL", retyped(iq4)),
            ("UD-Q8_K_XL", retyped(unsloth, scales)),
        ):
            with self.subTest(release=name):
                metadata = self.metadata(tensors=tensors)
                gguf.require_quantized([metadata])
                self.assertEqual(
                    gguf.model_config(metadata), gguf.model_config(self.metadata())
                )
        whole = self.metadata(tensors=retyped(lambda _name, kind: "BF16"))
        with self.assertRaisesRegex(models.ModelError, "unquantized"):
            gguf.require_quantized([whole])

    def test_the_derived_tokenizer_is_inkling_s_and_its_format_s(self):
        files = inkling.derived_files()
        backend = json.loads(files["tokenizer/tokenizer.json"])
        # Inkling's one split, then byte-level BPE, no normalizer, and a
        # split that is a token of its own is that token before any merge.
        self.assertIsNone(backend["normalizer"])
        self.assertEqual(
            backend["pre_tokenizer"]["pretokenizers"],
            [
                INKLING_SPLIT,
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        )
        self.assertIs(backend["model"]["ignore_merges"], True)
        tokenizer = inkling.inkling_tokenizer()
        ids = tokenizer.convert_tokens_to_ids
        self.assertEqual(
            tokenizer.encode("abc", add_special_tokens=False), [ids("abc")]
        )
        merged = Tokenizer.from_str(
            files["tokenizer/tokenizer.json"]
            .decode()
            .replace('"ignore_merges":true', '"ignore_merges":false')
        )
        self.assertEqual(merged.encode("abc").ids, [ids("ab"), ids("c")])
        # Digits in runs of three; a slash run stays with its punctuation.
        pieces = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
        self.assertEqual([span for _, span in pieces("12345")], [(0, 3), (3, 5)])
        self.assertEqual([piece for piece, _ in pieces(".//x")], [".//", "x"])
        # The frames the server reads decode as text; the stop token and the
        # padding, special, do not; no BOS or EOS is added.
        text = (
            "<|message_model|><|content_thinking|>abc<|end_message|>"
            "<|message_model|>t<|content_invoke_tool_json|>{}<|end_message|>"
        )
        tokens = tokenizer.encode(text + "<|content_model_end_sampling|><|endoftext|>")
        self.assertEqual(tokens[-2:], [200006, 199999])
        self.assertEqual(tokenizer.decode(tokens, skip_special_tokens=True), text)
        self.assertEqual(
            (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token),
            (None, "<|content_model_end_sampling|>", "<|endoftext|>"),
        )
        self.assertEqual(len(tokenizer), inkling.VOCABULARY)
        # The server reads it in Inkling's format, reuses its prompts' tokens
        # after <|end_message|> and serves its template unchanged, which has
        # no generation prompt.
        constraints.validate_tokenizer(tokenizer, INKLING)
        self.assertTrue(
            PromptTokenizer(tokenizer, INKLING.message_boundary, NATIVE_CONTEXT).enabled
        )
        self.assertEqual(
            files["tokenizer/chat_template.jinja"].decode(), inkling.TEMPLATE
        )
        template = ChatTemplates(tokenizer).select(None)
        self.assertEqual(template.later_system, NATIVE)
        self.assertEqual(template.generation_prompts, {})


@unittest.skipUnless(
    os.environ.get("SPLASH_INKLING_GGUF")
    and os.environ.get("SPLASH_INKLING_TOKENIZER"),
    "set SPLASH_INKLING_GGUF to an Inkling 2 GGUF and SPLASH_INKLING_TOKENIZER to "
    "the release's tokenizer.json",
)
class InklingTokenizerParityTests(unittest.TestCase):
    """The tokenizer the installer derives from a real Inkling 2 GGUF encodes
    as the release's own tokenizer.json, which transformers loads: on a
    varied corpus, and on every entry of the vocabulary as text, the case
    where llama.cpp, which merges every split, differs."""

    @classmethod
    def setUpClass(cls):
        metadata = gguf.Metadata(os.environ["SPLASH_INKLING_GGUF"])
        cls.config = gguf.model_config(metadata)
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        for name, data in gguf.tokenizer_files(metadata).items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(data)
        cls.derived = AutoTokenizer.from_pretrained(
            root / "tokenizer", local_files_only=True
        )
        cls.release = Tokenizer.from_file(os.environ["SPLASH_INKLING_TOKENIZER"])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_the_corpus_encodes_alike(self):
        for text in CORPUS:
            with self.subTest(text=text[:24]):
                self.assertEqual(
                    self.derived(text, add_special_tokens=False)["input_ids"],
                    self.release.encode(text, add_special_tokens=False).ids,
                )
        frames = "<|message_model|><|content_text|>hi<|end_message|>"
        self.assertEqual(
            self.derived(frames, add_special_tokens=False)["input_ids"],
            self.release.encode(frames, add_special_tokens=False).ids,
        )

    def test_every_vocabulary_entry_encodes_alike(self):
        byte_level = decoders.ByteLevel()
        texts = [
            byte_level.decode([token])
            for token, index in self.release.get_vocab(with_added_tokens=False).items()
        ]
        derived = self.derived.backend_tokenizer.encode_batch(
            texts, add_special_tokens=False
        )
        release = self.release.encode_batch(texts, add_special_tokens=False)
        differing = [
            text
            for text, ours, theirs in zip(texts, derived, release, strict=True)
            if ours.ids != theirs.ids
        ]
        self.assertEqual(differing, [])

    def test_the_header_derives_the_release_s_configuration(self):
        release = RELEASE["text_config"]
        for key, value in self.config["text_config"].items():
            if key in release:
                with self.subTest(key=key):
                    expected = release[key]
                    if isinstance(expected, float):
                        self.assertEqual(single(value), single(expected))
                    else:
                        self.assertEqual(value, expected)


if __name__ == "__main__":
    unittest.main()

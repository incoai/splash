"""Flint's GGUFs (general.architecture flint): the configuration and
tokenizer the installer derives from a header like unsloth/flint-GGUF's
flint-UD-Q4_K_XL.gguf (revision 20468edfc38ab43d6914cc88b08cbecb05d0fc62),
and its installation of one. Its scalar metadata and its tensor table, read
by HTTP range requests, are fixtures (dev/tests/fixtures/model-configs/flint),
and so are the config.json and the chat template of the release it was made
from, RecursiveSI/flint."""

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from tokenizers import pre_tokenizers
from transformers import AutoTokenizer

from dev.tests.installer_fixtures import (
    CONFIGS,
    CONTROL,
    GGML,
    NORMAL,
    USER_DEFINED,
    FakeHub,
    draft_dir,
    selection,
    tensor_table,
    vision_fixture,
    write_gguf,
    write_target,
)
from dev.tests.server_fixtures import NATIVE_CONTEXT
from install import assembly, families, gguf, local, models, upstream
from server import constraints
from server.chat_format import FLINT
from server.chat_templates import NATIVE, ChatTemplates
from server.tokenization import PromptTokenizer

FIXTURES = CONFIGS / "flint"
TEMPLATE = (CONFIGS.parent / "chat_templates/flint.jinja").read_text()
LAYERS = 49
# Flint's added tokens from ID 150000: its control tokens, special, and the
# tags its template writes in text, user-defined and not special.
ADDED = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<think>",
    "</think>",
    "<tools>",
    "</tools>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
    "<|developer|>",
    *(f"<|reserved_{index}|>" for index in range(150019, 150144)),
]
BASE = 150000
# The splits of RecursiveSI/flint's tokenizer.json (revision
# 2a247be0c0aec895ccb0ac7ca92229ec6df37288) before its byte-level BPE.
FLINT_SPLITS = [
    {
        "type": "Split",
        "pattern": {"Regex": "\\p{N}{1,3}"},
        "behavior": "Isolated",
        "invert": False,
    },
    {
        "type": "Split",
        "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"},
        "behavior": "Isolated",
        "invert": False,
    },
    {
        "type": "Split",
        "pattern": {
            "Regex": "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
            "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
            "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
            "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
            "| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        },
        "behavior": "Isolated",
        "invert": False,
    },
]


def header_values():
    """The header's metadata: the fixture's architecture metadata and a
    tokenizer of Flint's 150144 tokens, its added tokens at their IDs."""
    tokens = sorted(pre_tokenizers.ByteLevel.alphabet()) + ["12", "123"]
    tokens += [f"[base{index}]" for index in range(BASE - len(tokens))]
    return json.loads((FIXTURES / "gguf-metadata.json").read_text()) | {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "flint",
        "tokenizer.ggml.tokens": tokens + ADDED,
        "tokenizer.ggml.token_type": [NORMAL] * BASE
        + [USER_DEFINED if text in ADDED[6:18] else CONTROL for text in ADDED],
        "tokenizer.ggml.merges": ["1 2", "12 3"],
        "tokenizer.ggml.bos_token_id": 150000,
        "tokenizer.ggml.eos_token_id": 150001,
        "tokenizer.ggml.padding_token_id": 150001,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_eos_token": False,
        "tokenizer.ggml.eog_token_ids": [150001, 150003, 150005, 150018],
        "tokenizer.chat_template": TEMPLATE,
    }


class FlintGgufTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def metadata(self, values=None):
        path = write_target(
            self.root / "flint.gguf", header_values() if values is None else values, ()
        )
        return gguf.Metadata(path)

    def test_the_derived_config_is_flint_s_own(self):
        # The keys and values of the release's config.json, flat, that the
        # header states: every size, the layer patterns, the router and the
        # tokens that end a turn.
        config = gguf.model_config(self.metadata())
        release = json.loads((FIXTURES / "config.json").read_text())
        self.assertEqual(config, {key: release[key] for key in config})
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
                "num_experts",
                "num_experts_per_tok",
                "intermediate_size",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
                "norm_topk_prob",
                "router_score_function",
                "mlp_only_layers",
                "sliding_window",
                "layer_types",
                "no_rope_layer_indices",
                "eos_token_id",
            },
        )

    def test_each_derived_value_is_the_header_s(self):
        values = header_values()
        sliding = values["flint.attention.sliding_window_pattern"]
        rotated = [layer % 2 == 0 for layer in range(LAYERS)]
        values |= {
            "flint.attention.sliding_window_pattern": sliding,
            "flint.attention.rope_pattern": rotated,
            "flint.expert_gating_func": 1,
            "flint.expert_weights_norm": False,
            "flint.leading_dense_block_count": 3,
            "flint.attention.sliding_window": 4096,
            "tokenizer.ggml.eog_token_ids": [150001],
        }
        config = gguf.model_config(self.metadata(values))
        self.assertEqual(
            config["layer_types"],
            ["sliding_attention" if slides else "full_attention" for slides in sliding],
        )
        self.assertEqual(config["no_rope_layer_indices"], list(range(1, LAYERS, 2)))
        self.assertEqual(
            {
                key: config[key]
                for key in (
                    "router_score_function",
                    "norm_topk_prob",
                    "mlp_only_layers",
                    "sliding_window",
                    "eos_token_id",
                )
            },
            {
                "router_score_function": "softmax",
                "norm_topk_prob": False,
                "mlp_only_layers": [0, 1, 2],
                "sliding_window": 4096,
                "eos_token_id": [150001],
            },
        )

    def test_the_tokens_ending_a_turn_are_stated_or_derived(self):
        # unsloth/flint-GGUF states tokenizer.ggml.eog_token_ids, and the
        # config takes the list as it is.
        stated = header_values() | {"tokenizer.ggml.eog_token_ids": [150001, 150005]}
        config = gguf.model_config(self.metadata(stated))
        self.assertEqual(config["eos_token_id"], [150001, 150005])
        # lmstudio-community's, ggml-org's, bartowski's, mradermacher's and
        # prithivMLmods' state none: the IDs of Flint's stop tokens in the
        # vocabulary are unsloth's list.
        values = header_values()
        unsloth = values.pop("tokenizer.ggml.eog_token_ids")
        config = gguf.model_config(self.metadata(values))
        self.assertEqual(config["eos_token_id"], unsloth)

    def test_a_header_without_the_list_or_flint_s_stop_tokens_is_refused(self):
        def without_list(**changes):
            values = header_values()
            del values["tokenizer.ggml.eog_token_ids"]
            tokens = values["tokenizer.ggml.tokens"] = list(
                values["tokenizer.ggml.tokens"]
            )
            types = values["tokenizer.ggml.token_type"] = list(
                values["tokenizer.ggml.token_type"]
            )
            if "rename" in changes:
                tokens[changes["rename"]] = "<|renamed|>"
            if "text" in changes:
                types[changes["text"]] = USER_DEFINED
            if "twice" in changes:
                tokens[changes["twice"]] = "<|developer|>"
            if "eos" in changes:
                values["tokenizer.ggml.eos_token_id"] = changes["eos"]
            return values

        for changes, reason in (
            ({"rename": 150018}, re.escape("hold <|developer|> once")),
            ({"text": 150003}, re.escape("hold <|user|> once")),
            ({"twice": 150019}, re.escape("hold <|developer|> once")),
            ({"eos": 150000}, "its EOS is none of Flint's stop tokens"),
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(models.ModelError, reason):
                    gguf.model_config(self.metadata(without_list(**changes)))

    def test_metadata_the_config_cannot_state_is_refused(self):
        for key, value, reason in (
            ("tokenizer.ggml.eog_token_ids", [], "tokenizer.ggml.eog_token_ids"),
            (
                "tokenizer.ggml.eog_token_ids",
                [150001.0],
                "tokenizer.ggml.eog_token_ids",
            ),
            (
                "flint.attention.sliding_window_pattern",
                [True] * (LAYERS - 1),
                "flint.attention.sliding_window_pattern",
            ),
            (
                "flint.attention.rope_pattern",
                [1] * LAYERS,
                "flint.attention.rope_pattern",
            ),
            ("flint.attention.rope_pattern", None, "flint.attention.rope_pattern"),
            ("flint.expert_gating_func", 3, "expert gating function: 3"),
            ("flint.expert_weights_norm", 1, "flint.expert_weights_norm"),
            ("flint.leading_dense_block_count", LAYERS, "leading_dense_block_count"),
            ("flint.attention.sliding_window", None, "flint.attention.sliding_window"),
        ):
            with self.subTest(key=key, value=value):
                values = header_values()
                if value is None:
                    del values[key]
                else:
                    values[key] = value
                with self.assertRaisesRegex(models.ModelError, reason):
                    gguf.model_config(self.metadata(values))

    def test_the_derived_tokenizer_is_flint_s_and_its_format_s(self):
        metadata = self.metadata()
        files = gguf.tokenizer_files(metadata)
        for name, data in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        tokenizer = AutoTokenizer.from_pretrained(
            self.root / "tokenizer", local_files_only=True
        )
        backend = json.loads(files["tokenizer/tokenizer.json"])
        # Flint's splits, then byte-level BPE, and no normalizer: Qwen's NFC
        # would rewrite text Flint's tokenizer keeps.
        self.assertIsNone(backend["normalizer"])
        self.assertEqual(backend["pre_tokenizer"]["pretokenizers"][:3], FLINT_SPLITS)
        self.assertEqual(
            [part["type"] for part in backend["pre_tokenizer"]["pretokenizers"]],
            ["Split", "Split", "Split", "ByteLevel"],
        )
        # Digits in runs of three, and a run of Han and katakana apart.
        pieces = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
        for text, spans in (
            ("12345", [(0, 3), (3, 5)]),
            ("日本語テキストabc", [(0, 7), (7, 10)]),
        ):
            self.assertEqual([span for _, span in pieces(text)], spans)
        # No NFC: a combining accent stays apart.
        self.assertNotEqual(tokenizer.encode("e\u0301"), tokenizer.encode("\u00e9"))
        self.assertEqual(tokenizer.encode("123", add_special_tokens=False), [257])
        # Control tokens are special; the tags its template writes are text.
        think, call, end = 150006, 150010, 150001
        self.assertEqual(
            tokenizer.encode(
                "<think><tool_call><|end_of_text|>", add_special_tokens=False
            ),
            [think, call, end],
        )
        self.assertEqual(
            tokenizer.decode([think, call, end], skip_special_tokens=True),
            "<think><tool_call>",
        )
        self.assertEqual(
            (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token),
            ("<|begin_of_text|>", "<|end_of_text|>", "<|end_of_text|>"),
        )
        # The server reads it in Flint's chat format, reuses its prompts'
        # tokens after <|assistant|> and serves its template unchanged.
        constraints.validate_tokenizer(tokenizer, FLINT)
        self.assertTrue(
            PromptTokenizer(tokenizer, FLINT.message_boundary, NATIVE_CONTEXT).enabled
        )
        self.assertEqual(files["tokenizer/chat_template.jinja"].decode(), TEMPLATE)
        template = ChatTemplates(tokenizer).select(None)
        self.assertEqual(template.later_system, NATIVE)
        self.assertEqual(
            set(template.generation_prompts.values()), {("<|assistant|>", (150004,))}
        )

    @staticmethod
    def flint_gguf(path, values=None):
        """A GGUF target of Flint's header (values) and tensor table at path,
        and a vision projector beside it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        write_target(
            path,
            header_values() if values is None else values,
            tensor_table(families.named("Flint")),
        )
        write_gguf(
            path.parent / "mmproj-F32.gguf",
            vision_fixture(),
            [("v.patch_embd.weight", GGML["F32"])],
        )
        return path

    def test_a_repository_installs_text_only_without_language_only(self):
        # Flint's models have no vision tower: an installation that serves
        # images where it can serves Flint's text only, reads no projector its
        # repository holds and needs no --language-only.
        family = families.named("Flint")
        repo, variant = "unsloth/flint-GGUF", "UD-Q4_K_XL"
        target = f"flint-{variant}.gguf"
        fake = FakeHub(self, self.root / "hub")
        fake.publish(repo, "a" * 40, lambda root: self.flint_gguf(root / target))
        draft = draft_dir(self.root / "draft", family)
        chosen = selection(
            self.root,
            f"{repo}:{variant}",
            language_only=False,
            draft_model=str(draft),
        )
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            upstream.prepare(chosen)
        self.assertIn(
            f"as Flint (gguf); draft {draft}; vision disabled.",
            output.getvalue(),
        )
        record = assembly.verify(chosen.link, full=True)
        self.assertEqual((record["family"], record["vision_format"]), ("Flint", "none"))
        self.assertFalse((chosen.link / "vision").exists())
        self.assertEqual(fake.range_reads, [f"{repo}/{target}"])
        self.assertEqual(fake.downloads, [f"{repo}/{target}"])

    def test_a_header_without_the_list_installs_as_flint(self):
        # The engine's check holds the derived list to the family's.
        root = self.root.resolve()
        values = header_values()
        del values["tokenizer.ggml.eog_token_ids"]
        target = self.flint_gguf(root / "model" / "flint.gguf", values)
        draft = draft_dir(root / "draft", families.named("Flint"))
        with contextlib.redirect_stdout(io.StringIO()):
            link = local.prepare_local_model(target, draft, models_dir=root / "models")
        record = assembly.verify(link, full=True)
        self.assertEqual(record["family"], "Flint")
        config = json.loads((link / "config.json").read_text())
        self.assertEqual(config["eos_token_id"], [150001, 150003, 150005, 150018])

    def test_local_files_serve_text_only_and_take_no_projector(self):
        # Resolved, as prepare_local_model names the files it prepares.
        root = self.root.resolve()
        target = self.flint_gguf(root / "model" / "flint.gguf")
        mmproj = target.parent / "mmproj-F32.gguf"
        draft = draft_dir(root / "draft", families.named("Flint"))

        def prepare(**options):
            with contextlib.redirect_stdout(io.StringIO()):
                return local.prepare_local_model(
                    target, draft, models_dir=root / "models", **options
                )

        self.assertEqual(assembly.verify(prepare())["vision_format"], "none")
        with self.assertRaisesRegex(
            models.ModelError,
            re.escape(
                f"{mmproj} is a vision projector, which a Flint model, without a "
                "vision tower, does not load"
            ),
        ):
            prepare(mmproj=mmproj)


if __name__ == "__main__":
    unittest.main()

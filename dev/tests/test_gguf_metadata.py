import contextlib
import io
import re
import shutil
import struct
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from tokenizers import Tokenizer, pre_tokenizers
from transformers import AutoTokenizer

from dev.tests import fixture_files
from dev.tests.fixture_files import GGUF_TYPE_ARRAY, GGUF_TYPE_STRING
from dev.tests.installer_fixtures import (
    DRAFT_COMMIT,
    MOE,
    FakeHub,
    draft_dir,
    selection,
)
from install import assembly, families, gguf, hub, models, upstream

GGUF_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"

# The GGUF specification's token types (llama_token_type), stated
# independently of the reader under test.
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
# GGML tensor types (ggml_type), stated independently of gguf.TENSOR_TYPES.
GGML = {
    "F32": 0,
    "F16": 1,
    "Q5_0": 6,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q4_K": 12,
    "IQ2_XXS": 16,
    "IQ3_XXS": 18,
    "IQ4_XS": 23,
    "IQ1_M": 29,
    "BF16": 30,
    "MXFP4": 39,
}


def write_gguf(path, values, tensors=()):
    """A GGUF header of the values and the tensors' (name, GGML type), each
    tensor 256 values without data: the metadata reader reads no more."""
    table = [(name, [256], kind, b"") for name, kind in tensors]
    return fixture_files.write_gguf(path, values, table)


def fixture(*, native=False):
    tokens = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokens += ["ab", "<|endoftext|>", "<|im_end|>", "<think>", "<|im_start|>"]
    types = [NORMAL] * 257 + [CONTROL, CONTROL, USER_DEFINED, CONTROL]
    if native:
        padding = 248320 - len(tokens)
        tokens += [f"[unused{i}]" for i in range(padding)]
        types += [UNUSED] * padding
    return {
        "general.architecture": "qwen35moe",
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.block_count": 40,
        "qwen35moe.full_attention_interval": 4,
        "qwen35moe.context_length": 262144,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.expert_count": 256,
        "qwen35moe.expert_used_count": 8,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.eos_token_id": 258,
        "tokenizer.ggml.bos_token_id": 257,
        "tokenizer.ggml.padding_token_id": 257,
        "tokenizer.chat_template": "{% for message in messages %}{{ message.content }}<|im_end|>{% endfor %}",
    }


def loadable_tensors(values, directory):
    """A tensor table the native loader accepts for the header values: each
    tensor it reads, quantized as Q4_K, Q8_0 (GDN alpha and beta) or F32."""
    header = gguf.Metadata(write_gguf(directory / "header.gguf", values))
    return {
        name: GGML[next(t for t in ("Q4_K", "Q8_0", "F32") if t in types)]
        for name, types in gguf.loaded_tensors(header).items()
    }


def vision_fixture():
    return {
        "general.architecture": "clip",
        "clip.projector_type": "qwen3vl_merger",
        "clip.use_gelu": True,
        "clip.vision.block_count": 27,
        "clip.vision.embedding_length": 1152,
        "clip.vision.attention.head_count": 16,
        "clip.vision.feed_forward_length": 4304,
        "clip.vision.projection_dim": 2048,
        "clip.vision.patch_size": 16,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.image_size": 768,
        "clip.vision.is_deepstack_layers": [False] * 27,
        "clip.vision.image_mean": [0.5] * 3,
        "clip.vision.image_std": [0.5] * 3,
    }


class GgufMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def metadata(self, values):
        return gguf.Metadata(write_gguf(self.root / "model.gguf", values))

    def test_reader_stops_before_weights_and_checks_bounds(self):
        path = write_gguf(self.root / "small.gguf", {"key": "value"})
        expected = path.stat().st_size
        with path.open("ab") as stream:
            stream.write(b"tensor payload is not metadata")
        with mock.patch.object(gguf.Metadata, "MAX_BYTES", expected):
            self.assertEqual(gguf.Metadata(path).values, {"key": "value"})
        # A caller's stream stays open, at the end of the metadata.
        with path.open("rb") as stream:
            self.assertEqual(gguf.Metadata(stream).values, {"key": "value"})
            self.assertEqual(stream.tell(), expected)
        raw = path.read_bytes()[:expected]
        for size in range(len(raw)):
            path.write_bytes(raw[:size])
            with (
                self.subTest(size=size),
                self.assertRaisesRegex(models.ModelError, "truncated"),
            ):
                gguf.Metadata(path)
        path.write_bytes(raw)
        with mock.patch.object(gguf.Metadata, "MAX_BYTES", 24):
            with self.assertRaisesRegex(models.ModelError, "size limit"):
                gguf.Metadata(path)
        # Version 3, little-endian, as the native reader requires.
        for version in (struct.pack(">I", 3), struct.pack("<I", 2)):
            path.write_bytes(raw[:4] + version + raw[8:])
            with (
                self.subTest(version=version),
                self.assertRaisesRegex(models.ModelError, "unsupported GGUF header"),
            ):
                gguf.Metadata(path)

    def test_reader_rejects_duplicate_keys_unknown_types_and_nested_arrays(self):
        path = write_gguf(self.root / "bad.gguf", {"key": "value"})
        raw = path.read_bytes()
        # The header is the magic, the version, the tensor count and the key
        # count; the one key's value type follows its length and bytes.
        header = struct.calcsize("<4sIQQ")
        key_count = header - struct.calcsize("<Q")
        value_type = header + struct.calcsize("<Q") + len("key")
        path.write_bytes(raw[:key_count] + struct.pack("<Q", 2) + raw[header:] * 2)
        with self.assertRaisesRegex(models.ModelError, "duplicate"):
            gguf.Metadata(path)
        # Each is rejected for its own reason, not read on to the end.
        for kind, data, reason in (
            (99, b"", "unknown GGUF metadata type"),
            (
                GGUF_TYPE_ARRAY,
                struct.pack("<IQ", GGUF_TYPE_ARRAY, 1),
                "unsupported or oversized",
            ),
            (
                GGUF_TYPE_ARRAY,
                struct.pack("<IQ", GGUF_TYPE_STRING, gguf.Metadata.MAX_ITEMS + 1),
                "unsupported or oversized",
            ),
            (GGUF_TYPE_STRING, struct.pack("<Q", 1) + b"\xff", "invalid UTF-8"),
        ):
            path.write_bytes(raw[:value_type] + struct.pack("<I", kind) + data)
            with (
                self.subTest(kind=kind, data=data),
                self.assertRaisesRegex(models.ModelError, reason),
            ):
                gguf.Metadata(path)

    def test_reader_checks_counts_tensor_ranks_and_duplicate_tensors(self):
        path = self.root / "tensors.gguf"
        # The native reader's bound on each of the tensor and key counts.
        header = struct.pack("<4sI", b"GGUF", 3)
        for tensors, keys in ((16385, 0), (0, 16385)):
            path.write_bytes(header + struct.pack("<QQ", tensors, keys))
            with (
                self.subTest(tensors=tensors, keys=keys),
                self.assertRaisesRegex(models.ModelError, "too many fields"),
            ):
                gguf.Metadata(path, tensors=True)
        path.write_bytes(header + struct.pack("<QQ", 16384, 16384))
        with self.assertRaisesRegex(models.ModelError, "truncated"):
            gguf.Metadata(path, tensors=True)
        # Ranks up to the native reader's limit, GGML_MAX_DIMS.
        for rank in (1, 4):
            fixture_files.write_gguf(path, {}, [("t", [2] * rank, GGML["F32"], b"")])
            self.assertEqual(
                gguf.Metadata(path, tensors=True).tensors, {"t": GGML["F32"]}
            )
        fixture_files.write_gguf(path, {}, [("t", [2] * 5, GGML["F32"], b"")])
        with self.assertRaisesRegex(models.ModelError, "invalid GGUF tensor rank: t"):
            gguf.Metadata(path, tensors=True)
        write_gguf(path, {}, [("t", GGML["F32"]), ("t", GGML["F32"])])
        with self.assertRaisesRegex(models.ModelError, "duplicate GGUF tensor: t"):
            gguf.Metadata(path, tensors=True)

    def test_bpe_ids_special_tokens_normalization_and_local_reload(self):
        metadata = self.metadata(fixture())
        files = gguf.tokenizer_files(metadata)
        for name, data in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        with mock.patch(
            "socket.socket.connect", side_effect=AssertionError("network access")
        ):
            tokenizer = AutoTokenizer.from_pretrained(
                self.root / "tokenizer", local_files_only=True
            )
        backend = Tokenizer.from_str(files["tokenizer/tokenizer.json"].decode())
        tokens = metadata.values["tokenizer.ggml.tokens"]
        self.assertEqual(
            backend.get_vocab(), {token: i for i, token in enumerate(tokens)}
        )
        self.assertEqual(
            tokenizer.encode("ab<think><|im_end|>", add_special_tokens=False),
            [256, 259, 258],
        )
        self.assertEqual(tokenizer.encode("ab"), [256])  # No implicit BOS/EOS.
        # Control tokens are special whether or not a special-token ID names
        # them; the user-defined <think> is not.
        self.assertEqual(
            tokenizer.decode([260, 256, 259, 258], skip_special_tokens=True),
            "ab<think>",
        )
        self.assertEqual(tokenizer.encode("e\u0301"), tokenizer.encode("é"))
        # Combining marks stay with letters; Qwen2's older pattern splits these.
        for word in ("क्", "a\u035c"):
            self.assertEqual(len(backend.pre_tokenizer.pre_tokenize_str(word)), 1)
        self.assertEqual(len(backend.pre_tokenizer.pre_tokenize_str("123")), 3)
        text = "日本語 中文 👩🏽‍💻 \n\t 0123"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "ab"}], tokenize=True, return_dict=False
            ),
            [256, 258],
        )
        self.assertEqual(
            files["tokenizer/chat_template.jinja"].decode(),
            metadata.values["tokenizer.chat_template"],
        )

    def test_invalid_tokenizer_metadata_is_rejected(self):
        types = fixture()["tokenizer.ggml.token_type"]
        for key, value in (
            ("tokenizer.ggml.pre", "unknown"),
            ("tokenizer.ggml.model", "llama"),
            ("tokenizer.ggml.tokens", ["a", "a"]),
            ("tokenizer.ggml.token_type", [NORMAL]),
            # Token types outside the byte-level BPE profile.
            ("tokenizer.ggml.token_type", types[:-1] + [UNKNOWN]),
            ("tokenizer.ggml.token_type", types[:-1] + [BYTE]),
            ("tokenizer.ggml.merges", ["a"]),
            ("tokenizer.ggml.merges", ["a absent"]),
            ("tokenizer.ggml.eos_token_id", 99999),
            # A special token must be a control token: not a normal token,
            # not the user-defined <think>.
            ("tokenizer.ggml.eos_token_id", 0),
            ("tokenizer.ggml.eos_token_id", 259),
            ("tokenizer.ggml.add_bos_token", True),
            ("tokenizer.ggml.add_eos_token", True),
            ("tokenizer.chat_template", ""),
        ):
            with self.subTest(key=key, value=value):
                values = fixture()
                values[key] = value
                with self.assertRaises(models.ModelError):
                    gguf.tokenizer_files(self.metadata(values))

    def test_screening_accepts_each_tensor_as_the_native_loader_reads_it(self):
        values = fixture()
        values["qwen35moe.block_count"] = 41
        values["qwen35moe.nextn_predict_layers"] = 1
        tensors = loadable_tensors(values, self.root)
        # Layer 3 is full attention; the others around it GDN.
        self.assertIn("blk.3.attn_q.weight", tensors)
        self.assertNotIn("blk.2.attn_q.weight", tensors)
        # GDN alpha and beta may also both be F32, and the MTP layer (block
        # 40) is never loaded, so its types do not matter.
        tensors |= {"blk.1.ssm_alpha.weight": GGML["F32"]}
        tensors |= {"blk.1.ssm_beta.weight": GGML["F32"]}
        tensors |= {"blk.40.ffn_up_exps.weight": GGML["BF16"]}
        tensors |= {"blk.0.ffn_down_exps.weight": GGML["IQ4_XS"]}
        # The low-bit formats of Unsloth's smaller files, the embedding too.
        tensors |= {"blk.4.ffn_gate_exps.weight": GGML["IQ3_XXS"]}
        tensors |= {"blk.2.attn_qkv.weight": GGML["IQ1_M"]}
        tensors |= {"blk.5.ffn_up_exps.weight": GGML["IQ2_XXS"]}
        tensors |= {"token_embd.weight": GGML["Q2_K"]}
        path = write_gguf(self.root / "ok.gguf", values, tensors.items())
        gguf.require_loadable(gguf.Metadata(path, tensors=True))
        f32 = {name: GGML["F32"] for name in tensors}
        for changes, reason in (
            (
                {"blk.4.ffn_gate_exps.weight": GGML["Q5_0"]},
                "ffn_gate_exps.weight Q5_0 [(]1 tensor[)]",
            ),
            (
                {
                    "blk.0.attn_qkv.weight": GGML["BF16"],
                    "blk.1.attn_qkv.weight": GGML["BF16"],
                },
                "attn_qkv.weight BF16 [(]2 tensors[)]",
            ),
            ({"token_embd.weight": GGML["IQ4_XS"]}, "token_embd.weight IQ4_XS"),
            ({"blk.3.attn_q.weight": GGML["BF16"]}, "attn_q.weight BF16"),
            # F32 only where the loader reads floats: not a projection, not
            # a quantized router or norm, not half an alpha/beta pair.
            ({"blk.3.attn_q.weight": GGML["F32"]}, "attn_q.weight F32"),
            (
                {"blk.0.ffn_gate_inp.weight": GGML["Q8_0"]},
                "ffn_gate_inp.weight Q8_0",
            ),
            ({"output_norm.weight": GGML["F16"]}, "output_norm.weight F16"),
            (
                {"blk.1.ssm_alpha.weight": GGML["Q8_0"]},
                "ssm_alpha.weight and ssm_beta.weight of different types",
            ),
            # An all-F32 file, whose types the loader reads somewhere.
            (f32, "attn_output.weight F32 [(]10 tensors[)]"),
        ):
            with self.subTest(reason=reason):
                path = write_gguf(
                    self.root / "bad.gguf", values, (tensors | changes).items()
                )
                with self.assertRaisesRegex(
                    models.ModelError, "cannot load: .*" + reason
                ):
                    gguf.require_loadable(gguf.Metadata(path, tensors=True))
        # Every tensor the loader reads must be present.
        missing = {n: k for n, k in tensors.items() if n != "blk.7.attn_k.weight"}
        path = write_gguf(self.root / "bad.gguf", values, missing.items())
        with self.assertRaisesRegex(models.ModelError, "attn_k.weight missing"):
            gguf.require_loadable(gguf.Metadata(path, tensors=True))
        # Without tensors=True the reader never reads the tensor table.
        self.assertEqual(gguf.Metadata(path).tensors, {})

    def test_loadable_types_are_the_native_formats(self):
        # The installer's list must be the loader's own: kQuantFormats' GGML
        # types (runtime/metal/abi/QuantFormat.h).
        header = (
            Path(__file__).resolve().parents[2] / "runtime/metal/abi/QuantFormat.h"
        ).read_text()
        table = header.split("kQuantFormats[GGUF_FMT_COUNT] = {", 1)[1].split("};", 1)[
            0
        ]
        native = {gguf.TENSOR_TYPES[int(n)] for n in re.findall(r"\{(\d+),", table)}
        self.assertEqual(native, gguf.QUANTIZED_TYPES)
        self.assertLessEqual(gguf.EMBEDDING_TYPES, gguf.QUANTIZED_TYPES)

    def test_every_derivation_names_an_unsupported_architecture(self):
        values = fixture()
        values["general.architecture"] = "llama"
        metadata = self.metadata(values)
        for derive in (gguf.require_loadable, gguf.model_config, gguf.tokenizer_files):
            with (
                self.subTest(derive=derive.__name__),
                self.assertRaisesRegex(
                    models.ModelError, "unsupported GGUF model architecture: llama"
                ),
            ):
                derive(metadata)

    def test_moe_config_states_its_experts_and_identifies_the_family(self):
        config = gguf.model_config(self.metadata(fixture(native=True)))
        text = config["text_config"]
        self.assertEqual((text["num_experts"], text["num_experts_per_tok"]), (256, 8))
        self.assertEqual(families.family_for(config).name, "Qwen3.6-35B-A3B")

    def test_config_uses_metadata_and_subtracts_only_mtp_layers(self):
        values = fixture()
        values["qwen35moe.block_count"] = 41
        values["qwen35moe.nextn_predict_layers"] = 1
        text = gguf.model_config(self.metadata(values))["text_config"]
        self.assertEqual(text["num_hidden_layers"], 40)
        self.assertEqual(text["hidden_size"], 2048)
        self.assertEqual(text["max_position_embeddings"], 262144)
        # Tensor screening reads the layer count first, and rejects an
        # invalid MTP count as configuration derivation does.
        for mtp in (41, "1"):
            values["qwen35moe.nextn_predict_layers"] = mtp
            metadata = self.metadata(values)
            for derive in (gguf.require_loadable, gguf.model_config):
                with (
                    self.subTest(mtp=mtp, derive=derive.__name__),
                    self.assertRaisesRegex(
                        models.ModelError, "invalid GGUF MTP layer count"
                    ),
                ):
                    derive(metadata)

    def test_vision_config_derives_the_position_grid_and_rejects_other_towers(self):
        vision = self.metadata(vision_fixture())
        # 768-pixel images in 16-pixel patches: a 48 x 48 position grid.
        self.assertEqual(gguf.vision_config(vision)["num_position_embeddings"], 2304)
        for key, value, reason in (
            ("general.architecture", "qwen35", "unsupported GGUF vision architecture"),
            ("clip.projector_type", "mlp", "unsupported GGUF vision architecture"),
            ("clip.use_gelu", False, "unsupported GGUF vision architecture"),
            ("clip.vision.image_size", 770, "invalid GGUF vision position grid"),
            (
                "clip.vision.is_deepstack_layers",
                [True] + [False] * 26,
                "deepstack layers are unsupported",
            ),
        ):
            values = vision_fixture()
            values[key] = value
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(models.ModelError, reason),
            ):
                gguf.vision_config(self.metadata(values))

    def derived(self, models_root, path):
        """The metadata derived from a target GGUF, as installations derive
        it, under the lock."""
        models_root.mkdir(parents=True, exist_ok=True)
        with (
            models.installation_lock(models_root),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return assembly.derived_metadata(models_root, {"target/m.gguf": path})

    def test_vision_projector_is_chosen_by_its_header(self):
        tensors = (
            ("v.blk.0.attn_qkv.weight", GGML["BF16"]),
            ("v.patch_embd.weight", GGML["F32"]),
        )

        def projectors(**files):
            root = self.root / "projectors"
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir()
            for name, (values, kinds) in files.items():
                write_gguf(root / (name + ".gguf"), values, kinds)
            return hub.Repository.local_directory(root)

        bf16 = (vision_fixture(), tensors)
        f32 = (vision_fixture(), [(name, GGML["F32"]) for name, _ in tensors])
        f16 = (vision_fixture(), [(tensors[0][0], GGML["F16"]), tensors[1]])
        text = (fixture(), tensors)
        # Other publishers' names; F16 and non-vision files never count.
        repo = projectors(
            **{"mmproj-Model-bf16": bf16, "mmproj-Model-f16": f16, "mmproj-x": text}
        )
        name, header = upstream.select_vision(repo)
        self.assertEqual(name, "mmproj-Model-bf16.gguf")
        self.assertEqual(header.values["general.architecture"], "clip")
        self.assertEqual(
            upstream.select_vision(
                projectors(**{"mmproj-f16": f16, "mmproj-f32": f32})
            )[0],
            "mmproj-f32.gguf",
        )
        with self.assertRaisesRegex(
            models.ModelError,
            r"no BF16 or F32 vision projector \(mmproj-f16.gguf \(clip: F16, F32\); "
            r"mmproj-x.gguf \(qwen35moe: BF16, F32\)\); use --language-only",
        ):
            upstream.select_vision(projectors(**{"mmproj-f16": f16, "mmproj-x": text}))
        with self.assertRaisesRegex(
            models.ModelError, "several BF16 vision projectors"
        ):
            upstream.select_vision(projectors(**{"mmproj-a": bf16, "mmproj-b": bf16}))
        with self.assertRaisesRegex(models.ModelError, "no mmproj"):
            upstream.select_vision(projectors())

    def test_metadata_cache_hit_integrity_and_atomic_failure(self):
        path = write_gguf(self.root / "model.gguf", fixture())
        cache = self.root / "models"
        key, files = self.derived(cache, path)
        self.assertEqual(files["config.json"].parent.name, key)
        expected = files["tokenizer/tokenizer.json"].read_bytes()
        with mock.patch.object(
            gguf, "Metadata", side_effect=AssertionError("reparsed")
        ):
            self.assertEqual(self.derived(cache, path), (key, files))
            # The entry is keyed by content: the same file elsewhere shares it.
            copy = self.root / "elsewhere/model.gguf"
            copy.parent.mkdir()
            shutil.copyfile(path, copy)
            self.assertEqual(self.derived(cache, copy), (key, files))
        # A damaged entry is derived again, not left to block installation.
        for damage in (
            lambda: files["tokenizer/tokenizer.json"].write_text("corrupt"),
            lambda: files["config.json"].unlink(),
            lambda: (files["config.json"].parent / "files.json").write_text("{}"),
        ):
            damage()
            self.assertEqual(self.derived(cache, path), (key, files))
            self.assertEqual(files["tokenizer/tokenizer.json"].read_bytes(), expected)
        with mock.patch.object(
            assembly.os, "rename", side_effect=OSError("interrupted")
        ):
            with self.assertRaises(OSError):
                self.derived(self.root / "interrupted", path)
        self.assertEqual(list((self.root / "interrupted/.metadata").iterdir()), [])
        self.assertTrue(self.derived(self.root / "interrupted", path))

    def test_concurrent_preparation_derives_once_and_follows_the_source(self):
        path = write_gguf(self.root / "model.gguf", fixture())
        cache = self.root / "models"
        with (
            mock.patch.object(
                gguf, "tokenizer_files", wraps=gguf.tokenizer_files
            ) as derive,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            results = list(pool.map(lambda _: self.derived(cache, path), range(2)))
        # The lock serializes installations: the second reuses the entry.
        derive.assert_called_once()
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(list((cache / ".metadata").iterdir())), 1)
        values = fixture()
        values["tokenizer.chat_template"] = "updated template"
        write_gguf(path, values)
        _, changed = self.derived(cache, path)
        self.assertNotEqual(changed, results[0][1])
        self.assertEqual(
            changed["tokenizer/chat_template.jinja"].read_text(), "updated template"
        )

    def gguf_repository(self, *, vision=True):
        """unsloth/Qwen3.6-35B-A3B-GGUF on a FakeHub, with the drafts'
        repository: a loadable target GGUF, an F32 projector and conflicting
        sidecars, which must not override the selected GGUF's metadata."""

        def build(root):
            root.mkdir(parents=True)
            values = fixture(native=True)
            write_gguf(
                root / "model-Q4_K_M.gguf",
                values,
                loadable_tensors(values, self.root).items(),
            )
            if vision:
                write_gguf(
                    root / "mmproj-F32.gguf",
                    vision_fixture(),
                    [("v.patch_embd.weight", GGML["F32"])],
                )
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                (root / name).write_text("invalid sidecar")

        fake = FakeHub(self, self.root / "hub")
        fake.publish(GGUF_REPO, "a" * 40, build)
        fake.publish(MOE.draft.repo, DRAFT_COMMIT, lambda p: draft_dir(p, MOE))
        return fake

    @staticmethod
    def prepare(chosen):
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            upstream.prepare(chosen)
        return output.getvalue()

    def test_gguf_only_repository_assembly_never_resolves_other_target_sources(self):
        fake = self.gguf_repository()
        for language_only in (True, False):
            with self.subTest(language_only=language_only):
                fake.requests.clear()
                chosen = selection(
                    self.root, GGUF_REPO + ":Q4_K_M", language_only=language_only
                )
                output = self.prepare(chosen)
                self.assertIn(f"Selected model-Q4_K_M.gguf from {GGUF_REPO}.", output)
                self.assertEqual(
                    fake.requests,
                    [(GGUF_REPO, None), (MOE.draft.repo, None)],
                )
                assembly.verify(chosen.link, full=True)
                config = models.read_json(chosen.link / "config.json")
                self.assertEqual(config["text_config"]["num_hidden_layers"], 40)
                self.assertEqual("vision_config" in config, not language_only)
                self.assertEqual((chosen.link / "vision").exists(), not language_only)
                self.assertFalse((chosen.link / "processor").exists())
                self.assertEqual(
                    (chosen.link / "tokenizer/config.json").resolve(),
                    (chosen.link / "config.json").resolve(),
                )
        # Only the selected GGUF and its projector are downloaded.
        self.assertEqual(
            sorted(fake.downloads),
            sorted(
                [
                    f"{GGUF_REPO}/mmproj-F32.gguf",
                    f"{GGUF_REPO}/model-Q4_K_M.gguf",
                    f"{MOE.draft.repo}/config.json",
                    f"{MOE.draft.repo}/model.safetensors",
                ]
            ),
        )

    def test_a_layer_count_no_family_has_is_rejected_before_screening(self):
        # Screening lists every tensor of every layer a header claims; only
        # the family bounds the count.
        fake = self.gguf_repository(vision=False)
        values = fixture(native=True)
        values["qwen35moe.block_count"] = 2**32 - 1
        write_gguf(fake.remote / GGUF_REPO / ("a" * 40) / "model-Q4_K_M.gguf", values)
        with (
            mock.patch.object(
                gguf, "loaded_tensors", side_effect=AssertionError("screened")
            ),
            self.assertRaisesRegex(
                models.ModelError, "no supported model has this architecture"
            ),
        ):
            self.prepare(selection(self.root, GGUF_REPO + ":Q4_K_M"))
        self.assertEqual(fake.downloads, [])

    def test_an_unsupported_projector_normalization_is_rejected_before_download(self):
        fake = self.gguf_repository(vision=False)
        values = vision_fixture()
        values["clip.vision.image_mean"] = [0.48, 0.46, 0.41]
        write_gguf(
            fake.remote / GGUF_REPO / ("a" * 40) / "mmproj-F32.gguf",
            values,
            [("v.patch_embd.weight", GGML["F32"])],
        )
        with self.assertRaisesRegex(models.ModelError, "vision preprocessing"):
            self.prepare(
                selection(self.root, GGUF_REPO + ":Q4_K_M", language_only=False)
            )
        self.assertEqual(fake.downloads, [])

    def test_a_new_metadata_adapter_rebuilds_the_metadata_locally(self):
        self.rebuild_metadata_with_a_new_adapter(offline=False)

    def test_offline_a_new_metadata_adapter_rebuilds_the_metadata_locally(self):
        self.rebuild_metadata_with_a_new_adapter(offline=True)

    def rebuild_metadata_with_a_new_adapter(self, *, offline):
        fake = self.gguf_repository(vision=False)
        chosen = selection(self.root, GGUF_REPO + ":Q4_K_M")
        self.prepare(chosen)
        installed = assembly.verify(chosen.link)["metadata"]
        adapter = self.root / "gguf.py"
        adapter.write_text("a new adapter\n")
        fake.requests.clear(), fake.downloads.clear()
        with (
            mock.patch.object(gguf, "__file__", str(adapter)),
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", offline),
        ):
            output = self.prepare(chosen)
        # The unchanged target and draft are assembled again from the cache.
        self.assertEqual(
            fake.requests,
            [] if offline else [(GGUF_REPO, None), (MOE.draft.repo, None)],
        )
        self.assertEqual(fake.downloads, [])
        self.assertIn("the GGUF metadata adapter changed", output)
        rebuilt = assembly.verify(chosen.link)["metadata"]
        self.assertNotEqual(rebuilt, installed)
        self.assertEqual((chosen.link / "config.json").resolve().parent.name, rebuilt)
        # The entry no assembly links any more is removed.
        self.assertEqual(
            [p.name for p in (chosen.models_root / ".metadata").iterdir()], [rebuilt]
        )

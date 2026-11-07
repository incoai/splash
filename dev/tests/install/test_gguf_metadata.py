import contextlib
import io
import shutil
import struct
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from dev.tests import fixture_files
from dev.tests.fixture_files import GGUF_TYPE_ARRAY, GGUF_TYPE_STRING
from dev.tests.installer_fixtures import (
    BYTE,
    DRAFT_COMMIT,
    FLASH_NEXT,
    GGML,
    MOE,
    NORMAL,
    UNKNOWN,
    FakeHub,
    draft_dir,
    family_config,
    fixture,
    selection,
    tensor_table,
    vision_fixture,
    write_gguf,
    write_target,
)
from install import assembly, gguf, hub, models, upstream

GGUF_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"


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
        # Each tensor's extents, innermost first, and its data's offset in the
        # data section, where each tensor starts aligned.
        fixture_files.write_gguf(
            path,
            {},
            [("a", [3, 2], GGML["F32"], bytes(24)), ("b", [4], GGML["F32"], bytes(16))],
        )
        header = gguf.Metadata(path, tensors=True)
        self.assertEqual(header.extents, {"a": (3, 2), "b": (4,)})
        self.assertEqual(header.offsets, {"a": 0, "b": fixture_files.GGUF_ALIGNMENT})
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

    def test_a_target_quantized_in_its_experts_alone_is_served(self):
        # The routed experts hold most of a MoE's values: quantized alone, as
        # in llmfan46's Qwen3.6-35B-A3B NVFP4-Experts-Only, they make a
        # quantized target, here split with every quantized tensor in its
        # second part.
        table = tensor_table()
        parts = [
            gguf.Metadata(
                write_target(
                    self.root / f"model-0000{part + 1}-of-00002.gguf",
                    values,
                    [
                        (name, extents, GGML[kind])
                        for name, extents, _ in table
                        if ("_exps." in name) == bool(part)
                    ],
                ),
                tensors=True,
            )
            for part, (values, kind) in enumerate(((fixture(), "BF16"), ({}, "NVFP4")))
        ]
        gguf.require_quantized(parts)

    def test_every_derivation_names_an_unsupported_architecture(self):
        values = fixture()
        values["general.architecture"] = "llama"
        metadata = self.metadata(values)
        for derive in (gguf.model_config, gguf.tokenizer_files):
            with (
                self.subTest(derive=derive.__name__),
                self.assertRaisesRegex(
                    models.ModelError, "unsupported GGUF model architecture: llama"
                ),
            ):
                derive(metadata)

    def test_the_engine_checks_a_gguf_by_its_config_and_headers(self):
        # The derived config holds the sizes the engine reads in it and a MoE's
        # routed experts, fewer than its family's in a pruned release; the rest
        # of what the planner checks in the GGUF at every start, the engine
        # checks in the copy of its header.
        values = fixture(native=True)
        config = gguf.model_config(self.metadata(values))
        self.assertEqual(
            set(config["text_config"]),
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
            },
        )
        # Qwen3.8-Flash-Next's fixture config, which the native configuration
        # test reads, is the one derived from its fixture header.
        self.assertEqual(
            gguf.model_config(self.metadata(fixture(native=True, family=FLASH_NEXT))),
            family_config(FLASH_NEXT),
        )
        path, headers = self.root / "config.json", self.root / "target"
        headers.mkdir()
        for changes, refusal in (
            ({}, None),
            (
                {"qwen35moe.expert_used_count": 16},
                "GGUF metadata does not match the target: "
                "expert_used_count 16 [(]expected 8",
            ),
        ):
            with self.subTest(changes=changes):
                # Both from one header, as the installer derives them.
                header = write_target(
                    headers / "model.gguf", values | changes, tensor_table()
                )
                metadata = gguf.Metadata(header)
                path.write_bytes(models.json_bytes(gguf.model_config(metadata)))
                if refusal is None:
                    family = upstream.check_model(
                        "gguf", "none", path, gguf_headers=headers
                    )
                    self.assertEqual(family.name, "Qwen3.6-35B-A3B")
                    continue
                with self.assertRaisesRegex(models.ModelError, refusal):
                    upstream.check_model("gguf", "none", path, gguf_headers=headers)

    def test_config_uses_metadata_and_subtracts_only_mtp_layers(self):
        values = fixture()
        values["qwen35moe.block_count"] = 41
        values["qwen35moe.nextn_predict_layers"] = 1
        text = gguf.model_config(self.metadata(values))["text_config"]
        self.assertEqual(text["num_hidden_layers"], 40)
        self.assertEqual(text["hidden_size"], 2048)
        self.assertEqual(text["max_position_embeddings"], 262144)
        for mtp in (41, "1"):
            values["qwen35moe.nextn_predict_layers"] = mtp
            with (
                self.subTest(mtp=mtp),
                self.assertRaisesRegex(
                    models.ModelError, "invalid GGUF MTP layer count"
                ),
            ):
                gguf.model_config(self.metadata(values))

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
            return assembly.derived_metadata(
                models_root, {"target/m.gguf": path}, {path: assembly.file_record(path)}
            )

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
        quantized = (vision_fixture(), [(tensors[0][0], GGML["Q8_0"]), tensors[1]])
        # Other publishers' names; BF16 is preferred to F32 and F32 to F16, and
        # non-vision files never count.
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
        only_f16 = projectors(**{"mmproj-f16": f16, "mmproj-x": text})
        self.assertEqual(upstream.select_vision(only_f16)[0], "mmproj-f16.gguf")
        # Text only, the model is served: the caller says how to ask for it.
        with self.assertRaisesRegex(
            models.VisionError,
            r"no BF16, F32 or F16 vision projector \(mmproj-q8.gguf \(clip: F32, Q8_0\); "
            r"mmproj-x.gguf \(qwen35moe: BF16, F32\)\)$",
        ):
            upstream.select_vision(
                projectors(**{"mmproj-q8": quantized, "mmproj-x": text})
            )
        with self.assertRaisesRegex(
            models.ModelError, "several BF16 vision projectors"
        ):
            upstream.select_vision(projectors(**{"mmproj-a": bf16, "mmproj-b": bf16}))
        # Prism ML prefixes the model's name.
        self.assertEqual(
            upstream.select_vision(
                projectors(**{"Model-mmproj-BF16": bf16, "Model-PQ2_0": text})
            )[0],
            "Model-mmproj-BF16.gguf",
        )
        with self.assertRaisesRegex(models.ModelError, "no GGUF named mmproj"):
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
            write_target(
                root / "model-Q4_K_M.gguf", fixture(native=True), tensor_table()
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
        fake.publish(MOE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, MOE))
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
                    [(GGUF_REPO, None), (MOE.draft_repo, None)],
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
                    f"{MOE.draft_repo}/config.json",
                    f"{MOE.draft_repo}/model.safetensors",
                ]
            ),
        )

    def flash_next_repository(self, repo):
        """A Qwen3.8-Flash-Next GGUF in repo on a FakeHub, of a real file's
        tensors (tensor_table), with the family's DFlash2 draft repository, of
        the fixture's made-up sizes."""

        def build(root):
            root.mkdir(parents=True)
            write_target(
                root / "model-Q2_0.gguf",
                fixture(native=True, family=FLASH_NEXT),
                tensor_table(FLASH_NEXT),
            )

        fake = FakeHub(self, self.root / "hub")
        fake.publish(repo, "a" * 40, build)
        fake.publish(
            FLASH_NEXT.draft_repo,
            DRAFT_COMMIT,
            lambda root: draft_dir(root, FLASH_NEXT),
        )
        return fake

    def test_flash_next_drafts_with_its_released_draft(self):
        repo, draft = "someone/Flash-Next-GGUF", FLASH_NEXT.draft_repo
        fake = self.flash_next_repository(repo)
        chosen = selection(self.root, repo + ":Q2_0")
        output = self.prepare(chosen)
        self.assertIn(f"Selected model-Q2_0.gguf from {repo}.", output)
        self.assertEqual(fake.requests, [(repo, None), (draft, None)])
        # The target and the draft alone are downloaded.
        self.assertEqual(
            sorted(fake.downloads),
            sorted(
                [
                    f"{repo}/model-Q2_0.gguf",
                    f"{draft}/config.json",
                    f"{draft}/model.safetensors",
                ]
            ),
        )
        record = assembly.verify(chosen.link, full=True)
        self.assertEqual(record["family"], FLASH_NEXT.name)
        self.assertEqual(
            record["sources"]["draft"], {"repo": draft, "revision": DRAFT_COMMIT}
        )
        self.assertEqual(
            models.read_json(chosen.link / "draft" / "config.json"),
            family_config(FLASH_NEXT, "draft/config.json"),
        )

    def test_a_gguf_splash_cannot_serve_is_refused_before_its_projector(self):
        # Another architecture, with a projector Splash would not take either.
        fake = FakeHub(self, self.root / "hub")

        def build(root):
            root.mkdir(parents=True)
            write_gguf(root / "model-Q4_K_M.gguf", {"general.architecture": "gemma3"})
            projector = vision_fixture() | {"clip.projector_type": "gemma3"}
            write_gguf(
                root / "mmproj-F16.gguf",
                projector,
                [("v.patch_embd.weight", GGML["F16"])],
            )

        fake.publish("someone/gemma-GGUF", "a" * 40, build)
        with self.assertRaisesRegex(
            models.ModelError, "unsupported GGUF model architecture: gemma3"
        ):
            self.prepare(
                selection(self.root, "someone/gemma-GGUF:Q4_K_M", language_only=False)
            )
        self.assertEqual(fake.range_reads, ["someone/gemma-GGUF/model-Q4_K_M.gguf"])
        self.assertEqual(fake.downloads, [])

    def test_a_header_splash_refuses_is_refused_before_any_download(self):
        # A GGUF whose header states what the planner refuses at every start,
        # or a model unquantized, with its projector: the header is checked
        # before the projector is chosen, and neither a weight file nor the
        # draft is downloaded.
        fake = FakeHub(self, self.root / "hub")
        fake.publish(MOE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, MOE))

        def retyped(kind):
            return [(name, e, kind(name, k)) for name, e, k in tensor_table()]

        unquantized = (
            "^this GGUF stores its weights unquantized [(]BF16, F16 or F32[)]: "
            "Splash loads quantized GGUFs, not whole-BF16 or F16 models; choose "
            "a quantized variant of the repository$"
        )
        for index, (changes, tensors, refusal) in enumerate(
            (
                (
                    {"qwen35moe.expert_used_count": 16},
                    tensor_table(),
                    "GGUF metadata does not match the target: "
                    "expert_used_count 16 [(]expected 8",
                ),
                (
                    {"qwen35moe.rope.scaling.type": "yarn"},
                    tensor_table(),
                    "GGUF metadata does not match the target: rope.scaling.type yarn",
                ),
                (
                    {},
                    retyped(lambda n, k: GGML["Q8_1"] if "ffn_gate_exps" in n else k),
                    "GGUF tensors this build cannot load: "
                    "ffn_gate_exps.weight [(]Q8_1[)] in 40 layers$",
                ),
                ({}, retyped(lambda n, k: GGML["F32"]), unquantized),
                # BF16 with its token table quantized is still a BF16 model.
                (
                    {},
                    retyped(
                        lambda n, k: GGML[
                            "Q8_0" if n == "token_embd.weight" else "BF16"
                        ]
                    ),
                    unquantized,
                ),
            )
        ):
            repo = f"someone/model-{index}-GGUF"

            def build(root, values=fixture(native=True) | changes, tensors=tensors):
                root.mkdir(parents=True)
                write_target(root / "model-Q4_K_M.gguf", values, tensors)
                write_gguf(
                    root / "mmproj-F32.gguf",
                    vision_fixture(),
                    [("v.patch_embd.weight", GGML["F32"])],
                )

            fake.publish(repo, "a" * 40, build)
            fake.range_reads.clear()
            with self.subTest(refusal=refusal):
                with self.assertRaisesRegex(models.ModelError, refusal):
                    self.prepare(
                        selection(self.root, repo + ":Q4_K_M", language_only=False)
                    )
                self.assertEqual(fake.range_reads, [f"{repo}/model-Q4_K_M.gguf"])
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
        # The refusal names the projector, and how to serve text only.
        with self.assertRaisesRegex(
            models.ModelError,
            "vision projector mmproj-F32.gguf: unsupported vision preprocessing "
            "configuration; use --language-only to serve text only",
        ):
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
            [] if offline else [(GGUF_REPO, None), (MOE.draft_repo, None)],
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

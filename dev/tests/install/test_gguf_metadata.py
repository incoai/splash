import contextlib
import dataclasses
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
    prepared_towers,
    selection,
    tensor_table,
    vision_fixture,
    write_gguf,
    write_target,
)
from install import assembly, families, gguf, hub, models, upstream
from install import paths as paths_module

GGUF_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"

FAMILIES = families.FAMILIES  # as defined, whatever a test patches
CHECK_VISION = upstream.check_vision  # the engine's, which tests stand in for


class GgufMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        prepared_towers(self)

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
            for name, file in files.items():
                if isinstance(file, bytes):
                    (root / (name + ".gguf")).write_bytes(file)
                else:
                    write_gguf(root / (name + ".gguf"), *file)
            return hub.Repository.local_directory(root)

        bf16 = (vision_fixture(), tensors)
        f32 = (vision_fixture(), [(name, GGML["F32"]) for name, _ in tensors])
        f16 = (vision_fixture(), [(tensors[0][0], GGML["F16"]), tensors[1]])
        text = (fixture(), tensors)
        quantized = (vision_fixture(), [(tensors[0][0], GGML["Q8_0"]), tensors[1]])
        # Other publishers' names; BF16 is preferred to F32 and F32 to F16, and
        # non-vision files never count.
        qwen, inkling = families.named("Qwen3.8-27B"), families.named("Inkling-2-Mini")
        repo = projectors(
            **{"mmproj-Model-bf16": bf16, "mmproj-Model-f16": f16, "mmproj-x": text}
        )
        name, header = upstream.select_vision(repo, qwen)
        self.assertEqual(name, "mmproj-Model-bf16.gguf")
        self.assertEqual(header.values["general.architecture"], "clip")
        self.assertEqual(
            upstream.select_vision(
                projectors(**{"mmproj-f16": f16, "mmproj-f32": f32}), qwen
            )[0],
            "mmproj-f32.gguf",
        )
        only_f16 = projectors(**{"mmproj-f16": f16, "mmproj-x": text})
        self.assertEqual(upstream.select_vision(only_f16, qwen)[0], "mmproj-f16.gguf")
        # Text only, the model is served: the caller says how to ask for it.
        with self.assertRaisesRegex(
            models.VisionError,
            r"no BF16, F32 or F16 vision projector \(mmproj-q8.gguf \(clip: F32, Q8_0\); "
            r"mmproj-x.gguf \(qwen35moe: BF16, F32\)\)$",
        ):
            upstream.select_vision(
                projectors(**{"mmproj-q8": quantized, "mmproj-x": text}), qwen
            )
        # Inkling's tower takes a Q8_0 projector too, after an exact one.
        both = projectors(**{"mmproj-Model-Q8_0": quantized, "mmproj-Model-BF16": bf16})
        self.assertEqual(
            upstream.select_vision(both, inkling)[0], "mmproj-Model-BF16.gguf"
        )
        self.assertEqual(
            upstream.select_vision(both, qwen)[0], "mmproj-Model-BF16.gguf"
        )
        self.assertEqual(
            upstream.select_vision(
                projectors(**{"mmproj-Model-Q8_0": quantized}), inkling
            )[0],
            "mmproj-Model-Q8_0.gguf",
        )
        with self.assertRaisesRegex(
            models.ModelError, "several BF16 vision projectors"
        ):
            upstream.select_vision(
                projectors(**{"mmproj-a": bf16, "mmproj-b": bf16}), qwen
            )
        # A file named mmproj whose header Splash cannot read is no projector:
        # another is chosen, and alone it leaves the model its text.
        unreadable = b"GGUF" + bytes(20)
        self.assertEqual(
            upstream.select_vision(
                projectors(**{"mmproj-a": unreadable, "mmproj-b": bf16}), qwen
            )[0],
            "mmproj-b.gguf",
        )
        with self.assertRaisesRegex(
            models.VisionError,
            r"no BF16, F32 or F16 vision projector \(mmproj-a.gguf \(.+\)\)$",
        ):
            upstream.select_vision(projectors(**{"mmproj-a": unreadable}), qwen)
        # Prism ML prefixes the model's name.
        self.assertEqual(
            upstream.select_vision(
                projectors(**{"Model-mmproj-BF16": bf16, "Model-PQ2_0": text}), qwen
            )[0],
            "Model-mmproj-BF16.gguf",
        )
        with self.assertRaisesRegex(models.ModelError, "no GGUF named mmproj"):
            upstream.select_vision(projectors(), qwen)

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
        # None of its tensors is 16-bit: --weights auto loads it as the
        # default does, as the engine's check of its headers says.
        self.assertIs(record["quantizable_weights"], False)
        self.assertEqual(
            record["sources"]["draft"], {"repo": draft, "revision": DRAFT_COMMIT}
        )
        self.assertEqual(
            models.read_json(chosen.link / "draft" / "config.json"),
            family_config(FLASH_NEXT, "draft/config.json"),
        )

    def test_a_start_gives_an_older_record_what_weights_auto_changes(self):
        fake = self.gguf_repository()
        chosen = selection(self.root, GGUF_REPO + ":Q4_K_M", language_only=True)
        # An assembly made before 1.7.4 records nothing of --weights.
        build = assembly.build

        def older(models_root, record, files):
            del record["quantizable_weights"]
            return build(models_root, record, files)

        with mock.patch.object(assembly, "build", older):
            self.prepare(chosen)
        self.assertNotIn("quantizable_weights", assembly.verify(chosen.link))
        fake.downloads.clear()
        # The next start assembles it again from the snapshots it links.
        output = self.prepare(chosen)
        self.assertIn("its record gains what the app offers its options by", output)
        self.assertIs(assembly.verify(chosen.link)["quantizable_weights"], False)
        self.assertEqual(fake.downloads, [])
        self.assertIn("is already installed", self.prepare(chosen))

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

    def install_while_its_family_served_text_only(self, fake, revision=None):
        """A selection that asks for images, installed by a release in which
        its target's family served text only (Inkling-2-Mini's before
        1.7.3), with the requests and downloads it made forgotten; from here
        on the family serves images, falling back to text as Inkling 2's
        does (ModelFamily.text_fallback)."""
        chosen = selection(
            self.root, GGUF_REPO + ":Q4_K_M", language_only=False, revision=revision
        )
        self.as_family(vision=False)
        self.prepare(chosen)
        self.assertEqual(assembly.verify(chosen.link)["vision_format"], "none")
        self.assertNotIn(f"{GGUF_REPO}/mmproj-F32.gguf", fake.downloads)
        self.as_family(text_fallback=True)
        fake.requests.clear(), fake.downloads.clear()
        return chosen

    def as_family(self, **changes):
        """The test's family (MOE) with changes to its own definition, in
        place of any made before, until the test ends."""
        patched = tuple(
            dataclasses.replace(family, **changes)
            if family.name == MOE.name
            else family
            for family in FAMILIES
        )
        patcher = mock.patch.object(families, "FAMILIES", patched)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def prepare_warned(chosen):
        """prepare's output and its warnings."""
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()) as warnings,
        ):
            upstream.prepare(chosen)
        return output.getvalue(), warnings.getvalue()

    def test_an_installation_without_its_family_s_tower_takes_it_in(self):
        fake = self.gguf_repository()
        chosen = self.install_while_its_family_served_text_only(fake)
        output, warnings = self.prepare_warned(chosen)
        self.assertIn("its family serves images now", output)
        self.assertIn("vision enabled", output)
        self.assertEqual(warnings, "")
        # The projector alone downloads; the installed GGUF is linked again.
        self.assertEqual(fake.downloads, [f"{GGUF_REPO}/mmproj-F32.gguf"])
        self.assertEqual(assembly.verify(chosen.link)["vision_format"], "gguf")
        output, warnings = self.prepare_warned(chosen)
        self.assertIn("is already installed", output)
        self.assertEqual(warnings, "")

    def test_a_pinned_commit_takes_its_tower_in_too(self):
        # A commit revision resolves without the Hub; the tower needs its
        # listing, which one request gets.
        fake = self.gguf_repository()
        chosen = self.install_while_its_family_served_text_only(fake, "a" * 40)
        output, warnings = self.prepare_warned(chosen)
        self.assertIn("vision enabled", output)
        self.assertEqual(warnings, "")
        self.assertEqual(fake.downloads, [f"{GGUF_REPO}/mmproj-F32.gguf"])
        self.assertEqual(assembly.verify(chosen.link)["vision_format"], "gguf")

    def test_offline_the_newest_cached_snapshot_holding_the_variant_installs(self):
        """The cache's record of the default branch names a newer commit an
        interrupted download left with the projector alone: offline, the
        older snapshot that holds the GGUF installs, named in the log."""
        fake = self.gguf_repository()
        self.prepare(selection(self.root / "first", GGUF_REPO + ":Q4_K_M"))
        repository = self.root / "hub" / hub.folder_name(GGUF_REPO)
        moved = repository / "snapshots" / ("b" * 40)
        moved.mkdir()
        (moved / "mmproj-F32.gguf").write_bytes(b"the projector, fetched first")
        (repository / "refs" / "main").write_text("b" * 40)
        fake.requests.clear()
        chosen = selection(self.root / "second", GGUF_REPO + ":Q4_K_M")
        with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            output = self.prepare(chosen)
            self.assertIn(
                f"The Hub cache's {GGUF_REPO}@{'b' * 12}, its record of the default "
                f"branch, lacks {GGUF_REPO}:Q4_K_M; taking {'a' * 40}, the newest "
                "cached snapshot that holds it.",
                output,
            )
            self.assertEqual(
                assembly.verify(chosen.link)["sources"]["target"]["revision"],
                "a" * 40,
            )
            # A selection that names no variant never takes another
            # snapshot's GGUF: the one it holds is not what the Hub selects.
            with self.assertRaisesRegex(models.ModelError, "no single GGUF|select"):
                self.prepare(selection(self.root / "unnamed", GGUF_REPO))
            # With no snapshot that holds it, the refusal says so.
            (repository / "snapshots" / ("a" * 40) / "model-Q4_K_M.gguf").unlink()
            with self.assertRaisesRegex(
                models.ModelError,
                f"cannot install {GGUF_REPO}:Q4_K_M: could not reach the Hub "
                f"[(]HF_HUB_OFFLINE is set[)], and no cached commit of {GGUF_REPO} "
                f"holds :Q4_K_M whole [(]looked at {'b' * 12}, {'a' * 12}[)]",
            ):
                self.prepare(selection(self.root / "third", GGUF_REPO + ":Q4_K_M"))
        self.assertEqual(fake.requests, [])

    def test_offline_a_snapshot_holds_the_variant_only_with_its_tower(self):
        """The record's snapshot holds the GGUF without its projector, an
        older one both: offline, a selection with vision installs the older
        one; a text-only selection the record's."""
        fake = self.gguf_repository()
        self.prepare(
            selection(self.root / "first", GGUF_REPO + ":Q4_K_M", language_only=False)
        )
        repository = self.root / "hub" / hub.folder_name(GGUF_REPO)
        older = repository / "snapshots" / ("a" * 40)
        moved = repository / "snapshots" / ("b" * 40)
        moved.mkdir()
        (moved / "model-Q4_K_M.gguf").symlink_to(older / "model-Q4_K_M.gguf")
        (repository / "refs" / "main").write_text("b" * 40)
        fake.requests.clear()
        with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            for language_only, commit in ((False, "a" * 40), (True, "b" * 40)):
                with self.subTest(language_only=language_only):
                    chosen = selection(
                        self.root / f"text{language_only}",
                        GGUF_REPO + ":Q4_K_M",
                        language_only=language_only,
                    )
                    self.prepare(chosen)
                    record = assembly.verify(chosen.link)
                    self.assertEqual(record["sources"]["target"]["revision"], commit)
                    self.assertEqual(
                        record["vision_format"] != "none", not language_only
                    )
        self.assertEqual(fake.requests, [])

    def test_offline_an_installation_without_its_tower_serves_text_quietly(self):
        fake = self.gguf_repository()
        chosen = self.install_while_its_family_served_text_only(fake)
        adapter = self.root / "gguf.py"
        adapter.write_text("a new adapter\n")
        with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            output, warnings = self.prepare_warned(chosen)
            self.assertIn("is already installed", output)
            # A new metadata adapter assembles it again, text only, from the
            # snapshots it links, which hold no projector.
            with mock.patch.object(gguf, "__file__", str(adapter)):
                output, rebuilt = self.prepare_warned(chosen)
        self.assertIn("the GGUF metadata adapter changed", output)
        self.assertEqual(warnings + rebuilt, "")
        self.assertEqual(fake.downloads, [])
        record = assembly.verify(chosen.link)
        self.assertEqual(record["vision_format"], "none")
        # The cache held no projector: the Hub is asked again once it answers.
        self.assertNotIn("vision_unavailable", record)
        output, _ = self.prepare_warned(chosen)
        self.assertEqual(assembly.verify(chosen.link)["vision_format"], "gguf")

    def test_a_repository_without_a_projector_goes_on_serving_text(self):
        fake = self.gguf_repository(vision=False)
        chosen = self.install_while_its_family_served_text_only(fake)
        output, warnings = self.prepare_warned(chosen)
        self.assertIn(
            "vision disabled (the GGUF repository has no BF16, F32 or F16 vision "
            "projector",
            output,
        )
        self.assertEqual(warnings, "")
        record = assembly.verify(chosen.link)
        self.assertEqual(record["vision_format"], "none")
        self.assertIn(
            "no BF16, F32 or F16 vision projector", record["vision_unavailable"]
        )
        # Its commit's listing had none: later starts ask nothing more.
        fake.requests.clear()
        output, warnings = self.prepare_warned(chosen)
        self.assertIn("is already installed", output)
        self.assertNotIn("serves images now", output)
        self.assertEqual((warnings, fake.downloads), ("", []))
        self.assertEqual(len(fake.requests), 2)  # the target's and the draft's
        # An update that assembles it again from the snapshots it links, where
        # nothing lists the commit, keeps what the listing said of it.
        adapter = self.root / "gguf.py"
        adapter.write_text("a new adapter\n")
        with mock.patch.object(gguf, "__file__", str(adapter)):
            output, warnings = self.prepare_warned(chosen)
            self.assertIn("the GGUF metadata adapter changed", output)
            self.assertEqual(
                assembly.verify(chosen.link)["vision_unavailable"],
                record["vision_unavailable"],
            )
            output, again = self.prepare_warned(chosen)
        self.assertIn("is already installed", output)
        self.assertNotIn("serves images now", output)
        self.assertEqual((warnings + again, fake.downloads), ("", []))

    def test_a_fresh_installation_without_a_projector_serves_text(self):
        # Of a family that falls back to text: online, the listing's lack is
        # recorded; from the Hub cache alone, it is not, and the Hub is asked
        # once it answers.
        fake = self.gguf_repository(vision=False)
        self.as_family(text_fallback=True)
        chosen = selection(self.root, GGUF_REPO + ":Q4_K_M", language_only=False)
        output, warnings = self.prepare_warned(chosen)
        record = assembly.verify(chosen.link)
        self.assertEqual((record["vision_format"], warnings), ("none", ""))
        self.assertIn("vision_unavailable", record)
        self.assertIn("vision disabled (the GGUF repository has no", output)
        pinned = selection(
            self.root, GGUF_REPO + ":Q4_K_M", language_only=False, revision="a" * 40
        )
        # Another family's such source is refused, naming --language-only.
        self.as_family(text_fallback=False)
        with self.assertRaisesRegex(models.VisionError, "use --language-only"):
            self.prepare(pinned)
        self.as_family(text_fallback=True)
        fake.downloads.clear()
        with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            output, warnings = self.prepare_warned(pinned)
        self.assertIn("from the Hub cache", output)
        self.assertIn("vision disabled (the GGUF repository has no", output)
        record = assembly.verify(pinned.link)
        self.assertEqual((record["vision_format"], warnings), ("none", ""))
        self.assertNotIn("vision_unavailable", record)
        self.assertEqual(fake.downloads, [])

    def test_a_tower_the_engine_refuses_serves_text_where_its_family_does(self):
        # The engine prepares the downloaded tower once (check_vision): one it
        # refuses serves text, its refusal recorded and not asked again.
        self.gguf_repository()
        self.as_family(text_fallback=True)
        refusal = "its vision tower: the vision tower is quantized"
        upstream.check_vision.side_effect = models.VisionError(refusal)
        chosen = selection(self.root, GGUF_REPO + ":Q4_K_M", language_only=False)
        output, warnings = self.prepare_warned(chosen)
        checked, paths, _ = upstream.check_vision.call_args.args
        self.assertEqual(checked.files[assembly.GGUF_VISION], "mmproj-F32.gguf")
        self.assertTrue(paths["mmproj-F32.gguf"].is_file())
        self.assertIn(f"serves text only: {refusal}.", output)
        record = assembly.verify(chosen.link)
        self.assertEqual((record["vision_format"], warnings), ("none", ""))
        self.assertEqual(record["vision_unavailable"], refusal)
        self.assertNotIn(assembly.GGUF_VISION, record["files"])
        upstream.check_vision.reset_mock()
        output, _ = self.prepare_warned(chosen)
        self.assertIn("is already installed", output)
        upstream.check_vision.assert_not_called()
        # Another family's is refused, naming --language-only.
        self.as_family(text_fallback=False)
        pinned = selection(
            self.root, GGUF_REPO + ":Q4_K_M", language_only=False, revision="a" * 40
        )
        with self.assertRaisesRegex(
            models.VisionError, f"{refusal}; use --language-only to serve text only"
        ):
            self.prepare(pinned)
        self.assertFalse(pinned.link.exists())

    def test_an_installation_whose_tower_the_engine_refuses_stays_text(self):
        fake = self.gguf_repository()
        chosen = self.install_while_its_family_served_text_only(fake)
        refusal = "its vision tower: missing source tensor: v.blk.0.ln1.weight"
        upstream.check_vision.side_effect = models.VisionError(refusal)
        output, warnings = self.prepare_warned(chosen)
        self.assertIn("its family serves images now", output)
        self.assertIn(f"serves text only: {refusal}.", output)
        self.assertEqual(warnings, "")
        record = assembly.verify(chosen.link)
        self.assertEqual(record["vision_format"], "none")
        self.assertEqual(record["vision_unavailable"], refusal)
        upstream.check_vision.reset_mock()
        output, warnings = self.prepare_warned(chosen)
        self.assertIn("is already installed", output)
        self.assertEqual(warnings, "")
        upstream.check_vision.assert_not_called()
        # Assembled again from the snapshots it links, which hold the
        # projector, it stays text: the engine is not asked again.
        adapter = self.root / "gguf.py"
        adapter.write_text("a new adapter\n")
        with mock.patch.object(gguf, "__file__", str(adapter)):
            output, warnings = self.prepare_warned(chosen)
            self.assertIn("the GGUF metadata adapter changed", output)
            self.assertEqual(
                assembly.verify(chosen.link)["vision_unavailable"], refusal
            )
            output, again = self.prepare_warned(chosen)
        self.assertIn("is already installed", output)
        self.assertNotIn("serves images now", output)
        self.assertEqual(warnings + again, "")
        upstream.check_vision.assert_not_called()

    def test_the_engine_prepares_a_tower_before_it_is_installed(self):
        # The engine's own check (its vision-check command) refuses the
        # fixture's projector, the header of one tensor.
        fake = self.gguf_repository()
        repo = hub.Repository.resolve(GGUF_REPO)
        scratch = self.root / "scratch"
        scratch.mkdir()
        target = upstream.inspect_target(repo, "Q4_K_M", False, scratch)
        self.assertEqual(target.vision_format, "gguf")
        paths = repo.download(set(target.files.values()))
        with self.assertRaisesRegex(
            models.VisionError, "^its vision tower: GGUF data section is truncated$"
        ):
            CHECK_VISION(target, paths, scratch)
        self.assertIn(f"{GGUF_REPO}/mmproj-F32.gguf", fake.downloads)

    def test_only_the_engine_s_refusal_of_a_tower_refuses_it(self):
        # An engine that refuses the tower (EX_SOFTWARE, EX_DATAERR) refuses
        # it; one that cannot run the check fails the installation instead.
        self.gguf_repository()
        repo = hub.Repository.resolve(GGUF_REPO)
        scratch = self.root / "scratch"
        scratch.mkdir()
        target = upstream.inspect_target(repo, "Q4_K_M", False, scratch)
        paths = repo.download(set(target.files.values()))
        engine = self.root / "engine" / "splash"
        engine.parent.mkdir()
        for status, report, refused in (
            (70, "error: the vision tower is quantized", models.VisionError),
            (65, "error: vision metadata mismatch: x", models.VisionError),
            (74, "error: native runtime I/O failed: cannot map", models.ModelError),
            (64, "error: unknown command vision-check", models.ModelError),
            (-9, "", models.ModelError),
        ):
            stop = f"kill -{-status} $$" if status < 0 else f"exit {status}"
            engine.write_text(f"#!/bin/sh\necho '{report}' >&2\n{stop}\n")
            engine.chmod(0o755)
            with (
                self.subTest(status=status),
                mock.patch.object(paths_module, "BINARY", engine),
                self.assertRaises(models.ModelError) as caught,
            ):
                CHECK_VISION(target, paths, scratch)
            self.assertIs(type(caught.exception), refused)

    def test_a_tower_s_refusal_names_the_target_s_file(self):
        # The engine names the file it read, a link in the check's scratch
        # directory: the refusal, which a record keeps, names the target's.
        self.gguf_repository()
        repo = hub.Repository.resolve(GGUF_REPO)
        scratch = self.root / "scratch"
        scratch.mkdir()
        target = upstream.inspect_target(repo, "Q4_K_M", False, scratch)
        paths = repo.download(set(target.files.values()))
        engine = self.root / "engine" / "splash"
        engine.parent.mkdir()
        engine.write_text(
            "#!/bin/sh\n"
            f'echo "error: the vision tower is quantized (in $5/{assembly.GGUF_VISION})" >&2\n'
            "exit 70\n"
        )
        engine.chmod(0o755)
        with (
            mock.patch.object(paths_module, "BINARY", engine),
            self.assertRaisesRegex(
                models.VisionError,
                r"^its vision tower: the vision tower is quantized "
                r"\(in mmproj-F32\.gguf\)$",
            ),
        ):
            CHECK_VISION(target, paths, scratch)

    def test_an_engine_that_cannot_check_a_tower_s_configuration_fails(self):
        # Only the engine's refusal of a projector's configuration refuses
        # the tower, for a family that then serves text too: an engine that
        # ends otherwise fails the inspection, and nothing is recorded of it.
        self.gguf_repository()
        self.as_family(text_fallback=True)
        repo = hub.Repository.resolve(GGUF_REPO)
        engine = self.root / "engine" / "splash"
        engine.parent.mkdir()
        engine.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = model-check ] && [ "$3" != none ]; then kill -9 $$; fi\n'
            f'exec "{paths_module.BINARY}" "$@"\n'
        )
        engine.chmod(0o755)
        with (
            mock.patch.object(paths_module, "BINARY", engine),
            self.assertRaisesRegex(models.ModelError, "model check failed") as caught,
        ):
            upstream.inspect_target(repo, "Q4_K_M", False, self.root)
        self.assertIs(type(caught.exception), models.ModelError)

    def test_a_tower_that_cannot_download_leaves_the_rest_of_an_update(self):
        fake = self.gguf_repository()
        chosen = self.install_while_its_family_served_text_only(fake)
        installed = assembly.verify(chosen.link)["metadata"]
        adapter = self.root / "gguf.py"
        adapter.write_text("a new adapter\n")
        fake.download_failure = OSError(28, "No space left on device")
        with mock.patch.object(gguf, "__file__", str(adapter)):
            output, warnings = self.prepare_warned(chosen)
        self.assertIn("cannot take in its vision tower", warnings)
        self.assertIn("the GGUF metadata adapter changed", output)
        record = assembly.verify(chosen.link)
        self.assertEqual(record["vision_format"], "none")
        self.assertNotEqual(record["metadata"], installed)
        # A check for updates says the tower it was run for was not taken
        # in, whatever else it assembled again or found unchanged: kept.
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(upstream.prepare(chosen), upstream.KEPT)
            fake.download_failure = None
            self.assertEqual(upstream.prepare(chosen), upstream.UPDATED)
            self.assertEqual(upstream.prepare(chosen), upstream.UP_TO_DATE)
        self.assertEqual(assembly.verify(chosen.link)["vision_format"], "gguf")

    def test_a_text_reassembly_keeps_a_tower_another_start_took_in(self):
        # Two starts read the text installation; one takes the tower in, then
        # the other, offline, assembles its commit again text only.
        fake = self.gguf_repository()
        chosen = self.install_while_its_family_served_text_only(fake)
        stale = assembly.verify(chosen.link)
        self.prepare_warned(chosen)
        self.assertEqual(assembly.verify(chosen.link)["vision_format"], "gguf")
        with contextlib.redirect_stdout(io.StringIO()):
            upstream._install(
                chosen,
                assembly.source(stale, "target"),
                stale,
                language_only=True,
                keep_tower=True,
            )
        self.assertEqual(assembly.verify(chosen.link)["vision_format"], "gguf")

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

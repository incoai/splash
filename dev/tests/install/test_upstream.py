import contextlib
import dataclasses
import errno
import fcntl
import io
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
from huggingface_hub.errors import IncompleteSnapshotError

from dev.tests.installer_fixtures import (
    DENSE,
    DRAFT_COMMIT,
    MODEL,
    MOE,
    PROCESSOR,
    FakeHub,
    cached_snapshot,
    draft_dir,
    fake_hub,
    family_config,
    http_error,
    mlx_target,
    pins,
    selection,
)
from install import assembly, families, hub, models, paths, upstream

# The engine's refusal of a model of no family Splash serves, naming what
# tells it apart.
UNSUPPORTED = (
    "no supported model has this architecture [(]{}[)]; "
    "supported: Qwen3.8-27B, Qwen3.6-35B-A3B"
)


class UpstreamTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "hub"

    @staticmethod
    def prepare(chosen):
        """upstream.prepare's output and warnings."""
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()) as warnings,
        ):
            upstream.prepare(chosen)
        return output.getvalue(), warnings.getvalue()

    def test_every_family_names_its_own_draft_repository(self):
        repos = [family.draft_repo for family in families.FAMILIES]
        for repo in repos:
            with self.subTest(repo=repo):
                self.assertEqual(models.validate_repo_id(repo), repo)
        self.assertEqual(len(set(repos)), len(repos))

    def test_family_is_identified_by_architecture_not_name(self):
        # The engine's check (its model-check command), with and without the
        # draft.
        for family in families.FAMILIES:
            config = mlx_target(self.root / family.name, family) / "config.json"
            draft = draft_dir(self.root / family.name / "draft", family)
            for arguments in ({}, {"draft": draft / "config.json"}):
                self.assertIs(
                    upstream.check_model(
                        "mlx-affine", "safetensors", config, **arguments
                    ),
                    family,
                )
        # A differing size or model type is another model, whatever the
        # repository is called; the engine names it. A field a family's
        # fine-tune could change is refused as a mismatch.
        for changes, refusal in (
            (
                {"num_hidden_layers": 48},
                UNSUPPORTED.format(
                    "text config num_hidden_layers: MLX 48, Qwen3.8-27B 64"
                ),
            ),
            (
                {"max_position_embeddings": 131072},
                UNSUPPORTED.format(
                    "text config max_position_embeddings: MLX 131072, "
                    "Qwen3.8-27B 262144"
                ),
            ),
            (
                {"vocab_size": 151936},
                UNSUPPORTED.format(
                    "text config vocab_size: MLX 151936, Qwen3.8-27B 248320"
                ),
            ),
            (
                {"head_dim": 128},
                UNSUPPORTED.format("text config head_dim: MLX 128, Qwen3.8-27B 256"),
            ),
            (
                {"model_type": "qwen3_moe"},
                UNSUPPORTED.format("text model type qwen3_moe"),
            ),
            ({"num_experts": 128}, "num_experts mismatch: MLX 128, runtime 256"),
        ):
            family = MOE if "num_experts" in changes else DENSE
            root = mlx_target(self.root / "changed", family, changes=changes)
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(models.ModelError, refusal),
            ):
                upstream.check_model("mlx-affine", "none", root / "config.json")
        (self.root / "changed/config.json").write_text(
            json.dumps({"hidden_size": 5120})
        )
        with self.assertRaisesRegex(
            models.ModelError, UNSUPPORTED.format("its config has no text_config")
        ):
            upstream.check_model(
                "mlx-affine", "none", self.root / "changed/config.json"
            )

    def test_a_family_without_a_known_draft_is_refused(self):
        config = mlx_target(self.root / "target", MOE) / "config.json"
        with (
            mock.patch.object(families, "FAMILIES", (DENSE,)),
            self.assertRaisesRegex(
                models.ModelError,
                "no DFlash2 draft is known for the Qwen3.6-35B-A3B family",
            ),
        ):
            upstream.check_model("mlx-affine", "none", config)

    def test_gguf_selection_is_exact_and_ignores_subfolders(self):
        files = {
            "Qwen3.8-27B-UD-Q4_K_M.gguf",
            "Qwen3.8-27B-Q4_0.gguf",
            "Qwen3.8-27B-Q8_0.gguf",
            "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
            "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
            "mmproj-F16.gguf",
            "mmproj-BF16.gguf",
        }
        self.assertEqual(
            upstream.select_gguf(files, "UD-Q4_K_M"),
            ("Qwen3.8-27B-UD-Q4_K_M.gguf", False),
        )
        self.assertEqual(
            upstream.select_gguf(files, "q4_0"), ("Qwen3.8-27B-Q4_0.gguf", False)
        )
        # Q4_K_M names the plain file; without one, the one file ending so,
        # which the caller reports, and never one of several.
        both = files | {"Qwen3.8-27B-Q4_K_M.gguf"}
        self.assertEqual(
            upstream.select_gguf(both, "Q4_K_M"), ("Qwen3.8-27B-Q4_K_M.gguf", False)
        )
        self.assertEqual(
            upstream.select_gguf(files, "Q4_K_M"),
            ("Qwen3.8-27B-UD-Q4_K_M.gguf", True),
        )
        with self.assertRaisesRegex(
            models.ModelError,
            "no single GGUF matches :Q4_K_M .*Qwen3.8-27B-UD-Q4_K_M.gguf, "
            "Qwen3.8-27B-XL-Q4_K_M.gguf",
        ):
            upstream.select_gguf(files | {"Qwen3.8-27B-XL-Q4_K_M.gguf"}, "Q4_K_M")
        # A projector is never a target, however its publisher names it.
        self.assertEqual(
            upstream.select_gguf({"Model-PQ2_0.gguf", "Model-mmproj-BF16.gguf"}, None),
            ("Model-PQ2_0.gguf", False),
        )
        # A repository of one GGUF has no shared name to strip, and needs no
        # variant.
        for variant in ("UD-Q4_K_M", "Q4_K_M", None):
            self.assertEqual(
                upstream.select_gguf({"Qwen3.8-27B-UD-Q4_K_M.gguf"}, variant),
                ("Qwen3.8-27B-UD-Q4_K_M.gguf", False),
            )
        for variant in (None, "Q4", "BF16", "missing"):
            with (
                self.subTest(variant=variant),
                self.assertRaisesRegex(models.ModelError, "Qwen3.8-27B-Q8_0.gguf"),
            ):
                upstream.select_gguf(files, variant)
        # The native loader finds a target by its .gguf extension alone.
        for variant in ("Q4_K_M", None):
            with (
                self.subTest(variant=variant),
                self.assertRaisesRegex(models.ModelError, "repository root: none"),
            ):
                upstream.select_gguf({"Qwen3.8-27B-Q4_K_M.GGUF"}, variant)

    def test_architecture_is_checked_before_weight_downloads(self):
        # Another architecture, and fine-tunes of a supported one the engine
        # refuses at every start, such as one that ropes with YaRN, named by
        # the older `type` key.
        rope = family_config(DENSE)["text_config"]["rope_parameters"]
        del rope["rope_type"]
        fake = FakeHub(self, self.cache)
        for index, (changes, refusal) in enumerate(
            (
                (
                    {"num_hidden_layers": 48},
                    UNSUPPORTED.format(
                        "text config num_hidden_layers: MLX 48, Qwen3.8-27B 64"
                    ),
                ),
                (
                    {"rms_norm_eps": 1e-5},
                    "text config rms_norm_eps mismatch: MLX 1e-05, runtime 1e-06",
                ),
                (
                    {"tie_word_embeddings": True},
                    "text config tie_word_embeddings must be false",
                ),
                (
                    {"rope_parameters": rope | {"type": "yarn"}},
                    "text config rope_parameters type mismatch: MLX yarn, "
                    "runtime default",
                ),
            )
        ):
            model = f"someone/fine-tune-{index}"
            fake.publish(
                model, "a" * 40, lambda p: mlx_target(p, DENSE, changes=changes)
            )
            fake.requests.clear(), fake.downloads.clear()
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(models.ModelError, refusal),
            ):
                self.prepare(selection(self.root, model))
            self.assertEqual(fake.requests, [(model, None)])
            self.assertEqual(fake.downloads, [f"{model}/config.json"])

    def test_only_a_splash_manifest_is_refused_as_a_package(self):
        def target(root):
            mlx_target(root, DENSE)
            (root / "manifest.json").write_text(json.dumps({"name": "a tool's file"}))

        fake = fake_hub(self, self.cache)
        fake.publish(MODEL, "b" * 40, target)
        package = {"format": {"name": "splash-packed-q4"}}
        fake.publish(
            "someone/package",
            "c" * 40,
            lambda p: (
                p.mkdir(parents=True),
                (p / "manifest.json").write_text(json.dumps(package)),
            ),
        )
        self.prepare(selection(self.root))
        self.assertEqual(
            assembly.verify(selection(self.root).link)["sources"]["target"]["revision"],
            "b" * 40,
        )
        with self.assertRaisesRegex(
            models.ModelError,
            "someone/package is a Splash package, which Splash no longer loads; "
            "serve the MLX model of its family instead: splash serve --model "
            "mlx-community/Qwen3.8-27B-4bit",
        ):
            self.prepare(selection(self.root, "someone/package", language_only=False))
        self.assertFalse(selection(self.root, "someone/package").link.exists())

    def test_an_installed_splash_package_is_refused_before_any_request(self):
        # What an earlier release installed: a link to the package's snapshot.
        chosen = selection(self.root, "someone/package", language_only=False)
        snapshot = self.cache / "models--someone--package/snapshots" / ("c" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / "manifest.json").write_text(
            json.dumps({"format": {"name": "splash-packed-q4-moe"}})
        )
        chosen.link.parent.mkdir(parents=True)
        chosen.link.symlink_to(snapshot, target_is_directory=True)
        fake = FakeHub(self, self.cache)
        # The message names the Hub cache folder the package's files take.
        with self.assertRaisesRegex(
            models.ModelError,
            re.escape(
                "someone/package is a Splash package, which Splash no longer loads "
                f"(its files in {(self.cache / 'models--someone--package').resolve()} "
                "can be deleted); serve the MLX model of its family instead: "
                "splash serve --model mlx-community/Qwen3.6-35B-A3B-4bit"
            ),
        ):
            self.prepare(chosen)
        self.assertEqual((fake.requests, fake.downloads), ([], []))

    def test_only_mlx_formats_splash_loads_are_accepted(self):
        def target(quantization):
            def build(root):
                mlx_target(root, DENSE)
                config = json.loads((root / "config.json").read_text())
                del config["quantization"], config["quantization_config"]
                (root / "config.json").write_text(json.dumps(config | quantization))

            return build

        fake = fake_hub(self, self.cache)
        required = (
            r"this model requires an MLX checkpoint \(affine 2, 3, 4, 5, 6 or 8 bits "
            r"in groups of 32, 64 or 128, or mxfp4\) or a supported GGUF"
        )
        formats = (
            r"; MLX weights load as affine 2, 3, 4, 5, 6 or 8 bits in groups of 32, "
            "64 or 128, or as mxfp4"
        )
        for name, quantization, refusal in (
            # A transformers quantization_config alone is another method.
            (
                "gptq",
                {
                    "quantization_config": {
                        "quant_method": "gptq",
                        "bits": 4,
                        "group_size": 64,
                    }
                },
                required,
            ),
            (
                "awq",
                {
                    "quantization_config": {
                        "quant_method": "awq",
                        "bits": 4,
                        "group_size": 64,
                    }
                },
                required,
            ),
            (
                "nvfp4",
                {"quantization": {"mode": "nvfp4", "bits": 4, "group_size": 16}},
                "quantization is nvfp4 4-bit in groups of 16" + formats,
            ),
            (
                "mxfp4-g64",
                {"quantization": {"mode": "mxfp4", "bits": 4, "group_size": 64}},
                "quantization is mxfp4 4-bit in groups of 64" + formats,
            ),
            (
                "q7",
                {"quantization": {"bits": 7, "group_size": 64}},
                "quantization is affine 7-bit in groups of 64" + formats,
            ),
            (
                "q4-g256",
                {
                    "quantization": {
                        "bits": 4,
                        "group_size": 64,
                        "language_model.model.layers.0.mlp.down_proj": {
                            "bits": 4,
                            "group_size": 256,
                        },
                    }
                },
                "quantization language_model.model.layers.0.mlp.down_proj is "
                "affine 4-bit in groups of 256" + formats,
            ),
            # MLX writes false for a module it leaves unquantized.
            (
                "bf16-head",
                {
                    "quantization": {
                        "bits": 4,
                        "group_size": 64,
                        "language_model.lm_head": False,
                    }
                },
                "quantization language_model.lm_head is unquantized; Splash loads "
                "quantized MLX projections and token tables",
            ),
        ):
            with self.subTest(name=name):
                fake.publish(f"someone/{name}", "b" * 40, target(quantization))
                fake.downloads.clear()
                with self.assertRaisesRegex(models.ModelError, refusal):
                    self.prepare(selection(self.root, f"someone/{name}"))
                self.assertEqual(fake.downloads, [f"someone/{name}/config.json"])
        # MLX writes both keys, and states the mode only in newer versions; a
        # module's entry gives it its own format, affine unless it says.
        for name, affine in (
            ("mlx", {"bits": 4, "group_size": 64, "mode": "affine"}),
            ("older-mlx", {"bits": 4, "group_size": 64}),
            ("q8", {"bits": 8, "group_size": 64}),
            ("q3-g32", {"bits": 3, "group_size": 32, "mode": "affine"}),
            ("mxfp4", {"bits": 4, "group_size": 32, "mode": "mxfp4"}),
            (
                "mixed",
                {
                    "bits": 2,
                    "group_size": 128,
                    "mode": "affine",
                    "language_model.model.layers.0.mlp.down_proj": {
                        "bits": 6,
                        "group_size": 64,
                    },
                    "language_model.model.layers.1.mlp.down_proj": {
                        "bits": 4,
                        "group_size": 32,
                        "mode": "mxfp4",
                    },
                },
            ),
        ):
            fake.publish(
                f"someone/{name}",
                "c" * 40,
                target({"quantization": affine, "quantization_config": affine}),
            )
            self.prepare(selection(self.root, f"someone/{name}"))
            assembly.verify(selection(self.root, f"someone/{name}").link)

    def test_a_model_splash_cannot_serve_is_refused_before_its_vision_tower(self):
        # With vision or without, a BF16 release as transformers saves it,
        # its tower named model.visual.*, and a model of another size with an
        # MLX tower, as Qwen3.5-4B is, are refused for what they are, before
        # anything but config.json is fetched.
        def tower(root, prefix):
            (root / "model.safetensors").unlink()
            shards = {
                prefix + "blocks.0.attn.qkv.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": shards})
            )
            for name in set(shards.values()):
                (root / name).write_text(name)
            (root / "preprocessor_config.json").write_text(json.dumps(PROCESSOR))

        def transformers_release(root):
            mlx_target(root, DENSE)
            config = json.loads((root / "config.json").read_text())
            del config["quantization"], config["quantization_config"]
            (root / "config.json").write_text(json.dumps(config))
            tower(root, "model.visual.")

        def other_size(root):
            mlx_target(root, DENSE, changes={"hidden_size": 2560})
            tower(root, "vision_tower.")

        fake = FakeHub(self, self.cache)
        for model, build, refusal in (
            (
                "Qwen/Qwen3.8-27B",
                transformers_release,
                r"this model requires an MLX checkpoint \(affine 2, 3, 4, 5, 6 or 8 "
                r"bits in groups of 32, 64 or 128, or mxfp4\) or a supported GGUF",
            ),
            (
                "mlx-community/Qwen3.5-4B-MLX-4bit",
                other_size,
                UNSUPPORTED.format(
                    "text config hidden_size: MLX 2560, Qwen3.8-27B 5120"
                ),
            ),
        ):
            fake.publish(model, "a" * 40, build)
            fake.downloads.clear()
            with self.subTest(model=model):
                for language_only in (False, True):
                    with self.assertRaisesRegex(models.ModelError, refusal):
                        self.prepare(
                            selection(self.root, model, language_only=language_only)
                        )
                self.assertEqual(fake.downloads, [f"{model}/config.json"])

    def test_an_engine_that_cannot_check_a_model_is_named(self):
        # install/models.py prepare, run by hand, may find no engine, or one
        # built before the check it runs, which refuses the command line as
        # it does any it does not take (EX_USAGE).
        fake_hub(self, self.cache)
        missing, stale = self.root / "missing/splash", self.root / "stale/splash"
        stale.parent.mkdir()
        stale.write_text(
            "#!/bin/sh\n"
            "echo 'error: expected the serve-native command' >&2\n"
            'echo "usage: $0 serve-native MODEL_DIRECTORY" >&2\n'
            "exit 64\n"
        )
        stale.chmod(0o755)
        for engine, packaged, refusal in (
            (
                missing,
                False,
                f"cannot run the engine {missing} (No such file or directory); "
                "rebuild it with make",
            ),
            (
                missing,
                True,
                f"cannot run the engine {missing} (No such file or directory); "
                "reinstall Splash",
            ),
            (
                stale,
                False,
                f"the engine {stale} does not know this model check; "
                "rebuild it with make",
            ),
        ):
            with self.subTest(engine=engine.parent.name, packaged=packaged):
                with (
                    mock.patch.object(paths, "BINARY", engine),
                    mock.patch.object(paths, "PACKAGED", packaged),
                    self.assertRaises(models.ModelError) as refused,
                ):
                    self.prepare(selection(self.root))
                self.assertEqual(str(refused.exception), refusal)

    def test_missing_metadata_never_falls_back_to_another_repository(self):
        required = (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
        )
        fake = FakeHub(self, self.cache)
        for missing in required:
            with self.subTest(missing=missing):
                fake.publish(
                    "user/custom",
                    "a" * 40,
                    lambda p: (
                        p.mkdir(parents=True, exist_ok=True),
                        [
                            (p / name).write_text("{}")
                            for name in ("model.safetensors", *required)
                            if name != missing
                        ],
                    ),
                )
                chosen = selection(self.root, "user/custom", language_only=False)
                with self.assertRaisesRegex(
                    models.ModelError, "must come from the target repository"
                ) as refused:
                    self.prepare(chosen)
                # A text-only checkpoint lacks only the processor.
                self.assertEqual(
                    "use --language-only to serve text only" in str(refused.exception),
                    missing == "preprocessor_config.json",
                )
                self.assertEqual(fake.downloads, [])
                self.assertFalse(chosen.models_root.exists())
                shutil.rmtree(fake.remote)

    def test_source_assembly_pairs_the_draft_by_architecture(self):
        fake = FakeHub(self, self.cache)
        fake.publish(
            "someone/my-favourite-model", "a" * 40, lambda p: mlx_target(p, MOE)
        )
        fake.publish(MOE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, MOE))
        # The name says nothing about the model; the configuration does.
        chosen = selection(self.root, "someone/my-favourite-model")
        self.prepare(chosen)
        self.assertEqual(
            fake.requests,
            [
                ("someone/my-favourite-model", None),
                (MOE.draft_repo, None),
            ],
        )
        record = assembly.verify(chosen.link)
        self.assertEqual(record["family"], MOE.name)
        self.assertEqual(
            record["sources"],
            {
                "target": {"repo": chosen.model, "revision": "a" * 40},
                "draft": {"repo": MOE.draft_repo, "revision": DRAFT_COMMIT},
            },
        )
        self.assertEqual(record["vision_format"], "none")
        self.assertFalse((chosen.link / "manifest.json").exists())
        self.assertFalse((chosen.link / "vision").exists())
        snapshot = fake.snapshot(chosen.model, "a" * 40)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            self.assertEqual(
                (chosen.link / "tokenizer" / name).readlink(), snapshot / name
            )
        # The paths the native loaders and the tokenizer read (assembly.py).
        for name in ("config.json", "target/config.json", "tokenizer/config.json"):
            self.assertEqual((chosen.link / name).readlink(), snapshot / "config.json")
        self.assertEqual(
            (chosen.link / "draft/model.safetensors").read_bytes(), b"draft"
        )
        (snapshot / "tokenizer.json").write_text("changed size")
        with self.assertRaises(models.ModelError):
            assembly.verify(chosen.link)

    def test_mlx_vision_links_only_the_shards_holding_the_tower(self):
        shards = {
            "vision_tower.blocks.0.attn.qkv.weight": "model-00001-of-00002.safetensors",
            "language_model.model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "language_model.lm_head.weight": "model-00002-of-00002.safetensors",
        }

        def target(root):
            mlx_target(root, DENSE)
            (root / "model.safetensors").unlink()
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": shards})
            )
            for name in set(shards.values()):
                (root / name).write_text(name)
            (root / "preprocessor_config.json").write_text(json.dumps(PROCESSOR))

        fake = fake_hub(self, self.cache)
        fake.publish(MODEL, "a" * 40, target)
        chosen = selection(self.root, language_only=False)
        self.prepare(chosen)
        self.assertEqual(assembly.verify(chosen.link)["vision_format"], "safetensors")
        self.assertFalse((chosen.link / "processor").exists())
        self.assertEqual(
            sorted(p.name for p in (chosen.link / "vision").iterdir()),
            ["config.json", "model-00001-of-00002.safetensors"],
        )
        self.assertEqual(
            sorted(p.name for p in (chosen.link / "target").iterdir()),
            ["config.json", *sorted(set(shards.values()))],
        )
        # A checkpoint without the tower cannot serve images, nor can a
        # text-only conversion that keeps the processor but drops the tower's
        # configuration too: its text configuration is checked first.
        del shards["vision_tower.blocks.0.attn.qkv.weight"]
        fake.publish("someone/text-model", "b" * 40, target)

        def text_conversion(root):
            target(root)
            config = json.loads((root / "config.json").read_text())
            del config["vision_config"]
            (root / "config.json").write_text(json.dumps(config))

        fake.publish("someone/text-conversion", "c" * 40, text_conversion)
        for model in ("someone/text-model", "someone/text-conversion"):
            with (
                self.subTest(model=model),
                self.assertRaisesRegex(
                    models.ModelError, "no vision tower; use --language-only"
                ),
            ):
                self.prepare(selection(self.root, model, language_only=False))

    def test_a_tower_in_a_shard_of_its_own_is_not_the_targets(self):
        # OptiQ keeps the tower in a subdirectory shard: the target links the
        # text shards alone, and only serving images needs the tower's shard.
        shards = {
            "vision_tower.blocks.0.attn.qkv.weight": "optiq/optiq_vision.safetensors",
            "language_model.model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "language_model.lm_head.weight": "model-00002-of-00002.safetensors",
        }

        def target(root):
            mlx_target(root, DENSE)
            (root / "model.safetensors").unlink()
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": shards})
            )
            for name in set(shards.values()):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(name)
            (root / "preprocessor_config.json").write_text(json.dumps(PROCESSOR))

        fake = fake_hub(self, self.cache)
        fake.publish(MODEL, "a" * 40, target)
        chosen = selection(self.root)
        self.prepare(chosen)
        self.assertEqual(
            sorted(p.name for p in (chosen.link / "target").iterdir()),
            [
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
        )
        self.assertNotIn(f"{MODEL}/optiq/optiq_vision.safetensors", fake.downloads)
        with self.assertRaisesRegex(models.ModelError, "missing or unsupported shards"):
            self.prepare(selection(self.root, language_only=False))

    def test_a_shard_name_read_as_a_glob_is_never_downloaded(self):
        def target(root):
            mlx_target(root, DENSE)
            (root / "model.safetensors").unlink()
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"lm_head.weight": "model-*.safetensors"}})
            )
            (root / "model-*.safetensors").write_text("{}")

        fake = fake_hub(self, self.cache)
        fake.publish(MODEL, "a" * 40, target)
        with self.assertRaisesRegex(models.ModelError, "unsupported file name"):
            self.prepare(selection(self.root))
        self.assertNotIn(f"{MODEL}/model-*.safetensors", fake.downloads)

    def test_a_single_checkpoint_file_is_searched_for_the_tower_by_header(self):
        def checkpoint(tensors):
            def build(root):
                mlx_target(root, DENSE)
                header = json.dumps(
                    {
                        t: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}
                        for t in tensors
                    }
                    | {"__metadata__": {"format": "mlx"}}
                ).encode()
                (root / "model.safetensors").write_bytes(
                    len(header).to_bytes(8, "little") + header + b"\0"
                )
                (root / "preprocessor_config.json").write_text(json.dumps(PROCESSOR))

            return build

        fake = fake_hub(self, self.cache)
        fake.publish(
            MODEL,
            "b" * 40,
            checkpoint(
                ["vision_tower.patch_embed.weight", "language_model.lm_head.weight"]
            ),
        )
        chosen = selection(self.root, language_only=False)
        self.prepare(chosen)
        self.assertEqual(
            sorted(p.name for p in (chosen.link / "vision").iterdir()),
            ["config.json", "model.safetensors"],
        )
        fake.publish(
            "someone/text", "c" * 40, checkpoint(["language_model.lm_head.weight"])
        )
        with self.assertRaisesRegex(models.ModelError, "no vision tower"):
            self.prepare(selection(self.root, "someone/text", language_only=False))
        # Its header was read by range requests, not a download.
        self.assertNotIn("someone/text/model.safetensors", fake.downloads)
        self.assertIn("someone/text/model.safetensors", fake.range_reads)

    def test_a_cached_file_is_read_from_the_cache(self):
        fake = FakeHub(self, self.cache)
        fake.publish(MODEL, "a" * 40, lambda p: mlx_target(p, DENSE))
        repo = hub.Repository.resolve(MODEL)
        repo.download({"config.json"})
        with repo.open("config.json") as stream:
            self.assertEqual(json.load(stream)["quantization"]["bits"], 4)
        with repo.open("tokenizer.json") as stream:
            self.assertEqual(stream.read(), b"{}")
        self.assertEqual(fake.range_reads, [f"{MODEL}/tokenizer.json"])

    def test_unchanged_commits_start_with_a_request_each_and_no_download(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        fake.requests.clear(), fake.downloads.clear()
        output, _ = self.prepare(chosen)
        self.assertEqual(fake.requests, [(MODEL, None), (DENSE.draft_repo, None)])
        self.assertEqual(fake.downloads, [])
        self.assertIn("is already installed", output)

    def test_moved_commit_is_installed_and_published_atomically(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        output, _ = self.prepare(chosen)
        self.assertIn("Fetching 4 file(s), 0.00 GB, from " + MODEL, output)
        old = chosen.link.resolve()
        fake.publish(MODEL, "b" * 40, lambda p: mlx_target(p, DENSE))
        fake.requests.clear(), fake.downloads.clear()
        rename = assembly.os.rename

        def publish(stage, destination):
            # The old assembly stays in use until the new one is complete.
            self.assertEqual(chosen.link.resolve(), old)
            rename(stage, destination)

        with mock.patch.object(assembly.os, "rename", side_effect=publish):
            output, _ = self.prepare(chosen)
        self.assertIn(f"{MODEL} moved from {'a' * 12} to {'b' * 12}.", output)
        self.assertEqual(
            assembly.verify(chosen.link)["sources"]["target"]["revision"], "b" * 40
        )
        self.assertEqual(fake.requests, [(MODEL, None), (DENSE.draft_repo, None)])
        self.assertNotIn(f"{DENSE.draft_repo}/model.safetensors", fake.downloads)
        self.assertEqual(pins(self.cache), sorted(["b" * 40, DRAFT_COMMIT]))

    def test_publishing_removes_what_no_installation_uses(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        first = chosen.link.resolve()
        # A server holds the assembly it serves.
        _, server = assembly.hold(chosen.link, chosen.models_root)
        self.addCleanup(server.close)
        # Staging an interrupted installation left behind.
        stale = [
            chosen.models_root / ".resolved/.loading-x",
            chosen.models_root / ".metadata/.loading-y",
            chosen.models_root / "mlx-community/.prepare-Qwen3.8-27B-4bit-z",
            chosen.models_root / ".selections/.prepare-w",
        ]
        for path in stale:
            path.mkdir(parents=True)
        fake.publish(MODEL, "b" * 40, lambda p: mlx_target(p, DENSE))
        self.prepare(chosen)
        second = chosen.link.resolve()
        self.assertEqual(
            sorted(p.name for p in (chosen.models_root / ".resolved").iterdir()),
            sorted([first.name, second.name]),
        )
        self.assertFalse(any(path.exists() for path in stale))
        # Released by its server and linked by no selection, it goes next.
        server.close()
        fake.publish(MODEL, "c" * 40, lambda p: mlx_target(p, DENSE))
        self.prepare(chosen)
        self.assertEqual(
            [p.name for p in (chosen.models_root / ".resolved").iterdir()],
            [chosen.link.resolve().name],
        )
        assembly.verify(chosen.link)

    def test_unreachable_hub_starts_the_installed_assembly(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        for failure in (
            httpx.ConnectError("[Errno 8] nodename nor servname provided"),
            httpx.ConnectTimeout(""),
            http_error(500),
            http_error(401),
            http_error(403),
            http_error(404),
        ):
            with self.subTest(failure=failure):
                fake.failure = failure
                fake.requests.clear()
                output, _ = self.prepare(chosen)
                reason = hub.reason(failure)
                self.assertNotIn("\n", reason)
                self.assertIn(
                    f"Could not reach the Hub ({reason}); "
                    f"using the installed {MODEL}@{'a' * 12}.\n",
                    output,
                )
                self.assertIn("is already installed", output)
                # A Hub that did not answer for the target is not asked for the
                # draft.
                self.assertEqual(fake.requests, [(MODEL, None)])

    def test_unreachable_hub_without_an_installation_is_the_error(self):
        fake = fake_hub(self, self.cache)
        fake.failure = http_error(404)
        with self.assertRaisesRegex(
            models.ModelError,
            "cannot resolve someone/other: 404 Client Error.*; neither this "
            "installation nor the Hub cache records a commit for the default branch",
        ):
            self.prepare(selection(self.root, "someone/other"))

    def test_commit_and_offline_selections_make_no_request(self):
        fake = fake_hub(self, self.cache)
        pinned = selection(self.root, revision="a" * 40)
        self.prepare(pinned)
        self.prepare(selection(self.root))
        fake.requests.clear(), fake.downloads.clear()
        output, _ = self.prepare(pinned)
        self.assertIn("is already installed", output)
        with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            output, _ = self.prepare(selection(self.root))
        self.assertIn("is already installed", output)
        self.assertEqual((fake.requests, fake.downloads), ([], []))

    def test_new_commit_rejected_before_download_keeps_the_installed_one(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        installed = chosen.link.resolve()
        fake.publish(
            MODEL, "b" * 40, lambda p: mlx_target(p, DENSE, changes={"head_dim": 128})
        )
        _, warnings = self.prepare(chosen)
        self.assertIn(
            f"Warning: keeping the installed {MODEL}@{'a' * 12}; cannot install "
            f"{MODEL}@{'b' * 40}: no supported model has this architecture (text "
            "config head_dim: MLX 128, Qwen3.8-27B 256); supported: Qwen3.8-27B, "
            "Qwen3.6-35B-A3B",
            warnings,
        )
        self.assertEqual(chosen.link.resolve(), installed)

    def test_new_commit_whose_download_fails_keeps_the_installed_one(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        installed = chosen.link.resolve()
        fake.publish(MODEL, "c" * 40, lambda p: mlx_target(p, DENSE))
        fake.download_failure = httpx.ReadTimeout("timed out")
        _, warnings = self.prepare(chosen)
        self.assertIn(
            f"cannot install {MODEL}@{'c' * 40}: cannot install {MODEL}: timed out",
            warnings,
        )
        self.assertEqual(chosen.link.resolve(), installed)
        self.assertEqual(pins(self.cache), sorted(["a" * 40, DRAFT_COMMIT]))
        # Nothing installed: the rejection is the error.
        with self.assertRaisesRegex(models.ModelError, "timed out"):
            self.prepare(selection(self.root, revision="c" * 40))

    def test_a_moved_draft_reassembles_the_installed_target(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        fake.publish(DENSE.draft_repo, "e" * 40, lambda p: draft_dir(p, DENSE))
        fake.requests.clear(), fake.downloads.clear()
        output, _ = self.prepare(chosen)
        self.assertIn(
            f"{DENSE.draft_repo} moved from {DRAFT_COMMIT[:12]} to {'e' * 12}", output
        )
        self.assertEqual(
            assembly.verify(chosen.link)["sources"],
            {
                "target": {"repo": MODEL, "revision": "a" * 40},
                "draft": {"repo": DENSE.draft_repo, "revision": "e" * 40},
            },
        )
        # Only the new draft is fetched; the target is not downloaded again.
        self.assertEqual(fake.requests, [(MODEL, None), (DENSE.draft_repo, None)])
        self.assertTrue(
            all(name.startswith(DENSE.draft_repo + "/") for name in fake.downloads)
        )
        self.assertEqual(pins(self.cache), sorted(["a" * 40, "e" * 40]))

    def test_an_installation_of_another_draft_repository_moves_to_the_familys(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        other = dataclasses.replace(DENSE, draft_repo="someone/other-draft")
        fake.publish(other.draft_repo, "c" * 40, lambda p: draft_dir(p, DENSE))
        with mock.patch.object(families, "FAMILIES", (other, MOE)):
            self.prepare(chosen)
        # Another installation pins the same draft.
        hub.pin(
            fake.snapshot(other.draft_repo, "c" * 40),
            other.draft_repo,
            chosen.models_root / "someone/other",
        )
        fake.requests.clear()
        output, _ = self.prepare(chosen)
        self.assertIn(
            f"its draft is now {DENSE.draft_repo}@{DRAFT_COMMIT[:12]}", output
        )
        self.assertEqual(
            assembly.verify(chosen.link)["sources"]["draft"],
            {"repo": DENSE.draft_repo, "revision": DRAFT_COMMIT},
        )
        self.assertEqual(fake.requests, [(MODEL, None), (DENSE.draft_repo, None)])
        # Its pin of the repository it no longer links is retired too; the
        # other installation's pin stays.
        self.assertEqual(pins(self.cache), sorted(["a" * 40, "c" * 40, DRAFT_COMMIT]))

    def test_a_draft_the_hub_cannot_resolve_keeps_the_installed_one(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        # The draft's main names a commit the Hub cannot serve.
        fake.branches[DENSE.draft_repo, "main"] = "f" * 40
        fake.downloads.clear()
        output, _ = self.prepare(chosen)
        self.assertIn("Could not reach the Hub (404 Client Error", output)
        self.assertIn(
            f"using the installed draft {DENSE.draft_repo}@{DRAFT_COMMIT[:12]}.", output
        )
        self.assertIn("is already installed", output)
        self.assertEqual(
            assembly.verify(chosen.link)["sources"]["draft"]["revision"],
            DRAFT_COMMIT,
        )
        self.assertEqual(fake.downloads, [])

    def move_target_and_draft(self, build_draft, failing_revision=None):
        """Install MODEL, then start with MODEL moved to b*40 and DENSE's draft
        repository moved to e*40, built by build_draft; fetches of
        failing_revision fail. The warnings of that start."""
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        fake.publish(MODEL, "b" * 40, lambda p: mlx_target(p, DENSE))
        fake.publish(DENSE.draft_repo, "e" * 40, build_draft)
        fetch = fake.fetch

        def fetch_or_fail(repo_id, name, revision):
            if revision == failing_revision:
                raise OSError(errno.ECONNRESET, "connection reset by peer")
            return fetch(repo_id, name, revision)

        with mock.patch.object(fake, "fetch", side_effect=fetch_or_fail):
            _, warnings = self.prepare(chosen)
        # The new target is installed with the installed draft.
        self.assertEqual(
            assembly.verify(chosen.link)["sources"],
            {
                "target": {"repo": MODEL, "revision": "b" * 40},
                "draft": {"repo": DENSE.draft_repo, "revision": DRAFT_COMMIT},
            },
        )
        self.assertEqual(pins(self.cache), sorted(["b" * 40, DRAFT_COMMIT]))
        return warnings

    def test_a_new_draft_whose_download_fails_keeps_the_installed_draft(self):
        warnings = self.move_target_and_draft(
            lambda p: draft_dir(p, DENSE), failing_revision="e" * 40
        )
        self.assertIn(
            f"Warning: cannot use the {DENSE.name} draft {DENSE.draft_repo}@"
            f"{'e' * 12}; keeping the installed one: cannot fetch the {DENSE.name} "
            f"draft: [Errno {errno.ECONNRESET}] connection reset by peer",
            warnings,
        )

    def test_a_new_draft_without_weights_keeps_the_installed_draft(self):
        def incomplete(root):
            draft_dir(root, DENSE)
            (root / "model.safetensors").unlink()

        warnings = self.move_target_and_draft(incomplete)
        self.assertIn(
            f"Warning: cannot use the {DENSE.name} draft {DENSE.draft_repo}@"
            f"{'e' * 12}; keeping the installed one: {DENSE.draft_repo} does not "
            f"contain a DFlash2 checkpoint for {DENSE.name}",
            warnings,
        )

    def test_a_new_draft_of_another_architecture_keeps_the_installed_draft(self):
        warnings = self.move_target_and_draft(
            lambda p: draft_dir(p, DENSE, **{"dflash_config.block_size": 16})
        )
        self.assertIn(
            f"Warning: cannot use the {DENSE.name} draft {DENSE.draft_repo}@"
            f"{'e' * 12}; keeping the installed one: draft configuration is "
            f"incompatible with {DENSE.name}: draft config dflash_config block_size "
            "mismatch: checkpoint 16, runtime 8",
            warnings,
        )

    def test_a_draft_the_engine_refuses_is_never_downloaded(self):
        # JSON's true equals 1, the first layer this draft reads, in Python;
        # the engine takes no boolean for a number.
        layers = [True, 6, 11, 16, 22, 27, 32, 37]
        fake = FakeHub(self, self.cache)
        fake.publish(MODEL, "a" * 40, lambda p: mlx_target(p, MOE))
        fake.publish(
            MOE.draft_repo,
            DRAFT_COMMIT,
            lambda p: draft_dir(p, MOE, **{"dflash_config.target_layer_ids": layers}),
        )
        with self.assertRaisesRegex(
            models.ModelError,
            f"draft configuration is incompatible with {MOE.name}: draft config "
            "target_layer_ids 0 must be a number",
        ):
            self.prepare(selection(self.root))
        self.assertEqual(
            fake.downloads, [f"{MODEL}/config.json", f"{MOE.draft_repo}/config.json"]
        )

    def test_a_moved_hub_cache_starts_the_installation_it_links(self):
        # HF_HUB_CACHE may name another folder than the one an installation
        # was built from; its links still name its files there.
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        pinned = selection(self.root, revision="a" * 40)
        self.prepare(chosen)
        self.prepare(pinned)
        installed = pins(self.cache)
        moved = self.root / "moved"
        moved.mkdir()
        fake.requests.clear(), fake.downloads.clear()
        for name, start, requests in (
            (
                "the Hub answers",
                lambda: self.prepare(chosen),
                [(MODEL, None), (DENSE.draft_repo, None)],
            ),
            ("a commit revision", lambda: self.prepare(pinned), []),
            ("offline", lambda: self.prepare(chosen), []),
        ):
            with (
                self.subTest(start=name),
                mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(moved)),
                mock.patch(
                    "huggingface_hub.constants.HF_HUB_OFFLINE", name == "offline"
                ),
            ):
                output, _ = start()
                self.assertIn("is already installed", output)
                self.assertEqual(fake.requests, requests)
                self.assertEqual(fake.downloads, [])
                fake.requests.clear()
        self.assertEqual(pins(self.cache), installed)
        self.assertEqual(list(moved.iterdir()), [])

    def test_a_moved_hub_cache_and_an_unreachable_hub_start_the_installation(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        installed = pins(self.cache)
        moved = self.root / "moved"
        moved.mkdir()
        fake.failure = httpx.ConnectError("[Errno 8] nodename nor servname provided")
        fake.downloads.clear()
        with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(moved)):
            output, _ = self.prepare(chosen)
        self.assertIn(
            f"Could not reach the Hub ({hub.reason(fake.failure)}); "
            f"using the installed {MODEL}@{'a' * 12}.\n",
            output,
        )
        self.assertIn("is already installed", output)
        self.assertEqual(fake.downloads, [])
        self.assertEqual(pins(self.cache), installed)
        self.assertEqual(list(moved.iterdir()), [])

    def test_a_moved_hub_cache_keeps_the_installation_an_update_needs(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        installed = chosen.link.resolve()
        moved = self.root / "moved"
        moved.mkdir()
        fake.publish(DENSE.draft_repo, "e" * 40, lambda p: draft_dir(p, DENSE))
        fake.downloads.clear()
        with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(moved)):
            _, warnings = self.prepare(chosen)
        self.assertIn(
            f"Warning: keeping the installed {MODEL}@{'a' * 12}; cannot update it "
            f"({DENSE.draft_repo} moved from {DRAFT_COMMIT[:12]} to {'e' * 12}): "
            f"the Hub cache has no snapshot {'a' * 40} of {MODEL}",
            warnings,
        )
        self.assertEqual(chosen.link.resolve(), installed)
        self.assertEqual(fake.downloads, [])
        self.assertEqual(list(moved.iterdir()), [])

    def test_an_incompatible_draft_is_an_error_not_a_crash(self):
        fake_hub(self, self.cache)
        local = draft_dir(self.root / "draft", DENSE, num_hidden_layers=6)
        with self.assertRaisesRegex(
            models.ModelError,
            "draft configuration is incompatible with Qwen3.8-27B: draft config "
            "num_hidden_layers mismatch: checkpoint 6, runtime 5",
        ):
            self.prepare(selection(self.root, draft_model=str(local)))

    def test_hub_snapshots_are_pinned_and_old_pins_retired(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        for commit in ("a" * 40, "b" * 40):
            fake.publish(MODEL, commit, lambda p: mlx_target(p, DENSE))
            self.prepare(chosen)
            refs = sorted(
                (
                    self.cache / "models--mlx-community--Qwen3.8-27B-4bit/refs/splash"
                ).glob("*/*")
            )
            self.assertEqual([ref.name for ref in refs], [commit])
            draft_refs = sorted(
                (self.cache / hub.folder_name(DENSE.draft_repo) / "refs/splash").glob(
                    "*/*"
                )
            )
            self.assertEqual([ref.name for ref in draft_refs], [DRAFT_COMMIT])
        self.assertEqual(refs[0].parent.name, draft_refs[0].parent.name)
        self.assertEqual(
            assembly.verify(chosen.link)["sources"]["target"]["revision"], "b" * 40
        )

    def test_pins_are_required_before_publishing(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        with (
            mock.patch.object(
                hub, "pin", side_effect=PermissionError(errno.EACCES, "no")
            ),
            self.assertRaises(PermissionError),
        ):
            self.prepare(chosen)
        self.assertFalse(chosen.link.exists())

    def test_pins_change_under_the_lock_and_are_repaired_on_start(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        expected = sorted(["a" * 40, DRAFT_COMMIT])
        pin = hub.pin

        def locked(*arguments):
            with (chosen.models_root / ".install.lock").open("a+b") as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return pin(*arguments)

        with mock.patch.object(hub, "pin", side_effect=locked) as pinned:
            self.prepare(chosen)
        self.assertEqual(pinned.call_count, 2)
        self.assertEqual(pins(self.cache), expected)
        # A verified start restores lost pins.
        for ref in self.cache.glob("*/refs/splash/*/*"):
            ref.unlink()
        with mock.patch.object(hub, "pin", side_effect=locked):
            self.prepare(chosen)
        self.assertEqual(pins(self.cache), expected)

    def test_a_read_only_cache_leaves_the_installation_usable(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        for ref in self.cache.glob("*/refs/splash/*/*"):
            ref.unlink()
        with mock.patch.object(
            hub.os, "replace", side_effect=OSError(errno.EROFS, "read only")
        ):
            output, warnings = self.prepare(chosen)
        self.assertIn("is already installed", output)
        self.assertIn("external cache pruning", warnings)
        self.assertEqual(pins(self.cache), [])

    def test_a_damaged_assembly_and_an_unreachable_hub_cost_one_request(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        built = next((chosen.models_root / ".resolved").iterdir())
        (built / "tokenizer/tokenizer.json").unlink()
        fake.failure = httpx.ConnectTimeout("")
        fake.requests.clear()
        output, _ = self.prepare(chosen)
        self.assertIn("Reinstalling " + MODEL, output)
        # The draft's repository is not asked when the Hub did not answer for
        # the target: the commit the Hub cache holds for it stands in.
        self.assertEqual(fake.requests, [(MODEL, None)])
        self.assertEqual(
            assembly.verify(chosen.link)["sources"]["draft"],
            {"repo": DENSE.draft_repo, "revision": DRAFT_COMMIT},
        )

    def test_damaged_assembly_is_rebuilt(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        built = next((chosen.models_root / ".resolved").iterdir())
        (built / "tokenizer/tokenizer.json").unlink()
        fake.downloads.clear()
        output, _ = self.prepare(chosen)
        self.assertIn("Reinstalling " + MODEL, output)
        self.assertIn(f"Rebuilding the damaged {built}", output)
        assembly.verify(built)
        # The cached snapshot is complete; nothing is downloaded again.
        self.assertNotIn(f"{MODEL}/model.safetensors", fake.downloads)

    def test_verify_checks_full_content_and_rejects_same_size_changes(self):
        source = self.root / "source"
        source.write_bytes(b"abcd")
        built = self.root / "assembly"
        built.mkdir()
        (built / "weight").symlink_to(source)
        local = {"repo": str(self.root), "revision": None}
        record = {
            "version": 1,
            "model": MODEL,
            "family": DENSE.name,
            "target_format": "mlx-affine",
            "vision_format": "none",
            "sources": {"target": local, "draft": local},
            "files": {"weight": assembly.file_record(source)},
        }
        (built / "model.json").write_text(json.dumps(record))
        assembly.verify(built, full=True)
        source.write_bytes(b"abce")
        with self.assertRaises(models.ModelError):
            assembly.verify(built)
        stat = source.stat()
        record["files"]["weight"].update(
            mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns
        )
        (built / "model.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(models.ModelError, "hash mismatch"):
            assembly.verify(built, full=True)

    def test_verify_rejects_a_record_of_another_shape(self):
        fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        record = json.loads((chosen.link / "model.json").read_text())
        for change in (
            lambda r: r.pop("family"),
            lambda r: r.update(metadata="0" * 64),
            lambda r: r.update(target_format="safetensors"),
            lambda r: r.update(version=True),
            lambda r: r["sources"]["draft"].update(revision="main"),
            lambda r: r["files"]["config.json"].pop("mtime_ns"),
            lambda r: r["files"]["config.json"].update(digest="z" * 64),
            lambda r: r["files"].update({"../escape": r["files"]["config.json"]}),
            lambda r: r["files"].update({".": r["files"]["config.json"]}),
        ):
            changed = json.loads(json.dumps(record))
            change(changed)
            built = self.root / "copy"
            shutil.rmtree(built, ignore_errors=True)
            shutil.copytree(chosen.link, built, symlinks=True)
            (built / "model.json").unlink()
            (built / "model.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(models.ModelError, "invalid resolved model"):
                assembly.verify(built)

    def test_hub_resolution_pins_one_commit(self):
        fake = FakeHub(self, self.cache)
        fake.publish(MODEL, "a" * 40, lambda p: mlx_target(p, DENSE), branch="v2")
        repo = hub.Repository.resolve(MODEL, "v2")
        self.assertEqual(fake.requests, [(MODEL, "v2")])
        self.assertEqual(repo.revision, "a" * 40)
        self.assertEqual(
            repo.file("config.json"), fake.snapshot(MODEL, "a" * 40) / "config.json"
        )
        self.assertEqual(
            set(repo.download({"model.safetensors"})), {"model.safetensors"}
        )
        fake.publish(MODEL, "b" * 40, lambda p: mlx_target(p, DENSE), branch="v2")
        # The branch moved; the resolved repository still reads its commit.
        self.assertEqual(
            json.loads(repo.file("config.json").read_text())["quantization"]["bits"], 4
        )
        self.assertEqual(repo.revision, "a" * 40)

    def test_only_an_absolute_path_is_a_local_repository(self):
        # A relative path in the working directory is still a Hub repository ID.
        (self.root / MODEL).mkdir(parents=True)
        fake = FakeHub(self, self.cache)
        fake.publish(MODEL, "a" * 40, lambda p: mlx_target(p, DENSE), branch="branch")
        with contextlib.chdir(self.root):
            repo = hub.Repository.resolve(MODEL, "branch")
        self.assertEqual(fake.requests, [(MODEL, "branch")])
        self.assertEqual((repo.directory, repo.revision), (None, "a" * 40))
        local = hub.Repository.resolve(str(self.root / MODEL))
        self.assertEqual((local.directory, local.revision), (self.root / MODEL, None))
        with self.assertRaisesRegex(models.ModelError, "draft directory not found"):
            hub.Repository.resolve(str(self.root / "deleted-draft"))

    def test_unreachable_hub_uses_the_cache_or_explains_access(self):
        fake = FakeHub(self, self.cache)
        fake.failure = http_error(401)
        with self.assertRaisesRegex(
            models.ModelError,
            "cannot resolve owner/private: 401 Client Error.*; set HF_TOKEN .*; "
            "neither this installation nor the Hub cache records a commit "
            "for the default branch",
        ):
            hub.Repository.resolve("owner/private")
        # A branch the cache recorded resolves to its cached snapshot.
        cached_snapshot(
            self.cache, "owner/private", "c" * 40, lambda p: mlx_target(p, DENSE)
        )
        refs = self.cache / "models--owner--private/refs"
        refs.mkdir()
        (refs / "main").write_text("c" * 40)
        repo = hub.Repository.resolve("owner/private")
        self.assertEqual(repo.revision, "c" * 40)
        self.assertIn("model.safetensors", repo.files)
        self.assertIn("401 Client Error", repo.unreachable_reason)
        (refs / "main").write_text("d" * 40)
        with self.assertRaisesRegex(
            models.ModelError, "the Hub cache has no snapshot of " + "d" * 40
        ):
            hub.Repository.resolve("owner/private")

    def test_offline_rebuild_uses_the_recorded_or_pinned_snapshot(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        fake.requests.clear()
        # What huggingface_hub reports for a partial snapshot offline.
        incomplete = IncompleteSnapshotError("incomplete", snapshot_path="")
        with (
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
            mock.patch("huggingface_hub.snapshot_download", side_effect=incomplete),
        ):
            # A damaged assembly is rebuilt from the commit it recorded,
            (chosen.link / "target/model.safetensors").unlink()
            output, _ = self.prepare(chosen)
            self.assertIn(
                "Could not reach the Hub (HF_HUB_OFFLINE is set); installing "
                f"{MODEL}@{'a' * 12} from the Hub cache.",
                output,
            )
            assembly.verify(chosen.link)
            # a deleted one from the installation's pins,
            shutil.rmtree(chosen.link.resolve())
            self.prepare(chosen)
            self.assertEqual(
                assembly.verify(chosen.link)["sources"]["target"]["revision"],
                "a" * 40,
            )
            # A new selection needs the Hub once for its draft's default branch,
            # which no installation of it records.
            with self.assertRaisesRegex(
                models.ModelError,
                f"cannot resolve {DENSE.draft_repo}: HF_HUB_OFFLINE is set",
            ):
                self.prepare(selection(self.root, revision="a" * 40))
            with self.assertRaisesRegex(
                models.ModelError,
                f"cannot resolve {MODEL}: HF_HUB_OFFLINE is set; neither this "
                "installation nor the Hub cache records a commit for v2",
            ):
                self.prepare(selection(self.root, revision="v2"))
        self.assertEqual(fake.requests, [])

    def test_a_gguf_taken_by_its_ending_is_named_before_it_is_checked(self):
        def repository(root):
            root.mkdir(parents=True)
            (root / "m-UD-Q4_K_M.gguf").write_bytes(b"not a GGUF")
            (root / "m-Q8_0.gguf").touch()

        FakeHub(self, self.cache).publish("owner/model", "a" * 40, repository)
        with (
            contextlib.redirect_stdout(io.StringIO()) as output,
            self.assertRaisesRegex(models.ModelError, "unsupported GGUF header"),
        ):
            upstream.prepare(selection(self.root, "owner/model:Q4_K_M"))
        self.assertIn(
            "No GGUF is named for :Q4_K_M alone; using m-UD-Q4_K_M.gguf, the only "
            "one whose name ends in -Q4_K_M.",
            output.getvalue(),
        )

    def test_hub_failures_reading_a_header_are_model_errors(self):
        def gguf_repository(root):
            root.mkdir(parents=True, exist_ok=True)
            (root / "m-Q4_K_M.gguf").touch()

        for failure, hint in (
            (httpx.ConnectError("connection reset"), False),
            (http_error(401), True),
            (http_error(404), False),
        ):
            with self.subTest(failure=failure):
                fake = FakeHub(self, self.cache)
                fake.publish("owner/model", "a" * 40, gguf_repository)
                # The GGUF header read, before any download.
                with (
                    mock.patch.object(fake, "open", side_effect=failure),
                    self.assertRaises(models.ModelError) as raised,
                ):
                    self.prepare(selection(self.root, "owner/model:Q4_K_M"))
                message = str(raised.exception)
                self.assertTrue(
                    message.startswith("cannot install owner/model:Q4_K_M: "), message
                )
                self.assertIn(" ".join(str(failure).split()), message)
                self.assertEqual("hf auth login" in message, hint)

    def test_a_failed_download_is_reported_without_a_traceback(self):
        fake = fake_hub(self, self.cache)
        fake.download_failure = httpx.ReadTimeout("timed out")
        errors = io.StringIO()
        with (
            contextlib.redirect_stderr(errors),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = models.main(
                [
                    "--models",
                    str(self.root / "fresh"),
                    "--model",
                    MODEL,
                    "--language-only",
                    "prepare",
                ]
            )
        self.assertEqual(code, 1)
        self.assertEqual(
            errors.getvalue(), f"error: cannot install {MODEL}: timed out\n"
        )

    def test_verifying_a_missing_installation_says_so(self):
        for options in ([], ["--language-only"]):
            with self.subTest(options=options):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    code = models.main(
                        [
                            "--models",
                            str(self.root),
                            "--model",
                            MODEL,
                            *options,
                            "verify",
                        ]
                    )
                self.assertEqual(code, 1)
                self.assertEqual(
                    errors.getvalue(),
                    f"error: {MODEL} is not installed in {self.root}\n",
                )

    def test_selection_paths_do_not_conflict(self):
        root = Path("/models")
        paths = {
            models.selection_link(root, MODEL),
            models.selection_link(root, MODEL, language_only=True),
            models.selection_link(root, MODEL, revision="old"),
            models.selection_link(root, MODEL, draft_model="mine/draft"),
        }
        self.assertEqual(len(paths), 4)
        # Each is where garbage collection finds the selection links.
        links = {self.root / path.relative_to("/") for path in paths}
        for link in links:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(self.root)
        self.assertEqual(set(models.selection_links(self.root / "models")), links)


if __name__ == "__main__":
    unittest.main()

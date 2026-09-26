import contextlib
import dataclasses
import errno
import fcntl
import io
import json
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
    http_error,
    mlx_target,
    pins,
    selection,
    text_config,
)
from install import assembly, families, hub, legacy, models, upstream


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
        repos = [family.draft.repo for family in families.FAMILIES]
        for repo in repos:
            with self.subTest(repo=repo):
                self.assertEqual(models.validate_repo_id(repo), repo)
        self.assertEqual(len(set(repos)), len(repos))

    def test_family_is_identified_by_architecture_not_name(self):
        for family in families.FAMILIES:
            self.assertIs(
                families.family_for({"text_config": text_config(family)}), family
            )
        # A differing field is another architecture, whatever the repository is called.
        for changes in (
            {"num_hidden_layers": 48},
            {"max_position_embeddings": 131072},
            {"vocab_size": 151936},
            {"head_dim": 128},
            {"model_type": "qwen3_moe"},
            {"num_experts": 128},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(
                    models.ModelError, "no supported model has this architecture"
                ),
            ):
                family = MOE if "num_experts" in changes else DENSE
                families.family_for({"text_config": text_config(family, **changes)})
        with self.assertRaises(models.ModelError):
            families.family_for({"hidden_size": 5120})

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
        fake = FakeHub(self, self.cache)
        fake.publish(
            "someone/renamed-27b",
            "a" * 40,
            lambda p: mlx_target(p, DENSE, changes={"num_hidden_layers": 48}),
        )
        with self.assertRaisesRegex(models.ModelError, "no supported model"):
            self.prepare(selection(self.root, "someone/renamed-27b"))
        self.assertEqual(fake.requests, [("someone/renamed-27b", None)])
        self.assertEqual(fake.downloads, ["someone/renamed-27b/config.json"])

    def test_only_a_splash_manifest_makes_a_legacy_package(self):
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
        packaged = selection(self.root, "someone/package", language_only=False)
        with mock.patch.object(legacy, "prepare") as install_package:
            self.prepare(selection(self.root))
            self.prepare(packaged)
        install_package.assert_called_once_with(packaged)
        self.assertEqual(
            assembly.verify(selection(self.root).link)["sources"]["target"]["revision"],
            "b" * 40,
        )

    def test_a_package_rejects_source_options(self):
        fake = FakeHub(self, self.cache)
        fake.publish(
            "someone/package",
            "c" * 40,
            lambda p: (
                p.mkdir(parents=True),
                (p / "manifest.json").write_text(
                    json.dumps({"format": {"name": "splash-packed-q4"}})
                ),
            ),
        )
        for options in ({"revision": "c" * 40}, {"language_only": True}):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(models.ModelError, "require an upstream"),
            ):
                self.prepare(
                    selection(
                        self.root,
                        "someone/package",
                        **{"language_only": False} | options,
                    )
                )
        self.assertEqual(fake.downloads, ["someone/package/manifest.json"])

    def test_only_mlx_affine_quantization_is_accepted(self):
        def target(quantization):
            def build(root):
                mlx_target(root, DENSE)
                config = json.loads((root / "config.json").read_text())
                del config["quantization"]
                (root / "config.json").write_text(json.dumps(config | quantization))

            return build

        fake = fake_hub(self, self.cache)
        for name, quantization in (
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
            ),
            ("mxfp4", {"quantization": {"mode": "mxfp4", "bits": 4, "group_size": 64}}),
            ("q8", {"quantization": {"bits": 8, "group_size": 64}}),
        ):
            with self.subTest(name=name):
                fake.publish(f"someone/{name}", "b" * 40, target(quantization))
                with self.assertRaisesRegex(
                    models.ModelError, "requires an MLX affine 4-bit/group-64"
                ):
                    self.prepare(selection(self.root, f"someone/{name}"))
        # MLX writes both keys, and states the mode only in newer versions.
        affine = {"bits": 4, "group_size": 64, "mode": "affine"}
        fake.publish(
            "someone/mlx",
            "c" * 40,
            target({"quantization": affine, "quantization_config": affine}),
        )
        self.prepare(selection(self.root, "someone/mlx"))
        assembly.verify(selection(self.root, "someone/mlx").link)

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
                ):
                    self.prepare(chosen)
                self.assertEqual(fake.downloads, [])
                self.assertFalse(chosen.models_root.exists())
                shutil.rmtree(fake.remote)

    def test_source_assembly_pairs_the_draft_by_architecture(self):
        fake = FakeHub(self, self.cache)
        fake.publish(
            "someone/my-favourite-model", "a" * 40, lambda p: mlx_target(p, MOE)
        )
        fake.publish(MOE.draft.repo, DRAFT_COMMIT, lambda p: draft_dir(p, MOE))
        # The name says nothing about the model; the configuration does.
        chosen = selection(self.root, "someone/my-favourite-model")
        self.prepare(chosen)
        self.assertEqual(
            fake.requests,
            [
                ("someone/my-favourite-model", None),
                (MOE.draft.repo, None),
            ],
        )
        record = assembly.verify(chosen.link)
        self.assertEqual(record["family"], MOE.name)
        self.assertEqual(
            record["sources"],
            {
                "target": {"repo": chosen.model, "revision": "a" * 40},
                "draft": {"repo": MOE.draft.repo, "revision": DRAFT_COMMIT},
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
        # A checkpoint without the tower cannot serve images.
        del shards["vision_tower.blocks.0.attn.qkv.weight"]
        fake.publish("someone/text-model", "b" * 40, target)
        with self.assertRaisesRegex(models.ModelError, "no vision tower"):
            self.prepare(
                selection(self.root, "someone/text-model", language_only=False)
            )

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
        self.assertEqual(fake.requests, [(MODEL, None), (DENSE.draft.repo, None)])
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
        self.assertEqual(fake.requests, [(MODEL, None), (DENSE.draft.repo, None)])
        self.assertNotIn(f"{DENSE.draft.repo}/model.safetensors", fake.downloads)
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
            f"{MODEL}@{'b' * 40}: no supported model has this architecture",
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
        fake.publish(DENSE.draft.repo, "e" * 40, lambda p: draft_dir(p, DENSE))
        fake.requests.clear(), fake.downloads.clear()
        output, _ = self.prepare(chosen)
        self.assertIn(
            f"{DENSE.draft.repo} moved from {DRAFT_COMMIT[:12]} to {'e' * 12}", output
        )
        self.assertEqual(
            assembly.verify(chosen.link)["sources"],
            {
                "target": {"repo": MODEL, "revision": "a" * 40},
                "draft": {"repo": DENSE.draft.repo, "revision": "e" * 40},
            },
        )
        # Only the new draft is fetched; the target is not downloaded again.
        self.assertEqual(fake.requests, [(MODEL, None), (DENSE.draft.repo, None)])
        self.assertTrue(
            all(name.startswith(DENSE.draft.repo + "/") for name in fake.downloads)
        )
        self.assertEqual(pins(self.cache), sorted(["a" * 40, "e" * 40]))

    def test_an_installation_of_another_draft_repository_moves_to_the_familys(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        other = dataclasses.replace(
            DENSE, draft=dataclasses.replace(DENSE.draft, repo="someone/other-draft")
        )
        fake.publish(other.draft.repo, "c" * 40, lambda p: draft_dir(p, DENSE))
        with mock.patch.object(families, "FAMILIES", (other, MOE)):
            self.prepare(chosen)
        fake.requests.clear()
        output, _ = self.prepare(chosen)
        self.assertIn(
            f"its draft is now {DENSE.draft.repo}@{DRAFT_COMMIT[:12]}", output
        )
        self.assertEqual(
            assembly.verify(chosen.link)["sources"]["draft"],
            {"repo": DENSE.draft.repo, "revision": DRAFT_COMMIT},
        )
        self.assertEqual(fake.requests, [(MODEL, None), (DENSE.draft.repo, None)])
        # Its pin of the repository it no longer links is retired too.
        self.assertEqual(pins(self.cache), sorted(["a" * 40, DRAFT_COMMIT]))

    def test_a_draft_the_hub_cannot_resolve_keeps_the_installed_one(self):
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        # The draft's main names a commit the Hub cannot serve.
        fake.branches[DENSE.draft.repo, "main"] = "f" * 40
        fake.downloads.clear()
        output, _ = self.prepare(chosen)
        self.assertIn("Could not reach the Hub (404 Client Error", output)
        self.assertIn(
            f"using the installed draft {DENSE.draft.repo}@{DRAFT_COMMIT[:12]}.", output
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
        fake.publish(DENSE.draft.repo, "e" * 40, build_draft)
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
                "draft": {"repo": DENSE.draft.repo, "revision": DRAFT_COMMIT},
            },
        )
        self.assertEqual(pins(self.cache), sorted(["b" * 40, DRAFT_COMMIT]))
        return warnings

    def test_a_new_draft_whose_download_fails_keeps_the_installed_draft(self):
        warnings = self.move_target_and_draft(
            lambda p: draft_dir(p, DENSE), failing_revision="e" * 40
        )
        self.assertIn(
            f"Warning: cannot use the {DENSE.name} draft {DENSE.draft.repo}@"
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
            f"Warning: cannot use the {DENSE.name} draft {DENSE.draft.repo}@"
            f"{'e' * 12}; keeping the installed one: {DENSE.draft.repo} does not "
            f"contain a DFlash2 checkpoint for {DENSE.name}",
            warnings,
        )

    def test_a_new_draft_of_another_architecture_keeps_the_installed_draft(self):
        warnings = self.move_target_and_draft(
            lambda p: draft_dir(p, DENSE, **{"dflash_config.block_size": 16})
        )
        self.assertIn(
            f"Warning: cannot use the {DENSE.name} draft {DENSE.draft.repo}@"
            f"{'e' * 12}; keeping the installed one: draft configuration is "
            f"incompatible with {DENSE.name}: dflash_config.block_size 16, not 8",
            warnings,
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
                [(MODEL, None), (DENSE.draft.repo, None)],
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
        fake.publish(DENSE.draft.repo, "e" * 40, lambda p: draft_dir(p, DENSE))
        fake.downloads.clear()
        with mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(moved)):
            _, warnings = self.prepare(chosen)
        self.assertIn(
            f"Warning: keeping the installed {MODEL}@{'a' * 12}; cannot update it "
            f"({DENSE.draft.repo} moved from {DRAFT_COMMIT[:12]} to {'e' * 12}): "
            f"the Hub cache has no snapshot {'a' * 40} of {MODEL}",
            warnings,
        )
        self.assertEqual(chosen.link.resolve(), installed)
        self.assertEqual(fake.downloads, [])
        self.assertEqual(list(moved.iterdir()), [])

    def test_an_incompatible_draft_is_an_error_not_a_crash(self):
        fake_hub(self, self.cache)
        local = draft_dir(self.root / "draft", DENSE)
        config = json.loads((local / "config.json").read_text())
        layers = {"num_hidden_layers": DENSE.draft.layers + 1}
        (local / "config.json").write_text(json.dumps(config | layers))
        with self.assertRaisesRegex(models.ModelError, "draft configuration"):
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
                (self.cache / hub.folder_name(DENSE.draft.repo) / "refs/splash").glob(
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
            hub.os, "link", side_effect=OSError(errno.EROFS, "read only")
        ):
            output, warnings = self.prepare(chosen)
        self.assertIn("is already installed", output)
        self.assertIn("external cache pruning", warnings)
        self.assertEqual(pins(self.cache), [])

    def test_a_packed_draft_installation_is_installed_again(self):
        # Assemblies linked Splash-DFlash2's packed drafts before drafts were
        # prepared from their checkpoints; the runtime loads those no more.
        fake = fake_hub(self, self.cache)
        chosen = selection(self.root)
        self.prepare(chosen)
        record = assembly.verify(chosen.link)
        files = {
            name: Path(entry["path"])
            for name, entry in record["files"].items()
            if not name.startswith("draft/")
        }
        packed_repo = "company/Splash-DFlash2"
        packed = hub.snapshot(packed_repo, "c" * 40)
        packed.mkdir(parents=True)
        for name in ("config.json", "model.bin", "layer-0.bin"):
            (packed / name).write_text(name)
            files["draft/" + name] = packed / name
        old = record | {
            "sources": record["sources"]
            | {"draft": {"repo": packed_repo, "revision": "c" * 40}},
            "files": {name: assembly.file_record(path) for name, path in files.items()},
        }
        # As the installer that assembled such drafts did, without the check;
        # another installation pins the same draft.
        other = chosen.models_root / "someone/other"
        with (
            models.installation_lock(chosen.models_root),
            mock.patch.object(assembly, "_packed_draft", return_value=False),
        ):
            models.link_selection(
                chosen.link, assembly.build(chosen.models_root, old, files)
            )
            for installation in (chosen.link, other):
                hub.pin(packed, packed_repo, installation)
        fake.requests.clear()
        output, _ = self.prepare(chosen)
        self.assertIn(
            f"Reinstalling {MODEL}: its draft is not a DFlash2 checkpoint", output
        )
        self.assertIn("draft/model.safetensors", assembly.verify(chosen.link)["files"])
        self.assertEqual(fake.requests, [(MODEL, None), (DENSE.draft.repo, None)])
        # The replaced assembly's record names the pins this installation
        # retires; the other installation's pin stays.
        self.assertFalse(hub.pinned(packed, chosen.link).exists())
        self.assertTrue(hub.pinned(packed, other).exists())

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
            {"repo": DENSE.draft.repo, "revision": DRAFT_COMMIT},
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
                f"cannot resolve {DENSE.draft.repo}: HF_HUB_OFFLINE is set",
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

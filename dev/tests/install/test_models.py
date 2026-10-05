import argparse
import contextlib
import copy
import errno
import fcntl
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
from huggingface_hub.hf_api import RepoSibling

from dev.tests.installer_fixtures import http_error
from install import hub, legacy
from install import models as installer


class ModelArtifactTest(unittest.TestCase):
    MODEL_ID = "community/My-Splash.Model_1"
    REVISION = "a" * 40

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture_number = 0
        # Never use real user credentials or contact the network in these tests.
        for patch in (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("huggingface_hub.get_token", return_value=None),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.api = self.start_patch("huggingface_hub.HfApi")
        self.manifest_download = self.start_patch("huggingface_hub.hf_hub_download")
        self.download = self.start_patch("huggingface_hub.snapshot_download")

    def start_patch(self, name):
        patch = mock.patch(name)
        value = patch.start()
        self.addCleanup(patch.stop)
        return value

    @staticmethod
    def packed_file(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            file.write(struct.pack("<8sII", b"MDFT0001", 0, 0))
            file.seek(legacy.ALIGNMENT - 1)
            file.write(b"\0")

    def package_fixture(self, *, schema=3, model_id=None, revision=None):
        self.fixture_number += 1
        snapshot = (
            self.root
            / f"cache-{self.fixture_number}"
            / ("models--" + (model_id or self.MODEL_ID).replace("/", "--"))
            / "snapshots"
            / (revision or self.REVISION)
        )
        target_layers, draft_layers = (64, 5) if schema == 3 else (40, 6)
        target_files = (
            "embedding.bin",
            "head.bin",
            *(f"layer-{index}.bin" for index in range(target_layers)),
        )
        for name in (
            "draft/model.bin",
            "vision/model.bin",
            *(f"target/{file}" for file in target_files),
            *(f"draft/layer-{index}.bin" for index in range(draft_layers)),
        ):
            self.packed_file(snapshot / name)
        tokenizer = snapshot / "tokenizer"
        tokenizer.mkdir()
        for name in legacy.PACKAGE_TOKENIZER_FILES:
            (tokenizer / name).write_text(f"{name}\n")
        if schema == 4:
            (snapshot / "layout.json").write_text("{}\n")
        records = [
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size": path.stat().st_size,
                "sha256": installer.sha256(path),
            }
            for path in sorted(item for item in snapshot.rglob("*") if item.is_file())
        ]
        manifest = {
            "schema_version": schema,
            "model": "Community fine-tuned model",
            "format": {
                "name": "splash-packed-q4" + ("-moe" if schema == 4 else ""),
                "section_alignment_bytes": legacy.ALIGNMENT,
                "target_layer_magic": "MDFM0001" if schema == 4 else "MDFL0006",
                "draft_layer_magic": "MDFD0004",
                "vision_magic": "MDFV0001",
            },
            "execution_geometry": {},
            "artifacts": records,
        }
        if schema == 4:
            manifest["target"] = {"architecture": "qwen3_5_moe"}
            manifest["draft"] = {"architecture": "DFlash2DraftModel"}
        self.write_manifest(snapshot, manifest)
        return snapshot, manifest

    @staticmethod
    def hub_records(snapshot):
        manifest = json.loads((snapshot / "manifest.json").read_text())
        return list(manifest.get("artifacts", []))

    @staticmethod
    def write_manifest(snapshot, manifest):
        # Match the real Hub cache: installation -> snapshot -> blob.
        path = snapshot / "manifest.json"
        if not path.exists():
            blob = snapshot.parent.parent / "blobs" / "manifest"
            blob.parent.mkdir(exist_ok=True)
            blob.write_text("")
            path.symlink_to("../../blobs/manifest")
        path.write_text(json.dumps(manifest, sort_keys=True))

    def configure_hub(self, snapshot):
        self.api.return_value.model_info.return_value = SimpleNamespace(
            sha=self.REVISION,
            siblings=[
                RepoSibling(
                    rfilename="manifest.json",
                    size=(snapshot / "manifest.json").stat().st_size,
                )
            ]
            + [
                RepoSibling(rfilename=r["path"], size=r["size"])
                for r in self.hub_records(snapshot)
                if isinstance(r, dict) and "path" in r and "size" in r
            ],
        )
        self.manifest_download.return_value = str(snapshot / "manifest.json")
        self.download.return_value = str(snapshot)

    def test_accepts_full_hub_ids_without_name_or_owner_allowlist(self):
        for model_id in (
            self.MODEL_ID,
            "a/b",
            "other-team/finetune",
            "incoai/anything",
        ):
            with self.subTest(model=model_id):
                self.assertEqual(installer.validate_repo_id(model_id), model_id)
                self.assertEqual(installer.parse_model_id(model_id), model_id)
                self.assertEqual(
                    installer.selection_link(self.root, model_id), self.root / model_id
                )

    def test_missing_or_invalid_model_fails_before_creating_or_downloading(self):
        destination = self.root / "uncreated"
        base = ["--models", str(destination)]
        cases = [
            None,
            "short-name",
            "../outside",
            "owner/repo/extra",
            "owner/../repo",
            "https://huggingface.co/owner/repo",
            "owner/repo.git",
            "owner/repo--one",
            "owner/repo..one",
            "owner/-repo",
            "owner/repo.",
            "owner/" + "x" * 97,
            "owner/repo\n",
            "owner\\repo",
            " owner/repo",
            "",
        ]
        with mock.patch.object(legacy, "resolve_snapshot") as download:
            for model in cases:
                args = (
                    [*base, "prepare"]
                    if model is None
                    else [*base, "--model", model, "prepare"]
                )
                with (
                    self.subTest(model=model),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as raised:
                        installer.main(args)
                    self.assertEqual(raised.exception.code, 2)
            download.assert_not_called()
        self.assertFalse(destination.exists())
        with self.assertRaises(installer.ModelError):
            installer.validate_repo_id(None)
        with self.assertRaises(argparse.ArgumentTypeError):
            installer.parse_model_id("short-name")

    def test_a_selection_validates_its_model_before_creating_paths(self):
        models = self.root / "uncreated"
        with self.assertRaises(installer.ModelError):
            installer.Selection.of(models, "../outside")
        self.assertFalse(models.exists())

    def test_quick_and_full_verification(self):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"
        installer.link_selection(models / self.MODEL_ID, snapshot)
        for full in (False, True):
            legacy.verify(models / self.MODEL_ID, self.MODEL_ID, full=full)
        packed = snapshot / "target/embedding.bin"
        with packed.open("r+b") as file:
            file.seek(128)
            file.write(b"x")
        legacy.verify(models / self.MODEL_ID, self.MODEL_ID, full=False)
        with self.assertRaisesRegex(installer.ModelError, "checksum changed"):
            legacy.verify(models / self.MODEL_ID, self.MODEL_ID, full=True)

    def test_accepts_both_formats_with_arbitrary_display_name_and_metadata(self):
        for schema in (3, 4):
            with self.subTest(schema=schema):
                snapshot, manifest = self.package_fixture(schema=schema)
                # Aggregate digest algorithms and descriptive metadata are producer-owned.
                manifest["artifact_set_sha256"] = "producer-specific metadata"
                manifest["execution_geometry"]["extra_metadata"] = 1
                self.write_manifest(snapshot, manifest)
                validated = legacy.validate_manifest(snapshot / "manifest.json")
                legacy.verify_artifacts(snapshot, validated, full=True)

    def test_bad_manifest_fails_before_downloading_weights(self):
        snapshot, original = self.package_fixture()
        for field, value in (
            ("schema_version", 99),
            ("schema_version", True),
            ("model", ""),
            ("model", []),
            ("format", None),
            ("format", {"name": []}),
            ("format", {"name": {}}),
            ("format", {"name": "safetensors"}),
            ("execution_geometry", []),
            ("artifacts", []),
        ):
            with self.subTest(field=field, value=value):
                manifest = copy.deepcopy(original)
                manifest[field] = value
                self.write_manifest(snapshot, manifest)
                self.configure_hub(snapshot)
                with self.assertRaises(installer.ModelError):
                    legacy.resolve_snapshot(self.MODEL_ID)
                self.download.assert_not_called()

    def test_format_magic_and_moe_architecture_are_checked(self):
        snapshot, original = self.package_fixture(schema=4)
        for section, key, value in (
            ("format", "target_layer_magic", "WRONG"),
            ("format", "section_alignment_bytes", 1),
            ("target", "architecture", "unsupported"),
            ("draft", "architecture", "unsupported"),
        ):
            with self.subTest(section=section, key=key):
                manifest = copy.deepcopy(original)
                manifest[section][key] = value
                self.write_manifest(snapshot, manifest)
                with self.assertRaises(installer.ModelError):
                    legacy.validate_manifest(snapshot / "manifest.json")

    def test_manifest_must_list_every_file_the_runtime_reads(self):
        for schema in (3, 4):
            snapshot, original = self.package_fixture(schema=schema)
            for name in (
                "target/embedding.bin",
                "target/head.bin",
                "target/layer-0.bin",
                f"target/layer-{63 if schema == 3 else 39}.bin",
                "draft/model.bin",
                f"draft/layer-{4 if schema == 3 else 5}.bin",
                "vision/model.bin",
                "tokenizer/config.json",
            ):
                with self.subTest(schema=schema, missing=name):
                    manifest = {
                        **original,
                        "artifacts": [
                            r for r in original["artifacts"] if r["path"] != name
                        ],
                    }
                    self.write_manifest(snapshot, manifest)
                    self.configure_hub(snapshot)
                    with self.assertRaisesRegex(
                        installer.ModelError, "missing: " + name
                    ):
                        legacy.resolve_snapshot(self.MODEL_ID)
                    self.download.assert_not_called()

    def test_manifest_rejects_unsafe_ambiguous_and_glob_paths(self):
        snapshot, original = self.package_fixture()
        for path in (
            "../escape",
            "/absolute",
            "target/../escape",
            "./target/file",
            "target//file",
            "target/",
            "target/./file",
            "target\\file",
            "target/*",
            "target/?",
            "target/[abc]",
            ".",
            "target/\x00file",
            "target/\nfile",
            "manifest.json",
            "",
        ):
            with self.subTest(path=path):
                manifest = copy.deepcopy(original)
                manifest["artifacts"][0]["path"] = path
                self.write_manifest(snapshot, manifest)
                with self.assertRaisesRegex(installer.ModelError, "invalid artifact"):
                    legacy.validate_manifest(snapshot / "manifest.json")

    def test_manifest_rejects_duplicate_overlapping_and_invalid_records(self):
        snapshot, original = self.package_fixture()
        first = original["artifacts"][0]
        for records in (
            [*original["artifacts"], first],
            [*original["artifacts"], {**first, "path": "target"}],
            [{**first, "size": True}],
            [{**first, "size": -1}],
            [{**first, "size": 3}],
            [{**first, "sha256": "+" + "a" * 63}],
            [{**first, "sha256": "z" * 64}],
            [None],
            [{"path": "target/x.bin"}],
        ):
            with self.subTest(records=records):
                self.write_manifest(snapshot, {**original, "artifacts": records})
                with self.assertRaises(installer.ModelError):
                    legacy.validate_manifest(snapshot / "manifest.json")

    def test_full_verification_accepts_uppercase_sha256(self):
        snapshot, manifest = self.package_fixture()
        for record in manifest["artifacts"]:
            record["sha256"] = record["sha256"].upper()
        self.write_manifest(snapshot, manifest)
        legacy.verify_artifacts(
            snapshot,
            legacy.validate_manifest(snapshot / "manifest.json"),
            full=True,
        )

    def test_download_pins_manifest_and_artifacts_to_one_resolved_commit(self):
        snapshot, manifest = self.package_fixture()
        self.configure_hub(snapshot)
        self.assertEqual(legacy.resolve_snapshot(self.MODEL_ID), snapshot.resolve())
        self.api.assert_called_once_with(token=False)
        self.api.return_value.model_info.assert_called_once_with(
            self.MODEL_ID, revision="main", files_metadata=True
        )
        common = dict(
            repo_id=self.MODEL_ID,
            repo_type="model",
            revision=self.REVISION,
            token=False,
        )
        self.manifest_download.assert_called_once_with(
            filename="manifest.json", **{**common, "revision": "main"}
        )
        self.download.assert_called_once_with(
            allow_patterns=[
                "manifest.json",
                *(r["path"] for r in manifest["artifacts"]),
            ],
            **common,
        )

    def test_custom_hub_cache_installs_offline_without_copying_weights(self):
        for variable in ("HF_HOME", "HF_HUB_CACHE"):
            with self.subTest(variable=variable):
                snapshot, _ = self.package_fixture()
                location = self.root / variable / "external disk"
                cache = location / "hub" if variable == "HF_HOME" else location
                repository = cache / snapshot.parent.parent.name
                repository.parent.mkdir(parents=True)
                shutil.move(str(snapshot.parent.parent), repository)
                snapshot = repository / "snapshots" / self.REVISION
                (repository / "refs").mkdir()
                (repository / "refs/main").write_text(self.REVISION)
                models = self.root / variable / "installed"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(installer.__file__).resolve()),
                        "--models",
                        str(models),
                        "--model",
                        self.MODEL_ID,
                        "prepare",
                    ],
                    env={
                        "HOME": str(self.root / "home"),
                        "HF_HUB_OFFLINE": "1",
                        variable: str(location),
                    },
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                installed = models / self.MODEL_ID
                self.assertTrue(installed.is_symlink())
                self.assertEqual(installed.resolve(), snapshot.resolve())
                self.assertTrue(
                    (installed / "target/embedding.bin").samefile(
                        snapshot / "target/embedding.bin"
                    )
                )
                self.assertEqual(len(list((repository / "refs/splash").glob("*/*"))), 1)

    def test_missing_or_oversize_remote_manifest_fails_before_any_download(self):
        snapshot, _ = self.package_fixture()
        for siblings in (
            [],
            [
                SimpleNamespace(
                    rfilename="manifest.json", size=installer.MAX_JSON_BYTES + 1
                )
            ],
        ):
            with self.subTest(siblings=siblings):
                self.configure_hub(snapshot)
                self.api.return_value.model_info.return_value.siblings = siblings
                with self.assertRaisesRegex(installer.ModelError, "manifest.json"):
                    legacy.resolve_snapshot(self.MODEL_ID)
                self.manifest_download.assert_not_called()
                self.download.assert_not_called()

    def test_hub_must_return_a_concrete_commit(self):
        snapshot, _ = self.package_fixture()
        self.configure_hub(snapshot)
        self.api.return_value.model_info.return_value.sha = "main"
        with self.assertRaisesRegex(installer.ModelError, "snapshot commit"):
            legacy.resolve_snapshot(self.MODEL_ID)
        self.manifest_download.assert_not_called()
        self.download.assert_not_called()

    def test_corrupt_download_retries_same_commit_and_rejects_bad_artifacts(self):
        snapshot, _ = self.package_fixture()
        self.configure_hub(snapshot)
        (snapshot / "target/embedding.bin").write_bytes(b"bad")
        with self.assertRaisesRegex(installer.ModelError, "wrong size"):
            legacy.resolve_snapshot(self.MODEL_ID)
        self.assertEqual(self.download.call_count, 1)
        self.assertEqual(
            self.manifest_download.call_args_list[0].kwargs["revision"], "main"
        )
        for call in (
            *self.manifest_download.call_args_list[1:],
            *self.download.call_args_list,
        ):
            self.assertEqual(call.kwargs["revision"], self.REVISION)
        self.assertNotIn("force_download", self.download.call_args.kwargs)
        self.api.return_value.model_info.assert_called_once()

    def test_corrupt_cached_manifest_is_refetched_at_the_same_commit(self):
        snapshot, manifest = self.package_fixture()
        self.configure_hub(snapshot)
        manifest_path = snapshot / "manifest.json"
        manifest_path.write_text("{truncated")

        def download_manifest(**kwargs):
            if kwargs.get("force_download"):
                self.write_manifest(snapshot, manifest)
            return str(manifest_path)

        self.manifest_download.side_effect = download_manifest
        self.assertEqual(legacy.resolve_snapshot(self.MODEL_ID), snapshot.resolve())
        self.assertEqual(self.manifest_download.call_count, 2)
        self.assertEqual(
            self.manifest_download.call_args_list[0].kwargs["revision"], "main"
        )
        retry = self.manifest_download.call_args.kwargs
        self.assertEqual(retry["revision"], self.REVISION)
        self.assertTrue(retry["force_download"])
        self.download.assert_called_once()
        self.assertEqual(self.download.call_args.kwargs["revision"], self.REVISION)
        self.assertNotIn("force_download", self.download.call_args.kwargs)
        self.api.return_value.model_info.assert_called_once()

    def test_malformed_fresh_manifest_fails_after_one_forced_refetch(self):
        snapshot, _ = self.package_fixture()
        self.configure_hub(snapshot)
        (snapshot / "manifest.json").write_text("{truncated")
        with self.assertRaises(installer.ModelError):
            legacy.resolve_snapshot(self.MODEL_ID)
        self.assertEqual(self.manifest_download.call_count, 2)
        self.assertEqual(
            self.manifest_download.call_args.kwargs["revision"], self.REVISION
        )
        self.assertTrue(self.manifest_download.call_args.kwargs["force_download"])
        self.download.assert_not_called()
        self.api.return_value.model_info.assert_called_once()

    def test_manifest_cannot_change_between_preflight_and_snapshot(self):
        snapshot, manifest = self.package_fixture()
        other = self.root / "other"
        shutil.copytree(snapshot, other)
        manifest["model"] = "changed"
        self.write_manifest(other, manifest)
        self.configure_hub(snapshot)
        self.download.return_value = str(other)
        with self.assertRaisesRegex(installer.ModelError, "manifest changed"):
            legacy.resolve_snapshot(self.MODEL_ID)

    def test_incomplete_hub_package_fails_before_weights_download(self):
        snapshot, _ = self.package_fixture()
        self.configure_hub(snapshot)
        info = self.api.return_value.model_info.return_value
        info.siblings = [
            item for item in info.siblings if item.rfilename != "target/head.bin"
        ]
        with self.assertRaisesRegex(installer.ModelError, "Hub artifact"):
            legacy.resolve_snapshot(self.MODEL_ID)
        self.download.assert_not_called()

    def test_complete_cached_snapshot_can_be_relinked_offline(self):
        snapshot, _ = self.package_fixture()
        self.api.return_value.model_info.side_effect = httpx.ConnectError("offline")
        with mock.patch(
            "huggingface_hub.try_to_load_from_cache",
            return_value=str(snapshot / "manifest.json"),
        ):
            self.assertEqual(legacy.resolve_snapshot(self.MODEL_ID), snapshot.resolve())
        self.download.assert_not_called()

    def test_corrupt_artifact_is_repaired_without_redownloading_others(self):
        snapshot, _ = self.package_fixture()
        self.configure_hub(snapshot)
        damaged = snapshot / "target/head.bin"
        original = damaged.read_bytes()
        damaged.write_bytes(b"bad")

        def download_file(**kwargs):
            if kwargs.get("force_download"):
                self.assertEqual(kwargs["filename"], "target/head.bin")
                damaged.write_bytes(original)
            return str(snapshot / kwargs["filename"])

        self.manifest_download.side_effect = download_file
        self.assertEqual(legacy.resolve_snapshot(self.MODEL_ID), snapshot.resolve())
        self.download.assert_called_once()
        self.assertEqual(self.manifest_download.call_count, 2)

    def test_user_credentials_apply_to_any_repository_with_env_priority(self):
        cases = (
            (self.MODEL_ID, None, None, False),
            ("incoai/model", None, None, False),
            (self.MODEL_ID, None, "hf_testlogin", "hf_testlogin"),
            ("incoai/model", None, "hf_testlogin", "hf_testlogin"),
            ("incoai/model", "hf_testenv", "hf_testlogin", "hf_testenv"),
        )
        for model_id, env, login, expected in cases:
            with self.subTest(model=model_id, env=env, login=login):
                snapshot, _ = self.package_fixture(model_id=model_id)
                self.configure_hub(snapshot)
                environment = {"HF_ENDPOINT": "https://unused.invalid"}
                if env:
                    environment["HF_TOKEN"] = env
                with (
                    mock.patch.dict(os.environ, environment, clear=True),
                    mock.patch("huggingface_hub.get_token", return_value=login),
                ):
                    legacy.resolve_snapshot(model_id)
                for call in (
                    self.api.call_args,
                    self.manifest_download.call_args,
                    self.download.call_args,
                ):
                    self.assertEqual(call.kwargs["token"], expected)
                    # HF_ENDPOINT reaches the Hub library, which reads it.
                    self.assertNotIn("endpoint", call.kwargs)

    def test_download_failure_is_actionable_and_redacts_credentials(self):
        self.api.return_value.model_info.side_effect = RuntimeError(
            "network unavailable hf_testdistributiontoken"
        )
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_testdistributiontoken"}):
            with self.assertRaises(installer.ModelError) as raised:
                legacy.resolve_snapshot("incoai/model")
        self.assertIn("incoai/model@main", str(raised.exception))
        self.assertIn("network unavailable [redacted]", str(raised.exception))
        self.assertNotIn("hf_testdistributiontoken", str(raised.exception))
        self.assertNotIn("HF_TOKEN", os.environ)

    def test_command_line_takes_required_model_before_command(self):
        args = installer.parse_args(["--model", self.MODEL_ID, "prepare"])
        self.assertEqual((args.command, args.model), ("prepare", self.MODEL_ID))

    def test_prepare_atomically_installs_then_reuses_offline_without_hashing_weights(
        self,
    ):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"
        args = installer.Selection.of(models, self.MODEL_ID)
        with (
            mock.patch.object(
                legacy, "resolve_snapshot", return_value=snapshot
            ) as resolve,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            legacy.prepare(args)
        destination = models / self.MODEL_ID
        self.assertTrue(destination.is_symlink())
        self.assertEqual(destination.resolve(), snapshot.resolve())
        resolve.assert_called_once_with(self.MODEL_ID)
        with (
            mock.patch.object(legacy, "resolve_snapshot") as second,
            mock.patch.object(installer, "sha256") as hash_file,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            legacy.prepare(args)
            result = installer.main(
                ["--models", str(models), "--model", self.MODEL_ID, "verify"]
            )
        self.assertEqual(result, 0)
        second.assert_not_called()
        hash_file.assert_not_called()
        self.api.assert_not_called()

    def test_a_package_downloads_outside_the_installation_lock(self):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"

        def locked():
            with (models / ".install.lock").open("a+b") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return True
            return False

        def download(repo_id):
            # Every start takes the lock; none waits for a download.
            self.assertFalse(locked())
            return snapshot

        pin = hub.pin

        def pin_locked(*arguments):
            self.assertTrue(locked())
            return pin(*arguments)

        with (
            mock.patch.object(legacy, "resolve_snapshot", side_effect=download),
            mock.patch.object(hub, "pin", side_effect=pin_locked) as pinned,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            legacy.prepare(installer.Selection.of(models, self.MODEL_ID))
        pinned.assert_called_once()
        self.assertEqual((models / self.MODEL_ID).resolve(), snapshot.resolve())

    def test_installed_package_starts_through_prepare_without_the_hub(self):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"
        installer.link_selection(models / self.MODEL_ID, snapshot)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(
                installer.main(
                    ["--models", str(models), "--model", self.MODEL_ID, "prepare"]
                ),
                0,
            )
        self.assertIn("is already installed", output.getvalue())
        self.api.assert_not_called()
        self.manifest_download.assert_not_called()
        self.download.assert_not_called()

    def test_variant_model_ids_parse_and_name_selection_links(self):
        self.assertEqual(
            installer.split_model_id("owner/repo:UD-Q4_K_M"),
            ("owner/repo", "UD-Q4_K_M"),
        )
        self.assertEqual(installer.split_model_id("owner/repo"), ("owner/repo", None))
        self.assertEqual(
            installer.parse_model_id("owner/repo:UD-Q4_K_M"),
            "owner/repo:UD-Q4_K_M",
        )
        for bad in ("owner/repo:", "owner/repo:a b", "owner/repo:..", "owner:v"):
            with self.assertRaises(installer.ModelError):
                installer.split_model_id(bad)
        with self.assertRaises(argparse.ArgumentTypeError):
            installer.parse_model_id("owner/repo:")
        models = self.root / "models"
        self.assertEqual(
            installer.selection_link(models, "owner/repo:UD-Q4_K_M"),
            models / "owner" / "repo:UD-Q4_K_M",
        )
        self.assertEqual(
            installer.selection_link(models, "owner/repo"), models / "owner/repo"
        )

    def test_link_prints_the_selection_link_of_the_source_options(self):
        # make's MODEL_ROOT is this output; a relative draft folder names the
        # installation splash serve --draft-model selects from the same folder.
        models = self.root / "models"
        draft = self.root / "draft"
        draft.mkdir()
        for arguments, options in (
            (["--model", "owner/repo:UD-Q4_K_M"], {}),
            (
                ["--model", "owner/repo", "--revision", "b" * 40, "--language-only"]
                + ["--draft-model", os.path.relpath(draft)],
                {
                    "revision": "b" * 40,
                    "language_only": True,
                    "draft_model": str(draft.resolve()),
                },
            ),
        ):
            with (
                self.subTest(arguments=arguments),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(
                    installer.main(["--models", str(models), *arguments, "link"]), 0
                )
            link = installer.selection_link(models.resolve(), arguments[1], **options)
            self.assertEqual(output.getvalue(), f"{link}\n")

    def test_a_package_takes_no_variant(self):
        snapshot, _ = self.package_fixture()
        self.configure_hub(snapshot)
        models = self.root / "models"
        with self.assertRaisesRegex(installer.ModelError, "no variants"):
            legacy.prepare(installer.Selection.of(models, self.MODEL_ID + ":Q4_K_M"))
        self.api.assert_not_called()
        self.assertFalse(models.exists())

    def test_publish_refuses_to_replace_a_real_directory(self):
        snapshot, _ = self.package_fixture()
        destination = self.root / "occupied"
        destination.mkdir()
        with self.assertRaisesRegex(installer.ModelError, "non-symlink"):
            installer.link_selection(destination, snapshot)

    def test_legacy_packages_are_preserved_and_never_given_an_official_id(self):
        model_id = "incoai/Qwen3.6-35B-A3B-Splash"
        for kind in ("directory", "absolute link", "relative link"):
            with self.subTest(kind=kind):
                old, manifest = self.package_fixture(schema=4)
                manifest["draft"]["dummy"] = True
                self.write_manifest(old, manifest)
                models = self.root / f"models-{self.fixture_number}"
                models.mkdir()
                old_package = models / "qwen3.6-35b-a3b"
                if kind == "directory":
                    shutil.copytree(old, old_package)
                else:
                    old_package.symlink_to(
                        old
                        if kind == "absolute link"
                        else os.path.relpath(old, models),
                        target_is_directory=True,
                    )
                expected = (old_package / "manifest.json").read_bytes()
                official, _ = self.package_fixture(schema=4, model_id=model_id)
                self.configure_hub(official)
                args = installer.Selection.of(models, model_id)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    legacy.prepare(args)
                self.assertEqual((models / model_id).resolve(), official.resolve())
                self.assertEqual((old_package / "manifest.json").read_bytes(), expected)
                self.assertIn("missing artifacts will be downloaded", output.getvalue())
                with mock.patch.object(legacy, "resolve_snapshot") as download:
                    with contextlib.redirect_stdout(io.StringIO()):
                        legacy.prepare(args)
                    download.assert_not_called()

    def test_existing_link_to_a_different_repository_is_replaced_after_verification(
        self,
    ):
        old, _ = self.package_fixture(model_id="old/model")
        official, _ = self.package_fixture()
        self.configure_hub(official)
        models = self.root / "models"
        installer.link_selection(models / self.MODEL_ID, old)
        with contextlib.redirect_stdout(io.StringIO()):
            legacy.prepare(installer.Selection.of(models, self.MODEL_ID))
        self.assertEqual((models / self.MODEL_ID).resolve(), official.resolve())
        self.assertTrue(old.is_dir())
        self.assertFalse((old.parent.parent / "refs").exists())

    def test_unsupported_installed_format_is_replaced_with_verified_snapshot(self):
        old, manifest = self.package_fixture(revision="b" * 40)
        manifest["format"]["name"] = "unsupported-packed-q4"
        self.write_manifest(old, manifest)
        current, _ = self.package_fixture()
        self.configure_hub(current)
        models = self.root / "models"
        destination = models / self.MODEL_ID
        installer.link_selection(destination, old)
        with contextlib.redirect_stdout(io.StringIO()):
            legacy.prepare(installer.Selection.of(models, self.MODEL_ID))
        self.assertEqual(destination.resolve(), current.resolve())
        self.assertTrue(old.is_dir())
        self.assertEqual(json.loads((old / "manifest.json").read_text()), manifest)

    def test_unidentified_real_directory_is_preserved_without_downloading(self):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"
        destination = models / self.MODEL_ID
        shutil.copytree(snapshot, destination)
        with self.assertRaisesRegex(installer.ModelError, "move it aside"):
            legacy.prepare(installer.Selection.of(models, self.MODEL_ID))
        self.download.assert_not_called()
        self.assertTrue(destination.is_dir())

    def test_existing_canonical_snapshot_repairs_missing_ref_offline(self):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"
        installer.link_selection(models / self.MODEL_ID, snapshot)
        with mock.patch.object(legacy, "resolve_snapshot") as download:
            with contextlib.redirect_stdout(io.StringIO()):
                legacy.prepare(installer.Selection.of(models, self.MODEL_ID))
            download.assert_not_called()
        refs = list((snapshot.parent.parent / "refs/splash").glob("*/*"))
        self.assertEqual([ref.read_text() for ref in refs], [self.REVISION])

    def test_shared_hub_cache_keeps_each_installation_revision_pinned(self):
        from huggingface_hub import scan_cache_dir

        first, _ = self.package_fixture()
        second, _ = self.package_fixture(revision="b" * 40)
        second = second.rename(first.parent / second.name)
        cache = first.parent.parent.parent
        refs = first.parent.parent / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text("b" * 40)
        install_a = self.root / "install-a" / self.MODEL_ID
        install_b = self.root / "install-b" / self.MODEL_ID
        pin_a = hub.pin(first, self.MODEL_ID, install_a)
        pin_b = hub.pin(second, self.MODEL_ID, install_b)
        self.assertNotEqual(pin_a.parent, pin_b.parent)
        self.assertEqual((refs / "main").read_text(), "b" * 40)
        scanned = scan_cache_dir(cache)
        self.assertFalse(scanned.warnings)
        revisions = next(iter(scanned.repos)).revisions
        self.assertEqual(len(revisions), 2)
        self.assertTrue(all(revision.refs for revision in revisions))

    def test_existing_ref_does_not_require_cache_writes(self):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"
        destination = models / self.MODEL_ID
        installer.link_selection(destination, snapshot)
        ref = hub.pin(snapshot, self.MODEL_ID, destination)
        with mock.patch.object(
            hub.os, "link", side_effect=AssertionError("cache write")
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                legacy.prepare(installer.Selection.of(models, self.MODEL_ID))
        self.assertEqual(ref.read_text(), self.REVISION)

    def test_verified_model_starts_when_cache_ref_cannot_be_written(self):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"
        destination = models / self.MODEL_ID
        installer.link_selection(destination, snapshot)
        for code in (errno.EACCES, errno.EPERM, errno.EROFS):
            with self.subTest(errno=code):
                errors = io.StringIO()
                with mock.patch.object(
                    hub.os, "link", side_effect=OSError(code, "read only")
                ):
                    with (
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(errors),
                    ):
                        legacy.prepare(installer.Selection.of(models, self.MODEL_ID))
                self.assertEqual(destination.resolve(), snapshot.resolve())
                self.assertIn("external cache pruning", errors.getvalue())
        self.download.assert_not_called()

    def test_new_install_requires_ref_before_publishing(self):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"
        self.configure_hub(snapshot)
        with mock.patch.object(
            hub,
            "pin",
            side_effect=OSError(errno.EROFS, "read only"),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(OSError):
                    legacy.prepare(installer.Selection.of(models, self.MODEL_ID))
        self.assertFalse((models / self.MODEL_ID).exists())

    def test_invalid_existing_ref_is_not_ignored(self):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"
        destination = models / self.MODEL_ID
        installer.link_selection(destination, snapshot)
        ref = hub.pin(snapshot, self.MODEL_ID, destination)
        ref.write_text("wrong")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(
                installer.ModelError, "invalid installed snapshot reference"
            ):
                legacy.prepare(installer.Selection.of(models, self.MODEL_ID))

    def test_retiring_old_pin_failure_keeps_verified_installation_usable(self):
        snapshot, _ = self.package_fixture()
        models = self.root / "models"
        destination = models / self.MODEL_ID
        installer.link_selection(destination, snapshot)
        ref = hub.pin(snapshot, self.MODEL_ID, destination)
        old = ref.parent / ("b" * 40)
        old.write_text("b" * 40)
        errors = io.StringIO()
        with mock.patch.object(
            Path, "unlink", side_effect=PermissionError(errno.EACCES, "read only")
        ):
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(errors),
            ):
                legacy.prepare(installer.Selection.of(models, self.MODEL_ID))
        self.assertEqual(ref.read_text(), self.REVISION)
        self.assertTrue(old.exists())
        self.assertIn("could not retire", errors.getvalue())

    def test_updating_one_installation_retires_only_its_old_pin(self):
        first, _ = self.package_fixture()
        second, _ = self.package_fixture(revision="b" * 40)
        second = second.rename(first.parent / second.name)
        models = self.root / "install-a"
        destination = models / self.MODEL_ID
        other = self.root / "install-b" / self.MODEL_ID
        old_pin = hub.pin(first, self.MODEL_ID, destination)
        other_pin = hub.pin(first, self.MODEL_ID, other)
        installer.link_selection(destination, second)
        with contextlib.redirect_stdout(io.StringIO()):
            legacy.prepare(installer.Selection.of(models, self.MODEL_ID))
        self.assertFalse(old_pin.exists())
        self.assertTrue(other_pin.exists())
        self.assertEqual([p.read_text() for p in old_pin.parent.iterdir()], ["b" * 40])

    def test_main_update_retries_before_downloading_any_weights(self):
        current, _ = self.package_fixture(revision="b" * 40)
        self.configure_hub(current)
        info = self.api.return_value.model_info.return_value
        self.api.return_value.model_info.side_effect = [
            SimpleNamespace(sha=self.REVISION, siblings=info.siblings),
            SimpleNamespace(sha="b" * 40, siblings=info.siblings),
        ]
        self.assertEqual(legacy.resolve_snapshot(self.MODEL_ID), current.resolve())
        self.assertEqual(self.manifest_download.call_count, 2)
        self.assertEqual(self.download.call_count, 1)
        self.assertEqual(self.download.call_args.kwargs["revision"], "b" * 40)
        self.download.reset_mock()
        self.api.return_value.model_info.side_effect = None
        self.api.return_value.model_info.return_value = SimpleNamespace(
            sha=self.REVISION, siblings=info.siblings
        )
        with self.assertRaisesRegex(installer.ModelError, "main changed repeatedly"):
            legacy.resolve_snapshot(self.MODEL_ID)
        self.download.assert_not_called()

    def test_authentication_failure_is_actionable_and_does_not_change_identity(self):
        for explicit in (False, True):
            for status in (401, 403, 404, 500):
                with self.subTest(explicit=explicit, status=status):
                    self.api.reset_mock()
                    self.api.return_value.model_info.side_effect = http_error(status)
                    environment = {"HF_TOKEN": "test-explicit"} if explicit else {}
                    with (
                        mock.patch.dict(os.environ, environment, clear=True),
                        mock.patch(
                            "huggingface_hub.get_token", return_value="test-login"
                        ) as login,
                    ):
                        with self.assertRaises(installer.ModelError) as raised:
                            legacy.resolve_snapshot(self.MODEL_ID)
                    if explicit:
                        login.assert_not_called()
                    else:
                        login.assert_called_once()
                    self.api.assert_called_once_with(
                        token="test-explicit" if explicit else "test-login"
                    )
                    if status in (401, 403):
                        self.assertIn("hf auth login", str(raised.exception))

    def test_installation_lock_is_exclusive(self):
        with mock.patch.object(installer.fcntl, "flock") as flock:
            with installer.installation_lock(self.root):
                pass
        self.assertEqual(
            [call.args[1] for call in flock.call_args_list],
            [
                installer.fcntl.LOCK_EX | installer.fcntl.LOCK_NB,
                installer.fcntl.LOCK_UN,
            ],
        )


class ScriptEntryTests(unittest.TestCase):
    """The launcher runs install/models.py as a script: errors raised in the
    other installer modules must reach the user as one line, as they do
    through package imports."""

    def run_script(self, *arguments):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(installer.__file__).resolve()),
                    "--models",
                    str(root / "models"),
                    *arguments,
                ],
                env={
                    "HOME": str(root / "home"),
                    "HF_HUB_OFFLINE": "1",
                    "HF_HUB_CACHE": str(root / "hub"),
                },
                capture_output=True,
                text=True,
                timeout=60,
            )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertNotIn("Traceback", output)
        return output

    def test_an_uncached_model_offline_is_one_error_line(self):
        output = self.run_script("--model", "someone/not-cached", "prepare")
        self.assertIn("error: cannot resolve someone/not-cached", output)

    def test_verifying_nothing_installed_is_one_error_line(self):
        output = self.run_script("--model", "someone/model:Q4_K_M", "verify")
        self.assertIn("error: someone/model:Q4_K_M is not installed in ", output)

    def test_install_entry_points_run_as_scripts(self):
        # As the launcher, make and spawn_refresh run them: by path, from
        # any working directory.
        install = Path(installer.__file__).resolve().parent
        for script in ("catalog.py", "models.py"):
            with (
                self.subTest(script=script),
                tempfile.TemporaryDirectory() as temporary,
            ):
                result = subprocess.run(
                    [sys.executable, str(install / script), "--help"],
                    cwd=temporary,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

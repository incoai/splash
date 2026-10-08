"""A local projector override participates in the ordinary model assembly."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.install.test_gguf_metadata import (
    GGML,
    fixture,
    loadable_tensors,
    vision_fixture,
    write_gguf,
)
from dev.tests.installer_fixtures import MOE, FakeHub, draft_dir, mlx_target
from install import assembly, launcher, legacy, models, upstream


class MmprojTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.library = self.root / "library"
        self.model = "community/Test-GGUF"
        self.fake = FakeHub(self, self.root / "hub")
        draft_dir(self.library / MOE.draft_repo, MOE)
        self.projector = self.projector_file(self.root / "external vision.gguf")

    def projector_file(self, path, *, changes=None, kind=GGML["BF16"]):
        values = (
            vision_fixture()
            | {"clip.vision.attention.layer_norm_epsilon": 1e-6}
            | (changes or {})
        )
        return write_gguf(path, values, [("v.patch_embd.weight", kind)])

    def target(self, root):
        root.mkdir(parents=True, exist_ok=True)
        values = fixture(native=True)
        write_gguf(
            root / "model.gguf", values, loadable_tensors(values, self.root).items()
        )

    def selection(self, **options):
        return models.Selection.of(
            self.root / "models",
            self.model,
            **({"model_dir": self.library, "mmproj": self.projector} | options),
        )

    def prepare(self, chosen):
        with contextlib.redirect_stdout(io.StringIO()):
            upstream.prepare(chosen)
        return assembly.verify(chosen.link, full=True)

    def test_local_model_uses_external_projector_without_network_or_source_edits(self):
        self.target(self.library / self.model)
        before = self.projector.read_bytes()
        chosen = self.selection()
        record = self.prepare(chosen)
        self.assertEqual(record["vision_format"], "gguf")
        self.assertEqual((chosen.link / "vision/mmproj.gguf").resolve(), self.projector)
        config = json.loads((chosen.link / "config.json").read_text())
        self.assertEqual(config["vision_config"]["out_hidden_size"], 2048)
        self.assertEqual(self.projector.read_bytes(), before)
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(self.fake.downloads, [])

    def test_override_wins_over_bundled_projector(self):
        target = self.library / self.model
        self.target(target)
        self.projector_file(target / "mmproj-F16.gguf", kind=GGML["F16"])
        chosen = self.selection()
        self.prepare(chosen)
        self.assertEqual((chosen.link / "vision/mmproj.gguf").resolve(), self.projector)

    def test_remote_target_uses_local_projector_without_downloading_vision(self):
        self.fake.publish(self.model, "a" * 40, self.target)
        draft = self.library / MOE.draft_repo
        chosen = self.selection(model_dir=None, draft_model=str(draft))
        self.prepare(chosen)
        self.assertEqual((chosen.link / "vision/mmproj.gguf").resolve(), self.projector)
        self.assertEqual(self.fake.downloads, [self.model + "/model.gguf"])

    def test_unchanged_projector_reuses_assembly_and_replacement_rebuilds_it(self):
        self.target(self.library / self.model)
        chosen = self.selection()
        self.prepare(chosen)
        original = chosen.link.resolve()
        self.prepare(chosen)
        self.assertEqual(chosen.link.resolve(), original)
        self.projector_file(self.projector, kind=GGML["F32"])
        self.prepare(chosen)
        self.assertNotEqual(chosen.link.resolve(), original)

    def test_distinct_projectors_have_distinct_selection_links(self):
        other = self.projector_file(self.root / "other.gguf")
        self.assertNotEqual(self.selection().link, self.selection(mmproj=other).link)
        self.assertNotEqual(self.selection().link, self.selection(mmproj=None).link)
        alias = self.root / "alias.gguf"
        alias.symlink_to(self.projector)
        self.assertEqual(self.selection().link, self.selection(mmproj=alias).link)

    def test_missing_override_is_rejected(self):
        with self.assertRaisesRegex(models.ModelError, "mmproj.*file"):
            self.selection(mmproj=self.root / "missing.gguf")

    def test_language_only_conflicts_with_override(self):
        with self.assertRaisesRegex(models.ModelError, "mmproj.*language-only"):
            self.selection(language_only=True)

    def test_invalid_projector_is_rejected_before_weight_downloads(self):
        self.fake.publish(self.model, "a" * 40, self.target)
        for changes, kind in (
            ({"general.architecture": "qwen35"}, GGML["BF16"]),
            ({"clip.vision.projection_dim": 5120}, GGML["BF16"]),
            ({"clip.vision.block_count": 26}, GGML["BF16"]),
            ({"clip.vision.attention.layer_norm_epsilon": 1e-5}, GGML["BF16"]),
            ({"clip.vision.is_deepstack_layers": [False] * 2}, GGML["BF16"]),
            ({"clip.vision.is_deepstack_layers": [""] * 27}, GGML["BF16"]),
            ({}, GGML["F16"]),
            ({}, GGML["Q8_0"]),
        ):
            with self.subTest(changes=changes, kind=kind):
                self.projector_file(self.projector, changes=changes, kind=kind)
                chosen = self.selection(
                    model_dir=None, draft_model=str(self.library / MOE.draft_repo)
                )
                with self.assertRaises(models.ModelError):
                    self.prepare(chosen)
                self.assertFalse(chosen.link.exists())
                self.assertEqual(self.fake.downloads, [])

    def test_mlx_target_refuses_gguf_projector_override(self):
        mlx_target(self.library / self.model, MOE)
        with self.assertRaisesRegex(models.ModelError, "mmproj.*GGUF"):
            self.prepare(self.selection())

    def test_cli_projector_path_survives_installer_handoff(self):
        args = launcher.parse_args(
            ["serve", "--model", self.model, "--mmproj", str(self.projector)]
        )
        chosen = models.Selection.of(
            self.root / "models", args.model, mmproj=args.mmproj
        )
        command = launcher._model_command(chosen, "prepare")
        parsed = models.parse_args(command[2:])
        self.assertEqual(Path(parsed.mmproj), self.projector)
        self.assertEqual(parsed.command, "prepare")

    def test_serve_hands_projector_to_model_preparation(self):
        with (
            mock.patch.object(launcher, "RUNTIME_DIR", self.root / "runtime"),
            mock.patch.object(launcher, "_check_port"),
            mock.patch.object(launcher, "_ensure_installed") as install,
            mock.patch.object(launcher.os, "execve"),
            mock.patch.object(launcher.catalog, "spawn_refresh"),
            mock.patch.object(launcher.signal, "signal"),
            mock.patch.object(launcher.signal, "pthread_sigmask"),
        ):
            launcher.main(
                ["serve", "--model", self.model, "--mmproj", str(self.projector)]
            )
        self.assertEqual(install.call_args.args[0].mmproj, self.projector)

    def test_legacy_package_rejects_override(self):
        with self.assertRaisesRegex(models.ModelError, "upstream model ID"):
            legacy.prepare(self.selection())

    def test_cli_reports_missing_file_and_conflicting_options(self):
        for arguments in (
            ["--mmproj", str(self.root / "missing.gguf")],
            ["--mmproj", str(self.projector), "--language-only"],
        ):
            for parse, command in (
                (launcher.parse_args, ["serve", "--model", self.model, *arguments]),
                (models.parse_args, ["--model", self.model, *arguments, "prepare"]),
            ):
                with (
                    self.subTest(command=command),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as stopped:
                        parse(command)
                    self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

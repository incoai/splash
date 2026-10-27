import fcntl
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from desktop import model_tools
from desktop.model_tools import remove
from install import models


class ModelToolsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "models"
        assembly = self.root / ".resolved" / "shared"
        assembly.mkdir(parents=True)
        (assembly / "model.json").write_text("{}")
        self.assembly = assembly
        for name in ("first", "second"):
            link = self.root / "owner" / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(assembly)

    def test_remove_one_selection_keeps_shared_assembly(self):
        remove(self.root, "owner/first")
        self.assertFalse((self.root / "owner/first").is_symlink())
        self.assertTrue((self.root / "owner/second").is_symlink())
        self.assertTrue(self.assembly.is_dir())

    def test_held_assembly_survives_last_selection_removal(self):
        remove(self.root, "owner/first")
        with (self.assembly / "model.json").open("rb") as held:
            fcntl.flock(held, fcntl.LOCK_SH)
            remove(self.root, "owner/second")
            self.assertTrue(self.assembly.is_dir())

    def test_refuses_non_link(self):
        link = self.root / "owner" / "first"
        link.unlink()
        link.mkdir()
        with self.assertRaises(models.ModelError):
            remove(self.root, "owner/first")
        self.assertTrue(link.is_dir())

    def test_remove_command_keeps_existing_cli(self):
        subprocess.run(
            [sys.executable, "-m", "desktop.model_tools", "--models", str(self.root),
             "--model", "owner/first", "remove"],
            cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True, text=True,
        )
        self.assertFalse((self.root / "owner/first").is_symlink())
        self.assertTrue((self.root / "owner/second").is_symlink())

    def test_estimate_gguf_counts_only_selected_quantization_and_split_parts(self):
        sizes = {
            "Model-UD-Q4_K_M.gguf": 10,
            "Model-UD-Q5_K_M.gguf": 12,
            "UD-Q8_K_XL/Model-UD-Q8_K_XL-00001-of-00002.gguf": 20,
            "UD-Q8_K_XL/Model-UD-Q8_K_XL-00002-of-00002.gguf": 21,
        }
        repo = SimpleNamespace(files=set(sizes), sizes={name: (size, None) for name, size in sizes.items()})
        with patch("desktop.model_tools.hub.Repository.resolve", return_value=repo):
            result = model_tools.estimate_sizes("GGUF", "owner/Model-GGUF", ["UD-Q4_K_M", "UD-Q8_K_XL"])
        self.assertEqual(result, {"UD-Q4_K_M": 10, "UD-Q8_K_XL": 41})

    def test_estimate_mlx_queries_each_quantization_repository(self):
        repos = {
            "owner/Model-4bit": SimpleNamespace(files={"model.safetensors", "config.json"},
                                                 sizes={"model.safetensors": (30, None),
                                                        "config.json": (2, None)}),
            "owner/Model-8bit": SimpleNamespace(files={"model.safetensors"},
                                                 sizes={"model.safetensors": (60, None)}),
        }
        with patch("desktop.model_tools.hub.Repository.resolve", side_effect=repos.__getitem__) as resolve:
            result = model_tools.estimate_sizes("MLX", "owner/Model", ["4bit", "8bit"])
        self.assertEqual(result, {"4bit": 32, "8bit": 60})
        self.assertEqual([call.args[0] for call in resolve.call_args_list],
                         ["owner/Model-4bit", "owner/Model-8bit"])

    def test_plan_creates_one_draft_task_only_for_missing_files(self):
        config = Path(self.directory.name) / "config.json"
        config.write_text("{}")
        target_repo = SimpleNamespace(revision="target-abc", files=set())
        target = SimpleNamespace(family=object(), format="gguf", vision_format="none",
                                 config={}, gguf_metadata={})
        draft = SimpleNamespace(name="owner/shared-draft", revision="abc", paths=None,
                                files={"config.json", "model.safetensors"},
                                sizes={"config.json": (2, "config"),
                                       "model.safetensors": (500, "weights")},
                                file=lambda _: config)
        with (patch("desktop.model_tools.hub.Repository.resolve", return_value=target_repo),
              patch("install.upstream.inspect_target", return_value=target),
              patch("install.upstream._resolve_draft", return_value=draft),
              patch("install.upstream._weight_files", return_value={"model.safetensors"}),
              patch("install.upstream.check_model"),
              patch("desktop.model_tools.hub.folder", return_value=Path(self.directory.name)),
              patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(
                  try_to_load_from_cache=lambda _repo, name, revision:
                      str(config) if name == "config.json" else None)})):
            plan = model_tools.download_plan(self.root, "owner/first", False)
        self.assertEqual(plan, {"model_revision": "target-abc",
                                "draft_repository": "owner/shared-draft",
                                "draft_revision": "abc", "draft_needed": True,
                                "draft_size_bytes": 500})

    def test_fetches_draft_and_model_files_separately(self):
        draft = SimpleNamespace(files={"config.json", "model.safetensors"}, download=Mock())
        model = SimpleNamespace(download=Mock())
        target = SimpleNamespace(files={"target/model.gguf": "model.gguf"})
        with (patch("desktop.model_tools.hub.Repository.resolve", side_effect=[draft, model]),
              patch("install.upstream._weight_files", return_value={"model.safetensors"}),
              patch("install.upstream.inspect_target", return_value=target)):
            model_tools.fetch_draft("owner/shared-draft", "abc")
            model_tools.fetch_model(self.root, "owner/first", False, "target-abc")
        draft.download.assert_called_once_with({"config.json", "model.safetensors"})
        model.download.assert_called_once_with({"model.gguf"})

    def test_assemble_uses_pinned_cached_revisions(self):
        target = SimpleNamespace(files=set())
        draft = SimpleNamespace()
        with (patch("desktop.model_tools.hub.Repository.cached", side_effect=[target, draft]) as cached,
              patch("install.upstream._install") as install):
            model_tools.assemble_model(self.root, "owner/first", False,
                                       "target-abc", "owner/shared-draft", "draft-abc")
        self.assertEqual([call.args for call in cached.call_args_list],
                         [("owner/first", "target-abc"), ("owner/shared-draft", "draft-abc")])
        self.assertIs(install.call_args.args[1], target)
        self.assertIs(install.call_args.args[3], draft)

    def test_plan_rejects_legacy_package_before_fetching(self):
        manifest = Path(self.directory.name) / "manifest.json"
        manifest.write_text('{"format":{"name":"splash-packed-q4"}}')
        target = SimpleNamespace(files={"manifest.json"}, file=lambda _: manifest)
        with (patch("desktop.model_tools.hub.Repository.resolve", return_value=target),
              patch("install.upstream.inspect_target") as inspect):
            with self.assertRaises(models.ModelError):
                model_tools.download_plan(self.root, "owner/first", False)
        inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()

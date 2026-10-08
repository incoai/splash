"""External model libraries download physical, verified files without replacing user data."""

import errno
import hashlib
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from install import hub, library, models


class LibraryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "model library"
        self.content = {
            "config.json": b"{}",
            "sub/model.gguf": b"weights",
            "other.gguf": b"other",
        }
        self.repo = hub.Repository(
            "owner/model",
            "a" * 40,
            set(self.content),
            sizes={
                name: (len(data), hashlib.sha256(data).hexdigest())
                for name, data in self.content.items()
            },
        )
        self.calls = []
        patch = mock.patch("huggingface_hub.hf_hub_download", side_effect=self.download)
        self.fetch = patch.start()
        self.addCleanup(patch.stop)

    def download(self, repo_id, filename, *, revision, local_dir):
        self.calls.append((repo_id, filename, revision, Path(local_dir)))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.content[filename])
        return str(path)

    def test_selected_files_are_physical_pinned_and_reused(self):
        result = library.materialize(
            self.repo, {"config.json", "sub/model.gguf"}, self.root
        )
        self.assertEqual(result.directory, self.root / "owner/model")
        self.assertEqual(set(result.files), {"config.json", "sub/model.gguf"})
        self.assertEqual(library.revision(result.directory), self.repo.revision)
        self.assertTrue(all(call[2] == self.repo.revision for call in self.calls))
        self.assertTrue(
            all(self.root / ".splash" in call[3].parents for call in self.calls)
        )
        self.assertFalse((result.directory / "sub/model.gguf").is_symlink())
        library.materialize(self.repo, {"sub/model.gguf"}, self.root)
        self.assertEqual(self.fetch.call_count, 2)
        library.materialize(self.repo, {"other.gguf"}, self.root)
        self.assertEqual(library.revision(result.directory), self.repo.revision)

    def test_git_blob_hash_supported(self):
        data = self.content["config.json"]
        repo = hub.Repository(
            self.repo.name,
            self.repo.revision,
            {"config.json"},
            sizes={
                "config.json": (len(data), hashlib.sha1(b"blob 2\0" + data).hexdigest())
            },
        )
        library.materialize(repo, {"config.json"}, self.root)

    def test_cached_files_copied_without_network(self):
        cache = self.root.parent / "snapshot"
        cache.mkdir()
        (cache / "config.json").write_bytes(b"{}")
        repo = hub.Repository(
            self.repo.name, self.repo.revision, {"config.json"}, cache
        )
        result = library.materialize(repo, {"config.json"}, self.root)
        self.fetch.assert_not_called()
        self.assertFalse((result.directory / "config.json").is_symlink())
        self.assertEqual(library.revision(result.directory), self.repo.revision)

    def test_existing_mismatch_preserved(self):
        target = self.root / "owner/model/config.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"user data")
        with self.assertRaisesRegex(models.ModelError, "mismatch|different"):
            library.materialize(self.repo, {"config.json"}, self.root)
        self.assertEqual(target.read_bytes(), b"user data")
        self.fetch.assert_not_called()

    def test_corrupt_download_not_published(self):
        self.content["config.json"] = b"broken"
        with self.assertRaisesRegex(models.ModelError, "mismatch|different"):
            library.materialize(self.repo, {"config.json"}, self.root)
        self.assertFalse((self.root / "owner/model/config.json").exists())

    def test_changed_or_extra_files_remove_revision_claim(self):
        directory = library.materialize(self.repo, {"config.json"}, self.root).directory
        (directory / "extra").write_bytes(b"unknown")
        self.assertIsNone(library.revision(directory))
        (directory / "extra").unlink()
        (directory / "config.json").write_bytes(b"xx")
        self.assertIsNone(library.revision(directory))

    def test_unmanaged_missing_and_corrupt_provenance(self):
        self.assertIsNone(library.revision(self.root / "owner/model"))
        directory = library.materialize(self.repo, {"config.json"}, self.root).directory
        record = next((self.root / ".splash").glob("*.json"))
        record.write_text("{")
        self.assertIsNone(library.revision(directory))

    def test_refuse_revision_mixing(self):
        library.materialize(self.repo, {"config.json"}, self.root)
        newer = hub.Repository(
            self.repo.name, "b" * 40, self.repo.files, sizes=self.repo.sizes
        )
        with self.assertRaisesRegex(models.ModelError, "revision|commit"):
            library.materialize(newer, {"other.gguf"}, self.root)

    def test_symlink_components_and_files_rejected(self):
        outside = self.root.parent / "outside"
        outside.mkdir()
        for component in ("owner", "owner/model", ".splash", "owner/model/config.json"):
            with self.subTest(component=component):
                root = self.root / str(len(component))
                path = root / component
                path.parent.mkdir(parents=True)
                path.symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(models.ModelError, "symlink"):
                    library.materialize(self.repo, {"config.json"}, root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_unsafe_names_and_unpinned_sources_rejected(self):
        for name in ("../escape", "/escape", ".cache/file"):
            with self.subTest(name=name), self.assertRaises(models.ModelError):
                repo = hub.Repository(self.repo.name, self.repo.revision, {name})
                library.materialize(repo, {name}, self.root)
        with self.assertRaisesRegex(models.ModelError, "commit"):
            library.materialize(
                hub.Repository(self.repo.name, "main", self.repo.files),
                {"config.json"},
                self.root,
            )

    def test_interrupted_fetch_can_be_retried(self):
        self.fetch.side_effect = OSError("interrupted")
        with self.assertRaises(models.ModelError):
            library.materialize(self.repo, {"config.json"}, self.root)
        self.assertFalse((self.root / "owner/model/config.json").exists())
        self.fetch.side_effect = self.download
        library.materialize(self.repo, {"config.json"}, self.root)
        self.assertEqual(
            library.revision(self.root / "owner/model"), self.repo.revision
        )

    def test_missing_remote_hash_is_rejected(self):
        repo = hub.Repository(self.repo.name, self.repo.revision, self.repo.files)
        with self.assertRaisesRegex(models.ModelError, "metadata|hash"):
            library.materialize(repo, {"config.json"}, self.root)
        self.fetch.assert_not_called()

    def test_concurrent_materialization_fetches_once(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda _: library.materialize(
                        self.repo, {"config.json"}, self.root
                    ),
                    range(2),
                )
            )
        self.assertEqual(self.fetch.call_count, 1)
        self.assertEqual(results[0].directory, results[1].directory)
        self.assertEqual(library.revision(results[0].directory), self.repo.revision)

    def test_same_size_hash_mismatch_preserved(self):
        target = self.root / "owner/model/config.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"xx")
        with self.assertRaisesRegex(models.ModelError, "hash mismatch"):
            library.materialize(self.repo, {"config.json"}, self.root)
        self.assertEqual(target.read_bytes(), b"xx")

    def test_read_only_library_has_actionable_error(self):
        with mock.patch(
            "install.library.os.open",
            side_effect=OSError(errno.EROFS, "Read-only file system"),
        ):
            with self.assertRaisesRegex(models.ModelError, "cannot install.*Read-only"):
                library.materialize(self.repo, {"config.json"}, self.root)

    def test_download_returning_cache_path_rejected(self):
        self.fetch.side_effect = lambda *args, **kwargs: "/unexpected/cache/file"
        with self.assertRaisesRegex(models.ModelError, "requested local directory"):
            library.materialize(self.repo, {"config.json"}, self.root)

    def test_invalid_provenance_blocks_new_downloads(self):
        library.materialize(self.repo, {"config.json"}, self.root)
        path = next((self.root / ".splash").glob("*.json"))
        path.write_text("{}")
        with self.assertRaisesRegex(models.ModelError, "provenance"):
            library.materialize(self.repo, {"other.gguf"}, self.root)

    def test_changed_stat_invalidates_revision_even_if_size_unchanged(self):
        directory = library.materialize(self.repo, {"config.json"}, self.root).directory
        (directory / "config.json").write_bytes(b"xx")
        self.assertIsNone(library.revision(directory))

    def test_empty_selection_rejected(self):
        with self.assertRaisesRegex(models.ModelError, "no model files"):
            library.materialize(self.repo, set(), self.root)

    def test_cached_blob_digest_rejects_corrupted_source(self):
        for algorithm in ("sha256", "git-sha1"):
            with self.subTest(algorithm=algorithm):
                data = b"{}"
                digest = (
                    hashlib.sha256(data).hexdigest()
                    if algorithm == "sha256"
                    else hashlib.sha1(b"blob 2\0" + data).hexdigest()
                )
                cache = self.root.parent / algorithm / "models--owner--model"
                blob = cache / "blobs" / digest
                blob.parent.mkdir(parents=True)
                blob.write_bytes(b"xx")
                snapshot = cache / "snapshots" / self.repo.revision
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").symlink_to(blob)
                repo = hub.Repository(
                    self.repo.name, self.repo.revision, {"config.json"}, snapshot
                )
                with self.assertRaisesRegex(models.ModelError, "hash mismatch"):
                    library.materialize(repo, {"config.json"}, self.root / algorithm)
                self.assertFalse(
                    (self.root / algorithm / "owner/model/config.json").exists()
                )
                blob.write_bytes(data)
                result = library.materialize(
                    repo, {"config.json"}, self.root / algorithm
                )
                self.assertEqual((result.directory / "config.json").read_bytes(), data)
        self.fetch.assert_not_called()

    def test_cached_symlink_without_blob_digest_rejected(self):
        cache = self.root.parent / "snapshot"
        cache.mkdir()
        source = self.root.parent / "unknown-blob"
        source.write_bytes(b"{}")
        (cache / "config.json").symlink_to(source)
        repo = hub.Repository(
            self.repo.name, self.repo.revision, {"config.json"}, cache
        )
        with self.assertRaisesRegex(models.ModelError, "hash metadata"):
            library.materialize(repo, {"config.json"}, self.root)


if __name__ == "__main__":
    unittest.main()

import contextlib
import dataclasses
import fcntl
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import (
    DENSE,
    DRAFT_COMMIT,
    FLASH_NEXT,
    GGML,
    MOE,
    PROCESSOR,
    FakeHub,
    cached_snapshot,
    draft_dir,
    family_config,
    fixture,
    loadable_tensors,
    mlx_target,
    pins,
    selection,
    vision_fixture,
    write_gguf,
)
from install import assembly, families, hub, local, models, upstream

GGUF_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"


class LocalModelTest(unittest.TestCase):
    """prepare_local_model, which must never reach the network: every way out
    of this process to the Hub, or anywhere, fails the test."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # Resolved, as prepare_local_model names the files it prepares.
        self.root = Path(temporary.name).resolve()
        self.models = self.root / "models"
        self.cache = self.root / "hub"
        # A scratch directory for the headers loadable_tensors writes.
        self.headers = self.root / "headers"
        self.headers.mkdir()

        def offline(name):
            def refuse(*args, **options):
                raise AssertionError(f"prepare_local_model used the network: {name}")

            return refuse

        for name, replacement in (
            ("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)),
            ("huggingface_hub.constants.HF_HUB_OFFLINE", False),
            ("socket.socket.connect", offline("socket.connect")),
            ("socket.create_connection", offline("socket.create_connection")),
            ("huggingface_hub.HfApi", offline("HfApi")),
            ("huggingface_hub.HfFileSystem", offline("HfFileSystem")),
            ("huggingface_hub.snapshot_download", offline("snapshot_download")),
            ("huggingface_hub.hf_hub_download", offline("hf_hub_download")),
        ):
            patch = mock.patch(name, replacement)
            patch.start()
            self.addCleanup(patch.stop)

    def prepare(self, target, draft=None, **options):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return local.prepare_local_model(
                target, draft, models_dir=self.models, **options
            )

    def gguf(self, path, family=MOE):
        """A loadable GGUF target of family at path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        header = fixture(native=True, family=family)
        return write_gguf(path, header, loadable_tensors(header, self.headers).items())

    @staticmethod
    def projector(path, kind="F32"):
        """A vision projector at path, of one tensor of kind."""
        path.parent.mkdir(parents=True, exist_ok=True)
        return write_gguf(path, vision_fixture(), [("v.patch_embd.weight", GGML[kind])])

    def cache_main(self, repo_id, commit):
        """Record commit as repo_id's main branch in the Hub cache, as a
        download of the branch does."""
        refs = self.cache / hub.folder_name(repo_id) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(commit)

    def split_gguf(self, directory, name, count):
        """A loadable GGUF target split into count parts, NAME-0000K-of-0000N."""
        directory.mkdir(parents=True, exist_ok=True)
        header = fixture(native=True)
        tensors = list(loadable_tensors(header, self.headers).items())
        split = {"split.count": count, "split.tensors.count": len(tensors)}
        for number in range(count):
            values = (header if number == 0 else {}) | split | {"split.no": number}
            write_gguf(
                directory / f"{name}-{number + 1:05d}-of-{count:05d}.gguf",
                values,
                tensors[number::count],
            )

    def test_a_gguf_and_its_draft_are_prepared_as_the_engine_loads_them(self):
        target = self.gguf(self.root / "downloads" / "model-Q4_K_M.gguf")
        draft = draft_dir(self.root / "draft", MOE)
        prepared = self.prepare(target, draft)
        record = assembly.verify(prepared, full=True)
        # The selection link of these files names the directory, which lies
        # with Splash's other installations.
        link = models.local_selection_link(self.models, target, draft, None, False)
        self.assertEqual(link.resolve(), prepared)
        self.assertEqual(prepared.parent, self.models / ".resolved")
        self.assertEqual(
            (record["model"], record["family"], record["target_format"]),
            (str(target), MOE.name, "gguf"),
        )
        self.assertEqual(record["vision_format"], "none")
        self.assertEqual(
            record["sources"],
            {
                "target": {"repo": str(target), "revision": None},
                "draft": {"repo": str(draft), "revision": None},
            },
        )
        self.assertEqual(
            sorted(record["files"]),
            [
                "config.json",
                "draft/config.json",
                "draft/model.safetensors",
                "target/model-Q4_K_M.gguf",
                "tokenizer/chat_template.jinja",
                "tokenizer/config.json",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
            ],
        )
        self.assertEqual((prepared / "target/model-Q4_K_M.gguf").readlink(), target)
        self.assertEqual(
            models.read_json(prepared / "config.json")["text_config"][
                "num_hidden_layers"
            ],
            40,
        )
        # The GGUF's directory names the same target: the projector beside it
        # is not loaded unless mmproj names it, and the files of a
        # subdirectory, such as a draft's, are not the target's.
        self.projector(target.parent / "mmproj-F32.gguf")
        nested = draft_dir(target.parent / "DFlash2", MOE)
        record = assembly.verify(self.prepare(target.parent, nested))
        self.assertIn("target/model-Q4_K_M.gguf", record["files"])
        self.assertEqual(record["vision_format"], "none")

    def test_a_gguf_projector_is_loaded_when_mmproj_names_it(self):
        target = self.gguf(self.root / "model" / "model-Q4_K_M.gguf")
        draft = draft_dir(self.root / "draft", MOE)
        # A projector may lie anywhere, under any name.
        mmproj = self.projector(self.root / "vision" / "tower.gguf")
        prepared = self.prepare(target, draft, mmproj=mmproj)
        record = assembly.verify(prepared, full=True)
        self.assertEqual(record["vision_format"], "gguf")
        self.assertEqual((prepared / assembly.GGUF_VISION).readlink(), mmproj)
        self.assertIn("vision_config", models.read_json(prepared / "config.json"))
        # Only a clip GGUF of BF16, F32 or F16 weights is one; and a model that
        # serves text only loads none.
        quantized = self.projector(self.root / "vision" / "q8.gguf", "Q8_0")
        text = self.gguf(self.root / "vision" / "text.gguf")
        for mmproj_path, refusal in (
            (
                quantized,
                rf"{quantized} \(clip: Q8_0\) is not a vision projector Splash "
                "loads.*; omit mmproj to serve text only",
            ),
            (text, rf"{text} \(qwen35moe: F32, Q4_K\) is not a vision projector"),
            (target, f"{target} is named as one of the target's files"),
        ):
            with (
                self.subTest(mmproj=mmproj_path.name),
                self.assertRaisesRegex(models.ModelError, refusal),
            ):
                self.prepare(target, draft, mmproj=mmproj_path)
        with self.assertRaisesRegex(
            models.ModelError, f"{mmproj} is a vision projector, which a model"
        ):
            self.prepare(target, draft, mmproj=mmproj, language_only=True)

    def test_without_a_draft_the_familys_is_taken_from_the_hub_cache(self):
        target = self.gguf(self.root / "model" / "model-Q4_K_M.gguf")
        remedy = (
            f"download {MOE.draft_repo}, or pass the directory of a {MOE.name} "
            "DFlash2 draft as draft"
        )
        with self.assertRaisesRegex(
            models.ModelError,
            f"cannot resolve {MOE.draft_repo}: no draft was given, and preparing "
            "local files asks no Hub; the Hub cache records no commit for the "
            f"default branch; {remedy}",
        ):
            self.prepare(target)
        cached_snapshot(
            self.cache, MOE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, MOE)
        )
        self.cache_main(MOE.draft_repo, DRAFT_COMMIT)
        prepared = self.prepare(target)
        self.assertEqual(
            assembly.verify(prepared)["sources"]["draft"],
            {"repo": MOE.draft_repo, "revision": DRAFT_COMMIT},
        )
        # Its snapshot is pinned, as an installation's are.
        self.assertEqual(pins(self.cache), [DRAFT_COMMIT])
        # A download of a newer draft moves the branch, which preparing again
        # follows: the newer draft is pinned in place of the older one, and
        # the directory it replaces is removed.
        newer = "e" * 40
        cached_snapshot(self.cache, MOE.draft_repo, newer, lambda p: draft_dir(p, MOE))
        self.cache_main(MOE.draft_repo, newer)
        self.assertEqual(
            assembly.verify(self.prepare(target))["sources"]["draft"],
            {"repo": MOE.draft_repo, "revision": newer},
        )
        self.assertEqual(pins(self.cache), [newer])
        self.assertFalse(prepared.exists())
        # A branch naming a commit the cache does not hold is refused: no
        # other commit stands in for it.
        missing = "f" * 40
        self.cache_main(MOE.draft_repo, missing)
        with self.assertRaisesRegex(
            models.ModelError, f"the Hub cache has no snapshot of {missing}; {remedy}"
        ):
            self.prepare(target)

    def test_a_family_without_a_released_draft_needs_a_draft(self):
        target = self.gguf(self.root / "model-Q2_0.gguf", FLASH_NEXT)
        unreleased = dataclasses.replace(FLASH_NEXT, draft_repo=None)
        with mock.patch.object(families, "FAMILIES", (unreleased,)):
            with self.assertRaisesRegex(
                models.ModelError,
                "no DFlash2 draft is released for Qwen3.8-Flash-Next; pass the "
                "directory of one as draft",
            ):
                self.prepare(target)
            record = assembly.verify(
                self.prepare(target, draft_dir(self.root / "draft", FLASH_NEXT))
            )
        self.assertEqual(record["family"], FLASH_NEXT.name)

    def test_an_mlx_target_serves_images_with_its_own_tower(self):
        shards = {
            "vision_tower.blocks.0.attn.qkv.weight": "model-00001-of-00002.safetensors",
            "language_model.model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "language_model.lm_head.weight": "model-00002-of-00002.safetensors",
        }
        target = mlx_target(self.root / "Qwen3.8-27B-4bit", DENSE)
        (target / "model.safetensors").unlink()
        (target / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": shards})
        )
        for name in set(shards.values()):
            (target / name).write_text(name)
        draft = draft_dir(self.root / "draft", DENSE)
        with self.assertRaisesRegex(
            models.ModelError,
            "is missing: preprocessor_config.json or processor_config.json. .*; "
            "pass language_only=True to serve text only",
        ):
            self.prepare(target, draft)
        # A processor Splash cannot follow refuses the vision alone, as does
        # a tower configuration the engine refuses.
        (target / "preprocessor_config.json").write_text(
            json.dumps(PROCESSOR | {"patch_size": 14})
        )
        with self.assertRaisesRegex(
            models.VisionError,
            "unsupported vision preprocessing configuration; pass "
            "language_only=True to serve text only",
        ):
            self.prepare(target, draft)
        (target / "preprocessor_config.json").write_text(json.dumps(PROCESSOR))
        config = json.loads((target / "config.json").read_text())
        tower = config["vision_config"] | {"depth": 1}
        (target / "config.json").write_text(
            json.dumps(config | {"vision_config": tower})
        )
        with self.assertRaisesRegex(
            models.VisionError,
            "vision config.*; pass language_only=True to serve text only",
        ):
            self.prepare(target, draft)
        (target / "config.json").write_text(json.dumps(config))
        prepared = self.prepare(target, draft)
        record = assembly.verify(prepared, full=True)
        self.assertEqual(
            (record["target_format"], record["vision_format"]),
            ("mlx-affine", "safetensors"),
        )
        self.assertEqual(
            sorted(p.name for p in (prepared / "vision").iterdir()),
            ["config.json", "model-00001-of-00002.safetensors"],
        )
        self.assertEqual(
            sorted(p.name for p in (prepared / "target").iterdir()),
            ["config.json", *sorted(set(shards.values()))],
        )
        self.assertEqual(
            sorted(p.name for p in (prepared / "tokenizer").iterdir()),
            ["config.json", "tokenizer.json", "tokenizer_config.json"],
        )
        self.assertEqual(
            models.read_json(prepared / "draft/config.json"),
            family_config(DENSE, "draft/config.json"),
        )
        # Text only, the tower is left out; mmproj names a GGUF's projector.
        record = assembly.verify(self.prepare(target, draft, language_only=True))
        self.assertEqual(record["vision_format"], "none")
        self.assertFalse(any(name.startswith("vision/") for name in record["files"]))
        mmproj = self.projector(self.root / "mmproj-F32.gguf")
        with self.assertRaisesRegex(
            models.ModelError,
            f"cannot prepare {target}: {mmproj} is a GGUF target's vision "
            "projector; an MLX target keeps its vision tower in its checkpoint",
        ):
            self.prepare(target, draft, mmproj=mmproj)

    def test_a_split_gguf_is_named_by_any_part_or_its_directory(self):
        directory = self.root / "UD-Q4_K_M"
        self.split_gguf(directory, "model-UD-Q4_K_M", 3)
        parts = [f"model-UD-Q4_K_M-0000{n}-of-00003.gguf" for n in (1, 2, 3)]
        draft = draft_dir(self.root / "draft", MOE)
        for target in (directory, directory / parts[1]):
            with self.subTest(target=target.name):
                prepared = self.prepare(target, draft)
                assembly.verify(prepared, full=True)
                self.assertEqual(
                    sorted(p.name for p in (prepared / "target").iterdir()), parts
                )
        # Every part must be there, and a directory must hold one target.
        (directory / parts[2]).unlink()
        with self.assertRaisesRegex(
            models.ModelError,
            f"the split GGUF model-UD-Q4_K_M in {directory} is incomplete: "
            f"{parts[0]}, {parts[1]}",
        ):
            self.prepare(directory / parts[0], draft)
        self.split_gguf(directory, "model-UD-Q4_K_M", 3)
        self.gguf(directory / "model-Q8_0.gguf")
        with self.assertRaisesRegex(
            models.ModelError,
            f"cannot prepare {directory}: it holds several GGUF targets "
            ".*model-Q8_0.gguf.*; pass the target's file",
        ):
            self.prepare(directory, draft)

    def test_files_splash_cannot_serve_are_refused_by_name(self):
        target = self.gguf(self.root / "model" / "model-Q4_K_M.gguf")
        draft = draft_dir(self.root / "draft", MOE)
        weights = self.root / "model" / "notes.txt"
        weights.write_text("")
        empty = self.root / "empty"
        empty.mkdir()
        for arguments, refusal in (
            (
                (self.root / "absent.gguf", draft),
                f"cannot prepare {self.root / 'absent.gguf'}: it does not exist",
            ),
            ((weights, draft), f"cannot prepare {weights}: it is not a GGUF file"),
            ((empty, draft), f"cannot prepare {empty}: it holds no model"),
            (
                (target, target),
                f"cannot prepare {target}: draft {target} is not a directory",
            ),
            (
                (target, draft_dir(self.root / "dense-draft", DENSE)),
                f"{self.root / 'dense-draft'}'s draft configuration is "
                f"incompatible with {MOE.name}: draft config hidden_size mismatch",
            ),
            ((target, empty), f"{empty} does not contain a DFlash2 checkpoint"),
        ):
            with (
                self.subTest(refusal=refusal),
                self.assertRaisesRegex(models.ModelError, refusal),
            ):
                self.prepare(*arguments)
        # A projector is a GGUF file, and a refusal of one names it; one that
        # is no GGUF refuses the vision alone.
        absent, junk = self.root / "absent.gguf", self.root / "junk.gguf"
        junk.write_bytes(b"not a GGUF")
        for mmproj, refusal, kind in (
            (absent, f"mmproj {absent} is not a file", models.ModelError),
            (empty, f"mmproj {empty} is not a file", models.ModelError),
            (
                junk,
                f"{junk}: unsupported GGUF header .*; omit mmproj to serve text only",
                models.VisionError,
            ),
        ):
            with (
                self.subTest(mmproj=mmproj.name),
                self.assertRaisesRegex(kind, f"cannot prepare {target}: {refusal}"),
            ):
                self.prepare(target, draft, mmproj=mmproj)
        # A checkpoint whose index names a shard that is not there.
        mlx = mlx_target(self.root / "mlx", DENSE)
        (mlx / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"lm_head.weight": "model-00002.safetensors"}})
        )
        with self.assertRaisesRegex(
            models.ModelError,
            "checkpoint has missing or unsupported shards: model-00002.safetensors",
        ):
            self.prepare(mlx, draft_dir(self.root / "dense", DENSE), language_only=True)
        # Nothing was published.
        self.assertFalse((self.models / ".resolved").exists())

    def test_preparing_again_reads_no_content_until_a_file_changes(self):
        target = self.gguf(self.root / "model" / "model-Q4_K_M.gguf")
        draft = draft_dir(self.root / "draft", MOE)
        prepared = self.prepare(target, draft)
        hashed = []
        hash_file = models.hash_file

        def counted(path, digest):
            hashed.append(Path(path).resolve())
            return hash_file(path, digest)

        sources = {target, draft / "config.json", draft / "model.safetensors"}
        with mock.patch.object(models, "hash_file", counted):
            self.assertEqual(self.prepare(target, draft), prepared)
            self.assertFalse(sources & set(hashed))
            # A new draft is prepared anew, and only its files are read; the
            # directory it replaces stays while a server holds it.
            _, server = assembly.hold(prepared)
            (draft / "model.safetensors").write_bytes(b"retrained")
            hashed.clear()
            again = self.prepare(target, draft)
        self.assertNotEqual(again, prepared)
        self.assertEqual(sources & set(hashed), {draft / "model.safetensors"})
        assembly.verify(again, full=True)
        self.assertTrue(prepared.exists())
        # Released, it goes with the next preparation, and a server started on
        # it is refused.
        server.close()
        self.assertEqual(self.prepare(target, draft), again)
        self.assertFalse(prepared.exists())
        with self.assertRaisesRegex(
            models.ModelError, f"{prepared} names no model Splash prepared"
        ):
            assembly.hold(prepared)

    def test_a_files_digest_is_read_before_the_installation_lock(self):
        # A file's first digest is read before the installation lock, which
        # installations and starting servers wait for; its record is taken
        # again under the lock, where a file written since is read again.
        target = self.gguf(self.root / "model" / "model-Q4_K_M.gguf")
        draft = draft_dir(self.root / "draft", MOE)
        weights = draft / "model.safetensors"
        reads = []
        hash_file = models.hash_file

        def read(path, digest):
            with (self.models / ".install.lock").open("a+b") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    reads.append((path, "locked"))
                else:
                    reads.append((path, "free"))
            value = hash_file(path, digest)
            if path == weights and (weights, "locked") not in reads:
                weights.write_bytes(b"written while it was read")
            return value

        with mock.patch.object(models, "hash_file", read):
            prepared = self.prepare(target, draft)
        for path, expected in (
            (target, ["free"]),
            (draft / "config.json", ["free"]),
            (weights, ["free", "locked"]),
        ):
            self.assertEqual([lock for read, lock in reads if read == path], expected)
        assembly.verify(prepared, full=True)

    def test_local_files_assemble_as_their_hub_installation_does(self):
        # The same files installed from a FakeHub and prepared from the Hub
        # cache the installation left are linked and recorded alike.
        def build(root):
            self.gguf(root / "model-Q4_K_M.gguf")
            self.projector(root / "mmproj-F32.gguf")

        fake = FakeHub(self, self.cache)
        fake.publish(GGUF_REPO, "a" * 40, build)
        fake.publish(MOE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, MOE))
        chosen = selection(self.root, GGUF_REPO + ":Q4_K_M", language_only=False)
        with contextlib.redirect_stdout(io.StringIO()):
            upstream.prepare(chosen)
        installed = assembly.verify(chosen.link)
        snapshot = fake.snapshot(GGUF_REPO, "a" * 40)
        requests, downloads = list(fake.requests), list(fake.downloads)

        def prepare():
            return self.prepare(
                snapshot / "model-Q4_K_M.gguf", mmproj=snapshot / "mmproj-F32.gguf"
            )

        # The installation downloaded its draft by commit, which records no
        # branch: the family's draft is the one a download of the branch left.
        with self.assertRaisesRegex(
            models.ModelError,
            "the Hub cache records no commit for the default branch",
        ):
            prepare()
        self.cache_main(MOE.draft_repo, DRAFT_COMMIT)
        prepared = assembly.verify(prepare())
        self.assertEqual((fake.requests, fake.downloads), (requests, downloads))
        for key in ("family", "target_format", "vision_format", "metadata", "files"):
            self.assertEqual(prepared[key], installed[key], key)
        self.assertEqual(prepared["sources"]["draft"], installed["sources"]["draft"])


if __name__ == "__main__":
    unittest.main()

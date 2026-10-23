"""MiniMax-H3's assemblies (install/video.py) on fake layouts of its
repositories: installed as they are downloaded, with no conversion."""

import contextlib
import dataclasses
import functools
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import FakeHub, cached_snapshot, pins, selection
from install import assembly, hub, local, models, upstream, video

COMMITS = {video.REPOSITORY: "1" * 40, video.ENCODER: "2" * 40, video.RELEASE: "3" * 40}
LICENSE = """MiniMax H3 Community License Agreement
5. “Excluded Territories” means the European Union, the United Kingdom, the Republic of Korea and the United States of America.
"""
MARKER = {
    "format": video.FORMAT,
    "version": video.VERSION,
    "weights": video.WEIGHTS,
    "variants": ["turbo", "default"],
    "sources": {video.RELEASE: COMMITS[video.RELEASE]},
}


def dit(variant):
    return {f"dit-{variant}/{name}": "" for name in
            ("vdn-w8-block00.safetensors", "prep.safetensors", video.ADALN, video.DIT_MANIFEST)}  # fmt: skip


IMAGES = {
    **{f"vae/{name}": "" for name in video.IMAGES["vae"]},
    "vae/vae_dec_f16.bin": "",
    "audio/audio_dec_f16.bin": "",
    "preview/taeh3_dec_f16.bin": "",
}
# Each repository's files, and some the installer must not download.
SOURCES = {
    video.REPOSITORY: {video.MARKER: json.dumps(MARKER), "LICENSE": LICENSE,
                       "NOTICE": "", "README.md": "", "tokenizer/tokenizer.json": "{}",
                       **IMAGES, **dit("turbo"), **dit("default")},
    video.ENCODER: {"config.json": "{}", "model.safetensors": "",
                    "model.safetensors.index.json": "{}"},
    video.RELEASE: {"LICENSE": LICENSE, video.RELEASE_INDEX: "{}",
                    "FL2VA/transformer/t.safetensors": ""},
}  # fmt: skip


def tree(root: Path, files):
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(data)
    return root


class VideoModelTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.cache, self.models = self.root / "hub", self.root / "models"
        # Nothing converts: no converter process, no engine run.
        refused = AssertionError("an installation converted")
        for patch in (
            mock.patch("subprocess.run", side_effect=refused),
            mock.patch("install.models.run_engine", side_effect=refused),
            mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)),
            mock.patch.dict(os.environ, {"SPLASH_ACCEPT_LICENSE": video.ACCEPT}),
        ):
            self.enterContext(patch)

    def hub(self):
        fake = FakeHub(self, self.cache)
        for repo, files in SOURCES.items():
            fake.publish(repo, COMMITS[repo], functools.partial(tree, files=files))
        return fake

    def install(self, variants=(), model=video.REPOSITORY):
        chosen = selection(self.root, model, language_only=False)
        chosen = dataclasses.replace(chosen, variants=variants)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            upstream.prepare(chosen)
        return chosen.link, output.getvalue()

    def prepare_local(self, target):
        with contextlib.redirect_stdout(io.StringIO()):
            return local.prepare_local_model(target, models_dir=self.models)

    def test_the_hub_installs_the_repository_as_downloaded(self):
        fake = self.hub()
        link, output = self.install()
        record = assembly.verify(link, full=True)
        self.assertEqual(record["target_format"], video.FORMAT)
        self.assertEqual(record["sources"], {
            "target": {"repo": video.REPOSITORY, "revision": COMMITS[video.REPOSITORY]},
            "encoder": {"repo": video.ENCODER, "revision": COMMITS[video.ENCODER]},
        })  # fmt: skip
        self.assertEqual(record["video"]["variants"], ["turbo"])
        snapshot = fake.snapshot(video.REPOSITORY, COMMITS[video.REPOSITORY])
        self.assertEqual(
            {name: Path(entry["path"]) for name, entry in record["files"].items()},
            {
                **{f"weights/{n}": snapshot / n for n in dit("turbo")},
                **{f"weights/vae/{n}": snapshot / "vae" / n for n in video.IMAGES["vae"]},
                "weights/audio/audio_dec_f16.bin": snapshot / "audio/audio_dec_f16.bin",
                "weights/preview/taeh3_dec_f16.bin": snapshot / "preview/taeh3_dec_f16.bin",
                "tokenizer/tokenizer.json": snapshot / "tokenizer/tokenizer.json",
                "LICENSE": snapshot / "LICENSE",
                **{f"encoder/{n}": fake.snapshot(video.ENCODER, COMMITS[video.ENCODER]) / n
                   for n in ("config.json", "model.safetensors")},
            },
        )  # fmt: skip
        self.assertIn("“Excluded Territories” means the European Union", output)
        self.assertEqual(
            sorted(pins(self.cache)),
            [COMMITS[video.REPOSITORY], COMMITS[video.ENCODER]],
        )
        # What the engine does not read is not downloaded: not the default
        # variant, the unused VAE decoder or the model card.
        unread = ("dit-default/", "vae_dec_f16.bin", "README.md", "NOTICE",
                  "model.safetensors.index.json")  # fmt: skip
        self.assertEqual([d for d in fake.downloads if any(u in d for u in unread)], [])
        self.assertFalse((self.models / ".converted").exists())
        # A restart without the Hub starts the installation from the cache.
        fake.failure = OSError("unreachable")
        served = link.resolve()
        self.install()
        self.assertEqual(link.resolve(), served)

    def test_the_default_variant_downloads_once_asked_for(self):
        fake = self.hub()
        link, _ = self.install()
        self.install(variants=("default",))
        record = assembly.verify(link, full=True)
        self.assertEqual(record["video"]["variants"], ["turbo", "default"])
        self.assertIn(f"weights/dit-default/{video.ADALN}", record["files"])
        self.assertTrue(any("dit-default/" in d for d in fake.downloads))
        # The installation keeps both variants from then on.
        self.install()
        self.assertEqual(
            assembly.verify(link)["video"]["variants"], ["turbo", "default"]
        )
        with self.assertRaisesRegex(models.ModelError, "no variant fast"):
            self.install(variants=("fast",))

    def test_a_repository_is_refused_for_its_marker(self):
        fake = self.hub()
        changes = ({"weights": "h3-w4"}, {"version": 2}, {"variants": ["fast"]})
        for number, change in enumerate(changes, 6):
            fake.publish(f"owner/other-{number}", str(number) * 40, functools.partial(tree, files={
                **SOURCES[video.REPOSITORY], video.MARKER: json.dumps(MARKER | change)}))  # fmt: skip
            with self.assertRaisesRegex(
                models.ModelError, "does not describe MiniMax-H3"
            ):
                self.install(model=f"owner/other-{number}")
        partial = dict(SOURCES[video.REPOSITORY])
        del partial[f"dit-turbo/{video.ADALN}"]
        fake.publish("owner/partial", "5" * 40, functools.partial(tree, files=partial))
        with self.assertRaisesRegex(
            models.ModelError, f"missing .*dit-turbo/{video.ADALN}"
        ):
            self.install(model="owner/partial")
        self.assertEqual([d for d in fake.downloads if d.endswith(".safetensors")], [])

    def test_the_release_is_refused_with_the_way_to_the_engines_format(self):
        fake = self.hub()
        with self.assertRaisesRegex(models.ModelError, video.REPOSITORY):
            self.install(model=video.RELEASE)
        self.assertEqual(fake.downloads, [])
        release = tree(self.root / "MiniMax-H3", SOURCES[video.RELEASE])
        with self.assertRaisesRegex(models.ModelError, "publish_video_repo.py"):
            self.prepare_local(release)

    def test_the_license_is_refused_unacknowledged_and_recorded_once_acknowledged(self):
        fake = self.hub()
        unacknowledged = mock.patch.dict(os.environ, {"SPLASH_ACCEPT_LICENSE": ""})
        with (
            unacknowledged,
            mock.patch("sys.stdin", io.StringIO()),
            self.assertRaisesRegex(models.ModelError, "SPLASH_ACCEPT_LICENSE"),
        ):
            self.install()
        self.assertEqual([d for d in fake.downloads if d.endswith(".safetensors")], [])
        self.install()
        (record,) = models.read_json(self.models / ".licenses.json").values()
        self.assertEqual(record["by"], "SPLASH_ACCEPT_LICENSE")
        with unacknowledged:
            self.install()

    def test_local_repositories_and_converted_images_prepare_as_they_are(self):
        build = functools.partial(tree, files=SOURCES[video.ENCODER])
        cached_snapshot(self.cache, video.ENCODER, COMMITS[video.ENCODER], build)
        refs = self.cache / hub.folder_name(video.ENCODER) / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text(COMMITS[video.ENCODER])
        published = tree(self.root / "published", SOURCES[video.REPOSITORY])
        # h3's layout: the images, with fl2va/ linking to FL2VA, whose
        # tokenizer and license it takes, and a stream image nothing reads.
        release = tree(self.root / "release", {
            "LICENSE": LICENSE, "FL2VA/tokenizer/tokenizer.json": "{}"})  # fmt: skip
        converted = tree(self.root / "weights", {
            **IMAGES, **dit("turbo"), "dit-turbo/stream.bin": ""})  # fmt: skip
        (converted / "fl2va").symlink_to(release / "FL2VA")
        with mock.patch("huggingface_hub.HfApi", side_effect=AssertionError("network")):
            for target, variants in ((published, ["turbo", "default"]),
                                     (converted, ["turbo"])):  # fmt: skip
                prepared = self.prepare_local(target)
                self.assertEqual(self.prepare_local(target), prepared)
                record = assembly.verify(prepared, full=True)
                self.assertEqual(
                    record["sources"]["target"], {"repo": str(target), "revision": None}
                )
                self.assertEqual(record["video"]["variants"], variants)
                self.assertEqual(Path(record["files"]["LICENSE"]["path"]).parent,
                                 release if target == converted else published)  # fmt: skip
                self.assertIn("tokenizer/tokenizer.json", record["files"])
                self.assertNotIn("weights/dit-turbo/stream.bin", record["files"])


if __name__ == "__main__":
    unittest.main()

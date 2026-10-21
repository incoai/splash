"""MiniMax-H3's assemblies (install/video.py) on fake layouts of its sources.
The converters do not run: a stand-in writes the images they would."""

import contextlib
import functools
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import FakeHub, cached_snapshot, pins, selection
from install import assembly, hub, local, models, upstream, video

REPOS = (video.TARGET, video.VDN, video.PREVIEW, video.ENCODER)
COMMITS = {repo: str(number) * 40 for number, repo in enumerate(REPOS, 1)}
LICENSE = """MiniMax H3 Community License Agreement
5. “Excluded Territories” means the European Union, the United Kingdom, the Republic of Korea and the United States of America.
"""
# Each source's files the installer reads, and some it must not download.
SOURCES = {
    video.TARGET: {"LICENSE": LICENSE, "FL2VA/model_index.json": "{}",
                   "FL2VA/transformer/model.safetensors.index.json": '{"weight_map": {"a": "t"}}',
                   "FL2VA/transformer/t": "", "FL2VA/video_vae/source/model.safetensors": "",
                   "FL2VA/video_vae/minimax_h3_video_vae.py": "",
                   "FL2VA/audio_vae/config.json": "{}", "FL2VA/audio_vae/model.safetensors": "",
                   "FL2VA/tokenizer/tokenizer.json": "{}"},
    video.VDN: {"README.md": "", **{f"{stage}/{name}": "" for stage, _ in video.VARIANTS.values()
                                    for name in ("adapters/default/adapter.safetensors",
                                                 "linear_branch/model.safetensors")}},
    video.PREVIEW: {"safetensors/taeh3.safetensors": "", "safetensors/taehv.safetensors": ""},
    video.ENCODER: {"config.json": "{}", "model.safetensors": "",
                    "model.safetensors.index.json": "{}"},
}  # fmt: skip


def tree(root: Path, files):
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(data)
    return root


def images(root: Path):
    """The engine's images as the converters write them."""
    files = video.IMAGES.items()
    return tree(root, {f"{d}/{n}": "" for d, names in files for n in names or ["x"]})


class VideoModelTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.cache, self.models = self.root / "hub", self.root / "models"
        self.conversions = []

        def convert_into(staging, fl2va, vdn, taeh3):
            self.conversions.append((fl2va, vdn, taeh3))
            images(staging)

        for patch in (
            mock.patch("install.video._convert_into", convert_into),
            mock.patch("install.video._require_converter_environment", lambda: None),
            mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)),
            mock.patch.dict(os.environ, {"SPLASH_ACCEPT_LICENSE": video.ACCEPT}),
        ):
            self.enterContext(patch)

    def hub(self):
        fake = FakeHub(self, self.cache)
        for repo, files in SOURCES.items():
            fake.publish(repo, COMMITS[repo], functools.partial(tree, files=files))
        return fake

    def install(self):
        chosen = selection(self.root, video.TARGET, language_only=False)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            upstream.prepare(chosen)
        return chosen.link, output.getvalue()

    def prepare_local(self, target):
        with contextlib.redirect_stdout(io.StringIO()):
            return local.prepare_local_model(target, models_dir=self.models)

    def test_the_hub_installs_converted_images_once(self):
        fake = self.hub()
        link, output = self.install()
        record = assembly.verify(link, full=True)
        self.assertEqual(record["target_format"], video.FORMAT)
        self.assertEqual(record["sources"], {
            "target": {"repo": video.TARGET, "revision": COMMITS[video.TARGET]},
            "encoder": {"repo": video.ENCODER, "revision": COMMITS[video.ENCODER]},
        })  # fmt: skip
        self.assertEqual(
            record["video"]["images"]["commits"],
            {name: COMMITS[name] for name in (video.TARGET, video.VDN, video.PREVIEW)},
        )
        self.assertEqual(
            Path(record["files"]["weights/dit-turbo/x"]["path"]).parents[2],
            self.models / ".converted",
        )
        self.assertIn("“Excluded Territories” means the European Union", output)
        self.assertEqual(
            sorted(pins(self.cache)), [COMMITS[video.TARGET], COMMITS[video.ENCODER]]
        )
        # Only what the converters and the engine read is downloaded.
        unread = (".py", "README.md", "taehv.safetensors",
                  f"{video.ENCODER}/model.safetensors.index.json")  # fmt: skip
        self.assertEqual([d for d in fake.downloads if d.endswith(unread)], [])
        # A restart on the same commits converts nothing.
        served = link.resolve()
        self.install()
        self.assertEqual((link.resolve(), len(self.conversions)), (served, 1))

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

    def test_local_images_and_the_repository_layout_prepare(self):
        for repo, files in SOURCES.items():
            build = functools.partial(tree, files=files)
            cached_snapshot(self.cache, repo, COMMITS[repo], build)
            refs = self.cache / hub.folder_name(repo) / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text(COMMITS[repo])
        target = tree(self.root / "MiniMax-H3", SOURCES[video.TARGET])
        converted = images(self.root / "weights")
        (converted / "fl2va").symlink_to(target / "FL2VA")
        with mock.patch("huggingface_hub.HfApi", side_effect=AssertionError("network")):
            prepared = self.prepare_local(converted)
            self.assertEqual(self.prepare_local(converted), prepared)
            record = assembly.verify(prepared, full=True)
            self.assertEqual(record["video"]["images"], {"directory": str(converted)})
            self.assertEqual(self.conversions, [])
            record = assembly.verify(self.prepare_local(target), full=True)
        self.assertEqual(
            record["video"]["images"]["commits"][video.TARGET], str(target)
        )
        self.assertEqual(len(self.conversions), 1)

    def test_collection_keeps_the_images_an_installation_holds(self):
        fl2va = tree(self.root / "h3", SOURCES[video.TARGET]) / "FL2VA"
        with video.converted(self.models, fl2va, self.root, self.root, {}) as (held, _):
            with models.installation_lock(self.models):
                assembly.collect_garbage(self.models)
            self.assertTrue(held.is_dir())
        with models.installation_lock(self.models):
            assembly.collect_garbage(self.models)
        self.assertEqual(list((self.models / ".converted").iterdir()), [])


if __name__ == "__main__":
    unittest.main()

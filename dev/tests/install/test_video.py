"""MiniMax-H3's assemblies (install/video.py) on fake layouts of its sources.
The converters do not run: a stand-in writes the images they would."""

import contextlib
import dataclasses
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
                   "FL2VA/video_vae/source/config.json": "{}",
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


def images(root: Path, variants=("turbo",)):
    """The engine's images as the converters write them, with the DiT of
    each of variants."""
    files = {
        f"{d}/{n}": "" for d, names in video.IMAGES.items() for n in names or ["x"]
    }
    for variant in variants:
        files |= {f"dit-{variant}/x": "", f"dit-{variant}/{video.ADALN}": ""}
    return tree(root, files)


class VideoModelTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.cache, self.models = self.root / "hub", self.root / "models"
        # The conversion steps that ran: decoders, encoder, (dit, variant)
        # and (adaln, variant).
        self.conversions = []

        def step(name, files):
            def run(staging, *arguments):
                variant = arguments[0] if name in ("dit", "adaln") else None
                self.conversions.append((name, variant) if variant else name)
                tree(staging, {f.format(variant=variant): "" for f in files})

            return run

        steps = {
            "_convert_decoders": step("decoders", ["vae/manifest_w8.json", "vae/vae_dec_w8.bin",
                                                   "audio/x", "preview/x"]),
            "_convert_encoder": step("encoder", ["vae/manifest_enc.json", "vae/vae_enc_f16.bin"]),
            "_convert_dit": step("dit", ["dit-{variant}/x"]),
            "_tabulate_adaln": step("adaln", [f"dit-{{variant}}/{video.ADALN}"]),
        }  # fmt: skip
        for patch in (
            *(mock.patch(f"install.video.{name}", run) for name, run in steps.items()),
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

    def install(self, variants=()):
        chosen = selection(self.root, video.TARGET, language_only=False)
        chosen = dataclasses.replace(chosen, variants=variants)
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
        self.assertEqual(record["video"]["images"]["variants"], ["turbo"])
        self.assertEqual(
            Path(record["files"]["weights/dit-turbo/x"]["path"]).parents[2],
            self.models / ".converted",
        )
        self.assertIn(f"weights/dit-turbo/{video.ADALN}", record["files"])
        self.assertFalse(any(name.startswith(("weights/dit-default/", "weights/fl2va/"))
                             for name in record["files"]))  # fmt: skip
        self.assertEqual(self.conversions,
                         ["decoders", "encoder", ("dit", "turbo"), ("adaln", "turbo")])  # fmt: skip
        self.assertIn("“Excluded Territories” means the European Union", output)
        self.assertIn("FL2VA's transformer shards", output)
        self.assertEqual(
            sorted(pins(self.cache)), [COMMITS[video.TARGET], COMMITS[video.ENCODER]]
        )
        # Only what the converters and the engine read is downloaded: not the
        # default variant's stage.
        unread = (".py", "README.md", "taehv.safetensors",
                  f"{video.ENCODER}/model.safetensors.index.json")  # fmt: skip
        self.assertEqual([d for d in fake.downloads if d.endswith(unread)], [])
        self.assertEqual([d for d in fake.downloads if "stage-b-step-2000" in d], [])
        # A restart on the same commits converts nothing, and downloads
        # nothing only the conversion read, the transformer's deleted shard
        # included.
        served = link.resolve()
        (shard,) = self.cache.glob(
            "models--MiniMaxAI--MiniMax-H3/snapshots/*/FL2VA/transformer/t"
        )
        shard.unlink()
        downloaded = len(fake.downloads)
        self.install()
        self.assertEqual((link.resolve(), len(self.conversions)), (served, 4))
        self.assertEqual(fake.downloads[downloaded:], [])

    def test_the_default_variant_converts_once_asked_for(self):
        self.hub()
        link, _ = self.install()
        turbo = Path(assembly.verify(link)["files"]["weights/dit-turbo/x"]["path"])
        self.install(variants=("default",))
        record = assembly.verify(link, full=True)
        self.assertEqual(record["video"]["images"]["variants"], ["turbo", "default"])
        self.assertEqual(
            self.conversions[4:], [("dit", "default"), ("adaln", "default")]
        )
        # The turbo images are the earlier entry's, cloned.
        both = Path(record["files"]["weights/dit-turbo/x"]["path"])
        self.assertNotEqual(both.parents[1], turbo.parents[1])
        self.assertIn(f"weights/dit-default/{video.ADALN}", record["files"])
        # The installation keeps both variants from then on.
        self.install()
        self.assertEqual(len(self.conversions), 6)
        self.assertEqual(
            Path(assembly.verify(link)["files"]["weights/dit-turbo/x"]["path"]), both
        )
        with self.assertRaisesRegex(models.ModelError, "no variant fast"):
            self.install(variants=("fast",))

    def test_images_converted_before_the_encoder_are_completed_with_its_alone(self):
        self.hub()
        commits = {
            name: COMMITS[name] for name in (video.TARGET, video.VDN, video.PREVIEW)
        }
        identity = {"converters": video.ENCODERLESS_CONVERTERS, "commits": commits}
        # Both variants, their stream images and no adaLN tables.
        before = images(video._entry(self.models, identity), ("turbo", "default"))
        for name in ("vae/manifest_enc.json", "vae/vae_enc_f16.bin",
                     f"dit-turbo/{video.ADALN}", f"dit-default/{video.ADALN}"):  # fmt: skip
            (before / name).unlink()
        tree(before, {"dit-turbo/stream.bin": "", "dit-default/stream.bin": ""})
        (before / "model.json").write_bytes(models.json_bytes(identity))
        link, _ = self.install()
        record = assembly.verify(link, full=True)
        self.assertEqual(self.conversions, ["encoder", ("adaln", "turbo")])
        self.assertFalse(any("stream.bin" in name or "dit-default" in name
                             for name in record["files"]))  # fmt: skip
        linked = Path(record["files"]["weights/vae/vae_enc_f16.bin"]["path"])
        self.assertEqual(Path(record["files"]["weights/dit-turbo/x"]["path"]).parents[1],
                         linked.parents[1])  # fmt: skip
        self.assertNotEqual(linked.parents[1], before)

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
        self.assertEqual(len(self.conversions), 4)

    def test_collection_keeps_the_images_an_installation_holds(self):
        fl2va = tree(self.root / "h3", SOURCES[video.TARGET]) / "FL2VA"
        with video.converted(
            self.models, fl2va, self.root, self.root, {}, ("turbo",)
        ) as (
            held,
            _,
        ):
            with models.installation_lock(self.models):
                assembly.collect_garbage(self.models)
            self.assertTrue(held.is_dir())
        with models.installation_lock(self.models):
            assembly.collect_garbage(self.models)
        self.assertEqual(list((self.models / ".converted").iterdir()), [])


if __name__ == "__main__":
    unittest.main()

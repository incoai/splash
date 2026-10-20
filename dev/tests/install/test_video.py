"""MiniMax-H3, a video model's assemblies (install/video.py): from the Hub and
from local files, on fake layouts of its sources. The converters do not run:
a stand-in writes the images they would (convert_into)."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import FakeHub, cached_snapshot, pins, selection
from install import assembly, hub, local, models, upstream, video

COMMITS = {
    video.TARGET: "1" * 40,
    video.VDN: "2" * 40,
    video.PREVIEW: "3" * 40,
    video.ENCODER: "4" * 40,
}
LICENSE = """MiniMax H3 Community License Agreement
The scope of this License Agreement is expressly limited to the “Applicable Territory” as defined below.
3. “Applicable Territory” means worldwide, excluding the Excluded Territories.
5. “Excluded Territories” means the European Union, the United Kingdom, the Republic of Korea and the United States of America.
4. You may not use the MiniMax H3 Works outside the Applicable Territory.
"""


def write(path: Path, data="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        data = json.dumps(data)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


def h3_repo(root: Path):
    """MiniMaxAI/MiniMax-H3's layout: its license and FL2VA."""
    write(root / "LICENSE", LICENSE)
    fl2va = root / "FL2VA"
    write(fl2va / "model_index.json", {"_class_name": "FL2VAPipeline"})
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    write(
        fl2va / "transformer" / "model.safetensors.index.json",
        {"weight_map": {"a": shards[0], "b": shards[1]}},
    )
    for shard in shards:
        write(fl2va / "transformer" / shard, shard)
    write(fl2va / "transformer" / "config.json", {})
    write(fl2va / "video_vae" / "source" / "model.safetensors", "vae")
    write(fl2va / "video_vae" / "minimax_h3_video_vae.py", "code")
    write(fl2va / "audio_vae" / "config.json", {})
    write(fl2va / "audio_vae" / "model.safetensors", "audio")
    write(fl2va / "tokenizer" / "tokenizer.json", "{}")
    write(fl2va / "tokenizer" / "tokenizer_config.json", "{}")
    return root


def vdn_repo(root: Path):
    for stage, turbo in video.VARIANTS.values():
        write(root / stage / "adapters" / "default" / "adapter.safetensors", stage)
        if turbo:
            write(root / stage / "adapters" / "turbo" / "adapter.safetensors", stage)
        write(root / stage / "linear_branch" / "model.safetensors", stage)
        write(root / stage / "metadata.json", {})
        write(root / stage / "model_spec.json", {})
        write(root / stage / "diffusers" / "modeling_vdn_h3.py", "code")
    write(root / "README.md", "readme")
    return root


def preview_repo(root: Path):
    write(root / "safetensors" / "taeh3.safetensors", "taeh3")
    write(root / "safetensors" / "taehv.safetensors", "taehv")
    return root


def encoder_repo(root: Path):
    write(root / "config.json", {"model_type": "qwen3_vl"})
    write(root / "model-00001-of-00002.safetensors", "e1")
    write(root / "model-00002-of-00002.safetensors", "e2")
    # A shard index of other shards, as the repository holds one: unread.
    write(root / "model.safetensors.index.json", {"weight_map": {}})
    return root


def images(root: Path, tag="converted"):
    """The engine's images as the converters write them."""
    for variant in video.VARIANTS:
        write(root / f"dit-{variant}" / "manifest.json", {"files": {}})
        write(root / f"dit-{variant}" / "stream.bin", f"{tag} {variant}")
        write(root / f"dit-{variant}" / "prep.safetensors", tag)
    for name in (
        "manifest_w8.json",
        "vae_dec_w8.bin",
        "manifest_f16.json",
        "vae_dec_f16.bin",
    ):
        write(root / "vae" / name, tag)
    write(root / "audio" / "manifest.json", {})
    write(root / "audio" / "audio_dec_f16.bin", tag)
    write(root / "preview" / "manifest.json", {})
    write(root / "preview" / "taeh3_dec_f16.bin", tag)
    return root


class VideoModelTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cache = self.root / "hub"
        self.models = self.root / "models"
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
            patch.start()
            self.addCleanup(patch.stop)

    def hub(self):
        fake = FakeHub(self, self.cache)
        for repo, build in (
            (video.TARGET, h3_repo),
            (video.VDN, vdn_repo),
            (video.PREVIEW, preview_repo),
            (video.ENCODER, encoder_repo),
        ):
            fake.publish(repo, COMMITS[repo], build)
        return fake

    def install(self, **options):
        chosen = selection(self.root, video.TARGET, language_only=False, **options)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            upstream.prepare(chosen)
        return chosen.link, output.getvalue()

    def cache_main(self, repo_id, build):
        """repo_id at its commit in the Hub cache, recorded as its main
        branch, as a download of the branch leaves it."""
        cached_snapshot(self.cache, repo_id, COMMITS[repo_id], build)
        refs = self.cache / hub.folder_name(repo_id) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(COMMITS[repo_id])

    def prepare_local(self, target, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return local.prepare_local_model(target, models_dir=self.models, **options)

    def test_the_hub_installs_an_assembly_of_converted_images(self):
        fake = self.hub()
        link, output = self.install()
        record = assembly.verify(link, full=True)
        self.assertEqual(record["target_format"], video.FORMAT)
        self.assertEqual(
            record["sources"],
            {
                "target": {"repo": video.TARGET, "revision": COMMITS[video.TARGET]},
                "encoder": {"repo": video.ENCODER, "revision": COMMITS[video.ENCODER]},
            },
        )
        self.assertEqual(record["video"]["license"]["name"], video.LICENSE_NAME)
        images = record["video"]["images"]
        self.assertEqual(images["converters"], video.converters_identity())
        self.assertEqual(
            images["commits"],
            {name: COMMITS[name] for name in (video.TARGET, video.VDN, video.PREVIEW)},
        )
        converted = self.models / ".converted"
        self.assertEqual(
            Path(record["files"]["weights/dit-turbo/stream.bin"]["path"]).parents[1],
            next(converted.iterdir()),
        )
        self.assertEqual(
            sorted(n for n in record["files"] if not n.startswith("weights/dit")),
            [
                "LICENSE",
                "encoder/config.json",
                "encoder/model-00001-of-00002.safetensors",
                "encoder/model-00002-of-00002.safetensors",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
                "weights/audio/audio_dec_f16.bin",
                "weights/audio/manifest.json",
                "weights/fl2va/transformer/model-00001-of-00002.safetensors",
                "weights/fl2va/transformer/model-00002-of-00002.safetensors",
                "weights/fl2va/transformer/model.safetensors.index.json",
                "weights/preview/manifest.json",
                "weights/preview/taeh3_dec_f16.bin",
                "weights/vae/manifest_w8.json",
                "weights/vae/vae_dec_w8.bin",
            ],
        )
        # The converters read the sources' snapshots; nothing else of them
        # was downloaded than the model needs.
        ((fl2va, vdn, taeh3),) = self.conversions
        self.assertEqual(
            fl2va, fake.snapshot(video.TARGET, COMMITS[video.TARGET]) / "FL2VA"
        )
        self.assertEqual(vdn, fake.snapshot(video.VDN, COMMITS[video.VDN]))
        self.assertEqual(taeh3.name, "taeh3.safetensors")
        self.assertEqual(
            [
                d
                for d in fake.downloads
                if d.endswith((".py", "README.md", "taehv.safetensors"))
            ],
            [],
        )
        self.assertNotIn(
            f"{video.ENCODER}/model.safetensors.index.json", fake.downloads
        )
        # The license's territory is quoted before any weight is downloaded.
        self.assertIn("“Excluded Territories” means the European Union", output)
        self.assertEqual(
            sorted(pins(self.cache)),
            sorted([COMMITS[video.TARGET], COMMITS[video.ENCODER]]),
        )
        # A start of the same commits converts nothing and serves the same
        # assembly.
        served = link.resolve()
        self.install()
        self.assertEqual(link.resolve(), served)
        self.assertEqual(len(self.conversions), 1)

    def test_the_license_is_acknowledged_before_any_weight_download(self):
        fake = self.hub()
        unacknowledged = mock.patch.dict(os.environ, {"SPLASH_ACCEPT_LICENSE": ""})
        with (
            unacknowledged,
            mock.patch("sys.stdin", io.StringIO()),
            self.assertRaisesRegex(models.ModelError, "SPLASH_ACCEPT_LICENSE"),
        ):
            self.install()
        self.assertEqual(
            [d for d in fake.downloads if "safetensors" in d and "index" not in d], []
        )
        self.assertFalse((self.models / ".licenses.json").exists())
        # Acknowledged once, it is not asked again.
        self.install()
        self.assertEqual(
            list(models.read_json(self.models / ".licenses.json").values())[0]["by"],
            "SPLASH_ACCEPT_LICENSE",
        )
        with unacknowledged:
            self.install()

    def test_a_video_model_takes_no_language_model_options(self):
        self.hub()
        with self.assertRaisesRegex(
            models.ModelError, "is a video model: it takes no variant"
        ):
            self.install(draft_model="owner/draft")

    def test_missing_sources_are_refused_by_name(self):
        fake = self.hub()
        fake.publish(video.VDN, "5" * 40, lambda p: write(p / "README.md"))
        with self.assertRaisesRegex(
            models.ModelError, "linear_branch/model.safetensors"
        ):
            self.install()

    def test_converted_images_prepare_from_local_files(self):
        fl2va = h3_repo(self.root / "h3") / "FL2VA"
        converted = images(self.root / "weights", "local")
        (converted / "fl2va").symlink_to(fl2va)
        self.cache_main(video.ENCODER, encoder_repo)
        with mock.patch("huggingface_hub.HfApi", side_effect=AssertionError("network")):
            prepared = self.prepare_local(converted)
            self.assertEqual(self.prepare_local(converted), prepared)
        record = assembly.verify(prepared, full=True)
        self.assertEqual(
            record["sources"]["target"], {"repo": str(converted), "revision": None}
        )
        self.assertEqual(record["video"]["images"], {"directory": str(converted)})
        self.assertEqual(
            record["files"]["weights/dit-default/stream.bin"]["path"],
            str(converted / "dit-default" / "stream.bin"),
        )
        self.assertEqual(self.conversions, [])
        with self.assertRaisesRegex(
            models.ModelError, "is a video model: it takes no variant"
        ):
            self.prepare_local(converted, language_only=True)

    def test_the_repository_layout_prepares_converted_from_the_cache(self):
        target = h3_repo(self.root / "MiniMax-H3")
        for repo, build in (
            (video.VDN, vdn_repo),
            (video.PREVIEW, preview_repo),
            (video.ENCODER, encoder_repo),
        ):
            self.cache_main(repo, build)
        prepared = self.prepare_local(target)
        record = assembly.verify(prepared, full=True)
        self.assertEqual(
            record["video"]["images"]["commits"][video.TARGET], str(target)
        )
        self.assertEqual(len(self.conversions), 1)

    def test_collection_keeps_the_images_an_installation_holds(self):
        converted = self.models / ".converted"
        fl2va = h3_repo(self.root / "h3") / "FL2VA"
        with video.converted(self.models, fl2va, self.root, self.root, {}) as (held, _):
            with models.installation_lock(self.models):
                assembly.collect_garbage(self.models)
            self.assertTrue(held.is_dir())
        with models.installation_lock(self.models):
            assembly.collect_garbage(self.models)
        self.assertEqual(list(converted.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

"""dev/tools/publish_video_repo.py on fake images and sources: the
repository it assembles is install/video.py's layout, and a conversion
converts what the images lack."""

import contextlib
import functools
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import FakeHub
from dev.tools import publish_video_repo as publish
from install import hub, models, paths, video

VDN, PREVIEW = publish.VDN, publish.PREVIEW
COMMITS = {video.RELEASE: "1" * 40, VDN: "2" * 40, PREVIEW: "3" * 40}
HOME = "/Users/someone"
# The release's files a conversion reads, and some it must not download.
SOURCES = {
    video.RELEASE: {"LICENSE": "MiniMax H3 Community License Agreement\n",
                    video.RELEASE_INDEX: "{}", "FL2VA/tokenizer/tokenizer.json": "{}",
                    "FL2VA/transformer/model.safetensors.index.json": '{"weight_map": {"a": "t"}}',
                    "FL2VA/transformer/t": "", "FL2VA/video_vae/source/model.safetensors": "",
                    "FL2VA/video_vae/source/config.json": "{}",
                    "FL2VA/video_vae/minimax_h3_video_vae.py": "",
                    "FL2VA/audio_vae/config.json": "{}", "FL2VA/audio_vae/model.safetensors": ""},
    VDN: {"README.md": "", **{f"{stage}/{name}": "" for stage, _ in publish.STAGES.values()
                              for name in ("adapters/default/adapter.safetensors",
                                           "linear_branch/model.safetensors")}},
    PREVIEW: {publish.TAEH3: "", "safetensors/taehv.safetensors": ""},
}  # fmt: skip


def tree(root: Path, files):
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(data)
    return root


def images(root: Path, variants=("turbo",)):
    """The engine's images as h3's converters write them, with the DiT of
    each of variants, their manifests naming the converting machine's
    paths."""
    stage = f"{HOME}/exp/h3-vdn/weights/stage-dmd-step-250"
    source = (f"{HOME}/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots/"
              f"{COMMITS[video.RELEASE]}/FL2VA/video_vae/source/model.safetensors")  # fmt: skip
    files = {
        "vae/manifest_w8.json": json.dumps({"src": source, "had_seed": 0x48335138}),
        "vae/manifest_enc.json": "{}",
        "vae/vae_dec_w8.bin": "w8",
        "vae/vae_enc_f16.bin": "",
        "vae/vae_dec_f16.bin": "",
        "audio/manifest.json": "{}",
        "audio/audio_dec_f16.bin": "",
        "preview/manifest.json": "{}",
        "preview/taeh3_dec_f16.bin": "",
    }
    for variant in variants:
        manifest = {
            "format": video.WEIGHTS,
            "stage": stage,
            "adapters": [f"{stage}/adapters/turbo"],
        }
        files |= {f"dit-{variant}/manifest.json": json.dumps(manifest),
                  f"dit-{variant}/vdn-w8-block00.safetensors": "block",
                  f"dit-{variant}/{video.ADALN}": "", f"dit-{variant}/stream.bin": ""}  # fmt: skip
    return tree(root, files)


class PublishVideoRepoTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.cache = self.root / "hub"
        self.enterContext(
            mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache))
        )

    def publish(self, *arguments):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            status = publish.main([str(argument) for argument in arguments])
        return status, errors.getvalue()

    def test_converted_images_assemble_as_the_repository_installs_them(self):
        snapshot = tree(hub.snapshot(video.RELEASE, COMMITS[video.RELEASE]),
                        SOURCES[video.RELEASE])  # fmt: skip
        converted = images(self.root / "weights")
        (converted / "fl2va").symlink_to(snapshot / "FL2VA")
        out = self.root / "repository"
        status, errors = self.publish(out, "--images", converted)
        self.assertEqual(status, 1)
        self.assertIn(f"--commit {VDN}=COMMIT", errors)
        status, _ = self.publish(out, "--images", converted, "--commit",
                                 f"{VDN}={COMMITS[VDN]}", "--commit", f"{PREVIEW}={COMMITS[PREVIEW]}")  # fmt: skip
        self.assertEqual(status, 0)
        written = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
        self.assertEqual(written, {
            video.MARKER, "README.md", "NOTICE", "LICENSE", "tokenizer/tokenizer.json",
            "vae/manifest_w8.json", "vae/manifest_enc.json", "vae/vae_dec_w8.bin",
            "vae/vae_enc_f16.bin", "audio/manifest.json", "audio/audio_dec_f16.bin",
            "preview/manifest.json", "preview/taeh3_dec_f16.bin", "dit-turbo/manifest.json",
            "dit-turbo/vdn-w8-block00.safetensors", f"dit-turbo/{video.ADALN}"})  # fmt: skip
        self.assertEqual(models.read_json(out / video.MARKER), {
            "format": video.FORMAT, "version": video.VERSION, "weights": video.WEIGHTS,
            "variants": ["turbo"], "sources": COMMITS})  # fmt: skip
        self.assertEqual(
            (out / "dit-turbo/vdn-w8-block00.safetensors").read_text(), "block"
        )
        manifest = models.read_json(out / "dit-turbo/manifest.json")
        self.assertEqual((manifest["format"], manifest["stage"], manifest["adapters"]),
                         (video.WEIGHTS, "stage-dmd-step-250",
                          ["stage-dmd-step-250/adapters/turbo"]))  # fmt: skip
        self.assertEqual(models.read_json(out / "vae/manifest_w8.json")["src"],
                         "FL2VA/video_vae/source/model.safetensors")  # fmt: skip
        self.assertFalse(any(HOME in p.read_text() for p in out.rglob("*.json")))
        self.assertTrue((out / "NOTICE").read_text().startswith(publish.MINIMAX_NOTICE))
        self.assertIn("license_link: LICENSE", (out / "README.md").read_text())
        # The installer links the repository's files as they are.
        repo = hub.Repository.local_files(str(out), video.local_files(out))
        self.assertEqual(video.repository_files(repo, ["turbo"]),
                         written - {video.MARKER, "README.md", "NOTICE"})  # fmt: skip

    def test_a_conversion_converts_what_the_images_lack(self):
        fake = FakeHub(self, self.cache)
        for repo, files in SOURCES.items():
            fake.publish(repo, COMMITS[repo], functools.partial(tree, files=files))
        conversions = []

        def step(name, files):
            def run(converted, *arguments):
                variant = arguments[0] if name in ("dit", "adaln") else None
                conversions.append((name, variant) if variant else name)
                tree(converted, {f.format(variant=variant): "" for f in files})

            return run

        steps = {
            "_convert_decoders": step("decoders", ["vae/manifest_w8.json", "vae/vae_dec_w8.bin",
                                                   "audio/x", "preview/x"]),
            "_convert_encoder": step("encoder", ["vae/manifest_enc.json", "vae/vae_enc_f16.bin"]),
            "_convert_dit": step("dit", ["dit-{variant}/manifest.json"]),
            "_tabulate_adaln": step("adaln", [f"dit-{{variant}}/{video.ADALN}"]),
        }  # fmt: skip
        for patch in (
            *(mock.patch.object(publish, name, run) for name, run in steps.items()),
            mock.patch.object(publish, "_require_converter_environment", lambda: None),
        ):
            self.enterContext(patch)
        converted = self.root / "weights"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(publish.convert(converted, ("turbo",)), COMMITS)
        self.assertEqual(conversions,
                         ["decoders", "encoder", ("dit", "turbo"), ("adaln", "turbo")])  # fmt: skip
        fl2va = fake.snapshot(video.RELEASE, COMMITS[video.RELEASE]) / "FL2VA"
        self.assertEqual((converted / "fl2va").resolve(), fl2va)
        self.assertEqual(publish.release_commit(converted), COMMITS[video.RELEASE])
        self.assertEqual(sorted(p.name for p in converted.iterdir()),
                         ["audio", "dit-turbo", "fl2va", "preview", "vae"])  # fmt: skip
        # Only what the converters read is downloaded: not the default
        # variant's stage.
        unread = (".py", "README.md", "taehv.safetensors", "stage-b-step-2000")
        self.assertEqual([d for d in fake.downloads if d.endswith(unread)], [])
        # The default variant converts alone.
        with contextlib.redirect_stdout(io.StringIO()):
            publish.convert(converted, ("turbo", "default"))
        self.assertEqual(conversions[4:], [("dit", "default"), ("adaln", "default")])

    def test_the_weights_format_is_quant_formats_and_the_converters(self):
        header = (paths.ROOT / "runtime/metal/abi/QuantFormat.h").read_text()
        defined = dict(re.findall(r"#define (QUANT_VIDEO_W8\w*) (\S+)", header))
        converter = (publish.CONVERTERS / "convert_dit.py").read_text()
        self.assertEqual(defined["QUANT_VIDEO_W8"], f'"{video.WEIGHTS}"')
        self.assertEqual(defined["QUANT_VIDEO_W8_GROUP"],
                         re.search(r"^HAD_GROUP = (\d+)", converter, re.M)[1] + "u")  # fmt: skip
        self.assertEqual(defined["QUANT_VIDEO_W8_SEED"],
                         re.search(r"^HAD_SEED = (0x[0-9A-Fa-f]+)", converter, re.M)[1] + "u")  # fmt: skip
        self.assertEqual(defined["QUANT_VIDEO_W8_QMAX"], "127.0f")
        self.assertIn(f'man = {{"format": "{video.WEIGHTS}"', converter)


if __name__ == "__main__":
    unittest.main()

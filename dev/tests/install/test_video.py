"""MiniMax-H3's assemblies (install/video.py) on fake repositories: each
variant converted while its sources are read by range requests, into a
derived entry, by a small recipe in place of h3.py's (conversion.py)."""

import contextlib
import dataclasses
import functools
import hashlib
import io
import json
import os
import re
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dev.tests.installer_fixtures import FakeHub, cached_snapshot, pins, selection
from install import assembly, conversion, h3, hub, local, models, upstream, video
from install.families import H3

VDN = H3.variants["vdn-8"].companion
COMMITS = {H3.release: "1" * 40, H3.encoder: "2" * 40, VDN: "3" * 40}
LICENSE = """MiniMax H3 Community License Agreement
5. “Excluded Territories” means the European Union, the United Kingdom, the Republic of Korea and the United States of America.
"""
SHARD = "FL2VA/transformer/model-00001-of-00001.safetensors"
TENSORS = {f"w{i}": ("F32", np.full((1024,), i, np.float32)) for i in range(3)}
# The images the engine reads, as the fake recipe writes them.
IMAGES = [
    "prep.safetensors",
    "vdn-w8-block00.safetensors",
    H3.adaln,
    H3.dit_manifest,
]
OTHER = ["vae/v.bin", "audio/a.bin", "preview/p.bin"]


def encoder_tensors(layers, extra=()):
    """A shard of a Qwen3-VL checkpoint's layers (two tensors each, of
    distinct bytes) and the extra tensors named."""
    names = [f"language_model.model.layers.{i}.{n}" for i in layers
             for n in ("input_layernorm.weight", "mlp.down_proj.scales")]  # fmt: skip
    return {n: ("F32", np.full((16 + i,), i, np.float32))
            for i, n in enumerate([*names, *extra])}  # fmt: skip


# The text encoder's checkpoint, as mlx-community's are sharded: a shard the
# engine reads all of (the token table, the tower, early layers), one it
# reads some of (layers up to H3.encoder_layers - 1) and one it reads
# none of (the later layers, the final norm and lm_head).
ENCODER_SHARDS = {
    "model-00001-of-00003.safetensors": h3.safetensors(encoder_tensors(range(30), [
        "language_model.model.embed_tokens.weight", "language_model.model.embed_tokens.scales",
        "vision_tower.blocks.0.norm1.weight"]), {"format": "mlx"}),
    "model-00002-of-00003.safetensors": h3.safetensors(encoder_tensors(range(30, 53)), {"format": "mlx"}),
    "model-00003-of-00003.safetensors": h3.safetensors(encoder_tensors(range(53, 64), [
        "language_model.model.norm.weight", "language_model.lm_head.weight"]), {"format": "mlx"}),
}  # fmt: skip
SOURCES = {
    H3.release: {"LICENSE": LICENSE, H3.release_index: "{}", "README.md": "",
                 "FL2VA/tokenizer/tokenizer.json": "{}", SHARD: h3.safetensors(TENSORS)},
    H3.encoder: {"config.json": "{}", **ENCODER_SHARDS,
                 "model.safetensors.index.json": "{}"},
    VDN: {f"{stage}/{n}": h3.safetensors(TENSORS) for stage, _ in h3.STAGES.values()
         for n in ("linear_branch/model.safetensors", "adapters/default/adapter_spec.json",
                   "adapters/default/adapter_model.safetensors",
                   "adapters/turbo/adapter_spec.json", "adapters/turbo/adapter_model.safetensors")},
}  # fmt: skip
# The files h3.inputs names, of the release, besides SHARD.
SOURCES[H3.release] |= {f"{d}{n}": "{}" for d in (h3.VIDEO_VAE, h3.AUDIO_VAE)
                        for n in ("model.safetensors", "config.json")}  # fmt: skip
BRANCH = "stage-dmd-step-250/linear_branch/model.safetensors"


def tree(root: Path, files):
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            (root / name).write_bytes(data)
        else:
            (root / name).write_text(data)
    return root


class VideoModelTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.cache, self.models = self.root / "hub", self.root / "models"
        self.runs, self.engine, self.fail = [], [], set()
        for patch in (
            mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)),
            mock.patch.object(h3, "units", self.units),
            mock.patch.object(models, "run_engine", self.run_engine),
        ):
            self.enterContext(patch)

    def units(self, stage, variant, release, vdn, run_engine):
        """A recipe as h3.py's is laid out: units reading the sources' tensors,
        scratch that a later unit consumes and discards, an engine step."""
        weights = conversion.Source(release, SHARD)
        dit = f"dit-{variant}"

        def run(name, outputs):
            def write(tensors):
                if name in self.fail:
                    self.fail.discard(name)
                    raise models.ModelError("the network went away")
                self.runs.append((variant, name))
                data = (
                    b"".join(a.tobytes() for _, a in tensors.values())
                    + variant.encode()
                )
                return {output: data for output in outputs}

            return write

        def tabulate(_):
            steps = next(v.steps for v in H3.variants.values() if v.dit == variant)
            run_engine(["video-adaln", str(stage / ".scratch"), variant, str(steps)])
            table = stage / ".scratch" / "table"
            table.write_bytes((stage / ".scratch/base").read_bytes())
            return {f"{dit}/{H3.adaln}": table}

        U = conversion.Unit
        return [
            U("base", (".scratch/base",), run("base", [".scratch/base"]),
              lambda: [(weights, ["w2"])], consumed_by=f"{dit}/{H3.adaln}"),
            U("tables", (f"{dit}/{H3.adaln}",), tabulate, discard=(".scratch",)),
            *(U(name, (f"{dit}/{name}",), run(name, [f"{dit}/{name}"]),
                lambda: [(weights, ["w0", "w1"])]) for name in IMAGES if name != H3.adaln),
            U("decoders", tuple(OTHER), run("decoders", OTHER),
              lambda: [(conversion.Source(vdn, BRANCH), ["w1"])]),
        ]  # fmt: skip

    def run_engine(self, arguments, check):
        self.engine.append(arguments[0])

    def hub(self):
        fake = FakeHub(self, self.cache)
        for repo, files in SOURCES.items():
            fake.publish(repo, COMMITS[repo], functools.partial(tree, files=files))
        return fake

    def install(self, model=f"{H3.release}:vdn-8"):
        chosen = selection(self.root, model, language_only=False)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            upstream.prepare(chosen)
        return chosen.link, output.getvalue()

    def entries(self):
        return sorted(p.name for p in (self.models / assembly.CONVERTED).iterdir())

    def test_a_variant_is_converted_as_its_sources_are_read_and_linked(self):
        fake = self.hub()
        link, output = self.install()
        record = assembly.verify(link, full=True)
        listed = (hub.Repository.resolve(n) for n in (H3.release, VDN))
        key = conversion.key(
            h3.inputs("turbo", *listed), video.recipe_identity(h3), "turbo"
        )
        self.assertEqual(record["sources"], {
            "target": {"repo": H3.release, "revision": COMMITS[H3.release]},
            "encoder": {"repo": H3.encoder, "revision": COMMITS[H3.encoder]},
        })  # fmt: skip
        self.assertEqual(record["video"]["variants"], ["vdn-8"])
        self.assertEqual(record["video"]["key"], key)
        entry = self.models / assembly.CONVERTED / key
        package = self.models / assembly.CONVERTED / record["video"]["encoder"]["key"]
        release = fake.snapshot(H3.release, COMMITS[H3.release])
        self.assertEqual(
            {name: Path(e["path"]) for name, e in record["files"].items()},
            {
                **{f"weights/dit-turbo/{n}": entry / "dit-turbo" / n for n in IMAGES},
                **{f"weights/{n}": entry / n for n in OTHER},
                "tokenizer/tokenizer.json": release / "FL2VA/tokenizer/tokenizer.json",
                "LICENSE": release / "LICENSE",
                **{f"encoder/{n}": fake.snapshot(H3.encoder, COMMITS[H3.encoder]) / n
                   for n in ("config.json", "model-00001-of-00003.safetensors")},
                **{f"encoder/{n}": package / n
                   for n in ("model-00002-of-00003.safetensors", video.ENCODER_INDEX)},
            },
        )  # fmt: skip
        data = (entry / "dit-turbo/prep.safetensors").read_bytes()
        self.assertEqual(
            data, np.arange(2, dtype=np.float32).repeat(1024).tobytes() + b"turbo"
        )
        self.assertEqual((entry / "dit-turbo" / H3.adaln).read_bytes()[:4096],
                         np.full(1024, 2, np.float32).tobytes())  # fmt: skip
        self.assertEqual(self.engine, ["video-adaln"])
        self.assertIn(H3.license, output)
        # The weights are read by ranges, never downloaded; scratch is gone.
        self.assertIn(f"{H3.release}/{SHARD}", fake.range_reads)
        self.assertEqual([d for d in fake.downloads if d.endswith(".safetensors")
                          and not d.startswith(H3.encoder)], [])  # fmt: skip
        self.assertEqual(self.entries(), sorted([key, package.name]))
        self.assertFalse((entry / ".scratch").exists())
        self.assertEqual(
            pins(self.cache), sorted([COMMITS[H3.encoder], COMMITS[H3.release]])
        )
        # A restart converts nothing, and starts without the Hub too.
        runs = len(self.runs)
        served = link.resolve()
        self.install()
        fake.failure = OSError("unreachable")
        self.install()
        self.assertEqual((link.resolve(), len(self.runs)), (served, runs))

    def test_a_record_of_an_unhashable_target_format_is_invalid(self):
        self.hub()
        link, _ = self.install()
        record = assembly.verify(link)
        for target_format in (["gguf"], {"gguf": 1}):
            path = link / "model.json"
            path.write_text(json.dumps({**record, "target_format": target_format}))
            with self.assertRaisesRegex(models.ModelError, "invalid resolved"):
                assembly.verify(link)

    def test_a_commit_that_changes_no_file_read_converts_nothing(self):
        fake = self.hub()
        link, _ = self.install()
        key = assembly.verify(link)["video"]["key"]
        runs = len(self.runs)
        # A new commit of the release: its model card changed, the files the
        # recipe reads did not.
        card = SOURCES[H3.release] | {"README.md": "a new card"}
        fake.publish(H3.release, "5" * 40, functools.partial(tree, files=card))
        self.install()
        record = assembly.verify(link)
        self.assertEqual(record["sources"]["target"]["revision"], "5" * 40)
        self.assertEqual((record["video"]["key"], len(self.runs)), (key, runs))
        # One whose shard changed is converted again.
        shard = card | {SHARD: h3.safetensors({**TENSORS, "w0": TENSORS["w1"]})}
        fake.publish(H3.release, "6" * 40, functools.partial(tree, files=shard))
        self.install()
        self.assertNotEqual(assembly.verify(link)["video"]["key"], key)
        self.assertGreater(len(self.runs), runs)

    def test_prepare_reports_what_it_did(self):
        # As upstream.prepare reports a language model's start, which
        # `prepare --json-progress` sends the app last.
        fake = self.hub()
        chosen = selection(self.root, f"{H3.release}:vdn-8", language_only=False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(upstream.prepare(chosen), upstream.UPDATED)
            self.assertEqual(upstream.prepare(chosen), upstream.UP_TO_DATE)
            card = SOURCES[H3.release] | {"README.md": "a new card"}
            fake.publish(H3.release, "5" * 40, functools.partial(tree, files=card))
            self.assertEqual(upstream.prepare(chosen), upstream.UPDATED)
            with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
                self.assertEqual(upstream.prepare(chosen), upstream.UNREACHABLE)

    def test_an_interrupted_conversion_resumes_at_its_first_unit_not_written(self):
        self.hub()
        self.fail = {"vdn-w8-block00.safetensors"}
        with self.assertRaisesRegex(models.ModelError, "the network went away"):
            self.install()
        (stage,) = (self.models / assembly.CONVERTED).iterdir()
        self.assertTrue(stage.name.startswith(models.ENTRY_STAGING))
        # Garbage collection keeps a recent conversion's staging, held or not.
        with models.installation_lock(self.models):
            assembly.collect_garbage(self.models)
        self.assertTrue(stage.is_dir())
        self.runs.clear()
        self.install()
        self.assertEqual([name for _, name in self.runs],
                         ["vdn-w8-block00.safetensors", H3.dit_manifest, "decoders"])  # fmt: skip
        self.assertFalse(stage.exists())

    def test_each_variant_is_a_model_of_its_own_and_collected_once_unlinked(self):
        self.hub()
        turbo, _ = self.install()
        default, _ = self.install(f"{H3.release}:vdn-50")
        self.assertEqual(default, self.models / f"{H3.release}:vdn-50")
        self.assertEqual(assembly.verify(default)["video"]["variants"], ["vdn-50"])
        self.assertIn("weights/dit-default/vdn-w8-block00.safetensors",
                      assembly.verify(default)["files"])  # fmt: skip
        # Two DiTs' entries, and the encoder's package they share.
        self.assertEqual(len(self.entries()), 3)
        with self.assertRaisesRegex(models.ModelError, "no variant fast"):
            self.install(f"{H3.release}:fast")
        # A conversion's staging nothing resumed for a week is collected too.
        stale = self.models / assembly.CONVERTED / f"{models.ENTRY_STAGING}{'5' * 64}"
        stale.mkdir()
        week = time.time() - assembly.RESUMABLE_SECONDS - 60
        os.utime(stale, (week, week))
        turbo.unlink()
        with models.installation_lock(self.models):
            assembly.collect_garbage(self.models)
        video_record = assembly.verify(default)["video"]
        self.assertEqual(self.entries(),
                         sorted([video_record["key"], video_record["encoder"]["key"]]))  # fmt: skip

    def test_a_model_id_without_a_variant_keeps_the_one_installed(self):
        self.hub()
        # An installation of the model ID while vdn-8 was the default.
        old = dataclasses.replace(H3, default_variant="vdn-8")
        with mock.patch.object(video, "H3", old):
            link, _ = self.install(H3.release)
        runs = len(self.runs)
        self.install(H3.release)
        self.assertEqual(assembly.verify(link)["video"]["variants"], ["vdn-8"])
        self.assertEqual(len(self.runs), runs)

    def test_each_variant_keeps_the_key_and_manifest_it_was_installed_with(self):
        # Literal, as the recipe derived them at 0a784b69 on these sources,
        # whose files outside LFS the fake Hub names by their Git blob IDs
        # (installer_fixtures.blob_id): an installation's entry and record
        # must not change under a refactor.
        self.hub()
        self.assertEqual(video.recipe_identity(h3), "h3-w8a8-vdn-v1+r2")
        for model, variant, dit, key in (
            (f"{H3.release}:vdn-8", "vdn-8", "turbo", "6a813d51d9ab9cbec15813c249507eaafa465ba0f7d40104a7dc09211c3f3c87"),
            (f"{H3.release}:vdn-50", "vdn-50", "default", "9a006740b6f6211148ee2fa7f1c6314efbf70c3db4bb5924cd1bcfa2c303266f"),
        ):  # fmt: skip
            record = assembly.verify(self.install(model)[0])
            self.assertEqual(record["video"], {
                "variants": [variant], "key": key,
                "sources": {VDN: {"repo": VDN, "revision": "3" * 40},
                            h3.TAEH3_REPO: {"repo": h3.TAEH3_REPO, "revision": "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c"}},
                "encoder": {"key": "58a86c8f8cef3adcf876fb1ed7a3acfc33410e65fc4e6dbdf6115a029f96ed93",
                            "package": "qwen3-vl-encoder-package-v1", "layers": 50,
                            "linked": ["model-00001-of-00003.safetensors"],
                            "trimmed": ["model-00002-of-00003.safetensors"],
                            "omitted": ["model-00003-of-00003.safetensors"]}})  # fmt: skip
            self.assertEqual({n.split("/")[1] for n in record["files"] if n.startswith("weights/dit-")},
                             {f"dit-{dit}"})  # fmt: skip
        # The DiT's manifest of given adapters and QK norms, byte for byte.
        targets = {"transformer_blocks.45.attn.to_q.weight", "transformer_blocks.0.ff.net.2.weight",
                   "transformer_blocks.0.attn.to_gate.weight", "norm_out.linear.weight"}  # fmt: skip
        norms = {f"blocks.{b}.attn.{p}_norm.weight": qk_norm(b, p) for b in h3.QKFIX for p in "qk"}  # fmt: skip
        for dit, stage, digest in (
            ("turbo", "stage-dmd-step-250", "6e27f63f0d560c49188613ac63ea82627b3e316c10864f646cfebc917378812b"),
            ("default", "stage-b-step-2000", "45bb2529b1c21afe24c41533430009f2b1e74c6ba77935e567d4a96fa9c57eaf"),
        ):  # fmt: skip
            adapters = [SimpleNamespace(name=n, targets=lambda: targets)
                        for n in ("default", "turbo")[: 1 + (dit == "turbo")]]  # fmt: skip
            data = h3.dit_manifest(dit, adapters, norms)
            manifest = json.loads(data)
            self.assertEqual(
                (manifest["format"], manifest["stage"]), ("h3-w8a8-vdn-v1", stage)
            )
            self.assertEqual(manifest["qkfix"]["blocks"]["45"], {
                "file": "qkfix-block45.safetensors", "split": [126, 127, -1, -1], "gq_max": 47.0, "gk_max": 48.0})  # fmt: skip
            self.assertEqual(sorted(manifest["qkfix"]["blocks"]), ["45", "48", "49"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

    def test_the_encoder_package_holds_only_the_tensors_the_engine_reads(self):
        fake = self.hub()
        link, _ = self.install()
        record = assembly.verify(link, full=True)
        package = record["video"]["encoder"]
        self.assertEqual(package | {"key": None}, {
            "key": None, "package": H3.encoder_package, "layers": H3.encoder_layers,
            "linked": ["model-00001-of-00003.safetensors"],
            "trimmed": ["model-00002-of-00003.safetensors"],
            "omitted": ["model-00003-of-00003.safetensors"]})  # fmt: skip
        # The shard read in part is read by ranges; the one not read at all
        # is neither downloaded nor read past its header.
        self.assertEqual(sorted(d for d in fake.downloads if d.startswith(H3.encoder)),
                         [f"{H3.encoder}/{n}" for n in
                          ("config.json", "model-00001-of-00003.safetensors")])  # fmt: skip
        # Every tensor the engine reads, of its name, dtype, shape and bytes;
        # no other.
        source = fake.remote / H3.encoder / COMMITS[H3.encoder]
        expected, found = {}, {}
        for name in ENCODER_SHARDS:
            expected |= {n: v for n, v in tensors(source / name).items()
                         if video.encoder_reads(n)}  # fmt: skip
        for name in record["files"]:
            if name.startswith("encoder/") and name.endswith(".safetensors"):
                found |= tensors(link / name)
        self.assertEqual(found, expected)
        self.assertEqual(len(found), 3 + 2 * H3.encoder_layers)
        self.assertEqual(header(link / "encoder/model-00002-of-00003.safetensors")
                         .get("__metadata__"), {"format": "mlx"})  # fmt: skip
        index = models.read_json(link / "encoder" / video.ENCODER_INDEX)
        self.assertEqual(set(index["weight_map"]), set(expected))
        # Installing again writes nothing, with the Hub or without it.
        entry = self.models / assembly.CONVERTED / package["key"]
        before = sorted((p, p.stat().st_mtime_ns) for p in entry.rglob("*"))
        served = link.resolve()
        self.install()
        fake.failure = OSError("unreachable")
        self.install()
        self.assertEqual(link.resolve(), served)
        self.assertEqual(
            sorted((p, p.stat().st_mtime_ns) for p in entry.rglob("*")), before
        )

    def test_an_encoder_without_a_layer_the_engine_reads_is_refused(self):
        fake = self.hub()
        shards = dict(SOURCES[H3.encoder])
        del shards["model-00002-of-00003.safetensors"]
        fake.publish(H3.encoder, "7" * 40, functools.partial(tree, files=shards))
        with self.assertRaisesRegex(models.ModelError, r"lacks tensors the engine reads: "
                                    r"language_model\.model\.layers\.30\.\*"):  # fmt: skip
            self.install()

    def test_other_repositories_are_refused(self):
        fake = self.hub()
        # A repository of the engine's format earlier builds served.
        fake.publish("owner/converted", "6" * 40, functools.partial(tree, files={
            "splash-video.json": "{}", "LICENSE": LICENSE}))  # fmt: skip
        installed = {"target_format": H3.target_format, "sources": {}, "video": {}}
        with self.assertRaisesRegex(
            models.ModelError, f"serve {H3.release}:{H3.default_variant}"
        ):
            video.prepare(selection(self.root, "owner/converted", language_only=False),
                          hub.Repository.resolve("owner/converted"), installed)  # fmt: skip

    def test_images_converted_before_prepare_as_they_are(self):
        build = functools.partial(tree, files=SOURCES[H3.encoder])
        cached_snapshot(self.cache, H3.encoder, COMMITS[H3.encoder], build)
        refs = self.cache / hub.folder_name(H3.encoder) / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text(COMMITS[H3.encoder])
        dits = [v.dit for v in H3.variants.values()]
        images = {f"dit-{d}/{n}": "" for d in dits for n in IMAGES}
        images["dit-pika-6/adapter/lora-block00.safetensors"] = ""
        target = tree(self.root / "images", {
            **images, **{n: "" for n in OTHER}, "LICENSE": LICENSE, "NOTICE": "",
            "tokenizer/tokenizer.json": "{}"})  # fmt: skip
        with mock.patch("huggingface_hub.HfApi", side_effect=AssertionError("network")):
            with contextlib.redirect_stdout(io.StringIO()):
                prepared = local.prepare_local_model(target, models_dir=self.models)
            record = assembly.verify(prepared, full=True)
            self.assertEqual(record["video"]["variants"], list(H3.variants))
            for name in ("LICENSE", "NOTICE", "tokenizer/tokenizer.json"):
                self.assertEqual(Path(record["files"][name]["path"]), target / name)
            # A DiT's runtime adapter, in its adapter/ directory.
            adapter = "dit-pika-6/adapter/lora-block00.safetensors"
            self.assertEqual(
                Path(record["files"][f"weights/{adapter}"]["path"]), target / adapter
            )
            with self.assertRaisesRegex(models.ModelError, "converts as it installs"):
                local.prepare_local_model(tree(self.root / "MiniMax-H3", SOURCES[H3.release]),
                                          models_dir=self.models)  # fmt: skip


def qk_norm(block, p):
    """A QKFIX block's q or k norm weight, distinct by block and by p."""
    return np.linspace(1.0, 2.0 + block + (p == "k"), h3.HD, dtype=np.float32)


def header(path):
    """A safetensors file's header."""
    data = path.read_bytes()
    return json.loads(data[8 : 8 + int.from_bytes(data[:8], "little")])


def tensors(path):
    """Each tensor of a safetensors file: name -> (dtype, shape, bytes)."""
    data = path.read_bytes()
    base = 8 + int.from_bytes(data[:8], "little")
    return {name: (e["dtype"], e["shape"],
                   data[base + e["data_offsets"][0]:base + e["data_offsets"][1]])
            for name, e in header(path).items() if name != "__metadata__"}  # fmt: skip


class ConversionTest(unittest.TestCase):
    def test_adjacent_tensors_are_read_in_one_range_and_chunked(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (directory / "w.safetensors").write_bytes(h3.safetensors(TENSORS))
        repo = hub.Repository.local_files(
            "owner/w", {"w.safetensors": directory / "w.safetensors"}
        )
        source = conversion.Source(repo, "w.safetensors")
        reads = []
        read_into = source.read_into
        source.read_into = lambda start, view: (
            reads.append((start, len(view))),
            read_into(start, view),
        )
        with (
            mock.patch.object(conversion, "CHUNK", 3000),
            ThreadPoolExecutor(2) as pool,
        ):
            tensors = conversion.fetch([(source, ["w2", "w0", "w1"])], pool)
        self.assertEqual(
            sorted(reads),
            [(0, 3000), (3000, 3000), (6000, 3000), (9000, 3000), (12000, 288)],
        )
        for name, (dtype, array) in TENSORS.items():
            self.assertEqual(tensors[name][0], dtype)
            np.testing.assert_array_equal(tensors[name][1], array)

    def test_the_key_names_the_files_read_the_recipe_and_the_variant(self):
        key = conversion.key({"a/b": "1" * 40}, "r", "turbo")
        self.assertEqual(len({key, conversion.key({"a/b": "2" * 40}, "r", "turbo"),
                              conversion.key({"a/b": "1" * 40}, "r2", "turbo"),
                              conversion.key({"a/b": "1" * 40}, "r", "default"),
                              conversion.key({"a/b": "1" * 40}, "r", "turbo", {"name": "turbo"})}), 5)  # fmt: skip


class FamilyTest(unittest.TestCase):
    def test_the_assemblies_target_format_is_the_familys(self):
        # assembly.py names it as a literal, as the engine does.
        self.assertEqual(
            assembly.TARGET_FORMATS[H3.target_format],
            ({"video"}, {"target", "encoder"}),
        )


class FormatTest(unittest.TestCase):
    def test_the_converters_constants_are_the_headers(self):
        # QUANT_VIDEO_W8 of runtime/metal/abi/QuantFormat.h, which the
        # engine's readers and kernels take: the recipe's name, the Hadamard
        # group and seed of h3.py's rotation, and the int8 scale's divisor.
        header = Path(__file__).resolve().parents[3] / "runtime/metal/abi/QuantFormat.h"
        defines = dict(
            re.findall(r"^#define (QUANT_VIDEO_W8\w*) (\S+)$", header.read_text(), re.M)
        )
        self.assertEqual(defines["QUANT_VIDEO_W8"], f'"{video.RECIPE}"')
        self.assertEqual(defines["QUANT_VIDEO_W8_GROUP"], f"{h3.HAD_GROUP}u")
        self.assertEqual(defines["QUANT_VIDEO_W8_SEED"], f"{h3.HAD_SEED:#x}u")
        self.assertEqual(defines["QUANT_VIDEO_W8_QMAX"], "127.0f")
        q, scale = h3.quant_rows(np.array([[0.5, -2.0, 1.0]], np.float32))
        self.assertEqual(q.tolist(), [[32, -127, 64]])
        self.assertEqual(scale.tolist(), [np.float32(2.0) / np.float32(127.0)])


class QkfixTest(unittest.TestCase):
    def test_the_vdn_manifest_refuses_qk_gains_other_than_its_qkfix_blocks(self):
        cache = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve() / "hub"
        fake = FakeHub(self, cache)
        norms = [
            f"blocks.{b}.attn.{p}_norm.weight" for b in range(h3.BLOCKS) for p in "qk"
        ]
        weights = h3.safetensors({n: ("F32", np.zeros(1, np.float32)) for n in norms})
        stage, _ = h3.STAGES["turbo"]
        files = {
            H3.release: {f"{h3.TRANSFORMER}model.safetensors.index.json":
                         json.dumps({"weight_map": dict.fromkeys(norms, "model.safetensors")}),
                         f"{h3.TRANSFORMER}model.safetensors": weights,
                         **{f"{d}model.safetensors": weights for d in (h3.VIDEO_VAE, h3.AUDIO_VAE)}},
            VDN: {f"{stage}/linear_branch/model.safetensors": weights}
            | {f"{stage}/adapters/{a}/{n}": data for a in ("default", "turbo")
               for n, data in (("adapter_spec.json", "{}"), ("adapter_model.safetensors", weights))},
        }  # fmt: skip
        for repo, tree_files in files.items():
            fake.publish(repo, COMMITS[repo], functools.partial(tree, files=tree_files))
        units = h3.units(cache.parent / "stage", "turbo", hub.Repository.resolve(H3.release),
                         hub.Repository.resolve(VDN), None)  # fmt: skip
        (unit,) = [u for u in units if u.name == "manifest"]
        # It reads all 100 QK norms; the manifest it writes is dit_manifest's.
        self.assertEqual(
            sorted(n for _, names in unit.reads() for n in names), sorted(norms)
        )
        t = {f"blocks.{b}.attn.{p}_norm.weight": ("F32", qk_norm(b, p) if b in h3.QKFIX
             else np.full(h3.HD, 4.0, np.float32)) for b in range(h3.BLOCKS) for p in "qk"}  # fmt: skip
        adapters = [SimpleNamespace(name=n, targets=set) for n in ("default", "turbo")]
        want = h3.dit_manifest("turbo", adapters, h3.Tensors(t))
        self.assertEqual(unit.run(t), {"dit-turbo/manifest.json": want})
        # A block above the gain threshold that is not a QKFIX block, or a
        # QKFIX block below it, is refused.
        for b, norm in ((3, 6.0), (45, 4.0)):
            changed = t | {f"blocks.{b}.attn.{p}_norm.weight": ("F32", np.full(h3.HD, norm, np.float32))
                           for p in "qk"}  # fmt: skip
            with self.assertRaisesRegex(
                models.ModelError,
                rf"block {b} of .* its QKFIX blocks are not \[45, 48, 49\]",
            ):
                unit.run(changed)


if __name__ == "__main__":
    unittest.main()

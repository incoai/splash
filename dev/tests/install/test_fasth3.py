"""FastH3's variants of MiniMax-H3 (install/fasth3.py) on fake repositories:
each installed from its own companion, its recipe's units reading only the
tensors they convert, and the students' tensors and schedule as a reference
conversion of the students takes them."""

import contextlib
import functools
import hashlib
import io
import json
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np

from dev.tests.install.test_video import ENCODER_SHARDS, LICENSE, qk_norm, tree
from dev.tests.installer_fixtures import FakeHub, selection
from install import assembly, conversion, fasth3, h3, hub, models, upstream, video
from install.families import H3, PIKA

RELEASE, BASE = "1" * 40, fasth3.LORA_BASE
COMMITS = {H3.release: RELEASE, H3.encoder: "2" * 40, fasth3.STUDENT_8: "3" * 40,
           fasth3.LORA_4: "4" * 40, fasth3.STUDENT_4[0]: fasth3.STUDENT_4[1]}  # fmt: skip
ONE = ("F32", np.zeros(1, np.float32))
CONTRACT_8 = {"dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125], "transformer_forwards": 8,
              "num_inference_steps": 9, "attention_backend": "VIDEO_SPARSE_ATTN_H3", "vsa_tile_size": 64,
              "vsa_sparsity": 0.8, "video_scheduler_shift": 10.0, "audio_scheduler_shift": 3.0,
              "model_id": fasth3.STUDENT_8, "checkpoint_step": 1300, "guidance_scale": 1.0}  # fmt: skip
# The four-step contract predates the shifts in fastvideo_inference.json.
CONTRACT_4 = {k: v for k, v in CONTRACT_8.items() if not k.endswith("_shift")} | {
    "dmd_denoising_steps": [999, 749, 500, 250], "transformer_forwards": 4, "num_inference_steps": 5,
    "vsa_sparsity": 0.9, "model_id": fasth3.STUDENT_4[0]}  # fmt: skip
STUDENT_BLOCK = ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "attn.to_gate_compress.weight",
                 "attn.to_out.0.weight", "ff.net.0.proj.weight", "ff.net.2.weight", "norm1.weight", "norm2.weight",
                 "attn.norm_q.weight", "attn.norm_k.weight", "adaln_proj.linear.weight", "adaln_proj.linear.bias")  # fmt: skip
NATIVE_BLOCK = ("attn.qkv_proj.weight", "attn.out_proj.weight", "attn.q_norm.weight", "attn.k_norm.weight",
                "norm1.weight", "norm2.weight", "mlp.fc1.weight", "mlp.fc2.weight", "adaln_proj.linear.weight",
                "adaln_proj.linear.bias")  # fmt: skip
REFINER = ("norm1.weight", "norm2.weight", "attn.norm_q.weight", "attn.norm_k.weight", "attn.to_q.weight",
           "attn.to_k.weight", "attn.to_v.weight", "attn.to_out.0.weight", "ff.net.0.proj.weight", "ff.net.2.weight")  # fmt: skip
GLOBALS = [*fasth3.TIME, *(f"{p}.{q}" for p in ("context_embedder", "norm_out.linear", "proj_in", "audio_proj_in",
                                                "proj_out", "audio_proj_out") for q in ("weight", "bias")),
           "norm_out.norm.weight", "token_refiner.final_norm.weight"]  # fmt: skip


def safetensors(names):
    return h3.safetensors({n: ONE for n in names})


def student_names():
    names = [
        f"transformer_blocks.{i}.{n}"
        for i in range(fasth3.BLOCKS)
        for n in STUDENT_BLOCK
    ]
    names += [
        f"token_refiner.refiner_blocks.{b}.{n}" for b in range(2) for n in REFINER
    ]
    return names + GLOBALS


def native_names():
    names = [f"blocks.{i}.{n}" for i in range(fasth3.BLOCKS) for n in NATIVE_BLOCK]
    names += [f"token_refiner.blocks.{b}.{n}" for b in range(2) for n in
              ("norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight", "attn.qkv_proj.weight",
               "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")]  # fmt: skip
    names += [
        f"time_embedder.{p}.{q}"
        for p in ("proj_in", "proj_out")
        for q in ("weight", "bias")
    ]
    names += [
        f"{p}.{q}"
        for p in ("condition_proj", "final_layer.adaln_proj.linear")
        for q in ("weight", "bias")
    ]
    return names + list(h3.IO_KEYS) + ["token_refiner.final_norm.weight"]


def bundle_names():
    names = [f"transformer_blocks.{i}.{n}" for i in range(fasth3.BLOCKS) for n in
             ("attn.to_q.lora_A.weight", "attn.to_q.lora_B.weight", "attn.to_gate_compress.set_weight",
              "norm1.diff", "adaln_proj.linear.diff_b")]  # fmt: skip
    return names + [
        "time_embedder.linear_1.diff",
        "proj_in.diff_b",
        "token_refiner.refiner_blocks.0.attn.to_q.lora_A.weight",
    ]


def sharded(prefix, index_name, names):
    """A transformer directory of two shards and their index."""
    half = len(names) // 2
    shards = {"model-1.safetensors": names[:half], "model-2.safetensors": names[half:]}
    files = {f"{prefix}{f}": safetensors(n) for f, n in shards.items()}
    index = {"weight_map": {n: f for f, ns in shards.items() for n in ns}}
    return files | {f"{prefix}{index_name}": json.dumps(index)}


def contract(c, shifts=(10.0, 3.0)):
    return {fasth3.CONTRACT[0]: json.dumps(c), fasth3.CONTRACT[1]: json.dumps({"shift": shifts[0]}),
            fasth3.CONTRACT[2]: json.dumps({"shift": shifts[1]})}  # fmt: skip


DECODERS = {f"{d}{n}": safetensors(["w"]) if n.endswith(".safetensors") else "{}"
            for d in (h3.VIDEO_VAE, h3.AUDIO_VAE) for n in ("model.safetensors", "config.json")}  # fmt: skip
SOURCES = {
    H3.release: {"LICENSE": LICENSE, H3.release_index: "{}", "FL2VA/tokenizer/tokenizer.json": "{}",
                 **sharded(h3.TRANSFORMER, "model.safetensors.index.json", native_names()), **DECODERS},
    H3.encoder: {"config.json": "{}", **ENCODER_SHARDS},
    fasth3.STUDENT_8: {**sharded(fasth3.TRANSFORMER, "diffusion_pytorch_model.safetensors.index.json",
                                 student_names()), **contract(CONTRACT_8)},
    fasth3.LORA_4: {fasth3.LORA_FILE: safetensors(bundle_names())},
    fasth3.STUDENT_4[0]: contract(CONTRACT_4, (12.0, 3.0)),
}  # fmt: skip


class FastH3InstallTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.cache, self.models = self.root / "hub", self.root / "models"
        self.engine = []
        for patch in (
            mock.patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)),
            mock.patch.object(
                models,
                "run_engine",
                lambda arguments, check: self.engine.append(arguments),
            ),
        ):
            self.enterContext(patch)
        self.fake = FakeHub(self, self.cache)
        for repo, files in SOURCES.items():
            self.fake.publish(repo, COMMITS[repo], functools.partial(tree, files=files))
        # The commit the bundle's deltas apply to, of the same transformer shards.
        self.fake.publish(
            H3.release,
            BASE,
            functools.partial(tree, files=SOURCES[H3.release]),
            "base",
        )

    def listed(self, name, revision=None):
        return hub.Repository.resolve(name, revision)

    def test_each_variant_reads_the_release_and_its_companion(self):
        release = self.listed(H3.release)
        eight = fasth3.inputs("fasth3-8", release, self.listed(fasth3.STUDENT_8))
        four = fasth3.inputs("fasth3-4", release, self.listed(fasth3.LORA_4))
        transformer = {
            f"{H3.release}/{n}"
            for n in SOURCES[H3.release]
            if n.startswith(h3.TRANSFORMER)
        }
        self.assertTrue(transformer.isdisjoint(eight) and transformer <= set(four))
        self.assertIn(f"{fasth3.STUDENT_8}/transformer/model-1.safetensors", eight)
        self.assertIn(f"{fasth3.STUDENT_4[0]}/{fasth3.CONTRACT[0]}", four)
        self.assertIn(f"{fasth3.LORA_4}/{fasth3.LORA_FILE}", four)
        self.assertEqual(fasth3.pinned("fasth3-8"), h3.PINNED)
        self.assertEqual(
            fasth3.pinned("fasth3-4"),
            h3.PINNED | {fasth3.STUDENT_4[0]: fasth3.STUDENT_4[1]},
        )
        # A release of other transformer shards than the bundle's base is refused.
        changed = SOURCES[H3.release] | sharded(
            h3.TRANSFORMER, "model.safetensors.index.json", native_names()[1:]
        )
        self.fake.publish(H3.release, "5" * 40, functools.partial(tree, files=changed))
        with self.assertRaisesRegex(models.ModelError, f"not the one at {BASE[:12]}"):
            fasth3.inputs(
                "fasth3-4", self.listed(H3.release), self.listed(fasth3.LORA_4)
            )

    def test_the_units_read_what_they_convert_and_write_the_engines_images(self):
        stage = self.root / "stage"
        release = self.listed(H3.release)
        for variant, companion in fasth3.COMPANIONS.items():
            units = fasth3.units(stage, variant, release, self.listed(companion), None)
            dit = f"dit-{variant}"
            outputs = {o for u in units for o in u.outputs if not o.startswith(".")}
            want = {
                f"{dit}/fasth3-w8-block{i:02d}.safetensors"
                for i in range(fasth3.BLOCKS)
            }
            want |= {f"{dit}/qkfix-block{i}.safetensors" for i in fasth3.QKFIX}
            want |= {
                f"{dit}/{n}"
                for n in (
                    "io.safetensors",
                    "prep.safetensors",
                    H3.dit_manifest,
                    H3.adaln,
                )
            }
            want |= {
                o
                for u in h3.decoders(stage, release)
                for o in u.outputs
                if not o.startswith(".")
            }
            self.assertEqual(outputs, want)
            reads = {
                u.name: [(s.repo.name, sorted(n)) for s, n in u.reads()]
                for u in units
                if u.reads
            }
            for unit, found in reads.items():
                self.assertTrue(
                    any(n for _, n in found) or unit.startswith("video VAE"),
                    f"{variant} {unit}",
                )
            block = dict(reads["block 7"])
            if variant == "fasth3-8":
                self.assertEqual(
                    block,
                    {
                        fasth3.STUDENT_8: sorted(
                            f"transformer_blocks.7.{n}" for n in STUDENT_BLOCK
                        )
                    },
                )
            else:
                self.assertEqual(
                    block[H3.release], sorted(f"blocks.7.{n}" for n in NATIVE_BLOCK)
                )
                self.assertEqual(block[fasth3.LORA_4], sorted(n for n in bundle_names() if n.startswith(
                    "transformer_blocks.7.")))  # fmt: skip
                self.assertEqual(
                    dict(reads["time embedding"])[fasth3.LORA_4],
                    ["time_embedder.linear_1.diff"],
                )
                self.assertEqual(
                    dict(reads["patch projections"])[fasth3.LORA_4], ["proj_in.diff_b"]
                )
            # Each block's keyframe row is scratch the tabulation consumes.
            last = units[-1]
            self.assertEqual((last.name, last.discard), ("adaLN tables", (".adaln",)))
            self.assertTrue(
                all(
                    u.consumed_by == last.outputs[0]
                    for u in units
                    if u.name.startswith("block ")
                )
            )

    def test_a_variant_installs_as_a_model_of_its_own(self):
        recipe = []

        def units(stage, variant, release, companion, run_engine):
            recipe.append((variant, companion.name))
            dit = f"dit-{variant}"
            images = [
                f"{dit}/{n}"
                for n in (
                    "fasth3-w8-block00.safetensors",
                    H3.adaln,
                    H3.dit_manifest,
                )
            ]
            others = [
                "vae/v.bin",
                "audio/a.bin",
                "preview/p.bin",
            ]
            write = conversion.Unit(
                "all",
                (*images, *others),
                lambda t: {n: b"" for n in (*images, *others)},
            )
            return [write]

        with (
            mock.patch.object(fasth3, "units", units),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for variant in ("fasth3-4", "fasth3-8"):
                upstream.prepare(
                    selection(self.root, f"{H3.release}:{variant}", language_only=False)
                )
        self.assertEqual(
            recipe, [("fasth3-4", fasth3.LORA_4), ("fasth3-8", fasth3.STUDENT_8)]
        )
        link = self.models / f"{H3.release}:fasth3-4"
        record = assembly.verify(link)
        listed = (self.listed(n) for n in (H3.release, fasth3.LORA_4))
        key = conversion.key(
            fasth3.inputs("fasth3-4", *listed),
            video.recipe_identity(fasth3),
            "fasth3-4",
        )
        self.assertEqual(
            (record["video"]["variants"], record["video"]["key"]), (["fasth3-4"], key)
        )
        self.assertEqual(
            set(record["video"]["sources"]),
            {fasth3.LORA_4, h3.TAEH3_REPO, fasth3.STUDENT_4[0]},
        )
        self.assertIn(
            "weights/dit-fasth3-4/fasth3-w8-block00.safetensors", record["files"]
        )
        # Two variants' entries, and the text encoder's package they share.
        self.assertEqual(len(list((self.models / assembly.CONVERTED).iterdir())), 3)
        # Literal, as the recipe derived them at 0a784b69 on these sources:
        # an installation's entry and record must not change under a refactor.
        self.assertEqual(video.recipe_identity(fasth3), "h3-w8a8-fasth3-v1+r2")
        taeh3 = {h3.TAEH3_REPO: {"repo": h3.TAEH3_REPO, "revision": "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c"}}  # fmt: skip
        for variant, key, sources in (
            ("fasth3-8", "27b209a56e065e562f86817a2eb88dc1a972e943e8309db4c76fd6f8c0850e08",
             {fasth3.STUDENT_8: {"repo": fasth3.STUDENT_8, "revision": "3" * 40}}),
            ("fasth3-4", "1919250bef7a613dd9e62d7765fa092c28a7884263786881d1a1947be4c67693",
             {fasth3.LORA_4: {"repo": fasth3.LORA_4, "revision": "4" * 40},
              fasth3.STUDENT_4[0]: {"repo": fasth3.STUDENT_4[0], "revision": "5ea076f35b84da4c3c82217112fa733d8eea2ae1"}}),
        ):  # fmt: skip
            record = assembly.verify(self.models / f"{H3.release}:{variant}")
            self.assertEqual(record["video"] | {"encoder": None}, {
                "variants": [variant], "key": record["video"]["key"], "sources": sources | taeh3, "encoder": None})  # fmt: skip
            self.assertEqual({n.split("/")[1] for n in record["files"] if n.startswith("weights/dit-")},
                             {f"dit-{variant}"})  # fmt: skip

    def test_a_spec_variant_installs_under_a_key_naming_its_spec(self):
        # Pika's repository at its commit, and the release's transformer at
        # the commit pika-6's checkpoint names.
        pika = sharded(
            "transformer_ref/",
            "diffusion_pytorch_model.safetensors.index.json",
            student_names(),
        )
        pika |= {
            f: safetensors(["w"])
            for f in ("fl2va/adapter.safetensors", "ref2va/adapter.safetensors")
        }
        self.fake.publish(PIKA, "6" * 40, functools.partial(tree, files=pika))
        checkpoint = H3.variants["pika-6"].spec["checkpoint"]["revision"]
        self.fake.publish(
            H3.release,
            checkpoint,
            functools.partial(tree, files=SOURCES[H3.release]),
            "pika",
        )
        recipe = []

        def units(stage, variant, release, companion, run_engine, spec):
            recipe.append((variant, companion.name, spec["name"]))
            names = [
                f"dit-{variant}/{n}"
                for n in (H3.adaln, H3.dit_manifest, "adapter/manifest.json")
            ]
            names += ["vae/v.bin", "audio/a.bin", "preview/p.bin"]
            return [
                conversion.Unit(
                    "all", tuple(names), lambda t: dict.fromkeys(names, b"")
                )
            ]

        with (
            mock.patch.object(fasth3, "units", units),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for variant in ("pika-6", "pika-ref2va-6"):
                upstream.prepare(
                    selection(self.root, f"{H3.release}:{variant}", language_only=False)
                )
        self.assertEqual(
            recipe,
            [("pika-6", PIKA, "pika-6"), ("pika-ref2va-6", PIKA, "pika-ref2va-6")],
        )
        # Literal, as the recipe derives them on these sources, whose files
        # outside LFS the fake Hub names by their Git blob IDs
        # (installer_fixtures.blob_id).
        taeh3 = {
            h3.TAEH3_REPO: {
                "repo": h3.TAEH3_REPO,
                "revision": "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c",
            }
        }
        pika_source = {PIKA: {"repo": PIKA, "revision": "6" * 40}}
        for variant, key, sources in (
            (
                "pika-6",
                "f1c4062aa36d449bd25c646e21c1ab9c7c44e937ccd5b1121e3a3a300937b818",
                pika_source
                | {H3.release: {"repo": H3.release, "revision": checkpoint}},
            ),
            (
                "pika-ref2va-6",
                "e492c5d8c4ec4276bbf42f29d54689b50f14af894083922411f257b0f1e113ec",
                pika_source,
            ),
        ):
            record = assembly.verify(self.models / f"{H3.release}:{variant}")
            self.assertEqual(
                record["video"] | {"encoder": None},
                {
                    "variants": [variant],
                    "key": key,
                    "sources": sources | taeh3,
                    "encoder": None,
                },
            )
            self.assertIn(
                f"weights/dit-{variant}/adapter/manifest.json", record["files"]
            )
        # The spec is in the key: the same files and recipe under another
        # spec name another entry.
        listed = [hub.Repository.resolve(n) for n in (H3.release, PIKA)]
        spec = H3.variants["pika-6"].spec
        inputs = fasth3.inputs("pika-6", *listed, spec=spec)
        self.assertIn(f"{PIKA}/fl2va/adapter.safetensors", inputs)
        self.assertIn(f"{H3.release}/{h3.TRANSFORMER}model-1.safetensors", inputs)
        keys = {
            conversion.key(
                inputs, video.recipe_identity(fasth3), "pika-6", spec | change
            )
            for change in ({}, {"qkfix": [45, 48]}, {"references": True})
        }
        self.assertEqual(len(keys), 3)

    def test_a_spec_file_installs_a_checkpoint_splash_does_not_ship(self):
        files = sharded(
            "transformer/",
            "diffusion_pytorch_model.safetensors.index.json",
            student_names(),
        )
        files["adapter.safetensors"] = safetensors(["w"])
        self.fake.publish(
            "owner/finetune", "7" * 40, functools.partial(tree, files=files)
        )
        spec = {"checkpoint": {"repo": "owner/finetune", "prefix": "transformer/"},
                "adapter": {"file": "adapter.safetensors", "name": "default", "alpha": 64.0},
                "schedule": {k: v for k, v in PIKA_6["schedule"].items() if k != "source"}}  # fmt: skip
        path = self.root / "spec.json"
        recipe = []

        def units(stage, variant, release, companion, run_engine, spec):
            recipe.append((variant, companion.name))
            names = [f"dit-{variant}/{n}" for n in (H3.adaln, H3.dit_manifest)]
            names += ["vae/v.bin", "audio/a.bin", "preview/p.bin"]
            return [
                conversion.Unit(
                    "all", tuple(names), lambda t: dict.fromkeys(names, b"")
                )
            ]

        def install(model, given=None):
            path.write_text(json.dumps(given or spec))
            arguments = [
                "--models",
                str(self.models),
                "--model",
                model if "/" in model else f"{H3.release}:{model}",
            ]
            arguments += ["--spec", str(path)] if given is not False else []
            with (
                contextlib.redirect_stdout(io.StringIO()),
                mock.patch("sys.stderr", io.StringIO()) as stderr,
            ):
                return models.main([*arguments, "prepare"]), stderr.getvalue()

        with mock.patch.object(fasth3, "units", units):
            self.assertEqual(install("tuned"), (0, ""))
            record = assembly.verify(self.models / f"{H3.release}:tuned")
            # Installed again without the file, from the spec it records.
            self.assertEqual(install("tuned", False), (0, ""))
        self.assertEqual(recipe, [("tuned", "owner/finetune")])
        self.assertEqual(record["video"]["variants"], ["tuned"])
        self.assertEqual(
            record["video"]["spec"],
            {"name": "tuned", "references": False, "qkfix": [45, 48, 49]} | spec,
        )
        self.assertEqual(assembly.verify(self.models / f"{H3.release}:tuned"), record)
        schedule = spec["schedule"] | {"sampler": "dpm"}
        for model, given, error in (
            ("pika-6", spec, "pika-6 is a variant Splash ships"),
            ("turbo", spec, "turbo is the DiT of a variant Splash ships"),
            ("tuned", spec | {"sigmas": [1.0]}, "the spec has no field sigmas"),
            ("tuned", spec | {"schedule": schedule}, "a spec's sampler is euler or renoise"),
            ("tuned", ["x"], "expected a JSON object"),
            ("tuned", spec | {"adapter": spec["adapter"] | {"alpha": True}}, "alpha are numbers"),
            ("tuned", spec | {"checkpoint": {"repo": "owner/finetune", "prefix": 1}}, "are strings"),
            ("community/other:tuned", spec, f"--spec installs a variant {H3.release}:NAME"),
            (H3.release, spec, f"--spec installs a variant {H3.release}:NAME"),
        ):  # fmt: skip
            status, stderr = install(model, given)
            self.assertEqual((status, stderr.count("\n")), (1, 1))
            self.assertIn(error, stderr)


class FastH3TensorsTest(unittest.TestCase):
    def test_the_bundle_applies_to_the_base_as_fastvideo_lora_v2(self):
        K = 2
        rng = np.random.default_rng(0)
        qkv = rng.standard_normal((3 * fasth3.H * fasth3.HD, K)).astype(np.float32)
        a, b = (
            rng.standard_normal((4, K)).astype(np.float32),
            rng.standard_normal((fasth3.H * fasth3.HD, 4)),
        )
        gate, norm, diff = (
            rng.standard_normal(s).astype(np.float32) for s in ((7, K), (5,), (5,))
        )
        bf16 = functools.partial(h3.bf16_bits)
        src = fasth3.LoraStudent({
            "blocks.3.attn.qkv_proj.weight": ("BF16", bf16(qkv)), "blocks.3.norm1.weight": ("F32", norm),
            "transformer_blocks.3.attn.to_k.lora_A.weight": ("F32", a),
            "transformer_blocks.3.attn.to_k.lora_B.weight": ("F32", b.astype(np.float32)),
            "transformer_blocks.3.attn.to_gate_compress.set_weight": ("F32", gate),
            "transformer_blocks.3.norm1.diff": ("F32", diff),
        })  # fmt: skip
        rows = qkv.reshape(fasth3.H, 3, fasth3.HD, K)[:, 1].reshape(
            -1, K
        )  # per-head interleaved [q | k | v]
        k = h3.to_f32("BF16", h3.bf16_bits(rows))

        def student(x):  # merged in fp32, held in bf16 as the student's checkpoint
            return h3.to_f32("BF16", h3.bf16_bits(x))

        merged = student(k + b.astype(np.float32) @ a)
        self.assertFalse(np.array_equal(merged, k + b.astype(np.float32) @ a))
        np.testing.assert_array_equal(
            src.get("transformer_blocks.3.attn.to_k.weight"), merged
        )
        # Merged by rows, as the units convert, the same.
        n, rows = src.rows("transformer_blocks.3.attn.to_k.weight")
        np.testing.assert_array_equal(np.concatenate([rows(r, min(r + 1000, n)) for r in range(0, n, 1000)]),
                                      merged)  # fmt: skip
        np.testing.assert_array_equal(
            fasth3.swapped_rows(n, rows)(0, n), h3.swap_halves(rows(0, n))
        )
        np.testing.assert_array_equal(
            src.get("transformer_blocks.3.attn.to_gate_compress.weight"), student(gate)
        )
        np.testing.assert_array_equal(
            src.get("transformer_blocks.3.norm1.weight"), student(norm + diff)
        )

    def test_the_contract_and_its_schedule_are_the_students(self):
        files = {k: json.loads(v) for k, v in contract(CONTRACT_8).items()}
        c = fasth3.contract(files)
        self.assertEqual(
            (c["dmd_rungs"][0], c["video_shift"], c["vsa_sparsity"]), (999, 10.0, 0.8)
        )
        # a reference conversion's dit-fasth3-8 manifest.json adaln_timesteps
        self.assertEqual(fasth3.timesteps(c["dmd_rungs"], 10.0, 3.0).tolist()[:4], [
            0.00010013580322265625, 0.00033360719680786133, 0.014211654663085938, 0.032424747943878174])  # fmt: skip
        four = fasth3.contract(
            {k: json.loads(v) for k, v in contract(CONTRACT_4, (12.0, 3.0)).items()}
        )
        self.assertEqual(fasth3.timesteps(four["dmd_rungs"], four["video_shift"], 3.0).tolist(), [
            8.344650268554688e-05, 0.00033360719680786133, 0.02716749906539917, 0.07692307233810425,
            0.10048043727874756, 0.19999998807907104, 0.25, 0.5])  # fmt: skip
        files[fasth3.CONTRACT[1]] = {"shift": 12.0}
        with self.assertRaisesRegex(models.ModelError, "disagrees"):
            fasth3.contract(files)

    def test_each_variant_keeps_the_manifest_it_was_installed_with(self):
        # The DiT's manifest of a contract, QK norms and gains, byte for byte,
        # as the recipe wrote it at 0a784b69.
        src = {f"transformer_blocks.{b}.attn.norm_{p}.weight": qk_norm(b, p) for b in fasth3.QKFIX for p in "qk"}  # fmt: skip
        gains = [float(i) for i in range(fasth3.BLOCKS)]
        for variant, c, shifts, source, digest in (
            ("fasth3-8", CONTRACT_8, (10.0, 3.0), fasth3.STUDENT_8, "1b65af98c373849d12327aed0ce1a3de1fe367ea57c9effabe369f9524d9364b"),
            ("fasth3-4", CONTRACT_4, (12.0, 3.0), f"{fasth3.LORA_4}/{fasth3.LORA_FILE}", "9ae32f5974981ab8844c3daaec1e3aea71062cc1e2bc24c7dd1aeb94536181c5"),
        ):  # fmt: skip
            ctr = fasth3.contract(
                {k: json.loads(v) for k, v in contract(c, shifts).items()}
            )
            table = fasth3.timesteps(
                ctr["dmd_rungs"], ctr["video_shift"], ctr["audio_shift"]
            )
            data = fasth3.dit_manifest(variant, ctr, table, gains, src, source)
            manifest = json.loads(data)
            self.assertEqual((manifest["format"], manifest["variant"], manifest["source"]),
                             ("h3-w8a8-fasth3-v1", variant, source))  # fmt: skip
            self.assertEqual(manifest["qkfix"]["blocks"]["45"], {
                "file": "qkfix-block45.safetensors", "split": [126, 127, -1, -1], "gain": 2256.0,
                "gq_max": 47.0, "gk_max": 48.0})  # fmt: skip
            self.assertEqual(sorted(manifest["qkfix"]["blocks"]), ["45", "48", "49"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, variant)


# A small DiT the units run on (FastH3OutputsTest): its heads, width and MLP
# width (h3.H, h3.D, h3.F), the time embedding's width, the adaLN rows, the
# text and patch widths, and the bundle's rank.
SMALL = {"H": 2, "D": 128, "F": 384}
TD, ADA, TEXT, PATCH, RANK = 32, 36, 16, 16, 4


def small(seed, shapes, f32=()):
    """Random tensors of shapes, name -> (dtype, array): F32 for the names
    starting with one of f32, else BF16; a block's QK norms give the QKFIX
    blocks gains above h3.QKFIX_GAIN and the others gains below 3."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in shapes.items():
        x = rng.standard_normal(shape).astype(np.float32)
        norm = (
            r"(?:transformer_)?blocks\.(\d+)\.attn\.(?:norm_)?([qk])(?:_norm)?\.weight"
        )
        if m := re.fullmatch(norm, name):
            b = int(m.group(1))
            x = (
                qk_norm(b, m.group(2))
                if b in fasth3.QKFIX
                else rng.uniform(0.5, 1.5, shape).astype(np.float32)
            )
        out[name] = (
            ("F32", x) if name.startswith(tuple(f32)) else ("BF16", h3.bf16_bits(x))
        )
    return out  # fmt: skip


def small_linears(native):
    """A transformer or refiner block's linears and norms, name -> shape."""
    H, HD, D, F = SMALL["H"], h3.HD, SMALL["D"], SMALL["F"]
    if native:
        return {"attn.qkv_proj.weight": (3 * H * HD, D), "attn.out_proj.weight": (D, H * HD),
                "mlp.fc1.weight": (2 * F, D), "mlp.fc2.weight": (D, F), "norm1.weight": (D,), "norm2.weight": (D,),
                "attn.q_norm.weight": (HD,), "attn.k_norm.weight": (HD,)}  # fmt: skip
    return {**{f"attn.to_{p}.weight": (H * HD, D) for p in "qkv"}, "attn.to_out.0.weight": (D, H * HD),
            "ff.net.0.proj.weight": (2 * F, D), "ff.net.2.weight": (D, F), "norm1.weight": (D,), "norm2.weight": (D,),
            "attn.norm_q.weight": (HD,), "attn.norm_k.weight": (HD,)}  # fmt: skip


def small_native():
    """FL2VA's transformer (native names): bf16, but the time embedder,
    patch projections and output heads fp32."""
    D, ada = (
        SMALL["D"],
        {"adaln_proj.linear.weight": (ADA, TD), "adaln_proj.linear.bias": (ADA,)},
    )
    s = {
        f"blocks.{i}.{n}": v
        for i in range(fasth3.BLOCKS)
        for n, v in (small_linears(True) | ada).items()
    }
    s |= {
        f"token_refiner.blocks.{b}.{n}": v
        for b in range(2)
        for n, v in small_linears(True).items()
    }
    s |= {
        "time_embedder.proj_in.weight": (TD, 256),
        "time_embedder.proj_in.bias": (TD,),
        "time_embedder.proj_out.weight": (TD, TD),
        "time_embedder.proj_out.bias": (TD,),
        "condition_proj.weight": (D, TEXT),
        "condition_proj.bias": (D,),
        "final_layer.adaln_proj.linear.weight": (2 * D, TD),
        "final_layer.adaln_proj.linear.bias": (2 * D,),
        "token_refiner.final_norm.weight": (D,),
        "final_layer.norm.weight": (D,),
    }
    for p in ("video", "audio"):
        s |= {
            f"{p}_patch_proj.weight": (D, PATCH),
            f"{p}_patch_proj.bias": (D,),
            f"final_layer.{p}_out.weight": (PATCH, D),
            f"final_layer.{p}_out.bias": (PATCH,),
        }
    return small(1, s, ("time_embedder.", "video_patch", "audio_patch", "final_layer.video_out", "final_layer.audio_out"))  # fmt: skip


def small_diffusers(seed, gate, f32=()):
    """A diffusers-named trunk: a FastH3 student (with its compression
    gate) or Pika's Ref2VA base."""
    H, HD, D = SMALL["H"], h3.HD, SMALL["D"]
    block = small_linears(False) | {
        "adaln_proj.linear.weight": (ADA, TD),
        "adaln_proj.linear.bias": (ADA,),
    }
    block |= {"attn.to_gate_compress.weight": (H * HD, D)} if gate else {}
    s = {
        f"transformer_blocks.{i}.{n}": v
        for i in range(fasth3.BLOCKS)
        for n, v in block.items()
    }
    s |= {
        f"token_refiner.refiner_blocks.{b}.{n}": v
        for b in range(2)
        for n, v in small_linears(False).items()
    }
    for name, shape in (
        ("time_embedder.linear_1", (TD, 256)),
        ("time_embedder.linear_2", (TD, TD)),
        ("context_embedder", (D, TEXT)),
        ("norm_out.linear", (2 * D, TD)),
        ("proj_in", (D, PATCH)),
        ("audio_proj_in", (D, PATCH)),
        ("proj_out", (PATCH, D)),
        ("audio_proj_out", (PATCH, D)),
    ):
        s |= {f"{name}.weight": shape, f"{name}.bias": shape[:1]}
    return small(seed, s | {"norm_out.norm.weight": (D,), "token_refiner.final_norm.weight": (D,)}, f32)  # fmt: skip


def small_bundle():
    """fasth3-4's LoRA bundle (fp32): a LoRA, set_weight, diff and diff_b
    per block, and a delta of the time embedder, the patch projection and the
    token refiner."""
    H, HD, D = SMALL["H"], h3.HD, SMALL["D"]
    s = {"time_embedder.linear_1.diff": (TD, 256), "proj_in.diff_b": (D,),
         "token_refiner.refiner_blocks.0.attn.to_q.lora_A.weight": (RANK, D),
         "token_refiner.refiner_blocks.0.attn.to_q.lora_B.weight": (H * HD, RANK)}  # fmt: skip
    for i in range(fasth3.BLOCKS):
        t = f"transformer_blocks.{i}."
        s |= {t + "attn.to_q.lora_A.weight": (RANK, D), t + "attn.to_q.lora_B.weight": (H * HD, RANK),
              t + "attn.to_gate_compress.set_weight": (H * HD, D), t + "norm1.diff": (D,),
              t + "adaln_proj.linear.diff_b": (ADA,)}  # fmt: skip
    return small(3, s, ("",))


def small_adapter(seed, rank=RANK, extra=()):
    """A PEFT adapter of Pika's (fp32, adapter name default): a LoRA of
    each block's and each token-refiner block's q / k / v, out, fc1 and fc2,
    and of the modules extra."""
    H, HD, D, F = SMALL["H"], h3.HD, SMALL["D"], SMALL["F"]
    io = {**{f"attn.to_{p}": (D, H * HD) for p in "qkv"}, "attn.to_out.0": (H * HD, D),
          "ff.net.0.proj": (D, 2 * F), "ff.net.2": (F, D)}  # fmt: skip
    blocks = [f"transformer_blocks.{i}." for i in range(fasth3.BLOCKS)]
    modules = {p + m: v for p in blocks + ["token_refiner.refiner_blocks.0.", "token_refiner.refiner_blocks.1."]
               for m, v in io.items()} | dict(extra)  # fmt: skip
    shapes = {}
    for m, (k, n) in modules.items():
        shapes |= {
            f"{m}.lora_A.default.weight": (rank, k),
            f"{m}.lora_B.default.weight": (n, rank),
        }
    return small(seed, shapes, ("",))


def small_sharded(prefix, index_name, stem, tensors):
    """A transformer directory of tensors in two shards, stem-1 and stem-2,
    and their index."""
    names = sorted(tensors)
    shards = {
        f"{stem}-1.safetensors": names[: len(names) // 2],
        f"{stem}-2.safetensors": names[len(names) // 2 :],
    }
    files = {
        f"{prefix}{f}": h3.safetensors({n: tensors[n] for n in ns})
        for f, ns in shards.items()
    }
    return files | {f"{prefix}{index_name}": json.dumps({"weight_map": {n: f for f, ns in shards.items() for n in ns}})}  # fmt: skip


PIKA_6 = H3.variants["pika-6"].spec


class FastH3OutputsTest(unittest.TestCase):
    """Each variant's units run on a small DiT's fake repositories: every
    file they write, the entry's and the scratch the tabulation reads, by
    SHA-256."""

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        for module in (h3, fasth3):
            self.enterContext(mock.patch.multiple(module, **SMALL))
        self.fake = FakeHub(self, self.root / "hub")
        release = small_sharded(
            h3.TRANSFORMER, "model.safetensors.index.json", "model", small_native()
        )
        student = small_sharded(fasth3.TRANSFORMER, "diffusion_pytorch_model.safetensors.index.json",
                                "diffusion_pytorch_model", small_diffusers(2, gate=True))  # fmt: skip
        for repo, commit, files in (
            (H3.release, RELEASE, release | DECODERS),
            (fasth3.STUDENT_8, COMMITS[fasth3.STUDENT_8], student | contract(CONTRACT_8)),
            (fasth3.LORA_4, COMMITS[fasth3.LORA_4], {fasth3.LORA_FILE: h3.safetensors(small_bundle())}),
            (fasth3.STUDENT_4[0], fasth3.STUDENT_4[1], contract(CONTRACT_4, (12.0, 3.0))),
        ):  # fmt: skip
            self.fake.publish(repo, commit, functools.partial(tree, files=files))
        # pika-6's checkpoint, the release's transformer at its commit, and
        # Pika's Ref2VA base (diffusers names, its fp32 modules fp32) and
        # adapters.
        self.fake.publish(
            H3.release,
            PIKA_6["checkpoint"]["revision"],
            functools.partial(tree, files=release),
            "pika",
        )
        ref = small_sharded(
            "transformer_ref/",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model",
            small_diffusers(
                4,
                gate=False,
                f32=(
                    "time_embedder.",
                    "proj_in.",
                    "audio_proj_in.",
                    "proj_out.",
                    "audio_proj_out.",
                ),
            ),
        )
        adapters = {
            "fl2va/adapter.safetensors": h3.safetensors(
                small_adapter(5), {"format": "pt"}
            ),
            "ref2va/adapter.safetensors": h3.safetensors(small_adapter(6)),
        }
        self.fake.publish(PIKA, "6" * 40, functools.partial(tree, files=ref | adapters))  # fmt: skip

    def written(self, variant, companion, spec=None):
        """path -> SHA-256 of each file variant's units write, the decoders'
        aside; the fake engine's adaLN tables list the scratch it reads."""
        stage, release = self.root / variant, hub.Repository.resolve(H3.release)

        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        def engine(arguments):
            scratch = Path(arguments[1])
            read = [
                f"{p.relative_to(scratch)} {digest(p)}\n"
                for p in sorted(scratch.rglob("*"))
                if p.is_file()
            ]
            (scratch / f"dit-{variant}" / H3.adaln).write_text("".join(read))

        skip = {u.name for u in h3.decoders(stage, release)} | {"audio VAE encoder"}
        units = [u for u in fasth3.units(stage, variant, release, hub.Repository.resolve(companion), engine, spec)
                 if u.name not in skip]  # fmt: skip
        with contextlib.redirect_stdout(io.StringIO()):
            conversion.convert(stage, units, variant)
        return {
            p.relative_to(stage).as_posix(): digest(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }

    def test_each_variant_writes_the_bytes_it_wrote_at_0a784b69(self):
        # Literal, as the units wrote them at 1a53efd7 (fasth3.py as at
        # 0a784b69 but its QKFIX check): a refactor must not change a byte.
        for variant, companion, count, digest in (
            ("fasth3-8", fasth3.STUDENT_8, 58, "f21e0fb8cb1766fe03a0318e9e4f58975404a200287ae874d5ce3b30cfcd39ba"),
            ("fasth3-4", fasth3.LORA_4, 58, "46aca984bf3a496a3e99c17b98b20a4dd71976734eff9cbf14f29eb8a75e3614"),
        ):  # fmt: skip
            files = self.written(variant, companion)
            listing = "".join(f"{p} {d}\n" for p, d in files.items())
            self.assertEqual(len(files), count, variant)
            self.assertEqual(
                hashlib.sha256(listing.encode()).hexdigest(), digest, variant
            )

    def test_a_spec_writes_a_dense_dit_and_its_runtime_adapter(self):
        # pika-6's and pika-ref2va-6's files on these sources, literal. Their
        # blocks, QKFIX files, io, prep tensors, scratch rows and adapter
        # files are byte for byte what the dev tools they replace
        # (h3_base_dense.py, h3_lora_convert.py at 0a784b69) wrote; each
        # adapter manifest names its source by the blob ID the fake Hub lists
        # (installer_fixtures.blob_id).
        for variant, count, digest in (
            ("pika-6", 110, "95be911d5100ad0e25da7a0d094c173b91361f71ec6ffb4b14508ea7c33073ff"),
            ("pika-ref2va-6", 110, "eafef027c4e953049b54aaf2381a9b38b2a7a682fb0b2c328c54e68d3dab885d"),
        ):  # fmt: skip
            files = self.written(variant, PIKA, H3.variants[variant].spec)
            listing = "".join(f"{p} {d}\n" for p, d in files.items())
            dit = self.root / variant / f"dit-{variant}"
            manifest = json.loads((dit / H3.dit_manifest).read_text())
            self.assertEqual(
                (manifest["vsa"]["attention"], manifest["vsa"]["sparsity"]),
                ("dense", 0.0),
            )
            self.assertEqual(
                manifest["schedule"]["knots"], [1000, 750, 500, 250, 100, 40, 0]
            )
            self.assertEqual(manifest["adapters"], [{"path": "adapter", "scale": 1.0}])
            adapter = json.loads((dit / "adapter" / H3.dit_manifest).read_text())
            self.assertEqual(
                (adapter["rank"], adapter["scale"], adapter["refiner_blocks"]),
                (128, 32.0, 2),
            )
            self.assertIn(f"dit-{variant}/adapter/lora-block49.safetensors", files)
            self.assertEqual(len(files), count, variant)
            self.assertEqual(
                hashlib.sha256(listing.encode()).hexdigest(), digest, variant
            )

    def test_a_spec_the_recipe_or_the_engine_cannot_run_is_refused(self):
        release, pika = hub.Repository.resolve(H3.release), hub.Repository.resolve(PIKA)
        schedule = PIKA_6["schedule"]
        for change, error in (
            ({"schedule": schedule | {"guidance_scale": 3.5}}, "no CFG"),
            ({"schedule": schedule | {"sampler": "dpm++"}}, "euler or renoise"),
            ({"schedule": schedule | {"sigmas": [1.0, 0.5]}}, "no field sigmas"),
            ({"schedule": schedule | {"rungs": [1000, 1000, 40]}}, "strictly decreasing"),
            ({"schedule": schedule | {"vsa_sparsity": 0.9}}, "VSA sparsity is 0"),
            ({"guidance": 1}, "no field guidance"),
        ):  # fmt: skip
            with self.assertRaisesRegex(models.ModelError, error):
                fasth3.units(self.root, "pika-6", release, pika, None, PIKA_6 | change)
        # An adapter above the engine's rank, or adapting an adaLN projection.
        for adapter, error in (
            (small_adapter(5, rank=256), "rank 256, above the engine's 128"),
            (
                small_adapter(
                    5, extra={"transformer_blocks.0.adaln_proj.linear": (TD, ADA)}
                ),
                "no adaLN",
            ),
        ):
            self.fake.publish(
                PIKA,
                "8" * 40,
                functools.partial(
                    tree, files={"fl2va/adapter.safetensors": h3.safetensors(adapter)}
                ),
            )
            with self.assertRaisesRegex(models.ModelError, error):
                fasth3.units(
                    self.root,
                    "pika-6",
                    release,
                    hub.Repository.resolve(PIKA, "8" * 40),
                    None,
                    PIKA_6,
                )
        # A checkpoint that is not MiniMax-H3's DiT.
        with (
            mock.patch.object(fasth3, "F", 512),
            self.assertRaisesRegex(models.ModelError, "not MiniMax-H3's DiT"),
        ):
            fasth3.units(self.root, "pika-6", release, pika, None, PIKA_6)
        # Blocks whose QK gains are not the spec's QKFIX blocks.
        units = fasth3.units(
            self.root, "pika-6", release, pika, None, PIKA_6 | {"qkfix": [45, 48]}
        )
        (manifest,) = [u for u in units if u.name == "manifest"]
        with self.assertRaisesRegex(
            models.ModelError, r"block 49 of the base DiT .* not \[45, 48\]"
        ):
            manifest.run(conversion.fetch(manifest.reads(), ThreadPoolExecutor(1)))  # fmt: skip

    def test_a_spec_with_references_writes_the_audio_vaes_encoder(self):
        release = hub.Repository.resolve(H3.release)
        units = fasth3.units(self.root, "pika-ref2va-6", release, hub.Repository.resolve(PIKA), None,
                             H3.variants["pika-ref2va-6"].spec)  # fmt: skip
        (unit,) = [u for u in units if u.name == "audio VAE encoder"]
        self.assertEqual(
            [(s.name, n) for s, n in unit.reads()],
            [(h3.AUDIO_VAE + "model.safetensors", [])],
        )
        weights = {"encoder.block.0.weight_g": ("F32", np.ones(2, np.float32))}
        with mock.patch.object(
            h3,
            "audio_encoder",
            lambda t, config: ([b"".join(t.get(n).tobytes() for n in t)], b"{}"),
        ):
            self.assertEqual(unit.run(weights), {"audio/audio_enc_f16.bin": [np.ones(2, np.float32).tobytes()],
                                                 "audio/manifest_enc.json": b"{}"})  # fmt: skip


if __name__ == "__main__":
    unittest.main()

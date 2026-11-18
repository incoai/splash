"""A LiLiCorr draft GGUF converted into a DFlash2 checkpoint: the
configuration from its header, its NVFP4, Q8_0, BF16 and F32 tensors
dequantized, scaled and rounded to BF16, its KV heads repeated, and the
checkpoint an installation derives from it once (assembly.derived_draft).
The GGUFs are small synthetic ones laid out like
nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF's."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dev.tests.fixture_files import write_gguf
from dev.tests.installer_fixtures import (
    BOLT,
    CONFIGS,
    fixture,
    tensor_table,
    write_target,
)
from install import assembly, dflash_gguf, hub, models, upstream

F32, F16, Q8_0, BF16, NVFP4 = 0, 1, 8, 30, 40
HIDDEN = 64
# The Bolt release's header, but one layer.
METADATA = {
    "general.architecture": "dflash",
    "dflash.block_count": 1,
    "dflash.embedding_length": 4096,
    "dflash.feed_forward_length": 12288,
    "dflash.attention.head_count": 32,
    "dflash.attention.head_count_kv": 2,
    "dflash.attention.key_length": 128,
    "dflash.rope.freq_base": 10000.0,
    "dflash.block_size": 8,
    "dflash.conv_kernel_size": 2,
    "dflash.conv_group_size": 16,
    "dflash.target_layers": [2, 19, 36, 52, 69, 86],
    "dflash.attention.sliding_window": 1024,
    "lilicorr.hidden_size": 1024,
    "lilicorr.candidate_topk": 8,
    "lilicorr.num_heads": 8,
    "lilicorr.num_layers": 2,
}


def nvfp4_block(scales, nibbles):
    """One NVFP4 block: four E4M3 scale bytes, then 64 four-bit codes,
    each 16-value group's first eight in its bytes' low nibbles."""
    packed = bytearray()
    for group in range(4):
        codes = nibbles[group * 16 : group * 16 + 16]
        packed += bytes(int(codes[j] | codes[j + 8] << 4) for j in range(8))
    return bytes(int(s) for s in scales) + bytes(packed)


def bf16(values):
    """BF16 words of values, rounded to nearest even."""
    words = np.asarray(values, np.float32).view(np.uint32)
    return ((words + 0x7FFF + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def rounded(values):
    return (bf16(values).astype(np.uint32) << 16).view(np.float32)


def lilicorr_gguf(path, metadata=METADATA, *, without=()):
    """A LiLiCorr GGUF: blk.0.attn_k NVFP4 [2 heads x 2 rows, 64] with a
    scale of 2, fc Q8_0 [4, 64] with a scale of 0.5, the norms F32 and
    every other weight BF16 [4, 64], each of distinct values."""
    rng = np.random.default_rng(0)
    tensors = []
    for _, source in dflash_gguf._tensors(metadata["dflash.block_count"]):
        if source in without:
            continue
        if source == "blk.0.attn_k.weight":
            raw = b"".join(
                nvfp4_block(rng.integers(0x30, 0x48, 4), rng.integers(0, 16, 64))
                for _ in range(4)
            )
            tensors.append((source, [HIDDEN, 4], NVFP4, raw))
            tensors.append(("blk.0.attn_k.scale", [1], F32, struct.pack("<f", 2.0)))
        elif source == "fc.weight":
            raw = b"".join(
                struct.pack("<e", 0.25 * (row + 1))
                + rng.integers(-128, 128, 32, dtype=np.int8).tobytes()
                for row in range(8)
            )
            tensors.append((source, [HIDDEN, 4], Q8_0, raw))
            tensors.append(("fc.scale", [1], F32, struct.pack("<f", 0.5)))
        elif "norm" in source:
            values = rng.standard_normal(HIDDEN).astype(np.float32)
            tensors.append((source, [HIDDEN], F32, values.tobytes()))
        else:
            values = rng.standard_normal((4, HIDDEN)).astype(np.float32)
            tensors.append((source, [HIDDEN, 4], BF16, bf16(values).tobytes()))
    return write_gguf(path, metadata, tensors)


def read_safetensors(path):
    """name -> (shape, float32 values) of a BF16 safetensors file."""
    data = path.read_bytes()
    length = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8 : 8 + length])
    tensors = {}
    for name, entry in header.items():
        begin, end = (8 + length + offset for offset in entry["data_offsets"])
        words = np.frombuffer(data[begin:end], np.uint16).astype(np.uint32) << 16
        tensors[name] = words.view(np.float32).reshape(entry["shape"])
    return tensors


class ConversionTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def convert(self, source):
        stage = self.root / "stage"
        stage.mkdir()
        dflash_gguf.write_checkpoint(source, stage)
        return json.loads((stage / "config.json").read_text()), read_safetensors(
            stage / "model.safetensors"
        )

    def test_dequantizes_scales_repeats_and_rounds(self):
        source = lilicorr_gguf(self.root / "draft.gguf")
        config, tensors = self.convert(source)
        self.assertEqual(
            config["dflash_config"]["target_layer_ids"], [1, 18, 35, 51, 68, 85]
        )
        self.assertEqual(config["num_key_value_heads"], 8)
        self.assertEqual(config["dflash_config"]["mask_token_id"], 990)
        self.assertEqual(config["dflash_config"]["projector_type"], "lilicorr")
        self.assertEqual(set(tensors), {name for name, _ in dflash_gguf._tensors(1)})
        raw = source.read_bytes()

        def stored(name, size):
            header = dflash_gguf.gguf.Metadata(source, tensors=True)
            alignment = 32
            data = (len(header.raw) + alignment - 1) // alignment * alignment
            offset = data + header.offsets[name]
            return raw[offset : offset + size]

        # NVFP4: each block's 16-value groups, halved E4M3 scale times the
        # doubled E2M1 code, low nibbles first; then the tensor's scale; then
        # each of the two heads' rows four times.
        e2m1 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
        blocks = stored("blk.0.attn_k.weight", 4 * 36)
        expected = []
        for block in range(4):
            data = blocks[block * 36 : block * 36 + 36]
            for group in range(4):
                exponent, mantissa = data[group] >> 3, data[group] & 7
                scale = (1 + mantissa / 8) * 2.0 ** (exponent - 7) * 0.5
                for j in range(16):
                    byte = data[4 + group * 8 + j % 8]
                    code = byte & 15 if j < 8 else byte >> 4
                    expected.append(e2m1[code] * scale * 2.0)
        rows = np.array(expected, np.float32).reshape(4, HIDDEN)
        heads = rows.reshape(2, 2, HIDDEN)
        repeated = np.repeat(heads, 4, axis=0).reshape(16, HIDDEN)
        np.testing.assert_array_equal(
            tensors["layers.0.self_attn.k_proj.weight"], rounded(repeated)
        )
        # Q8_0: an f16 scale and 32 int8 values a block, times the tensor's
        # scale, rounded to nearest even.
        blocks = stored("fc.weight", 8 * 34)
        values = []
        for block in range(8):
            data = blocks[block * 34 : block * 34 + 34]
            scale = struct.unpack_from("<e", data)[0]
            values += [scale * q * 0.5 for q in np.frombuffer(data[2:], np.int8)]
        values = np.array(values, np.float32).reshape(4, HIDDEN)
        np.testing.assert_array_equal(tensors["fc.weight"], rounded(values))
        # BF16 stays as it is; F32 rounds to nearest even.
        norm = np.frombuffer(stored("output_norm.weight", 4 * HIDDEN), np.float32)
        np.testing.assert_array_equal(tensors["norm.weight"], rounded(norm))
        # A value just past half a BF16 step rounds up, one short of it down.
        self.assertEqual(rounded([1 + 3 * 2**-9, 1 + 2**-9]).tolist(), [1 + 2**-7, 1])

    def test_conversion_is_deterministic(self):
        source = lilicorr_gguf(self.root / "draft.gguf")
        outputs = []
        for name in ("a", "b"):
            stage = self.root / name
            stage.mkdir()
            dflash_gguf.write_checkpoint(source, stage)
            outputs.append(
                [(stage / f).read_bytes() for f in ("config.json", "model.safetensors")]
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_other_drafts_are_refused(self):
        for metadata, without, refusal in (
            (METADATA | {"general.architecture": "llama"}, (), "not a LiLiCorr"),
            (
                {k: v for k, v in METADATA.items() if k != "lilicorr.hidden_size"},
                (),
                "not a LiLiCorr",
            ),
            (METADATA | {"dflash.attention.head_count_kv": 3}, (), "3 KV heads"),
            (METADATA, ("blk.0.attn_q.weight",), "lacks tensors: blk.0.attn_q"),
        ):
            with self.subTest(refusal=refusal):
                source = lilicorr_gguf(
                    self.root / "draft.gguf", metadata, without=without
                )
                stage = self.root / "stage"
                stage.mkdir(exist_ok=True)
                with self.assertRaisesRegex(models.ModelError, refusal):
                    dflash_gguf.write_checkpoint(source, stage)


class DraftFilesTest(unittest.TestCase):
    """A repository holding one LiLiCorr GGUF and no config.json is a draft
    whose configuration the engine checks before the GGUF downloads."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        configs = CONFIGS / BOLT.name.lower()
        headers = self.root / "target"
        headers.mkdir()
        write_target(headers / "model.gguf", fixture(family=BOLT), tensor_table(BOLT))
        self.target = upstream.Target(
            "gguf",
            "none",
            configs / "config.json",
            headers,
            BOLT,
            {},
        )

    def files(self, metadata):
        release = self.root / "release"
        release.mkdir(exist_ok=True)
        lilicorr_gguf(release / "draft.gguf", metadata)
        (release / "README.md").write_text("LiLiCorr")
        repo = hub.Repository.local_directory(release)
        return upstream.draft_files(repo, self.target)

    def test_the_gguf_is_linked_once_its_configuration_is_accepted(self):
        files = self.files(METADATA)
        self.assertEqual(set(files), {"draft/draft.gguf"})
        with self.assertRaisesRegex(
            models.ModelError, "draft configuration is incompatible with Nemotron"
        ):
            self.files(METADATA | {"dflash.embedding_length": 2048})
        with self.assertRaisesRegex(
            models.ModelError, "draft.gguf: it is not a LiLiCorr"
        ):
            self.files(METADATA | {"general.architecture": "llama"})


class DerivedDraftTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = lilicorr_gguf(self.root / "draft.gguf")
        self.files = {"draft/draft.gguf": self.source}
        self.records = {self.source: assembly.file_record(self.source)}

    def derive(self):
        models_root = self.root / "models"
        models_root.mkdir(exist_ok=True)
        with models.installation_lock(models_root):
            return assembly.derived_draft(models_root, self.files, self.records)

    def test_derived_once_and_rebuilt_when_damaged(self):
        derived = self.derive()
        self.assertEqual(set(derived), {"draft/config.json", "draft/model.safetensors"})
        key = assembly.draft_key({"draft/draft.gguf": self.records[self.source]})
        self.assertEqual(derived["draft/config.json"].parent.name, key)
        weights = derived["draft/model.safetensors"]
        written = weights.stat().st_mtime_ns
        self.assertEqual(self.derive(), derived)
        self.assertEqual(weights.stat().st_mtime_ns, written)
        weights.write_bytes(b"damaged")
        self.derive()
        self.assertNotEqual(weights.read_bytes(), b"damaged")
        # A checkpoint released as one is not derived.
        self.files = {"draft/model.safetensors": self.source}
        self.assertEqual(self.derive(), {})

    def test_key_follows_the_source_and_the_converter(self):
        record = self.records[self.source]
        key = assembly.draft_key({"draft/draft.gguf": record})
        self.assertIsNone(assembly.draft_key({"draft/model.safetensors": record}))
        self.assertNotEqual(
            assembly.draft_key({"draft/draft.gguf": record | {"digest": "0" * 64}}),
            key,
        )
        with mock.patch.object(
            assembly.models, "sha256", side_effect=lambda path: "1" * 64
        ):
            self.assertNotEqual(assembly.draft_key({"draft/draft.gguf": record}), key)

    def test_a_start_rederives_a_draft_of_another_converter(self):
        derived = self.derive()
        draft = hub.Repository.local_directory(self.root)
        record = self.records[self.source]
        installed = {
            "family": BOLT.name,
            "target_format": "mlx-affine",
            "sources": {"draft": draft.identity()},
            "files": {
                "draft/draft.gguf": record,
                "draft/config.json": {"path": str(derived["draft/config.json"])},
            },
        }
        self.assertEqual(upstream._changes(installed, BOLT, draft), [])
        installed["files"]["draft/config.json"]["path"] = "/m/.metadata/old/config.json"
        self.assertEqual(
            upstream._changes(installed, BOLT, draft), ["the draft converter changed"]
        )


if __name__ == "__main__":
    unittest.main()

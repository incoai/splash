"""Synthetic, stdlib-only tests for the raw Flash-Next checkpoint importer."""

import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tools import import_flash_next as importer

ALIGNMENT = 16 * 1024


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_shard(path, tensors, *, raw_header=None, raw_payload=None):
    """Build safetensors independently; malformed fixtures need no SDK writer."""
    header = {"__metadata__": {"format": "mlx"}}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    if raw_header is None:
        raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        raw_header += b" " * (-len(raw_header) % 8)
    if raw_payload is None:
        raw_payload = bytes(payload)
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + raw_payload)
    return header


def rewrite_header(path, mutate):
    raw = path.read_bytes()
    size = struct.unpack_from("<Q", raw)[0]
    header = json.loads(raw[8 : 8 + size])
    mutate(header)
    write_shard(
        path,
        [],
        raw_header=json.dumps(header).encode("utf-8"),
        raw_payload=raw[8 + size :],
    )


class FlashNextImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="splash-flash-import-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.destination = self.root / "bundle"
        self.source.mkdir()
        self.first_shard = "model-00001-of-00002.safetensors"
        self.second_shard = "model-00002-of-00002.safetensors"
        self.shards = {
            self.first_shard: [
                (
                    "model.layers.0.mlp.experts.gate_proj.weight",
                    "U32",
                    [4, 8],
                    struct.pack(
                        "<32I", *([0x76543210, 0xFEDCBA98, 0x01234567, 0x89ABCDEF] * 8)
                    ),
                ),
                (
                    "model.layers.0.mlp.experts.gate_proj.scales",
                    "BF16",
                    [4, 2],
                    struct.pack(
                        "<8H",
                        0x3F80,
                        0x3C07,
                        0x0080,
                        0x4001,
                        0x3D55,
                        0x3F01,
                        0x4040,
                        0x3EAA,
                    ),
                ),
                (
                    "model.layers.0.mlp.experts.gate_proj.biases",
                    "BF16",
                    [4, 2],
                    struct.pack(
                        "<8H",
                        0xBF80,
                        0x8000,
                        0x0000,
                        0x3F80,
                        0xC001,
                        0xBEAA,
                        0x3D55,
                        0xBF01,
                    ),
                ),
            ],
            self.second_shard: [
                (
                    "model.ple.embed_tokens.weight",
                    "U32",
                    [2, 4],
                    struct.pack(
                        "<8I",
                        0xFEDCBA98,
                        0x76543210,
                        0x11111111,
                        0xEEEEEEEE,
                        0x0F0F0F0F,
                        0xF0F0F0F0,
                        0x80000001,
                        0x12345678,
                    ),
                ),
                (
                    "model.ple.embed_tokens.scales",
                    "BF16",
                    [2, 1],
                    struct.pack("<2H", 0x3C07, 0x3F81),
                ),
                (
                    "model.ple.embed_tokens.biases",
                    "BF16",
                    [2, 1],
                    struct.pack("<2H", 0xBF81, 0x8000),
                ),
                (
                    "model.ple.indices",
                    "I64",
                    [3],
                    struct.pack("<3q", 7, 1 << 40, (1 << 62) - 1),
                ),
            ],
        }
        for name, tensors in self.shards.items():
            write_shard(self.source / name, tensors)
        self.tensors = {
            name: {"shard": shard, "dtype": dtype, "shape": shape, "data": data}
            for shard, tensors in self.shards.items()
            for name, dtype, shape, data in tensors
        }
        self.index = {
            "metadata": {
                "total_size": sum(
                    len(tensor["data"]) for tensor in self.tensors.values()
                )
            },
            "weight_map": {
                name: tensor["shard"] for name, tensor in self.tensors.items()
            },
        }
        self.write_index()
        self.config = {
            "model_type": "qwen4_exp",
            "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        }
        self.write_config()
        self.small_files = {
            "tokenizer.json": b'{ "version" : "1.0", "model" : {} }\n',
            "tokenizer_config.json": b'{"chat_template": "synthetic-template"}\n',
            "generation_config.json": b'{ "do_sample": false }\n',
            "vocab.json": b'{"tiny": 7}\n',
            "merges.txt": b"#version: 0.2\nsynthetic pair\n",
            "chat_template.jinja": b"{% for message in messages %}{{ message }}{% endfor %}\n",
        }
        for name, data in self.small_files.items():
            (self.source / name).write_bytes(data)
        self.source_snapshot = {
            path.name: path.read_bytes() for path in self.source.iterdir()
        }
        self.source_stat_snapshot = {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in self.source.iterdir()
        }

    def write_index(self):
        (self.source / "model.safetensors.index.json").write_text(
            json.dumps(self.index, indent=3) + "\n", encoding="utf-8"
        )

    def write_config(self):
        (self.source / "config.json").write_text(
            json.dumps(self.config, indent=3) + "\n", encoding="utf-8"
        )

    def reset_source(self):
        for path in self.source.iterdir():
            path.unlink()
        for name, data in self.source_snapshot.items():
            (self.source / name).write_bytes(data)

    def assert_no_publication(self):
        self.assertFalse(self.destination.exists())
        self.assertEqual({path.name for path in self.root.iterdir()}, {"source"})

    def test_raw_round_trip_alignment_and_identities(self):
        manifest = importer.import_bundle(self.source, self.destination, chunk_bytes=19)
        self.assertEqual(manifest["alignment"], ALIGNMENT)
        self.assertEqual(manifest["schema"], "splash-local-qwen4-affine-v1")
        self.assertEqual(set(manifest["tensors"]), set(self.tensors))
        self.assertEqual(len(manifest["shards"]), 2)
        self.assertEqual(manifest["quantization"], self.config["quantization"])
        by_path = {record["path"]: record for record in manifest["shards"]}
        for record in manifest["shards"]:
            payload = (self.destination / record["path"]).read_bytes()
            original_shard = (self.source / record["source_path"]).read_bytes()
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(record["sha256"], sha256(payload))
            self.assertEqual(record["source_bytes"], len(original_shard))
            self.assertEqual(record["source_sha256"], sha256(original_shard))
            self.assertEqual(len(payload) % ALIGNMENT, 0)
            records = sorted(
                (
                    tensor
                    for tensor in manifest["tensors"].values()
                    if tensor["shard"] == record["path"]
                ),
                key=lambda tensor: tensor["offset"],
            )
            cursor = 0
            for tensor in records:
                self.assertEqual(tensor["offset"], cursor)
                end = cursor + tensor["length"]
                cursor = (end + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
                self.assertEqual(payload[end:cursor], b"\0" * (cursor - end))
            self.assertEqual(cursor, len(payload))
        for name, expected in self.tensors.items():
            tensor = manifest["tensors"][name]
            self.assertEqual(tensor["dtype"], expected["dtype"])
            self.assertEqual(tensor["shape"], expected["shape"])
            self.assertEqual(tensor["source_shard"], expected["shard"])
            self.assertEqual(tensor["length"], len(expected["data"]))
            source_bytes = self.source_snapshot[expected["shard"]]
            header_bytes = struct.unpack_from("<Q", source_bytes)[0]
            source_header = json.loads(source_bytes[8 : 8 + header_bytes])
            self.assertEqual(
                tensor["source_offset"],
                8 + header_bytes + source_header[name]["data_offsets"][0],
            )
            self.assertEqual(by_path[tensor["shard"]]["source_path"], expected["shard"])
            payload = (self.destination / tensor["shard"]).read_bytes()
            self.assertEqual(
                payload[tensor["offset"] : tensor["offset"] + tensor["length"]],
                expected["data"],
            )
        for name, data in self.source_snapshot.items():
            if not name.endswith(".safetensors"):
                self.assertEqual((self.destination / name).read_bytes(), data)
        for record in manifest["small_files"]:
            data = (self.source / record["path"]).read_bytes()
            self.assertEqual(record["bytes"], len(data))
            self.assertEqual(record["sha256"], sha256(data))
        source_identity = {
            "schema": "splash-local-qwen4-affine-v1",
            "source_files": [
                {"path": name, "bytes": len(data), "sha256": sha256(data)}
                for name, data in sorted(self.source_snapshot.items())
            ],
        }
        self.assertEqual(
            manifest["source_identity_sha256"],
            sha256(
                json.dumps(
                    source_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ),
        )
        self.assertEqual(
            importer.verify_bundle(
                self.destination, source=self.source, chunk_bytes=19
            ),
            manifest,
        )
        self.assertEqual(
            self.source_snapshot,
            {path.name: path.read_bytes() for path in self.source.iterdir()},
        )
        self.assertEqual(
            self.source_stat_snapshot,
            {
                path.name: (path.stat().st_size, path.stat().st_mtime_ns)
                for path in self.source.iterdir()
            },
        )
        # PLE words and int64 values remain opaque raw tensors, including size.
        self.assertEqual(
            manifest["tensors"]["model.ple.embed_tokens.weight"]["length"], 32
        )
        self.assertEqual(manifest["tensors"]["model.ple.indices"]["dtype"], "I64")

    def test_verified_existing_bundle_is_reused_without_rewriting(self):
        first = importer.import_bundle(self.source, self.destination)
        snapshot = {
            path.relative_to(self.destination): (
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )
            for path in self.destination.rglob("*")
            if path.is_file()
        }
        with mock.patch.object(
            importer, "_copy_shard", side_effect=AssertionError("copied again")
        ):
            second = importer.import_bundle(
                self.source, self.destination, chunk_bytes=19
            )
        self.assertEqual(first, second)
        self.assertEqual(
            snapshot,
            {
                path.relative_to(self.destination): (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
                for path in self.destination.rglob("*")
                if path.is_file()
            },
        )

    def test_existing_missing_or_tampered_payload_is_refused_without_repair(self):
        for mutation in ("missing", "tampered"):
            with self.subTest(mutation=mutation):
                destination = self.root / mutation
                manifest = importer.import_bundle(self.source, destination)
                payload = destination / manifest["shards"][0]["path"]
                if mutation == "missing":
                    payload.unlink()
                else:
                    damaged = bytearray(payload.read_bytes())
                    damaged[0] ^= 1
                    payload.write_bytes(damaged)
                snapshot = {
                    path.relative_to(destination): path.read_bytes()
                    for path in destination.rglob("*")
                    if path.is_file()
                }
                with self.assertRaises(importer.BundleError):
                    importer.verify_bundle(
                        destination, source=self.source, chunk_bytes=19
                    )
                with self.assertRaises(importer.BundleError):
                    importer.import_bundle(self.source, destination, chunk_bytes=19)
                self.assertEqual(
                    snapshot,
                    {
                        path.relative_to(destination): path.read_bytes()
                        for path in destination.rglob("*")
                        if path.is_file()
                    },
                )

    def test_missing_copied_config_is_not_considered_a_complete_bundle(self):
        importer.import_bundle(self.source, self.destination)
        (self.destination / "config.json").unlink()
        with self.assertRaises(importer.BundleError):
            importer.verify_bundle(self.destination, source=self.source)
        with self.assertRaises(importer.BundleError):
            importer.import_bundle(self.source, self.destination)
        self.assertFalse((self.destination / "config.json").exists())

    def test_existing_bundle_from_changed_source_is_refused(self):
        importer.import_bundle(self.source, self.destination)
        path = self.source / "tokenizer_config.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(importer.BundleError):
            importer.import_bundle(self.source, self.destination)

    def test_destination_is_absent_until_all_shards_finish(self):
        original_copy = importer._copy_shard
        observed = []

        def checked_copy(*args, **kwargs):
            self.assertFalse(self.destination.exists())
            result = original_copy(*args, **kwargs)
            self.assertFalse(self.destination.exists())
            observed.append(args[0])
            return result

        with mock.patch.object(importer, "_copy_shard", side_effect=checked_copy):
            manifest = importer.import_bundle(
                self.source, self.destination, chunk_bytes=19
            )
        self.assertEqual(len(observed), 2)
        self.assertEqual(
            importer.verify_bundle(self.destination, source=self.source), manifest
        )

    def test_source_modification_during_copy_rejects_atomic_publication(self):
        original_copy = importer._copy_shard
        modified = False

        def changed_copy(*args, **kwargs):
            nonlocal modified
            result = original_copy(*args, **kwargs)
            if not modified:
                modified = True
                path = Path(args[0])
                damaged = bytearray(path.read_bytes())
                damaged[-1] ^= 1
                path.write_bytes(damaged)
            return result

        with mock.patch.object(importer, "_copy_shard", side_effect=changed_copy):
            with self.assertRaises(importer.BundleError):
                importer.import_bundle(self.source, self.destination, chunk_bytes=19)
        self.assertTrue(modified)
        self.assert_no_publication()

    def test_injected_copy_failure_leaves_no_partial_bundle(self):
        with mock.patch.object(
            importer, "_copy_shard", side_effect=OSError("synthetic failure")
        ):
            with self.assertRaises((importer.BundleError, OSError)):
                importer.import_bundle(self.source, self.destination, chunk_bytes=19)
        self.assert_no_publication()

    def test_payload_and_header_reads_obey_tiny_chunk_bound(self):
        original_open = Path.open
        observed = []

        class BoundedReader:
            def __init__(reader_self, stream):
                reader_self.stream = stream

            def read(reader_self, size=-1):
                if not 0 < size <= 19:
                    raise AssertionError(f"unbounded tensor stream read: {size}")
                observed.append(size)
                return reader_self.stream.read(size)

            def __getattr__(reader_self, name):
                return getattr(reader_self.stream, name)

            def __enter__(reader_self):
                reader_self.stream.__enter__()
                return reader_self

            def __exit__(reader_self, *args):
                return reader_self.stream.__exit__(*args)

        def bounded_open(path, mode="r", *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            if mode == "rb" and (
                path.suffix == ".safetensors" or "weights" in path.parts
            ):
                return BoundedReader(stream)
            return stream

        with mock.patch.object(Path, "open", bounded_open):
            manifest = importer.import_bundle(
                self.source, self.destination, chunk_bytes=19
            )
            self.assertEqual(
                importer.verify_bundle(
                    self.destination, source=self.source, chunk_bytes=19
                ),
                manifest,
            )
        self.assertIn(19, observed)
        self.assertTrue(all(0 < size <= 19 for size in observed))

    def test_corrupt_manifest_or_unknown_schema_is_not_reused(self):
        for mutation in ("checksum", "schema"):
            with self.subTest(mutation=mutation):
                destination = self.root / mutation
                importer.import_bundle(self.source, destination)
                path = destination / "manifest.json"
                if mutation == "checksum":
                    path.write_bytes(path.read_bytes() + b" ")
                else:
                    manifest = json.loads(path.read_bytes())
                    manifest["schema"] = "future-unknown-schema"
                    raw = json.dumps(manifest).encode("utf-8")
                    path.write_bytes(raw)
                    (destination / "manifest.sha256").write_text(sha256(raw) + "\n")
                snapshot = {
                    file.relative_to(destination): file.read_bytes()
                    for file in destination.rglob("*")
                    if file.is_file()
                }
                with self.assertRaises(importer.BundleError):
                    importer.verify_bundle(destination, source=self.source)
                with self.assertRaises(importer.BundleError):
                    importer.import_bundle(self.source, destination)
                self.assertEqual(
                    snapshot,
                    {
                        file.relative_to(destination): file.read_bytes()
                        for file in destination.rglob("*")
                        if file.is_file()
                    },
                )

    def test_race_created_empty_destination_is_never_replaced(self):
        original_publish = importer._publish_exclusive

        def race_publish(source, destination):
            destination.mkdir()
            original_publish(source, destination)

        with mock.patch.object(
            importer, "_publish_exclusive", side_effect=race_publish
        ):
            with self.assertRaises(OSError):
                importer.import_bundle(self.source, self.destination, chunk_bytes=19)
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertEqual(
            {path.name for path in self.root.iterdir()}, {"source", "bundle"}
        )

    def test_mixed_module_override_is_preserved(self):
        prefix = "model.layers.0.mlp.experts.gate_proj"
        override = {"bits": 8, "group_size": 32, "mode": "affine"}
        self.config["quantization"][prefix] = override
        self.write_config()
        tensors = [
            self.shards[self.first_shard][0],
            (
                prefix + ".scales",
                "BF16",
                [4, 1],
                struct.pack("<4H", 0x3F80, 0x3C07, 0x4001, 0x3D55),
            ),
            (
                prefix + ".biases",
                "BF16",
                [4, 1],
                struct.pack("<4H", 0xBF80, 0x8000, 0x0000, 0x3F80),
            ),
        ]
        write_shard(self.source / self.first_shard, tensors)
        self.index["metadata"]["total_size"] = sum(
            len(tensor[3]) for tensor in tensors + self.shards[self.second_shard]
        )
        self.write_index()
        manifest = importer.import_bundle(self.source, self.destination, chunk_bytes=19)
        self.assertEqual(manifest["quantization"], self.config["quantization"])
        self.assertEqual(manifest["tensors"][prefix + ".scales"]["shape"], [4, 1])
        self.assertEqual(
            importer.verify_bundle(
                self.destination, source=self.source, chunk_bytes=19
            ),
            manifest,
        )

    def test_malformed_headers_are_rejected_before_publication(self):
        weight = "model.layers.0.mlp.experts.gate_proj.weight"
        scales = "model.layers.0.mlp.experts.gate_proj.scales"
        cases = {
            "short_prefix": lambda path: path.write_bytes(b"1234"),
            "oversized_header": lambda path: path.write_bytes(
                struct.pack("<Q", 1 << 40)
            ),
            "bad_json": lambda path: path.write_bytes(struct.pack("<Q", 2) + b"{]"),
            "negative_shape": lambda path: rewrite_header(
                path, lambda h: h[weight].update(shape=[-1, 8])
            ),
            "boolean_shape": lambda path: rewrite_header(
                path, lambda h: h[weight].update(shape=[True, 8])
            ),
            "dtype_bytes": lambda path: rewrite_header(
                path, lambda h: h[weight].update(shape=[4, 9])
            ),
            "unknown_dtype": lambda path: rewrite_header(
                path, lambda h: h[weight].update(dtype="F99")
            ),
            "negative_offset": lambda path: rewrite_header(
                path, lambda h: h[weight].update(data_offsets=[-1, 127])
            ),
            "reversed_offset": lambda path: rewrite_header(
                path, lambda h: h[weight].update(data_offsets=[128, 0])
            ),
            "out_of_bounds": lambda path: rewrite_header(
                path, lambda h: h[weight].update(data_offsets=[1024, 1152])
            ),
            "overlap": lambda path: rewrite_header(
                path, lambda h: h[scales].update(data_offsets=[0, 16])
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(case=label):
                self.reset_source()
                mutate(self.source / self.first_shard)
                with self.assertRaises(importer.BundleError):
                    importer.import_bundle(
                        self.source, self.destination, chunk_bytes=19
                    )
                self.assert_no_publication()

    def test_duplicate_tensor_header_keys_are_rejected(self):
        descriptor = json.dumps({"dtype": "U32", "shape": [1], "data_offsets": [0, 4]})
        name = json.dumps("model.layers.0.mlp.experts.gate_proj.weight")
        duplicate = (
            "{" + name + ":" + descriptor + "," + name + ":" + descriptor + "}"
        ).encode()
        write_shard(
            self.source / self.first_shard,
            [],
            raw_header=duplicate,
            raw_payload=struct.pack("<I", 0xFEDCBA98),
        )
        with self.assertRaises(importer.BundleError):
            importer.import_bundle(self.source, self.destination)
        self.assert_no_publication()

    def test_duplicate_tensor_across_shards_is_rejected(self):
        second = list(self.shards[self.second_shard])
        second.append(self.shards[self.first_shard][0])
        write_shard(self.source / self.second_shard, second)
        with self.assertRaises(importer.BundleError):
            importer.import_bundle(self.source, self.destination)
        self.assert_no_publication()

    def test_index_mismatches_and_unsafe_shard_names_are_rejected(self):
        weight = "model.layers.0.mlp.experts.gate_proj.weight"
        for mapping in (
            None,
            self.second_shard,
            "missing.safetensors",
            "../outside.safetensors",
            "/outside.safetensors",
        ):
            with self.subTest(mapping=mapping):
                self.reset_source()
                index = copy.deepcopy(self.index)
                if mapping is None:
                    index["weight_map"].pop(weight)
                else:
                    index["weight_map"][weight] = mapping
                (self.source / "model.safetensors.index.json").write_text(
                    json.dumps(index)
                )
                with self.assertRaises(importer.BundleError):
                    importer.import_bundle(self.source, self.destination)
                self.assert_no_publication()

    def test_invalid_quantization_is_rejected_independently_of_tensor_headers(self):
        for field, value in (
            ("bits", 0),
            ("bits", True),
            ("group_size", 0),
            ("group_size", True),
            ("mode", "unknown-mode"),
        ):
            with self.subTest(field=field, value=value):
                self.reset_source()
                config = copy.deepcopy(self.config)
                config["quantization"][field] = value
                (self.source / "config.json").write_text(json.dumps(config))
                with self.assertRaises(importer.BundleError):
                    importer.import_bundle(self.source, self.destination)
                self.assert_no_publication()


if __name__ == "__main__":
    unittest.main()

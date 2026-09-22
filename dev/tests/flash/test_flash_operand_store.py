import copy
import hashlib
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from dev.tests.flash import flash_operand_store_fixtures as fixture


class FlashOperandStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "store"
        self.manifest = fixture.create_fixture(self.root)

    def assert_native(
        self,
        valid,
        *,
        source=fixture.SOURCE_IDENTITY,
        fingerprint=fixture.WEIGHTS_FINGERPRINT,
        verify=True,
    ):
        binary = os.environ.get("FLASH_OPERAND_STORE_NATIVE_CHECKER")
        if not binary:
            return
        command = [binary, "--check", str(self.root), source, fingerprint]
        if verify:
            command.append("--verify-payloads")
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertEqual(
            result.returncode == 0,
            valid,
            " ".join(command) + "\n" + result.stdout + result.stderr,
        )

    def assert_rejected(self, *, manifest=None, **expected):
        if manifest is not None:
            fixture.write_manifest(self.root, manifest)
        with self.assertRaises(fixture.StoreError):
            fixture.verify_store(self.root, **expected)
        self.assert_native(
            False,
            source=expected.get("source_identity", fixture.SOURCE_IDENTITY),
            fingerprint=expected.get(
                "weights_fingerprint", fixture.WEIGHTS_FINGERPRINT
            ),
            verify=expected.get("verify_payloads", True),
        )

    def mutated(self, path, value):
        manifest = copy.deepcopy(self.manifest)
        target = manifest
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return manifest

    def test_bf16_and_f32_fixture_is_byte_verified_and_source_immutable(self):
        original = {path.name: path.read_bytes() for path in self.root.iterdir()}
        self.assertEqual(fixture.verify_store(self.root), self.manifest)
        self.assert_native(True)
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.root.iterdir()}, original
        )
        bf = (self.root / "bf16.bin").read_bytes()
        f32 = (self.root / "f32.bin").read_bytes()
        self.assertEqual(len(bf), fixture.ALIGNMENT)
        self.assertFalse(any(bf[64 * 64 * 2 :]))
        for index in range(64 * 64):
            value = struct.unpack_from("<f", f32, index * 4)[0]
            self.assertEqual(bf[index * 2 : index * 2 + 2], fixture._bf16(value))

    def test_fixture_exercises_signed_scales_and_bf16_midpoint_parities(self):
        bf = (self.root / "bf16.bin").read_bytes()
        f32 = (self.root / "f32.bin").read_bytes()
        for index, full, rounded in (
            (11, 1.00390625, 0x3F80),
            (64 + 4, 1.01171875, 0x3F82),
            (128 + 13, -1.00390625, 0xBF80),
        ):
            with self.subTest(index=index):
                self.assertEqual(struct.unpack_from("<f", f32, index * 4)[0], full)
                self.assertEqual(struct.unpack_from("<H", bf, index * 2)[0], rounded)

    def test_math_digest_matches_native_header_constants(self):
        repo = Path(__file__).resolve().parents[3]
        for filename, value in (
            ("FlashDenseCache.hpp", fixture.MATH_FORMATS["BF16"]),
            ("FlashFloatDenseCache.hpp", fixture.MATH_FORMATS["F32"]),
            ("FlashOperandStore.hpp", fixture.MATH_VERSION),
        ):
            self.assertIn(
                '"' + value + '"', (repo / "runtime/flash" / filename).read_text()
            )

    def test_schema_math_and_source_identity_cannot_be_reused(self):
        for path, value in (
            (("schema",), "splash-local-affine-operands-v2"),
            (("alignment_bytes",), 4096),
            (("math_version_sha256",), "3" * 64),
            (("source_identity_sha256",), "3" * 64),
            (("weights_manifest_fingerprint",), "3" * 64),
            (("entries", 0, "format"), "INT8"),
            (("entries", 0, "operand_math"), fixture.MATH_FORMATS["F32"]),
        ):
            with self.subTest(path=path):
                self.assert_rejected(manifest=self.mutated(path, value))

    def test_expected_source_arguments_are_strict_and_bound(self):
        for field in ("source_identity", "weights_fingerprint"):
            for value in ("3" * 64, "A" * 64, "1" * 63, "1" * 65, "", "g" * 64):
                with self.subTest(field=field, value=value):
                    fixture.write_manifest(self.root, self.manifest)
                    self.assert_rejected(**{field: value})

    def test_digest_fields_require_lowercase_exact_sha256(self):
        paths = (
            ("source_identity_sha256",),
            ("weights_manifest_fingerprint",),
            ("math_version_sha256",),
            ("entries", 0, "payload_sha256"),
        )
        for path in paths:
            for value in ("A" * 64, "0" * 63, "0" * 65, "g" * 64, True, None):
                with self.subTest(path=path, value=value):
                    self.assert_rejected(manifest=self.mutated(path, value))

    def test_integer_metadata_never_coerces_booleans_negative_or_fractional(self):
        paths = [("alignment_bytes",)]
        paths += [
            ("entries", 0, name)
            for name in ("logical_bytes", "allocated_bytes", "offset_bytes")
        ]
        paths += [("entries", 0, "shape", axis) for axis in (0, 1)]
        paths += [("entries", 0, "source", name) for name in fixture.SOURCE_KEYS]
        for path in paths:
            for value in (True, False, -1, 1.0, "1", None, 1 << 64):
                with self.subTest(path=path, value=value):
                    self.assert_rejected(manifest=self.mutated(path, value))

    def test_exact_keys_and_bounded_nonempty_inventory(self):
        for path in ((), ("entries", 0), ("entries", 0, "source")):
            for mode in ("missing", "extra"):
                manifest = copy.deepcopy(self.manifest)
                target = manifest
                for key in path:
                    target = target[key]
                if mode == "missing":
                    del target[next(iter(target))]
                else:
                    target["unexpected"] = "field"
                with self.subTest(path=path, mode=mode):
                    self.assert_rejected(manifest=manifest)
        for entries in ([], {}, None, [None], [None] * 8193):
            with self.subTest(entries_type=type(entries).__name__):
                self.assert_rejected(manifest=self.mutated(("entries",), entries))

    def test_duplicate_operand_and_payload_file_are_rejected(self):
        duplicate = copy.deepcopy(self.manifest)
        duplicate["entries"].append(copy.deepcopy(duplicate["entries"][0]))
        self.assert_rejected(manifest=duplicate)
        # A different name/format still cannot alias another payload.
        aliased = copy.deepcopy(self.manifest)
        aliased["entries"][1]["file"] = "bf16.bin"
        self.assert_rejected(manifest=aliased)

    def test_duplicate_json_keys_rejected_even_when_manifest_checksum_matches(self):
        for raw in (
            fixture.canonical(self.manifest).replace(
                b'"alignment_bytes":16384',
                b'"alignment_bytes":1,"alignment_bytes":16384',
                1,
            ),
            fixture.canonical(self.manifest).replace(
                b'"experts":1', b'"experts":2,"experts":1', 1
            ),
            fixture.canonical(self.manifest).replace(
                b'"experts":1', b'"experts":2,"\\u0065xperts":1', 1
            ),
        ):
            with self.subTest(raw=raw[:80]):
                fixture.write_manifest(self.root, self.manifest, raw=raw)
                self.assert_rejected()

    def test_invalid_utf8_and_nonfinite_json_constants_are_rejected(self):
        for raw in (
            b"\xff",
            b"[]",
            b"null",
            b"{",
            fixture.canonical(self.manifest).replace(
                b'"experts":1', b'"experts":NaN', 1
            ),
            fixture.canonical(self.manifest).replace(
                b'"offset_bytes":0', b'"offset_bytes":-0', 1
            ),
            fixture.canonical(self.manifest).replace(
                b'"experts":1', b'"experts":1e0', 1
            ),
        ):
            with self.subTest(raw=raw[:80]):
                fixture.write_manifest(self.root, self.manifest, raw=raw)
                self.assert_rejected()

    def test_missing_and_nonregular_payloads_are_rejected(self):
        path = self.root / "bf16.bin"
        path.unlink()
        self.assert_rejected()
        path.mkdir()
        self.assert_rejected()

    def test_source_geometry_quantization_and_strides_are_consistent(self):
        source_mutations = {
            "experts": (0, 2),
            "output_size": (63, 128),
            "input_size": (32, 128),
            "bits": (1, 3, 7, 16),
            "group_size": (1, 16, 256),
            "weight_row_stride_bytes": (0, 28, 33),
            "weight_expert_stride_bytes": (1, 2047),
            "parameter_row_stride_bytes": (0, 2, 5),
            "parameter_expert_stride_bytes": (1, 255),
        }
        for name, values in source_mutations.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    self.assert_rejected(
                        manifest=self.mutated(("entries", 0, "source", name), value)
                    )
        for shape in (
            [],
            [64],
            [64, 64, 1],
            [0, 64],
            [63, 64],
            [64, 33],
            [64, 32769],
            [64, 1 << 32],
            "64,64",
        ):
            with self.subTest(shape=shape):
                self.assert_rejected(
                    manifest=self.mutated(("entries", 0, "shape"), shape)
                )

    def test_projection_and_file_strings_are_strict(self):
        for value in (
            "",
            "a\0b",
            "a" * 1025,
            "fixture.embed_tokens",
            "fixture.ngram_embedding",
            True,
            None,
            3,
        ):
            with self.subTest(projection=value):
                self.assert_rejected(
                    manifest=self.mutated(("entries", 0, "projection"), value)
                )
        for name in (
            "../bf16.bin",
            "sub/bf16.bin",
            "./bf16.bin",
            "/tmp/bf16.bin",
            "bf16.bin/../bf16.bin",
            "bf16.BIN",
            "bf16",
            "f ü.bin",
            "bf16\0.bin",
            ".bf16.bin",
            "a..bin",
            "a" * 125 + ".bin",
            "",
            True,
            None,
        ):
            with self.subTest(file=name):
                self.assert_rejected(
                    manifest=self.mutated(("entries", 0, "file"), name)
                )

    def test_symlink_payload_manifest_checksum_and_root_are_rejected(self):
        for name in ("bf16.bin", "manifest.json", "manifest.sha256"):
            original = self.root / name
            moved = Path(self.temporary.name) / name
            original.rename(moved)
            original.symlink_to(moved)
            with self.subTest(name=name):
                self.assert_rejected()
            original.unlink()
            moved.rename(original)
        linked = Path(self.temporary.name) / "linked"
        linked.symlink_to(self.root, target_is_directory=True)
        original_root = self.root
        self.root = linked
        self.assert_rejected()
        self.root = original_root

    def test_manifest_checksum_covers_exact_bytes_and_newline(self):
        original = (self.root / "manifest.sha256").read_bytes()
        for checksum in (
            original[:-1],
            original + b"\n",
            b"0" * 64 + b"\n",
            original.upper(),
        ):
            with self.subTest(checksum=checksum[:16]):
                (self.root / "manifest.sha256").write_bytes(checksum)
                self.assert_rejected()
        (self.root / "manifest.sha256").write_bytes(original)
        (self.root / "manifest.json").write_bytes(
            (self.root / "manifest.json").read_bytes() + b" "
        )
        self.assert_rejected()

    def test_payload_hash_truncation_growth_and_padding_corruption(self):
        path = self.root / "bf16.bin"
        original = path.read_bytes()
        corrupted = bytearray(original)
        corrupted[0] ^= 1
        for payload in (original[:-1], original + b"\0", bytes(corrupted)):
            with self.subTest(size=len(payload)):
                path.write_bytes(payload)
                fixture.write_manifest(self.root, self.manifest)
                self.assert_rejected()
        # Re-signing an invalid layout must not legitimize nonzero padding.
        corrupted = bytearray(original)
        corrupted[-1] = 1
        path.write_bytes(corrupted)
        manifest = copy.deepcopy(self.manifest)
        manifest["entries"][0]["payload_sha256"] = hashlib.sha256(corrupted).hexdigest()
        self.assert_rejected(manifest=manifest)

    def test_offset_logical_allocation_and_file_extent_cannot_overlap(self):
        for name, values in {
            "offset_bytes": (1, 16384),
            "logical_bytes": (0, 8191, 8193),
            "allocated_bytes": (0, 8192, 16383, 32768),
        }.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    self.assert_rejected(
                        manifest=self.mutated(("entries", 0, name), value)
                    )

    def test_metadata_only_check_does_not_claim_payload_integrity(self):
        path = self.root / "bf16.bin"
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)
        self.assertEqual(
            fixture.verify_store(self.root, verify_payloads=False), self.manifest
        )
        self.assert_native(True, verify=False)
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()

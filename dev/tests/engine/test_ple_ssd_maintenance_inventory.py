"""Source checks for exact reflected owner counts and no SSD table binding."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class PLESSDMaintenanceInventoryTests(unittest.TestCase):
    def test_both_shader_entry_points_have_exact_pointer_and_read_counts(self):
        shader = (
            ROOT / "runtime/metal/kernels/shared/flash_idle_residency_maintenance.metal"
        ).read_text()
        for structure, kernel, count in (
            ("ImmutableOwners1134", "flash_idle_immutable_touch_v1", 1134),
            (
                "ImmutableNonPLEOwners1141",
                "flash_idle_immutable_touch_ple_ssd_v1",
                1141,
            ),
        ):
            with self.subTest(kernel=kernel):
                self.assertRegex(
                    shader,
                    rf"struct {structure} \{{ array<device const uint \*, {count}> p \[\[id\(0\)\]\]; \}};",
                )
                body = shader.split(f"kernel void {kernel}(", 1)[1].split("\n}", 1)[0]
                self.assertIn(f"constant {structure} &owners [[buffer(0)]]", body)
                self.assertIn(f"index < {count}; index += 256", body)
                self.assertIn("owners.p[index][0]", body)
                self.assertIn(f"output[2] = {count}", body)
                self.assertNotIn("nullptr", body)
                self.assertNotIn("SSDStore", body)

    def test_worker_chooses_explicit_storage_mode_and_only_immutable_owners(self):
        worker = (ROOT / "runtime/flash/FlashWorker.mm").read_text()
        body = worker.split("if (idleMaintenanceRequested) {", 1)[1].split(
            "const uint64_t instance", 1
        )[0]
        self.assertIn("weights.pleSSDStreamingEnabled()", body)
        self.assertIn("StorageMode::PLESSD : idle_maintenance::StorageMode::Raw", body)
        self.assertIn("weights.immutableWeightBuffers()", body)
        self.assertIn("forward.cachedOperandsOnly()", body)
        self.assertIn("head->cachedOperandsOnly()", body)
        self.assertIn("hiddenSize, maintenanceStorage", body)
        self.assertIn("plannedBytes(backend, maintenanceStorage)", body)
        self.assertNotRegex(
            body,
            re.compile(
                r"pleSSDStore|diskProjection|lookupRows|packedRows|staging|cacheBudget"
            ),
        )

    def test_backend_reflection_rejects_missing_or_padded_pointer_lists(self):
        backend = (ROOT / "runtime/metal/MetalBackend.mm").read_text()
        body = backend.split(
            "MetalBuffer MetalBackend::makeReadOnlyArgumentBuffer(", 1
        )[1].split("auto result = allocateBuffer", 1)[0]
        self.assertIn("resources.size() != pointerLayout.size()", body)
        self.assertIn("argument-buffer resource is empty", body)
        self.assertIn("argument-buffer indices must be unique and dense", body)


if __name__ == "__main__":
    unittest.main()

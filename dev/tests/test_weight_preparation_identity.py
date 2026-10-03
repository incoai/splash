import re
import shutil
import tempfile
import unittest
from pathlib import Path

from dev.tools import build_identity
from dev.tools.weight_preparation_identity import INPUTS, header, identity

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = re.compile(r'^\s*#\s*(?:include|import)\s+"([^"]+)"', re.MULTILINE)
GENERATED = {"WeightPreparationIdentity.hpp"}
# Project headers the preparation inputs include that define no prepared
# byte. Keys record every value these declarations carry into a plan;
# golden hashes of prepared fixtures cover the shared I/O.
REVIEWED = {
    "runtime/model/PreparedWeights.hpp": "the store, staging bound and bounded I/O",
    "runtime/model/GgufFile.hpp": "keys record each tensor's bytes and type",
    "runtime/model/GgufImage.hpp": "the GGUF plan, which keys record whole",
    "runtime/model/AffinePreparation.hpp": "the affine plan, which keys record whole",
    "runtime/model/GgufPreparation.hpp": "the executor's declarations",
    "runtime/model/VisionPreparation.hpp": "the vision plan, which keys record whole",
    "runtime/model/WeightStore.hpp": "the reader; preparation uses its error type",
    "runtime/metal/MetalBackend.hpp": "the backend the repack is dispatched through",
    "runtime/metal/DeviceCapabilities.hpp": "device queries of the backend",
    "runtime/metal/CommandGraph.hpp": "dispatch recording of the backend",
    "runtime/metal/abi/ExecutionGeometry.h": "the batch and prefill geometry",
    "runtime/ops/Weights.hpp": "inference weight views",
    "runtime/ops/Linear.hpp": "inference projections",
    "runtime/ops/Normalization.hpp": "inference norms",
}


def identities(root):
    """Each preparation's generated identity, by name."""
    return {
        define.removeprefix("SPLASH_").removesuffix("_PREPARATION_ID"): value.strip('"')
        for _, define, value in (
            line.split() for line in header(root).decode().splitlines()[2:]
        )
    }


def included(name):
    """The project files a source includes, transitively, itself included."""
    seen, pending = set(), [name]
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        source = ROOT / path
        for include in INCLUDE.findall(source.read_text()):
            if include in GENERATED:
                continue
            for base in (source.parent, ROOT / "runtime"):
                if (base / include).is_file():
                    pending.append(
                        (base / include).resolve().relative_to(ROOT).as_posix()
                    )
                    break
            else:
                raise AssertionError(
                    f"{path} includes {include}, which is not in the tree"
                )
    return seen


class PreparationIdentityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for name in {
            *(path for paths in INPUTS.values() for path in paths),
            "runtime/metal/abi/Gguf.h",
            "runtime/metal/kernels/shared/gguf_linear.metal",
            "runtime/model/WeightStore.hpp",
            "runtime/model/GgufImage.cpp",
            "runtime/model/SafetensorsCheckpoint.mm",
            "runtime/model/VisionLoader.cpp",
            "runtime/model/AffineTarget.cpp",
        }:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, self.root / name)

    def tearDown(self):
        self.directory.cleanup()

    def changed_by(self, name, text="\n// changed\n"):
        """The preparations whose identity changes when text is appended to name."""
        original = identities(self.root)
        path = self.root / name
        previous = path.read_bytes()
        path.write_bytes(previous + text.encode())
        changed = identities(self.root)
        path.write_bytes(previous)
        return {kind for kind in original if changed[kind] != original[kind]}

    def test_inference_code_parsers_and_planners_keep_every_identity(self):
        for name in (
            "runtime/metal/abi/Gguf.h",
            "runtime/metal/kernels/shared/gguf_linear.metal",
            "runtime/model/WeightStore.hpp",
            "runtime/model/GgufImage.cpp",
            "runtime/model/SafetensorsCheckpoint.mm",
            "runtime/model/VisionLoader.cpp",
            "runtime/model/AffineTarget.cpp",
        ):
            with self.subTest(unrelated=name):
                self.assertEqual(self.changed_by(name), set())

    def test_preparation_code_changes_the_identities_it_defines(self):
        for name in {path for paths in INPUTS.values() for path in paths}:
            with self.subTest(preparation=name):
                kinds = {kind for kind, paths in INPUTS.items() if name in paths}
                self.assertEqual(self.changed_by(name), kinds)

    def test_names_do_not_enter_the_identity(self):
        original = identities(self.root)
        for kind, paths in INPUTS.items():
            with self.subTest(kind=kind):
                renamed = []
                for index, name in enumerate(reversed(paths)):
                    path = self.root / "renamed" / kind / f"{index}.src"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.root / name, path)
                    renamed.append(path.relative_to(self.root).as_posix())
                self.assertEqual(identity(self.root, renamed), original[kind])

    def test_identity_is_not_a_build_identity(self):
        for kind, paths in INPUTS.items():
            with self.subTest(kind=kind):
                self.assertNotEqual(
                    identities(self.root)[kind],
                    build_identity.source_digest(self.root, paths),
                )

    def test_every_input_says_which_models_editing_it_prepares_again(self):
        names = {"AFFINE": "affine", "GGUF": "GGUF", "VISION": "vision"}
        for name in {path for paths in INPUTS.values() for path in paths}:
            with self.subTest(input=name):
                kinds = " and ".join(
                    names[kind] for kind, paths in INPUTS.items() if name in paths
                )
                text = (ROOT / name).read_text()
                self.assertEqual(text.count("re-prepares every"), 1)
                self.assertIn(
                    f"Editing this file re-prepares every {kinds} model.", text
                )

    def test_every_included_header_is_fingerprinted_or_reviewed(self):
        reached = set()
        for kind, paths in INPUTS.items():
            for name in paths:
                for path in included(name) - set(paths):
                    reached.add(path)
                    with self.subTest(kind=kind, source=name, header=path):
                        self.assertIn(path, REVIEWED)
        self.assertEqual(set(REVIEWED) - reached, set(), "stale reviewed headers")

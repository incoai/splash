import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tools import check_architecture


class ArchitectureTests(unittest.TestCase):
    def test_serving_modules_depend_on_lower_layers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = root / "server"
            server.mkdir()
            (server / "server.py").write_text("from .frontend import Frontend")
            (server / "frontend.py").write_text("from .backend import Job")
            (server / "backend.py").write_text("from . import runtime")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for statement, module in (
                    ("from .frontend import Frontend", None),
                    ("from frontend import Frontend", "frontend"),
                    ("from server.frontend import Frontend", "server.frontend"),
                    ("from server import frontend", "server"),
                    ("import server.frontend", "server.frontend"),
                    (
                        "import importlib\nimportlib.import_module('server.frontend')",
                        None,
                    ),
                    (
                        "from importlib import import_module\n"
                        "import_module('.frontend', 'server')",
                        None,
                    ),
                    ("__import__('server.frontend')", None),
                ):
                    with self.subTest(statement=statement):
                        (server / "backend.py").write_text(statement)
                        # The package's own modules are imported relatively.
                        style = (
                            [
                                f"server/backend.py: imports {module} without a "
                                "relative import"
                            ]
                            if module
                            else []
                        )
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "server/backend.py: imports upper serving layer frontend",
                                *style,
                            ],
                        )

    def test_package_modules_import_their_siblings_relatively(self):
        header = (
            "import sys\nfrom pathlib import Path\n\n"
            'if __name__ == "__main__" and not __package__:\n'
            "    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
            '    __package__ = "install"\n\n'
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for package in ("server", "install"):
                (root / package).mkdir()
            (root / "server/errors.py").write_text("")
            (root / "server/backend.py").write_text("from .errors import APIError\n")
            (root / "install/paths.py").write_text("")
            (root / "install/clients.py").write_text("")
            # Another package's modules are imported by their package.
            (root / "install/launcher.py").write_text(
                header + "from server import serve_options\n\nfrom . import paths\n"
            )
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for path, text, errors in (
                    (
                        "server/backend.py",
                        "from errors import APIError",
                        ["imports errors without a relative import"],
                    ),
                    (
                        "server/backend.py",
                        "import errors",
                        ["imports errors without a relative import"],
                    ),
                    (
                        "server/backend.py",
                        "if __package__:\n    from .errors import APIError\n"
                        "else:\n    from errors import APIError\n",
                        [
                            "reads __package__",
                            "imports errors without a relative import",
                        ],
                    ),
                    (
                        "install/launcher.py",
                        header + "import paths\n",
                        ["imports paths without a relative import"],
                    ),
                    # Nor by the package's own name.
                    (
                        "install/clients.py",
                        "from install import paths",
                        ["imports install without a relative import"],
                    ),
                    (
                        "install/clients.py",
                        "import install.paths",
                        ["imports install.paths without a relative import"],
                    ),
                    # Only the script entry points take the header.
                    (
                        "install/clients.py",
                        header + "from . import paths\n",
                        ["reads __package__", "reads __package__"],
                    ),
                ):
                    original = (root / path).read_text()
                    with self.subTest(path=path, text=text):
                        (root / path).write_text(text)
                        self.assertCountEqual(
                            check_architecture.check(),
                            [f"{path}: {error}" for error in errors],
                        )
                    (root / path).write_text(original)

    def test_serving_entrypoint_is_not_a_shared_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "server/output.py"
            source.parent.mkdir()
            source.write_text("def parse():\n    from .server import FrontendHandler\n")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(
                    check_architecture.check(),
                    ["server/output.py: imports upper serving layer server"],
                )

    def test_workspace_policy_belongs_to_operators(self):
        symbols = check_architecture.OPERATOR_WORKSPACE_POLICY_NAMES
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            operator = root / "runtime/ops/Attention.cpp"
            operator.parent.mkdir(parents=True)
            operator.write_text("\n".join(symbols))
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for relative in (
                    "runtime/model/Runtime.cpp",
                    "runtime/engine/Scheduler.cpp",
                    "runtime/engine/Bootstrap.hpp",
                    "runtime/engine/Bootstrap.mm",
                    "runtime/engine/RuntimeResources.hpp",
                    "runtime/engine/RuntimeResources.mm",
                ):
                    source = root / relative
                    source.parent.mkdir(parents=True, exist_ok=True)
                    for symbol in symbols:
                        with self.subTest(source=relative, symbol=symbol):
                            source.write_text(f"auto size = ops::{symbol};")
                            errors = check_architecture.check()
                            layer = relative.split("/")[1]
                            self.assertEqual(
                                errors,
                                [
                                    f"{relative}: {layer} owns an operator workspace policy"
                                ],
                            )
                    source.unlink()

    def test_workspace_policy_names_exist_in_operators(self):
        self.assertEqual(check_architecture.stale_policy_names(), [])

    def test_workspace_policy_reports_a_vanished_name(self):
        symbols = check_architecture.OPERATOR_WORKSPACE_POLICY_NAMES
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            operator = root / "runtime/ops/Attention.cpp"
            operator.parent.mkdir(parents=True)
            operator.write_text("\n".join(symbols[1:]))
            # A name that only survives in model/ or engine/ is not an operator's.
            model = root / "runtime/model/Runtime.cpp"
            model.parent.mkdir(parents=True)
            model.write_text(symbols[0])
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.stale_policy_names(), [symbols[0]])

    def test_production_measures_time_the_mac_is_awake(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # What production measures time with: durations on the awake
            # clock and its waits, wall-clock instants on the system clock.
            allowed = root / "runtime/engine/Clocks.cpp"
            allowed.parent.mkdir(parents=True)
            allowed.write_text(
                "AwakeClock::now();\n"
                "wake.wait_until(lock, AwakeClock::now() + kSlice, ready);\n"
                "std::this_thread::sleep_for(kPoll);\n"
                "std::chrono::system_clock::now();\n"
            )
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for relative, text in (
                    (
                        "runtime/metal/MetalBackend.mm",
                        "std::chrono::steady_clock::now();\n",
                    ),
                    (
                        "runtime/engine/Engine.cpp",
                        "using Clock = std::chrono::steady_clock;\n",
                    ),
                    ("runtime/main.mm", "std::chrono::high_resolution_clock::now();\n"),
                    # A wait for a duration measures it on the steady clock.
                    (
                        "runtime/engine/RuntimeResources.mm",
                        "wake.wait_for(lock, kProbation, stopped);\n",
                    ),
                    ("runtime/engine/Cache.cpp", "mutex.try_lock_for(kWait);\n"),
                    ("runtime/engine/KvPool.cpp", "slots.try_acquire_for(kWait);\n"),
                ):
                    with self.subTest(source=relative):
                        source = root / relative
                        source.parent.mkdir(parents=True, exist_ok=True)
                        source.write_text(text)
                        self.assertEqual(
                            check_architecture.check(),
                            [f"{relative}: measures time on a clock that counts sleep"],
                        )
                        source.unlink()

    def test_only_engine_assembly_depends_on_concrete_models(self):
        headers = (
            "model/DFlashDraft.hpp",
            "model/ModelFactory.hpp",
            "model/Qwen3_6Moe.hpp",
            "model/Qwen3_8.hpp",
            "model/QwenHybridLayout.hpp",
            "model/QwenState.hpp",
            "model/QwenTarget.hpp",
            "model/QwenTargetFiles.hpp",
            "model/QwenTargetLoader.hpp",
            "model/Runtime.hpp",
            "model/WeightStore.hpp",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime/engine").mkdir(parents=True)
            for assembly in (
                "Bootstrap.hpp",
                "Bootstrap.mm",
                "RuntimeResources.hpp",
                "RuntimeResources.mm",
            ):
                (root / "runtime/engine" / assembly).write_text(
                    "".join(f'#include "{header}"\n' for header in headers)
                )
            policy = root / "runtime/engine/Scheduler.cpp"
            startup = root / "runtime/main.mm"
            generic = (
                '#include "model/Model.hpp"\n#include "model/ModelDescriptor.hpp"\n'
            )
            policy.write_text(generic)
            startup.write_text(generic + "model::ModelDescriptor model;\n")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for header in headers:
                    with self.subTest(source="engine policy", header=header):
                        policy.write_text(f'#include "{header}"\n')
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/engine/Scheduler.cpp: engine policy "
                                f"depends on concrete model {header}"
                            ],
                        )
                policy.write_text(generic)
                for header in headers:
                    with self.subTest(source="startup", header=header):
                        startup.write_text(f'#include "{header}"\n')
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/main.mm: startup depends on concrete "
                                f"model {header}"
                            ],
                        )
                for name in ("QwenTarget", "DFlashDraft", "Runtime"):
                    with self.subTest(source="startup", symbol=name):
                        startup.write_text(f"auto instance = model::{name}{{}};\n")
                        self.assertEqual(
                            check_architecture.check(),
                            ["runtime/main.mm: startup names a concrete model type"],
                        )

    def test_metal_depends_on_no_upper_production_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime/model").mkdir(parents=True)
            (root / "runtime/model/Model.hpp").write_text("")
            kernel = root / "runtime/metal/kernels/shared/rope.metal"
            kernel.parent.mkdir(parents=True)
            with mock.patch.object(check_architecture, "ROOT", root):
                for include, header in (
                    ('#include "engine/Engine.hpp"', "engine/Engine.hpp"),
                    ('#include "model/Model.hpp"', "model/Model.hpp"),
                    ('#include "ops/Linear.hpp"', "ops/Linear.hpp"),
                    ('#include "../../../model/Model.hpp"', "model/Model.hpp"),
                ):
                    with self.subTest(include=include):
                        kernel.write_text(include + "\n")
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/metal/kernels/shared/rope.metal: Metal "
                                f"depends on production layer {header}"
                            ],
                        )

    def test_includes_are_read_where_the_compiler_finds_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime/model").mkdir(parents=True)
            (root / "runtime/model/QwenTarget.hpp").write_text("")
            policy = root / "runtime/engine/Scheduler.cpp"
            policy.parent.mkdir()
            with mock.patch.object(check_architecture, "ROOT", root):
                for include in (
                    '# include "model/QwenTarget.hpp"',
                    '#include"model/QwenTarget.hpp"',
                    "#  import <model/QwenTarget.hpp>",
                    '#include "../model/QwenTarget.hpp"',
                    '#include "model/../model/QwenTarget.hpp"',
                ):
                    with self.subTest(include=include):
                        policy.write_text(include + "\n")
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/engine/Scheduler.cpp: engine policy "
                                "depends on concrete model model/QwenTarget.hpp"
                            ],
                        )

    def test_production_cannot_include_test_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime/ops/Linear.cpp"
            source.parent.mkdir(parents=True)
            source.write_text('#include "tests/engine/LinearNumerics.hpp"\n')
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(
                    check_architecture.check(),
                    [
                        "runtime/ops/Linear.cpp: production depends on "
                        "test code tests/engine/LinearNumerics.hpp"
                    ],
                )

    def test_plan_orchestration_and_storage_geometry_remain_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "runtime/model/Runtime.cpp"
            assembly = root / "runtime/engine/RuntimeResources.mm"
            model.parent.mkdir(parents=True)
            assembly.parent.mkdir(parents=True)
            model.write_text(
                "auto bytes = PagedAttention::prefillWorkspace();\n"
                "constexpr auto tileRows = kv::kPageTokens;\n"
                "auto rows = ExecutionLimits::draftQueryRows;\n"
                "auto rank = layout.selectorRank;\n"
            )
            assembly.write_text(
                '#include "model/Runtime.hpp"\nops::ExecutionPlans plans(device);\n'
            )
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])

    def test_no_workflow_passes_a_hugging_face_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflows = root / ".github/workflows"
            workflows.mkdir(parents=True)
            # The path that turns the token store off names no token.
            (workflows / "ci.yml").write_text("env:\n  HF_TOKEN_PATH: /nonexistent\n")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for name, text, token in (
                    (
                        "release.yml",
                        "env:\n  HF_TOKEN: ${{ secrets.HF_TOKEN }}\n",
                        "HF_TOKEN",
                    ),
                    (
                        "release.yaml",
                        "run: HUGGING_FACE_HUB_TOKEN=x make\n",
                        "HUGGING_FACE_HUB_TOKEN",
                    ),
                    (
                        "release.yml",
                        "with:\n  token: ${{ secrets.HF_READ }}\n",
                        "secrets.HF_",
                    ),
                    (
                        "release.yml",
                        "with:\n  token: ${{ secrets['HF_READ'] }}\n",
                        "secrets['HF_",
                    ),
                ):
                    with self.subTest(text=text):
                        workflow = workflows / name
                        workflow.write_text(text)
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                f".github/workflows/{name}: passes a Hugging Face token ({token})"
                            ],
                        )
                        workflow.unlink()

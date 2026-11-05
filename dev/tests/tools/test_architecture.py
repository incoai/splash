import subprocess
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

    def test_dialects_render_without_the_handler_or_the_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "server/dialects").mkdir(parents=True)
            with mock.patch.object(check_architecture, "ROOT", root):
                for name, text, errors in (
                    (
                        "dialects/openai.py",
                        "from ..chat_messages import normalize_messages\n",
                        [],
                    ),
                    (
                        "dialects/openai.py",
                        "from ..server import FrontendHandler\n",
                        ["imports upper serving layer server"],
                    ),
                    (
                        "dialects/openai.py",
                        "from ..backend import NativeResult\n",
                        ["imports upper serving layer backend"],
                    ),
                    (
                        "dialects/openai.py",
                        "from server.dialects import anthropic\n",
                        [
                            "imports sibling dialect anthropic",
                            "imports server.dialects without a relative import",
                        ],
                    ),
                    # A subpackage's own name is its folder's, so a module
                    # may import the library a dialect is named after.
                    ("dialects/openai.py", "import openai\n", []),
                    # The layers below the dialects never import them.
                    (
                        "output.py",
                        "from .dialects import openai\n",
                        ["imports upper serving layer dialects"],
                    ),
                    # Each API has one owner: no dialect imports another,
                    # and the modules every dialect reaches import none.
                    (
                        "dialects/openai.py",
                        "from .anthropic import ERRORS\n",
                        ["imports sibling dialect anthropic"],
                    ),
                    (
                        "dialects/openai.py",
                        "from ..dialects import anthropic\n",
                        ["imports sibling dialect anthropic"],
                    ),
                    (
                        "dialects/openai.py",
                        "from . import responses, scoring\n",
                        [
                            "imports sibling dialect responses",
                            "imports sibling dialect scoring",
                        ],
                    ),
                    ("dialects/__init__.py", "from . import anthropic, openai\n", []),
                    (
                        "frontend.py",
                        "from .dialects import responses\n",
                        ["imports upper serving layer dialects"],
                    ),
                    (
                        "chat_messages.py",
                        "from .frontend import Frontend\n",
                        ["imports upper serving layer frontend"],
                    ),
                ):
                    with self.subTest(name=name, text=text):
                        path = root / "server" / name
                        path.write_text(text)
                        try:
                            self.assertEqual(
                                check_architecture.check(),
                                [f"server/{name}: {error}" for error in errors],
                            )
                        finally:
                            path.unlink()

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
                            # Only this rule's findings: in an engine policy file a
                            # symbol such as MoeExpertTile also trips R3, which
                            # test_engine_policy_names_no_model_mechanism covers.
                            errors = [
                                error
                                for error in check_architecture.check()
                                if error.endswith("owns an operator workspace policy")
                            ]
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

    def test_no_residency_set_is_attached_to_a_queue_or_command_buffer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # What a command makes resident: the memory its encoders use.
            allowed = root / "runtime/metal/MetalBackend.mm"
            allowed.parent.mkdir(parents=True)
            allowed.write_text(
                "[encoder useResources:resources count:count usage:usage];\n"
            )
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for text in (
                    "[commands_ addResidencySet:addressed_];\n",
                    "[command useResidencySet:set];\n",
                ):
                    with self.subTest(text=text):
                        source = root / "runtime/metal/Residency.hpp"
                        source.write_text(text)
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/metal/Residency.hpp: attaches a residency set to a queue or command buffer"
                            ],
                        )
                        source.unlink()

    def test_only_page_storage_writes_gpu_addresses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "runtime/ops/PageStorage.cpp"
            allowed.parent.mkdir(parents=True)
            allowed.write_text("const uint64_t address = allocated.gpuAddress();\n")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                source = root / "runtime/ops/BlockIndexer.cpp"
                source.write_text("entries.push_back(extent.gpuAddress());\n")
                self.assertEqual(
                    check_architecture.check(),
                    [
                        "runtime/ops/BlockIndexer.cpp: writes a GPU address outside page storage"
                    ],
                )

    def test_layers_depend_only_on_the_layers_below(self):
        allowed = (
            ("runtime/main.mm", "engine/Bootstrap.hpp"),
            ("runtime/main.mm", "engine/FdTransport.hpp"),
            ("runtime/engine/Bootstrap.mm", "model/QwenTarget.hpp"),
            ("runtime/engine/RuntimeResources.mm", "ops/ExecutionPlans.hpp"),
            ("runtime/engine/RuntimeResources.mm", "metal/abi/PagedAttention.h"),
            ("runtime/engine/Scheduler.cpp", "model/Model.hpp"),
            ("runtime/engine/Scheduler.cpp", "AwakeClock.hpp"),
            ("runtime/model/QwenTarget.cpp", "ops/Linear.hpp"),
            ("runtime/model/QwenTarget.cpp", "metal/MetalBackend.hpp"),
            ("runtime/model/Model.hpp", "ops/PagedKv.hpp"),
            ("runtime/ops/Linear.cpp", "metal/abi/Gguf.h"),
            ("runtime/ops/AneFfn.hpp", "ane/Program.hpp"),
            ("runtime/ane/Program.hpp", "metal/MetalBackend.hpp"),
            ("runtime/metal/MetalBackend.mm", "Supervision.hpp"),
            ("runtime/metal/kernels/shared/rope.metal", "metal/abi/RoPE.h"),
            ("runtime/metal/abi/Gguf.h", "metal/kernels/common/quant_formats.h"),
            ("runtime/metal/kernels/shared/rope.metal", "metal_stdlib"),
        )
        forbidden = (
            ("runtime/main.mm", "model/QwenTarget.hpp", "startup", "model"),
            ("runtime/engine/Scheduler.cpp", "model/QwenTarget.hpp", "engine", "model"),
            ("runtime/engine/Status.cpp", "model/QwenTarget.hpp", "engine", "model"),
            ("runtime/engine/Scheduler.cpp", "ops/GDN.hpp", "engine", "ops"),
            (
                "runtime/engine/Scheduler.cpp",
                "metal/MetalBackend.hpp",
                "engine",
                "metal",
            ),
            (
                "runtime/engine/Scheduler.cpp",
                "engine/Bootstrap.hpp",
                "engine",
                "engine assembly",
            ),
            ("runtime/model/QwenTarget.cpp", "engine/Engine.hpp", "model", "engine"),
            (
                "runtime/model/Model.hpp",
                "model/QwenTarget.hpp",
                "model interface",
                "model",
            ),
            ("runtime/ops/Linear.cpp", "model/Model.hpp", "ops", "model interface"),
            ("runtime/ops/Linear.cpp", "engine/Engine.hpp", "ops", "engine"),
            ("runtime/metal/MetalBackend.mm", "ops/Linear.hpp", "metal", "ops"),
            ("runtime/ane/Handoff.mm", "model/QwenTarget.hpp", "ane", "model"),
            ("runtime/metal/MetalBackend.mm", "ane/Program.hpp", "metal", "ane"),
            ("runtime/model/QwenTarget.cpp", "ane/Program.hpp", "model", "ane"),
            (
                "runtime/metal/kernels/shared/rope.metal",
                "engine/Engine.hpp",
                "kernels",
                "engine",
            ),
            (
                "runtime/metal/kernels/shared/rope.metal",
                "ops/Linear.hpp",
                "kernels",
                "ops",
            ),
            (
                "runtime/metal/kernels/shared/rope.metal",
                "metal/MetalBackend.hpp",
                "kernels",
                "metal",
            ),
            ("runtime/metal/abi/RoPE.h", "AwakeClock.hpp", "kernels", "shared"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime").mkdir()
            for shared in ("AwakeClock.hpp", "Supervision.hpp"):
                (root / "runtime" / shared).write_text("")
            with mock.patch.object(check_architecture, "ROOT", root):
                for name, include in allowed:
                    with self.subTest(source=name, include=include):
                        self.assertEqual(
                            check_architecture.check_layers(
                                name, [f"runtime/{include}"]
                            ),
                            [],
                        )
                for name, include, layer, target in forbidden:
                    with self.subTest(source=name, include=include):
                        source = root / name
                        source.parent.mkdir(parents=True, exist_ok=True)
                        source.write_text(f'#include "{include}"\n')
                        self.assertIn(
                            f"{name}: {layer} depends on {target} ({include})",
                            check_architecture.check(),
                        )
                        source.unlink()

    def test_layer_exceptions_name_crossings_that_exist(self):
        self.assertEqual(check_architecture.stale_layer_exceptions(), [])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime/engine/KvPool.hpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                '#include "ops/PagedKv.hpp"\n#include "model/Model.hpp"\n'
            )
            exceptions = frozenset(
                {
                    ("runtime/engine/KvPool.hpp", "ops/PagedKv.hpp"),
                    ("runtime/engine/KvPool.hpp", "metal/MetalBackend.hpp"),
                    ("runtime/engine/KvPool.hpp", "model/Model.hpp"),
                    ("runtime/engine/Gone.hpp", "ops/PagedKv.hpp"),
                }
            )
            with (
                mock.patch.object(check_architecture, "ROOT", root),
                mock.patch.object(check_architecture, "LAYER_EXCEPTIONS", exceptions),
            ):
                self.assertEqual(check_architecture.check(), [])
                self.assertEqual(
                    check_architecture.stale_layer_exceptions(),
                    [
                        ("runtime/engine/Gone.hpp", "ops/PagedKv.hpp"),
                        ("runtime/engine/KvPool.hpp", "metal/MetalBackend.hpp"),
                        # The interface is the engine's to include.
                        ("runtime/engine/KvPool.hpp", "model/Model.hpp"),
                    ],
                )

    def test_frontend_reaches_the_runtime_through_the_wire_protocol_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            for directory in ("server", "install", "app/Sources/Splash", "app/.build"):
                (root / directory).mkdir(parents=True)
            (root / ".gitignore").write_text("app/.build/\n")
            (root / "runtime").mkdir()
            (root / "runtime/Supervision.hpp").write_text("")
            # Prose may cite the runtime, and an ignored build is no source.
            (root / "server/protocol.py").write_text(
                '"""Mirrors runtime/engine/Protocol.hpp."""\n'
                "# runtime/Supervision.hpp owns the stop budget.\n"
                'LOCKS = "build/runtime"\n'
            )
            (root / "app/Sources/Splash/Runtime.swift").write_text(
                "// runtime/Supervision.hpp `kStop`\n"
                'let locks = root.appending(path: "build/runtime")\n'
            )
            (root / "app/.build/Bridge.h").write_text('#include "engine/Engine.hpp"\n')
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check_frontend(), [])
                for name, text, target, path in (
                    (
                        "server/protocol.py",
                        'HEADER = ROOT / "runtime/engine/Protocol.hpp"\n',
                        "engine",
                        "engine/Protocol.hpp",
                    ),
                    (
                        "install/build.py",
                        'SOURCES = ["runtime/Supervision.hpp"]\n',
                        "shared",
                        "Supervision.hpp",
                    ),
                    (
                        "install/build.sh",
                        "# Builds the server.\nclang++ -c runtime/engine/Engine.cpp\n",
                        "engine",
                        "engine/Engine.cpp",
                    ),
                    (
                        "app/Package.swift",
                        '.target(name: "Engine", path: "../runtime/engine")\n',
                        "engine",
                        "engine",
                    ),
                    (
                        "app/Sources/Splash/Bridge.h",
                        '#include "model/Model.hpp"\n',
                        "model interface",
                        "model/Model.hpp",
                    ),
                ):
                    with self.subTest(source=name):
                        source = root / name
                        original = source.read_text() if source.exists() else None
                        source.write_text(text)
                        self.assertEqual(
                            check_architecture.check_frontend(),
                            [f"{name}: frontend depends on {target} ({path})"],
                        )
                        if original is None:
                            source.unlink()
                        else:
                            source.write_text(original)

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
                                "runtime/engine/Scheduler.cpp: engine depends on "
                                "model (model/QwenTarget.hpp)"
                            ],
                        )

    def test_engine_includes_model_headers_from_the_interface(self):
        boundary = "".join(
            f'#include "{header}"\n'
            for header in sorted(check_architecture.ENGINE_MODEL_BOUNDARY)
        )
        internals = (
            "model/ExpertStore.hpp",
            "model/KvHistoryStream.hpp",
            "model/ModelDescriptor.hpp",
            "model/SparseRowStore.hpp",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "runtime/engine/MemoryControl.cpp"
            policy.parent.mkdir(parents=True)
            policy.write_text(boundary)
            # Assembly includes what it assembles.
            for source in check_architecture.ENGINE_ASSEMBLY:
                (root / source).write_text(
                    "".join(f'#include "{header}"\n' for header in internals)
                )
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.check(), [])
                for header in internals:
                    with self.subTest(header=header):
                        policy.write_text(boundary + f'#include "{header}"\n')
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/engine/MemoryControl.cpp: engine depends "
                                f"on model ({header})"
                            ],
                        )

    def test_startup_and_status_name_no_concrete_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime/engine").mkdir(parents=True)
            with mock.patch.object(check_architecture, "ROOT", root):
                for name, role in (
                    ("runtime/main.mm", "startup"),
                    ("runtime/engine/Status.cpp", "status writer"),
                ):
                    source = root / name
                    # Both pass on or render mechanisms and model telemetry.
                    source.write_text(
                        "auto expertMemory = 0, historyStream = 0;\n"
                        "model::KvReadStats reads;\n"
                    )
                    self.assertEqual(check_architecture.check(), [])
                    for symbol, named in (
                        ("QwenTarget", "a model family"),
                        ("DFlashDraft", "a model family"),
                        ("FlintLayout", "a model family"),
                        ("Runtime", "a concrete model type"),
                    ):
                        with self.subTest(source=name, symbol=symbol):
                            source.write_text(f"auto instance = model::{symbol}{{}};\n")
                            self.assertEqual(
                                check_architecture.check(),
                                [f"{name}: {role} names {named} ({symbol})"],
                            )
                    source.unlink()

    def test_engine_policy_names_no_model_family(self):
        identifiers = "model::QwenTarget target;\nauto rows = kDFlashRows;\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "runtime/engine/Scheduler.cpp"
            policy.parent.mkdir(parents=True)
            with mock.patch.object(check_architecture, "ROOT", root):
                for text in (
                    "// Qwen3.8-Flash-Next's prompts\nint rows = 0;\n",
                    "/* a DFlash draft's\n   window */ int rows = 0;\n",
                    "auto label = \"flint\";\nauto first = 'q';\n",
                    # Words that only contain a family's name.
                    "auto flinty = 0, qwenish = 0;\n",
                    # A name's words that no concrete model is named by.
                    "auto nextFlash = 0, b = 0, incoai = 0;\nuint8_t a3 = 0;\n",
                ):
                    with self.subTest(text=text):
                        policy.write_text(text)
                        self.assertEqual(check_architecture.check(), [])
                for text, identifier in (
                    ("model::QwenTarget target;\n", "QwenTarget"),
                    ("auto rows = kDFlashRows;\n", "kDFlashRows"),
                    ("bool dflash_draft = false;\n", "dflash_draft"),
                    ("constexpr int FLINT_LAYERS = 4;\n", "FLINT_LAYERS"),
                    ("struct OrnithLayout {};\n", "OrnithLayout"),
                    ("auto qwen2Rows = 0;\n", "qwen2Rows"),
                    # A digit separator starts no character literal.
                    ("auto rows = 1'000 * qwenRows / 1'000;\n", "qwenRows"),
                ):
                    with self.subTest(text=text):
                        policy.write_text(text)
                        self.assertEqual(
                            check_architecture.check(),
                            [
                                "runtime/engine/Scheduler.cpp: engine policy names "
                                f"a model family ({identifier})"
                            ],
                        )
                policy.unlink()
                for source in sorted(check_architecture.ENGINE_ASSEMBLY):
                    with self.subTest(source=source):
                        (root / source).write_text(identifiers)
                        self.assertEqual(check_architecture.check(), [])

    def test_every_family_names_itself_with_a_family_word(self):
        families = check_architecture.family_names(
            check_architecture.ROOT / "install/families.py"
        )
        self.assertTrue(families)
        for name, _ in families:
            with self.subTest(family=name):
                self.assertTrue(
                    check_architecture.MODEL_FAMILY_PARTS.intersection(
                        check_architecture.identifier_parts(name)
                    )
                )

    def test_family_words_are_brands_and_what_a_concrete_model_goes_by(self):
        self.assertEqual(
            check_architecture.family_words(
                [("Gemma4-12B-Flash", "incoai/Opal-2-12B-DFlash2")],
                (
                    "model/Gemma4Target.hpp",
                    "model/DFlashDraft.hpp",
                    "model/Runtime.hpp",
                ),
            ),
            {"gemma", "opal", "dflash"},
        )

    def test_engine_policy_names_no_model_mechanism(self):
        name = "runtime/engine/Engine.cpp"
        for text, identifier in (
            ("void dropExperts();\n", "dropExperts"),
            ("auto experts = 0;\n", "experts"),
            ("bool hasIndexer = false;\n", "hasIndexer"),
            ("auto PLERows = 0;\n", "PLERows"),
            ("auto kvHistoryStream = 0;\n", "kvHistoryStream"),
            ("auto history_staging_bytes = 0;\n", "history_staging_bytes"),
            ("auto kvBlockCacheBytes = 0;\n", "kvBlockCacheBytes"),
            ("model::SparseRowStore *rows = nullptr;\n", "SparseRowStore"),
            ("auto hyperConnections = 0;\n", "hyperConnections"),
            ("bool valuesTokenMajor = false;\n", "valuesTokenMajor"),
            # A mechanism's words in any order, singular or plural.
            ("bool streamsHistory = false;\n", "streamsHistory"),
            ("auto historyStreams = 0;\n", "historyStreams"),
            ("auto historyStaging = 0;\n", "historyStaging"),
            ("auto cacheBlocks = 0;\n", "cacheBlocks"),
            ("auto rowsSparse = 0;\n", "rowsSparse"),
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    check_architecture.check_model_names(name, text),
                    [f"{name}: engine policy names a model mechanism ({identifier})"],
                )
        # Words that only contain or neighbour a mechanism's, and comments.
        text = (
            "auto historyBytes = 0, streamConflicts = 0, blockSize = 0;\n"
            "auto sampled = 0, expertise = 0, indexes = 0;\n"
            "bool gainsIndexPages = false;\nauto indexPageLanes = 0;\n"
            "// an expert's history stream\n"
        )
        self.assertEqual(check_architecture.check_model_names(name, text), [])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / name
            source.parent.mkdir(parents=True)
            source.write_text("void dropExperts();\n")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(
                    check_architecture.check(),
                    [f"{name}: engine policy names a model mechanism (dropExperts)"],
                )

    def test_identifiers_split_into_words(self):
        for identifier, parts in (
            ("kvHistoryStream", ["kv", "history", "stream"]),
            ("history_staging_bytes", ["history", "staging", "bytes"]),
            ("DFlashDraft", ["dflash", "draft"]),
            ("kDFlashRows", ["k", "dflash", "rows"]),
            ("PLERows", ["ple", "rows"]),
            ("Qwen3_6Moe", ["qwen", "3", "6", "moe"]),
            ("FLINT_LAYERS", ["flint", "layers"]),
        ):
            with self.subTest(identifier=identifier):
                self.assertEqual(check_architecture.identifier_parts(identifier), parts)

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

    def test_notes_outside_the_repository_are_reported(self):
        # Each note's name is joined at run time, so that this file names none.
        amendment = "am" + "end/"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "docs").mkdir()
            (root / "docs/validate.md").write_text("See [the API](api.md).\n")
            (root / "docs/api.md").write_text("Fields: `runtime/engine/Gone.hpp`.\n")
            (root / "runtime/engine").mkdir(parents=True)
            (root / "runtime/engine/Engine.hpp").write_text(
                "// The rule of docs/validate.md and engine/Engine.hpp.\n"
                'auto path = "docs/gone.md";\n'
            )
            # An ignored file is not the repository's.
            (root / ".gitignore").write_text("scratch.md\n")
            (root / "scratch.md").write_text(f"{amendment}V.md\n")
            with (
                mock.patch.object(check_architecture, "ROOT", root),
                mock.patch.object(check_architecture, "NOTE_ALLOWANCE", {}),
            ):
                self.assertEqual(
                    check_architecture.check_note_references(),
                    [
                        "docs/api.md: names a missing repository file "
                        "(runtime/engine/Gone.hpp)"
                    ],
                )
                (root / "docs/api.md").write_text("Fields.\n")
                for reference in (
                    amendment,
                    "AMEND" + "MENTS",
                    "ATTEMP" + "TS",
                    "DESIGN" + "-1",
                    "DESIGN" + " section",
                    "merge" + " report",
                    "owner" + " 11-03 17:30",
                    "coordinator" + " 09:20",
                    "Coordinator" + " 11-03 09:20",
                    "owner" + "'s trade (20:30",
                ):
                    with self.subTest(reference=reference):
                        (root / "runtime/engine/Cache.cpp").write_text(
                            f"// as ({reference} 2) says\n"
                        )
                        self.assertEqual(
                            check_architecture.check_note_references(),
                            [
                                "runtime/engine/Cache.cpp: refers to a note "
                                f"outside the repository ({reference})"
                            ],
                        )
                for text, path in (
                    ("// the ledger (design/M.md B5)\n", "design/M.md"),
                    ("// see runtime/engine/Gone.cpp:40\n", "runtime/engine/Gone.cpp"),
                    ("/* engine/Gone.hpp */\n", "engine/Gone.hpp"),
                ):
                    with self.subTest(text=text):
                        (root / "runtime/engine/Cache.cpp").write_text(text)
                        self.assertEqual(
                            check_architecture.check_note_references(),
                            [
                                "runtime/engine/Cache.cpp: names a missing "
                                f"repository file ({path})"
                            ],
                        )

    def test_note_allowance_only_shrinks(self):
        note = "# " + "am" + "end/V.md\n"
        allowed = {"dev/benchmarks/run.py": 2}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            source = root / "dev/benchmarks/run.py"
            source.parent.mkdir(parents=True)
            with mock.patch.object(check_architecture, "ROOT", root):
                for count, allowance, errors in (
                    (2, allowed, []),
                    (
                        1,
                        allowed,
                        [
                            "dev/benchmarks/run.py: holds 1 references to notes "
                            "outside the repository, its allowance 2: lower it"
                        ],
                    ),
                    (
                        2,
                        allowed | {"dev/gone.py": 1},
                        [
                            "dev/gone.py: has a note allowance but is not in the repository"
                        ],
                    ),
                    # Above its allowance, a file's references are each reported.
                    (3, allowed, ["dev/benchmarks/run.py: refers to a note"] * 3),
                ):
                    with (
                        self.subTest(count=count, allowance=allowance),
                        mock.patch.object(
                            check_architecture, "NOTE_ALLOWANCE", allowance
                        ),
                    ):
                        source.write_text(note * count)
                        self.assertEqual(
                            [
                                error.partition(" outside the repository (")[0]
                                for error in check_architecture.check_note_references()
                            ],
                            errors,
                        )

    def test_engine_size_stays_within_its_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime/engine").mkdir(parents=True)
            (root / "runtime/model").mkdir()
            (root / "runtime/engine/Engine.cpp").write_text("int a;\nint b;\n")
            (root / "runtime/engine/Engine.hpp").write_text("#pragma once")
            (root / "runtime/model/Model.hpp").write_text("int c;\n" * 9)
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(check_architecture.engine_lines(), 3)
            with mock.patch.object(check_architecture, "ENGINE_LINE_BUDGET", 3):
                self.assertEqual(check_architecture.check_engine_size(3), [])
                self.assertEqual(
                    check_architecture.check_engine_size(4),
                    ["runtime/engine holds 4 production lines, over its budget 3"],
                )

    def test_rules_name_paths_that_exist(self):
        self.assertEqual(check_architecture.missing_listed_paths(), [])
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(check_architecture, "ROOT", Path(temporary)):
                missing = check_architecture.missing_listed_paths()
        for path in (
            "runtime/engine/Bootstrap.hpp",  # engine assembly
            "runtime/model/Model.hpp",  # the model's interface
            "runtime/engine/Status.cpp",  # the status writer
            "runtime/model/QwenTarget.hpp",  # a concrete model
        ):
            self.assertIn(path, missing)

    def test_every_runtime_source_belongs_to_a_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime/scratch/Helper.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("int helper();\n")
            with mock.patch.object(check_architecture, "ROOT", root):
                self.assertEqual(
                    check_architecture.check(),
                    ["runtime/scratch/Helper.cpp: belongs to no layer"],
                )

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dev.tools import build_config, build_identity


class BuildIdentityTests(unittest.TestCase):
    def test_copied_input_is_stable_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = Path("runtime/model/Qwen3_8.hpp")
            destination = root / relative
            destination.parent.mkdir(parents=True)
            shutil.copy2(build_identity.ROOT / relative, destination)
            constants = {
                "metal_flags": "-std=metal4.0 -O3",
                "engine_cxxflags": "-std=c++20 -O3",
            }
            first = build_identity.build_id(root, [relative.as_posix()], constants)
            second = build_identity.build_id(
                root,
                [relative.as_posix()],
                dict(reversed(tuple(constants.items()))),
            )
            self.assertEqual(first, second)
            destination.write_bytes(destination.read_bytes() + b"\n// mutation\n")
            changed = build_identity.build_id(root, [relative.as_posix()], constants)
            self.assertNotEqual(first, changed)

    def test_generated_header_and_stamp_change_only_with_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / "BuildIdentity.hpp"
            stamp = root / "build-identity.json"
            first = "src-" + "a" * 64
            second = "src-" + "b" * 64
            self.assertTrue(
                build_identity.update_if_changed(
                    header, build_identity.header_bytes(first)
                )
            )
            self.assertTrue(
                build_identity.update_if_changed(
                    stamp, build_identity.stamp_bytes(first)
                )
            )
            self.assertFalse(
                build_identity.update_if_changed(
                    header, build_identity.header_bytes(first)
                )
            )
            self.assertFalse(
                build_identity.update_if_changed(
                    stamp, build_identity.stamp_bytes(first)
                )
            )
            self.assertTrue(
                build_identity.update_if_changed(
                    header, build_identity.header_bytes(second)
                )
            )
            self.assertIn(second, header.read_text())

    def test_default_inputs_are_production_only(self):
        inputs = set(build_identity.production_input_paths())
        self.assertIn("runtime/main.mm", inputs)
        self.assertIn("runtime/metal/abi/ExecutionGeometry.h", inputs)
        self.assertIn("runtime/model/Runtime.mm", inputs)
        self.assertIn("runtime/metal/kernels/decode/linear_q4.metal", inputs)
        self.assertIn("runtime/metal/kernels/prefill/linear_q4.metal", inputs)
        self.assertIn("runtime/metal/abi/KernelABI.h", inputs)
        self.assertIn("runtime/metal/kernels/common/paged_attention_tile.h", inputs)
        self.assertIn("dev/tools/build_identity.py", inputs)
        self.assertFalse(any(path.startswith("dev/tests/") for path in inputs))
        self.assertFalse(any(path.startswith("dev/benchmarks/") for path in inputs))
        self.assertFalse(any(path.startswith("install/models/") for path in inputs))

    def test_c_execution_geometry_header_changes_default_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "runtime" / "metal" / "abi"
            sources.mkdir(parents=True)
            geometry = sources / "ExecutionGeometry.h"
            geometry.write_text("#define SPLASH_DFLASH_QUERY_ROWS 8\n")
            tool = root / "dev/tools/build_identity.py"
            tool.parent.mkdir(parents=True)
            tool.write_text("fixture tool")
            first = build_identity.build_id(root)
            geometry.write_text("#define SPLASH_DFLASH_QUERY_ROWS 7\n")
            self.assertNotEqual(first, build_identity.build_id(root))

    def test_alternate_root_shader_add_edit_and_remove_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tool = root / "dev/tools/build_identity.py"
            tool.parent.mkdir(parents=True)
            shutil.copy2(build_identity.ROOT / "dev/tools/build_identity.py", tool)
            shader = root / "runtime/metal/kernels/shared/alternate.metal"
            shader.parent.mkdir(parents=True)
            header = root / "generated/BuildIdentity.hpp"
            stamp = root / "generated/build-identity.json"

            def invoke(mode):
                return subprocess.run(
                    (
                        sys.executable,
                        str(build_identity.ROOT / "dev/tools/build_identity.py"),
                        mode,
                        "--root",
                        str(root),
                        "--header",
                        str(header),
                        "--stamp",
                        str(stamp),
                    ),
                    capture_output=True,
                    text=True,
                )

            initial = previous = build_identity.build_id(root)
            written = invoke("write")
            self.assertEqual(written.returncode, 0, written.stderr)
            self.assertEqual(json.loads(written.stdout)["build_id"], initial)
            for content in ("kernel version A", "kernel version B", None):
                with self.subTest(content=content):
                    if content is None:
                        shader.unlink()
                    else:
                        shader.write_text(content)
                    current = build_identity.build_id(root)
                    self.assertNotEqual(current, previous)
                    stale = invoke("check")
                    self.assertNotEqual(stale.returncode, 0)
                    self.assertIn("generated build identity is stale", stale.stderr)
                    written = invoke("write")
                    self.assertEqual(written.returncode, 0, written.stderr)
                    self.assertEqual(json.loads(written.stdout)["build_id"], current)
                    checked = invoke("check")
                    self.assertEqual(checked.returncode, 0, checked.stderr)
                    previous = current
            self.assertEqual(previous, initial)

    def test_make_tracks_generated_header_for_production_binary(self):
        makefile = (build_identity.ROOT / "Makefile").read_text()
        test_makefile = (build_identity.ROOT / "dev/native.mk").read_text()
        self.assertIn(
            "$(ENGINE_MAIN_OBJECT): runtime/main.mm $(BUILD_ID_HEADER)", makefile
        )
        self.assertIn("-include $(BUILD_ID_HEADER)", makefile)
        self.assertIn("verify-build-identity", test_makefile)


class CompileConfigurationTests(unittest.TestCase):
    FLAG_SETS = (
        ("ENGINE_CXXFLAGS", "engine/engine/Status.o"),
        ("PROD_METALFLAGS", "metal/shared/rope.air"),
        ("ENGINE_TEST_CXXFLAGS", "engine-tests/operator-tuning"),
        ("TEST_METALFLAGS", "engine-tests/metal-backend.air"),
        ("ENGINE_SANITIZER_CXXFLAGS", "sanitizers/operator-tuning-asan-ubsan"),
    )

    def make(self, *arguments):
        return subprocess.run(
            ("make", *arguments),
            cwd=build_identity.ROOT,
            capture_output=True,
            text=True,
        )

    def build(self, *arguments):
        result = self.make("-j4", *arguments)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def query(self, *arguments):
        result = self.make("-q", *arguments)
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return result.returncode

    def snapshot(self, build):
        return {
            path.relative_to(build).as_posix(): (
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in build.rglob("*")
            if path.is_file()
        }

    def recording_compiler(self, directory):
        log = directory / "calls.jsonl"
        compiler = directory / "compiler.py"
        compiler.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "output = Path(args[1] if args[0] == 'rcs' "
            "else args[args.index('-o') + 1])\n"
            "output.write_text(json.dumps(args))\n"
            "os.utime(output, (2000000000, 2000000000))\n"
            f"with Path({str(log)!r}).open('a') as stream:\n"
            "    stream.write(str(output) + '\\n')\n"
        )
        return compiler, log

    def test_actual_flag_changes_and_reversions(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build"
            build_arg = f"BUILD={build}"
            targets = [str(build / output) for _, output in self.FLAG_SETS]
            database = self.make("-np", build_arg, targets[0]).stdout
            self.build(build_arg, *targets)
            for variable, output in self.FLAG_SETS:
                with self.subTest(flags=variable):
                    target = str(build / output)
                    default = re.search(
                        rf"^{variable} := (.*)$", database, re.MULTILINE
                    ).group(1)
                    changed = f"{variable}={default} -O0"
                    before = self.snapshot(build)
                    self.assertEqual(self.query(build_arg, target), 0)
                    self.assertEqual(self.query(build_arg, changed, target), 1)
                    dry = self.make("-n", build_arg, changed, target)
                    self.assertEqual(dry.returncode, 0, dry.stderr)
                    self.assertEqual(self.snapshot(build), before)
                    original_config = build_config.record_path(target).read_text()
                    switched = self.build(build_arg, changed, target)
                    self.assertIn(" record --output ", switched.stdout)
                    self.assertNotEqual(
                        build_config.record_path(target).read_text(), original_config
                    )
                    self.assertEqual(self.query(build_arg, changed, target), 0)
                    self.assertEqual(self.query(build_arg, target), 1)
                    reverted = self.build(build_arg, target)
                    self.assertIn(" record --output ", reverted.stdout)
                    self.assertEqual(
                        build_config.record_path(target).read_text(), original_config
                    )
                    self.assertEqual(self.query(build_arg, target), 0)

    def test_partial_builds_and_links_with_one_output_timestamp(self):
        # Exercise the actual Make graph without compiling the whole engine.
        # This tool records invocations and pins every output to the same time,
        # making configuration tests independent of the host's timestamp tick.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            build = directory / "build"
            compiler, log = self.recording_compiler(directory)
            options = [f"BUILD={build}"] + [
                f"{name}={sys.executable} {compiler}"
                for name in ("CXX", "METAL", "METALLIB", "AR")
            ]

            def calls():
                return log.read_text().splitlines()

            outputs = (
                "splash",
                "engine-tests/vision-encoder",
                "engine-tests/attention-sweep",
                "engine-tests/metal-backend.metallib",
                "sanitizers/operator-tuning-asan-ubsan",
                "engine-tests/backend-benchmark",
            )
            targets = [str(build / name) for name in outputs]
            self.build(*options, *targets)
            baseline_calls = calls()
            self.build(*options, *targets)
            self.assertEqual(calls(), baseline_calls)

            vision, sweep = targets[1:3]
            changed = "ENGINE_TEST_CXXFLAGS=-O0"
            self.build(*options, changed, vision)
            self.assertEqual(self.query(*options, changed, vision), 0)
            self.assertEqual(self.query(*options, changed, sweep), 1)
            self.assertEqual(self.query(*options, sweep), 0)
            self.build(*options, changed, sweep)
            self.build(*options, vision)
            # Restoring only vision must not mark the still-O0 sweep current.
            self.assertEqual(self.query(*options, sweep), 1)
            self.assertEqual(self.query(*options, changed, sweep), 0)
            self.build(*options, sweep)

            for flags, linked in (
                ("ENGINE_LINKFLAGS=-Wl,-dead_strip", targets[0]),
                ("TEST_METALFLAGS=-O0", targets[3]),
                ("ENGINE_SANITIZER_CXXFLAGS=-O0", targets[4]),
            ):
                with self.subTest(flags=flags):
                    for configuration in ([flags], []):
                        start = len(calls())
                        self.build(*options, *configuration, linked)
                        self.assertIn(linked, calls()[start:])
                        if linked == targets[0]:
                            self.assertIn(
                                str(build / "engine/libsplash.a"), calls()[start:]
                            )
                            self.assertIn(
                                str(build / "splash.metallib"), calls()[start:]
                            )
                        if linked == targets[3]:
                            self.assertIn(
                                str(build / "engine-tests/metal-backend.air"),
                                calls()[start:],
                            )
                        start = len(calls())
                        self.build(*options, *configuration, linked)
                        self.assertEqual(len(calls()), start)

    def test_metal_input_removal_and_restore_rebuild_only_affected_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "Makefile",
                "dev/Makefile",
                "dev/native.mk",
                "dev/tools/build_config.py",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(build_identity.ROOT / relative, destination)
            kernel_root = Path("runtime/metal/kernels")
            q8_sources = [
                kernel_root / phase / name
                for phase in ("prefill", "decode")
                for name in ("attention_q8.metal", "attention_q8_store.metal")
            ]
            removed_source = kernel_root / "shared/removed.metal"
            removed_header = kernel_root / "common/removed.h"
            for relative in (
                *q8_sources,
                removed_source,
                removed_header,
                Path("runtime/metal/abi/ExecutionGeometry.h"),
                Path("runtime/engine/Status.cpp"),
                Path("dev/tests/engine/q8_page_format_oracle.metal"),
                Path("dev/tests/engine/metal_backend_test.metal"),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("// fixture\n")

            build = root / "build"
            compiler, log = self.recording_compiler(root)
            options = ["-C", str(root), f"BUILD={build}"] + [
                f"{name}={sys.executable} {compiler}"
                for name in ("CXX", "METAL", "METALLIB", "AR")
            ]
            library = str(build / "splash.metallib")
            attention = str(build / "engine-tests/q8-attention.metallib")
            native = str(build / "engine/engine/Status.o")
            unrelated = str(build / "engine-tests/metal-backend.metallib")
            targets = (library, attention, native, unrelated)
            production_airs = {
                str(
                    build
                    / "metal"
                    / source.relative_to(kernel_root).with_suffix(".air")
                )
                for source in (*q8_sources, removed_source)
            }
            test_airs = {
                str(
                    build
                    / "engine-tests/kernels"
                    / source.relative_to(kernel_root).with_suffix(".air")
                )
                for source in q8_sources
            }

            def rebuild(*selected):
                before = log.read_text().splitlines() if log.exists() else []
                self.build(*options, *selected)
                return set(log.read_text().splitlines()[len(before) :])

            rebuild(*targets)
            self.assertEqual(rebuild(*targets), set())
            (root / removed_source).unlink()
            before = self.snapshot(build)
            self.assertEqual(self.query(*options, library), 1)
            self.assertEqual(self.query(*options, native, unrelated), 0)
            dry = self.make("-n", *options, *targets)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertEqual(self.snapshot(build), before)
            self.assertEqual(rebuild(*targets), {library})
            removed_air = str(build / "metal/shared/removed.air")
            self.assertNotIn(removed_air, json.loads(Path(library).read_text()))

            (root / removed_source).write_text("// fixture\n")
            self.assertEqual(rebuild(*targets), {library})
            self.assertIn(removed_air, json.loads(Path(library).read_text()))
            (root / removed_header).unlink()
            self.assertEqual(self.query(*options, library, attention), 1)
            self.assertEqual(rebuild(attention), test_airs | {attention})
            # Updating the shared test library must not mark the production
            # library current, even with identical output timestamps.
            self.assertEqual(self.query(*options, library), 1)
            self.assertEqual(rebuild(library), production_airs | {library})
            self.assertEqual(rebuild(native, unrelated), set())

            (root / removed_header).write_text("// fixture\n")
            self.assertEqual(rebuild(library), production_airs | {library})
            self.assertEqual(self.query(*options, attention), 1)
            self.assertEqual(rebuild(attention), test_airs | {attention})
            self.assertEqual(self.query(*options, *targets), 0)
            self.assertEqual(rebuild(*targets), set())

    def test_failed_compiler_invalidates_previous_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            command = [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
            ]
            self.assertEqual(build_config.run_configured(output, "A", command), 0)
            self.assertTrue(build_config.matches(output, "A"))
            self.assertEqual(
                build_config.run_configured(
                    output,
                    "A",
                    [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(output)!r}).write_text('partial'); raise SystemExit(1)",
                    ],
                ),
                1,
            )
            self.assertEqual(output.read_text(), "partial")
            self.assertFalse(build_config.matches(output, "A"))
            self.assertEqual(build_config.run_configured(output, "A", command), 0)
            self.assertTrue(build_config.matches(output, "A"))


if __name__ == "__main__":
    unittest.main()

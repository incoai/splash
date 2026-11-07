import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from dev.tests.server_fixtures import (
    empty_page_cache,
    full_render_limits,
    pdf_bytes,
    render_pdf,
)
from server import document_worker, documents
from server.errors import APIError

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"
# A variable another process of the user's started with, as the server did
# with SPLASH_API_KEY.
OTHERS = {"SPLASH_PROBE": "another process's secret"}
# What an exploited PDFium could try, each attempt True when it succeeds; then
# it starts another program, which prints nothing, or says it was refused.
PROBE = r"""
import ctypes, json, os, socket, sys

home, private, folder, port, beside, other = sys.argv[1:]
# OTHERS, as the environment holds it.
SECRET = "SPLASH_PROBE=another process's secret"
system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def attempt(action):
    try:
        action()
    except OSError:
        return False
    return True


def fork():
    if os.fork() == 0:
        os._exit(0)
    os.wait()


def service():
    port = ctypes.c_uint()
    if system.bootstrap_look_up(
        ctypes.c_uint.in_dll(system, "bootstrap_port"),
        b"com.apple.system.opendirectoryd.libinfo",
        ctypes.byref(port),
    ):
        raise OSError("bootstrap_look_up")


def environment(pid):
    # kern.procargs2 (CTL_KERN, KERN_PROCARGS2): a process's arguments and
    # the environment it started with.
    name = (ctypes.c_int * 3)(1, 49, pid)
    size = ctypes.c_size_t(1 << 16)
    found = ctypes.create_string_buffer(size.value)
    if system.sysctl(name, 3, found, ctypes.byref(size), None, 0):
        raise OSError("sysctl")
    if SECRET.encode() not in found.raw[: size.value]:
        raise OSError("not its environment")


print(json.dumps({
    "home": attempt(lambda: os.listdir(home)),
    "private": attempt(lambda: open(private).read()),
    "beside": attempt(lambda: open(beside).read()),
    "urandom": attempt(lambda: open("/dev/urandom", "rb").read(1)),
    "write": attempt(lambda: open(os.path.join(folder, "written"), "w").close()),
    "connect": attempt(
        lambda: socket.create_connection(("127.0.0.1", int(port)), 10).close()
    ),
    "fork": attempt(fork),
    "service": attempt(service),
    "others": attempt(lambda: environment(int(other))),
    "environment": dict(os.environ),
}), flush=True)
try:
    os.execv("/usr/bin/true", ["true"])
except OSError:
    print("exec refused")
"""


def run(arguments, payload=b"", sandboxed=True):
    """The exit status and output of the interpreter, run with `arguments` as
    render runs the worker, under the sandbox or not."""
    with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as output:
        source.write(payload)
        source.seek(0)
        with (
            nullcontext()
            if sandboxed
            else mock.patch.object(document_worker, "_sandbox", return_value=[])
        ):
            process = document_worker._start(arguments, source, output)
        try:
            process.wait(timeout=60)
        finally:
            document_worker._stop(process)
        output.seek(0)
        return process.returncode, output.read()


def worker(payload, sandboxed=True):
    """The worker's exit status and output for `payload`."""
    limits = json.dumps(asdict(full_render_limits()))
    return run(
        ["-m", "server.document_worker", limits, "30"], payload, sandboxed=sandboxed
    )


def parameters(command):
    """The parameters sandbox-exec's `command` gives its profile, by name."""
    return dict(
        argument[2:].split("=", 1) for argument in command if argument.startswith("-D")
    )


class DocumentWorkerTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(empty_page_cache())

    def sleeping_worker(self, children):
        start = subprocess.Popen

        def spawn(command, **kwargs):
            process = start(
                [sys.executable, "-c", "import time; time.sleep(10)"], **kwargs
            )
            children.append(process)
            self.addCleanup(document_worker._stop, process)
            return process

        return mock.patch.object(document_worker.subprocess, "Popen", spawn)

    def assert_released(self, children):
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertFalse(document_worker._workers)
        self.assertFalse(documents._cache)
        self.assertFalse(documents._render_lock.locked())
        self.assertIn("ALPHA 42", render_pdf()[0]["text"])

    def test_deadline_stops_worker_and_next_document_succeeds(self):
        children = []
        with self.sleeping_worker(children):
            with self.assertRaises(APIError) as caught:
                document_worker.render(pdf_bytes(), asdict(full_render_limits()), 0.1)
        self.assertEqual(caught.exception.status, 504)
        self.assertEqual(caught.exception.code, "request_timeout")
        self.assert_released(children)

    def test_worker_time_limit_without_a_deadline_is_a_request_error(self):
        children = []
        with (
            self.sleeping_worker(children),
            mock.patch.object(document_worker, "MAX_SECONDS", 0.1),
        ):
            with self.assertRaises(APIError) as caught:
                document_worker.render(
                    pdf_bytes(), asdict(full_render_limits()), math.inf
                )
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(
            caught.exception.message, "PDF processing exceeded the time limit"
        )
        self.assert_released(children)

    def test_memory_budget_stops_worker_without_caching_partial_output(self):
        children = []
        with (
            self.sleeping_worker(children),
            mock.patch.object(
                document_worker,
                "_memory_bytes",
                return_value=document_worker.MAX_MEMORY_BYTES + 1,
            ),
        ):
            with self.assertRaisesRegex(APIError, "memory limit"):
                render_pdf()
        self.assert_released(children)

    def test_failed_measurement_stops_worker(self):
        children = []
        with (
            self.sleeping_worker(children),
            mock.patch.object(document_worker, "_memory_bytes", side_effect=OSError),
        ):
            with self.assertRaises(APIError) as caught:
                render_pdf()
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(caught.exception.code, "document_unavailable")
        self.assert_released(children)

    def test_unreadable_result_is_a_server_error(self):
        start = subprocess.Popen

        def spawn(command, **kwargs):
            return start([sys.executable, "-c", "print('not JSON')"], **kwargs)

        with mock.patch.object(document_worker.subprocess, "Popen", spawn):
            with self.assertRaises(APIError) as caught:
                render_pdf()
        self.assertEqual(
            (caught.exception.status, caught.exception.code),
            (500, "document_worker_failed"),
        )
        self.assertFalse(document_worker._workers)
        self.assertFalse(documents._cache)

    def test_spawn_failure_is_transient_and_leaves_no_worker(self):
        with mock.patch.object(
            document_worker.subprocess, "Popen", side_effect=OSError
        ):
            with self.assertRaises(APIError) as caught:
                render_pdf()
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(caught.exception.code, "document_unavailable")
        self.assertFalse(document_worker._workers)
        self.assertFalse(documents._cache)


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(empty_page_cache())

    def test_pages_are_byte_identical_outside_the_sandbox(self):
        payloads = (
            pdf_bytes(pages=3),
            # A font PDFium looks up among the system's, and text beyond ASCII.
            pdf_bytes(r"caf\351 na\357ve \200 50", font="Georgia"),
            (FIXTURES / "owner-password.pdf").read_bytes(),
            # Errors too.
            (FIXTURES / "open-password.pdf").read_bytes(),
            b"%PDF-1.4 not a document",
        )
        for payload in payloads:
            with self.subTest(payload=payload[:40]):
                sandboxed = worker(payload)
                self.assertEqual(sandboxed[0], 0)
                self.assertEqual(sandboxed, worker(payload, sandboxed=False))
        text = json.loads(worker(payloads[1])[1])["pages"][0]["text"]
        self.assertEqual(text, "PDF page 1:\ncafé naïve € 50 page 1\n")

    def test_worker_reaches_nothing_rendering_does_without(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            socket.create_server(("127.0.0.1", 0)) as listener,
            mock.patch.dict(os.environ, {"SPLASH_API_KEY": "server secret"}),
        ):
            other = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"], env=OTHERS
            )
            self.addCleanup(document_worker._stop, other)
            private = Path(folder, "private")
            private.write_text("private")
            arguments = [
                "-c",
                PROBE,
                str(Path.home()),
                str(private),
                folder,
                str(listener.getsockname()[1]),
                # The folder holding server/, beside the server package.
                str(document_worker.ROOT / "docs" / "api.md"),
                str(other.pid),
            ]
            outputs = {
                sandboxed: run(arguments, sandboxed=sandboxed)
                for sandboxed in (False, True)
            }
        attempts = (
            "home",
            "private",
            "beside",
            "urandom",
            "write",
            "connect",
            "fork",
            "service",
            "others",
        )
        for sandboxed, (status, output) in outputs.items():
            self.assertEqual(status, 0)
            lines = output.decode().splitlines()
            result = json.loads(lines[0])
            # Outside the sandbox each attempt succeeds, so that inside it each
            # fails for the sandbox alone.
            self.assertEqual(
                {name: result[name] for name in attempts},
                dict.fromkeys(attempts, not sandboxed),
            )
            self.assertEqual(lines[1:], ["exec refused"] if sandboxed else [])
            # No variable of the server's: Python sets LC_CTYPE itself when it
            # can (PEP 538), and CoreFoundation __CF_USER_TEXT_ENCODING.
            environment = result["environment"]
            self.assertLessEqual(
                set(environment), {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}
            )
            self.assertNotIn("server secret", environment.values())

    def test_links_are_those_resolving_a_path_follows(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(os.path.realpath(folder))
            (base / "real" / "bin").mkdir(parents=True)
            (base / "real" / "bin" / "tool").touch()
            (base / "real" / "bin" / "next").symlink_to("tool")
            (base / "real" / "bin" / "run").symlink_to("../bin/next")
            (base / "alias").symlink_to("real")
            (base / "absolute").symlink_to(base / "alias")
            (base / "loop").symlink_to("loop")
            self.assertEqual(
                document_worker._links(base / "absolute" / "bin" / "run"),
                [
                    base / "absolute",
                    base / "alias",
                    base / "real" / "bin" / "run",
                    base / "real" / "bin" / "next",
                ],
            )
            self.assertEqual(document_worker._links(base / "real" / "bin" / "tool"), [])
            self.assertEqual(
                len(document_worker._links(base / "loop")), document_worker.MAX_LINKS
            )

    def test_each_start_follows_an_upgrade_of_the_interpreter(self):
        # An upgrade under a running server repoints a link on the way to the
        # interpreter (Homebrew's opt, uv's minor version), and the kernel
        # checks the next worker by the new real paths.
        with tempfile.TemporaryDirectory() as folder:
            base = Path(os.path.realpath(folder))
            old, new = base / "3.13.1", base / "3.13.2"
            opt = base / "opt"
            commands = []
            for tree in (old, new):
                (tree / "bin").mkdir(parents=True)
                (tree / "bin" / "python3.13").touch()
                opt.unlink(missing_ok=True)
                opt.symlink_to(tree)
                with mock.patch.multiple(
                    sys,
                    executable=str(opt / "bin" / "python3.13"),
                    base_prefix=str(opt),
                ):
                    commands.append(document_worker._sandbox())
        before, after = (set(parameters(command).values()) for command in commands)
        self.assertLessEqual({str(old / "bin" / "python3.13"), str(old)}, before)
        self.assertLessEqual({str(new / "bin" / "python3.13"), str(new)}, after)
        self.assertEqual([value for value in after if str(old) in value], [])

    def test_worker_starts_through_a_linked_folder(self):
        # The same interpreter, reached through a link to its prefix: a venv's
        # own, whose interpreter is a link, or the installation's.
        venv = sys.prefix != sys.base_prefix
        prefix = sys.prefix if venv else os.path.realpath(sys.prefix)
        executable = sys.executable if venv else os.path.realpath(sys.executable)
        with tempfile.TemporaryDirectory() as folder:
            link = Path(folder, "runtime")
            link.symlink_to(prefix)
            linked = link / os.path.relpath(executable, prefix)
            with mock.patch.object(sys, "executable", str(linked)):
                started = worker(pdf_bytes())
                with mock.patch.object(document_worker, "_links", return_value=[]):
                    refused = worker(pdf_bytes())
        self.assertEqual(started, worker(pdf_bytes(), sandboxed=False))
        # Without the links, sandbox-exec cannot start it.
        self.assertEqual(refused, (document_worker.SANDBOX_FAILED, b""))

    def test_a_framework_interpreter_starts_its_app(self):
        # Homebrew's and python.org's interpreter is a launcher that resolves
        # its folder and starts the framework's Python.app, as CI's does;
        # Homebrew's prefix is a link (/opt/homebrew/opt), and the kernel
        # checks what starts by its real path.
        with tempfile.TemporaryDirectory() as folder:
            real = Path(os.path.realpath(folder), "Cellar", "Versions", "3.13")
            app = real / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
            launcher = real / "bin" / "python3.13"
            app.parent.mkdir(parents=True)
            launcher.parent.mkdir()
            app.touch()
            launcher.touch()
            Path(folder, "opt").symlink_to(real.parent.parent)
            base = Path(folder, "opt", "Versions", "3.13")
            with mock.patch.multiple(
                sys, executable=str(base / "bin" / "python3.13"), base_prefix=str(base)
            ):
                named = parameters(document_worker._sandbox())
        started = {value for name, value in named.items() if name.startswith("PYTHON")}
        self.assertEqual(started, {str(launcher), str(app)})
        # The folders by their real paths, the linked one where its link leads.
        folders = {value for name, value in named.items() if name.startswith("FOLDER")}
        self.assertLessEqual({str(launcher.parent), str(real), "/"}, folders)
        self.assertNotIn(str(base / "bin"), folders)

    def test_a_sandbox_that_cannot_start_is_unavailable(self):
        refusing = [
            document_worker.SANDBOX_EXEC,
            "-p",
            "(version 1)(deny default)",
            "--",
        ]
        with mock.patch.object(document_worker, "_sandbox", return_value=refusing):
            with self.assertRaises(APIError) as caught:
                render_pdf()
        self.assertEqual(
            (caught.exception.status, caught.exception.code),
            (503, "document_unavailable"),
        )
        self.assertFalse(document_worker._workers)


if __name__ == "__main__":
    unittest.main()

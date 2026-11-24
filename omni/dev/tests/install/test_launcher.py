import fcntl
import io
import json
import math
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import (
    COMMIT,
    TTS_MODEL,
    FakeHub,
    audio_repo,
    cached_model,
    no_hub,
)
from dev.tests.server_fixtures import keep_stop_signals, server_arguments
from install import families, launcher
from server import audio_server as api
from server import serve_options

MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-4bit"


def with_stop_held(command, number):
    """The command, started with a stop signal already sent and held, as the
    launcher holds one sent while it starts a program."""
    # A test run as a background job inherits SIGINT ignored, which would
    # discard the signal instead of holding it.
    return [
        sys.executable,
        "-c",
        "import os, signal, sys; "
        f"signal.signal({int(number)}, signal.SIG_DFL); "
        f"signal.pthread_sigmask(signal.SIG_BLOCK, ({int(number)},)); "
        f"os.kill(os.getpid(), {int(number)}); "
        "os.execv(sys.argv[1], sys.argv[1:])",
        *command,
    ]


def selection(models_root, **options):
    return launcher.model_artifacts.Selection.of(models_root, MODEL_ID, **options)


MODEL_IDS = (
    *families.SUPPORTED_MODELS,
    # The launcher takes any repository ID; the installer refuses one of no
    # supported family.
    "community/custom-model",
)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        # The launcher's defaults ignore the caller's Splash settings.
        self.enterContext(mock.patch.dict(os.environ))
        for name in ("SPLASH_PORT", "SPLASH_API_KEY"):
            os.environ.pop(name, None)
        keep_stop_signals(self)

    def test_serve_requires_exact_repository_id_before_build(self):
        for arguments in (
            ["serve"],
            ["serve", "--model", "qwen3.6-35b-a3b"],
            ["serve", "--model", "Qwen3.6-35B-A3B"],
            ["serve", "--model", "https://huggingface.co/community/model"],
            ["serve", "--model", "community/../model"],
            # A GGUF variant, which the language engine took.
            ["serve", "--model", "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"],
            ["serve", "--model", "owner/repo:"],
        ):
            with (
                self.subTest(arguments=arguments),
                mock.patch.object(launcher, "_ensure_installed") as install,
                mock.patch("sys.stderr", io.StringIO()) as error,
                self.assertRaises(SystemExit) as failed,
            ):
                launcher.main(arguments)
            self.assertEqual(failed.exception.code, 2)
            self.assertIn("--model", error.getvalue())
            install.assert_not_called()
        for model in MODEL_IDS:
            self.assertEqual(
                launcher.parse_args(["serve", "--model", model]).model, model
            )

    def test_only_serve_is_public(self):
        for command in (
            "start",
            "stop",
            "status",
            "logs",
            "doctor",
            "model",
            "models",
            "uninstall",
            # The coding clients the language server connected.
            "claude",
            "codex",
            "opencode",
            "hermes",
            "pi",
        ):
            with self.subTest(command=command), mock.patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.parse_args([command])
        args = launcher.parse_args(
            [
                "serve",
                "--model",
                MODEL_ID,
                "--max-memory",
                "28G",
                "--lanes",
                "4",
            ]
        )
        self.assertEqual(args.model, MODEL_ID)
        self.assertEqual(args.max_memory, 28 * 1024**3)
        self.assertEqual(args.lanes, 4)
        # The language server's options, and arguments after --, which the
        # coding clients took (--draft-model returned for transcription).
        for removed in (
            ("--max-context", "100K"),
            ("--language-only",),
            ("--no-webui",),
            ("--", "--port", "4096"),
        ):
            with (
                self.subTest(option=removed[0]),
                mock.patch("sys.stderr", io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                launcher.parse_args(["serve", "--model", MODEL_ID, *removed])

    def test_host_controls_probe_and_server_independently_of_allowed_host(self):
        for host in (None, "0.0.0.0", "192.0.2.10", "localhost"):
            with (
                self.subTest(host=host),
                tempfile.TemporaryDirectory() as temporary,
                mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
                mock.patch.object(launcher.socket, "socket") as factory,
                mock.patch.object(launcher, "_ensure_installed"),
                mock.patch.object(launcher.os, "execve") as execute,
            ):
                arguments = [
                    "serve",
                    "--model",
                    MODEL_ID,
                    "--port",
                    "9123",
                    "--allowed-host",
                    "proxy.example",
                ]
                if host is not None:
                    arguments.extend(["--host", host])
                launcher.main(arguments)
                expected = host or "127.0.0.1"
                factory.return_value.__enter__.return_value.bind.assert_called_once_with(
                    (expected, 9123)
                )
                served = api.parse_args(server_arguments(execute.call_args.args[1]))
                self.assertEqual(served.host, expected)
                self.assertEqual(served.allowed_host, ["proxy.example"])

    def test_invalid_bind_address_fails_before_model_work(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
            mock.patch.object(launcher, "_ensure_installed") as install,
            mock.patch.object(launcher.os, "execve") as execute,
            mock.patch("sys.stderr", io.StringIO()) as error,
        ):
            self.assertEqual(
                launcher.main(
                    [
                        "serve",
                        "--model",
                        MODEL_ID,
                        "--host",
                        "http://127.0.0.1",
                    ]
                ),
                1,
            )
            self.assertIn("cannot bind http://127.0.0.1:", error.getvalue())
            install.assert_not_called()
            execute.assert_not_called()

    def test_foreground_exec_preserves_terminal_and_holds_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            lock_path = runtime / "serve-8000.lock"
            lock_path.write_text(
                json.dumps(
                    {"pid": os.getpid(), "model": "stale-model" * 100, "port": 65535}
                )
            )
            original_inode = lock_path.stat().st_ino
            owner = {
                "pid": os.getpid(),
                "model": MODEL_ID,
                "port": serve_options.DEFAULT_PORT,
            }

            def check_install(chosen, draft):
                self.assertEqual(json.loads(lock_path.read_text()), owner)

            def check_exec(binary, argv, environment):
                self.assertEqual(binary, str(launcher.paths.PYTHON))
                # test_serve_options.py checks every shared option; these
                # are the launcher's own and the shared options it was given.
                with mock.patch.dict(os.environ, environment):
                    served = api.parse_args(server_arguments(argv))
                self.assertEqual(
                    (served.binary, served.model, served.port),
                    (str(launcher.paths.BINARY), MODEL_ID, serve_options.DEFAULT_PORT),
                )
                self.assertEqual(
                    (served.lanes, served.max_memory, served.idle_release),
                    (4, 28 * 1024**3, math.inf),
                )
                self.assertEqual(
                    served.allowed_origin, [("tauri", "localhost", None), "*"]
                )
                self.assertNotIn("start_new_session", environment)
                self.assertFalse(
                    any("test-server-key" in argument for argument in argv)
                )
                self.assertEqual(served.api_key, "test-server-key")
                self.assertEqual(json.loads(lock_path.read_text()), owner)
                self.assertEqual(lock_path.stat().st_ino, original_inode)
                with (runtime / "serve.lock").open("a+") as upgrade:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(upgrade, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(upgrade, fcntl.LOCK_SH | fcntl.LOCK_NB)
                with (runtime / "serve-8000.lock").open("a+") as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

            with (
                mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                mock.patch.object(launcher.socket, "socket"),
                mock.patch.object(
                    launcher, "_ensure_installed", side_effect=check_install
                ) as install,
                mock.patch.object(
                    launcher.os, "execve", side_effect=check_exec
                ) as execute,
                mock.patch("sys.stdout", io.StringIO()),
            ):
                launcher.main(
                    [
                        "serve",
                        "--model",
                        MODEL_ID,
                        "--lanes",
                        "4",
                        "--api-key",
                        "test-server-key",
                        "--idle-release",
                        "off",
                        "--max-memory",
                        "28G",
                        "--allowed-origin",
                        "tauri://localhost",
                        "--allowed-origin",
                        "*",
                    ]
                )
            install.assert_called_once_with(
                selection(launcher.paths.MODELS), draft=True
            )
            execute.assert_called_once()
            self.assertEqual(
                {p.name for p in runtime.iterdir()}, {"serve.lock", "serve-8000.lock"}
            )
            with (runtime / "serve-8000.lock").open("a+") as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_upgrade_lock_blocks_every_port_before_model_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            with (runtime / "serve.lock").open("a+") as installation:
                installation.write('{"pid":123,"model":"stale/model","port":8000}')
                installation.flush()
                fcntl.flock(installation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch.object(launcher, "_ensure_installed") as install,
                    mock.patch.object(launcher.os, "execve") as execute,
                    mock.patch("sys.stderr", io.StringIO()) as error,
                ):
                    for port in (8000, 9123):
                        self.assertEqual(
                            launcher.main(
                                ["serve", "--model", MODEL_ID, "--port", str(port)]
                            ),
                            1,
                        )
                self.assertIn(
                    "Splash is being upgraded; wait for the upgrade to finish",
                    error.getvalue(),
                )
                self.assertNotIn("stale/model", error.getvalue())
                install.assert_not_called()
                execute.assert_not_called()

    def test_duplicate_serve_reports_existing_owner_without_model_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            lock_path = runtime / "serve-8000.lock"
            owner = {
                "pid": os.getpid(),
                "model": "incoai/Qwen3.8-27B-Splash",
                "port": 8000,
            }
            with lock_path.open("w+") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                json.dump(owner, held)
                held.flush()
                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch.object(launcher.socket, "socket") as probe,
                    mock.patch.object(launcher, "_ensure_installed") as install,
                    mock.patch.object(launcher.os, "execve") as execute,
                    mock.patch("sys.stderr", io.StringIO()) as error,
                ):
                    self.assertEqual(launcher.main(["serve", "--model", MODEL_ID]), 1)
                self.assertIn(f"PID {os.getpid()}", error.getvalue())
                self.assertIn("model incoai/Qwen3.8-27B-Splash", error.getvalue())
                self.assertIn("port 8000", error.getvalue())
                self.assertEqual(json.loads(lock_path.read_text()), owner)
                probe.assert_not_called()
                install.assert_not_called()
                execute.assert_not_called()

    def test_duplicate_serve_tolerates_missing_or_invalid_metadata(self):
        for content in (
            b"",
            b"old lock file",
            b"{",
            b"[]",
            b"null",
            b"\xff",
            b'{"pid": 123}',
            b'{"pid": true, "model": "old", "port": 8000}',
            b'{"pid": 123, "model": "old", "port": "8000"}',
            b'{"pid": 123, "model": "old", "port": 0}',
        ):
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as temporary,
            ):
                runtime = Path(temporary)
                lock_path = runtime / "serve-8000.lock"
                lock_path.write_bytes(content)
                with lock_path.open("a+") as held:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with (
                        mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                        mock.patch.object(launcher, "_ensure_installed") as install,
                        mock.patch("sys.stderr", io.StringIO()) as error,
                    ):
                        self.assertEqual(
                            launcher.main(["serve", "--model", MODEL_ID]), 1
                        )
                    self.assertEqual(
                        error.getvalue(),
                        "error: Splash is already serving; stop it with Ctrl+C first\n",
                    )
                    self.assertEqual(lock_path.read_bytes(), content)
                    install.assert_not_called()

    def test_busy_port_and_duplicate_serve_fail_before_model_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
                mock.patch.object(launcher.socket, "socket") as socket,
                mock.patch.object(launcher, "_ensure_installed") as install,
                mock.patch("sys.stderr", io.StringIO()),
            ):
                socket.return_value.__enter__.return_value.bind.side_effect = OSError(
                    "busy"
                )
                self.assertEqual(launcher.main(["serve", "--model", MODEL_ID]), 1)
                with (Path(temporary) / "serve-8000.lock").open("a+") as held:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.assertEqual(launcher.main(["serve", "--model", MODEL_ID]), 1)
                install.assert_not_called()

    def test_real_port_probe_allows_time_wait_but_rejects_live_listener(self):
        for host, closed in (
            ("127.0.0.1", False),
            ("127.0.0.1", True),
            ("0.0.0.0", False),
            ("0.0.0.0", True),
        ):
            with (
                self.subTest(host=host, closed=closed),
                tempfile.TemporaryDirectory() as temporary,
                socket.socket() as listener,
            ):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.settimeout(2)
                listener.bind((host, 0))
                port = listener.getsockname()[1]
                listener.listen()
                if closed:
                    with socket.create_connection(
                        ("127.0.0.1", port), timeout=2
                    ) as client:
                        connection, _ = listener.accept()
                        connection.close()
                        self.assertEqual(client.recv(1), b"")
                    listener.close()
                # Exercise the production probe against a real ephemeral port,
                # without interfering with a user's server on port 8000.
                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
                    mock.patch.object(launcher, "_ensure_installed") as install,
                    mock.patch.object(launcher.os, "execve") as execute,
                    mock.patch("sys.stderr", io.StringIO()),
                ):
                    result = launcher.main(
                        [
                            "serve",
                            "--model",
                            MODEL_ID,
                            "--host",
                            host,
                            "--port",
                            str(port),
                        ]
                    )
                if closed:
                    self.assertIsNone(result)
                    install.assert_called_once()
                    execute.assert_called_once()
                else:
                    self.assertEqual(result, 1)
                    install.assert_not_called()
                    execute.assert_not_called()

    def test_real_port_probe_rejects_a_listener_on_another_address_of_the_port(self):
        # SO_REUSEADDR lets 127.0.0.1 bind beside another process's wildcard
        # listener, IPv4 or dual-stack (python -m http.server), and take its
        # loopback clients; and 0.0.0.0 bind beside a loopback listener,
        # which keeps the clients the launcher connects to 127.0.0.1.
        for listen, host in (
            ("0.0.0.0", "127.0.0.1"),
            ("::", "127.0.0.1"),
            ("127.0.0.1", "0.0.0.0"),
        ):
            family = socket.AF_INET6 if ":" in listen else socket.AF_INET
            with (
                self.subTest(listen=listen, host=host),
                tempfile.TemporaryDirectory() as temporary,
                socket.socket(family) as listener,
            ):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                listener.bind((listen, 0))
                port = listener.getsockname()[1]
                listener.listen()
                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
                    mock.patch.object(launcher, "_ensure_installed") as install,
                    mock.patch.object(launcher.os, "execve") as execute,
                    mock.patch("sys.stderr", io.StringIO()) as error,
                ):
                    result = launcher.main(
                        ["serve", "--model", MODEL_ID, "--host", host]
                        + ["--port", str(port)]
                    )
                self.assertEqual(result, 1)
                self.assertIn(f"cannot bind {host}:{port}: ", error.getvalue())
                install.assert_not_called()
                execute.assert_not_called()

    def test_port_selection_validates_environment_and_explicit_override(self):
        with mock.patch.dict(os.environ, {"SPLASH_PORT": "8123"}):
            self.assertEqual(
                launcher.parse_args(["serve", "--model", MODEL_ID]).port, 8123
            )
            self.assertEqual(
                launcher.parse_args(
                    ["serve", "--model", MODEL_ID, "--port", "9123"]
                ).port,
                9123,
            )
        for value in ("", "0", "-1", "65536", "invalid", "1.5"):
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {"SPLASH_PORT": value}),
            ):
                for arguments in (["serve", "--model", MODEL_ID],):
                    with (
                        mock.patch("sys.stderr", io.StringIO()) as error,
                        self.assertRaises(SystemExit),
                    ):
                        launcher.parse_args(arguments)
                    self.assertIn("SPLASH_PORT: ", error.getvalue())
                    self.assertEqual(
                        launcher.parse_args([*arguments, "--port", "9123"]).port,
                        9123,
                    )
                    with (
                        mock.patch("sys.stderr", io.StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        launcher.parse_args([*arguments, "--port", value])

    def test_serve_help_names_the_default_port(self):
        with (
            mock.patch.dict(os.environ),
            mock.patch.object(serve_options, "DEFAULT_PORT", 8123),
            mock.patch("sys.stdout", io.StringIO()) as output,
        ):
            os.environ.pop("SPLASH_PORT", None)
            with self.assertRaises(SystemExit):
                launcher.parse_args(["serve", "--help"])
        self.assertIn("SPLASH_PORT or 8123", " ".join(output.getvalue().split()))

    def test_distinct_ports_have_independent_locks_and_same_port_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.close()
            with socket.socket() as second:
                second.bind(("127.0.0.1", 0))
                other_port = second.getsockname()[1]
            self.assertNotEqual(port, other_port)
            runtime = Path(temporary)
            seen = []

            def execute(binary, argv, environment):
                selected = int(argv[argv.index("--port") + 1])
                seen.append(selected)
                lock_path = runtime / f"serve-{selected}.lock"
                self.assertEqual(json.loads(lock_path.read_text())["port"], selected)
                if selected == port:
                    with socket.socket() as active:
                        active.bind(("127.0.0.1", port))
                        active.listen()
                        self.assertEqual(
                            launcher.main(
                                ["serve", "--model", MODEL_ID, "--port", str(port)]
                            ),
                            1,
                        )
                        self.assertIsNone(
                            launcher.main(
                                [
                                    "serve",
                                    "--model",
                                    MODEL_ID,
                                    "--port",
                                    str(other_port),
                                ]
                            )
                        )

            with (
                mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                mock.patch.object(launcher, "_ensure_installed") as install,
                mock.patch.object(launcher.os, "execve", side_effect=execute),
                mock.patch("sys.stderr", io.StringIO()) as error,
            ):
                self.assertIsNone(
                    launcher.main(["serve", "--model", MODEL_ID, "--port", str(port)])
                )
                for selected in (port, other_port):
                    with (runtime / f"serve-{selected}.lock").open("a+") as released:
                        fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(seen, [port, other_port])
            self.assertEqual(install.call_count, 2)
            self.assertIn(f"port {port}", error.getvalue())

    def test_exec_keeps_installation_locked_until_every_server_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "server").mkdir()
            (root / "server/__init__.py").write_text("")
            (root / "server/audio_server.py").write_text(
                "import sys\nprint('ready', flush=True)\nsys.stdin.read()\n"
            )
            runtime = root / "runtime"
            script = (
                "import sys\nfrom pathlib import Path\nfrom install import launcher\n"
                "launcher.ROOT = Path(sys.argv[1])\n"
                "launcher.RUNTIME_DIR = launcher.ROOT / 'runtime'\n"
                "launcher.paths.PYTHON = Path(sys.executable)\n"
                "launcher._ensure_installed = lambda selection, draft: None\n"
                "launcher.model_artifacts.selection_link = lambda *a, **k: launcher.ROOT\n"
                "launcher.main(['serve', '--model', 'test/model', '--port', sys.argv[2]])\n"
            )
            processes = []
            ports = set()
            try:
                while len(ports) < 2:
                    with socket.socket() as probe:
                        probe.bind(("127.0.0.1", 0))
                        port = probe.getsockname()[1]
                    if port in ports:
                        continue
                    ports.add(port)
                    process = subprocess.Popen(
                        [sys.executable, "-c", script, str(root), str(port)],
                        cwd=Path(launcher.__file__).resolve().parents[1],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    processes.append(process)
                    self.assertTrue(select.select([process.stdout], [], [], 10)[0])
                    self.assertEqual(process.stdout.readline(), "ready\n")
                    with (runtime / f"serve-{port}.lock").open("a+") as lock:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (runtime / "serve.lock").open("a+") as installation:
                    for process in processes:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(installation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        _, error = process.communicate(input="", timeout=10)
                        self.assertEqual(process.returncode, 0, error)
                    fcntl.flock(installation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for port in ports:
                    with (runtime / f"serve-{port}.lock").open("a+") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=10)

    def test_source_script_runs_its_checkout_launcher_through_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkout = root / "checkout"
            (checkout / "install").mkdir(parents=True)
            (checkout / "install/launcher.py").write_text(
                "import sys\nprint(__file__, *sys.argv[1:])\n"
            )
            (checkout / ".venv/bin").mkdir(parents=True)
            (checkout / ".venv/bin/python").symlink_to(sys.executable)
            script = checkout / "splash-omni"
            script.write_bytes((launcher.ROOT / "splash-omni").read_bytes())
            script.chmod(0o755)
            # A relative link into the checkout, reached through an absolute one.
            (root / "bin").mkdir()
            (root / "bin/splash-omni").symlink_to("../checkout/splash-omni")
            (root / "path").mkdir()
            (root / "path/splash-omni").symlink_to(root / "bin/splash-omni")
            # The relative link through a linked directory: its .. is physical.
            (root / "path/linked").symlink_to("../bin")
            for command in (
                script,
                root / "bin/splash-omni",
                root / "path/splash-omni",
                root / "path/linked/splash-omni",
            ):
                with self.subTest(command=command):
                    result = subprocess.run(
                        [command, "serve", "--help"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        result.stdout,
                        f"{checkout / 'install/launcher.py'} serve --help\n",
                    )

    def test_source_build_lock_covers_make_and_releases_on_failure(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as temporary, self.subTest(fail=fail):
                runtime = Path(temporary)
                calls = []

                def run(command, **options):
                    calls.append(command)
                    with (runtime / "build.lock").open("a+") as probe:
                        if command[0] == "make":
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            (descriptor,) = options["pass_fds"]
                            self.assertEqual(
                                os.fstat(descriptor).st_ino,
                                os.fstat(probe.fileno()).st_ino,
                            )
                        else:
                            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return subprocess.CompletedProcess(command, int(fail))

                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch.object(launcher.subprocess, "run", side_effect=run),
                    mock.patch.object(
                        launcher,
                        "_run_held",
                        side_effect=lambda command, **options: run(command).returncode,
                    ),
                ):
                    if fail:
                        with self.assertRaisesRegex(
                            launcher.LauncherError, "source build failed"
                        ):
                            launcher._ensure_installed(selection(runtime))
                    else:
                        launcher._ensure_installed(selection(runtime))
                self.assertEqual(len(calls), 1 if fail else 4)
                if not fail:
                    # The device check runs on the built binary, unlocked.
                    self.assertEqual(
                        calls[2], [str(launcher.paths.BINARY), "device-check"]
                    )
                with (runtime / "build.lock").open("a+") as probe:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_source_build_says_when_it_waits_for_another(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            with (runtime / "build.lock").open("a+") as other:
                fcntl.flock(other, fcntl.LOCK_EX)
                # The other build ends once the launcher says it waits, or
                # after ten seconds: one that waits without a word fails
                # instead of hanging.
                ended = threading.Timer(10, fcntl.flock, (other, fcntl.LOCK_UN))
                ended.start()
                self.addCleanup(ended.cancel)

                class Output(io.StringIO):
                    def write(self, text):
                        fcntl.flock(other, fcntl.LOCK_UN)
                        return super().write(text)

                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch.object(
                        launcher.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess([], 0),
                    ) as run,
                    mock.patch.object(launcher, "_run_held", return_value=0),
                    mock.patch("sys.stdout", Output()) as output,
                ):
                    launcher._ensure_installed(selection(runtime))
            self.assertEqual(
                output.getvalue(), "Another Splash build is running; waiting...\n"
            )
            self.assertEqual(run.call_args_list[0].args[0][0], "make")

    def test_source_selection_reaches_installation_and_served_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            options = {"revision": "v2"}
            with (
                mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                mock.patch.object(launcher.socket, "socket"),
                mock.patch.object(launcher, "_ensure_installed") as install,
                mock.patch.object(
                    launcher.model_artifacts,
                    "selection_link",
                    return_value=runtime / "selected",
                ) as root,
                mock.patch.object(launcher.os, "execve") as execute,
            ):
                launcher.main(
                    [
                        "serve",
                        "--model",
                        MODEL_ID,
                        "--revision",
                        "v2",
                    ]
                )
            (chosen,) = install.call_args.args
            self.assertEqual(
                (chosen.model, chosen.link),
                (MODEL_ID, runtime / "selected"),
            )
            root.assert_called_once_with(chosen.models_root, MODEL_ID, **options)
            argv = execute.call_args.args[1]
            self.assertEqual(server_arguments(argv)[0], str(runtime / "selected"))

            with (
                mock.patch.object(
                    launcher.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0),
                ),
                mock.patch.object(launcher, "_run_held", return_value=0) as run,
                mock.patch.object(launcher, "_build_from_source"),
            ):
                launcher._ensure_installed(chosen)
            command = run.call_args.args[0]
            self.assertEqual(run.call_args.kwargs["cwd"], launcher.ROOT)
            parsed = launcher.model_artifacts.parse_args(command[2:])
            self.assertEqual(
                (parsed.command, parsed.model, parsed.revision),
                ("prepare", MODEL_ID, "v2"),
            )
            self.assertFalse(parsed.no_draft_model)

    def test_the_installer_fetches_the_default_draft_unless_told_otherwise(self):
        # The installer downloads the default draft beside the model unless
        # serve turns it off or names another, which the server fetches.
        for options, draft in (
            ([], True),
            (["--no-draft-model"], False),
            (["--draft-model", "owner/draft"], False),
        ):
            with (
                self.subTest(options=options),
                tempfile.TemporaryDirectory() as temporary,
            ):
                runtime = Path(temporary)
                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch.object(launcher.socket, "socket"),
                    mock.patch.object(launcher, "_ensure_installed") as install,
                    mock.patch.object(
                        launcher.model_artifacts,
                        "selection_link",
                        return_value=runtime / "selected",
                    ),
                    mock.patch.object(launcher.os, "execve"),
                ):
                    launcher.main(["serve", "--model", MODEL_ID, *options])
                self.assertEqual(install.call_args.kwargs, {"draft": draft})
                with (
                    mock.patch.object(launcher, "_run_held", return_value=0) as run,
                    mock.patch.object(launcher, "_build_from_source"),
                    mock.patch.object(launcher.model_artifacts, "run_engine"),
                ):
                    launcher._ensure_installed(install.call_args.args[0], draft=draft)
                parsed = launcher.model_artifacts.parse_args(run.call_args.args[0][2:])
                self.assertEqual(parsed.command, "prepare")
                self.assertEqual(parsed.no_draft_model, not draft)

    def test_offline_serve_forbids_the_hub_to_installer_and_server(self):
        for offline in (False, True):
            with (
                self.subTest(offline=offline),
                tempfile.TemporaryDirectory() as temporary,
            ):
                runtime = Path(temporary)
                os.environ.pop("HF_HUB_OFFLINE", None)
                seen = {}

                def install(selection, draft):
                    seen["installer"] = os.environ.get("HF_HUB_OFFLINE")

                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch.object(launcher.socket, "socket"),
                    mock.patch.object(
                        launcher, "_ensure_installed", side_effect=install
                    ),
                    mock.patch.object(
                        launcher.model_artifacts,
                        "selection_link",
                        return_value=runtime / "selected",
                    ),
                    mock.patch.object(launcher.os, "execve") as execute,
                ):
                    launcher.main(
                        [
                            "serve",
                            "--model",
                            MODEL_ID,
                            *(["--offline"] if offline else []),
                        ]
                    )
                argv, environment = execute.call_args.args[1:]
                expected = "1" if offline else None
                self.assertEqual(seen, {"installer": expected})
                self.assertEqual(environment.get("HF_HUB_OFFLINE"), expected)
                # The server has no such option: the environment carries it.
                self.assertNotIn("--offline", argv)

    def test_server_holds_the_assembly_it_serves(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            assembly = runtime / "models/.resolved/assembly"
            assembly.mkdir(parents=True)
            (assembly / "model.json").write_text("{}")
            selection = runtime / "models/owner/model"
            selection.parent.mkdir()
            selection.symlink_to(assembly)
            held = []

            def execute(program, argv, environment):
                # Installations remove no assembly a server holds.
                with (assembly / "model.json").open("rb") as record:
                    try:
                        fcntl.flock(record, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        held.append(server_arguments(argv)[0])

            with (
                mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                mock.patch.object(launcher.paths, "MODELS", runtime / "models"),
                mock.patch.object(launcher.socket, "socket"),
                mock.patch.object(launcher, "_ensure_installed"),
                mock.patch.object(
                    launcher.model_artifacts, "selection_link", return_value=selection
                ),
                mock.patch.object(launcher.os, "execve", side_effect=execute),
            ):
                launcher.main(["serve", "--model", MODEL_ID])
            self.assertEqual(held, [str(assembly.resolve())])

    def test_serve_takes_the_stop_signals_and_holds_them_across_the_exec(self):
        # A non-interactive shell starts background jobs with SIGINT ignored.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        held = {}

        def execute(*_):
            held["handlers"] = {
                number: signal.getsignal(number) for number in launcher.STOP_SIGNALS
            }
            held["mask"] = signal.pthread_sigmask(signal.SIG_BLOCK, ())

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
            mock.patch.object(launcher.socket, "socket"),
            mock.patch.object(launcher, "_ensure_installed"),
            mock.patch.object(launcher.os, "execve", side_effect=execute),
        ):
            launcher.main(["serve", "--model", MODEL_ID])
        self.assertEqual(
            held["handlers"], dict.fromkeys(launcher.STOP_SIGNALS, launcher._interrupt)
        )
        self.assertLessEqual(set(launcher.STOP_SIGNALS), held["mask"])

    def test_a_stop_ends_serve_with_the_status_of_its_signal(self):
        for number, status in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            with (
                self.subTest(signal=number.name),
                tempfile.TemporaryDirectory() as temporary,
                mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
                mock.patch.object(launcher.socket, "socket"),
                mock.patch.object(
                    launcher,
                    "_ensure_installed",
                    side_effect=lambda _, draft: signal.raise_signal(number),
                ),
            ):
                self.assertEqual(launcher.main(["serve", "--model", MODEL_ID]), status)

    def test_a_held_program_starts_with_the_stop_signals_blocked(self):
        check = (
            "import signal, sys; "
            "sys.exit(not {signal.SIGINT, signal.SIGTERM} "
            "<= signal.pthread_sigmask(signal.SIG_BLOCK, ()))"
        )
        self.assertEqual(launcher._run_held([sys.executable, "-c", check]), 0)
        self.assertFalse(
            set(launcher.STOP_SIGNALS) & signal.pthread_sigmask(signal.SIG_BLOCK, ())
        )

    def test_a_stop_the_launcher_takes_ends_the_held_program(self):
        signal.signal(signal.SIGINT, launcher._interrupt)
        main = threading.main_thread().ident
        popen = subprocess.Popen
        for moment in ("spawn", "run"):
            with self.subTest(moment=moment):
                programs = []

                def spawn(*args, **options):
                    programs.append(popen(*args, **options))
                    if moment == "spawn":
                        # Held: it arrives once the program exists.
                        signal.pthread_kill(main, signal.SIGINT)
                    else:
                        threading.Timer(
                            0.2, signal.pthread_kill, (main, signal.SIGINT)
                        ).start()
                    return programs[-1]

                with (
                    mock.patch.object(launcher.subprocess, "Popen", side_effect=spawn),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    launcher._run_held(
                        [sys.executable, "-c", "import time; time.sleep(60)"]
                    )
                self.assertEqual(programs[0].wait(timeout=5), -signal.SIGKILL)

    def test_programs_take_a_stop_held_from_their_start(self):
        home = self.enterContext(tempfile.TemporaryDirectory())
        # Unstopped, each would go on to fail at its first step.
        server = ["-m", "server.audio_server", "model", "--tokenizer", "tokenizer"]
        server += ["--model", MODEL_ID, "--port", "0"]
        installer = ["install/models.py", "--models", home, "--model", MODEL_ID]
        installer.append("prepare")
        environment = {
            **os.environ,
            "HOME": home,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_VERBOSITY": "error",
        }
        for program, number, status in (
            (server, signal.SIGINT, 0),
            (server, signal.SIGTERM, 0),
            (installer, signal.SIGINT, 130),
        ):
            with self.subTest(program=program[:2], signal=number.name):
                result = subprocess.run(
                    with_stop_held([sys.executable, *program], number),
                    cwd=launcher.ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(
                    (result.returncode, result.stdout, result.stderr), (status, "", "")
                )

    def test_unsupported_mac_is_refused_before_any_download(self):
        reason = (
            "Splash needs Apple GPU family 9 or newer (M3 or later) on macOS 26.4 "
            "or newer; this Mac has Apple M2 Max (Apple GPU family 8) on macOS "
            "26.4.1 (apple_gpu_family_9_required)"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary, python, prepared = root / "splash", root / "python", root / "ran"
            python.write_text(f"#!/bin/sh\ntouch '{prepared}'\n")
            python.chmod(0o755)
            # The binary's own line is the error, without its error: prefix;
            # a binary killed before main() is reported whole.
            for check, refusal in (
                (f"echo 'error: {reason}' >&2; exit 70", reason),
                (
                    "echo 'error: Metal device unavailable' >&2; exit 70",
                    "Metal device unavailable",
                ),
                (
                    "printf 'dyld: Symbol not found\\n  Expected in: Metal\\n' >&2; kill -ABRT $$",
                    "the engine's device check failed: dyld: Symbol not found\n"
                    "  Expected in: Metal",
                ),
                ("exit 0", None),
            ):
                with (
                    self.subTest(check=check),
                    mock.patch.object(launcher, "_build_from_source"),
                    mock.patch.object(launcher.paths, "BINARY", binary),
                    mock.patch.object(launcher.paths, "PYTHON", python),
                ):
                    binary.write_text(
                        f'#!/bin/sh\ntest "$*" = device-check || exit 2\n{check}\n'
                    )
                    binary.chmod(0o755)
                    prepared.unlink(missing_ok=True)
                    if refusal is None:
                        launcher._ensure_installed(selection(root))
                    else:
                        with self.assertRaises(launcher.LauncherError) as refused:
                            launcher._ensure_installed(selection(root))
                        self.assertEqual(str(refused.exception), refusal)
                    # Preparation, which downloads, runs only on a supported Mac.
                    self.assertEqual(prepared.exists(), refusal is None)

    def test_failed_download_never_executes_server(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
            mock.patch.object(launcher.socket, "socket"),
            mock.patch.object(
                launcher,
                "_ensure_installed",
                side_effect=launcher.LauncherError("download failed"),
            ),
            mock.patch.object(launcher.os, "execve") as execute,
            mock.patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(launcher.main(["serve", "--model", MODEL_ID]), 1)
            with (Path(temporary) / "serve-8000.lock").open("a+") as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute.assert_not_called()

    def test_metadata_write_failure_releases_lock_before_model_work(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
            mock.patch.object(launcher.json, "dump", side_effect=OSError("disk full")),
            mock.patch.object(launcher, "_ensure_installed") as install,
            mock.patch.object(launcher.os, "execve") as execute,
            mock.patch("sys.stderr", io.StringIO()) as error,
        ):
            self.assertEqual(launcher.main(["serve", "--model", MODEL_ID]), 1)
            self.assertIn("disk full", error.getvalue())
            with (Path(temporary) / "serve-8000.lock").open("a+") as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
        install.assert_not_called()
        execute.assert_not_called()


class CacheFirstServeTests(unittest.TestCase):
    """`splash-omni serve` asks the Hub only for what this machine lacks: the
    launcher's whole start, with the installer command it runs run in this
    process against a stand-in Hub, and the exec of the server stubbed."""

    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ))
        for name in ("SPLASH_PORT", "SPLASH_API_KEY", "HF_HUB_OFFLINE"):
            os.environ.pop(name, None)
        keep_stop_signals(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "hub"
        self.hub = FakeHub(self, self.cache)
        self.draft = self.enterContext(
            mock.patch.object(serve_options, "download_model")
        )
        for target, name, value in (
            (launcher, "RUNTIME_DIR", self.root / "runtime"),
            (launcher.paths, "MODELS", self.root / "models"),
        ):
            self.enterContext(mock.patch.object(target, name, value))
        self.enterContext(mock.patch.object(launcher.socket, "socket"))
        # The source build, done.
        self.enterContext(
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            )
        )

    def serve(self, model, *options):
        """The status, output and served root of `splash-omni serve --model MODEL
        OPTIONS`; the root is None when it did not start."""
        family = "qwen3-asr" if "ASR" in model else "qwen3-tts"

        def engine(arguments, check):
            if arguments[0] == "omni-check":
                return json.dumps({"family": family})
            return ""

        def installer(command, **options):
            self.assertEqual(Path(command[1]), launcher.ROOT / "install/models.py")
            # A new process reads HF_HUB_OFFLINE when it starts.
            offline = os.environ.get("HF_HUB_OFFLINE") == "1"
            with mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", offline):
                return launcher.model_artifacts.main(command[2:])

        with (
            mock.patch.object(launcher.model_artifacts, "run_engine", engine),
            mock.patch.object(launcher, "_run_held", side_effect=installer),
            mock.patch.object(launcher.os, "execve") as execute,
            mock.patch("sys.stdout", io.StringIO()) as output,
            mock.patch("sys.stderr", output),
        ):
            status = launcher.main(["serve", "--model", model, *options])
        root = None
        if execute.called:
            root = server_arguments(execute.call_args.args[1])[0]
        return status, output.getvalue(), root

    def test_a_cached_model_starts_without_any_hub_call(self):
        cached_model(self.cache, TTS_MODEL, COMMIT, lambda p: audio_repo(p))
        # The Hub has a newer commit; it is never asked about it.
        self.hub.publish(TTS_MODEL, "b" * 40, lambda p: audio_repo(p))
        for words in (
            f"Using {TTS_MODEL}@{COMMIT[:12]} from the Hugging Face cache; "
            "the Hub is not asked.\n",
            f"Splash model {TTS_MODEL} is already installed in ",
        ):
            with self.subTest(words=words), no_hub():
                status, output, root = self.serve(TTS_MODEL)
                self.assertIsNone(status, output)
                self.assertIsNotNone(root, output)
                self.assertIn(words, output)
        self.assertEqual(self.hub.calls(), [])
        record = json.loads((Path(root) / "model.json").read_text())
        self.assertEqual(record["sources"]["target"]["revision"], COMMIT)
        self.draft.assert_not_called()

    def test_a_model_not_here_is_the_one_download(self):
        self.hub.publish(TTS_MODEL, COMMIT, lambda p: audio_repo(p))
        other = families.SUPPORTED_MODELS[1]
        self.hub.publish(other, COMMIT, lambda p: audio_repo(p))
        status, output, root = self.serve(TTS_MODEL)
        self.assertIsNotNone(root, output)
        self.assertEqual(self.hub.requests, [(TTS_MODEL, None)])
        self.assertTrue(self.hub.downloads)
        self.assertTrue(
            all(name.startswith(TTS_MODEL + "/") for name in self.hub.downloads)
        )
        self.draft.assert_not_called()

    def test_qwen3_asr_1_7b_fetches_its_draft_unless_off_or_cached(self):
        config = {
            "model_type": "qwen3_asr",
            "architectures": ["Qwen3ASRForConditionalGeneration"],
            "thinker_config": {"text_config": {"hidden_size": 2048}},
        }
        cached_model(self.cache, MODEL_ID, COMMIT, lambda p: audio_repo(p, config))
        draft = serve_options.DEFAULT_DRAFT_MODEL
        with no_hub():
            _, _, root = self.serve(MODEL_ID, "--no-draft-model")
            self.assertIsNotNone(root)
            self.draft.assert_not_called()
            _, output, root = self.serve(MODEL_ID)
            self.assertIsNotNone(root)
            self.draft.assert_called_once_with(draft)
            self.assertIn(f"Downloading {draft}, the speculative draft", output)
            self.draft.reset_mock()
            cached_model(self.cache, draft, "c" * 40, lambda p: audio_repo(p, config))
            _, _, root = self.serve(MODEL_ID)
            self.assertIsNotNone(root)
            self.draft.assert_not_called()
        self.assertEqual(self.hub.calls(), [])

    def test_offline_starts_a_cached_model_and_refuses_a_missing_one(self):
        self.hub.publish(MODEL_ID, COMMIT, lambda p: audio_repo(p))
        cached_model(self.cache, TTS_MODEL, COMMIT, lambda p: audio_repo(p))
        with no_hub():
            status, output, root = self.serve(TTS_MODEL, "--offline")
            self.assertIsNotNone(root, output)
            status, output, root = self.serve(MODEL_ID, "--offline")
        self.assertIsNone(root)
        self.assertEqual(status, 1)
        self.assertIn(
            f"error: {MODEL_ID} is neither installed nor in the Hugging Face "
            "cache, and HF_HUB_OFFLINE is set (splash-omni serve --offline); start "
            "without it to download the model\n",
            output,
        )
        self.assertIn("error: model download or verification failed\n", output)
        self.assertEqual(self.hub.calls(), [])


if __name__ == "__main__":
    unittest.main()

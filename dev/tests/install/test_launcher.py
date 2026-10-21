import contextlib
import errno
import fcntl
import io
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from dev.tests.server_fixtures import (
    console_written,
    keep_stop_signals,
    server_arguments,
)
from install import launcher
from server import serve_options
from server import server as api

MODEL_ID = "community/custom-splash"


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
    "mlx-community/Qwen3.8-27B-4bit",
    "mlx-community/Qwen3.6-35B-A3B-4bit",
    "community/custom-splash",
)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        # The launcher's defaults ignore the caller's Splash settings.
        self.enterContext(mock.patch.dict(os.environ))
        for name in (
            "SPLASH_PORT",
            "SPLASH_API_KEY",
            "SPLASH_DEFAULT_REASONING_EFFORT",
        ):
            os.environ.pop(name, None)
        keep_stop_signals(self)
        # Serve installs models under a models root of the test's, and keeps
        # its locks and records in a runtime directory of the test's.
        models = Path(self.enterContext(tempfile.TemporaryDirectory())) / "models"
        self.enterContext(mock.patch.object(launcher.paths, "MODELS", models))
        runtime = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(launcher, "RUNTIME_DIR", runtime))

    def test_serve_requires_exact_repository_id_before_build(self):
        for arguments in (
            ["serve"],
            ["serve", "--model", "qwen3.6-35b-a3b"],
            ["serve", "--model", "Qwen3.6-35B-A3B"],
            ["serve", "--model", "https://huggingface.co/community/model"],
            ["serve", "--model", "community/../model"],
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

    def test_only_serve_and_clients_are_public(self):
        for command in (
            "start",
            "stop",
            "status",
            "logs",
            "doctor",
            "model",
            "models",
            "uninstall",
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
                "--max-context",
                "100K",
            ]
        )
        self.assertEqual(args.model, MODEL_ID)
        self.assertEqual(args.max_memory, 28 * 1024**3)
        self.assertEqual(args.max_context, 102400)

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
                # The app reads why a start failed beside the port's lock.
                self.assertEqual(
                    served.serve_record, Path(temporary) / "serve-9123.json"
                )

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
            console_written()
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

            def check_install(chosen):
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
                    (served.max_context, served.max_memory, served.kv_format),
                    (102400, 28 * 1024**3, "bf16"),
                )
                self.assertEqual(
                    served.allowed_origin, [("tauri", "localhost", None), "*"]
                )
                self.assertTrue(served.no_webui)
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
                        "--kv-format",
                        "bf16",
                        "--api-key",
                        "test-server-key",
                        "--no-webui",
                        "--max-context",
                        "100K",
                        "--max-memory",
                        "28G",
                        "--allowed-origin",
                        "tauri://localhost",
                        "--allowed-origin",
                        "*",
                    ]
                )
            install.assert_called_once_with(selection(launcher.paths.MODELS))
            execute.assert_called_once()
            self.assertEqual(
                {p.name for p in runtime.iterdir()},
                {"serve.lock", "serve-8000.lock", "serve-8000.json"},
            )
            # The start's record, which the server, keeping the pid, ends.
            self.assertEqual(
                json.loads((runtime / "serve-8000.json").read_text()),
                {"pid": os.getpid(), "phase": "starting"},
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
                console_written()
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
                "model": "mlx-community/Qwen3.8-27B-4bit",
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
                console_written()
                self.assertIn(f"PID {os.getpid()}", error.getvalue())
                self.assertIn("model mlx-community/Qwen3.8-27B-4bit", error.getvalue())
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
                    console_written()
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
                console_written()
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
            for name in launcher.clients.INSTALL_URLS:
                self.assertEqual(launcher.parse_args([name]).port, 8123)
                args = launcher.parse_args([name, "--port", "7777"])
                self.assertEqual(args.port, 7777)
                self.assertEqual(args.client_args, [])
        for value in ("", "0", "-1", "65536", "invalid", "1.5"):
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {"SPLASH_PORT": value}),
            ):
                for arguments in (
                    ["serve", "--model", MODEL_ID],
                    *([name] for name in launcher.clients.INSTALL_URLS),
                ):
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

    def test_serve_refuses_abbreviated_options(self):
        # serve takes its options by their full names, as the desktop app
        # passes every one: an abbreviation would change meaning as options are
        # added.
        for arguments in (
            ["--po", "9000"],
            ["--po=9000"],
            ["--api-k", "k"],
            ["--hos", "0.0.0.0"],
            ["--mod=" + MODEL_ID],
        ):
            with (
                self.subTest(arguments=arguments),
                mock.patch("sys.stderr", io.StringIO()) as error,
                self.assertRaises(SystemExit) as failed,
            ):
                launcher.parse_args(["serve", "--model", MODEL_ID, *arguments])
            self.assertEqual(failed.exception.code, 2)
            self.assertIn("unrecognized arguments", error.getvalue())

    def test_client_port_selection_preserves_agent_arguments(self):
        cases = (
            (["--port", "2268", "--help"], 2268, ["--help"]),
            (["--port=2268", "--version"], 2268, ["--version"]),
            (["resume", "--last", "--port", "2268"], 2268, ["resume", "--last"]),
            (["--por", "2268"], 8000, ["--por", "2268"]),
            (["--", "--port", "7777"], 8000, ["--port", "7777"]),
            (
                ["--port", "2268", "--", "--port=7777", "--help"],
                2268,
                ["--port=7777", "--help"],
            ),
            (
                ["--port=2268", "exec", "--", "--port", "7777"],
                2268,
                ["exec", "--", "--port", "7777"],
            ),
        )
        for name in launcher.clients.INSTALL_URLS:
            for arguments, port, forwarded in cases:
                with self.subTest(name=name, arguments=arguments):
                    args = launcher.parse_args([name, *arguments])
                    self.assertEqual(args.port, port)
                    self.assertEqual(args.client_args, forwarded)
            for arguments in (["--port"], ["--port", "--help"]):
                with (
                    self.subTest(name=name, arguments=arguments),
                    mock.patch("sys.stderr", io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as failed:
                        launcher.parse_args([name, *arguments])
                    self.assertEqual(failed.exception.code, 2)

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

    def test_the_apps_launch_options_are_the_serve_options(self):
        # The desktop app offers every option of splash serve but those of one
        # model and those it sets itself, from one table (LaunchOption.all,
        # docs/package.md): a flag it keeps that serve no longer takes would
        # fail every start, and a new option would never show.
        with (
            mock.patch("sys.stdout", io.StringIO()) as output,
            self.assertRaises(SystemExit),
        ):
            launcher.parse_args(["serve", "--help"])
        flags = set(re.findall(r"(?m)^  (--[a-z0-9-]+)", output.getvalue())) - {
            "--help",
            "--model",
            "--model-dir",
            "--offline",
            "--revision",
            "--draft-model",
            "--mmproj",
        }
        self.assertIn("--max-memory", flags)
        table = Path(launcher.__file__).resolve().parents[1] / (
            "app/Sources/Splash/Preferences.swift"
        )
        options = set(re.findall(r'flag: "(--[a-z0-9-]+)"', table.read_text()))
        self.assertEqual(options, flags)

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
            console_written()
            self.assertIn(f"port {port}", error.getvalue())

    def test_exec_keeps_installation_locked_until_every_server_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "server").mkdir()
            (root / "server/__init__.py").write_text("")
            (root / "server/server.py").write_text(
                "import sys\nprint('ready', flush=True)\nsys.stdin.read()\n"
            )
            runtime = root / "runtime"
            script = (
                "import sys\nfrom pathlib import Path\nfrom install import launcher\n"
                "launcher.ROOT = Path(sys.argv[1])\n"
                "launcher.RUNTIME_DIR = launcher.ROOT / 'runtime'\n"
                "launcher.paths.MODELS = launcher.ROOT / 'models'\n"
                "launcher.paths.PYTHON = Path(sys.executable)\n"
                "launcher._ensure_installed = lambda selection: None\n"
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

    def test_clients_discover_and_connect_to_selected_http_port(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append((self.path, self.headers.get("Authorization")))
                payload = {
                    "data": [
                        {
                            "id": MODEL_ID,
                            "owned_by": "splash",
                            "context_length": 102400,
                            "input_modalities": ["text", "image", "pdf"],
                        }
                    ]
                }
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with (
            tempfile.TemporaryDirectory() as temporary,
            ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server,
        ):
            port = server.server_port
            worker = threading.Thread(target=server.serve_forever)
            worker.start()
            try:
                with (
                    mock.patch.dict(
                        os.environ,
                        {
                            "SPLASH_PORT": str(port),
                            "SPLASH_API_KEY": "test-key",
                            "PI_CODING_AGENT_DIR": str(Path(temporary) / "pi"),
                            # Hermes's root.
                            "HOME": temporary,
                        },
                    ),
                    mock.patch.object(
                        launcher.clients, "find_executable", return_value="/bin/echo"
                    ),
                    mock.patch.object(
                        launcher.clients, "command", wraps=launcher.clients.command
                    ) as command,
                    mock.patch.object(launcher.os, "execvpe") as execute,
                    mock.patch("sys.stdout", io.StringIO()),
                ):
                    os.environ.pop("HERMES_HOME", None)
                    for name in launcher.clients.INSTALL_URLS:
                        for arguments in ([], ["--port", str(port)]):
                            with self.subTest(name=name, arguments=arguments):
                                # An explicit port must override the environment
                                # for both discovery and the agent's connection.
                                os.environ["SPLASH_PORT"] = (
                                    "invalid" if arguments else str(port)
                                )
                                launcher.main([name, *arguments])
                                self.assertEqual(
                                    command.call_args.args[2],
                                    f"http://127.0.0.1:{port}",
                                )
                                self.assertEqual(
                                    command.call_args.args[3:5], (MODEL_ID, 102400)
                                )
                                self.assertEqual(
                                    command.call_args.kwargs["client_args"], []
                                )
                    self.assertEqual(
                        execute.call_count, 2 * len(launcher.clients.INSTALL_URLS)
                    )
            finally:
                server.shutdown()
                worker.join(timeout=5)
            self.assertEqual(
                requests,
                [
                    request
                    for name in launcher.clients.INSTALL_URLS
                    for _ in range(2)
                    # Pi also asks how the server renders later system messages.
                    for request in [("/v1/models", "Bearer test-key")]
                    + [("/status", "Bearer test-key")] * (name == "pi")
                ],
            )

    def test_only_a_given_port_that_finds_no_server_mentions_the_agents_own(self):
        with (
            mock.patch.object(
                launcher.clients, "find_executable", return_value="/bin/echo"
            ),
            mock.patch.object(launcher, "_request_json", return_value=None),
            mock.patch.object(launcher.os, "execvpe") as execute,
        ):
            for environment, arguments, mentioned in (
                ({}, [], False),
                ({"SPLASH_PORT": "2268"}, [], False),
                ({}, ["--port", "4096"], True),
            ):
                with (
                    self.subTest(environment=environment, arguments=arguments),
                    mock.patch.dict(os.environ, environment),
                    mock.patch("sys.stderr", io.StringIO()) as error,
                ):
                    self.assertEqual(launcher.main(["opencode", *arguments]), 1)
                    console_written()
                    self.assertIn("No ready Splash server", error.getvalue())
                    self.assertEqual(
                        "If --port was meant for opencode itself, put it after --."
                        in error.getvalue(),
                        mentioned,
                    )
        execute.assert_not_called()

    def test_client_setup_works_while_status_is_unavailable(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/status":
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps(
                    {
                        "data": [
                            {
                                "id": MODEL_ID,
                                "owned_by": "splash",
                                "context_length": 4096,
                                "input_modalities": ["text"],
                            }
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever)
            worker.start()
            try:
                with (
                    mock.patch.dict(
                        os.environ, {"SPLASH_PORT": str(server.server_port)}
                    ),
                    mock.patch.object(
                        launcher.clients, "find_executable", return_value="/bin/echo"
                    ),
                    mock.patch.object(
                        launcher.clients, "command", return_value=(["codex"], {})
                    ) as command,
                    mock.patch.object(launcher.os, "execvpe") as execute,
                    mock.patch("sys.stdout", io.StringIO()),
                ):
                    self.assertIsNone(launcher.main(["codex"]))
            finally:
                server.shutdown()
                worker.join(timeout=5)
        self.assertEqual(command.call_args.args[3:5], (MODEL_ID, 4096))
        execute.assert_called_once()

    def test_server_probe_never_goes_through_a_proxy(self):
        """The server is on loopback: the launcher reaches it directly, and a
        proxy the environment names never receives the server's key."""
        requests = []

        class Recorder(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(
                    (
                        self.server.server_port,
                        self.path,
                        self.headers.get("Authorization"),
                    )
                )
                body = b'{"data": []}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        with (
            ThreadingHTTPServer(("127.0.0.1", 0), Recorder) as server,
            ThreadingHTTPServer(("127.0.0.1", 0), Recorder) as proxy,
        ):
            port = server.server_port
            workers = [
                threading.Thread(target=listener.serve_forever)
                for listener in (server, proxy)
            ]
            for worker in workers:
                worker.start()
            try:
                # A new process, as the launcher is: urlopen keeps the
                # proxies it finds on its first call.
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys\nfrom install import launcher\n"
                        "print(launcher._request_json('/v1/models', "
                        "port=int(sys.argv[1])))",
                        str(port),
                    ],
                    cwd=Path(launcher.__file__).resolve().parents[1],
                    env={
                        "SPLASH_API_KEY": "test-key",
                        "http_proxy": f"http://127.0.0.1:{proxy.server_port}",
                    },
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            finally:
                for listener in (server, proxy):
                    listener.shutdown()
                for worker in workers:
                    worker.join(timeout=5)
        self.assertEqual(result.stdout, "{'data': []}\n", result.stderr)
        self.assertEqual(requests, [(port, "/v1/models", "Bearer test-key")])

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
            script = checkout / "splash"
            script.write_bytes((launcher.ROOT / "splash").read_bytes())
            script.chmod(0o755)
            # A relative link into the checkout, reached through an absolute one.
            (root / "bin").mkdir()
            (root / "bin/splash").symlink_to("../checkout/splash")
            (root / "path").mkdir()
            (root / "path/splash").symlink_to(root / "bin/splash")
            # The relative link through a linked directory: its .. is physical.
            (root / "path/linked").symlink_to("../bin")
            for command in (
                script,
                root / "bin/splash",
                root / "path/splash",
                root / "path/linked/splash",
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
                    mock.patch.object(launcher.paths, "PACKAGED", False),
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
                    mock.patch.object(launcher.paths, "PACKAGED", False),
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
            console_written()
            self.assertEqual(
                output.getvalue(), "Another Splash build is running; waiting...\n"
            )
            self.assertEqual(run.call_args_list[0].args[0][0], "make")

    def test_source_selection_reaches_installation_and_served_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            draft = runtime / "local-draft"
            draft.mkdir()
            options = {
                "revision": "v2",
                "language_only": True,
                "draft_model": str(draft.resolve()),
            }
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
                        "--draft-model",
                        str(draft),
                        "--language-only",
                    ]
                )
            (chosen,) = install.call_args.args
            self.assertEqual(
                (chosen.model, chosen.link),
                (MODEL_ID, runtime / "selected"),
            )
            root.assert_called_once_with(chosen.models_root, MODEL_ID, **options)
            # The server serves the assembly the selection link names.
            argv = execute.call_args.args[1]
            self.assertEqual(server_arguments(argv)[0], str(chosen.link))

            with (
                mock.patch.object(
                    launcher.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0),
                ),
                mock.patch.object(launcher, "_run_held", return_value=0) as run,
                mock.patch.object(launcher.paths, "PACKAGED", True),
            ):
                launcher._ensure_installed(chosen)
            command = run.call_args.args[0]
            self.assertEqual(run.call_args.kwargs["cwd"], launcher.ROOT)
            parsed = launcher.model_artifacts.parse_args(command[2:])
            self.assertEqual(
                (
                    parsed.command,
                    parsed.model,
                    parsed.revision,
                    parsed.language_only,
                    parsed.draft_model,
                ),
                ("prepare", MODEL_ID, "v2", True, options["draft_model"]),
            )

    def test_offline_serve_forbids_the_hub_to_installer_and_server(self):
        for offline in (False, True):
            with (
                self.subTest(offline=offline),
                tempfile.TemporaryDirectory() as temporary,
            ):
                runtime = Path(temporary)
                os.environ.pop("HF_HUB_OFFLINE", None)
                seen = {}

                def install(selection):
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

    def test_server_is_given_the_selection_link_whose_assembly_it_holds(self):
        # The server holds the assembly the link names when it starts, and
        # serves it however an installation moves the link.
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            selection = runtime / "models/owner/model"
            with (
                mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                mock.patch.object(launcher.socket, "socket"),
                mock.patch.object(launcher, "_ensure_installed"),
                mock.patch.object(
                    launcher.model_artifacts, "selection_link", return_value=selection
                ),
                mock.patch.object(launcher.os, "execve") as execute,
            ):
                launcher.main(["serve", "--model", MODEL_ID])
            served = api.parse_args(server_arguments(execute.call_args.args[1]))
            self.assertEqual(served.model_root, str(selection))

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
                    side_effect=lambda _: signal.raise_signal(number),
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
        # In a launcher process of its own, with the stop sent to the process,
        # as Ctrl+C and the app send it: the kernel hands it to any thread that
        # has it unblocked. It comes as the program is spawned, held until the
        # program exists, or while the program runs. The 50 ms give a stop
        # that another thread took the time to be raised in the spawn, where
        # nothing would end the program.
        program = "\n".join(
            (
                "import os, signal, subprocess, sys, threading, time",
                "from unittest import mock",
                "from install import launcher",
                "signal.signal(signal.SIGINT, launcher._interrupt)",
                "popen = subprocess.Popen",
                "programs = []",
                "stop = (os.getpid(), signal.SIGINT)",
                "def spawn(*args, **options):",
                "    programs.append(popen(*args, **options))",
                "    if sys.argv[1] == 'spawn':",
                "        os.kill(*stop)",
                "        time.sleep(0.05)",
                "    else:",
                "        threading.Timer(0.2, os.kill, stop).start()",
                "    return programs[-1]",
                "held = [sys.executable, '-c', 'import time; time.sleep(10)']",
                "with mock.patch.object(subprocess, 'Popen', side_effect=spawn):",
                "    try:",
                "        launcher._run_held(held)",
                "    except KeyboardInterrupt:",
                "        pass",
                "try:",
                "    print(programs[0].wait(5))",
                "except subprocess.TimeoutExpired:",
                "    programs[0].kill()",
                "    print('still running')",
            )
        )
        for moment in ("spawn", "run"):
            with self.subTest(moment=moment):
                result = subprocess.run(
                    [sys.executable, "-c", program, moment],
                    cwd=launcher.ROOT,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.stdout, f"{-signal.SIGKILL}\n", result.stderr)

    def test_programs_take_a_stop_held_from_their_start(self):
        home = self.enterContext(tempfile.TemporaryDirectory())
        # Unstopped, each would go on to fail at its first step.
        server = ["-m", "server.server", "model", "--model", MODEL_ID, "--port", "0"]
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

    def test_relative_draft_directory_is_resolved_for_the_installer(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            contextlib.chdir(temporary),
        ):
            base = Path(temporary).resolve()
            (base / "drafts/local").mkdir(parents=True)
            for value in ("drafts/local", "./drafts/local", "drafts/../drafts/local"):
                with self.subTest(value=value):
                    args = launcher.parse_args(
                        ["serve", "--model", MODEL_ID, "--draft-model", value]
                    )
                    self.assertEqual(args.draft_model, str(base / "drafts/local"))
                    # The installer, run directly, records the same directory.
                    args = launcher.model_artifacts.parse_args(
                        ["--model", MODEL_ID, "--draft-model", value, "prepare"]
                    )
                    self.assertEqual(args.draft_model, str(base / "drafts/local"))
            repository = "incoai/Qwen3.8-27B-DFlash2"
            args = launcher.parse_args(
                ["serve", "--model", MODEL_ID, "--draft-model", repository]
            )
            self.assertEqual(args.draft_model, repository)
            # none selects no draft, even beside a directory of that name,
            # which ./none names.
            (base / "none").mkdir()
            for value, draft in (("none", "none"), ("./none", str(base / "none"))):
                with self.subTest(value=value):
                    args = launcher.parse_args(
                        ["serve", "--model", MODEL_ID, "--draft-model", value]
                    )
                    self.assertEqual(args.draft_model, draft)
            for value in ("./missing", "missing", "", "drafts/local/../../missing/"):
                with (
                    self.subTest(value=value),
                    mock.patch.object(launcher, "_ensure_installed") as install,
                    mock.patch("sys.stderr", io.StringIO()) as error,
                    self.assertRaises(SystemExit) as failed,
                ):
                    launcher.main(
                        ["serve", "--model", MODEL_ID, "--draft-model", value]
                    )
                self.assertEqual(failed.exception.code, 2)
                self.assertIn(
                    "argument --draft-model: must be a local DFlash2 draft "
                    "directory, a Hugging Face repository ID (owner/repo) or none",
                    error.getvalue(),
                )
                install.assert_not_called()

    def test_packaged_serve_never_invokes_make_or_system_python(self):
        with (
            mock.patch.object(launcher.paths, "PACKAGED", True),
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
            mock.patch.object(launcher, "_run_held", return_value=0) as run_held,
        ):
            launcher._ensure_installed(selection(launcher.paths.MODELS))
        (check,) = (call.args[0] for call in run.call_args_list)
        command = run_held.call_args.args[0]
        self.assertEqual(check, [str(launcher.paths.BINARY), "device-check"])
        self.assertEqual(command[0], str(launcher.paths.PYTHON))
        self.assertIn("prepare", command)
        self.assertNotIn("make", command)

    def test_python_programs_write_bytecode_only_as_the_launcher_does(self):
        # A runtime inside a signed app must not gain __pycache__ folders: a
        # launcher run with -B or PYTHONDONTWRITEBYTECODE gives the installer
        # and the server PYTHONDONTWRITEBYTECODE (the server's PDF workers,
        # started with no environment, have -B of their own).
        for flag in (False, True):
            with (
                self.subTest(dont_write_bytecode=flag),
                tempfile.TemporaryDirectory() as temporary,
            ):
                runtime = Path(temporary)
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
                seen = {}

                def install(selection):
                    seen["installer"] = os.environ.get("PYTHONDONTWRITEBYTECODE")

                with (
                    mock.patch.object(launcher.sys, "dont_write_bytecode", flag),
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
                    launcher.main(["serve", "--model", MODEL_ID])
                argv, environment = execute.call_args.args[1:]
                expected = "1" if flag else None
                self.assertEqual(seen, {"installer": expected})
                self.assertEqual(environment.get("PYTHONDONTWRITEBYTECODE"), expected)
                self.assertNotIn("-B", argv)

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
                    mock.patch.object(launcher.paths, "PACKAGED", True),
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

    def test_a_refused_start_says_why_in_its_serve_record(self):
        """The app words a start the launcher refuses by the record's kind."""

        @contextlib.contextmanager
        def upgrading(runtime):
            with (runtime / "serve.lock").open("a+") as installation:
                fcntl.flock(installation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield

        def helper(output):
            def run(command, *, stdout, **_options):
                stdout.write(output)
                return 1

            return mock.patch.object(launcher, "_run_held", side_effect=run)

        def installer(runtime):
            # Its messages, and its failure as an error event, on stdout.
            program = runtime / "python"
            program.write_text(
                "#!/bin/sh\necho 'Selected a/b.'\necho '"
                '{"event":"error","kind":null,"message":"a/b is not installed"}'
                "'\nexit 1\n"
            )
            program.chmod(0o755)
            return mock.patch.object(launcher.paths, "PYTHON", program)

        unsupported = "Splash needs Apple GPU family 9 or newer (M3 or later)"
        for case, setup, arguments, failure, said, output in (
            (
                "upgrade",
                lambda runtime: [upgrading(runtime)],
                [],
                {
                    "kind": "upgrade",
                    "message": "Splash is being upgraded; wait for the upgrade to finish",
                },
                "error: Splash is being upgraded; wait for the upgrade to finish\n",
                "",
            ),
            (
                "port",
                lambda _: [
                    mock.patch.object(
                        launcher,
                        "_check_port",
                        side_effect=OSError(errno.EADDRINUSE, "Address already in use"),
                    )
                ],
                [],
                {
                    "kind": "port",
                    "message": "cannot bind 127.0.0.1:8000: [Errno 48] Address already in use",
                },
                "error: cannot bind 127.0.0.1:8000: [Errno 48] Address already in use\n",
                "",
            ),
            (
                "device",
                lambda _: [
                    mock.patch.object(
                        launcher.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess([], 0),
                    ),
                    mock.patch.object(
                        launcher.model_artifacts,
                        "run_engine",
                        side_effect=launcher.model_artifacts.ModelError(unsupported),
                    ),
                ],
                [],
                {"kind": "device", "message": unsupported},
                f"error: {unsupported}\n",
                "",
            ),
            (
                "install",
                lambda runtime: [
                    mock.patch.object(launcher, "_ensure_runtime"),
                    installer(runtime),
                ],
                [],
                {"kind": "install", "message": "a/b is not installed"},
                # Said by the installer (test_an_installer_failure_is_said_once).
                "",
                "Selected a/b.\n",
            ),
            (
                "local",
                lambda _: [
                    mock.patch.object(launcher, "_ensure_runtime"),
                    helper('{"kind":"model","message":"it is not a GGUF file"}'),
                ],
                ["--model-dir", "/m.gguf"],
                {"kind": "model", "message": "it is not a GGUF file"},
                # The line names the files, which the record leaves to the app.
                "error: cannot prepare /m.gguf: it is not a GGUF file\n",
                "",
            ),
            (
                "local crash",
                lambda _: [mock.patch.object(launcher, "_ensure_runtime"), helper("")],
                ["--model-dir", "/m.gguf"],
                {"kind": "internal", "message": "local model preparation failed"},
                "error: local model preparation failed\n",
                "",
            ),
            (
                "exec",
                lambda _: [
                    mock.patch.object(
                        launcher, "_prepare_start", return_value=Path("/m")
                    )
                ],
                [],
                {"kind": "internal", "message": "[Errno 2] No such file or directory"},
                "error: [Errno 2] No such file or directory\n",
                "",
            ),
        ):
            with (
                self.subTest(case),
                tempfile.TemporaryDirectory() as temporary,
                contextlib.ExitStack() as stack,
            ):
                runtime = Path(temporary)
                stack.enter_context(mock.patch.object(launcher, "RUNTIME_DIR", runtime))
                stack.enter_context(mock.patch.object(launcher.socket, "socket"))
                stack.enter_context(
                    mock.patch.object(
                        launcher.os,
                        "execve",
                        side_effect=FileNotFoundError(
                            errno.ENOENT, "No such file or directory"
                        ),
                    )
                )
                error = stack.enter_context(mock.patch("sys.stderr", io.StringIO()))
                relayed = stack.enter_context(mock.patch("sys.stdout", io.StringIO()))
                for patch in setup(runtime):
                    stack.enter_context(patch)
                self.assertEqual(
                    launcher.main(["serve", "--model", MODEL_ID, *arguments]), 1
                )
                self.assertEqual(
                    json.loads((runtime / "serve-8000.json").read_text()),
                    {"pid": os.getpid(), "phase": "failed", "failure": failure},
                )
                console_written()
                self.assertEqual(error.getvalue(), said)
                self.assertEqual(relayed.getvalue(), output)

    def test_an_installer_failure_is_said_once(self):
        """The installer says its failure on stderr itself, beside the error
        event the launcher records it by, so the launcher does not say it
        again; an installer that ended without one is said by the launcher."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            script = (
                "import sys\nfrom pathlib import Path\nfrom install import launcher\n"
                "root = Path(sys.argv[1])\n"
                "launcher.RUNTIME_DIR = root / 'runtime'\n"
                "launcher.paths.MODELS = root / 'models'\n"
                "launcher.paths.PYTHON = Path(sys.argv[3])\n"
                "launcher._ensure_runtime = lambda: None\n"
                "raise SystemExit(launcher.main(\n"
                "    ['serve', '--model', 'test/model', '--offline', '--port', sys.argv[2]]\n"
                "))\n"
            )
            crashed = root / "crashed"
            crashed.write_text("#!/bin/sh\nexit 1\n")
            crashed.chmod(0o755)
            unresolved = (
                "cannot resolve test/model: HF_HUB_OFFLINE is set; neither this "
                "installation nor the Hub cache records a commit for the default branch"
            )
            for case, python, message in (
                # The real installer, offline, with an empty Hub cache.
                ("installer", sys.executable, unresolved),
                ("crashed", crashed, "model download or verification failed"),
            ):
                with self.subTest(case):
                    result = subprocess.run(
                        [sys.executable, "-c", script, str(root), str(port), python],
                        cwd=Path(launcher.__file__).resolve().parents[1],
                        env=dict(os.environ, HF_HUB_CACHE=str(root / "hub")),
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertEqual(
                        result.stderr.splitlines().count(f"error: {message}"),
                        1,
                        result.stderr,
                    )
                    record = root / f"runtime/serve-{port}.json"
                    self.assertEqual(
                        json.loads(record.read_text())["failure"],
                        {"kind": "install", "message": message},
                    )

    def test_a_start_refused_the_lock_leaves_the_servers_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            record = runtime / "serve-8000.json"
            record.write_text('{"pid":123,"phase":"starting"}')
            with (runtime / "serve-8000.lock").open("a+") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch("sys.stderr", io.StringIO()) as error,
                ):
                    self.assertEqual(launcher.main(["serve", "--model", MODEL_ID]), 1)
            console_written()
            self.assertIn("Splash is already serving", error.getvalue())
            self.assertEqual(record.read_text(), '{"pid":123,"phase":"starting"}')

    def test_a_refused_serve_command_line_is_recorded_at_its_port(self):
        """A serve command line the launcher refuses is recorded with the
        message it printed at the port it names, under that port's lock, over
        a record an earlier process left; one another process holds keeps
        that process's."""
        stale = json.dumps({"pid": os.getpid(), "phase": "ready"})
        for arguments in (
            ["--port", "9123"],
            ["--model", MODEL_ID, "--port=9123", "--max-memory", "1Q"],
            ["--model", MODEL_ID, "--port", "9123", "--revision", "main"]
            + ["--model-dir", "/m"],
        ):
            for held in (False, True):
                with (
                    self.subTest(arguments=arguments, held=held),
                    tempfile.TemporaryDirectory() as temporary,
                    mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
                    mock.patch("sys.stderr", io.StringIO()) as error,
                    (Path(temporary) / "serve-9123.lock").open("a+") as holder,
                ):
                    record = Path(temporary) / "serve-9123.json"
                    record.write_text(stale)
                    if held:
                        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(SystemExit) as refused:
                        launcher.main(["serve", *arguments])
                    self.assertEqual(refused.exception.code, 2)
                    _, refusal, printed = (
                        error.getvalue().splitlines()[-1].partition(": error: ")
                    )
                    self.assertTrue(refusal)
                    self.assertEqual(
                        json.loads(record.read_text()),
                        json.loads(stale)
                        if held
                        else {
                            "pid": os.getpid(),
                            "phase": "failed",
                            "failure": {
                                "kind": "configuration",
                                "message": printed,
                            },
                        },
                    )

    def test_a_refusal_whose_record_cannot_be_written_stays_a_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            # A runtime directory under a file cannot be created; a record
            # with a directory in its place cannot be replaced.
            (Path(temporary) / "file").touch()
            (Path(temporary) / "run/serve-9123.json").mkdir(parents=True)
            for runtime in (Path(temporary) / "file/run", Path(temporary) / "run"):
                with (
                    self.subTest(runtime=runtime.name),
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch("sys.stderr", io.StringIO()) as error,
                    self.assertRaises(SystemExit) as refused,
                ):
                    launcher.main(["serve", "--port", "9123"])
                self.assertEqual(refused.exception.code, 2)
                self.assertIn("splash serve: error: ", error.getvalue())

    def test_a_start_whose_starting_record_cannot_be_written_fails_before_model_work(
        self,
    ):
        """A start that cannot record it began, a directory in its record's
        place, fails at once with one error naming the record and the
        system's error, before any model work, and releases its port's lock:
        otherwise an earlier process's record, a stale ready one of the same
        pid among them, would say how this start goes."""
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            record = runtime / "serve-8000.json"
            record.mkdir()
            with (
                mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                mock.patch.object(launcher, "_prepare_start") as prepare,
                mock.patch.object(launcher.os, "execve") as execute,
                mock.patch("sys.stderr", io.StringIO()) as error,
            ):
                self.assertEqual(launcher.main(["serve", "--model", MODEL_ID]), 1)
            prepare.assert_not_called()
            execute.assert_not_called()
            console_written()
            self.assertEqual(
                error.getvalue(),
                f"error: cannot write the serve record {record}: Is a directory\n",
            )
            with (runtime / "serve-8000.lock").open("a+") as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_a_start_refused_whose_record_cannot_be_written_says_its_refusal(self):
        """A start the launcher refuses once its record cannot be written, a
        directory in its place after it recorded the start, prints its
        refusal, not the record's failure."""
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            record = runtime / "serve-8000.json"
            starting = launcher.serve_record.starting

            def started(path):
                starting(path)
                path.unlink()
                path.mkdir()

            with (
                (runtime / "serve.lock").open("a+") as installation,
                mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                mock.patch.object(
                    launcher.serve_record, "starting", side_effect=started
                ),
                mock.patch("sys.stderr", io.StringIO()) as error,
            ):
                fcntl.flock(installation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(launcher.main(["serve", "--model", MODEL_ID]), 1)
            console_written()
            written, refusal = error.getvalue().splitlines()
            self.assertEqual(
                refusal,
                "error: Splash is being upgraded; wait for the upgrade to finish",
            )
            # The record of its refusal.
            self.assertTrue(
                written.endswith(
                    f" Warning · cannot write the serve record {record}: Is a directory"
                ),
                written,
            )

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
            console_written()
            self.assertIn("disk full", error.getvalue())
            with (Path(temporary) / "serve-8000.lock").open("a+") as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
        install.assert_not_called()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()

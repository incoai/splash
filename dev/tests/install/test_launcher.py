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

from dev.tests.server import test_videos
from dev.tests.server_fixtures import (
    console_written,
    keep_stop_signals,
    server_arguments,
)
from install import launcher
from install.families import H3
from server import serve_options, serve_record, videos
from server import server as api

MODEL_ID = "community/custom-splash"
MODEL = test_videos.MODEL
SEVEN = test_videos.SEVEN


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
            "SPLASH_REASONING_EFFORT",
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
                "host": "127.0.0.1",
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
                "host": "unix:/tmp/splash.sock",
            }
            (runtime / "serve-8000.json").write_text(
                json.dumps({"pid": os.getpid(), "phase": "loading"})
            )
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
                self.assertIn("host unix:/tmp/splash.sock", error.getvalue())
                self.assertIn("phase loading", error.getvalue())
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
                        "error: Splash is already serving; stop it with Ctrl+C "
                        "first, or serve on another --port\n",
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

    def test_a_unix_socket_is_bound_and_given_up_before_model_work(self):
        """A --host of unix:PATH is tried as the server binds it: a free path
        and one a killed server left its socket at go on, the path left clear
        for the server, under the port's lock and record; a path a server
        holds, one still loading too, fails the start before any model work,
        named without the port it does not listen on, and stays that
        server's."""
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            runtime = Path(temporary) / "runtime"
            path = Path(temporary) / "splash.sock"
            arguments = ["serve", "--model", MODEL_ID, "--port", "9123"]
            arguments += ["--host", f"unix:{path}"]
            for leftover in (False, True):
                if leftover:
                    with socket.socket(socket.AF_UNIX) as killed:
                        killed.bind(str(path))
                with (
                    self.subTest(leftover=leftover),
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch.object(launcher, "_ensure_installed") as install,
                    mock.patch.object(launcher.os, "execve") as execute,
                ):
                    self.assertIsNone(launcher.main(arguments))
                    install.assert_called_once()
                    served = api.parse_args(server_arguments(execute.call_args.args[1]))
                    self.assertEqual(served.host, f"unix:{path}")
                    self.assertEqual(served.serve_record, runtime / "serve-9123.json")
                    self.assertFalse(path.exists())
            holder = launcher.unix_socket.Claim(str(path))
            self.addCleanup(holder.close)
            with socket.socket(socket.AF_UNIX) as loading:
                holder.bind(loading)
                bound = path.lstat().st_ino
                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch.object(launcher, "_ensure_installed") as install,
                    mock.patch.object(launcher.os, "execve") as execute,
                    mock.patch("sys.stderr", io.StringIO()) as error,
                ):
                    self.assertEqual(launcher.main(arguments), 1)
                console_written()
                message = f"cannot bind unix:{path}: [Errno 48] Address already in use"
                self.assertEqual(error.getvalue(), f"error: {message}\n")
                install.assert_not_called()
                execute.assert_not_called()
                self.assertEqual(path.lstat().st_ino, bound)
            self.assertEqual(
                json.loads((runtime / "serve-9123.json").read_text())["failure"],
                {"kind": "port", "message": message},
            )

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
        # model, those it sets itself and the effort of every request, which
        # it leaves to the command line, from one table (LaunchOption.all,
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
            "--chat-template-file",
            "--reasoning-effort",
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

    def test_an_agent_command_names_the_server_that_holds_its_port(self):
        """An agent command that finds no ready server at 127.0.0.1:PORT,
        where a server holds the port nonetheless, names it: its PID and
        model, where it listens, a UNIX socket or another address, and its
        phase as its serve record says, from the start on. Of one that
        listens where the command asks and says it is ready, it says only
        that it does not answer, as a server that is stopping does not (it
        was told that agent commands need a ready server on loopback, which
        that server said it was)."""
        with (
            tempfile.TemporaryDirectory(dir="/tmp") as temporary,
            socket.socket() as unused,
        ):
            unused.bind(("127.0.0.1", 0))
            port = unused.getsockname()[1]
            unused.close()
            os.environ["SPLASH_PORT"] = str(port)
            record = launcher.RUNTIME_DIR / f"serve-{port}.json"
            refusals = {}

            def execute(binary, argv, environment):
                # The record the server, keeping the launcher's pid, writes as
                # its start goes on.
                for phase, write in (
                    ("starting", lambda: None),
                    ("loading", lambda: serve_record.entered(record, "loading")),
                    ("warming", lambda: serve_record.warming(record, 1, 4)),
                    ("ready", lambda: serve_record.ready(record)),
                ):
                    write()
                    with mock.patch("sys.stderr", io.StringIO()) as error:
                        self.assertEqual(launcher.main(["claude"]), 1)
                    console_written()
                    refusals[phase] = error.getvalue()

            for host in (
                f"unix:{temporary}/splash.sock",
                "192.0.2.1",
                "127.0.0.1",
                "0.0.0.0",
            ):
                refusals.clear()
                with (
                    self.subTest(host=host),
                    # Whether the address can be bound is not this test's.
                    mock.patch.object(launcher, "_check_port"),
                    mock.patch.object(launcher, "_ensure_installed"),
                    mock.patch.object(launcher.os, "execve", side_effect=execute),
                    mock.patch.object(
                        launcher.clients, "find_executable", return_value="/bin/echo"
                    ),
                    mock.patch.object(launcher.os, "execvpe") as agent,
                ):
                    self.assertIsNone(
                        launcher.main(["serve", "--model", MODEL_ID, "--host", host])
                    )
                    agent.assert_not_called()
                    self.assertEqual(
                        list(refusals), ["starting", "loading", "warming", "ready"]
                    )
                    url = f"http://127.0.0.1:{port}"
                    for phase, said in refusals.items():
                        named = (
                            f"(PID {os.getpid()}, model {MODEL_ID}, port {port}, "
                            f"host {host}, phase {phase})"
                        )
                        asked = host in ("127.0.0.1", "0.0.0.0") and phase == "ready"
                        self.assertEqual(
                            said,
                            f"error: Splash is serving {named} but does not "
                            f"answer at {url}: it may be stopping.\n"
                            if asked
                            else f"error: No ready Splash server at {url}, but "
                            f"Splash is serving {named}. Agent commands connect "
                            "over TCP on loopback to a server that is ready.\n",
                        )

    def test_an_exited_servers_lock_names_no_server_nor_another_pids_record_a_phase(
        self,
    ):
        """The lock and the record a server that exited leaves name no server
        to an agent command, and a record of another process's says nothing
        of the server holding the lock."""
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            port = unused.getsockname()[1]
        os.environ["SPLASH_PORT"] = str(port)
        lock = launcher.RUNTIME_DIR / f"serve-{port}.lock"
        lock.write_text(
            json.dumps(
                {"pid": 4242, "model": MODEL_ID, "port": port, "host": "192.0.2.1"}
            )
        )
        (launcher.RUNTIME_DIR / f"serve-{port}.json").write_text(
            json.dumps({"pid": 123, "phase": "ready"})
        )
        with (
            mock.patch.object(
                launcher.clients, "find_executable", return_value="/bin/echo"
            ),
            mock.patch.object(launcher.os, "execvpe") as agent,
        ):
            for held in (False, True):
                with (
                    self.subTest(held=held),
                    lock.open() as holder,
                    mock.patch("sys.stderr", io.StringIO()) as error,
                ):
                    if held:
                        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.assertEqual(launcher.main(["codex"]), 1)
                    console_written()
                    self.assertEqual(
                        error.getvalue(),
                        f"error: No ready Splash server at http://127.0.0.1:{port}"
                        + (
                            f", but Splash is serving (PID 4242, model {MODEL_ID}, "
                            f"port {port}, host 192.0.2.1). Agent commands connect "
                            "over TCP on loopback to a server that is ready.\n"
                            if held
                            else ". Run 'splash serve --model <HF_REPO_ID>' in "
                            "another terminal first.\n"
                        ),
                    )
        agent.assert_not_called()

    def test_an_agent_command_says_what_is_known_of_a_ready_161_server(self):
        """A 1.6.1 server's lock names no --host. Of one whose record says
        ready and that does not answer at 127.0.0.1:PORT, an agent command
        says that it does not answer, which it may not as it stops or where
        it listens on another address (it said there was no ready server and
        that agent commands need a ready one, beside `phase ready`)."""
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            port = unused.getsockname()[1]
        os.environ["SPLASH_PORT"] = str(port)
        lock = launcher.RUNTIME_DIR / f"serve-{port}.lock"
        lock.write_text(json.dumps({"pid": 4242, "model": MODEL_ID, "port": port}))
        (launcher.RUNTIME_DIR / f"serve-{port}.json").write_text(
            json.dumps({"pid": 4242, "phase": "ready"})
        )
        with (
            mock.patch.object(
                launcher.clients, "find_executable", return_value="/bin/echo"
            ),
            mock.patch.object(launcher.os, "execvpe") as agent,
            lock.open() as holder,
            mock.patch("sys.stderr", io.StringIO()) as error,
        ):
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(launcher.main(["codex"]), 1)
            console_written()
        self.assertEqual(
            error.getvalue(),
            f"error: Splash is serving (PID 4242, model {MODEL_ID}, port {port}, "
            f"phase ready) but does not answer at http://127.0.0.1:{port}: it may "
            "be stopping, or listen on another address.\n",
        )
        agent.assert_not_called()

    def test_an_ipv6_host_is_refused_before_model_work(self):
        """The server listens on IPv4 alone: a start with --host :: or ::1 is
        refused as it checks the address, so a server that holds the port
        under either is never ready, as an agent command that finds one
        says it is not (a ready one on ::1 would be on loopback, where the
        command does not reach it)."""
        for host in ("::", "::1"):
            with (
                self.subTest(host=host),
                tempfile.TemporaryDirectory() as temporary,
                mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
                mock.patch.object(launcher, "_ensure_installed") as install,
                mock.patch.object(launcher.os, "execve") as execute,
                mock.patch("sys.stderr", io.StringIO()) as error,
            ):
                result = launcher.main(["serve", "--model", MODEL_ID, "--host", host])
                console_written()
                self.assertEqual(result, 1)
                self.assertIn(f"cannot bind {host}:", error.getvalue())
                install.assert_not_called()
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

    def test_generate_installs_as_serve_does_and_starts_the_server_serve_would(self):
        model = "MiniMaxAI/MiniMax-H3"
        with (
            tempfile.TemporaryDirectory() as runtime,
            contextlib.chdir(runtime),
            mock.patch.object(launcher, "RUNTIME_DIR", Path(runtime)),
            mock.patch.object(launcher, "_check_port") as probe,
            mock.patch.object(launcher, "_ensure_installed") as install,
            mock.patch.object(launcher, "_holds_video_model", return_value=True),
            mock.patch.object(
                launcher, "_spawn_server", side_effect=OSError("not here")
            ) as spawn,
            mock.patch.object(launcher.os, "execve") as execute,
            mock.patch("sys.stderr", io.StringIO()) as stderr,
        ):
            status = launcher.main([
                "generate", "--model", model, "--prompt", "a cat", "--out", "clip.mp4",
            ])  # fmt: skip
            console_written()
            held = [file.name for file in spawn.call_args.args[2]]
        self.assertEqual(status, 1)
        self.assertIn(H3.license, stderr.getvalue())
        probe.assert_not_called()
        execute.assert_not_called()
        selection = install.call_args.args[0]
        command, environment, _ = spawn.call_args.args
        self.assertEqual(command[4:8], ["server.server", str(selection.link), "--model", model])  # fmt: skip
        served = api.parse_args(server_arguments(command))
        self.assertEqual((served.model, served.host), (model, "127.0.0.1"))
        self.assertEqual(served.serve_record.name, f"serve-{served.port}.json")
        # The server stops itself once generate has exited.
        self.assertEqual(environment["SPLASH_APP_PID"], str(os.getpid()))
        self.assertEqual(held, [str(Path(runtime) / "serve.lock")])
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(["generate", "--model", model])
        # --duration, ComfyUI's name, is --seconds.
        command = ["generate", "--model", model, "--prompt", "a cat", "--out", "a.mp4"]
        lengths = [
            launcher.parse_args([*command, flag, "9"]).seconds
            for flag in ("--seconds", "--duration")
        ]
        self.assertEqual(lengths, ["9", "9"])

    def test_generate_defaults_to_10_seconds_at_832x480(self):
        self.assertEqual((H3.default_seconds, H3.size), ("10", "832x480"))
        with (
            mock.patch("sys.stdout", io.StringIO()) as help_text,
            self.assertRaises(SystemExit),
        ):
            launcher.parse_args(["generate", "--help"])
        text = " ".join(help_text.getvalue().split())
        self.assertIn("From the prompt alone: 10 s at 832x480", text)
        self.assertIn("(10 by default)", text)
        self.assertIn("480p 832x480 (the default)", text)

    def test_generate_refuses_a_command_line_that_cannot_make_a_clip_at_once(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(contextlib.chdir(directory))
        Path("first.png").touch()
        Path("cat.png").touch()
        for arguments, message in (
            (["--last-frame", "first.png"], "--last-frame needs --first-frame"),
            (["--first-frame", "first.png", "--reference", "cat.png"],
             "--reference cannot be used with --first-frame or --last-frame"),
            (["--first-frame", "nope.png"], "--first-frame nope.png: no such file"),
            (["--reference", "."], "--reference . is a directory"),
            (["--reference", "cat.png"] * 13, "--reference: at most 12 files"),
            (["--out", "."], "--out . is a directory: name the mp4, such as clip.mp4"),
            (["--out", "nodir/clip.mp4"], "--out nodir/clip.mp4: nodir is not a directory"),
            (["--model", f"{MODEL}:pika-ref2va-6"],
             f"{MODEL}:pika-ref2va-6 takes --reference FILE"),
            (["--model", f"{MODEL}:vdn-8", "--reference", "cat.png"],
             f"--reference takes the pika-ref2va-6 variant: --model {MODEL}:pika-ref2va-6"),
        ):  # fmt: skip
            arguments = [
                "generate",
                "--prompt",
                "a cat",
                "--out",
                "clip.mp4",
                *arguments,
            ]
            with (
                self.subTest(arguments=arguments),
                mock.patch("sys.stderr", io.StringIO()) as error,
                self.assertRaises(SystemExit) as refused,
            ):
                launcher.parse_args(arguments)
            self.assertEqual(refused.exception.code, 2)
            self.assertIn(message, error.getvalue())
        # A pipe, such as <(curl ...), is an input.
        os.mkfifo("pipe")
        piped = launcher.parse_args([
            "generate", "--prompt", "a cat", "--out", "clip.mp4", "--first-frame", "pipe",
        ])  # fmt: skip
        self.assertEqual(piped.first_frame, Path("pipe"))
        os.remove("pipe")
        # A write that fails leaves no temporary file behind.
        with self.assertRaises(IsADirectoryError):
            launcher._write_whole(Path(directory), b"mp4")
        self.assertEqual(sorted(os.listdir(directory)), ["cat.png", "first.png"])

    def test_generate_refuses_before_installing_what_the_clip_would_refuse(self):
        models = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(launcher.paths, "MODELS", models))
        self.enterContext(mock.patch.object(launcher, "RUNTIME_DIR", models))
        cat = models / "cat.png"
        cat.touch()
        for arguments, message in (
            (["--prompt", " "], "--prompt: prompt must be a non-empty string"),
            (["--seed", "4294967296"], "--seed 4294967296: seed must be an integer in [0, 4294967295]"),
            (["--seed", "x"], "--seed x: seed must be an integer"),
            (["--seconds", "20"], "--seconds 20: seconds must be a whole number from 4 to 15"),
            (["--duration", "20"], "--duration 20: seconds must be a whole number"),
            (["--size", "1920x1080"], "--size 1920x1080: size must be WIDTHxHEIGHT"),
            # MiniMax-H3 without a variant installs pika-6, which takes none.
            (["--reference", str(cat)], f"--model {MODEL}:pika-ref2va-6"),
        ):  # fmt: skip
            with (
                self.subTest(arguments=arguments),
                mock.patch.object(launcher, "_ensure_installed") as install,
                mock.patch("sys.stderr", io.StringIO()) as error,
            ):
                status = launcher.main(["generate", "--port", "1", "--model", MODEL,
                                        "--prompt", "a cat", "--out", "clip.mp4",
                                        *arguments])  # fmt: skip
                console_written()
            self.assertEqual(status, 1)
            self.assertIn(message, error.getvalue())
            install.assert_not_called()
        # Another model's length is the clip program's to refuse (a language
        # model's as one).
        language = launcher.model_artifacts.Selection.of(models, "Qwen/Qwen3-8B")
        launcher._refuse_before_install(
            launcher.argparse.Namespace(seconds="20", size=None, reference=[]), language
        )
        # The launcher refuses what server/videos.py refuses, and no more.
        for seconds, size in (("4", "768x768"), ("15", "1536x768"), ("3", "480x960"),
                              ("05", "832x480"), ("16", "864x480"), ("5", "800x480"),
                              ("5", "1568x768"), ("5", "640x320"), ("5", "832X480")):  # fmt: skip
            args = launcher.argparse.Namespace(
                prompt="a cat", seed=None, seconds=seconds, size=size, reference=[]
            )
            selection = launcher.model_artifacts.Selection.of(models, MODEL)
            try:
                videos.canvas(size)
                videos.clip_seconds(seconds)
                accepted = True
            except videos.APIError:
                accepted = False
            with self.subTest(seconds=seconds, size=size):
                try:
                    launcher._refuse_before_install(args, selection)
                    self.assertTrue(accepted)
                except launcher.LauncherError:
                    self.assertFalse(accepted)

    def test_install_installs_as_serve_does_and_starts_nothing(self):
        model = "MiniMaxAI/MiniMax-H3:my-finetune"
        spec = Path(self.enterContext(tempfile.TemporaryDirectory())) / "spec.json"
        spec.write_text("{}")
        for extra, forwarded in (
            ([], None),
            (["--spec", str(spec)], str(spec.resolve())),
        ):
            with (
                self.subTest(extra=extra),
                mock.patch.object(launcher, "_check_port") as probe,
                mock.patch.object(launcher, "_ensure_installed") as install,
                mock.patch.object(launcher.os, "execve") as execute,
            ):
                self.assertEqual(launcher.main(["install", model, *extra]), 0)
            probe.assert_not_called()
            execute.assert_not_called()
            want = launcher.model_artifacts.Selection.of(launcher.paths.MODELS, model)
            install.assert_called_once_with(want, forwarded)
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(["install"])

    def test_arguments_after_the_separator_are_refused_after_each_commands_own(self):
        # serve refuses them last, after SPLASH_PORT and its own options;
        # install, which reads no port, at once.
        for environment, arguments, message in (
            ({"SPLASH_PORT": "bad"}, ["serve", "--model", MODEL_ID], "SPLASH_PORT: "),
            ({}, ["serve", "--model", MODEL_ID, "--mmproj", "x"],
             "--mmproj requires --model-dir"),
            ({}, ["serve", "--model", MODEL_ID], "only supported for coding clients"),
            ({"SPLASH_PORT": "bad"}, ["generate", "--prompt", "p", "--out", "o.mp4"],
             "SPLASH_PORT: "),
            ({}, ["generate", "--prompt", "p", "--out", "o.mp4"],
             "only supported for coding clients"),
            ({"SPLASH_PORT": "bad"}, ["install", MODEL_ID],
             "only supported for coding clients"),
        ):  # fmt: skip
            with (
                self.subTest(arguments=arguments, environment=environment),
                mock.patch.dict(os.environ, environment),
                mock.patch("sys.stderr", io.StringIO()) as error,
                self.assertRaises(SystemExit) as failed,
            ):
                launcher.parse_args([*arguments, "--", "z"])
            self.assertEqual(failed.exception.code, 2)
            self.assertIn(message, error.getvalue())

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
                    "directory or a Hugging Face repository ID",
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


class FakeServer:
    """A server program generate starts (_spawn_server), served in this
    process: its pid, and whether a SIGTERM stopped it."""

    pids = iter(range(1 << 20, 1 << 21))

    def __init__(self, stop=None, returncode=None):
        self.pid = next(self.pids)
        self.stop = stop
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if self.stop is not None:
            self.stop()
        self.returncode = -signal.SIGTERM

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.terminate()


class GenerateOnServerTests(unittest.TestCase):
    """splash generate sends its clip over HTTP: as a job of the splash serve
    of a video model on its port, or else of a server it starts on a free
    port and stops once the clip ends (server/videos.py over the fake
    engine)."""

    def setUp(self):
        self.enterContext(mock.patch.dict(os.environ, {"SPLASH_API_KEY": "key"}))
        os.environ.pop("SPLASH_PORT", None)
        keep_stop_signals(self)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(launcher, "RUNTIME_DIR", self.directory))
        self.enterContext(mock.patch.object(launcher, "POLL_SECONDS", 0.02))
        self.log = self.directory / "log"
        # The servers generate starts, each (command, environment, process,
        # app), and a port no server holds.
        self.spawned = []
        self.port = launcher._free_port()

    def app(self, variants, step_seconds, fail):
        engine = test_videos.engine(self.log, step_seconds, fail)
        variants = {n: H3.variants[n] for n in variants}
        return videos.VideoApp(MODEL, (), engine, test_videos.tokenize, variants)

    def listen(self, app, port):
        """Serve app on 127.0.0.1:port: the server, and what stops it."""
        server = api.FrontendServer(("127.0.0.1", port), app, api_key="key")
        server.RequestHandlerClass = videos.handler(api.FrontendHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        stopped = threading.Event()

        def stop():
            if not stopped.is_set():
                stopped.set()
                server.shutdown()
                server.server_close()
                app.close()

        self.addCleanup(stop)
        return server, stop

    def serve(self, variants=("pika-6",), step_seconds=0.0, fail=False):
        self.served = app = self.app(variants, step_seconds, fail)
        self.server, _ = self.listen(app, 0)
        self.port = self.server.server_port
        # The lock splash serve holds on its port, with its owner record.
        lock = self.enterContext((self.directory / f"serve-{self.port}.lock").open("w"))
        fcntl.flock(lock, fcntl.LOCK_EX)
        json.dump({"pid": os.getpid(), "model": MODEL, "port": self.port}, lock)
        lock.flush()

    def spawn(self, step_seconds=0.0, fail=False, refused=0):
        """_spawn_server serving the clip in this process as server.server
        would, on the command's --port, ready in its --serve-record; the
        first `refused` find their port taken."""

        def spawn(command, environment, held):
            port = int(command[command.index("--port") + 1])
            record = Path(command[command.index("--serve-record") + 1])
            if len(self.spawned) < refused:
                process, app = FakeServer(returncode=1), None
                failure = serve_record.failure(serve_record.PORT, f"{port} is taken")
                state = {"pid": process.pid, "phase": "failed", "failure": failure}
            else:
                app = self.app(("pika-6",), step_seconds, fail)
                process = FakeServer(self.listen(app, port)[1])
                state = {"pid": process.pid, "phase": "ready"}
            record.write_text(json.dumps(state))
            self.spawned.append((command, environment, process, app))
            return process

        return mock.patch.object(launcher, "_spawn_server", side_effect=spawn)

    def generate(self, *arguments, spawn=None):
        with (
            mock.patch.object(launcher, "_ensure_installed") as install,
            mock.patch.object(launcher, "_holds_video_model", return_value=True),
            spawn or mock.patch.object(launcher, "_spawn_server") as started,
            mock.patch.object(launcher.os, "execve") as execute,
            mock.patch("sys.stderr", io.StringIO()) as stderr,
        ):
            status = launcher.main([
                "generate", "--port", str(self.port), "--prompt", "a cat",
                "--out", str(self.directory / "clip.mp4"), *arguments,
            ])  # fmt: skip
            console_written()
        execute.assert_not_called()
        if spawn is None:
            # The clip went to the server on the port: nothing installed or
            # started.
            install.assert_not_called()
            started.assert_not_called()
        return status, stderr.getvalue()

    def own(self, *arguments, **spawn):
        """generate --model with no server on its port: status and stderr."""
        return self.generate("--model", MODEL, *arguments, spawn=self.spawn(**spawn))

    def started(self, spawn):
        """_spawn_server returning spawn's process for each command."""

        def started(command, environment, held):
            self.spawned.append((command, environment, spawn(), None))
            return self.spawned[-1][2]

        return mock.patch.object(launcher, "_spawn_server", side_effect=started)

    def assert_stopped(self):
        """Every server generate started has stopped, its engine with it."""
        self.assertTrue(self.spawned)
        for _, _, process, app in self.spawned:
            self.assertIsNotNone(process.poll())
            if app is not None:
                self.assertTrue(process.terminated)
                self.assertTrue(app.videos.closing)
                self.assertFalse(app.videos.engine.alive())
                self.assertIsNotNone(app.videos.engine.process.poll())

    def test_the_clip_and_its_preview_are_a_job_of_the_running_server(self):
        self.serve(step_seconds=0.05)
        first = self.directory / "first.png"
        first.write_bytes(test_videos.png(832, 480, bytes(832 * 480 * 3)))
        status, stderr = self.generate(
            "--seed", "7", "--size", "832x480", "--preview", "--first-frame", str(first)
        )
        self.assertEqual(status, 0, stderr)
        self.assertEqual((self.directory / "clip.mp4").read_bytes(), SEVEN)
        umask = os.umask(0)
        os.umask(umask)
        self.assertEqual(
            (self.directory / "clip.mp4").stat().st_mode & 0o777, 0o666 & ~umask
        )
        preview = (self.directory / "clip.preview.mp4").read_bytes()
        self.assertEqual(preview, b"preview" + SEVEN)
        (taken,) = test_videos.taken(self.log)
        self.assertEqual((taken["size"], taken["seed"]), ("832x480", 7))
        self.assertIn("first_frame", taken)
        # Each progress line with the time since the job was queued.
        self.assertRegex(stderr, r"Video · denoise · \d+% · \d+ s\n")
        self.assertRegex(
            stderr.splitlines()[-1], r"Video · .*clip.mp4 · 243 frames · 0.0 s$"
        )
        self.assertNotIn(H3.license, stderr)

    def test_another_model_a_variant_not_installed_or_a_failure_stop_in_one_line(self):
        self.serve(fail=True)
        for arguments, message in (
            (["--model", "community/other"], f"Splash serves {MODEL} on port {self.port}, not community/other"),
            (["--model", f"{MODEL}:vdn-8"], "the vdn-8 variant of MiniMaxAI/MiniMax-H3 is not installed"),
            ([], "the video VAE failed"),
        ):  # fmt: skip
            with self.subTest(arguments=arguments):
                status, stderr = self.generate(*arguments)
                self.assertEqual(status, 1)
                self.assertEqual(len(stderr.splitlines()[-1:]), 1)
                self.assertIn(message, stderr.splitlines()[-1])
                self.assertEqual(list(self.directory.glob("*.mp4*")), [])
                self.assertEqual(list(self.directory.glob("tmp*")), [])

    def listing(self, models):
        """_request_json answering /v1/models with `models` from the server
        on the port alone."""
        request = launcher._request_json

        def answer(path, timeout=2, *, port=launcher.serve_options.DEFAULT_PORT):
            return models if port == self.port else request(path, timeout, port=port)

        return mock.patch.object(launcher, "_request_json", side_effect=answer)

    def test_a_language_model_server_is_refused_and_model_starts_one_beside_it(self):
        self.serve()
        language = {"data": [{"id": "community/llm", "owned_by": "splash"}]}
        with self.listing(language):
            status, stderr = self.generate()
            self.assertEqual(status, 1)
            self.assertIn(
                f"community/llm on port {self.port}, a language model", stderr
            )
            status, stderr = self.own()
        self.assertEqual(status, 0, stderr)
        ((command, _, _, _),) = self.spawned
        self.assertEqual(command[4], "server.server")
        self.assertNotEqual(api.parse_args(server_arguments(command)).port, self.port)
        self.assertEqual((self.directory / "clip.mp4").read_bytes(), bytes(8))
        self.assert_stopped()

    def test_a_server_still_starting_is_refused_unless_it_starts_a_language_model(self):
        self.serve()
        with self.listing(None):
            status, stderr = self.generate("--model", MODEL)
            self.assertEqual(status, 1)
            self.assertIn(f"Splash is starting on port {self.port} (PID", stderr)
            lock = self.directory / f"serve-{self.port}.lock"
            lock.write_text(
                json.dumps(
                    {"pid": os.getpid(), "model": "community/llm", "port": self.port}
                )
            )
            self.assertEqual(self.generate()[0], 1)
            status, stderr = self.own()
        self.assertEqual((status, len(self.spawned)), (0, 1), stderr)
        self.assert_stopped()

    def test_a_server_that_stops_answering_ends_the_wait(self):
        self.serve(step_seconds=0.5)
        threading.Timer(
            0.2, lambda: (self.server.shutdown(), self.server.server_close())
        ).start()
        with mock.patch.object(launcher, "SILENCE_SECONDS", 0.2):
            status, stderr = self.generate()
        self.assertEqual(status, 1)
        self.assertIn("stopped answering", stderr)
        self.assertFalse((self.directory / "clip.mp4").exists())

    def test_a_stop_deletes_the_job_and_ends_with_the_status_of_its_signal(self):
        self.serve(step_seconds=0.5)
        stop = launcher.StopSignal(signal.SIGTERM)
        with mock.patch.object(launcher.time, "sleep", side_effect=stop):
            status, stderr = self.generate()
        self.assertEqual(status, 143)
        self.assertIn("Video · cancelling", stderr)
        self.assertEqual(self.served.videos.list({})["data"], [])
        self.assertFalse((self.directory / "clip.mp4").exists())

    def test_without_a_server_generate_starts_one_on_a_free_port_and_stops_it(self):
        status, stderr = self.own(
            "--seconds", "15", "--size", "864x480", "--seed", "7", "--no-audio"
        )
        self.assertEqual(status, 0, stderr)
        self.assertEqual((self.directory / "clip.mp4").read_bytes(), SEVEN)
        self.assertIn(H3.license, stderr)
        ((command, environment, _, _),) = self.spawned
        port = api.parse_args(server_arguments(command)).port
        self.assertNotEqual(port, self.port)
        self.assertIn(f"Splash serves {MODEL} on port {port}", stderr)
        self.assertEqual(environment["SPLASH_APP_PID"], str(os.getpid()))
        (taken,) = test_videos.taken(self.log)
        self.assertEqual((taken["frames"], taken["size"], taken["seed"], taken["outputs"]),
                         (362, "864x480", 7, 0))  # fmt: skip
        self.assert_stopped()
        # Nothing listens on its port any more.
        with socket.socket() as client:
            self.assertNotEqual(client.connect_ex(("127.0.0.1", port)), 0)

    def test_the_own_servers_clip_takes_the_defaults(self):
        status, stderr = self.own()
        self.assertEqual(status, 0, stderr)
        (taken,) = test_videos.taken(self.log)
        self.assertEqual((taken["frames"], taken["size"]), (243, "832x480"))
        self.assertRegex(stderr.splitlines()[-1], r"clip.mp4 · 243 frames")
        self.assert_stopped()

    def test_the_own_server_is_stopped_after_a_failed_job(self):
        status, stderr = self.own(fail=True)
        self.assertEqual(status, 1)
        self.assertIn("the video VAE failed", stderr.splitlines()[-1])
        self.assertFalse((self.directory / "clip.mp4").exists())
        self.assert_stopped()

    def test_the_own_server_is_stopped_after_a_stop(self):
        # The stop arrives as generate waits for the job; the server's own
        # waits keep the time module's sleep.
        stop = launcher.StopSignal(signal.SIGINT)
        clock = mock.Mock(wraps=launcher.time, sleep=mock.Mock(side_effect=stop))
        with mock.patch.object(launcher, "time", clock):
            status, stderr = self.own(step_seconds=0.5)
        self.assertEqual(status, 130)
        self.assertIn("Video · cancelling", stderr)
        ((_, _, _, app),) = self.spawned
        self.assertEqual(app.videos.list({})["data"], [])
        self.assertFalse((self.directory / "clip.mp4").exists())
        self.assert_stopped()

    def test_a_stop_while_the_own_server_starts_stops_it(self):
        stop = launcher.StopSignal(signal.SIGINT)
        with mock.patch.object(launcher.time, "sleep", side_effect=stop):
            status, _ = self.generate("--model", MODEL, spawn=self.started(FakeServer))
        self.assertEqual(status, 130)
        ((_, _, process, _),) = self.spawned
        self.assertTrue(process.terminated)

    def test_a_port_taken_before_the_own_server_binds_it_is_tried_again(self):
        status, stderr = self.own(refused=1)
        self.assertEqual((status, len(self.spawned)), (0, 2), stderr)
        self.assert_stopped()
        self.spawned.clear()
        status, _ = self.own(refused=launcher.PORT_ATTEMPTS)
        self.assertEqual((status, len(self.spawned)), (1, launcher.PORT_ATTEMPTS))
        self.assert_stopped()

    def test_a_server_that_exits_before_it_is_ready_fails_the_clip(self):
        exited = self.started(lambda: FakeServer(returncode=1))
        status, stderr = self.generate("--model", MODEL, spawn=exited)
        self.assertEqual(status, 1)
        self.assertIn("the video server exited with status 1", stderr)

    def test_a_language_model_is_refused_once_installed(self):
        with (
            mock.patch.object(launcher, "_ensure_installed"),
            mock.patch.object(launcher, "_spawn_server") as spawn,
            mock.patch("sys.stderr", io.StringIO()) as stderr,
        ):
            status = launcher.main(["generate", "--port", str(self.port), "--model",
                                    "Qwen/Qwen3-8B", "--prompt", "a cat", "--out",
                                    str(self.directory / "clip.mp4")])  # fmt: skip
            console_written()
        self.assertEqual(status, 1)
        self.assertIn("Qwen/Qwen3-8B is a language model: splash generate takes a video model",
                      stderr.getvalue())  # fmt: skip
        spawn.assert_not_called()
        link = self.directory / "link"
        link.mkdir()
        (link / "model.json").write_text(
            json.dumps({"target_format": H3.target_format})
        )
        self.assertTrue(launcher._holds_video_model(link))

    def test_without_a_server_or_a_model_generate_refuses(self):
        with mock.patch("sys.stderr", io.StringIO()) as stderr:
            status = launcher.main(["generate", "--port", str(self.port), "--prompt",
                                    "a cat", "--out", str(self.directory / "clip.mp4")])  # fmt: skip
            console_written()
        # The status argparse gives a missing argument.
        self.assertEqual(status, 2)
        self.assertIn("--model is required", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

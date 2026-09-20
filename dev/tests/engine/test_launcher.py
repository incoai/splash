import fcntl
import io
import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from install import launcher

MODEL_ID = "community/custom-splash"
MODEL_IDS = (
    "incoai/Qwen3.8-27B-Splash",
    "incoai/Qwen3.6-35B-A3B-Splash",
    "community/custom-splash",
)


class LauncherTests(unittest.TestCase):
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

    def test_size_validation(self):
        for value in ("1G", "1GB", "1GiB", "1073741824"):
            self.assertEqual(launcher._parse_max_memory(value), 1024**3)
        for flag, values in (
            ("--max-context", ("0", "-1", "257K", "bad")),
            ("--max-memory", ("0", "-1G", "bad", str(2**64))),
            ("--max-request-size", ("auto", "0", "-1G", "bad", str(2**64))),
        ):
            for value in values:
                with self.subTest(value=value), mock.patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        launcher.parse_args(
                            ["serve", "--model", MODEL_ID, f"{flag}={value}"]
                        )

    def test_host_controls_probe_and_server_independently_of_allowed_host(self):
        for host in (None, "0.0.0.0", "192.0.2.10", "localhost"):
            with (
                self.subTest(host=host),
                tempfile.TemporaryDirectory() as temporary,
                mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
                mock.patch.object(launcher.socket, "socket") as factory,
                mock.patch.object(launcher, "_ensure_installed"),
                mock.patch.object(launcher.catalog, "spawn_refresh"),
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
                argv = execute.call_args.args[1]
                self.assertEqual(argv[argv.index("--host") + 1], expected)
                self.assertEqual(
                    argv[argv.index("--allowed-host") + 1], "proxy.example"
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
                "port": launcher.PORT,
            }

            def check_install(model):
                self.assertEqual(json.loads(lock_path.read_text()), owner)

            def check_exec(binary, argv, environment):
                refresh.assert_called_once_with()
                self.assertEqual(binary, str(launcher.paths.PYTHON))
                self.assertEqual(argv[argv.index("--max-context") + 1], "102400")
                self.assertEqual(
                    argv[argv.index("--max-memory") + 1], str(28 * 1024**3)
                )
                self.assertEqual(
                    argv[argv.index("--binary") + 1], str(launcher.paths.BINARY)
                )
                self.assertEqual(argv[argv.index("--model") + 1], MODEL_ID)
                self.assertEqual(
                    argv[argv.index("--max-request-size") + 1], str(256 * 1024**2)
                )
                self.assertEqual(
                    argv[-4:],
                    ["--allowed-host", "splash.local", "--allowed-host", "proxy.local"],
                )
                self.assertNotIn("start_new_session", environment)
                self.assertIn("--no-webui", argv)
                self.assertNotIn("test-server-key", argv)
                self.assertEqual(environment["SPLASH_API_KEY"], "test-server-key")
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
                mock.patch.object(launcher.catalog, "spawn_refresh") as refresh,
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
                        "--api-key",
                        "test-server-key",
                        "--no-webui",
                        "--max-request-size",
                        "256M",
                        "--max-context",
                        "100K",
                        "--max-memory",
                        "28G",
                        "--allowed-host",
                        "splash.local",
                        "--allowed-host",
                        "proxy.local",
                    ]
                )
            install.assert_called_once_with(MODEL_ID)
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
                self.assertIn("installation is busy", error.getvalue())
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
                args = launcher.parse_args([name, "--port", "7777"])
                self.assertEqual(args.port, 8123)
                self.assertEqual(args.client_args, ["--port", "7777"])
        for value in ("", "0", "-1", "65536", "invalid", "1.5"):
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {"SPLASH_PORT": value}),
                mock.patch("sys.stderr", io.StringIO()),
            ):
                for arguments in (["serve", "--model", MODEL_ID], ["claude"]):
                    with self.assertRaises(SystemExit):
                        launcher.parse_args(arguments)
                self.assertEqual(
                    launcher.parse_args(
                        ["serve", "--model", MODEL_ID, "--port", "9123"]
                    ).port,
                    9123,
                )
                with self.assertRaises(SystemExit):
                    launcher.parse_args(["serve", "--model", MODEL_ID, "--port", value])

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
                mock.patch.object(launcher.catalog, "spawn_refresh"),
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
            (root / "server/server.py").write_text(
                "import sys\nprint('ready', flush=True)\nsys.stdin.read()\n"
            )
            runtime = root / "runtime"
            script = (
                "import sys\nfrom pathlib import Path\nfrom install import launcher\n"
                "launcher.ROOT = Path(sys.argv[1])\n"
                "launcher.RUNTIME_DIR = launcher.ROOT / 'runtime'\n"
                "launcher.paths.PYTHON = Path(sys.executable)\n"
                "launcher._ensure_installed = lambda model: None\n"
                "launcher.model_artifacts.installed_root = lambda *args: launcher.ROOT\n"
                "launcher.catalog.spawn_refresh = lambda: None\n"
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
                payload = (
                    {"maximum_context_tokens": 102400}
                    if self.path == "/status"
                    else {"data": [{"id": MODEL_ID, "owned_by": "splash"}]}
                )
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
                        {"SPLASH_PORT": str(port), "SPLASH_API_KEY": "test-key"},
                    ),
                    mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
                    mock.patch.object(
                        launcher.clients, "find_executable", return_value="/bin/echo"
                    ),
                    mock.patch.object(
                        launcher.clients, "command", wraps=launcher.clients.command
                    ) as command,
                    mock.patch.object(launcher.os, "execvpe") as execute,
                    mock.patch("sys.stdout", io.StringIO()),
                ):
                    for name in launcher.clients.INSTALL_URLS:
                        launcher.main([name])
                        self.assertEqual(
                            command.call_args.args[2], f"http://127.0.0.1:{port}"
                        )
                        self.assertEqual(
                            command.call_args.args[3:5], (MODEL_ID, 102400)
                        )
                        self.assertEqual(
                            command.call_args.args[5], launcher._runtime_dir(port)
                        )
                    self.assertEqual(execute.call_count, 4)
            finally:
                server.shutdown()
                worker.join(timeout=5)
            self.assertEqual(
                requests,
                [
                    (path, "Bearer test-key")
                    for _ in range(4)
                    for path in ("/status", "/v1/models")
                ],
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
                ):
                    if fail:
                        with self.assertRaisesRegex(
                            launcher.LauncherError, "source build failed"
                        ):
                            launcher._ensure_installed(MODEL_ID)
                    else:
                        launcher._ensure_installed(MODEL_ID)
                self.assertEqual(len(calls), 1 if fail else 3)
                with (runtime / "build.lock").open("a+") as probe:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_packaged_serve_never_invokes_make_or_system_python(self):
        with (
            mock.patch.object(launcher.paths, "PACKAGED", True),
            mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
        ):
            launcher._ensure_installed(MODEL_ID)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[0], str(launcher.paths.PYTHON))
        self.assertIn("prepare", command)
        self.assertNotIn("make", command)

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


if __name__ == "__main__":
    unittest.main()

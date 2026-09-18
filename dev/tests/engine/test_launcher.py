import fcntl
import io
import json
import os
import socket
import subprocess
import tempfile
import unittest
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

    def test_cache_disk_quota(self):
        required = ["serve", "--model", MODEL_ID]
        self.assertEqual(launcher.parse_args(required).max_cache_disk, 0)
        self.assertEqual(
            launcher.parse_args([*required, "--max-cache-disk", "5G"]).max_cache_disk,
            5 * 1024**3,
        )
        # The earlier name still works.
        self.assertEqual(
            launcher.parse_args([*required, "--max-state-disk", "5G"]).max_cache_disk,
            5 * 1024**3,
        )
        for invalid in ("auto", "-1", "nan"):
            with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args([*required, "--max-cache-disk", invalid])

    def test_size_validation(self):
        for value in ("1G", "1GB", "1GiB", "1073741824"):
            self.assertEqual(launcher._parse_max_memory(value), 1024**3)
        for flag, values in (
            ("--max-context", ("0", "-1", "257K", "bad")),
            ("--max-memory", ("0", "-1G", "bad", str(2**64))),
        ):
            for value in values:
                with self.subTest(value=value), mock.patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        launcher.parse_args(
                            ["serve", "--model", MODEL_ID, f"{flag}={value}"]
                        )

    def test_foreground_exec_preserves_terminal_and_holds_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            lock_path = runtime / "serve.lock"
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
                    argv[argv.index("--max-cache-disk") + 1], str(5 * 1024**3)
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
                with (runtime / "serve.lock").open("a+") as other:
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
                        "--max-context",
                        "100K",
                        "--max-memory",
                        "28G",
                        "--max-cache-disk",
                        "5G",
                        "--allowed-host",
                        "splash.local",
                        "--allowed-host",
                        "proxy.local",
                    ]
                )
            install.assert_called_once_with(MODEL_ID)
            execute.assert_called_once()
            self.assertEqual({p.name for p in runtime.iterdir()}, {"serve.lock"})
            with (runtime / "serve.lock").open("a+") as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_duplicate_serve_reports_existing_owner_without_model_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            lock_path = runtime / "serve.lock"
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
                lock_path = runtime / "serve.lock"
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
                with (Path(temporary) / "serve.lock").open("a+") as held:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.assertEqual(launcher.main(["serve", "--model", MODEL_ID]), 1)
                install.assert_not_called()

    def test_real_port_probe_allows_time_wait_but_rejects_live_listener(self):
        for closed in (False, True):
            with (
                self.subTest(closed=closed),
                tempfile.TemporaryDirectory() as temporary,
                socket.socket() as listener,
                socket.socket() as probe,
            ):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.settimeout(2)
                listener.bind(("127.0.0.1", 0))
                address = listener.getsockname()
                listener.listen()
                if closed:
                    with socket.create_connection(address, timeout=2) as client:
                        connection, _ = listener.accept()
                        connection.close()
                        self.assertEqual(client.recv(1), b"")
                    listener.close()
                # Exercise the production probe against a real ephemeral port,
                # without interfering with a user's server on port 8000.
                mapped_probe = mock.Mock(wraps=probe)
                mapped_probe.bind.side_effect = lambda _: probe.bind(address)
                with (
                    mock.patch.object(launcher, "RUNTIME_DIR", Path(temporary)),
                    mock.patch.object(launcher.socket, "socket") as factory,
                    mock.patch.object(launcher, "_ensure_installed") as install,
                    mock.patch.object(launcher.os, "execve") as execute,
                    mock.patch("sys.stderr", io.StringIO()),
                ):
                    factory.return_value.__enter__.return_value = mapped_probe
                    result = launcher.main(["serve", "--model", MODEL_ID])
                if closed:
                    self.assertIsNone(result)
                    install.assert_called_once()
                    execute.assert_called_once()
                else:
                    self.assertEqual(result, 1)
                    install.assert_not_called()
                    execute.assert_not_called()

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
            with (Path(temporary) / "serve.lock").open("a+") as released:
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
            with (Path(temporary) / "serve.lock").open("a+") as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
        install.assert_not_called()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()

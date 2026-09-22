import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import launcher
from server import protocol, server


def local_fixture(root, source=None):
    root.mkdir(parents=True)
    files = {
        "config.json": '{"model_type":"qwen4_exp"}',
        "model.safetensors.index.json": '{"weight_map":{}}',
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "chat_template.jinja": "test template",
    }
    records = []
    for name, text in files.items():
        data = text.encode()
        (root / name).write_bytes(data)
        if source is not None:
            source.mkdir(exist_ok=True)
            (source / name).write_bytes(data)
        records.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (root / "weights").mkdir()
    weights = b"test weights" + bytes(16384 - len(b"test weights"))
    (root / "weights/model.bin").write_bytes(weights)
    if source is not None:
        (source / "model.safetensors").write_bytes(b"original weights")
    manifest = {
        "schema": launcher.LOCAL_SCHEMA,
        "alignment": 16384,
        "source_identity_sha256": "a" * 64,
        "small_files": records,
        "shards": [
            {
                "path": "weights/model.bin",
                "bytes": len(weights),
                "sha256": hashlib.sha256(weights).hexdigest(),
                "source_path": "model.safetensors",
                "source_bytes": 16,
                "source_sha256": hashlib.sha256(b"original weights").hexdigest(),
            }
        ],
    }
    source_records = [
        *records,
        {
            "path": "model.safetensors",
            "bytes": 16,
            "sha256": hashlib.sha256(b"original weights").hexdigest(),
        },
    ]
    manifest["source_identity_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema": launcher.LOCAL_SCHEMA,
                "source_files": sorted(
                    source_records, key=lambda record: record["path"]
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    data = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(data)
    (root / "manifest.sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")
    return root.resolve()


class LocalLauncherTests(unittest.TestCase):
    def test_existing_source_selects_derived_bundle_without_import_or_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original"
            bundle = local_fixture(root / "install/local-models/derived", source)
            with (
                mock.patch.object(launcher, "ROOT", root),
                mock.patch.object(launcher.paths, "PACKAGED", False),
                mock.patch.object(launcher.subprocess, "run") as run,
            ):
                self.assertEqual(launcher._ensure_local_installed(source, None), bundle)
            run.assert_not_called()
            self.assertEqual(
                (source / "model.safetensors").read_bytes(), b"original weights"
            )

    def test_explicit_bundle_must_match_original_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original"
            bundle = local_fixture(root / "bundle", source)
            (source / "tokenizer.json").write_text("changed")
            with self.assertRaisesRegex(
                launcher.LauncherError, "original model metadata"
            ):
                launcher._ensure_local_installed(source, bundle)

    def test_same_size_changed_original_weights_do_not_select_stale_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original"
            bundle = local_fixture(root / "bundle", source)
            (source / "model.safetensors").write_bytes(b"different weight")
            self.assertEqual((source / "model.safetensors").stat().st_size, 16)
            with self.assertRaisesRegex(launcher.LauncherError, "weight checksums"):
                launcher._ensure_local_installed(source, bundle)
            # An explicitly selected imported snapshot is independent of the
            # original directory and still validates without modifying either.
            self.assertEqual(launcher._ensure_local_installed(None, bundle), bundle)

    def test_changed_manifest_or_bundle_metadata_fails_closed(self):
        for name in ("manifest.json", "tokenizer.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                bundle = local_fixture(Path(directory) / "bundle")
                (bundle / name).write_text('{"changed":true}')
                with self.assertRaises(launcher.LauncherError):
                    launcher.local_bundle_manifest(bundle)

    def test_ambiguous_sources_require_explicit_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original"
            local_fixture(root / "install/local-models/first", source)
            local_fixture(root / "install/local-models/second", source)
            with (
                mock.patch.object(launcher, "ROOT", root),
                mock.patch.object(launcher.paths, "PACKAGED", False),
            ):
                with self.assertRaisesRegex(
                    launcher.LauncherError, "multiple local packages"
                ):
                    launcher._ensure_local_installed(source, None)

    def test_local_serve_uses_worker_bundle_and_independent_port_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = local_fixture(root / "bundle")
            worker = root / "build/flash-next/splash-flash"
            worker.parent.mkdir(parents=True)
            worker.write_bytes(b"unused CPU fixture")
            worker.with_name("splash.metallib").write_bytes(b"unused CPU fixture")
            python = root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"unused CPU fixture")
            runtime = root / "runtime"
            runtime.mkdir()
            installation_lock = runtime / "serve.lock"
            installation_lock.write_text("existing installation state")
            with installation_lock.open("a+") as held:
                launcher.fcntl.flock(
                    held, launcher.fcntl.LOCK_SH | launcher.fcntl.LOCK_NB
                )
                with (
                    mock.patch.object(launcher, "ROOT", root),
                    mock.patch.object(launcher.paths, "PYTHON", python),
                    mock.patch.object(launcher, "RUNTIME_DIR", runtime),
                    mock.patch.object(launcher.socket, "socket") as probe,
                    mock.patch.object(launcher, "_ensure_installed") as hub,
                    mock.patch.object(launcher.catalog, "spawn_refresh") as refresh,
                    mock.patch.object(launcher.os, "execve") as execute,
                ):
                    launcher.main(
                        [
                            "serve",
                            "--model",
                            "local/Flash-Next",
                            "--local-package",
                            str(bundle),
                            "--port",
                            "8011",
                        ]
                    )
            argv = execute.call_args.args[1]
            self.assertEqual(argv[argv.index("--local-package") + 1], str(bundle))
            self.assertEqual(argv[argv.index("--tokenizer") + 1], str(bundle))
            self.assertEqual(
                argv[argv.index("--binary") + 1],
                str(root / "build/flash-next/splash-flash"),
            )
            self.assertEqual(argv[argv.index("--port") + 1], "8011")
            probe.return_value.__enter__.return_value.bind.assert_called_once_with(
                ("127.0.0.1", 8011)
            )
            self.assertEqual(
                json.loads((runtime / "serve-8011.lock").read_text())["port"], 8011
            )
            self.assertEqual(
                installation_lock.read_text(), "existing installation state"
            )
            hub.assert_not_called()
            refresh.assert_not_called()

    def test_missing_local_worker_builds_only_the_flash_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "build/flash-next/splash-flash"
            python = root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"unused CPU fixture")

            def build(command, **kwargs):
                self.assertEqual(
                    command, ["make", "-j4", "flash-next", "BUILD=build/flash-next"]
                )
                self.assertEqual(kwargs, {"cwd": root, "check": False})
                worker.parent.mkdir(parents=True)
                worker.write_bytes(b"unused CPU fixture")
                worker.with_name("splash.metallib").write_bytes(b"unused CPU fixture")
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(launcher, "ROOT", root),
                mock.patch.object(launcher.paths, "PYTHON", python),
                mock.patch.object(launcher.paths, "PACKAGED", False),
                mock.patch.object(launcher.subprocess, "run", side_effect=build) as run,
            ):
                launcher._ensure_local_runtime(worker)
                launcher._ensure_local_runtime(worker)
            run.assert_called_once()

    def test_clean_checkout_sets_up_python_before_build_and_packaged_mode_never_builds(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / ".venv/bin/python"
            worker = root / "build/flash-next/splash-flash"
            commands = []

            def setup(command, **_kwargs):
                commands.append(command)
                if "install-environment" in command:
                    python.parent.mkdir(parents=True)
                    python.write_bytes(b"unused CPU fixture")
                else:
                    self.assertTrue(python.is_file())
                    worker.parent.mkdir(parents=True)
                    worker.write_bytes(b"unused CPU fixture")
                    worker.with_name("splash.metallib").write_bytes(
                        b"unused CPU fixture"
                    )
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(launcher, "ROOT", root),
                mock.patch.object(launcher.paths, "PYTHON", python),
                mock.patch.object(launcher.paths, "PACKAGED", False),
                mock.patch.object(launcher.subprocess, "run", side_effect=setup),
            ):
                launcher._ensure_local_runtime(worker)
            self.assertEqual(
                commands,
                [
                    ["make", "platform-check", "install-environment"],
                    ["make", "-j4", "flash-next", "BUILD=build/flash-next"],
                ],
            )
            worker.unlink()
            with (
                mock.patch.object(launcher.paths, "PYTHON", python),
                mock.patch.object(launcher.paths, "PACKAGED", True),
                mock.patch.object(launcher.subprocess, "run") as run,
            ):
                with self.assertRaises(launcher.LauncherError):
                    launcher._ensure_local_runtime(worker)
            run.assert_not_called()

    def test_missing_bundle_import_sanitizes_alias_and_preserves_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Source with spaces Ω"
            source.mkdir()
            (source / "config.json").write_text('{"model_type":"qwen4_exp"}')
            importer = root / "dev/tools/import_flash_next.py"
            importer.parent.mkdir(parents=True)
            importer.write_text("unused CPU fixture")
            packages = root / "install/local-models"
            identity = launcher.model_artifacts.sha256(source / "config.json")[:12]
            collision = packages / f"Source-with-spaces-{identity}"
            collision.mkdir(parents=True)
            (collision / "preserved").write_text("previous snapshot")

            def import_model(command, **kwargs):
                alias = command[command.index("--alias") + 1]
                self.assertRegex(alias, r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
                self.assertEqual(alias, collision.name + "-abcdef")
                local_fixture(packages / alias, source)
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(launcher, "ROOT", root),
                mock.patch.object(launcher.paths, "PACKAGED", False),
                mock.patch.object(launcher.secrets, "token_hex", return_value="abcdef"),
                mock.patch.object(
                    launcher.subprocess, "run", side_effect=import_model
                ) as run,
                mock.patch("sys.stdout", io.StringIO()),
            ):
                selected = launcher._ensure_local_installed(source, None)
            self.assertEqual(selected.name, collision.name + "-abcdef")
            self.assertEqual((collision / "preserved").read_text(), "previous snapshot")
            run.assert_called_once()

    def test_port_and_local_alias_parser_contract(self):
        args = launcher.parse_args(
            [
                "serve",
                "--model",
                "local/Flash-Next",
                "--local-model",
                "/original",
                "--port",
                "8011",
            ]
        )
        self.assertEqual(args.local_model, Path("/original"))
        self.assertEqual(args.port, 8011)
        for value in ("0", "-1", "65536"):
            with self.subTest(port=value), mock.patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    launcher.parse_args(
                        ["serve", "--model", "local/Flash-Next", "--port", value]
                    )

    def test_native_command_selects_local_bundle_and_preserves_legacy_route(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = local_fixture(Path(directory) / "bundle")
            args = server.parse_args(
                [
                    "--local-package",
                    str(bundle),
                    "--model",
                    "local/Flash-Next",
                    "--max-context",
                    "8192",
                    "--max-memory",
                    "32G",
                ]
            )
            self.assertEqual(
                server._native_command(args),
                [
                    str(server.ROOT / "build/flash-next/splash-flash"),
                    "serve-flash-native",
                    str(bundle),
                    "8192",
                    str(32 * 1024**3),
                ],
            )
            self.assertEqual(args.tokenizer, str(bundle))
            for extra in (["target", "draft"], ["--tokenizer", "/different"]):
                with self.subTest(extra=extra), mock.patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        server.parse_args(
                            [
                                "--local-package",
                                str(bundle),
                                "--model",
                                "local/Flash-Next",
                                *extra,
                            ]
                        )
        args = server.parse_args(
            [
                "target",
                "draft",
                "--tokenizer",
                "tokenizer",
                "--model",
                "owner/repository",
            ]
        )
        self.assertEqual(
            server._native_command(args),
            [
                str(server.ROOT / "build/splash"),
                "serve-native",
                "target",
                "draft",
                "auto",
                "auto",
            ],
        )

    def test_local_startup_enforces_native_modalities_and_passes_effective_context(
        self,
    ):
        for modalities in (["text"], ["text", "image"]):
            with (
                self.subTest(modalities=modalities),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle = local_fixture(Path(directory) / "bundle")
                args = server.parse_args(
                    [
                        "--local-package",
                        str(bundle),
                        "--model",
                        "local/Flash-Next",
                        "--port",
                        "8011",
                    ]
                )
                runtime = mock.Mock()
                runtime.readiness = protocol.ReadyEvent(1, 4, 8192, 15)
                runtime.status.return_value = protocol.StatusJsonEvent(
                    1,
                    protocol.STATUS_SCHEMA_VERSION,
                    json.dumps(
                        {"capabilities": {"input_modalities": modalities}}
                    ).encode(),
                )
                backend = mock.Mock()
                http = mock.Mock(server_port=8011)
                http.serve_forever.side_effect = KeyboardInterrupt
                with (
                    mock.patch.object(server, "parse_args", return_value=args),
                    mock.patch.object(server, "load_thinking_key", return_value=None),
                    mock.patch.object(server, "ThinkingCodec"),
                    mock.patch.object(
                        server.AutoTokenizer, "from_pretrained"
                    ) as tokenizer,
                    mock.patch.object(server, "validate_tokenizer"),
                    mock.patch.object(
                        server.engine_runtime,
                        "MultiplexedRuntime",
                        return_value=runtime,
                    ) as worker,
                    mock.patch.object(server, "NativeBackend", return_value=backend),
                    mock.patch.object(server, "ConstraintFactory"),
                    mock.patch.object(server, "Frontend") as frontend,
                    mock.patch.object(server, "FrontendServer", return_value=http),
                    mock.patch.object(server.signal, "signal"),
                    mock.patch("builtins.print"),
                ):
                    if modalities == ["text"]:
                        server.main()
                        self.assertEqual(
                            frontend.call_args.kwargs["input_modalities"], ("text",)
                        )
                        self.assertEqual(frontend.call_args.args[3], 8192)
                        http.server_activate.assert_called_once()
                    else:
                        with self.assertRaises(SystemExit):
                            server.main()
                        frontend.assert_not_called()
                        http.server_activate.assert_not_called()
                tokenizer.assert_called_once_with(
                    str(bundle), local_files_only=True, trust_remote_code=False
                )
                self.assertEqual(
                    worker.call_args.args[0],
                    [args.binary, "serve-flash-native", str(bundle), "auto", "auto"],
                )
                runtime.status.assert_called_once()
                backend.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""Oh My Pi's launcher/configuration contract and installed-client smoke."""

import io
import json
import os
import secrets
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from install import clients, launcher

MODEL = "community/model"


class OmpTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.enterContext(mock.patch.dict(os.environ, {"HOME": temporary.name}))
        self.models = self.home / ".omp/agent/models.yml"

    def command(self, environment=None, arguments=(), port=8000):
        return clients.command(
            "omp",
            "/bin/omp",
            f"http://127.0.0.1:{port}",
            MODEL,
            102400,
            {} if environment is None else environment,
            input_modalities=["text", "image", "pdf"],
            client_args=arguments,
        )

    def test_launch_configures_the_served_model_without_saving_the_key(self):
        original = {"SPLASH_API_KEY": "!private-$key", "OTHER": "kept"}
        argv, environment = self.command(original)
        self.assertEqual(argv, ["/bin/omp", "--provider", "splash", "--model", MODEL])
        self.assertEqual(environment, original)
        self.assertNotIn("!private-$key", self.models.read_text())
        provider = yaml.safe_load(self.models.read_text())["providers"]["splash"]
        self.assertEqual(provider["baseUrl"], "http://127.0.0.1:8000/v1")
        self.assertEqual(provider["api"], "openai-completions")
        self.assertEqual(provider["apiKey"], "SPLASH_API_KEY")
        self.assertEqual(provider["models"][0]["id"], MODEL)
        self.assertEqual(provider["models"][0]["input"], ["text", "image"])
        self.assertEqual(provider["models"][0]["contextWindow"], 102400)
        self.assertEqual(provider["models"][0]["maxTokens"], 25600)
        self.assertEqual(self.models.stat().st_mode & 0o777, 0o600)
        self.assertFalse((self.home / ".pi").exists())

    def test_updates_preserve_other_providers_and_choose_omps_active_file(self):
        self.models.parent.mkdir(parents=True)
        other = {"baseUrl": "https://example.test/v1", "apiKey": "OTHER_KEY"}
        for suffix in (".yml", ".yaml", ".json"):
            with self.subTest(suffix=suffix):
                for existing in self.models.parent.glob("models.*"):
                    existing.unlink()
                source = self.models.with_suffix(suffix)
                # JSON is also valid YAML; OMP migrates it when no YAML exists.
                source.write_text(
                    '{"providers": {"other": {"baseUrl": '
                    '"https://example.test/v1", "apiKey": "OTHER_KEY"}}, '
                    '"future": {"keep": true}}'
                )
                self.command()
                self.command(port=8001)
                active = source if suffix == ".yaml" else self.models
                config = yaml.safe_load(active.read_text())
                self.assertEqual(config["providers"]["other"], other)
                self.assertEqual(config["future"], {"keep": True})
                self.assertEqual(
                    set(config["providers"]), {"other", "splash", "splash-8001"}
                )
                if suffix == ".json":
                    self.assertEqual(
                        yaml.safe_load(source.read_text())["providers"],
                        {"other": other},
                    )

    def test_configuration_follows_the_selected_agent_directory_and_profile(self):
        cases = (
            (
                {"PI_CODING_AGENT_DIR": str(self.home / "custom")},
                [],
                self.home / "custom",
            ),
            (
                {"OMP_PROFILE": "work", "PI_CODING_AGENT_DIR": "/unused"},
                [],
                self.home / ".omp/profiles/work/agent",
            ),
            ({"PI_PROFILE": "legacy"}, [], self.home / ".omp/profiles/legacy/agent"),
            ({"OMP_PROFILE": "", "PI_PROFILE": "legacy"}, [], self.home / ".omp/agent"),
            (
                {
                    "OMP_PROFILE": "default",
                    "PI_PROFILE": "work",
                    "PI_CODING_AGENT_DIR": str(self.home / ".omp/profiles/work/agent"),
                },
                [],
                self.home / ".omp/agent",
            ),
            (
                {
                    "OMP_PROFILE": "work",
                    "PI_CODING_AGENT_DIR": str(self.home / ".omp/profiles/work/agent"),
                },
                ["--profile", "default"],
                self.home / ".omp/agent",
            ),
            (
                {"OMP_PROFILE": "work"},
                ["--profile", "other"],
                self.home / ".omp/profiles/other/agent",
            ),
            (
                {"PI_CONFIG_DIR": ".custom-omp"},
                ["--profile=work"],
                self.home / ".custom-omp/profiles/work/agent",
            ),
            (
                {"PI_CONFIG_DIR": str(self.home / "custom-omp")},
                [],
                self.home / str(self.home).lstrip("/") / "custom-omp/agent",
            ),
        )
        for environment, arguments, directory in cases:
            with self.subTest(environment=environment, arguments=arguments):
                for existing in self.home.rglob("models.yml"):
                    existing.unlink()
                argv, _ = self.command(environment, arguments)
                self.assertTrue((directory / "models.yml").is_file())
                if arguments:
                    self.assertEqual(argv[-len(arguments) :], arguments)

    def test_config_only_writes_the_profile_without_launching(self):
        model = {"id": MODEL, "owned_by": "splash", "input_modalities": ["text"]}
        with (
            mock.patch.object(
                launcher,
                "_running_status",
                return_value={"maximum_context_tokens": 102400},
            ),
            mock.patch.object(
                launcher, "_request_json", return_value={"data": [model]}
            ),
            mock.patch.object(clients, "find_executable", return_value="/bin/omp"),
            mock.patch.object(launcher.os, "execvpe") as execute,
            mock.patch("sys.stdout", io.StringIO()) as output,
            mock.patch.dict(
                os.environ,
                {
                    "OMP_PROFILE": "",
                    "PI_CODING_AGENT_DIR": str(self.home / "configured"),
                },
            ),
        ):
            self.assertEqual(launcher.main(["omp", "--config-only"]), 0)
            execute.assert_not_called()
            self.assertIn("Configured omp", output.getvalue())
            self.assertTrue((self.home / "configured/models.yml").is_file())
        self.assertEqual(
            launcher.parse_args(["omp", "--config", "extra.yml"]).client_args,
            ["--config", "extra.yml"],
        )
        self.assertEqual(
            launcher.parse_args(["omp", "--", "--config-only"]).client_args,
            ["--config-only"],
        )

    def test_invalid_configuration_is_not_overwritten(self):
        self.models.parent.mkdir(parents=True)
        for original in (
            "[broken",
            "[]",
            "null",
            "providers: []",
            "providers: null",
            "!!python/object:unsafe {}",
        ):
            with self.subTest(original=original):
                self.models.write_text(original)
                with self.assertRaisesRegex(clients.ClientError, "Invalid OMP"):
                    self.command()
                self.assertEqual(self.models.read_text(), original)

    def test_yaml_strings_keep_omps_yaml_12_meaning(self):
        self.models.parent.mkdir(parents=True)
        self.models.write_text(
            "providers:\n  other:\n    apiKey: yes\n    models:\n      - id: off\n        name: 2026-09-29\n"
        )
        self.command()
        # YAML 1.2 (OMP) keeps these plain scalars as strings; PyYAML's
        # normal object loader would convert them to bools and dates.
        config = yaml.load(self.models.read_text(), Loader=yaml.BaseLoader)
        other = config["providers"]["other"]
        self.assertEqual(other["apiKey"], "yes")
        self.assertEqual(other["models"][0], {"id": "off", "name": "2026-09-29"})

    def test_invalid_profiles_do_not_create_configuration(self):
        for arguments in (
            ["--profile"],
            ["--profile", "../elsewhere"],
            ["--profile=BAD"],
            ["--profile=con"],
            ["--profile=work."],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(clients.ClientError):
                    self.command(arguments=arguments)
                self.assertFalse(self.models.parent.exists())

    def test_symlink_and_atomic_replacement_preserve_the_users_file(self):
        target = self.home / "dotfiles/models.yml"
        target.parent.mkdir()
        original = "providers: {}\n"
        target.write_text(original)
        self.models.parent.mkdir(parents=True)
        self.models.symlink_to(target)
        with mock.patch.object(Path, "replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.command()
        self.assertEqual(target.read_text(), original)
        self.assertEqual(list(target.parent.iterdir()), [target])
        self.command()
        self.assertTrue(self.models.is_symlink())
        self.assertIn("splash", yaml.safe_load(target.read_text())["providers"])


@unittest.skipUnless(
    os.environ.get("SPLASH_OMP_BINARY"), "set SPLASH_OMP_BINARY for OMP HTTP test"
)
class InstalledOmpTests(unittest.TestCase):
    def test_text_tools_and_resume_against_splash_http(self):
        from dev.tests.agent_real import executed_commands
        from dev.tests.test_server import (
            FakeRuntime,
            FakeTokenizer,
            Harness,
            Plan,
            _byte_backend,
        )

        tokenizer = FakeTokenizer()
        tokenizer.fragments[5] = (
            "<tool_call>\n<function=bash>\n<parameter=command>\n"
            "printf 'omp-tool-ok'\n</parameter>\n<parameter=i>\nPrint the test marker\n</parameter>\n</function>\n</tool_call>\n"
        )
        tokenizer.backend_tokenizer = _byte_backend(tokenizer.fragments)
        runtime = FakeRuntime(Plan([[5]]), Plan([[4]]), Plan([[26, 4]]))
        api_key = "!test-$" + secrets.token_hex(16)
        harness = Harness(
            runtime,
            tokenizer=tokenizer,
            max_context=131072,
            timeout=60,
            api_key=api_key,
            model=MODEL,
        )
        self.addCleanup(harness.close)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "project"
            work.mkdir()
            environment = {
                key: os.environ[key] for key in ("PATH", "TMPDIR") if key in os.environ
            }
            environment.update(
                HOME=directory,
                PI_CODING_AGENT_DIR=str(work / "agent"),
                PI_OFFLINE="1",
                PI_TELEMETRY="0",
                SPLASH_API_KEY=api_key,
            )
            argv, environment = clients.command(
                "omp",
                os.environ["SPLASH_OMP_BINARY"],
                f"http://127.0.0.1:{harness.server.server_port}",
                MODEL,
                131072,
                environment,
                input_modalities=["text"],
                client_args=[
                    "--print",
                    "--mode",
                    "json",
                    "--tools",
                    "bash",
                    "--approval-mode",
                    "yolo",
                    "--no-pty",
                    "--no-title",
                    "--no-lsp",
                    "--no-extensions",
                    "--no-skills",
                    "--no-rules",
                ],
            )
            session = None
            for thinking in ("off", "low"):
                result = subprocess.run(
                    [
                        *argv,
                        "--thinking",
                        thinking,
                        *(["--resume", session] if session else []),
                    ],
                    input="Reply briefly.\n",
                    cwd=work,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                parsed = [
                    json.loads(line)
                    for line in result.stdout.splitlines()
                    if line.startswith("{")
                ]
                header = next(
                    event for event in parsed if event.get("type") == "session"
                )
                if session:
                    self.assertEqual(header["id"], session)
                session = header["id"]
                messages = [
                    event["message"]
                    for event in parsed
                    if event.get("type") == "message_end"
                    and event.get("message", {}).get("role") == "assistant"
                ]
                self.assertEqual(messages[-1]["stopReason"], "stop", result.stdout)
                self.assertIn(
                    "plain answer",
                    " ".join(item.get("text", "") for item in messages[-1]["content"]),
                )
                if thinking == "off":
                    self.assertEqual(
                        executed_commands("omp", parsed), ["printf 'omp-tool-ok'"]
                    )
            self.assertEqual(len(runtime.requests), 3)
            self.assertEqual(
                [options["enable_thinking"] for _, options in tokenizer.templates],
                [False, False, True],
            )


if __name__ == "__main__":
    unittest.main()

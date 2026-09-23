import io
import json
import os
import secrets
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import clients, launcher


class PiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root / "pi home"
        self.config = self.home / "models.json"
        self.environment = {"PI_CODING_AGENT_DIR": str(self.home)}

    def command(self, **kwargs):
        return clients.command(
            "pi",
            "/bin/pi",
            "http://127.0.0.1:18997/",
            "org/model",
            102400,
            self.root / "runtime",
            self.environment,
            **kwargs,
        )

    def test_configures_served_model_and_preserves_environment_and_arguments(self):
        original = dict(self.environment)
        arguments = ["--thinking", "low", "--", "literal"]
        argv, env = self.command(client_args=arguments)
        self.assertEqual(
            argv,
            ["/bin/pi", "--provider", "splash", "--model", "org/model", *arguments],
        )
        self.assertEqual(self.environment, original)
        self.assertEqual(env["PI_CODING_AGENT_DIR"], str(self.home))
        provider = json.loads(self.config.read_text())["providers"]["splash"]
        self.assertEqual(provider["baseUrl"], "http://127.0.0.1:18997/v1")
        self.assertEqual(provider["api"], "openai-completions")
        self.assertEqual(provider["apiKey"], "local")
        model = provider["models"][0]
        self.assertEqual(model["id"], "org/model")
        self.assertEqual(model["contextWindow"], 102400)
        self.assertEqual(model["maxTokens"], 25600)
        self.assertEqual(model["input"], ["text", "image"])
        self.assertTrue(model["reasoning"])
        self.assertEqual(model["thinkingLevelMap"], {"off": "none"})

    def test_merges_other_providers_without_changing_settings_or_auth(self):
        self.home.mkdir()
        original = {
            "providers": {
                "other": {"models": [{"id": "keep"}]},
                "splash": {"baseUrl": "old"},
            },
            "extra": {"keep": True},
        }
        self.config.write_text(json.dumps(original))
        for name in ("settings.json", "auth.json"):
            (self.home / name).write_text('{"keep":true}')
        self.command()
        self.command()
        config = json.loads(self.config.read_text())
        self.assertEqual(config["providers"]["other"], original["providers"]["other"])
        self.assertEqual(config["extra"], original["extra"])
        self.assertEqual(len(config["providers"]["splash"]["models"]), 1)
        for name in ("settings.json", "auth.json"):
            self.assertEqual((self.home / name).read_text(), '{"keep":true}')
        self.assertEqual(self.config.stat().st_mode & 0o777, 0o600)

    def test_key_is_an_environment_reference_not_saved_or_passed_in_argv(self):
        self.environment["SPLASH_API_KEY"] = "!test-$secret"
        argv, env = self.command()
        self.assertNotIn("!test-$secret", self.config.read_text())
        self.assertNotIn("!test-$secret", " ".join(argv))
        self.assertEqual(env["SPLASH_API_KEY"], "!test-$secret")
        self.assertEqual(
            json.loads(self.config.read_text())["providers"]["splash"]["apiKey"],
            "$SPLASH_API_KEY",
        )

    def test_invalid_config_is_preserved(self):
        self.home.mkdir()
        for value in (
            b"broken",
            b"[]",
            b"null",
            b'{"providers":[]}',
            b'{"providers":null}',
            b"\xff",
        ):
            with self.subTest(value=value):
                self.config.write_bytes(value)
                with self.assertRaisesRegex(clients.ClientError, "Invalid Pi config"):
                    self.command()
                self.assertEqual(self.config.read_bytes(), value)

    def test_failed_atomic_replace_preserves_original_and_removes_temporary(self):
        self.home.mkdir()
        self.config.write_text("{}")
        with mock.patch.object(Path, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.command()
        self.assertEqual(self.config.read_text(), "{}")
        self.assertEqual(list(self.home.iterdir()), [self.config])

    def test_default_and_tilde_agent_directory(self):
        with mock.patch.object(Path, "home", return_value=self.root):
            self.assertEqual(
                clients.pi_config_path({}), self.root / ".pi/agent/models.json"
            )
        with mock.patch.dict(os.environ, {"HOME": str(self.root)}):
            self.assertEqual(
                clients.pi_config_path({"PI_CODING_AGENT_DIR": "~/custom"}),
                self.root / "custom/models.json",
            )

    def test_launch_and_config_only_use_ready_server_and_selected_port(self):
        for config_only in (False, True):
            with (
                self.subTest(config_only=config_only),
                mock.patch.dict(
                    os.environ,
                    {
                        **self.environment,
                        "SPLASH_PORT": "18997",
                        "SPLASH_API_KEY": "test-key",
                    },
                ),
                mock.patch.object(
                    clients, "find_executable", return_value="/bin/pi"
                ) as find,
                mock.patch.object(
                    launcher,
                    "_running_status",
                    return_value={"ready": True, "maximum_context_tokens": 65536},
                ) as status,
                mock.patch.object(
                    launcher,
                    "_request_json",
                    return_value={
                        "data": [{"id": "actual/model", "owned_by": "splash"}]
                    },
                ) as models,
                mock.patch.object(launcher.os, "execvpe") as execute,
                mock.patch("sys.stdout", io.StringIO()) as output,
            ):
                launcher.main(
                    ["pi", "--config"] if config_only else ["pi", "--print", "hello"]
                )
                status.assert_called_once_with(18997)
                models.assert_called_once_with("/v1/models", port=18997)
                if config_only:
                    execute.assert_not_called()
                    find.assert_not_called()
                    self.assertIn(str(self.config), output.getvalue())
                else:
                    find.assert_called_once_with("pi")
                    self.assertEqual(
                        execute.call_args.args[1],
                        [
                            "/bin/pi",
                            "--provider",
                            "splash",
                            "--model",
                            "actual/model",
                            "--print",
                            "hello",
                        ],
                    )
                provider = json.loads(self.config.read_text())["providers"]["splash"]
                self.assertEqual(provider["models"][0]["contextWindow"], 65536)
                self.assertEqual(provider["baseUrl"], "http://127.0.0.1:18997/v1")

    def test_config_only_requires_ready_server_without_writing(self):
        with (
            mock.patch.dict(os.environ, self.environment),
            mock.patch.object(launcher, "_running_status", return_value=None),
            mock.patch.object(launcher.os, "execvpe") as execute,
            mock.patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(launcher.main(["pi", "--config"]), 1)
        execute.assert_not_called()
        self.assertFalse(self.config.exists())

    def test_config_option_is_scoped_and_separator_keeps_pi_flags(self):
        self.assertTrue(launcher.parse_args(["pi", "--config"]).config_only)
        args = launcher.parse_args(["pi", "--", "--config"])
        self.assertFalse(args.config_only)
        self.assertEqual(args.client_args, ["--config"])
        args = launcher.parse_args(["codex", "--config", 'model="other"'])
        self.assertFalse(args.config_only)
        self.assertEqual(args.client_args, ["--config", 'model="other"'])
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(["pi", "--config", "ignored prompt"])


@unittest.skipUnless(
    os.environ.get("SPLASH_PI_BINARY"), "set SPLASH_PI_BINARY for Pi HTTP test"
)
class InstalledPiTests(unittest.TestCase):
    def test_text_tools_and_session_resume_against_splash_http(self):
        from dev.tests.agent_real import events, executed_commands
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
            "printf 'pi-tool-ok'\n</parameter>\n</function>\n</tool_call>\n"
        )
        tokenizer.backend_tokenizer = _byte_backend(tokenizer.fragments)
        runtime = FakeRuntime(Plan([[5]]), Plan([[4]]), Plan([[26, 4]]))
        test_key = "!test-$" + secrets.token_hex(16)
        harness = Harness(
            runtime,
            tokenizer=tokenizer,
            max_context=131072,
            timeout=60,
            api_key=test_key,
        )
        self.addCleanup(harness.close)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            env = {
                key: os.environ[key] for key in ("PATH", "TMPDIR") if key in os.environ
            }
            env.update(
                HOME=directory,
                PI_CODING_AGENT_DIR=str(work / "pi"),
                PI_OFFLINE="1",
                PI_TELEMETRY="0",
                SPLASH_API_KEY=test_key,
            )
            argv, env = clients.command(
                "pi",
                os.environ["SPLASH_PI_BINARY"],
                f"http://127.0.0.1:{harness.server.server_port}",
                "test-model",
                131072,
                work / "runtime",
                env,
                client_args=[
                    "--print",
                    "--mode",
                    "json",
                    "--no-extensions",
                    "--no-skills",
                    "--no-prompt-templates",
                    "--no-themes",
                    "--no-context-files",
                    "--thinking",
                    "off",
                ],
            )
            session = None
            for index in range(2):
                arguments = argv if index == 0 else [*argv[:-1], "low"]
                result = subprocess.run(
                    [*arguments, *(["--session", session] if session else [])],
                    input="Reply briefly.\n",
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                parsed = events(result.stdout)
                header = next(e for e in parsed if e.get("type") == "session")
                if session:
                    self.assertEqual(header["id"], session)
                session = header["id"]
                messages = [
                    e["message"]
                    for e in parsed
                    if e.get("type") == "message_end"
                    and e.get("message", {}).get("role") == "assistant"
                ]
                self.assertTrue(messages, result.stdout + result.stderr)
                self.assertEqual(
                    messages[-1]["stopReason"], "stop", result.stdout + result.stderr
                )
                self.assertTrue(
                    any(
                        c.get("text", "").strip() == "plain answer"
                        for c in messages[-1]["content"]
                    ),
                    result.stdout,
                )
                if index == 0:
                    self.assertEqual(
                        executed_commands("pi", parsed), ["printf 'pi-tool-ok'"]
                    )
            self.assertEqual(len(runtime.requests), 3)
            self.assertEqual(
                [options["enable_thinking"] for _, options in tokenizer.templates],
                [False, False, True],
            )

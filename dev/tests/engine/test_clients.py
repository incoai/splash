import http.server
import io
import json
import os
import secrets
import signal
import subprocess
import tempfile
import threading
import tomllib
import unittest
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from unittest import mock

import yaml

from install import clients, launcher

MODEL = "incoai/Qwen3.6-35B-A3B-Splash"


class ClientTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runtime = Path(directory.name)
        # Pi's default models.json is in the home directory; no launch here
        # touches the developer's.
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.home = Path(home.name)
        self.enterContext(mock.patch.dict(os.environ, {"HOME": home.name}))
        self.pi_models = self.home / ".pi/agent/models.json"

    def command(
        self,
        name,
        context=102400,
        model=MODEL,
        env=None,
        client_args=(),
        client_version=None,
        input_modalities=None,
    ):
        return clients.command(
            name,
            f"/bin/{name}",
            "http://127.0.0.1:8000/",
            model,
            context,
            self.runtime,
            {} if env is None else env,
            client_args=client_args,
            client_version=client_version,
            input_modalities=["text", "image", "pdf"]
            if input_modalities is None
            else input_modalities,
        )

    def test_server_key_reaches_every_provider_without_entering_argv(self):
        for name in clients.INSTALL_URLS:
            with self.subTest(client=name):
                argv, env = self.command(
                    name, env={"SPLASH_API_KEY": "test-server-key"}
                )
                self.assertNotIn("test-server-key", " ".join(argv))
                if name == "claude":
                    self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "test-server-key")
                elif name == "codex":
                    self.assertEqual(env["SPLASH_API_KEY"], "test-server-key")
                elif name == "opencode":
                    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
                    self.assertEqual(
                        config["provider"]["splash"]["options"]["apiKey"],
                        "test-server-key",
                    )
                elif name == "pi":
                    # Pi reads the key from the environment for each request.
                    self.assertEqual(env["SPLASH_API_KEY"], "test-server-key")
                    self.assertNotIn("test-server-key", self.pi_models.read_text())
                    config = json.loads(self.pi_models.read_text())
                    self.assertEqual(
                        config["providers"]["splash"]["apiKey"], "$SPLASH_API_KEY"
                    )
                else:
                    self.assertEqual(env["OPENAI_API_KEY"], "test-server-key")
                    profile = Path(env["HERMES_HOME"]) / "config.yaml"
                    self.assertEqual(
                        yaml.safe_load(profile.read_text())["model"]["api_key"],
                        "test-server-key",
                    )
                    self.assertEqual(profile.stat().st_mode & 0o777, 0o600)

    def test_missing_clients_have_actionable_install_message(self):
        for name in clients.INSTALL_URLS:
            with (
                self.subTest(name=name),
                mock.patch.object(clients.shutil, "which", return_value=None),
            ):
                with self.assertRaisesRegex(
                    clients.ClientError, "not installed.*PATH"
                ) as error:
                    clients.find_executable(name)
                self.assertIn(clients.INSTALL_URLS[name], str(error.exception))

    def test_context_is_required_not_guessed(self):
        for context in (None, 0, -1, "100000", True):
            for name in clients.INSTALL_URLS:
                with self.subTest(name=name, context=context):
                    with self.assertRaisesRegex(clients.ClientError, "context"):
                        self.command(name, context=context)

    def test_model_is_required(self):
        for model in (None, "", 123):
            with (
                self.subTest(model=model),
                self.assertRaisesRegex(clients.ClientError, "model"),
            ):
                self.command("codex", model=model)

    def test_unknown_client_is_rejected(self):
        with self.assertRaisesRegex(clients.ClientError, "Unknown coding client"):
            self.command("aider")

    def test_default_environment_is_a_copy_of_the_process_environment(self):
        with mock.patch.dict(os.environ, {"SPLASH_API_KEY": "process-key"}):
            before = dict(os.environ)
            _, env = clients.command(
                "codex",
                "/bin/codex",
                "http://127.0.0.1:8000",
                MODEL,
                102400,
                self.runtime,
                input_modalities=["text"],
            )
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(env, before)

    def test_claude_launch_selects_the_local_server_for_every_model_alias(self):
        # A shell configured for a cloud provider keeps its unrelated
        # settings, and compaction stays on with the served window.
        command, env = self.command(
            "claude",
            env={
                "ANTHROPIC_API_KEY": "unrelated-key",
                "CLAUDE_CODE_USE_VERTEX": "1",
                "CLAUDE_CONFIG_DIR": "/custom/claude",
                "PATH": "/bin",
            },
        )
        self.assertEqual(
            command,
            [
                "/bin/claude",
                "--disallowedTools",
                "WebSearch",
                "--model",
                MODEL,
                "--permission-mode",
                "default",
            ],
        )
        self.assertEqual(
            env,
            {
                "CLAUDE_CONFIG_DIR": "/custom/claude",
                "PATH": "/bin",
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
                "ANTHROPIC_AUTH_TOKEN": "local",
                "ANTHROPIC_MODEL": MODEL,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": MODEL,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": MODEL,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL,
                "ANTHROPIC_SMALL_FAST_MODEL": MODEL,
                "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "102400",
                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "102400",
                "CLAUDE_CODE_USE_BEDROCK": "0",
                "CLAUDE_CODE_USE_VERTEX": "0",
                "CLAUDE_CODE_USE_FOUNDRY": "0",
            },
        )

    def test_opencode_launch_registers_the_served_model_for_every_agent(self):
        argv, env = self.command("opencode", env={"PATH": "/bin"})
        self.assertEqual(argv, ["/bin/opencode"])
        self.assertEqual(env.keys(), {"PATH", "OPENCODE_CONFIG_CONTENT"})
        self.assertEqual(env["PATH"], "/bin")
        served = f"splash/{MODEL}"
        self.assertEqual(
            json.loads(env["OPENCODE_CONFIG_CONTENT"]),
            {
                "model": served,
                "small_model": served,
                "agent": {
                    name: {"model": served}
                    for name in (
                        "build",
                        "plan",
                        "general",
                        "explore",
                        "title",
                        "compaction",
                    )
                },
                "provider": {
                    "splash": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Splash",
                        "options": {
                            "baseURL": "http://127.0.0.1:8000/v1",
                            "apiKey": "local",
                        },
                        "models": {
                            MODEL: {
                                "name": MODEL,
                                "reasoning": True,
                                # Efforts are offered, none is selected.
                                "variants": {
                                    "none": {"reasoningEffort": "none"},
                                    "low": {"reasoningEffort": "low"},
                                    "medium": {"reasoningEffort": "medium"},
                                    "high": {"reasoningEffort": "high"},
                                    "xhigh": {"reasoningEffort": "xhigh"},
                                },
                                "attachment": True,
                                "modalities": {
                                    "input": ["text", "image", "pdf"],
                                    "output": ["text"],
                                },
                                "limit": {
                                    "context": 102400,
                                    "input": 76800,
                                    "output": 25600,
                                },
                            }
                        },
                    }
                },
            },
        )

    def test_codex_launch_passes_the_connection_as_root_config_overrides(self):
        argv, env = self.command("codex", env={"PATH": "/bin"})
        self.assertEqual(
            argv,
            [
                "/bin/codex",
                "-c",
                f'model="{MODEL}"',
                "-c",
                'web_search="disabled"',
                "-c",
                'model_provider="splash"',
                "-c",
                'model_providers.splash={name="Splash",'
                'base_url="http://127.0.0.1:8000/v1",'
                'env_key="SPLASH_API_KEY",wire_api="responses"}',
                "-c",
                "model_context_window=102400",
                "-c",
                "model_auto_compact_token_limit=92160",
            ],
        )
        self.assertEqual(env, {"PATH": "/bin", "SPLASH_API_KEY": "local"})
        self.assertEqual(
            tomllib.loads("\n".join(argv[2::2])),
            {
                "model": MODEL,
                "web_search": "disabled",
                "model_provider": "splash",
                "model_providers": {
                    "splash": {
                        "name": "Splash",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "env_key": "SPLASH_API_KEY",
                        "wire_api": "responses",
                    }
                },
                "model_context_window": 102400,
                "model_auto_compact_token_limit": 92160,
            },
        )
        self.assertEqual(list(self.runtime.iterdir()), [])

    def test_hermes_launch_writes_a_private_profile(self):
        argv, env = self.command("hermes", env={"PATH": "/bin"})
        home = self.runtime / "hermes"
        self.assertEqual(
            argv,
            ["/bin/hermes", "chat", "--provider", "custom", "--model", MODEL],
        )
        self.assertEqual(
            env,
            {
                "PATH": "/bin",
                "HERMES_HOME": str(home),
                "CUSTOM_BASE_URL": "http://127.0.0.1:8000/v1",
                "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
                "OPENAI_API_KEY": "local",
            },
        )
        self.assertEqual(
            yaml.safe_load((home / "config.yaml").read_text()),
            {
                "model": {
                    "default": MODEL,
                    "provider": "custom",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "api_key": "local",
                    "api_mode": "chat_completions",
                    "supports_vision": True,
                    "context_length": 102400,
                    "max_tokens": 25600,
                }
            },
        )
        self.assertEqual(home.stat().st_mode & 0o777, 0o700)
        self.assertEqual({path.name for path in home.iterdir()}, {"config.yaml"})

    def test_pi_launch_adds_the_served_model_to_the_users_pi_models(self):
        argv, env = self.command("pi", env={"PATH": "/bin"})
        self.assertEqual(argv, ["/bin/pi", "--provider", "splash", "--model", MODEL])
        self.assertEqual(env, {"PATH": "/bin"})
        self.assertEqual(
            json.loads(self.pi_models.read_text()),
            {
                "providers": {
                    "splash": {
                        "baseUrl": "http://127.0.0.1:8000/v1",
                        "api": "openai-completions",
                        "apiKey": "local",
                        "models": [
                            {
                                "id": MODEL,
                                "reasoning": True,
                                "thinkingLevelMap": {"off": "none"},
                                "input": ["text", "image"],
                                "contextWindow": 102400,
                                "maxTokens": 25600,
                            }
                        ],
                    }
                }
            },
        )
        agent = self.pi_models.parent
        self.assertEqual(agent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.pi_models.stat().st_mode & 0o777, 0o600)
        self.assertEqual([path.name for path in agent.iterdir()], ["models.json"])
        self.assertEqual(list(self.runtime.iterdir()), [])

    def test_pi_names_a_provider_per_port(self):
        self.command("pi")
        argv, _ = clients.command(
            "pi",
            "/bin/pi",
            "http://127.0.0.1:8001/",
            MODEL,
            102400,
            self.runtime,
            {},
            input_modalities=["text"],
        )
        self.assertEqual(argv[:3], ["/bin/pi", "--provider", "splash-8001"])
        providers = json.loads(self.pi_models.read_text())["providers"]
        self.assertEqual(
            {name: provider["baseUrl"] for name, provider in providers.items()},
            {
                "splash": "http://127.0.0.1:8000/v1",
                "splash-8001": "http://127.0.0.1:8001/v1",
            },
        )

    def test_pi_models_follow_the_agent_directory_setting(self):
        self.command("pi", env={"PI_CODING_AGENT_DIR": "~/custom agent"})
        path = self.home / "custom agent/models.json"
        self.assertEqual(
            json.loads(path.read_text())["providers"]["splash"]["models"][0]["id"],
            MODEL,
        )
        self.assertFalse(self.pi_models.exists())

    def test_pi_update_replaces_only_the_splash_provider(self):
        agent = self.pi_models.parent
        agent.mkdir(parents=True)
        other = {"baseUrl": "http://other/v1", "models": [{"id": "keep"}]}
        self.pi_models.write_text(
            json.dumps(
                {
                    "providers": {"other": other, "splash": {"baseUrl": "stale"}},
                    "future": {"keep": True},
                }
            )
        )
        for name in ("settings.json", "auth.json"):
            (agent / name).write_text('{"keep": true}')
        self.command("pi")
        self.command("pi", context=262144, model="incoai/Qwen3.8-27B-Splash")
        config = json.loads(self.pi_models.read_text())
        self.assertEqual(config["providers"]["other"], other)
        self.assertEqual(config["future"], {"keep": True})
        splash = config["providers"]["splash"]
        self.assertEqual(splash["baseUrl"], "http://127.0.0.1:8000/v1")
        self.assertEqual(
            [(m["id"], m["contextWindow"], m["maxTokens"]) for m in splash["models"]],
            [("incoai/Qwen3.8-27B-Splash", 262144, 32768)],
        )
        for name in ("settings.json", "auth.json"):
            self.assertEqual((agent / name).read_text(), '{"keep": true}')
        self.assertEqual(
            {path.name for path in agent.iterdir()},
            {"models.json", "settings.json", "auth.json"},
        )

    def test_invalid_pi_models_are_not_overwritten(self):
        self.pi_models.parent.mkdir(parents=True)
        for models in (
            b"broken",
            b"[]",
            b"null",
            b'{"providers": []}',
            b'{"providers": null}',
            # Pi strips comments; rewriting the file would drop them.
            b'{"providers": {}} // mine\n',
            b"\xff",
        ):
            self.pi_models.write_bytes(models)
            with (
                self.subTest(models=models),
                self.assertRaisesRegex(clients.ClientError, "Invalid Pi models.json"),
            ):
                self.command("pi")
            self.assertEqual(self.pi_models.read_bytes(), models)

    def test_pi_updates_a_symlinked_models_file_in_place(self):
        dotfiles = self.home / "dotfiles"
        dotfiles.mkdir()
        (dotfiles / "models.json").write_text('{"providers": {}}')
        self.pi_models.parent.mkdir(parents=True)
        self.pi_models.symlink_to(dotfiles / "models.json")
        self.command("pi")
        self.assertTrue(self.pi_models.is_symlink())
        config = json.loads((dotfiles / "models.json").read_text())
        self.assertEqual(config["providers"]["splash"]["models"][0]["id"], MODEL)
        self.assertEqual([path.name for path in dotfiles.iterdir()], ["models.json"])

    def test_opencode_preserves_unrelated_inline_config(self):
        user = {
            "permission": {"bash": "ask"},
            "mcp": {"test": {"type": "local"}},
            "provider": {"other": {"name": "Other"}},
        }
        original = {"OPENCODE_CONFIG_CONTENT": json.dumps(user), "CUSTOM": "kept"}
        before = dict(original)
        _, env = self.command("opencode", env=original)
        self.assertEqual(original, before)
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(config["permission"], user["permission"])
        self.assertEqual(config["mcp"], user["mcp"])
        self.assertEqual(config["provider"]["other"], user["provider"]["other"])
        self.assertEqual(config["model"], config["small_model"])
        provider = config["provider"]["splash"]
        self.assertEqual(provider["options"]["baseURL"], "http://127.0.0.1:8000/v1")
        self.assertEqual(
            provider["models"][MODEL]["limit"]["context"],
            102400,
        )

    def test_opencode_preserves_inline_variant_definitions_and_selection(self):
        variants = {
            "low": {"reasoningEffort": "low", "temperature": 0.2},
            "xhigh": {"disabled": True},
            "brief": {"reasoningEffort": "none"},
        }
        user = {
            "agent": {"build": {"variant": "brief", "prompt": "Custom prompt"}},
            "provider": {"splash": {"models": {MODEL: {"variants": variants}}}},
        }
        original = {"OPENCODE_CONFIG_CONTENT": json.dumps(user)}
        before = dict(original)
        args = ["run", "--variant", "none", "A prompt"]
        argv, env = self.command("opencode", env=original, client_args=args)
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        configured = config["provider"]["splash"]["models"][MODEL]["variants"]
        self.assertEqual(configured["none"], {"reasoningEffort": "none"})
        self.assertEqual(configured["medium"], {"reasoningEffort": "medium"})
        self.assertEqual(configured["high"], {"reasoningEffort": "high"})
        for name, value in variants.items():
            self.assertEqual(configured[name], value)
        self.assertEqual(config["agent"]["build"]["variant"], "brief")
        self.assertEqual(config["agent"]["build"]["prompt"], "Custom prompt")
        self.assertEqual(argv, ["/bin/opencode", *args])
        self.assertEqual(original, before)

    def test_opencode_points_built_in_agents_at_the_served_model(self):
        # An agent-level model in the user's global config outranks the
        # top-level one, so a stale pin would send every request to a model
        # the server does not serve.
        user = {
            "agent": {
                "build": {
                    "model": "splash/incoai/Qwen3.8-27B-Splash",
                    "variant": "off",
                },
                "title": {"model": "splash/incoai/Qwen3.8-27B-Splash"},
                "compaction": {"model": "other/cloud-model", "temperature": 0.2},
                "reviewer": {"model": "other/model", "prompt": "review"},
            }
        }
        _, env = self.command(
            "opencode", env={"OPENCODE_CONFIG_CONTENT": json.dumps(user)}
        )
        agents = json.loads(env["OPENCODE_CONFIG_CONTENT"])["agent"]
        for name in ("build", "plan", "general", "explore", "title", "compaction"):
            self.assertEqual(agents[name]["model"], f"splash/{MODEL}")
        self.assertEqual(agents["build"]["variant"], "off")
        self.assertEqual(agents["compaction"]["temperature"], 0.2)
        self.assertEqual(agents["reviewer"], user["agent"]["reviewer"])

    def test_opencode_reports_shared_input_output_budget(self):
        # The output allowance is a quarter of the window, from 1 to 32768.
        for context, output in ((3, 1), (4096, 1024), (262144, 32768)):
            with self.subTest(context=context):
                _, env = self.command("opencode", context=context)
                config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
                self.assertEqual(
                    config["provider"]["splash"]["models"][MODEL]["limit"],
                    {"context": context, "input": context - output, "output": output},
                )
                self.assertNotIn("compaction", config)

    def test_clients_offer_the_served_input_modalities(self):
        for modalities in (["text", "image", "pdf"], ["text"]):
            vision = "image" in modalities
            with self.subTest(modalities=modalities):
                _, env = self.command("opencode", input_modalities=modalities)
                model = json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"][
                    "splash"
                ]["models"][MODEL]
                self.assertIs(model["attachment"], vision)
                self.assertEqual(
                    model["modalities"], {"input": modalities, "output": ["text"]}
                )
                _, env = self.command("hermes", input_modalities=modalities)
                profile = Path(env["HERMES_HOME"]) / "config.yaml"
                config = yaml.safe_load(profile.read_text())
                self.assertIs(config["model"]["supports_vision"], vision)
                # Pi's schema has no PDF input.
                self.command("pi", input_modalities=modalities)
                config = json.loads(self.pi_models.read_text())
                self.assertEqual(
                    config["providers"]["splash"]["models"][0]["input"],
                    ["text", "image"] if vision else ["text"],
                )
        # Claude Code and Codex configurations declare no input modalities.
        for name in ("claude", "codex"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.command(name, input_modalities=["text", "image", "pdf"]),
                    self.command(name, input_modalities=["text"]),
                )

    def test_server_without_text_input_modalities_predates_the_launcher(self):
        for name in clients.INSTALL_URLS:
            for modalities in ([], ["image"], "text", [None], ["text", 3]):
                with self.subTest(name=name, modalities=modalities):
                    with self.assertRaisesRegex(clients.ClientError, "restart it"):
                        self.command(name, input_modalities=modalities)

    def test_opencode_preserves_user_compaction_preferences(self):
        compaction = {"auto": True, "reserved": 12000, "prune": False}
        _, env = self.command(
            "opencode",
            env={"OPENCODE_CONFIG_CONTENT": json.dumps({"compaction": compaction})},
        )
        self.assertEqual(
            json.loads(env["OPENCODE_CONFIG_CONTENT"])["compaction"], compaction
        )

    def test_invalid_opencode_inline_config_is_not_discarded(self):
        # Every object the launcher reads or extends must be one.
        for config in (
            "invalid",
            "[]",
            "null",
            '"text"',
            {"agent": []},
            {"agent": None},
            {"agent": {"plan": "custom"}},
            {"agent": {"compaction": []}},
            {"provider": []},
            {"provider": {"splash": []}},
            {"provider": {"splash": {"models": 1}}},
            {"provider": {"splash": {"models": {MODEL: []}}}},
            {"provider": {"splash": {"models": {MODEL: {"variants": None}}}}},
        ):
            value = config if isinstance(config, str) else json.dumps(config)
            with (
                self.subTest(config=value),
                self.assertRaisesRegex(clients.ClientError, "OPENCODE_CONFIG_CONTENT"),
            ):
                self.command("opencode", env={"OPENCODE_CONFIG_CONTENT": value})

    def test_opencode_accepts_any_value_outside_the_entries_it_extends(self):
        user = {"agent": {"reviewer": "custom"}, "provider": {"other": []}}
        _, env = self.command(
            "opencode", env={"OPENCODE_CONFIG_CONTENT": json.dumps(user)}
        )
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(config["agent"]["reviewer"], "custom")
        self.assertEqual(config["provider"]["other"], [])
        # The served provider replaces a user's splash entry as a whole.
        user = {"provider": {"splash": {"models": {"other-model": []}, "npm": 1}}}
        _, env = self.command(
            "opencode", env={"OPENCODE_CONFIG_CONTENT": json.dumps(user)}
        )
        _, default = self.command("opencode")
        self.assertEqual(
            json.loads(env["OPENCODE_CONFIG_CONTENT"])["provider"]["splash"],
            json.loads(default["OPENCODE_CONFIG_CONTENT"])["provider"]["splash"],
        )

    def test_opencode_two_requests_a_private_server_for_the_inline_config(self):
        # OpenCode 2 loads OPENCODE_CONFIG_CONTENT inside its server process;
        # a persistent background service started earlier never sees this
        # environment, so the provider block must go to a private server.
        for version in (2, 3):
            with self.subTest(version=version):
                argv, env = self.command("opencode", client_version=version)
                self.assertEqual(argv, ["/bin/opencode", "--standalone"])
                self.assertIn("OPENCODE_CONFIG_CONTENT", env)

    def test_opencode_one_or_unknown_versions_keep_the_environment_launch(self):
        # OpenCode 1 reads the environment in-process and rejects the flag;
        # an unparsable or failed probe must not change the launch either.
        for version in (None, 0, 1, True, "2", 1.5):
            with self.subTest(version=version):
                argv, _ = self.command("opencode", client_version=version)
                self.assertEqual(argv, ["/bin/opencode"])

    def test_opencode_two_respects_a_user_selected_server(self):
        cases = {
            ("--standalone",): ["/bin/opencode", "--standalone"],
            ("--server", "http://127.0.0.1:9999"): [
                "/bin/opencode",
                "--server",
                "http://127.0.0.1:9999",
            ],
            ("--server=http://127.0.0.1:9999",): [
                "/bin/opencode",
                "--server=http://127.0.0.1:9999",
            ],
        }
        for args, expected in cases.items():
            with self.subTest(args=args):
                argv, _ = self.command(
                    "opencode", client_args=list(args), client_version=2
                )
                self.assertEqual(argv, expected)

    def test_opencode_two_places_private_server_flag_after_the_subcommand(self):
        args = ["run", "A prompt"]
        argv, _ = self.command("opencode", client_args=args, client_version=2)
        self.assertEqual(argv, ["/bin/opencode", *args, "--standalone"])

    def test_opencode_two_preserves_the_end_of_options_separator(self):
        args = ["run", "--", "--server"]
        argv, _ = self.command("opencode", client_args=args, client_version=2)
        self.assertEqual(
            argv, ["/bin/opencode", "run", "--standalone", "--", "--server"]
        )

    def test_version_probe_parses_client_output_and_fails_closed(self):
        def completed(stdout, returncode=0):
            return subprocess.CompletedProcess(
                ["opencode", "--version"], returncode, stdout=stdout, stderr=""
            )

        outputs = {
            "1.18.31\n": 1,
            "2.0.12": 2,
            "opencode v2.0.12\n": 2,
            "v2.0.12": 2,
            "opencode v2.0.12-beta.1\n": 2,
            "opencode 10.0.1 (build 7)\n": 10,
            "": None,
            "unknown\n": None,
            "2\n": None,
            "warning: requires macOS 26.4\n": None,
        }
        for stdout, expected in outputs.items():
            with self.subTest(stdout=stdout):
                with mock.patch.object(
                    clients.subprocess, "run", return_value=completed(stdout)
                ) as run:
                    self.assertEqual(
                        clients.probe_major_version("/bin/opencode"), expected
                    )
                self.assertEqual(run.call_args.args[0], ["/bin/opencode", "--version"])
        with mock.patch.object(
            clients.subprocess, "run", return_value=completed("2.0.12", 1)
        ):
            self.assertIsNone(clients.probe_major_version("/bin/opencode"))
        for error in (
            FileNotFoundError,
            PermissionError,
            subprocess.TimeoutExpired(cmd="opencode", timeout=5),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"),
        ):
            with (
                self.subTest(error=error),
                mock.patch.object(clients.subprocess, "run", side_effect=error),
            ):
                self.assertIsNone(clients.probe_major_version("/bin/opencode"))

    def test_hermes_profile_preserves_preferences_and_sessions_on_model_change(self):
        _, env = self.command("hermes")
        home = Path(env["HERMES_HOME"])
        path = home / "config.yaml"
        config = yaml.safe_load(path.read_text())
        config["display"] = {"interface": "tui"}
        config["mcp_servers"] = {"test": {"command": "test-server"}}
        path.write_text(yaml.safe_dump(config))
        (home / "state.db").write_bytes(b"session data")
        self.command("hermes", context=262144, model="incoai/Qwen3.8-27B-Splash")
        changed = yaml.safe_load(path.read_text())
        self.assertEqual(changed["model"]["context_length"], 262144)
        self.assertEqual(changed["model"]["max_tokens"], 32768)
        self.assertEqual(changed["model"]["default"], "incoai/Qwen3.8-27B-Splash")
        self.assertTrue(changed["model"]["supports_vision"])
        self.assertEqual(changed["display"], config["display"])
        self.assertEqual(changed["mcp_servers"], config["mcp_servers"])
        self.assertEqual((home / "state.db").read_bytes(), b"session data")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual({p.name for p in home.iterdir()}, {"state.db", "config.yaml"})

    def test_invalid_hermes_profile_is_not_overwritten(self):
        home = self.runtime / "hermes"
        home.mkdir()
        path = home / "config.yaml"
        for profile in (
            b"model: broken\n",
            b"model: false\n",
            b"- a list\n",
            b"model: [\n",
            b"created: 2026-13-01\n",
            b"\xff\n",
        ):
            path.write_bytes(profile)
            with (
                self.subTest(profile=profile),
                self.assertRaisesRegex(clients.ClientError, "Invalid Hermes profile"),
            ):
                self.command("hermes")
            self.assertEqual(path.read_bytes(), profile)

    def test_empty_hermes_profile_is_configured_from_scratch(self):
        home = self.runtime / "hermes"
        home.mkdir()
        path = home / "config.yaml"
        for profile, kept in (
            ("", {}),
            ("~\n", {}),
            ("model:\n", {}),
            ("display: {interface: tui}\n", {"display": {"interface": "tui"}}),
        ):
            path.write_text(profile)
            with self.subTest(profile=profile):
                self.command("hermes")
                configured = yaml.safe_load(path.read_text())
                self.assertEqual(configured.pop("model")["default"], MODEL)
                self.assertEqual(configured, kept)

    def test_failed_profile_replacement_keeps_the_previous_profile(self):
        home = self.runtime / "hermes"
        home.mkdir()
        path = home / "config.yaml"
        path.write_text("display: {interface: tui}\n")
        with (
            mock.patch.object(Path, "replace", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            self.command("hermes")
        self.assertEqual(path.read_text(), "display: {interface: tui}\n")
        self.assertEqual(list(home.iterdir()), [path])

    def test_no_client_filters_tools_bypasses_permissions_or_changes_cwd(self):
        cwd = Path.cwd()
        original = {"PATH": "/bin", "USER_SETTING": "keep"}
        for name in clients.INSTALL_URLS:
            with self.subTest(name=name):
                argv, env = self.command(name, env=original)
                self.assertEqual(env["USER_SETTING"], "keep")
                for word in (
                    "--tools",
                    "--toolsets",
                    "skip-permissions",
                    "bypassPermissions",
                    "--bare",
                    "--safe-mode",
                    "--system-prompt",
                ):
                    self.assertNotIn(word, " ".join(argv))
        self.assertEqual(Path.cwd(), cwd)
        self.assertEqual(original, {"PATH": "/bin", "USER_SETTING": "keep"})

    def test_codex_combines_user_and_launcher_config_before_subcommands(self):
        for prefix in (
            [],
            ["exec"],
            ["resume", "--last"],
            ["exec", "resume", "id"],
            ["review"],
            ["exec", "review"],
        ):
            for flag in (
                ["-c", 'model_reasoning_effort="low"'],
                ["--config", 'model_reasoning_effort="low"'],
                ['--config=model_reasoning_effort="low"'],
                ['-cmodel_reasoning_effort="low"'],
                ['-c=model_reasoning_effort="low"'],
            ):
                with self.subTest(prefix=prefix, flag=flag):
                    args = [*prefix, *flag, "a prompt with -c and --config words"]
                    argv, _ = self.command("codex", client_args=args)
                    self.assertEqual(argv[-len(prefix) - 1 :], [*prefix, args[-1]])
                    self.assertLess(
                        argv.index('model_provider="splash"'),
                        argv.index('model_reasoning_effort="low"'),
                    )
                    self.assertNotIn("-c", argv[-len(prefix) - 1 :])
                    self.assertEqual(args, [*prefix, *flag, args[-1]])

    def test_codex_preserves_config_precedence_and_literal_arguments(self):
        argv, _ = self.command(
            "codex",
            client_args=[
                "-c",
                "model_auto_compact_token_limit=18000",
                "exec",
                "-c",
                "model_auto_compact_token_limit=24000",
                "--",
                "-c",
                "literal",
            ],
        )
        values = [
            argv[index + 1] for index, value in enumerate(argv[:-4]) if value == "-c"
        ]
        self.assertEqual(
            [
                value
                for value in values
                if value.startswith("model_auto_compact_token_limit=")
            ],
            [
                "model_auto_compact_token_limit=92160",
                "model_auto_compact_token_limit=18000",
                "model_auto_compact_token_limit=24000",
            ],
        )
        self.assertEqual(argv[-4:], ["exec", "--", "-c", "literal"])
        with self.assertRaisesRegex(clients.ClientError, "requires a config override"):
            self.command("codex", client_args=["exec", "-c"])

    def test_other_clients_preserve_passthrough_arguments(self):
        for name in ("claude", "opencode", "hermes", "pi"):
            with self.subTest(name=name):
                args = ["--help", "--", "literal"]
                argv, _ = self.command(name, client_args=args)
                self.assertEqual(argv[-len(args) :], args)


class ClientLifecycleTests(unittest.TestCase):
    @contextmanager
    def ready_server(self, client, context=102400, model=None):
        """Serve model from a ready server; yields the patched exec."""
        if model is None:
            model = {
                "id": MODEL,
                "owned_by": "splash",
                "input_modalities": ["text", "image", "pdf"],
            }
        status = {"ready": True, "maximum_context_tokens": context}
        with (
            tempfile.TemporaryDirectory() as runtime,
            mock.patch.object(launcher, "PROFILES_DIR", Path(runtime)),
            mock.patch.object(
                clients, "find_executable", return_value=f"/bin/{client}"
            ),
            mock.patch.object(launcher, "_running_status", return_value=status),
            mock.patch.object(
                launcher, "_request_json", return_value={"data": [model]}
            ),
            mock.patch.object(launcher.os, "execvpe") as execute,
            mock.patch("sys.stdout", io.StringIO()),
        ):
            yield execute

    def test_clients_accept_arguments_with_or_without_separator(self):
        payloads = (
            [],
            ["--help"],
            ["--resume"],
            ["-r", "session-id"],
            ["resume", "--last"],
            ["--max-context", "100K"],
            ["run", "--variant", "none", "Explain this project"],
            ["exec", "-c", 'model_reasoning_effort="low"', "--", "-c", "literal"],
        )
        for name, separator, payload in product(
            clients.INSTALL_URLS, ([], ["--"]), payloads
        ):
            with self.subTest(name=name, separator=separator, payload=payload):
                args = launcher.parse_args([name, *separator, *payload])
                self.assertEqual(args.command, name)
                self.assertEqual(args.client_args, payload)

    def test_clients_remove_only_one_leading_separator(self):
        for name in clients.INSTALL_URLS:
            with self.subTest(name=name):
                args = launcher.parse_args([name, "--", "--", "-c", "literal"])
                self.assertEqual(args.client_args, ["--", "-c", "literal"])

    def test_serve_keeps_strict_argument_validation(self):
        for payload in (["--resume"], ["--", "--resume"], ["claude", "--resume"]):
            with (
                self.subTest(payload=payload),
                mock.patch("sys.stderr", io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                launcher.parse_args(["serve", "--model", "community/model", *payload])
            self.assertEqual(error.exception.code, 2)

    def test_splash_help_does_not_require_a_server(self):
        for arguments in (["--help"], ["serve", "--help"]):
            with (
                self.subTest(arguments=arguments),
                mock.patch.object(launcher, "_running_status") as status,
                mock.patch("sys.stdout", io.StringIO()),
                self.assertRaises(SystemExit) as result,
            ):
                launcher.main(arguments)
            self.assertEqual(result.exception.code, 0)
            status.assert_not_called()

    def test_missing_client_is_checked_before_connection(self):
        with (
            mock.patch.object(clients.shutil, "which", return_value=None),
            mock.patch.object(launcher, "_running_status") as status,
            mock.patch("sys.stderr", io.StringIO()) as error,
        ):
            self.assertEqual(launcher.main(["claude"]), 1)
        status.assert_not_called()
        self.assertIn("claude is not installed", error.getvalue())

    def test_ready_server_is_source_of_model_and_context(self):
        for context, separator in product((102400, 262144), ([], ["--"])):
            with (
                self.subTest(context=context, separator=separator),
                self.ready_server("codex", context=context) as execute,
            ):
                launcher.main(
                    [
                        "codex",
                        *separator,
                        "exec",
                        "-c",
                        'model_reasoning_effort="low"',
                        "hello",
                    ]
                )
            argv = execute.call_args.args[1]
            self.assertEqual(argv[1:3], ["-c", f'model="{MODEL}"'])
            self.assertIn(f"model_context_window={context}", argv)
            self.assertLess(
                argv.index('model_reasoning_effort="low"'), argv.index("exec")
            )
            self.assertEqual(argv[-2:], ["exec", "hello"])

    def test_opencode_launch_probes_the_installed_version(self):
        for version, expected in (
            (2, ["/bin/opencode", "--standalone"]),
            (1, ["/bin/opencode"]),
            (None, ["/bin/opencode"]),
        ):
            with (
                self.subTest(version=version),
                mock.patch.object(
                    clients, "probe_major_version", return_value=version
                ) as probe,
                self.ready_server("opencode") as execute,
            ):
                launcher.main(["opencode"])
            self.assertEqual(execute.call_args.args[1], expected)
            probe.assert_called_once_with("/bin/opencode")

    def test_launcher_configures_clients_from_the_served_input_modalities(self):
        for reported in (["text", "image", "pdf"], ["text"], None):
            model = {"id": MODEL, "owned_by": "splash"}
            if reported is not None:
                model["input_modalities"] = reported
            with (
                self.subTest(modalities=reported),
                mock.patch.object(clients, "probe_major_version", return_value=1),
                self.ready_server("opencode", model=model) as execute,
                mock.patch("sys.stderr", io.StringIO()) as error,
            ):
                result = launcher.main(["opencode"])
            if reported is None:
                # A server started by an older Splash reports only vision.
                self.assertEqual(result, 1)
                self.assertIn(
                    "restart it with this version of Splash", error.getvalue()
                )
                execute.assert_not_called()
                continue
            config = json.loads(execute.call_args.args[2]["OPENCODE_CONFIG_CONTENT"])
            served = config["provider"]["splash"]["models"][model["id"]]
            self.assertIs(served["attachment"], "image" in reported)
            self.assertEqual(served["modalities"]["input"], reported)

    def test_unready_server_never_probes_or_launches(self):
        with (
            mock.patch.object(clients, "find_executable", return_value="/bin/opencode"),
            mock.patch.object(clients, "probe_major_version") as probe,
            mock.patch.object(launcher, "_running_status", return_value=None),
            mock.patch.object(launcher.os, "execvpe") as execute,
            mock.patch("sys.stderr", io.StringIO()),
        ):
            self.assertEqual(launcher.main(["opencode"]), 1)
        probe.assert_not_called()
        execute.assert_not_called()

    def test_version_probe_is_scoped_to_opencode(self):
        for name in ("claude", "codex", "hermes"):
            with (
                self.subTest(name=name),
                mock.patch.object(clients, "probe_major_version") as probe,
                self.ready_server(name) as execute,
            ):
                launcher.main([name])
            execute.assert_called_once()
            probe.assert_not_called()

    def test_source_checkout_keeps_hermes_sessions_out_of_build(self):
        # make clean removes build/, and a Hermes home there with it.
        self.assertFalse(launcher.paths.PACKAGED)
        model = {"id": MODEL, "owned_by": "splash", "input_modalities": ["text"]}
        status = {"ready": True, "maximum_context_tokens": 102400}
        for port in (launcher.PORT, launcher.PORT + 1):
            with (
                self.subTest(port=port),
                mock.patch.dict(os.environ, {"SPLASH_PORT": str(port)}),
                mock.patch.object(
                    clients, "find_executable", return_value="/bin/hermes"
                ),
                # The launcher's own home; nothing is written into the checkout.
                mock.patch.object(clients, "_write_hermes_profile") as write,
                mock.patch.object(launcher, "_running_status", return_value=status),
                mock.patch.object(
                    launcher, "_request_json", return_value={"data": [model]}
                ),
                mock.patch.object(launcher.os, "execvpe") as execute,
                mock.patch("sys.stdout", io.StringIO()),
            ):
                launcher.main(["hermes"])
            home = Path(execute.call_args.args[2]["HERMES_HOME"])
            write.assert_called_once_with(home, mock.ANY)
            self.assertTrue(home.is_relative_to(launcher.ROOT))
            self.assertFalse(home.is_relative_to(launcher.ROOT / "build"))

    def test_unready_server_never_launches_or_downloads(self):
        for payload in ([], ["--help"]):
            with (
                self.subTest(payload=payload),
                mock.patch.object(
                    clients, "find_executable", return_value="/bin/claude"
                ),
                mock.patch.object(launcher, "_running_status", return_value=None),
                mock.patch.object(launcher, "_ensure_installed") as install,
                mock.patch.object(launcher.os, "execvpe") as execute,
                mock.patch("sys.stderr", io.StringIO()) as error,
            ):
                self.assertEqual(launcher.main(["claude", *payload]), 1)
            self.assertIn("splash serve", error.getvalue())
            execute.assert_not_called()
            install.assert_not_called()

    def test_unidentified_server_never_launches_client(self):
        for catalog in (
            None,
            {},
            {"data": None},
            {"data": []},
            {"data": [{"id": "other", "owned_by": "other"}]},
        ):
            with (
                self.subTest(catalog=catalog),
                mock.patch.object(
                    clients, "find_executable", return_value="/bin/codex"
                ),
                mock.patch.object(
                    launcher, "_running_status", return_value={"ready": True}
                ),
                mock.patch.object(launcher, "_request_json", return_value=catalog),
                mock.patch.object(launcher.os, "execvpe") as execute,
                mock.patch("sys.stderr", io.StringIO()),
            ):
                self.assertEqual(launcher.main(["codex"]), 1)
            execute.assert_not_called()


@unittest.skipUnless(
    os.environ.get("SPLASH_OPENCODE_BINARY"),
    "set SPLASH_OPENCODE_BINARY for private-server configuration test",
)
class InstalledOpenCodeTests(unittest.TestCase):
    def test_private_server_configuration_is_isolated_from_the_existing_service(self):
        binary = str(Path(os.environ["SPLASH_OPENCODE_BINARY"]).resolve())
        version = clients.probe_major_version(binary)
        self.assertIsNotNone(version)
        if version < 2:
            self.skipTest("private servers require OpenCode 2")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            environment = {"PATH": os.environ["PATH"]}
            for name in ("config", "data", "cache", "state"):
                environment[f"XDG_{name.upper()}_HOME"] = str(work / name)
            (work / "tmp").mkdir()
            environment["TMPDIR"] = str(work / "tmp")

            def run(argv, env):
                result = subprocess.run(
                    argv,
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout

            query = [binary, "api", "GET", "/api/config"]
            model = "incoai/Qwen3.8-27B-Splash"
            argv, configured = clients.command(
                "opencode",
                binary,
                "http://127.0.0.1:18997",
                model,
                102400,
                work / "runtime",
                environment,
                client_version=version,
                client_args=query[1:],
                input_modalities=["text", "image", "pdf"],
            )
            try:
                run([binary, "service", "start"], environment)
                before = json.loads(run(query, environment))
                self.assertEqual(json.loads(run(query, configured)), before)
                sources = json.loads(run(argv, configured))
                info = next(
                    source["info"]
                    for source in sources
                    if source.get("type") == "document"
                    and "splash" in source["info"].get("providers", {})
                )
                self.assertEqual(
                    info["model"], {"providerID": "splash", "model": model}
                )
                self.assertEqual(
                    info["providers"]["splash"]["settings"]["baseURL"],
                    "http://127.0.0.1:18997/v1",
                )
                self.assertEqual(json.loads(run(query, environment)), before)
            finally:
                run([binary, "service", "stop"], environment)


@unittest.skipUnless(
    os.environ.get("SPLASH_CODEX_BINARY"),
    "set SPLASH_CODEX_BINARY for CLI routing test",
)
class InstalledCodexTests(unittest.TestCase):
    def test_truncated_response_has_a_bounded_client_outcome(self):
        from dev.tests.test_server import FakeRuntime, Harness, Plan

        # The installed client and production HTTP adapter are real. A
        # controlled length stop keeps this boundary test independent of model text.
        runtime = FakeRuntime(*(Plan([[4]], reason="length") for _ in range(32)))
        harness = Harness(runtime, max_context=131072, default_max_new=16, timeout=60)
        self.addCleanup(harness.close)
        base_url = f"http://127.0.0.1:{harness.server.server_port}"
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "codex").mkdir()
            environment = {
                key: os.environ[key]
                for key in ("PATH", "HOME", "TMPDIR")
                if key in os.environ
            }
            environment.update(
                CODEX_HOME=str(work / "codex"),
                HTTP_PROXY=base_url,
                HTTPS_PROXY=base_url,
                ALL_PROXY=base_url,
                NO_PROXY="127.0.0.1,localhost",
            )
            argv, environment = clients.command(
                "codex",
                os.environ["SPLASH_CODEX_BINARY"],
                base_url,
                "test-model",
                131072,
                work / "runtime",
                environment,
                input_modalities=["text", "image", "pdf"],
                client_args=[
                    "exec",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--sandbox",
                    "read-only",
                    "--json",
                    "-c",
                    'cli_auth_credentials_store="ephemeral"',
                    "-c",
                    "model_providers.splash.stream_max_retries=1",
                    "-",
                ],
            )
            process = subprocess.Popen(
                argv,
                cwd=work,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(
                    "Reply with a long paragraph, without tools.\n", timeout=45
                )
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=5)
            self.assertTrue(runtime.requests, stderr[-2000:])
            self.assertLess(
                len(runtime.requests),
                32,
                "client exhausted the bounded truncation fixture",
            )
            rows = [
                json.loads(line) for line in stdout.splitlines() if line.startswith("{")
            ]
            failed = [row for row in rows if row.get("type") == "turn.failed"]
            completed = [row for row in rows if row.get("type") == "turn.completed"]
            self.assertTrue(failed or completed, stdout + stderr[-2000:])
            if failed:
                # Current clients surface incomplete as an error; a future
                # client may accept the partial turn. Neither needs a server dialect.
                self.assertIn("max_output_tokens", json.dumps(failed))
                self.assertNotEqual(process.returncode, 0)
            else:
                self.assertEqual(process.returncode, 0, stderr[-2000:])
            self.assertEqual(runtime.pending_count, 0)

    def test_config_overrides_keep_requests_on_the_local_provider(self):
        requests, external = [], []
        received = threading.Event()

        class Recorder(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers.get("Authorization"), body))
                payload = json.dumps(
                    {
                        "error": {
                            "type": "invalid_request_error",
                            "message": "routing test complete",
                        }
                    }
                ).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                received.set()

            def do_CONNECT(self):
                external.append(self.path)
                self.send_error(503)

            def do_GET(self):
                self.send_error(404)

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "codex").mkdir()
            (work / "state").mkdir()
            (work / "logs").mkdir()
            subprocess.run(["git", "init", "-q", str(work)], check=True)
            (work / "example.py").write_text("value = 42\n")
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(thread.join)
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            base_url = f"http://127.0.0.1:{server.server_port}"
            env = {
                key: os.environ[key]
                for key in ("PATH", "HOME", "TMPDIR")
                if key in os.environ
            }
            env.update(
                CODEX_HOME=str(work / "codex"),
                HTTP_PROXY=base_url,
                HTTPS_PROXY=base_url,
                ALL_PROXY=base_url,
                NO_PROXY="127.0.0.1,localhost",
                OPENAI_BASE_URL=base_url + "/v1",
            )
            flags = [
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "-C",
                str(work),
            ]
            isolation = [
                "-c",
                'cli_auth_credentials_store="ephemeral"',
                "-c",
                "sqlite_home=" + json.dumps(str(work / "state")),
                "-c",
                "log_dir=" + json.dumps(str(work / "logs")),
            ]
            cases = [
                [*isolation, "exec", *flags, "-"],
                ["exec", *flags, *isolation, "-"],
                [
                    "exec",
                    *flags,
                    *isolation,
                    '--config=model_reasoning_effort="low"',
                    "-",
                ],
                ["exec", *flags, "review", "--uncommitted", *isolation],
                [
                    "exec",
                    *flags,
                    *isolation,
                    "--",
                    "Reply OK; -c is literal prompt text.",
                ],
            ]
            for args in cases:
                with self.subTest(args=args):
                    requests.clear()
                    external.clear()
                    received.clear()
                    argv, environment = clients.command(
                        "codex",
                        os.environ["SPLASH_CODEX_BINARY"],
                        base_url,
                        MODEL,
                        102400,
                        work / "runtime",
                        env,
                        input_modalities=["text", "image", "pdf"],
                        client_args=args,
                    )
                    with (
                        tempfile.TemporaryFile(mode="w+") as prompt,
                        tempfile.TemporaryFile(mode="w+") as output,
                        tempfile.TemporaryFile(mode="w+") as error,
                    ):
                        prompt.write("Reply OK without tools.\n")
                        prompt.seek(0)
                        process = subprocess.Popen(
                            argv,
                            stdin=prompt,
                            stdout=output,
                            stderr=error,
                            cwd=work,
                            env=environment,
                            start_new_session=True,
                        )
                        try:
                            # A fresh state directory initializes Codex's databases.
                            received.wait(45)
                        finally:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait(timeout=5)
                        error.seek(0)
                        stderr = error.read()
                    self.assertTrue(requests, stderr[-2000:])
                    # Metadata/update probes may run, but the inference
                    # provider must remain local. The proxy blocks all hosts.
                    self.assertNotIn("api.openai.com:443", external)
                    self.assertIn("provider: splash", stderr)
                    for path, authorization, body in requests:
                        self.assertEqual(path, "/v1/responses")
                        self.assertEqual(authorization, "Bearer local")
                        self.assertEqual(body["model"], MODEL)
                        self.assertFalse(
                            any(
                                tool["type"].startswith("web_search")
                                for tool in body.get("tools", [])
                            )
                        )


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

        # The installed client and production HTTP adapter are real; the
        # controlled runtime answers with one tool call, then plain text.
        tokenizer = FakeTokenizer()
        tokenizer.fragments[5] = (
            "<tool_call>\n<function=bash>\n<parameter=command>\n"
            "printf 'pi-tool-ok'\n</parameter>\n</function>\n</tool_call>\n"
        )
        tokenizer.backend_tokenizer = _byte_backend(tokenizer.fragments)
        runtime = FakeRuntime(Plan([[5]]), Plan([[4]]), Plan([[26, 4]]))
        # A key Pi would run as a command or expand if it were saved.
        api_key = "!test-$" + secrets.token_hex(16)
        harness = Harness(
            runtime,
            tokenizer=tokenizer,
            max_context=131072,
            timeout=60,
            api_key=api_key,
        )
        self.addCleanup(harness.close)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            environment = {
                key: os.environ[key] for key in ("PATH", "TMPDIR") if key in os.environ
            }
            environment.update(
                HOME=directory,
                PI_CODING_AGENT_DIR=str(work / "pi"),
                PI_OFFLINE="1",
                PI_TELEMETRY="0",
                SPLASH_API_KEY=api_key,
            )
            argv, environment = clients.command(
                "pi",
                os.environ["SPLASH_PI_BINARY"],
                f"http://127.0.0.1:{harness.server.server_port}",
                "test-model",
                131072,
                work / "runtime",
                environment,
                input_modalities=["text", "image", "pdf"],
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
            self.assertNotIn(api_key, (work / "pi/models.json").read_text())
            session = None
            for thinking in ("off", "low"):
                result = subprocess.run(
                    [
                        *argv[:-1],
                        thinking,
                        *(["--session", session] if session else []),
                    ],
                    input="Reply briefly.\n",
                    cwd=work,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, output)
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
                self.assertTrue(messages, output)
                self.assertEqual(messages[-1]["stopReason"], "stop", output)
                self.assertIn(
                    "plain answer",
                    [c.get("text", "").strip() for c in messages[-1]["content"]],
                    output,
                )
                if thinking == "off":
                    self.assertEqual(
                        executed_commands("pi", parsed), ["printf 'pi-tool-ok'"]
                    )
            self.assertEqual(len(runtime.requests), 3)
            # Thinking off reaches the template as reasoning_effort "none".
            self.assertEqual(
                [options["enable_thinking"] for _, options in tokenizer.templates],
                [False, False, True],
            )


if __name__ == "__main__":
    unittest.main()

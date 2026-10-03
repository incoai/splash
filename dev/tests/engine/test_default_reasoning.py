import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests import test_server as fixtures
from dev.tests.engine.test_launcher import keep_stop_signals, server_arguments
from install import launcher
from server import server as api

SERVER_ARGS = ["model", "--tokenizer", "tokenizer", "--model", "owner/repo"]
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class DefaultReasoningTests(unittest.TestCase):
    def harness(self, effort=None, **kwargs):
        harness = fixtures.Harness(
            fixtures.FakeRuntime(), default_reasoning_effort=effort, **kwargs
        )
        self.addCleanup(harness.close)
        return harness

    def assert_effort(self, harness, expected):
        template = harness.tokenizer.templates[-1][1]
        if expected is None:
            self.assertNotIn("enable_thinking", template)
            self.assertNotIn("reasoning_effort", template)
        else:
            self.assertEqual(template["enable_thinking"], expected != "none")
            self.assertEqual(
                template.get("reasoning_effort"),
                None if expected == "none" else expected,
            )

    def test_default_and_every_explicit_effort_across_chat_and_responses(self):
        for default in (None, *EFFORTS):
            harness = self.harness(default)
            for path in ("/v1/chat/completions", "/v1/responses"):
                for effort in (None, *EFFORTS):
                    for stream in (False, True):
                        with self.subTest(
                            default=default, path=path, effort=effort, stream=stream
                        ):
                            if path.endswith("responses"):
                                body = fixtures.ServerTest.responses_body(stream=stream)
                                if effort is not None:
                                    body["reasoning"] = {"effort": effort}
                            else:
                                body = fixtures.ServerTest.body(stream=stream)
                                if effort is not None:
                                    body["reasoning_effort"] = effort
                            status, _, payload = harness.request("POST", path, body)
                            self.assertEqual(status, 200, payload)
                            self.assert_effort(
                                harness, effort if effort is not None else default
                            )
            harness.close()

    def test_null_effort_and_empty_responses_options_use_default(self):
        harness = self.harness("none")
        for path, options in (
            ("/v1/chat/completions", {"reasoning_effort": None}),
            ("/v1/responses", {"reasoning": None}),
            ("/v1/responses", {"reasoning": {}}),
            ("/v1/responses", {"reasoning": {"effort": None}}),
        ):
            body = (
                fixtures.ServerTest.responses_body()
                if path.endswith("responses")
                else fixtures.ServerTest.body()
            )
            status, _, payload = harness.request("POST", path, {**body, **options})
            self.assertEqual(status, 200, payload)
            self.assert_effort(harness, "none")

    def test_invalid_request_effort_is_not_replaced_by_default(self):
        harness = self.harness("none")
        for path in ("/v1/chat/completions", "/v1/responses"):
            for effort in ("", "invalid", False, 1, [], {}):
                body = (
                    fixtures.ServerTest.responses_body()
                    if path.endswith("responses")
                    else fixtures.ServerTest.body()
                )
                body.update(
                    {"reasoning": {"effort": effort}}
                    if path.endswith("responses")
                    else {"reasoning_effort": effort}
                )
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 400, payload)
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_anthropic_thinking_and_count_tokens_ignore_openai_default(self):
        for default in ("none", "xhigh"):
            harness = self.harness(default)
            for thinking, expected in (
                (None, "none"),
                ({"type": "disabled"}, "none"),
                ({"type": "adaptive"}, "low"),
                ({"type": "enabled", "budget_tokens": 1024}, "low"),
            ):
                for path in ("/v1/messages", "/v1/messages/count_tokens?beta=true"):
                    body = fixtures.ServerTest.anthropic_body(
                        max_tokens=1100, output_config={"effort": "low"}
                    )
                    if thinking is not None:
                        body["thinking"] = thinking
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200, payload)
                    self.assert_effort(harness, expected)

    def test_apply_template_matches_generation(self):
        harness = self.harness("none")
        for effort in (None, "low"):
            body = fixtures.ServerTest.body()
            if effort:
                body["reasoning_effort"] = effort
            status, _, payload = harness.request("POST", "/apply-template", body)
            self.assertEqual(status, 200, payload)
            template = harness.tokenizer.templates[-1][1]
            status, _, payload = harness.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, 200, payload)
            self.assertEqual(harness.tokenizer.templates[-1][1], template)
            self.assert_effort(harness, effort or "none")

    def test_unset_preserves_model_template_including_thinking_off(self):
        for default in (False, True):
            tokenizer = fixtures.TemplateTokenizer(
                fixtures.ServerTest.reasoning_template(default=default)
            )
            harness = self.harness(tokenizer=tokenizer)
            job = harness.app.prepare(
                fixtures.ServerTest.body(), deadline=fixtures.FOREVER
            )
            self.assertEqual(job.thinking, default)
            job = harness.app.prepare_responses(
                fixtures.ServerTest.responses_body(), deadline=fixtures.FOREVER
            )
            self.assertEqual(job.thinking, default)

    def test_scoring_is_independent_of_generation_default(self):
        jobs = []
        for effort in (None, "none", "xhigh"):
            harness = self.harness(
                effort, tokenizer=fixtures.ServerTest.CharTokenizer(), max_context=8192
            )
            job, _ = harness.app.prepare_judgment(
                fixtures.ServerTest.judgment_body(), deadline=fixtures.FOREVER
            )
            self.assertFalse(job.thinking)
            jobs.append((job.prompt_tokens, job.score_tokens))
            systemone = harness.app.prepare_systemone(
                {
                    "model": "test-model",
                    "state": {},
                    "questions": {"q": {"type": "noul"}},
                },
                deadline=fixtures.FOREVER,
            )
            self.assertFalse(systemone[0][2].thinking)
        self.assertEqual(jobs, [jobs[0]] * 3)

    def test_invalid_server_defaults_fail_before_startup(self):
        for value in ("", "off", "None", "HIGH", " low"):
            for parse, args in (
                (api.parse_args, SERVER_ARGS),
                (launcher.parse_args, ["serve", "--model", "owner/repo"]),
            ):
                with (
                    self.subTest(value=value, parse=parse.__module__),
                    mock.patch.dict(
                        os.environ, {"SPLASH_DEFAULT_REASONING_EFFORT": value}
                    ),
                    mock.patch("sys.stderr", io.StringIO()),
                ):
                    with self.assertRaises(SystemExit):
                        parse(args)
                    with self.assertRaises(SystemExit):
                        parse([*args, "--default-reasoning-effort", value])
                    self.assertEqual(
                        parse(
                            [*args, "--default-reasoning-effort", "none"]
                        ).default_reasoning_effort,
                        "none",
                    )

    def test_cli_over_environment_and_launcher_forwarding(self):
        keep_stop_signals(self)
        for env, explicit, expected in (
            (None, None, None),
            ("none", None, "none"),
            ("low", "xhigh", "xhigh"),
        ):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SPLASH_DEFAULT_REASONING_EFFORT", None)
                if env is not None:
                    os.environ["SPLASH_DEFAULT_REASONING_EFFORT"] = env
                options = ["--default-reasoning-effort", explicit] if explicit else []
                self.assertEqual(
                    api.parse_args([*SERVER_ARGS, *options]).default_reasoning_effort,
                    expected,
                )
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    mock.patch.object(launcher, "RUNTIME_DIR", Path(tmp)),
                    mock.patch.object(launcher.socket, "socket"),
                    mock.patch.object(launcher, "_ensure_installed"),
                    mock.patch.object(launcher.catalog, "spawn_refresh"),
                    mock.patch.object(launcher.os, "execve") as execute,
                ):
                    launcher.main(["serve", "--model", "owner/repo", *options])
                    argv = execute.call_args.args[1]
                    self.assertEqual(
                        api.parse_args(server_arguments(argv)).default_reasoning_effort,
                        expected,
                    )
                    # The server reads the environment's default itself.
                    self.assertEqual(
                        any(
                            argument.startswith("--default-reasoning-effort")
                            for argument in argv
                        ),
                        explicit is not None,
                    )

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests import server_fixtures as fixtures
from install import launcher
from server import server as api
from server.dialects import responses as responses_dialect
from server.dialects import scoring

SERVER_ARGS = ["model", "--model", "owner/repo"]
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
            self.assertEqual(template["reasoning_effort"], expected)

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
                                body = fixtures.responses_body(stream=stream)
                                if effort is not None:
                                    body["reasoning"] = {"effort": effort}
                            else:
                                body = fixtures.chat_body(stream=stream)
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
                fixtures.responses_body()
                if path.endswith("responses")
                else fixtures.chat_body()
            )
            status, _, payload = harness.request("POST", path, {**body, **options})
            self.assertEqual(status, 200, payload)
            self.assert_effort(harness, "none")

    def test_invalid_request_effort_is_not_replaced_by_default(self):
        harness = self.harness("none")
        for path in ("/v1/chat/completions", "/v1/responses"):
            for effort in ("", "invalid", False, 1, [], {}):
                body = (
                    fixtures.responses_body()
                    if path.endswith("responses")
                    else fixtures.chat_body()
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
                    body = fixtures.anthropic_body(
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
            body = fixtures.chat_body()
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
                fixtures.reasoning_template(default=default)
            )
            harness = self.harness(tokenizer=tokenizer)
            job = harness.app.prepare(fixtures.chat_body(), deadline=fixtures.FOREVER)
            self.assertEqual(job.thinking, default)
            job, _ = responses_dialect.prepare(
                harness.app,
                fixtures.responses_body(),
                deadline=fixtures.FOREVER,
                reserve_input=fixtures.reserve_unbounded,
            )
            self.assertEqual(job.thinking, default)

    def test_scoring_is_independent_of_generation_default(self):
        jobs = []
        for effort in (None, "none", "xhigh"):
            harness = self.harness(
                effort, tokenizer=fixtures.CharTokenizer(), max_context=8192
            )
            job, _ = scoring.prepare_judgment(
                harness.app,
                fixtures.judgment_body(),
                deadline=fixtures.FOREVER,
                disconnected=fixtures.stays_connected,
            )
            self.assertFalse(job.thinking)
            jobs.append((job.prompt_tokens, job.score_tokens))
            systemone = scoring.prepare_systemone(
                harness.app,
                {
                    "model": "test-model",
                    "state": {},
                    "questions": {"q": {"type": "noul"}},
                },
                deadline=fixtures.FOREVER,
                disconnected=fixtures.stays_connected,
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
        fixtures.keep_stop_signals(self)
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
                    mock.patch.object(launcher.paths, "MODELS", Path(tmp) / "models"),
                    mock.patch.object(launcher.socket, "socket"),
                    mock.patch.object(launcher, "_ensure_installed"),
                    mock.patch.object(launcher.os, "execve") as execute,
                ):
                    launcher.main(["serve", "--model", "owner/repo", *options])
                    argv = execute.call_args.args[1]
                    self.assertEqual(
                        api.parse_args(
                            fixtures.server_arguments(argv)
                        ).default_reasoning_effort,
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


class ServerReasoningEffortTests(unittest.TestCase):
    """--reasoning-effort: the effort of every request, whatever it asks."""

    # What a request may say of reasoning, by route.
    CHAT = (
        {},
        {"reasoning_effort": "none"},
        {"reasoning_effort": "max"},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {
            "reasoning_effort": "low",
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "max",
            },
        },
    )
    RESPONSES = (
        {},
        {"reasoning": {"effort": "none"}},
        {"reasoning": {"effort": "xhigh"}},
        {"chat_template_kwargs": {"enable_thinking": True}},
    )
    MESSAGES = (
        {},
        {"thinking": {"type": "disabled"}},
        {"thinking": {"type": "adaptive"}},
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}},
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        {"chat_template_kwargs": {"enable_thinking": True}},
    )
    ROUTES = (
        ("/v1/chat/completions", fixtures.chat_body, CHAT),
        ("/apply-template", fixtures.chat_body, CHAT),
        ("/v1/responses", fixtures.responses_body, RESPONSES),
        ("/v1/messages", fixtures.anthropic_body, MESSAGES),
        ("/v1/messages/count_tokens", fixtures.anthropic_body, MESSAGES),
    )
    # Fields of reasoning a route refuses, with or without the option.
    MALFORMED = (
        ("/v1/chat/completions", fixtures.chat_body, {"reasoning_effort": "turbo"}),
        ("/v1/chat/completions", fixtures.chat_body, {"reasoning_effort": 1}),
        (
            "/v1/chat/completions",
            fixtures.chat_body,
            {"chat_template_kwargs": {"enable_thinking": "yes"}},
        ),
        ("/v1/responses", fixtures.responses_body, {"reasoning": "high"}),
        ("/v1/responses", fixtures.responses_body, {"reasoning": {"effort": "on"}}),
        ("/v1/messages", fixtures.anthropic_body, {"thinking": {"type": "on"}}),
        (
            "/v1/messages",
            fixtures.anthropic_body,
            {"thinking": {"type": "enabled", "budget_tokens": 10}},
        ),
        (
            "/v1/messages",
            fixtures.anthropic_body,
            {"thinking": {"type": "disabled", "display": "omitted"}},
        ),
        ("/v1/messages", fixtures.anthropic_body, {"output_config": {"effort": "on"}}),
    )

    def harness(self, effort=None, **kwargs):
        harness = fixtures.Harness(
            fixtures.FakeRuntime(), reasoning_effort=effort, **kwargs
        )
        self.addCleanup(harness.close)
        return harness

    def test_every_route_renders_every_request_with_the_servers_effort(self):
        for effort in ("none", "low", "max"):
            harness = self.harness(effort)
            for path, make, asks in self.ROUTES:
                for ask in asks:
                    with self.subTest(effort=effort, path=path, ask=ask):
                        status, _, payload = harness.request(
                            "POST", path, make(max_tokens=1100, **ask)
                        )
                        self.assertEqual(status, 200, payload)
                        template = harness.tokenizer.templates[-1][1]
                        self.assertEqual(template["enable_thinking"], effort != "none")
                        self.assertEqual(template["reasoning_effort"], effort)
            harness.close()

    def test_the_servers_effort_decides_whether_a_request_thinks(self):
        asks_to_think = fixtures.chat_body(
            reasoning_effort="max", chat_template_kwargs={"enable_thinking": True}
        )
        asks_not_to = fixtures.chat_body(
            reasoning_effort="none", chat_template_kwargs={"enable_thinking": False}
        )
        for effort, thinks in (("none", False), ("high", True)):
            # A template that thinks unless told not to, and one that does
            # not unless told to.
            for default in (True, False):
                harness = self.harness(
                    effort,
                    tokenizer=fixtures.TemplateTokenizer(
                        fixtures.reasoning_template(default=default)
                    ),
                )
                for body in (fixtures.chat_body(), asks_to_think, asks_not_to):
                    with self.subTest(effort=effort, default=default, body=body):
                        job = harness.app.prepare(body, deadline=fixtures.FOREVER)
                        self.assertIs(job.thinking, thinks)

    def test_a_malformed_reasoning_field_is_refused_as_without_it(self):
        plain, forced = self.harness(), self.harness("none")
        for path, make, ask in self.MALFORMED:
            with self.subTest(path=path, ask=ask):
                body = make(max_tokens=1100, **ask)
                expected = plain.request("POST", path, body)
                self.assertEqual(expected[0], 400, expected[2])
                self.assertEqual(forced.request("POST", path, body), expected)
        self.assertEqual(forced.backend.runtime.requests, [])

    def test_models_and_status_report_the_effort_in_force(self):
        # A request's effort decides nothing, so /v1/models lists none for a
        # client to offer, as for a model that decides itself; /status names
        # the effort every request gets.
        for effort in (None, "none", "high"):
            with self.subTest(effort=effort):
                harness = self.harness(effort)
                _, _, payload = harness.request("GET", "/status")
                status = json.loads(payload)
                self.assertEqual(status["reasoning_effort"], effort)
                self.assertIs(status["reasoning_switch"], effort is None)
                _, _, payload = harness.request("GET", "/v1/models")
                reasoning = json.loads(payload)["data"][0]["reasoning"]
                self.assertEqual(
                    reasoning == {"thinks": None, "efforts": [], "default": None},
                    effort is not None,
                )

    def test_a_thinking_budget_still_caps_the_thinking_it_sets(self):
        # The effort says whether a request thinks, and a budget, the
        # server's or the request's own, how long.
        harness = self.harness(
            "high",
            default_thinking_token_budget=64,
            tokenizer=fixtures.TemplateTokenizer(fixtures.reasoning_template()),
        )
        for ask, budget in (({}, 64), ({"thinking_token_budget": 32}, 32)):
            with self.subTest(ask=ask):
                job = harness.app.prepare(
                    fixtures.chat_body(reasoning_effort="none", max_tokens=100, **ask),
                    deadline=fixtures.FOREVER,
                )
                self.assertEqual(
                    (job.thinking, job.thinking_token_budget), (True, budget)
                )

    def test_started_records_show_what_was_sent_and_what_rendered(self):
        records = []
        harness = self.harness("none", request_logger=records.append)
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            fixtures.chat_body(
                reasoning_effort="high", chat_template_kwargs={"enable_thinking": True}
            ),
        )
        self.assertEqual(status, 200, payload)
        (started,) = (record for record in records if record["outcome"] == "started")
        self.assertEqual(
            started["reasoning"],
            {
                "sent_effort": "high",
                "effort": "none",
                "enable_thinking": None,
                "template_effort": None,
                "thinking": False,
            },
        )

    def test_it_is_refused_with_a_default_effort_and_taken_with_a_budget(self):
        names = ("SPLASH_REASONING_EFFORT", "SPLASH_DEFAULT_REASONING_EFFORT")
        for parse, args in (
            (api.parse_args, SERVER_ARGS),
            (launcher.parse_args, ["serve", "--model", "owner/repo"]),
        ):
            for environment, options in (
                ({}, ["--reasoning-effort=none", "--default-reasoning-effort=low"]),
                ({names[1]: "low"}, ["--reasoning-effort=none"]),
                ({names[0]: "none"}, ["--default-reasoning-effort=low"]),
            ):
                with (
                    self.subTest(parse=parse.__module__, options=options),
                    mock.patch.dict(os.environ),
                    mock.patch("sys.stderr", io.StringIO()) as stderr,
                ):
                    for name in names:
                        os.environ.pop(name, None)
                    os.environ.update(environment)
                    with self.assertRaises(SystemExit):
                        parse([*args, *options])
                    self.assertIn("--default-reasoning-effort", stderr.getvalue())
                    self.assertIn("--reasoning-effort", stderr.getvalue())
            with mock.patch.dict(os.environ):
                for name in names:
                    os.environ.pop(name, None)
                parsed = parse(
                    [*args, "--reasoning-effort=none", "--thinking-token-budget=64"]
                )
            self.assertEqual(
                (parsed.reasoning_effort, parsed.thinking_token_budget), ("none", 64)
            )

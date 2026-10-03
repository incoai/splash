import argparse
import io
import json
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest import mock
from urllib.parse import quote

from dev.tests import test_server as fixtures
from dev.tests.engine.test_json_responses import (
    PATHS,
    request_body,
    stream_events,
    tool_request,
)
from dev.tests.engine.test_launcher import keep_stop_signals, server_arguments
from install import launcher
from server import protocol as native_wire
from server import serve_options
from server import server as api

ALIASES = ("local", "community/stable:v1", "模型", "-local")
ANNOUNCED = ALIASES[0]
SERVER_ARGS = ["model", "--tokenizer", "tokenizer", "--model", "owner/repo"]


def expected(announce):
    """The model name responses report."""
    return ANNOUNCED if announce else "test-model"


class ServedModelNamesTests(unittest.TestCase):
    def harness(self, runtime=None, announce=False, **kwargs):
        harness = fixtures.Harness(
            runtime or fixtures.FakeRuntime(),
            served_model_names=ALIASES,
            announce_served_name=announce,
            **kwargs,
        )
        self.addCleanup(harness.close)
        return harness

    def assert_reported_model(self, payload, stream, model):
        """Every model field of the response, or of each streamed event, is model."""
        rows = stream_events(payload) if stream else [json.loads(payload)]
        models = [
            item["model"]
            for row in rows
            for item in (row, row.get("response", {}), row.get("message", {}))
            if "model" in item
        ]
        self.assertTrue(models, payload)
        self.assertEqual(set(models), {model})

    def test_catalog_lists_the_response_model_first_and_deduplicates_aliases(self):
        for announce, names in (
            (False, ["test-model", "local"]),
            (True, ["local", "test-model"]),
        ):
            with self.subTest(announce=announce):
                harness = fixtures.Harness(
                    fixtures.FakeRuntime(),
                    served_model_names=("local", "test-model", "local"),
                    announce_served_name=announce,
                )
                self.addCleanup(harness.close)
                status, _, payload = harness.request("GET", "/v1/models")
                self.assertEqual(status, 200)
                data = json.loads(payload)
                self.assertEqual([m["id"] for m in data["data"]], names)
                self.assertEqual([m["name"] for m in data["models"]], names)
                self.assertNotIn("root", data["data"][0])
                self.assertEqual(data["data"][1]["root"], names[0])

    def test_status_reports_the_loaded_model(self):
        for announce in (False, True):
            with self.subTest(announce=announce):
                harness = self.harness(announce=announce)
                status, _, payload = harness.request("GET", "/status")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(payload)["instance"]["model"], "test-model")

    def test_alias_lookup_including_encoded_slashes(self):
        harness = self.harness()
        for name in ("test-model", *ALIASES):
            with self.subTest(name=name):
                status, _, payload = harness.request(
                    "GET", "/v1/models/" + quote(name, safe="")
                )
                self.assertEqual(status, 200, payload)
                self.assertEqual(json.loads(payload)["id"], name)
        self.assertEqual(harness.request("GET", "/v1/models/missing")[0], 404)

    def test_catalog_and_aliases_report_the_effective_context_limit(self):
        for context in (32768, 102400, 262144):
            with self.subTest(context=context):
                harness = self.harness(max_context=context)
                status, _, payload = harness.request("GET", "/v1/models")
                self.assertEqual(status, 200)
                for model in json.loads(payload)["data"]:
                    self.assertEqual(model["max_model_len"], context)
                    self.assertEqual(model["context_length"], context)
                    status, _, detail = harness.request(
                        "GET", "/v1/models/" + quote(model["id"], safe="")
                    )
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(detail), model)

    def test_all_generation_apis_accept_every_name_and_report_the_response_model(
        self,
    ):
        for announce in (False, True):
            harness = self.harness(announce=announce)
            for path, stream, name in product(
                (*PATHS, "/v1/completions"), (False, True), ("test-model", *ALIASES)
            ):
                with self.subTest(
                    announce=announce, path=path, stream=stream, name=name
                ):
                    body = (
                        {"prompt": "hello", "max_tokens": 4, "stream": stream}
                        if path == "/v1/completions"
                        else request_body(path, stream)
                    )
                    body["model"] = name
                    if stream and path in (PATHS[0], "/v1/completions"):
                        body["stream_options"] = {"include_usage": True}
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200, payload)
                    self.assert_reported_model(payload, stream, expected(announce))

    def test_streamed_tool_calls_report_the_response_model(self):
        tokenizer = fixtures.FakeTokenizer()
        tokenizer.fragments[40] = (
            "<tool_call>\n<function=echo>\n<parameter=value>\n"
            "1\n</parameter>\n</function>\n</tool_call>\n"
        )
        tokenizer.backend_tokenizer = fixtures._byte_backend(tokenizer.fragments)
        for announce in (False, True):
            harness = self.harness(announce=announce, tokenizer=tokenizer)
            for path in PATHS:
                with self.subTest(announce=announce, path=path):
                    harness.backend.runtime.plans.append(fixtures.Plan([[40]]))
                    status, _, payload = harness.request(
                        "POST", path, tool_request(path, stream=True)
                    )
                    self.assertEqual(status, 200, payload)
                    # The stream carries the call, not an error event.
                    self.assertIn(b'"echo"', payload)
                    self.assert_reported_model(payload, True, expected(announce))

    def test_responses_without_aliases_report_the_loaded_model(self):
        harness = fixtures.Harness(fixtures.FakeRuntime())
        self.addCleanup(harness.close)
        status, _, payload = harness.request("POST", PATHS[0], request_body(PATHS[0]))
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["model"], "test-model")

    def test_template_and_token_count_accept_aliases(self):
        harness = self.harness()
        for path in ("/apply-template", "/v1/messages/count_tokens?beta=true"):
            for alias in ALIASES:
                body = fixtures.ServerTest.body(model=alias)
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 200, payload)
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_alias_and_reasoning_default_work_together(self):
        harness = self.harness(default_reasoning_effort="none")
        for path in ("/v1/chat/completions", "/v1/responses"):
            for effort in (None, "low"):
                for stream in (False, True):
                    body = request_body(path, stream)
                    body["model"] = "local"
                    if effort is not None:
                        body.update(
                            {"reasoning": {"effort": effort}}
                            if path.endswith("responses")
                            else {"reasoning_effort": effort}
                        )
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200, payload)
                    template = harness.tokenizer.templates[-1][1]
                    self.assertEqual(template["enable_thinking"], effort is not None)
                    self.assertEqual(template.get("reasoning_effort"), effort)

    def test_scoring_accepts_alias_and_reports_the_response_model(self):
        for announce in (False, True):
            with self.subTest(announce=announce):
                runtime = fixtures.FakeRuntime(
                    fixtures.Plan(logits=(1.0, -1.0)), fixtures.Plan(logits=(1.0, -1.0))
                )
                harness = self.harness(
                    runtime,
                    announce=announce,
                    tokenizer=fixtures.ServerTest.CharTokenizer(),
                    max_context=8192,
                )
                status, _, payload = harness.request(
                    "POST",
                    "/v1/judgments",
                    fixtures.ServerTest.judgment_body(model="local"),
                )
                self.assertEqual(status, 200, payload)
                self.assertEqual(json.loads(payload)["model"]["id"], expected(announce))
                status, _, payload = harness.request(
                    "POST",
                    "/v1/systemone",
                    {
                        "model": "local",
                        "state": {},
                        "questions": {"q": {"type": "noul"}},
                    },
                )
                self.assertEqual(status, 200, payload)
                self.assertEqual(json.loads(payload)["model"], expected(announce))

    def test_unknown_or_malformed_model_rejected_without_inference(self):
        runtime = fixtures.FakeRuntime()
        harness = self.harness(runtime)
        for path in PATHS:
            for value in ("unknown", "LOCAL", [], {}, 1):
                with self.subTest(path=path, value=value):
                    body = request_body(path)
                    body["model"] = value
                    status, _, payload = harness.request("POST", path, body)
                    self.assertIn(status, (400, 404), payload)
        self.assertEqual(runtime.requests, [])
        self.assertEqual(
            harness.request("POST", PATHS[0], request_body(PATHS[0]))[0], 200
        )

    def test_response_history_works_across_aliases(self):
        for announce in (False, True):
            with self.subTest(announce=announce):
                harness = self.harness(announce=announce)
                body = request_body(PATHS[1])
                body.update(model="local", store=True)
                status, _, payload = harness.request("POST", PATHS[1], body)
                self.assertEqual(status, 200, payload)
                first = json.loads(payload)
                self.assertEqual(first["model"], expected(announce))
                body.update(model=ALIASES[1], previous_response_id=first["id"])
                status, _, payload = harness.request("POST", PATHS[1], body)
                self.assertEqual(status, 200, payload)
                self.assertEqual(json.loads(payload)["model"], expected(announce))
                status, _, payload = harness.request(
                    "GET", "/v1/responses/" + first["id"]
                )
                self.assertEqual(status, 200, payload)
                self.assertEqual(json.loads(payload)["model"], expected(announce))

    def test_alias_validation_before_startup(self):
        for name in (
            "",
            "has space",
            "newline\n",
            "a?b",
            "a#b",
            "a%b",
            "a\\b",
            "/a",
            "a/",
            "a//b",
            ".",
            "a/../b",
        ):
            with self.subTest(name=name):
                with self.assertRaises(argparse.ArgumentTypeError):
                    serve_options.parse_served_model_name(name)
                for parse, args in (
                    (api.parse_args, SERVER_ARGS),
                    (launcher.parse_args, ["serve", "--model", "owner/repo"]),
                ):
                    with (
                        mock.patch("sys.stderr", io.StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        parse([*args, "--served-model-name", name])
        for name in ALIASES:
            self.assertEqual(serve_options.parse_served_model_name(name), name)

    def test_announce_requires_a_served_name(self):
        for parse, args in (
            (api.parse_args, SERVER_ARGS),
            (launcher.parse_args, ["serve", "--model", "owner/repo"]),
        ):
            with self.subTest(parse=parse.__module__):
                with (
                    mock.patch("sys.stderr", io.StringIO()) as stderr,
                    self.assertRaises(SystemExit) as raised,
                ):
                    parse([*args, "--announce-served-name"])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(
                    "--announce-served-name needs --served-model-name",
                    stderr.getvalue(),
                )
        with self.assertRaises(ValueError):
            fixtures.make_frontend(
                fixtures.FakeTokenizer(),
                None,
                "test-model",
                128,
                1,
                2,
                vision=True,
                announce_served_name=True,
            )

    def test_main_wires_served_model_names_into_the_frontend(self):
        runtime = mock.Mock()
        runtime.readiness = native_wire.ReadyEvent(4, 131072, False)
        args = fixtures.main_args(
            served_model_name=["local"], announce_served_name=True
        )
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=object()
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "ChatTemplates"),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", return_value=runtime
            ),
            mock.patch.object(api, "NativeBackend"),
            mock.patch.object(api, "ConstraintFactory"),
            mock.patch.object(api, "Frontend") as app_type,
            mock.patch.object(
                api, "FrontendServer", return_value=mock.Mock(server_port=8000)
            ),
            mock.patch.object(api.signal, "signal"),
            mock.patch.object(api, "print_status"),
        ):
            api.main()
        self.assertEqual(app_type.call_args.kwargs["served_model_names"], ["local"])
        self.assertIs(app_type.call_args.kwargs["announce_served_name"], True)

    def test_launcher_forwards_repeated_aliases(self):
        keep_stop_signals(self)
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(launcher, "RUNTIME_DIR", Path(tmp)),
            mock.patch.object(launcher.socket, "socket"),
            mock.patch.object(launcher, "_ensure_installed"),
            mock.patch.object(launcher.catalog, "spawn_refresh"),
            mock.patch.object(launcher.os, "execve") as execute,
        ):
            launcher.main(
                [
                    "serve",
                    "--model",
                    "owner/repo",
                    "--served-model-name",
                    "local",
                    "--served-model-name",
                    "stable",
                    "--served-model-name=-local",
                ]
            )
            argv = execute.call_args.args[1]
            parsed = api.parse_args(server_arguments(argv))
            self.assertEqual(parsed.model, "owner/repo")
            self.assertEqual(parsed.served_model_name, ["local", "stable", "-local"])

    def test_launcher_forwards_the_announced_name_only_when_asked(self):
        keep_stop_signals(self)
        for flags, announced in (([], False), (["--announce-served-name"], True)):
            with (
                self.subTest(flags=flags),
                tempfile.TemporaryDirectory() as tmp,
                mock.patch.object(launcher, "RUNTIME_DIR", Path(tmp)),
                mock.patch.object(launcher.socket, "socket"),
                mock.patch.object(launcher, "_ensure_installed"),
                mock.patch.object(launcher.catalog, "spawn_refresh"),
                mock.patch.object(launcher.os, "execve") as execute,
            ):
                launcher.main(
                    [
                        "serve",
                        "--model",
                        "owner/repo",
                        "--served-model-name",
                        "local",
                        *flags,
                    ]
                )
                parsed = api.parse_args(server_arguments(execute.call_args.args[1]))
                self.assertEqual(parsed.served_model_name, ["local"])
                self.assertIs(parsed.announce_served_name, announced)

    def test_client_launcher_configures_the_name_responses_report(self):
        for announce in (False, True):
            with self.subTest(announce=announce):
                harness = self.harness(announce=announce)
                status, _, payload = harness.request("GET", "/v1/models")
                self.assertEqual(status, 200)
                with (
                    mock.patch.object(
                        launcher.clients, "find_executable", return_value="codex"
                    ),
                    mock.patch.object(
                        launcher, "_request_json", return_value=json.loads(payload)
                    ),
                    mock.patch.object(
                        launcher.clients, "command", return_value=(["codex"], {})
                    ) as command,
                    mock.patch.object(launcher.os, "execvpe"),
                    mock.patch("sys.stdout", io.StringIO()),
                ):
                    launcher.coding_client(launcher.parse_args(["codex"]))
                model = command.call_args.args[3]
                self.assertEqual(model, expected(announce))
                body = request_body(PATHS[0])
                body["model"] = model
                status, _, payload = harness.request("POST", PATHS[0], body)
                self.assertEqual(status, 200, payload)
                self.assertEqual(json.loads(payload)["model"], model)

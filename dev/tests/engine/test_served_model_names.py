import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote

from dev.tests import test_server as fixtures
from dev.tests.engine.test_json_responses import PATHS, request_body, stream_events
from install import launcher
from server import frontend
from server import server as api

ALIASES = ("local", "community/stable:v1", "模型", "-local")
SERVER_ARGS = ["target", "draft", "--tokenizer", "tokenizer", "--model", "owner/repo"]


class ServedModelNamesTests(unittest.TestCase):
    def harness(self, runtime=None, **kwargs):
        harness = fixtures.Harness(
            runtime or fixtures.FakeRuntime(), served_model_names=ALIASES, **kwargs
        )
        self.addCleanup(harness.close)
        return harness

    def test_catalog_keeps_canonical_identity_and_deduplicates_aliases(self):
        harness = fixtures.Harness(
            fixtures.FakeRuntime(), served_model_names=("local", "test-model", "local")
        )
        self.addCleanup(harness.close)
        status, _, payload = harness.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        data = json.loads(payload)
        self.assertEqual([m["id"] for m in data["data"]], ["test-model", "local"])
        self.assertEqual([m["name"] for m in data["models"]], ["test-model", "local"])
        self.assertNotIn("root", data["data"][0])
        self.assertEqual(data["data"][1]["root"], "test-model")

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

    def test_all_generation_apis_accept_aliases_but_report_real_model(self):
        harness = self.harness()
        for path in PATHS:
            for stream in (False, True):
                for alias in ALIASES:
                    with self.subTest(path=path, stream=stream, alias=alias):
                        body = request_body(path, stream)
                        body["model"] = alias
                        status, _, payload = harness.request("POST", path, body)
                        self.assertEqual(status, 200, payload)
                        rows = (
                            stream_events(payload) if stream else [json.loads(payload)]
                        )
                        models = []
                        for row in rows:
                            for item in (
                                row,
                                row.get("response", {}),
                                row.get("message", {}),
                            ):
                                if "model" in item:
                                    models.append(item["model"])
                        self.assertTrue(models, payload)
                        self.assertEqual(set(models), {"test-model"})

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

    def test_scoring_accepts_alias_and_reports_real_model(self):
        runtime = fixtures.FakeRuntime(
            fixtures.Plan(logits=(1.0, -1.0)), fixtures.Plan(logits=(1.0, -1.0))
        )
        harness = self.harness(
            runtime, tokenizer=fixtures.ServerTest.CharTokenizer(), max_context=8192
        )
        status, _, payload = harness.request(
            "POST", "/v1/judgments", fixtures.ServerTest.judgment_body(model="local")
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["model"]["id"], "test-model")
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {"model": "local", "state": {}, "questions": {"q": {"type": "noul"}}},
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["model"], "test-model")

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
        harness = self.harness()
        body = request_body(PATHS[1])
        body.update(model="local", store=True)
        status, _, payload = harness.request("POST", PATHS[1], body)
        self.assertEqual(status, 200, payload)
        first = json.loads(payload)
        body.update(model=ALIASES[1], previous_response_id=first["id"])
        status, _, payload = harness.request("POST", PATHS[1], body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["model"], "test-model")
        status, _, payload = harness.request("GET", "/v1/responses/" + first["id"])
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["model"], "test-model")

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
                with self.assertRaises(ValueError):
                    frontend.validate_served_model_name(name)
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
            self.assertEqual(frontend.validate_served_model_name(name), name)
            self.assertEqual(launcher._parse_served_model_name(name), name)

    def test_launcher_forwards_repeated_aliases(self):
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
            parsed = api.parse_args(argv[3:])
            self.assertEqual(parsed.model, "owner/repo")
            self.assertEqual(parsed.served_model_name, ["local", "stable", "-local"])

    def test_client_launcher_uses_canonical_model_when_aliases_are_listed(self):
        models = [
            {"id": name, "owned_by": "splash"} for name in ("owner/repo", "local")
        ]
        with (
            mock.patch.object(
                launcher.clients, "find_executable", return_value="codex"
            ),
            mock.patch.object(
                launcher,
                "_running_status",
                return_value={"maximum_context_tokens": 4096},
            ),
            mock.patch.object(launcher, "_request_json", return_value={"data": models}),
            mock.patch.object(
                launcher.clients, "command", return_value=(["codex"], {})
            ) as command,
            mock.patch.object(launcher.os, "execvpe"),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            launcher.coding_client(launcher.parse_args(["codex"]))
            self.assertEqual(command.call_args.args[3], "owner/repo")

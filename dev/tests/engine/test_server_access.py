import concurrent.futures
import http.client
import io
import json
import os
import unittest
from unittest import mock

from openai import AuthenticationError, OpenAI

from dev.tests.test_server import FakeRuntime, Harness, Plan
from install import launcher
from server import server


class ServerAccessTests(unittest.TestCase):
    def test_wildcard_listener_keeps_host_and_api_key_validation(self):
        harness = self.harness(
            host="0.0.0.0", allowed_hosts=("splash.local",), api_key="test-server-key"
        )
        self.assertEqual(harness.server.server_address[0], "0.0.0.0")
        port = harness.server.server_address[1]
        for host, key, expected in (
            (f"127.0.0.1:{port}", "test-server-key", 200),
            (f"127.0.0.1:{port}", "wrong", 401),
            (f"splash.local:{port}", "test-server-key", 200),
            (f"unknown.example:{port}", "test-server-key", 403),
        ):
            with self.subTest(host=host, expected=expected):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                try:
                    connection.request(
                        "GET",
                        "/v1/models",
                        headers={
                            "Host": host,
                            "Authorization": f"Bearer {key}",
                        },
                    )
                    response = connection.getresponse()
                    self.assertEqual(response.status, expected, response.read())
                finally:
                    connection.close()

    def test_authenticated_streams_and_disconnect_cleanup(self):
        paths = (
            (
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hello"}]},
            ),
            ("/v1/responses", {"input": "hello"}),
            (
                "/v1/messages",
                {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16},
            ),
        )
        for path, fields in paths:
            with self.subTest(path=path):
                blocked = Plan(block=True)
                runtime = FakeRuntime(Plan([[1, 2, 3]]), blocked, Plan([[1, 2, 3]]))
                harness = Harness(runtime, api_key="test-server-key")
                self.addCleanup(harness.close)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test-server-key",
                }
                body = {"model": "test-model", "stream": True, **fields}
                status, content_type, data = harness.request(
                    "POST", path, body, headers
                )
                self.assertEqual(status, 200, data)
                self.assertTrue(content_type.startswith("text/event-stream"))
                self.assertIn(b"answer", data)
                self.assertNotIn(b"test-server-key", data)
                connection = http.client.HTTPConnection(
                    *harness.server.server_address, timeout=3
                )
                connection.request("POST", path, json.dumps(body), headers)
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.close()
                connection.close()
                self.assertTrue(
                    blocked.cancelled.wait(2),
                    "stream disconnect did not cancel generation",
                )
                self.assertTrue(harness.server.requests.idle.wait(2))
                self.assertEqual(harness.request("POST", path, body, headers)[0], 200)

    def test_openai_sdk_recognizes_auth_errors_and_recovers_with_valid_key(self):
        harness = self.harness(api_key="test-server-key")
        base_url = "http://%s:%s/v1" % harness.server.server_address
        with OpenAI(base_url=base_url, api_key="incorrect", max_retries=0) as client:
            with self.assertRaises(AuthenticationError):
                client.models.list()
        with OpenAI(
            base_url=base_url, api_key="test-server-key", max_retries=0
        ) as client:
            self.assertEqual(client.models.list().data[0].id, "test-model")

    def test_concurrent_clients_do_not_share_authentication_state(self):
        harness = self.harness(api_key="test-server-key")

        def query(index):
            authorized = index % 3 != 0
            key = "test-server-key" if authorized else "incorrect"
            status, _, _ = harness.request(
                "GET", "/v1/models", headers={"Authorization": f"Bearer {key}"}
            )
            return status, 200 if authorized else 401

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for actual, expected in pool.map(query, range(64)):
                self.assertEqual(actual, expected)
        self.assertTrue(harness.server.connections.idle.wait(2))
        self.assertFalse(harness.backend.runtime.requests)

    def test_authenticated_generation_uses_all_three_protocols(self):
        runtime = FakeRuntime(*(Plan([[1, 2, 3]]) for _ in range(3)))
        harness = Harness(runtime, api_key="test-server-key")
        self.addCleanup(harness.close)
        messages = [{"role": "user", "content": "hello"}]
        for path, body in (
            ("/v1/chat/completions", {"messages": messages}),
            ("/v1/responses", {"input": "hello"}),
            ("/v1/messages", {"messages": messages, "max_tokens": 16}),
        ):
            with self.subTest(path=path):
                status, _, data = harness.request(
                    "POST",
                    path,
                    {"model": "test-model", **body},
                    {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer test-server-key",
                    },
                )
                self.assertEqual(status, 200, data)
                self.assertIn(b"answer", data)
        self.assertEqual(len(runtime.requests), 3)

    def harness(self, **kwargs):
        harness = Harness(FakeRuntime(), **kwargs)
        self.addCleanup(harness.close)
        return harness

    def test_authentication_precedes_body_parsing_and_admission(self):
        harness = self.harness(api_key="test-server-key")
        for path in (
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
            "/v1/messages/count_tokens?beta=true",
            "/tokenize",
            "/apply-template",
        ):
            with (
                self.subTest(path=path),
                mock.patch.object(server.FrontendHandler, "_read_json_body") as read,
            ):
                status, _, data = harness.request("POST", path)
                self.assertEqual(status, 401)
                self.assertNotIn("test-server-key", data.decode())
                read.assert_not_called()
                if path.startswith("/v1/messages"):
                    self.assertEqual(
                        json.loads(data)["error"]["type"], "authentication_error"
                    )

    def test_credentials_protect_control_and_history_routes(self):
        harness = self.harness(api_key="test-server-key")
        for method, path in (
            ("GET", "/status"),
            ("GET", "/metrics"),
            ("HEAD", "/v1/models"),
            ("GET", "/v1/responses/resp_missing"),
            ("DELETE", "/v1/responses/resp_missing"),
        ):
            with self.subTest(method=method, path=path):
                self.assertEqual(harness.request(method, path)[0], 401)
        for headers in (
            {"Authorization": "Bearer test-server-key"},
            {"Authorization": "bearer test-server-key"},
            {"x-api-key": "test-server-key"},
        ):
            self.assertEqual(
                harness.request("GET", "/v1/models", headers=headers)[0], 200
            )
        for headers in (
            {"Authorization": "Bearer incorrect"},
            {"Authorization": "Basic test-server-key"},
            {"Authorization": "Bearer test-server-key", "x-api-key": "test-server-key"},
        ):
            self.assertEqual(
                harness.request("GET", "/v1/models", headers=headers)[0], 401
            )

    def test_public_probes_and_optional_webui(self):
        harness = self.harness(api_key="test-server-key", webui=False)
        for method in ("GET", "HEAD"):
            for path in ("/health", "/ready"):
                self.assertEqual(harness.request(method, path)[0], 200)
            for path in ("/", "/index.html?test=1"):
                self.assertEqual(harness.request(method, path)[0], 404)
        default = self.harness()
        self.assertEqual(default.request("GET", "/")[0], 200)
        self.assertEqual(default.request("GET", "/v1/models")[0], 200)
        protected = self.harness(api_key="test-server-key")
        status, _, html = protected.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertNotIn(b"test-server-key", html)

    def test_cli_key_precedence_and_validation(self):
        for parse, arguments in (
            (launcher.parse_args, ["serve", "--model", "owner/repo"]),
            (
                server.parse_args,
                [
                    "target",
                    "draft",
                    "--tokenizer",
                    "tokenizer",
                    "--model",
                    "owner/repo",
                ],
            ),
        ):
            with mock.patch.dict(os.environ, {"SPLASH_API_KEY": "environment-key"}):
                self.assertEqual(parse(arguments).api_key, "environment-key")
                args = parse([*arguments, "--api-key", "argument-key", "--no-webui"])
                self.assertEqual(args.api_key, "argument-key")
                self.assertTrue(args.no_webui)
            for key in ("", "two words", "key\n", "非ASCII"):
                with self.subTest(key=key), mock.patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse([*arguments, "--api-key", key])

    def test_launcher_uses_key_and_reports_auth_failure(self):
        harness = self.harness(api_key="test-server-key")
        port = harness.server.server_port
        with mock.patch.dict(os.environ, {"SPLASH_API_KEY": "test-server-key"}):
            self.assertIn("data", launcher._request_json("/v1/models", port=port))
        with mock.patch.dict(os.environ, {"SPLASH_API_KEY": "incorrect"}):
            with self.assertRaisesRegex(launcher.LauncherError, "SPLASH_API_KEY"):
                launcher._request_json("/v1/models", port=port)

    def test_unauthenticated_request_has_bearer_challenge(self):
        harness = self.harness(api_key="test-server-key")
        connection = http.client.HTTPConnection(*harness.server.server_address)
        self.addCleanup(connection.close)
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        response.read()

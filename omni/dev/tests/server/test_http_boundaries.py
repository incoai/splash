import http.client
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest import mock

from dev.tests.audio_fixtures import (
    AudioHarness,
    AudioHarnessTestCase,
    FakeAudioRuntime,
)
from server import crash_trace, http_security, origins
from server.errors import APIError

INVALID_ORIGINS = (
    "",
    "null",
    "localhost:3000",
    "http://",
    "http://localhost:3000/",
    "http://localhost/app",
    "http://localhost?debug",
    "http://user@localhost",
    "http://localhost:99999",
    "http://local host",
)


class HttpBoundaryTests(AudioHarnessTestCase):
    def test_model_retrieval_matches_listing_without_inference(self):
        harness = AudioHarness(FakeAudioRuntime(), model="community/example-model")
        self.addCleanup(harness.close)
        status, _, body = harness.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        model = json.loads(body)["data"][0]
        for suffix in (
            "community/example-model",
            "community%2Fexample-model?beta=true",
        ):
            with self.subTest(suffix=suffix):
                status, _, body = harness.request("GET", "/v1/models/" + suffix)
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body), model)
        for suffix in ("", "community/other-model", "community/example-model/extra"):
            with self.subTest(suffix=suffix):
                status, _, body = harness.request("GET", "/v1/models/" + suffix)
                self.assertEqual(status, 404)
                self.assertEqual(json.loads(body)["error"]["code"], "model_not_found")
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_authority_and_origin_validation(self):
        allowed = {"localhost", "127.0.0.1", "::1", "serving.example"}

        def elsewhere(origin):
            return (
                f"Origin {origin} is not allowed; restart the server with "
                f"--allowed-origin {origin} to accept it"
            )

        cases = (
            ("localhost:8000", "http://localhost:8000", None),
            ("[::1]:8000", "http://[::1]:8000", None),
            ("serving.example", "https://serving.example", None),
            ("127.0.0.1:8000", None, None),
            (
                "unconfigured.example:8000",
                None,
                "Host unconfigured.example is not allowed; restart the server "
                "with --allowed-host unconfigured.example to accept it",
            ),
            ("user@localhost:8000", None, "invalid Host header"),
            (
                "localhost:8000",
                "http://localhost:9000",
                elsewhere("http://localhost:9000"),
            ),
            (
                "localhost:8000",
                "http://other.example:8000",
                elsewhere("http://other.example:8000"),
            ),
            ("localhost:8000", "tauri://localhost", elsewhere("tauri://localhost")),
            ("localhost:8000", "null", "invalid Origin header"),
            ("localhost:8000", "tauri://*", "invalid Origin header"),
            ("localhost:8000", "http://user@localhost:8000", "invalid Origin header"),
            ("localhost:8000", "http://localhost:8000/path", "invalid Origin header"),
            ("localhost:8000", "http://localhost:99999", "invalid Origin header"),
        )
        for host, origin, rejection in cases:
            with self.subTest(host=host, origin=origin):
                headers = Message()
                headers["Host"] = host
                if origin is not None:
                    headers["Origin"] = origin
                if rejection is None:
                    self.assertIsNone(
                        http_security.validate_headers(headers, allowed, frozenset())
                    )
                    continue
                with self.assertRaises(APIError) as caught:
                    http_security.validate_headers(headers, allowed, frozenset())
                error = caught.exception
                self.assertEqual(
                    (error.status, error.code, error.message),
                    (403, "forbidden", rejection),
                )
                # Only a well-formed origin that is not admitted is refused
                # by name, for the server to print.
                refused = rejection == elsewhere(origin)
                self.assertIs(isinstance(error, http_security.OriginRefused), refused)
                if refused:
                    self.assertEqual(error.origin, origin)
        for name in ("Host", "Origin"):
            headers = Message()
            headers["Host"] = "localhost"
            headers["Origin"] = "http://localhost"
            headers[name] = headers[name]
            with self.assertRaises(APIError):
                http_security.validate_headers(headers, allowed, frozenset())

    def test_only_the_origins_named_are_admitted_from_elsewhere(self):
        def answer(origin, allowed):
            headers = Message()
            headers["Host"] = "localhost:8000"
            headers["Origin"] = origin
            return http_security.validate_headers(headers, {"localhost"}, allowed)

        named = frozenset(
            map(
                origins.parse_allowed_origin,
                ("tauri://localhost", "HTTP://Localhost:3000", "https://chat.example"),
            )
        )
        for origin in (
            "tauri://localhost",
            "http://localhost:3000",
            "https://chat.example",
            "https://chat.example:443",
        ):
            with self.subTest(origin=origin):
                self.assertEqual(answer(origin, named), origin)
        for origin in (
            "tauri://other",
            "app://localhost",
            "http://localhost:3001",
            "http://chat.example",
            "https://chat.example:8443",
        ):
            with self.subTest(origin=origin), self.assertRaises(APIError) as caught:
                answer(origin, named)
            self.assertIn(f"--allowed-origin {origin} ", caught.exception.message)
        # The server's own pages owe their browser nothing, named or not.
        self.assertIsNone(answer("http://localhost:8000", named))
        # Every origin: whatever a browser sends, a sandboxed page's null too.
        every = frozenset({origins.ANY_ORIGIN})
        for origin in ("https://anywhere.example", "null", "http://localhost:8000"):
            with self.subTest(origin=origin):
                self.assertEqual(answer(origin, every), "*")

    def test_allowed_origin_grammar(self):
        # An origin as a browser sends it, whatever the letter case and
        # with its scheme's default port; '*' stands for every origin.
        for value, parsed in (
            ("*", origins.ANY_ORIGIN),
            ("HTTP://Localhost:3000", ("http", "localhost", 3000)),
            ("tauri://localhost", ("tauri", "localhost", None)),
            ("https://chat.example", ("https", "chat.example", 443)),
        ):
            with self.subTest(value=value):
                self.assertEqual(origins.parse_allowed_origin(value), parsed)
        for value in INVALID_ORIGINS:
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "expected a scheme and a host"),
            ):
                origins.parse_allowed_origin(value)
        # Origins match exactly: a pattern, which would match nothing, is refused.
        for value in (
            "tauri://*",
            "app://*",
            "http://*.example.com",
            "http://localhost:*",
            "*://localhost",
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, r"only a bare '\*'"),
            ):
                origins.parse_allowed_origin(value)

    def test_http_rejection_precedes_routing_and_local_access_still_works(self):
        harness = self.harness()
        status, _, body = harness.request(
            "GET", "/health", headers={"Host": "unconfigured.example"}
        )
        self.assertEqual(status, 403)
        error = json.loads(body)["error"]
        self.assertEqual(error["code"], "forbidden")
        self.assertIn("--allowed-host unconfigured.example", error["message"])
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_crash_content_is_disabled_without_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with mock.patch.object(crash_trace, "DEFAULT_TRACE_DIRECTORY", directory):
                ring = crash_trace.CrashTraceRing(("splash", "serve-omni"))
                ring.start_generation(1, 1)
                ring.record_bytes(1, "client_to_engine", b"private synthetic content")
                self.assertIsNone(
                    ring.dump(
                        1,
                        RuntimeError("failed"),
                        process_returncode=1,
                        last_status=None,
                    )
                )
                self.assertEqual(list(directory.iterdir()), [])

    def test_opt_in_crash_files_have_a_total_retention_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                mock.patch.object(crash_trace, "DEFAULT_TRACE_DIRECTORY", directory),
                mock.patch.object(crash_trace, "MAX_TRACE_FILES", 2),
            ):
                ring = crash_trace.CrashTraceRing(
                    ("splash", "serve-omni"), enabled=True
                )
                for generation in range(1, 5):
                    ring.start_generation(generation, 1)
                    ring.record_bytes(generation, "client_to_engine", b"synthetic")
                    ring.dump(
                        generation,
                        RuntimeError("failed"),
                        process_returncode=1,
                        last_status=None,
                    )
                files = list(directory.glob("*.json"))
                self.assertEqual(len(files), 2)
                self.assertEqual(
                    {json.loads(path.read_text())["generation"] for path in files},
                    {3, 4},
                )

    def test_models_routes_use_normalized_paths(self):
        harness = self.harness()

        def get(path):
            connection = http.client.HTTPConnection(
                *harness.server.server_address, timeout=3
            )
            connection.request("GET", path)
            response = connection.getresponse()
            result = (response.status, response.read())
            connection.close()
            return result

        status, payload = get("/v1/models")
        self.assertEqual(status, 200)
        body = json.loads(payload)
        self.assertEqual(body["data"][0]["id"], "test-model")

        # Encoded spellings of the route and model id resolve identically.
        for path in (
            "/v1/models/test-model",
            "/v1/models/test-%6Dodel",
            "/v1%2Fmodels",
            "/v1/models/../models",
        ):
            with self.subTest(path=path):
                self.assertEqual(get(path)[0], 200)

        # Unknown models still 404 under the normalized prefix.
        self.assertEqual(get("/v1/models/unknown")[0], 404)
        self.assertEqual(get("/v1/models/")[0], 404)

        # Paths that normalize away from the catalog are routed where they
        # lead.
        for path in ("/health", "/v1/models/%2e%2e/%2e%2e/health"):
            with self.subTest(path=path):
                self.assertEqual(get(path), (200, b'{"status":"ok"}'))

    def test_models_route_lists_the_served_model(self):
        harness = self.harness()
        status, _, payload = harness.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        model = json.loads(payload)["data"][0]
        self.assertEqual(model["id"], "test-model")
        self.assertEqual(model["owned_by"], "splash")


if __name__ == "__main__":
    unittest.main()

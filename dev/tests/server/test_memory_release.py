"""POST /v1/memory/release over HTTP: a request takes no fields, and the answer
says whether the engine gave its weights back."""

import http.client
import json
import unittest
from unittest import mock

from dev.tests import native_peer
from dev.tests.server_fixtures import FakeRuntime, HarnessTestCase
from server import runtime as engine_runtime

RELEASED = {"released": True, "restores": 0, "restore_failures": 0}


class MemoryReleaseTests(HarnessTestCase):
    def post(self, harness, headers, data=None):
        """A POST of exactly these headers and bytes, and its status and body."""
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        self.addCleanup(connection.close)
        connection.putrequest("POST", "/v1/memory/release")
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders(data)
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def test_an_idle_engine_releases_for_a_request_without_fields(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for headers, data in (
            ({}, None),
            ({"Content-Length": "0"}, None),
            ({"Content-Type": "application/json", "Content-Length": "2"}, b"{}"),
        ):
            with self.subTest(headers=headers):
                self.assertEqual(
                    self.post(harness, headers, data),
                    (200, {"released": True, "weights": RELEASED}),
                )
        self.assertEqual(runtime.release_calls, 3)

    def test_a_body_with_fields_or_not_json_is_refused_before_the_engine(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for headers, data, status in (
            (
                {"Content-Type": "application/json", "Content-Length": "8"},
                b'{"x": 1}',
                400,
            ),
            ({"Content-Type": "application/json", "Content-Length": "2"}, b"[]", 400),
            ({"Content-Type": "text/plain", "Content-Length": "2"}, b"{}", 415),
        ):
            with self.subTest(data=data):
                self.assertEqual(self.post(harness, headers, data)[0], status)
        self.assertEqual(runtime.release_calls, 0)

    def test_a_busy_engine_keeps_its_weights(self):
        runtime = FakeRuntime()
        runtime.release_event = native_peer.status_event(
            weights={"released": False, "restores": 0, "restore_failures": 0}
        )
        harness = self.harness(runtime)
        status, payload = self.post(harness, {})
        self.assertEqual((status, payload["error"]["code"]), (409, "engine_busy"))

    def test_an_engine_that_does_not_answer_or_fails_is_unavailable(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for error, code in (
            (TimeoutError("native status response timed out"), "engine_busy"),
            (
                engine_runtime.EngineUnhealthy("native loop failed"),
                "runtime_unavailable",
            ),
        ):
            with self.subTest(code=code):
                with mock.patch.object(runtime, "release_weights", side_effect=error):
                    status, payload = self.post(harness, {})
                self.assertEqual((status, payload["error"]["code"]), (503, code))

    def test_the_release_needs_the_api_key(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, api_key="test-server-key")
        self.assertEqual(self.post(harness, {})[0], 401)
        self.assertEqual(
            self.post(harness, {"Authorization": "Bearer test-server-key"})[0], 200
        )
        self.assertEqual(runtime.release_calls, 1)


if __name__ == "__main__":
    unittest.main()

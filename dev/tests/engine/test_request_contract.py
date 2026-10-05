import array
import base64
import errno
import http.client
import json
import select
import socket
import struct
import sys
import unittest
from functools import partial
from unittest import mock

from dev.tests.engine import native_peer
from dev.tests.engine.test_documents import pdf_bytes
from dev.tests.engine.test_runtime import FakeFactory, request
from dev.tests.test_server import (
    FOREVER,
    FakeConstraintFactory,
    FakeRuntime,
    Harness,
    no_signed_thinking,
)
from server import documents, runtime, schema_validation, tool_schema
from server import frontend as request_frontend
from server import protocol as wire
from server import server as api
from server.api_shapes import anthropic_to_chat_prompt, normalize_messages
from server.errors import APIError


class RequestContractTests(unittest.TestCase):
    def test_optional_null_parameters_and_timeout_cap(self):
        harness = Harness(FakeRuntime())
        self.addCleanup(harness.close)
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
            "stream": None,
            "temperature": None,
            "top_p": None,
            "n": None,
            "parallel_tool_calls": None,
            "presence_penalty": None,
            "frequency_penalty": None,
        }
        status, _, payload = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200, payload)
        deadline = harness.app.request_deadline({"timeout": 1e6}, 10)
        self.assertEqual(deadline, 10 + harness.app.request_timeout)

    def test_invalid_generation_fields_fail_before_document_rendering(self):
        constraints = FakeConstraintFactory()
        harness = Harness(FakeRuntime(), constraint_factory=constraints)
        self.addCleanup(harness.close)
        pdf = base64.b64encode(pdf_bytes(pages=1)).decode()
        document = {
            "type": "file",
            "file": {"file_data": "data:application/pdf;base64," + pdf},
        }
        tool = {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
        rendering = "the PDF reached rendering"
        invalid = (
            ({"temperature": 5}, "temperature must be"),
            ({"seed": -1}, "seed must be"),
            ({"priority": "urgent"}, "priority must be"),
            ({"n": 2}, "n and logprobs"),
        )
        # The valid request shows that the PDF otherwise reaches the renderer.
        for fields, message in (*invalid, ({}, rendering)):
            with (
                self.subTest(fields=fields),
                mock.patch.dict(documents._cache, clear=True),
                mock.patch.object(
                    documents, "_render", side_effect=APIError(400, rendering)
                ) as render,
            ):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": [document]}],
                        **fields,
                    },
                )
                self.assertEqual(status, 400, payload)
                self.assertIn(message, json.loads(payload)["error"]["message"])
                self.assertEqual(render.called, not fields)
        for fields, message in invalid:
            with self.subTest(fields=fields, tools=True):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Look it up."}],
                        "tools": [tool],
                        **fields,
                    },
                )
                self.assertEqual(status, 400, payload)
                self.assertIn(message, json.loads(payload)["error"]["message"])
                self.assertEqual(constraints.grammars, [])

    def test_enabled_thinking_honors_effort(self):
        for effort in ("low", "medium", "high", "xhigh", "max"):
            converted = anthropic_to_chat_prompt(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 4096,
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "output_config": {"effort": effort},
                },
                thinking_resolver=no_signed_thinking,
            )
            self.assertEqual(converted["reasoning_effort"], effort)

    def test_tool_call_id_survives_normalization(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_42",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"value":1}'},
                    }
                ],
            }
        ]
        self.assertEqual(
            normalize_messages(
                [{"role": "user", "content": "Hi"}, *messages],
                vision=True,
                deadline=FOREVER,
            )[1]["tool_calls"][0]["id"],
            "call_42",
        )

    def test_trailing_bytes_do_not_hide_connection_close(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        handler = object.__new__(api.FrontendHandler)
        handler.connection = local
        self.assertFalse(handler._client_disconnected())
        peer.sendall(b"trailing")
        peer.close()
        self.assertFalse(handler._client_disconnected())
        self.assertTrue(handler._client_disconnected())

    def test_descriptor_exhaustion_does_not_read_as_a_disconnect(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        handler = object.__new__(api.FrontendHandler)
        handler.connection = local
        exhausted = OSError(errno.EMFILE, "Too many open files")
        with mock.patch("select.kqueue", side_effect=exhausted):
            self.assertFalse(handler._client_disconnected())

    def test_reset_client_reads_as_disconnected(self):
        listener = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(listener.close)
        client = socket.create_connection(listener.getsockname())
        accepted, _ = listener.accept()
        self.addCleanup(accepted.close)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        self.assertTrue(select.select([accepted], [], [], 2)[0])
        handler = object.__new__(api.FrontendHandler)
        handler.connection = accepted
        self.assertTrue(handler._client_disconnected())

    def test_unsupported_http_version_is_rejected_before_header_validation(self):
        handler = object.__new__(api.FrontendHandler)
        handler._header_timer = mock.Mock()
        handler.server = mock.Mock()
        handler.connection = mock.Mock()
        handler.request_version = "HTTP/0.9"
        handler.headers = {}
        handler.send_error = mock.Mock()
        with mock.patch.object(
            api.BaseHTTPRequestHandler, "parse_request", return_value=True
        ):
            self.assertFalse(handler.parse_request())
        handler.send_error.assert_called_once_with(505, "HTTP version not supported")
        self.assertTrue(handler.close_connection)

    def test_unparsable_requests_get_an_http_1_1_json_error(self):
        harness = Harness(FakeRuntime())
        self.addCleanup(harness.close)
        # Each request is all the server reads: bytes it left unread would
        # reset the connection under its response.
        for data, status, anthropic in (
            (b"GARBAGE\r\n", 400, False),
            (b"GET / FOO/1.1\r\n", 400, False),
            # HTTP/0.9's request line, which has no version. Python before
            # 3.13 reads headers after it all the same.
            (b"GET /\r\n" + b"\r\n" * (sys.version_info < (3, 13)), 505, False),
            # One byte over the stdlib's request and header line limits.
            (b"x" * 65537, 414, False),
            (b"POST /v1/messages HTTP/1.1\r\n" + b"x" * 65537, 431, True),
        ):
            with self.subTest(data=data[:24]):
                client = socket.create_connection(
                    harness.server.server_address, timeout=2
                )
                self.addCleanup(client.close)
                client.sendall(data)
                response = http.client.HTTPResponse(client)
                response.begin()
                self.assertEqual((response.version, response.status), (11, status))
                self.assertEqual(response.getheader("Connection"), "close")
                self.assertEqual(response.getheader("Content-Type"), "application/json")
                payload = json.loads(response.read())
                self.assertEqual(payload.get("type"), "error" if anthropic else None)
                self.assertTrue(payload["error"]["message"])

    def test_missing_native_executable_has_upgrade_guidance(self):
        with self.assertRaisesRegex(
            runtime.EngineUnhealthy, "restart Splash from the current installation"
        ):
            runtime.MultiplexedRuntime(
                process_factory=mock.Mock(side_effect=FileNotFoundError()),
                eager_start=True,
            )

    def test_head_and_options(self):
        harness = Harness(FakeRuntime())
        self.addCleanup(harness.close)
        status, headers, payload = harness.request("HEAD", "/health")
        self.assertEqual((status, payload), (200, b""))
        self.assertNotIn("Python", str(headers))
        self.assertEqual(harness.request("OPTIONS", "/v1/messages")[0], 204)

    def test_stream_failure_before_admission_preserves_retryable_http_status(self):
        class DelayedHandler(api.FrontendHandler):
            def setup(self):
                super().setup()
                self._last_sse_write = 0.0

        class RejectBeforeStart(FakeRuntime):
            def _run(self, call):
                call.complete(
                    error=api.APIError(503, "retry later", "resource_timeout")
                )

        harness = Harness(RejectBeforeStart())
        harness.server.RequestHandlerClass = DelayedHandler
        self.addCleanup(harness.close)
        for path in (
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
        ):
            body = {"model": "test-model", "stream": True, "max_tokens": 16}
            if path == "/v1/responses":
                body["input"] = "hello"
            elif path == "/v1/completions":
                body["prompt"] = "hello"
            else:
                body["messages"] = [{"role": "user", "content": "hello"}]
            with self.subTest(path=path):
                status, headers, payload = harness.request("POST", path, body)
                self.assertEqual(status, 503, payload)
                self.assertIn("application/json", headers)

    def test_template_diagnostic_points_to_template_without_logging_content(self):
        harness = Harness(FakeRuntime())
        self.addCleanup(harness.close)

        def helper():
            raise ValueError("private message contents")

        namespace = {"helper": helper}
        exec(
            compile("def render(*args):\n    helper()\n", "<template>", "exec"),
            namespace,
        )
        with (
            mock.patch.object(
                harness.app, "_apply_chat_template", side_effect=namespace["render"]
            ),
            mock.patch.object(request_frontend, "print_status") as log,
        ):
            with self.assertRaises(api.APIError):
                harness.app.prepare(
                    {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    deadline=FOREVER,
                )
        self.assertEqual(
            log.call_args.args[0], "Template error · ValueError · <template>:2"
        )

    def test_schema_validation_is_bounded_and_preserves_patterns(self):
        schema = {
            "type": "object",
            "patternProperties": {"^key": {"type": "integer"}},
            "additionalProperties": False,
        }
        validator = schema_validation.build_validator(schema)
        self.assertTrue(validator.is_valid({"key_one": 1}))
        self.assertFalse(validator.is_valid({"other": 1}))
        self.assertFalse(validator.is_valid({"key_one": "1"}))
        # Exercise the timeout path without an adversarial pattern or workload.
        with mock.patch.object(
            schema_validation.regex, "search", side_effect=TimeoutError
        ):
            with self.assertRaises(schema_validation.SchemaEvaluationError):
                validator.is_valid({"key_one": 1})

    def test_build_validator_reuses_a_cached_instance_for_the_same_schema(self):
        schema_a = {"type": "object", "properties": {"x": {"type": "integer"}}}
        schema_b = {"type": "object", "properties": {"x": {"type": "string"}}}
        first = schema_validation.build_validator(schema_a)
        second = schema_validation.build_validator(dict(schema_a))
        third = schema_validation.build_validator(schema_b)
        self.assertIs(first, second)
        self.assertIsNot(first, third)

    def test_nonstring_schema_is_a_request_error(self):
        for value in ([], {}, 1):
            schema = {"type": "object", "$schema": value}
            for normalize in (
                partial(
                    tool_schema.normalize_tools,
                    [
                        {
                            "type": "function",
                            "function": {"name": "test", "parameters": schema},
                        }
                    ],
                    "auto",
                    True,
                ),
                partial(
                    tool_schema.normalize_response_format,
                    {"type": "json_schema", "json_schema": {"schema": schema}},
                ),
            ):
                with (
                    self.subTest(value=value),
                    self.assertRaises(api.APIError) as caught,
                ):
                    normalize()
                self.assertEqual(caught.exception.status, 400)

    def test_schemas_nest_no_deeper_than_they_are_read(self):
        def nested(depth):
            schema = {"type": "string"}
            for _ in range(depth - 1):
                schema = {"type": "array", "items": schema}
            return schema

        for normalize, invalid in (
            (
                lambda schema: tool_schema.normalize_tools(
                    [
                        {
                            "type": "function",
                            "function": {"name": "test", "parameters": schema},
                        }
                    ],
                    "auto",
                    True,
                ),
                "invalid tool schema for test",
            ),
            (
                lambda schema: tool_schema.normalize_response_format(
                    {"type": "json_schema", "json_schema": {"schema": schema}}
                ),
                "invalid response schema",
            ),
        ):
            with self.subTest(invalid=invalid):
                normalize(nested(tool_schema.MAX_SCHEMA_DEPTH))
                with self.assertRaises(api.APIError) as caught:
                    normalize(nested(tool_schema.MAX_SCHEMA_DEPTH + 1))
                self.assertEqual(
                    (caught.exception.status, caught.exception.message),
                    (
                        400,
                        f"{invalid}: nested more than "
                        f"{tool_schema.MAX_SCHEMA_DEPTH} levels deep",
                    ),
                )
        # A schema the parser reads but that would exhaust the stack where it
        # is read recursively is refused as a schema, not as unreadable JSON.
        harness = Harness(FakeRuntime())
        self.addCleanup(harness.close)
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": nested(600)},
                },
            },
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(
            json.loads(payload)["error"]["message"],
            "invalid response schema: nested more than 64 levels deep",
        )
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_mask_byte_payload_round_trips(self):
        response = wire.MaskResponseFrame(
            1, 2, array.array("I", (0, 1, 0xFFFFFFFF, 42)).tobytes()
        )
        encoded = wire.serialize_message(response)
        ((decoded, raw),) = native_peer.ClientFrameReader().feed(encoded)
        self.assertEqual(raw, encoded)
        self.assertEqual(decoded, response)

    def test_unacknowledged_cancel_fails_generation_and_releases_calls(self):
        factory = FakeFactory()
        engine = runtime.MultiplexedRuntime(process_factory=factory, eager_start=True)
        self.addCleanup(engine.close)
        engine._cancel_grace_seconds = 0.02
        call = engine.submit(request(10))
        call.cancel()
        with self.assertRaises(runtime.EngineUnhealthy):
            call.result(1)
        self.assertTrue(call.done)


if __name__ == "__main__":
    unittest.main()

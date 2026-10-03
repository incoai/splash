import json
import unittest
from itertools import product
from unittest import mock

from dev.tests.test_server import FakeConstraintFactory, FakeRuntime, Harness, Plan
from server import output as model_output
from server import schema_validation, tool_schema
from server import server as api


class OutputValidationTests(unittest.TestCase):
    def test_evaluation_failures_are_distinct_from_invalid_output(self):
        schema = {"type": "string", "pattern": "^Paris$"}
        _, validator = tool_schema.normalize_response_format(
            {"type": "json_schema", "json_schema": {"schema": schema}}
        )
        _, policy = tool_schema.normalize_tools(
            [{"type": "function", "function": {"name": "echo", "parameters": schema}}],
            "auto",
            True,
        )

        def validate_tool(value):
            model_output.validate_tool_calls(
                [{"function": {"name": "echo", "arguments": value}}], policy
            )

        for validate in (
            lambda value: model_output.validate_response_content(value, validator),
            validate_tool,
        ):
            for failure in (TimeoutError(), schema_validation.regex.error("test")):
                with (
                    self.subTest(failure=type(failure).__name__),
                    mock.patch.object(
                        schema_validation.regex, "search", side_effect=failure
                    ),
                ):
                    with self.assertRaises(api.APIError) as caught:
                        validate('"Paris"')
                    self.assertEqual(caught.exception.status, 500)
                    self.assertEqual(caught.exception.code, "output_validation_failed")
            with self.assertRaises(api.APIError) as caught:
                validate('"London"')
            self.assertEqual(caught.exception.status, 500)
            self.assertEqual(caught.exception.code, "invalid_model_output")

    def test_reference_lookup_crashes_are_evaluation_failures(self):
        # referencing's draft 3 crawls an extends object's keys as schemas once a
        # lookup scans the document, here for the nested id; the AttributeError
        # escaped validation as an internal server error.
        schema = {
            "$schema": "http://json-schema.org/draft-03/schema#",
            "type": "object",
            "extends": {"type": "object"},
            "definitions": {"q": {"type": "string"}},
            "properties": {
                "a": {
                    "id": "https://other.invalid/x.json",
                    "type": "object",
                    "properties": {"b": {"$ref": "#/definitions/q"}},
                }
            },
        }
        _, validator = tool_schema.normalize_response_format(
            {"type": "json_schema", "json_schema": {"schema": schema}}
        )
        _, policy = tool_schema.normalize_tools(
            [{"type": "function", "function": {"name": "echo", "parameters": schema}}],
            "auto",
            True,
        )
        for validate in (
            lambda value: model_output.validate_response_content(value, validator),
            lambda value: model_output.validate_tool_calls(
                [{"function": {"name": "echo", "arguments": value}}], policy
            ),
        ):
            with self.assertRaises(api.APIError) as caught:
                validate('{"a": {"b": "y"}}')
            self.assertEqual(caught.exception.status, 500)
            self.assertEqual(caught.exception.code, "output_validation_failed")

    def test_invalid_request_schemas_remain_client_errors(self):
        schema = {"type": 7}
        for normalize in (
            lambda: tool_schema.normalize_response_format(
                {"type": "json_schema", "json_schema": {"schema": schema}}
            ),
            lambda: tool_schema.normalize_tools(
                [
                    {
                        "type": "function",
                        "function": {"name": "echo", "parameters": schema},
                    }
                ],
                "auto",
                True,
            ),
        ):
            with self.assertRaises(api.APIError) as caught:
                normalize()
            self.assertEqual(caught.exception.status, 400)
            self.assertEqual(caught.exception.code, "invalid_request_error")

    def test_protocols_report_evaluation_failure_and_accept_next_request(self):
        schema = {"type": "object", "patternProperties": {".*": {}}}
        function = {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "pattern": "^Paris$"}},
            },
        }
        for path, tools, stream in product(
            ("/v1/chat/completions", "/v1/responses", "/v1/messages"),
            (False, True),
            (False, True),
        ):
            with self.subTest(path=path, tools=tools, stream=stream):
                token = 5 if tools else 10
                harness = Harness(
                    FakeRuntime(Plan([[token]]), Plan([[token]])),
                    constraint_factory=FakeConstraintFactory(),
                )
                body = {"model": "test-model", "stream": stream}
                if path == "/v1/responses":
                    body.update(
                        input="hello",
                        max_output_tokens=16,
                        reasoning={"effort": "none"},
                    )
                    if tools:
                        body["tools"] = [{"type": "function", **function}]
                    else:
                        body["text"] = {
                            "format": {
                                "type": "json_schema",
                                "name": "answer",
                                "schema": schema,
                            }
                        }
                else:
                    body.update(
                        messages=[{"role": "user", "content": "hello"}], max_tokens=16
                    )
                    if path == "/v1/messages":
                        body["thinking"] = {"type": "disabled"}
                        if tools:
                            body["tools"] = [
                                {
                                    "name": "weather",
                                    "input_schema": function["parameters"],
                                }
                            ]
                        else:
                            body["output_config"] = {
                                "format": {"type": "json_schema", "schema": schema}
                            }
                    else:
                        body["reasoning_effort"] = "none"
                        if tools:
                            body["tools"] = [{"type": "function", "function": function}]
                        else:
                            body["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {"name": "answer", "schema": schema},
                            }
                try:
                    with mock.patch.object(
                        schema_validation.regex, "search", side_effect=TimeoutError
                    ):
                        status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200 if stream else 500, payload)
                    if stream:
                        events = [
                            json.loads(line[6:])
                            for line in payload.decode().splitlines()
                            if line.startswith("data: ") and line != "data: [DONE]"
                        ]
                        errors = [
                            event["error"] for event in events if "error" in event
                        ]
                        errors += [
                            event["response"]["error"]
                            for event in events
                            if event.get("type") == "response.failed"
                        ]
                    else:
                        errors = [json.loads(payload)["error"]]
                    self.assertEqual(len(errors), 1, payload)
                    if path != "/v1/messages":
                        self.assertEqual(errors[0]["code"], "output_validation_failed")
                    self.assertEqual(
                        errors[0]["message"],
                        "schema pattern exceeded its evaluation time limit",
                    )
                    self.assertEqual(
                        errors[0]["type"],
                        "api_error" if path == "/v1/messages" else "server_error",
                    )
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200, payload)
                    self.assertNotIn(b"output_validation_failed", payload)
                    self.assertNotIn(b'"type":"error"', payload)
                finally:
                    harness.close()


if __name__ == "__main__":
    unittest.main()

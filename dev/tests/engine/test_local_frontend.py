import base64
import copy
import http.client
import json
import threading
import unittest
from unittest import mock

from jinja2 import TemplateError

from server import api_shapes
from server import frontend as request_frontend
from server import server as api
from server.errors import APIError
from server.thinking import ThinkingCodec


class StrictSystemTokenizer:
    """Model the Flash chat template's single leading system message rule."""

    def __init__(self):
        self.attempts = []
        self.templates = []

    def apply_chat_template(self, messages, **kwargs):
        self.attempts.append(copy.deepcopy(messages))
        if any(message["role"] == "system" for message in messages[1:]):
            raise TemplateError("System message must be at the beginning.")
        self.templates.append((copy.deepcopy(messages), kwargs))
        prefix = "<|im_start|>assistant\n<think>\n"
        if not kwargs.get("enable_thinking", True):
            prefix += "\n</think>\n\n"
        return prefix

    def __call__(self, text, **kwargs):
        return {"input_ids": [101, 102]}

    def convert_tokens_to_ids(self, token):
        return 248069 if token == "</think>" else None


class LocalHTTPFixture:
    """Exercise HTTP conversion with a mock backend and no native worker."""

    def __init__(self, app):
        self.server = api.FrontendServer(("127.0.0.1", 0), app, webui=False)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def request(self, path, body):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request(
                "POST", path, json.dumps(body), {"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "local_answer", "strict": True, "schema": SCHEMA},
}
IMAGE_URL = "data:image/png;base64,cG5n"
PDF_FILE = {
    "filename": "local.pdf",
    "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
}
THINKING_KEY = base64.urlsafe_b64encode(bytes(range(32)))


class LocalFrontendTests(unittest.TestCase):
    def frontend(self, tokenizer=None, backend=None):
        tokenizer = tokenizer or StrictSystemTokenizer()
        backend = backend if backend is not None else mock.Mock()
        app = request_frontend.Frontend(
            tokenizer,
            backend,
            "test-model",
            512,
            16,
            10,
            2,
            thinking_codec=ThinkingCodec(key=THINKING_KEY),
            input_modalities=("text",),
        )
        return app, tokenizer, backend

    def body(self, **fields):
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello."}],
            "reasoning_effort": "none",
            "max_completion_tokens": 16,
            **fields,
        }

    def test_text_only_frontend_prepares_text_without_image_or_native_work(self):
        app, tokenizer, backend = self.frontend()
        with mock.patch.object(app.images, "prepare") as images:
            job, thinking, has_tools = app.prepare(
                self.body(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Discuss images and PDF files as plain text.",
                                }
                            ],
                        }
                    ]
                )
            )
        self.assertFalse(thinking)
        self.assertFalse(has_tools)
        self.assertEqual(job.image_spans, ())
        self.assertEqual(job.image_pixels, b"")
        self.assertTrue(tokenizer.templates)
        images.assert_not_called()
        self.assertEqual(backend.mock_calls, [])

    def test_chat_modalities_reject_before_document_image_or_template_processing(self):
        parts = (
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            {"type": "file", "file": PDF_FILE},
        )
        for role in ("user", "tool"):
            for part in parts:
                for method in ("prepare", "count_tokens", "apply_template"):
                    with self.subTest(role=role, part=part["type"], method=method):
                        app, tokenizer, backend = self.frontend()
                        messages = [{"role": role, "content": [part]}]
                        if role == "tool":
                            messages.insert(0, {"role": "user", "content": "Read it."})
                            messages[-1]["tool_call_id"] = "call1"
                        with (
                            mock.patch.object(api_shapes, "file_content") as files,
                            mock.patch.object(
                                api_shapes, "document_content"
                            ) as documents,
                            mock.patch.object(app.images, "prepare") as images,
                            self.assertRaises(APIError) as caught,
                        ):
                            getattr(app, method)(self.body(messages=messages))
                        self.assertEqual(caught.exception.status, 400)
                        files.assert_not_called()
                        documents.assert_not_called()
                        images.assert_not_called()
                        self.assertEqual(tokenizer.attempts, [])
                        self.assertEqual(backend.mock_calls, [])

    def test_responses_modalities_reject_before_document_or_image_processing(self):
        parts = (
            {"type": "input_image", "image_url": IMAGE_URL},
            {"type": "input_file", **PDF_FILE},
        )
        for tool_output in (False, True):
            for part in parts:
                with self.subTest(tool_output=tool_output, part=part["type"]):
                    app, tokenizer, backend = self.frontend()
                    item = (
                        {
                            "type": "function_call_output",
                            "call_id": "call1",
                            "output": [part],
                        }
                        if tool_output
                        else {"role": "user", "content": [part]}
                    )
                    inputs = [{"role": "user", "content": "Read it."}, item]
                    with (
                        mock.patch.object(api_shapes, "file_content") as files,
                        mock.patch.object(api_shapes, "document_content") as documents,
                        mock.patch.object(app.images, "prepare") as images,
                        self.assertRaises(APIError) as caught,
                    ):
                        app.prepare_responses(
                            {
                                "model": "test-model",
                                "input": inputs,
                                "max_output_tokens": 16,
                            }
                        )
                    self.assertEqual(caught.exception.status, 400)
                    files.assert_not_called()
                    documents.assert_not_called()
                    images.assert_not_called()
                    self.assertEqual(tokenizer.attempts, [])
                    self.assertEqual(backend.mock_calls, [])

    def test_responses_inherited_attachments_reject_before_processing(self):
        for part in (
            {"type": "input_image", "image_url": IMAGE_URL},
            {"type": "input_file", **PDF_FILE},
        ):
            with self.subTest(part=part["type"]):
                app, tokenizer, backend = self.frontend()
                app.response_store.put(
                    {"id": "resp-old"},
                    [{"role": "user", "content": [part]}],
                )
                with (
                    mock.patch.object(api_shapes, "file_content") as files,
                    mock.patch.object(api_shapes, "document_content") as documents,
                    mock.patch.object(app.images, "prepare") as images,
                    self.assertRaises(APIError) as caught,
                ):
                    app.prepare_responses(
                        {
                            "model": "test-model",
                            "previous_response_id": "resp-old",
                            "input": "Continue.",
                            "max_output_tokens": 16,
                        }
                    )
                self.assertEqual(caught.exception.status, 400)
                files.assert_not_called()
                documents.assert_not_called()
                images.assert_not_called()
                self.assertEqual(tokenizer.attempts, [])
                self.assertEqual(backend.mock_calls, [])

    def test_image_like_tool_arguments_and_schemas_are_plain_data(self):
        app, tokenizer, backend = self.frontend()
        image_like = {"type": "image", "image_url": {"url": IMAGE_URL}}
        schema = {"type": "object", "examples": [image_like]}
        body = self.body(
            messages=[
                {"role": "user", "content": "Describe the data."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call1",
                            "type": "function",
                            "function": {"name": "describe", "arguments": image_like},
                        }
                    ],
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "describe", "parameters": schema},
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "description", "schema": schema},
            },
        )
        job, _, has_tools = app.prepare(body)
        self.assertTrue(has_tools)
        self.assertEqual(job.image_spans, ())
        self.assertTrue(tokenizer.templates)
        self.assertEqual(backend.mock_calls, [])

    def test_default_frontend_keeps_existing_multimodal_input_policy(self):
        app = request_frontend.Frontend(
            StrictSystemTokenizer(),
            None,
            "test-model",
            512,
            16,
            10,
            2,
            thinking_codec=ThinkingCodec(key=THINKING_KEY),
        )
        for part in (
            {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            {"type": "file", "file": PDF_FILE},
        ):
            with self.subTest(part=part["type"]):
                app.validate_input(
                    self.body(messages=[{"role": "user", "content": [part]}])
                )

    def test_status_reports_text_only_capabilities(self):
        backend = mock.Mock()
        backend.status.return_value = {"ready": True}
        app, _, _ = self.frontend(backend=backend)
        status = app.status()
        self.assertEqual(status["capabilities"]["input_modalities"], ["text"])
        self.assertEqual(status["capabilities"]["output_modalities"], ["text"])

    def test_anthropic_routes_reject_raw_modalities_before_conversion(self):
        tokenizer = StrictSystemTokenizer()
        app, _, backend = self.frontend(tokenizer)
        harness = LocalHTTPFixture(app)
        self.addCleanup(harness.close)
        parts = (
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "cG5n"},
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjQ=",
                },
            },
        )
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            for tool_result in (False, True):
                for part in parts:
                    with self.subTest(
                        path=path, tool_result=tool_result, part=part["type"]
                    ):
                        content = (
                            [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "call1",
                                    "content": [part],
                                }
                            ]
                            if tool_result
                            else [part]
                        )
                        with (
                            mock.patch.object(
                                api_shapes, "document_content"
                            ) as documents,
                            mock.patch.object(app.images, "prepare") as images,
                        ):
                            status, payload = harness.request(
                                path,
                                {
                                    "model": "test-model",
                                    "messages": [{"role": "user", "content": content}],
                                    "max_tokens": 16,
                                },
                            )
                        self.assertEqual(status, 400, payload)
                        self.assertIn(
                            "support", json.loads(payload)["error"]["message"].lower()
                        )
                        documents.assert_not_called()
                        images.assert_not_called()
        self.assertEqual(tokenizer.attempts, [])
        backend.submit.assert_not_called()
        backend.cancel.assert_not_called()

    def test_leading_system_developer_and_schema_coalesce_in_order(self):
        app, tokenizer, _ = self.frontend()
        body = self.body(
            messages=[
                {"role": "system", "content": "First system instruction."},
                {
                    "role": "developer",
                    "content": [
                        {"type": "text", "text": "Second developer instruction."}
                    ],
                },
                {"role": "system", "content": "Third system instruction."},
                {"role": "user", "content": "Answer with JSON."},
            ],
            response_format=RESPONSE_FORMAT,
        )
        original = copy.deepcopy(body)
        job, _, _ = app.prepare(body)
        self.assertIsNotNone(job.response_validator)
        messages = tokenizer.attempts[-1]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        system = messages[0]["content"]
        instructions = (
            "First system instruction.",
            "Second developer instruction.",
            "Third system instruction.",
            "Your final answer must be a JSON value",
        )
        positions = [system.index(instruction) for instruction in instructions]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('"answer"', system)
        self.assertEqual(messages[1]["content"], "Answer with JSON.")
        self.assertEqual(body, original)

    def test_single_leading_developer_with_schema_renders(self):
        app, tokenizer, _ = self.frontend()
        app.apply_template(
            self.body(
                messages=[
                    {
                        "role": "developer",
                        "content": "Follow the developer instruction.",
                    },
                    {"role": "user", "content": "Answer."},
                ],
                response_format=RESPONSE_FORMAT,
            )
        )
        messages = tokenizer.attempts[-1]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Follow the developer instruction.", messages[0]["content"])
        self.assertIn("Your final answer must be a JSON value", messages[0]["content"])

    def test_later_system_or_developer_is_not_hoisted_or_coalesced(self):
        for role in ("system", "developer"):
            with self.subTest(role=role):
                app, tokenizer, _ = self.frontend()
                body = self.body(
                    messages=[
                        {"role": "system", "content": "Leading instruction."},
                        {"role": "user", "content": "First turn."},
                        {"role": role, "content": "Later instruction."},
                        {"role": "user", "content": "Second turn."},
                    ]
                )
                with self.assertRaises(APIError) as caught:
                    app.apply_template(body)
                self.assertEqual(caught.exception.status, 400)
                messages = tokenizer.attempts[-1]
                self.assertEqual(
                    [message["role"] for message in messages],
                    ["system", "user", "system", "user"],
                )
                self.assertEqual(messages[2]["content"], "Later instruction.")
                self.assertNotIn("Later instruction.", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()

import base64
import json
import time
import unittest
from unittest import mock

from dev.tests import test_server
from dev.tests.engine.test_documents import pdf_bytes, render_pdf
from dev.tests.test_server import FOREVER, FakeRuntime, Harness, Plan
from server import api_shapes, documents
from server.errors import APIError


class PdfProtocolTests(unittest.TestCase):
    def setUp(self):
        self.encoded = base64.b64encode(pdf_bytes(pages=2)).decode()
        self.file = {
            "filename": "receipt.pdf",
            "file_data": "data:application/pdf;base64," + self.encoded,
        }

    def chat(self, file=None, deadline=FOREVER):
        return api_shapes.normalize_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": self.file if file is None else file}
                    ],
                }
            ],
            vision=True,
            deadline=deadline,
        )

    def test_protocols_share_rendered_pages_and_preserve_order(self):
        expected = render_pdf(pdf_bytes(pages=2))
        chat = self.chat()[0]["content"]
        self.assertEqual(chat, expected)
        response, _ = api_shapes.responses_to_chat_body(
            {},
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "before"},
                        {"type": "input_file", **self.file},
                        {"type": "input_text", "text": "after"},
                    ],
                }
            ],
        )
        actual = api_shapes.normalize_messages(
            response["messages"], vision=True, deadline=FOREVER
        )[0]["content"]
        self.assertEqual(
            actual,
            [
                {"type": "text", "text": "before"},
                *expected,
                {"type": "text", "text": "after"},
            ],
        )
        self.assertEqual(sum(p["type"] == "image_url" for p in actual), 2)

    def test_http_generation_prepares_page_images_in_both_protocols(self):
        for dialect in ("chat", "responses"):
            for stream in (False, True):
                with self.subTest(dialect=dialect, stream=stream):
                    runtime = FakeRuntime(Plan([[1, 2, 3]]))
                    harness = Harness(
                        runtime,
                        tokenizer=test_server.ServerTest.ImagePadTokenizer(),
                        max_context=65536,
                        timeout=10,
                    )
                    try:
                        body = {"model": "test-model", "stream": stream}
                        if dialect == "chat":
                            body.update(
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "file", "file": self.file}
                                        ],
                                    }
                                ],
                                max_tokens=16,
                            )
                            path = "/v1/chat/completions"
                        else:
                            body.update(
                                input=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "input_file", **self.file}
                                        ],
                                    }
                                ],
                                max_output_tokens=16,
                            )
                            path = "/v1/responses"
                        status, _, payload = harness.request("POST", path, body)
                        self.assertEqual(status, 200, payload)
                        self.assertEqual(len(runtime.requests), 1)
                        self.assertEqual(len(runtime.requests[0].frame.image_spans), 2)
                        if stream:
                            self.assertIn(b"data:", payload)
                        else:
                            self.assertIn("id", json.loads(payload))
                    finally:
                        harness.close()

    def test_responses_history_keeps_pdf_on_followup(self):
        runtime = FakeRuntime(Plan([[1]]), Plan([[1]]))
        harness = Harness(
            runtime,
            tokenizer=test_server.ServerTest.ImagePadTokenizer(),
            max_context=65536,
            timeout=10,
        )
        self.addCleanup(harness.close)
        body = {
            "model": "test-model",
            "max_output_tokens": 16,
            "input": [
                {"role": "user", "content": [{"type": "input_file", **self.file}]}
            ],
        }
        status, _, payload = harness.request("POST", "/v1/responses", body)
        self.assertEqual(status, 200, payload)
        response = json.loads(payload)
        self.assertTrue(response["store"])
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            {
                "model": "test-model",
                "max_output_tokens": 16,
                "previous_response_id": response["id"],
                "input": "Read the second page again.",
            },
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual([len(r.frame.image_spans) for r in runtime.requests], [2, 2])

    def test_unsupported_file_rejected_before_stream_or_runtime(self):
        runtime = FakeRuntime()
        harness = Harness(runtime)
        self.addCleanup(harness.close)
        for path, body in (
            (
                "/v1/chat/completions",
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "file", "file": {"file_id": "file-123"}}
                            ],
                        }
                    ]
                },
            ),
            (
                "/v1/responses",
                {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_file",
                                    "file_url": "https://example.com/file.pdf",
                                }
                            ],
                        }
                    ]
                },
            ),
        ):
            with self.subTest(path=path):
                status, content_type, payload = harness.request(
                    "POST", path, {"model": "test-model", "stream": True, **body}
                )
                self.assertEqual(status, 400, payload)
                self.assertIn("application/json", content_type)
        self.assertEqual(runtime.requests, [])

    def test_plain_base64_with_optional_filename(self):
        expected = self.chat()[0]["content"]
        for file in (
            {"file_data": self.encoded},
            {"filename": "scan.PDF", "file_data": self.encoded},
        ):
            with self.subTest(filename=file.get("filename")):
                self.assertEqual(self.chat(file)[0]["content"], expected)

    def test_responses_defers_rendering_until_request_preparation(self):
        with mock.patch.object(
            documents, "_render", side_effect=AssertionError("early render")
        ):
            response, _ = api_shapes.responses_to_chat_body(
                {},
                [
                    {
                        "role": "user",
                        "content": [{"type": "input_file", **self.file}],
                    }
                ],
            )
        self.assertEqual(
            response["messages"][0]["content"], [{"type": "file", "file": self.file}]
        )

    def test_user_and_tool_files_share_one_request_budget(self):
        budget = documents.DocumentBudget(deadline=FOREVER)
        documents.file_content(self.file, budget=budget)
        one_file = documents.MAX_REQUEST_DOCUMENT_BYTES - budget.remaining_bytes

        def normalize(messages, remaining_bytes):
            request_budget = documents.DocumentBudget(
                deadline=FOREVER, remaining_bytes=remaining_bytes
            )
            with mock.patch.object(
                api_shapes, "DocumentBudget", return_value=request_budget
            ):
                api_shapes.normalize_messages(messages, vision=True, deadline=FOREVER)

        for dialect in ("chat", "responses"):
            with self.subTest(dialect=dialect):
                if dialect == "chat":
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "file", "file": self.file}],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "read",
                            "content": [{"type": "file", "file": self.file}],
                        },
                    ]
                else:
                    items = [
                        {
                            "role": "user",
                            "content": [{"type": "input_file", **self.file}],
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "read",
                            "output": [{"type": "input_file", **self.file}],
                        },
                    ]
                    messages = api_shapes.responses_to_chat_body({}, items)[0][
                        "messages"
                    ]
                normalize(messages, 2 * one_file)
                with self.assertRaisesRegex(APIError, "request size limit"):
                    normalize(messages, 2 * one_file - 1)

    def test_deadline_applies_even_to_cached_pdf(self):
        self.chat()
        with self.assertRaises(APIError) as raised:
            self.chat(deadline=time.monotonic() - 1)
        self.assertEqual(raised.exception.status, 504)

    def test_invalid_and_unsupported_files_fail_before_render(self):
        values = [
            None,
            "file.pdf",
            {},
            {"filename": 7, "file_data": self.encoded},
            {"file_id": "file-123", **self.file},
            {"file_url": "https://example.com/a.pdf", **self.file},
            {"file_data": "data:text/plain;base64,SGk="},
            {"filename": "receipt.pdf", "file_data": 7},
            {"filename": "receipt.pdf", "file_data": "***"},
        ]
        with mock.patch.object(
            documents,
            "_render",
            side_effect=AssertionError("invalid input reached PDF worker"),
        ):
            for file in values:
                with self.subTest(file=file), self.assertRaises(APIError):
                    api_shapes.normalize_messages(
                        [{"role": "user", "content": [{"type": "file", "file": file}]}],
                        vision=True,
                        deadline=FOREVER,
                    )

    def test_input_bound_checked_before_decode(self):
        with (
            mock.patch.object(documents, "MAX_PDF_BYTES", 1),
            mock.patch.object(
                documents.base64,
                "b64decode",
                side_effect=AssertionError("oversize decoded"),
            ),
            self.assertRaisesRegex(APIError, "size limit"),
        ):
            self.chat()


if __name__ == "__main__":
    unittest.main()

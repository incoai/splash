import json
from unittest import mock

from dev.tests.server_fixtures import (
    FOREVER,
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    ImagePadTokenizer,
    chat_body,
    png_data_url,
    serve_context,
)
from server import frontend
from server.errors import APIError


class ChatCountTokensTest(HarnessTestCase):
    def test_counts_the_generation_prompt_without_inference(self):
        class InputTokenizer(FakeTokenizer):
            def apply_chat_template(
                self, messages, add_generation_prompt=False, **options
            ):
                rendered = super().apply_chat_template(
                    messages, add_generation_prompt=add_generation_prompt, **options
                )
                return json.dumps([messages, options], sort_keys=True) + rendered

            def __call__(self, text, **options):
                return {"input_ids": list(text.encode())}

        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=InputTokenizer(), max_context=32768)
        history = [
            {"role": "user", "content": "你好 👋"},
            {
                "role": "assistant",
                "content": "Hello",
                "reasoning_content": "A greeting.",
            },
            {"role": "user", "content": "And then?"},
        ]
        counts = []
        for effort in ("none", "high"):
            with self.subTest(effort=effort):
                body = chat_body(messages=history, reasoning_effort=effort)
                with mock.patch.object(
                    frontend, "Job", side_effect=AssertionError("inference")
                ):
                    status, _, payload = harness.request(
                        "POST", "/v1/chat/completions/count_tokens?beta=true", body
                    )
                self.assertEqual(status, 200, payload)
                count = json.loads(payload)
                job = harness.app.prepare(body, deadline=FOREVER)
                self.assertEqual(count, {"input_tokens": len(job.prompt_tokens)})
                counts.append(count["input_tokens"])
        self.assertEqual(len(counts), 2)
        self.assertNotEqual(counts[0], counts[1])
        self.assertEqual(runtime.requests, [])

    def test_counts_images_even_when_the_prompt_exceeds_context(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=ImagePadTokenizer(), max_context=16)
        image = {"type": "image_url", "image_url": {"url": png_data_url()}}
        body = chat_body(messages=[{"role": "user", "content": [image, image]}])
        with mock.patch.object(
            harness.app, "_expand_image_pads", side_effect=AssertionError("expansion")
        ):
            status, _, payload = harness.request(
                "POST", "/v1/chat/completions/count_tokens", body
            )
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload), {"input_tokens": 130})
        with self.assertRaisesRegex(APIError, "context window"):
            harness.app.prepare(body, deadline=FOREVER)
        serve_context(harness.app, 256)
        job = harness.app.prepare(body, deadline=FOREVER)
        self.assertEqual(len(job.prompt_tokens), 130)
        self.assertEqual(runtime.requests, [])

    def test_authentication_validation_and_capacity_recovery(self):
        harness = self.harness(api_key="secret")
        path = "/v1/chat/completions/count_tokens"
        self.assertEqual(harness.request("POST", path, chat_body())[0], 401)
        headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
        status, _, payload = harness.request(
            "POST", path, {"messages": "bad"}, headers=headers
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(json.loads(payload)["error"]["type"], "invalid_request_error")
        status, _, payload = harness.request("POST", path, chat_body(), headers=headers)
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload), {"input_tokens": 2})
        self._wait_for_http_active(harness.server.token_counts, 0)

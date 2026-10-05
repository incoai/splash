import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from dev.tests.server_fixtures import (
    FOREVER,
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    anthropic_body,
    chat_body,
    make_frontend,
    responses_body,
)
from server import server as api


class BlockingTokenizer(FakeTokenizer):
    """Holds each render until released, once armed after its chat template
    was probed."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.entered = threading.Event()
        self.active = 0
        self.maximum_active = 0
        self.calls = 0
        self.armed = False

    def apply_chat_template(self, messages, **kwargs):
        if not self.armed:
            return super().apply_chat_template(messages, **kwargs)
        with self.lock:
            self.calls += 1
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            if self.calls >= 2:
                self.entered.set()
        self.release.wait(2)
        try:
            return super().apply_chat_template(messages, **kwargs)
        finally:
            with self.lock:
                self.active -= 1


class PreparationTests(HarnessTestCase):
    def test_frontend_limits_generation_and_token_count_preparation_to_two(self):
        tokenizer = BlockingTokenizer()
        app = make_frontend(
            tokenizer,
            SimpleNamespace(status=lambda: {}),
            "test-model",
            32768,
            2.0,
            2,
            vision=True,
        )
        tokenizer.armed = True
        results = []
        errors = []

        def prepare(index):
            try:
                operation = app.count_tokens if index == 1 else app.prepare
                results.append(operation(chat_body(), deadline=FOREVER))
            except Exception as error:
                errors.append(error)

        threads = [
            threading.Thread(target=prepare, args=(index,)) for index in range(3)
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(tokenizer.entered.wait(1))
        time.sleep(0.05)
        self.assertEqual(tokenizer.calls, 2)
        self.assertEqual(
            app.status()["frontend"],
            {
                "preparation_capacity": 2,
                "active": 2,
                "waiting": 1,
            },
        )
        tokenizer.release.set()
        for thread in threads:
            thread.join(2)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 3)
        self.assertEqual(tokenizer.maximum_active, 2)

    def test_frontend_rejects_invalid_preparation_capacity(self):
        with self.assertRaisesRegex(ValueError, "preparation capacity"):
            make_frontend(
                FakeTokenizer(),
                SimpleNamespace(status=lambda: {}),
                "test-model",
                128,
                1,
                0,
                vision=True,
            )

    def test_preparation_consumes_original_deadline_and_releases_slots(self):
        app = make_frontend(
            FakeTokenizer(), None, "test-model", 128, 10, 1, vision=True
        )
        for elapsed in (0.25, 5):
            clock = [100.0]

            def tokenize(*_args, **_kwargs):
                clock[0] += elapsed
                return "<|im_start|>assistant\n<think>\n"

            with (
                self.subTest(elapsed=elapsed),
                mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]),
                mock.patch.object(
                    app.tokenizer, "apply_chat_template", side_effect=tokenize
                ),
            ):
                body = chat_body(timeout=1)
                deadline = app.request_deadline(body, clock[0])
                if elapsed < 1:
                    job = app.prepare(body, deadline=deadline)
                    self.assertEqual(job.deadline, 101.0)
                else:
                    with self.assertRaises(api.APIError) as error:
                        app.prepare(body, deadline=deadline)
                    self.assertEqual(
                        (error.exception.status, error.exception.code),
                        (504, "request_timeout"),
                    )
                self.assertEqual(
                    (app.preparation_active, app.preparation_waiting), (0, 0)
                )
                self.assertTrue(app.preparation_slots.acquire(blocking=False))
                app.preparation_slots.release()

    def test_preparation_queue_respects_request_timeout(self):
        app = make_frontend(
            FakeTokenizer(), None, "test-model", 128, 10, 1, vision=True
        )
        app.tokenizer.templates.clear()
        app.preparation_slots.acquire()
        try:
            body = chat_body(timeout=0.02)
            with self.assertRaises(api.APIError) as error:
                app.prepare(body, deadline=app.request_deadline(body, time.monotonic()))
            self.assertEqual(
                (error.exception.status, error.exception.code), (504, "request_timeout")
            )
            self.assertEqual(app.tokenizer.templates, [])
            self.assertEqual((app.preparation_active, app.preparation_waiting), (0, 0))
        finally:
            app.preparation_slots.release()

    def test_expired_preparation_skips_later_stages(self):
        app = make_frontend(
            FakeTokenizer(), None, "test-model", 128, 10, 1, vision=True
        )
        for stage in ("grammar", "images"):
            clock = [100.0]

            def expire(*_args, **_kwargs):
                clock[0] += 5
                return []

            factory = mock.Mock()
            app.constraint_factory = factory
            with (
                self.subTest(stage=stage),
                mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]),
                mock.patch.object(app, "_prepare_images", return_value=[]) as images,
                mock.patch.object(
                    app.tokenizer,
                    "apply_chat_template",
                    return_value="<|im_start|>assistant\n<think>\n",
                ) as tokenize,
            ):
                if stage == "grammar":
                    factory.create.side_effect = expire
                else:
                    images.side_effect = expire
                body = chat_body(timeout=1, response_format={"type": "json_object"})
                with self.assertRaises(api.APIError) as error:
                    app.prepare(body, deadline=app.request_deadline(body, clock[0]))
                self.assertEqual(error.exception.status, 504)
                if stage == "grammar":
                    images.assert_called_once()
                    tokenize.assert_called_once()
                else:
                    tokenize.assert_not_called()
                    factory.create.assert_not_called()
                self.assertEqual(app.preparation_active, 0)

    def test_http_body_time_is_included_before_native_admission(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        original = api.FrontendHandler._read_json_body

        def read_body(handler, deadline):
            body = original(handler, deadline)
            time.sleep(0.05)
            return body

        for path, body in (
            ("/v1/chat/completions", chat_body(timeout=0.01)),
            ("/v1/responses", responses_body(timeout=0.01)),
            ("/v1/messages", anthropic_body(timeout=0.01)),
            ("/v1/messages/count_tokens", anthropic_body(timeout=0.01)),
        ):
            with (
                self.subTest(path=path),
                mock.patch.object(api.FrontendHandler, "_read_json_body", read_body),
            ):
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 504, payload)
                self.assertEqual(
                    json.loads(payload)["error"]["type"],
                    "timeout_error"
                    if path.startswith("/v1/messages")
                    else "server_error",
                )
        self.assertEqual(runtime.requests, [])
        self.assertEqual(harness.tokenizer.templates, [])


if __name__ == "__main__":
    unittest.main()

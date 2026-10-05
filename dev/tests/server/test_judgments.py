import hashlib
import http.client
import json
import math
import unittest
from unittest import mock

from dev.tests.server_fixtures import (
    FOREVER,
    CharTokenizer,
    FakeRuntime,
    HarnessTestCase,
    Plan,
    judgment_body,
    make_frontend,
    stays_connected,
)
from server import judgments
from server import server as api


class BoundaryCountingTokenizer(CharTokenizer):
    """Counts whole-prompt tokenizations: one to prepare the prompt plus
    one per answer-slot boundary check. An optional clock advances on each
    of them so a test can expire a deadline inside the boundary pass."""

    def __init__(self, clock=None, step=0.0):
        super().__init__()
        self.clock = clock
        self.step = step
        self.prompt = None
        self.prompt_encodes = 0

    def apply_chat_template(self, messages, **kwargs):
        rendered = super().apply_chat_template(messages, **kwargs)
        if kwargs.get("tokenize") is False:
            self.prompt = rendered
        return rendered

    def encode(self, text, **kwargs):
        if self.prompt is not None and text.startswith(self.prompt):
            self.prompt_encodes += 1
            if self.clock is not None:
                self.clock[0] += self.step
        return super().encode(text, **kwargs)


class JudgmentTests(HarnessTestCase):
    def test_judgments_scores_options_without_generation(self):
        runtime = FakeRuntime(Plan(logits=(1.5, -2.25)))
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        status, content_type, payload = harness.request(
            "POST", "/v1/judgments", judgment_body()
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(content_type, "application/json")
        response = json.loads(payload)
        self.assertEqual(response["id"], "row-1")
        self.assertEqual(response["option_ids"], ["yes", "no"])
        self.assertEqual(response["option_logits"], [1.5, -2.25])
        probabilities = response["probabilities"]
        self.assertEqual(len(probabilities), 2)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])
        expected = math.exp(1.5) / (math.exp(1.5) + math.exp(-2.25))
        self.assertAlmostEqual(probabilities[0], expected)
        self.assertEqual(response["prompt_version"], "direct-options-v1")
        self.assertEqual(
            response["usage"],
            {
                "prompt_tokens": response["input_tokens"],
                "completion_tokens": 0,
                "total_tokens": response["input_tokens"],
            },
        )

        request = runtime.requests[0].frame
        # The rendered prompt plus one slot token is exactly the boundary the
        # engine scores; verify it against the tokenizer, not the response.
        prompt_text = harness.tokenizer.decode(request.prompt_tokens)
        self.assertEqual(
            response["prompt_sha256"],
            hashlib.sha256(prompt_text.encode()).hexdigest(),
        )
        self.assertEqual(response["answer_token_ids"], list(request.score_tokens))
        self.assertEqual(
            harness.tokenizer.encode(prompt_text + "A"),
            list(request.prompt_tokens) + [ord("A")],
        )

    def test_judgments_rejects_invalid_rows_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        cases = (
            {},
            judgment_body(id=""),
            judgment_body(state={}),
            judgment_body(question=42),
            judgment_body(options=[{"id": "a", "description": "x"}]),
            judgment_body(
                options=[
                    {"id": "a", "description": "x"},
                    {"id": "a", "description": "y"},
                ]
            ),
            judgment_body(
                options=[
                    {"id": "a", "description": "x"},
                    {"id": "b"},
                ]
            ),
            judgment_body(
                options=[{"id": str(i), "description": "x"} for i in range(17)]
            ),
            judgment_body(model="other-model"),
            judgment_body(stream=True),
            judgment_body(priority="urgent"),
        )
        for body in cases:
            with self.subTest(body=body):
                status, _, payload = harness.request("POST", "/v1/judgments", body)
                self.assertIn(status, (400, 404), payload)
        self.assertEqual(runtime.requests, [])

    def test_judgments_rejects_nonfinite_state(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        connection.request(
            "POST",
            "/v1/judgments",
            '{"id":"r","state":NaN,"question":"q",'
            '"options":[{"id":"a","description":"x"},'
            '{"id":"b","description":"y"}]}',
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        connection.close()
        self.assertEqual(runtime.requests, [])

    def test_judgments_requires_auth_when_configured(self):
        runtime = FakeRuntime(Plan(logits=(0.0, 1.0)))
        harness = self.harness(
            runtime,
            tokenizer=CharTokenizer(),
            max_context=8192,
            api_key="secret",
        )
        status, _, _ = harness.request("POST", "/v1/judgments", judgment_body())
        self.assertEqual(status, 401)
        status, _, payload = harness.request(
            "POST",
            "/v1/judgments",
            judgment_body(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer secret",
            },
        )
        self.assertEqual(status, 200, payload)

    def test_judgments_deadline_cancels_the_score_request(self):
        plan = Plan(logits=(0.0, 1.0), block=True)
        runtime = FakeRuntime(plan)
        harness = self.harness(
            runtime,
            tokenizer=CharTokenizer(),
            max_context=8192,
            timeout=0.05,
        )
        status, _, payload = harness.request("POST", "/v1/judgments", judgment_body())
        self.assertEqual(status, 504, payload)
        self.assertEqual(runtime.cancel_count, 1)

    def test_judgment_deadline_stops_the_slot_boundary_pass(self):
        clock = [100.0]
        tokenizer = BoundaryCountingTokenizer(clock, 0.5)
        app = make_frontend(tokenizer, None, "test-model", 8192, 10, 2, vision=True)
        body = judgment_body(
            options=[
                {"id": f"opt{index}", "description": f"case {index}"}
                for index in range(16)
            ],
            timeout=1.0,
        )
        deadline = app.request_deadline(body, clock[0])
        with (
            mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaises(api.APIError) as error,
        ):
            app.prepare_judgment(body, deadline=deadline, disconnected=stays_connected)
        self.assertEqual(
            (error.exception.status, error.exception.code), (504, "request_timeout")
        )
        # The prepare pass and one boundary check; all 17 ran before the fix.
        self.assertEqual(tokenizer.prompt_encodes, 2)

    def test_client_disconnect_stops_the_slot_boundary_pass(self):
        tokenizer = BoundaryCountingTokenizer()
        app = make_frontend(tokenizer, None, "test-model", 8192, 10, 2, vision=True)
        body = judgment_body(
            options=[
                {"id": f"opt{index}", "description": f"case {index}"}
                for index in range(16)
            ]
        )
        # Connected through admission and the first boundary check.
        disconnected = iter((False, False, True))
        with self.assertRaises(ConnectionResetError):
            app.prepare_judgment(
                body, deadline=FOREVER, disconnected=lambda: next(disconnected)
            )
        # The prepare pass and one boundary check, and the slot returned.
        self.assertEqual(tokenizer.prompt_encodes, 2)
        self.assertEqual(app.preparation_active, 0)

    def test_judgment_context_budget_precedes_the_slot_boundary_pass(self):
        tokenizer = BoundaryCountingTokenizer()
        app = make_frontend(tokenizer, None, "test-model", 8, 10, 2, vision=True)
        with self.assertRaises(api.APIError) as error:
            app.prepare_judgment(
                judgment_body(
                    options=[
                        {"id": f"opt{index}", "description": f"case {index}"}
                        for index in range(16)
                    ]
                ),
                deadline=FOREVER,
                disconnected=stays_connected,
            )
        self.assertEqual(
            (error.exception.status, error.exception.code),
            (400, "context_length_exceeded"),
        )
        # Only the prepare pass; the 16 boundary checks ran first before the fix.
        self.assertEqual(tokenizer.prompt_encodes, 1)

    def test_systemone_validates_all_questions_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        for invalid in (
            {"type": []},
            {"type": "noul", "criteria": ["yes"]},
            {"type": "score", "criteria": [None]},
            {"type": "choice", "criteria": {str(i): None for i in range(256)}},
        ):
            with self.subTest(question=invalid):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/systemone",
                    {
                        "model": "test-model",
                        "state": {},
                        "questions": {
                            "valid": {"type": "noul", "criteria": {}},
                            "invalid": invalid,
                        },
                    },
                )
                self.assertEqual(status, 422, payload)
                self.assertTrue(
                    any(
                        "invalid" in error["loc"]
                        for error in json.loads(payload)["detail"]
                    )
                )
        self.assertEqual(runtime.requests, [])

    def test_systemone_singleton_domains_need_no_native_request(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer())
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": [],
                "questions": {
                    "choice": {"type": "choice", "criteria": {"only": None}},
                    "score": {
                        "type": "score",
                        "criteria": [{"description": "Only level"}],
                    },
                },
            },
        )
        self.assertEqual(status, 200, payload)
        response = json.loads(payload)
        self.assertEqual(response["answers"]["choice"]["probabilities"], {"only": 1.0})
        self.assertEqual(response["answers"]["score"]["score"], 0.0)
        self.assertEqual(response["usage"], {"input_tokens": 0, "output_tokens": 0})
        self.assertEqual(runtime.requests, [])

    def test_systemone_render_failure_is_a_request_error(self):
        class QuestionRenderFails(CharTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                if messages[0].get("content") == judgments.SYSTEMONE_SYSTEM:
                    raise ValueError("the template cannot render this question")
                return super().apply_chat_template(messages, **kwargs)

        runtime = FakeRuntime()
        harness = self.harness(
            runtime, tokenizer=QuestionRenderFails(), max_context=8192
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": "Some evidence",
                "questions": {"supported": {"type": "noul"}},
            },
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(
            json.loads(payload)["error"]["message"],
            "question prompt could not be rendered",
        )
        self.assertEqual(runtime.requests, [])

    def test_systemone_shared_deadline_cancels_only_current_question(self):
        blocked = Plan(logits=(0.0, 1.0), block=True)
        runtime = FakeRuntime(Plan(logits=(1.0, 0.0)), blocked)
        harness = self.harness(
            runtime,
            tokenizer=CharTokenizer(),
            max_context=8192,
            queue_size=1,
            timeout=1,
        )
        judgments.slot_labels(harness.tokenizer)
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": "Some evidence",
                "questions": {
                    "first": {"type": "noul", "criteria": {}},
                    "blocked": {"type": "noul"},
                    "never_started": {"type": "noul"},
                },
            },
        )
        self.assertEqual(status, 504, payload)
        self.assertEqual(len(runtime.requests), 2)
        self.assertTrue(blocked.cancelled.is_set())
        self.assertEqual(runtime.cancel_count, 1)

    def test_systemone_question_count_budget_rejects_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=8192)
        fitting = {
            f"q{index}": {"type": "choice", "criteria": {"only": None}}
            for index in range(judgments.MAX_SYSTEMONE_QUESTIONS)
        }
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {"model": "test-model", "state": [], "questions": fitting},
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            len(json.loads(payload)["answers"]), judgments.MAX_SYSTEMONE_QUESTIONS
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": [],
                "questions": {**fitting, "extra": {"type": "noul"}},
            },
        )
        self.assertEqual(status, 422, payload)
        self.assertIn("questions", json.loads(payload)["detail"][0]["loc"])
        self.assertEqual(runtime.requests, [])

    def test_systemone_total_token_budget_rejects_before_inference(self):
        class PaddedPromptTokenizer(BoundaryCountingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                self.templates.append((messages, kwargs))
                self.prompt = "p" * 600_000
                return self.prompt

        runtime = FakeRuntime()
        tokenizer = PaddedPromptTokenizer()
        harness = self.harness(runtime, tokenizer=tokenizer, max_context=700_000)
        status, _, payload = harness.request(
            "POST",
            "/v1/systemone",
            {
                "model": "test-model",
                "state": "evidence",
                "questions": {"a": {"type": "noul"}, "b": {"type": "noul"}},
            },
        )
        self.assertEqual(status, 422, payload)
        self.assertIn("total prepared", json.loads(payload)["detail"][0]["msg"])
        self.assertEqual(runtime.requests, [])
        # Question a prepares and checks both slot boundaries; question b is
        # rejected on its prepare pass. Before the fix b also ran both checks.
        self.assertEqual(tokenizer.prompt_encodes, 4)


if __name__ == "__main__":
    unittest.main()

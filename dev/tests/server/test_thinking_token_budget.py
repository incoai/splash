import json
import unittest

from dev.tests import server_fixtures as fixtures
from server import protocol as wire
from server.tool_schema import THINK_END_TOKEN_ID, TOOL_CALL_OPEN_TOKEN_ID

TOOLS = [{"type": "function", "function": {"name": "weather"}}]


def budget(tokens, also_ends=THINK_END_TOKEN_ID):
    return wire.ThinkingBudget(tokens, THINK_END_TOKEN_ID, also_ends)


class ThinkingTokenBudgetTests(unittest.TestCase):
    def harness(self, **options):
        runtime = fixtures.FakeRuntime()
        harness = fixtures.Harness(runtime, **options)
        self.addCleanup(harness.close)
        return harness, runtime

    def post(self, harness, path, body):
        status, _, payload = harness.request("POST", path, body)
        self.assertEqual(status, 200, payload)

    def refused(self, harness, path, body, message):
        status, _, payload = harness.request("POST", path, body)
        self.assertEqual(status, 400, payload)
        error = json.loads(payload)["error"]
        self.assertEqual(error["message"], message)

    def frames(self, runtime):
        return [request.frame.thinking_budget for request in runtime.requests]

    def test_chat_and_responses_carry_the_budget_of_a_thinking_request(self):
        harness, runtime = self.harness()
        for budget_tokens in (0, 7, None, -1):
            self.post(
                harness,
                "/v1/chat/completions",
                fixtures.chat_body(max_tokens=32, thinking_token_budget=budget_tokens),
            )
            self.post(
                harness,
                "/v1/responses",
                fixtures.responses_body(
                    max_output_tokens=32, thinking_token_budget=budget_tokens
                ),
            )
        self.assertEqual(
            self.frames(runtime),
            [budget(0)] * 2 + [budget(7)] * 2 + [None] * 4,
        )

    def test_the_server_default_applies_unless_the_request_sets_a_budget(self):
        harness, runtime = self.harness(default_thinking_token_budget=5)
        for fields in ({}, {"thinking_token_budget": None}):
            self.post(
                harness,
                "/v1/chat/completions",
                fixtures.chat_body(max_tokens=32, **fields),
            )
        for budget_tokens in (9, -1):
            self.post(
                harness,
                "/v1/chat/completions",
                fixtures.chat_body(max_tokens=32, thinking_token_budget=budget_tokens),
            )
        self.assertEqual(self.frames(runtime), [budget(5), budget(5), budget(9), None])

    def test_only_a_budget_below_the_output_limit_of_a_thinking_request_counts(self):
        harness, runtime = self.harness(default_thinking_token_budget=5)
        # Thinking off has no block to close, and a budget the output limit
        # reaches first closes nothing.
        self.post(
            harness,
            "/v1/chat/completions",
            fixtures.chat_body(max_tokens=32, reasoning_effort="none"),
        )
        for max_tokens, expected in ((5, None), (6, budget(5))):
            self.post(
                harness,
                "/v1/chat/completions",
                fixtures.chat_body(max_tokens=max_tokens),
            )
        self.assertEqual(self.frames(runtime), [None, None, budget(5)])

    def test_a_call_opening_also_ends_thinking_where_a_call_may_follow(self):
        harness, runtime = self.harness()
        for choice in ("auto", "none"):
            self.post(
                harness,
                "/v1/chat/completions",
                fixtures.chat_body(
                    max_tokens=32,
                    tools=TOOLS,
                    tool_choice=choice,
                    thinking_token_budget=3,
                ),
            )
        self.assertEqual(
            self.frames(runtime), [budget(3, TOOL_CALL_OPEN_TOKEN_ID), budget(3)]
        )

    def test_a_budget_must_be_a_non_negative_integer_or_minus_one(self):
        harness, runtime = self.harness()
        for value in (-2, 1.5, "8", True, [8]):
            for path, body in (
                ("/v1/chat/completions", fixtures.chat_body()),
                ("/v1/responses", fixtures.responses_body()),
                ("/v1/completions", {"model": "test-model", "prompt": "Hi"}),
            ):
                with self.subTest(path=path, value=value):
                    self.refused(
                        harness,
                        path,
                        {**body, "thinking_token_budget": value},
                        "thinking_token_budget must be a non-negative integer "
                        "or -1 for no cap",
                    )
        self.assertEqual(runtime.requests, [])

    def test_text_completions_refuse_a_budget(self):
        harness, runtime = self.harness(default_thinking_token_budget=5)
        completion = {"model": "test-model", "prompt": "Hi", "max_tokens": 4}
        self.refused(
            harness,
            "/v1/completions",
            {**completion, "thinking_token_budget": 8},
            "thinking_token_budget is not supported for text completions",
        )
        for fields in (
            {},
            {"thinking_token_budget": None},
            {"thinking_token_budget": -1},
        ):
            self.post(harness, "/v1/completions", {**completion, **fields})
        self.assertEqual(self.frames(runtime), [None] * 3)

    def test_messages_enabled_thinking_caps_thinking_at_its_budget_tokens(self):
        harness, runtime = self.harness(
            default_thinking_token_budget=5, max_context=4096
        )
        for thinking in (
            {"type": "enabled", "budget_tokens": 1024},
            {"type": "adaptive"},
            {"type": "disabled"},
        ):
            self.post(
                harness,
                "/v1/messages",
                fixtures.anthropic_body(max_tokens=2048, thinking=thinking),
            )
        self.assertEqual(self.frames(runtime), [budget(1024), budget(5), None])

    def test_messages_budget_tokens_follow_the_messages_rules(self):
        harness, runtime = self.harness(max_context=4096)
        for budget_tokens, message in (
            (None, "thinking.budget_tokens must be an integer of at least 1024"),
            (1023, "thinking.budget_tokens must be an integer of at least 1024"),
            (True, "thinking.budget_tokens must be an integer of at least 1024"),
            (2048, "thinking.budget_tokens must be less than max_tokens"),
        ):
            with self.subTest(budget_tokens=budget_tokens):
                thinking = {"type": "enabled"}
                if budget_tokens is not None:
                    thinking["budget_tokens"] = budget_tokens
                self.refused(
                    harness,
                    "/v1/messages",
                    fixtures.anthropic_body(max_tokens=2048, thinking=thinking),
                    message,
                )
        self.assertEqual(runtime.requests, [])


if __name__ == "__main__":
    unittest.main()

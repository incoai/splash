import json
import unittest

from dev.tests.test_server import FakeConstraintFactory, FakeRuntime, Harness, Plan
from server import protocol as wire

CHAT = ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]})
COMPLETION = ("/v1/completions", {"prompt": "Hi"})
IGNORE_EOS = wire.RequestFlag.IGNORE_END_OF_SEQUENCE


class IgnoreEosTests(unittest.TestCase):
    def harness(self, *plans, **options):
        runtime = FakeRuntime(*plans)
        harness = Harness(runtime, **options)
        self.addCleanup(harness.close)
        return harness, runtime

    def test_chat_and_text_completions_carry_the_request_flag(self):
        harness, runtime = self.harness(
            *(Plan([[4]], reason="length") for _ in range(6))
        )
        for path, body in (CHAT, COMPLETION):
            for fields in ({}, {"ignore_eos": False}, {"ignore_eos": True}):
                with self.subTest(path=path, fields=fields):
                    status, _, payload = harness.request(
                        "POST", path, {**body, "max_tokens": 1, **fields}
                    )
                    self.assertEqual(status, 200, payload)
                    choice = json.loads(payload)["choices"][0]
                    self.assertEqual(choice["finish_reason"], "length")
        self.assertEqual(
            [request.frame.flags for request in runtime.requests],
            [0, 0, IGNORE_EOS] * 2,
        )

    def test_value_must_be_a_boolean(self):
        harness, runtime = self.harness()
        for path, body in (CHAT, COMPLETION):
            for value in (1, "true", None, {}):
                with self.subTest(path=path, value=value):
                    status, _, payload = harness.request(
                        "POST", path, {**body, "ignore_eos": value}
                    )
                    self.assertEqual(status, 400)
                    self.assertEqual(
                        json.loads(payload)["error"]["message"],
                        "ignore_eos must be a boolean",
                    )
        self.assertEqual(runtime.requests, [])

    def test_grammar_constrained_generation_rejects_it(self):
        # The grammar decides where tool calls and structured output end.
        factory = FakeConstraintFactory()
        harness, runtime = self.harness(Plan([[4]]), constraint_factory=factory)
        tools = [{"type": "function", "function": {"name": "weather"}}]
        schema = {"type": "json_schema", "json_schema": {"schema": {}}}
        for fields in (
            {"tools": tools},
            {"tools": tools, "tool_choice": "none"},
            {"tools": tools, "tool_choice": "required"},
            {"response_format": schema},
            {"response_format": {"type": "json_object"}},
        ):
            with self.subTest(fields=fields):
                status, _, payload = harness.request(
                    "POST", CHAT[0], {**CHAT[1], **fields, "ignore_eos": True}
                )
                self.assertEqual(status, 400)
                self.assertEqual(
                    json.loads(payload)["error"]["message"],
                    "ignore_eos cannot be combined with tools or structured output",
                )
        self.assertEqual((runtime.requests, factory.grammars), ([], []))
        # Unconstrained text ignores end-of-sequence as asked.
        status, _, payload = harness.request(
            "POST",
            CHAT[0],
            {**CHAT[1], "response_format": {"type": "text"}, "ignore_eos": True},
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(runtime.requests[0].frame.flags, IGNORE_EOS)
        self.assertEqual(runtime.requests[0].frame.constraint, wire.ConstraintMode.NONE)


if __name__ == "__main__":
    unittest.main()

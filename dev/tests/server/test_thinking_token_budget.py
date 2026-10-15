import io
import json
import unittest
from unittest import mock

from dev.tests import server_fixtures as fixtures
from server import backend as backend_api
from server import protocol as wire
from server import server
from server.backend import ThinkingEndError, thinking_end_tokens
from server.tool_schema import THINK_END_TOKEN_ID, TOOL_CALL_OPEN_TOKEN_ID

TOOLS = [{"type": "function", "function": {"name": "weather"}}]


def budget(tokens, also_ends=THINK_END_TOKEN_ID, end=(THINK_END_TOKEN_ID,)):
    return wire.ThinkingBudget(tokens, end, also_ends)


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

    def test_a_budget_ends_with_the_servers_thinking_end(self):
        end = (14, 15, THINK_END_TOKEN_ID)
        harness, runtime = self.harness(thinking_end=end)
        for tools in ({}, {"tools": TOOLS}):
            self.post(
                harness,
                "/v1/chat/completions",
                fixtures.chat_body(max_tokens=32, thinking_token_budget=3, **tools),
            )
        self.assertEqual(
            self.frames(runtime),
            [budget(3, end=end), budget(3, TOOL_CALL_OPEN_TOKEN_ID, end)],
        )

    def test_reasoning_tokens_count_the_end_before_its_close(self):
        # What the engine returns under a budget of 3 that ends with "first
        # second\n" and then </think>, the fake vocabulary's token 26: three
        # thinking tokens, the end, then the answer.
        end = (14, 15, 26)
        output = [[1, 1, 1], list(end), [27]]
        runtime = fixtures.FakeRuntime(*(fixtures.Plan(output) for _ in range(4)))
        harness = fixtures.Harness(runtime, thinking_end=end)
        self.addCleanup(harness.close)
        chat = fixtures.chat_body(max_tokens=32, thinking_token_budget=3)
        responses = fixtures.responses_body(
            max_output_tokens=32, thinking_token_budget=3
        )
        usages = []
        with mock.patch.object(backend_api, "THINK_END_TOKEN_ID", 26):
            for path, body, stream in (
                ("/v1/chat/completions", chat, False),
                ("/v1/chat/completions", chat, True),
                ("/v1/responses", responses, False),
                ("/v1/responses", responses, True),
            ):
                if stream:
                    body = {**body, "stream": True}
                    if path == "/v1/chat/completions":
                        body["stream_options"] = {"include_usage": True}
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 200, payload)
                if not stream:
                    usages.append(json.loads(payload)["usage"])
                elif path == "/v1/chat/completions":
                    usages.append(fixtures.stream_events(payload)[-1]["usage"])
                else:
                    events = fixtures.response_events(payload)
                    usages.append(events[-1]["response"]["usage"])
        self.assertEqual(self.frames(runtime), [budget(3, 26, end)] * 4)
        self.assertEqual(
            [usage["completion_tokens_details"] for usage in usages[:2]]
            + [usage["output_tokens_details"] for usage in usages[2:]],
            [{"reasoning_tokens": 5}] * 4,
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

    def test_messages_budget_tokens_at_or_above_max_tokens_set_no_cap(self):
        # Messages allows such a budget with interleaved thinking, and one the
        # output limit reaches first closes nothing.
        harness, runtime = self.harness(
            default_thinking_token_budget=5, max_context=4096
        )
        for budget_tokens in (2047, 2048, 4096):
            body = fixtures.anthropic_body(
                max_tokens=2048,
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
            )
            for path in ("/v1/messages", "/v1/messages/count_tokens"):
                with self.subTest(budget_tokens=budget_tokens, path=path):
                    self.post(harness, path, body)
        self.assertEqual(self.frames(runtime), [budget(2047), None, None])


class ThinkingEndTextTests(unittest.TestCase):
    # How the tokenizer encodes each text; its added tokens are the close of
    # thinking, a call's opening and the end of a turn.
    ENCODINGS = {
        "Time is up.\n</think>": [14, 15, THINK_END_TOKEN_ID],
        "</think>": [THINK_END_TOKEN_ID],
        "longest": [14] * 63 + [THINK_END_TOKEN_ID],
        "": [],
        "Time is up.": [14, 15],
        "</think>\n\n": [THINK_END_TOKEN_ID, 271],
        "<tool_call></think>": [TOOL_CALL_OPEN_TOKEN_ID, THINK_END_TOKEN_ID],
        "<|im_end|></think>": [248046, THINK_END_TOKEN_ID],
        "too long": [14] * 64 + [THINK_END_TOKEN_ID],
    }

    def tokenizer(self):
        return mock.Mock(
            encode=mock.Mock(side_effect=lambda text, **_: self.ENCODINGS[text]),
            added_tokens_decoder=dict.fromkeys(
                (THINK_END_TOKEN_ID, TOOL_CALL_OPEN_TOKEN_ID, 248046)
            ),
        )

    def test_without_a_text_thinking_ends_at_its_close(self):
        tokenizer = self.tokenizer()
        self.assertEqual(thinking_end_tokens(tokenizer, None), (THINK_END_TOKEN_ID,))
        tokenizer.encode.assert_not_called()

    def test_a_text_ends_with_the_close_and_holds_no_other_special_token(self):
        for text in ("Time is up.\n</think>", "</think>", "longest"):
            with self.subTest(text=text):
                tokenizer = self.tokenizer()
                self.assertEqual(
                    thinking_end_tokens(tokenizer, text), tuple(self.ENCODINGS[text])
                )
                tokenizer.encode.assert_called_once_with(text, add_special_tokens=False)
        for text in (
            "",
            "Time is up.",
            "</think>\n\n",
            "<tool_call></think>",
            "<|im_end|></think>",
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(
                    ThinkingEndError,
                    "^--thinking-end-text must end with </think> and hold no other "
                    "special token$",
                ):
                    thinking_end_tokens(self.tokenizer(), text)
        with self.assertRaisesRegex(
            ThinkingEndError, "^--thinking-end-text takes 65 tokens; the most is 64$"
        ):
            thinking_end_tokens(self.tokenizer(), "too long")

    def test_startup_refuses_a_text_before_starting_the_native_worker(self):
        args = server.parse_args(
            [
                "model",
                "--tokenizer",
                "tokenizer",
                "--model",
                "owner/model",
                "--thinking-end-text=Time is up.",
            ]
        )
        http = mock.Mock()
        with (
            mock.patch.object(server, "parse_args", return_value=args),
            mock.patch.object(server.signal, "signal"),
            mock.patch.object(server, "load_thinking_key", return_value=None),
            mock.patch.object(
                server.AutoTokenizer, "from_pretrained", return_value=self.tokenizer()
            ),
            mock.patch.object(server, "validate_tokenizer"),
            mock.patch.object(server, "FrontendServer", return_value=http),
            mock.patch.object(server.engine_runtime, "MultiplexedRuntime") as native,
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            mock.patch("sys.stdout", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            server.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn(
            "--thinking-end-text must end with </think> and hold no other special token",
            stderr.getvalue(),
        )
        native.assert_not_called()
        http.server_activate.assert_not_called()


if __name__ == "__main__":
    unittest.main()

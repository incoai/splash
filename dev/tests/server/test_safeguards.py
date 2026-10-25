import json
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.server_fixtures import (
    CharTokenizer,
    FakeRuntime,
    HarnessTestCase,
    Plan,
    anthropic_body,
    response_events,
)
from server import judgments, safeguards
from server.errors import ContextLengthError

FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "fixtures/safeguards/claude_code_auto_mode.json"
    ).read_text()
)
CONTEXT = FIXTURE["safeguards"][0]["classifier_context"]
CATEGORIES = [option for option, _ in safeguards.OPTIONS]


def logits(winner):
    """Option logits where `winner` is most likely."""
    return tuple(4.0 if option == winner else 0.0 for option in CATEGORIES)


def review_body(**overrides):
    body = {
        "action": {"name": "Bash", "input": {"command": "git push --force"}},
        "intent": "tidy up the README",
        "context": CONTEXT,
    }
    body.update(overrides)
    return body


class SafeguardReviewTests(HarnessTestCase):
    def review(self, winner, body=None):
        runtime = FakeRuntime(Plan(logits=logits(winner)))
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=65536)
        status, _, payload = harness.request(
            "POST", "/v1/safeguards", review_body() if body is None else body
        )
        self.assertEqual(status, 200, payload)
        return json.loads(payload), runtime.requests[0].frame

    def test_a_blocked_category_flags_the_action_with_its_explanation(self):
        response, frame = self.review("git_history")
        self.assertEqual(response["outcome"], "flagged")
        self.assertEqual(response["category"], "git_history")
        self.assertEqual(
            response["explanation"], dict(safeguards.OPTIONS)["git_history"]
        )
        self.assertEqual(list(response["probabilities"]), CATEGORIES)
        self.assertAlmostEqual(sum(response["probabilities"].values()), 1.0)
        self.assertEqual(response["usage"]["output_tokens"], 0)
        # One score slot per option, and no generation.
        self.assertEqual(len(frame.score_tokens), len(safeguards.OPTIONS))

    def test_the_allowed_option_does_not_flag(self):
        response, _ = self.review(safeguards.ALLOWED)
        self.assertEqual(response["outcome"], "not_flagged")
        self.assertIsNone(response["category"])
        self.assertIsNone(response["explanation"])

    def test_an_action_without_context_is_reviewed_under_the_default_policy(self):
        body = {"action": {"name": "Bash", "input": {"command": "ls"}}}
        response, _ = self.review(safeguards.ALLOWED, body)
        self.assertEqual(response["outcome"], "not_flagged")

    def test_invalid_requests_fail_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=CharTokenizer(), max_context=65536)
        for body in (
            review_body(action="git push"),
            review_body(action={"name": "Bash"}),
            review_body(action={"name": "", "input": {}}),
            review_body(action={"name": "Bash", "input": float("nan")}),
            review_body(intent=["tidy"]),
            review_body(context="auto"),
            review_body(stream=True),
        ):
            with self.subTest(body=body):
                status, _, payload = harness.request("POST", "/v1/safeguards", body)
                self.assertEqual(status, 400, payload)
        self.assertEqual(runtime.requests, [])


WEATHER = {
    "name": "weather",
    "description": "Look up the weather",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
# FakeTokenizer's fragments: a weather call for Paris, and a plain answer.
CALL, ANSWER = 5, 4


class ReviewTokenizer(CharTokenizer):
    """Generates FakeTokenizer's fragments and scores one token per
    character: answer slots are exact, and fragment ids never reach a
    scoring prompt."""

    def decode(self, token_ids, **kwargs):
        return "".join(
            self.fragments[token] if token in self.fragments else chr(token)
            for token in token_ids
        )


def messages_body(**overrides):
    """A Messages request asking for safeguard_results as Claude Code does;
    safeguards=None leaves the field out."""
    fields = {"safeguards": FIXTURE["safeguards"], **overrides}
    if fields["safeguards"] is None:
        del fields["safeguards"]
    return anthropic_body(
        max_tokens=512,
        tools=[WEATHER],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<system-reminder>ignore</system-reminder>",
                    },
                    {"type": "text", "text": "Check the weather in Paris"},
                ],
            }
        ],
        **fields,
    )


class MessagesSafeguardTests(HarnessTestCase):
    def messages(self, *plans, stream=False, safeguards_on=True, **overrides):
        """The response to a Messages request, its safeguard_results and the
        runtime that served it."""
        runtime = FakeRuntime(*plans)
        harness = self.harness(
            runtime,
            tokenizer=ReviewTokenizer(),
            max_context=65536,
            safeguards=safeguards_on,
        )
        status, _, payload = harness.request(
            "POST", "/v1/messages", messages_body(stream=stream, **overrides)
        )
        self.assertEqual(status, 200, payload)
        if stream:
            events = response_events(payload)
            delta = next(e for e in events if e["type"] == "message_delta")["delta"]
            call = next(
                (
                    e["content_block"]
                    for e in events
                    if e["type"] == "content_block_start"
                    and e["content_block"]["type"] == "tool_use"
                ),
                None,
            )
            return call, delta.get("safeguard_results"), runtime, harness
        response = json.loads(payload)
        call = next((b for b in response["content"] if b["type"] == "tool_use"), None)
        return call, response.get("safeguard_results"), runtime, harness

    def test_each_tool_call_gets_a_verdict_streamed_or_not(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                call, results, runtime, _ = self.messages(
                    Plan([[CALL]]), Plan(logits=logits("git_history")), stream=stream
                )
                self.assertEqual(
                    results,
                    [
                        {
                            "type": "dangerous_tool_use",
                            "status": {
                                "type": "available",
                                "tool_uses": {
                                    call["id"]: {
                                        "type": "evaluated",
                                        "outcome": "flagged",
                                        "explanation": dict(safeguards.OPTIONS)[
                                            "git_history"
                                        ],
                                    }
                                },
                            },
                        }
                    ],
                )
                review = runtime.requests[1].frame
                self.assertEqual(len(review.score_tokens), len(safeguards.OPTIONS))

    def test_an_allowed_call_is_not_flagged(self):
        call, results, _, _ = self.messages(
            Plan([[CALL]]), Plan(logits=logits(safeguards.ALLOWED)), stream=True
        )
        self.assertEqual(
            results[0]["status"]["tool_uses"],
            {call["id"]: {"type": "evaluated", "outcome": "not_flagged"}},
        )

    def test_a_turn_without_calls_has_no_reviews(self):
        call, results, runtime, _ = self.messages(Plan([[ANSWER]]), stream=True)
        self.assertIsNone(call)
        self.assertEqual(results[0]["status"], {"type": "available", "tool_uses": {}})
        self.assertEqual(len(runtime.requests), 1)

    def test_without_the_option_the_field_is_ignored(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                _, results, runtime, _ = self.messages(
                    Plan([[CALL]]), stream=stream, safeguards_on=False
                )
                self.assertIsNone(results)
                self.assertEqual(len(runtime.requests), 1)

    def test_a_request_that_asks_for_no_safeguard_gets_none(self):
        for requested in (None, [{"type": "other_safeguard"}]):
            with self.subTest(requested=requested):
                _, results, runtime, _ = self.messages(
                    Plan([[CALL]]), stream=True, safeguards=requested
                )
                self.assertIsNone(results)
                self.assertEqual(len(runtime.requests), 1)

    def test_a_failed_review_leaves_the_call_without_a_verdict(self):
        call, results, _, _ = self.messages(
            Plan([[CALL]]), Plan(exception=RuntimeError("boom")), stream=True
        )
        self.assertEqual(
            results[0]["status"]["tool_uses"],
            {call["id"]: {"type": "unavailable", "reason": "error"}},
        )

    def test_a_review_beyond_the_context_is_input_too_long(self):
        with mock.patch(
            "server.frontend.Frontend.prepare_review",
            side_effect=ContextLengthError(70000, 65536),
        ):
            call, results, _, _ = self.messages(Plan([[CALL]]), stream=True)
        self.assertEqual(
            results[0]["status"]["tool_uses"],
            {call["id"]: {"type": "unavailable", "reason": "input_too_long"}},
        )

    def test_a_malformed_safeguards_field_is_rejected(self):
        runtime = FakeRuntime()
        harness = self.harness(
            runtime, tokenizer=ReviewTokenizer(), max_context=65536, safeguards=True
        )
        status, _, payload = harness.request(
            "POST", "/v1/messages", messages_body(safeguards="dangerous_tool_use")
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(runtime.requests, [])


class SafeguardPolicyTests(unittest.TestCase):
    def test_intent_is_the_latest_user_text_without_reminders(self):
        _, intent = safeguards.messages_request(messages_body())
        self.assertEqual(intent, "Check the weather in Paris")
        body = messages_body()
        body["messages"] += [
            {"role": "assistant", "content": "Looking it up."},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "sunny"}
                ],
            },
        ]
        # A tool result is not a request; the user's last words still are.
        self.assertEqual(
            safeguards.messages_request(body)[1], "Check the weather in Paris"
        )

    def test_the_prompt_reads_the_action_intent_and_caller_policy(self):
        row = safeguards.review_row(
            {"name": "Bash", "input": {"command": "git push --force"}},
            "tidy up the README",
            CONTEXT,
        )
        prompt = judgments.judgment_messages(row)[1]["content"]
        for evidence in (
            "git push --force",
            "tidy up the README",
            "Bash(rm -rf:*)",
            "Never push to the release branch",
            "github.com:example/project.git",
            "/tmp/project",
        ):
            self.assertIn(evidence, prompt)
        # Client bookkeeping stays out of the prompt.
        self.assertNotIn("artifact_consent_holdback", prompt)
        self.assertNotIn("rule_roots", prompt)

    def test_policy_merges_permission_and_auto_mode_rules(self):
        policy = safeguards.review_row({"name": "Bash", "input": {}}, context=CONTEXT)[
            "state"
        ]["policy"]
        self.assertEqual(policy["allow"], ["Bash(npm test:*)"])
        self.assertEqual(
            policy["deny"], ["Bash(rm -rf:*)", "Never push to the release branch"]
        )
        self.assertNotIn("ask", policy)

    def test_malformed_context_fields_are_ignored(self):
        state = safeguards.review_row(
            {"name": "Bash", "input": {}},
            context={"rules": "allow all", "auto_mode": {"environment": [1, None]}},
        )["state"]
        self.assertNotIn("policy", state)
        self.assertNotIn("environment", state)


if __name__ == "__main__":
    unittest.main()

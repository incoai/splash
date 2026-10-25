import json
import unittest
from pathlib import Path

from dev.tests.server_fixtures import CharTokenizer, FakeRuntime, HarnessTestCase, Plan
from server import judgments, safeguards

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


class SafeguardPolicyTests(unittest.TestCase):
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

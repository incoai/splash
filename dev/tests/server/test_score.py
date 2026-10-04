"""POST /v1/score and POST /v1/decisions: labels read after the Chat prompt
and the assistant prefix, in one prefill (server/scoring.py)."""

import json
import math

from dev.tests.server_fixtures import CharTokenizer, FakeRuntime, HarnessTestCase, Plan


class ScoreTests(HarnessTestCase):
    class ScoreTokenizer(CharTokenizer):
        """Character tokens through every entry point Chat and /v1/score use."""

        def __call__(self, text, **kwargs):
            return {"input_ids": self.encode(text)}

        def __len__(self):
            return 0x110000

    def score_body(self, **overrides):
        body = {
            "messages": [
                {"role": "system", "content": "Judge products."},
                {"role": "user", "content": "Same item?"},
            ],
            "assistant_prefix": '{"same":',
            "labels": ["T", "F"],
            "reasoning_effort": "none",
        }
        body.update(overrides)
        return body

    def test_score_reads_labels_after_the_chat_prompt_and_prefix(self):
        top = ((ord("T"), 2.0), (ord("x"), 1.0), (ord("F"), 0.5))
        runtime = FakeRuntime(
            Plan(logits=(2.0, 0.5), normalizer=3.0, top=top, matched_tokens=5),
            Plan([[ord("x")]]),
        )
        harness = self.harness(
            runtime, tokenizer=self.ScoreTokenizer(), max_context=8192
        )
        status, content_type, payload = harness.request(
            "POST", "/v1/score", self.score_body(return_top_k=2)
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(content_type, "application/json")
        response = json.loads(payload)
        self.assertEqual(response["object"], "score")
        self.assertEqual(response["argmax"], "T")
        labels = response["labels"]
        self.assertEqual([entry["label"] for entry in labels], ["T", "F"])
        self.assertEqual([entry["token_id"] for entry in labels], [ord("T"), ord("F")])
        # Full-vocabulary log probabilities subtract the native normalizer.
        self.assertAlmostEqual(labels[0]["logprob"], 2.0 - 3.0)
        self.assertAlmostEqual(labels[1]["logprob"], 0.5 - 3.0)
        self.assertAlmostEqual(labels[0]["prob"] + labels[1]["prob"], 1.0)
        self.assertAlmostEqual(
            labels[0]["prob"], math.exp(2.0) / (math.exp(2.0) + math.exp(0.5))
        )
        self.assertAlmostEqual(response["label_mass"], math.exp(-1.0) + math.exp(-2.5))
        self.assertEqual([entry["token"] for entry in response["top_k"]], ["T", "x"])
        self.assertAlmostEqual(response["top_k"][1]["logprob"], -2.0)
        self.assertEqual(response["usage"]["cached_tokens"], 5)
        self.assertEqual(
            set(response["timing"]), {"queue_ms", "prefill_ms", "total_ms"}
        )

        request = runtime.requests[0].frame
        # The engine scores the Chat prompt followed by the prefix, prefill only.
        text = harness.tokenizer.decode(request.prompt_tokens)
        self.assertTrue(text.endswith('</think>\n\n{"same":'), text)
        self.assertEqual(request.score_tokens, (ord("T"), ord("F")))
        self.assertEqual(request.logical_max_output_tokens, 0)
        # Reusable state stays before the assistant turn: Chat's generation
        # prompt plus the prefix, or unknown (0) where Chat's is unknown.
        chat = self.score_body()
        for key in ("assistant_prefix", "labels"):
            chat.pop(key)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", {**chat, "max_tokens": 1}
        )
        self.assertEqual(status, 200, payload)
        generation = runtime.requests[1].frame.generation_prompt_tokens
        self.assertEqual(
            request.generation_prompt_tokens,
            generation + len('{"same":') if generation else 0,
        )
        self.assertEqual(
            list(runtime.requests[1].frame.prompt_tokens),
            list(request.prompt_tokens[: -len('{"same":')]),
        )
        status, _, payload = harness.request("GET", "/status")
        score = json.loads(payload)["score"]
        self.assertEqual(score["score_requests"], 1)
        self.assertEqual(score["score_cached_tokens"], 5)
        self.assertEqual(score["score_prompt_tokens"], len(request.prompt_tokens))

    def test_score_renders_tools_as_chat_does(self):
        # The template receives a score request's tools as it receives
        # Chat's, so a score request with an agent's system prompt and tools
        # renders, and warms, the start of the agent's requests.
        runtime = FakeRuntime(
            Plan(logits=(1.0, 0.0), normalizer=2.0), Plan([[ord("x")]])
        )
        harness = self.harness(
            runtime, tokenizer=self.ScoreTokenizer(), max_context=8192
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        status, _, payload = harness.request(
            "POST", "/v1/score", self.score_body(tools=tools)
        )
        self.assertEqual(status, 200, payload)
        score_tools = harness.tokenizer.templates[-1][1].get("tools")
        chat = self.score_body(tools=tools)
        for key in ("assistant_prefix", "labels"):
            chat.pop(key)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", {**chat, "max_tokens": 1}
        )
        self.assertEqual(status, 200, payload)
        chat_tools = harness.tokenizer.templates[-1][1].get("tools")
        self.assertTrue(score_tools)
        self.assertEqual(score_tools, chat_tools)
        self.assertEqual(score_tools[0]["function"]["name"], "weather")
        self.assertEqual(
            list(runtime.requests[1].frame.prompt_tokens),
            list(runtime.requests[0].frame.prompt_tokens[: -len('{"same":')]),
        )

    def test_score_temperature_scales_only_label_probabilities(self):
        runtime = FakeRuntime(Plan(logits=(2.0, 0.5), normalizer=3.0))
        harness = self.harness(
            runtime, tokenizer=self.ScoreTokenizer(), max_context=8192
        )
        status, _, payload = harness.request(
            "POST", "/v1/score", self.score_body(temperature=2.0)
        )
        self.assertEqual(status, 200, payload)
        labels = json.loads(payload)["labels"]
        self.assertAlmostEqual(
            labels[0]["prob"], math.exp(1.0) / (math.exp(1.0) + math.exp(0.25))
        )
        self.assertAlmostEqual(labels[0]["logprob"], -1.0)

    def test_score_label_token_ids_override_labels(self):
        runtime = FakeRuntime(Plan(logits=(0.0, 1.0, -1.0), normalizer=2.0))
        harness = self.harness(
            runtime, tokenizer=self.ScoreTokenizer(), max_context=8192
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/score",
            self.score_body(labels=None, label_token_ids=[65, 66, 67]),
        )
        self.assertEqual(status, 200, payload)
        response = json.loads(payload)
        self.assertEqual(response["argmax"], "B")
        self.assertEqual(runtime.requests[0].frame.score_tokens, (65, 66, 67))

    def test_score_rejects_a_multi_token_label_with_its_tokens(self):
        runtime = FakeRuntime()
        harness = self.harness(
            runtime, tokenizer=self.ScoreTokenizer(), max_context=8192
        )
        status, _, payload = harness.request(
            "POST", "/v1/score", self.score_body(labels=[" true", "F"])
        )
        self.assertEqual(status, 400, payload)
        error = json.loads(payload)["error"]
        self.assertIn("exactly one token", error["message"])
        detail = error["details"]["labels"][0]
        self.assertEqual(detail["label"], " true")
        self.assertEqual([token["text"] for token in detail["tokens"]], list(" true"))
        self.assertEqual(runtime.requests, [])

    def test_score_rejects_invalid_requests_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(
            runtime, tokenizer=self.ScoreTokenizer(), max_context=8192
        )
        for overrides, fragment in (
            ({"labels": ["T"]}, "at least two"),
            ({"labels": ["T", "T"]}, "distinct"),
            ({"labels": None}, "labels or label_token_ids"),
            ({"temperature": 0}, "temperature"),
            ({"return_top_k": 21}, "return_top_k"),
            ({"stream": True}, "unsupported fields"),
            ({"assistant_prefix": 3}, "assistant_prefix"),
            ({"reasoning_effort": "high"}, "think block"),
        ):
            with self.subTest(overrides=overrides):
                status, _, payload = harness.request(
                    "POST", "/v1/score", self.score_body(**overrides)
                )
                self.assertEqual(status, 400, payload)
                self.assertIn(fragment, json.loads(payload)["error"]["message"])
        self.assertEqual(runtime.requests, [])

    def test_score_batch_runs_every_request_and_keeps_order(self):
        runtime = FakeRuntime(
            Plan(logits=(1.0, 0.0), normalizer=2.0),
            Plan(logits=(0.0, 1.0), normalizer=2.0),
            Plan(logits=(3.0, 0.0), normalizer=4.0),
        )
        harness = self.harness(
            runtime, tokenizer=self.ScoreTokenizer(), max_context=8192
        )
        items = [
            {k: v for k, v in self.score_body().items() if k != "model"}
            for _ in range(3)
        ]
        items[1]["messages"] = [{"role": "user", "content": "Other candidate?"}]
        status, _, payload = harness.request("POST", "/v1/score", {"requests": items})
        self.assertEqual(status, 200, payload)
        response = json.loads(payload)
        self.assertEqual(response["object"], "list")
        self.assertEqual(
            [entry["argmax"] for entry in response["data"]], ["T", "F", "T"]
        )
        self.assertEqual(len(runtime.requests), 3)
        status, _, payload = harness.request(
            "POST", "/v1/score", {"requests": [{"labels": ["T"]}]}
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("requests[0].", json.loads(payload)["error"]["message"])

    class WordTokenizer(ScoreTokenizer):
        """Character tokens, except that "yes" and "no" are one token each."""

        WORDS = {"yes": 0x10FF00, "no": 0x10FF01}

        def encode(self, text, **kwargs):
            tokens, index = [], 0
            while index < len(text):
                for word, token in self.WORDS.items():
                    if text.startswith(word, index):
                        tokens.append(token)
                        index += len(word)
                        break
                else:
                    tokens.append(ord(text[index]))
                    index += 1
            return tokens

        def decode(self, token_ids, **kwargs):
            names = {token: word for word, token in self.WORDS.items()}
            return "".join(
                names.get(token, chr(token) if token < 0x10FF00 else "")
                for token in token_ids
            )

    def test_decisions_answer_each_question_type_over_the_shared_input(self):
        runtime = FakeRuntime(
            Plan(logits=(2.0, 0.0), normalizer=2.5),
            Plan(logits=(0.0, 3.0, 1.0), normalizer=3.5),
            Plan(logits=(0.0, 1.0, 2.0), normalizer=2.5),
        )
        harness = self.harness(
            runtime, tokenizer=self.WordTokenizer(), max_context=8192
        )
        body = {
            "system": "You classify product listings.",
            "input": "Listing: cordless drill 18V, 2 batteries",
            "reasoning_effort": "none",
            "questions": {
                "is_tool": {"type": "yes_no", "question": "Is it a power tool?"},
                "category": {
                    "type": "choice",
                    "question": "Which category?",
                    "options": ["accessory", "drill", "battery"],
                },
                "quality": {
                    "type": "score",
                    "question": "How complete is the listing?",
                    "levels": ["poor", "fair", "good"],
                },
            },
        }
        status, _, payload = harness.request("POST", "/v1/decisions", body)
        self.assertEqual(status, 200, payload)
        answers = json.loads(payload)["answers"]
        self.assertEqual(answers["is_tool"]["answer"], "yes")
        self.assertAlmostEqual(
            answers["is_tool"]["p_yes"], math.exp(2) / (math.exp(2) + 1)
        )
        self.assertEqual(answers["category"]["answer"], "B")
        self.assertEqual(answers["category"]["option"], "drill")
        self.assertEqual(answers["quality"]["answer"], 2)
        self.assertEqual(answers["quality"]["level"], "good")
        self.assertGreater(answers["quality"]["expected"], 1.0)
        # Every question follows the same system and input; only its suffix
        # differs (the messages each request's template rendered).
        rendered = [
            messages
            for messages, options in harness.tokenizer.templates
            if options.get("tokenize") is False and options.get("add_generation_prompt")
        ][-3:]
        users = [messages[-1]["content"] for messages in rendered]
        shared = "Listing: cordless drill 18V, 2 batteries\n\n"
        self.assertTrue(all(user.startswith(shared) for user in users), users)
        self.assertTrue(
            all(
                messages[0]["content"] == "You classify product listings."
                for messages in rendered
            )
        )
        self.assertIn(
            "A) accessory\nB) drill\nC) battery\nAnswer with the letter only.", users[1]
        )
        self.assertIn(
            "0 = poor\n1 = fair\n2 = good\nAnswer with the number only.", users[2]
        )
        self.assertEqual(runtime.requests[0].frame.score_tokens, (0x10FF00, 0x10FF01))
        self.assertEqual(
            runtime.requests[1].frame.score_tokens, (ord("A"), ord("B"), ord("C"))
        )

    def test_decisions_reject_invalid_questions(self):
        runtime = FakeRuntime()
        harness = self.harness(
            runtime, tokenizer=self.WordTokenizer(), max_context=8192
        )
        for body, fragment in (
            ({"questions": {"q": {"type": "yes_no", "question": "x"}}}, "input"),
            ({"input": "x", "questions": {}}, "nonempty object"),
            (
                {"input": "x", "questions": {"q": {"type": "maybe", "question": "x"}}},
                "type must be",
            ),
            (
                {
                    "input": "x",
                    "questions": {
                        "q": {"type": "choice", "question": "x", "options": ["a"]}
                    },
                },
                "options",
            ),
            (
                {
                    "input": "x",
                    "questions": {
                        "q": {"type": "score", "question": "x", "levels": ["a"] * 11}
                    },
                },
                "levels",
            ),
        ):
            with self.subTest(body=body):
                status, _, payload = harness.request("POST", "/v1/decisions", body)
                self.assertEqual(status, 400, payload)
                self.assertIn(fragment, json.loads(payload)["error"]["message"])
        self.assertEqual(runtime.requests, [])

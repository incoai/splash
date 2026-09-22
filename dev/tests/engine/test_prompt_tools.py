import concurrent.futures
import json
import threading
import time
import unittest
from unittest import mock

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from dev.tests.test_server import FakeRuntime, Harness


class PromptToolsTests(unittest.TestCase):
    def test_parallel_tokenization_keeps_special_token_options_independent(self):
        text = "hello world " * 2048
        expected = {
            option: self.tokenizer(text, add_special_tokens=option)["input_ids"]
            for option in (False, True)
        }

        def tokenize(index):
            option = bool(index % 2)
            return option, self.harness.app.tokenize(
                {"content": text, "add_special": option}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for option, tokens in pool.map(tokenize, range(32)):
                self.assertEqual(tokens, expected[option])
        self.assertEqual(self.harness.app.preparation_active, 0)
        self.assertEqual(self.harness.app.preparation_waiting, 0)

    def test_preparation_saturation_times_out_then_recovers(self):
        entered = threading.Barrier(3)
        release = threading.Event()
        render = self.tokenizer.apply_chat_template

        def blocking_render(*args, **kwargs):
            entered.wait(timeout=3)
            if not release.wait(3):
                raise TimeoutError("test render not released")
            return render(*args, **kwargs)

        body = {"messages": [{"role": "user", "content": "hello"}]}
        with mock.patch.object(
            self.tokenizer, "apply_chat_template", side_effect=blocking_render
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pending = [
                    pool.submit(self.harness.request, "POST", "/apply-template", body)
                    for _ in range(2)
                ]
                try:
                    entered.wait(timeout=3)
                    self.assertEqual(self.harness.request("GET", "/health")[0], 200)
                    self.post("/tokenize", {"content": "hello", "timeout": 0.05}, 504)
                    self.assertEqual(self.harness.app.preparation_active, 2)
                    self.assertEqual(self.harness.app.preparation_waiting, 0)
                finally:
                    release.set()
                for response in pending:
                    self.assertEqual(response.result()[0], 200)
        self.assertTrue(self.harness.server.token_counts.idle.wait(2))
        self.post("/tokenize", {"content": "hello"})
        self.post("/apply-template", body)

    def setUp(self):
        vocabulary = {
            token: index
            for index, token in enumerate(
                [
                    "[UNK]",
                    "[BOS]",
                    "[EOS]",
                    "<think>",
                    "</think>",
                    "user",
                    "assistant",
                    "hello",
                    "world",
                ]
            )
        }
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A", special_tokens=[("[BOS]", 1)]
        )
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            additional_special_tokens=["<think>", "</think>"],
            chat_template=(
                "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
                "{% if tools %}{{ tools | tojson }}\n{% endif %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n"
                "{% if enable_thinking | default(true) %}<think>\n{% endif %}{% endif %}"
            ),
        )
        self.runtime = FakeRuntime()
        self.harness = Harness(self.runtime, tokenizer=self.tokenizer)
        self.addCleanup(self.harness.close)

    def post(self, path, body, status=200):
        actual, _, data = self.harness.request("POST", path, body)
        self.assertEqual(actual, status, data)
        return json.loads(data)

    def test_raw_tokens_preserve_text_and_special_tokens(self):
        for text in (
            "",
            "hello world",
            "  hello\nworld  ",
            "你好 🌊",
            "<think>hello</think>",
        ):
            for add_special in (False, True):
                with self.subTest(text=text, add_special=add_special):
                    data = self.post(
                        "/tokenize?test=1",
                        {"content": text, "add_special": add_special},
                    )
                    self.assertEqual(
                        data["tokens"],
                        self.tokenizer(text, add_special_tokens=add_special)[
                            "input_ids"
                        ],
                    )
        self.assertFalse(self.runtime.requests)

    def test_template_and_tokens_match_generation_preparation(self):
        for effort in (None, "none", "high"):
            body = {"messages": [{"role": "user", "content": "hello world"}]}
            if effort is not None:
                body["reasoning_effort"] = effort
            with self.subTest(effort=effort):
                rendered = self.post("/apply-template", body)["prompt"]
                tokens = self.post("/tokenize", {"content": rendered})["tokens"]
                job, _, _ = self.harness.app.prepare(body)
                self.assertEqual(tokens, job.prompt_tokens)
                self.assertEqual("<think>" in rendered, effort != "none")
        self.assertFalse(self.runtime.requests)

    def test_template_without_generation_prefix_and_with_tools(self):
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
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
            ],
            "reasoning_effort": "high",
            "add_generation_prompt": False,
        }
        rendered = self.post("/apply-template", body)["prompt"]
        self.assertIn("weather", rendered)
        self.assertNotIn("<think>", rendered)
        self.assertNotIn("<|im_start|>assistant\n", rendered)

    def test_grammar_timing_is_separate_and_records_failed_preparation(self):
        app = self.harness.app
        app.constraint_factory = mock.Mock()
        app.constraint_factory.create.return_value = None
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "note"}}],
        }
        for _ in range(2):
            app.prepare(body)
        self.assertEqual(app.latencies.snapshot()["grammar"]["count"], 2)
        app.prepare({"messages": body["messages"]})
        self.assertEqual(app.latencies.snapshot()["grammar"]["count"], 2)
        app.constraint_factory.create.side_effect = ValueError("compile failed")
        with self.assertRaisesRegex(ValueError, "compile failed"):
            app.prepare(body)
        sample = app.latencies.snapshot()["grammar"]
        self.assertEqual(sample["count"], 3)
        self.assertGreaterEqual(sample["sum"], 0)

    def test_input_validation_and_template_errors(self):
        for body in (
            {},
            {"content": None},
            {"content": []},
            {"content": "hello", "add_special": 1},
            {"content": "hello", "parse_special": False},
            {"content": "hello", "with_pieces": True},
        ):
            self.post("/tokenize", body, 400)
        for body in (
            {},
            {"messages": "bad"},
            {"messages": [], "add_generation_prompt": "false"},
            {"model": "other", "messages": [{"role": "user", "content": "hello"}]},
        ):
            self.post("/apply-template", body, 404 if body.get("model") else 400)
        with mock.patch.object(
            self.tokenizer,
            "apply_chat_template",
            side_effect=ValueError("private text"),
        ):
            data = self.post(
                "/apply-template",
                {"messages": [{"role": "user", "content": "hello"}]},
                400,
            )
            self.assertNotIn("private text", json.dumps(data))

    def test_utilities_do_not_require_engine_admission_or_fit_context(self):
        self.harness.app.max_context = 1
        with (
            mock.patch.object(self.harness.backend, "can_submit", return_value=False),
            mock.patch.object(
                self.harness.server.requests,
                "acquire",
                side_effect=AssertionError("generation admission"),
            ),
        ):
            self.post("/tokenize", {"content": "hello " * 500})
            self.post(
                "/apply-template",
                {"messages": [{"role": "user", "content": "hello " * 500}]},
            )
        self.assertFalse(self.runtime.requests)

    def test_utilities_share_bounded_ingress_and_preparation_deadlines(self):
        for path, body in (
            ("/tokenize", {"content": "hello"}),
            ("/apply-template", {"messages": [{"role": "user", "content": "hello"}]}),
        ):
            with mock.patch.object(
                self.harness.server.token_counts, "acquire", return_value=False
            ):
                self.post(path, body, 503)
            with mock.patch.object(
                self.harness.app, "request_deadline", return_value=time.monotonic() - 1
            ):
                self.post(path, body, 504)
            self.post(path, body)

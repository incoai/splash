import http.client
import json
import time
import unittest
from unittest import mock

from openai import OpenAI

from dev.tests.test_server import FakeRuntime, FakeTokenizer, Harness, Plan
from server import protocol as wire
from server import server as api

CHAT = ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}]})
COMPLETION = ("/v1/completions", {"prompt": "once upon a time"})


class PromptTokenizer(FakeTokenizer):
    """Encodes one id per word, after its BOS when special tokens are added,
    and records every encoding it is asked for."""

    WORDS = {"once": 14, "upon": 15, "a": 16, "time": 17}
    VOCABULARY = 28

    def __init__(self):
        super().__init__()
        self.bos = 9
        self.encodings = []

    def __call__(self, text, add_special_tokens=True, **kwargs):
        self.encodings.append((text, add_special_tokens))
        ids = [self.WORDS.get(word, 1) for word in text.split()]
        if add_special_tokens and self.bos is not None:
            ids.insert(0, self.bos)
        return {"input_ids": ids}

    def __len__(self):
        return self.VOCABULARY


def data_events(payload):
    return [
        line[6:].decode() for line in payload.splitlines() if line.startswith(b"data: ")
    ]


def next_data(response, seconds=2.0):
    """The next data event of a stream, skipping comments; fails when none
    comes within `seconds`."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and (line := response.readline()):
        if line.startswith(b"data: "):
            return json.loads(line[6:])
    raise AssertionError("no data event")


class TextCompletionTests(unittest.TestCase):
    def harness(self, *plans, **options):
        runtime = FakeRuntime(*plans)
        harness = Harness(runtime, tokenizer=PromptTokenizer(), **options)
        self.addCleanup(harness.close)
        # The chat templates were probed at startup; keep only request work.
        harness.tokenizer.encodings.clear()
        return harness, runtime

    def complete(self, harness, **fields):
        body = {"model": "test-model", "prompt": "once upon a time", **fields}
        status, _, payload = harness.request("POST", "/v1/completions", body)
        return status, json.loads(payload)

    def test_prompt_generates_as_given_without_the_chat_template(self):
        harness, runtime = self.harness(Plan([[1], [2], [3]]))
        status, response = self.complete(harness, max_tokens=8, temperature=0)
        self.assertEqual(status, 200, response)
        self.assertTrue(response["id"].startswith("cmpl-"))
        self.assertEqual(response["object"], "text_completion")
        self.assertEqual(response["model"], "test-model")
        # The text keeps its think delimiter: nothing splits reasoning out.
        self.assertEqual(
            response["choices"],
            [
                {
                    "index": 0,
                    "text": "because </think>answer\n",
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        )
        self.assertEqual(response["usage"]["prompt_tokens"], 5)
        self.assertEqual(response["usage"]["completion_tokens"], 3)
        self.assertEqual(
            response["usage"]["completion_tokens_details"], {"reasoning_tokens": 0}
        )
        self.assertIn("timings", response)
        self.assertEqual(harness.tokenizer.templates, [])
        self.assertEqual(harness.tokenizer.encodings, [("once upon a time", True)])
        request = runtime.requests[0]
        self.assertEqual(request.prompt_tokens, (9, 14, 15, 16, 17))
        self.assertEqual(request.generation_prompt_tokens, 0)
        self.assertEqual(request.logical_max_output_tokens, 8)
        self.assertEqual(request.cohort, wire.Cohort.GREEDY)
        self.assertEqual(request.constraint, wire.ConstraintMode.NONE)

    def test_token_ids_are_sent_as_given_within_the_vocabulary(self):
        harness, runtime = self.harness(Plan([[4]]))
        status, response = self.complete(harness, prompt=[3, 1, 27])
        self.assertEqual(status, 200, response)
        self.assertEqual(response["choices"][0]["text"], "plain answer\n")
        self.assertEqual(runtime.requests[0].prompt_tokens, (3, 1, 27))
        self.assertEqual(harness.tokenizer.encodings, [])
        for prompt in ([28], [-1], [3, 2**32]):
            with self.subTest(prompt=prompt):
                status, response = self.complete(harness, prompt=prompt)
                self.assertEqual(status, 400)
                self.assertEqual(
                    response["error"]["message"],
                    "prompt token ids must be in the vocabulary",
                )
        self.assertEqual(len(runtime.requests), 1)

    def test_stream_sends_text_completion_chunks_usage_and_done(self):
        harness, _ = self.harness(Plan([[14], [15]], reason="length"))
        body = {
            "model": "test-model",
            "prompt": "once upon a time",
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 2,
        }
        status, content_type, payload = harness.request("POST", "/v1/completions", body)
        self.assertEqual((status, content_type), (200, "text/event-stream"))
        events = data_events(payload)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(event) for event in events[:-1]]
        self.assertEqual(len({chunk["id"] for chunk in chunks}), 1)
        self.assertTrue(chunks[0]["id"].startswith("cmpl-"))
        self.assertTrue(all(chunk["object"] == "text_completion" for chunk in chunks))
        texts = [chunk["choices"][0] for chunk in chunks[:-2]]
        self.assertEqual("".join(choice["text"] for choice in texts), "first second\n")
        self.assertTrue(
            all(
                choice["logprobs"] is None and choice["finish_reason"] is None
                for choice in texts
            )
        )
        finish = chunks[-2]
        self.assertEqual(
            finish["choices"],
            [{"index": 0, "text": "", "logprobs": None, "finish_reason": "length"}],
        )
        self.assertEqual(finish["timings"]["predicted_n"], 2)
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 2)
        self.assertEqual(chunks[-1]["usage"]["prompt_tokens"], 5)

    def test_stream_usage_is_opt_in(self):
        harness, _ = self.harness(Plan([[4]]))
        for options in ({}, {"stream_options": None}, {"stream_options": {}}):
            with self.subTest(options=options):
                body = {"model": "test-model", "prompt": "once", "stream": True}
                status, _, payload = harness.request(
                    "POST", "/v1/completions", {**body, **options}
                )
                self.assertEqual(status, 200, payload)
                chunks = [json.loads(event) for event in data_events(payload)[:-1]]
                self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
                self.assertTrue(all("usage" not in chunk for chunk in chunks))

    def test_stop_string_ends_the_text_and_cancels_generation(self):
        harness, runtime = self.harness(
            Plan([[14], [15], [4]], reason="length", delay=0.01)
        )
        status, response = self.complete(harness, stop=["unused", "second"])
        self.assertEqual(status, 200, response)
        self.assertEqual(response["choices"][0]["text"], "first ")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"]["completion_tokens"], 2)
        self.assertEqual(runtime.cancel_count, 1)
        for stop in ("", ["x"] * 5, 3):
            with self.subTest(stop=stop):
                status, response = self.complete(harness, stop=stop)
                self.assertEqual(status, 400)
                self.assertEqual(
                    response["error"]["message"],
                    "stop must be a string or up to four strings",
                )

    def test_unsupported_fields_are_rejected_by_name(self):
        harness, runtime = self.harness()
        cases = (
            ("suffix", "!", "suffix is not supported"),
            ("echo", True, "echo is not supported"),
            ("logprobs", 0, "logprobs is not supported"),
            ("logprobs", 5, "logprobs is not supported"),
            ("best_of", 2, "best_of must be 1"),
            ("n", 2, "n must be 1"),
            ("n", True, "n must be 1"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                status, response = self.complete(harness, **{field: value})
                self.assertEqual(status, 400)
                self.assertEqual(response["error"]["message"], message)
        batches = (["once", "upon"], ["once"], [[14, 15]], [14, "upon"], 3, None)
        for prompt in batches:
            with self.subTest(prompt=prompt):
                status, response = self.complete(harness, prompt=prompt)
                self.assertEqual(status, 400)
                self.assertEqual(
                    response["error"]["message"],
                    "prompt must be one string or one array of token ids",
                )
        # An empty string is a prompt only for a tokenizer that adds a BOS.
        harness.tokenizer.bos = None
        for prompt in ("", []):
            with self.subTest(prompt=prompt):
                status, response = self.complete(harness, prompt=prompt)
                self.assertEqual(status, 400)
                self.assertEqual(
                    response["error"]["message"], "prompt must not be empty"
                )
        self.assertEqual(runtime.requests, [])
        harness.tokenizer.bos = 9
        status, response = self.complete(harness, prompt="")
        self.assertEqual(status, 200, response)
        self.assertEqual(runtime.requests[0].prompt_tokens, (9,))

    def test_default_values_of_unsupported_fields_are_accepted(self):
        harness, runtime = self.harness()
        defaults = {
            "suffix": None,
            "echo": False,
            "logprobs": None,
            "best_of": 1,
            "repetition_penalty": 1,
        }
        for fields in (defaults, {key: None for key in defaults}, {"n": 1}):
            with self.subTest(fields=fields):
                status, response = self.complete(harness, **fields)
                self.assertEqual(status, 200, response)
        self.assertEqual(len(runtime.requests), 3)

    def test_omitted_max_tokens_is_openais_default_within_the_context(self):
        # 16 tokens, not all the context leaves as in chat; a prompt that
        # leaves less context generates up to the rest instead of failing.
        harness, runtime = self.harness(max_context=64)
        for fields in ({}, {"max_tokens": None}):
            status, response = self.complete(harness, **fields)
            self.assertEqual(status, 200, response)
            self.assertEqual(runtime.requests[-1].logical_max_output_tokens, 16)
        harness, runtime = self.harness(max_context=16)
        status, response = self.complete(harness)
        self.assertEqual(status, 200, response)
        request = runtime.requests[0]
        self.assertEqual(
            request.logical_max_output_tokens, 16 - len(request.prompt_tokens)
        )

    def test_sampling_budget_seed_and_priority_are_validated_as_in_chat(self):
        harness, runtime = self.harness(max_context=16)
        status, response = self.complete(
            harness,
            temperature=0.7,
            top_p=0.5,
            top_k=5,
            seed=7,
            priority="foreground",
        )
        self.assertEqual(status, 200, response)
        request = runtime.requests[0]
        self.assertEqual(request.cohort, wire.Cohort.SAMPLING)
        self.assertAlmostEqual(request.sampling.temperature, 0.7)
        self.assertEqual((request.sampling.top_p, request.sampling.top_k), (0.5, 5))
        self.assertEqual(request.seed, 7)
        self.assertEqual(request.priority, wire.RequestPriority.FOREGROUND)
        invalid = (
            ({"temperature": -1}, "invalid sampling parameters"),
            ({"top_k": 33}, "invalid sampling parameters"),
            ({"presence_penalty": 1}, "output transformation is not supported"),
            ({"repetition_penalty": 1.1}, "output transformation is not supported"),
            ({"logit_bias": {"1": 2}}, "output transformation is not supported"),
            ({"seed": 2**64}, "seed must be an unsigned 64-bit integer"),
            ({"priority": "urgent"}, "priority must be"),
            ({"max_tokens": 0}, "max_tokens must be a positive integer"),
            ({"max_tokens": 12}, "prompt and max_tokens exceed the context window"),
            ({"model": "other-model"}, "model other-model not found"),
            ({"timeout": 0}, "timeout must be positive"),
        )
        for fields, message in invalid:
            with self.subTest(fields=fields):
                status, response = self.complete(harness, **fields)
                self.assertIn(status, (400, 404))
                self.assertIn(message, response["error"]["message"])
        self.assertEqual(len(runtime.requests), 1)

    def test_openai_sdk_reads_completions_and_streams(self):
        harness, _ = self.harness(Plan([[14], [15]]), Plan([[14], [15]]))
        host, port = harness.server.server_address
        with OpenAI(
            api_key="test", base_url=f"http://{host}:{port}/v1", max_retries=0
        ) as client:
            completion = client.completions.create(
                model="test-model", prompt="once upon a time", max_tokens=4
            )
            self.assertEqual(completion.choices[0].text, "first second\n")
            self.assertEqual(completion.choices[0].finish_reason, "stop")
            self.assertEqual(completion.usage.completion_tokens, 2)
            chunks = list(
                client.completions.create(
                    model="test-model",
                    prompt=[14, 15],
                    max_tokens=4,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )
        self.assertEqual(
            "".join(chunk.choices[0].text for chunk in chunks if chunk.choices),
            "first second\n",
        )
        self.assertEqual(chunks[-1].usage.prompt_tokens, 2)


class TextCompletionLifecycleTests(unittest.TestCase):
    """Deadlines, disconnects and failures end a text completion as they end
    the same chat request."""

    def harness(self, *plans, **options):
        runtime = FakeRuntime(*plans)
        harness = Harness(runtime, **options)
        self.addCleanup(harness.close)
        return harness, runtime

    @staticmethod
    def wait_for_cancel(runtime):
        deadline = time.monotonic() + 2
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        return runtime.cancel_count

    def test_deadline_cancels_and_the_next_request_runs(self):
        for path, body in (CHAT, COMPLETION):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    harness, runtime = self.harness(
                        Plan([[4]], block=True), Plan([[4]])
                    )
                    request = {**body, "stream": stream, "timeout": 0.3}
                    status, _, payload = harness.request("POST", path, request)
                    if stream:
                        self.assertEqual(status, 200)
                        error = json.loads(data_events(payload)[-2])["error"]
                        self.assertEqual(error["code"], "request_timeout")
                    else:
                        self.assertEqual(status, 504)
                    self.assertEqual(runtime.cancel_count, 1)
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200, payload)
                    self.assertIn(b"plain answer", payload)

    def test_disconnect_cancels_and_the_next_request_runs(self):
        for path, body in (CHAT, COMPLETION):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    plan = (
                        Plan([[4]] * 100, delay=0.02)
                        if stream
                        else Plan([[4]], block=True)
                    )
                    harness, runtime = self.harness(plan, Plan([[4]]), timeout=10)
                    connection = http.client.HTTPConnection(
                        *harness.server.server_address, timeout=2
                    )
                    connection.request(
                        "POST",
                        path,
                        json.dumps({**body, "stream": stream}),
                        {"Content-Type": "application/json"},
                    )
                    if stream:
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        response.readline()
                        response.close()
                    else:
                        self.assertTrue(plan.started.wait(1))
                    connection.close()
                    self.assertEqual(self.wait_for_cancel(runtime), 1)
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 200, payload)
                    self.assertIn(b"plain answer", payload)

    def test_idle_stream_keeps_alive_with_a_comment_then_empty_chunks(self):
        # As the chat stream does: an SSE comment until the request starts,
        # then empty text chunks, data events that clients which time out on
        # missing data events accept.
        for before_start in (True, False):
            with self.subTest(before_start=before_start):
                plan = Plan([[4]], block=True, before_start=before_start)
                harness, _ = self.harness(plan)
                with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.02):
                    connection, response = harness.open_stream(
                        COMPLETION[0], {**COMPLETION[1], "stream": True}
                    )
                    try:
                        self.assertEqual(response.status, 200)
                        if before_start:
                            self.assertEqual(
                                response.readline(), b": splash-keepalive\n"
                            )
                            self.assertFalse(plan.started.is_set())
                            plan.start_release.set()
                        chunk = next_data(response)
                        self.assertEqual(chunk.get("object"), "text_completion", chunk)
                        self.assertEqual(chunk["choices"][0]["text"], "")
                        self.assertIsNone(chunk["choices"][0]["finish_reason"])
                    finally:
                        plan.start_release.set()
                        plan.release.set()
                        response.read()
                        connection.close()

    def test_native_failure_is_reported_as_chat_reports_it(self):
        for path, body in (CHAT, COMPLETION):
            with self.subTest(path=path):
                harness, _ = self.harness(
                    Plan(error=("runtime_error", "broken")),
                    Plan(error=("runtime_error", "broken")),
                )
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 500)
                self.assertEqual(json.loads(payload)["error"]["message"], "broken")
                status, _, payload = harness.request(
                    "POST", path, {**body, "stream": True}
                )
                self.assertEqual(status, 200)
                events = data_events(payload)
                self.assertEqual(events[-1], "[DONE]")
                self.assertEqual(
                    json.loads(events[-2])["error"],
                    {
                        "message": "broken",
                        "type": "server_error",
                        "code": "runtime_error",
                    },
                )


if __name__ == "__main__":
    unittest.main()

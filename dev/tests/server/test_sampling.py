import json
import unittest
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
from server import backend as backend_api
from server import frontend as request_frontend
from server import protocol as native_wire


class SamplingTests(HarnessTestCase):
    def test_sampling_fields_reach_the_engine_and_are_validated_per_field(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        # Qwen's recommended non-thinking sampling, whose min_p of 0 drops
        # nothing, penalties beyond it, and min_p up to its limit.
        accepted = (
            (
                {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                    "repetition_penalty": 1.0,
                },
                {"min_p": 0.0},
            ),
            ({"repetition_penalty": 1.1}, {"min_p": None}),
            ({"repetition_penalty": 2.5, "frequency_penalty": -2}, {}),
            (
                {"presence_penalty": 2, "frequency_penalty": 2, "top_k": 32},
                {"repetition_penalty": None},
            ),
            ({"repetition_penalty": 1e-40, "frequency_penalty": 0.25}, {}),
            ({"temperature": 0.7, "min_p": 0.25}, {}),
            ({"min_p": 1}, {}),
        )
        for fields, neutral in accepted:
            with self.subTest(fields=fields):
                runtime.requests.clear()
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(**fields, **neutral)
                )
                self.assertEqual(status, 200, payload)
                sampling = runtime.requests[0].frame.sampling
                for name, value in fields.items():
                    self.assertEqual(getattr(sampling, name), value)
        # A nonzero temperature below 0.01 samples at 0.01; zero stays
        # greedy.
        for temperature, sampled in (
            (1e-46, 0.01),
            (1e-40, 0.01),
            (2.0**-126, 0.01),
            (0.005, 0.01),
            (0.01, 0.01),
            (0.0, 0.0),
        ):
            with self.subTest(temperature=temperature):
                runtime.requests.clear()
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(temperature=temperature)
                )
                self.assertEqual(status, 200, payload)
                self.assertEqual(
                    runtime.requests[0].frame.sampling.temperature, sampled
                )
        refused = (
            ({"temperature": 2.5}, "temperature must be a number in [0, 2]"),
            ({"temperature": -0.5}, "temperature must be a number in [0, 2]"),
            ({"temperature": 1e300}, "temperature must be a number in [0, 2]"),
            ({"top_p": 0}, "top_p must be a number in (0, 1]"),
            ({"top_p": 1e-46}, "top_p must be a number in (0, 1]"),
            ({"min_p": 1.5}, "min_p must be a number in [0, 1]"),
            ({"min_p": -0.1}, "min_p must be a number in [0, 1]"),
            ({"min_p": "0.1"}, "min_p must be a number in [0, 1]"),
            ({"min_p": True}, "min_p must be a number in [0, 1]"),
            ({"presence_penalty": 2.5}, "presence_penalty must be a number in [-2, 2]"),
            (
                {"presence_penalty": False},
                "presence_penalty must be a number in [-2, 2]",
            ),
            (
                {"frequency_penalty": -3},
                "frequency_penalty must be a number in [-2, 2]",
            ),
            ({"frequency_penalty": "1"}, "frequency_penalty must be a number"),
            ({"repetition_penalty": 0}, "repetition_penalty must be a positive number"),
            (
                {"repetition_penalty": -1},
                "repetition_penalty must be a positive number",
            ),
            ({"repetition_penalty": 1e-46}, "repetition_penalty must be a positive"),
            ({"repetition_penalty": 1e39}, "repetition_penalty must be a positive"),
            (
                {"repetition_penalty": True},
                "repetition_penalty must be a positive number",
            ),
            ({"logit_bias": {"1": 2}}, "logit_bias is not supported"),
        )
        for fields, message in refused:
            with self.subTest(fields=fields):
                runtime.requests.clear()
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(**fields)
                )
                self.assertEqual(status, 400, payload)
                self.assertIn(message, json.loads(payload)["error"]["message"])
                self.assertEqual(runtime.requests, [])
        # Messages has temperature, top_p and top_k, validated as Chat's;
        # no penalty or min_p, which keep their defaults.
        for top_k, sent in ((20, 20), (0, 0), (-1, 0)):
            with self.subTest(api="messages", top_k=top_k):
                runtime.requests.clear()
                status, _, payload = harness.request(
                    "POST",
                    "/v1/messages",
                    anthropic_body(temperature=0.7, top_p=0.8, top_k=top_k),
                )
                self.assertEqual(status, 200, payload)
                self.assertEqual(
                    runtime.requests[0].frame.sampling,
                    native_wire.SamplingParameters(
                        temperature=0.7, top_p=0.8, top_k=sent
                    ),
                )
        runtime.requests.clear()
        status, _, payload = harness.request(
            "POST", "/v1/messages", anthropic_body(top_p=0)
        )
        self.assertEqual(status, 400, payload)
        self.assertIn(
            "top_p must be a number in (0, 1]",
            json.loads(payload)["error"]["message"],
        )
        self.assertEqual(runtime.requests, [])

    def test_top_k_takes_any_positive_integer_and_0_or_minus_1_disables_it(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        # 0 and -1 keep every token, which the frame says with 0; a top_k
        # past the vocabulary keeps every token too, however large.
        for top_k, sent in (
            (1, 1),
            (33, 33),
            (1000, 1000),
            (0, 0),
            (-1, 0),
            (2**40, 0xFFFFFFFF),
        ):
            with self.subTest(top_k=top_k):
                runtime.requests.clear()
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(top_k=top_k)
                )
                self.assertEqual(status, 200, payload)
                self.assertEqual(runtime.requests[0].frame.sampling.top_k, sent)
        for top_k in (-2, 2.5, True, "20"):
            with self.subTest(top_k=top_k):
                runtime.requests.clear()
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(top_k=top_k)
                )
                self.assertEqual(status, 400, payload)
                self.assertEqual(
                    json.loads(payload)["error"]["message"],
                    "top_k must be 0 or -1 (disabled) or a positive integer",
                )
                self.assertEqual(runtime.requests, [])

    def test_sampling_field_lists_follow_the_wire_dataclass(self):
        self.assertEqual(
            set(request_frontend.SAMPLING_NUMBERS) | {"top_k"},
            set(native_wire.SAMPLING_FIELDS),
        )
        runtime = FakeRuntime()
        defaults = {"temperature": 0.7, "top_p": 0.8, "top_k": 40}
        harness = self.harness(runtime, sampling_defaults=defaults)
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(**dict.fromkeys(native_wire.SAMPLING_FIELDS)),
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            runtime.requests[0].frame.sampling,
            native_wire.SamplingParameters(**defaults),
        )

    def test_responses_forward_the_sampling_fields_chat_validates(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        # A value apart from its default for every sampling option the frame
        # carries.
        fields = {
            "temperature": 0.5,
            "top_p": 0.5,
            "top_k": 5,
            "presence_penalty": 1.5,
            "frequency_penalty": 0.5,
            "repetition_penalty": 1.05,
            "min_p": 0.25,
        }
        self.assertEqual(tuple(fields), native_wire.SAMPLING_FIELDS)
        status, _, payload = harness.request(
            "POST", "/v1/responses", responses_body(store=False, **fields)
        )
        self.assertEqual(status, 200, payload)
        sampling = runtime.requests[0].frame.sampling
        for name, value in fields.items():
            self.assertEqual(getattr(sampling, name), value)
        status, _, payload = harness.request(
            "POST", "/v1/responses", responses_body(logit_bias={"1": 2})
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(
            json.loads(payload)["error"]["message"], "logit_bias is not supported"
        )
        self.assertEqual(len(runtime.requests), 1)

    def test_requests_get_the_served_familys_sampling_unless_they_name_their_own(
        self,
    ):
        tokenizer = FakeTokenizer()
        backend = backend_api.NativeBackend(
            FakeRuntime(), tokenizer, lambda _record: None
        )
        self.addCleanup(backend.close)
        app = make_frontend(
            tokenizer,
            backend,
            "test-model",
            128,
            1,
            2,
            vision=True,
            sampling_defaults={"temperature": 0.7, "top_p": 0.8, "top_k": 40},
        )
        body = chat_body()
        body.pop("temperature")
        job = app.prepare(body, deadline=FOREVER)
        self.assertEqual(job.sampling, native_wire.SamplingParameters(0.7, 0.8, 40))
        # A request's own fields replace them.
        job = app.prepare(
            chat_body(temperature=0.3, top_p=0.5, top_k=5), deadline=FOREVER
        )
        self.assertEqual(job.sampling, native_wire.SamplingParameters(0.3, 0.5, 5))

    def test_a_familys_sampling_is_what_a_request_may_send(self):
        # A default a request could not send would refuse every request that
        # leaves the field out, blaming the client.
        tokenizer = FakeTokenizer()
        for defaults in ({"top_p": 0}, {"top_k": 1.5}, {"seed": 1}):
            with self.subTest(defaults=defaults):
                with self.assertRaisesRegex(ValueError, "invalid sampling default"):
                    make_frontend(
                        tokenizer,
                        None,
                        "test-model",
                        128,
                        1,
                        2,
                        vision=True,
                        sampling_defaults=defaults,
                    )

    def test_a_request_without_a_seed_gets_a_random_one(self):
        tokenizer = FakeTokenizer()
        backend = backend_api.NativeBackend(
            FakeRuntime(), tokenizer, lambda _record: None
        )
        self.addCleanup(backend.close)
        app = make_frontend(tokenizer, backend, "test-model", 128, 1, 2, vision=True)
        with mock.patch("server.frontend.secrets.randbits", return_value=123):
            job = app.prepare(chat_body(), deadline=FOREVER)
            self.assertEqual(job.seed, 123)
            # -1 asks for a random seed too; another negative one is taken
            # as its 64 bits.
            for seed, expected in (
                (-1, 123),
                (-2, 2**64 - 2),
                (-(2**63), 2**63),
                (2**64 - 1, 2**64 - 1),
            ):
                with self.subTest(seed=seed):
                    job = app.prepare(chat_body(seed=seed), deadline=FOREVER)
                    self.assertEqual(job.seed, expected)


if __name__ == "__main__":
    unittest.main()

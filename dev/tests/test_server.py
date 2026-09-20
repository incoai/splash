import base64
import concurrent.futures
import hashlib
import http.client
import io
import json
import math
import os
import random
import signal
import socket
import struct
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from openai import OpenAI
from tokenizers import Tokenizer, decoders, models

from server import api_shapes, diagnostics, tool_schema
from server import backend as backend_api
from server import constraints as generation_constraints
from server import errors as api_errors
from server import frontend as request_frontend
from server import output as model_output
from server import protocol as native_wire
from server import server as api
from server.api_shapes import _namespace_alias, normalize_responses_input
from server.tool_schema import MAX_JSON_NESTING, _grammar_compatible_schema


def _byte_alphabet():
    byte_values = [
        *range(ord("!"), ord("~") + 1),
        *range(0xA1, 0xAD),
        *range(0xAE, 0x100),
    ]
    codepoints = list(byte_values)
    offset = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + offset)
            offset += 1
    return dict(zip(byte_values, map(chr, codepoints), strict=True))


def _byte_backend(fragments):
    alphabet = _byte_alphabet()
    vocabulary = {}
    for token_id in range(max(fragments) + 1):
        value = fragments.get(token_id, f"<unused_{token_id}>")
        token = "".join(alphabet[byte] for byte in value.encode())
        vocabulary[token] = token_id
    vocabulary["[UNK]"] = len(vocabulary)
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.decoder = decoders.ByteLevel()
    return backend


class FakeTokenizer:
    def __init__(self):
        self.fragments = {
            1: "because ",
            2: "</thi",
            3: "nk>answer\n",
            4: "plain answer\n",
            5: (
                "<tool_call>\n<function=weather>\n<parameter=city>\n"
                "Paris\n</parameter>\n</function>\n</tool_call>\n"
            ),
            7: "<tool_call><function=time></function></tool_call>",
            8: (
                "<tool_call><function=weather><parameter=city>"
                "3</parameter></function></tool_call>"
            ),
            9: (
                "<tool_call><function=weather></function></tool_call>"
                "<tool_call><function=time></function></tool_call>"
            ),
            10: '{"x":3}',
            11: (
                "<tool_call>\n<function=multi_agent_v1__spawn_agent>\n"
                "<parameter=message>\ninspect\n</parameter>\n"
                "</function>\n</tool_call>\n"
            ),
            12: '{"x":',
            13: ("<tool_call>\n<function=weather>\n<parameter=city>\nPar"),
            14: "first ",
            15: "second\n",
            16: ("<tool_call>\n<function=time>\n</function>\n</tool_call>\n"),
            17: "\n<tool_",
            18: (
                "call>\n<function=weather>\n<parameter=city>\n"
                "Paris\n</parameter>\n</function>\n</tool_call>\n"
            ),
            19: "<tool_ca",
            20: " \nalpha \t",
            21: "<tool_",
            22: (
                "call>\n<function=weather>\n<parameter=city>\n"
                "Paris\n</parameter>\n</function>\n</tool_call>"
            ),
            23: "\n beta \t\n",
            24: ("answer <<tool_call>\n<function=weather>\n<parameter=city>\nPar"),
            25: "nk>",
            26: "</think>",
            27: "answer\n",
        }
        self.backend_tokenizer = _byte_backend(self.fragments)
        self.templates = []

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        rendered = "<|im_start|>assistant\n<think>\n"
        if not kwargs.get("enable_thinking", True):
            rendered += "\n</think>\n\n"
        return rendered if kwargs.get("tokenize") is False else [101, 102]

    def __call__(self, text, **kwargs):
        return {"input_ids": [101, 102]}

    def decode(self, token_ids, **kwargs):
        return "".join(self.fragments[token] for token in token_ids)

    def convert_tokens_to_ids(self, token):
        return 26 if token == "</think>" else None


class EndFailTokenizer(FakeTokenizer):
    def __init__(self):
        super().__init__()
        self.fragments[6] = "tail"
        self.backend_tokenizer = _byte_backend(self.fragments)
        self.tail_decodes = 0

    def decode(self, token_ids, **kwargs):
        if token_ids == [6]:
            self.tail_decodes += 1
            if self.tail_decodes == 1:
                raise RuntimeError("decode failed during end")
        return super().decode(token_ids, **kwargs)


class RenderFailTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if any(
            isinstance(message.get("content"), str) and "\ud800" in message["content"]
            for message in messages
        ):
            raise TypeError("invalid surrogate")
        return super().apply_chat_template(messages, **kwargs)


class TemplateTokenizer(FakeTokenizer):
    """Render real HF/Jinja templates while keeping the test runtime's tokens."""

    def __init__(self, template):
        super().__init__()
        from transformers import PreTrainedTokenizerFast

        self.renderer = PreTrainedTokenizerFast(tokenizer_object=self.backend_tokenizer)
        self.renderer.chat_template = template

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return self.renderer.apply_chat_template(messages, **kwargs)


class BlockingTokenizer(FakeTokenizer):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.entered = threading.Event()
        self.active = 0
        self.maximum_active = 0
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        with self.lock:
            self.calls += 1
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            if self.calls >= 2:
                self.entered.set()
        self.release.wait(2)
        try:
            return super().apply_chat_template(messages, **kwargs)
        finally:
            with self.lock:
                self.active -= 1


class PassthroughStreamer:
    def __init__(self, tokenizer, callback, *args):
        self.tokenizer = tokenizer
        self.callback = callback
        self.stop_sequence = None

    def put_tokens(self, token_ids):
        text = self.tokenizer.decode(token_ids)
        if text:
            self.callback(text)

    def end(self):
        pass

    @staticmethod
    def count_reasoning_tokens(_enabled):
        return 0


class SeededRandomChunkStreamer(PassthroughStreamer):
    def __init__(self, tokenizer, callback, *args):
        super().__init__(tokenizer, callback)
        self.random = random.Random(20260821)

    def put_tokens(self, token_ids):
        text = self.tokenizer.decode(token_ids)
        while text:
            length = self.random.randint(1, 4)
            self.callback(text[:length])
            text = text[length:]


class ByteLevelTestTokenizer:
    def __init__(self, raw_tokens):
        alphabet = _byte_alphabet()
        vocabulary = {"[UNK]": 0, "<eos>": 1}
        self.token_ids = []
        for raw in raw_tokens:
            token = "".join(alphabet[value] for value in raw)
            token_id = vocabulary.setdefault(token, len(vocabulary))
            self.token_ids.append(token_id)
        self.backend_tokenizer = Tokenizer(
            models.WordLevel(vocabulary, unk_token="[UNK]")
        )
        self.backend_tokenizer.add_special_tokens(["<eos>"])
        self.backend_tokenizer.decoder = decoders.ByteLevel()
        self.eos_token_id = vocabulary["<eos>"]

    def decode(self, token_ids, skip_special_tokens=True, **_kwargs):
        return self.backend_tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )


class Plan:
    def __init__(
        self,
        batches=(),
        reason="stop",
        block=False,
        error=None,
        exception=None,
        delay=0,
        before_start=False,
        after_terminal=False,
        matched_tokens=1,
    ):
        self.batches = list(batches)
        self.reason = reason
        self.error = error
        self.exception = exception
        self.delay = delay
        self.matched_tokens = matched_tokens
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.cancelled = threading.Event()
        self.start_release = threading.Event()
        self.terminal = threading.Event()
        self.terminal_release = threading.Event()
        if not before_start:
            self.start_release.set()
        if not after_terminal:
            self.terminal_release.set()


class FakeCall:
    def __init__(self, runtime, plan, request_id, request, on_event, on_complete):
        self.runtime = runtime
        self.plan = plan
        self.request_id = request_id
        self.request = request
        self.on_event = on_event
        self.on_complete = on_complete
        self.callback_errors = ()
        self.cancel_requested = False
        self._result = None
        self._error = None
        self._done = False
        self._lock = threading.Lock()

    @property
    def done(self):
        with self._lock:
            return self._done

    def emit(self, event):
        try:
            self.on_event(self, event)
        except Exception as error:
            self.callback_errors = (*self.callback_errors, error)

    def complete(self, *, result=None, error=None):
        with self._lock:
            if self._done:
                return False
            self._result = result
            self._error = error
            self._done = True
        self.on_complete(self)
        return True

    def result(self, _timeout=None):
        with self._lock:
            if not self._done:
                raise TimeoutError("fake current runtime call is not complete")
            if self._error is not None:
                raise self._error
            return self._result

    def cancel(self):
        with self._lock:
            if self._done or self.cancel_requested:
                return False
            self.cancel_requested = True
        self.runtime.cancel_count += 1
        self.plan.cancelled.set()
        self.plan.start_release.set()
        self.plan.release.set()
        self.plan.terminal_release.set()
        return True


class FakeRuntime:
    def __init__(self, *plans):
        self.plans = list(plans)
        self.requests = []
        self.calls = []
        self.cancel_count = 0
        self.closed = False
        self.ready = True
        self.pending_limit = 32
        self.restart_count = 0
        self.last_crash_trace = None
        self.status_event = native_wire.StatusJsonEvent(
            1,
            native_wire.STATUS_SCHEMA_VERSION,
            json.dumps(
                {
                    "schema_version": native_wire.STATUS_SCHEMA_VERSION,
                    "ready": True,
                    "memory_pressure": "normal",
                    "metal": {"healthy": True},
                    "memory": {
                        "plan_bytes": 100,
                        "actual_bytes": 99,
                        "peak_bytes": 100,
                        "headroom_bytes": 20,
                    },
                    "kv": {
                        "total_pages": 8,
                        "free_pages": 8,
                        "active_pages": 0,
                        "prefix_pages": 0,
                        "pinned_pages": 0,
                    },
                    "requests": {"queued": 0, "resident": 0, "runnable": 0},
                },
                separators=(",", ":"),
            ).encode(),
        )
        self.threads = []

    @property
    def pending_count(self):
        return sum(not call.done for call in self.calls)

    def status(self, timeout=5.0):
        if timeout <= 0:
            raise ValueError("status timeout must be positive")
        return self.status_event

    def submit(self, request, *, on_event, on_complete):
        if self.pending_count >= self.pending_limit:
            raise api.engine_runtime.PendingLimitExceeded(
                "fake current runtime runtime is full"
            )
        plan = self.plans.pop(0) if self.plans else Plan([[4]])
        call = FakeCall(
            self,
            plan,
            len(self.calls) + 1,
            request,
            on_event,
            on_complete,
        )
        self.calls.append(call)
        self.requests.append(request)
        thread = threading.Thread(target=self._run, args=(call,), daemon=True)
        self.threads.append(thread)
        thread.start()
        return call

    def _run(self, call):
        plan = call.plan
        request = call.request
        plan.start_release.wait(2)
        if call.done:
            return
        call.emit(
            native_wire.StartEvent(
                call.request_id,
                (
                    native_wire.CacheDisposition.PREFIX_HIT
                    if plan.matched_tokens
                    else native_wire.CacheDisposition.MISS
                ),
                0,
                plan.matched_tokens,
                8192,
            )
        )
        plan.started.set()
        plan.release.wait(2)
        if plan.exception:
            call.complete(error=plan.exception)
            return
        if plan.error:
            call.complete(error=api_errors.NativeError(*plan.error))
            return
        completion = 0
        if not plan.cancelled.is_set():
            offset = 0
            for batch in plan.batches:
                if plan.cancelled.is_set():
                    break
                tokens = tuple(batch)
                call.emit(native_wire.TokensEvent(call.request_id, offset, tokens))
                offset += len(tokens)
                completion += len(tokens)
                plan.cancelled.wait(plan.delay)
        reason = (
            native_wire.FinishReason.CANCELLED
            if plan.cancelled.is_set()
            else {
                "stop": native_wire.FinishReason.STOP,
                "length": native_wire.FinishReason.LENGTH,
                "cancelled": native_wire.FinishReason.CANCELLED,
            }[plan.reason]
        )
        plan.terminal.set()
        plan.terminal_release.wait(2)
        if reason == native_wire.FinishReason.CANCELLED:
            completion = 0
        done = native_wire.DoneEvent(
            call.request_id,
            reason,
            len(request.prompt_tokens),
            completion,
            1_000,
            2_000,
            3_000,
        )
        call.complete(
            result=api.engine_runtime.GenerationResult(
                call.request_id,
                None,
                tuple(token for batch in plan.batches for token in batch),
                done,
            )
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.ready = False
        for call in self.calls:
            call.cancel()
        for thread in self.threads:
            thread.join(2)


class FakeConstraintFactory:
    def __init__(self):
        self.grammars = []

    def create(self, grammar):
        self.grammars.append(grammar)
        return SimpleNamespace(consume=lambda _tokens: None)

    def stats(self):
        return {}


class Harness:
    def __init__(
        self,
        runtime,
        tokenizer=None,
        queue_size=4,
        timeout=2,
        max_context=128,
        default_max_new=16,
        request_logger=None,
        constraint_factory=None,
        io_timeout=api.HTTP_IO_TIMEOUT,
        thinking_codec=None,
        api_key=None,
        webui=True,
    ):
        self.tokenizer = tokenizer or FakeTokenizer()
        runtime.pending_limit = queue_size
        self.backend = backend_api.NativeBackend(
            runtime, self.tokenizer, request_logger=request_logger
        )
        self.app = request_frontend.Frontend(
            self.tokenizer,
            self.backend,
            "test-model",
            max_context,
            default_max_new,
            timeout,
            2,
            constraint_factory,
            thinking_codec=thinking_codec,
        )
        self.server = api.FrontendServer(
            ("127.0.0.1", 0),
            self.app,
            io_timeout,
            request_capacity=queue_size,
            api_key=api_key,
            webui=webui,
        )
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        if headers is None:
            headers = {"Content-Type": "application/json"} if body is not None else {}
        data = json.dumps(body) if body is not None else None
        connection.request(method, path, data, headers)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, response.getheader("Content-Type"), payload

    def raw_post(self, payload, length):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        connection.putrequest("POST", "/v1/chat/completions")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(length))
        connection.endheaders(payload)
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response.status, body

    def open_stream(self, path, body):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        connection.request(
            "POST",
            path,
            json.dumps(body),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return connection, response

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.backend.close()


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.harnesses = []

    def tearDown(self):
        for harness in reversed(self.harnesses):
            harness.close()

    def harness(self, runtime, **kwargs):
        harness = Harness(runtime, **kwargs)
        self.harnesses.append(harness)
        return harness

    def test_expected_client_disconnect_does_not_print_server_traceback(self):
        harness = self.harness(FakeRuntime())
        with mock.patch.object(api.ThreadingHTTPServer, "handle_error") as parent:
            with mock.patch.object(
                api.sys,
                "exc_info",
                return_value=(ConnectionResetError, ConnectionResetError(), None),
            ):
                harness.server.handle_error(None, ("127.0.0.1", 1))
            parent.assert_not_called()

            unexpected = RuntimeError("unexpected")
            with mock.patch.object(
                api.sys,
                "exc_info",
                return_value=(RuntimeError, unexpected, None),
            ):
                harness.server.handle_error(None, ("127.0.0.1", 1))
            parent.assert_called_once()

    def test_constraint_factory_caches_pristine_matchers_with_lru(self):
        testcase = self

        class Matcher:
            builds = []

            @staticmethod
            def validate_grammar(grammar, tokenizer):
                testcase.assertEqual(tokenizer, "guidance-tokenizer")
                return "invalid" if grammar == "bad" else None

            def __init__(self, tokenizer, grammar, log_level=0):
                testcase.assertEqual((tokenizer, log_level), ("guidance-tokenizer", 0))
                self.grammar = grammar
                self.builds.append(grammar)

            def is_error(self):
                return False

            def deep_copy(self):
                return SimpleNamespace(grammar=self.grammar, copy=True)

        with (
            mock.patch(
                "server.constraints.guidance_tokenizer",
                return_value="guidance-tokenizer",
            ),
            mock.patch("server.constraints.LLMatcher", Matcher),
            mock.patch("server.constraints.LLExecutor", return_value="executor"),
            mock.patch(
                "server.constraints.TokenConstraint",
                side_effect=lambda matcher, executor: (matcher, executor),
            ),
        ):
            factory = generation_constraints.ConstraintFactory(object(), cache_size=2)
            first = factory.create("one")
            second = factory.create("one")
            factory.create("two")
            factory.create("three")
            factory.create("one")
            with self.assertRaisesRegex(api.APIError, "invalid"):
                factory.create("bad")

        self.assertIsNot(first[0], second[0])
        self.assertEqual(Matcher.builds, ["one", "two", "three", "one"])
        self.assertEqual(
            factory.stats(),
            {"entries": 2, "capacity": 2, "hits": 1, "misses": 4},
        )

    def test_constraint_factory_shares_concurrent_cold_grammar_build(self):
        class Matcher:
            builds = 0

            @staticmethod
            def validate_grammar(_grammar, _tokenizer):
                return None

            def __init__(self, _tokenizer, grammar, log_level=0):
                self.grammar = grammar
                type(self).builds += 1

            def is_error(self):
                return False

            def deep_copy(self):
                return SimpleNamespace(grammar=self.grammar)

        with (
            mock.patch("server.constraints.guidance_tokenizer", return_value=object()),
            mock.patch("server.constraints.LLMatcher", Matcher),
            mock.patch("server.constraints.LLExecutor", return_value=object()),
            mock.patch(
                "server.constraints.TokenConstraint",
                side_effect=lambda matcher, _: matcher,
            ),
        ):
            factory = generation_constraints.ConstraintFactory(object())
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(factory.create, ["shared"] * 4))

        self.assertTrue(all(result.grammar == "shared" for result in results))
        self.assertEqual(Matcher.builds, 1)
        self.assertEqual(factory.stats()["hits"], 3)

    @staticmethod
    def openai_client(harness):
        host, port = harness.server.server_address
        return OpenAI(
            api_key="test",
            base_url=f"http://{host}:{port}/v1",
            max_retries=0,
            timeout=3,
        )

    @staticmethod
    def body(**extra):
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        }
        body.update(extra)
        return body

    @staticmethod
    def responses_body(**extra):
        body = {
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            "temperature": 0,
        }
        body.update(extra)
        return body

    @staticmethod
    def anthropic_body(**extra):
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "temperature": 0,
        }
        body.update(extra)
        return body

    @staticmethod
    def next_sse_data(response):
        data = []
        while True:
            raw = response.readline()
            if not raw:
                return None
            line = raw.decode().rstrip("\r\n")
            if not line:
                if data:
                    return "\n".join(data)
                continue
            if line.startswith("data:"):
                data.append(line[5:].lstrip())

    @staticmethod
    def response_events(payload):
        events = []
        for block in payload.decode().strip().split("\n\n"):
            lines = block.splitlines()
            event = next(line[7:] for line in lines if line.startswith("event: "))
            data = json.loads(
                next(line[6:] for line in lines if line.startswith("data: "))
            )
            if data["type"] != event:
                raise AssertionError(f"SSE event mismatch: {event} != {data['type']}")
            events.append(data)
        return events

    def test_callback_streamer_emits_no_space_text_before_end(self):
        for raw_tokens in (
            [b"abcdefghijklmnop", b"qrst", b"uv", b"wx"],
            [b'{"', b"abcdefgh", b"ij", b'":', b"1234", b"567890", b"}"],
        ):
            tokenizer = ByteLevelTestTokenizer(raw_tokens)
            chunks = []
            streamer = backend_api.CallbackStreamer(tokenizer, chunks.append)
            for token_id in tokenizer.token_ids:
                streamer.put_tokens([token_id])
            expected = tokenizer.decode(tokenizer.token_ids)
            self.assertGreater(len(chunks), 1)
            self.assertEqual("".join(chunks), expected)
            before_end = list(chunks)
            streamer.end()
            self.assertEqual(chunks, before_end)

    def test_callback_streamer_matches_one_shot_for_random_token_batches(self):
        family = "👨‍👩‍👧‍👦".encode()
        cases = (
            [bytes([value]) for value in family],
            [
                b"e",
                b"\xcc",
                b"\x81",
                b" + ",
                *[bytes([value]) for value in "中文".encode()],
            ],
            [b"prefix", b" ", b"suffix"],
            [b"\xf0", b"\x9f"],
        )
        for case_index, raw_tokens in enumerate(cases):
            tokenizer = ByteLevelTestTokenizer(raw_tokens)
            token_ids = list(tokenizer.token_ids)
            token_ids.insert(max(1, len(token_ids) // 2), tokenizer.eos_token_id)
            expected = tokenizer.decode(token_ids)
            for seed in range(50):
                randomizer = random.Random(seed)
                chunks = []
                streamer = backend_api.CallbackStreamer(tokenizer, chunks.append)
                offset = 0
                while offset < len(token_ids):
                    size = randomizer.randint(1, min(7, len(token_ids) - offset))
                    streamer.put_tokens(token_ids[offset : offset + size])
                    offset += size
                streamer.end()
                self.assertEqual("".join(chunks), expected, (raw_tokens, seed))
                if case_index == 0:
                    self.assertNotIn("�", "".join(chunks))

        tokenizer = ByteLevelTestTokenizer([bytes([value]) for value in range(256)])
        for seed in range(100):
            randomizer = random.Random(seed)
            token_ids = [
                tokenizer.token_ids[randomizer.randrange(256)]
                for _ in range(randomizer.randint(1, 128))
            ]
            for _ in range(randomizer.randrange(4)):
                token_ids.insert(
                    randomizer.randrange(len(token_ids) + 1), tokenizer.eos_token_id
                )
            chunks = []
            streamer = backend_api.CallbackStreamer(tokenizer, chunks.append)
            offset = 0
            while offset < len(token_ids):
                size = randomizer.randint(1, min(11, len(token_ids) - offset))
                streamer.put_tokens(token_ids[offset : offset + size])
                offset += size
            streamer.end()
            self.assertEqual("".join(chunks), tokenizer.decode(token_ids), seed)

    def test_callback_streamer_reconciles_an_invalid_decoder_prefix(self):
        tokenizer = ByteLevelTestTokenizer([b'{"', b"value", b'":', b"123", b"}"])
        chunks = []
        streamer = backend_api.CallbackStreamer(tokenizer, chunks.append)
        streamer.decode_stream = SimpleNamespace(
            step=mock.Mock(side_effect=ValueError("Invalid prefix encountered"))
        )
        streamer.put_tokens(tokenizer.token_ids)
        self.assertEqual(chunks, [])
        streamer.end()
        self.assertEqual("".join(chunks), tokenizer.decode(tokenizer.token_ids))

    def test_callback_streamer_stops_across_token_boundaries(self):
        tokenizer = ByteLevelTestTokenizer([b"alpha <ST", b"OP>hidden", b" trailing"])
        chunks = []
        stopped = []
        streamer = backend_api.CallbackStreamer(
            tokenizer,
            chunks.append,
            ("<STOP>", "unused"),
            lambda: stopped.append(True),
        )
        streamer.put_tokens(tokenizer.token_ids)
        streamer.end()
        self.assertEqual("".join(chunks), "alpha ")
        self.assertEqual(stopped, [True])
        self.assertEqual(streamer.stop_sequence, "<STOP>")

    def test_models_and_nonstream_reasoning(self):
        runtime = FakeRuntime(Plan([[1], [2], [3]]))
        harness = self.harness(runtime)
        status, content_type, payload = harness.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/html; charset=utf-8")
        self.assertIn(b"/v1/chat/completions", payload)
        for effort in (b"xhigh", b"medium", b"low", b"none"):
            self.assertIn(b'value="' + effort + b'"', payload)
        for label in (b"XHigh", b"Medium", b"Low", b"Off"):
            self.assertIn(b">" + label + b"</option>", payload)
        for label in (b"Splash", b"New chat", b"Recents"):
            self.assertIn(label, payload)
        self.assertIn(b"reasoning_effort", payload)
        self.assertIn(b"include_usage", payload)
        self.assertIn(b"tokens_per_second", payload)
        self.assertIn(b"allowedEfforts.has(savedEffort)", payload)
        self.assertIn(b"function storageGet", payload)
        self.assertIn(b"function storageSet", payload)
        self.assertIn(b"JSON.parse(storageGet(storeKey))", payload)
        self.assertIn(
            b"Chat still works if browser storage is full or disabled", payload
        )
        self.assertIn(b"function cleanMessage", payload)
        self.assertIn(b"if (input.value === '') input.value = text", payload)
        self.assertIn(b'id="image-input"', payload)
        self.assertIn(b'accept="image/*"', payload)
        self.assertIn(b"reader.readAsDataURL(file)", payload)
        self.assertIn(b"type: 'image_url'", payload)
        self.assertIn(b"input.addEventListener('paste'", payload)
        self.assertIn(b"className = 'waiting'", payload)
        self.assertIn(b"Generating response", payload)
        self.assertIn(b"@keyframes spin", payload)

        status, _, payload = harness.request("GET", "/health")
        self.assertEqual((status, json.loads(payload)), (200, {"status": "ok"}))
        status, _, payload = harness.request("GET", "/ready")
        self.assertEqual((status, json.loads(payload)), (200, {"status": "ready"}))
        status, _, payload = harness.request("GET", "/status")
        snapshot = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(snapshot["schema_version"], native_wire.STATUS_SCHEMA_VERSION)
        self.assertIs(snapshot["ready"], True)
        self.assertEqual(snapshot["memory_pressure"], "normal")
        self.assertIs(snapshot["metal"]["healthy"], True)
        self.assertEqual(
            snapshot["transport"],
            {
                "ready": True,
                "recovering": False,
                "pending": 0,
                "pending_limit": 4,
                "restarts": 0,
                "last_crash_trace": None,
                "status_stale": False,
                "status_age_ms": 0.0,
            },
        )

        runtime.status_event = native_wire.StatusJsonEvent(
            2,
            native_wire.STATUS_SCHEMA_VERSION,
            json.dumps(
                {
                    "schema_version": native_wire.STATUS_SCHEMA_VERSION,
                    "ready": True,
                    "memory_pressure": "normal",
                    "metal": {"healthy": True},
                    "requests": {"submitted": 7, "completed": 5},
                    "admission": {
                        "waiting_memory": 2,
                        "waiting_concurrency": 1,
                        "suspended": 1,
                        "oldest_wait_ms": 1250.0,
                    },
                    "scheduler": {
                        "queued": 1,
                        "waiting_resources": 3,
                        "waiting_prefix": 2,
                        "prefilling": 2,
                        "decoding": 1,
                        "waiting_mask": 0,
                        "prefill_batches": 10,
                        "prefill_rows": 2048,
                        "decode_batches": 7,
                        "decode_batches_by_width": {
                            "b1": 1,
                            "b2": 2,
                            "b3": 3,
                            "b4": 1,
                        },
                    },
                    "kv": {
                        "blocks": 6,
                        "pages_total": 32,
                        "pages_free": 24,
                        "pages_active": 4,
                        "pages_cache": 4,
                        "pages_resident": 8,
                        "resident_backing_bytes": 8192,
                    },
                    "state": {
                        "entries": 2,
                        "pinned": 1,
                        "bytes": 4096,
                        "active_cells": 2,
                        "hits": 7,
                        "misses": 4,
                        "publications": 3,
                        "evictions": 1,
                    },
                    "cache": {
                        "hits": 7,
                        "cold_misses": 4,
                        "reused_tokens": 1024,
                        "lazy_junctions": 2,
                    },
                    "draft_context": {
                        "target_prefill_rows": 10000,
                        "active_rows": 2048,
                        "materialization_rows": 31,
                        "avoided_rows": 7921,
                        "restore_skipped": 1,
                        "resets": 2,
                    },
                    "constraint_masks": {
                        "overlap_batches": 5,
                        "overlap_requests": 8,
                        "last_target_forward_gpu_ms": 72.5,
                        "total_target_forward_gpu_ms": 250.0,
                        "last_residual_wait_ms": 1.5,
                        "total_residual_wait_ms": 12.0,
                    },
                    "memory_actual": {
                        "current_bytes": 700,
                        "peak_bytes": 800,
                    },
                    "memory_governor": {"limit_bytes": 1000, "headroom_bytes": 200},
                    "metrics": {
                        "ttft_ms": {"p50": 10.0, "p95": 12.5, "samples": 5},
                        "prefill_input_tokens": 2048,
                        "prefill_wall_ms": 500.0,
                        "prefill_tokens_per_second": 4096.0,
                        "decode_output_tokens": 32,
                        "decode_wall_ms": 64.0,
                        "decode_tokens_per_second": 500.0,
                        "draft_acceptance_rate": 0.875,
                        "capacity_failures": 1,
                        "metal_failures": 2,
                    },
                },
                separators=(",", ":"),
            ).encode(),
        )
        status, content_type, payload = harness.request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/plain; version=0.0.4; charset=utf-8")
        metrics = payload.decode().splitlines()
        self.assertIn("splash_ready 1", metrics)
        self.assertIn('splash_memory_pressure{state="normal"} 1', metrics)
        self.assertIn("splash_requests_submitted_total 7", metrics)
        self.assertIn("splash_scheduler_waiting_resources 3", metrics)
        self.assertIn("splash_scheduler_waiting_prefix 2", metrics)
        self.assertIn("splash_admission_waiting_memory 2", metrics)
        self.assertIn("splash_admission_waiting_concurrency 1", metrics)
        self.assertIn("splash_admission_suspended 1", metrics)
        self.assertIn("splash_admission_oldest_wait_milliseconds 1250.0", metrics)
        self.assertIn("splash_scheduler_prefill_rows_total 2048", metrics)
        self.assertIn("splash_scheduler_decode_b3_total 3", metrics)
        self.assertIn("splash_kv_pages_free 24", metrics)
        self.assertIn("splash_state_entries 2", metrics)
        self.assertIn("splash_state_hits_total 7", metrics)
        self.assertIn("splash_cache_hits_total 7", metrics)
        self.assertIn("splash_cache_cold_misses_total 4", metrics)
        self.assertIn("splash_cache_reused_tokens_total 1024", metrics)
        self.assertIn("splash_cache_lazy_junctions_total 2", metrics)
        self.assertIn("splash_target_prefill_rows_total 10000", metrics)
        self.assertIn("splash_draft_context_active_rows_total 2048", metrics)
        self.assertIn("splash_draft_context_avoided_rows_total 7921", metrics)
        self.assertIn("splash_draft_state_restore_skipped_total 1", metrics)
        self.assertIn("splash_constraint_mask_overlap_batches_total 5", metrics)
        self.assertIn("splash_constraint_mask_overlap_requests_total 8", metrics)
        self.assertIn(
            "splash_constraint_mask_target_forward_gpu_milliseconds 72.5", metrics
        )
        self.assertIn("splash_constraint_mask_residual_wait_milliseconds 1.5", metrics)
        self.assertIn("splash_prefill_input_tokens_total 2048", metrics)
        self.assertIn("splash_prefill_tokens_per_second 4096.0", metrics)
        self.assertIn("splash_decode_output_tokens_total 32", metrics)
        self.assertIn("splash_decode_tokens_per_second 500.0", metrics)
        self.assertIn("splash_capacity_failures_total 1", metrics)
        self.assertIn("splash_metal_failures_total 2", metrics)
        self.assertIn("splash_response_store_entries 0", metrics)
        self.assertIn("splash_memory_headroom_bytes 200", metrics)
        self.assertIn("splash_ttft_p95_milliseconds 12.5", metrics)
        self.assertIn("splash_draft_acceptance_ratio 0.875", metrics)

        status, _, payload = harness.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        model = json.loads(payload)["data"][0]
        self.assertEqual(model["id"], "test-model")
        self.assertEqual(model["owned_by"], "splash")

        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body(seed=7)
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        message = response["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "because ")
        self.assertEqual(message["content"], "answer\n")
        self.assertEqual(
            response["usage"],
            {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 1},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        )
        self.assertEqual(response["metrics"]["prefill"], {"tokens": 1})
        self.assertEqual(response["metrics"]["decode"], {"tokens": 2})
        self.assertEqual(
            response["metrics"]["request_latency"],
            {
                "start_to_first_token_ms": 1.0,
                "first_token_to_done_ms": 2.0,
                "wall_ms": 3.0,
                "ttft_ms": 1.0,
                "queue_to_start_ms": 0.0,
                "stream_tokens_per_second": 1000.0,
            },
        )
        self.assertNotIn("tokens_per_second", response["metrics"])
        self.assertNotIn("speculative", response["metrics"])
        self.assertNotIn("batch", response["metrics"])
        self.assertEqual(
            response["metrics"]["cache"],
            {"status": "hit", "matched_tokens": 1, "capacity": 8192, "slot": 0},
        )
        self.assertNotIn("queue_ms", response["metrics"])
        self.assertEqual(runtime.requests[0].seed, 7)

    class ImagePadTokenizer(FakeTokenizer):
        """Renders one image placeholder per image part like the pinned
        Qwen template, with pad id 50."""

        def get_chat_template(self, **kwargs):
            return api_shapes.IMAGE_PAD_TOKEN

        def __call__(self, text, **kwargs):
            count = text.count(api_shapes.IMAGE_PAD_TOKEN)
            width = len(api_shapes.IMAGE_PAD_TOKEN)
            return {
                "input_ids": [101, *([50] * count), 102],
                "offset_mapping": [
                    (0, 1),
                    *((1 + i * width, 1 + (i + 1) * width) for i in range(count)),
                    (len(text) - 1, len(text)),
                ],
            }

        def apply_chat_template(self, messages, **kwargs):
            self.templates.append((messages, kwargs))
            rendered = "A"
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    rendered += "".join(
                        kwargs.get("chat_template", api_shapes.IMAGE_PAD_TOKEN)
                        for part in content
                        if part.get("type") == "image_url"
                    )
            rendered += "Z"
            rendered += "<|im_start|>assistant\n<think>\n"
            if not kwargs.get("enable_thinking", True):
                rendered += "\n</think>\n\n"
            return (
                rendered
                if kwargs.get("tokenize") is False
                else self(rendered)["input_ids"]
            )

        def convert_tokens_to_ids(self, token):
            if token == api_shapes.IMAGE_PAD_TOKEN:
                return 50
            return super().convert_tokens_to_ids(token)

    @staticmethod
    def _png_data_url(color=(200, 30, 30)):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def _image_message(self, url=None):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": url or self._png_data_url()},
                },
            ],
        }

    def test_image_parts_expand_placeholders_and_carry_spans_and_pixels(self):
        runtime = FakeRuntime(Plan([[4]]))
        harness = self.harness(runtime, tokenizer=self.ImagePadTokenizer())
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", self.body(messages=[self._image_message()])
        )
        self.assertEqual(status, 200)
        request = runtime.requests[0]
        # 64x64 upscales to the 256x256 minimum: a 16x16 patch grid, 64 tokens.
        self.assertEqual(request.prompt_tokens, (101, *([50] * 64), 102))
        (span,) = request.image_spans
        self.assertEqual(
            (span.offset, span.tokens, span.grid_height, span.grid_width),
            (1, 64, 16, 16),
        )
        self.assertEqual(len(request.image_pixels), 256 * 256 * 3)
        self.assertEqual(request.image_pixels[:3], bytes((200, 30, 30)))
        digest = hashlib.sha256(
            struct.pack("<II", 16, 16) + request.image_pixels
        ).digest()
        self.assertEqual(
            (span.digest_lo, span.digest_hi), struct.unpack_from("<QQ", digest)
        )
        rendered, _ = harness.tokenizer.templates[-1]
        self.assertEqual(
            [part["type"] for part in rendered[0]["content"]], ["text", "image_url"]
        )
        # Repeats reuse the prepared image instead of decoding again.
        harness.request(
            "POST", "/v1/chat/completions", self.body(messages=[self._image_message()])
        )
        self.assertEqual(harness.app.images.stats()["entries"], 1)
        self.assertEqual(runtime.requests[1].image_spans, request.image_spans)

    def test_image_positions_ignore_quoted_vision_tokens(self):
        from tokenizers import pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            additional_special_tokens=[
                api_shapes.IMAGE_PAD_TOKEN,
                "<|vision_start|>",
                "<|vision_end|>",
            ],
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message.role }}: "
            "{% if message.content is string %}{{ message.content }}"
            "{% else %}{% for part in message.content %}"
            "{% if part.type == 'image_url' %}"
            "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
            "{% else %}{{ part.text }}{% endif %}{% endfor %}{% endif %}"
            "{{ '\\n' }}{% endfor %}"
        )
        app = object.__new__(request_frontend.Frontend)
        app.tokenizer = tokenizer
        app.max_context = 1024
        quoted = "中文 📷 <|vision_start|><|image_pad|><|vision_end|>"
        messages = [
            {"role": "system", "content": "Document: " + api_shapes.IMAGE_PAD_TOKEN},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "unused"}},
                    {"type": "text", "text": quoted},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "image_url", "image_url": {"url": "unused"}},
                    {"type": "text", "text": quoted},
                ],
            },
        ]
        template = {"tokenize": True, "return_dict": False}
        baseline = tokenizer.apply_chat_template(messages, **template)
        pad_id = tokenizer.convert_tokens_to_ids(api_shapes.IMAGE_PAD_TOKEN)
        all_pads = [i for i, token in enumerate(baseline) if token == pad_id]
        self.assertEqual(len(all_pads), 5)
        tokens, positions, rendered = app._render_image_tokens(messages, template)
        self.assertEqual(tokens, baseline)
        self.assertEqual(
            rendered, tokenizer.apply_chat_template(messages, tokenize=False)
        )
        self.assertEqual(positions, [all_pads[1], all_pads[3]])
        # Render markers must not change prompt/cache identity on a repeat.
        self.assertEqual(
            app._render_image_tokens(messages, template), (tokens, positions, rendered)
        )
        prepared = [
            SimpleNamespace(
                tokens=4,
                grid_height=4,
                grid_width=4,
                digest_lo=i,
                digest_hi=0,
                pixels=bytes([i]),
            )
            for i in (1, 2)
        ]
        expanded, spans, pixels = app._expand_image_pads(tokens, prepared, positions)
        expected = list(baseline)
        for position in reversed(positions):
            expected[position : position + 1] = [pad_id] * 4
        self.assertEqual(expanded, expected)
        self.assertEqual(
            [span.offset for span in spans], [positions[0], positions[1] + 3]
        )
        self.assertEqual(pixels, b"\x01\x02")
        with self.assertRaisesRegex(api.APIError, "image count"):
            app._expand_image_pads(tokens, prepared[:1], positions)

    def test_image_size_is_checked_before_pixel_concatenation(self):
        app = self.harness(FakeRuntime(), tokenizer=self.ImagePadTokenizer()).app

        class UnmaterializedPixels:
            def __len__(self):
                return native_wire.ABSOLUTE_MAX_FRAME_PAYLOAD_BYTES

        # No huge buffer: trying to concatenate this sentinel raises TypeError.
        # The size guard must reject using lengths alone, before any copy.
        image = SimpleNamespace(
            tokens=4,
            pixels=UnmaterializedPixels(),
            grid_height=4,
            grid_width=4,
            digest_lo=0,
            digest_hi=0,
        )
        pad = app.tokenizer.convert_tokens_to_ids(api_shapes.IMAGE_PAD_TOKEN)
        with self.assertRaisesRegex(api.APIError, "request size limit"):
            app._expand_image_pads([pad], [image], [0])

    def test_image_count_is_checked_before_decoding(self):
        app = self.harness(FakeRuntime(), tokenizer=self.ImagePadTokenizer()).app
        part = self._image_message()["content"][1]
        limit = native_wire.ProtocolLimits().max_image_spans
        messages = [
            {"role": "user", "content": [part] * limit},
            {"role": "tool", "content": [part]},
        ]
        with mock.patch.object(api.image_input, "decode_data_url") as decode:
            with self.assertRaisesRegex(api.APIError, f"at most {limit} images"):
                app._prepare_images(messages, check_context=False)
            decode.assert_not_called()
        self.assertEqual(app.images.stats()["request_bytes"], 0)
        prepared = app._prepare_images(messages[:1], check_context=False)
        self.assertEqual(len(prepared), limit)
        del prepared
        self.assertEqual(app.images.stats()["request_bytes"], 0)

    def test_image_context_is_checked_before_expanding_placeholders(self):
        app = self.harness(FakeRuntime(), tokenizer=self.ImagePadTokenizer()).app
        image = SimpleNamespace(
            tokens=app.max_context,
            pixels=b"",
            grid_height=16,
            grid_width=32,
            digest_lo=0,
            digest_hi=0,
        )
        pad = app.tokenizer.convert_tokens_to_ids(api_shapes.IMAGE_PAD_TOKEN)
        with self.assertRaisesRegex(api.APIError, "context window"):
            app._expand_image_pads([pad], [image], [0])

    def test_image_preparation_stops_at_aggregate_budget(self):
        app = self.harness(FakeRuntime()).app
        image = SimpleNamespace(tokens=1, pixels=b"x" * 512)
        messages = [
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    }
                ]
                * 3
            }
        ]
        with (
            mock.patch.object(native_wire, "ABSOLUTE_MAX_FRAME_PAYLOAD_BYTES", 1024),
            mock.patch.object(app.images, "prepare", return_value=image) as prepare,
            self.assertRaisesRegex(api.APIError, "request size limit"),
        ):
            app._prepare_images(messages)
        self.assertEqual(prepare.call_count, 2)

    def test_image_request_budget_survives_native_owner_and_recovers(self):
        app = self.harness(
            FakeRuntime(), tokenizer=self.ImagePadTokenizer(), max_context=1024
        ).app
        app.images = api.image_input.ImageCache(request_budget_bytes=256 * 256 * 3)
        job, _, _ = app.prepare(self.body(messages=[self._image_message()]))
        native_request = app.backend._generation_request(job)
        self.assertIs(native_request.image_owner, job.image_owner)
        del job
        with self.assertRaisesRegex(api.APIError, "image memory budget") as failure:
            app.prepare(self.body(messages=[self._image_message()]))
        self.assertEqual(failure.exception.status, 503)
        del native_request
        self.assertEqual(app.images.stats()["request_bytes"], 0)
        job, _, _ = app.prepare(self.body(messages=[self._image_message()]))
        self.assertGreater(app.images.stats()["request_bytes"], 0)
        del job
        self.assertEqual(app.images.stats()["request_bytes"], 0)

    def test_anthropic_and_responses_images_reach_the_same_pipeline(self):
        runtime = FakeRuntime(Plan([[4]]), Plan([[4]]))
        harness = self.harness(runtime, tokenizer=self.ImagePadTokenizer())
        url = self._png_data_url((30, 30, 200))
        media_type, _, data = url.partition(";base64,")
        status, _, _ = harness.request(
            "POST",
            "/v1/messages",
            self.anthropic_body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "before"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type.removeprefix("data:"),
                                    "data": data,
                                },
                            },
                            {"type": "text", "text": "after"},
                        ],
                    }
                ]
            ),
        )
        self.assertEqual(status, 200)
        rendered, _ = harness.tokenizer.templates[-1]
        self.assertEqual(
            [part["type"] for part in rendered[0]["content"]],
            ["text", "image_url", "text"],
        )
        self.assertEqual(len(runtime.requests[0].image_spans), 1)

        status, _, _ = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "look"},
                            {"type": "input_image", "image_url": url},
                        ],
                    }
                ]
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(runtime.requests[1].image_spans), 1)
        self.assertEqual(
            runtime.requests[1].image_spans, runtime.requests[0].image_spans
        )

    def test_tool_results_carry_images_like_user_content(self):
        runtime = FakeRuntime(Plan([[4]]), Plan([[4]]))
        harness = self.harness(runtime, tokenizer=self.ImagePadTokenizer())
        url = self._png_data_url((30, 200, 30))
        media_type, _, data = url.partition(";base64,")
        status, _, _ = harness.request(
            "POST",
            "/v1/messages",
            self.anthropic_body(
                messages=[
                    {"role": "user", "content": "screenshot it"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "screenshot",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [
                                    {"type": "text", "text": "captured"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type.removeprefix(
                                                "data:"
                                            ),
                                            "data": data,
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                ]
            ),
        )
        self.assertEqual(status, 200)
        rendered, _ = harness.tokenizer.templates[-1]
        self.assertEqual(rendered[-1]["role"], "tool")
        self.assertEqual(
            [part["type"] for part in rendered[-1]["content"]], ["text", "image_url"]
        )
        request = runtime.requests[0]
        self.assertEqual(request.prompt_tokens, (101, *([50] * 64), 102))
        (span,) = request.image_spans
        self.assertEqual((span.offset, span.tokens), (1, 64))

        status, _, _ = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(
                messages=[
                    {"role": "user", "content": "screenshot it"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "screenshot", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": [
                            {"type": "text", "text": "captured"},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    },
                ]
            ),
        )
        self.assertEqual(status, 200)
        rendered, _ = harness.tokenizer.templates[-1]
        self.assertEqual(rendered[-1]["role"], "tool")
        self.assertEqual(
            [part["type"] for part in rendered[-1]["content"]], ["text", "image_url"]
        )
        self.assertEqual(runtime.requests[1].image_spans, request.image_spans)

        # Responses tool outputs use the same image pixels and span geometry,
        # including when the tool result is replayed from stored history.
        for stream in (False, True):
            status, _, payload = harness.request(
                "POST",
                "/v1/responses",
                self.responses_body(
                    stream=stream,
                    reasoning={"effort": "none"},
                    input=[
                        {"role": "user", "content": "screenshot it"},
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "screenshot",
                            "arguments": "{}",
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": [
                                {"type": "input_text", "text": "captured"},
                                {"type": "input_image", "image_url": url},
                            ],
                        },
                    ],
                ),
            )
            self.assertEqual(status, 200, payload)
            response = (
                self.response_events(payload)[-1]["response"]
                if stream
                else json.loads(payload)
            )
            rendered, _ = harness.tokenizer.templates[-1]
            self.assertEqual(rendered[-1]["role"], "tool")
            self.assertEqual(rendered[-1]["tool_call_id"], "call_1")
            self.assertEqual(
                [part["type"] for part in rendered[-1]["content"]],
                ["text", "image_url"],
            )
            self.assertEqual(runtime.requests[-1].image_spans, request.image_spans)
            self.assertEqual(runtime.requests[-1].image_pixels, request.image_pixels)
            status, _, payload = harness.request(
                "POST",
                "/v1/responses",
                self.responses_body(
                    previous_response_id=response["id"], input="look again"
                ),
            )
            self.assertEqual(status, 200, payload)
            self.assertEqual(runtime.requests[-1].image_spans, request.image_spans)

    def test_responses_tool_images_reject_invalid_content_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=self.ImagePadTokenizer())
        for content in (
            {"type": "input_image", "image_url": "https://example.com/image.png"},
            {"type": "input_image", "image_url": "data:image/png;base64,@@@"},
            {"type": "input_file", "file_id": "not-supported"},
        ):
            with self.subTest(content=content):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/responses",
                    self.responses_body(
                        input=[
                            {
                                "type": "function_call_output",
                                "call_id": "call_1",
                                "output": [content],
                            }
                        ]
                    ),
                )
                self.assertEqual(status, 400, payload)
        self.assertEqual(runtime.requests, [])

    def test_image_inputs_are_validated_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=self.ImagePadTokenizer())
        cases = (
            (
                [self._image_message("https://example.com/x.png")],
                "only data: image URLs",
            ),
            ([self._image_message("data:image/png;base64,@@@")], "not valid base64"),
            (
                [{"role": "user", "content": [{"type": "video", "video": "x"}]}],
                "video content is not supported",
            ),
            (
                [{"role": "user", "content": [{"type": "image_url", "image_url": 5}]}],
                "invalid image content part",
            ),
            (
                [
                    {
                        "role": "system",
                        "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                    }
                ],
                "only text message content",
            ),
        )
        for messages, message in cases:
            with self.subTest(message=message):
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", self.body(messages=messages)
                )
                self.assertEqual(status, 400)
                self.assertIn(message, json.loads(payload)["error"]["message"])
        self.assertEqual(runtime.requests, [])

        # A template that drops the placeholder cannot carry the image.
        plain = self.harness(runtime)
        status, _, payload = plain.request(
            "POST", "/v1/chat/completions", self.body(messages=[self._image_message()])
        )
        self.assertEqual(status, 400)
        self.assertIn(
            "does not define the image pad", json.loads(payload)["error"]["message"]
        )

    def test_anthropic_messages_nonstream_stream_and_errors(self):
        runtime = FakeRuntime(Plan([[4]]), Plan([[4]]))
        harness = self.harness(runtime)
        status, content_type, payload = harness.request(
            "POST", "/v1/messages", self.anthropic_body()
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(content_type, "application/json")
        message = json.loads(payload)
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(
            message["content"], [{"type": "text", "text": "plain answer\n"}]
        )
        self.assertEqual(message["stop_reason"], "end_turn")
        self.assertEqual(
            message["usage"],
            {"input_tokens": 1, "cache_read_input_tokens": 1, "output_tokens": 1},
        )

        # Standard Anthropic query parameters do not change endpoint routing.
        status, _, payload = harness.request(
            "POST", "/v1/messages?beta=true", self.anthropic_body()
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["type"], "message")

        status, content_type, payload = harness.request(
            "POST", "/v1/messages", self.anthropic_body(stream=True)
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(content_type, "text/event-stream")
        events = self.response_events(payload)
        self.assertEqual(events[0]["type"], "message_start")
        self.assertEqual(events[-2]["type"], "message_delta")
        self.assertEqual(events[-2]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(events[-1], {"type": "message_stop"})
        text = "".join(
            event["delta"]["text"]
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "text_delta"
        )
        self.assertEqual(text, "plain answer\n")

        body = self.anthropic_body()
        del body["max_tokens"]
        status, _, payload = harness.request("POST", "/v1/messages", body)
        self.assertEqual(status, 400)
        error = json.loads(payload)
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["error"]["type"], "invalid_request_error")

    def test_anthropic_count_tokens_matches_generation_without_admission(self):
        class InputTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                prefix = super().apply_chat_template(messages, **kwargs)
                return json.dumps([messages, kwargs], sort_keys=True) + prefix

            def __call__(self, text, **kwargs):
                return {"input_ids": list(text.encode())}

        runtime = FakeRuntime()
        factory = FakeConstraintFactory()
        harness = self.harness(
            runtime,
            tokenizer=InputTokenizer(),
            constraint_factory=factory,
            max_context=32768,
        )
        tools = [
            {
                "name": "read_file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ]
        history = [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Inspect the file."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "read_file",
                        "input": {"path": "hello.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "hello"},
                    {"type": "text", "text": "Summarize."},
                ],
            },
        ]
        cases = [
            {},
            {
                "system": [{"type": "text", "text": "Be concise."}],
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            },
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
            {
                "tools": tools,
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                        },
                    }
                },
            },
            *(
                {"messages": history, "tools": tools, "tool_choice": choice}
                for choice in (
                    {"type": "none"},
                    {"type": "auto"},
                    {"type": "any"},
                    {"type": "tool", "name": "read_file"},
                )
            ),
        ]
        counts = []
        for index, extra in enumerate(cases):
            with self.subTest(extra=extra):
                body = self.anthropic_body(**extra)
                del body["max_tokens"]
                with (
                    mock.patch.object(
                        request_frontend, "Job", side_effect=AssertionError("Job")
                    ),
                    mock.patch.object(
                        factory, "create", side_effect=AssertionError("grammar")
                    ),
                ):
                    for suffix in ("", "?beta=true"):
                        status, _, payload = harness.request(
                            "POST", "/v1/messages/count_tokens" + suffix, body
                        )
                        self.assertEqual(status, 200, payload)
                        counted = json.loads(payload)
                        self.assertEqual(set(counted), {"input_tokens"})
                job, *_ = harness.app.prepare(
                    api.anthropic_to_chat_body({**body, "max_tokens": 8})
                )
                self.assertEqual(counted["input_tokens"], len(job.prompt_tokens))
                self.assertEqual(job.request_id, index + 1)
                counts.append(counted["input_tokens"])
        self.assertGreater(len(set(counts)), 2)
        self.assertEqual(runtime.requests, [])
        self._wait_for_http_active(harness.server.requests, 0)

    def test_anthropic_count_tokens_can_measure_over_context_images(self):
        harness = self.harness(
            FakeRuntime(), tokenizer=self.ImagePadTokenizer(), max_context=16
        )
        data = self._png_data_url().partition(";base64,")[2]
        image = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": data,
            },
        }
        body = self.anthropic_body(
            messages=[{"role": "user", "content": [image, image]}]
        )
        del body["max_tokens"]
        with mock.patch.object(
            harness.app, "_expand_image_pads", side_effect=AssertionError("expansion")
        ):
            status, _, payload = harness.request(
                "POST", "/v1/messages/count_tokens", body
            )
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload), {"input_tokens": 130})
        self.assertEqual(harness.app.images.stats()["request_bytes"], 0)
        with self.assertRaisesRegex(api.APIError, "context window"):
            harness.app.prepare(api.anthropic_to_chat_body({**body, "max_tokens": 1}))
        harness.app.max_context = 256
        job, *_ = harness.app.prepare(
            api.anthropic_to_chat_body({**body, "max_tokens": 1})
        )
        self.assertEqual(len(job.prompt_tokens), 130)
        del job
        self.assertEqual(harness.app.images.stats()["request_bytes"], 0)
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_anthropic_count_tokens_validates_input_and_releases_capacity(self):
        harness = self.harness(FakeRuntime())
        valid = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        for body, expected in (
            ([], 400),
            ({}, 400),
            ({"messages": valid["messages"]}, 400),
            ({"model": "test-model"}, 400),
            ({**valid, "messages": []}, 400),
            ({**valid, "model": "other-model"}, 404),
            ({**valid, "tools": "bad"}, 400),
            ({**valid, "thinking": {"type": "bad"}}, 400),
            (
                {
                    **valid,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "invalid",
                                    },
                                }
                            ],
                        }
                    ],
                },
                400,
            ),
        ):
            with self.subTest(body=body):
                status, _, payload = harness.request(
                    "POST", "/v1/messages/count_tokens?beta=true", body
                )
                self.assertEqual(status, expected, payload)
                self.assertEqual(json.loads(payload)["type"], "error")
                self._wait_for_http_active(harness.server.requests, 0)
                self.assertEqual(harness.app.preparation_active, 0)
        with mock.patch.object(
            FakeTokenizer, "__call__", return_value={"input_ids": list(range(256))}
        ):
            status, _, payload = harness.request(
                "POST", "/v1/messages/count_tokens", valid
            )
        self.assertEqual((status, json.loads(payload)), (200, {"input_tokens": 256}))
        for _ in range(harness.app.preparation_capacity):
            harness.app.preparation_slots.acquire()
        try:
            status, _, payload = harness.request(
                "POST", "/v1/messages/count_tokens", {**valid, "timeout": 0.02}
            )
            self.assertEqual(status, 504, payload)
        finally:
            for _ in range(harness.app.preparation_capacity):
                harness.app.preparation_slots.release()
        self._wait_for_http_active(harness.server.requests, 0)
        self.assertEqual(harness.app.preparation_waiting, 0)
        self.assertEqual(harness.backend.runtime.requests, [])

    def test_anthropic_tool_history_and_choice_use_one_chat_pipeline(self):
        translated = api.anthropic_to_chat_body(
            self.anthropic_body(
                system=[{"type": "text", "text": "be exact"}],
                messages=[
                    {"role": "user", "content": "read it"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "read_file",
                                "input": {"path": "a.py"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [{"type": "text", "text": "contents"}],
                            },
                            {"type": "text", "text": "summarize"},
                        ],
                    },
                ],
                tools=[
                    {
                        "name": "read_file",
                        "description": "Read a local file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                tool_choice={
                    "type": "tool",
                    "name": "read_file",
                    "disable_parallel_tool_use": True,
                },
            )
        )
        self.assertEqual(
            translated["messages"][0], {"role": "system", "content": "be exact"}
        )
        self.assertEqual(translated["messages"][2]["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(translated["messages"][3]["role"], "tool")
        self.assertEqual(translated["messages"][4]["content"], "summarize")
        self.assertEqual(
            translated["tool_choice"],
            {"type": "function", "function": {"name": "read_file"}},
        )
        self.assertFalse(translated["parallel_tool_calls"])

    def test_anthropic_adaptive_thinking_maps_reasoning_effort(self):
        translated = api.anthropic_to_chat_body(
            self.anthropic_body(
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
            )
        )
        self.assertEqual(translated["reasoning_effort"], "high")
        translated = api.anthropic_to_chat_body(
            self.anthropic_body(
                thinking={"type": "adaptive"},
                output_config={"effort": "low"},
            )
        )
        self.assertEqual(translated["reasoning_effort"], "low")

    def test_anthropic_adaptive_effort_rejects_non_strings_before_admission(self):
        runtime = FakeRuntime(Plan(batches=[(3,)]))
        harness = self.harness(runtime)
        for effort in ([], {}, None, True, 1):
            with self.subTest(effort=effort):
                status, _, raw = harness.request(
                    "POST",
                    "/v1/messages",
                    self.anthropic_body(
                        thinking={"type": "adaptive"},
                        output_config={"effort": effort},
                    ),
                )
                self.assertEqual(status, 400)
                self.assertEqual(
                    json.loads(raw)["error"]["type"], "invalid_request_error"
                )
        self.assertEqual(runtime.requests, [])
        status, _, _ = harness.request(
            "POST",
            "/v1/messages",
            self.anthropic_body(
                thinking={"type": "adaptive"}, output_config={"effort": "low"}
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(runtime.requests), 1)

    def test_anthropic_strips_nonsemantic_billing_system_block(self):
        def translated(metadata):
            return api.anthropic_to_chat_body(
                self.anthropic_body(
                    system=[
                        {
                            "type": "text",
                            "text": f"x-anthropic-billing-header: {metadata}",
                        },
                        {
                            "type": "text",
                            "text": "stable agent system",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "dynamic status"},
                    ]
                )
            )

        first = translated("one")
        self.assertEqual(first, translated("two"))
        self.assertEqual(
            first["messages"][0],
            {"role": "system", "content": "stable agent systemdynamic status"},
        )

    def test_anthropic_preserves_inline_system_messages(self):
        translated = api.anthropic_to_chat_body(
            self.anthropic_body(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "system", "content": "dynamic system update"},
                    {"role": "assistant", "content": "acknowledged"},
                ]
            )
        )
        self.assertEqual(
            translated["messages"],
            [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": "dynamic system update"},
                {"role": "assistant", "content": "acknowledged"},
            ],
        )

        body = self.anthropic_body(messages=[{"content": "missing role"}])
        with self.assertRaisesRegex(
            api.APIError, "require user/assistant/system roles"
        ):
            api.anthropic_to_chat_body(body)

    def test_health_stays_live_when_native_is_not_ready(self):
        runtime = FakeRuntime()
        runtime.closed = True
        runtime.ready = False
        harness = self.harness(runtime)
        status, _, payload = harness.request("GET", "/health")
        self.assertEqual((status, json.loads(payload)), (200, {"status": "ok"}))
        status, _, payload = harness.request("GET", "/ready")
        self.assertEqual(
            (status, json.loads(payload)), (503, {"status": "unavailable"})
        )
        status, _, payload = harness.request("GET", "/status")
        self.assertFalse(json.loads(payload)["ready"])

    def test_status_identifies_the_http_instance_independently_of_readiness(self):
        first = self.harness(FakeRuntime())
        second = self.harness(FakeRuntime())
        before = first.server.status()["instance"]
        first.backend.runtime.ready = False
        status, _, payload = first.request("GET", "/status")
        snapshot = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["instance"], before)
        self.assertNotEqual(before["id"], second.server.status()["instance"]["id"])
        self.assertEqual(before["pid"], os.getpid())
        self.assertEqual(before["model"], "test-model")
        self.assertEqual((before["host"], before["port"]), first.server.server_address)
        self.assertGreater(before["started_at"], 0)
        self.assertLessEqual(before["started_at"], time.time())

    def test_http_error_logs_method_and_path_without_query_or_body(self):
        harness = self.harness(FakeRuntime())
        for method in ("GET", "POST", "DELETE"):
            with (
                self.subTest(method=method),
                mock.patch.object(api, "print_status") as output,
            ):
                status, _, _ = harness.request(
                    method, "/missing?token=private-query", {"private-body": "secret"}
                )
                self.assertEqual(status, 404)
                output.assert_called_once_with(
                    f"Error · not_found · {method} /missing", error=True
                )
        with mock.patch.object(api, "print_status") as output:
            status, _, _ = harness.request("POST", "/v1/messages?beta=true", {})
            self.assertEqual(status, 400)
            output.assert_called_once_with(
                "Error · invalid_request_error · POST /v1/messages", error=True
            )

    def test_streaming_sse_and_usage(self):
        harness = self.harness(FakeRuntime(Plan([[1], [2], [3]], reason="length")))
        status, content_type, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(stream=True, stream_options={"include_usage": True}),
        )
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/event-stream")
        events = [
            line[6:]
            for line in payload.decode().splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(event) for event in events[:-1]]
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
        self.assertEqual(deltas[0], {"role": "assistant", "content": ""})
        self.assertIn({"reasoning_content": "because "}, deltas)
        self.assertIn({"content": "answer\n"}, deltas)
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 3)
        self.assertAlmostEqual(
            chunks[-1]["metrics"]["request_latency"]["stream_tokens_per_second"],
            1000.0,
        )

    def test_stop_sequence_cancels_native_without_emitting_the_marker(self):
        runtime = FakeRuntime(Plan([[14], [15], [4]], reason="length", delay=0.01))
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(stop="second", reasoning_effort="none"),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["content"], "first ")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"]["completion_tokens"], 2)
        self.assertEqual(runtime.cancel_count, 1)

    def test_anthropic_cache_usage_and_matched_stop_sequence(self):
        for stream in (False, True):
            for matched_tokens in (0, 1):
                with self.subTest(stream=stream, matched_tokens=matched_tokens):
                    runtime = FakeRuntime(
                        Plan(
                            [[14], [15], [4]], delay=0.01, matched_tokens=matched_tokens
                        )
                    )
                    harness = self.harness(runtime)
                    status, _, payload = harness.request(
                        "POST",
                        "/v1/messages",
                        self.anthropic_body(
                            stream=stream, stop_sequences=["unused", "second"]
                        ),
                    )
                    self.assertEqual(status, 200, payload)
                    if stream:
                        events = self.response_events(payload)
                        usage = {
                            **events[0]["message"]["usage"],
                            **events[-2]["usage"],
                        }
                        finish = events[-2]["delta"]
                        content = "".join(
                            event["delta"]["text"]
                            for event in events
                            if event["type"] == "content_block_delta"
                        )
                    else:
                        finish = json.loads(payload)
                        usage = finish["usage"]
                        content = finish["content"][0]["text"]
                    self.assertEqual(
                        usage,
                        {
                            "input_tokens": 2 - matched_tokens,
                            "cache_read_input_tokens": matched_tokens,
                            "output_tokens": 2,
                        },
                    )
                    self.assertEqual(finish["stop_reason"], "stop_sequence")
                    self.assertEqual(finish["stop_sequence"], "second")
                    self.assertEqual(content, "first ")
                    self.assertEqual(runtime.cancel_count, 1)

    def test_usage_counts_only_tokens_before_the_thinking_delimiter(self):
        harness = self.harness(FakeRuntime(Plan([[1], [26], [27]])))
        with self.openai_client(harness) as client:
            response = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
            )
        self.assertEqual(response.choices[0].message.content, "answer\n")
        self.assertEqual(
            response.usage.completion_tokens_details.reasoning_tokens,
            1,
        )
        self.assertEqual(response.usage.prompt_tokens_details.cached_tokens, 1)

    def test_stream_options_null_uses_defaults(self):
        harness = self.harness(FakeRuntime(Plan([[3]])))
        status, content_type, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(
                stream=True,
                stream_options=None,
                reasoning_effort="none",
            ),
        )
        self.assertEqual((status, content_type), (200, "text/event-stream"))
        events = [
            line[6:]
            for line in payload.decode().splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(event) for event in events[:-1]]
        self.assertEqual(
            chunks[0]["choices"][0]["delta"],
            {"role": "assistant", "content": ""},
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_tool_capable_chat_streams_text_before_done_without_repeating(self):
        plan = Plan([[14], [15]], delay=0.01, after_terminal=True)
        harness = self.harness(FakeRuntime(plan))
        tools = [{"type": "function", "function": {"name": "weather"}}]
        started = time.monotonic()
        connection, response = harness.open_stream(
            "/v1/chat/completions",
            self.body(
                stream=True,
                tools=tools,
                reasoning_effort="none",
            ),
        )
        parts = []
        try:
            self.assertEqual(response.status, 200)
            while not parts:
                data = self.next_sse_data(response)
                self.assertIsNotNone(data)
                self.assertNotEqual(data, "[DONE]")
                chunk = json.loads(data)
                if chunk["choices"]:
                    text = chunk["choices"][0]["delta"].get("content")
                    if text:
                        parts.append(text)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(plan.terminal.wait(1))
            self.assertFalse(plan.terminal_release.is_set())
            plan.terminal_release.set()
            while True:
                data = self.next_sse_data(response)
                self.assertIsNotNone(data)
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk["choices"]:
                    text = chunk["choices"][0]["delta"].get("content")
                    if text:
                        parts.append(text)
        finally:
            plan.terminal_release.set()
            response.close()
            connection.close()
        self.assertEqual(parts, ["first ", "second\n"])

    def test_tool_capable_responses_streams_text_before_done_without_repeating(self):
        plan = Plan([[14], [15]], delay=0.01, after_terminal=True)
        harness = self.harness(FakeRuntime(plan))
        tools = [{"type": "function", "name": "weather"}]
        started = time.monotonic()
        connection, response = harness.open_stream(
            "/v1/responses",
            self.responses_body(
                stream=True,
                tools=tools,
                reasoning={"effort": "none"},
            ),
        )
        parts, completed = [], None
        try:
            self.assertEqual(response.status, 200)
            while not parts:
                data = self.next_sse_data(response)
                self.assertIsNotNone(data)
                event = json.loads(data)
                if event["type"] == "response.output_text.delta":
                    parts.append(event["delta"])
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(plan.terminal.wait(1))
            self.assertFalse(plan.terminal_release.is_set())
            plan.terminal_release.set()
            while completed is None:
                data = self.next_sse_data(response)
                self.assertIsNotNone(data)
                event = json.loads(data)
                if event["type"] == "response.output_text.delta":
                    parts.append(event["delta"])
                if event["type"] == "response.completed":
                    completed = event["response"]
        finally:
            plan.terminal_release.set()
            response.close()
            connection.close()
        self.assertEqual(parts, ["first ", "second\n"])
        message = next(
            item for item in completed["output"] if item["type"] == "message"
        )
        self.assertEqual(message["content"][0]["text"], "first second\n")

    def test_tps_omits_prefill_token_and_invalid_timings(self):
        result = backend_api.NativeResult(
            "stop", 1, 1, 1.0, 2.0, 3.0, 1, first_token_batch_tokens=1
        )
        metrics = api.metrics_dict(result)
        self.assertNotIn("stream_tokens_per_second", metrics["request_latency"])
        self.assertEqual(metrics["prefill"]["tokens"], 1)
        result.completion_tokens = 3
        for interval in (0.0, math.inf, 5e-324):
            result.first_token_to_done_ms = interval
            self.assertNotIn(
                "stream_tokens_per_second", api.metrics_dict(result)["request_latency"]
            )

    def test_stream_rate_excludes_the_whole_first_speculative_batch(self):
        for batches, remaining in (([[14, 15]], 0), ([[14, 15], [14, 15, 4]], 3)):
            with self.subTest(batches=batches):
                harness = self.harness(FakeRuntime(Plan(batches)))
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", self.body(reasoning_effort="none")
                )
                self.assertEqual(status, 200)
                metrics = json.loads(payload)["metrics"]
                self.assertEqual(metrics["decode"]["tokens"], remaining)
                latency = metrics["request_latency"]
                if remaining:
                    self.assertEqual(latency["stream_tokens_per_second"], 1500.0)
                else:
                    self.assertNotIn("stream_tokens_per_second", latency)

    def test_request_record_contains_metrics_but_console_omits_prompt_and_id(self):
        records = []
        harness = self.harness(FakeRuntime(), request_logger=records.append)
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", self.body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["event"], "request")
        self.assertEqual(record["outcome"], "stop")
        self.assertNotIn("queue_ms", record)
        self.assertNotIn("queue_ms", record["metrics"])
        self.assertNotIn("hello", json.dumps(record))

        with mock.patch("builtins.print") as output:
            diagnostics.print_request(record)
        line = output.call_args.args[0]
        self.assertRegex(line, r"^\d{2}:\d{2}:\d{2} Done · input ")
        self.assertNotIn("request_id", line)
        self.assertNotIn("hello", line)
        self.assertNotIn("\n", line)
        self.assertTrue(output.call_args.kwargs["flush"])

    def test_console_request_summary(self):
        record = {
            "request_id": 123,
            "outcome": "stop",
            "prompt_tokens": 10240,
            "completion_tokens": 320,
            "metrics": {
                "cache": {"matched_tokens": 8192},
                "request_latency": {
                    "ttft_ms": 800,
                    "stream_tokens_per_second": 85,
                },
            },
        }
        with (
            mock.patch.object(api.time, "strftime", return_value="14:32:08"),
            mock.patch("builtins.print") as output,
        ):
            diagnostics.print_request(record)
        self.assertEqual(
            output.call_args.args[0],
            "14:32:08 Done · input 10,240 · cached 8,192 · output 320"
            " · TTFT 0.8s · 85.0 tok/s",
        )

    def test_console_shows_the_tool_block_signature(self):
        record = {
            "outcome": "stop",
            "prompt_tokens": 33_799,
            "completion_tokens": 12,
            "tools": {"count": 27, "signature": "1a2b3c4d"},
            "metrics": {"cache": {"matched_tokens": 6_656}},
        }
        with (
            mock.patch.object(api.time, "strftime", return_value="14:32:08"),
            mock.patch("builtins.print") as output,
        ):
            diagnostics.print_request(record)
        self.assertEqual(
            output.call_args.args[0],
            "14:32:08 Done · input 33,799 · cached 6,656 · output 12"
            " · tools 27·1a2b3c4d",
        )

    def test_console_cancellation_without_tokens_or_latency(self):
        with mock.patch("builtins.print") as output:
            diagnostics.print_request({"outcome": "cancelled", "prompt_tokens": 32})
        line = output.call_args.args[0]
        self.assertIn("Cancelled · input 32 · cached 0 · output 0", line)
        self.assertNotIn("TTFT", line)
        self.assertNotIn("tok/s", line)

    def test_console_error_omits_private_details(self):
        with mock.patch("builtins.print") as output:
            diagnostics.print_request(
                {
                    "outcome": "error",
                    "error_code": "context_length_exceeded",
                    "error_message": "private prompt\n用户输入",
                    "request_id": 123,
                }
            )
        line = output.call_args.args[0]
        self.assertRegex(line, r"^\d{2}:\d{2}:\d{2} Error · context_length_exceeded$")
        self.assertIs(output.call_args.kwargs["file"], api.sys.stderr)

    def test_server_requires_explicit_model_and_paths(self):
        model = "community/custom-splash"
        package = api.ROOT / "install/models" / model
        required = [
            str(package / "target"),
            str(package / "draft"),
            "--tokenizer",
            str(package / "tokenizer"),
            "--model",
            model,
        ]
        for arguments in (
            [],
            required[:-2],
            [*required[:-1], "qwen3.8-27b"],
            [*required[:-1], "Qwen3.8-27B"],
            [*required[:-1], "https://huggingface.co/community/model"],
            [*required[:-1], "community/../model"],
        ):
            with (
                self.subTest(arguments=arguments),
                mock.patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                api.parse_args(arguments)
        args = api.parse_args(required)
        self.assertEqual(Path(args.target), package / "target")
        self.assertEqual(Path(args.draft), package / "draft")
        self.assertEqual(Path(args.tokenizer), package / "tokenizer")
        self.assertIsNone(args.max_context)
        self.assertIsNone(args.max_memory)
        self.assertEqual(
            api.parse_args([*required, "--max-context", "262144"]).max_context, 262144
        )
        self.assertEqual(
            api.parse_args([*required, "--max-memory", "32G"]).max_memory, 32 * 1024**3
        )
        self.assertEqual(args.max_new_tokens, 32768)
        self.assertEqual(args.request_timeout, 1800)
        self.assertEqual(args.model, model)
        self.assertEqual(Path(args.binary).name, "splash")
        tokenizer = FakeTokenizer()
        backend = backend_api.NativeBackend(FakeRuntime(), tokenizer)
        self.addCleanup(backend.close)
        app = request_frontend.Frontend(
            tokenizer, backend, "test-model", 40000, 32768, 1, 2
        )
        self.assertEqual(app.prepare(self.body())[0].max_new_tokens, 32768)
        app = request_frontend.Frontend(
            tokenizer, backend, "test-model", 10, 32768, 1, 2
        )
        self.assertEqual(app.prepare(self.body())[0].max_new_tokens, 8)
        with mock.patch("sys.stderr"):
            for option, value in (
                ("--max-context", "0"),
                ("--max-context", "262145"),
                ("--max-context", "not-a-number"),
                ("--max-memory", "0"),
                ("--max-memory", "not-a-number"),
                ("--max-new-tokens", "0"),
                ("--request-timeout", "0"),
                ("--request-timeout", "nan"),
                ("--request-timeout", "inf"),
                ("--queue-size", "0"),
                ("--port", "-1"),
                ("--port", "65536"),
            ):
                with self.assertRaises(SystemExit):
                    api.parse_args([*required, option, value])

    def test_public_ids_are_stable_and_unique_across_app_instances(self):
        tokenizer = FakeTokenizer()
        first_backend = backend_api.NativeBackend(FakeRuntime(), tokenizer)
        second_backend = backend_api.NativeBackend(FakeRuntime(), tokenizer)
        self.addCleanup(first_backend.close)
        self.addCleanup(second_backend.close)
        with mock.patch.object(
            api.secrets, "token_hex", side_effect=("boot_a", "boot_b")
        ):
            first = request_frontend.Frontend(
                tokenizer, first_backend, "test-model", 128, 16, 1, 2
            )
            second = request_frontend.Frontend(
                tokenizer, second_backend, "test-model", 128, 16, 1, 2
            )
        first_job, _, _ = first.prepare(self.body(seed=1))
        second_job, _, _ = second.prepare(self.body(seed=1))
        self.assertEqual((first_job.request_id, second_job.request_id), (1, 1))
        self.assertNotEqual(first_job.public_id, second_job.public_id)
        result = backend_api.NativeResult("stop", 2, 1, 1, 1, 1)
        first_id = api.completion_response(
            "test-model",
            first_job,
            result,
            {"role": "assistant", "content": "x"},
            False,
        )["id"]
        second_id = api.completion_response(
            "test-model",
            second_job,
            result,
            {"role": "assistant", "content": "x"},
            False,
        )["id"]
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(
            api.stream_chunk("test-model", first_job.public_id, 1, {})["id"], first_id
        )
        first_job.created_at = 123
        with mock.patch.object(api.time, "time", return_value=999):
            response = api.responses_response(
                "test-model", first_job, "in_progress", []
            )
        self.assertEqual(response["created_at"], 123)

    def test_sigterm_uses_the_normal_main_cleanup_path(self):
        args = SimpleNamespace(
            target="target",
            draft="draft",
            tokenizer="tokenizer",
            model="test-model",
            max_context=None,
            max_memory=None,
            max_image_pixels=api.image_input.MAX_PIXELS,
            max_new_tokens=16,
            request_timeout=2,
            queue_size=1,
            host="127.0.0.1",
            allowed_host=[],
            api_key=None,
            no_webui=False,
            port=0,
            binary="splash",
        )
        runtime = mock.Mock()
        runtime.readiness = native_wire.ReadyEvent(
            engine_instance_id=1,
            max_concurrent_requests=4,
            max_context_tokens=262144,
            feature_bits=15,
        )
        backend = mock.Mock()
        server = mock.Mock(server_port=8000)
        tokenizer = object()
        handlers = {}
        order = []

        def install(signum, handler):
            if handler is api._interrupt:
                handlers[signum] = handler
                return signal.SIG_DFL
            handlers[signum] = handler
            self.assertIn(signum, (signal.SIGTERM, signal.SIGINT))
            self.assertEqual(handler, signal.SIG_IGN)

        def serve():
            self.assertEqual(set(handlers), {signal.SIGTERM, signal.SIGINT})
            handlers[signal.SIGTERM](signal.SIGTERM, None)

        def bind(*_, **__):
            return server

        server.server_bind.side_effect = lambda: order.append("bind")

        runtime.wait_ready.side_effect = lambda: order.append("runtime") or True
        server.serve_forever.side_effect = serve
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", return_value=runtime
            ) as runtime_type,
            mock.patch.object(
                api, "NativeBackend", return_value=backend
            ) as backend_type,
            mock.patch.object(api, "ConstraintFactory", return_value=object()),
            mock.patch.object(api, "Frontend", return_value=object()) as app_type,
            mock.patch.object(api, "FrontendServer", side_effect=bind),
            mock.patch.object(api.signal, "signal", side_effect=install),
            mock.patch("builtins.print"),
        ):
            api.main()
        self.assertEqual(
            handlers, {signal.SIGTERM: signal.SIG_IGN, signal.SIGINT: signal.SIG_IGN}
        )
        runtime_type.assert_called_once_with(
            [
                "splash",
                "serve-native",
                "target",
                "draft",
                "auto",
                "auto",
            ],
            startup_timeout=api.NATIVE_START_TIMEOUT,
            pending_limit=1,
            eager_start=False,
        )
        backend_type.assert_called_once_with(
            runtime, tokenizer, request_logger=diagnostics.print_request
        )
        self.assertEqual(app_type.call_args.args[3], 262144)
        self.assertEqual(app_type.call_args.args[6], 4)
        runtime.wait_ready.assert_called_once_with()
        server.serve_forever.assert_called_once()
        server.server_close.assert_called_once()
        backend.close.assert_called_once()
        server.server_bind.assert_called_once_with()
        server.server_activate.assert_called_once_with()
        self.assertEqual(order, ["bind", "runtime"])

    def test_main_cleans_up_when_native_startup_fails_after_reserved_bind(self):
        args = SimpleNamespace(
            target="target",
            draft="draft",
            tokenizer="tokenizer",
            model="test-model",
            max_context=128,
            max_memory=32 * 1024**3,
            max_new_tokens=16,
            request_timeout=2,
            queue_size=1,
            host="127.0.0.1",
            allowed_host=[],
            api_key=None,
            no_webui=False,
            port=0,
            binary="splash",
        )
        runtime = mock.Mock()
        runtime.wait_ready.side_effect = api.engine_runtime.EngineUnhealthy("late")
        backend = mock.Mock()
        server = mock.Mock()
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(
                api.AutoTokenizer, "from_pretrained", return_value=object()
            ),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", return_value=runtime
            ),
            mock.patch.object(api, "NativeBackend", return_value=backend),
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(api.signal, "signal"),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            api.main()
        server.server_bind.assert_called_once_with()
        server.server_activate.assert_not_called()
        server.server_close.assert_called_once_with()
        backend.close.assert_called_once()
        self.assertIn("Error · late", stderr.getvalue())

    def test_main_rejects_port_conflict_before_loading_or_starting_native(self):
        args = SimpleNamespace(
            target="target",
            draft="draft",
            tokenizer="tokenizer",
            model="test-model",
            max_context=None,
            max_memory=None,
            max_image_pixels=api.image_input.MAX_PIXELS,
            max_new_tokens=16,
            request_timeout=2,
            queue_size=1,
            host="127.0.0.1",
            allowed_host=[],
            api_key=None,
            no_webui=False,
            port=8000,
            binary="splash",
        )
        server = mock.Mock()
        server.server_bind.side_effect = OSError(48, "Address already in use")
        with (
            mock.patch.object(api, "parse_args", return_value=args),
            mock.patch.object(api, "FrontendServer", return_value=server),
            mock.patch.object(api.AutoTokenizer, "from_pretrained") as tokenizer,
            mock.patch.object(api.engine_runtime, "MultiplexedRuntime") as runtime,
            mock.patch.object(api.signal, "signal"),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            api.main()
        tokenizer.assert_not_called()
        runtime.assert_not_called()
        server.server_activate.assert_not_called()
        server.server_close.assert_called_once_with()
        self.assertIn("Address already in use", stderr.getvalue())

    def test_tools_history_choice_and_output(self):
        runtime = FakeRuntime(Plan([[5]]))
        harness = self.harness(runtime)
        messages = [
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"Rome"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "old", "content": "sunny"},
            {"role": "user", "content": "again"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            },
            {"type": "function", "function": {"name": "time", "parameters": {}}},
        ]
        choice = {"type": "function", "function": {"name": "weather"}}
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(
                messages=messages,
                tools=tools,
                tool_choice=choice,
                reasoning_effort="none",
            ),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
        call = response["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(
            call["function"], {"name": "weather", "arguments": '{"city":"Paris"}'}
        )
        rendered_messages, kwargs = harness.tokenizer.templates[-1]
        self.assertEqual(
            rendered_messages[0], {"role": "system", "content": "Be concise."}
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in kwargs["tools"]], ["weather", "time"]
        )
        self.assertEqual(
            rendered_messages[2]["tool_calls"][0]["function"]["arguments"],
            {"city": "Rome"},
        )
        self.assertFalse(kwargs["enable_thinking"])

    def test_streaming_tool_call_has_index(self):
        harness = self.harness(FakeRuntime(Plan([[5]])))
        tools = [{"type": "function", "function": {"name": "weather"}}]
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(
                stream=True,
                tools=tools,
                reasoning_effort="none",
            ),
        )
        self.assertEqual(status, 200)
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        tool_deltas = [
            call
            for chunk in chunks
            for call in chunk["choices"][0]["delta"].get("tool_calls", [])
        ]
        header = tool_deltas[0]
        self.assertEqual(header["index"], 0)
        self.assertTrue(header["id"].startswith("call_"))
        self.assertEqual(header["type"], "function")
        self.assertEqual(header["function"], {"name": "weather"})
        self.assertEqual(
            "".join(
                delta["function"].get("arguments", "") for delta in tool_deltas[1:]
            ),
            '{"city":"Paris"}',
        )

    def test_streaming_tool_call_arrives_before_native_done(self):
        plan = Plan([[13]], reason="length", after_terminal=True)
        harness = self.harness(FakeRuntime(plan))
        with mock.patch.object(backend_api, "CallbackStreamer", PassthroughStreamer):
            connection, response = harness.open_stream(
                "/v1/chat/completions",
                self.body(
                    stream=True,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                    "additionalProperties": False,
                                },
                            },
                        }
                    ],
                    reasoning_effort="none",
                ),
            )
            try:
                role = json.loads(self.next_sse_data(response))
                self.assertEqual(role["choices"][0]["delta"]["role"], "assistant")
                header = json.loads(self.next_sse_data(response))
                call = header["choices"][0]["delta"]["tool_calls"][0]
                self.assertEqual(call["function"]["name"], "weather")
                arguments = ""
                while "Par" not in arguments:
                    chunk = json.loads(self.next_sse_data(response))
                    delta = chunk["choices"][0]["delta"]["tool_calls"][0]
                    arguments += delta["function"].get("arguments", "")
                self.assertEqual(arguments, '{"city":"Par')
                self.assertTrue(plan.terminal.wait(1))
                self.assertFalse(plan.terminal_release.is_set())
            finally:
                plan.terminal_release.set()
                response.read()
                connection.close()

    def test_stream_sends_keepalive_while_native_is_idle(self):
        plan = Plan([[4]], block=True)
        harness = self.harness(FakeRuntime(plan))
        with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.02):
            connection, response = harness.open_stream(
                "/v1/chat/completions",
                self.body(stream=True, reasoning_effort="none"),
            )
            try:
                role = json.loads(self.next_sse_data(response))
                self.assertEqual(role["choices"][0]["delta"]["role"], "assistant")
                self.assertEqual(
                    response.readline().decode().rstrip("\r\n"),
                    ": splash-keepalive",
                )
            finally:
                plan.release.set()
                response.read()
                connection.close()

    def test_responses_stream_heartbeats_before_native_start(self):
        plan = Plan([[4]], before_start=True)
        harness = self.harness(FakeRuntime(plan))

        def next_event(response):
            payload = json.loads(self.next_sse_data(response))
            return payload["type"], payload

        with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.02):
            connection, response = harness.open_stream(
                "/v1/responses",
                self.responses_body(stream=True, reasoning={"effort": "none"}),
            )
            try:
                created, _ = next_event(response)
                initial, _ = next_event(response)
                heartbeat, payload = next_event(response)
                self.assertEqual(created, "response.created")
                self.assertEqual(initial, "response.in_progress")
                self.assertEqual(heartbeat, "response.in_progress")
                self.assertEqual(payload["response"]["status"], "in_progress")
                self.assertFalse(plan.started.is_set())
            finally:
                plan.start_release.set()
                response.read()
                connection.close()

    def test_responses_stream_heartbeats_after_start_until_first_output(self):
        handler = object.__new__(api.FrontendHandler)
        handler.wfile = io.BytesIO()
        handler._last_sse_write = 0.0
        handler._response_started = False
        handler._client_disconnected = lambda: False
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        app = SimpleNamespace(
            model="test-model",
            backend=SimpleNamespace(cancel=mock.Mock()),
            persist_response=mock.Mock(),
        )
        handler.server = SimpleNamespace(app=app)
        job = backend_api.Job(
            request_id=1,
            prompt_tokens=[101, 102],
            max_new_tokens=16,
            seed=0,
            temperature=0,
            top_p=1,
            top_k=0,
            deadline=100,
            public_id="prefill-heartbeat",
        )
        clock = [0.0]
        events = iter(
            [
                ("start", "miss"),
                None,
                None,
                ("text", "answer"),
                None,
                None,
                ("done", backend_api.NativeResult("stop", 2, 1, 1, 1, 1)),
            ]
        )

        def next_event(**_kwargs):
            event = next(events)
            if event is None:
                clock[0] += api.SSE_KEEPALIVE_SECONDS + 0.1
                raise api.queue.Empty
            return event

        job.events = SimpleNamespace(get=next_event)
        with mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]):
            handler._responses_stream(job, False, False)

        raw = handler.wfile.getvalue()
        stream = io.BytesIO(raw)
        parsed = []
        while data := self.next_sse_data(stream):
            parsed.append(json.loads(data))
        kinds = [event["type"] for event in parsed]
        first_output = kinds.index("response.output_item.added")
        self.assertEqual(
            kinds[:first_output],
            ["response.created", *(["response.in_progress"] * 3)],
        )
        for event in parsed[:first_output]:
            self.assertEqual(event["response"]["id"], "resp_prefill-heartbeat")
            self.assertEqual(event["response"]["status"], "in_progress")
            self.assertEqual(event["response"]["output"], [])
        self.assertNotIn("response.in_progress", kinds[first_output:])
        self.assertEqual(
            [event["sequence_number"] for event in parsed], list(range(len(parsed)))
        )
        before_output, _, after_output = raw.partition(
            b"event: response.output_item.added\n"
        )
        self.assertNotIn(b": splash-keepalive", before_output)
        self.assertEqual(after_output.count(b": splash-keepalive"), 2)
        self.assertEqual(kinds[-1], "response.completed")
        response = parsed[-1]["response"]
        self.assertEqual(response["id"], "resp_prefill-heartbeat")
        self.assertEqual(response["output"][0]["content"][0]["text"], "answer")
        app.backend.cancel.assert_not_called()
        app.persist_response.assert_called_once()

    def test_buffered_tool_generation_keeps_stream_alive(self):
        schema = {
            "type": "object",
            "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
            "required": ["questions"],
        }
        _, policy = tool_schema.normalize_tools(
            [{"type": "function", "function": {"name": "ask", "parameters": schema}}],
            "required",
            False,
        )
        handler = object.__new__(api.FrontendHandler)
        handler.wfile = io.BytesIO()
        handler._last_sse_write = 0.0
        handler._response_started = True
        handler._client_disconnected = lambda: False
        clock = [0.0]
        snapshots = []
        fragments = [
            '<tool_call>\n<function=ask>\n<parameter=questions>\n["',
            *("a" for _ in range(24)),
            '"]\n</parameter>\n</function>\n</tool_call>',
        ]
        events = iter(
            [
                *(("text", fragment) for fragment in fragments),
                ("done", backend_api.NativeResult("stop", 1, 26, 1, 1, 1)),
            ]
        )

        def next_event(**_kwargs):
            snapshots.append(handler.wfile.getvalue())
            clock[0] += 0.5
            return next(events)

        job = SimpleNamespace(
            public_id="request",
            tool_policy=policy,
            response_validator=None,
            deadline=100,
            events=SimpleNamespace(get=next_event),
        )
        with mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]):
            _, _, calls, _, _ = handler._collect(
                job,
                False,
                True,
                lambda field, text: handler._sse({field: text}),
                handler._sse,
                handler._sse_keepalive,
            )
        # All native events were immediately available, but the JSON array
        # remained buffered across several heartbeat periods.
        self.assertGreaterEqual(snapshots[-3].count(b": splash-keepalive"), 3)
        self.assertNotIn(b"questions", snapshots[-3])
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"questions": ["a" * 24]}
        )

    def test_visible_stream_output_resets_keepalive_clock(self):
        handler = object.__new__(api.FrontendHandler)
        handler.wfile = io.BytesIO()
        handler._last_sse_write = 0.0
        handler._client_disconnected = lambda: False
        clock = [0.0]
        job = SimpleNamespace(
            deadline=100, events=SimpleNamespace(get=lambda **_: ("text", "x"))
        )
        heartbeat = mock.Mock(wraps=handler._sse_keepalive)
        with mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]):
            for index in range(12):
                clock[0] += 0.5
                handler._next_event(job, heartbeat)
                if index % 2:
                    handler._responses_sse("event", {"text": "x"})
                else:
                    handler._sse({"text": "x"})
        heartbeat.assert_not_called()

    def test_anthropic_stream_pings_during_prefill_and_decode_waits(self):
        plan = Plan([[4]], before_start=True, block=True)
        harness = self.harness(FakeRuntime(plan))
        with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.02):
            connection, response = harness.open_stream(
                "/v1/messages",
                self.anthropic_body(stream=True),
            )
            try:
                self.assertEqual(
                    json.loads(self.next_sse_data(response)), {"type": "ping"}
                )
                self.assertFalse(plan.started.is_set())
                plan.start_release.set()
                self.assertTrue(plan.started.wait(1))
                start = json.loads(self.next_sse_data(response))
                while start["type"] == "ping":
                    start = json.loads(self.next_sse_data(response))
                self.assertEqual(start["type"], "message_start")
                self.assertEqual(
                    start["message"]["usage"],
                    {
                        "input_tokens": 1,
                        "cache_read_input_tokens": 1,
                        "output_tokens": 0,
                    },
                )
                for _ in range(2):
                    self.assertEqual(
                        json.loads(self.next_sse_data(response)), {"type": "ping"}
                    )
                self.assertFalse(plan.release.is_set())
            finally:
                plan.start_release.set()
                plan.release.set()
                response.read()
                connection.close()

    def test_responses_tool_arguments_arrive_before_native_done(self):
        plan = Plan([[13]], reason="length", after_terminal=True)
        harness = self.harness(FakeRuntime(plan))
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
        with mock.patch.object(backend_api, "CallbackStreamer", PassthroughStreamer):
            connection, response = harness.open_stream(
                "/v1/responses",
                self.responses_body(
                    stream=True,
                    tools=[tool],
                    reasoning={"effort": "none"},
                ),
            )
            arguments = ""
            try:
                while "Par" not in arguments:
                    event = json.loads(self.next_sse_data(response))
                    if event["type"] == "response.function_call_arguments.delta":
                        arguments += event["delta"]
                self.assertEqual(arguments, '{"city":"Par')
                self.assertTrue(plan.terminal.wait(1))
                self.assertFalse(plan.terminal_release.is_set())
            finally:
                plan.terminal_release.set()
                response.read()
                connection.close()

    def test_anthropic_tool_arguments_arrive_before_native_done(self):
        plan = Plan([[13]], reason="length", after_terminal=True)
        harness = self.harness(FakeRuntime(plan))
        tool = {
            "name": "weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }
        with mock.patch.object(backend_api, "CallbackStreamer", PassthroughStreamer):
            connection, response = harness.open_stream(
                "/v1/messages",
                self.anthropic_body(stream=True, tools=[tool]),
            )
            arguments = ""
            try:
                while "Par" not in arguments:
                    event = json.loads(self.next_sse_data(response))
                    if (
                        event["type"] == "content_block_delta"
                        and event["delta"]["type"] == "input_json_delta"
                    ):
                        arguments += event["delta"]["partial_json"]
                self.assertEqual(arguments, '{"city":"Par')
                self.assertTrue(plan.terminal.wait(1))
                self.assertFalse(plan.terminal_release.is_set())
            finally:
                plan.terminal_release.set()
                response.read()
                connection.close()

    def test_openai_sdk_tool_streams_match_nonstream_canonical_arguments(self):
        value = "x" * (model_output.TOOL_ARGUMENT_DELTA_CHARS * 2 + 137)
        tokenizer = FakeTokenizer()
        tokenizer.fragments[25] = (
            "<tool_call>\n<function=echo>\n"
            f"<parameter=text>\n{value}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        tokenizer.backend_tokenizer = _byte_backend(tokenizer.fragments)
        harness = self.harness(
            FakeRuntime(*(Plan([[25]]) for _ in range(4))), tokenizer=tokenizer
        )
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
        chat_tools = [
            {
                "type": "function",
                "function": {"name": "echo", "parameters": schema},
            }
        ]
        responses_tools = [{"type": "function", "name": "echo", "parameters": schema}]
        host, port = harness.server.server_address
        with OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="test",
            timeout=5,
            max_retries=0,
        ) as client:
            chat = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                tools=chat_tools,
                extra_body={"reasoning_effort": "none"},
            )
            canonical_chat = chat.choices[0].message.tool_calls[0]
            chat_deltas = []
            finish_reasons = []
            for chunk in client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                tools=chat_tools,
                stream=True,
                extra_body={"reasoning_effort": "none"},
            ):
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reasons.append(choice.finish_reason)
                chat_deltas.extend(choice.delta.tool_calls or [])

            header = chat_deltas[0]
            argument_deltas = [
                delta.function.arguments
                for delta in chat_deltas[1:]
                if delta.function and delta.function.arguments is not None
            ]
            self.assertEqual((header.index, header.type), (0, "function"))
            self.assertTrue(header.id.startswith("call_"))
            self.assertEqual(header.function.name, "echo")
            self.assertIsNone(header.function.arguments)
            self.assertGreater(len(argument_deltas), 1)
            self.assertLessEqual(
                max(map(len, argument_deltas)), model_output.TOOL_ARGUMENT_DELTA_CHARS
            )
            self.assertEqual(
                "".join(argument_deltas), canonical_chat.function.arguments
            )
            self.assertEqual(finish_reasons[-1], "tool_calls")

            response = client.responses.create(
                model="test-model",
                input="hello",
                tools=responses_tools,
                reasoning={"effort": "none"},
            )
            canonical_response = next(
                item for item in response.output if item.type == "function_call"
            )
            response_events = list(
                client.responses.create(
                    model="test-model",
                    input="hello",
                    tools=responses_tools,
                    reasoning={"effort": "none"},
                    stream=True,
                )
            )

        added = next(
            event.item
            for event in response_events
            if event.type == "response.output_item.added"
            and event.item.type == "function_call"
        )
        response_deltas = [
            event.delta
            for event in response_events
            if event.type == "response.function_call_arguments.delta"
        ]
        arguments_done = next(
            event
            for event in response_events
            if event.type == "response.function_call_arguments.done"
        )
        item_done = next(
            event.item
            for event in response_events
            if event.type == "response.output_item.done"
            and event.item.type == "function_call"
        )
        self.assertEqual((added.name, added.arguments), ("echo", ""))
        self.assertEqual(added.id, item_done.id)
        self.assertGreater(len(response_deltas), 1)
        self.assertLessEqual(
            max(map(len, response_deltas)), model_output.TOOL_ARGUMENT_DELTA_CHARS
        )
        self.assertEqual("".join(response_deltas), canonical_response.arguments)
        self.assertEqual(arguments_done.item_id, item_done.id)
        self.assertEqual(arguments_done.name, item_done.name)
        self.assertEqual(arguments_done.arguments, canonical_response.arguments)
        self.assertEqual(item_done.arguments, canonical_response.arguments)
        self.assertEqual(response_events[-1].type, "response.completed")

    def test_split_tool_marker_omits_template_whitespace_and_parses_call(self):
        batches = [[17], [18]]
        harness = self.harness(FakeRuntime(Plan(batches), Plan(batches)))
        tools = [{"type": "function", "function": {"name": "weather"}}]
        body = self.body(tools=tools, reasoning_effort="none")
        with mock.patch.object(backend_api, "CallbackStreamer", PassthroughStreamer):
            status, _, payload = harness.request("POST", "/v1/chat/completions", body)
            stream_status, _, stream_payload = harness.request(
                "POST", "/v1/chat/completions", dict(body, stream=True)
            )
        self.assertEqual((status, stream_status), (200, 200))
        content = json.loads(payload)["choices"][0]["message"]["content"]
        chunks = [
            json.loads(line[6:])
            for line in stream_payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
        streamed = "".join(delta.get("content", "") for delta in deltas)
        self.assertIsNone(content)
        self.assertEqual(streamed, "")
        calls = next(delta["tool_calls"] for delta in deltas if "tool_calls" in delta)
        self.assertEqual(calls[0]["function"]["name"], "weather")
        self.assertNotIn("<tool_call>", stream_payload.decode())

    def test_chat_tool_stream_preserves_mixed_whitespace_exactly(self):
        batches = [[20], [21], [22], [23]]
        harness = self.harness(FakeRuntime(Plan(batches), Plan(batches)))
        tools = [{"type": "function", "function": {"name": "weather"}}]
        body = self.body(tools=tools, reasoning_effort="none")
        with mock.patch.object(backend_api, "CallbackStreamer", PassthroughStreamer):
            status, _, payload = harness.request("POST", "/v1/chat/completions", body)
            stream_body = dict(body, stream=True)
            stream_status, _, stream_payload = harness.request(
                "POST", "/v1/chat/completions", stream_body
            )
        self.assertEqual((status, stream_status), (200, 200))
        message = json.loads(payload)["choices"][0]["message"]
        chunks = [
            json.loads(line[6:])
            for line in stream_payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
        streamed = "".join(delta.get("content", "") for delta in deltas)
        expected = " \nalpha \t\n beta \t\n"
        self.assertEqual(message["content"], expected)
        self.assertEqual(streamed, message["content"])
        self.assertEqual(
            [
                call["function"]["name"]
                for delta in deltas
                for call in delta.get("tool_calls", [])
                if "name" in call["function"]
            ],
            ["weather"],
        )
        self.assertNotIn("<tool_call>", stream_payload.decode())

    def test_responses_tool_stream_preserves_mixed_whitespace_exactly(self):
        batches = [[20], [21], [22], [23]]
        harness = self.harness(FakeRuntime(Plan(batches), Plan(batches)))
        tools = [{"type": "function", "name": "weather"}]
        body = self.responses_body(
            tools=tools,
            reasoning={"effort": "none"},
        )
        with mock.patch.object(backend_api, "CallbackStreamer", PassthroughStreamer):
            status, _, payload = harness.request("POST", "/v1/responses", body)
            stream_status, _, stream_payload = harness.request(
                "POST", "/v1/responses", dict(body, stream=True)
            )
        self.assertEqual((status, stream_status), (200, 200))
        response = json.loads(payload)
        nonstream_message = next(
            item for item in response["output"] if item["type"] == "message"
        )
        events = self.response_events(stream_payload)
        streamed = "".join(
            event["delta"]
            for event in events
            if event["type"] == "response.output_text.delta"
        )
        completed = events[-1]["response"]
        stream_messages = [
            item for item in completed["output"] if item["type"] == "message"
        ]
        expected = " \nalpha \t\n beta \t\n"
        self.assertEqual(nonstream_message["content"][0]["text"], expected)
        self.assertEqual(
            "".join(item["content"][0]["text"] for item in stream_messages),
            expected,
        )
        self.assertEqual(streamed, expected)
        self.assertEqual(
            [item["type"] for item in completed["output"]],
            ["message", "function_call", "message"],
        )
        self.assertNotIn("<tool_call>", stream_payload.decode())

    def test_streaming_multiple_tools_never_exposes_xml(self):
        harness = self.harness(FakeRuntime(Plan([[5], [16]])))
        tools = [
            {"type": "function", "function": {"name": "weather"}},
            {"type": "function", "function": {"name": "time"}},
        ]
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(
                stream=True,
                tools=tools,
                reasoning_effort="none",
            ),
        )
        self.assertEqual(status, 200, payload)
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks if chunk["choices"]]
        self.assertEqual("".join(delta.get("content", "") for delta in deltas), "")
        calls = [
            call
            for delta in deltas
            for call in delta.get("tool_calls", [])
            if "name" in call["function"]
        ]
        self.assertEqual(
            [call["function"]["name"] for call in calls], ["weather", "time"]
        )
        self.assertNotIn("<tool_call>", payload.decode())

    def test_streaming_malformed_tool_keeps_safe_prefix_without_xml(self):
        harness = self.harness(FakeRuntime(Plan([[14], [8]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(
                stream=True,
                tools=[self.rich_weather_tool()],
                reasoning_effort="none",
            ),
        )
        self.assertEqual(status, 200, payload)
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        deltas = [
            chunk["choices"][0]["delta"] for chunk in chunks if chunk.get("choices")
        ]
        self.assertEqual(
            [delta["content"] for delta in deltas if delta.get("content")],
            ["first "],
        )
        error = next(chunk["error"] for chunk in chunks if "error" in chunk)
        self.assertEqual(error["code"], "invalid_model_output")
        self.assertNotIn("<tool_call>", payload.decode())

    def test_incomplete_tool_prefix_preserves_whitespace_without_xml(self):
        plans = [Plan([[20], [19]], reason="length") for _ in range(4)]
        harness = self.harness(FakeRuntime(*plans))
        chat_tools = [{"type": "function", "function": {"name": "weather"}}]
        responses_tools = [{"type": "function", "name": "weather"}]
        chat_body = self.body(tools=chat_tools, reasoning_effort="none")
        responses_body = self.responses_body(
            tools=responses_tools, reasoning={"effort": "none"}
        )
        with mock.patch.object(backend_api, "CallbackStreamer", PassthroughStreamer):
            status, _, payload = harness.request(
                "POST", "/v1/chat/completions", chat_body
            )
            stream_status, _, stream_payload = harness.request(
                "POST", "/v1/chat/completions", dict(chat_body, stream=True)
            )
            responses_status, _, responses_payload = harness.request(
                "POST", "/v1/responses", responses_body
            )
            responses_stream_status, _, responses_stream_payload = harness.request(
                "POST", "/v1/responses", dict(responses_body, stream=True)
            )
        self.assertEqual(
            (status, stream_status, responses_status, responses_stream_status),
            (200, 200, 200, 200),
        )
        expected = " \nalpha \t"
        chat_message = json.loads(payload)["choices"][0]["message"]
        chunks = [
            json.loads(line[6:])
            for line in stream_payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        chat_streamed = "".join(
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks
            if chunk["choices"]
        )
        self.assertEqual(chat_message["content"], expected)
        self.assertEqual(chat_streamed, expected)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")

        response = json.loads(responses_payload)
        response_message = next(
            item for item in response["output"] if item["type"] == "message"
        )
        events = self.response_events(responses_stream_payload)
        responses_streamed = "".join(
            event["delta"]
            for event in events
            if event["type"] == "response.output_text.delta"
        )
        completed_message = next(
            item
            for item in events[-1]["response"]["output"]
            if item["type"] == "message"
        )
        self.assertEqual(response_message["content"][0]["text"], expected)
        self.assertEqual(completed_message["content"][0]["text"], expected)
        self.assertEqual(responses_streamed, expected)
        for wire in (stream_payload, responses_stream_payload):
            self.assertNotIn(b"<tool_ca", wire)

    def test_incomplete_full_tool_marker_preserves_literal_prefix_exactly(self):
        plans = [Plan([[24]], reason="length") for _ in range(4)]
        harness = self.harness(FakeRuntime(*plans))
        chat_body = self.body(
            tools=[{"type": "function", "function": {"name": "weather"}}],
            reasoning_effort="none",
        )
        responses_body = self.responses_body(
            tools=[{"type": "function", "name": "weather"}],
            reasoning={"effort": "none"},
        )
        with mock.patch.object(
            backend_api, "CallbackStreamer", SeededRandomChunkStreamer
        ):
            chat_status, _, chat_payload = harness.request(
                "POST", "/v1/chat/completions", chat_body
            )
            chat_stream_status, _, chat_stream_payload = harness.request(
                "POST", "/v1/chat/completions", dict(chat_body, stream=True)
            )
            responses_status, _, responses_payload = harness.request(
                "POST", "/v1/responses", responses_body
            )
            responses_stream_status, _, responses_stream_payload = harness.request(
                "POST", "/v1/responses", dict(responses_body, stream=True)
            )
        self.assertEqual(
            (
                chat_status,
                chat_stream_status,
                responses_status,
                responses_stream_status,
            ),
            (200, 200, 200, 200),
        )

        expected = "answer <"
        chat_message = json.loads(chat_payload)["choices"][0]["message"]
        chat_chunks = [
            json.loads(line[6:])
            for line in chat_stream_payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        chat_streamed = "".join(
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chat_chunks
            if chunk["choices"]
        )
        self.assertEqual(chat_message["content"], expected)
        self.assertEqual(chat_message["tool_calls"][0]["function"]["arguments"], "{")
        self.assertEqual(chat_streamed, expected)
        self.assertEqual(chat_chunks[-1]["choices"][0]["finish_reason"], "length")

        response = json.loads(responses_payload)
        self.assertEqual(response["status"], "incomplete")
        self.assertEqual(
            [item["type"] for item in response["output"]],
            ["message", "function_call"],
        )
        nonstream_text = response["output"][0]["content"][0]["text"]
        response_events = self.response_events(responses_stream_payload)
        streamed_text = "".join(
            event["delta"]
            for event in response_events
            if event["type"] == "response.output_text.delta"
        )
        completed = response_events[-1]["response"]
        self.assertEqual(completed["status"], "incomplete")
        self.assertEqual(
            [item["type"] for item in completed["output"]],
            ["message", "function_call"],
        )
        self.assertEqual(completed["output"][1]["status"], "incomplete")
        completed_text = completed["output"][0]["content"][0]["text"]
        self.assertEqual(nonstream_text, expected)
        self.assertEqual(streamed_text, expected)
        self.assertEqual(completed_text, expected)
        for wire in (chat_stream_payload, responses_stream_payload):
            self.assertNotIn(b"<tool_call>", wire)

    def test_tool_arguments_always_serialize_as_strict_json(self):
        nested = "[" * 10000 + "0" + "]" * 10000
        huge_integer = "9" * 5000
        text = (
            "<tool_call>\n<function=f>\n"
            "<parameter=constant>\nNaN\n</parameter>\n"
            "<parameter=overflow>\n1e10000\n</parameter>\n"
            f"<parameter=integer>\n{huge_integer}\n</parameter>\n"
            f"<parameter=nested>\n{nested}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, calls = model_output.parse_tool_calls(text, 1)
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(
            arguments,
            {
                "constant": "NaN",
                "overflow": "1e10000",
                "integer": huge_integer,
                "nested": nested,
            },
        )

    def test_json_value_has_an_explicit_nesting_limit(self):
        allowed = "[" * MAX_JSON_NESTING + "0" + "]" * MAX_JSON_NESTING
        deep_array = "[" * (MAX_JSON_NESTING + 1) + "0" + "]" * (MAX_JSON_NESTING + 1)
        deep_object = (
            '{"value":' * (MAX_JSON_NESTING + 1) + "0" + "}" * (MAX_JSON_NESTING + 1)
        )
        escaped = r"brackets in a string: \"[{]}\"" * (MAX_JSON_NESTING + 1)

        self.assertIsInstance(tool_schema.json_value(allowed), list)
        self.assertEqual(tool_schema.json_value(deep_array), deep_array)
        self.assertEqual(tool_schema.json_value(deep_object), deep_object)
        self.assertEqual(tool_schema.json_value(json.dumps(escaped)), escaped)

    def test_tool_parser_preserves_schema_typed_strings(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["text", "count"],
        }
        policy = tool_schema.ToolPolicy({}, {"echo": schema}, True, False)
        text = (
            "<tool_call>\n<function=echo>\n"
            "<parameter=text>\n123\n</parameter>\n"
            "<parameter=count>\n3\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, calls = model_output.parse_tool_calls(text, 1, policy)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"text": "123", "count": 3},
        )

    def test_tool_parser_accepts_stock_client_string_constraints(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "minLength": 4,
                    "maxLength": 16,
                    "pattern": "^[A-Z]+$",
                }
            },
            "required": ["text"],
        }
        tools, policy = tool_schema.normalize_tools(
            [
                {
                    "type": "function",
                    "function": {"name": "echo", "parameters": schema},
                }
            ],
            "auto",
            True,
        )
        self.assertEqual(len(tools), 1)
        self.assertIn("/(?s:.*)/", tool_schema._tool_arguments_grammar(schema))
        valid = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"text":"VALID"}'},
        }
        model_output.validate_tool_calls([valid], policy)
        invalid = json.loads(json.dumps(valid))
        invalid["function"]["arguments"] = '{"text":"x"}'
        with self.assertRaisesRegex(api.APIError, "invalid arguments for echo"):
            model_output.validate_tool_calls([invalid], policy)

    def test_tool_parser_accepts_all_of_string_constraints(self):
        value_schema = {
            "type": "string",
            "description": "Recipient",
            "allOf": [
                {"pattern": r"^[^\n\r]*$"},
                {"pattern": r"^[\s\S]{0,300}$"},
            ],
        }
        schema = {
            "type": "object",
            "properties": {"to": value_schema},
            "required": ["to"],
        }
        tools, policy = tool_schema.normalize_tools(
            [
                {
                    "type": "function",
                    "function": {"name": "send_message", "parameters": schema},
                }
            ],
            "auto",
            True,
        )
        self.assertEqual(len(tools), 1)
        self.assertIn("/(?s:.*)/", tool_schema._tool_arguments_grammar(schema))

        valid = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "send_message",
                "arguments": '{"to":"main [agent-1]"}',
            },
        }
        model_output.validate_tool_calls([valid], policy)
        for value in ("main\nagent", "x" * 301):
            invalid = json.loads(json.dumps(valid))
            invalid["function"]["arguments"] = json.dumps({"to": value})
            with self.assertRaisesRegex(
                api.APIError, "invalid arguments for send_message"
            ):
                model_output.validate_tool_calls([invalid], policy)

    def test_tool_grammar_defers_property_name_assertions_to_validation(self):
        schema = {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "object",
                    "propertyNames": {"pattern": "^[a-z]+$"},
                    "additionalProperties": {"type": "string"},
                }
            },
        }
        tools, policy = tool_schema.normalize_tools(
            [
                {
                    "type": "function",
                    "function": {"name": "ask", "parameters": schema},
                }
            ],
            "auto",
            True,
        )
        self.assertEqual(len(tools), 1)
        grammar = tool_schema._tool_arguments_grammar(schema)
        self.assertNotIn("propertyNames", grammar)
        invalid = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "ask",
                "arguments": '{"answers":{"NOT_LOWER":"value"}}',
            },
        }
        with self.assertRaisesRegex(api.APIError, "invalid arguments for ask"):
            model_output.validate_tool_calls([invalid], policy)

    def test_tool_grammar_defers_nested_lookaround_patterns_to_validation(self):
        pattern = (
            r"^(?!\.\.?(?:/|$))[A-Za-z0-9_\-.~:@+]{1,200}"
            r"(?:/(?!\.\.?(?:/|$))[A-Za-z0-9_\-.~:@+]{1,200}){0,14}$"
        )
        schema = {
            "type": "object",
            "properties": {
                "writes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"path": {"$ref": "#/$defs/path"}},
                        "required": ["path"],
                    },
                }
            },
            "$defs": {"path": {"type": "string", "pattern": pattern}},
            "required": ["writes"],
        }
        original = json.dumps(schema)
        _, policy = tool_schema.normalize_tools(
            [{"type": "function", "function": {"name": "write", "parameters": schema}}],
            "required",
            False,
        )
        grammar = tool_schema._tool_arguments_grammar(schema)
        self.assertFalse(generation_constraints.LLMatcher.validate_grammar(grammar))
        self.assertEqual(json.dumps(schema), original)
        for path in ("website/index.html", "../secret", "website/../secret"):
            with self.subTest(path=path):
                arguments = json.dumps({"writes": [{"path": path}]})
                call = {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "write", "arguments": arguments},
                }
                xml = (
                    "<tool_call>\n<function=write>\n<parameter=writes>\n"
                    + json.dumps([{"path": path}])
                    + "\n</parameter>\n</function>\n</tool_call>"
                )
                projector = model_output.StreamingToolCallProjector(policy, "request_1")
                if path == "website/index.html":
                    model_output.validate_tool_calls([call], policy)
                    projector.put(xml)
                    self.assertEqual(len(projector.closed_calls), 1)
                else:
                    with self.assertRaisesRegex(
                        api.APIError, "invalid arguments for write"
                    ):
                        model_output.validate_tool_calls([call], policy)
                    with self.assertRaisesRegex(
                        api.APIError, "invalid arguments for write"
                    ):
                        projector.put(xml)
                    self.assertEqual(projector.closed_calls, [])

    def test_tool_grammar_preserves_keyword_named_properties_and_literals(self):
        literal = {"pattern": "(?=literal)", "propertyNames": "literal"}
        references = [{"$ref": "#/x"}, {"$ref": "https://example.com/x.json"}]
        schema = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "propertyNames": {"const": literal},
                "nested": {"enum": [literal]},
                "constant": {"const": references[0]},
                "choice": {"enum": references},
                "defaulted": {"type": "object", "default": references[1]},
                "shown": {"type": "object", "examples": references},
                "linked": {"type": "array", "items": {"$ref": "#/$defs/x"}},
            },
            "$defs": {"x": {"type": "integer"}},
            "required": ["pattern", "propertyNames", "nested", "constant", "choice"],
        }
        projected = _grammar_compatible_schema(schema)
        self.assertEqual(projected.pop("x-guidance"), {"lenient": True})
        self.assertEqual(projected, schema)
        _, policy = tool_schema.normalize_tools(
            [{"type": "function", "function": {"name": "echo", "parameters": schema}}],
            "required",
            False,
        )
        grammar = tool_schema._tool_arguments_grammar(schema)
        self.assertFalse(generation_constraints.LLMatcher.validate_grammar(grammar))
        for reference in references:
            self.assertIn(json.dumps(reference, separators=(",", ":")), grammar)
        self.assertIn('"items":{"$ref":"#/$defs/__splash_root/$defs/x"}', grammar)
        self.assertNotIn("__splash_root/x", grammar)
        arguments = {
            "pattern": "literal",
            "propertyNames": literal,
            "nested": literal,
            "constant": references[0],
            "choice": references[1],
        }
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo", "arguments": json.dumps(arguments)},
        }
        model_output.validate_tool_calls([call], policy)
        _, validator = tool_schema.normalize_response_format(
            {"type": "json_schema", "json_schema": {"name": "echo", "schema": schema}}
        )
        validator.validate(arguments)

    def test_tool_parser_resolves_root_and_chained_string_refs(self):
        schema = {
            "type": "object",
            "properties": {
                "base": {"type": "string"},
                "direct": {"$ref": "#/properties/base"},
                "chained": {"$ref": "#/$defs/alias"},
            },
            "$defs": {
                "alias": {"$ref": "#/$defs/text"},
                "text": {"type": "string"},
            },
        }
        policy = tool_schema.ToolPolicy({}, {"echo": schema}, True, False)
        text = (
            "<tool_call>\n<function=echo>\n"
            "<parameter=direct>\n123\n</parameter>\n"
            "<parameter=chained>\n456\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, calls = model_output.parse_tool_calls(text, 1, policy)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"direct": "123", "chained": "456"},
        )
        grammar = tool_schema._tool_arguments_grammar(schema)
        self.assertIn('[suffix="\\n</parameter>\\n"]', grammar)
        self.assertEqual(grammar.count("/(?s:.*)/"), 3)
        self.assertNotIn(r'[^"\s]', grammar)
        self.assertNotIn("RAW_MIDDLE", grammar)

    def test_tool_parser_preserves_raw_string_edge_whitespace(self):
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
        policy = tool_schema.ToolPolicy({}, {"echo": schema}, True, False)
        text = (
            "<tool_call>\n<function=echo>\n"
            '<parameter=text>\n "line"\n\n</parameter>\n'
            "</function>\n</tool_call>"
        )
        _, calls = model_output.parse_tool_calls(text, 1, policy)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"text": ' "line"\n'},
        )

    def test_tool_grammar_excludes_text_spellings_of_control_tokens(self):
        schema = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        policy = tool_schema.ToolPolicy({}, {"bash": schema}, False, False)
        grammar = json.loads(tool_schema.tool_grammar(policy, True))
        main = grammar["grammars"][0]["lark_grammar"]
        self.assertIn(
            r"TEXT: /(?s:.*)/ & ~/(?s:.*)(<tool_call>|<\/think>)(?s:.*)/",
            main,
        )

    def test_tool_parser_keeps_nested_closing_tags_inside_raw_value(self):
        text = (
            "<tool_call>\n<function=echo>\n<parameter=x>\nhello"
            "</function>\n</tool_call>world\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        content, calls = model_output.parse_tool_calls(text, 1)
        self.assertEqual(content, "")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"x": "hello</function>\n</tool_call>world"},
        )

    def test_tool_parser_handles_mixed_parameter_types(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
                "flags": {"type": "array", "items": {"type": "boolean"}},
                "maybe": {"type": ["number", "null"]},
            },
            "required": ["text", "count", "flags", "maybe"],
            "additionalProperties": False,
        }
        tool = {
            "type": "function",
            "function": {"name": "echo", "parameters": schema},
        }
        _, policy = tool_schema.normalize_tools([tool], "required", True)
        raw = (
            "visible prefix "
            "<tool_call>\n<function=echo>\n"
            "<parameter=text>\nsnow </function>\n</tool_call> tail\n</parameter>\n"
            "<parameter=count>\n3\n</parameter>\n"
            "<parameter=flags>\n[true,false]\n</parameter>\n"
            "<parameter=maybe>\nnull\n</parameter>\n"
            "</function>\n</tool_call> visible suffix"
        )
        content, calls = model_output.parse_tool_calls(raw, "fuzz", policy)
        model_output.validate_tool_calls(calls, policy)
        self.assertEqual(content, "visible prefix  visible suffix")

    def test_tool_parser_preserves_enum_edge_whitespace(self):
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string", "enum": [" edge "]}},
        }
        policy = tool_schema.ToolPolicy({}, {"echo": schema}, True, False)
        text = (
            "<tool_call>\n<function=echo>\n"
            "<parameter=text>\n edge \n</parameter>\n"
            "</function>\n</tool_call>"
        )
        _, calls = model_output.parse_tool_calls(text, 1, policy)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"text": " edge "}
        )
        with self.assertRaisesRegex(api.APIError, "XML framing"):
            tool_schema._tool_arguments_grammar(
                {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "enum": ["bad\n</parameter>value"],
                        }
                    },
                }
            )

    def test_tool_choice_shapes_the_grammar_policy_not_the_prompt(self):
        tokenizer = FakeTokenizer()
        backend = backend_api.NativeBackend(FakeRuntime(), tokenizer)
        self.addCleanup(backend.close)
        app = request_frontend.Frontend(tokenizer, backend, "test-model", 128, 16, 1, 2)
        tools = [
            {"type": "function", "function": {"name": "f"}},
            {"type": "function", "function": {"name": "g"}},
        ]
        app.prepare(self.body(tools=tools, tool_choice="none"))
        self.assertNotIn("tools", tokenizer.templates[-1][1])
        named = {"type": "function", "function": {"name": "g"}}
        cases = (
            ({"tool_choice": "required"}, True, True, "(tool_0 | tool_1)+"),
            ({"tool_choice": named}, True, True, "(tool_0)+"),
            ({"parallel_tool_calls": False}, False, False, "(tool_0 | tool_1)? tail"),
        )
        for extra, required, parallel, start in cases:
            with self.subTest(**extra):
                job, _, _ = app.prepare(self.body(tools=tools, **extra))
                rendered, kwargs = tokenizer.templates[-1]
                self.assertEqual(rendered, [{"role": "user", "content": "hello"}])
                self.assertEqual(kwargs["tools"], tools)
                policy = job.tool_policy
                self.assertEqual(
                    (policy.required, policy.parallel), (required, parallel)
                )
                main = json.loads(tool_schema.tool_grammar(policy, True))["grammars"][0]
                self.assertIn(f"start: think {start}", main["lark_grammar"])
                self.assertEqual(
                    [
                        name
                        for name in "fg"
                        if f"<function={name}>" in main["lark_grammar"]
                    ],
                    list(policy.schemas),
                )

    def test_tool_names_accept_long_mcp_names_up_to_128_characters(self):
        runtime = FakeRuntime(Plan([[4]]))
        harness = self.harness(runtime)
        name = (
            "mcp__github_enterprise_server__list_pull_request_review_comments_for_repos"
        )
        self.assertEqual(len(name), 74)
        status, _, _ = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(
                tools=[{"type": "function", "function": {"name": name}}],
                reasoning_effort="none",
            ),
        )
        self.assertEqual(status, 200)
        _, kwargs = harness.tokenizer.templates[-1]
        self.assertEqual(kwargs["tools"][0]["function"]["name"], name)
        cases = (
            (
                {"type": "function", "function": {"name": "mcp__" + "a" * 124}},
                "tool name must match [A-Za-z0-9_-]{1,128}",
            ),
            ({"type": "web_search"}, "only function tools are supported"),
        )
        for tool, message in cases:
            with self.subTest(message=message):
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", self.body(tools=[tool])
                )
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(payload)["error"]["message"], message)
        self.assertEqual(len(runtime.requests), 1)

    @staticmethod
    def rich_weather_tool():
        return {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "$defs": {
                        "city": {
                            "anyOf": [
                                {"type": "string", "enum": ["Paris", "Rome"]},
                                {"type": "null"},
                            ]
                        }
                    },
                    "type": "object",
                    "properties": {"city": {"$ref": "#/$defs/city"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }

    def test_pydantic_style_schema_and_invalid_output(self):
        tool = self.rich_weather_tool()
        harness = self.harness(FakeRuntime(Plan([[8]]), Plan([[5]])))
        for expected in (500, 200):
            status, _, payload = harness.request(
                "POST",
                "/v1/chat/completions",
                self.body(tools=[tool], reasoning_effort="none"),
            )
            self.assertEqual(status, expected)
            if status == 500:
                self.assertEqual(
                    json.loads(payload)["error"]["code"], "invalid_model_output"
                )

    def test_required_named_and_parallel_tool_policies(self):
        tools = [
            {"type": "function", "function": {"name": "weather"}},
            {"type": "function", "function": {"name": "time"}},
        ]
        named = {"type": "function", "function": {"name": "weather"}}
        cases = (
            ([4], {"tool_choice": "required"}),
            ([7], {"tool_choice": named}),
            ([9], {"parallel_tool_calls": False}),
        )
        for batch, extra in cases:
            harness = self.harness(FakeRuntime(Plan([batch])))
            status, _, payload = harness.request(
                "POST",
                "/v1/chat/completions",
                self.body(tools=tools, reasoning_effort="none", **extra),
            )
            self.assertEqual(status, 500)
            self.assertEqual(
                json.loads(payload)["error"]["code"], "invalid_model_output"
            )

    def test_remote_tool_schema_ref_is_rejected_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        remote = {"$ref": "https://example.com/city.json"}
        tool = self.rich_weather_tool()
        tool["function"]["parameters"]["$defs"]["city"] = remote
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "city",
                "schema": {"type": "object", "properties": {"city": remote}},
            },
        }
        for parameter, value in (
            ("tools", [tool]),
            ("response_format", response_format),
        ):
            with self.subTest(parameter=parameter):
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", self.body(**{parameter: value})
                )
                self.assertEqual(status, 400)
                self.assertIn(
                    "schema reference is not allowed: https://example.com/city.json",
                    json.loads(payload)["error"]["message"],
                )
        self.assertEqual(runtime.requests, [])

    def test_streaming_invalid_tool_is_an_sse_error(self):
        harness = self.harness(FakeRuntime(Plan([[8]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(
                stream=True,
                tools=[self.rich_weather_tool()],
                reasoning_effort="none",
            ),
        )
        events = [
            line[6:]
            for line in payload.decode().splitlines()
            if line.startswith("data: ")
        ]
        errors = [json.loads(event) for event in events if event.startswith('{"error"')]
        self.assertEqual(status, 200)
        self.assertEqual(errors[0]["error"]["code"], "invalid_model_output")
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(event) for event in events[:-1]]
        self.assertFalse(
            any(
                choice.get("delta", {}).get("tool_calls")
                for chunk in chunks
                for choice in chunk.get("choices", [])
            )
        )
        self.assertNotIn("<tool_call>", payload.decode())

    def test_streaming_tool_error_cancels_inflight_native_and_recovers(self):
        function = self.rich_weather_tool()["function"]
        cases = (
            (
                "/v1/chat/completions",
                self.body(
                    stream=True,
                    tools=[{"type": "function", "function": function}],
                    reasoning_effort="none",
                ),
            ),
            (
                "/v1/responses",
                self.responses_body(
                    stream=True,
                    tools=[{"type": "function", **function}],
                    reasoning={"effort": "none"},
                ),
            ),
            (
                "/v1/messages",
                self.anthropic_body(
                    stream=True,
                    tools=[
                        {
                            "name": function["name"],
                            "input_schema": function["parameters"],
                        }
                    ],
                ),
            ),
        )
        for path, body in cases:
            with self.subTest(path=path):
                # The invalid call is emitted while native still has work.
                # A one-batch immediate completion cannot catch missing aborts.
                plan = Plan([[8], [4]], delay=1)
                runtime = FakeRuntime(plan)
                harness = self.harness(runtime, queue_size=1)
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 200, payload)
                self.assertIn(
                    b"api_error" if path == "/v1/messages" else b"invalid_model_output",
                    payload,
                )
                self.assertEqual(runtime.cancel_count, 1)
                runtime.threads[0].join(1)
                self.assertEqual(runtime.pending_count, 0)
                for _ in range(100):
                    if not harness.backend.active:
                        break
                    time.sleep(0.01)
                self.assertEqual(harness.backend.active, {})
                self.assertEqual(harness.server.requests.stats()["active"], 0)
                self.assertFalse(runtime.calls[0].cancel())
                self.assertEqual(runtime.cancel_count, 1)
                status, _, _ = harness.request(
                    "POST", "/v1/chat/completions", self.body(reasoning_effort="none")
                )
                self.assertEqual(status, 200)

    def test_parallel_tool_stream_fails_before_releasing_a_valid_first_call(self):
        tokenizer = FakeTokenizer()
        tokenizer.fragments[25] = (
            "<tool_call>\n<function=weather>\n"
            "<parameter=city>\nParis\n</parameter>\n"
            "</function>\n</tool_call>"
            "<tool_call>\n<function=echo>\n"
            "<parameter=count>\ninvalid\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        tokenizer.backend_tokenizer = _byte_backend(tokenizer.fragments)
        schemas = {
            "weather": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                "required": ["city"],
            },
            "echo": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        }
        chat_tools = [
            {
                "type": "function",
                "function": {"name": name, "parameters": schema},
            }
            for name, schema in schemas.items()
        ]
        harness = self.harness(
            FakeRuntime(Plan([[25]]), Plan([[25]])), tokenizer=tokenizer
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(
                stream=True,
                tools=chat_tools,
                reasoning_effort="none",
            ),
        )
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(status, 200, payload)
        self.assertEqual(chunks[-1]["error"]["code"], "invalid_model_output")
        self.assertFalse(
            any(
                choice.get("delta", {}).get("tool_calls")
                for chunk in chunks
                for choice in chunk.get("choices", [])
            )
        )
        self.assertNotIn("<tool_call>", payload.decode())

        responses_tools = [
            {"type": "function", "name": name, "parameters": schema}
            for name, schema in schemas.items()
        ]
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                stream=True,
                tools=responses_tools,
                reasoning={"effort": "none"},
            ),
        )
        events = self.response_events(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(
            events[-1]["response"]["error"]["code"], "invalid_model_output"
        )
        self.assertFalse(
            any(
                event["type"].startswith("response.function_call_arguments")
                or event.get("item", {}).get("type") == "function_call"
                for event in events
            )
        )
        self.assertNotIn("<tool_call>", payload.decode())

    def test_qwen_reasoning_efforts(self):
        tokenizer = FakeTokenizer()
        backend = backend_api.NativeBackend(FakeRuntime(), tokenizer)
        self.addCleanup(backend.close)
        app = request_frontend.Frontend(tokenizer, backend, "test-model", 128, 16, 1, 2)
        for effort in ("xhigh", "medium", "low"):
            app.prepare(self.body(reasoning_effort=effort))
            template = tokenizer.templates[-1][1]
            self.assertTrue(template["enable_thinking"])
            self.assertEqual(template["reasoning_effort"], effort)
        app.prepare(self.body(reasoning_effort="none"))
        template = tokenizer.templates[-1][1]
        self.assertFalse(template["enable_thinking"])
        self.assertNotIn("reasoning_effort", template)

    @staticmethod
    def reasoning_template(*, default=True, efforts=None):
        validation = ""
        if efforts is not None:
            validation = (
                "{% if reasoning_effort is defined and reasoning_effort not in "
                + repr(efforts)
                + " %}{{ raise_exception('unsupported effort') }}{% endif %}"
            )
        return (
            validation + "{% for message in messages %}"
            "{{ '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
            "{% if enable_thinking | default("
            + ("true" if default else "false")
            + ") %}{{ '<think>\\n' }}"
            "{% else %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}{% endif %}"
        )

    def test_native_thinking_default_controls_parser_and_grammar(self):
        schema = {"type": "object", "properties": {"x": {"const": 3}}}
        for default in (False, True):
            for tools in (False, True):
                with self.subTest(default=default, tools=tools):
                    tokenizer = TemplateTokenizer(
                        self.reasoning_template(default=default)
                    )
                    factory = FakeConstraintFactory()
                    batches = [[1], [26], [10]] if default else [[10]]
                    harness = self.harness(
                        FakeRuntime(Plan(batches), Plan(batches)),
                        tokenizer=tokenizer,
                        constraint_factory=factory,
                    )
                    body = self.body(
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"schema": schema},
                        }
                    )
                    if tools:
                        body["tools"] = [self.rich_weather_tool()]
                    for stream in (False, True):
                        status, _, payload = harness.request(
                            "POST", "/v1/chat/completions", {**body, "stream": stream}
                        )
                        self.assertEqual(status, 200, payload)
                        self.assertEqual("think:" in factory.grammars[-1], default)
                        if stream:
                            self.assertEqual(b'"reasoning_content"' in payload, default)
                            self.assertIn(b"[DONE]", payload)
                        else:
                            message = json.loads(payload)["choices"][0]["message"]
                            self.assertEqual(message["content"], '{"x":3}')
                            self.assertEqual("reasoning_content" in message, default)
                    for _, kwargs in tokenizer.templates:
                        self.assertNotIn("enable_thinking", kwargs)
                        self.assertNotIn("reasoning_effort", kwargs)

    def test_dictionary_template_uses_the_selected_generation_prefix(self):
        tokenizer = TemplateTokenizer(
            {
                "default": self.reasoning_template(default=False),
                "tool_use": self.reasoning_template(default=True),
            }
        )
        factory = FakeConstraintFactory()
        app = request_frontend.Frontend(
            tokenizer, None, "test-model", 128, 16, 1, 2, factory
        )
        for tools, expected in (([], False), ([self.rich_weather_tool()], True)):
            body = self.body(tools=tools)
            body["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": "A historical <think> must not open thinking.",
                },
            )
            job, thinking, _ = app.prepare(body)
            self.assertEqual((thinking, job.thinking), (expected, expected))
            self.assertEqual(app.count_tokens(body), len(job.prompt_tokens))
        self.assertIn("think:", factory.grammars[-1])

    def test_responses_omitted_or_null_effort_preserves_native_thinking_off(self):
        tokenizer = TemplateTokenizer(self.reasoning_template(default=False))
        harness = self.harness(
            FakeRuntime(Plan([[27]]), Plan([[27]]), Plan([[27]])), tokenizer=tokenizer
        )
        for extra in ({}, {"reasoning": None}, {"reasoning": {"effort": None}}):
            with self.subTest(extra=extra):
                status, _, payload = harness.request(
                    "POST", "/v1/responses", self.responses_body(**extra)
                )
                self.assertEqual(status, 200, payload)
                response = json.loads(payload)
                self.assertEqual(response["status"], "completed")
                self.assertEqual(
                    [item["type"] for item in response["output"]], ["message"]
                )
                self.assertEqual(
                    response["output"][0]["content"][0]["text"], "answer\n"
                )
                kwargs = tokenizer.templates[-1][1]
                self.assertNotIn("enable_thinking", kwargs)
                self.assertNotIn("reasoning_effort", kwargs)

    def test_reasoning_aliases_only_retry_when_the_template_rejects_native_values(self):
        for accepted, aliases in (
            (("minimal", "low", "medium", "high", "xhigh", "max"), {}),
            (
                ("low", "medium", "xhigh"),
                {"high": "xhigh", "max": "xhigh", "minimal": "low"},
            ),
        ):
            tokenizer = TemplateTokenizer(self.reasoning_template(efforts=accepted))
            app = request_frontend.Frontend(
                tokenizer, None, "test-model", 128, 16, 1, 2
            )
            for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
                with self.subTest(accepted=accepted, effort=effort):
                    tokenizer.templates.clear()
                    job, thinking, _ = app.prepare(self.body(reasoning_effort=effort))
                    self.assertTrue(thinking)
                    self.assertTrue(job.thinking)
                    self.assertEqual(
                        [
                            kwargs["reasoning_effort"]
                            for _, kwargs in tokenizer.templates
                        ],
                        [effort, aliases[effort]] if effort in aliases else [effort],
                    )

    def test_reasoning_template_errors_do_not_silently_drop_effort(self):
        tokenizer = TemplateTokenizer(self.reasoning_template(efforts=("medium",)))
        app = request_frontend.Frontend(tokenizer, None, "test-model", 128, 16, 1, 2)
        with self.assertRaises(api.APIError):
            app.prepare(self.body(reasoning_effort="high"))
        self.assertEqual(
            [kwargs["reasoning_effort"] for _, kwargs in tokenizer.templates],
            ["high", "xhigh"],
        )
        with mock.patch.object(
            tokenizer, "apply_chat_template", side_effect=ValueError("bad content")
        ) as render:
            with self.assertRaises(api.APIError) as caught:
                app.prepare(self.body(reasoning_effort="high"))
            render.assert_called_once()
            self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_reasoning_effort_accepts_only_standard_protocol_values(self):
        tokenizer = FakeTokenizer()
        app = request_frontend.Frontend(tokenizer, None, "test-model", 128, 16, 1, 2)
        for effort in ("", "on", "off", "ultra", True, 1, [], {}):
            with (
                self.subTest(effort=effort),
                self.assertRaisesRegex(api.APIError, "invalid reasoning_effort"),
            ):
                app.prepare(self.body(reasoning_effort=effort))
        self.assertEqual(tokenizer.templates, [])
        job, thinking, _ = app.prepare(self.body(reasoning_effort=None))
        self.assertTrue(thinking)
        self.assertTrue(job.thinking)
        self.assertNotIn("enable_thinking", tokenizer.templates[-1][1])

    def test_explicit_thinking_mode_must_be_honored(self):
        for prefix, effort in (
            ("<think>\\n", "none"),
            ("<think>\\n\\n</think>\\n\\n", "low"),
        ):
            tokenizer = TemplateTokenizer(
                "{{ '<|im_start|>assistant\\n" + prefix + "' }}"
            )
            app = request_frontend.Frontend(
                tokenizer, None, "test-model", 128, 16, 1, 2
            )
            with (
                self.subTest(effort=effort),
                self.assertRaisesRegex(api.APIError, "requested thinking mode"),
            ):
                app.prepare(self.body(reasoning_effort=effort))

    def test_thinking_prefix_excludes_completed_or_nonassistant_history(self):
        for suffix, expected in (
            ("", False),
            ("<think>\n", True),
            ("<think>\n\n</think>\n\n", False),
        ):
            history = "<|im_start|>assistant\n<think>old</think>answer<|im_end|>\n"
            self.assertEqual(
                request_frontend._thinking_from_prefix(
                    history + "<|im_start|>assistant\n" + suffix
                ),
                expected,
            )
        for rendered in (
            "",
            "<|im_start|>user\n<think>",
            "<|im_start|>assistant\n<think>old<|im_end|>\n",
        ):
            with self.subTest(rendered=rendered), self.assertRaises(api.APIError):
                request_frontend._thinking_from_prefix(rendered)

    def test_anthropic_thinking_off_is_not_reenabled_by_effort(self):
        tokenizer = TemplateTokenizer(self.reasoning_template())
        app = request_frontend.Frontend(tokenizer, None, "test-model", 128, 16, 1, 2)
        for thinking in (None, {"type": "disabled"}):
            body = self.anthropic_body(output_config={"effort": "high"})
            if thinking is not None:
                body["thinking"] = thinking
            chat = api.anthropic_to_chat_body(body)
            job, active, _ = app.prepare(chat)
            self.assertFalse(active)
            self.assertFalse(job.thinking)
            self.assertEqual(
                app.count_tokens(api.anthropic_to_chat_prompt(body)),
                len(job.prompt_tokens),
            )
            self.assertFalse(tokenizer.templates[-1][1]["enable_thinking"])

    def test_anthropic_effort_is_validated_even_when_thinking_is_off(self):
        for thinking in (
            None,
            {"type": "disabled"},
            {"type": "enabled"},
            {"type": "adaptive"},
        ):
            for effort in ("invalid", "none", "ultra", {}, None):
                body = self.anthropic_body(output_config={"effort": effort})
                if thinking is not None:
                    body["thinking"] = thinking
                with (
                    self.subTest(thinking=thinking, effort=effort),
                    self.assertRaisesRegex(api.APIError, "output_config.effort"),
                ):
                    api.anthropic_to_chat_prompt(body)

    def test_structured_output_validation_and_constraint(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        }
        factory = FakeConstraintFactory()
        harness = self.harness(FakeRuntime(Plan([[10]])), constraint_factory=factory)
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(reasoning_effort="none", response_format=response_format),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"], '{"x":3}'
        )
        self.assertIn("%json", factory.grammars[0])

        invalid = self.harness(FakeRuntime(Plan([[4]])))
        status, _, payload = invalid.request(
            "POST",
            "/v1/chat/completions",
            self.body(reasoning_effort="none", response_format=response_format),
        )
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(payload)["error"]["code"], "invalid_model_output")

    def test_tools_and_response_format_allow_calls_or_a_valid_final_answer(self):
        factory = FakeConstraintFactory()
        harness = self.harness(
            FakeRuntime(Plan([[10]]), Plan([[5]]), Plan([[4]])),
            constraint_factory=factory,
        )
        body = self.body(
            tools=[self.rich_weather_tool()],
            reasoning_effort="none",
            response_format={"type": "json_object"},
        )
        status, _, payload = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"], '{"x":3}'
        )
        status, _, payload = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200, payload)
        call = json.loads(payload)["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Paris"})
        status, _, payload = harness.request("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 500, payload)
        self.assertEqual(json.loads(payload)["error"]["code"], "invalid_model_output")
        grammar = json.loads(factory.grammars[0])["grammars"][0]["lark_grammar"]
        self.assertIn("| answer)", grammar)
        self.assertIn("%json", grammar)

    def test_structured_output_schema_is_visible_without_changing_input_history(self):
        harness = self.harness(FakeRuntime())
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        for system in ([], [{"role": "system", "content": "Original instructions."}]):
            with self.subTest(system=system):
                body = self.body(
                    messages=[*system, {"role": "user", "content": "Calculate 6 * 7."}],
                    tools=[self.rich_weather_tool()],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"schema": schema},
                    },
                )
                original = json.dumps(body, sort_keys=True)
                prompt = harness.app._prepare_prompt(body)
                self.assertEqual(json.dumps(body, sort_keys=True), original)
                self.assertEqual(prompt.messages[: len(system)], system)
                self.assertEqual(prompt.messages[-1], body["messages"][-1])
                instruction = prompt.messages[len(system)]
                self.assertEqual(instruction["role"], "system")
                self.assertEqual(
                    json.loads(instruction["content"].split("\n", 1)[1]), schema
                )
                self.assertEqual(prompt.response_schema, schema)
                self.assertIsNotNone(prompt.response_validator)

    def test_structured_output_keeps_required_tool_choice(self):
        harness = self.harness(FakeRuntime(Plan([[10]]), Plan([[5]])))
        body = self.body(
            tools=[self.rich_weather_tool()],
            tool_choice="required",
            reasoning_effort="none",
            response_format={"type": "json_object"},
        )
        for expected in (500, 200):
            status, _, payload = harness.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, expected, payload)

    def test_structured_stream_keeps_tool_markers_inside_json_strings(self):
        _, policy = tool_schema.normalize_tools(
            [self.rich_weather_tool()], "auto", True
        )
        payload = ' \n{"text":"<tool_call>\\n<function=weather> and </think>"}'
        for width in (1, 2, 7, len(payload)):
            with self.subTest(width=width):
                projector = model_output.StreamingToolCallProjector(
                    policy, "test", structured=True
                )
                events = []
                for start in range(0, len(payload), width):
                    events.extend(projector.put(payload[start : start + width]))
                tail = projector.finish(payload, [], False)
                self.assertTrue(all(kind == "content" for kind, _ in events))
                self.assertEqual(
                    "".join(value for _, value in events) + "".join(tail), payload
                )

    def test_structured_tool_response_streams_and_recovers_from_length(self):
        harness = self.harness(FakeRuntime(Plan([[5]]), Plan([[13]], reason="length")))
        body = self.body(
            tools=[self.rich_weather_tool()],
            reasoning_effort="none",
            response_format={"type": "json_object"},
            stream=True,
        )
        for incomplete in (False, True):
            status, _, payload = harness.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, 200, payload)
            self.assertNotIn(b"invalid_model_output", payload)
            chunks = [
                json.loads(line[6:])
                for line in payload.decode().splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            self.assertEqual(
                chunks[-1]["choices"][0]["finish_reason"],
                "length" if incomplete else "tool_calls",
            )

    def test_responses_tools_and_schema_work_in_stream_and_nonstream(self):
        harness = self.harness(FakeRuntime(Plan([[10]]), Plan([[10]]), Plan([[5]])))
        tool = self.rich_weather_tool()["function"]
        body = self.responses_body(
            tools=[{"type": "function", **tool}],
            reasoning={"effort": "none"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"x": {"const": 3}},
                        "required": ["x"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        status, _, payload = harness.request("POST", "/v1/responses", body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(
            json.loads(payload)["output"][-1]["content"][0]["text"], '{"x":3}'
        )
        status, _, payload = harness.request(
            "POST", "/v1/responses", {**body, "stream": True}
        )
        self.assertEqual(status, 200, payload)
        self.assertIn(b"response.completed", payload)
        self.assertNotIn(b"invalid_model_output", payload)
        status, _, payload = harness.request("POST", "/v1/responses", body)
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["output"][-1]["type"], "function_call")

    def test_reasoning_effort_is_preserved_until_the_template_renders(self):
        harness = self.harness(FakeRuntime())
        prompt = harness.app._prepare_prompt(self.body(reasoning_effort="minimal"))
        self.assertEqual(prompt.reasoning_effort, "minimal")
        _, thinking, _ = harness.app.prepare(self.body(reasoning_effort="minimal"))
        self.assertTrue(thinking)

    def test_chat_length_safely_finishes_partial_structured_and_tool_output(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        }
        runtime = FakeRuntime(
            Plan([[12]], reason="length"),
            Plan([[13]], reason="length"),
            Plan([[13]], reason="length"),
        )
        harness = self.harness(runtime, constraint_factory=FakeConstraintFactory())
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(reasoning_effort="none", response_format=response_format),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(response["choices"][0]["finish_reason"], "length")
        self.assertEqual(response["choices"][0]["message"]["content"], '{"x":')

        tool = self.rich_weather_tool()
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(tools=[tool], reasoning_effort="none"),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(response["choices"][0]["finish_reason"], "length")
        self.assertNotIn("<tool_call>", payload.decode())

        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(tools=[tool], reasoning_effort="none", stream=True),
        )
        self.assertEqual(status, 200, payload)
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertNotIn("<tool_call>", payload.decode())

    def test_tool_constraint_requires_object_parameters(self):
        factory = FakeConstraintFactory()
        harness = self.harness(FakeRuntime(), constraint_factory=factory)
        cases = (
            (
                {"name": "bad", "parameters": {"type": "string"}},
                "top-level JSON object",
            ),
            (
                {
                    "name": "bad",
                    "parameters": {
                        "type": "object",
                        "allOf": [{"type": "object"}],
                    },
                },
                "unsupported top-level tool schema",
            ),
            (
                {
                    "name": "bad",
                    "parameters": {
                        "type": "object",
                        "properties": {" leading": {"type": "string"}},
                    },
                },
                "invalid tool parameter name",
            ),
            ({"name": "bad>name", "parameters": {}}, "tool name must match"),
        )
        for function, message in cases:
            tool = {"type": "function", "function": function}
            status, _, payload = harness.request(
                "POST", "/v1/chat/completions", self.body(tools=[tool])
            )
            self.assertEqual(status, 400)
            self.assertIn(message, json.loads(payload)["error"]["message"])
        self.assertEqual(factory.grammars, [])

    def test_tool_constraint_accepts_nullable_non_strings(self):
        for keyword, value_type in (("anyOf", "integer"), ("oneOf", "object")):
            schema = {
                "type": "object",
                "properties": {
                    "value": {keyword: [{"type": value_type}, {"type": "null"}]}
                },
            }
            grammar = tool_schema._tool_arguments_grammar(schema)
            self.assertIn("%json", grammar)
            self.assertNotIn("RAW_START", grammar)

    def test_nullable_strings_use_json_and_preserve_string_null(self):
        variants = (
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
            {"oneOf": [{"type": "string"}, {"type": "null"}]},
            {"type": ["string", "null"]},
            {"enum": ["null", None]},
        )
        for value_schema in variants:
            with self.subTest(schema=value_schema):
                schema = {
                    "type": "object",
                    "properties": {"value": value_schema},
                    "required": ["value"],
                }
                grammar = tool_schema._tool_arguments_grammar(schema)
                self.assertIn("%json", grammar)
                policy = tool_schema.ToolPolicy({}, {"echo": schema}, True, False)
                values = []
                for wire in ("null", '"null"'):
                    text = (
                        "<tool_call>\n<function=echo>\n<parameter=value>\n"
                        f"{wire}\n</parameter>\n</function>\n</tool_call>"
                    )
                    _, calls = model_output.parse_tool_calls(text, 1, policy)
                    values.append(
                        json.loads(calls[0]["function"]["arguments"])["value"]
                    )
                self.assertEqual(values, [None, "null"])

    def test_validation(self):
        harness = self.harness(FakeRuntime())
        invalid = [
            self.body(temperature=-1),
            self.body(top_p=0),
            self.body(top_k=33),
            self.body(stop=""),
            self.body(stop=3),
            self.body(stop=["x"] * 5),
            self.body(stop=["x", ""]),
            self.body(reasoning_effort="bad"),
            self.body(seed=2**64),
            self.body(n=True),
            self.body(n=1.0),
            self.body(logprobs=0),
            self.body(temperature=1e300),
            self.body(temperature=float("nan")),
            self.body(temperature=1e-46),
            self.body(top_p=1e-46),
            self.body(presence_penalty=1),
            self.body(presence_penalty=False),
            self.body(min_p=0.1),
            self.body(logit_bias={"1": 2}),
            self.body(stream="true"),
            self.body(messages=[{"role": "system", "content": "instructions"}]),
            self.body(messages=[{"role": "assistant", "content": "answer"}]),
            self.body(messages=[{"role": "user", "content": [{"type": "image_url"}]}]),
            self.body(
                messages=[{"role": "user", "content": [{"type": "text", "text": 3}]}]
            ),
            self.body(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": None, "tool_calls": 3},
                ]
            ),
            self.body(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "tool", "content": "result", "tool_call_id": {}},
                ]
            ),
            self.body(tools=[{"type": "function", "function": "bad"}]),
            self.body(tools=[{"type": "function", "function": {"name": ""}}]),
            self.body(
                tools=[
                    {"type": "function", "function": {"name": "same"}},
                    {"type": "function", "function": {"name": "same"}},
                ]
            ),
            self.body(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {"type": "unknown"},
                        },
                    }
                ]
            ),
            self.body(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": []},
                    }
                ]
            ),
            self.body(parallel_tool_calls=1),
            self.body(
                tools=[{"type": "function", "function": {"name": "weather"}}],
                tool_choice={"type": "function", "function": {"name": []}},
            ),
            self.body(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "tool_calls": [{"function": "bad"}]},
                ]
            ),
        ]
        for body in invalid:
            status, _, _ = harness.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, 400)
        for stop in (None, [], "x", ["x", "y"]):
            body = self.body()
            if stop is not None:
                body["stop"] = stop
            status, _, _ = harness.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, 200)

    def test_frontend_limits_generation_and_token_count_preparation_to_two(self):
        tokenizer = BlockingTokenizer()
        app = request_frontend.Frontend(
            tokenizer,
            SimpleNamespace(status=lambda: {}),
            "test-model",
            32768,
            16,
            2.0,
            2,
        )
        results = []
        errors = []

        def prepare(index):
            try:
                operation = app.count_tokens if index == 1 else app.prepare
                results.append(operation(self.body()))
            except Exception as error:
                errors.append(error)

        threads = [
            threading.Thread(target=prepare, args=(index,)) for index in range(3)
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(tokenizer.entered.wait(1))
        time.sleep(0.05)
        self.assertEqual(tokenizer.calls, 2)
        self.assertEqual(
            app.status()["frontend"],
            {
                "preparation_capacity": 2,
                "active": 2,
                "waiting": 1,
            },
        )
        tokenizer.release.set()
        for thread in threads:
            thread.join(2)
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 3)
        self.assertEqual(tokenizer.maximum_active, 2)

    def test_frontend_rejects_invalid_preparation_capacity(self):
        with self.assertRaisesRegex(ValueError, "preparation capacity"):
            request_frontend.Frontend(
                FakeTokenizer(),
                SimpleNamespace(status=lambda: {}),
                "test-model",
                128,
                16,
                1,
                0,
            )

    def test_context_window_rejects_output_budget_without_truncating(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, max_context=10, default_max_new=9)
        cases = (
            (
                "/v1/chat/completions",
                self.body(max_completion_tokens=9),
            ),
            ("/v1/chat/completions", self.body(max_tokens=9)),
            ("/v1/responses", self.responses_body(max_output_tokens=9)),
        )
        for path, body in cases:
            with self.subTest(path=path, body=body):
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 400, payload)
                self.assertIn(
                    "prompt and max_completion_tokens exceed the context window",
                    json.loads(payload)["error"]["message"],
                )
                error = json.loads(payload)["error"]
                self.assertEqual(error["type"], "invalid_request_error")
                if path != "/v1/messages":
                    self.assertEqual(error["code"], "context_length_exceeded")
        self.assertEqual(runtime.requests, [])

    def test_anthropic_output_budget_is_clamped_to_the_remaining_window(self):
        # Claude Code sends max_tokens 32K on every turn and does not compact
        # for it; the request must proceed with what the window allows.
        runtime = FakeRuntime()
        harness = self.harness(runtime, max_context=10, default_max_new=9)
        status, _, payload = harness.request(
            "POST", "/v1/messages", self.anthropic_body(max_tokens=9)
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(runtime.requests), 1)
        request = runtime.requests[0]
        self.assertEqual(
            len(request.prompt_tokens) + request.logical_max_output_tokens, 10
        )
        self.assertGreaterEqual(request.logical_max_output_tokens, 1)

    def test_default_output_budget_uses_remaining_context(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, max_context=10, default_max_new=9)
        for path, body in (
            ("/v1/chat/completions", self.body()),
            ("/v1/responses", self.responses_body()),
        ):
            with self.subTest(path=path):
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 200, payload)
                self.assertEqual(runtime.requests[-1].logical_max_output_tokens, 8)
                self.assertEqual(runtime.requests[-1].prompt_tokens, (101, 102))

        harness.app.max_context = 100000
        harness.app.default_max_new = 32768
        for length, expected in ((90000, 10000), (99999, 1)):
            with mock.patch.object(
                FakeTokenizer, "__call__", return_value={"input_ids": [101] * length}
            ):
                job, *_ = harness.app.prepare(self.body())
            self.assertEqual(job.max_new_tokens, expected)
            self.assertEqual(len(job.prompt_tokens), length)
        with mock.patch.object(
            FakeTokenizer, "__call__", return_value={"input_ids": [101] * 100000}
        ):
            with self.assertRaisesRegex(api.APIError, "prompt exceeds") as caught:
                harness.app.prepare(self.body())
            self.assertEqual(caught.exception.code, "context_length_exceeded")

    def test_preparation_consumes_original_deadline_and_releases_slots(self):
        app = request_frontend.Frontend(
            FakeTokenizer(), None, "test-model", 128, 16, 10, 1
        )
        for elapsed in (0.25, 5):
            clock = [100.0]

            def tokenize(*_args, **_kwargs):
                clock[0] += elapsed
                return "<|im_start|>assistant\n<think>\n"

            with (
                self.subTest(elapsed=elapsed),
                mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]),
                mock.patch.object(
                    app.tokenizer, "apply_chat_template", side_effect=tokenize
                ),
            ):
                if elapsed < 1:
                    job, *_ = app.prepare(self.body(timeout=1))
                    self.assertEqual(job.deadline, 101.0)
                else:
                    with self.assertRaises(api.APIError) as error:
                        app.prepare(self.body(timeout=1))
                    self.assertEqual(
                        (error.exception.status, error.exception.code),
                        (504, "request_timeout"),
                    )
                self.assertEqual(
                    (app.preparation_active, app.preparation_waiting), (0, 0)
                )
                self.assertTrue(app.preparation_slots.acquire(blocking=False))
                app.preparation_slots.release()

    def test_preparation_queue_respects_request_timeout(self):
        app = request_frontend.Frontend(
            FakeTokenizer(), None, "test-model", 128, 16, 10, 1
        )
        app.preparation_slots.acquire()
        try:
            with self.assertRaises(api.APIError) as error:
                app.prepare(self.body(timeout=0.02))
            self.assertEqual(
                (error.exception.status, error.exception.code), (504, "request_timeout")
            )
            self.assertEqual(app.tokenizer.templates, [])
            self.assertEqual((app.preparation_active, app.preparation_waiting), (0, 0))
        finally:
            app.preparation_slots.release()

    def test_expired_preparation_skips_later_stages(self):
        app = request_frontend.Frontend(
            FakeTokenizer(), None, "test-model", 128, 16, 10, 1
        )
        for stage in ("grammar", "images"):
            clock = [100.0]

            def expire(*_args, **_kwargs):
                clock[0] += 5
                return []

            factory = mock.Mock()
            app.constraint_factory = factory
            with (
                self.subTest(stage=stage),
                mock.patch.object(api.time, "monotonic", side_effect=lambda: clock[0]),
                mock.patch.object(app, "_prepare_images", return_value=[]) as images,
                mock.patch.object(
                    app.tokenizer,
                    "apply_chat_template",
                    return_value="<|im_start|>assistant\n<think>\n",
                ) as tokenize,
            ):
                if stage == "grammar":
                    factory.create.side_effect = expire
                else:
                    images.side_effect = expire
                with self.assertRaises(api.APIError) as error:
                    app.prepare(
                        self.body(timeout=1, response_format={"type": "json_object"})
                    )
                self.assertEqual(error.exception.status, 504)
                if stage == "grammar":
                    images.assert_called_once()
                    tokenize.assert_called_once()
                else:
                    tokenize.assert_not_called()
                    factory.create.assert_not_called()
                self.assertEqual(app.preparation_active, 0)

    def test_http_body_time_is_included_before_native_admission(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        original = api.FrontendHandler._read_json_body

        def read_body(handler):
            body = original(handler)
            time.sleep(0.05)
            return body

        for path, body in (
            ("/v1/chat/completions", self.body(timeout=0.01)),
            ("/v1/responses", self.responses_body(timeout=0.01)),
            ("/v1/messages", self.anthropic_body(timeout=0.01)),
            ("/v1/messages/count_tokens", self.anthropic_body(timeout=0.01)),
        ):
            with (
                self.subTest(path=path),
                mock.patch.object(api.FrontendHandler, "_read_json_body", read_body),
            ):
                status, _, payload = harness.request("POST", path, body)
                self.assertEqual(status, 504, payload)
                self.assertEqual(
                    json.loads(payload)["error"]["type"],
                    "timeout_error"
                    if path.startswith("/v1/messages")
                    else "server_error",
                )
        self.assertEqual(runtime.requests, [])
        self.assertEqual(harness.tokenizer.templates, [])

    def test_request_body_validation(self):
        harness = self.harness(FakeRuntime())
        for length in (0, -1):
            status, _ = harness.raw_post(b"", length)
            self.assertEqual(status, 400)
        status, _ = harness.raw_post(b"", api.MAX_REQUEST_BYTES + 1)
        self.assertEqual(status, 413)
        status, _ = harness.raw_post(b"\xff", 1)
        self.assertEqual(status, 400)
        deeply_nested = ("[" * 10000 + "0" + "]" * 10000).encode()
        status, _ = harness.raw_post(deeply_nested, len(deeply_nested))
        self.assertEqual(status, 400)

    def test_huge_numbers_and_nonstandard_json_are_rejected(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        huge = 10**400
        for field in (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "min_p",
            "timeout",
        ):
            with self.subTest(field=field):
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", self.body(**{field: huge})
                )
                self.assertEqual(status, 400, payload)
                self.assertIn("error", json.loads(payload))

        nonstandard = json.dumps(self.body()).replace(
            '"temperature": 0', '"temperature": NaN'
        )
        status, payload = harness.raw_post(
            nonstandard.encode(), len(nonstandard.encode())
        )
        self.assertEqual(status, 400, payload)

        history = [
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"x":NaN}'},
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ]
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body(messages=history)
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(runtime.requests, [])

    def test_http_media_type_encoding_and_transfer_are_strict(self):
        harness = self.harness(FakeRuntime())
        for headers, expected in (
            ({}, 415),
            ({"Content-Type": "text/plain"}, 415),
            ({"Content-Type": "application/json", "Content-Encoding": "gzip"}, 415),
            ({"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, 400),
        ):
            with self.subTest(headers=headers):
                status, _, _ = harness.request(
                    "POST", "/v1/chat/completions", self.body(), headers
                )
                self.assertEqual(status, expected)

        status, _, _ = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(reasoning_effort="none"),
            {"Content-Type": "application/vnd.splash+json; charset=utf-8"},
        )
        self.assertEqual(status, 200)

    def test_http_body_is_exact_and_io_has_a_deadline(self):
        harness = self.harness(FakeRuntime(), io_timeout=0.1)
        payload = json.dumps(self.body()).encode()

        digits = str(len(payload))
        for invalid_length in (f"+{digits}", f"{digits[0]}_{digits[1:]}"):
            with self.subTest(content_length=invalid_length):
                connection = socket.create_connection(
                    harness.server.server_address, timeout=2
                )
                connection.sendall(
                    b"POST /v1/chat/completions HTTP/1.1\r\n"
                    b"Host: localhost\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {invalid_length}\r\n\r\n".encode()
                    + payload
                )
                self.assertIn(b" 400 ", connection.recv(4096))
                connection.close()

        connection = socket.create_connection(harness.server.server_address, timeout=2)
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(payload) + 1}\r\n\r\n".encode()
            + payload
        )
        connection.shutdown(socket.SHUT_WR)
        self.assertIn(b" 400 ", connection.recv(4096))
        connection.close()

        connection = socket.create_connection(harness.server.server_address, timeout=2)
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\nContent-Type: application/json\r\n"
            b"Content-Length: 100\r\n\r\n{"
        )
        self.assertIn(b" 408 ", connection.recv(4096))
        connection.close()

        connection = socket.create_connection(harness.server.server_address, timeout=2)
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
        )

        def drip():
            for byte in payload:
                try:
                    connection.sendall(bytes((byte,)))
                except OSError:
                    return
                time.sleep(0.02)

        sender = threading.Thread(target=drip)
        sender.start()
        started = time.monotonic()
        self.assertIn(b" 408 ", connection.recv(4096))
        self.assertLess(time.monotonic() - started, 0.5)
        connection.close()
        sender.join(1)

        connection = socket.create_connection(harness.server.server_address, timeout=2)
        connection.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost:")
        started = time.monotonic()
        self.assertEqual(connection.recv(4096), b"")
        self.assertLess(time.monotonic() - started, 1)
        connection.close()

    def test_control_plane_survives_a_full_client_connection_burst(self):
        harness = self.harness(FakeRuntime())

        def status():
            code, _, payload = harness.request("GET", "/status")
            return code, json.loads(payload)

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            results = list(executor.map(lambda _: status(), range(64)))
        self.assertTrue(all(code == 200 for code, _ in results))
        self.assertTrue(all(payload["ready"] for _, payload in results))
        self.assertGreaterEqual(api.FrontendServer.request_queue_size, 64)

    def _wait_for_http_active(self, admission, expected):
        deadline = time.monotonic() + 1
        while admission.stats()["active"] != expected and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(admission.stats()["active"], expected)

    def test_ingress_rejects_before_reading_body_and_keeps_control_reachable(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, queue_size=1)
        upload = socket.create_connection(harness.server.server_address, timeout=2)
        self.addCleanup(upload.close)
        upload.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n"
        )
        self._wait_for_http_active(harness.server.requests, 1)
        for path in (
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
        ):
            with self.subTest(path=path):
                connection = http.client.HTTPConnection(
                    *harness.server.server_address, timeout=1
                )
                self.addCleanup(connection.close)
                connection.putrequest("POST", path)
                connection.putheader("Content-Type", "application/json")
                connection.putheader("Content-Length", str(api.MAX_REQUEST_BYTES))
                connection.endheaders()  # Do not send any body to an overloaded server.
                response = connection.getresponse()
                self.assertEqual(response.status, 503)
                self.assertEqual(response.getheader("Connection"), "close")
                self.assertEqual(
                    json.loads(response.read())["error"]["type"],
                    "overloaded_error" if path == "/v1/messages" else "server_error",
                )
                connection.close()
        self.assertEqual(runtime.requests, [])
        self.assertEqual(harness.tokenizer.templates, [])
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        status, _, payload = harness.request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["http"]["requests"], {"active": 1, "capacity": 1}
        )
        upload.close()
        self._wait_for_http_active(harness.server.requests, 0)
        self.assertEqual(
            harness.request("POST", "/v1/chat/completions", self.body())[0], 200
        )

    def test_ingress_slot_covers_stream_and_releases_on_disconnect(self):
        blocking = Plan([[4]], block=True)
        runtime = FakeRuntime(blocking)
        harness = self.harness(runtime, queue_size=1)
        connection, response = harness.open_stream(
            "/v1/chat/completions", self.body(stream=True)
        )
        self.assertTrue(blocking.started.wait(1))
        self.assertEqual(response.status, 200)
        self.assertEqual(
            harness.request("POST", "/v1/chat/completions", self.body())[0], 503
        )
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        response.close()
        connection.close()
        self._wait_for_http_active(harness.server.requests, 0)
        self.assertTrue(blocking.cancelled.is_set())
        self.assertEqual(
            harness.request("POST", "/v1/chat/completions", self.body())[0], 200
        )

    def test_ingress_releases_slot_after_parse_preparation_and_native_errors(self):
        harness = self.harness(FakeRuntime(), queue_size=1)
        self.assertEqual(harness.raw_post(b"{", 1)[0], 400)
        self._wait_for_http_active(harness.server.requests, 0)
        with mock.patch.object(
            harness.app, "prepare", side_effect=RuntimeError("test")
        ):
            with mock.patch.object(api, "log_unexpected"):
                self.assertEqual(
                    harness.request("POST", "/v1/chat/completions", self.body())[0], 500
                )
        self._wait_for_http_active(harness.server.requests, 0)
        with mock.patch.object(harness.backend, "submit", return_value=False):
            self.assertEqual(
                harness.request("POST", "/v1/chat/completions", self.body())[0], 429
            )
        self._wait_for_http_active(harness.server.requests, 0)
        self.assertEqual(
            harness.request("POST", "/v1/chat/completions", self.body())[0], 200
        )

    def test_header_only_connections_are_bounded_before_thread_creation(self):
        with mock.patch.object(api.FrontendServer, "control_connection_capacity", 2):
            harness = self.harness(FakeRuntime(), queue_size=1)
        sockets = []
        try:
            for _ in range(harness.server.connections.capacity):
                connection = socket.create_connection(
                    harness.server.server_address, timeout=2
                )
                sockets.append(connection)
                connection.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost:")
            self._wait_for_http_active(harness.server.connections, 3)
            excess = socket.create_connection(harness.server.server_address, timeout=1)
            self.addCleanup(excess.close)
            response = http.client.HTTPResponse(excess)
            response.begin()
            self.assertEqual(response.status, 503)
            error = json.loads(response.read())["error"]
            self.assertEqual(error["type"], "server_error")
            self.assertEqual(error["code"], "frontend_overloaded")
            self.assertEqual(response.getheader("Retry-After"), "1")
            excess.close()
            self.assertEqual(harness.server.connections.stats()["active"], 3)
            self.assertEqual(harness.server.requests.stats()["active"], 0)
        finally:
            for connection in sockets:
                connection.close()
        self._wait_for_http_active(harness.server.connections, 0)
        self.assertEqual(harness.request("GET", "/health")[0], 200)

    def test_thread_start_failure_returns_connection_slot(self):
        harness = self.harness(FakeRuntime())
        with mock.patch.object(
            api.ThreadingHTTPServer, "process_request", side_effect=RuntimeError("test")
        ):
            with self.assertRaisesRegex(RuntimeError, "test"):
                harness.server.process_request(mock.Mock(), ("127.0.0.1", 1))
        self.assertEqual(harness.server.connections.stats()["active"], 0)

    def test_http_admission_capacity_is_exact_and_validated(self):
        for invalid in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "HTTP admission capacity"):
                api.HttpAdmission(invalid)
        admission = api.HttpAdmission(1)
        self.assertTrue(admission.acquire())
        self.assertFalse(admission.acquire())
        admission.release()
        self.assertEqual(admission.stats(), {"active": 0, "capacity": 1})
        with self.assertRaisesRegex(RuntimeError, "without acquisition"):
            admission.release()

    def test_sse_write_timeout_cancels_the_submitted_job(self):
        blocking = Plan([[4]], block=True)
        runtime = FakeRuntime(blocking)
        harness = self.harness(runtime)
        with mock.patch.object(api.FrontendHandler, "_sse", side_effect=TimeoutError):
            status, _, _ = harness.request(
                "POST",
                "/v1/chat/completions",
                self.body(stream=True, reasoning_effort="none"),
            )
        self.assertEqual(status, 200)
        deadline = time.monotonic() + 1
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)

    def test_template_render_error_is_structured_and_server_recovers(self):
        harness = self.harness(FakeRuntime(), tokenizer=RenderFailTokenizer())
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(messages=[{"role": "user", "content": "\ud800"}]),
        )
        self.assertEqual(status, 400)
        self.assertEqual(
            json.loads(payload)["error"]["message"], "messages could not be rendered"
        )
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", self.body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)

    def test_very_large_timeout_does_not_overflow_wait(self):
        blocking = Plan([[4]], block=True)
        harness = self.harness(FakeRuntime(blocking))
        release = threading.Timer(0.03, blocking.release.set)
        release.start()
        self.addCleanup(release.cancel)
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", self.body(timeout=1e308)
        )
        self.assertEqual(status, 200)

    def test_official_sampling_defaults_and_random_seed(self):
        tokenizer = FakeTokenizer()
        backend = backend_api.NativeBackend(FakeRuntime(), tokenizer)
        self.addCleanup(backend.close)
        app = request_frontend.Frontend(tokenizer, backend, "test-model", 128, 16, 1, 2)
        body = self.body()
        body.pop("temperature")
        with mock.patch("server.frontend.secrets.randbits", return_value=123):
            job, _, _ = app.prepare(body)
        self.assertEqual(
            (job.temperature, job.top_p, job.top_k, job.seed), (1.0, 0.95, 20, 123)
        )

    def test_pending_limit_returns_retryable_http_overload(self):
        blocking = Plan([[4]], block=True)
        runtime = FakeRuntime(blocking)
        harness = self.harness(runtime, queue_size=1)
        head, _, _ = harness.app.prepare(self.body(timeout=2))
        self.assertTrue(harness.backend.submit(head))
        self.assertTrue(blocking.started.wait(1))

        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body()
        )
        self.assertEqual(status, 429)
        self.assertEqual(json.loads(payload)["error"]["code"], "rate_limit_exceeded")
        self.assertEqual(len(runtime.requests), 1)
        blocking.release.set()

    def test_priority_validation_and_responses_forwarding(self):
        harness = self.harness(FakeRuntime())
        for value in ("urgent", 0, None, []):
            with self.assertRaisesRegex(api.APIError, "priority"):
                harness.app.prepare(self.body(priority=value))
        job, _, _ = harness.app.prepare_responses(
            self.responses_body(priority="foreground")
        )
        self.assertEqual(job.priority, backend_api.REQUEST_PRIORITIES["foreground"])

    def test_active_timeout_signals_and_next_request_runs(self):
        blocking = Plan([[4]], block=True)
        runtime = FakeRuntime(blocking, Plan([[4]]))
        harness = self.harness(runtime)
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", self.body(timeout=0.03)
        )
        self.assertEqual(status, 504)
        self.assertEqual(runtime.cancel_count, 1)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"],
            "plain answer\n",
        )

    def test_http_stream_graceful_disconnect_cancels_and_next_request_runs(self):
        streaming = Plan([[4]] * 100, delay=0.02)
        runtime = FakeRuntime(streaming, Plan([[4]]))
        harness = self.harness(runtime, timeout=10)
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=2
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(self.body(stream=True, reasoning_effort="none")),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        response.readline()
        response.close()
        connection.close()
        deadline = time.monotonic() + 2
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"],
            "plain answer\n",
        )

    def test_http_nonstream_disconnect_cancels_and_next_request_runs(self):
        blocking = Plan([[4]], block=True)
        runtime = FakeRuntime(blocking, Plan([[4]]))
        harness = self.harness(runtime, timeout=10)
        connection = http.client.HTTPConnection(*harness.server.server_address)
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(self.body(reasoning_effort="none")),
            {"Content-Type": "application/json"},
        )
        self.assertTrue(blocking.started.wait(1))
        connection.close()
        deadline = time.monotonic() + 2
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"],
            "plain answer\n",
        )

    def test_stream_disconnect_before_headers_cancels(self):
        waiting = Plan([[4]], block=True, before_start=True)
        runtime = FakeRuntime(waiting)
        harness = self.harness(runtime, timeout=10)
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=2
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(self.body(stream=True, reasoning_effort="none")),
            {"Content-Type": "application/json"},
        )
        deadline = time.monotonic() + 1
        while not runtime.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(runtime.calls), 1)
        connection.sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
        connection.close()
        waiting.start_release.set()
        deadline = time.monotonic() + 2
        while runtime.cancel_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(runtime.cancel_count, 1)

    def test_runtime_error_does_not_poison_next_request(self):
        runtime = FakeRuntime(Plan(error=("bad_request", "broken")), Plan([[4]]))
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body()
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)["error"]["code"], "bad_request")
        status, _, _ = harness.request("POST", "/v1/chat/completions", self.body())
        self.assertEqual(status, 200)

    def test_retryable_capacity_failure_maps_to_503(self):
        runtime = FakeRuntime(
            Plan(
                exception=api.engine_runtime.RequestFailed(
                    1,
                    b"capacity_exhausted",
                    b"system memory pressure is critical",
                    retryable=True,
                )
            ),
            Plan([[4]]),
        )
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body()
        )
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(payload)["error"]["code"], "capacity_exhausted")
        status, _, _ = harness.request("POST", "/v1/chat/completions", self.body())
        self.assertEqual(status, 200)

    def test_critical_memory_pressure_marks_server_unready(self):
        class PressuredRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                self.status_event = native_wire.StatusJsonEvent(
                    1,
                    native_wire.STATUS_SCHEMA_VERSION,
                    json.dumps(
                        {
                            "schema_version": native_wire.STATUS_SCHEMA_VERSION,
                            "ready": True,
                            "memory_pressure": "critical",
                            "metal": {"healthy": True},
                        },
                        separators=(",", ":"),
                    ).encode(),
                )

        harness = self.harness(PressuredRuntime())
        status, _, payload = harness.request("GET", "/ready")
        self.assertEqual(
            (status, json.loads(payload)), (503, {"status": "unavailable"})
        )
        status, _, payload = harness.request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(payload)["ready"])

    def test_unexpected_runtime_error_does_not_kill_backend(self):
        runtime = FakeRuntime(
            Plan(exception=RuntimeError("boom")),
            Plan([[4]]),
        )
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body()
        )
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(payload)["error"]["code"], "runtime_error")
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(reasoning_effort="none"),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"],
            "plain answer\n",
        )

    def test_unexpected_handler_error_cancels_and_is_structured(self):
        harness = self.harness(FakeRuntime(Plan([[4]], block=True)))
        stderr = io.StringIO()
        with (
            mock.patch.object(
                harness.backend, "cancel", wraps=harness.backend.cancel
            ) as cancel,
            mock.patch.object(
                api.FrontendHandler, "_complete", side_effect=RuntimeError("boom")
            ),
            mock.patch.object(api.sys, "stderr", stderr),
        ):
            status, _, payload = harness.request(
                "POST", "/v1/chat/completions", self.body()
            )
        self.assertEqual(status, 500, payload)
        self.assertEqual(json.loads(payload)["error"]["code"], "internal_server_error")
        cancel.assert_called_once()
        self.assertRegex(
            stderr.getvalue(),
            r"^\d{2}:\d{2}:\d{2} Error · internal_server_error · RuntimeError\n$",
        )
        self.assertNotIn("boom", stderr.getvalue())

    def test_unexpected_responses_stream_error_is_failed_and_cancels(self):
        harness = self.harness(FakeRuntime(Plan([[4]], block=True)))
        stderr = io.StringIO()
        with (
            mock.patch.object(
                harness.backend, "cancel", wraps=harness.backend.cancel
            ) as cancel,
            mock.patch.object(
                api.FrontendHandler, "_collect", side_effect=RuntimeError("boom")
            ),
            mock.patch.object(api.sys, "stderr", stderr),
        ):
            status, _, payload = harness.request(
                "POST",
                "/v1/responses",
                self.responses_body(stream=True, reasoning={"effort": "none"}),
            )
        self.assertEqual(status, 200)
        events = self.response_events(payload)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(
            events[-1]["response"]["error"]["code"], "internal_server_error"
        )
        cancel.assert_called_once()
        self.assertRegex(
            stderr.getvalue(),
            r"^\d{2}:\d{2}:\d{2} Error · internal_server_error · RuntimeError\n$",
        )

    def test_streamer_end_error_does_not_kill_backend(self):
        runtime = FakeRuntime(Plan([[6]]), Plan([[4]]))
        harness = self.harness(runtime, tokenizer=EndFailTokenizer())
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body()
        )
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(payload)["error"]["code"], "runtime_error")
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            self.body(reasoning_effort="none"),
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["choices"][0]["message"]["content"],
            "plain answer\n",
        )

    def test_responses_nonstream_contract_and_reasoning(self):
        runtime = FakeRuntime(Plan([[1], [2], [3]]))
        harness = self.harness(runtime)
        body = self.responses_body(
            instructions="Be concise.",
            tool_choice="auto",
            parallel_tool_calls=True,
            reasoning={"effort": "medium", "summary": "auto", "context": "auto"},
            store=False,
            stream=False,
            stream_options={"reasoning_summary_delivery": "sequential_cutoff"},
            include=["reasoning.encrypted_content"],
            service_tier="auto",
            prompt_cache_key="turn-1",
            text={"verbosity": "low"},
            client_metadata={"origin": "test-client"},
            unknown_option={"future": True},
            max_output_tokens=8,
            seed=7,
        )
        status, content_type, payload = harness.request("POST", "/v1/responses", body)
        self.assertEqual((status, content_type), (200, "application/json"))
        response = json.loads(payload)
        self.assertTrue(response["id"].startswith("resp_"))
        self.assertEqual(
            (response["object"], response["status"]), ("response", "completed")
        )
        self.assertTrue(response["end_turn"])
        self.assertEqual(
            [item["type"] for item in response["output"]],
            ["reasoning", "message"],
        )
        self.assertEqual(response["output"][0]["summary"][0]["text"], "because ")
        self.assertEqual(response["output"][0]["content"][0]["text"], "because ")
        self.assertEqual(response["output"][1]["content"][0]["text"], "answer\n")
        self.assertEqual(
            response["usage"],
            {
                "input_tokens": 2,
                "input_tokens_details": {
                    "cached_tokens": 1,
                    "cache_write_tokens": 1,
                },
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 5,
            },
        )
        rendered, kwargs = harness.tokenizer.templates[-1]
        self.assertEqual(rendered[0], {"role": "system", "content": "Be concise."})
        self.assertEqual(rendered[1]["content"], "hello")
        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertEqual(
            (
                runtime.requests[0].logical_max_output_tokens,
                runtime.requests[0].seed,
            ),
            (8, 7),
        )

    def test_responses_stream_emits_protocol_events_and_usage(self):
        harness = self.harness(FakeRuntime(Plan([[1], [4]], reason="length")))
        status, content_type, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                stream=True,
                reasoning={"effort": "none"},
                store=False,
            ),
        )
        self.assertEqual((status, content_type), (200, "text/event-stream"))
        self.assertNotIn(b"[DONE]", payload)
        events = self.response_events(payload)
        kinds = [event["type"] for event in events]
        self.assertEqual(
            kinds,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.incomplete",
            ],
        )
        self.assertEqual(
            [
                event["delta"]
                for event in events
                if event["type"] == "response.output_text.delta"
            ],
            ["because ", "plain answer\n"],
        )
        incomplete = events[-1]["response"]
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertEqual(
            incomplete["incomplete_details"], {"reason": "max_output_tokens"}
        )
        self.assertFalse(incomplete["end_turn"])
        self.assertEqual(incomplete["usage"]["output_tokens"], 2)
        message = next(
            item for item in incomplete["output"] if item["type"] == "message"
        )
        self.assertEqual(message["status"], "incomplete")
        self.assertEqual(events[0]["response"]["id"], incomplete["id"])
        added = next(
            event["item"]
            for event in events
            if event["type"] == "response.output_item.added"
        )
        done = next(
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
        )
        self.assertEqual(added["id"], done["id"])
        self.assertEqual(added["content"], [])
        text_done = next(
            event for event in events if event["type"] == "response.output_text.done"
        )
        part_done = next(
            event for event in events if event["type"] == "response.content_part.done"
        )
        self.assertEqual(text_done["text"], "because plain answer\n")
        self.assertEqual(text_done["logprobs"], [])
        self.assertEqual(part_done["part"], done["content"][0])
        self.assertEqual(
            {
                (event.get("item_id"), event.get("output_index"))
                for event in events
                if "item_id" in event
            },
            {(done["id"], 0)},
        )
        self.assertEqual(
            [event["sequence_number"] for event in events], list(range(len(events)))
        )

    def test_responses_nonstream_length_is_incomplete_even_for_partial_json(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        harness = self.harness(
            FakeRuntime(Plan([[12]], reason="length")),
            constraint_factory=FakeConstraintFactory(),
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                reasoning={"effort": "none"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "answer",
                        "schema": schema,
                        "strict": True,
                    }
                },
            ),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(response["status"], "incomplete")
        self.assertEqual(
            response["incomplete_details"], {"reason": "max_output_tokens"}
        )
        self.assertFalse(response["end_turn"])
        self.assertEqual(response["output"][0]["status"], "incomplete")
        self.assertEqual(response["output"][0]["content"][0]["text"], '{"x":')

    def test_responses_length_marks_a_streamed_tool_call_incomplete(self):
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        harness = self.harness(
            FakeRuntime(Plan([[5]], reason="length"), Plan([[5]], reason="length"))
        )
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                tools=[tool],
                reasoning={"effort": "none"},
            ),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(response["status"], "incomplete")
        self.assertEqual(
            [item["type"] for item in response["output"]], ["function_call"]
        )
        self.assertEqual(response["output"][0]["status"], "incomplete")
        self.assertNotIn("<tool_call>", payload.decode())

        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                stream=True,
                tools=[tool],
                reasoning={"effort": "none"},
            ),
        )
        events = self.response_events(payload)
        self.assertEqual(status, 200, payload)
        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertNotIn("<tool_call>", payload.decode())
        calls = [
            event["item"]
            for event in events
            if event.get("item", {}).get("type") == "function_call"
        ]
        self.assertTrue(calls)
        self.assertEqual(calls[-1]["status"], "incomplete")
        self.assertEqual(
            [item["type"] for item in events[-1]["response"]["output"]],
            ["function_call"],
        )

    def test_responses_streams_reasoning_summary_before_answer(self):
        harness = self.harness(FakeRuntime(Plan([[1], [2], [3]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                stream=True,
                reasoning={"effort": "xhigh", "summary": "auto"},
            ),
        )
        self.assertEqual(status, 200)
        events = self.response_events(payload)
        kinds = [event["type"] for event in events]
        self.assertEqual(
            kinds,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        summary_delta = next(
            event
            for event in events
            if event["type"] == "response.reasoning_summary_text.delta"
        )
        self.assertEqual(summary_delta["delta"], "because ")
        self.assertLess(
            kinds.index("response.reasoning_summary_text.delta"),
            kinds.index("response.output_text.delta"),
        )
        self.assertNotIn("response.reasoning_text.delta", kinds)
        self.assertNotIn("response.reasoning_text.done", kinds)
        summary_done = events[5]
        self.assertEqual(summary_done["text"], "because ")
        self.assertEqual(
            events[6]["part"], {"type": "summary_text", "text": "because "}
        )
        reasoning_id = events[2]["item"]["id"]
        message_id = events[8]["item"]["id"]
        self.assertEqual(
            [(event["output_index"], event.get("item_id")) for event in events[2:8]],
            [
                (0, None),
                (0, reasoning_id),
                (0, reasoning_id),
                (0, reasoning_id),
                (0, reasoning_id),
                (0, None),
            ],
        )
        self.assertEqual(
            [(event["output_index"], event.get("item_id")) for event in events[8:14]],
            [
                (1, None),
                (1, message_id),
                (1, message_id),
                (1, message_id),
                (1, message_id),
                (1, None),
            ],
        )
        completed = events[-1]["response"]
        self.assertEqual(
            [item["type"] for item in completed["output"]],
            ["reasoning", "message"],
        )
        self.assertEqual(completed["output"][0]["summary"][0]["text"], "because ")
        self.assertEqual(completed["output"][1]["content"][0]["text"], "answer\n")
        self.assertEqual(
            [event["sequence_number"] for event in events], list(range(len(events)))
        )
        self.assertEqual(
            {events[0]["response"]["id"], events[1]["response"]["id"], completed["id"]},
            {completed["id"]},
        )

    def test_responses_reasoning_length_marks_item_and_part_incomplete(self):
        harness = self.harness(FakeRuntime(Plan([[1]], reason="length")))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                stream=True,
                reasoning={"effort": "xhigh", "summary": "auto"},
            ),
        )
        self.assertEqual(status, 200, payload)
        events = self.response_events(payload)
        part_done = next(
            event
            for event in events
            if event["type"] == "response.reasoning_summary_part.done"
        )
        item_done = next(
            event
            for event in events
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "reasoning"
        )
        terminal = events[-1]
        self.assertEqual(part_done["status"], "incomplete")
        self.assertEqual(item_done["item"]["status"], "incomplete")
        self.assertEqual(terminal["type"], "response.incomplete")
        self.assertEqual(terminal["response"]["output"][0]["status"], "incomplete")

    def test_responses_reasoning_status_tracks_the_truncated_phase(self):
        answer_harness = self.harness(
            FakeRuntime(
                Plan([[1], [2], [3]], reason="length"),
                Plan([[1], [2], [3]], reason="length"),
            )
        )
        for stream in (False, True):
            status, _, payload = answer_harness.request(
                "POST",
                "/v1/responses",
                self.responses_body(
                    stream=stream,
                    reasoning={"effort": "xhigh", "summary": "auto"},
                ),
            )
            self.assertEqual(status, 200, payload)
            response = (
                self.response_events(payload)[-1]["response"]
                if stream
                else json.loads(payload)
            )
            self.assertEqual(response["status"], "incomplete")
            self.assertEqual(
                [(item["type"], item["status"]) for item in response["output"]],
                [("reasoning", "completed"), ("message", "incomplete")],
            )

        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        tool_harness = self.harness(
            FakeRuntime(
                Plan([[1], [2], [25], [13]], reason="length"),
                Plan([[1], [2], [25], [13]], reason="length"),
            )
        )
        for stream in (False, True):
            status, _, payload = tool_harness.request(
                "POST",
                "/v1/responses",
                self.responses_body(
                    stream=stream,
                    tools=[tool],
                    reasoning={"effort": "xhigh", "summary": "auto"},
                ),
            )
            self.assertEqual(status, 200, payload)
            response = (
                self.response_events(payload)[-1]["response"]
                if stream
                else json.loads(payload)
            )
            expected_items = [
                ("reasoning", "completed"),
                ("function_call", "incomplete"),
            ]
            self.assertEqual(
                [(item["type"], item["status"]) for item in response["output"]],
                expected_items,
            )
            self.assertNotIn("<tool_call>", payload.decode())

    def test_openai_sdk_parses_responses_reasoning_lifecycle(self):
        harness = self.harness(FakeRuntime(Plan([[1], [2], [3]])))
        with self.openai_client(harness) as client:
            events = list(
                client.responses.create(
                    model="test-model",
                    input="hello",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "medium", "summary": "auto"},
                )
            )
        self.assertEqual(
            [type(event).__name__ for event in events],
            [
                "ResponseCreatedEvent",
                "ResponseInProgressEvent",
                "ResponseOutputItemAddedEvent",
                "ResponseReasoningSummaryPartAddedEvent",
                "ResponseReasoningSummaryTextDeltaEvent",
                "ResponseReasoningSummaryTextDoneEvent",
                "ResponseReasoningSummaryPartDoneEvent",
                "ResponseOutputItemDoneEvent",
                "ResponseOutputItemAddedEvent",
                "ResponseContentPartAddedEvent",
                "ResponseTextDeltaEvent",
                "ResponseTextDoneEvent",
                "ResponseContentPartDoneEvent",
                "ResponseOutputItemDoneEvent",
                "ResponseCompletedEvent",
            ],
        )
        self.assertEqual(
            [event.sequence_number for event in events], list(range(len(events)))
        )
        self.assertEqual(events[5].text, "because ")
        self.assertEqual(events[11].text, "answer\n")
        self.assertIsNone(events[6].status)
        self.assertEqual(events[7].item.status, "completed")
        self.assertEqual(events[-1].response.status, "completed")
        self.assertEqual(
            events[-1].response.usage.output_tokens_details.reasoning_tokens, 3
        )
        self.assertEqual(
            events[-1].response.usage.input_tokens_details.cached_tokens, 1
        )
        self.assertEqual(
            events[0].response.id,
            events[1].response.id,
        )
        self.assertEqual(events[1].response.id, events[-1].response.id)

        incomplete_harness = self.harness(FakeRuntime(Plan([[4]], reason="length")))
        with self.openai_client(incomplete_harness) as client:
            incomplete = list(
                client.responses.create(
                    model="test-model",
                    input="hello",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "none"},
                )
            )[-1]
        self.assertEqual(type(incomplete).__name__, "ResponseIncompleteEvent")
        self.assertEqual(incomplete.response.status, "incomplete")
        self.assertEqual(
            incomplete.response.incomplete_details.reason, "max_output_tokens"
        )

        reasoning_harness = self.harness(FakeRuntime(Plan([[1]], reason="length")))
        with self.openai_client(reasoning_harness) as client:
            reasoning_events = list(
                client.responses.create(
                    model="test-model",
                    input="hello",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "xhigh", "summary": "auto"},
                )
            )
        part_done = next(
            event
            for event in reasoning_events
            if event.type == "response.reasoning_summary_part.done"
        )
        item_done = next(
            event
            for event in reasoning_events
            if event.type == "response.output_item.done"
            and event.item.type == "reasoning"
        )
        self.assertEqual(part_done.status, "incomplete")
        self.assertEqual(item_done.item.status, "incomplete")
        self.assertEqual(reasoning_events[-1].response.status, "incomplete")

        phase_harness = self.harness(
            FakeRuntime(Plan([[1], [2], [3]], reason="length"))
        )
        with self.openai_client(phase_harness) as client:
            phase_events = list(
                client.responses.create(
                    model="test-model",
                    input="hello",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "xhigh", "summary": "auto"},
                )
            )
        self.assertEqual(
            [(item.type, item.status) for item in phase_events[-1].response.output],
            [("reasoning", "completed"), ("message", "incomplete")],
        )

    def test_openai_sdk_parses_function_call_arguments_done(self):
        harness = self.harness(FakeRuntime(Plan([[5]])))
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        with self.openai_client(harness) as client:
            events = list(
                client.responses.create(
                    model="test-model",
                    input="weather",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "none"},
                    tools=[tool],
                )
            )
        event_types = [event.type for event in events]
        self.assertEqual(event_types[:2], ["response.created", "response.in_progress"])
        self.assertEqual(event_types[-1], "response.completed")
        self.assertLess(
            event_types.index("response.function_call_arguments.delta"),
            event_types.index("response.function_call_arguments.done"),
        )
        arguments_done = next(
            event
            for event in events
            if event.type == "response.function_call_arguments.done"
        )
        call_done = next(
            event.item
            for event in events
            if event.type == "response.output_item.done"
            and event.item.type == "function_call"
        )
        self.assertEqual(
            type(arguments_done).__name__, "ResponseFunctionCallArgumentsDoneEvent"
        )
        self.assertEqual(
            (
                arguments_done.item_id,
                arguments_done.output_index,
                arguments_done.name,
                arguments_done.arguments,
            ),
            (call_done.id, 0, "weather", '{"city":"Paris"}'),
        )
        self.assertEqual(events[-1].response.status, "completed")

        failed_harness = self.harness(FakeRuntime(Plan([[8]])))
        with self.openai_client(failed_harness) as client:
            failed = list(
                client.responses.create(
                    model="test-model",
                    input="weather",
                    temperature=0,
                    stream=True,
                    store=False,
                    reasoning={"effort": "none"},
                    tools=[tool],
                )
            )[-1]
        self.assertEqual(type(failed).__name__, "ResponseFailedEvent")
        self.assertEqual(failed.response.status, "failed")
        self.assertEqual(failed.response.error.code, "invalid_model_output")

    def test_responses_tool_history_named_choice_and_output(self):
        runtime = FakeRuntime(Plan([[5]]))
        harness = self.harness(runtime)
        history = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "weather"}],
                "phase": "final_answer",
            },
            {
                "type": "reasoning",
                "id": "rs_old",
                "summary": [{"type": "summary_text", "text": "Use weather."}],
                "encrypted_content": None,
            },
            {
                "type": "function_call",
                "id": "fc_old",
                "call_id": "call_old",
                "name": "weather",
                "arguments": '{"city":"Rome"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call_old",
                "output": [{"type": "input_text", "text": "sunny"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "again"}],
            },
        ]
        tools = [
            {
                "type": "function",
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                "strict": True,
            },
            {"type": "function", "name": "time", "parameters": {}},
        ]
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                input=history,
                tools=tools,
                tool_choice={"type": "function", "name": "weather"},
                parallel_tool_calls=False,
                reasoning={"effort": "none"},
            ),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertFalse(response["end_turn"])
        self.assertFalse(response["parallel_tool_calls"])
        call = response["output"][-1]
        self.assertEqual(call["type"], "function_call")
        self.assertTrue(call["id"].startswith("fc_"))
        self.assertTrue(call["call_id"].startswith("call_"))
        self.assertEqual(
            (call["name"], call["arguments"]),
            ("weather", '{"city":"Paris"}'),
        )
        rendered, kwargs = harness.tokenizer.templates[-1]
        self.assertEqual(rendered[0], {"role": "user", "content": "weather"})
        self.assertEqual(
            [tool["function"]["name"] for tool in kwargs["tools"]], ["weather", "time"]
        )
        self.assertEqual(rendered[1]["reasoning_content"], "Use weather.")
        self.assertEqual(rendered[2]["tool_call_id"], "call_old")
        self.assertEqual(rendered[2]["content"], "sunny")
        self.assertEqual(kwargs["tools"][0]["function"]["name"], "weather")
        self.assertTrue(kwargs["tools"][0]["function"]["strict"])

    def test_responses_accepts_shorthand_messages_and_retained_history(self):
        harness = self.harness(FakeRuntime())
        with self.openai_client(harness) as client:
            first = client.responses.create(
                model="test-model",
                input=[{"role": "user", "content": "hello"}],
                reasoning={"effort": "none"},
            )
            second = client.responses.create(
                model="test-model",
                previous_response_id=first.id,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "again"}],
                    }
                ],
                reasoning={"effort": "none"},
            )
        self.assertEqual((first.status, second.status), ("completed", "completed"))
        messages, _ = harness.tokenizer.templates[-1]
        self.assertEqual(
            [(item["role"], item["content"]) for item in messages],
            [("user", "hello"), ("assistant", "plain answer\n"), ("user", "again")],
        )

    def test_responses_rejects_hosted_tools_without_native_work(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for tool in (
            {"type": "web_search"},
            {"type": "web_search", "external_web_access": False},
            {"type": "web_search", "external_web_access": True},
            {"type": "web_search_preview"},
            {"type": "file_search"},
        ):
            for stream in (False, True):
                with self.subTest(tool=tool, stream=stream):
                    status, _, payload = harness.request(
                        "POST",
                        "/v1/responses",
                        self.responses_body(tools=[tool], stream=stream),
                    )
                    self.assertEqual(status, 400, payload)
                    self.assertEqual(
                        json.loads(payload)["error"]["code"], "invalid_request_error"
                    )
        self.assertEqual(runtime.requests, [])
        self.assertEqual(harness.tokenizer.templates, [])

    def test_responses_supports_namespaces(self):
        tools = [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "required": ["message"],
                        },
                    }
                ],
            },
        ]
        runtime = FakeRuntime(Plan([[1]]))
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                tools=tools,
                tool_choice="auto",
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200, payload)
        _, template = harness.tokenizer.templates[-1]
        self.assertEqual(
            [tool["function"]["name"] for tool in template["tools"]],
            ["exec_command", "multi_agent_v1__spawn_agent"],
        )

        before = len(runtime.requests)
        named = dict(self.responses_body())
        named["tools"] = tools
        named["tool_choice"] = {"type": "web_search"}
        status, _, payload = harness.request("POST", "/v1/responses", named)
        self.assertEqual(status, 400, payload)

        namespace_runtime = FakeRuntime(Plan([[11]]), Plan([[4]]))
        namespace_harness = self.harness(namespace_runtime)
        namespace_body = self.responses_body(
            stream=True,
            tools=tools,
            tool_choice="auto",
            reasoning={"effort": "none"},
        )
        status, _, payload = namespace_harness.request(
            "POST", "/v1/responses", namespace_body
        )
        self.assertEqual(status, 200, payload)
        events = self.response_events(payload)
        call = next(
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "function_call"
        )
        self.assertEqual(
            (call["namespace"], call["name"], call["arguments"]),
            ("multi_agent_v1", "spawn_agent", '{"message":"inspect"}'),
        )
        followup = dict(namespace_body)
        followup["stream"] = False
        followup["input"] = [
            *namespace_body["input"],
            call,
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": "done",
            },
        ]
        status, _, payload = namespace_harness.request(
            "POST", "/v1/responses", followup
        )
        self.assertEqual(status, 200, payload)
        rendered, _ = namespace_harness.tokenizer.templates[-1]
        self.assertEqual(
            rendered[-2]["tool_calls"][0]["function"]["name"],
            "multi_agent_v1__spawn_agent",
        )
        self.assertEqual(rendered[-1]["content"], "done")
        self.assertEqual(len(runtime.requests), before)

        bad = dict(self.responses_body())
        bad["tools"] = [{"type": "computer_use_preview"}]
        status, _, payload = harness.request("POST", "/v1/responses", bad)
        self.assertEqual(status, 400, payload)

    def test_responses_shortens_long_namespace_aliases_without_collisions(self):
        namespace = "n" * 64
        first_name = "same_prefix_" + "a" * 51
        second_name = "same_prefix_" + "a" * 50 + "b"
        tools = [
            {
                "type": "namespace",
                "name": namespace,
                "tools": [
                    {"type": "function", "name": first_name, "parameters": {}},
                    {"type": "function", "name": second_name, "parameters": {}},
                ],
            }
        ]
        translated = api_shapes.responses_to_chat_body(
            self.responses_body(
                tools=tools,
                tool_choice={
                    "type": "function",
                    "namespace": namespace,
                    "name": first_name,
                },
            )
        )
        aliases = [tool["function"]["name"] for tool in translated["tools"]]
        self.assertEqual([len(alias) for alias in aliases], [64, 64])
        self.assertNotEqual(aliases[0], aliases[1])
        self.assertEqual(
            translated["tool_choice"],
            {"type": "function", "function": {"name": aliases[0]}},
        )
        self.assertEqual(
            translated["_tool_namespaces"][aliases[0]],
            (namespace, first_name),
        )
        self.assertEqual(_namespace_alias(namespace, first_name), aliases[0])

    def test_responses_streaming_tool_call_and_failed_event(self):
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                "required": ["city"],
            },
        }
        harness = self.harness(FakeRuntime(Plan([[5]]), Plan([[8]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                stream=True,
                tools=[tool],
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200)
        events = self.response_events(payload)
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[:2], ["response.created", "response.in_progress"])
        self.assertEqual(kinds[-1], "response.completed")
        added_index, added_output_index, added = next(
            (index, event["output_index"], event["item"])
            for index, event in enumerate(events)
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "function_call"
        )
        delta_indexes = [
            index
            for index, event in enumerate(events)
            if event["type"] == "response.function_call_arguments.delta"
        ]
        arguments_done_index, arguments_done = next(
            (index, event)
            for index, event in enumerate(events)
            if event["type"] == "response.function_call_arguments.done"
        )
        item_done_index, done = next(
            (index, event["item"])
            for index, event in enumerate(events)
            if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "function_call"
        )
        self.assertEqual(added["arguments"], "")
        self.assertEqual(
            (done["name"], done["arguments"]),
            ("weather", '{"city":"Paris"}'),
        )
        self.assertEqual(
            "".join(events[index]["delta"] for index in delta_indexes),
            done["arguments"],
        )
        self.assertEqual(
            (
                arguments_done["item_id"],
                arguments_done["output_index"],
                arguments_done["name"],
                arguments_done["arguments"],
            ),
            (
                done["id"],
                added_output_index,
                done["name"],
                done["arguments"],
            ),
        )
        self.assertEqual(added["id"], done["id"])
        self.assertTrue(added_index < delta_indexes[0] < arguments_done_index)
        self.assertLess(arguments_done_index, item_done_index)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertNotIn("<tool_call>", payload.decode())
        self.assertEqual(
            [item["type"] for item in events[-1]["response"]["output"]],
            ["function_call"],
        )
        self.assertEqual(
            [event["sequence_number"] for event in events], list(range(len(events)))
        )

        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                stream=True,
                tools=[tool],
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200)
        events = self.response_events(payload)
        self.assertEqual(
            [event["type"] for event in events],
            ["response.created", "response.in_progress", "response.failed"],
        )
        self.assertEqual(events[-1]["response"]["status"], "failed")
        self.assertEqual(
            events[-1]["response"]["error"]["code"], "invalid_model_output"
        )
        self.assertEqual([event["sequence_number"] for event in events], [0, 1, 2])
        self.assertNotIn("<tool_call>", payload.decode())
        self.assertNotIn("response.completed", [event["type"] for event in events])
        self.assertFalse(
            any(
                event["type"].startswith("response.function_call_arguments")
                or event.get("item", {}).get("type") == "function_call"
                for event in events
            )
        )

    def test_responses_output_items_carry_distinct_ids(self):
        tool = {
            "type": "function",
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                "required": ["city"],
            },
        }
        # Text, a tool call, then the template newline that follows the call.
        harness = self.harness(FakeRuntime(Plan([[4, 5]]), Plan([[1, 2, 3, 5]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                stream=True,
                tools=[tool],
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200)
        events = self.response_events(payload)
        response = events[-1]["response"]
        public_id = response["id"].removeprefix("resp_")
        ids = [item["id"] for item in response["output"]]
        self.assertEqual(
            ids, [f"msg_{public_id}_0", f"fc_{public_id}_1", f"msg_{public_id}_2"]
        )
        ids_by_index = {}
        for event in events:
            if "output_index" in event:
                item_id = event["item"]["id"] if "item" in event else event["item_id"]
                ids_by_index.setdefault(event["output_index"], set()).add(item_id)
        self.assertEqual(
            ids_by_index, {index: {item_id} for index, item_id in enumerate(ids)}
        )
        status, _, stored = harness.request("GET", f"/v1/responses/{response['id']}")
        self.assertEqual(status, 200)
        self.assertEqual([item["id"] for item in json.loads(stored)["output"]], ids)

        status, _, payload = harness.request(
            "POST", "/v1/responses", self.responses_body(tools=[tool])
        )
        self.assertEqual(status, 200)
        response = json.loads(payload)
        public_id = response["id"].removeprefix("resp_")
        self.assertEqual(
            [item["id"] for item in response["output"]],
            [f"rs_{public_id}_0", f"msg_{public_id}_1", f"fc_{public_id}_2"],
        )

    def test_responses_structured_text_uses_decode_constraint(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        factory = FakeConstraintFactory()
        harness = self.harness(FakeRuntime(Plan([[10]])), constraint_factory=factory)
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                reasoning={"effort": "none"},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "answer",
                        "schema": schema,
                        "strict": True,
                    }
                },
            ),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(response["output"][0]["content"][0]["text"], '{"x":3}')
        self.assertEqual(response["text"]["format"]["schema"], schema)
        self.assertIn("%json", factory.grammars[0])

    def test_responses_rejects_unsupported_features_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        cases = (
            ({"store": "yes"}, "store must be a boolean"),
            ({"previous_response_id": ""}, "previous_response_id"),
            ({"conversation": "conv_old"}, "conversation"),
            ({"background": True}, "background"),
            (
                {"text": {"format": {"type": "json_schema"}}},
                "response_format.json_schema.schema is required",
            ),
            ({"tools": [{"type": "custom", "name": "shell"}]}, "only function"),
            (
                {
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": "x"}],
                        }
                    ]
                },
                "only data: image URLs",
            ),
        )
        for extra, message in cases:
            with self.subTest(extra=extra):
                status, _, payload = harness.request(
                    "POST", "/v1/responses", self.responses_body(**extra)
                )
                self.assertEqual(status, 400)
                self.assertIn(message, json.loads(payload)["error"]["message"])
        self.assertEqual(runtime.requests, [])

    def test_responses_store_retrieve_delete_and_store_false(self):
        runtime = FakeRuntime(Plan([[4]]), Plan([[4]]))
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(previous_response_id="resp_missing"),
        )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(payload)["error"]["code"], "not_found_error")
        self.assertEqual(runtime.requests, [])

        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(reasoning={"effort": "none"}),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertIs(response["store"], True)
        response_id = response["id"]

        status, _, payload = harness.request(
            "GET", f"/v1/responses/{response_id}?include=reasoning.encrypted_content"
        )
        self.assertEqual((status, json.loads(payload)), (200, response))
        status, _, payload = harness.request("GET", "/status")
        store_status = json.loads(payload)["response_store"]
        self.assertEqual(store_status["entries"], 1)
        self.assertLessEqual(store_status["bytes"], store_status["budget_bytes"])

        status, _, payload = harness.request("DELETE", f"/v1/responses/{response_id}")
        self.assertEqual(
            (status, json.loads(payload)),
            (200, {"id": response_id, "object": "response", "deleted": True}),
        )
        status, _, _ = harness.request("GET", f"/v1/responses/{response_id}")
        self.assertEqual(status, 404)

        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(store=False, reasoning={"effort": "none"}),
        )
        response = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertIs(response["store"], False)
        status, _, _ = harness.request("GET", f"/v1/responses/{response['id']}")
        self.assertEqual(status, 404)

    def test_response_store_is_strictly_byte_bounded_lru(self):
        store = request_frontend.ResponseStore(2048)
        for index in range(10):
            response = {
                "id": f"resp_{index}",
                "object": "response",
                "output": [{"type": "message", "text": "x" * 128}],
            }
            self.assertTrue(store.put(response, [{"text": "y" * 128}]))
        stats = store.stats()
        self.assertLessEqual(stats["bytes"], stats["budget_bytes"])
        self.assertGreater(stats["evictions"], 0)
        self.assertIsNone(store.get("resp_0"))
        self.assertIsNotNone(store.get("resp_9"))
        self.assertFalse(
            request_frontend.ResponseStore(64).put(
                {"id": "resp_large", "text": "x" * 128}, []
            )
        )

    def test_response_store_concurrent_churn_remains_bounded(self):
        store = request_frontend.ResponseStore(8192)
        errors = []

        def churn(shard):
            try:
                for index in range(100):
                    response_id = f"resp_{shard}_{index}"
                    self.assertTrue(
                        store.put(
                            {"id": response_id, "text": "x" * 128},
                            [{"shard": shard, "index": index}],
                        )
                    )
                    store.get(response_id)
                    if index % 3 == 0:
                        store.delete(response_id)
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=churn, args=(shard,)) for shard in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        stats = store.stats()
        self.assertLessEqual(stats["bytes"], stats["budget_bytes"])
        self.assertGreater(stats["evictions"], 0)

    def test_previous_response_replays_history_but_not_old_instructions(self):
        tokenizer = FakeTokenizer()
        runtime = FakeRuntime(Plan([[4]]), Plan([[4]]))
        harness = self.harness(runtime, tokenizer=tokenizer)
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                instructions="old instruction",
                reasoning={"effort": "none"},
            ),
        )
        first = json.loads(payload)
        self.assertEqual(status, 200)

        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                input="follow up",
                instructions="new instruction",
                previous_response_id=first["id"],
                store=False,
                reasoning={"effort": "none"},
            ),
        )
        second = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(second["previous_response_id"], first["id"])
        rendered, _ = tokenizer.templates[-1]
        self.assertEqual(
            rendered,
            [
                {"role": "system", "content": "new instruction"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "plain answer\n"},
                {"role": "user", "content": "follow up"},
            ],
        )

    def test_responses_system_items_keep_their_input_position(self):
        messages = normalize_responses_input(
            "Be concise.",
            [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": "Answer in French.",
                },
                {"type": "message", "role": "user", "content": "again"},
            ],
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "system", "content": "Answer in French."},
                {"role": "user", "content": "again"},
            ],
        )

    def test_streaming_response_is_stored_only_at_terminal_event(self):
        harness = self.harness(FakeRuntime(Plan([[4]])))
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(stream=True, reasoning={"effort": "none"}),
        )
        self.assertEqual(status, 200)
        events = self.response_events(payload)
        terminal = events[-1]["response"]
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertIs(terminal["store"], True)
        status, _, payload = harness.request("GET", f"/v1/responses/{terminal['id']}")
        self.assertEqual((status, json.loads(payload)), (200, terminal))

    def test_previous_response_preserves_function_call_and_output(self):
        tokenizer = FakeTokenizer()
        runtime = FakeRuntime(Plan([[5]]), Plan([[4]]))
        harness = self.harness(runtime, tokenizer=tokenizer)
        tools = [
            {
                "type": "function",
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                tools=tools,
                reasoning={"effort": "none"},
            ),
        )
        first = json.loads(payload)
        self.assertEqual(status, 200)
        call = next(item for item in first["output"] if item["type"] == "function_call")
        status, _, _ = harness.request(
            "POST",
            "/v1/responses",
            self.responses_body(
                previous_response_id=first["id"],
                tools=tools,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": "sunny",
                    }
                ],
                store=False,
                reasoning={"effort": "none"},
            ),
        )
        self.assertEqual(status, 200)
        rendered, _ = tokenizer.templates[-1]
        self.assertEqual(rendered[-2]["role"], "assistant")
        self.assertEqual(
            rendered[-2]["tool_calls"][0]["function"],
            {"name": "weather", "arguments": {"city": "Paris"}},
        )
        self.assertEqual(
            rendered[-1],
            {"role": "tool", "tool_call_id": call["call_id"], "content": "sunny"},
        )


class MessageNormalizationTest(unittest.TestCase):
    def test_unfinished_tool_call_arguments_stay_history_text(self):
        # A tool call cut short by the client's output limit comes back as
        # history. The model must see the text it produced; rejecting the
        # request would leave the agent unable to continue at all.
        truncated = '{"command": "cat <<EOF > archive/batch4.jsonl\\n{\\"id\\": 1'
        messages = api_shapes.normalize_messages(
            [
                {"role": "user", "content": "save it"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": truncated},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "interrupted"},
                {"role": "user", "content": "continue"},
            ]
        )
        call = messages[1]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "shell")
        self.assertEqual(call["arguments"], truncated)
        complete = api_shapes.normalize_messages(
            [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"a": 1}'},
                        }
                    ],
                },
            ]
        )
        self.assertEqual(
            complete[1]["tool_calls"][0]["function"]["arguments"], {"a": 1}
        )
        for malformed in ('{"x": 1,}', '{"x": NaN}', "not json"):
            with self.subTest(arguments=malformed), self.assertRaises(api.APIError):
                api_shapes.normalize_messages(
                    [
                        {"role": "user", "content": "x"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "shell",
                                        "arguments": malformed,
                                    },
                                }
                            ],
                        },
                    ]
                )
        with self.assertRaises(api.APIError):
            api_shapes.normalize_messages(
                [
                    {"role": "user", "content": "x"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "shell", "arguments": ["list"]},
                            }
                        ],
                    },
                ]
            )


if __name__ == "__main__":
    unittest.main()

"""What the server tests share: fakes of the native engine, its process, the
tokenizer and the grammar compiler; the harness that serves a frontend over
them; request bodies and response readers for each API; and the PDFs and
launcher arguments several tests build."""

import base64
import http.client
import inspect
import io
import json
import math
import os
import signal
import subprocess
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from llguidance import LLTokenizer
from openai import OpenAI
from openai._streaming import SSEDecoder
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from dev.tests import native_peer
from install import launcher
from server import api_shapes, documents, images, serve_options
from server import backend as backend_api
from server import errors as api_errors
from server import frontend as request_frontend
from server import protocol as wire
from server import runtime as engine_runtime
from server import server as api
from server.chat_templates import ChatTemplates
from server.lru import LRUCache
from server.origins import parse_allowed_origin
from server.thinking import ThinkingCodec

# The deadline of a request prepared directly, which never expires.
FOREVER = math.inf
READY = wire.ReadyEvent(4, 131_072, False)
# The generation APIs: Chat Completions, Responses and Messages.
PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages")


def stays_connected():
    """Whether the client of a request prepared directly has left: never."""
    return False


def reserve_unbounded(size):
    """Reserve `size` more input bytes for a request prepared directly, which
    no input budget bounds."""


def byte_alphabet():
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


def byte_backend(fragments):
    alphabet = byte_alphabet()
    vocabulary = {}
    for token_id in range(max(fragments) + 1):
        value = fragments.get(token_id, f"<unused_{token_id}>")
        token = "".join(alphabet[byte] for byte in value.encode())
        vocabulary[token] = token_id
    vocabulary["[UNK]"] = len(vocabulary)
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.decoder = decoders.ByteLevel()
    return backend


def grammar_tokenizers():
    """A tokenizer with a token per byte, the call and thinking delimiters
    and <eos>, and its llguidance tokenizer, to compile and match grammars."""
    vocabulary = {char: token for token, char in byte_alphabet().items()}
    vocabulary["[UNK]"] = len(vocabulary)
    vocabulary["<eos>"] = len(vocabulary)
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            pre_tokenizers.Split(Regex(""), behavior="isolated"),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<tool_call>", "</tool_call>", "</think>", "<eos>"])
    guidance = LLTokenizer(tokenizer.to_str(), eos_token=tokenizer.token_to_id("<eos>"))
    return tokenizer, guidance


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
            28: (
                "<tool_call>\n<function=weather>\n<parameter=city>\n"
                "3\n</parameter>\n</function>\n</tool_call>\n"
            ),
        }
        self.backend_tokenizer = byte_backend(self.fragments)
        self.templates = []

    # Requests render as their generation prompt alone, a fixed prefix; the
    # source only has to be a template ChatTemplates can probe at startup.
    chat_template = (
        "{%- for message in messages %}"
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
    )

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        rendered = ""
        if kwargs.get("add_generation_prompt"):
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


class TemplateTokenizer(FakeTokenizer):
    """Render real HF/Jinja templates while keeping the test runtime's tokens."""

    def __init__(self, template):
        super().__init__()
        from transformers import PreTrainedTokenizerFast

        self.renderer = PreTrainedTokenizerFast(tokenizer_object=self.backend_tokenizer)
        self.renderer.chat_template = template

    @property
    def chat_template(self):
        return self.renderer.chat_template

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((messages, kwargs))
        return self.renderer.apply_chat_template(messages, **kwargs)


class CharTokenizer(FakeTokenizer):
    """One token per character, so single-letter answer slots are exact."""

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)


class ImagePadTokenizer(FakeTokenizer):
    """Renders one image placeholder per image part like the pinned
    Qwen template, with pad id 50."""

    # Stands for the template: rendering emits the source per image part,
    # so the frontend's image render marker appears where it replaced it.
    chat_template = api_shapes.IMAGE_PAD_TOKEN

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
        if kwargs.get("add_generation_prompt"):
            rendered += "<|im_start|>assistant\n<think>\n"
            if not kwargs.get("enable_thinking", True):
                rendered += "\n</think>\n\n"
        return (
            rendered if kwargs.get("tokenize") is False else self(rendered)["input_ids"]
        )

    def convert_tokens_to_ids(self, token):
        if token == api_shapes.IMAGE_PAD_TOKEN:
            return 50
        return super().convert_tokens_to_ids(token)


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
    def count_reasoning_tokens(_enabled, _tool_calls=False):
        return 0


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
        logits=None,
    ):
        self.batches = list(batches)
        self.reason = reason
        self.error = error
        self.exception = exception
        self.delay = delay
        self.matched_tokens = matched_tokens
        self.logits = logits
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
    """A RuntimeCall the fake engine runs by its plan or, without one, that
    the test drives through emit() and complete()."""

    def __init__(self, runtime, request_id, request, on_event, on_complete, plan):
        self.runtime = runtime
        self.request_id = request_id
        self.request = request
        self.on_event = on_event
        self.on_complete = on_complete
        self.plan = plan
        self.callback_error = None
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
        if self.on_event is None:
            return
        try:
            self.on_event(self, event)
        except Exception as error:
            if self.callback_error is None:
                self.callback_error = error
            self.cancel()

    def complete(self, *, result=None, error=None):
        with self._lock:
            if self._done:
                return False
            self._result = result
            self._error = error
            self._done = True
        if self.on_complete is not None:
            self.on_complete(self)
        return True

    def result(self, timeout=None):
        with self._lock:
            if not self._done:
                raise TimeoutError("fake runtime call is not complete")
            if self._error is not None:
                raise self._error
            return self._result

    def cancel(self):
        with self._lock:
            if self._done or self.cancel_requested:
                return False
            self.cancel_requested = True
        self.runtime.cancel_count += 1
        if self.plan is not None:
            self.plan.cancelled.set()
            self.plan.start_release.set()
            self.plan.release.set()
            self.plan.terminal_release.set()
        return True


class FakeRuntime:
    """MultiplexedRuntime's stand-in. Each call runs the next of `plans`, or
    Plan([[4]]) once they run out, on a thread of its own; with manual=True
    the test drives every call."""

    def __init__(self, *plans, manual=False):
        self.plans = list(plans)
        self.manual = manual
        self.requests = []
        self.calls = []
        self.cancel_count = 0
        self.closed = False
        self.ready = True
        self.readiness = READY
        self.fatal_error = None
        self.pending_limit = 32
        self.restart_count = 0
        self.last_crash_trace = None
        self.status_event = native_peer.status_event()
        self.threads = []

    @property
    def pending_count(self):
        return sum(not call.done for call in self.calls)

    def wait_ready(self, timeout=None):
        # The fake starts no engine: it is as ready as the test sets it.
        return self.ready

    def status(self, timeout=5.0, *, fail_unanswered=False):
        if timeout <= 0:
            raise ValueError("status timeout must be positive")
        if not self.ready:
            raise engine_runtime.EngineUnhealthy("native process is not ready")
        return self.status_event

    def submit(self, request, *, on_event=None, on_complete=None):
        if self.pending_count >= self.pending_limit:
            raise engine_runtime.PendingLimitExceeded("fake runtime is full")
        plan = None
        if not self.manual:
            plan = self.plans.pop(0) if self.plans else Plan([[4]])
        call = FakeCall(
            self,
            len(self.calls) + 1,
            request,
            on_event,
            on_complete,
            plan,
        )
        self.calls.append(call)
        self.requests.append(request)
        if plan is not None:
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
        call.emit(wire.StartEvent(call.request_id, 0, plan.matched_tokens))
        plan.started.set()
        plan.release.wait(2)
        if plan.exception:
            call.complete(error=plan.exception)
            return
        if plan.error:
            call.complete(error=api_errors.ConstraintError(plan.error))
            return
        completion = 0
        if not plan.cancelled.is_set():
            offset = 0
            for batch in plan.batches:
                if plan.cancelled.is_set():
                    break
                tokens = tuple(batch)
                call.emit(wire.TokensEvent(call.request_id, offset, tokens))
                offset += len(tokens)
                completion += len(tokens)
                plan.cancelled.wait(plan.delay)
        reason = (
            wire.FinishReason.CANCELLED
            if plan.cancelled.is_set()
            else {
                "stop": wire.FinishReason.STOP,
                "length": wire.FinishReason.LENGTH,
            }[plan.reason]
        )
        plan.terminal.set()
        plan.terminal_release.wait(2)
        if reason == wire.FinishReason.CANCELLED:
            completion = 0
        done = wire.DoneEvent(
            call.request_id,
            reason,
            len(request.frame.prompt_tokens),
            completion,
            1_000,
            2_000,
            3_000,
            tuple(plan.logits) if plan.logits is not None else (),
        )
        call.complete(result=done)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.ready = False
        for call in self.calls:
            call.cancel()
        for thread in self.threads:
            thread.join(2)

    def kill(self):
        # The engine is gone; its calls end as the test ends them.
        self.ready = False


class FakeOutput:
    def __init__(self):
        self._condition = threading.Condition()
        self._buffer = bytearray()
        self._closed = False

    def feed(self, data):
        with self._condition:
            if self._closed:
                raise BrokenPipeError("fake stdout is closed")
            self._buffer.extend(data)
            self._condition.notify_all()

    def read1(self, size):
        with self._condition:
            while not self._buffer and not self._closed:
                self._condition.wait()
            if not self._buffer:
                return b""
            count = min(size, len(self._buffer))
            result = bytes(self._buffer[:count])
            del self._buffer[:count]
            return result

    read = read1

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class FakeInput:
    def __init__(self, process):
        self._process = process
        self._condition = threading.Condition()
        self._reader = native_peer.ClientFrameReader()
        self._closed = False
        self.records = []
        self._read_fd, self._write_fd = os.pipe()

    def fileno(self):
        return self._write_fd

    def write(self, data):
        with self._condition:
            if self._closed:
                raise BrokenPipeError("fake stdin is closed")
            records = self._reader.feed(data)
            self.records.extend(records)
            self._condition.notify_all()
        for message, _raw in records:
            handler = self._process.input_handler
            if handler:
                handler(self._process, message)
        return len(data)

    def flush(self):
        return None

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            os.close(self._write_fd)
            os.close(self._read_fd)
            self._condition.notify_all()

    def messages(self, message_type):
        with self._condition:
            return [
                message
                for message, _raw in self.records
                if isinstance(message, message_type)
            ]

    def records_of(self, message_type):
        with self._condition:
            return [
                record for record in self.records if isinstance(record[0], message_type)
            ]

    def wait_for(self, message_type, count=1, timeout=1.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                matches = [
                    message
                    for message, _raw in self.records
                    if isinstance(message, message_type)
                ]
                if len(matches) >= count:
                    return matches
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"did not receive {count} {message_type.__name__} frames"
                    )
                self._condition.wait(remaining)


class FakeProcess:
    _pids = iter(range(50_000, 60_000))

    def __init__(self, input_handler=None, initial_output=None):
        self.pid = next(self._pids)
        self.input_handler = input_handler
        self.stdout = FakeOutput()
        self.stdin = FakeInput(self)
        self._condition = threading.Condition()
        self._exit_code = None
        if initial_output is None:
            self.send(READY)
        elif initial_output:
            self.stdout.feed(initial_output)

    def send(self, event):
        self.stdout.feed(native_peer.serialize_event(event))

    def close_stdout(self):
        self.stdout.close()

    def poll(self):
        with self._condition:
            return self._exit_code

    def terminate(self):
        self._exit(-15)

    def kill(self):
        self._exit(-9)

    def _exit(self, code):
        with self._condition:
            if self._exit_code is not None:
                return
            self._exit_code = code
            self._condition.notify_all()
        self.stdin.close()
        self.stdout.close()

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._exit_code is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise subprocess.TimeoutExpired("fake-native", timeout)
                self._condition.wait(remaining)
            return self._exit_code


class FakeFactory:
    def __init__(self, handler=None, initial_output=None):
        self.handler = handler
        self.initial_output = initial_output
        self.processes = []

    def __call__(self):
        process = FakeProcess(self.handler, self.initial_output)
        self.processes.append(process)
        return process


def request(
    token,
    *,
    prompt_tokens=None,
    logical_max_output_tokens=32,
    priority=wire.RequestPriority.NORMAL,
    deadline=45.0,
    sampling=None,
    seed=0,
    constraint=wire.ConstraintMode.NONE,
    mask_provider=None,
    image_owner=None,
    return_progress=False,
    score_tokens=(),
    generation_prompt_tokens=0,
):
    """A request whose deadline is `deadline` seconds from now."""
    frame = native_peer.request_frame(
        request_id=0,
        priority=priority,
        absolute_deadline_unix_micros=0,
        remaining_deadline_micros=0,
        logical_max_output_tokens=logical_max_output_tokens,
        prompt_tokens=prompt_tokens or (token, token + 1),
        sampling=sampling or wire.SamplingParameters(),
        seed=seed,
        constraint=constraint,
        score_tokens=score_tokens,
        flags=wire.RequestFlag.RETURN_PROGRESS
        if return_progress
        else wire.RequestFlag(0),
        generation_prompt_tokens=generation_prompt_tokens,
    )
    return engine_runtime.GenerationRequest(
        frame, time.monotonic() + deadline, mask_provider, image_owner
    )


def written_frame(process, call):
    return next(
        frame
        for frame in process.stdin.messages(wire.RequestFrame)
        if frame.request_id == call.request_id
    )


def send_success(process, call, *, lane=0, tokens=(10, 11, 12)):
    process.send(wire.StartEvent(call.request_id, lane, 0))
    process.send(wire.TokensEvent(call.request_id, 0, tokens[:2]))
    if tokens[2:]:
        process.send(wire.TokensEvent(call.request_id, 2, tokens[2:]))
    process.send(
        wire.DoneEvent(
            call.request_id,
            wire.FinishReason.STOP,
            len(written_frame(process, call).prompt_tokens),
            len(tokens),
            100,
            200,
            350,
            (),
        )
    )


class FakeConstraintFactory:
    def __init__(self):
        self.grammars = []

    def create(self, grammar, *, timeout=None, prefixes=None):
        self.grammars.append(grammar)
        return SimpleNamespace(commit=lambda _tokens: None, finish=lambda: None)

    def stats(self):
        return {}


class PassThroughConstraintFactory:
    """Leaves generation unconstrained, for tests that do not check grammars."""

    def create(self, grammar, *, timeout=None, prefixes=None):
        return None

    def stats(self):
        return {}


def main_args(**overrides):
    """Parsed command-line arguments for main() tests."""
    return SimpleNamespace(
        **{
            "model_root": "model",
            "tokenizer": "tokenizer",
            "model": "test-model",
            "served_model_name": [],
            "announce_served_name": False,
            "default_reasoning_effort": None,
            "max_context": None,
            "max_memory": None,
            "idle_release": None,
            "max_cache_disk": 0,
            "persistent_cache": False,
            "cache_dir": None,
            "decode_share": None,
            "disable_ane": False,
            "max_image_pixels": images.MAX_PIXELS,
            "request_timeout": None,
            "queue_size": 1,
            "host": "127.0.0.1",
            "allowed_host": [],
            "allowed_origin": [],
            "api_key": None,
            "allow_idle_sleep": False,
            "no_webui": False,
            "max_request_size": serve_options.DEFAULT_MAX_REQUEST_BYTES,
            "port": 0,
            "binary": "splash",
            "kv_format": "int8",
            **overrides,
        }
    )


def make_frontend(
    tokenizer, *args, constraint_factory=None, thinking_codec=None, **options
):
    """A frontend over the tokenizer, with its chat templates probed as
    startup probes them. Unless a test passes its own, generation is
    unconstrained and thinking is signed with a fresh key."""
    if constraint_factory is None:
        constraint_factory = PassThroughConstraintFactory()
    if thinking_codec is None:
        thinking_codec = ThinkingCodec()
    return request_frontend.Frontend(
        tokenizer,
        *args,
        constraint_factory=constraint_factory,
        chat_templates=ChatTemplates(tokenizer),
        thinking_codec=thinking_codec,
        **options,
    )


def no_signed_thinking(signature):
    """The thinking resolver for converted requests that carry no signature."""
    raise AssertionError(f"unexpected thinking signature {signature!r}")


def make_job(request_id=101, *, constraint=None, temperature=0.0):
    return backend_api.Job(
        request_id=request_id,
        prompt_tokens=[11, 12, 13, 14],
        max_new_tokens=37,
        seed=0x123456789ABCDEF0,
        sampling=wire.SamplingParameters(
            temperature,
            0.75,
            17,
            presence_penalty=1.5,
            frequency_penalty=-0.25,
            repetition_penalty=1.125,
            min_p=0.125,
        ),
        deadline=time.monotonic() + 10.0,
        priority=wire.RequestPriority.FOREGROUND,
        constraint=constraint,
    )


class Harness:
    def __init__(
        self,
        runtime,
        tokenizer=None,
        queue_size=4,
        timeout=2,
        max_context=128,
        model="test-model",
        request_logger=lambda _record: None,
        constraint_factory=None,
        io_timeout=None,
        thinking_codec=None,
        api_key=None,
        webui=True,
        max_request_bytes=serve_options.DEFAULT_MAX_REQUEST_BYTES,
        host="127.0.0.1",
        allowed_hosts=(),
        allowed_origins=(),
        vision=True,
        **frontend_options,
    ):
        self.tokenizer = tokenizer or FakeTokenizer()
        runtime.pending_limit = queue_size
        self.backend = backend_api.NativeBackend(
            runtime, self.tokenizer, request_logger=request_logger
        )
        self.app = make_frontend(
            self.tokenizer,
            self.backend,
            model,
            max_context,
            timeout,
            2,
            constraint_factory=constraint_factory,
            thinking_codec=thinking_codec,
            vision=vision,
            **frontend_options,
        )
        # The chat templates are probed once, before the frontend is built;
        # keep only request renders in a recording tokenizer.
        if isinstance(getattr(self.tokenizer, "templates", None), list):
            self.tokenizer.templates.clear()
        self.server = api.FrontendServer(
            (host, 0),
            self.app,
            request_capacity=queue_size,
            api_key=api_key,
            webui=webui,
            max_request_bytes=max_request_bytes,
            allowed_hosts=allowed_hosts,
            allowed_origins=map(parse_allowed_origin, allowed_origins),
        )
        # The server reads HTTP_IO_TIMEOUT per connection. A harness given its
        # own holds it until close(); patching only once setup has succeeded
        # never leaves it patched.
        self._io_timeout = None
        if io_timeout is not None:
            self._io_timeout = mock.patch.object(api, "HTTP_IO_TIMEOUT", io_timeout)
            self._io_timeout.start()
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
        if self._io_timeout is not None:
            self._io_timeout.stop()


class HarnessTestCase(unittest.TestCase):
    """A test case whose harnesses close when its test ends."""

    def harness(self, runtime=None, **kwargs):
        harness = Harness(runtime or FakeRuntime(), **kwargs)
        self.addCleanup(harness.close)
        return harness

    def _wait_for_http_active(self, admission, expected):
        deadline = time.monotonic() + 1
        while admission.stats()["active"] != expected and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(admission.stats()["active"], expected)


def chat_body(**extra):
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }
    body.update(extra)
    return body


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


def anthropic_body(**extra):
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "temperature": 0,
    }
    body.update(extra)
    return body


def judgment_body(**overrides):
    body = {
        "id": "row-1",
        "state": {"evidence": "the sky is blue"},
        "question": "Is the claim supported?",
        "options": [
            {"id": "yes", "description": "supported"},
            {"id": "no", "description": "not supported"},
        ],
    }
    body.update(overrides)
    return body


def request_body(path, stream=False):
    if path == PATHS[1]:
        return responses_body(stream=stream, reasoning={"effort": "none"})
    body = chat_body(stream=stream, max_tokens=16)
    body.update(
        {"thinking": {"type": "disabled"}}
        if path == PATHS[2]
        else {"reasoning_effort": "none"}
    )
    return body


def tool_request(path, stream=False, schema=None):
    body = request_body(path, stream)
    parameters = {
        "type": "object",
        "properties": {"value": schema or {}},
        "required": ["value"],
    }
    tool = {"name": "echo", "parameters": parameters}
    if path == PATHS[0]:
        body["tools"] = [{"type": "function", "function": tool}]
    elif path == PATHS[1]:
        body["tools"] = [{"type": "function", **tool}]
    else:
        body["tools"] = [{"name": "echo", "input_schema": parameters}]
    return body


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


def png_data_url(color=(200, 30, 30)):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


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


def response_events(payload):
    events = []
    for block in payload.decode().strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        if data["type"] != event:
            raise AssertionError(f"SSE event mismatch: {event} != {data['type']}")
        events.append(data)
    return events


def stream_events(payload):
    chunks = (payload[i : i + 1] for i in range(len(payload)))
    return [
        event.json()
        for event in SSEDecoder().iter_bytes(chunks)
        if event.data != "[DONE]"
    ]


def openai_client(harness):
    host, port = harness.server.server_address
    return OpenAI(
        api_key="test",
        base_url=f"http://{host}:{port}/v1",
        max_retries=0,
        timeout=3,
    )


def pdf_bytes(
    text="ALPHA 42", *, pages=1, width=200, height=200, encrypted=False, padding_bytes=0
):
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for index in range(pages):
        page_id = len(objects) + 1
        kids.append(f"{page_id} 0 R")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode()
        )
        content = (
            f"1 0 0 rg 0 0 {width / 2} {height / 2} re f\n"
            f"0 0 1 rg {width / 2} 0 {width / 2} {height / 2} re f\n"
            f"0 0 0 rg BT /F1 16 Tf 10 {height - 30} Td ({text} page {index + 1}) Tj ET"
        ).encode("ascii")
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream"
        )
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {pages} >>".encode()
    if padding_bytes:
        objects.append(
            f"<< /Length {padding_bytes} >>\nstream\n".encode()
            + b" " * padding_bytes
            + b"\nendstream"
        )
    encryption = ""
    if encrypted:
        objects.append(
            b"<< /Filter /Standard /V 1 /R 2 /Length 40 /P -4 "
            b"/O <0000000000000000000000000000000000000000000000000000000000000000> "
            b"/U <0000000000000000000000000000000000000000000000000000000000000000> >>"
        )
        encryption = (
            f"/Encrypt {len(objects)} 0 R /ID [<0123456789abcdef> <0123456789abcdef>]"
        )
    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(
        f"trailer\n<< /Root 1 0 R /Size {len(offsets)} {encryption} >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(result)


def document_block(payload=None, **fields):
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(
                pdf_bytes() if payload is None else payload
            ).decode(),
        },
        **fields,
    }


def full_render_limits():
    """The worker's limits for a request whose budget is untouched."""
    return documents.RenderLimits(
        documents.MAX_PAGES,
        documents.MAX_PAGE_PIXELS,
        documents.MAX_TEXT_CHARACTERS,
        documents.MAX_RENDERED_BYTES,
        documents.MAX_REQUEST_DOCUMENT_BYTES,
    )


def empty_page_cache():
    """A patch of the rendered-page cache with an empty one, as the server
    starts with."""
    return mock.patch.object(
        documents, "_cache", LRUCache(documents.CACHE_BYTES, documents.CACHE_ENTRIES)
    )


def render_pdf(payload=None, budget=None):
    """A PDF's page text and image parts, rendered as request preparation
    renders a file part."""
    encoded = base64.b64encode(pdf_bytes() if payload is None else payload).decode()
    return documents.file_content(
        {"file_data": encoded},
        budget=documents.DocumentBudget(deadline=math.inf)
        if budget is None
        else budget,
    )


def keep_stop_signals(test):
    """Give the process its stop-signal handlers and mask back after `test`:
    serve takes the signals, and blocks them for the exec a test stubs."""
    for number in launcher.STOP_SIGNALS:
        test.addCleanup(signal.signal, number, signal.getsignal(number))
    test.addCleanup(
        signal.pthread_sigmask,
        signal.SIG_SETMASK,
        signal.pthread_sigmask(signal.SIG_BLOCK, ()),
    )


def server_arguments(argv):
    """The server's own arguments in the launcher's command for it."""
    return argv[argv.index("server.server") + 1 :]


def interface_drift(fake, real):
    """Where `fake` departs from the class `real` it stands in for: each of
    real's public methods it lacks or takes other parameters for, and each of
    its properties and declared attributes it lacks. Annotations aside."""

    def parameters(function):
        return [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(function).parameters.values()
        ]

    drift = []
    members = dict(inspect.getmembers_static(real))
    for name in sorted({*members, *inspect.get_annotations(real)}):
        if name.startswith("_"):
            continue
        member = members.get(name)
        if inspect.isfunction(member):
            own = inspect.getattr_static(type(fake), name, None)
            if not inspect.isfunction(own) or parameters(own) != parameters(member):
                drift.append(f"method {name}")
        elif (member is None or isinstance(member, property)) and not hasattr(
            fake, name
        ):
            drift.append(f"attribute {name}")
    return drift

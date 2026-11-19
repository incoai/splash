"""Prepare API requests for generation."""

import hashlib
import json
import secrets
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from jinja2 import TemplateError

from . import images as image_input
from . import protocol as wire
from . import request_fields
from .backend import Job, ReasoningRequest
from .chat_format import REASONING_EFFORTS, THINK_END, THINK_START, TOOL_CALL_OPEN
from .chat_messages import normalize_messages, template_messages
from .chat_templates import (
    LATER_SYSTEM_UNSUPPORTED,
    RESERVED_TEMPLATE_KWARGS,
    render_chat_template,
    template_options,
)
from .diagnostics import print_status
from .documents import render_files
from .errors import (
    APIError,
    ContextLengthError,
    RequestValidationError,
    field_error,
    remaining_request_time,
)
from .latency import LatencyMetrics
from .metrics import is_finite_number
from .output import MAX_STOP_CHARACTERS
from .tokenization import PromptTokenizer, tokenizable, unquoted
from .tool_schema import (
    ToolPolicy,
    json_grammar,
    normalize_response_format,
    normalize_tools,
    tool_grammar,
)

PREPARATION_WAIT_SECONDS = 30.0

# vLLM raises a nonzero temperature below this to it (_MAX_TEMP in
# vllm/sampling_params.py); Splash does the same.
MIN_SAMPLING_TEMPERATURE = 0.01

# Text completions' output budget when max_tokens is omitted: OpenAI's
# default for the endpoint, which vLLM and SGLang also use.
COMPLETION_DEFAULT_MAX_TOKENS = 16


def _generation_prompt(probed, rendered, tokens):
    """Whether the generation prompt opens a think block, and how many of the
    prompt's last tokens it is (zero where they differ from its tokens).
    `probed` is its text and tokens, as the startup probe found them for the
    request's template options, and the rendered prompt must end with that
    text."""
    text, ids = probed
    if not text or not rendered.endswith(text):
        raise APIError(
            400, "chat template must end with an assistant generation prefix"
        )
    count = len(ids)
    return (
        text.rfind(THINK_START) > text.rfind(THINK_END),
        count if count < len(tokens) and tuple(tokens[-count:]) == ids else 0,
    )


@dataclass
class Prompt:
    messages: list
    tools: list | None
    tool_policy: ToolPolicy | None
    reasoning_effort: str | None
    response_schema: dict | bool | None = None
    response_validator: object = None
    preserve_thinking: bool | None = None
    # The request's own chat_template_kwargs enable_thinking, which outranks
    # its reasoning effort (template_options); None where it sets none.
    enable_thinking: bool | None = None
    # Its other template variables, which outrank Splash's own.
    template_kwargs: dict = field(default_factory=dict)
    # The reasoning effort the request sent; None where it sent none.
    sent_effort: str | None = None
    # The reasoning effort its chat_template_kwargs set, which the template
    # renders over the others; None where they set none Splash names, or the
    # template reads none (ChatTemplate.reads_effort).
    template_effort: str | None = None


def _head_tokens(messages, tools, tokens, openings):
    """Where the prompt's head ends: its system message and the tools the
    template renders before any other message, which other conversations of
    the same agent or app share, end where the first other message opens
    (ChatFormat.turn_openings, as `openings`' tokens). The engine keeps a
    state there on first use, so the next conversation resumes past the
    head rather than prefill it again. Zero without a system message first
    or tools, and where no turn opens the format's way, as a custom template
    (--chat-template-file) may write them. A head holds no image: system
    content is text, and images come in user and tool turns."""
    system = bool(messages) and messages[0].get("role") == "system"
    if not (system or tools):
        return 0
    end = len(tokens)
    for ids in openings:
        index = -1
        # Its first token found by list.index, the rest checked there: on an
        # M5 Max a 15,000-token head takes 0.09 ms, where comparing at every
        # token took 3.1 ms.
        while True:
            try:
                index = tokens.index(ids[0], index + 1, end)
            except ValueError:
                break
            if tuple(tokens[index : index + len(ids)]) == ids:
                end = index
                break
    return 0 if end >= len(tokens) else end


@dataclass
class RenderedPrompt:
    text: str
    tokens: list[int]
    images: list
    image_positions: list[int]
    thinking: bool
    # Images precede the generation prompt; expanding them keeps this count.
    generation_prompt_tokens: int
    # Where the head ends (_head_tokens); zero without one.
    head_tokens: int = 0


@dataclass(frozen=True)
class GenerationOptions:
    """Sampling, stop and scheduling options, validated alike by every
    generation API."""

    sampling: wire.SamplingParameters
    stop_sequences: tuple[str, ...]
    ignore_eos: bool
    seed: int
    priority: wire.RequestPriority
    cache_salt: int
    # The thinking tokens of a request that thinks, after which its thinking
    # closes; None for no cap.
    thinking_token_budget: int | None


class Frontend:
    def __init__(
        self,
        tokenizer,
        backend,
        model,
        request_timeout,
        *,
        constraint_factory,
        chat_templates,
        chat_format,
        thinking_codec,
        response_store,
        max_image_pixels=image_input.MAX_PIXELS,
        served_model_names=(),
        announce_served_name=False,
        reasoning_effort=None,
        default_reasoning_effort=None,
        default_thinking_token_budget=None,
        sampling_defaults=None,
    ):
        # What the engine serves, from its Ready (backend.runtime.capabilities):
        # the context follows the latest engine (max_context); its lanes and
        # vision are the model's. Without vision, message normalization
        # rejects image and PDF input before any decoding.
        served = backend.runtime.capabilities
        preparation_capacity = served.max_concurrent_requests
        self.vision = served.vision
        self.latencies = LatencyMetrics()
        self.tokenizer = tokenizer
        # Probed at startup; requests choose among these, never the
        # tokenizer's own.
        self.chat_templates = chat_templates
        # The served family's: how its output reasons and calls tools.
        self.chat_format = chat_format
        # Sized by the first engine's context, which every relaunch asks for.
        self.prompt_tokenizer = PromptTokenizer(
            tokenizer, chat_format.message_boundary, served.max_context_tokens
        )
        # The tokens of each of its turn openings, where a prompt's head ends.
        self.turn_openings = tuple(
            tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in chat_format.turn_openings
        )
        self.backend = backend
        # The model id the engine loaded. /status reports it, so an alias
        # can never hide what served a request.
        self.model = model
        names = tuple(served_model_names)
        if announce_served_name and not names:
            raise ValueError("announce_served_name needs a served model name")
        # The name generation and scoring responses report.
        self.response_model = names[0] if announce_served_name else model
        # /v1/models order: the name responses report first, so a client
        # configured from data[0] (splash <client>) requests the name it gets
        # back. Every name stays accepted.
        self.model_names = tuple(dict.fromkeys((self.response_model, model, *names)))
        # The effort the server gives every chat request in place of whatever
        # the request says of reasoning (--reasoning-effort); None where each
        # request's own, or the default below, decides.
        self.reasoning_effort = reasoning_effort
        if (
            default_reasoning_effort is not None
            and default_reasoning_effort not in REASONING_EFFORTS
        ):
            raise ValueError("invalid default_reasoning_effort")
        self.default_reasoning_effort = default_reasoning_effort
        if default_thinking_token_budget is not None and (
            not isinstance(default_thinking_token_budget, int)
            or isinstance(default_thinking_token_budget, bool)
            or default_thinking_token_budget < 0
        ):
            raise ValueError("invalid default_thinking_token_budget")
        self.default_thinking_token_budget = default_thinking_token_budget
        # The served family's published sampling, by field: what a request
        # gets for each of these it leaves out. Each is a value a request may
        # send, so a request that leaves it out is never refused for it.
        self.sampling_defaults = dict(sampling_defaults or {})
        for name, value in self.sampling_defaults.items():
            if request_fields.sampling_refusal(name, value) is not None:
                raise ValueError(f"invalid sampling default {name}={value!r}")
        self.request_timeout = request_timeout
        self.constraint_factory = constraint_factory
        self.max_image_pixels = max_image_pixels
        self.images = image_input.ImageCache()
        self.ids = count(1)
        self.preparation_capacity = preparation_capacity
        self.preparation_slots = threading.BoundedSemaphore(preparation_capacity)
        self.preparation_lock = threading.Lock()
        self.preparation_active = 0
        self.preparation_waiting = 0
        # The Responses API's stored responses, which /status reports.
        self.response_store = response_store
        self.thinking_codec = thinking_codec

    def accepts_model(self, model):
        """Whether a request may name `model`: a name the server answers to,
        or none, absent or empty, which means the served model."""
        return model in (None, "") or (
            isinstance(model, str) and model in self.model_names
        )

    def check_model(self, body):
        """Refuse a request whose model is one the server does not answer to
        (accepts_model)."""
        if not self.accepts_model(body.get("model")):
            raise APIError(404, f"model {body['model']} not found", "model_not_found")

    @property
    def max_context(self):
        """The context the latest engine serves: a relaunch whose plan holds
        less serves that, until a later one regains the first's."""
        return self.backend.runtime.capabilities.max_context_tokens

    @property
    def input_modalities(self):
        """The one list /status, /v1/models and client setup report."""
        return ["text", "image", "pdf"] if self.vision else ["text"]

    @property
    def reasoning_switch(self):
        """Whether a request's reasoning effort or enable_thinking turns the
        model's reasoning on and off, as /status reports: where the family's
        model leaves that to its prompt (ChatFormat.reasoning_switch) and the
        server has not set every request's effort itself."""
        return self.chat_format.reasoning_switch and self.reasoning_effort is None

    @property
    def reasoning(self):
        """What /v1/models reports of reasoning: the default template's
        (ChatTemplate.reasoning), with the server's default effort in place of
        the template's own, named by the listed effort it renders like; no
        efforts where a request's effort decides nothing (reasoning_switch),
        so that a client offers none."""
        if not self.reasoning_switch:
            return {"thinks": None, "efforts": [], "default": None}
        template = self.chat_templates.select(None)
        reasoning = dict(template.reasoning)
        if (effort := self.default_reasoning_effort) is not None:
            reasoning["thinks"] = effort != "none"
            reasoning["default"] = template.effort_names.get(effort)
        return reasoning

    def status(self):
        status = self.backend.status()
        # The app shows the context the server serves now.
        status["capabilities"] = {"max_context_tokens": self.max_context}
        status["vision"] = self.vision
        status["input_modalities"] = self.input_modalities
        status["reasoning_switch"] = self.reasoning_switch
        status["reasoning_effort"] = self.reasoning_effort
        status["chat_template"] = self.chat_templates.status()
        with self.preparation_lock:
            status["frontend"] = {
                "preparation_capacity": self.preparation_capacity,
                "active": self.preparation_active,
                "waiting": self.preparation_waiting,
            }
        status["grammar_cache"] = self.constraint_factory.stats()
        status["response_store"] = self.response_store.stats()
        status["image_cache"] = self.images.stats()
        status["tokenizer_cache"] = self.prompt_tokenizer.stats()
        status["latency"] = self.latencies.snapshot()
        return status

    def _prepare_images(self, messages, deadline, *, check_context=True):
        """Prepared images in template render order: content parts in message
        order, images in document order."""
        parts = [
            part
            for message in messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        limit = wire.MAX_IMAGE_SPANS
        if len(parts) > limit:
            raise APIError(400, f"requests support at most {limit} images")
        prepared = self.images.request_batch()
        tokens = pixel_bytes = 0
        for part in parts:
            # An image can take a few hundred milliseconds to decode; an
            # expired request starts no more of them.
            remaining_request_time(deadline)
            try:
                payload = image_input.decode_data_url(part["image_url"]["url"])
                image = self.images.prepare(payload, self.max_image_pixels)
                tokens += image.tokens
                pixel_bytes += len(image.pixels)
                self._check_image_request_size(
                    tokens,
                    len(prepared) + 1,
                    pixel_bytes,
                    check_context=check_context,
                    image_tokens_only=True,
                )
                prepared.append(image)
            except image_input.ImageCapacityError as error:
                raise APIError(503, str(error), "frontend_overloaded") from error
            except image_input.ImageError as error:
                raise APIError(400, str(error)) from error
        return prepared

    def _check_image_request_size(
        self,
        tokens,
        image_count,
        pixel_bytes,
        *,
        check_context=True,
        image_tokens_only=False,
    ):
        if check_context and tokens >= self.max_context:
            raise ContextLengthError(
                tokens, self.max_context - 1, image_tokens_only=image_tokens_only
            )
        # A thinking budget's end tokens follow, at most this many.
        frame_bytes = (
            wire.REQUEST_FIXED_BYTES
            + 4 * tokens
            + wire.IMAGE_SPAN_BYTES * image_count
            + pixel_bytes
            + 4 * wire.MAX_THINKING_END_TOKENS
        )
        if frame_bytes > wire.MAX_FRAME_PAYLOAD_BYTES:
            raise APIError(400, "images exceed the request size limit")

    @staticmethod
    def _image_token_count(prompt_tokens, prepared):
        """The prompt's length once each image's placeholder is widened."""
        return len(prompt_tokens) + sum(image.tokens - 1 for image in prepared)

    def _expand_image_pads(self, prompt_tokens, prepared, positions):
        """Widens the template's single placeholder per image to the image's
        merged token count and returns the spans the engine injects into."""
        token_count = self._image_token_count(prompt_tokens, prepared)
        # Validate lengths before expanding tokens or copying repeated pixels.
        # HTTP body and image-cache limits do not bound decoded request size.
        self._check_image_request_size(
            token_count,
            len(prepared),
            sum(len(image.pixels) for image in prepared),
        )
        expanded, spans, cursor = [], [], 0
        pad_id = self.tokenizer.convert_tokens_to_ids(self.chat_format.image_pad)
        for position, image in zip(positions, prepared):
            expanded.extend(prompt_tokens[cursor:position])
            spans.append(
                wire.ImageSpan(
                    len(expanded),
                    image.tokens,
                    image.grid_height,
                    image.grid_width,
                    image.digest_lo,
                    image.digest_hi,
                )
            )
            expanded.extend([pad_id] * image.tokens)
            cursor = position + 1
        expanded.extend(prompt_tokens[cursor:])
        pixels = b"".join(image.pixels for image in prepared)
        return expanded, tuple(spans), pixels

    def request_deadline(self, body, started_at):
        timeout = body.get("timeout")
        if timeout is None:
            timeout = self.request_timeout
        elif not is_finite_number(timeout) or timeout <= 0:
            raise RequestValidationError(
                [field_error(["timeout"], "timeout must be positive")]
            )
        return started_at + min(timeout, self.request_timeout)

    def prepare(
        self,
        body,
        *,
        deadline,
        flags=wire.RequestFlag(0),
        output_field=None,
        stop_field="stop",
    ):
        """A Chat request, or a Messages request converted to Chat, with the
        request's own flags (request_fields.generation). Errors on the output
        limit name output_field, and errors on the stops' limits stop_field,
        that API's own fields."""
        with self.preparation(deadline) as without_slot:
            return self.chat_job(
                body,
                deadline=deadline,
                without_slot=without_slot,
                flags=flags,
                output_field=output_field,
                stop_field=stop_field,
            )

    def prepare_completion(self, body, *, deadline, flags=wire.RequestFlag(0)):
        """A text completion, with the request's own flags
        (request_fields.generation): the prompt generates as given, with no
        chat template, reasoning split, tools or images."""
        with self.preparation(deadline):
            self.check_model(body)
            # A request generates one text and returns only that text.
            for field in ("suffix", "logprobs"):
                if field in body:
                    raise APIError(400, f"{field} is not supported")
            if body.get("echo", False) is not False:
                raise APIError(400, "echo is not supported")
            for field in ("best_of", "n"):
                value = body.get(field, 1)
                if not isinstance(value, int) or isinstance(value, bool) or value != 1:
                    raise APIError(400, f"{field} must be 1")
            options = self._generation_options(body)
            # A text completion has no reasoning split, so no thinking is
            # tracked to close.
            if body.get("thinking_token_budget") not in (None, -1):
                raise APIError(
                    400, "thinking_token_budget is not supported for text completions"
                )
            prompt_tokens = self._completion_prompt(body.get("prompt"))
            remaining_request_time(deadline)
            requested = body.get("max_tokens")
            max_new = self._output_budget(
                COMPLETION_DEFAULT_MAX_TOKENS if requested is None else requested,
                prompt_tokens,
                "max_tokens",
            )
            return self._generation_job(
                options, prompt_tokens, max_new, deadline, flags
            )

    def _completion_prompt(self, prompt):
        """A string encodes as a raw prompt, with the tokenizer's own special
        tokens such as a BOS; token ids must be in its vocabulary."""
        if isinstance(prompt, str):
            try:
                tokens = self._tokenize(prompt, add_special_tokens=True)["input_ids"]
            except Exception as error:
                raise APIError(400, "prompt could not be tokenized") from error
        elif isinstance(prompt, list) and all(type(token) is int for token in prompt):
            vocabulary = len(self.tokenizer)
            if any(not 0 <= token < vocabulary for token in prompt):
                raise APIError(400, "prompt token ids must be in the vocabulary")
            tokens = prompt
        else:
            raise APIError(400, "prompt must be one string or one array of token ids")
        if not tokens:
            raise APIError(400, "prompt must not be empty")
        return list(tokens)

    def count_tokens(self, body, *, deadline):
        with self.preparation(deadline) as without_slot:
            prompt = self._prepare_prompt(
                body, None, deadline=deadline, without_slot=without_slot
            )
            rendered = self._render_prompt(prompt, deadline, check_context=False)
            return self._image_token_count(rendered.tokens, rendered.images)

    def tokenize(self, body, *, deadline):
        content = body.get("content")
        if not isinstance(content, str):
            raise APIError(400, "content must be a string")
        add_special = body.get("add_special", False)
        if not isinstance(add_special, bool):
            raise APIError(400, "add_special must be a boolean")
        for option, supported in (("parse_special", True), ("with_pieces", False)):
            if body.get(option, supported) is not supported:
                raise APIError(
                    400, f"only {option}={str(supported).lower()} is supported"
                )
        with self.preparation(deadline):
            try:
                tokens = self._tokenize(content, add_special_tokens=add_special)[
                    "input_ids"
                ]
            except Exception as error:
                raise APIError(400, "content could not be tokenized") from error
            remaining_request_time(deadline)
            return tokens

    def score_job(self, prompt_tokens, slot_ids, deadline, priority, cache_salt):
        """A job that reads the logits of `slot_ids` where the prompt ends and
        generates nothing: what a scoring API submits."""
        return Job(
            request_id=next(self.ids),
            prompt_tokens=prompt_tokens,
            max_new_tokens=0,
            seed=0,
            sampling=wire.SamplingParameters(),
            deadline=deadline,
            priority=priority,
            score_tokens=tuple(slot_ids),
            public_id=secrets.token_hex(16),
            cache_salt=cache_salt,
        )

    def apply_template(self, body, *, deadline):
        add_generation_prompt = body.get("add_generation_prompt", True)
        if not isinstance(add_generation_prompt, bool):
            raise APIError(400, "add_generation_prompt must be a boolean")
        with self.preparation(deadline) as without_slot:
            prompt = self._prepare_prompt(
                body, None, deadline=deadline, without_slot=without_slot
            )
            return self._render_prompt(
                prompt,
                deadline,
                check_context=False,
                add_generation_prompt=add_generation_prompt,
            ).text

    @contextmanager
    def preparation(self, deadline):
        """Hold a preparation slot while the request prepares. Yields a
        context manager under which the request holds none: it gives its
        slot back to wait for and use the PDF worker, which slots do not
        bound, and takes one again after, unless it failed meanwhile."""
        self._take_preparation_slot(deadline)
        held = True

        @contextmanager
        def without_slot():
            nonlocal held
            self._give_preparation_slot()
            held = False
            yield
            self._take_preparation_slot(deadline)
            held = True

        try:
            remaining_request_time(deadline)
            with self.latencies.measure("preparation"):
                yield without_slot
        finally:
            if held:
                self._give_preparation_slot()

    def _take_preparation_slot(self, deadline):
        remaining = remaining_request_time(deadline)
        with self.preparation_lock:
            self.preparation_waiting += 1
        with self.latencies.measure("preparation_queue"):
            acquired = self.preparation_slots.acquire(
                timeout=min(remaining, PREPARATION_WAIT_SECONDS)
            )
        with self.preparation_lock:
            self.preparation_waiting -= 1
            if acquired:
                self.preparation_active += 1
        if not acquired:
            remaining_request_time(deadline)
            raise APIError(
                503,
                "frontend preparation capacity is exhausted",
                "frontend_overloaded",
            )

    def _give_preparation_slot(self):
        with self.preparation_lock:
            self.preparation_active -= 1
        self.preparation_slots.release()

    def _prepare_prompt(self, body, tool_namespaces, *, deadline, without_slot):
        self.check_model(body)
        reasoning_effort = body.get("reasoning_effort")
        if reasoning_effort is None:
            reasoning_effort = self.default_reasoning_effort
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str)
            or reasoning_effort not in REASONING_EFFORTS
        ):
            raise APIError(400, "invalid reasoning_effort")
        preserve_thinking = body.get("preserve_thinking")
        if preserve_thinking is not None and not isinstance(preserve_thinking, bool):
            raise APIError(400, "preserve_thinking must be a boolean")
        template_kwargs = body.get("chat_template_kwargs")
        if template_kwargs is None:
            template_kwargs = {}
        elif not isinstance(template_kwargs, dict):
            raise APIError(400, "chat_template_kwargs must be an object")
        elif reserved := sorted(RESERVED_TEMPLATE_KWARGS & template_kwargs.keys()):
            raise APIError(400, f"chat_template_kwargs cannot set {reserved[0]}")
        template_kwargs = dict(template_kwargs)
        enable_thinking = template_kwargs.pop("enable_thinking", None)
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise APIError(
                400, "chat_template_kwargs enable_thinking must be a boolean"
            )
        if self.reasoning_effort is not None:
            # The server's effort is every request's, as though each had sent
            # it and nothing else of reasoning: it replaces the request's
            # effort, its enable_thinking, which would outrank an effort, and
            # a reasoning_effort among its template variables, which would
            # outrank Splash's own. It follows the request's own checks, so a
            # malformed field is refused as without it.
            reasoning_effort, enable_thinking = self.reasoning_effort, None
            template_kwargs.pop("reasoning_effort", None)
        template_effort = template_kwargs.get("reasoning_effort")
        messages = normalize_messages(body.get("messages"), vision=self.vision)
        tools, tool_policy = normalize_tools(
            body.get("tools"),
            body.get("tool_choice"),
            body.get("parallel_tool_calls", True),
            tool_namespaces,
        )
        response_schema, response_validator = normalize_response_format(
            body.get("response_format")
        )
        # PDFs render only once the whole request is checked: a render holds
        # the one PDF worker for up to 30 s.
        messages = template_messages(
            render_files(messages, deadline=deadline, without_slot=without_slot)
        )
        return Prompt(
            messages,
            tools,
            tool_policy,
            reasoning_effort,
            response_schema,
            response_validator,
            preserve_thinking,
            enable_thinking,
            template_kwargs,
            body.get("reasoning_effort"),
            template_effort
            if template_effort in REASONING_EFFORTS
            and self.chat_templates.select(tools).reads_effort
            else None,
        )

    def _tokenize(self, text, **options):
        with self.latencies.measure("tokenization"):
            return self.tokenizer(tokenizable(text), **options)

    def _apply_chat_template(self, messages, source, options):
        """The prompt the template `source` renders from the request's
        messages and template variables, each quoted (PromptTokenizer.quote)
        as text a client sent: what a message, a tool definition or a
        variable spells is never the template's own tag."""
        with self.latencies.measure("template"):
            quote = self.prompt_tokenizer.quote
            return render_chat_template(
                self.tokenizer,
                quote(messages),
                {
                    "tokenize": False,
                    "return_dict": False,
                    "chat_template": source,
                    **{name: quote(value) for name, value in options.items()},
                },
            )

    def _render_prompt(
        self, prompt, deadline, *, check_context=True, add_generation_prompt=True
    ):
        chat_template = self.chat_templates.select(prompt.tools)
        if not chat_template.accepts(prompt.messages):
            raise APIError(400, LATER_SYSTEM_UNSUPPORTED)
        options = {
            **template_options(
                reasoning_effort=prompt.reasoning_effort,
                preserve_thinking=prompt.preserve_thinking,
                tools=prompt.tools,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=prompt.enable_thinking,
            ),
            **prompt.template_kwargs,
        }
        with self.latencies.measure("images"):
            images = self._prepare_images(
                prompt.messages, deadline, check_context=check_context
            )
        remaining_request_time(deadline)
        try:
            rendered = self._apply_chat_template(
                prompt.messages, chat_template.source, options
            )
            with self.latencies.measure("tokenization"):
                tokens = self.prompt_tokenizer.encode(rendered)
        except APIError:
            raise
        except Exception as error:
            frame = error.__traceback__
            template_frame = None
            while True:
                if frame.tb_frame.f_code.co_filename == "<template>":
                    template_frame = frame
                if frame.tb_next is None:
                    break
                frame = frame.tb_next
            frame = template_frame or frame
            location = Path(frame.tb_frame.f_code.co_filename).name
            print_status(
                f"Template error · {type(error).__name__} · {location}:{frame.tb_lineno}",
                error=True,
            )
            # The client reads a template's own refusal, such as a system
            # message it does not take, with what it names of the request
            # as the request sent it; other errors may carry internals, and
            # the log keeps no request content.
            reason = (
                f": {unquoted(str(error))}" if isinstance(error, TemplateError) else ""
            )
            raise APIError(400, f"messages could not be rendered{reason}") from error
        # The text the model reads; the marks said how it encodes.
        rendered = unquoted(rendered)
        # Text a client sent encodes as no added token, so each pad among the
        # tokens is one the template wrote for an image.
        positions = []
        if images:
            pad_id = self.tokenizer.convert_tokens_to_ids(self.chat_format.image_pad)
            positions = [index for index, token in enumerate(tokens) if token == pad_id]
            if len(positions) != len(images):
                raise APIError(400, "image count does not match the rendered template")
        remaining_request_time(deadline)
        thinking, generation_prompt_tokens = False, 0
        if add_generation_prompt:
            thinking, generation_prompt_tokens = _generation_prompt(
                chat_template.generation_prompt(options), rendered, tokens
            )
        head_tokens = _head_tokens(
            prompt.messages, prompt.tools, tokens, self.turn_openings
        )
        return RenderedPrompt(
            rendered,
            tokens,
            images,
            positions,
            thinking,
            generation_prompt_tokens,
            head_tokens,
        )

    def chat_job(
        self,
        body,
        *,
        deadline,
        without_slot,
        flags=wire.RequestFlag(0),
        output_field=None,
        stop_field="stop",
        tool_namespaces=None,
    ):
        """The generation job of a Chat body, prepared while the caller holds
        a preparation slot (preparation(), which yields `without_slot`), with
        the request's own flags (request_fields.generation). tool_namespaces
        names each tool the model sees under another name, as the request
        names it."""
        # Fields checked without the prompt fail before it is prepared, which
        # can render documents and build validators.
        options = self._generation_options(body, stop_field)
        n = body.get("n", 1)
        logprobs = body.get("logprobs")
        if (
            not isinstance(n, int)
            or isinstance(n, bool)
            or n != 1
            or (logprobs is not None and (not isinstance(logprobs, bool) or logprobs))
        ):
            raise APIError(400, "n and logprobs are not currently supported")
        prompt = self._prepare_prompt(
            body, tool_namespaces, deadline=deadline, without_slot=without_slot
        )
        tools, tool_policy = prompt.tools, prompt.tool_policy
        response_schema, response_validator = (
            prompt.response_schema,
            prompt.response_validator,
        )
        # Constrained tool calls and structured output generate under a
        # grammar, which decides where the output ends; tools beside a
        # response schema share one.
        tool_constrained = tool_policy is not None and (
            tool_policy.constrained or response_schema is not None
        )
        constrained = tool_constrained or response_schema is not None
        if options.ignore_eos and constrained:
            raise APIError(
                400,
                "ignore_eos cannot be combined with constrained tool calls, "
                "tool_choice none or structured output",
            )
        rendered = self._render_prompt(prompt, deadline)
        prompt_tokens, prepared_images = rendered.tokens, rendered.images
        image_positions, thinking = rendered.image_positions, rendered.thinking
        constraint = None
        remaining_request_time(deadline)
        if constrained:
            with self.latencies.measure("grammar"):
                if tool_constrained:
                    constraint = self.constraint_factory.create(
                        tool_grammar(
                            tool_policy,
                            thinking,
                            response_schema,
                            chat_format=self.chat_format,
                        ),
                        timeout=remaining_request_time(deadline),
                        prefixes=lambda: self._call_openings(tool_policy, thinking),
                    )
                else:
                    constraint = self.constraint_factory.create(
                        json_grammar(
                            response_schema, thinking, chat_format=self.chat_format
                        ),
                        timeout=remaining_request_time(deadline),
                    )
        remaining_request_time(deadline)
        tools_signature = None
        if tools:
            digest = hashlib.sha1(
                json.dumps(tools, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:8]
            tools_signature = (len(tools), digest)
        image_spans, image_pixels = (), b""
        if prepared_images:
            prompt_tokens, image_spans, image_pixels = self._expand_image_pads(
                prompt_tokens, prepared_images, image_positions
            )
        remaining_request_time(deadline)
        requested = body.get("max_completion_tokens", body.get("max_tokens"))
        if output_field is None:
            # Chat takes either field; errors name the one the client sent.
            output_field = (
                "max_completion_tokens"
                if "max_completion_tokens" in body
                else "max_tokens"
            )
        max_new = self._output_budget(requested, prompt_tokens, output_field)
        budget = options.thinking_token_budget
        if budget is not None and requested is not None:
            # Where the context leaves less room than the request's limit, the
            # thinking keeps its share of that room, and the answer the rest; a
            # limit past the budget plus the room counts as the budget plus the
            # room.
            budget = budget * max_new // min(requested, budget + max_new)
        # A budget the output limit reaches first closes nothing.
        return self._generation_job(
            options,
            prompt_tokens,
            max_new,
            deadline,
            flags,
            thinking=thinking,
            reasoning=ReasoningRequest(
                prompt.sent_effort,
                prompt.reasoning_effort,
                prompt.enable_thinking,
                prompt.template_effort,
            ),
            chat_format=self.chat_format,
            thinking_token_budget=(
                budget if thinking and budget is not None and budget < max_new else None
            ),
            tool_policy=tool_policy,
            response_validator=response_validator,
            response_format=body.get("response_format"),
            constraint=constraint,
            image_spans=image_spans,
            image_pixels=image_pixels,
            image_owner=prepared_images if prepared_images else None,
            tools_signature=tools_signature,
            generation_prompt_tokens=rendered.generation_prompt_tokens,
            head_tokens=rendered.head_tokens,
            output_clamped_to_context=requested is not None and max_new < requested,
        )

    def _call_openings(self, policy, thinking):
        """The tokens that begin each strict tool's call, after the close of
        the reasoning the generation prompt opened. Its parameter names may
        come next, so a tool with more of them than the parser admits fails
        there."""
        chat_format = self.chat_format
        reasoning = [chat_format.think_end_id] if thinking else []
        return [
            (
                reasoning
                + self._tokenize(
                    TOOL_CALL_OPEN + chat_format.call_syntax.function_opening(name),
                    add_special_tokens=False,
                )["input_ids"],
                f"tool {name} has too many parameters to constrain",
            )
            for name in policy.schemas
            if name in policy.strict
        ]

    def _generation_options(self, body, stop_field="stop"):
        numbers = {}
        for name, (default, _, _) in request_fields.SAMPLING_NUMBERS.items():
            value = body.get(name, self.sampling_defaults.get(name, default))
            if (refusal := request_fields.sampling_refusal(name, value)) is not None:
                raise APIError(400, refusal)
            numbers[name] = float(value)
        if 0 < numbers["temperature"] < MIN_SAMPLING_TEMPERATURE:
            numbers["temperature"] = MIN_SAMPLING_TEMPERATURE
        # A top_k of 0 or -1 keeps every token, which the frame says with 0. A
        # top_k past the vocabulary keeps every token too, so one past the
        # frame's u32 is sent as its largest value.
        top_k = body.get("top_k", self.sampling_defaults.get("top_k", 0))
        if (refusal := request_fields.sampling_refusal("top_k", top_k)) is not None:
            raise APIError(400, refusal)
        top_k = min(max(top_k, 0), 0xFFFFFFFF)
        # Chat's stop, which another API's dialect converts its own field to.
        stop = body.get("stop")
        if stop is None:
            stop_sequences = ()
        elif isinstance(stop, str) and stop:
            stop_sequences = (stop,)
        elif isinstance(stop, list) and all(
            isinstance(value, str) and value for value in stop
        ):
            stop_sequences = tuple(stop)
        else:
            raise APIError(400, "stop must be a string or up to four strings")
        # Splash's limits on stops, for every API; errors name stop_field, the
        # API's own field.
        if len(stop_sequences) > 4:
            raise APIError(400, f"{stop_field} must be up to four strings")
        if any(len(value) > MAX_STOP_CHARACTERS for value in stop_sequences):
            raise APIError(
                400,
                f"each of {stop_field} must be at most {MAX_STOP_CHARACTERS} "
                "characters",
            )
        # Splash does not implement logit_bias, so only an empty one (null or
        # {}) is accepted; a non-empty one is refused rather than ignored.
        if body.get("logit_bias") not in (None, {}):
            raise APIError(400, "logit_bias is not supported")
        ignore_eos = body.get("ignore_eos", False)
        if not isinstance(ignore_eos, bool):
            raise APIError(400, "ignore_eos must be a boolean")
        seed = body.get("seed", -1)
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not -(2**63) <= seed < 2**64
        ):
            raise APIError(400, "seed must be a 64-bit integer")
        # -1 asks for a random seed, and another negative one is taken as
        # its 64 bits.
        seed = secrets.randbits(64) if seed == -1 else seed % 2**64
        # -1 sets no cap; an absent or null budget takes the server's.
        budget = body.get("thinking_token_budget")
        if budget is None:
            budget = self.default_thinking_token_budget
        elif not isinstance(budget, int) or isinstance(budget, bool) or budget < -1:
            raise APIError(
                400,
                "thinking_token_budget must be a non-negative integer or -1 for no cap",
            )
        elif budget == -1:
            budget = None
        return GenerationOptions(
            wire.SamplingParameters(top_k=top_k, **numbers),
            stop_sequences,
            ignore_eos,
            seed,
            request_fields.priority(body),
            request_fields.cache_salt(body),
            budget,
        )

    def _output_budget(self, requested, prompt_tokens, field):
        """The output token budget requested under the API's field name. It is
        a ceiling: a request generates up to the end of the context window
        its prompt leaves, all of it when the request names no budget."""
        if len(prompt_tokens) >= self.max_context:
            raise ContextLengthError(len(prompt_tokens), self.max_context - 1)
        remaining = self.max_context - len(prompt_tokens)
        max_new = remaining if requested is None else requested
        if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new <= 0:
            raise APIError(400, f"{field} must be a positive integer")
        return min(max_new, remaining)

    def _generation_job(
        self, options, prompt_tokens, max_new, deadline, flags, **fields
    ):
        """The job for a prepared prompt, with the endpoint's own fields: its
        flags are the request's own (request_fields.generation) and those its
        generation options set."""
        if options.ignore_eos:
            flags |= wire.RequestFlag.IGNORE_END_OF_SEQUENCE
        return Job(
            request_id=next(self.ids),
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new,
            seed=options.seed,
            sampling=options.sampling,
            deadline=deadline,
            priority=options.priority,
            cache_salt=options.cache_salt,
            stop_sequences=options.stop_sequences,
            flags=flags,
            public_id=secrets.token_hex(16),
            **fields,
        )

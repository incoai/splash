"""Prepare API requests for generation and manage Responses history."""

import copy
import hashlib
import json
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

if __package__:
    from . import images as image_input
    from . import json_codec, judgments
    from . import protocol as wire
    from .api_shapes import (
        IMAGE_PAD_TOKEN,
        canonical_responses_input,
        normalize_messages,
        responses_to_chat_body,
        template_messages,
    )
    from .backend import REQUEST_PRIORITIES, Job, remaining_request_time
    from .chat_templates import (
        LATER_SYSTEM_UNSUPPORTED,
        REASONING_EFFORTS,
        RESERVED_TEMPLATE_KWARGS,
        render_chat_template,
        template_options,
    )
    from .diagnostics import print_status
    from .errors import APIError, ContextLengthError
    from .latency import LatencyMetrics
    from .metrics import is_finite_number
    from .tokenization import PromptTokenizer
    from .tool_schema import (
        THINK_END,
        THINK_END_TOKEN_ID,
        TOOL_CALL_OPEN,
        ToolPolicy,
        function_opening,
        json_grammar,
        normalize_response_format,
        normalize_tools,
        tool_grammar,
    )
else:
    import images as image_input
    import json_codec
    import judgments
    import protocol as wire
    from api_shapes import (
        IMAGE_PAD_TOKEN,
        canonical_responses_input,
        normalize_messages,
        responses_to_chat_body,
        template_messages,
    )
    from backend import REQUEST_PRIORITIES, Job, remaining_request_time
    from chat_templates import (
        LATER_SYSTEM_UNSUPPORTED,
        REASONING_EFFORTS,
        RESERVED_TEMPLATE_KWARGS,
        render_chat_template,
        template_options,
    )
    from diagnostics import print_status
    from errors import APIError, ContextLengthError
    from latency import LatencyMetrics
    from metrics import is_finite_number
    from tokenization import PromptTokenizer
    from tool_schema import (
        THINK_END,
        THINK_END_TOKEN_ID,
        TOOL_CALL_OPEN,
        ToolPolicy,
        function_opening,
        json_grammar,
        normalize_response_format,
        normalize_tools,
        tool_grammar,
    )


PREPARATION_WAIT_SECONDS = 30.0


MIN_FLOAT32_SUBNORMAL = float.fromhex("0x1p-149")


RESPONSE_STORE_BUDGET_BYTES = 64 * 1024 * 1024


# Text completions' output budget when max_tokens is omitted: OpenAI's
# default for the endpoint, which vLLM and SGLang also use.
COMPLETION_DEFAULT_MAX_TOKENS = 16


# A stable marker lets repeated image requests reuse the compiled template.
IMAGE_RENDER_MARKER = f"__splash_image_{secrets.token_hex(16)}__"


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
        text.rfind("<think>") > text.rfind(THINK_END),
        count if count < len(tokens) and tuple(tokens[-count:]) == ids else 0,
    )


@dataclass(frozen=True, slots=True)
class StoredResponse:
    response_json: bytes
    history_json: bytes

    @property
    def size(self):
        return len(self.response_json) + len(self.history_json)

    @property
    def response(self):
        return json_codec.loads(self.response_json)


class ResponseStore:
    """Process-local Responses state with one strict byte-budgeted LRU."""

    def __init__(self, budget_bytes=RESPONSE_STORE_BUDGET_BYTES):
        if (
            not isinstance(budget_bytes, int)
            or isinstance(budget_bytes, bool)
            or budget_bytes <= 0
        ):
            raise ValueError("response store budget must be positive")
        self.budget_bytes = budget_bytes
        self.records = OrderedDict()
        self.bytes = 0
        self.evictions = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, response_id):
        with self.lock:
            record = self.records.pop(response_id, None)
            if record is None:
                self.misses += 1
                return None
            self.records[response_id] = record
            self.hits += 1
        return record

    def put(self, response, history_items):
        record = StoredResponse(
            json_codec.encode(response), json_codec.encode(history_items)
        )
        if record.size > self.budget_bytes:
            return False
        response_id = response["id"]
        with self.lock:
            previous = self.records.pop(response_id, None)
            if previous is not None:
                self.bytes -= previous.size
            self.records[response_id] = record
            self.bytes += record.size
            while self.bytes > self.budget_bytes:
                _, evicted = self.records.popitem(last=False)
                self.bytes -= evicted.size
                self.evictions += 1
        return True

    def delete(self, response_id):
        with self.lock:
            record = self.records.pop(response_id, None)
            if record is None:
                return False
            self.bytes -= record.size
            return True

    def stats(self):
        with self.lock:
            return {
                "entries": len(self.records),
                "bytes": self.bytes,
                "budget_bytes": self.budget_bytes,
                "evictions": self.evictions,
                "hits": self.hits,
                "misses": self.misses,
            }


@dataclass
class Prompt:
    messages: list
    tools: list | None
    tool_policy: ToolPolicy | None
    reasoning_effort: str | None
    response_schema: dict | bool | None = None
    response_validator: object = None
    preserve_thinking: bool | None = None
    # Template variables from the request, which outrank Splash's own.
    template_kwargs: dict = field(default_factory=dict)


@dataclass
class RenderedPrompt:
    text: str
    tokens: list[int]
    images: list
    image_positions: list[int]
    thinking: bool
    # Images precede the generation prompt; expanding them keeps this count.
    generation_prompt_tokens: int


@dataclass(frozen=True)
class GenerationOptions:
    """Sampling and stop options, validated alike by every generation API."""

    temperature: float
    top_p: float
    top_k: int
    stop_sequences: tuple[str, ...]
    ignore_eos: bool


def validate_served_model_name(value):
    if (
        not isinstance(value, str)
        or not value
        or any(not c.isprintable() or c.isspace() or c in "\\%?#" for c in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(
            "model alias must be a non-empty name without whitespace or URL delimiters"
        )
    return value


class Frontend:
    def __init__(
        self,
        tokenizer,
        backend,
        model,
        max_context,
        request_timeout,
        preparation_capacity,
        *,
        constraint_factory,
        chat_templates,
        thinking_codec,
        vision,
        max_image_pixels=image_input.MAX_PIXELS,
        served_model_names=(),
        default_reasoning_effort=None,
    ):
        if not isinstance(preparation_capacity, int) or preparation_capacity <= 0:
            raise ValueError("frontend preparation capacity must be positive")
        # Announced by the engine in its Ready event. Without it, message
        # normalization rejects image and PDF input before any decoding.
        self.vision = vision
        self.latencies = LatencyMetrics()
        self.tokenizer = tokenizer
        # Probed at startup; requests choose among these, never the
        # tokenizer's own.
        self.chat_templates = chat_templates
        self.prompt_tokenizer = PromptTokenizer(tokenizer)
        self.backend = backend
        self.model = model
        self.model_names = tuple(
            dict.fromkeys(
                [
                    model,
                    *(validate_served_model_name(name) for name in served_model_names),
                ]
            )
        )
        if (
            default_reasoning_effort is not None
            and default_reasoning_effort not in REASONING_EFFORTS
        ):
            raise ValueError("invalid default_reasoning_effort")
        self.default_reasoning_effort = default_reasoning_effort
        self.max_context = max_context
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
        self.response_store = ResponseStore()
        self.thinking_codec = thinking_codec

    def accepts_model(self, model):
        return isinstance(model, str) and model in self.model_names

    @property
    def input_modalities(self):
        """The one list /status, /v1/models and client setup report."""
        return ["text", "image", "pdf"] if self.vision else ["text"]

    def status(self):
        status = self.backend.status()
        status["vision"] = self.vision
        status["input_modalities"] = self.input_modalities
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

    def _prepare_images(self, messages, *, check_context=True):
        """Prepared images in template render order: content parts in message
        order, images in document order."""
        parts = [
            part
            for message in messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        limit = wire.ProtocolLimits().max_image_spans
        if len(parts) > limit:
            raise APIError(400, f"requests support at most {limit} images")
        prepared = self.images.request_batch()
        tokens = pixel_bytes = 0
        for part in parts:
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
        frame_bytes = (
            wire.REQUEST_FIXED_BYTES
            + 4 * tokens
            + wire.IMAGE_SPAN_BYTES * image_count
            + pixel_bytes
        )
        if frame_bytes > wire.ABSOLUTE_MAX_FRAME_PAYLOAD_BYTES:
            raise APIError(400, "images exceed the request size limit")

    def _render_image_tokens(self, messages, template):
        """Track placeholders emitted by the template, not quoted in input text.

        A temporary render marker is removed before tokenization, so the pinned
        template's final text and token IDs remain unchanged. Token offsets tie
        each real image to its placeholder even when a coding agent has read
        documentation or source containing literal vision tokens.
        """
        rendered = self._apply_chat_template(
            messages,
            {
                **template,
                "tokenize": False,
                "chat_template": template["chat_template"].replace(
                    IMAGE_PAD_TOKEN, IMAGE_RENDER_MARKER
                ),
            },
        )
        parts = rendered.split(IMAGE_RENDER_MARKER)
        image_offsets = set()
        offset = 0
        for part in parts[:-1]:
            offset += len(part)
            image_offsets.add((offset, offset + len(IMAGE_PAD_TOKEN)))
            offset += len(IMAGE_PAD_TOKEN)
        rendered = IMAGE_PAD_TOKEN.join(parts)
        encoded = self._tokenize(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        positions = [
            index
            for index, span in enumerate(encoded["offset_mapping"])
            if tuple(span) in image_offsets
        ]
        return list(encoded["input_ids"]), positions, rendered

    def _image_token_count(self, prompt_tokens, prepared, positions):
        pad_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
        if len(positions) != len(prepared) or any(
            prompt_tokens[position] != pad_id for position in positions
        ):
            raise APIError(400, "image count does not match the rendered template")
        return len(prompt_tokens) + sum(image.tokens - 1 for image in prepared)

    def _expand_image_pads(self, prompt_tokens, prepared, positions):
        """Widens the template's single placeholder per image to the image's
        merged token count and returns the spans the engine injects into."""
        token_count = self._image_token_count(prompt_tokens, prepared, positions)
        # Validate lengths before expanding tokens or copying repeated pixels.
        # HTTP body and image-cache limits do not bound decoded request size.
        self._check_image_request_size(
            token_count,
            len(prepared),
            sum(len(image.pixels) for image in prepared),
        )
        expanded, spans, cursor = [], [], 0
        pad_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
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

    def request_deadline(self, body, started_at=None):
        if started_at is None:
            started_at = time.monotonic()
        timeout = body.get("timeout")
        if timeout is None:
            timeout = self.request_timeout
        elif not is_finite_number(timeout) or timeout <= 0:
            raise APIError(400, "timeout must be positive")
        return started_at + min(timeout, self.request_timeout)

    def prepare(
        self,
        body,
        tool_namespaces=None,
        *,
        deadline=None,
        output_field=None,
        clamp_output_budget=False,
    ):
        """A Chat request, or another API's request converted to Chat.
        Output limit errors name output_field, that API's own field; with
        clamp_output_budget, a limit larger than what the context leaves is
        lowered to it instead of refused."""
        if deadline is None:
            deadline = self.request_deadline(body)
        with self._preparation(deadline):
            return self._prepare(
                body,
                tool_namespaces,
                deadline,
                output_field=output_field,
                clamp_output_budget=clamp_output_budget,
            )

    def prepare_completion(self, body, *, deadline=None):
        """A text completion: the prompt generates as given, with no chat
        template, reasoning split, tools or images."""
        if deadline is None:
            deadline = self.request_deadline(body)
        with self._preparation(deadline):
            nullable = {
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "n",
                "best_of",
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
                "max_tokens",
                "suffix",
                "echo",
                "logprobs",
            }
            body = {
                key: value
                for key, value in body.items()
                if value is not None or key not in nullable
            }
            if not self.accepts_model(body.get("model", self.model)):
                raise APIError(
                    404, f"model {body['model']} not found", "model_not_found"
                )
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
            prompt_tokens = self._completion_prompt(body.get("prompt"))
            remaining_request_time(deadline)
            # The default is a ceiling: a prompt that leaves less context
            # generates up to the rest.
            requested = body.get("max_tokens")
            max_new = self._output_budget(
                COMPLETION_DEFAULT_MAX_TOKENS if requested is None else requested,
                prompt_tokens,
                "max_tokens",
                clamp=requested is None,
            )
            return self._generation_job(body, options, prompt_tokens, max_new, deadline)

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

    def count_tokens(self, body, *, deadline=None):
        if deadline is None:
            deadline = self.request_deadline(body)
        with self._preparation(deadline):
            prompt = self._prepare_prompt(body, deadline=deadline)
            rendered = self._render_prompt(prompt, deadline, check_context=False)
            return self._image_token_count(
                rendered.tokens, rendered.images, rendered.image_positions
            )

    def tokenize(self, body, *, deadline=None):
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
        if deadline is None:
            deadline = self.request_deadline(body)
        with self._preparation(deadline):
            try:
                tokens = self._tokenize(content, add_special_tokens=add_special)[
                    "input_ids"
                ]
            except Exception as error:
                raise APIError(400, "content could not be tokenized") from error
            remaining_request_time(deadline)
            return tokens

    def _priority(self, body):
        priority_name = body.get("priority", "normal")
        if (
            not isinstance(priority_name, str)
            or priority_name not in REQUEST_PRIORITIES
        ):
            raise APIError(400, "priority must be foreground, normal, or background")
        return REQUEST_PRIORITIES[priority_name]

    def _score_job(self, prompt_tokens, slot_ids, deadline, priority, meta):
        return Job(
            request_id=next(self.ids),
            prompt_tokens=prompt_tokens,
            max_new_tokens=0,
            seed=0,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            deadline=deadline,
            priority=priority,
            score_tokens=tuple(slot_ids),
            public_id=secrets.token_hex(16),
            meta=meta,
        )

    def prepare_judgment(self, body, *, deadline=None):
        unknown = sorted(
            set(body)
            - {"id", "state", "question", "options", "model", "timeout", "priority"}
        )
        if unknown:
            raise APIError(400, f"unsupported fields: {', '.join(unknown)}")
        if not self.accepts_model(body.get("model", self.model)):
            raise APIError(404, f"model {body['model']} not found", "model_not_found")
        try:
            judgments.validate_row(body)
        except ValueError as error:
            raise APIError(400, str(error)) from error
        if deadline is None:
            deadline = self.request_deadline(body)
        priority = self._priority(body)
        with self._preparation(deadline):

            def admit(prompt_tokens):
                remaining_request_time(deadline)
                if prompt_tokens > self.max_context:
                    raise ContextLengthError(prompt_tokens, self.max_context)

            try:
                tokens, slots, prompt = judgments.encode_prompt(
                    self.tokenizer,
                    self.chat_templates.select(None).source,
                    judgments.judgment_messages(body),
                    judgments.LETTERS[: len(body["options"])],
                    admit=admit,
                    checkpoint=lambda: remaining_request_time(deadline),
                )
            except judgments.ScoringUnsupported as error:
                raise APIError(500, str(error), "scoring_unsupported") from error
            except APIError:
                raise
            except Exception as error:
                raise APIError(400, "judgment prompt could not be rendered") from error
            remaining_request_time(deadline)
            job = self._score_job(
                tokens,
                slots,
                deadline,
                priority,
                {
                    "prompt_sha256": judgments.digest(prompt),
                    "answer_token_ids": tuple(slots),
                },
            )
        return job, body

    def prepare_systemone(self, body, *, deadline=None):
        details = []
        model = body.get("model")
        if not isinstance(model, str) or not model:
            details.append(judgments.detail(["model"], "field required", "missing"))
        elif not self.accepts_model(model):
            details.append(
                judgments.detail(
                    ["model"], f"model {model} is not served by this endpoint"
                )
            )
        state, specs, question_details = judgments.validate_systemone(body)
        details.extend(question_details)
        priority_name = body.get("priority", "normal")
        if (
            not isinstance(priority_name, str)
            or priority_name not in REQUEST_PRIORITIES
        ):
            details.append(
                judgments.detail(
                    ["priority"],
                    "priority must be foreground, normal, or background",
                )
            )
        if details:
            raise judgments.SystemOneError(details)
        if deadline is None:
            deadline = self.request_deadline(body)
        priority = REQUEST_PRIORITIES[priority_name]
        jobs = []
        total_tokens = 0
        with self._preparation(deadline):
            for qid, spec in specs:
                if spec.deterministic:
                    jobs.append((qid, spec, None))
                    continue
                slots = judgments.slot_labels(self.tokenizer)
                if len(spec.labels) > len(slots):
                    raise judgments.SystemOneError(
                        [
                            judgments.detail(
                                ["questions", qid, "criteria"],
                                f"the served tokenizer supports "
                                f"{len(slots)} answer slots; "
                                f"{len(spec.labels)} were requested",
                            )
                        ]
                    )
                labels = slots[: len(spec.labels)]

                def admit(prompt_tokens, qid=qid, prepared=total_tokens):
                    remaining_request_time(deadline)
                    if prompt_tokens > self.max_context:
                        raise ContextLengthError(prompt_tokens, self.max_context)
                    if prepared + prompt_tokens > judgments.MAX_SYSTEMONE_TOTAL_TOKENS:
                        raise judgments.SystemOneError(
                            [
                                judgments.detail(
                                    ["questions", qid],
                                    "total prepared question tokens exceed "
                                    f"{judgments.MAX_SYSTEMONE_TOTAL_TOKENS}",
                                )
                            ]
                        )

                try:
                    tokens, slot_ids, prompt = judgments.encode_prompt(
                        self.tokenizer,
                        self.chat_templates.select(None).source,
                        judgments.systemone_messages(state, spec, labels),
                        labels,
                        admit=admit,
                        checkpoint=lambda: remaining_request_time(deadline),
                    )
                except judgments.ScoringUnsupported as error:
                    raise APIError(500, str(error), "scoring_unsupported") from error
                except (APIError, judgments.SystemOneError):
                    raise
                except Exception as error:
                    raise APIError(
                        500, "question prompt could not be rendered"
                    ) from error
                remaining_request_time(deadline)
                total_tokens += len(tokens)
                job = self._score_job(
                    tokens,
                    slot_ids,
                    deadline,
                    priority,
                    {
                        "prompt_sha256": judgments.digest(prompt),
                        "answer_token_ids": tuple(slot_ids),
                    },
                )
                jobs.append((qid, spec, job))
        return jobs

    def apply_template(self, body, *, deadline=None):
        add_generation_prompt = body.get("add_generation_prompt", True)
        if not isinstance(add_generation_prompt, bool):
            raise APIError(400, "add_generation_prompt must be a boolean")
        if deadline is None:
            deadline = self.request_deadline(body)
        with self._preparation(deadline):
            prompt = self._prepare_prompt(body, deadline=deadline)
            return self._render_prompt(
                prompt,
                deadline,
                check_context=False,
                add_generation_prompt=add_generation_prompt,
            ).text

    @contextmanager
    def _preparation(self, deadline):
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
        try:
            remaining_request_time(deadline)
            with self.latencies.measure("preparation"):
                yield
        finally:
            with self.preparation_lock:
                self.preparation_active -= 1
            self.preparation_slots.release()

    def _prepare_prompt(self, body, tool_namespaces=None, *, deadline=None):
        if not self.accepts_model(body.get("model", self.model)):
            raise APIError(404, f"model {body['model']} not found", "model_not_found")
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
        messages = template_messages(
            normalize_messages(
                body.get("messages"), vision=self.vision, deadline=deadline
            )
        )
        tools, tool_policy = normalize_tools(
            body.get("tools"),
            body.get("tool_choice"),
            body.get("parallel_tool_calls", True),
            tool_namespaces,
        )
        response_schema, response_validator = normalize_response_format(
            body.get("response_format")
        )
        return Prompt(
            messages,
            tools,
            tool_policy,
            reasoning_effort,
            response_schema,
            response_validator,
            preserve_thinking,
            template_kwargs,
        )

    def _tokenize(self, text, **options):
        with self.latencies.measure("tokenization"):
            return self.tokenizer(text, **options)

    def _apply_chat_template(self, messages, template):
        with self.latencies.measure("template"):
            return render_chat_template(self.tokenizer, messages, template)

    def _render_prompt(
        self, prompt, deadline, *, check_context=True, add_generation_prompt=True
    ):
        chat_template = self.chat_templates.select(prompt.tools)
        if not chat_template.accepts(prompt.messages):
            raise APIError(400, LATER_SYSTEM_UNSUPPORTED)
        template = {
            "tokenize": False,
            "return_dict": False,
            "chat_template": chat_template.source,
            **template_options(
                reasoning_effort=prompt.reasoning_effort,
                preserve_thinking=prompt.preserve_thinking,
                tools=prompt.tools,
                add_generation_prompt=add_generation_prompt,
            ),
            **prompt.template_kwargs,
        }
        with self.latencies.measure("images"):
            images = self._prepare_images(prompt.messages, check_context=check_context)
        remaining_request_time(deadline)
        if images and self.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN) is None:
            raise APIError(400, "the tokenizer does not define the image pad token")
        positions = []
        try:
            if images:
                tokens, positions, rendered = self._render_image_tokens(
                    prompt.messages, template
                )
            else:
                rendered = self._apply_chat_template(prompt.messages, template)
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
            raise APIError(400, "messages could not be rendered") from error
        remaining_request_time(deadline)
        thinking, generation_prompt_tokens = False, 0
        if add_generation_prompt:
            thinking, generation_prompt_tokens = _generation_prompt(
                chat_template.generation_prompt(template), rendered, tokens
            )
        requested = template.get("enable_thinking")
        if (
            add_generation_prompt
            and requested is not None
            and thinking != bool(requested)
        ):
            raise APIError(
                400, "chat template does not support the requested thinking mode"
            )
        return RenderedPrompt(
            rendered, tokens, images, positions, thinking, generation_prompt_tokens
        )

    def _prepare(
        self,
        body,
        tool_namespaces,
        deadline,
        *,
        output_field=None,
        clamp_output_budget=False,
    ):
        nullable = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "parallel_tool_calls",
        }
        body = {
            key: value
            for key, value in body.items()
            if value is not None or key not in nullable
        }
        prompt = self._prepare_prompt(body, tool_namespaces, deadline=deadline)
        options = self._generation_options(body)
        n = body.get("n", 1)
        logprobs = body.get("logprobs")
        if (
            not isinstance(n, int)
            or isinstance(n, bool)
            or n != 1
            or (logprobs is not None and (not isinstance(logprobs, bool) or logprobs))
        ):
            raise APIError(400, "n and logprobs are not currently supported")
        tools, tool_policy = prompt.tools, prompt.tool_policy
        response_schema, response_validator = (
            prompt.response_schema,
            prompt.response_validator,
        )
        # A stop sequence could cut a tool call or a structured result short;
        # under tool_choice none the tools are only described, never called.
        if options.stop_sequences and (
            (tools and tool_policy.schemas) or response_schema is not None
        ):
            raise APIError(
                400, "stop cannot be combined with tools or structured output"
            )
        # Tools and structured output generate under a grammar, which decides
        # where the output ends.
        constrained = bool(tools) or response_schema is not None
        if options.ignore_eos and constrained:
            raise APIError(
                400, "ignore_eos cannot be combined with tools or structured output"
            )
        rendered = self._render_prompt(prompt, deadline)
        prompt_tokens, prepared_images = rendered.tokens, rendered.images
        image_positions, thinking = rendered.image_positions, rendered.thinking
        constraint = None
        remaining_request_time(deadline)
        if constrained:
            with self.latencies.measure("grammar"):
                if tools:
                    constraint = self.constraint_factory.create(
                        tool_grammar(tool_policy, thinking, response_schema),
                        timeout=remaining_request_time(deadline),
                        prefixes=lambda: self._call_openings(tool_policy, thinking),
                    )
                elif response_schema is not None:
                    constraint = self.constraint_factory.create(
                        json_grammar(response_schema, thinking),
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
        max_new = self._output_budget(
            requested, prompt_tokens, output_field, clamp_output_budget
        )
        job = self._generation_job(
            body,
            options,
            prompt_tokens,
            max_new,
            deadline,
            thinking=thinking,
            thinking_display=(
                "omitted" if body.get("thinking_display") == "omitted" else "summarized"
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
            output_clamped_to_context=requested is not None and max_new < requested,
        )
        return job, thinking, bool(tools)

    def _call_openings(self, policy, thinking):
        """The tokens that begin each callable tool's call. Its parameter
        names are all possible next, so a tool with more of them than the
        parser admits fails there."""
        reasoning = [THINK_END_TOKEN_ID] if thinking else []
        return [
            (
                reasoning
                + self._tokenize(
                    TOOL_CALL_OPEN + function_opening(name), add_special_tokens=False
                )["input_ids"],
                f"tool {name} has too many parameters to constrain",
            )
            for name in policy.schemas
        ]

    def _generation_options(self, body):
        temperature = body.get("temperature", 1.0)
        top_p, top_k = body.get("top_p", 0.95), body.get("top_k", 20)
        if (
            not is_finite_number(temperature)
            or not is_finite_number(top_p)
            or not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or temperature < 0
            or temperature > 2
            or (temperature != 0 and temperature < MIN_FLOAT32_SUBNORMAL)
            or not 0 < top_p <= 1
            or top_p < MIN_FLOAT32_SUBNORMAL
            or not 1 <= top_k <= wire.MAX_TOP_K
        ):
            raise APIError(400, "invalid sampling parameters")
        stop = body.get("stop")
        if stop in (None, []):
            stop_sequences = ()
        elif isinstance(stop, str) and stop:
            stop_sequences = (stop,)
        elif (
            isinstance(stop, list)
            and 1 <= len(stop) <= 4
            and all(isinstance(value, str) and value for value in stop)
        ):
            stop_sequences = tuple(stop)
        else:
            raise APIError(400, "stop must be a string or up to four strings")
        # Each with the value that leaves the logits unchanged.
        penalties = (
            (body.get("presence_penalty", 0), 0),
            (body.get("frequency_penalty", 0), 0),
            (body.get("repetition_penalty", 1), 1),
            (body.get("min_p", 0), 0),
        )
        if any(
            not is_finite_number(value) or value != neutral
            for value, neutral in penalties
        ) or body.get("logit_bias") not in (None, {}):
            raise APIError(
                400, "the requested logits or output transformation is not supported"
            )
        ignore_eos = body.get("ignore_eos", False)
        if not isinstance(ignore_eos, bool):
            raise APIError(400, "ignore_eos must be a boolean")
        return GenerationOptions(temperature, top_p, top_k, stop_sequences, ignore_eos)

    def _output_budget(self, requested, prompt_tokens, field, clamp=False):
        """The output token budget requested under the API's field name,
        within the context window the prompt leaves. A request that names
        none may use all of that window, as in vLLM and SGLang."""
        if len(prompt_tokens) >= self.max_context:
            raise ContextLengthError(len(prompt_tokens), self.max_context - 1)
        remaining = self.max_context - len(prompt_tokens)
        max_new = remaining if requested is None else requested
        if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new <= 0:
            raise APIError(400, f"{field} must be a positive integer")
        if max_new > remaining:
            if not clamp:
                raise APIError(
                    400,
                    f"prompt and {field} exceed the context window: "
                    f"{len(prompt_tokens)} + {max_new} > {self.max_context} tokens",
                    "context_length_exceeded",
                )
            # This API treats the output budget as a ceiling. Generate up to
            # the remaining context and report the length stop if it is reached.
            max_new = remaining
        return max_new

    def _generation_job(
        self, body, options, prompt_tokens, max_new, deadline, **fields
    ):
        """The job for a prepared prompt, with the endpoint's own fields."""
        seed = body.get("seed")
        if seed is None:
            seed = secrets.randbits(64)
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise APIError(400, "seed must be an unsigned 64-bit integer")
        priority = self._priority(body)
        return Job(
            request_id=next(self.ids),
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new,
            seed=seed,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
            deadline=deadline,
            priority=priority,
            stop_sequences=options.stop_sequences,
            flags=(
                wire.RequestFlag.IGNORE_END_OF_SEQUENCE
                if options.ignore_eos
                else wire.RequestFlag(0)
            ),
            public_id=secrets.token_hex(16),
            **fields,
        )

    def prepare_responses(self, body, *, deadline=None, reserve_input=None):
        if deadline is None:
            deadline = self.request_deadline(body)
        store = body.get("store")
        if store is not None and not isinstance(store, bool):
            raise APIError(400, "store must be a boolean")
        store = True if store is None else store
        previous_id = body.get("previous_response_id")
        if previous_id is not None and (
            not isinstance(previous_id, str) or not previous_id
        ):
            raise APIError(400, "previous_response_id must be a non-empty string")
        with self._preparation(deadline):
            previous_items = []
            if previous_id is not None:
                previous = self.response_store.get(previous_id)
                if previous is None:
                    # Clients key on this code to resend the full history.
                    raise APIError(
                        404,
                        "previous response not found",
                        "previous_response_not_found",
                    )
                # The immutable record remains valid if the store evicts it.
                # Reserve its input bytes before materializing the history.
                if reserve_input is not None:
                    reserve_input(len(previous.history_json))
                previous_items = json_codec.loads(previous.history_json)
            chat = responses_to_chat_body(body, previous_items)
            namespaces = chat.pop("_tool_namespaces")
            job, thinking, has_tools = self._prepare(
                chat, namespaces, deadline, output_field="max_output_tokens"
            )
            job.response_store = store
            job.response_previous_id = previous_id
            if store:
                job.response_history_items = [
                    *previous_items,
                    *canonical_responses_input(body.get("input")),
                ]
            return job, thinking, has_tools

    def persist_response(self, job, response, output):
        if not job.response_store:
            return
        history = [*job.response_history_items, *copy.deepcopy(output)]
        if not self.response_store.put(response, history):
            response["store"] = False
            job.response_store = False

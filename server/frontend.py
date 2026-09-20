"""Prepare API requests for generation and manage Responses history."""

import copy
import hashlib
import json
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
from pathlib import Path

from jinja2 import TemplateError

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
    from .diagnostics import print_status
    from .errors import APIError, ContextLengthError
    from .latency import LatencyMetrics
    from .metrics import is_finite_number
    from .thinking import ThinkingCodec
    from .tool_schema import (
        THINK_END,
        ToolPolicy,
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
    from diagnostics import print_status
    from errors import APIError, ContextLengthError
    from latency import LatencyMetrics
    from metrics import is_finite_number
    from thinking import ThinkingCodec
    from tool_schema import (
        THINK_END,
        ToolPolicy,
        json_grammar,
        normalize_response_format,
        normalize_tools,
        tool_grammar,
    )


PREPARATION_WAIT_SECONDS = 30.0
REASONING_EFFORT_ALIASES = {"high": "xhigh", "max": "xhigh", "minimal": "low"}


MIN_FLOAT32_SUBNORMAL = float.fromhex("0x1p-149")


RESPONSE_STORE_BUDGET_BYTES = 64 * 1024 * 1024


# A stable marker lets repeated image requests reuse the compiled template.
IMAGE_RENDER_MARKER = f"__splash_image_{secrets.token_hex(16)}__"


def _thinking_from_prefix(rendered):
    marker = "<|im_start|>"
    start = rendered.rfind(marker)
    prefix = rendered[start + len(marker) :] if start >= 0 else ""
    if not prefix.startswith("assistant\n") or "<|im_end|>" in prefix:
        raise APIError(
            400, "chat template must end with an assistant generation prefix"
        )
    content = prefix[len("assistant\n") :]
    return content.rfind("<think>") > content.rfind(THINK_END)


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


@dataclass
class RenderedPrompt:
    text: str
    tokens: list[int]
    images: list
    image_positions: list[int]
    thinking: bool


class Frontend:
    def __init__(
        self,
        tokenizer,
        backend,
        model,
        max_context,
        default_max_new,
        request_timeout,
        preparation_capacity,
        constraint_factory=None,
        max_image_pixels=image_input.MAX_PIXELS,
        thinking_codec=None,
    ):
        if not isinstance(preparation_capacity, int) or preparation_capacity <= 0:
            raise ValueError("frontend preparation capacity must be positive")
        self.latencies = LatencyMetrics()
        self.tokenizer = tokenizer
        self.backend = backend
        self.model = model
        self.max_context = max_context
        self.default_max_new = default_max_new
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
        self.thinking_codec = (
            ThinkingCodec() if thinking_codec is None else thinking_codec
        )

    def status(self):
        status = self.backend.status()
        with self.preparation_lock:
            status["frontend"] = {
                "preparation_capacity": self.preparation_capacity,
                "active": self.preparation_active,
                "waiting": self.preparation_waiting,
            }
        if self.constraint_factory is not None:
            status["grammar_cache"] = self.constraint_factory.stats()
        status["response_store"] = self.response_store.stats()
        status["image_cache"] = self.images.stats()
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
        source = self.tokenizer.get_chat_template(tools=template.get("tools"))
        rendered = self._apply_chat_template(
            messages,
            {
                **template,
                "tokenize": False,
                "chat_template": source.replace(IMAGE_PAD_TOKEN, IMAGE_RENDER_MARKER),
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
        if not is_finite_number(timeout) or timeout <= 0:
            raise APIError(400, "timeout must be positive")
        return started_at + min(timeout, self.request_timeout)

    def prepare(
        self, body, tool_namespaces=None, *, deadline=None, clamp_output_budget=False
    ):
        if deadline is None:
            deadline = self.request_deadline(body)
        with self._preparation(deadline):
            return self._prepare(body, tool_namespaces, deadline, clamp_output_budget)

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
        if body.get("model", self.model) != self.model:
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
        elif model != self.model:
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
        if body.get("model", self.model) != self.model:
            raise APIError(404, f"model {body['model']} not found", "model_not_found")
        reasoning_effort = body.get("reasoning_effort")
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str)
            or reasoning_effort
            not in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
        ):
            raise APIError(400, "invalid reasoning_effort")
        preserve_thinking = body.get("preserve_thinking")
        if preserve_thinking is not None and not isinstance(preserve_thinking, bool):
            raise APIError(400, "preserve_thinking must be a boolean")
        messages = template_messages(
            normalize_messages(body.get("messages"), deadline=deadline)
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
        if response_schema is not None:
            instruction = (
                "Your final answer must be a JSON value matching the following "
                "JSON schema, without Markdown fences."
            )
            if tools:
                instruction += (
                    " You may call tools first when needed. Tool calls use their "
                    "own argument schemas; this schema applies only to your final answer."
                )
            instruction += "\n" + json.dumps(response_schema, separators=(",", ":"))
            # The template already describes tools. Describe the answer format
            # too, so the model can choose between a tool and a final answer.
            index = next(
                (
                    i
                    for i, message in enumerate(messages)
                    if message["role"] != "system"
                ),
                len(messages),
            )
            messages.insert(index, {"role": "system", "content": instruction})
        return Prompt(
            messages,
            tools,
            tool_policy,
            reasoning_effort,
            response_schema,
            response_validator,
            preserve_thinking,
        )

    def _tokenize(self, text, **options):
        with self.latencies.measure("tokenization"):
            return self.tokenizer(text, **options)

    def _apply_chat_template(self, messages, template):
        with self.latencies.measure("template"):
            return self._render_template(messages, template)

    def _render_template(self, messages, template):
        try:
            return self.tokenizer.apply_chat_template(messages, **template)
        except TemplateError:
            alias = REASONING_EFFORT_ALIASES.get(template.get("reasoning_effort"))
            if alias is None:
                raise
            return self.tokenizer.apply_chat_template(
                messages, **{**template, "reasoning_effort": alias}
            )

    def _render_prompt(
        self, prompt, deadline, *, check_context=True, add_generation_prompt=True
    ):
        template = {
            "tokenize": False,
            "return_dict": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if prompt.reasoning_effort is not None:
            template["enable_thinking"] = prompt.reasoning_effort != "none"
            if prompt.reasoning_effort != "none":
                template["reasoning_effort"] = prompt.reasoning_effort
        if prompt.preserve_thinking is not None:
            template["preserve_thinking"] = prompt.preserve_thinking
        if prompt.tools:
            template["tools"] = prompt.tools
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
                tokens = self._tokenize(rendered, add_special_tokens=False)["input_ids"]
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
        thinking = _thinking_from_prefix(rendered) if add_generation_prompt else False
        if (
            add_generation_prompt
            and prompt.reasoning_effort is not None
            and thinking != (prompt.reasoning_effort != "none")
        ):
            raise APIError(
                400, "chat template does not support the requested thinking mode"
            )
        return RenderedPrompt(rendered, tokens, images, positions, thinking)

    def _prepare(self, body, tool_namespaces, deadline, clamp_output_budget=False):
        nullable = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "n",
            "presence_penalty",
            "frequency_penalty",
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
        n = body.get("n", 1)
        logprobs = body.get("logprobs")
        if (
            not isinstance(n, int)
            or isinstance(n, bool)
            or n != 1
            or (logprobs is not None and (not isinstance(logprobs, bool) or logprobs))
        ):
            raise APIError(400, "n and logprobs are not currently supported")
        penalties = (
            body.get("presence_penalty", 0),
            body.get("frequency_penalty", 0),
            body.get("min_p", 0),
        )
        if any(
            not is_finite_number(value) or value != 0 for value in penalties
        ) or body.get("logit_bias") not in (None, {}):
            raise APIError(
                400, "the requested logits or output transformation is not supported"
            )
        tools, tool_policy = prompt.tools, prompt.tool_policy
        response_schema, response_validator = (
            prompt.response_schema,
            prompt.response_validator,
        )
        if stop_sequences and (tools or response_schema is not None):
            raise APIError(
                400, "stop cannot be combined with tools or structured output"
            )
        rendered = self._render_prompt(prompt, deadline)
        prompt_tokens, prepared_images = rendered.tokens, rendered.images
        image_positions, thinking = rendered.image_positions, rendered.thinking
        constraint = None
        remaining_request_time(deadline)
        if self.constraint_factory is not None:
            if tools:
                constraint = self.constraint_factory.create(
                    tool_grammar(tool_policy, thinking, response_schema),
                    timeout=remaining_request_time(deadline),
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
        if len(prompt_tokens) >= self.max_context:
            raise ContextLengthError(len(prompt_tokens), self.max_context - 1)
        max_new = body.get(
            "max_completion_tokens",
            body.get(
                "max_tokens",
                min(self.default_max_new, self.max_context - len(prompt_tokens)),
            ),
        )
        if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new <= 0:
            raise APIError(400, "max_completion_tokens must be a positive integer")
        if len(prompt_tokens) + max_new > self.max_context:
            if not clamp_output_budget:
                raise APIError(
                    400,
                    "prompt and max_completion_tokens exceed the context window",
                    "context_length_exceeded",
                )
            # This API treats the output budget as a ceiling. Generate up to
            # the remaining context and report the length stop if it is reached.
            max_new = self.max_context - len(prompt_tokens)
        seed = body.get("seed")
        if seed is None:
            seed = secrets.randbits(64)
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise APIError(400, "seed must be an unsigned 64-bit integer")
        priority = self._priority(body)
        request_id = next(self.ids)
        job = Job(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            deadline=deadline,
            priority=priority,
            stop_sequences=stop_sequences,
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
            public_id=secrets.token_hex(16),
            tools_signature=tools_signature,
        )
        return job, thinking, bool(tools)

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
                    raise APIError(404, "response not found", "not_found_error")
                # The immutable record remains valid if the store evicts it.
                # Reserve its input bytes before materializing the history.
                if reserve_input is not None:
                    reserve_input(len(previous.history_json))
                previous_items = json_codec.loads(previous.history_json)
            chat = responses_to_chat_body(body, previous_items)
            namespaces = chat.pop("_tool_namespaces")
            job, thinking, has_tools = self._prepare(chat, namespaces, deadline)
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

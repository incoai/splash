"""Extended /v1/systemone executor.

TypeLLM-style semantics over the native score/generate transport: thinking
jobs, finite-option scoring with permutations, open JSON answers under a
token grammar, nullable decisions, layered batch/sequential/DAG execution,
prefix warm-ups, and per-request usage accounting. The handler supplies
`run_jobs`, which submits a phase of native jobs with bounded parallelism
and returns their results in job order.
"""

from __future__ import annotations

import hashlib
import json
import random
import secrets
from dataclasses import dataclass, field

if __package__:
    from . import judgments
    from .backend import remaining_request_time
    from .errors import APIError, ContextLengthError
    from .frontend import _thinking_from_prefix
else:
    import judgments
    from backend import remaining_request_time
    from errors import APIError, ContextLengthError
    from frontend import _thinking_from_prefix


THINKING_TEMPERATURE = 0.6
THINKING_TOP_P = 0.95
THINKING_TOP_K = 20
VALUE_SAMPLE_TOP_K = 32
# A shared prompt prefix this long earns a score-only warm-up job, which
# publishes the recurrent state the hybrid runtime resumes from.
WARMUP_MIN_TOKENS = 256


@dataclass
class _Variant:
    """One prepared question variant (permutation order or single open)."""

    qid: str
    spec: object
    order: tuple | None
    index: int
    thinking: bool
    budget: int | None
    prompt_tokens: list
    slots: tuple
    image_spans: tuple
    image_pixels: bytes
    image_owner: object
    family: tuple
    think_max_new: int = 0
    constraint: object = None
    prefix: list = field(default_factory=list)
    reasoning: str | None = None
    reasoning_truncated: bool = False
    probabilities: list | None = None
    null_probability: float | None = None
    is_null: bool = False
    value: object = None


def _host_rng(seed, qid, purpose):
    if seed is None:
        return random.Random()
    return random.Random(f"{seed}:{qid}:{purpose}")


def _native_seed(seed, qid, variant, phase):
    if seed is None:
        return secrets.randbits(64)
    digest = hashlib.sha256(f"{seed}:{qid}:{variant}:{phase}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _result_or_raise(result, job):
    if result.reason == "cancelled":
        if job.timed_out:
            raise APIError(504, "request timed out", "request_timeout")
        raise APIError(500, "request cancelled", "request_cancelled")


def _logits_or_raise(result, job):
    _result_or_raise(result, job)
    if result.reason != "stop" or len(result.option_logits) != len(job.score_tokens):
        raise APIError(500, "runtime protocol error", "protocol_error")
    return list(result.option_logits)


def _longest_common_prefix(prompts):
    shortest = min(len(tokens) for tokens in prompts)
    length = 0
    while length < shortest and len({tokens[length] for tokens in prompts}) == 1:
        length += 1
    return length


def warmup_job(frontend, jobs, priority, deadline, warmed=None):
    """One score-only job over the jobs' shared token prefix, or None.

    A published recurrent state lets the jobs resume past the shared prefill.
    `warmed`, when given, deduplicates prefixes already warmed this request."""
    if len(jobs) < 2:
        return None
    prompts = [job.prompt_tokens for job in jobs]
    common = _longest_common_prefix(prompts)
    if common < WARMUP_MIN_TOKENS or common >= min(len(tokens) for tokens in prompts):
        return None
    # A published recurrent state must contain every image span whole;
    # partial images cannot be staged mid-prefix.
    if not all(span.offset + span.tokens <= common for span in jobs[0].image_spans):
        return None
    key = tuple(prompts[0][:common])
    if warmed is not None and key in warmed:
        return None
    slots = list(
        dict.fromkeys(
            frontend.tokenizer.encode(label, add_special_tokens=False)[0]
            for label in judgments.slot_labels(frontend.tokenizer)
        )
    )
    if len(slots) < 2:
        # A phase without finite questions still needs two distinct
        # score tokens; reuse the prefix's own in-vocabulary ids.
        slots = list(dict.fromkeys(prompts[0][:common]))
    if len(slots) < 2:
        return None
    if warmed is not None:
        warmed.add(key)
    return frontend._score_job(
        prompts[0][:common],
        slots[:2],
        deadline,
        priority,
        None,
        image_spans=jobs[0].image_spans,
        image_pixels=jobs[0].image_pixels,
        image_owner=jobs[0].image_owner,
    )


def _run_phase(frontend, run_jobs, entries, plan, warmed, deadline):
    """Submit one phase's jobs after prefix warm-ups. `entries` is a list of
    (job, family) and the runner returns results in the same order."""
    groups = {}
    for job, family in entries:
        groups.setdefault(family, []).append(job)
    warmups = []
    for jobs in groups.values():
        warmup = warmup_job(frontend, jobs, plan.priority, deadline, warmed)
        if warmup is not None:
            warmups.append(warmup)
    if warmups:
        for job, result in zip(warmups, run_jobs(warmups)):
            _result_or_raise(result, job)
    return run_jobs([job for job, _ in entries]) if entries else []


def _base_helpers(frontend):
    """Tokenizer constants every extended request needs."""
    tokenizer = frontend.tokenizer
    eos = {
        getattr(tokenizer, "eos_token_id", None),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    }
    return {
        "eos_ids": {token for token in eos if isinstance(token, int)},
        "prefill": tokenizer.encode(judgments.VALUE_PREFILL, add_special_tokens=False),
    }


def _thinking_helpers(frontend, state):
    """Tokenizer constants for the thinking protocol, derived once."""
    if state.get("think_end") is not None:
        return state
    tokenizer = frontend.tokenizer
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    if not isinstance(think_end, int) or tokenizer.encode(
        "</think>", add_special_tokens=False
    ) != [think_end]:
        raise APIError(
            500,
            "the tokenizer cannot express the thinking close marker",
            "thinking_unsupported",
        )
    state.update(
        think_end=think_end,
        natural=[think_end, *tokenizer.encode("\n\n", add_special_tokens=False)],
        forced=tokenizer.encode(
            judgments.FORCED_THINKING_CLOSE, add_special_tokens=False
        ),
    )
    return state


def _prepare_variants(frontend, plan, qid, spec, deps, deadline, total_tokens):
    """Render every variant prompt for one question; the caller holds a
    Frontend._preparation slot for the whole layer."""
    options = plan.options
    tokenizer = frontend.tokenizer
    thinking = spec.resolved_thinking(options)
    budget = spec.resolved_budget(options)
    if spec.kind in judgments.FINITE_QUESTION_TYPES:
        labels = judgments.slot_labels(tokenizer)
        if len(spec.labels) > len(labels):
            raise judgments.SystemOneError(
                [
                    judgments.detail(
                        ["questions", qid, "criteria"],
                        f"the served tokenizer supports {len(labels)} answer "
                        f"slots; {len(spec.labels)} were requested",
                    )
                ]
            )
        labels = labels[: len(spec.labels)]
        orders = judgments.permutation_orders(
            spec.permutations,
            len(spec.labels),
            _host_rng(options.seed, qid, "orders"),
        )
    else:
        labels = ()
        orders = [None]
    variants = []
    for index, order in enumerate(orders):
        if spec.kind in judgments.FINITE_QUESTION_TYPES:
            messages = judgments.systemone_messages(
                plan.state,
                spec,
                labels,
                thinking=thinking,
                order=order,
                dependencies=deps,
                image_urls=options.images,
            )
            system = (
                judgments.SYSTEMONE_THINKING_SYSTEM
                if thinking
                else judgments.SYSTEMONE_SYSTEM
            )
        else:
            messages = judgments.systemone_value_messages(
                plan.state,
                spec,
                thinking=thinking,
                dependencies=deps,
                image_urls=options.images,
            )
            system = (
                judgments.SYSTEMONE_VALUE_THINKING_SYSTEM
                if thinking
                else judgments.SYSTEMONE_VALUE_SYSTEM
            )
        tokens, rendered, spans, pixels = frontend._systemone_render(
            messages,
            thinking,
            options.reasoning_effort if thinking else None,
            plan.images,
        )
        remaining_request_time(deadline)
        if thinking:
            try:
                thinking_prefix = _thinking_from_prefix(rendered)
            except APIError as error:
                raise APIError(
                    error.status,
                    f"question {qid}: {error.message}",
                    error.code,
                ) from error
            if not thinking_prefix:
                raise APIError(
                    400,
                    f"question {qid}: the chat template does not support thinking",
                )
        elif len(tokens) > frontend.max_context:
            raise ContextLengthError(len(tokens), frontend.max_context)
        total_tokens += len(tokens)
        if total_tokens > judgments.MAX_SYSTEMONE_TOTAL_TOKENS:
            raise judgments.SystemOneError(
                [
                    judgments.detail(
                        ["questions", qid],
                        "total prepared question tokens exceed "
                        f"{judgments.MAX_SYSTEMONE_TOTAL_TOKENS}",
                    )
                ]
            )
        slots = ()
        if spec.kind in judgments.FINITE_QUESTION_TYPES:
            try:
                slots = judgments.answer_slots(
                    tokenizer, rendered, tokens, len(labels), thinking=thinking
                )
            except judgments.ScoringUnsupported as error:
                raise APIError(500, str(error), "scoring_unsupported") from error
        variant = _Variant(
            qid=qid,
            spec=spec,
            order=order,
            index=index,
            thinking=thinking,
            budget=budget,
            prompt_tokens=tokens,
            slots=slots,
            image_spans=spans,
            image_pixels=pixels,
            image_owner=plan.images or None,
            family=(system, thinking),
        )
        variants.append(variant)
    return variants, total_tokens


def _value_budget(options, spec):
    return (
        options.text_max_tokens
        if spec.kind == "string"
        else options.numeric_max_digits + 8
    )


def _think_jobs(frontend, plan, variants, helpers, deadline):
    jobs = []
    for variant in variants:
        reserve = 0
        if variant.spec.kind in judgments.OPEN_QUESTION_TYPES:
            reserve = len(helpers["prefill"]) + _value_budget(
                plan.options, variant.spec
            )
        available = (
            frontend.max_context
            - len(variant.prompt_tokens)
            - reserve
            - len(helpers["forced"])
            - judgments.THINKING_TAIL_RESERVE
        )
        if available < 1:
            raise APIError(
                400,
                f"question {variant.qid} leaves no room for thinking and the answer",
                "context_length_exceeded",
            )
        variant.think_max_new = min(
            available,
            variant.budget if variant.budget is not None else available,
            frontend.default_max_new,
        )
        jobs.append(
            frontend._systemone_job(
                variant.prompt_tokens,
                variant.think_max_new,
                seed=_native_seed(
                    plan.options.seed, variant.qid, variant.index, "thinking"
                ),
                temperature=THINKING_TEMPERATURE,
                top_p=THINKING_TOP_P,
                top_k=THINKING_TOP_K,
                deadline=deadline,
                priority=plan.priority,
                thinking=True,
                stop_token_ids=(helpers["think_end"],),
                image_spans=variant.image_spans,
                image_pixels=variant.image_pixels,
                image_owner=variant.image_owner,
            )
        )
    return jobs


def _finish_thinking(tokenizer, variant, job, result, helpers):
    _result_or_raise(result, job)
    output = list(result.output_tokens)
    if result.stop_sequence is not None:
        reasoning = output[:-1]
        variant.prefix = [
            *variant.prompt_tokens,
            *reasoning,
            *helpers["natural"],
        ]
        variant.reasoning_truncated = False
    else:
        while output and output[-1] in helpers["eos_ids"]:
            output.pop()
        reasoning = output
        if not tokenizer.decode(reasoning, skip_special_tokens=True).strip():
            raise APIError(
                500,
                "the model produced empty reasoning",
                "invalid_model_output",
            )
        variant.prefix = [
            *variant.prompt_tokens,
            *reasoning,
            *helpers["forced"],
        ]
        variant.reasoning_truncated = True
    variant.reasoning = tokenizer.decode(reasoning, skip_special_tokens=True).strip()


def _value_prompt(variant, helpers):
    return [*variant.prefix, *helpers["prefill"]]


def _nullable_candidates(tokenizer, spec):
    starts = judgments.value_start_tokens(tokenizer)
    value_ids = (
        starts["string"]
        if spec.kind == "string"
        else [*starts["positive"], *starts["negative"], *starts["direct"]]
    )
    if not starts["null"] or not value_ids:
        raise APIError(
            500,
            "the tokenizer cannot express null and value token starts",
            "scoring_unsupported",
        )
    return starts["null"], value_ids


def _value_job(frontend, plan, variant, helpers, options, deadline):
    prompt = _value_prompt(variant, helpers)
    max_new = _value_budget(options, variant.spec)
    if len(prompt) + max_new > frontend.max_context:
        raise APIError(
            400,
            f"question {variant.qid} leaves no room for its answer",
            "context_length_exceeded",
        )
    sample = options.mode == "sample"
    return frontend._systemone_job(
        prompt,
        max_new,
        seed=_native_seed(options.seed, variant.qid, variant.index, "value"),
        temperature=options.temperature if sample else 0.0,
        top_p=1.0,
        top_k=VALUE_SAMPLE_TOP_K if sample else 1,
        deadline=deadline,
        priority=plan.priority,
        constraint=variant.constraint,
        image_spans=variant.image_spans,
        image_pixels=variant.image_pixels,
        image_owner=variant.image_owner,
    )


def _parse_value(tokenizer, spec, job, result, helpers):
    _result_or_raise(result, job)
    if result.reason == "length":
        raise APIError(
            500,
            f"the model did not complete its answer within {job.max_new_tokens} tokens",
            "invalid_model_output",
        )
    if result.reason != "stop":
        raise APIError(500, "runtime protocol error", "protocol_error")
    tokens = list(result.output_tokens)
    while tokens and tokens[-1] in helpers["eos_ids"]:
        tokens.pop()
    try:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        parsed = json.loads(judgments.VALUE_PREFILL + text)
        if not isinstance(parsed, dict) or list(parsed) != ["answer"]:
            raise ValueError("answer is not a single-key object")
        value = parsed["answer"]
    except ValueError as error:
        raise APIError(
            500,
            f"the model produced an invalid answer: {error}",
            "invalid_model_output",
        ) from error
    try:
        return judgments.check_open_value(spec, value)
    except ValueError as error:
        raise APIError(
            500,
            f"the model produced an invalid answer: {error}",
            "invalid_model_output",
        ) from error


def _selected_finite(spec, probabilities, options, rng):
    index = (
        judgments.sample_index(probabilities, rng)
        if options.mode == "sample"
        else judgments.argmax_index(probabilities)
    )
    if spec.kind == "noul":
        return index == 0, index
    if spec.kind == "choice":
        return spec.labels[index], index
    return spec.descriptions[index], index


def execute(frontend, run_jobs, plan, deadline):
    """Run the validated plan and return the /v1/systemone response body."""
    options = plan.options
    tokenizer = frontend.tokenizer
    spec_map = dict(plan.specs)
    execution = judgments.resolve_execution(plan.specs, options.execution)
    layers = judgments.execution_layers(plan.specs, execution)
    visible = judgments.visible_dependencies(plan.specs, execution)
    temperature = options.temperature if options.mode == "sample" else 1.0
    answers = {}
    selected = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    reasoning_tokens = 0
    any_thinking = any(spec.resolved_thinking(options) for _, spec in plan.specs)
    total_tokens = 0
    warmed = set()
    helpers = _base_helpers(frontend)
    for layer in layers:
        variants = []
        with frontend._preparation(deadline):
            for qid, spec in layer:
                if spec.deterministic:
                    continue
                dependencies = tuple(
                    (spec_map[dep], selected[dep]) for dep in visible[qid]
                )
                made, total_tokens = _prepare_variants(
                    frontend, plan, qid, spec, dependencies, deadline, total_tokens
                )
                for variant in made:
                    if variant.spec.kind in judgments.OPEN_QUESTION_TYPES:
                        if frontend.constraint_factory is None:
                            raise APIError(
                                500,
                                "open questions need an output constraint factory",
                                "scoring_unsupported",
                            )
                        variant.constraint = frontend.constraint_factory.create(
                            judgments.value_grammar(
                                variant.spec, options.numeric_max_digits
                            ),
                            timeout=remaining_request_time(deadline),
                        )
                variants.extend(made)
        for qid, spec in layer:
            if spec.deterministic:
                answers[qid] = judgments.deterministic_answer(spec)
                selected[qid] = judgments.deterministic_value(spec)
        if not variants:
            continue
        # Think phase: one job per thinking variant.
        thinking_variants = [variant for variant in variants if variant.thinking]
        if thinking_variants:
            _thinking_helpers(frontend, helpers)
            think_jobs = _think_jobs(
                frontend, plan, thinking_variants, helpers, deadline
            )
            entries = [
                (job, variant.family)
                for job, variant in zip(think_jobs, thinking_variants)
            ]
            results = _run_phase(frontend, run_jobs, entries, plan, warmed, deadline)
            for variant, job, result in zip(thinking_variants, think_jobs, results):
                usage["input_tokens"] += result.prompt_tokens
                usage["output_tokens"] += result.completion_tokens
                reasoning_tokens += result.completion_tokens
                _finish_thinking(tokenizer, variant, job, result, helpers)
        for variant in variants:
            if not variant.prefix:
                variant.prefix = list(variant.prompt_tokens)
        # Decision phase: finite scores, nullable decisions, open values.
        actions = []
        entries = []
        pending_values = []
        for variant in variants:
            spec = variant.spec
            if spec.kind in judgments.FINITE_QUESTION_TYPES:
                job = frontend._score_job(
                    variant.prefix,
                    variant.slots,
                    deadline,
                    plan.priority,
                    None,
                    image_spans=variant.image_spans,
                    image_pixels=variant.image_pixels,
                    image_owner=variant.image_owner,
                )
                actions.append(("score", variant, job, ()))
                entries.append((job, variant.family))
            elif spec.nullable:
                null_ids, value_ids = _nullable_candidates(tokenizer, spec)
                job = frontend._score_job(
                    _value_prompt(variant, helpers),
                    list(dict.fromkeys([*null_ids, *value_ids])),
                    deadline,
                    plan.priority,
                    None,
                    image_spans=variant.image_spans,
                    image_pixels=variant.image_pixels,
                    image_owner=variant.image_owner,
                )
                actions.append(("decide", variant, job, (null_ids, value_ids)))
                entries.append((job, variant.family))
            else:
                job = _value_job(frontend, plan, variant, helpers, options, deadline)
                actions.append(("value", variant, job, ()))
                entries.append((job, variant.family))
        results = _run_phase(frontend, run_jobs, entries, plan, warmed, deadline)
        for (kind, variant, job, extra), result in zip(actions, results):
            usage["input_tokens"] += result.prompt_tokens
            if kind == "score":
                logits = _logits_or_raise(result, job)
                variant.probabilities = judgments.softmax(
                    [logit / temperature for logit in logits]
                )
            elif kind == "decide":
                logits = _logits_or_raise(result, job)
                null_ids, value_ids = extra
                p_null = judgments.softmax(
                    [
                        judgments.logsumexp(logits[: len(null_ids)]) / temperature,
                        judgments.logsumexp(logits[len(null_ids) :]) / temperature,
                    ]
                )[0]
                variant.null_probability = p_null
                if options.mode == "sample":
                    variant.is_null = (
                        _host_rng(options.seed, variant.qid, "null").random() < p_null
                    )
                else:
                    variant.is_null = p_null >= 1.0 - p_null
                if not variant.is_null:
                    pending_values.append(variant)
            else:
                usage["output_tokens"] += result.completion_tokens
                variant.value = _parse_value(
                    tokenizer, variant.spec, job, result, helpers
                )
        # Value phase: nullable questions that decided to answer.
        if pending_values:
            entries = []
            jobs = []
            for variant in pending_values:
                job = _value_job(frontend, plan, variant, helpers, options, deadline)
                jobs.append(job)
                entries.append((job, variant.family))
            results = _run_phase(frontend, run_jobs, entries, plan, warmed, deadline)
            for variant, job, result in zip(pending_values, jobs, results):
                usage["input_tokens"] += result.prompt_tokens
                usage["output_tokens"] += result.completion_tokens
                variant.value = _parse_value(
                    tokenizer, variant.spec, job, result, helpers
                )
        # Answers and selected values.
        for qid, spec in layer:
            if spec.deterministic:
                continue
            group = [variant for variant in variants if variant.qid == qid]
            if spec.kind in judgments.FINITE_QUESTION_TYPES:
                mean = [0.0] * len(spec.labels)
                for variant in group:
                    for position, original in enumerate(variant.order):
                        mean[original] += variant.probabilities[position]
                mean = [value / len(group) for value in mean]
                rng = _host_rng(options.seed, qid, "select")
                value, index = _selected_finite(spec, mean, options, rng)
                answer = judgments.systemone_answer(spec, mean, selected=index)
                selected[qid] = value
            else:
                variant = group[0]
                answer = judgments.open_answer(
                    spec, variant.value, variant.null_probability
                )
                selected[qid] = variant.value
            if options.return_reasoning and spec.resolved_thinking(options):
                identity = group[0]
                answer["reasoning"] = identity.reasoning
                answer["reasoning_truncated"] = identity.reasoning_truncated
            answers[qid] = answer
    if any_thinking:
        usage["reasoning_tokens"] = reasoning_tokens
    return {
        "model": frontend.model,
        "answers": {qid: answers[qid] for qid, _ in plan.specs},
        "usage": usage,
    }

# SemIf direct-options prompt and helpers:
# https://github.com/TheoLeeCJ/SemIf
# MIT License
# Copyright (c) 2026 TheoLeeCJ
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Finite-option scoring: Splash's /v1/judgments and TypeSafe's
/v1/systemone, their preparation, answers and error shape.

Both endpoints render the SemIf direct-options-v1 prompt shape (a fixed
system instruction plus one JSON user payload, thinking disabled) and read
raw final-position logits at verified single-token answer slots. Question
and row identifiers never enter the prompt.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import string
import weakref
from dataclasses import dataclass

from .. import request_fields
from ..chat_format import THINK_END, THINK_START
from ..chat_templates import render_chat_template, template_options
from ..errors import (
    APIError,
    ContextLengthError,
    ErrorDialect,
    RequestValidationError,
    field_error,
    remaining_request_time,
)
from ..protocol import MAX_SCORE_TOKENS
from ..tokenization import tokenizable

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one "
    "listed option. Respond with only its uppercase letter, with no explanation "
    "or reasoning."
)
SYSTEMONE_SYSTEM = DIRECT_SYSTEM.replace("uppercase letter", "uppercase slot")
PROMPT_VERSION = "direct-options-v1"
READOUT = (
    "native full-vocabulary last-position logits restricted to declared answer slots"
)
PROBABILITY_STATUS = "conditional option score; uncalibrated as decision confidence"
# A /v1/systemone batch prepares every question before the first inference
# and runs them under one shared deadline, so the batch carries its own
# caps: at most this many questions holding at most this many prepared
# prompt tokens in total.
MAX_SYSTEMONE_QUESTIONS = 64
MAX_SYSTEMONE_TOTAL_TOKENS = 1 << 20

_MISSING = object()


class ScoringUnsupported(RuntimeError):
    """The served tokenizer cannot express exact single-token answer slots."""


def validate_row(row):
    """SemIf semif_phase1.core.validate_row, verbatim."""
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("options must contain 2-16 entries")
    ids = []
    for option in options:
        if (
            not isinstance(option, dict)
            or not isinstance(option.get("id"), str)
            or not isinstance(option.get("description"), str)
        ):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def judgment_messages(row):
    """SemIf semif_phase1.core.direct_messages, verbatim."""
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def softmax(values):
    """SemIf semif_phase1.core.softmax, verbatim."""
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def digest(text):
    """SemIf semif_phase1.core.digest, verbatim."""
    return hashlib.sha256(text.encode()).hexdigest()


def concentration(probabilities):
    """Normalized-entropy concentration in [0, 1]: how spread the option
    distribution is, not a calibrated confidence."""
    if len(probabilities) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    return max(0.0, 1.0 - entropy / math.log(len(probabilities)))


_SLOT_LABELS = weakref.WeakKeyDictionary()


def _derive_slot_labels(tokenizer):
    labels = []
    for length in (1, 2, 3):
        for letters in itertools.product(string.ascii_uppercase, repeat=length):
            label = "".join(letters)
            encoded = tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) == 1 and tokenizer.decode(encoded) == label:
                labels.append(label)
                if len(labels) >= MAX_SCORE_TOKENS:
                    return labels
    return labels


def slot_labels(tokenizer):
    """Stable single-token answer slots for the served tokenizer, sorted by
    (length, label): A..Z, AA, AB, ... The list is derived once per tokenizer
    and cached; every use still verifies the prompt boundary."""
    try:
        labels = _SLOT_LABELS.get(tokenizer)
    except TypeError:
        labels = None
    if labels is None:
        labels = tuple(_derive_slot_labels(tokenizer))
        try:
            _SLOT_LABELS[tokenizer] = labels
        except TypeError:
            pass
    return labels


def encode_prompt(
    prompt_tokenizer,
    chat_template,
    messages,
    labels,
    *,
    chat_format,
    admit,
    checkpoint,
):
    """Render messages with chat_template without thinking, as a request
    with effort none renders, and verify single-token answer slots. A model
    that opens its reasoning itself (chat_format.think_start_id), which
    effort none leaves reasoning, gets an empty think block at the prompt's
    end, after which it answers. The
    prompt is the rendered text as every prompt is tokenized, each surrogate
    code point as U+FFFD: the one text its tokens, the slot checks and its
    digest read.

    As SemIf's semif_phase1.direct.encode_prompt: each slot label must be
    one exact round-trip token, and appending it to the rendered prompt must
    extend the token ids by exactly that token. A label can change only the
    tokens after the prompt's last message boundary, which encodes on its
    own (PromptTokenizer.split), so only that end is encoded with each
    label, the whole prompt where the tokenizer has no such boundary.
    `admit` receives the prompt token count before that pass and
    `checkpoint` runs once per slot inside it; either may raise to abandon
    preparation.
    """
    tokenizer = prompt_tokenizer.tokenizer
    rendered = render_chat_template(
        tokenizer,
        messages,
        {
            "chat_template": chat_template,
            "tokenize": False,
            **template_options(
                reasoning_effort="none",
                preserve_thinking=None,
                tools=None,
                add_generation_prompt=True,
            ),
        },
    )
    if chat_format.think_start_id is not None:
        rendered += THINK_START + THINK_END
    prompt = tokenizable(rendered)
    head, tail = prompt_tokenizer.split(prompt)
    tail_ids = list(tokenizer.encode(tail, add_special_tokens=False))
    ids = head + tail_ids
    if not ids:
        raise ScoringUnsupported("the tokenizer produced an empty prompt")
    slots = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != label:
            raise ScoringUnsupported(
                f"answer slot {label!r} is not one exact round-trip token"
            )
        slots.append(encoded[0])
    if len(slots) != len(set(slots)):
        raise ScoringUnsupported("answer-slot tokens collide")
    admit(len(ids))
    for label, token in zip(labels, slots):
        checkpoint()
        encoded = tokenizer.encode(tail + label, add_special_tokens=False)
        if encoded != tail_ids + [token]:
            raise ScoringUnsupported(
                f"answer boundary changes tokenization for slot {label!r}"
            )
    return ids, slots, prompt


def judgment_response(model, row, job, result, prompt_sha256):
    logits = list(result.option_logits)
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(logits),
        "option_logits": logits,
        "input_tokens": result.prompt_tokens,
        "answer_token_ids": list(job.score_tokens),
        "prompt_sha256": prompt_sha256,
        "prompt_version": PROMPT_VERSION,
        "model": {"id": model},
        "readout": READOUT,
        "probability_status": PROBABILITY_STATUS,
        "forward_seconds": result.start_to_first_token_ms / 1000.0,
        "total_seconds": result.request_wall_ms / 1000.0,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": result.prompt_tokens,
        },
    }


@dataclass(frozen=True)
class SystemOneQuestion:
    kind: str
    instructions: object
    labels: tuple
    descriptions: tuple
    legend: dict | None
    deterministic: bool


_QUESTION_TYPES = {"noul", "choice", "score"}


def _json_description(value):
    return value is None or isinstance(value, (str, dict, list))


def _question_spec(qid, question):
    """Validate one question; returns (spec, details). Ids never infer."""
    loc = ["questions", qid]
    if not isinstance(question, dict):
        return None, [field_error(loc, "question must be an object", "model_type")]
    details = []
    kind = question.get("type")
    if not isinstance(kind, str) or kind not in _QUESTION_TYPES:
        details.append(
            field_error([*loc, "type"], "type must be noul, choice, or score")
        )
        return None, details
    instructions = question.get("instructions")
    if instructions is not None and not isinstance(instructions, (str, dict, list)):
        details.append(
            field_error(
                [*loc, "instructions"],
                "instructions must be a string, object, or array",
            )
        )
    criteria = question.get("criteria")
    labels = descriptions = None
    legend = None
    deterministic = False
    if kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict)
            or any(
                not _json_description(criteria.get(label))
                for label in ("true", "false")
            )
        ):
            details.append(
                field_error(
                    [*loc, "criteria"],
                    "noul criteria must map true/false to a string, object, "
                    "array, or null",
                )
            )
        labels = ("true", "false")
        descriptions = tuple(
            criteria.get(label)
            if isinstance(criteria, dict) and criteria.get(label) is not None
            else label
            for label in labels
        )
    elif kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            details.append(
                field_error(
                    [*loc, "criteria"],
                    "choice criteria must be a nonempty object mapping labels "
                    "to descriptions",
                )
            )
        elif len(criteria) > MAX_SCORE_TOKENS:
            details.append(
                field_error(
                    [*loc, "criteria"],
                    f"choice supports at most {MAX_SCORE_TOKENS} options",
                )
            )
        elif any(not isinstance(label, str) for label in criteria):
            details.append(
                field_error([*loc, "criteria"], "choice labels must be strings")
            )
        elif any(not _json_description(value) for value in criteria.values()):
            details.append(
                field_error(
                    [*loc, "criteria"],
                    "choice descriptions must be strings, objects, arrays, or null",
                )
            )
        else:
            labels = tuple(criteria)
            descriptions = tuple(
                value if value is not None else label
                for label, value in criteria.items()
            )
            deterministic = len(labels) == 1
    else:
        if not isinstance(criteria, list) or not criteria:
            details.append(
                field_error(
                    [*loc, "criteria"],
                    "score criteria must be a nonempty array of level descriptions",
                )
            )
        elif len(criteria) > MAX_SCORE_TOKENS:
            details.append(
                field_error(
                    [*loc, "criteria"],
                    f"score supports at most {MAX_SCORE_TOKENS} levels",
                )
            )
        elif any(not isinstance(value, (str, dict, list)) for value in criteria):
            details.append(
                field_error(
                    [*loc, "criteria"],
                    "score descriptions must be strings, objects, or arrays",
                )
            )
        else:
            labels = tuple(str(index) for index in range(len(criteria)))
            descriptions = tuple(criteria)
            legend = {str(index): value for index, value in enumerate(criteria)}
            deterministic = len(criteria) == 1
    if details:
        return None, details
    spec = SystemOneQuestion(
        kind, instructions, labels, descriptions, legend, deterministic
    )
    return spec, []


def validate_systemone(body):
    """Validate every field and question before any inference.

    Returns (state, [(question_id, spec), ...], details); callers raise
    RequestValidationError when details is nonempty.
    """
    details = []
    state = body.get("state", _MISSING)
    if state is _MISSING:
        details.append(field_error(["state"], "field required", "missing"))
    elif not isinstance(state, (str, dict, list)):
        details.append(
            field_error(["state"], "state must be a string, object, or array")
        )
        state = None
    questions = body.get("questions", _MISSING)
    specs = []
    if questions is _MISSING:
        details.append(field_error(["questions"], "field required", "missing"))
    elif not isinstance(questions, dict) or not questions:
        details.append(
            field_error(["questions"], "questions must be a nonempty object")
        )
    elif len(questions) > MAX_SYSTEMONE_QUESTIONS:
        details.append(
            field_error(
                ["questions"],
                f"questions must contain at most {MAX_SYSTEMONE_QUESTIONS} entries",
            )
        )
    else:
        for qid, question in questions.items():
            spec, errors = _question_spec(qid, question)
            details.extend(errors)
            specs.append((qid, spec))
    return state, specs, details


def systemone_messages(state, spec, slots):
    """Render one question in the direct-options-v1 shape. Structured
    instructions and descriptions stay JSON values; labels keep their
    meaning alongside the answer slot."""
    options = [
        {"slot": slot, "label": label, "description": description}
        for slot, label, description in zip(slots, spec.labels, spec.descriptions)
    ]
    payload = {"evidence": state}
    if spec.instructions is not None:
        payload["criterion"] = spec.instructions
    payload["options"] = options
    return [
        {"role": "system", "content": SYSTEMONE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def systemone_answer(spec, probabilities):
    if spec.kind == "noul":
        return {"type": "noul", "noul": probabilities[0]}
    if spec.kind == "choice":
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": spec.labels[best],
            "probabilities": dict(zip(spec.labels, probabilities)),
            "confidence": concentration(probabilities),
        }
    return {
        "type": "score",
        "score": sum(
            index * probability for index, probability in enumerate(probabilities)
        ),
        "legend": spec.legend,
        "probabilities": {
            str(index): probability for index, probability in enumerate(probabilities)
        },
        "confidence": concentration(probabilities),
    }


def deterministic_answer(spec):
    """A singleton option domain has one certain answer; no inference."""
    return systemone_answer(spec, [1.0])


# TypeSafe's SDK reads a fresh request id in this header on the routes it
# calls: /v1/systemone and the model list.
REQUEST_ID_HEADER = "x-typesafe-request-id"


class SystemOneErrors(ErrorDialect):
    """/v1/systemone answers as the FastAPI service it stands in for:
    invalid fields with 422 and what is wrong with each, an overload with
    TypeSafe's 529, and other errors as OpenAI's APIs do."""

    def answer(self, error):
        if isinstance(error, RequestValidationError):
            return 422, "unprocessable_entity", {"detail": error.details}
        status, code, payload = super().answer(error)
        return 529 if status == 503 else status, code, payload


ERRORS = SystemOneErrors()


def typed_models(models):
    """The model list's "models" entries, which TypeSafe's SDK reads
    (models.list())."""
    return [
        {
            "name": item["id"],
            "description": "Splash resident model",
            "release_date": "",
        }
        for item in models
    ]


def post_judgments(exchange, body, deadline):
    app = exchange.app
    job, prompt_sha256 = prepare_judgment(
        app, body, deadline=deadline, disconnected=exchange.client_disconnected
    )
    exchange.submit(job)
    result = await_done(exchange, job)
    exchange.send_json(
        200, judgment_response(app.response_model, body, job, result, prompt_sha256)
    )


def post_systemone(exchange, body, deadline):
    answers = {}
    input_tokens = 0
    entries = prepare_systemone(
        exchange.app,
        body,
        deadline=deadline,
        disconnected=exchange.client_disconnected,
    )
    for qid, spec, job in entries:
        if job is None:
            answers[qid] = deterministic_answer(spec)
            continue
        # One admitted job per HTTP request preserves the existing
        # queue bound and lets later questions reuse the state prefix.
        exchange.submit(job)
        result = await_done(exchange, job)
        input_tokens += result.prompt_tokens
        answers[qid] = systemone_answer(spec, softmax(list(result.option_logits)))
    exchange.send_json(
        200,
        {
            "model": exchange.app.response_model,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    )


def await_done(exchange, job):
    """The result of a score job, which emits only start and done."""
    while (event := exchange.next_event(job))[0] != "done":
        pass
    return event[1]


def _preparation_checkpoint(deadline, disconnected):
    """A check that a request's preparation may go on: its deadline has not
    passed, and its client has not left."""

    def checkpoint():
        remaining_request_time(deadline)
        if disconnected():
            raise ConnectionResetError("client disconnected during preparation")

    return checkpoint


def _encode_score_prompt(app, messages, labels, admit, checkpoint, what):
    """The tokens, answer-slot token ids and text of a scoring prompt. The
    request's input drives the render, so a failure is its error."""
    try:
        return encode_prompt(
            app.prompt_tokenizer,
            app.chat_templates.select(None).source,
            messages,
            labels,
            chat_format=app.chat_format,
            admit=admit,
            checkpoint=checkpoint,
        )
    except ScoringUnsupported as error:
        raise APIError(500, str(error), "scoring_unsupported") from error
    except (APIError, ConnectionResetError):
        # The request's own limits, deadline and client end preparation.
        raise
    except Exception as error:
        raise APIError(400, f"{what} prompt could not be rendered") from error


def prepare_judgment(app, body, *, deadline, disconnected):
    """A judgment's score job, and the digest of its rendered prompt."""
    unknown = sorted(
        set(body)
        - {
            "id",
            "state",
            "question",
            "options",
            "model",
            "timeout",
            "priority",
            "cache_salt",
        }
    )
    if unknown:
        raise APIError(400, f"unsupported fields: {', '.join(unknown)}")
    app.check_model(body)
    try:
        validate_row(body)
    except ValueError as error:
        raise APIError(400, str(error)) from error
    priority = request_fields.priority(body)
    cache_salt = request_fields.cache_salt(body)
    checkpoint = _preparation_checkpoint(deadline, disconnected)
    with app.preparation(deadline):

        def admit(prompt_tokens):
            checkpoint()
            if prompt_tokens > app.max_context:
                raise ContextLengthError(prompt_tokens, app.max_context)

        tokens, slots, prompt = _encode_score_prompt(
            app,
            judgment_messages(body),
            LETTERS[: len(body["options"])],
            admit,
            checkpoint,
            "judgment",
        )
        checkpoint()
        job = app.score_job(tokens, slots, deadline, priority, cache_salt)
    return job, digest(prompt)


def prepare_systemone(app, body, *, deadline, disconnected):
    """Each question's id, spec and score job; no job for a question its
    spec answers without the model."""
    details = []
    model = body.get("model")
    if not isinstance(model, str) or not model:
        details.append(field_error(["model"], "field required", "missing"))
    elif not app.accepts_model(model):
        details.append(
            field_error(["model"], f"model {model} is not served by this endpoint")
        )
    state, specs, question_details = validate_systemone(body)
    details.extend(question_details)
    try:
        priority = request_fields.priority(body)
    except APIError as error:
        details.append(field_error(["priority"], error.message))
    try:
        cache_salt = request_fields.cache_salt(body)
    except APIError as error:
        details.append(field_error(["cache_salt"], error.message))
    if details:
        raise RequestValidationError(details)
    jobs = []
    total_tokens = 0
    checkpoint = _preparation_checkpoint(deadline, disconnected)
    with app.preparation(deadline):
        for qid, spec in specs:
            if spec.deterministic:
                jobs.append((qid, spec, None))
                continue
            slots = slot_labels(app.tokenizer)
            if len(spec.labels) > len(slots):
                raise RequestValidationError(
                    [
                        field_error(
                            ["questions", qid, "criteria"],
                            f"the served tokenizer supports "
                            f"{len(slots)} answer slots; "
                            f"{len(spec.labels)} were requested",
                        )
                    ]
                )
            labels = slots[: len(spec.labels)]

            def admit(prompt_tokens, qid=qid, prepared=total_tokens):
                checkpoint()
                if prompt_tokens > app.max_context:
                    raise ContextLengthError(prompt_tokens, app.max_context)
                if prepared + prompt_tokens > MAX_SYSTEMONE_TOTAL_TOKENS:
                    raise RequestValidationError(
                        [
                            field_error(
                                ["questions", qid],
                                "total prepared question tokens exceed "
                                f"{MAX_SYSTEMONE_TOTAL_TOKENS}",
                            )
                        ]
                    )

            tokens, slot_ids, _ = _encode_score_prompt(
                app,
                systemone_messages(state, spec, labels),
                labels,
                admit,
                checkpoint,
                "question",
            )
            checkpoint()
            total_tokens += len(tokens)
            jobs.append(
                (
                    qid,
                    spec,
                    app.score_job(tokens, slot_ids, deadline, priority, cache_salt),
                )
            )
    return jobs

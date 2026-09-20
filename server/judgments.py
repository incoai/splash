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

"""Direct finite-option scoring shared by /v1/judgments and /v1/systemone.

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
# Native score-only requests carry at most this many option tokens.
MAX_OPTIONS = 255
# A /v1/systemone batch prepares every question before the first inference
# and runs them under one shared deadline, so the batch carries its own
# caps: at most this many questions holding at most this many prepared
# prompt tokens in total.
MAX_SYSTEMONE_QUESTIONS = 64
MAX_SYSTEMONE_TOTAL_TOKENS = 1 << 20

_MISSING = object()


class ScoringUnsupported(RuntimeError):
    """The served tokenizer cannot express exact single-token answer slots."""


class SystemOneError(Exception):
    """One or more /v1/systemone request fields failed validation."""

    def __init__(self, details):
        self.details = list(details)
        super().__init__(self.details[0]["msg"] if self.details else "invalid")


def detail(loc, msg, error_type="value_error"):
    return {"loc": ["body", *loc], "msg": msg, "type": error_type}


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
    """Normalized-entropy concentration in [0, 1].

    This is a local measure of how spread the option distribution is. It is
    not a calibrated confidence and makes no parity claim with any hosted
    judgment service.
    """
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
                if len(labels) >= MAX_OPTIONS:
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


def encode_prompt(tokenizer, messages, labels, *, admit=None, checkpoint=None):
    """Render messages and verify single-token answer slots.

    Mirrors SemIf semif_phase1.direct.encode_prompt: each slot label must be
    one exact round-trip token, and appending the label to the rendered
    prompt must extend the token ids by exactly that token.

    The boundary pass re-tokenizes the whole prompt once per slot, so a long
    prompt with many options costs far more than the prompt itself. `admit`
    receives the prepared prompt token count before that pass begins and
    `checkpoint` runs once per slot inside it; either may raise to abandon
    preparation. Omitting both reproduces the upstream behavior exactly.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = list(tokenizer.encode(prompt, add_special_tokens=False))
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
    if admit is not None:
        admit(len(ids))
    for label, token in zip(labels, slots):
        if checkpoint is not None:
            checkpoint()
        if tokenizer.encode(prompt + label, add_special_tokens=False) != ids + [token]:
            raise ScoringUnsupported(
                f"answer boundary changes tokenization for slot {label!r}"
            )
    return ids, slots, prompt


def judgment_response(model, row, meta, result):
    logits = list(result.option_logits)
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(logits),
        "option_logits": logits,
        "input_tokens": result.prompt_tokens,
        "answer_token_ids": list(meta["answer_token_ids"]),
        "prompt_sha256": meta["prompt_sha256"],
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
        return None, [detail(loc, "question must be an object", "model_type")]
    details = []
    kind = question.get("type")
    if not isinstance(kind, str) or kind not in _QUESTION_TYPES:
        details.append(detail([*loc, "type"], "type must be noul, choice, or score"))
        return None, details
    instructions = question.get("instructions")
    if instructions is not None and not isinstance(instructions, (str, dict, list)):
        details.append(
            detail(
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
                detail(
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
                detail(
                    [*loc, "criteria"],
                    "choice criteria must be a nonempty object mapping labels "
                    "to descriptions",
                )
            )
        elif len(criteria) > MAX_OPTIONS:
            details.append(
                detail(
                    [*loc, "criteria"],
                    f"choice supports at most {MAX_OPTIONS} options",
                )
            )
        elif any(not isinstance(label, str) for label in criteria):
            details.append(detail([*loc, "criteria"], "choice labels must be strings"))
        elif any(not _json_description(value) for value in criteria.values()):
            details.append(
                detail(
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
                detail(
                    [*loc, "criteria"],
                    "score criteria must be a nonempty array of level descriptions",
                )
            )
        elif len(criteria) > MAX_OPTIONS:
            details.append(
                detail(
                    [*loc, "criteria"],
                    f"score supports at most {MAX_OPTIONS} levels",
                )
            )
        elif any(not isinstance(value, (str, dict, list)) for value in criteria):
            details.append(
                detail(
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
    SystemOneError when details is nonempty.
    """
    details = []
    state = body.get("state", _MISSING)
    if state is _MISSING:
        details.append(detail(["state"], "field required", "missing"))
    elif not isinstance(state, (str, dict, list)):
        details.append(detail(["state"], "state must be a string, object, or array"))
        state = None
    questions = body.get("questions", _MISSING)
    specs = []
    if questions is _MISSING:
        details.append(detail(["questions"], "field required", "missing"))
    elif not isinstance(questions, dict) or not questions:
        details.append(detail(["questions"], "questions must be a nonempty object"))
    elif len(questions) > MAX_SYSTEMONE_QUESTIONS:
        details.append(
            detail(
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

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
from dataclasses import dataclass, replace

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one "
    "listed option. Respond with only its uppercase letter, with no explanation "
    "or reasoning."
)
SYSTEMONE_SYSTEM = DIRECT_SYSTEM.replace("uppercase letter", "uppercase slot")
SYSTEMONE_THINKING_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one "
    "listed option. Think it through, then respond with only its uppercase slot."
)
SYSTEMONE_VALUE_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Respond with only "
    "the requested JSON answer, with no explanation or reasoning."
)
SYSTEMONE_VALUE_THINKING_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Think it through, "
    "then respond with only the requested JSON answer."
)
FORCED_THINKING_CLOSE = "\n\nI will now give the final answer.\n</think>\n\n"
VALUE_PREFILL = '{"answer":'
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
# TypeLLM bounds: at most 6! permutation variants per question, 64 request
# images, and a 32 KB open string answer.
MAX_SYSTEMONE_PERMUTATIONS = 720
MAX_SYSTEMONE_IMAGES = 64
MAX_SYSTEMONE_TEXT_LENGTH = 32768
DEFAULT_NUMERIC_MAX_DIGITS = 32
DEFAULT_TEXT_MAX_TOKENS = 512
THINKING_TAIL_RESERVE = 16

FINITE_QUESTION_TYPES = frozenset(("noul", "choice", "score"))
OPEN_QUESTION_TYPES = frozenset(("string", "integer", "number"))
SYSTEMONE_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
SYSTEMONE_EXECUTIONS = ("auto", "batch", "sequential", "dag")
SYSTEMONE_MODES = ("argmax", "sample")
_EXTENDED_REQUEST_FIELDS = frozenset(
    (
        "thinking",
        "thinking_budget",
        "reasoning_effort",
        "execution",
        "mode",
        "temperature",
        "seed",
        "images",
        "numeric_max_digits",
        "text_max_tokens",
        "return_reasoning",
    )
)
_EXTENDED_QUESTION_FIELDS = frozenset(
    (
        "thinking",
        "thinking_budget",
        "permutations",
        "depends_on",
        "nullable",
        "maxLength",
        "minimum",
        "maximum",
    )
)

_MISSING = object()
_INHERIT = object()


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


def encode_prompt(tokenizer, chat_template, messages, labels, *, admit, checkpoint):
    """Render messages with chat_template and verify single-token answer slots.

    Mirrors SemIf semif_phase1.direct.encode_prompt: each slot label must be
    one exact round-trip token, and appending the label to the rendered
    prompt must extend the token ids by exactly that token.

    The boundary pass re-tokenizes the whole prompt once per slot, so a long
    prompt with many options costs far more than the prompt itself. `admit`
    receives the prepared prompt token count before that pass begins and
    `checkpoint` runs once per slot inside it; either may raise to abandon
    preparation.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
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
    admit(len(ids))
    for label, token in zip(labels, slots):
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
    thinking: bool | None = None
    # Three states: _INHERIT (absent), None (explicit null clears the
    # inherited budget), or a positive integer.
    thinking_budget: object = _INHERIT
    permutations: int | str | None = None
    depends_on: tuple | None = None
    nullable: bool = False
    max_length: int | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None

    def resolved_thinking(self, options):
        return options.thinking if self.thinking is None else self.thinking

    def resolved_budget(self, options):
        if self.thinking_budget is _INHERIT:
            return options.thinking_budget
        return self.thinking_budget


@dataclass(frozen=True)
class SystemOneOptions:
    """Top-level extension fields; every field defaults to its documented
    value so a no-extension request could carry this object unchanged."""

    thinking: bool = False
    thinking_budget: int | None = None
    reasoning_effort: str | None = None
    execution: str = "auto"
    mode: str = "argmax"
    temperature: float = 1.0
    seed: int | None = None
    images: tuple = ()
    numeric_max_digits: int = DEFAULT_NUMERIC_MAX_DIGITS
    text_max_tokens: int = DEFAULT_TEXT_MAX_TOKENS
    return_reasoning: bool = False


@dataclass
class SystemOnePlan:
    """Validated extended request; the executor prepares prompts per layer."""

    state: object
    specs: list
    options: SystemOneOptions
    images: object
    priority: int


_QUESTION_TYPES = FINITE_QUESTION_TYPES | OPEN_QUESTION_TYPES


def systemone_is_extended(body):
    """Extension detection: any extension key, even at its default value,
    or an open question type selects the new executor."""
    if not isinstance(body, dict):
        return False
    if any(field in body for field in _EXTENDED_REQUEST_FIELDS):
        return True
    questions = body.get("questions")
    if isinstance(questions, dict):
        for question in questions.values():
            if not isinstance(question, dict):
                continue
            if question.get("type") in OPEN_QUESTION_TYPES:
                return True
            if any(field in question for field in _EXTENDED_QUESTION_FIELDS):
                return True
    return False


def _json_description(value):
    return value is None or isinstance(value, (str, dict, list))


def _question_spec(qid, question, known_ids):
    """Validate one question; returns (spec, details). Ids never infer."""
    loc = ["questions", qid]
    if not isinstance(question, dict):
        return None, [detail(loc, "question must be an object", "model_type")]
    details = []
    kind = question.get("type")
    if not isinstance(kind, str) or kind not in _QUESTION_TYPES:
        details.append(
            detail(
                [*loc, "type"],
                "type must be noul, choice, score, string, integer, or number",
            )
        )
        return None, details
    open_type = kind in OPEN_QUESTION_TYPES
    instructions = question.get("instructions")
    if instructions is not None and not isinstance(instructions, (str, dict, list)):
        details.append(
            detail(
                [*loc, "instructions"],
                "instructions must be a string, object, or array",
            )
        )
    thinking = question.get("thinking")
    if "thinking" in question and not isinstance(thinking, bool):
        details.append(detail([*loc, "thinking"], "thinking must be a boolean"))
    thinking_budget = _INHERIT
    if "thinking_budget" in question:
        thinking_budget = question["thinking_budget"]
        if thinking_budget is not None and (
            not isinstance(thinking_budget, int)
            or isinstance(thinking_budget, bool)
            or thinking_budget < 1
        ):
            details.append(
                detail(
                    [*loc, "thinking_budget"],
                    "thinking_budget must be an integer >= 1 or null",
                )
            )
            thinking_budget = _INHERIT
    permutations = None
    if "permutations" in question:
        permutations = question["permutations"]
        if open_type:
            details.append(
                detail(
                    [*loc, "permutations"],
                    "permutations is only supported for noul, choice, or score",
                )
            )
            permutations = None
        elif not (
            permutations == "all"
            or (
                isinstance(permutations, int)
                and not isinstance(permutations, bool)
                and permutations >= 1
            )
        ):
            details.append(
                detail(
                    [*loc, "permutations"],
                    "permutations must be a positive integer or 'all'",
                )
            )
            permutations = None
    depends_on = None
    if "depends_on" in question:
        depends_on = question["depends_on"]
        if (
            not isinstance(depends_on, list)
            or any(
                not isinstance(dependency, str) or not dependency
                for dependency in depends_on
            )
            or len(set(depends_on)) != len(depends_on)
        ):
            details.append(
                detail(
                    [*loc, "depends_on"],
                    "depends_on must be a list of unique, nonempty question ids",
                )
            )
            depends_on = None
        else:
            depends_on = tuple(depends_on)
            for dependency in depends_on:
                if dependency == qid:
                    details.append(
                        detail(
                            [*loc, "depends_on"],
                            "a question cannot depend on itself",
                        )
                    )
                elif dependency not in known_ids:
                    details.append(
                        detail(
                            [*loc, "depends_on"],
                            f"depends_on references unknown question {dependency!r}",
                        )
                    )
    nullable = question.get("nullable")
    if "nullable" in question:
        if open_type:
            if not isinstance(nullable, bool):
                details.append(detail([*loc, "nullable"], "nullable must be a boolean"))
                nullable = False
        else:
            details.append(
                detail(
                    [*loc, "nullable"],
                    "nullable is only supported for open types",
                )
            )
            nullable = False
    max_length = None
    if "maxLength" in question:
        max_length = question["maxLength"]
        if kind != "string":
            details.append(
                detail(
                    [*loc, "maxLength"],
                    "maxLength is only supported for string questions",
                )
            )
            max_length = None
        elif (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or not 0 <= max_length <= MAX_SYSTEMONE_TEXT_LENGTH
        ):
            details.append(
                detail(
                    [*loc, "maxLength"],
                    f"maxLength must be an integer in [0, {MAX_SYSTEMONE_TEXT_LENGTH}]",
                )
            )
            max_length = None
    minimum = maximum = None
    for keyword in ("minimum", "maximum"):
        if keyword not in question:
            continue
        bound = question[keyword]
        if kind not in ("integer", "number"):
            details.append(
                detail(
                    [*loc, keyword],
                    f"{keyword} is only supported for integer and number questions",
                )
            )
            continue
        valid = (
            type(bound) is int
            if kind == "integer"
            else type(bound) is int or (type(bound) is float and math.isfinite(bound))
        )
        if not valid:
            details.append(
                detail(
                    [*loc, keyword],
                    f"{keyword} must be an integer"
                    if kind == "integer"
                    else f"{keyword} must be a finite number",
                )
            )
            continue
        if keyword == "minimum":
            minimum = bound
        else:
            maximum = bound
    if (
        kind in ("integer", "number")
        and minimum is not None
        and maximum is not None
        and minimum > maximum
    ):
        details.append(detail([*loc, "minimum"], "minimum must not exceed maximum"))
        minimum = maximum = None
    criteria = question.get("criteria")
    if open_type:
        if criteria is not None:
            details.append(
                detail(
                    [*loc, "criteria"],
                    f"{kind} questions do not accept criteria",
                )
            )
        labels = descriptions = ()
        legend = None
        deterministic = False
        if details:
            return None, details
        spec = SystemOneQuestion(
            kind,
            instructions,
            labels,
            descriptions,
            legend,
            deterministic,
            thinking,
            thinking_budget,
            None,
            depends_on,
            bool(nullable),
            max_length,
            minimum,
            maximum,
        )
        return spec, []
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
    if permutations is not None and labels is not None:
        total = math.factorial(len(labels))
        count = total if permutations == "all" else min(permutations, total)
        if count > MAX_SYSTEMONE_PERMUTATIONS:
            details.append(
                detail(
                    [*loc, "permutations"],
                    f"permutations exceeds {MAX_SYSTEMONE_PERMUTATIONS} variants",
                )
            )
    if details:
        return None, details
    spec = SystemOneQuestion(
        kind,
        instructions,
        labels,
        descriptions,
        legend,
        deterministic,
        thinking,
        thinking_budget,
        permutations,
        depends_on,
        False,
        None,
        None,
        None,
    )
    return spec, []


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _options_spec(body, specs):
    """Validate the top-level extension fields; returns (options, details)."""
    details = []
    thinking = body.get("thinking", False)
    if not isinstance(thinking, bool):
        details.append(detail(["thinking"], "thinking must be a boolean"))
        thinking = False
    thinking_budget = body.get("thinking_budget")
    if thinking_budget is not None and (
        not _is_int(thinking_budget) or thinking_budget < 1
    ):
        details.append(
            detail(
                ["thinking_budget"],
                "thinking_budget must be an integer >= 1 or null",
            )
        )
        thinking_budget = None
    reasoning_effort = body.get("reasoning_effort")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str)
        or reasoning_effort not in SYSTEMONE_REASONING_EFFORTS
    ):
        details.append(
            detail(
                ["reasoning_effort"],
                "reasoning_effort must be minimal, low, medium, high, xhigh, or max",
            )
        )
        reasoning_effort = None
    execution = body.get("execution", "auto")
    if not isinstance(execution, str) or execution not in SYSTEMONE_EXECUTIONS:
        details.append(
            detail(
                ["execution"],
                "execution must be auto, batch, sequential, or dag",
            )
        )
        execution = "auto"
    mode = body.get("mode", "argmax")
    if not isinstance(mode, str) or mode not in SYSTEMONE_MODES:
        details.append(detail(["mode"], "mode must be argmax or sample"))
        mode = "argmax"
    temperature = body.get("temperature", 1.0)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0 < temperature <= 2
    ):
        details.append(
            detail(["temperature"], "temperature must be a finite number in (0, 2]")
        )
        temperature = 1.0
    seed = body.get("seed")
    if seed is not None and (not _is_int(seed) or not 0 <= seed < 2**64):
        details.append(detail(["seed"], "seed must be an integer in [0, 2**64)"))
        seed = None
    images = body.get("images")
    if images is None:
        images = ()
    elif (
        not isinstance(images, list)
        or not 1 <= len(images) <= MAX_SYSTEMONE_IMAGES
        or any(
            not isinstance(url, str) or not url.startswith("data:") for url in images
        )
    ):
        details.append(
            detail(["images"], f"images must be 1-{MAX_SYSTEMONE_IMAGES} data: URLs")
        )
        images = ()
    else:
        images = tuple(images)
    numeric_max_digits = body.get("numeric_max_digits", DEFAULT_NUMERIC_MAX_DIGITS)
    if not _is_int(numeric_max_digits) or not 1 <= numeric_max_digits <= 64:
        details.append(
            detail(
                ["numeric_max_digits"],
                "numeric_max_digits must be an integer in [1, 64]",
            )
        )
        numeric_max_digits = DEFAULT_NUMERIC_MAX_DIGITS
    options = SystemOneOptions(
        thinking=thinking,
        thinking_budget=thinking_budget,
        reasoning_effort=reasoning_effort,
        execution=execution,
        mode=mode,
        temperature=float(temperature),
        seed=seed,
        images=images,
        numeric_max_digits=numeric_max_digits,
        return_reasoning=_return_reasoning(body, details),
    )
    if reasoning_effort is not None and not any(
        spec is not None and spec.resolved_thinking(options) for _, spec in specs
    ):
        details.append(
            detail(
                ["reasoning_effort"],
                "reasoning_effort requires a question with thinking enabled",
            )
        )
    if execution in ("batch", "sequential") and any(
        spec is not None and spec.depends_on is not None for _, spec in specs
    ):
        details.append(
            detail(
                ["execution"],
                "depends_on requires execution auto or dag",
            )
        )
    return options, details


def _return_reasoning(body, details):
    value = body.get("return_reasoning", False)
    if not isinstance(value, bool):
        details.append(
            detail(["return_reasoning"], "return_reasoning must be a boolean")
        )
        return False
    return value


def validate_systemone(body, text_max_token_cap):
    """Validate every field and question before any inference.

    Returns (state, [(question_id, spec), ...], options, details); callers
    raise SystemOneError when details is nonempty.
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
        known_ids = set(questions)
        for qid, question in questions.items():
            spec, errors = _question_spec(qid, question, known_ids)
            details.extend(errors)
            specs.append((qid, spec))
    options, option_details = _options_spec(body, specs)
    details.extend(option_details)
    text_max_tokens = body.get("text_max_tokens", DEFAULT_TEXT_MAX_TOKENS)
    if "text_max_tokens" in body and (
        not _is_int(text_max_tokens) or not 1 <= text_max_tokens <= text_max_token_cap
    ):
        details.append(
            detail(
                ["text_max_tokens"],
                f"text_max_tokens must be an integer in [1, {text_max_token_cap}]",
            )
        )
        text_max_tokens = DEFAULT_TEXT_MAX_TOKENS
    options = replace(options, text_max_tokens=text_max_tokens)
    # Cross-question check: a dependency cycle lands on the first question in
    # declaration order that still cannot resolve. Skipped when a depends_on
    # field already failed shape or reference validation.
    if not any("depends_on" in error["loc"] for error in details):
        remaining = {qid for qid, spec in specs if spec is not None}
        done = set()
        while remaining:
            ready = {
                qid
                for qid, spec in specs
                if spec is not None
                and qid in remaining
                and set(spec.depends_on or ()) <= done
            }
            if not ready:
                first = next(qid for qid, _ in specs if qid in remaining)
                details.append(
                    detail(
                        ["questions", first, "depends_on"],
                        "question dependencies contain a cycle",
                    )
                )
                break
            done.update(ready)
            remaining -= ready
    return state, specs, options, details


def _user_content(payload, image_urls):
    text = json.dumps(payload, ensure_ascii=False)
    if not image_urls:
        return text
    return [
        *({"type": "image_url", "image_url": {"url": url}} for url in image_urls),
        {"type": "text", "text": text},
    ]


def _dependency_results(dependencies):
    entries = []
    for dep_spec, selected in dependencies:
        entry = {}
        if dep_spec.instructions is not None:
            entry["criterion"] = dep_spec.instructions
        entry["answer"] = selected
        entries.append(entry)
    return entries


def systemone_messages(
    state, spec, slots, *, thinking=False, order=None, dependencies=(), image_urls=()
):
    """Render one question in the direct-options-v1 shape. Structured
    instructions and descriptions stay JSON values; labels keep their
    meaning alongside the answer slot. The optional keyword arguments add
    thinking, permutation, dependency and image support without changing
    the default rendering."""
    if order is None:
        order = range(len(spec.labels))
    options = [
        {
            "slot": slots[position],
            "label": spec.labels[order[position]],
            "description": spec.descriptions[order[position]],
        }
        for position in range(len(slots))
    ]
    payload = {"evidence": state}
    if dependencies:
        payload["dependency_results"] = _dependency_results(dependencies)
    if spec.instructions is not None:
        payload["criterion"] = spec.instructions
    payload["options"] = options
    return [
        {
            "role": "system",
            "content": SYSTEMONE_THINKING_SYSTEM if thinking else SYSTEMONE_SYSTEM,
        },
        {"role": "user", "content": _user_content(payload, image_urls)},
    ]


def value_type_text(spec):
    """TypeLLM Type line for open answers, e.g. 'number or null, minimum 0'."""
    kind = spec.kind + (" or null" if spec.nullable else "")
    if spec.kind == "string":
        return kind + (
            "" if spec.max_length is None else f", at most {spec.max_length} characters"
        )
    bounds = [
        f"{word} {json.dumps(value)}"
        for word, value in (("minimum", spec.minimum), ("maximum", spec.maximum))
        if value is not None
    ]
    return ", ".join([kind, *bounds])


def value_format_text(spec):
    """TypeLLM Answer line for open answers."""
    text = (
        f'Answer as {{"answer": <{spec.kind}{" or null" if spec.nullable else ""}>}}.'
    )
    if spec.kind == "number":
        text += " Do not use exponent notation."
    if spec.nullable:
        text += " Return null only if there is no value."
    return text


def systemone_value_messages(
    state, spec, *, thinking=False, dependencies=(), image_urls=()
):
    """Render an open (string/integer/number) question."""
    payload = {"evidence": state}
    if dependencies:
        payload["dependency_results"] = _dependency_results(dependencies)
    if spec.instructions is not None:
        payload["criterion"] = spec.instructions
    payload["answer_type"] = value_type_text(spec)
    payload["answer_format"] = value_format_text(spec)
    return [
        {
            "role": "system",
            "content": (
                SYSTEMONE_VALUE_THINKING_SYSTEM if thinking else SYSTEMONE_VALUE_SYSTEM
            ),
        },
        {"role": "user", "content": _user_content(payload, image_urls)},
    ]


def systemone_answer(spec, probabilities, *, selected=None):
    if spec.kind == "noul":
        return {"type": "noul", "noul": probabilities[0]}
    if spec.kind == "choice":
        if selected is None:
            selected = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": spec.labels[selected],
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


def deterministic_value(spec):
    """The selected value a singleton question feeds to its dependents."""
    if spec.kind == "choice":
        return spec.labels[0]
    if spec.kind == "score":
        return spec.descriptions[0]
    return spec.labels[0]


def open_answer(spec, value, null_probability=None):
    answer = {"type": spec.kind, spec.kind: value}
    if spec.nullable:
        answer["null_probability"] = null_probability
    return answer


def argmax_index(probabilities):
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def sample_index(probabilities, rng):
    """TypeLLM _sample: a cumulative-sum threshold over option order."""
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


def logsumexp(values):
    pivot = max(values)
    return pivot + math.log(math.fsum(math.exp(value - pivot) for value in values))


def resolve_execution(specs, execution):
    if execution != "auto":
        return execution
    return "dag" if any(spec.depends_on is not None for _, spec in specs) else "batch"


def dependency_layers(pairs):
    """Stable topological layers, or None on a cycle (validation reports
    cycles first, so a caller after validation always gets layers)."""
    remaining = list(pairs)
    done = set()
    layers = []
    while remaining:
        layer = [pair for pair in remaining if set(pair[1].depends_on or ()) <= done]
        if not layer:
            return None
        layers.append(layer)
        done.update(qid for qid, _ in layer)
        remaining = [pair for pair in remaining if pair[0] not in done]
    return layers


def execution_layers(pairs, execution):
    """Batch keeps one layer; sequential one per question; dag topological."""
    if execution == "batch":
        return [list(pairs)]
    if execution == "sequential":
        return [[pair] for pair in pairs]
    layers = dependency_layers(pairs)
    if layers is None:
        raise SystemOneError(
            [detail(["questions", pairs[0][0], "depends_on"], "dependency cycle")]
        )
    return layers


def visible_dependencies(pairs, execution):
    """Question ids whose answers a question sees, in declaration order."""
    order = [qid for qid, _ in pairs]
    if execution == "sequential":
        return {qid: order[:index] for index, (qid, _) in enumerate(pairs)}
    if execution == "dag":
        by_qid = dict(pairs)
        visible = {}
        for qid, spec in pairs:
            ancestors = set()
            stack = list(spec.depends_on or ())
            while stack:
                dep = stack.pop()
                if dep in ancestors or dep not in by_qid:
                    continue
                ancestors.add(dep)
                stack.extend(by_qid[dep].depends_on or ())
            visible[qid] = [dep for dep in order if dep in ancestors]
        return visible
    return {qid: [] for qid, _ in pairs}


def _unrank_permutation(rank, size):
    available = list(range(size))
    order = []
    while available:
        index, rank = divmod(rank, math.factorial(len(available) - 1))
        order.append(available.pop(index))
    return tuple(order)


def permutation_orders(permutations, size, rng):
    """Display orders σ where σ[i] is the original option index at position
    i. The identity order always leads; 'all' enumerates lexicographically;
    an integer budget fills with distinct random ranks."""
    total = math.factorial(size)
    count = total if permutations == "all" else min(permutations or 1, total)
    if count <= 1:
        return [tuple(range(size))]
    if count == total:
        return list(itertools.permutations(range(size)))
    ranks = []
    seen = {0}
    while len(ranks) < count - 1:
        rank = rng.randrange(total)
        if rank not in seen:
            seen.add(rank)
            ranks.append(rank)
    return [tuple(range(size)), *(_unrank_permutation(rank, size) for rank in ranks)]


_SLOT_BOUNDARY = weakref.WeakKeyDictionary()


def _check_slot_boundary(tokenizer):
    labels = slot_labels(tokenizer)
    close = tokenizer.encode("</think>\n\n", add_special_tokens=False)
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
    for label, token in zip(labels, slots):
        if tokenizer.encode("</think>\n\n" + label, add_special_tokens=False) != list(
            close
        ) + [token]:
            raise ScoringUnsupported(
                f"answer boundary changes tokenization for slot {label!r}"
            )
    return tuple(close), tuple(slots)


def slot_boundary(tokenizer):
    """(close_ids, slot_ids) verified once per tokenizer: each slot label is
    one round-trip token right after the thinking close."""
    try:
        cached = _SLOT_BOUNDARY.get(tokenizer)
    except TypeError:
        cached = None
    if cached is None:
        cached = _check_slot_boundary(tokenizer)
        try:
            _SLOT_BOUNDARY[tokenizer] = cached
        except TypeError:
            pass
    return cached


def answer_slots(tokenizer, rendered, ids, count, *, thinking):
    """Slot ids for one prepared prompt, verified at its answer boundary.

    Thinking prompts close later with the same '</think>\n\n' boundary the
    cached check covers; a non-thinking prompt that ends there needs no
    re-tokenization either. Anything else falls back to the legacy
    whole-prompt boundary check."""
    close, slots = slot_boundary(tokenizer)
    labels = slot_labels(tokenizer)
    if thinking or (
        rendered.endswith("</think>\n\n") and tuple(ids[-len(close) :]) == close
    ):
        return slots[:count]
    for index in range(count):
        if tokenizer.encode(rendered + labels[index], add_special_tokens=False) != (
            list(ids) + [slots[index]]
        ):
            raise ScoringUnsupported(
                f"answer boundary changes tokenization for slot {labels[index]!r}"
            )
    return slots[:count]


_VALUE_STARTS = weakref.WeakKeyDictionary()


def _derive_value_starts(tokenizer):
    key = tokenizer.encode(VALUE_PREFILL, add_special_tokens=False)
    starts = {"null": [], "positive": [], "negative": [], "string": []}
    probes = (
        ("null", " null}", lambda piece: piece.strip() == "null"),
        ("positive", " 1}", lambda piece: piece != "" and piece.strip() == ""),
        ("negative", " -1}", lambda piece: piece.strip() == "-"),
        ("string", ' "a"}', lambda piece: piece.strip() == '"'),
        ("string", ' ""}', lambda piece: piece.strip() == '""'),
    )
    for kind, suffix, accept in probes:
        ids = tokenizer.encode(VALUE_PREFILL + suffix, add_special_tokens=False)
        if ids[: len(key)] != key or len(ids) <= len(key):
            continue
        token = ids[len(key)]
        if accept(tokenizer.decode([token])) and token not in starts[kind]:
            starts[kind].append(token)
    direct = []
    for char in "-" + string.digits:
        encoded = tokenizer.encode(char, add_special_tokens=False)
        if len(encoded) == 1 and tokenizer.decode(encoded) == char:
            direct.append(encoded[0])
    return {
        "null": tuple(starts["null"]),
        "positive": tuple(starts["positive"]),
        "negative": tuple(starts["negative"]),
        "string": tuple(starts["string"]),
        "direct": tuple(direct),
    }


def value_start_tokens(tokenizer):
    """First-token ids for each JSON value kind after '{"answer":', cached
    per tokenizer."""
    try:
        cached = _VALUE_STARTS.get(tokenizer)
    except TypeError:
        cached = None
    if cached is None:
        cached = _derive_value_starts(tokenizer)
        try:
            _VALUE_STARTS[tokenizer] = cached
        except TypeError:
            pass
    return cached


def value_grammar(spec, max_digits):
    """llguidance grammar for the '{"answer": VALUE}' continuation."""
    if spec.kind == "string":
        schema = {"type": "string"}
        if spec.max_length is not None:
            schema["maxLength"] = spec.max_length
        rule = "%json " + json.dumps(schema, separators=(",", ":"))
        extra = ""
    elif spec.minimum is not None or spec.maximum is not None:
        schema = {"type": spec.kind}
        if spec.minimum is not None:
            schema["minimum"] = spec.minimum
        if spec.maximum is not None:
            schema["maximum"] = spec.maximum
        rule = "%json " + json.dumps(schema, separators=(",", ":"))
        extra = ""
    else:
        rule = "NUMBER"
        if spec.kind == "integer":
            pattern = f"-?(?:0|[1-9][0-9]{{0,{max_digits - 1}}})"
        else:
            alternatives = [
                "0" + (f"(?:\\.[0-9]{{1,{max_digits - 1}}})?" if max_digits > 1 else "")
            ]
            for digits in range(1, max_digits + 1):
                alternative = "[1-9]" + (f"[0-9]{{{digits - 1}}}" if digits > 1 else "")
                if max_digits - digits >= 1:
                    alternative += f"(?:\\.[0-9]{{1,{max_digits - digits}}})?"
                alternatives.append(alternative)
            pattern = "-?(?:" + "|".join(alternatives) + ")"
        extra = f"NUMBER: /{pattern}/\n"
    return f'%llguidance {{}}\nstart: WS {rule} "}}"\nWS: /[ ]?/\n{extra}'


def check_open_value(spec, value):
    """Validate a generated open value; returns it normalized."""
    if spec.kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"answer is not a string: {value!r}")
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ValueError("answer contains an unpaired surrogate")
        if spec.max_length is not None and len(value) > spec.max_length:
            raise ValueError(f"answer exceeds maxLength {spec.max_length}")
        return value
    if spec.kind == "integer":
        if type(value) is not int:
            raise ValueError(f"answer is not an integer: {value!r}")
    else:
        if type(value) not in (int, float):
            raise ValueError(f"answer is not a number: {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("answer is not finite")
    if spec.minimum is not None and value < spec.minimum:
        raise ValueError(f"answer {value} is below minimum {spec.minimum}")
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError(f"answer {value} is above maximum {spec.maximum}")
    return value

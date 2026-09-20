"""Incremental model-output parsing and final tool/answer validation."""

import re

from jsonschema.exceptions import ValidationError
from referencing.exceptions import Unresolvable

if __package__:
    from . import json_codec
    from .errors import APIError
    from .schema_validation import SchemaEvaluationError
    from .tool_schema import (
        FUNCTION_CLOSE,
        FUNCTION_OPEN,
        PARAMETER_CLOSE,
        PARAMETER_OPEN,
        THINK_END,
        TOOL_CALL_CLOSE,
        TOOL_CALL_OPEN,
        json_value,
        raw_string_schema,
    )
else:
    import json_codec
    from errors import APIError
    from schema_validation import SchemaEvaluationError
    from tool_schema import (
        FUNCTION_CLOSE,
        FUNCTION_OPEN,
        PARAMETER_CLOSE,
        PARAMETER_OPEN,
        THINK_END,
        TOOL_CALL_CLOSE,
        TOOL_CALL_OPEN,
        json_value,
        raw_string_schema,
    )


TOOL_ARGUMENT_DELTA_CHARS = 16 * 1024
_SURROGATE = re.compile("[\ud800-\udfff]")


def _validate_tool_unicode(value):
    # Validate decoded values, so literal backslash-u text remains unchanged.
    if isinstance(value, str):
        if _SURROGATE.search(value):
            raise APIError(
                500,
                "model returned invalid Unicode in tool arguments",
                "invalid_model_output",
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_tool_unicode(key)
            _validate_tool_unicode(item)
    elif isinstance(value, list):
        for item in value:
            _validate_tool_unicode(item)


def _tool_json(value):
    _validate_tool_unicode(value)
    return json_codec.dumps(value)


def hold_partial(text, marker):
    for length in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:length]):
            return text[:-length], text[-length:]
    return text, ""


class ReasoningSplitter:
    def __init__(self, thinking):
        self.reasoning = thinking
        self.pending = ""

    def put(self, text):
        if not self.reasoning:
            return [("content", text)]
        self.pending += text
        end = self.pending.find(THINK_END)
        if end >= 0:
            output = [("reasoning_content", self.pending[:end])]
            content = self.pending[end + len(THINK_END) :]
            if content:
                output.append(("content", content))
            self.pending = ""
            self.reasoning = False
            return [(kind, value) for kind, value in output if value]
        ready, self.pending = hold_partial(self.pending, THINK_END)
        return [("reasoning_content", ready)] if ready else []

    def finish(self):
        if not self.pending:
            return []
        kind = "reasoning_content" if self.reasoning else "content"
        text, self.pending = self.pending, ""
        return [(kind, text)]


class StreamingToolCallProjector:
    """Stream Qwen tool XML as OpenAI JSON argument deltas.

    Emit function names before their arguments finish. Validate each closed
    call before its closing JSON brace, then validate the complete response
    at request completion.
    """

    _FUNCTION_PREFIX = FUNCTION_OPEN
    _PARAMETER_PREFIX = PARAMETER_OPEN
    _PARAMETER_CLOSE = PARAMETER_CLOSE
    _FUNCTION_CLOSE = FUNCTION_CLOSE

    def __init__(self, policy, request_id, structured=False):
        self.policy = policy
        self.request_id = request_id
        self.pending = ""
        self.state = "output" if structured else "content"
        self.call_index = 0
        self.call_id = None
        self.function_name = None
        self.parameter_name = None
        self.parameter_schema = None
        self.parameter_root = None
        self.parameter_value_fragments = []
        self.streaming_string = False
        self.arguments = {}
        self.argument_fragments = []
        self.content_fragments = []
        self.streamed_content_fragments = []
        self.closed_calls = []

    @staticmethod
    def _malformed():
        raise APIError(500, "model returned malformed tool XML", "invalid_model_output")

    def _literal(self, value):
        if self.pending.startswith(value):
            self.pending = self.pending[len(value) :]
            return True
        if value.startswith(self.pending):
            return False
        self._malformed()

    def _emit_content(self, value, events):
        if value:
            self.content_fragments.append(value)
            # Publish text after a tool call only after full completion; a
            # max-token boundary can leave it in an unfinished tool suffix.
            if not self.closed_calls:
                if not self.streamed_content_fragments:
                    # Hold template whitespace until there is visible text.
                    # A tool-only turn must not create an empty text item.
                    if not value.strip():
                        return
                    value = "".join(self.content_fragments)
                self.streamed_content_fragments.append(value)
                events.append(("content", value))

    def _begin_call(self, events):
        name_end = self.pending.find(">\n")
        if name_end < 0:
            return False
        name = self.pending[:name_end]
        if not name or self.policy.validators.get(name) is None:
            raise APIError(
                500,
                f"model called unknown tool {name}",
                "invalid_model_output",
            )
        self.pending = self.pending[name_end + 2 :]
        self.function_name = name
        self.call_id = f"call_{self.request_id}_{self.call_index}"
        self.arguments = {}
        self.argument_fragments = ["{"]
        events.append(
            (
                "tool",
                {
                    "index": self.call_index,
                    "id": self.call_id,
                    "type": "function",
                    "function": {"name": name},
                },
            )
        )
        events.append(
            ("tool", {"index": self.call_index, "function": {"arguments": "{"}})
        )
        self.state = "body"
        return True

    def _emit_argument(self, fragment, events):
        self.argument_fragments.append(fragment)
        for chunk in argument_deltas(fragment):
            events.append(
                (
                    "tool",
                    {
                        "index": self.call_index,
                        "function": {"arguments": chunk},
                    },
                )
            )

    def _emit_string_value(self, value, events):
        if not value:
            return
        self.parameter_value_fragments.append(value)
        self._emit_argument(_tool_json(value)[1:-1], events)

    def _finish_parameter(self, events):
        value_end = self.pending.find(self._PARAMETER_CLOSE)
        if self.streaming_string and value_end < 0:
            ready, self.pending = hold_partial(self.pending, self._PARAMETER_CLOSE)
            self._emit_string_value(ready, events)
            return False
        if value_end < 0:
            return False
        raw_value = self.pending[:value_end]
        self.pending = self.pending[value_end + len(self._PARAMETER_CLOSE) :]
        if self.streaming_string:
            self._emit_string_value(raw_value, events)
            value = "".join(self.parameter_value_fragments)
            self._emit_argument('"', events)
        else:
            value = _typed_tool_value(
                raw_value, self.parameter_schema, self.parameter_root
            )
            prefix = "" if len(self.arguments) == 0 else ","
            fragment = (
                prefix + _tool_json(self.parameter_name) + ":" + _tool_json(value)
            )
            self._emit_argument(fragment, events)
        self.arguments[self.parameter_name] = value
        self.parameter_name = None
        self.parameter_schema = None
        self.parameter_root = None
        self.parameter_value_fragments = []
        self.streaming_string = False
        self.state = "body"
        return True

    def _finish_call(self, events):
        arguments = _tool_json(self.arguments)
        call = {
            "id": self.call_id,
            "type": "function",
            "function": {"name": self.function_name, "arguments": arguments},
        }
        validate_tool_calls([call], self.policy)
        self.argument_fragments.append("}")
        if "".join(self.argument_fragments) != arguments:
            raise APIError(
                500,
                "streamed tool arguments do not match canonical arguments",
                "internal_server_error",
            )
        events.append(
            ("tool", {"index": self.call_index, "function": {"arguments": "}"}})
        )
        self.closed_calls.append(call)
        self.call_index += 1
        self.call_id = None
        self.function_name = None
        self.arguments = {}
        self.argument_fragments = []
        self.state = "content"

    def put(self, text):
        self.pending += text
        events = []
        while self.pending:
            if self.state == "output":
                first = self.pending.lstrip()
                if not first:
                    break
                # A structured answer is one JSON value or tool calls. Once
                # JSON starts, XML spellings inside its strings are just data.
                self.state = "content" if first.startswith("<") else "json"
            if self.state == "json":
                self._emit_content(self.pending, events)
                self.pending = ""
                break
            if self.state == "content":
                start = self.pending.find(TOOL_CALL_OPEN)
                if start >= 0:
                    self._emit_content(self.pending[:start], events)
                    self.pending = self.pending[start + len(TOOL_CALL_OPEN) :]
                    self.state = "function_prefix"
                    continue
                ready, self.pending = hold_partial(self.pending, TOOL_CALL_OPEN)
                self._emit_content(ready, events)
                break
            if self.state == "function_prefix":
                if not self._literal(self._FUNCTION_PREFIX):
                    break
                self.state = "function_name"
                continue
            if self.state == "function_name":
                if not self._begin_call(events):
                    break
                continue
            if self.state == "body":
                if self.pending.startswith(self._PARAMETER_PREFIX):
                    self.pending = self.pending[len(self._PARAMETER_PREFIX) :]
                    self.state = "parameter_name"
                    continue
                if self.pending.startswith(self._FUNCTION_CLOSE):
                    self.pending = self.pending[len(self._FUNCTION_CLOSE) :]
                    self._finish_call(events)
                    continue
                if self._PARAMETER_PREFIX.startswith(
                    self.pending
                ) or self._FUNCTION_CLOSE.startswith(self.pending):
                    break
                self._malformed()
            if self.state == "parameter_name":
                name_end = self.pending.find(">\n")
                if name_end < 0:
                    break
                name = self.pending[:name_end]
                if not name or name in self.arguments:
                    raise APIError(
                        500,
                        "model repeated a tool parameter",
                        "invalid_model_output",
                    )
                self.pending = self.pending[name_end + 2 :]
                self.parameter_name = name
                self.parameter_schema, self.parameter_root = _tool_property_schema(
                    self.policy, self.function_name, name
                )
                string_schema = raw_string_schema(
                    self.parameter_schema, self.parameter_root
                )
                self.streaming_string = bool(
                    string_schema is not None and string_schema[0] == "raw"
                )
                self.parameter_value_fragments = []
                if self.streaming_string:
                    prefix = "" if len(self.arguments) == 0 else ","
                    self._emit_argument(
                        prefix + _tool_json(name) + ':"',
                        events,
                    )
                self.state = "parameter_value"
                continue
            if self.state == "parameter_value":
                if not self._finish_parameter(events):
                    break
                continue
        return events

    def interrupted_result(self):
        content = "".join(self.streamed_content_fragments)
        if not self.closed_calls and self.call_id is None:
            content = "".join(self.content_fragments)
        if (
            not self.closed_calls
            and self.state in ("content", "output", "json")
            and not TOOL_CALL_OPEN.startswith(self.pending)
        ):
            content += self.pending
        calls = list(self.closed_calls)
        if self.call_id is not None:
            calls.append(
                {
                    "id": self.call_id,
                    "type": "function",
                    "function": {
                        "name": self.function_name,
                        "arguments": "".join(self.argument_fragments),
                    },
                }
            )
        return ("" if calls and not content.strip() else content), calls

    def finish(self, canonical_content, canonical_calls, incomplete):
        content = []
        if self.state in ("content", "output", "json"):
            if self.pending and not (
                incomplete and TOOL_CALL_OPEN.startswith(self.pending)
            ):
                self.content_fragments.append(self.pending)
            self.pending = ""
        elif not incomplete:
            self._malformed()
        parsed_content = "".join(self.content_fragments)
        if canonical_calls and not parsed_content.strip():
            parsed_content = ""
        emitted = "".join(self.streamed_content_fragments)
        if (
            not incomplete and parsed_content != canonical_content
        ) or not canonical_content.startswith(emitted):
            raise APIError(
                500,
                "streamed content does not match canonical content",
                "internal_server_error",
            )
        remaining = canonical_content[len(emitted) :]
        if remaining:
            self.streamed_content_fragments.append(remaining)
            content.append(remaining)
        if self.closed_calls != canonical_calls:
            if not incomplete:
                raise APIError(
                    500,
                    "streamed tool calls do not match canonical tool calls",
                    "internal_server_error",
                )
        return content


def argument_deltas(arguments):
    # Keep individual SSE frames bounded even when a tool has a large string
    # argument. Callers preserve fragment order and validate the canonical JSON.
    for offset in range(0, len(arguments), TOOL_ARGUMENT_DELTA_CHARS):
        yield arguments[offset : offset + TOOL_ARGUMENT_DELTA_CHARS]


def _tool_property_schema(policy, tool_name, parameter_name):
    if policy is None:
        return None, None
    root = policy.argument_schemas.get(tool_name)
    if not isinstance(root, dict):
        return None, None
    schema = root.get("properties", {}).get(
        parameter_name, root.get("additionalProperties", {})
    )
    return schema, schema


def _typed_tool_value(value, schema, root):
    parsed = json_value(value)
    string_schema = raw_string_schema(schema, root) if root is not None else None
    if string_schema is None:
        return parsed
    if string_schema[0] == "raw" or value in string_schema[1]:
        return value
    if value == "null" and None in string_schema[1]:
        return None
    return parsed


def parse_tool_calls(text, request_id, policy=None):
    calls = []
    content, cursor = [], 0
    opening = TOOL_CALL_OPEN + FUNCTION_OPEN
    while (start := text.find(TOOL_CALL_OPEN, cursor)) >= 0:
        content.append(text[cursor:start])
        if not text.startswith(opening, start):
            raise APIError(
                500, "model returned malformed tool XML", "invalid_model_output"
            )
        name_start = start + len(opening)
        name_end = text.find(">\n", name_start)
        if name_end < 0:
            raise APIError(
                500, "model returned malformed tool XML", "invalid_model_output"
            )
        name = text[name_start:name_end]
        cursor = name_end + 2
        arguments = {}
        while text.startswith(PARAMETER_OPEN, cursor):
            parameter_start = cursor + len(PARAMETER_OPEN)
            parameter_end = text.find(">\n", parameter_start)
            if parameter_end < 0:
                raise APIError(
                    500, "model returned malformed tool XML", "invalid_model_output"
                )
            parameter_name = text[parameter_start:parameter_end]
            if parameter_name in arguments:
                raise APIError(
                    500, "model repeated a tool parameter", "invalid_model_output"
                )
            value_start = parameter_end + 2
            value_end = text.find(PARAMETER_CLOSE, value_start)
            if value_end < 0:
                raise APIError(
                    500, "model returned malformed tool XML", "invalid_model_output"
                )
            schema, root = _tool_property_schema(policy, name, parameter_name)
            arguments[parameter_name] = _typed_tool_value(
                text[value_start:value_end], schema, root
            )
            cursor = value_end + len(PARAMETER_CLOSE)
        if not text.startswith(FUNCTION_CLOSE, cursor):
            raise APIError(
                500, "model returned malformed tool XML", "invalid_model_output"
            )
        cursor += len(FUNCTION_CLOSE)
        index = len(calls)
        calls.append(
            {
                "id": f"call_{request_id}_{index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _tool_json(arguments),
                },
            }
        )
    content.append(text[cursor:])
    content = "".join(content)
    if calls and any(
        tag in content
        for tag in (
            TOOL_CALL_OPEN,
            TOOL_CALL_CLOSE,
            "<function=",
            "</function>",
            PARAMETER_OPEN,
            "</parameter>",
        )
    ):
        raise APIError(500, "model returned malformed tool XML", "invalid_model_output")
    return ("" if calls and not content.strip() else content), calls


def validate_tool_calls(calls, policy):
    if policy.required and not calls:
        raise APIError(
            500, "model did not call a required tool", "invalid_model_output"
        )
    if not policy.parallel and len(calls) > 1:
        raise APIError(
            500, "model returned parallel tool calls", "invalid_model_output"
        )
    for call in calls:
        function = call["function"]
        name = function["name"]
        validator = policy.validators.get(name)
        if validator is None:
            raise APIError(
                500, f"model called unknown tool {name}", "invalid_model_output"
            )
        try:
            arguments = json_codec.loads(function["arguments"])
            _validate_tool_unicode(arguments)
            validator.validate(arguments)
        except SchemaEvaluationError as error:
            raise APIError(500, str(error), "output_validation_failed") from error
        except ValidationError as error:
            raise APIError(
                500,
                f"invalid arguments for {name} at {error.json_path}: {error.message}",
                "invalid_model_output",
            ) from error
        except (Unresolvable, RecursionError) as error:
            raise APIError(
                500, f"could not validate tool {name}", "invalid_model_output"
            ) from error


def validate_response_content(content, validator):
    if validator is None:
        return
    try:
        value = json_codec.loads(content)
        validator.validate(value)
    except SchemaEvaluationError as error:
        raise APIError(500, str(error), "output_validation_failed") from error
    except (ValueError, ValidationError, Unresolvable, RecursionError) as error:
        raise APIError(
            500,
            f"model returned invalid structured output: {error}",
            "invalid_model_output",
        ) from error

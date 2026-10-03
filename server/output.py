"""Incremental model-output parsing, output blocks and final tool/answer
validation."""

import re
from dataclasses import dataclass

from jsonschema.exceptions import ValidationError
from referencing.exceptions import Unresolvable

from . import json_codec
from .errors import APIError
from .schema_validation import SchemaEvaluationError
from .tool_schema import (
    FUNCTION_CLOSE,
    FUNCTION_OPEN,
    PARAMETER_CLOSE,
    PARAMETER_OPEN,
    THINK_END,
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
        # Whether the newlines after </think>, which set the answer apart in
        # the chat template's layout of a turn, are still to be dropped.
        self.separator = False

    def put(self, text):
        if not self.reasoning:
            return self._content(text)
        self.pending += text
        end = self.pending.find(THINK_END)
        if end >= 0:
            reasoning = self.pending[:end]
            content = self.pending[end + len(THINK_END) :]
            self.pending = ""
            self.reasoning = False
            self.separator = True
            output = [("reasoning_content", reasoning)] if reasoning else []
            return output + self._content(content)
        ready, self.pending = hold_partial(self.pending, THINK_END)
        return [("reasoning_content", ready)] if ready else []

    def _content(self, text):
        if self.separator:
            text = text.lstrip("\n")
            if not text:
                return []
            self.separator = False
        return [("content", text)]

    def finish(self):
        if not self.pending:
            return []
        kind = "reasoning_content" if self.reasoning else "content"
        text, self.pending = self.pending, ""
        return [(kind, text)]


class StreamingToolCallProjector:
    """Parse Qwen tool XML as it arrives into OpenAI JSON argument deltas,
    for streamed and complete responses alike.

    Emit function names before their arguments finish. Validate each closed
    call before its closing JSON brace, then validate the complete response
    at request completion. Text outside calls streams as it arrives, after a
    call as before one.
    """

    def __init__(self, policy, request_id, structured=False):
        self.policy = policy
        self.request_id = request_id
        self.pending = ""
        self.state = "output" if structured else "content"
        self.call_index = 0
        self.call_id = None
        self.function_name = None
        self.parameter_name = None
        self.string_schema = None
        self.parameter_value_fragments = []
        self.streaming_string = False
        self.arguments = {}
        self.argument_fragments = []
        self.content_fragments = []
        # How many content fragments the stream has published (the rest are
        # whitespace it holds), and whether the text since the start of the
        # output or the last call has shown a visible character yet.
        self.streamed_count = 0
        self.text_visible = False
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
        if not value:
            return
        self.content_fragments.append(value)
        # The chat template sets calls apart from text with whitespace. Hold
        # whitespace that starts the output or follows a call until visible
        # text arrives. Before the first text or after the last, it only
        # frames the calls and is dropped; between two texts it separates
        # them and streams with the later one.
        if not self.text_visible:
            if not value.strip():
                return
            self.text_visible = True
        unsent = self.content_fragments[self.streamed_count :]
        self.streamed_count = len(self.content_fragments)
        events.append(("content", "".join(unsent)))

    def _streamed_content(self):
        return "".join(self.content_fragments[: self.streamed_count])

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
        if not self.streamed_count:
            # Whitespace before the first text only framed the calls.
            self.content_fragments.clear()
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
        # Only text that may begin the closing marker stays pending, so each
        # character of a value is scanned and copied a bounded number of times.
        value_end = self.pending.find(PARAMETER_CLOSE)
        if value_end < 0:
            ready, self.pending = hold_partial(self.pending, PARAMETER_CLOSE)
            if self.streaming_string:
                self._emit_string_value(ready, events)
            elif ready:
                self.parameter_value_fragments.append(ready)
            return False
        tail = self.pending[:value_end]
        self.pending = self.pending[value_end + len(PARAMETER_CLOSE) :]
        if self.streaming_string:
            self._emit_string_value(tail, events)
            value = "".join(self.parameter_value_fragments)
            self._emit_argument('"', events)
        else:
            self.parameter_value_fragments.append(tail)
            value = _typed_tool_value(
                "".join(self.parameter_value_fragments), self.string_schema
            )
            prefix = "" if len(self.arguments) == 0 else ","
            fragment = (
                prefix + _tool_json(self.parameter_name) + ":" + _tool_json(value)
            )
            self._emit_argument(fragment, events)
        self.arguments[self.parameter_name] = value
        self.parameter_name = None
        self.string_schema = None
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
        self.text_visible = False
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
                if not self._literal(FUNCTION_OPEN):
                    break
                self.state = "function_name"
                continue
            if self.state == "function_name":
                if not self._begin_call(events):
                    break
                continue
            if self.state == "body":
                if self.pending.startswith(PARAMETER_OPEN):
                    self.pending = self.pending[len(PARAMETER_OPEN) :]
                    self.state = "parameter_name"
                    continue
                if self.pending.startswith(FUNCTION_CLOSE):
                    self.pending = self.pending[len(FUNCTION_CLOSE) :]
                    self._finish_call(events)
                    continue
                if any(
                    marker.startswith(self.pending)
                    for marker in (PARAMETER_OPEN, FUNCTION_CLOSE)
                ):
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
                self.string_schema = raw_string_schema(
                    _tool_property_schema(self.policy, self.function_name, name)
                )
                self.streaming_string = (
                    self.string_schema is not None and self.string_schema[0] == "raw"
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

    def finish(self, incomplete):
        """The content and calls of the output, and the content the stream
        still owes, which put() held back to see what followed.

        Closed calls are complete, and output cut at the token limit keeps
        an open call with the arguments it has. Cut output with a call has
        the content the stream published, which leaves out whitespace that
        no visible text has followed since the start or the last call.
        Otherwise the content is the text outside calls without whitespace
        that only frames them and, when cut, a trailing partial call marker
        or unfinished call header."""
        if self.state in ("content", "output", "json"):
            if self.pending and not (
                incomplete and TOOL_CALL_OPEN.startswith(self.pending)
            ):
                self.content_fragments.append(self.pending)
            elif self.closed_calls:
                # Whitespace held after the last text only framed the calls.
                del self.content_fragments[self.streamed_count :]
            self.pending = ""
        elif not incomplete:
            self._malformed()
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
        streamed = self._streamed_content()
        content = streamed if calls and incomplete else "".join(self.content_fragments)
        return content, calls, content[len(streamed) :]


def argument_deltas(arguments):
    # Keep individual SSE frames bounded even when a tool has a large string
    # argument. Callers preserve fragment order and validate the canonical JSON.
    for offset in range(0, len(arguments), TOOL_ARGUMENT_DELTA_CHARS):
        yield arguments[offset : offset + TOOL_ARGUMENT_DELTA_CHARS]


def _tool_property_schema(policy, tool_name, parameter_name):
    if policy is None:
        return None
    root = policy.argument_schemas.get(tool_name)
    if not isinstance(root, dict):
        return None
    return root.get("properties", {}).get(
        parameter_name, root.get("additionalProperties", {})
    )


def _typed_tool_value(value, string_schema):
    parsed = json_value(value)
    if string_schema is None:
        return parsed
    return value if string_schema[0] == "raw" or value in string_schema[1] else parsed


@dataclass(slots=True)
class Block:
    """One block of output: the reasoning, a run of text or a tool call."""

    kind: str  # "reasoning", "text" or "tool"
    # The text streamed into the block, or a call's argument fragments.
    parts: list[str]
    call_id: str | None = None
    name: str | None = None
    # "completed" or "incomplete" once the block closes.
    status: str = "in_progress"

    @property
    def text(self):
        return "".join(self.parts)


class BlockSequencer:
    """Output in order as blocks, one open at a time: the reasoning, each run
    of text, each tool call. A new kind or a tool header closes the open
    block. Messages and Responses render the same sequence, streamed or not;
    each callback receives a block with its position."""

    def __init__(self, on_open=None, on_delta=None, on_close=None):
        self.on_open = on_open
        self.on_delta = on_delta
        self.on_close = on_close
        self.blocks = []
        self.open = None

    def _start(self, block):
        self._close("completed")
        self.blocks.append(block)
        self.open = block
        if self.on_open is not None:
            self.on_open(len(self.blocks) - 1, block)

    def _append(self, text):
        self.open.parts.append(text)
        if self.on_delta is not None:
            self.on_delta(len(self.blocks) - 1, self.open, text)

    def _close(self, status):
        block, self.open = self.open, None
        if block is None:
            return
        block.status = status
        if self.on_close is not None:
            self.on_close(len(self.blocks) - 1, block)

    def text(self, field, text):
        """Collected text, in field "reasoning_content" or "content"."""
        kind = "reasoning" if field == "reasoning_content" else "text"
        if self.open is None or self.open.kind != kind:
            self._start(Block(kind, []))
        self._append(text)

    def tool(self, delta):
        """A projected tool delta: a call's header or its arguments."""
        function = delta["function"]
        if "name" in function:
            self._start(Block("tool", [], call_id=delta["id"], name=function["name"]))
        if arguments := function.get("arguments"):
            self._append(arguments)

    def finish(self, incomplete, reasoning_open):
        """Close the open block, cut if the output was cut inside it, and end
        with an empty text block when there is neither text nor a call."""
        if self.open is not None:
            cut = incomplete and (self.open.kind != "reasoning" or reasoning_open)
            self._close("incomplete" if cut else "completed")
        if all(block.kind == "reasoning" for block in self.blocks):
            self._start(Block("text", []))
            self._close("incomplete" if incomplete else "completed")
        return self.blocks


def _validate(validator, value):
    try:
        validator.validate(value)
    except AttributeError as error:
        # referencing's draft 3 crawls the keys of an extends object as schemas
        # whenever a reference lookup scans the document for identifiers.
        raise SchemaEvaluationError(
            "schema reference could not be evaluated"
        ) from error


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
            _validate(validator, arguments)
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
        _validate(validator, value)
    except SchemaEvaluationError as error:
        raise APIError(500, str(error), "output_validation_failed") from error
    except (ValueError, ValidationError, Unresolvable, RecursionError) as error:
        raise APIError(
            500,
            f"model returned invalid structured output: {error}",
            "invalid_model_output",
        ) from error

"""Incremental model-output parsing, output blocks and final answer
validation."""

import re
from dataclasses import dataclass

from jsonschema.exceptions import ValidationError
from referencing.exceptions import Unresolvable

from . import json_codec
from .errors import APIError
from .schema_validation import SchemaEvaluationError
from .tool_schema import (
    CALL_OPEN,
    FUNCTION_END,
    JSON_TYPES,
    MAX_NAME_LENGTH,
    NAME_SPACE,
    PARAMETER_CLOSE,
    PARAMETER_OPEN,
    THINK_END,
    TOOL_CALL_CLOSE,
)

TOOL_ARGUMENT_DELTA_CHARS = 16 * 1024
# The deepest a call's arguments nest as an object the server writes, so that
# they read back within json_codec.MAX_DEPTH where a request carries them
# deepest: as a tool_use block's input in a Messages history, inside the body,
# its messages, a message, the message's content and the block. Chat and
# Responses carry arguments as strings, read on their own.
MAX_ARGUMENTS_DEPTH = json_codec.MAX_DEPTH - 5
_SURROGATE = re.compile("[\ud800-\udfff]")


def hold_partial(text, *markers):
    """`text` split before its longest end that begins one of `markers`, which
    more text may complete."""
    longest = max(len(marker) for marker in markers)
    for length in range(min(len(text), longest - 1), 0, -1):
        if any(
            length < len(marker) and text.endswith(marker[:length])
            for marker in markers
        ):
            return text[:-length], text[-length:]
    return text, ""


class ReasoningSplitter:
    def __init__(self, thinking, tool_calls=False):
        self.reasoning = thinking
        self.pending = ""
        # Whether the newlines after </think>, which set the answer apart in
        # the chat template's layout of a turn, are still to be dropped.
        self.separator = False
        # Where a call may follow, a call's opening also ends the reasoning
        # and begins the answer.
        self.ends = (THINK_END, CALL_OPEN) if tool_calls else (THINK_END,)

    def put(self, text):
        if not self.reasoning:
            return self._content(text)
        self.pending += text
        ends = [
            (index, marker)
            for marker in self.ends
            if (index := self.pending.find(marker)) >= 0
        ]
        if ends:
            end, marker = min(ends)
            reasoning = self.pending[:end]
            content = self.pending[end:]
            if marker == THINK_END:
                content = content[len(THINK_END) :]
            self.pending = ""
            self.reasoning = False
            self.separator = marker == THINK_END
            output = [("reasoning_content", reasoning)] if reasoning else []
            return output + self._content(content)
        ready, self.pending = hold_partial(self.pending, *self.ends)
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


# Where a value ends: the close the template writes after it, then the next
# parameter, the function's close, or the call's close a model may write
# without the function's. A value may hold the tags in any other order.
_VALUE_ENDS = tuple(
    PARAMETER_CLOSE + tag for tag in (PARAMETER_OPEN, FUNCTION_END, TOOL_CALL_CLOSE)
)
# Text outside calls: a call's opening opens one, and a </think>, which a
# model that called a tool from its reasoning may still write, is dropped.
_TEXT_TAGS = (CALL_OPEN, THINK_END)
_NOT_JSON = object()
# A template that writes a value through Jinja's string filter, as Nex's and
# some of Qwen's do, spells a boolean or null as Python does.
_PYTHON_LITERALS = {"True": True, "False": False, "None": None}


def _json_value(text):
    """The JSON value `text` spells, if it can be written back as JSON in its
    call's arguments: its numbers finite, its strings Unicode and its
    containers, in the arguments object, nested at most MAX_ARGUMENTS_DEPTH
    deep. Otherwise _NOT_JSON."""
    try:
        value = json_codec.loads(text)
    except ValueError:
        return _NOT_JSON
    # The value's level in the arguments object.
    pending = [(value, 2)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, str):
            if _SURROGATE.search(item):
                return _NOT_JSON
        elif isinstance(item, (dict, list)):
            if depth > MAX_ARGUMENTS_DEPTH:
                return _NOT_JSON
            children = [*item, *item.values()] if isinstance(item, dict) else item
            pending.extend((child, depth + 1) for child in children)
    return value


def convert_value(text, types):
    """A parameter's text as the JSON types its schema declares read it, any
    type where `types` is None: a string as the text, and another value as
    the JSON the text spells when that is of a declared type or when no
    declared type is a string. A boolean or null declared without a string
    may also be spelled as Python spells it. Otherwise the text."""
    value = _json_value(text)
    if value is _NOT_JSON:
        value = _PYTHON_LITERALS.get(text.strip(), _NOT_JSON)
        declared = (
            value is not _NOT_JSON
            and types is not None
            and "string" not in types
            and JSON_TYPES[type(value)] in types
        )
        return value if declared else text
    if types is None or "string" in types:
        kind = JSON_TYPES[type(value)]
        declared = (
            types is None or kind in types or (kind == "integer" and "number" in types)
        )
        return value if declared and kind != "string" else text
    return value


class StreamingToolCallProjector:
    """Parse Qwen's XML tool calls out of the output as it arrives into
    OpenAI JSON argument deltas, for streamed and complete responses alike.

    Calls read as the chat template lays them out. One opens at CALL_OPEN
    and names its function up to ">". Each parameter is a <parameter=NAME>
    tag and a value, without the newline the template sets after the tag,
    that ends at PARAMETER_CLOSE where the next parameter, the function's
    close or the call's close follows, so a value may hold the tags in any
    other order. </function> closes the call, and the </tool_call> after it
    may be left out. Names lose the space around them; a call that names no
    function is dropped, other text inside a call is dropped, and a repeated
    parameter keeps its first value, which may have streamed. A value
    converts by the types its tool declares for it (convert_value), and one
    that may only be a string streams as it is written. Text outside calls
    streams as it arrives, after a call as before one, and a </think> there
    is dropped.
    """

    def __init__(self, policy, request_id, structured=False):
        self.policy = policy
        self.request_id = request_id
        self.pending = ""
        self.state = "output" if structured else "content"
        self.call_index = 0
        # The open call, None while the text names none or names no function.
        self.call_id = None
        self.function_name = None
        # The types by which its declared parameters and any others convert.
        self.parameter_types = {}
        self.other_types = None
        self.parameter_names = set()
        self.argument_fragments = []
        self.parameter_count = 0
        # The open parameter, None while its value is dropped.
        self.parameter_name = None
        self.value_types = None
        self.value_streams = False
        self.value_started = False
        self.value_parts = []
        self.content_fragments = []
        # How many content fragments the stream has published (the rest are
        # whitespace it holds), and whether the text since the start of the
        # output or the last call has shown a visible character yet.
        self.streamed_count = 0
        self.text_visible = False
        self.closed_calls = []

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

    def _emit_argument(self, fragment, events):
        if self.call_id is None:
            return
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

    def _begin_call(self, name, events):
        self.state = "arguments"
        self.argument_fragments = []
        self.parameter_count = 0
        self.parameter_names = set()
        name = name.strip(NAME_SPACE)
        if not name:
            return
        if not self.streamed_count:
            # Whitespace before the first text only framed the calls.
            self.content_fragments.clear()
        self.function_name = name
        self.call_id = f"call_{self.request_id}_{self.call_index}"
        self.parameter_types, self.other_types = self.policy.parameter_types(name)
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
        self._emit_argument("{", events)

    def _key(self, name):
        separator = "," if self.parameter_count else ""
        self.parameter_count += 1
        return f"{separator}{json_codec.dumps(name)}:"

    def _begin_parameter(self, name, events):
        name = name.strip(NAME_SPACE)
        # A repeated parameter keeps its first value, which may have streamed;
        # a parameter without a name, or of a call that names no function,
        # keeps none.
        kept = name and name not in self.parameter_names and self.call_id is not None
        self.parameter_names.add(name)
        self.parameter_name = name if kept else None
        self.value_types = self.parameter_types.get(name, self.other_types)
        # A value that may only be a string streams as it is written; any
        # other waits for its end to convert.
        self.value_streams = self.value_types == {"string"}
        self.value_started = False
        self.value_parts = []
        if self.value_streams and self.parameter_name is not None:
            self._emit_argument(self._key(name) + '"', events)
        self.state = "value"

    def _put_value(self, text, events):
        if self.parameter_name is None:
            return
        if not self.value_started:
            if not text:
                return
            self.value_started = True
            text = text.removeprefix("\n")
        if not text:
            return
        if self.value_streams:
            self._emit_argument(json_codec.dumps(text)[1:-1], events)
        else:
            self.value_parts.append(text)

    def _end_parameter(self, text, events):
        if self.parameter_name is not None:
            if not self.value_started:
                self.value_started = True
                text = text.removeprefix("\n")
            self._put_value(text, events)
            if self.value_streams:
                self._emit_argument('"', events)
            else:
                value = convert_value("".join(self.value_parts), self.value_types)
                self._emit_argument(
                    self._key(self.parameter_name) + json_codec.dumps(value), events
                )
        self.parameter_name = None
        self.value_parts = []
        self.state = "arguments"

    def _finish_call(self, events):
        self._emit_argument("}", events)
        if self.call_id is not None:
            self.closed_calls.append(
                {
                    "id": self.call_id,
                    "type": "function",
                    "function": {
                        "name": self.function_name,
                        "arguments": "".join(self.argument_fragments),
                    },
                }
            )
            self.call_index += 1
        self.call_id = None
        self.function_name = None
        self.argument_fragments = []
        self.text_visible = False
        self.state = "closing"

    def _next_tag(self, *tags):
        """The tag of `tags` that the pending text spells first, and where."""
        found = [(index, tag) for tag in tags if (index := self.pending.find(tag)) >= 0]
        return min(found) if found else (-1, None)

    def _name(self):
        """The name the pending text spells up to ">", taken off the text; ""
        for a longer one, and None while it may yet end."""
        end = self.pending.find(">", 0, MAX_NAME_LENGTH + 1)
        if end < 0:
            return None if len(self.pending) <= MAX_NAME_LENGTH else ""
        name, self.pending = self.pending[:end], self.pending[end + 1 :]
        return name

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
                start, tag = self._next_tag(*_TEXT_TAGS)
                if tag is None:
                    ready, self.pending = hold_partial(self.pending, *_TEXT_TAGS)
                    self._emit_content(ready, events)
                    break
                self._emit_content(self.pending[:start], events)
                self.pending = self.pending[start + len(tag) :]
                if tag == CALL_OPEN:
                    self.state = "name"
                continue
            if self.state == "name":
                if (name := self._name()) is None:
                    break
                self._begin_call(name, events)
                continue
            if self.state == "arguments":
                # Text between parameters is dropped, but for a tag it may
                # begin.
                tags = (PARAMETER_OPEN, FUNCTION_END, TOOL_CALL_CLOSE)
                start, tag = self._next_tag(*tags)
                if tag is None:
                    self.pending = hold_partial(self.pending, *tags)[1]
                    break
                self.pending = self.pending[start + len(tag) :]
                if tag == PARAMETER_OPEN:
                    self.state = "parameter"
                else:
                    self._finish_call(events)
                    if tag == TOOL_CALL_CLOSE:
                        self.state = "content"
                continue
            if self.state == "parameter":
                if (name := self._name()) is None:
                    break
                self._begin_parameter(name, events)
                continue
            if self.state == "value":
                start, tag = self._next_tag(*_VALUE_ENDS)
                if tag is None:
                    held = hold_partial(self.pending, *_VALUE_ENDS)[1]
                    self._put_value(
                        self.pending[: len(self.pending) - len(held)], events
                    )
                    self.pending = held
                    break
                self._end_parameter(self.pending[:start], events)
                # The tag after the close stays for the arguments to read.
                self.pending = self.pending[start + len(PARAMETER_CLOSE) :]
                continue
            if self.state == "closing":
                # After </function>, the template closes the call's block.
                rest = self.pending.lstrip(NAME_SPACE)
                if rest.startswith(TOOL_CALL_CLOSE):
                    self.pending = rest[len(TOOL_CALL_CLOSE) :]
                elif TOOL_CALL_CLOSE.startswith(rest):
                    break
                self.state = "content"
        return events

    def finish(self, incomplete):
        """The content and calls of the output, and the events the stream
        still owes for what put() held back.

        Output cut at the token limit keeps an open call with the arguments
        it has, and no call whose name it cut; other output that ends inside
        a call closes the call, and what it held back after a value could
        only begin the value's close. Cut output with a call has the content
        the stream published, which leaves out whitespace that no visible
        text has followed since the start or the last call. Otherwise the
        content is the text outside calls without whitespace that only frames
        them and, when cut, a trailing partial tag."""
        events = []
        if not incomplete:
            if self.state == "name":
                name = self.pending if len(self.pending) <= MAX_NAME_LENGTH else ""
                self.pending = ""
                self._begin_call(name, events)
            if self.state == "value":
                self.pending = ""
                self._end_parameter("", events)
            if self.state in ("arguments", "parameter"):
                self.pending = ""
                self._finish_call(events)
        if self.state == "closing":
            self.pending = ""
            self.state = "content"
        if (
            self.state in ("content", "output", "json")
            and self.pending
            and not (
                incomplete and any(tag.startswith(self.pending) for tag in _TEXT_TAGS)
            )
        ):
            self.content_fragments.append(self.pending)
        elif self.closed_calls:
            # Whitespace held after the last text only framed the calls.
            del self.content_fragments[self.streamed_count :]
        self.pending = ""
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
        if unsent := content[len(streamed) :]:
            events.append(("content", unsent))
        return content, calls, events


def argument_deltas(arguments):
    # Keep individual SSE frames bounded even when a tool has a large string
    # argument. Callers preserve fragment order.
    for offset in range(0, len(arguments), TOOL_ARGUMENT_DELTA_CHARS):
        yield arguments[offset : offset + TOOL_ARGUMENT_DELTA_CHARS]


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

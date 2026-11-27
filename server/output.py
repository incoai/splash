"""Incremental model-output parsing, output blocks and final answer
validation."""

import re
import threading
from dataclasses import dataclass

from jsonschema.exceptions import ValidationError
from referencing.exceptions import Unresolvable

from . import json_codec
from .chat_format import THINK_END, THINK_START
from .errors import APIError
from .schema_validation import SchemaEvaluationError
from .tool_schema import (
    JSON_TYPES,
    MAX_NAME_LENGTH,
    NAME_SPACE,
    ParameterTypes,
)

TOOL_ARGUMENT_DELTA_CHARS = 16 * 1024
# The deepest a call's arguments nest as an object the server writes, so that
# they read back within json_codec.MAX_DEPTH where a request carries them
# deepest: as a tool_use block's input in a Messages history, inside the body,
# its messages, a message, the message's content and the block. Chat and
# Responses carry arguments as strings, read on their own.
MAX_ARGUMENTS_DEPTH = json_codec.MAX_DEPTH - 5
_SURROGATE = re.compile("[\ud800-\udfff]")
# The longest stop sequence a request may carry: StopSearch keeps a table entry
# of about 40 bytes and a match step for every character of each stop, so
# four stops this long hold about 3 MB.
MAX_STOP_CHARACTERS = 16 * 1024


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


class StopSearch:
    """The first of a request's stop sequences that the output contains, as
    the output arrives, and how much of its end may begin one. Each stop
    keeps the length of the output's longest end that begins it (a
    Knuth-Morris-Pratt state), so a character costs the same however long
    the stops are: rescanning every length of a 50,000-character stop on
    every token would copy about a gigabyte per token."""

    def __init__(self, stops):
        self.stops = tuple(stops)
        self._fallbacks = [_prefix_function(stop) for stop in self.stops]
        self._states = [0] * len(self.stops)

    def search(self, text):
        """The stop that the output, `text` appended, contains first, and how
        many characters before the end of `text` it begins (more than
        `text` holds where it began earlier); None where it contains none."""
        first = None
        for index, (stop, fallback) in enumerate(zip(self.stops, self._fallbacks)):
            state = self._states[index]
            for position, character in enumerate(text):
                while state and stop[state] != character:
                    state = fallback[state - 1]
                if stop[state] == character:
                    state += 1
                if state == len(stop):
                    back = len(text) - position - 1 + len(stop)
                    if first is None or back > first[1]:
                        first = (stop, back)
                    break
            self._states[index] = state
        return first

    @property
    def held(self):
        """The length of the output's longest end that begins a stop."""
        return max(self._states)


def _prefix_function(text):
    """For each prefix of `text`, the length of its longest proper end that
    is also a prefix of `text`."""
    lengths = [0] * len(text)
    length = 0
    for index in range(1, len(text)):
        while length and text[index] != text[length]:
            length = lengths[length - 1]
        if text[index] == text[length]:
            length += 1
        lengths[index] = length
    return lengths


class ReasoningSplitter:
    """The output's reasoning and its answer, as the output arrives, in the
    blocks of its chat format. The reasoning starts the output where the
    generation prompt opened it (thinking), and where the model opens its
    reasoning itself (chat_format.think_start_ids), it follows the format's
    think_open that starts the output, which is dropped; where the format's
    reasoning returns after text (reasoning_returns), each think_open in the
    answer starts it again. It ends at the format's think_close, which is
    dropped too, and where a call may follow (tool_calls) and a call's
    opening ends it (chat_format.calls_end_reasoning), at a call's opening,
    which begins the answer. The tags that frame the answer's text
    (text_frames, text_close) are dropped from it, and where they are, a
    call, from its call_tag through its call_close, is the answer's as it is
    written, for the projector to read. A text completion, which has no chat
    format, reasons only where thinking says so, and calls no tool."""

    def __init__(self, thinking, tool_calls=False, chat_format=None):
        self.reasoning = thinking
        self.think_open = THINK_START if chat_format is None else chat_format.think_open
        self.think_close = THINK_END if chat_format is None else chat_format.think_close
        # Whether the output may yet open its reasoning: until it shows
        # whether it starts with think_open. A format whose reasoning returns
        # opens it wherever its answer holds think_open (content_tags).
        self.opening = (
            not thinking
            and chat_format is not None
            and bool(chat_format.think_start_ids)
            and not chat_format.reasoning_returns
        )
        self.pending = ""
        # Whether the newlines after the reasoning's close, which set the
        # answer apart in the chat template's layout of a turn, are still to
        # be dropped.
        self.separator = False
        self.ends = (
            (self.think_close, chat_format.call_syntax.call_open)
            if tool_calls and chat_format.calls_end_reasoning
            else (self.think_close,)
        )
        # The tags the answer holds that are not its text: those that frame
        # its text, think_open where reasoning returns after text, and a
        # call's call_tag where the answer's text is framed, whose call stays
        # whole through its call_close (call_close), which may be a frame's
        # close too.
        self.content_tags = ()
        self.call_tag = self.call_close = None
        self.in_call = False
        if chat_format is not None:
            self.content_tags = (
                chat_format.text_frames
                + tuple(tag for tag in (chat_format.text_close,) if tag)
                + ((self.think_open,) if chat_format.reasoning_returns else ())
            )
            if chat_format.text_frames:
                syntax = chat_format.call_syntax
                self.call_tag, self.call_close = syntax.call_tag, syntax.call_close
                self.content_tags += (self.call_tag,)

    def put(self, text):
        if self.opening:
            self.pending += text
            if len(self.pending) < len(self.think_open) and self.think_open.startswith(
                self.pending
            ):
                return []
            self.opening = False
            text, self.pending = self.pending, ""
            if text.startswith(self.think_open):
                self.reasoning = True
                text = text[len(self.think_open) :]
        if not self.reasoning:
            return self._content(text)
        return self._reasoning(text)

    def _reasoning(self, text):
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
            if marker == self.think_close:
                content = content[len(self.think_close) :]
            self.pending = ""
            self.reasoning = False
            self.separator = marker == self.think_close
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
        if not self.content_tags:
            return [("content", text)]
        # The answer without its tags: at each position the longest tag the
        # text holds whole, and an end that may begin one held for the text
        # after it.
        self.pending += text
        output = []
        while True:
            if self.in_call:
                end = self.pending.find(self.call_close)
                if end < 0:
                    ready, self.pending = hold_partial(self.pending, self.call_close)
                    return output + ([("content", ready)] if ready else [])
                end += len(self.call_close)
                output.append(("content", self.pending[:end]))
                self.pending = self.pending[end:]
                self.in_call = False
                continue
            found = [
                (index, -len(tag), tag)
                for tag in self.content_tags
                if (index := self.pending.find(tag)) >= 0
            ]
            if not found:
                ready, self.pending = hold_partial(self.pending, *self.content_tags)
                return output + ([("content", ready)] if ready else [])
            index, _, tag = min(found)
            if index:
                output.append(("content", self.pending[:index]))
            self.pending = self.pending[index:]
            # A tag that a longer one begins, at the text's end, waits for
            # the text that tells them apart.
            if any(
                len(longer) > len(tag) and longer.startswith(self.pending)
                for longer in self.content_tags
            ):
                return output
            self.pending = self.pending[len(tag) :]
            if tag == self.think_open:
                self.reasoning = True
                text, self.pending = self.pending, ""
                return output + self._reasoning(text)
            if tag == self.call_tag:
                # A call, its tag and its text, as it is written.
                output.append(("content", tag))
                self.in_call = True

    def finish(self):
        if not self.pending:
            return []
        kind = "reasoning_content" if self.reasoning else "content"
        text, self.pending = self.pending, ""
        return [(kind, text)]


_NOT_JSON = object()
# A template that writes a value through Jinja's string filter, as Nex's and
# some of Qwen's do, spells a boolean or null as Python does.
_PYTHON_LITERALS = {"True": True, "False": False, "None": None}


def _json_value(text, depth=2):
    """The JSON value `text` spells, if it can be written back as JSON in its
    call's arguments: its numbers finite, its strings Unicode and its
    containers, in the arguments object, nested at most MAX_ARGUMENTS_DEPTH
    deep. Otherwise _NOT_JSON. `depth` is the value's level: 2 for a value
    in the arguments object, 1 for the object."""
    try:
        value = json_codec.loads(text)
    except ValueError:
        return _NOT_JSON
    pending = [(value, depth)]
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


def _arguments_object(text):
    """Whether `text`, a closed call's arguments, is the JSON object every
    dialect returns a call's arguments as, within the bounds _json_value
    reads a value in it by."""
    return isinstance(_json_value(text, depth=1), dict)


def convert_value(text, types):
    """A parameter's text as the JSON types its schema gives for it read it,
    `types` holding at least one; a value of none is its text, which
    StreamingToolCallProjector streams as it is written. A string reads as
    the text, and another value as the JSON the text spells when that is of
    a declared type or when no declared type is a string. A boolean or null
    declared without a string may also be spelled as Python spells it.
    Otherwise the text."""
    value = _json_value(text)
    if value is _NOT_JSON:
        value = _PYTHON_LITERALS.get(text.strip(), _NOT_JSON)
        declared = (
            value is not _NOT_JSON
            and "string" not in types
            and JSON_TYPES[type(value)] in types
        )
        return value if declared else text
    if "string" in types:
        kind = JSON_TYPES[type(value)]
        declared = kind in types or (kind == "integer" and "number" in types)
        return value if declared and kind != "string" else text
    return value


class StreamingToolCallProjector:
    """Parse the tool calls a model writes in its chat template's layout
    (syntax, a ToolCallSyntax) out of the output as it arrives into OpenAI
    JSON argument deltas, for streamed and complete responses alike.

    A call opens at the syntax's call_open and names its function up to its
    name_end. Each parameter is key_open, a name up to key_close, and a
    value, without the template's value_open, that ends at value_close where
    the next parameter's opening, the arguments' close or the call's close
    follows (value_ends), so a value may hold the tags in any other order.
    The arguments' close or the call's close ends the call; the call's close
    after an arguments' close (Qwen's </function>) may be left out. A call
    whose arguments are one JSON object (the "json" kind) streams them as
    written after the object's name and ends at the call's close; one whose
    header names no function, its name's close straight after the call's
    opening, is held until it closes and named by its object, as a header
    names one. Names lose the space around them; any other opening that
    names no function, or a name too long to be one, reads as text; other
    text inside a call is dropped,
    and a repeated parameter keeps its first value. A value typed as string
    alone, or untyped, streams as written; any other converts at its end
    (convert_value). Text outside calls streams as it arrives, and the
    reasoning's close there (think_close) is dropped: a model may write one
    again after its reasoning has ended, or with thinking off.

    Without a tool policy the output has no calls: a call's opening is text,
    and the projector only drops the reasoning's closes, so that every
    answer reads its text the same way. A structured answer's JSON is left as
    it is.
    """

    def __init__(
        self,
        policy,
        request_id,
        structured=False,
        syntax=None,
        think_close=THINK_END,
        streaming=False,
    ):
        # The types of the tools its calls name, read as they name them.
        self.tool_types = ParameterTypes(policy) if policy is not None else None
        # The tools a call may name: those the request declares, whose names
        # its calls are returned under (ToolPolicy.declared_name).
        self.policy = policy
        self.declared = policy.names if policy is not None else frozenset()
        # The output from the open call on (put), and where in it the call's
        # opening begins: a call that is not returned is the text the model
        # wrote.
        self.text = ""
        self.call_start = None
        # A closed call returned as text, until the text after its close
        # shows where the call ends; and why each call that was not returned
        # was not (REJECTED_CALL_REASONS).
        self.rejected_call = None
        self.rejections = []
        # Whether the output streams: a call of a declared tool then streams
        # once its arguments begin as an object (_start_call), as OpenAI
        # and Anthropic stream them; otherwise a call is held
        # until it closes. The open call streams (call_streams), whether its
        # JSON arguments' start has been read (arguments_read), and the id of
        # a streamed call the output cut, which ends incomplete.
        self.streaming = streaming
        self.call_streams = False
        self.arguments_read = False
        self.cut_call = None
        self.request_id = request_id
        # The layout its calls are read in, which a tool policy needs.
        self.syntax = syntax
        # The close of the chat format's reasoning, dropped where text holds it.
        self.think_close = think_close
        self.text_tags = (
            (syntax.call_open, think_close) if policy is not None else (think_close,)
        )
        self.pending = ""
        self.state = "output" if structured else "content"
        self.call_index = 0
        # The open call, None while the text names none: the declared tool it
        # names, else its name as written, and its name as written.
        self.call_id = None
        self.function_name = None
        self.written_name = None
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
        # output or the last call has shown a visible character yet. Without
        # calls, no whitespace frames them.
        self.streamed_count = 0
        self.text_visible = policy is None
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
        # A held call streams once it has closed and been read as one
        # (_finish_call): one the output cuts, or that is not returned, has
        # sent nothing. A started one streams each fragment.
        self.argument_fragments.append(fragment)
        if self.call_streams:
            events.extend(
                ("tool", {"index": self.call_index, "function": {"arguments": chunk}})
                for chunk in argument_deltas(fragment)
            )

    def _start_call(self, events):
        """Stream the open call from here on: its header, then the arguments
        read so far and each fragment after them."""
        if not self.streamed_count:
            # Whitespace before the first text only framed the calls.
            self.content_fragments.clear()
        events.append(self._header())
        events.extend(
            ("tool", {"index": self.call_index, "function": {"arguments": chunk}})
            for chunk in argument_deltas("".join(self.argument_fragments))
            if chunk
        )
        self.call_streams = True

    def _header(self):
        return (
            "tool",
            {
                "index": self.call_index,
                "id": self.call_id,
                "type": "function",
                "function": {"name": self.function_name},
            },
        )

    def _consumed(self):
        """Where in the output the text put() has read ends."""
        return len(self.text) - len(self.pending)

    def _not_a_call(self, name, events):
        """An opening that names no function, and the text after it, as
        content: nothing of a call has streamed yet."""
        self.state = "content"
        self.call_start = None
        self.call_id = self.function_name = self.written_name = None
        self._emit_content(self.syntax.call_open + name, events)

    def _begin_call(self, name, events):
        # A call of the "json" kind whose header names no function may name
        # it in its object, once the header closes (body_open, _end_body).
        if not name.strip(NAME_SPACE) and self.syntax.arguments != "json":
            self._not_a_call(name, events)
            return
        self.state = "arguments"
        self.argument_fragments = []
        self.parameter_count = 0
        self.parameter_names = set()
        self.written_name = name.strip(NAME_SPACE)
        name = self.policy.declared_name(self.written_name) or self.written_name
        self.function_name = name
        self.call_id = f"call_{self.request_id}_{self.call_index}"
        self.parameter_types, self.other_types = self.tool_types.of(name)
        self.arguments_read = False
        if self.streaming and name in self.declared and self.syntax.arguments != "json":
            # Arguments the projector writes are an object from their start.
            self._start_call(events)
        if self.syntax.arguments == "json":
            # The name's close, then the call's JSON object.
            self.state = "body_open"
        else:
            self._emit_argument("{", events)

    def _key(self, name):
        separator = "," if self.parameter_count else ""
        self.parameter_count += 1
        return f"{separator}{json_codec.dumps(name)}:"

    def _begin_parameter(self, name, events):
        name = name.strip(NAME_SPACE)
        # A repeated parameter keeps its first value, which may have streamed;
        # a parameter without a name keeps none.
        kept = name and name not in self.parameter_names
        self.parameter_names.add(name)
        self.parameter_name = name if kept else None
        self.value_types = self.parameter_types.get(name, self.other_types)
        # A value whose schema gives it no type, or the type string alone, is
        # its text and streams as it is written; any other waits for its end
        # to convert.
        self.value_streams = not self.value_types or self.value_types == {"string"}
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
            text = text.removeprefix(self.syntax.value_open)
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
                text = text.removeprefix(self.syntax.value_open)
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

    def _end_body(self, text, events):
        """End a call of the "json" kind at its close, `text` the rest of its
        object. Its arguments streamed as written, after the object's name
        (body_arguments), end at the object's own close; an object laid out
        otherwise is read whole: its arguments, under the syntax's
        json_arguments_key, or its text as written where it holds none, as a
        cut call keeps its text."""
        if self.state == "body_arguments":
            text = text.rstrip()
            self._emit_argument(text.removesuffix("}"), events)
        else:
            try:
                value = json_codec.loads(text)
            except ValueError:
                value = None
            key = self.syntax.json_arguments_key
            arguments = value.get(key) if isinstance(value, dict) else None
            if isinstance(arguments, dict):
                text = json_codec.dumps(arguments)
            # A call whose header names no function is the call of the
            # declared tool its object names, as a header's name would be
            # (ToolPolicy.declared_name); one that names none is its text.
            named = (
                value.get(self.syntax.json_name_key)
                if isinstance(value, dict)
                else None
            )
            if not self.written_name and isinstance(named, str):
                self.function_name = self.policy.declared_name(named) or ""
            self._emit_argument(text.strip() or "{}", events)
        self._finish_call(events)

    def _finish_call(self, events):
        """Close the open call. One that has streamed (_start_call) ends as
        sent; arguments that turned out not to be JSON are counted. A held
        one is returned, and streamed whole, where it names a declared tool
        (ToolPolicy.declared_name) and its arguments read as a JSON object;
        otherwise it is the text the model wrote, once the text after it
        shows where the call ends (_settle_rejected)."""
        if self.syntax.arguments != "json":
            self._emit_argument("}", events)
        arguments = "".join(self.argument_fragments)
        if self.function_name not in self.declared:
            reason = "undeclared"
        elif not _arguments_object(arguments):
            reason = "invalid_arguments"
        else:
            reason = None
        if self.call_streams and reason is not None:
            self.rejections.append(reason)
            reason = None
        elif reason is None and not self.call_streams:
            if not self.streamed_count:
                # Whitespace before the first text only framed the calls.
                self.content_fragments.clear()
            events.append(self._header())
            events.extend(
                ("tool", {"index": self.call_index, "function": {"arguments": chunk}})
                for chunk in argument_deltas(arguments)
            )
        if reason is None:
            self.closed_calls.append(
                {
                    "id": self.call_id,
                    "type": "function",
                    "function": {"name": self.function_name, "arguments": arguments},
                }
            )
            self.call_index += 1
            self.text_visible = False
            self.call_start = None
        else:
            self.rejections.append(reason)
            self.rejected_call = self.call_start
        self.call_id = None
        self.function_name = None
        self.written_name = None
        self.call_streams = False
        self.argument_fragments = []
        # A call of the "json" kind ended at its call_close.
        self.state = "content" if self.syntax.arguments == "json" else "closing"

    def _settle_rejected(self, events):
        """The text of the call _finish_call did not return, as content, once
        put() has read past the call's close."""
        start, self.rejected_call, self.call_start = self.rejected_call, None, None
        self._emit_content(self.text[start : self._consumed()], events)

    def _next_tag(self, *tags):
        """The tag of `tags` that the pending text spells first, and where."""
        found = [(index, tag) for tag in tags if (index := self.pending.find(tag)) >= 0]
        return min(found) if found else (-1, None)

    def _name(self, end, keep=False):
        """The name the pending text spells up to `end`, taken off the text
        with `end` unless `end` is kept; "" for a longer one, and None while
        it may yet end."""
        index = self.pending.find(end, 0, MAX_NAME_LENGTH + len(end))
        if index < 0:
            return None if len(self.pending) < MAX_NAME_LENGTH + len(end) else ""
        name = self.pending[:index]
        self.pending = self.pending[index if keep else index + len(end) :]
        return name

    def put(self, text):
        if self.call_start is None and self.rejected_call is None:
            # Outside a call only the text not yet read is kept.
            self.text = self.pending
        self.pending += text
        self.text += text
        events = []
        syntax = self.syntax
        while self.pending:
            if self.rejected_call is not None and self.state == "content":
                self._settle_rejected(events)
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
                start, tag = self._next_tag(*self.text_tags)
                if tag is None:
                    ready, self.pending = hold_partial(self.pending, *self.text_tags)
                    self._emit_content(ready, events)
                    break
                self._emit_content(self.pending[:start], events)
                if tag != self.think_close:
                    # A call's opening, the other tag.
                    self.call_start = self._consumed() + start
                    self.state = "name"
                self.pending = self.pending[start + len(tag) :]
                continue
            if self.state == "name":
                # The character that ends the name stays: the arguments drop
                # it, or read the tag it begins.
                if (name := self._name(syntax.name_end, keep=True)) is None:
                    break
                self._begin_call(name, events)
                continue
            if self.state == "body_open":
                # The name's close, which the name left in the text. A header
                # that names no function is a call's only where it closes:
                # the call is then held until its object names it.
                close = syntax.name_close
                if len(self.pending) < len(close) and close.startswith(self.pending):
                    break
                if not self.written_name and not self.pending.startswith(close):
                    self._not_a_call("", events)
                    continue
                self.pending = self.pending.removeprefix(close)
                self.state = "body_prefix"
                continue
            if self.state == "body_prefix":
                # The object's name, as the template writes it, before the
                # arguments, which then stream as written; an object laid out
                # otherwise is read whole at the call's close.
                prefix = syntax.json_prefix(self.written_name)
                if self.pending.startswith(prefix):
                    self.pending = self.pending[len(prefix) :]
                    self.state = "body_arguments"
                elif prefix.startswith(self.pending):
                    break
                else:
                    self.state = "body"
                continue
            if (
                self.state == "body_arguments"
                and self.streaming
                and not self.arguments_read
                and self.function_name in self.declared
            ):
                # A streaming call starts once its arguments begin as an
                # object; any other is held, and is text once it closes.
                first = self.pending.lstrip()
                if not first:
                    break
                self.arguments_read = True
                if first.startswith("{"):
                    self._start_call(events)
            if self.state in ("body_arguments", "body"):
                end = self.pending.find(syntax.call_close)
                if end >= 0:
                    self._end_body(self.pending[:end], events)
                    self.pending = self.pending[end + len(syntax.call_close) :]
                    continue
                if self.state == "body_arguments":
                    # The object's own close, the last character before the
                    # call's but for whitespace, waits with that whitespace
                    # for the text after it.
                    ready, held = hold_partial(self.pending, syntax.call_close)
                    close = len(ready.rstrip()) - 1
                    if close >= 0 and ready[close] == "}":
                        ready, held = ready[:close], ready[close:] + held
                    if ready:
                        self._emit_argument(ready, events)
                    self.pending = held
                break
            if self.state == "arguments":
                # Text between parameters is dropped, but for a tag it may
                # begin.
                tags = syntax.argument_tags
                start, tag = self._next_tag(*tags)
                if tag is None:
                    self.pending = hold_partial(self.pending, *tags)[1]
                    break
                self.pending = self.pending[start + len(tag) :]
                if tag == syntax.key_open:
                    self.state = "parameter"
                else:
                    self._finish_call(events)
                    if tag == syntax.call_close:
                        self.state = "content"
                continue
            if self.state == "parameter":
                if (name := self._name(syntax.key_close)) is None:
                    break
                self._begin_parameter(name, events)
                continue
            if self.state == "value":
                # Before the value starts, the start of the value_open the
                # template sets before it waits for the rest.
                if (
                    not self.value_started
                    and len(self.pending) < len(syntax.value_open)
                    and syntax.value_open.startswith(self.pending)
                ):
                    break
                start, tag = self._next_tag(*syntax.value_ends)
                if tag is None:
                    held = hold_partial(self.pending, *syntax.value_ends)[1]
                    self._put_value(
                        self.pending[: len(self.pending) - len(held)], events
                    )
                    self.pending = held
                    break
                self._end_parameter(self.pending[:start], events)
                # The tag after the close stays for the arguments to read.
                self.pending = self.pending[start + len(syntax.value_close) :]
                continue
            if self.state == "closing":
                # After </function>, the template closes the call's block.
                rest = self.pending.lstrip(NAME_SPACE)
                if rest.startswith(syntax.call_close):
                    self.pending = rest[len(syntax.call_close) :]
                elif syntax.call_close.startswith(rest):
                    break
                self.state = "content"
        if self.rejected_call is not None and self.state == "content":
            self._settle_rejected(events)
        return events

    @property
    def in_call(self):
        """Whether the output so far ends inside a call, past its opening."""
        return self.state in (
            "name",
            "arguments",
            "parameter",
            "value",
            "body_open",
            "body_prefix",
            "body_arguments",
            "body",
        )

    def finish(self, incomplete):
        """The content and calls of the output, and the events the stream
        still owes for what put() held back.

        Output cut short, by the token limit or a stop sequence inside a
        call, drops the call it cut where it was held, having streamed
        nothing: a call that did not close is not a call. A call that had
        started streaming ends with the arguments it streamed (cut_call). Other output that ends inside a call
        closes the call, what it held back after a value being the start of
        the value's close. Cut output with a call, made or cut, has the
        content the stream published, which leaves out whitespace that no
        visible text has followed since the start or the last call.
        Otherwise the content is the text outside calls without whitespace
        that only frames them and, when cut, a trailing partial tag."""
        events = []
        cut_call = incomplete and self.in_call
        streamed_cut = None
        if cut_call:
            self.rejections.append("cut")
            if self.call_streams:
                # Nothing is retracted: the call ends incomplete, with the
                # arguments it streamed.
                streamed_cut = {
                    "id": self.call_id,
                    "type": "function",
                    "function": {
                        "name": self.function_name,
                        "arguments": "".join(self.argument_fragments),
                    },
                }
                self.cut_call = self.call_id
            self.pending = ""
            self.state = "content"
            self.call_id = self.function_name = self.written_name = self.call_start = (
                None
            )
            self.call_streams = False
            self.argument_fragments = []
        if not incomplete:
            if self.state == "name":
                # A name too long to be one stays text, as the content reads it.
                name = ""
                if len(self.pending) <= MAX_NAME_LENGTH:
                    name, self.pending = self.pending, ""
                self._begin_call(name, events)
            if self.state == "value":
                self.pending = ""
                self._end_parameter("", events)
            if self.state in ("arguments", "parameter"):
                self.pending = ""
                self._finish_call(events)
            if self.state == "body_open" and not self.written_name:
                self._not_a_call("", events)
            if self.state in ("body_open", "body_prefix", "body_arguments", "body"):
                text, self.pending = self.pending, ""
                if self.state in ("body_open", "body_prefix"):
                    text = ""
                self._end_body(text, events)
        if self.state == "closing":
            self.pending = ""
            self.state = "content"
        if self.rejected_call is not None:
            self._settle_rejected(events)
        if (
            self.state in ("content", "output", "json")
            and self.pending
            and not (
                incomplete
                and any(tag.startswith(self.pending) for tag in self.text_tags)
            )
        ):
            self.content_fragments.append(self.pending)
        elif self.closed_calls or cut_call:
            # Whitespace held after the last text only framed the calls.
            del self.content_fragments[self.streamed_count :]
        self.pending = ""
        calls = list(self.closed_calls)
        if streamed_cut is not None:
            calls.append(streamed_cut)
        streamed = self._streamed_content()
        content = (
            streamed
            if (calls or cut_call) and incomplete
            else "".join(self.content_fragments)
        )
        if unsent := content[len(streamed) :]:
            events.append(("content", unsent))
        return content, calls, events


# Why a call the model wrote was not returned whole: it named a tool the
# request does not declare (returned as text), its arguments are not a JSON
# object (as text where it was held, as sent where it had streamed), or the
# output ended before it closed (dropped where it was held, ended incomplete
# where it had streamed).
REJECTED_CALL_REASONS = ("undeclared", "invalid_arguments", "cut")
_REJECTED_CALL_WORDS = {
    "undeclared": "undeclared tool",
    "invalid_arguments": "arguments not a JSON object",
    "cut": "cut before its close",
}


class RejectedCalls:
    """How many calls the server has not returned, by reason, for /status
    and Prometheus."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = dict.fromkeys(REJECTED_CALL_REASONS, 0)

    def add(self, reasons):
        with self._lock:
            for reason in reasons:
                self._counts[reason] += 1

    def stats(self):
        with self._lock:
            return dict(self._counts)


class KeywordCounts:
    """How many times the server has generated a best-effort answer without
    each schema keyword its grammar does not enforce, for /status and
    Prometheus."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts = {}

    def add(self, keywords):
        with self._lock:
            for keyword in keywords:
                self._counts[keyword] = self._counts.get(keyword, 0) + 1

    def stats(self):
        with self._lock:
            return dict(self._counts)


def rejected_calls_line(reasons):
    """The part of a log line that says which calls were not returned."""
    counts = {reason: reasons.count(reason) for reason in REJECTED_CALL_REASONS}
    return " · ".join(
        f"{count} {_REJECTED_CALL_WORDS[reason]}"
        for reason, count in counts.items()
        if count
    )


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

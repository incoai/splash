"""OpenAI's Responses API: its routes, the responses it stores, how its
requests convert to Chat, and its answers, a response or its server-sent
events."""

import re
import threading
from dataclasses import dataclass
from functools import partial

from .. import json_codec, request_fields
from .. import protocol as wire
from ..chat_messages import image_part, media_content
from ..errors import APIError
from ..lru import LRUCache
from ..metrics import timings_dict
from ..output import BlockSequencer
from ..tool_schema import TAG_NAME_RULE, is_tag_name

RESPONSE_PATH = re.compile(r"/v1/responses/(resp_[A-Za-z0-9_]+)")


def post(exchange, body, deadline):
    streamed, flags = request_fields.generation(body)
    job, conversation = prepare(
        exchange.app,
        body,
        deadline=deadline,
        flags=flags,
        reserve_input=exchange.reserve_input,
    )
    respond = stream if streamed else complete
    return job, partial(respond, exchange, conversation=conversation)


def get(exchange, response_id):
    stored = exchange.app.response_store.get(response_id)
    if stored is None:
        exchange.safe_error(APIError(404, "response not found", "not_found_error"))
    else:
        exchange.send_json(200, stored.response)


def delete(exchange, response_id):
    if not exchange.app.response_store.delete(response_id):
        exchange.safe_error(APIError(404, "response not found", "not_found_error"))
        return
    exchange.send_json(
        200,
        {"id": response_id, "object": "response", "deleted": True},
    )


@dataclass(slots=True)
class Conversation:
    """What a Responses request's answer needs of its conversation beside
    the job: the response it continues, whether the answer is stored (until
    the store refuses it), and the items stored with the answer's output,
    None when it is not stored."""

    previous_id: str | None
    store: bool
    items: list | None


def prepare(app, body, *, deadline, reserve_input, flags=wire.RequestFlag(0)):
    """The generation job of a Responses request, after the input of the
    stored response it continues, and its Conversation. `reserve_input(size)`
    reserves request body budget for that input; flags are the request's own
    (request_fields.generation)."""
    store = body.get("store")
    if store is not None and not isinstance(store, bool):
        raise APIError(400, "store must be a boolean")
    store = True if store is None else store
    previous_id = body.get("previous_response_id")
    if previous_id is not None and (
        not isinstance(previous_id, str) or not previous_id
    ):
        raise APIError(400, "previous_response_id must be a non-empty string")
    with app.preparation(deadline) as without_slot:
        previous_items = []
        if previous_id is not None:
            previous = app.response_store.get(previous_id)
            if previous is None:
                # Clients key on this code to resend the full history.
                raise APIError(
                    404,
                    "previous response not found",
                    "previous_response_not_found",
                )
            # The immutable record remains valid if the store evicts it.
            # Reserve its input bytes before materializing the history.
            reserve_input(len(previous.history_json))
            previous_items = json_codec.loads(previous.history_json)
        items = [*previous_items, *canonical_input(body.get("input"))]
        chat, namespaces = to_chat_body(body, items)
        job = app.chat_job(
            chat,
            deadline=deadline,
            without_slot=without_slot,
            flags=flags,
            output_field="max_output_tokens",
            tool_namespaces=namespaces,
        )
        conversation = Conversation(previous_id, store, items if store else None)
        # The items stay charged to the request body budget until the job is
        # released.
        job.retained_input = conversation.items
        return job, conversation


def persist(app, conversation, response, output):
    if not conversation.store:
        return
    # The store encodes the history at once, so later changes to the
    # request's items or the output cannot reach it.
    history = [*conversation.items, *output]
    if not app.response_store.put(response, history):
        response["store"] = False
        conversation.store = False


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

    BUDGET_BYTES = 64 * 1024 * 1024

    def __init__(self):
        self.records = LRUCache(self.BUDGET_BYTES)
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, response_id):
        with self.lock:
            record = self.records.get(response_id)
            if record is None:
                self.misses += 1
            else:
                self.hits += 1
        return record

    def put(self, response, history_items):
        record = StoredResponse(
            json_codec.encode(response), json_codec.encode(history_items)
        )
        with self.lock:
            return self.records.put(response["id"], record, record.size)

    def delete(self, response_id):
        with self.lock:
            return self.records.pop(response_id) is not None

    def stats(self):
        with self.lock:
            return {
                "entries": len(self.records),
                "bytes": self.records.bytes,
                "budget_bytes": self.records.budget_bytes,
                "evictions": self.records.evictions,
                "hits": self.hits,
                "misses": self.misses,
            }


def _text(value, kinds=("input_text", "output_text", "text")):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise APIError(400, "Responses content must be text")
    if any(
        not isinstance(part, dict)
        or part.get("type") not in kinds
        or not isinstance(part.get("text"), str)
        for part in value
    ):
        raise APIError(400, "only text Responses content is supported")
    return "".join(part["text"] for part in value)


def _content(content):
    """User and tool-result content share the same text/image conversion."""
    if not isinstance(content, list):
        return _text(content)
    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise APIError(400, "Responses content parts must be objects")
        kind = part.get("type")
        if kind == "input_image":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            parts.append(image_part(url))
        elif kind == "input_file":
            parts.append(
                {"type": "file", "file": {k: v for k, v in part.items() if k != "type"}}
            )
        elif kind in ("input_text", "output_text", "text") and isinstance(
            part.get("text"), str
        ):
            parts.append({"type": "text", "text": part["text"]})
        else:
            raise APIError(
                400, "only text, image and inline PDF Responses content is supported"
            )
    return media_content(parts)


def normalize_input(instructions, items):
    instructions = "" if instructions is None else instructions
    if not isinstance(instructions, str):
        raise APIError(400, "instructions must be a string")
    if not isinstance(items, list) or not items:
        raise APIError(400, "input must be a non-empty string or array")
    messages, pending = [], None
    if instructions:
        messages.append({"role": "system", "content": instructions})

    def assistant():
        nonlocal pending
        pending = pending or {"role": "assistant", "content": ""}
        return pending

    def flush():
        nonlocal pending
        if pending is not None:
            messages.append(pending)
            pending = None

    for item in items:
        if not isinstance(item, dict):
            raise APIError(400, "each input item must be an object")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role == "user":
                flush()
                messages.append(
                    {"role": role, "content": _content(item.get("content"))}
                )
                continue
            text = _text(item.get("content"))
            if role in ("system", "developer"):
                flush()
                messages.append({"role": "system", "content": text})
            elif role == "assistant":
                assistant()["content"] += text
            else:
                raise APIError(400, f"unsupported message role: {role}")
        elif kind == "reasoning":
            content = item.get("content")
            text = (
                _text(content, ("reasoning_text", "text"))
                if content is not None
                else ""
            )
            text = text or _text(item.get("summary", []), ("summary_text",))
            if text:
                current = assistant()
                current["reasoning_content"] = (
                    current.get("reasoning_content", "") + text
                )
        elif kind == "function_call":
            call_id, name, arguments = (
                item.get("call_id"),
                item.get("name"),
                item.get("arguments"),
            )
            if not all(isinstance(value, str) and value for value in (call_id, name)):
                raise APIError(400, "function_call requires call_id and name")
            if not isinstance(arguments, str):
                raise APIError(400, "function_call arguments must be JSON text")
            namespace = item.get("namespace")
            if namespace is not None:
                name = _namespace_alias(namespace, name)
            assistant().setdefault("tool_calls", []).append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        elif kind == "custom_tool_call":
            call_id, name, text = (
                item.get("call_id"),
                item.get("name"),
                item.get("input"),
            )
            if not all(isinstance(value, str) and value for value in (call_id, name)):
                raise APIError(400, "custom_tool_call requires call_id and name")
            if not isinstance(text, str):
                raise APIError(400, "custom_tool_call input must be a string")
            # The call as the function the model made it (_custom_function).
            assistant().setdefault("tool_calls", []).append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": {CUSTOM_INPUT: text}},
                }
            )
        elif kind in ("function_call_output", "custom_tool_call_output"):
            call_id, output = item.get("call_id"), item.get("output")
            if not isinstance(call_id, str) or not call_id:
                raise APIError(400, f"{kind} requires call_id")
            output = output.get("content") if isinstance(output, dict) else output
            flush()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _content(output),
                }
            )
        else:
            raise APIError(
                400,
                "only message, reasoning, function_call, function_call_output, "
                "custom_tool_call and custom_tool_call_output input items are "
                "supported",
            )
    flush()
    return messages


def canonical_input(items):
    """A Responses input as its list of items, a string as one user message."""
    if isinstance(items, str):
        return [{"type": "message", "role": "user", "content": items}]
    if not isinstance(items, list):
        raise APIError(400, "input must be a non-empty string or array")
    return items


# A custom (freeform) Responses tool is the function of one string parameter
# the model calls (_custom_function); the request's tool names map it to
# CUSTOM_TOOL, where a namespaced function's alias maps to its namespace.
CUSTOM_TOOL = None
CUSTOM_INPUT = "input"


def _namespace_alias(namespace, name):
    """The name a namespace's function goes by in the prompt and its calls,
    which the tools' own name rule then applies to."""
    if not (
        isinstance(namespace, str) and namespace and isinstance(name, str) and name
    ):
        raise APIError(400, "invalid namespace tool name")
    return f"{namespace}__{name}"


def _response_function(tool, name=None):
    function = {
        key: tool[key]
        for key in ("name", "description", "parameters", "strict")
        if key in tool
    }
    if name is not None:
        function["name"] = name
    return {"type": "function", "function": function}


def _custom_function(tool):
    """A custom tool as the function the model calls: one string, its input,
    which the tool's format, a grammar or plain text, describes. Its calls
    return as custom_tool_call items (output_item)."""
    name = tool.get("name")
    if not is_tag_name(name):
        raise APIError(400, f"custom tool name must be {TAG_NAME_RULE}")
    description = tool.get("description", "")
    if not isinstance(description, str):
        raise APIError(400, f"custom tool {name} description must be a string")
    form = tool.get("format")
    if form is None:
        form = {"type": "text"}
    if not isinstance(form, dict) or form.get("type") not in ("text", "grammar"):
        raise APIError(400, f"custom tool {name} format must be text or a grammar")
    explained = "The tool's input, as raw text."
    if form["type"] == "grammar":
        syntax, definition = form.get("syntax"), form.get("definition")
        if not isinstance(syntax, str) or not isinstance(definition, str):
            raise APIError(
                400, f"custom tool {name} grammar needs syntax and definition"
            )
        explained = (
            "The tool's input, as raw text that this "
            f"{syntax} grammar accepts:\n{definition}"
        )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    CUSTOM_INPUT: {"type": "string", "description": explained}
                },
                "required": [CUSTOM_INPUT],
                "additionalProperties": False,
            },
        },
    }


def to_chat_tools(tools):
    """The Chat function tools of a Responses request's tools, and how the
    request names each one the model sees under another name or shape: a
    namespaced function's alias maps to its namespace and name, and a custom
    tool's name to CUSTOM_TOOL and that name."""
    if not isinstance(tools, list):
        raise APIError(400, "tools must be an array")
    output, namespaces = [], {}
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "custom":
            function = _custom_function(tool)
            name = function["function"]["name"]
            namespaces[name] = (CUSTOM_TOOL, name)
            output.append(function)
            continue
        if isinstance(tool, dict) and tool.get("type") == "namespace":
            namespace, children = tool.get("name"), tool.get("tools")
            if not isinstance(children, list):
                raise APIError(400, "namespace tools must be an array")
            for child in children:
                if not isinstance(child, dict) or child.get("type") != "function":
                    raise APIError(400, "only function namespace tools are supported")
                name = child.get("name")
                alias = _namespace_alias(namespace, name)
                namespaces[alias] = (namespace, name)
                output.append(_response_function(child, alias))
            continue
        if not isinstance(tool, dict) or tool.get("type") != "function":
            kind = tool.get("type") if isinstance(tool, dict) else type(tool).__name__
            raise APIError(
                400, f"only function and custom tools are supported, not {kind!r}"
            )
        output.append(_response_function(tool))
    return output or None, namespaces


def _response_format(text):
    if text is None:
        return None
    if not isinstance(text, dict) or not isinstance(
        text.get("format"), (dict, type(None))
    ):
        raise APIError(400, "text.format must be an object")
    value = text.get("format")
    if value is None or value.get("type") == "text":
        return None
    if value.get("type") == "json_object":
        return {"type": "json_object"}
    if value.get("type") != "json_schema":
        raise APIError(400, "only text and JSON text formats are supported")
    return {
        "type": "json_schema",
        "json_schema": {
            key: value[key]
            for key in ("name", "description", "schema", "strict")
            if key in value
        },
    }


def to_chat_body(body, items):
    """A Responses request as a Chat body, and how the request names each
    tool the model sees under another name or shape (to_chat_tools).
    `items` are the request's canonical input items after those of the
    response it continues."""
    if body.get("conversation") is not None:
        raise APIError(400, "conversation is not supported")
    if body.get("background") not in (None, False):
        raise APIError(400, "background responses are not supported")
    if body.get("truncation") not in (None, "disabled"):
        raise APIError(400, "only disabled truncation is supported")
    chat = {
        "messages": normalize_input(body.get("instructions"), items),
        "parallel_tool_calls": body.get("parallel_tool_calls"),
    }
    namespaces = {}
    if body.get("tools") is not None:
        chat["tools"], namespaces = to_chat_tools(body["tools"])
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") not in ("function", "custom") or not isinstance(
            choice.get("name"), str
        ):
            raise APIError(400, "invalid named tool_choice")
        name = choice["name"]
        namespace = choice.get("namespace")
        if choice["type"] == "custom":
            if namespace is not None or namespaces.get(name) != (CUSTOM_TOOL, name):
                raise APIError(400, "invalid named tool_choice")
        elif namespace is not None:
            name = _namespace_alias(namespace, name)
            if namespaces.get(name) != (namespace, choice["name"]):
                raise APIError(400, "invalid named tool_choice")
        choice = {"type": "function", "function": {"name": name}}
    if choice is not None:
        chat["tool_choice"] = choice
    reasoning = body.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        raise APIError(400, "reasoning must be an object")
    if reasoning and reasoning.get("effort") is not None:
        chat["reasoning_effort"] = reasoning["effort"]
    response_format = _response_format(body.get("text"))
    if response_format is not None:
        chat["response_format"] = response_format
    aliases = {"max_output_tokens": "max_completion_tokens"}
    # Chat validates what it receives, so a field Splash cannot honor, such
    # as logit_bias, is refused rather than dropped.
    for field_name in (
        "model",
        *wire.SAMPLING_FIELDS,
        "logit_bias",
        "seed",
        "priority",
        "cache_salt",
        "stop",
        "max_output_tokens",
        "ignore_eos",
        "chat_template_kwargs",
        "preserve_thinking",
        "thinking_token_budget",
    ):
        if field_name in body and body[field_name] is not None:
            chat[aliases.get(field_name, field_name)] = body[field_name]
    return chat, namespaces


def complete(exchange, job, *, conversation):
    app = exchange.app
    sequencer = BlockSequencer()
    collected = exchange.collect(
        job,
        on_text=sequencer.text,
        on_tool_delta=sequencer.tool,
    )
    result = collected.result
    incomplete = result.reason == "length"
    output = output_items(
        job, sequencer.finish(collected.cut, collected.reasoning_open)
    )
    response = response_object(
        app.response_model,
        job,
        conversation,
        "incomplete" if incomplete else "completed",
        output,
        result=result,
    )
    persist(app, conversation, response, output)
    exchange.send_json(200, response)


def stream(exchange, job, *, conversation):
    app = exchange.app

    def snapshot(status, items, **fields):
        """The response as it stands (response_object)."""
        return response_object(
            app.response_model, job, conversation, status, items, **fields
        )

    output, sequence = [], 0

    def send(event, **payload):
        nonlocal sequence
        exchange.event_sse(
            event, {"type": event, "sequence_number": sequence, **payload}
        )
        sequence += 1

    def begin():
        if exchange.response_started:
            return
        exchange.start_event_stream()
        send("response.created", response=snapshot("in_progress", []))
        send("response.in_progress", response=snapshot("in_progress", []))

    def open_item(index, block):
        item = output_item(job, block, index)
        send("response.output_item.added", output_index=index, item=item)
        if block.kind == "reasoning":
            send(
                "response.reasoning_summary_part.added",
                item_id=item["id"],
                output_index=index,
                summary_index=0,
                part={"type": "summary_text", "text": ""},
            )
        elif block.kind == "text":
            send(
                "response.content_part.added",
                item_id=item["id"],
                output_index=index,
                content_index=0,
                part={"type": "output_text", "text": "", "annotations": []},
            )

    def item_delta(index, block, text):
        if is_custom_call(job, block):
            # A custom tool's input is the string inside the arguments;
            # it streams whole when the call closes.
            return
        if block.kind == "tool":
            event, extra = "response.function_call_arguments.delta", {}
        elif block.kind == "reasoning":
            event, extra = (
                "response.reasoning_summary_text.delta",
                {"summary_index": 0},
            )
        else:
            event, extra = (
                "response.output_text.delta",
                {"content_index": 0, "logprobs": []},
            )
        send(
            event,
            item_id=output_item_id(job, block.kind, index),
            output_index=index,
            delta=text,
            **extra,
        )

    def close_item(index, block):
        item = output_item(job, block, index)
        if is_custom_call(job, block):
            for event, field in (("delta", "delta"), ("done", "input")):
                send(
                    f"response.custom_tool_call_input.{event}",
                    item_id=item["id"],
                    output_index=index,
                    **{field: item["input"]},
                )
        elif block.kind == "tool":
            send(
                "response.function_call_arguments.done",
                item_id=item["id"],
                output_index=index,
                name=item["name"],
                arguments=item["arguments"],
            )
        elif block.kind == "reasoning":
            part = {"type": "summary_text", "text": block.text}
            send(
                "response.reasoning_summary_text.done",
                item_id=item["id"],
                output_index=index,
                summary_index=0,
                text=block.text,
            )
            payload = {
                "item_id": item["id"],
                "output_index": index,
                "summary_index": 0,
                "part": part,
            }
            if block.status == "incomplete":
                payload["status"] = "incomplete"
            send("response.reasoning_summary_part.done", **payload)
        else:
            part = item["content"][0]
            send(
                "response.output_text.done",
                item_id=item["id"],
                output_index=index,
                content_index=0,
                text=part["text"],
                logprobs=[],
            )
            send(
                "response.content_part.done",
                item_id=item["id"],
                output_index=index,
                content_index=0,
                part=part,
            )
        send("response.output_item.done", output_index=index, item=item)
        output.append(item)

    sequencer = BlockSequencer(open_item, item_delta, close_item)

    def keepalive():
        begin()
        if not sequencer.blocks:
            send("response.in_progress", response=snapshot("in_progress", []))
        else:
            # The data event that adds nothing, response.in_progress, carries
            # a snapshot of the response, which would replace the output
            # streamed so far. A comment keeps the stream alive instead,
            # though clients that time out on missing data events ignore it.
            exchange.sse_keepalive()

    def run():
        collected = exchange.collect(
            job,
            on_start=begin,
            on_text=sequencer.text,
            on_tool_delta=sequencer.tool,
            on_idle=keepalive,
            on_progress=lambda progress: send(
                "response.in_progress",
                response=snapshot("in_progress", []),
                prompt_progress=progress,
            ),
        )
        result = collected.result
        incomplete = result.reason == "length"
        sequencer.finish(collected.cut, collected.reasoning_open)
        status = "incomplete" if incomplete else "completed"
        response = snapshot(status, output, result=result)
        persist(app, conversation, response, output)
        send(f"response.{status}", response=response)

    def send_error(error):
        send(
            "response.failed",
            response=snapshot(
                "failed", output, error=exchange.errors.payload(error)["error"]
            ),
        )

    exchange.guarded_stream(job, run, send_error)


def response_object(model, job, conversation, status, output, result=None, error=None):
    usage = None
    if result is not None:
        cache = result.cache
        usage = {
            "input_tokens": result.prompt_tokens,
            "input_tokens_details": {
                "cached_tokens": cache.matched_tokens,
                # Clients price these tokens apart from the rest of the
                # uncached input. The prefix cache keeps a prompt as prefill
                # computes it, at no cost of its own, so uncached input is
                # input_tokens less cached_tokens, as in Messages, which
                # reports no cache_creation_input_tokens.
                "cache_write_tokens": 0,
            },
            "output_tokens": result.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": job.reasoning_tokens},
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        }
    text_format = job.response_format or {"type": "text"}
    if text_format.get("type") == "json_schema":
        text_format = {"type": "json_schema", **text_format["json_schema"]}
    return {
        "id": f"resp_{job.public_id}",
        "object": "response",
        "created_at": job.created_at,
        "status": status,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if status == "incomplete" else None
        ),
        "error": error,
        "model": model,
        "output": output,
        "parallel_tool_calls": (
            job.tool_policy.parallel if job.tool_policy is not None else True
        ),
        "store": conversation.store,
        "previous_response_id": conversation.previous_id,
        "text": {"format": text_format},
        "usage": usage,
        # Once complete, Chat's timings (timings_dict).
        **({} if result is None else {"timings": timings_dict(result)}),
        "end_turn": result is not None
        and result.reason == "stop"
        and not any(
            item["type"] in ("function_call", "custom_tool_call") for item in output
        ),
    }


_ITEM_PREFIXES = {
    "reasoning": "rs",
    "text": "msg",
    "tool": "fc",
    "custom": "ctc",
}


def output_item_id(job, kind, index):
    """The id of the Responses item of a block of `kind` at `index`."""
    return f"{_ITEM_PREFIXES[kind]}_{job.public_id}_{index}"


def is_custom_call(job, block):
    """Whether `block` calls one of the request's custom tools."""
    policy = job.tool_policy
    return (
        block.kind == "tool"
        and policy is not None
        and policy.namespaces.get(block.name) == (CUSTOM_TOOL, block.name)
    )


def custom_input(arguments):
    """A custom tool's input from the arguments of the function the model
    called: their input string, or the arguments as written when they are
    cut short or hold none."""
    try:
        value = json_codec.loads(arguments)
    except ValueError:
        return arguments
    if isinstance(value, dict) and isinstance(value.get(CUSTOM_INPUT), str):
        return value[CUSTOM_INPUT]
    return arguments


def output_item(job, block, index):
    """The Responses output item of `block` at `index`. An item in progress
    carries no text or arguments yet."""
    status = block.status
    text = "" if status == "in_progress" else block.text
    if is_custom_call(job, block):
        return {
            "id": output_item_id(job, "custom", index),
            "type": "custom_tool_call",
            "status": status,
            "call_id": block.call_id,
            "name": block.name,
            "input": custom_input(text),
        }
    item_id = output_item_id(job, block.kind, index)
    if block.kind == "reasoning":
        return {
            "id": item_id,
            "type": "reasoning",
            "status": status,
            "summary": ([{"type": "summary_text", "text": text}] if text else []),
            "content": ([{"type": "reasoning_text", "text": text}] if text else []),
            "encrypted_content": None,
        }
    if block.kind == "text":
        return {
            "id": item_id,
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": (
                []
                if status == "in_progress"
                else [{"type": "output_text", "text": text, "annotations": []}]
            ),
        }
    namespaced = job.tool_policy.namespaces.get(block.name) if job.tool_policy else None
    item = {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": block.call_id,
        "name": namespaced[1] if namespaced else block.name,
        "arguments": text,
    }
    if namespaced:
        item["namespace"] = namespaced[0]
    return item


def output_items(job, blocks):
    return [output_item(job, block, index) for index, block in enumerate(blocks)]

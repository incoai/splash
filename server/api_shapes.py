"""Chat message normalization, which every API's request reaches, and the
conversion of Responses requests and answers."""

from __future__ import annotations

import json

from . import json_codec
from . import protocol as wire
from .documents import checked_file
from .errors import APIError
from .metrics import timings_dict
from .tool_schema import TAG_NAME_RULE, is_tag_name

VISION_UNAVAILABLE = "this model is serving without vision"


def field_value(value, key, default):
    """`value`'s field `key`, or `default` where it is absent or null, as a
    null field is one left out."""
    field = value.get(key)
    return default if field is None else field


# Chat's and Responses' kinds of text part. An assistant's refusal is text it
# wrote, in a field of its own.
TEXT_PARTS = ("text", "input_text", "output_text", "refusal")


def _part_text(part):
    text = field_value(part, "refusal" if part["type"] == "refusal" else "text", "")
    if not isinstance(text, str):
        raise APIError(400, "text message content must be a string")
    return text


def _text_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind not in TEXT_PARTS:
                raise APIError(400, "only text message content is supported")
            parts.append(_part_text(part))
        return "".join(parts)
    raise APIError(400, "message content must be text")


def image_part(url):
    if not isinstance(url, str):
        raise APIError(400, "invalid image content part")
    return {"type": "image_url", "image_url": {"url": url}}


def media_content(parts):
    """Canonical parts, or their text when they carry no image or file."""
    if any(part["type"] != "text" for part in parts):
        return parts
    return "".join(part["text"] for part in parts)


def _require_vision(modality, vision):
    """Reject image or PDF input before it is decoded or rendered."""
    if not vision:
        raise APIError(400, f"{modality} input is not supported: {VISION_UNAVAILABLE}")


def _user_content(content, vision):
    """Text for plain user content; a canonical parts list when it carries
    images or PDFs, so the chat template places each where the author put it.

    The one place media is accepted or rejected: every API shape converts its
    image and PDF parts to image_url and file parts that reach this point
    before anything is decoded or rendered (documents.render_files renders
    the file parts once the whole request is checked)."""
    if not isinstance(content, list):
        return _text_content(content)
    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise APIError(400, "message content parts must be objects")
        kind = part.get("type")
        if kind in ("image_url", "input_image"):
            _require_vision("image", vision)
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            parts.append(image_part(url))
        elif kind in TEXT_PARTS:
            parts.append({"type": "text", "text": _part_text(part)})
        elif kind == "file":
            _require_vision("PDF", vision)
            data = checked_file(part.get("file"))
            parts.append({"type": "file", "file": {"file_data": data}})
        elif kind in ("video", "video_url", "input_audio"):
            raise APIError(400, f"{kind} content is not supported")
        else:
            raise APIError(400, "unsupported message content part")
    return media_content(parts)


def _call_arguments(arguments):
    """A history call's arguments: an object, as sent or as JSON text of one;
    otherwise their text, the string as sent or another value's JSON, which
    template_messages keeps as text, so that a conversation continues past a
    call the output limit cut or the model wrote wrong."""
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            # Some providers send a call without arguments as "".
            parsed = json_codec.loads(arguments.strip() or "{}")
        except ValueError:
            return arguments
        return parsed if isinstance(parsed, dict) else arguments
    return arguments if isinstance(arguments, dict) else json_codec.dumps(arguments)


def normalize_messages(messages, *, vision):
    """`messages` checked and in canonical form, each PDF a file part of its
    checked data, not yet rendered."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "messages must be a non-empty array")
    normalized = []
    has_user = False
    for message in messages:
        if not isinstance(message, dict):
            raise APIError(400, "each message must be an object")
        role = message.get("role")
        if role in ("system", "developer"):
            text = _text_content(message.get("content"))
            if len(normalized) == 1 and normalized[0]["role"] == "system":
                # Responses instructions and developer items, or an Anthropic
                # system and a system message, lead as one system message.
                normalized[0]["content"] = "\n\n".join(
                    part for part in (normalized[0]["content"], text) if part
                )
            else:
                normalized.append({"role": "system", "content": text})
            continue
        if role not in ("user", "assistant", "tool"):
            raise APIError(400, f"unsupported message role: {role}")
        has_user |= role == "user"
        content = message.get("content")
        item = {
            "role": role,
            "content": _user_content(content, vision)
            if role in ("user", "tool")
            else _text_content(content),
        }
        if role == "assistant":
            # Clients send an assistant's reasoning back as Splash's
            # reasoning_content or as reasoning text.
            reasoning = message.get("reasoning_content")
            if reasoning is None and isinstance(message.get("reasoning"), str):
                reasoning = message["reasoning"]
            if reasoning is not None:
                item["reasoning_content"] = _text_content(reasoning)
        if role == "assistant" and message.get("tool_calls") is not None:
            if not isinstance(message["tool_calls"], list):
                raise APIError(400, "assistant tool_calls must be an array")
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict) or not isinstance(
                    function.get("name"), str
                ):
                    raise APIError(400, "invalid assistant tool call")
                calls.append(
                    {
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": _call_arguments(function.get("arguments")),
                        },
                    }
                )
                if call.get("id") is not None:
                    if not isinstance(call["id"], str):
                        raise APIError(400, "tool call id must be a string")
                    calls[-1]["id"] = call["id"]
            item["tool_calls"] = calls
        if role == "tool" and message.get("tool_call_id") is not None:
            tool_call_id = message["tool_call_id"]
            if not isinstance(tool_call_id, str):
                raise APIError(400, "tool_call_id must be a string")
            item["tool_call_id"] = tool_call_id
        normalized.append(item)
    if not has_user:
        raise APIError(400, "messages must include a user message")
    return normalized


def template_messages(messages):
    """Keep a call whose arguments are text, not an object, as text, without
    presenting it as a complete call."""
    result = []
    for message in messages:
        calls = message.get("tool_calls", [])
        if not any(isinstance(call["function"]["arguments"], str) for call in calls):
            result.append(message)
            continue
        current = {key: value for key, value in message.items() if key != "tool_calls"}
        for call in calls:
            if not isinstance(call["function"]["arguments"], str):
                current.setdefault("tool_calls", []).append(call)
                continue
            if any(
                current.get(key)
                for key in ("content", "reasoning_content", "tool_calls")
            ):
                result.append(current)
            result.append(
                {
                    "role": "assistant",
                    "content": "Incomplete tool call:\n"
                    + json.dumps(call["function"], ensure_ascii=False),
                }
            )
            current = {"role": "assistant", "content": ""}
        if any(
            current.get(key) for key in ("content", "reasoning_content", "tool_calls")
        ):
            result.append(current)
    return result


def _responses_text(value, kinds=("input_text", "output_text", "text")):
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


def _responses_content(content):
    """User and tool-result content share the same text/image conversion."""
    if not isinstance(content, list):
        return _responses_text(content)
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


def normalize_responses_input(instructions, items):
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
                    {"role": role, "content": _responses_content(item.get("content"))}
                )
                continue
            text = _responses_text(item.get("content"))
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
                _responses_text(content, ("reasoning_text", "text"))
                if content is not None
                else ""
            )
            text = text or _responses_text(item.get("summary", []), ("summary_text",))
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
                    "content": _responses_content(output),
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


def canonical_responses_input(items):
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
    return as custom_tool_call items (responses_item)."""
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


def normalize_responses_tools(tools):
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


def _responses_format(text):
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


def responses_to_chat_body(body, items):
    """A Responses request as a Chat body, and how the request names each
    tool the model sees under another name or shape (normalize_responses_tools).
    `items` are the request's canonical input items after those of the
    response it continues."""
    if body.get("conversation") is not None:
        raise APIError(400, "conversation is not supported")
    if body.get("background") not in (None, False):
        raise APIError(400, "background responses are not supported")
    if body.get("truncation") not in (None, "disabled"):
        raise APIError(400, "only disabled truncation is supported")
    if body.get("stream") is not None and not isinstance(body["stream"], bool):
        raise APIError(400, "stream must be a boolean")
    chat = {
        "messages": normalize_responses_input(body.get("instructions"), items),
        "parallel_tool_calls": body.get("parallel_tool_calls"),
    }
    namespaces = {}
    if body.get("tools") is not None:
        chat["tools"], namespaces = normalize_responses_tools(body["tools"])
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
    response_format = _responses_format(body.get("text"))
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
        "timeout",
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


def responses_response(model, job, status, output, result=None, error=None):
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
        "store": job.response_store,
        "previous_response_id": job.response_previous_id,
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


_RESPONSES_ITEM_PREFIXES = {
    "reasoning": "rs",
    "text": "msg",
    "tool": "fc",
    "custom": "ctc",
}


def responses_item_id(job, kind, index):
    """The id of the Responses item of a block of `kind` at `index`."""
    return f"{_RESPONSES_ITEM_PREFIXES[kind]}_{job.public_id}_{index}"


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


def responses_item(job, block, index):
    """The Responses output item of `block` at `index`. An item in progress
    carries no text or arguments yet."""
    status = block.status
    text = "" if status == "in_progress" else block.text
    if is_custom_call(job, block):
        return {
            "id": responses_item_id(job, "custom", index),
            "type": "custom_tool_call",
            "status": status,
            "call_id": block.call_id,
            "name": block.name,
            "input": custom_input(text),
        }
    item_id = responses_item_id(job, block.kind, index)
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


def responses_output(job, blocks):
    return [responses_item(job, block, index) for index, block in enumerate(blocks)]

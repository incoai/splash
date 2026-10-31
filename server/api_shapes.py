"""Request and response shape conversion between the chat, Responses and Anthropic dialects."""

from __future__ import annotations

import json

from . import json_codec
from . import protocol as wire
from .documents import DocumentBudget, document_parts, file_content
from .errors import APIError
from .llama_cpp_compat import stamp_id_slot
from .metrics import timings_dict, usage_dict
from .tool_schema import TAG_NAME_RULE, is_tag_name

IMAGE_PAD_TOKEN = "<|image_pad|>"
VISION_UNAVAILABLE = (
    "this model is serving without vision (started with --language-only)"
)


def _field(value, key, default):
    """`value`'s field `key`, or `default` where it is absent or null, as a
    null field is one left out."""
    field = value.get(key)
    return default if field is None else field


# Chat's and Responses' kinds of text part. An assistant's refusal is text it
# wrote, in a field of its own.
TEXT_PARTS = ("text", "input_text", "output_text", "refusal")


def _part_text(part):
    text = _field(part, "refusal" if part["type"] == "refusal" else "text", "")
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


def _image_part(url):
    if not isinstance(url, str):
        raise APIError(400, "invalid image content part")
    return {"type": "image_url", "image_url": {"url": url}}


def _media_content(parts):
    """Canonical parts, or their text when they carry no image or file."""
    if any(part["type"] != "text" for part in parts):
        return parts
    return "".join(part["text"] for part in parts)


def _require_vision(modality, vision):
    """Reject image or PDF input before it is decoded or rendered."""
    if not vision:
        raise APIError(400, f"{modality} input is not supported: {VISION_UNAVAILABLE}")


def _user_content(content, document_budget, vision):
    """Text for plain user content; a canonical parts list when it carries
    images, so the chat template places each image where the author put it.

    The one place media is accepted or rejected: every API shape converts its
    image and PDF parts to image_url and file parts that reach this point
    before anything is decoded or rendered."""
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
            parts.append(_image_part(url))
        elif kind in TEXT_PARTS:
            parts.append({"type": "text", "text": _part_text(part)})
        elif kind == "file":
            _require_vision("PDF", vision)
            parts.extend(file_content(part.get("file"), budget=document_budget))
        elif kind in ("video", "video_url", "input_audio"):
            raise APIError(400, f"{kind} content is not supported")
        else:
            raise APIError(400, "unsupported message content part")
    return _media_content(parts)


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


def normalize_messages(messages, *, vision, deadline, without_slot):
    """`messages` checked and in canonical form, each PDF rendered to its
    pages under `without_slot`, which gives a request in preparation its
    slot back meanwhile."""
    document_budget = DocumentBudget(deadline=deadline, without_slot=without_slot)
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
            "content": _user_content(content, document_budget, vision)
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
            parts.append(_image_part(url))
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
    return _media_content(parts)


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
            if name in namespaces:
                raise APIError(400, f"duplicate tool name: {name}")
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
                if alias in namespaces:
                    raise APIError(400, f"duplicate tool name: {alias}")
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


def responses_to_chat_body(body, items, llama_cpp_compat=False):
    """A Responses request as a Chat body, and how the request names each
    tool the model sees under another name or shape (normalize_responses_tools).
    `items` are the request's canonical input items after those of the
    response it continues. With llama_cpp_compat the body's id_slot - the lane
    a proxy pinned the request to - carries over into the Chat body, where the
    generation reads it (llama_cpp_compat)."""
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
    if llama_cpp_compat:
        chat["id_slot"] = body.get("id_slot")
    return chat, namespaces


def _anthropic_system_text(value, label):
    """System text without billing-header blocks; None when nothing remains."""
    if isinstance(value, str):
        value = [{"type": "text", "text": value}]
    if not isinstance(value, list):
        raise APIError(400, f"{label} must be text content")
    parts = []
    for part in value:
        if (
            not isinstance(part, dict)
            or part.get("type") != "text"
            or not isinstance(part.get("text"), str)
        ):
            raise APIError(400, f"{label} contains unsupported content")
        # Claude Code prepends a billing-header text block; it is not user text.
        if not part["text"].startswith("x-anthropic-billing-header"):
            parts.append(part["text"])
    return "".join(parts) or None


def _anthropic_content(value, label):
    """Text for plain Anthropic content; canonical parts in document order when
    its blocks carry images or documents. User messages and tool results share
    it, so the template places each image where the author put it."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise APIError(400, f"{label} must be text or content blocks")
    parts = []
    for block in value:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            source = block.get("source")
            if (
                not isinstance(source, dict)
                or source.get("type") != "base64"
                or not isinstance(source.get("media_type"), str)
                or not isinstance(source.get("data"), str)
            ):
                raise APIError(400, "Anthropic image blocks must be base64 sources")
            parts.append(
                _image_part(f"data:{source['media_type']};base64,{source['data']}")
            )
        elif kind == "document":
            parts.extend(document_parts(block))
        else:
            raise APIError(400, f"{label} contains unsupported content")
    return _media_content(parts)


def anthropic_to_chat_body(body, *, thinking_resolver, llama_cpp_compat=False):
    """A Messages generation request as a Chat body, and how its response
    shows reasoning. count_tokens converts only the prompt, so it still
    counts a final assistant message. With llama_cpp_compat the body's
    id_slot - the lane a proxy pinned the request to - carries over into the
    Chat body, where the generation reads it (llama_cpp_compat)."""
    max_tokens = body.get("max_tokens")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise APIError(400, "max_tokens must be a positive integer")
    if not isinstance(body.get("stream", False), bool):
        raise APIError(400, "stream must be a boolean")
    chat, thinking_display = _anthropic_chat(body, thinking_resolver)
    # Anthropic continues a final assistant message (a prefill); Splash would
    # close that turn and start another, so it refuses it.
    if chat["messages"] and chat["messages"][-1]["role"] == "assistant":
        raise APIError(
            400,
            "a final assistant message (prefill) is not supported; "
            "end messages with a user turn",
        )
    chat.update(
        max_completion_tokens=max_tokens,
        stop=body.get("stop_sequences"),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        seed=body.get("seed"),
        timeout=body.get("timeout"),
        priority=body.get("priority"),
        cache_salt=body.get("cache_salt"),
    )
    if llama_cpp_compat:
        chat["id_slot"] = body.get("id_slot")
    chat = {key: value for key, value in chat.items() if value is not None}
    return chat, thinking_display


def _anthropic_preserve_thinking(context_management):
    """True when a Messages request keeps every turn's thinking: its
    context_management clears none (clear_thinking_20251015 with keep: all).
    Splash makes no other edit, so the model reads the conversation as
    sent."""
    edits = (
        context_management.get("edits")
        if isinstance(context_management, dict)
        else None
    )
    if isinstance(edits, list) and any(
        isinstance(edit, dict)
        and edit.get("type") == "clear_thinking_20251015"
        and edit.get("keep") == "all"
        for edit in edits
    ):
        return True
    return None


def _anthropic_thinking(thinking, effort):
    """The reasoning effort and the reasoning display of a Messages
    request's thinking, which is off when omitted."""
    if thinking is None:
        return "none", "summarized"
    if not isinstance(thinking, dict) or thinking.get("type") not in (
        "enabled",
        "disabled",
        "adaptive",
    ):
        raise APIError(400, "thinking.type must be enabled, disabled, or adaptive")
    thinking_type = thinking["type"]
    display = thinking.get("display")
    if display is not None:
        if thinking_type == "disabled":
            raise APIError(
                400, "thinking.display requires enabled or adaptive thinking"
            )
        if display not in ("summarized", "omitted", "updates"):
            raise APIError(
                400,
                "thinking.display must be summarized, omitted, updates, or null",
            )
    return (
        "none" if thinking_type == "disabled" else effort,
        # Models expose reasoning and text, not separate progress-update blocks.
        "omitted" if display in ("omitted", "updates") else "summarized",
    )


def _anthropic_thinking_budget(thinking):
    """The budget_tokens of enabled thinking, which caps it: at least 1024,
    as Messages requires. One at or above max_tokens, which Messages allows
    with interleaved thinking, caps nothing. Adaptive thinking sets none."""
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return None
    budget = thinking.get("budget_tokens")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1024:
        raise APIError(
            400, "thinking.budget_tokens must be an integer of at least 1024"
        )
    return budget


def anthropic_to_chat_prompt(body, *, thinking_resolver):
    """The prompt of a Messages request as a Chat body, as count_tokens
    counts it."""
    return _anthropic_chat(body, thinking_resolver)[0]


def _anthropic_chat(body, thinking_resolver):
    """A Messages request's prompt as a Chat body, and how its response
    shows reasoning."""
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise APIError(400, "model must be a non-empty string")
    preserve_thinking = _anthropic_preserve_thinking(body.get("context_management"))
    output_config = body.get("output_config", {})
    if not isinstance(output_config, dict):
        raise APIError(400, "output_config must be an object")
    effort = _field(output_config, "effort", "high")
    if not isinstance(effort, str) or effort not in (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ):
        raise APIError(400, "output_config.effort is invalid")
    response_format = _anthropic_response_format(body, output_config)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "messages must be a non-empty array")
    system = body.get("system")
    system_text = None if system is None else _anthropic_system_text(system, "system")
    translated = [
        translation
        for message in messages
        for translation in _anthropic_message(message, thinking_resolver)
    ]
    if system_text:
        translated.insert(0, {"role": "system", "content": system_text})
    reasoning_effort, thinking_display = _anthropic_thinking(
        body.get("thinking"), effort
    )
    chat = {
        "model": body["model"],
        "messages": translated,
        "tools": _anthropic_offered_tools(body.get("tools"), messages),
        "response_format": response_format,
        "preserve_thinking": preserve_thinking,
        "reasoning_effort": reasoning_effort,
        "chat_template_kwargs": body.get("chat_template_kwargs"),
        "thinking_token_budget": _anthropic_thinking_budget(body.get("thinking")),
    }
    if body.get("tool_choice") is not None:
        chat.update(_anthropic_tool_choice(body["tool_choice"]))
    chat = {key: value for key, value in chat.items() if value is not None}
    return chat, thinking_display


def _anthropic_response_format(body, output_config):
    """The format of a Messages request's output, in output_config.format or
    output_format, as a Chat response_format; None without one."""
    output_format, legacy = output_config.get("format"), body.get("output_format")
    if output_format is not None and legacy is not None:
        raise APIError(400, "output_config.format and output_format cannot be combined")
    output_format = legacy if output_format is None else output_format
    if output_format is None:
        return None
    if (
        not isinstance(output_format, dict)
        or output_format.get("type") != "json_schema"
        or not isinstance(output_format.get("schema"), (dict, bool))
    ):
        raise APIError(400, "output format must be a json_schema object with a schema")
    return {"type": "json_schema", "json_schema": {"schema": output_format["schema"]}}


def _anthropic_message(message, thinking_resolver):
    """The Chat messages a Messages message stands for: a system message's
    text; an assistant's text, reasoning and calls as one message; a user's
    tool results, then the rest of its content."""
    if not isinstance(message, dict):
        raise APIError(400, "Anthropic messages must be objects")
    role = message.get("role")
    content = message.get("content")
    if role not in ("user", "assistant", "system"):
        raise APIError(400, "Anthropic messages require user/assistant/system roles")
    if role == "system":
        if isinstance(content, list):
            # Its tool changes apply to the request's tools.
            content = [block for block in content if not _is_tool_change(block)]
        text = _anthropic_system_text(content, "system message")
        return [{"role": "system", "content": text}] if text else []
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        raise APIError(400, "Anthropic message content must be text or blocks")
    text_parts = []
    user_blocks = []
    reasoning_parts = []
    tool_calls = []
    tool_results = []
    for block in content:
        if not isinstance(block, dict):
            raise APIError(400, "Anthropic content blocks must be objects")
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
            user_blocks.append(block)
        elif role == "user" and kind in ("image", "document"):
            user_blocks.append(block)
        elif (
            role == "assistant"
            and kind == "thinking"
            and isinstance(block.get("thinking"), str)
        ):
            signature = _field(block, "signature", "")
            if not isinstance(signature, str):
                raise APIError(400, "invalid thinking signature")
            if signature:
                try:
                    reasoning = thinking_resolver(signature)
                except APIError as error:
                    # Other providers' signatures are opaque, as are ours
                    # under another key. Preserve their visible history;
                    # hidden content is dropped, as redacted_thinking is.
                    if error.code != "invalid_thinking_signature":
                        raise
                    reasoning = block["thinking"]
                if reasoning:
                    reasoning_parts.append(reasoning)
            else:
                reasoning_parts.append(block["thinking"])
        elif role == "assistant" and kind == "redacted_thinking":
            if not isinstance(block.get("data"), str) or not block["data"]:
                raise APIError(400, "invalid redacted_thinking data")
            # Provider-encrypted reasoning cannot be rendered in this
            # model's prompt. Keep the surrounding visible history.
        elif role == "assistant" and kind == "tool_use":
            if (
                not isinstance(block.get("id"), str)
                or not isinstance(block.get("name"), str)
                or not isinstance(block.get("input"), dict)
            ):
                raise APIError(400, "invalid Anthropic tool_use block")
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {"name": block["name"], "arguments": block["input"]},
                }
            )
        elif role == "user" and kind == "tool_result":
            tool_results.append(_anthropic_tool_result(block))
        else:
            raise APIError(400, f"unsupported Anthropic content block: {kind}")
    if role == "assistant":
        item = {"role": role, "content": "".join(text_parts)}
        if reasoning_parts:
            item["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            item["tool_calls"] = tool_calls
        return [item]
    if user_blocks or not tool_results:
        user = {
            "role": role,
            "content": _anthropic_content(user_blocks, "user message"),
        }
        return [*tool_results, user]
    return tool_results


def _anthropic_tool_result(block):
    """A user's tool_result block as a Chat tool message, whose content
    says first that the tool failed where it did."""
    if not isinstance(block.get("tool_use_id"), str):
        raise APIError(400, "invalid Anthropic tool_result block")
    is_error = _field(block, "is_error", False)
    if not isinstance(is_error, bool):
        raise APIError(400, "tool_result.is_error must be a boolean")
    content = _anthropic_content(
        _tool_references_as_text(_field(block, "content", "")), "tool_result"
    )
    if is_error:
        failed = "Tool execution failed:\n"
        content = (
            failed + content
            if isinstance(content, str)
            else [{"type": "text", "text": failed}, *content]
        )
    return {"role": "tool", "tool_call_id": block["tool_use_id"], "content": content}


def _tool_references_as_text(content):
    """tool_result content with each tool_reference block, a tool search's
    result, as text naming the tool. The prompt already declares every tool of
    the request, deferred ones too, so the name is what the model needs to call
    the tool it found."""
    if not isinstance(content, list):
        return content
    return [
        _tool_reference_text(part)
        if isinstance(part, dict) and part.get("type") == "tool_reference"
        else part
        for part in content
    ]


def _tool_reference_text(block):
    name = block.get("tool_name")
    if not is_tag_name(name):
        raise APIError(400, f"tool_reference.tool_name must be {TAG_NAME_RULE}")
    return {"type": "text", "text": f"\nAvailable tool: {name}\n"}


def _anthropic_tools(tools):
    """Messages tools, custom ones only, as Chat function tools."""
    if not isinstance(tools, list):
        raise APIError(400, "tools must be an array")
    functions = []
    for tool in tools:
        if isinstance(tool, dict) and _field(tool, "type", "custom") != "custom":
            raise APIError(
                400,
                "only custom function tools are supported; "
                f"received type {tool['type']!r}",
            )
        if (
            not isinstance(tool, dict)
            or not isinstance(tool.get("name"), str)
            or not isinstance(_field(tool, "input_schema", {}), (dict, bool))
        ):
            raise APIError(400, "invalid Anthropic tool")
        function = {
            "name": tool["name"],
            "parameters": _field(tool, "input_schema", {}),
        }
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        if "strict" in tool:
            function["strict"] = tool["strict"]
        functions.append({"type": "function", "function": function})
    return functions


_TOOL_CHANGES = ("tool_addition", "tool_removal")


def _is_tool_change(block):
    return isinstance(block, dict) and block.get("type") in _TOOL_CHANGES


def _anthropic_offered_tools(tools, messages):
    """A Messages request's tools as Chat function tools, changed in order by
    the tool_addition and tool_removal blocks of its system messages. A chat
    template declares one tool list, so a tool a message defines joins it,
    replacing a tool of its name, and a removed tool leaves it until added
    again. None without tools."""
    declared = None if tools is None else _anthropic_tools(tools)
    changes = [
        block
        for message in messages
        if message["role"] == "system" and isinstance(message["content"], list)
        for block in message["content"]
        if _is_tool_change(block)
    ]
    if not changes:
        return declared
    offered = {}
    for tool in declared or []:
        name = tool["function"]["name"]
        # Keyed by name, the tools would hide a repeated one from
        # normalize_tools.
        if name in offered:
            raise APIError(400, f"duplicate tool name: {name}")
        offered[name] = tool
    removed = set()
    for block in changes:
        name, definition = _tool_change(block)
        if definition is not None:
            offered[name] = definition
        elif name not in offered:
            raise APIError(
                400, f"{block['type']} names tool {name!r}, which is not declared"
            )
        if block["type"] == "tool_removal":
            removed.add(name)
        else:
            removed.discard(name)
    return [tool for name, tool in offered.items() if name not in removed]


def _tool_change(block):
    """The name of the tool a tool_addition or tool_removal block changes, and
    the Chat tool it defines; None when it names the tool."""
    tool = block.get("tool")
    form = tool.get("type") if isinstance(tool, dict) else None
    if block["type"] == "tool_addition" and form == "tool_definition":
        (definition,) = _anthropic_tools([tool.get("definition")])
        return definition["function"]["name"], definition
    if form != "tool_reference" or not isinstance(tool.get("name"), str):
        forms = (
            "a named tool_reference or a tool_definition"
            if block["type"] == "tool_addition"
            else "a named tool_reference"
        )
        raise APIError(400, f"{block['type']}.tool must be {forms}")
    return tool["name"], None


def _anthropic_tool_choice(choice):
    """A Messages tool_choice as Chat's parallel_tool_calls and
    tool_choice."""
    if not isinstance(choice, dict):
        raise APIError(400, "tool_choice must be an object")
    disable_parallel = _field(choice, "disable_parallel_tool_use", False)
    if not isinstance(disable_parallel, bool):
        raise APIError(400, "disable_parallel_tool_use must be a boolean")
    kind = choice.get("type")
    if kind == "auto":
        tool_choice = "auto"
    elif kind == "any":
        tool_choice = "required"
    elif kind == "none":
        tool_choice = "none"
    elif kind == "tool" and isinstance(choice.get("name"), str):
        tool_choice = {"type": "function", "function": {"name": choice["name"]}}
    else:
        raise APIError(400, "invalid Anthropic tool_choice")
    return {"parallel_tool_calls": not disable_parallel, "tool_choice": tool_choice}


def responses_response(model, job, status, output, result=None, error=None):
    """A Responses API response document; stamp_id_slot adds the compat lane
    to every one, the streamed response.created included."""
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
    return stamp_id_slot(
        {
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
        },
        job,
    )


def finish_reason(result, called):
    """OpenAI finish_reason: the output's whole tool calls count only when
    the token limit did not end it."""
    if called and result.reason != "length":
        return "tool_calls"
    return result.reason


def completion_response(model, job, result, message, called):
    return stamp_id_slot(
        {
            "id": f"chatcmpl-{job.public_id}",
            "object": "chat.completion",
            "created": job.created_at,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason(result, called),
                }
            ],
            "usage": usage_dict(result, job),
            "metrics": result.metrics,
            "timings": timings_dict(result),
        },
        job,
    )


def text_completion_response(model, job, result, text):
    return stamp_id_slot(
        {
            "id": f"cmpl-{job.public_id}",
            "object": "text_completion",
            "created": job.created_at,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": result.reason,
                }
            ],
            "usage": usage_dict(result, job),
            "metrics": result.metrics,
            "timings": timings_dict(result),
        },
        job,
    )


def _chunk(
    object_type, chunk_id, created, model, choice, usage, metrics, timings, job=None
):
    chunk = {
        "id": chunk_id,
        "object": object_type,
        "created": created,
        "model": model,
        "choices": [choice],
    }
    if usage is not None:
        chunk["choices"] = []
        chunk["usage"] = usage
    if metrics is not None:
        chunk["metrics"] = metrics
    if timings is not None:
        chunk["timings"] = timings
    # A proxy scans every stream frame for id_slot and the last occurrence
    # wins, so every chunk of a compat generation carries its lane.
    return stamp_id_slot(chunk, job) if job is not None else chunk


def stream_chunk(
    model,
    request_id,
    created,
    delta,
    finish_reason=None,
    usage=None,
    metrics=None,
    timings=None,
    job=None,
):
    return _chunk(
        "chat.completion.chunk",
        f"chatcmpl-{request_id}",
        created,
        model,
        {"index": 0, "delta": delta, "finish_reason": finish_reason},
        usage,
        metrics,
        timings,
        job,
    )


def text_completion_chunk(
    model,
    request_id,
    created,
    text,
    finish_reason=None,
    usage=None,
    metrics=None,
    timings=None,
    job=None,
):
    return _chunk(
        "text_completion",
        f"cmpl-{request_id}",
        created,
        model,
        {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason},
        usage,
        metrics,
        timings,
        job,
    )


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


def anthropic_stop(result, called, output_clamped_to_context):
    """Anthropic's stop_reason: the output's whole tool calls count only when
    the token limit did not end it."""
    if called and result.reason != "length":
        return "tool_use"
    if result.reason == "length":
        # Anthropic's reason when the context window, not max_tokens, ends
        # the response: max_tokens was lowered to what the context leaves, or
        # the engine ended it at what it holds of the context.
        if output_clamped_to_context or result.bounded:
            return "model_context_window_exceeded"
        return "max_tokens"
    if result.stop_sequence is not None:
        return "stop_sequence"
    return "end_turn"


def anthropic_usage(prompt_tokens, output_tokens, cache, thinking_tokens=None):
    """Messages usage; with thinking_tokens, also the output tokens the
    response spent thinking, in output_tokens_details as Anthropic reports
    them (Chat and Responses report them as reasoning_tokens)."""
    usage = {
        "input_tokens": prompt_tokens - cache.matched_tokens,
        "cache_read_input_tokens": cache.matched_tokens,
        "output_tokens": output_tokens,
    }
    if thinking_tokens is not None:
        usage["output_tokens_details"] = {"thinking_tokens": thinking_tokens}
    return usage


def anthropic_block(job, block, signature=""):
    """The Messages content block of `block` as a complete response shows it
    or, while it is in progress, as a stream starts it, before its text or
    input. Reasoning whose display is omitted shows only `signature`. A call
    no longer in progress has complete arguments: the projector writes each
    call it closes as JSON, and a complete response leaves out a call the
    token limit or a stop sequence cut."""
    if block.kind == "reasoning":
        thinking = "" if job.thinking_display == "omitted" else block.text
        return {"type": "thinking", "thinking": thinking, "signature": signature}
    if block.kind == "text":
        return {"type": "text", "text": block.text}
    return {
        "type": "tool_use",
        "id": block.call_id,
        "name": block.name,
        "input": {} if block.status == "in_progress" else json_codec.loads(block.text),
    }


def anthropic_response(model, job, blocks, result, called, thinking_signature):
    content = [anthropic_block(job, block, thinking_signature) for block in blocks]
    if all(item["type"] == "thinking" for item in content):
        content.append({"type": "text", "text": ""})
    return stamp_id_slot(
        {
            "id": f"msg_{job.public_id}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": anthropic_stop(
                result, called, job.output_clamped_to_context
            ),
            "stop_sequence": result.stop_sequence,
            "usage": anthropic_usage(
                result.prompt_tokens,
                result.completion_tokens,
                result.cache,
                job.reasoning_tokens,
            ),
            "timings": timings_dict(result),
        },
        job,
    )

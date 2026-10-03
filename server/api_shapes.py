"""Request and response shape conversion between the chat, Responses and Anthropic dialects."""

from __future__ import annotations

import hashlib
import json
import re

from . import json_codec
from . import protocol as wire
from .documents import DocumentBudget, document_parts, file_content
from .errors import APIError
from .metrics import timings_dict, usage_dict

IMAGE_PAD_TOKEN = "<|image_pad|>"
VISION_UNAVAILABLE = (
    "this model is serving without vision (started with --language-only)"
)


def _text_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in (
                "text",
                "input_text",
            ):
                raise APIError(400, "only text message content is supported")
            text = part.get("text", "")
            if not isinstance(text, str):
                raise APIError(400, "text message content must be a string")
            parts.append(text)
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
        if kind == "image_url":
            _require_vision("image", vision)
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            parts.append(_image_part(url))
        elif kind in ("text", "input_text"):
            text = part.get("text", "")
            if not isinstance(text, str):
                raise APIError(400, "text message content must be a string")
            parts.append({"type": "text", "text": text})
        elif kind == "file":
            _require_vision("PDF", vision)
            parts.extend(file_content(part.get("file"), budget=document_budget))
        elif kind in ("video", "video_url", "input_audio"):
            raise APIError(400, f"{kind} content is not supported")
        else:
            raise APIError(400, "unsupported message content part")
    return _media_content(parts)


_JSON_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_JSON_STRING_SPECIAL = re.compile(r'["\\\x00-\x1f]')


def _unfinished_json(text):
    """Recognize unfinished JSON in linear space/time, without building values.

    The stack holds one expectation per container, so flat arrays and long
    strings use constant auxiliary memory. Only a valid prefix is accepted.
    """
    stack = ["value"]
    pos, size = 0, len(text)
    while pos < size:
        char = text[pos]
        if char in " \t\r\n":
            pos += 1
            continue
        expected = stack[-1]
        if expected in ("array_end", "object_end"):
            if char == ("]" if expected == "array_end" else "}"):
                stack.pop()
            elif char == ",":
                stack[-1] = "array_value" if expected == "array_end" else "key"
            else:
                return False
            pos += 1
            continue
        if expected == "colon":
            if char != ":":
                return False
            stack[-1] = "object_value"
            pos += 1
            continue
        if expected in ("array_first", "object_first") and char == (
            "]" if expected == "array_first" else "}"
        ):
            stack.pop()
            pos += 1
            continue
        if expected == "done":
            return False
        key = expected in ("key", "object_first")
        if key and char != '"':
            return False
        # Record the continuation before consuming a scalar or entering a
        # nested container. Root completion stays on the stack as a sentinel.
        if key:
            stack[-1] = "colon"
        elif expected == "value":
            stack[-1] = "done"
        elif expected == "object_value":
            stack[-1] = "object_end"
        else:
            stack[-1] = "array_end"
        if char in "[{":
            stack.append("array_first" if char == "[" else "object_first")
            pos += 1
        elif char == '"':
            pos += 1
            while True:
                special = _JSON_STRING_SPECIAL.search(text, pos)
                if special is None:
                    return True
                pos = special.start() + 1
                char = special.group()
                if char == '"':
                    break
                if char != "\\":
                    return False
                if pos == size:
                    return True
                escape = text[pos]
                pos += 1
                if escape == "u":
                    end = min(pos + 4, size)
                    if any(c not in "0123456789abcdefABCDEF" for c in text[pos:end]):
                        return False
                    if end - pos < 4:
                        return True
                    pos = end
                elif escape not in '"\\/bfnrt':
                    return False
        elif char in "tfn":
            literal = {"t": "true", "f": "false", "n": "null"}[char]
            available = text[pos : pos + len(literal)]
            if not literal.startswith(available):
                return False
            pos += len(available)
            if len(available) < len(literal):
                return True
        else:
            number = _JSON_NUMBER.match(text, pos)
            if number is None:
                return char == "-" and pos + 1 == size
            start, pos = pos, number.end()
            if size - pos <= 2:
                suffix = text[pos:]
                exponent = (
                    text.find("e", start, pos) >= 0 or text.find("E", start, pos) >= 0
                )
                if suffix in ("e", "E", "e+", "e-", "E+", "E-"):
                    return not exponent
                if suffix == ".":
                    return not exponent and text.find(".", start, pos) < 0
    return stack != ["done"]


def normalize_messages(messages, *, vision, deadline):
    document_budget = DocumentBudget(deadline=deadline)
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
        if role == "assistant" and message.get("reasoning_content") is not None:
            item["reasoning_content"] = _text_content(message["reasoning_content"])
        if role == "assistant" and message.get("tool_calls") is not None:
            if not isinstance(message["tool_calls"], list):
                raise APIError(400, "assistant tool_calls must be an array")
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    raise APIError(400, "invalid assistant tool call")
                arguments = function.get("arguments", {})
                unfinished = False
                if isinstance(arguments, str):
                    try:
                        # Some providers send a call without arguments as "".
                        arguments = json_codec.loads(arguments.strip() or "{}")
                    except ValueError as error:
                        # Preserve calls truncated by the output limit in history
                        # so the conversation can continue. Invalid complete JSON
                        # is still rejected.
                        if not _unfinished_json(arguments):
                            raise APIError(
                                400, "tool call arguments must be valid JSON"
                            ) from error
                        unfinished = True
                if not (isinstance(arguments, dict) or unfinished) or not isinstance(
                    function.get("name"), str
                ):
                    raise APIError(400, "invalid assistant tool call")
                calls.append(
                    {
                        "type": "function",
                        "function": {"name": function["name"], "arguments": arguments},
                    }
                )
                if "id" in call:
                    if not isinstance(call["id"], str):
                        raise APIError(400, "tool call id must be a string")
                    calls[-1]["id"] = call["id"]
            item["tool_calls"] = calls
        if role == "tool" and "tool_call_id" in message:
            tool_call_id = message["tool_call_id"]
            if not isinstance(tool_call_id, str):
                raise APIError(400, "tool_call_id must be a string")
            item["tool_call_id"] = tool_call_id
        normalized.append(item)
    if not has_user:
        raise APIError(400, "messages must include a user message")
    return normalized


def template_messages(messages):
    """Keep unfinished call text without presenting it as complete arguments."""
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
        elif kind == "function_call_output":
            call_id, output = item.get("call_id"), item.get("output")
            if not isinstance(call_id, str) or not call_id:
                raise APIError(400, "function_call_output requires call_id")
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
                "only message, reasoning, function_call, and "
                "function_call_output input items are supported",
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


def _namespace_alias(namespace, name):
    if (
        not isinstance(namespace, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", namespace) is None
        or not isinstance(name, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None
    ):
        raise APIError(400, "invalid namespace tool name")
    alias = f"{namespace}__{name}"
    if len(alias) <= 64:
        return alias
    digest = hashlib.sha256(f"{namespace}\0{name}".encode()).hexdigest()[:16]
    return f"{namespace[:20]}__{name[:24]}__{digest}"


def _response_function(tool, name=None):
    function = {
        key: tool[key]
        for key in ("name", "description", "parameters", "strict")
        if key in tool
    }
    if name is not None:
        function["name"] = name
    return {"type": "function", "function": function}


def normalize_responses_tools(tools):
    if not isinstance(tools, list):
        raise APIError(400, "tools must be an array")
    output, namespaces = [], {}
    for tool in tools:
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
            raise APIError(400, f"only function tools are supported, not {kind!r}")
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
    """A Responses request as a Chat body, and the namespace and name of
    each namespaced tool by its alias. `items` are the request's canonical
    input items after those of the response it continues."""
    if body.get("conversation") is not None:
        raise APIError(400, "conversation is not supported")
    if body.get("background") not in (None, False):
        raise APIError(400, "background responses are not supported")
    if body.get("truncation") not in (None, "disabled"):
        raise APIError(400, "only disabled truncation is supported")
    if body.get("context_management") not in (None, []):
        raise APIError(400, "Responses context_management edits are not supported")
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
        if choice.get("type") != "function" or not isinstance(choice.get("name"), str):
            raise APIError(400, "invalid named tool_choice")
        name = choice["name"]
        namespace = choice.get("namespace")
        if namespace is not None:
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
        "stop",
        "max_output_tokens",
    ):
        if field_name in body and body[field_name] is not None:
            chat[aliases.get(field_name, field_name)] = body[field_name]
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


def anthropic_to_chat_body(body, *, thinking_resolver):
    """A Messages generation request as a Chat body, and how its response
    shows reasoning. count_tokens converts only the prompt, so it still
    counts a final assistant message."""
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
    if body["messages"][-1]["role"] == "assistant":
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
    )
    chat = {key: value for key, value in chat.items() if value is not None}
    return chat, thinking_display


def _anthropic_preserve_thinking(context_management):
    if context_management is None:
        return None
    if not isinstance(context_management, dict):
        raise APIError(400, "context_management must be an object")
    edits = context_management.get("edits", [])
    if not isinstance(edits, list):
        raise APIError(400, "context_management.edits must be an array")
    for edit in edits:
        if (
            not isinstance(edit, dict)
            or edit.get("type") != "clear_thinking_20251015"
            or edit.get("keep") != "all"
        ):
            raise APIError(
                400,
                "only clear_thinking_20251015 with keep: all is supported "
                "in context_management.edits",
            )
    return True if edits else None


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
    if "display" in thinking:
        if thinking_type == "disabled":
            raise APIError(
                400, "thinking.display requires enabled or adaptive thinking"
            )
        if display not in (None, "summarized", "omitted", "updates"):
            raise APIError(
                400,
                "thinking.display must be summarized, omitted, updates, or null",
            )
    return (
        "none" if thinking_type == "disabled" else effort,
        # Models expose reasoning and text, not separate progress-update blocks.
        "omitted" if display in ("omitted", "updates") else "summarized",
    )


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
    effort = output_config.get("effort", "high")
    if not isinstance(effort, str) or effort not in (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ):
        raise APIError(400, "output_config.effort is invalid")
    if "format" in output_config and "output_format" in body:
        raise APIError(400, "output_config.format and output_format cannot be combined")
    response_format = None
    if "format" in output_config or "output_format" in body:
        output_format = output_config.get("format", body.get("output_format"))
        if (
            not isinstance(output_format, dict)
            or output_format.get("type") != "json_schema"
            or not isinstance(output_format.get("schema"), (dict, bool))
        ):
            raise APIError(
                400, "output format must be a json_schema object with a schema"
            )
        response_format = {
            "type": "json_schema",
            "json_schema": {"schema": output_format["schema"]},
        }
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "messages must be a non-empty array")

    translated = []
    system_text = None
    system = body.get("system")
    if system is not None:
        system_text = _anthropic_system_text(system, "system")
    for message in messages:
        if not isinstance(message, dict):
            raise APIError(400, "Anthropic messages must be objects")
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant", "system"):
            raise APIError(
                400, "Anthropic messages require user/assistant/system roles"
            )
        if role == "system":
            text = _anthropic_system_text(content, "system message")
            if text:
                translated.append({"role": "system", "content": text})
            continue
        if isinstance(content, str):
            translated.append({"role": role, "content": content})
            continue
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
                signature = block.get("signature", "")
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
                        "function": {
                            "name": block["name"],
                            "arguments": block["input"],
                        },
                    }
                )
            elif role == "user" and kind == "tool_result":
                if not isinstance(block.get("tool_use_id"), str):
                    raise APIError(400, "invalid Anthropic tool_result block")
                if not isinstance(block.get("is_error", False), bool):
                    raise APIError(400, "tool_result.is_error must be a boolean")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": _anthropic_content(
                            block.get("content", ""), "tool_result"
                        ),
                    }
                )
                if block.get("is_error"):
                    result = tool_results[-1]
                    content = result["content"]
                    result["content"] = (
                        "Tool execution failed:\n" + content
                        if isinstance(content, str)
                        else [
                            {"type": "text", "text": "Tool execution failed:\n"},
                            *content,
                        ]
                    )
            else:
                raise APIError(400, f"unsupported Anthropic content block: {kind}")
        if role == "assistant":
            item = {"role": role, "content": "".join(text_parts)}
            if reasoning_parts:
                item["reasoning_content"] = "".join(reasoning_parts)
            if tool_calls:
                item["tool_calls"] = tool_calls
            translated.append(item)
        else:
            translated.extend(tool_results)
            if user_blocks or not tool_results:
                translated.append(
                    {
                        "role": role,
                        "content": _anthropic_content(user_blocks, "user message"),
                    }
                )

    if system_text:
        translated.insert(0, {"role": "system", "content": system_text})

    reasoning_effort, thinking_display = _anthropic_thinking(
        body.get("thinking"), effort
    )
    chat = {
        "model": body["model"],
        "messages": translated,
        "response_format": response_format,
        "preserve_thinking": preserve_thinking,
        "reasoning_effort": reasoning_effort,
    }

    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise APIError(400, "tools must be an array")
        chat["tools"] = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type", "custom") != "custom":
                raise APIError(
                    400,
                    "only custom function tools are supported; "
                    f"received type {tool['type']!r}",
                )
            if (
                not isinstance(tool, dict)
                or not isinstance(tool.get("name"), str)
                or not isinstance(tool.get("input_schema", {}), (dict, bool))
            ):
                raise APIError(400, "invalid Anthropic tool")
            function = {
                "name": tool["name"],
                "parameters": tool.get("input_schema", {}),
            }
            if isinstance(tool.get("description"), str):
                function["description"] = tool["description"]
            if "strict" in tool:
                function["strict"] = tool["strict"]
            chat["tools"].append({"type": "function", "function": function})
    choice = body.get("tool_choice")
    if choice is not None:
        if not isinstance(choice, dict):
            raise APIError(400, "tool_choice must be an object")
        disable_parallel = choice.get("disable_parallel_tool_use", False)
        if not isinstance(disable_parallel, bool):
            raise APIError(400, "disable_parallel_tool_use must be a boolean")
        chat["parallel_tool_calls"] = not disable_parallel
        kind = choice.get("type")
        if kind == "auto":
            chat["tool_choice"] = "auto"
        elif kind == "any":
            chat["tool_choice"] = "required"
        elif kind == "none":
            chat["tool_choice"] = "none"
        elif kind == "tool" and isinstance(choice.get("name"), str):
            chat["tool_choice"] = {
                "type": "function",
                "function": {"name": choice["name"]},
            }
        else:
            raise APIError(400, "invalid Anthropic tool_choice")
    chat = {key: value for key, value in chat.items() if value is not None}
    return chat, thinking_display


def responses_response(model, job, status, output, result=None, error=None):
    usage = None
    if result is not None:
        cache = result.cache
        usage = {
            "input_tokens": result.prompt_tokens,
            "input_tokens_details": {
                "cached_tokens": cache.matched_tokens,
                "cache_write_tokens": max(
                    0, result.prompt_tokens - cache.matched_tokens
                ),
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
        "end_turn": result is not None
        and result.reason == "stop"
        and not any(item["type"] == "function_call" for item in output),
    }


def finish_reason(result, tool_calls):
    """OpenAI finish_reason: tool calls count only when the output was not cut."""
    if tool_calls and result.reason != "length":
        return "tool_calls"
    return result.reason


def completion_response(model, job, result, message, tool_calls):
    return {
        "id": f"chatcmpl-{job.public_id}",
        "object": "chat.completion",
        "created": job.created_at,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason(result, tool_calls),
            }
        ],
        "usage": usage_dict(result, job),
        "metrics": result.metrics,
        "timings": timings_dict(result),
    }


def text_completion_response(model, job, result, text):
    return {
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
    }


def _chunk(object_type, chunk_id, created, model, choice, usage, metrics, timings):
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
    return chunk


def stream_chunk(
    model,
    request_id,
    created,
    delta,
    finish_reason=None,
    usage=None,
    metrics=None,
    timings=None,
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
    )


_RESPONSES_ITEM_PREFIXES = {"reasoning": "rs", "text": "msg", "tool": "fc"}


def responses_item_id(job, kind, index):
    """The id of the Responses item of a block of `kind` at `index`."""
    return f"{_RESPONSES_ITEM_PREFIXES[kind]}_{job.public_id}_{index}"


def responses_item(job, block, index):
    """The Responses output item of `block` at `index`. An item in progress
    carries no text or arguments yet."""
    item_id = responses_item_id(job, block.kind, index)
    status = block.status
    text = "" if status == "in_progress" else block.text
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


def anthropic_stop(result, tool_calls, output_clamped_to_context):
    if tool_calls and result.reason != "length":
        return "tool_use"
    if result.reason == "length":
        # Anthropic's reason when the context window, not max_tokens, ends
        # the response.
        if output_clamped_to_context:
            return "model_context_window_exceeded"
        return "max_tokens"
    if result.stop_sequence is not None:
        return "stop_sequence"
    return "end_turn"


def anthropic_usage(prompt_tokens, output_tokens, cache):
    return {
        "input_tokens": prompt_tokens - cache.matched_tokens,
        "cache_read_input_tokens": cache.matched_tokens,
        "output_tokens": output_tokens,
    }


def anthropic_response(model, job, blocks, result, tool_calls, thinking_signature):
    content = []
    for block in blocks:
        if block.kind == "reasoning":
            content.append(
                {
                    "type": "thinking",
                    "thinking": (
                        "" if job.thinking_display == "omitted" else block.text
                    ),
                    "signature": thinking_signature,
                }
            )
        elif block.kind == "text":
            content.append({"type": "text", "text": block.text})
        else:
            try:
                arguments = json_codec.loads(block.text)
            except ValueError:
                # Only a call the token limit cut has unfinished arguments,
                # and a complete message leaves it out.
                if result.reason != "length":
                    raise
                continue
            content.append(
                {
                    "type": "tool_use",
                    "id": block.call_id,
                    "name": block.name,
                    "input": arguments,
                }
            )
    if all(item["type"] == "thinking" for item in content):
        content.append({"type": "text", "text": ""})
    return {
        "id": f"msg_{job.public_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": anthropic_stop(
            result, tool_calls, job.output_clamped_to_context
        ),
        "stop_sequence": result.stop_sequence,
        "usage": anthropic_usage(
            result.prompt_tokens, result.completion_tokens, result.cache
        ),
    }

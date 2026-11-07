"""Chat message normalization, which every API's request reaches."""

from __future__ import annotations

import json

from . import json_codec
from .documents import checked_file
from .errors import APIError

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

"""Anthropic's Messages API: its routes, how its requests convert to Chat,
its answers, a message or its server-sent events, and its error shape."""

from functools import partial

from .. import json_codec, request_fields
from ..chat_messages import field_value, image_part, media_content
from ..documents import document_parts
from ..errors import APIError, ContextLengthError, ErrorDialect
from ..metrics import timings_dict
from ..output import BlockSequencer
from ..tool_schema import TAG_NAME_RULE, is_tag_name


def post_messages(exchange, body, deadline):
    app = exchange.app
    streamed, flags = request_fields.generation(body)
    chat, omitted = to_chat_body(body, thinking_resolver=app.thinking_codec.decode)
    job = app.prepare(
        chat,
        deadline=deadline,
        flags=flags,
        output_field="max_tokens",
        stop_field="stop_sequences",
    )
    respond = stream if streamed else complete
    return job, partial(respond, exchange, omitted=omitted)


def post_count_tokens(exchange, body, deadline):
    app = exchange.app
    prompt = to_chat_prompt(body, thinking_resolver=app.thinking_codec.decode)
    tokens = app.count_tokens(prompt, deadline=deadline)
    exchange.send_json(200, {"input_tokens": tokens})


def _system_text(value, label):
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


def _content(value, label):
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
                image_part(f"data:{source['media_type']};base64,{source['data']}")
            )
        elif kind == "document":
            parts.extend(document_parts(block))
        else:
            raise APIError(400, f"{label} contains unsupported content")
    return media_content(parts)


def to_chat_body(body, *, thinking_resolver):
    """A Messages generation request as a Chat body, and whether its
    response omits reasoning text. count_tokens converts only the prompt, so
    it still counts a final assistant message."""
    max_tokens = body.get("max_tokens")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise APIError(400, "max_tokens must be a positive integer")
    chat, omitted = _chat(body, thinking_resolver)
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
        stop=_stop_sequences(body.get("stop_sequences")),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        seed=body.get("seed"),
        priority=body.get("priority"),
        cache_salt=body.get("cache_salt"),
    )
    chat = {key: value for key, value in chat.items() if value is not None}
    return chat, omitted


def _stop_sequences(stops):
    """A Messages request's stop_sequences, as Chat's stop: an array of
    strings that each hold a character other than whitespace, as Anthropic
    requires. Chat holds them to Splash's limits on stops."""
    if stops is None:
        return None
    if not isinstance(stops, list) or not all(isinstance(stop, str) for stop in stops):
        raise APIError(400, "stop_sequences must be an array of strings")
    if not all(stop.strip() for stop in stops):
        raise APIError(400, "each of stop_sequences must contain non-whitespace")
    return stops


def _preserve_thinking(context_management):
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


def _thinking(thinking, effort):
    """The reasoning effort of a Messages request's thinking, which is off
    when the field is absent, and whether its response omits reasoning text."""
    if thinking is None:
        return "none", False
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
        display in ("omitted", "updates"),
    )


def _thinking_budget(thinking):
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


def to_chat_prompt(body, *, thinking_resolver):
    """The prompt of a Messages request as a Chat body, as count_tokens
    counts it."""
    return _chat(body, thinking_resolver)[0]


def _chat(body, thinking_resolver):
    """A Messages request's prompt as a Chat body, and whether its response
    omits reasoning text."""
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise APIError(400, "model must be a non-empty string")
    preserve_thinking = _preserve_thinking(body.get("context_management"))
    output_config = body.get("output_config", {})
    if not isinstance(output_config, dict):
        raise APIError(400, "output_config must be an object")
    effort = field_value(output_config, "effort", "high")
    if not isinstance(effort, str) or effort not in (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ):
        raise APIError(400, "output_config.effort is invalid")
    response_format = _response_format(body, output_config)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "messages must be a non-empty array")
    system = body.get("system")
    system_text = None if system is None else _system_text(system, "system")
    translated = [
        translation
        for message in messages
        for translation in _message(message, thinking_resolver)
    ]
    if system_text:
        translated.insert(0, {"role": "system", "content": system_text})
    reasoning_effort, omitted = _thinking(body.get("thinking"), effort)
    chat = {
        "model": body["model"],
        "messages": translated,
        "tools": _offered_tools(body.get("tools"), messages),
        "response_format": response_format,
        "preserve_thinking": preserve_thinking,
        "reasoning_effort": reasoning_effort,
        "chat_template_kwargs": body.get("chat_template_kwargs"),
        "thinking_token_budget": _thinking_budget(body.get("thinking")),
    }
    if body.get("tool_choice") is not None:
        chat.update(_tool_choice(body["tool_choice"]))
    chat = {key: value for key, value in chat.items() if value is not None}
    return chat, omitted


def _response_format(body, output_config):
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


def _message(message, thinking_resolver):
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
        text = _system_text(content, "system message")
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
            signature = field_value(block, "signature", "")
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
            tool_results.append(_tool_result(block))
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
            "content": _content(user_blocks, "user message"),
        }
        return [*tool_results, user]
    return tool_results


def _tool_result(block):
    """A user's tool_result block as a Chat tool message, whose content
    says first that the tool failed where it did."""
    if not isinstance(block.get("tool_use_id"), str):
        raise APIError(400, "invalid Anthropic tool_result block")
    is_error = field_value(block, "is_error", False)
    if not isinstance(is_error, bool):
        raise APIError(400, "tool_result.is_error must be a boolean")
    content = _content(
        _tool_references_as_text(field_value(block, "content", "")), "tool_result"
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


def _tools(tools):
    """Messages tools, custom ones only, as Chat function tools."""
    if not isinstance(tools, list):
        raise APIError(400, "tools must be an array")
    functions = []
    for tool in tools:
        if isinstance(tool, dict) and field_value(tool, "type", "custom") != "custom":
            raise APIError(
                400,
                "only custom function tools are supported; "
                f"received type {tool['type']!r}",
            )
        if (
            not isinstance(tool, dict)
            or not isinstance(tool.get("name"), str)
            or not isinstance(field_value(tool, "input_schema", {}), (dict, bool))
        ):
            raise APIError(400, "invalid Anthropic tool")
        function = {
            "name": tool["name"],
            "parameters": field_value(tool, "input_schema", {}),
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


def _offered_tools(tools, messages):
    """A Messages request's tools as Chat function tools, changed in order by
    the tool_addition and tool_removal blocks of its system messages. A chat
    template declares one tool list, so a tool a message defines joins it,
    replacing a tool of its name, and a removed tool leaves it until added
    again. None without tools."""
    declared = None if tools is None else _tools(tools)
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
        (definition,) = _tools([tool.get("definition")])
        return definition["function"]["name"], definition
    if form != "tool_reference" or not isinstance(tool.get("name"), str):
        forms = (
            "a named tool_reference or a tool_definition"
            if block["type"] == "tool_addition"
            else "a named tool_reference"
        )
        raise APIError(400, f"{block['type']}.tool must be {forms}")
    return tool["name"], None


def _tool_choice(choice):
    """A Messages tool_choice as Chat's parallel_tool_calls and
    tool_choice."""
    if not isinstance(choice, dict):
        raise APIError(400, "tool_choice must be an object")
    disable_parallel = field_value(choice, "disable_parallel_tool_use", False)
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


def complete(exchange, job, *, omitted):
    app = exchange.app
    sequencer = BlockSequencer()
    collected = exchange.collect(
        job,
        on_text=sequencer.text,
        on_tool_delta=sequencer.tool,
    )
    result = collected.result
    blocks = sequencer.finish(collected.cut, collected.reasoning_open)
    if collected.cut_call is not None:
        # A complete message leaves out the call the output cut.
        blocks = [block for block in blocks if block.call_id != collected.cut_call]
    signature = (
        app.thinking_codec.encode(collected.reasoning)
        if collected.reasoning and omitted
        else ""
    )
    exchange.send_json(
        200,
        message_response(
            app.response_model,
            job,
            blocks,
            result,
            collected.called,
            omitted,
            signature,
        ),
    )


def stream(exchange, job, *, omitted):
    app = exchange.app

    def send(event, payload):
        exchange.event_sse(event, {"type": event, **payload})

    def start():
        # Cache accounting is known at native admission. Keep the socket
        # alive while queued, but do not publish guessed input usage.
        exchange.start_event_stream()
        send(
            "message_start",
            {
                "message": {
                    "id": f"msg_{job.public_id}",
                    "type": "message",
                    "role": "assistant",
                    "model": app.response_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": message_usage(len(job.prompt_tokens), 0, job.cache),
                }
            },
        )

    def keepalive():
        exchange.start_event_stream()
        send("ping", {})

    def open_block(index, block):
        send(
            "content_block_start",
            {"index": index, "content_block": content_block(block, omitted)},
        )
        if block.kind == "reasoning" and omitted:
            send(
                "content_block_delta",
                {
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": ""},
                },
            )

    def block_delta(index, block, text):
        if block.kind == "reasoning":
            if omitted:
                return
            delta = {"type": "thinking_delta", "thinking": text}
        elif block.kind == "text":
            delta = {"type": "text_delta", "text": text}
        else:
            delta = {"type": "input_json_delta", "partial_json": text}
        send("content_block_delta", {"index": index, "delta": delta})

    def close_block(index, block):
        if block.kind == "reasoning" and omitted:
            signature = app.thinking_codec.encode(block.text)
            send(
                "content_block_delta",
                {
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": signature},
                },
            )
        send("content_block_stop", {"index": index})

    sequencer = BlockSequencer(open_block, block_delta, close_block)

    def run():
        collected = exchange.collect(
            job,
            on_start=start,
            on_text=sequencer.text,
            on_tool_delta=sequencer.tool,
            on_idle=keepalive,
            on_progress=lambda progress: send("ping", {"prompt_progress": progress}),
        )
        result = collected.result
        sequencer.finish(collected.cut, collected.reasoning_open)
        send(
            "message_delta",
            {
                "delta": {
                    "stop_reason": stop_reason(
                        result,
                        collected.called,
                        job.output_clamped_to_context,
                    ),
                    "stop_sequence": result.stop_sequence,
                },
                "usage": {
                    "output_tokens": result.completion_tokens,
                    "output_tokens_details": {"thinking_tokens": job.reasoning_tokens},
                },
                "timings": timings_dict(result),
            },
        )
        send("message_stop", {})

    def send_error(error):
        exchange.event_sse("error", exchange.errors.payload(error))

    exchange.guarded_stream(job, run, send_error)


def stop_reason(result, called, output_clamped_to_context):
    """Anthropic's stop_reason: tool_use when the output's tool calls count as
    made (Collected.called)."""
    if called:
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


def message_usage(prompt_tokens, output_tokens, cache, thinking_tokens=None):
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


def content_block(block, omitted, signature=""):
    """The Messages content block of `block` as a complete response shows it
    or, while it is in progress, as a stream starts it, before its text or
    input. Reasoning whose display is `omitted` shows only `signature`. A call
    no longer in progress has complete arguments: the projector writes each
    call it closes as JSON, and a complete response leaves out a call the
    token limit or a stop sequence cut."""
    if block.kind == "reasoning":
        thinking = "" if omitted else block.text
        return {"type": "thinking", "thinking": thinking, "signature": signature}
    if block.kind == "text":
        return {"type": "text", "text": block.text}
    return {
        "type": "tool_use",
        "id": block.call_id,
        "name": block.name,
        "input": {} if block.status == "in_progress" else json_codec.loads(block.text),
    }


def message_response(model, job, blocks, result, called, omitted, thinking_signature):
    content = [content_block(block, omitted, thinking_signature) for block in blocks]
    if all(item["type"] == "thinking" for item in content):
        content.append({"type": "text", "text": ""})
    return {
        "id": f"msg_{job.public_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(result, called, job.output_clamped_to_context),
        "stop_sequence": result.stop_sequence,
        "usage": message_usage(
            result.prompt_tokens,
            result.completion_tokens,
            result.cache,
            job.reasoning_tokens,
        ),
        "timings": timings_dict(result),
    }


class AnthropicErrors(ErrorDialect):
    # The type of an error of each status Anthropic's API names one for; any
    # other is an api_error from 500, else an invalid_request_error.
    TYPES = {
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        408: "timeout_error",
        413: "request_too_large",
        429: "rate_limit_error",
        503: "overloaded_error",
        504: "timeout_error",
    }

    def payload(self, error):
        message = error.message
        if isinstance(error, ContextLengthError):
            message = (
                f"prompt is too long: {error.input_tokens} tokens > "
                f"{error.maximum_input_tokens} maximum input tokens"
            )
            if error.image_tokens_only:
                message += " (image tokens alone; text not yet counted)"
        error_type = self.TYPES.get(
            error.status,
            "api_error" if error.status >= 500 else "invalid_request_error",
        )
        return {"type": "error", "error": {"type": error_type, "message": message}}


ERRORS = AnthropicErrors()

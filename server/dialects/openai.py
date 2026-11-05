"""OpenAI's Chat and text completions: their routes, answers and builders.
A completion answers whole, or as chunks in server-sent events. Text
completions have no tools."""

from functools import partial

from .. import request_fields
from ..errors import APIError
from ..metrics import timings_dict, usage_dict


def post_chat(exchange, body, deadline):
    return _post(exchange, body, deadline, exchange.app.prepare, chat=True)


def post_completion(exchange, body, deadline):
    return _post(exchange, body, deadline, exchange.app.prepare_completion, chat=False)


def post_chat_count_tokens(exchange, body, deadline):
    tokens = exchange.app.count_tokens(body, deadline=deadline)
    exchange.send_json(200, {"input_tokens": tokens})


def _post(exchange, body, deadline, prepare, *, chat):
    """Prepare a Chat or text completion, and the answer that writes its
    output: whole, or as a stream."""
    streamed, flags = request_fields.generation(body)
    stream_options = _stream_options(body)
    job = prepare(body, deadline=deadline, flags=flags)
    if streamed:
        return job, partial(stream, exchange, stream_options=stream_options, chat=chat)
    return job, partial(complete, exchange, chat=chat)


def _stream_options(body):
    """A Chat or text completion request's stream_options."""
    options = body.get("stream_options", {})
    if not isinstance(options, dict):
        raise APIError(400, "invalid streaming options")
    # A null include_usage is one left out, as a null field is.
    include_usage = options.get("include_usage")
    if include_usage is None:
        include_usage = False
    if not isinstance(include_usage, bool):
        raise APIError(400, "invalid streaming options")
    return {"include_usage": include_usage}


def complete(exchange, job, *, chat):
    collected = exchange.collect(job)
    model = exchange.app.response_model
    if not chat:
        exchange.send_json(
            200,
            text_completion_response(model, job, collected.result, collected.content),
        )
        return
    message = {"role": "assistant", "content": collected.content or None}
    if collected.reasoning:
        message["reasoning_content"] = collected.reasoning
    if collected.tool_calls:
        message["tool_calls"] = collected.tool_calls
    exchange.send_json(
        200,
        completion_response(model, job, collected.result, message, collected.called),
    )


def stream(exchange, job, stream_options, *, chat):
    chunk = partial(
        stream_chunk if chat else text_completion_chunk,
        exchange.app.response_model,
        job.public_id,
        job.created_at,
    )
    empty = {} if chat else ""
    started = False

    def payload(field, text):
        return {field: text} if chat else text

    def start():
        nonlocal started
        started = True
        exchange.start_event_stream()
        if chat:
            exchange.sse(chunk({"role": "assistant", "content": ""}))

    def keepalive():
        # After the start, a chunk that adds nothing: tool arguments of
        # arrays and objects are buffered until complete, which can take
        # minutes, and clients that time out on missing data events ignore
        # SSE comments.
        if started:
            exchange.sse(chunk(empty))
        else:
            exchange.sse_keepalive()

    def run():
        collected = exchange.collect(
            job,
            on_start=start,
            on_text=lambda field, text: exchange.sse(chunk(payload(field, text))),
            on_tool_delta=lambda delta: exchange.sse(chunk({"tool_calls": [delta]})),
            on_idle=keepalive,
            on_progress=lambda progress: exchange.sse(
                chunk(empty) | {"prompt_progress": progress}
            ),
        )
        result = collected.result
        exchange.sse(
            chunk(
                empty,
                finish_reason(result, collected.called),
                timings=timings_dict(result),
            )
        )
        if stream_options.get("include_usage"):
            exchange.sse(
                chunk(empty, usage=usage_dict(result, job), metrics=result.metrics)
            )
        exchange.sse("[DONE]")

    def send_error(error):
        exchange.sse(exchange.errors.payload(error))
        exchange.sse("[DONE]")

    exchange.guarded_stream(job, run, send_error)


def finish_reason(result, called):
    """OpenAI finish_reason: tool_calls when the output's tool calls count as
    made (Collected.called)."""
    if called:
        return "tool_calls"
    return result.reason


def completion_response(model, job, result, message, called):
    return {
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

"""The OpenAI Chat and text completion answers: a completion, or its chunks
as server-sent events. Text completions have no tools."""

from functools import partial

from ..api_shapes import (
    completion_response,
    finish_reason,
    stream_chunk,
    text_completion_chunk,
    text_completion_response,
)
from ..metrics import timings_dict, usage_dict


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

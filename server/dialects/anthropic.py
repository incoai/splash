"""The Anthropic Messages answer: a message, or its server-sent events."""

from ..api_shapes import (
    anthropic_block,
    anthropic_response,
    anthropic_stop,
    anthropic_usage,
)
from ..metrics import timings_dict
from ..output import BlockSequencer


def complete(exchange, job):
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
        if collected.reasoning and job.thinking_display == "omitted"
        else ""
    )
    exchange.send_json(
        200,
        anthropic_response(
            app.response_model,
            job,
            blocks,
            result,
            collected.called,
            signature,
        ),
    )


def stream(exchange, job):
    app = exchange.app
    omitted = job.thinking_display == "omitted"

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
                    "usage": anthropic_usage(len(job.prompt_tokens), 0, job.cache),
                }
            },
        )

    def keepalive():
        exchange.start_event_stream()
        send("ping", {})

    def open_block(index, block):
        send(
            "content_block_start",
            {"index": index, "content_block": anthropic_block(job, block)},
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
                    "stop_reason": anthropic_stop(
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

"""The OpenAI Responses answer: a response, or its server-sent events."""

from ..api_shapes import (
    is_custom_call,
    responses_item,
    responses_item_id,
    responses_output,
    responses_response,
)
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
    incomplete = result.reason == "length"
    output = responses_output(
        job, sequencer.finish(collected.cut, collected.reasoning_open)
    )
    response = responses_response(
        app.response_model,
        job,
        "incomplete" if incomplete else "completed",
        output,
        result=result,
    )
    app.persist_response(job, response, output)
    exchange.send_json(200, response)


def stream(exchange, job):
    app = exchange.app
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
        send(
            "response.created",
            response=responses_response(app.response_model, job, "in_progress", []),
        )
        send(
            "response.in_progress",
            response=responses_response(app.response_model, job, "in_progress", []),
        )

    def open_item(index, block):
        item = responses_item(job, block, index)
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
            item_id=responses_item_id(job, block.kind, index),
            output_index=index,
            delta=text,
            **extra,
        )

    def close_item(index, block):
        item = responses_item(job, block, index)
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
            send(
                "response.in_progress",
                response=responses_response(app.response_model, job, "in_progress", []),
            )
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
                response=responses_response(app.response_model, job, "in_progress", []),
                prompt_progress=progress,
            ),
        )
        result = collected.result
        incomplete = result.reason == "length"
        sequencer.finish(collected.cut, collected.reasoning_open)
        status = "incomplete" if incomplete else "completed"
        response = responses_response(
            app.response_model, job, status, output, result=result
        )
        app.persist_response(job, response, output)
        send(f"response.{status}", response=response)

    def send_error(error):
        send(
            "response.failed",
            response=responses_response(
                app.response_model,
                job,
                "failed",
                output,
                error=exchange.errors.payload(error)["error"],
            ),
        )

    exchange.guarded_stream(job, run, send_error)

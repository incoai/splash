"""Bounded qualification of an existing local Flash-Next HTTP server.

This script sends inference requests only when executed. It never starts,
restarts, builds, or reconfigures a server. Short qualification timings are
reported as observations, not as a throughput benchmark or speed comparison.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import ipaddress
import json
import math
import os
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
COUNTERS = (
    "requests.submitted",
    "requests.completed",
    "requests.cancelled",
    "requests.failed",
    "scheduler.prefill_batches",
    "scheduler.prefill_rows",
    "scheduler.decode_batches",
    "scheduler.decode_batches_by_width.b1",
    "scheduler.decode_batches_by_width.b2",
    "scheduler.decode_batches_by_width.b3",
    "scheduler.decode_batches_by_width.b4",
    "metrics.autoregressive_output_tokens",
    "metrics.prefill_input_tokens",
    "metrics.drafted_tokens",
    "metrics.accepted_draft_tokens",
    "metrics.metal_failures",
    "model_timing.prefill.total_gpu_ms",
    "model_timing.prefill.total_wall_ms",
    "model_timing.prefill.forward_host_wall_ms",
    "model_timing.decode.total_gpu_ms",
    "model_timing.decode.total_wall_ms",
    "model_timing.decode.forward_host_wall_ms",
)


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def same_json(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            same_json(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_json(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def get_path(document, path):
    value = document
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def counter_delta(before, after):
    result = {}
    for path in COUNTERS:
        left, right = get_path(before, path), get_path(after, path)
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and math.isfinite(left)
            and math.isfinite(right)
        ):
            result[path] = right - left
    return result


def validate_status(status, expected_identity=None):
    errors = []
    for path in ("ready", "metal.healthy", "transport.ready", "memory_audit.valid"):
        if get_path(status, path) is not True:
            errors.append(f"status {path} is not true")
    if get_path(status, "transport.status_stale") is not False:
        errors.append("native status is stale or freshness is unavailable")
    if get_path(status, "transport.recovering") is not False:
        errors.append("native transport is recovering or recovery state is unavailable")
    if expected_identity is not None and status.get("identity") != expected_identity:
        errors.append("native runtime identity changed")
    return errors


def idle(status):
    paths = (
        "transport.pending",
        "frontend.active",
        "frontend.waiting",
        "http.requests.active",
        "scheduler.queued",
        "scheduler.active_requests",
        "scheduler.prefilling",
        "scheduler.decoding",
        "scheduler.waiting_mask",
    )
    return all(get_path(status, path) in (None, 0) for path in paths) and (
        get_path(status, "scheduler.command_in_flight") in (None, False)
    )


def _blank_record(stream):
    return {
        "stream": stream,
        "text": "",
        "tool_calls": [],
        "finish_reason": None,
        "usage": None,
        "metrics": {},
        "events": [],
        "done": False,
        "first_content_ms": None,
        "request_ids": [],
        "model_ids": [],
        "error_frames": [],
        "errors": [],
        "reasoning_text": "",
    }


def _remember(record, document):
    for field, target in (("id", "request_ids"), ("model", "model_ids")):
        value = document.get(field)
        if isinstance(value, str) and value not in record[target]:
            record[target].append(value)


def parse_sse(lines):
    """Parse timestamped SSE lines; elapsed times share one client clock origin."""
    record = _blank_record(True)
    data, calls = [], {}

    def event(elapsed):
        if not data:
            return
        raw = "\n".join(data)
        data.clear()
        if record["done"]:
            record["errors"].append("SSE data arrived after [DONE]")
            return
        if raw == "[DONE]":
            record["done"] = True
            record["done_ms"] = elapsed
            return
        try:
            document = strict_json(raw)
            if not isinstance(document, dict):
                raise ValueError("SSE JSON must be an object")
        except (ValueError, TypeError) as error:
            record["errors"].append(f"invalid SSE JSON: {error}")
            return
        record["events"].append({"elapsed_ms": elapsed, "data": document})
        _remember(record, document)
        if "error" in document:
            record["error_frames"].append(document["error"])
        if document.get("usage") is not None:
            if record["usage"] is not None:
                record["errors"].append("multiple final usage frames")
            record["usage"] = document["usage"]
            record["metrics"] = document.get("metrics", {})
        choices = document.get("choices", [])
        if not isinstance(choices, list):
            record["errors"].append("SSE choices must be an array")
            return
        for choice in choices:
            if not isinstance(choice, dict):
                record["errors"].append("SSE choice must be an object")
                continue
            if choice.get("index", 0) != 0:
                record["errors"].append("unexpected choice index")
            reason = choice.get("finish_reason")
            if reason is not None:
                if record["finish_reason"] is not None:
                    record["errors"].append("multiple finish frames")
                record["finish_reason"] = reason
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                record["errors"].append("SSE delta must be an object")
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                if record["first_content_ms"] is None:
                    record["first_content_ms"] = elapsed
                record["text"] += content
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str):
                record["reasoning_text"] += reasoning
            parts = delta.get("tool_calls", [])
            if not isinstance(parts, list):
                record["errors"].append("SSE tool_calls must be an array")
                continue
            for part in parts:
                if not isinstance(part, dict) or not isinstance(
                    part.get("function", {}), dict
                ):
                    record["errors"].append("SSE tool delta must be an object")
                    continue
                index = part.get("index", 0)
                if type(index) is not int or not 0 <= index < 16:
                    record["errors"].append("invalid streamed tool index")
                    continue
                call = calls.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if "id" in part:
                    if not isinstance(part["id"], str) or (
                        call["id"] and call["id"] != part["id"]
                    ):
                        record["errors"].append(
                            "streamed tool ID changed or is invalid"
                        )
                    else:
                        call["id"] = part["id"]
                for key in ("name", "arguments"):
                    value = part.get("function", {}).get(key, "")
                    if not isinstance(value, str):
                        record["errors"].append(
                            "streamed tool name/arguments must be strings"
                        )
                    else:
                        call["function"][key] += value

    last = 0.0
    for elapsed, raw in lines:
        last = elapsed
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            record["errors"].append("invalid UTF-8 SSE line")
            continue
        if not line:
            event(elapsed)
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif line.startswith(":") or line.startswith(("event:", "id:", "retry:")):
            continue
        else:
            record["errors"].append("unexpected SSE field")
    if data:
        event(last)
        record["errors"].append("SSE ended without an event separator")
    record["tool_calls"] = [calls[index] for index in sorted(calls)]
    return record


def normalize_response(status, content_type, raw_bytes):
    record = _blank_record(False)
    record.update(http_status=status, content_type=content_type)
    try:
        document = strict_json(raw_bytes)
        if not isinstance(document, dict):
            raise ValueError("response JSON must be an object")
    except (ValueError, TypeError, UnicodeError) as error:
        record["errors"].append(f"invalid response JSON: {error}")
        record["raw_response"] = raw_bytes.decode("utf-8", errors="replace")
        return record
    record["response"] = document
    _remember(record, document)
    if "error" in document:
        record["error_frames"].append(document["error"])
    choices = document.get("choices", [])
    if not isinstance(choices, list):
        record["errors"].append("completion choices must be an array")
    elif (
        len(choices) == 1
        and isinstance(choices[0], dict)
        and isinstance(choices[0].get("message"), dict)
    ):
        message = choices[0].get("message", {})
        text, tool_calls = message.get("content") or "", message.get("tool_calls", [])
        reasoning = message.get("reasoning_content") or ""
        if not isinstance(text, str) or not isinstance(reasoning, str):
            record["errors"].append("completion content/reasoning must be strings")
            text, reasoning = "", ""
        if not isinstance(tool_calls, list) or any(
            not isinstance(call, dict) or not isinstance(call.get("function"), dict)
            for call in tool_calls
        ):
            record["errors"].append(
                "completion tool_calls must contain function objects"
            )
            tool_calls = []
        for call in tool_calls:
            if not isinstance(call.get("id"), str):
                record["errors"].append("completion tool ID must be a string")
            if any(
                not isinstance(call["function"].get(key), str)
                for key in ("name", "arguments")
            ):
                record["errors"].append(
                    "completion tool name/arguments must be strings"
                )
        record.update(
            text=text,
            tool_calls=tool_calls,
            finish_reason=choices[0].get("finish_reason"),
            reasoning_text=reasoning,
        )
        record["message"] = message
    elif status == 200:
        record["errors"].append("expected exactly one completion choice")
    record["usage"] = document.get("usage")
    record["metrics"] = document.get("metrics", {})
    return record


def check_answer(record, expected, budget):
    errors = list(record.get("errors", []))
    if record.get("http_status") != 200:
        errors.append(f"HTTP status {record.get('http_status')} instead of 200")
    if record.get("error_frames"):
        errors.append("server returned an error frame")
    if record.get("stream"):
        if not record.get("content_type", "").startswith("text/event-stream"):
            errors.append("stream response is not text/event-stream")
        if not record.get("done"):
            errors.append("stream lacks [DONE]")
        if record.get("first_content_ms") is None and not record.get("tool_calls"):
            errors.append("stream emitted no content")
    elif not record.get("content_type", "").startswith("application/json"):
        errors.append("completion response is not application/json")
    if len(record.get("request_ids", [])) != 1 or len(record.get("model_ids", [])) != 1:
        errors.append("response lacks a single consistent request/model identity")
    if record.get("expected_model") is not None and record.get("model_ids") != [
        record["expected_model"]
    ]:
        errors.append("response model differs from requested model")
    if record.get("reasoning_text"):
        errors.append("reasoning text appeared for reasoning_effort=none")
    usage = record.get("usage")
    if not isinstance(usage, dict):
        errors.append("response lacks final usage")
    else:
        prompt, completion, total = (
            usage.get(key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        if not all(type(value) is int for value in (prompt, completion, total)):
            errors.append("usage token counts are not integers")
        elif not (
            prompt > 0 and 0 < completion <= budget and total == prompt + completion
        ):
            errors.append("usage accounting or completion budget is invalid")
        if get_path(usage, "completion_tokens_details.reasoning_tokens") not in (
            None,
            0,
        ):
            errors.append("reasoning token usage is nonzero for reasoning_effort=none")
        if record.get("cache_disabled"):
            if get_path(record, "metrics.cache.matched_tokens") != 0:
                errors.append("cache-disabled request reports reused tokens")
            if get_path(record, "metrics.prefill.tokens") != prompt:
                errors.append("cache-disabled prefill work differs from prompt usage")
    kind = expected["kind"]
    if kind == "tool":
        calls = record.get("tool_calls", [])
        if record.get("finish_reason") != "tool_calls" or len(calls) != 1:
            errors.append("expected exactly one tool call and tool_calls finish reason")
        else:
            call = calls[0]
            function = call.get("function", {})
            if call.get("type") != "function":
                errors.append("tool call type is not function")
            if not isinstance(call.get("id"), str) or not call["id"]:
                errors.append("tool call lacks an ID")
            if function.get("name") != expected["name"]:
                errors.append("tool call name differs")
            try:
                arguments = strict_json(function.get("arguments", ""))
                if not same_json(arguments, expected["arguments"]):
                    errors.append("tool call arguments differ")
            except (ValueError, TypeError):
                errors.append("tool call arguments are invalid JSON")
    else:
        if record.get("tool_calls"):
            errors.append("unexpected tool call in final answer")
        if record.get("finish_reason") != "stop":
            errors.append("short coherent answer did not finish with stop")
        if kind == "text":
            if record.get("text", "").strip() != expected["value"]:
                errors.append("answer text differs from expected value")
        elif kind == "json":
            try:
                if not same_json(
                    strict_json(record.get("text", "")), expected["value"]
                ):
                    errors.append("answer JSON differs in value, keys, or types")
            except (ValueError, TypeError):
                errors.append("answer is not strict JSON")
    return errors


def chat_body(model, messages, budget, **options):
    return {
        "model": model,
        "messages": messages,
        "max_completion_tokens": budget,
        "reasoning_effort": "none",
        "temperature": 0,
        "seed": 0,
        **options,
    }


def build_plan(model, nonce):
    def messages(name, question):
        return [
            {"role": "user", "content": f"Qualification {nonce}/{name}.\n{question}"}
        ]

    structured = {
        "type": "object",
        "properties": {
            "answer": {"type": "integer"},
            "label": {"type": "string", "const": "system-developer"},
            "ok": {"type": "boolean"},
        },
        "required": ["answer", "label", "ok"],
        "additionalProperties": False,
    }
    instructions = [
        {"role": "system", "content": "Answer accurately using the requested format."},
        {
            "role": "developer",
            "content": "Use integers for arithmetic answers. Include no Markdown or explanations.",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    tool_messages = [
        *instructions,
        *messages("tool", "Call add_numbers with a=19 and b=23. Wait for its result."),
    ]
    lanes = []
    for lane, (question, answer) in enumerate(
        (("14 + 37", 51), ("8 * 9", 72), ("144 / 12", 12), ("91 - 28", 63))
    ):
        schema = {
            "type": "object",
            "properties": {
                "lane": {"type": "integer", "const": lane},
                "answer": {"type": "integer"},
            },
            "required": ["lane", "answer"],
            "additionalProperties": False,
        }
        lanes.append(
            {
                "lane": lane,
                "body": chat_body(
                    model,
                    messages(
                        f"concurrent/{lane}",
                        f"Calculate {question}. Return JSON with lane={lane} and the integer answer.",
                    ),
                    96,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": f"lane_{lane}",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                ),
                "expected": {"kind": "json", "value": {"lane": lane, "answer": answer}},
            }
        )
    long_messages = messages(
        "control",
        "Write consecutive integers from 1 to 100000, one per line. Keep counting until the output limit. Include no other text.",
    )
    return [
        {
            "id": "stream_arithmetic",
            "body": chat_body(
                model,
                messages(
                    "stream", "Calculate 17 * 23. Reply with only the integer answer."
                ),
                64,
                stream=True,
                stream_options={"include_usage": True},
            ),
            "expected": {"kind": "text", "value": "391"},
        },
        {
            "id": "structured_system_developer",
            "feature": "structured_output",
            "body": chat_body(
                model,
                [
                    *instructions,
                    *messages(
                        "structured",
                        "Calculate 6 * 7. Return JSON with answer, label=system-developer and ok=true.",
                    ),
                ],
                96,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "instructions_probe",
                        "strict": True,
                        "schema": structured,
                    },
                },
            ),
            "expected": {
                "kind": "json",
                "value": {"answer": 42, "label": "system-developer", "ok": True},
            },
        },
        {"id": "concurrent_four", "lanes": lanes},
        {
            "id": "tool_call",
            "feature": "tools",
            "body": chat_body(
                model,
                tool_messages,
                192,
                tools=tools,
                parallel_tool_calls=False,
                tool_choice={"type": "function", "function": {"name": "add_numbers"}},
            ),
            "expected": {
                "kind": "tool",
                "name": "add_numbers",
                "arguments": {"a": 19, "b": 23},
            },
        },
        {
            "id": "tool_continuation",
            "feature": "tools",
            "depends_on": "tool_call",
            "body": chat_body(
                model,
                tool_messages,
                64,
                tools=tools,
                tool_choice="none",
                parallel_tool_calls=False,
            ),
            "expected": {"kind": "text", "value": "42"},
        },
        {
            "id": "cancellation",
            "control": True,
            "body": chat_body(
                model,
                long_messages,
                256,
                stream=True,
                stream_options={"include_usage": True},
            ),
        },
        {
            "id": "deadline",
            "control": True,
            "body": chat_body(
                model, messages("deadline", long_messages[0]["content"]), 256
            ),
        },
        {
            "id": "recovery_arithmetic",
            "body": chat_body(
                model,
                messages(
                    "recovery", "Calculate 90 - 48. Reply with only the integer answer."
                ),
                64,
            ),
            "expected": {"kind": "text", "value": "42"},
        },
    ]


class HTTPClient:
    def __init__(self, args):
        endpoint = urlsplit(args.base_url)
        self.host, self.port = endpoint.hostname, endpoint.port
        self.timeout = args.timeout
        self.model = args.model
        self.headers = {"Content-Type": "application/json"}
        if key := os.environ.get("SPLASH_API_KEY"):
            self.headers["Authorization"] = f"Bearer {key}"

    def send(self, method, path, body=None, *, disconnect=False):
        payload = None if body is None else json.dumps(body, allow_nan=False).encode()
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=self.timeout
        )
        began = time.monotonic()
        try:
            connection.request(method, path, payload, self.headers)
            response = connection.getresponse()
            headers_ms = (time.monotonic() - began) * 1000
            content_type = response.getheader("Content-Type", "")
            if (
                body is not None
                and body.get("stream")
                and response.status == 200
                and content_type.startswith("text/event-stream")
            ):
                lines, size = [], 0
                while True:
                    line = response.readline(MAX_RESPONSE_BYTES + 1)
                    if not line:
                        break
                    size += len(line)
                    if size > MAX_RESPONSE_BYTES:
                        raise ValueError(
                            "stream response exceeds qualification byte budget"
                        )
                    lines.append(((time.monotonic() - began) * 1000, line))
                    if (
                        disconnect
                        and line in (b"\n", b"\r\n")
                        and parse_sse(lines)["text"]
                    ):
                        break
                record = parse_sse(lines)
                record.update(
                    http_status=response.status,
                    content_type=content_type,
                    disconnected_after_content=disconnect and bool(record["text"]),
                )
                if disconnect and connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                response.close()
            else:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError("response exceeds qualification byte budget")
                record = normalize_response(response.status, content_type, raw)
            record.update(
                http_headers_ms=headers_ms,
                http_total_ms=(time.monotonic() - began) * 1000,
                client_started_monotonic=began,
                client_ended_monotonic=time.monotonic(),
                expected_model=self.model,
            )
            record["response_sha256"] = hashlib.sha256(
                json.dumps(
                    {"text": record["text"], "tool_calls": record["tool_calls"]},
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            if body is not None:
                record["request_body"] = copy.deepcopy(body)
            return record
        finally:
            connection.close()

    def status(self):
        record = self.send("GET", "/status")
        if record["http_status"] != 200 or not isinstance(record.get("response"), dict):
            raise ValueError("status request failed")
        return record["response"]


def maximum_overlap(records):
    events = []
    for record in records:
        events.extend(
            (
                (record["client_started_monotonic"], 1),
                (record["client_ended_monotonic"], -1),
            )
        )
    active, maximum = 0, 0
    for _, change in sorted(events):
        active += change
        maximum = max(maximum, active)
    return maximum


class Qualification:
    def __init__(self, args, document):
        self.args, self.document = args, document
        self.client = HTTPClient(args)
        self.initial = self.client.status()
        errors = validate_status(self.initial)
        if not idle(self.initial):
            errors.append("qualification requires an idle dedicated server")
        if get_path(self.initial, "instance.model") != args.model:
            errors.append("server model differs from requested model")
        if get_path(self.initial, "capabilities.native_route") != "flash-next":
            errors.append("server does not identify the native Flash-Next route")
        if errors:
            raise ValueError("; ".join(errors))
        self.identity = self.initial["identity"]
        self.cache_disabled = (
            get_path(self.initial, "capabilities.prefix_cache") is False
        )
        self.tool_message = None
        document["initial_status"] = self.initial
        document["controls_support"] = {
            "basis": "native Flash-Next mandatory protocol-v5 cancellation and per-request timeout contract; Ready bits are not exposed by HTTP",
            "negotiated_ready_bits_observed": False,
        }

    def wait_idle(self, *, after_snapshot=None, completed=None, cancelled=None):
        began = time.monotonic()
        while True:
            status = self.client.status()
            errors = validate_status(status, self.identity)
            if errors:
                raise ValueError("; ".join(errors))
            if get_path(status, "transport.restarts") != get_path(
                self.initial, "transport.restarts"
            ):
                raise ValueError("native worker restarted during qualification")
            snapshot = get_path(status, "status_snapshot.steady_seconds")
            fresh = after_snapshot is None or (
                isinstance(snapshot, (int, float)) and snapshot > after_snapshot
            )
            terminals = (
                completed is None or get_path(status, "requests.completed") >= completed
            ) and (
                cancelled is None or get_path(status, "requests.cancelled") >= cancelled
            )
            if idle(status) and fresh and terminals:
                return status
            if time.monotonic() - began > self.args.cleanup_timeout:
                raise ValueError(
                    "request cleanup did not reach idle within its timeout"
                )
            time.sleep(0.25)

    def run_case(self, plan):
        feature = plan.get("feature")
        if feature and get_path(self.initial, "capabilities." + feature) is not True:
            return {
                "id": plan["id"],
                "status": "skipped",
                "reason": f"{feature} is not advertised",
            }
        if plan.get("control") and self.args.skip_controls:
            return {
                "id": plan["id"],
                "status": "skipped",
                "reason": "controls explicitly disabled",
            }
        before = self.wait_idle()
        row = {"id": plan["id"], "status_before": before, "records": [], "errors": []}
        if plan["id"] == "concurrent_four":
            barrier = threading.Barrier(5)

            def lane_request(lane):
                barrier.wait(timeout=5)
                record = self.client.send("POST", "/v1/chat/completions", lane["body"])
                record["lane"] = lane["lane"]
                record["cache_disabled"] = self.cache_disabled
                record["checks"] = check_answer(
                    record, lane["expected"], lane["body"]["max_completion_tokens"]
                )
                return record

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(lane_request, lane) for lane in plan["lanes"]]
                barrier.wait(timeout=5)
                row["records"] = [
                    future.result(timeout=self.args.timeout + 5) for future in futures
                ]
            row["client_maximum_in_flight"] = maximum_overlap(row["records"])
            if row["client_maximum_in_flight"] != 4:
                row["errors"].append(
                    "four simultaneous client requests were not observed"
                )
            identifiers = [record.get("request_ids") for record in row["records"]]
            if (
                any(len(values) != 1 for values in identifiers)
                or len({values[0] for values in identifiers if values}) != 4
            ):
                row["errors"].append(
                    "concurrent requests do not have four distinct IDs"
                )
        else:
            body = copy.deepcopy(plan["body"])
            if plan["id"] == "tool_continuation":
                if self.tool_message is None:
                    return {
                        "id": plan["id"],
                        "status": "skipped",
                        "reason": "required tool call did not pass",
                    }
                call_id = self.tool_message["tool_calls"][0]["id"]
                body["messages"].extend(
                    (
                        copy.deepcopy(self.tool_message),
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": '{"sum":42}',
                        },
                        {
                            "role": "user",
                            "content": "Reply with only the integer total reported by the tool.",
                        },
                    )
                )
            if plan["id"] == "deadline":
                body["timeout"] = self.args.deadline_seconds
            record = self.client.send(
                "POST",
                "/v1/chat/completions",
                body,
                disconnect=plan["id"] == "cancellation",
            )
            record["cache_disabled"] = self.cache_disabled
            if "expected" in plan:
                record["checks"] = check_answer(
                    record, plan["expected"], body["max_completion_tokens"]
                )
            elif plan["id"] == "cancellation":
                record["checks"] = list(record["errors"])
                if record["http_status"] != 200 or not record.get(
                    "disconnected_after_content"
                ):
                    record["checks"].append(
                        "cancellation did not disconnect an active content stream"
                    )
                if (
                    record["done"]
                    or record["finish_reason"] is not None
                    or record["error_frames"]
                ):
                    record["checks"].append(
                        "stream terminated before the cancellation probe"
                    )
                if len(record["request_ids"]) != 1 or record["model_ids"] != [
                    self.args.model
                ]:
                    record["checks"].append(
                        "cancellation stream request/model identity is invalid"
                    )
            else:
                record["checks"] = list(record["errors"])
                error = get_path(record, "response.error.code")
                if record["http_status"] != 504 or error != "request_timeout":
                    record["checks"].append(
                        "deadline did not return HTTP504/request_timeout"
                    )
            if plan["id"] == "tool_call" and not record["checks"]:
                self.tool_message = copy.deepcopy(record["message"])
            row["records"] = [record]
        row["http_wave_wall_ms"] = (
            max(record["client_ended_monotonic"] for record in row["records"])
            - min(record["client_started_monotonic"] for record in row["records"])
        ) * 1000
        completed = (
            None
            if plan.get("control")
            else get_path(before, "requests.completed") + len(row["records"])
        )
        cancelled = (
            get_path(before, "requests.cancelled") + 1
            if plan["id"] == "cancellation"
            else None
        )
        after = self.wait_idle(
            after_snapshot=get_path(before, "status_snapshot.steady_seconds"),
            completed=completed,
            cancelled=cancelled,
        )
        row["status_after"] = after
        delta = row["native_counter_duration_delta"] = counter_delta(before, after)
        if any(value < 0 for value in delta.values()):
            row["errors"].append(
                "native counters or cumulative phase durations decreased"
            )
        for record in row["records"]:
            row["errors"].extend(record.get("checks", []))
        submitted = delta.get("requests.submitted")
        if plan["id"] == "cancellation":
            if submitted != 1 or delta.get("requests.cancelled") != 1:
                row["errors"].append(
                    "native cancellation counter did not confirm one cancelled request"
                )
            row["coverage_scope"] = (
                "disconnect after client-observed content, native cancellation acknowledgement and idle cleanup"
            )
        elif plan["id"] == "deadline":
            work = delta.get("scheduler.prefill_rows", 0) + delta.get(
                "scheduler.decode_batches", 0
            )
            row["native_request_observed"] = submitted == 1
            row["native_model_work_observed"] = submitted == 1 and work > 0
            row["coverage_scope"] = (
                "native model work began before timeout and terminal cleanup"
                if submitted == 1 and work > 0
                else "native transport/admission timeout; no model work observed"
                if submitted == 1
                else "frontend deadline before native request accounting; native deadline not exercised"
            )
            if submitted not in (0, 1):
                row["errors"].append("unexpected native request count during deadline")
            if (
                submitted == 1
                and delta.get("requests.cancelled", 0) + delta.get("requests.failed", 0)
                != 1
            ):
                row["errors"].append(
                    "admitted deadline request lacks a native terminal counter"
                )
        else:
            count = len(row["records"])
            if (
                submitted != count
                or delta.get("requests.completed") != count
                or delta.get("requests.cancelled") != 0
                or delta.get("requests.failed") != 0
            ):
                row["errors"].append(
                    "native request counters differ from isolated completed HTTP work"
                )
        if plan["id"] == "concurrent_four":
            row["native_batch_width_delta"] = {
                f"b{width}": delta.get(f"scheduler.decode_batches_by_width.b{width}")
                for width in range(1, 5)
            }
            row["native_scheduler_mode"] = get_path(after, "scheduler.mode")
            row["coverage_scope"] = (
                "four concurrent HTTP requests; native batch widths reported separately"
            )
        completions = [
            get_path(record, "usage.completion_tokens") for record in row["records"]
        ]
        if (
            all(type(value) is int for value in completions)
            and row["http_wave_wall_ms"] > 0
        ):
            row["observed_http_output_tokens_per_second"] = (
                sum(completions) * 1000 / row["http_wave_wall_ms"]
            )
        row["status"] = "failed" if row["errors"] else "passed"
        return row


def save_report(document, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--model", default="local/Qwen3.8-Flash-Next-oQ4e-mtp")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build/release/flash-next/http-qualification.json",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--cleanup-timeout", type=float, default=30)
    parser.add_argument("--deadline-seconds", type=float, default=0.2)
    parser.add_argument("--nonce", default=None)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write prompts/settings without contacting any server",
    )
    args = parser.parse_args(argv)
    try:
        endpoint = urlsplit(args.base_url)
        if (
            endpoint.scheme != "http"
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
            or endpoint.path not in ("", "/")
        ):
            raise ValueError("base URL must be a local HTTP origin")
        if (
            endpoint.hostname != "localhost"
            and not ipaddress.ip_address(endpoint.hostname).is_loopback
        ):
            raise ValueError("only loopback servers are supported")
        if (
            endpoint.port is None
            or not 1 <= endpoint.port <= 65535
            or endpoint.port == 8000
        ):
            raise ValueError("an explicit port other than 8000 is required")
        if not all(
            math.isfinite(value) and 0 < value <= 600
            for value in (args.timeout, args.cleanup_timeout, args.deadline_seconds)
        ):
            raise ValueError("timeouts must be finite and in (0,600]")
        if args.deadline_seconds >= args.timeout:
            raise ValueError("deadline must be shorter than the HTTP timeout")
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    args.nonce = args.nonce or uuid.uuid4().hex
    return args


def main(argv=None):
    args = parse_args(argv)
    document = {
        "schema_version": 1,
        "completed": False,
        "pass": False,
        "settings": {
            "base_url": args.base_url,
            "model": args.model,
            "nonce": args.nonce,
            "http_timeout_seconds": args.timeout,
            "cleanup_timeout_seconds": args.cleanup_timeout,
            "deadline_seconds": args.deadline_seconds,
            "skip_controls": args.skip_controls,
            "reasoning_effort": "none",
            "temperature": 0,
            "seed": 0,
        },
        "plan": build_plan(args.model, args.nonce),
        "cases": [],
        "timing_notes": [
            "Client HTTP timings use time.monotonic and include connect/send/read overhead.",
            "Client streaming TTFT is first nonempty content, excluding headers, role frames and keepalives.",
            "Native model_timing deltas are summed GPU/command/forward-host durations from the isolated case.",
            "Short stopped answers use actual output counts; these are qualification observations, not matched throughput benchmarks.",
            "Four HTTP requests do not imply a four-lane native GPU batch; scheduler mode and actual width counters are saved.",
            "Executable hashes and negotiated Ready bits are not exposed by HTTP; identities are the server-provided model, semantics and instance fields.",
        ],
    }
    save_report(document, args.output)
    if args.plan_only:
        print(f"Plan written without server requests: {args.output}")
        return 0
    try:
        qualification = Qualification(args, document)
        for plan in document["plan"]:
            print(f"{plan['id']}: starting", flush=True)
            try:
                row = qualification.run_case(plan)
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                http.client.HTTPException,
            ) as error:
                row = {
                    "id": plan["id"],
                    "status": "failed",
                    "errors": [f"{type(error).__name__}: {error}"],
                }
            document["cases"].append(row)
            save_report(document, args.output)
            print(
                f"{row['id']}: {row['status']} · {row.get('http_wave_wall_ms', 0):.1f}ms",
                flush=True,
            )
        document["final_status"] = qualification.wait_idle()
        document["completed"] = True
        document["coverage_complete"] = all(
            row["status"] == "passed" for row in document["cases"]
        )
        document["pass"] = document["coverage_complete"]
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        http.client.HTTPException,
    ) as error:
        document["fatal_error"] = f"{type(error).__name__}: {error}"
    save_report(document, args.output)
    print(f"Report: {args.output} · pass={document['pass']}", flush=True)
    return 0 if document["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

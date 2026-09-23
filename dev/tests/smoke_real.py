#!/usr/bin/env python3

"""Small real-model smoke test for the generic Splash HTTP frontend."""

from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from install import models as model_artifacts  # noqa: E402


class SmokeFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def request(
    port: int, method: str, path: str, body: dict | None = None, *, timeout: float = 60
):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    payload = None if body is None else json.dumps(body).encode()
    headers = {} if payload is None else {"Content-Type": "application/json"}
    try:
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        raw = response.read()
    finally:
        connection.close()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SmokeFailure(f"{method} {path} returned invalid JSON") from error
    return response.status, document


def stream_request(port: int, path: str, body: dict) -> tuple[int, str, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    payload = json.dumps(body).encode()
    try:
        connection.request("POST", path, payload, {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type", ""), response.read()
    finally:
        connection.close()


class RealServer:
    def __init__(self, arguments):
        package = arguments.package.resolve()
        binary = arguments.binary.resolve()
        self.port = available_port()
        self.log = tempfile.NamedTemporaryFile(
            mode="w+", prefix="splash-http-smoke-", suffix=".log"
        )
        command = [
            sys.executable,
            str(ROOT / "server/server.py"),
            str(package / "target"),
            str(package / "draft"),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--binary",
            str(binary),
            "--tokenizer",
            str(package / "tokenizer"),
            "--model",
            arguments.model,
        ]
        if arguments.max_context is not None:
            command.extend(("--max-context", str(arguments.max_context)))
        if arguments.max_memory is not None:
            command.extend(("--max-memory", arguments.max_memory))
        command.extend(("--kv-format", arguments.kv_format))
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def tail(self) -> str:
        self.log.flush()
        self.log.seek(0)
        return "".join(self.log.readlines()[-40:])

    def wait_ready(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise SmokeFailure(f"server exited during startup\n{self.tail()}")
            try:
                status, document = request(self.port, "GET", "/status")
                if status == 200 and document.get("ready") is True:
                    return document
            except (ConnectionError, OSError, SmokeFailure):
                pass
            time.sleep(0.25)
        raise SmokeFailure(f"server did not become ready\n{self.tail()}")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(10)
        self.log.close()


def kv_identity(identity: dict) -> dict:
    # Older INT8 builds expose only q8 and have no explicit format field.
    return {"format": "int8", **identity.get("kv", identity.get("q8", {}))}


def validate_status(status: dict, kv_format: str | None = None) -> None:
    require(status.get("ready") is True, "runtime is not ready")
    require(status.get("metal", {}).get("healthy") is True, "Metal is unhealthy")
    require(
        status.get("transport", {}).get("restarts") == 0,
        "native runtime restarted during the real-model gate",
    )
    require(
        status.get("identity", {}).get("cache", {}).get("block_tokens") == 32,
        "runtime did not expose Page32 KV identity",
    )
    identity = status.get("identity", {})
    kv = kv_identity(identity)
    actual = kv["format"]
    require(actual in ("int8", "bf16"), "unknown KV format")
    if kv_format is not None:
        require(actual == kv_format, "runtime KV format differs from requested format")
    quantization, scale_type = (
        ("symmetric_int8", "float32") if actual == "int8" else ("none", "none")
    )
    require(kv.get("quantization") == quantization, "wrong KV quantization")
    require(kv.get("scale_type") == scale_type, "wrong KV scale type")


def chat_body(model: str, prompt: str, **extra) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 32,
        "temperature": 0,
        "reasoning_effort": "none",
    }
    body.update(extra)
    return body


def image_data_url(kind: str) -> str:
    """Synthetic 256x256 PNGs: solid red, solid blue, or red-left/blue-right."""
    from PIL import Image

    image = Image.new("RGB", (256, 256), (245, 245, 245))
    if kind == "red":
        image.paste((220, 30, 30), (0, 0, 256, 256))
    elif kind == "blue":
        image.paste((30, 60, 220), (0, 0, 256, 256))
    else:
        image.paste((220, 30, 30), (0, 0, 128, 256))
        image.paste((30, 60, 220), (128, 0, 256, 256))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def image_chat_body(model: str, prompt: str, url: str, **extra) -> dict:
    body = chat_body(model, prompt, **extra)
    body["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]
    return body


def answer_text(chat: dict) -> str:
    return (
        chat.get("choices", [{}])[0].get("message", {}).get("content") or ""
    ).lower()


def run_images(port: int, model: str, nonce: str) -> None:
    question = f"What color is this image? Answer with one word. Request {nonce}."
    code, red = request(
        port,
        "POST",
        "/v1/chat/completions",
        image_chat_body(model, question, image_data_url("red")),
    )
    require(code == 200, f"image Chat failed with HTTP {code}: {red!r}")
    require("red" in answer_text(red), f"red image was not described as red: {red!r}")
    metrics = red.get("metrics", {})
    print(
        "image chat: PASS "
        f"(prompt_tokens={red.get('usage', {}).get('prompt_tokens')}, "
        f"ttft_ms={metrics.get('request_latency', {}).get('ttft_ms')}, "
        f"cache={metrics.get('cache', {}).get('status')})",
        flush=True,
    )

    # The same image and prompt reuse the image-aware prefix without running
    # the vision tower again; a different image behind identical placeholder
    # tokens must not reuse KV.
    code, before = request(port, "GET", "/status")
    require(code == 200 and "images" in before, "status lacks image telemetry")
    code, repeat = request(
        port,
        "POST",
        "/v1/chat/completions",
        image_chat_body(model, question, image_data_url("red")),
    )
    require(
        code == 200 and "red" in answer_text(repeat), "repeated image request failed"
    )
    repeat_cache = repeat.get("metrics", {}).get("cache", {})
    require(
        repeat_cache.get("status") == "hit"
        and repeat_cache.get("matched_tokens", 0) > 0,
        f"identical image did not reuse the prefix: {repeat_cache!r}",
    )
    code, after = request(port, "GET", "/status")
    require(
        code == 200
        and after["images"]["encodes"] == before["images"]["encodes"]
        and after["images"]["embedding_reuses"] > before["images"]["embedding_reuses"],
        f"repeated image re-ran the vision tower: {before['images']} -> {after['images']}",
    )
    code, blue = request(
        port,
        "POST",
        "/v1/chat/completions",
        image_chat_body(model, question, image_data_url("blue")),
    )
    require(code == 200, f"blue image Chat failed with HTTP {code}")
    require(
        "blue" in answer_text(blue), f"blue image was not described as blue: {blue!r}"
    )
    blue_cache = blue.get("metrics", {}).get("cache", {})
    require(
        blue_cache.get("matched_tokens", 0) < repeat_cache["matched_tokens"],
        f"a different image reused image KV: {blue_cache!r}",
    )
    print(
        "image prefix cache: PASS "
        f"(repeat matched={repeat_cache['matched_tokens']}, "
        f"different image matched={blue_cache.get('matched_tokens')})",
        flush=True,
    )

    layout = image_data_url("layout")
    media_type, _, data = layout.partition(";base64,")
    code, anthropic = request(
        port,
        "POST",
        "/v1/messages",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type.removeprefix("data:"),
                                "data": data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Which color fills the left half of this image, "
                            f"red or blue? Answer with one word. Request {nonce}.",
                        },
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
            "thinking": {"type": "disabled"},
        },
    )
    require(code == 200 and anthropic.get("type") == "message", "image Messages failed")
    text = "".join(
        block.get("text", "") for block in anthropic.get("content", [])
    ).lower()
    require(
        "red" in text and "blue" not in text.split("red")[0], f"layout answer: {text!r}"
    )
    print("anthropic image messages: PASS", flush=True)

    code, responses = request(
        port,
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {"type": "input_image", "image_url": image_data_url("blue")},
                    ],
                }
            ],
            # Responses requests reason by default; leave room for the
            # thinking block before the one-word answer.
            "max_output_tokens": 256,
            "temperature": 0,
            "store": False,
        },
    )
    require(
        code == 200 and responses.get("object") == "response", "image Responses failed"
    )
    message = next(
        (
            item
            for item in reversed(responses.get("output", []))
            if item.get("type") == "message" and item.get("role") == "assistant"
        ),
        {},
    )
    output_text = "".join(
        block.get("text", "")
        for block in message.get("content", [])
        if block.get("type") == "output_text"
    ).lower()
    require("blue" in output_text, f"Responses image answer: {output_text!r}")
    print("responses image input: PASS", flush=True)


def run(port: int, model: str) -> None:
    nonce = uuid.uuid4().hex
    code, models = request(port, "GET", "/v1/models")
    require(code == 200 and models.get("data"), "model discovery failed")

    code, chat = request(
        port,
        "POST",
        "/v1/chat/completions",
        chat_body(model, f"Reply with one short word. Request {nonce}."),
    )
    require(code == 200 and len(chat.get("choices", [])) == 1, "Chat failed")
    require(chat.get("usage", {}).get("prompt_tokens", 0) > 0, "Chat usage missing")
    print("chat completions: PASS", flush=True)

    code, content_type, payload = stream_request(
        port,
        "/v1/chat/completions",
        chat_body(
            model,
            f"Reply with one short word. Streaming request {nonce}.",
            stream=True,
            stream_options={"include_usage": True},
        ),
    )
    require(code == 200, f"streaming Chat failed with HTTP {code}")
    require(
        content_type.startswith("text/event-stream"),
        f"streaming Chat returned {content_type!r}",
    )
    events = [
        line[6:] for line in payload.decode().splitlines() if line.startswith("data: ")
    ]
    require(events and events[-1] == "[DONE]", "streaming Chat did not finish")
    chunks = [json.loads(event) for event in events[:-1]]
    text = "".join(
        choice.get("delta", {}).get("content", "")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    require(text, "streaming Chat emitted no assistant text")
    require(
        any(chunk.get("usage", {}).get("completion_tokens", 0) > 0 for chunk in chunks),
        "streaming Chat emitted no final usage",
    )
    print("chat streaming: PASS", flush=True)

    tool = {
        "type": "function",
        "function": {
            "name": "record_probe",
            "description": "Record the fixed smoke-test value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string", "const": "ok"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
    code, tool_response = request(
        port,
        "POST",
        "/v1/chat/completions",
        chat_body(
            model,
            f"Call record_probe for request {nonce}.",
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "record_probe"}},
            max_completion_tokens=96,
        ),
    )
    calls = (
        tool_response.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
    )
    require(code == 200 and len(calls) == 1, "generic tool call failed")
    require(calls[0].get("function", {}).get("name") == "record_probe", "wrong tool")
    require(
        json.loads(calls[0]["function"]["arguments"]) == {"value": "ok"},
        "wrong tool arguments",
    )
    print("tool protocol: PASS", flush=True)

    schema = {
        "type": "object",
        "properties": {"result": {"type": "string", "const": "ok"}},
        "required": ["result"],
        "additionalProperties": False,
    }
    code, structured = request(
        port,
        "POST",
        "/v1/chat/completions",
        chat_body(
            model,
            f"Return the required JSON for request {nonce}.",
            max_completion_tokens=64,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "probe", "strict": True, "schema": schema},
            },
        ),
    )
    content = structured.get("choices", [{}])[0].get("message", {}).get("content")
    require(code == 200 and json.loads(content) == {"result": "ok"}, "JSON failed")
    print("structured output: PASS", flush=True)

    code, responses = request(
        port,
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": f"Reply briefly. Request {nonce}.",
            "max_output_tokens": 32,
            "temperature": 0,
            "store": False,
        },
    )
    require(
        code == 200 and responses.get("object") == "response",
        f"Responses failed: code={code}, body={responses!r}",
    )
    print("responses: PASS", flush=True)

    code, anthropic = request(
        port,
        "POST",
        "/v1/messages",
        {
            "model": model,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: ignored"},
                {"type": "text", "text": "Answer briefly."},
            ],
            "messages": [
                {"role": "user", "content": f"Request {nonce}."},
                {"role": "assistant", "content": "Ready."},
                {"role": "system", "content": "Reply with only PONG."},
                {"role": "user", "content": "Go."},
            ],
            "max_tokens": 32,
            "temperature": 0,
            "thinking": {"type": "disabled"},
        },
    )
    require(code == 200 and anthropic.get("type") == "message", "Messages failed")
    require(isinstance(anthropic.get("content"), list), "Messages content missing")
    require(
        "PONG" in "".join(block.get("text", "") for block in anthropic["content"]),
        "Messages inline system instruction was not followed",
    )
    print("anthropic messages: PASS", flush=True)

    run_images(port, model, nonce)
    run_protocol_extensions(port, model)
    run_judgments(port, model, nonce)


def run_protocol_extensions(port: int, model: str) -> None:
    messages = [{"role": "user", "content": "What is 2 + 2? Answer briefly."}]
    for suffix in ("", "?beta=true"):
        code, count = request(
            port,
            "POST",
            "/v1/messages/count_tokens" + suffix,
            {"model": model, "messages": messages},
        )
        require(
            code == 200 and count.get("input_tokens", 0) > 0,
            f"Anthropic token counting failed: {count!r}",
        )
    code, hidden = request(
        port,
        "POST",
        "/v1/messages",
        {
            "model": model,
            "messages": messages,
            "max_tokens": 256,
            "thinking": {"type": "enabled", "display": "omitted"},
            "output_config": {"effort": "low"},
        },
    )
    require(code == 200, f"hidden thinking failed: {hidden!r}")
    blocks = [b for b in hidden["content"] if b["type"] == "thinking"]
    require(
        blocks and all(b["thinking"] == "" and b.get("signature") for b in blocks),
        "hidden thinking leaked text or omitted its continuation signature",
    )
    code, count = request(
        port,
        "POST",
        "/v1/messages/count_tokens?beta=true",
        {
            "model": model,
            "messages": [
                *messages,
                {"role": "assistant", "content": hidden["content"]},
                {"role": "user", "content": "Continue."},
            ],
        },
    )
    require(
        code == 200 and count.get("input_tokens", 0) > 0,
        "hidden-thinking continuation could not be counted",
    )
    schema = {
        "type": "object",
        "properties": {"answer": {"const": "yes"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    code, structured = request(
        port,
        "POST",
        "/v1/messages",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Answer yes in JSON."}],
            "max_tokens": 64,
            "thinking": {"type": "disabled"},
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        },
    )
    require(code == 200, f"Anthropic structured output failed: {structured!r}")
    text = "".join(b.get("text", "") for b in structured["content"])
    require(json.loads(text) == {"answer": "yes"}, "Anthropic schema was not enforced")
    code, nullable = request(
        port,
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "Reply OK.",
            "max_output_tokens": 16,
            "stream": None,
            "parallel_tool_calls": None,
            "temperature": None,
            "reasoning": {"effort": "none"},
            "store": False,
        },
    )
    require(
        code == 200 and nullable.get("object") == "response",
        f"nullable Responses parameters failed: {nullable!r}",
    )
    # Counting and generation must render the same prompt, including defaults.
    counted = {"model": model, "messages": messages, "thinking": {"type": "disabled"}}
    code, count = request(port, "POST", "/v1/messages/count_tokens", counted)
    require(code == 200, f"count parity setup failed: {count!r}")
    code, reply = request(port, "POST", "/v1/messages", {**counted, "max_tokens": 32})
    require(code == 200, f"count parity generation failed: {reply!r}")
    usage = reply["usage"]
    actual = sum(
        usage.get(key, 0)
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    )
    require(
        count["input_tokens"] == actual, f"count/usage mismatch: {count!r} vs {usage!r}"
    )

    pdf = base64.b64encode(
        (ROOT / "dev/tests/fixtures/documents/plain.pdf").read_bytes()
    ).decode()
    code, document = request(
        port,
        "POST",
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 64,
            "thinking": {"type": "disabled"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf,
                            },
                        },
                        {"type": "text", "text": "Briefly describe the page."},
                    ],
                }
            ],
        },
    )
    require(
        code == 200 and document.get("content"), f"PDF generation failed: {document!r}"
    )

    code, combined = request(
        port,
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 64,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "Call lookup."}],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            },
        },
    )
    require(code == 200, f"tools plus response schema failed: {combined!r}")
    calls = combined["choices"][0]["message"].get("tool_calls", [])
    require(
        len(calls) == 1 and calls[0]["function"]["name"] == "lookup",
        "combined tools/schema lost the required call",
    )
    require(
        json.loads(calls[0]["function"]["arguments"]) == {},
        "combined tools/schema changed tool arguments",
    )

    code, high = request(
        port,
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": 256,
            "reasoning_effort": "high",
        },
    )
    require(code == 200 and high["choices"], f"explicit high effort failed: {high!r}")

    code, foreign = request(
        port,
        "POST",
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 32,
            "thinking": {"type": "disabled"},
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            "messages": [
                *messages,
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Adding two and two gives four.",
                            "signature": "opaque-provider-signature",
                        },
                        {"type": "text", "text": "4"},
                    ],
                },
                {"role": "user", "content": "Repeat your answer."},
            ],
        },
    )
    require(
        code == 200 and foreign.get("content"),
        f"visible external thinking continuation failed: {foreign!r}",
    )
    print(
        "Count/usage parity, PDF, tools+schema, high effort and external thinking history: PASS",
        flush=True,
    )
    print(
        "Anthropic count/beta, hidden-thinking continuation, structured output and nullable Responses: PASS",
        flush=True,
    )


JUDGMENT_QUESTION = "What is the approval status of the proposal?"
JUDGMENT_OPTIONS = [
    {"id": "approved", "description": "Approval was explicitly given."},
    {"id": "pending", "description": "Approval has not been given yet."},
    {"id": "rejected", "description": "The proposal was turned down."},
]


def judgment_body(model: str, state: str, **extra) -> dict:
    body = {
        "id": "smoke",
        "model": model,
        "state": state,
        "question": JUDGMENT_QUESTION,
        "options": JUDGMENT_OPTIONS,
    }
    body.update(extra)
    return body


def filler(nonce: str, label: str, items: int) -> str:
    """Distinct bulk evidence: the label recurs in every item, so two fillers
    share no reusable prefix."""
    return " ".join(
        f"Appendix {label} item {index} of request {nonce} records routine "
        "correspondence that has no bearing on the decision."
        for index in range(items)
    )


def runtime_status(port: int) -> dict:
    code, status = request(port, "GET", "/status")
    require(code == 200, f"status read failed with HTTP {code}")
    return status


def counters(port: int) -> dict:
    status = runtime_status(port)
    return {
        "submitted": status["requests"]["submitted"],
        "cancelled": status["requests"]["cancelled"],
        "failed": status["requests"]["failed"],
        "reused_tokens": status["cache"]["reused_tokens"],
        "restarts": status["transport"]["restarts"],
    }


def wait_for_timeout_cleanup(port: int, before: dict) -> None:
    limit = time.monotonic() + 120
    while True:
        after = runtime_status(port)
        require(
            after["instance"] == before["instance"]
            and after["transport"]["restarts"] == 0
            and after["ready"]
            and after["metal"]["healthy"],
            "scoring timeout destabilized the runtime",
        )
        delta = {
            key: after["requests"][key] - before["requests"][key]
            for key in ("submitted", "completed", "cancelled", "failed")
        }
        require(
            delta["submitted"] == 1 and delta["completed"] == 0,
            f"unexpected scoring timeout outcome: {delta!r}",
        )
        terminal = (delta["cancelled"], delta["failed"])
        require(
            terminal in ((0, 0), (1, 0), (0, 1)),
            f"score did not terminate exactly once: {delta!r}",
        )
        # Frontend cancellation or the native deadline can finish first.
        if terminal != (0, 0) and after["transport"]["pending"] == 0:
            return
        require(time.monotonic() < limit, "scoring timeout cleanup did not finish")
        time.sleep(0.5)


def run_judgments(port: int, model: str, nonce: str) -> None:
    approved = (
        "Minutes of the review board: the board voted unanimously to approve "
        f"the proposal and entered the approval in the register. Request {nonce}."
    )
    code, judgment = request(
        port, "POST", "/v1/judgments", judgment_body(model, approved)
    )
    require(code == 200, f"judgment failed with HTTP {code}: {judgment!r}")
    ids, logits = judgment["option_ids"], judgment["option_logits"]
    probabilities = judgment["probabilities"]
    require(
        len(ids) == len(logits) == len(probabilities) == len(JUDGMENT_OPTIONS),
        f"judgment returned a ragged score: {judgment!r}",
    )
    require(
        abs(sum(probabilities) - 1.0) < 1e-6,
        f"judgment probabilities are not normalized: {probabilities!r}",
    )
    best = ids[max(range(len(ids)), key=probabilities.__getitem__)]
    require(
        best == "approved",
        f"judgment scored the wrong option: {dict(zip(ids, probabilities))}",
    )
    require(
        len(judgment["prompt_sha256"]) == 64
        and len(judgment["answer_token_ids"]) == len(ids)
        and judgment["prompt_version"] == "direct-options-v1",
        f"judgment provenance is incomplete: {judgment!r}",
    )
    usage = judgment["usage"]
    require(
        usage["completion_tokens"] == 0 and usage["prompt_tokens"] > 0,
        f"scoring generated tokens: {usage!r}",
    )
    print("judgments scoring: PASS", flush=True)

    code, systemone = request(
        port,
        "POST",
        "/v1/systemone",
        {
            "model": model,
            "state": {"message": f"I was charged twice. Fix this today. {nonce}"},
            "questions": {
                "billing": {
                    "type": "noul",
                    "instructions": "Is this message about billing?",
                },
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {"billing": None, "technical": None, "sales": None},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is the request?",
                    "criteria": ["No urgency", "This week", "Today"],
                },
                "sole": {"type": "choice", "criteria": {"only": None}},
            },
        },
    )
    require(code == 200, f"System One failed with HTTP {code}: {systemone!r}")
    answers = systemone["answers"]
    require(
        answers["billing"]["noul"] > 0.5,
        f"System One missed the billing topic: {answers['billing']!r}",
    )
    require(
        answers["department"]["choice"] == "billing",
        f"System One routed to the wrong team: {answers['department']!r}",
    )
    require(
        answers["urgency"]["score"] >= 1.0,
        f"System One under-scored an urgent request: {answers['urgency']!r}",
    )
    require(
        answers["sole"]["choice"] == "only" and answers["sole"]["confidence"] == 1.0,
        f"a singleton domain did not answer itself: {answers['sole']!r}",
    )
    require(
        systemone["usage"]["output_tokens"] == 0,
        f"System One generated tokens: {systemone['usage']!r}",
    )
    print("system one answers: PASS", flush=True)

    shared = judgment_body(model, f"{approved} {filler(nonce, 'reuse', 200)}")
    code, cold = request(port, "POST", "/v1/judgments", shared, timeout=300)
    require(code == 200, f"cold scored prompt failed with HTTP {code}: {cold!r}")
    before = counters(port)
    code, warm = request(port, "POST", "/v1/judgments", shared, timeout=300)
    require(code == 200, f"repeated scored prompt failed with HTTP {code}: {warm!r}")
    after = counters(port)
    prompt_tokens = warm["usage"]["prompt_tokens"]
    reused = after["reused_tokens"] - before["reused_tokens"]
    require(
        cold["usage"]["prompt_tokens"] == prompt_tokens,
        "the repeated scored prompt changed length",
    )
    # The tail below the last reusable block boundary is always recomputed.
    require(
        reused * 10 >= prompt_tokens * 9,
        f"scoring reused only {reused} of {prompt_tokens} prompt tokens",
    )
    print(f"judgments cache reuse: PASS ({reused}/{prompt_tokens})", flush=True)

    outcomes = {}
    quiet = counters(port)

    def score() -> None:
        outcomes["score"] = request(
            port,
            "POST",
            "/v1/judgments",
            judgment_body(model, f"{approved} Concurrent."),
            timeout=300,
        )

    def chat() -> None:
        outcomes["chat"] = request(
            port,
            "POST",
            "/v1/chat/completions",
            chat_body(model, f"Reply with one short word. Concurrent {nonce}."),
            timeout=300,
        )

    workers = [threading.Thread(target=score), threading.Thread(target=chat)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(300)
    require(len(outcomes) == 2, "a concurrent scoring/chat request did not finish")
    score_code, scored = outcomes["score"]
    chat_code, chatted = outcomes["chat"]
    require(score_code == 200, f"concurrent scoring failed: {scored!r}")
    require(chat_code == 200, f"concurrent chat failed: {chatted!r}")
    require(
        scored["usage"]["completion_tokens"] == 0,
        f"concurrent scoring generated tokens: {scored['usage']!r}",
    )
    require(
        chatted["usage"]["completion_tokens"] > 0,
        f"concurrent chat generated nothing: {chatted['usage']!r}",
    )
    mixed = counters(port)
    require(
        mixed["failed"] == quiet["failed"] and mixed["restarts"] == 0,
        f"mixed scoring and chat destabilized the runtime: {quiet!r} -> {mixed!r}",
    )
    print("mixed scoring and chat: PASS", flush=True)

    code, long_score = request(
        port,
        "POST",
        "/v1/judgments",
        judgment_body(model, filler(nonce, "warm", 250)),
        timeout=300,
    )
    require(code == 200, f"long scored prompt failed with HTTP {code}: {long_score!r}")
    forward = long_score["forward_seconds"]
    require(forward > 0.5, f"prefill is too fast to cancel: {forward:.2f}s")
    before = runtime_status(port)
    started = time.monotonic()
    code, timed_out = request(
        port,
        "POST",
        "/v1/judgments",
        judgment_body(model, filler(nonce, "stop", 250), timeout=forward / 3),
        timeout=300,
    )
    elapsed = time.monotonic() - started
    require(code == 504, f"expected a scoring timeout, got HTTP {code}: {timed_out!r}")
    require(
        elapsed < forward,
        f"the deadline did not cut prefill short: {elapsed:.2f}s of {forward:.2f}s",
    )
    require(
        timed_out.get("error", {}).get("code") == "request_timeout",
        f"504 did not identify a request timeout: {timed_out!r}",
    )
    wait_for_timeout_cleanup(port, before)
    code, recovered = request(
        port, "POST", "/v1/judgments", judgment_body(model, approved)
    )
    require(
        code == 200 and recovered["usage"]["completion_tokens"] == 0,
        f"scoring did not recover after timeout: {recovered!r}",
    )
    after = runtime_status(port)
    require(
        after["instance"] == before["instance"] and after["transport"]["restarts"] == 0,
        "scoring recovery replaced the runtime",
    )
    print(f"judgments timeout recovery: PASS (504 after {elapsed:.2f}s)", flush=True)


def add_server_arguments(parser):
    parser.add_argument("--binary", type=Path, default=ROOT / "build/splash")
    parser.add_argument(
        "--package",
        type=Path,
        help="installed model package root (target, draft and tokenizer)",
    )
    parser.add_argument("--model", type=model_artifacts.parse_repo_id, required=True)
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--max-memory")
    parser.add_argument("--kv-format", choices=("int8", "bf16"), default="int8")
    parser.add_argument("--startup-timeout", type=float, default=1800)


def resolve_server_arguments(arguments):
    if arguments.package is None:
        arguments.package = model_artifacts.installed_root(
            model_artifacts.MODELS, arguments.model
        )
    return arguments


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_server_arguments(parser)
    return resolve_server_arguments(parser.parse_args(argv))


def main(argv=None) -> int:
    arguments = parse_args(argv)
    server = RealServer(arguments)
    try:
        validate_status(
            server.wait_ready(arguments.startup_timeout), arguments.kv_format
        )
        run(server.port, arguments.model)
        validate_status(request(server.port, "GET", "/status")[1], arguments.kv_format)
        print("http smoke: PASS", flush=True)
        return 0
    except Exception:
        print(server.tail(), file=sys.stderr)
        raise
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(main())

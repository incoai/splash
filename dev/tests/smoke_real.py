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
        if arguments.max_cache_disk is not None:
            command.extend(("--max-cache-disk", arguments.max_cache_disk))
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


def validate_status(status: dict) -> None:
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
    q8 = status.get("identity", {}).get("q8", {})
    require(q8.get("quantization") == "symmetric_int8", "wrong KV quantization")
    require(q8.get("scale_type") == "float32", "wrong KV scale type")


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
    parser.add_argument("--max-cache-disk")
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
        validate_status(server.wait_ready(arguments.startup_timeout))
        run(server.port, arguments.model)
        validate_status(request(server.port, "GET", "/status")[1])
        print("http smoke: PASS", flush=True)
        return 0
    except Exception:
        print(server.tail(), file=sys.stderr)
        raise
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""HTTP routes, protocol responses and serving-process startup."""

import argparse
import json
import os
import queue
import re
import secrets
import selectors
import signal
import socket
import sys
import threading
import time
import weakref
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from huggingface_hub.utils import validate_repo_id
from transformers import AutoTokenizer

if __package__:
    from . import images as image_input
    from . import json_codec, judgments
    from . import runtime as engine_runtime
    from .api_shapes import (
        anthropic_response,
        anthropic_stop,
        anthropic_to_chat_body,
        anthropic_to_chat_prompt,
        anthropic_usage,
        completion_response,
        finish_reason,
        responses_item,
        responses_output,
        responses_response,
        stream_chunk,
    )
    from .backend import NativeBackend, remaining_request_time
    from .constraints import ConstraintFactory, validate_tokenizer
    from .diagnostics import log_unexpected, print_request, print_status
    from .errors import APIError, ContextLengthError
    from .frontend import REASONING_EFFORTS, Frontend, validate_served_model_name
    from .http_security import authenticate, validate_api_key, validate_headers
    from .latency import RequestLatency
    from .metrics import (
        is_finite_number,
        metrics_dict,
        prometheus_metrics,
        usage_dict,
    )
    from .output import (
        ReasoningSplitter,
        StreamingToolCallProjector,
        argument_deltas,
        parse_tool_calls,
        validate_response_content,
        validate_tool_calls,
    )
    from .thinking import ThinkingCodec, ThinkingKeyError, load_thinking_key
else:
    import images as image_input
    import json_codec
    import judgments
    from api_shapes import (
        anthropic_response,
        anthropic_stop,
        anthropic_to_chat_body,
        anthropic_to_chat_prompt,
        anthropic_usage,
        completion_response,
        finish_reason,
        responses_item,
        responses_output,
        responses_response,
        stream_chunk,
    )
    from backend import NativeBackend, remaining_request_time
    from constraints import ConstraintFactory, validate_tokenizer
    from diagnostics import log_unexpected, print_request, print_status
    from errors import APIError, ContextLengthError
    from frontend import REASONING_EFFORTS, Frontend, validate_served_model_name
    from http_security import authenticate, validate_api_key, validate_headers
    from latency import RequestLatency
    from metrics import is_finite_number, metrics_dict, prometheus_metrics, usage_dict
    from output import (
        ReasoningSplitter,
        StreamingToolCallProjector,
        argument_deltas,
        parse_tool_calls,
        validate_response_content,
        validate_tool_calls,
    )
    from thinking import ThinkingCodec, ThinkingKeyError, load_thinking_key

    import runtime as engine_runtime


DEFAULT_MAX_REQUEST_BYTES = 128 * 1024 * 1024
# Match the former generation ingress envelope (32 slots × 16 MiB).
DEFAULT_REQUEST_BODY_BUDGET = 512 * 1024 * 1024
MAX_CONTEXT_TOKENS = 262144
HTTP_IO_TIMEOUT = 30.0
HTTP_UPLOAD_BYTES_PER_SECOND = 512 * 1024
CLIENT_DISCONNECT_POLL = 0.01
SSE_KEEPALIVE_SECONDS = 2.0
NATIVE_START_TIMEOUT = 600.0
ROOT = Path(__file__).parents[1]
CHAT_HTML = Path(__file__).with_name("chat.html").read_bytes()


def _normalize_path(raw_path):
    """Canonicalize a request target for route and header decisions.

    Strips any query string or fragment, percent-decodes, and resolves
    ``.`` and ``..`` segments so encoded or dotted spellings of a route
    are treated exactly like the route itself.
    """
    decoded = unquote(raw_path.partition("?")[0].partition("#")[0])
    if not decoded.startswith("/"):
        return decoded
    trailing = decoded.endswith("/") and len(decoded) > 1
    segments = []
    for segment in decoded.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    normalized = "/" + "/".join(segments)
    if trailing and normalized != "/":
        normalized += "/"
    return normalized


class FrontendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        self._response_started = False
        self._last_sse_write = time.monotonic()
        super().setup()
        self.connection.settimeout(self.server.io_timeout)
        self._header_timer = threading.Timer(
            self.server.io_timeout, self._expire_headers
        )
        self._header_timer.daemon = True
        self._header_timer.start()

    def _expire_headers(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def finish(self):
        self._header_timer.cancel()
        super().finish()

    def log_message(self, format, *args):
        pass

    def parse_request(self):
        try:
            parsed = super().parse_request()
        finally:
            self._header_timer.cancel()
        if not parsed:
            return False
        if self.request_version not in {"HTTP/1.0", "HTTP/1.1"}:
            self.close_connection = True
            self.send_error(505, "HTTP version not supported")
            return False
        try:
            allowed_hosts = self.server.allowed_hosts | {
                self.connection.getsockname()[0].lower()
            }
            validate_headers(self.headers, allowed_hosts)
            path = self.path.partition("?")[0]
            public = self.command == "OPTIONS" or (
                self.command in ("GET", "HEAD")
                and path in ("/", "/index.html", "/health", "/ready")
            )
            if not public:
                authenticate(self.headers, self.server.api_key)
        except APIError as error:
            self.close_connection = True
            self._safe_error(
                error, self.path.partition("?")[0].startswith("/v1/messages"), log=False
            )
            return False
        return True

    @property
    def app(self):
        return self.server.app

    def _send(self, status, data, content_type):
        self.send_response(status)
        if status == 503:
            self.send_header("Retry-After", "1")
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        path = _normalize_path(getattr(self, "path", ""))
        if (
            path == "/v1/systemone"
            or path == "/v1/models"
            or path.startswith("/v1/models/")
        ):
            self.send_header("x-typesafe-request-id", f"req_{secrets.token_hex(12)}")
        self._response_started = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, status, payload):
        try:
            data = json_codec.encode(payload)
        except json_codec.JSONEncodingError as error:
            log_unexpected(error)
            self._error(
                APIError(500, "internal server error", "internal_server_error"),
                self.path.partition("?")[0].startswith("/v1/messages"),
            )
            return
        self._send(status, data, "application/json")

    def _error(self, error, anthropic=False):
        error_type = error.protocol_type(anthropic)
        message = error.message
        if anthropic and isinstance(error, ContextLengthError):
            message = (
                f"prompt is too long: {error.input_tokens} tokens > "
                f"{error.maximum_input_tokens} maximum input tokens"
            )
            if error.image_tokens_only:
                message += " (image tokens alone; text not yet counted)"
        self._json(
            error.status,
            {"type": "error", "error": {"type": error_type, "message": message}}
            if anthropic
            else {
                "error": {
                    "message": error.message,
                    "type": error_type,
                    "code": error.code,
                }
            },
        )

    def _log_api_error(self, error):
        path = self.path.partition("?")[0].partition("#")[0]
        path = "".join(char if char.isprintable() else "?" for char in path)
        print_status(f"Error · {error.code} · {self.command} {path[:256]}", error=True)

    def _safe_error(self, error, anthropic=False, *, log=True):
        if self._response_started:
            return
        if log:
            self._log_api_error(error)
        try:
            self._error(error, anthropic)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def _read_json_body(self, deadline):
        if self.headers.get_all("Transfer-Encoding"):
            raise APIError(400, "transfer encoding is not supported")
        encodings = self.headers.get_all("Content-Encoding", [])
        if len(encodings) > 1 or (
            encodings and encodings[0].strip().lower() != "identity"
        ):
            raise APIError(415, "content encoding is not supported")
        content_types = self.headers.get_all("Content-Type", [])
        content_type = self.headers.get_content_type().lower()
        if len(content_types) != 1 or not (
            content_type == "application/json"
            or (
                content_type.startswith("application/")
                and content_type.endswith("+json")
            )
        ):
            raise APIError(415, "Content-Type must be application/json")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            raise APIError(400, "exactly one Content-Length header is required")
        if not lengths[0].isascii() or not lengths[0].isdigit():
            raise APIError(400, "invalid Content-Length header")
        length = int(lengths[0])
        if length <= 0:
            raise APIError(400, "request body must not be empty")
        if length > self.server.max_request_bytes:
            raise APIError(
                413,
                f"request body is {length} bytes; limit is "
                f"{self.server.max_request_bytes} bytes (--max-request-size)",
                "request_too_large",
            )
        # Bound total upload time even when a client keeps the socket active.
        deadline = min(
            deadline,
            time.monotonic()
            + self.server.io_timeout
            + length / HTTP_UPLOAD_BYTES_PER_SECOND,
        )
        self._body_reservation = RequestBodyReservation(
            self.server.request_bodies, length
        )
        payload = bytearray()
        try:
            while len(payload) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                self.connection.settimeout(min(remaining, self.server.io_timeout))
                chunk = self.rfile.read1(min(65536, length - len(payload)))
                if not chunk:
                    raise APIError(400, "request body ended before Content-Length")
                payload.extend(chunk)
        finally:
            self.connection.settimeout(self.server.io_timeout)
        text = payload.decode(json.detect_encoding(payload), "surrogatepass")
        payload.clear()
        return json_codec.loads(text)

    def do_HEAD(self):
        self.do_GET()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Allow", "GET, HEAD, POST, DELETE, OPTIONS")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def version_string(self):
        return "Splash"

    def do_GET(self):
        path = self.path.partition("?")[0]
        if path in ("/", "/index.html"):
            if self.server.webui:
                self._send(200, CHAT_HTML, "text/html; charset=utf-8")
            else:
                self._safe_error(APIError(404, "not found", "not_found"))
            return
        if path == "/health":
            self._json(200, {"status": "ok"})
            return
        if path == "/ready":
            ready = self.app.backend.is_ready()
            self._json(
                200 if ready else 503,
                {"status": "ready" if ready else "unavailable"},
            )
            return
        if path == "/status":
            self._json(200, self.server.status())
            return
        if path == "/metrics":
            self._send(
                200,
                prometheus_metrics(self.server.status()).encode(),
                "text/plain; version=0.0.4; charset=utf-8",
            )
            return
        response_match = re.fullmatch(r"/v1/responses/(resp_[A-Za-z0-9_]+)", path)
        if response_match:
            stored = self.app.response_store.get(response_match.group(1))
            if stored is None:
                self._safe_error(APIError(404, "response not found", "not_found_error"))
            else:
                self._json(200, stored.response)
            return
        model_path = _normalize_path(self.path)
        if model_path == "/v1/models" or model_path.startswith("/v1/models/"):
            models = [
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "splash",
                    "max_model_len": self.app.max_context,
                    **({"root": self.app.model} if name != self.app.model else {}),
                }
                for name in self.app.model_names
            ]
            if model_path == "/v1/models":
                # TypeSafe SDK compatibility: models.list() reads "models" entries.
                typed = [
                    {
                        "name": item["id"],
                        "description": "Splash resident model",
                        "release_date": "",
                    }
                    for item in models
                ]
                self._json(200, {"object": "list", "data": models, "models": typed})
            else:
                name = model_path.removeprefix("/v1/models/")
                model = next((item for item in models if item["id"] == name), None)
                if model is None:
                    self._safe_error(
                        APIError(404, "model not found", "model_not_found")
                    )
                else:
                    self._json(200, model)
            return
        self._safe_error(APIError(404, "not found", "not_found"))

    def do_DELETE(self):
        path = self.path.partition("?")[0]
        response_match = re.fullmatch(r"/v1/responses/(resp_[A-Za-z0-9_]+)", path)
        if response_match is None:
            self._safe_error(APIError(404, "not found", "not_found"))
            return
        response_id = response_match.group(1)
        if not self.app.response_store.delete(response_id):
            self._safe_error(APIError(404, "response not found", "not_found_error"))
            return
        self._json(
            200,
            {"id": response_id, "object": "response", "deleted": True},
        )

    def do_POST(self):
        started_at = time.monotonic()
        job = None
        body = None
        self._body_reservation = None
        submitted = False
        # Route on the URL path so standard protocol query parameters do not
        # turn a supported endpoint into an unknown one.
        path = self.path.partition("?")[0]
        count_tokens = path == "/v1/messages/count_tokens"
        prompt_only = count_tokens or path in ("/tokenize", "/apply-template")
        anthropic = path == "/v1/messages" or count_tokens
        systemone = path == "/v1/systemone"
        if path not in (
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/tokenize",
            "/apply-template",
            "/v1/judgments",
            "/v1/systemone",
        ):
            self._safe_error(APIError(404, "not found", "not_found"))
            return
        if not prompt_only and not self.app.backend.can_submit():
            self._safe_error(
                APIError(
                    529 if systemone else 503,
                    "engine is recovering; retry shortly",
                    "engine_recovering",
                ),
                anthropic,
                log=False,
            )
            return
        # Hold one ingress slot through body parsing, preparation, and the
        # complete response. Slow uploads/readers cannot accumulate outside
        # the native pending limit, and control endpoints need no such slot.
        admission = self.server.token_counts if prompt_only else self.server.requests
        if not admission.acquire():
            self._safe_error(
                APIError(
                    529 if systemone else 503,
                    "frontend request capacity is exhausted",
                    "frontend_overloaded",
                ),
                anthropic,
            )
            return
        try:
            with self.app.latencies.measure("upload"):
                body = self._read_json_body(started_at + self.app.request_timeout)
            if not isinstance(body, dict):
                if systemone:
                    raise judgments.SystemOneError(
                        [judgments.detail([], "request body must be an object")]
                    )
                raise APIError(400, "request body must be an object")
            try:
                deadline = self.app.request_deadline(body, started_at)
            except APIError as error:
                if systemone:
                    raise judgments.SystemOneError(
                        [judgments.detail(["timeout"], error.message)]
                    ) from error
                raise
            if path == "/tokenize":
                self._json(200, {"tokens": self.app.tokenize(body, deadline=deadline)})
                return
            if path == "/apply-template":
                self._json(
                    200, {"prompt": self.app.apply_template(body, deadline=deadline)}
                )
                return
            if count_tokens:
                tokens = self.app.count_tokens(
                    anthropic_to_chat_prompt(
                        body,
                        deadline=deadline,
                        thinking_resolver=self.app.thinking_codec.decode,
                    ),
                    deadline=deadline,
                )
                self._json(200, {"input_tokens": tokens})
                return
            if path == "/v1/judgments":
                job, row = self.app.prepare_judgment(body, deadline=deadline)
                remaining_request_time(deadline)
                if self._client_disconnected():
                    raise ConnectionResetError("client disconnected before submission")
                if not self.app.backend.submit(job):
                    raise APIError(429, "request queue is full", "rate_limit_exceeded")
                submitted = True
                self._judgment_complete(job, row)
                return
            if systemone:
                self._systemone(body, deadline)
                return
            responses = path == "/v1/responses"
            stream = body.get("stream", False)
            if stream is None and not anthropic:
                stream = False
            return_progress = body.get("return_progress", False)
            if not isinstance(return_progress, bool) or (
                return_progress and stream is not True
            ):
                raise APIError(
                    400, "return_progress requires stream: true and must be a boolean"
                )
            if anthropic:
                job, thinking, has_tools = self.app.prepare(
                    anthropic_to_chat_body(
                        body,
                        deadline=deadline,
                        thinking_resolver=self.app.thinking_codec.decode,
                    ),
                    deadline=deadline,
                    clamp_output_budget=True,
                )
                stream_options = None
            elif responses:
                job, thinking, has_tools = self.app.prepare_responses(
                    body,
                    deadline=deadline,
                    reserve_input=self._body_reservation.grow,
                )
                stream_options = None
            else:
                stream_options = body.get("stream_options")
                if stream_options is None:
                    stream_options = {}
                if (
                    not isinstance(stream, bool)
                    or not isinstance(stream_options, dict)
                    or not isinstance(stream_options.get("include_usage", False), bool)
                ):
                    raise APIError(400, "invalid streaming options")
                stream_options = {
                    "include_usage": stream_options.get("include_usage", False)
                }
                job, thinking, has_tools = self.app.prepare(body, deadline=deadline)
            body = None
            self._body_reservation.retain_for(job)
            self._body_reservation = None
            job.return_progress = return_progress
            job.latency = RequestLatency(self.app.latencies, started_at)
            remaining_request_time(deadline)
            if self._client_disconnected():
                raise ConnectionResetError("client disconnected before submission")
            if not self.app.backend.submit(job):
                raise APIError(429, "request queue is full", "rate_limit_exceeded")
            submitted = True
            if anthropic and stream:
                self._anthropic_stream(job, thinking, has_tools)
            elif anthropic:
                self._anthropic_complete(job, thinking, has_tools)
            elif responses and stream:
                self._responses_stream(job, thinking, has_tools)
            elif responses:
                self._responses_complete(job, thinking, has_tools)
            elif stream:
                self._stream(job, thinking, has_tools, stream_options)
            else:
                self._complete(job, thinking, has_tools)
        except judgments.SystemOneError as error:
            if submitted:
                self.app.backend.cancel(job)
            self._systemone_error(error)
        except (BrokenPipeError, ConnectionResetError):
            if submitted:
                self.app.backend.cancel(job)
        except TimeoutError:
            if submitted:
                self.app.backend.cancel(job)
            error = APIError(408, "HTTP I/O timed out", "request_timeout")
            self._safe_error(error, anthropic, log=not submitted)
        except APIError as error:
            if submitted:
                self.app.backend.cancel(job)
            if systemone and error.status == 503:
                error = APIError(529, error.message, error.code)
            # The native outcome was already logged; a server-side failure
            # after submission must still reach the console.
            self._safe_error(error, anthropic, log=not submitted or error.status >= 500)
        except (ValueError, RecursionError):
            if submitted:
                self.app.backend.cancel(job)
            if systemone:
                self._systemone_error(
                    judgments.SystemOneError(
                        [judgments.detail([], "invalid JSON request body")]
                    )
                )
            else:
                error = APIError(400, "invalid JSON request body")
                self._safe_error(error, anthropic, log=not submitted)
        except Exception as error:
            if submitted:
                self.app.backend.cancel(job)
            log_unexpected(error)
            error = APIError(500, "internal server error", "internal_server_error")
            self._safe_error(error, anthropic, log=False)
        finally:
            body = None
            if self._body_reservation is not None:
                self._body_reservation.release()
                self._body_reservation = None
            admission.release()
            self.app.latencies.observe("http_request", time.monotonic() - started_at)

    def _judgment_complete(self, job, row):
        result = None
        while result is None:
            kind, value = self._next_event(job)
            if kind == "done":
                result = value
        if result.reason == "cancelled":
            if job.timed_out:
                raise APIError(504, "request timed out", "request_timeout")
            raise APIError(500, "request cancelled", "request_cancelled")
        if result.reason != "stop" or len(result.option_logits) != len(
            job.score_tokens
        ):
            raise APIError(500, "runtime protocol error", "protocol_error")
        self._json(
            200,
            judgments.judgment_response(self.app.model, row, job.meta, result),
        )

    def _systemone(self, body, deadline):
        active_job = None
        try:
            entries = self.app.prepare_systemone(body, deadline=deadline)
            remaining_request_time(deadline)
            if self._client_disconnected():
                raise ConnectionResetError("client disconnected before submission")
            answers = {}
            input_tokens = 0
            for qid, spec, job in entries:
                if job is None:
                    answers[qid] = judgments.deterministic_answer(spec)
                    continue
                # One admitted job per HTTP request preserves the existing
                # queue bound and lets later questions reuse the state prefix.
                active_job = job
                if not self.app.backend.submit(job):
                    raise APIError(429, "request queue is full", "rate_limit_exceeded")
                result = None
                while result is None:
                    kind, value = self._next_event(job)
                    if kind == "done":
                        result = value
                if result.reason == "cancelled":
                    if job.timed_out:
                        raise APIError(504, "request timed out", "request_timeout")
                    raise APIError(500, "request cancelled", "request_cancelled")
                if result.reason != "stop" or len(result.option_logits) != len(
                    job.score_tokens
                ):
                    raise APIError(500, "runtime protocol error", "protocol_error")
                input_tokens += result.prompt_tokens
                answers[qid] = judgments.systemone_answer(
                    spec, judgments.softmax(list(result.option_logits))
                )
                active_job = None
        except BaseException:
            if active_job is not None:
                self.app.backend.cancel(active_job)
            raise
        self._json(
            200,
            {
                "model": self.app.model,
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        )

    def _systemone_error(self, error):
        if self._response_started:
            return
        self._log_api_error(
            APIError(422, error.details[0]["msg"], "unprocessable_entity")
        )
        try:
            self._json(422, {"detail": error.details})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def _next_event(self, job, on_idle=None):
        while True:
            if self._client_disconnected():
                self.app.backend.cancel(job)
                raise ConnectionResetError
            remaining = job.deadline - time.monotonic()
            if remaining <= 0:
                self.app.backend.cancel(job, timed_out=True)
                raise APIError(504, "request timed out", "request_timeout")
            try:
                event = job.events.get(timeout=min(remaining, CLIENT_DISCONNECT_POLL))
            except queue.Empty:
                event = None
            # A queued failure takes precedence over a keepalive: until headers
            # are sent, the caller can still return its HTTP error status.
            if event is not None and event[0] == "error":
                raise event[1]
            if time.monotonic() >= job.deadline:
                continue
            # Native token activity can be buffered by the tool projector.
            # Measure silence on the HTTP stream across calls, not time spent
            # waiting for an empty native-event queue.
            if (
                on_idle is not None
                and time.monotonic() - self._last_sse_write >= SSE_KEEPALIVE_SECONDS
            ):
                on_idle()
            if event is not None:
                return event

    def _client_disconnected(self):
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.connection, selectors.EVENT_READ)
                readable = selector.select(0)
            # The body is already consumed and every response closes the
            # connection. Drain unexpected trailing bytes so they cannot hide EOF.
            return bool(readable) and not self.connection.recv(
                65536, socket.MSG_DONTWAIT
            )
        except BlockingIOError:
            return False
        except (OSError, ValueError):
            return True

    def _finalize_content(self, content, job, has_tools, incomplete, projector=None):
        structured = job.response_validator is not None
        if not has_tools or (structured and not content.lstrip().startswith("<")):
            if not incomplete:
                validate_response_content(content, job.response_validator)
                if has_tools:
                    validate_tool_calls([], job.tool_policy)
            return content, []
        if incomplete:
            if projector is None:
                projector = StreamingToolCallProjector(
                    job.tool_policy, job.public_id, structured
                )
                projector.put(content)
            return projector.interrupted_result()
        content, tool_calls = parse_tool_calls(content, job.public_id, job.tool_policy)
        validate_tool_calls(tool_calls, job.tool_policy)
        if structured:
            if not tool_calls:
                validate_response_content(content, job.response_validator)
            elif content.strip():
                raise APIError(
                    500,
                    "structured tool output contains text outside tool calls",
                    "invalid_model_output",
                )
        return content, tool_calls

    def _collect(
        self,
        job,
        thinking,
        has_tools,
        on_text=None,
        on_tool_delta=None,
        on_idle=None,
        on_progress=None,
    ):
        splitter = ReasoningSplitter(thinking)
        tool_projector = (
            StreamingToolCallProjector(
                job.tool_policy, job.public_id, job.response_validator is not None
            )
            if has_tools and on_text is not None
            else None
        )
        reasoning, content, result = [], [], None

        def append(field, text):
            (reasoning if field == "reasoning_content" else content).append(text)
            if on_text is None:
                return
            if field == "reasoning_content":
                on_text(field, text)
                return
            if tool_projector is not None:
                for kind, value in tool_projector.put(text):
                    if kind == "content":
                        on_text("content", value)
                    else:
                        on_tool_delta(value)
                return
            on_text(field, text)

        while result is None:
            kind, value = self._next_event(job, on_idle)
            if kind == "text":
                for field, text in splitter.put(value):
                    append(field, text)
            elif kind == "progress" and on_progress is not None:
                on_progress(value)
            elif kind == "done":
                result = value
        for field, text in splitter.finish():
            append(field, text)
        if result.reason == "cancelled":
            if job.timed_out:
                raise APIError(504, "request timed out", "request_timeout")
            raise APIError(500, "request cancelled", "request_cancelled")
        content_text = "".join(content)
        incomplete = result.reason == "length"
        content_text, tool_calls = self._finalize_content(
            content_text, job, has_tools, incomplete, tool_projector
        )
        if tool_projector is not None:
            for ready in tool_projector.finish(content_text, tool_calls, incomplete):
                on_text("content", ready)
        return "".join(reasoning), content_text, tool_calls, result, splitter.reasoning

    def _complete(self, job, thinking, has_tools):
        reasoning_text, content_text, tool_calls, result, _ = self._collect(
            job, thinking, has_tools
        )
        message = {"role": "assistant", "content": content_text or None}
        if reasoning_text:
            message["reasoning_content"] = reasoning_text
        if tool_calls:
            message["tool_calls"] = tool_calls
        self._json(
            200,
            completion_response(self.app.model, job, result, message, bool(tool_calls)),
        )

    def _anthropic_complete(self, job, thinking, has_tools):
        reasoning, content, tool_calls, result, _ = self._collect(
            job, thinking, has_tools
        )
        signature = (
            self.app.thinking_codec.encode(reasoning)
            if reasoning and job.thinking_display == "omitted"
            else ""
        )
        self._json(
            200,
            anthropic_response(
                self.app.model, job, reasoning, content, tool_calls, result, signature
            ),
        )

    def _anthropic_stream(self, job, thinking, has_tools):
        content_index = 0
        active_kind = None
        active_tool_index = None
        streamed_tool_calls = 0
        hidden_thinking = []
        omitted = job.thinking_display == "omitted"

        def send(event, payload):
            self._responses_sse(event, {"type": event, **payload})

        def keepalive():
            self._start_event_stream()
            send("ping", {})

        def finish_active():
            nonlocal active_kind, active_tool_index, content_index
            if active_kind is None:
                return
            if active_kind == "thinking" and omitted:
                signature = self.app.thinking_codec.encode("".join(hidden_thinking))
                send(
                    "content_block_delta",
                    {
                        "index": content_index,
                        "delta": {"type": "signature_delta", "signature": signature},
                    },
                )
                hidden_thinking.clear()
            send("content_block_stop", {"index": content_index})
            content_index += 1
            active_kind = None
            active_tool_index = None

        def put_text(field, text):
            nonlocal active_kind
            kind = "thinking" if field == "reasoning_content" else "text"
            if active_kind != kind:
                finish_active()
                active_kind = kind
                block = (
                    {"type": "thinking", "thinking": "", "signature": ""}
                    if kind == "thinking"
                    else {"type": "text", "text": ""}
                )
                send(
                    "content_block_start",
                    {"index": content_index, "content_block": block},
                )
                if kind == "thinking" and omitted:
                    send(
                        "content_block_delta",
                        {
                            "index": content_index,
                            "delta": {"type": "thinking_delta", "thinking": ""},
                        },
                    )
            if kind == "thinking" and omitted:
                hidden_thinking.append(text)
                return
            delta = (
                {"type": "thinking_delta", "thinking": text}
                if kind == "thinking"
                else {"type": "text_delta", "text": text}
            )
            send("content_block_delta", {"index": content_index, "delta": delta})

        def put_tool_delta(delta):
            nonlocal active_kind, active_tool_index, streamed_tool_calls
            function = delta.get("function") or {}
            index = delta["index"]
            if function.get("name") is not None:
                finish_active()
                active_kind = "tool"
                active_tool_index = index
                streamed_tool_calls = max(streamed_tool_calls, index + 1)
                send(
                    "content_block_start",
                    {
                        "index": content_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": delta["id"],
                            "name": function["name"],
                            "input": {},
                        },
                    },
                )
            arguments = function.get("arguments")
            if arguments:
                if active_kind != "tool" or active_tool_index != index:
                    raise APIError(
                        500,
                        "tool argument delta arrived before its tool header",
                        "internal_server_error",
                    )
                send(
                    "content_block_delta",
                    {
                        "index": content_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments,
                        },
                    },
                )

        def run():
            nonlocal content_index
            # Cache accounting is known at native admission. Keep the socket
            # alive while queued, but do not publish guessed input usage.
            if self._next_event(job, keepalive)[0] != "start":
                raise APIError(500, "runtime protocol error", "protocol_error")
            self._start_event_stream()
            send(
                "message_start",
                {
                    "message": {
                        "id": f"msg_{job.public_id}",
                        "type": "message",
                        "role": "assistant",
                        "model": self.app.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": anthropic_usage(len(job.prompt_tokens), 0, job.cache),
                    }
                },
            )
            _, content, tool_calls, result, _ = self._collect(
                job,
                thinking,
                has_tools,
                put_text,
                put_tool_delta,
                keepalive,
                lambda progress: send("ping", {"prompt_progress": progress}),
            )
            finish_active()
            if not content and not tool_calls and content_index == 0:
                send(
                    "content_block_start",
                    {
                        "index": content_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                send("content_block_stop", {"index": content_index})
                content_index += 1
            for call in tool_calls[streamed_tool_calls:]:
                send(
                    "content_block_start",
                    {
                        "index": content_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "input": {},
                        },
                    },
                )
                send(
                    "content_block_delta",
                    {
                        "index": content_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": call["function"]["arguments"],
                        },
                    },
                )
                send("content_block_stop", {"index": content_index})
                content_index += 1
            send(
                "message_delta",
                {
                    "delta": {
                        "stop_reason": anthropic_stop(result, tool_calls),
                        "stop_sequence": result.stop_sequence,
                    },
                    "usage": {"output_tokens": result.completion_tokens},
                },
            )
            send("message_stop", {})

        def send_error(error):
            send(
                "error",
                {
                    "error": {
                        "type": error.protocol_type(True),
                        "message": error.message,
                    }
                },
            )

        self._guarded_stream(job, run, send_error)

    def _responses_complete(self, job, thinking, has_tools):
        reasoning, content, tool_calls, result, reasoning_active = self._collect(
            job, thinking, has_tools
        )
        status = "incomplete" if result.reason == "length" else "completed"
        reasoning_status = (
            "incomplete" if status == "incomplete" and reasoning_active else "completed"
        )
        output = responses_output(
            job, reasoning, content, tool_calls, status, reasoning_status
        )
        response = responses_response(
            self.app.model,
            job,
            status,
            output,
            result=result,
        )
        self.app.persist_response(job, response, output)
        self._json(
            200,
            response,
        )

    def _start_event_stream(self):
        if self._response_started:
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._response_started = True

    def _sse(self, payload):
        data = (
            payload.encode("utf-8")
            if isinstance(payload, str)
            else json_codec.encode(payload)
        )
        self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()
        self._last_sse_write = time.monotonic()

    def _responses_sse(self, event, payload):
        data = json_codec.encode(payload)
        self.wfile.write(f"event: {event}\ndata: ".encode() + data + b"\n\n")
        self.wfile.flush()
        self._last_sse_write = time.monotonic()

    def _sse_keepalive(self):
        # SSE comments are invisible to SDK event decoders but still count as
        # transport progress. Long prefill and resource waits must not look
        # like dead connections to strict local-agent idle timers.
        self._start_event_stream()
        self.wfile.write(b": splash-keepalive\n\n")
        self.wfile.flush()
        self._last_sse_write = time.monotonic()

    def _guarded_stream(self, job, run, send_error):
        try:
            run()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.app.backend.cancel(job)
        except APIError as error:
            self.app.backend.cancel(job)
            if not self._response_started:
                raise
            if error.status >= 500:
                self._log_api_error(error)
            try:
                send_error(error)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
        except Exception as error:
            self.app.backend.cancel(job)
            if not self._response_started:
                raise
            log_unexpected(error)
            try:
                send_error(
                    APIError(500, "internal server error", "internal_server_error")
                )
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

    def _responses_stream(self, job, thinking, has_tools):
        output, sequence = [], 0
        active_kind, active_parts = None, []
        active_call = None
        streamed_tool_calls = 0

        def send(event, **payload):
            nonlocal sequence
            self._responses_sse(
                event, {"type": event, "sequence_number": sequence, **payload}
            )
            sequence += 1

        def begin():
            if self._response_started:
                return
            self._start_event_stream()
            send(
                "response.created",
                response=responses_response(self.app.model, job, "in_progress", []),
            )
            send(
                "response.in_progress",
                response=responses_response(self.app.model, job, "in_progress", []),
            )

        def keepalive():
            begin()
            if active_kind is None and not output:
                send(
                    "response.in_progress",
                    response=responses_response(self.app.model, job, "in_progress", []),
                )
            else:
                # Do not replace already-streamed output with an empty snapshot.
                self._sse_keepalive()

        def start_part(kind, item, index):
            if kind == "reasoning":
                send(
                    "response.reasoning_summary_part.added",
                    item_id=item["id"],
                    output_index=index,
                    summary_index=0,
                    part={"type": "summary_text", "text": ""},
                )
            elif kind == "message":
                send(
                    "response.content_part.added",
                    item_id=item["id"],
                    output_index=index,
                    content_index=0,
                    part={"type": "output_text", "text": "", "annotations": []},
                )

        def finish_part(kind, item, index):
            if kind == "function_call":
                send(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=index,
                    name=item["name"],
                    arguments=item["arguments"],
                )
            elif kind == "reasoning":
                text = item["summary"][0]["text"] if item["summary"] else ""
                part = {"type": "summary_text", "text": text}
                send(
                    "response.reasoning_summary_text.done",
                    item_id=item["id"],
                    output_index=index,
                    summary_index=0,
                    text=text,
                )
                payload = {
                    "item_id": item["id"],
                    "output_index": index,
                    "summary_index": 0,
                    "part": part,
                }
                if item["status"] == "incomplete":
                    payload["status"] = "incomplete"
                send("response.reasoning_summary_part.done", **payload)
            elif kind == "message":
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

        def finish_active(status="completed", value=None):
            nonlocal active_call, active_kind, active_parts
            if active_kind is None:
                return
            index = len(output)
            if active_kind == "function_call":
                if value is None:
                    value = {
                        "id": active_call["id"],
                        "type": "function",
                        "function": {
                            "name": active_call["function"]["name"],
                            "arguments": "".join(active_parts),
                        },
                    }
            item = responses_item(
                job,
                active_kind,
                "".join(active_parts) if value is None else value,
                index,
                status,
            )
            finish_part(active_kind, item, index)
            send("response.output_item.done", output_index=index, item=item)
            output.append(item)
            active_call, active_kind, active_parts = None, None, []

        def emit_delta(kind, item, index, text):
            if kind == "function_call":
                event, extra = "response.function_call_arguments.delta", {}
            elif kind == "reasoning":
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
                item_id=item["id"],
                output_index=index,
                delta=text,
                **extra,
            )

        def put_text(field, text):
            nonlocal active_kind
            kind = "reasoning" if field == "reasoning_content" else "message"
            if kind != active_kind:
                finish_active()
                active_kind = kind
                item = responses_item(job, kind, "", len(output), "in_progress")
                if kind == "message":
                    item["content"] = []
                send(
                    "response.output_item.added",
                    output_index=len(output),
                    item=item,
                )
                start_part(kind, item, len(output))
            else:
                item = responses_item(job, kind, "", len(output), "in_progress")
            active_parts.append(text)
            emit_delta(kind, item, len(output), text)

        def put_tool_delta(delta):
            nonlocal active_call, active_kind, active_parts, streamed_tool_calls
            function = delta.get("function") or {}
            index = delta["index"]
            if function.get("name") is not None:
                finish_active()
                active_kind = "function_call"
                active_parts = []
                active_call = {
                    "id": delta["id"],
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": "",
                    },
                }
                streamed_tool_calls = max(streamed_tool_calls, index + 1)
                item = responses_item(
                    job,
                    "function_call",
                    active_call,
                    len(output),
                    "in_progress",
                )
                send(
                    "response.output_item.added",
                    output_index=len(output),
                    item=item,
                )
            arguments = function.get("arguments")
            if arguments:
                if active_kind != "function_call" or index + 1 != streamed_tool_calls:
                    raise APIError(
                        500,
                        "tool argument delta arrived before its tool header",
                        "internal_server_error",
                    )
                active_parts.append(arguments)
                item = responses_item(
                    job,
                    "function_call",
                    active_call,
                    len(output),
                    "in_progress",
                )
                emit_delta("function_call", item, len(output), arguments)

        def emit_item(kind, value, status="completed"):
            index = len(output)
            pending = value if kind == "function_call" else ""
            pending_item = responses_item(job, kind, pending, index, "in_progress")
            if kind == "message":
                pending_item["content"] = []
            send(
                "response.output_item.added",
                output_index=index,
                item=pending_item,
            )
            start_part(kind, pending_item, index)
            item = responses_item(job, kind, value, index, status)
            if kind == "function_call":
                for arguments in argument_deltas(item["arguments"]):
                    emit_delta(kind, item, index, arguments)
            elif value:
                emit_delta(kind, item, index, value)
            finish_part(kind, item, index)
            send("response.output_item.done", output_index=index, item=item)
            output.append(item)

        def run():
            if self._next_event(job, keepalive)[0] != "start":
                raise APIError(500, "runtime protocol error", "protocol_error")
            begin()
            _, content, calls, result, reasoning_active = self._collect(
                job,
                thinking,
                has_tools,
                put_text,
                put_tool_delta,
                keepalive,
                lambda progress: send(
                    "response.in_progress",
                    response=responses_response(self.app.model, job, "in_progress", []),
                    prompt_progress=progress,
                ),
            )
            status = "incomplete" if result.reason == "length" else "completed"
            active_status = (
                "completed"
                if active_kind == "reasoning" and not reasoning_active
                else status
            )
            active_value = None
            if active_kind == "function_call" and streamed_tool_calls <= len(calls):
                active_value = calls[streamed_tool_calls - 1]
            finish_active(active_status, active_value)
            if has_tools:
                has_message = any(item["type"] == "message" for item in output)
                if (
                    content or (not calls and streamed_tool_calls == 0)
                ) and not has_message:
                    emit_item("message", content, status)
                for call in calls[streamed_tool_calls:]:
                    emit_item("function_call", call, status)
            elif not any(item["type"] == "message" for item in output):
                emit_item("message", "", status)
            event = (
                "response.incomplete"
                if status == "incomplete"
                else "response.completed"
            )
            response = responses_response(
                self.app.model, job, status, output, result=result
            )
            self.app.persist_response(job, response, output)
            send(
                event,
                response=response,
            )

        def send_error(error):
            send(
                "response.failed",
                response=responses_response(
                    self.app.model,
                    job,
                    "failed",
                    output,
                    error={
                        "type": error.protocol_type(),
                        "code": error.code,
                        "message": error.message,
                    },
                ),
            )

        self._guarded_stream(job, run, send_error)

    def _stream(self, job, thinking, has_tools, stream_options):
        public_id = job.public_id

        def run():
            first = self._next_event(job, self._sse_keepalive)
            if first[0] != "start":
                raise APIError(500, "runtime protocol error", "protocol_error")
            self._start_event_stream()
            created = job.created_at
            self._sse(
                stream_chunk(
                    self.app.model,
                    public_id,
                    created,
                    {"role": "assistant", "content": ""},
                )
            )

            def put_progress(progress):
                chunk = stream_chunk(self.app.model, public_id, created, {})
                chunk["prompt_progress"] = progress
                self._sse(chunk)

            def put_text(field, text):
                self._sse(
                    stream_chunk(
                        self.app.model,
                        public_id,
                        created,
                        {field: text},
                    )
                )

            def put_tool_delta(delta):
                self._sse(
                    stream_chunk(
                        self.app.model,
                        public_id,
                        created,
                        {"tool_calls": [delta]},
                    )
                )

            _, _, tool_calls, result, _ = self._collect(
                job,
                thinking,
                has_tools,
                put_text,
                put_tool_delta,
                self._sse_keepalive,
                put_progress,
            )
            self._sse(
                stream_chunk(
                    self.app.model,
                    public_id,
                    created,
                    {},
                    finish_reason(result, tool_calls),
                )
            )
            if stream_options.get("include_usage"):
                self._sse(
                    stream_chunk(
                        self.app.model,
                        public_id,
                        created,
                        {},
                        usage=usage_dict(result, job),
                        metrics=metrics_dict(result),
                    )
                )
            self._sse("[DONE]")

        def send_error(error):
            self._sse(
                {
                    "error": {
                        "message": error.message,
                        "type": error.protocol_type(),
                        "code": error.code,
                    }
                }
            )
            self._sse("[DONE]")

        self._guarded_stream(job, run, send_error)


class HttpAdmission:
    """Nonwaiting capacity gate, in request counts or input bytes."""

    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("HTTP admission capacity must be a positive integer")
        self.capacity = capacity
        self.active = 0
        # Input finalizers can run during a stats snapshot on this thread.
        self.lock = threading.RLock()
        self.idle = threading.Event()
        self.idle.set()

    def acquire(self, amount=1):
        with self.lock:
            if self.active + amount > self.capacity:
                return False
            self.active += amount
            self.idle.clear()
            return True

    def release(self, amount=1):
        with self.lock:
            if amount > self.active:
                raise RuntimeError("HTTP admission slot released without acquisition")
            self.active -= amount
            if self.active == 0:
                self.idle.set()

    def stats(self):
        with self.lock:
            return {"active": self.active, "capacity": self.capacity}


class RequestBodyReservation:
    """Account input bytes until preparation and any retained input are released."""

    def __init__(self, admission, size):
        if not admission.acquire(size):
            raise APIError(
                503,
                "request body capacity is exhausted; retry shortly",
                "frontend_overloaded",
            )
        self.admission = admission
        self.size = size

    def release(self):
        self.admission.release(self.size)
        self.size = 0

    def grow(self, size):
        if not self.admission.acquire(size):
            raise APIError(
                503,
                "retained input capacity is exhausted; retry shortly",
                "frontend_overloaded",
            )
        self.size += size

    def retain_for(self, job):
        # Generation retains schemas and, for Responses, conversation history.
        # Text/image prompts have otherwise become tokens and prepared pixels.
        policy = job.tool_policy
        retained = (
            job.response_history_items,
            job.response_format,
            policy.schemas if policy else None,
            policy.namespaces if policy else None,
            job.stop_sequences,
        )
        retained = [value for value in retained if value]
        size = json_codec.encoded_size(retained) if retained else 0
        if size > self.size:
            self.grow(size - self.size)
        else:
            self.admission.release(self.size - size)
        self.size = size
        if size:
            weakref.finalize(job, self.release)


class FrontendServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # Keep control/catalog capacity separate from generation capacity.
    # Neither gate allocates workers in advance.
    request_queue_size = 64
    control_connection_capacity = 64

    def __init__(
        self,
        address,
        app,
        io_timeout=HTTP_IO_TIMEOUT,
        bind_and_activate=True,
        request_capacity=32,
        allowed_hosts=(),
        api_key=None,
        webui=True,
        max_request_bytes=DEFAULT_MAX_REQUEST_BYTES,
    ):
        if not is_finite_number(io_timeout) or io_timeout <= 0:
            raise ValueError("io_timeout must be positive and finite")
        if (
            isinstance(max_request_bytes, bool)
            or not isinstance(max_request_bytes, int)
            or max_request_bytes <= 0
        ):
            raise ValueError("max_request_bytes must be a positive integer")
        self.max_request_bytes = max_request_bytes
        self.request_bodies = HttpAdmission(
            max(DEFAULT_REQUEST_BODY_BUDGET, 2 * max_request_bytes)
        )
        self.io_timeout = io_timeout
        self.api_key = validate_api_key(api_key) if api_key is not None else None
        self.webui = webui
        self.allowed_hosts = {
            host.lower().rstrip(".")
            for host in (*allowed_hosts, address[0], "localhost", "127.0.0.1", "::1")
            if host not in ("0.0.0.0", "::")
        }
        self.instance_id = secrets.token_hex(12)
        self.started_at = time.time()
        self.requests = HttpAdmission(request_capacity)
        self.token_counts = HttpAdmission(request_capacity)
        self.connections = HttpAdmission(
            request_capacity + self.control_connection_capacity
        )
        super().__init__(address, FrontendHandler, bind_and_activate)
        self.app = app

    def status(self):
        status = self.app.status()
        status["instance"] = {
            "id": self.instance_id,
            "pid": os.getpid(),
            "model": self.app.model,
            "host": self.server_address[0],
            "port": self.server_address[1],
            "started_at": self.started_at,
        }
        status["http"] = {
            "requests": self.requests.stats(),
            "request_body_bytes": self.request_bodies.stats(),
            "max_request_bytes": self.max_request_bytes,
            "token_counts": self.token_counts.stats(),
            "connections": self.connections.stats(),
        }
        return status

    def process_request(self, request, client_address):
        if not self.connections.acquire():
            # Header-only/idle connections must also be bounded. Do not create
            # a thread or block the accept loop to reject an excess socket.
            # The request path has not been read, so no API dialect is known.
            # Keep a generic server error and its stable diagnostic code.
            payload = b'{"error":{"type":"server_error","code":"frontend_overloaded","message":"HTTP connection capacity is exhausted"}}'
            response = (
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Type: application/json\r\nConnection: close\r\nRetry-After: 1\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                + payload
            )
            try:
                request.setblocking(False)
                request.sendall(response)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def server_close(self):
        super().server_close()
        self.connections.idle.wait(min(2.0, self.io_timeout))

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def _parse_max_context(value):
    if value == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"must be 'auto' or an integer in [1, {MAX_CONTEXT_TOKENS}]"
        ) from error
    if not 1 <= parsed <= MAX_CONTEXT_TOKENS:
        raise argparse.ArgumentTypeError(
            f"must be 'auto' or an integer in [1, {MAX_CONTEXT_TOKENS}]"
        )
    return parsed


def _parse_max_memory(value):
    if value == "auto":
        return None
    normalized = value.strip().upper()
    multipliers = {
        "K": 1024,
        "KB": 1024,
        "KIB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "MIB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "GIB": 1024**3,
    }
    suffix = ""
    for candidate in sorted(multipliers, key=len, reverse=True):
        if normalized.endswith(candidate):
            suffix = candidate
            normalized = normalized[: -len(candidate)]
            break
    try:
        number = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a positive byte count such as 32G"
        ) from error
    result = number * multipliers.get(suffix, 1)
    if number <= 0 or result > 2**63 - 1:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a positive byte count such as 32G"
        )
    return result


def _parse_request_size(value):
    size = _parse_max_memory(value)
    if size is None:
        raise argparse.ArgumentTypeError("must be a positive byte count such as 128M")
    return size


def _parse_model_id(value):
    if value.count("/") != 1:
        raise argparse.ArgumentTypeError(
            "use a full Hugging Face repository ID: owner/repo"
        )
    try:
        validate_repo_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("draft")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--model", type=_parse_model_id, required=True, metavar="OWNER/REPO"
    )
    parser.add_argument(
        "--served-model-name",
        action="append",
        default=[],
        type=validate_served_model_name,
        help="additional API model name; responses still identify the loaded model (repeatable)",
    )
    parser.add_argument(
        "--default-reasoning-effort",
        choices=REASONING_EFFORTS,
        default=os.environ.get("SPLASH_DEFAULT_REASONING_EFFORT"),
        help="Chat/Responses effort when unspecified (default: SPLASH_DEFAULT_REASONING_EFFORT or model template)",
    )
    parser.add_argument("--max-context", type=_parse_max_context, default=None)
    parser.add_argument("--max-memory", type=_parse_max_memory, default=None)
    parser.add_argument(
        "--kv-format",
        choices=("int8", "bf16"),
        default="int8",
        help="target KV cache storage (default: int8); bf16 uses more memory",
    )
    parser.add_argument(
        "--max-request-size",
        type=_parse_request_size,
        default=DEFAULT_MAX_REQUEST_BYTES,
        help="maximum HTTP request body size (default: 128M); "
        "shared input budget is max(512M, twice this limit)",
    )
    parser.add_argument("--max-image-pixels", type=int, default=image_input.MAX_PIXELS)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--allowed-host", action="append", default=[])
    parser.add_argument("--api-key", default=os.environ.get("SPLASH_API_KEY"))
    parser.add_argument("--no-webui", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--binary", default=str(ROOT / "build" / "splash"))
    args = parser.parse_args(argv)
    if (
        args.default_reasoning_effort is not None
        and args.default_reasoning_effort not in REASONING_EFFORTS
    ):
        parser.error(
            "invalid --default-reasoning-effort / SPLASH_DEFAULT_REASONING_EFFORT"
        )
    if args.api_key is not None:
        try:
            validate_api_key(args.api_key)
        except ValueError as error:
            parser.error(str(error))
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not image_input.MIN_PIXELS <= args.max_image_pixels <= image_input.MAX_PIXELS:
        parser.error(
            "--max-image-pixels must be in "
            f"[{image_input.MIN_PIXELS}, {image_input.MAX_PIXELS}]"
        )
    if not is_finite_number(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be positive and finite")
    if args.queue_size <= 0:
        parser.error("--queue-size must be positive")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    return args


def _native_command(args):
    command = [
        args.binary,
        "serve-native",
        args.target,
        args.draft,
        "auto" if args.max_context is None else str(args.max_context),
        "auto" if args.max_memory is None else str(args.max_memory),
    ]
    if args.kv_format != "int8":
        command.extend(("--kv-format", args.kv_format))
    return command


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main():
    args = parse_args()
    server = None
    backend = None
    # A server started in the background from a non-interactive shell inherits
    # SIGINT as ignored and Python then leaves it alone; install both stop
    # signals explicitly so scripts and supervisors can interrupt it.
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    try:
        # Bind before loading the tokenizer or model so duplicates fail early.
        # Activate only after the runtime is ready, keeping a partially started
        # service from receiving requests.
        server = FrontendServer(
            (args.host, args.port),
            None,
            bind_and_activate=False,
            request_capacity=args.queue_size,
            allowed_hosts=args.allowed_host,
            api_key=args.api_key,
            webui=not args.no_webui,
            max_request_bytes=args.max_request_size,
        )
        server.server_bind()
        thinking_codec = ThinkingCodec(load_thinking_key())
        print_status(f"Loading · {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True, trust_remote_code=False
        )
        validate_tokenizer(tokenizer)
        runtime = engine_runtime.MultiplexedRuntime(
            _native_command(args),
            startup_timeout=NATIVE_START_TIMEOUT,
            pending_limit=args.queue_size,
            eager_start=False,
        )
        backend = NativeBackend(
            runtime,
            tokenizer,
            request_logger=print_request,
        )
        if not runtime.wait_ready():
            raise engine_runtime.EngineUnhealthy("native runtime did not become ready")
        readiness = runtime.readiness
        if (
            readiness is None
            or not 1 <= readiness.max_context_tokens <= MAX_CONTEXT_TOKENS
            or (
                args.max_context is not None
                and readiness.max_context_tokens != args.max_context
            )
        ):
            raise engine_runtime.EngineUnhealthy(
                "native runtime reported an invalid context window"
            )
        effective_context = readiness.max_context_tokens
        constraint_factory = ConstraintFactory(tokenizer)
        app = Frontend(
            tokenizer,
            backend,
            args.model,
            effective_context,
            args.max_new_tokens,
            args.request_timeout,
            readiness.max_concurrent_requests,
            constraint_factory,
            max_image_pixels=args.max_image_pixels,
            thinking_codec=thinking_codec,
            served_model_names=args.served_model_name,
            default_reasoning_effort=args.default_reasoning_effort,
        )
        server.app = app
        server.server_activate()
        address = f"http://{args.host}:{server.server_port}"
        context = (
            f"{effective_context // 1024}K"
            if effective_context % 1024 == 0
            else f"{effective_context:,}"
        )
        print_status(f"Ready · {args.model} · context {context} · {address}")
        server.serve_forever()
    except (engine_runtime.EngineUnhealthy, ThinkingKeyError) as error:
        print_status(f"Error · {error}", error=True)
        raise SystemExit(1) from None
    except OSError as error:
        print_status(f"Error · unable to start HTTP server: {error}", error=True)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        pass
    finally:
        # main owns this process. Keep stop signals idempotent through child
        # cleanup and interpreter teardown, including after this function returns.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if backend is not None:
                print_status("Stopping · releasing engine resources")
                backend.close()
        finally:
            if server is not None:
                server.server_close()


if __name__ == "__main__":
    main()

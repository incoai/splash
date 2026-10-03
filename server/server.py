"""HTTP routes, protocol responses and serving-process startup."""

import argparse
import json
import math
import os
import queue
import re
import secrets
import select
import shlex
import signal
import socket
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from huggingface_hub.utils import validate_repo_id
from transformers import AutoTokenizer

from . import images as image_input
from . import json_codec, judgments, serve_options
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
    responses_item_id,
    responses_output,
    responses_response,
    stream_chunk,
    text_completion_chunk,
    text_completion_response,
)
from .backend import NativeBackend, NativeResult, remaining_request_time
from .chat_templates import ChatTemplateError, ChatTemplates
from .constraints import ConstraintFactory, validate_tokenizer
from .diagnostics import log_unexpected, print_request, print_status
from .errors import APIError, ContextLengthError
from .frontend import Frontend
from .http_security import OriginRefused, authenticate, validate_headers
from .latency import RequestLatency
from .metrics import prometheus_metrics, timings_dict, usage_dict
from .origins import ANY_ORIGIN
from .output import (
    BlockSequencer,
    ReasoningSplitter,
    StreamingToolCallProjector,
    validate_response_content,
    validate_tool_calls,
)
from .thinking import ThinkingCodec, ThinkingKeyError, load_thinking_key

# Match the former generation ingress envelope (32 slots × 16 MiB).
DEFAULT_REQUEST_BODY_BUDGET = 512 * 1024 * 1024
HTTP_IO_TIMEOUT = 30.0
HTTP_UPLOAD_BYTES_PER_SECOND = 512 * 1024
# Native events wake a waiting request at once; this only bounds how late a
# client disconnect is noticed.
CLIENT_DISCONNECT_POLL = 0.1
# How long a connection refused unread may take its client to close.
REFUSED_LINGER_SECONDS = 2.0
SSE_KEEPALIVE_SECONDS = 2.0
NATIVE_START_TIMEOUT = 600.0
ROOT = Path(__file__).parents[1]
CHAT_HTML = Path(__file__).with_name("chat.html").read_bytes()
# The chat page's brand mark, in its text colors, which the page shows too.
# Browsers, and other clients, ask for a site's icon at /favicon.ico.
FAVICON_SVG = Path(__file__).with_name("favicon.svg").read_bytes()
# The answer to a connection that gets no slot, or gives up its slot before
# its request is read. With no request path, no API dialect is known: a
# generic server error with its stable diagnostic code.
_CONNECTION_OVERLOADED_PAYLOAD = (
    b'{"error":{"type":"server_error","code":"frontend_overloaded",'
    b'"message":"HTTP connection capacity is exhausted"}}'
)
CONNECTION_OVERLOADED_RESPONSE = (
    b"HTTP/1.1 503 Service Unavailable\r\n"
    b"Content-Type: application/json\r\nConnection: close\r\nRetry-After: 1\r\n"
    b"Content-Length: %d\r\n\r\n%s"
    % (len(_CONNECTION_OVERLOADED_PAYLOAD), _CONNECTION_OVERLOADED_PAYLOAD)
)


def _content_length(value):
    """A Content-Length value as a byte count; None when it is not a
    decimal count, or has more digits than int() converts."""
    if not value.isascii() or not value.isdigit():
        return None
    try:
        return int(value)
    except ValueError:
        return None


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


@dataclass(slots=True)
class Collected:
    """A generation as FrontendHandler._collect gathered it."""

    reasoning: str
    content: str
    tool_calls: list
    result: NativeResult
    # The output ended inside its reasoning.
    reasoning_open: bool


class FrontendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    methods = "GET, HEAD, POST, DELETE, OPTIONS"

    def setup(self):
        self._response_started = False
        # What the response owes its request's origin, once the request has
        # passed validate_headers.
        self._allow_origin = None
        self._unread_body = 0
        self._last_sse_write = time.monotonic()
        super().setup()
        self.connection.settimeout(HTTP_IO_TIMEOUT)
        self._header_timer = threading.Timer(HTTP_IO_TIMEOUT, self._expire_headers)
        self._header_timer.daemon = True
        self._header_timer.start()

    def _expire_headers(self):
        self.server.connections.expire(self.connection)

    def finish(self):
        self._header_timer.cancel()
        if self._unread_body:
            self._discard_unread_body()
        super().finish()

    def _discard_unread_body(self):
        # Closing with request bytes unread resets the connection, and the
        # reset can destroy the response before a client still uploading
        # reads it. Half-close, then receive the rest of the upload on the
        # terms a body is read, waiting on the client as a connection with
        # no request yet does.
        self.server.connections.draining(self.connection)
        try:
            self.connection.shutdown(socket.SHUT_WR)
            while self._unread_body > 0:
                remaining = self._upload_deadline - time.monotonic()
                if remaining <= 0:
                    return
                self.connection.settimeout(min(remaining, HTTP_IO_TIMEOUT))
                chunk = self.rfile.read1(min(65536, self._unread_body))
                if not chunk:
                    return
                self._unread_body -= len(chunk)
        except OSError:
            pass

    def log_message(self, format, *args):
        pass

    def parse_request(self):
        try:
            parsed = super().parse_request()
        finally:
            self._header_timer.cancel()
        if not self.server.connections.serving(self.connection):
            self.close_connection = True
            return False
        if not parsed:
            return False
        if self.request_version not in {"HTTP/1.0", "HTTP/1.1"}:
            self.close_connection = True
            self.send_error(505, "HTTP version not supported")
            return False
        # The body still to come, which finish() receives if no handler
        # reads it before responding: its stated length or, without a valid
        # one, as much as the server accepts, in the time an upload gets,
        # counted from here.
        lengths = self.headers.get_all("Content-Length", [])
        length = _content_length(lengths[0]) if len(lengths) == 1 else None
        if length is not None:
            self._unread_body = length
        elif lengths or self.headers.get_all("Transfer-Encoding"):
            self._unread_body = self.server.max_request_bytes
        self._upload_deadline = time.monotonic() + self._upload_seconds(
            min(self._unread_body, self.server.max_request_bytes)
        )
        try:
            allowed_hosts = self.server.allowed_hosts | {
                self.connection.getsockname()[0].lower()
            }
            self._allow_origin = validate_headers(
                self.headers, allowed_hosts, self.server.allowed_origins
            )
            public = self.command == "OPTIONS" or (
                self.command in ("GET", "HEAD")
                and self.route
                in ("/", "/index.html", "/favicon.ico", "/health", "/ready")
            )
            if not public:
                authenticate(self.headers, self.server.api_key)
        except APIError as error:
            if isinstance(error, OriginRefused):
                self.server.refused_origins.report(error.origin)
            self.close_connection = True
            self._safe_error(error, self.route.startswith("/v1/messages"), log=False)
            return False
        return True

    def send_error(self, code, message=None, explain=None):
        # The stdlib's send_error, which answers requests it cannot parse or
        # route and parse_request's 505, writes an HTML page. After a request
        # line it cannot parse, or HTTP/0.9's, request_version is HTTP/0.9,
        # and it writes that page with no status line or headers. Answer as
        # any other error, over HTTP/1.1.
        self.request_version = self.protocol_version
        self._safe_error(
            APIError(code, message or self.responses[code][0]),
            self.route.startswith("/v1/messages"),
            log=False,
        )

    @property
    def app(self):
        return self.server.app

    @property
    def route(self):
        """The request's path as routing, authentication and error dialects
        all read it; empty before a request line parses."""
        return _normalize_path(getattr(self, "path", ""))

    def end_headers(self):
        # A browser hands a page the response from another origin only when
        # the response names that origin, so every response to an admitted
        # origin does, errors and event streams too, and exposes the retry
        # and authentication hints, which CORS hides by default.
        if self._allow_origin is not None:
            self.send_header("Access-Control-Allow-Origin", self._allow_origin)
            if self._allow_origin != ANY_ORIGIN:
                self.send_header("Vary", "Origin")
            self.send_header(
                "Access-Control-Expose-Headers", "Retry-After, WWW-Authenticate"
            )
        super().end_headers()

    def _send(self, status, data, content_type):
        self.send_response(status)
        if status == 503:
            self.send_header("Retry-After", "1")
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        route = self.route
        if (
            route == "/v1/systemone"
            or route == "/v1/models"
            or route.startswith("/v1/models/")
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
                self.route.startswith("/v1/messages"),
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
        path = "".join(char if char.isprintable() else "?" for char in self.route)
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

    def _upload_seconds(self, length):
        # Inactivity allowed at any point, plus the body at the upload rate.
        return HTTP_IO_TIMEOUT + length / HTTP_UPLOAD_BYTES_PER_SECOND

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
        length = _content_length(lengths[0])
        if length is None:
            raise APIError(400, "invalid Content-Length header")
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
        deadline = min(deadline, time.monotonic() + self._upload_seconds(length))
        self._body_reservation = RequestBodyReservation(
            self.server.request_bodies, length
        )
        payload = bytearray()
        try:
            while len(payload) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                self.connection.settimeout(min(remaining, HTTP_IO_TIMEOUT))
                chunk = self.rfile.read1(min(65536, length - len(payload)))
                if not chunk:
                    raise APIError(400, "request body ended before Content-Length")
                payload.extend(chunk)
        finally:
            self._unread_body = length - len(payload)
            self.connection.settimeout(HTTP_IO_TIMEOUT)
        text = payload.decode(json.detect_encoding(payload), "surrogatepass")
        payload.clear()
        return json_codec.loads(text)

    def do_HEAD(self):
        self.do_GET()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Allow", self.methods)
        if (
            self._allow_origin is not None
            and "Access-Control-Request-Method" in self.headers
        ):
            # A browser's preflight, which asks what the request it holds back
            # may use: every method the server has and, as in vLLM, any header.
            self.send_header("Access-Control-Allow-Methods", self.methods)
            requested = self.headers.get("Access-Control-Request-Headers")
            if requested is not None and requested.isprintable():
                self.send_header("Access-Control-Allow-Headers", requested)
            self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def version_string(self):
        return "Splash"

    def do_GET(self):
        path = self.route
        if path in ("/", "/index.html", "/favicon.ico"):
            if not self.server.webui:
                self._safe_error(APIError(404, "not found", "not_found"))
            elif path == "/favicon.ico":
                self._send(200, FAVICON_SVG, "image/svg+xml")
            else:
                self._send(200, CHAT_HTML, "text/html; charset=utf-8")
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
        if path == "/v1/models" or path.startswith("/v1/models/"):
            models = [
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "splash",
                    "max_model_len": self.app.max_context,
                    "context_length": self.app.max_context,
                    "vision": self.app.vision,
                    "input_modalities": self.app.input_modalities,
                    **(
                        {"root": self.app.response_model}
                        if name != self.app.response_model
                        else {}
                    ),
                }
                for name in self.app.model_names
            ]
            if path == "/v1/models":
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
                name = path.removeprefix("/v1/models/")
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
        path = self.route
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
        path = self.route
        count_tokens = path == "/v1/messages/count_tokens"
        prompt_only = count_tokens or path in ("/tokenize", "/apply-template")
        anthropic = path == "/v1/messages" or count_tokens
        systemone = path == "/v1/systemone"
        completions = path == "/v1/completions"
        if path not in (
            "/v1/chat/completions",
            "/v1/completions",
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
        refusal = None if prompt_only else self.app.backend.refusal()
        if refusal is not None:
            if systemone and refusal.status == 503:
                refusal = APIError(529, refusal.message, refusal.code)
            self._safe_error(refusal, anthropic, log=False)
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
                        body, thinking_resolver=self.app.thinking_codec.decode
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
                self.app.backend.submit(job)
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
                chat, thinking_display = anthropic_to_chat_body(
                    body, thinking_resolver=self.app.thinking_codec.decode
                )
                job = self.app.prepare(
                    chat,
                    deadline=deadline,
                    output_field="max_tokens",
                    clamp_output_budget=True,
                    thinking_display=thinking_display,
                )
                stream_options = None
            elif responses:
                job = self.app.prepare_responses(
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
                if completions:
                    job = self.app.prepare_completion(body, deadline=deadline)
                else:
                    job = self.app.prepare(body, deadline=deadline)
            body = None
            self._body_reservation.retain_for(job)
            self._body_reservation = None
            job.return_progress = return_progress
            job.latency = RequestLatency(self.app.latencies, started_at)
            remaining_request_time(deadline)
            if self._client_disconnected():
                raise ConnectionResetError("client disconnected before submission")
            self.app.backend.submit(job)
            submitted = True
            if anthropic and stream:
                self._anthropic_stream(job)
            elif anthropic:
                self._anthropic_complete(job)
            elif responses and stream:
                self._responses_stream(job)
            elif responses:
                self._responses_complete(job)
            elif stream:
                self._openai_stream(job, stream_options, chat=not completions)
            elif completions:
                self._text_completion(job)
            else:
                self._complete(job)
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

    def _await_done(self, job):
        """The result of a score job, which emits only start and done."""
        while (event := self._next_event(job))[0] != "done":
            pass
        return event[1]

    def _judgment_complete(self, job, row):
        result = self._await_done(job)
        self._json(
            200,
            judgments.judgment_response(self.app.response_model, row, job, result),
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
                self.app.backend.submit(job)
                result = self._await_done(job)
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
                "model": self.app.response_model,
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
                self.app.backend.cancel(job)
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
        # A poll object holds no descriptor: running out of descriptors
        # must not read as a disconnect.
        poller = select.poll()
        poller.register(self.connection, select.POLLIN)
        try:
            # The body is already consumed and every response closes the
            # connection. Drain unexpected trailing bytes so they cannot hide EOF.
            return bool(poller.poll(0)) and not self.connection.recv(
                65536, socket.MSG_DONTWAIT
            )
        except BlockingIOError:
            return False
        except ConnectionError:
            return True

    def _finalize_content(self, content, job, incomplete, projector):
        """The content and calls of the output, validated unless it was cut,
        and the content the stream still owes. `content` is read only without
        tools: with tools, the projector holds the content."""
        if job.tool_policy is None:
            if not incomplete:
                validate_response_content(content, job.response_validator)
            return content, [], ""
        content, tool_calls, unsent = projector.finish(incomplete)
        if not incomplete:
            validate_tool_calls(tool_calls, job.tool_policy)
            if not tool_calls:
                validate_response_content(content, job.response_validator)
            elif job.response_validator is not None and content.strip():
                raise APIError(
                    500,
                    "structured tool output contains text outside tool calls",
                    "invalid_model_output",
                )
        return content, tool_calls, unsent

    def _collect(
        self,
        job,
        *,
        on_start=None,
        on_text=None,
        on_tool_delta=None,
        on_idle=None,
        on_progress=None,
    ):
        """Gather a generation until it is done. The callbacks receive the
        start, each piece of output, the idle waits and prompt progress:
        streams send them, and complete Messages and Responses gather the
        output into blocks."""
        splitter = ReasoningSplitter(job.thinking)
        # Output with tools is parsed as it arrives whether it streams or not.
        projector = (
            StreamingToolCallProjector(
                job.tool_policy, job.public_id, job.response_validator is not None
            )
            if job.tool_policy is not None
            else None
        )
        reasoning, content, result = [], [], None

        def append(field, text):
            if field == "content" and projector is not None:
                events = projector.put(text)
            else:
                (reasoning if field == "reasoning_content" else content).append(text)
                events = [(field, text)]
            if on_text is None:
                return
            for kind, value in events:
                if kind == "tool":
                    on_tool_delta(value)
                else:
                    on_text(kind, value)

        while result is None:
            kind, value = self._next_event(job, on_idle)
            if kind == "start" and on_start is not None:
                on_start()
            elif kind == "text":
                for field, text in splitter.put(value):
                    append(field, text)
            elif kind == "progress" and on_progress is not None:
                on_progress(value)
            elif kind == "done":
                result = value
        for field, text in splitter.finish():
            append(field, text)
        content_text, tool_calls, unsent = self._finalize_content(
            "".join(content), job, result.reason == "length", projector
        )
        if unsent and on_text is not None:
            on_text("content", unsent)
        return Collected(
            "".join(reasoning), content_text, tool_calls, result, splitter.reasoning
        )

    def _complete(self, job):
        collected = self._collect(job)
        message = {"role": "assistant", "content": collected.content or None}
        if collected.reasoning:
            message["reasoning_content"] = collected.reasoning
        if collected.tool_calls:
            message["tool_calls"] = collected.tool_calls
        self._json(
            200,
            completion_response(
                self.app.response_model,
                job,
                collected.result,
                message,
                bool(collected.tool_calls),
            ),
        )

    def _text_completion(self, job):
        collected = self._collect(job)
        self._json(
            200,
            text_completion_response(
                self.app.response_model, job, collected.result, collected.content
            ),
        )

    def _anthropic_complete(self, job):
        sequencer = BlockSequencer()
        collected = self._collect(
            job,
            on_text=sequencer.text,
            on_tool_delta=sequencer.tool,
        )
        result = collected.result
        blocks = sequencer.finish(result.reason == "length", collected.reasoning_open)
        signature = (
            self.app.thinking_codec.encode(collected.reasoning)
            if collected.reasoning and job.thinking_display == "omitted"
            else ""
        )
        self._json(
            200,
            anthropic_response(
                self.app.response_model,
                job,
                blocks,
                result,
                collected.tool_calls,
                signature,
            ),
        )

    def _anthropic_stream(self, job):
        omitted = job.thinking_display == "omitted"

        def send(event, payload):
            self._event_sse(event, {"type": event, **payload})

        def start():
            # Cache accounting is known at native admission. Keep the socket
            # alive while queued, but do not publish guessed input usage.
            self._start_event_stream()
            send(
                "message_start",
                {
                    "message": {
                        "id": f"msg_{job.public_id}",
                        "type": "message",
                        "role": "assistant",
                        "model": self.app.response_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": anthropic_usage(len(job.prompt_tokens), 0, job.cache),
                    }
                },
            )

        def keepalive():
            self._start_event_stream()
            send("ping", {})

        def open_block(index, block):
            if block.kind == "reasoning":
                content_block = {"type": "thinking", "thinking": "", "signature": ""}
            elif block.kind == "text":
                content_block = {"type": "text", "text": ""}
            else:
                content_block = {
                    "type": "tool_use",
                    "id": block.call_id,
                    "name": block.name,
                    "input": {},
                }
            send(
                "content_block_start", {"index": index, "content_block": content_block}
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
                signature = self.app.thinking_codec.encode(block.text)
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
            collected = self._collect(
                job,
                on_start=start,
                on_text=sequencer.text,
                on_tool_delta=sequencer.tool,
                on_idle=keepalive,
                on_progress=lambda progress: send(
                    "ping", {"prompt_progress": progress}
                ),
            )
            result = collected.result
            sequencer.finish(result.reason == "length", collected.reasoning_open)
            send(
                "message_delta",
                {
                    "delta": {
                        "stop_reason": anthropic_stop(
                            result,
                            collected.tool_calls,
                            job.output_clamped_to_context,
                        ),
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

    def _responses_complete(self, job):
        sequencer = BlockSequencer()
        collected = self._collect(
            job,
            on_text=sequencer.text,
            on_tool_delta=sequencer.tool,
        )
        result = collected.result
        incomplete = result.reason == "length"
        output = responses_output(
            job, sequencer.finish(incomplete, collected.reasoning_open)
        )
        response = responses_response(
            self.app.response_model,
            job,
            "incomplete" if incomplete else "completed",
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

    def _write_sse(self, frame):
        self.wfile.write(frame)
        self.wfile.flush()
        self._last_sse_write = time.monotonic()

    def _sse(self, payload):
        data = (
            payload.encode("utf-8")
            if isinstance(payload, str)
            else json_codec.encode(payload)
        )
        self._write_sse(b"data: " + data + b"\n\n")

    def _event_sse(self, event, payload):
        data = json_codec.encode(payload)
        self._write_sse(f"event: {event}\ndata: ".encode() + data + b"\n\n")

    def _sse_keepalive(self):
        # An SSE comment is traffic, so socket read timeouts and proxies do
        # not take a long prefill or resource wait for a dead connection, but
        # event decoders skip it and clients that time out on missing data
        # events ignore it. Streams send it only where no data event fits: the
        # chat and text completion streams until the request starts, the
        # Responses stream once output has begun.
        self._start_event_stream()
        self._write_sse(b": splash-keepalive\n\n")

    def _sse_error(self, error):
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

    def _responses_stream(self, job):
        output, sequence = [], 0

        def send(event, **payload):
            nonlocal sequence
            self._event_sse(
                event, {"type": event, "sequence_number": sequence, **payload}
            )
            sequence += 1

        def begin():
            if self._response_started:
                return
            self._start_event_stream()
            send(
                "response.created",
                response=responses_response(
                    self.app.response_model, job, "in_progress", []
                ),
            )
            send(
                "response.in_progress",
                response=responses_response(
                    self.app.response_model, job, "in_progress", []
                ),
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
            if block.kind == "tool":
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
                    response=responses_response(
                        self.app.response_model, job, "in_progress", []
                    ),
                )
            else:
                # The data event that adds nothing, response.in_progress,
                # carries a snapshot of the response, which would replace the
                # output streamed so far. A comment keeps the stream alive
                # instead, though clients that time out on missing data events
                # ignore it.
                self._sse_keepalive()

        def run():
            collected = self._collect(
                job,
                on_start=begin,
                on_text=sequencer.text,
                on_tool_delta=sequencer.tool,
                on_idle=keepalive,
                on_progress=lambda progress: send(
                    "response.in_progress",
                    response=responses_response(
                        self.app.response_model, job, "in_progress", []
                    ),
                    prompt_progress=progress,
                ),
            )
            result = collected.result
            incomplete = result.reason == "length"
            sequencer.finish(incomplete, collected.reasoning_open)
            status = "incomplete" if incomplete else "completed"
            response = responses_response(
                self.app.response_model, job, status, output, result=result
            )
            self.app.persist_response(job, response, output)
            send(
                f"response.{status}",
                response=response,
            )

        def send_error(error):
            send(
                "response.failed",
                response=responses_response(
                    self.app.response_model,
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

    def _openai_stream(self, job, stream_options, *, chat):
        """A Chat or text completion stream; text completions have no tools."""
        chunk = partial(
            stream_chunk if chat else text_completion_chunk,
            self.app.response_model,
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
            self._start_event_stream()
            if chat:
                self._sse(chunk({"role": "assistant", "content": ""}))

        def keepalive():
            # After the start, a chunk that adds nothing: tool arguments of
            # arrays and objects are buffered until complete, which can take
            # minutes, and clients that time out on missing data events
            # ignore SSE comments.
            if started:
                self._sse(chunk(empty))
            else:
                self._sse_keepalive()

        def run():
            collected = self._collect(
                job,
                on_start=start,
                on_text=lambda field, text: self._sse(chunk(payload(field, text))),
                on_tool_delta=lambda delta: self._sse(chunk({"tool_calls": [delta]})),
                on_idle=keepalive,
                on_progress=lambda progress: self._sse(
                    chunk(empty) | {"prompt_progress": progress}
                ),
            )
            result = collected.result
            self._sse(
                chunk(
                    empty,
                    finish_reason(result, collected.tool_calls),
                    timings=timings_dict(result),
                )
            )
            if stream_options.get("include_usage"):
                self._sse(
                    chunk(empty, usage=usage_dict(result, job), metrics=result.metrics)
                )
            self._sse("[DONE]")

        self._guarded_stream(job, run, self._sse_error)


class HttpAdmission:
    """Nonwaiting capacity gate, in request counts or input bytes."""

    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("HTTP admission capacity must be a positive integer")
        self.capacity = capacity
        self.active = 0
        # Input finalizers can run during a stats snapshot on this thread.
        self.lock = threading.RLock()

    def acquire(self, amount=1):
        with self.lock:
            if self.active + amount > self.capacity:
                return False
            self.active += amount
            return True

    def release(self, amount=1):
        with self.lock:
            if amount > self.active:
                raise RuntimeError("HTTP admission slot released without acquisition")
            self.active -= amount

    def stats(self):
        with self.lock:
            return {"active": self.active, "capacity": self.capacity}


def _refuse_connection(connection):
    """Send CONNECTION_OVERLOADED_RESPONSE without waiting: the server has
    read no request from the connection and written it no response, so the
    response fits in its send buffer."""
    try:
        connection.send(CONNECTION_OVERLOADED_RESPONSE, socket.MSG_DONTWAIT)
    except OSError:
        pass


def _has_input(connection):
    """Whether `connection` holds input its thread has yet to read, such as
    a request that arrived before its thread ran."""
    # A poll object holds no descriptor.
    poller = select.poll()
    poller.register(connection, select.POLLIN)
    return bool(poller.poll(0))


class LingeringCloser:
    """Closes connections answered without reading their requests.

    Closing a connection with request bytes unread resets it, and the reset
    can destroy the answer before the client reads it. Each connection given
    here is half-closed instead; one thread reads and drops what its client
    still sends, and closes it once the client has closed or `linger`
    seconds after its answer. At most `capacity` wait at once; any beyond
    them are closed at once.
    """

    # How soon the thread first reads a connection that arrives while it
    # waits on others.
    TICK = 0.05

    def __init__(self, capacity, linger):
        self.capacity = capacity
        self.linger = linger
        self.changed = threading.Condition()
        self.arrivals = []
        self.held = 0
        self.stopped = False
        self.thread = threading.Thread(
            target=self._run, name="lingering close", daemon=True
        )
        self.thread.start()

    def close(self, connection):
        try:
            connection.shutdown(socket.SHUT_WR)
            connection.setblocking(False)
        except OSError:
            connection.close()
            return
        with self.changed:
            if self.stopped or self.held >= self.capacity:
                connection.close()
                return
            self.held += 1
            self.arrivals.append((connection, time.monotonic() + self.linger))
            self.changed.notify()

    def stop(self):
        """Close every waiting connection and end the thread."""
        with self.changed:
            self.stopped = True
            self.changed.notify()
        self.thread.join()

    def _run(self):
        # A poll object holds no descriptor.
        poller = select.poll()
        waiting = {}
        while True:
            with self.changed:
                while not (waiting or self.arrivals or self.stopped):
                    self.changed.wait()
                arrivals, self.arrivals = self.arrivals, []
                stopped = self.stopped
            for connection, deadline in arrivals:
                poller.register(connection, select.POLLIN)
                waiting[connection.fileno()] = connection, deadline
            if not stopped:
                for descriptor, _ in poller.poll(self.TICK * 1000):
                    connection, _ = waiting[descriptor]
                    try:
                        if connection.recv(65536):
                            continue
                    except BlockingIOError:
                        continue
                    except OSError:
                        pass
                    waiting[descriptor] = connection, 0.0
            now = time.monotonic()
            done = [
                descriptor
                for descriptor, (_, deadline) in waiting.items()
                if stopped or deadline <= now
            ]
            for descriptor in done:
                poller.unregister(descriptor)
                waiting.pop(descriptor)[0].close()
            with self.changed:
                self.held -= len(done)
            if stopped:
                return


class ConnectionSlots:
    """The connections the server gives a thread, at most `capacity`.

    One that waits for its request with nothing yet to read, or drains an
    upload refused unread, gives its slot to a new connection when no slot
    is free, the longest waiting first, so stalled connections, however
    many and from however many addresses, cannot keep others out. One whose
    request has arrived keeps its slot, although its thread may not have
    run yet: when every slot has a request, arrived or in progress, the new
    connection is refused. One that gives way still awaiting its request
    gets the 503 of a connection refused at the accept, as its request may
    be on its way. One draining a refused upload already has its response.
    """

    # What a connection with a slot is doing.
    AWAITING_REQUEST = "awaiting request"
    SERVING = "serving"
    DRAINING = "draining"

    def __init__(self, capacity):
        self.capacity = capacity
        # Each connection with a slot, mapped to what it is doing; those that
        # wait on their client in the order they began to.
        self.holders = {}
        self.lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()

    def admit(self, connection):
        """Give `connection` a slot, awaiting its request; False when every
        slot has a request, arrived or in progress."""
        with self.lock:
            if len(self.holders) >= self.capacity:
                waiting = next(
                    (
                        held
                        for held, state in self.holders.items()
                        if state == self.DRAINING
                        or (state == self.AWAITING_REQUEST and not _has_input(held))
                    ),
                    None,
                )
                if waiting is None:
                    return False
                if self.holders[waiting] == self.AWAITING_REQUEST:
                    _refuse_connection(waiting)
                self._close(waiting)
            self.holders[connection] = self.AWAITING_REQUEST
            self.idle.clear()
            return True

    def expire(self, connection):
        """Close `connection`, and free its slot, if it still awaits its
        request."""
        with self.lock:
            if self.holders.get(connection) == self.AWAITING_REQUEST:
                self._close(connection)

    def _close(self, connection):
        # Under the lock, which a connection's release takes before the
        # connection is closed, so the descriptor is still its own. Its
        # thread sees the end of input; it has lost its slot, so a request
        # whose headers the shutdown cut short is not served.
        del self.holders[connection]
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def serving(self, connection):
        """Mark the request on `connection` in progress; False once the
        connection has lost its slot."""
        with self.lock:
            if connection not in self.holders:
                return False
            self.holders[connection] = self.SERVING
            return True

    def draining(self, connection):
        """`connection`, answered, drains an upload refused unread: it waits
        on its client again, last in line."""
        with self.lock:
            if self.holders.pop(connection, None) is not None:
                self.holders[connection] = self.DRAINING

    def release(self, connection):
        """Give back the slot of `connection`, if it still has one, before
        the connection is closed."""
        with self.lock:
            self.holders.pop(connection, None)
            if not self.holders:
                self.idle.set()

    def stats(self):
        with self.lock:
            return {"active": len(self.holders), "capacity": self.capacity}


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


class RefusedOriginLog:
    """Prints each origin the server refuses, once. Its browser hides the 403
    from the page, which sees a network error, so the operator learns here
    which --allowed-origin would admit it. Any client can send any origin, so
    past LIMIT origins no more are printed, and the flag value is quoted for a
    shell."""

    LIMIT = 32

    def __init__(self):
        self.lock = threading.Lock()
        # The origins printed, and the first one past LIMIT.
        self.origins = set()

    def report(self, origin):
        # Printing under the lock keeps the closing line last.
        with self.lock:
            if origin in self.origins or len(self.origins) > self.LIMIT:
                return
            self.origins.add(origin)
            if len(self.origins) > self.LIMIT:
                print_status("Refused · further Origins are not logged", error=True)
                return
            # Visible ASCII, as parse_origin admits, but of any length.
            shown = origin[:256]
            print_status(
                f"Refused · Origin {shown} · restart with --allowed-origin "
                f"{shlex.quote(shown)} to accept it",
                error=True,
            )


class FrontendServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # Connections the kernel holds until the accept loop takes them. A burst
    # beyond this queue is reset by the kernel, unseen by the server, so ask
    # for as many as uvicorn does; the kernel caps it (128 on macOS).
    # Queued connections take no thread or descriptor.
    request_queue_size = 2048
    # Keep control/catalog capacity separate from generation capacity.
    # Neither gate allocates workers in advance.
    control_connection_capacity = 64
    # Connections refused at the accept that wait at once, on one thread,
    # for their clients to close.
    refused_connection_capacity = 64

    def __init__(
        self,
        address,
        app,
        bind_and_activate=True,
        request_capacity=32,
        allowed_hosts=(),
        api_key=None,
        webui=True,
        max_request_bytes=serve_options.DEFAULT_MAX_REQUEST_BYTES,
        allowed_origins=(),
    ):
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
        self.api_key = api_key
        self.webui = webui
        self.allowed_hosts = {
            host.lower().rstrip(".")
            for host in (*allowed_hosts, address[0], "localhost", "127.0.0.1", "::1")
            if host not in ("0.0.0.0", "::")
        }
        # As serve_options.parse_allowed_origin returns them.
        self.allowed_origins = frozenset(allowed_origins)
        self.refused_origins = RefusedOriginLog()
        self.instance_id = secrets.token_hex(12)
        self.started_at = time.time()
        self.requests = HttpAdmission(request_capacity)
        self.token_counts = HttpAdmission(request_capacity)
        self.connections = ConnectionSlots(
            request_capacity + self.control_connection_capacity
        )
        self.refused = LingeringCloser(
            self.refused_connection_capacity, REFUSED_LINGER_SECONDS
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
        if not self.connections.admit(request):
            # Do not create a thread or block the accept loop to reject an
            # excess socket.
            _refuse_connection(request)
            self.refused.close(request)
            return
        super().process_request(request, client_address)

    def shutdown_request(self, request):
        self.connections.release(request)
        super().shutdown_request(request)

    def server_close(self):
        super().server_close()
        self.refused.stop()
        self.connections.idle.wait(min(2.0, HTTP_IO_TIMEOUT))

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def _parse_model_id(value):
    repo_id, separator, variant = value.partition(":")
    if repo_id.count("/") != 1:
        raise argparse.ArgumentTypeError(
            "use a full Hugging Face repository ID: owner/repo[:variant]"
        )
    try:
        validate_repo_id(repo_id)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    if separator and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", variant):
        raise argparse.ArgumentTypeError(
            "model variant must be a short name such as UD-Q4_K_M"
        )
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_root",
        metavar="MODEL_DIRECTORY",
        help="installed model directory holding target/ and draft/",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--model", type=_parse_model_id, required=True, metavar="OWNER/REPO"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--binary", default=str(ROOT / "build" / "splash"))
    serve_options.add_serve_arguments(parser)
    args = parser.parse_args(argv)
    serve_options.check_serve_arguments(parser, args)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    return args


def _native_command(args):
    command = [
        args.binary,
        "serve-native",
        args.model_root,
        "auto" if args.max_context is None else str(args.max_context),
        "auto" if args.max_memory is None else str(args.max_memory),
    ]
    if args.max_cache_disk:
        command.append(str(args.max_cache_disk))
    if args.kv_format != "int8":
        command.extend(("--kv-format", args.kv_format))
    if args.decode_share is not None:
        command.extend(("--decode-share", str(args.decode_share)))
    if args.max_image_pixels != image_input.MAX_PIXELS:
        command.extend(
            ("--max-image-patches", str(image_input.max_patches(args.max_image_pixels)))
        )
    return command


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main():
    args = parse_args()
    server = None
    runtime = None
    backend = None
    # A server started in the background from a non-interactive shell inherits
    # SIGINT as ignored and Python then leaves it alone; install both stop
    # signals explicitly so scripts and supervisors can interrupt it.
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    try:
        # The launcher blocks both across its exec: one sent while this module
        # imported arrives here and ends the startup cleanly.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
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
            allowed_origins=args.allowed_origin,
        )
        server.server_bind()
        if ANY_ORIGIN in args.allowed_origin and args.api_key is None:
            print_status(
                "Warning · --allowed-origin '*' without --api-key lets every web "
                "page open in a browser that reaches this server use it",
                error=True,
            )
        thinking_codec = ThinkingCodec(load_thinking_key())
        print_status(f"Loading · {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True, trust_remote_code=False
        )
        validate_tokenizer(tokenizer)
        chat_templates = ChatTemplates(tokenizer)
        print_status(f"Chat template · {chat_templates.describe()}")
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
            or not 1 <= readiness.max_context_tokens <= serve_options.MAX_CONTEXT_TOKENS
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
            # No deadline unless given, as in vLLM and SGLang; a request still
            # ends when its client disconnects.
            math.inf if args.request_timeout is None else args.request_timeout,
            readiness.max_concurrent_requests,
            constraint_factory=constraint_factory,
            chat_templates=chat_templates,
            max_image_pixels=args.max_image_pixels,
            thinking_codec=thinking_codec,
            served_model_names=args.served_model_name,
            announce_served_name=args.announce_served_name,
            default_reasoning_effort=args.default_reasoning_effort,
            vision=readiness.vision,
        )
        server.app = app
        server.server_activate()
        address = f"http://{args.host}:{server.server_port}"
        context = (
            f"{effective_context // 1024}K"
            if effective_context % 1024 == 0
            else f"{effective_context:,}"
        )
        mode = "" if readiness.vision else " · language only"
        print_status(f"Ready · {args.model} · context {context}{mode} · {address}")
        server.serve_forever()
    except (
        engine_runtime.EngineRuntimeError,
        ThinkingKeyError,
        ChatTemplateError,
    ) as error:
        print_status(f"Error · {error}", error=True)
        raise SystemExit(1) from None
    except OSError as error:
        print_status(f"Error · unable to start HTTP server: {error}", error=True)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        pass
    finally:
        # main owns this process. Keep stop signals idempotent through child
        # cleanup and interpreter teardown, including after this function
        # returns, except that a second Ctrl+C during cleanup stops the engine
        # without waiting for its graceful exit.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(
            signal.SIGINT,
            signal.SIG_IGN if runtime is None else lambda *_: runtime.kill(),
        )
        try:
            if backend is not None:
                print_status("Stopping · releasing engine resources")
                backend.close()
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if server is not None:
                server.server_close()


if __name__ == "__main__":
    main()

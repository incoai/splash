"""HTTP routes, protocol responses and serving-process startup."""

import argparse
import json
import math
import os
import queue
import re
import resource
import secrets
import select
import signal
import socket
import sys
import time
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

from transformers import AutoTokenizer

from . import images as image_input
from . import json_codec, judgments, serve_options
from . import protocol as wire
from . import runtime as engine_runtime
from .api_shapes import (
    anthropic_block,
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
from .connections import ConnectionSlots, HttpAdmission, RequestBodyReservation
from .constraints import ConstraintFactory, validate_tokenizer
from .diagnostics import log_unexpected, print_request, print_status
from .errors import (
    ANTHROPIC_ERRORS,
    OPENAI_ERRORS,
    RETRY_STATUSES,
    SYSTEMONE_ERRORS,
    APIError,
    ErrorDialect,
    RequestValidationError,
    field_error,
)
from .frontend import Frontend
from .http_security import (
    OriginRefused,
    RefusedOriginLog,
    authenticate,
    validate_headers,
)
from .latency import RequestLatency
from .metrics import prometheus_metrics, timings_dict, usage_dict
from .origins import ANY_ORIGIN
from .output import (
    BlockSequencer,
    ReasoningSplitter,
    StreamingToolCallProjector,
    validate_response_content,
)
from .thinking import ThinkingCodec, ThinkingKeyError, load_thinking_key

# Request bodies held at once, from upload through preparation and, for what
# a generation retains of them, until it ends, take at most max(this, twice
# --max-request-size) bytes, so two of the largest requests always fit.
DEFAULT_REQUEST_BODY_BUDGET = 512 * 1024 * 1024
HTTP_IO_TIMEOUT = 30.0
HTTP_UPLOAD_BYTES_PER_SECOND = 512 * 1024
# Native events wake a waiting request at once; this only bounds how late a
# client disconnect is noticed.
CLIENT_DISCONNECT_POLL = 0.1
# How long a connection refused unread may take its client to close.
REFUSED_LINGER_SECONDS = 2.0
# The soft descriptor limit the server raises its own to: macOS starts a
# process with 256, fewer than its connection slots, the refused connections
# lingering after a burst and the engine's pipes and files can need. 10240 is
# macOS's OPEN_MAX.
DESCRIPTOR_LIMIT = 10240
SSE_KEEPALIVE_SECONDS = 2.0
NATIVE_START_TIMEOUT = 600.0
ROOT = Path(__file__).parents[1]
CHAT_HTML = Path(__file__).with_name("chat.html").read_bytes()
CHAT_ASSETS = {
    f"/chat-assets/{name}": (Path(__file__).parent / "chat-assets" / name).read_bytes()
    for name in (
        "markdown-it-15.0.2.min.js",
        "highlight-11.12.0.min.js",
        "github-11.12.0.min.css",
        "github-dark-11.12.0.min.css",
    )
}
# The chat page's brand mark, in its text colors, which the page shows too.
# Browsers, and other clients, ask for a site's icon at /favicon.ico.
FAVICON_SVG = Path(__file__).with_name("favicon.svg").read_bytes()


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


def _progress_flag(body):
    """The request flag of a generation request's return_progress, which only
    a stream takes."""
    value = body.get("return_progress", False)
    if not isinstance(value, bool) or (value and body.get("stream") is not True):
        raise APIError(
            400, "return_progress requires stream: true and must be a boolean"
        )
    return wire.RequestFlag.RETURN_PROGRESS if value else wire.RequestFlag(0)


def _stream_options(body):
    """A Chat or text completion request's stream and its stream_options."""
    stream = body.get("stream", False)
    if stream is None:
        stream = False
    options = body.get("stream_options")
    if options is None:
        options = {}
    if (
        not isinstance(stream, bool)
        or not isinstance(options, dict)
        or not isinstance(options.get("include_usage", False), bool)
    ):
        raise APIError(400, "invalid streaming options")
    return stream, {"include_usage": options.get("include_usage", False)}


@dataclass(frozen=True, slots=True)
class PostRoute:
    """A POST endpoint, as FrontendHandler serves it."""

    # The handler's method that serves a request, given its body and
    # deadline: it answers the request, or prepares the generation the
    # request asks for and returns its job and the method that answers with
    # the job's output.
    method: str
    # How the endpoint's API answers errors.
    errors: ErrorDialect
    # Whether it answers from the prompt alone: it needs no engine, so the
    # engine's state refuses none of its requests, which hold slots of the
    # token-count gate instead of the request gate.
    prompt_only: bool = False


POST_ROUTES = {
    "/v1/chat/completions": PostRoute("_post_chat_completions", OPENAI_ERRORS),
    "/v1/completions": PostRoute("_post_completions", OPENAI_ERRORS),
    "/v1/responses": PostRoute("_post_responses", OPENAI_ERRORS),
    "/v1/messages": PostRoute("_post_messages", ANTHROPIC_ERRORS),
    "/v1/messages/count_tokens": PostRoute(
        "_post_count_tokens", ANTHROPIC_ERRORS, prompt_only=True
    ),
    "/tokenize": PostRoute("_post_tokenize", OPENAI_ERRORS, prompt_only=True),
    "/apply-template": PostRoute(
        "_post_apply_template", OPENAI_ERRORS, prompt_only=True
    ),
    "/v1/judgments": PostRoute("_post_judgments", OPENAI_ERRORS),
    "/v1/systemone": PostRoute("_post_systemone", SYSTEMONE_ERRORS),
}


def path_errors(path):
    """How every error on `path` is answered, whatever its method and
    whether it arises before a handler runs or in one: as the API of its
    POST route answers it, on Anthropic's other paths, those under
    /v1/messages, as that API does, and elsewhere as OpenAI's."""
    route = POST_ROUTES.get(path)
    if route is not None:
        return route.errors
    return ANTHROPIC_ERRORS if path.startswith("/v1/messages/") else OPENAI_ERRORS


@dataclass(slots=True)
class Collected:
    """A generation as FrontendHandler._collect gathered it."""

    reasoning: str
    content: str
    tool_calls: list
    result: NativeResult
    # The output ended inside its reasoning.
    reasoning_open: bool
    # The id of the call the token limit cut, whose arguments are
    # unfinished; None when it cut none.
    cut_call: str | None


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

    def finish(self):
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
        if not super().parse_request():
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
                and (
                    self.route
                    in ("/", "/index.html", "/favicon.ico", "/health", "/ready")
                    or self.route in CHAT_ASSETS
                )
            )
            if not public:
                authenticate(self.headers, self.server.api_key)
        except APIError as error:
            if isinstance(error, OriginRefused):
                self.server.refused_origins.report(error.origin)
            self.close_connection = True
            self._safe_error(error, log=False)
            return False
        return True

    def send_error(self, code, message=None, explain=None):
        # The stdlib's send_error, which answers requests it cannot parse or
        # route and parse_request's 505, writes an HTML page. After a request
        # line it cannot parse, or HTTP/0.9's, request_version is HTTP/0.9,
        # and it writes that page with no status line or headers. Answer as
        # any other error, over HTTP/1.1, a method or version the server does
        # not implement as a server error.
        self.request_version = self.protocol_version
        status = HTTPStatus(code)
        error_code = status.name.lower() if code >= 500 else "invalid_request_error"
        self._safe_error(
            APIError(code, message or status.phrase, error_code), log=False
        )

    @property
    def app(self):
        return self.server.app

    @property
    def route(self):
        """The request's path as routing, authentication and error dialects
        all read it; empty before a request line parses."""
        return _normalize_path(getattr(self, "path", ""))

    @property
    def errors(self):
        """How errors on the request's path are answered (path_errors)."""
        return path_errors(self.route)

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
        if status in RETRY_STATUSES:
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
            self._safe_error(
                APIError(500, "internal server error", "internal_server_error"),
                log=False,
            )
            return
        self._send(status, data, "application/json")

    def _log_api_error(self, code):
        path = "".join(char if char.isprintable() else "?" for char in self.route)
        print_status(f"Error · {code} · {self.command} {path[:256]}", error=True)

    def _safe_error(self, error, *, log=True):
        """Answer with `error` as its path's API answers it, unless the
        response has begun."""
        if self._response_started:
            return
        status, code, payload = self.errors.answer(error)
        if log:
            self._log_api_error(code)
        try:
            self._json(status, payload)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def _upload_seconds(self, length):
        # Inactivity allowed at any point, plus the body at the upload rate.
        return HTTP_IO_TIMEOUT + length / HTTP_UPLOAD_BYTES_PER_SECOND

    def _read_json_body(self, deadline):
        """The request's body, a JSON object, read by `deadline`."""
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
        try:
            text = payload.decode(json.detect_encoding(payload), "surrogatepass")
            payload.clear()
            body = json_codec.loads(text)
        except ValueError:
            # Text that is not JSON, or JSON nested past json_codec.MAX_DEPTH.
            raise RequestValidationError(
                [field_error([], "invalid JSON request body")]
            ) from None
        if not isinstance(body, dict):
            raise RequestValidationError(
                [field_error([], "request body must be an object")]
            )
        return body

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
        if path in ("/", "/index.html", "/favicon.ico") or path in CHAT_ASSETS:
            if not self.server.webui:
                self._safe_error(APIError(404, "not found", "not_found"))
            elif path == "/favicon.ico":
                self._send(200, FAVICON_SVG, "image/svg+xml")
            elif path in CHAT_ASSETS:
                mime = "text/css" if path.endswith(".css") else "text/javascript"
                self._send(200, CHAT_ASSETS[path], f"{mime}; charset=utf-8")
            else:
                self._send(200, CHAT_HTML, "text/html; charset=utf-8")
            return
        if path == "/health":
            self._json(200, {"status": "ok"})
            return
        if path == "/ready":
            ready = self.app.backend.is_ready()
            self._json(
                200 if ready else 503, {"status": "ready" if ready else "unavailable"}
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
        route = POST_ROUTES.get(self.route)
        if route is None:
            self._safe_error(APIError(404, "not found", "not_found"))
            return
        refusal = None if route.prompt_only else self.app.backend.refusal()
        if refusal is not None:
            self._safe_error(refusal, log=False)
            return
        # Hold one ingress slot through body parsing, preparation, and the
        # complete response. Slow uploads/readers cannot accumulate outside
        # the native pending limit, and control endpoints need no such slot.
        admission = (
            self.server.token_counts if route.prompt_only else self.server.requests
        )
        if not admission.acquire():
            self._safe_error(
                APIError(
                    503,
                    "frontend request capacity is exhausted",
                    "frontend_overloaded",
                )
            )
            return
        self._body_reservation = None
        # The native job submitted last, which a failure cancels.
        self._submitted = None
        try:
            generation = self._serve(route, started_at)
            if generation is not None:
                self._generate(*generation, started_at)
        except (BrokenPipeError, ConnectionResetError):
            self._cancel_submitted()
        except TimeoutError:
            self._cancel_submitted()
            error = APIError(408, "HTTP I/O timed out", "request_timeout")
            self._safe_error(error, log=self._submitted is None)
        except APIError as error:
            self._cancel_submitted()
            # The native outcome was already logged; a server-side failure
            # after submission must still reach the console.
            self._safe_error(error, log=self._submitted is None or error.status >= 500)
        except Exception as error:
            self._cancel_submitted()
            log_unexpected(error)
            error = APIError(500, "internal server error", "internal_server_error")
            self._safe_error(error, log=False)
        finally:
            if self._body_reservation is not None:
                self._body_reservation.release()
                self._body_reservation = None
            # The handler outlives the request, through what finish()
            # still reads of an upload refused unread; the job must not, as
            # freeing it returns the input it retains.
            self._submitted = None
            admission.release()
            self.app.latencies.observe("http_request", time.monotonic() - started_at)

    def _serve(self, route, started_at):
        """Read the request and serve it: answer it, or prepare the
        generation it asks for and return its job and the method that answers
        with the job's output."""
        with self.app.latencies.measure("upload"):
            body = self._read_json_body(started_at + self.app.request_timeout)
        serve = getattr(self, route.method)
        return serve(body, self.app.request_deadline(body, started_at))

    def _generate(self, job, respond, started_at):
        """Submit a prepared generation and answer with its output. The
        request's input bytes stay reserved only for what the job retains."""
        self._body_reservation.retain_for(job)
        self._body_reservation = None
        job.latency = RequestLatency(self.app.latencies, started_at)
        self._submit(job)
        respond(job)

    def _submit(self, job):
        """Submit `job` unless its deadline has passed or its client has
        left; from then on a failure cancels it."""
        remaining_request_time(job.deadline)
        if self._client_disconnected():
            raise ConnectionResetError("client disconnected before submission")
        self.app.backend.submit(job)
        self._submitted = job

    def _cancel_submitted(self):
        if self._submitted is not None:
            self.app.backend.cancel(self._submitted)

    def _post_chat_completions(self, body, deadline):
        progress = _progress_flag(body)
        stream, stream_options = _stream_options(body)
        job = self.app.prepare(body, deadline=deadline)
        job.flags |= progress
        if stream:
            return job, partial(
                self._openai_stream, stream_options=stream_options, chat=True
            )
        return job, self._complete

    def _post_completions(self, body, deadline):
        progress = _progress_flag(body)
        stream, stream_options = _stream_options(body)
        job = self.app.prepare_completion(body, deadline=deadline)
        job.flags |= progress
        if stream:
            return job, partial(
                self._openai_stream, stream_options=stream_options, chat=False
            )
        return job, self._text_completion

    def _post_responses(self, body, deadline):
        progress = _progress_flag(body)
        job = self.app.prepare_responses(
            body, deadline=deadline, reserve_input=self._body_reservation.grow
        )
        job.flags |= progress
        if body.get("stream"):
            return job, self._responses_stream
        return job, self._responses_complete

    def _post_messages(self, body, deadline):
        progress = _progress_flag(body)
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
        job.flags |= progress
        if body.get("stream"):
            return job, self._anthropic_stream
        return job, self._anthropic_complete

    def _post_count_tokens(self, body, deadline):
        prompt = anthropic_to_chat_prompt(
            body, thinking_resolver=self.app.thinking_codec.decode
        )
        tokens = self.app.count_tokens(prompt, deadline=deadline)
        self._json(200, {"input_tokens": tokens})

    def _post_tokenize(self, body, deadline):
        self._json(200, {"tokens": self.app.tokenize(body, deadline=deadline)})

    def _post_apply_template(self, body, deadline):
        self._json(200, {"prompt": self.app.apply_template(body, deadline=deadline)})

    def _post_judgments(self, body, deadline):
        job, row = self.app.prepare_judgment(
            body, deadline=deadline, disconnected=self._client_disconnected
        )
        self._submit(job)
        result = self._await_done(job)
        self._json(
            200,
            judgments.judgment_response(self.app.response_model, row, job, result),
        )

    def _post_systemone(self, body, deadline):
        answers = {}
        input_tokens = 0
        entries = self.app.prepare_systemone(
            body, deadline=deadline, disconnected=self._client_disconnected
        )
        for qid, spec, job in entries:
            if job is None:
                answers[qid] = judgments.deterministic_answer(spec)
                continue
            # One admitted job per HTTP request preserves the existing
            # queue bound and lets later questions reuse the state prefix.
            self._submit(job)
            result = self._await_done(job)
            input_tokens += result.prompt_tokens
            answers[qid] = judgments.systemone_answer(
                spec, judgments.softmax(list(result.option_logits))
            )
        self._json(
            200,
            {
                "model": self.app.response_model,
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        )

    def _await_done(self, job):
        """The result of a score job, which emits only start and done."""
        while (event := self._next_event(job))[0] != "done":
            pass
        return event[1]

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
        """The content and calls of the output, and the events the stream
        still owes. An answer without calls is validated against the response
        format unless it was cut. `content` is read only without tools: with
        tools, the projector holds the content."""
        if job.tool_policy is None:
            if not incomplete:
                validate_response_content(content, job.response_validator)
            return content, [], []
        content, tool_calls, events = projector.finish(incomplete)
        if not incomplete and not tool_calls:
            validate_response_content(content, job.response_validator)
        return content, tool_calls, events

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
        policy = job.tool_policy
        splitter = ReasoningSplitter(job.thinking, job.may_call_tools)
        # Output with tools is parsed as it arrives whether it streams or not.
        projector = (
            StreamingToolCallProjector(
                policy, job.public_id, job.response_validator is not None
            )
            if policy is not None
            else None
        )
        reasoning, content, result = [], [], None

        def publish(events):
            if on_text is None:
                return
            for kind, value in events:
                if kind == "tool":
                    on_tool_delta(value)
                else:
                    on_text(kind, value)

        def append(field, text):
            if field == "content" and projector is not None:
                publish(projector.put(text))
            else:
                (reasoning if field == "reasoning_content" else content).append(text)
                publish([(field, text)])

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
        content_text, tool_calls, events = self._finalize_content(
            "".join(content), job, result.reason == "length", projector
        )
        publish(events)
        return Collected(
            "".join(reasoning),
            content_text,
            tool_calls,
            result,
            splitter.reasoning,
            None if projector is None else projector.call_id,
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
        if collected.cut_call is not None:
            # A complete message leaves out the call the token limit cut.
            blocks = [block for block in blocks if block.call_id != collected.cut_call]
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
            self._event_sse("error", self.errors.payload(error))

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
        self._sse(self.errors.payload(error))
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
                self._log_api_error(error.code)
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
                    error=self.errors.payload(error)["error"],
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


class FrontendServer(HTTPServer):
    allow_reuse_address = True
    # Connections the kernel holds until the accept loop takes them. A burst
    # beyond this queue is reset by the kernel, unseen by the server, so ask
    # for as many as uvicorn does; the kernel caps it (128 on macOS).
    # Queued connections take no thread or descriptor.
    request_queue_size = 2048
    # Keep control/catalog capacity separate from generation capacity.
    # Neither gate allocates workers in advance.
    control_connection_capacity = 64
    # Connections answered without their requests read that wait at once,
    # on one thread, for their clients to close. A burst refuses all that
    # arrive while every slot is held, and a client that sends its request
    # after its connection closed is reset before it reads the 503, so
    # enough linger for a burst well beyond the kernel's queue, within the
    # descriptors the server raises its limit to.
    refused_connection_capacity = 1024

    def __init__(
        self,
        address,
        app,
        bind_and_activate=True,
        request_capacity=serve_options.DEFAULT_QUEUE_SIZE,
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
            request_capacity + self.control_connection_capacity,
            self._serve_connection,
            self.refused_connection_capacity,
            REFUSED_LINGER_SECONDS,
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

    def server_activate(self):
        super().server_activate()
        # accept() then returns at once when the kernel holds no connection.
        self.socket.setblocking(False)

    def _handle_request_noblock(self):
        """Take every connection the kernel holds at each wakeup of the accept
        loop, as asyncio's accept does, not one: a burst arrives faster than a
        select per connection takes it, and macOS resets what overflows the
        kernel's queue."""
        while True:
            try:
                request, client_address = self.get_request()
            except OSError:
                return
            # macOS gives a connection accepted from a nonblocking socket that
            # socket's mode.
            request.setblocking(True)
            if not self.verify_request(request, client_address):
                self.shutdown_request(request)
                continue
            try:
                self.process_request(request, client_address)
            except Exception:
                self.handle_error(request, client_address)
                self.shutdown_request(request)
            except BaseException:
                self.shutdown_request(request)
                raise

    def process_request(self, request, client_address):
        if not self.connections.admit(
            request, client_address, time.monotonic() + HTTP_IO_TIMEOUT
        ):
            self.connections.refuse(request)

    def _serve_connection(self, request, client_address):
        """Serve the connection `request`, ready for its thread, on a
        connection worker, as ThreadingMixIn's thread for a request does."""
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def shutdown_request(self, request):
        self.connections.release(request)
        super().shutdown_request(request)

    def server_close(self):
        super().server_close()
        self.connections.stop()
        self.connections.idle.wait(min(2.0, HTTP_IO_TIMEOUT))

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_root",
        metavar="MODEL_DIRECTORY",
        help="installed model directory holding target/ and draft/",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--model",
        type=serve_options.parse_model_id,
        required=True,
        metavar="OWNER/REPO",
    )
    parser.add_argument("--port", type=int, default=serve_options.DEFAULT_PORT)
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
    if args.persistent_cache:
        command.extend(
            ("--cache-dir", str(args.cache_dir or serve_options.DEFAULT_CACHE_DIR))
        )
    if args.kv_format != "int8":
        command.extend(("--kv-format", args.kv_format))
    if args.idle_release is not None:
        command.extend(
            ("--idle-release", serve_options.idle_release_text(args.idle_release))
        )
    if args.disable_ane:
        command.extend(("--ane", "off"))
    if args.decode_share is not None:
        command.extend(("--decode-share", str(args.decode_share)))
    if args.max_image_pixels != image_input.MAX_PIXELS:
        command.extend(
            ("--max-image-patches", str(image_input.max_patches(args.max_image_pixels)))
        )
    if args.allow_idle_sleep:
        command.extend(("--idle-sleep", "allow"))
    return command


def _raise_descriptor_limit():
    """Raise the soft descriptor limit to DESCRIPTOR_LIMIT, or to the hard
    limit when that is lower; a higher soft limit stays."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = DESCRIPTOR_LIMIT
    if hard != resource.RLIM_INFINITY:
        wanted = min(wanted, hard)
    if soft < wanted:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))


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
        _raise_descriptor_limit()
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

"""HTTP routing and serving-process startup. One table names each endpoint's
handler; each API's handlers, request conversion and answers live in its
module in dialects/."""

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
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

from transformers import AutoTokenizer

from install import assembly
from install.models import ModelError

from . import dialects, json_codec, serve_options, serve_record
from . import runtime as engine_runtime
from .backend import (
    NativeBackend,
    NativeResult,
    ThinkingEndError,
    thinking_end_tokens,
)
from .chat_templates import ChatTemplateError, ChatTemplates
from .connections import ConnectionSlots, HttpAdmission, RequestBodyReservation
from .constraints import (
    ConstraintFactory,
    TokenizerContractError,
    check_capabilities,
    validate_tokenizer,
)
from .diagnostics import console_stats, log_unexpected, print_request, print_status
from .errors import (
    OPENAI_ERRORS,
    RETRY_STATUSES,
    APIError,
    ErrorDialect,
    RequestValidationError,
    field_error,
    remaining_request_time,
)
from .frontend import Frontend
from .http_security import (
    OriginRefused,
    RefusedOriginLog,
    authenticate,
    validate_headers,
)
from .latency import RequestLatency
from .metrics import prometheus_metrics
from .origins import ANY_ORIGIN
from .output import (
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
# How long an answered connection may take its client to close.
ANSWERED_LINGER_SECONDS = 2.0
# The soft descriptor limit the server raises its own to: macOS starts a
# process with 256, fewer than its connection slots, the answered connections
# lingering after a burst and the engine's pipes and files can need. 10240 is
# macOS's OPEN_MAX.
DESCRIPTOR_LIMIT = 10240
# The status of a server that stopped restarting its engine, apart from a
# stop's 0 and a failed start's 1: what supervises the server, as launchd's
# KeepAlive does, decides whether to start it again. 69 is sysexits'
# EX_UNAVAILABLE.
ENGINE_STOPPED_STATUS = os.EX_UNAVAILABLE
SSE_KEEPALIVE_SECONDS = 2.0
ROOT = Path(__file__).parents[1]
CHAT_HTML = Path(__file__).with_name("chat.html").read_bytes()
CHAT_ASSETS = {
    f"/chat-assets/{name}": (Path(__file__).parent / "chat-assets" / name).read_bytes()
    for name in (
        "markdown-it-15.0.2.min.js",
        "highlight-11.12.0.min.js",
        "markdown-links.js",
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


@dataclass(frozen=True, slots=True)
class Route:
    """An endpoint, as FrontendHandler serves it. Its errors and
    request_id_header hold for its whole path, so every route on one path
    states the same ones: path_errors and _send read the first."""

    # What serves a request, called with the FrontendHandler that reads it.
    # A POST route's also takes the request's body and deadline: it answers
    # the request, or prepares the generation the request asks for and
    # returns its job and the function that answers with the job's output.
    # Another route's takes what its path pattern captures.
    handler: Callable
    # How the endpoint's API answers errors, on its path by any method.
    errors: ErrorDialect = OPENAI_ERRORS
    # Whether it answers without the API key: the chat page and the probes.
    public: bool = False
    # Whether it answers from the prompt alone: it needs no engine, so the
    # engine's state refuses none of its requests, which hold slots of the
    # token-count gate instead of the request gate.
    prompt_only: bool = False
    # Whether it takes no fields: a request carries no body or an empty JSON
    # object.
    fieldless: bool = False
    # A header that gives every answer on its path, by any method and errors
    # too, a fresh request id: TypeSafe's SDK reads one on the routes it
    # calls.
    request_id_header: str | None = None


@dataclass(slots=True)
class Collected:
    """A generation as FrontendHandler.collect gathered it."""

    reasoning: str
    content: str
    tool_calls: list
    result: NativeResult
    # The output ended inside its reasoning.
    reasoning_open: bool
    # The token limit or a stop sequence ended the output inside a call.
    cut_in_call: bool
    # The id of that call, whose arguments are unfinished; None when it cut
    # none, or none whose name it had read.
    cut_call: str | None

    @property
    def cut(self):
        """Whether the output's end cut its last block short: the token limit
        cuts any block, a stop sequence only a call, as text may end where
        the client asks. A call cut at its name has no block, and a stop
        sequence there leaves the block before it whole."""
        return self.result.reason == "length" or self.cut_call is not None

    @property
    def called(self):
        """Whether the output's tool calls count as made: the token limit did
        not end the output, and it cut none."""
        return (
            bool(self.tool_calls)
            and not self.cut_in_call
            and self.result.reason != "length"
        )


class FrontendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    methods = "GET, HEAD, POST, DELETE, OPTIONS"

    def setup(self):
        self.response_started = False
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
            route, _ = find_route(self.route, self.command)
            public = self.command == "OPTIONS" or (route is not None and route.public)
            if not public:
                authenticate(self.headers, self.server.api_key)
        except APIError as error:
            if isinstance(error, OriginRefused):
                self.server.refused_origins.report(error.origin)
            self.close_connection = True
            self.safe_error(error, log=False)
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
        self.safe_error(APIError(code, message or status.phrase, error_code), log=False)

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

    def _send(self, status, data, content_type, retry_after=1):
        self.send_response(status)
        if status in RETRY_STATUSES:
            self.send_header("Retry-After", str(retry_after))
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        route, _ = find_route(self.route)
        if route is not None and route.request_id_header is not None:
            self.send_header(route.request_id_header, f"req_{secrets.token_hex(12)}")
        self.response_started = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_json(self, status, payload, retry_after=1):
        try:
            data = json_codec.encode(payload)
        except json_codec.JSONEncodingError as error:
            log_unexpected(error)
            self.safe_error(
                APIError(500, "internal server error", "internal_server_error"),
                log=False,
            )
            return
        self._send(status, data, "application/json", retry_after)

    def _log_api_error(self, code):
        path = "".join(char if char.isprintable() else "?" for char in self.route)
        print_status(f"Error · {code} · {self.command} {path[:256]}", error=True)

    def safe_error(self, error, *, log=True):
        """Answer with `error` as its path's API answers it, unless the
        response has begun."""
        if self.response_started:
            return
        status, code, payload = self.errors.answer(error)
        if log:
            self._log_api_error(code)
        try:
            self.send_json(status, payload, error.retry_after)
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
        self._dispatch("GET")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method):
        """Serve a request of `method`, which carries no body, by its route."""
        route, captured = find_route(self.route, method)
        if route is None:
            self.safe_error(APIError(404, "not found", "not_found"))
        else:
            route.handler(self, *captured)

    def _get_page(self):
        path = self.route
        if not self.server.webui:
            self.safe_error(APIError(404, "not found", "not_found"))
        elif path == "/favicon.ico":
            self._send(200, FAVICON_SVG, "image/svg+xml")
        elif path in CHAT_ASSETS:
            mime = "text/css" if path.endswith(".css") else "text/javascript"
            self._send(200, CHAT_ASSETS[path], f"{mime}; charset=utf-8")
        else:
            self._send(200, CHAT_HTML, "text/html; charset=utf-8")

    def _get_health(self):
        self.send_json(200, {"status": "ok"})

    def _get_ready(self):
        ready = self.app.backend.is_ready()
        self.send_json(
            200 if ready else 503, {"status": "ready" if ready else "unavailable"}
        )

    def _get_status(self):
        self.send_json(200, self.server.status())

    def _get_metrics(self):
        self._send(
            200,
            prometheus_metrics(self.server.status()).encode(),
            "text/plain; version=0.0.4; charset=utf-8",
        )

    def _models(self):
        return [
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "splash",
                "max_model_len": self.app.max_context,
                "context_length": self.app.max_context,
                "vision": self.app.vision,
                "input_modalities": self.app.input_modalities,
                "reasoning": self.app.reasoning,
                **(
                    {"root": self.app.response_model}
                    if name != self.app.response_model
                    else {}
                ),
            }
            for name in self.app.model_names
        ]

    def _get_models(self):
        models = self._models()
        self.send_json(
            200,
            {
                "object": "list",
                "data": models,
                "models": dialects.scoring.typed_models(models),
            },
        )

    def _get_model(self, name):
        model = next((item for item in self._models() if item["id"] == name), None)
        if model is None:
            self.safe_error(APIError(404, "model not found", "model_not_found"))
        else:
            self.send_json(200, model)

    def do_POST(self):
        started_at = time.monotonic()
        route, _ = find_route(self.route, "POST")
        if route is None:
            self.safe_error(APIError(404, "not found", "not_found"))
            return
        refusal = None if route.prompt_only else self.app.backend.refusal()
        if refusal is not None:
            self.safe_error(refusal, log=False)
            return
        # Hold one ingress slot through body parsing, preparation, and the
        # complete response. Slow uploads/readers cannot accumulate outside
        # the native pending limit, and control endpoints need no such slot.
        admission = (
            self.server.token_counts if route.prompt_only else self.server.requests
        )
        if not admission.acquire():
            self.safe_error(
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
            self.safe_error(error, log=self._submitted is None)
        except APIError as error:
            self._cancel_submitted()
            # The native outcome was already logged; a server-side failure
            # after submission must still reach the console.
            self.safe_error(error, log=self._submitted is None or error.status >= 500)
        except Exception as error:
            self._cancel_submitted()
            log_unexpected(error)
            error = APIError(500, "internal server error", "internal_server_error")
            self.safe_error(error, log=False)
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
        generation it asks for and return its job and the function that
        answers with the job's output."""
        if route.fieldless and not self._unread_body:
            body = {}
        else:
            with self.app.latencies.measure("upload"):
                body = self._read_json_body(started_at + self.app.request_timeout)
            # A field set to null is one left out, as clients that write every
            # field they do not set as null mean it.
            body = {key: value for key, value in body.items() if value is not None}
            if route.fieldless and body:
                raise APIError(400, "this endpoint takes no fields")
        return route.handler(self, body, self.app.request_deadline(body, started_at))

    def _generate(self, job, respond, started_at):
        """Submit a prepared generation and answer with its output. The
        request's input bytes stay reserved only for what the job retains."""
        self._body_reservation.retain_for(job)
        self._body_reservation = None
        job.latency = RequestLatency(self.app.latencies, started_at)
        self.submit(job)
        respond(job)

    def submit(self, job):
        """Submit `job` unless its deadline has passed or its client has
        left; from then on a failure cancels it."""
        remaining_request_time(job.deadline)
        if self.client_disconnected():
            raise ConnectionResetError("client disconnected before submission")
        self.app.backend.submit(job)
        self._submitted = job

    def reserve_input(self, size):
        """Reserve `size` more bytes of the request body budget for input the
        request's preparation adds, such as the stored history it continues."""
        self._body_reservation.grow(size)

    def _cancel_submitted(self):
        if self._submitted is not None:
            self.app.backend.cancel(self._submitted)

    def _post_tokenize(self, body, deadline):
        self.send_json(200, {"tokens": self.app.tokenize(body, deadline=deadline)})

    def _post_apply_template(self, body, deadline):
        self.send_json(
            200, {"prompt": self.app.apply_template(body, deadline=deadline)}
        )

    def _post_memory_release(self, body, deadline):
        weights = self.app.backend.release_weights().get("weights") or {}
        # A restore leaves the weights released until its last image is back.
        if weights.get("released") is True and weights.get("restoring") is False:
            self.send_json(200, {"released": True, "weights": weights})
            return
        # An answer about the engine's state, not a failure: the console does
        # not log it.
        self.safe_error(
            APIError(
                409,
                "the engine is serving requests or restoring its weights; it "
                "releases them only while idle",
                "engine_busy",
            ),
            log=False,
        )

    def next_event(self, job, on_idle=None):
        while True:
            if self.client_disconnected():
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

    def client_disconnected(self):
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

    def _finalize_content(self, content, job, result, projector, cut_in_call):
        """The content and calls of the output, and the events the stream
        still owes. The token limit cuts the output wherever it ends, and a
        stop sequence the call it ends inside; text before a stop sequence is
        the text the client asked for. An answer without calls is validated
        against the response format only when the model ended it. `content`
        is read only without a projector, which otherwise holds it."""
        ended = result.reason != "length" and result.stop_sequence is None
        if projector is None:
            if ended:
                validate_response_content(content, job.response_validator)
            return content, [], []
        content, tool_calls, events = projector.finish(
            result.reason == "length" or cut_in_call
        )
        if ended and not tool_calls:
            validate_response_content(content, job.response_validator)
        return content, tool_calls, events

    def collect(
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
        splitter = ReasoningSplitter(job.thinking, job.may_call_tools, job.chat_format)
        # A chat answer is parsed as it arrives whether it streams or not:
        # its calls where tools may be called, and in any answer, a </think>
        # the model writes after its reasoning ended is dropped. A text
        # completion, without a chat template, is its text as written.
        projector = (
            StreamingToolCallProjector(
                job.tool_policy,
                job.public_id,
                job.response_validator is not None,
                job.chat_format.call_syntax,
            )
            if job.tool_policy is not None or job.reasoning is not None
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
            kind, value = self.next_event(job, on_idle)
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
        cut_in_call = (
            projector is not None
            and projector.in_call
            and (result.reason == "length" or result.stop_sequence is not None)
        )
        content_text, tool_calls, events = self._finalize_content(
            "".join(content), job, result, projector, cut_in_call
        )
        publish(events)
        return Collected(
            "".join(reasoning),
            content_text,
            tool_calls,
            result,
            splitter.reasoning,
            cut_in_call,
            None if projector is None else projector.call_id,
        )

    def start_event_stream(self):
        if self.response_started:
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.response_started = True

    def _write_sse(self, frame):
        self.wfile.write(frame)
        self.wfile.flush()
        self._last_sse_write = time.monotonic()

    def sse(self, payload):
        data = (
            payload.encode("utf-8")
            if isinstance(payload, str)
            else json_codec.encode(payload)
        )
        self._write_sse(b"data: " + data + b"\n\n")

    def event_sse(self, event, payload):
        data = json_codec.encode(payload)
        self._write_sse(f"event: {event}\ndata: ".encode() + data + b"\n\n")

    def sse_keepalive(self):
        # An SSE comment is traffic, so socket read timeouts and proxies do
        # not take a long prefill or resource wait for a dead connection, but
        # event decoders skip it and clients that time out on missing data
        # events ignore it. Streams send it only where no data event fits: the
        # chat and text completion streams until the request starts, the
        # Responses stream once output has begun.
        self.start_event_stream()
        self._write_sse(b": splash-keepalive\n\n")

    def guarded_stream(self, job, run, send_error):
        """Run a dialect's stream. A failure cancels the job; once the
        response has begun, send_error writes it as the dialect's error
        event, and before, the handler answers with its HTTP status."""
        try:
            run()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.app.backend.cancel(job)
        except APIError as error:
            self.app.backend.cancel(job)
            if not self.response_started:
                raise
            if error.status >= 500:
                self._log_api_error(error.code)
            try:
                send_error(error)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
        except Exception as error:
            self.app.backend.cancel(job)
            if not self.response_started:
                raise
            log_unexpected(error)
            try:
                send_error(
                    APIError(500, "internal server error", "internal_server_error")
                )
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass


_MODEL_PATH = re.compile(r"/v1/models/(.*)", re.DOTALL)
_PAGE = Route(FrontendHandler._get_page, public=True)

# Every endpoint, by its method and its path or path pattern.
ROUTES = {
    **{
        ("GET", path): _PAGE
        for path in ("/", "/index.html", "/favicon.ico", *CHAT_ASSETS)
    },
    ("GET", "/health"): Route(FrontendHandler._get_health, public=True),
    ("GET", "/ready"): Route(FrontendHandler._get_ready, public=True),
    ("GET", "/status"): Route(FrontendHandler._get_status),
    ("GET", "/metrics"): Route(FrontendHandler._get_metrics),
    ("GET", dialects.responses.RESPONSE_PATH): Route(dialects.responses.get),
    ("DELETE", dialects.responses.RESPONSE_PATH): Route(dialects.responses.delete),
    ("GET", "/v1/models"): Route(
        FrontendHandler._get_models,
        request_id_header=dialects.scoring.REQUEST_ID_HEADER,
    ),
    ("GET", _MODEL_PATH): Route(
        FrontendHandler._get_model, request_id_header=dialects.scoring.REQUEST_ID_HEADER
    ),
    ("POST", "/v1/chat/completions"): Route(dialects.openai.post_chat),
    ("POST", "/v1/chat/completions/count_tokens"): Route(
        dialects.openai.post_chat_count_tokens, prompt_only=True
    ),
    ("POST", "/v1/completions"): Route(dialects.openai.post_completion),
    ("POST", "/v1/responses"): Route(dialects.responses.post),
    ("POST", "/v1/messages"): Route(
        dialects.anthropic.post_messages, dialects.anthropic.ERRORS
    ),
    ("POST", "/v1/messages/count_tokens"): Route(
        dialects.anthropic.post_count_tokens,
        dialects.anthropic.ERRORS,
        prompt_only=True,
    ),
    ("POST", "/tokenize"): Route(FrontendHandler._post_tokenize, prompt_only=True),
    ("POST", "/apply-template"): Route(
        FrontendHandler._post_apply_template, prompt_only=True
    ),
    ("POST", "/v1/judgments"): Route(dialects.scoring.post_judgments),
    ("POST", "/v1/systemone"): Route(
        dialects.scoring.post_systemone,
        dialects.scoring.ERRORS,
        request_id_header=dialects.scoring.REQUEST_ID_HEADER,
    ),
    ("POST", "/v1/memory/release"): Route(
        FrontendHandler._post_memory_release, fieldless=True
    ),
}


def find_route(path, method=None):
    """The route serving `method` on `path`, a route of any method when None,
    and what its path pattern captures; (None, ()) when no route does. HEAD
    is served by GET's route."""
    if method == "HEAD":
        method = "GET"
    for (route_method, key), route in ROUTES.items():
        if method not in (None, route_method):
            continue
        if isinstance(key, str):
            if key == path:
                return route, ()
        elif match := key.fullmatch(path):
            return route, match.groups()
    return None, ()


def path_errors(path):
    """How every error on `path` is answered, whatever its method and
    whether it arises before a handler runs or in one: as the API of its
    route answers it, on Anthropic's other paths, those under /v1/messages,
    as that API does, and elsewhere as OpenAI's."""
    route, _ = find_route(path)
    if route is not None:
        return route.errors
    if path.startswith("/v1/messages/"):
        return dialects.anthropic.ERRORS
    return OPENAI_ERRORS


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
    # Answered connections, refused or served, that wait at once, on one
    # thread, for their clients to close. A burst refuses all that arrive
    # while every slot is held, and a client that sends its request after
    # its connection closed is reset before it reads the 503, so enough
    # linger for a burst well beyond the kernel's queue, within the
    # descriptors the server raises its limit to.
    answered_connection_capacity = 1024

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
            self.answered_connection_capacity,
            ANSWERED_LINGER_SECONDS,
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
        status["console"] = console_stats()
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
        self.connections.close(request)

    def server_close(self):
        super().server_close()
        self.connections.stop()
        self.connections.idle.wait(min(2.0, HTTP_IO_TIMEOUT))

    def handle_error(self, request, client_address):
        # A client that left is no error. Any other is one console line, as a
        # request's unexpected error is: socketserver's own writes a traceback
        # straight to stderr, on the connection worker.
        error = sys.exc_info()[1]
        if not isinstance(error, (BrokenPipeError, ConnectionResetError)):
            log_unexpected(error)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_root",
        metavar="MODEL_DIRECTORY",
        help="the directory of a model Splash prepared, or a selection link to "
        "one: the server holds and serves the directory it names at startup",
    )
    parser.add_argument(
        "--model",
        type=serve_options.parse_model_id,
        required=True,
        metavar="OWNER/REPO",
    )
    parser.add_argument("--port", type=int, default=serve_options.DEFAULT_PORT)
    parser.add_argument("--binary", default=str(ROOT / "build" / "splash"))
    # The launcher's, not a serve option: where the app follows a start and
    # reads why one failed (serve_record).
    parser.add_argument("--serve-record", type=Path, metavar="PATH")
    serve_options.add_serve_arguments(parser)
    args = parser.parse_args(argv)
    serve_options.check_serve_arguments(parser, args)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    return args


def _native_command(args, model_root, relaunch_context=None):
    """The engine's command line serving model_root, a relaunch's with
    `relaunch_context`, the context the first engine served. A relaunch is
    planned as the first engine was, for --max-context or the automatic
    context, and serves what its plan holds of the first engine's context,
    where its own automatic one would follow a Neural Engine split calibrated
    anew."""
    command = [
        args.binary,
        "serve-native",
        str(model_root),
        *serve_options.engine_argv(args),
    ]
    if relaunch_context is not None:
        command.extend(("--relaunch-context", str(relaunch_context)))
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


def _stop_with(pid):
    """Stops the server with SIGTERM once process `pid` has ended: one stop,
    which a server already stopping ignores.

    The desktop app passes its own process ID as SPLASH_APP_PID, so a model
    it started gives its memory back when the app crashes, rather than at
    the app's next launch. A process already gone stops the server at once:
    kqueue returns its watch at once, as an error event.
    """

    def watch():
        queue = select.kqueue()
        exited = select.kevent(
            pid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
            fflags=select.KQ_NOTE_EXIT,
        )
        queue.control([exited], 1, None)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=watch, name="stop-with-app", daemon=True).start()


class ListenError(Exception):
    """The HTTP server cannot listen at its address."""


def _start_failed(args, error, failure):
    """End a start that failed, saying why in the log and, for the app, in
    the launcher's serve record (serve_record)."""
    print_status(f"Error · {error}", error=True)
    serve_record.failed(args.serve_record, failure)
    raise SystemExit(1) from None


def main():
    args = parse_args()
    server = None
    runtime = None
    backend = None
    # Why the backend stopped restarting the engine, once it has.
    stopped = None

    def stop_serving(reason):
        nonlocal stopped
        stopped = reason
        os.kill(os.getpid(), signal.SIGTERM)

    # A server started in the background from a non-interactive shell inherits
    # SIGINT as ignored and Python then leaves it alone; install both stop
    # signals explicitly so scripts and supervisors can interrupt it.
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    try:
        # The launcher blocks both across its exec: one sent while this module
        # imported arrives here and ends the startup cleanly.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
        if app := os.environ.get("SPLASH_APP_PID"):
            _stop_with(int(app))
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
        try:
            server.server_bind()
        except OSError as error:
            raise ListenError(f"unable to start HTTP server: {error}") from None
        if ANY_ORIGIN in args.allowed_origin and args.api_key is None:
            print_status(
                "Warning · --allowed-origin '*' without --api-key lets every web "
                "page open in a browser that reaches this server use it",
                error=True,
            )
        thinking_codec = ThinkingCodec(load_thinking_key())
        print_status(f"Loading · {args.model}")
        serve_record.entered(args.serve_record, "loading")
        # Nothing reads `held`: it keeps the assembly held until the server
        # exits, so that no installation collects the assembly its tokenizer
        # and engines, a relaunched one too, serve.
        model_root, held = assembly.hold(Path(args.model_root))
        # The family the installation found the model to be, whose chat
        # format its output is read in.
        family = assembly.family(model_root)
        if family is None:
            raise ModelError(
                f"{args.model} is not prepared as a model of a family this "
                "release serves: prepare it again"
            )
        # A tokenizer the loader cannot read fails with whatever its parsers
        # raise, such as a KeyError for a tokenizer.json without added_tokens:
        # each is the model's.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_root / "tokenizer", local_files_only=True, trust_remote_code=False
            )
        except Exception as error:
            raise ModelError(
                f"cannot load the tokenizer of {args.model}: "
                f"{type(error).__name__}: {error}"
            ) from error
        validate_tokenizer(tokenizer, family.chat_format)
        thinking_end = thinking_end_tokens(
            tokenizer, args.thinking_end_text, family.chat_format
        )
        chat_templates = ChatTemplates(tokenizer)
        print_status(f"Chat template · {chat_templates.describe()}")
        # Unset, a budget ends thinking as the template closes it, so that the
        # next request renders that close back as the tokens the reply ended
        # its thinking with; Qwen's templates still trim a space or line break
        # the cut leaves before it.
        if thinking_end is None:
            thinking_end = thinking_end_tokens(
                tokenizer, chat_templates.select(None).thinking_end, family.chat_format
            )
        runtime = engine_runtime.MultiplexedRuntime(
            _native_command(args, model_root),
            relaunch_command=lambda first: _native_command(
                args, model_root, first.max_context_tokens
            ),
            pending_limit=args.queue_size,
        )
        runtime.on_warming = lambda: serve_record.entered(args.serve_record, "warming")
        backend = NativeBackend(
            runtime,
            tokenizer,
            request_logger=print_request,
            thinking_end=thinking_end,
        )
        # A server that serves nothing more stops as on SIGTERM and exits
        # ENGINE_STOPPED_STATUS below.
        backend.on_stopped = stop_serving
        if not runtime.wait_ready():
            raise engine_runtime.EngineUnhealthy("native runtime did not become ready")
        # The engine is authoritative for what it serves: it refuses a
        # --max-context it cannot serve.
        served = runtime.capabilities
        check_capabilities(tokenizer, family.chat_format, served)
        app = Frontend(
            tokenizer,
            backend,
            args.model,
            # No deadline unless given, as in vLLM and SGLang; a request still
            # ends when its client disconnects.
            math.inf if args.request_timeout is None else args.request_timeout,
            constraint_factory=ConstraintFactory(tokenizer, served),
            chat_templates=chat_templates,
            chat_format=family.chat_format,
            sampling_defaults=asdict(family.sampling),
            max_image_pixels=args.max_image_pixels,
            thinking_codec=thinking_codec,
            response_store=dialects.responses.ResponseStore(),
            served_model_names=args.served_model_name,
            announce_served_name=args.announce_served_name,
            default_reasoning_effort=args.default_reasoning_effort,
            default_thinking_token_budget=args.thinking_token_budget,
        )
        server.app = app
        server.server_activate()
        # The app takes the server for ready from its record alone: one that
        # cannot say so has not started (serve_record).
        serve_record.ready(args.serve_record)
        address = f"http://{args.host}:{server.server_port}"
        tokens = served.max_context_tokens
        context = f"{tokens // 1024}K" if tokens % 1024 == 0 else f"{tokens:,}"
        mode = "" if served.vision else " · text only"
        print_status(f"Ready · {args.model} · context {context}{mode} · {address}")
        server.serve_forever()
    except engine_runtime.EngineStartFailed as error:
        _start_failed(args, error, serve_record.engine_failure(error.event, str(error)))
    except ListenError as error:
        _start_failed(args, error, serve_record.failure(serve_record.PORT, str(error)))
    except (
        ModelError,
        ChatTemplateError,
        ThinkingEndError,
        TokenizerContractError,
    ) as error:
        _start_failed(args, error, serve_record.failure(serve_record.MODEL, str(error)))
    except (engine_runtime.EngineRuntimeError, ThinkingKeyError, OSError) as error:
        _start_failed(
            args, error, serve_record.failure(serve_record.INTERNAL, str(error))
        )
    except serve_record.NotRecorded as error:
        # The record that cannot say ready cannot say why either: the app
        # takes the exit for the failure, which the log words.
        print_status(f"Error · {error}", error=True)
        raise SystemExit(1) from None
    except Exception as error:
        # An error no clause above names is a bug: the record still names the
        # failure, so that the app reports it instead of a crash, and the
        # traceback reaches the log.
        serve_record.failed(
            args.serve_record,
            serve_record.failure(
                serve_record.INTERNAL, f"{type(error).__name__}: {error}"
            ),
        )
        raise
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
            # Stop accepting before the engine stops, which can take seconds:
            # a client that connects meanwhile is refused at once instead of
            # waiting in the backlog to be reset. server_close still answers
            # the requests in flight.
            if server is not None:
                server.socket.close()
            if backend is not None:
                print_status("Stopping · releasing engine resources")
                backend.close()
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if server is not None:
                server.server_close()
    if stopped is not None:
        serve_record.failed(
            args.serve_record, serve_record.failure(serve_record.STOPPED, stopped)
        )
        raise SystemExit(ENGINE_STOPPED_STATUS)


if __name__ == "__main__":
    main()

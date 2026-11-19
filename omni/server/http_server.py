"""The HTTP server the audio server builds on: the request handler base
(header validation, authentication, body reading charged to the shared
budget, admission, the wait for a job's events and the cleanup a failure
owes), the connection-slotted server, and the process helpers its startup
uses."""

import json
import os
import queue
import resource
import secrets
import select
import socket
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote

from . import json_codec, serve_options
from .backend import remaining_request_time
from .connections import ConnectionSlots, HttpAdmission, RequestBodyReservation
from .diagnostics import CONSOLE, log_unexpected, print_status
from .errors import (
    OPENAI_ERRORS,
    RETRY_STATUSES,
    APIError,
    ErrorDialect,
    RequestValidationError,
    field_error,
)
from .http_security import (
    OriginRefused,
    RefusedOriginLog,
    authenticate,
    validate_headers,
)
from .origins import ANY_ORIGIN

# Request bodies held at once, from upload through preparation and, for what
# a request retains of them, until it ends, take at most max(this, twice
# --max-request-size) bytes, so two of the largest requests always fit.
DEFAULT_REQUEST_BODY_BUDGET = 512 * 1024 * 1024
HTTP_IO_TIMEOUT = 30.0
# A client that vanishes without closing (a sleeping laptop, a dropped
# network) sends nothing a non-streaming request reads, which may run for
# minutes; keepalive probes the idle connection and drops it 60 s after its
# last data (30 s idle, then 3 probes 10 s apart), and the request is
# cancelled as for a disconnect.
TCP_KEEPALIVE_OPTIONS = (
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30),
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
)
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
NATIVE_START_TIMEOUT = 600.0
# The routes a client reaches without the API key: the liveness probes.
PUBLIC_ROUTES = ("/health", "/ready")


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
class PostRoute:
    """A POST endpoint, as FrontendHandler serves it."""

    # The handler's method that serves a request, given the time it
    # started: it reads the body, submits what the request asks for and
    # answers.
    method: str
    # How the endpoint's API answers errors.
    errors: ErrorDialect = OPENAI_ERRORS


class FrontendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    methods = "GET, HEAD, POST, OPTIONS"
    # The POST endpoints served, by path.
    post_routes = {}
    # The response headers a page of an admitted origin may read.
    exposed_headers = "Retry-After, WWW-Authenticate"

    def setup(self):
        self._response_started = False
        # What the response owes its request's origin, once the request has
        # passed validate_headers.
        self._allow_origin = None
        self._unread_body = 0
        super().setup()
        self.connection.settimeout(HTTP_IO_TIMEOUT)
        for level, option, value in TCP_KEEPALIVE_OPTIONS:
            self.connection.setsockopt(level, option, value)

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
                self.command in ("GET", "HEAD") and self.route in PUBLIC_ROUTES
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
        """How errors on the request's path are answered: as the API of its
        POST route answers them, and elsewhere as OpenAI's."""
        route = self.post_routes.get(self.route)
        return OPENAI_ERRORS if route is None else route.errors

    def end_headers(self):
        # A browser hands a page the response from another origin only when
        # the response names that origin, so every response to an admitted
        # origin does, errors too, and exposes the retry and authentication
        # hints, which CORS hides by default.
        if self._allow_origin is not None:
            self.send_header("Access-Control-Allow-Origin", self._allow_origin)
            if self._allow_origin != ANY_ORIGIN:
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Expose-Headers", self.exposed_headers)
        super().end_headers()

    def _send(self, status, data, content_type, headers=()):
        self.send_response(status)
        if status in RETRY_STATUSES:
            self.send_header("Retry-After", "1")
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
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

    def _require_json_content_type(self):
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

    def _read_json_body(self, deadline):
        """The request's body, a JSON object, read by `deadline`."""
        payload = self._read_body(deadline, self._require_json_content_type)
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

    def _read_body(self, deadline, require_content_type):
        """The request's body, read by `deadline` and charged to the body
        budget until the handler releases its reservation; a stated length
        above --max-request-size is refused unread. require_content_type
        refuses a body of the wrong media type before any of it is read."""
        if self.headers.get_all("Transfer-Encoding"):
            raise APIError(400, "transfer encoding is not supported")
        encodings = self.headers.get_all("Content-Encoding", [])
        if len(encodings) > 1 or (
            encodings and encodings[0].strip().lower() != "identity"
        ):
            raise APIError(415, "content encoding is not supported")
        require_content_type()
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
        return payload

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
        return "Splash Omni"

    def do_GET(self):
        # The serving handler routes GET; the base answers nothing.
        self._safe_error(APIError(404, "not found", "not_found"))

    def do_POST(self):
        started_at = time.monotonic()
        route = self.post_routes.get(self.route)
        if route is None:
            self._safe_error(APIError(404, "not found", "not_found"))
            return
        refusal = self.app.backend.refusal()
        if refusal is not None:
            self._safe_error(refusal, log=False)
            return
        # Hold one ingress slot through body parsing, preparation, and the
        # complete response. Slow uploads/readers cannot accumulate outside
        # the native pending limit, and control endpoints need no such slot.
        admission = self.server.requests
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
            getattr(self, route.method)(started_at)
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

    def _next_event(self, job):
        """The job's next event; a queued failure is raised, a client that
        left or a deadline that passed cancels the job."""
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
                continue
            if event[0] == "error":
                raise event[1]
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
        except (ConnectionError, TimeoutError):
            # A connection keepalive dropped fails its read with ETIMEDOUT.
            return True

    def _guarded_stream(self, job, run, send_error):
        """Run `run`, which streams the job's output: a failure before the
        response began is raised to do_POST, which answers it; after, it is
        told with `send_error`, as the stream can tell it."""
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


class FrontendServer(HTTPServer):
    allow_reuse_address = True
    # Connections the kernel holds until the accept loop takes them. A burst
    # beyond this queue is reset by the kernel, unseen by the server, so ask
    # for as many as uvicorn does; the kernel caps it (128 on macOS).
    # Queued connections take no thread or descriptor.
    request_queue_size = 2048
    # Keep control capacity separate from request capacity. Neither gate
    # allocates workers in advance.
    control_connection_capacity = 64
    # Connections answered without their requests read that wait at once,
    # on one thread, for their clients to close. A burst refuses all that
    # arrive while every slot is held, and a client that sends its request
    # after its connection closed is reset before it reads the 503, so
    # enough linger for a burst well beyond the kernel's queue, within the
    # descriptors the server raises its limit to.
    refused_connection_capacity = 1024
    # The handler that serves the connections; a serving server sets its own.
    handler_type = FrontendHandler

    def __init__(
        self,
        address,
        app,
        bind_and_activate=True,
        request_capacity=serve_options.DEFAULT_QUEUE_SIZE,
        allowed_hosts=(),
        api_key=None,
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
        self.connections = ConnectionSlots(
            request_capacity + self.control_connection_capacity,
            self._serve_connection,
            self.refused_connection_capacity,
            REFUSED_LINGER_SECONDS,
        )
        super().__init__(address, self.handler_type, bind_and_activate)
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
            "connections": self.connections.stats(),
        }
        status["console"] = CONSOLE.stats()
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


def raise_descriptor_limit():
    """Raise the soft descriptor limit to DESCRIPTOR_LIMIT, or to the hard
    limit when that is lower; a higher soft limit stays."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = DESCRIPTOR_LIMIT
    if hard != resource.RLIM_INFINITY:
        wanted = min(wanted, hard)
    if soft < wanted:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))


def interrupt(_signum, _frame):
    """The stop signals' handler: each ends the server as Ctrl+C does."""
    raise KeyboardInterrupt

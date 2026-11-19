"""Forward the page's API requests to a server, unchanged.

A browser lets a page read another origin's answers only when that server
allows the page's origin (CORS), and hides its response headers unless the
server exposes them. The page therefore calls the API through this proxy by
default: the request's method, path, query, body and headers go to the
server the page names, and its status, headers and body come back as they
arrive, so streamed audio streams. Only the hop-by-hop headers and the
browser's own (Host, Origin, Referer, Cookie) are not forwarded.
"""

from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit

# The header naming the server a request goes to: its base URL, such as
# http://127.0.0.1:8000.
UPSTREAM_HEADER = "X-Splash-Upstream"
# Headers that belong to one connection, or to the browser's page, and are
# not forwarded either way.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
_REQUEST_ONLY = frozenset(
    {"host", "origin", "referer", "cookie", "content-length", UPSTREAM_HEADER.lower()}
)
_RESPONSE_ONLY = frozenset({"content-length", "set-cookie"})
# Seconds an upstream may stay silent: a long speech request without
# streaming answers only when it is done.
UPSTREAM_TIMEOUT = 3600
CHUNK = 64 * 1024


class UpstreamError(ValueError):
    """An upstream base URL the proxy does not forward to."""


def parse_upstream(base):
    """(scheme, host, port, path prefix) of a base URL such as
    http://127.0.0.1:8000 or https://example.org/api; UpstreamError for
    anything else."""
    if not isinstance(base, str) or not base.strip():
        raise UpstreamError(f"{UPSTREAM_HEADER} must name the server's base URL")
    try:
        parts = urlsplit(base.strip())
        port = parts.port
    except ValueError:
        raise UpstreamError(f"invalid server URL {base!r}") from None
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise UpstreamError(
            f"invalid server URL {base!r}: expected http(s)://host[:port][/path]"
        )
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return parts.scheme, parts.hostname, port, parts.path.rstrip("/")


def target_path(prefix, path):
    """The upstream request path: the base URL's path, then the API path
    (with its query) the page asked for."""
    if not path.startswith("/"):
        path = "/" + path
    return prefix + path


def request_headers(headers):
    """The headers forwarded upstream, as (name, value) pairs."""
    return [
        (name, value)
        for name, value in headers.items()
        if name.lower() not in _HOP_BY_HOP and name.lower() not in _REQUEST_ONLY
    ]


def response_headers(headers):
    return [
        (name, value)
        for name, value in headers
        if name.lower() not in _HOP_BY_HOP and name.lower() not in _RESPONSE_ONLY
    ]


def _error(handler, status, message):
    data = json.dumps(
        {"error": {"message": message, "type": "proxy_error", "code": status}}
    ).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(data)


def forward(handler, base, path, body, *, connection_type=None):
    """Answer `handler`'s request (a BaseHTTPRequestHandler speaking
    HTTP/1.1) with the answer the server at `base` gives the same request
    for `path`; `body` is the request's body, read. An unreachable server is
    answered 502."""
    handler.close_connection = True
    try:
        scheme, host, port, prefix = parse_upstream(base)
    except UpstreamError as error:
        _error(handler, 400, str(error))
        return
    if connection_type is None:
        connection_type = (
            http.client.HTTPSConnection
            if scheme == "https"
            else http.client.HTTPConnection
        )
    connection = connection_type(host, port, timeout=UPSTREAM_TIMEOUT)
    try:
        try:
            connection.putrequest(
                handler.command,
                target_path(prefix, path),
                skip_accept_encoding=True,
            )
            for name, value in request_headers(handler.headers):
                connection.putheader(name, value)
            if body or handler.command in ("POST", "PUT", "PATCH"):
                connection.putheader("Content-Length", str(len(body)))
            connection.endheaders(body or None)
            answer = connection.getresponse()
        except (OSError, http.client.HTTPException) as error:
            _error(handler, 502, f"cannot reach {base}: {error}")
            return
        _relay(handler, answer)
    finally:
        connection.close()


def _relay(handler, answer):
    """Send the upstream answer on: its status and headers, then its body
    as it arrives, with its length when the server gave one and chunked
    otherwise."""
    length = answer.getheader("Content-Length")
    head = handler.command == "HEAD" or answer.status in (204, 304)
    handler.send_response(answer.status, answer.reason)
    for name, value in response_headers(answer.getheaders()):
        handler.send_header(name, value)
    chunked = not head and length is None
    if length is not None:
        handler.send_header("Content-Length", length)
    elif chunked:
        handler.send_header("Transfer-Encoding", "chunked")
    handler.send_header("Connection", "close")
    handler.end_headers()
    if head:
        return
    try:
        while True:
            data = answer.read1(CHUNK)
            if not data:
                break
            if chunked:
                handler.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            else:
                handler.wfile.write(data)
            handler.wfile.flush()
        if chunked:
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
    except (OSError, http.client.HTTPException):
        # The page went away (a stopped request), or the server cut its
        # answer short: closing the upstream connection cancels it there.
        pass

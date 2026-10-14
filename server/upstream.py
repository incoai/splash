"""Relay requests for the models this server does not serve to a cloud API.

With --anthropic-upstream or --openai-upstream, one base URL carries both a
coding agent's cloud model and the subagents it runs on Splash: a POST under
/v1/ whose model this server does not serve goes to its API's upstream with
the client's own credentials, and the upstream's response returns as it
arrives. Splash keeps no credential and logs no header or body.
"""

import threading
from contextlib import contextmanager
from urllib.parse import quote, urlsplit

from .errors import APIError

ANTHROPIC = "anthropic"
OPENAI = "openai"
# Seconds to connect, and to wait for each read of a response: a streamed
# generation sends events or pings well within this, and the Anthropic and
# OpenAI SDKs give a whole request ten minutes by default.
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 600.0
# Hop-by-hop headers (RFC 9110, 7.6.1), which name this connection only.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-connection",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
# What the relay sets itself. It asks for an identity body, which it passes
# on unread, and ends each response by closing its connection, as every
# Splash response does.
REQUEST_HEADERS_SET = frozenset({"host", "content-length", "accept-encoding", "expect"})
RESPONSE_HEADERS_SET = frozenset({"content-length", "date", "server", "via"})


def api_of(path):
    """The API whose upstream a POST to `path` belongs to: Anthropic's
    Messages paths, and OpenAI's every other path under /v1/; None for any
    other path. The server never relays its own endpoints (PostRoute)."""
    if path == "/v1/messages" or path.startswith("/v1/messages/"):
        return ANTHROPIC
    if path.startswith("/v1/"):
        return OPENAI
    return None


class Upstreams:
    """The upstream base URL of each API that has one, as serve_options
    parses them, and the relay to them."""

    def __init__(self, anthropic=None, openai=None):
        self.urls = {
            api: url
            for api, url in ((ANTHROPIC, anthropic), (OPENAI, openai))
            if url is not None
        }
        self._lock = threading.Lock()
        self._client = None
        self._counts = {api: {"relayed": 0, "failed": 0} for api in self.urls}

    def __bool__(self):
        return bool(self.urls)

    def api(self, path):
        """The API whose upstream may take a POST to `path`; None when it has
        none."""
        api = api_of(path)
        return api if api in self.urls else None

    def url(self, api, path, query=""):
        """Where a request to `path` goes: the base URL with the path as the
        API's SDKs append it to theirs. An Anthropic base URL ends before
        /v1, an OpenAI one includes its version, as https://api.openai.com/v1
        or https://chatgpt.com/backend-api/codex does."""
        if api == OPENAI:
            path = path.removeprefix("/v1")
        return self.urls[api] + quote(path) + (f"?{query}" if query else "")

    def host(self, api):
        return urlsplit(self.urls[api]).netloc

    def status(self):
        with self._lock:
            return {
                api: {"url": url, **self._counts[api]} for api, url in self.urls.items()
            }

    def _http(self):
        # httpx loads only when a request is relayed.
        import httpx

        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
                    follow_redirects=False,
                )
            return self._client

    @contextmanager
    def relay(self, api, path, query, headers, body):
        """The upstream's response to the request, open for reading. The
        request goes as the client sent it but for its hop-by-hop headers;
        a failure to reach the upstream, or one inside the body, raises
        APIError, which answers the client unless its response has begun."""
        import httpx

        forwarded = [
            (name, value)
            for name, value in headers.items()
            if name.lower() not in HOP_BY_HOP | REQUEST_HEADERS_SET
        ]
        forwarded.append(("Accept-Encoding", "identity"))
        try:
            with self._http().stream(
                "POST", self.url(api, path, query), headers=forwarded, content=body
            ) as response:
                with self._lock:
                    self._counts[api]["relayed"] += 1
                yield response
        except httpx.HTTPError as error:
            with self._lock:
                self._counts[api]["failed"] += 1
            if isinstance(error, httpx.TimeoutException):
                raise APIError(
                    504, f"{self.host(api)} did not answer in time", "upstream_timeout"
                ) from None
            raise APIError(
                502,
                f"cannot reach {self.host(api)}: {type(error).__name__}",
                "upstream_unavailable",
            ) from None

    def close(self):
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


def response_headers(response):
    """The upstream response's headers that pass to the client."""
    return [
        (name, value)
        for name, value in response.headers.multi_items()
        if name.lower() not in HOP_BY_HOP | RESPONSE_HEADERS_SET
    ]

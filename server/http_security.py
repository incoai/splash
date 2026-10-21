"""Validate HTTP authority, browser origin and optional API credentials."""

import hmac
import shlex
import threading

from .diagnostics import print_status
from .errors import APIError
from .origins import ANY_ORIGIN, DEFAULT_PORTS, parse_authority, parse_origin


def validate_api_key(value):
    if not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError("API key must contain only visible ASCII characters")
    return value


def authenticate(headers, key):
    if key is None:
        return
    authorization = headers.get_all("Authorization", [])
    api_keys = headers.get_all("x-api-key", [])
    # Anthropic's SDK sends both headers when it has an API key and an auth
    # token: every credential given must be the key, each header only once.
    supplied = list(api_keys)
    for header in authorization:
        scheme, separator, value = header.partition(" ")
        supplied.append(value if separator and scheme.lower() == "bearer" else None)
    if (
        not supplied
        or len(authorization) > 1
        or len(api_keys) > 1
        or not all(
            value is not None and hmac.compare_digest(value.encode(), key.encode())
            for value in supplied
        )
    ):
        raise APIError(401, "invalid or missing API key", "authentication_error")


def _forbidden(message):
    return APIError(403, message, "forbidden")


class OriginRefused(APIError):
    """A request from a page of an origin --allowed-origin does not admit."""

    def __init__(self, origin):
        # Name the fix for the operator.
        super().__init__(
            403,
            f"Origin {origin} is not allowed; restart the server with "
            f"--allowed-origin {origin} to accept it",
            "forbidden",
        )
        self.origin = origin


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


def validate_headers(headers, allowed_hosts, allowed_origins):
    """Refuses a request whose Host or Origin the server does not serve, with
    OriginRefused for a well-formed Origin that no --allowed-origin admits.
    Returns what its response owes a browser in Access-Control-Allow-Origin:
    the origin of a page elsewhere that --allowed-origin admits, or None."""
    hosts = headers.get_all("Host", [])
    origins = headers.get_all("Origin", [])
    if len(hosts) != 1 or len(origins) > 1:
        raise _forbidden("expected one Host header and at most one Origin header")
    try:
        host, port = parse_authority(hosts[0])
    except ValueError:
        raise _forbidden("invalid Host header") from None
    if host not in allowed_hosts:
        # Only the bind address, loopback and --allowed-host names are served,
        # which keeps DNS-rebound pages out. Name the fix for the operator.
        raise _forbidden(
            f"Host {host} is not allowed; restart the server with "
            f"--allowed-host {host} to accept it"
        )
    if not origins:
        return None
    if ANY_ORIGIN in allowed_origins:
        return ANY_ORIGIN
    try:
        origin = parse_origin(origins[0])
    except ValueError:
        raise _forbidden("invalid Origin header") from None
    default_port = DEFAULT_PORTS.get(origin[0])
    if default_port is not None and origin[1:] == (
        host,
        default_port if port is None else port,
    ):
        # The server's own pages.
        return None
    if origin not in allowed_origins:
        # A browser sends Origin with what a page asks of another origin. Only
        # the origins --allowed-origin names may, which keeps the pages of
        # other sites out.
        raise OriginRefused(origins[0])
    return origins[0]

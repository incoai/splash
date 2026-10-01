"""Validate HTTP authority, browser origin and optional API credentials."""

import hmac
from urllib.parse import urlsplit

if __package__:
    from .errors import APIError
else:
    from errors import APIError


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


def _authority(value):
    if not value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError("invalid authority")
    parsed = urlsplit("//" + value)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid authority")
    return parsed.hostname.lower().rstrip("."), parsed.port


def _forbidden(message):
    return APIError(403, message, "forbidden")


def validate_headers(headers, allowed_hosts):
    hosts = headers.get_all("Host", [])
    origins = headers.get_all("Origin", [])
    if len(hosts) != 1 or len(origins) > 1:
        raise _forbidden("expected one Host header and at most one Origin header")
    try:
        host, port = _authority(hosts[0])
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
        return
    try:
        origin = urlsplit(origins[0])
        if origin.scheme not in ("http", "https") or origin.path:
            raise ValueError("invalid origin")
        if origin.query or origin.fragment:
            raise ValueError("invalid origin")
        origin_host, origin_port = _authority(origin.netloc)
    except ValueError:
        raise _forbidden("invalid Origin header") from None
    default_port = 443 if origin.scheme == "https" else 80
    if (origin_host, default_port if origin_port is None else origin_port) != (
        host,
        default_port if port is None else port,
    ):
        raise _forbidden("cross-origin requests are not allowed")

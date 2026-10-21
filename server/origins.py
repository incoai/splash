"""Parse HTTP authorities, browser origins and --allowed-origin values.

Standard library only: the launcher checks --allowed-origin with it before
.venv exists.
"""

from urllib.parse import urlsplit

# Every origin: in --allowed-origin, and in Access-Control-Allow-Origin.
ANY_ORIGIN = "*"
DEFAULT_PORTS = {"http": 80, "https": 443}


def parse_authority(value):
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


def parse_origin(value):
    # An origin as a browser serializes it: a scheme and an authority and
    # nothing after them. A port it does not name is its scheme's default.
    # No browser sends a *, which only a pattern holds.
    if "*" in value or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise ValueError("invalid origin")
    parsed = urlsplit(value)
    if not parsed.scheme or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("invalid origin")
    host, port = parse_authority(parsed.netloc)
    return (
        parsed.scheme,
        host,
        DEFAULT_PORTS.get(parsed.scheme) if port is None else port,
    )


def parse_allowed_origin(value):
    """An --allowed-origin value as validate_headers compares it: ANY_ORIGIN,
    or an origin's scheme, host and port."""
    if value == ANY_ORIGIN:
        return ANY_ORIGIN
    if "*" in value:
        raise ValueError(
            f"{value} is not an origin: only a bare '*' admits every origin; "
            "origins are matched exactly, so patterns such as tauri://* or "
            "http://*.example.com are not supported"
        )
    try:
        return parse_origin(value)
    except ValueError:
        raise ValueError(
            f"{value} is not an origin: expected a scheme and a host, as in "
            "tauri://localhost or http://localhost:3000, or '*' for every origin"
        ) from None

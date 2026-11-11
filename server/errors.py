from __future__ import annotations

import threading
import time

# The statuses of a refusal the client may retry shortly, which a response
# invites with Retry-After: the server is overloaded, restarting its engine
# or shutting down. Each API answers these with 503, /v1/systemone with 529.
RETRY_STATUSES = (503, 529)


class APIError(Exception):
    def __init__(self, status, message, code="invalid_request_error", retry_after=1):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        # The seconds a retry status invites the client to wait (Retry-After).
        self.retry_after = retry_after
        # Whether the console has its line: the backend logs the end of a
        # request the engine took, a failure included, under the engine's
        # id, and the HTTP handler does not log that failure again.
        self.logged = False


class RequestValidationError(APIError):
    """Request fields that fail validation, each described as FastAPI
    describes one (field_error). /v1/systemone answers with every one, as
    the FastAPI service it stands in for does; the other APIs with the first
    one's message, as with any invalid request."""

    def __init__(self, details):
        super().__init__(400, details[0]["msg"])
        self.details = list(details)


def remaining_request_time(deadline):
    """The seconds left before `deadline`; a 504 once it has passed."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise APIError(504, "request timed out", "request_timeout")
    # Callers wait this long, and waits reject a timeout above TIMEOUT_MAX;
    # a request without a deadline has infinite time left.
    return min(remaining, threading.TIMEOUT_MAX)


def field_error(loc, msg, error_type="value_error"):
    """What is wrong with the body field at path `loc`, the body itself when
    `loc` is empty."""
    return {"loc": ["body", *loc], "msg": msg, "type": error_type}


class ConstraintError(Exception):
    """The output grammar rejected a token or has no valid next token."""


class ContextLengthError(APIError):
    def __init__(self, input_tokens, maximum_input_tokens, *, image_tokens_only=False):
        super().__init__(
            400, "prompt exceeds the context window", "context_length_exceeded"
        )
        self.input_tokens = input_tokens
        self.maximum_input_tokens = maximum_input_tokens
        self.image_tokens_only = image_tokens_only


class ErrorDialect:
    """How an API answers errors; this one, OpenAI's, also answers on the
    paths of no other API (server.path_errors)."""

    def answer(self, error):
        """The status of the response that answers `error`, the code the
        console logs for it, and its payload."""
        return error.status, error.code, self.payload(error)

    def payload(self, error):
        """`error` as a response or an event stream carries it."""
        error_type = "server_error" if error.status >= 500 else "invalid_request_error"
        return {
            "error": {"message": error.message, "type": error_type, "code": error.code}
        }


OPENAI_ERRORS = ErrorDialect()

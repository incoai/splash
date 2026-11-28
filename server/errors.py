from __future__ import annotations

import threading
import time

from . import protocol as wire

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
    """A prompt longer than the context, worded with both counts in each
    API's shape: agents that trim history by the numbers in the message can."""

    def __init__(self, input_tokens, maximum_input_tokens, *, image_tokens_only=False):
        super().__init__(
            400,
            f"prompt exceeds the context window: {input_tokens} tokens > "
            f"{maximum_input_tokens} maximum input tokens"
            + (
                " (image tokens alone; text not yet counted)"
                if image_tokens_only
                else ""
            ),
            "context_length_exceeded",
        )
        self.input_tokens = input_tokens
        self.maximum_input_tokens = maximum_input_tokens
        self.image_tokens_only = image_tokens_only

    def anthropic_message(self):
        """The message in Anthropic's own wording, which Claude Code reads to
        compact a conversation."""
        return (
            f"prompt is too long: {self.input_tokens} tokens > "
            f"{self.maximum_input_tokens} maximum input tokens"
            + (
                " (image tokens alone; text not yet counted)"
                if self.image_tokens_only
                else ""
            )
        )


# What helps a request that does not fit even alone, by what it lacked
# (wire.CapacityLimit): its client and the console are told alike. Never
# --max-context, which sizes no memory a lone request lacks, nor an image's
# vision scratch.
CAPACITY_REMEDIES = {
    wire.CapacityLimit.MEMORY: "shorten the conversation, or restart the server "
    "with a larger --max-memory if one was set below this Mac's memory",
    wire.CapacityLimit.IMAGE_MEMORY: "send a smaller image, or restart the server "
    "with a larger --max-memory or a smaller --max-image-pixels",
    wire.CapacityLimit.CLOSED_DISK: "shorten the conversation, or restart the "
    "server, which reopens the SSD cache that stopped taking writes",
    wire.CapacityLimit.DISK_QUOTA: "shorten the conversation, or restart the "
    "server with a larger --max-cache-disk",
}


class CapacityError(ContextLengthError):
    """A request of `input_tokens` that does not fit even alone, in memory or
    in the SSD cache's quota (the engine's capacity_exhausted and
    kv_disk_quota, `failure`), answered as each API answers a prompt over the
    context, so that coding agents compact the conversation and retry rather
    than stop: its tokens, the prompt tokens memory or the quota holds it now
    where the engine says so, and what helps. The console keeps the engine's
    code and sentence (engine_code, engine_reason) beside the same advice."""

    def __init__(self, input_tokens, failure):
        limit = failure.capacity_limit
        # The counts say why where its prompt is what does not fit; a request
        # whose output outgrew what it holds had a prompt that fit.
        fits_tokens = (
            failure.fits_tokens if 0 < failure.fits_tokens < input_tokens else None
        )
        super().__init__(input_tokens, fits_tokens)
        self.fits_tokens = fits_tokens
        self.remedy = CAPACITY_REMEDIES[limit]
        self.engine_code = failure.code.decode("ascii", "replace")
        self.engine_reason = failure.message_bytes.decode("utf-8", "replace")
        self.where = (
            "the SSD cache"
            if limit is wire.CapacityLimit.DISK_QUOTA
            else "this server's memory"
        )
        where = self.where
        self.message = (
            f"prompt does not fit in {where}: {input_tokens} tokens > "
            f"{fits_tokens} tokens it holds for this request now; {self.remedy}"
            if fits_tokens is not None
            else f"prompt does not fit in {where}: {input_tokens} tokens; {self.remedy}"
        )
        self.args = (self.message,)

    def anthropic_message(self):
        if self.fits_tokens is None:
            return (
                f"prompt is too long: {self.input_tokens} tokens do not fit in "
                f"{self.where} now; {self.remedy}"
            )
        return (
            f"prompt is too long: {self.input_tokens} tokens > "
            f"{self.fits_tokens} maximum input tokens {self.where} holds for it "
            f"now; {self.remedy}"
        )


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

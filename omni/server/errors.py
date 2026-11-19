from __future__ import annotations

# The statuses of a refusal the client may retry shortly, which a response
# invites with Retry-After: the server is overloaded, restarting its engine
# or shutting down.
RETRY_STATUSES = (503,)


class APIError(Exception):
    def __init__(self, status, message, code="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


class RequestValidationError(APIError):
    """Request fields that fail validation, each described as FastAPI
    describes one (field_error); the response carries the first one's
    message, as with any invalid request."""

    def __init__(self, details):
        super().__init__(400, details[0]["msg"])
        self.details = list(details)


def field_error(loc, msg, error_type="value_error"):
    """What is wrong with the body field at path `loc`, the body itself when
    `loc` is empty."""
    return {"loc": ["body", *loc], "msg": msg, "type": error_type}


class ErrorDialect:
    """How an API answers errors: OpenAI's shape, on every path."""

    def answer(self, error):
        """The status of the response that answers `error`, the code the
        console logs for it, and its payload."""
        return error.status, error.code, self.payload(error)

    def payload(self, error):
        """`error` as a response carries it."""
        error_type = "server_error" if error.status >= 500 else "invalid_request_error"
        return {
            "error": {"message": error.message, "type": error_type, "code": error.code}
        }


OPENAI_ERRORS = ErrorDialect()

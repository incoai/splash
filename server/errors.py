from __future__ import annotations

# The statuses of a refusal the client may retry shortly, which a response
# invites with Retry-After: the server is overloaded, restarting its engine
# or shutting down. Each API answers these with 503, /v1/systemone with 529.
RETRY_STATUSES = (503, 529)


class APIError(Exception):
    def __init__(self, status, message, code="invalid_request_error", *, details=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        # Optional structured detail an OpenAI-shaped error body carries as
        # error.details (for example a /v1/score label's token breakdown).
        self.response_details = details


class RequestValidationError(APIError):
    """Request fields that fail validation, each described as FastAPI
    describes one (field_error). /v1/systemone answers with every one, as
    the FastAPI service it stands in for does; the other APIs with the first
    one's message, as with any invalid request."""

    def __init__(self, details):
        super().__init__(400, details[0]["msg"])
        self.details = list(details)


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
        body = {"message": error.message, "type": error_type, "code": error.code}
        if error.response_details is not None:
            body["details"] = error.response_details
        return {"error": body}


class AnthropicErrors(ErrorDialect):
    # The type of an error of each status Anthropic's API names one for; any
    # other is an api_error from 500, else an invalid_request_error.
    TYPES = {
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        408: "timeout_error",
        413: "request_too_large",
        429: "rate_limit_error",
        503: "overloaded_error",
        504: "timeout_error",
    }

    def payload(self, error):
        message = error.message
        if isinstance(error, ContextLengthError):
            message = (
                f"prompt is too long: {error.input_tokens} tokens > "
                f"{error.maximum_input_tokens} maximum input tokens"
            )
            if error.image_tokens_only:
                message += " (image tokens alone; text not yet counted)"
        error_type = self.TYPES.get(
            error.status,
            "api_error" if error.status >= 500 else "invalid_request_error",
        )
        return {"type": "error", "error": {"type": error_type, "message": message}}


class SystemOneErrors(ErrorDialect):
    """/v1/systemone answers as the FastAPI service it stands in for:
    invalid fields with 422 and what is wrong with each, an overload with
    TypeSafe's 529, and other errors as OpenAI's APIs do."""

    def answer(self, error):
        if isinstance(error, RequestValidationError):
            return 422, "unprocessable_entity", {"detail": error.details}
        status, code, payload = super().answer(error)
        return 529 if status == 503 else status, code, payload


OPENAI_ERRORS = ErrorDialect()
ANTHROPIC_ERRORS = AnthropicErrors()
SYSTEMONE_ERRORS = SystemOneErrors()

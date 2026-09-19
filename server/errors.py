from __future__ import annotations


class APIError(Exception):
    def __init__(self, status, message, code="invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code

    def protocol_type(self, anthropic=False):
        if not anthropic:
            return "server_error" if self.status >= 500 else "invalid_request_error"
        return {
            401: "authentication_error",
            403: "permission_error",
            404: "not_found_error",
            408: "timeout_error",
            413: "request_too_large",
            429: "rate_limit_error",
            503: "overloaded_error",
            504: "timeout_error",
        }.get(
            self.status, "api_error" if self.status >= 500 else "invalid_request_error"
        )


class NativeError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class ContextLengthError(APIError):
    def __init__(self, input_tokens, maximum_input_tokens, *, image_tokens_only=False):
        super().__init__(
            400, "prompt exceeds the context window", "context_length_exceeded"
        )
        self.input_tokens = input_tokens
        self.maximum_input_tokens = maximum_input_tokens
        self.image_tokens_only = image_tokens_only

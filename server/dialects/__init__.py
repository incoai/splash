"""How each API dialect answers a generation, whole or as a stream.

Each module's complete(exchange, job, ...) and stream(exchange, job, ...)
read the generation and write the response through the exchange, the HTTP
handler (server.FrontendHandler) and its public methods; these modules hold
no HTTP.
"""

from . import anthropic, openai, responses

__all__ = ("anthropic", "openai", "responses")

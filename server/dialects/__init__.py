"""Each API's own module: how it reads its requests and writes its answers.

A module's functions take the exchange, the HTTP handler
(server.FrontendHandler), and use only its public methods: a route's
handler answers its request or returns the generation it prepared, and
complete(exchange, job, ...) and stream(exchange, job, ...) write a
generation's answer whole or as a stream. The HTTP itself stays in the
router.
"""

from . import anthropic, openai, responses, scoring

__all__ = ("anthropic", "openai", "responses", "scoring")

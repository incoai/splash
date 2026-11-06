"""Each API's own module: its routes, how it reads its requests and how it
writes its answers.

A route's handler takes the exchange, the HTTP handler
(server.FrontendHandler), and uses only its public methods: it answers its
request or returns the generation it prepared, and complete(exchange, job,
...) and stream(exchange, job, ...) write a generation's answer whole or as
a stream. A request's preparation takes the app (server.frontend.Frontend)
and no exchange. The HTTP itself stays in the router.
"""

from . import anthropic, openai, responses, scoring

__all__ = ("anthropic", "openai", "responses", "scoring")

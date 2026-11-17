"""Prefill-only decision scoring: POST /v1/score and /v1/decisions.

The requests, labels and answers live in server/scoring.py, and a score
job's preparation in Frontend.prepare_score: messages render exactly as Chat
renders them, so a score request also warms its agent's prompt in the prefix
cache. These handlers submit a request's jobs together, so a "requests"
batch runs concurrently through the scheduler, and answer in order.
"""

from __future__ import annotations

from .. import scoring
from ..errors import APIError, remaining_request_time


def post_score(exchange, body, deadline):
    """POST /v1/score: one request, or a batch under "requests" whose jobs
    run concurrently through the scheduler (scoring.py)."""
    batched = "requests" in body
    if batched:
        unknown = sorted(set(body) - scoring.BATCH_FIELDS)
        if unknown:
            raise APIError(400, f"unsupported fields: {', '.join(unknown)}")
        items = body["requests"]
        if (
            not isinstance(items, list)
            or not items
            or len(items) > scoring.MAX_BATCH_REQUESTS
            or any(not isinstance(item, dict) for item in items)
        ):
            raise APIError(
                400,
                "requests must be an array of 1 to "
                f"{scoring.MAX_BATCH_REQUESTS} request objects",
            )
        shared = {
            key: body[key] for key in ("model", "priority", "cache_salt") if key in body
        }
        bodies = [{**shared, **item} for item in items]
    else:
        bodies = [body]
    responses = run_scores(exchange, bodies, deadline, indexed=batched)
    exchange.send_json(
        200,
        {"object": "list", "data": responses} if batched else responses[0],
    )


def run_scores(exchange, bodies, deadline, *, indexed):
    """Prepare every score request, submit them together and return their
    /v1/score results in order; errors name requests[i] when indexed."""
    app = exchange.app
    jobs = [
        app.prepare_score(item, deadline=deadline, index=index if indexed else None)
        for index, item in enumerate(bodies)
    ]
    remaining_request_time(deadline)
    if exchange.client_disconnected():
        raise ConnectionResetError("client disconnected before submission")
    submitted = []
    try:
        for job in jobs:
            app.backend.submit(job)
            submitted.append(job)
        results = [await_done(exchange, job) for job in jobs]
    except BaseException:
        for job in submitted:
            app.backend.cancel(job)
        raise
    responses = []
    for job, result in zip(jobs, results):
        cached = job.cache.matched_tokens
        prefill = result.start_to_first_token_ms
        total = result.request_wall_ms
        responses.append(
            scoring.score_response(
                app.response_model,
                job.score_meta["labels"],
                job.score_tokens,
                job.score_meta["options"],
                result,
                tokenizer=app.tokenizer,
                cached_tokens=cached,
                queue_ms=max(0.0, total - prefill),
            )
        )
        app.record_score(result.prompt_tokens, cached, total / 1000.0)
    return responses


def post_decisions(exchange, body, deadline):
    """POST /v1/decisions: each question a single-token score over the
    shared input (scoring.decision_requests)."""
    try:
        bodies, questions = scoring.decision_requests(body)
    except scoring.ScoreRequestError as error:
        raise APIError(400, str(error)) from error
    scores = run_scores(exchange, bodies, deadline, indexed=False)
    exchange.send_json(
        200,
        {
            "object": "decisions",
            "model": exchange.app.response_model,
            "answers": {
                question.qid: scoring.decision_answer(question, score)
                for question, score in zip(questions, scores)
            },
            "usage": {
                "prompt_tokens": sum(
                    score["usage"]["prompt_tokens"] for score in scores
                ),
                "cached_tokens": sum(
                    score["usage"]["cached_tokens"] for score in scores
                ),
            },
        },
    )


def await_done(exchange, job):
    """The result of a score job, which emits only start and done."""
    while (event := exchange.next_event(job))[0] != "done":
        pass
    return event[1]

"""The serving benchmark's clients: what they send, what they record.

Four scenarios drive a running server over HTTP, stdlib only, so that they
measure every tree the same way (1.5.0, main and the candidate serve the same
OpenAI API but different /status documents):

- churn: three workers, each sending a long-output request whose prompt grows
  every round and every fourth round resending its previous conversation, the
  first requests in a fixed arrival order (churn_ord.py of the 1.6.0 work,
  whose request bodies these are byte for byte). /status every 2 s.
- ttft: N conversations cache an ~8K-token prefix one at a time, then send it
  with distinct ~300-token suffixes at the same instant, ROUNDS times
  (ttft.py, whose prompts these are). /status every 0.5 s.
- serve1: one request at a time, cold prompts of about 300, 2K and 14K tokens:
  time to first token and decode milliseconds per token.
- warm: a cached long prefix per depth, then short suffixes after it: the
  server's prompt time of each (the prefill-splits measurement's protocol).
  serve1 and warm record the /status each request waited idle for, and the
  one after the last.

Every request timestamps each content chunk of its stream, so a request's
gaps, the intervals between consecutive chunks after its first, are measured
by the client whatever the tree reports.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import random
import statistics
import sys
import threading
import time

API_KEY = "splash-local"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
# Gaps of at least this long are listed one by one in a request's record.
LONG_GAP_SECONDS = 1.0
# Log2 buckets of gap milliseconds: [0, 2), [2, 4), ..., [2^20, inf).
GAP_BUCKETS = 21
# How long a request may take from its send before it is one the server did
# not answer in time: twice the longest request measured, 1068.4 s on the M5
# Pro at --max-memory 18G, so that a slow Mac is not mistaken for a hang.
DEADLINE_FACTOR = 2
REQUEST_SECONDS = DEADLINE_FACTOR * 1068.4


class RequestFailed(RuntimeError):
    """A request the server answered with an error, or without a token,
    that ends a scenario: the server's own failure (G8)."""


class Deadline(TimeoutError):
    """A request the server did not end within its time, though it kept the
    stream alive (server.py writes a keepalive every 2 s while a request
    runs, so no socket timeout fires): the server's own failure."""


class Unanswered(Exception):
    """A churn request the client got no whole answer to, recorded as such
    (run_churn): it stops its worker, as the server no longer answers it."""


class ClientError(RuntimeError):
    """A client's own failure, not the server's, such as an exception in
    recording a request the server answered: the run measured nothing."""


def dig(document, path: str):
    """The value at a dotted path of nested dicts, or None."""
    for key in path.split("."):
        document = document.get(key) if isinstance(document, dict) else None
    return document


def get_json(port: int, path: str, timeout: float = 60):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", path, headers=HEADERS)
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise RuntimeError(f"GET {path}: {response.status} {body[:200]!r}")
    return json.loads(body)


def served_model(port: int) -> str:
    return get_json(port, "/v1/models")["data"][0]["id"]


def gap_summary(chunks: list[float], run_start: float) -> dict:
    """A request's gaps from the clock times of its content chunks: count,
    max, p99, log2 histogram in milliseconds and every gap of at least
    LONG_GAP_SECONDS as [seconds into the run at its end, length]."""
    gaps = [later - earlier for earlier, later in zip(chunks, chunks[1:])]
    histogram = [0] * GAP_BUCKETS
    for gap in gaps:
        milliseconds = gap * 1000
        bucket = 0 if milliseconds < 2 else int(math.log2(milliseconds))
        histogram[min(bucket, GAP_BUCKETS - 1)] += 1
    ordered = sorted(gaps)
    return {
        "n": len(gaps),
        "max_s": round(ordered[-1], 3) if gaps else 0,
        "p99_s": round(ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))], 3)
        if gaps
        else 0,
        "hist_ms_log2": histogram,
        "long": [
            [round(end - run_start, 2), round(end - start, 3)]
            for start, end in zip(chunks, chunks[1:])
            if end - start >= LONG_GAP_SECONDS
        ],
    }


class Stream:
    """One streamed chat completion: its HTTP status, content chunk times
    (appended to `chunks` as they arrive), text, usage and the server's own
    metrics, and why the server failed it: the body of an HTTP error, or the
    error event of a stream it had begun, which keeps status 200 (the server
    sends it, then [DONE], when the request fails after its first event).
    With a barrier the request is sent once every sender has connected, and
    its clock starts after the connection; without one, before it. A request
    still streaming `timeout` seconds after its send raises Deadline; the
    socket's own timeout, as long, ends one the server went silent on. A
    stream that ends before its `data: [DONE]` without an error event raises
    ConnectionError, as http.client does for a connection the server closed
    before its response."""

    def __init__(
        self,
        port: int,
        body: str,
        timeout: float = REQUEST_SECONDS,
        barrier=None,
        chunks=None,
    ):
        self.chunks: list[float] = [] if chunks is None else chunks
        self.text: list[str] = []
        self.usage: dict = {}
        self.metrics: dict = {}
        self.finish_reason: str | None = None
        self.error_body = ""
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            if barrier is None:
                self.sent = time.time()
                connection.connect()
            else:
                connection.connect()
                barrier.wait()
                self.sent = time.time()
            connection.request("POST", "/v1/chat/completions", body, HEADERS)
            response = connection.getresponse()
            self.status = response.status
            if response.status != 200:
                self.error_body = response.read().decode(errors="replace")[:200]
                return
            finished = False
            for line in response:
                if time.time() - self.sent > timeout:
                    raise Deadline(
                        f"a request still streaming {timeout:g} s after its send"
                    )
                finished = finished or line.startswith(b"data: [DONE]")
                if not line.startswith(b"data: {"):
                    continue
                event = json.loads(line[6:])
                if "error" in event:
                    self.error_body = json.dumps(event["error"])[:200]
                self.usage = event.get("usage") or self.usage
                self.metrics = event.get("metrics") or self.metrics
                for choice in event.get("choices", []):
                    self.finish_reason = (
                        choice.get("finish_reason") or self.finish_reason
                    )
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        self.chunks.append(time.time())
                        self.text.append(piece)
            if not finished and not self.error_body:
                raise ConnectionError("the stream ended before its data: [DONE]")
        finally:
            connection.close()
            self.ended = time.time()

    @property
    def ok(self) -> bool:
        """Whether the server served the request: status 200 and no error."""
        return self.status == 200 and not self.error_body

    @property
    def first(self) -> float | None:
        return self.chunks[0] if self.chunks else None


class Recorder:
    """A run's events (LABEL.events.jsonl) and /status samples
    (LABEL.status.jsonl), each line stamped with seconds since the run began."""

    def __init__(self, directory: str, label: str, status_name: str = "status"):
        self.start = time.time()
        self.lock = threading.Lock()
        self.events = open(os.path.join(directory, f"{label}.events.jsonl"), "w")
        self.status = open(os.path.join(directory, f"{label}.{status_name}.jsonl"), "w")

    def seconds(self, at: float | None = None) -> float:
        return round((time.time() if at is None else at) - self.start, 2)

    def emit(self, kind: str, **fields) -> None:
        with self.lock:
            self.events.write(
                json.dumps({"t": self.seconds(), "kind": kind, **fields}) + "\n"
            )
            self.events.flush()

    def sample(self, status: dict, **fields) -> None:
        with self.lock:
            self.status.write(
                json.dumps({"t": self.seconds(), **fields, "status": status}) + "\n"
            )
            self.status.flush()

    def close(self) -> None:
        self.events.close()
        self.status.close()


def poll_status(
    port: int, recorder: Recorder, until, period: float, observe=None
) -> None:
    """Records /status every `period` seconds until until() is true; with
    `observe`, also hands it each sample."""
    while not until():
        try:
            status = get_json(port, "/status", timeout=30)
            recorder.sample(status)
            if observe is not None:
                observe(status)
        except Exception as error:  # recorded: a run with gaps in its samples says so
            recorder.emit("status_error", error=str(error)[:200])
        time.sleep(period)


# ---------------------------------------------------------------- churn


CHURN_WORDS = ("amber", "cobalt", "jade", "ruby", "slate")
CHURN_DEFAULT_BASE, CHURN_DEFAULT_STEP = 1100, 900
CHURN_OUTPUT_TOKENS = 1500
# The first requests arrive in the order's sequence, this far apart.
CHURN_STAGGER_SECONDS = 0.6
CHURN_WORKERS = 3


def churn_doc(seed: int, words: int) -> str:
    rng = random.Random(seed)
    return " ".join(
        f"{rng.choice(CHURN_WORDS)}-{rng.randrange(10**6)}" for _ in range(words)
    )


def churn_messages(worker: int, round_number: int, base: int, step: int) -> list[dict]:
    """A worker's request of one round, base + step x round words: its
    first, round 1, about 16.2K tokens at the default base and step (2,000
    words; DOC_TOKENS_PER_WORD), then 7.3K more each round."""
    words = base + step * round_number
    log = churn_doc(worker * 1000 + round_number, words)
    return [
        {
            "role": "user",
            "content": f"Log {worker}-{round_number}:\n{log}\nIgnore the log. Write the numbers "
            f"from 1 to {CHURN_OUTPUT_TOKENS}, one per line, nothing else.",
        }
    ]


def churn_resend(messages: list[dict], answer: str) -> list[dict]:
    """The conversation a worker resends every fourth round."""
    return messages + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "Again, the same numbers."},
    ]


def churn_body(model: str, messages: list[dict]) -> str:
    return json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": CHURN_OUTPUT_TOKENS,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )


def churn_status_event(status: dict) -> dict:
    """The few /status values each churn sample also logs as an event."""
    experts = status.get("streamed_experts") or {}
    return {
        "chunks": experts.get("chunks"),
        "cap_chunks": experts.get("cap_chunks"),
        "shrinks": experts.get("shrinks"),
        "regrows": experts.get("regrows"),
        "arena": experts.get("arena_bytes"),
        "hit": dig(experts, "decode.hit_rate"),
        "pressure": status.get("memory_pressure"),
        "req": status.get("requests"),
        "metal_ok": dig(status, "metal.healthy"),
    }


def run_churn(
    port: int,
    directory: str,
    label: str,
    minutes: float,
    order: list[int],
    base: int = CHURN_DEFAULT_BASE,
    step: int = CHURN_DEFAULT_STEP,
    poll_seconds: float = 2.0,
    timeout: float = REQUEST_SECONDS,
) -> None:
    """A churn run. Workers send requests until `minutes` have passed and
    finish the one they have in flight; the run ends 4 s after the last.
    Each request ends in one record: done, http_error (the server answered
    it with an error) or unanswered (chat), which also stops its worker.
    Any other exception is the client's own: it ends its worker with a
    traceback and, once the others finish, the run with ClientError. Each
    /status sample also logs churn_status_event's values as an event."""
    model = served_model(port)
    recorder = Recorder(directory, label)
    end = recorder.start + minutes * 60

    def chat(messages, tag):
        """Sends one request and records how it ended: done, returning the
        answer; http_error, the server's error, returning None; or
        unanswered, when the client got no whole answer: its deadline
        passed (Deadline), or its connection failed or closed before the
        stream's end. That record keeps its chunk times, as a done record
        does for its gaps, and why, and chat raises Unanswered: the server
        no longer answers it. Only the stream's own failures are
        unanswered."""
        body = churn_body(model, messages)
        chunks: list[float] = []
        sent = time.time()
        try:
            stream = Stream(port, body, timeout, chunks=chunks)
        except Exception as error:
            recorder.emit(
                "unanswered",
                tag=tag,
                start=round(sent - recorder.start, 1),
                chunks_ms=[round((t - sent) * 1000) for t in chunks],
                error=f"{type(error).__name__}: {error}",
            )
            raise Unanswered(tag) from error
        if not stream.ok:
            # Code 200 for an error event in a stream the server had begun.
            recorder.emit(
                "http_error", tag=tag, code=stream.status, body=stream.error_body
            )
            return None
        out = stream.usage.get("completion_tokens") or 0
        first = stream.first or stream.ended
        recorder.emit(
            "done",
            tag=tag,
            prompt=stream.usage.get("prompt_tokens"),
            cached=dig(stream.usage, "prompt_tokens_details.cached_tokens"),
            out=out,
            ttft=round(first - stream.sent, 2),
            decode_tps=round((out - 1) / max(1e-3, stream.ended - first), 1)
            if out > 1
            else None,
            start=round(stream.sent - recorder.start, 1),
            gap=gap_summary(stream.chunks, recorder.start),
            chunks_ms=[round((t - stream.sent) * 1000) for t in stream.chunks],
        )
        return "".join(stream.text)

    def worker(index):
        time.sleep(CHURN_STAGGER_SECONDS * order.index(index))
        round_number, previous = 0, None
        while time.time() < end:
            round_number += 1
            resend = previous is not None and round_number % 4 == 0
            if resend:
                messages, tag = previous, f"w{index}-resend{round_number}"
            else:
                messages = churn_messages(index, round_number, base, step)
                tag = f"w{index}-r{round_number}"
            try:
                answer = chat(messages, tag)
            except Unanswered:
                return
            if answer is not None and not resend:
                previous = churn_resend(messages, answer)

    # The exceptions of the clients' own that ended their workers.
    failures: list[Exception] = []

    def client(index):
        try:
            worker(index)
        except Exception as error:
            failures.append(error)
            raise  # threading prints its traceback

    def observe(status):
        recorder.emit("status", **churn_status_event(status))

    done = threading.Event()
    poller = threading.Thread(
        target=poll_status,
        args=(port, recorder, done.is_set, poll_seconds, observe),
        daemon=True,
    )
    poller.start()
    workers = [
        threading.Thread(target=client, args=(w,), daemon=True)
        for w in range(CHURN_WORKERS)
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    time.sleep(4)
    recorder.emit("end")
    done.set()
    poller.join(timeout=poll_seconds + 35)
    recorder.close()
    if failures:
        error = failures[0]
        raise ClientError(
            f"a churn client failed: {type(error).__name__}: {error}"
        ) from error


# ----------------------------------------------------------------- ttft


TTFT_FILES = (
    "argparse.py",
    "typing.py",
    "inspect.py",
    "tarfile.py",
    "pydoc.py",
    "_pydecimal.py",
    "subprocess.py",
    "difflib.py",
    "threading.py",
    "dataclasses.py",
    "configparser.py",
    "statistics.py",
)
# Prefix slots before the suffixes; suffix slots per conversation.
TTFT_PREFIX_SLOTS, TTFT_ROUND_SLOTS = 16, 8
TTFT_PREFIX_CHARS = 36500  # about 8K tokens of Python source
TTFT_SUFFIX_CHARS = 1100  # about 300 tokens
TTFT_MAX_TOKENS = 16
# /status counters whose change a round reports, and gauges read after it.
TTFT_COUNTERS = (
    "scheduler.prefill_batches",
    "scheduler.prefill_rows",
    "scheduler.decode_batches",
    "model_timing.prefill.total_gpu_ms",
    "model_timing.decode.total_gpu_ms",
    "streamed_experts.prefill.loads",
    "streamed_experts.prefill.routed",
    "streamed_experts.prefill.read",
    "streamed_experts.prefill.read_bytes",
    "streamed_experts.shrinks",
    "streamed_experts.regrows",
    "streamed_table.prefill.unique_rows",
    "streamed_table.prefill.read_rows",
    "cache.hits",
    "cache.cold_misses",
    "cache.kv_hit_tokens",
    "cache.kv_disk_hit_tokens",
    "state.disk_hits",
    "disk.kv_restores",
    "disk.read_bytes",
    "memory_governor.denied_reservations",
)
TTFT_GAUGES = (
    "memory_pressure",
    "streamed_experts.chunks",
    "streamed_experts.slots",
    "state.entries",
    "kv.pages_cache",
)
# memory_governor.claims' fields a round reports at their largest over the
# /status samples taken while it ran: a round whose starts waited for
# memory measured the wait, not the concurrent prefill.
TTFT_CLAIMS = ("waiting", "bytes", "oldest_ms")


def ttft_corpus() -> str:
    """Slices of this interpreter's stdlib sources: the same prompts on every
    run with the same interpreter, whose digest each summary records."""
    root = os.path.dirname(os.__file__)
    parts = []
    for name in TTFT_FILES:
        with open(os.path.join(root, name), encoding="utf-8") as source:
            parts.append(source.read())
    return "\n".join(parts)


def excerpt(text: str, start: int, chars: int) -> str:
    """`chars` characters of `text` from `start`, widened to whole lines."""
    begin = text.find("\n", start) + 1
    end = text.rfind("\n", begin, begin + chars)
    if begin <= 0 or end <= begin:
        raise ValueError("corpus too short for these slices")
    return text[begin:end]


def ttft_conversation(
    text: str, n: int, conversation: int, prefix_chars: int = TTFT_PREFIX_CHARS
):
    """The warm-up of conversation `conversation` at concurrency `n`, and its
    slot, distinct for every (N, conversation)."""
    slot = n * (n - 1) // 2 + conversation
    prefix = excerpt(text, slot * prefix_chars, prefix_chars)
    content = f"Conversation {n}.{conversation}. A source excerpt follows.\n\n{prefix}\n\nReply with OK."
    return [{"role": "user", "content": content}], slot


def ttft_follow_up(
    text: str,
    warm: list[dict],
    slot: int,
    round_number: int,
    prefix_chars: int = TTFT_PREFIX_CHARS,
    suffix_chars: int = TTFT_SUFFIX_CHARS,
):
    start = (
        TTFT_PREFIX_SLOTS * prefix_chars
        + (slot * TTFT_ROUND_SLOTS + round_number) * suffix_chars
    )
    suffix = excerpt(text, start, suffix_chars)
    return warm + [
        {"role": "assistant", "content": "OK."},
        {
            "role": "user",
            "content": f"Round {round_number}. Another excerpt:\n\n{suffix}\n\n"
            "In one sentence, what does it do?",
        },
    ]


def ttft_body(model: str, messages: list[dict], max_tokens: int) -> str:
    return json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )


def busy(status: dict) -> bool:
    """Whether the server still has work: a request, a queued, prefilling,
    decoding or waiting lane, an admission wait, or (schema 7) memory owed to
    a transfer. Restore points waiting for a copy (disk.write_behind.waiting)
    are not work: nothing needs them yet."""
    scheduler, requests = status.get("scheduler") or {}, status.get("requests") or {}
    return bool(
        status.get("active_requests")
        or scheduler.get("queued")
        or scheduler.get("prefilling")
        or scheduler.get("decoding")
        or scheduler.get("waiting_resources")
        or scheduler.get("waiting_prefix")
        or scheduler.get("waiting_mask")
        or dig(status, "admission.waiting")
        or dig(status, "memory_governor.owed.transfers")
        or requests.get("submitted", 0)
        != requests.get("completed", 0)
        + requests.get("cancelled", 0)
        + requests.get("failed", 0)
    )


def wait_idle(port: int, settle: float, timeout: float = 600) -> dict:
    """/status once the server has had no work for `settle` seconds."""
    deadline = time.time() + timeout
    while True:
        status = get_json(port, "/status")
        if not busy(status):
            if settle <= 0:
                return status
            time.sleep(settle)
            settle = 0
            continue
        if time.time() > deadline:
            raise RuntimeError("server did not go idle")
        time.sleep(0.25)


def counter_deltas(before: dict, after: dict) -> dict:
    out = {}
    for path in TTFT_COUNTERS:
        a, b = dig(before, path), dig(after, path)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[path] = round(b - a, 3)
    routed, read = (
        out.get("streamed_experts.prefill.routed"),
        out.get("streamed_experts.prefill.read"),
    )
    if routed:
        out["expert_prefill_hit_rate"] = round(1 - read / routed, 4)
    for path in TTFT_GAUGES:
        out[path] = dig(after, path)
    return out


def largest_claims(claims: list) -> dict | None:
    """Each TTFT_CLAIMS field at its largest over these
    memory_governor.claims, None where no sample had any (a tree before
    schema 7)."""
    claims = [c for c in claims if c]
    if not claims:
        return None
    return {field: max(c.get(field, 0) for c in claims) for field in TTFT_CLAIMS}


def ttft_request_record(stream: Stream, recorder: Recorder) -> dict:
    lat = stream.metrics.get("request_latency") or {}
    first = stream.first or stream.ended
    return {
        "sent_s": recorder.seconds(stream.sent),
        "first_s": recorder.seconds(first),
        "ttft_ms": round((first - stream.sent) * 1000, 1),
        "total_ms": round((stream.ended - stream.sent) * 1000, 1),
        "prompt": stream.usage.get("prompt_tokens"),
        "cached": dig(stream.usage, "prompt_tokens_details.cached_tokens"),
        "out": stream.usage.get("completion_tokens"),
        "prefilled": dig(stream.metrics, "prefill.tokens"),
        "cache_status": dig(stream.metrics, "cache.status"),
        "server_ttft_ms": lat.get("ttft_ms"),
        "queue_ms": lat.get("queue_to_start_ms"),
        "start_to_first_ms": lat.get("start_to_first_token_ms"),
        "gap": gap_summary(stream.chunks, recorder.start),
    }


def run_ttft(
    port: int,
    directory: str,
    label: str,
    n: int,
    rounds: int = 3,
    settle: float = 3.0,
    poll_seconds: float = 0.5,
    text: str | None = None,
    timeout: float = REQUEST_SECONDS,
) -> dict:
    """One concurrency level of the TTFT scenario against a fresh server:
    records LABEL.events.jsonl (start, warm-ups, requests, rounds, summary) and
    LABEL.status.jsonl (/status every `poll_seconds`), and returns the
    summary. Each request may take `timeout` seconds (Stream)."""
    if rounds > TTFT_ROUND_SLOTS:
        raise ValueError(f"at most {TTFT_ROUND_SLOTS} rounds")
    text = ttft_corpus() if text is None else text
    model = served_model(port)
    recorder = Recorder(directory, label)
    done = threading.Event()
    # The memory claims of the samples since the round began.
    claims: list = []

    def observe(status):
        claims.append(dig(status, "memory_governor.claims"))

    poller = threading.Thread(
        target=poll_status,
        args=(port, recorder, done.is_set, poll_seconds, observe),
        daemon=True,
    )
    poller.start()
    try:
        recorder.emit(
            "start", n=n, rounds=rounds, model=model, python=sys.version.split()[0]
        )
        wait_idle(port, 0)
        conversations, digest = [], hashlib.sha256()
        for c in range(n):
            warm, slot = ttft_conversation(text, n, c)
            stream = Stream(port, ttft_body(model, warm, 1), timeout)
            if not stream.ok:
                raise RequestFailed(f"warm-up {c}: {stream.status} {stream.error_body}")
            recorder.emit(
                "warmup", n=n, conv=c, **ttft_request_record(stream, recorder)
            )
            conversations.append((warm, slot))
            digest.update(json.dumps(warm).encode())
        records = []
        for r in range(rounds):
            requests = [
                ttft_follow_up(text, warm, slot, r) for warm, slot in conversations
            ]
            for messages in requests:
                digest.update(json.dumps(messages).encode())
            before = wait_idle(port, settle)
            claims.clear()
            barrier, streams = threading.Barrier(n), [None] * n

            def send(i, requests=requests, barrier=barrier, streams=streams):
                try:
                    streams[i] = Stream(
                        port,
                        ttft_body(model, requests[i], TTFT_MAX_TOKENS),
                        timeout,
                        barrier=barrier,
                    )
                except Exception as error:  # the round fails and says why
                    # A sender that failed before the barrier releases the
                    # others waiting there.
                    barrier.abort()
                    streams[i] = error

            threads = [threading.Thread(target=send, args=(i,)) for i in range(n)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            after = wait_idle(port, 0)
            failures = [
                str(s)[:300]
                if isinstance(s, Exception)
                else f"{s.status} {s.error_body}"
                for s in streams
                if isinstance(s, Exception) or not s.ok
            ]
            if failures:
                recorder.emit("round_failed", n=n, round=r, failures=failures)
                continue
            results = [ttft_request_record(s, recorder) for s in streams]
            for i, result in enumerate(results):
                recorder.emit("request", n=n, round=r, conv=i, **result)
            ranked = sorted(results, key=lambda x: x["ttft_ms"])
            record = {
                "n": n,
                "round": r,
                "window_s": [
                    min(x["sent_s"] for x in results),
                    max(x["first_s"] for x in results),
                ],
                "wall_ms": round(
                    (max(s.ended for s in streams) - min(s.sent for s in streams))
                    * 1000,
                    1,
                ),
                "ttft_ms": [x["ttft_ms"] for x in ranked],
                "queue_ms": [round(x["queue_ms"] or 0) for x in ranked],
                "cached": [x["cached"] for x in ranked],
                "prompt": [x["prompt"] for x in ranked],
                "prefilled": [x["prefilled"] for x in ranked],
                **counter_deltas(before, after),
                "memory_governor.claims": largest_claims(
                    [*claims, dig(after, "memory_governor.claims")]
                ),
            }
            recorder.emit("round", **record)
            records.append(record)
        summary = {
            "n": n,
            "rounds": len(records),
            "prompts_sha256": digest.hexdigest()[:16],
        }
        if records:
            every = [t for record in records for t in record["ttft_ms"]]
            summary.update(
                ttft_median_ms=round(statistics.median(every), 1),
                ttft_max_ms=max(every),
                round_worst_ms=[max(record["ttft_ms"]) for record in records],
                cached_min=min(c for record in records for c in record["cached"]),
            )
        recorder.emit("summary", **summary)
        return summary
    finally:
        done.set()
        poller.join(timeout=poll_seconds + 35)
        recorder.close()


# --------------------------------------------------------------- serve1


# Prompt sizes, in tokens, of the one-request-at-a-time scenario.
SERVE1_SIZES = (300, 2048, 14336)
SERVE1_OUTPUT_TOKENS = 256
# A prompt of churn_doc words ("amber-123456") takes DOC_FRAME_TOKENS +
# DOC_TOKENS_PER_WORD x words tokens under the Qwen tokenizers, which split
# digits one by one: fitted to the prompt_tokens the servers recorded, 75
# words -> 662 and 512 -> 4,190 (serve1, smoke-s-serve1-pass-1-132500 and
# smoke-a-mini-serve1-cand-1-201707), 2,000 -> 16,205-16,259 and 7,300 ->
# 59,043-59,144 (churn first rounds, smoke-a-fn-churn-A1 and
# smoke-a-s-fn-churn-18g-cand-A1-193918). The frame is the chat template
# and the prompt's own text around the words.
DOC_TOKENS_PER_WORD = 8.09
DOC_FRAME_TOKENS = 50


def doc_words(tokens: int) -> int:
    """The churn_doc words of a prompt of about `tokens` tokens, at most
    that many."""
    return max(1, int((tokens - DOC_FRAME_TOKENS) / DOC_TOKENS_PER_WORD))


def serve1_messages(size: int, sample: int) -> list[dict]:
    log = churn_doc(10**6 + size * 100 + sample, doc_words(size))
    return [
        {
            "role": "user",
            "content": f"Log s{size}-{sample}:\n{log}\nIgnore the log. Write the numbers "
            f"from 1 to {CHURN_OUTPUT_TOKENS}, one per line, nothing else.",
        }
    ]


def run_serve1(
    port: int,
    directory: str,
    label: str,
    samples: int = 3,
    settle: float = 2.0,
    sizes=SERVE1_SIZES,
    timeout: float = REQUEST_SECONDS,
) -> None:
    """Each size's cold prompt `samples` times, one request at a time on an
    idle server: time to first token and decode milliseconds per token. The
    /status each request waited idle for, and the one after the last, are
    the run's samples. Each request may take `timeout` seconds (Stream)."""
    model = served_model(port)
    recorder = Recorder(directory, label)
    try:
        for sample in range(samples):
            for size in sizes:
                recorder.sample(wait_idle(port, settle))
                body = json.dumps(
                    {
                        "model": model,
                        "messages": serve1_messages(size, sample),
                        "max_tokens": SERVE1_OUTPUT_TOKENS,
                        "temperature": 0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                )
                stream = Stream(port, body, timeout)
                if not stream.ok or not stream.chunks:
                    raise RequestFailed(
                        f"serve1 {size}: {stream.status} {stream.error_body}"
                    )
                out = stream.usage.get("completion_tokens") or 0
                decode = (
                    (stream.chunks[-1] - stream.chunks[0]) * 1000 / (out - 1)
                    if out > 1
                    else None
                )
                recorder.emit(
                    "request",
                    size=size,
                    sample=sample,
                    ttft_ms=round((stream.chunks[0] - stream.sent) * 1000, 1),
                    decode_ms_per_token=round(decode, 3) if decode else None,
                    prompt=stream.usage.get("prompt_tokens"),
                    cached=dig(stream.usage, "prompt_tokens_details.cached_tokens"),
                    out=out,
                )
        recorder.sample(wait_idle(port, 0))
        recorder.emit("end")
    finally:
        recorder.close()


# ----------------------------------------------------------------- warm


# The warm-suffix protocol: per trial a fresh cold
# prefix, cached by one request, then suffixes of about 300 tokens after it,
# the first WARM_DISCARDED of them discarded; each suffix's prompt time is the
# server's start-to-first-token time (timings.prompt_ms).
WARM_DISCARDED = 2
WARM_SUFFIXES = 8
WARM_SUFFIX_TOKENS = 300


def warm_prefix(depth: int, trial: int) -> list[dict]:
    log = churn_doc(2 * 10**6 + depth + trial, doc_words(depth))
    return [
        {"role": "user", "content": f"Archive {depth}-{trial}:\n{log}\nReply with OK."}
    ]


def warm_suffix(prefix: list[dict], depth: int, trial: int, index: int) -> list[dict]:
    log = churn_doc(
        3 * 10**6 + depth * 100 + trial * 10 + index, doc_words(WARM_SUFFIX_TOKENS)
    )
    return prefix + [
        {"role": "assistant", "content": "OK."},
        {
            "role": "user",
            "content": f"Note {index}:\n{log}\nIn one sentence, what is it?",
        },
    ]


def run_warm(
    port: int,
    directory: str,
    label: str,
    depths,
    trial: int,
    settle: float = 2.0,
    timeout: float = REQUEST_SECONDS,
) -> None:
    """The warm-suffix trial `trial` at each cached depth: the same texts for
    every tree, so trees compare trial by trial. The /status each request
    waited idle for, and the one after the last, are the run's samples.
    Each request may take `timeout` seconds (Stream)."""
    model = served_model(port)
    recorder = Recorder(directory, label)
    try:
        for depth in depths:
            prefix = warm_prefix(depth, trial)
            recorder.sample(wait_idle(port, settle))
            cached = Stream(port, ttft_body(model, prefix, 1), timeout)
            if not cached.ok:
                raise RequestFailed(
                    f"warm {depth}: {cached.status} {cached.error_body}"
                )
            for index in range(WARM_DISCARDED + WARM_SUFFIXES):
                recorder.sample(wait_idle(port, settle))
                stream = Stream(
                    port,
                    ttft_body(model, warm_suffix(prefix, depth, trial, index), 1),
                    timeout,
                )
                if not stream.ok:
                    raise RequestFailed(
                        f"warm {depth} suffix {index}: {stream.status} {stream.error_body}"
                    )
                recorder.emit(
                    "warm",
                    depth=depth,
                    trial=trial,
                    index=index,
                    discarded=index < WARM_DISCARDED,
                    prompt_ms=dig(
                        stream.metrics, "request_latency.start_to_first_token_ms"
                    ),
                    prompt=stream.usage.get("prompt_tokens"),
                    cached=dig(stream.usage, "prompt_tokens_details.cached_tokens"),
                )
        recorder.sample(wait_idle(port, 0))
        recorder.emit("end")
    finally:
        recorder.close()

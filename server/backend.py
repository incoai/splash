"""Native request submission, cancellation, recovery and completion ownership."""

import copy
import functools
import queue
import threading
import time
from dataclasses import dataclass, field

from tokenizers.decoders import DecodeStream

from . import json_codec
from . import protocol as wire
from . import runtime as engine_runtime
from .constraints import TokenConstraint
from .diagnostics import print_status
from .errors import APIError, ConstraintError
from .latency import RequestLatency
from .metrics import metrics_dict
from .output import hold_partial
from .tool_schema import (
    CALL_OPEN,
    THINK_END_TOKEN_ID,
    TOOL_CALL_OPEN_TOKEN_ID,
    ToolPolicy,
)

# A failure counts toward a crash loop unless its engine served this long.
CRASH_LOOP_WINDOW_SECONDS = 60.0
# Consecutive such failures, failed relaunches included, that stop relaunching.
CRASH_LOOP_LIMIT = 3
# The n-th consecutive relaunch waits RESTART_BACKOFF_SECONDS * 2**(n-2); the
# first waits 0.
RESTART_BACKOFF_SECONDS = 5.0


# Control requests use a short live probe and explicitly label stale snapshots.
STATUS_REFRESH_TIMEOUT_SECONDS = 0.05


def remaining_request_time(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise APIError(504, "request timed out", "request_timeout")
    # Callers wait this long, and waits reject a timeout above TIMEOUT_MAX;
    # a request without a deadline has infinite time left.
    return min(remaining, threading.TIMEOUT_MAX)


@dataclass(frozen=True)
class CacheInfo:
    status: str = "unknown"
    matched_tokens: int = 0
    lane: int = -1


@dataclass
class NativeResult:
    reason: str
    prompt_tokens: int
    completion_tokens: int
    start_to_first_token_ms: float
    first_token_to_done_ms: float
    request_wall_ms: float
    prefill_tokens: int = 0
    cache: CacheInfo = field(default_factory=CacheInfo)
    stop_sequence: str | None = None
    first_token_batch_tokens: int = 0
    # Raw option logits for score-only jobs, in requested token order.
    option_logits: tuple = ()

    @functools.cached_property
    def metrics(self):
        """The per-request metrics every response and log record reports."""
        return metrics_dict(self)


@dataclass
class Job:
    request_id: int
    prompt_tokens: list
    max_new_tokens: int
    seed: int
    sampling: wire.SamplingParameters
    deadline: float
    priority: wire.RequestPriority = wire.RequestPriority.NORMAL
    stop_sequences: tuple[str, ...] = ()
    thinking: bool = False
    thinking_display: str = "summarized"
    reasoning_tokens: int = 0
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    tool_policy: ToolPolicy | None = None
    response_validator: object | None = None
    response_format: dict | None = None
    constraint: TokenConstraint | None = None
    cache: CacheInfo = field(default_factory=CacheInfo)
    # Image placeholder spans with their grids and digests, plus the
    # concatenated resized pixels the engine encodes during prefill.
    image_spans: tuple = ()
    image_pixels: bytes = b""
    image_owner: object | None = None
    public_id: str = ""
    created_at: int = field(default_factory=lambda: int(time.time()))
    response_store: bool = False
    # (count, short hash) of the normalized tool list, so the console shows
    # when a client's tool block changes between turns and breaks its prefix.
    tools_signature: tuple | None = None
    response_previous_id: str | None = None
    response_history_items: list | None = None
    # Option token ids for score-only jobs; empty means ordinary generation.
    score_tokens: tuple = ()
    # Trailing prompt tokens of the chat template's generation prompt; zero
    # when unknown.
    generation_prompt_tokens: int = 0
    flags: wire.RequestFlag = wire.RequestFlag(0)
    # The request asked for more output than the context leaves, and
    # max_new_tokens was lowered to what it leaves.
    output_clamped_to_context: bool = False
    # The digest of a judgment's rendered prompt, which its response reports.
    prompt_sha256: str | None = None
    latency: RequestLatency | None = None

    @property
    def may_call_tools(self):
        """Whether the output may call a tool: tools are offered under a
        choice other than none."""
        return self.tool_policy is not None and bool(self.tool_policy.schemas)


class CallbackStreamer:
    def __init__(self, tokenizer, callback, stop_sequences=(), on_stop=None):
        self.tokenizer = tokenizer
        self.callback = callback
        self.stop_sequences = tuple(stop_sequences)
        self.on_stop = on_stop
        self.backend = tokenizer.backend_tokenizer
        self.decode_stream = DecodeStream(skip_special_tokens=True)
        self.token_ids = []
        self.emitted = []
        self.pending_text = ""
        self.stop_sequence = None

    def _send(self, text):
        if text:
            self.callback(text)
            self.emitted.append(text)

    def _emit(self, text):
        if not text or self.stop_sequence is not None:
            return
        if not self.stop_sequences:
            self._send(text)
            return
        self.pending_text += text
        matches = [
            (self.pending_text.find(stop), stop)
            for stop in self.stop_sequences
            if stop in self.pending_text
        ]
        if matches:
            offset, self.stop_sequence = min(matches, key=lambda match: match[0])
            self._send(self.pending_text[:offset])
            self.pending_text = ""
            if self.on_stop is not None:
                self.on_stop()
            return

        retained = max(
            len(hold_partial(self.pending_text, stop)[1])
            for stop in self.stop_sequences
        )
        ready = self.pending_text[:-retained] if retained else self.pending_text
        self.pending_text = self.pending_text[-retained:] if retained else ""
        self._send(ready)

    def put_tokens(self, token_ids):
        if self.stop_sequence is not None:
            return
        for token_id in token_ids:
            token_id = int(token_id)
            self.token_ids.append(token_id)
            if self.decode_stream is None:
                continue
            try:
                text = self.decode_stream.step(self.backend, token_id)
            except Exception as error:
                # tokenizers raises a plain Exception with this message when an
                # incremental decode cannot continue; fall back to decoding the
                # whole sequence at the end.
                if not str(error).startswith("Invalid prefix encountered"):
                    raise
                self.decode_stream = None
                continue
            if text:
                self._emit(text)
                if self.stop_sequence is not None:
                    break

    def end(self):
        if self.stop_sequence is not None:
            return
        decoded = self.tokenizer.decode(
            self.token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        handled = "".join(self.emitted) + self.pending_text
        if not decoded.startswith(handled):
            raise RuntimeError("incremental tokenizer output diverged")
        # Bytes of a multi-byte character the output ends inside decode to
        # U+FFFD; DecodeStream held them back for the rest. Drop the trailing
        # U+FFFD, as vLLM's detokenizer does; one that text follows stays.
        self._emit(decoded[len(handled) :].rstrip("\ufffd"))
        if self.stop_sequence is None:
            self._send(self.pending_text)
            self.pending_text = ""

    def count_reasoning_tokens(self, enabled, tool_calls=False):
        """The tokens before the reasoning's end, as ReasoningSplitter reads
        the text: its close, or where a call may follow, a call's opening."""
        if not enabled:
            return 0
        for index, token in enumerate(self.token_ids):
            if token == THINK_END_TOKEN_ID or (
                tool_calls
                and token == TOOL_CALL_OPEN_TOKEN_ID
                and self._opens_call(index)
            ):
                return index
        return len(self.token_ids)

    def _opens_call(self, index):
        """Whether the text from the call-open token at `index` on begins
        CALL_OPEN, decoded from as few tokens as decide it."""
        text = ""
        for end in range(index + 1, len(self.token_ids) + 1):
            text = self.tokenizer.decode(self.token_ids[index:end])
            if len(text) >= len(CALL_OPEN) or not CALL_OPEN.startswith(text):
                break
        return text.startswith(CALL_OPEN)


@dataclass
class _JobState:
    job: Job
    streamer: CallbackStreamer
    call: object | None = None
    terminal_enqueued: bool = False
    detached: bool = False
    shutdown_requested: bool = False
    first_token_batch_tokens: int = 0

    def detach(self):
        self.detached = True
        self.call = None
        self.streamer.on_stop = None


class NativeBackend:
    """Write admitted requests directly to native inference.

    A CPU finalizer runs tokenizer flushes and request logging off the reader
    thread.
    """

    _FINISH_NAMES = {
        wire.FinishReason.STOP: "stop",
        wire.FinishReason.LENGTH: "length",
        wire.FinishReason.CANCELLED: "cancelled",
    }

    def __init__(self, runtime, tokenizer, request_logger):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.request_logger = request_logger
        self.active = {}
        self.closing = False
        self.lock = threading.RLock()
        self.status_snapshot = None
        self.status_snapshot_at = None
        # Since when the engine loop has left status probes unanswered.
        self.status_unanswered_since = None
        # Recovery state, under self.lock: the one background worker, when a
        # relaunch is owed (monotonic), whether a status refresh is owed,
        # consecutive failures without a full window of service, and why
        # relaunching stopped for good.
        self.recovery = None
        self.restart_due = None
        self.refresh_due = False
        self.failures = 0
        self.fatal_error = None
        # Set by close() to cut a relaunch's backoff short.
        self.wakeup = threading.Event()
        # Why the engine cannot serve: its failure, the last failed restart or
        # why restarts stopped.
        self.engine_error = None
        self.terminals = queue.Queue()
        self.finalizer = threading.Thread(
            target=self._finalize_loop,
            name="splash-http-finalizer",
            daemon=True,
        )
        self.finalizer.start()
        runtime.on_engine_failure = self._engine_failed

    def _engine_failed(self, error, served_seconds):
        """The runtime's failure listener: an engine that reached Ready failed
        after serving for `served_seconds`."""
        self._failed("Engine failed", error, served_seconds)

    def _failed(self, event, error, served_seconds):
        # Clients get the reason with every refusal until a status succeeds.
        with self.lock:
            if self.closing or self.fatal_error is not None:
                return
            self.engine_error = str(error)
            if self.status_snapshot is not None:
                # A failed engine's readiness is no longer evidence.
                self.status_snapshot["ready"] = False
            if (unchangeable := self.runtime.fatal_error) is not None:
                # Changed limits would recur at every relaunch; the error says
                # how to recover.
                self._stop_restarting_locked(str(unchangeable))
            elif (delay := self._restart_delay_locked(served_seconds)) is None:
                # The runtime writes an engine's crash trace before it reports
                # that engine's failure, to the listener or from wait_ready.
                trace = self.runtime.last_crash_trace
                self._stop_restarting_locked(
                    f"the inference engine failed {self.failures} times in a row, "
                    f"each within {CRASH_LOOP_WINDOW_SECONDS:.0f} s of starting "
                    f"({error}); Splash stopped restarting it. "
                    + (
                        f"Crash trace: {trace}. "
                        if trace
                        else "Set SPLASH_CRASH_TRACE=1 to record a crash trace. "
                    )
                    + "Restart the Splash server after fixing the cause."
                )
            else:
                self.restart_due = time.monotonic() + delay
            stopped = self.fatal_error
            self._ensure_recovery_locked()
        print_status(f"{event} · {error}", error=True)
        if stopped is not None:
            print_status(f"Engine stopped · {stopped}", error=True)

    def _restart_delay_locked(self, served_seconds):
        """Seconds until the relaunch a failure owes; None once a crash loop
        stops relaunching."""
        if served_seconds >= CRASH_LOOP_WINDOW_SECONDS:
            self.failures = 0
        self.failures += 1
        if self.failures >= CRASH_LOOP_LIMIT:
            return None
        if self.failures == 1:
            return 0.0
        return RESTART_BACKOFF_SECONDS * 2 ** (self.failures - 2)

    def _stop_restarting_locked(self, message):
        """Stop relaunching for good; there is no way back but a server
        restart."""
        self.restart_due = None
        self.fatal_error = message
        self.engine_error = message

    def _ensure_recovery_locked(self):
        """Start the recovery worker when work is owed and none runs.

        Whoever finds work owed calls this, so a worker that failed to start
        is started by the next probe or request."""
        if (
            self.closing
            or self.fatal_error is not None
            or (self.restart_due is None and not self.refresh_due)
            or (self.recovery is not None and self.recovery.is_alive())
        ):
            return
        self.recovery = threading.Thread(
            target=self._recover,
            name="splash-engine-recovery",
            daemon=True,
        )
        try:
            self.recovery.start()
        except BaseException:
            self.recovery = None
            raise

    def _recover(self):
        """Run owed relaunches, then owed status refreshes, until none is
        owed."""
        while True:
            with self.lock:
                if self.closing or self.fatal_error is not None:
                    self.recovery = None
                    return
                due = self.restart_due
                refresh = due is None and self.refresh_due
                if refresh:
                    self.refresh_due = False
                elif due is None:
                    self.recovery = None
                    return
            if due is not None:
                if not self.wakeup.wait(max(0.0, due - time.monotonic())):
                    self._relaunch(due)
                continue
            try:
                restarts = self.runtime.restart_count
                event = self.runtime.status(
                    timeout=engine_runtime.STATUS_ANSWER_LIMIT_SECONDS,
                    fail_unanswered=True,
                )
                self._cache_status(self._decode_status_event(event), restarts)
            except Exception as error:
                # The snapshot stays. A refresh left unanswered has failed its
                # engine, whose relaunch is now owed; after any other failure
                # no status request is left waiting.
                if not isinstance(error, TimeoutError):
                    with self.lock:
                        self.status_unanswered_since = None

    def _relaunch(self, due):
        try:
            # It returns once the engine is Ready. The failure listener counts
            # a failure after Ready, even one before wait_ready returns.
            self.runtime.wait_ready()
        except Exception as error:
            self._failed("Engine restart failed", error, 0.0)
            return
        with self.lock:
            # A failure after Ready may already owe the next relaunch.
            if self.restart_due == due:
                self.restart_due = None
            self.refresh_due = True

    def refusal(self):
        """The error a request not yet admitted gets now; None while the
        engine serves."""
        with self.lock:
            if self.closing:
                return APIError(503, "server is shutting down", "server_shutdown")
            if self.fatal_error is not None:
                return APIError(500, self.fatal_error, "engine_failed")
            self._ensure_recovery_locked()
            if self.runtime.ready:
                return None
            failure = self.engine_error
        return APIError(
            503,
            "engine is recovering; retry shortly"
            + (f" (last failure: {failure})" if failure else ""),
            "engine_recovering",
        )

    def is_ready(self):
        # The engine's own ready folds in memory pressure and Metal health.
        return self.status()["ready"] is True

    @staticmethod
    def _decode_status_event(event):
        snapshot = json_codec.loads(event.json)
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema_version") != wire.STATUS_SCHEMA_VERSION
        ):
            raise ValueError("native status does not match the current schema")
        return snapshot

    def _cache_status(self, snapshot, restarts):
        """Cache `snapshot` unless the engine that answered it has failed or
        been replaced since `restarts` was read; return whether it was cached."""
        with self.lock:
            # An answer from an engine that has failed since is no evidence,
            # even once a relaunched one serves. Ready is read first, so an
            # unchanged count means the engine found Ready is the one that
            # answered.
            if (
                self.closing
                or not self.runtime.ready
                or self.runtime.restart_count != restarts
            ):
                return False
            self.status_snapshot = copy.deepcopy(snapshot)
            self.status_snapshot_at = time.monotonic()
            self.status_unanswered_since = None
            restarted = self.engine_error is not None
            self.engine_error = None
        if restarted:
            print_status("Engine restarted")
        return True

    def status(self, timeout=STATUS_REFRESH_TIMEOUT_SECONDS):
        stale_error = None
        stale_age_ms = None
        with self.lock:
            # The worker refreshes status unless a relaunch is owed. A probe
            # would wait behind its request in a busy loop.
            refresh_pending = (
                self.recovery is not None
                and self.restart_due is None
                and self.status_snapshot is not None
            )
        try:
            if refresh_pending:
                raise TimeoutError("native status refresh is pending")
            restarts = self.runtime.restart_count
            event = self.runtime.status(timeout=timeout)
            snapshot = self._decode_status_event(event)
        except Exception as error:
            stale_error = error
            busy = isinstance(error, TimeoutError)
            now = time.monotonic()
            with self.lock:
                snapshot = copy.deepcopy(self.status_snapshot)
                captured_at = self.status_snapshot_at
                transport_ready = not self.closing and self.runtime.ready
                # A probe skipped behind the refresh sent nothing to answer.
                if busy and not refresh_pending:
                    if self.status_unanswered_since is None:
                        self.status_unanswered_since = now
                    # The refresh waits as long as a busy loop may take, and
                    # fails the engine of a loop that never answers.
                    if transport_ready:
                        self.refresh_due = True
                unanswered = self.status_unanswered_since
                self._ensure_recovery_locked()
            if snapshot is None or captured_at is None:
                snapshot = {
                    "schema_version": wire.STATUS_SCHEMA_VERSION,
                    "ready": False,
                }
            else:
                stale_age_ms = max(0.0, (now - captured_at) * 1000.0)
                # A busy loop keeps its last readiness until it has left a
                # status request unanswered as long as a refresh waits for one;
                # any other failure is not evidence of readiness.
                snapshot["ready"] = (
                    busy
                    and snapshot.get("ready") is True
                    and (
                        unanswered is None
                        or now - unanswered < engine_runtime.STATUS_ANSWER_LIMIT_SECONDS
                    )
                )
        else:
            if not self._cache_status(snapshot, restarts):
                snapshot["ready"] = False
        with self.lock:
            transport_ready = not self.closing and self.runtime.ready
            engine_error = self.engine_error
            stopped = self.fatal_error is not None
        snapshot["transport"] = {
            "ready": transport_ready,
            "recovering": not self.closing and not stopped and not transport_ready,
            "stopped": stopped,
            "pending": self.runtime.pending_count,
            "pending_limit": self.runtime.pending_limit,
            "restarts": self.runtime.restart_count,
            "last_crash_trace": self.runtime.last_crash_trace,
            "status_stale": stale_error is not None,
            "status_age_ms": (
                0.0 if stale_error is None or stale_age_ms is None else stale_age_ms
            ),
        }
        if stale_error is not None:
            snapshot["transport"]["error"] = engine_error or str(stale_error)
        if not transport_ready:
            snapshot["ready"] = False
        return snapshot

    @staticmethod
    def _mask_provider(job):
        if job.constraint is None:
            return None

        def provide(event):
            # runtime sends the simulated context sequence, not merely draft
            # proposals: [] for the initial mask, then [pending anchor,
            # draft...] for verification. TokenConstraint returns the mask
            # before the first simulated token and after each token.
            return job.constraint.masks(event.simulation_tokens)

        return provide

    def _generation_request(self, job):
        constraint = (
            wire.ConstraintMode.TOKEN_MASK
            if job.constraint is not None
            else wire.ConstraintMode.NONE
        )
        frame = wire.RequestFrame(
            request_id=0,
            priority=job.priority,
            absolute_deadline_unix_micros=0,
            remaining_deadline_micros=0,
            logical_max_output_tokens=job.max_new_tokens,
            prompt_tokens=tuple(job.prompt_tokens),
            sampling=job.sampling,
            seed=job.seed,
            constraint=constraint,
            image_spans=job.image_spans,
            image_pixels=job.image_pixels,
            score_tokens=job.score_tokens,
            generation_prompt_tokens=job.generation_prompt_tokens,
            flags=job.flags,
        )
        return engine_runtime.GenerationRequest(
            frame, job.deadline, self._mask_provider(job), job.image_owner
        )

    def submit(self, job):
        state = None

        def stop_matched():
            call = None
            with self.lock:
                call = state.call
            if call is not None:
                call.cancel()

        def emit(text):
            job.events.put(("text", text))

        streamer = CallbackStreamer(
            self.tokenizer, emit, job.stop_sequences, stop_matched
        )
        state = _JobState(job, streamer)
        request = self._generation_request(job)
        # The engine copies the pixels at admission; only the frame written
        # below needs them, and the job lives until the response ends.
        job.image_pixels = b""

        def on_event(call, event):
            with self.lock:
                if state.detached:
                    return
                if state.call is None:
                    state.call = call
                cancel = job.cancelled.is_set() or self.closing
            # A raise here is the call's callback error, which cancels it.
            self._on_event(state, event)
            if cancel:
                call.cancel()

        def on_complete(call):
            with self.lock:
                if state.detached or state.terminal_enqueued:
                    return
                if state.call is None:
                    state.call = call
                state.terminal_enqueued = True
                self.terminals.put((state, call))

        with self.lock:
            if self.closing:
                state.detach()
                job.events.put(
                    (
                        "error",
                        APIError(
                            503,
                            "server is shutting down",
                            "server_shutdown",
                        ),
                    )
                )
                return
            self.active[job.request_id] = state
        try:
            call = self.runtime.submit(
                request, on_event=on_event, on_complete=on_complete
            )
            with self.lock:
                if not state.detached:
                    state.call = call
                cancel = state.detached or self.closing or job.cancelled.is_set()
            if cancel:
                call.cancel()
        except Exception as error:
            with self.lock:
                deliver = not state.detached and not state.terminal_enqueued
                if deliver:
                    self._detach_locked(state)
            if deliver:
                refusal = None
                if isinstance(
                    error,
                    (
                        engine_runtime.EngineUnhealthy,
                        engine_runtime.ProtocolFatal,
                        engine_runtime.RuntimeClosed,
                    ),
                ):
                    # Not admitted: refused as a request arriving now would be.
                    refusal = self.refusal()
                job.events.put(("error", refusal or self._api_error(error)))

    def _detach_locked(self, state):
        state.detach()
        if self.active.get(state.job.request_id) is state:
            del self.active[state.job.request_id]

    def _on_event(self, state, event):
        job = state.job
        if isinstance(event, wire.StartEvent):
            cache = CacheInfo(
                "hit" if event.matched_prompt_tokens else "miss",
                event.matched_prompt_tokens,
                event.lane,
            )
            with self.lock:
                job.cache = cache
            job.events.put(("start", None))
        elif isinstance(event, wire.PromptProgressEvent):
            job.events.put(
                (
                    "progress",
                    {
                        "total": len(job.prompt_tokens),
                        "cache": job.cache.matched_tokens,
                        "processed": event.processed_tokens,
                        "time_ms": event.elapsed_micros / 1000.0,
                    },
                )
            )
        elif isinstance(event, wire.TokensEvent):
            if job.latency is not None and event.tokens:
                job.latency.tokens()
            if event.sequence_offset == 0:
                state.first_token_batch_tokens = len(event.tokens)
            if job.constraint is not None:
                job.constraint.commit(event.tokens)
            state.streamer.put_tokens(event.tokens)

    def cancel(self, job):
        call = None
        with self.lock:
            job.cancelled.set()
            state = self.active.get(job.request_id)
            if state is not None:
                call = state.call
        if call is not None:
            call.cancel()

    def _finalize_loop(self):
        while True:
            item = self.terminals.get()
            if item is None:
                return
            state, call = item
            self._finalize(state, call)
            # Do not keep the finished request alive while waiting for the
            # next terminal: its image batch returns the request budget only
            # once nothing references it.
            del item, state, call

    def _finalize(self, state, call):
        job = state.job
        error = None
        result = None
        try:
            done = call.result(0)
            if call.callback_error is not None:
                raise self._api_error(call.callback_error)
            if (
                job.constraint is not None
                and done.reason != wire.FinishReason.CANCELLED
            ):
                # The grammar checks what was generated after the last mask. A
                # cancelled request needs no check, and its last mask may
                # still be computing.
                job.constraint.finish()
            state.streamer.end()
            job.reasoning_tokens = state.streamer.count_reasoning_tokens(
                job.thinking, job.may_call_tools
            )
            stop_sequence = state.streamer.stop_sequence
            result = NativeResult(
                reason=(
                    "stop"
                    if stop_sequence is not None
                    else self._FINISH_NAMES[done.reason]
                ),
                prompt_tokens=done.prompt_tokens,
                completion_tokens=(
                    len(state.streamer.token_ids)
                    if stop_sequence is not None
                    else done.completion_tokens
                ),
                option_logits=done.option_logits,
                start_to_first_token_ms=done.prefill_micros / 1000.0,
                first_token_to_done_ms=done.decode_micros / 1000.0,
                request_wall_ms=done.wall_micros / 1000.0,
                prefill_tokens=max(0, done.prompt_tokens - job.cache.matched_tokens),
                cache=job.cache,
                stop_sequence=stop_sequence,
                first_token_batch_tokens=state.first_token_batch_tokens,
            )
            if job.latency is not None:
                latency = result.metrics["request_latency"]
                queued = latency.get("queue_to_start_ms")
                if queued is not None:
                    job.latency.metrics.observe("native_queue", queued / 1000.0)
        except (engine_runtime.EngineUnhealthy, engine_runtime.ProtocolFatal):
            # An admitted request ends with EngineUnhealthy or ProtocolFatal
            # only when the engine running it fails: it stopped or broke the
            # protocol. The listener has already decided whether that engine
            # restarts; the console names the failure.
            with self.lock:
                fatal_error = self.fatal_error
            if fatal_error is not None:
                error = APIError(500, fatal_error, "engine_failed")
            else:
                error = APIError(
                    503,
                    "the inference engine stopped unexpectedly and is restarting; "
                    "retry the request",
                    "runtime_unavailable",
                )
        except Exception as unexpected:
            error = self._api_error(unexpected)
        finally:
            with self.lock:
                shutdown_requested = state.shutdown_requested
                self._detach_locked(state)
        if shutdown_requested:
            # Once close() has claimed an active HTTP request, its public
            # terminal is deterministic even if the native cancellation and
            # RuntimeClosed delivery race each other. A completion delivered
            # before close() acquires the state lock remains a normal result.
            result = None
            error = APIError(503, "server is shutting down", "server_shutdown")
        self._record(job, result=result, error=error)
        job.events.put(("error", error) if error else ("done", result))

    @staticmethod
    def _api_error(error):
        if isinstance(error, APIError):
            return APIError(error.status, error.message, error.code)
        if isinstance(error, ConstraintError):
            return APIError(400, str(error), "constraint_error")
        if isinstance(error, engine_runtime.RequestFailed):
            code = error.code.decode("ascii", "replace")
            message = error.message_bytes.decode("utf-8", "replace")
            if code == "deadline_exceeded":
                return APIError(504, message, "request_timeout")
            if code == "capacity_exhausted":
                return APIError(
                    400,
                    "the request does not fit in the memory this server may use, even "
                    "after every cached prefix was evicted; restart the server with a "
                    f"larger --max-memory or a smaller --max-context ({message})",
                    code,
                )
            request_codes = {
                "invalid_constraint",
                "invalid_count",
                "invalid_deadline",
                "invalid_enum_value",
                "invalid_request",
                "invalid_sampling",
                "limit_exceeded",
            }
            status = 503 if error.retryable else 400 if code in request_codes else 500
            return APIError(status, message, code)
        if isinstance(error, engine_runtime.MaskComputationFailed):
            if error.retryable:
                return APIError(503, str(error), "runtime_busy")
            return APIError(400, str(error), "constraint_error")
        if isinstance(error, engine_runtime.PendingLimitExceeded):
            # The native pending limit is --queue-size, the HTTP request
            # gate's capacity, so only a race with the gate reaches it, as
            # when a cancelled request still holds its native slot: an
            # overload like the gate's own, retried the same way.
            return APIError(503, "request queue is full", "frontend_overloaded")
        if isinstance(
            error,
            (
                engine_runtime.EngineUnhealthy,
                engine_runtime.ProtocolFatal,
                engine_runtime.RuntimeClosed,
            ),
        ):
            return APIError(503, str(error), "runtime_unavailable")
        if isinstance(error, TimeoutError):
            return APIError(504, "request timed out", "request_timeout")
        return APIError(500, str(error), "runtime_error")

    def _record(self, job, result=None, error=None):
        record = {
            "outcome": result.reason if result else "error",
            "prompt_tokens": len(job.prompt_tokens),
        }
        if result:
            record["completion_tokens"] = result.completion_tokens
            record["metrics"] = result.metrics
        if job.tools_signature:
            record["tools"] = {
                "count": job.tools_signature[0],
                "signature": job.tools_signature[1],
            }
        if error:
            record["error_code"] = error.code
        try:
            self.request_logger(record)
        except Exception:
            pass

    def close(self):
        with self.lock:
            if self.closing:
                return
            self.closing = True
            calls = []
            for state in self.active.values():
                state.shutdown_requested = True
                calls.append(state.call)
        self.wakeup.set()
        for call in calls:
            if call is not None:
                call.cancel()
        try:
            # Also fails a relaunch in progress.
            self.runtime.close()
        finally:
            with self.lock:
                recovery = self.recovery
            if recovery is not None and recovery is not threading.current_thread():
                recovery.join(timeout=1.0)
            with self.lock:
                stranded = []
                for state in list(self.active.values()):
                    if state.terminal_enqueued:
                        continue
                    self._detach_locked(state)
                    stranded.append(state)
            for state in stranded:
                error = APIError(503, "server is shutting down", "server_shutdown")
                self._record(state.job, error=error)
                state.job.events.put(("error", error))
            self.terminals.put(None)
            self.finalizer.join()

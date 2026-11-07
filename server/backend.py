"""Native request submission, cancellation, recovery and completion ownership."""

import copy
import functools
import math
import queue
import threading
import time
from dataclasses import dataclass, field, replace

from tokenizers.decoders import DecodeStream

from . import json_codec, supervision
from . import protocol as wire
from . import runtime as engine_runtime
from .chat_format import THINK_END, ChatFormat
from .constraints import TokenConstraint
from .diagnostics import print_status
from .errors import APIError, ConstraintError
from .latency import RequestLatency
from .metrics import metrics_dict
from .output import StopSearch
from .tool_schema import ToolPolicy

# Engine outcomes whose message the engine words from its own limits and
# counts, never from a request: the client gets it as it is, and the console
# prints it beside the code.
_ENGINE_RESOURCE_CODES = frozenset(
    {"capacity_exhausted", "kv_disk_quota", "kv_write_budget", "resource_timeout"}
)


class ThinkingEndError(ValueError):
    """A thinking end, --thinking-end-text or the chat template's close, the
    model's tokenizer cannot end thinking with."""


def thinking_end_tokens(tokenizer, text, chat_format):
    """The tokens the engine selects where a thinking budget runs out: those
    of the text, --thinking-end-text or else the close the chat template
    writes after a reasoning (ChatTemplate.thinking_end), which must encode
    as text then chat_format's close of thinking as its own token, at most
    wire.MAX_THINKING_END_TOKENS tokens. The text holds none of the
    tokenizer's added tokens, the template's special tokens. None without a
    text, where the close alone ends thinking (NativeBackend)."""
    if text is None:
        return None
    tokens = tuple(tokenizer.encode(text, add_special_tokens=False))
    if (
        not tokens
        or tokens[-1] != chat_format.think_end_id
        or not tokenizer.added_tokens_decoder.keys().isdisjoint(tokens[:-1])
    ):
        raise ThinkingEndError(
            f"the thinking end {text!r} must end with {THINK_END} and hold no "
            "other special token"
        )
    if len(tokens) > wire.MAX_THINKING_END_TOKENS:
        raise ThinkingEndError(
            f"the thinking end {text!r} takes {len(tokens)} tokens; the most is "
            f"{wire.MAX_THINKING_END_TOKENS}"
        )
    return tokens


@dataclass(frozen=True)
class CacheInfo:
    status: str = "unknown"
    matched_tokens: int = 0
    lane: int = -1
    # Tokens whose KV the request read from cached blocks another request
    # computed, past its cached prefix; known once it is done.
    deduplicated_tokens: int = 0


@dataclass(frozen=True)
class ReasoningRequest:
    """How a chat request asked to reason: the reasoning effort it sent, None
    when absent; the effort its template rendered with, the server's default
    for an absent one and None for the template's own; and its
    chat_template_kwargs enable_thinking, None when unset."""

    sent: str | None
    effort: str | None
    enable_thinking: bool | None


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
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    draft_cycles: int = 0
    # The engine ended the response at a bound of its own, short of
    # max_new_tokens: what the SSD cache or memory holds of its context.
    bounded: bool = False

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
    # A chat request's reasoning options, which its Started line shows; None
    # for a request without a chat template.
    reasoning: ReasoningRequest | None = None
    # The format a chat request's output is read in; None for a request
    # without a chat template.
    chat_format: ChatFormat | None = None
    # The thinking tokens the job may select before its thinking closes, set
    # only where that comes before the output limit; None for no cap.
    thinking_token_budget: int | None = None
    reasoning_tokens: int = 0
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    tool_policy: ToolPolicy | None = None
    response_validator: object | None = None
    response_format: dict | None = None
    constraint: TokenConstraint | None = None
    cache: CacheInfo = field(default_factory=CacheInfo)
    # Image placeholder spans with their grids and digests, plus the
    # concatenated resized pixels the engine encodes during prefill and their
    # charge against the in-flight image budget, held until the frame is
    # written.
    image_spans: tuple = ()
    image_pixels: bytes = b""
    image_owner: object | None = None
    public_id: str = ""
    created_at: int = field(default_factory=lambda: int(time.time()))
    # (count, short hash) of the normalized tool list, so the console shows
    # when a client's tool block changes between turns and breaks its prefix.
    tools_signature: tuple | None = None
    # Input an API keeps past preparation for its answer, such as the
    # conversation a stored Responses answer is stored with, charged to the
    # request body budget until the job is released (retain_for).
    retained_input: list | None = None
    # Option token ids for score-only jobs; empty means ordinary generation.
    score_tokens: tuple = ()
    # Trailing prompt tokens of the chat template's generation prompt; zero
    # when unknown.
    generation_prompt_tokens: int = 0
    flags: wire.RequestFlag = wire.RequestFlag(0)
    # The request asked for more output than the context leaves, and
    # max_new_tokens was lowered to what it leaves.
    output_clamped_to_context: bool = False
    # The request's cache_salt hashed to 128 bits; zero without one.
    cache_salt: int = 0
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
        self.stops = StopSearch(stop_sequences) if stop_sequences else None
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
        if self.stops is None:
            self._send(text)
            return
        self.pending_text += text
        found = self.stops.search(text)
        if found is not None:
            self.stop_sequence, back = found
            self._send(self.pending_text[: len(self.pending_text) - back])
            self.pending_text = ""
            if self.on_stop is not None:
                self.on_stop()
            return

        retained = self.stops.held
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

    def count_reasoning_tokens(self, chat_format, thinking, tool_calls=False):
        """The reasoning's tokens, as ReasoningSplitter reads the text: from
        the start of the output where the generation prompt opened the
        reasoning (thinking), or after the model's own opening of it
        (chat_format.think_start_id), to its close or, where a call may follow
        and a call's opening ends the reasoning, a call's opening; none for
        a request without a chat format."""
        if chat_format is None:
            return 0
        if thinking:
            start = 0
        elif chat_format.think_start_id is not None and self.token_ids[:1] == [
            chat_format.think_start_id
        ]:
            start = 1
        else:
            return 0
        calls_end = tool_calls and chat_format.calls_end_reasoning
        for index in range(start, len(self.token_ids)):
            token = self.token_ids[index]
            if token == chat_format.think_end_id or (
                calls_end
                and token == chat_format.call_open_id
                and self._opens_call(index, chat_format.call_syntax.call_open)
            ):
                return index - start
        return len(self.token_ids) - start

    def _opens_call(self, index, call_open):
        """Whether the text from the call-open token at `index` on begins
        call_open, decoded from as few tokens as decide it."""
        text = ""
        for end in range(index + 1, len(self.token_ids) + 1):
            text = self.tokenizer.decode(self.token_ids[index:end])
            if len(text) >= len(call_open) or not call_open.startswith(text):
                break
        return text.startswith(call_open)


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


def _failure_code(error):
    """How an engine failed (supervision.EngineFailure): the code its
    failure carries, engine_unhealthy for one the runtime did not class."""
    if isinstance(
        error, (engine_runtime.EngineUnhealthy, engine_runtime.ProtocolFatal)
    ):
        return error.code
    return "engine_unhealthy"


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

    def __init__(self, runtime, tokenizer, request_logger, thinking_end=None):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.request_logger = request_logger
        # What the engine selects where a thinking budget runs out
        # (thinking_end_tokens); None for the close of thinking alone, in
        # each job's chat format.
        self.thinking_end = thinking_end
        self.active = {}
        self.closing = False
        self.lock = threading.RLock()
        self.status_snapshot = None
        self.status_snapshot_at = None
        # Since when the engine loop has left status probes unanswered.
        self.status_unanswered_since = None
        # Recovery state, under self.lock: the one background worker, when a
        # relaunch is owed (monotonic), whether a status refresh is owed, when
        # to relaunch after a failure, and why relaunching stopped for good.
        self.recovery = None
        self.restart_due = None
        self.refresh_due = False
        self.policy = supervision.RestartPolicy()
        self.stop_reason = None
        # The requests refused for having been in flight at two engine
        # failures in a row, and the last engine failure, until its engine has
        # exited and it is counted (_count_failure).
        self.quarantine = supervision.Quarantine()
        self.failure_to_count = None
        # Set by close() to cut a relaunch's backoff short.
        self.wakeup = threading.Event()
        # Why the engine cannot serve: its failure, the last failed restart or
        # why restarts stopped.
        self.engine_error = None
        # What the finalizer runs, in order: requests' Started lines and
        # terminals, which must not hold up the runtime's reader thread.
        self.finalizer_queue = queue.Queue()
        self.finalizer = threading.Thread(
            target=self._finalize_loop,
            name="splash-http-finalizer",
            daemon=True,
        )
        self.finalizer.start()
        runtime.on_engine_failure = self._engine_failed

    def _engine_failed(self, error, served_seconds, started, exit_status):
        """The runtime's failure listener (engine_runtime.FailureListener): an
        engine that reached Ready failed after serving for `served_seconds`."""
        self._failed(
            "Engine failed",
            supervision.EngineFailure(
                _failure_code(error),
                str(error),
                served_seconds,
                started=started,
                exit_status=exit_status,
            ),
        )

    def _failed(self, event, failure):
        # Clients get the reason with every refusal until a status succeeds.
        with self.lock:
            if self.closing or self.stop_reason is not None:
                return
            self.engine_error = failure.message
            if self.status_snapshot is not None:
                # A failed engine's readiness is no longer evidence.
                self.status_snapshot["ready"] = False
            if failure.exit_status is not None:
                self.failure_to_count = failure
            delay = self.policy.after(failure)
            if delay is None:
                self._stop_restarting_locked(self._stop_message(failure))
            else:
                self.restart_due = time.monotonic() + delay
            stopped = self.stop_reason
            self._ensure_recovery_locked()
        print_status(f"{event} · {failure.message}", error=True)
        if stopped is not None:
            print_status(f"Engine stopped · {stopped}", error=True)

    def _stop_message(self, failure):
        """Why relaunching stopped after `failure`, and what to do."""
        kind = failure.start.kind if failure.start is not None else None
        if kind is wire.StartFailureKind.COMMAND_MEMORY:
            return (
                "the engine's start ran out of GPU memory another process holds "
                "(another Splash server?); Splash stopped restarting it. Quit that "
                "process, then restart the Splash server."
            )
        if kind is wire.StartFailureKind.MODEL:
            return (
                f"the model's files can no longer be served ({failure.message}); "
                "Splash stopped restarting the engine. Restore or reinstall the "
                "model, then restart the Splash server."
            )
        cause = (
            failure.message
            if supervision.permanent(failure)
            else f"it failed {supervision.CRASH_LOOP_FAILURES} times in a row, each "
            f"within {self.policy.thresholds.short_lived:.0f} s of starting "
            f"({failure.message})"
        )
        # The runtime writes an engine's crash trace before it reports that
        # engine's failure, to the listener or from wait_ready.
        trace = self.runtime.last_crash_trace
        return (
            f"Splash stopped restarting the engine: {cause}. "
            + (
                f"Crash trace: {trace}. "
                if trace
                else "Set SPLASH_CRASH_TRACE=1 to record a crash trace. "
            )
            + "Restart the Splash server after fixing the cause."
        )

    def _stop_restarting_locked(self, message):
        """Stop relaunching for good; there is no way back but a server
        restart."""
        self.restart_due = None
        self.stop_reason = message
        self.engine_error = message

    def _ensure_recovery_locked(self):
        """Start the recovery worker when work is owed and none runs.

        Whoever finds work owed calls this, so a worker that failed to start
        is started by the next probe or request."""
        if (
            self.closing
            or self.stop_reason is not None
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
                if self.closing or self.stop_reason is not None:
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
                    timeout=self.policy.thresholds.status_answer,
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

    def _count_failure(self):
        """Count the last engine failure toward the quarantine once its engine
        has exited, which it waits for at most the stop grace, and before the
        relaunch admits requests: how the engine exited decides whether a
        request it ran may have caused the failure
        (supervision.request_caused). Its output may end before its exit
        status is there to read."""
        with self.lock:
            failure = self.failure_to_count
            self.failure_to_count = None
        if failure is None or not supervision.request_caused(
            failure.code, failure.exit_status()
        ):
            return
        refused = self.quarantine.after(failure.started)
        if refused:
            print_status(
                f"Quarantine · {len(refused)} request"
                f"{'' if len(refused) == 1 else 's'} in flight at two engine "
                "failures in a row refused until the server restarts",
                error=True,
            )

    def _relaunch(self, due):
        self._count_failure()
        try:
            # It returns once the engine is Ready. The failure listener counts
            # a failure after Ready, even one before wait_ready returns.
            self.runtime.wait_ready()
        except Exception as error:
            self._failed(
                "Engine restart failed",
                supervision.EngineFailure(
                    _failure_code(error),
                    str(error),
                    0.0,
                    start=error.event
                    if isinstance(error, engine_runtime.EngineStartFailed)
                    else None,
                    launch=isinstance(error, engine_runtime.EngineLaunchFailed),
                ),
            )
            return
        with self.lock:
            # A failure after Ready may already owe the next relaunch.
            if self.restart_due == due:
                self.restart_due = None
            self.refresh_due = True

    def _recovery_retry_after_locked(self):
        """The Retry-After of a refusal while the engine restarts: the seconds
        until the next bound on the relaunched engine's start still ahead. It
        should be Ready as long after its spawn as the engine before took to
        start; past that it waits for memory the host or Metal refused, up to
        the start retry window more; past that the start timeout ends it.
        Before the spawn, while the relaunch waits for its backoff or for the
        engine before to exit, it is as long after the later of now and the
        time the relaunch is due; a relaunch not owed yet, before the failure
        listener has run, is due now. At least 1, and at most
        supervision.RETRY_AFTER_LIMIT_SECONDS."""
        now = time.monotonic()
        start = self.runtime.last_start_seconds
        spawned = self.runtime.spawned_at
        if spawned is None:
            due = now if self.restart_due is None else max(self.restart_due, now)
            ready = due + start
        else:
            bounds = (
                spawned + start,
                spawned + start + self.policy.thresholds.start_retry_window,
                spawned + supervision.START_TIMEOUT_SECONDS,
            )
            ready = next((bound for bound in bounds if bound > now), now)
        return min(
            max(math.ceil(ready - now), 1), supervision.RETRY_AFTER_LIMIT_SECONDS
        )

    def refusal(self):
        """The error a request not yet admitted gets now; None while the
        engine serves."""
        with self.lock:
            if self.closing:
                return APIError(503, "server is shutting down", "server_shutdown")
            if self.stop_reason is not None:
                return APIError(500, self.stop_reason, "engine_failed")
            self._ensure_recovery_locked()
            if self.runtime.ready:
                return None
            failure = self.engine_error
            retry_after = self._recovery_retry_after_locked()
        return APIError(
            503,
            "engine is recovering; retry shortly"
            + (f" (last failure: {failure})" if failure else ""),
            "engine_recovering",
            retry_after,
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

    def status(self, timeout=supervision.STATUS_PROBE_SECONDS):
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
                        or now - unanswered < self.policy.thresholds.status_answer
                    )
                )
        else:
            if not self._cache_status(snapshot, restarts):
                snapshot["ready"] = False
        with self.lock:
            transport_ready = not self.closing and self.runtime.ready
            engine_error = self.engine_error
            stopped = self.stop_reason is not None
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
            cache_salt=job.cache_salt,
            thinking_budget=self._thinking_budget(job),
        )
        return engine_runtime.GenerationRequest(
            frame,
            job.deadline,
            self._mask_provider(job),
            wire.request_fingerprint(frame),
        )

    def _thinking_budget(self, job):
        """The engine's cutoff of the thinking a job generates in, which it
        ends with the server's thinking end. The engine's thinking ends at its
        close or, where a call may follow and a call's opening ends the
        reasoning, at any call-open token, while count_reasoning_tokens ends
        the reasoning at a call-open token only where it begins the call's
        opening: a stray call-open token in the thinking ends the engine's
        count, and the reasoning can then run past the budget."""
        if job.thinking_token_budget is None:
            return None
        chat_format = job.chat_format
        return wire.ThinkingBudget(
            job.thinking_token_budget,
            self.thinking_end or (chat_format.think_end_id,),
            chat_format.call_open_id
            if job.may_call_tools and chat_format.calls_end_reasoning
            else chat_format.think_end_id,
        )

    def release_weights(self, timeout=5.0):
        """Asks the engine to give its weights back now; returns its status.

        It releases them only while no request is in service and no restore is
        under way, with `--idle-release off` too; the returned snapshot's
        `weights` says whether they are released: `released` and not
        `restoring`. The next request restores the weights as after the idle
        release.
        """
        with self.lock:
            if self.closing:
                raise APIError(503, "server is shutting down", "server_shutdown")
        restarts = self.runtime.restart_count
        try:
            event = self.runtime.release_weights(timeout=timeout)
        except TimeoutError:
            # The engine answers between commands: it is running a long one.
            raise APIError(
                503,
                f"the engine did not answer within {timeout:g} s; retry shortly",
                "engine_busy",
            ) from None
        except (
            engine_runtime.EngineUnhealthy,
            engine_runtime.ProtocolFatal,
            engine_runtime.RuntimeClosed,
        ) as error:
            raise self.refusal() or self._api_error(error) from None
        snapshot = self._decode_status_event(event)
        self._cache_status(snapshot, restarts)
        return snapshot

    def submit(self, job):
        state = None

        def stop_matched():
            call = None
            with self.lock:
                call = state.call
            if call is not None:
                call.stop()

        def emit(text):
            job.events.put(("text", text))

        streamer = CallbackStreamer(
            self.tokenizer, emit, job.stop_sequences, stop_matched
        )
        state = _JobState(job, streamer)
        request = self._generation_request(job)

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
                self.finalizer_queue.put(functools.partial(self._finalize, state, call))

        try:
            self._admit(state, request, on_event, on_complete)
        finally:
            # The frame, the request's last copy of its pixels, is written
            # or refused: the pixels and their charge end here, not with the
            # response, which may decode for minutes.
            job.image_pixels = b""
            job.image_owner = None

    def _admit(self, state, request, on_event, on_complete):
        job = state.job
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
            if self.quarantine.refuses(request.fingerprint):
                # Retrying fails the same way, as capacity_exhausted does.
                state.detach()
                trace = self.runtime.last_crash_trace
                job.events.put(
                    (
                        "error",
                        APIError(
                            400,
                            "this request was in flight at two engine failures in "
                            "a row; Splash refuses it until the server restarts"
                            + (f" (crash trace: {trace})" if trace else ""),
                            "request_quarantined",
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
            record = {
                "outcome": "started",
                "prompt_tokens": len(job.prompt_tokens),
                "cached_tokens": cache.matched_tokens,
            }
            if job.reasoning is not None:
                record["reasoning"] = {
                    "sent_effort": job.reasoning.sent,
                    "effort": job.reasoning.effort,
                    "enable_thinking": job.reasoning.enable_thinking,
                    "thinking": job.thinking,
                }
            # The finalizer prints it, as it prints the Done line: a console
            # that stops draining must not hold up the reader thread, which
            # delivers every request's tokens and masks.
            self.finalizer_queue.put(functools.partial(self._log, record))
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
        while (work := self.finalizer_queue.get()) is not None:
            work()
            # Do not keep a finished request, its output and its events,
            # alive while waiting for the next one.
            del work

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
                and state.streamer.stop_sequence is None
            ):
                # The grammar checks what was generated after the last mask. A
                # cancelled request, or one a stop sequence ended, needs no
                # check, and its last mask may still be computing.
                job.constraint.finish()
            state.streamer.end()
            job.reasoning_tokens = state.streamer.count_reasoning_tokens(
                job.chat_format, job.thinking, job.may_call_tools
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
                cache=replace(job.cache, deduplicated_tokens=done.deduplicated_tokens),
                stop_sequence=stop_sequence,
                first_token_batch_tokens=state.first_token_batch_tokens,
                drafted_tokens=done.drafted_tokens,
                accepted_draft_tokens=done.accepted_draft_tokens,
                draft_cycles=done.draft_cycles,
                bounded=done.bounded,
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
                if self.stop_reason is not None:
                    error = APIError(500, self.stop_reason, "engine_failed")
                else:
                    error = APIError(
                        503,
                        "the inference engine stopped unexpectedly and is "
                        "restarting; retry the request",
                        "runtime_unavailable",
                        self._recovery_retry_after_locked(),
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
            # A request that does not fit even alone, in memory or in the SSD
            # cache, fails the same way on a retry; the engine's message names
            # what it lacked and the limits that would fit it.
            if code in ("capacity_exhausted", "kv_disk_quota"):
                return APIError(400, message, code)
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
            # The engine says when a retry may succeed where that is not at
            # once, as when the SSD cache's write budget refills.
            return APIError(status, message, code, error.retry_after_seconds or 1)
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
            if error.code in _ENGINE_RESOURCE_CODES:
                record["engine_reason"] = error.message
        self._log(record)

    def _log(self, record):
        # A request's start, when the engine admits it, and its end.
        try:
            self.request_logger(record)
        except Exception:
            pass

    def close(self):
        with self.lock:
            if self.closing:
                return
            self.closing = True
            for state in self.active.values():
                state.shutdown_requested = True
        self.wakeup.set()
        try:
            # Fails every call and a relaunch in progress, and stops the
            # engine, which takes no more work. A cancel per call first could
            # wait the write budget (write_progress) for a frame being written.
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
            self.finalizer_queue.put(None)
            self.finalizer.join()

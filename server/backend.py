"""Native request submission, cancellation, recovery and completion ownership."""

import copy
import queue
import threading
import time
from dataclasses import dataclass, field

from tokenizers.decoders import DecodeStream

if __package__:
    from . import protocol as wire
    from . import runtime as engine_runtime
    from .constraints import TokenConstraint
    from .errors import APIError, NativeError
    from .latency import RequestLatency
    from .metrics import metrics_dict
    from .output import hold_partial
    from .tool_schema import THINK_END, ToolPolicy, strict_json_loads
else:
    import protocol as wire
    from constraints import TokenConstraint
    from errors import APIError, NativeError
    from latency import RequestLatency
    from metrics import metrics_dict
    from output import hold_partial
    from tool_schema import THINK_END, ToolPolicy, strict_json_loads

    import runtime as engine_runtime


# Retry only startup failures before submission; admitted work is never replayed.
NATIVE_RECOVERY_RETRIES = 1
NATIVE_RECOVERY_GRACE_SECONDS = 2.0


# Control requests use a short live probe and explicitly label stale snapshots.
STATUS_REFRESH_TIMEOUT_SECONDS = 0.05
STATUS_BACKGROUND_TIMEOUT_SECONDS = 30.0


REQUEST_PRIORITIES = {"foreground": 0, "normal": 1, "background": 2}
REQUEST_PRIORITY_NAMES = {value: name for name, value in REQUEST_PRIORITIES.items()}


MAX_PROTOCOL_U64 = (1 << 64) - 1


def remaining_request_time(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise APIError(504, "request timed out", "request_timeout")
    return remaining


@dataclass(frozen=True)
class CacheInfo:
    status: str = "unknown"
    matched_tokens: int = 0
    capacity: int = 0
    slot: int = -1


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


@dataclass
class Job:
    request_id: int
    prompt_tokens: list
    max_new_tokens: int
    seed: int
    temperature: float
    top_p: float
    top_k: int
    deadline: float
    priority: int = REQUEST_PRIORITIES["normal"]
    stop_sequences: tuple[str, ...] = ()
    thinking: bool = False
    thinking_display: str = "summarized"
    reasoning_tokens: int = 0
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    timed_out: bool = False
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
    return_progress: bool = False
    # Option token ids for score-only jobs; empty means ordinary generation.
    score_tokens: tuple = ()
    # Endpoint-specific metadata carried to the response builder.
    meta: dict | None = None
    latency: RequestLatency | None = None


class CallbackStreamer:
    def __init__(self, tokenizer, callback, stop_sequences=(), on_stop=None):
        self.tokenizer = tokenizer
        self.callback = callback
        self.stop_sequences = tuple(stop_sequences)
        self.on_stop = on_stop
        self.backend = getattr(tokenizer, "backend_tokenizer", None)
        self.decode_stream = (
            DecodeStream(skip_special_tokens=True) if self.backend is not None else None
        )
        self.token_ids = []
        self.emitted = []
        self.pending_text = ""
        self.stop_sequence = None
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        self.think_end_token = convert(THINK_END) if callable(convert) else None

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
        self._emit(decoded[len(handled) :])
        if self.stop_sequence is None:
            self._send(self.pending_text)
            self.pending_text = ""

    def count_reasoning_tokens(self, enabled):
        if not enabled or not isinstance(self.think_end_token, int):
            return 0
        try:
            return self.token_ids.index(self.think_end_token)
        except ValueError:
            return len(self.token_ids)


@dataclass
class _JobState:
    job: Job
    streamer: CallbackStreamer
    call: object | None = None
    callback_error: Exception | None = None
    terminal_enqueued: bool = False
    detached: bool = False
    shutdown_requested: bool = False
    first_token_batch_tokens: int = 0

    def detach(self):
        self.detached = True
        self.call = None
        self.callback_error = None
        self.streamer.on_stop = None


class NativeBackend:
    """Write admitted requests directly to native inference.

    A CPU finalizer runs tokenizer flushes and request logging off the reader
    thread.
    """

    _CACHE_NAMES = {
        wire.CacheDisposition.MISS: "miss",
        wire.CacheDisposition.PREFIX_HIT: "hit",
    }
    _FINISH_NAMES = {
        wire.FinishReason.STOP: "stop",
        wire.FinishReason.LENGTH: "length",
        wire.FinishReason.CANCELLED: "cancelled",
    }

    def __init__(self, runtime, tokenizer, request_logger=None):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.request_logger = request_logger
        self.active = {}
        self.closing = False
        self.lock = threading.RLock()
        self.status_snapshot = None
        self.status_snapshot_at = None
        self.status_refresh_inflight = False
        self.status_refresh_thread = None
        self.status_refresh_failures = 0
        self.status_refresh_after = 0.0
        self.terminals = queue.Queue()
        self.finalizer = threading.Thread(
            target=self._finalize_loop,
            name="splash-http-finalizer",
            daemon=True,
        )
        self.finalizer.start()

    def can_submit(self):
        with self.lock:
            if self.closing:
                return False
        if not self.runtime.ready:
            self._ensure_background_status_refresh()
            return False
        return True

    def is_ready(self):
        if not self.can_submit():
            return False
        snapshot = self.status()
        pressure = snapshot.get("memory_pressure")
        return snapshot.get("ready") is True and pressure in {"normal", "warning"}

    @staticmethod
    def _decode_status_event(event):
        snapshot = strict_json_loads(event.json)
        if (
            event.schema_version != wire.STATUS_SCHEMA_VERSION
            or not isinstance(snapshot, dict)
            or snapshot.get("schema_version") != wire.STATUS_SCHEMA_VERSION
        ):
            raise ValueError("native status does not match the current schema")
        return snapshot

    def _cache_status(self, snapshot):
        with self.lock:
            if self.closing:
                return
            self.status_snapshot = copy.deepcopy(snapshot)
            self.status_snapshot_at = time.monotonic()
            self.status_refresh_failures = 0
            self.status_refresh_after = 0.0

    def _background_status_refresh(self):
        try:
            if not self.runtime.ready:
                self.runtime.wait_ready()
            event = self.runtime.status(timeout=STATUS_BACKGROUND_TIMEOUT_SECONDS)
            self._cache_status(self._decode_status_event(event))
        except Exception:
            with self.lock:
                self.status_refresh_failures = min(5, self.status_refresh_failures + 1)
                self.status_refresh_after = time.monotonic() + 2 ** (
                    self.status_refresh_failures - 1
                )
        finally:
            with self.lock:
                self.status_refresh_inflight = False
                self.status_refresh_thread = None

    def _ensure_background_status_refresh(self):
        with self.lock:
            if (
                self.closing
                or self.status_refresh_inflight
                or time.monotonic() < self.status_refresh_after
            ):
                return
            thread = threading.Thread(
                target=self._background_status_refresh,
                name="splash-status-refresh",
                daemon=True,
            )
            self.status_refresh_inflight = True
            self.status_refresh_thread = thread
            try:
                thread.start()
            except BaseException:
                self.status_refresh_inflight = False
                self.status_refresh_thread = None
                raise

    def status(self, timeout=STATUS_REFRESH_TIMEOUT_SECONDS):
        stale_error = None
        stale_age_ms = None
        with self.lock:
            refresh_pending = (
                self.status_refresh_inflight and self.status_snapshot is not None
            )
        try:
            if refresh_pending:
                raise TimeoutError("native status refresh is pending")
            event = self.runtime.status(timeout=timeout)
            snapshot = self._decode_status_event(event)
        except Exception as error:
            stale_error = error
            with self.lock:
                snapshot = copy.deepcopy(self.status_snapshot)
                captured_at = self.status_snapshot_at
            if snapshot is None or captured_at is None:
                snapshot = {
                    "schema_version": wire.STATUS_SCHEMA_VERSION,
                    "ready": False,
                }
            else:
                stale_age_ms = max(0.0, (time.monotonic() - captured_at) * 1000.0)
                # A stale snapshot is useful telemetry but never evidence that
                # the service is currently ready.
                snapshot["ready"] = False
            if not self.runtime.ready or isinstance(error, TimeoutError):
                self._ensure_background_status_refresh()
        else:
            self._cache_status(snapshot)
        with self.lock:
            transport_ready = not self.closing and self.runtime.ready
        snapshot["transport"] = {
            "ready": transport_ready,
            "recovering": not self.closing and not transport_ready,
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
            snapshot["transport"]["error"] = str(stale_error)
        metal = snapshot.get("metal")
        if (
            not transport_ready
            or snapshot.get("memory_pressure") == "critical"
            or not isinstance(metal, dict)
            or metal.get("healthy") is not True
        ):
            snapshot["ready"] = False
        return snapshot

    @staticmethod
    def _deadline(job):
        remaining = remaining_request_time(job.deadline)
        wall_micros = time.time_ns() // 1000
        maximum_remaining = MAX_PROTOCOL_U64 - wall_micros
        if remaining >= maximum_remaining / 1_000_000:
            remaining_micros = maximum_remaining
        else:
            remaining_micros = max(1, int(remaining * 1_000_000))
        return engine_runtime.Deadline(
            wall_micros + remaining_micros,
            remaining_micros,
        )

    @staticmethod
    def _mask_provider(job):
        if job.constraint is None:
            return None

        def provide(event):
            # runtime sends the simulated context sequence, not merely draft
            # proposals: [] for the initial mask, then [pending anchor,
            # draft...] for verification. TokenConstraint returns the mask
            # before the first simulated token and after each token.
            payload = job.constraint.masks(event.simulation_tokens)
            expected_bytes = event.words_per_mask * event.mask_rows * 4
            if len(payload) != expected_bytes:
                raise NativeError(
                    "constraint_error",
                    f"grammar produced {len(payload)} mask bytes; "
                    f"expected {expected_bytes}",
                )
            return payload

        return provide

    def _generation_request(self, job):
        priority = wire.RequestPriority(job.priority)
        if job.constraint is not None:
            cohort = wire.Cohort.CONSTRAINED
            constraint = wire.ConstraintMode.TOKEN_MASK
        elif job.temperature > 0:
            cohort = wire.Cohort.SAMPLING
            constraint = wire.ConstraintMode.NONE
        else:
            cohort = wire.Cohort.GREEDY
            constraint = wire.ConstraintMode.NONE
        return engine_runtime.GenerationRequest(
            prompt_tokens=tuple(job.prompt_tokens),
            logical_max_output_tokens=job.max_new_tokens,
            deadline=self._deadline(job),
            priority=priority,
            sampling=wire.SamplingParameters(
                float(job.temperature), float(job.top_p), job.top_k
            ),
            seed=job.seed,
            cohort=cohort,
            constraint=constraint,
            mask_provider=self._mask_provider(job),
            image_spans=job.image_spans,
            image_pixels=job.image_pixels,
            image_owner=job.image_owner,
            return_progress=job.return_progress,
            score_tokens=job.score_tokens,
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
        try:
            request = self._generation_request(job)
        except APIError as error:
            state.detach()
            job.events.put(("error", self._api_error(error)))
            return True

        def on_event(call, event):
            with self.lock:
                if state.detached:
                    return
                if state.call is None:
                    state.call = call
                cancel = job.cancelled.is_set() or self.closing
            self._on_event(state, call, event)
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
                return True
            self.active[job.request_id] = state
        try:
            recovery_attempt = 0
            while True:
                try:
                    call = self.runtime.submit(
                        request, on_event=on_event, on_complete=on_complete
                    )
                    break
                except engine_runtime.EngineUnhealthy:
                    with self.lock:
                        retry = (
                            recovery_attempt < NATIVE_RECOVERY_RETRIES
                            and not state.detached
                            and not state.terminal_enqueued
                            and not self.closing
                            and not job.cancelled.is_set()
                        )
                    if not retry:
                        raise
                    recovery_attempt += 1
                    # Do not sleep while holding transport state. Concurrent
                    # requests remain independent, and cancellation ends the
                    # recovery grace promptly.
                    if job.cancelled.wait(NATIVE_RECOVERY_GRACE_SECONDS):
                        raise
                    # Refresh the relative deadline for the replacement
                    # generation while preserving the original absolute job
                    # deadline.
                    request = self._generation_request(job)
            with self.lock:
                if not state.detached:
                    state.call = call
                cancel = state.detached or self.closing or job.cancelled.is_set()
            if cancel:
                call.cancel()
            return True
        except engine_runtime.PendingLimitExceeded:
            with self.lock:
                reject = not state.detached and not state.terminal_enqueued
                if reject:
                    self._detach_locked(state)
            return not reject
        except Exception as error:
            with self.lock:
                deliver = not state.detached and not state.terminal_enqueued
                if deliver:
                    self._detach_locked(state)
            if deliver:
                job.events.put(("error", self._api_error(error)))
            return True

    def _detach_locked(self, state):
        state.detach()
        if self.active.get(state.job.request_id) is state:
            del self.active[state.job.request_id]

    def _on_event(self, state, call, event):
        job = state.job
        try:
            if isinstance(event, wire.StartEvent):
                cache = CacheInfo(
                    self._CACHE_NAMES[event.cache_disposition],
                    event.matched_prompt_tokens,
                    event.capacity_tokens,
                    event.slot_index,
                )
                with self.lock:
                    job.cache = cache
                job.events.put(("start", cache.status))
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
                    job.constraint.consume(event.tokens)
                state.streamer.put_tokens(event.tokens)
        except Exception as error:
            with self.lock:
                if not state.detached and state.callback_error is None:
                    state.callback_error = error
            call.cancel()

    def cancel(self, job, timed_out=False):
        call = None
        with self.lock:
            job.timed_out |= timed_out
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
            native = call.result(0)
            if state.callback_error is not None:
                raise self._api_error(state.callback_error)
            if call.callback_errors:
                raise self._api_error(call.callback_errors[0])
            state.streamer.end()
            job.reasoning_tokens = state.streamer.count_reasoning_tokens(job.thinking)
            done = native.done
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
                latency = metrics_dict(result)["request_latency"]
                queued = latency.get("queue_to_start_ms")
                if queued is not None:
                    job.latency.metrics.observe("native_queue", queued / 1000.0)
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
        if isinstance(error, NativeError):
            status = (
                400
                if error.code in ("bad_request", "unsupported", "constraint_error")
                else 500
            )
            return APIError(status, error.message, error.code)
        if isinstance(error, engine_runtime.RequestFailed):
            code = error.code.decode("ascii", "replace")
            message = error.message_bytes.decode("utf-8", "replace")
            if code == "deadline_exceeded":
                return APIError(504, message, "request_timeout")
            request_codes = {
                "integer_overflow",
                "invalid_cohort_constraint",
                "invalid_deadline",
                "invalid_enum_value",
                "invalid_request",
                "invalid_request_id",
                "invalid_sampling",
                "limit_exceeded",
            }
            status = 503 if error.retryable else 400 if code in request_codes else 500
            return APIError(status, message, code)
        if isinstance(error, engine_runtime.CapacityExhausted):
            return APIError(503, str(error), "capacity_exhausted")
        if isinstance(error, engine_runtime.MaskComputationFailed):
            return APIError(400, str(error), "constraint_error")
        if isinstance(
            error, (engine_runtime.EngineUnhealthy, engine_runtime.RuntimeClosed)
        ):
            return APIError(503, str(error), "runtime_unavailable")
        if isinstance(error, engine_runtime.ProtocolFatal):
            return APIError(500, str(error), "protocol_error")
        if isinstance(error, TimeoutError):
            return APIError(504, "request timed out", "request_timeout")
        return APIError(500, str(error), "runtime_error")

    def _record(self, job, result=None, error=None):
        if self.request_logger is None:
            return
        record = {
            "event": "request",
            "request_id": job.request_id,
            "outcome": result.reason if result else "error",
            "prompt_tokens": len(job.prompt_tokens),
            "priority": REQUEST_PRIORITY_NAMES[job.priority],
        }
        if result:
            record["completion_tokens"] = result.completion_tokens
            record["metrics"] = metrics_dict(result)
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
        for call in calls:
            if call is not None:
                call.cancel()
        try:
            self.runtime.close()
        finally:
            with self.lock:
                status_thread = self.status_refresh_thread
            if (
                status_thread is not None
                and status_thread is not threading.current_thread()
            ):
                status_thread.join(timeout=1.0)
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

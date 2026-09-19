"""Thin multiplexed subprocess client for the Splash native protocol.

The native scheduler owns execution. Calls accepted under ``pending_limit``
are written directly to the child; the Python executor handles CPU grammar masks.

Callbacks can run on the reader thread and must return quickly. Blocking
callers should use ``RuntimeCall.result`` on their own thread.
"""

from __future__ import annotations

import itertools
import math
import os
import selectors
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import BinaryIO, Callable, Protocol, Sequence, TypeAlias

if __package__:
    from . import protocol as wire
    from .crash_trace import CrashTraceRing
else:  # Direct execution from the server directory.
    import protocol as wire
    from crash_trace import CrashTraceRing


class ProcessLike(Protocol):
    stdin: BinaryIO
    stdout: BinaryIO
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class EngineRuntimeError(RuntimeError):
    def restate(self) -> EngineRuntimeError:
        """Copy this failure without retaining a traceback.

        Stored failures are shared across calls. Re-raising one exception
        retains each traceback's requests and prepared images; store and raise
        fresh copies instead.
        """
        return type(self)(*self.args)


class RuntimeClosed(EngineRuntimeError):
    pass


class PendingLimitExceeded(EngineRuntimeError):
    pass


class RequestFailed(EngineRuntimeError):
    def __init__(
        self,
        request_id: int,
        code: bytes,
        message: bytes,
        *,
        retryable: bool = False,
    ):
        self.request_id = request_id
        self.code = code
        self.message_bytes = message
        self.retryable = retryable
        text = message.decode("utf-8", errors="replace")
        machine_code = code.decode("ascii", errors="replace")
        super().__init__(f"request {request_id} failed [{machine_code}]: {text}")

    def restate(self) -> RequestFailed:
        return RequestFailed(
            self.request_id,
            self.code,
            self.message_bytes,
            retryable=self.retryable,
        )


class CapacityExhausted(EngineRuntimeError):
    retryable = True

    def __init__(self, event: wire.CapacityExhaustedEvent):
        self.event = event
        super().__init__(
            f"request {event.request_id} could not allocate its KV target "
            f"(target_pages={event.required_kv_pages}, "
            f"logical_pages_free={event.available_kv_pages}); retry after "
            "other requests finish or system memory becomes available"
        )

    def restate(self) -> CapacityExhausted:
        return CapacityExhausted(self.event)


class EngineUnhealthy(EngineRuntimeError):
    pass


class ProtocolFatal(EngineRuntimeError):
    pass


class MaskComputationFailed(EngineRuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class Deadline:
    absolute_unix_micros: int
    remaining_micros: int

    @classmethod
    def after(
        cls,
        seconds: float,
        *,
        wall_time_ns: Callable[[], int] = time.time_ns,
    ) -> Deadline:
        if seconds <= 0:
            raise ValueError("deadline duration must be positive")
        duration_micros = max(1, int(seconds * 1_000_000))
        return cls(wall_time_ns() // 1000 + duration_micros, duration_micros)


MaskProvider: TypeAlias = Callable[[wire.MaskRequestEvent], Sequence[int] | bytes]
EventCallback: TypeAlias = Callable[["RuntimeCall", wire.Message], None]
CompletionCallback: TypeAlias = Callable[["RuntimeCall"], None]

_REQUIRED_READY_FEATURES = int(
    wire.ReadyFeature.CANCELLATION
    | wire.ReadyFeature.TOKEN_MASKS
    | wire.ReadyFeature.STATUS_JSON
    | wire.ReadyFeature.MULTIPLEXING
)
_READ_CHUNK_BYTES = 64 * 1024


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("native operation timed out")
    return remaining


@dataclass(slots=True, frozen=True)
class GenerationRequest:
    prompt_tokens: tuple[int, ...]
    logical_max_output_tokens: int
    deadline: Deadline
    priority: wire.RequestPriority = wire.RequestPriority.NORMAL
    sampling: wire.SamplingParameters = field(default_factory=wire.SamplingParameters)
    seed: int = 0
    cohort: wire.Cohort = wire.Cohort.GREEDY
    constraint: wire.ConstraintMode = wire.ConstraintMode.NONE
    mask_provider: MaskProvider | None = None
    # Image spans in prompt order with their concatenated resized pixels.
    image_spans: tuple[wire.ImageSpan, ...] = ()
    image_pixels: bytes = b""
    # Keeps the frontend's byte reservation alive across cancellation and CPU
    # mask work. It is ownership only and is never serialized to the engine.
    image_owner: object | None = None
    return_progress: bool = False
    # Option token ids for score-only requests; empty means generation.
    score_tokens: tuple[int, ...] = ()


@dataclass(slots=True, frozen=True)
class GenerationResult:
    request_id: int
    start: wire.StartEvent | None
    tokens: tuple[int, ...]
    done: wire.DoneEvent


class RuntimeCall:
    """Future-like handle for one directly admitted native request."""

    def __init__(
        self,
        client: MultiplexedRuntime,
        request_id: int,
        generation: int,
        request: GenerationRequest,
        on_event: EventCallback | None,
        on_complete: CompletionCallback | None,
    ):
        self._client = client
        self.request_id = request_id
        self.generation = generation
        self.request = request
        self._on_event = on_event
        self._completion_callbacks = []
        if on_complete:
            self._completion_callbacks.append(on_complete)
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._result: GenerationResult | None = None
        self._error: EngineRuntimeError | None = None
        self._start: wire.StartEvent | None = None
        self._progress: wire.PromptProgressEvent | None = None
        self._token_chunks: dict[int, tuple[int, ...]] = {}
        self._next_token_offset = 0
        self._mask_error: MaskComputationFailed | None = None
        self._callback_errors: list[BaseException] = []
        self._cancel_requested = False
        self._cancel_timer = None

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @property
    def cancel_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested

    @property
    def callback_errors(self) -> tuple[BaseException, ...]:
        with self._lock:
            return tuple(self._callback_errors)

    def result(self, timeout: float | None = None) -> GenerationResult:
        if not self._event.wait(timeout):
            raise TimeoutError(f"request {self.request_id} did not finish in time")
        with self._lock:
            if self._error:
                raise self._error.restate()
            assert self._result is not None
            return self._result

    def cancel(self) -> bool:
        with self._lock:
            if self._event.is_set() or self._cancel_requested:
                return False
            self._cancel_requested = True
        self._client._cancel_call(self)
        return True

    def _emit(self, message: wire.Message) -> None:
        with self._lock:
            callback = self._on_event
        if callback is None:
            return
        try:
            callback(self, message)
        except BaseException as error:
            self._record_callback_error(error)

    def _record_callback_error(self, error: BaseException) -> None:
        # Keep diagnostics without retaining the callback's request frames.
        error.__traceback__ = None
        error.__context__ = None
        error.__cause__ = None
        with self._lock:
            self._callback_errors.append(error)

    def _invoke_completion(self, callback: CompletionCallback) -> None:
        try:
            callback(self)
        except BaseException as error:
            self._record_callback_error(error)

    def _record_start(self, event: wire.StartEvent) -> bool:
        with self._lock:
            if self._event.is_set() or self._start is not None:
                return False
            self._start = event
        self._emit(event)
        return True

    def _record_progress(self, event: wire.PromptProgressEvent) -> str | None:
        with self._lock:
            if not self.request.return_progress:
                return "unsolicited prompt progress"
            if self._event.is_set() or self._start is None or self._next_token_offset:
                return "prompt progress outside prefill"
            if (
                not self._start.matched_prompt_tokens
                <= event.processed_tokens
                <= len(self.request.prompt_tokens)
            ):
                return "prompt progress is outside the request's token range"
            if self._progress is not None and (
                event.processed_tokens <= self._progress.processed_tokens
                or event.elapsed_micros < self._progress.elapsed_micros
            ):
                return "prompt progress moved backwards or repeated"
            self._progress = event
        self._emit(event)
        return None

    def _record_tokens(self, event: wire.TokensEvent) -> str | None:
        with self._lock:
            if self._event.is_set():
                return "TokensEvent arrived after the request became terminal"
            if self._start is None:
                return "TokensEvent arrived before StartEvent"
            if event.sequence_offset != self._next_token_offset:
                relation = (
                    "overlaps or duplicates prior tokens"
                    if event.sequence_offset < self._next_token_offset
                    else "leaves a gap in the token stream"
                )
                return (
                    f"TokensEvent offset {event.sequence_offset} {relation}; "
                    f"expected {self._next_token_offset}"
                )
            next_offset = self._next_token_offset + len(event.tokens)
            if next_offset > self.request.logical_max_output_tokens:
                return (
                    f"TokensEvent stream length {next_offset} exceeds logical "
                    f"maximum {self.request.logical_max_output_tokens}"
                )
            self._token_chunks[event.sequence_offset] = event.tokens
            self._next_token_offset = next_offset
        self._emit(event)
        return None

    def _record_mask_error(self, error: MaskComputationFailed) -> None:
        with self._lock:
            if not self._event.is_set() and self._mask_error is None:
                self._mask_error = error

    def _build_result(self, done: wire.DoneEvent) -> GenerationResult:
        with self._lock:
            prompt_tokens = len(self.request.prompt_tokens)
            logical_max = self.request.logical_max_output_tokens
            completion_tokens = self._next_token_offset
            if done.prompt_tokens != prompt_tokens:
                raise ProtocolFatal(
                    f"DoneEvent prompt count {done.prompt_tokens} does not match "
                    f"request count {prompt_tokens}"
                )
            if done.completion_tokens > logical_max:
                raise ProtocolFatal(
                    f"DoneEvent completion count {done.completion_tokens} exceeds "
                    f"logical maximum {logical_max}"
                )
            if done.completion_tokens != completion_tokens:
                raise ProtocolFatal(
                    f"DoneEvent completion count {done.completion_tokens} does not "
                    f"match streamed count {completion_tokens}"
                )
            if self._start is None and not (
                self._cancel_requested
                and done.reason is wire.FinishReason.CANCELLED
                and not completion_tokens
            ):
                raise ProtocolFatal("DoneEvent arrived before StartEvent")
            if (
                done.reason is wire.FinishReason.LENGTH
                and completion_tokens != logical_max
            ):
                raise ProtocolFatal(
                    f"length-finished DoneEvent has {completion_tokens} tokens; "
                    f"expected logical maximum {logical_max}"
                )
            expected_scores = len(self.request.score_tokens)
            if expected_scores:
                if done.decode_micros:
                    raise ProtocolFatal(
                        "score DoneEvent reported autoregressive decoding"
                    )
                if done.reason not in (
                    wire.FinishReason.STOP,
                    wire.FinishReason.CANCELLED,
                ):
                    raise ProtocolFatal("score DoneEvent has an invalid finish reason")
                if done.reason is wire.FinishReason.STOP:
                    if len(done.option_logits) != expected_scores:
                        raise ProtocolFatal(
                            f"score DoneEvent returned {len(done.option_logits)} "
                            f"option logits; expected {expected_scores}"
                        )
                elif done.option_logits:
                    raise ProtocolFatal(
                        "unfinished score DoneEvent returned option logits"
                    )
            elif done.option_logits:
                raise ProtocolFatal(
                    "DoneEvent returned option logits for a generation request"
                )
            tokens = tuple(
                token
                for offset in sorted(self._token_chunks)
                for token in self._token_chunks[offset]
            )
            return GenerationResult(self.request_id, self._start, tokens, done)

    def _terminal_mask_error(self) -> MaskComputationFailed | None:
        with self._lock:
            return self._mask_error

    def _set_terminal(
        self,
        *,
        result: GenerationResult | None = None,
        error: EngineRuntimeError | None = None,
    ) -> bool:
        with self._lock:
            if self._event.is_set():
                return False
            self._result = result
            self._error = error
            self._on_event = None
            callbacks = tuple(self._completion_callbacks)
            self._completion_callbacks.clear()
            self._event.set()
            if self._cancel_timer is not None:
                self._cancel_timer.cancel()
                self._cancel_timer = None
        for callback in callbacks:
            self._invoke_completion(callback)
        return True


@dataclass(slots=True)
class _StatusWaiter:
    generation: int
    event: threading.Event = field(default_factory=threading.Event)
    result: wire.StatusJsonEvent | None = None
    error: EngineRuntimeError | None = None


@dataclass(slots=True)
class _StartupAttempt:
    deadline: float
    event: threading.Event = field(default_factory=threading.Event)
    generation: int | None = None
    error: EngineRuntimeError | None = None


class MultiplexedRuntime:
    """One-reader, direct-admission client for the native protocol."""

    _shutdown_grace_seconds = 15.0
    # Allow the native 120-second command watchdog to finish before fencing it.
    _cancel_grace_seconds = 150.0

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        process_factory: Callable[[], ProcessLike] | None = None,
        startup_timeout: float = 30.0,
        io_timeout: float = 5.0,
        pending_limit: int = 64,
        mask_workers: int | None = None,
        eager_start: bool = True,
    ):
        if process_factory is None and not command:
            raise ValueError("command or process_factory is required")
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        if not math.isfinite(io_timeout) or io_timeout <= 0:
            raise ValueError("io_timeout must be positive")
        if pending_limit <= 0:
            raise ValueError("pending_limit must be positive")
        if mask_workers is not None and mask_workers <= 0:
            raise ValueError("mask_workers must be positive")

        self._command = tuple(command) if command else None
        self._process_factory = process_factory or self._default_process_factory
        self._startup_timeout = startup_timeout
        self._io_timeout = io_timeout
        self._pending_limit = pending_limit
        self._admission_slots = threading.BoundedSemaphore(pending_limit)
        # Native cancellation frees a request slot before its CPU mask job
        # necessarily finishes. Bound queued + running jobs independently.
        self._mask_slots = threading.BoundedSemaphore(pending_limit)
        self._mask_executor = ThreadPoolExecutor(
            max_workers=mask_workers,
            thread_name_prefix="splash-mask",
        )

        self._state_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._closed = False
        self._process: ProcessLike | None = None
        self._reader_thread: threading.Thread | None = None
        self._generation = 0
        self._ever_started = False
        self._restart_count = 0
        self._startup_attempt: _StartupAttempt | None = None
        self._ready_message: wire.ReadyEvent | None = None
        self._context_limit: int | None = None
        self._terminal_error: EngineRuntimeError | None = None
        self._pending: dict[int, RuntimeCall] = {}
        self._status_waiters: dict[int, _StatusWaiter] = {}
        self._last_status_id = 0
        self._last_status: wire.StatusJsonEvent | None = None
        self._request_ids = itertools.count(1)
        self._status_ids = itertools.count(1)
        self._crash_trace = CrashTraceRing(
            self._command, enabled=os.environ.get("SPLASH_CRASH_TRACE") == "1"
        )

        if eager_start:
            self._ensure_process()

    def _default_process_factory(self) -> ProcessLike:
        assert self._command is not None
        return subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )

    @property
    def pending_limit(self) -> int:
        return self._pending_limit

    @property
    def pending_count(self) -> int:
        with self._state_lock:
            return len(self._pending)

    @property
    def restart_count(self) -> int:
        with self._state_lock:
            return self._restart_count

    @property
    def readiness(self) -> wire.ReadyEvent | None:
        with self._state_lock:
            return self._ready_message

    @property
    def ready(self) -> bool:
        with self._state_lock:
            process = self._process
            return (
                not self._closed
                and process is not None
                and process.poll() is None
                and self._ready_message is not None
                and self._terminal_error is None
            )

    @property
    def last_status(self) -> wire.StatusJsonEvent | None:
        with self._state_lock:
            return self._last_status

    @property
    def last_crash_trace(self) -> str | None:
        path = self._crash_trace.last_dump
        return str(path) if path is not None else None

    def __enter__(self) -> MultiplexedRuntime:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def wait_ready(self, timeout: float | None = None) -> bool:
        self._ensure_process(timeout=timeout)
        return self.ready

    def submit(
        self,
        request: GenerationRequest,
        *,
        on_event: EventCallback | None = None,
        on_complete: CompletionCallback | None = None,
    ) -> RuntimeCall:
        """Admit and immediately write one request without client scheduling."""

        deadline = (
            time.monotonic()
            + min(
                request.deadline.remaining_micros,
                request.deadline.absolute_unix_micros - time.time_ns() // 1000,
            )
            / 1_000_000
        )
        request_id = next(self._request_ids)
        if not self._admission_slots.acquire(blocking=False):
            raise PendingLimitExceeded(
                f"pending request limit {self._pending_limit} is full"
            )

        call: RuntimeCall | None = None
        try:
            # Admission must precede serialization, which copies image payloads.
            _remaining(deadline)
            protocol_request = wire.RequestFrame(
                request_id=request_id,
                priority=request.priority,
                absolute_deadline_unix_micros=request.deadline.absolute_unix_micros,
                remaining_deadline_micros=request.deadline.remaining_micros,
                logical_max_output_tokens=request.logical_max_output_tokens,
                prompt_tokens=request.prompt_tokens,
                sampling=request.sampling,
                seed=request.seed,
                cohort=request.cohort,
                constraint=request.constraint,
                image_spans=request.image_spans,
                image_pixels=request.image_pixels,
                return_progress=request.return_progress,
                score_tokens=request.score_tokens,
            )
            try:
                encoded = wire.serialize_message(protocol_request)
            except wire.ProtocolError as error:
                raise self._request_protocol_error(request_id, error.issue) from error
            generation = self._ensure_process(timeout=_remaining(deadline))
            _remaining(deadline)
            with self._state_lock:
                self._require_generation_ready_locked(generation)
                # Created under the lock so that a call exists exactly when
                # _pending owns its admission slot; the release below relies
                # on that.
                call = RuntimeCall(
                    self,
                    request_id,
                    generation,
                    request,
                    on_event,
                    on_complete,
                )
                self._pending[request_id] = call
            self._write_bytes(encoded, generation, call=call, deadline=deadline)
            return call
        except BaseException:
            if call is not None:
                removed = False
                with self._state_lock:
                    removed = self._pending.pop(request_id, None) is call
                if removed:
                    self._admission_slots.release()
            else:
                self._admission_slots.release()
            raise

    def status(self, timeout: float = 5.0) -> wire.StatusJsonEvent:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("status timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._state_lock:
            if self._closed:
                raise RuntimeClosed("runtime is closed")
            generation = self._generation
            self._require_generation_ready_locked(generation)
            correlation_id = next(self._status_ids)
            self._last_status_id = correlation_id
            waiter = _StatusWaiter(generation)
            self._status_waiters[correlation_id] = waiter
        try:
            encoded = wire.serialize_message(wire.StatusRequestFrame(correlation_id))
            self._write_bytes(encoded, generation, deadline=deadline)
        except BaseException:
            with self._state_lock:
                self._status_waiters.pop(correlation_id, None)
            raise
        if not waiter.event.wait(max(0.0, deadline - time.monotonic())):
            expired = False
            with self._state_lock:
                if self._status_waiters.pop(correlation_id, None) is waiter:
                    expired = True
            if expired:
                raise TimeoutError("native status response timed out")
        if waiter.error:
            raise waiter.error.restate()
        assert waiter.result is not None
        return waiter.result

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            generation = self._generation
            process = self._process
            reader = self._reader_thread
            attempt = self._startup_attempt
            if attempt is not None and not attempt.event.is_set():
                attempt.error = RuntimeClosed("runtime is closed")
                attempt.event.set()
        if process is not None:
            self._fail_generation(generation, RuntimeClosed("runtime is closed"))
            self._stop_process(process, reader)
        with self._state_lock:
            self._process = None
            self._reader_thread = None
            self._ready_message = None
        self._mask_executor.shutdown(wait=True, cancel_futures=True)

    def _request_protocol_error(
        self, request_id: int, issue: wire.ProtocolIssue
    ) -> EngineRuntimeError:
        if issue.failure_class is wire.FailureClass.REQUEST_ERROR:
            return RequestFailed(
                request_id,
                wire.issue_code_name(issue.code).encode(),
                issue.message.encode(),
            )
        if issue.failure_class is wire.FailureClass.ENGINE_UNHEALTHY:
            return EngineUnhealthy(issue.describe())
        return ProtocolFatal(issue.describe())

    def _ensure_process(self, timeout: float | None = None) -> int:
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("ready timeout must be positive")
        caller_deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            stale_generation: int | None = None
            owner = False
            with self._state_lock:
                if self._closed:
                    raise RuntimeClosed("runtime is closed")
                process = self._process
                if (
                    process is not None
                    and process.poll() is None
                    and self._ready_message is not None
                    and self._terminal_error is None
                ):
                    return self._generation
                attempt = self._startup_attempt
                if attempt is not None and not attempt.event.is_set():
                    pass
                elif process is not None and self._terminal_error is None:
                    stale_generation = self._generation
                else:
                    attempt = _StartupAttempt(time.monotonic() + self._startup_timeout)
                    self._startup_attempt = attempt
                    old_process = process
                    old_reader = self._reader_thread
                    owner = True

            if stale_generation is not None:
                # Deliver terminal callbacks without any startup ownership
                # held. A completion callback may safely re-enter submit().
                self._fail_generation(
                    stale_generation,
                    EngineUnhealthy("native process is no longer ready"),
                )
                continue

            assert attempt is not None
            if owner:
                # A caller can leave without cancelling the shared startup.
                # The control thread ends on Ready, failure or the startup deadline.
                threading.Thread(
                    target=self._run_startup_attempt,
                    args=(attempt, old_process, old_reader),
                    name="splash-native-startup",
                    daemon=True,
                ).start()
            deadline = min(attempt.deadline, caller_deadline or attempt.deadline)
            if not attempt.event.wait(max(0.0, deadline - time.monotonic())):
                if time.monotonic() < attempt.deadline:
                    raise TimeoutError("native ready wait timed out")
                self._expire_startup_attempt(attempt)
            with self._state_lock:
                if attempt.error is not None:
                    raise attempt.error.restate()
                if self._closed:
                    raise RuntimeClosed("runtime is closed")
                if (
                    attempt.generation == self._generation
                    and self._process is not None
                    and self._process.poll() is None
                    and self._ready_message is not None
                    and self._terminal_error is None
                ):
                    return self._generation
                if self._terminal_error is not None:
                    raise self._terminal_error.restate()
                raise EngineUnhealthy("native process failed before ReadyEvent")

    def _run_startup_attempt(self, attempt, old_process, old_reader) -> None:
        self._launch_startup_attempt(attempt, old_process, old_reader)
        if not attempt.event.wait(max(0.0, attempt.deadline - time.monotonic())):
            self._expire_startup_attempt(attempt)

    def _expire_startup_attempt(self, attempt: _StartupAttempt) -> None:
        error = EngineUnhealthy("native ReadyEvent timed out")
        with self._state_lock:
            if self._startup_attempt is not attempt or attempt.event.is_set():
                return
            attempt.error = error
            attempt.event.set()
            generation = attempt.generation
        if generation is not None:
            self._fail_generation(generation, error)

    def _complete_startup_attempt(
        self,
        attempt: _StartupAttempt,
        error: EngineRuntimeError,
    ) -> None:
        with self._state_lock:
            if self._startup_attempt is attempt and not attempt.event.is_set():
                attempt.error = error
                attempt.event.set()

    def _launch_startup_attempt(
        self,
        attempt: _StartupAttempt,
        old_process: ProcessLike | None,
        old_reader: threading.Thread | None,
    ) -> None:
        if old_process is not None:
            self._stop_process(old_process, old_reader)
        with self._state_lock:
            # close() or a newer attempt may have won while the old engine was
            # being stopped; spawning now would orphan a fresh engine.
            if self._closed or self._startup_attempt is not attempt:
                return

        process = None
        try:
            process = self._process_factory()
            if process.stdin is None or process.stdout is None:
                raise EngineUnhealthy(
                    "native process did not expose binary stdin/stdout"
                )
            os.set_blocking(process.stdin.fileno(), False)
        except BaseException as error:
            if process is not None:
                self._stop_process(process, None)
            message = (
                "native engine executable is missing; the installation may have been "
                "upgraded or removed. Stop the server and restart Splash from the "
                "current installation"
                if isinstance(error, FileNotFoundError)
                else f"could not launch native engine: {error}"
            )
            failure = EngineUnhealthy(message)
            self._complete_startup_attempt(attempt, failure)
            return
        if time.monotonic() >= attempt.deadline:
            self._expire_startup_attempt(attempt)
        reject = False
        with self._state_lock:
            if (
                self._closed
                or self._startup_attempt is not attempt
                or attempt.event.is_set()
            ):
                reject = True
            else:
                self._generation += 1
                generation = self._generation
                if self._ever_started:
                    self._restart_count += 1
                self._ever_started = True
                self._process = process
                self._terminal_error = None
                self._ready_message = None
                self._last_status = None
                attempt.generation = generation
                self._crash_trace.start_generation(generation, process.pid)
                reader = threading.Thread(
                    target=self._reader_loop,
                    args=(process, generation),
                    name=f"splash-native-reader-{generation}",
                    daemon=True,
                )
                self._reader_thread = reader
                reader.start()
        if reject:
            self._stop_process(process, None)
            self._complete_startup_attempt(
                attempt,
                RuntimeClosed("runtime is closed"),
            )
            return

    def _require_generation_ready_locked(self, generation: int) -> None:
        if self._closed:
            raise RuntimeClosed("runtime is closed")
        if generation != self._generation or self._process is None:
            raise EngineUnhealthy("native process generation is unavailable")
        if self._terminal_error:
            raise self._terminal_error.restate()
        if self._ready_message is None or self._process.poll() is not None:
            raise EngineUnhealthy("native process is not ready")

    def _write_bytes(
        self,
        encoded: bytes,
        generation: int,
        *,
        call: RuntimeCall | None = None,
        deadline: float | None = None,
    ) -> None:
        io_deadline = time.monotonic() + self._io_timeout
        deadline = min(deadline, io_deadline) if deadline is not None else io_deadline
        if not self._write_lock.acquire(timeout=_remaining(deadline)):
            raise TimeoutError("native write lock timed out")
        failure: BaseException | None = None
        finish_failure = None
        offset = 0
        try:
            try:
                with self._state_lock:
                    self._require_generation_ready_locked(generation)
                    if (
                        call is not None
                        and self._pending.get(call.request_id) is not call
                    ):
                        return
                    assert self._process is not None
                    stream = self._process.stdin
                _remaining(deadline)
                # Record under the same lock that defines native wire order.
                # Concurrent callers can arrive in any Python scheduling
                # order, but a replay must reproduce the order actually
                # written to the engine.
                view = memoryview(encoded)
                while offset < len(view):
                    _remaining(deadline)
                    try:
                        written = stream.write(view[offset:])
                    except BlockingIOError:
                        written = None
                    if written is None:
                        # stdin is an unbuffered nonblocking pipe. None means
                        # EAGAIN, never a successful write of the whole frame.
                        with selectors.DefaultSelector() as selector:
                            selector.register(stream, selectors.EVENT_WRITE)
                            selector.select(_remaining(deadline))
                        continue
                    if written <= 0:
                        raise BrokenPipeError("native stdin accepted zero bytes")
                    offset += written
                self._crash_trace.record_bytes(generation, "client_to_engine", encoded)
            except TimeoutError:
                if offset == 0:
                    raise
                # Abandoning part of a frame would corrupt every later write.
                failure = EngineUnhealthy(
                    f"native frame write timed out after {offset} bytes"
                )
            except BaseException as error:
                with self._state_lock:
                    terminal = (
                        self._terminal_error if generation == self._generation else None
                    )
                if terminal is not None:
                    failure = terminal.restate()
                elif isinstance(error, EngineRuntimeError):
                    failure = error
                else:
                    failure = EngineUnhealthy(f"native protocol write failed: {error}")
            if failure:
                # Fence/detach the generation before another writer can enter.
                # Completion callbacks may submit again, so deliver them only
                # after releasing the write lock.
                finish_failure = self._begin_generation_failure(generation, failure)
        finally:
            self._write_lock.release()
        if finish_failure:
            finish_failure()
        if failure:
            raise failure

    def _cancel_call(self, call: RuntimeCall) -> None:
        try:
            encoded = wire.serialize_message(wire.CancelFrame(call.request_id))
            self._write_bytes(encoded, call.generation, call=call)
            with call._lock:
                if not call.done:
                    timer = threading.Timer(
                        self._cancel_grace_seconds, self._cancel_unacknowledged, (call,)
                    )
                    timer.daemon = True
                    call._cancel_timer = timer
                    timer.start()
        except TimeoutError:
            # Unlike a request that has not been sent, cancellation cannot be
            # dropped while leaving the generation healthy and doing work.
            self._fail_generation(
                call.generation, EngineUnhealthy("native cancel write timed out")
            )
        except EngineRuntimeError:
            # Generation failure is already delivered to every in-flight call.
            return

    def _cancel_unacknowledged(self, call):
        with self._state_lock:
            if self._pending.get(call.request_id) is not call or call.done:
                return
            finish = self._begin_generation_failure(
                call.generation,
                EngineUnhealthy("native did not acknowledge cancellation"),
            )
        if finish:
            finish()

    def _reader_loop(self, process: ProcessLike, generation: int) -> None:
        parser = wire.FrameParser()
        try:
            stream = process.stdout
            read = getattr(stream, "read1", stream.read)
            while True:
                chunk = read(_READ_CHUNK_BYTES)
                if not chunk:
                    if issue := parser.finish():
                        raise self._issue_error(issue)
                    raise EngineUnhealthy("native protocol reached EOF")
                offset = 0
                while offset < len(chunk):
                    step = parser.consume(memoryview(chunk)[offset:])
                    offset += step.consumed_bytes
                    if step.issue:
                        raise self._issue_error(step.issue)
                    if not step.consumed_bytes:
                        raise ProtocolFatal("native parser made no progress")
                    if step.frame:
                        self._crash_trace.record_frame(
                            generation, "engine_to_client", step.frame
                        )
                        try:
                            message = wire.decode_frame(step.frame)
                        except wire.ProtocolError as error:
                            raise self._issue_error(error.issue) from error
                        self._dispatch_message(generation, message)
        except BaseException as error:
            if not isinstance(error, EngineRuntimeError):
                error = EngineUnhealthy(f"native reader failed: {error}")
            self._fail_generation(generation, error)

    def _issue_error(self, issue: wire.ProtocolIssue) -> EngineRuntimeError:
        if issue.failure_class is wire.FailureClass.ENGINE_UNHEALTHY:
            return EngineUnhealthy(issue.describe())
        # Request-scoped decode issues cannot be trusted on the engine->client
        # stream unless they arrive as a valid ErrorEvent.
        return ProtocolFatal(issue.describe())

    def _dispatch_message(self, generation: int, message: wire.Message) -> None:
        if isinstance(message, wire.ReadyEvent):
            if int(message.feature_bits) & _REQUIRED_READY_FEATURES != (
                _REQUIRED_READY_FEATURES
            ):
                raise ProtocolFatal(
                    "native ReadyEvent is missing required native protocol features"
                )
            with self._state_lock:
                if generation != self._generation or self._terminal_error is not None:
                    return
                if self._ready_message is not None:
                    raise ProtocolFatal("native sent more than one ReadyEvent")
                assert self._process is not None
                attempt = self._startup_attempt
                if attempt is None or attempt.generation != generation:
                    raise ProtocolFatal(
                        "native ReadyEvent has no matching startup attempt"
                    )
                if attempt.error is not None:
                    raise attempt.error.restate()
                if time.monotonic() >= attempt.deadline:
                    raise EngineUnhealthy("native ReadyEvent timed out")
                if (
                    self._context_limit is not None
                    and message.max_context_tokens != self._context_limit
                ):
                    raise EngineUnhealthy(
                        "native context window changed; restart the Splash server"
                    )
                self._context_limit = message.max_context_tokens
                self._ready_message = message
                attempt.event.set()
            return

        with self._state_lock:
            if generation != self._generation or self._terminal_error is not None:
                return
            if self._ready_message is None:
                raise ProtocolFatal("native event arrived before ReadyEvent")

        if isinstance(message, wire.StatusJsonEvent):
            self._dispatch_status(generation, message)
            return
        if isinstance(message, wire.ErrorEvent):
            self._dispatch_error(generation, message)
            return
        if isinstance(message, wire.CapacityExhaustedEvent):
            call = self._require_call(generation, message.request_id)
            call._emit(message)
            self._finish_call(call, error=CapacityExhausted(message))
            return
        if isinstance(message, wire.StartEvent):
            call = self._require_call(generation, message.request_id)
            if not call._record_start(message):
                raise ProtocolFatal("duplicate or late StartEvent")
            return
        if isinstance(message, wire.PromptProgressEvent):
            call = self._require_call(generation, message.request_id)
            if issue := call._record_progress(message):
                raise ProtocolFatal(issue)
            return
        if isinstance(message, wire.TokensEvent):
            call = self._require_call(generation, message.request_id)
            if issue := call._record_tokens(message):
                raise ProtocolFatal(issue)
            return
        if isinstance(message, wire.MaskRequestEvent):
            call = self._require_call(generation, message.request_id)
            call._emit(message)
            self._submit_mask(call, message)
            return
        if isinstance(message, wire.DoneEvent):
            call = self._require_call(generation, message.request_id)
            result = call._build_result(message)
            call._emit(message)
            if error := call._terminal_mask_error():
                self._finish_call(call, error=error)
            else:
                self._finish_call(call, result=result)
            return
        raise ProtocolFatal(
            f"native sent client-direction frame {type(message).__name__}"
        )

    def _require_call(self, generation: int, request_id: int) -> RuntimeCall:
        with self._state_lock:
            call = self._pending.get(request_id)
            if call is None or call.generation != generation:
                raise ProtocolFatal(
                    f"native event references unknown request {request_id}"
                )
            return call

    def _dispatch_error(self, generation: int, event: wire.ErrorEvent) -> None:
        if event.failure_class is wire.FailureClass.REQUEST_ERROR:
            call = self._require_call(generation, event.request_id)
            call._emit(event)
            self._finish_call(
                call,
                error=RequestFailed(
                    event.request_id,
                    event.code,
                    event.message,
                    retryable=event.retryable,
                ),
            )
            return
        if event.failure_class is wire.FailureClass.ENGINE_UNHEALTHY:
            raise EngineUnhealthy(event.message.decode("utf-8", errors="replace"))
        raise ProtocolFatal(event.message.decode("utf-8", errors="replace"))

    def _dispatch_status(self, generation: int, event: wire.StatusJsonEvent) -> None:
        with self._state_lock:
            self._last_status = event
            waiter = self._status_waiters.pop(event.correlation_id, None)
            if waiter is None:
                if event.correlation_id == 0:
                    return
                if 0 < event.correlation_id <= self._last_status_id:
                    return
                raise ProtocolFatal(
                    f"native status references unknown correlation "
                    f"{event.correlation_id}"
                )
            if waiter.generation != generation:
                return
            waiter.result = event
            waiter.event.set()

    def _submit_mask(self, call: RuntimeCall, event: wire.MaskRequestEvent) -> None:
        if call.cancel_requested or call.done:
            return
        provider = call.request.mask_provider
        if provider is None:
            self._mask_failed(
                call,
                MaskComputationFailed(
                    f"request {call.request_id} has no token-mask provider"
                ),
            )
            return
        if not self._mask_slots.acquire(blocking=False):
            self._mask_failed(call, MaskComputationFailed("token-mask queue is full"))
            return

        def compute():
            if call.cancel_requested or call.done:
                return ()
            return provider(event)

        try:
            future = self._mask_executor.submit(compute)
        except RuntimeError as error:
            self._mask_slots.release()
            self._mask_failed(call, MaskComputationFailed(str(error)))
            return

        def complete(completed) -> None:
            try:
                if call.cancel_requested or call.done:
                    return
                result = completed.result()
                words = result if isinstance(result, bytes) else tuple(result)
                count = len(words) // 4 if isinstance(words, bytes) else len(words)
                expected = event.words_per_mask * event.mask_rows
                if count != expected:
                    raise ValueError(
                        f"mask provider returned {count} words; expected {expected}"
                    )
                response = wire.MaskResponseFrame(
                    call.request_id, event.mask_request_id, words
                )
                encoded = wire.serialize_message(response)
                if call.cancel_requested or call.done:
                    return
                self._write_bytes(encoded, call.generation, call=call)
            except BaseException as error:
                if isinstance(error, EngineRuntimeError):
                    failure = MaskComputationFailed(str(error))
                elif isinstance(error, wire.ProtocolError):
                    failure = MaskComputationFailed(error.issue.describe())
                else:
                    failure = MaskComputationFailed(f"mask computation failed: {error}")
                self._mask_failed(call, failure)
            finally:
                self._mask_slots.release()

        future.add_done_callback(complete)

    def _mask_failed(self, call: RuntimeCall, error: MaskComputationFailed) -> None:
        call._record_mask_error(error)
        self._cancel_call(call)

    def _finish_call(
        self,
        call: RuntimeCall,
        *,
        result: GenerationResult | None = None,
        error: EngineRuntimeError | None = None,
    ) -> None:
        with self._state_lock:
            removed = self._pending.pop(call.request_id, None) is call
        if not removed:
            return
        self._admission_slots.release()
        call._set_terminal(result=result, error=error)

    def _fail_generation(self, generation: int, error: EngineRuntimeError) -> None:
        finish = self._begin_generation_failure(generation, error)
        if finish:
            finish()

    def _begin_generation_failure(
        self, generation: int, error: EngineRuntimeError
    ) -> Callable[[], None] | None:
        with self._state_lock:
            if generation != self._generation or self._terminal_error is not None:
                return
            # The raise that produced this failure is over; only its facts are
            # kept, so its frames do not outlive the requests they ran for.
            failure = error.restate()
            self._terminal_error = failure
            self._ready_message = None
            attempt = self._startup_attempt
            if (
                attempt is not None
                and attempt.generation == generation
                and not attempt.event.is_set()
            ):
                attempt.error = failure
                attempt.event.set()
            process = self._process
            calls = tuple(self._pending.values())
            self._pending.clear()
            waiters = tuple(self._status_waiters.values())
            self._status_waiters.clear()
            for waiter in waiters:
                waiter.error = failure
                waiter.event.set()
            last_status = self._last_status.json if self._last_status else None
            returncode = process.poll() if process is not None else None

        def finish() -> None:
            if not isinstance(failure, RuntimeClosed):
                self._crash_trace.dump(
                    generation,
                    failure,
                    process_returncode=returncode,
                    last_status=last_status,
                )
            for call in calls:
                self._admission_slots.release()
                call._set_terminal(error=failure)
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
                self._arm_kill_fallback(process)

        return finish

    def _arm_kill_fallback(self, process: ProcessLike) -> None:
        """SIGKILL an engine that ignores SIGTERM, without blocking the caller.

        A crashed or hung engine is asked to exit gracefully; if it is still
        alive after the shutdown grace it is killed so it cannot keep GPU
        memory while the next request is waiting for a restart."""

        def kill_if_alive() -> None:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass

        timer = threading.Timer(self._shutdown_grace_seconds, kill_if_alive)
        timer.daemon = True
        timer.start()

    def _stop_process(
        self,
        process: ProcessLike,
        reader: threading.Thread | None,
    ) -> None:
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (OSError, ValueError):
            pass
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        # Bound the child's exit time before escalating to SIGKILL.
        try:
            process.wait(timeout=self._shutdown_grace_seconds)
        except (OSError, subprocess.TimeoutExpired, TimeoutError, KeyboardInterrupt):
            # A second Ctrl+C during the grace means "stop now".
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired, TimeoutError):
                pass
        if reader and reader is not threading.current_thread():
            reader.join(timeout=1.0)

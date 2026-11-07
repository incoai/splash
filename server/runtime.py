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
from dataclasses import dataclass, field, replace
from typing import BinaryIO, Callable, Protocol, Sequence, TypeAlias

from . import protocol as wire
from . import supervision
from .crash_trace import CrashTraceRing


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
        retry_after_seconds: int = 0,
    ):
        self.request_id = request_id
        self.code = code
        self.message_bytes = message
        self.retryable = retryable
        # When a retry may succeed, in seconds; 0 for at once.
        self.retry_after_seconds = retry_after_seconds
        text = message.decode("utf-8", errors="replace")
        machine_code = code.decode("ascii", errors="replace")
        super().__init__(f"request {request_id} failed [{machine_code}]: {text}")

    def restate(self) -> RequestFailed:
        return RequestFailed(
            self.request_id,
            self.code,
            self.message_bytes,
            retryable=self.retryable,
            retry_after_seconds=self.retry_after_seconds,
        )


class EngineUnhealthy(EngineRuntimeError):
    """The engine failed; `code` says how: the code of the engine's
    ErrorEvent, or engine_unhealthy for a failure the server saw."""

    def __init__(self, message: str, *, code: str = "engine_unhealthy"):
        super().__init__(message)
        self.code = code

    def restate(self) -> EngineUnhealthy:
        return EngineUnhealthy(str(self), code=self.code)


class EngineStartFailed(EngineUnhealthy):
    """The engine's StartFailedEvent, in place of its Ready."""

    def __init__(self, event: wire.StartFailedEvent):
        super().__init__(
            event.message.decode("utf-8", errors="replace"), code="start_failed"
        )
        self.event = event

    def restate(self) -> EngineStartFailed:
        return EngineStartFailed(self.event)


class EngineLaunchFailed(EngineUnhealthy):
    """The engine's executable is missing."""

    def __init__(self, message: str):
        super().__init__(message, code="launch_failed")

    def restate(self) -> EngineLaunchFailed:
        return EngineLaunchFailed(str(self))


class ProtocolFatal(EngineRuntimeError):
    code = "protocol_fatal"


class MaskComputationFailed(EngineRuntimeError):
    """A token mask was not delivered.

    ``retryable`` marks a server condition, such as a full mask queue or a
    stalled transport, rather than the request's constraint or provider.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        self.retryable = retryable
        super().__init__(message)

    def restate(self) -> MaskComputationFailed:
        return MaskComputationFailed(*self.args, retryable=self.retryable)


MaskProvider: TypeAlias = Callable[[wire.MaskRequestEvent], bytes]
EventCallback: TypeAlias = Callable[["RuntimeCall", wire.EngineEvent], None]
CompletionCallback: TypeAlias = Callable[["RuntimeCall"], None]
# The failure, the seconds the engine served, the fingerprints of the requests
# it had started (GenerationRequest.fingerprint), and its exit status: a call
# that waits for the engine to exit, at most the stop grace, and returns its
# returncode, None if it still runs. The engine's output may end before its
# exit status is there to read, so the status is read only once it exited.
FailureListener: TypeAlias = Callable[
    [EngineRuntimeError, float, frozenset[bytes], Callable[[], int | None]], None
]

_READ_CHUNK_BYTES = 64 * 1024
_MAX_U64 = (1 << 64) - 1


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("native operation timed out")
    return remaining


def _wire_deadline(remaining: float) -> tuple[int, int]:
    """(absolute unix µs, remaining µs) for a deadline `remaining` seconds
    away; an infinite one saturates at the wire's u64 maximum."""
    wall = time.time_ns() // 1000
    limit = _MAX_U64 - wall
    micros = (
        limit if remaining >= limit / 1_000_000 else max(1, int(remaining * 1_000_000))
    )
    return wall + micros, micros


@dataclass(slots=True, frozen=True)
class GenerationRequest:
    """One native request: its wire frame, without the request id and the
    deadline stamp MultiplexedRuntime.submit fills in, plus what stays in
    Python."""

    # request_id and both deadline fields are 0.
    frame: wire.RequestFrame
    # time.monotonic() deadline; math.inf for none.
    deadline: float
    mask_provider: MaskProvider | None
    # What identifies its content across retries (protocol.request_fingerprint).
    fingerprint: bytes

    def __post_init__(self):
        if (self.frame.constraint is wire.ConstraintMode.TOKEN_MASK) != (
            self.mask_provider is not None
        ):
            raise ValueError("a token-mask request needs exactly one mask provider")


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
        # The counts native events are checked against, not the request:
        # its pixels must not outlive the written frame.
        frame = request.frame
        self._prompt_tokens = len(frame.prompt_tokens)
        self._logical_max = frame.logical_max_output_tokens
        self._score_tokens = len(frame.score_tokens)
        self._return_progress = bool(frame.flags & wire.RequestFlag.RETURN_PROGRESS)
        self.fingerprint = request.fingerprint
        self.mask_provider = request.mask_provider
        self._on_event = on_event
        self._on_complete = on_complete
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._result: wire.DoneEvent | None = None
        self._error: EngineRuntimeError | None = None
        self._start: wire.StartEvent | None = None
        self._progress: wire.PromptProgressEvent | None = None
        self._next_token_offset = 0
        self._mask_error: MaskComputationFailed | None = None
        self._callback_error: BaseException | None = None
        self._end_requested = False
        self._end_timer = None

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @property
    def end_requested(self) -> bool:
        """Whether a cancel or a stop was asked for: the call takes one."""
        with self._lock:
            return self._end_requested

    @property
    def started(self) -> bool:
        """Whether the engine started the request (its StartEvent)."""
        with self._lock:
            return self._start is not None

    @property
    def callback_error(self) -> BaseException | None:
        """The first error an event or completion callback raised."""
        with self._lock:
            return self._callback_error

    def result(self, timeout: float | None = None) -> wire.DoneEvent:
        if not self._event.wait(timeout):
            raise TimeoutError(f"request {self.request_id} did not finish in time")
        with self._lock:
            if self._error:
                raise self._error.restate()
            assert self._result is not None
            return self._result

    def cancel(self) -> bool:
        """Ask native to abandon this request; it finishes as cancelled.

        Returns False, writing nothing, once the call is done or a cancel or
        stop was already requested.
        """
        return self._request_end(wire.CancelFrame(self.request_id))

    def stop(self) -> bool:
        """Ask native to end this request at a client stop condition.

        Native finishes it as stopped, a normal completion, or as cancelled
        before its first token. Shares `cancel`'s latch: returns False, writing
        nothing, once the call is done or a cancel or stop was already
        requested.
        """
        return self._request_end(wire.StopFrame(self.request_id))

    def _request_end(self, frame: wire.CancelFrame | wire.StopFrame) -> bool:
        with self._lock:
            if self._event.is_set() or self._end_requested:
                return False
            self._end_requested = True
        self._client._end_call(self, frame)
        return True

    def _emit(self, message: wire.EngineEvent) -> None:
        with self._lock:
            callback = self._on_event
        if callback is None:
            return
        try:
            callback(self, message)
        except BaseException as error:
            # A request whose consumer failed cannot be delivered; stop it.
            self._record_callback_error(error)
            self.cancel()

    def _record_callback_error(self, error: BaseException) -> None:
        # Keep diagnostics without retaining the callback's request frames.
        error.__traceback__ = None
        error.__context__ = None
        error.__cause__ = None
        with self._lock:
            if self._callback_error is None:
                self._callback_error = error

    def _record_start(self, event: wire.StartEvent) -> bool:
        with self._lock:
            if self._event.is_set() or self._start is not None:
                return False
            self._start = event
        self._emit(event)
        return True

    def _record_progress(self, event: wire.PromptProgressEvent) -> str | None:
        with self._lock:
            if not self._return_progress:
                return "unsolicited prompt progress"
            if self._event.is_set() or self._start is None or self._next_token_offset:
                return "prompt progress outside prefill"
            if (
                not self._start.matched_prompt_tokens
                <= event.processed_tokens
                <= self._prompt_tokens
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
            if next_offset > self._logical_max:
                return (
                    f"TokensEvent stream length {next_offset} exceeds logical "
                    f"maximum {self._logical_max}"
                )
            self._next_token_offset = next_offset
        self._emit(event)
        return None

    def _record_mask_error(self, error: MaskComputationFailed) -> None:
        with self._lock:
            if not self._event.is_set() and self._mask_error is None:
                self._mask_error = error

    def _check_done(self, done: wire.DoneEvent) -> wire.DoneEvent:
        with self._lock:
            prompt_tokens = self._prompt_tokens
            logical_max = self._logical_max
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
                self._end_requested
                and done.reason is wire.FinishReason.CANCELLED
                and not completion_tokens
            ):
                raise ProtocolFatal("DoneEvent arrived before StartEvent")
            # Only a bound of the engine's own ends a request with LENGTH short of
            # its logical maximum.
            if done.reason is wire.FinishReason.LENGTH and (
                completion_tokens >= logical_max
                if done.bounded
                else completion_tokens != logical_max
            ):
                raise ProtocolFatal(
                    f"length-finished DoneEvent has {completion_tokens} tokens; "
                    f"expected {'fewer than ' if done.bounded else ''}logical "
                    f"maximum {logical_max}"
                )
            expected_scores = self._score_tokens
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
            return done

    def _terminal_mask_error(self) -> MaskComputationFailed | None:
        with self._lock:
            return self._mask_error

    def _set_terminal(
        self,
        *,
        result: wire.DoneEvent | None = None,
        error: EngineRuntimeError | None = None,
    ) -> bool:
        with self._lock:
            if self._event.is_set():
                return False
            self._result = result
            self._error = error
            self._on_event = None
            callback, self._on_complete = self._on_complete, None
            self._event.set()
            if self._end_timer is not None:
                self._end_timer.cancel()
                self._end_timer = None
        if callback is not None:
            try:
                callback(self)
            except BaseException as error:
                self._record_callback_error(error)
        return True


@dataclass(slots=True)
class _StatusWaiter:
    event: threading.Event = field(default_factory=threading.Event)
    result: wire.StatusJsonEvent | None = None
    error: EngineRuntimeError | None = None


@dataclass(slots=True)
class _StartupAttempt:
    deadline: float
    event: threading.Event = field(default_factory=threading.Event)
    generation: int | None = None
    error: EngineRuntimeError | None = None
    # The engine before, while the attempt waits for it to exit.
    stopping: ProcessLike | None = None
    # When it spawned its engine (time.monotonic()), once the engine before
    # had exited.
    spawned: float | None = None


class MultiplexedRuntime:
    """One-reader, direct-admission client for the native protocol."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        # The command of every engine started once one was Ready, from that
        # first engine's ReadyEvent: the server's asks for its context. None
        # starts `command` again.
        relaunch_command: Callable[[wire.ReadyEvent], Sequence[str]] | None = None,
        process_factory: Callable[[], ProcessLike] | None = None,
        startup_timeout: float = supervision.START_TIMEOUT_SECONDS,
        pending_limit: int = 64,
        eager_start: bool = False,
        # CPU token-mask workers; None lets the executor choose.
        mask_workers: int | None = None,
    ):
        if process_factory is None and not command:
            raise ValueError("command or process_factory is required")
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        if pending_limit <= 0:
            raise ValueError("pending_limit must be positive")

        self._command = tuple(command) if command else None
        # Read at construction, where a test that patched the module's sees it.
        self._thresholds = supervision.THRESHOLDS
        self._relaunch_command = relaunch_command
        self._process_factory = process_factory
        self._startup_timeout = startup_timeout
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
        self._startup_attempt: _StartupAttempt | None = None
        self._ready_message: wire.ReadyEvent | None = None
        # When the current generation's ReadyEvent arrived.
        self._ready_at: float | None = None
        # The first engine's Ready, whose context every relaunch asks for.
        self._first_ready: wire.ReadyEvent | None = None
        # The latest engine's Ready, kept while a failed one is relaunched.
        self._latest_ready: wire.ReadyEvent | None = None
        # The seconds the latest engine took from its spawn to Ready, which a
        # relaunch is expected to take again from its own.
        self._last_start_seconds: float | None = None
        self._terminal_error: EngineRuntimeError | None = None
        self._pending: dict[int, RuntimeCall] = {}
        self._status_waiters: dict[int, _StatusWaiter] = {}
        self._liveness_timer: threading.Timer | None = None
        self._last_status_id = 0
        self._last_status: wire.StatusJsonEvent | None = None
        self._request_ids = itertools.count(1)
        self._status_ids = itertools.count(1)
        self._crash_trace = CrashTraceRing(
            self._command, enabled=os.environ.get("SPLASH_CRASH_TRACE") == "1"
        )
        # Called once an engine that reached Ready has failed for any reason
        # but close() (FailureListener). It runs, without locks, on the
        # thread that saw the failure, before the engine's calls end, and
        # must not block.
        self.on_engine_failure: FailureListener | None = None
        # Called once the first engine holds its weights and begins its
        # warm-up runs (WarmingEvent), on the reader thread, without locks;
        # it must not block. A relaunched engine's warm-up is a recovery,
        # which status reports.
        self.on_warming: Callable[[], None] | None = None

        if eager_start:
            self._ensure_process()

    def _launch_command(self) -> tuple[str, ...] | None:
        """The command of the engine to start, a relaunch's once an engine
        was Ready; None without a command, which a process factory stands
        in for."""
        with self._state_lock:
            first = self._first_ready
        if self._command is None or first is None or self._relaunch_command is None:
            return self._command
        return tuple(self._relaunch_command(first))

    @staticmethod
    def _spawn(command: Sequence[str]) -> ProcessLike:
        return subprocess.Popen(
            command,
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
            return max(0, self._generation - 1)

    @property
    def capabilities(self) -> wire.ReadyEvent | None:
        """The latest engine's Ready, kept while a failed one is relaunched;
        None until an engine was Ready."""
        with self._state_lock:
            return self._latest_ready

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
    def last_start_seconds(self) -> float | None:
        """The seconds the latest engine took from its spawn to Ready; None
        until an engine was Ready."""
        with self._state_lock:
            return self._last_start_seconds

    @property
    def spawned_at(self) -> float | None:
        """When the engine a startup under way spawned (time.monotonic());
        None once its startup has ended, and while it waits for the engine
        before to exit."""
        with self._state_lock:
            attempt = self._startup_attempt
            return None if attempt.event.is_set() else attempt.spawned

    @property
    def last_crash_trace(self) -> str | None:
        path = self._crash_trace.last_dump
        return str(path) if path is not None else None

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Start an engine unless one serves, and wait for its Ready.

        Raises why the startup ended before Ready. Returns whether the engine
        still serves: on_engine_failure reports a failure after Ready.
        """
        self._ensure_process(timeout=timeout)
        return self.ready

    def submit(
        self,
        request: GenerationRequest,
        *,
        on_event: EventCallback | None = None,
        on_complete: CompletionCallback | None = None,
    ) -> RuntimeCall:
        """Admit and immediately write one request without client scheduling.

        A runtime that is not ready refuses at once; only wait_ready() starts
        an engine.
        """

        deadline = request.deadline
        request_id = next(self._request_ids)
        if not self._admission_slots.acquire(blocking=False):
            raise PendingLimitExceeded(
                f"pending request limit {self._pending_limit} is full"
            )

        call: RuntimeCall | None = None
        try:
            # Admission must precede serialization, which copies image payloads.
            absolute, remaining = _wire_deadline(_remaining(deadline))
            frame = replace(
                request.frame,
                request_id=request_id,
                absolute_deadline_unix_micros=absolute,
                remaining_deadline_micros=remaining,
            )
            try:
                encoded = wire.serialize_message(frame)
            except wire.ProtocolError as error:
                raise self._request_protocol_error(request_id, error.issue) from error
            with self._state_lock:
                if self._closed:
                    raise RuntimeClosed("runtime is closed")
                if not self.ready:
                    raise EngineUnhealthy("native process is not ready")
                generation = self._generation
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
                self._arm_liveness_probe_locked(generation)
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

    def status(
        self, timeout: float = 5.0, *, fail_unanswered: bool = False
    ) -> wire.StatusJsonEvent:
        """The engine's status JSON. With fail_unanswered, a loop that leaves
        the request unanswered for `timeout` fails the engine it ran."""
        return self._status(timeout, None, fail_unanswered)

    def release_weights(self, timeout: float = 5.0) -> wire.StatusJsonEvent:
        """Asks the engine to give its weights back now. It does so only while
        it holds no request and no restore is under way, with
        `--idle-release off` too; the status JSON that answers shows which
        (`weights`: `released` and not `restoring`). The next request restores
        them, as after the idle release."""
        return self._status(timeout, None, False, release=True)

    def _status(
        self,
        timeout: float,
        generation: int | None,
        fail_unanswered: bool,
        release: bool = False,
    ) -> wire.StatusJsonEvent:
        """Asks the engine of ``generation``, or the current one when None, so
        that a probe never measures a newer engine."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("status timeout must be positive")
        deadline = time.monotonic() + timeout
        with self._state_lock:
            if self._closed:
                raise RuntimeClosed("runtime is closed")
            if generation is None:
                generation = self._generation
            self._require_generation_ready_locked(generation)
            correlation_id = next(self._status_ids)
            self._last_status_id = correlation_id
            waiter = _StatusWaiter()
            self._status_waiters[correlation_id] = waiter
        try:
            encoded = wire.serialize_message(
                wire.ReleaseRequestFrame(correlation_id)
                if release
                else wire.StatusRequestFrame(correlation_id)
            )
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
                if fail_unanswered:
                    self._fail_generation(
                        generation,
                        EngineUnhealthy(
                            f"native loop did not answer status within {timeout:g} s"
                        ),
                    )
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

    def kill(self) -> None:
        """SIGKILL the engine now, without waiting for its graceful exit.

        Safe in a signal handler: close() may be waiting for that exit.
        """
        process = self._process
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass

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

    def _ensure_process(self, timeout: float | None = None) -> None:
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
                    return
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
                # Another thread may still be delivering the engine's failure.
                attempt.event.wait()
            with self._state_lock:
                if attempt.error is not None:
                    raise attempt.error.restate()
                if self._closed:
                    raise RuntimeClosed("runtime is closed")
            # The engine reached Ready; on_engine_failure reports any failure
            # since.
            return

    def _run_startup_attempt(self, attempt, old_process, old_reader) -> None:
        self._launch_startup_attempt(attempt, old_process, old_reader)
        if not attempt.event.wait(max(0.0, attempt.deadline - time.monotonic())):
            self._expire_startup_attempt(attempt)

    def _expire_startup_attempt(self, attempt: _StartupAttempt) -> None:
        error = EngineUnhealthy("native ReadyEvent timed out")
        with self._state_lock:
            if self._startup_attempt is not attempt or attempt.event.is_set():
                return
            generation = attempt.generation
            if generation is None:
                stopping = attempt.stopping
                attempt.error = (
                    EngineUnhealthy(
                        f"previous native engine (pid {stopping.pid}) has not "
                        "exited; not launching a second engine beside it"
                    )
                    if stopping is not None
                    else error
                )
                attempt.event.set()
                return
        # The engine's failure ends the attempt.
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
            # A new engine loaded beside the old one holds two models at once,
            # which on a Mac sized for one exhausts memory: it spawns once the
            # old one has exited. One that has not by the attempt's deadline
            # stays the engine the next attempt waits for.
            with self._state_lock:
                attempt.stopping = old_process
            if not self._stop_process(old_process, old_reader, attempt.deadline):
                return
            with self._state_lock:
                attempt.stopping = None
        with self._state_lock:
            # close(), a newer attempt or the attempt's deadline may have won
            # while the old engine was being stopped; spawning now would
            # orphan a fresh engine.
            if (
                self._closed
                or self._startup_attempt is not attempt
                or attempt.event.is_set()
                or time.monotonic() >= attempt.deadline
            ):
                return
            attempt.spawned = time.monotonic()

        process = None
        try:
            command = self._launch_command()
            process = (
                self._spawn(command)
                if self._process_factory is None
                else self._process_factory()
            )
            if process.stdin is None or process.stdout is None:
                raise EngineUnhealthy(
                    "native process did not expose binary stdin/stdout"
                )
            os.set_blocking(process.stdin.fileno(), False)
        except BaseException as error:
            if process is not None:
                self._stop_process(process, None)
            # A missing executable stays missing; any other spawn error, as
            # fork's ENOMEM under pressure, may not.
            failure = (
                EngineLaunchFailed(
                    "native engine executable is missing; the installation may "
                    "have been upgraded or removed. Stop the server and restart "
                    "Splash from the current installation"
                )
                if isinstance(error, FileNotFoundError)
                else EngineUnhealthy(f"could not launch native engine: {error}")
            )
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
                self._process = process
                self._terminal_error = None
                self._ready_message = None
                self._last_status = None
                attempt.generation = generation
                self._crash_trace.start_generation(generation, process.pid, command)
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
        encoded: bytes | bytearray,
        generation: int,
        *,
        call: RuntimeCall | None = None,
        deadline: float | None = None,
    ) -> None:
        # The caller's deadline bounds only the start of a frame. A started
        # frame must be finished: then only the I/O timeout, counted from the
        # last write that made progress, bounds it.
        limit = time.monotonic() + self._thresholds.write_progress
        if deadline is not None:
            limit = min(deadline, limit)
        if not self._write_lock.acquire(timeout=_remaining(limit)):
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
                _remaining(limit)
                # Record under the same lock that defines native wire order.
                # Concurrent callers can arrive in any Python scheduling
                # order, but a replay must reproduce the order actually
                # written to the engine.
                view = memoryview(encoded)
                while offset < len(view):
                    _remaining(limit)
                    try:
                        written = stream.write(view[offset:])
                    except BlockingIOError:
                        written = None
                    if written is None:
                        # stdin is an unbuffered nonblocking pipe. None means
                        # EAGAIN, never a successful write of the whole frame.
                        with selectors.DefaultSelector() as selector:
                            selector.register(stream, selectors.EVENT_WRITE)
                            selector.select(_remaining(limit))
                        continue
                    if written <= 0:
                        raise BrokenPipeError("native stdin accepted zero bytes")
                    offset += written
                    limit = time.monotonic() + self._thresholds.write_progress
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

    def _end_call(
        self, call: RuntimeCall, frame: wire.CancelFrame | wire.StopFrame
    ) -> None:
        try:
            encoded = wire.serialize_message(frame)
            self._write_bytes(encoded, call.generation, call=call)
            with call._lock:
                if not call.done:
                    timer = threading.Timer(
                        self._thresholds.end_acknowledgement,
                        self._end_unacknowledged,
                        (call,),
                    )
                    timer.daemon = True
                    call._end_timer = timer
                    timer.start()
        except TimeoutError:
            # Unlike a request that has not been sent, a cancel or stop cannot
            # be dropped while leaving the generation healthy and doing work.
            self._fail_generation(
                call.generation, EngineUnhealthy("native end write timed out")
            )
        except EngineRuntimeError:
            # Generation failure is already delivered to every in-flight call.
            return

    def _end_unacknowledged(self, call):
        with self._state_lock:
            if self._pending.get(call.request_id) is not call or call.done:
                return
            finish = self._begin_generation_failure(
                call.generation,
                EngineUnhealthy("native did not acknowledge the request's end"),
            )
        if finish:
            finish()

    def _arm_liveness_probe_locked(self, generation: int) -> None:
        if self._liveness_timer is None:
            timer = threading.Timer(
                supervision.LIVENESS_INTERVAL_SECONDS,
                self._probe_liveness,
                (generation,),
            )
            timer.daemon = True
            self._liveness_timer = timer
            timer.start()

    def _liveness_wanted_locked(self, generation: int) -> bool:
        return (
            not self._closed
            and generation == self._generation
            and self._terminal_error is None
            and bool(self._pending)
        )

    def _probe_liveness(self, generation: int) -> None:
        """The engine's reader thread drains stdin even while its loop is
        stuck, so writes keep progressing; only an answer from the loop shows
        it runs."""
        try:
            with self._state_lock:
                if not self._liveness_wanted_locked(generation):
                    return
            try:
                self._status(
                    self._thresholds.status_answer, generation, fail_unanswered=True
                )
            except (EngineRuntimeError, TimeoutError):
                # Left unanswered, the generation has failed. Otherwise it
                # failed meanwhile, or the write lock stayed taken;
                # _write_bytes fences a started write that stalls.
                pass
        finally:
            with self._state_lock:
                # A failure cancels this timer, and the next generation may
                # then arm its own: only the armed timer arms the next one.
                if self._liveness_timer is threading.current_thread():
                    self._liveness_timer = None
                    if self._liveness_wanted_locked(generation):
                        self._arm_liveness_probe_locked(generation)

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
        # The parser and the decoder fail the stream, or the engine when an
        # allocation fails; a request's own failure arrives as an ErrorEvent.
        if issue.failure_class is wire.FailureClass.ENGINE_UNHEALTHY:
            return EngineUnhealthy(issue.describe())
        return ProtocolFatal(issue.describe())

    def _dispatch_message(self, generation: int, message: wire.EngineEvent) -> None:
        if isinstance(message, wire.WarmingEvent):
            with self._state_lock:
                if generation != self._generation or self._terminal_error is not None:
                    return
                if self._ready_message is not None:
                    raise ProtocolFatal("native warming arrived after ReadyEvent")
                listener = self.on_warming if self._first_ready is None else None
            if listener is not None:
                listener()
            return
        if isinstance(message, wire.ReadyEvent):
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
                if self._first_ready is None:
                    self._first_ready = message
                # What the server serves follows the latest engine: a relaunch
                # whose plan holds less context serves that.
                self._latest_ready = message
                self._ready_message = message
                self._ready_at = time.monotonic()
                self._last_start_seconds = self._ready_at - attempt.spawned
                attempt.event.set()
            return

        with self._state_lock:
            if generation != self._generation or self._terminal_error is not None:
                return
            # Before its ReadyEvent an engine sends, besides its warming,
            # nothing but why its start failed (runtime/main.mm refuseStart),
            # and that only then.
            if (self._ready_message is None) != isinstance(
                message, wire.StartFailedEvent
            ):
                raise ProtocolFatal(
                    "native start failure arrived after ReadyEvent"
                    if self._ready_message is not None
                    else "native event arrived before ReadyEvent"
                )

        if isinstance(message, wire.StartFailedEvent):
            raise EngineStartFailed(message)
        if isinstance(message, wire.StatusJsonEvent):
            self._dispatch_status(message)
            return
        if isinstance(message, wire.ErrorEvent):
            self._dispatch_error(generation, message)
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
            if call.mask_provider is None:
                raise ProtocolFatal(
                    "native requested a token mask for an unconstrained request"
                )
            self._submit_mask(call, message)
            return
        if isinstance(message, wire.DoneEvent):
            call = self._require_call(generation, message.request_id)
            result = call._check_done(message)
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
            self._finish_call(
                call,
                error=RequestFailed(
                    event.request_id,
                    event.code,
                    event.message,
                    retryable=event.retryable,
                    retry_after_seconds=event.retry_after_seconds,
                ),
            )
            return
        if event.failure_class is wire.FailureClass.ENGINE_UNHEALTHY:
            raise EngineUnhealthy(
                event.message.decode("utf-8", errors="replace"),
                code=event.code.decode("ascii", errors="replace"),
            )
        raise ProtocolFatal(event.message.decode("utf-8", errors="replace"))

    def _dispatch_status(self, event: wire.StatusJsonEvent) -> None:
        with self._state_lock:
            self._last_status = event
            waiter = self._status_waiters.pop(event.correlation_id, None)
            if waiter is None:
                if 0 < event.correlation_id <= self._last_status_id:
                    return
                raise ProtocolFatal(
                    f"native status references unknown correlation "
                    f"{event.correlation_id}"
                )
            waiter.result = event
            waiter.event.set()

    def _submit_mask(self, call: RuntimeCall, event: wire.MaskRequestEvent) -> None:
        if call.end_requested or call.done:
            return
        provider = call.mask_provider
        if not self._mask_slots.acquire(blocking=False):
            self._mask_failed(
                call, MaskComputationFailed("token-mask queue is full", retryable=True)
            )
            return

        def compute():
            if call.end_requested or call.done:
                return b""
            return provider(event)

        try:
            future = self._mask_executor.submit(compute)
        except RuntimeError as error:
            self._mask_slots.release()
            self._mask_failed(call, MaskComputationFailed(str(error), retryable=True))
            return

        def complete(completed) -> None:
            try:
                if call.end_requested or call.done:
                    return
                result = completed.result()
                expected = 4 * event.words_per_mask * event.mask_rows
                if not isinstance(result, bytes):
                    raise ValueError(
                        f"mask provider returned {type(result).__name__}; "
                        f"expected {expected} bytes"
                    )
                if len(result) != expected:
                    raise ValueError(
                        f"mask provider returned {len(result)} bytes; "
                        f"expected {expected}"
                    )
                response = wire.MaskResponseFrame(
                    call.request_id, event.mask_request_id, result
                )
                encoded = wire.serialize_message(response)
                if call.end_requested or call.done:
                    return
                self._write_bytes(encoded, call.generation, call=call)
            except BaseException as error:
                if isinstance(error, (EngineRuntimeError, TimeoutError)):
                    # The transport, not the request's constraint, failed.
                    failure = MaskComputationFailed(str(error), retryable=True)
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
        call.cancel()

    def _finish_call(
        self,
        call: RuntimeCall,
        *,
        result: wire.DoneEvent | None = None,
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
            served_seconds = (
                None if self._ready_at is None else time.monotonic() - self._ready_at
            )
            self._ready_message = None
            self._ready_at = None
            attempt = self._startup_attempt
            process = self._process
            calls = tuple(self._pending.values())
            self._pending.clear()
            # A queued request was in no command: only the started may have
            # caused the failure (supervision.Quarantine).
            started = frozenset(call.fingerprint for call in calls if call.started)
            if self._liveness_timer is not None:
                self._liveness_timer.cancel()
                self._liveness_timer = None
            waiters = tuple(self._status_waiters.values())
            self._status_waiters.clear()
            for waiter in waiters:
                waiter.error = failure
                waiter.event.set()
            last_status = self._last_status.json if self._last_status else None
            returncode = process.poll() if process is not None else None

        def exit_status() -> int | None:
            # finish() has asked the engine to stop, and kills it past the
            # stop grace: an engine that exited on its own did so by then.
            if process is None:
                return None
            self._wait_exit(process, time.monotonic() + self._thresholds.stop_grace)
            return process.poll()

        def finish() -> None:
            if not isinstance(failure, RuntimeClosed):
                try:
                    self._crash_trace.dump(
                        generation,
                        failure,
                        process_returncode=returncode,
                        last_status=last_status,
                    )
                except Exception:
                    # A diagnostic: one not written must not keep the failure
                    # from reaching the listener, the calls and the engine.
                    pass
                listener = self.on_engine_failure
                if served_seconds is not None and listener:
                    # Before the calls end, so their terminals can follow
                    # what the listener decided about the engine.
                    try:
                        listener(failure, served_seconds, started, exit_status)
                    except Exception:
                        pass
            if attempt is not None and attempt.generation == generation:
                # After the crash trace, so a startup's waiter that learns of
                # this failure finds its trace written.
                self._complete_startup_attempt(attempt, failure)
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

        timer = threading.Timer(self._thresholds.stop_grace, kill_if_alive)
        timer.daemon = True
        timer.start()

    def _stop_process(
        self,
        process: ProcessLike,
        reader: threading.Thread | None,
        deadline: float | None = None,
    ) -> bool:
        """Whether `process` exits by `deadline` (time.monotonic()): asked
        with SIGTERM, killed past the shutdown grace, then waited for, without
        a deadline a shutdown grace more. A killed engine can take long to
        exit while it gives its memory back, and neither the next engine nor
        the server's exit should run beside it."""
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
        grace = time.monotonic() + self._thresholds.stop_grace
        exited = self._wait_exit(
            process, grace if deadline is None else min(grace, deadline)
        )
        if not exited:
            try:
                process.kill()
            except OSError:
                pass
            exited = self._wait_exit(
                process,
                time.monotonic() + self._thresholds.stop_grace
                if deadline is None
                else deadline,
            )
        if reader and reader is not threading.current_thread():
            reader.join(timeout=1.0)
        return exited

    @staticmethod
    def _wait_exit(process: ProcessLike, deadline: float) -> bool:
        """Whether `process` exits by `deadline` (time.monotonic())."""
        while True:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
                return True
            except (subprocess.TimeoutExpired, TimeoutError):
                if time.monotonic() >= deadline:
                    return False
            except OSError:
                # Reaped already: nothing is left to wait for.
                return True

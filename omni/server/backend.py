"""The engine's recovery, status and request ownership, which the audio
backend builds on: relaunches after a failure, the crash-loop stop, the
status snapshots and the refusals while the engine recovers, and the
finalizer thread that delivers each request's terminal off the reader
thread. What is about one request's frames and events is the subclass's."""

import copy
import queue
import threading
import time

from . import json_codec
from . import protocol as wire
from . import runtime as engine_runtime
from .diagnostics import print_status
from .errors import APIError

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


class NativeBackend:
    """Own the engine's recovery and the requests admitted to it.

    A CPU finalizer delivers each request's terminal and logs it off the
    reader thread. Subclasses submit their requests' frames (submit), read
    their events (_on_event), finish them (_finalize) and log them
    (_record).
    """

    _FINISH_NAMES = {
        wire.FinishReason.STOP: "stop",
        wire.FinishReason.LENGTH: "length",
        wire.FinishReason.CANCELLED: "cancelled",
    }

    def __init__(self, runtime, request_logger):
        self.runtime = runtime
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
        # Called once relaunching has stopped for good, after the console
        # says why: the server stops serving and exits, so that what
        # supervises it decides whether to start it again.
        self.on_stopped = None
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
                    + "The server exits; start it again after fixing the cause."
                )
            else:
                self.restart_due = time.monotonic() + delay
            stopped = self.fatal_error
            self._ensure_recovery_locked()
        print_status(f"{event} · {error}", error=True)
        if stopped is not None:
            print_status(f"Engine stopped · {stopped}", error=True)
            if self.on_stopped is not None:
                self.on_stopped()

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
        # The engine's status document carries its own schema; the server
        # reports it as it is, once it says whether the engine is ready.
        snapshot = json_codec.loads(event.json)
        if not isinstance(snapshot, dict) or not isinstance(
            snapshot.get("ready"), bool
        ):
            raise ValueError("native status does not say whether the engine is ready")
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
                snapshot = {"ready": False}
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

    def submit(self, job):
        """Submit `job` to the engine; its events go to `job.events`."""
        raise NotImplementedError

    def _detach_locked(self, state):
        state.detach()
        if self.active.get(state.job.request_id) is state:
            del self.active[state.job.request_id]

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
            # next terminal: its input returns the request budget only once
            # nothing references it.
            del item, state, call

    def _finalize(self, state, call):
        """Deliver the terminal of the request `state` from its `call`."""
        raise NotImplementedError

    @staticmethod
    def _api_error(error):
        if isinstance(error, APIError):
            return APIError(error.status, error.message, error.code)
        if isinstance(error, engine_runtime.RequestFailed):
            code = error.code.decode("ascii", "replace")
            message = error.message_bytes.decode("utf-8", "replace")
            if code == "deadline_exceeded":
                return APIError(504, message, "request_timeout")
            if code == "capacity_exhausted":
                return APIError(
                    400,
                    "the request does not fit in the memory this server may use; "
                    "restart the server with a larger --max-memory or fewer "
                    f"--lanes ({message})",
                    code,
                )
            request_codes = {
                "invalid_count",
                "invalid_deadline",
                "invalid_enum_value",
                "invalid_request",
                "invalid_sampling",
                "limit_exceeded",
            }
            status = 503 if error.retryable else 400 if code in request_codes else 500
            return APIError(status, message, code)
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
        """Log how `job` ended through the request logger."""
        raise NotImplementedError

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

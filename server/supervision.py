"""Every threshold the server fails, stops or relaunches the engine by,
derived from the engine's supervision budgets (protocol.SUPERVISION, which
runtime/Supervision.hpp owns), the restart policy, and the quarantine of
requests that keep failing the engine."""

import math
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from . import protocol as wire

# How long a start may take from its spawn to Ready, model load and warmup
# included, before the server gives up on it.
START_TIMEOUT_SECONDS = 600.0
# While requests are pending, how often the server asks the engine's loop for
# its status: only an answer shows the loop runs.
LIVENESS_INTERVAL_SECONDS = 10.0
# How long /status waits for a live answer before it reports the last one,
# labeled stale.
STATUS_PROBE_SECONDS = 0.05
# The second relaunch in a row waits this long, each after it twice the one
# before, up to Thresholds.backoff_cap.
BACKOFF_BASE_SECONDS = 5.0
# The longest Retry-After a refusal while the engine restarts asks for, the
# longest every OpenAI and Anthropic SDK follows: the JavaScript ones follow
# less than a minute, the Python ones up to a minute.
RETRY_AFTER_LIMIT_SECONDS = 59


@dataclass(frozen=True, slots=True)
class Thresholds:
    # A status request unanswered this long fails the engine: its loop takes
    # input between steps, and a step may last the loop-stall budget.
    status_answer: float
    # How long a frame write may wait for the write lock, or stall once
    # started, for the same reason (requests, cancels and stops alike).
    write_progress: float
    # A request's end, a cancel or a stop, that the engine leaves
    # unacknowledged this long fails it: the command in flight may run the
    # command watchdog out first.
    end_acknowledgement: float
    # From SIGTERM to SIGKILL of an engine, and how long its exit is then
    # waited for.
    stop_grace: float
    # An engine that fails sooner after Ready counts toward the backoff.
    short_lived: float
    backoff_cap: float
    # How much longer than its usual start a relaunch may take while it waits
    # for memory the host or Metal refused.
    start_retry_window: float

    @classmethod
    def of(cls, budgets: wire.Supervision) -> "Thresholds":
        loop = budgets.loop_stall_ms / 1000
        return cls(
            status_answer=loop,
            write_progress=loop,
            end_acknowledgement=budgets.command_timeout_ms / 1000 + loop,
            stop_grace=budgets.stop_ms / 1000,
            short_lived=budgets.probation_ms / 1000,
            backoff_cap=budgets.probation_ms / 1000,
            start_retry_window=budgets.start_retry_window_ms / 1000,
        )


THRESHOLDS = Thresholds.of(wire.SUPERVISION)

# Start failures a relaunch would meet again whatever this Mac's memory or
# disk: a setting the engine refuses, a Mac that cannot run Splash, model
# files it cannot serve, which every relaunch reads again, and a start whose
# commands another process's GPU memory starved, which a retry would take
# from that process.
PERMANENT_START_FAILURES = frozenset(
    {
        wire.StartFailureKind.CONFIGURATION,
        wire.StartFailureKind.DEVICE,
        wire.StartFailureKind.MODEL,
        wire.StartFailureKind.COMMAND_MEMORY,
    }
)

# Start failures for want of memory the host, Metal or the budget refused: a
# relaunch waits for it as long as it is refused, and its failure says
# nothing about the engine.
MEMORY_START_FAILURES = frozenset(
    {
        wire.StartFailureKind.HOST_MEMORY,
        wire.StartFailureKind.DRIVER_ALLOCATION,
        wire.StartFailureKind.MEMORY_BUDGET,
        wire.StartFailureKind.CONTEXT_BUDGET,
    }
)

# How often the relaunch backoff doubles from BACKOFF_BASE_SECONDS before it
# reaches its cap.
BACKOFF_DOUBLINGS = math.ceil(math.log2(THRESHOLDS.backoff_cap / BACKOFF_BASE_SECONDS))
# The short-lived failure in a row that stops the relaunches: the one whose
# backoff would reach the cap (the first relaunches at once, the second
# after BACKOFF_BASE_SECONDS). Relaunching an engine that has failed this
# often within its probation, at the cap, would go on doing so.
CRASH_LOOP_FAILURES = 2 + BACKOFF_DOUBLINGS

# Engine failures a request it ran may have caused: a death by a fault signal,
# and of an engine that exited on its own, an execution failure it reported
# or output that broke the protocol. Not a kill (the user's, jetsam's or the
# server's own, after a fence or past the stop grace), whatever its output
# showed, a command short of GPU memory, or a start, which runs no request.
REQUEST_FAILURE_CODES = frozenset(
    {"engine_execution_failed", "metal_execution_failed", "protocol_fatal"}
)
FAULT_SIGNALS = frozenset(
    {
        signal.SIGSEGV,
        signal.SIGBUS,
        signal.SIGILL,
        signal.SIGABRT,
        signal.SIGFPE,
        signal.SIGTRAP,
    }
)


@dataclass(frozen=True, slots=True)
class EngineFailure:
    # The engine's ErrorEvent code, "start_failed", "launch_failed",
    # "protocol_fatal", or "engine_unhealthy".
    code: str
    message: str
    # 0 for a start that failed.
    served_seconds: float
    # The engine's StartFailedEvent for a start it refused.
    start: wire.StartFailedEvent | None = None
    # The engine's executable is missing.
    launch: bool = False
    # The fingerprints (protocol.request_fingerprint) of the requests the
    # engine had started when it failed.
    started: frozenset[bytes] = frozenset()
    # For an engine that failed after Ready, its exit status once it has
    # exited (runtime.FailureListener), which decides with the code whether
    # a request it ran may have caused the failure.
    exit_status: Callable[[], int | None] | None = None


def request_caused(code: str, returncode: int | None) -> bool:
    """Whether an engine that failed with `code` and exited with `returncode`
    failed in a way a request it ran may have caused. A signal it died by
    decides: a fault signal counts, any other is a kill. Only an engine that
    exited on its own is judged by its code; one that had not exited within
    the stop grace (None) is killed, and does not count."""
    if returncode is None:
        return False
    if returncode < 0:
        return -returncode in FAULT_SIGNALS
    return code in REQUEST_FAILURE_CODES


def permanent(failure: EngineFailure) -> bool:
    """Whether every relaunch would meet `failure` again: the engine's
    executable is missing, or a start failure in PERMANENT_START_FAILURES."""
    return failure.launch or (
        failure.start is not None and failure.start.kind in PERMANENT_START_FAILURES
    )


class RestartPolicy:
    """When to relaunch a failed engine: at once after a failure that followed
    a full probation of service, then after a backoff that doubles with each
    short-lived failure in a row, up to the cap. Never after a permanent
    failure, nor at the CRASH_LOOP_FAILURES-th short-lived failure in a row.
    A start that failed for want of memory (MEMORY_START_FAILURES) backs off
    as any, and is relaunched for as long as it fails, but neither adds to
    the failures in a row nor clears them."""

    def __init__(self):
        self.thresholds = THRESHOLDS
        # Failures in a row since a full probation of service, which the
        # backoff doubles by.
        self.streak = 0
        # Of those, the short-lived ones a start's want of memory did not
        # cause.
        self.short_lived_failures = 0

    def after(self, failure: EngineFailure) -> float | None:
        """Seconds until the relaunch `failure` owes; None for none."""
        if permanent(failure):
            return None
        if failure.served_seconds >= self.thresholds.short_lived:
            self.streak = 0
            self.short_lived_failures = 0
        elif failure.start is None or failure.start.kind not in MEMORY_START_FAILURES:
            self.short_lived_failures += 1
            if self.short_lived_failures == CRASH_LOOP_FAILURES:
                return None
        self.streak += 1
        if self.streak == 1:
            return 0.0
        # Doubled until it reaches the cap and no further: doubled on, it no
        # longer fits a float from the 1026th failure in a row.
        return min(
            self.thresholds.backoff_cap,
            BACKOFF_BASE_SECONDS * 2 ** min(self.streak - 2, BACKOFF_DOUBLINGS),
        )


# How long a quarantined request is refused, in time the Mac is awake. Every
# request the engine had started at both failures is refused, not only the
# one that caused them, so with several lanes a refusal can fall on a request
# that did nothing, whose retries (the same request but for what its
# fingerprint leaves out) would fail until the server restarts. Let back, a
# request that does fail the engine costs at most two failures an hour: one
# while still a suspect, two once a later failure has replaced the suspects.
QUARANTINE_SECONDS = 3600.0


class Quarantine:
    """Refuses a request, for QUARANTINE_SECONDS, whose fingerprint was among
    the started requests at each of the last two engine failures a request
    may have caused. A failure no request could have caused neither adds to
    the suspects nor clears them."""

    def __init__(self):
        self._lock = threading.Lock()
        self._suspects: frozenset[bytes] = frozenset()
        # Each refused fingerprint and when it is let back (time.monotonic).
        # Replaced, never changed in place, so refuses() reads it unlocked.
        self._refused: dict[bytes, float] = {}

    def refuses(self, fingerprint: bytes) -> bool:
        return time.monotonic() < self._refused.get(fingerprint, -math.inf)

    def after(self, started: frozenset[bytes]) -> frozenset[bytes]:
        """Counts a failure a request may have caused, with the fingerprints
        of the requests the engine had started; returns those it now
        refuses."""
        with self._lock:
            now = time.monotonic()
            refused = {
                fingerprint: until
                for fingerprint, until in self._refused.items()
                if now < until
            }
            newly = (self._suspects & started) - refused.keys()
            refused.update(dict.fromkeys(newly, now + QUARANTINE_SECONDS))
            self._refused = refused
            self._suspects = started
            return frozenset(newly)

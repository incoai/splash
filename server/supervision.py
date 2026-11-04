"""Every threshold the server fails, stops or relaunches the engine by,
derived from the engine's supervision budgets (protocol.SUPERVISION, which
runtime/Supervision.hpp owns), the restart policy, and the quarantine of
requests that keep failing the engine."""

import math
import signal
import threading
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
    # started, for the same reason (requests and cancels alike).
    write_progress: float
    # A cancel the engine leaves unacknowledged this long fails it: the
    # command in flight may run the command watchdog out first.
    cancel_acknowledgement: float
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
            cancel_acknowledgement=budgets.command_timeout_ms / 1000 + loop,
            stop_grace=budgets.stop_ms / 1000,
            short_lived=budgets.probation_ms / 1000,
            backoff_cap=budgets.probation_ms / 1000,
            start_retry_window=budgets.start_retry_window_ms / 1000,
        )


THRESHOLDS = Thresholds.of(wire.SUPERVISION)

# Start failures a relaunch would meet again whatever this Mac's memory or
# disk: a setting the engine refuses, a Mac that cannot run Splash, and a
# start whose commands another process's GPU memory starved, which a retry
# would take from that process.
PERMANENT_START_FAILURES = frozenset(
    {
        wire.StartFailureKind.CONFIGURATION,
        wire.StartFailureKind.DEVICE,
        wire.StartFailureKind.COMMAND_MEMORY,
    }
)

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


class RestartPolicy:
    """When to relaunch a failed engine: at once after a failure that followed
    a full probation of service, then after a backoff that doubles with each
    short-lived failure in a row, up to the cap, for as long as failures go
    on. Never once the engine's executable is missing, or after a start
    failure in PERMANENT_START_FAILURES."""

    def __init__(self):
        self.thresholds = THRESHOLDS
        self.streak = 0

    def after(self, failure: EngineFailure) -> float | None:
        """Seconds until the relaunch `failure` owes; None for none."""
        if failure.launch or (
            failure.start is not None and failure.start.kind in PERMANENT_START_FAILURES
        ):
            return None
        if failure.served_seconds >= self.thresholds.short_lived:
            self.streak = 0
        self.streak += 1
        if self.streak == 1:
            return 0.0
        cap = self.thresholds.backoff_cap
        # Doubled until it reaches the cap and no further: doubled on, it no
        # longer fits a float from the 1026th failure in a row.
        doublings = math.ceil(math.log2(cap / BACKOFF_BASE_SECONDS))
        return min(cap, BACKOFF_BASE_SECONDS * 2 ** min(self.streak - 2, doublings))


class Quarantine:
    """Refuses a request, until the server restarts, whose fingerprint was
    among the started requests at each of the last two engine failures a
    request may have caused. A failure no request could have caused neither
    adds to the suspects nor clears them."""

    def __init__(self):
        self._lock = threading.Lock()
        self._suspects: frozenset[bytes] = frozenset()
        # Replaced, never changed in place, so refuses() reads it unlocked.
        self._refused: frozenset[bytes] = frozenset()

    def refuses(self, fingerprint: bytes) -> bool:
        return fingerprint in self._refused

    def after(self, started: frozenset[bytes]) -> frozenset[bytes]:
        """Counts a failure a request may have caused, with the fingerprints
        of the requests the engine had started; returns those it now
        refuses."""
        with self._lock:
            refused = (self._suspects & started) - self._refused
            self._refused |= refused
            self._suspects = started
            return refused

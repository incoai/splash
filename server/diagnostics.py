"""Concise console diagnostics without request-body logging."""

import atexit
import collections
import io
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

# The server package, whose innermost frame, in it or one of its
# subpackages, locates an unexpected error.
_PACKAGE = Path(__file__).resolve().parent

# One thread writes every console line, the server's, its engine's
# (relay_engine_lines) and the launcher's, in the order they come: stdout or
# stderr that stops draining, as a pipe whose reader stalled does, then blocks
# that thread alone, never a request's completion, the engine loop or an HTTP
# answer. It writes both streams, so that they keep one order, which a log of
# both (2>&1, the app's) shows: a stalled stdout holds up stderr too. A line
# that finds 1024 waiting, minutes of a busy server's, is dropped and counted.
CONSOLE_LINES = 1024
# How long a wait for console lines lasts at most: the exit's for the last
# lines, its engine's included, a failed engine's report's for that engine's,
# and the launcher's before an exec, which drops those still waiting. A
# console that drains takes them in milliseconds; one that stalled, or an
# engine slow to exit, must not hold the process up longer.
LAST_LINES_SECONDS = 1.0


class _Console:
    def __init__(self):
        self._waiting = collections.deque()
        self._changed = threading.Condition()
        # Lines taken, written and dropped since the process started.
        self._taken = 0
        self._written = 0
        self._dropped = 0
        self._relays = []
        writer = threading.Thread(
            target=self._write, name="splash-console", daemon=True
        )
        # It starts with the stop signals blocked, and keeps them so: a signal
        # sent to the process goes to a thread that has it unblocked, and the
        # launcher blocks them in its own across a spawn or an exec so that one
        # sent meanwhile stays pending for the program it starts
        # (launcher.STOP_SIGNALS).
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGINT, signal.SIGTERM))
        try:
            writer.start()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)

    def put(self, line, *, error):
        stream = sys.stderr if error else sys.stdout
        with self._changed:
            if len(self._waiting) >= CONSOLE_LINES:
                self._dropped += 1
                return
            self._waiting.append((stream, f"{line}\n"))
            self._taken += 1
            self._changed.notify_all()

    def flush(self, timeout):
        with self._changed:
            taken = self._taken
            return self._changed.wait_for(lambda: self._written >= taken, timeout)

    def stats(self):
        with self._changed:
            return {"dropped_lines": self._dropped}

    def relay(self, stream):
        def forward():
            with io.BufferedReader(stream) as lines:
                for line in lines:
                    text = line.decode("utf-8", "replace").removesuffix("\n")
                    self.put(text, error=True)

        thread = threading.Thread(
            target=forward, name="splash-engine-console", daemon=True
        )
        thread.start()
        # Listed once started: the exit joins each one listed.
        with self._changed:
            self._relays = [relay for relay in self._relays if relay.is_alive()]
            self._relays.append(thread)
        return thread

    def finish(self, timeout):
        # An engine stopped before the exit: its last lines, such as the
        # persistent cache's at a stop, are still on their way.
        deadline = time.monotonic() + timeout
        with self._changed:
            relays = list(self._relays)
        for relay in relays:
            relay.join(max(0.0, deadline - time.monotonic()))
        self.flush(max(0.0, deadline - time.monotonic()))

    def _write(self):
        while True:
            with self._changed:
                self._changed.wait_for(lambda: self._waiting)
                stream, line = self._waiting.popleft()
            # Logging is best effort: a full disk (ENOSPC), a closed stream or
            # any other failure loses the line, never the thread.
            try:
                stream.write(line)
                stream.flush()
            except Exception:
                pass
            with self._changed:
                self._written += 1
                self._changed.notify_all()


_console = _Console()
atexit.register(_console.finish, LAST_LINES_SECONDS)


def flush_console(timeout):
    """Wait up to `timeout` seconds for the console lines printed so far to
    be written; whether they were."""
    return _console.flush(timeout)


def console_stats():
    """The console lines dropped since the server started (/status)."""
    return _console.stats()


def relay_engine_lines(stream):
    """Print each line an engine writes to `stream`, its stderr, as it comes,
    on stderr, until the engine closes it; the thread that does, which ends
    then. A line that finds the console full is dropped: the thread never
    waits on the console."""
    return _console.relay(stream)


def log_unexpected(error):
    """Print the type of an unexpected error and the innermost line of the
    server it passed through, never its message, which may carry request
    data."""
    try:
        location = ""
        for frame, line in traceback.walk_tb(error.__traceback__):
            path = Path(frame.f_code.co_filename).resolve()
            if path.is_relative_to(_PACKAGE):
                location = f" · {path.relative_to(_PACKAGE.parent)}:{line}"
        print_status(
            f"Error · internal_server_error · {type(error).__name__}{location}",
            error=True,
        )
    except Exception:
        pass


def print_line(line, *, error=False):
    """Print `line` as it is on stdout, or on stderr for an error. The
    console thread, the process's one writer, writes it in one write, newline
    included, so that it stays whole beside the lines other processes write
    to the same terminal or log, as the launcher's programs do."""
    _console.put(line, error=error)


def print_status(message, *, error=False):
    # Timed when it is printed, not when it is written.
    print_line(f"{time.strftime('%H:%M:%S')} {message}", error=error)


def _reasoning_parts(reasoning):
    """A chat request's reasoning: the effort it sent, or absent and the
    server's default where that applied, its enable_thinking where it set
    one, and whether it thinks."""
    if reasoning["sent_effort"] is not None:
        parts = [f"effort {reasoning['sent_effort']}"]
    elif reasoning["effort"] is not None:
        parts = [f"effort absent → {reasoning['effort']}"]
    else:
        parts = ["effort absent"]
    if reasoning["enable_thinking"] is not None:
        parts.append(f"enable_thinking {str(reasoning['enable_thinking']).lower()}")
    parts.append("thinking on" if reasoning["thinking"] else "thinking off")
    return parts


def print_request(record):
    outcome = record["outcome"]
    if outcome == "error":
        parts = ["Error", record.get("error_code", "runtime_error")]
        # Only the engine's own wording of a resource failure, never the
        # request's error message, which may quote it.
        if record.get("engine_reason"):
            parts.append(record["engine_reason"])
        print_status(" · ".join(parts), error=True)
        return
    if outcome == "started":
        # The engine admitted the request: a long prompt's prefill runs
        # from here until its Done line.
        parts = [
            "Started",
            f"input {record['prompt_tokens']:,}",
            f"cached {record['cached_tokens']:,}",
        ]
        if "reasoning" in record:
            parts.extend(_reasoning_parts(record["reasoning"]))
        print_status(" · ".join(parts))
        return
    metrics = record.get("metrics", {})
    latency = metrics.get("request_latency", {})
    cache = metrics.get("cache", {})
    parts = [
        "Cancelled" if outcome == "cancelled" else "Done",
        f"input {record['prompt_tokens']:,}",
        f"cached {cache.get('matched_tokens', 0):,}",
    ]
    # Tokens whose KV the request read from cached blocks another request
    # computed, past its cached prefix.
    if cache.get("deduplicated_tokens"):
        parts.append(f"deduplicated {cache['deduplicated_tokens']:,}")
    parts.append(f"output {record.get('completion_tokens', 0):,}")
    tools = record.get("tools")
    if isinstance(tools, dict) and tools.get("count"):
        parts.append(f"tools {tools['count']}·{tools.get('signature', '')}")
    ttft = latency.get("ttft_ms")
    speed = latency.get("stream_tokens_per_second")
    if ttft is not None:
        parts.append(f"TTFT {ttft / 1000:.1f}s")
    if speed is not None:
        parts.append(f"{speed:.1f} tok/s")
    speculation = metrics.get("speculation", {})
    if speculation.get("cycles"):
        # Each verified draft yields the tokens it accepted and one more.
        per_cycle = 1 + speculation["accepted_draft_tokens"] / speculation["cycles"]
        parts.append(f"{per_cycle:.1f} tok/cycle")
    print_status(" · ".join(parts))

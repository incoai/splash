"""Concise console diagnostics without request-body logging."""

import atexit
import collections
import io
import itertools
import os
import select
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
# (relay_engine_lines), the launcher's and Python's tracebacks, in the order
# they come: stdout or stderr that stops draining, as a pipe whose reader
# stalled does, then blocks that thread alone, never a request's completion,
# the engine loop or an HTTP answer. It writes both streams, so that they keep
# one order, which a log of both (2>&1, the app's) shows: a stalled stdout
# holds up stderr too. A line that finds 1024 waiting, minutes of a busy
# server's, is dropped and counted, and once the console writes again a
# warning in the dropped lines' place says how many.
CONSOLE_LINES = 1024
# How long a wait for console lines lasts at most: the exit's for the last
# lines, its engine's included, the launcher's before an exec, which drops
# those still waiting, and how long a failed engine's last lines hold the
# lines printed after its failure (hold_behind_engine). A console that drains
# takes them in milliseconds; one that stalled, or an engine slow to exit,
# must not hold the process or the log up longer.
LAST_LINES_SECONDS = 1.0


class _Console:
    def __init__(self):
        # The lines to write, in order, and those printed while a failed
        # engine's last lines hold them (hold), each as (when it was taken,
        # its stream, its text).
        self._waiting = collections.deque()
        self._held = collections.deque()
        self._changed = threading.Condition()
        # The lines taken and dropped since the process started, those dropped
        # since a warning last said so, and which one the thread writes.
        self._taken = 0
        self._dropped = 0
        self._unsaid = 0
        self._writing = None
        # The engines' relays still reading, and when the lines a failed
        # engine's relay holds go out at the latest.
        self._relays = set()
        self._holds = {}
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

    def put(self, line, *, error, relay=None):
        stream = sys.stderr if error else sys.stdout
        with self._changed:
            if len(self._waiting) + len(self._held) >= CONSOLE_LINES:
                self._dropped += 1
                self._unsaid += 1
                return
            self._append(stream, f"{line}\n", relay)

    def _append(self, stream, line, relay):
        # While a failed engine's relay holds them, lines wait behind its
        # own, which pass.
        self._taken += 1
        entry = (self._taken, stream, line)
        if self._holds and relay not in self._holds:
            self._held.append(entry)
        else:
            self._waiting.append(entry)
        self._changed.notify_all()

    def _written_up_to(self, last):
        # Whether every line taken up to the `last`th is written: the lines of
        # a failed engine pass those its failure holds (hold), so they are not
        # written in the order taken.
        return (self._writing is None or self._writing > last) and all(
            taken > last for taken, _, _ in itertools.chain(self._waiting, self._held)
        )

    def flush(self, timeout):
        with self._changed:
            # And the warning that lines were dropped, which takes the first
            # place to free (_write).
            last = self._taken + (1 if self._unsaid else 0)
            return self._changed.wait_for(lambda: self._written_up_to(last), timeout)

    def stats(self):
        with self._changed:
            return {"dropped_lines": self._dropped}

    def relay(self, stream):
        def forward():
            try:
                with io.BufferedReader(stream) as lines:
                    for line in lines:
                        text = line.decode("utf-8", "replace").removesuffix("\n")
                        self.put(text, error=True, relay=thread)
            finally:
                with self._changed:
                    self._relays.discard(thread)
                    self._release(thread)
                    self._changed.notify_all()

        thread = threading.Thread(
            target=forward, name="splash-engine-console", daemon=True
        )
        # Listed before it starts: it may end at once.
        with self._changed:
            self._relays.add(thread)
        try:
            thread.start()
        except BaseException:
            with self._changed:
                self._relays.discard(thread)
            raise
        return thread

    def hold(self, relay):
        with self._changed:
            if relay in self._relays:
                self._holds[relay] = time.monotonic() + LAST_LINES_SECONDS
                self._changed.notify_all()

    def _release(self, relay):
        if self._holds.pop(relay, None) is not None and not self._holds:
            self._waiting.extend(self._held)
            self._held.clear()

    def finish(self, timeout):
        # An engine stopped before the exit: its last lines, such as the
        # persistent cache's at a stop, are still on their way.
        with self._changed:
            self._changed.wait_for(
                lambda: (
                    not self._relays
                    and not self._unsaid
                    and self._written_up_to(self._taken)
                ),
                timeout,
            )

    def _write(self):
        while True:
            with self._changed:
                while True:
                    now = time.monotonic()
                    for relay, due in list(self._holds.items()):
                        if due <= now:
                            self._release(relay)
                    if self._waiting:
                        break
                    due = min(self._holds.values(), default=None)
                    self._changed.wait(None if due is None else due - now)
                self._writing, stream, line = self._waiting.popleft()
                if self._unsaid:
                    self._append(sys.stderr, _dropped_warning(self._unsaid), None)
                    self._unsaid = 0
            _write_line(stream, line)
            with self._changed:
                self._writing = None
                self._changed.notify_all()


def _dropped_warning(count):
    lines = "line" if count == 1 else "lines"
    return (
        f"{time.strftime('%H:%M:%S')} Warning · {count:,} console {lines} "
        "dropped while the output stopped draining\n"
    )


def _write_line(stream, line):
    # Straight to the stream's file descriptor, in the stream's encoding: a
    # write that blocks on a stalled pipe then holds no lock of Python's
    # buffered stdio, which the interpreter's exit takes to flush it, so the
    # exit's wait (LAST_LINES_SECONDS) bounds the exit with or without -u. A
    # stream without one, such as a test's StringIO, takes the text. Logging
    # is best effort: a full disk (ENOSPC), a closed stream or any other
    # failure loses the line, never the thread.
    try:
        try:
            descriptor = stream.fileno()
        except io.UnsupportedOperation:
            stream.write(line)
            stream.flush()
            return
        data = memoryview(line.encode(stream.encoding, stream.errors))
        while data:
            try:
                data = data[os.write(descriptor, data) :]
            except BlockingIOError:
                # A descriptor another program left nonblocking, as a
                # terminal it shares can be: wait until it takes more.
                select.select((), (descriptor,), ())
    except Exception:
        pass


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


def hold_behind_engine(relay):
    """Write the lines printed from now on, by any thread, after those the
    engine of `relay` (relay_engine_lines) has yet to write, until its stderr
    ends, LAST_LINES_SECONDS from now at most: a failed engine says why as it
    exits, and what it saved as it stops. No caller waits; the console holds
    the lines."""
    _console.hold(relay)


def _print_uncaught(kind, error, trace, heading=""):
    """sys.excepthook: Python's report of an uncaught exception, worded as its
    own hooks word it, printed whole through the console, in order with the
    lines before it, so that a stalled stderr never holds up the exit."""
    lines = traceback.format_exception(kind, error, trace)
    print_line(heading + "".join(lines).rstrip("\n"), error=True)


def _print_uncaught_in_thread(uncaught):
    """threading.excepthook: _print_uncaught, headed by the thread's name;
    nothing for a thread that raised SystemExit, as threading's own."""
    if uncaught.exc_type is SystemExit:
        return
    thread = uncaught.thread
    name = thread.name if thread is not None else threading.get_ident()
    _print_uncaught(
        uncaught.exc_type,
        uncaught.exc_value,
        uncaught.exc_traceback,
        f"Exception in thread {name}:\n",
    )


sys.excepthook = _print_uncaught
threading.excepthook = _print_uncaught_in_thread


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
    included, continued if the stream takes part of it, so that it stays
    whole beside the lines other processes write to the same terminal or log,
    as the launcher's programs do."""
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


def _request(record):
    """The request a line names, where the engine took it: its id, as the
    engine's own lines and `/status` give it, so that the lines that start
    and end it, among those of others running at once, go together."""
    if record.get("request_id") is None:
        return []
    return [f"request {record['request_id']}"]


def print_request(record):
    outcome = record["outcome"]
    if outcome == "error":
        parts = ["Error", *_request(record), record.get("error_code", "runtime_error")]
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
            *_request(record),
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
        *_request(record),
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

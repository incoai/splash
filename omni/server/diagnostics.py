"""Concise console diagnostics without request-body logging."""

import collections
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

# The lines that may wait for the console. A console whose reader stopped
# draining (a stalled log shipper or supervisor pipe) blocks its writes; past
# these, lines are dropped and counted rather than wait.
CONSOLE_LINES = 1024

# The server package, whose innermost frame locates an unexpected error.
_PACKAGE = Path(__file__).resolve().parent


def log_unexpected(error):
    """Print the type of an unexpected error and the innermost line of the
    server it passed through, never its message, which may carry request
    data."""
    try:
        location = ""
        for frame, line in traceback.walk_tb(error.__traceback__):
            path = Path(frame.f_code.co_filename).resolve()
            if path.parent == _PACKAGE:
                location = f" · {_PACKAGE.name}/{path.name}:{line}"
        print_status(
            f"Error · internal_server_error · {type(error).__name__}{location}",
            error=True,
        )
    except Exception:
        pass


class Console:
    """stdout and stderr, written by one thread of their own, so that no
    request, finalizer or engine listener waits on a console that has stopped
    draining. Past CONSOLE_LINES waiting lines, a line is dropped and counted;
    the next line written says how many were."""

    def __init__(self):
        self._waiting = collections.deque()
        self._changed = threading.Condition()
        # Lines taken and written, to know when a flush is done; lines
        # dropped since the start, and since the console last said so.
        self._taken = 0
        self._written = 0
        self._dropped = 0
        self._unsaid = 0
        writer = threading.Thread(
            target=self._write, name="splash-console", daemon=True
        )
        # The writer keeps the stop signals blocked, so that one sent to the
        # process reaches the thread that handles it.
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGINT, signal.SIGTERM))
        try:
            writer.start()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)

    def put(self, line, *, error):
        with self._changed:
            if len(self._waiting) >= CONSOLE_LINES:
                self._dropped += 1
                self._unsaid += 1
                return
            self._taken += 1
            # The stream as it is now: lines go where they were printed.
            self._waiting.append((sys.stderr if error else sys.stdout, line))
            self._changed.notify_all()

    def _write(self):
        while True:
            with self._changed:
                self._changed.wait_for(lambda: self._waiting)
                stream, line = self._waiting.popleft()
                unsaid, self._unsaid = self._unsaid, 0
            if unsaid:
                self._emit(
                    sys.stderr,
                    f"{time.strftime('%H:%M:%S')} Warning · {unsaid} console "
                    "lines dropped while the console was not reading",
                )
            self._emit(stream, line)
            with self._changed:
                self._written += 1
                self._changed.notify_all()

    @staticmethod
    def _emit(stream, line):
        # One write per line, newline included, so that lines stay whole
        # beside the native runtime's on the shared stderr.
        try:
            stream.write(f"{line}\n")
            stream.flush()
        except (OSError, ValueError):
            # A closed or broken console loses the line, as print would.
            pass

    def flush(self, timeout):
        """Wait up to `timeout` seconds for the lines taken so far; True once
        they are written."""
        with self._changed:
            last = self._taken
            return self._changed.wait_for(lambda: self._written >= last, timeout)

    def stats(self):
        with self._changed:
            return {"dropped_lines": self._dropped}


CONSOLE = Console()


def print_status(message, *, error=False):
    """Queue one console line; it never waits for the console."""
    CONSOLE.put(f"{time.strftime('%H:%M:%S')} {message}", error=error)

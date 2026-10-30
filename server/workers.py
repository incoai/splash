"""Run a decoder over untrusted input in a disposable process, within
supervised memory, time and output budgets: PDFium for PDF pages
(document_worker.py) and AVFoundation for video frames (videos.py)."""

import atexit
import ctypes
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import APIError

POLL_SECONDS = 0.01
# The most of a worker's stderr kept for its error.
MAX_ERROR_BYTES = 4096
_workers = set()
_workers_lock = threading.Lock()


class _RusageInfo(ctypes.Structure):
    # Darwin's public rusage_info_v2 layout, queried only for our own child.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
            "child_user_time",
            "child_system_time",
            "child_pkg_idle_wkups",
            "child_interrupt_wkups",
            "child_pageins",
            "child_elapsed_abstime",
            "diskio_bytesread",
            "diskio_byteswritten",
        )
    ]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_libproc.proc_pid_rusage.restype = ctypes.c_int


def _memory_bytes(pid):
    usage = _RusageInfo()
    if _libproc.proc_pid_rusage(pid, 2, ctypes.byref(usage)):
        raise OSError(ctypes.get_errno(), "cannot inspect worker")
    return max(usage.resident_size, usage.phys_footprint)


def _stop(process):
    if process.poll() is None:
        process.kill()
    process.wait()


@atexit.register
def _close_workers():
    with _workers_lock:
        workers = tuple(_workers)
    for process in workers:
        _stop(process)


@dataclass(frozen=True, slots=True)
class Work:
    """What a worker does, as its errors name it."""

    # What exceeds a limit: "PDF processing".
    subject: str
    # The process: "document worker".
    worker: str
    # The error code of a worker that cannot start or be measured.
    unavailable: str
    # The error for output over its limit.
    oversized: str


@dataclass(frozen=True, slots=True)
class Finished:
    returncode: int
    output: bytes
    # The start of what the worker wrote to stderr.
    error: str


def run(
    command,
    stdin,
    work,
    *,
    remaining,
    max_seconds,
    max_memory_bytes,
    max_output_bytes,
    cwd=None,
):
    """Run command with stdin's bytes as its input, within the `remaining`
    seconds of its request, infinite when it has none, and max_seconds, its
    resident or physical footprint within max_memory_bytes: its exit status,
    at most max_output_bytes of stdout, and its error. A worker over a limit is
    killed; nothing it wrote is returned."""
    duration = min(max_seconds, remaining)
    deadline = time.monotonic() + duration
    # TemporaryFile is private and unlinked. File-backed transport also lets
    # the child enforce its output limit without blocking on a pipe reader.
    with (
        tempfile.TemporaryFile() as source,
        tempfile.TemporaryFile() as output,
        tempfile.TemporaryFile() as error,
    ):
        source.write(stdin)
        source.seek(0)
        try:
            process = subprocess.Popen(
                command, stdin=source, stdout=output, stderr=error, cwd=cwd
            )
        except OSError:
            raise APIError(
                503, f"{work.worker} could not start", work.unavailable
            ) from None
        with _workers_lock:
            _workers.add(process)
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    if remaining <= max_seconds:
                        raise APIError(504, "request timed out", "request_timeout")
                    raise APIError(400, f"{work.subject} exceeded the time limit")
                try:
                    memory = _memory_bytes(process.pid)
                except OSError:
                    if process.poll() is not None:
                        break
                    raise APIError(
                        503,
                        f"{work.worker} memory measurement failed",
                        work.unavailable,
                    ) from None
                if memory > max_memory_bytes:
                    raise APIError(400, f"{work.subject} exceeded the memory limit")
                try:
                    process.wait(timeout=POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
            output.seek(0)
            written = output.read(max_output_bytes + 1)
            if len(written) > max_output_bytes:
                raise APIError(400, work.oversized)
            error.seek(0)
            message = error.read(MAX_ERROR_BYTES).decode("utf-8", "replace").strip()
            return Finished(process.returncode, written, message)
        finally:
            _stop(process)
            with _workers_lock:
                _workers.discard(process)


# The directory that holds the server package, from which a Python worker
# imports it however the server was started.
SERVER_ROOT = Path(__file__).resolve().parents[1]

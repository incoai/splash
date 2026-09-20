"""Run PDFium in a disposable process with supervised memory and time budgets."""

import atexit
import ctypes
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

if __package__:
    from .errors import APIError
else:
    from errors import APIError

MAX_SECONDS = 30.0
MAX_MEMORY_BYTES = 1024 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
POLL_SECONDS = 0.01
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


if sys.platform == "darwin":
    _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _libproc.proc_pid_rusage.restype = ctypes.c_int


def _memory_bytes(pid):
    if sys.platform == "darwin":
        usage = _RusageInfo()
        if _libproc.proc_pid_rusage(pid, 2, ctypes.byref(usage)):
            raise OSError(ctypes.get_errno(), "cannot inspect document worker")
        return max(usage.resident_size, usage.phys_footprint)
    pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


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


def render(payload, limits, remaining):
    duration = MAX_SECONDS if remaining < 0 else min(MAX_SECONDS, remaining)
    deadline = time.monotonic() + duration
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        json.dumps(limits),
        str(duration),
    ]
    # TemporaryFile is private and unlinked. File-backed transport also lets
    # the child enforce its output limit without blocking on a pipe reader.
    with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as output:
        source.write(payload)
        source.seek(0)
        try:
            process = subprocess.Popen(
                command, stdin=source, stdout=output, stderr=subprocess.DEVNULL
            )
        except OSError:
            raise APIError(
                503, "document worker could not start", "document_unavailable"
            ) from None
        with _workers_lock:
            _workers.add(process)
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    if 0 <= remaining <= MAX_SECONDS:
                        raise APIError(504, "request timed out", "request_timeout")
                    raise APIError(400, "PDF processing exceeded the time limit")
                try:
                    memory = _memory_bytes(process.pid)
                except OSError:
                    if process.poll() is not None:
                        break
                    raise APIError(
                        503,
                        "document worker memory measurement failed",
                        "document_unavailable",
                    ) from None
                if memory > MAX_MEMORY_BYTES:
                    raise APIError(400, "PDF processing exceeded the memory limit")
                try:
                    process.wait(timeout=POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode != 0:
                raise APIError(
                    400, "PDF processing failed or exceeded its resource limit"
                )
            output.seek(0)
            encoded = output.read(MAX_OUTPUT_BYTES + 1)
            if len(encoded) > MAX_OUTPUT_BYTES:
                raise APIError(400, "rendered PDF exceeds the size limit")
            try:
                result = json.loads(encoded)
            except (ValueError, UnicodeError):
                raise APIError(
                    500, "document worker returned an invalid result"
                ) from None
            if "error" in result:
                error = result["error"]
                raise APIError(error["status"], error["message"], error["code"])
            return result["pages"]
        finally:
            _stop(process)
            with _workers_lock:
                _workers.discard(process)


def main():
    from documents import MAX_PDF_BYTES, DocumentBudget, RenderLimits, render_pages

    limits = RenderLimits(**json.loads(sys.argv[1]))
    duration = float(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    cpu_seconds = max(1, math.ceil(duration))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, MAX_MEMORY_BYTES))
    # On macOS RLIMIT_AS aliases the advisory RSS limit. The parent instead
    # measures resident/physical footprint and terminates an over-budget child.
    budget = DocumentBudget(
        deadline=time.monotonic() + duration, remaining_bytes=limits.request_bytes
    )
    try:
        payload = sys.stdin.buffer.read(MAX_PDF_BYTES + 1)
        if len(payload) > MAX_PDF_BYTES:
            raise APIError(400, "PDF document exceeds the size limit")
        pages = render_pages(payload, budget, limits)
        result = {"pages": [asdict(page) for page in pages]}
    except APIError as error:
        result = {
            "error": {
                "status": error.status,
                "message": error.message,
                "code": error.code,
            }
        }
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()

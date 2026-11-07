"""Run PDFium in a disposable, sandboxed process with supervised memory and
time budgets."""

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

from .errors import APIError

MAX_SECONDS = 30.0
MAX_MEMORY_BYTES = 1024 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
POLL_SECONDS = 0.01
# The folder that holds the server package: the worker runs there, so that -m
# imports the package however the server was started.
ROOT = Path(__file__).resolve().parents[1]
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
# sandbox-exec's status (EX_OSERR) when it cannot apply the profile or start
# the interpreter, as inside another sandbox; the worker never exits with it.
SANDBOX_FAILED = 71
# PDFium parses whatever PDF a request sends, which a model may have fetched
# from the web, so the worker runs under a profile that refuses all that
# rendering does without: it reads the system's libraries and fonts (PDFium
# names ~/Library/Fonts too, but never expands it), the interpreter and the
# server package, and nothing of the user's; it writes only to the descriptors
# it is given, and reaches no network, no service and no other program. Of the
# sysctls it reads those os.uname() needs. The rest it tries (the time zone,
# the locale's data, the user database, the log) is refused and changes
# nothing it writes: the pages are byte-identical outside the sandbox. It
# learns nothing of other processes: kern.procargs2 tells any process of the
# user another's arguments and the environment it started with, the server's
# included, and the kernel checks it as process-info-pidinfo, which (deny
# default) leaves open and (deny process-info*) refuses. Of itself it may
# learn all: PDFium asks as it loads a page, and stops the worker (SIGTRAP)
# when refused. _sandbox() adds the interpreter, which it starts, its
# prefixes, which it reads, and the symbolic links on the way to them.
PROFILE = """(version 1)
(deny default)
(allow sysctl-read
  (sysctl-name "hw.machine")
  (sysctl-name "kern.hostname")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.version"))
(allow file-read*
  (literal "/") ; the interpreter aborts as it starts without it
  (subpath "/System")
  (subpath "/usr/lib")
  (subpath "/Library/Fonts")
  (literal (param "ROOT"))
  (subpath (param "SERVER")))
(deny process-info*)
(allow process-info* (target self))
"""
# The most symbolic links the kernel follows in resolving a path (MAXSYMLINKS).
MAX_LINKS = 32
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
        raise OSError(ctypes.get_errno(), "cannot inspect document worker")
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


def _links(path):
    """The symbolic links that resolving the absolute `path` follows."""
    links = []
    resolved = Path("/")
    pending = list(Path(path).parts[1:])
    while pending and len(links) < MAX_LINKS:
        part = pending.pop(0)
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        if not candidate.is_symlink():
            resolved = candidate
            continue
        links.append(candidate)
        target = Path(os.readlink(candidate))
        if target.is_absolute():
            resolved = Path("/")
            pending[:0] = target.parts[1:]
        else:
            pending[:0] = target.parts
    return links


def _sandbox():
    """sandbox-exec's command that runs the interpreter under PROFILE. Each
    worker builds its own: an upgrade under a running server moves the real
    paths it names.

    The kernel checks a path the worker opens by its real path, and each
    symbolic link it follows to get there as a read of the link: a venv's
    interpreter is one, and a folder on the way to it or to a prefix may be
    one. So the profile lets it read its prefixes by their real paths and
    the links on the way to them, and start the interpreter's real file.
    A framework build's interpreter, Homebrew's and python.org's, is a
    launcher: it resolves the folder it was started from, which takes the
    metadata of each folder on the way, and starts the interpreter in the
    framework's Python.app. Paths go in as parameters, which need no
    quoting; -- ends the options."""
    prefixes = (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
    links = [link for path in (sys.executable, *prefixes) for link in _links(path)]
    app = Path(sys.base_prefix, "Resources/Python.app/Contents/MacOS/Python")
    # By their real paths, which the kernel checks: Homebrew's prefix is a link.
    interpreters = [
        os.path.realpath(path) for path in (sys.executable, app) if Path(path).is_file()
    ]
    # The kernel checks each folder by its real path, the links on the way
    # apart (links): a folder reached through a link, as /var is, is checked
    # where the link leads.
    folders = {
        str(folder)
        for path in (sys.executable, os.path.realpath(sys.executable))
        for parent in Path(path).parents
        for real in [Path(os.path.realpath(parent))]
        for folder in (real, *real.parents)
    }
    parameters = {
        "ROOT": str(ROOT),
        "SERVER": str(ROOT / "server"),
    }

    def allow(operation, kind, name, paths):
        """The rule that allows `operation` on each of `paths`, a filter of
        `kind` each, whose paths go in as the parameters NAME0, NAME1, ...;
        none for no paths, as an allow without a filter allows the
        operation everywhere."""
        filters = []
        for index, path in enumerate(dict.fromkeys(paths)):
            parameters[f"{name}{index}"] = path
            filters.append(f'\n  ({kind} (param "{name}{index}"))')
        return f"(allow {operation}{''.join(filters)})\n" if filters else ""

    profile = (
        PROFILE
        + allow("process-exec", "literal", "PYTHON", interpreters)
        + allow("file-read*", "subpath", "PREFIX", map(os.path.realpath, prefixes))
        + allow("file-read-metadata", "literal", "LINK", map(str, links))
        + allow("file-read-metadata", "literal", "FOLDER", sorted(folders))
    )
    return [
        SANDBOX_EXEC,
        *(f"-D{name}={value}" for name, value in parameters.items()),
        "-p",
        profile,
        "--",
    ]


def _start(arguments, source, output):
    """The interpreter, started with `arguments` as the worker starts: under
    the sandbox, in ROOT, and with no environment, so that no secret of the
    server's, such as SPLASH_API_KEY, reaches a process parsing untrusted
    PDFs; with -B it writes no bytecode, which the profile would refuse."""
    return subprocess.Popen(
        [*_sandbox(), sys.executable, "-B", *arguments],
        stdin=source,
        stdout=output,
        stderr=subprocess.DEVNULL,
        cwd=ROOT,
        env={},
    )


def render(payload, limits, remaining):
    """Render a PDF in a worker within the `remaining` seconds of its request,
    infinite when it has no deadline, and the worker's own time limit."""
    duration = min(MAX_SECONDS, remaining)
    deadline = time.monotonic() + duration
    # TemporaryFile is private and unlinked. File-backed transport also lets
    # the child enforce its output limit without blocking on a pipe reader.
    with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as output:
        source.write(payload)
        source.seek(0)
        try:
            process = _start(
                ["-m", "server.document_worker", json.dumps(limits), str(duration)],
                source,
                output,
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
                    if remaining <= MAX_SECONDS:
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
            if process.returncode == SANDBOX_FAILED:
                raise APIError(
                    503, "document worker could not start", "document_unavailable"
                )
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
                    500,
                    "document worker returned an invalid result",
                    "document_worker_failed",
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
    from .documents import MAX_PDF_BYTES, DocumentBudget, RenderLimits, render_pages

    limits = RenderLimits(**json.loads(sys.argv[1]))
    duration = float(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    cpu_seconds = max(1, math.ceil(duration))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    # On macOS RLIMIT_AS aliases the advisory RSS limit. The parent instead
    # measures resident/physical footprint and terminates an over-budget child.
    budget = DocumentBudget(
        deadline=time.monotonic() + duration,
        remaining_bytes=limits.request_bytes,
        remaining_pages=limits.request_pages,
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

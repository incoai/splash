"""The serve record, `<runtime>/serve-PORT.json` beside the launcher's lock,
from which the desktop app follows a server it started through its start and
words why one did not start. The launcher writes it, atomically, once it holds
the lock; the server it executes, which keeps its pid, as each later phase of
the start begins; either, when the start fails: a kind the app words and a
message for the rest.

The app learns a start's outcome from the record and the process's exit
alone: it waits until the record says ready and takes an exit before that for
the start's failure. So the starting and ready records that cannot be written
fail the start (NotRecorded): the launcher ends before any model loads,
rather than leave a record of an earlier process, a stale ready one of the
same pid among them, to say how this start goes; the server ends before it
serves instead of serving while the app waits on it. Every other record is
best effort, one that cannot be written a line in the log: a refusal or a
failure stays the one the process prints and exits with, and a later phase
before ready, which only words the wait, leaves the record as it was. The
standard library and the protocol and diagnostics modules alone: the launcher
runs before .venv exists."""

import contextlib
import json
import os
from pathlib import Path

from .diagnostics import print_status
from .protocol import StartFailureKind


def _name(kind: StartFailureKind) -> str:
    return kind.name.lower()


# The kinds of a failed start: the engine's StartFailureKind names, which the
# launcher and the server share where their failure is of the same class, and
# the frontend's own.
CONFIGURATION = _name(StartFailureKind.CONFIGURATION)
MODEL = _name(StartFailureKind.MODEL)
DEVICE = _name(StartFailureKind.DEVICE)
INTERNAL = _name(StartFailureKind.INTERNAL)
# The server cannot listen at its address.
PORT = "port"
# An upgrade of Splash holds the installation.
UPGRADE = "upgrade"
# The model's download or verification failed.
INSTALL = "install"


class NotRecorded(Exception):
    """A record that cannot be written: its path and the system's error."""


def starting(path: Path | None):
    """Record that this process's start began, which no stale record of its
    pid can be; NotRecorded where it cannot be, and the start has failed."""
    _write(path, {"pid": os.getpid(), "phase": "starting"})


def entered(path: Path | None, phase: str):
    """Record, best effort, that this process's start entered a later
    `phase`: "loading", the tokenizer, then the engine's weights; or
    "warming", the engine runs the model once at full size (WarmingEvent)."""
    _write_best_effort(path, {"pid": os.getpid(), "phase": phase})


def ready(path: Path | None):
    """Record that this process's start is ready, answering; NotRecorded
    where it cannot be, and the start has failed."""
    _write(path, {"pid": os.getpid(), "phase": "ready"})


def failed(path: Path | None, failure: dict):
    """Record, best effort, that this process's start failed (failure,
    engine_failure)."""
    _write_best_effort(
        path, {"pid": os.getpid(), "phase": "failed", "failure": failure}
    )


def _write(path, record):
    """Replace the record at `path` (a sibling written, then renamed over it);
    nothing without a path. A record that cannot be written raises
    NotRecorded, its sibling removed where it can be."""
    if path is None:
        return
    sibling = path.with_name(f".{path.name}.{record['pid']}")
    try:
        sibling.write_text(json.dumps(record, separators=(",", ":")))
        os.replace(sibling, path)
    except OSError as error:
        with contextlib.suppress(OSError):
            sibling.unlink(missing_ok=True)
        raise NotRecorded(
            f"cannot write the serve record {path}: {error.strerror or error}"
        ) from None


def _write_best_effort(path, record):
    """_write, a record that cannot be written a warning line of its own."""
    try:
        _write(path, record)
    except NotRecorded as error:
        print_status(f"Warning · {error}", error=True)


def failure(kind: str, message: str) -> dict:
    """A failure of `kind`, one of this module's, with its message."""
    return {"kind": kind, "message": message}


def engine_failure(event, message: str) -> dict:
    """The engine's StartFailedEvent, with its message: its kind, and the
    bound and numbers of a shortfall by name."""
    record = {"kind": _name(event.kind)}
    if event.required_bytes or event.budget_bytes or event.plan_context_tokens:
        record |= {
            "bound": event.bound.name.lower(),
            "required_bytes": event.required_bytes,
            "budget_bytes": event.budget_bytes,
            "plan_context_tokens": event.plan_context_tokens,
        }
    return record | {"message": message}

"""The serve record, `<runtime>/serve-PORT.json` beside the launcher's lock,
from which the desktop app follows a server it started through its start and
words why one did not start. The launcher writes it, atomically, once it holds
the lock; the server it executes, which keeps its pid, as each later phase of
the start begins; either, when the start fails: a kind the app words and a
message for the rest. The standard library and the protocol module alone: the
launcher runs before .venv exists."""

import json
import os
from pathlib import Path

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


def entered(path: Path | None, phase: str):
    """Record that this process's start entered `phase`: "starting", which no
    stale record of its pid can be; "loading", the tokenizer, then the
    engine's weights; "warming", the engine runs the model once at full size
    (WarmingEvent); or "ready", answering."""
    _write(path, {"pid": os.getpid(), "phase": phase})


def failed(path: Path | None, failure: dict):
    """Record that this process's start failed (failure, engine_failure)."""
    _write(path, {"pid": os.getpid(), "phase": "failed", "failure": failure})


def _write(path, record):
    """Replace the record at `path` (a sibling written, then renamed over it);
    nothing without a path."""
    if path is None:
        return
    sibling = path.with_name(f".{path.name}.{record['pid']}")
    sibling.write_text(json.dumps(record, separators=(",", ":")))
    os.replace(sibling, path)


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

"""Bounded native-protocol crash traces and deterministic local replay.

The ring keeps the newest MAX_TRACE_ENTRIES frames within MAX_TRACE_BYTES of a
process generation, so the trace of a long-lived engine starts wherever the
ring did: earlier frames, and whatever engine state they built up, are gone.
A frame larger than MAX_TRACE_BYTES leaves only a marker with its header, size
and digest; a trace that lost engine input that way cannot be replayed.
A replay feeds the recorded client-to-engine frames back at their recorded
offsets, re-stamping each request's absolute deadline from replay time, since
the recorded wall-clock deadline has usually elapsed by then.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import selectors
import subprocess
import tempfile
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from . import protocol as wire

TRACE_SCHEMA_VERSION = 1
MAX_TRACE_ENTRIES = 512
MAX_TRACE_BYTES = 16 * 1024 * 1024
MAX_TRACE_FILES = 4
MAX_TRACE_DISK_BYTES = 64 * 1024 * 1024
REPLAY_TIMEOUT_SECONDS = 600
SHUTDOWN_GRACE_SECONDS = 15.0
DEFAULT_TRACE_DIRECTORY = Path.home() / "Library" / "Logs" / "Splash" / "crash"


@dataclass(frozen=True, slots=True)
class _TraceEntry:
    monotonic_ns: int
    direction: str
    frame: bytes
    # A frame over MAX_TRACE_BYTES keeps only its header, size and digest.
    omitted_bytes: int = 0
    sha256: str = ""

    def as_dict(self, started_ns: int) -> dict:
        item = {
            "offset_micros": max(0, (self.monotonic_ns - started_ns) // 1000),
            "direction": self.direction,
        }
        encoded = base64.b64encode(self.frame).decode("ascii")
        if self.omitted_bytes:
            item.update(
                header_base64=encoded,
                omitted_bytes=self.omitted_bytes,
                sha256=self.sha256,
            )
        else:
            item["frame_base64"] = encoded
        return item


class CrashTraceRing:
    """A process-generation-local, byte-bounded ring of protocol frames."""

    def __init__(self, command: Sequence[str] | None, *, enabled: bool = False):
        self._command = tuple(command or ())
        self._enabled = enabled
        self._lock = threading.Lock()
        self._generation = 0
        self._pid = 0
        self._started_ns = 0
        self._entries: deque[_TraceEntry] = deque()
        self._bytes = 0
        self._dumped_generation = 0
        self._last_dump: Path | None = None

    @property
    def active(self) -> bool:
        # Production subprocesses always have an exact replay command. Test
        # process factories intentionally do not write machine-global logs.
        return self._enabled and bool(self._command)

    @property
    def last_dump(self) -> Path | None:
        with self._lock:
            return self._last_dump

    def start_generation(self, generation: int, pid: int) -> None:
        if not self.active:
            return
        with self._lock:
            self._generation = generation
            self._pid = pid
            self._started_ns = time.monotonic_ns()
            self._entries.clear()
            self._bytes = 0

    def record_bytes(
        self, generation: int, direction: str, frame: bytes | bytearray
    ) -> None:
        if not self.active:
            return
        omitted, digest = 0, ""
        if len(frame) > MAX_TRACE_BYTES:
            # Only the header is kept, so only the header is copied.
            omitted, digest = len(frame), hashlib.sha256(frame).hexdigest()
            encoded = bytes(memoryview(frame)[: wire.FRAME_HEADER_BYTES])
        else:
            encoded = bytes(frame)
        with self._lock:
            if generation != self._generation:
                return
            self._entries.append(
                _TraceEntry(time.monotonic_ns(), direction, encoded, omitted, digest)
            )
            self._bytes += len(encoded)
            while (
                len(self._entries) > MAX_TRACE_ENTRIES or self._bytes > MAX_TRACE_BYTES
            ):
                self._bytes -= len(self._entries.popleft().frame)

    def record_frame(self, generation: int, direction: str, frame: wire.Frame) -> None:
        if self.active:
            self.record_bytes(generation, direction, wire.serialize_frame(frame))

    def dump(
        self,
        generation: int,
        error: BaseException,
        *,
        process_returncode: int | None,
        last_status: bytes | None,
    ) -> Path | None:
        if not self.active:
            return None
        with self._lock:
            if generation != self._generation or self._dumped_generation == generation:
                return self._last_dump
            self._dumped_generation = generation
            started_ns = self._started_ns
            pid = self._pid
            entries = tuple(self._entries)
        document = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "generation": generation,
            "pid": pid,
            "process_returncode": process_returncode,
            "command": list(self._command),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "last_status_json": (
                last_status.decode("utf-8", errors="replace")
                if last_status is not None
                else None
            ),
            "frames": [entry.as_dict(started_ns) for entry in entries],
            "omitted_frames": sum(entry.omitted_bytes > 0 for entry in entries),
        }
        try:
            DEFAULT_TRACE_DIRECTORY.mkdir(parents=True, exist_ok=True, mode=0o700)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
            target = DEFAULT_TRACE_DIRECTORY / (
                f"splash-crash-g{generation}-{timestamp}.json"
            )
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", dir=DEFAULT_TRACE_DIRECTORY
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump(document, output, separators=(",", ":"))
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, target)
                with suppress(OSError):
                    self._prune()
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        except OSError:
            return None
        with self._lock:
            self._last_dump = target
        return target

    @staticmethod
    def _prune():
        files = sorted(
            DEFAULT_TRACE_DIRECTORY.glob("splash-crash-g*-*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        retained_bytes = 0
        for index, path in enumerate(files):
            retained_bytes += path.stat().st_size
            if index >= MAX_TRACE_FILES or retained_bytes > MAX_TRACE_DISK_BYTES:
                path.unlink(missing_ok=True)


def _load_trace(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != TRACE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported Splash crash trace")
    command = document.get("command")
    frames = document.get("frames")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
        or not isinstance(frames, list)
    ):
        raise ValueError("crash trace is missing its replay command or frames")
    return document


def replay(path: Path) -> int:
    document = _load_trace(path)
    outgoing = []
    for item in document["frames"]:
        if not isinstance(item, dict) or item.get("direction") != "client_to_engine":
            continue
        if "omitted_bytes" in item:
            raise ValueError(
                "crash trace omits a client-to-engine frame too large to record; "
                "it cannot be replayed"
            )
        offset = item.get("offset_micros")
        encoded = item.get("frame_base64")
        if type(offset) is not int or offset < 0 or not isinstance(encoded, str):
            raise ValueError("crash trace contains an invalid outgoing frame")
        outgoing.append((offset, base64.b64decode(encoded, validate=True)))
    if not outgoing:
        raise ValueError("crash trace contains no client-to-engine frames")

    process = subprocess.Popen(
        document["command"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
    )
    assert process.stdin is not None and process.stdout is not None
    reader: threading.Thread | None = None
    try:
        changed = threading.Condition()
        ready = False
        reader_done = False
        reader_error: BaseException | None = None

        def drain() -> None:
            nonlocal ready, reader_done, reader_error
            parser = wire.FrameParser()
            try:
                while chunk := process.stdout.read(64 * 1024):
                    offset = 0
                    while offset < len(chunk):
                        step = parser.consume(memoryview(chunk)[offset:])
                        offset += step.consumed_bytes
                        if step.issue:
                            raise RuntimeError(step.issue.describe())
                        if step.frame and step.frame.type is wire.FrameType.READY:
                            with changed:
                                ready = True
                                changed.notify_all()
                if issue := parser.finish():
                    raise RuntimeError(issue.describe())
            except BaseException as error:
                with changed:
                    reader_error = error
            finally:
                with changed:
                    reader_done = True
                    changed.notify_all()

        def check_reader(eof_message=None):
            # Called with changed held. EOF is only an error while replay
            # still needs Ready or has input left to send.
            if reader_error is not None:
                raise reader_error
            if reader_done and eof_message is not None:
                returncode = process.poll()
                suffix = "" if returncode is None else f" (exit status {returncode})"
                raise RuntimeError(eof_message + suffix)

        reader = threading.Thread(target=drain, name="splash-trace-replay", daemon=True)
        reader.start()
        with changed:
            if not changed.wait_for(
                lambda: ready or reader_done, timeout=REPLAY_TIMEOUT_SECONDS
            ):
                raise TimeoutError("replay engine did not become ready")
            check_reader(
                None if ready else "replay engine reached EOF before ReadyEvent"
            )
        os.set_blocking(process.stdin.fileno(), False)
        first_offset = outgoing[0][0]
        started = time.monotonic()
        for offset, encoded in outgoing:
            target = started + (offset - first_offset) / 1_000_000
            remaining = target - time.monotonic()
            with changed:
                if remaining > 0:
                    changed.wait_for(lambda: reader_done, timeout=remaining)
                check_reader("replay engine reached EOF before all input was sent")
            pending = memoryview(
                wire.refresh_request_deadline(encoded, time.time_ns() // 1000)
            )
            deadline = time.monotonic() + REPLAY_TIMEOUT_SECONDS
            while pending:
                with changed:
                    check_reader("replay engine reached EOF before all input was sent")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("replay engine input write timed out")
                try:
                    written = process.stdin.write(pending)
                except BlockingIOError:
                    written = None
                if written is None:
                    # Poll in bounded slices so reader completion also ends a
                    # write when the engine keeps stdin open without reading.
                    with selectors.DefaultSelector() as selector:
                        selector.register(process.stdin, selectors.EVENT_WRITE)
                        selector.select(min(remaining, 0.1))
                elif written <= 0:
                    raise BrokenPipeError("replay engine input accepted zero bytes")
                else:
                    pending = pending[written:]
        process.stdin.close()
        deadline = time.monotonic() + REPLAY_TIMEOUT_SECONDS
        while (returncode := process.poll()) is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, REPLAY_TIMEOUT_SECONDS)
            with changed:
                check_reader()
                # Clean EOF is expected here; wait for process teardown too.
                changed.wait(timeout=min(remaining, 0.1))
        reader.join(timeout=5)
        with changed:
            check_reader()
        return returncode
    finally:
        # Native SIGTERM requests graceful teardown. A hung engine still needs
        # a kill fallback, and cleanup must preserve the original replay error.
        with suppress(OSError):
            process.stdin.close()
        if process.poll() is None:
            with suppress(OSError):
                process.terminate()
            try:
                process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
            except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
                with suppress(OSError):
                    process.kill()
                with suppress(OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
                    process.wait(timeout=1.0)
        with suppress(OSError):
            process.stdout.close()
        if reader is not None and reader.is_alive():
            reader.join(timeout=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a Splash native crash trace")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    return replay(args.trace)


if __name__ == "__main__":
    raise SystemExit(main())

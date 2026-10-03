import base64
import hashlib
import io
import json
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

from dev.tests.engine import native_peer
from server import crash_trace
from server import protocol as wire


def ready_frame():
    """The frame the server reads for a Ready event."""
    return (
        wire.FrameParser()
        .consume(native_peer.serialize_event(wire.ReadyEvent(4, 131_072, False)))
        .frame
    )


class CrashTraceTest(unittest.TestCase):
    def _trace_for_command(self, command, *, frames=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "trace.json"
        if frames is None:
            frames = [(0, wire.serialize_message(wire.StatusRequestFrame(7)))]
        path.write_text(
            json.dumps(
                {
                    "schema_version": crash_trace.TRACE_SCHEMA_VERSION,
                    "command": list(command),
                    "frames": [
                        {
                            "direction": "client_to_engine",
                            "offset_micros": offset,
                            "frame_base64": base64.b64encode(frame).decode("ascii"),
                        }
                        for offset, frame in frames
                    ],
                }
            )
        )
        return path

    @contextmanager
    def _replay_peer(self, program, *, timeout=2.0, frames=None):
        path = self._trace_for_command((sys.executable, "-c", program), frames=frames)
        engines = []
        spawn = subprocess.Popen
        watchdog_fired = threading.Event()

        def spawn_and_record(*arguments, **keywords):
            engines.append(spawn(*arguments, **keywords))
            return engines[-1]

        def kill_stuck_peer():
            watchdog_fired.set()
            for engine in engines:
                if engine.poll() is None:
                    engine.kill()

        watchdog = threading.Timer(5.0, kill_stuck_peer)
        watchdog.start()
        try:
            with (
                mock.patch.object(crash_trace.subprocess, "Popen", spawn_and_record),
                mock.patch.object(crash_trace, "REPLAY_TIMEOUT_SECONDS", timeout),
                mock.patch.object(crash_trace, "SHUTDOWN_GRACE_SECONDS", 0.05),
            ):
                yield path, engines
                self.assertFalse(
                    watchdog_fired.is_set(), "replay exceeded test watchdog"
                )
        finally:
            watchdog.cancel()
            watchdog.join()
            # A failed assertion must not leave this regression's peer alive.
            for engine in engines:
                if engine.poll() is None:
                    engine.kill()
                engine.wait(timeout=5)
                engine.stdin.close()
                engine.stdout.close()

    def test_ring_enforces_entry_and_byte_limits_before_dump(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                mock.patch.object(crash_trace, "DEFAULT_TRACE_DIRECTORY", directory),
                mock.patch.object(crash_trace, "MAX_TRACE_ENTRIES", 3),
                mock.patch.object(crash_trace, "MAX_TRACE_BYTES", 12),
            ):
                ring = crash_trace.CrashTraceRing(
                    ("splash", "serve-native"), enabled=True
                )
                ring.start_generation(1, 50)
                for frame in (b"one", b"two2", b"three", b"four"):
                    ring.record_bytes(1, "client_to_engine", frame)
                path = ring.dump(
                    1,
                    RuntimeError("bounded"),
                    process_returncode=1,
                    last_status=None,
                )
            document = json.loads(path.read_text())
            decoded = [
                base64.b64decode(item["frame_base64"]) for item in document["frames"]
            ]
            self.assertEqual(decoded, [b"three", b"four"])
            self.assertLessEqual(sum(map(len, decoded)), 12)
            self.assertLessEqual(len(decoded), 3)

    def test_dump_is_bounded_private_atomic_and_once_per_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "crash"
            with mock.patch.object(crash_trace, "DEFAULT_TRACE_DIRECTORY", directory):
                ring = crash_trace.CrashTraceRing(
                    ("splash", "serve-native"), enabled=True
                )
                ring.start_generation(3, 1234)
                outgoing = wire.serialize_message(wire.StatusRequestFrame(9))
                ring.record_bytes(3, "client_to_engine", outgoing)
                ring.record_frame(3, "engine_to_client", ready_frame())
                path = ring.dump(
                    3,
                    RuntimeError("engine failed"),
                    process_returncode=-6,
                    last_status=b'{"schema_version":2,"ready":false}',
                )
                self.assertIsNotNone(path)
                self.assertEqual(
                    stat.S_IMODE(path.stat().st_mode),
                    0o600,
                )
                document = json.loads(path.read_text())
                self.assertEqual(document["schema_version"], 1)
                self.assertEqual(document["generation"], 3)
                self.assertEqual(document["pid"], 1234)
                self.assertEqual(document["error"]["type"], "RuntimeError")
                self.assertEqual(
                    [frame["direction"] for frame in document["frames"]],
                    ["client_to_engine", "engine_to_client"],
                )
                self.assertEqual(
                    ring.dump(
                        3,
                        RuntimeError("second failure"),
                        process_returncode=-6,
                        last_status=None,
                    ),
                    path,
                )
                self.assertEqual(len(list(directory.glob("*.json"))), 1)

    def test_oversized_frame_leaves_a_marker_that_replay_refuses(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                mock.patch.object(crash_trace, "DEFAULT_TRACE_DIRECTORY", directory),
                mock.patch.object(crash_trace, "MAX_TRACE_BYTES", 64),
            ):
                ring = crash_trace.CrashTraceRing(
                    ("splash", "serve-native"), enabled=True
                )
                ring.start_generation(1, 50)
                small = wire.serialize_message(wire.StatusRequestFrame(1))
                large = wire.serialize_message(
                    wire.MaskResponseFrame(2, 3, bytes(4 * 64))
                )
                ring.record_bytes(1, "client_to_engine", small)
                ring.record_bytes(1, "client_to_engine", large)
                path = ring.dump(
                    1, RuntimeError("failed"), process_returncode=1, last_status=None
                )
            document = json.loads(path.read_text())
            self.assertEqual(document["omitted_frames"], 1)
            kept, marker = document["frames"]
            self.assertEqual(base64.b64decode(kept["frame_base64"]), small)
            self.assertNotIn("frame_base64", marker)
            self.assertEqual(marker["omitted_bytes"], len(large))
            self.assertEqual(marker["sha256"], hashlib.sha256(large).hexdigest())
            self.assertEqual(
                base64.b64decode(marker["header_base64"]),
                large[: wire.FRAME_HEADER_BYTES],
            )
            with (
                mock.patch.object(crash_trace.subprocess, "Popen") as spawn,
                self.assertRaisesRegex(ValueError, "cannot be replayed"),
            ):
                crash_trace.replay(path)
            spawn.assert_not_called()

    def test_disabled_ring_does_not_reencode_received_frames(self):
        ring = crash_trace.CrashTraceRing(("splash", "serve-native"))
        frame = ready_frame()
        with mock.patch.object(crash_trace.wire, "serialize_frame") as serialize:
            ring.record_frame(1, "engine_to_client", frame)
        serialize.assert_not_called()

    def test_fake_process_factory_has_no_machine_global_trace(self):
        ring = crash_trace.CrashTraceRing(None)
        self.assertFalse(ring.active)
        ring.start_generation(1, 1)
        ring.record_bytes(
            1,
            "client_to_engine",
            wire.serialize_message(wire.StatusRequestFrame(1)),
        )
        self.assertIsNone(
            ring.dump(
                1,
                RuntimeError("ignored fake failure"),
                process_returncode=None,
                last_status=None,
            )
        )

    def test_replay_waits_for_ready_and_replays_recorded_input(self):
        ready_program = (
            "import sys; from dev.tests.engine import native_peer as n; "
            "from server import protocol as w; "
            "sys.stdout.buffer.write(n.serialize_event(w.ReadyEvent(4,131072,False))); "
            "sys.stdout.buffer.flush(); sys.stdin.buffer.read()"
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with mock.patch.object(crash_trace, "DEFAULT_TRACE_DIRECTORY", directory):
                ring = crash_trace.CrashTraceRing(
                    (sys.executable, "-c", ready_program), enabled=True
                )
                ring.start_generation(1, 50)
                ring.record_bytes(
                    1,
                    "client_to_engine",
                    wire.serialize_message(wire.StatusRequestFrame(7)),
                )
                path = ring.dump(
                    1,
                    RuntimeError("replay me"),
                    process_returncode=None,
                    last_status=None,
                )
            self.assertEqual(crash_trace.replay(path), 0)

    def test_replay_stops_the_engine_when_the_stream_fails(self):
        for ignore_term in (False, True):
            with self.subTest(ignore_term=ignore_term):
                program = (
                    "import signal, sys, time; "
                    + (
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                        if ignore_term
                        else ""
                    )
                    + "sys.stdout.buffer.write(b'NOPE' + bytes(20)); "
                    "sys.stdout.buffer.flush(); time.sleep(60)"
                )
                with self._replay_peer(program) as (path, engines):
                    with self.assertRaisesRegex(
                        RuntimeError, "protocol_fatal:bad_magic"
                    ):
                        crash_trace.replay(path)
                    (engine,) = engines
                    expected_signal = signal.SIGKILL if ignore_term else signal.SIGTERM
                    self.assertEqual(engine.poll(), -expected_signal)
                    self.assertTrue(engine.stdin.closed and engine.stdout.closed)

    def test_replay_kills_an_engine_after_a_write_failure(self):
        program = (
            "import os, signal, sys, time; "
            "from dev.tests.engine import native_peer as n; "
            "from server import protocol as w; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); os.close(0); "
            "sys.stdout.buffer.write(n.serialize_event(w.ReadyEvent(4,131072,False))); "
            "sys.stdout.buffer.flush(); time.sleep(60)"
        )
        with self._replay_peer(program) as (path, engines):
            with self.assertRaises(BrokenPipeError):
                crash_trace.replay(path)
            (engine,) = engines
            self.assertEqual(engine.poll(), -signal.SIGKILL)
            self.assertTrue(engine.stdin.closed and engine.stdout.closed)

    def test_replay_interrupts_a_blocked_write(self):
        request = native_peer.request_frame(
            request_id=7,
            priority=wire.RequestPriority.FOREGROUND,
            remaining_deadline_micros=45_000_000,
            logical_max_output_tokens=16,
            prompt_tokens=(1,) * 262_144,
            seed=99,
        )
        for outcome in ("timeout", "protocol_error", "eof"):
            with self.subTest(outcome=outcome):
                program = (
                    "import os, signal, sys, time; "
                    "from dev.tests.engine import native_peer as n; "
                    "from server import protocol as w; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "sys.stdout.buffer.write("
                    "n.serialize_event(w.ReadyEvent(4,1048576,False))); "
                    "sys.stdout.buffer.flush(); "
                    "sys.stdin.buffer.read(1); "
                    + {
                        "timeout": "",
                        "protocol_error": (
                            "sys.stdout.buffer.write(b'NOPE' + bytes(20)); "
                            "sys.stdout.buffer.flush(); "
                        ),
                        "eof": "os.close(1); ",
                    }[outcome]
                    + "time.sleep(60)"
                )
                with self._replay_peer(
                    program,
                    timeout=0.5,
                    frames=[(0, wire.serialize_message(request))],
                ) as (path, engines):
                    expected = TimeoutError if outcome == "timeout" else RuntimeError
                    message = {
                        "timeout": "input write timed out",
                        "protocol_error": "protocol_fatal:bad_magic",
                        "eof": "EOF before all input was sent",
                    }[outcome]
                    with self.assertRaisesRegex(expected, message):
                        crash_trace.replay(path)
                    (engine,) = engines
                    self.assertEqual(engine.poll(), -signal.SIGKILL)
                    self.assertTrue(engine.stdin.closed and engine.stdout.closed)

    def test_replay_reports_eof_before_ready_without_waiting_for_timeout(self):
        with self._replay_peer("raise SystemExit(23)") as (path, engines):
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "EOF before ReadyEvent"):
                crash_trace.replay(path)
            self.assertLess(time.monotonic() - started, 1.0)
            (engine,) = engines
            self.assertEqual(engine.poll(), 23)
            self.assertTrue(engine.stdin.closed and engine.stdout.closed)

    def test_replay_eof_interrupts_the_recorded_input_delay(self):
        frame = wire.serialize_message(wire.StatusRequestFrame(7))
        program = (
            "import sys; from dev.tests.engine import native_peer as n; "
            "from server import protocol as w; "
            "sys.stdout.buffer.write(n.serialize_event(w.ReadyEvent(4,131072,False))); "
            "sys.stdout.buffer.flush(); "
            f"sys.stdin.buffer.read({len(frame)}); raise SystemExit(23)"
        )
        with self._replay_peer(program, frames=[(0, frame), (2_000_000, frame)]) as (
            path,
            engines,
        ):
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "EOF before all input was sent"):
                crash_trace.replay(path)
            self.assertLess(time.monotonic() - started, 1.0)
            (engine,) = engines
            self.assertEqual(engine.poll(), 23)
            self.assertTrue(engine.stdin.closed and engine.stdout.closed)

    def test_replay_allows_eof_after_input_while_the_engine_finishes(self):
        for returncode in (0, 23):
            with self.subTest(returncode=returncode):
                program = (
                    "import os, sys, time; "
                    "from dev.tests.engine import native_peer as n; "
                    "from server import protocol as w; "
                    "sys.stdout.buffer.write("
                    "n.serialize_event(w.ReadyEvent(4,131072,False))); "
                    "sys.stdout.buffer.flush(); "
                    "sys.stdin.buffer.read(); os.close(1); "
                    f"time.sleep(0.05); raise SystemExit({returncode})"
                )
                with self._replay_peer(program) as (path, engines):
                    self.assertEqual(crash_trace.replay(path), returncode)
                    (engine,) = engines
                    self.assertEqual(engine.poll(), returncode)
                    self.assertTrue(engine.stdin.closed and engine.stdout.closed)

    def test_replay_reader_error_interrupts_schedule_and_process_wait(self):
        frame = wire.serialize_message(wire.StatusRequestFrame(7))
        program = (
            "import signal, sys, time; from dev.tests.engine import native_peer as n; "
            "from server import protocol as w; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "sys.stdout.buffer.write(n.serialize_event(w.ReadyEvent(4,131072,False))); "
            "sys.stdout.buffer.flush(); "
            f"sys.stdin.buffer.read({len(frame)}); "
            "sys.stdout.buffer.write(b'NOPE' + bytes(20)); "
            "sys.stdout.buffer.flush(); time.sleep(60)"
        )
        for delayed_frame in (False, True):
            with self.subTest(delayed_frame=delayed_frame):
                frames = [(0, frame)]
                if delayed_frame:
                    frames.append((60_000_000, frame))
                with self._replay_peer(program, frames=frames) as (path, engines):
                    with self.assertRaisesRegex(
                        RuntimeError, "protocol_fatal:bad_magic"
                    ):
                        crash_trace.replay(path)
                    (engine,) = engines
                    self.assertEqual(engine.poll(), -signal.SIGKILL)
                    self.assertTrue(engine.stdin.closed and engine.stdout.closed)

    def test_replay_reaps_an_engine_that_never_becomes_ready(self):
        program = (
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(60)"
        )
        with self._replay_peer(program, timeout=0.1) as (path, engines):
            with self.assertRaisesRegex(TimeoutError, "did not become ready"):
                crash_trace.replay(path)
            (engine,) = engines
            self.assertIsNotNone(engine.poll())
            self.assertTrue(engine.stdin.closed and engine.stdout.closed)

    def test_replay_reaps_an_engine_that_never_finishes(self):
        program = (
            "import signal, sys, time; from dev.tests.engine import native_peer as n; "
            "from server import protocol as w; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "sys.stdout.buffer.write(n.serialize_event(w.ReadyEvent(4,131072,False))); "
            "sys.stdout.buffer.flush(); sys.stdin.buffer.read(); time.sleep(60)"
        )
        with self._replay_peer(program) as (path, engines):
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                crash_trace.replay(path)
            self.assertEqual(caught.exception.timeout, 2.0)
            (engine,) = engines
            self.assertEqual(engine.poll(), -signal.SIGKILL)
            self.assertTrue(engine.stdin.closed and engine.stdout.closed)

    def test_replay_preserves_protocol_error_when_cleanup_raises(self):
        path = self._trace_for_command(("fake-native",))
        output = io.BytesIO(b"NOPE" + bytes(20))
        engine = mock.Mock()
        engine.stdout.read.side_effect = output.read
        engine.stdout.close.side_effect = OSError("stdout close failed")
        engine.stdin.close.side_effect = OSError("stdin close failed")
        engine.poll.return_value = None
        engine.terminate.side_effect = OSError("terminate failed")
        engine.wait.side_effect = [OSError("wait failed"), 0]
        with mock.patch.object(crash_trace.subprocess, "Popen", return_value=engine):
            with self.assertRaisesRegex(RuntimeError, "protocol_fatal:bad_magic"):
                crash_trace.replay(path)
        engine.kill.assert_called_once_with()
        self.assertEqual(engine.wait.call_count, 2)
        engine.stdout.close.assert_called_once_with()

    def test_replay_restamps_request_deadlines_from_replay_time(self):
        capture_program = (
            "import sys; from dev.tests.engine import native_peer as n; "
            "from server import protocol as w; "
            "sys.stdout.buffer.write(n.serialize_event(w.ReadyEvent(4,131072,False))); "
            "sys.stdout.buffer.flush(); "
            "open(sys.argv[1], 'wb').write(sys.stdin.buffer.read())"
        )
        request = native_peer.request_frame(
            request_id=7,
            priority=wire.RequestPriority.FOREGROUND,
            remaining_deadline_micros=45_000_000,
            logical_max_output_tokens=16,
            # Exceed pipe capacity to exercise successful partial writes too.
            prompt_tokens=(1, 2, 3) * 100_000,
            sampling=wire.SamplingParameters(0.5, 0.25, 8),
            seed=99,
            constraint=wire.ConstraintMode.TOKEN_MASK,
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            captured = directory / "replayed.bin"
            with mock.patch.object(crash_trace, "DEFAULT_TRACE_DIRECTORY", directory):
                ring = crash_trace.CrashTraceRing(
                    (sys.executable, "-c", capture_program, str(captured)), enabled=True
                )
                ring.start_generation(1, 50)
                for message in (request, wire.StatusRequestFrame(7)):
                    ring.record_bytes(
                        1, "client_to_engine", wire.serialize_message(message)
                    )
                path = ring.dump(
                    1,
                    RuntimeError("replay me"),
                    process_returncode=None,
                    last_status=None,
                )
            started = time.time_ns() // 1000
            self.assertEqual(crash_trace.replay(path), 0)
            finished = time.time_ns() // 1000
            replayed = captured.read_bytes()
        (replayed_request, _), (status_request, _) = (
            native_peer.ClientFrameReader().feed(replayed)
        )
        self.assertEqual(status_request, wire.StatusRequestFrame(7))
        budget = request.remaining_deadline_micros
        self.assertGreaterEqual(
            replayed_request.absolute_deadline_unix_micros, started + budget
        )
        self.assertLessEqual(
            replayed_request.absolute_deadline_unix_micros, finished + budget
        )
        self.assertEqual(
            replace(
                replayed_request,
                absolute_deadline_unix_micros=request.absolute_deadline_unix_micros,
            ),
            request,
        )


if __name__ == "__main__":
    unittest.main()

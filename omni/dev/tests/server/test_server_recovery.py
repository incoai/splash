import concurrent.futures
import dataclasses
import http.client
import io
import json
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from dev.tests import native_peer
from dev.tests.audio_fixtures import (
    TTS_PROFILE,
    AudioHarnessTestCase,
    FakeAudioRuntime,
    audio_main_args,
    speech_body,
)
from dev.tests.server_fixtures import AUDIO_READY, FakeFactory
from server import audio_backend, audio_runtime, audio_server
from server import audio_protocol as audio_wire
from server import backend as backend_api
from server import protocol as wire
from server import runtime as engine_runtime


def answer_status(process, message):
    if isinstance(message, wire.StatusRequestFrame):
        process.send(native_peer.status_event(message.correlation_id))


def speech_job(request_id=101, *, deadline=10.0):
    """A speech request's job, as the frontend prepares one."""
    return audio_backend.AudioJob(
        request_id,
        audio_wire.SpeechRequest(0, (11, 12, 13, 14)),
        time.monotonic() + deadline,
    )


class RecoveringRuntime(FakeAudioRuntime):
    """A serving engine the test fails. Each relaunch waits for the test to
    put its outcome: None for Ready, or the error the relaunch raises."""

    def __init__(self):
        super().__init__()
        self.outcomes = queue.Queue()
        self.startup_calls = 0

    def fail(self, backend, error=None, served=1.0):
        self.ready = False
        backend._engine_failed(
            error or engine_runtime.EngineUnhealthy("native protocol reached EOF"),
            served,
        )

    def wait_ready(self, timeout=None):
        self.startup_calls += 1
        outcome = self.outcomes.get(timeout=3)
        if outcome is not None:
            raise outcome
        self.ready = True
        return True

    def close(self):
        self.outcomes.put(engine_runtime.RuntimeClosed("runtime is closed"))
        super().close()


class ServerRecoveryTests(AudioHarnessTestCase):
    @staticmethod
    def wait_until(predicate, timeout=1):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError("test condition did not become true")
            time.sleep(0.005)

    def backend(self, runtime):
        backend = audio_backend.AudioBackend(runtime, lambda _record: None)
        self.addCleanup(backend.close)
        return backend

    @staticmethod
    def engine(factory, **options):
        return audio_runtime.AudioRuntime(
            process_factory=factory, eager_start=True, **options
        )

    def relaunched(self, runtime, backend):
        runtime.outcomes.put(None)
        self.wait_until(lambda: backend.recovery is None and runtime.ready)

    def test_probes_share_one_relaunch_without_waiting_for_it(self):
        for first_path in ("/ready", "/status"):
            with self.subTest(first_path=first_path):
                runtime = RecoveringRuntime()
                harness = self.harness(runtime)
                runtime.fail(harness.backend)
                started = time.monotonic()
                response = harness.request("GET", first_path)
                self.assertLess(time.monotonic() - started, 0.5)
                self.assertEqual(response[0], 503 if first_path == "/ready" else 200)
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    probes = [
                        pool.submit(harness.request, "GET", "/status")
                        for _ in range(16)
                    ]
                    for probe in probes:
                        status, _, payload = probe.result(1)
                        self.assertEqual(status, 200)
                        snapshot = json.loads(payload)
                        self.assertFalse(snapshot["ready"])
                        self.assertTrue(snapshot["transport"]["recovering"])
                self.assertEqual(runtime.startup_calls, 1)
                self.relaunched(runtime, harness.backend)
                self.assertEqual(harness.request("GET", "/ready")[0], 200)
                snapshot = json.loads(harness.request("GET", "/status")[2])
                self.assertFalse(snapshot["transport"]["recovering"])
                self.assertEqual(runtime.startup_calls, 1)
                self.assertEqual(runtime.requests, [])

    def test_fast_failures_back_off_then_stop_restarting(self):
        runtime = RecoveringRuntime()
        harness = self.harness(runtime)
        backend = harness.backend
        clock = [100.0]
        with (
            mock.patch.object(backend_api, "RESTART_BACKOFF_SECONDS", 0.01),
            mock.patch.object(
                backend_api, "time", SimpleNamespace(monotonic=lambda: clock[0])
            ),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            for delay in (0.0, 0.01):
                runtime.fail(backend, served=1.0)
                self.assertAlmostEqual(backend.restart_due - clock[0], delay)
                self.relaunched(runtime, backend)
            runtime.fail(backend, served=1.0)
        self.assertEqual(runtime.startup_calls, 2)
        refusal = backend.refusal()
        self.assertEqual((refusal.status, refusal.code), (500, "engine_failed"))
        self.assertIn("failed 3 times in a row", refusal.message)
        self.assertRegex(refusal.message, "Crash trace|SPLASH_CRASH_TRACE")
        self.assertEqual(
            console.call_args_list[-1].args[0], f"Engine stopped · {refusal.message}"
        )
        self.assertEqual(harness.request("GET", "/ready")[0], 503)
        transport = json.loads(harness.request("GET", "/status")[2])["transport"]
        self.assertTrue(transport["stopped"])
        self.assertFalse(transport["recovering"])
        self.assertEqual(transport["error"], refusal.message)
        status, _, payload = harness.request("POST", "/v1/audio/speech", speech_body())
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(payload)["error"]["code"], "engine_failed")
        self.assertEqual(runtime.startup_calls, 2)

    def test_the_server_is_told_once_relaunching_stops(self):
        runtime = RecoveringRuntime()
        backend = self.harness(runtime).backend
        stopped = []
        backend.on_stopped = lambda: stopped.append(backend.fatal_error)
        with (
            mock.patch.object(backend_api, "RESTART_BACKOFF_SECONDS", 0.0),
            mock.patch.object(backend_api, "print_status"),
        ):
            for _ in range(backend_api.CRASH_LOOP_LIMIT - 1):
                runtime.fail(backend, served=1.0)
                self.relaunched(runtime, backend)
                self.assertEqual(stopped, [])
            runtime.fail(backend, served=1.0)
        self.assertEqual(len(stopped), 1)
        self.assertIn("The server exits", stopped[0])

    def test_full_window_of_service_resets_the_crash_count(self):
        runtime = RecoveringRuntime()
        backend = self.backend(runtime)
        with mock.patch.object(backend_api, "print_status"):
            for _ in range(5):
                runtime.fail(backend, served=61.0)
                self.assertEqual(backend.failures, 1)
                self.assertLessEqual(backend.restart_due, time.monotonic())
                self.relaunched(runtime, backend)
        self.assertIsNone(backend.fatal_error)
        self.assertEqual(runtime.startup_calls, 5)

    def test_failure_during_inflight_refresh_restarts_at_once(self):
        runtime = RecoveringRuntime()
        backend = self.backend(runtime)
        refreshing = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        status = runtime.status

        def unanswered(timeout, fail_unanswered=False):
            if not fail_unanswered:
                raise TimeoutError("native status response timed out")
            refreshing.set()
            release.wait(2)
            return status(timeout)

        with (
            mock.patch.object(runtime, "status", side_effect=unanswered),
            mock.patch.object(backend_api, "print_status"),
        ):
            backend.status()
            self.assertTrue(refreshing.wait(1))
            runtime.fail(backend, served=120.0)
            release.set()
            self.wait_until(lambda: runtime.startup_calls == 1, 0.5)

    def test_failed_relaunch_counts_toward_the_crash_loop(self):
        runtime = RecoveringRuntime()
        for reason in ("a", "b"):
            runtime.outcomes.put(engine_runtime.EngineUnhealthy(reason))
        backend = self.backend(runtime)
        with (
            mock.patch.object(backend_api, "RESTART_BACKOFF_SECONDS", 0.01),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            runtime.fail(backend, served=1.0)
            self.wait_until(lambda: backend.fatal_error is not None)
        self.assertEqual(runtime.startup_calls, 2)
        self.assertIn("failed 3 times in a row", backend.fatal_error)
        self.assertIn("(b)", backend.fatal_error)
        self.assertEqual(
            [call.args[0] for call in console.call_args_list],
            [
                "Engine failed · native protocol reached EOF",
                "Engine restart failed · a",
                "Engine restart failed · b",
                f"Engine stopped · {backend.fatal_error}",
            ],
        )

    def test_changed_ready_limits_stop_restarting_at_once(self):
        factory = FakeFactory(handler=answer_status, audio=True)
        runtime = self.engine(factory)
        backend = self.backend(runtime)
        factory.initial_output = native_peer.serialize_audio_event(
            audio_wire.ReadyEvent(4, 65_536, audio_wire.ModelKind.TTS)
        )
        with (
            mock.patch.object(runtime._crash_trace, "dump"),
            mock.patch.object(backend_api, "print_status"),
        ):
            factory.processes[0].kill()
            self.wait_until(lambda: backend.fatal_error is not None, 2)
        self.assertEqual(len(factory.processes), 2)
        # The runtime's error already says how to recover.
        self.assertEqual(
            backend.fatal_error,
            "native context window, concurrency or model kind changed; "
            "restart the Splash server",
        )
        transport = backend.status()["transport"]
        self.assertTrue(transport["stopped"])
        self.assertFalse(transport["recovering"])
        self.assertEqual(transport["error"], backend.fatal_error)
        refusal = backend.refusal()
        self.assertEqual((refusal.status, refusal.code), (500, "engine_failed"))

    def test_stop_after_failed_relaunches_names_the_last_crash_trace(self):
        factory = FakeFactory(audio=True)

        def launch():
            process = factory()
            if len(factory.processes) > 1:
                # Every relaunch fails before Ready.
                process.close_stdout()
            return process

        runtime = self.engine(launch)
        factory.initial_output = b""
        trace = mock.Mock(last_dump=None)

        def dump(generation, _error, **_details):
            # A real trace is fsynced before it is named.
            time.sleep(0.05)
            trace.last_dump = f"/traces/g{generation}.json"

        trace.dump.side_effect = dump
        backend = self.backend(runtime)
        with (
            mock.patch.object(runtime, "_crash_trace", trace),
            mock.patch.object(backend_api, "RESTART_BACKOFF_SECONDS", 0.01),
            mock.patch.object(backend_api, "print_status"),
        ):
            factory.processes[0].kill()
            self.wait_until(lambda: backend.fatal_error is not None, 2)
        self.assertEqual(len(factory.processes), 3)
        self.assertIn("Crash trace: /traces/g3.json.", backend.fatal_error)

    def test_failure_right_after_a_relaunch_ready_is_counted_once(self):
        factory = FakeFactory(audio=True)

        def launch():
            process = factory()
            if len(factory.processes) == 2:
                # The relaunched engine fails as soon as it is Ready.
                process.close_stdout()
            return process

        runtime = self.engine(launch)
        backend = self.backend(runtime)
        relaunch = backend._relaunch
        relaunched = threading.Event()

        def relaunch_then_signal(due):
            relaunch(due)
            relaunched.set()

        dump = runtime._crash_trace.dump

        def dump_after_the_relaunch(generation, *args, **kwargs):
            # The listener hears of the relaunched engine's failure only
            # after the relaunch has seen its outcome.
            if generation == 2:
                relaunched.wait(1)
            return dump(generation, *args, **kwargs)

        with (
            mock.patch.object(backend, "_relaunch", side_effect=relaunch_then_signal),
            mock.patch.object(
                runtime._crash_trace, "dump", side_effect=dump_after_the_relaunch
            ),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            factory.processes[0].kill()
            self.wait_until(lambda: console.call_count >= 2, 2)
        self.assertEqual(
            [call.args[0] for call in console.call_args_list],
            ["Engine failed · native protocol reached EOF"] * 2,
        )
        self.assertEqual(backend.failures, 2)
        self.assertIsNone(backend.fatal_error)

    def test_crash_that_stops_restarts_fails_its_request_without_retry(self):
        factory = FakeFactory(handler=answer_status, audio=True)
        runtime = self.engine(factory)
        backend = self.backend(runtime)
        job = speech_job()
        with (
            mock.patch.object(backend_api, "RESTART_BACKOFF_SECONDS", 0.01),
            mock.patch.object(runtime._crash_trace, "dump"),
            mock.patch.object(backend_api, "print_status"),
        ):
            for launched in (2, 3):
                factory.processes[-1].kill()
                self.wait_until(
                    lambda: len(factory.processes) == launched and runtime.ready, 2
                )
            backend.submit(job)
            factory.processes[-1].stdin.wait_for(audio_wire.SpeechRequest)
            factory.processes[-1].close_stdout()
            kind, error = job.events.get(timeout=1)
        self.assertEqual(
            (kind, error.status, error.code), ("error", 500, "engine_failed")
        )
        self.assertIn("failed 3 times in a row", error.message)
        self.assertEqual(len(factory.processes), 3)

    def test_recovery_rejects_before_body_or_ingress_and_advertises_retry(self):
        runtime = RecoveringRuntime()
        harness = self.harness(runtime, queue_size=1)
        runtime.fail(harness.backend)
        with (
            mock.patch.object(harness.app, "prepare_speech") as prepare_speech,
            mock.patch.object(harness.app, "prepare_transcription") as prepare_upload,
        ):
            for path in ("/v1/audio/speech", "/v1/audio/transcriptions"):
                with self.subTest(path=path):
                    connection = http.client.HTTPConnection(
                        *harness.server.server_address, timeout=1
                    )
                    try:
                        connection.putrequest("POST", path)
                        connection.putheader("Content-Type", "application/json")
                        connection.putheader("Content-Length", "100")
                        connection.endheaders()
                        response = connection.getresponse()
                        self.assertEqual(response.status, 503)
                        self.assertEqual(response.getheader("Retry-After"), "1")
                        self.assertEqual(
                            json.loads(response.read())["error"]["type"],
                            "server_error",
                        )
                    finally:
                        connection.close()
                    self.assertEqual(harness.server.requests.stats()["active"], 0)
                    self.assertEqual(runtime.pending_count, 0)
            prepare_speech.assert_not_called()
            prepare_upload.assert_not_called()
        self.wait_until(lambda: runtime.startup_calls == 1)
        self.assertEqual(runtime.requests, [])
        # The control routes answer while the engine recovers.
        self.assertEqual(harness.request("GET", "/v1/models")[0], 200)
        self.assertEqual(harness.request("GET", "/health")[0], 200)

    def test_successful_probe_clears_unanswered_status_and_engine_error(self):
        runtime = RecoveringRuntime()
        backend = self.backend(runtime)
        with (
            mock.patch.object(runtime, "status", side_effect=TimeoutError),
            mock.patch.object(backend_api, "print_status"),
        ):
            runtime.fail(backend)
            # The relaunch succeeds; its status refresh is not answered.
            self.relaunched(runtime, backend)
            self.assertFalse(backend.status()["ready"])
            self.wait_until(lambda: backend.recovery is None)
        self.assertIsNotNone(backend.status_unanswered_since)
        self.assertEqual(backend.engine_error, "native protocol reached EOF")
        with mock.patch.object(backend_api, "print_status") as console:
            self.assertTrue(backend.status()["ready"])
        console.assert_called_once_with("Engine restarted")
        self.assertIsNone(backend.status_unanswered_since)
        self.assertIsNone(backend.engine_error)

    def test_background_recovery_and_waiters_share_the_native_startup(self):
        factory = FakeFactory(handler=answer_status, audio=True)
        runtime = self.engine(factory)
        backend = self.backend(runtime)
        self.assertTrue(backend.status()["ready"])
        factory.initial_output = b""
        with mock.patch.object(backend_api, "print_status"):
            factory.processes[0].kill()
            self.wait_until(lambda: len(factory.processes) == 2)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                waiters = [pool.submit(runtime.wait_ready, 1) for _ in range(4)]
                for _ in range(10):
                    self.assertEqual(backend.refusal().code, "engine_recovering")
                self.assertEqual(len(factory.processes), 2)
                factory.processes[1].send(AUDIO_READY)
                for waiter in waiters:
                    self.assertTrue(waiter.result(1))
            self.wait_until(lambda: backend.recovery is None)
        self.assertTrue(backend.status()["ready"])
        self.assertEqual(runtime.pending_count, 0)
        self.assertEqual(
            factory.processes[1].stdin.messages(audio_wire.SpeechRequest), []
        )

    def test_idle_engine_death_restarts_before_traffic_arrives(self):
        factory = FakeFactory(audio=True)
        runtime = self.engine(factory)
        backend = self.backend(runtime)
        factory.processes[0].kill()
        self.wait_until(lambda: len(factory.processes) == 2 and runtime.ready, 2)
        self.assertEqual(runtime.restart_count, 1)
        self.assertIsNone(backend.refusal())

    def test_idle_engine_that_stops_answering_status_is_restarted(self):
        def answer_after_the_first(process, message):
            if process is not factory.processes[0]:
                answer_status(process, message)

        factory = FakeFactory(handler=answer_after_the_first, audio=True)
        runtime = self.engine(factory)
        backend = self.backend(runtime)
        with (
            mock.patch.object(engine_runtime, "STATUS_ANSWER_LIMIT_SECONDS", 0.05),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            # The probe's timeout owes a refresh, which goes unanswered too.
            self.assertFalse(backend.status()["ready"])
            self.wait_until(lambda: len(factory.processes) == 2 and runtime.ready, 2)
            self.wait_until(lambda: backend.recovery is None)
        self.assertEqual(
            console.call_args_list[0].args[0],
            "Engine failed · native loop did not answer status within 0.05 s",
        )
        self.assertEqual(runtime.pending_count, 0)
        self.assertTrue(backend.status()["ready"])

    def test_engine_failure_and_failed_restart_are_reported(self):
        factory = FakeFactory(audio=True)

        def launch():
            if factory.processes:
                raise FileNotFoundError("splash")
            return factory()

        runtime = self.engine(launch)
        backend = self.backend(runtime)
        with mock.patch.object(backend_api, "print_status") as console:
            factory.processes[0].kill()
            self.wait_until(lambda: console.call_count >= 2)
            transport = backend.status()["transport"]
        failed, restart = (call.args[0] for call in console.call_args_list[:2])
        self.assertEqual(failed, "Engine failed · native protocol reached EOF")
        self.assertRegex(
            restart, "^Engine restart failed · native engine executable is missing"
        )
        self.assertTrue(transport["recovering"])
        self.assertIn("executable is missing", transport["error"])

    def test_failed_restarts_end_in_one_stop_line(self):
        factory = FakeFactory(audio=True)

        def launch():
            if factory.processes:
                raise FileNotFoundError("splash")
            return factory()

        runtime = self.engine(launch)
        backend = self.backend(runtime)
        with (
            mock.patch.object(backend_api, "RESTART_BACKOFF_SECONDS", 0.01),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            factory.processes[0].kill()
            self.wait_until(lambda: console.call_count >= 4)
        lines = [call.args[0] for call in console.call_args_list]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "Engine failed · native protocol reached EOF")
        for line in lines[1:3]:
            self.assertRegex(
                line, "^Engine restart failed · native engine executable is missing"
            )
        self.assertEqual(lines[3], f"Engine stopped · {backend.fatal_error}")
        transport = backend.status()["transport"]
        self.assertTrue(transport["stopped"])
        self.assertFalse(transport["recovering"])

    def test_engine_failure_under_a_request_asks_its_client_to_retry(self):
        # The engine stops, or breaks the protocol; either way it fails, and
        # its requests with it.
        protocol_failure = wire.ErrorEvent(
            wire.FailureClass.PROTOCOL_FATAL,
            0,
            False,
            b"invalid_request_id",
            b"request id is already active",
        )
        for fail, reason in (
            (
                lambda process: process.close_stdout(),
                "native protocol reached EOF",
            ),
            (
                lambda process: process.send(protocol_failure),
                "request id is already active",
            ),
        ):
            with self.subTest(reason=reason):
                factory = FakeFactory(audio=True)
                runtime = self.engine(factory)
                backend = self.backend(runtime)
                job = speech_job()
                with mock.patch.object(backend_api, "print_status") as console:
                    backend.submit(job)
                    factory.processes[0].stdin.wait_for(audio_wire.SpeechRequest)
                    fail(factory.processes[0])
                    kind, error = job.events.get(timeout=1)
                    self.wait_until(lambda: console.called)
                self.assertEqual(kind, "error")
                self.assertEqual(
                    (error.status, error.code, error.message),
                    (
                        503,
                        "runtime_unavailable",
                        "the inference engine stopped unexpectedly and is "
                        "restarting; retry the request",
                    ),
                )
                # The engine's own reason stays on the console.
                self.assertEqual(
                    console.call_args_list[0].args[0], f"Engine failed · {reason}"
                )

    def test_recovery_refusals_carry_the_last_engine_failure(self):
        runtime = RecoveringRuntime()
        runtime.outcomes.put(engine_runtime.EngineUnhealthy("GPU is gone"))
        harness = self.harness(runtime)
        with (
            mock.patch.object(backend_api, "RESTART_BACKOFF_SECONDS", 0.01),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            runtime.fail(harness.backend)
            # The second relaunch waits for its outcome.
            self.wait_until(lambda: runtime.startup_calls == 2)
            console.assert_called_with(
                "Engine restart failed · GPU is gone", error=True
            )
            status, _, payload = harness.request(
                "POST", "/v1/audio/speech", speech_body()
            )
            self.assertEqual(status, 503)
            self.assertEqual(
                json.loads(payload)["error"]["message"],
                "engine is recovering; retry shortly (last failure: GPU is gone)",
            )
            transport = json.loads(harness.request("GET", "/status")[2])["transport"]
            self.assertEqual(transport["error"], "GPU is gone")
            self.relaunched(runtime, harness.backend)
            self.assertIsNone(harness.backend.refusal())
            console.assert_called_with("Engine restarted")
            transport = json.loads(harness.request("GET", "/status")[2])["transport"]
            self.assertNotIn("error", transport)

    def test_startup_protocol_failure_ends_with_one_error_line(self):
        runtime_type = audio_runtime.AudioRuntime
        factory = FakeFactory(
            initial_output=b"not a frame".ljust(wire.FRAME_HEADER_BYTES, b"\0"),
            audio=True,
        )
        with (
            mock.patch.object(
                audio_server, "parse_args", return_value=audio_main_args()
            ),
            mock.patch.object(
                audio_server.ModelProfile, "load", return_value=TTS_PROFILE
            ),
            mock.patch.object(
                audio_server.AutoTokenizer, "from_pretrained", return_value=object()
            ),
            mock.patch.object(
                audio_server.audio_runtime,
                "AudioRuntime",
                side_effect=lambda _command, **options: runtime_type(
                    process_factory=factory, **options
                ),
            ),
            mock.patch.object(audio_server, "AudioServer"),
            mock.patch.object(audio_server.signal, "signal"),
            mock.patch("sys.stdout", new_callable=io.StringIO),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            audio_server.main()
        (line,) = stderr.getvalue().splitlines()
        self.assertIn("Error · ", line)
        self.assertIn("bad_magic", line)
        self.assertIsNotNone(factory.processes[0].poll())

    def test_startup_ends_with_the_failure_its_engine_reports(self):
        # An engine whose start fails (a --max-memory below what its model
        # takes) sends why in place of its ReadyEvent; any other event before
        # Ready still breaks the protocol.
        reason = (
            "--max-memory 1.00 GiB is below the 4.06 GiB this model takes "
            "with 2 lanes; raise it, or lower --lanes"
        )
        cases = (
            (wire.FailureClass.ENGINE_UNHEALTHY, 0, reason, f"Error · {reason}"),
            (
                wire.FailureClass.REQUEST_ERROR,
                7,
                "no such request",
                "Error · native event arrived before ReadyEvent",
            ),
        )
        runtime_type = audio_runtime.AudioRuntime
        for failure, request_id, message, expected in cases:
            factory = FakeFactory(
                initial_output=native_peer.serialize_audio_event(
                    wire.ErrorEvent(
                        failure, request_id, False, b"start_failed", message.encode()
                    )
                ),
                audio=True,
            )
            with (
                self.subTest(failure=failure),
                mock.patch.object(
                    audio_server, "parse_args", return_value=audio_main_args()
                ),
                mock.patch.object(
                    audio_server.ModelProfile, "load", return_value=TTS_PROFILE
                ),
                mock.patch.object(
                    audio_server.AutoTokenizer, "from_pretrained", return_value=object()
                ),
                mock.patch.object(
                    audio_server.audio_runtime,
                    "AudioRuntime",
                    side_effect=lambda _command, **options: runtime_type(
                        process_factory=factory, **options
                    ),
                ),
                mock.patch.object(audio_server, "AudioServer"),
                mock.patch.object(audio_server.signal, "signal"),
                mock.patch("sys.stdout", new_callable=io.StringIO),
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
                self.assertRaisesRegex(SystemExit, "1"),
            ):
                audio_server.main()
            (line,) = stderr.getvalue().splitlines()
            self.assertTrue(line.endswith(expected), line)

    def test_restarted_native_must_match_the_original_ready_event(self):
        for restarted in (
            AUDIO_READY,
            dataclasses.replace(AUDIO_READY, max_context_tokens=65536),
            dataclasses.replace(AUDIO_READY, max_concurrent_requests=1),
            dataclasses.replace(AUDIO_READY, kind=audio_wire.ModelKind.ASR),
        ):
            with self.subTest(restarted=restarted):
                factory = FakeFactory(audio=True)
                runtime = self.engine(factory)
                try:
                    self.assertEqual(runtime.readiness.max_context_tokens, 8192)
                    with mock.patch.object(runtime._crash_trace, "dump"):
                        factory.processes[0].kill()
                        self.wait_until(lambda: not runtime.ready)
                        factory.initial_output = native_peer.serialize_audio_event(
                            restarted
                        )
                        if restarted is AUDIO_READY:
                            self.assertTrue(runtime.wait_ready(1))
                        else:
                            # The difference would recur on every relaunch.
                            for _ in range(2):
                                with self.assertRaisesRegex(
                                    engine_runtime.EngineUnhealthy,
                                    "restart the Splash server",
                                ):
                                    runtime.wait_ready(1)
                            self.assertFalse(runtime.ready)
                    self.assertEqual(runtime.pending_count, 0)
                    self.assertEqual(len(factory.processes), 2)
                    self.assertEqual(
                        factory.processes[1].stdin.messages(audio_wire.SpeechRequest),
                        [],
                    )
                finally:
                    runtime.close()


if __name__ == "__main__":
    unittest.main()

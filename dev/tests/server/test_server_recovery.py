import concurrent.futures
import dataclasses
import http.client
import io
import json
import queue
import signal
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from dev.tests import native_peer
from dev.tests.server_fixtures import (
    READY,
    FakeFactory,
    FakeRuntime,
    FakeTokenizer,
    HarnessTestCase,
    Plan,
    chat_body,
    console_written,
    main_args,
    make_frontend,
    make_job,
    unprepared_hold,
)
from server import backend as backend_api
from server import protocol as wire
from server import runtime as engine_runtime
from server import server as api
from server import supervision


def answer_status(process, message):
    if isinstance(message, wire.StatusRequestFrame):
        process.send(native_peer.status_event(message.correlation_id))


class RecoveringRuntime(FakeRuntime):
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
            frozenset(),
            lambda: None,
        )

    def wait_ready(self, timeout=None):
        self.startup_calls += 1
        outcome = self.outcomes.get(timeout=3)
        # The startup has ended; no engine is being started.
        self.spawned_at = None
        if outcome is not None:
            raise outcome
        self.ready = True
        return True

    def close(self):
        self.outcomes.put(engine_runtime.RuntimeClosed("runtime is closed"))
        super().close()


class ServerRecoveryTests(HarnessTestCase):
    @staticmethod
    def wait_until(predicate, timeout=1):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError("test condition did not become true")
            time.sleep(0.005)

    def backend(self, runtime, on_exit=None):
        backend = backend_api.NativeBackend(
            runtime, FakeTokenizer(), lambda _record: None, on_exit=on_exit
        )
        self.addCleanup(backend.close)
        return backend

    def relaunched(self, runtime, backend):
        runtime.outcomes.put(None)
        self.wait_until(lambda: backend.recovery is None and runtime.ready)

    @staticmethod
    def body():
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 4,
            "reasoning_effort": "none",
        }

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

    def test_the_sixth_fast_failure_in_a_row_stops(self):
        """Short-lived failures in a row double the backoff, and the sixth,
        whose wait would have reached the cap, stops the relaunches: requests
        get 500 engine_failed with its reason, which /status reports, and the
        backend tells the server, which exits (they went on every 60 s for
        good, each loading the weights again)."""
        runtime = RecoveringRuntime()
        harness = self.harness(runtime)
        backend = harness.backend
        backend.on_stopped = stopped = mock.Mock()
        clock = [100.0]
        with (
            mock.patch.object(supervision, "BACKOFF_BASE_SECONDS", 0.01),
            mock.patch.object(
                backend_api, "time", SimpleNamespace(monotonic=lambda: clock[0])
            ),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            for delay in (0.0, 0.01, 0.02, 0.04, 0.08):
                runtime.fail(backend, served=1.0)
                self.assertAlmostEqual(backend.restart_due - clock[0], delay)
                refusal = backend.refusal()
                self.assertEqual(
                    (refusal.status, refusal.code), (503, "engine_recovering")
                )
                self.relaunched(runtime, backend)
            self.assertIsNone(backend.stop_reason)
            stopped.assert_not_called()
            runtime.fail(backend, served=1.0)
        self.assertEqual(runtime.startup_calls, 5)
        reason = backend.stop_reason
        self.assertEqual(
            reason,
            "Splash stopped restarting the engine: it failed 6 times in a row, "
            "each within 60 s of starting (native protocol reached EOF). "
            "Set SPLASH_CRASH_TRACE=1 to record a crash trace. "
            "Start the Splash server again after fixing the cause.",
        )
        self.assertEqual(
            console.call_args_list[-1].args[0], f"Engine stopped · {reason}"
        )
        stopped.assert_called_once_with(reason)
        refusal = backend.refusal()
        self.assertEqual(
            (refusal.status, refusal.code, refusal.message),
            (500, "engine_failed", reason),
        )
        transport = json.loads(harness.request("GET", "/status")[2])["transport"]
        self.assertTrue(transport["stopped"])
        self.assertEqual(transport["error"], reason)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", self.body()
        )
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(payload)["error"]["code"], "engine_failed")

    def test_recovery_refusals_ask_for_the_time_a_relaunch_takes(self):
        # Until the next bound on the relaunched engine's start: as long after
        # its spawn as the engine before took to start, then the start retry
        # window more; before the spawn as long after the later of now and
        # the relaunch's due time, rounded up: at least 1 s, at most 59 s, the
        # longest every SDK follows.
        runtime = RecoveringRuntime()
        backend = self.backend(runtime)
        clock = [100.0]
        with (
            mock.patch.object(
                backend_api, "time", SimpleNamespace(monotonic=lambda: clock[0])
            ),
            mock.patch.object(backend_api, "print_status"),
        ):
            runtime.last_start_seconds = 20.5
            runtime.fail(backend)
            self.assertEqual(backend.refusal().retry_after, 21)
            # The engine before ignores SIGTERM and is still being stopped
            # past a start's time: the whole start is still to come.
            clock[0] += 25.0
            self.assertEqual(backend.refusal().retry_after, 21)
            # Ten seconds after the spawn, the rest of the start is left.
            runtime.spawned_at = clock[0]
            clock[0] += 10.0
            self.assertEqual(backend.refusal().retry_after, 11)
            # A spawned engine slower than the one before waits for memory,
            # up to the start retry window more.
            clock[0] += 20.0
            self.assertEqual(backend.refusal().retry_after, 21)
            self.relaunched(runtime, backend)
            # The second relaunch in a row is due BACKOFF_BASE_SECONDS on.
            runtime.fail(backend)
            for start, retry_after in ((20.5, 26), (0.2, 6), (100.0, 59)):
                runtime.last_start_seconds = start
                self.assertEqual(backend.refusal().retry_after, retry_after)

    def test_retry_after_past_the_expected_ready_covers_the_start_retry_window(self):
        """A relaunch past its usual start waits for memory the host or Metal
        refused, up to the start retry window (it said 1 s, and clients spent
        their retries); past that, up to the start timeout."""
        runtime = RecoveringRuntime()
        backend = self.backend(runtime)
        clock = [0.0]
        with (
            mock.patch.object(
                backend_api, "time", SimpleNamespace(monotonic=lambda: clock[0])
            ),
            mock.patch.object(backend_api, "print_status"),
        ):
            runtime.last_start_seconds = 8.0
            runtime.fail(backend)
            runtime.spawned_at = 0.0
            clock[0] = 9.0
            self.assertEqual(backend.refusal().retry_after, 29)
            clock[0] = 40.0
            self.assertEqual(backend.refusal().retry_after, 59)

    def test_full_window_of_service_resets_the_backoff(self):
        runtime = RecoveringRuntime()
        backend = self.backend(runtime)
        with mock.patch.object(backend_api, "print_status"):
            for _ in range(5):
                runtime.fail(backend, served=61.0)
                self.assertEqual(backend.policy.streak, 1)
                self.assertLessEqual(backend.restart_due, time.monotonic())
                self.relaunched(runtime, backend)
        self.assertIsNone(backend.stop_reason)
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

    def test_relaunch_refused_for_host_memory_keeps_relaunching(self):
        """A relaunch the host refuses memory backs off and relaunches again:
        its retry runs no command (failed relaunches stopped at the third)."""
        runtime = RecoveringRuntime()
        host = wire.StartFailedEvent(
            wire.StartFailureKind.HOST_MEMORY,
            wire.ShortfallBound.DEVICE_MEMORY,
            0,
            0,
            0,
            b"the host has too little free memory to start",
        )
        for _ in range(3):
            runtime.outcomes.put(engine_runtime.EngineStartFailed(host))
        backend = self.backend(runtime)
        with (
            mock.patch.object(supervision, "BACKOFF_BASE_SECONDS", 0.01),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            runtime.fail(backend, served=1.0)
            self.wait_until(lambda: runtime.startup_calls == 4)
            self.relaunched(runtime, backend)
        self.assertIsNone(backend.stop_reason)
        self.assertEqual(
            [call.args[0] for call in console.call_args_list],
            [
                "Engine failed · native protocol reached EOF",
                *[
                    "Engine restart failed · the host has too little free memory "
                    "to start"
                ]
                * 3,
                "Engine restarted",
            ],
        )
        self.assertIsNone(backend.refusal())

    def test_engine_failure_with_exit_calls_on_exit_and_owes_no_relaunch(self):
        runtime = RecoveringRuntime()
        exits = []
        backend = self.backend(runtime, on_exit=lambda: exits.append(True))
        with mock.patch.object(backend_api, "print_status") as console:
            runtime.fail(backend)
            self.assertEqual(exits, [True])
            self.assertIsNone(backend.restart_due)
            refusal = backend.refusal()
        self.assertEqual((refusal.status, refusal.code), (503, "engine_recovering"))
        self.assertIn("native protocol reached EOF", refusal.message)
        self.assertIn(
            "Exiting · --engine-failure exit",
            [call.args[0] for call in console.call_args_list][-1],
        )
        self.assertEqual(exits, [True])
        time.sleep(0.05)
        self.assertEqual(runtime.startup_calls, 0)

    def test_engine_restart_failure_with_exit_follows_the_normal_path(self):
        runtime = RecoveringRuntime()
        exits = []
        backend = self.backend(runtime, on_exit=lambda: exits.append(True))
        with mock.patch.object(backend_api, "print_status"):
            backend._failed(
                "Engine restart failed",
                supervision.EngineFailure("engine_unhealthy", "bad start", 0.0),
            )
        self.assertEqual(exits, [])
        self.assertIsNotNone(backend.restart_due)
        self.relaunched(runtime, backend)
        self.assertEqual(runtime.startup_calls, 1)

    def test_engine_failure_without_exit_still_schedules_a_relaunch(self):
        runtime = RecoveringRuntime()
        backend = self.backend(runtime)
        with mock.patch.object(backend_api, "print_status"):
            runtime.fail(backend)
        self.assertIsNotNone(backend.restart_due)
        self.relaunched(runtime, backend)
        self.assertEqual(runtime.startup_calls, 1)

    def stopped_by(self, outcome):
        """The stop reason of a backend whose first relaunch ends with
        `outcome`, and its console lines; the backend tells the server once."""
        runtime = RecoveringRuntime()
        runtime.outcomes.put(outcome)
        backend = self.backend(runtime)
        backend.on_stopped = stopped = mock.Mock()
        with mock.patch.object(backend_api, "print_status") as console:
            runtime.fail(backend, served=1.0)
            self.wait_until(lambda: stopped.called)
        stopped.assert_called_once_with(backend.stop_reason)
        self.assertEqual(runtime.startup_calls, 1)
        refusal = backend.refusal()
        self.assertEqual(
            (refusal.status, refusal.code, refusal.message),
            (500, "engine_failed", backend.stop_reason),
        )
        transport = backend.status()["transport"]
        self.assertTrue(transport["stopped"])
        self.assertFalse(transport["recovering"])
        return backend.stop_reason, [call.args[0] for call in console.call_args_list]

    def test_command_memory_start_failure_on_relaunch_stops(self):
        """A relaunch whose start ran out of GPU memory another process holds
        is not retried: a retry would take that memory from it (it was
        retried twice more, then stopped)."""
        reason, lines = self.stopped_by(
            engine_runtime.EngineStartFailed(
                wire.StartFailedEvent(
                    wire.StartFailureKind.COMMAND_MEMORY,
                    wire.ShortfallBound.DEVICE_MEMORY,
                    0,
                    0,
                    0,
                    b"a start command ran out of GPU memory",
                )
            )
        )
        self.assertEqual(
            reason,
            "the engine's start ran out of GPU memory another process holds "
            "(another Splash server?); Splash stopped restarting it. Quit that "
            "process, then start the Splash server again.",
        )
        self.assertEqual(lines[-1], f"Engine stopped · {reason}")

    def test_configuration_start_failure_on_relaunch_stops(self):
        reason, _ = self.stopped_by(
            engine_runtime.EngineStartFailed(
                wire.StartFailedEvent(
                    wire.StartFailureKind.CONFIGURATION,
                    wire.ShortfallBound.DEVICE_MEMORY,
                    0,
                    0,
                    0,
                    b"--kv-format bf16 is refused",
                )
            )
        )
        self.assertTrue(
            reason.startswith(
                "Splash stopped restarting the engine: --kv-format bf16 is refused. "
            )
        )
        self.assertRegex(reason, "Crash trace|SPLASH_CRASH_TRACE")

    def test_model_start_failure_on_relaunch_stops(self):
        """A relaunch that finds the model's files it cannot serve is not
        retried: every relaunch reads the same files (it was retried every
        60 s for good)."""
        reason, lines = self.stopped_by(
            engine_runtime.EngineStartFailed(
                wire.StartFailedEvent(
                    wire.StartFailureKind.MODEL,
                    wire.ShortfallBound.DEVICE_MEMORY,
                    0,
                    0,
                    0,
                    b"MODEL_DIRECTORY must name an existing directory",
                )
            )
        )
        self.assertEqual(
            reason,
            "the model's files can no longer be served (MODEL_DIRECTORY must name "
            "an existing directory); Splash stopped restarting the engine. Restore "
            "or reinstall the model, then start the Splash server again.",
        )
        self.assertEqual(lines[-1], f"Engine stopped · {reason}")

    def test_missing_executable_on_relaunch_stops(self):
        reason, _ = self.stopped_by(
            engine_runtime.EngineLaunchFailed("native engine executable is missing")
        )
        self.assertIn("native engine executable is missing", reason)

    def test_relaunch_serving_less_context_is_adopted(self):
        """A relaunch whose plan holds less context serves that: the context
        /v1/models and /status report and requests are checked against
        follows it, and the server keeps serving (it stopped restarting at
        once)."""
        factory = FakeFactory(handler=answer_status)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        backend = self.backend(runtime)
        app = make_frontend(
            FakeTokenizer(), backend, "test-model", 128, 1, 2, vision=False
        )
        self.assertEqual(app.max_context, READY.max_context_tokens)
        smaller = dataclasses.replace(READY, max_context_tokens=65_536)
        factory.initial_output = native_peer.serialize_event(smaller)
        with (
            mock.patch.object(runtime._crash_trace, "dump"),
            mock.patch.object(backend_api, "print_status"),
        ):
            factory.processes[0].kill()
            self.wait_until(
                lambda: (
                    len(factory.processes) == 2
                    and runtime.ready
                    and backend.recovery is None
                ),
                2,
            )
        self.assertIsNone(backend.stop_reason)
        self.assertIsNone(backend.refusal())
        self.assertEqual(app.max_context, 65_536)
        self.assertEqual(app.status()["capabilities"], {"max_context_tokens": 65_536})

    def test_relaunch_asks_for_the_first_engines_context(self):
        # A relaunch is asked for the context the first engine served, with
        # --max-context as without, and serves what its plan holds of it. It
        # keeps --max-context, so that it is planned as the first engine was.
        for max_context in (None, 131072):
            args = main_args(max_context=max_context)
            first = tuple(api._native_command(args, args.model_root))
            relaunch = (*first, "--relaunch-context", str(READY.max_context_tokens))
            for relaunched in (
                READY,
                dataclasses.replace(READY, max_context_tokens=65_536),
            ):
                with self.subTest(max_context=max_context, relaunched=relaunched):
                    factory = FakeFactory(handler=answer_status)
                    commands = []

                    def spawn(command, factory=factory, commands=commands, **_):
                        commands.append(command)
                        return factory()

                    runtime = engine_runtime.MultiplexedRuntime(
                        first,
                        relaunch_command=lambda ready, args=args: api._native_command(
                            args, args.model_root, ready.max_context_tokens
                        ),
                    )
                    trace = mock.Mock(last_dump=None)
                    with (
                        mock.patch.object(
                            engine_runtime.subprocess, "Popen", side_effect=spawn
                        ),
                        mock.patch.object(runtime, "_crash_trace", trace),
                        mock.patch.object(backend_api, "print_status"),
                    ):
                        self.assertTrue(runtime.wait_ready(1))
                        backend = self.backend(runtime)
                        factory.initial_output = native_peer.serialize_event(relaunched)
                        factory.processes[0].kill()
                        self.wait_until(
                            lambda: (
                                len(factory.processes) == 2
                                and runtime.ready
                                and backend.recovery is None
                            ),
                            2,
                        )
                    self.assertEqual(commands, [first, relaunch])
                    # A crash trace of the relaunch replays its own command.
                    self.assertEqual(
                        [
                            call.args[2]
                            for call in trace.start_generation.call_args_list
                        ],
                        [first, relaunch],
                    )
                    self.assertIsNone(backend.stop_reason)
                    self.assertIsNone(backend.refusal())
                    self.assertEqual(runtime.capabilities, relaunched)

    def test_stop_after_a_failed_relaunch_names_the_last_crash_trace(self):
        factory = FakeFactory()
        configuration = wire.StartFailedEvent(
            wire.StartFailureKind.CONFIGURATION,
            wire.ShortfallBound.DEVICE_MEMORY,
            0,
            0,
            0,
            b"refused",
        )

        def launch():
            process = factory()
            if len(factory.processes) > 1:
                # The relaunch refuses its configuration.
                process.send(configuration)
            return process

        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=launch, eager_start=True
        )
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
            mock.patch.object(backend_api, "print_status"),
        ):
            factory.processes[0].kill()
            self.wait_until(lambda: backend.stop_reason is not None, 2)
        self.assertEqual(len(factory.processes), 2)
        self.assertIn("Crash trace: /traces/g2.json.", backend.stop_reason)

    def test_failure_right_after_a_relaunch_ready_is_counted_once(self):
        factory = FakeFactory()

        def launch():
            process = factory()
            if len(factory.processes) == 2:
                # The relaunched engine fails as soon as it is Ready.
                process.close_stdout()
            return process

        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=launch, eager_start=True
        )
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
        self.assertEqual(backend.policy.streak, 2)
        self.assertIsNone(backend.stop_reason)

    def quarantine_server(self, outcome):
        """A backend over engines that answer each request with
        `outcome(process, frame)`, relaunched without backoff to speak of."""
        factory = FakeFactory(
            handler=lambda process, message: (
                answer_status(process, message),
                isinstance(message, wire.RequestFrame) and outcome(process, message),
            )
        )
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        backend = self.backend(runtime)
        for patch in (
            mock.patch.object(supervision, "BACKOFF_BASE_SECONDS", 0.01),
            mock.patch.object(runtime._crash_trace, "dump"),
            mock.patch.object(backend_api, "print_status"),
        ):
            self.enterContext(patch)
        return factory, runtime, backend

    def outcome_of(self, backend, runtime, prompt):
        """What a request for `prompt` gets once the engine serves."""
        self.wait_until(lambda: runtime.ready and backend.recovery is None, 2)
        job = make_job()
        job.prompt_tokens = list(prompt)
        backend.submit(job)
        kind, value = job.events.get(timeout=2)
        while kind not in ("done", "error"):
            kind, value = job.events.get(timeout=2)
        return kind, value

    @staticmethod
    def serve(process, frame):
        process.send(wire.StartEvent(frame.request_id, 0, 0))
        process.send(wire.TokensEvent(frame.request_id, 0, (4,)))
        process.send(
            wire.DoneEvent(
                frame.request_id,
                wire.FinishReason.STOP,
                len(frame.prompt_tokens),
                1,
                100,
                200,
                350,
                (),
            )
        )

    @staticmethod
    def crash(code):
        def fail(process, frame):
            process.send(wire.StartEvent(frame.request_id, 0, 0))
            process.send(
                wire.ErrorEvent(
                    wire.FailureClass.ENGINE_UNHEALTHY, 0, False, code, b"boom"
                )
            )

        return fail

    def test_a_request_started_at_two_engine_failures_in_a_row_is_refused(self):
        """A request in flight at two engine failures in a row is refused
        for an hour, without reaching the engine; another, started only
        after the first relaunch, is served."""
        poison = (1, 2, 3)
        crash = self.crash(b"engine_execution_failed")
        factory, runtime, backend = self.quarantine_server(
            lambda process, frame: (
                crash if frame.prompt_tokens == poison else self.serve
            )(process, frame)
        )
        for _ in range(2):
            kind, error = self.outcome_of(backend, runtime, poison)
            self.assertEqual((kind, error.code), ("error", "runtime_unavailable"))
        self.assertEqual(self.outcome_of(backend, runtime, (4, 5))[0], "done")
        sent = len(factory.processes[-1].stdin.messages(wire.RequestFrame))
        kind, error = self.outcome_of(backend, runtime, poison)
        self.assertEqual(
            (kind, error.status, error.code), ("error", 400, "request_quarantined")
        )
        self.assertIn("two engine failures in a row", error.message)
        self.assertEqual(
            len(factory.processes[-1].stdin.messages(wire.RequestFrame)), sent
        )
        self.assertEqual(self.outcome_of(backend, runtime, (4, 5))[0], "done")

    def test_a_request_retried_through_failures_is_refused_before_they_stop(self):
        """A request whose every run fails the engine, however often its client
        retries it, is refused at its second failure, long before the failures
        in a row would stop the relaunches."""
        poison = (1, 2, 3)
        crash = self.crash(b"engine_execution_failed")
        factory, runtime, backend = self.quarantine_server(
            lambda process, frame: (
                crash if frame.prompt_tokens == poison else self.serve
            )(process, frame)
        )
        outcomes = [
            self.outcome_of(backend, runtime, poison)[1].code
            for _ in range(supervision.CRASH_LOOP_FAILURES + 1)
        ]
        self.assertEqual(
            outcomes,
            ["runtime_unavailable"] * 2
            + ["request_quarantined"] * (supervision.CRASH_LOOP_FAILURES - 1),
        )
        self.assertIsNone(backend.stop_reason)
        self.assertEqual(len(factory.processes), 3)
        self.assertEqual(self.outcome_of(backend, runtime, (4, 5))[0], "done")

    def test_a_request_started_at_one_failure_is_served_after_the_relaunch(self):
        poison = (1, 2, 3)
        crashes = [self.crash(b"engine_execution_failed")]
        factory, runtime, backend = self.quarantine_server(
            lambda process, frame: (
                crashes.pop()
                if frame.prompt_tokens == poison and crashes
                else self.serve
            )(process, frame)
        )
        self.assertEqual(self.outcome_of(backend, runtime, poison)[0], "error")
        self.assertEqual(self.outcome_of(backend, runtime, poison)[0], "done")
        self.assertEqual(len(factory.processes), 2)

    def test_requests_started_at_two_kills_are_not_quarantined(self):
        """A kill (jetsam's, the user's, the server's own fence) says nothing
        about the requests in flight."""
        poison = (1, 2, 3)

        def start_then_die(process, frame):
            process.send(wire.StartEvent(frame.request_id, 0, 0))
            process.kill()

        factory, runtime, backend = self.quarantine_server(
            lambda process, frame: (
                start_then_die if len(factory.processes) <= 2 else self.serve
            )(process, frame)
        )
        for _ in range(2):
            kind, error = self.outcome_of(backend, runtime, poison)
            self.assertEqual((kind, error.code), ("error", "runtime_unavailable"))
        self.assertEqual(self.outcome_of(backend, runtime, poison)[0], "done")

    def test_a_request_started_at_two_fault_deaths_in_a_row_is_refused(self):
        """An engine that dies by a fault signal ends its output before its
        exit status is there to read: its failure counts once it has
        exited."""
        poison = (1, 2, 3)

        def segfault(process, frame):
            process.send(wire.StartEvent(frame.request_id, 0, 0))
            process.crash(signal.SIGSEGV)

        factory, runtime, backend = self.quarantine_server(
            lambda process, frame: (
                segfault if frame.prompt_tokens == poison else self.serve
            )(process, frame)
        )
        for _ in range(2):
            kind, error = self.outcome_of(backend, runtime, poison)
            self.assertEqual((kind, error.code), ("error", "runtime_unavailable"))
        kind, error = self.outcome_of(backend, runtime, poison)
        self.assertEqual(
            (kind, error.status, error.code), ("error", 400, "request_quarantined")
        )
        self.assertEqual(self.outcome_of(backend, runtime, (4, 5))[0], "done")

    def test_requests_started_at_two_kills_mid_frame_are_not_quarantined(self):
        """A kill that cuts a frame short breaks the protocol, and still says
        nothing about the requests in flight: the signal decides."""
        poison = (1, 2, 3)

        def killed_mid_frame(process, frame):
            process.send(wire.StartEvent(frame.request_id, 0, 0))
            done = native_peer.serialize_event(
                wire.DoneEvent(
                    frame.request_id, wire.FinishReason.STOP, 3, 1, 100, 200, 350, ()
                )
            )
            process.stdout.feed(done[: len(done) // 2])
            process.kill()

        factory, runtime, backend = self.quarantine_server(
            lambda process, frame: (
                killed_mid_frame if len(factory.processes) <= 2 else self.serve
            )(process, frame)
        )
        for _ in range(2):
            kind, error = self.outcome_of(backend, runtime, poison)
            self.assertEqual((kind, error.code), ("error", "runtime_unavailable"))
        self.assertEqual(self.outcome_of(backend, runtime, poison)[0], "done")
        # Both failures were the frame cut short.
        failures = [
            call.args[0]
            for call in backend_api.print_status.call_args_list
            if call.args[0].startswith("Engine failed")
        ]
        self.assertEqual(len(failures), 2)
        self.assertTrue(
            all("stream ended after" in line for line in failures), failures
        )

    def test_a_queued_request_is_never_quarantined(self):
        """A request the engine had not started was in no command."""
        poison, queued = (1, 2, 3), (7, 8)
        crash = self.crash(b"engine_execution_failed")
        factory, runtime, backend = self.quarantine_server(
            lambda process, frame: (
                frame.prompt_tokens == poison and crash(process, frame)
            )
        )
        for _ in range(2):
            self.wait_until(lambda: runtime.ready and backend.recovery is None, 2)
            waiting = make_job(201)
            waiting.prompt_tokens = list(queued)
            backend.submit(waiting)
            factory.processes[-1].stdin.wait_for(wire.RequestFrame)
            self.assertEqual(self.outcome_of(backend, runtime, poison)[0], "error")
            self.assertEqual(waiting.events.get(timeout=2)[0], "error")
        fingerprint = backend._generation_request(waiting).fingerprint
        self.assertFalse(backend.quarantine.refuses(fingerprint))

    def test_metal_out_of_memory_failures_do_not_count(self):
        """A command short of GPU memory says nothing about the requests it
        ran."""
        poison = (1, 2, 3)
        crash = self.crash(b"metal_out_of_memory")
        factory, runtime, backend = self.quarantine_server(
            lambda process, frame: (
                crash if len(factory.processes) <= 2 else self.serve
            )(process, frame)
        )
        for _ in range(2):
            self.assertEqual(self.outcome_of(backend, runtime, poison)[0], "error")
        self.assertEqual(self.outcome_of(backend, runtime, poison)[0], "done")

    def test_generation_recovery_rejects_before_body_or_ingress_and_advertises_retry(
        self,
    ):
        runtime = RecoveringRuntime()
        # A relaunch is due now and takes as long as the engine before's start.
        runtime.last_start_seconds = 20.0
        harness = self.harness(runtime, queue_size=1)
        runtime.fail(harness.backend)
        with mock.patch.object(harness.app, "prepare") as prepare:
            for path, status in (
                ("/v1/messages", 503),
                ("/v1/chat/completions", 503),
                ("/v1/responses", 503),
                # TypeSafe's overload status, which invites a retry as well.
                ("/v1/systemone", 529),
            ):
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
                        self.assertEqual(response.status, status)
                        self.assertEqual(response.getheader("Retry-After"), "20")
                        self.assertEqual(
                            json.loads(response.read())["error"]["type"],
                            "overloaded_error"
                            if path.startswith("/v1/messages")
                            else "server_error",
                        )
                    finally:
                        connection.close()
                    self.assertEqual(harness.server.requests.stats()["active"], 0)
                    self.assertEqual(runtime.pending_count, 0)
            prepare.assert_not_called()
        self.wait_until(lambda: runtime.startup_calls == 1)
        self.assertEqual(runtime.requests, [])
        for path, body in (
            ("/v1/messages/count_tokens", self.body()),
            ("/v1/chat/completions/count_tokens", chat_body()),
        ):
            with self.subTest(path=path):
                self.assertEqual(harness.request("POST", path, body)[0], 200)
        self.assertEqual(harness.request("GET", "/v1/models")[0], 200)

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

    def test_token_count_remains_available_while_generation_slots_are_full(self):
        plan = Plan([[4]], block=True)
        runtime = FakeRuntime(plan)
        harness = self.harness(runtime, queue_size=1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            generation = pool.submit(
                harness.request, "POST", "/v1/chat/completions", self.body()
            )
            try:
                self.assertTrue(plan.started.wait(1))
                self.assertEqual(harness.server.requests.stats()["active"], 1)
                for path, body in (
                    ("/v1/messages/count_tokens?beta=true", self.body()),
                    ("/v1/chat/completions/count_tokens", chat_body()),
                ):
                    with self.subTest(path=path):
                        status, _, payload = harness.request("POST", path, body)
                        self.assertEqual(status, 200)
                        self.assertEqual(json.loads(payload), {"input_tokens": 2})
                        self.assertEqual(len(runtime.requests), 1)
                        self.wait_until(
                            lambda: harness.server.token_counts.stats()["active"] == 0
                        )
                        self.assertEqual(harness.server.requests.stats()["active"], 1)
            finally:
                plan.release.set()
            self.assertEqual(generation.result(1)[0], 200)
        self.wait_until(lambda: harness.server.requests.stats()["active"] == 0)

    def test_systemone_overload_is_529_and_invites_a_retry(self):
        plan = Plan([[4]], block=True)
        harness = self.harness(FakeRuntime(plan), queue_size=1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            generation = pool.submit(
                harness.request, "POST", "/v1/chat/completions", self.body()
            )
            try:
                self.assertTrue(plan.started.wait(1))
                connection = http.client.HTTPConnection(
                    *harness.server.server_address, timeout=3
                )
                self.addCleanup(connection.close)
                connection.request(
                    "POST",
                    "/v1/systemone",
                    json.dumps(
                        {
                            "model": "test-model",
                            "state": "evidence",
                            "questions": {"supported": {"type": "noul"}},
                        }
                    ),
                    {"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 529)
                self.assertEqual(response.getheader("Retry-After"), "1")
                self.assertEqual(
                    json.loads(response.read())["error"]["code"],
                    "frontend_overloaded",
                )
            finally:
                plan.release.set()
            self.assertEqual(generation.result(1)[0], 200)

    def test_token_count_has_its_own_bounded_ingress_and_releases_on_disconnect(self):
        harness = self.harness(FakeRuntime(), queue_size=1)
        upload = http.client.HTTPConnection(*harness.server.server_address, timeout=1)
        try:
            upload.putrequest("POST", "/v1/messages/count_tokens")
            upload.putheader("Content-Type", "application/json")
            upload.putheader("Content-Length", "100")
            upload.endheaders()
            self.wait_until(lambda: harness.server.token_counts.stats()["active"] == 1)
            response = harness.request("POST", "/v1/messages/count_tokens", self.body())
            self.assertEqual(response[0], 503)
            self.assertEqual(
                json.loads(response[2])["error"]["type"], "overloaded_error"
            )
            self.assertEqual(
                harness.request("POST", "/v1/chat/completions", self.body())[0], 200
            )
            self.assertEqual(harness.request("GET", "/v1/models")[0], 200)
            snapshot = json.loads(harness.request("GET", "/status")[2])
            self.assertEqual(
                snapshot["http"]["token_counts"], {"active": 1, "capacity": 1}
            )
        finally:
            upload.close()
        self.wait_until(lambda: harness.server.token_counts.stats()["active"] == 0)
        self.assertEqual(
            harness.request("POST", "/v1/messages/count_tokens", self.body())[0], 200
        )

    def test_background_recovery_and_waiters_share_the_native_startup(self):
        factory = FakeFactory(handler=answer_status)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
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
                factory.processes[1].send(READY)
                for waiter in waiters:
                    self.assertTrue(waiter.result(1))
            self.wait_until(lambda: backend.recovery is None)
        self.assertTrue(backend.status()["ready"])
        self.assertEqual(runtime.pending_count, 0)
        self.assertEqual(factory.processes[1].stdin.messages(wire.RequestFrame), [])

    def test_idle_engine_death_restarts_before_traffic_arrives(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        backend = self.backend(runtime)
        factory.processes[0].kill()
        self.wait_until(lambda: len(factory.processes) == 2 and runtime.ready, 2)
        self.assertEqual(runtime.restart_count, 1)
        self.assertIsNone(backend.refusal())

    def test_idle_engine_that_stops_answering_status_is_restarted(self):
        def answer_after_the_first(process, message):
            if process is not factory.processes[0]:
                answer_status(process, message)

        factory = FakeFactory(handler=answer_after_the_first)
        with mock.patch.object(
            supervision,
            "THRESHOLDS",
            dataclasses.replace(supervision.THRESHOLDS, status_answer=0.05),
        ):
            runtime = engine_runtime.MultiplexedRuntime(
                process_factory=factory, eager_start=True
            )
            backend = self.backend(runtime)
        with mock.patch.object(backend_api, "print_status") as console:
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
        factory = FakeFactory()

        def launch():
            if factory.processes:
                raise OSError(12, "Cannot allocate memory")
            return factory()

        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=launch, eager_start=True
        )
        backend = self.backend(runtime)
        with mock.patch.object(backend_api, "print_status") as console:
            factory.processes[0].kill()
            self.wait_until(lambda: console.call_count >= 2)
            transport = backend.status()["transport"]
        failed, restart = (call.args[0] for call in console.call_args_list[:2])
        self.assertEqual(failed, "Engine failed · native protocol reached EOF")
        self.assertEqual(
            restart,
            "Engine restart failed · could not launch native engine: "
            "[Errno 12] Cannot allocate memory",
        )
        # A spawn may fail for want of memory; it backs off and retries.
        self.assertTrue(transport["recovering"])
        self.assertIn("Cannot allocate memory", transport["error"])

    def test_a_missing_executable_ends_in_one_stop_line(self):
        factory = FakeFactory()

        def launch():
            if factory.processes:
                raise FileNotFoundError("splash")
            return factory()

        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=launch, eager_start=True
        )
        backend = self.backend(runtime)
        with mock.patch.object(backend_api, "print_status") as console:
            factory.processes[0].kill()
            self.wait_until(lambda: console.call_count >= 3)
        lines = [call.args[0] for call in console.call_args_list]
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "Engine failed · native protocol reached EOF")
        self.assertRegex(
            lines[1], "^Engine restart failed · native engine executable is missing"
        )
        self.assertEqual(lines[2], f"Engine stopped · {backend.stop_reason}")
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
                factory = FakeFactory()
                runtime = engine_runtime.MultiplexedRuntime(
                    process_factory=factory, eager_start=True
                )
                backend = self.backend(runtime)
                job = make_job()
                with (
                    mock.patch.object(backend_api, "print_status") as console,
                    # A start as long as a large model's, its relaunch not
                    # spawned yet.
                    mock.patch.object(
                        engine_runtime.MultiplexedRuntime,
                        "last_start_seconds",
                        mock.PropertyMock(return_value=30.0),
                    ),
                    mock.patch.object(
                        engine_runtime.MultiplexedRuntime,
                        "spawned_at",
                        mock.PropertyMock(return_value=None),
                    ),
                ):
                    backend.submit(job)
                    factory.processes[0].stdin.wait_for(wire.RequestFrame)
                    fail(factory.processes[0])
                    kind, error = job.events.get(timeout=1)
                    self.wait_until(lambda: console.called)
                self.assertEqual(kind, "error")
                self.assertEqual(
                    (error.status, error.code, error.message, error.retry_after),
                    (
                        503,
                        "runtime_unavailable",
                        "the inference engine stopped unexpectedly and is "
                        "restarting; retry the request",
                        30,
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
            mock.patch.object(supervision, "BACKOFF_BASE_SECONDS", 0.01),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            runtime.fail(harness.backend)
            # The second relaunch waits for its outcome.
            self.wait_until(lambda: runtime.startup_calls == 2)
            console.assert_called_with(
                "Engine restart failed · GPU is gone", error=True
            )
            status, _, payload = harness.request(
                "POST", "/v1/chat/completions", self.body()
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

    def test_startup_failure_ends_with_one_error_line(self):
        # A broken stream, and an engine that says why its start failed.
        refusal = (
            "--max-context 1000000 exceeds the 790521 tokens the model and "
            "this Mac's memory allow; omit it or pass at most 790521"
        )
        runtime_type = engine_runtime.MultiplexedRuntime
        for initial_output, reason in (
            (
                b"not a frame".ljust(wire.FRAME_HEADER_BYTES, b"\0"),
                "protocol_fatal:bad_magic",
            ),
            (
                native_peer.serialize_event(
                    wire.StartFailedEvent(
                        wire.StartFailureKind.CONTEXT_BUDGET,
                        wire.ShortfallBound.DEVICE_MEMORY,
                        0,
                        0,
                        790_521,
                        refusal.encode(),
                    )
                ),
                refusal,
            ),
        ):
            with self.subTest(reason=reason):
                factory = FakeFactory(initial_output=initial_output)
                with (
                    mock.patch.object(api, "parse_args", return_value=main_args()),
                    unprepared_hold(),
                    mock.patch.object(api, "load_thinking_key", return_value=None),
                    mock.patch.object(
                        api.AutoTokenizer, "from_pretrained", return_value=object()
                    ),
                    mock.patch.object(api, "validate_tokenizer"),
                    mock.patch.object(api, "thinking_end_tokens", return_value=None),
                    mock.patch.object(api, "ChatTemplates"),
                    mock.patch.object(
                        api.engine_runtime,
                        "MultiplexedRuntime",
                        side_effect=lambda _command, **options: runtime_type(
                            process_factory=factory, **options
                        ),
                    ),
                    mock.patch.object(api, "FrontendServer"),
                    mock.patch.object(api.signal, "signal"),
                    mock.patch("sys.stdout", new_callable=io.StringIO),
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
                    self.assertRaisesRegex(SystemExit, "1"),
                ):
                    api.main()
                console_written()
                (line,) = stderr.getvalue().splitlines()
                self.assertIn(f"Error · {reason}", line)
                self.assertNotIn("reached EOF", line)
                self.assertIsNotNone(factory.processes[0].poll())

    def test_a_relaunch_is_served_as_it_announces(self):
        """The runtime serves what the latest engine announces; it compares
        nothing with the first engine's Ready."""
        for restarted in (
            READY,
            dataclasses.replace(READY, max_context_tokens=65536),
            dataclasses.replace(READY, vision=True),
        ):
            with self.subTest(restarted=restarted):
                factory = FakeFactory()
                runtime = engine_runtime.MultiplexedRuntime(
                    process_factory=factory, eager_start=True
                )
                try:
                    self.assertEqual(runtime.capabilities, READY)
                    with mock.patch.object(runtime._crash_trace, "dump"):
                        factory.processes[0].kill()
                        self.wait_until(lambda: not runtime.ready)
                        factory.initial_output = native_peer.serialize_event(restarted)
                        self.assertTrue(runtime.wait_ready(1))
                    self.assertEqual(runtime.capabilities, restarted)
                    self.assertEqual(len(factory.processes), 2)
                finally:
                    runtime.close()


if __name__ == "__main__":
    unittest.main()

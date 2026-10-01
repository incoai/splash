import concurrent.futures
import dataclasses
import http.client
import io
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from dev.tests.engine.test_native_backend import FakeTokenizer as NativeTokenizer
from dev.tests.engine.test_native_backend import make_job
from dev.tests.engine.test_runtime import READY_FEATURES, FakeFactory
from dev.tests.test_server import FakeRuntime, Harness, Plan, main_args
from install import launcher
from server import backend as backend_api
from server import protocol as wire
from server import runtime as engine_runtime
from server import server as api


class RecoveringRuntime(FakeRuntime):
    def __init__(self, failures=()):
        super().__init__()
        self.ready = False
        self.failures = list(failures)
        self.startup_calls = 0
        self.startup_entered = threading.Event()
        self.startup_release = threading.Event()

    def wait_ready(self):
        self.startup_calls += 1
        self.startup_entered.set()
        if not self.startup_release.wait(3):
            raise TimeoutError("test did not release startup")
        if self.failures:
            raise self.failures.pop(0)
        self.ready = True
        return True

    def status(self, timeout=5):
        if not self.ready:
            raise engine_runtime.EngineUnhealthy("native process is not ready")
        return super().status(timeout)

    def close(self):
        self.startup_release.set()
        super().close()


class ServerRecoveryTests(unittest.TestCase):
    @staticmethod
    def wait_until(predicate, timeout=1):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError("test condition did not become true")
            time.sleep(0.005)

    def harness(self, runtime, **kwargs):
        harness = Harness(runtime, **kwargs)

        def close():
            if isinstance(runtime, RecoveringRuntime):
                runtime.startup_release.set()
            harness.close()

        self.addCleanup(close)
        return harness

    @staticmethod
    def body():
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 4,
            "reasoning_effort": "none",
        }

    def test_ready_and_status_start_one_recovery_without_waiting_for_startup(self):
        for first_path in ("/ready", "/status"):
            with self.subTest(first_path=first_path):
                runtime = RecoveringRuntime()
                harness = self.harness(runtime)
                started = time.monotonic()
                response = harness.request("GET", first_path)
                self.assertLess(time.monotonic() - started, 0.5)
                self.assertEqual(response[0], 503 if first_path == "/ready" else 200)
                self.assertTrue(runtime.startup_entered.wait(0.5))
                self.assertFalse(runtime.startup_release.is_set())
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
                runtime.startup_release.set()
                self.wait_until(lambda: not harness.backend.status_refresh_inflight)
                self.assertEqual(harness.request("GET", "/ready")[0], 200)
                snapshot = json.loads(harness.request("GET", "/status")[2])
                self.assertFalse(snapshot["transport"]["recovering"])
                self.assertEqual(runtime.requests, [])

    def test_failed_recovery_is_backed_off_and_success_resets_backoff(self):
        runtime = RecoveringRuntime(
            [
                engine_runtime.EngineUnhealthy("first"),
                engine_runtime.EngineUnhealthy("second"),
            ]
        )
        runtime.startup_release.set()
        backend = backend_api.NativeBackend(runtime, NativeTokenizer())
        self.addCleanup(backend.close)
        clock = [100.0]
        with mock.patch.object(
            backend_api, "time", SimpleNamespace(monotonic=lambda: clock[0])
        ):
            self.assertFalse(backend.can_submit())
            self.wait_until(lambda: not backend.status_refresh_inflight)
            first_retry = backend.status_refresh_after
            self.assertGreater(first_retry, clock[0])
            for _ in range(10):
                backend.status()
                self.assertFalse(backend.can_submit())
            self.assertEqual(runtime.startup_calls, 1)
            clock[0] = first_retry
            self.assertFalse(backend.can_submit())
            self.wait_until(lambda: not backend.status_refresh_inflight)
            second_retry = backend.status_refresh_after
            self.assertGreater(second_retry - clock[0], first_retry - 100)
            self.assertEqual(runtime.startup_calls, 2)
            clock[0] = second_retry
            backend.can_submit()
            self.wait_until(lambda: not backend.status_refresh_inflight)
            self.assertTrue(backend.can_submit())
            self.assertEqual(runtime.startup_calls, 3)
            self.assertEqual(backend.status_refresh_failures, 0)
            self.assertEqual(backend.status_refresh_after, 0)
        backend.close()
        self.assertFalse(backend.can_submit())
        self.assertFalse(backend.status()["transport"]["recovering"])
        self.assertEqual(runtime.startup_calls, 3)

    def test_generation_recovery_rejects_before_body_or_ingress_and_advertises_retry(
        self,
    ):
        runtime = RecoveringRuntime()
        harness = self.harness(runtime, queue_size=1)
        with mock.patch.object(harness.app, "prepare") as prepare:
            for path in ("/v1/messages", "/v1/chat/completions", "/v1/responses"):
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
                            "overloaded_error"
                            if path.startswith("/v1/messages")
                            else "server_error",
                        )
                    finally:
                        connection.close()
                    self.assertEqual(harness.server.requests.stats()["active"], 0)
                    self.assertEqual(runtime.pending_count, 0)
            prepare.assert_not_called()
        self.assertEqual(runtime.startup_calls, 1)
        self.assertEqual(runtime.requests, [])
        self.assertEqual(
            harness.request("POST", "/v1/messages/count_tokens", self.body())[0], 200
        )
        self.assertEqual(harness.request("GET", "/v1/models")[0], 200)

    def test_fresh_control_status_clears_old_refresh_backoff(self):
        runtime = RecoveringRuntime()
        runtime.ready = True
        runtime.startup_release.set()
        backend = backend_api.NativeBackend(runtime, NativeTokenizer())
        self.addCleanup(backend.close)
        with mock.patch.object(
            backend_api, "time", SimpleNamespace(monotonic=lambda: 100.0)
        ):
            with mock.patch.object(runtime, "status", side_effect=TimeoutError):
                backend._ensure_background_status_refresh()
                self.wait_until(lambda: not backend.status_refresh_inflight)
            self.assertGreater(backend.status_refresh_after, 100.0)
            self.assertTrue(backend.status()["ready"])
            self.assertEqual(backend.status_refresh_failures, 0)
            self.assertEqual(backend.status_refresh_after, 0)
            runtime.ready = False
            self.assertFalse(backend.can_submit())
            self.wait_until(lambda: not backend.status_refresh_inflight)
            self.assertEqual(runtime.startup_calls, 1)
            self.assertTrue(backend.can_submit())

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
                status, _, payload = harness.request(
                    "POST", "/v1/messages/count_tokens?beta=true", self.body()
                )
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

    def test_launcher_accepts_a_recovering_instance_without_calling_it_ready(self):
        snapshot = {
            "ready": False,
            "transport": {"ready": False, "recovering": True, "status_stale": False},
        }
        with mock.patch.object(launcher, "_request_json", return_value=snapshot):
            self.assertIs(launcher._running_status(), snapshot)
            self.assertFalse(snapshot["ready"])
            snapshot["transport"]["recovering"] = False
            self.assertEqual(launcher._running_status(), snapshot)

    def test_background_recovery_and_waiters_share_the_native_startup(self):
        payload = json.dumps(
            {
                "schema_version": wire.STATUS_SCHEMA_VERSION,
                "ready": True,
                "memory_pressure": "normal",
                "metal": {"healthy": True},
            }
        ).encode()

        def respond(process, message):
            if isinstance(message, wire.StatusRequestFrame):
                process.send(
                    wire.StatusJsonEvent(
                        message.correlation_id, wire.STATUS_SCHEMA_VERSION, payload
                    )
                )

        factory = FakeFactory(handler=respond, initial_output=b"")
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=False
        )
        backend = backend_api.NativeBackend(runtime, NativeTokenizer())
        self.addCleanup(backend.close)
        self.assertFalse(backend.status()["ready"])
        self.wait_until(lambda: len(factory.processes) == 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            waiters = [pool.submit(runtime.wait_ready, 1) for _ in range(4)]
            for _ in range(10):
                self.assertFalse(backend.can_submit())
            self.assertEqual(len(factory.processes), 1)
            factory.processes[0].send(wire.ReadyEvent(1000, 4, 131072, READY_FEATURES))
            for waiter in waiters:
                self.assertTrue(waiter.result(1))
        self.wait_until(lambda: not backend.status_refresh_inflight)
        self.assertTrue(backend.status()["ready"])
        self.assertEqual(runtime.pending_count, 0)
        self.assertEqual(factory.processes[0].stdin.messages(wire.RequestFrame), [])

    def test_idle_engine_death_restarts_before_traffic_arrives(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        backend = backend_api.NativeBackend(runtime, NativeTokenizer())
        self.addCleanup(backend.close)
        factory.processes[0].kill()
        self.wait_until(lambda: len(factory.processes) == 2 and runtime.ready, 2)
        self.assertEqual(runtime.restart_count, 1)
        self.assertTrue(backend.can_submit())

    def test_engine_failure_and_failed_restart_are_reported(self):
        factory = FakeFactory()

        def launch():
            if factory.processes:
                raise FileNotFoundError("splash")
            return factory()

        runtime = engine_runtime.MultiplexedRuntime(process_factory=launch)
        backend = backend_api.NativeBackend(runtime, NativeTokenizer())
        self.addCleanup(backend.close)
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

    def test_engine_failure_under_a_request_asks_its_client_to_retry(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        backend = backend_api.NativeBackend(runtime, NativeTokenizer())
        self.addCleanup(backend.close)
        job = make_job()
        with mock.patch.object(backend_api, "print_status") as console:
            self.assertTrue(backend.submit(job))
            factory.processes[0].stdin.wait_for(wire.RequestFrame)
            factory.processes[0].close_stdout()
            kind, error = job.events.get(timeout=1)
            self.wait_until(lambda: console.called)
        self.assertEqual(kind, "error")
        self.assertEqual(
            (error.status, error.code, error.message),
            (
                503,
                "runtime_unavailable",
                "the inference engine stopped unexpectedly and is restarting; "
                "retry the request",
            ),
        )
        # The engine's own reason stays on the console.
        self.assertEqual(
            console.call_args_list[0].args[0],
            "Engine failed · native protocol reached EOF",
        )

    def test_recovery_refusals_carry_the_last_engine_failure(self):
        runtime = RecoveringRuntime([engine_runtime.EngineUnhealthy("GPU is gone")])
        runtime.startup_release.set()
        harness = self.harness(runtime)
        clock = [100.0]
        with (
            mock.patch.object(
                backend_api, "time", SimpleNamespace(monotonic=lambda: clock[0])
            ),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            self.assertFalse(harness.backend.can_submit())
            self.wait_until(lambda: not harness.backend.status_refresh_inflight)
            console.assert_called_once_with(
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
            clock[0] = harness.backend.status_refresh_after
            self.assertFalse(harness.backend.can_submit())
            self.wait_until(lambda: not harness.backend.status_refresh_inflight)
            self.assertTrue(harness.backend.can_submit())
            console.assert_called_with("Engine restarted")
            transport = json.loads(harness.request("GET", "/status")[2])["transport"]
            self.assertNotIn("error", transport)

    def test_startup_protocol_failure_ends_with_one_error_line(self):
        missing = READY_FEATURES & ~wire.ReadyFeature.MULTIPLEXING
        runtime_type = engine_runtime.MultiplexedRuntime
        for output, reason in (
            (
                wire.serialize_message(wire.ReadyEvent(1000, 4, 131072, missing)),
                "missing required native protocol features",
            ),
            (b"not a frame".ljust(wire.FRAME_HEADER_BYTES, b"\0"), "bad_magic"),
        ):
            with self.subTest(reason=reason):
                factory = FakeFactory(initial_output=output)
                with (
                    mock.patch.object(api, "parse_args", return_value=main_args()),
                    mock.patch.object(api, "load_thinking_key", return_value=None),
                    mock.patch.object(
                        api.AutoTokenizer, "from_pretrained", return_value=object()
                    ),
                    mock.patch.object(api, "validate_tokenizer"),
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
                (line,) = stderr.getvalue().splitlines()
                self.assertIn("Error · ", line)
                self.assertIn(reason, line)
                self.assertIsNotNone(factory.processes[0].poll())

    def test_restarted_native_must_match_the_original_ready_event(self):
        original = wire.ReadyEvent(1001, 4, 131072, READY_FEATURES)
        for restarted in (
            original,
            dataclasses.replace(original, max_context_tokens=65536),
            dataclasses.replace(original, max_concurrent_requests=1),
            dataclasses.replace(
                original, feature_bits=READY_FEATURES | wire.ReadyFeature.VISION
            ),
        ):
            with self.subTest(restarted=restarted):
                factory = FakeFactory()
                runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
                try:
                    self.assertEqual(runtime.readiness.max_context_tokens, 131072)
                    with mock.patch.object(runtime._crash_trace, "dump"):
                        factory.processes[0].kill()
                        self.wait_until(lambda: not runtime.ready)
                        factory.initial_output = wire.serialize_message(restarted)
                        if restarted is original:
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
                        factory.processes[1].stdin.messages(wire.RequestFrame), []
                    )
                finally:
                    runtime.close()


if __name__ == "__main__":
    unittest.main()

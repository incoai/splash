import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from dev.tests import native_peer
from dev.tests.audio_fixtures import FakeAudioRuntime, pcm
from dev.tests.server_fixtures import FOREVER, FakeFactory, interface_drift
from server import audio_backend, audio_runtime, runtime
from server import audio_protocol as audio_wire
from server import backend as backend_api
from server import protocol as wire
from server.runtime import STATUS_ANSWER_LIMIT_SECONDS

CHUNK = pcm([3] * 1920)


def speech_job(request_id=101, *, deadline=10.0, max_frames=0):
    """A speech request's job, as the frontend prepares one."""
    return audio_backend.AudioJob(
        request_id,
        audio_wire.SpeechRequest(
            0,
            (11, 12, 13, 14),
            language_id=2050,
            temperature=0.6,
            top_k=17,
            seed=0x123456789ABCDEF0,
            max_frames=max_frames,
        ),
        FOREVER if deadline is None else time.monotonic() + deadline,
    )


class DrivenRuntime(FakeAudioRuntime):
    """A runtime whose calls the test drives, and whose submit, status and
    close go as `mode` says."""

    def __init__(self, mode="normal", *, complete_on_close=True):
        super().__init__(manual=True)
        self.mode = mode
        self.complete_on_close = complete_on_close
        self.pending_limit = 8
        self.restart_count = 3
        self.status_calls = 0
        self.submit_attempts = 0
        self.submit_entered = threading.Event()
        self.release_submit = threading.Event()

    def submit(self, request, *, on_event=None, on_complete=None):
        self.submit_attempts += 1
        if self.mode == "pending_limit":
            raise runtime.PendingLimitExceeded("full")
        if self.mode == "raise_before_call":
            raise runtime.EngineUnhealthy("native process is not ready")
        if self.mode == "protocol_failure_before_call":
            raise runtime.ProtocolFatal("native sent more than one ReadyEvent")
        call = super().submit(request, on_event=on_event, on_complete=on_complete)
        if self.mode == "block_after_call":
            self.submit_entered.set()
            if not self.release_submit.wait(2.0):
                raise TimeoutError("test did not release blocked submit")
        if self.mode == "complete_inline":
            call.complete(result=success_result(call))
        if self.mode == "complete_then_raise":
            call.complete(error=runtime.EngineUnhealthy("native write failed"))
            raise runtime.EngineUnhealthy("native write failed")
        return call

    def status(self, timeout=5.0, *, fail_unanswered=False):
        self.status_calls += 1
        if self.mode == "status_timeout":
            raise TimeoutError("native status response timed out")
        if self.mode == "status_timeout_once":
            self.mode = "normal"
            raise TimeoutError("native status response timed out")
        return super().status(timeout, fail_unanswered=fail_unanswered)

    def close(self):
        if self.complete_on_close and not self.closed:
            for call in self.calls:
                call.complete(error=runtime.RuntimeClosed("runtime is closed"))
        super().close()


def success_result(call, *, reason=wire.FinishReason.STOP, frames=0):
    return wire.DoneEvent(call.request_id, reason, 4, frames, 1_250, 2_500, 4_000, ())


class AudioBackendContractTests(unittest.TestCase):
    def make_transport(self, runtime=None):
        runtime = runtime or DrivenRuntime()
        transport = audio_backend.AudioBackend(runtime, lambda _record: None)
        self.addCleanup(transport.close)
        return transport, runtime

    def wait_for_recovery(self, transport):
        deadline = time.monotonic() + 1.0
        while transport.recovery is not None and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertIsNone(transport.recovery)

    @staticmethod
    def terminal(job, timeout=1.0):
        while True:
            event = job.events.get(timeout=timeout)
            if event[0] in ("done", "error"):
                return event

    def test_job_maps_to_its_native_request(self):
        transport, runtime = self.make_transport()
        job = speech_job()

        transport.submit(job)
        request = runtime.calls[0].request
        self.assertIsInstance(request, audio_runtime.AudioRequest)
        self.assertIs(request.frame, job.frame)
        self.assertTrue(request.speech)
        self.assertEqual(request.deadline, job.deadline)

    def test_adapter_through_real_runtime_serializes_the_speech_frame(self):
        factory = FakeFactory(audio=True)
        native = audio_runtime.AudioRuntime(
            process_factory=factory,
            pending_limit=4,
            eager_start=True,
        )
        transport, _runtime = self.make_transport(native)
        job = speech_job(404, max_frames=50)

        transport.submit(job)
        process = factory.processes[0]
        frame = process.stdin.wait_for(audio_wire.SpeechRequest)[0]
        # The runtime stamps the request id; the rest is the job's.
        self.assertGreater(frame.request_id, 0)
        self.assertEqual(frame.text_tokens, (11, 12, 13, 14))
        self.assertEqual(frame.language_id, 2050)
        self.assertAlmostEqual(frame.temperature, 0.6)
        self.assertEqual(frame.top_k, 17)
        self.assertEqual(frame.seed, 0x123456789ABCDEF0)
        self.assertEqual(frame.max_frames, 50)

        process.send(audio_wire.StartEvent(frame.request_id, 0, 0))
        process.send(audio_wire.AudioEvent(frame.request_id, 0, CHUNK))
        process.send(
            wire.DoneEvent(
                frame.request_id, wire.FinishReason.STOP, 4, 1, 100, 200, 350, ()
            )
        )
        kind, result = self.terminal(job)
        self.assertEqual(kind, "done")
        self.assertEqual((result.reason, result.completion_count), ("stop", 1))
        self.assertEqual(result.streamed, 1920)

    def test_request_without_a_deadline_reaches_the_engine(self):
        # Without --request-timeout no request has a deadline: it waits as
        # long as a wait can, and the audio frames carry none.
        factory = FakeFactory(audio=True)
        native = audio_runtime.AudioRuntime(
            process_factory=factory,
            pending_limit=4,
            eager_start=True,
        )
        transport, _runtime = self.make_transport(native)
        job = speech_job(deadline=None)
        self.assertEqual(
            backend_api.remaining_request_time(job.deadline), threading.TIMEOUT_MAX
        )
        transport.submit(job)
        frame = factory.processes[0].stdin.wait_for(audio_wire.SpeechRequest)[0]
        self.assertEqual(frame.text_tokens, (11, 12, 13, 14))

    def test_start_audio_and_done_are_finalized_off_callback_path(self):
        transport, runtime = self.make_transport()
        job = speech_job()
        transport.submit(job)
        call = runtime.calls[0]

        call.emit(audio_wire.StartEvent(call.request_id, 2, 0))
        call.emit(audio_wire.AudioEvent(call.request_id, 0, CHUNK))
        call.emit(audio_wire.AudioEvent(call.request_id, 1920, CHUNK))
        call.complete(result=success_result(call, frames=2))

        self.assertEqual(job.events.get(timeout=1.0), ("start", None))
        self.assertEqual(job.events.get(timeout=1.0), ("audio", CHUNK))
        self.assertEqual(job.events.get(timeout=1.0), ("audio", CHUNK))
        kind, result = job.events.get(timeout=1.0)
        self.assertEqual(kind, "done")
        self.assertEqual(result.reason, "stop")
        self.assertEqual(result.prompt_tokens, 4)
        self.assertEqual(result.completion_count, 2)
        self.assertEqual(result.streamed, 3840)
        self.assertEqual(result.prefill_ms, 1.25)
        self.assertEqual(result.decode_ms, 2.5)
        self.assertEqual(result.wall_ms, 4.0)
        self.assertIsNotNone(result.first_output_ms)
        self.assertFalse(transport.active)

    def test_a_cancelled_request_is_a_result_without_output(self):
        transport, runtime = self.make_transport()
        job = speech_job()
        transport.submit(job)
        call = runtime.calls[0]

        call.emit(audio_wire.StartEvent(call.request_id, 0, 0))
        transport.cancel(job)
        call.complete(result=success_result(call, reason=wire.FinishReason.CANCELLED))

        kind, result = self.terminal(job)
        self.assertEqual(kind, "done")
        self.assertEqual(result.reason, "cancelled")
        self.assertEqual(result.streamed, 0)
        self.assertIsNone(result.first_output_ms)
        self.assertFalse(transport.active)

    def test_inline_completion_cannot_leave_stale_active_job(self):
        transport, _runtime = self.make_transport(DrivenRuntime("complete_inline"))
        job = speech_job()

        transport.submit(job)
        self.assertEqual(self.terminal(job)[0], "done")
        self.assertFalse(transport.active)

    def test_completion_then_submit_error_delivers_exactly_one_terminal(self):
        transport, _runtime = self.make_transport(DrivenRuntime("complete_then_raise"))
        job = speech_job()

        transport.submit(job)
        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(error.status, 503)
        self.assertEqual(error.code, "runtime_unavailable")
        with self.assertRaises(queue.Empty):
            job.events.get(timeout=0.05)
        self.assertFalse(transport.active)

    def test_pre_admission_native_failure_is_refused_once_without_retry(self):
        # The engine died, or broke the protocol, after do_POST's own refusal
        # check passed.
        for mode in ("raise_before_call", "protocol_failure_before_call"):
            with self.subTest(mode=mode):
                runtime_client = DrivenRuntime(mode)
                runtime_client.ready = False
                transport, _runtime = self.make_transport(runtime_client)
                job = speech_job()

                transport.submit(job)
                self.assertEqual(runtime_client.submit_attempts, 1)
                kind, error = self.terminal(job)
                self.assertEqual(kind, "error")
                self.assertEqual((error.status, error.code), (503, "engine_recovering"))
                self.assertFalse(transport.active)

    def test_pending_limit_does_not_leave_active_state(self):
        transport, _runtime = self.make_transport(DrivenRuntime("pending_limit"))
        job = speech_job()

        self.assertIsNone(transport.submit(job))
        self.assertFalse(transport.active)
        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(
            (error.status, error.message, error.code),
            (503, "request queue is full", "frontend_overloaded"),
        )

    def test_callback_error_cancels_and_is_the_terminal(self):
        transport, runtime = self.make_transport()
        job = speech_job()
        transport.submit(job)
        call = runtime.calls[0]

        failure = RuntimeError("reader failed")
        with mock.patch.object(transport, "_on_event", side_effect=failure):
            call.emit(audio_wire.StartEvent(call.request_id, 0, 0))
        self.assertEqual(runtime.cancel_count, 1)
        call.complete(result=success_result(call, reason=wire.FinishReason.CANCELLED))

        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual((error.status, error.code), (500, "runtime_error"))
        self.assertFalse(transport.active)

    def test_cancel_is_idempotent(self):
        transport, runtime = self.make_transport()
        job = speech_job()
        transport.submit(job)

        transport.cancel(job)
        transport.cancel(job)
        self.assertTrue(job.cancelled.is_set())
        self.assertEqual(runtime.cancel_count, 1)

    def test_cancel_during_submit_is_sent_when_call_becomes_available(self):
        runtime = DrivenRuntime("block_after_call")
        transport, _runtime = self.make_transport(runtime)
        job = speech_job()
        thread = threading.Thread(target=transport.submit, args=(job,))
        thread.start()
        self.assertTrue(runtime.submit_entered.wait(1.0))

        transport.cancel(job)
        runtime.release_submit.set()
        thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(runtime.cancel_count, 1)

    def test_close_can_own_terminal_while_submit_is_still_returning(self):
        runtime = DrivenRuntime("block_after_call")
        transport, _runtime = self.make_transport(runtime)
        job = speech_job()
        submit_thread = threading.Thread(target=transport.submit, args=(job,))
        submit_thread.start()
        self.assertTrue(runtime.submit_entered.wait(1.0))
        close_thread = threading.Thread(target=transport.close)

        close_thread.start()
        close_thread.join(1.0)
        runtime.release_submit.set()
        submit_thread.join(1.0)

        self.assertFalse(close_thread.is_alive())
        self.assertFalse(submit_thread.is_alive())
        self.assertEqual(self.terminal(job)[0], "error")
        self.assertFalse(transport.active)

    def test_close_drains_runtime_terminal_before_stopping_finalizer(self):
        transport, _runtime = self.make_transport()
        job = speech_job()
        transport.submit(job)

        transport.close()

        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(error.status, 503)
        self.assertEqual(error.code, "server_shutdown")
        self.assertFalse(transport.finalizer.is_alive())
        self.assertFalse(transport.active)

    def test_close_owns_terminal_even_when_native_cancel_wins_race(self):
        transport, runtime = self.make_transport()
        job = speech_job()
        transport.submit(job)
        call = runtime.calls[0]

        # The fake native side completes successfully as soon as close starts.
        # The HTTP contract must still report the shutdown that claimed this
        # active request, independent of which native terminal arrived first.
        original_close = runtime.close

        def close_with_done():
            call.complete(result=success_result(call))
            original_close()

        runtime.close = close_with_done
        transport.close()

        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual((error.status, error.code), (503, "server_shutdown"))
        self.assertFalse(transport.active)

    def test_close_fails_stranded_call_without_hanging_http_waiter(self):
        runtime = DrivenRuntime(complete_on_close=False)
        transport, _runtime = self.make_transport(runtime)
        job = speech_job()
        transport.submit(job)

        transport.close()

        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(error.status, 503)
        self.assertEqual(error.code, "server_shutdown")
        self.assertFalse(transport.finalizer.is_alive())
        self.assertFalse(transport.active)

    def test_native_ready_is_authoritative_and_malformed_status_fails_closed(self):
        transport, runtime = self.make_transport()
        runtime.status_event = wire.StatusJsonEvent(1, b"[]")
        self.assertFalse(transport.is_ready())

        # Native ready already folds in memory pressure and Metal health.
        runtime.status_event = native_peer.status_event(ready=False)
        self.assertFalse(transport.is_ready())

        runtime.status_event = native_peer.status_event(ready="false")
        self.assertFalse(transport.is_ready())

        # A document without a readiness says nothing about it.
        runtime.status_event = wire.StatusJsonEvent(1, b'{"schema_version":1}')
        status = transport.status()
        self.assertFalse(status["ready"])
        self.assertIn("error", status["transport"])

        runtime.status_event = native_peer.status_event()
        self.assertTrue(transport.is_ready())
        self.assertEqual(runtime.status_calls, 5)

    def test_status_adds_only_transport_accounting(self):
        transport, runtime = self.make_transport()
        status = transport.status()

        self.assertTrue(status["ready"])
        self.assertEqual(
            status["transport"],
            {
                "ready": True,
                "recovering": False,
                "stopped": False,
                "pending": 0,
                "pending_limit": 8,
                "restarts": 3,
                "last_crash_trace": None,
                "status_stale": False,
                "status_age_ms": 0.0,
            },
        )

    def test_busy_loop_keeps_last_readiness_until_status_is_unanswered_for_the_bound(
        self,
    ):
        runtime = DrivenRuntime()
        transport, _runtime = self.make_transport(runtime)
        clock = [100.0]
        with mock.patch.object(
            backend_api, "time", SimpleNamespace(monotonic=lambda: clock[0])
        ):
            self.assertTrue(transport.status()["ready"])
            runtime.mode = "status_timeout"

            status = transport.status(timeout=0.01)

            self.assertTrue(status["ready"])
            self.assertTrue(status["transport"]["status_stale"])
            # The transport itself is still live: launchers use this to tell a
            # busy engine from a missing server.
            self.assertTrue(status["transport"]["ready"])
            self.assertIsInstance(status["transport"]["status_age_ms"], float)
            self.assertEqual(
                status["transport"]["error"], "native status response timed out"
            )
            self.assertTrue(transport.is_ready())

            clock[0] += STATUS_ANSWER_LIMIT_SECONDS
            status = transport.status(timeout=0.01)
            self.assertFalse(status["ready"])
            self.assertTrue(status["transport"]["status_stale"])
            self.assertFalse(transport.is_ready())

    def test_failed_engine_snapshot_is_not_ready_after_restart(self):
        native = DrivenRuntime()
        transport, _runtime = self.make_transport(native)
        self.assertTrue(transport.status()["ready"])
        native.mode = "status_timeout"
        with mock.patch.object(backend_api, "print_status"):
            transport._engine_failed(runtime.EngineUnhealthy("gpu fault"), 120.0)
            # The relaunch succeeds; its status refresh is not answered.
            self.wait_for_recovery(transport)
        self.assertTrue(native.ready)
        self.assertFalse(transport.status(timeout=0.01)["ready"])
        self.assertFalse(transport.is_ready())

    def test_late_status_answer_cannot_undo_an_engine_failure(self):
        native = DrivenRuntime()
        transport, _runtime = self.make_transport(native)
        self.assertTrue(transport.status()["ready"])
        decode = transport._decode_status_event
        answered = []

        def fail_after_the_answer(event):
            if not answered:
                answered.append(event)
                native.ready = False
                transport._engine_failed(runtime.EngineUnhealthy("gpu fault"), 120.0)
            return decode(event)

        with (
            mock.patch.object(
                transport, "_decode_status_event", side_effect=fail_after_the_answer
            ),
            mock.patch.object(backend_api, "print_status") as console,
        ):
            self.assertFalse(transport.status()["ready"])
            self.wait_for_recovery(transport)
        self.assertNotIn(mock.call("Engine restarted"), console.call_args_list)
        self.assertEqual(transport.engine_error, "gpu fault")
        self.assertFalse(transport.status_snapshot["ready"])
        # The relaunched engine is busy: the failed one's answer is no evidence.
        native.ready = True
        native.mode = "status_timeout"
        self.assertFalse(transport.status(timeout=0.01)["ready"])

    def test_answer_from_a_replaced_engine_is_not_evidence(self):
        native = DrivenRuntime()
        transport, _runtime = self.make_transport(native)
        native.status_event = native_peer.status_event(ready=False)
        self.assertFalse(transport.status()["ready"])
        status = native.status

        def relaunched_after_the_answer(*args, **kwargs):
            event = status(*args, **kwargs)
            # The engine that answered fails, and a relaunched one is Ready
            # before the answer is cached.
            native.restart_count += 1
            return event

        native.status_event = native_peer.status_event(ready=True)
        with mock.patch.object(
            native, "status", side_effect=relaunched_after_the_answer
        ):
            self.assertFalse(transport.status()["ready"])
        self.assertFalse(transport.status_snapshot["ready"])
        # The relaunched engine is busy: the replaced one's answer is no evidence.
        native.mode = "status_timeout"
        self.assertFalse(transport.status(timeout=0.01)["ready"])

    def test_skipped_probe_does_not_start_the_unanswered_clock(self):
        native = DrivenRuntime()
        transport, _runtime = self.make_transport(native)
        clock = [100.0]
        cached = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        cache_status = transport._cache_status

        def cache_then_hold_the_refresh(snapshot, restarts):
            cached_status = cache_status(snapshot, restarts)
            if threading.current_thread() is transport.recovery:
                cached.set()
                release.wait(2.0)
            return cached_status

        with (
            mock.patch.object(
                backend_api, "time", SimpleNamespace(monotonic=lambda: clock[0])
            ),
            mock.patch.object(
                transport, "_cache_status", side_effect=cache_then_hold_the_refresh
            ),
        ):
            self.assertTrue(transport.status()["ready"])
            native.mode = "status_timeout_once"
            self.assertTrue(transport.status()["ready"])
            self.assertTrue(cached.wait(1.0))
            # Skipped: the refresh has answered but its worker still runs.
            self.assertTrue(transport.status()["ready"])
            release.set()
            self.wait_for_recovery(transport)
            self.assertIsNone(transport.status_unanswered_since)
            clock[0] += STATUS_ANSWER_LIMIT_SECONDS
            native.mode = "status_timeout_once"
            # One missed probe later on is a busy loop, not a wedged one.
            self.assertTrue(transport.status()["ready"])

    def test_answered_refresh_stops_the_unanswered_clock(self):
        native = DrivenRuntime()
        transport, _runtime = self.make_transport(native)
        clock = [100.0]

        def status(timeout, fail_unanswered=False):
            if fail_unanswered:
                # Answered, though not with a status this server reads.
                return wire.StatusJsonEvent(1, b"[]")
            raise TimeoutError("native status response timed out")

        self.assertTrue(transport.status()["ready"])
        with (
            mock.patch.object(
                backend_api, "time", SimpleNamespace(monotonic=lambda: clock[0])
            ),
            mock.patch.object(native, "status", side_effect=status),
        ):
            self.assertTrue(transport.status()["ready"])
            self.wait_for_recovery(transport)
            self.assertIsNone(transport.status_unanswered_since)
            clock[0] += STATUS_ANSWER_LIMIT_SECONDS
            self.assertTrue(transport.status()["ready"])
            self.wait_for_recovery(transport)

    def test_malformed_status_fails_closed_while_the_loop_is_busy(self):
        native = DrivenRuntime()
        transport, _runtime = self.make_transport(native)
        self.assertTrue(transport.status()["ready"])
        native.mode = "status_timeout"
        self.assertTrue(transport.status()["ready"])
        self.wait_for_recovery(transport)
        self.assertIsNotNone(transport.status_unanswered_since)
        native.mode = "normal"
        native.status_event = wire.StatusJsonEvent(1, b"[]")
        self.assertFalse(transport.status()["ready"])

    def test_status_timeout_without_cache_keeps_type_stable(self):
        transport, _runtime = self.make_transport(DrivenRuntime("status_timeout"))

        status = transport.status(timeout=0.01)

        self.assertFalse(status["ready"])
        self.assertTrue(status["transport"]["status_stale"])
        self.assertEqual(status["transport"]["status_age_ms"], 0.0)

    def test_status_timeout_without_valid_cache_is_fail_closed(self):
        runtime = DrivenRuntime("status_timeout")
        transport, _runtime = self.make_transport(runtime)
        self.assertFalse(transport.is_ready())

    def test_timed_out_status_keeps_one_background_refresh_alive(self):
        runtime = DrivenRuntime()
        transport, _runtime = self.make_transport(runtime)
        self.assertTrue(transport.status()["ready"])
        runtime.mode = "status_timeout_once"

        stale = transport.status(timeout=0.01)
        self.assertTrue(stale["transport"]["status_stale"])
        self.wait_for_recovery(transport)
        refreshed = transport.status()
        self.assertFalse(refreshed["transport"]["status_stale"])
        self.assertTrue(refreshed["ready"])

    def test_close_waits_for_the_recovery_worker_to_start(self):
        transport, _runtime = self.make_transport(DrivenRuntime("status_timeout_once"))
        starting = threading.Event()
        release_start = threading.Event()
        closing = threading.Event()
        closed = threading.Event()
        errors = []
        start = threading.Thread.start

        def gated_start(thread):
            if thread.name == "splash-engine-recovery":
                starting.set()
                if not release_start.wait(2.0):
                    raise TimeoutError("test did not release the worker start")
            return start(thread)

        def status():
            try:
                transport.status(timeout=0.01)
            except BaseException as error:
                errors.append(error)

        def close():
            closing.set()
            try:
                transport.close()
            except BaseException as error:
                errors.append(error)
            finally:
                closed.set()

        status_thread = threading.Thread(target=status)
        close_thread = threading.Thread(target=close)
        try:
            with mock.patch.object(threading.Thread, "start", gated_start):
                status_thread.start()
                self.assertTrue(starting.wait(1.0))
                close_thread.start()
                self.assertTrue(closing.wait(1.0))
                self.assertFalse(closed.wait(0.05))
                release_start.set()
                status_thread.join(1.0)
                close_thread.join(1.0)
            self.assertFalse(status_thread.is_alive())
            self.assertFalse(close_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertFalse(transport.finalizer.is_alive())
        finally:
            release_start.set()
            for thread in (status_thread, close_thread):
                if thread.ident is not None:
                    thread.join(2.0)
            transport.close()
            # Keep this regression's fixture bounded even if close regresses.
            if transport.finalizer.is_alive():
                transport.terminals.put(None)
                transport.finalizer.join(1.0)

    def test_failed_status_thread_start_releases_refresh_ownership(self):
        transport, native = self.make_transport(DrivenRuntime("status_timeout"))
        with mock.patch.object(
            threading.Thread, "start", side_effect=RuntimeError("cannot start thread")
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot start thread"):
                transport.status(timeout=0.01)
        self.assertIsNone(transport.recovery)
        self.assertTrue(transport.refresh_due)

        refreshed = threading.Event()

        def status(timeout, fail_unanswered=False):
            if not fail_unanswered:
                raise TimeoutError("native status response timed out")
            refreshed.set()
            return native.status_event

        with mock.patch.object(native, "status", side_effect=status):
            # The next probe starts the refresh still owed.
            transport.status(timeout=0.01)
            self.assertTrue(refreshed.wait(1.0))
            transport.close()
        self.assertFalse(transport.finalizer.is_alive())

    def test_runtime_errors_map_to_stable_http_statuses(self):
        cases = (
            (
                runtime.RequestFailed(1, b"invalid_sampling", b"sampling is invalid"),
                (400, "invalid_sampling"),
            ),
            (
                runtime.RequestFailed(1, b"invalid_request", b"text is empty"),
                (400, "invalid_request"),
            ),
            (
                runtime.RequestFailed(1, b"limit_exceeded", b"too many samples"),
                (400, "limit_exceeded"),
            ),
            (
                runtime.RequestFailed(1, b"deadline_exceeded", b"deadline elapsed"),
                (504, "request_timeout"),
            ),
            (
                runtime.RequestFailed(1, b"temporarily_busy", b"retry", retryable=True),
                (503, "temporarily_busy"),
            ),
            (
                runtime.RequestFailed(1, b"capacity_exhausted", b"no pages"),
                (400, "capacity_exhausted"),
            ),
            (
                runtime.RequestFailed(1, b"internal", b"a bug"),
                (500, "internal"),
            ),
            (runtime.EngineUnhealthy("gpu failed"), (503, "runtime_unavailable")),
            (runtime.ProtocolFatal("bad frame"), (503, "runtime_unavailable")),
            (runtime.RuntimeClosed("closed"), (503, "runtime_unavailable")),
            (TimeoutError(), (504, "request_timeout")),
            (RuntimeError("boom"), (500, "runtime_error")),
        )
        for native, expected in cases:
            with self.subTest(native=native):
                error = audio_backend.AudioBackend._api_error(native)
                self.assertEqual((error.status, error.code), expected)

    def test_fakes_offer_the_interfaces_they_stand_in_for(self):
        runtime_fake = DrivenRuntime()
        for fake, real in (
            (runtime_fake, audio_runtime.AudioRuntime),
            (runtime_fake.submit(None), runtime.RuntimeCall),
        ):
            with self.subTest(fake=type(fake).__name__):
                self.assertEqual(interface_drift(fake, real), [])


if __name__ == "__main__":
    unittest.main()

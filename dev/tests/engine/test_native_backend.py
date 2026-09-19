import array
import gc
import queue
import struct
import threading
import time
import unittest
from unittest import mock

from dev.tests.engine.test_runtime import FakeFactory
from server import backend as backend_api
from server import constraints as generation_constraints
from server import errors as api_errors
from server import frontend as request_frontend
from server import images, runtime
from server import protocol as wire


class FakeTokenizer:
    backend_tokenizer = None

    @staticmethod
    def decode(token_ids, **_kwargs):
        return "|".join(str(token) for token in token_ids)

    @staticmethod
    def convert_tokens_to_ids(_token):
        return None

    @staticmethod
    def apply_chat_template(_messages, **kwargs):
        rendered = "<|im_start|>assistant\n<think>\n"
        if not kwargs.get("enable_thinking", True):
            rendered += "\n</think>\n\n"
        return rendered if kwargs.get("tokenize") is False else [31, 32, 33]

    def __call__(self, _text, **_kwargs):
        return {"input_ids": [31, 32, 33]}


class FakeConstraint:
    def __init__(self, words_per_mask=2, *, omit_anchor=False, consume_error=None):
        self.words_per_mask = words_per_mask
        self.omit_anchor = omit_anchor
        self.consume_error = consume_error
        self.mask_calls = []
        self.consumed = []

    def masks(self, simulation_tokens):
        simulation_tokens = tuple(simulation_tokens)
        self.mask_calls.append(simulation_tokens)
        rows = len(simulation_tokens) + (0 if self.omit_anchor else 1)
        words = tuple(range(1, rows * self.words_per_mask + 1))
        return struct.pack(f"<{len(words)}I", *words)

    def consume(self, tokens):
        if self.consume_error is not None:
            raise self.consume_error
        self.consumed.append(tuple(tokens))


class FakeCall:
    def __init__(self, request_id, request, on_event, on_complete):
        self.request_id = request_id
        self.request = request
        self.on_event = on_event
        self.on_complete = on_complete
        self.callback_errors = ()
        self.cancel_requested = False
        self.cancel_writes = 0
        self._result = None
        self._error = None
        self._done = False

    @property
    def done(self):
        return self._done

    def emit(self, event):
        self.on_event(self, event)

    def complete(self, *, result=None, error=None):
        if self._done:
            return False
        self._result = result
        self._error = error
        self._done = True
        self.on_complete(self)
        return True

    def result(self, _timeout=None):
        if not self._done:
            raise TimeoutError("fake call is not complete")
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self):
        if self._done or self.cancel_requested:
            return False
        self.cancel_requested = True
        self.cancel_writes += 1
        return True


class FakeRuntime:
    def __init__(self, mode="normal", *, complete_on_close=True):
        self.mode = mode
        self.complete_on_close = complete_on_close
        self.calls = []
        self.ready = True
        self.last_status = None
        self.status_calls = 0
        self.status_event = wire.StatusJsonEvent(
            1,
            wire.STATUS_SCHEMA_VERSION,
            b'{"schema_version":5,"ready":true,"memory_pressure":"normal",'
            b'"metal":{"healthy":true}}',
        )
        self.pending_limit = 8
        self.restart_count = 3
        self.last_crash_trace = None
        self.closed = False
        self.submit_attempts = 0
        self.submit_entered = threading.Event()
        self.release_submit = threading.Event()

    @property
    def pending_count(self):
        return sum(not call.done for call in self.calls)

    def submit(self, request, *, on_event, on_complete):
        self.submit_attempts += 1
        if self.mode == "pending_limit":
            raise runtime.PendingLimitExceeded("full")
        if self.mode == "raise_before_call" or (
            self.mode == "raise_once_before_call" and self.submit_attempts == 1
        ):
            raise runtime.EngineUnhealthy("native startup failed")
        call = FakeCall(len(self.calls) + 1, request, on_event, on_complete)
        self.calls.append(call)
        if self.mode == "block_after_call":
            self.submit_entered.set()
            if not self.release_submit.wait(2.0):
                raise TimeoutError("test did not release blocked submit")
        if self.mode == "complete_inline":
            call.complete(result=success_result(call))
        if self.mode == "complete_then_raise":
            call.complete(error=runtime.EngineUnhealthy("native write failed"))
            raise runtime.EngineUnhealthy("native write failed")
        if self.mode == "raise":
            raise runtime.EngineUnhealthy("submit failed")
        return call

    def status(self, timeout=5.0):
        if timeout <= 0:
            raise ValueError("status timeout must be positive")
        self.status_calls += 1
        if self.mode == "status_timeout":
            raise TimeoutError("native status response timed out")
        if self.mode == "status_timeout_once":
            self.mode = "normal"
            raise TimeoutError("native status response timed out")
        self.last_status = self.status_event
        return self.status_event

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.ready = False
        if self.complete_on_close:
            for call in self.calls:
                call.complete(error=runtime.RuntimeClosed("runtime is closed"))


def make_job(request_id=101, *, constraint=None, temperature=0.0):
    return backend_api.Job(
        request_id=request_id,
        prompt_tokens=[11, 12, 13, 14],
        max_new_tokens=37,
        seed=0x123456789ABCDEF0,
        temperature=temperature,
        top_p=0.75,
        top_k=17,
        deadline=time.monotonic() + 10.0,
        priority=backend_api.REQUEST_PRIORITIES["foreground"],
        constraint=constraint,
    )


def success_result(call, *, reason=wire.FinishReason.STOP, tokens=()):
    done = wire.DoneEvent(call.request_id, reason, 4, len(tokens), 1_250, 2_500, 4_000)
    return runtime.GenerationResult(call.request_id, None, tuple(tokens), done)


class NativeBackendContractTests(unittest.TestCase):
    def make_transport(self, runtime=None):
        runtime = runtime or FakeRuntime()
        transport = backend_api.NativeBackend(runtime, FakeTokenizer())
        self.addCleanup(transport.close)
        return transport, runtime

    @staticmethod
    def terminal(job, timeout=1.0):
        while True:
            event = job.events.get(timeout=timeout)
            if event[0] in ("done", "error"):
                return event

    def test_job_maps_to_generation_request_and_mask_has_anchor_row(self):
        constraint = FakeConstraint(words_per_mask=2)
        transport, runtime = self.make_transport()
        job = make_job(constraint=constraint, temperature=0.6)

        self.assertTrue(transport.submit(job))
        request = runtime.calls[0].request
        self.assertEqual(request.prompt_tokens, (11, 12, 13, 14))
        self.assertEqual(request.logical_max_output_tokens, 37)
        self.assertEqual(request.priority, wire.RequestPriority.FOREGROUND)
        self.assertEqual(request.sampling, wire.SamplingParameters(0.6, 0.75, 17))
        self.assertEqual(request.seed, 0x123456789ABCDEF0)
        self.assertEqual(request.cohort, wire.Cohort.CONSTRAINED)
        self.assertEqual(request.constraint, wire.ConstraintMode.TOKEN_MASK)
        self.assertGreater(request.deadline.remaining_micros, 0)

        event = wire.MaskRequestEvent(runtime.calls[0].request_id, 91, 2, (5, 6))
        self.assertEqual(
            request.mask_provider(event), array.array("I", (1, 2, 3, 4, 5, 6)).tobytes()
        )
        self.assertEqual(constraint.mask_calls, [(5, 6)])

    def test_mask_simulation_empty_and_eight_tokens_produce_one_and_nine_rows(self):
        constraint = FakeConstraint(words_per_mask=2)
        provider = backend_api.NativeBackend._mask_provider(
            make_job(constraint=constraint)
        )

        initial = wire.MaskRequestEvent(1, 20, 2, ())
        simulation = wire.MaskRequestEvent(1, 21, 2, tuple(range(8)))
        self.assertEqual(provider(initial), array.array("I", (1, 2)).tobytes())
        self.assertEqual(provider(simulation), array.array("I", range(1, 19)).tobytes())
        self.assertEqual(constraint.mask_calls, [(), tuple(range(8))])

        with self.assertRaises(api_errors.NativeError) as caught:
            generation_constraints.TokenConstraint(None, None).masks(tuple(range(9)))
        self.assertEqual(caught.exception.code, "constraint_error")

    def test_http_fields_reach_native_generation_request(self):
        transport, runtime = self.make_transport()
        app = request_frontend.Frontend(
            FakeTokenizer(), transport, "test-model", 128, 32, 10, 2
        )
        job, _thinking, _tools = app.prepare(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "none",
                "max_completion_tokens": 19,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 13,
                "seed": 99,
                "priority": "background",
            }
        )

        self.assertTrue(transport.submit(job))
        request = runtime.calls[0].request
        self.assertEqual(request.prompt_tokens, (31, 32, 33))
        self.assertEqual(request.logical_max_output_tokens, 19)
        self.assertEqual(request.priority, wire.RequestPriority.BACKGROUND)
        self.assertEqual(request.sampling, wire.SamplingParameters(0.7, 0.8, 13))
        self.assertEqual(request.seed, 99)
        self.assertEqual(request.cohort, wire.Cohort.SAMPLING)

    def test_adapter_through_real_runtime_serializes_request_frame(self):
        factory = FakeFactory()
        native = runtime.MultiplexedRuntime(
            process_factory=factory,
            pending_limit=4,
        )
        transport, _runtime = self.make_transport(native)
        job = make_job(404, temperature=0.6)

        self.assertTrue(transport.submit(job))
        process = factory.processes[0]
        frame = process.stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(frame.priority, wire.RequestPriority.FOREGROUND)
        self.assertEqual(frame.logical_max_output_tokens, 37)
        self.assertEqual(frame.prompt_tokens, (11, 12, 13, 14))
        self.assertAlmostEqual(frame.sampling.temperature, 0.6)
        self.assertEqual((frame.sampling.top_p, frame.sampling.top_k), (0.75, 17))
        self.assertEqual(frame.seed, 0x123456789ABCDEF0)
        self.assertEqual(frame.cohort, wire.Cohort.SAMPLING)
        self.assertEqual(frame.constraint, wire.ConstraintMode.NONE)
        self.assertGreater(frame.absolute_deadline_unix_micros, 0)
        self.assertGreater(frame.remaining_deadline_micros, 0)

        process.send(
            wire.StartEvent(frame.request_id, wire.CacheDisposition.MISS, 0, 0, 4096)
        )
        process.send(wire.TokensEvent(frame.request_id, 0, (7,)))
        process.send(
            wire.DoneEvent(
                frame.request_id,
                wire.FinishReason.STOP,
                4,
                1,
                100,
                200,
                350,
            )
        )
        self.assertEqual(self.terminal(job)[0], "done")

    def test_score_job_maps_to_score_only_request_and_returns_logits(self):
        factory = FakeFactory()
        native = runtime.MultiplexedRuntime(
            process_factory=factory,
            pending_limit=4,
        )
        transport, _runtime = self.make_transport(native)
        job = make_job(404)
        job.max_new_tokens = 0
        job.temperature = 0.0
        job.top_p = 1.0
        job.top_k = 0
        job.score_tokens = (101, 202, 303)

        self.assertTrue(transport.submit(job))
        process = factory.processes[0]
        frame = process.stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(frame.score_tokens, (101, 202, 303))
        self.assertEqual(frame.logical_max_output_tokens, 0)
        self.assertEqual(frame.cohort, wire.Cohort.GREEDY)

        process.send(
            wire.StartEvent(frame.request_id, wire.CacheDisposition.MISS, 0, 0, 4096)
        )
        process.send(
            wire.DoneEvent(
                frame.request_id,
                wire.FinishReason.STOP,
                4,
                0,
                100,
                0,
                350,
                (1.5, -2.25, 0.5),
            )
        )
        kind, result = self.terminal(job)
        self.assertEqual(kind, "done")
        self.assertEqual(result.option_logits, (1.5, -2.25, 0.5))
        self.assertEqual(result.completion_tokens, 0)

    def test_real_runtime_correlates_initial_and_verify_mask_rows(self):
        factory = FakeFactory()
        native = runtime.MultiplexedRuntime(
            process_factory=factory,
            pending_limit=4,
        )
        transport, _runtime = self.make_transport(native)
        constraint = FakeConstraint(words_per_mask=2)
        job = make_job(405, constraint=constraint)

        self.assertTrue(transport.submit(job))
        process = factory.processes[0]
        request = process.stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(request.cohort, wire.Cohort.CONSTRAINED)
        self.assertEqual(request.constraint, wire.ConstraintMode.TOKEN_MASK)
        process.send(
            wire.StartEvent(
                request.request_id,
                wire.CacheDisposition.MISS,
                0,
                0,
                4096,
            )
        )

        process.send(wire.MaskRequestEvent(request.request_id, 71, 2, ()))
        initial = process.stdin.wait_for(wire.MaskResponseFrame)[0]
        self.assertEqual(initial.request_id, request.request_id)
        self.assertEqual(initial.mask_request_id, 71)
        self.assertEqual(initial.mask_words, (1, 2))

        simulation = tuple(range(100, 108))
        process.send(wire.MaskRequestEvent(request.request_id, 72, 2, simulation))
        verify = process.stdin.wait_for(wire.MaskResponseFrame, count=2)[1]
        self.assertEqual(verify.request_id, request.request_id)
        self.assertEqual(verify.mask_request_id, 72)
        self.assertEqual(verify.mask_words, tuple(range(1, 19)))
        self.assertEqual(constraint.mask_calls, [(), simulation])

        process.send(
            wire.DoneEvent(
                request.request_id,
                wire.FinishReason.STOP,
                4,
                0,
                100,
                0,
                150,
            )
        )
        self.assertEqual(self.terminal(job)[0], "done")

    def test_mask_provider_rejects_missing_n_plus_one_anchor_row(self):
        job = make_job(constraint=FakeConstraint(omit_anchor=True))
        provider = backend_api.NativeBackend._mask_provider(job)
        event = wire.MaskRequestEvent(1, 2, 2, (5, 6))

        with self.assertRaises(api_errors.NativeError) as caught:
            provider(event)
        self.assertEqual(caught.exception.code, "constraint_error")
        self.assertIn("expected 24", caught.exception.message)

    def test_greedy_and_sampling_cohorts_follow_temperature(self):
        transport, _runtime = self.make_transport()
        greedy = transport._generation_request(make_job(1, temperature=0.0))
        sampling = transport._generation_request(make_job(2, temperature=0.5))

        self.assertEqual(greedy.cohort, wire.Cohort.GREEDY)
        self.assertEqual(sampling.cohort, wire.Cohort.SAMPLING)
        self.assertEqual(greedy.constraint, wire.ConstraintMode.NONE)
        self.assertEqual(sampling.constraint, wire.ConstraintMode.NONE)

    def test_start_tokens_and_done_are_finalized_off_callback_path(self):
        constraint = FakeConstraint()
        transport, runtime = self.make_transport()
        job = make_job(constraint=constraint)
        self.assertTrue(transport.submit(job))
        call = runtime.calls[0]

        call.emit(
            wire.StartEvent(call.request_id, wire.CacheDisposition.PREFIX_HIT, 2, 2, 99)
        )
        call.emit(wire.TokensEvent(call.request_id, 0, (7, 8)))
        call.complete(result=success_result(call, tokens=(7, 8)))

        self.assertEqual(job.events.get(timeout=1.0), ("start", "hit"))
        self.assertEqual(job.events.get(timeout=1.0), ("text", "7|8"))
        kind, result = job.events.get(timeout=1.0)
        self.assertEqual(kind, "done")
        self.assertEqual(result.reason, "stop")
        self.assertEqual(result.prefill_tokens, 2)
        self.assertEqual(result.start_to_first_token_ms, 1.25)
        self.assertEqual(result.first_token_to_done_ms, 2.5)
        self.assertEqual(result.request_wall_ms, 4.0)
        self.assertEqual(constraint.consumed, [(7, 8)])
        self.assertFalse(transport.active)

    def test_finalized_request_returns_its_image_budget(self):
        factory = FakeFactory()
        transport, _runtime = self.make_transport(
            runtime.MultiplexedRuntime(process_factory=factory, pending_limit=4)
        )
        cache = images.ImageCache(budget_bytes=0, request_budget_bytes=4)
        job = make_job()
        job.image_owner = cache.request_batch()
        job.image_owner.append(images.PreparedImage(2, 2, b"abcd", 0, 0))
        self.assertEqual(cache.stats()["request_bytes"], 4)

        self.assertTrue(transport.submit(job))
        process = factory.processes[0]
        frame = process.stdin.wait_for(wire.RequestFrame)[0]
        process.send(
            wire.StartEvent(frame.request_id, wire.CacheDisposition.MISS, 0, 0, 4096)
        )
        process.send(
            wire.DoneEvent(
                frame.request_id, wire.FinishReason.STOP, 4, 0, 100, 200, 350
            )
        )
        self.assertEqual(self.terminal(job)[0], "done")
        del job

        # The done event precedes the finalizer dropping the request; the
        # batch then hangs only on the state/call callback cycle.
        deadline = time.monotonic() + 1.0
        while cache.stats()["request_bytes"] and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.001)
        self.assertEqual(cache.stats()["request_bytes"], 0)

    def test_inline_completion_cannot_leave_stale_active_job(self):
        transport, _runtime = self.make_transport(FakeRuntime("complete_inline"))
        job = make_job()

        self.assertTrue(transport.submit(job))
        self.assertEqual(self.terminal(job)[0], "done")
        self.assertFalse(transport.active)

    def test_completion_then_submit_error_delivers_exactly_one_terminal(self):
        transport, _runtime = self.make_transport(FakeRuntime("complete_then_raise"))
        job = make_job()

        self.assertTrue(transport.submit(job))
        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(error.status, 503)
        self.assertEqual(error.code, "runtime_unavailable")
        with self.assertRaises(queue.Empty):
            job.events.get(timeout=0.05)
        self.assertFalse(transport.active)

    def test_pre_admission_native_failure_retries_once_then_succeeds(self):
        runtime_client = FakeRuntime("raise_once_before_call")
        transport, _runtime = self.make_transport(runtime_client)
        job = make_job()

        with mock.patch.object(backend_api, "NATIVE_RECOVERY_GRACE_SECONDS", 0):
            self.assertTrue(transport.submit(job))
        self.assertEqual(runtime_client.submit_attempts, 2)
        self.assertEqual(len(runtime_client.calls), 1)
        runtime_client.calls[0].complete(result=success_result(runtime_client.calls[0]))
        self.assertEqual(self.terminal(job)[0], "done")
        self.assertFalse(transport.active)

    def test_pre_admission_native_failure_has_one_bounded_retry(self):
        runtime_client = FakeRuntime("raise_before_call")
        transport, _runtime = self.make_transport(runtime_client)
        job = make_job()

        with mock.patch.object(backend_api, "NATIVE_RECOVERY_GRACE_SECONDS", 0):
            self.assertTrue(transport.submit(job))
        self.assertEqual(runtime_client.submit_attempts, 2)
        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(error.status, 503)
        self.assertEqual(error.code, "runtime_unavailable")
        self.assertFalse(transport.active)

    def test_pending_limit_does_not_leave_active_state(self):
        transport, _runtime = self.make_transport(FakeRuntime("pending_limit"))
        job = make_job()

        self.assertFalse(transport.submit(job))
        self.assertFalse(transport.active)
        self.assertTrue(job.events.empty())

    def test_constraint_callback_error_cancels_and_maps_to_bad_request(self):
        constraint = FakeConstraint(
            consume_error=api_errors.NativeError("constraint_error", "invalid token")
        )
        transport, runtime = self.make_transport()
        job = make_job(constraint=constraint)
        self.assertTrue(transport.submit(job))
        call = runtime.calls[0]

        call.emit(wire.TokensEvent(call.request_id, 0, (7,)))
        self.assertEqual(call.cancel_writes, 1)
        call.complete(
            result=success_result(call, reason=wire.FinishReason.CANCELLED, tokens=(7,))
        )

        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(error.status, 400)
        self.assertEqual(error.code, "constraint_error")

    def test_cancel_is_idempotent_and_preserves_timeout_flag(self):
        transport, runtime = self.make_transport()
        job = make_job()
        self.assertTrue(transport.submit(job))
        call = runtime.calls[0]

        transport.cancel(job, timed_out=True)
        transport.cancel(job)
        self.assertTrue(job.cancelled.is_set())
        self.assertTrue(job.timed_out)
        self.assertEqual(call.cancel_writes, 1)

    def test_cancel_during_submit_is_sent_when_call_becomes_available(self):
        runtime = FakeRuntime("block_after_call")
        transport, _runtime = self.make_transport(runtime)
        job = make_job()
        submitted = []
        thread = threading.Thread(
            target=lambda: submitted.append(transport.submit(job))
        )
        thread.start()
        self.assertTrue(runtime.submit_entered.wait(1.0))

        transport.cancel(job, timed_out=True)
        runtime.release_submit.set()
        thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(submitted, [True])
        self.assertEqual(runtime.calls[0].cancel_writes, 1)
        self.assertTrue(job.timed_out)

    def test_close_can_own_terminal_while_submit_is_still_returning(self):
        runtime = FakeRuntime("block_after_call")
        transport, _runtime = self.make_transport(runtime)
        job = make_job()
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
        job = make_job()
        self.assertTrue(transport.submit(job))

        transport.close()

        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(error.status, 503)
        self.assertEqual(error.code, "server_shutdown")
        self.assertFalse(transport.finalizer.is_alive())
        self.assertFalse(transport.active)

    def test_close_owns_terminal_even_when_native_cancel_wins_race(self):
        transport, runtime = self.make_transport()
        job = make_job()
        self.assertTrue(transport.submit(job))
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
        runtime = FakeRuntime(complete_on_close=False)
        transport, _runtime = self.make_transport(runtime)
        job = make_job()
        self.assertTrue(transport.submit(job))

        transport.close()

        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(error.status, 503)
        self.assertEqual(error.code, "server_shutdown")
        self.assertFalse(transport.finalizer.is_alive())
        self.assertFalse(transport.active)

    def test_status_schema_and_memory_pressure_are_fail_closed(self):
        transport, runtime = self.make_transport()
        runtime.status_event = wire.StatusJsonEvent(
            1, wire.STATUS_SCHEMA_VERSION, b"[]"
        )
        self.assertFalse(transport.is_ready())

        runtime.status_event = wire.StatusJsonEvent(
            1,
            wire.STATUS_SCHEMA_VERSION,
            b'{"schema_version":5,"ready":true,"memory_pressure":"critical",'
            b'"metal":{"healthy":true}}',
        )
        self.assertFalse(transport.is_ready())

        runtime.status_event = wire.StatusJsonEvent(
            1,
            wire.STATUS_SCHEMA_VERSION,
            b'{"schema_version":5,"ready":"false",'
            b'"memory_pressure":"normal","metal":{"healthy":true}}',
        )
        self.assertFalse(transport.is_ready())

        runtime.status_event = wire.StatusJsonEvent(
            1,
            wire.STATUS_SCHEMA_VERSION,
            b'{"schema_version":5,"ready":true,"metal":{"healthy":true}}',
        )
        self.assertFalse(transport.is_ready())

        runtime.status_event = wire.StatusJsonEvent(
            1,
            wire.STATUS_SCHEMA_VERSION,
            b'{"schema_version":5,"ready":true,"memory_pressure":"normal",'
            b'"metal":{"healthy":true}}',
        )
        self.assertTrue(transport.is_ready())
        self.assertEqual(runtime.status_calls, 5)

        runtime.status_event = wire.StatusJsonEvent(1, 1, b'{"schema_version":3}')
        status = transport.status()
        self.assertFalse(status["ready"])
        self.assertIn("error", status["transport"])

    def test_status_adds_only_transport_accounting(self):
        transport, runtime = self.make_transport()
        status = transport.status()

        self.assertTrue(status["ready"])
        self.assertEqual(
            status["transport"],
            {
                "ready": True,
                "recovering": False,
                "pending": 0,
                "pending_limit": 8,
                "restarts": 3,
                "last_crash_trace": None,
                "status_stale": False,
                "status_age_ms": 0.0,
            },
        )

    def test_status_timeout_fails_closed_even_with_a_cached_snapshot(self):
        runtime = FakeRuntime()
        transport, _runtime = self.make_transport(runtime)
        self.assertTrue(transport.status()["ready"])
        runtime.mode = "status_timeout"

        status = transport.status(timeout=0.01)

        self.assertFalse(status["ready"])
        self.assertEqual(status["schema_version"], wire.STATUS_SCHEMA_VERSION)
        self.assertTrue(status["transport"]["status_stale"])
        # The transport itself is still live: launchers use this to tell a
        # busy engine from a missing server.
        self.assertTrue(status["transport"]["ready"])
        self.assertIsInstance(status["transport"]["status_age_ms"], float)
        self.assertEqual(
            status["transport"]["error"], "native status response timed out"
        )
        self.assertFalse(transport.is_ready())

    def test_status_timeout_without_cache_keeps_type_stable(self):
        transport, _runtime = self.make_transport(FakeRuntime("status_timeout"))

        status = transport.status(timeout=0.01)

        self.assertFalse(status["ready"])
        self.assertTrue(status["transport"]["status_stale"])
        self.assertEqual(status["transport"]["status_age_ms"], 0.0)

    def test_status_timeout_without_valid_cache_is_fail_closed(self):
        runtime = FakeRuntime("status_timeout")
        transport, _runtime = self.make_transport(runtime)
        self.assertFalse(transport.is_ready())

        runtime.last_status = wire.StatusJsonEvent(
            1,
            wire.STATUS_SCHEMA_VERSION,
            b'{"schema_version":4,"ready":true}',
        )
        self.assertFalse(transport.is_ready())

    def test_timed_out_status_keeps_one_background_refresh_alive(self):
        runtime = FakeRuntime()
        transport, _runtime = self.make_transport(runtime)
        self.assertTrue(transport.status()["ready"])
        runtime.mode = "status_timeout_once"

        stale = transport.status(timeout=0.01)
        self.assertTrue(stale["transport"]["status_stale"])
        deadline = time.monotonic() + 1.0
        while transport.status_refresh_inflight and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertFalse(transport.status_refresh_inflight)
        refreshed = transport.status()
        self.assertFalse(refreshed["transport"]["status_stale"])
        self.assertTrue(refreshed["ready"])

    def test_close_waits_for_background_status_thread_to_start(self):
        transport, _runtime = self.make_transport(FakeRuntime("status_timeout_once"))
        starting = threading.Event()
        release_start = threading.Event()
        closing = threading.Event()
        closed = threading.Event()
        errors = []
        start = threading.Thread.start

        def gated_start(thread):
            if thread.name == "splash-status-refresh":
                starting.set()
                if not release_start.wait(2.0):
                    raise TimeoutError("test did not release status thread start")
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
        transport, native = self.make_transport(FakeRuntime("status_timeout"))
        with mock.patch.object(
            threading.Thread, "start", side_effect=RuntimeError("cannot start thread")
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot start thread"):
                transport.status(timeout=0.01)
        self.assertFalse(transport.status_refresh_inflight)
        self.assertIsNone(transport.status_refresh_thread)

        refreshed = threading.Event()
        native.mode = "normal"
        status = native.status

        def refresh(**kwargs):
            result = status(**kwargs)
            refreshed.set()
            return result

        with mock.patch.object(native, "status", side_effect=refresh):
            transport._ensure_background_status_refresh()
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
                runtime.RequestFailed(1, b"deadline_exceeded", b"deadline elapsed"),
                (504, "request_timeout"),
            ),
            (
                runtime.RequestFailed(1, b"temporarily_busy", b"retry", retryable=True),
                (503, "temporarily_busy"),
            ),
            (runtime.EngineUnhealthy("gpu failed"), (503, "runtime_unavailable")),
            (runtime.ProtocolFatal("bad frame"), (500, "protocol_error")),
        )
        for native, expected in cases:
            with self.subTest(native=native):
                error = backend_api.NativeBackend._api_error(native)
                self.assertEqual((error.status, error.code), expected)


if __name__ == "__main__":
    unittest.main()

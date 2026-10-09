import array
import gc
import math
import queue
import random
import struct
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from tokenizers import Tokenizer, decoders, models

from dev.tests import native_peer
from dev.tests.server_fixtures import (
    FOREVER,
    FakeFactory,
    FakeRuntime,
    byte_alphabet,
    byte_backend,
    interface_drift,
    make_frontend,
    make_job,
)
from server import backend as backend_api
from server import constraints as generation_constraints
from server import errors as api_errors
from server import images, runtime, tool_schema
from server import protocol as wire
from server.runtime import STATUS_ANSWER_LIMIT_SECONDS


class DigitTokenizer:
    # Token ids up to 15 decode to their digits.
    backend_tokenizer = byte_backend({token: str(token) for token in range(16)})
    # Rendering ignores it; only the startup probe reads it.
    chat_template = "{%- for message in messages %}{{- message.content }}{%- endfor %}"

    @classmethod
    def decode(cls, token_ids, **_kwargs):
        return cls.backend_tokenizer.decode(list(token_ids))

    @staticmethod
    def convert_tokens_to_ids(_token):
        return None

    @staticmethod
    def apply_chat_template(_messages, **kwargs):
        rendered = ""
        if kwargs.get("add_generation_prompt"):
            rendered = "<|im_start|>assistant\n<think>\n"
            if not kwargs.get("enable_thinking", True):
                rendered += "\n</think>\n\n"
        return rendered if kwargs.get("tokenize") is False else [31, 32, 33]

    def __call__(self, _text, **_kwargs):
        return {"input_ids": [31, 32, 33]}


class FakeConstraint:
    def __init__(self, words_per_mask=2, *, commit_error=None, finish_error=None):
        self.words_per_mask = words_per_mask
        self.commit_error = commit_error
        self.finish_error = finish_error
        self.mask_calls = []
        self.committed = []
        self.finish_calls = 0

    def masks(self, simulation_tokens):
        simulation_tokens = tuple(simulation_tokens)
        self.mask_calls.append(simulation_tokens)
        rows = len(simulation_tokens) + 1
        words = tuple(range(1, rows * self.words_per_mask + 1))
        return struct.pack(f"<{len(words)}I", *words)

    def commit(self, token_ids):
        if self.commit_error is not None:
            raise self.commit_error
        self.committed.append(tuple(token_ids))

    def finish(self):
        self.finish_calls += 1
        if self.finish_error is not None:
            raise self.finish_error


class FakeMatcher:
    """Records the batches a TokenConstraint consumes; accepts every one."""

    def __init__(self):
        self.consumed = []

    def consume_tokens(self, tokens):
        self.consumed.append(tuple(tokens))
        return True


class DrivenRuntime(FakeRuntime):
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


def success_result(call, *, reason=wire.FinishReason.STOP, tokens=()):
    return wire.DoneEvent(
        call.request_id, reason, 4, len(tokens), 1_250, 2_500, 4_000, ()
    )


class NativeBackendContractTests(unittest.TestCase):
    def make_transport(self, runtime=None):
        runtime = runtime or DrivenRuntime()
        transport = backend_api.NativeBackend(
            runtime, DigitTokenizer(), lambda _record: None
        )
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

    def test_job_maps_to_generation_request_and_mask_has_anchor_row(self):
        constraint = FakeConstraint(words_per_mask=2)
        transport, runtime = self.make_transport()
        job = make_job(constraint=constraint, temperature=0.6)

        transport.submit(job)
        request = runtime.calls[0].request
        frame = request.frame
        self.assertEqual(frame.prompt_tokens, (11, 12, 13, 14))
        self.assertEqual(frame.logical_max_output_tokens, 37)
        self.assertEqual(frame.priority, wire.RequestPriority.FOREGROUND)
        self.assertEqual(frame.sampling, job.sampling)
        self.assertEqual(frame.seed, 0x123456789ABCDEF0)
        self.assertEqual(frame.constraint, wire.ConstraintMode.TOKEN_MASK)
        self.assertEqual(request.deadline, job.deadline)

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

        with self.assertRaises(api_errors.ConstraintError):
            generation_constraints.TokenConstraint(None, None).masks(tuple(range(9)))

    def test_http_fields_reach_native_generation_request(self):
        transport, runtime = self.make_transport()
        app = make_frontend(
            DigitTokenizer(), transport, "test-model", 128, 10, 2, vision=True
        )
        job = app.prepare(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "none",
                "max_completion_tokens": 19,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 13,
                "presence_penalty": 1.5,
                "frequency_penalty": 0.5,
                "repetition_penalty": 1.05,
                "seed": 99,
                "priority": "background",
            },
            deadline=FOREVER,
        )

        transport.submit(job)
        frame = runtime.calls[0].request.frame
        self.assertEqual(frame.prompt_tokens, (31, 32, 33))
        self.assertEqual(frame.logical_max_output_tokens, 19)
        self.assertEqual(frame.priority, wire.RequestPriority.BACKGROUND)
        self.assertEqual(
            frame.sampling,
            wire.SamplingParameters(0.7, 0.8, 13, 1.5, 0.5, 1.05),
        )
        self.assertEqual(frame.seed, 99)

    def test_adapter_through_real_runtime_serializes_request_frame(self):
        factory = FakeFactory()
        native = runtime.MultiplexedRuntime(
            process_factory=factory,
            pending_limit=4,
            eager_start=True,
        )
        transport, _runtime = self.make_transport(native)
        job = make_job(404, temperature=0.6)
        job.generation_prompt_tokens = 2
        job.flags = wire.RequestFlag.IGNORE_END_OF_SEQUENCE

        transport.submit(job)
        process = factory.processes[0]
        frame = process.stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(frame.priority, wire.RequestPriority.FOREGROUND)
        self.assertEqual(frame.logical_max_output_tokens, 37)
        self.assertEqual(frame.prompt_tokens, (11, 12, 13, 14))
        self.assertEqual(frame.generation_prompt_tokens, 2)
        self.assertEqual(frame.flags, wire.RequestFlag.IGNORE_END_OF_SEQUENCE)
        self.assertAlmostEqual(frame.sampling.temperature, 0.6)
        self.assertEqual(
            frame.sampling,
            wire.SamplingParameters(
                frame.sampling.temperature, 0.75, 17, 1.5, -0.25, 1.125, 0.125
            ),
        )
        self.assertEqual(frame.seed, 0x123456789ABCDEF0)
        self.assertEqual(frame.constraint, wire.ConstraintMode.NONE)
        self.assertGreater(frame.absolute_deadline_unix_micros, 0)
        self.assertGreater(frame.remaining_deadline_micros, 0)

        process.send(wire.StartEvent(frame.request_id, 0, 0))
        process.send(wire.TokensEvent(frame.request_id, 0, (7,)))
        process.send(
            wire.DoneEvent(
                frame.request_id, wire.FinishReason.STOP, 4, 1, 100, 200, 350, ()
            )
        )
        self.assertEqual(self.terminal(job)[0], "done")

    def test_request_without_a_deadline_reaches_the_engine_without_one(self):
        # Without --request-timeout only a request's own timeout sets one. A
        # request without either waits as long as a wait can, and the engine
        # gets the wire's maximum.
        factory = FakeFactory()
        native = runtime.MultiplexedRuntime(
            process_factory=factory,
            pending_limit=4,
            eager_start=True,
        )
        transport, _runtime = self.make_transport(native)
        app = make_frontend(
            DigitTokenizer(), transport, "test-model", 128, math.inf, 2, vision=True
        )
        self.assertEqual(app.request_deadline({"timeout": 5}, 10), 15)
        job = app.prepare(
            {"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
            deadline=FOREVER,
        )
        self.assertEqual(
            backend_api.remaining_request_time(job.deadline), threading.TIMEOUT_MAX
        )

        transport.submit(job)
        frame = factory.processes[0].stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(frame.absolute_deadline_unix_micros, runtime._MAX_U64)

    def test_score_job_maps_to_score_only_request_and_returns_logits(self):
        factory = FakeFactory()
        native = runtime.MultiplexedRuntime(
            process_factory=factory,
            pending_limit=4,
            eager_start=True,
        )
        transport, _runtime = self.make_transport(native)
        job = make_job(404)
        job.max_new_tokens = 0
        job.sampling = wire.SamplingParameters()
        job.score_tokens = (101, 202, 303)

        transport.submit(job)
        process = factory.processes[0]
        frame = process.stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(frame.score_tokens, (101, 202, 303))
        self.assertEqual(frame.logical_max_output_tokens, 0)

        process.send(wire.StartEvent(frame.request_id, 0, 0))
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
            eager_start=True,
        )
        transport, _runtime = self.make_transport(native)
        constraint = FakeConstraint(words_per_mask=2)
        job = make_job(405, constraint=constraint)

        transport.submit(job)
        process = factory.processes[0]
        request = process.stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(request.constraint, wire.ConstraintMode.TOKEN_MASK)
        process.send(wire.StartEvent(request.request_id, 0, 0))

        process.send(wire.MaskRequestEvent(request.request_id, 71, 2, ()))
        initial = process.stdin.wait_for(wire.MaskResponseFrame)[0]
        self.assertEqual(initial.request_id, request.request_id)
        self.assertEqual(initial.mask_request_id, 71)
        self.assertEqual(initial.mask_words, array.array("I", (1, 2)).tobytes())

        simulation = tuple(range(100, 108))
        process.send(wire.MaskRequestEvent(request.request_id, 72, 2, simulation))
        verify = process.stdin.wait_for(wire.MaskResponseFrame, count=2)[1]
        self.assertEqual(verify.request_id, request.request_id)
        self.assertEqual(verify.mask_request_id, 72)
        self.assertEqual(verify.mask_words, array.array("I", range(1, 19)).tobytes())
        self.assertEqual(constraint.mask_calls, [(), simulation])

        process.send(
            wire.DoneEvent(
                request.request_id, wire.FinishReason.STOP, 4, 0, 100, 0, 150, ()
            )
        )
        self.assertEqual(self.terminal(job)[0], "done")

    def test_start_tokens_and_done_are_finalized_off_callback_path(self):
        constraint = FakeConstraint()
        transport, runtime = self.make_transport()
        job = make_job(constraint=constraint)
        transport.submit(job)
        call = runtime.calls[0]

        call.emit(wire.StartEvent(call.request_id, 2, 2))
        call.emit(wire.TokensEvent(call.request_id, 0, (7, 8)))
        call.complete(result=success_result(call, tokens=(7, 8)))

        self.assertEqual(job.events.get(timeout=1.0), ("start", None))
        self.assertEqual(job.events.get(timeout=1.0), ("text", "7"))
        self.assertEqual(job.events.get(timeout=1.0), ("text", "8"))
        kind, result = job.events.get(timeout=1.0)
        self.assertEqual(kind, "done")
        self.assertEqual(result.reason, "stop")
        self.assertEqual(result.cache.status, "hit")
        self.assertEqual(result.prefill_tokens, 2)
        self.assertEqual(result.start_to_first_token_ms, 1.25)
        self.assertEqual(result.first_token_to_done_ms, 2.5)
        self.assertEqual(result.request_wall_ms, 4.0)
        self.assertEqual(constraint.committed, [(7, 8)])
        self.assertEqual(constraint.finish_calls, 1)
        self.assertFalse(transport.active)

    def test_finalized_request_returns_its_image_budget(self):
        factory = FakeFactory()
        transport, _runtime = self.make_transport(
            runtime.MultiplexedRuntime(
                process_factory=factory, pending_limit=4, eager_start=True
            )
        )
        self.enterContext(mock.patch.object(images.ImageCache, "BUDGET_BYTES", 0))
        self.enterContext(
            mock.patch.object(images.ImageCache, "REQUEST_BUDGET_BYTES", 4)
        )
        cache = images.ImageCache()
        job = make_job()
        job.image_owner = cache.request_batch()
        job.image_owner.append(images.PreparedImage(2, 2, b"abcd", 0, 0))
        self.assertEqual(cache.stats()["request_bytes"], 4)

        transport.submit(job)
        process = factory.processes[0]
        frame = process.stdin.wait_for(wire.RequestFrame)[0]
        process.send(wire.StartEvent(frame.request_id, 0, 0))
        process.send(
            wire.DoneEvent(
                frame.request_id, wire.FinishReason.STOP, 4, 0, 100, 200, 350, ()
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
        transport, _runtime = self.make_transport(DrivenRuntime("complete_inline"))
        job = make_job()

        transport.submit(job)
        self.assertEqual(self.terminal(job)[0], "done")
        self.assertFalse(transport.active)

    def test_completion_then_submit_error_delivers_exactly_one_terminal(self):
        transport, _runtime = self.make_transport(DrivenRuntime("complete_then_raise"))
        job = make_job()

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
                job = make_job()

                transport.submit(job)
                self.assertEqual(runtime_client.submit_attempts, 1)
                kind, error = self.terminal(job)
                self.assertEqual(kind, "error")
                self.assertEqual((error.status, error.code), (503, "engine_recovering"))
                self.assertFalse(transport.active)

    def test_pending_limit_does_not_leave_active_state(self):
        transport, _runtime = self.make_transport(DrivenRuntime("pending_limit"))
        job = make_job()

        self.assertIsNone(transport.submit(job))
        self.assertFalse(transport.active)
        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(
            (error.status, error.message, error.code),
            (503, "request queue is full", "frontend_overloaded"),
        )

    def test_constraint_callback_error_cancels_and_maps_to_bad_request(self):
        constraint = FakeConstraint(
            commit_error=api_errors.ConstraintError("invalid token")
        )
        transport, runtime = self.make_transport()
        job = make_job(constraint=constraint)
        transport.submit(job)
        call = runtime.calls[0]

        call.emit(wire.TokensEvent(call.request_id, 0, (7,)))
        self.assertEqual(runtime.cancel_count, 1)
        call.complete(
            result=success_result(call, reason=wire.FinishReason.CANCELLED, tokens=(7,))
        )

        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(error.status, 400)
        self.assertEqual(error.code, "constraint_error")

    def test_trailing_tokens_the_grammar_rejects_fail_the_request(self):
        constraint = FakeConstraint(
            finish_error=api_errors.ConstraintError("invalid token")
        )
        transport, runtime = self.make_transport()
        job = make_job(constraint=constraint)
        transport.submit(job)
        call = runtime.calls[0]

        call.emit(wire.TokensEvent(call.request_id, 0, (7,)))
        call.complete(result=success_result(call, tokens=(7,)))

        kind, error = self.terminal(job)
        self.assertEqual(kind, "error")
        self.assertEqual(error.status, 400)
        self.assertEqual(error.code, "constraint_error")

    def test_cancelled_request_skips_the_trailing_grammar_check(self):
        constraint = FakeConstraint(
            finish_error=api_errors.ConstraintError("invalid token")
        )
        transport, runtime = self.make_transport()
        job = make_job(constraint=constraint)
        transport.submit(job)
        call = runtime.calls[0]

        call.emit(wire.TokensEvent(call.request_id, 0, (7,)))
        call.complete(
            result=success_result(call, reason=wire.FinishReason.CANCELLED, tokens=(7,))
        )

        kind, result = self.terminal(job)
        self.assertEqual(kind, "done")
        self.assertEqual(result.reason, "cancelled")
        self.assertEqual(constraint.finish_calls, 0)

    def test_token_constraint_consumes_committed_batches_only_at_mask_or_finish(self):
        matcher = FakeMatcher()
        constraint = generation_constraints.TokenConstraint(matcher, None)
        constraint.commit((7, 8))
        constraint.commit((9,))
        self.assertEqual(matcher.consumed, [])
        constraint.finish()
        self.assertEqual(matcher.consumed, [(7, 8), (9,)])

    def test_cancel_is_idempotent(self):
        transport, runtime = self.make_transport()
        job = make_job()
        transport.submit(job)

        transport.cancel(job)
        transport.cancel(job)
        self.assertTrue(job.cancelled.is_set())
        self.assertEqual(runtime.cancel_count, 1)

    def test_cancel_during_submit_is_sent_when_call_becomes_available(self):
        runtime = DrivenRuntime("block_after_call")
        transport, _runtime = self.make_transport(runtime)
        job = make_job()
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
        job = make_job()
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
        job = make_job()
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
        runtime.status_event = native_peer.status_event(
            ready=False, memory_pressure="critical"
        )
        self.assertFalse(transport.is_ready())

        runtime.status_event = native_peer.status_event(ready="false")
        self.assertFalse(transport.is_ready())

        runtime.status_event = native_peer.status_event(memory_pressure=None)
        self.assertTrue(transport.is_ready())

        runtime.status_event = native_peer.status_event()
        self.assertTrue(transport.is_ready())
        self.assertEqual(runtime.status_calls, 5)

        runtime.status_event = wire.StatusJsonEvent(1, b'{"schema_version":3}')
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
            self.assertEqual(status["schema_version"], wire.STATUS_SCHEMA_VERSION)
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
                runtime.RequestFailed(1, b"deadline_exceeded", b"deadline elapsed"),
                (504, "request_timeout"),
            ),
            (
                runtime.RequestFailed(1, b"temporarily_busy", b"retry", retryable=True),
                (503, "temporarily_busy"),
            ),
            (
                runtime.RequestFailed(1, b"invalid_count", b"too many score tokens"),
                (400, "invalid_count"),
            ),
            (runtime.EngineUnhealthy("gpu failed"), (503, "runtime_unavailable")),
            (runtime.ProtocolFatal("bad frame"), (503, "runtime_unavailable")),
            (
                runtime.MaskComputationFailed("grammar has no valid token"),
                (400, "constraint_error"),
            ),
            (
                runtime.MaskComputationFailed("queue is full", retryable=True),
                (503, "runtime_busy"),
            ),
        )
        for native, expected in cases:
            with self.subTest(native=native):
                error = backend_api.NativeBackend._api_error(native)
                self.assertEqual((error.status, error.code), expected)

    def test_fakes_offer_the_interfaces_they_stand_in_for(self):
        for fake, real in (
            (DrivenRuntime(), runtime.MultiplexedRuntime),
            (FakeConstraint(), generation_constraints.TokenConstraint),
        ):
            with self.subTest(fake=type(fake).__name__):
                self.assertEqual(interface_drift(fake, real), [])


class ByteLevelTestTokenizer:
    def __init__(self, raw_tokens):
        alphabet = byte_alphabet()
        vocabulary = {"[UNK]": 0, "<eos>": 1}
        self.token_ids = []
        for raw in raw_tokens:
            token = "".join(alphabet[value] for value in raw)
            token_id = vocabulary.setdefault(token, len(vocabulary))
            self.token_ids.append(token_id)
        self.backend_tokenizer = Tokenizer(
            models.WordLevel(vocabulary, unk_token="[UNK]")
        )
        self.backend_tokenizer.add_special_tokens(["<eos>"])
        self.backend_tokenizer.decoder = decoders.ByteLevel()
        self.eos_token_id = vocabulary["<eos>"]

    def decode(self, token_ids, skip_special_tokens=True, **_kwargs):
        return self.backend_tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )


class CallbackStreamerTests(unittest.TestCase):
    def test_callback_streamer_emits_no_space_text_before_end(self):
        for raw_tokens in (
            [b"abcdefghijklmnop", b"qrst", b"uv", b"wx"],
            [b'{"', b"abcdefgh", b"ij", b'":', b"1234", b"567890", b"}"],
        ):
            tokenizer = ByteLevelTestTokenizer(raw_tokens)
            chunks = []
            streamer = backend_api.CallbackStreamer(tokenizer, chunks.append)
            for token_id in tokenizer.token_ids:
                streamer.put_tokens([token_id])
            expected = tokenizer.decode(tokenizer.token_ids)
            self.assertGreater(len(chunks), 1)
            self.assertEqual("".join(chunks), expected)
            before_end = list(chunks)
            streamer.end()
            self.assertEqual(chunks, before_end)

    def test_reasoning_tokens_end_where_the_splitter_ends_reasoning(self):
        # A call-open token ends the reasoning where a call may follow and its
        # text opens one; a </think> ends it always.
        think_end = tool_schema.THINK_END_TOKEN_ID
        call = tool_schema.TOOL_CALL_OPEN_TOKEN_ID
        texts = {
            1: "a",
            2: " b",
            3: "\n<function=",
            call: "<tool_call>",
            think_end: "</think>",
        }
        tokenizer = SimpleNamespace(
            backend_tokenizer=None,
            decode=lambda ids: "".join(texts[token] for token in ids),
        )
        for token_ids, tool_calls, expected in (
            ([1, 2, think_end, 1], False, 2),
            ([1, call, 3, think_end], False, 3),
            ([1, call, 3, think_end], True, 1),
            ([1, call, 2, think_end], True, 3),
            ([1, 2], True, 2),
        ):
            with self.subTest(token_ids=token_ids, tool_calls=tool_calls):
                streamer = backend_api.CallbackStreamer(tokenizer, lambda *_: None)
                streamer.token_ids = token_ids
                self.assertEqual(
                    streamer.count_reasoning_tokens(True, tool_calls), expected
                )
                self.assertEqual(streamer.count_reasoning_tokens(False, True), 0)

    def test_callback_streamer_matches_one_shot_for_random_token_batches(self):
        family = "👨‍👩‍👧‍👦".encode()
        cases = (
            [bytes([value]) for value in family],
            [
                b"e",
                b"\xcc",
                b"\x81",
                b" + ",
                *[bytes([value]) for value in "中文".encode()],
            ],
            [b"prefix", b" ", b"suffix"],
            [b"\xf0", b"\x9f"],
        )
        for case_index, raw_tokens in enumerate(cases):
            tokenizer = ByteLevelTestTokenizer(raw_tokens)
            token_ids = list(tokenizer.token_ids)
            token_ids.insert(max(1, len(token_ids) // 2), tokenizer.eos_token_id)
            # The one-shot decode, less a character the output ends inside.
            expected = tokenizer.decode(token_ids).rstrip("\ufffd")
            for seed in range(50):
                randomizer = random.Random(seed)
                chunks = []
                streamer = backend_api.CallbackStreamer(tokenizer, chunks.append)
                offset = 0
                while offset < len(token_ids):
                    size = randomizer.randint(1, min(7, len(token_ids) - offset))
                    streamer.put_tokens(token_ids[offset : offset + size])
                    offset += size
                streamer.end()
                self.assertEqual("".join(chunks), expected, (raw_tokens, seed))
                if case_index == 0:
                    self.assertNotIn("�", "".join(chunks))

        tokenizer = ByteLevelTestTokenizer([bytes([value]) for value in range(256)])
        for seed in range(100):
            randomizer = random.Random(seed)
            token_ids = [
                tokenizer.token_ids[randomizer.randrange(256)]
                for _ in range(randomizer.randint(1, 128))
            ]
            for _ in range(randomizer.randrange(4)):
                token_ids.insert(
                    randomizer.randrange(len(token_ids) + 1), tokenizer.eos_token_id
                )
            chunks = []
            streamer = backend_api.CallbackStreamer(tokenizer, chunks.append)
            offset = 0
            while offset < len(token_ids):
                size = randomizer.randint(1, min(11, len(token_ids) - offset))
                streamer.put_tokens(token_ids[offset : offset + size])
                offset += size
            streamer.end()
            self.assertEqual(
                "".join(chunks), tokenizer.decode(token_ids).rstrip("\ufffd"), seed
            )

    def test_callback_streamer_drops_a_character_the_output_ends_inside(self):
        # "é" is b"\xc3\xa9". Output that ends after its first byte decodes it
        # to U+FFFD, which is dropped; one that text follows stays.
        tokenizer = ByteLevelTestTokenizer([b"caf", b"\xc3", b" noir"])
        caf, lead, noir = tokenizer.token_ids
        for incremental in (True, False):
            for token_ids, expected in (
                ([caf, lead], "caf"),
                ([caf, lead, noir, lead], "caf\ufffd noir"),
            ):
                chunks = []
                streamer = backend_api.CallbackStreamer(tokenizer, chunks.append)
                if not incremental:
                    # As an invalid DecodeStream prefix leaves it: the end
                    # decodes the whole output.
                    streamer.decode_stream = None
                streamer.put_tokens(token_ids)
                streamer.end()
                self.assertEqual("".join(chunks), expected)

    def test_callback_streamer_reconciles_an_invalid_decoder_prefix(self):
        tokenizer = ByteLevelTestTokenizer([b'{"', b"value", b'":', b"123", b"}"])
        chunks = []
        streamer = backend_api.CallbackStreamer(tokenizer, chunks.append)
        streamer.decode_stream = SimpleNamespace(
            step=mock.Mock(side_effect=ValueError("Invalid prefix encountered"))
        )
        streamer.put_tokens(tokenizer.token_ids)
        self.assertEqual(chunks, [])
        streamer.end()
        self.assertEqual("".join(chunks), tokenizer.decode(tokenizer.token_ids))

    def test_callback_streamer_stops_across_token_boundaries(self):
        tokenizer = ByteLevelTestTokenizer([b"alpha <ST", b"OP>hidden", b" trailing"])
        chunks = []
        stopped = []
        streamer = backend_api.CallbackStreamer(
            tokenizer,
            chunks.append,
            ("<STOP>", "unused"),
            lambda: stopped.append(True),
        )
        streamer.put_tokens(tokenizer.token_ids)
        streamer.end()
        self.assertEqual("".join(chunks), "alpha ")
        self.assertEqual(stopped, [True])
        self.assertEqual(streamer.stop_sequence, "<STOP>")


if __name__ == "__main__":
    unittest.main()

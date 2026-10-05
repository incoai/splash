import dataclasses
import math
import os
import select
import struct
import threading
import time
import typing
import unittest
import weakref
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from dev.tests import native_peer
from dev.tests.server_fixtures import (
    READY,
    FakeFactory,
    FakeProcess,
    request,
    send_success,
)
from server import crash_trace
from server import protocol as wire
from server import runtime as engine_runtime


def f32_sampling(sampling):
    """The sampling a request frame carries: its float fields as f32."""
    hints = typing.get_type_hints(wire.SamplingParameters)
    return dataclasses.replace(
        sampling,
        **{
            field.name: struct.unpack(
                "<f", struct.pack("<f", getattr(sampling, field.name))
            )[0]
            for field in dataclasses.fields(wire.SamplingParameters)
            if hints[field.name] is float
        },
    )


READY_STATUS = b'{"schema_version":%d,"ready":true}' % wire.STATUS_SCHEMA_VERSION


def answer_status(process, message):
    process.send(wire.StatusJsonEvent(message.correlation_id, READY_STATUS))


def fast_liveness_probe(test):
    """Probes the loop every 50 ms and fails it after 200 ms without an answer."""
    test = mock.patch.object(engine_runtime, "STATUS_ANSWER_LIMIT_SECONDS", 0.2)(test)
    return mock.patch.object(
        engine_runtime.MultiplexedRuntime, "_liveness_interval_seconds", 0.05
    )(test)


class RuntimeTests(unittest.TestCase):
    def test_direct_admission_and_out_of_order_demultiplexing(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory,
            pending_limit=4,
            eager_start=True,
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]

        requests = (
            request(
                100,
                priority=wire.RequestPriority.FOREGROUND,
                deadline=9.0,
                seed=55,
            ),
            request(
                200,
                sampling=wire.SamplingParameters(0.7, 0.9, 16, 0.5, -0.25, 1.1, 0.05),
                seed=66,
            ),
            request(
                300,
                constraint=wire.ConstraintMode.TOKEN_MASK,
                mask_provider=lambda event: array(
                    "I", (1,) * (event.words_per_mask * event.mask_rows)
                ).tobytes(),
            ),
            request(400, priority=wire.RequestPriority.BACKGROUND),
        )
        callbacks = []
        event_types = {}
        starts = {}
        streamed = {}
        callback_lock = threading.Lock()
        callbacks_complete = threading.Event()

        def submit(index):
            def event(call, message):
                with callback_lock:
                    event_types.setdefault(call.request_id, []).append(type(message))
                    if isinstance(message, wire.StartEvent):
                        starts[call.request_id] = message
                    if isinstance(message, wire.TokensEvent):
                        streamed.setdefault(call.request_id, []).extend(message.tokens)

            def completed(call):
                with callback_lock:
                    callbacks.append(call.request_id)
                    if len(callbacks) == len(requests):
                        callbacks_complete.set()

            return runtime.submit(
                requests[index],
                on_event=event,
                on_complete=completed,
            )

        submitted_wall = time.time_ns() // 1000
        submitted_at = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            calls = list(executor.map(submit, range(4)))
        written_wall = time.time_ns() // 1000

        frames = process.stdin.wait_for(wire.RequestFrame, count=4)
        self.assertEqual(runtime.pending_count, 4)
        self.assertEqual(
            {frame.request_id for frame in frames}, {c.request_id for c in calls}
        )
        with mock.patch.object(wire, "serialize_message") as serialize:
            with self.assertRaises(engine_runtime.PendingLimitExceeded):
                runtime.submit(request(500))
            serialize.assert_not_called()

        expected = {item.frame.prompt_tokens: item for item in requests}
        for frame, raw in process.stdin.records_of(wire.RequestFrame):
            source = expected[frame.prompt_tokens]
            self.assertEqual(frame.priority, source.frame.priority)
            self.assertEqual(
                raw[wire.FRAME_HEADER_BYTES + 8], int(source.frame.priority)
            )
            # Stamped once, at submission, from the monotonic deadline.
            self.assertGreater(frame.remaining_deadline_micros, 0)
            self.assertLessEqual(
                frame.remaining_deadline_micros,
                (source.deadline - submitted_at) * 1_000_000,
            )
            self.assertLessEqual(
                submitted_wall,
                frame.absolute_deadline_unix_micros - frame.remaining_deadline_micros,
            )
            self.assertLessEqual(
                frame.absolute_deadline_unix_micros - frame.remaining_deadline_micros,
                written_wall,
            )
            self.assertEqual(
                frame.logical_max_output_tokens,
                source.frame.logical_max_output_tokens,
            )
            self.assertEqual(frame.sampling, f32_sampling(source.frame.sampling))
            self.assertEqual(frame.seed, source.frame.seed)
            self.assertEqual(frame.constraint, source.frame.constraint)

        reverse_calls = list(reversed(calls))
        for index, call in enumerate(reverse_calls):
            send_success(
                process,
                call,
                lane=index,
                tokens=(1000 + index, 2000 + index, 3000 + index),
            )

        for index, call in enumerate(reverse_calls):
            call.result(1.0)
            self.assertEqual(
                streamed[call.request_id], [1000 + index, 2000 + index, 3000 + index]
            )
            self.assertEqual(starts[call.request_id].lane, index)
        self.assertTrue(callbacks_complete.wait(1.0))
        self.assertEqual(callbacks, [call.request_id for call in reverse_calls])
        for call in calls:
            self.assertEqual(
                event_types[call.request_id],
                [wire.StartEvent, wire.TokensEvent, wire.TokensEvent],
            )
        self.assertEqual(runtime.pending_count, 0)
        self.assertTrue(runtime.ready)
        self.assertEqual(runtime.restart_count, 0)

    def test_result_is_available_before_completion_callback_finishes(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        callback_entered = threading.Event()
        callback_release = threading.Event()
        callback_complete = threading.Event()

        def completed(call):
            callback_entered.set()
            if callback_release.wait(5.0):
                callback_complete.set()

        call = runtime.submit(request(100), on_complete=completed)
        send_success(factory.processes[0], call)
        try:
            self.assertTrue(callback_entered.wait(1.0))
            self.assertEqual(call.result(1.0).completion_tokens, 3)
            self.assertFalse(callback_complete.is_set())
        finally:
            callback_release.set()
        self.assertTrue(callback_complete.wait(1.0))

    def test_tokens_require_start_and_exactly_contiguous_offsets(self):
        cases = (
            ("before_start", (), 0, (10,), "before StartEvent"),
            ("gap", ((0, (10,)),), 2, (11,), "leaves a gap"),
            (
                "overlap",
                ((0, (10, 11)),),
                1,
                (12,),
                "overlaps or duplicates",
            ),
            (
                "duplicate",
                ((0, (10,)),),
                0,
                (10,),
                "overlaps or duplicates",
            ),
        )
        for name, valid_chunks, bad_offset, bad_tokens, message in cases:
            with self.subTest(name=name):
                factory = FakeFactory()
                runtime = engine_runtime.MultiplexedRuntime(
                    process_factory=factory, eager_start=True
                )
                process = factory.processes[0]
                call = runtime.submit(request(11))
                try:
                    if name != "before_start":
                        process.send(wire.StartEvent(call.request_id, 0, 0))
                    for offset, tokens in valid_chunks:
                        process.send(wire.TokensEvent(call.request_id, offset, tokens))
                    process.send(
                        wire.TokensEvent(call.request_id, bad_offset, bad_tokens)
                    )
                    with self.assertRaisesRegex(engine_runtime.ProtocolFatal, message):
                        call.result(1.0)
                    self.assertFalse(runtime.ready)
                finally:
                    runtime.close()

    def test_done_requires_consistent_order_counts_and_finish_reason(self):
        cases = (
            ("before_start", False, (), wire.FinishReason.STOP, 2, 0, "before Start"),
            (
                "uncancelled_before_start",
                False,
                (),
                wire.FinishReason.CANCELLED,
                2,
                0,
                "before Start",
            ),
            ("prompt_count", True, (), wire.FinishReason.STOP, 1, 0, "prompt count"),
            (
                "completion_count",
                True,
                ((0, (10, 11)),),
                wire.FinishReason.STOP,
                2,
                1,
                "streamed count 2",
            ),
            (
                "logical_max",
                True,
                (),
                wire.FinishReason.STOP,
                2,
                33,
                "exceeds logical maximum 32",
            ),
            (
                "short_length",
                True,
                ((0, (10,)),),
                wire.FinishReason.LENGTH,
                2,
                1,
                "expected logical maximum 32",
            ),
        )
        for (
            name,
            started,
            chunks,
            reason,
            prompt_count,
            completion_count,
            message,
        ) in cases:
            with self.subTest(name=name):
                factory = FakeFactory()
                runtime = engine_runtime.MultiplexedRuntime(
                    process_factory=factory, eager_start=True
                )
                process = factory.processes[0]
                call = runtime.submit(request(12))
                try:
                    if started:
                        process.send(wire.StartEvent(call.request_id, 0, 0))
                    for offset, tokens in chunks:
                        process.send(wire.TokensEvent(call.request_id, offset, tokens))
                    process.send(
                        wire.DoneEvent(
                            call.request_id,
                            reason,
                            prompt_count,
                            completion_count,
                            10,
                            20,
                            30,
                            (),
                        )
                    )
                    with self.assertRaisesRegex(engine_runtime.ProtocolFatal, message):
                        call.result(1.0)
                    self.assertFalse(runtime.ready)
                finally:
                    runtime.close()

    def test_length_done_at_logical_max_is_valid(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(request(13, logical_max_output_tokens=3))
        process.send(wire.StartEvent(call.request_id, 0, 0))
        process.send(wire.TokensEvent(call.request_id, 0, (20, 21)))
        process.send(wire.TokensEvent(call.request_id, 2, (22,)))
        process.send(
            wire.DoneEvent(
                call.request_id, wire.FinishReason.LENGTH, 2, 3, 10, 20, 30, ()
            )
        )

        result = call.result(1.0)
        self.assertEqual(result.completion_tokens, 3)
        self.assertEqual(result.reason, wire.FinishReason.LENGTH)

    def test_stop_done_at_logical_max_is_valid(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(request(14, logical_max_output_tokens=3))
        send_success(process, call, tokens=(20, 21, 22))

        result = call.result(1.0)
        self.assertEqual(result.completion_tokens, 3)
        self.assertEqual(result.reason, wire.FinishReason.STOP)

    def test_cancel_is_a_correlated_frame(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        events = []
        call = runtime.submit(
            request(10), on_event=lambda _call, event: events.append(event)
        )

        self.assertTrue(call.cancel())
        self.assertFalse(call.cancel())
        cancel = process.stdin.wait_for(wire.CancelFrame)[0]
        self.assertEqual(cancel.request_id, call.request_id)
        process.send(
            wire.DoneEvent(
                call.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 10, ()
            )
        )
        result = call.result(1.0)
        self.assertEqual(result.reason, wire.FinishReason.CANCELLED)
        self.assertEqual(events, [])
        self.assertEqual(result.completion_tokens, 0)
        self.assertFalse(call.cancel())

    def test_generation_prompt_tokens_reach_the_request_frame(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        runtime.submit(request(10, generation_prompt_tokens=1))
        frame = factory.processes[0].stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(frame.generation_prompt_tokens, 1)

    def test_score_request_passes_slots_and_returns_option_logits(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(
            request(10, logical_max_output_tokens=0, score_tokens=(101, 202, 303))
        )
        frame = process.stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(frame.score_tokens, (101, 202, 303))
        self.assertEqual(frame.logical_max_output_tokens, 0)
        process.send(wire.StartEvent(call.request_id, 0, 0))
        process.send(
            wire.DoneEvent(
                call.request_id,
                wire.FinishReason.STOP,
                2,
                0,
                100,
                0,
                350,
                (1.5, -2.25, 0.5),
            )
        )
        result = call.result(1.0)
        self.assertEqual(result.completion_tokens, 0)
        self.assertEqual(result.option_logits, (1.5, -2.25, 0.5))

    def test_score_done_rejects_invalid_terminal_results(self):
        for reason, logits, decode_micros in (
            (wire.FinishReason.STOP, (1.5, -2.25), 0),
            (wire.FinishReason.LENGTH, (), 0),
            (wire.FinishReason.STOP, (1.5, -2.25, 0.5), 1),
        ):
            with self.subTest(
                reason=reason, logits=logits, decode_micros=decode_micros
            ):
                factory = FakeFactory()
                runtime = engine_runtime.MultiplexedRuntime(
                    process_factory=factory, eager_start=True
                )
                process = factory.processes[0]
                call = runtime.submit(
                    request(
                        10,
                        logical_max_output_tokens=0,
                        score_tokens=(101, 202, 303),
                    )
                )
                try:
                    process.send(wire.StartEvent(call.request_id, 0, 0))
                    process.send(
                        wire.DoneEvent(
                            call.request_id,
                            reason,
                            2,
                            0,
                            100,
                            decode_micros,
                            350,
                            logits,
                        )
                    )
                    with self.assertRaises(engine_runtime.ProtocolFatal):
                        call.result(1.0)
                    self.assertFalse(runtime.ready)
                finally:
                    runtime.close()

    def test_generation_done_with_option_logits_is_fatal(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        process = factory.processes[0]
        call = runtime.submit(request(10))
        try:
            process.send(wire.StartEvent(call.request_id, 0, 0))
            process.send(
                wire.DoneEvent(
                    call.request_id,
                    wire.FinishReason.STOP,
                    2,
                    0,
                    100,
                    0,
                    350,
                    (1.5, -2.25),
                )
            )
            with self.assertRaisesRegex(engine_runtime.ProtocolFatal, "option logits"):
                call.result(1.0)
            self.assertFalse(runtime.ready)
        finally:
            runtime.close()

    def test_cancelled_score_done_returns_no_logits(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(
            request(10, logical_max_output_tokens=0, score_tokens=(101, 202))
        )
        self.assertTrue(call.cancel())
        process.send(
            wire.DoneEvent(
                call.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 10, ()
            )
        )
        result = call.result(1.0)
        self.assertEqual(result.reason, wire.FinishReason.CANCELLED)
        self.assertEqual(result.option_logits, ())

    def test_generate_is_the_blocking_convenience_over_direct_submit(self):
        def handler(process, message):
            if not isinstance(message, wire.RequestFrame):
                return
            process.send(wire.StartEvent(message.request_id, 2, 1))
            process.send(wire.TokensEvent(message.request_id, 0, (41, 42)))
            process.send(
                wire.DoneEvent(
                    message.request_id,
                    wire.FinishReason.STOP,
                    len(message.prompt_tokens),
                    2,
                    10,
                    20,
                    35,
                    (),
                )
            )

        factory = FakeFactory(handler)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        events = []
        call = runtime.submit(
            request(15), on_event=lambda _call, event: events.append(event)
        )
        result = call.result(1.0)
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(events[0].matched_prompt_tokens, 1)

    def test_mask_response_preserves_both_ids_and_exact_word_count(self):
        provider_thread = []
        provider_called = threading.Event()

        def provider(event):
            provider_thread.append(threading.current_thread().name)
            provider_called.set()
            return array(
                "I", range(1, event.words_per_mask * event.mask_rows + 1)
            ).tobytes()

        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True, mask_workers=1
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(
            request(
                20,
                constraint=wire.ConstraintMode.TOKEN_MASK,
                mask_provider=provider,
            )
        )

        process.send(wire.MaskRequestEvent(call.request_id, 987, 2, (5, 6, 7)))
        response = process.stdin.wait_for(wire.MaskResponseFrame)[0]
        self.assertTrue(provider_called.is_set())
        self.assertTrue(provider_thread[0].startswith("splash-mask"))
        self.assertEqual(response.request_id, call.request_id)
        self.assertEqual(response.mask_request_id, 987)
        self.assertEqual(response.mask_words, array("I", range(1, 9)).tobytes())
        send_success(process, call)
        self.assertEqual(call.result(1.0).completion_tokens, 3)

    def test_bad_mask_size_cancels_and_fails_only_that_call(self):
        for mask, message in (
            (array("I", (1,)).tobytes(), "returned 4 bytes; expected 24$"),
            (None, "returned NoneType; expected 24 bytes$"),
        ):
            with self.subTest(message=message):
                factory = FakeFactory()
                runtime = engine_runtime.MultiplexedRuntime(
                    process_factory=factory, eager_start=True
                )
                self.addCleanup(runtime.close)
                process = factory.processes[0]
                bad = runtime.submit(
                    request(
                        30,
                        constraint=wire.ConstraintMode.TOKEN_MASK,
                        mask_provider=lambda _event, mask=mask: mask,
                    )
                )
                healthy = runtime.submit(request(40))

                process.send(wire.StartEvent(bad.request_id, 0, 0))
                process.send(wire.MaskRequestEvent(bad.request_id, 88, 2, (5, 6)))
                cancel = process.stdin.wait_for(wire.CancelFrame)[0]
                self.assertEqual(cancel.request_id, bad.request_id)
                process.send(
                    wire.DoneEvent(
                        bad.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 10, ()
                    )
                )
                with self.assertRaisesRegex(
                    engine_runtime.MaskComputationFailed, message
                ):
                    bad.result(1.0)
                send_success(process, healthy)
                self.assertEqual(healthy.result(1.0).completion_tokens, 3)
                self.assertTrue(runtime.ready)

    def test_token_mask_mode_and_provider_go_together(self):
        def provider(_event):
            return b""

        for constraint, mask_provider in (
            (wire.ConstraintMode.TOKEN_MASK, None),
            (wire.ConstraintMode.NONE, provider),
        ):
            with self.subTest(constraint=constraint):
                with self.assertRaisesRegex(ValueError, "exactly one mask provider"):
                    request(
                        10,
                        constraint=constraint,
                        mask_provider=mask_provider,
                    )

    def test_mask_request_for_unconstrained_request_is_protocol_fatal(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(request(10))
        process.send(wire.StartEvent(call.request_id, 0, 0))
        process.send(wire.MaskRequestEvent(call.request_id, 88, 2, ()))
        with self.assertRaisesRegex(
            engine_runtime.ProtocolFatal, "token mask for an unconstrained request"
        ):
            call.result(1.0)
        self.assertFalse(runtime.ready)

    def test_mask_failure_marks_the_call_cancelled_and_writes_one_cancel(self):
        computed = []

        def provider(event):
            computed.append(event.mask_request_id)
            return b""

        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(
            request(
                30,
                constraint=wire.ConstraintMode.TOKEN_MASK,
                mask_provider=provider,
            )
        )
        process.send(wire.StartEvent(call.request_id, 0, 0))
        process.send(wire.MaskRequestEvent(call.request_id, 88, 2, ()))
        process.stdin.wait_for(wire.CancelFrame)
        self.assertTrue(call.cancel_requested)
        self.assertFalse(call.cancel())
        process.send(wire.MaskRequestEvent(call.request_id, 89, 2, ()))
        process.send(
            wire.DoneEvent(
                call.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 10, ()
            )
        )
        with self.assertRaises(engine_runtime.MaskComputationFailed):
            call.result(1.0)
        self.assertEqual(len(process.stdin.messages(wire.CancelFrame)), 1)
        self.assertEqual(computed, [88])

    def test_event_callback_error_cancels_once_and_is_the_call_result(self):
        failures = [ValueError("first"), ValueError("second")]

        def on_event(_call, event):
            if isinstance(event, wire.TokensEvent):
                raise failures[event.sequence_offset]

        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(request(10), on_event=on_event)
        process.send(wire.StartEvent(call.request_id, 0, 0))
        process.send(wire.TokensEvent(call.request_id, 0, (20,)))
        process.stdin.wait_for(wire.CancelFrame)
        process.send(wire.TokensEvent(call.request_id, 1, (21,)))
        process.send(
            wire.DoneEvent(
                call.request_id, wire.FinishReason.CANCELLED, 2, 2, 0, 0, 10, ()
            )
        )
        self.assertEqual(call.result(1.0).reason, wire.FinishReason.CANCELLED)
        self.assertIs(call.callback_error, failures[0])
        self.assertIsNone(call.callback_error.__traceback__)
        self.assertEqual(len(process.stdin.messages(wire.CancelFrame)), 1)
        self.assertTrue(runtime.ready)

    def test_cancelled_call_never_writes_a_late_mask_response(self):
        provider_started = threading.Event()
        provider_release = threading.Event()
        provider_finished = threading.Event()

        def provider(event):
            provider_started.set()
            provider_release.wait(1.0)
            provider_finished.set()
            return array("I", (1,) * (event.words_per_mask * event.mask_rows)).tobytes()

        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True, mask_workers=1
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(
            request(
                31,
                constraint=wire.ConstraintMode.TOKEN_MASK,
                mask_provider=provider,
            )
        )
        process.send(wire.StartEvent(call.request_id, 0, 0))
        process.send(wire.MaskRequestEvent(call.request_id, 89, 2, (5, 6)))
        self.assertTrue(provider_started.wait(1.0))
        self.assertTrue(call.cancel())
        cancel = process.stdin.wait_for(wire.CancelFrame)[0]
        self.assertEqual(cancel.request_id, call.request_id)
        process.send(
            wire.DoneEvent(
                call.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 10, ()
            )
        )
        self.assertEqual(call.result(1.0).reason, wire.FinishReason.CANCELLED)
        provider_release.set()
        self.assertTrue(provider_finished.wait(1.0))
        time.sleep(0.02)
        self.assertEqual(process.stdin.messages(wire.MaskResponseFrame), [])
        self.assertTrue(runtime.ready)

    def test_cancel_churn_bounds_mask_queue_and_skips_cancelled_work(self):
        started, release = threading.Event(), threading.Event()
        calls_to_provider = []

        def provider(event):
            calls_to_provider.append(event.request_id)
            started.set()
            release.wait(5.0)
            return array("I", (1,) * (event.words_per_mask * event.mask_rows)).tobytes()

        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, pending_limit=2, eager_start=True, mask_workers=1
        )
        self.addCleanup(runtime.close)
        self.addCleanup(release.set)
        process = factory.processes[0]

        def submit_mask():
            call = runtime.submit(
                request(
                    31,
                    constraint=wire.ConstraintMode.TOKEN_MASK,
                    mask_provider=provider,
                )
            )
            call._record_start(wire.StartEvent(call.request_id, 0, 0))
            runtime._submit_mask(
                call, wire.MaskRequestEvent(call.request_id, 89, 2, (5, 6))
            )
            return call

        with mock.patch.object(
            runtime._mask_executor, "submit", wraps=runtime._mask_executor.submit
        ) as submit:
            first = submit_mask()
            self.assertTrue(started.wait(1.0))
            for _ in range(16):
                call = submit_mask()
                call.cancel()
                process.send(
                    wire.DoneEvent(
                        call.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 10, ()
                    )
                )
                try:
                    call.result(1.0)
                except engine_runtime.MaskComputationFailed:
                    pass  # Queue saturation fails only this request.
            self.assertLessEqual(submit.call_count, 2)
        release.set()
        runtime._mask_executor.submit(lambda: None).result(1.0)  # drain small queue
        self.assertEqual(calls_to_provider, [first.request_id])
        first.cancel()
        process.send(
            wire.DoneEvent(
                first.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 10, ()
            )
        )
        first.result(1.0)
        recovered = submit_mask()
        runtime._mask_executor.submit(lambda: None).result(1.0)
        self.assertIn(recovered.request_id, calls_to_provider)
        self.assertTrue(runtime.ready)

    def test_full_mask_queue_fails_the_request_as_retryable(self):
        started, release = threading.Event(), threading.Event()

        def provider(event):
            started.set()
            release.wait(5.0)
            return array("I", (1,) * (event.words_per_mask * event.mask_rows)).tobytes()

        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, pending_limit=1, eager_start=True
        )
        self.addCleanup(runtime.close)
        self.addCleanup(release.set)
        process = factory.processes[0]

        def constrained(token):
            call = runtime.submit(
                request(
                    token,
                    constraint=wire.ConstraintMode.TOKEN_MASK,
                    mask_provider=provider,
                )
            )
            process.send(wire.StartEvent(call.request_id, 0, 0))
            process.send(wire.MaskRequestEvent(call.request_id, 1, 2, ()))
            return call

        def cancelled(call):
            process.send(
                wire.DoneEvent(
                    call.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 10, ()
                )
            )

        abandoned = constrained(1)
        self.assertTrue(started.wait(1.0))
        # Cancellation frees the request slot; its mask job keeps running.
        abandoned.cancel()
        cancelled(abandoned)
        abandoned.result(1.0)
        waiting = constrained(3)
        process.stdin.wait_for(wire.CancelFrame, count=2)
        cancelled(waiting)
        with self.assertRaisesRegex(
            engine_runtime.MaskComputationFailed, "queue is full"
        ) as caught:
            waiting.result(1.0)
        self.assertTrue(caught.exception.retryable)
        self.assertTrue(runtime.ready)

    def test_correlated_status_and_expired_response(self):
        respond = threading.Event()

        def handler(process, message):
            if isinstance(message, wire.StatusRequestFrame) and respond.is_set():
                answer_status(process, message)

        factory = FakeFactory(handler)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]

        with self.assertRaises(TimeoutError):
            runtime.status(timeout=0.02)
        first = process.stdin.messages(wire.StatusRequestFrame)[0]
        process.send(
            wire.StatusJsonEvent(
                first.correlation_id,
                b'{"schema_version":%d,"late":true}' % wire.STATUS_SCHEMA_VERSION,
            )
        )
        respond.set()
        status = runtime.status(timeout=1.0)
        self.assertNotEqual(status.correlation_id, first.correlation_id)
        self.assertEqual(status.json, READY_STATUS)
        self.assertTrue(runtime.ready)

    @fast_liveness_probe
    def test_liveness_probe_fails_a_generation_whose_loop_stops_answering(self):
        alive = threading.Event()
        alive.set()

        def handler(process, message):
            if isinstance(message, wire.StatusRequestFrame) and alive.is_set():
                answer_status(process, message)

        factory = FakeFactory(handler)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        call = runtime.submit(request(1))
        # The reader thread keeps taking writes; only the loop stops.
        factory.processes[0].stdin.wait_for(wire.StatusRequestFrame, count=2)
        self.assertFalse(call.done)
        alive.clear()
        with self.assertRaisesRegex(
            engine_runtime.EngineUnhealthy, "did not answer status"
        ):
            call.result(2.0)
        self.assertFalse(runtime.ready)

    @fast_liveness_probe
    def test_liveness_probe_runs_only_while_calls_are_pending(self):
        def handler(process, message):
            if isinstance(message, wire.StatusRequestFrame):
                answer_status(process, message)

        factory = FakeFactory(handler)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        time.sleep(0.3)
        self.assertEqual(process.stdin.messages(wire.StatusRequestFrame), [])
        call = runtime.submit(request(1))
        process.stdin.wait_for(wire.StatusRequestFrame, count=2, timeout=0.5)
        send_success(process, call)
        call.result(1.0)
        # A probe that saw the call pending may still be writing.
        time.sleep(0.1)
        probes = len(process.stdin.messages(wire.StatusRequestFrame))
        time.sleep(0.3)
        self.assertEqual(len(process.stdin.messages(wire.StatusRequestFrame)), probes)
        self.assertTrue(runtime.ready)

    def test_status_with_correlation_zero_is_protocol_fatal(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        call = runtime.submit(request(10))
        factory.processes[0].send(wire.StatusJsonEvent(0, READY_STATUS))
        with self.assertRaisesRegex(
            engine_runtime.ProtocolFatal, "unknown correlation 0"
        ):
            call.result(1.0)
        self.assertFalse(runtime.ready)

    def test_request_failures_are_scoped(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        request_error = runtime.submit(request(50))
        healthy = runtime.submit(request(70))

        process.send(
            wire.ErrorEvent(
                wire.FailureClass.REQUEST_ERROR,
                request_error.request_id,
                True,
                b"deadline_exceeded",
                b"request deadline expired",
            )
        )
        send_success(process, healthy)

        with self.assertRaises(engine_runtime.RequestFailed) as caught:
            request_error.result(1.0)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.code, b"deadline_exceeded")
        self.assertEqual(healthy.result(1.0).completion_tokens, 3)
        self.assertTrue(runtime.ready)

    def test_unhealthy_fatal_and_eof_fail_all_then_restart_on_wait_ready(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)

        def relaunch():
            # A failed engine refuses new work and starts nothing by itself.
            with self.assertRaisesRegex(engine_runtime.EngineUnhealthy, "not ready"):
                runtime.submit(request(1))
            launched = len(factory.processes)
            self.assertTrue(runtime.wait_ready(1))
            self.assertEqual(len(factory.processes), launched + 1)

        first = factory.processes[0]
        first_calls = (runtime.submit(request(80)), runtime.submit(request(90)))
        first.send(
            wire.ErrorEvent(
                wire.FailureClass.ENGINE_UNHEALTHY,
                0,
                False,
                b"gpu_fault",
                b"Metal command buffer failed",
            )
        )
        for call in first_calls:
            with self.assertRaises(engine_runtime.EngineUnhealthy):
                call.result(1.0)
        self.assertFalse(runtime.ready)
        self.assertEqual(len(factory.processes), 1)

        relaunch()
        after_unhealthy = runtime.submit(request(100))
        second = factory.processes[1]
        self.assertEqual(runtime.restart_count, 1)
        self.assertEqual(len(factory.processes), 2)
        second.send(
            wire.ErrorEvent(
                wire.FailureClass.PROTOCOL_FATAL,
                0,
                False,
                b"bad_frame",
                b"stream cannot be trusted",
            )
        )
        with self.assertRaises(engine_runtime.ProtocolFatal):
            after_unhealthy.result(1.0)

        relaunch()
        after_fatal = runtime.submit(request(110))
        third = factory.processes[2]
        self.assertEqual(runtime.restart_count, 2)
        third.close_stdout()
        with self.assertRaises(engine_runtime.EngineUnhealthy):
            after_fatal.result(1.0)

        relaunch()
        after_eof = runtime.submit(request(120))
        fourth = factory.processes[3]
        self.assertEqual(runtime.restart_count, 3)
        send_success(fourth, after_eof)
        self.assertEqual(after_eof.result(1.0).completion_tokens, 3)

    def test_engine_failure_listener_runs_before_calls_end_with_served_seconds(
        self,
    ):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        order = []
        completed = threading.Event()
        runtime.on_engine_failure = lambda error, served: order.append(
            ("listener", str(error), served)
        )

        def complete(_call):
            order.append(("complete",))
            completed.set()

        call = runtime.submit(request(10), on_complete=complete)
        factory.processes[0].close_stdout()
        with self.assertRaises(engine_runtime.EngineUnhealthy):
            call.result(1.0)
        self.assertTrue(completed.wait(1.0))
        self.assertEqual(len(order), 2)
        kind, message, served = order[0]
        self.assertEqual((kind, message), ("listener", "native protocol reached EOF"))
        self.assertGreaterEqual(served, 0)
        self.assertEqual(order[1], ("complete",))
        # close() is not an engine failure.
        self.assertTrue(runtime.wait_ready(1))
        runtime.close()
        self.assertEqual(len(order), 2)

    def test_fatal_error_property_reports_changed_limits(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        self.assertIsNone(runtime.fatal_error)
        factory.initial_output = native_peer.serialize_event(
            wire.ReadyEvent(4, 65_536, False)
        )
        with mock.patch.object(runtime._crash_trace, "dump"):
            factory.processes[0].kill()
            with self.assertRaisesRegex(
                engine_runtime.EngineUnhealthy, "restart the Splash server"
            ):
                runtime.wait_ready(1)
        self.assertIsInstance(runtime.fatal_error, engine_runtime.EngineUnhealthy)
        self.assertIn("context window", str(runtime.fatal_error))

    def test_unanswered_status_with_fail_unanswered_fails_only_its_generation(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        failures = []
        runtime.on_engine_failure = lambda error, _served: failures.append(error)

        with self.assertRaises(TimeoutError):
            runtime.status(timeout=0.05)
        self.assertTrue(runtime.ready)
        self.assertEqual(failures, [])

        with self.assertRaises(TimeoutError):
            runtime.status(timeout=0.05, fail_unanswered=True)
        self.assertFalse(runtime.ready)
        (failure,) = failures
        self.assertIsInstance(failure, engine_runtime.EngineUnhealthy)
        self.assertEqual(
            str(failure), "native loop did not answer status within 0.05 s"
        )
        self.assertTrue(runtime.wait_ready(1))
        self.assertEqual(len(failures), 1)

        # A request whose engine fails before it expires ends with that
        # failure and never fails the next engine.
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(runtime.status, 0.5, fail_unanswered=True)
            factory.processes[1].stdin.wait_for(wire.StatusRequestFrame)
            factory.processes[1].kill()
            with self.assertRaisesRegex(engine_runtime.EngineUnhealthy, "EOF"):
                pending.result(1)
        self.assertTrue(runtime.wait_ready(1))
        self.assertEqual(
            [str(failure) for failure in failures[1:]], ["native protocol reached EOF"]
        )
        self.assertTrue(runtime.ready)

    def test_invalidation_before_registration_returns_the_admission_slot(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, pending_limit=1, eager_start=True
        )
        self.addCleanup(runtime.close)
        serialize = wire.serialize_message

        def serialize_then_fail(message):
            encoded = serialize(message)
            runtime._fail_generation(
                runtime._generation,
                engine_runtime.EngineUnhealthy("lost after admission"),
            )
            return encoded

        with mock.patch.object(wire, "serialize_message", serialize_then_fail):
            with self.assertRaises(engine_runtime.EngineUnhealthy):
                runtime.submit(request(125))
        self.assertEqual(runtime.pending_count, 0)
        self.assertEqual(runtime._admission_slots._value, runtime.pending_limit)

        self.assertTrue(runtime.wait_ready(1))
        replacement = runtime.submit(request(126))
        self.assertEqual(runtime.restart_count, 1)
        send_success(factory.processes[1], replacement)
        self.assertEqual(replacement.result(1.0).completion_tokens, 3)

    def test_eof_restarts_and_close_is_terminal(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        call = runtime.submit(request(130))
        factory.processes[0].close_stdout()
        with self.assertRaisesRegex(engine_runtime.EngineUnhealthy, "reached EOF"):
            call.result(1.0)
        self.assertTrue(runtime.wait_ready(1))
        replacement = runtime.submit(request(140))
        self.assertEqual(runtime.restart_count, 1)
        send_success(factory.processes[1], replacement)
        replacement.result(1.0)
        runtime.close()
        runtime.close()
        with self.assertRaises(engine_runtime.RuntimeClosed):
            runtime.submit(request(150))

    def test_failed_call_releases_its_request_when_the_caller_drops_it(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)

        class ImageOwner:
            """Stands in for the frontend's byte-budgeted image batch."""

        owner = ImageOwner()
        released = weakref.ref(owner)
        call = runtime.submit(request(160, image_owner=owner))
        del owner
        factory.processes[0].close_stdout()

        def finalize(call):
            # The frontend finalizer raises the failure and returns. Whatever
            # the runtime keeps of that failure must not keep this frame, and
            # with it the request's image owner, alive.
            try:
                call.result(1.0)
            except engine_runtime.EngineUnhealthy as failure:
                return str(failure)
            return None

        self.assertIn("reached EOF", finalize(call))
        # The reader thread hands out the terminal before it has finished
        # tearing the generation down; wait for it so only kept state is left.
        runtime._reader_thread.join(1.0)
        del call
        self.assertIsNone(released())

    def test_fatal_generation_writes_replay_trace_when_enabled(self):
        factory = FakeFactory()
        with (
            TemporaryDirectory() as temporary,
            mock.patch.dict(os.environ, {"SPLASH_CRASH_TRACE": "1"}),
            mock.patch.object(
                crash_trace,
                "DEFAULT_TRACE_DIRECTORY",
                Path(temporary),
            ),
        ):
            runtime = engine_runtime.MultiplexedRuntime(
                command=("fake-native", "serve-native"),
                process_factory=factory,
                eager_start=True,
            )
            call = runtime.submit(request(130))
            factory.processes[0].close_stdout()
            with self.assertRaisesRegex(engine_runtime.EngineUnhealthy, "reached EOF"):
                call.result(1.0)
            path = runtime.last_crash_trace
            self.assertIsNotNone(path)
            self.assertTrue(Path(path).is_file())
            runtime.close()

    def test_a_crash_trace_not_written_still_ends_the_generation(self):
        factory = FakeFactory()
        failures = []
        with (
            TemporaryDirectory() as temporary,
            mock.patch.dict(os.environ, {"SPLASH_CRASH_TRACE": "1"}),
            mock.patch.object(crash_trace, "DEFAULT_TRACE_DIRECTORY", Path(temporary)),
            mock.patch.object(crash_trace.base64, "b64encode", side_effect=MemoryError),
        ):
            runtime = engine_runtime.MultiplexedRuntime(
                command=("fake-native", "serve-native"),
                process_factory=factory,
                pending_limit=1,
                eager_start=True,
            )
            runtime.on_engine_failure = lambda error, _served: failures.append(
                str(error)
            )
            call = runtime.submit(request(160))
            factory.processes[0].close_stdout()
            with self.assertRaisesRegex(engine_runtime.EngineUnhealthy, "reached EOF"):
                call.result(1.0)
            runtime._reader_thread.join(1.0)
            self.assertEqual(failures, ["native protocol reached EOF"])
            self.assertIsNone(runtime.last_crash_trace)
            self.assertIsNotNone(factory.processes[0].poll())
            self.assertTrue(runtime._admission_slots.acquire(blocking=False))
            runtime._admission_slots.release()
            runtime.close()

    def test_close_callback_can_reenter_without_process_lock_deadlock(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        callback_finished = threading.Event()
        callback_error = []

        def completed(_call):
            try:
                runtime.submit(request(151))
            except engine_runtime.RuntimeClosed:
                callback_finished.set()
            except BaseException as error:
                callback_error.append(error)
                callback_finished.set()

        call = runtime.submit(request(150), on_complete=completed)
        closing = threading.Thread(target=runtime.close)
        closing.start()
        closing.join(1.0)
        self.assertFalse(closing.is_alive(), "close deadlocked in completion callback")
        self.assertTrue(callback_finished.is_set())
        self.assertEqual(callback_error, [])
        with self.assertRaises(engine_runtime.RuntimeClosed):
            call.result(1.0)

    def test_startup_accepts_only_binary_ready_event(self):
        invalid_header = b"ready text line\n".ljust(wire.FRAME_HEADER_BYTES, b"\0")
        factory = FakeFactory(initial_output=invalid_header)
        with self.assertRaises(engine_runtime.ProtocolFatal):
            engine_runtime.MultiplexedRuntime(
                process_factory=factory,
                startup_timeout=0.2,
                eager_start=True,
            )
        self.assertIsNotNone(factory.processes[0].poll())

    def test_concurrent_waiters_share_one_failed_startup(self):
        factory = FakeFactory(initial_output=b"")
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory,
            startup_timeout=0.2,
        )
        self.addCleanup(runtime.close)
        barrier = threading.Barrier(4)

        def wait_ready(_index):
            barrier.wait()
            with self.assertRaises(engine_runtime.EngineUnhealthy):
                runtime.wait_ready()

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(wait_ready, range(4)))
        elapsed = time.monotonic() - started
        self.assertEqual(len(factory.processes), 1)
        self.assertLess(elapsed, 0.5)

    def test_close_interrupts_an_in_progress_startup(self):
        factory = FakeFactory(initial_output=b"")
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory,
            startup_timeout=10.0,
        )
        failure = []

        def start():
            try:
                runtime.wait_ready()
            except BaseException as error:
                failure.append(error)

        thread = threading.Thread(target=start)
        thread.start()
        deadline = time.monotonic() + 1.0
        while not factory.processes and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(len(factory.processes), 1)
        started = time.monotonic()
        runtime.close()
        elapsed = time.monotonic() - started
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        self.assertLess(elapsed, 0.5)
        self.assertEqual(len(failure), 1)
        self.assertIsInstance(failure[0], engine_runtime.RuntimeClosed)
        self.assertIsNotNone(factory.processes[0].poll())

    def test_stop_process_kills_an_engine_that_ignores_sigterm(self):
        class StubbornProcess(FakeProcess):
            def terminate(self):
                pass  # stuck in a GPU command; SIGTERM is never acted on

        process = StubbornProcess(1)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=FakeFactory(),
            startup_timeout=10.0,
        )
        try:
            with mock.patch.object(
                engine_runtime.MultiplexedRuntime, "_shutdown_grace_seconds", 0.05
            ):
                started = time.monotonic()
                runtime._stop_process(process, None)
            self.assertEqual(process.poll(), -9)
            self.assertLess(time.monotonic() - started, 2.0)
        finally:
            runtime.close()

    def test_kill_ends_a_close_waiting_for_engine_teardown(self):
        waiting = threading.Event()

        class SlowTeardown(FakeProcess):
            def terminate(self):
                pass  # Still exiting gracefully; SIGTERM only asked it to.

            def wait(self, timeout=None):
                waiting.set()
                return super().wait(timeout)

        process = SlowTeardown(1000)
        self.addCleanup(process.kill)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=lambda: process, eager_start=True
        )
        closing = threading.Thread(target=runtime.close)
        closing.start()
        self.assertTrue(waiting.wait(1.0))
        runtime.kill()
        closing.join(1.0)
        self.assertFalse(closing.is_alive())
        self.assertEqual(process.poll(), -9)

    def test_kill_fallback_ends_an_engine_that_survives_terminate(self):
        class StubbornProcess(FakeProcess):
            def terminate(self):
                pass

        process = StubbornProcess(1)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=FakeFactory(),
            startup_timeout=10.0,
        )
        try:
            with mock.patch.object(
                engine_runtime.MultiplexedRuntime, "_shutdown_grace_seconds", 0.05
            ):
                runtime._arm_kill_fallback(process)
            deadline = time.monotonic() + 2.0
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(process.poll(), -9)
        finally:
            runtime.close()

    def test_wire_deadline_is_stamped_once_at_submission(self):
        wall = 1_700_000_000_000_000
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        with mock.patch.object(
            engine_runtime.time, "time_ns", return_value=wall * 1000
        ):
            call = runtime.submit(
                request(160, priority=wire.RequestPriority.FOREGROUND, deadline=1.25)
            )
            unbounded = runtime.submit(request(170, deadline=math.inf))
        first, second = factory.processes[0].stdin.wait_for(wire.RequestFrame, 2)
        self.assertEqual(first.request_id, call.request_id)
        self.assertEqual(first.priority, wire.RequestPriority.FOREGROUND)
        self.assertGreater(first.remaining_deadline_micros, 0)
        self.assertLessEqual(first.remaining_deadline_micros, 1_250_000)
        self.assertEqual(
            first.absolute_deadline_unix_micros - first.remaining_deadline_micros,
            wall,
        )
        self.assertEqual(second.request_id, unbounded.request_id)
        self.assertEqual(
            second.remaining_deadline_micros, engine_runtime._MAX_U64 - wall
        )
        self.assertEqual(second.absolute_deadline_unix_micros, engine_runtime._MAX_U64)

    def test_short_wait_leaves_shared_startup_running(self):
        factory = FakeFactory(initial_output=b"")
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory,
            startup_timeout=1,
            pending_limit=2,
        )
        self.addCleanup(runtime.close)
        with ThreadPoolExecutor(max_workers=2) as executor:
            short = executor.submit(runtime.wait_ready, 0.03)
            long = executor.submit(runtime.wait_ready, 1.0)
            with self.assertRaises(TimeoutError):
                short.result(0.5)
            self.assertEqual(len(factory.processes), 1)
            process = factory.processes[0]
            self.assertIsNone(process.poll())
            process.send(READY)
            self.assertTrue(long.result(0.5))
        call = runtime.submit(request(2))
        frames = process.stdin.messages(wire.RequestFrame)
        self.assertEqual([frame.prompt_tokens for frame in frames], [(2, 3)])
        send_success(process, call)
        call.result(0.5)
        self.assertEqual(runtime.pending_count, 0)

    def test_startup_expires_even_after_every_waiter_has_left(self):
        factory = FakeFactory(initial_output=b"")
        created = threading.Event()

        def create_process():
            process = factory()
            created.set()
            return process

        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=create_process, startup_timeout=2.0
        )
        self.addCleanup(runtime.close)
        with self.assertRaises(TimeoutError):
            runtime.wait_ready(0.01)
        self.assertTrue(created.wait(1.0))
        process = factory.processes[0]
        self.assertIsNone(process.poll())
        self.assertIsNotNone(process.wait(5.0))
        self.assertFalse(runtime.ready)

    def test_slow_process_factory_does_not_hold_a_waiter(self):
        release = threading.Event()
        factory = FakeFactory()

        def slow_factory():
            release.wait(1)
            return factory()

        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=slow_factory, startup_timeout=1
        )
        self.addCleanup(runtime.close)
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(runtime.wait_ready, 0.01)
                with self.assertRaises(TimeoutError):
                    future.result(0.5)
        finally:
            release.set()
        self.assertTrue(runtime.wait_ready(0.5))
        self.assertEqual(len(factory.processes), 1)

    def test_submit_on_a_not_ready_runtime_fails_fast_and_launches_nothing(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        started = time.monotonic()
        with self.assertRaisesRegex(engine_runtime.EngineUnhealthy, "not ready"):
            runtime.submit(request(1))
        self.assertLess(time.monotonic() - started, 0.5)
        with self.assertRaises(TimeoutError):
            runtime.submit(request(1, deadline=-1.0))
        self.assertEqual(factory.processes, [])
        self.assertEqual(runtime._admission_slots._value, runtime.pending_limit)

    def test_late_process_factory_cannot_publish_ready_after_startup_expiry(self):
        release = threading.Event()
        created = threading.Event()
        factory = FakeFactory()

        def slow_factory():
            release.wait(1)
            process = factory()
            created.set()
            return process

        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=slow_factory, startup_timeout=0.02
        )
        self.addCleanup(runtime.close)
        try:
            with self.assertRaisesRegex(
                engine_runtime.EngineUnhealthy, "ReadyEvent timed out"
            ):
                runtime.wait_ready()
        finally:
            release.set()
        self.assertTrue(created.wait(0.5))
        self.assertIsNotNone(factory.processes[0].wait(0.5))
        self.assertFalse(runtime.ready)

    def test_write_lock_waits_consume_request_and_status_deadlines(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        with runtime._write_lock:
            for action in (
                lambda: runtime.status(timeout=0.01),
                lambda: runtime.submit(request(1, deadline=0.01)),
            ):
                started = time.monotonic()
                with self.assertRaises(TimeoutError):
                    action()
                self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(runtime.ready)
        self.assertEqual(runtime.pending_count, 0)
        self.assertEqual(runtime._status_waiters, {})
        self.assertEqual(factory.processes[0].stdin.records, [])
        call = runtime.submit(request(2))
        send_success(factory.processes[0], call)
        call.result(0.5)

    def test_cancel_write_timeout_fails_generation(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        runtime._io_timeout_seconds = 0.02
        call = runtime.submit(request(1))
        with runtime._write_lock:
            self.assertTrue(call.cancel())
        with self.assertRaisesRegex(
            engine_runtime.EngineUnhealthy, "cancel write timed out"
        ):
            call.result(0.5)
        self.assertEqual(runtime.pending_count, 0)
        self.assertFalse(runtime.ready)

    def test_nonblocking_retries_and_short_writes_preserve_exact_frame(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        stdin = factory.processes[0].stdin
        write = stdin.write
        attempts = 0

        def short_write(data):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return None
            if attempts == 2:
                raise BlockingIOError()
            return write(data[:7])

        with mock.patch.object(stdin, "write", side_effect=short_write):
            call = runtime.submit(request(1))
        self.assertGreater(attempts, 3)
        (frame,) = stdin.messages(wire.RequestFrame)
        self.assertEqual(frame.request_id, call.request_id)
        self.assertEqual(frame.prompt_tokens, (1, 2))

    def test_partial_os_pipe_write_fails_generation_and_releases_all_calls(self):
        process = FakeProcess(1000)
        process.stdin.close()
        read_fd, write_fd = os.pipe()
        process.stdin = os.fdopen(write_fd, "wb", buffering=0)
        self.addCleanup(os.close, read_fd)
        replacement_factory = FakeFactory()
        processes = [process]

        def factory():
            return processes.pop() if processes else replacement_factory()

        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=True
        )
        self.addCleanup(runtime.close)
        runtime._io_timeout_seconds = 0.15
        failure_state = []

        def completed(_call):
            failure_state.append((runtime._write_lock.locked(), runtime.ready))
            # Re-entering submit from a terminal callback is refused at once.
            try:
                runtime.submit(request(2))
            except engine_runtime.EngineUnhealthy as error:
                failure_state.append(type(error))

        active = runtime.submit(request(1), on_complete=completed)
        os.read(read_fd, 65536)  # Drain just the first complete request.
        large = request(0, prompt_tokens=tuple(range(131072)), deadline=1.0)
        with ThreadPoolExecutor(max_workers=1) as executor:
            writer = executor.submit(runtime.submit, large)
            self.assertTrue(select.select([read_fd], [], [], 0.5)[0])
            # The pipe now holds a real partial frame; no reader drains it.
            with self.assertRaises(TimeoutError):
                runtime.status(timeout=0.01)
            with self.assertRaisesRegex(
                engine_runtime.EngineUnhealthy, "frame write timed out"
            ):
                writer.result(0.5)
        with self.assertRaises(engine_runtime.EngineUnhealthy):
            active.result(0.5)
        self.assertEqual(
            failure_state, [(False, False), engine_runtime.EngineUnhealthy]
        )
        self.assertIsNotNone(process.poll())
        self.assertTrue(runtime.wait_ready(1))
        recovered = runtime.submit(request(2))
        send_success(replacement_factory.processes[0], recovered)
        recovered.result(0.5)
        self.assertEqual(runtime.pending_count, 0)
        self.assertEqual(runtime.restart_count, 1)

    def test_request_deadline_does_not_abandon_a_started_frame(self):
        process = FakeProcess(1000)
        process.stdin.close()
        read_fd, write_fd = os.pipe()
        process.stdin = os.fdopen(write_fd, "wb", buffering=0)
        self.addCleanup(os.close, read_fd)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=lambda: process, eager_start=True
        )
        self.addCleanup(runtime.close)
        active = runtime.submit(request(1))
        os.read(read_fd, 65536)  # Drain just the first complete request.
        large = request(0, prompt_tokens=tuple(range(131072)), deadline=0.1)
        received = bytearray()
        with ThreadPoolExecutor(max_workers=1) as executor:
            writer = executor.submit(runtime.submit, large)
            self.assertTrue(select.select([read_fd], [], [], 0.5)[0])
            # The frame has started; let its request deadline pass mid-frame.
            time.sleep(0.3)
            self.assertFalse(writer.done())
            while not writer.done():
                if select.select([read_fd], [], [], 0.05)[0]:
                    received += os.read(read_fd, 1 << 20)
            call = writer.result(0)
        while select.select([read_fd], [], [], 0)[0]:
            received += os.read(read_fd, 1 << 20)
        frames = [
            message for message, _raw in native_peer.ClientFrameReader().feed(received)
        ]
        self.assertEqual([frame.request_id for frame in frames], [call.request_id])
        self.assertEqual(frames[0].prompt_tokens, large.frame.prompt_tokens)
        # The engine, not the transport, now fails the expired request alone.
        self.assertTrue(runtime.ready)
        self.assertFalse(active.done)
        self.assertEqual(runtime.pending_count, 2)


if __name__ == "__main__":
    unittest.main()

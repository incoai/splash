import os
import select
import subprocess
import threading
import time
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from server import crash_trace
from server import protocol as wire
from server import runtime as engine_runtime

READY_FEATURES = int(
    wire.ReadyFeature.CANCELLATION
    | wire.ReadyFeature.TOKEN_MASKS
    | wire.ReadyFeature.STATUS_JSON
    | wire.ReadyFeature.MULTIPLEXING
)


class FakeOutput:
    def __init__(self):
        self._condition = threading.Condition()
        self._buffer = bytearray()
        self._closed = False

    def feed(self, data):
        with self._condition:
            if self._closed:
                raise BrokenPipeError("fake stdout is closed")
            self._buffer.extend(data)
            self._condition.notify_all()

    def read1(self, size):
        with self._condition:
            while not self._buffer and not self._closed:
                self._condition.wait()
            if not self._buffer:
                return b""
            count = min(size, len(self._buffer))
            result = bytes(self._buffer[:count])
            del self._buffer[:count]
            return result

    read = read1

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class FakeInput:
    def __init__(self, process):
        self._process = process
        self._condition = threading.Condition()
        self._parser = wire.FrameParser()
        self._closed = False
        self.records = []
        self._read_fd, self._write_fd = os.pipe()

    def fileno(self):
        return self._write_fd

    def write(self, data):
        data = bytes(data)
        dispatched = []
        with self._condition:
            if self._closed:
                raise BrokenPipeError("fake stdin is closed")
            offset = 0
            while offset < len(data):
                step = self._parser.consume(memoryview(data)[offset:])
                if step.issue:
                    raise wire.ProtocolError(step.issue)
                if not step.consumed_bytes:
                    raise AssertionError("fake input parser made no progress")
                offset += step.consumed_bytes
                if step.frame:
                    message = wire.decode_frame(step.frame)
                    raw = wire.serialize_frame(step.frame)
                    self.records.append((message, raw))
                    dispatched.append(message)
            self._condition.notify_all()
        for message in dispatched:
            handler = self._process.input_handler
            if handler:
                handler(self._process, message)
        return len(data)

    def flush(self):
        return None

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            os.close(self._write_fd)
            os.close(self._read_fd)
            self._condition.notify_all()

    def messages(self, message_type):
        with self._condition:
            return [
                message
                for message, _raw in self.records
                if isinstance(message, message_type)
            ]

    def records_of(self, message_type):
        with self._condition:
            return [
                record for record in self.records if isinstance(record[0], message_type)
            ]

    def wait_for(self, message_type, count=1, timeout=1.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                matches = [
                    message
                    for message, _raw in self.records
                    if isinstance(message, message_type)
                ]
                if len(matches) >= count:
                    return matches
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"did not receive {count} {message_type.__name__} frames"
                    )
                self._condition.wait(remaining)


class FakeProcess:
    _pids = iter(range(50_000, 60_000))

    def __init__(self, engine_instance_id, input_handler=None, initial_output=None):
        self.pid = next(self._pids)
        self.engine_instance_id = engine_instance_id
        self.input_handler = input_handler
        self.stdout = FakeOutput()
        self.stdin = FakeInput(self)
        self._condition = threading.Condition()
        self._exit_code = None
        if initial_output is None:
            self.send(
                wire.ReadyEvent(
                    engine_instance_id,
                    4,
                    131_072,
                    READY_FEATURES,
                )
            )
        elif initial_output:
            self.stdout.feed(initial_output)

    def send(self, message):
        self.stdout.feed(wire.serialize_message(message))

    def close_stdout(self):
        self.stdout.close()

    def poll(self):
        with self._condition:
            return self._exit_code

    def terminate(self):
        self._exit(-15)

    def kill(self):
        self._exit(-9)

    def _exit(self, code):
        with self._condition:
            if self._exit_code is not None:
                return
            self._exit_code = code
            self._condition.notify_all()
        self.stdin.close()
        self.stdout.close()

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._exit_code is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise subprocess.TimeoutExpired("fake-native", timeout)
                self._condition.wait(remaining)
            return self._exit_code


class FakeFactory:
    def __init__(self, handler=None, initial_output=None):
        self.handler = handler
        self.initial_output = initial_output
        self.processes = []

    def __call__(self):
        process = FakeProcess(
            1000 + len(self.processes),
            self.handler,
            self.initial_output,
        )
        self.processes.append(process)
        return process


def request(
    token,
    *,
    logical_max_output_tokens=32,
    priority=wire.RequestPriority.NORMAL,
    deadline=None,
    sampling=None,
    seed=0,
    cohort=wire.Cohort.GREEDY,
    constraint=wire.ConstraintMode.NONE,
    mask_provider=None,
    image_owner=None,
    score_tokens=(),
):
    return engine_runtime.GenerationRequest(
        prompt_tokens=(token, token + 1),
        logical_max_output_tokens=logical_max_output_tokens,
        deadline=deadline or engine_runtime.Deadline.after(45),
        priority=priority,
        sampling=sampling or wire.SamplingParameters(),
        seed=seed,
        cohort=cohort,
        constraint=constraint,
        mask_provider=mask_provider,
        image_owner=image_owner,
        score_tokens=score_tokens,
    )


def send_success(process, call, *, slot=0, tokens=(10, 11, 12)):
    process.send(
        wire.StartEvent(
            call.request_id,
            wire.CacheDisposition.MISS,
            slot,
            0,
            4096,
        )
    )
    process.send(wire.TokensEvent(call.request_id, 0, tokens[:2]))
    if tokens[2:]:
        process.send(wire.TokensEvent(call.request_id, 2, tokens[2:]))
    process.send(
        wire.DoneEvent(
            call.request_id,
            wire.FinishReason.STOP,
            len(call.request.prompt_tokens),
            len(tokens),
            100,
            200,
            350,
        )
    )


class RuntimeTests(unittest.TestCase):
    def test_direct_admission_and_out_of_order_demultiplexing(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory,
            pending_limit=4,
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]

        requests = (
            request(
                100,
                priority=wire.RequestPriority.FOREGROUND,
                deadline=engine_runtime.Deadline(1_900_000_000_000_001, 9_000_001),
                seed=55,
            ),
            request(
                200,
                sampling=wire.SamplingParameters(0.7, 0.9, 16),
                cohort=wire.Cohort.SAMPLING,
                seed=66,
            ),
            request(
                300,
                cohort=wire.Cohort.CONSTRAINED,
                constraint=wire.ConstraintMode.TOKEN_MASK,
                mask_provider=lambda event: (
                    (1,) * (event.words_per_mask * event.mask_rows)
                ),
            ),
            request(400, priority=wire.RequestPriority.BACKGROUND),
        )
        callbacks = []
        event_types = {}
        callback_lock = threading.Lock()
        callbacks_complete = threading.Event()

        def submit(index):
            def event(call, message):
                with callback_lock:
                    event_types.setdefault(call.request_id, []).append(type(message))

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

        with ThreadPoolExecutor(max_workers=4) as executor:
            calls = list(executor.map(submit, range(4)))

        frames = process.stdin.wait_for(wire.RequestFrame, count=4)
        self.assertEqual(runtime.pending_count, 4)
        self.assertEqual(
            {frame.request_id for frame in frames}, {c.request_id for c in calls}
        )
        with mock.patch.object(wire, "serialize_message") as serialize:
            with self.assertRaises(engine_runtime.PendingLimitExceeded):
                runtime.submit(request(500))
            serialize.assert_not_called()

        expected = {item.prompt_tokens: item for item in requests}
        for frame, raw in process.stdin.records_of(wire.RequestFrame):
            source = expected[frame.prompt_tokens]
            self.assertEqual(frame.priority, source.priority)
            self.assertEqual(raw[wire.FRAME_HEADER_BYTES + 8], int(source.priority))
            self.assertEqual(
                frame.absolute_deadline_unix_micros,
                source.deadline.absolute_unix_micros,
            )
            self.assertEqual(
                frame.remaining_deadline_micros, source.deadline.remaining_micros
            )
            self.assertEqual(
                frame.logical_max_output_tokens, source.logical_max_output_tokens
            )
            self.assertAlmostEqual(
                frame.sampling.temperature, source.sampling.temperature, places=6
            )
            self.assertAlmostEqual(
                frame.sampling.top_p, source.sampling.top_p, places=6
            )
            self.assertEqual(frame.sampling.top_k, source.sampling.top_k)
            self.assertEqual(frame.seed, source.seed)
            self.assertEqual(frame.cohort, source.cohort)
            self.assertEqual(frame.constraint, source.constraint)

        reverse_calls = list(reversed(calls))
        for index, call in enumerate(reverse_calls):
            send_success(
                process,
                call,
                slot=index,
                tokens=(1000 + index, 2000 + index, 3000 + index),
            )

        for index, call in enumerate(reverse_calls):
            result = call.result(1.0)
            self.assertEqual(result.tokens, (1000 + index, 2000 + index, 3000 + index))
            self.assertEqual(result.start.slot_index, index)
        self.assertTrue(callbacks_complete.wait(1.0))
        self.assertEqual(callbacks, [call.request_id for call in reverse_calls])
        for call in calls:
            self.assertEqual(
                event_types[call.request_id],
                [
                    wire.StartEvent,
                    wire.TokensEvent,
                    wire.TokensEvent,
                    wire.DoneEvent,
                ],
            )
        self.assertEqual(runtime.pending_count, 0)
        self.assertTrue(runtime.ready)
        self.assertEqual(runtime.readiness.engine_instance_id, 1000)

    def test_result_is_available_before_completion_callback_finishes(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
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
            self.assertEqual(call.result(1.0).tokens, (10, 11, 12))
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
                runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
                process = factory.processes[0]
                call = runtime.submit(request(11))
                try:
                    if name != "before_start":
                        process.send(
                            wire.StartEvent(
                                call.request_id,
                                wire.CacheDisposition.MISS,
                                0,
                                0,
                                4096,
                            )
                        )
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
                runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
                process = factory.processes[0]
                call = runtime.submit(request(12))
                try:
                    if started:
                        process.send(
                            wire.StartEvent(
                                call.request_id,
                                wire.CacheDisposition.MISS,
                                0,
                                0,
                                4096,
                            )
                        )
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
                        )
                    )
                    with self.assertRaisesRegex(engine_runtime.ProtocolFatal, message):
                        call.result(1.0)
                    self.assertFalse(runtime.ready)
                finally:
                    runtime.close()

    def test_length_done_at_logical_max_is_valid(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(request(13, logical_max_output_tokens=3))
        process.send(
            wire.StartEvent(
                call.request_id,
                wire.CacheDisposition.MISS,
                0,
                0,
                4096,
            )
        )
        process.send(wire.TokensEvent(call.request_id, 0, (20, 21)))
        process.send(wire.TokensEvent(call.request_id, 2, (22,)))
        process.send(
            wire.DoneEvent(
                call.request_id,
                wire.FinishReason.LENGTH,
                2,
                3,
                10,
                20,
                30,
            )
        )

        result = call.result(1.0)
        self.assertEqual(result.tokens, (20, 21, 22))
        self.assertEqual(result.done.reason, wire.FinishReason.LENGTH)

    def test_stop_done_at_logical_max_is_valid(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(request(14, logical_max_output_tokens=3))
        send_success(process, call, tokens=(20, 21, 22))

        result = call.result(1.0)
        self.assertEqual(result.tokens, (20, 21, 22))
        self.assertEqual(result.done.reason, wire.FinishReason.STOP)

    def test_cancel_is_a_correlated_frame(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(request(10))

        self.assertTrue(call.cancel())
        self.assertFalse(call.cancel())
        cancel = process.stdin.wait_for(wire.CancelFrame)[0]
        self.assertEqual(cancel.request_id, call.request_id)
        process.send(
            wire.DoneEvent(
                call.request_id,
                wire.FinishReason.CANCELLED,
                2,
                0,
                0,
                0,
                10,
            )
        )
        result = call.result(1.0)
        self.assertEqual(result.done.reason, wire.FinishReason.CANCELLED)
        self.assertIsNone(result.start)
        self.assertEqual(result.tokens, ())
        self.assertFalse(call.cancel())

    def test_score_request_passes_slots_and_returns_option_logits(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(
            request(10, logical_max_output_tokens=0, score_tokens=(101, 202, 303))
        )
        frame = process.stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(frame.score_tokens, (101, 202, 303))
        self.assertEqual(frame.logical_max_output_tokens, 0)
        process.send(
            wire.StartEvent(call.request_id, wire.CacheDisposition.MISS, 0, 0, 4096)
        )
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
        self.assertEqual(result.tokens, ())
        self.assertEqual(result.done.option_logits, (1.5, -2.25, 0.5))

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
                runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
                process = factory.processes[0]
                call = runtime.submit(
                    request(
                        10,
                        logical_max_output_tokens=0,
                        score_tokens=(101, 202, 303),
                    )
                )
                try:
                    process.send(
                        wire.StartEvent(
                            call.request_id,
                            wire.CacheDisposition.MISS,
                            0,
                            0,
                            4096,
                        )
                    )
                    if decode_micros:
                        # A scored DoneEvent with decode activity is invalid at
                        # encode time, so inject the malformed frame as raw
                        # wire bytes to exercise the runtime's real parser.
                        done = bytearray(
                            wire.serialize_message(
                                wire.DoneEvent(
                                    call.request_id,
                                    reason,
                                    2,
                                    0,
                                    100,
                                    0,
                                    350,
                                    logits,
                                )
                            )
                        )
                        done[
                            wire.FRAME_HEADER_BYTES + 25 : wire.FRAME_HEADER_BYTES + 33
                        ] = decode_micros.to_bytes(8, "little")
                        process.stdout.feed(done)
                    else:
                        process.send(
                            wire.DoneEvent(
                                call.request_id,
                                reason,
                                2,
                                0,
                                100,
                                0,
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
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        process = factory.processes[0]
        call = runtime.submit(request(10))
        try:
            process.send(
                wire.StartEvent(call.request_id, wire.CacheDisposition.MISS, 0, 0, 4096)
            )
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
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(
            request(10, logical_max_output_tokens=0, score_tokens=(101, 202))
        )
        self.assertTrue(call.cancel())
        process.send(
            wire.DoneEvent(
                call.request_id,
                wire.FinishReason.CANCELLED,
                2,
                0,
                0,
                0,
                10,
            )
        )
        result = call.result(1.0)
        self.assertEqual(result.done.reason, wire.FinishReason.CANCELLED)
        self.assertEqual(result.done.option_logits, ())

    def test_generate_is_the_blocking_convenience_over_direct_submit(self):
        def handler(process, message):
            if not isinstance(message, wire.RequestFrame):
                return
            process.send(
                wire.StartEvent(
                    message.request_id,
                    wire.CacheDisposition.PREFIX_HIT,
                    2,
                    1,
                    4096,
                )
            )
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
                )
            )

        factory = FakeFactory(handler)
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        call = runtime.submit(request(15))
        result = call.result(1.0)
        self.assertEqual(result.tokens, (41, 42))
        self.assertEqual(
            result.start.cache_disposition, wire.CacheDisposition.PREFIX_HIT
        )

    def test_mask_response_preserves_both_ids_and_exact_word_count(self):
        provider_thread = []
        provider_called = threading.Event()

        def provider(event):
            provider_thread.append(threading.current_thread().name)
            provider_called.set()
            return tuple(range(1, event.words_per_mask * event.mask_rows + 1))

        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory,
            mask_workers=1,
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(
            request(
                20,
                cohort=wire.Cohort.CONSTRAINED,
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
        self.assertEqual(response.mask_words, (1, 2, 3, 4, 5, 6, 7, 8))
        send_success(process, call)
        self.assertEqual(call.result(1.0).tokens, (10, 11, 12))

    def test_bad_mask_size_cancels_and_fails_only_that_call(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        bad = runtime.submit(
            request(
                30,
                cohort=wire.Cohort.CONSTRAINED,
                constraint=wire.ConstraintMode.TOKEN_MASK,
                mask_provider=lambda _event: (1,),
            )
        )
        healthy = runtime.submit(request(40))

        process.send(
            wire.StartEvent(
                bad.request_id,
                wire.CacheDisposition.MISS,
                0,
                0,
                4096,
            )
        )
        process.send(wire.MaskRequestEvent(bad.request_id, 88, 2, (5, 6)))
        cancel = process.stdin.wait_for(wire.CancelFrame)[0]
        self.assertEqual(cancel.request_id, bad.request_id)
        process.send(
            wire.DoneEvent(
                bad.request_id,
                wire.FinishReason.CANCELLED,
                2,
                0,
                0,
                0,
                10,
            )
        )
        with self.assertRaises(engine_runtime.MaskComputationFailed):
            bad.result(1.0)
        send_success(process, healthy)
        self.assertEqual(healthy.result(1.0).tokens, (10, 11, 12))
        self.assertTrue(runtime.ready)

    def test_cancelled_call_never_writes_a_late_mask_response(self):
        provider_started = threading.Event()
        provider_release = threading.Event()
        provider_finished = threading.Event()

        def provider(event):
            provider_started.set()
            provider_release.wait(1.0)
            provider_finished.set()
            return (1,) * (event.words_per_mask * event.mask_rows)

        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory,
            mask_workers=1,
        )
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        call = runtime.submit(
            request(
                31,
                cohort=wire.Cohort.CONSTRAINED,
                constraint=wire.ConstraintMode.TOKEN_MASK,
                mask_provider=provider,
            )
        )
        process.send(
            wire.StartEvent(
                call.request_id,
                wire.CacheDisposition.MISS,
                0,
                0,
                4096,
            )
        )
        process.send(wire.MaskRequestEvent(call.request_id, 89, 2, (5, 6)))
        self.assertTrue(provider_started.wait(1.0))
        self.assertTrue(call.cancel())
        cancel = process.stdin.wait_for(wire.CancelFrame)[0]
        self.assertEqual(cancel.request_id, call.request_id)
        process.send(
            wire.DoneEvent(
                call.request_id,
                wire.FinishReason.CANCELLED,
                2,
                0,
                0,
                0,
                10,
            )
        )
        self.assertEqual(call.result(1.0).done.reason, wire.FinishReason.CANCELLED)
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
            return (1,) * (event.words_per_mask * event.mask_rows)

        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, mask_workers=1, pending_limit=2
        )
        self.addCleanup(runtime.close)
        self.addCleanup(release.set)
        process = factory.processes[0]

        def submit_mask():
            call = runtime.submit(
                request(
                    31,
                    cohort=wire.Cohort.CONSTRAINED,
                    constraint=wire.ConstraintMode.TOKEN_MASK,
                    mask_provider=provider,
                )
            )
            call._record_start(
                wire.StartEvent(call.request_id, wire.CacheDisposition.MISS, 0, 0, 4096)
            )
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
                        call.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 10
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
                first.request_id, wire.FinishReason.CANCELLED, 2, 0, 0, 0, 10
            )
        )
        first.result(1.0)
        recovered = submit_mask()
        runtime._mask_executor.submit(lambda: None).result(1.0)
        self.assertIn(recovered.request_id, calls_to_provider)
        self.assertTrue(runtime.ready)

    def test_correlated_status_and_expired_response(self):
        respond = threading.Event()

        def handler(process, message):
            if isinstance(message, wire.StatusRequestFrame) and respond.is_set():
                process.send(
                    wire.StatusJsonEvent(
                        message.correlation_id,
                        wire.STATUS_SCHEMA_VERSION,
                        b'{"schema_version":4,"ready":true}',
                    )
                )

        factory = FakeFactory(handler)
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        process = factory.processes[0]

        with self.assertRaises(TimeoutError):
            runtime.status(timeout=0.02)
        first = process.stdin.messages(wire.StatusRequestFrame)[0]
        process.send(
            wire.StatusJsonEvent(
                first.correlation_id,
                wire.STATUS_SCHEMA_VERSION,
                b'{"schema_version":4,"late":true}',
            )
        )
        respond.set()
        status = runtime.status(timeout=1.0)
        self.assertNotEqual(status.correlation_id, first.correlation_id)
        self.assertEqual(status.json, b'{"schema_version":4,"ready":true}')
        self.assertEqual(runtime.last_status, status)
        self.assertTrue(runtime.ready)

    def test_request_and_capacity_failures_are_scoped(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        process = factory.processes[0]
        request_error = runtime.submit(request(50))
        capacity = runtime.submit(request(60))
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
        process.send(wire.CapacityExhaustedEvent(capacity.request_id, 40, 12, 50_000))
        send_success(process, healthy)

        with self.assertRaises(engine_runtime.RequestFailed) as caught:
            request_error.result(1.0)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.code, b"deadline_exceeded")
        with self.assertRaises(engine_runtime.CapacityExhausted) as caught:
            capacity.result(1.0)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.event.retry_after_micros, 50_000)
        self.assertIn("logical_pages_free=12", str(caught.exception))
        self.assertIn("system memory becomes available", str(caught.exception))
        self.assertEqual(healthy.result(1.0).tokens, (10, 11, 12))
        self.assertTrue(runtime.ready)

    def test_unhealthy_fatal_and_eof_fail_all_then_restart_lazily(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)

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

        after_unhealthy = runtime.submit(request(100))
        second = factory.processes[1]
        self.assertEqual(runtime.restart_count, 1)
        self.assertEqual(runtime.readiness.engine_instance_id, 1001)
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

        after_fatal = runtime.submit(request(110))
        third = factory.processes[2]
        self.assertEqual(runtime.restart_count, 2)
        third.close_stdout()
        with self.assertRaises(engine_runtime.EngineUnhealthy):
            after_fatal.result(1.0)

        after_eof = runtime.submit(request(120))
        fourth = factory.processes[3]
        self.assertEqual(runtime.restart_count, 3)
        send_success(fourth, after_eof)
        self.assertEqual(after_eof.result(1.0).tokens, (10, 11, 12))

    def test_invalidation_before_registration_returns_the_admission_slot(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, pending_limit=1
        )
        self.addCleanup(runtime.close)
        ensure_process = runtime._ensure_process

        def ensure_then_fail(timeout=None):
            generation = ensure_process(timeout=timeout)
            runtime._fail_generation(
                generation, engine_runtime.EngineUnhealthy("lost after readiness")
            )
            return generation

        with mock.patch.object(runtime, "_ensure_process", ensure_then_fail):
            with self.assertRaises(engine_runtime.EngineUnhealthy):
                runtime.submit(request(125))
        self.assertEqual(runtime.pending_count, 0)
        self.assertEqual(runtime._admission_slots._value, runtime.pending_limit)

        replacement = runtime.submit(request(126))
        self.assertEqual(runtime.restart_count, 1)
        send_success(factory.processes[1], replacement)
        self.assertEqual(replacement.result(1.0).tokens, (10, 11, 12))

    def test_eof_restarts_and_close_is_terminal(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        call = runtime.submit(request(130))
        factory.processes[0].close_stdout()
        with self.assertRaisesRegex(engine_runtime.EngineUnhealthy, "reached EOF"):
            call.result(1.0)
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
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
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
            )
            call = runtime.submit(request(130))
            factory.processes[0].close_stdout()
            with self.assertRaisesRegex(engine_runtime.EngineUnhealthy, "reached EOF"):
                call.result(1.0)
            path = runtime.last_crash_trace
            self.assertIsNotNone(path)
            self.assertTrue(Path(path).is_file())
            runtime.close()

    def test_close_callback_can_reenter_without_process_lock_deadlock(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
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
            )
        self.assertIsNotNone(factory.processes[0].poll())

    def test_concurrent_submitters_share_one_failed_startup(self):
        factory = FakeFactory(initial_output=b"")
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory,
            startup_timeout=0.2,
            eager_start=False,
        )
        self.addCleanup(runtime.close)
        barrier = threading.Barrier(4)

        def submit(index):
            barrier.wait()
            with self.assertRaises(engine_runtime.EngineUnhealthy):
                runtime.submit(request(200 + index))

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(submit, range(4)))
        elapsed = time.monotonic() - started
        self.assertEqual(len(factory.processes), 1)
        self.assertLess(elapsed, 0.5)

    def test_close_interrupts_an_in_progress_startup(self):
        factory = FakeFactory(initial_output=b"")
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory,
            startup_timeout=10.0,
            eager_start=False,
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
            eager_start=False,
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

    def test_kill_fallback_ends_an_engine_that_survives_terminate(self):
        class StubbornProcess(FakeProcess):
            def terminate(self):
                pass

        process = StubbornProcess(1)
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=FakeFactory(),
            startup_timeout=10.0,
            eager_start=False,
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

    def test_startup_rejects_ready_missing_required_features(self):
        ready = wire.serialize_message(
            wire.ReadyEvent(
                engine_instance_id=1,
                max_concurrent_requests=4,
                max_context_tokens=131_072,
                feature_bits=int(wire.ReadyFeature.CANCELLATION),
            )
        )
        factory = FakeFactory(initial_output=ready)
        with self.assertRaisesRegex(
            engine_runtime.ProtocolFatal, "missing required native protocol features"
        ):
            engine_runtime.MultiplexedRuntime(
                process_factory=factory,
                startup_timeout=0.2,
            )
        self.assertIsNotNone(factory.processes[0].poll())

    def test_deadline_helper_serializes_absolute_and_remaining_clocks(self):
        deadline = engine_runtime.Deadline.after(
            1.25,
            wall_time_ns=lambda: 1_700_000_000_000_000_000,
        )
        self.assertEqual(deadline.absolute_unix_micros, 1_700_000_001_250_000)
        self.assertEqual(deadline.remaining_micros, 1_250_000)

        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        with mock.patch.object(
            engine_runtime.time, "time_ns", return_value=1_700_000_000_000_000_000
        ):
            call = runtime.submit(
                request(
                    160, priority=wire.RequestPriority.FOREGROUND, deadline=deadline
                )
            )
        frame = factory.processes[0].stdin.wait_for(wire.RequestFrame)[0]
        self.assertEqual(frame.request_id, call.request_id)
        self.assertEqual(frame.priority, wire.RequestPriority.FOREGROUND)
        self.assertEqual(
            frame.absolute_deadline_unix_micros, deadline.absolute_unix_micros
        )
        self.assertEqual(frame.remaining_deadline_micros, deadline.remaining_micros)

    def test_short_request_leaves_shared_startup_running(self):
        factory = FakeFactory(initial_output=b"")
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory,
            eager_start=False,
            startup_timeout=1,
            pending_limit=2,
        )
        self.addCleanup(runtime.close)
        with ThreadPoolExecutor(max_workers=2) as executor:
            short = executor.submit(
                runtime.submit, request(1, deadline=engine_runtime.Deadline.after(0.03))
            )
            long = executor.submit(
                runtime.submit, request(2, deadline=engine_runtime.Deadline.after(1))
            )
            with self.assertRaises(TimeoutError):
                short.result(0.5)
            self.assertEqual(len(factory.processes), 1)
            process = factory.processes[0]
            self.assertIsNone(process.poll())
            self.assertEqual(process.stdin.messages(wire.RequestFrame), [])
            process.send(wire.ReadyEvent(1000, 4, 131072, READY_FEATURES))
            call = long.result(0.5)
        frames = process.stdin.messages(wire.RequestFrame)
        self.assertEqual([frame.prompt_tokens for frame in frames], [(2, 3)])
        send_success(process, call)
        call.result(0.5)
        self.assertEqual(runtime.pending_count, 0)

    def test_startup_expires_even_after_every_caller_has_left(self):
        factory = FakeFactory(initial_output=b"")
        created = threading.Event()

        def create_process():
            process = factory()
            created.set()
            return process

        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=create_process, eager_start=False, startup_timeout=2.0
        )
        self.addCleanup(runtime.close)
        with self.assertRaises(TimeoutError):
            runtime.submit(request(1, deadline=engine_runtime.Deadline.after(0.01)))
        self.assertTrue(created.wait(1.0))
        process = factory.processes[0]
        self.assertIsNone(process.poll())
        self.assertIsNotNone(process.wait(5.0))
        self.assertFalse(runtime.ready)
        self.assertEqual(process.stdin.messages(wire.RequestFrame), [])

    def test_slow_process_factory_does_not_hold_request_deadline(self):
        release = threading.Event()
        factory = FakeFactory()

        def slow_factory():
            release.wait(1)
            return factory()

        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=slow_factory, eager_start=False, startup_timeout=1
        )
        self.addCleanup(runtime.close)
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    runtime.submit,
                    request(1, deadline=engine_runtime.Deadline.after(0.01)),
                )
                with self.assertRaises(TimeoutError):
                    future.result(0.5)
        finally:
            release.set()
        self.assertTrue(runtime.wait_ready(0.5))
        self.assertEqual(len(factory.processes), 1)
        self.assertEqual(factory.processes[0].stdin.messages(wire.RequestFrame), [])

    def test_expired_request_does_not_launch_or_write(self):
        factory = FakeFactory()
        runtime = engine_runtime.MultiplexedRuntime(
            process_factory=factory, eager_start=False
        )
        self.addCleanup(runtime.close)
        with self.assertRaises(TimeoutError):
            runtime.submit(request(1, deadline=engine_runtime.Deadline(1, 1_000_000)))
        self.assertEqual(factory.processes, [])

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
            process_factory=slow_factory, eager_start=False, startup_timeout=0.02
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
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
        self.addCleanup(runtime.close)
        with runtime._write_lock:
            for action in (
                lambda: runtime.status(timeout=0.01),
                lambda: runtime.submit(
                    request(1, deadline=engine_runtime.Deadline.after(0.01))
                ),
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
            process_factory=factory, io_timeout=0.02
        )
        self.addCleanup(runtime.close)
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
        runtime = engine_runtime.MultiplexedRuntime(process_factory=factory)
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
            process_factory=factory, io_timeout=0.15
        )
        self.addCleanup(runtime.close)
        recovered_calls = []
        failure_state = []

        def completed(_call):
            failure_state.append((runtime._write_lock.locked(), runtime.ready))
            recovered_calls.append(runtime.submit(request(2)))

        active = runtime.submit(request(1), on_complete=completed)
        os.read(read_fd, 65536)  # Drain just the first complete request.
        large = engine_runtime.GenerationRequest(
            prompt_tokens=tuple(range(131072)),
            logical_max_output_tokens=32,
            deadline=engine_runtime.Deadline.after(1),
        )
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
        self.assertEqual(failure_state, [(False, False)])
        self.assertIsNotNone(process.poll())
        self.assertEqual(len(recovered_calls), 1)
        recovered = recovered_calls[0]
        send_success(replacement_factory.processes[0], recovered)
        recovered.result(0.5)
        self.assertEqual(runtime.pending_count, 0)
        self.assertEqual(runtime.restart_count, 1)


if __name__ == "__main__":
    unittest.main()

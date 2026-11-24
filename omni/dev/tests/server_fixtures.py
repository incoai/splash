"""What the server tests share: fakes of the native engine and its process,
the requests the runtime tests write, and the launcher arguments and
interface checks several tests use. The audio fakes and the harness that
serves the audio frontend are in audio_fixtures.py."""

import inspect
import math
import os
import signal
import subprocess
import threading
import time

from dev.tests import native_peer
from install import launcher
from server import audio_protocol as audio_wire
from server import audio_runtime
from server import protocol as wire
from server import runtime as engine_runtime

# The deadline of a request prepared directly, which never expires.
FOREVER = math.inf
# A version-8 Ready, the base protocol's, which the crash trace tests speak.
READY = wire.ReadyEvent(4, 131_072, False)
AUDIO_READY = audio_wire.ReadyEvent(8, 8192, audio_wire.ModelKind.TTS)


class FakeCall:
    """A RuntimeCall the fake engine runs by its plan or, without one, that
    the test drives through emit() and complete()."""

    def __init__(self, runtime, request_id, request, on_event, on_complete, plan):
        self.runtime = runtime
        self.request_id = request_id
        self.request = request
        self.on_event = on_event
        self.on_complete = on_complete
        self.plan = plan
        self.callback_error = None
        self.cancel_requested = False
        self._result = None
        self._error = None
        self._done = False
        self._lock = threading.Lock()

    @property
    def done(self):
        with self._lock:
            return self._done

    def emit(self, event):
        if self.on_event is None:
            return
        try:
            self.on_event(self, event)
        except Exception as error:
            if self.callback_error is None:
                self.callback_error = error
            self.cancel()

    def complete(self, *, result=None, error=None):
        with self._lock:
            if self._done:
                return False
            self._result = result
            self._error = error
            self._done = True
        if self.on_complete is not None:
            self.on_complete(self)
        return True

    def result(self, timeout=None):
        with self._lock:
            if not self._done:
                raise TimeoutError("fake runtime call is not complete")
            if self._error is not None:
                raise self._error
            return self._result

    def cancel(self):
        with self._lock:
            if self._done or self.cancel_requested:
                return False
            self.cancel_requested = True
        self.runtime.cancel_count += 1
        if self.plan is not None:
            self.plan.cancelled.set()
            self.plan.start_release.set()
            self.plan.release.set()
            self.plan.terminal_release.set()
        return True


class FakeRuntime:
    """MultiplexedRuntime's stand-in. Each call runs the next of `plans`, or
    the subclass's default plan once they run out, on a thread of its own;
    with manual=True the test drives every call."""

    def __init__(self, *plans, manual=False):
        self.plans = list(plans)
        self.manual = manual
        self.requests = []
        self.calls = []
        self.cancel_count = 0
        self.closed = False
        self.ready = True
        self.readiness = READY
        self.fatal_error = None
        self.pending_limit = 32
        self.restart_count = 0
        self.last_crash_trace = None
        self.status_event = native_peer.status_event()
        self.threads = []

    @property
    def pending_count(self):
        return sum(not call.done for call in self.calls)

    def wait_ready(self, timeout=None):
        # The fake starts no engine: it is as ready as the test sets it.
        return self.ready

    def status(self, timeout=5.0, *, fail_unanswered=False):
        if timeout <= 0:
            raise ValueError("status timeout must be positive")
        if not self.ready:
            raise engine_runtime.EngineUnhealthy("native process is not ready")
        return self.status_event

    def submit(self, request, *, on_event=None, on_complete=None):
        if self.pending_count >= self.pending_limit:
            raise engine_runtime.PendingLimitExceeded("fake runtime is full")
        plan = None
        if not self.manual:
            plan = self.plans.pop(0) if self.plans else self.default_plan(request)
        call = FakeCall(self, len(self.calls) + 1, request, on_event, on_complete, plan)
        self.calls.append(call)
        self.requests.append(request)
        if plan is not None:
            thread = threading.Thread(target=self._run, args=(call,), daemon=True)
            self.threads.append(thread)
            thread.start()
        return call

    @staticmethod
    def default_plan(request):
        """The plan of a call once the given plans ran out."""
        raise NotImplementedError

    def _run(self, call):
        """Run `call` by its plan."""
        raise NotImplementedError

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.ready = False
        for call in self.calls:
            call.cancel()
        for thread in self.threads:
            thread.join(2)

    def kill(self):
        # The engine is gone; its calls end as the test ends them.
        self.ready = False


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
    def __init__(self, process, *, audio=False):
        self._process = process
        self._condition = threading.Condition()
        self._reader = native_peer.ClientFrameReader(audio=audio)
        self._closed = False
        self.records = []
        self._read_fd, self._write_fd = os.pipe()

    def fileno(self):
        return self._write_fd

    def write(self, data):
        with self._condition:
            if self._closed:
                raise BrokenPipeError("fake stdin is closed")
            records = self._reader.feed(data)
            self.records.extend(records)
            self._condition.notify_all()
        for message, _raw in records:
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
    """A fake of the engine process: one speaking the base protocol, version
    8, and announcing READY, or with audio=True the audio engine's, speaking
    version 9 and announcing AUDIO_READY, which the runtime tests drive."""

    _pids = iter(range(50_000, 60_000))

    def __init__(self, input_handler=None, initial_output=None, *, audio=False):
        self.pid = next(self._pids)
        self.input_handler = input_handler
        self.audio = audio
        self.stdout = FakeOutput()
        self.stdin = FakeInput(self, audio=audio)
        self._condition = threading.Condition()
        self._exit_code = None
        if initial_output is None:
            self.send(AUDIO_READY if audio else READY)
        elif initial_output:
            self.stdout.feed(initial_output)

    def send(self, event):
        self.stdout.feed(
            native_peer.serialize_audio_event(event)
            if self.audio
            else native_peer.serialize_event(event)
        )

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
    def __init__(self, handler=None, initial_output=None, *, audio=False):
        self.handler = handler
        self.initial_output = initial_output
        self.audio = audio
        self.processes = []

    def __call__(self):
        process = FakeProcess(self.handler, self.initial_output, audio=self.audio)
        self.processes.append(process)
        return process


def request(token, *, samples=4, max_tokens=32, deadline=45.0):
    """A transcription request whose deadline is `deadline` seconds from now,
    as the runtime tests write one: its prefix tokens, (token, token + 1),
    tell the frames apart, and its `samples` float32 samples are silence."""
    frame = audio_wire.TranscribeRequest(
        0, (token, token + 1), (151670,), bytes(4 * samples), max_tokens
    )
    return audio_runtime.AudioRequest(frame, time.monotonic() + deadline)


def written_frame(process, call):
    return next(
        frame
        for frame in process.stdin.messages(audio_wire.TranscribeRequest)
        if frame.request_id == call.request_id
    )


def send_success(process, call, *, lane=0, tokens=(10, 11, 12)):
    """The events of a transcription that streams `tokens` and stops."""
    process.send(audio_wire.StartEvent(call.request_id, lane, 0))
    process.send(audio_wire.TokensEvent(call.request_id, 0, tokens[:2]))
    if tokens[2:]:
        process.send(audio_wire.TokensEvent(call.request_id, 2, tokens[2:]))
    frame = written_frame(process, call)
    process.send(
        audio_wire.DoneEvent(
            call.request_id,
            audio_wire.FinishReason.STOP,
            len(frame.prefix_tokens) + len(frame.suffix_tokens),
            len(tokens),
            100,
            200,
            350,
            (),
        )
    )


def keep_stop_signals(test):
    """Give the process its stop-signal handlers and mask back after `test`:
    serve takes the signals, and blocks them for the exec a test stubs."""
    for number in launcher.STOP_SIGNALS:
        test.addCleanup(signal.signal, number, signal.getsignal(number))
    test.addCleanup(
        signal.pthread_sigmask,
        signal.SIG_SETMASK,
        signal.pthread_sigmask(signal.SIG_BLOCK, ()),
    )


def server_arguments(argv):
    """The server's own arguments in the launcher's command for it."""
    return argv[argv.index("server.audio_server") + 1 :]


def interface_drift(fake, real):
    """Where `fake` departs from the class `real` it stands in for: each of
    real's public methods it lacks or takes other parameters for, and each of
    its properties and declared attributes it lacks. Annotations aside."""

    def parameters(function):
        return [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(function).parameters.values()
        ]

    drift = []
    members = dict(inspect.getmembers_static(real))
    for name in sorted({*members, *inspect.get_annotations(real)}):
        if name.startswith("_"):
            continue
        member = members.get(name)
        if inspect.isfunction(member):
            own = inspect.getattr_static(type(fake), name, None)
            if not inspect.isfunction(own) or parameters(own) != parameters(member):
                drift.append(f"method {name}")
        elif (member is None or isinstance(member, property)) and not hasattr(
            fake, name
        ):
            drift.append(f"attribute {name}")
    return drift

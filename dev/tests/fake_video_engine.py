"""A stand-in for a video model's engine process (runtime/video/
VideoRuntime.hpp) on the native protocol, for the server's tests.

  fake_video_engine.py LOG STEP_SECONDS

It announces FeatureVideo, then runs each VideoRequest's phases, its
denoising in four steps of STEP_SECONDS, then its preview when asked and its
clip: mp4s of the request's seed. A Cancel ends a request at its next step
or phase. LOG gets one JSON line per request taken and per cancel."""

import json
import os
import select
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dev.tests import native_peer  # noqa: E402
from server import protocol as wire  # noqa: E402

DENOISE_STEPS = 4


class Engine:
    def __init__(self, log, step_seconds):
        self.log = log
        self.step_seconds = step_seconds
        self.reader = native_peer.ClientFrameReader()
        self.requests, self.cancelled = [], set()

    def send(self, event):
        sys.stdout.buffer.write(native_peer.serialize_event(event))
        sys.stdout.buffer.flush()

    def note(self, record):
        self.log.write(json.dumps(record) + "\n")
        self.log.flush()

    def poll(self, timeout):
        """Take what the server wrote within timeout (None: until it writes)."""
        if select.select([sys.stdin], [], [], timeout)[0]:
            data = os.read(sys.stdin.fileno(), 1 << 16)
            if not data:
                raise SystemExit(0)
            for message, _ in self.reader.feed(data):
                if isinstance(message, wire.CancelFrame):
                    self.cancelled.add(message.request_id)
                    self.note({"cancel": message.request_id})
                else:
                    self.requests.append(message)

    def phase(self, request, phase, steps=1):
        for step in range(steps + 1):
            if request.request_id in self.cancelled:
                return False
            self.send(
                wire.VideoProgressEvent(
                    request.request_id, phase, step, steps, 1000 * step, 1 << 30
                )
            )
            if step < steps:
                self.poll(self.step_seconds if phase is wire.VideoPhase.DENOISE else 0)
        return True

    def run(self, request):
        self.note({"request": request.request_id, "tokens": list(request.prompt_tokens),
                   "frames": request.frames, "variant": int(request.variant),
                   "outputs": int(request.outputs), "seed": request.seed})  # fmt: skip
        phases = [(wire.VideoPhase.ENCODE, 1), (wire.VideoPhase.PREPARE, 1),
                  (wire.VideoPhase.DENOISE, DENOISE_STEPS)]  # fmt: skip
        for phase, steps in phases:
            if not self.phase(request, phase, steps):
                return self.cancel(request)
        mp4 = b"\0\0\0\x18ftypmp42" + request.seed.to_bytes(8, "little")
        if request.outputs & wire.VideoOutputFlag.PREVIEW:
            self.phase(request, wire.VideoPhase.PREVIEW)
            self.send(
                wire.VideoOutputEvent(
                    request.request_id,
                    wire.VideoOutputKind.PREVIEW,
                    25,
                    0,
                    b"preview" + mp4,
                )
            )
        for phase in (wire.VideoPhase.DECODE, wire.VideoPhase.MUX):
            if not self.phase(request, phase):
                return self.cancel(request)
        self.send(
            wire.VideoOutputEvent(
                request.request_id, wire.VideoOutputKind.VIDEO, request.frames, 7, mp4
            )
        )

    def cancel(self, request):
        self.send(wire.ErrorEvent(wire.FailureClass.REQUEST_ERROR, request.request_id, False,
                                  b"cancelled", b"the video request was cancelled"))  # fmt: skip

    def serve(self):
        self.send(wire.ReadyEvent(1, 2048, False, True))
        while True:
            while not self.requests:
                self.poll(None)
            self.run(self.requests.pop(0))


if __name__ == "__main__":
    with open(sys.argv[1], "a") as log:
        Engine(log, float(sys.argv[2])).serve()

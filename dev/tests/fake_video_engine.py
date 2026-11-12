"""A video model's engine (runtime/video/VideoRuntime.hpp) for the server's
tests: fake_video_engine.py LOG STEP_SECONDS [fail | refuse | hold]. Each
request denoises in four steps of STEP_SECONDS, unless a Cancel ends it, then
sends its preview when asked and its clip, each the request's seed; with
`fail`, the engine's error instead. With `refuse`, the engine sends a start's
failure in place of Ready and exits. With `hold`, the first request waits at
its first step for its Cancel, a minute at most. LOG gets a JSON line per request, with its canvas and the SHA-256 of
each keyframe it carries, and per cancel."""

import hashlib
import json
import os
import select
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dev.tests import native_peer  # noqa: E402
from server import protocol as wire  # noqa: E402


def main(log, step_seconds, mode):
    reader, pending, cancelled = native_peer.ClientFrameReader(), [], set()

    def send(event):
        sys.stdout.buffer.write(native_peer.serialize_event(event))
        sys.stdout.buffer.flush()

    def note(**record):
        log.write(json.dumps(record) + "\n")
        log.flush()

    def poll(timeout):
        if select.select([sys.stdin], [], [], timeout)[0]:
            data = os.read(sys.stdin.fileno(), 1 << 16)
            if not data:
                raise SystemExit(0)
            for message, _ in reader.feed(data):
                if isinstance(message, wire.CancelFrame):
                    cancelled.add(message.request_id)
                    note(cancel=message.request_id)
                else:
                    pending.append(message)

    if mode == "refuse":
        send(wire.StartFailedEvent(wire.StartFailureKind.DEVICE,
                                   wire.ShortfallBound.DEVICE_MEMORY, 0, 0, 0,
                                   b"no Apple10 GPU"))  # fmt: skip
        return
    send(wire.ReadyEvent(1, 2048, False, 0, 0, (), video=True))
    while True:
        while not pending:
            poll(None)
        request = pending.pop(0)
        number, seed = request.request_id, request.seed.to_bytes(8, "little")
        keyframes = {name: hashlib.sha256(pixels).hexdigest()
                     for name, pixels in (("first_frame", request.first_frame),
                                          ("last_frame", request.last_frame)) if pixels}  # fmt: skip
        note(request=number, tokens=list(request.prompt_tokens), frames=request.frames,
             size=f"{request.width}x{request.height}", outputs=int(request.outputs),
             seed=request.seed, **keyframes)  # fmt: skip
        for step in range(5):
            if number in cancelled:
                send(wire.ErrorEvent(wire.FailureClass.REQUEST_ERROR, number, False,
                                     b"cancelled", b"the request was cancelled"))  # fmt: skip
                break
            send(
                wire.VideoProgressEvent(number, wire.VideoPhase.DENOISE, step, 4, 0, 0)
            )
            if mode == "hold" and number == 1 and step == 0:
                held = time.monotonic() + 60
                while number not in cancelled and time.monotonic() < held:
                    poll(max(0.0, held - time.monotonic()))
            poll(step_seconds)
        else:
            if mode == "fail":
                send(wire.ErrorEvent(wire.FailureClass.REQUEST_ERROR, number, False,
                                     b"video_engine_failed", b"the video VAE failed"))  # fmt: skip
                continue
            if request.outputs & wire.VideoOutputFlag.PREVIEW:
                kind = wire.VideoOutputKind.PREVIEW
                send(wire.VideoOutputEvent(number, kind, 25, 0, b"preview" + seed))
            kind = wire.VideoOutputKind.VIDEO
            send(wire.VideoOutputEvent(number, kind, request.frames, 7, seed))


if __name__ == "__main__":
    with open(sys.argv[1], "a") as log:
        main(log, float(sys.argv[2]), (sys.argv[3:] or [None])[0])

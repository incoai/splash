"""A video model's engine (runtime/video/VideoRuntime.hpp) for the server's
tests: fake_video_engine.py LOG STEP_SECONDS [fail]. Each request denoises in
four steps of STEP_SECONDS, unless a Cancel ends it, then sends its preview
when asked and its clip, each the request's seed; with `fail`, the engine's
error instead. LOG gets a JSON line per request, with the SHA-256 of each
keyframe it carries, and per cancel."""

import hashlib
import json
import os
import select
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dev.tests import native_peer  # noqa: E402
from server import protocol as wire  # noqa: E402


def main(log, step_seconds, fail):
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

    send(wire.ReadyEvent(1, 2048, False, True))
    while True:
        while not pending:
            poll(None)
        request = pending.pop(0)
        number, seed = request.request_id, request.seed.to_bytes(8, "little")
        keyframes = {name: hashlib.sha256(pixels).hexdigest()
                     for name, pixels in (("first_frame", request.first_frame),
                                          ("last_frame", request.last_frame)) if pixels}  # fmt: skip
        note(request=number, tokens=list(request.prompt_tokens), frames=request.frames,
             outputs=int(request.outputs), seed=request.seed, **keyframes)  # fmt: skip
        for step in range(5):
            if number in cancelled:
                send(wire.ErrorEvent(wire.FailureClass.REQUEST_ERROR, number, False,
                                     b"cancelled", b"the request was cancelled"))  # fmt: skip
                break
            send(
                wire.VideoProgressEvent(number, wire.VideoPhase.DENOISE, step, 4, 0, 0)
            )
            poll(step_seconds)
        else:
            if fail:
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
        main(log, float(sys.argv[2]), sys.argv[3:] == ["fail"])

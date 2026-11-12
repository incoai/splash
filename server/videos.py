"""A video model's server: OpenAI-shaped /v1/videos jobs on the video engine.

`python -m server.server` of a video model's assembly (install/video.py)
serves these routes alone, beside /health, /ready, /status and /v1/models:
POST /v1/videos, GET /v1/videos, GET /v1/videos/{id},
GET /v1/videos/{id}/content and DELETE /v1/videos/{id}. A language model's
routes answer 404: a language model is served by an instance of its own.
A job may start from an image: OpenAI's `input_reference`, a PNG or JPEG
file of the multipart body, is the clip's first frame, and Splash's
`last_frame`, another, its last; each is put on the canvas here
(images.keyframe) and travels with the job's request to the engine. A
Ref2VA job is conditioned on references instead: repeated `references`
files of the multipart body, images, videos with their soundtracks and
audio clips,
decoded and normalized here as Pika H3 Flex does (references.normalize), on
the pika-ref2va-6 variant (MiniMax-H3's Ref2VA base).
The engine process (runtime/video/VideoRuntime.hpp) runs one job at a time
on the GPU, the others wait in a short queue in their order of arrival. A
job's progress follows the engine's phases, its denoising step by step; its
preview and mp4 stay in a store of bounded bytes, which evicts the oldest
finished job first; DELETE cancels a job, a running one at its next step.

`python -m server.videos`, which `splash generate` starts, runs one clip on
the engine instead and writes its mp4 (main, generate).
"""

from __future__ import annotations

import argparse
import email.parser
import email.policy
import itertools
import json
import os
import re
import secrets
import select
import signal
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import parse_qs

from install import assembly, models, video
from install.families import H3

from . import images, supervision
from . import protocol as wire
from . import references as refs
from .diagnostics import print_status
from .errors import APIError

# OpenAI's `seconds` names a clip: a whole number of seconds (H3.seconds),
# and for the golden-clip tests alone (SPLASH_GOLDEN_CLIPS=1,
# dev/tests/video_real.py) the 3 s golden clips'.
GOLDEN_SECONDS = 3 if os.environ.get("SPLASH_GOLDEN_CLIPS") == "1" else None
SECONDS_PATTERN = re.compile(r"[1-9][0-9]?")
# OpenAI's `size`, WIDTHxHEIGHT, a canvas of one of H3.canvas_tiers.
SIZE_PATTERN = re.compile(r"([1-9][0-9]{0,4})x([1-9][0-9]{0,4})")
# The shipped variant whose DiT runs reference rows (MiniMax-H3's Ref2VA base
# with Pika H3 Flex's ref2va adapter). A request with references takes the
# first installed variant whose spec has references, else this one; such a
# variant takes no other request.
REF2VA_VARIANT = "pika-ref2va-6"
FIELDS = {
    "prompt",
    "model",
    "seconds",
    "size",
    "variant",
    "seed",
    "steps",
    "preview",
    "audio",
    "input_reference",
    "last_frame",
    "references",
}
# The image fields of a multipart body: the clip's first and last frames.
KEYFRAMES = ("input_reference", "last_frame")
# The repeated file field of a Ref2VA job's references, in presentation order.
REFERENCES = "references"
# The jobs that may wait for the engine, and the bytes of finished jobs' mp4s
# kept for download: the oldest finished job goes first.
MAX_QUEUED_JOBS = 4
STORE_BUDGET_BYTES = 4 << 30
MAX_LISTED_JOBS = 100
# A job's progress once each phase has completed; the denoising's grows with
# its steps.
PHASE_PROGRESS = {
    wire.VideoPhase.ENCODE: 2,
    wire.VideoPhase.PREPARE: 4,
    wire.VideoPhase.DENOISE: 90,
    wire.VideoPhase.PREVIEW: 91,
    wire.VideoPhase.DECODE: 97,
    wire.VideoPhase.AUDIO: 98,
    wire.VideoPhase.MUX: 100,
}
# How long a stopping engine may take to end its request and exit, and a
# starting one to announce Ready, the latter as long as a language model's.
CLOSE_SECONDS = 30.0
START_SECONDS = supervision.START_TIMEOUT_SECONDS
JOB_ID = re.compile(r"/v1/videos/(video_[0-9a-f]{32})(/content)?")


def serves(model_root: Path):
    """Whether the assembly at model_root holds a video model."""
    record = assembly.recorded(model_root)
    return record is not None and record["target_format"] == H3.target_format


class VideoFailed(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class VideoStartFailed(VideoFailed):
    """The engine's StartFailedEvent, in place of its Ready."""

    def __init__(self, event):
        super().__init__("start_failed", event.message.decode("utf-8", "replace"))
        self.event = event


def canvas(size):
    """The (width, height) of a `size` field, or APIError(400) for one
    outside both tiers."""
    match = SIZE_PATTERN.fullmatch(size) if isinstance(size, str) else None
    if match:
        width, height = int(match[1]), int(match[2])
        short, long = min(width, height), max(width, height)
        if (
            width % H3.canvas_step == 0
            and height % H3.canvas_step == 0
            and short in H3.canvas_tiers
            and long <= 2 * short
        ):
            return width, height
    raise APIError(
        400,
        "size must be WIDTHxHEIGHT of a 768p or 480p canvas: both multiples of "
        "32, the short side 768 or 480 and the long at most twice it, such as "
        + ", ".join(H3.sizes),
    )


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    seconds: str
    variant: str
    # The DiT the engine runs (the variant's row: dit-<dit>/).
    dit: str
    seed: int
    steps: int
    preview: bool
    audio: bool
    size: str = H3.size
    # Its keyframes on the canvas, uint8 RGB, or empty.
    first_frame: bytes = field(default=b"", repr=False)
    last_frame: bytes = field(default=b"", repr=False)
    # Its Ref2VA references, normalized (references.Normalized), in
    # presentation order.
    references: tuple = field(default=(), repr=False)


def _integer(fields, name, default, maximum):
    value = fields.get(name, default)
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if type(value) is not int or not 0 <= value <= maximum:
        raise APIError(400, f"{name} must be an integer in [0, {maximum}]")
    return value


def _boolean(fields, name, default):
    value = fields.get(name, default)
    if isinstance(value, str) and value in ("true", "false"):
        value = value == "true"
    if not isinstance(value, bool):
        raise APIError(400, f"{name} must be a boolean")
    return value


def _keyframes(fields, width, height):
    """The request's first and last frames on its canvas, or empty: the
    image bytes of input_reference and last_frame, the first stretched onto
    the canvas and the last cropped to it, as MiniMax-H3's port does."""
    if "last_frame" in fields and "input_reference" not in fields:
        raise APIError(400, "last_frame needs input_reference, the clip's first frame")
    keyframes = []
    for name in KEYFRAMES:
        payload = fields.get(name)
        if payload is None:
            keyframes.append(b"")
            continue
        if not isinstance(payload, bytes):
            raise APIError(
                400, f"{name} must be an image file of a multipart/form-data body"
            )
        try:
            stretch = name == "input_reference"
            keyframes.append(images.keyframe(payload, width, height, stretch))
        except images.ImageError as error:
            raise APIError(400, f"{name}: {error}") from error
    return keyframes


def _references(fields, frames, width, height):
    """The request's references, decoded and normalized
    (references.normalize), or none: the files of the repeated multipart
    field `references`, each an image or a video by its content type."""
    files = fields.get(REFERENCES)
    if files is None:
        return ()
    if not isinstance(files, (list, tuple)) or not all(
        isinstance(f, refs.ReferenceFile) for f in files
    ):
        raise APIError(
            400,
            "references must be image, video or audio files of a multipart/form-data body",
        )
    if not files:
        return ()
    if any(name in fields for name in KEYFRAMES):
        raise APIError(
            400, "references condition a clip without input_reference or last_frame"
        )
    return tuple(refs.normalize(files, frames, width, height))


def clip_seconds(value):
    """A request's `seconds`, a whole number (or its decimal string) from 4 to
    15, or the golden clips' 3 under SPLASH_GOLDEN_CLIPS=1, as its decimal
    string."""
    text = str(value)
    if (
        isinstance(value, (int, str))
        and not isinstance(value, bool)
        and SECONDS_PATTERN.fullmatch(text)
        and (int(text) in H3.seconds or int(text) == GOLDEN_SECONDS)
    ):
        return text
    allowed = f"from {H3.seconds[0]} to {H3.seconds[-1]}"
    if GOLDEN_SECONDS is not None:
        allowed += f" or {GOLDEN_SECONDS}"
    raise APIError(400, f"seconds must be a whole number {allowed}")


def clip_frames(seconds):
    """The frames of a clip of `seconds` (a whole number or its decimal
    string): the smallest 17 n + 5 at least seconds x 24, Pika's
    align_num_frames of its duration x FPS."""
    frames = int(seconds) * H3.fps
    while frames % H3.frame_period != H3.frame_offset:
        frames += 1
    return frames


def parse_request(fields, names, variants=H3.variants):
    """A POST /v1/videos body's fields (JSON, or form fields, all strings but
    the keyframes' image bytes and the references' files) as a
    VideoRequest; `names` are the names the model answers to, and its
    installed `variants` (name -> families.Variant), of which a request
    without one takes H3.default_variant, or else the first. A request with
    references takes a variant whose spec has references (REF2VA_VARIANT),
    which takes no other. A VDN-H3 variant's steps (recipe h3) are its
    row's; a FastH3 one's 0 leaves its trained rungs to the engine."""
    if unknown := sorted(set(fields) - FIELDS):
        raise APIError(
            400, f"unsupported parameter: {unknown[0]}", "unsupported_parameter"
        )
    prompt = fields.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise APIError(400, "prompt must be a non-empty string")
    if fields.get("model", names[0]) not in names:
        raise APIError(
            404, f"model {fields['model']!r} is not served", "model_not_found"
        )
    seconds = clip_seconds(fields.get("seconds", H3.default_seconds))
    size = fields.get("size", H3.size)
    width, height = canvas(size)
    referenced = bool(fields.get(REFERENCES))
    if referenced:
        default = next(
            (n for n, v in variants.items() if _takes_references(v)), REF2VA_VARIANT
        )
    else:
        default = (
            H3.default_variant
            if H3.default_variant in variants
            else next(iter(variants))
        )
    variant = fields.get("variant", default)
    if variant not in H3.variants and variant not in variants:
        raise APIError(
            400, "variant must be one of " + ", ".join(H3.variants | variants)
        )
    row = variants.get(variant) or H3.variants[variant]
    if referenced and not _takes_references(row):
        raise APIError(
            400,
            f"references condition the {REF2VA_VARIANT} variant (MiniMax-H3's Ref2VA base), not {variant}",
        )
    if _takes_references(row) and not referenced:
        raise APIError(
            400,
            f"the {variant} variant takes references: one image or video at least",
        )
    steps = _integer(fields, "steps", 0, 1000)
    if steps not in (0, row.steps):
        raise APIError(
            400,
            f"steps must be 0 or {row.steps}, the {variant} variant's",
        )
    return VideoRequest(
        prompt,
        seconds,
        variant,
        row.dit,
        _integer(fields, "seed", 0, 2**32 - 1),
        steps or (row.steps if row.recipe == "h3" else 0),
        _boolean(fields, "preview", False),
        _boolean(fields, "audio", True),
        f"{width}x{height}",
        *_keyframes(fields, width, height),
        _references(fields, clip_frames(seconds), width, height),
    )


def _takes_references(row):
    """Whether a variant's DiT runs reference rows: its spec's references."""
    return bool(row.spec and row.spec["references"])


def installed_variants(model_root: Path):
    """The variants the assembly at model_root serves, name ->
    families.Variant (video.installed)."""
    return video.installed(model_root)


def require_variant(request, model, variants):
    """Refuses a request of a variant not installed: each variant is a model
    of its own (install/video.py)."""
    if request.variant not in variants:
        repo = model.partition(":")[0]
        raise APIError(
            400,
            f"the {request.variant} variant of {model} is not installed: "
            f"serve {repo}:{request.variant}",
        )


# Qwen3-VL's processor's image grid (MiniMax-H3's processor: patch 16, merge
# 2, smart_resize factor 32, the canvas step, over 65,536 to 16,777,216
# pixels).
VISION_PIXELS = (65536, 16777216)


def image_tokens(width, height):
    """The <|image_pad|> tokens of a keyframe on a width x height canvas: one
    per 2 x 2 merged 16-pixel patches. The processor's smart_resize leaves a
    canvas as it is (both sides multiples of 32, its area within the range);
    the engine's vision tower takes no other."""
    if (
        width % H3.canvas_step
        or height % H3.canvas_step
        or not (VISION_PIXELS[0] <= width * height <= VISION_PIXELS[1])
    ):
        raise APIError(
            400, f"a {width}x{height} keyframe would be resized by the vision processor"
        )
    return (width // H3.canvas_step) * (height // H3.canvas_step)


def presentation(tokenizer, prompt, keyframes=0, size=None):
    """MiniMax-H3's presentation of a request (diffusers
    MiniMaxH3FL2VATextEncoderStep, Pika H3 Flex's i2va): per keyframe, in
    packed order (the first frame, then the last), the label "<Picture i>: "
    and a vision block, <|vision_start|>, one <|image_pad|> per merged token
    of the keyframe on the canvas, <|vision_end|>; then the prompt verbatim;
    no chat template, no special tokens. The engine's text encoder runs the
    vision tower on the keyframes into the pad rows and tags the blocks' rows
    as video. SPLASH_VIDEO_PROMPT_ONLY=1 keeps the earlier presentation of a
    clip from keyframes, the prompt alone (the reference first-frame clip's
    presentation)."""
    tokens = []
    if keyframes and os.environ.get("SPLASH_VIDEO_PROMPT_ONLY") != "1":
        pads = image_tokens(*size)
        for index in range(keyframes):
            tokens += tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)[
                "input_ids"
            ]
            tokens += [H3.vision_start] + [H3.image_pad] * pads + [H3.vision_end]
    return tokens + list(tokenizer(prompt, add_special_tokens=False)["input_ids"])


# MiniMax-H3's Ref2VA presentation (diffusers MiniMaxH3Ref2VATextEncoderStep):
# reference videos are normalized to 24 fps (H3.fps) and read by Qwen3-VL at
# 2 fps, their sampled frames merged in pairs (the temporal patch), one vision
# block of <|video_pad|> per pair.
VIDEO_SAMPLE_FPS, TEMPORAL_PATCH = 2.0, 2


def video_blocks(frames, fps=H3.fps, sample_fps=VIDEO_SAMPLE_FPS):
    """A reference video's vision blocks (diffusers
    _sample_video_condition_frames): every fps / sample_fps-th of its frames,
    Python-rounded and deduplicated, padded to an even count by repeating the
    last; per merged pair (first frame index, second frame index, timestamp),
    the timestamp the mean of the pair's sample times i / sample_fps. The
    engine's TextEncoder::videoBlockFrames returns the same."""
    stride = fps / sample_fps
    indices, cursor = [], 0.0
    while round(cursor) < frames:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    if len(indices) < TEMPORAL_PATCH:
        raise APIError(
            400,
            f"a reference video of {frames} frames is too short for the vision tower",
        )
    blocks = []
    for i in range(0, len(indices), TEMPORAL_PATCH):
        j = min(i + TEMPORAL_PATCH - 1, len(indices) - 1)
        blocks.append((indices[i], indices[j], (i / sample_fps + j / sample_fps) / 2))
    return blocks


def ref2va_presentation(tokenizer, prompt, references):
    """MiniMax-H3's Ref2VA presentation's token ids (diffusers
    MiniMaxH3Ref2VATextEncoderStep._build_presentation): per reference, in
    packed order, numbered per modality, "<Audio j>: " for a soundtrack (a
    video's before its "<Video k>: "), "<Picture i>: " and an image block, or
    "<Video k>: " and per merged frame pair "<t seconds>" (Python's "{:.1f}",
    half to even) and a block of <|video_pad|>; then the prompt verbatim; no
    chat template, no special tokens. A block holds one pad per merged token of
    its image or frame (image_tokens: Qwen3-VL's processors leave every tier
    size and canvas as it is). The engine's text encoder takes the vision
    blocks' images in token order, a video block's two frames per
    video_blocks."""

    def text(value):
        return list(tokenizer(value, add_special_tokens=False)["input_ids"])

    tokens, counts = [], {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.kind not in counts:
            raise APIError(400, f"a reference of kind {reference.kind!r}")
        if reference.has_audio:
            counts["audio"] += 1
            tokens += text(f"<Audio {counts['audio']}>: ")
        if reference.kind == "image":
            counts["image"] += 1
            tokens += text(f"<Picture {counts['image']}>: ")
            tokens += (
                [H3.vision_start]
                + [H3.image_pad] * image_tokens(*reference.size)
                + [H3.vision_end]
            )
        elif reference.kind == "video":
            counts["video"] += 1
            tokens += text(f"<Video {counts['video']}>: ")
            pads = image_tokens(*reference.size)
            for _, _, seconds in video_blocks(reference.frames):
                tokens += text(f"<{seconds:.1f} seconds>")
                tokens += [H3.vision_start] + [H3.video_pad] * pads + [H3.vision_end]
    return tokens + text(prompt)


def prompt_tokens(tokenizer, prompt, ready, request=None):
    """The presentation's token ids, as the text encoder of the engine that
    announced ready takes them: the prompt's, and with request's keyframes
    their labels and vision blocks (presentation), or with its references
    theirs (ref2va_presentation). Its text rows are at most H3.max_text_tokens
    (with keyframes, all its rows), and all its rows within the engine's
    bound."""
    if request is not None and request.references:
        tokens = ref2va_presentation(tokenizer, prompt, request.references)
        # The vision blocks' rows, their start and end included, are the video's (TextEncoder::tags).
        text = sum(
            t not in (H3.image_pad, H3.video_pad, H3.vision_start, H3.vision_end)
            for t in tokens
        )
        if text > H3.max_text_tokens or len(tokens) > ready.max_context_tokens:
            raise APIError(
                400,
                f"the presentation is {len(tokens)} tokens ({text} of text); the text "
                f"encoder takes {H3.max_text_tokens} of text and {ready.max_context_tokens} in all",
            )
        return tuple(tokens)
    keyframes = (
        0 if request is None else bool(request.first_frame) + bool(request.last_frame)
    )
    size = None if request is None else canvas(request.size)
    tokens = presentation(tokenizer, prompt, keyframes, size)
    bound = min(H3.max_text_tokens, ready.max_context_tokens)
    if not 2 <= len(tokens) <= bound:
        raise APIError(
            400,
            f"the prompt is {len(tokens)} tokens; the text encoder takes 2 to {bound}",
        )
    return tuple(tokens)


def wire_references(references):
    """The normalized references as the engine takes them: their pixels
    and soundtracks as views, copied once into the request's frame."""
    out = []
    for r in references:
        sound = (
            b""
            if r.soundtrack is None
            else memoryview(r.soundtrack.astype("<f4", copy=False))
        )
        out.append(
            wire.VideoReference(
                wire.VideoReferenceKind[r.kind.upper()],
                r.frames,
                *r.size,
                memoryview(r.pixels) if r.pixels.size else b"",
                sound,
            )
        )
    return tuple(out)


def clip(request, tokens):
    """A VideoRequestFrame's fields but its id: the clip of request."""
    outputs = wire.VideoOutputFlag(0)
    if request.preview:
        outputs |= wire.VideoOutputFlag.PREVIEW
    if request.audio:
        outputs |= wire.VideoOutputFlag.AUDIO
    width, height = canvas(request.size)
    return (request.dit, outputs, clip_frames(request.seconds), width,
            height, request.steps, request.seed, tokens, request.first_frame,
            request.last_frame, wire_references(request.references))  # fmt: skip


def load(model_root, binary, startup_timeout):
    """The engine of the assembly at model_root, started and ready, and the
    tokenizer its prompts are encoded with."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_root / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    engine = VideoEngine(
        [str(binary), "serve-native", str(model_root)], startup_timeout
    )
    engine.start()
    return engine, tokenizer


def form_fields(content_type, payload):
    """The fields of a multipart/form-data body, as OpenAI's SDK sends a
    video's: its text fields, the keyframes' files as their bytes, and the
    references' files, in their order, as references.ReferenceFile of their
    content types; any other file field is refused."""
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + bytes(payload)
    )
    if not message.is_multipart():
        raise APIError(400, "invalid multipart/form-data body")
    fields = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name is None or (
            part.get_filename() is not None and name not in (*KEYFRAMES, REFERENCES)
        ):
            raise APIError(
                400, f"{name or 'a field'} is not supported", "unsupported_parameter"
            )
        if name == REFERENCES:
            if part.get_filename() is None:
                raise APIError(400, "references must be image, video or audio files")
            data, declared = part.get_payload(decode=True), part.get_content_type()
            # A file of no type or of application/octet-stream has the one its
            # bytes or its name give.
            if part["content-type"] is None or declared == "application/octet-stream":
                declared = refs.content_type(data, part.get_filename())
            fields.setdefault(REFERENCES, []).append(refs.ReferenceFile(declared, data))
        elif name in KEYFRAMES:
            fields[name] = part.get_payload(decode=True)
        else:
            fields[name] = part.get_content().strip()
    return fields


@dataclass
class Job:
    id: str
    model: str
    request: VideoRequest
    # The prompt's tokens, as the text encoder takes them.
    tokens: tuple
    created_at: int = field(default_factory=lambda: int(time.time()))
    status: str = "queued"
    progress: int = 0
    phase: wire.VideoPhase | None = None
    completed_at: int | None = None
    error: dict | None = None
    # Each phase's seconds and the engine's peak footprint at its end.
    phases: dict = field(default_factory=dict)
    preview: bytes | None = None
    video: bytes | None = None

    @property
    def finished(self):
        return self.status in ("completed", "failed")

    @property
    def stored_bytes(self):
        return len(self.preview or b"") + len(self.video or b"")

    def public(self):
        """The video object, with OpenAI's fields and Splash's beside them:
        its phase, its request's options and each phase's timing."""
        return {
            "id": self.id,
            "object": "video",
            "model": self.model,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "expires_at": None,
            "size": self.request.size,
            "seconds": self.request.seconds,
            "remixed_from_video_id": None,
            "error": self.error,
            "phase": self.phase.name.lower() if self.phase else None,
            "variant": self.request.variant,
            "seed": self.request.seed,
            "preview": self.preview is not None,
            "phases": self.phases,
        }


class VideoEngine:
    """The video engine process on the native protocol: started with the
    server, and again for the next job once it has exited."""

    def __init__(self, command, startup_timeout):
        self.command = command
        self.startup_timeout = startup_timeout
        self.process = None
        self.parser = None
        self.pending = b""
        self.ready = None
        self.request_ids = itertools.count(1)
        # Writes to the engine, and the request in flight and whether it is
        # cancelled, which a cancel from another thread reads.
        self.lock = threading.Lock()
        self.current = None
        self.cancelled = False

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def start(self):
        """Start the engine unless it runs, and wait for its Ready."""
        if self.alive():
            return
        if self.process is not None:
            self.close()
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
        )
        self.parser, self.pending = wire.FrameParser(), b""
        try:
            self.ready = self._await_ready()
        except BaseException:
            # An engine that did not start is stopped, not left running.
            self.close()
            raise

    def _await_ready(self):
        deadline = time.monotonic() + self.startup_timeout
        ready = self._event(deadline)
        # Its warm-up's steps come before Ready, within the same deadline.
        while isinstance(ready, wire.WarmingEvent):
            ready = self._event(deadline)
        # An engine that cannot start sends its reason in place of Ready.
        if isinstance(ready, wire.StartFailedEvent):
            raise VideoStartFailed(ready)
        if isinstance(ready, wire.ErrorEvent):
            raise VideoFailed(ready.code.decode(), ready.message.decode())
        if not isinstance(ready, wire.ReadyEvent) or not ready.video:
            raise VideoFailed("video_engine_failed", "the engine serves no video model")
        return ready

    def run(self, request, on_event):
        """The clip of request, a VideoRequestFrame's fields but its id: its
        VideoOutputEvent once every other event went to on_event; raises
        VideoFailed with the engine's error. A cancel() before it is sent
        cancels it as it is sent."""
        try:
            self.start()
            frame = wire.VideoRequestFrame(next(self.request_ids), *request)
            with self.lock:
                self.current = frame.request_id
                self._send(frame)
                if self.cancelled:
                    self._send(wire.CancelFrame(frame.request_id))
            while True:
                event = self._event(None)
                if isinstance(event, wire.ErrorEvent):
                    if event.failure_class is not wire.FailureClass.REQUEST_ERROR:
                        self.close()
                    raise VideoFailed(event.code.decode(), event.message.decode())
                if (
                    isinstance(event, wire.VideoOutputEvent)
                    and event.output is wire.VideoOutputKind.VIDEO
                ):
                    return event
                on_event(event)
        finally:
            with self.lock:
                self.current, self.cancelled = None, False

    def cancel(self):
        """Cancel the request in flight, or the next one run() sends."""
        with self.lock:
            self.cancelled = True
            if self.current is not None:
                self._send(wire.CancelFrame(self.current))

    def stop(self):
        """End the engine's input: it cancels its request and exits, or is
        killed after CLOSE_SECONDS."""
        if self.alive():
            with self.lock:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=CLOSE_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def close(self):
        """Stop the engine and close its pipes, once nothing reads it: an
        engine that exited by itself still holds its input."""
        self.stop()
        if self.process is not None:
            with self.lock:
                self.process.stdin.close()
            self.process.stdout.close()

    def _send(self, message):
        """Write a frame; call it with the lock held."""
        try:
            self.process.stdin.write(wire.serialize_message(message))
        except (OSError, ValueError) as error:
            raise VideoFailed(
                "video_engine_failed", f"the engine stopped: {error}"
            ) from None

    def _event(self, deadline):
        """The engine's next event, by deadline (time.monotonic) if given."""
        while True:
            while self.pending:
                step = self.parser.consume(self.pending)
                self.pending = self.pending[step.consumed_bytes :]
                if step.issue:
                    raise VideoFailed("video_engine_failed", step.issue.describe())
                if step.frame:
                    return wire.decode_frame(step.frame)
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not select.select([self.process.stdout], [], [], wait)[0]:
                raise VideoFailed(
                    "video_engine_failed", "the engine did not start in time"
                )
            chunk = os.read(self.process.stdout.fileno(), 1 << 20)
            if not chunk:
                status = self.process.wait()
                raise VideoFailed(
                    "video_engine_failed", f"the engine exited with status {status}"
                )
            self.pending += chunk


class VideoService:
    """Admits jobs, runs them one at a time on the engine, and keeps their
    results within the store's budget."""

    def __init__(
        self,
        engine,
        tokenizer,
        names,
        *,
        variants=H3.variants,
        budget_bytes=STORE_BUDGET_BYTES,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.names = names
        self.variants = variants
        self.budget_bytes = budget_bytes
        self.jobs = OrderedDict()
        self.queue = []
        self.running = None
        self.evictions = 0
        self.closing = False
        self.lock = threading.Condition()
        self.worker = threading.Thread(
            target=self._work, name="splash-videos", daemon=True
        )
        self.worker.start()

    def submit(self, fields):
        request = parse_request(fields, self.names, self.variants)
        require_variant(request, self.names[0], self.variants)
        tokens = prompt_tokens(
            self.tokenizer, request.prompt, self.engine.ready, request
        )
        with self.lock:
            if self.closing:
                raise APIError(503, "the server is shutting down", "server_shutdown")
            if len(self.queue) >= MAX_QUEUED_JOBS:
                raise APIError(
                    503,
                    f"{MAX_QUEUED_JOBS} video jobs are already queued",
                    "video_queue_full",
                )
            job = Job("video_" + secrets.token_hex(16), self.names[0], request, tokens)
            self.jobs[job.id] = job
            self.queue.append(job)
            self.lock.notify_all()
            return job.public()

    def get(self, job_id):
        with self.lock:
            return self._job(job_id).public()

    def list(self, query):
        try:
            limit = int(query.get("limit", 20))
        except ValueError:
            raise APIError(400, "limit must be an integer") from None
        if not 1 <= limit <= MAX_LISTED_JOBS:
            raise APIError(400, f"limit must be in [1, {MAX_LISTED_JOBS}]")
        order = query.get("order", "desc")
        if order not in ("asc", "desc"):
            raise APIError(400, "order must be asc or desc")
        with self.lock:
            jobs = list(self.jobs.values())
            if order == "desc":
                jobs.reverse()
            if (after := query.get("after")) is not None:
                ids = [job.id for job in jobs]
                if after not in ids:
                    raise APIError(404, "video not found", "not_found")
                jobs = jobs[ids.index(after) + 1 :]
            data = [job.public() for job in jobs[:limit]]
        return {
            "object": "list",
            "data": data,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
            "has_more": len(jobs) > limit,
        }

    def content(self, job_id, variant):
        """The mp4 of a completed job, or its preview once decoded."""
        if variant not in ("video", "preview"):
            raise APIError(
                400, "variant must be video or preview", "unsupported_parameter"
            )
        with self.lock:
            job = self._job(job_id)
            data = job.preview if variant == "preview" else job.video
        if data is None:
            raise APIError(
                404 if job.finished else 409,
                f"the {variant} of {job_id} is not available (status {job.status})",
                "video_not_ready",
            )
        return data

    def delete(self, job_id):
        """Forgets a job, cancelling it if it has not finished."""
        with self.lock:
            job = self._job(job_id)
            del self.jobs[job_id]
            if job in self.queue:
                self.queue.remove(job)
            elif job is self.running:
                self.engine.cancel()
        return {"id": job_id, "object": "video.deleted", "deleted": True}

    def status(self):
        with self.lock:
            finished = [job for job in self.jobs.values() if job.finished]
            return {
                "queued": len(self.queue),
                "running": self.running.id if self.running else None,
                "stored": len(finished),
                "stored_bytes": sum(job.stored_bytes for job in finished),
                "budget_bytes": self.budget_bytes,
                "evictions": self.evictions,
                "engine": "ready" if self.engine.alive() else "stopped",
            }

    def close(self):
        """Drop the queued jobs and stop the engine, which ends the running
        one."""
        with self.lock:
            self.closing = True
            self.queue.clear()
            self.lock.notify_all()
        self.engine.stop()
        self.worker.join()
        self.engine.close()

    def _job(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            raise APIError(404, "video not found", "not_found")
        return job

    def _work(self):
        while True:
            with self.lock:
                while not self.queue and not self.closing:
                    self.lock.wait()
                if self.closing:
                    return
                job = self.running = self.queue.pop(0)
                job.status = "in_progress"
            self._run(job)
            with self.lock:
                # A delete after the engine's run ended cancelled no request:
                # the next one must not take its cancel.
                self.running, self.engine.cancelled = None, False
                self._evict()

    def _run(self, job):
        try:
            output = self.engine.run(
                clip(job.request, job.tokens), lambda event: self._report(job, event)
            )
        except VideoFailed as error:
            self._finish(job, None, {"code": error.code, "message": str(error)})
        else:
            self._finish(job, output, None)

    def _report(self, job, event):
        with self.lock:
            if isinstance(event, wire.VideoOutputEvent):
                job.preview = event.mp4
            else:
                job.phase = event.phase
                start = (
                    0
                    if event.phase == 1
                    else PHASE_PROGRESS[wire.VideoPhase(event.phase - 1)]
                )
                share = (PHASE_PROGRESS[event.phase] - start) * event.step / event.steps
                job.progress = max(job.progress, int(start + share))
                if event.step == event.steps:
                    job.phases[event.phase.name.lower()] = {
                        "seconds": round(event.elapsed_micros / 1e6, 3),
                        "peak_footprint_bytes": event.peak_footprint_bytes,
                    }

    def _finish(self, job, output, error):
        with self.lock:
            job.status = "failed" if error else "completed"
            job.error, job.phase = error, None
            job.completed_at = int(time.time())
            # A finished job keeps its mp4s, not its keyframes or references.
            job.request = replace(
                job.request, first_frame=b"", last_frame=b"", references=()
            )
            if output is not None:
                job.video, job.progress = output.mp4, 100
                job.phases["total"] = {"seconds": round(output.wall_micros / 1e6, 3)}
            print_status(
                f"Video · {job.id} · {job.status}"
                + (
                    f" · {job.phases['total']['seconds']} s"
                    if output
                    else f" · {error['code']}"
                )
            )

    def _evict(self):
        """Drop the oldest finished jobs past the store's budget."""
        finished = [job for job in self.jobs.values() if job.finished]
        total = sum(job.stored_bytes for job in finished)
        for job in finished:
            if total <= self.budget_bytes:
                break
            total -= job.stored_bytes
            del self.jobs[job.id]
            self.evictions += 1


class VideoApp:
    """What a video model's server serves (FrontendServer.app): the model
    under its names, and its jobs on the engine."""

    def __init__(self, model, names, engine, tokenizer, variants=H3.variants):
        self.model = model
        self.names = tuple(dict.fromkeys((model, *names)))
        self.videos = VideoService(engine, tokenizer, self.names, variants=variants)

    @classmethod
    def start(cls, model, names, model_root, binary, startup_timeout):
        """The app of the assembly at model_root, its engine started and
        ready, and its tokenizer loaded."""
        engine, tokenizer = load(model_root, binary, startup_timeout)
        return cls(model, names, engine, tokenizer, installed_variants(model_root))

    def status(self):
        return {"videos": self.videos.status()}

    def close(self):
        self.videos.close()


def handler(base):
    """The request handler of a video model's server: base's HTTP handling
    (FrontendHandler), its own routes."""

    class VideoHandler(base):
        def do_GET(self):
            path = self.route
            if path == "/health":
                self.send_json(200, {"status": "ok"})
            elif path == "/ready":
                ready = self.app.videos.engine.alive()
                self.send_json(
                    200 if ready else 503,
                    {"status": "ready" if ready else "unavailable"},
                )
            elif path == "/status":
                self.send_json(200, self.server.status())
            elif path == "/v1/models" or path.startswith("/v1/models/"):
                self._video_models(path)
            elif path == "/v1/videos":
                self._answer(
                    lambda: self.send_json(200, self.app.videos.list(self._query()))
                )
            elif match := JOB_ID.fullmatch(path):
                job_id, content = match.groups()
                self._answer(
                    lambda: (
                        self._send(
                            200,
                            self.app.videos.content(
                                job_id, self._query().get("variant", "video")
                            ),
                            "video/mp4",
                        )
                        if content
                        else self.send_json(200, self.app.videos.get(job_id))
                    )
                )
            else:
                self._absent()

        def do_POST(self):
            if self.route != "/v1/videos":
                self._absent()
                return
            self._body_reservation = None
            try:
                self._answer(
                    lambda: self.send_json(200, self.app.videos.submit(self._fields()))
                )
            finally:
                if self._body_reservation is not None:
                    self._body_reservation.release()

        def do_DELETE(self):
            if match := JOB_ID.fullmatch(self.route):
                if not match.group(2):
                    self._answer(
                        lambda: self.send_json(
                            200, self.app.videos.delete(match.group(1))
                        )
                    )
                    return
            self._absent()

        def _absent(self):
            self.safe_error(
                APIError(
                    404,
                    f"not found: this server serves the video model {self.app.model} "
                    "at /v1/videos; a language model is served by a server of its own",
                    "not_found",
                )
            )

        def _answer(self, respond):
            try:
                respond()
            except APIError as error:
                self.safe_error(error)
            except TimeoutError:
                self.safe_error(APIError(408, "HTTP I/O timed out", "request_timeout"))

        def _query(self):
            return {
                key: values[-1]
                for key, values in parse_qs(self.path.partition("?")[2]).items()
            }

        def _fields(self):
            """The POST body's fields: JSON, or OpenAI's SDK's multipart form."""
            deadline = time.monotonic() + 600.0
            content_type = self.headers.get("Content-Type", "")
            if self.headers.get_content_type() == "multipart/form-data":
                return form_fields(
                    content_type, self._read_body(deadline, ("multipart/form-data",))
                )
            body = self._read_json_body(deadline)
            return {key: value for key, value in body.items() if value is not None}

        def _video_models(self, path):
            entries = [
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "splash",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["video", "audio"],
                }
                for name in self.app.names
            ]
            if path == "/v1/models":
                self.send_json(200, {"object": "list", "data": entries})
            elif entry := next(
                (e for e in entries if path == f"/v1/models/{e['id']}"), None
            ):
                self.send_json(200, entry)
            else:
                self.safe_error(APIError(404, "model not found", "model_not_found"))

    return VideoHandler


def generate(engine, tokenizer, request, out: Path):
    """Run request on the ready engine as a job runs, its progress on
    stderr, and write its mp4 to out; its preview, when asked, to
    <out stem>.preview.mp4 as soon as it is decoded. Raises VideoFailed with
    the engine's error, APIError for a prompt the encoder refuses."""
    tokens = prompt_tokens(tokenizer, request.prompt, engine.ready, request)
    preview = out.with_suffix(".preview.mp4")

    def report(event):
        if isinstance(event, wire.VideoOutputEvent):
            preview.write_bytes(event.mp4)
            print_status(f"Video · preview · {preview}", error=True)
        else:
            print_status(
                f"Video · {event.phase.name.lower()} · step {event.step}/"
                f"{event.steps} · {event.elapsed_micros / 1e6:.1f} s · peak "
                f"{event.peak_footprint_bytes / 1e9:.2f} GB",
                error=True,
            )

    output = engine.run(clip(request, tokens), report)
    out.write_bytes(output.mp4)
    print_status(
        f"Video · {out} · {output.frames} frames · {output.wall_micros / 1e6:.1f} s",
        error=True,
    )


def _stop(number, _frame):
    raise KeyboardInterrupt(number)


def main(argv=None):
    """One clip of the video model at MODEL_ROOT: what `splash generate`
    starts, with the POST /v1/videos fields its flags give, its first and
    last frames read from image files, its references (--reference, in
    presentation order) from image and video files, each of the content type
    references.content_type finds. Exits 0 once the mp4 is written, 1 on
    a refusal or the engine's error, and 128 + the signal's number on SIGINT
    or SIGTERM, which cancel the clip at its next step."""
    parser = argparse.ArgumentParser(prog="python -m server.videos")
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request", type=json.loads, required=True)
    parser.add_argument("--first-frame", type=Path)
    parser.add_argument("--last-frame", type=Path)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, _stop)
    engine = None
    try:
        # The launcher blocks both across its exec.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
        fields = dict(args.request)
        for name, path in zip(KEYFRAMES, (args.first_frame, args.last_frame)):
            if path is not None:
                fields[name] = path.read_bytes()
        if args.reference:
            fields[REFERENCES] = [refs.read(path) for path in args.reference]
        if not args.out.parent.is_dir():
            raise APIError(400, f"{args.out.parent} is not a directory")
        # Nothing reads `held`: it keeps the assembly from collection until
        # this process exits, as the server's does.
        model_root, held = assembly.hold(args.model_root)
        if not serves(model_root):
            raise APIError(
                400,
                f"{args.model} is a language model: splash generate takes a video "
                f"model, such as {H3.release}; splash serve serves this one",
            )
        variants = installed_variants(model_root)
        request = parse_request(fields, (args.model,), variants)
        require_variant(request, args.model, variants)
        print_status(H3.license, error=True)
        print_status(f"Loading · {args.model}", error=True)
        engine, tokenizer = load(model_root, args.binary, START_SECONDS)
        failure = []

        def run():
            try:
                generate(engine, tokenizer, request, args.out)
            except Exception as error:
                failure.append(error)

        # The clip runs beside this thread, which takes the stop signals and
        # cancels it.
        worker = threading.Thread(target=run, name="splash-generate", daemon=True)
        worker.start()
        try:
            worker.join()
        except KeyboardInterrupt:
            print_status("Video · cancelling at the next step", error=True)
            engine.cancel()
            worker.join()
            raise
        if failure:
            raise failure[0]
        return 0
    except (APIError, VideoFailed, models.ModelError, OSError) as error:
        print_status(f"Error · {error}", error=True)
        return 1
    except KeyboardInterrupt as stop:
        return 128 + (stop.args[0] if stop.args else signal.SIGINT)
    finally:
        # A stop during cleanup leaves it to end the engine.
        for number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(number, signal.SIG_IGN)
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())

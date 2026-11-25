"""A video model's server: OpenAI-shaped /v1/videos jobs on the video engine.

`python -m server.server` of a video model's assembly (install/video.py)
serves these routes alone, beside its trial page (video.html), /health,
/ready, /status and /v1/models:
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
on the GPU, the others wait in a short queue in their order of arrival. It
runs for its job alone: the server starts it once as it starts, for the
limits its VideoReady announces, which check requests while no engine runs,
then one per job, which exits once its job has ended. A
job's progress follows the engine's phases, its denoising step by step; its
preview and mp4 stay in a store of bounded bytes, which evicts the oldest
finished job first; DELETE cancels a job, a running one at its next DiT block or decode clip.
`splash generate` sends its clip to such a server, one it starts itself
where none runs (install/launcher.py).
"""

from __future__ import annotations

import email.parser
import email.policy
import itertools
import os
import re
import secrets
import select
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import parse_qs

from install import assembly, video
from install.families import H3

from . import images, protocol, supervision
from . import references as refs
from . import video_protocol as wire
from .diagnostics import print_status
from .errors import APIError

# OpenAI's `seconds` names a clip: a whole number of seconds (H3.seconds),
# and for the golden-clip tests alone (SPLASH_GOLDEN_CLIPS=1) the 3 s golden
# clips'.
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
# The trial page, which a video model's server serves in place of the chat
# page, at its routes and as it does: public, and gone with --no-webui.
VIDEO_HTML = Path(__file__).with_name("video.html").read_bytes()


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


# The vision blocks' rows, their start and end included: the video's
# (TextEncoder::tags); every other row is text.
VISION_ROWS = frozenset((H3.image_pad, H3.video_pad, H3.vision_start, H3.vision_end))


def bounded(tokens, ready):
    """tokens as a tuple, or APIError(400) for a presentation the text encoder
    of the engine that announced ready does not take: 2 to its bound of rows,
    at most H3.max_text_tokens of them text (Pika H3 Flex's encode_text, for
    a clip from keyframes and a Ref2VA clip alike)."""
    text = sum(t not in VISION_ROWS for t in tokens)
    if not 2 <= len(tokens) <= ready.max_prompt_tokens or text > H3.max_text_tokens:
        raise APIError(
            400,
            f"the presentation is {len(tokens)} tokens ({text} of text); the text "
            f"encoder takes 2 to {ready.max_prompt_tokens}, at most "
            f"{H3.max_text_tokens} of text",
        )
    return tuple(tokens)


def prompt_tokens(tokenizer, prompt, ready, request=None):
    """The presentation's token ids, as the text encoder of the engine that
    announced ready takes them (bounded): the prompt's, and with request's
    keyframes their labels and vision blocks (presentation), or with its
    references theirs (ref2va_presentation)."""
    if request is not None and request.references:
        return bounded(
            ref2va_presentation(tokenizer, prompt, request.references), ready
        )
    keyframes = (
        0 if request is None else bool(request.first_frame) + bool(request.last_frame)
    )
    size = None if request is None else canvas(request.size)
    return bounded(presentation(tokenizer, prompt, keyframes, size), ready)


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
            # A browser's FormData sends a text field's UTF-8 bytes with no
            # charset, which get_content() would read as ASCII.
            charset = part.get_content_charset() or "utf-8"
            try:
                fields[name] = part.get_payload(decode=True).decode(charset).strip()
            except (LookupError, UnicodeError):
                raise APIError(400, f"{name} is not {charset} text") from None
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
    """The video engine process on its own stream (video_protocol): started
    for a job and closed once it ends, until shut_down."""

    def __init__(self, command, startup_timeout):
        self.command = command
        self.startup_timeout = startup_timeout
        self.process = None
        self.parser = None
        self.pending = b""
        # Whether the engine's first header is still to come: a start that
        # fails before it runs as the video engine says why in the engine
        # protocol.
        self.first = True
        self.ready = None
        self.request_ids = itertools.count(1)
        # Writes to the engine, and the request in flight and whether it is
        # cancelled, which a cancel from another thread reads.
        self.lock = threading.Lock()
        self.current = None
        self.cancelled = False
        # Set by shut_down: no engine starts after it.
        self.shut = False

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def start(self):
        """Start the engine unless it runs, and wait for its Ready."""
        if self.alive():
            return
        if self.process is not None:
            self.close()
        with self.lock:
            if self.shut:
                raise VideoFailed("server_shutdown", "the server is shutting down")
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
            )
        self.parser, self.pending, self.first = wire.FrameParser(), b"", True
        try:
            self.ready = self._await_ready()
        except BaseException:
            # An engine that did not start is stopped, not left running.
            self.close()
            raise

    def _await_ready(self):
        ready = self._event(time.monotonic() + self.startup_timeout)
        # An engine that cannot start sends its reason in place of Ready.
        if isinstance(ready, protocol.StartFailedEvent):
            raise VideoStartFailed(ready)
        if isinstance(ready, protocol.ErrorEvent):
            raise VideoFailed(ready.code.decode(), ready.message.decode())
        if not isinstance(ready, wire.VideoReadyEvent):
            raise VideoFailed("video_engine_failed", "the engine sent no VideoReady")
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
                    self._send(protocol.CancelFrame(frame.request_id))
            while True:
                event = self._event(None)
                if isinstance(event, protocol.ErrorEvent):
                    if event.failure_class is not protocol.FailureClass.REQUEST_ERROR:
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
                self._send(protocol.CancelFrame(self.current))

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

    def shut_down(self):
        """Stop the engine, and start none from here on."""
        with self.lock:
            self.shut = True
        self.stop()

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
            # The first header's magic says which stream the engine speaks.
            magic = len(wire.ENGINE_MAGIC)
            while self.pending and (not self.first or len(self.pending) >= magic):
                if self.first:
                    if wire.starts_engine_protocol(self.pending):
                        return self._start_refusal(deadline)
                    self.first = False
                step = self.parser.consume(self.pending)
                self.pending = self.pending[step.consumed_bytes :]
                if step.issue:
                    raise VideoFailed("video_engine_failed", step.issue.describe())
                if step.frame:
                    return wire.decode_frame(step.frame)
            self._read(deadline)

    def _start_refusal(self, deadline):
        """The StartFailed or Error of a start that failed before it ran as
        the video engine (runtime/main.mm refuseStart), in the engine
        protocol; any other frame there is not the video engine's."""
        parser = protocol.FrameParser()
        while True:
            step = parser.consume(self.pending)
            self.pending = self.pending[step.consumed_bytes :]
            if step.issue:
                raise VideoFailed("video_engine_failed", step.issue.describe())
            if step.frame:
                event = protocol.decode_frame(step.frame)
                if isinstance(event, (protocol.StartFailedEvent, protocol.ErrorEvent)):
                    return event
                raise VideoFailed(
                    "video_engine_failed", "the engine serves no video model"
                )
            self._read(deadline)

    def _read(self, deadline):
        """Add what the engine writes next to pending, by deadline
        (time.monotonic) if given."""
        wait = None if deadline is None else max(0.0, deadline - time.monotonic())
        if not select.select([self.process.stdout], [], [], wait)[0]:
            raise VideoFailed("video_engine_failed", "the engine did not start in time")
        chunk = os.read(self.process.stdout.fileno(), 1 << 20)
        if not chunk:
            status = self.process.wait()
            raise VideoFailed(
                "video_engine_failed", f"the engine exited with status {status}"
            )
        self.pending += chunk


class VideoService:
    """Admits jobs, runs them one at a time, each on an engine of its own,
    and keeps their results within the store's budget."""

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
        # The limits the engine announces (VideoReady), which requests are
        # checked against while no engine runs: one start for them, which
        # also refuses an engine that cannot start, then none until a job.
        engine.start()
        self.ready = engine.ready
        engine.close()
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
        tokens = prompt_tokens(self.tokenizer, request.prompt, self.ready, request)
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
                "engine": "running" if self.engine.alive() else "idle",
            }

    def accepting(self):
        """Whether a job may be submitted: until the server shuts down."""
        with self.lock:
            return not self.closing

    def close(self):
        """Drop the queued jobs and stop the engine, which ends the running
        one."""
        with self.lock:
            self.closing = True
            self.queue.clear()
            self.lock.notify_all()
        self.engine.shut_down()
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
        """Run job on an engine started for it, which exits once it ends."""
        try:
            try:
                self.engine.start()
                self._require_limits(self.engine.ready)
                output = self.engine.run(
                    clip(job.request, job.tokens),
                    lambda event: self._report(job, event),
                )
            finally:
                self.engine.close()
        except VideoFailed as error:
            self._finish(job, None, {"code": error.code, "message": str(error)})
        except OSError as error:
            message = f"the engine did not start: {error}"
            self._finish(job, None, {"code": "video_engine_failed", "message": message})
        else:
            self._finish(job, output, None)

    def _require_limits(self, ready):
        """Refuse an engine whose VideoReady differs from the one the
        server's start announced, which its job's request was checked
        against."""
        if ready != self.ready:
            raise VideoFailed(
                "video_engine_changed",
                "the engine started for this job announces a bound of "
                f"{ready.max_prompt_tokens} prompt tokens and "
                f"{ready.max_concurrent_requests} concurrent requests, the one "
                f"started with the server {self.ready.max_prompt_tokens} and "
                f"{self.ready.max_concurrent_requests}: restart the server",
            )

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
            # A job deleted, or running as the server stops, was cancelled:
            # how its engine then ends, at the cancel or exiting as its input
            # closes, is no failure of the clip.
            cancelled = error is not None and (self.closing or job.id not in self.jobs)
            if cancelled:
                error = {"code": "cancelled", "message": "the clip was cancelled"}
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
            if output:
                outcome = f"{job.status} · {job.phases['total']['seconds']} s"
            elif cancelled:
                outcome = "cancelled"
            else:
                outcome = f"{job.status} · {error['code']}"
            print_status(f"Video · {job.id} · {outcome}")

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
        """The app of the assembly at model_root, its tokenizer loaded and
        the limits its engine announces read (VideoService)."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_root / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        engine = VideoEngine(
            [str(binary), "serve-native", str(model_root)], startup_timeout
        )
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
            if path in ("/", "/index.html", "/favicon.ico"):
                self._get_page()
            elif path == "/health":
                self.send_json(200, {"status": "ok"})
            elif path == "/ready":
                # Ready once started: an idle server runs no engine, and
                # starts one per job.
                ready = self.app.videos.accepting()
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

        def _get_page(self):
            if self.server.webui and self.route != "/favicon.ico":
                self._send(200, VIDEO_HTML, "text/html; charset=utf-8")
            else:
                super()._get_page()

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

"""ComfyUI V3 nodes: clients of a local Splash video server (POST /v1/videos)."""

from __future__ import annotations

import io
import time
from io import BytesIO

import comfy.model_management
import comfy.utils
from comfy_api.latest import IO, ComfyExtension, InputImpl
from server import PromptServer

from . import convert, launcher, options
from .options import MAX_SECONDS, MIN_SECONDS, SEED_MAX
from .splash_client import FilePart, SplashClient, SplashError, SplashInterrupted

CATEGORY = "video/Splash"
PREVIEW_MAX_SIZE = 512
POLL_INTERVAL = 1.0
# The type of an mp4's first box (ISO BMFF), at bytes 4-8 of the file.
MP4_BOX = b"ftyp"


# ---- schema pieces -------------------------------------------------------


def _prompt(tooltip="What happens in the clip, in any language; sent as typed."):
    return IO.String.Input("prompt", multiline=True, default="", tooltip=tooltip)


def _size():
    return IO.Combo.Input(
        "size",
        options=options.SIZES,
        default=options.SIZES[0],
        tooltip="Width x height of the clip in pixels: 768p or 480p, landscape or portrait.",
    )


def _resolution():
    return IO.Combo.Input(
        "resolution",
        options=list(options.RESOLUTIONS),
        default=next(iter(options.RESOLUTIONS)),
        tooltip="768p (1344x768) or 480p (832x480); the first frame's shape picks landscape or portrait.",
    )


def _frame(name, tooltip, optional=False):
    return IO.Image.Input(name, optional=optional, tooltip=tooltip)


def _seconds():
    return IO.Int.Input(
        "seconds", min=MIN_SECONDS, max=MAX_SECONDS, step=1, default=5, display_mode=IO.NumberDisplay.slider,
        tooltip=f"Length of the clip, {MIN_SECONDS} to {MAX_SECONDS} seconds.",
    )


def _seed():
    return IO.Int.Input(
        "seed", min=0, max=SEED_MAX, step=1, default=42, display_mode=IO.NumberDisplay.number,
        control_after_generate=True, tooltip="Random seed of the clip.",
    )


def _advanced(variant: bool):
    inputs = []
    if variant:
        inputs.append(
            IO.Combo.Input(
                "variant",
                options=options.VARIANTS,
                default="auto",
                tooltip=f"Denoising variant; auto lets the server pick (a server started here starts with "
                f"{options.DEFAULT_VARIANT}). The server serves every variant installed when it started.",
                advanced=True,
            )
        )
    inputs += [
        IO.Boolean.Input("generate_audio", default=True, tooltip="Generate the soundtrack.", advanced=True),
        IO.Boolean.Input(
            "preview", default=False, tooltip="Ask the server for a preview and show its middle frame.", advanced=True
        ),
        IO.String.Input(
            "base_url", default=options.DEFAULT_BASE_URL, tooltip="URL of the Splash video server.", advanced=True
        ),
        IO.String.Input("api_key", default="", tooltip="Splash server API key, if it has one.", advanced=True),
        IO.Boolean.Input(
            "auto_start", advanced=True, default=True,
            tooltip="Start `splash serve` with this node's variant when nothing listens at a local "
            "base_url. It is left running.",
        ),
        IO.Boolean.Input(
            "unload_comfy_models", advanced=True, default=True,
            tooltip="Unload ComfyUI's models before the request to free memory for Splash.",
        ),
    ]
    return inputs


# The reference kinds the server takes (references.normalize, server/references.py):
# (input id, slot prefix, ComfyUI socket type, at most, the prompt's label, notes).
REFERENCE_KINDS = (
    ("pictures", "picture", IO.Image, 9, "Picture", "Subjects or styles."),
    ("videos", "video", IO.Video, 3, "Video",
     "Motion or scenes, 2-15 s each and 15 s together; a video's soundtrack takes the next <Audio> number."),
    ("audio_clips", "audio", IO.Audio, 3, "Audio",
     "Voices or music beside a picture or a video, 2-15 s each and 15 s together; numbered after the "
     "videos' soundtracks."),
)


def _references():
    """One growing group of slots per reference kind; the prompt names a
    reference by its label and number, counted per kind in slot order."""
    groups = []
    for group, slot, kind, most, label, notes in REFERENCE_KINDS:
        names = [f"{slot}_{n}" for n in range(1, most + 1)]
        groups.append(
            IO.Autogrow.Input(
                group,
                template=IO.Autogrow.TemplateNames(kind.Input(slot), names=names, min=0),
                tooltip=f"Up to {most}, named <{label} 1>, <{label} 2>, ... in the prompt. {notes}",
            )
        )
    return groups


def _schema(node_id, display_name, description, inputs):
    return IO.Schema(
        node_id=node_id,
        display_name=display_name,
        category=CATEGORY,
        description=description,
        inputs=inputs,
        outputs=[IO.Video.Output()],
        hidden=[IO.Hidden.unique_id],
    )


# ---- execution -----------------------------------------------------------


def _preview_image(mp4: bytes):
    """The middle frame of an mp4 as a PIL image, or None."""
    try:
        import av

        with av.open(io.BytesIO(mp4)) as container:
            stream = container.streams.video[0]
            frames = [frame.to_image() for frame in container.decode(stream)]
        return frames[len(frames) // 2] if frames else None
    except Exception:
        return None


def _log_dir():
    try:
        import folder_paths

        return folder_paths.get_user_directory()
    except Exception:
        return None


def _node_id(cls) -> str:
    """The node's schema id; ComfyUI runs a "<class>Clone" copy of the class."""
    try:
        return cls.define_schema().node_id
    except Exception:
        return cls.__name__


async def _prepare(settings: dict, is_interrupted, text):
    """Make sure the server answers (starting it when allowed), then free
    ComfyUI's models when asked."""
    await launcher.ensure_server(
        settings["base_url"],
        settings["variant"],
        auto_start=settings["auto_start"],
        api_key=settings["api_key"],
        is_interrupted=is_interrupted,
        on_status=text,
        log_dir=_log_dir(),
    )
    if settings["unload_comfy_models"]:
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()


async def _run(cls, settings: dict, fields: dict, files: list) -> IO.NodeOutput:
    node_id = cls.hidden.unique_id
    server = PromptServer.instance

    def text(message):
        if node_id is not None:
            server.send_progress_text(message, node_id)

    is_interrupted = comfy.model_management.processing_interrupted
    bar = comfy.utils.ProgressBar(100)
    try:
        await _prepare(settings, is_interrupted, text)

        started = time.monotonic()
        last = {"progress": 0}

        def on_progress(job):
            progress = int(job.get("progress") or 0)
            last["progress"] = progress
            if job.get("status") == "queued":
                text("queued")
            else:
                phase = job.get("phase") or job.get("status") or ""
                text(f"{phase} · {progress}% · {time.monotonic() - started:.0f}s")
            bar.update_absolute(progress, 100)

        def on_preview(mp4):
            image = _preview_image(mp4)
            if image is not None:
                bar.update_absolute(last["progress"], 100, ("JPEG", image, PREVIEW_MAX_SIZE))

        client = SplashClient(
            settings["base_url"], settings["api_key"], poll_interval=POLL_INTERVAL, node=_node_id(cls),
            serve_model=launcher.model_ref(settings["variant"]),
        )
        mp4 = await client.generate(
            fields, files, on_progress=on_progress, is_interrupted=is_interrupted, on_preview=on_preview
        )
        if mp4[4:8] != MP4_BOX:
            raise SplashError(
                "bad_response",
                f"the clip from the Splash server at {client.base_url} is not an mp4: {len(mp4)} bytes "
                "without an ftyp box at bytes 4-8",
            )
    except SplashInterrupted:
        comfy.model_management.throw_exception_if_processing_interrupted()
        raise comfy.model_management.InterruptProcessingException()
    except SplashError as error:
        raise RuntimeError(str(error)) from error
    return IO.NodeOutput(InputImpl.VideoFromFile(BytesIO(mp4)))


def _settings(base_url, api_key, auto_start, unload_comfy_models, variant):
    """`variant` is the one a server started here starts with."""
    return {
        "base_url": base_url.strip() or options.DEFAULT_BASE_URL,
        "variant": variant,
        "api_key": api_key.strip(),
        "auto_start": bool(auto_start),
        "unload_comfy_models": bool(unload_comfy_models),
    }


class SplashH3TextToVideo(IO.ComfyNode):
    @classmethod
    async def execute(
        cls, prompt, size, seconds, seed, variant="auto", generate_audio=True, preview=False,
        base_url=options.DEFAULT_BASE_URL, api_key="", auto_start=True, unload_comfy_models=True,
    ) -> IO.NodeOutput:
        fields = options.video_fields(prompt, seconds, size, seed, generate_audio, preview, variant)
        settings = _settings(base_url, api_key, auto_start, unload_comfy_models, options.serve_variant(variant))
        return await _run(cls, settings, fields, [])

    @classmethod
    def define_schema(cls):
        return _schema(
            "SplashH3TextToVideo",
            "Splash H3 Text to Video",
            "Generate a video with audio from a text prompt with MiniMax-H3 on a local Splash server.",
            [_prompt(), _size(), _seconds(), _seed(), *_advanced(variant=True)],
        )


class SplashH3FirstLastFrameToVideo(IO.ComfyNode):
    @classmethod
    async def execute(
        cls, prompt, first_frame, resolution, seconds, seed, last_frame=None, variant="auto",
        generate_audio=True, preview=False, base_url=options.DEFAULT_BASE_URL, api_key="", auto_start=True,
        unload_comfy_models=True,
    ) -> IO.NodeOutput:
        size = options.keyframe_size(resolution, *convert.image_size(first_frame))
        fields = options.video_fields(prompt, seconds, size, seed, generate_audio, preview, variant)
        files = [FilePart("input_reference", "first_frame.png", "image/png", convert.image_to_png(first_frame))]
        if last_frame is not None:
            files.append(FilePart("last_frame", "last_frame.png", "image/png", convert.image_to_png(last_frame)))
        settings = _settings(base_url, api_key, auto_start, unload_comfy_models, options.serve_variant(variant))
        return await _run(cls, settings, fields, files)

    @classmethod
    def define_schema(cls):
        return _schema(
            "SplashH3FirstLastFrameToVideo",
            "Splash H3 First-Last-Frame to Video",
            "Generate a video from a first frame and an optional last frame with MiniMax-H3 on a local "
            "Splash server. Landscape or portrait follows the first frame.",
            [
                _prompt(),
                _frame("first_frame", "The picture the clip opens on; its shape sets the orientation."),
                _frame("last_frame", "A picture for the clip to arrive at; leave it unconnected to let the clip "
                       "end freely.", optional=True),
                _resolution(),
                _seconds(),
                _seed(),
                *_advanced(variant=True),
            ],
        )


class SplashH3ReferenceToVideo(IO.ComfyNode):
    @classmethod
    async def execute(
        cls, prompt, size, seconds, seed, pictures=None, videos=None, audio_clips=None, generate_audio=True,
        preview=False, base_url=options.DEFAULT_BASE_URL, api_key="", auto_start=True, unload_comfy_models=True,
    ) -> IO.NodeOutput:
        if not pictures and not videos:
            # The server would refuse audio clips alone, and run no reference job without any.
            raise ValueError("Connect a picture or a video to the Reference node; audio clips go beside one.")
        fields = options.video_fields(prompt, seconds, size, seed, generate_audio, preview)
        files = convert.reference_parts(pictures, videos, audio_clips)
        settings = _settings(base_url, api_key, auto_start, unload_comfy_models, options.REFERENCE_VARIANT)
        return await _run(cls, settings, fields, files)

    @classmethod
    def define_schema(cls):
        return _schema(
            "SplashH3ReferenceToVideo",
            "Splash H3 Reference to Video",
            "Generate a video conditioned on reference images, videos and audio with MiniMax-H3 on a "
            "local Splash server. The prompt refers to them as <Picture 1>, <Video 1>, <Audio 1>, ...",
            [
                _prompt("What happens in the clip, in any language; name the references as <Picture 1>, "
                        "<Video 1>, <Audio 1>, ..."),
                *_references(),
                _size(),
                _seconds(),
                _seed(),
                *_advanced(variant=False),
            ],
        )


NODES = [SplashH3TextToVideo, SplashH3FirstLastFrameToVideo, SplashH3ReferenceToVideo]


class SplashExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return list(NODES)

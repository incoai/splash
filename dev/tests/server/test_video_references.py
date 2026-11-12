"""Ref2VA references in the video server (server/references.py, server/videos.py): their kinds, Pika H3 Flex's limits
and normalization rules, the multipart field, the variant they take, the presentation, and the wire frame.

With SPLASH_REF2VA_RECORD_SHOP (the directory of a Ref2VA example's inputs, which are not part of this repository:
motion-transfer-input-1.mp4, -2.png, -3.png, prompt.txt, and a reference run's saved conditioning.npz of them) and
SPLASH_REF2VA_TOKENIZER (FL2VA's tokenizer), the server's token ids of that clip at 832x480 5 s must be the saved ones.
"""

import importlib.util
import io
import os
import shlex
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from install.families import H3
from server import protocol as wire
from server import references as refs
from server import videos
from server.errors import APIError

# PyAV decodes video and audio references and encodes these tests' clips; the release leaves it out
# (dev/requirements.txt installs it), so the tests that need it skip where it is missing.
needs_pyav = unittest.skipUnless(
    importlib.util.find_spec("av"), "needs PyAV (dev/requirements.txt)"
)


def tokenize(text, add_special_tokens):
    """One token per character: what the presentation's tests need of a tokenizer."""
    return {"input_ids": [ord(c) for c in text]}


def png(width, height, value=128):
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (width, height), (value, 255 - value, 7)).save(out, "PNG")
    return out.getvalue()


def media(seconds, fps=24.0, size=(64, 36), audio=True, video=True, container="mp4"):
    """A small clip encoded with PyAV: frames of a gradient that moves, and a stereo tone."""
    from fractions import Fraction

    import av

    out = io.BytesIO()
    with av.open(out, "w", format=container) as c:
        if video:
            vs = c.add_stream("mpeg4", rate=Fraction(fps).limit_denominator(1001))
            vs.width, vs.height, vs.pix_fmt = size[0], size[1], "yuv420p"
        if audio:
            codec = "mp3" if container == "mp3" else "aac"
            as_ = c.add_stream(codec, rate=44100)
            as_.layout = "stereo"
        if video:
            for i in range(round(seconds * fps)):
                x = np.zeros((size[1], size[0], 3), np.uint8)
                x[..., 0] = (np.arange(size[0]) + 4 * i) % 256
                for p in vs.encode(av.VideoFrame.from_ndarray(x, format="rgb24")):
                    c.mux(p)
            for p in vs.encode():
                c.mux(p)
        if audio:
            n, t = 1024, 0
            fmt = "fltp" if codec == "aac" else "s16p"
            while t < seconds * 44100:
                k = np.arange(t, t + n) / 44100
                tone = (
                    np.stack([np.sin(2 * np.pi * 440 * k), np.sin(2 * np.pi * 660 * k)])
                    * 0.25
                )
                data = (
                    tone.astype(np.float32)
                    if fmt == "fltp"
                    else (tone * 32767).astype(np.int16)
                )
                frame = av.AudioFrame.from_ndarray(data, format=fmt, layout="stereo")
                frame.sample_rate, frame.pts = 44100, t
                for p in as_.encode(frame):
                    c.mux(p)
                t += n
            for p in as_.encode():
                c.mux(p)
    return out.getvalue()


def file(content_type, data):
    return refs.ReferenceFile(content_type, data)


class KindsTest(unittest.TestCase):
    def test_kinds_by_content_type(self):
        self.assertEqual(refs.kind("image/png"), "image")
        self.assertEqual(refs.kind("image/jpeg"), "image")
        for t in ("video/mp4", "video/quicktime", "video/webm"):
            self.assertEqual(refs.kind(t), "video")
        for t in ("audio/mpeg", "audio/wav", "audio/mp4", "audio/aac"):
            self.assertEqual(refs.kind(t), "audio")
        for t in ("application/octet-stream", "image/gif", "text/plain", ""):
            with self.assertRaises(APIError) as caught:
                refs.kind(t)
            self.assertEqual(caught.exception.status, 400)

    def test_content_type_sniffed_from_the_file(self):
        self.assertEqual(refs.content_type(png(4, 4)), "image/png")
        self.assertEqual(
            refs.content_type(b"\xff\xd8\xff\xe0" + bytes(12)), "image/jpeg"
        )
        self.assertEqual(
            refs.content_type(bytes(4) + b"ftypisom" + bytes(4)), "video/mp4"
        )
        self.assertEqual(
            refs.content_type(bytes(4) + b"ftypqt  " + bytes(4)), "video/quicktime"
        )
        self.assertEqual(
            refs.content_type(bytes(4) + b"ftypM4A " + bytes(4)), "audio/mp4"
        )
        self.assertEqual(
            refs.content_type(b"\x1a\x45\xdf\xa3" + bytes(12)), "video/webm"
        )
        self.assertEqual(refs.content_type(b"RIFF\0\0\0\0WAVEfmt "), "audio/wav")
        self.assertEqual(refs.content_type(b"ID3\x04" + bytes(12)), "audio/mpeg")
        self.assertEqual(refs.content_type(b"\xff\xf1" + bytes(14)), "audio/aac")
        self.assertEqual(refs.content_type(b"????", "clip.webm"), "video/webm")


class GeometryTest(unittest.TestCase):
    def test_canvas_rules(self):
        # The example's 16:9 sources: the video on the 480p tier canvas, the images at the tier's 1024 short edge.
        self.assertEqual(
            refs.resolve_canvas_size(1920, 1080, 480, 832 * 480), (480, 832)
        )
        self.assertEqual(
            refs.resolve_canvas_size(1080, 1920, 768, 1344 * 768), (1344, 768)
        )
        self.assertEqual(refs.reference_canvas_size(1920, 1080, 1024), (1024, 1824))
        self.assertEqual(
            refs.reference_canvas_size(500, 400, 2048), (2048, 2560)
        )  # upscaled, no area cap
        with self.assertRaises(APIError):
            refs.reference_canvas_size(1000, 200, 1024)  # 5:1

    def test_video_onto_the_24_fps_grid(self):
        # 30 fps: each frame held until the next one's slot (np.floor(i * 24 / 30 + 0.5)), truncated to the clip.
        frames = np.arange(30, dtype=np.uint8)[:, None, None, None] * np.ones(
            (1, 480, 832, 3), np.uint8
        )
        out = refs.normalize_video(frames, 30.0, 22, 480, 832 * 480)
        slots = np.floor(np.arange(30) * 0.8 + 0.5).astype(int)
        want = np.repeat(np.arange(30), np.diff(slots, append=24))[:22]
        self.assertEqual(out[:, 0, 0, 0].tolist(), want.tolist())
        self.assertIs(out.dtype, np.dtype(np.uint8))


@needs_pyav
class NormalizeTest(unittest.TestCase):
    def test_video_image_and_audio(self):
        normalized = refs.normalize(
            [
                file("video/mp4", media(3.0)),
                file("image/png", png(64, 36)),
                file("audio/mp4", media(4.0, video=False)),
            ],
            124,
            832,
            480,
        )
        video, image, audio = normalized
        self.assertEqual(
            (video.kind, video.frames, video.size), ("video", 72, (832, 480))
        )
        self.assertEqual(video.soundtrack.shape[0], 2)
        self.assertLessEqual(video.soundtrack.shape[1], round(124 / 24 * 32000))
        self.assertEqual(
            (image.kind, image.frames, image.size), ("image", 1, (1824, 1024))
        )
        self.assertIsNone(image.soundtrack)
        self.assertEqual((audio.kind, audio.frames), ("audio", 0))
        self.assertEqual(audio.soundtrack.dtype, np.float32)
        # Truncated to the clip: 5 s of 32 kHz.
        self.assertEqual(audio.soundtrack.shape[0], 2)
        self.assertGreater(audio.soundtrack.shape[1], 3.5 * 32000)

    def test_a_silent_video_has_no_soundtrack(self):
        (video,) = refs.normalize(
            [file("video/mp4", media(2.5, audio=False))], 124, 832, 480
        )
        self.assertIsNone(video.soundtrack)

    def test_audio_is_truncated_to_the_clip(self):
        image = file("image/png", png(32, 32))
        _, audio = refs.normalize(
            [image, file("audio/mpeg", media(8.0, video=False, container="mp3"))],
            124,
            832,
            480,
        )
        self.assertEqual(audio.soundtrack.shape, (2, round(124 / 24 * 32000)))

    def test_limits(self):
        image = file("image/png", png(32, 32))
        cases = [
            ([image] * 10, "at most 9 images"),
            ([file("video/mp4", b"")] * 4, "at most 9 images, 3 videos"),
            ([image] + [file("audio/wav", b"")] * 4, "3 audio clips"),
            (
                [file("audio/mp4", media(3.0, video=False))],
                "an image or video reference",
            ),
            ([file("video/mp4", media(1.5))], "2 to 15 seconds"),
            ([image, file("audio/mp4", media(1.5, video=False))], "2 to 15 seconds"),
            # Decoded for 16 s at most: past 15 s in all, as Pika checks it before each video's length.
            (
                [file("video/mp4", media(16.5, size=(32, 32), audio=False))],
                "15 s at most in all",
            ),
            (
                [file("video/mp4", media(8.0, size=(32, 32), audio=False))] * 2,
                "15 s at most in all",
            ),
            ([file("image/png", b"not a png")], "could not be decoded"),
            ([file("video/mp4", b"not a video")], "could not be decoded"),
        ]
        for files, message in cases:
            with (
                self.subTest(message=message, files=len(files)),
                self.assertRaises(APIError) as caught,
            ):
                refs.normalize(files, 124, 832, 480)
            self.assertEqual(caught.exception.status, 400)
            self.assertIn(message, str(caught.exception))


class PyAVTest(unittest.TestCase):
    def test_without_pyav_video_and_audio_references_name_the_install_command(self):
        image = file("image/png", png(32, 32))
        message = (
            "video and audio references need PyAV, which Splash does not ship; install it with: "
            f"{shlex.quote(sys.executable)} -m pip install av==19.0.2"
        )
        with mock.patch.dict(sys.modules, {"av": None}):
            for files in (
                [file("video/mp4", b"\0\0\0\x18ftypisom")],
                [image, file("audio/mpeg", b"ID3")],
            ):
                with (
                    self.subTest(kind=files[-1].content_type),
                    self.assertRaises(APIError) as caught,
                ):
                    refs.normalize(files, 124, 832, 480)
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(str(caught.exception), message)
            # An image reference never imports it.
            (normalized,) = refs.normalize([image], 124, 832, 480)
        self.assertEqual(normalized.kind, "image")

    def test_the_pin_is_the_development_requirement(self):
        pins = (Path(__file__).parents[2] / "requirements.txt").read_text().splitlines()
        self.assertIn(refs.PYAV, pins)


class RequestTest(unittest.TestCase):
    NAMES = ("MiniMaxAI/MiniMax-H3",)
    VARIANTS = {n: H3.variants[n] for n in ("vdn-8", "pika-6", "pika-ref2va-6")}

    def fields(self, **extra):
        return {
            "prompt": "a record shop",
            "size": "832x480",
            "references": [
                file("video/mp4", media(3.0)),
                file("image/png", png(64, 36)),
            ],
            **extra,
        }

    @needs_pyav
    def test_references_take_the_ref2va_variant(self):
        request = videos.parse_request(self.fields(), self.NAMES, self.VARIANTS)
        self.assertEqual(request.variant, "pika-ref2va-6")
        self.assertEqual([r.kind for r in request.references], ["video", "image"])
        self.assertEqual(videos.clip(request, (1, 2))[0], "pika-ref2va-6")

    def test_a_request_without_a_variant_takes_pika_6_where_installed(self):
        for variants, variant in (
            (("vdn-8", "pika-6"), "pika-6"),
            (("vdn-8", "vdn-50"), "vdn-8"),
        ):
            request = videos.parse_request(
                {"prompt": "x"}, self.NAMES, {n: H3.variants[n] for n in variants}
            )
            self.assertEqual(request.variant, variant)

    @needs_pyav
    def test_refusals(self):
        cases = [
            (
                self.fields(variant="pika-6"),
                "references condition the pika-ref2va-6 variant",
            ),
            (
                self.fields(variant="vdn-8"),
                "references condition the pika-ref2va-6 variant",
            ),
            ({"prompt": "x", "variant": "pika-ref2va-6"}, "takes references"),
            (
                self.fields(input_reference=png(832, 480)),
                "without input_reference or last_frame",
            ),
            (
                self.fields(references=["data:image/png;base64,AAAA"]),
                "multipart/form-data",
            ),
            (self.fields(references=[file("text/plain", b"x")]), "not supported"),
        ]
        for fields, message in cases:
            with self.subTest(message=message), self.assertRaises(APIError) as caught:
                videos.parse_request(fields, self.NAMES, self.VARIANTS)
            self.assertEqual(caught.exception.status, 400)
            self.assertIn(message, str(caught.exception))

    def test_the_multipart_field_repeats_in_order(self):
        boundary = "XyZ"
        parts = [("prompt", None, None, b"a shop"), ("references", "a.png", "image/png", png(8, 8)),
                 ("references", "b.mp4", "video/mp4", b"\0\0\0\x18ftypisom"),
                 ("references", "c.mp3", "audio/mpeg", b"ID3"),
                 # A file of no type or of application/octet-stream: its bytes', else its name's.
                 ("references", "d", None, b"RIFF\0\0\0\0WAVE"),
                 ("references", "e.png", "application/octet-stream", b"x")]  # fmt: skip
        body = b""
        for name, filename, ctype, data in parts:
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'.encode()
            if filename:
                body += f'; filename="{filename}"'.encode()
            if ctype:
                body += f"\r\nContent-Type: {ctype}".encode()
            body += b"\r\n\r\n" + data + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        fields = videos.form_fields(f"multipart/form-data; boundary={boundary}", body)
        self.assertEqual(fields["prompt"], "a shop")
        self.assertEqual(
            [f.content_type for f in fields["references"]],
            ["image/png", "video/mp4", "audio/mpeg", "audio/wav", "image/png"],
        )
        self.assertEqual(fields["references"][2].data, b"ID3")

    @needs_pyav
    def test_the_wire_frame_carries_the_normalized_references(self):
        fields = self.fields()
        fields["references"].append(file("audio/mp4", media(3.0, video=False)))
        request = videos.parse_request(fields, self.NAMES, self.VARIANTS)
        frame = wire.VideoRequestFrame(7, *videos.clip(request, (1, 2)))
        decoded = wire.serialize_message(frame)
        from dev.tests import native_peer

        ((message, _),) = native_peer.ClientFrameReader().feed(bytes(decoded))
        kinds = [r.kind for r in message.references]
        self.assertEqual(kinds, [wire.VideoReferenceKind.VIDEO, wire.VideoReferenceKind.IMAGE,
                                 wire.VideoReferenceKind.AUDIO])  # fmt: skip
        video, image, audio = request.references
        self.assertEqual(message.references[0].pixels, video.pixels.tobytes())
        self.assertEqual(
            message.references[0].soundtrack, video.soundtrack.astype("<f4").tobytes()
        )
        self.assertEqual(
            (message.references[1].width, message.references[1].height), (1824, 1024)
        )
        self.assertEqual(
            message.references[2].soundtrack, audio.soundtrack.astype("<f4").tobytes()
        )
        self.assertEqual(
            (message.references[2].frames, message.references[2].pixels), (0, b"")
        )


class PresentationTest(unittest.TestCase):
    def ready(self, tokens):
        return wire.ReadyEvent(1, tokens, False, 0, 0, (), video=True)

    def test_labels_and_blocks(self):
        sound = np.zeros((2, 1), np.float32)
        presented = [refs.Normalized("video", np.zeros((24, 256, 256, 3), np.uint8), sound),
                     refs.Normalized("audio", refs.NO_PIXELS, sound),
                     refs.Normalized("image", np.zeros((1, 256, 512, 3), np.uint8))]  # fmt: skip
        ids = videos.ref2va_presentation(tokenize, "go", presented)
        text = "".join(chr(t) if t < 0x10000 else "#" for t in ids)
        # A video's soundtrack is labelled before the video, an audio reference by its label alone.
        self.assertTrue(text.startswith("<Audio 1>: <Video 1>: <0.2 seconds>#"))
        self.assertIn("#<Audio 2>: <Picture 1>: #", text)
        self.assertTrue(text.endswith("go"))
        # 24 frames: one pair (frames 0 and 12) of 8 x 8 merged tokens; the image 16 x 8.
        self.assertEqual(ids.count(H3.video_pad), 64)
        self.assertEqual(ids.count(H3.image_pad), 128)

    def test_token_ids_are_the_golden(self):
        sound = np.zeros((2, 1), np.float32)
        presented = [refs.Normalized("video", np.zeros((48, 256, 256, 3), np.uint8), sound),
                     refs.Normalized("audio", refs.NO_PIXELS, sound),
                     refs.Normalized("image", np.zeros((1, 256, 512, 3), np.uint8))]  # fmt: skip
        # Literal ids: 48 frames at 24 fps are two pairs at 0.25 and 1.25 s of 8 x 8 merged <|video_pad|>s, the
        # image 16 x 8 <|image_pad|>s, each between <|vision_start|> and <|vision_end|>.
        video = [151652, *[151656] * 64, 151653]
        self.assertEqual(
            videos.ref2va_presentation(tokenize, "go", presented),
            [*b"<Audio 1>: <Video 1>: <0.2 seconds>", *video, *b"<1.2 seconds>", *video,
             *b"<Audio 2>: <Picture 1>: ", 151652, *[151655] * 128, 151653, *b"go"],
        )  # fmt: skip

    def test_bounds(self):
        tokenizer = tokenize
        request = videos.parse_request(
            {
                "prompt": "x" * 2100,
                "size": "832x480",
                "references": [file("image/png", png(64, 36))],
            },
            ("m",),
            {"pika-ref2va-6": H3.variants["pika-ref2va-6"]},
        )
        with self.assertRaises(APIError) as caught:
            videos.prompt_tokens(tokenizer, request.prompt, self.ready(32768), request)
        self.assertIn("2048 of text", str(caught.exception))
        short = videos.parse_request(
            {"prompt": "x", "size": "832x480", "references": [file("image/png", png(64, 36))]}, ("m",),
            {"pika-ref2va-6": H3.variants["pika-ref2va-6"]},
        )  # fmt: skip
        tokens = videos.prompt_tokens(tokenizer, short.prompt, self.ready(32768), short)
        self.assertEqual(tokens.count(H3.image_pad), 32 * 57)
        with self.assertRaises(APIError):
            videos.prompt_tokens(tokenizer, short.prompt, self.ready(100), short)
        # Without references the presentation keeps within one pass of the text encoder.
        with self.assertRaises(APIError):
            videos.prompt_tokens(tokenizer, "x" * 2049, self.ready(32768))


@unittest.skipUnless(
    all(
        os.environ.get(v)
        for v in ("SPLASH_REF2VA_RECORD_SHOP", "SPLASH_REF2VA_TOKENIZER")
    ),
    "needs a Ref2VA example's inputs and the tokenizer",
)
class RecordShopTest(unittest.TestCase):
    def test_token_ids_are_the_dump(self):
        from transformers import AutoTokenizer

        inputs = Path(os.environ["SPLASH_REF2VA_RECORD_SHOP"])
        files = [
            refs.read(inputs / f"motion-transfer-input-{name}")
            for name in ("1.mp4", "2.png", "3.png")
        ]
        self.assertEqual(
            [f.content_type for f in files], ["video/mp4", "image/png", "image/png"]
        )
        video, first, second = refs.normalize(files, 124, 832, 480)
        tokenizer = AutoTokenizer.from_pretrained(
            os.environ["SPLASH_REF2VA_TOKENIZER"], local_files_only=True
        )
        prompt = (inputs / "prompt.txt").read_text()
        ids = videos.ref2va_presentation(tokenizer, prompt, (video, first, second))
        self.assertEqual(len(ids), 6393)
        self.assertEqual(
            ids, np.load(inputs / "conditioning.npz")["token_ids"].reshape(-1).tolist()
        )


if __name__ == "__main__":
    unittest.main()

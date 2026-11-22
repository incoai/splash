"""Inkling 2's image preparation (server/images.py prepare_inkling) against
its reference processor, transformers' Inkling2ImageProcessor
(image_processing_inkling2.py), transcribed below in numpy, and its images
through the server: an image part's placeholder widened to its tokens in
Chat and Messages requests, and refused when the model serves text only."""

import base64
import hashlib
import io
import json
import math
import struct
import unittest

import numpy as np

from dev.tests import server_fixtures as fixtures
from dev.tests.inkling_fixtures import inkling_tokenizer
from server import images
from server.chat_format import INKLING

IMAGE_TOKEN = 200054
CONTENT_IMAGE = 200005
END_MESSAGE = 200010
STOP = 200006


def encoded(image, format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def pattern(width, height, mode="RGB", seed=0):
    """An image of random values of `mode` (RGB, RGBA or LA)."""
    from PIL import Image

    channels = {"RGB": 3, "RGBA": 4, "LA": 2}[mode]
    values = np.random.default_rng(seed).integers(
        0, 256, (height, width, channels), dtype=np.uint8
    )
    return Image.fromarray(values, mode)


def reference_rgb(image):
    """transformers' Inkling2ImageProcessor's RGB of a PIL image: RGBA where
    it has transparency, composited onto an 8-pixel checkerboard of 204 and
    153 as (rgb a + background (255 - a) + 127) // 255 (convert_to_rgb), else
    RGB as converted."""
    if not image.has_transparency_data:
        return np.asarray(image.convert("RGB"), dtype=np.int64)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.int64)
    height, width = rgba.shape[:2]
    rows = np.arange(height)[:, None] // 8
    columns = np.arange(width)[None, :] // 8
    background = np.where((rows + columns) % 2 == 0, 204, 153)[..., None]
    alpha = rgba[..., 3:4]
    return (rgba[..., :3] * alpha + background * (255 - alpha) + 127) // 255


def reference_pixel_values(image):
    """Its pixel_values and image_grid_thw (_preprocess): the image padded
    right and bottom to whole 40-pixel tokens with -1 on the 0-255 scale,
    rescaled and normalized by mean and deviation 0.5 (fused, (x - 127.5) /
    127.5 in float32), then one 40x40 patch per token, raster order."""
    rgb = reference_rgb(image).astype(np.float32)
    height, width = rgb.shape[:2]
    rows, columns = math.ceil(height / 40), math.ceil(width / 40)
    padded = np.full((rows * 40, columns * 40, 3), -1.0, np.float32)
    padded[:height, :width] = rgb
    normalized = (padded - np.float32(127.5)) / np.float32(127.5)
    tokens = normalized.reshape(rows, 40, columns, 40, 3).transpose(0, 2, 1, 3, 4)
    return tokens.reshape(rows * columns, 40, 40, 3), (1, rows, columns)


def splash_pixel_values(prepared):
    """Splash's prepared pixels as the encoder normalizes them (x / 127.5 - 1,
    shared/inkling_vision.metal), cut into the reference's token patches."""
    height, width = prepared.grid_height * 20, prepared.grid_width * 20
    pixels = np.frombuffer(prepared.pixels, np.uint8).reshape(height, width, 3)
    normalized = pixels.astype(np.float32) * np.float32(1 / 127.5) - np.float32(1)
    rows, columns = height // 40, width // 40
    tokens = normalized.reshape(rows, 40, columns, 40, 3).transpose(0, 2, 1, 3, 4)
    return tokens.reshape(rows * columns, 40, 40, 3), (1, rows, columns)


class InklingPreparationTest(unittest.TestCase):
    def check_against_reference(self, image, payload=None):
        prepared = images.prepare_inkling(payload or encoded(image), images.MAX_PIXELS)
        expected, grid = reference_pixel_values(image)
        actual, splash_grid = splash_pixel_values(prepared)
        self.assertEqual(splash_grid, grid)
        self.assertEqual(prepared.tokens, grid[1] * grid[2])
        self.assertEqual(
            (prepared.grid_height, prepared.grid_width), (2 * grid[1], 2 * grid[2])
        )
        self.assertEqual(prepared.patch, images.INKLING_PATCH)
        # The image's own pixels normalize as the reference's, to float32
        # rounding; the reference's padding is -1.0078, which no uint8 holds,
        # and Splash's zero padding is -1.
        pad = expected < -1.0
        np.testing.assert_allclose(actual[~pad], expected[~pad], rtol=0, atol=2e-7)
        np.testing.assert_array_equal(expected[pad], np.float32(-128.5 / 127.5))
        np.testing.assert_array_equal(actual[pad], -1.0)
        return prepared

    def test_images_are_padded_to_whole_tokens_never_resized(self):
        for width, height in (
            (1, 1),
            (39, 41),
            (40, 40),
            (50, 30),
            (120, 80),
            (457, 123),
        ):
            with self.subTest(size=(width, height)):
                prepared = self.check_against_reference(
                    pattern(width, height, seed=width)
                )
                self.assertEqual(
                    prepared.tokens, math.ceil(width / 40) * math.ceil(height / 40)
                )
                self.assertEqual(
                    len(prepared.pixels),
                    prepared.grid_height * 20 * prepared.grid_width * 20 * 3,
                )

    def test_transparency_composites_onto_the_checkerboard(self):
        from PIL import Image

        self.check_against_reference(pattern(45, 37, "RGBA", seed=3))
        self.check_against_reference(pattern(19, 26, "LA", seed=4))
        # A palette image with a transparent color.
        palette = pattern(33, 17, seed=5).convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=8
        )
        palette.info["transparency"] = 2
        self.assertTrue(palette.has_transparency_data)
        self.check_against_reference(palette, encoded(palette))
        # Fully transparent: the checkerboard itself.
        clear = Image.new("RGBA", (16, 16), (10, 20, 30, 0))
        prepared = images.prepare_inkling(encoded(clear), images.MAX_PIXELS)
        pixels = np.frombuffer(prepared.pixels, np.uint8).reshape(40, 40, 3)
        self.assertEqual(pixels[0, 0].tolist(), [204] * 3)
        self.assertEqual(pixels[0, 8].tolist(), [153] * 3)
        self.assertEqual(pixels[8, 8].tolist(), [204] * 3)

    def test_the_encoder_normalizes_as_the_reference(self):
        # x / 127.5 - 1 against the reference's fused (x - 127.5) / 127.5.
        values = np.arange(256, dtype=np.float32)
        splash = values * np.float32(1 / 127.5) - np.float32(1)
        reference = (values - np.float32(127.5)) / np.float32(127.5)
        self.assertLessEqual(float(np.abs(splash - reference).max()), 1.2e-7)

    def test_images_past_the_patch_budget_resize_inside_it(self):
        # The default cap's 16,384 patches: 4,096 tokens, as llama.cpp's.
        self.assertEqual(images.max_patches(images.MAX_PIXELS) // 4, 4096)
        for width, height, cap in (
            (5000, 4000, images.MAX_PIXELS),
            (900, 700, images.MIN_PIXELS),
        ):
            with self.subTest(size=(width, height), cap=cap):
                payload = encoded(pattern(width, height, seed=1), "JPEG")
                prepared = images.prepare_inkling(payload, cap)
                patches = prepared.grid_height * prepared.grid_width
                self.assertLessEqual(patches, images.max_patches(cap))
                self.assertGreater(patches, images.max_patches(cap) // 2)
                resized = images.inkling_size(height, width, cap)
                self.assertEqual(resized[0] % 40 + resized[1] % 40, 0)
                self.assertAlmostEqual(
                    resized[1] / resized[0], width / height, delta=0.1 * width / height
                )
        # A thin image keeps one token of height and the budget's width.
        self.assertEqual(
            images.inkling_size(40, 200_000, images.MAX_PIXELS), (40, 4096 * 40)
        )
        self.assertEqual(
            images.inkling_size(200_000, 39, images.MAX_PIXELS), (4096 * 40, 40)
        )
        # Inside it, the size stays.
        self.assertEqual(
            images.inkling_size(1234, 2345, images.MAX_PIXELS), (1234, 2345)
        )
        self.assertEqual(
            images.inkling_size(40, 163_840, images.MAX_PIXELS), (40, 163_840)
        )

    def test_the_digest_covers_the_grid_and_pixels(self):
        prepared = images.prepare_inkling(encoded(pattern(50, 30)), images.MAX_PIXELS)
        digest = hashlib.sha256(struct.pack("<II", 2, 4) + prepared.pixels).digest()
        self.assertEqual(
            (prepared.digest_lo, prepared.digest_hi), struct.unpack_from("<QQ", digest)
        )
        buffer = io.BytesIO()
        pattern(50, 30).save(buffer, format="PNG", compress_level=1)
        same = images.prepare_inkling(buffer.getvalue(), images.MAX_PIXELS)
        self.assertEqual(
            (same.digest_lo, same.digest_hi), (prepared.digest_lo, prepared.digest_hi)
        )
        other = images.prepare_inkling(
            encoded(pattern(50, 30, seed=9)), images.MAX_PIXELS
        )
        self.assertNotEqual(other.digest_lo, prepared.digest_lo)

    def test_undecodable_and_oversized_payloads_are_image_errors(self):
        with self.assertRaises(images.ImageError):
            images.prepare_inkling(b"not an image", images.MAX_PIXELS)
        with self.assertRaises(images.ImageError):
            images.inkling_size(100, 100, images.MAX_PIXELS + 1)


def data_url(image):
    return "data:image/png;base64," + base64.b64encode(encoded(image)).decode()


class InklingImageRequestTest(fixtures.HarnessTestCase):
    """Images through the server with Inkling's format, template and
    tokenizer: each image part renders as its own block, whose one
    placeholder the frontend widens to the image's tokens."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = inkling_tokenizer()

    def harness(self, *plans, **options):
        return super().harness(
            fixtures.FakeRuntime(*plans),
            tokenizer=self.tokenizer,
            chat_format=INKLING,
            image_format=images.INKLING_IMAGES,
            max_context=4096,
            **options,
        )

    def check_image_blocks(self, frame, prepared_images):
        """The frame's prompt holds each image as <|content_image|>, its
        tokens' placeholders and <|end_message|>, its span over them, and the
        pixels of its 20-pixel patches in span order."""
        prompt = list(frame.prompt_tokens)
        self.assertEqual(len(frame.image_spans), len(prepared_images))
        pixels = b""
        for span, prepared in zip(frame.image_spans, prepared_images):
            self.assertEqual(prompt[span.offset - 1], CONTENT_IMAGE)
            self.assertEqual(
                prompt[
                    span.offset : span.end
                    if hasattr(span, "end")
                    else span.offset + span.tokens
                ],
                [IMAGE_TOKEN] * prepared.tokens,
            )
            self.assertEqual(prompt[span.offset + span.tokens], END_MESSAGE)
            self.assertEqual(
                (
                    span.tokens,
                    span.grid_height,
                    span.grid_width,
                    span.patch,
                    span.digest_lo,
                    span.digest_hi,
                ),
                (
                    prepared.tokens,
                    prepared.grid_height,
                    prepared.grid_width,
                    20,
                    prepared.digest_lo,
                    prepared.digest_hi,
                ),
            )
            self.assertEqual(
                span.pixel_bytes, span.grid_height * 20 * span.grid_width * 20 * 3
            )
            pixels += prepared.pixels
        self.assertEqual(frame.image_pixels, pixels)
        self.assertEqual(
            prompt.count(IMAGE_TOKEN), sum(p.tokens for p in prepared_images)
        )

    def test_chat_image_parts_widen_to_their_tokens(self):
        first, second = pattern(90, 50, seed=1), pattern(41, 41, seed=2)
        harness = self.harness(fixtures.Plan([[STOP]]))
        message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(first)}},
                {"type": "text", "text": "the"},
                {"type": "image_url", "image_url": {"url": data_url(second)}},
            ],
        }
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", fixtures.chat_body(messages=[message])
        )
        self.assertEqual(status, 200, payload)
        prepared = [
            images.prepare_inkling(encoded(image), images.MAX_PIXELS)
            for image in (first, second)
        ]
        self.assertEqual([p.tokens for p in prepared], [6, 4])
        self.check_image_blocks(harness.backend.runtime.requests[0].frame, prepared)

    def test_messages_image_blocks_reach_the_same_spans(self):
        image = pattern(50, 30, seed=7)
        harness = self.harness(fixtures.Plan([[STOP]]), fixtures.Plan([[STOP]]))
        data = base64.b64encode(encoded(image)).decode()
        body = fixtures.anthropic_body(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": data,
                            },
                        },
                        {"type": "text", "text": "the"},
                    ],
                }
            ]
        )
        status, _, payload = harness.request("POST", "/v1/messages", body)
        self.assertEqual(status, 200, payload)
        prepared = images.prepare_inkling(encoded(image), images.MAX_PIXELS)
        messages_frame = harness.backend.runtime.requests[0].frame
        self.check_image_blocks(messages_frame, [prepared])
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            fixtures.chat_body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url(image)},
                            },
                            {"type": "text", "text": "the"},
                        ],
                    }
                ]
            ),
        )
        self.assertEqual(status, 200, payload)
        chat_frame = harness.backend.runtime.requests[1].frame
        self.check_image_blocks(chat_frame, [prepared])
        self.assertEqual(chat_frame.image_pixels, messages_frame.image_pixels)

    def test_a_model_serving_text_only_refuses_images_before_decoding(self):
        harness = self.harness(vision=False)
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                }
            ],
        }
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", fixtures.chat_body(messages=[message])
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("image", json.loads(payload)["error"]["message"])
        self.assertEqual(harness.backend.runtime.requests, [])


if __name__ == "__main__":
    unittest.main()

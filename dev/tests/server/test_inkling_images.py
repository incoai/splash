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
from dev.tests.server.test_chat_formats import inkling2_format, inkling2_tokenizer
from server import images


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
    shared/inkling_vision.metal), its padding past the image's own pixels -1,
    cut into the reference's token patches."""
    height, width = prepared.grid_height * 20, prepared.grid_width * 20
    pixels = np.frombuffer(prepared.pixels, np.uint8).reshape(height, width, 3)
    values = pixels.astype(np.float32)
    values[prepared.content_height :] = -1
    values[:, prepared.content_width :] = -1
    normalized = values * np.float32(1 / 127.5) - np.float32(1)
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
        # Every value normalizes as the reference's, to float32 rounding: the
        # image's own pixels, and its padding past them, -1 on the 0-255
        # scale, which the frame holds as zero and the encoder takes as -1
        # from the span's content size.
        self.assertEqual(
            (prepared.content_height, prepared.content_width), image.size[::-1]
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
        pixels = np.frombuffer(prepared.pixels, np.uint8).reshape(
            prepared.grid_height * 20, prepared.grid_width * 20, 3
        )
        self.assertFalse(pixels[prepared.content_height :].any())
        self.assertFalse(pixels[:, prepared.content_width :].any())
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
            (60_000, 120, images.MAX_PIXELS),
            (120, 60_000, images.MAX_PIXELS),
            (2377, 98, 100_000),
            (200_000, 40, images.MAX_PIXELS),
            (39, 200_000, images.MAX_PIXELS),
        ):
            with self.subTest(size=(width, height), cap=cap):
                tokens = images.max_patches(cap) // 4
                resized_height, resized_width = images.inkling_size(height, width, cap)
                used = math.ceil(resized_height / 40) * math.ceil(resized_width / 40)
                self.assertLessEqual(used, tokens)
                # Its sides scaled alike: the shorter one rounded to a pixel,
                # and no pixel more on the longer one fits.
                self.assertLess(resized_width, width + 1)
                self.assertLessEqual(
                    abs(resized_width * height - resized_height * width),
                    max(width, height),
                )
                longer = max(resized_height, resized_width) + 1
                shorter = max(
                    1,
                    (min(width, height) * longer + max(width, height) // 2)
                    // max(width, height),
                )
                self.assertGreater(
                    math.ceil(longer / 40) * math.ceil(shorter / 40), tokens
                )
        # The thinnest keep a pixel: a 1-pixel row of 200,000 takes the
        # budget's 4,096 tokens of width and one of height, its padding.
        self.assertEqual(
            images.inkling_size(1, 200_000, images.MAX_PIXELS), (1, 163_840)
        )
        self.assertEqual(
            images.inkling_size(120, 60_000, images.MAX_PIXELS), (109, 54_600)
        )
        payload = encoded(pattern(5000, 4000, seed=1), "JPEG")
        prepared = images.prepare_inkling(payload, images.MAX_PIXELS)
        self.assertLessEqual(prepared.tokens, 4096)
        self.assertGreater(prepared.tokens, 4000)
        # Inside it, the size stays.
        self.assertEqual(
            images.inkling_size(1234, 2345, images.MAX_PIXELS), (1234, 2345)
        )
        self.assertEqual(
            images.inkling_size(40, 163_840, images.MAX_PIXELS), (40, 163_840)
        )

    def test_a_resized_transparent_image_keeps_its_checkerboard_squares(self):
        from PIL import Image

        # Past the budget, resized first: the checkerboard is composited at
        # the size the encoder sees, its squares 8 pixels as ever.
        clear = Image.new("RGBA", (900, 700), (10, 20, 30, 0))
        prepared = images.prepare_inkling(encoded(clear), images.MIN_PIXELS)
        height, width = images.inkling_size(700, 900, images.MIN_PIXELS)
        self.assertLess(width, 900)
        pixels = np.frombuffer(prepared.pixels, np.uint8).reshape(
            prepared.grid_height * 20, prepared.grid_width * 20, 3
        )
        for row, column, grey in ((0, 0, 204), (0, 8, 153), (8, 8, 204), (7, 7, 204)):
            self.assertEqual(pixels[row, column].tolist(), [grey] * 3)
        # Banded, the composite of a large image is the one-pass formula's.
        rgba = pattern(1500, 1100, "RGBA", seed=7)
        values = np.asarray(rgba, dtype=np.int64)
        rows = np.arange(1100)[:, None] // 8
        columns = np.arange(1500)[None, :] // 8
        grey = np.where((rows + columns) % 2 == 0, 204, 153)[..., None]
        alpha = values[..., 3:]
        expected = (values[..., :3] * alpha + grey * (255 - alpha) + 127) // 255
        self.assertTrue(
            np.array_equal(np.asarray(images.composite_on_checkerboard(rgba)), expected)
        )

    def test_the_digest_covers_the_grid_and_pixels(self):
        prepared = images.prepare_inkling(encoded(pattern(50, 30)), images.MAX_PIXELS)
        # The grid, the image's own height and width, and the pixels.
        digest = hashlib.sha256(
            struct.pack("<4I", 2, 4, 30, 50) + prepared.pixels
        ).digest()
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
        # Black images 40 and 39 pixels wide hold the same padded pixels, and
        # the encoder pads the narrower one's last column: they differ.
        from PIL import Image

        wide, narrow = (
            images.prepare_inkling(
                encoded(Image.new("RGB", (width, 30))), images.MAX_PIXELS
            )
            for width in (40, 39)
        )
        self.assertEqual(wide.pixels, narrow.pixels)
        self.assertNotEqual(
            (wide.digest_lo, wide.digest_hi), (narrow.digest_lo, narrow.digest_hi)
        )

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
        cls.tokenizer = inkling2_tokenizer()
        ids = cls.tokenizer.convert_tokens_to_ids
        cls.image_token = ids("<|unused_200054|>")
        cls.content_image = ids("<|content_image|>")
        cls.end_message = ids("<|end_message|>")
        cls.stop = ids("<|content_model_end_sampling|>")

    def harness(self, *plans, **options):
        return super().harness(
            fixtures.FakeRuntime(*plans),
            tokenizer=self.tokenizer,
            chat_format=inkling2_format(self.tokenizer),
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
            self.assertEqual(prompt[span.offset - 1], self.content_image)
            self.assertEqual(
                prompt[
                    span.offset : span.end
                    if hasattr(span, "end")
                    else span.offset + span.tokens
                ],
                [self.image_token] * prepared.tokens,
            )
            self.assertEqual(prompt[span.offset + span.tokens], self.end_message)
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
            prompt.count(self.image_token), sum(p.tokens for p in prepared_images)
        )

    def test_chat_image_parts_widen_to_their_tokens(self):
        first, second = pattern(90, 50, seed=1), pattern(41, 41, seed=2)
        harness = self.harness(fixtures.Plan([[self.stop]]))
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
        harness = self.harness(
            fixtures.Plan([[self.stop]]), fixtures.Plan([[self.stop]])
        )
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

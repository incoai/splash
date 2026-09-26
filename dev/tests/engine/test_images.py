import base64
import gc
import hashlib
import io
import random
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from server import images


def png_bytes(width, height, color=(200, 30, 30)):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


class SmartResizeTest(unittest.TestCase):
    def test_sides_stay_patch_and_merge_aligned_inside_the_budget(self):
        for height, width in ((100, 300), (10_000, 300), (17, 3_000), (3000, 4000)):
            resized_height, resized_width = images.smart_resize(
                height, width, images.MAX_PIXELS
            )
            self.assertEqual(resized_height % 32, 0)
            self.assertEqual(resized_width % 32, 0)
            self.assertGreaterEqual(resized_height * resized_width, images.MIN_PIXELS)
            self.assertLessEqual(resized_height * resized_width, images.MAX_PIXELS)

    def test_small_images_are_upscaled_to_the_minimum_budget(self):
        self.assertEqual(images.smart_resize(64, 64, images.MAX_PIXELS), (256, 256))

    def test_serving_cap_bounds_large_images(self):
        height, width = images.smart_resize(2880, 1800, 1_000_000)
        self.assertLessEqual(height * width, 1_000_000)
        self.assertAlmostEqual(height / width, 2880 / 1800, delta=0.1)

    def test_extreme_aspect_ratio_is_rejected(self):
        with self.assertRaises(images.ImageError):
            images.smart_resize(10, 3000, images.MAX_PIXELS)

    def test_alignment_never_exceeds_a_small_serving_cap(self):
        rng = random.Random(42)
        shapes = [(1, 100), (100, 1), (1, 200), (200, 1)]
        shapes += [(rng.randint(1, 10000), rng.randint(1, 10000)) for _ in range(300)]
        for shape in shapes:
            if max(shape) / min(shape) > 200:
                continue
            for cap in (images.MIN_PIXELS, 70000, 100000, images.MAX_PIXELS):
                with self.subTest(shape=shape, cap=cap):
                    height, width = images.smart_resize(*shape, cap)
                    self.assertGreaterEqual(min(height, width), images.FACTOR)
                    self.assertEqual(height % images.FACTOR, 0)
                    self.assertEqual(width % images.FACTOR, 0)
                    self.assertLessEqual(height * width, cap)


class PrepareTest(unittest.TestCase):
    def test_prepare_returns_grid_pixels_and_content_digest(self):
        prepared = images.prepare(png_bytes(64, 64), images.MAX_PIXELS)
        # 64x64 upscales to the 256x256 minimum: a 16x16 patch grid.
        self.assertEqual((prepared.grid_height, prepared.grid_width), (16, 16))
        self.assertEqual(prepared.tokens, 64)
        self.assertEqual(len(prepared.pixels), 256 * 256 * 3)
        self.assertEqual(prepared.pixels[:3], bytes((200, 30, 30)))
        digest = hashlib.sha256(struct.pack("<II", 16, 16) + prepared.pixels).digest()
        self.assertEqual(
            (prepared.digest_lo, prepared.digest_hi),
            struct.unpack_from("<QQ", digest),
        )

    def test_digest_depends_on_content_not_encoding(self):
        red = images.prepare(png_bytes(64, 64), images.MAX_PIXELS)
        red_again = images.prepare(png_bytes(64, 64), images.MAX_PIXELS)
        blue = images.prepare(png_bytes(64, 64, (30, 30, 200)), images.MAX_PIXELS)
        self.assertEqual(
            (red.digest_lo, red.digest_hi), (red_again.digest_lo, red_again.digest_hi)
        )
        self.assertNotEqual(
            (red.digest_lo, red.digest_hi), (blue.digest_lo, blue.digest_hi)
        )

    def test_undecodable_payload_is_an_image_error(self):
        with self.assertRaises(images.ImageError):
            images.prepare(b"not an image", images.MAX_PIXELS)

    def test_source_limit_and_aspect_ratio_are_checked_before_decode(self):
        from PIL import Image

        normal = png_bytes(64, 64)
        wide = png_bytes(201, 1)
        with mock.patch.object(images, "MAX_SOURCE_PIXELS", 64 * 64):
            self.assertEqual(images.prepare(normal).tokens, 64)
        with mock.patch.object(images, "MAX_SOURCE_PIXELS", 64 * 64 - 1):
            with mock.patch.object(Image.Image, "convert") as convert:
                with self.assertRaisesRegex(images.ImageError, "source image exceeds"):
                    images.prepare(normal)
                convert.assert_not_called()
        with mock.patch.object(Image.Image, "convert") as convert:
            with self.assertRaisesRegex(images.ImageError, "aspect ratio"):
                images.prepare(wide)
            convert.assert_not_called()

    def test_exif_orientation_is_applied_before_resizing(self):
        from PIL import Image

        def prepared(image, exif=None):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", **({"exif": exif} if exif else {}))
            image = images.prepare(buffer.getvalue(), images.MAX_PIXELS)
            return image.grid_height, image.grid_width, image.digest_lo

        stored = Image.new("RGB", (80, 40), (30, 30, 200))
        stored.paste((200, 30, 30), (0, 0, 80, 1))
        # Orientation 6 displays the stored image turned a quarter clockwise.
        tag = Image.Exif()
        tag[0x0112] = 6
        upright = stored.transpose(Image.Transpose.ROTATE_270)
        self.assertEqual(prepared(stored, tag), prepared(upright))
        # A malformed tag leaves the image as stored instead of failing it.
        malformed = b"Exif\x00\x00not a tiff header"
        self.assertEqual(prepared(stored, malformed), prepared(stored))

    def test_transparent_pixels_are_composited_onto_white(self):
        from PIL import Image

        # Opaque black on the top half, transparent pixels storing black below.
        rgba = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        rgba.paste((0, 0, 0, 255), (0, 0, 256, 128))
        palette = Image.new("P", (256, 256), 0)
        palette.putpalette([0, 0, 0, 0, 0, 0])
        palette.paste(1, (0, 0, 256, 128))
        expected = bytes(256 * 128 * 3) + b"\xff" * (256 * 128 * 3)
        for image, encoding, params in (
            (rgba, "PNG", {}),
            (rgba.convert("LA"), "PNG", {}),
            (palette, "PNG", {"transparency": 0}),
            (palette, "GIF", {"transparency": 0}),
        ):
            with self.subTest(mode=image.mode, encoding=encoding):
                buffer = io.BytesIO()
                image.save(buffer, format=encoding, **params)
                prepared = images.prepare(buffer.getvalue(), images.MAX_PIXELS)
                self.assertEqual(prepared.pixels, expected)

    def test_small_image_preparation_obeys_cap_after_upscale(self):
        prepared = images.prepare(png_bytes(1, 100), images.MIN_PIXELS)
        self.assertLessEqual(len(prepared.pixels), 3 * images.MIN_PIXELS)


class DataUrlTest(unittest.TestCase):
    def test_base64_data_urls_decode_and_others_are_rejected(self):
        payload = png_bytes(64, 64)
        url = "data:image/png;base64," + base64.b64encode(payload).decode()
        self.assertEqual(images.decode_data_url(url), payload)
        for invalid in (
            "https://example.com/x.png",
            "data:image/png,rawbytes",
            "data:image/png;base64,not*base64",
            None,
        ):
            with self.subTest(url=invalid):
                with self.assertRaises(images.ImageError):
                    images.decode_data_url(invalid)


class ImageCacheTest(unittest.TestCase):
    def test_request_budget_counts_repeated_images_and_all_live_batches(self):
        cache = images.ImageCache(budget_bytes=0, request_budget_bytes=12)
        image = images.PreparedImage(2, 2, b"abcd", 0, 0)
        first, second = cache.request_batch(), cache.request_batch()
        first.append(image)
        first.append(image)
        second.append(image)
        self.assertEqual(cache.stats()["request_bytes"], 12)
        with self.assertRaises(images.ImageCapacityError):
            second.append(image)
        self.assertEqual(cache.stats()["request_bytes"], 12)
        holder = first
        del first
        self.assertEqual(cache.stats()["request_bytes"], 12)
        del holder
        self.assertEqual(cache.stats()["request_bytes"], 4)
        second.append(image)
        del second
        self.assertEqual(cache.stats()["request_bytes"], 0)

    def test_request_budget_is_returned_after_exception_and_cyclic_owner(self):
        cache = images.ImageCache(request_budget_bytes=4)

        def prepare_then_fail():
            batch = cache.request_batch()
            batch.append(images.PreparedImage(2, 2, b"abcd", 0, 0))
            batch.owner = batch  # Model a cancelled callback ownership cycle.
            raise images.ImageError("injected render failure")

        with self.assertRaises(images.ImageError):
            prepare_then_fail()
        with cache._lock:
            gc.collect()
        self.assertEqual(cache.stats()["request_bytes"], 0)

    def test_concurrent_request_batches_do_not_oversell_image_budget(self):
        cache = images.ImageCache(request_budget_bytes=12)
        batches = [cache.request_batch() for _ in range(16)]

        def append(batch):
            try:
                batch.append(images.PreparedImage(2, 2, b"abcd", 0, 0))
                return True
            except images.ImageCapacityError:
                return False

        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(sum(executor.map(append, batches)), 3)
        self.assertEqual(cache.stats()["request_bytes"], 12)
        batches.clear()
        self.assertEqual(cache.stats()["request_bytes"], 0)

    def test_cache_reuses_prepared_images_and_evicts_by_bytes(self):
        cache = images.ImageCache(budget_bytes=2 * 256 * 256 * 3)
        first = cache.prepare(png_bytes(64, 64), images.MAX_PIXELS)
        self.assertIs(cache.prepare(png_bytes(64, 64), images.MAX_PIXELS), first)
        self.assertEqual(cache.stats()["entries"], 1)
        cache.prepare(png_bytes(64, 64, (0, 200, 0)), images.MAX_PIXELS)
        cache.prepare(png_bytes(64, 64, (0, 0, 200)), images.MAX_PIXELS)
        self.assertEqual(cache.stats()["entries"], 2)
        self.assertLessEqual(cache.stats()["bytes"], cache.budget_bytes)
        # A different serving cap is a different preparation.
        cache.prepare(png_bytes(64, 64), 65_536)
        self.assertEqual(cache.stats()["entries"], 2)


if __name__ == "__main__":
    unittest.main()

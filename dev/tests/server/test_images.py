import base64
import gc
import hashlib
import io
import json
import random
import struct
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

from tokenizers import Tokenizer, models

from dev.tests.server_fixtures import (
    FOREVER,
    FakeRuntime,
    HarnessTestCase,
    ImagePadTokenizer,
    Plan,
    anthropic_body,
    chat_body,
    document_block,
    no_signed_thinking,
    pdf_bytes,
    png_data_url,
    render_pdf,
    response_events,
    responses_body,
)
from server import api_shapes, documents, images
from server import backend as backend_api
from server import frontend as request_frontend
from server import protocol as native_wire
from server import server as api


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
        self.enterContext(mock.patch.object(images.ImageCache, "BUDGET_BYTES", 0))
        self.enterContext(
            mock.patch.object(images.ImageCache, "REQUEST_BUDGET_BYTES", 12)
        )
        cache = images.ImageCache()
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
        self.enterContext(
            mock.patch.object(images.ImageCache, "REQUEST_BUDGET_BYTES", 4)
        )
        cache = images.ImageCache()

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
        self.enterContext(
            mock.patch.object(images.ImageCache, "REQUEST_BUDGET_BYTES", 12)
        )
        cache = images.ImageCache()
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
        self.enterContext(
            mock.patch.object(images.ImageCache, "BUDGET_BYTES", 2 * 256 * 256 * 3)
        )
        cache = images.ImageCache()
        first = cache.prepare(png_bytes(64, 64), images.MAX_PIXELS)
        self.assertIs(cache.prepare(png_bytes(64, 64), images.MAX_PIXELS), first)
        self.assertEqual(cache.stats()["entries"], 1)
        cache.prepare(png_bytes(64, 64, (0, 200, 0)), images.MAX_PIXELS)
        cache.prepare(png_bytes(64, 64, (0, 0, 200)), images.MAX_PIXELS)
        self.assertEqual(cache.stats()["entries"], 2)
        self.assertLessEqual(cache.stats()["bytes"], cache.BUDGET_BYTES)
        # A different serving cap is a different preparation.
        cache.prepare(png_bytes(64, 64), 65_536)
        self.assertEqual(cache.stats()["entries"], 2)


def image_message(url=None):
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {
                "type": "image_url",
                "image_url": {"url": url or png_data_url()},
            },
        ],
    }


class ImageRequestTest(HarnessTestCase):
    def test_image_parts_expand_placeholders_and_carry_spans_and_pixels(self):
        runtime = FakeRuntime(Plan([[4]]))
        harness = self.harness(runtime, tokenizer=ImagePadTokenizer())
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(messages=[image_message()])
        )
        self.assertEqual(status, 200)
        request = runtime.requests[0].frame
        # 64x64 upscales to the 256x256 minimum: a 16x16 patch grid, 64 tokens.
        self.assertEqual(request.prompt_tokens, (101, *([50] * 64), 102))
        (span,) = request.image_spans
        self.assertEqual(
            (span.offset, span.tokens, span.grid_height, span.grid_width),
            (1, 64, 16, 16),
        )
        self.assertEqual(len(request.image_pixels), 256 * 256 * 3)
        self.assertEqual(request.image_pixels[:3], bytes((200, 30, 30)))
        digest = hashlib.sha256(
            struct.pack("<II", 16, 16) + request.image_pixels
        ).digest()
        self.assertEqual(
            (span.digest_lo, span.digest_hi), struct.unpack_from("<QQ", digest)
        )
        rendered, _ = harness.tokenizer.templates[-1]
        self.assertEqual(
            [part["type"] for part in rendered[0]["content"]], ["text", "image_url"]
        )
        # Repeats reuse the prepared image instead of decoding again.
        harness.request(
            "POST", "/v1/chat/completions", chat_body(messages=[image_message()])
        )
        self.assertEqual(harness.app.images.stats()["entries"], 1)
        self.assertEqual(runtime.requests[1].frame.image_spans, request.image_spans)

    def test_image_positions_ignore_quoted_vision_tokens(self):
        from tokenizers import pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            additional_special_tokens=[
                api_shapes.IMAGE_PAD_TOKEN,
                "<|vision_start|>",
                "<|vision_end|>",
            ],
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message.role }}: "
            "{% if message.content is string %}{{ message.content }}"
            "{% else %}{% for part in message.content %}"
            "{% if part.type == 'image_url' %}"
            "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
            "{% else %}{{ part.text }}{% endif %}{% endfor %}{% endif %}"
            "{{ '\\n' }}{% endfor %}"
        )
        app = object.__new__(request_frontend.Frontend)
        app.tokenizer = tokenizer
        from server.latency import LatencyMetrics

        app.latencies = LatencyMetrics()
        app.max_context = 1024
        quoted = "中文 📷 <|vision_start|><|image_pad|><|vision_end|>"
        messages = [
            {"role": "system", "content": "Document: " + api_shapes.IMAGE_PAD_TOKEN},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "unused"}},
                    {"type": "text", "text": quoted},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "image_url", "image_url": {"url": "unused"}},
                    {"type": "text", "text": quoted},
                ],
            },
        ]
        template = {
            "tokenize": True,
            "return_dict": False,
            "chat_template": tokenizer.chat_template,
        }
        baseline = tokenizer.apply_chat_template(messages, **template)
        pad_id = tokenizer.convert_tokens_to_ids(api_shapes.IMAGE_PAD_TOKEN)
        all_pads = [i for i, token in enumerate(baseline) if token == pad_id]
        self.assertEqual(len(all_pads), 5)
        tokens, positions, rendered = app._render_image_tokens(messages, template)
        self.assertEqual(tokens, baseline)
        self.assertEqual(
            rendered, tokenizer.apply_chat_template(messages, tokenize=False)
        )
        self.assertEqual(positions, [all_pads[1], all_pads[3]])
        # Render markers must not change prompt/cache identity on a repeat.
        self.assertEqual(
            app._render_image_tokens(messages, template), (tokens, positions, rendered)
        )
        prepared = [
            SimpleNamespace(
                tokens=4,
                grid_height=4,
                grid_width=4,
                digest_lo=i,
                digest_hi=0,
                pixels=bytes([i]),
            )
            for i in (1, 2)
        ]
        expanded, spans, pixels = app._expand_image_pads(tokens, prepared, positions)
        expected = list(baseline)
        for position in reversed(positions):
            expected[position : position + 1] = [pad_id] * 4
        self.assertEqual(expanded, expected)
        self.assertEqual(
            [span.offset for span in spans], [positions[0], positions[1] + 3]
        )
        self.assertEqual(pixels, b"\x01\x02")
        with self.assertRaisesRegex(api.APIError, "image count"):
            app._expand_image_pads(tokens, prepared[:1], positions)

    def test_image_render_marker_is_stable_across_requests(self):
        app = self.harness(FakeRuntime(), tokenizer=ImagePadTokenizer()).app
        template = {
            "tokenize": False,
            "return_dict": False,
            "chat_template": app.tokenizer.chat_template,
        }
        app._render_image_tokens([image_message()], template)
        app._render_image_tokens([image_message()], template)
        first_source = app.tokenizer.templates[-2][1]["chat_template"]
        second_source = app.tokenizer.templates[-1][1]["chat_template"]
        self.assertEqual(first_source, second_source)

    def test_image_size_is_checked_before_pixel_concatenation(self):
        app = self.harness(FakeRuntime(), tokenizer=ImagePadTokenizer()).app

        class UnmaterializedPixels:
            def __len__(self):
                return native_wire.MAX_FRAME_PAYLOAD_BYTES

        # No huge buffer: trying to concatenate this sentinel raises TypeError.
        # The size guard must reject using lengths alone, before any copy.
        image = SimpleNamespace(
            tokens=4,
            pixels=UnmaterializedPixels(),
            grid_height=4,
            grid_width=4,
            digest_lo=0,
            digest_hi=0,
        )
        pad = app.tokenizer.convert_tokens_to_ids(api_shapes.IMAGE_PAD_TOKEN)
        with self.assertRaisesRegex(api.APIError, "request size limit"):
            app._expand_image_pads([pad], [image], [0])

    def test_language_only_rejects_media_before_decoding_or_rendering(self):
        runtime = FakeRuntime()
        harness = self.harness(
            runtime, tokenizer=ImagePadTokenizer(), max_context=65536, vision=False
        )
        image = png_data_url()
        pdf = base64.b64encode(pdf_bytes()).decode()
        pdf_url = "data:application/pdf;base64," + pdf
        chat = {
            "image_url": {"type": "image_url", "image_url": {"url": image}},
            "file": {
                "type": "file",
                "file": {"filename": "a.pdf", "file_data": pdf_url},
            },
        }
        responses = {
            "input_image": {"type": "input_image", "image_url": image},
            "input_file": {
                "type": "input_file",
                "filename": "a.pdf",
                "file_data": pdf_url,
            },
        }
        anthropic = {
            "image": {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image.partition(",")[2],
                },
            },
            "document": {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf,
                },
            },
        }
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "look", "arguments": "{}"},
        }
        modality = {
            "image_url": "image",
            "input_image": "image",
            "image": "image",
            "file": "PDF",
            "input_file": "PDF",
            "document": "PDF",
        }
        # Every API, with each part in a user turn and in a tool result.
        cases = []
        for kind, part in chat.items():
            tool_result = [
                {"role": "user", "content": "look"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_1", "content": [part]},
            ]
            for path in ("/v1/chat/completions", "/apply-template"):
                for messages in ([{"role": "user", "content": [part]}], tool_result):
                    cases.append((path, kind, chat_body(messages=messages)))
        for kind, part in responses.items():
            tool_result = [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "look",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": [part]},
            ]
            for items in ([{"role": "user", "content": [part]}], tool_result):
                cases.append(("/v1/responses", kind, responses_body(input=items)))
            # Stored history is normalized with the new input, like any item.
            history_id = f"resp_{kind}"
            harness.app.response_store.put(
                {"id": history_id}, [{"role": "user", "content": [part]}]
            )
            cases.append(
                (
                    "/v1/responses",
                    kind,
                    responses_body(input="again", previous_response_id=history_id),
                )
            )
        for kind, block in anthropic.items():
            tool_result = [
                {"role": "user", "content": "look"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "look",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [block],
                        }
                    ],
                },
            ]
            for path in ("/v1/messages", "/v1/messages/count_tokens"):
                for messages in ([{"role": "user", "content": [block]}], tool_result):
                    cases.append((path, kind, anthropic_body(messages=messages)))
        with (
            mock.patch.object(
                images,
                "decode_data_url",
                wraps=images.decode_data_url,
            ) as decode,
            mock.patch.object(
                documents, "pdf_content", wraps=documents.pdf_content
            ) as render,
        ):
            for path, kind, body in cases:
                with self.subTest(path=path, kind=kind):
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 400, payload)
                    self.assertEqual(
                        json.loads(payload)["error"]["message"],
                        f"{modality[kind]} input is not supported: "
                        "this model is serving without vision "
                        "(started with --language-only)",
                    )
            decode.assert_not_called()
            render.assert_not_called()
        self.assertEqual(runtime.requests, [])
        self.assertEqual(harness.app.images.stats()["request_bytes"], 0)
        self._wait_for_http_active(harness.server.request_bodies, 0)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body()
        )
        self.assertEqual(status, 200, payload)

    def test_conversions_leave_media_to_message_normalization(self):
        image = png_data_url()
        document = document_block(title="Report", context="Fixture")
        anthropic = api_shapes.anthropic_to_chat_prompt
        responses = api_shapes.responses_to_chat_body
        with (
            mock.patch.object(images, "decode_data_url", side_effect=AssertionError),
            mock.patch.object(documents, "_render", side_effect=AssertionError),
        ):
            converted = {
                "responses": responses(
                    {},
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_image", "image_url": image},
                                {"type": "input_file", "file_data": image},
                            ],
                        }
                    ],
                )[0]["messages"],
                "anthropic": anthropic(
                    {
                        "model": "m",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": image.partition(",")[2],
                                        },
                                    },
                                    document,
                                ],
                            }
                        ],
                    },
                    thinking_resolver=no_signed_thinking,
                )["messages"],
            }
        for shape, messages in converted.items():
            with self.subTest(shape=shape):
                kinds = [part["type"] for part in messages[0]["content"]]
                self.assertEqual(kinds[0], "image_url")
                self.assertEqual(kinds[-1], "file")
                with self.assertRaisesRegex(api.APIError, "^image input"):
                    api_shapes.normalize_messages(
                        messages, vision=False, deadline=FOREVER
                    )
                with self.assertRaisesRegex(api.APIError, "^PDF input"):
                    api_shapes.normalize_messages(
                        [{"role": "user", "content": messages[0]["content"][1:]}],
                        vision=False,
                        deadline=FOREVER,
                    )
        # Normalization renders the document's PDF after its title and context.
        self.assertEqual(
            api_shapes.normalize_messages(
                converted["anthropic"], vision=True, deadline=FOREVER
            )[0]["content"][1:],
            [
                {"type": "text", "text": "Report\n"},
                {"type": "text", "text": "Fixture\n"},
                *render_pdf(),
            ],
        )

    def test_image_count_is_checked_before_decoding(self):
        app = self.harness(FakeRuntime(), tokenizer=ImagePadTokenizer()).app
        part = image_message()["content"][1]
        limit = native_wire.MAX_IMAGE_SPANS
        messages = [
            {"role": "user", "content": [part] * limit},
            {"role": "tool", "content": [part]},
        ]
        with mock.patch.object(images, "decode_data_url") as decode:
            with self.assertRaisesRegex(api.APIError, f"at most {limit} images"):
                app._prepare_images(messages, FOREVER, check_context=False)
            decode.assert_not_called()
        self.assertEqual(app.images.stats()["request_bytes"], 0)
        prepared = app._prepare_images(messages[:1], FOREVER, check_context=False)
        self.assertEqual(len(prepared), limit)
        del prepared
        self.assertEqual(app.images.stats()["request_bytes"], 0)

    def test_image_context_is_checked_before_expanding_placeholders(self):
        app = self.harness(FakeRuntime(), tokenizer=ImagePadTokenizer()).app
        image = SimpleNamespace(
            tokens=app.max_context,
            pixels=b"",
            grid_height=16,
            grid_width=32,
            digest_lo=0,
            digest_hi=0,
        )
        pad = app.tokenizer.convert_tokens_to_ids(api_shapes.IMAGE_PAD_TOKEN)
        with self.assertRaisesRegex(api.APIError, "context window"):
            app._expand_image_pads([pad], [image], [0])

    def test_image_preparation_stops_at_aggregate_budget(self):
        app = self.harness(FakeRuntime()).app
        image = SimpleNamespace(tokens=1, pixels=b"x" * 512)
        messages = [
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    }
                ]
                * 3
            }
        ]
        with (
            mock.patch.object(native_wire, "MAX_FRAME_PAYLOAD_BYTES", 1024),
            mock.patch.object(app.images, "prepare", return_value=image) as prepare,
            self.assertRaisesRegex(api.APIError, "request size limit"),
        ):
            app._prepare_images(messages, FOREVER)
        self.assertEqual(prepare.call_count, 2)

    def test_image_preparation_stops_once_the_request_expires(self):
        app = self.harness(FakeRuntime()).app
        image = SimpleNamespace(tokens=1, pixels=b"x")
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        clock = [100.0]

        def prepare(*_args):
            # Each image takes a second.
            clock[0] += 1.0
            return image

        with (
            mock.patch.object(
                backend_api, "time", SimpleNamespace(monotonic=lambda: clock[0])
            ),
            mock.patch.object(app.images, "prepare", side_effect=prepare) as prepared,
            self.assertRaises(api.APIError) as caught,
        ):
            app._prepare_images([{"content": [part] * 3}], 100.5)
        self.assertEqual(caught.exception.status, 504)
        self.assertEqual(prepared.call_count, 1)
        self.assertEqual(app.images.stats()["request_bytes"], 0)

    def test_image_request_budget_survives_native_owner_and_recovers(self):
        app = self.harness(
            FakeRuntime(), tokenizer=ImagePadTokenizer(), max_context=1024
        ).app
        self.enterContext(
            mock.patch.object(images.ImageCache, "REQUEST_BUDGET_BYTES", 256 * 256 * 3)
        )
        app.images = images.ImageCache()
        job = app.prepare(chat_body(messages=[image_message()]), deadline=FOREVER)
        native_request = app.backend._generation_request(job)
        self.assertIs(native_request.image_owner, job.image_owner)
        del job
        with self.assertRaisesRegex(api.APIError, "image memory budget") as failure:
            app.prepare(chat_body(messages=[image_message()]), deadline=FOREVER)
        self.assertEqual(failure.exception.status, 503)
        del native_request
        self.assertEqual(app.images.stats()["request_bytes"], 0)
        job = app.prepare(chat_body(messages=[image_message()]), deadline=FOREVER)
        self.assertGreater(app.images.stats()["request_bytes"], 0)
        del job
        self.assertEqual(app.images.stats()["request_bytes"], 0)

    def test_anthropic_and_responses_images_reach_the_same_pipeline(self):
        runtime = FakeRuntime(Plan([[4]]), Plan([[4]]))
        harness = self.harness(runtime, tokenizer=ImagePadTokenizer())
        url = png_data_url((30, 30, 200))
        media_type, _, data = url.partition(";base64,")
        status, _, _ = harness.request(
            "POST",
            "/v1/messages",
            anthropic_body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "before"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type.removeprefix("data:"),
                                    "data": data,
                                },
                            },
                            {"type": "text", "text": "after"},
                        ],
                    }
                ]
            ),
        )
        self.assertEqual(status, 200)
        rendered, _ = harness.tokenizer.templates[-1]
        self.assertEqual(
            [part["type"] for part in rendered[0]["content"]],
            ["text", "image_url", "text"],
        )
        self.assertEqual(len(runtime.requests[0].frame.image_spans), 1)

        status, _, _ = harness.request(
            "POST",
            "/v1/responses",
            responses_body(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "look"},
                            {"type": "input_image", "image_url": url},
                        ],
                    }
                ]
            ),
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(runtime.requests[1].frame.image_spans), 1)
        self.assertEqual(
            runtime.requests[1].frame.image_spans, runtime.requests[0].frame.image_spans
        )

    def test_tool_results_carry_images_like_user_content(self):
        runtime = FakeRuntime(Plan([[4]]), Plan([[4]]))
        harness = self.harness(runtime, tokenizer=ImagePadTokenizer())
        url = png_data_url((30, 200, 30))
        media_type, _, data = url.partition(";base64,")
        status, _, _ = harness.request(
            "POST",
            "/v1/messages",
            anthropic_body(
                messages=[
                    {"role": "user", "content": "screenshot it"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "screenshot",
                                "input": {},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [
                                    {"type": "text", "text": "captured"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type.removeprefix(
                                                "data:"
                                            ),
                                            "data": data,
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                ]
            ),
        )
        self.assertEqual(status, 200)
        rendered, _ = harness.tokenizer.templates[-1]
        self.assertEqual(rendered[-1]["role"], "tool")
        self.assertEqual(
            [part["type"] for part in rendered[-1]["content"]], ["text", "image_url"]
        )
        request = runtime.requests[0].frame
        self.assertEqual(request.prompt_tokens, (101, *([50] * 64), 102))
        (span,) = request.image_spans
        self.assertEqual((span.offset, span.tokens), (1, 64))

        status, _, _ = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(
                messages=[
                    {"role": "user", "content": "screenshot it"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "screenshot", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": [
                            {"type": "text", "text": "captured"},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    },
                ]
            ),
        )
        self.assertEqual(status, 200)
        rendered, _ = harness.tokenizer.templates[-1]
        self.assertEqual(rendered[-1]["role"], "tool")
        self.assertEqual(
            [part["type"] for part in rendered[-1]["content"]], ["text", "image_url"]
        )
        self.assertEqual(runtime.requests[1].frame.image_spans, request.image_spans)

        # Responses tool outputs use the same image pixels and span geometry,
        # including when the tool result is replayed from stored history.
        for stream in (False, True):
            status, _, payload = harness.request(
                "POST",
                "/v1/responses",
                responses_body(
                    stream=stream,
                    reasoning={"effort": "none"},
                    input=[
                        {"role": "user", "content": "screenshot it"},
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "screenshot",
                            "arguments": "{}",
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": [
                                {"type": "input_text", "text": "captured"},
                                {"type": "input_image", "image_url": url},
                            ],
                        },
                    ],
                ),
            )
            self.assertEqual(status, 200, payload)
            response = (
                response_events(payload)[-1]["response"]
                if stream
                else json.loads(payload)
            )
            rendered, _ = harness.tokenizer.templates[-1]
            self.assertEqual(rendered[-1]["role"], "tool")
            self.assertEqual(rendered[-1]["tool_call_id"], "call_1")
            self.assertEqual(
                [part["type"] for part in rendered[-1]["content"]],
                ["text", "image_url"],
            )
            self.assertEqual(
                runtime.requests[-1].frame.image_spans, request.image_spans
            )
            self.assertEqual(
                runtime.requests[-1].frame.image_pixels, request.image_pixels
            )
            status, _, payload = harness.request(
                "POST",
                "/v1/responses",
                responses_body(previous_response_id=response["id"], input="look again"),
            )
            self.assertEqual(status, 200, payload)
            self.assertEqual(
                runtime.requests[-1].frame.image_spans, request.image_spans
            )

    def test_responses_tool_images_reject_invalid_content_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=ImagePadTokenizer())
        for content in (
            {"type": "input_image", "image_url": "https://example.com/image.png"},
            {"type": "input_image", "image_url": "data:image/png;base64,@@@"},
            {"type": "input_file", "file_id": "not-supported"},
        ):
            with self.subTest(content=content):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/responses",
                    responses_body(
                        input=[
                            {
                                "type": "function_call_output",
                                "call_id": "call_1",
                                "output": [content],
                            }
                        ]
                    ),
                )
                self.assertEqual(status, 400, payload)
        self.assertEqual(runtime.requests, [])

    def test_image_inputs_are_validated_before_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, tokenizer=ImagePadTokenizer())
        cases = (
            (
                [image_message("https://example.com/x.png")],
                "only data: image URLs",
            ),
            ([image_message("data:image/png;base64,@@@")], "not valid base64"),
            (
                [{"role": "user", "content": [{"type": "video", "video": "x"}]}],
                "video content is not supported",
            ),
            (
                [{"role": "user", "content": [{"type": "image_url", "image_url": 5}]}],
                "invalid image content part",
            ),
            (
                [
                    {
                        "role": "system",
                        "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                    }
                ],
                "only text message content",
            ),
        )
        for messages, message in cases:
            with self.subTest(message=message):
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(messages=messages)
                )
                self.assertEqual(status, 400)
                self.assertIn(message, json.loads(payload)["error"]["message"])
        self.assertEqual(runtime.requests, [])

        # A template that drops the placeholder cannot carry the image.
        plain = self.harness(runtime)
        status, _, payload = plain.request(
            "POST", "/v1/chat/completions", chat_body(messages=[image_message()])
        )
        self.assertEqual(status, 400)
        self.assertIn(
            "does not define the image pad", json.loads(payload)["error"]["message"]
        )


if __name__ == "__main__":
    unittest.main()

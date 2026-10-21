import io
import math
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from PIL import Image

from dev.tests.server_fixtures import (
    document_block,
    empty_page_cache,
    full_render_limits,
    pdf_bytes,
    render_pdf,
)
from server import documents, images
from server.errors import APIError
from server.lru import LRUCache


class DocumentTests(unittest.TestCase):
    def test_owner_password_does_not_prevent_opening_but_user_password_does(self):
        fixtures = Path(__file__).parents[1] / "fixtures" / "documents"
        owner = (fixtures / "owner-password.pdf").read_bytes()
        locked = (fixtures / "open-password.pdf").read_bytes()
        # Both fixtures contain the same page, encrypted with AES-128. The
        # first has an empty opening password; the second requires one.
        self.assertIn("ALPHA 42", render_pdf(owner)[0]["text"])
        with self.assertRaisesRegex(APIError, "opening password"):
            render_pdf(locked)

    def setUp(self):
        self.enterContext(empty_page_cache())

    def test_document_block_puts_title_and_context_before_its_pdf(self):
        block = document_block(title="Report", context="Local fixture")
        self.assertEqual(
            documents.document_parts(block),
            [
                {"type": "text", "text": "Report\n"},
                {"type": "text", "text": "Local fixture\n"},
                {
                    "type": "file",
                    "file": {
                        "file_data": documents.PDF_DATA_URL_PREFIX
                        + block["source"]["data"]
                    },
                },
            ],
        )

    def test_pdf_keeps_page_text_and_images(self):
        parts = render_pdf(pdf_bytes(pages=2))
        self.assertIn("ALPHA 42 page 1", parts[0]["text"])
        self.assertIn("ALPHA 42 page 2", parts[2]["text"])
        for part in (parts[1], parts[3]):
            payload = images.decode_data_url(part["image_url"]["url"])
            with Image.open(io.BytesIO(payload)) as page:
                self.assertLessEqual(
                    page.width * page.height, documents.MAX_PAGE_PIXELS
                )
                self.assertEqual(
                    page.getpixel((page.width // 4, page.height * 3 // 4))[:3],
                    (255, 0, 0),
                )
                self.assertEqual(
                    page.getpixel((page.width * 3 // 4, page.height * 3 // 4))[:3],
                    (0, 0, 255),
                )
            self.assertGreater(images.prepare(payload).tokens, 0)

    def test_large_page_is_bounded_before_bitmap_allocation(self):
        import pypdfium2 as pdfium

        sizes = []
        create_bitmap = pdfium.PdfBitmap.new_native

        def bounded_bitmap(width, height, **kwargs):
            sizes.append((width, height))
            self.assertLessEqual(width * height, documents.MAX_PAGE_PIXELS)
            return create_bitmap(width, height, **kwargs)

        original = pdfium.PdfPage.render

        def render(page, **kwargs):
            return original(page, bitmap_maker=bounded_bitmap, **kwargs)

        with mock.patch.object(pdfium.PdfPage, "render", render):
            documents.render_pages(
                pdf_bytes(width=14400, height=14400),
                documents.DocumentBudget(deadline=math.inf),
                full_render_limits(),
            )
        self.assertEqual(len(sizes), 1)

    def test_invalid_document_blocks_are_rejected(self):
        numeric_data = document_block()
        numeric_data["source"]["data"] = 4
        invalid = [
            ({"type": "document"}, "base64 application/pdf"),
            (document_block(title=123), "title must be a string"),
            (
                document_block(citations={"enabled": True}),
                "citations are not supported",
            ),
            (numeric_data, "PDF data must be a base64 string"),
        ]
        for block, message in invalid:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(APIError, message),
            ):
                documents.document_parts(block)

    def test_invalid_pdf_and_encryption_are_rejected(self):
        for payload, message in (
            (b"not a PDF", "could not be decoded"),
            (b"", "could not be decoded"),
            (pdf_bytes(encrypted=True), "opening password"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(APIError, message),
            ):
                render_pdf(payload)
        for encoded in ("***", "é"):
            with (
                self.subTest(encoded=encoded),
                self.assertRaisesRegex(APIError, "PDF data is not valid base64"),
            ):
                documents.file_content(
                    {"file_data": encoded},
                    budget=documents.DocumentBudget(deadline=math.inf),
                )

    def test_document_bounds_fail_before_processing_more_content(self):
        import pypdfium2 as pdfium

        with (
            mock.patch.object(documents, "MAX_PDF_BYTES", 3),
            mock.patch.object(documents, "_render") as render,
        ):
            with self.assertRaisesRegex(APIError, "size limit"):
                render_pdf()
            render.assert_not_called()
        for payload, message in (
            (pdf_bytes(pages=0), "could not be decoded"),
            (pdf_bytes(pages=documents.MAX_PAGES + 1), "pages"),
        ):
            with mock.patch.object(pdfium.PdfPage, "render") as render:
                with self.assertRaisesRegex(APIError, message):
                    render_pdf(payload)
                render.assert_not_called()
        with (
            mock.patch.object(documents, "MAX_TEXT_CHARACTERS", 3),
            mock.patch.object(pdfium.PdfPage, "render") as render,
        ):
            with self.assertRaisesRegex(APIError, "text exceeds"):
                render_pdf()
            render.assert_not_called()
        with mock.patch.object(documents, "MAX_RENDERED_BYTES", 1):
            with self.assertRaisesRegex(APIError, "rendered PDF exceeds"):
                render_pdf()

    def test_cache_reuses_rendered_pages_and_returns_independent_parts(self):
        with mock.patch.object(documents, "_render", wraps=documents._render) as render:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(render_pdf, [pdf_bytes()] * 8))
        self.assertEqual(render.call_count, 1)
        self.assertIs(results[0][0]["text"], results[1][0]["text"])
        self.assertIs(
            results[0][1]["image_url"]["url"], results[1][1]["image_url"]["url"]
        )
        results[0][0]["text"] = "changed"
        results[0][1]["image_url"]["url"] = "changed"
        self.assertIn("ALPHA 42", results[1][0]["text"])
        self.assertTrue(results[1][1]["image_url"]["url"].startswith("data:image/png"))
        self.assertIn("ALPHA 42", render_pdf()[0]["text"])

    def test_small_pdf_uses_native_page_limit_instead_of_twenty_pages(self):
        from server.protocol import MAX_IMAGE_SPANS

        self.assertEqual(documents.MAX_PAGES, MAX_IMAGE_SPANS)
        for pages in (21, documents.MAX_PAGES):
            with self.subTest(pages=pages):
                parts = render_pdf(pdf_bytes(pages=pages, width=64, height=64))
                self.assertEqual(len(parts), pages * 2)
                self.assertIn(f"PDF page {pages}:", parts[-2]["text"])

    def test_source_larger_than_ten_mib_within_conversion_budget(self):
        payload = pdf_bytes(padding_bytes=11 * 1024 * 1024)
        budget = documents.DocumentBudget(deadline=math.inf)
        parts = render_pdf(payload, budget)
        self.assertIn("ALPHA 42", parts[0]["text"])
        self.assertLess(
            budget.remaining_bytes, documents.MAX_REQUEST_DOCUMENT_BYTES - len(payload)
        )

    def test_pdf_pages_and_source_bytes_share_request_budget_on_cache_hits(self):
        payload = pdf_bytes(pages=2)
        render_pdf(payload)
        with mock.patch.object(documents, "_render") as render:
            budget = documents.DocumentBudget(deadline=math.inf, remaining_pages=3)
            render_pdf(payload, budget)
            self.assertEqual(budget.remaining_pages, 1)
            with self.assertRaisesRegex(APIError, "request page limit"):
                render_pdf(payload, budget)
            with self.assertRaisesRegex(APIError, "request size limit"):
                render_pdf(
                    payload,
                    documents.DocumentBudget(deadline=math.inf, remaining_bytes=1),
                )
            render.assert_not_called()

    def test_cold_pdf_obeys_remaining_page_budget(self):
        with self.assertRaisesRegex(APIError, "pages"):
            render_pdf(
                pdf_bytes(pages=3),
                documents.DocumentBudget(deadline=math.inf, remaining_pages=2),
            )
        self.assertFalse(documents._cache)

    def test_request_budget_counts_repeated_pdfs_with_and_without_cache(self):
        render_pdf()
        size = documents._cache.bytes
        source_size = len(pdf_bytes())
        for keep_cache in (False, True):
            self.setUp()
            budget = documents.DocumentBudget(
                deadline=math.inf, remaining_bytes=2 * (size + source_size) - 1
            )
            with mock.patch.object(
                documents, "_render", wraps=documents._render
            ) as render:
                render_pdf(budget=budget)
                if not keep_cache:
                    self.setUp()
                with self.assertRaisesRegex(APIError, "size limit"):
                    render_pdf(budget=budget)
            self.assertEqual(render.call_count, 1 if keep_cache else 2)
            self.assertEqual(budget.remaining_bytes, size - 1)

    def test_shared_budget_error_matches_on_cold_and_cached_pdf(self):
        payload = pdf_bytes()
        for cached in (False, True):
            for remaining in (len(payload), len(payload) + 1):
                with self.subTest(cached=cached, remaining=remaining):
                    self.setUp()
                    if cached:
                        render_pdf(payload)
                    with self.assertRaises(APIError) as raised:
                        render_pdf(
                            payload,
                            documents.DocumentBudget(
                                deadline=math.inf, remaining_bytes=remaining
                            ),
                        )
                    self.assertEqual(raised.exception.status, 400)
                    self.assertEqual(
                        raised.exception.message,
                        "PDF sources and rendered pages exceed the shared request size limit",
                    )
                    self.assertEqual(bool(documents._cache), cached)
                    self.assertIn("ALPHA 42", render_pdf(payload)[0]["text"])

    def test_only_an_uncached_pdf_waits_for_the_renderer(self):
        render_pdf()
        with documents._render_lock, mock.patch.object(documents, "_render") as render:
            # While another PDF renders, a cached one is answered at once; one
            # that waited for the renderer would time out.
            budget = documents.DocumentBudget(deadline=time.monotonic() + 2)
            self.assertIn("ALPHA 42", render_pdf(budget=budget)[0]["text"])
            # Another waits for the renderer within its deadline.
            with self.assertRaises(APIError) as raised:
                render_pdf(
                    pdf_bytes("BETA 7"),
                    documents.DocumentBudget(deadline=time.monotonic() + 0.01),
                )
            self.assertEqual(
                (raised.exception.status, raised.exception.code),
                (504, "request_timeout"),
            )
            render.assert_not_called()
        self.assertFalse(documents._render_lock.locked())

    def test_expired_render_closes_native_objects_and_does_not_cache_partial_pdf(self):
        import pypdfium2 as pdfium

        budget = documents.DocumentBudget(deadline=time.monotonic() + 10)
        original = pdfium.PdfPage.render
        handles = []

        def render(page, **kwargs):
            bitmap = original(page, **kwargs)
            handles.extend((page, page.pdf, bitmap))
            budget.deadline = time.monotonic() - 1
            return bitmap

        with mock.patch.object(pdfium.PdfPage, "render", render):
            with self.assertRaises(APIError) as raised:
                documents.render_pages(pdf_bytes(pages=2), budget, full_render_limits())
        self.assertEqual(raised.exception.code, "request_timeout")
        self.assertEqual(len(handles), 3)
        self.assertTrue(all(handle.raw is None for handle in handles))
        self.assertFalse(documents._cache)
        self.assertEqual(documents._cache.bytes, 0)
        self.assertFalse(documents._render_lock.locked())
        self.assertIn("ALPHA 42", render_pdf()[0]["text"])

    def test_cache_evicts_by_bytes_and_entry_count(self):
        page = documents.Page("text", "image")
        for budget, entries in ((page.size, 16), (documents.CACHE_BYTES, 1)):
            with (
                mock.patch.object(documents, "_render", return_value=(page,)) as render,
                mock.patch.object(documents, "_cache", LRUCache(budget, entries)),
            ):
                for payload in (b"first", b"second", b"first"):
                    render_pdf(payload)
                self.assertEqual(render.call_count, 3)
                self.assertEqual(len(documents._cache), 1)
                self.assertLessEqual(documents._cache.bytes, budget)


if __name__ == "__main__":
    unittest.main()

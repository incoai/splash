import base64
import io
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from PIL import Image

from server import documents, images
from server.errors import APIError


def pdf_bytes(
    text="ALPHA 42", *, pages=1, width=200, height=200, encrypted=False, padding_bytes=0
):
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for index in range(pages):
        page_id = len(objects) + 1
        kids.append(f"{page_id} 0 R")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode()
        )
        content = (
            f"1 0 0 rg 0 0 {width / 2} {height / 2} re f\n"
            f"0 0 1 rg {width / 2} 0 {width / 2} {height / 2} re f\n"
            f"0 0 0 rg BT /F1 16 Tf 10 {height - 30} Td ({text} page {index + 1}) Tj ET"
        ).encode("ascii")
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream"
        )
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {pages} >>".encode()
    if padding_bytes:
        objects.append(
            f"<< /Length {padding_bytes} >>\nstream\n".encode()
            + b" " * padding_bytes
            + b"\nendstream"
        )
    encryption = ""
    if encrypted:
        objects.append(
            b"<< /Filter /Standard /V 1 /R 2 /Length 40 /P -4 "
            b"/O <0000000000000000000000000000000000000000000000000000000000000000> "
            b"/U <0000000000000000000000000000000000000000000000000000000000000000> >>"
        )
        encryption = (
            f"/Encrypt {len(objects)} 0 R /ID [<0123456789abcdef> <0123456789abcdef>]"
        )
    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(
        f"trailer\n<< /Root 1 0 R /Size {len(offsets)} {encryption} >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(result)


def document_block(payload=None, **fields):
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(
                pdf_bytes() if payload is None else payload
            ).decode(),
        },
        **fields,
    }


class DocumentTests(unittest.TestCase):
    def test_owner_password_does_not_prevent_opening_but_user_password_does(self):
        fixtures = Path(__file__).parents[1] / "fixtures" / "documents"
        owner = (fixtures / "owner-password.pdf").read_bytes()
        locked = (fixtures / "open-password.pdf").read_bytes()
        # Both fixtures contain the same page, encrypted with AES-128. The
        # first has an empty opening password; the second requires one.
        parts = documents.document_content(document_block(owner))
        self.assertIn("ALPHA 42", parts[0]["text"])
        with self.assertRaisesRegex(APIError, "opening password"):
            documents.document_content(document_block(locked))

    def setUp(self):
        with documents._pdf_lock:
            documents._cache.clear()
            documents._cache_bytes = 0

    def test_pdf_keeps_page_text_images_and_document_context(self):
        parts = documents.document_content(
            document_block(pdf_bytes(pages=2), title="Report", context="Local fixture")
        )
        self.assertEqual(
            parts[:2],
            [
                {"type": "text", "text": "Report\n"},
                {"type": "text", "text": "Local fixture\n"},
            ],
        )
        self.assertIn("ALPHA 42 page 1", parts[2]["text"])
        self.assertIn("ALPHA 42 page 2", parts[4]["text"])
        for part in (parts[3], parts[5]):
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
                pdf_bytes(width=14400, height=14400), documents.DocumentBudget()
            )
        self.assertEqual(len(sizes), 1)

    def test_input_types_invalid_pdf_and_encryption_are_rejected(self):
        invalid = [
            ({"type": "document"}, "base64 application/pdf"),
            (document_block(b"not a PDF"), "could not be decoded"),
            (document_block(b""), "could not be decoded"),
            (document_block(pdf_bytes(encrypted=True)), "opening password"),
            (document_block(title=123), "title must be a string"),
            (
                document_block(citations={"enabled": True}),
                "citations are not supported",
            ),
        ]
        for block, message in invalid:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(APIError, message),
            ):
                documents.document_content(block)
        for encoded in ("***", "é", 4):
            block = document_block()
            block["source"]["data"] = encoded
            with (
                self.subTest(encoded=encoded),
                self.assertRaisesRegex(APIError, "base64"),
            ):
                documents.document_content(block)

    def test_document_bounds_fail_before_processing_more_content(self):
        import pypdfium2 as pdfium

        block = document_block()
        with (
            mock.patch.object(documents, "MAX_PDF_BYTES", 3),
            mock.patch.object(documents, "_render") as render,
        ):
            with self.assertRaisesRegex(APIError, "size limit"):
                documents.document_content(block)
            render.assert_not_called()
        for payload, message in (
            (pdf_bytes(pages=0), "could not be decoded"),
            (pdf_bytes(pages=documents.MAX_PAGES + 1), "pages"),
        ):
            with mock.patch.object(pdfium.PdfPage, "render") as render:
                with self.assertRaisesRegex(APIError, message):
                    documents.document_content(document_block(payload))
                render.assert_not_called()
        with (
            mock.patch.object(documents, "MAX_TEXT_CHARACTERS", 3),
            mock.patch.object(pdfium.PdfPage, "render") as render,
        ):
            with self.assertRaisesRegex(APIError, "text exceeds"):
                documents.document_content(block)
            render.assert_not_called()
        with mock.patch.object(documents, "MAX_RENDERED_BYTES", 1):
            with self.assertRaisesRegex(APIError, "rendered PDF exceeds"):
                documents.document_content(block)

    def test_cache_reuses_rendered_pages_and_returns_independent_parts(self):
        block = document_block()
        with mock.patch.object(documents, "_render", wraps=documents._render) as render:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(documents.document_content, [block] * 8))
        self.assertEqual(render.call_count, 1)
        self.assertIs(results[0][0]["text"], results[1][0]["text"])
        self.assertIs(
            results[0][1]["image_url"]["url"], results[1][1]["image_url"]["url"]
        )
        results[0][0]["text"] = "changed"
        results[0][1]["image_url"]["url"] = "changed"
        self.assertIn("ALPHA 42", results[1][0]["text"])
        self.assertTrue(results[1][1]["image_url"]["url"].startswith("data:image/png"))
        self.assertIn("ALPHA 42", documents.document_content(block)[0]["text"])

    def test_small_pdf_uses_native_page_limit_instead_of_twenty_pages(self):
        from server.protocol import ProtocolLimits

        self.assertEqual(documents.MAX_PAGES, ProtocolLimits().max_image_spans)
        for pages in (21, documents.MAX_PAGES):
            with self.subTest(pages=pages):
                parts = documents.document_content(
                    document_block(pdf_bytes(pages=pages, width=64, height=64))
                )
                self.assertEqual(len(parts), pages * 2)
                self.assertIn(f"PDF page {pages}:", parts[-2]["text"])

    def test_source_larger_than_ten_mib_within_conversion_budget(self):
        payload = pdf_bytes(padding_bytes=11 * 1024 * 1024)
        budget = documents.DocumentBudget()
        parts = documents.document_content(document_block(payload), budget=budget)
        self.assertIn("ALPHA 42", parts[0]["text"])
        self.assertLess(
            budget.remaining_bytes, documents.MAX_REQUEST_DOCUMENT_BYTES - len(payload)
        )

    def test_pdf_pages_and_source_bytes_share_request_budget_on_cache_hits(self):
        block = document_block(pdf_bytes(pages=2))
        documents.document_content(block)
        with mock.patch.object(documents, "_render") as render:
            budget = documents.DocumentBudget(remaining_pages=3)
            documents.document_content(block, budget=budget)
            self.assertEqual(budget.remaining_pages, 1)
            with self.assertRaisesRegex(APIError, "request page limit"):
                documents.document_content(block, budget=budget)
            with self.assertRaisesRegex(APIError, "request size limit"):
                documents.document_content(
                    block, budget=documents.DocumentBudget(remaining_bytes=1)
                )
            render.assert_not_called()

    def test_cold_pdf_obeys_remaining_page_budget(self):
        with self.assertRaisesRegex(APIError, "pages"):
            documents.document_content(
                document_block(pdf_bytes(pages=3)),
                budget=documents.DocumentBudget(remaining_pages=2),
            )
        self.assertFalse(documents._cache)

    def test_request_budget_counts_repeated_pdfs_with_and_without_cache(self):
        block = document_block()
        documents.document_content(block)
        size = sum(page.size for pages in documents._cache.values() for page in pages)
        source_size = len(base64.b64decode(block["source"]["data"]))
        for keep_cache in (False, True):
            self.setUp()
            budget = documents.DocumentBudget(
                remaining_bytes=2 * (size + source_size) - 1
            )
            with mock.patch.object(
                documents, "_render", wraps=documents._render
            ) as render:
                documents.document_content(block, budget=budget)
                if not keep_cache:
                    self.setUp()
                with self.assertRaisesRegex(APIError, "size limit"):
                    documents.document_content(block, budget=budget)
            self.assertEqual(render.call_count, 1 if keep_cache else 2)
            self.assertEqual(budget.remaining_bytes, size - 1)

    def test_shared_budget_error_matches_on_cold_and_cached_pdf(self):
        payload = pdf_bytes()
        block = document_block(payload)
        for cached in (False, True):
            for remaining in (len(payload), len(payload) + 1):
                with self.subTest(cached=cached, remaining=remaining):
                    self.setUp()
                    if cached:
                        documents.document_content(block)
                    with self.assertRaises(APIError) as raised:
                        documents.document_content(
                            block,
                            budget=documents.DocumentBudget(remaining_bytes=remaining),
                        )
                    self.assertEqual(raised.exception.status, 400)
                    self.assertEqual(
                        raised.exception.message,
                        "PDF sources and rendered pages exceed the shared request size limit",
                    )
                    self.assertEqual(bool(documents._cache), cached)
                    self.assertIn(
                        "ALPHA 42", documents.document_content(block)[0]["text"]
                    )

    def test_waiting_for_renderer_expires_without_decoding_more_pdf_bytes(self):
        block = document_block()
        with (
            documents._pdf_lock,
            mock.patch.object(documents.base64, "b64decode") as decode,
        ):
            with self.assertRaises(APIError) as raised:
                documents.document_content(
                    block,
                    budget=documents.DocumentBudget(deadline=time.monotonic() + 0.01),
                )
            self.assertEqual(raised.exception.status, 504)
            self.assertEqual(raised.exception.code, "request_timeout")
            decode.assert_not_called()
        self.assertFalse(documents._pdf_lock.locked())

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
                documents.render_pages(pdf_bytes(pages=2), budget)
        self.assertEqual(raised.exception.code, "request_timeout")
        self.assertEqual(len(handles), 3)
        self.assertTrue(all(handle.raw is None for handle in handles))
        self.assertFalse(documents._cache)
        self.assertEqual(documents._cache_bytes, 0)
        self.assertFalse(documents._pdf_lock.locked())
        self.assertIn(
            "ALPHA 42", documents.document_content(document_block())[0]["text"]
        )

    def test_cache_evicts_by_bytes_and_entry_count(self):
        page = documents.Page("text", "image")
        for budget, entries in ((page.size, 16), (documents.CACHE_BYTES, 1)):
            self.setUp()
            with (
                mock.patch.object(documents, "_render", return_value=(page,)) as render,
                mock.patch.object(documents, "CACHE_BYTES", budget),
                mock.patch.object(documents, "CACHE_ENTRIES", entries),
            ):
                for payload in (b"first", b"second", b"first"):
                    documents.document_content(document_block(payload))
                self.assertEqual(render.call_count, 3)
                self.assertEqual(len(documents._cache), 1)
                self.assertLessEqual(documents._cache_bytes, budget)


if __name__ == "__main__":
    unittest.main()

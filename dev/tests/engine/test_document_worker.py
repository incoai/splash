import math
import subprocess
import sys
import unittest
from dataclasses import asdict
from unittest import mock

from dev.tests.engine.test_documents import (
    full_render_limits,
    pdf_bytes,
    render_pdf,
)
from server import document_worker, documents
from server.errors import APIError


class DocumentWorkerTests(unittest.TestCase):
    def setUp(self):
        with documents._pdf_lock:
            documents._cache.clear()
            documents._cache_bytes = 0

    def sleeping_worker(self, children):
        start = subprocess.Popen

        def spawn(command, **kwargs):
            process = start(
                [sys.executable, "-c", "import time; time.sleep(10)"], **kwargs
            )
            children.append(process)
            self.addCleanup(document_worker._stop, process)
            return process

        return mock.patch.object(document_worker.subprocess, "Popen", spawn)

    def assert_released(self, children):
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertFalse(document_worker._workers)
        self.assertFalse(documents._cache)
        self.assertFalse(documents._pdf_lock.locked())
        self.assertIn("ALPHA 42", render_pdf()[0]["text"])

    def test_deadline_stops_worker_and_next_document_succeeds(self):
        children = []
        with self.sleeping_worker(children):
            with self.assertRaises(APIError) as caught:
                document_worker.render(pdf_bytes(), asdict(full_render_limits()), 0.1)
        self.assertEqual(caught.exception.status, 504)
        self.assertEqual(caught.exception.code, "request_timeout")
        self.assert_released(children)

    def test_worker_time_limit_without_a_deadline_is_a_request_error(self):
        children = []
        with (
            self.sleeping_worker(children),
            mock.patch.object(document_worker, "MAX_SECONDS", 0.1),
        ):
            with self.assertRaises(APIError) as caught:
                document_worker.render(
                    pdf_bytes(), asdict(full_render_limits()), math.inf
                )
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(
            caught.exception.message, "PDF processing exceeded the time limit"
        )
        self.assert_released(children)

    def test_memory_budget_stops_worker_without_caching_partial_output(self):
        children = []
        with (
            self.sleeping_worker(children),
            mock.patch.object(
                document_worker,
                "_memory_bytes",
                return_value=document_worker.MAX_MEMORY_BYTES + 1,
            ),
        ):
            with self.assertRaisesRegex(APIError, "memory limit"):
                render_pdf()
        self.assert_released(children)

    def test_failed_measurement_stops_worker(self):
        children = []
        with (
            self.sleeping_worker(children),
            mock.patch.object(document_worker, "_memory_bytes", side_effect=OSError),
        ):
            with self.assertRaises(APIError) as caught:
                render_pdf()
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(caught.exception.code, "document_unavailable")
        self.assert_released(children)

    def test_spawn_failure_is_transient_and_leaves_no_worker(self):
        with mock.patch.object(
            document_worker.subprocess, "Popen", side_effect=OSError
        ):
            with self.assertRaises(APIError) as caught:
                render_pdf()
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(caught.exception.code, "document_unavailable")
        self.assertFalse(document_worker._workers)
        self.assertFalse(documents._cache)


if __name__ == "__main__":
    unittest.main()

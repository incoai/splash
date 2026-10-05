import base64
import concurrent.futures
import copy
import gc
import json
import threading
import time
import unittest
import weakref
from unittest import mock

from dev.tests import server_fixtures as fixtures
from server import frontend


class ResponseLifecycleTests(unittest.TestCase):
    def harness(self, **kwargs):
        harness = fixtures.Harness(fixtures.FakeRuntime(), **kwargs)
        self.addCleanup(harness.close)
        return harness

    def test_waiting_requests_do_not_load_history(self):
        harness = self.harness(queue_size=8)
        harness.app.response_store.put(
            {"id": "resp_previous"}, [{"role": "user", "content": "x" * 2048}]
        )
        for _ in range(2):
            harness.app.preparation_slots.acquire()
        body = {"input": "continue", "previous_response_id": "resp_previous"}
        with (
            mock.patch.object(
                harness.app.response_store, "get", wraps=harness.app.response_store.get
            ) as get,
            concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool,
        ):
            calls = [
                pool.submit(harness.request, "POST", "/v1/responses", body)
                for _ in range(4)
            ]
            try:
                deadline = time.monotonic() + 1
                while (
                    harness.app.preparation_waiting != 4 and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
                self.assertEqual(harness.app.preparation_waiting, 4)
                get.assert_not_called()
            finally:
                for _ in range(2):
                    harness.app.preparation_slots.release()
            self.assertTrue(all(call.result()[0] == 200 for call in calls))

    def test_response_lookup_does_not_decode_history(self):
        harness = self.harness()
        expected = {"id": "resp_previous", "status": "completed"}
        harness.app.response_store.put(
            expected, [{"role": "user", "content": "x" * 2048}]
        )
        record = harness.app.response_store.get("resp_previous")
        with mock.patch.object(frontend.json, "loads", wraps=json.loads) as decode:
            status, _, payload = harness.request("GET", "/v1/responses/resp_previous")
            self.assertEqual((status, json.loads(payload)), (200, expected))
            self.assertFalse(
                any(
                    call.args[0] is record.history_json
                    for call in decode.call_args_list
                )
            )
        result = record.response
        result["status"] = "changed"
        harness.app.response_store.delete("resp_previous")
        self.assertEqual(record.response, expected)
        self.assertEqual(json.loads(record.history_json)[0]["content"], "x" * 2048)

    def test_store_false_uses_history_without_retaining_it(self):
        harness = self.harness()
        harness.app.response_store.put(
            {"id": "resp_previous"}, [{"role": "user", "content": "old history"}]
        )
        job = harness.app.prepare_responses(
            {
                "input": "continue",
                "previous_response_id": "resp_previous",
                "store": False,
            },
            deadline=fixtures.FOREVER,
        )
        self.assertIn("old history", str(harness.tokenizer.templates))
        self.assertIsNone(job.response_history_items)
        self.assertFalse(job.response_store)

    def test_stored_history_is_independent_of_the_request_body(self):
        harness = self.harness()
        question = {"type": "message", "role": "user", "content": "question"}
        answer = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer"}],
        }
        body = {"input": [dict(question)]}
        job = harness.app.prepare_responses(body, deadline=fixtures.FOREVER)
        output = [copy.deepcopy(answer)]
        harness.app.persist_response(job, {"id": "resp_stored"}, output)
        # The record was encoded when it was stored.
        body["input"][0]["content"] = "changed"
        output[0]["content"][0]["text"] = "changed"
        record = harness.app.response_store.get("resp_stored")
        self.assertEqual(json.loads(record.history_json), [question, answer])

    def test_stored_history_is_the_input_as_posted(self):
        # Preparation renders the PDF and decodes the image of the items the
        # record stores, and leaves them as they were posted.
        harness = self.harness(
            tokenizer=fixtures.ImagePadTokenizer(), max_context=65536
        )
        pdf = base64.b64encode(fixtures.pdf_bytes()).decode()
        posted = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Compare these."},
                    {
                        "type": "input_file",
                        "filename": "receipt.pdf",
                        "file_data": "data:application/pdf;base64," + pdf,
                    },
                    {
                        "type": "input_image",
                        "image_url": fixtures.png_data_url(),
                    },
                ],
            }
        ]
        status, _, payload = harness.request(
            "POST", "/v1/responses", {"input": posted, "max_output_tokens": 16}
        )
        self.assertEqual(status, 200, payload)
        record = harness.app.response_store.get(json.loads(payload)["id"])
        self.assertEqual(json.loads(record.history_json)[: len(posted)], posted)

    def test_completion_before_submit_returns_releases_job_without_gc(self):
        harness = self.harness()
        submit_native = harness.backend.runtime.submit
        submit_job = harness.backend.submit
        references = []

        def completed(*args, **kwargs):
            call = submit_native(*args, **kwargs)
            deadline = time.monotonic() + 1
            while harness.backend.active and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertFalse(harness.backend.active)
            return call

        def track(job):
            references.append(weakref.ref(job))
            return submit_job(job)

        enabled = gc.isenabled()
        gc.disable()
        try:
            with (
                mock.patch.object(harness.backend.runtime, "submit", completed),
                mock.patch.object(harness.backend, "submit", track),
            ):
                status, _, payload = harness.request(
                    "POST", "/v1/responses", {"input": "history " * 100}
                )
                self.assertEqual(status, 200, payload)
            for thread in harness.backend.runtime.threads:
                thread.join(1)
            harness.backend.runtime.calls.clear()
            self.assertEqual(len(references), 1)
            deadline = time.monotonic() + 1
            while references[0]() is not None and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIsNone(references[0]())
        finally:
            if enabled:
                gc.enable()

    def test_response_store_is_strictly_byte_bounded_lru(self):
        self.enterContext(
            mock.patch.object(frontend.ResponseStore, "BUDGET_BYTES", 2048)
        )
        store = frontend.ResponseStore()
        for index in range(10):
            response = {
                "id": f"resp_{index}",
                "object": "response",
                "output": [{"type": "message", "text": "x" * 128}],
            }
            self.assertTrue(store.put(response, [{"text": "y" * 128}]))
        stats = store.stats()
        self.assertLessEqual(stats["bytes"], stats["budget_bytes"])
        self.assertGreater(stats["evictions"], 0)
        self.assertIsNone(store.get("resp_0"))
        self.assertIsNotNone(store.get("resp_9"))
        with mock.patch.object(frontend.ResponseStore, "BUDGET_BYTES", 64):
            self.assertFalse(
                frontend.ResponseStore().put(
                    {"id": "resp_large", "text": "x" * 128}, []
                )
            )

    def test_response_store_concurrent_churn_remains_bounded(self):
        self.enterContext(
            mock.patch.object(frontend.ResponseStore, "BUDGET_BYTES", 8192)
        )
        store = frontend.ResponseStore()
        errors = []

        def churn(shard):
            try:
                for index in range(100):
                    response_id = f"resp_{shard}_{index}"
                    self.assertTrue(
                        store.put(
                            {"id": response_id, "text": "x" * 128},
                            [{"shard": shard, "index": index}],
                        )
                    )
                    store.get(response_id)
                    if index % 3 == 0:
                        store.delete(response_id)
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=churn, args=(shard,)) for shard in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        stats = store.stats()
        self.assertLessEqual(stats["bytes"], stats["budget_bytes"])
        self.assertGreater(stats["evictions"], 0)

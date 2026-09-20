import concurrent.futures
import gc
import json
import time
import unittest
import weakref
from unittest import mock

from dev.tests import test_server as fixtures
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
        job, *_ = harness.app.prepare_responses(
            {
                "input": "continue",
                "previous_response_id": "resp_previous",
                "store": False,
            }
        )
        self.assertIn("old history", str(harness.tokenizer.templates))
        self.assertIsNone(job.response_history_items)
        self.assertFalse(job.response_store)

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

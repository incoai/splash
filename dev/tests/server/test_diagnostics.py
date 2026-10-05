import io
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from dev.tests.server_fixtures import FakeRuntime, HarnessTestCase, chat_body
from server import diagnostics
from server import server as api


class DiagnosticsTests(HarnessTestCase):
    def test_expected_client_disconnect_does_not_print_server_traceback(self):
        harness = self.harness(FakeRuntime())
        with mock.patch.object(api.ThreadingHTTPServer, "handle_error") as parent:
            with mock.patch.object(
                api.sys,
                "exc_info",
                return_value=(ConnectionResetError, ConnectionResetError(), None),
            ):
                harness.server.handle_error(None, ("127.0.0.1", 1))
            parent.assert_not_called()

            unexpected = RuntimeError("unexpected")
            with mock.patch.object(
                api.sys,
                "exc_info",
                return_value=(RuntimeError, unexpected, None),
            ):
                harness.server.handle_error(None, ("127.0.0.1", 1))
            parent.assert_called_once()

    def test_http_error_logs_method_and_path_without_query_or_body(self):
        harness = self.harness(FakeRuntime())
        for method in ("GET", "POST", "DELETE"):
            with (
                self.subTest(method=method),
                mock.patch.object(api, "print_status") as output,
            ):
                status, _, _ = harness.request(
                    method, "/missing?token=private-query", {"private-body": "secret"}
                )
                self.assertEqual(status, 404)
                output.assert_called_once_with(
                    f"Error · not_found · {method} /missing", error=True
                )
        with mock.patch.object(api, "print_status") as output:
            status, _, _ = harness.request("POST", "/v1/messages?beta=true", {})
            self.assertEqual(status, 400)
            output.assert_called_once_with(
                "Error · invalid_request_error · POST /v1/messages", error=True
            )

    def test_request_record_contains_metrics_but_console_omits_prompt_and_id(self):
        records = []
        harness = self.harness(FakeRuntime(), request_logger=records.append)
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["outcome"], "stop")
        self.assertNotIn("queue_ms", record)
        self.assertNotIn("queue_ms", record["metrics"])
        self.assertNotIn("hello", json.dumps(record))

        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            diagnostics.print_request(record)
        line = output.getvalue()
        self.assertRegex(line, r"^\d{2}:\d{2}:\d{2} Done · input [^\n]*\n$")
        self.assertNotIn("request_id", line)
        self.assertNotIn("hello", line)

    def test_console_request_summary(self):
        record = {
            "request_id": 123,
            "outcome": "stop",
            "prompt_tokens": 10240,
            "completion_tokens": 320,
            "metrics": {
                "cache": {"matched_tokens": 8192},
                "request_latency": {
                    "ttft_ms": 800,
                    "stream_tokens_per_second": 85,
                },
            },
        }
        with (
            mock.patch.object(api.time, "strftime", return_value="14:32:08"),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            diagnostics.print_request(record)
        self.assertEqual(
            output.getvalue(),
            "14:32:08 Done · input 10,240 · cached 8,192 · output 320"
            " · TTFT 0.8s · 85.0 tok/s\n",
        )

    def test_console_shows_the_tool_block_signature(self):
        record = {
            "outcome": "stop",
            "prompt_tokens": 33_799,
            "completion_tokens": 12,
            "tools": {"count": 27, "signature": "1a2b3c4d"},
            "metrics": {"cache": {"matched_tokens": 6_656}},
        }
        with (
            mock.patch.object(api.time, "strftime", return_value="14:32:08"),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            diagnostics.print_request(record)
        self.assertEqual(
            output.getvalue(),
            "14:32:08 Done · input 33,799 · cached 6,656 · output 12"
            " · tools 27·1a2b3c4d\n",
        )

    def test_console_cancellation_without_tokens_or_latency(self):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            diagnostics.print_request({"outcome": "cancelled", "prompt_tokens": 32})
        line = output.getvalue()
        self.assertIn("Cancelled · input 32 · cached 0 · output 0", line)
        self.assertNotIn("TTFT", line)
        self.assertNotIn("tok/s", line)

    def test_console_error_omits_private_details(self):
        with (
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
            mock.patch("sys.stderr", new_callable=io.StringIO) as errors,
        ):
            diagnostics.print_request(
                {
                    "outcome": "error",
                    "error_code": "context_length_exceeded",
                    "error_message": "private prompt\n用户输入",
                    "request_id": 123,
                }
            )
        self.assertRegex(
            errors.getvalue(), r"^\d{2}:\d{2}:\d{2} Error · context_length_exceeded\n$"
        )
        self.assertEqual(output.getvalue(), "")

    def test_concurrent_console_lines_stay_whole(self):
        # Status lines on stdout and errors on stderr, both on one unbuffered
        # file, as `python -u server.py > log 2>&1` or launchd writes them.
        with tempfile.TemporaryFile() as log:
            streams = [
                io.TextIOWrapper(
                    io.FileIO(os.dup(log.fileno()), "w"), write_through=True
                )
                for _ in range(2)
            ]
            self.addCleanup(lambda: [stream.close() for stream in streams])

            def write(worker):
                for line in range(300):
                    diagnostics.print_status(
                        f"worker {worker} line {line}", error=worker % 2 == 1
                    )

            workers = [
                threading.Thread(target=write, args=(worker,)) for worker in range(8)
            ]
            with (
                mock.patch("sys.stdout", streams[0]),
                mock.patch("sys.stderr", streams[1]),
            ):
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join()
            log.seek(0)
            lines = log.read().decode().splitlines()
        self.assertEqual(len(lines), 8 * 300)
        for line in lines:
            self.assertRegex(line, r"^\d{2}:\d{2}:\d{2} worker \d line \d+$")


if __name__ == "__main__":
    unittest.main()

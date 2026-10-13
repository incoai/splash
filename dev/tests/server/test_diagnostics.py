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
        with mock.patch.object(api.HTTPServer, "handle_error") as parent:
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
        # The request's start, when the engine admits it, then its end.
        self.assertEqual([record["outcome"] for record in records], ["started", "stop"])
        started, record = records
        self.assertNotIn("queue_ms", record)
        self.assertNotIn("queue_ms", record["metrics"])
        self.assertNotIn("hello", json.dumps(records))

        for logged, pattern in (
            (
                started,
                r"^\d{2}:\d{2}:\d{2} Started · input \d+ · cached \d+"
                r" · effort none · thinking off\n$",
            ),
            (record, r"^\d{2}:\d{2}:\d{2} Done · input [^\n]*\n$"),
        ):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                diagnostics.print_request(logged)
            line = output.getvalue()
            self.assertRegex(line, pattern)
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

    def test_console_names_tokens_read_from_cached_blocks(self):
        # A cold request whose first blocks matched cached ones read those
        # blocks' KV in its later chunks: cached 0 alone would hide it.
        record = {
            "outcome": "stop",
            "prompt_tokens": 168,
            "completion_tokens": 0,
            "metrics": {"cache": {"matched_tokens": 0, "deduplicated_tokens": 64}},
        }
        with (
            mock.patch.object(api.time, "strftime", return_value="08:02:27"),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            diagnostics.print_request(record)
        self.assertEqual(
            output.getvalue(),
            "08:02:27 Done · input 168 · cached 0 · deduplicated 64 · output 0\n",
        )

    def test_console_names_tokens_per_speculative_cycle(self):
        # Five verified drafts accepted 20 tokens: each cycle yielded 5.
        record = {
            "outcome": "stop",
            "prompt_tokens": 168,
            "completion_tokens": 25,
            "metrics": {
                "cache": {"matched_tokens": 0},
                "speculation": {
                    "drafted_tokens": 35,
                    "accepted_draft_tokens": 20,
                    "cycles": 5,
                },
            },
        }
        with (
            mock.patch.object(api.time, "strftime", return_value="08:02:27"),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            diagnostics.print_request(record)
        self.assertEqual(
            output.getvalue(),
            "08:02:27 Done · input 168 · cached 0 · output 25 · 5.0 tok/cycle\n",
        )

    def test_console_request_start(self):
        with (
            mock.patch.object(api.time, "strftime", return_value="19:03:32"),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            diagnostics.print_request(
                {
                    "outcome": "started",
                    "prompt_tokens": 153_545,
                    "cached_tokens": 23_456,
                }
            )
        self.assertEqual(
            output.getvalue(), "19:03:32 Started · input 153,545 · cached 23,456\n"
        )

    def test_console_start_shows_how_a_chat_request_reasons(self):
        def line(**reasoning):
            with (
                mock.patch.object(api.time, "strftime", return_value="19:03:32"),
                mock.patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                diagnostics.print_request(
                    {
                        "outcome": "started",
                        "prompt_tokens": 2_473,
                        "cached_tokens": 0,
                        "reasoning": {
                            "sent_effort": None,
                            "effort": None,
                            "enable_thinking": None,
                            "thinking": True,
                            **reasoning,
                        },
                    }
                )
            return output.getvalue().removeprefix(
                "19:03:32 Started · input 2,473 · cached 0"
            )

        self.assertEqual(line(), " · effort absent · thinking on\n")
        self.assertEqual(line(effort="low"), " · effort absent → low · thinking on\n")
        self.assertEqual(
            line(sent_effort="none", effort="none", thinking=False),
            " · effort none · thinking off\n",
        )
        self.assertEqual(
            line(
                sent_effort="xhigh",
                effort="xhigh",
                enable_thinking=False,
                thinking=False,
            ),
            " · effort xhigh · enable_thinking false · thinking off\n",
        )

    def test_started_records_carry_chat_reasoning_options(self):
        records = []
        harness = self.harness(
            FakeRuntime(),
            request_logger=records.append,
            default_reasoning_effort="low",
        )
        for body in (
            chat_body(),
            chat_body(reasoning_effort="none"),
            chat_body(chat_template_kwargs={"enable_thinking": False}),
        ):
            status, _, payload = harness.request("POST", "/v1/chat/completions", body)
            self.assertEqual(status, 200, payload)
        status, _, payload = harness.request(
            "POST", "/v1/completions", {"model": "test-model", "prompt": "Hi"}
        )
        self.assertEqual(status, 200, payload)
        started = [record for record in records if record["outcome"] == "started"]
        self.assertEqual(
            [record.get("reasoning") for record in started],
            [
                {
                    "sent_effort": None,
                    "effort": "low",
                    "enable_thinking": None,
                    "thinking": True,
                },
                {
                    "sent_effort": "none",
                    "effort": "none",
                    "enable_thinking": None,
                    "thinking": False,
                },
                {
                    "sent_effort": None,
                    "effort": "low",
                    "enable_thinking": False,
                    "thinking": False,
                },
                None,
            ],
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

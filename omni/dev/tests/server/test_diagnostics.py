import io
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from dev.tests.audio_fixtures import (
    AudioHarnessTestCase,
    AudioPlan,
    FakeAudioRuntime,
    pcm,
    speech_body,
)
from server import audio_backend, diagnostics, http_server


class DiagnosticsTests(AudioHarnessTestCase):
    def test_expected_client_disconnect_does_not_print_server_traceback(self):
        harness = self.harness(FakeAudioRuntime())
        with mock.patch.object(http_server.HTTPServer, "handle_error") as parent:
            with mock.patch.object(
                http_server.sys,
                "exc_info",
                return_value=(ConnectionResetError, ConnectionResetError(), None),
            ):
                harness.server.handle_error(None, ("127.0.0.1", 1))
            parent.assert_not_called()

            unexpected = RuntimeError("unexpected")
            with mock.patch.object(
                http_server.sys,
                "exc_info",
                return_value=(RuntimeError, unexpected, None),
            ):
                harness.server.handle_error(None, ("127.0.0.1", 1))
            parent.assert_called_once()

    def test_http_error_logs_method_and_path_without_query_or_body(self):
        harness = self.harness(FakeAudioRuntime())
        for method in ("GET", "POST", "DELETE"):
            with (
                self.subTest(method=method),
                mock.patch.object(http_server, "print_status") as output,
            ):
                status, _, _ = harness.request(
                    method, "/missing?token=private-query", {"private-body": "secret"}
                )
                self.assertEqual(status, 404)
                output.assert_called_once_with(
                    f"Error · not_found · {method} /missing", error=True
                )
        with mock.patch.object(http_server, "print_status") as output:
            status, _, _ = harness.request(
                "POST", "/v1/audio/speech?beta=true", {"input": ["private text"]}
            )
            self.assertEqual(status, 400)
            output.assert_called_once_with(
                "Error · invalid_request_error · POST /v1/audio/speech", error=True
            )

    def test_request_record_counts_the_text_but_console_omits_it(self):
        records = []
        plan = AudioPlan(chunks=[pcm([0] * 48_000)])
        harness = self.harness(FakeAudioRuntime(plan), request_logger=records.append)
        status, _, _ = harness.request(
            "POST", "/v1/audio/speech", speech_body(input="hello there")
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["outcome"], "stop")
        self.assertEqual(record["kind"], "speech")
        self.assertEqual(record["streamed"], 48_000)
        self.assertNotIn("hello", json.dumps(record))

        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            audio_backend.print_audio_request(record)
        line = output.getvalue()
        self.assertRegex(line, r"^\d{2}:\d{2}:\d{2} Done · text [^\n]*\n$")
        self.assertNotIn("request_id", line)
        self.assertNotIn("hello", line)

    def test_console_request_summary(self):
        speech = {
            "request_id": 123,
            "outcome": "stop",
            "kind": "speech",
            "text_tokens": 1024,
            "prompt_tokens": 1040,
            "completion": 25,
            "streamed": 48_000,
            "wall_ms": 1000.0,
            "first_output_ms": 120.0,
        }
        transcription = {
            "outcome": "stop",
            "kind": "transcription",
            "seconds": 3.5,
            "completion": 1200,
            "streamed": 1200,
            "wall_ms": 500.0,
            "first_output_ms": 80.0,
        }
        for record, line in (
            (
                speech,
                "14:32:08 Done · text 1,024 tokens · audio 2.0s · 2.0x realtime"
                " · first output 0.12s\n",
            ),
            (
                transcription,
                "14:32:08 Done · audio 3.5s · output 1,200 tokens"
                " · first output 0.08s\n",
            ),
        ):
            with (
                self.subTest(kind=record["kind"]),
                mock.patch.object(
                    diagnostics.time, "strftime", return_value="14:32:08"
                ),
                mock.patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                audio_backend.print_audio_request(record)
                self.assertEqual(output.getvalue(), line)

    def test_console_cancellation_without_audio_or_latency(self):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            audio_backend.print_audio_request(
                {"outcome": "cancelled", "kind": "speech", "text_tokens": 32}
            )
        line = output.getvalue()
        self.assertIn("Cancelled · text 32 tokens · audio 0.0s", line)
        self.assertNotIn("realtime", line)
        self.assertNotIn("first output", line)

    def test_console_error_omits_private_details(self):
        with (
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
            mock.patch("sys.stderr", new_callable=io.StringIO) as errors,
        ):
            audio_backend.print_audio_request(
                {
                    "outcome": "error",
                    "kind": "speech",
                    "error_code": "limit_exceeded",
                    "error_message": "private text\n用户输入",
                    "request_id": 123,
                }
            )
        self.assertRegex(
            errors.getvalue(), r"^\d{2}:\d{2}:\d{2} Error · limit_exceeded\n$"
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

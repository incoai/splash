import errno
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.server_fixtures import (
    FakeRuntime,
    HarnessTestCase,
    StalledConsole,
    chat_body,
    console_written,
    full_pipe,
)
from server import diagnostics
from server import server as api


def printed(record):
    """Print a request's line and wait for the console to write it."""
    diagnostics.print_request(record)
    console_written()


def dropped_warning(count):
    """The console's warning that `count` lines were dropped, as a pattern."""
    return (
        r"\d{2}:\d{2}:\d{2} Warning · "
        f"{count:,} console lines dropped while the output stopped draining"
    )


def exited(command, both):
    """Run `command` from the checkout with its stdout, and its stderr too when
    `both`, on a pipe nobody reads, full already: its exit status, None once
    10 s pass first, the seconds it ran and what it wrote on a separate
    stderr."""
    unread, written, _ = full_pipe()
    try:
        started = time.monotonic()
        try:
            child = subprocess.Popen(
                command,
                cwd=Path(diagnostics.__file__).resolve().parents[1],
                stdout=written,
                stderr=written if both else subprocess.PIPE,
            )
        finally:
            os.close(written)
        try:
            status = child.wait(10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
            status = None
        elapsed = time.monotonic() - started
        if both:
            return status, elapsed, b""
        with child.stderr:
            return status, elapsed, child.stderr.read()
    finally:
        os.close(unread)


class DiagnosticsTests(HarnessTestCase):
    def test_a_failed_connection_is_one_console_line_and_a_disconnect_none(self):
        # Never socketserver's traceback, which quotes the error's message and
        # is written straight to stderr, on the connection worker.
        harness = self.harness(FakeRuntime())
        for error, written in (
            (ConnectionResetError(), ""),
            (BrokenPipeError(), ""),
            (
                RuntimeError("private request text"),
                "14:32:08 Error · internal_server_error · RuntimeError\n",
            ),
        ):
            with self.subTest(error=type(error).__name__):
                with (
                    mock.patch.object(api.time, "strftime", return_value="14:32:08"),
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    try:
                        raise error
                    except Exception:
                        harness.server.handle_error(None, ("127.0.0.1", 1))
                    console_written()
                self.assertEqual(stderr.getvalue(), written)

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

    def test_request_record_contains_metrics_but_console_omits_the_prompt(self):
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
                r"^\d{2}:\d{2}:\d{2} Started · request (\d+) · input \d+ · cached \d+"
                r" · effort none · thinking off\n$",
            ),
            (record, r"^\d{2}:\d{2}:\d{2} Done · request (\d+) · input [^\n]*\n$"),
        ):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                printed(logged)
            line = output.getvalue()
            self.assertRegex(line, pattern)
            self.assertNotIn("hello", line)
        # Both lines name the request the engine took, as its own lines do.
        self.assertEqual(started["request_id"], record["request_id"])

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
            printed(record)
        self.assertEqual(
            output.getvalue(),
            "14:32:08 Done · request 123 · input 10,240 · cached 8,192 · output 320"
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
            printed(record)
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
            printed(record)
        self.assertEqual(
            output.getvalue(),
            "08:02:27 Done · input 168 · cached 0 · output 25 · 5.0 tok/cycle\n",
        )

    def test_console_request_start(self):
        with (
            mock.patch.object(api.time, "strftime", return_value="19:03:32"),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            printed(
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
                printed(
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
            printed(record)
        self.assertEqual(
            output.getvalue(),
            "14:32:08 Done · input 33,799 · cached 6,656 · output 12"
            " · tools 27·1a2b3c4d\n",
        )

    def test_console_cancellation_without_tokens_or_latency(self):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            printed({"outcome": "cancelled", "prompt_tokens": 32})
        line = output.getvalue()
        self.assertIn("Cancelled · input 32 · cached 0 · output 0", line)
        self.assertNotIn("TTFT", line)
        self.assertNotIn("tok/s", line)

    def test_console_error_omits_private_details(self):
        with (
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
            mock.patch("sys.stderr", new_callable=io.StringIO) as errors,
        ):
            printed(
                {
                    "outcome": "error",
                    "error_code": "context_length_exceeded",
                    "error_message": "private prompt\n用户输入",
                    "request_id": 123,
                }
            )
        self.assertRegex(
            errors.getvalue(),
            r"^\d{2}:\d{2}:\d{2} Error · request 123 · context_length_exceeded\n$",
        )
        self.assertEqual(output.getvalue(), "")

    def test_console_error_names_the_engines_resource_reason(self):
        reason = (
            "requests past 147449 tokens keep their KV on the SSD cache, "
            "which takes no writes"
        )
        with mock.patch("sys.stderr", new_callable=io.StringIO) as errors:
            printed(
                {
                    "outcome": "error",
                    "error_code": "capacity_exhausted",
                    "engine_reason": reason,
                    "error_message": "private prompt",
                }
            )
        self.assertRegex(
            errors.getvalue(),
            r"^\d{2}:\d{2}:\d{2} Error · capacity_exhausted · " + reason + "\n$",
        )

    def test_concurrent_console_lines_stay_whole(self):
        # Status lines on stdout and errors on stderr, both on one unbuffered
        # file, as `python -u server.py > log 2>&1` or launchd writes them:
        # 800 lines, fewer than the console holds waiting, so that none is
        # dropped however slowly it writes them.
        with tempfile.TemporaryFile() as log:
            streams = [
                io.TextIOWrapper(
                    io.FileIO(os.dup(log.fileno()), "w"), write_through=True
                )
                for _ in range(2)
            ]
            self.addCleanup(lambda: [stream.close() for stream in streams])

            def write(worker):
                for line in range(100):
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
            console_written()
            log.seek(0)
            lines = log.read().decode().splitlines()
        self.assertEqual(len(lines), 8 * 100)
        for line in lines:
            self.assertRegex(line, r"^\d{2}:\d{2}:\d{2} worker \d line \d+$")

    def test_a_full_log_disk_leaves_the_caller_and_the_console_running(self):
        class FullDisk(io.StringIO):
            def flush(self):
                raise OSError(errno.ENOSPC, "No space left on device")

        with mock.patch("sys.stderr", FullDisk()), mock.patch("sys.stdout", FullDisk()):
            for _ in range(2):
                diagnostics.print_status("Stopping", error=False)
                diagnostics.print_request(
                    {"outcome": "error", "error_code": "invalid_request_error"}
                )
        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            diagnostics.print_status("Ready")
            console_written()
        self.assertRegex(output.getvalue(), r"^\d{2}:\d{2}:\d{2} Ready\n$")

    def test_a_console_that_stops_draining_holds_up_no_caller(self):
        # With a line being written, the next 1024 wait and the 100 after them
        # are dropped and counted; the waiting ones are written, in order,
        # once the console drains, and a warning on stderr says how many were
        # dropped.
        drained = threading.Event()
        console = StalledConsole(drained)
        self.addCleanup(console_written)
        self.addCleanup(drained.set)
        dropped = diagnostics.console_stats()["dropped_lines"]
        kept = diagnostics.CONSOLE_LINES + 1
        printer = threading.Thread(
            target=lambda: [
                diagnostics.print_status(f"line {n}") for n in range(1, kept + 100)
            ]
        )
        with (
            mock.patch("sys.stdout", console),
            mock.patch("sys.stderr", new_callable=io.StringIO) as errors,
        ):
            diagnostics.print_status("line 0")
            self.assertTrue(console.writing.wait(5.0))
            printer.start()
            printer.join(5.0)
            self.assertFalse(printer.is_alive())
            self.assertEqual(
                diagnostics.console_stats()["dropped_lines"] - dropped, 100
            )
            drained.set()
            console_written()
        written = [line.split(" ", 1)[1] for line in console.getvalue().splitlines()]
        self.assertEqual(written, [f"line {n}" for n in range(kept)])
        self.assertRegex(errors.getvalue(), rf"^{dropped_warning(100)}\n$")

    def test_the_exit_writes_the_last_lines_and_its_engines(self):
        # An engine writes its last lines, such as the persistent cache's, as
        # it stops, here once its input ends, while the process exits: the
        # exit writes them, and keeps the process's status.
        engine = "import sys; sys.stdin.read(); sys.stderr.write('Stopped\\n')"
        program = (
            "import subprocess, sys; from server import diagnostics; "
            f"engine = subprocess.Popen([sys.executable, '-c', {engine!r}], "
            "stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0); "
            "diagnostics.relay_engine_lines(engine.stderr); "
            "diagnostics.print_status('Stopping'); "
            "engine.stdin.close(); "
            "raise SystemExit(69)"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=Path(diagnostics.__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 69, result.stderr)
        self.assertRegex(result.stdout, r"^\d{2}:\d{2}:\d{2} Stopping\n$")
        self.assertEqual(result.stderr, "Stopped\n")

    def test_an_exit_on_a_console_that_stopped_draining_takes_its_second(self):
        # Its stdout, or both streams as 2>&1 gives them, on a pipe nobody
        # reads: the exit gives up on its lines after the console's second
        # and keeps the process's status, without -u, as docs/models.md
        # starts the server, as with it.
        program = (
            "from server import diagnostics; "
            "[diagnostics.print_line(f'line {n}') for n in range(10)]; "
            "raise SystemExit(3)"
        )
        for case, options, both in (
            ("stdout", [], False),
            ("2>&1", [], True),
            ("-u 2>&1", ["-u"], True),
        ):
            with self.subTest(case=case):
                status, elapsed, errors = exited(
                    [sys.executable, *options, "-c", program], both
                )
                self.assertEqual(status, 3, errors)
                self.assertLess(elapsed, 5.0)

    def test_an_uncaught_exception_is_a_console_line(self):
        # Python's report of an exception nothing caught, in the main thread,
        # as server.main's of a bug, or in another, comes after the lines
        # printed before it; on a stderr nobody reads, the process still exits
        # with its status after the console's second.
        before = "\n".join(
            (
                "import threading",
                "from server import diagnostics",
                "for n in range(100):",
                "    diagnostics.print_line(f'line {n} ' + 'x' * 1000, error=True)",
            )
        )
        lines = [f"line {n} " + "x" * 1000 for n in range(100)]
        for case, uncaught, status, report in (
            (
                "main",
                "raise RuntimeError('bug')",
                1,
                ("Traceback (most recent call last):", "RuntimeError: bug"),
            ),
            (
                "thread",
                "worker = threading.Thread(target=lambda: 1 / 0, name='worker')\n"
                "worker.start()\n"
                "worker.join()",
                0,
                ("Exception in thread worker:", "ZeroDivisionError: division by zero"),
            ),
        ):
            program = f"{before}\n{uncaught}\n"
            with self.subTest(case=case, stderr="read"):
                result = subprocess.run(
                    [sys.executable, "-u", "-c", program],
                    cwd=Path(diagnostics.__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, status, result.stderr)
                written = result.stderr.splitlines()
                self.assertEqual(written[:100], lines)
                self.assertEqual((written[100], written[-1]), report)
            with self.subTest(case=case, stderr="unread"):
                exit_status, elapsed, _ = exited(
                    [sys.executable, "-u", "-c", program], True
                )
                self.assertEqual(exit_status, status)
                self.assertLess(elapsed, 5.0)

    def test_requests_complete_while_the_console_stops_draining(self):
        harness = self.harness(FakeRuntime(), request_logger=diagnostics.print_request)
        drained = threading.Event()
        console = StalledConsole(drained)
        self.addCleanup(console_written)
        self.addCleanup(drained.set)
        before = diagnostics.console_stats()["dropped_lines"]
        with mock.patch("sys.stdout", console), mock.patch("sys.stderr", console):
            for _ in range(2):
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body()
                )
                self.assertEqual(status, 200, payload)
            # With the first of their lines being written, the console fills
            # up; the next request's Started and Done lines are dropped.
            self.assertTrue(console.writing.wait(5.0))
            for n in range(diagnostics.CONSOLE_LINES):
                diagnostics.print_status(f"line {n}")
            dropped = diagnostics.console_stats()["dropped_lines"]
            status, _, payload = harness.request(
                "POST", "/v1/chat/completions", chat_body()
            )
            self.assertEqual(status, 200, payload)
            status, _, payload = harness.request("GET", "/status")
            self.assertEqual(status, 200)
            self.assertEqual(
                json.loads(payload)["console"]["dropped_lines"] - dropped, 2
            )
            # Once it drains, a warning after the lines that waited, where the
            # dropped ones would have been, says how many there were.
            drained.set()
            console_written()
        count = diagnostics.console_stats()["dropped_lines"] - before
        self.assertRegex(
            console.getvalue().splitlines()[-1], rf"^{dropped_warning(count)}$"
        )

    def test_request_lines_are_the_ones_the_app_reads(self):
        """The server writes its request lines as log_lines_golden.txt holds
        them, in its order, which the app parses (RequestEntry)."""
        golden = [
            line
            for line in (Path(__file__).parent / "log_lines_golden.txt")
            .read_text()
            .splitlines()
            if line and not line.startswith("#")
        ]
        records = [
            {
                "outcome": "started",
                "request_id": 7,
                "prompt_tokens": 4096,
                "cached_tokens": 1024,
                "reasoning": {
                    "sent_effort": "none",
                    "effort": None,
                    "enable_thinking": None,
                    "thinking": False,
                },
            },
            {
                "outcome": "stop",
                "request_id": 7,
                "prompt_tokens": 4096,
                "completion_tokens": 574,
                "metrics": {
                    "cache": {"matched_tokens": 1024},
                    "request_latency": {
                        "ttft_ms": 400,
                        "stream_tokens_per_second": 36.8,
                    },
                    "speculation": {"accepted_draft_tokens": 9, "cycles": 10},
                },
            },
            {
                "outcome": "cancelled",
                "request_id": 8,
                "prompt_tokens": 69,
                "completion_tokens": 12,
                "metrics": {"cache": {"matched_tokens": 0}},
            },
            # A request the engine took, and one refused before it.
            {"outcome": "error", "request_id": 9, "error_code": "runtime_unavailable"},
            {"outcome": "error", "error_code": "invalid_request_error"},
        ]
        self.assertEqual(len(golden), len(records))
        for line, record in zip(golden, records):
            with (
                self.subTest(line=line),
                mock.patch.object(api.time, "strftime", return_value=line.split()[0]),
                mock.patch("sys.stdout", new_callable=io.StringIO) as output,
                mock.patch("sys.stderr", new_callable=io.StringIO) as errors,
            ):
                printed(record)
                self.assertEqual(output.getvalue() + errors.getvalue(), line + "\n")


if __name__ == "__main__":
    unittest.main()

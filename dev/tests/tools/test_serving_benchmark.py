import http.client
import http.server
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from dev.benchmarks.serving import check, clients, run


def synthetic_corpus() -> str:
    return "\n".join(
        f"line {i}: " + "abcdefghij"[i % 10] * (i % 53) for i in range(40000)
    )


def write_jsonl(path: Path, records) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def status(t, rows=0, tokens=0, **fields):
    document = {
        "scheduler": {"prefill_rows": rows, "prefill_batches": 0},
        "metrics": {"decode_output_tokens": tokens, "metal_failures": 0},
        "metal": {"healthy": True},
        "requests": {"failed": 0},
        "thermal_state": "nominal",
        "memory_pressure": "normal",
        "transport": {"restarts": 0},
    }
    for path, value in fields.items():
        node = document
        keys = path.split("__")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return {"t": t, "status": document}


def churn_run(directory: Path, label: str, done, samples, events=(), meta=None):
    """A complete churn run's files: done records (tag, start, t and
    optional chunks_ms, ttft, decode_tps), other `events`, /status samples,
    an end record at 960 s and meta."""
    events = [{"kind": "done", "out": 100, **record} for record in done] + [
        *events,
        {"kind": "end", "t": 960},
    ]
    write_jsonl(directory / f"{label}.events.jsonl", events)
    write_jsonl(directory / f"{label}.status.jsonl", samples)
    (directory / f"{label}.out").write_text("DONE\n")
    (directory / f"{label}.meta.json").write_text(
        json.dumps({"scenario": "churn", "build_id": "b1", **(meta or {})})
    )
    return check.Run(directory, label)


class ClientRequestTests(unittest.TestCase):
    def test_gaps_follow_the_chunks_after_the_first(self):
        start = 1000.0
        chunks = [start + t for t in (1.0, 1.01, 1.03, 2.53, 2.54)]
        summary = clients.gap_summary(chunks, start)
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["max_s"], 1.5)
        self.assertEqual(summary["long"], [[2.53, 1.5]])
        # 10 ms and 20 ms gaps fall in [8, 16) and [16, 32); 1.5 s in [1024, 2048).
        self.assertEqual(summary["hist_ms_log2"][3], 2)
        self.assertEqual(summary["hist_ms_log2"][4], 1)
        self.assertEqual(summary["hist_ms_log2"][10], 1)
        self.assertEqual(sum(summary["hist_ms_log2"]), 4)
        self.assertEqual(clients.gap_summary(chunks[:1], start)["n"], 0)

    def test_idle_ignores_restore_points_waiting_for_a_copy(self):
        idle = {
            "requests": {"submitted": 2, "completed": 2},
            "disk": {"write_behind": {"waiting": 3}},
        }
        self.assertFalse(clients.busy(idle))
        self.assertTrue(
            clients.busy({**idle, "memory_governor": {"owed": {"transfers": 1}}})
        )
        self.assertTrue(clients.busy({**idle, "admission": {"waiting": 1}}))
        self.assertTrue(
            clients.busy({**idle, "requests": {"submitted": 3, "completed": 2}})
        )

    def test_serve1_samples_its_idle_status_and_fails_on_an_error(self):
        # serve1 records the /status each request waited idle for and one
        # after the last, so that G8 sees a restart or a Metal failure; a
        # request the server fails ends it as the server's failure.
        statuses = iter({"n": i} for i in range(10))

        class FakeStream:
            def __init__(self, port, body, timeout):
                self.sent, self.chunks, self.error_body = 0.0, [0.5, 0.6, 0.7], ""
                self.status = 500 if "Log s2048" in body else 200
                self.ok = self.status == 200
                self.usage = {"completion_tokens": 3, "prompt_tokens": 300}

        def sampled(directory, label):
            return [
                json.loads(line)["status"]["n"]
                for line in (Path(directory) / f"{label}.status.jsonl").open()
            ]

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.multiple(
                clients,
                served_model=lambda port: "m",
                wait_idle=lambda port, settle: next(statuses),
                Stream=FakeStream,
            ),
        ):
            clients.run_serve1(1, directory, "ok", samples=2, sizes=(300,))
            self.assertEqual(sampled(directory, "ok"), [0, 1, 2])
            with self.assertRaisesRegex(clients.RequestFailed, "serve1 2048: 500"):
                clients.run_serve1(1, directory, "bad", samples=1, sizes=(300, 2048))
            self.assertEqual(sampled(directory, "bad"), [3, 4])


# A request that fails after its stream began, as the OpenAI dialect's stream
# writes it (server/dialects/openai.py).
FAILED_EVENT = {
    "error": {
        "message": "internal server error",
        "type": "server_error",
        "code": "internal_server_error",
    }
}


def content_event(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


KEEPALIVE = object()
# The server closes the connection there, before the stream's end.
CLOSE = object()


def keepalives_forever():
    while True:
        yield KEEPALIVE


def finished(target, seconds=30):
    """target()'s outcome, ("returned", value) or ("raised", error), or
    ("running", None) if it has not ended within `seconds`."""
    outcome = ["running", None]

    def call():
        try:
            outcome[:] = ["returned", target()]
        except Exception as error:
            outcome[:] = ["raised", error]

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(seconds)
    return tuple(outcome)


class StreamingServer:
    """A server on 127.0.0.1 streaming each chat completion, status 200, as
    server.py does: the role chunk, then the events answer(body) returns
    (KEEPALIVE: server.py's keepalive, 0.05 s after the last write; CLOSE:
    the connection closes there), then [DONE]; a request for whose body
    dropped(body) is true has its connection closed before any response;
    /status what status() returns, idle."""

    def __init__(self, answer, status=dict, dropped=lambda body: False):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def close(self):
                self.connection.shutdown(socket.SHUT_RDWR)
                self.close_connection = True

            def do_GET(self):
                models = {"data": [{"id": "m"}]}
                document = models if self.path == "/v1/models" else status()
                data = json.dumps(document).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"])).decode()
                if dropped(body):
                    self.close()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                role = {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}
                try:
                    self.wfile.write(f"data: {json.dumps(role)}\n\n".encode())
                    for event in answer(body):
                        if event is CLOSE:
                            self.close()
                            return
                        if event is KEEPALIVE:
                            time.sleep(0.05)
                            self.wfile.write(b": splash-keepalive\n\n")
                        else:
                            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client ended the request

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class ErrorEventTests(unittest.TestCase):
    """A request the server fails after the role chunk went out answers 200
    and ends with an error event: every scenario records it as failed."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name

    def events(self, label):
        path = Path(self.directory) / f"{label}.events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_the_stream_records_the_error_event(self):
        with StreamingServer(lambda body: [content_event("1"), FAILED_EVENT]) as fake:
            stream = clients.Stream(fake.port, "{}")
        self.assertEqual((stream.status, stream.ok), (200, False))
        self.assertEqual(json.loads(stream.error_body), FAILED_EVENT["error"])

    def test_churn_records_an_http_error(self):
        # The engine fails each request during its
        # prefill, after the role chunk.
        def failing(body):
            time.sleep(0.1)
            return [FAILED_EVENT]

        with (
            StreamingServer(failing) as fake,
            mock.patch.object(clients, "CHURN_STAGGER_SECONDS", 0),
        ):
            clients.run_churn(
                fake.port, self.directory, "c", 0.01, [0, 1, 2], poll_seconds=0.05
            )
        kinds = [e["kind"] for e in self.events("c")]
        self.assertNotIn("done", kinds)
        errors = [e for e in self.events("c") if e["kind"] == "http_error"]
        self.assertGreaterEqual(len(errors), 3)
        self.assertTrue(
            all(
                e["code"] == 200 and "internal_server_error" in e["body"]
                for e in errors
            )
        )
        self.assertEqual(len(check.Run(Path(self.directory), "c").errors), len(errors))

    def test_ttft_fails_the_round(self):
        # At N=2 the engine refuses one lane's memory claim
        # in round 0 after the role chunk went out.
        usage = {
            "choices": [],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 1900},
            },
        }

        def answer(body):
            if "Round 0." in body and "Conversation 2.1." in body:
                return [FAILED_EVENT]
            return [content_event("OK"), usage]

        with StreamingServer(answer) as fake:
            summary = clients.run_ttft(
                fake.port,
                self.directory,
                "t-n2",
                2,
                rounds=1,
                settle=0,
                poll_seconds=0.05,
                text=synthetic_corpus(),
            )
        self.assertEqual(summary["rounds"], 0)
        (failed,) = [e for e in self.events("t-n2") if e["kind"] == "round_failed"]
        self.assertEqual(len(failed["failures"]), 1)
        self.assertTrue(failed["failures"][0].startswith("200 "))
        self.assertIn("internal_server_error", failed["failures"][0])

    def test_serve1_and_warm_end_on_it(self):
        # serve1's 300-token request fails after its first token; warm's
        # suffix request fails after its role chunk.
        with StreamingServer(lambda body: [content_event("1"), FAILED_EVENT]) as fake:
            with self.assertRaisesRegex(
                clients.RequestFailed, "serve1 300: 200 .*internal_server_error"
            ):
                clients.run_serve1(
                    fake.port, self.directory, "s", samples=1, settle=0, sizes=(300,)
                )

        def answer(body):
            return [FAILED_EVENT] if "Note 0:" in body else [content_event("OK")]

        with StreamingServer(answer) as fake:
            with self.assertRaisesRegex(
                clients.RequestFailed, "warm 300 suffix 0: 200 .*internal_server_error"
            ):
                clients.run_warm(fake.port, self.directory, "w", [300], 0, settle=0)


class TtftRoundTests(unittest.TestCase):
    def test_each_round_records_the_largest_memory_claims_it_saw(self):
        # At N=2, in round 0 the second start waits 0.3 s for memory, a
        # memory-staggered round; round 1's
        # starts take theirs at once. A tree before schema 7 reports none.
        usage = {
            "choices": [],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 1900},
            },
        }
        claims = {"waiting": 0, "bytes": 0, "oldest_ms": 0}

        def answer(body):
            if "Round 0." in body and "Conversation 2.1." in body:
                claims.update(waiting=1, bytes=3 << 30, oldest_ms=250)
                time.sleep(0.3)
                claims.update(waiting=0, bytes=0, oldest_ms=0)
            return [content_event("OK"), usage]

        def rounds(status):
            with (
                tempfile.TemporaryDirectory() as directory,
                StreamingServer(answer, status) as fake,
            ):
                clients.run_ttft(
                    fake.port,
                    directory,
                    "t-n2",
                    2,
                    rounds=2,
                    settle=0,
                    poll_seconds=0.02,
                    text=synthetic_corpus(),
                )
                events = (Path(directory) / "t-n2.events.jsonl").read_text()
            return [
                event["memory_governor.claims"]
                for event in map(json.loads, events.splitlines())
                if event["kind"] == "round"
            ]

        self.assertEqual(
            rounds(lambda: {"memory_governor": {"claims": dict(claims)}}),
            [
                {"waiting": 1, "bytes": 3 << 30, "oldest_ms": 250},
                {"waiting": 0, "bytes": 0, "oldest_ms": 0},
            ],
        )
        self.assertEqual(rounds(dict), [None, None])


class ChurnStatusTests(unittest.TestCase):
    def test_each_sample_also_logs_its_status_event(self):
        # Beside each /status sample a churn run logs
        # the store's chunks, cap, shrinks and regrows, its arena, the decode
        # hit rate, the pressure, the requests and Metal's health as an
        # event.
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": 9000, "completion_tokens": 2},
        }
        document = {
            "streamed_experts": {
                "chunks": 40,
                "cap_chunks": 48,
                "shrinks": 2,
                "regrows": 1,
                "arena_bytes": 1 << 30,
                "decode": {"hit_rate": 0.9},
            },
            "memory_pressure": "warning",
            "requests": {"active": 3, "failed": 0},
            "metal": {"healthy": True},
        }

        def answer(body):
            time.sleep(0.05)
            return [content_event("1"), usage]

        with (
            tempfile.TemporaryDirectory() as directory,
            StreamingServer(answer, lambda: document) as fake,
            mock.patch.object(clients, "CHURN_STAGGER_SECONDS", 0),
        ):
            clients.run_churn(
                fake.port, directory, "c", 0.01, [0, 1, 2], poll_seconds=0.05
            )
            events = (Path(directory) / "c.events.jsonl").read_text().splitlines()
            samples = (Path(directory) / "c.status.jsonl").read_text().splitlines()
        logged = [
            {key: value for key, value in event.items() if key != "t"}
            for event in map(json.loads, events)
            if event["kind"] == "status"
        ]
        self.assertGreater(len(samples), 0)
        self.assertEqual(
            logged,
            [
                {
                    "kind": "status",
                    "chunks": 40,
                    "cap_chunks": 48,
                    "shrinks": 2,
                    "regrows": 1,
                    "arena": 1 << 30,
                    "hit": 0.9,
                    "pressure": "warning",
                    "req": {"active": 3, "failed": 0},
                    "metal_ok": True,
                }
            ]
            * len(samples),
        )


class UnansweredTests(unittest.TestCase):
    """A request the client gets no whole answer to is unanswered, the
    server's failure (G8): one the server keeps open past its time, as
    server.py keeps every stream alive with a keepalive every 2 s and sets
    no deadline of its own, so that a request a scheduler or ledger bug
    starves does not hold the run, and its machine's GPU lock, forever; and
    one whose connection closes before the stream's end."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name

    def events(self, label):
        path = Path(self.directory) / f"{label}.events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def churn(self, label, answer, dropped=lambda body: False, timeout=30):
        """A churn run of 0.01 min against StreamingServer(answer, dropped),
        completed as run.py completes one, and its check."""
        with (
            StreamingServer(answer, dropped=dropped) as fake,
            mock.patch.object(clients, "CHURN_STAGGER_SECONDS", 0),
        ):
            outcome = finished(
                lambda: clients.run_churn(
                    fake.port,
                    self.directory,
                    label,
                    0.01,
                    [0, 1, 2],
                    poll_seconds=0.05,
                    timeout=timeout,
                )
            )
        self.assertEqual(outcome, ("returned", None))
        directory = Path(self.directory)
        (directory / f"{label}.out").write_text("DONE\n")
        (directory / f"{label}.meta.json").write_text(json.dumps({"scenario": "churn"}))
        return self.events(label), check.check_run(check.Run(directory, label))

    def test_keepalives_do_not_hold_a_request_open(self):
        with StreamingServer(lambda body: keepalives_forever()) as fake:
            outcome = finished(lambda: clients.Stream(fake.port, "{}", timeout=0.5), 10)
        self.assertEqual(
            outcome[0], "raised", "keepalives held a request open past its time"
        )
        self.assertIsInstance(outcome[1], clients.Deadline)

    def test_a_stream_cut_before_its_end_raises(self):
        with StreamingServer(lambda body: [content_event("1"), CLOSE]) as fake:
            with self.assertRaisesRegex(ConnectionError, r"before its data: \[DONE\]"):
                clients.Stream(fake.port, "{}")

    def test_churn_records_requests_the_server_keeps_open(self):
        # w1's first request stays queued and w2's stops
        # after its first token while w0 keeps working, and the server keeps
        # both streams alive. Each unanswered record keeps the chunk times its
        # request had, for its gaps.
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": 9000, "completion_tokens": 2},
        }

        def answer(body):
            if "Log 1-" in body:
                yield from keepalives_forever()
            if "Log 2-" in body:
                yield content_event("1")
                yield from keepalives_forever()
            time.sleep(0.2)
            yield from (content_event("1"), content_event("2"), usage)

        events, result = self.churn("c", answer, timeout=1)
        unanswered = {e["tag"]: e for e in events if e["kind"] == "unanswered"}
        self.assertEqual(sorted(unanswered), ["w1-r1", "w2-r1"])
        self.assertTrue(
            all(e["error"].startswith("Deadline: ") for e in unanswered.values())
        )
        self.assertEqual(unanswered["w1-r1"]["chunks_ms"], [])
        self.assertEqual(len(unanswered["w2-r1"]["chunks_ms"]), 1)
        self.assertLess(unanswered["w2-r1"]["chunks_ms"][0], 1000)
        done = {e["tag"].split("-")[0] for e in events if e["kind"] == "done"}
        self.assertEqual(done, {"w0"})
        (g8,) = [f for f in result["failures"] if f.startswith("G8")]
        self.assertTrue(g8.startswith("G8: 2 requests unanswered at the client: "))
        self.assertIn("w1-r1 (Deadline: ", g8)
        self.assertIn("w2-r1 (Deadline: ", g8)

    def test_churn_records_requests_whose_connection_closed(self):
        # w1's second request: its connection closes before any response,
        # or after five tokens, before the stream's end; w0 and w2 work on.
        tokens = [content_event(f"{i}\n") for i in range(30)]
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        }

        def answer(body):
            return [*tokens[:5], CLOSE] if "Log 1-2:" in body else [*tokens, usage]

        for name, dropped, error in (
            ("no-response", lambda body: "Log 1-2:" in body, "RemoteDisconnected: "),
            (
                "cut",
                lambda body: False,
                "ConnectionError: the stream ended before its data: [DONE]",
            ),
        ):
            with self.subTest(name):
                events, result = self.churn(name, answer, dropped)
                (unanswered,) = [e for e in events if e["kind"] == "unanswered"]
                self.assertEqual(unanswered["tag"], "w1-r2")
                self.assertTrue(unanswered["error"].startswith(error), unanswered)
                done = {e["tag"] for e in events if e["kind"] == "done"}
                self.assertEqual(
                    ("w1-r1" in done, "w1-r2" in done, "w0-r2" in done),
                    (True, False, True),
                )
                self.assertIn(
                    f"G8: 1 requests unanswered at the client: w1-r2 ({error}",
                    "\n".join(result["failures"]),
                )

    def test_a_clients_own_exception_is_not_an_unanswered_request(self):
        # The client fails building a request's body, or recording one the
        # server answered whole: each worker ends with a traceback, nothing
        # is recorded unanswered, and the run ends as the client's failure.
        usage = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 2}}

        def answer(body):
            return [content_event("1"), content_event("2"), usage]

        for name, patched in (("body", "churn_body"), ("record", "gap_summary")):
            with (
                self.subTest(name),
                StreamingServer(answer) as fake,
                mock.patch.object(clients, "CHURN_STAGGER_SECONDS", 0),
                mock.patch.object(clients, patched, side_effect=ValueError("bug")),
                mock.patch.object(threading, "excepthook") as excepthook,
            ):
                outcome = finished(
                    lambda: clients.run_churn(
                        fake.port,
                        self.directory,
                        name,
                        0.01,
                        [0, 1, 2],
                        poll_seconds=0.05,
                    )
                )
                self.assertEqual(outcome[0], "raised", outcome)
                self.assertIsInstance(outcome[1], clients.ClientError)
                self.assertEqual(
                    str(outcome[1]), "a churn client failed: ValueError: bug"
                )
                self.assertEqual(
                    [call.args[0].exc_type for call in excepthook.call_args_list],
                    [ValueError] * clients.CHURN_WORKERS,
                )
                kinds = {e["kind"] for e in self.events(name)}
                self.assertFalse(kinds & {"done", "unanswered"}, kinds)

    def test_a_sender_that_fails_before_the_barrier_releases_the_others(self):
        # At N=2 one sender of a round cannot connect; the
        # one that did does not wait at the barrier for it forever.
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 1},
        }
        connect, lock, refused = (
            http.client.HTTPConnection.connect,
            threading.Lock(),
            [],
        )

        def refuse_one_sender(connection):
            with lock:
                refuse = (
                    threading.current_thread().name.endswith("(send)") and not refused
                )
                if refuse:
                    refused.append(threading.current_thread().name)
            if refuse:
                raise ConnectionRefusedError(61, "Connection refused")
            return connect(connection)

        with (
            StreamingServer(lambda body: [content_event("OK"), usage]) as fake,
            mock.patch.object(http.client.HTTPConnection, "connect", refuse_one_sender),
        ):
            outcome = finished(
                lambda: clients.run_ttft(
                    fake.port,
                    self.directory,
                    "t-n2",
                    2,
                    rounds=1,
                    settle=0,
                    poll_seconds=0.05,
                    text=synthetic_corpus(),
                )
            )
        self.assertEqual(outcome[0], "returned", outcome)
        self.assertEqual((len(refused), outcome[1]["rounds"]), (1, 0))
        (failed,) = [e for e in self.events("t-n2") if e["kind"] == "round_failed"]
        self.assertEqual(len(failed["failures"]), 2)


def store(t, size):
    """A sample whose only account, the experts', holds `size` bytes."""
    return status(
        t,
        memory_governor__accounts=[{"name": "experts", "current_bytes": size}],
        memory_governor__charged_bytes=size,
    )


class CheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def served(self, label="r", gaps=(0.05,) * 20, samples=None, ttft=1.0, **files):
        """A churn run in which every worker completed a request whose token
        events came `gaps` seconds apart."""
        chunks = [0.0]
        for gap in gaps:
            chunks.append(chunks[-1] + gap * 1000)
        done = [
            {
                "tag": f"w{w}-r1",
                "start": 10.0,
                "t": 11.0 + chunks[-1] / 1000,
                "ttft": ttft,
                "decode_tps": 20.0,
                "chunks_ms": chunks,
            }
            for w in range(clients.CHURN_WORKERS)
        ]
        samples = samples or [status(t) for t in range(0, 961, 60)]
        return churn_run(self.directory, label, done, samples, **files)

    def test_a_healthy_run_passes(self):
        result = check.check_run(self.served(), minutes=15, build="b1")
        self.assertEqual(result["invalid"] + result["failures"], [])

    def test_g4_fails_a_freeze_and_frequent_stalls(self):
        for name, served in (
            ("freeze", self.served("freeze", gaps=(0.05,) * 500 + (12,))),
            # A request the client stopped waiting for 11 s after its last
            # token.
            (
                "open",
                self.served(
                    "open",
                    gaps=(0.05,) * 500,
                    events=[
                        {
                            "kind": "unanswered",
                            "tag": "w0-r2",
                            "start": 40.0,
                            "t": 52.0,
                            "chunks_ms": [0, 50, 1000],
                            "error": "Deadline: a request still streaming",
                        }
                    ],
                ),
            ),
            ("p99", self.served("p99", gaps=(0.05,) * 90 + (6,) * 10)),
            # Chunks of one token event are one event: they do not dilute
            # the p99 of the gaps between events.
            (
                "events",
                self.served("events", gaps=(0.001,) * 2000 + (0.05,) * 90 + (6,) * 10),
            ),
        ):
            with self.subTest(name):
                self.assertTrue(
                    any(f.startswith("G4") for f in check.check_run(served)["failures"])
                )
        stalled = check.check_run(self.served("stall", gaps=(0.05,) * 200 + (3,)))
        self.assertEqual(stalled["failures"], [])
        self.assertTrue(any("gap of 3.00 s" in note for note in stalled["notes"]))

    def test_g8_fails_the_servers_own_failures(self):
        failing = {
            "restart": [status(0), status(60, transport__restarts=1)],
            "failed": [status(0), status(60, requests__failed=1)],
            "metal": [status(0), status(60, metal__healthy=False)],
            "tick": [status(0), status(60, loop__max_tick_ms=6000)],
        }
        for name, samples in failing.items():
            with self.subTest(name):
                run = self.served(name, samples=samples)
                self.assertTrue(
                    any(f.startswith("G8") for f in check.check_run(run)["failures"])
                )
        exited = self.served("exited", meta={"failure": "server", "error": "exited"})
        (self.directory / "exited.out").write_text("")
        result = check.check_run(exited)
        self.assertEqual(result["invalid"], [])
        self.assertIn("G8: exited", result["failures"])

    def test_g9_fails_a_starved_claim_unsummed_accounts_and_owed_memory(self):
        failing = {
            "starved": [
                store(0, 8),
                status(60, memory_governor__claims__oldest_ms=30_000),
            ],
            "unsummed": [
                store(0, 8),
                status(
                    60,
                    memory_governor__charged_bytes=9,
                    memory_governor__accounts=[{"name": "experts", "current_bytes": 8}],
                ),
            ],
            "owed": [store(0, 8), status(60, memory_governor__owed__bytes=4096)],
        }
        for name, samples in failing.items():
            with self.subTest(name):
                run = self.served(name, samples=samples)
                self.assertTrue(
                    any(f.startswith("G9") for f in check.check_run(run)["failures"])
                )

    def test_a_clients_own_last_request_is_not_in_flight(self):
        # A send is kept to 0.1 s and an answer to 0.01 s, so w0's second
        # request reads as sent at 90.0 s, before its first was answered at
        # 90.04 s; the work ahead of it is still only the 24,576 rows of
        # later arrivals and its own 9,000.
        run = churn_run(
            self.directory,
            "own",
            [
                {
                    "tag": "w0-r1",
                    "start": 0.0,
                    "t": 90.04,
                    "ttft": 15.0,
                    "prompt": 9000,
                    "cached": 0,
                },
                {
                    "tag": "w0-r2",
                    "start": 90.0,
                    "t": 180.0,
                    "ttft": 15.0,
                    "prompt": 9000,
                    "cached": 0,
                },
            ],
            [status(t) for t in range(0, 181, 60)],
        )
        waits = dict(
            (record["tag"], seconds) for record, seconds in check.first_token_waits(run)
        )
        self.assertAlmostEqual(waits["w0-r2"], (24576 + 9000) / (18000 / 180.0))

    def test_a_starved_first_token_fails(self):
        # w0 and w1 each send ten cold 9,000-token prompts, one every 90 s,
        # each answered after 15 s. w2 sends one at 0 s, then at 90 s, beside
        # w0's and w1's in flight, a cold 9,000-token prompt or one of 18,000
        # whose first 17,000 the cache holds. The run prefilled 198,000
        # tokens in 900 s, 220 a second, or with the cache hit 190,000, 211 a
        # second. The work ahead of w2's second request is the 18,000 in
        # flight, 24,576 of later arrivals and its own 9,000, 234 s, or its
        # own 1,000, 206 s: a first token after 450 s, 1.92 times 234 s, is
        # slow, after 490 s, 2.09 times, starved, as 450 s is for the cache
        # hit, 2.18 times 206 s.
        def requests(ttft, prompt, cached):
            return [
                {
                    "tag": f"w{w}-r{r}",
                    "start": 90.0 * (r - 1),
                    "t": 90.0 * r,
                    "ttft": 15.0,
                    "prompt": 9000,
                    "cached": 0,
                }
                for w in (0, 1)
                for r in range(1, 11)
            ] + [
                {
                    "tag": "w2-r1",
                    "start": 0.0,
                    "t": 90.0,
                    "ttft": 15.0,
                    "prompt": 9000,
                    "cached": 0,
                },
                {
                    "tag": "w2-r2",
                    "start": 90.0,
                    "t": 150.0 + ttft,
                    "ttft": ttft,
                    "prompt": prompt,
                    "cached": cached,
                },
            ]

        for name, ttft, prompt, cached, ratio, seconds in (
            ("slow", 450.0, 9000, 0, 1.92, 234),
            ("starved", 490.0, 9000, 0, 2.09, 234),
            ("cache hit", 450.0, 18000, 17000, 2.18, 206),
        ):
            with self.subTest(name):
                run = churn_run(
                    self.directory,
                    name,
                    requests(ttft, prompt, cached),
                    [status(t) for t in range(0, 961, 60)],
                )
                result = check.check_run(run, minutes=15, build="b1")
                self.assertIn(
                    f"first token: the longest wait {ratio:.2f} x the prefill time "
                    f"of the work ahead of it (w2-r2, {ttft:g} s, {seconds} s)",
                    result["notes"],
                )
                self.assertEqual(
                    [f for f in result["failures"] if f.startswith("first token")],
                    []
                    if name == "slow"
                    else [
                        f"first token: w2-r2 waited {ttft:g} s, past 2 x {seconds} s, "
                        "the prefill time of the work ahead of it at the run's rate"
                    ],
                )

    def test_a_store_that_oscillates_fails(self):
        sizes = {0: 8, 60: 10, 80: 8, 100: 10, 120: 8, 960: 8}
        run = self.served("osc", samples=[store(t, s) for t, s in sizes.items()])
        self.assertTrue(
            any(f.startswith("store") for f in check.check_run(run)["failures"])
        )
        # A take back long after the one before is counted, not failed.
        sizes = {0: 8, 60: 10, 80: 8, 400: 10, 420: 8, 960: 8}
        run = self.served("isolated", samples=[store(t, s) for t, s in sizes.items()])
        self.assertEqual(check.check_run(run)["failures"], [])

    def test_gaps_past_the_target_take_their_cause(self):
        # The 1 s gap ends 11.05 s into the run, while a claim waits.
        samples = [status(t) for t in (0, 8, 10, 12, 960)]
        samples[2]["status"]["memory_governor"] = {"claims": {"waiting": 1}}
        run = self.served("cause", gaps=(0.05, 1.0), samples=samples)
        self.assertEqual(check.gap_causes(run), {"claim wait": clients.CHURN_WORKERS})

    def test_runs_that_measured_nothing_are_invalid(self):
        other = self.served("build", meta={"build_id": "b2"})
        self.assertTrue(check.check_run(other, build="b1")["invalid"])
        self.served("cut")
        (self.directory / "cut.out").write_text("")
        self.assertTrue(check.check_run(check.Run(self.directory, "cut"))["invalid"])
        self.assertTrue(
            check.check_run(self.served("resident"), streamed=True)["invalid"]
        )

    def test_only_a_comparison_needs_a_quiet_machine(self):
        # A run's own check notes what made its machine loud, and passes;
        # compare fails it. The pressure run of 11-04 on the M5 Pro: its
        # display kept WindowServer at 36.6 % of a core.
        windowserver = [[410, 36.6, 0.2, "WindowServer", False]]
        busy = {"gpu_idle_check": {"ok": False, "max_percent": 40, "waited_s": 600}}
        # On the M5 Pro: a touch within a sampling period, or a sleep
        # assertion of a process the run did not start other than powerd.
        powerd = "pid 351(powerd): PreventUserIdleSystemSleep named: display on"
        ours = "pid 7(python3): PreventUserIdleSystemSleep named: benchmark"
        cases = {
            "hot": ({"samples": [status(0), status(60, thermal_state="serious")]}, []),
            "display": ({"meta": {"ps": [{"rows": windowserver}]}}, []),
            "busy": ({"meta": busy}, []),
            "touched": ({}, [{"hid_idle_s": 3.0, "holders": [powerd]}]),
            "held": ({}, [{"hid_idle_s": 900.0, "holders": ["pid 8(Music): Prevent"]}]),
            "quiet": ({}, [{"hid_idle_s": 900.0, "holders": [powerd, ours]}]),
        }
        for name, (files, idle) in cases.items():
            with self.subTest(name):
                write_jsonl(self.directory / f"{name}.idle.jsonl", idle)
                meta = {"pids": [7], **files.pop("meta", {})}
                run = self.served(name, meta=meta, **files)
                result = check.check_run(run, minutes=15, build="b1")
                self.assertEqual(result["invalid"] + result["failures"], [])
                noted = [n for n in result["notes"] if n.startswith("not quiet: ")]
                self.assertEqual(bool(noted), name != "quiet", result["notes"])
                self.assertEqual(bool(check.not_quiet(run)), name != "quiet")
        display = self.served(
            "display-a", meta={"build_id": "a", "ps": [{"rows": windowserver}]}
        )
        rounds = [
            self.served(f"q{i}", meta={"build_id": build})
            for i, build in enumerate("abb")
        ]
        problems, _ = check.compare([*rounds, display])
        self.assertEqual(
            problems,
            [
                "display-a: a process this run did not start in the top five "
                "by CPU: WindowServer at 36.6 %"
            ],
        )

    def test_the_cli_exits_1_on_a_failed_run(self):
        self.served(
            "good", samples=[status(t, streamed_experts__cap_bytes=1) for t in (0, 960)]
        )
        self.served("bad", gaps=(12,))
        for label, code in (("good", 0), ("bad", 1)):
            with self.subTest(label), mock.patch("sys.stdout", io.StringIO()) as out:
                argv = [
                    "run",
                    str(self.directory),
                    label,
                    "--minutes",
                    "15",
                    "--streamed",
                ]
                self.assertEqual(check.main(argv), code, out.getvalue())

    def test_compare_judges_two_builds_in_abba_order(self):
        def rounds(name, builds, slower=1.0):
            return [
                self.served(
                    f"{name}{i}",
                    ttft=1.0 * (slower if build == "b" else 1.0),
                    meta={"build_id": build},
                )
                for i, build in enumerate(builds)
            ]

        problems, lines = check.compare(rounds("same", "abba"))
        self.assertEqual(problems, [])
        self.assertTrue(lines)
        problems, _ = check.compare(rounds("slow", "abba", slower=1.2))
        self.assertIn("churn_ttft_s: fail", problems)
        problems, _ = check.compare(rounds("one", "aaaa"))
        self.assertTrue(any("both builds" in p for p in problems))
        problems, _ = check.compare(rounds("mixed", "abab"))
        self.assertTrue(any("A B B A" in p for p in problems))

        def ttft_rounds(name, slower):
            for i, build in enumerate("abba"):
                label = f"{name}{i}"
                factor = slower if build == "b" else 1.0
                requests = [
                    {"kind": "request", "n": 2, "round": r, "conv": c}
                    | {"ttft_ms": (900 + 10 * r + c) * factor}
                    for r in range(3)
                    for c in range(2)
                ]
                write_jsonl(self.directory / f"{label}-n2.events.jsonl", requests)
                write_jsonl(self.directory / f"{label}-n2.status.jsonl", [status(0)])
                (self.directory / f"{label}.out").write_text("DONE\n")
                (self.directory / f"{label}.meta.json").write_text(
                    json.dumps({"scenario": "ttft", "build_id": build})
                )
                yield check.Run(self.directory, label)

        problems, lines = check.compare(list(ttft_rounds("ttft", 1.0)))
        self.assertEqual(problems, [])
        self.assertTrue(any(line.startswith("ttft_n2_ttft_ms") for line in lines))
        problems, _ = check.compare(list(ttft_rounds("ttftslow", 1.2)))
        self.assertIn("ttft_n2_ttft_ms: fail", problems)


class RunTests(unittest.TestCase):
    def test_outputs_resolving_into_models_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "tree" / "install" / "models" / "blob"
            models.mkdir(parents=True)
            (root / "results").symlink_to(models)
            with self.assertRaisesRegex(run.RunError, "install/models"):
                run.refuse_protected(root / "results" / "run1")
            home = root / "home"
            (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
            (root / "cache").symlink_to(home / ".cache" / "huggingface" / "hub")
            with mock.patch.dict(os.environ, {"HOME": str(home)}):
                with self.assertRaisesRegex(run.RunError, "huggingface"):
                    run.refuse_protected(root / "cache")
                with self.assertRaisesRegex(run.RunError, "install/models"):
                    run.main(
                        [
                            "--tree",
                            directory,
                            "--machine",
                            "m6",
                            "--model",
                            "m",
                            "--scenario",
                            "churn",
                            "--label",
                            "x",
                            "--out",
                            str(root / "results"),
                        ]
                    )
            self.assertEqual(
                run.refuse_protected(root / "elsewhere"),
                Path(os.path.realpath(root / "elsewhere")),
            )

    def test_snapshots_are_taken_while_the_server_serves(self):
        # The process snapshots that void a run see its server and engine
        # among the run's own: none before it is ready or after it stopped.
        events = []

        class FakeServer:
            def __init__(self, tree, root, model, port, server_args, log):
                self.process = mock.Mock(pid=4242, **{"poll.return_value": None})

            def wait_ready(self):
                events.append("ready")
                return 1.0

            def engine_pids(self):
                return {4242, 4243}

            def stop(self):
                events.append("stop")
                return {"signals": ["SIGINT"], "exit": 0, "left_running": []}

        def snapshot(ours):
            events.append(frozenset(ours))
            return {"at": 0.0, "rows": [], "therm": ""}

        with tempfile.TemporaryDirectory() as directory:
            tree = Path(directory)
            (tree / "build").mkdir()
            (tree / "build" / "splash").write_bytes(b"splash")
            with (
                mock.patch.multiple(
                    run,
                    Server=FakeServer,
                    ps_snapshot=snapshot,
                    port_answers=lambda port: False,
                    tree_build_id=lambda tree: "src-1",
                    host_memory=lambda: {},
                    check_identity=lambda port, server, build_id: {
                        "pid": 4242,
                        "build_id": build_id,
                        "thermal_state": "nominal",
                    },
                ),
                mock.patch.object(run.clients, "run_serve1"),
            ):
                code = run.main(
                    [
                        "--tree",
                        directory,
                        "--machine",
                        "studio",
                        "--model",
                        "m",
                        "--model-root",
                        directory,
                        "--scenario",
                        "serve1",
                        "--label",
                        "s",
                        "--out",
                        f"{directory}/out",
                    ]
                )
            meta = json.loads((tree / "out" / "s.meta.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual(events[0], "ready")
        self.assertEqual(events[-1], "stop")
        snapshots = events[1:-1]
        self.assertEqual(len(snapshots), 2)
        self.assertTrue(all({4242, 4243} <= ours for ours in snapshots), snapshots)
        self.assertTrue({4242, 4243} <= set(meta["pids"]))

    def test_the_servers_own_failures_are_recorded_as_such(self):
        # A request the server failed, a server gone by the end of the
        # scenario, or anything else that ends a scenario once the server is
        # ready, ends the run as the server's failure, which fails G8, but a
        # client's own failure, the client's; run.py's refusals are recorded
        # by their own kinds.
        exited = {}

        class FakeServer:
            def __init__(self, tree, root, model, port, server_args, log):
                self.process = mock.Mock(pid=4242, returncode=-9)
                self.process.poll = lambda: exited.get("code")

            def wait_ready(self):
                return 1.0

            def engine_pids(self):
                return {4242}

            def stop(self):
                return {"signals": [], "exit": -9, "left_running": []}

        def recorded(label, serve1):
            with tempfile.TemporaryDirectory() as directory:
                tree = Path(directory)
                (tree / "build").mkdir()
                (tree / "build" / "splash").write_bytes(b"splash")
                with (
                    mock.patch.multiple(
                        run,
                        Server=FakeServer,
                        ps_snapshot=lambda ours: {"at": 0.0, "rows": [], "therm": ""},
                        port_answers=lambda port: False,
                        tree_build_id=lambda tree: "src-1",
                        host_memory=lambda: {},
                        check_identity=lambda port, server, build_id: {
                            "pid": 4242,
                            "build_id": build_id,
                            "thermal_state": "nominal",
                        },
                    ),
                    mock.patch.object(run.clients, "run_serve1", side_effect=serve1),
                ):
                    code = run.main(
                        [
                            "--tree",
                            directory,
                            "--machine",
                            "studio",
                            "--model",
                            "m",
                            "--model-root",
                            directory,
                            "--scenario",
                            "serve1",
                            "--label",
                            label,
                            "--out",
                            f"{directory}/out",
                        ]
                    )
                meta = json.loads((tree / "out" / f"{label}.meta.json").read_text())
            return code, meta.get("failure"), meta.get("error")

        def failed_request(*args, **kwargs):
            raise clients.RequestFailed("serve1 14336: 500 engine_execution_failed")

        self.assertEqual(
            recorded("request", failed_request),
            (1, "request", "serve1 14336: 500 engine_execution_failed"),
        )

        def server_gone(*args, **kwargs):
            exited["code"] = -9

        self.assertEqual(
            recorded("gone", server_gone),
            (1, "server", "the server exited -9 during the scenario"),
        )

        def connection_refused(*args, **kwargs):
            raise ConnectionRefusedError(61, "Connection refused")

        code, failure, error = recorded("refused", connection_refused)
        self.assertEqual((code, failure), (1, "server"))
        self.assertTrue(error.startswith("the server exited -9 during the scenario: "))
        # Alive on 127.0.0.1, the server refuses a request,
        # leaves its 14K request's stream to time out, or never goes idle.
        exited.clear()
        self.assertEqual(
            recorded("alive", connection_refused),
            (1, "server", "ConnectionRefusedError: [Errno 61] Connection refused"),
        )

        def wedged(*args, **kwargs):
            raise TimeoutError("timed out")

        self.assertEqual(
            recorded("wedged", wedged), (1, "server", "TimeoutError: timed out")
        )

        # A request it kept streaming past --request-seconds.
        def expired(*args, **kwargs):
            raise clients.Deadline("a request still streaming 2137 s after its send")

        self.assertEqual(
            recorded("expired", expired),
            (1, "server", "Deadline: a request still streaming 2137 s after its send"),
        )

        def owing(*args, **kwargs):
            raise RuntimeError("server did not go idle")

        self.assertEqual(
            recorded("owing", owing),
            (1, "server", "RuntimeError: server did not go idle"),
        )

        def client_failed(*args, **kwargs):
            raise clients.ClientError("a churn client failed: ValueError: bug")

        self.assertEqual(
            recorded("client", client_failed),
            (1, "client", "a churn client failed: ValueError: bug"),
        )

    def test_a_server_must_stop_on_sigint_with_exit_0_and_its_engine_gone(self):
        # The run records each server's stop, and G8 fails one that needed
        # more than SIGINT, did not exit 0 or left its engine running: here, a
        # server that ignores SIGINT, stopped by SIGTERM, and one that exits 0
        # on SIGINT without stopping its engine. Each stands in for its engine
        # with a child.
        engine = "engine = subprocess.Popen(['sleep', '600']); "
        children = {
            "clean": engine + "signal.signal(signal.SIGINT, lambda *_: "
            "(engine.terminate(), engine.wait(), sys.exit(0)))",
            "deaf": engine + "signal.signal(signal.SIGINT, signal.SIG_IGN)",
            "stray": engine + "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))",
            "gone": "sys.exit(3)",
        }
        stops, engines = {}, {}
        with tempfile.TemporaryDirectory() as directory:
            for name, setup in children.items():
                server = run.Server.__new__(run.Server)
                server.log = (Path(directory) / f"{name}.log").open("w")
                server.process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        f"import signal, subprocess, sys, time; {setup}; "
                        "print(engine.pid, flush=True); time.sleep(600)",
                    ],
                    stdout=subprocess.PIPE,
                )
                # Its engine runs and its handler is installed, or it exited.
                started = server.process.stdout.readline()
                if started:
                    engines[name] = int(started)
                else:
                    server.process.wait()
                server.process.stdout.close()
                with mock.patch.object(run, "STOP_GRACE_SECONDS", 1):
                    stops[name] = server.stop()
        self.assertEqual(
            stops,
            {
                "clean": {"signals": ["SIGINT"], "exit": 0, "left_running": []},
                "deaf": {
                    "signals": ["SIGINT", "SIGTERM"],
                    "exit": -15,
                    "left_running": [engines["deaf"]],
                },
                "stray": {
                    "signals": ["SIGINT"],
                    "exit": 0,
                    "left_running": [engines["stray"]],
                },
                "gone": {"signals": [], "exit": 3, "left_running": []},
            },
        )
        # The engines left running were killed.
        deadline = time.monotonic() + 10
        while any(map(run.running, engines.values())) and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertFalse([pid for pid in engines.values() if run.running(pid)])

        def stopped_by(stop):
            class FakeServer:
                def __init__(self, tree, root, model, port, server_args, log):
                    self.process = mock.Mock(pid=4242, **{"poll.return_value": None})

                def wait_ready(self):
                    return 1.0

                def engine_pids(self):
                    return {4242}

                def stop(self):
                    return stop

            return FakeServer

        for name, failed in (
            ("clean", False),
            ("deaf", True),
            ("stray", True),
            ("gone", True),
        ):
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                tree = Path(directory)
                (tree / "build").mkdir()
                (tree / "build" / "splash").write_bytes(b"splash")
                with (
                    mock.patch.multiple(
                        run,
                        Server=stopped_by(stops[name]),
                        ps_snapshot=lambda ours: {"at": 0.0, "rows": [], "therm": ""},
                        port_answers=lambda port: False,
                        tree_build_id=lambda tree: "src-1",
                        host_memory=lambda: {},
                        check_identity=lambda port, server, build_id: {
                            "pid": 4242,
                            "build_id": build_id,
                            "thermal_state": "nominal",
                        },
                    ),
                    mock.patch.object(run.clients, "run_serve1"),
                ):
                    code = run.main(
                        [
                            *("--tree", directory, "--machine", "other"),
                            *("--model", "m", "--model-root", directory),
                            *("--scenario", "serve1", "--label", "s"),
                            *("--out", f"{directory}/out"),
                        ]
                    )
                meta = json.loads((tree / "out" / "s.meta.json").read_text())
                self.assertEqual(code, 0)
                self.assertEqual(meta["parts"][0]["stop"], stops[name])
                result = check.check_run(check.Run(tree / "out", "s"))
                stopped = [f for f in result["failures"] if "did not stop" in f]
                self.assertEqual(bool(stopped), failed, result)

    def test_gpu_utilization_reads_ioreg(self):
        line = (
            '  | "PerformanceStatistics" = {"In use system memory"=1,'
            '"Device Utilization %"=37,"Renderer Utilization %"=12}'
        )
        with mock.patch.object(run, "command_output", return_value=line):
            self.assertEqual(run.gpu_utilization(), 37)

    def test_a_port_that_answers_is_refused(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertTrue(run.port_answers(port))
            with tempfile.TemporaryDirectory() as directory:
                code = run.main(
                    [
                        "--tree",
                        directory,
                        "--machine",
                        "m6",
                        "--model",
                        "m",
                        "--scenario",
                        "churn",
                        "--label",
                        "busy",
                        "--out",
                        directory,
                        "--port",
                        str(port),
                    ]
                )
                self.assertEqual(code, 1)
                self.assertIn(
                    "already answers", (Path(directory) / "busy.out").read_text()
                )
                meta = json.loads((Path(directory) / "busy.meta.json").read_text())
                self.assertEqual((meta["complete"], meta["failure"]), (False, "port"))
        self.assertFalse(run.port_answers(port))


if __name__ == "__main__":
    unittest.main()

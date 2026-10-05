import concurrent.futures
import http.client
import json
import socket
import threading
import time
import unittest
from unittest import mock

from dev.tests.server_fixtures import FakeRuntime, HarnessTestCase, Plan, chat_body
from server import connections, serve_options
from server import server as api


class HttpTransportTests(HarnessTestCase):
    def test_request_body_validation(self):
        harness = self.harness(FakeRuntime())
        for length in (0, -1):
            status, _ = harness.raw_post(b"", length)
            self.assertEqual(status, 400)
        status, _ = harness.raw_post(b"", serve_options.DEFAULT_MAX_REQUEST_BYTES + 1)
        self.assertEqual(status, 413)
        status, _ = harness.raw_post(b"\xff", 1)
        self.assertEqual(status, 400)
        deeply_nested = ("[" * 10000 + "0" + "]" * 10000).encode()
        status, _ = harness.raw_post(deeply_nested, len(deeply_nested))
        self.assertEqual(status, 400)

    def test_huge_numbers_and_nonstandard_json_are_rejected(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        huge = 10**400
        for field in (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "min_p",
            "timeout",
        ):
            with self.subTest(field=field):
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(**{field: huge})
                )
                self.assertEqual(status, 400, payload)
                self.assertIn("error", json.loads(payload))

        nonstandard = json.dumps(chat_body()).replace(
            '"temperature": 0', '"temperature": NaN'
        )
        status, payload = harness.raw_post(
            nonstandard.encode(), len(nonstandard.encode())
        )
        self.assertEqual(status, 400, payload)

        history = [
            {"role": "user", "content": "run"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"x":NaN}'},
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ]
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(messages=history)
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(runtime.requests, [])

    def test_http_media_type_encoding_and_transfer_are_strict(self):
        harness = self.harness(FakeRuntime())
        for headers, expected in (
            ({}, 415),
            ({"Content-Type": "text/plain"}, 415),
            ({"Content-Type": "application/json", "Content-Encoding": "gzip"}, 415),
            ({"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, 400),
        ):
            with self.subTest(headers=headers):
                status, _, _ = harness.request(
                    "POST", "/v1/chat/completions", chat_body(), headers
                )
                self.assertEqual(status, expected)

        status, _, _ = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(reasoning_effort="none"),
            {"Content-Type": "application/vnd.splash+json; charset=utf-8"},
        )
        self.assertEqual(status, 200)

    def test_http_body_is_exact_and_io_has_a_deadline(self):
        harness = self.harness(FakeRuntime(), io_timeout=0.1, timeout=0.3)
        payload = json.dumps(chat_body()).encode()

        digits = str(len(payload))
        for invalid_length in (f"+{digits}", f"{digits[0]}_{digits[1:]}"):
            with self.subTest(content_length=invalid_length):
                connection = socket.create_connection(
                    harness.server.server_address, timeout=2
                )
                connection.sendall(
                    b"POST /v1/chat/completions HTTP/1.1\r\n"
                    b"Host: localhost\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {invalid_length}\r\n\r\n".encode()
                    + payload
                )
                self.assertIn(b" 400 ", connection.recv(4096))
                connection.close()

        connection = socket.create_connection(harness.server.server_address, timeout=2)
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(payload) + 1}\r\n\r\n".encode()
            + payload
        )
        connection.shutdown(socket.SHUT_WR)
        self.assertIn(b" 400 ", connection.recv(4096))
        connection.close()

        connection = socket.create_connection(harness.server.server_address, timeout=2)
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\nContent-Type: application/json\r\n"
            b"Content-Length: 100\r\n\r\n{"
        )
        self.assertIn(b" 408 ", connection.recv(4096))
        connection.close()

        connection = socket.create_connection(harness.server.server_address, timeout=2)
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
        )

        def drip():
            for byte in payload:
                try:
                    connection.sendall(bytes((byte,)))
                except OSError:
                    return
                time.sleep(0.02)

        sender = threading.Thread(target=drip)
        sender.start()
        started = time.monotonic()
        self.assertIn(b" 408 ", connection.recv(4096))
        self.assertLess(time.monotonic() - started, 0.5)
        connection.close()
        sender.join(1)

        connection = socket.create_connection(harness.server.server_address, timeout=2)
        connection.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost:")
        started = time.monotonic()
        self.assertEqual(connection.recv(4096), b"")
        self.assertLess(time.monotonic() - started, 1)
        connection.close()

    def test_control_plane_survives_a_full_client_connection_burst(self):
        harness = self.harness(FakeRuntime())

        def status():
            code, _, payload = harness.request("GET", "/status")
            return code, json.loads(payload)

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            results = list(executor.map(lambda _: status(), range(64)))
        self.assertTrue(all(code == 200 for code, _ in results))
        self.assertTrue(all(payload["ready"] for _, payload in results))
        self.assertGreaterEqual(api.FrontendServer.request_queue_size, 64)

    def test_ingress_rejects_before_reading_body_and_keeps_control_reachable(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, queue_size=1)
        upload = socket.create_connection(harness.server.server_address, timeout=2)
        self.addCleanup(upload.close)
        upload.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n"
        )
        self._wait_for_http_active(harness.server.requests, 1)
        for path in (
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
        ):
            with self.subTest(path=path):
                connection = http.client.HTTPConnection(
                    *harness.server.server_address, timeout=1
                )
                self.addCleanup(connection.close)
                connection.putrequest("POST", path)
                connection.putheader("Content-Type", "application/json")
                connection.putheader(
                    "Content-Length", str(serve_options.DEFAULT_MAX_REQUEST_BYTES)
                )
                connection.endheaders()  # Do not send any body to an overloaded server.
                response = connection.getresponse()
                self.assertEqual(response.status, 503)
                self.assertEqual(response.getheader("Connection"), "close")
                self.assertEqual(
                    json.loads(response.read())["error"]["type"],
                    "overloaded_error" if path == "/v1/messages" else "server_error",
                )
                connection.close()
        self.assertEqual(runtime.requests, [])
        self.assertEqual(harness.tokenizer.templates, [])
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        status, _, payload = harness.request("GET", "/status")
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(payload)["http"]["requests"], {"active": 1, "capacity": 1}
        )
        upload.close()
        self._wait_for_http_active(harness.server.requests, 0)
        self.assertEqual(
            harness.request("POST", "/v1/chat/completions", chat_body())[0], 200
        )

    def test_ingress_slot_covers_stream_and_releases_on_disconnect(self):
        blocking = Plan([[4]], block=True)
        runtime = FakeRuntime(blocking)
        harness = self.harness(runtime, queue_size=1)
        connection, response = harness.open_stream(
            "/v1/chat/completions", chat_body(stream=True)
        )
        self.assertTrue(blocking.started.wait(1))
        self.assertEqual(response.status, 200)
        self.assertEqual(
            harness.request("POST", "/v1/chat/completions", chat_body())[0], 503
        )
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        response.close()
        connection.close()
        self._wait_for_http_active(harness.server.requests, 0)
        self.assertTrue(blocking.cancelled.is_set())
        self.assertEqual(
            harness.request("POST", "/v1/chat/completions", chat_body())[0], 200
        )

    def test_ingress_releases_slot_after_parse_preparation_and_native_errors(self):
        harness = self.harness(FakeRuntime(), queue_size=1)
        self.assertEqual(harness.raw_post(b"{", 1)[0], 400)
        self._wait_for_http_active(harness.server.requests, 0)
        with mock.patch.object(
            harness.app, "prepare", side_effect=RuntimeError("test")
        ):
            with mock.patch.object(api, "log_unexpected"):
                self.assertEqual(
                    harness.request("POST", "/v1/chat/completions", chat_body())[0], 500
                )
        self._wait_for_http_active(harness.server.requests, 0)
        with mock.patch.object(
            harness.backend.runtime,
            "submit",
            side_effect=api.engine_runtime.PendingLimitExceeded("full"),
        ):
            self.assertEqual(
                harness.request("POST", "/v1/chat/completions", chat_body())[0], 503
            )
        self._wait_for_http_active(harness.server.requests, 0)
        self.assertEqual(
            harness.request("POST", "/v1/chat/completions", chat_body())[0], 200
        )

    def _assert_refused(self, connection):
        # A connection given no slot, or losing its slot before its request is
        # read, is answered as the accept answers an excess one, then closed.
        response = http.client.HTTPResponse(connection)
        response.begin()
        self.assertEqual(response.status, 503)
        self.assertEqual(response.getheader("Retry-After"), "1")
        error = json.loads(response.read())["error"]
        self.assertEqual(error["type"], "server_error")
        self.assertEqual(error["code"], "frontend_overloaded")
        self.assertEqual(connection.recv(1), b"")

    def _assert_open(self, connection):
        connection.settimeout(0.2)
        with self.assertRaises(TimeoutError):
            connection.recv(1)

    def _upload_in_progress(self, address):
        upload = socket.create_connection(address, timeout=2)
        self.addCleanup(upload.close)
        upload.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{"
        )
        return upload

    def _stalled(self, address):
        stalled = socket.create_connection(address, timeout=2)
        self.addCleanup(stalled.close)
        stalled.sendall(b"GET /health HTTP/1.1\r\nHost:")
        return stalled

    def _wait_until_read(self, server):
        # Until the thread of every connection with a slot has read what its
        # client sent.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with server.connections.lock:
                held = list(server.connections.holders)
            if not any(connections._has_input(connection) for connection in held):
                return
            time.sleep(0.005)
        self.fail("a connection's input stayed unread")

    def test_stalled_connections_give_their_slots_to_new_ones(self):
        with mock.patch.object(api.FrontendServer, "control_connection_capacity", 2):
            harness = self.harness(FakeRuntime(), queue_size=1)
        address = harness.server.server_address
        upload = self._upload_in_progress(address)
        self._wait_for_http_active(harness.server.requests, 1)
        waiting = [self._stalled(address), self._stalled(address)]
        self._wait_for_http_active(harness.server.connections, 3)
        self._wait_until_read(harness.server)
        # Every slot is taken; the longest waiting stalled one gives way, never
        # the older connection with a request in progress.
        for _ in range(3):
            self.assertEqual(harness.request("GET", "/health")[0], 200)
            self._assert_refused(waiting.pop(0))
            # The answered connection keeps its slot until its thread sees its
            # client close; a stalled one arriving before then would take the
            # slot of the longest waiting one.
            self._wait_for_http_active(harness.server.connections, 2)
            waiting.append(self._stalled(address))
            self._wait_for_http_active(harness.server.connections, 3)
            self._wait_until_read(harness.server)
            self._assert_open(waiting[0])
        self._assert_open(upload)
        self.assertEqual(harness.server.requests.stats()["active"], 1)
        # Before the server closes, so it has no connection to wait for.
        for connection in (upload, *waiting):
            connection.close()

    def test_complete_requests_beyond_capacity_are_answered(self):
        with mock.patch.object(api.FrontendServer, "control_connection_capacity", 0):
            harness = self.harness(FakeRuntime(), queue_size=2)
        address = harness.server.server_address
        # Each connection's thread waits, as one not yet scheduled does, until
        # every client has sent its whole request.
        sent = threading.Event()
        setup = api.FrontendHandler.setup

        def scheduled_late(handler):
            sent.wait(2)
            setup(handler)

        clients = []
        with mock.patch.object(api.FrontendHandler, "setup", scheduled_late):
            for _ in range(8):
                client = socket.create_connection(address, timeout=2)
                self.addCleanup(client.close)
                client.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                clients.append(client)
            # The first two keep their slots, their requests having arrived;
            # the six after them are refused at the accept, and answered
            # although their requests arrived first.
            for client in clients[2:]:
                self._assert_refused(client)
            sent.set()
            for client in clients[:2]:
                response = http.client.HTTPResponse(client)
                response.begin()
                self.assertEqual(response.status, 200)
                response.read()
        for client in clients:
            client.close()

    def test_a_connection_draining_a_refused_upload_gives_its_slot_away(self):
        with mock.patch.object(api.FrontendServer, "control_connection_capacity", 1):
            harness = self.harness(FakeRuntime(), queue_size=1)
        address = harness.server.server_address
        upload = self._upload_in_progress(address)
        self._wait_for_http_active(harness.server.requests, 1)
        refused = socket.create_connection(address, timeout=2)
        self.addCleanup(refused.close)
        refused.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n"
        )
        response = http.client.HTTPResponse(refused)
        response.begin()
        self.assertEqual(response.status, 503)
        response.read()
        # The server half-closes once it waits for the upload to drain.
        self.assertEqual(refused.recv(1), b"")
        self.assertEqual(harness.server.connections.stats()["active"], 2)
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        self._wait_for_http_active(harness.server.connections, 1)
        upload.close()

    def test_requests_in_progress_fill_every_connection_slot(self):
        with mock.patch.object(api.FrontendServer, "control_connection_capacity", 0):
            harness = self.harness(FakeRuntime(), queue_size=3)
        address = harness.server.server_address
        uploads = [self._upload_in_progress(address) for _ in range(3)]
        self._wait_for_http_active(harness.server.requests, 3)
        excess = socket.create_connection(address, timeout=1)
        self.addCleanup(excess.close)
        response = http.client.HTTPResponse(excess)
        response.begin()
        self.assertEqual(response.status, 503)
        error = json.loads(response.read())["error"]
        self.assertEqual(error["type"], "server_error")
        self.assertEqual(error["code"], "frontend_overloaded")
        self.assertEqual(response.getheader("Retry-After"), "1")
        self.assertEqual(harness.server.connections.stats()["active"], 3)
        for upload in uploads:
            self._assert_open(upload)
        uploads[0].close()
        self._wait_for_http_active(harness.server.connections, 2)
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        for upload in uploads:
            upload.close()

    @mock.patch.object(connections, "_has_input", return_value=False)
    def test_connection_slots_close_the_longest_waiting_connection(self, _):
        slots = api.ConnectionSlots(2)
        first, second, third, fourth = (mock.Mock() for _ in range(4))
        self.assertTrue(slots.admit(first))
        self.assertTrue(slots.admit(second))
        self.assertTrue(slots.serving(first))
        self.assertTrue(slots.admit(third))
        # Still awaiting its request, it is answered before it is closed.
        second.send.assert_called_once_with(
            connections.CONNECTION_OVERLOADED_RESPONSE, socket.MSG_DONTWAIT
        )
        second.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        # A connection that lost its slot is not served.
        self.assertFalse(slots.serving(second))
        self.assertTrue(slots.serving(third))
        self.assertFalse(slots.admit(fourth))
        # One draining an upload, which has its response, is last in line
        # and closed without another.
        slots.draining(first)
        slots.expire(third)
        third.shutdown.assert_not_called()
        self.assertTrue(slots.admit(fourth))
        first.send.assert_not_called()
        first.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        # The header timeout closes a connection awaiting its request.
        slots.expire(fourth)
        fourth.send.assert_not_called()
        fourth.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        self.assertEqual(slots.stats(), {"active": 1, "capacity": 2})
        for connection in (first, second, third, fourth):
            slots.release(connection)
        self.assertTrue(slots.idle.is_set())
        self.assertEqual(slots.stats(), {"active": 0, "capacity": 2})

    def _socket_pairs(self, count):
        pairs = [socket.socketpair() for _ in range(count)]
        for pair in pairs:
            for end in pair:
                self.addCleanup(end.close)
        return pairs

    def test_a_connection_whose_request_arrived_keeps_its_slot(self):
        slots = api.ConnectionSlots(2)
        pairs = self._socket_pairs(4)
        (arrived, arrived_client), (idle, idle_client) = pairs[:2]
        (new, new_client), (excess, _) = pairs[2:]
        arrived_client.sendall(b"GET /health HTTP/1.1\r\n\r\n")
        self.assertTrue(slots.admit(arrived))
        self.assertTrue(slots.admit(idle))
        # The idle connection gives way, although it waited less long.
        self.assertTrue(slots.admit(new))
        self.assertFalse(slots.serving(idle))
        self.assertEqual(
            idle_client.recv(65536), connections.CONNECTION_OVERLOADED_RESPONSE
        )
        self.assertEqual(idle_client.recv(1), b"")
        # When every slot has a request, the new connection is refused.
        new_client.sendall(b"GET /health HTTP/1.1\r\n\r\n")
        self.assertFalse(slots.admit(excess))
        self.assertTrue(slots.serving(arrived))
        self.assertTrue(slots.serving(new))

    def test_refused_connections_linger_until_their_clients_close(self):
        closer = api.LingeringCloser(2, 0.5)
        self.addCleanup(closer.stop)
        (answered, answered_client), (silent, silent_client), (excess, _) = (
            self._socket_pairs(3)
        )
        answered.sendall(b"answer")
        closer.close(answered)
        closer.close(silent)
        # Beyond its capacity, the closer closes at once.
        closer.close(excess)
        self.assertEqual(excess.fileno(), -1)
        # Its client reads the answer to its end, and what it sends after
        # it is read and dropped until it closes.
        answered_client.sendall(b"x" * 100_000)
        self.assertEqual(answered_client.recv(64), b"answer")
        self.assertEqual(answered_client.recv(1), b"")
        answered_client.close()
        deadline = time.monotonic() + 0.4
        while answered.fileno() != -1 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(answered.fileno(), -1)
        # A connection whose client never closes is closed after the linger.
        self.assertNotEqual(silent.fileno(), -1)
        self.assertEqual(silent_client.recv(1), b"")
        deadline = time.monotonic() + 1
        while silent.fileno() != -1 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(silent.fileno(), -1)
        closer.stop()
        stopped, _ = self._socket_pairs(1)[0]
        closer.close(stopped)
        self.assertEqual(stopped.fileno(), -1)

    def test_a_burst_waits_in_the_kernel_queue_instead_of_being_reset(self):
        # Nothing accepts while the connections arrive, as when the accept
        # loop falls behind a burst: the kernel holds them all, where a short
        # queue had it reset those beyond it before the server saw them.
        server = api.FrontendServer(("127.0.0.1", 0), None)
        self.addCleanup(server.server_close)
        for _ in range(100):
            client = socket.create_connection(server.server_address, timeout=2)
            self.addCleanup(client.close)
            client.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")

    def test_thread_start_failure_returns_connection_slot(self):
        server = api.FrontendServer(("127.0.0.1", 0), None)
        self.addCleanup(server.server_close)
        client = socket.create_connection(server.server_address, timeout=2)
        self.addCleanup(client.close)
        with (
            mock.patch.object(
                api.ThreadingHTTPServer,
                "process_request",
                side_effect=RuntimeError("test"),
            ),
            mock.patch.object(api.ThreadingHTTPServer, "handle_error") as handle_error,
        ):
            server.handle_request()
        handle_error.assert_called_once()
        self.assertEqual(server.connections.stats()["active"], 0)
        self.assertEqual(client.recv(1), b"")

    def test_http_admission_capacity_is_exact_and_validated(self):
        for invalid in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "HTTP admission capacity"):
                api.HttpAdmission(invalid)
        admission = api.HttpAdmission(1)
        self.assertTrue(admission.acquire())
        self.assertFalse(admission.acquire())
        admission.release()
        self.assertEqual(admission.stats(), {"active": 0, "capacity": 1})
        with self.assertRaisesRegex(RuntimeError, "without acquisition"):
            admission.release()


if __name__ == "__main__":
    unittest.main()

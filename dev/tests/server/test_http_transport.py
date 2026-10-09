import concurrent.futures
import http.client
import json
import queue
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

    def test_stalled_connections_give_their_slots_to_new_ones(self):
        with mock.patch.object(api.FrontendServer, "control_connection_capacity", 2):
            harness = self.harness(FakeRuntime(), queue_size=1)
        address = harness.server.server_address
        upload = self._upload_in_progress(address)
        self._wait_for_http_active(harness.server.requests, 1)
        waiting = [self._stalled(address), self._stalled(address)]
        self._wait_for_http_active(harness.server.connections, 3)
        # Every slot is taken; the longest waiting stalled one gives way, never
        # the older connection with a request in progress.
        for _ in range(3):
            self.assertEqual(harness.request("GET", "/health")[0], 200)
            self._assert_refused(waiting.pop(0))
            # The answered connection keeps its slot until its thread has
            # closed it; a stalled one arriving before then would take the
            # slot of the longest waiting one.
            self._wait_for_http_active(harness.server.connections, 2)
            waiting.append(self._stalled(address))
            self._wait_for_http_active(harness.server.connections, 3)
            self._assert_open(waiting[0])
        self._assert_open(upload)
        self.assertEqual(harness.server.requests.stats()["active"], 1)
        # Before the server closes, so it has no connection to wait for.
        for connection in (upload, *waiting):
            connection.close()

    def test_a_connection_giving_way_before_its_request_reads_its_answer(self):
        body = json.dumps(chat_body()).encode()
        head = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
        )
        # Its client has sent nothing yet, or part of the head.
        for sent in (0, head.index(b"Content-Type")):
            with self.subTest(sent=sent):
                with mock.patch.object(
                    api.FrontendServer, "control_connection_capacity", 0
                ):
                    harness = self.harness(FakeRuntime(), queue_size=1)
                address = harness.server.server_address
                waiting = socket.create_connection(address, timeout=2)
                self.addCleanup(waiting.close)
                waiting.sendall(head[:sent])
                self._wait_for_http_active(harness.server.connections, 1)
                time.sleep(0.05)
                self.assertEqual(harness.request("GET", "/health")[0], 200)
                # The rest of its request arrives only now, the rest of its
                # head and its body in two writes as http.client sends them,
                # the second after a reset would have come back: the answer
                # still arrives whole, and nothing is reset.
                waiting.sendall(head[sent:])
                time.sleep(0.05)
                waiting.sendall(body)
                self._assert_refused(waiting)

    def test_a_connection_whose_head_arrived_keeps_its_slot(self):
        with mock.patch.object(api.FrontendServer, "control_connection_capacity", 0):
            harness = self.harness(FakeRuntime(), queue_size=1)
        address = harness.server.server_address
        body = json.dumps(chat_body()).encode()
        parsed, resume = threading.Event(), threading.Event()
        parse = api.BaseHTTPRequestHandler.parse_request

        def descheduled(handler):
            # Its thread loses the processor between reading the head and
            # acting on it.
            result = parse(handler)
            parsed.set()
            resume.wait(2)
            return result

        with mock.patch.object(
            api.BaseHTTPRequestHandler, "parse_request", descheduled
        ):
            arrived = socket.create_connection(address, timeout=2)
            self.addCleanup(arrived.close)
            # Its head, with its body to follow, as http.client sends them.
            arrived.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
            )
            self.assertTrue(parsed.wait(1))
            # Its request arrived, so a new connection finds no slot to take.
            excess = socket.create_connection(address, timeout=2)
            self.addCleanup(excess.close)
            self._assert_refused(excess)
            resume.set()
            arrived.sendall(body)
            response = http.client.HTTPResponse(arrived)
            response.begin()
            self.assertEqual(response.status, 200)
            response.read()

    def test_a_head_the_waiting_thread_has_yet_to_see_keeps_its_slot(self):
        with mock.patch.object(api.FrontendServer, "control_connection_capacity", 0):
            harness = self.harness(FakeRuntime(), queue_size=1)
        address = harness.server.server_address
        looking, resume = threading.Event(), threading.Event()
        ready = connections._ready

        def delayed(connection):
            # The waiting thread loses the processor before it looks.
            if threading.current_thread() is harness.server.connections.waiting.thread:
                looking.set()
                resume.wait(2)
            return ready(connection)

        with mock.patch.object(connections, "_ready", delayed):
            arrived = socket.create_connection(address, timeout=2)
            self.addCleanup(arrived.close)
            arrived.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
            self.assertTrue(looking.wait(1))
            # Its head arrived, so a new connection finds no slot to take.
            excess = socket.create_connection(address, timeout=2)
            self.addCleanup(excess.close)
            self._assert_refused(excess)
            resume.set()
            response = http.client.HTTPResponse(arrived)
            response.begin()
            self.assertEqual(response.status, 200)
            response.read()

    def test_the_workers_start_with_the_server(self):
        # Threads started while a burst arrives hold off the accept loop, and
        # the kernel resets what overflows its queue meanwhile: a thread per
        # slot starts with the server, and none while it serves.
        harness = self.harness(FakeRuntime())
        workers = [
            thread
            for thread in harness.server.connections.workers.threads
            if thread.is_alive() and thread.name == "connection worker"
        ]
        self.assertEqual(len(workers), harness.server.connections.capacity)
        with mock.patch.object(threading.Thread, "start") as start:
            for _ in range(20):
                self.assertEqual(harness.request("GET", "/health")[0], 200)
        start.assert_not_called()

    def test_the_accept_loop_takes_every_waiting_connection_at_once(self):
        server = api.FrontendServer(("127.0.0.1", 0), None)
        self.addCleanup(server.server_close)
        clients = []
        for _ in range(5):
            client = socket.create_connection(server.server_address, timeout=2)
            self.addCleanup(client.close)
            clients.append(client)
        # One wakeup takes all five, as a burst needs, and returns once the
        # kernel holds none.
        server._handle_request_noblock()
        self.assertEqual(server.connections.stats()["active"], 5)
        server._handle_request_noblock()
        self.assertEqual(server.connections.stats()["active"], 5)

    def test_a_request_head_is_due_by_the_deadline_of_its_accept(self):
        harness = self.harness(FakeRuntime(), io_timeout=1.0)
        address = harness.server.server_address
        silent = socket.create_connection(address, timeout=3)
        self.addCleanup(silent.close)
        late = socket.create_connection(address, timeout=3)
        self.addCleanup(late.close)
        accepted = time.monotonic()
        # Its first bytes come just before the deadline and the rest never:
        # it has only what is left of the time since its accept. Each is
        # closed with what it sent read, so that it sees no reset.
        time.sleep(0.7)
        late.sendall(b"GET /health HTTP/1.1\r\nHost:")
        for connection in (late, silent):
            self.assertEqual(connection.recv(1), b"")
            self.assertLess(time.monotonic() - accepted, 1.4)

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

    @mock.patch.object(connections, "Workers")
    @mock.patch.object(connections, "WaitingConnections")
    def test_connection_slots_close_the_longest_waiting_connection(self, *_):
        slots = api.ConnectionSlots(2, None, 1, 0.5)
        waiting = slots.waiting
        # Connections whose clients have sent nothing yet.
        first, second, third, fourth, refused = (
            mock.Mock(**{"recv.side_effect": BlockingIOError}) for _ in range(5)
        )
        self.assertTrue(slots.admit(first, "first", 10.0))
        waiting.wait.assert_called_once_with(first, "first", 10.0)
        self.assertTrue(slots.admit(second, "second", 10.0))
        # The head of its request has arrived: it keeps its slot.
        self.assertTrue(slots.arrived(first))
        self.assertTrue(slots.admit(third, "third", 10.0))
        # Waiting for its request without a thread, it is answered and closed
        # as a refused connection is.
        second.send.assert_called_once_with(
            connections.CONNECTION_OVERLOADED_RESPONSE, socket.MSG_DONTWAIT
        )
        waiting.close.assert_called_once_with(second)
        second.shutdown.assert_not_called()
        # A connection that lost its slot is neither served nor expired.
        self.assertFalse(slots.arrived(second))
        self.assertFalse(slots.expire(second))
        # When every slot has a request whose head arrived, the new
        # connection is refused, whether or not the waiting thread has looked
        # at the head yet.
        third.recv.side_effect = None
        third.recv.return_value = b"GET /health HTTP/1.1\r\n\r\n"
        self.assertFalse(slots.admit(fourth, "fourth", 10.0))
        self.assertTrue(slots.arrived(third))
        self.assertFalse(slots.admit(fourth, "fourth", 10.0))
        # One draining an upload, which has its response, is last in line,
        # and shut down without another, for its thread to see the end.
        slots.draining(first)
        self.assertTrue(slots.admit(fourth, "fourth", 10.0))
        first.send.assert_not_called()
        first.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        waiting.close.assert_called_once_with(second)
        # The head deadline frees the slot of one still waiting, for the
        # caller to close it.
        self.assertTrue(slots.expire(fourth))
        fourth.shutdown.assert_not_called()
        self.assertEqual(slots.stats(), {"active": 1, "capacity": 2})
        for connection in (first, second, third, fourth):
            slots.release(connection)
        self.assertTrue(slots.idle.is_set())
        self.assertEqual(slots.stats(), {"active": 0, "capacity": 2})
        # One refused at the accept is answered and closed as one that gives
        # way without a thread.
        slots.refuse(refused)
        refused.send.assert_called_once_with(
            connections.CONNECTION_OVERLOADED_RESPONSE, socket.MSG_DONTWAIT
        )
        waiting.close.assert_called_with(refused)

    def _socket_pairs(self, count):
        pairs = [socket.socketpair() for _ in range(count)]
        for pair in pairs:
            for end in pair:
                self.addCleanup(end.close)
        return pairs

    def test_a_connection_whose_request_arrived_keeps_its_slot(self):
        pairs = self._socket_pairs(4)
        (arrived, arrived_client), (idle, idle_client) = pairs[:2]
        (new, new_client), (excess, _) = pairs[2:]
        # Its head arrived, a connection goes to a worker, which here only
        # records it, and keeps its slot.
        served = queue.SimpleQueue()
        slots = api.ConnectionSlots(
            2, lambda connection, _: served.put(connection), 1, 0.5
        )
        self.addCleanup(slots.stop)
        deadline = time.monotonic() + 10
        arrived_client.sendall(b"GET /health HTTP/1.1\r\n\r\n")
        self.assertTrue(slots.admit(arrived, None, deadline))
        self.assertIs(served.get(timeout=1), arrived)
        self.assertTrue(slots.admit(idle, None, deadline))
        # The idle connection gives way, although it waited less long.
        self.assertTrue(slots.admit(new, None, deadline))
        self.assertFalse(slots.arrived(idle))
        self.assertEqual(
            idle_client.recv(65536), connections.CONNECTION_OVERLOADED_RESPONSE
        )
        self.assertEqual(idle_client.recv(1), b"")
        # When every slot has a request whose head arrived, the new connection
        # is refused.
        new_client.sendall(b"GET /health HTTP/1.1\r\n\r\n")
        self.assertIs(served.get(timeout=1), new)
        self.assertFalse(slots.admit(excess, None, deadline))

    def _wait_closed(self, connection, seconds):
        deadline = time.monotonic() + seconds
        while connection.fileno() != -1 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(connection.fileno(), -1)

    def test_refused_connections_linger_until_their_clients_close(self):
        (answered, answered_client), (silent, silent_client) = self._socket_pairs(2)
        waiting = connections.WaitingConnections(None, None, 2, 0.5)
        self.addCleanup(waiting.stop)
        answered.sendall(b"answer")
        waiting.close(answered)
        waiting.close(silent)
        # Its client reads the answer to its end, and what it sends after
        # it is read and dropped until it closes.
        answered_client.sendall(b"x" * 100_000)
        self.assertEqual(answered_client.recv(64), b"answer")
        self.assertEqual(answered_client.recv(1), b"")
        answered_client.close()
        self._wait_closed(answered, 0.4)
        # A connection whose client never closes is closed after the linger.
        self.assertNotEqual(silent.fileno(), -1)
        self.assertEqual(silent_client.recv(1), b"")
        self._wait_closed(silent, 1)
        waiting.stop()
        stopped, _ = self._socket_pairs(1)[0]
        waiting.close(stopped)
        self.assertEqual(stopped.fileno(), -1)

    def test_the_longest_lingering_connection_gives_way(self):
        pairs = self._socket_pairs(3)
        waiting = connections.WaitingConnections(None, None, 2, 5.0)
        self.addCleanup(waiting.stop)
        for connection, client in pairs:
            connection.sendall(b"answer")
            waiting.close(connection)
            # The thread takes it before the next is answered.
            deadline = time.monotonic() + 1
            while waiting.answered and time.monotonic() < deadline:
                time.sleep(0.005)
        # Beyond its capacity, the connection answered first is closed, its
        # client having had the longest to read its answer; the newest one,
        # whose request may still be on its way, lingers.
        (first, first_client), *rest = pairs
        self._wait_closed(first, 1)
        self.assertEqual(first_client.recv(64), b"answer")
        self.assertEqual(first_client.recv(1), b"")
        for connection, client in rest:
            self.assertNotEqual(connection.fileno(), -1)
            client.sendall(b"a request sent late")
        time.sleep(0.05)
        for connection, client in rest:
            self.assertEqual(client.recv(64), b"answer")

    def test_answered_connections_waiting_for_the_thread_are_bounded(self):
        # The thread is busy, here handing a connection to its worker.
        handing, resume = threading.Event(), threading.Event()

        def dispatch(connection, address):
            handing.set()
            resume.wait(2)

        (ready, ready_client), *answered = self._socket_pairs(4)
        waiting = connections.WaitingConnections(mock.Mock(), dispatch, 2, 5.0)
        self.addCleanup(waiting.stop)
        self.addCleanup(resume.set)
        ready_client.sendall(b"GET /health HTTP/1.1\r\n\r\n")
        waiting.wait(ready, None, time.monotonic() + 10)
        self.assertTrue(handing.wait(1))
        for connection, _ in answered:
            waiting.close(connection)
        # Beyond its capacity of answered connections waiting for the busy
        # thread, a connection is closed as it is handed over.
        self.assertEqual(
            [connection.fileno() == -1 for connection, _ in answered],
            [False, False, True],
        )

    def test_a_burst_beyond_capacity_is_answered_within_bounded_descriptors(self):
        with (
            mock.patch.object(api.FrontendServer, "control_connection_capacity", 1),
            mock.patch.object(api.FrontendServer, "refused_connection_capacity", 4),
        ):
            harness = self.harness(FakeRuntime(), queue_size=1)
        address = harness.server.server_address
        # Each connection the server accepts, to count those it holds open:
        # at most its 2 slots, 4 lingering and 4 waiting to linger, besides the
        # one it accepts and one a worker has released and is closing.
        accepted, peak, done = [], 0, threading.Event()
        get_request = api.FrontendServer.get_request

        def recorded(server):
            request = get_request(server)
            accepted.append(request[0])
            return request

        def sample():
            nonlocal peak
            while not done.is_set():
                held = sum(connection.fileno() != -1 for connection in list(accepted))
                peak = max(peak, held)
                time.sleep(0.0005)

        outcomes, go = [], threading.Event()

        def client():
            go.wait(2)
            connection = socket.create_connection(address, timeout=5)
            try:
                connection.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                response = http.client.HTTPResponse(connection)
                response.begin()
                response.read()
                outcomes.append((response.status, response.getheader("Retry-After")))
            except (OSError, http.client.HTTPException) as error:
                outcomes.append(error)
            finally:
                connection.close()

        clients = [threading.Thread(target=client) for _ in range(60)]
        for thread in [*clients, sampler := threading.Thread(target=sample)]:
            thread.start()
        start = threading.Thread.start

        def slow_start(thread):
            # Thread starts slow, as under load.
            time.sleep(0.005)
            start(thread)

        with (
            mock.patch.object(api.FrontendServer, "get_request", recorded),
            mock.patch.object(threading.Thread, "start", slow_start),
        ):
            go.set()
            for thread in clients:
                thread.join(10)
        done.set()
        sampler.join()
        self.assertEqual(len(outcomes), 60)
        for outcome in outcomes:
            self.assertIn(outcome, ((200, None), (503, "1")))
        self.assertLessEqual(peak, 2 + 2 * 4 + 2)
        self._wait_for_http_active(harness.server.connections, 0)

    def test_the_server_raises_its_descriptor_limit(self):
        unlimited = api.resource.RLIM_INFINITY
        for limits, raised in (
            ((256, unlimited), (api.DESCRIPTOR_LIMIT, unlimited)),
            ((256, 4096), (4096, 4096)),
            ((1 << 20, unlimited), None),
        ):
            with (
                self.subTest(limits=limits),
                mock.patch.object(api.resource, "getrlimit", return_value=limits),
                mock.patch.object(api.resource, "setrlimit") as setrlimit,
            ):
                api._raise_descriptor_limit()
                if raised is None:
                    setrlimit.assert_not_called()
                else:
                    setrlimit.assert_called_once_with(
                        api.resource.RLIMIT_NOFILE, raised
                    )

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

    def test_a_connection_whose_serving_fails_returns_its_slot(self):
        server = api.FrontendServer(("127.0.0.1", 0), None)
        self.addCleanup(server.server_close)
        client = socket.create_connection(server.server_address, timeout=2)
        self.addCleanup(client.close)
        # A connection goes to a worker once its client sends or closes.
        client.shutdown(socket.SHUT_WR)
        failed = threading.Event()
        with (
            mock.patch.object(
                api.FrontendServer, "finish_request", side_effect=RuntimeError("test")
            ),
            mock.patch.object(
                api.HTTPServer, "handle_error", side_effect=lambda *_: failed.set()
            ) as handle_error,
        ):
            server.handle_request()
            self.assertTrue(failed.wait(1))
        handle_error.assert_called_once()
        self._wait_for_http_active(server.connections, 0)
        self.assertEqual(client.recv(1), b"")

    def test_stopping_closes_waiting_connections_and_ends_workers(self):
        before = set(threading.enumerate())
        harness = self.harness(FakeRuntime())
        address = harness.server.server_address
        self.assertEqual(harness.request("GET", "/health")[0], 200)
        workers = [
            thread
            for thread in set(threading.enumerate()) - before
            if thread.name == "connection worker"
        ]
        self.assertTrue(workers)
        silent = socket.create_connection(address, timeout=2)
        self.addCleanup(silent.close)
        stalled = self._stalled(address)
        self._wait_for_http_active(harness.server.connections, 2)
        harness.close()
        self.assertEqual(harness.server.connections.stats()["active"], 0)
        # Each is closed without a response, and without a reset.
        for connection in (silent, stalled):
            self.assertEqual(connection.recv(1), b"")
        for worker in workers:
            worker.join(1)
            self.assertFalse(worker.is_alive())

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

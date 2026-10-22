"""The relay of requests for models served elsewhere (server/upstream.py)."""

import argparse
import http.client
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from dev.tests.server_fixtures import (
    HarnessTestCase,
    anthropic_body,
    chat_body,
    responses_body,
)
from server import serve_options, upstream
from server import server as api
from server.errors import APIError

CLOUD_MODEL = "cloud-model-1"
SECRET = "sk-test-cloud-credential"


class Cloud:
    """An upstream API that records each request and answers with
    `answer(handler)`, by default a short event stream."""

    def __init__(self, answer=None):
        self.requests = []
        self.answer = answer or self.stream
        cloud = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                cloud.requests.append((self.path, self.headers, body))
                cloud.answer(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    @staticmethod
    def stream(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("request-id", "req_cloud")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        for event in (b"event: message_start\ndata: {}\n\n", b"event: done\n\n"):
            handler.wfile.write(b"%x\r\n%s\r\n" % (len(event), event))
            handler.wfile.flush()
        handler.wfile.write(b"0\r\n\r\n")

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def post(harness, path, body, headers=(), raw=None):
    connection = http.client.HTTPConnection(*harness.server.server_address, timeout=5)
    data = raw if raw is not None else json.dumps(body).encode()
    connection.request(
        "POST", path, data, {"Content-Type": "application/json", **dict(headers)}
    )
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response, payload


class RelayTests(HarnessTestCase):
    def cloud(self, answer=None):
        cloud = Cloud(answer)
        self.addCleanup(cloud.close)
        return cloud

    def relaying(self, anthropic=None, openai=None, **options):
        return self.harness(
            upstreams=upstream.Upstreams(anthropic, openai),
            served_model_names=["local-alias"],
            **options,
        )

    def test_messages_for_another_model_go_upstream_with_the_clients_credentials(self):
        cloud = self.cloud()
        harness = self.relaying(anthropic=cloud.url + "/")
        raw = json.dumps(anthropic_body(model=CLOUD_MODEL, stream=True)).encode()
        response, payload = post(
            harness,
            "/v1/messages?beta=true",
            None,
            {
                "Authorization": f"Bearer {SECRET}",
                "x-api-key": SECRET,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "interleaved-thinking-2025-05-14",
                "Keep-Alive": "timeout=99",
                "Proxy-Authorization": "Basic cHJveHk6c2VjcmV0",
                "Accept-Encoding": "gzip",
            },
            raw=raw,
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        self.assertEqual(response.getheader("request-id"), "req_cloud")
        self.assertEqual(response.getheader("Via"), "1.1 splash")
        self.assertIsNone(response.getheader("Transfer-Encoding"))
        self.assertEqual(payload, b"event: message_start\ndata: {}\n\nevent: done\n\n")
        [(path, headers, body)] = cloud.requests
        self.assertEqual(path, "/v1/messages?beta=true")
        # The body as the client sent it, and every end-to-end header.
        self.assertEqual(body, raw)
        self.assertEqual(headers["Authorization"], f"Bearer {SECRET}")
        self.assertEqual(headers["x-api-key"], SECRET)
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(headers["anthropic-beta"], "interleaved-thinking-2025-05-14")
        # The hop-by-hop ones are the relay's own.
        self.assertEqual(headers["Host"], cloud.url.removeprefix("http://"))
        self.assertEqual(headers["Accept-Encoding"], "identity")
        self.assertIsNone(headers.get("Keep-Alive"))
        self.assertIsNone(headers.get("Proxy-Authorization"))
        self.assertEqual(
            harness.server.status()["upstreams"]["anthropic"]["relayed"], 1
        )

    def test_count_tokens_for_another_model_go_upstream(self):
        cloud = self.cloud()
        harness = self.relaying(anthropic=cloud.url)
        response, _ = post(
            harness,
            "/v1/messages/count_tokens",
            {"model": CLOUD_MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(cloud.requests[0][0], "/v1/messages/count_tokens")

    def test_openai_paths_go_to_the_openai_base_url_without_its_version(self):
        cloud = self.cloud()
        harness = self.relaying(openai=cloud.url + "/backend-api/codex")
        for path, body in (
            ("/v1/responses", responses_body(model="cloud-model-2")),
            ("/v1/chat/completions", chat_body(model="cloud-model-2")),
            # A path Splash itself does not serve, such as Codex's remote
            # compaction, goes too.
            ("/v1/responses/compact", responses_body(model="cloud-model-2")),
        ):
            with self.subTest(path=path):
                response, _ = post(harness, path, body)
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    cloud.requests[-1][0],
                    "/backend-api/codex" + path.removeprefix("/v1"),
                )

    def test_the_served_model_and_its_aliases_stay_local(self):
        cloud = self.cloud()
        harness = self.relaying(anthropic=cloud.url, openai=cloud.url + "/v1")
        for path, body in (
            ("/v1/messages", anthropic_body()),
            ("/v1/messages", anthropic_body(model="local-alias")),
            ("/v1/chat/completions", chat_body()),
            ("/v1/responses", responses_body(model="local-alias")),
            # Chat and Responses without a model name the served one.
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]}),
        ):
            with self.subTest(path=path, model=body.get("model")):
                response, payload = post(harness, path, body)
                self.assertEqual(response.status, 200, payload)
                self.assertIsNone(response.getheader("Via"))
        self.assertEqual(cloud.requests, [])
        # A path only an upstream serves has nothing for the local model.
        response, _ = post(harness, "/v1/responses/compact", responses_body())
        self.assertEqual(response.status, 404)

    def test_without_an_upstream_other_models_are_not_found(self):
        harness = self.harness()
        for path, body in (
            ("/v1/messages", anthropic_body(model=CLOUD_MODEL)),
            ("/v1/responses", responses_body(model="cloud-model-2")),
            ("/v1/responses/compact", responses_body(model="cloud-model-2")),
        ):
            with self.subTest(path=path):
                response, _ = post(harness, path, body)
                self.assertEqual(response.status, 404)
        self.assertNotIn("upstreams", harness.server.status())

    def test_an_api_without_an_upstream_keeps_its_answers(self):
        cloud = self.cloud()
        harness = self.relaying(anthropic=cloud.url)
        response, _ = post(
            harness, "/v1/responses", responses_body(model="cloud-model-2")
        )
        self.assertEqual(response.status, 404)
        self.assertEqual(cloud.requests, [])

    def test_splash_endpoints_are_never_relayed(self):
        relayed = {
            "/v1/chat/completions": "openai",
            "/v1/completions": "openai",
            "/v1/responses": "openai",
            "/v1/messages": "anthropic",
            "/v1/messages/count_tokens": "anthropic",
        }
        for path, route in api.POST_ROUTES.items():
            with self.subTest(path=path):
                self.assertEqual(route.relayed, path in relayed)
                if route.relayed:
                    self.assertEqual(upstream.api_of(path), relayed[path])
        self.assertEqual(upstream.api_of("/v1/responses/compact"), "openai")
        self.assertIsNone(upstream.api_of("/tokenize"))
        cloud = self.cloud()
        harness = self.relaying(anthropic=cloud.url, openai=cloud.url + "/v1")
        for path in ("/v1/judgments", "/v1/memory/release"):
            with self.subTest(path=path):
                response, _ = post(harness, path, {"model": "cloud-model-2"})
                self.assertNotEqual(response.getheader("Via"), "1.1 splash")
        self.assertEqual(cloud.requests, [])

    def test_upstream_errors_and_redirects_pass_through_unchanged(self):
        def answer(handler):
            status = 429 if handler.path.endswith("messages") else 307
            data = b'{"type":"error","error":{"type":"rate_limit_error"}}'
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("retry-after", "7")
            handler.send_header("Location", "https://elsewhere.example/v1/messages")
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)

        cloud = self.cloud(answer)
        harness = self.relaying(anthropic=cloud.url)
        response, payload = post(harness, "/v1/messages", anthropic_body(model="m"))
        self.assertEqual(response.status, 429)
        self.assertEqual(response.getheader("retry-after"), "7")
        self.assertEqual(
            payload, b'{"type":"error","error":{"type":"rate_limit_error"}}'
        )
        # A redirect returns to the client, and no credential follows it.
        response, _ = post(harness, "/v1/messages/count_tokens", {"model": "m"})
        self.assertEqual(response.status, 307)
        self.assertEqual(len(cloud.requests), 2)

    def test_an_unreachable_upstream_answers_in_the_apis_dialect(self):
        cloud = self.cloud()
        url = cloud.url
        cloud.close()
        harness = self.relaying(anthropic=url, openai=url + "/v1")
        response, payload = post(harness, "/v1/messages", anthropic_body(model="m"))
        self.assertEqual(response.status, 502)
        self.assertEqual(json.loads(payload)["error"]["type"], "api_error")
        self.assertIn("cannot reach 127.0.0.1", json.loads(payload)["error"]["message"])
        response, payload = post(harness, "/v1/responses", responses_body(model="m"))
        self.assertEqual(response.status, 502)
        self.assertEqual(json.loads(payload)["error"]["code"], "upstream_unavailable")
        self.assertEqual(harness.server.status()["upstreams"]["openai"]["failed"], 1)

    def test_a_recovering_engine_refuses_only_local_requests(self):
        cloud = self.cloud()
        harness = self.relaying(anthropic=cloud.url)
        refusal = APIError(
            503, "engine is recovering; retry shortly", "engine_recovering"
        )
        with mock.patch.object(harness.backend, "refusal", return_value=refusal):
            response, _ = post(harness, "/v1/messages", anthropic_body(model="m"))
            self.assertEqual(response.status, 200)
            response, payload = post(harness, "/v1/messages", anthropic_body())
            self.assertEqual(response.status, 503)
            self.assertEqual(json.loads(payload)["error"]["type"], "overloaded_error")

    def test_a_stream_reaches_the_client_as_the_upstream_sends_it(self):
        first_read = threading.Event()

        def answer(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            handler.wfile.write(b"data: first\n\n")
            handler.wfile.flush()
            # The rest only once the client has the first event.
            first_read.wait(5)
            handler.wfile.write(b"data: second\n\n")
            handler.close_connection = True

        cloud = self.cloud(answer)
        harness = self.relaying(openai=cloud.url + "/v1")
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=5
        )
        connection.request(
            "POST",
            "/v1/responses",
            json.dumps(responses_body(model="cloud-model-2", stream=True)),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        self.assertEqual(response.read1(), b"data: first\n\n")
        first_read.set()
        self.assertEqual(response.read(), b"data: second\n\n")
        connection.close()

    def test_no_header_or_body_reaches_the_console(self):
        cloud = self.cloud()
        harness = self.relaying(anthropic=cloud.url)
        with mock.patch.object(api, "print_status") as printed:
            post(
                harness,
                "/v1/messages",
                anthropic_body(model=CLOUD_MODEL, system=SECRET),
                {"x-api-key": SECRET},
            )
        lines = " ".join(str(call) for call in printed.call_args_list)
        self.assertIn("Relayed", lines)
        self.assertIn(CLOUD_MODEL, lines)
        self.assertNotIn(SECRET, lines)


class UpstreamOptionTests(unittest.TestCase):
    def test_upstream_urls_are_https_or_loopback_base_urls(self):
        for value, expected in (
            ("https://api.anthropic.com/", "https://api.anthropic.com"),
            ("https://api.openai.com/v1", "https://api.openai.com/v1"),
            ("http://127.0.0.1:4000", "http://127.0.0.1:4000"),
            ("http://localhost:4000/v1/", "http://localhost:4000/v1"),
            ("http://[::1]:4000", "http://[::1]:4000"),
        ):
            with self.subTest(value=value):
                self.assertEqual(serve_options.parse_upstream_url(value), expected)
        for value in (
            "http://api.anthropic.com",
            "ftp://api.anthropic.com",
            "https://user:pass@api.anthropic.com",
            "https://api.anthropic.com?key=1",
            "https://api.anthropic.com/#x",
            "https://",
            "api.anthropic.com",
            "https://api.anthropic.com:99999",
        ):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    serve_options.parse_upstream_url(value)

    def test_an_api_key_cannot_guard_a_relaying_server(self):
        parser = argparse.ArgumentParser()
        serve_options.add_serve_arguments(parser)
        args = parser.parse_args(
            ["--api-key", "k", "--anthropic-upstream", "https://api.anthropic.com"]
        )
        with (
            mock.patch.object(parser, "exit", side_effect=SystemExit),
            mock.patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            serve_options.check_serve_arguments(parser, args)

    def test_the_launcher_passes_upstreams_to_the_server(self):
        parser = argparse.ArgumentParser()
        serve_options.add_serve_arguments(parser)
        args = parser.parse_args(
            ["--openai-upstream", "https://chatgpt.com/backend-api/codex/"]
        )
        self.assertEqual(
            serve_options.serve_argv(args),
            ["--openai-upstream=https://chatgpt.com/backend-api/codex"],
        )


if __name__ == "__main__":
    unittest.main()

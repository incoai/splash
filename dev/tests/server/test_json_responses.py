import concurrent.futures
import gc
import http.client
import json
import math
import time
import unittest
from unittest import mock

import httpx

from dev.tests.server_fixtures import (
    PATHS,
    FakeRuntime,
    FakeTokenizer,
    Harness,
    HarnessTestCase,
    Plan,
    byte_backend,
    chat_body,
    openai_client,
    request_body,
    response_events,
    responses_body,
    stream_events,
    tool_request,
)
from server import frontend, json_codec
from server import server as api

TEXT = '中文 café e\u0301 🌍 "quoted" C:\\Users\\file\n\r\t\x00 NEL\x85 LS\u2028 PS\u2029 end'


def call_arguments(path, payload, stream):
    """The arguments of the response's first tool call."""
    if not stream:
        response = json.loads(payload)
        if path == PATHS[0]:
            call = response["choices"][0]["message"]["tool_calls"][0]
            return json.loads(call["function"]["arguments"])
        if path == PATHS[1]:
            return json.loads(response["output"][0]["arguments"])
        return response["content"][0]["input"]
    rows = stream_events(payload)
    if path == PATHS[0]:
        fragments = (
            call.get("function", {}).get("arguments", "")
            for row in rows
            for choice in row.get("choices", [])
            for call in choice["delta"].get("tool_calls", [])
        )
    elif path == PATHS[1]:
        fragments = (
            row["delta"]
            for row in rows
            if row["type"] == "response.function_call_arguments.delta"
        )
    else:
        fragments = (
            row["delta"]["partial_json"]
            for row in rows
            if row.get("delta", {}).get("type") == "input_json_delta"
        )
    return json.loads("".join(fragments))


def response_text(path, payload, stream):
    if stream:
        rows = stream_events(payload)
        if path == PATHS[0]:
            return "".join(
                choice["delta"].get("content", "")
                for row in rows
                for choice in row.get("choices", [])
            )
        if path == PATHS[1]:
            return "".join(
                row["delta"]
                for row in rows
                if row["type"] == "response.output_text.delta"
            )
        return "".join(
            row["delta"].get("text", "")
            for row in rows
            if row["type"] == "content_block_delta"
        )
    result = json.loads(payload)
    if path == PATHS[0]:
        return result["choices"][0]["message"]["content"]
    if path == PATHS[1]:
        return "".join(
            part["text"]
            for item in result["output"]
            if item["type"] == "message"
            for part in item["content"]
        )
    return "".join(part["text"] for part in result["content"] if part["type"] == "text")


class JsonResponseTests(unittest.TestCase):
    def harness(self, fragments=None):
        tokenizer = FakeTokenizer()
        if fragments:
            tokenizer.fragments.update(enumerate(fragments, 40))
            tokenizer.backend_tokenizer = byte_backend(tokenizer.fragments)
        harness = Harness(FakeRuntime(), tokenizer=tokenizer, queue_size=32)
        self.addCleanup(harness.close)
        return harness

    def plan(self, harness, count=1):
        harness.backend.runtime.plans.append(
            Plan([[token] for token in range(40, 40 + count)])
        )

    def assert_released(self, harness):
        for thread in harness.backend.runtime.threads:
            thread.join(1)
        # The fake runtime retains completed callbacks for test inspection.
        harness.backend.runtime.calls.clear()
        deadline = time.monotonic() + 2
        while (
            harness.server.request_bodies.stats()["active"]
            and time.monotonic() < deadline
        ):
            gc.collect()
            time.sleep(0.005)
        self.assertEqual(harness.server.request_bodies.stats()["active"], 0)
        self.assertEqual(harness.backend.runtime.pending_count, 0)

    def test_unicode_text_all_protocols_and_stream_decoders(self):
        harness = self.harness([TEXT[:15], TEXT[15:]])
        for path in PATHS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    self.plan(harness, 2)
                    status, _, payload = harness.request(
                        "POST", path, request_body(path, stream)
                    )
                    self.assertEqual(status, 200, payload)
                    self.assertEqual(response_text(path, payload, stream), TEXT)
                    self.assertIn("中文".encode(), payload)
                    self.assertNotIn(b"\\u4e2d", payload)
                    if stream:
                        lines = httpx.Response(status, content=payload).iter_lines()
                        rows = [
                            json.loads(line[6:])
                            for line in lines
                            if line.startswith("data: {")
                        ]
                        self.assertEqual(rows, stream_events(payload))
        self.assert_released(harness)

    def test_openai_sdk_streams_and_content_length(self):
        harness = self.harness([TEXT])
        client = openai_client(harness)
        self.addCleanup(client.close)
        self.plan(harness)
        chunks = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            extra_body={"reasoning_effort": "none"},
        )
        self.assertEqual(
            "".join(
                chunk.choices[0].delta.content or ""
                for chunk in chunks
                if chunk.choices
            ),
            TEXT,
        )
        self.plan(harness)
        events = client.responses.create(
            model="test-model", input="hello", stream=True, reasoning={"effort": "none"}
        )
        self.assertEqual(
            "".join(
                event.delta
                for event in events
                if event.type == "response.output_text.delta"
            ),
            TEXT,
        )
        self.plan(harness)
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        self.addCleanup(connection.close)
        connection.request(
            "POST",
            PATHS[0],
            json.dumps(request_body(PATHS[0])),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = response.read()
        self.assertEqual(response.status, 200)
        self.assertEqual(int(response.getheader("Content-Length")), len(payload))
        self.assertEqual(response_text(PATHS[0], payload, False), TEXT)
        self.assert_released(harness)

    def test_surrogate_model_errors_and_stop_do_not_break_requests(self):
        harness = self.harness()
        for path in PATHS:
            for model in ("\ud800", "\udfff", "中文\ud800"):
                with self.subTest(path=path, model=ascii(model)):
                    body = request_body(path)
                    body["model"] = model
                    status, _, payload = harness.request("POST", path, body)
                    self.assertEqual(status, 404)
                    self.assertIn(model, json.loads(payload)["error"]["message"])
        self.assertEqual(harness.backend.runtime.requests, [])
        body = request_body(PATHS[0])
        body["stop"] = "\ud800"
        self.assertEqual(harness.request("POST", PATHS[0], body)[0], 200)
        self.assert_released(harness)
        self.assertEqual(
            harness.request("POST", PATHS[0], request_body(PATHS[0]))[0], 200
        )

    def test_stored_response_roundtrip_and_atomic_failures(self):
        store = frontend.ResponseStore()
        for value in (TEXT, "\ud800", "\udfff"):
            response = {"id": "resp_test", "value": value}
            history = [{"role": "assistant", "content": value}]
            self.assertTrue(store.put(response, history))
            record = store.get("resp_test")
            self.assertEqual(record.response, response)
            self.assertEqual(json.loads(record.history_json), history)
            self.assertEqual(store.stats()["bytes"], record.size)
            for bad in (math.nan, math.inf, object()):
                with self.assertRaises(json_codec.JSONEncodingError):
                    store.put({"id": "resp_test", "value": bad}, history)
                self.assertEqual(store.get("resp_test").response, response)
                self.assertEqual(store.stats()["bytes"], record.size)
        self.assertTrue(store.delete("resp_test"))
        self.assertEqual(store.stats()["bytes"], 0)

    def test_responses_store_retrieve_continue_and_delete(self):
        harness = self.harness([TEXT])
        self.plan(harness)
        status, _, payload = harness.request("POST", PATHS[1], request_body(PATHS[1]))
        self.assertEqual(status, 200)
        response = json.loads(payload)
        path = PATHS[1] + "/" + response["id"]
        status, _, stored = harness.request("GET", path)
        self.assertEqual((status, json.loads(stored)), (200, response))
        body = request_body(PATHS[1])
        body["previous_response_id"] = response["id"]
        self.assertEqual(harness.request("POST", PATHS[1], body)[0], 200)
        self.assertEqual(harness.request("DELETE", path)[0], 200)
        self.assertEqual(harness.request("GET", path)[0], 404)
        self.assert_released(harness)

    def test_nonfinite_response_failure_is_request_scoped(self):
        harness = self.harness()
        functions = ("completion_response", "responses_response", "anthropic_response")
        for path, function in zip(PATHS, functions, strict=True):
            for stream in (False, True):
                with (
                    self.subTest(path=path, stream=stream),
                    mock.patch.object(api, "log_unexpected", lambda _error: None),
                ):
                    if stream:
                        # Poison the complete event before any part of it is written.
                        original = api.json_codec.encode
                        failed = False

                        def encode(value):
                            nonlocal failed
                            if not failed and isinstance(value, dict):
                                failed = True
                                return original({"value": math.nan})
                            return original(value)

                        context = mock.patch.object(
                            api.json_codec, "encode", side_effect=encode
                        )
                    else:
                        context = mock.patch.object(
                            api, function, return_value={"value": math.nan}
                        )
                    with context:
                        status, _, payload = harness.request(
                            "POST", path, request_body(path, stream)
                        )
                    if stream:
                        self.assertEqual(status, 200)
                        rows = stream_events(payload)
                        self.assertTrue(rows)
                        self.assertIn(
                            "internal_server_error"
                            if path != PATHS[2]
                            else "api_error",
                            str(rows),
                        )
                    else:
                        self.assertEqual(status, 500, payload)
                        self.assertEqual(
                            json.loads(payload)["error"]["message"],
                            "internal server error",
                        )
                    self.assertNotIn(b"NaN", payload)
                    self.assert_released(harness)
                    self.assertEqual(
                        harness.request("POST", path, request_body(path))[0], 200
                    )

    def test_get_and_head_serialization_errors_return_json_500(self):
        harness = self.harness()
        for method in ("GET", "HEAD"):
            with (
                self.subTest(method=method),
                mock.patch.object(
                    harness.server, "status", return_value={"value": math.inf}
                ),
                mock.patch.object(api, "log_unexpected", lambda _error: None),
            ):
                status, content_type, payload = harness.request(method, "/status")
                self.assertEqual(status, 500)
                self.assertEqual(content_type, "application/json")
                if method == "HEAD":
                    self.assertEqual(payload, b"")
                else:
                    self.assertEqual(
                        json.loads(payload)["error"]["code"], "internal_server_error"
                    )
        self.assertEqual(harness.request("GET", "/health")[0], 200)

    def test_malformed_requests_restore_capacity(self):
        harness = self.harness()
        invalid = [b"{", b"null", b"[]", b'{"x":NaN}', b'{"x":Infinity}', b"\xff"] * 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(
                pool.map(lambda data: harness.raw_post(data, len(data))[0], invalid)
            )
        self.assertEqual(set(statuses), {400})
        self.assertEqual(harness.backend.runtime.requests, [])
        self.assert_released(harness)
        self.assertEqual(
            harness.request("POST", PATHS[0], request_body(PATHS[0]))[0], 200
        )

    def test_tool_arguments_roundtrip_across_protocols_and_fragment_boundaries(self):
        nested = {
            "text": TEXT,
            "number": 17,
            "float": 1.25,
            "bool": True,
            "none": None,
            "list": ["🌍", {"quoted": '"\\'}],
        }
        cases = (
            ({"type": "string"}, TEXT, TEXT),
            ({"type": "object"}, json.dumps(nested), nested),
            ({"type": "object"}, json.dumps({"pair": "🌍"}), {"pair": "🌍"}),
            ({"type": "string"}, r"\ud800", r"\ud800"),
            ({"type": "object"}, json.dumps({"text": r"\ud800"}), {"text": r"\ud800"}),
            # A value of any type is the JSON its text spells.
            ({}, json.dumps(nested), nested),
            ({"type": "string"}, TEXT * 400, TEXT * 400),
        )
        for schema, raw, expected in cases:
            output = (
                "<tool_call>\n<function=echo>\n<parameter=value>\n"
                + raw
                + "\n</parameter>\n</function>\n</tool_call>\n"
            )
            # Split model output inside escaped strings as well as tool delimiters.
            width = max(1, len(output) // 7)
            fragments = [output[i : i + width] for i in range(0, len(output), width)]
            harness = self.harness(fragments)
            for path in PATHS:
                canonical = None
                for stream in (False, True):
                    with self.subTest(
                        path=path, stream=stream, size=len(raw), schema=schema
                    ):
                        body = tool_request(path, stream, schema)
                        self.plan(harness, len(fragments))
                        status, _, payload = harness.request("POST", path, body)
                        self.assertEqual(status, 200, payload)
                        if not stream:
                            response = json.loads(payload)
                            if path == PATHS[0]:
                                call = response["choices"][0]["message"]["tool_calls"][
                                    0
                                ]
                                canonical = call["function"]["arguments"]
                            elif path == PATHS[1]:
                                call = next(
                                    item
                                    for item in response["output"]
                                    if item["type"] == "function_call"
                                )
                                canonical = call["arguments"]
                            else:
                                call = next(
                                    item
                                    for item in response["content"]
                                    if item["type"] == "tool_use"
                                )
                                self.assertEqual(call["input"], {"value": expected})
                                canonical = json_codec.dumps(call["input"])
                            self.assertEqual(json.loads(canonical), {"value": expected})
                        else:
                            rows = stream_events(payload)
                            if path == PATHS[0]:
                                arguments = "".join(
                                    call.get("function", {}).get("arguments", "")
                                    for row in rows
                                    for choice in row.get("choices", [])
                                    for call in choice["delta"].get("tool_calls", [])
                                )
                                self.assertEqual(
                                    rows[-1]["choices"][0]["finish_reason"],
                                    "tool_calls",
                                )
                            elif path == PATHS[1]:
                                arguments = "".join(
                                    row["delta"]
                                    for row in rows
                                    if row["type"]
                                    == "response.function_call_arguments.delta"
                                )
                                self.assertEqual(rows[-1]["type"], "response.completed")
                            else:
                                arguments = "".join(
                                    row["delta"]["partial_json"]
                                    for row in rows
                                    if row.get("delta", {}).get("type")
                                    == "input_json_delta"
                                )
                                self.assertEqual(rows[-1]["type"], "message_stop")
                            self.assertEqual(arguments, canonical)
                            self.assertEqual(json.loads(arguments), {"value": expected})
            self.assert_released(harness)

    def test_overflowed_input_numbers_reject_before_inference(self):
        harness = self.harness()
        for path in PATHS:
            for number in ("1e400", "-1e400", "NaN", "Infinity", "-Infinity"):
                with self.subTest(path=path, number=number):
                    body = request_body(path)
                    # Responses retains the original input item for continuation.
                    if path == PATHS[1]:
                        body["input"][0]["extra"] = "NUMBER"
                    else:
                        body["messages"][0]["extra"] = "NUMBER"
                    data = json.dumps(body).replace('"NUMBER"', number).encode()
                    connection = http.client.HTTPConnection(
                        *harness.server.server_address, timeout=3
                    )
                    try:
                        connection.request(
                            "POST", path, data, {"Content-Type": "application/json"}
                        )
                        response = connection.getresponse()
                        payload = response.read()
                        self.assertEqual(response.status, 400, payload)
                    finally:
                        connection.close()
        self.assertEqual(harness.backend.runtime.requests, [])
        self.assert_released(harness)
        self.assertEqual(
            harness.request("POST", PATHS[1], request_body(PATHS[1]))[0], 200
        )

    def test_midstream_encoding_failure_does_not_interrupt_another_request(self):
        harness = self.harness()
        for path in PATHS:
            with self.subTest(path=path):
                plan = Plan([[14], [15]], delay=0.1)
                harness.backend.runtime.plans.append(plan)
                original = json_codec.encode

                def encode(value):
                    if isinstance(value, dict):
                        delta = value.get("delta")
                        content = (
                            delta.get("text") if isinstance(delta, dict) else delta
                        )
                        if value.get("choices"):
                            content = (
                                value["choices"][0].get("delta", {}).get("content")
                            )
                        if content == "second\n":
                            return original({"value": math.nan})
                    return original(value)

                with (
                    mock.patch.object(json_codec, "encode", side_effect=encode),
                    mock.patch.object(api, "log_unexpected", lambda _error: None),
                    concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
                ):
                    failing = pool.submit(
                        harness.request, "POST", path, request_body(path, True)
                    )
                    self.assertTrue(plan.started.wait(1))
                    status, _, normal = harness.request(
                        "POST", PATHS[0], request_body(PATHS[0])
                    )
                    self.assertEqual(status, 200)
                    self.assertEqual(
                        response_text(PATHS[0], normal, False), "plain answer\n"
                    )
                    status, _, payload = failing.result(timeout=3)
                self.assertEqual(status, 200)
                self.assertEqual(response_text(path, payload, True), "first ")
                self.assertIn("internal server error", str(stream_events(payload)))
                self.assert_released(harness)
                self.assertEqual(
                    harness.request("POST", path, request_body(path))[0], 200
                )

    def test_unicode_that_cannot_be_written_back_keeps_its_text(self):
        # JSON that decodes to a lone surrogate would not survive the next
        # turn, so the value keeps the text the model wrote.
        value = r'{"nested":[{"\ud800":"bad"}]}'
        fragments = [
            "<tool_call>\n<function=echo>\n<parameter=value>\n",
            value,
            "\n</parameter>\n</function>\n</tool_call>\n",
        ]
        harness = self.harness(fragments)
        for path in PATHS:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    plan = Plan([[40], [41], [42]], delay=0.05)
                    harness.backend.runtime.plans.append(plan)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        calling = pool.submit(
                            harness.request,
                            "POST",
                            path,
                            tool_request(path, stream, {"type": "object"}),
                        )
                        self.assertTrue(plan.started.wait(1))
                        status, _, normal = harness.request(
                            "POST", PATHS[0], request_body(PATHS[0])
                        )
                        self.assertEqual(status, 200)
                        self.assertEqual(
                            response_text(PATHS[0], normal, False), "plain answer\n"
                        )
                        status, _, payload = calling.result(timeout=3)
                    self.assertEqual(status, 200, payload)
                    self.assertEqual(
                        call_arguments(path, payload, stream), {"value": value}
                    )
                    self.assert_released(harness)
                    self.assertEqual(harness.request("GET", "/health")[0], 200)
                    self.assertEqual(
                        harness.request("POST", path, request_body(path))[0], 200
                    )


class UnicodeWireTests(HarnessTestCase):
    def test_unicode_output_stays_utf8_on_the_wire(self):
        tokenizer = FakeTokenizer()
        tokenizer.fragments[40] = (
            "Hello 你好世界 こんにちは世界 안녕하세요 세계 Café 🌍\n"
        )
        tokenizer.fragments[41] = (
            "<tool_call>\n<function=echo>\n<parameter=text>\n"
            "你好世界\n</parameter>\n</function>\n</tool_call>\n"
        )
        tokenizer.backend_tokenizer = byte_backend(tokenizer.fragments)
        runtime = FakeRuntime(
            Plan([[40]]), Plan([[40]]), Plan([[40]]), Plan([[41]]), Plan([[41]])
        )
        harness = self.harness(runtime, tokenizer=tokenizer)
        text = tokenizer.fragments[40]
        wire_text = json.dumps(text, ensure_ascii=False)[1:-1].encode()
        tools = [{"type": "function", "function": {"name": "echo"}}]

        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(reasoning_effort="none")
        )
        self.assertEqual(status, 200)
        self.assertIn(wire_text, payload)
        self.assertNotIn(b"\\u4f60", payload)
        self.assertNotIn(b"\\ud83c", payload)
        self.assertEqual(json.loads(payload)["choices"][0]["message"]["content"], text)

        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(stream=True, reasoning_effort="none"),
        )
        self.assertEqual(status, 200)
        self.assertIn("🌍".encode(), payload)
        self.assertNotIn(b"\\ud83c", payload)
        chunks = [
            json.loads(line[6:])
            for line in payload.decode().splitlines()
            if line.startswith("data: {")
        ]
        streamed = "".join(
            chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
        )
        self.assertEqual(streamed, text)

        status, _, payload = harness.request(
            "POST",
            "/v1/responses",
            responses_body(stream=True, reasoning={"effort": "none"}),
        )
        self.assertEqual(status, 200)
        self.assertIn("🌍".encode(), payload)
        self.assertNotIn(b"\\ud83c", payload)
        events = response_events(payload)
        streamed = "".join(
            event["delta"]
            for event in events
            if event["type"] == "response.output_text.delta"
        )
        self.assertEqual(streamed, text)

        for stream in (False, True):
            with self.subTest(stream=stream):
                status, _, payload = harness.request(
                    "POST",
                    "/v1/chat/completions",
                    chat_body(tools=tools, stream=stream, reasoning_effort="none"),
                )
                self.assertEqual(status, 200)
                self.assertIn("你好世界".encode(), payload)
                self.assertNotIn(b"\\u4f60", payload)
                if stream:
                    chunks = [
                        json.loads(line[6:])
                        for line in payload.decode().splitlines()
                        if line.startswith("data: {")
                    ]
                    arguments = "".join(
                        delta["function"].get("arguments", "")
                        for chunk in chunks
                        for delta in chunk["choices"][0]["delta"].get("tool_calls", [])
                    )
                else:
                    message = json.loads(payload)["choices"][0]["message"]
                    arguments = message["tool_calls"][0]["function"]["arguments"]
                self.assertEqual(arguments, '{"text":"你好世界"}')

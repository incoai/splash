import copy
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from dev.benchmarks import qualify_flash_http as qualification

MODEL = "local/test-flash"
USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 4,
    "total_tokens": 14,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 0},
}
METRICS = {
    "cache": {"matched_tokens": 0, "status": "miss"},
    "prefill": {"tokens": 10},
    "request_latency": {
        "ttft_ms": 12,
        "start_to_first_token_ms": 12,
        "first_token_to_done_ms": 20,
        "wall_ms": 32,
        "stream_tokens_per_second": 150,
    },
}


def chunk(delta=None, *, finish=None, usage=None, identifier="req-1", model=MODEL):
    value = {
        "id": identifier,
        "model": model,
        "object": "chat.completion.chunk",
        "created": 1,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }
    if usage is not None:
        value.update(
            choices=[], usage=copy.deepcopy(usage), metrics=copy.deepcopy(METRICS)
        )
    return value


def sse_lines(events):
    lines = [(0, b": keepalive\r\n"), (0, b"\r\n")]
    for index, event in enumerate(events, 1):
        data = event if isinstance(event, bytes) else json.dumps(event).encode()
        lines.extend(((index * 10, b"data: " + data + b"\r\n"), (index * 10, b"\r\n")))
    return lines


def successful_stream(text="42"):
    return sse_lines(
        [
            chunk({"role": "assistant", "content": ""}),
            chunk({"content": text}),
            chunk(finish="stop"),
            chunk(usage=USAGE),
            b"[DONE]",
        ]
    )


def response_bytes(content="42", *, tools=None, usage=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tools is not None:
        message["tool_calls"] = tools
    return json.dumps(
        {
            "id": "req-1",
            "model": MODEL,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": copy.deepcopy(USAGE if usage is None else usage),
            "metrics": copy.deepcopy(METRICS),
        }
    ).encode()


def healthy_status():
    return {
        "schema_version": 5,
        "ready": True,
        "maximum_context_tokens": 32768,
        "metal": {"healthy": True},
        "memory_audit": {"valid": True},
        "memory_pressure": "normal",
        "identity": {
            "source": "a" * 64,
            "loaded_model_layout_sha256": "b" * 64,
            "engine_instance_id": 1,
            "forward_semantics": "fixture-forward",
            "execution": "native_metal_autoregression",
        },
        "capabilities": {
            "native_route": "flash-next",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "tools": True,
            "structured_output": True,
            "mtp": False,
            "prefix_cache": False,
        },
        "transport": {
            "ready": True,
            "status_stale": False,
            "recovering": False,
            "restarts": 0,
            "pending": 0,
        },
        "scheduler": {
            "queued": 0,
            "active_requests": 0,
            "prefilling": 0,
            "decoding": 0,
            "waiting_mask": 0,
            "command_in_flight": False,
        },
        "requests": {"submitted": 4, "completed": 4, "cancelled": 0, "failed": 0},
    }


class FlashHTTPQualificationTests(unittest.TestCase):
    def setUp(self):
        network = mock.patch(
            "http.client.HTTPConnection",
            side_effect=AssertionError(
                "qualification fixtures must not contact a server"
            ),
        )
        network.start()
        self.addCleanup(network.stop)

    def check_text(self, record, expected="42", budget=64):
        return qualification.check_answer(
            record, {"kind": "text", "value": expected}, budget
        )

    def parsed_stream(self, lines):
        record = qualification.parse_sse(lines)
        record.update(
            http_status=200, content_type="text/event-stream", expected_model=MODEL
        )
        return record

    def test_cli_accepts_loopback_and_rejects_shared_or_remote_targets(self):
        for url in (
            "http://127.0.0.1:8011",
            "http://localhost:8011",
            "http://[::1]:8011",
        ):
            with self.subTest(url=url):
                qualification.parse_args(["--base-url", url])
        for url in (
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://[::1]:8000",
            "http://example.com:8011",
            "http://localhost.example.com:8011",
        ):
            with self.subTest(url=url), self.assertRaises(SystemExit):
                qualification.parse_args(["--base-url", url])

    def test_plan_is_deterministic_and_has_distinct_concurrent_answers(self):
        plan = qualification.build_plan(MODEL, "fixture-nonce")
        self.assertEqual(plan, qualification.build_plan(MODEL, "fixture-nonce"))
        cases = {row["id"]: row for row in plan}
        self.assertEqual(len(cases), len(plan))
        self.assertIn("structured_system_developer", cases)
        lanes = cases["concurrent_four"]["lanes"]
        self.assertEqual(len(lanes), 4)
        values = [
            json.dumps(lane["expected"]["value"], sort_keys=True) for lane in lanes
        ]
        self.assertEqual(len(set(values)), 4)
        call = cases["tool_call"]["expected"]
        self.assertEqual(call["name"], "add_numbers")
        self.assertEqual(call["arguments"], {"a": 19, "b": 23})
        self.assertEqual(cases["tool_continuation"]["depends_on"], "tool_call")
        self.assertEqual(cases["tool_continuation"]["expected"]["value"], "42")
        structured = cases["structured_system_developer"]["body"]
        self.assertEqual(
            [message["role"] for message in structured["messages"][:2]],
            ["system", "developer"],
        )
        self.assertEqual(structured["response_format"]["type"], "json_schema")
        streaming = cases["stream_arithmetic"]["body"]
        self.assertTrue(streaming["stream"])
        self.assertTrue(streaming["stream_options"]["include_usage"])
        self.assertEqual(
            cases["tool_call"]["body"]["tool_choice"],
            {"type": "function", "function": {"name": "add_numbers"}},
        )
        bodies = [row["body"] for row in plan if "body" in row]
        bodies.extend(lane["body"] for lane in lanes)
        for body in bodies:
            self.assertEqual(body["model"], MODEL)
            self.assertEqual(body["reasoning_effort"], "none")
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["seed"], 0)
        self.assertEqual(
            len({json.dumps(lane["body"], sort_keys=True) for lane in lanes}), 4
        )

    def test_stream_success_ignores_keepalive_and_role_only_frames(self):
        record = self.parsed_stream(successful_stream())
        self.assertEqual(self.check_text(record), [])
        self.assertEqual(record["text"], "42")
        self.assertTrue(record["done"])
        self.assertEqual(record["first_content_ms"], 20)
        self.assertEqual(set(record["request_ids"]), {"req-1"})
        self.assertEqual(set(record["model_ids"]), {MODEL})

    def test_stream_error_followed_by_done_is_not_success(self):
        lines = sse_lines(
            [
                chunk({"role": "assistant", "content": ""}),
                chunk({"content": "42"}),
                chunk(finish="stop"),
                chunk(usage=USAGE),
                {"error": {"code": "constraint_error", "message": "failed"}},
                b"[DONE]",
            ]
        )
        record = self.parsed_stream(lines)
        self.assertTrue(record["done"])
        self.assertTrue(record["error_frames"])
        self.assertTrue(self.check_text(record))

    def test_missing_stream_done_finish_or_usage_is_rejected(self):
        events = [
            chunk({"content": "42"}),
            chunk(finish="stop"),
            chunk(usage=USAGE),
            b"[DONE]",
        ]
        for missing in (1, 2, 3):
            with self.subTest(missing=missing):
                record = self.parsed_stream(
                    sse_lines(
                        [
                            event
                            for index, event in enumerate(events)
                            if index != missing
                        ]
                    )
                )
                self.assertTrue(self.check_text(record))

    def test_stream_identity_changes_are_rejected(self):
        for changed in (
            chunk({"content": "2"}, identifier="req-2"),
            chunk({"content": "2"}, model="local/other"),
        ):
            with self.subTest(changed=changed):
                record = self.parsed_stream(
                    sse_lines(
                        [
                            chunk({"content": "4"}),
                            changed,
                            chunk(finish="stop"),
                            chunk(usage=USAGE),
                            b"[DONE]",
                        ]
                    )
                )
                self.assertTrue(self.check_text(record))

    def test_malformed_sse_json_is_preserved_as_failure(self):
        record = self.parsed_stream(sse_lines([b"{not-json", b"[DONE]"]))
        self.assertTrue(record["errors"])
        self.assertTrue(self.check_text(record))

    def test_json_valid_malformed_sse_shapes_record_errors_and_raw_events(self):
        events = []
        for choices in (None, {}, "choices", [1]):
            events.append({**chunk(), "choices": choices})
        for delta in (None, "delta", 1):
            event = chunk()
            event["choices"][0]["delta"] = delta
            events.append(event)
        for calls in ({}, "calls", [None], [1]):
            events.append(chunk({"tool_calls": calls}))
        call = {
            "index": 0,
            "id": "call-1",
            "type": "function",
            "function": {"name": "add_numbers", "arguments": '{"a":19,"b":23}'},
        }
        for field in ("id", "name", "arguments"):
            for value in (None, 1, {}):
                broken = copy.deepcopy(call)
                target = broken if field == "id" else broken["function"]
                target[field] = value
                events.append(chunk({"tool_calls": [broken]}))
        for index, event in enumerate(events):
            with self.subTest(index=index, event=event):
                record = self.parsed_stream(sse_lines([event, b"[DONE]"]))
                self.assertTrue(record["errors"])
                self.assertEqual(record["events"][0]["data"], event)

    def test_json_valid_malformed_responses_record_errors_and_raw_response(self):
        base = json.loads(response_bytes())
        documents = []
        for choices in (None, {}, "choices", [1]):
            documents.append({**copy.deepcopy(base), "choices": choices})
        for message in (None, "message", 1):
            document = copy.deepcopy(base)
            document["choices"][0]["message"] = message
            documents.append(document)
        for content in (1, True, ["42"]):
            document = copy.deepcopy(base)
            document["choices"][0]["message"]["content"] = content
            documents.append(document)
        for calls in ({}, "calls", [None], [1]):
            document = copy.deepcopy(base)
            document["choices"][0]["message"]["tool_calls"] = calls
            documents.append(document)
        call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "add_numbers", "arguments": '{"a":19,"b":23}'},
        }
        for field in ("id", "name", "arguments"):
            for value in (None, 1, {}):
                broken = copy.deepcopy(call)
                target = broken if field == "id" else broken["function"]
                target[field] = value
                document = copy.deepcopy(base)
                document["choices"][0]["message"]["tool_calls"] = [broken]
                documents.append(document)
        for index, document in enumerate(documents):
            with self.subTest(index=index, document=document):
                record = qualification.normalize_response(
                    200, "application/json", json.dumps(document).encode()
                )
                self.assertTrue(record["errors"])
                self.assertEqual(record["response"], document)

    def test_nonstream_response_has_exact_arithmetic_oracle(self):
        record = qualification.normalize_response(
            200, "application/json", response_bytes()
        )
        self.assertEqual(self.check_text(record), [])
        self.assertTrue(self.check_text(record, expected="43"))

    def test_json_oracle_preserves_types_and_rejects_duplicates_and_nan(self):
        expected = {"kind": "json", "value": {"answer": 1}}
        for text, valid in (
            ('{"answer":1}', True),
            ('{"answer":true}', False),
            ('{"answer":1.0}', False),
            ('{"answer":1,"answer":1}', False),
            ('{"answer":NaN}', False),
        ):
            with self.subTest(text=text):
                record = qualification.normalize_response(
                    200, "application/json", response_bytes(text)
                )
                errors = qualification.check_answer(record, expected, 64)
                self.assertEqual(not errors, valid)

    def test_usage_accounting_and_output_budget_fail_closed(self):
        for field, value in (
            ("completion_tokens", 0),
            ("completion_tokens", True),
            ("completion_tokens", 65),
            ("total_tokens", 99),
            ("prompt_tokens", 0),
        ):
            with self.subTest(field=field, value=value):
                usage = {**copy.deepcopy(USAGE), field: value}
                record = qualification.normalize_response(
                    200, "application/json", response_bytes(usage=usage)
                )
                self.assertTrue(self.check_text(record))

    def test_required_tool_call_validates_name_arguments_and_call_id(self):
        expected = {
            "kind": "tool",
            "name": "add_numbers",
            "arguments": {"a": 19, "b": 23},
        }
        call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "add_numbers", "arguments": '{"a":19,"b":23}'},
        }
        record = qualification.normalize_response(
            200,
            "application/json",
            response_bytes("", tools=[call], finish="tool_calls"),
        )
        self.assertEqual(qualification.check_answer(record, expected, 128), [])
        for failure in ("name", "arguments", "id", "duplicate"):
            with self.subTest(failure=failure):
                calls = [copy.deepcopy(call)]
                if failure == "name":
                    calls[0]["function"]["name"] = "other"
                elif failure == "arguments":
                    calls[0]["function"]["arguments"] = '{"a":19,"b":24}'
                elif failure == "id":
                    calls[0]["id"] = ""
                else:
                    calls.append(copy.deepcopy(call))
                record = qualification.normalize_response(
                    200,
                    "application/json",
                    response_bytes("", tools=calls, finish="tool_calls"),
                )
                self.assertTrue(qualification.check_answer(record, expected, 128))

    def test_status_rejects_staleness_and_changed_identity(self):
        status = healthy_status()
        self.assertEqual(qualification.validate_status(status, status["identity"]), [])
        for field, value in (("status_stale", True), ("ready", False)):
            with self.subTest(field=field):
                broken = copy.deepcopy(status)
                broken["transport"][field] = value
                self.assertTrue(
                    qualification.validate_status(broken, status["identity"])
                )
        changed = copy.deepcopy(status)
        changed["identity"]["source"] = "changed-source"
        self.assertTrue(qualification.validate_status(changed, status["identity"]))

    def test_wait_idle_rejects_restart_with_unchanged_native_identity(self):
        initial = healthy_status()
        restarted = copy.deepcopy(initial)
        restarted["transport"]["restarts"] = 1
        runner = object.__new__(qualification.Qualification)
        runner.args = SimpleNamespace(cleanup_timeout=1)
        runner.initial, runner.identity = initial, initial["identity"]
        runner.client = mock.Mock()
        runner.client.status.return_value = restarted
        with self.assertRaisesRegex(ValueError, "restart"):
            runner.wait_idle()

    def test_wait_idle_observes_busy_cleanup_before_accepting_fresh_idle(self):
        initial = healthy_status()
        busy = copy.deepcopy(initial)
        busy["scheduler"]["active_requests"] = 1
        runner = object.__new__(qualification.Qualification)
        runner.args = SimpleNamespace(cleanup_timeout=1)
        runner.initial, runner.identity = initial, initial["identity"]
        runner.client = mock.Mock()
        runner.client.status.side_effect = [busy, copy.deepcopy(initial)]
        with (
            mock.patch.object(qualification.time, "monotonic", side_effect=[0, 0.1]),
            mock.patch.object(qualification.time, "sleep") as sleep,
        ):
            result = runner.wait_idle()
        self.assertEqual(result, initial)
        self.assertEqual(runner.client.status.call_count, 2)
        sleep.assert_called_once_with(0.25)

    def test_counter_delta_isolated_request_changes_do_not_mutate_snapshots(self):
        before = healthy_status()
        after = copy.deepcopy(before)
        after["requests"]["submitted"] += 1
        after["requests"]["cancelled"] += 1
        original = copy.deepcopy(before)
        deltas = qualification.counter_delta(before, after)
        self.assertEqual(deltas["requests.submitted"], 1)
        self.assertEqual(deltas["requests.cancelled"], 1)
        self.assertEqual(deltas["requests.completed"], 0)
        self.assertEqual(before, original)


if __name__ == "__main__":
    unittest.main()

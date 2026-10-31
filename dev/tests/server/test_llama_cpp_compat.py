"""--llama-cpp-compat: llama.cpp's /slots and /props probes, the lane map
behind their rows, and the id_slot stamped on every generation response.

Ported from llama-cm's test-splash-slots-shim.sh and
test-splash-lane-degrade.sh, which drove the same translation and lane rules
from the child's side; here they run against the real routes, the real status
document and the real generation paths."""

import json
import time
import unittest
from unittest import mock

from dev.tests.server_fixtures import (
    FakeRuntime,
    HarnessTestCase,
    Plan,
    anthropic_body,
    chat_body,
    response_events,
    responses_body,
    stream_events,
)
from server import llama_cpp_compat as compat
from server import protocol as wire
from server import server as api

# A status document of a ready engine with one request decoding, the shape a
# live multi-lane child reports (its fixture in llama-cm).
DECODING = {
    "id": 172,
    "phase": "decode",
    "priority": "normal",
    "prompt_tokens": 40000,
    "prompt_processed": 40000,
    "generated_tokens": 4321,
    "max_new_tokens": 32000,
    "age_ms": 89744.11521,
}
QUEUED = {
    "id": 173,
    "phase": "queued",
    "priority": "normal",
    "prompt_tokens": 100000,
    "prompt_processed": 0,
    "generated_tokens": 0,
    "max_new_tokens": 32000,
    "age_ms": 12000.5,
}
TWO = [DECODING, QUEUED]
MEMORY_PLAN = {"model": {"model_name": "/models/Qwen.gguf"}}
INSTANCE = {"model": "local/Qwen"}


def status_document(active_requests, window=262144):
    return {
        "schema_version": wire.STATUS_SCHEMA_VERSION,
        "ready": True,
        "maximum_context_tokens": window,
        "memory_plan": MEMORY_PLAN,
        "instance": INSTANCE,
        "active_requests": active_requests,
    }


def slots(status, lanes=None, queue_size=1):
    return json.loads(compat.slots_body(status, lanes, queue_size))["slots"]


class SlotsBodyTests(unittest.TestCase):
    def test_one_active_request_yields_a_row_of_its_own_lane(self):
        status = status_document([DECODING])
        row = slots(status)[0]
        # The row id is the LANE index, never Splash's request id: a proxy
        # joins /slots rows to sessions by slot id, and a request id like 172
        # would size its slot table 173 wide and join nothing.
        self.assertEqual(row["id"], 0)
        self.assertNotEqual(DECODING["id"], 0)
        self.assertEqual(row["n_ctx"], status["maximum_context_tokens"])
        self.assertIs(row["is_processing"], True)
        self.assertEqual(row["n_prompt_tokens"], DECODING["prompt_tokens"])
        self.assertEqual(row["n_prompt_tokens_processed"], DECODING["prompt_processed"])
        self.assertEqual(row["n_prompt_tokens_cache"], 0)
        self.assertEqual(
            row["next_token"], [{"n_decoded": DECODING["generated_tokens"]}]
        )

    def test_an_idle_status_yields_one_idle_row_per_lane(self):
        status = status_document([])
        rows = slots(status)
        self.assertEqual(len(rows), 1)
        self.assertIs(rows[0]["is_processing"], False)
        self.assertEqual(rows[0]["n_ctx"], status["maximum_context_tokens"])
        rows = slots(status, None, 2)
        self.assertEqual([row["id"] for row in rows], [0, 1])
        self.assertFalse(any(row["is_processing"] for row in rows))

    def test_a_queued_request_is_not_a_processing_slot(self):
        # A proxy's prefill-stall guard cuts any processing slot whose
        # n_prompt_tokens_processed stays flat; an orderly wait for a lane is
        # not a frozen prefill. The row stays, so n_ctx keeps the window.
        status = status_document([QUEUED])
        row = slots(status)[0]
        self.assertIs(row["is_processing"], False)
        self.assertEqual(row["n_ctx"], status["maximum_context_tokens"])
        self.assertEqual(row["id"], 0)
        self.assertEqual(row["n_prompt_tokens"], QUEUED["prompt_tokens"])
        self.assertEqual(row["n_prompt_tokens_processed"], 0)
        self.assertEqual(row["next_token"], [])

    def test_a_live_prefill_reports_its_progress(self):
        prefill = {
            "id": 7,
            "phase": "prefill",
            "priority": "foreground",
            "prompt_tokens": 35000,
            "prompt_processed": 12000,
            "generated_tokens": 0,
            "max_new_tokens": 32000,
            "age_ms": 1500.5,
        }
        row = slots(status_document([prefill]))[0]
        self.assertEqual(row["id"], 0)
        self.assertIs(row["is_processing"], True)
        self.assertEqual(row["n_prompt_tokens_processed"], 12000)
        self.assertEqual(row["n_prompt_tokens"], 35000)
        self.assertEqual(row["next_token"], [])
        self.assertEqual(row["n_ctx"], 262144)

    def test_no_lane_sidecar(self):
        # Context, phase and rates come from the proxy's own fold of /slots;
        # the server writes no lane file of its own.
        self.assertFalse(
            any(
                hasattr(compat, name)
                for name in ("LaneSampler", "write_lane_file", "lane_file_path")
            )
        )

    def test_props_reports_the_window_the_slots_and_the_model(self):
        props = json.loads(compat.props_body(status_document([DECODING])))
        self.assertEqual(props["total_slots"], 1)
        self.assertEqual(props["default_generation_settings"]["n_ctx"], 262144)
        self.assertEqual(props["model_alias"], "local/Qwen")
        self.assertEqual(props["model_path"], "/models/Qwen.gguf")
        self.assertIn(
            f"schema_version={wire.STATUS_SCHEMA_VERSION}", props["build_info"]
        )
        two = json.loads(compat.props_body(status_document([DECODING]), 2))
        self.assertEqual(two["total_slots"], 2)

    def test_two_requests_are_keyed_by_their_lanes_not_their_position(self):
        status = status_document(TWO)
        rows = slots(status, {172: 0, 173: 1}, 2)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(set(by_id), {0, 1})
        self.assertEqual(by_id[0]["n_prompt_tokens"], DECODING["prompt_tokens"])
        self.assertEqual(
            by_id[0]["n_prompt_tokens_processed"], DECODING["prompt_processed"]
        )
        self.assertIs(by_id[0]["is_processing"], True)
        self.assertEqual(by_id[0]["next_token"], [{"n_decoded": 4321}])
        self.assertIs(by_id[1]["is_processing"], False)
        self.assertEqual(by_id[1]["n_ctx"], 262144)
        self.assertEqual(by_id[1]["n_prompt_tokens"], QUEUED["prompt_tokens"])
        self.assertEqual(by_id[1]["n_prompt_tokens_processed"], 0)
        self.assertEqual(by_id[1]["next_token"], [])

        # The order flip: the status lists requests in the scheduler's order,
        # which can change between polls; the lane map keeps each request on
        # its own row.
        flipped = status_document(list(reversed(TWO)))
        rows = slots(flipped, {172: 0, 173: 1}, 2)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(set(by_id), {0, 1})
        self.assertEqual(by_id[0]["n_prompt_tokens"], DECODING["prompt_tokens"])
        self.assertEqual(by_id[1]["n_prompt_tokens"], QUEUED["prompt_tokens"])

    def test_a_finished_request_lane_is_reused_by_the_next(self):
        status = status_document(
            [QUEUED, {**DECODING, "id": 174, "phase": "prefill", "prompt_tokens": 500}]
        )
        rows = slots(status, {173: 1, 174: 0}, 2)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(set(by_id), {0, 1})
        self.assertEqual(by_id[0]["n_prompt_tokens"], 500)
        self.assertEqual(by_id[1]["n_prompt_tokens"], QUEUED["prompt_tokens"])

    def test_an_unmapped_request_gets_no_row(self):
        rows = slots(status_document(TWO), {172: 0}, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], 0)

    def test_without_a_lane_map_rows_follow_list_position(self):
        # A one-lane child, or a read before anything was mapped: the old
        # single row at id 0, and never a second row past the queue size.
        rows = slots(status_document(TWO))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], 0)
        rows = slots(status_document(TWO), None, 2)
        self.assertEqual(sorted(row["id"] for row in rows), [0, 1])
        self.assertIs(rows[0]["is_processing"], True)
        self.assertIs(rows[1]["is_processing"], False)


class LaneMapTests(unittest.TestCase):
    def test_lanes_follow_the_proxy_not_the_engine(self):
        lanes = compat.LaneMap(2)
        self.assertEqual(lanes.snapshot(), {})
        lanes.note_slot(172, 1)  # the proxy asked for lane 1
        lanes.note_start(172, 0)  # the engine started it on lane 0
        self.assertEqual(lanes.snapshot(), {172: 1})
        lanes.note_slot(173, 0)
        lanes.note_start(173, 1)
        self.assertEqual(lanes.snapshot(), {172: 1, 173: 0})
        self.assertEqual(lanes.stamp_id_slot(172), 1)
        self.assertIsNone(lanes.stamp_id_slot(999))
        lanes.note_done(172)
        self.assertEqual(lanes.snapshot(), {173: 0})
        # The freed lane is the next request's, but 172's own response still
        # stamps the lane it held (the engine's terminal record runs before
        # the response is built). A request that LOST its lane to a newer
        # one stamps nothing (test_a_reassigned_lane_belongs_to_the_newest).
        self.assertEqual(lanes.stamp_id_slot(172), 1)
        lanes.note_slot(174, None)
        self.assertEqual(lanes.snapshot()[174], 1)
        lanes.note_slot(175, 0)  # both lanes taken; the proxy's cap bounds this
        self.assertEqual(lanes.snapshot()[175], 0)
        lanes.note_slot(176, None)  # no id_slot and no free lane: no row
        self.assertNotIn(176, lanes.snapshot())
        for request_id in (173, 174, 175, 176):
            lanes.note_done(request_id)
        self.assertEqual(lanes.snapshot(), {})
        # A request that never held a lane stamps nothing.
        self.assertIsNone(lanes.stamp_id_slot(176))

    def test_a_reassigned_lane_belongs_to_the_newest_request(self):
        lanes = compat.LaneMap(1)
        lanes.note_slot(1, 0)
        lanes.note_slot(2, 0)
        self.assertEqual(lanes.snapshot(), {2: 0})

    def test_queue_size_zero_maps_nothing(self):
        lanes = compat.LaneMap(0)
        lanes.note_slot(1, 0)
        self.assertEqual(lanes.snapshot(), {})

    def test_an_engine_lane_mismatch_is_logged_once(self):
        console = []
        lanes = compat.LaneMap(2, log=console.append)
        lanes.note_slot(172, 0)
        lanes.note_start(172, 1)
        lanes.note_start(172, 1)
        self.assertEqual(len(console), 1)
        self.assertIn("proxy lane 0", console[0])
        self.assertIn("engine lane 1", console[0])
        self.assertIn("172", console[0])

    def test_lane_of_reads_only_an_integer_id_slot(self):
        for body, lane in (
            ({"id_slot": 3}, 3),
            ({"id_slot": 0}, 0),
            ({}, None),
            ({"id_slot": None}, None),
            ({"id_slot": "1"}, None),
            ({"id_slot": True}, None),
            ({"id_slot": 1.0}, None),
        ):
            with self.subTest(body=body):
                self.assertEqual(compat.lane_of(body), lane)


class CompatRouteTests(HarnessTestCase):
    def harness(self, runtime=None, **kwargs):
        kwargs.setdefault("llama_cpp_compat", True)
        return super().harness(runtime, **kwargs)

    def slot_rows(self, harness):
        status, _, payload = harness.request("GET", "/slots")
        self.assertEqual(status, 200)
        return json.loads(payload)["slots"]

    def test_slots_and_props_answer_from_the_status_document(self):
        harness = self.harness()
        rows = self.slot_rows(harness)
        # An idle child answers one idle row per lane, so the window stays
        # readable whatever the proxy's slot table looks like.
        self.assertEqual([row["id"] for row in rows], [0, 1, 2, 3])
        for row in rows:
            self.assertIs(row["is_processing"], False)
            self.assertEqual(row["n_ctx"], 128)  # the harness' max_context
        status, _, payload = harness.request("GET", "/props")
        self.assertEqual(status, 200)
        props = json.loads(payload)
        self.assertEqual(props["total_slots"], 4)  # the harness' queue_size
        self.assertEqual(props["default_generation_settings"]["n_ctx"], 128)
        self.assertEqual(props["model_path"], "test-model")
        self.assertEqual(props["model_alias"], "test-model")

    def test_probes_stay_quiet_in_the_console(self):
        harness = self.harness()
        with (
            mock.patch.object(harness.app, "log_request") as logged,
            mock.patch("sys.stderr") as console,
        ):
            for _ in range(3):
                self.assertEqual(harness.request("GET", "/slots")[0], 200)
                self.assertEqual(harness.request("GET", "/props")[0], 200)
        self.assertEqual(console.write.call_count, 0)
        self.assertEqual(logged.call_count, 0)

    def test_a_generation_holds_its_lane_and_stamps_it_back(self):
        blocked = Plan([[4]], block=True)
        runtime = FakeRuntime(blocked)
        harness = self.harness(runtime)
        connection, response = harness.open_stream(
            "/v1/chat/completions", chat_body(stream=True, id_slot=1)
        )
        # The stream opens at the engine's start, so the first SSE frame
        # proves the request reached the engine and holds its lane.
        frame = self.read_frame(response)
        # The fake engine answers a fixed status document with no
        # active_requests, so the live row is checked against the lane map
        # (what the rows are keyed by) rather than a per-request row.
        self.assertEqual(harness.app.lane_map.snapshot(), {1: 1})
        # The engine's lane (the fake always starts on 0) is only a
        # cross-check; the proxy's lane is what /slots reports.
        self.assertEqual(
            json.loads(frame.decode().removeprefix("data: "))["id_slot"], 1
        )
        # The lane is free again once the engine reports the request done.
        blocked.release.set()
        response.read()
        connection.close()
        self.wait_for_lane_free(harness, 1)

    def test_a_request_without_id_slot_takes_the_lowest_free_lane(self):
        harness = self.harness()
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body()
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["id_slot"], 0)

    def test_every_stream_kind_carries_the_lane(self):
        cases = (
            ("/v1/chat/completions", chat_body(stream=True, id_slot=2)),
            (
                "/v1/completions",
                {
                    "model": "test-model",
                    "prompt": "hello",
                    "stream": True,
                    "id_slot": 2,
                },
            ),
            (
                "/v1/responses",
                responses_body(stream=True, reasoning={"effort": "none"}, id_slot=2),
            ),
            (
                "/v1/messages",
                anthropic_body(stream=True, thinking={"type": "disabled"}, id_slot=2),
            ),
        )
        for path, body in cases:
            with self.subTest(path=path):
                harness = self.harness()
                _, response = harness.open_stream(path, body)
                payload = response.read()
                if path == "/v1/messages":
                    started = [
                        event
                        for event in response_events(payload)
                        if event["type"] == "message_start"
                    ]
                    self.assertEqual(started[0]["message"]["id_slot"], 2)
                elif path == "/v1/responses":
                    created = [
                        event
                        for event in response_events(payload)
                        if event["type"] == "response.created"
                    ]
                    self.assertEqual(created[0]["response"]["id_slot"], 2)
                else:
                    chunks = stream_events(payload)
                    self.assertTrue(chunks)
                    self.assertTrue(all(chunk["id_slot"] == 2 for chunk in chunks))

    def test_a_lane_frees_on_an_engine_error(self):
        runtime = FakeRuntime(Plan([[4]], error="the grammar refused the output"))
        harness = self.harness(runtime)
        status, _, _ = harness.request(
            "POST", "/v1/chat/completions", chat_body(id_slot=1)
        )
        self.assertEqual(status, 400)
        self.wait_for_idle(harness)

    def read_frame(self, response):
        # One SSE frame, read byte-wise: the stream opens at the engine's
        # start, so the first frame proves the request reached the engine.
        frame = b""
        while not frame.endswith(b"\n\n"):
            byte = response.read(1)
            if not byte:
                self.fail("the stream closed before its first frame")
            frame += byte
        return frame

    def test_a_lane_frees_when_the_client_leaves(self):
        # SSE_KEEPALIVE_SECONDS bounds how long a stream waits between two
        # disconnect polls; shorten it so the test does not wait on the
        # production interval.
        with mock.patch.object(api, "SSE_KEEPALIVE_SECONDS", 0.1):
            blocked = Plan([[4]], block=True)
            runtime = FakeRuntime(blocked)
            harness = self.harness(runtime)
            connection, response = harness.open_stream(
                "/v1/chat/completions", chat_body(stream=True, id_slot=1)
            )
            self.read_frame(response)
            connection.close()
            # The server notices the closed client on its disconnect poll and
            # cancels the job itself; the lane frees when the engine reports
            # the cancellation, exactly as the proxy's own cancel would.
            self.wait_for_lane_free(harness, 1)
        self.assertEqual({row["id"] for row in self.slot_rows(harness)}, {0, 1, 2, 3})

    def wait_for_lane_free(self, harness, lane):
        # Generous: a lane the client left frees only once the server's
        # disconnect poll notices it, which runs on the stream's own cadence.
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if lane not in harness.app.lane_map.snapshot().values():
                return
            time.sleep(0.02)
        self.fail(f"lane {lane} stayed held")

    def wait_for_idle(self, harness):
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if not harness.app.lane_map.snapshot():
                return
            time.sleep(0.02)
        self.fail("a lane stayed held")

    def test_a_reassigned_lane_goes_to_the_newest_request(self):
        first, second = Plan([[4]], block=True), Plan([[4]], block=True)
        runtime = FakeRuntime(first, second)
        harness = self.harness(runtime)
        left, left_response = harness.open_stream(
            "/v1/chat/completions", chat_body(stream=True, id_slot=0)
        )
        self.read_frame(left_response)
        right, right_response = harness.open_stream(
            "/v1/chat/completions", chat_body(stream=True, id_slot=0)
        )
        self.read_frame(right_response)
        # The older request lost the lane to the newer one: it keeps no row
        # and stamps nothing, and the newer one is the only request on lane 0.
        self.assertEqual(harness.app.lane_map.snapshot(), {2: 0})
        self.assertIsNone(harness.app.lane_map.stamp_id_slot(1))
        self.assertEqual(harness.app.lane_map.stamp_id_slot(2), 0)
        left.close()
        right.close()
        for plan in (first, second):
            plan.release.set()
            plan.start_release.set()

    def test_count_tokens_takes_no_lane(self):
        harness = self.harness()
        status, _, _ = harness.request(
            "POST", "/v1/messages/count_tokens", anthropic_body(id_slot=0)
        )
        self.assertEqual(status, 200)
        self.assertFalse(any(row["is_processing"] for row in self.slot_rows(harness)))


class CompatOffTests(HarnessTestCase):
    def test_without_the_option_the_routes_are_404_and_nothing_is_stamped(self):
        harness = self.harness()
        for path in ("/slots", "/props"):
            status, _, payload = harness.request("GET", path)
            self.assertEqual(status, 404)
            self.assertIn("not_found", json.loads(payload)["error"]["code"])
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(id_slot=1)
        )
        self.assertEqual(status, 400)
        self.assertIn("id_slot", json.loads(payload)["error"]["message"])
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body()
        )
        self.assertEqual(status, 200)
        self.assertNotIn("id_slot", json.loads(payload))

    def test_without_the_option_a_probe_error_logs_as_any_other_404(self):
        # The quiet rule belongs to the option: without it /slots is just
        # another unknown path, and its 404 reaches the console.
        harness = self.harness()
        # The handler runs on its own thread and imports print_status into
        # the server module, so patch it where it runs.
        with mock.patch("server.server.print_status") as console:
            self.assertEqual(harness.request("GET", "/slots")[0], 404)
        self.assertIn(
            "Error · not_found · GET /slots",
            [call.args[0] for call in console.call_args_list],
        )


if __name__ == "__main__":
    unittest.main()

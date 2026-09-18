import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from dev.tests import smoke_real
from server import api_shapes, chat_templates
from server.errors import APIError

VISION = ["text", "image", "pdf"]
MEDIA_PARTS = {
    "image_url": "image",
    "image": "image",
    "input_image": "image",
    "file": "PDF",
    "document": "PDF",
}
# A Qwen chat template's rendering of the later-system conversation.
IN_PLACE = (
    "<|im_start|>system\nYou are a terse assistant.<|im_end|>\n"
    "<|im_start|>user\nWhat is two plus two?<|im_end|>\n"
    "<|im_start|>assistant\nThe answer is four.<|im_end|>\n"
    "<|im_start|>system\nFrom now on answer in French.<|im_end|>\n"
    "<|im_start|>user\nWhat is three plus three?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def media(body):
    """The modality of the first image or PDF part of a request, if any."""
    messages = body.get("messages") or body.get("input") or []
    for message in messages if isinstance(messages, list) else []:
        content = message.get("content")
        for part in content if isinstance(content, list) else []:
            if part.get("type") in MEDIA_PARTS:
                return MEDIA_PARTS[part["type"]]
    return None


def refusal(modality, anthropic=False):
    """The body the server answers media with when it serves without vision."""
    try:
        api_shapes._require_vision(modality, False)
    except APIError as error:
        message = error.message
    if anthropic:
        return {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }
    return {
        "error": {"message": message, "type": "invalid_request_error", "code": None}
    }


class FakeServer:
    """Answers smoke_real.request like a text-only or vision server; records
    each request."""

    def __init__(self, modalities=VISION, later_system="native", prompt=IN_PLACE):
        self.modalities = modalities
        self.later_system = later_system
        self.prompt = prompt
        self.requests = {"submitted": 3, "cancelled": 0, "failed": 0}
        self.calls = []
        self.refuse = True

    def status(self):
        return {
            "vision": "image" in self.modalities,
            "input_modalities": self.modalities,
            "chat_template": {"later_system": self.later_system},
            "requests": dict(self.requests),
            "cache": {"reused_tokens": 0},
            "transport": {"restarts": 0},
        }

    def __call__(self, port, method, path, body=None, *, timeout=60):
        self.calls.append((method, path, body))
        if path == "/v1/models":
            return 200, {
                "data": [{"id": "test-model", "input_modalities": self.modalities}]
            }
        if path == "/status":
            return 200, self.status()
        if path == "/apply-template":
            return 200, {"prompt": self.prompt}
        modality = media(body)
        if modality and "image" not in self.modalities and self.refuse:
            return 400, refusal(modality, path == "/v1/messages")
        self.requests["submitted"] += 1
        return 200, {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }


class SmokeRealTests(unittest.TestCase):
    def test_status_checks_selected_kv_format_and_legacy_int8(self):
        for format in ("int8", "bf16"):
            kv = {
                "format": format,
                "quantization": "symmetric_int8" if format == "int8" else "none",
                "scale_type": "float32" if format == "int8" else "none",
            }
            status = {
                "ready": True,
                "metal": {"healthy": True},
                "transport": {"restarts": 0},
                "identity": {"cache": {"block_tokens": 32}, "kv": kv},
            }
            smoke_real.validate_status(status, format)
            with self.assertRaises(smoke_real.SmokeFailure):
                smoke_real.validate_status(
                    status, "bf16" if format == "int8" else "int8"
                )
            if format == "int8":
                old = {k: v for k, v in kv.items() if k != "format"}
                self.assertEqual(smoke_real.kv_identity({"q8": old}), kv)
                status["identity"] = {"cache": {"block_tokens": 32}, "q8": old}
                smoke_real.validate_status(status, "int8")
            else:
                kv["scale_type"] = "float32"
                with self.assertRaises(smoke_real.SmokeFailure):
                    smoke_real.validate_status(status, "bf16")

    def test_server_paths_are_resolved_from_caller_directory(self):
        with TemporaryDirectory() as directory, contextlib.chdir(directory):
            package = Path("model package")
            binary = Path("native build/splash")
            for absolute in (False, True):
                with self.subTest(absolute=absolute):
                    arguments = SimpleNamespace(
                        package=package.resolve() if absolute else package,
                        binary=binary.resolve() if absolute else binary,
                        model="test-model",
                        max_context=None,
                        max_memory=None,
                        max_cache_disk=None,
                        kv_format="bf16" if absolute else "int8",
                    )
                    with (
                        mock.patch.object(
                            smoke_real, "available_port", return_value=8000
                        ),
                        mock.patch.object(smoke_real.subprocess, "Popen") as popen,
                    ):
                        popen.return_value.poll.return_value = 0
                        server = smoke_real.RealServer(arguments)
                        try:
                            command = popen.call_args.args[0]
                            self.assertEqual(
                                command[command.index("--kv-format") + 1],
                                arguments.kv_format,
                            )
                            self.assertEqual(
                                popen.call_args.kwargs["cwd"], smoke_real.ROOT
                            )
                            self.assertEqual(
                                command[2], str(package.resolve() / "target")
                            )
                            self.assertEqual(
                                command[3], str(package.resolve() / "draft")
                            )
                            self.assertEqual(
                                command[command.index("--tokenizer") + 1],
                                str(package.resolve() / "tokenizer"),
                            )
                            self.assertEqual(
                                command[command.index("--binary") + 1],
                                str(binary.resolve()),
                            )
                        finally:
                            server.close()

    @staticmethod
    def timeout_status(**changes):
        status = {
            "instance": "test-instance",
            "ready": True,
            "metal": {"healthy": True},
            "transport": {"restarts": 0, "pending": 0},
            "requests": {
                "submitted": 7,
                "completed": 5,
                "cancelled": 1,
                "failed": 1,
            },
        }
        for key, value in changes.items():
            if isinstance(value, dict):
                status[key].update(value)
            else:
                status[key] = value
        return status

    def test_scoring_timeout_accepts_either_terminal_path_after_cleanup(self):
        for outcome in ("cancelled", "failed"):
            with self.subTest(outcome=outcome):
                before = self.timeout_status()
                active = self.timeout_status(
                    requests={"submitted": 8}, transport={"pending": 1}
                )
                finishing = self.timeout_status(
                    requests={"submitted": 8, outcome: 2}, transport={"pending": 1}
                )
                idle = self.timeout_status(requests={"submitted": 8, outcome: 2})
                with (
                    mock.patch.object(
                        smoke_real,
                        "request",
                        side_effect=[
                            (200, status) for status in (active, finishing, idle)
                        ],
                    ) as request,
                    mock.patch.object(smoke_real.time, "sleep") as sleep,
                ):
                    smoke_real.wait_for_timeout_cleanup(8000, before)
                self.assertEqual(request.call_count, 3)
                self.assertEqual(sleep.call_count, 2)

    def test_scoring_timeout_rejects_invalid_outcomes_and_unhealthy_runtime(self):
        cases = {
            "not_admitted": {"requests": {"submitted": 7}},
            "extra_request": {"requests": {"submitted": 9}},
            "completed": {"requests": {"completed": 6}},
            "double_terminal": {"requests": {"failed": 2}},
            "double_cancel": {"requests": {"cancelled": 3}},
            "new_instance": {"instance": "replacement"},
            "native_restart": {"transport": {"restarts": 1}},
            "not_ready": {"ready": False},
            "metal_unhealthy": {"metal": {"healthy": False}},
        }
        for name, changes in cases.items():
            with self.subTest(name=name):
                after = self.timeout_status(requests={"submitted": 8, "cancelled": 2})
                for key, value in changes.items():
                    if isinstance(value, dict):
                        after[key].update(value)
                    else:
                        after[key] = value
                with (
                    mock.patch.object(smoke_real, "request", return_value=(200, after)),
                    self.assertRaises(smoke_real.SmokeFailure),
                ):
                    smoke_real.wait_for_timeout_cleanup(8000, self.timeout_status())

    def test_scoring_timeout_cleanup_has_a_deadline(self):
        for cancelled, pending in ((1, 0), (1, 1), (2, 1)):
            with self.subTest(terminal=cancelled == 2, pending=pending):
                stuck = self.timeout_status(
                    requests={"submitted": 8, "cancelled": cancelled},
                    transport={"pending": pending},
                )
                with (
                    mock.patch.object(smoke_real, "request", return_value=(200, stuck)),
                    mock.patch.object(
                        smoke_real.time, "monotonic", side_effect=[0, 120]
                    ),
                    self.assertRaisesRegex(
                        smoke_real.SmokeFailure, "cleanup did not finish"
                    ),
                ):
                    smoke_real.wait_for_timeout_cleanup(8000, self.timeout_status())

    @staticmethod
    def assistant_message(text):
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }

    def run_images(self, output):
        red = {"choices": [{"message": {"content": "red"}}]}
        repeat = {
            **red,
            "metrics": {"cache": {"status": "hit", "matched_tokens": 24}},
        }
        blue = {
            "choices": [{"message": {"content": "blue"}}],
            "metrics": {"cache": {"matched_tokens": 0}},
        }
        documents = [
            red,
            {"images": {"encodes": 1, "embedding_reuses": 0}},
            repeat,
            {"images": {"encodes": 1, "embedding_reuses": 1}},
            blue,
            {"type": "message", "content": [{"type": "text", "text": "red"}]},
            {"object": "response", "output": output},
        ]
        with (
            mock.patch.object(
                smoke_real, "request", side_effect=[(200, item) for item in documents]
            ),
            mock.patch.object(
                smoke_real, "image_data_url", return_value="data:image/png;base64,AA=="
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            smoke_real.run_images(8000, "test-model", "test-request")

    def test_responses_image_accepts_final_assistant_text(self):
        self.run_images(
            [
                {"type": "reasoning", "summary": [{"text": "It may be red."}]},
                self.assistant_message("Blue."),
            ]
        )

    def test_responses_image_rejects_blue_outside_final_assistant_text(self):
        blue_message = self.assistant_message("blue")
        red_message = self.assistant_message("red")
        reasoning = {"type": "reasoning", "summary": [{"text": "Maybe blue."}]}
        cases = {
            "reasoning": [reasoning, red_message],
            "earlier_assistant": [blue_message, red_message],
            "no_assistant": [reasoning],
            "different_role": [red_message, {**blue_message, "role": "user"}],
            "other_content": [
                {
                    **red_message,
                    "content": [
                        {"type": "reasoning_text", "text": "blue"},
                        {"type": "output_text", "text": "red"},
                    ],
                }
            ],
        }
        for name, output in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    smoke_real.SmokeFailure, "Responses image answer"
                ):
                    self.run_images(output)

    def test_requests_send_the_api_key_the_server_requires(self):
        for key in (None, "", "secret"):
            with self.subTest(key=key):
                environment = {} if key is None else {"SPLASH_API_KEY": key}
                with (
                    mock.patch.dict(smoke_real.os.environ, environment, clear=True),
                    mock.patch.object(
                        smoke_real.http.client, "HTTPConnection"
                    ) as connection,
                ):
                    response = connection.return_value.getresponse.return_value
                    response.status = 200
                    response.read.return_value = b"{}"
                    smoke_real.request(8000, "GET", "/status")
                    smoke_real.request(8000, "POST", "/v1/models", {"a": 1})
                    smoke_real.stream_request(8000, "/v1/chat/completions", {})
                calls = connection.return_value.request.call_args_list
                headers = [call.args[3] for call in calls]
                self.assertEqual(
                    [header.get("Authorization") for header in headers],
                    [f"Bearer {key}" if key else None] * 3,
                )
                self.assertNotIn("Content-Type", headers[0])
                self.assertEqual(headers[1]["Content-Type"], "application/json")

    def test_discovery_reads_the_served_model_modalities(self):
        for modalities in (VISION, ["text"]):
            with self.subTest(modalities=modalities):
                server = FakeServer(modalities)
                with mock.patch.object(smoke_real, "request", server):
                    self.assertEqual(
                        smoke_real.input_modalities(8000, "test-model"), modalities
                    )
        cases = {
            "unlisted model": ("other-model", {}),
            "no modalities": ("test-model", {"input_modalities": None}),
            "vision disagrees": ("test-model", {"vision": True}),
        }
        for name, (model, status) in cases.items():
            with self.subTest(name=name):
                server = FakeServer(["text"] if name == "vision disagrees" else VISION)
                if name == "no modalities":
                    server.modalities = None
                original = server.status
                server.status = lambda original=original, status=status: {
                    **original(),
                    **status,
                }
                with (
                    mock.patch.object(smoke_real, "request", server),
                    self.assertRaises(smoke_real.SmokeFailure),
                ):
                    smoke_real.input_modalities(8000, model)

    def test_run_branches_on_modalities_and_the_template_probe(self):
        stream = (
            200,
            "text/event-stream",
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n'
            b'data: {"choices":[],"usage":{"completion_tokens":1}}\n'
            b"data: [DONE]\n",
        )

        def chat(body):
            if body.get("tools"):
                arguments = json.dumps({"value": "ok"})
                call = {"function": {"name": "record_probe", "arguments": arguments}}
                return {"choices": [{"message": {"tool_calls": [call]}}]}
            if body.get("response_format"):
                return {"choices": [{"message": {"content": '{"result": "ok"}'}}]}
            return {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 5},
            }

        for modalities, later_system in (
            (VISION, "native"),
            (["text"], "patched"),
            (["text"], "unsupported"),
        ):
            with self.subTest(modalities=modalities, later_system=later_system):
                server = FakeServer(modalities, later_system)

                def answer(port, method, path, body=None, **_kwargs):
                    if path == "/v1/chat/completions":
                        return 200, chat(body)
                    if path == "/v1/responses":
                        return 200, {"object": "response"}
                    if path == "/v1/messages":
                        if later_system == "unsupported":
                            message = chat_templates.LATER_SYSTEM_UNSUPPORTED
                            return 400, {"type": "error", "error": {"message": message}}
                        return 200, {"type": "message", "content": [{"text": "PONG"}]}
                    return server(port, method, path, body)

                with (
                    mock.patch.object(smoke_real, "request", answer),
                    mock.patch.object(
                        smoke_real, "stream_request", return_value=stream
                    ),
                    mock.patch.object(smoke_real, "run_images") as images,
                    mock.patch.object(smoke_real, "run_text_only") as text_only,
                    mock.patch.object(
                        smoke_real, "run_protocol_extensions"
                    ) as extensions,
                    mock.patch.object(smoke_real, "run_judgments"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    smoke_real.run(8000, "test-model")
                vision = "image" in modalities
                self.assertEqual(images.called, vision)
                self.assertEqual(text_only.called, not vision)
                self.assertEqual(extensions.call_args.args[2], vision)
                self.assertEqual(
                    any(path == "/apply-template" for _, path, _ in server.calls),
                    later_system != "unsupported",
                )

    def test_text_only_refuses_every_media_request_and_serves_text(self):
        def run(server):
            with (
                mock.patch.object(smoke_real, "request", server),
                mock.patch.object(
                    smoke_real,
                    "image_data_url",
                    return_value="data:image/png;base64,AA==",
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                smoke_real.run_text_only(8000, "test-model", "nonce")

        server = FakeServer(["text"])
        run(server)
        sent = [(path, media(body)) for method, path, body in server.calls if body]
        self.assertEqual(
            sent,
            [
                ("/v1/chat/completions", "image"),
                ("/v1/chat/completions", "PDF"),
                ("/v1/messages", "image"),
                ("/v1/messages", "PDF"),
                ("/v1/responses", "image"),
                ("/v1/chat/completions", None),
            ],
        )

        served = FakeServer(["text"])
        served.refuse = False
        with self.assertRaisesRegex(smoke_real.SmokeFailure, "not refused"):
            run(served)

        for name, document in {
            "other message": {"error": {"message": "image input is not supported"}},
            "other modality": refusal("PDF"),
            "no error": {},
        }.items():
            with self.subTest(name=name):
                server = FakeServer(["text"])
                original = server.__call__

                def answer(port, method, path, body=None, document=document, **_):
                    if (
                        body
                        and media(body) == "image"
                        and path == "/v1/chat/completions"
                    ):
                        return 400, document
                    return original(port, method, path, body)

                with self.assertRaisesRegex(smoke_real.SmokeFailure, "not refused"):
                    run(answer)

        for name, change in {
            "text fails": lambda server, path, body: (
                (500, {"error": {}}) if body and not media(body) else None
            ),
            "media reached the runtime": lambda server, path, body: (
                server.requests.update(submitted=server.requests["submitted"] + 1)
                if body and media(body)
                else None
            ),
            "media failed in the runtime": lambda server, path, body: (
                server.requests.update(failed=server.requests["failed"] + 1)
                if body and media(body)
                else None
            ),
        }.items():
            with self.subTest(name=name):
                server = FakeServer(["text"])
                original = server.__call__

                def answer(
                    port, method, path, body=None, change=change, server=server, **_
                ):
                    return change(server, path, body) or original(
                        port, method, path, body
                    )

                with self.assertRaises(smoke_real.SmokeFailure):
                    run(answer)

    def test_refusal_matches_the_server_messages(self):
        for modality in ("image", "PDF"):
            for anthropic in (False, True):
                document = refusal(modality, anthropic)
                self.assertTrue(
                    smoke_real.language_only_refusal(400, document, modality)
                )
                self.assertFalse(
                    smoke_real.language_only_refusal(200, document, modality)
                )
        self.assertIn(smoke_real.VISION_UNAVAILABLE, api_shapes.VISION_UNAVAILABLE)
        self.assertIn(
            smoke_real.LATER_SYSTEM_UNSUPPORTED, chat_templates.LATER_SYSTEM_UNSUPPORTED
        )
        self.assertEqual(
            set(smoke_real.LATER_SYSTEM_IN_PLACE) | {"unsupported"},
            {chat_templates.NATIVE, chat_templates.PATCHED, chat_templates.UNSUPPORTED},
        )

    def test_chat_template_probe_and_in_place_rendering(self):
        def run(server):
            with (
                mock.patch.object(smoke_real, "request", server),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                return smoke_real.run_chat_template(8000, "test-model")

        for later_system in (
            "native",
            "patched",
            {"default": "patched", "tool_use": "native"},
        ):
            with self.subTest(later_system=later_system):
                server = FakeServer(later_system=later_system)
                self.assertIn(run(server), ("native", "patched"))
                rendered = [
                    body for _, path, body in server.calls if path == "/apply-template"
                ]
                self.assertEqual(len(rendered), 1)
                roles = [message["role"] for message in rendered[0]["messages"]]
                self.assertEqual(
                    roles, ["system", "user", "assistant", "system", "user"]
                )
        # A template that does not render later system messages is not asked.
        server = FakeServer(
            later_system={"default": "unsupported", "tool_use": "native"}
        )
        self.assertEqual(run(server), "unsupported")
        self.assertFalse(any(path == "/apply-template" for _, path, _ in server.calls))

        later = "<|im_start|>system\nFrom now on answer in French.<|im_end|>\n"
        misplaced = IN_PLACE.replace(later, "")
        misplaced = misplaced.replace(
            "<|im_start|>system\nYou are a terse assistant.<|im_end|>\n",
            "<|im_start|>system\nYou are a terse assistant.\n\n"
            "From now on answer in French.<|im_end|>\n",
        )
        inline = IN_PLACE.replace(
            later, "<|im_start|>user\nFrom now on answer in French.<|im_end|>\n"
        )
        for name, (later_system, prompt) in {
            "dropped": ("native", IN_PLACE.replace(later, "")),
            "rendered twice": ("native", IN_PLACE.replace(later, later + later)),
            "moved to the front": ("patched", misplaced),
            "rendered as a user turn": ("patched", inline),
            "no probe": (None, IN_PLACE),
            "unknown probe result": ("renders", IN_PLACE),
            "named without default": ({"tool_use": "native"}, IN_PLACE),
        }.items():
            with self.subTest(name=name), self.assertRaises(smoke_real.SmokeFailure):
                run(FakeServer(later_system=later_system, prompt=prompt))


if __name__ == "__main__":
    unittest.main()

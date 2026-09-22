import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from dev.tests import smoke_real


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


if __name__ == "__main__":
    unittest.main()

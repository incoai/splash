import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from dev.tests import smoke_real


class SmokeRealTests(unittest.TestCase):
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

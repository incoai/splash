"""Native text-marker fixtures come from the installed tokenizer."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.engine import run_model_runtime_oracle as oracle
from dev.tests.server.test_chat_formats import ripple_tokenizer
from dev.tests.server.test_token_markers import advance
from install.families import named


class ModelRuntimeOracleTests(unittest.TestCase):
    def fixture(self):
        tokenizer = ripple_tokenizer()
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message.content }}{% endfor %}"
            "{% if add_generation_prompt %}assistant\n{% endif %}"
        )
        return tokenizer, oracle.fixture(tokenizer, named("Ripple-65B").sampling)

    def test_prompts_keep_reasoning_open_and_ids_in_their_own_vocabulary(self):
        tokenizer, fixture = self.fixture()
        self.assertTrue(
            tokenizer.decode(fixture["thinking_prompt"]).endswith("<think>\n")
        )
        self.assertTrue(
            tokenizer.decode(fixture["sampling_seed"]).endswith("assistant\n")
        )
        self.assertEqual(tokenizer.decode(fixture["thinking_end"]), "</think>")
        self.assertGreater(len(fixture["thinking_end"]), 1)
        self.assertGreater(len(fixture["long_end"]), 16)
        self.assertIn(fixture["thinking_end"][-1], fixture["filler"])
        self.assertEqual(fixture["vocabulary"], len(tokenizer))
        for name in (
            "thinking_prompt",
            "sampling_seed",
            "production_seed",
            "thinking_end",
            "long_end",
        ):
            self.assertTrue(all(0 <= token < len(tokenizer) for token in fixture[name]))
        self.assertGreater(fixture["temperature"], 0)
        self.assertEqual(fixture["top_p"], 0.95)
        self.assertEqual(fixture["top_k"], 20)

    def test_initial_states_and_long_close_reach_the_same_text_marker(self):
        _, fixture = self.fixture()
        for name, tokens in (
            (
                "thinking_markers",
                fixture["opening"] + fixture["filler"] + fixture["thinking_end"],
            ),
            ("opened_markers", fixture["filler"] + fixture["thinking_end"]),
            ("long_markers", fixture["long_end"]),
        ):
            program = fixture[name]
            state = program[5]
            for token in tokens:
                state = advance(program, state, token)
            self.assertEqual(state, program[3], name)

    def test_token_tag_family_preserves_existing_native_invocation(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            (root / "model.json").write_text(json.dumps({"family": "Qwen3.8-27B"}))
            with mock.patch.object(oracle.subprocess, "call", return_value=7) as call:
                self.assertEqual(
                    oracle.main(
                        ["binary", "library", str(root), "--state-format", "bf16"]
                    ),
                    7,
                )
            call.assert_called_once_with(
                ["binary", "library", str(root), "--state-format", "bf16"]
            )

    def test_text_marker_fixture_lives_until_native_process_exits(self):
        tokenizer, _ = self.fixture()
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            (root / "model.json").write_text(json.dumps({"family": "Ripple-65B"}))

            def native(command):
                path = Path(command[command.index("--token-fixture") + 1])
                self.assertEqual(
                    json.loads(path.read_text())["vocabulary"], len(tokenizer)
                )
                self.assertEqual(command[-2:], ["--kv-format", "bf16"])
                return 0

            with mock.patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
            ) as load:
                with mock.patch.object(oracle.subprocess, "call", side_effect=native):
                    self.assertEqual(
                        oracle.main(
                            ["binary", "library", str(root), "--kv-format", "bf16"]
                        ),
                        0,
                    )
            load.assert_called_once_with(
                root / "tokenizer", local_files_only=True, trust_remote_code=False
            )

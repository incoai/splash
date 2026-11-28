import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dev.tools import draft_golden


class TargetTokensTests(unittest.TestCase):
    """draft_golden.target_tokens takes the tokens a draft's candidates may name
    as the engine does: the draft config's valid_vocab_size where it states it,
    else the target's (Inkling-2-Mini's release leaves it to its target)."""

    def root(self, config: dict) -> Path:
        directory = Path(tempfile.mkdtemp())
        (directory / "config.json").write_text(json.dumps(config))
        return directory

    def test_a_stated_valid_vocab_size_is_taken(self):
        root = self.root(
            {"text_config": {"vocab_size": 201024, "unpadded_vocab_size": 200058}}
        )
        self.assertEqual(
            draft_golden.target_tokens(root, {"valid_vocab_size": 200058}), 200058
        )

    def test_an_absent_one_is_the_targets_unpadded_vocabulary(self):
        root = self.root(
            {"text_config": {"vocab_size": 201024, "unpadded_vocab_size": 200058}}
        )
        self.assertEqual(draft_golden.target_tokens(root, {}), 200058)

    def test_a_target_whose_rows_are_all_tokens_gives_its_vocabulary(self):
        root = self.root({"vocab_size": 248320})
        self.assertEqual(draft_golden.target_tokens(root, {}), 248320)

    def test_a_target_that_states_only_its_unpadded_vocabulary_gives_it(self):
        root = self.root({"text_config": {"unpadded_vocab_size": 200058}})
        self.assertEqual(draft_golden.target_tokens(root, {}), 200058)


def write_safetensors(path: Path, tensors: dict) -> None:
    """A safetensors file of int32 and float32 numpy arrays."""
    header, blobs, offset = {}, [], 0
    for name, value in tensors.items():
        value = np.ascontiguousarray(value)
        dtype = {"int32": "I32", "float32": "F32"}[value.dtype.name]
        blob = value.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


class MainTests(unittest.TestCase):
    """draft_golden's whole run over a golden prompt of 12 tokens, through a
    teacher-logits that writes a given lattice: the candidates are held to the
    target's vocabulary, whatever the prompt's length."""

    VOCABULARY = 60
    CAP = 30.0
    P, C = draft_golden.POSITIONS, draft_golden.CANDIDATES

    def golden(self, directory: Path) -> dict:
        ids = np.stack([np.arange(p, p + self.C) for p in range(self.P)]).astype(
            np.int32
        )
        values = np.tile(np.linspace(8.0, 0.0, self.C, dtype=np.float32), (self.P, 1))
        top24 = np.stack([np.arange(p, p + 24) for p in range(self.P)]).astype(np.int32)
        top24_values = np.tile(
            np.linspace(8.0, -2.0, 24, dtype=np.float32), (self.P, 1)
        )
        proposals = ids[:, 0]
        tensors = {
            "tokens": np.arange(12, dtype=np.int32),
            "captures": np.zeros((12, 4), np.float32),
            "block_anchor_position": np.array([4], np.int32),
            "logits_topk_ids": ids[None],
            "logits_topk_values": values[None],
            "logits_top24_ids": top24[None],
            "logits_top24_values": top24_values[None],
            "selector_predecessor_ids": np.concatenate([[0], proposals[:-1]]).astype(
                np.int32
            )[None],
            "selector_capped": draft_golden.softcap(values, self.CAP).astype(
                np.float32
            )[None],
            "selector_margin": np.full((1, self.P), 2.0, np.float32),
            "proposals": proposals[None],
        }
        write_safetensors(directory / "prompt.safetensors", tensors)
        (directory / "manifest.json").write_text(
            json.dumps({"prompts": {"prompt": {"file": "prompt.safetensors"}}})
        )
        return {
            "candidates": ids.ravel().tolist(),
            "unary": values.ravel().tolist(),
            "edges": [0.0] * (self.P * self.C * self.C),
        }

    def run_main(self, lattice: dict) -> tuple[int, str]:
        directory = Path(tempfile.mkdtemp())
        root, golden, work = (
            directory / "root",
            directory / "golden",
            directory / "work",
        )
        (root / "draft").mkdir(parents=True)
        golden.mkdir()
        (root / "config.json").write_text(
            json.dumps(
                {
                    "text_config": {
                        "vocab_size": 64,
                        "unpadded_vocab_size": self.VOCABULARY,
                    }
                }
            )
        )
        (root / "draft" / "config.json").write_text(
            json.dumps({"dflash_config": {"final_logit_softcapping": self.CAP}})
        )
        reference = self.golden(golden)
        (directory / "lattice.json").write_text(json.dumps(lattice or reference))
        # teacher-logits METALLIB ROOT TOKENS OUT --window END:ROWS ...: writes OUT-END.draft.json.
        teacher = directory / "teacher-logits"
        teacher.write_text(
            f"#!{sys.executable}\n"
            "import sys, shutil\n"
            "end = sys.argv[sys.argv.index('--window') + 1].split(':')[0]\n"
            f"shutil.copy({str(directory / 'lattice.json')!r}, f'{{sys.argv[4]}}-{{end}}.draft.json')\n"
        )
        teacher.chmod(0o755)
        argv = [
            "draft_golden",
            "lib",
            str(root),
            str(golden),
            str(work),
            "--set",
            "",
            "--teacher-logits",
            str(teacher),
        ]
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            status = draft_golden.main()
        return status, output.getvalue()

    def test_the_reference_lattice_passes(self):
        status, output = self.run_main({})
        self.assertEqual(status, 0, output)
        self.assertIn("draft golden: passed", output)

    def test_a_candidate_past_the_targets_vocabulary_fails(self):
        lattice = self.golden(Path(tempfile.mkdtemp()))
        lattice["candidates"][-1] = self.VOCABULARY
        status, output = self.run_main(lattice)
        self.assertEqual(status, 1, output)
        self.assertIn("a candidate past the head's tokens", output)


if __name__ == "__main__":
    unittest.main()

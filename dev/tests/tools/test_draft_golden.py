import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

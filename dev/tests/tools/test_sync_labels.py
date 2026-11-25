import json
import tempfile
import unittest
from pathlib import Path

from dev.tools import sync_labels


class SyncLabelsTests(unittest.TestCase):
    def test_repository_labels_are_valid(self):
        labels = sync_labels.load_labels()
        names = [label["name"] for label in labels]
        self.assertIn("type/bug", names)
        self.assertIn("area/ci", names)
        self.assertEqual(len(names), len(set(names)))
        for label in labels:
            self.assertRegex(label["color"], r"^[0-9a-f]{6}$")

    def test_rejects_duplicate_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.json"
            path.write_text(
                json.dumps(
                    [
                        {"name": "a", "color": "ffffff", "description": "one"},
                        {"name": "a", "color": "000000", "description": "two"},
                    ]
                )
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                sync_labels.load_labels(path)

    def test_rejects_an_invalid_color(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.json"
            path.write_text(
                json.dumps([{"name": "a", "color": "#ffffff", "description": "one"}])
            )
            with self.assertRaisesRegex(ValueError, "invalid color"):
                sync_labels.load_labels(path)

    def test_rejects_a_missing_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.json"
            path.write_text(json.dumps([{"name": "a", "color": "ffffff"}]))
            with self.assertRaisesRegex(ValueError, "missing"):
                sync_labels.load_labels(path)


if __name__ == "__main__":
    unittest.main()

"""install/suggested-models.tsv, the models the desktop app offers and shell
completion completes: its rows' format, and README.md's Models table, which
must list only its IDs."""

import collections
import json
import re
import tempfile
import unittest
from pathlib import Path

from install import models, upstream
from server import serve_options

REPO = Path(__file__).resolve().parents[3]


def suggested_rows():
    """Every line of the list but its # comments, split at tabs."""
    lines = (REPO / "install/suggested-models.tsv").read_text().splitlines()
    return [line.split("\t") for line in lines if not line.startswith("#")]


def readme_table():
    """The rows of the table under README.md's ## Models, header and rule
    left out, each split into its cells."""
    section = (REPO / "README.md").read_text().split("\n## Models\n", 1)[1]
    lines = [
        line
        for line in section.split("\n## ", 1)[0].splitlines()
        if line.startswith("|")
    ]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines[2:]]


def readme_models():
    """The backticked model IDs of the table under README.md's ## Models."""
    return [
        model
        for row in readme_table()
        for cell in row
        for model in re.findall(r"`([^`]+)`", cell)
    ]


class SuggestedModelsTests(unittest.TestCase):
    def test_rows_are_model_name_memory_size_and_about(self):
        rows = suggested_rows()
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(row=row):
                self.assertEqual(len(row), 5)
                model, name, memory, size, about = row
                self.assertEqual(serve_options.parse_model_id(model), model)
                self.assertTrue(name)
                self.assertRegex(memory, r"^[1-9][0-9]*$")
                self.assertRegex(size, r"^[1-9][0-9]*$")
                self.assertTrue(about)
                self.assertFalse(about.endswith("."))
        counts = collections.Counter(row[0] for row in rows)
        self.assertEqual([model for model, count in counts.items() if count > 1], [])
        abouts = collections.defaultdict(set)
        for _, name, _, _, about in rows:
            abouts[name].add(about)
        self.assertEqual([name for name, about in abouts.items() if len(about) > 1], [])

    def test_readme_models_table_lists_only_suggested_models(self):
        suggested = {row[0] for row in suggested_rows()}
        models = readme_models()
        self.assertTrue(models)
        self.assertEqual([model for model in models if model not in suggested], [])

    def test_the_unsupported_model_error_names_each_model_the_readme_lists(self):
        # A user who read a model's name in the README finds it in the
        # engine's refusal of a model it does not serve.
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps({"hidden_size": 5120}))
            with self.assertRaises(models.ModelError) as refused:
                upstream.check_model("mlx-affine", "none", config)
        supported = str(refused.exception).partition("; supported: ")[2]
        self.assertTrue(supported, refused.exception)
        names = set(re.findall(r"[^,() ][^,()]*[^,() ]", supported))
        self.assertEqual([row[0] for row in readme_table() if row[0] not in names], [])


if __name__ == "__main__":
    unittest.main()

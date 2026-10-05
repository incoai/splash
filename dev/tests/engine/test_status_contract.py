"""The engine's status document as the server reads it."""

import json
import unittest
from pathlib import Path

from server import protocol as wire
from server.metrics import PROMETHEUS_SERIES, prometheus_metrics

# A representative status document, every section of it set, which
# runtime_status_test.cpp checks the engine writes.
GOLDEN = Path(__file__).parent / "status_golden.json"
# The sections the server adds to the engine's document before it renders it.
SERVER_SECTIONS = {"transport", "frontend", "response_store"}


class StatusContractTests(unittest.TestCase):
    def test_every_series_renders_from_the_engine_status(self):
        status = json.loads(GOLDEN.read_text())
        self.assertEqual(status["schema_version"], wire.STATUS_SCHEMA_VERSION)
        lines = prometheus_metrics(status).splitlines()
        rendered = {line.split()[0] for line in lines if not line.startswith("#")}
        self.assertEqual(
            [
                name
                for name, path in PROMETHEUS_SERIES.items()
                if path[0] not in SERVER_SECTIONS and name not in rendered
            ],
            [],
        )
        self.assertIn('splash_memory_pressure{state="normal"} 1', lines)


if __name__ == "__main__":
    unittest.main()

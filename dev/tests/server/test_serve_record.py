import io
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest import mock

from dev.tests.server_fixtures import console_written
from server import serve_record


class ServeRecordTests(unittest.TestCase):
    def setUp(self):
        # A record that cannot be written: its directory gone, or a directory
        # in its place.
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.directory / "taken.json").mkdir()
        self.unwritable = (
            (self.directory / "gone/serve-8000.json", "No such file or directory"),
            (self.directory / "taken.json", "Is a directory"),
        )

    def test_a_record_other_than_starting_or_ready_is_best_effort(self):
        """A later phase before ready, a step of the warm-up, a refusal or a
        failure that cannot be recorded is a warning line of its own, naming
        the record and the system's error, and raises nothing: the start, or
        the failure it records, stays as it was."""
        for path, reason in self.unwritable:
            for write in (
                partial(serve_record.entered, phase="loading"),
                partial(serve_record.warming, steps_done=3, steps=9),
                partial(
                    serve_record.failed,
                    failure=serve_record.failure(serve_record.PORT, "in use"),
                ),
            ):
                with (
                    self.subTest(path=path.name, write=write.func.__name__),
                    mock.patch("sys.stderr", io.StringIO()) as log,
                ):
                    write(path)
                    console_written()
                    (line,) = log.getvalue().splitlines()
                    self.assertTrue(
                        line.endswith(
                            f" Warning · cannot write the serve record {path}: {reason}"
                        ),
                        line,
                    )
                    self.assertEqual(
                        list(self.directory.iterdir()), [self.directory / "taken.json"]
                    )

    def test_a_starting_or_ready_record_that_cannot_be_written_fails_the_start(
        self,
    ):
        """The app takes a start for ready from its record alone: a starting
        record, which replaces an earlier process's, or a ready record that
        cannot be written raises, naming the record and the system's error,
        and logs nothing of its own."""
        for path, reason in self.unwritable:
            for write in (serve_record.starting, serve_record.ready):
                with (
                    self.subTest(path=path.name, write=write.__name__),
                    mock.patch("sys.stderr", io.StringIO()) as log,
                    self.assertRaises(serve_record.NotRecorded) as failed,
                ):
                    write(path)
                self.assertEqual(
                    str(failed.exception),
                    f"cannot write the serve record {path}: {reason}",
                )
                console_written()
                self.assertEqual(log.getvalue(), "")
                self.assertEqual(
                    list(self.directory.iterdir()), [self.directory / "taken.json"]
                )

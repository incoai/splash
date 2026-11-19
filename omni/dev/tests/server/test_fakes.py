"""The shared fakes against the classes they stand in for. A fake that lacks a
method, or takes other parameters for one, fails the code that calls it in a
way production never would, and code that catches every error can hide it."""

import unittest

from dev.tests.audio_fixtures import FakeAudioRuntime
from dev.tests.server_fixtures import FakeProcess, interface_drift
from server import audio_runtime
from server import runtime as engine_runtime


class FakeInterfaceTests(unittest.TestCase):
    def test_each_fake_offers_the_interface_of_the_class_it_stands_in_for(self):
        runtime = FakeAudioRuntime(manual=True)
        process = FakeProcess(audio=True)
        self.addCleanup(process.kill)
        for fake, real in (
            (runtime, audio_runtime.AudioRuntime),
            (runtime.submit(None), engine_runtime.RuntimeCall),
            (process, engine_runtime.ProcessLike),
        ):
            with self.subTest(fake=type(fake).__name__):
                self.assertEqual(interface_drift(fake, real), [])

    def test_a_method_taking_other_parameters_is_drift(self):
        class Drifted(FakeAudioRuntime):
            def status(self, timeout=5.0):
                return self.status_event

        self.assertEqual(
            interface_drift(Drifted(), audio_runtime.AudioRuntime),
            ["method status"],
        )


if __name__ == "__main__":
    unittest.main()

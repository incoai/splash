"""The shared fakes against the classes they stand in for. A fake that lacks a
method, or takes other parameters for one, fails the code that calls it in a
way production never would, and code that catches every error can hide it."""

import unittest

from dev.tests.server_fixtures import (
    FakeConstraintFactory,
    FakeProcess,
    FakeRuntime,
    PassThroughConstraintFactory,
    interface_drift,
)
from server import constraints
from server import runtime as engine_runtime


class FakeInterfaceTests(unittest.TestCase):
    def test_each_fake_offers_the_interface_of_the_class_it_stands_in_for(self):
        runtime = FakeRuntime(manual=True)
        process = FakeProcess()
        self.addCleanup(process.kill)
        for fake, real in (
            (runtime, engine_runtime.MultiplexedRuntime),
            (runtime.submit(None), engine_runtime.RuntimeCall),
            (process, engine_runtime.ProcessLike),
            (FakeConstraintFactory(), constraints.ConstraintFactory),
            (PassThroughConstraintFactory(), constraints.ConstraintFactory),
        ):
            with self.subTest(fake=type(fake).__name__):
                self.assertEqual(interface_drift(fake, real), [])

    def test_a_method_taking_other_parameters_is_drift(self):
        class Drifted(FakeRuntime):
            def status(self, timeout=5.0):
                return self.status_event

        self.assertEqual(
            interface_drift(Drifted(), engine_runtime.MultiplexedRuntime),
            ["method status"],
        )


if __name__ == "__main__":
    unittest.main()

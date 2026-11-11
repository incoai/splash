import signal
import unittest
from types import SimpleNamespace
from unittest import mock

from server import protocol as wire
from server import supervision


def start_failure(kind):
    return supervision.EngineFailure(
        "start_failed",
        "refused",
        0.0,
        start=wire.StartFailedEvent(
            kind, wire.ShortfallBound.DEVICE_MEMORY, 0, 0, 0, b""
        ),
    )


class ThresholdTests(unittest.TestCase):
    def test_every_threshold_derives_from_the_engines_budgets(self):
        """A frame write may wait as long as a status answer: the engine's
        loop takes input between steps, and a step may last the loop-stall
        budget (5 s before, which fenced an engine in a legitimate step)."""
        thresholds = supervision.Thresholds.of(wire.SUPERVISION)
        self.assertEqual(
            (
                thresholds.status_answer,
                thresholds.write_progress,
                thresholds.end_acknowledgement,
                thresholds.stop_grace,
                thresholds.short_lived,
                thresholds.backoff_cap,
                thresholds.start_retry_window,
            ),
            (30.0, 30.0, 150.0, 15.0, 60.0, 60.0, 30.0),
        )
        self.assertEqual(thresholds.write_progress, thresholds.status_answer)
        self.assertEqual(supervision.THRESHOLDS, thresholds)


class RestartPolicyTests(unittest.TestCase):
    def test_the_sixth_short_lived_failure_in_a_row_stops(self):
        """Short-lived failures in a row double the backoff, and the sixth,
        whose wait would have reached the 60 s cap, stops the relaunches
        (they went on every 60 s for good, each loading the weights again)."""
        self.assertEqual(supervision.CRASH_LOOP_FAILURES, 6)
        policy = supervision.RestartPolicy()
        failure = supervision.EngineFailure("engine_unhealthy", "crashed", 1.0)
        self.assertEqual(
            [policy.after(failure) for _ in range(6)],
            [0.0, 5.0, 10.0, 20.0, 40.0, None],
        )

    def test_a_full_probation_of_service_starts_the_count_anew(self):
        policy = supervision.RestartPolicy()
        failure = supervision.EngineFailure("engine_unhealthy", "crashed", 1.0)
        served = supervision.EngineFailure("engine_unhealthy", "crashed", 60.0)
        self.assertEqual(
            [policy.after(failure) for _ in range(5)], [0.0, 5.0, 10.0, 20.0, 40.0]
        )
        self.assertEqual(policy.after(served), 0.0)
        self.assertEqual(
            [policy.after(failure) for _ in range(6)],
            [5.0, 10.0, 20.0, 40.0, 60.0, None],
        )

    def test_the_backoff_stays_at_the_cap_however_long_failures_go_on(self):
        """However long a relaunch waits for memory, the backoff stays at the
        cap (from the 1026th in a row, at least 17 h of relaunches in, it no
        longer fit a float, and the policy raised: the server relaunched with
        no backoff, or never again)."""
        policy = supervision.RestartPolicy()
        failure = start_failure(wire.StartFailureKind.HOST_MEMORY)
        delays = [policy.after(failure) for _ in range(2000)]
        self.assertEqual(set(delays[5:]), {60.0})
        served = supervision.EngineFailure("engine_unhealthy", "crashed", 60.0)
        self.assertEqual(policy.after(served), 0.0)
        self.assertEqual(policy.after(failure), 5.0)

    def test_a_start_that_waited_for_memory_neither_counts_nor_clears(self):
        """A relaunch refused memory is retried, every 60 s once the backoff
        reached its cap, for as long as it is refused: it says nothing about
        the engine, and the short-lived failures around it stay in a row."""
        for kind in (
            wire.StartFailureKind.HOST_MEMORY,
            wire.StartFailureKind.DRIVER_ALLOCATION,
            wire.StartFailureKind.MEMORY_BUDGET,
            wire.StartFailureKind.CONTEXT_BUDGET,
        ):
            with self.subTest(kind=kind):
                policy = supervision.RestartPolicy()
                failure = supervision.EngineFailure("engine_unhealthy", "crashed", 1.0)
                delays = [policy.after(failure) for _ in range(5)]
                delays += [policy.after(start_failure(kind)) for _ in range(2000)]
                self.assertEqual(delays[:5], [0.0, 5.0, 10.0, 20.0, 40.0])
                self.assertEqual(set(delays[5:]), {60.0})
                self.assertIsNone(policy.after(failure))

    def test_only_a_failure_a_relaunch_would_meet_again_stops_at_once(self):
        for kind in (
            wire.StartFailureKind.CONFIGURATION,
            wire.StartFailureKind.DEVICE,
            wire.StartFailureKind.COMMAND_MEMORY,
            wire.StartFailureKind.MODEL,
        ):
            with self.subTest(kind=kind):
                self.assertIsNone(
                    supervision.RestartPolicy().after(start_failure(kind))
                )
        missing = supervision.EngineFailure(
            "launch_failed", "missing", 0.0, launch=True
        )
        self.assertIsNone(supervision.RestartPolicy().after(missing))
        for kind in (
            wire.StartFailureKind.HOST_MEMORY,
            wire.StartFailureKind.DRIVER_ALLOCATION,
            wire.StartFailureKind.MEMORY_BUDGET,
            wire.StartFailureKind.CONTEXT_BUDGET,
            wire.StartFailureKind.INTERNAL,
        ):
            with self.subTest(kind=kind):
                policy = supervision.RestartPolicy()
                self.assertEqual(
                    [policy.after(start_failure(kind)) for _ in range(3)],
                    [0.0, 5.0, 10.0],
                )


class QuarantineTests(unittest.TestCase):
    def test_a_request_started_at_two_failures_in_a_row_is_refused(self):
        quarantine = supervision.Quarantine()
        self.assertEqual(quarantine.after(frozenset({b"x", b"y"})), frozenset())
        self.assertFalse(quarantine.refuses(b"x"))
        self.assertEqual(quarantine.after(frozenset({b"x", b"z"})), frozenset({b"x"}))
        self.assertTrue(quarantine.refuses(b"x"))
        self.assertFalse(quarantine.refuses(b"y") or quarantine.refuses(b"z"))
        # Refused for the hour, whatever fails next.
        self.assertEqual(quarantine.after(frozenset({b"y"})), frozenset())
        self.assertTrue(quarantine.refuses(b"x"))
        # Only the last two failures count: y was started at the first and
        # the third, not at two in a row.
        self.assertFalse(quarantine.refuses(b"y"))

    def patch_clock(self):
        """Patches the clock the quarantine reads; returns the one-element
        list that holds its time."""
        now = [0.0]
        self.enterContext(
            mock.patch.object(
                supervision, "time", SimpleNamespace(monotonic=lambda: now[0])
            )
        )
        return now

    def test_a_refusal_ends_after_the_quarantine(self):
        now = self.patch_clock()
        quarantine = supervision.Quarantine()
        quarantine.after(frozenset({b"x"}))
        quarantine.after(frozenset({b"x"}))
        now[0] = supervision.QUARANTINE_SECONDS - 1
        self.assertTrue(quarantine.refuses(b"x"))
        now[0] = supervision.QUARANTINE_SECONDS
        self.assertFalse(quarantine.refuses(b"x"))
        # Let back, it is still a suspect: the next failure it is started at
        # refuses it again, for a new hour.
        self.assertEqual(quarantine.after(frozenset({b"x"})), frozenset({b"x"}))
        now[0] = 2 * supervision.QUARANTINE_SECONDS - 1
        self.assertTrue(quarantine.refuses(b"x"))

    def test_a_request_let_back_once_no_longer_a_suspect_costs_two_failures(self):
        """A failure it was not started at replaces the suspects: let back,
        a request is refused again only at the second failure in a row it is
        started at, as at first."""
        now = self.patch_clock()
        quarantine = supervision.Quarantine()
        quarantine.after(frozenset({b"x"}))
        quarantine.after(frozenset({b"x"}))
        now[0] = supervision.QUARANTINE_SECONDS
        self.assertFalse(quarantine.refuses(b"x"))
        self.assertEqual(quarantine.after(frozenset({b"y"})), frozenset())
        self.assertEqual(quarantine.after(frozenset({b"x"})), frozenset())
        self.assertFalse(quarantine.refuses(b"x"))
        self.assertEqual(quarantine.after(frozenset({b"x"})), frozenset({b"x"}))
        self.assertTrue(quarantine.refuses(b"x"))

    def test_only_a_failure_a_request_may_have_caused_counts(self):
        """How the engine exited decides first: a fault signal counts, a kill
        never does, whatever its output showed, nor does an engine that had
        not exited. Only an engine that exited on its own (EX_SOFTWARE after
        an engine failure, EX_OK after the server's SIGTERM) is judged by its
        code."""
        for code in (
            "engine_execution_failed",
            "metal_execution_failed",
            "protocol_fatal",
        ):
            for returncode in (70, 0):
                with self.subTest(code=code, returncode=returncode):
                    self.assertTrue(supervision.request_caused(code, returncode))
        for fault in (
            signal.SIGSEGV,
            signal.SIGBUS,
            signal.SIGILL,
            signal.SIGABRT,
            signal.SIGFPE,
            signal.SIGTRAP,
        ):
            with self.subTest(signal=fault):
                self.assertTrue(supervision.request_caused("engine_unhealthy", -fault))
        for code, returncode in (
            ("metal_out_of_memory", 70),
            ("protocol_fatal", -signal.SIGKILL),
            ("engine_execution_failed", -signal.SIGTERM),
            ("engine_execution_failed", None),
            ("engine_unhealthy", -signal.SIGKILL),
            ("engine_unhealthy", None),
            ("engine_unhealthy", 1),
            ("start_failed", 70),
        ):
            with self.subTest(code=code, returncode=returncode):
                self.assertFalse(supervision.request_caused(code, returncode))


if __name__ == "__main__":
    unittest.main()

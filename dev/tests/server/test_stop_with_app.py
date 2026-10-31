"""The server stops, as Ctrl+C stops it, once the app that started it has ended
(SPLASH_APP_PID): a model the desktop app started gives its memory back when
the app crashes."""

import signal
import subprocess
import sys
import threading
import unittest

from server import server as api


class StopWithAppTest(unittest.TestCase):
    def watch(self, pid):
        """Watches `pid` as the server does; the event is set by the SIGINT."""
        interrupted = threading.Event()
        previous = signal.signal(signal.SIGINT, lambda *_: interrupted.set())
        self.addCleanup(signal.signal, signal.SIGINT, previous)
        api._stop_with(pid)
        return interrupted

    def test_stops_once_the_app_ends(self):
        app = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(app.wait)
        self.addCleanup(app.kill)
        interrupted = self.watch(app.pid)
        self.assertFalse(interrupted.wait(0.5), "stopped while the app runs")
        app.kill()
        self.assertTrue(interrupted.wait(5), "still running after the app ended")

    def test_stops_at_once_when_the_app_is_gone(self):
        app = subprocess.Popen([sys.executable, "-c", "pass"])
        app.wait()
        self.assertTrue(self.watch(app.pid).wait(5))


if __name__ == "__main__":
    unittest.main()

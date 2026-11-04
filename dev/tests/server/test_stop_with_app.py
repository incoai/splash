"""The server stops, as Ctrl+C stops it, once the app that started it has ended
(SPLASH_APP_PID): a model the desktop app started gives its memory back when
the app crashes."""

import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

from dev.tests.server_fixtures import main_args
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

    def test_the_server_watches_the_app_it_is_given(self):
        app = subprocess.Popen([sys.executable, "-c", "pass"])
        app.wait()
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))

        # main() reads SPLASH_APP_PID before it binds: the app gone, the startup
        # stops there.
        def bind(*_, **__):
            time.sleep(10)
            self.fail("still starting after the app ended")

        started = time.monotonic()
        with (
            mock.patch.dict(os.environ, {"SPLASH_APP_PID": str(app.pid)}),
            mock.patch.object(api, "parse_args", return_value=main_args()),
            mock.patch.object(api, "FrontendServer", side_effect=bind),
        ):
            api.main()
        self.assertLess(time.monotonic() - started, 5)


if __name__ == "__main__":
    unittest.main()

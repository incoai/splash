"""The server stops, as SIGTERM stops it, once the app that started it has
ended (SPLASH_APP_PID): a model the desktop app started gives its memory back
when the app crashes."""

import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

from dev.tests.server_fixtures import (
    NO_THINKING_END,
    keep_stop_signals,
    main_args,
    unprepared_hold,
)
from server import server as api


def join_watchers():
    """Waits for the server's watches of an app (_stop_with), each of which
    ends once its app has ended and the signal is sent."""
    for thread in threading.enumerate():
        if thread.name == "stop-with-app":
            thread.join()


class StopWithAppTest(unittest.TestCase):
    def end_after_test(self, app):
        """Ends `app` after the test and waits for the server's watch of it to
        send its signal, then gives the process its stop-signal handlers and
        mask back."""
        keep_stop_signals(self)
        self.addCleanup(join_watchers)
        self.addCleanup(app.wait)
        self.addCleanup(app.kill)

    def watch(self, app):
        """Watches `app` as the server does; the event is set by the SIGTERM
        it sends once `app` has ended."""
        self.end_after_test(app)
        stopped = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stopped.set())
        api._stop_with(app.pid)
        return stopped

    def test_stops_once_the_app_ends(self):
        app = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        stopped = self.watch(app)
        self.assertFalse(stopped.wait(0.5), "stopped while the app runs")
        app.kill()
        self.assertTrue(stopped.wait(5), "still running after the app ended")

    def test_stops_at_once_when_the_app_is_gone(self):
        app = subprocess.Popen([sys.executable, "-c", "pass"])
        app.wait()
        self.assertTrue(self.watch(app).wait(5))

    def test_the_server_watches_the_app_it_is_given(self):
        app = subprocess.Popen([sys.executable, "-c", "pass"])
        app.wait()
        self.end_after_test(app)

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

    def test_an_app_that_ends_while_the_server_stops_leaves_the_stop_graceful(self):
        """The app's end during a stop is not a second Ctrl+C: the engine
        exits gracefully, a persistent cache closed."""
        app = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.end_after_test(app)
        runtime = mock.Mock()
        # Stopped while the engine starts.
        runtime.wait_ready.side_effect = KeyboardInterrupt
        backend = mock.Mock()

        def close():
            # The app ends while the engine exits.
            app.kill()
            join_watchers()

        backend.close.side_effect = close
        with (
            mock.patch.dict(os.environ, {"SPLASH_APP_PID": str(app.pid)}),
            mock.patch.object(api, "parse_args", return_value=main_args()),
            mock.patch.object(api, "FrontendServer"),
            unprepared_hold(),
            mock.patch.object(api, "load_thinking_key", return_value=None),
            mock.patch.object(api.AutoTokenizer, "from_pretrained"),
            mock.patch.object(api, "validate_tokenizer"),
            mock.patch.object(api, "ChatTemplates", **NO_THINKING_END),
            mock.patch.object(
                api.engine_runtime, "MultiplexedRuntime", return_value=runtime
            ),
            mock.patch.object(api, "NativeBackend", return_value=backend),
            mock.patch.object(api, "print_status"),
        ):
            api.main()
        backend.close.assert_called_once_with()
        runtime.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()

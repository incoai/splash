# What the fakes share: each ends with the process that started it, as `splash serve` ends with
# the app. FakeRuntime copies it beside the launcher; each fake runs it from beside itself (runpy).
import os
import threading
import time


def end_with_app():
    """Ends this process once the process that started it is gone, as when a test run fails or is
    cut short by its time limit: the app that names itself by SPLASH_APP_PID, which starts a
    server in a group of its own that a signal to the tests never reaches; else the parent, the
    tests' process, which starts a fake itself. A thread watches from the call on, so that a fake
    held loading, warming or waiting for a request ends too."""
    app = int(os.environ.get("SPLASH_APP_PID") or os.getppid())

    def watch():
        # At 1 the parent was gone before this call, and launchd took this process on.
        while app != 1:
            try:
                os.kill(app, 0)
            except ProcessLookupError:
                break
            except PermissionError:
                pass
            time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()

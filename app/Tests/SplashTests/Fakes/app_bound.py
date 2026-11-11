# What the fake launchers share: they end with the app that started them, as `splash serve`
# does. FakeRuntime copies it beside the launcher, which runs it (runpy).
import os
import threading
import time


def end_with_app():
    """Ends this process once the app that started it is gone (SPLASH_APP_PID): the tests'
    process, as when a run fails. A thread watches from the call on, so that a fake held loading
    or warming ends too."""

    def watch():
        app = int(os.environ.get("SPLASH_APP_PID", "0"))
        while app:
            try:
                os.kill(app, 0)
            except ProcessLookupError:
                os._exit(0)
            except PermissionError:
                pass
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()

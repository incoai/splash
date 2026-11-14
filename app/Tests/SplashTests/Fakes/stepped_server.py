# A launcher that starts as `splash serve` does, one step at a time as the test signals it with
# a file in the runtime's folder, then answers as a ready server of a model that sees images and
# thinks at three efforts (MainViewTests, ComposerTests).
import json
import os
import runpy
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

root = Path(__file__).resolve().parents[1]
port = int(sys.argv[sys.argv.index("--port") + 1])


def after(name):
    while not (root / name).exists():
        time.sleep(0.02)


def record(phase, **fields):
    path = root / "build/runtime" / f"serve-{port}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "phase": phase} | fields))


# From the start, so that one held loading or warming ends with the app too.
runpy.run_path(str(Path(__file__).with_name("app_bound.py")))["end_with_app"]()


record("loading")
after("loaded")
if (root / "refused").exists():
    sys.exit(1)  # as the engine refuses a model it cannot serve
record("warming", warmup={"steps_done": 3, "steps": 10})
after("warm")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = {}
        if self.path == "/v1/models":
            body = {
                "data": [
                    {
                        "input_modalities": ["text", "image"],
                        "reasoning": {
                            "thinks": True,
                            "efforts": ["low", "medium", "high"],
                            "default": "medium",
                        },
                    }
                ]
            }
        if self.path == "/status":
            # A request answering, which this status goes on reporting once the test has
            # finished it, from a snapshot taken before, as the server answers while its engine
            # is busy, saying how long before; or those the test writes, from a snapshot as many
            # milliseconds old as it writes to "stale", ten minutes where it writes none.
            running = []
            age = None
            if (root / "stale").exists():
                age = float((root / "stale").read_text() or 600_000)
            if (root / "answering").exists():
                running = [
                    {
                        "id": 1,
                        "phase": "decode",
                        "prompt_tokens": 13,
                        "prompt_processed": 13,
                        "generated_tokens": 1,
                        "age_ms": 400,
                    }
                ]
                if age is None and (root / "finished").exists():
                    age = 600_000.0
            if (root / "running.json").exists():
                running = json.loads((root / "running.json").read_text())
            body = {
                "active_requests": running,
                "metrics": {"decode_tokens_per_second": 41.6},
                "transport": {
                    "status_stale": age is not None,
                    "status_age_ms": age or 0.0,
                },
            }
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def stop():
    # As a server that stops restarting its engine: its record says why, then it exits.
    after("broken")
    record("failed", failure={"kind": "stopped", "message": "x"})
    os._exit(69)


def answer():
    # As the server logs a request the engine takes, then ends.
    after("answering")
    print(
        time.strftime("%H:%M:%S") + " Started · request 1 · input 13 · cached 0",
        flush=True,
    )
    after("finished")
    print(
        time.strftime("%H:%M:%S")
        + " Done · request 1 · input 13 · cached 0 · output 1 · TTFT 0.4s · 2.5 tok/s",
        flush=True,
    )


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
threading.Thread(target=answer, daemon=True).start()
record("ready")
threading.Thread(target=stop, daemon=True).start()
server.serve_forever()

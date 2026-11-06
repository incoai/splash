# A launcher that starts as `splash serve` does, one step at a time as the test signals it with
# a file in the runtime's folder, then answers as a ready server of a model that sees images and
# thinks at three efforts (MainViewTests, ComposerTests).
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

root = Path(__file__).resolve().parents[1]
port = int(sys.argv[sys.argv.index("--port") + 1])


def after(name):
    while not (root / name).exists():
        time.sleep(0.02)


def record(phase):
    path = root / "build/runtime" / f"serve-{port}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "phase": phase}))


record("loading")
after("loaded")
if (root / "refused").exists():
    sys.exit(1)  # as the engine refuses a model it cannot serve
record("warming")
after("warm")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = {}
        if self.path == "/v1/models":
            body = {"data": [{
                "input_modalities": ["text", "image"],
                "reasoning": {"thinks": True, "efforts": ["low", "medium", "high"], "default": "medium"},
            }]}
        if self.path == "/status":
            running = [{}] if (root / "answering").exists() else []
            body = {
                "active_requests": running,
                "metrics": {"decode_tokens_per_second": 41.6},
            }
            if (root / "broken").exists():
                body["transport"] = {"stopped": True, "error": "x"}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
record("ready")
server.serve_forever()

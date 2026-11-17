# A launcher that serves as a ready `splash serve` of a model that sees images and thinks: it
# takes a message, reads its prompt for 0.3 s, then streams a reply in parts over 0.7 s, and
# titles a chat after 0.4 s (ChatScreenTests). Files in the runtime's folder stop it at each step,
# so that a test can draw what one changes before the next changes the window again: while `hold`
# is there, it holds a message before it takes or refuses it; while `refuse` is there, it refuses
# the message, as one longer than the model's context; and while `wait` is there, the reply to a
# message it has taken waits to start.
import json
import os
import runpy
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

root = Path(__file__).resolve().parents[1]
port = int(sys.argv[sys.argv.index("--port") + 1])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            return self.reply(
                {
                    "data": [
                        {
                            "input_modalities": ["text", "image", "pdf"],
                            "reasoning": {
                                "thinks": True,
                                "efforts": ["low", "medium", "high"],
                                "default": "medium",
                            },
                        }
                    ]
                }
            )
        self.reply({})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not body.get("stream"):
            time.sleep(0.4)
            title = {"choices": [{"message": {"content": "A Title"}, "finish_reason": "stop"}]}
            return self.reply(title)
        while (root / "hold").exists():
            time.sleep(0.01)
        if (root / "refuse").exists():
            error = {
                "message": "prompt exceeds the context window",
                "code": "context_length_exceeded",
            }
            return self.reply({"error": error}, 400)
        # As the server takes a message: its response starts, with a chunk that says nothing.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        start = {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
        self.wfile.write(f"data: {json.dumps(start)}\n\n".encode())
        self.wfile.flush()
        while (root / "wait").exists():
            time.sleep(0.01)
        for processed in range(1, 4):
            progress = {"choices": [], "prompt_progress": {"total": 3, "processed": processed}}
            self.wfile.write(f"data: {json.dumps(progress)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.1)
        for part in ["Spring ", "in Kyoto ", "is short: ", "the blossoms ", "peak ", "in early ", "April."]:
            delta = {"choices": [{"delta": {"content": part}}]}
            self.wfile.write(f"data: {json.dumps(delta)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.1)
        self.wfile.write(b"data: [DONE]\n\n")


runpy.run_path(str(Path(__file__).with_name("app_bound.py")))["end_with_app"]()

server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
record = root / "build/runtime" / f"serve-{port}.json"
record.parent.mkdir(parents=True, exist_ok=True)
record.write_text(json.dumps({"pid": os.getpid(), "phase": "ready"}))
server.serve_forever()

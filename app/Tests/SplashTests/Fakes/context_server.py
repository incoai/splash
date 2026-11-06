# A launcher that serves as a ready `splash serve` does: it counts prompts and answers each,
# and writes what it was asked to the runtime's folder (ChatContextFlowTests).
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

root = Path(__file__).resolve().parents[1]
port = int(sys.argv[sys.argv.index("--port") + 1])


def size(content):
    if isinstance(content, str):
        return len(content.encode())
    return sum(
        64 if part["type"] == "image_url" else len(part["text"].encode())
        for part in content
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body, status=200, kind="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self.reply(
                {
                    "data": [
                        {"context_length": 1024, "input_modalities": ["text", "image"]}
                    ]
                }
            )
        else:
            self.reply({})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/count_tokens"):
            with (root / "counted.jsonl").open("a") as output:
                output.write(json.dumps(body["messages"]) + "\n")
            text = body["messages"][-1]["content"]
            if text == "offline":
                return self.reply({"error": {"message": "Try again shortly"}}, 503)
            if text == "slow question":
                (root / "counting").touch()
                time.sleep(5)
            count = 100 + sum(
                size(m["content"]) + len(m.get("reasoning_content", "").encode())
                for m in body["messages"]
            )
            return self.reply({"input_tokens": count})
        with (root / "sent.jsonl").open("a") as output:
            output.write(json.dumps(body["messages"]) + "\n")
        self.reply(
            'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\ndata: [DONE]\n\n',
            kind="text/event-stream",
        )


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
record = root / "build/runtime" / f"serve-{port}.json"
record.parent.mkdir(parents=True, exist_ok=True)
record.write_text(json.dumps({"pid": os.getpid(), "phase": "ready"}))
server.serve_forever()

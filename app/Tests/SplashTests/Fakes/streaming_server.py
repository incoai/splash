# Stream replies, optionally waiting for the test to acknowledge each chunk.
# Title and endless modes also serve the chat lifecycle tests.
import itertools
import json
import select
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not body.get("stream"):
            data = json.dumps({"choices": [{"message": {"content": title}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(data):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        parts = ["One", " two", " three"]
        try:
            for part in itertools.cycle(parts) if "--endless" in sys.argv else parts:
                chunk(
                    (
                        'data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % part
                    ).encode()
                )
                if "--ack" in sys.argv:
                    while not select.select([sys.stdin], [], [], 0.05)[0]:
                        chunk(b"data: {}\n\n")
                    if not sys.stdin.readline():
                        return
                else:
                    time.sleep(0.4)
            chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client stopped


title = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else None
server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.handle_request()
if title is not None:
    server.handle_request()

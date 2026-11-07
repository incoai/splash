# Stream 3 parts, waiting for an observation before the next part or the end,
# so a busy test runner can inspect each intermediate reply (SplashTests).
import select
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(data):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        for part in ["One", " two", " three"]:
            chunk(
                ('data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % part).encode()
            )
            # Progress keeps the 50 ms redraw gate live while this part is held.
            while not select.select([sys.stdin], [], [], 0.05)[0]:
                chunk(b"data: {}\n\n")
            if not sys.stdin.readline():
                return
        chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.handle_request()

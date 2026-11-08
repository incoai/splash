# A server that streams one reply in three parts, apart in time, on the port it prints, and
# ends (SplashTests); with --endless, the parts go round until the client goes.
import itertools
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        parts = ["One", " two", " three"]
        try:
            for part in itertools.cycle(parts) if "--endless" in sys.argv else parts:
                self.wfile.write(
                    ('data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % part).encode()
                )
                self.wfile.flush()
                time.sleep(0.4)
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client stopped


server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.handle_request()

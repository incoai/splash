# A server that streams one reply in three parts, apart in time, on the port it prints, and
# ends (SplashTests); with --endless, the parts go round until the client goes, and with
# --title TITLE, it then answers the request for a title, which streams nothing, with TITLE.
import itertools
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
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


title = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else None
server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.handle_request()
if title is not None:
    server.handle_request()

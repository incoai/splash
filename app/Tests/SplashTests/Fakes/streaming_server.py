# A server that streams one reply in three parts, apart in time, on the port it prints, and
# ends (SplashTests).
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
        for part in ["One", " two", " three"]:
            self.wfile.write(
                ('data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % part).encode()
            )
            self.wfile.flush()
            time.sleep(0.4)
        self.wfile.write(b"data: [DONE]\n\n")


server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.handle_request()

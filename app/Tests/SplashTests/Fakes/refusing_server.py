# A server on the port it prints that answers each streamed request as the next of the actions
# its command line lists after a folder (ChatStoreTests). "answer" streams a short reply at once.
# The others hold the request first: they make the file "holding" in the folder and wait until
# the test has made "release" there, then take both away. Then "refuse" refuses the request, as
# one over the server's limits, and makes "refused"; "take" streams the reply; "drop" closes the
# connection without a byte; "fail" starts the response and ends it with an error, as the server
# does once it has taken a request; and "wait" starts the response with a keepalive, as the
# server does for a request that waits for its engine, and holds it again before the reply. A
# request for a title, which streams nothing, gets "A Title". It ends once the tests that
# started it are gone.
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

folder = Path(sys.argv[1])
actions = sys.argv[2:]
taking = threading.Lock()
ANSWER = b'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\ndata: [DONE]\n\n'


def hold():
    (folder / "holding").touch()
    while not (folder / "release").exists():
        time.sleep(0.02)
    (folder / "holding").unlink()
    (folder / "release").unlink()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, kind, data):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def stream(self, *events):
        """Starts the response, or goes on with it, with `events`."""
        if not self.started:
            self.started = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
        for event in events:
            self.wfile.write(event)
        self.wfile.flush()

    def do_POST(self):
        self.started = False
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not body.get("stream"):
            title = {"choices": [{"message": {"content": "A Title"}}]}
            return self.reply(200, "application/json", json.dumps(title).encode())
        with taking:
            action = actions.pop(0)
        self.close_connection = True
        if action == "answer":
            return self.stream(ANSWER)
        hold()
        try:
            if action == "refuse":
                error = {"error": {"message": "Request body too large"}}
                self.reply(413, "application/json", json.dumps(error).encode())
                (folder / "refused").touch()
            elif action == "take":
                self.stream(ANSWER)
            elif action == "drop":
                self.connection.shutdown(socket.SHUT_RDWR)
            elif action == "fail":
                error = {"error": {"message": "The engine stopped"}}
                self.stream(
                    f"data: {json.dumps(error)}\n\n".encode(), b"data: [DONE]\n\n"
                )
            elif action == "wait":
                self.stream(b": splash-keepalive\n\n")
                hold()
                self.stream(ANSWER)
        except OSError:
            pass  # the client stopped the request meanwhile


def end_with_tests():
    while os.getppid() != 1:
        time.sleep(0.5)
    os._exit(0)


threading.Thread(target=end_with_tests, daemon=True).start()
server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()

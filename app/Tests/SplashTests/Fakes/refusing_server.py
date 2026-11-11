# A server on the port it prints that answers each streamed request as the next of the actions
# its command line lists after a folder (ChatStoreTests): "answer" streams a short reply, and
# "refuse" makes the file "holding" in the folder, refuses the request, as one over the server's
# limits, once the test has made "release" there, then makes "refused". A request for a title,
# which streams nothing, gets "A Title". It ends once the tests that started it are gone.
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

folder = Path(sys.argv[1])
actions = sys.argv[2:]
taking = threading.Lock()


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

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not body.get("stream"):
            title = {"choices": [{"message": {"content": "A Title"}}]}
            return self.reply(200, "application/json", json.dumps(title).encode())
        with taking:
            action = actions.pop(0)
        if action == "answer":
            data = b'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\ndata: [DONE]\n\n'
            return self.reply(200, "text/event-stream", data)
        (folder / "holding").touch()
        while not (folder / "release").exists():
            time.sleep(0.02)
        error = {"error": {"message": "Request body too large"}}
        self.reply(413, "application/json", json.dumps(error).encode())
        (folder / "refused").touch()


def end_with_tests():
    while os.getppid() != 1:
        time.sleep(0.5)
    os._exit(0)


threading.Thread(target=end_with_tests, daemon=True).start()
server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()

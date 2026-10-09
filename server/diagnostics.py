"""Concise console diagnostics without request-body logging."""

import sys
import time
import traceback
from pathlib import Path

# The server package, whose innermost frame locates an unexpected error.
_PACKAGE = Path(__file__).resolve().parent


def log_unexpected(error):
    """Print the type of an unexpected error and the innermost line of the
    server it passed through, never its message, which may carry request
    data."""
    try:
        location = ""
        for frame, line in traceback.walk_tb(error.__traceback__):
            path = Path(frame.f_code.co_filename).resolve()
            if path.parent == _PACKAGE:
                location = f" · {_PACKAGE.name}/{path.name}:{line}"
        print_status(
            f"Error · internal_server_error · {type(error).__name__}{location}",
            error=True,
        )
    except Exception:
        pass


def print_status(message, *, error=False):
    # One write per line, newline included, so that lines written at once by
    # request threads, or by the native runtime on the shared stderr, stay
    # whole in a terminal or in one log file.
    stream = sys.stderr if error else sys.stdout
    stream.write(f"{time.strftime('%H:%M:%S')} {message}\n")
    stream.flush()


def print_request(record):
    outcome = record["outcome"]
    if outcome == "error":
        print_status(f"Error · {record.get('error_code', 'runtime_error')}", error=True)
        return
    metrics = record.get("metrics", {})
    latency = metrics.get("request_latency", {})
    parts = [
        "Cancelled" if outcome == "cancelled" else "Done",
        f"input {record['prompt_tokens']:,}",
        f"cached {metrics.get('cache', {}).get('matched_tokens', 0):,}",
        f"output {record.get('completion_tokens', 0):,}",
    ]
    tools = record.get("tools")
    if isinstance(tools, dict) and tools.get("count"):
        parts.append(f"tools {tools['count']}·{tools.get('signature', '')}")
    ttft = latency.get("ttft_ms")
    speed = latency.get("stream_tokens_per_second")
    if ttft is not None:
        parts.append(f"TTFT {ttft / 1000:.1f}s")
    if speed is not None:
        parts.append(f"{speed:.1f} tok/s")
    print_status(" · ".join(parts))

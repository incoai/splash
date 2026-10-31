"""Run PDFium in a disposable process with supervised memory and time budgets."""

import json
import math
import resource
import sys
import time
from dataclasses import asdict

from . import workers
from .errors import APIError

MAX_SECONDS = 30.0
MAX_MEMORY_BYTES = 1024 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
WORK = workers.Work(
    subject="PDF processing",
    worker="document worker",
    unavailable="document_unavailable",
    oversized="rendered PDF exceeds the size limit",
)


def render(payload, limits, remaining):
    """Render a PDF in a worker within the `remaining` seconds of its request,
    infinite when it has no deadline, and the worker's own time limit."""
    duration = min(MAX_SECONDS, remaining)
    finished = workers.run(
        [
            sys.executable,
            "-m",
            "server.document_worker",
            json.dumps(limits),
            str(duration),
        ],
        payload,
        WORK,
        remaining=remaining,
        max_seconds=MAX_SECONDS,
        max_memory_bytes=MAX_MEMORY_BYTES,
        max_output_bytes=MAX_OUTPUT_BYTES,
        cwd=workers.SERVER_ROOT,
    )
    if finished.returncode != 0:
        raise APIError(400, "PDF processing failed or exceeded its resource limit")
    try:
        result = json.loads(finished.output)
    except (ValueError, UnicodeError):
        raise APIError(
            500,
            "document worker returned an invalid result",
            "document_worker_failed",
        ) from None
    if "error" in result:
        error = result["error"]
        raise APIError(error["status"], error["message"], error["code"])
    return result["pages"]


def main():
    from .documents import MAX_PDF_BYTES, DocumentBudget, RenderLimits, render_pages

    limits = RenderLimits(**json.loads(sys.argv[1]))
    duration = float(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    cpu_seconds = max(1, math.ceil(duration))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    # On macOS RLIMIT_AS aliases the advisory RSS limit. The parent instead
    # measures resident/physical footprint and terminates an over-budget child.
    budget = DocumentBudget(
        deadline=time.monotonic() + duration, remaining_bytes=limits.request_bytes
    )
    try:
        payload = sys.stdin.buffer.read(MAX_PDF_BYTES + 1)
        if len(payload) > MAX_PDF_BYTES:
            raise APIError(400, "PDF document exceeds the size limit")
        pages = render_pages(payload, budget, limits)
        result = {"pages": [asdict(page) for page in pages]}
    except APIError as error:
        result = {
            "error": {
                "status": error.status,
                "message": error.message,
                "code": error.code,
            }
        }
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()

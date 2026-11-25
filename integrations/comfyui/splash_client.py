"""Async HTTP client of a Splash video server (`splash serve`): its
/v1/videos jobs.

Pure aiohttp; imports nothing from ComfyUI. The server is the source of
truth for validation: the client builds the request, polls the job and
downloads the mp4.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
from urllib.parse import urlsplit

import aiohttp

POLL_INTERVAL = 1.0
DELETE_TIMEOUT = 5.0
VIDEO_MODALITY = "video"
# The characters of a non-JSON error body an error message quotes.
ERROR_TEXT_CHARS = 200

log = logging.getLogger("comfyui_splash")


class SplashError(Exception):
    """An error reported by, or about, the Splash server."""

    def __init__(self, code: str, message: str, status: Optional[int] = None):
        super().__init__(code, message, status)
        self.code = code
        self.message = message
        self.status = status

    def __str__(self) -> str:
        return f"Splash {self.code}: {self.message}"


class SplashInterrupted(Exception):
    """The caller interrupted the run; the job was deleted (best effort)."""


@dataclass(frozen=True)
class FilePart:
    """A file field of the multipart body."""

    name: str
    filename: str
    content_type: str
    data: bytes


def _text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def encode_multipart(fields: dict, files: Iterable[FilePart] = ()) -> tuple[str, bytes]:
    """A multipart/form-data body as (Content-Type, bytes): the text fields
    as strings (booleans "true"/"false") in order, then the files in order."""
    boundary = "splash-" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{name}"\r\n'.encode()
        out += b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        out += _text(value).encode("utf-8") + b"\r\n"
    for part in files:
        out += f"--{boundary}\r\n".encode()
        out += (
            f'Content-Disposition: form-data; name="{part.name}"; '
            f'filename="{part.filename}"\r\n'
        ).encode()
        out += f"Content-Type: {part.content_type}\r\n\r\n".encode()
        out += bytes(part.data) + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", bytes(out)


def encode_request(fields: dict, files: Iterable[FilePart] = ()) -> tuple[str, bytes]:
    """The POST /v1/videos body: JSON without files, multipart with them."""
    files = list(files)
    if files:
        return encode_multipart(fields, files)
    return "application/json", json.dumps(fields).encode("utf-8")


def request_line(node, fields: dict, files, base_url: str, job_id=None, error=None) -> str:
    """The `splash request` log line of a POST: what was sent, never the
    prompt text or the api key; the job id, or the error that answered it."""
    files = list(files)
    prompt = _text(fields.get("prompt", ""))
    items = [
        ("node", node or "-"),
        ("format", "multipart" if files else "json"),
        ("prompt_chars", len(prompt)),
        ("prompt_sha256", hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]),
    ]
    items += [(name, _text(value)) for name, value in fields.items() if name != "prompt"]
    items.append(("files", len(files)))
    if files:
        items.append(("parts", ",".join(f"{p.name}:{p.filename}:{len(p.data)}" for p in files)))
    items.append(("base_url", base_url))
    if error is None:
        items.append(("job", job_id or "-"))
    elif isinstance(error, SplashError):
        items += [("error", error.code), ("status", error.status if error.status is not None else "-")]
    else:
        items.append(("error", type(error).__name__))
    return "splash request " + " ".join(f"{key}={value}" for key, value in items)


def _plain(text: str, limit: int = ERROR_TEXT_CHARS) -> str:
    """The text of a body with its markup tags dropped and its whitespace
    collapsed, cut at `limit` characters."""
    plain = " ".join(re.sub(r"<[^>]*>", " ", text).split())
    return plain if len(plain) <= limit else plain[:limit] + "…"


async def _error_of(response: aiohttp.ClientResponse) -> SplashError:
    """The SplashError of a non-2xx response, from its {"error": {...}} body;
    any other body (an HTML page, say) is named by its URL and a short plain
    text of it."""
    text = await response.text(errors="replace")
    try:
        error = json.loads(text).get("error")
        if isinstance(error, dict):
            code = error.get("code") or error.get("type") or f"http_{response.status}"
            message = error.get("message") or text.strip() or response.reason or ""
            return SplashError(str(code), str(message), response.status)
    except (ValueError, AttributeError):
        pass
    plain = _plain(text) or response.reason or "no text"
    return SplashError(
        f"http_{response.status}",
        f"HTTP {response.status} from {response.url}, not a Splash error answer: {plain}",
        response.status,
    )


class SplashClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        poll_interval: float = POLL_INTERVAL,
        session: Optional[aiohttp.ClientSession] = None,
        node: str = "",
        serve_model: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.poll_interval = poll_interval
        self._session = session
        self.node = node  # the ComfyUI node in the `splash request` log line
        self.serve_model = serve_model  # the `--model` a node's server is started with

    def _headers(self, extra: Optional[dict] = None) -> dict:
        headers = dict(extra or {})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _request(self, session, method, path, *, body=None, content_type=None, params=None):
        headers = self._headers({"Content-Type": content_type} if content_type else None)
        try:
            async with session.request(
                method, self.base_url + path, data=body, headers=headers, params=params
            ) as response:
                if response.status >= 400:
                    raise await _error_of(response)
                if response.content_type == "application/json":
                    return await response.json()
                return await response.read()
        except aiohttp.ClientConnectionError as error:
            raise SplashError(
                "unreachable", f"cannot reach the Splash server at {self.base_url}: {error}"
            ) from error

    async def _post(self, session, path, fields: dict, files: Iterable[FilePart] = ()):
        """POST the request and log one `splash request` line on its answer."""
        files = list(files)
        content_type, body = encode_request(fields, files)
        try:
            answer = await self._request(session, "POST", path, body=body, content_type=content_type)
        except BaseException as error:
            log.info("%s", request_line(self.node, fields, files, self.base_url, error=error))
            raise
        job_id = answer.get("id") if isinstance(answer, dict) else None
        log.info("%s", request_line(self.node, fields, files, self.base_url, job_id=job_id))
        return answer

    async def submit(self, session, fields: dict, files: Iterable[FilePart] = ()) -> dict:
        return await self._post(session, "/v1/videos", fields, files)

    async def get(self, session, job_id: str) -> dict:
        return await self._request(session, "GET", f"/v1/videos/{job_id}")

    async def content(self, session, job_id: str, variant: str = "video") -> bytes:
        return await self._request(
            session, "GET", f"/v1/videos/{job_id}/content", params={"variant": variant}
        )

    async def delete(self, session, job_id: str):
        return await self._request(session, "DELETE", f"/v1/videos/{job_id}")

    async def generate(
        self,
        fields: dict,
        files: Iterable[FilePart] = (),
        *,
        on_progress: Optional[Callable[[dict], None]] = None,
        is_interrupted: Optional[Callable[[], bool]] = None,
        on_preview: Optional[Callable[[bytes], None]] = None,
    ) -> bytes:
        """Submit a job, poll it every poll_interval seconds and return its
        mp4. On interrupt, cancellation or any other exception from the
        loop or a callback, the job is deleted (best effort) and the
        exception re-raised; a 503 queue-full answer is raised, not retried."""
        session = self._session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
        )
        try:
            try:
                job = await self.submit(session, fields, files)
            except SplashError as error:
                if error.status == 404:
                    await self._check_video_server(session, error)
                raise
            job_id = job["id"]
            try:
                return await self._follow(
                    session, job, fields, on_progress, is_interrupted, on_preview
                )
            except SplashError:
                raise  # the server ended or forgot the job
            except BaseException:
                await self._forget(session, job_id)
                raise
        finally:
            if self._session is None:
                await session.close()

    async def _check_video_server(self, session, error: SplashError):
        """After a 404 from POST /v1/videos: SplashError("not_video_server")
        naming what the server's /v1/models lists, unless a model of it has
        "video" among its output_modalities. A Splash language-model server
        (port 8000 by default) answers POST /v1/videos with 404 not_found."""
        try:
            entries = await self.models(session)
        except SplashError:
            entries = []
        if any(VIDEO_MODALITY in (entry.get("output_modalities") or ()) for entry in entries):
            return
        model = self.serve_model or "MiniMaxAI/MiniMax-H3:<variant>"
        parts = urlsplit(self.base_url)
        port = parts.port or (443 if parts.scheme == "https" else 80)
        raise SplashError(
            "not_video_server",
            f"the server at {self.base_url} serves {_served(entries)}, not an H3 video model: it answers "
            f"POST /v1/videos with HTTP 404 ({error.code}). The video nodes need a Splash video server: stop "
            f"this one and run `splash serve --model {model} --port {port}`, or set base_url to where one runs.",
            error.status,
        ) from error

    async def _forget(self, session, job_id):
        try:
            await asyncio.wait_for(self.delete(session, job_id), DELETE_TIMEOUT)
        except BaseException:  # best effort; the original exception wins
            pass

    async def _follow(self, session, job, fields, on_progress, is_interrupted, on_preview):
        job_id = job["id"]
        want_preview = bool(fields.get("preview")) and on_preview is not None
        previewed = False
        while True:
            if on_progress is not None:
                on_progress(job)
            status = job.get("status")
            if status == "completed":
                return await self.content(session, job_id, "video")
            if status == "failed":
                error = job.get("error") or {}
                raise SplashError(
                    str(error.get("code") or "video_failed"),
                    str(error.get("message") or "the job failed"),
                )
            if want_preview and not previewed and job.get("preview"):
                previewed = True
                try:
                    on_preview(await self.content(session, job_id, "preview"))
                except SplashError:
                    pass  # the preview is cosmetic
            deadline = time.monotonic() + self.poll_interval
            while True:
                if is_interrupted is not None and is_interrupted():
                    raise SplashInterrupted(f"job {job_id} interrupted")
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                await asyncio.sleep(min(left, 0.1))
            job = await self.get(session, job_id)

    async def models(self, session) -> list:
        """The entries of the server's /v1/models."""
        answer = await self._request(session, "GET", "/v1/models")
        data = answer.get("data") if isinstance(answer, dict) else None
        return [entry for entry in data or () if isinstance(entry, dict)]


def _served(entries) -> str:
    """The models of a /v1/models listing with their output modalities."""
    return ", ".join(
        f"{entry.get('id')} ({'/'.join(entry.get('output_modalities') or ()) or 'no output modality'})"
        for entry in entries
    ) or "no model it lists"


async def probe_ready(base_url: str, api_key: str = "", timeout: float = 2.0) -> str:
    """The state of a server's /ready: "ready" (200), "starting" (503 with a
    JSON body, as a Splash server answers until it accepts requests; or no
    answer in time) or "refused" (nothing listening). Any other answer
    raises SplashError: 401/403 about the api key, else "not_splash"."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(base_url.rstrip("/") + "/ready", headers=headers) as response:
                if response.status == 200:
                    return "ready"
                if response.status == 503 and response.content_type == "application/json":
                    return "starting"
                error = await _error_of(response)
    except aiohttp.ClientConnectorError:
        return "refused"
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return "starting"
    base_url = base_url.rstrip("/")
    if error.status in (401, 403):
        raise SplashError(
            error.code,
            f"the Splash server at {base_url} refused GET /ready with HTTP {error.status} "
            f"({error.message}): set the node's api_key to the server's --api-key.",
            error.status,
        )
    raise SplashError(
        "not_splash",
        f"the program at {base_url} is not a Splash server: it answers GET /ready with HTTP "
        f"{error.status} (a Splash server answers 200, or 503 while it starts). Stop that program, "
        "or set base_url to the address of a Splash server.",
        error.status,
    )

"""Attach an approved Hermes visible check to a patch in an isolated checkout."""

from collections import OrderedDict
import ipaddress
import json
from pathlib import Path
import subprocess
from threading import Lock
from urllib.parse import urlsplit

CHECK = "run_visible_tests"


def loopback_endpoint(value):
    if not isinstance(value, str):
        return None
    try:
        url = urlsplit(value.rstrip("/"))
        if (
            url.scheme == "http"
            and url.hostname
            and url.port
            and url.path == "/v1"
            and not (url.username or url.password or url.query or url.fragment)
            and (
                url.hostname == "localhost"
                or ipaddress.ip_address(url.hostname).is_loopback
            )
        ):
            return url.geturl()
    except ValueError:
        pass
    return None


class PatchFeedback:
    """Only offer automatic checks in the configured route, session and checkout."""

    def __init__(self, endpoint, model, workspace, cwd, check):
        self.endpoint = loopback_endpoint(endpoint)
        self.model = model if isinstance(model, str) and model else None
        self.workspace = (
            Path(workspace).resolve()
            if isinstance(workspace, str) and Path(workspace).is_absolute()
            else None
        )
        self.cwd = cwd
        self.check = check
        self.sessions = OrderedDict()
        self.lock = Lock()

    def current_workspace(self):
        try:
            cwd = self.cwd()
            return (
                cwd is not None
                and self.workspace is not None
                and Path(cwd).resolve() == self.workspace
            )
        except (OSError, RuntimeError, ValueError):
            return False

    def request(self, request, session_id="", model="", base_url="", api_mode="", **_):
        if not (
            session_id
            and self.endpoint
            and self.model == model
            and loopback_endpoint(base_url) == self.endpoint
            and api_mode == "chat_completions"
            and self.current_workspace()
            and any(
                isinstance(tool, dict)
                and isinstance(tool.get("function"), dict)
                and tool["function"].get("name") == CHECK
                for tool in request.get("tools") or ()
            )
        ):
            self.end(session_id)
            return
        with self.lock:
            self.sessions[session_id] = None
            self.sessions.move_to_end(session_id)
            if len(self.sessions) > 64:
                self.sessions.popitem(last=False)

    def paths_are_local(self, files):
        try:
            return (
                isinstance(files, list)
                and bool(files)
                and all(
                    isinstance(path, str)
                    and Path(path).is_absolute()
                    and Path(path).resolve(strict=True).is_relative_to(self.workspace)
                    for path in files
                )
            )
        except (OSError, RuntimeError, ValueError):
            return False

    def after_tool(self, tool_name="", result=None, session_id="", task_id=None, **_):
        if (
            tool_name != "patch"
            or not isinstance(result, str)
            or not self.current_workspace()
        ):
            return None
        with self.lock:
            if session_id not in self.sessions:
                return None
        try:
            payload = json.loads(result)
        except ValueError:
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("error")
            or not self.paths_are_local(payload.get("files_modified"))
        ):
            return None
        try:
            feedback = self.check(session_id, task_id)
        except (OSError, ValueError, subprocess.SubprocessError):
            feedback = {
                "error": "The visible check did not complete. Run it separately."
            }
        if not isinstance(feedback, dict):
            return None
        return json.dumps({**payload, "visible_tests": feedback}, ensure_ascii=False)

    def end(self, session_id="", **_):
        with self.lock:
            self.sessions.pop(session_id, None)


def register(ctx):
    if ctx.get_config("enabled") is not True:
        return
    from agent.runtime_cwd import resolve_context_cwd

    def check(session_id, task_id):
        from model_tools import handle_function_call
        from tools.registry import registry

        entry = registry.get_entry(CHECK)
        if entry is None or entry.toolset != "file" or entry.is_async:
            return None
        result = handle_function_call(
            CHECK, {}, task_id=task_id, session_id=session_id, enabled_tools=[CHECK]
        )
        return json.loads(result) if isinstance(result, str) else None

    policy = PatchFeedback(
        ctx.get_config("endpoint"),
        ctx.get_config("route_model"),
        ctx.get_config("workspace"),
        resolve_context_cwd,
        check,
    )
    ctx.register_middleware("llm_request", policy.request)
    ctx.register_hook("transform_tool_result", policy.after_tool)
    ctx.register_hook("on_session_end", policy.end)

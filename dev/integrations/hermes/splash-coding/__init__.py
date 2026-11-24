"""Optional local Splash policy for approved Hermes coding workspaces."""

import hashlib
import ipaddress
import json
import stat
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit


def local_endpoint(value):
    try:
        url = urlsplit(value.rstrip("/"))
        port = url.port
        host = url.hostname
        if (
            url.scheme != "http"
            or port is None
            or host is None
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path != "/v1"
        ):
            return None
        if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
            return None
    except (AttributeError, ValueError):
        return None
    return url.geturl()


def git_revision(root):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@dataclass
class Session:
    revision: str
    offered: set[Path] = field(default_factory=set)


class CodingPolicy:
    def __init__(self, config, cwd, revision=git_revision):
        self.endpoint = local_endpoint(config.get("endpoint", ""))
        self.model = config.get("route_model")
        self.group = config.get("cache_group")
        root = config.get("workspace")
        self.root = (
            Path(root).resolve()
            if isinstance(root, str) and Path(root).is_absolute()
            else None
        )
        self.share = (
            config.get("shared_prefix") is True
            and isinstance(self.group, str)
            and bool(self.group)
        )
        self.bound = config.get("bounded_reads") is True
        self.prefix = config.get("system_prefix", "")
        if not isinstance(self.prefix, str) or len(self.prefix) > 120_000:
            self.prefix = ""
            self.share = False
        self.cwd = cwd
        self.revision = revision
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.lock = Lock()

    def _workspace_matches(self):
        try:
            cwd = self.cwd()
            return (
                cwd is not None
                and self.root is not None
                and Path(cwd).resolve() == self.root
            )
        except (OSError, RuntimeError, ValueError):
            return False

    def request(self, request, session_id="", model="", base_url="", api_mode="", **_):
        if not (
            session_id
            and (self.share or self.bound)
            and self.endpoint
            and self.model
            and model == self.model
            and local_endpoint(base_url) == self.endpoint
            and api_mode == "chat_completions"
            and self._workspace_matches()
        ):
            self.end(session_id)
            return None

        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            revision = self.revision(self.root)
            if not revision:
                return None
            with self.lock:
                session = self.sessions.setdefault(session_id, Session(revision))
                self.sessions.move_to_end(session_id)
                while len(self.sessions) > 128:
                    self.sessions.popitem(last=False)

        extra = request.get("extra_body")
        messages = request.get("messages")
        if (
            not self.share
            or request.get("cache_salt") is not None
            or (extra is not None and not isinstance(extra, dict))
            or (extra is not None and "cache_salt" in extra)
            or not isinstance(messages, list)
            or not messages
            or not isinstance(messages[0], dict)
            or messages[0].get("role") != "system"
            or not isinstance(messages[0].get("content"), str)
        ):
            return None
        # Hash the ordered wire tool definitions. Leave provider arguments and
        # the existing transcript intact, including all message roles.
        material = json.dumps(
            [
                self.group,
                self.endpoint,
                self.model,
                str(self.root),
                session.revision,
                self.prefix,
                request.get("tools", []),
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        salt = "hermes-repo-v1-" + hashlib.sha256(material.encode()).hexdigest()
        updated = {**request, "extra_body": {**(extra or {}), "cache_salt": salt}}
        if self.prefix:
            system = messages[0]
            updated["messages"] = [
                {**system, "content": self.prefix + "\n\n" + system["content"]},
                *messages[1:],
            ]
        return {
            "request": updated,
            "source": "splash-coding",
            "reason": "workspace cache scope",
        }

    def before_tool(self, tool_name, args, session_id="", **_):
        if not self.bound or tool_name != "read_file" or not self._workspace_matches():
            return None
        with self.lock:
            if session_id not in self.sessions:
                return None
        path = args.get("path")
        offset = args.get("offset", 1)
        limit = args.get("limit")
        if (
            not isinstance(path, str)
            or not path
            or type(offset) is not int
            or offset != 1
        ):
            return None
        if limit is not None and (type(limit) is not int or limit <= 120):
            return None
        try:
            candidate = (self.root / path).resolve(strict=True)
            if not candidate.is_relative_to(self.root):
                return None
            info = candidate.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size < 12_000:
                return None
        except (OSError, ValueError, RuntimeError):
            return None

        with self.lock:
            session = self.sessions.get(session_id)
            if (
                session is None
                or candidate in session.offered
                or len(session.offered) >= 64
            ):
                return None
            session.offered.add(candidate)
        return {
            "action": "block",
            "message": (
                "This is a large file. Search for the relevant symbol, then read up to 120 lines "
                "around it using offset and limit. If the excerpt lacks necessary context, a patch "
                "does not match, or you cannot locate the symbol, retry this read or request a wider "
                "range. The next read of this file can proceed normally."
            ),
        }

    def end(self, session_id="", **_):
        with self.lock:
            self.sessions.pop(session_id, None)


def register(ctx):
    from agent.runtime_cwd import resolve_context_cwd

    keys = (
        "endpoint",
        "route_model",
        "cache_group",
        "workspace",
        "shared_prefix",
        "bounded_reads",
        "system_prefix",
    )
    policy = CodingPolicy(
        {key: ctx.get_config(key) for key in keys}, resolve_context_cwd
    )
    if policy.share or policy.bound:
        ctx.register_middleware("llm_request", policy.request)
        ctx.register_hook("pre_tool_call", policy.before_tool)
        ctx.register_hook("on_session_end", policy.end)

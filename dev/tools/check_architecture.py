#!/usr/bin/env python3
"""Enforce production dependency boundaries, and keep Hugging Face tokens out
of CI."""

from __future__ import annotations

import ast
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm", ".metal"}
INCLUDE = re.compile(r'^\s*#\s*(?:include|import)\s*(["<])([^">]+)[">]', re.MULTILINE)
# Tile, split and kernel-configuration names runtime/ops owns: model/ and
# engine/ size workspaces through the operators, never through these.
OPERATOR_WORKSPACE_POLICY_NAMES: tuple[str, ...] = (
    "prefillAttentionTiles",
    "kVerifySplits",
    "kPrefillAttentionTileRows",
    "moeMaximumTiles",
    "LinearTile",
    "LinearConfig",
    "MoeExpertTile",
    "MoeConfig",
)
OPERATOR_WORKSPACE_POLICY = re.compile(
    r"\b(?:" + "|".join(OPERATOR_WORKSPACE_POLICY_NAMES) + r")\b"
)
# The standard library's steady clocks, which count sleep on macOS, and the
# timed waits that measure on them; production measures durations on
# AwakeClock and wall-clock instants on system_clock.
SLEEP_COUNTING_CLOCK = re.compile(
    r"\b(?:steady_clock|high_resolution_clock|wait_for|try_lock_for"
    r"|try_acquire_for)\b"
)
# The variables huggingface_hub reads a token from, and the repository
# secrets a workflow would pass one in.
HUGGING_FACE_TOKEN = re.compile(
    r"\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\b|\bsecrets\s*(?:\.\s*|\[\s*['\"])HF_"
)


def production_sources() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "runtime").rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def include_paths(path: Path, text: str) -> list[str]:
    """Each include of a runtime source as the path below runtime/ the
    compiler reads: a quoted include beside the including file if it is
    there, else on -Iruntime."""
    result = []
    for delimiter, include in INCLUDE.findall(text):
        beside = posixpath.normpath((path.parent / include).as_posix())
        if delimiter == '"' and Path(beside).is_file():
            include = posixpath.relpath(beside, (ROOT / "runtime").as_posix())
        result.append(posixpath.normpath(include))
    return result


def imported_module(call: ast.Call) -> str | None:
    """The module an importlib.import_module or __import__ call names."""
    name = getattr(call.func, "attr", getattr(call.func, "id", None))
    if name in ("import_module", "__import__") and call.args:
        argument = call.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return argument.value
    return None


def check_server_dependencies() -> list[str]:
    # Request bridges and model-output handling must not import HTTP or prompt
    # preparation. The wire transport stays independent of those adapters.
    forbidden = {
        "backend": {"frontend", "api_shapes"},
        "constraints": {"frontend", "backend", "output", "api_shapes"},
        "output": {"frontend", "backend", "constraints", "api_shapes"},
        "runtime": {"frontend", "backend", "constraints", "output", "api_shapes"},
    }
    errors = []
    for path in sorted((ROOT / "server").glob("*.py")):
        if path.stem == "server":
            continue
        blocked = {"server", *forbidden.get(path.stem, ())}
        dependencies = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.module is None or (node.module == "server" and not node.level):
                    dependencies.update(alias.name for alias in node.names)
                else:
                    dependencies.add(node.module.removeprefix("server.").split(".")[0])
            elif isinstance(node, ast.Import):
                dependencies.update(
                    alias.name.removeprefix("server.").split(".")[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.Call) and (module := imported_module(node)):
                dependencies.add(
                    module.lstrip(".").removeprefix("server.").split(".")[0]
                )
        for dependency in sorted(dependencies & blocked):
            errors.append(f"{relative(path)}: imports upper serving layer {dependency}")
    return errors


def check_package_imports() -> list[str]:
    # The server and the installer each import their own modules one way,
    # relatively, never by their package's name. The installer's script
    # entry points name their package in a PEP 366 header, their one use of
    # __package__.
    header = ast.dump(
        ast.parse('__name__ == "__main__" and not __package__', mode="eval").body
    )
    entry_points = {"install/launcher.py", "install/models.py"}
    errors = []
    for package in ("server", "install"):
        paths = sorted((ROOT / package).glob("*.py"))
        own_names = {package, *(path.stem for path in paths)}
        for path in paths:
            name = relative(path)
            tree = ast.parse(path.read_text())
            allowed = set()
            if name in entry_points:
                allowed = {
                    id(node)
                    for statement in tree.body
                    if isinstance(statement, ast.If)
                    and ast.dump(statement.test) == header
                    for node in ast.walk(statement)
                }
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and not node.level:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    modules = []
                for module in modules:
                    if module.split(".")[0] in own_names:
                        errors.append(
                            f"{name}: imports {module} without a relative import"
                        )
                if (
                    isinstance(node, ast.Name)
                    and node.id == "__package__"
                    and id(node) not in allowed
                ):
                    errors.append(f"{name}: reads __package__")
    return errors


def check_workflows() -> list[str]:
    # CI installs public models alone, which need no token, and a token a
    # workflow passes reaches every program its job runs.
    return [
        f"{relative(path)}: passes a Hugging Face token ({match.group()})"
        for path in sorted((ROOT / ".github/workflows").glob("*.y*ml"))
        if (match := HUGGING_FACE_TOKEN.search(path.read_text()))
    ]


def check() -> list[str]:
    errors = check_server_dependencies() + check_package_imports() + check_workflows()
    forbidden_metal_dependencies = ("engine/", "model/", "ops/")
    forbidden_model_dependencies = ("engine/",)
    forbidden_operator_dependencies = ("engine/", "model/")
    engine_assembly_sources = {
        "runtime/engine/Bootstrap.hpp",
        "runtime/engine/Bootstrap.mm",
        "runtime/engine/RuntimeResources.hpp",
        "runtime/engine/RuntimeResources.mm",
    }
    concrete_model_headers = (
        "model/DFlashDraft.hpp",
        "model/ModelFactory.hpp",
        "model/Qwen3_6Moe.hpp",
        "model/Qwen3_8.hpp",
        "model/QwenHybridLayout.hpp",
        "model/QwenState.hpp",
        "model/QwenTarget.hpp",
        "model/QwenTargetFiles.hpp",
        "model/QwenTargetLoader.hpp",
        "model/Runtime.hpp",
        "model/WeightStore.hpp",
    )
    concrete_model_symbols = re.compile(r"\bmodel::(?:Qwen\w*|DFlash\w*|Runtime)\b")
    client_names = re.compile(r"\b(?:Claude Code|OpenCode|Codex|Hermes)\b", re.I)

    for path in production_sources():
        name = relative(path)
        text = path.read_text(errors="replace")
        includes = include_paths(path, text)
        for include in includes:
            if include.startswith("tests/"):
                errors.append(f"{name}: production depends on test code {include}")
        if name.startswith("runtime/metal/"):
            for include in includes:
                if include.startswith(forbidden_metal_dependencies):
                    errors.append(
                        f"{name}: Metal depends on production layer {include}"
                    )
        if name.startswith("runtime/model/"):
            for include in includes:
                if include.startswith(forbidden_model_dependencies):
                    errors.append(f"{name}: model depends on engine layer {include}")
            if re.search(r"\b(?:graph|commandGraph)\.add\s*\(", text):
                errors.append(f"{name}: model dispatches a Metal pipeline directly")
        if name.startswith(("runtime/model/", "runtime/engine/")):
            if OPERATOR_WORKSPACE_POLICY.search(text):
                layer = name.split("/")[1]
                errors.append(f"{name}: {layer} owns an operator workspace policy")
        if name.startswith("runtime/ops/"):
            for include in includes:
                if include.startswith(forbidden_operator_dependencies):
                    errors.append(
                        f"{name}: operator depends on upper production layer {include}"
                    )
        if name.startswith("runtime/engine/") and name not in engine_assembly_sources:
            for include in includes:
                if include in concrete_model_headers:
                    errors.append(
                        f"{name}: engine policy depends on concrete model {include}"
                    )
        if name == "runtime/main.mm":
            for include in includes:
                if include in concrete_model_headers:
                    errors.append(
                        f"{name}: startup depends on concrete model {include}"
                    )
            if concrete_model_symbols.search(text):
                errors.append(f"{name}: startup names a concrete model type")
        if name.startswith("runtime/engine/") and re.search(
            r"^namespace splash\s*\{", text, re.MULTILINE
        ):
            errors.append(f"{name}: engine declarations leak into root namespace")
        if client_names.search(text):
            errors.append(
                f"{name}: production backend contains client-specific behavior"
            )
        # Every timeout and duration counts time the Mac is awake, as the
        # server's time.monotonic() does.
        if SLEEP_COUNTING_CLOCK.search(text):
            errors.append(f"{name}: measures time on a clock that counts sleep")
    return errors


def stale_policy_names() -> list[str]:
    """The guarded operator policy names no source under runtime/ops uses,
    which the rule would keep guarding for nothing."""
    operators = [
        path.read_text(errors="replace")
        for path in production_sources()
        if relative(path).startswith("runtime/ops/")
    ]
    return [
        name
        for name in OPERATOR_WORKSPACE_POLICY_NAMES
        if not any(re.search(rf"\b{name}\b", text) for text in operators)
    ]


def main() -> int:
    errors = check() + [
        f"rule guards vanished operator policy symbol {name}"
        for name in stale_policy_names()
    ]
    if errors:
        for error in errors:
            print(f"architecture error: {error}", file=sys.stderr)
        return 1
    print("backend architecture: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

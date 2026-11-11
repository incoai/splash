#!/usr/bin/env python3
"""Enforce production dependency boundaries, keep notes that live outside the
repository out of it, bound the engine's size, and keep Hugging Face tokens
out of CI."""

from __future__ import annotations

import ast
import io
import posixpath
import re
import subprocess
import sys
import tokenize
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
# Residency sets attached to a command queue or a command buffer: Metal makes
# all of one resident before each command buffer starts, so every command
# would wait for memory it does not use. A command makes resident what its
# encoders use (MetalBackend::setTableReach, AddressTable).
ATTACHED_RESIDENCY_SET = re.compile(r"\b(?:addResidencySets?|useResidencySets?)\b")
# A GPU address written into a table: the backend declares the memory a table
# reaches only as its writer names it (MetalBackend::setTableReach), so page
# storage is the one production writer of addresses.
GPU_ADDRESS = re.compile(r"\bgpuAddress\s*\(\s*\)")
GPU_ADDRESS_WRITERS = frozenset({"runtime/ops/PageStorage.cpp"})
# The variables huggingface_hub reads a token from, and the repository
# secrets a workflow would pass one in.
HUGGING_FACE_TOKEN = re.compile(
    r"\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\b|\bsecrets\s*(?:\.\s*|\[\s*['\"])HF_"
)
# Engine files that assemble concrete models or parse their options: they may
# name models and mechanisms. Every other engine file but the status writer
# (ENGINE_REPORTING, ruled in check()) is policy.
ENGINE_ASSEMBLY = frozenset(
    {
        "runtime/engine/Bootstrap.hpp",
        "runtime/engine/Bootstrap.mm",
        "runtime/engine/RuntimeResources.hpp",
        "runtime/engine/RuntimeResources.mm",
        "runtime/engine/StreamedMemoryPlan.hpp",
        "runtime/engine/StreamedMemoryPlan.cpp",
        "runtime/engine/NativeArguments.hpp",
        "runtime/engine/NativeArguments.cpp",
    }
)
ENGINE_REPORTING = frozenset({"runtime/engine/Status.hpp", "runtime/engine/Status.cpp"})
# Identifier word parts (identifier_parts): engine policy decides from what
# the model reports (costs, pools, shape), never from which model
# (MODEL_FAMILY_PARTS, below) or mechanism serves it. An identifier names a
# mechanism when it holds every word of one of these, in any order
# (names_mechanism).
MODEL_MECHANISM_WORDS = (
    ("expert",),
    ("indexer",),
    ("ple",),
    ("history", "stream"),
    ("history", "staging"),
    ("block", "cache"),
    ("sparse", "row"),
    ("hyper", "connection"),
    ("values", "token", "major"),
)
# The model headers engine policy may include: the boundary, the types it
# exposes, and the storage the KV tier uses.
ENGINE_MODEL_BOUNDARY = frozenset(
    {
        "model/Model.hpp",
        "model/CommandCost.hpp",
        "model/ElasticPool.hpp",
        "model/LatencyHistogram.hpp",
        "model/SlotFile.hpp",
        "model/SsdReaders.hpp",
        "model/KvCopyRead.hpp",
        "model/StateTransfer.hpp",
        "model/WeightMemory.hpp",
    }
)
# The concrete models' headers: their names hold the words a model family
# goes by (family_words), and startup and the status writer name none of the
# types they define (concrete_model_types).
CONCRETE_MODEL_HEADERS = (
    "model/DFlashDraft.hpp",
    "model/Flint.hpp",
    "model/ModelFactory.hpp",
    "model/Qwen3_6Moe.hpp",
    "model/Qwen3_8.hpp",
    "model/Qwen4Exp.hpp",
    "model/Qwen4ExpTarget.hpp",
    "model/Qwen4Steps.hpp",
    "model/QwenHybridLayout.hpp",
    "model/QwenState.hpp",
    "model/QwenTarget.hpp",
    "model/QwenTargetFiles.hpp",
    "model/QwenTargetLoader.hpp",
    "model/Runtime.hpp",
    "model/WeightStore.hpp",
)
# The layers, top down: each layer's sources (a directory, a file, or with
# "/*" the files directly in a directory) and the layers they may include or
# import besides their own. A source is in the layer of the most specific
# member that holds it, so a file listed in a layer leaves its directory's.
# The frontend reaches the engine through the wire protocol alone; engine
# assembly, which builds the concrete model and its operators, is the one
# part of the engine that reaches below the model's interface; the interface
# exposes no model internals; ANE programs run on Metal's buffers; kernels
# are the shaders and their ABI headers.
LAYERS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("frontend", ("server/", "splash", "install/", "app/"), ()),
    ("startup", ("runtime/main.mm",), ("engine assembly", "engine", "shared")),
    (
        "engine assembly",
        tuple(sorted(ENGINE_ASSEMBLY)),
        ("engine", "model interface", "model", "ops", "metal", "kernels", "shared"),
    ),
    ("engine", ("runtime/engine/",), ("model interface", "shared")),
    (
        "model interface",
        tuple(sorted(f"runtime/{header}" for header in ENGINE_MODEL_BOUNDARY)),
        ("ops", "metal", "kernels", "shared"),
    ),
    (
        "model",
        ("runtime/model/",),
        ("model interface", "ops", "metal", "kernels", "shared"),
    ),
    ("ops", ("runtime/ops/",), ("ane", "metal", "kernels", "shared")),
    ("ane", ("runtime/ane/",), ("metal", "shared")),
    ("metal", ("runtime/metal/",), ("kernels", "shared")),
    ("kernels", ("runtime/metal/kernels/", "runtime/metal/abi/"), ()),
    ("shared", ("runtime/*",), ()),
)
# The includes that cross a layer today, as (source, include): this list only
# shrinks. An entry goes when the dependency goes, not only its include (an
# interface header may still bring the same header in), and an entry whose
# include is gone or no longer crosses a layer is an error.
LAYER_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # Types the model interface exposes from below it: model/Model.hpp
        # includes ops/PagedKv.hpp and ops/Vision.hpp, ElasticPool.hpp and
        # WeightMemory.hpp include metal/Allocation.hpp.
        ("runtime/engine/DiskLabels.hpp", "ops/PagedKv.hpp"),
        ("runtime/engine/Engine.hpp", "ops/Vision.hpp"),
        ("runtime/engine/KvCache.hpp", "ops/PagedKv.hpp"),
        ("runtime/engine/KvPool.hpp", "ops/PagedKv.hpp"),
        ("runtime/engine/MemoryLedger.hpp", "ops/PagedKv.hpp"),
        ("runtime/engine/MemoryPlan.hpp", "ops/PagedKv.hpp"),
        ("runtime/engine/MemoryControl.hpp", "metal/MetalBackend.hpp"),
        ("runtime/engine/MemoryGovernor.hpp", "metal/MetalBackend.hpp"),
        ("runtime/engine/NativeRuntime.cpp", "metal/MetalBackend.hpp"),
        ("runtime/engine/WeightResidency.hpp", "metal/Allocation.hpp"),
        # Operator and device facts the engine reads past the interface.
        ("runtime/engine/AneFfnStartup.cpp", "ane/Program.hpp"),
        ("runtime/engine/AneFfnStartup.hpp", "ops/AneFfn.hpp"),
        ("runtime/engine/AneFfnStartup.hpp", "ops/AneFfnCalibration.hpp"),
        ("runtime/engine/KvPageTier.hpp", "ops/KvSlotFormat.hpp"),
        ("runtime/engine/KvPageTier.hpp", "ops/PageStorage.hpp"),
        ("runtime/engine/MemoryPlan.hpp", "metal/DeviceCapabilities.hpp"),
        ("runtime/engine/MemoryPlan.hpp", "ops/GDN.hpp"),
        # The status writer renders model, device and assembly telemetry.
        ("runtime/engine/Status.cpp", "engine/RuntimeResources.hpp"),
        ("runtime/engine/Status.hpp", "metal/MetalBackend.hpp"),
        ("runtime/engine/Status.hpp", "model/ExpertStoreStats.hpp"),
        ("runtime/engine/Status.hpp", "model/KvReadStats.hpp"),
        ("runtime/engine/Status.hpp", "model/SparseRowStore.hpp"),
        # Startup reads the model's descriptor past engine assembly.
        ("runtime/main.mm", "model/ModelDescriptor.hpp"),
    }
)
# A runtime source a frontend file names outside its comments, as a path to
# read or build from.
RUNTIME_PATH = re.compile(r"(?<![\w.-])runtime/[\w./-]*\w")
# References to notes kept outside the repository: amendments, design drafts,
# attempt logs, reports of a merge and rulings stamped with a time, the stamp
# a few words after whose ruling it is. Code, comments and docs carry their
# own reasons.
OUTSIDE_NOTE = re.compile(
    r"\bamend/|\bDESIGN(?:-\w|[ ]§|[ ]section\b)|\bAMENDMENTS\b|\bATTEMPTS\b"
    r"|(?i:\bmerge[ ]report)"
    r"|\b(?i:owner|coordinator)(?:'s)?(?:[^\W\d_]|[ ]){1,12}?\(?"
    r"(?:\d{1,2}-\d{2}[ ])?\d{1,2}:\d{2}\b"
)
# A path in prose (comments, docstrings, docs) whose first directory is the
# repository's, at its root or an include root: it names a file that exists.
PROSE_PATH = re.compile(
    r"(?<![\w./~:@$-])((?:\.\.?/)*(?:[A-Za-z_][\w.-]*/)+[\w.-]*\w\."
    r"(?:md|py|hpp|h|cpp|cc|c|mm|m|metal|swift|json|jsonl|txt|sh|mk|yml|toml"
    r"|js|css|html|jinja|tsv))(?![\w/])"
)
# The files that hold such references today, and how many: the list only
# shrinks, a file above or below its count is an error, and the check is
# strict once the list is empty.
NOTE_ALLOWANCE: dict[str, int] = {
    "dev/tests/engine/TimingProfile.cpp": 6,
    "dev/tests/engine/native_loop_memory_test.cpp": 1,
    "dev/tests/engine/pressure_serving_test.mm": 3,
    "dev/tests/engine/q8_page_storage_test.mm": 1,
    "dev/tests/engine/qwen_state_storage_test.mm": 1,
    "dev/tests/engine/runtime_resources_test.mm": 1,
    "dev/tests/engine/timing_profile_test.cpp": 1,
}
# runtime/engine's production lines: the budget only goes down, unless the
# owner raises it.
ENGINE_LINE_BUDGET = 30_105
C_LIKE_SUFFIXES = SOURCE_SUFFIXES | {".swift"}
HASH_COMMENT_SUFFIXES = {"", ".sh", ".bash", ".fish", ".mk", ".yml", ".yaml", ".toml"}
C_LIKE_COMMENT = re.compile(
    r"\"(?:\\.|[^\"\\\n])*\"|(?<!\w)'(?:\\.|[^'\\\n])*'|(//[^\n]*|/\*.*?\*/)", re.S
)
HASH_COMMENT = re.compile(r"(?:^|(?<=\s))#[^\n]*", re.M)
# Comments and string and character literals, which the identifier rules do
# not read; a quote after a word character separates digits.
NOT_CODE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|(?<!\w)'(?:\\.|[^'\\\n])*'", re.S
)
IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*")
# A word of an identifier: an uppercase run before a capitalized word, a
# capitalized or lowercase word, an uppercase run, or a number.
IDENTIFIER_PART = re.compile(r"[A-Z]{2,}(?=[A-Z][a-z])|[A-Z]*[a-z]+|[A-Z]+|\d+")


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


def holds(member: str, path: str) -> bool:
    """Whether a layer's member holds a repository path: the file itself, a
    directory and what it contains, or with "/*" a file directly in the
    directory that exists, since a standard header's name (<cstdint>) would
    look like one."""
    if member.endswith("/*"):
        return posixpath.dirname(path) == member[:-2] and (ROOT / path).is_file()
    return path in (member, member.rstrip("/")) or (
        member.endswith("/") and path.startswith(member)
    )


def layer_of(path: str) -> str | None:
    """The layer of the most specific, longest, member that holds a path."""
    return max(
        (
            (len(member), layer)
            for layer, members, _ in LAYERS
            for member in members
            if holds(member, path)
        ),
        default=(0, None),
    )[1]


def crossing(name: str, dependency: str) -> str | None:
    """The layer a source's dependency, as a repository path, is on when the
    source's own layer may not depend on it."""
    layer, target = layer_of(name), layer_of(dependency)
    edges = next((edges for each, _, edges in LAYERS if each == layer), ())
    if target is None or target in (layer, *edges):
        return None
    return target


def check_layers(name: str, dependencies: list[str]) -> list[str]:
    """A source's dependencies, as repository paths, on layers its own may not
    depend on, besides the exceptions; a source in no layer is one error."""
    layer = layer_of(name)
    if layer is None:
        return [f"{name}: belongs to no layer"]
    return [
        f"{name}: {layer} depends on {target} ({dependency.removeprefix('runtime/')})"
        for dependency in dependencies
        if (target := crossing(name, dependency)) is not None
        and (name, dependency.removeprefix("runtime/")) not in LAYER_EXCEPTIONS
    ]


def stale_layer_exceptions() -> list[tuple[str, str]]:
    """The layer exceptions whose source no longer includes the header, or
    whose include no longer crosses a layer."""
    return sorted(
        (source, include)
        for source, include in LAYER_EXCEPTIONS
        if not (ROOT / source).is_file()
        or include not in include_paths(ROOT / source, (ROOT / source).read_text())
        or crossing(source, f"runtime/{include}") is None
    )


def split_prose(name: str, text: str) -> tuple[str, str]:
    """A file's code, and its prose: comments, Python docstrings, and the
    whole of a Markdown document. Data files have no prose."""
    suffix = Path(name).suffix
    if suffix == ".md":
        return "", text
    if suffix == ".py":
        code, prose = [], []
        statement_start = True
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT or (
                token.type == tokenize.STRING and statement_start
            ):
                prose.append(token.string)
            else:
                code.append(token.string)
            if token.type not in (tokenize.NL, tokenize.COMMENT):
                statement_start = token.type in (
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                )
        return "\n".join(code), "\n".join(prose)
    if suffix in C_LIKE_SUFFIXES:
        comments = [match for match in C_LIKE_COMMENT.finditer(text) if match[1]]
    elif suffix in HASH_COMMENT_SUFFIXES:
        comments = list(HASH_COMMENT.finditer(text))
    else:
        return text, ""
    code, start = [], 0
    for comment in comments:
        code.append(text[start : comment.start()])
        start = comment.end()
    code.append(text[start:])
    return " ".join(code), "\n".join(comment.group() for comment in comments)


def repository_files() -> list[str]:
    """The checkout's files that git does not ignore, which leaves out builds,
    downloaded models and tokens."""
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return sorted(name for name in listing.split("\0") if (ROOT / name).is_file())


def note_references(
    name: str, text: str, files: set[str], directories: set[str]
) -> list[str]:
    """A file's references to notes outside the repository: their names
    anywhere, and paths in its prose that name no repository file."""
    found = [
        f"refers to a note outside the repository ({match.group()})"
        for match in OUTSIDE_NOTE.finditer(text)
    ]
    _, prose = split_prose(name, text)
    # The checkout's root and the include roots (-Iruntime, -Idev).
    bases = ("", "runtime/", "dev/")
    here = posixpath.dirname(name)
    for match in PROSE_PATH.finditer(prose):
        path = match[1]
        if OUTSIDE_NOTE.search(path):
            continue
        candidates = [posixpath.normpath(posixpath.join(here, path))]
        if not path.startswith("."):
            # A document is the repository's wherever it is named; another
            # file is when its first directory is.
            if not path.endswith(".md") and not any(
                base + path.split("/")[0] in directories for base in bases
            ):
                continue
            candidates += [posixpath.normpath(base + path) for base in bases]
        if not any(each in files or each in directories for each in candidates):
            found.append(f"names a missing repository file ({path})")
    return found


def check_note_references() -> list[str]:
    """Every file's references to notes outside the repository beyond its
    allowance, and allowances above what their files hold."""
    files = repository_files()
    known = set(files)
    directories = {
        parent.as_posix()
        for name in files
        for parent in Path(name).parents
        if parent != Path(".")
    }
    errors = []
    for name in files:
        try:
            text = (ROOT / name).read_text()
        except UnicodeDecodeError:
            continue
        found = note_references(name, text, known, directories)
        allowance = NOTE_ALLOWANCE.get(name, 0)
        if len(found) > allowance:
            errors += [f"{name}: {reference}" for reference in found]
        elif len(found) < allowance:
            errors.append(
                f"{name}: holds {len(found)} references to notes outside the "
                f"repository, its allowance {allowance}: lower it"
            )
    errors += [
        f"{name}: has a note allowance but is not in the repository"
        for name in sorted(set(NOTE_ALLOWANCE) - set(files))
    ]
    return errors


def engine_lines() -> int:
    """runtime/engine's production lines."""
    return sum(
        len(path.read_text(errors="replace").splitlines())
        for path in production_sources()
        if relative(path).startswith("runtime/engine/")
    )


def check_engine_size(lines: int) -> list[str]:
    """runtime/engine's lines over its budget."""
    if lines <= ENGINE_LINE_BUDGET:
        return []
    return [
        f"runtime/engine holds {lines} production lines, over its budget "
        f"{ENGINE_LINE_BUDGET}"
    ]


def identifier_parts(identifier: str) -> list[str]:
    """The lower-cased words of an identifier, split at underscores, numbers
    and case changes. An uppercase run before a capitalized word is a word of
    its own (PLERows: ple, rows); a single capital is the start of the word
    after it (DFlashDraft: dflash, draft)."""
    return [part.lower() for part in IDENTIFIER_PART.findall(identifier)]


def family_names(path: Path) -> list[tuple[str, str]]:
    """The name and the draft's repository of each family the installer
    serves (FAMILIES in install/families.py)."""
    families = next(
        node.value
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "FAMILIES" for target in node.targets)
    )
    return [(family.args[0].value, family.args[1].value) for family in families.elts]


def family_words(
    families: list[tuple[str, str]], headers: tuple[str, ...]
) -> frozenset[str]:
    """The words a family goes by: the first word of a family's name and of
    its draft's repository after the owner, the model's brand (Qwen, Ornith,
    Flint), and those of either that a concrete model header's name holds
    too (DFlash). Numbers are not words, and a name's other words (Flash,
    Next, a size, the repository's owner) are no family's."""
    header_words = {
        part for header in headers for part in identifier_parts(Path(header).stem)
    }

    def words(name: str) -> list[str]:
        return [part for part in identifier_parts(name) if part.isalpha()]

    return frozenset(
        part
        for family, repository in families
        for name in (words(family), words(repository.rpartition("/")[2]))
        for part in name
        if part == name[0] or part in header_words
    )


MODEL_FAMILY_PARTS = family_words(
    family_names(ROOT / "install/families.py"), CONCRETE_MODEL_HEADERS
)
TYPE_DEFINITION = re.compile(
    r"^\s*(?:class|struct|enum\s+class)\s+(\w+)(?:\s+final)?\s*[:{]", re.MULTILINE
)
MODEL_SYMBOL = re.compile(r"\bmodel::(\w+)")


def concrete_model_types(headers: tuple[str, ...]) -> frozenset[str]:
    """The types the concrete model headers define, nested ones included."""
    return frozenset(
        symbol
        for header in headers
        if (path := ROOT / "runtime" / header).is_file()
        for symbol in TYPE_DEFINITION.findall(path.read_text())
    )


CONCRETE_MODEL_TYPES = concrete_model_types(CONCRETE_MODEL_HEADERS)


def singular(word: str) -> str:
    """A word without a trailing s, so that a plural names what its singular
    does."""
    return word.removesuffix("s")


def names_mechanism(parts: list[str]) -> bool:
    """Whether an identifier of these words holds every word of a model
    mechanism, in any order, singular or plural."""
    words = {singular(part) for part in parts}
    return any(
        all(singular(word) in words for word in mechanism)
        for mechanism in MODEL_MECHANISM_WORDS
    )


def code_identifiers(text: str) -> list[str]:
    """The identifiers of a source, without its comments, literals and
    include paths."""
    return IDENTIFIER.findall(NOT_CODE.sub(" ", INCLUDE.sub(" ", text)))


def check_model_names(
    name: str, text: str, role: str = "engine policy", mechanisms: bool = True
) -> list[str]:
    """A source's identifiers that name a model family, or with mechanisms a
    model mechanism, which its role may not."""
    errors = []
    for identifier in dict.fromkeys(code_identifiers(text)):
        parts = identifier_parts(identifier)
        if MODEL_FAMILY_PARTS.intersection(parts):
            errors.append(f"{name}: {role} names a model family ({identifier})")
        elif mechanisms and names_mechanism(parts):
            errors.append(f"{name}: {role} names a model mechanism ({identifier})")
    return errors


def check_concrete_types(name: str, text: str, role: str) -> list[str]:
    """The concrete model types a source names in the model's namespace, those
    that name a family aside (check_model_names reports them)."""
    return [
        f"{name}: {role} names a concrete model type ({symbol})"
        for symbol in dict.fromkeys(MODEL_SYMBOL.findall(NOT_CODE.sub(" ", text)))
        if symbol in CONCRETE_MODEL_TYPES
        and not MODEL_FAMILY_PARTS.intersection(identifier_parts(symbol))
    ]


def imported_module(call: ast.Call) -> str | None:
    """The module an importlib.import_module or __import__ call names."""
    name = getattr(call.func, "attr", getattr(call.func, "id", None))
    if name in ("import_module", "__import__") and call.args:
        argument = call.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return argument.value
    return None


def absolute_module(name: str, package: str) -> str:
    """The full name of a module that a module of package names, relatively
    or not."""
    level = len(name) - len(name.lstrip("."))
    if not level:
        return name
    base = package.rsplit(".", level - 1)[0]
    return f"{base}.{name[level:]}" if name[level:] else base


def imported_modules(tree: ast.AST, package: str) -> set[str]:
    """The full names of the modules a module of package imports. A name
    imported from a module may be a submodule of it, so it counts as one."""
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = absolute_module("." * node.level + (node.module or ""), package)
            modules.add(base)
            modules.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and (module := imported_module(node)):
            modules.add(absolute_module(module, package))
    return modules


def check_server_dependencies() -> list[str]:
    # Request bridges and model-output handling must not import HTTP, prompt
    # preparation or the dialects that render answers. The wire transport
    # stays independent of those adapters. The dialects (server/dialects)
    # read a request and render a generation the handler collects, and reach
    # the engine through nothing else. Each API has one owner: a dialect
    # imports no other dialect, and frontend and chat_messages, which every
    # dialect reaches, import none.
    upper = {"frontend", "chat_messages", "dialects"}
    forbidden = {
        "backend": upper,
        "constraints": {*upper, "backend", "output"},
        "output": {*upper, "backend", "constraints"},
        "runtime": {*upper, "backend", "constraints", "output"},
        "frontend": {"dialects"},
        "chat_messages": {"frontend", "dialects"},
        "dialects": {"frontend", "backend", "runtime"},
    }
    errors = []
    for path in sorted((ROOT / "server").rglob("*.py")):
        # A module of a subpackage is held to its subpackage's rule.
        layer = path.relative_to(ROOT / "server").with_suffix("").parts[0]
        if layer == "server":
            continue
        blocked = {"server", *forbidden.get(layer, ())}
        package = ".".join(path.relative_to(ROOT).parent.parts)
        modules = imported_modules(ast.parse(path.read_text()), package)
        if layer == "dialects" and path.name != "__init__.py":
            dialects = {
                module.split(".")[2]
                for module in modules
                if module.startswith("server.dialects.")
            }
            errors.extend(
                f"{relative(path)}: imports sibling dialect {name}"
                for name in sorted(dialects)
            )
        # With server/ on the path a bare name imports a server module too.
        dependencies = {
            module.removeprefix("server.").split(".")[0]
            for module in modules
            if module != "server"
        }
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
    entry_points = {
        "install/launcher.py",
        "install/models.py",
        "install/local.py",
        "install/cached.py",
        "install/families.py",
    }
    errors = []
    for package in ("server", "install"):
        # A package's subpackages are its own modules too.
        paths = sorted((ROOT / package).rglob("*.py"))
        own_names = {
            package,
            *(
                path.relative_to(ROOT / package).with_suffix("").parts[0]
                for path in paths
            ),
        }
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


def check_frontend() -> list[str]:
    """The includes and named paths into the runtime of the frontend's files
    git does not ignore."""
    errors = []
    for name in repository_files():
        if layer_of(name) != "frontend":
            continue
        path = ROOT / name
        text = path.read_text(errors="replace")
        if path.suffix in SOURCE_SUFFIXES:
            dependencies = [
                f"runtime/{include}" for include in include_paths(path, text)
            ]
        else:
            code, _ = split_prose(name, text)
            dependencies = [match.group() for match in RUNTIME_PATH.finditer(code)]
        errors += check_layers(
            name, [posixpath.normpath(each) for each in dependencies]
        )
    return errors


def check() -> list[str]:
    errors = check_server_dependencies() + check_package_imports() + check_workflows()
    client_names = re.compile(r"\b(?:Claude Code|OpenCode|Codex|Hermes)\b", re.I)

    for path in production_sources():
        name = relative(path)
        text = path.read_text(errors="replace")
        includes = include_paths(path, text)
        for include in includes:
            if include.startswith("tests/"):
                errors.append(f"{name}: production depends on test code {include}")
        errors += check_layers(name, [f"runtime/{include}" for include in includes])
        if name.startswith("runtime/model/"):
            if re.search(r"\b(?:graph|commandGraph)\.add\s*\(", text):
                errors.append(f"{name}: model dispatches a Metal pipeline directly")
        if name.startswith(("runtime/model/", "runtime/engine/")):
            if OPERATOR_WORKSPACE_POLICY.search(text):
                layer = name.split("/")[1]
                errors.append(f"{name}: {layer} owns an operator workspace policy")
        if name.startswith("runtime/engine/") and name not in (
            ENGINE_ASSEMBLY | ENGINE_REPORTING
        ):
            errors.extend(check_model_names(name, text))
        # Startup passes the model's options through and the status writer
        # renders its telemetry: both may name mechanisms, but neither a family
        # nor a concrete model's type.
        if name == "runtime/main.mm" or name in ENGINE_REPORTING:
            role = "startup" if name == "runtime/main.mm" else "status writer"
            errors.extend(check_model_names(name, text, role, mechanisms=False))
            errors.extend(check_concrete_types(name, text, role))
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
        if ATTACHED_RESIDENCY_SET.search(text):
            errors.append(
                f"{name}: attaches a residency set to a queue or command buffer"
            )
        if (
            not name.startswith("runtime/metal/")
            and name not in GPU_ADDRESS_WRITERS
            and GPU_ADDRESS.search(text)
        ):
            errors.append(f"{name}: writes a GPU address outside page storage")
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


def missing_listed_paths() -> list[str]:
    """The paths the rules name, as layer members, the status writer or
    concrete model headers, that are not in the repository."""
    listed = {
        member.removesuffix("/*").rstrip("/")
        for _, members, _ in LAYERS
        for member in members
    }
    listed |= ENGINE_REPORTING
    listed |= {f"runtime/{header}" for header in CONCRETE_MODEL_HEADERS}
    return sorted(path for path in listed if not (ROOT / path).exists())


def main() -> int:
    errors = check() + check_frontend() + check_note_references()
    errors += [
        f"rule guards vanished operator policy symbol {name}"
        for name in stale_policy_names()
    ]
    errors += [
        f"layer exception {source} -> {include} no longer crosses a layer: remove it"
        for source, include in stale_layer_exceptions()
    ]
    errors += [
        f"rule names {path}, which is not in the repository: remove it"
        for path in missing_listed_paths()
    ]
    lines = engine_lines()
    print(f"runtime/engine: {lines} production lines, budget {ENGINE_LINE_BUDGET}")
    errors += check_engine_size(lines)
    if errors:
        for error in errors:
            print(f"architecture error: {error}", file=sys.stderr)
        return 1
    print("backend architecture: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

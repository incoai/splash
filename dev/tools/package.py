#!/usr/bin/env python3
"""Build the runtime-only macOS archive and its Homebrew formula."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20260825/"
    "cpython-3.13.15%2B20260825-aarch64-apple-darwin-install_only_stripped.tar.gz"
)
PYTHON_SHA256 = "149038dd0c194c25d4616d7e42a35f67f2edee96412788f74115819b6a4c8548"
INSTALL_FILES = (
    "__init__.py",
    "launcher.py",
    "clients.py",
    "paths.py",
    "models.py",
    "hub.py",
    "families.py",
    "assembly.py",
    "upstream.py",
    "local.py",
    "cached.py",
    "gguf.py",
    "requirements.txt",
    "suggested-models.tsv",
)
COMPLETION_FILES = (
    "models",
    "_splash",
    "splash.bash",
    "splash.fish",
)
SERVER_FILES = (
    "__init__.py",
    "server.py",
    "backend.py",
    "constraints.py",
    "output.py",
    "frontend.py",
    "chat_templates.py",
    "judgments.py",
    "diagnostics.py",
    "api_shapes.py",
    "llama_cpp_compat.py",
    "chat_format.py",
    "tool_schema.py",
    "tokenization.py",
    "json_codec.py",
    "lru.py",
    "latency.py",
    "metrics.py",
    "errors.py",
    "runtime.py",
    "protocol.py",
    "images.py",
    "documents.py",
    "document_worker.py",
    "http_security.py",
    "connections.py",
    "origins.py",
    "serve_options.py",
    "thinking.py",
    "schema_validation.py",
    "crash_trace.py",
    "chat.html",
    "favicon.svg",
)
CHAT_ASSET_FILES = (
    "markdown-it-15.0.2.min.js",
    "highlight-11.12.0.min.js",
    "github-11.12.0.min.css",
    "github-dark-11.12.0.min.css",
)
# Splash's license and the notices of the third-party code it ships.
LICENSE_FILES = ("LICENSE", "THIRD_PARTY_NOTICES")
# The GitHub repository releases are published from.
REPOSITORY = "incoai/splash"
# The engine's files in build/, and the host-step spin's library beside them
# when the build has one: a Metal compiler that refuses its kernel leaves none
# (Makefile, HOST_STEP_LIB), and the engine then runs awake host steps without
# the spin.
ENGINE_FILES = ("splash", "splash.metallib")
OPTIONAL_ENGINE_FILES = ("host-step.metallib",)
# Imports that exercise the staged runtime's native libraries and Splash's own
# entry modules, run from the runtime's directory.
SMOKE_IMPORTS = (
    "import sys; sys.path.insert(0, '.'); "
    "import tokenizers, llguidance, numpy, PIL, transformers, huggingface_hub, pypdfium2; "
    "import server.server, install.launcher"
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def python_distribution():
    """The pinned standalone Python, downloaded into build/release once and
    checked against PYTHON_SHA256."""
    cached = ROOT / "build/release/python-runtime.tar.gz"
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists() or digest(cached) != PYTHON_SHA256:
        with tempfile.NamedTemporaryFile(dir=cached.parent) as download:
            with urllib.request.urlopen(PYTHON_URL, timeout=60) as response:
                shutil.copyfileobj(response, download)
            download.flush()
            if digest(Path(download.name)) != PYTHON_SHA256:
                raise ValueError("Python distribution checksum mismatch")
            # Linked into place whole, so a concurrent build never reads part of it;
            # the download itself goes away with its temporary file.
            temporary = cached.with_name(f".{Path(download.name).name}")
            os.link(download.name, temporary)
            temporary.replace(cached)
    return cached


def stage_runtime(destination, version):
    """Copy the allowlisted files Splash ships into destination and write its
    release.json."""
    for folder, names in (
        ("install", INSTALL_FILES),
        ("install/completions", COMPLETION_FILES),
        ("server", SERVER_FILES),
        ("server/chat-assets", CHAT_ASSET_FILES),
        ("engine", ENGINE_FILES),
    ):
        (destination / folder).mkdir()
        source = ROOT / ("build" if folder == "engine" else folder)
        for name in names:
            shutil.copy2(source / name, destination / folder / name)
    for name in OPTIONAL_ENGINE_FILES:
        if (ROOT / "build" / name).exists():
            shutil.copy2(ROOT / "build" / name, destination / "engine" / name)
    for name in LICENSE_FILES:
        shutil.copy2(ROOT / name, destination / name)
    (destination / "release.json").write_text(
        json.dumps({"version": version}, indent=2) + "\n"
    )


def build_runtime(destination, version):
    """Build the complete runtime in destination, an empty directory: the
    files stage_runtime copies and the standalone Python with
    install/requirements.txt, smoke-tested in place. The release archive and
    Splash.app (package_app.py) ship this tree as it is: it holds no bytecode
    and no console script, and the runtime, which runs with
    PYTHONDONTWRITEBYTECODE=1, writes nothing into it."""
    distribution = python_distribution()
    stage_runtime(destination, version)
    with tarfile.open(distribution) as python:
        python.extractall(destination, filter="data")
    python = destination / "python/bin/python3"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--no-compile",
            "-r",
            str(destination / "install/requirements.txt"),
        ],
        check=True,
    )
    # Exercise imports and the public entry point from the actual staged
    # runtime, not the developer's virtualenv or source PYTHONPATH, and, as
    # the runtime runs, without writing bytecode.
    subprocess.run(
        [str(python), "-I", "-B", "-c", SMOKE_IMPORTS], cwd=destination, check=True
    )
    # The server's entry point as the launcher starts it, which the
    # import above does not run. Not isolated: -I would ignore PYTHONPATH.
    subprocess.run(
        [str(python), "-B", "-P", "-m", "server.server", "--help"],
        cwd=destination,
        env={**os.environ, "PYTHONPATH": str(destination)},
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [str(python), "-B", str(destination / "install/launcher.py"), "--help"],
        cwd=destination,
        check=True,
    )
    # Only the interpreter stays in python/bin. The other entries are scripts
    # Splash never runs: the distribution's (pip, pydoc, ...) and the console
    # scripts pip installs (hf, tqdm, ...), whose shebangs name this stage.
    for entry in (destination / "python/bin").iterdir():
        if not re.fullmatch(r"python3(\.\d+)?", entry.name):
            entry.unlink()
    # The bytecode the distribution ships and pip's run writes.
    for cache in list(destination.rglob("__pycache__")):
        shutil.rmtree(cache)


def formula(version, url, checksum, macos_min):
    # Inputs are encoded as Ruby double-quoted literals with the backslash,
    # the quote and the interpolation opener escaped, so no path or URL can
    # become executable Ruby; double quotes are what `brew audit` expects.
    def quote(text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("#{", "\\#{")
        return '"' + escaped + '"'

    return f"""class SplashMacOSRequirement < Requirement
  fatal true
  satisfy(build_env: false) {{ OS.mac? && MacOS.full_version >= {quote(macos_min)} }}

  def message
    {quote(f"Splash requires macOS {macos_min} or newer.")}
  end
end

class Splash < Formula
  desc "Local inference engine for Apple silicon, built around the model"
  homepage "https://github.com/{REPOSITORY}"
  url {quote(url)}
  sha256 {quote(checksum)}
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on macos: :tahoe
  depends_on SplashMacOSRequirement

  def install
    libexec.install Dir["*"]
    (bin/"splash").write <<~SH
      #!/bin/sh
      export PYTHONDONTWRITEBYTECODE=1
      exec "#{{opt_libexec}}/python/bin/python3" -u "#{{opt_libexec}}/install/launcher.py" "$@"
    SH
    chmod 0755, bin/"splash"
    zsh_completion.install_symlink libexec/"install/completions/_splash"
    bash_completion.install_symlink libexec/"install/completions/splash.bash" => "splash"
    fish_completion.install_symlink libexec/"install/completions/splash.fish"
  end

  def caveats
    <<~CAVEAT
      Serve a model:
        splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
    CAVEAT
  end

  test do
    assert_match version.to_s, shell_output("#{{bin}}/splash --version")
    assert_match "serve", shell_output("#{{bin}}/splash --help")
  end
end
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--macos-min", required=True, help="minimum macOS, e.g. 26.4")
    parser.add_argument(
        "--url", help="published tarball URL; defaults to GitHub Releases"
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        parser.error("invalid release version")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    name = f"splash-{args.version}-arm64-macos26"
    archive = dist / f"{name}.tar.gz"
    if archive.exists():
        parser.error(f"release already exists: {archive}")
    with tempfile.TemporaryDirectory(prefix=".package-", dir=dist) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        build_runtime(stage, args.version)
        temporary_archive = Path(temporary) / archive.name
        with tarfile.open(temporary_archive, "w:gz") as release:
            release.add(stage, arcname=name)
        temporary_archive.replace(archive)
    checksum = digest(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(checksum + "\n")
    url = (
        args.url
        or f"https://github.com/{REPOSITORY}/releases/download/{args.version}/{archive.name}"
    )
    (dist / "splash.rb").write_text(
        formula(args.version, url, checksum, args.macos_min)
    )
    print(
        f"Built {archive}; run make package-bottle RELEASE_VERSION={args.version} "
        "before publishing the Homebrew formula."
    )


if __name__ == "__main__":
    main()

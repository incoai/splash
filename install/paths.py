"""Immutable program files and writable per-user data, for source or release."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (ROOT / "release.json").is_file()
_DATA_ENV = os.environ.get("SPLASH_DATA_DIR")
DATA = (
    Path(_DATA_ENV).expanduser()
    if _DATA_ENV
    else (Path.home() / "Library/Application Support/Splash" if PACKAGED else ROOT)
)
MODELS = DATA / "models" if (PACKAGED or _DATA_ENV) else ROOT / "install/models"
RUNTIME = DATA / "runtime" if (PACKAGED or _DATA_ENV) else ROOT / "build/runtime"
PYTHON = ROOT / ("python/bin/python3" if PACKAGED else ".venv/bin/python")
BINARY = ROOT / ("engine/splash" if PACKAGED else "build/splash")

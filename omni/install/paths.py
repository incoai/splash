"""The checkout's program files and the data it writes beside them."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "install/models"
RUNTIME = ROOT / "build/runtime"
PYTHON = ROOT / ".venv/bin/python"
BINARY = ROOT / "build/splash"

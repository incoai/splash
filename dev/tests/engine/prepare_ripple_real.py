"""Prepares the real Ripple-65B model root for an engine-level inference
request: the HF cache's files reach it by symlink, and a prompt's token ids
come from the release tokenizer. Nothing copies the weights.

usage: python prepare_ripple_real.py MODEL_ROOT
       then: python prepare_ripple_real.py MODEL_ROOT --prompt "..."
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GGUF_REPO = "models--AtomicChat--Ripple-65B-Preview-GGUF"
GGUF_FILE = "Ripple-65B-Preview-AD-Q4_K_M.gguf"
DRAFT_REPO = "models--ricursiveAI--Ripple-65B-Preview-DFlash2"
BASE_REPO = "models--ricursiveAI--Ripple-65B-Preview"


def snapshot(repo):
    """The repo's snapshot directory, the cache's one revision."""
    base = Path.home() / ".cache/huggingface/hub" / repo / "snapshots"
    revisions = sorted(path for path in base.iterdir() if path.is_dir())
    if not revisions:
        raise SystemExit(f"{repo} is not in the HF cache")
    return revisions[-1]


def cached_file(repo, name):
    path = snapshot(repo) / name
    if not path.exists():
        raise SystemExit(f"{repo}/{name} is not downloaded")
    return path


def link(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target)
    return path


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    link(cached_file(GGUF_REPO, GGUF_FILE), root / "target" / "ripple.gguf")
    link(cached_file(DRAFT_REPO, "model.safetensors"), root / "draft" / "model.safetensors")

    tokenizer_path = cached_file(BASE_REPO, "tokenizer.json")
    if "--prompt" not in sys.argv:
        print(f"model root ready at {root}")
        return
    prompt = sys.argv[sys.argv.index("--prompt") + 1]
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos = 100257
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    out = root / "prompt.tokens"
    out.write_text("\n".join(str(token) for token in [bos, *ids]) + "\n")
    print(f"{len(ids)} prompt tokens (plus bos) at {out}")
    print("decoded back:", tokenizer.decode([bos, *ids], skip_special_tokens=True))


if __name__ == "__main__":
    main()

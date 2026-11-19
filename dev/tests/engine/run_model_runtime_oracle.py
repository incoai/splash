"""Run the native oracle with tokenizer-derived fixtures for text markers.

Token-tag families retain their existing native fixtures. Text-marker models
use their installed template and tokenizer, including the production marker
compiler, so real-model checks never borrow another vocabulary's token IDs.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from install.families import named as family_named  # noqa: E402
from server.chat_format import THINK_END, THINK_START  # noqa: E402
from server.chat_templates import ChatTemplates  # noqa: E402
from server.token_markers import compile_markers  # noqa: E402


def fixture(tokenizer, sampling):
    def prompt(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    thinking = prompt(
        "List every prime number between 1 and 60, then check each one by trial "
        "division before you answer."
    )
    marker_close = encode(THINK_END)
    close = encode(ChatTemplates(tokenizer).select(None).thinking_end or THINK_END)
    markers = compile_markers(tokenizer, THINK_START, THINK_END, close)
    opened = list(markers)
    opened[5] = opened[2]
    step = encode(" step")
    long_end = step * ((32 + len(step) - 1) // len(step)) + close
    long_markers = list(compile_markers(tokenizer, THINK_START, THINK_END, long_end))
    long_markers[5] = long_markers[2]
    return {
        "vocabulary": len(tokenizer),
        "thinking_prompt": encode(thinking + THINK_START + "\n"),
        "sampling_seed": encode(
            prompt("Explain why 17 is prime, checking each possible divisor.")
        ),
        "production_seed": encode(
            prompt("Count upward from 1, writing each number and its English name.")
        ),
        "thinking_end": close,
        "marker_close": marker_close,
        "long_end": long_end,
        "long_markers": long_markers,
        "opening": encode(THINK_START + "\n"),
        "filler": encode(" Let us check.") + encode(">") + encode(" We continue."),
        "thinking_markers": markers,
        "opened_markers": opened,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("library")
    parser.add_argument("model", type=Path)
    parser.add_argument("options", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    binary, library, model, options = (
        arguments.binary,
        arguments.library,
        arguments.model,
        arguments.options,
    )
    record = json.loads((model / "model.json").read_text())
    family = family_named(record["family"])
    if family is None:
        raise ValueError("unknown installed model family")
    if family.chat_format.think_end_id is not None:
        return subprocess.call([binary, library, str(model), *options])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    with tempfile.TemporaryDirectory(prefix="splash-oracle-") as work:
        path = Path(work) / "tokens.json"
        payload = json.dumps(fixture(tokenizer, family.sampling), sort_keys=True)
        path.write_text(payload)
        print(
            "oracle_token_fixture_sha256="
            + hashlib.sha256(payload.encode()).hexdigest(),
            flush=True,
        )
        return subprocess.call(
            [binary, library, str(model), "--token-fixture", str(path), *options]
        )


if __name__ == "__main__":
    raise SystemExit(main())

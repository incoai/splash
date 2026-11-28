"""The chat requests backend-benchmark and decode-profile send, in a model's
own tokens.

The benchmark's decode throughput prompt and its transcripts' requests are
built in as Qwen's tokens (TranscriptPrompts.hpp), which another family's
vocabulary lacks or reads as other text. These are their texts, each one user
message rendered with thinking off by the model's chat template as a request
is (server.chat_templates), and encoded by its tokenizer; backend_regression
passes them to every round with --prompts where they are not the built-in
ones, so a Qwen model runs exactly as before.

decode-profile's chat request and counting request are built in as the Qwen
families' and Flint's tokens (ChatTokens in decode_profile.mm). Its chat
request repeats a sentence as the user's text to the prompt's length, so it
takes the rendering's parts around that text; `python -m
dev.benchmarks.chat_prompts decode-profile MODEL_ROOT PATH`, which make
benchmark-decode-profile runs, prints --chat-tokens PATH, the file of the
model's own written there, where they are not the built-in ones, else
nothing.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "dev/benchmarks/TranscriptPrompts.hpp"
PROFILE = ROOT / "dev/benchmarks/decode_profile.mm"

# The decode throughput prompt, then the transcripts' requests in their order.
TEXTS = (
    "Count upward from 1, writing each number and its English name on a new line. "
    "Continue until you reach 200.",
    "Write a Python function that merges overlapping intervals, then explain its "
    "time complexity.",
    "Implement an LRU cache in C++ with O(1) get and put, and show a short usage "
    "example.",
    "Write a Go HTTP handler that accepts a JSON body with a name field and replies "
    "with a JSON greeting.",
    "用中文解释TCP三次握手的过程，并说明为什么不能只用两次握手。",
    "请写一首关于秋天的七言绝句，然后逐句解释其中的意象。",
    "A train leaves at 9:40 and covers 210 km at 84 km/h. When does it arrive? "
    "Explain step by step.",
)

# decode-profile's chat request repeats this sentence as the user's text to its
# prompt's length; its counting request is TEXTS[0].
PROFILE_SENTENCE = (
    "Write a detailed, multi-paragraph essay about the history of the Roman Empire. "
)
# ChatTokens' parts, in --chat-tokens' order of lines.
PROFILE_PARTS = ("userHeader", "sentence", "assistantHeader", "counting")
# The user's text a chat request is rendered with to find its parts: a text no
# template's trimming or escaping changes.
_USER_TEXT = "decode_profile_user_text"

_ARRAY = re.compile(
    r"inline constexpr uint32_t (kDecodeThroughputPrompt|kTranscriptPrompt\d+)\[\] = \{([^}]*)\};"
)
_CHAT = re.compile(r"const ChatTokens (k\w+Chat)\{(.*?)\};", re.DOTALL)
_PART = re.compile(r"\.(\w+) = \{([^}]*)\}")


def builtin(header: Path = HEADER) -> list[list[int]]:
    """The built-in prompts, as the benchmark holds them: the decode
    throughput prompt, then the transcripts' requests."""
    arrays = {
        name: [int(token) for token in body.replace(",", " ").split()]
        for name, body in _ARRAY.findall(header.read_text())
    }
    names = ["kDecodeThroughputPrompt"] + [
        f"kTranscriptPrompt{index}" for index in range(len(TEXTS) - 1)
    ]
    if sorted(arrays) != sorted(names):
        raise ValueError(f"{header} does not hold the benchmark's {len(TEXTS)} prompts")
    return [arrays[name] for name in names]


def profile_builtin(source: Path = PROFILE) -> list[list[list[int]]]:
    """decode-profile's built-in chat tokens, as it holds them: the Qwen
    families', then Flint's, each ChatTokens' PROFILE_PARTS."""
    chats = {
        name: {
            part: [int(token) for token in body.replace(",", " ").split()]
            for part, body in _PART.findall(re.sub(r"//[^\n]*", "", chat))
        }
        for name, chat in _CHAT.findall(source.read_text())
    }
    names = ("kQwenChat", "kFlintChat")
    if sorted(chats) != sorted(names) or any(
        sorted(chats[name]) != sorted(PROFILE_PARTS) for name in names
    ):
        raise ValueError(
            f"{source} does not hold decode-profile's Qwen and Flint chat tokens"
        )
    return [[chats[name][part] for part in PROFILE_PARTS] for name in names]


def render_text(tokenizer, text: str) -> str:
    """One user message of text with thinking off, rendered as a chat
    request's prompt is."""
    from server.chat_templates import render_chat_template, template_options

    return render_chat_template(
        tokenizer,
        [{"role": "user", "content": text}],
        {
            "tokenize": False,
            **template_options(
                reasoning_effort="none",
                preserve_thinking=None,
                tools=None,
                add_generation_prompt=True,
            ),
        },
    )


def encode(tokenizer, text: str) -> list[int]:
    """Rendered text's tokens, as a chat request's prompt is encoded."""
    from server.tokenization import tokenizable

    return tokenizer(tokenizable(text), add_special_tokens=False)["input_ids"]


def render(tokenizer, text: str) -> list[int]:
    """One user message of text with thinking off, rendered and encoded as a
    chat request's prompt is."""
    return encode(tokenizer, render_text(tokenizer, text))


def profile_tokens(tokenizer) -> list[list[int]]:
    """decode-profile's chat tokens in the tokenizer's model, PROFILE_PARTS:
    the chat request's rendering before the user's text and after it, each
    encoded, PROFILE_SENTENCE encoded alone, and the counting request."""
    rendered = render_text(tokenizer, _USER_TEXT)
    if rendered.count(_USER_TEXT) != 1:
        raise ValueError("the chat template does not render the user's text once")
    header, assistant = rendered.split(_USER_TEXT)
    if render_text(tokenizer, TEXTS[0]) != header + TEXTS[0] + assistant:
        raise ValueError("the chat template does not render the user's text as it is")
    return [
        encode(tokenizer, header),
        encode(tokenizer, PROFILE_SENTENCE),
        encode(tokenizer, assistant),
        render(tokenizer, TEXTS[0]),
    ]


def model_tokenizer(model_root: Path):
    """The tokenizer and chat template a server of the model at model_root
    loads."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_root / "tokenizer", local_files_only=True, trust_remote_code=False
    )


def model_prompts(model_root: Path) -> list[list[int]]:
    """TEXTS in the tokens of the model at model_root, by the tokenizer and
    chat template a server of it loads."""
    tokenizer = model_tokenizer(model_root)
    return [render(tokenizer, text) for text in TEXTS]


def model_profile_tokens(model_root: Path) -> list[list[int]]:
    """decode-profile's chat tokens in the tokens of the model at
    model_root."""
    return profile_tokens(model_tokenizer(model_root))


def profile_options(model_root: Path, path: Path) -> list[str]:
    """decode-profile's options of the model at model_root: none where its
    tokenizer and chat template give built-in tokens, else --chat-tokens and
    the file of its own written at path."""
    tokens = model_profile_tokens(model_root)
    if tokens in profile_builtin():
        return []
    write(tokens, path)
    return ["--chat-tokens", str(path)]


def write(prompts: list[list[int]], path: Path) -> None:
    """The --prompts or --chat-tokens file: one line of tokens a prompt, or a
    part of one."""
    path.write_text("".join(" ".join(map(str, tokens)) + "\n" for tokens in prompts))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m dev.benchmarks.chat_prompts",
        description="A tool's options that send its chat requests in a model's own tokens.",
    )
    tools = parser.add_subparsers(dest="tool", required=True)
    profile = tools.add_parser(
        "decode-profile",
        help="print --chat-tokens PATH, writing the model's chat tokens to PATH, where they "
        "are not decode-profile's built-in ones; else print nothing",
    )
    profile.add_argument("model_root", type=Path)
    profile.add_argument("path", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = parse_args(argv)
    print(" ".join(profile_options(arguments.model_root, arguments.path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

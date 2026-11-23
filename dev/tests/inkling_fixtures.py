"""What the Inkling 2 tests share: a GGUF header like ggml-org's
Inkling-2-Mini-Q8_0.gguf (ggml-org/Inkling-2-Mini-GGUF, revision
e8a2cecf0e78914f07cc27b9be711a9b2c7561d9), its architecture metadata the
fixture's (dev/tests/fixtures/model-configs/inkling-2-mini) and its
vocabulary a stand-in of the real one's size with Inkling's 60 added tokens at
their IDs, and the tokenizer the installer derives from it."""

import functools
import json
import tempfile
from pathlib import Path

from tokenizers import pre_tokenizers
from transformers import AutoTokenizer

from dev.tests import fixture_files
from install import gguf

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIGS = FIXTURES / "model-configs" / "inkling-2-mini"
# The chat template of thinkingmachines/Inkling-2-Mini (revision
# b85c3cfa615f), which the GGUF's tokenizer.chat_template holds byte for byte.
TEMPLATE = (FIXTURES / "chat_templates" / "inkling.jinja").read_text()
# Inkling's added tokens from ID 199998, each a control token in its GGUFs.
NAMED = {
    199998: "<|unused|>",
    199999: "<|endoftext|>",
    200000: "<|message_user|>",
    200001: "<|message_model|>",
    200002: "<|message_system|>",
    200003: "<|message_tool|>",
    200004: "<|content_text|>",
    200005: "<|content_image|>",
    200006: "<|content_model_end_sampling|>",
    200008: "<|content_thinking|>",
    200010: "<|end_message|>",
    200022: "<|content_tool_error|>",
    200024: "<|content_xml|>",
    200028: "<|begin_of_text|>",
    200049: "<|content_invoke_tool_json|>",
    200057: "<|content_invoke_tool_text|>",
}
BASE = 199998
VOCABULARY = 200058
ADDED = [NAMED.get(index, f"<|unused_{index}|>") for index in range(BASE, VOCABULARY)]
# The GGUF token types (llama_token_type) of its vocabulary.
NORMAL, CONTROL = 1, 3
# Words of the stand-in vocabulary, beside the bytes: "abc" its merges do not
# build, which a tokenizer that looks a split up before any merge
# (ignore_merges) still takes whole, and words the tests' prompts use.
WORDS = ["ab", "abc", "ĠParis", "Ġweather", "hello", "th", "the", "Ġthe"]
MERGES = ["a b", "t h", "th e", "Ġ the"]


def header_values():
    """The header's metadata: the fixture's architecture metadata and a
    tokenizer of Inkling's 200058 tokens, its added tokens at their IDs."""
    tokens = sorted(pre_tokenizers.ByteLevel.alphabet()) + WORDS
    tokens += [f"[base{index}]" for index in range(BASE - len(tokens))]
    return json.loads((CONFIGS / "gguf-metadata.json").read_text()) | {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "inkling2",
        "tokenizer.ggml.tokens": tokens + ADDED,
        "tokenizer.ggml.token_type": [NORMAL] * BASE + [CONTROL] * len(ADDED),
        "tokenizer.ggml.merges": MERGES,
        "tokenizer.ggml.eos_token_id": 200006,
        "tokenizer.ggml.eot_token_id": 200006,
        "tokenizer.ggml.padding_token_id": 199999,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_eos_token": False,
        "tokenizer.chat_template": TEMPLATE,
    }


def tensor_table():
    """The tensors of ggml-org's Inkling-2-Mini-Q8_0.gguf, in its order: name,
    extents (innermost first) and GGML type ID."""
    ids = {name: kind for kind, name in gguf.TENSOR_TYPES.items()}
    return [
        (name, extents, ids[kind])
        for name, (kind, extents) in json.loads(
            (CONFIGS / "gguf-tensors.json").read_text()
        ).items()
    ]


def write_header(path, values=None, tensors=()):
    """A GGUF header of the values (header_values by default) and the
    tensors' (name, extents, GGML type), without data."""
    table = [(name, extents, kind, b"") for name, extents, kind in tensors]
    return fixture_files.write_gguf(
        path, header_values() if values is None else values, table
    )


@functools.cache
def derived_files():
    """The tokenizer files the installer derives from the header."""
    with tempfile.TemporaryDirectory() as directory:
        path = write_header(Path(directory) / "inkling.gguf")
        return gguf.tokenizer_files(gguf.Metadata(path))


def inkling_tokenizer():
    """A tokenizer loaded from the files the installer derives from the
    header, as the server loads an installed model's."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, data in derived_files().items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(data)
        return AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)

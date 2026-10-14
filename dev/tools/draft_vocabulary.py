"""Rank Qwen3.8's vocabulary for the MTP head's draft vocabulary.

Qwen3.8-Flash-Next's MTP head drafts its argmax over the most frequent tokens
alone (runtime/model/Qwen4DraftVocabulary.hpp). This ranks the tokenizer's
tokens by how often text decodes to them: it tokenizes each corpus source with
the model's tokenizer (tokenizer.json), mixes the sources' token frequencies
with fixed weights, ranks by the mix (ties toward the lower id) with the
tokenizer's added tokens first, and writes the top --tokens tokens as the
bitmap runtime/model/Qwen4DraftVocabularyData.cpp holds. With --traces it also
reports the top-K coverage of held-out decode tokens (ID.ids.npy token ids
beside ID.phase.npy phases, decode = 1).

The corpus (2026-10, in its own directory, from Hugging Face datasets):
  web-<lang>.parquet  HuggingFaceFW/fineweb-2 data/<lang>/test/000_00000.parquet
                      for 16 languages (deu_Latn fra_Latn spa_Latn jpn_Jpan
                      cmn_Hani hin_Deva rus_Cyrl por_Latn ita_Latn kor_Hang
                      arb_Arab tur_Latn vie_Latn pol_Latn nld_Latn ind_Latn)
  web-eng.json.gz     allenai/c4 en/c4-validation.00000-of-00008.json.gz
  reasoning.parquet   open-thoughts/OpenThoughts-114k data/train-00005-of-00006
  chat-smoltalk.parquet  HuggingFaceTB/smoltalk data/all/test-00000-of-00001
  chat-aya.parquet    CohereLabs/aya_dataset data/train-00000-of-00001
  math-gsm8k.parquet  openai/gsm8k main/train-00000-of-00001
plus source trees named with --code (an equal share of the code budget each).

Needs tokenizers, pyarrow and numpy (a venv of its own; not Splash's).
usage: draft_vocabulary.py --tokenizer tokenizer.json --corpus DIR --code DIR...
           [--traces DIR] [--tokens 32768] [--cpp OUT.cpp] [--ranking OUT.npy]
"""

import argparse
import glob
import gzip
import json
import os
import sys

if __package__ in (None, ""):
    from draft_vocabulary_manifest import capture_inputs, manifest_path, write_manifest
else:
    from .draft_vocabulary_manifest import capture_inputs, manifest_path, write_manifest

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

VOCABULARY = 248320  # Qwen3.8's vocabulary: its vocabulary projection's rows
MB = 1 << 20
CODE_SUFFIXES = {
    ".py",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".m",
    ".mm",
    ".metal",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".jsx",
    ".rs",
    ".go",
    ".java",
    ".swift",
    ".sh",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".html",
    ".css",
    ".sql",
}
SKIPPED_DIRECTORIES = {
    "node_modules",
    "build",
    "venv",
    ".venv",
    "__pycache__",
    "models",
}


def take(texts, budget):
    """The texts in order until they hold `budget` UTF-8 bytes."""
    taken, size = [], 0
    for text in texts:
        if not text:
            continue
        taken.append(text)
        size += len(text.encode())
        if size >= budget:
            break
    return taken


def parquet_rows(path, columns):
    table = pq.ParquetFile(path)
    for group in range(table.num_row_groups):
        yield from table.read_row_group(group, columns=columns).to_pylist()


def messages(path, column, author_key, author):
    for row in parquet_rows(path, [column]):
        for message in row[column]:
            if message.get(author_key) == author:
                yield message.get("value" if author_key == "from" else "content")


def source_paths(root):
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(
            s
            for s in subdirectories
            if not s.startswith(".") and s not in SKIPPED_DIRECTORIES
        )
        for name in sorted(files):
            path = os.path.join(directory, name)
            if (
                os.path.splitext(name)[1] not in CODE_SUFFIXES
                or os.path.getsize(path) > 400_000
            ):
                continue
            try:
                with open(path, encoding="utf-8") as file:
                    file.read()
            except (UnicodeDecodeError, OSError):
                continue
            yield path


def source_files(root):
    for path in source_paths(root):
        with open(path, encoding="utf-8") as file:
            yield file.read()


def corpus_sources(corpus, code_roots):
    """Each source's texts and its weight in the mix."""

    def path(name):
        return os.path.join(corpus, name)

    def web_english():
        with gzip.open(path("web-eng.json.gz"), "rt") as file:
            return take((json.loads(line)["text"] for line in file), 24 * MB)

    def web_languages():
        texts = []
        for name in sorted(glob.glob(path("web-*.parquet"))):
            texts += take((row["text"] for row in parquet_rows(name, ["text"])), 4 * MB)
        return texts

    def code():
        share = 80 * MB // len(code_roots)
        return [text for root in code_roots for text in take(source_files(root), share)]

    return {
        "reasoning": (
            lambda: take(
                messages(
                    path("reasoning.parquet"), "conversations", "from", "assistant"
                ),
                80 * MB,
            ),
            0.30,
        ),
        "code": (code, 0.25),
        "chat": (
            lambda: (
                take(
                    messages(
                        path("chat-smoltalk.parquet"), "messages", "role", "assistant"
                    ),
                    30 * MB,
                )
                + take(
                    (
                        row["targets"]
                        for row in parquet_rows(path("chat-aya.parquet"), ["targets"])
                    ),
                    30 * MB,
                )
            ),
            0.15,
        ),
        "web-english": (web_english, 0.10),
        "web-languages": (web_languages, 0.15),
        "math": (
            lambda: take(
                (
                    row["question"] + "\n" + row["answer"]
                    for row in parquet_rows(
                        path("math-gsm8k.parquet"), ["question", "answer"]
                    )
                ),
                8 * MB,
            ),
            0.05,
        ),
    }


def token_counts(tokenizer, texts):
    counts = np.zeros(VOCABULARY, np.int64)
    for start in range(0, len(texts), 512):
        for encoding in tokenizer.encode_batch(
            texts[start : start + 512], add_special_tokens=False
        ):
            counts += np.bincount(
                np.asarray(encoding.ids, np.int64), minlength=VOCABULARY
            )[:VOCABULARY]
    return counts


def rank(tokenizer_path, corpus, code_roots):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    with open(tokenizer_path, encoding="utf-8") as file:
        spec = json.load(file)
    added = sorted(token["id"] for token in spec.get("added_tokens", []))
    if max(max(spec["model"]["vocab"].values()), max(added)) >= VOCABULARY:
        raise SystemExit(
            "the tokenizer is not Qwen3.8's: a token id lies past the vocabulary"
        )
    mix = np.zeros(VOCABULARY)
    for name, (texts_of, weight) in corpus_sources(corpus, code_roots).items():
        texts = texts_of()
        counts = token_counts(tokenizer, texts)
        mix += weight * counts / max(1, counts.sum())
        print(
            f"{name}: {len(texts)} texts, {int(counts.sum())} tokens, {int((counts > 0).sum())} distinct, weight {weight}"
        )
    mix[added] = 2.0  # the chat template's, thinking and tool-call tokens first
    return np.lexsort((np.arange(VOCABULARY), -mix)).astype(np.uint32)


def coverage(ranking, traces, sizes):
    position = np.empty(VOCABULARY, np.int64)
    position[ranking] = np.arange(VOCABULARY)
    decoded = []
    for name in sorted(glob.glob(os.path.join(traces, "*.ids.npy"))):
        ids = np.load(name).astype(np.int64)
        decoded.append(
            ids[(np.load(name.replace(".ids.", ".phase.")) == 1) & (ids < VOCABULARY)]
        )
    decoded = np.concatenate(decoded)
    print(f"held-out decode tokens: {len(decoded)}")
    for size in sizes:
        print(
            f"  top {size}: {np.mean(position[decoded] < size):.4f} (token-id order {np.mean(decoded < size):.4f})"
        )


def write_cpp(path, tokens):
    words = np.zeros(VOCABULARY // 64, np.uint64)
    for token in tokens:
        words[token // 64] |= np.uint64(1) << np.uint64(token % 64)
    lines = [
        "// Generated by dev/tools/draft_vocabulary.py; do not edit. Qwen3.8's draft",
        f"// vocabulary: a bitmap of its {len(tokens)} tokens, bit t % 64 of word t / 64.",
        '#include "model/Qwen4DraftVocabulary.hpp"',
        "",
        "namespace splash::model::detail {",
        "",
        f"extern const uint64_t kQwen4DraftVocabularyWords[{len(words)}] = {{",
    ]
    for start in range(0, len(words), 4):
        lines.append(
            "    "
            + " ".join(f"0x{int(word):016x}ull," for word in words[start : start + 4])
        )
    lines += ["};", "", "} // namespace splash::model::detail", ""]
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--code", nargs="+", required=True)
    parser.add_argument("--traces")
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--cpp")
    parser.add_argument("--ranking")
    parser.add_argument(
        "--manifest",
        help="generation record (default: replace --cpp suffix with .manifest.json)",
    )
    arguments = parser.parse_args()
    code_paths = [path for root in arguments.code for path in source_paths(root)]
    inputs = (
        capture_inputs(arguments.tokenizer, arguments.corpus, code_paths)
        if arguments.cpp
        else None
    )
    ranking = rank(arguments.tokenizer, arguments.corpus, arguments.code)
    if inputs is not None and inputs != capture_inputs(
        arguments.tokenizer, arguments.corpus, code_paths
    ):
        raise RuntimeError("generation inputs changed while ranking")
    if arguments.ranking:
        np.save(arguments.ranking, ranking)
    if arguments.traces:
        coverage(ranking, arguments.traces, [16384, 32768, 49152, 65536])
    if arguments.cpp:
        write_cpp(arguments.cpp, np.sort(ranking[: arguments.tokens]))
        write_manifest(
            arguments.manifest or manifest_path(arguments.cpp),
            arguments.tokenizer,
            arguments.corpus,
            code_paths,
            arguments.cpp,
            [sys.executable, *sys.argv],
            inputs=inputs,
        )


if __name__ == "__main__":
    main()

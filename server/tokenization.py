"""Prompt tokenization: a rendered prompt's tokens, in which only the text
the chat template wrote reads as a token the tokenizer adds, and bounded token
reuse at literal chat-message boundaries."""

import json
import re
import secrets
import sys
import threading
from array import array

from tokenizers import Tokenizer

from .lru import LRUCache

# A surrogate code point. Request text keeps a lone one that JSON spells, such
# as half of an emoji ("\ud83d") a client cut, but the tokenizer refuses any
# text that holds one, and the request with it.
_SURROGATE = re.compile("[\ud800-\udfff]")


def tokenizable(text):
    """text with each surrogate code point as U+FFFD, as a UTF-8 decoder reads
    an invalid byte. One character replaces one, so offsets into text hold."""
    return _SURROGATE.sub("\ufffd", text)


# The mark PromptTokenizer.quote sets after the first character of an added
# token's text in what a client sent: a text no request can hold, as its 128
# random bits are drawn when the server starts, and of characters that a
# template's JSON or trimming of the text leaves as they are.
QUOTE = f"__splash_quoted_{secrets.token_hex(16)}__"


def unquoted(text):
    """`text` without its quote marks: the text the model reads."""
    return text.replace(QUOTE, "")


def added_tokens(tokenizer):
    """The tokens `tokenizer` adds beside its model's vocabulary, by their
    text: a chat template's turn and role tokens, its tags, an image's pad.
    It splits its input at each before anything else reads the text, and
    PromptTokenizer keeps that split from the text a client sent. A
    ValueError names what it cannot keep it from: a tokenizer without a fast
    backend, an added token that is not found as its literal text or that
    QUOTE cannot go inside, and a pipeline where the text between two added
    tokens does not encode on its own. A tokenizer that adds none splits
    nothing."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("tokenizer must be a fast tokenizer")
    tokens = {}
    for token_id, token in backend.get_added_tokens_decoder().items():
        text = token.content
        if token.normalized or token.lstrip or token.rstrip or token.single_word:
            raise ValueError(
                f"tokenizer must find the added token {text!r} as its literal text"
            )
        # The mark follows the token's first character, and its own text
        # must not complete a token with the characters around it.
        if (
            len(text) < 2
            or text in QUOTE
            or any(
                QUOTE.startswith(text[-length:]) or QUOTE.endswith(text[:length])
                for length in range(1, len(text))
            )
        ):
            raise ValueError(f"tokenizer adds a token that cannot be quoted: {text!r}")
        tokens[text] = token_id
    if tokens and not _encodes_segments_alone(backend):
        raise ValueError(
            "tokenizer must encode the text between two added tokens on its own"
        )
    return tokens


def _encodes_segments_alone(backend):
    """Whether the text between two literal added tokens encodes as it does
    alone: byte-level BPE after regex splits, Qwen's one or Flint's three,
    each of which reads a segment on its own, and no normalizer or NFC.
    Another pipeline may not: for example, Metaspace's prepend_scheme="first"
    depends on whether a segment starts the complete input, not just a
    split. A post-processor has no part in it: prompts encode without the
    tokenizer's own special tokens, and it adds no other."""
    if type(backend.model).__name__ != "BPE" or backend.model.dropout:
        return False
    if backend.pre_tokenizer is None:
        return False
    pre = json.loads(backend.pre_tokenizer.__getstate__())
    parts = pre["pretokenizers"] if pre.get("type") == "Sequence" else [pre]
    if [part.get("type") for part in parts] != ["Split"] * (len(parts) - 1) + [
        "ByteLevel"
    ] or parts[-1].get("add_prefix_space"):
        return False
    return (
        backend.normalizer is None
        or json.loads(backend.normalizer.__getstate__()).get("type") == "NFC"
    )


class PromptTokenizer:
    """Encodes rendered prompts so that only the chat template's own text
    reads as a token the tokenizer adds (added_tokens). A tokenizer finds an
    added token wherever the text spells it, so one that a message only
    quotes, as a tool result showing a parser's `tool_call_start =
    "<tool_call>"` does, would reach the model as the control token, and a
    `<|im_end|>` in a document as the end of its turn. `quote` marks each in
    what a client sent before the template renders it; `encode` reads the
    template's text between two of the tokens the template wrote as ordinary
    text, a marked token as its characters.

    It reuses a cached prompt's tokens up to its last `marker`, the chat
    template's special token at a message boundary
    (ChatFormat.message_boundary), where the tokenizer splits its input.
    It keeps up to CAPACITY prompts, in storage for BUDGET_PROMPTS of
    `max_context` tokens, the served context. A cached prompt is its text
    with its marks, which say how it encodes, so a conversation that quotes
    a tag reuses its tokens as any other does: a later turn of a 66K-token
    one tokenizes in 1.0 ms either way."""

    MIN_PREFIX_CHARS = 4096
    # A conversation keeps one prompt, its newest, so the count bounds the
    # conversations whose next turn reuses it, and agents interleave more of
    # them than there are lanes: with 4, eight coding agents on 4 lanes
    # reused 6% of their 44K-token prompts, each miss encoding one in 27 ms;
    # with 16 every turn found its prompt, 1.8 ms a request.
    CAPACITY = 16
    # The storage, which bounds the memory however many prompts it holds.
    BUDGET_PROMPTS = 4
    # The bytes a kept prompt takes per token, at most: the token's 4 and its
    # text, up to about 5 characters (code's; prose's are 4) at the 4 bytes
    # CPython stores each of a text that holds any past U+FFFF, as an emoji.
    BYTES_PER_TOKEN = 24

    def __init__(self, tokenizer, marker, max_context):
        self.tokenizer = tokenizer
        self.marker = marker
        self.added = added_tokens(tokenizer)
        # The marker is a boundary where it is one added token and no other
        # holds it.
        self.enabled = [text for text in self.added if marker in text] == [marker]
        self.marker_id = self.added[marker] if self.enabled else None
        # Where an added token's text begins, to mark it there: for each
        # first character of one, that character before the rest of any.
        rests = {}
        for text in self.added:
            rests.setdefault(text[0], []).append(re.escape(text[1:]))
        self._begins = [
            (first, re.compile(f"{re.escape(first)}(?={'|'.join(rest)})"))
            for first, rest in rests.items()
        ]
        self._marked = r"\g<0>" + QUOTE
        if self.added:
            # Each whole one, the longest first as the tokenizer finds them.
            self._tokens = re.compile(
                "("
                + "|".join(map(re.escape, sorted(self.added, key=len, reverse=True)))
                + ")"
            )
            # The tokenizer's own normalizer, pre-tokenizer and model, which
            # it shares, without its added tokens: how it encodes ordinary
            # text.
            backend = tokenizer.backend_tokenizer
            self._plain = Tokenizer(backend.model)
            if backend.normalizer is not None:
                self._plain.normalizer = backend.normalizer
            self._plain.pre_tokenizer = backend.pre_tokenizer
        self.entries = LRUCache(
            self.BUDGET_PROMPTS * max_context * self.BYTES_PER_TOKEN, self.CAPACITY
        )
        self.hits = self.misses = self.reused_tokens = 0
        self.lock = threading.Lock()

    def quote(self, value):
        """`value`, what a request gives its chat template (messages, tool
        definitions, template variables), with QUOTE inside every added
        token that a string or a key of it spells, wherever one begins. The
        template renders the marked text as it renders any other, and no
        longer finds a tag there itself."""
        if isinstance(value, str):
            for first, begins in self._begins:
                # Few strings hold the character at all: looking for it
                # first quotes a 66K-token conversation of 2,600 strings in
                # 0.4 ms, where the search alone took 2.8 ms.
                if first in value:
                    value = begins.sub(self._marked, value)
            return value
        if isinstance(value, dict):
            return {self.quote(key): self.quote(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.quote(item) for item in value]
        return value

    def tokens(self, text):
        """The tokens of `text`, a rendered prompt or a part of one that
        encodes on its own. Without a mark it encodes as the tokenizer
        encodes it. With one, each added token it still spells whole is the
        template's, and the text between two of them, its marks taken out,
        encodes as ordinary text: what the tokenizer does with a text it
        splits at those tokens alone."""
        text = tokenizable(text)
        if QUOTE not in text:
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        parts = self._tokens.split(text)
        plain = self._plain.encode_batch(
            [unquoted(part) for part in parts[::2]], add_special_tokens=False
        )
        tokens = list(plain[0].ids)
        for token, following in zip(parts[1::2], plain[1:]):
            tokens.append(self.added[token])
            tokens.extend(following.ids)
        return tokens

    def encode(self, text):
        """The tokens of the rendered prompt `text` (tokens), reusing a
        cached prompt's."""
        boundary = text.rfind(self.marker)
        if not self.enabled or boundary < 0:
            return self.tokens(text)
        boundary += len(self.marker)
        if boundary < self.MIN_PREFIX_CHARS:
            return self.tokens(text)
        prefix = text[:boundary]
        with self.lock:
            key = max(
                (key for key in self.entries if prefix.startswith(key)),
                key=len,
                default="",
            )
            cached = self.entries.get(key)
            if cached is not None:
                self.hits += 1
                self.reused_tokens += len(cached) // array("I").itemsize
            else:
                self.misses += 1
        tokens = array("I", cached).tolist() if cached is not None else []
        if key != prefix:
            tokens.extend(self.tokens(prefix[len(key) :]))
            # A different added token can consume part of the marker. In that
            # case this literal occurrence is not a tokenizer boundary.
            if not tokens or tokens[-1] != self.marker_id:
                return self.tokens(text)
            token_bytes = array("I", tokens).tobytes()
            size = sys.getsizeof(prefix) + sys.getsizeof(token_bytes)
            if size <= self.entries.budget_bytes:
                with self.lock:
                    # An extension replaces its earlier prefix; unrelated
                    # concurrent conversations retain their own LRU entries.
                    self.entries.pop(key)
                    self.entries.put(prefix, token_bytes, size)
        return tokens + self.tokens(text[boundary:])

    def split(self, text):
        """`text` cut just past its last marker where the tokenizer splits it
        there: the tokens before the cut, and the text after it, which
        encodes on its own (tokens). No tokens and all of `text` where it
        does not."""
        boundary = text.rfind(self.marker)
        if self.enabled and boundary >= 0:
            boundary += len(self.marker)
            head = self.tokens(text[:boundary])
            # As in encode, another added token may consume part of the
            # marker, which is then no boundary.
            if head and head[-1] == self.marker_id:
                return head, text[boundary:]
        return [], text

    def stats(self):
        with self.lock:
            return {
                "enabled": self.enabled,
                "entries": len(self.entries),
                "bytes": self.entries.bytes,
                "budget_bytes": self.entries.budget_bytes,
                "capacity": self.entries.capacity,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.entries.evictions,
                "reused_tokens": self.reused_tokens,
            }

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
# token that begins in what a client sent: a text no request can hold, as its
# 128 random bits are drawn when the server starts, and of characters that a
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
    backend, an added token that takes the spaces beside it or stands only
    as a word, which its literal text does not find, or that QUOTE cannot
    go inside, and a pipeline where the text between two added tokens does
    not encode on its own. A normalized one, as add_tokens adds them, is
    found in the normalized text, its literal text among others
    (PromptTokenizer.literal). A tokenizer that adds none splits nothing."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("tokenizer must be a fast tokenizer")
    tokens = {}
    for token_id, token in backend.get_added_tokens_decoder().items():
        text = token.content
        if token.lstrip or token.rstrip or token.single_word:
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


def _rest_pattern(rests):
    """A pattern for what follows an added token's first character where the
    token begins in a text: the rest of one, each of `rests`, or as much of
    a rest as the text holds where it ends first, which what follows the
    text in a prompt could complete. It reads the characters that rests
    share once: 1 MB of markup takes 1.7 ms with Flint's 144 tokens, where
    trying each rest in turn took 23 ms."""
    shared = {}
    for rest in rests:
        node = shared
        for character in rest:
            node = node.setdefault(character, {})
        # A token ends here: nothing more to read.
        node[""] = {}

    def pattern(node):
        if "" in node:
            return ""
        following = "|".join(
            re.escape(character) + pattern(after) for character, after in node.items()
        )
        return rf"(?:\Z|{following})"

    return pattern(shared)


class PromptTokenizer:
    """Encodes rendered prompts so that only the chat template's own text
    reads as a token the tokenizer adds (added_tokens). A tokenizer finds an
    added token wherever the text spells it, so one that a message only
    quotes, as a tool result showing a parser's `tool_call_start =
    "<tool_call>"` does, would reach the model as the control token, and a
    `<|im_end|>` in a document as the end of its turn. `quote` marks each
    where it begins in what a client sent, before the template renders it;
    `encode` reads the template's text between two of the tokens the template
    wrote as ordinary text, a marked token as its characters.

    It reuses a cached prompt's tokens up to a message end: a `marker`, the
    chat template's special token at a message boundary
    (ChatFormat.message_boundary), where the tokenizer splits its input. A
    prompt that extends a cached prompt reuses it whole; any other prompt
    reuses the longest run of whole messages it shares with a
    cached prompt, which keeps where each of its markers ends. So a branch
    of a conversation, or a return to an earlier turn, reuses the messages
    they share: on an M5 Max a 13.8K-token Qwen3.8 prompt whose last
    message differs encodes in 0.30 ms where it took 9.9 ms. It keeps up
    to CAPACITY prompts, in storage for BUDGET_PROMPTS of `max_context`
    tokens, the served context. A cached prompt is its text
    with its marks, which say how it encodes, so a conversation that quotes
    a tag reuses its tokens as any other does: a later turn of a 66K-token
    one tokenizes in 1.0 ms either way."""

    MIN_PREFIX_CHARS = 4096
    # A conversation keeps one prompt, its newest, and each branch of it its
    # own, so the count bounds the conversations whose next turn reuses it,
    # and agents interleave more of them than there are lanes: with 4, eight
    # coding agents on 4 lanes reused 6% of their 44K-token prompts, each
    # miss encoding one in 27 ms; with 16 every turn found its prompt, 1.8 ms
    # a request.
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
        backend = tokenizer.backend_tokenizer
        # Whether the tokenizer finds its added tokens as their literal text
        # alone. A normalized one, as add_tokens adds them, it finds in the
        # normalized text, where NFC makes another spelling of it the token
        # too, as U+212A for a K, which no mark breaks: such a tokenizer's
        # prompts all encode as the text between the template's tokens.
        self.literal = backend.normalizer is None or not any(
            token.normalized for token in backend.get_added_tokens_decoder().values()
        )
        # The marker is a boundary where it is one added token and no other
        # holds it.
        self.enabled = [text for text in self.added if marker in text] == [marker]
        self.marker_id = self.added[marker] if self.enabled else None
        # Where an added token begins in a text, to mark it there: for each
        # first character of one, that character before the rest of any, or
        # before the text's end where that comes first (_rest_pattern).
        rests = {}
        for text in self.added:
            rests.setdefault(text[0], []).append(text[1:])
        self._begins = [
            (first, re.compile(f"{re.escape(first)}(?={_rest_pattern(rest)})"))
            for first, rest in rests.items()
        ]
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
            self._plain = Tokenizer(backend.model)
            if backend.normalizer is not None:
                self._plain.normalizer = backend.normalizer
            self._plain.pre_tokenizer = backend.pre_tokenizer
        self.entries = LRUCache(
            self.BUDGET_PROMPTS * max_context * self.BYTES_PER_TOKEN, self.CAPACITY
        )
        self.hits = self.misses = self.reused_tokens = 0
        self.lock = threading.Lock()

    @staticmethod
    def _marked(begun):
        """The first character of a token that begins, with the mark after
        it. A function, as `sub` looks a replacement text up by a hash of
        its whole pattern at every call: 2 to 4 microseconds for these."""
        return begun[0] + QUOTE

    def quote(self, value):
        """`value`, what a request gives its chat template (messages, tool
        definitions, template variables), with QUOTE inside every added
        token that a string or a key of it begins: one it spells whole, and
        one its end cuts off, which what the template writes next could
        complete, as the `>` Qwen's templates write after a call's name
        would, or the next of a message's text parts. A U+0000 is U+FFFD
        first: a text that holds one inside a token spells no token to mark,
        and a template that deletes the character, as Flint's does, would
        write the token whole. The template renders the marked text as it
        renders any other, and no longer finds a tag there itself."""
        if isinstance(value, str):
            if "\x00" in value:
                value = value.replace("\x00", "\ufffd")
            for first, begins in self._begins:
                # Few strings hold the character at all: looking for it
                # first quotes a 66K-token conversation of 2,600 strings in
                # 0.24 ms, where searching each took 0.40 ms.
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
        encodes it, where that finds its added tokens as their literal text
        alone (literal). Otherwise each added token it still spells whole is
        the template's, and the text between two of them, its marks taken
        out, encodes as ordinary text: what the tokenizer does with a text
        it splits at those tokens alone."""
        text = tokenizable(text)
        if QUOTE not in text and self.literal:
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
            # An extension, an agent's ordinary request, reuses the longest
            # cached prompt it extends, whole: one comparison a prompt, which
            # stops at the first character that differs. Only a prompt that
            # extends none, which would encode from its start, compares
            # message by message: over 16 cached 42K-token prompts that
            # share a head, every request doing so took 0.21 ms more on an
            # M5 Max. A shorter cached prompt the prompt extends is taken
            # even where another shares more of its messages: that needs a
            # stale entry beside a longer branch, and costs encode time,
            # never a token.
            key = max(
                (kept for kept in self.entries if prefix.startswith(kept)),
                key=len,
                default=None,
            )
            # How many of its messages: all ([:None]) of one the prompt extends.
            shared = None
            reused_chars = 0 if key is None else len(key)
            if key is None:
                for kept, (_, ends, _) in self.entries.items():
                    count = start = 0
                    for end in ends:
                        if prefix[start:end] != kept[start:end]:
                            break
                        count, start = count + 1, end
                    if start > reused_chars:
                        key, shared, reused_chars = kept, count, start
            if key is None:
                self.misses += 1
                reused, char_ends, token_ends = array("I"), array("Q"), array("Q")
            else:
                cached_tokens, char_ends, token_ends = self.entries.get(key)
                char_ends, token_ends = char_ends[:shared], token_ends[:shared]
                reused = cached_tokens[: token_ends[-1]]
                self.hits += 1
                self.reused_tokens += len(reused)
        tokens = reused.tolist()
        if reused_chars < boundary:
            # The rest of the prefix encodes in one call: one call per message
            # took a cold 800-message, 15.9K-token Qwen3.8 prompt from 11.9 to
            # 17.1 ms on an M5 Max. Each marker in it is the marker's token,
            # which no other text encodes to (enabled), so the n-th marker's
            # token ends the n-th message. A different added token can
            # consume part of a marker, which is then no tokenizer boundary:
            # fewer tokens than markers are left, the last marker finds none,
            # and the prompt encodes whole.
            index = len(tokens) - 1
            tokens.extend(self.tokens(prefix[reused_chars:]))
            end = reused_chars
            while end < boundary:
                end = prefix.find(self.marker, end) + len(self.marker)
                try:
                    index = tokens.index(self.marker_id, index + 1)
                except ValueError:
                    return self.tokens(text)
                char_ends.append(end)
                token_ends.append(index + 1)
            # A cached prompt: its tokens, and where each of its messages
            # ends, in characters and in tokens.
            entry = (array("I", tokens), char_ends, token_ends)
            size = sys.getsizeof(prefix) + sys.getsizeof(entry)
            size += sum(map(sys.getsizeof, entry))
            if size <= self.entries.budget_bytes:
                with self.lock:
                    # An extension replaces the prompt it extends, whose
                    # message ends it holds; a branch keeps its own.
                    if key is not None and prefix.startswith(key):
                        self.entries.pop(key)
                    self.entries.put(prefix, entry, size)
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

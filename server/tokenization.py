"""Bounded token reuse at literal chat-message boundaries."""

import json
import re
import sys
import threading
from array import array

from .lru import LRUCache

# A surrogate code point. Request text keeps a lone one that JSON spells, such
# as half of an emoji ("\ud83d") a client cut, but the tokenizer refuses any
# text that holds one, and the request with it.
_SURROGATE = re.compile("[\ud800-\udfff]")


def tokenizable(text):
    """text with each surrogate code point as U+FFFD, as a UTF-8 decoder reads
    an invalid byte. One character replaces one, so offsets into text hold."""
    return _SURROGATE.sub("\ufffd", text)


class PromptTokenizer:
    """Encodes prompts reusing a cached prompt's tokens up to its last
    `marker`, the chat template's special token at a message boundary
    (ChatFormat.message_boundary), where the tokenizer splits its input.
    It keeps up to CAPACITY prompts, in storage for BUDGET_PROMPTS of
    `max_context` tokens, the served context."""

    MIN_PREFIX_CHARS = 4096
    # Each branch keeps its newest prompt and earlier message ends. Agents
    # interleave more branches than there are lanes: sixteen entries let
    # their later turns reuse context within the unchanged byte budget.
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
        self.enabled = self._supports_boundaries(tokenizer, marker)
        self.marker_id = (
            tokenizer.backend_tokenizer.token_to_id(marker) if self.enabled else None
        )
        self.entries = LRUCache(
            self.BUDGET_PROMPTS * max_context * self.BYTES_PER_TOKEN, self.CAPACITY
        )
        self.hits = self.misses = self.reused_tokens = 0
        self.lock = threading.Lock()

    @staticmethod
    def _supports_boundaries(tokenizer, marker):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if not getattr(tokenizer, "is_fast", False) or backend is None:
            return False
        # Literal, non-stripping added tokens split BPE input before ordinary
        # normalization/pretokenization. Do not assume this for other pipelines.
        if type(backend.model).__name__ != "BPE" or backend.model.dropout:
            return False
        if backend.pre_tokenizer is None:
            return False
        pre = json.loads(backend.pre_tokenizer.__getstate__())
        parts = pre["pretokenizers"] if pre.get("type") == "Sequence" else [pre]
        # Regex splits, Qwen's one or Flint's three, then byte-level BPE: each
        # split reads a segment on its own. Another pre-tokenizer may not:
        # for example, Metaspace's prepend_scheme="first" depends on whether
        # a segment starts the complete input, not just a split.
        if [part.get("type") for part in parts] != ["Split"] * (len(parts) - 1) + [
            "ByteLevel"
        ] or parts[-1].get("add_prefix_space"):
            return False
        for component, allowed in (
            (backend.normalizer, {"NFC"}),
            (backend.post_processor, {"ByteLevel"}),
        ):
            if (
                component is not None
                and json.loads(component.__getstate__()).get("type") not in allowed
            ):
                return False
        markers = []
        for token in backend.get_added_tokens_decoder().values():
            if marker not in token.content:
                continue
            if token.content != marker:
                return False
            markers.append(token)
        return len(markers) == 1 and not any(
            (
                markers[0].normalized,
                markers[0].lstrip,
                markers[0].rstrip,
                markers[0].single_word,
            )
        )

    def _encode(self, text):
        return self.tokenizer(tokenizable(text), add_special_tokens=False)["input_ids"]

    @staticmethod
    def _shared_boundary(prefix, key, boundaries):
        # Common message ends are ordered: a binary search compares their
        # text in C, including when the next turn branches or gets shorter.
        low, high = 0, len(boundaries) // 2
        while low < high:
            middle = (low + high) // 2
            end = boundaries[2 * middle]
            if end <= len(prefix) and prefix.startswith(key[:end]):
                low = middle + 1
            else:
                high = middle
        return low

    def encode(self, text):
        boundary = text.rfind(self.marker)
        if not self.enabled or boundary < 0:
            return self._encode(text)
        boundary += len(self.marker)
        if boundary < self.MIN_PREFIX_CHARS:
            return self._encode(text)
        prefix = text[:boundary]
        with self.lock:
            key, cached, shared = "", None, 0
            reused_chars = 0
            for candidate, entry in self.entries.items():
                count = self._shared_boundary(prefix, candidate, entry[1])
                if count and entry[1][2 * count - 2] > reused_chars:
                    key, cached, shared = candidate, entry, count
                    reused_chars = entry[1][2 * count - 2]
            if cached is not None:
                self.entries.get(key)
                self.hits += 1
                self.reused_tokens += cached[1][2 * shared - 1]
            else:
                self.misses += 1
        boundaries = array("Q", cached[1][: 2 * shared]) if cached else array("Q")
        token_count = boundaries[-1] if boundaries else 0
        tokens = (
            array("I", cached[0][: token_count * array("I").itemsize]).tolist()
            if cached
            else []
        )
        position = reused_chars
        while position < boundary:
            end = prefix.find(self.marker, position) + len(self.marker)
            part = self._encode(prefix[position:end])
            # A different added token can consume part of the marker. In that
            # case this literal occurrence is not a tokenizer boundary.
            if not part or part[-1] != self.marker_id:
                return self._encode(text)
            tokens.extend(part)
            boundaries.extend((end, len(tokens)))
            position = end
        if reused_chars < boundary:
            token_bytes = array("I", tokens).tobytes()
            entry = (token_bytes, boundaries)
            size = (
                sys.getsizeof(prefix)
                + sys.getsizeof(token_bytes)
                + sys.getsizeof(boundaries)
                + sys.getsizeof(entry)
            )
            if size <= self.entries.budget_bytes:
                with self.lock:
                    # A growing conversation retains every earlier message
                    # boundary in its newest entry. Branches keep their own.
                    if key and prefix.startswith(key):
                        self.entries.pop(key)
                    self.entries.put(prefix, entry, size)
        return tokens + self._encode(text[boundary:])

    def split(self, text):
        """`text` cut just past its last marker where the tokenizer splits it
        there: the tokens before the cut, and the text after it, which
        encodes on its own. No tokens and all of `text` where it does not."""
        boundary = text.rfind(self.marker)
        if self.enabled and boundary >= 0:
            boundary += len(self.marker)
            head = self._encode(text[:boundary])
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

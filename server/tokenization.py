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
    MARKER = "<|im_end|>"
    MIN_PREFIX_CHARS = 4096
    BUDGET_BYTES = 8 * 1024 * 1024
    CAPACITY = 4

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.enabled = self._supports_boundaries(tokenizer)
        self.marker_id = (
            tokenizer.backend_tokenizer.token_to_id(self.MARKER)
            if self.enabled
            else None
        )
        self.entries = LRUCache(self.BUDGET_BYTES, self.CAPACITY)
        self.hits = self.reused_tokens = 0
        self.lock = threading.Lock()

    @classmethod
    def _supports_boundaries(cls, tokenizer):
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
        if [part.get("type") for part in parts] not in (
            ["ByteLevel"],
            ["Split", "ByteLevel"],
        ) or parts[-1].get("add_prefix_space"):
            # For example, Metaspace's prepend_scheme="first" depends on
            # whether a segment starts the complete input, not just a split.
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
            if cls.MARKER not in token.content:
                continue
            if token.content != cls.MARKER:
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

    def encode(self, text):
        boundary = text.rfind(self.MARKER)
        if not self.enabled or boundary < 0:
            return self._encode(text)
        boundary += len(self.MARKER)
        if boundary < self.MIN_PREFIX_CHARS:
            return self._encode(text)
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
        tokens = array("I", cached).tolist() if cached is not None else []
        if key != prefix:
            tokens.extend(self._encode(prefix[len(key) :]))
            # A different added token can consume part of the marker. In that
            # case this literal occurrence is not a tokenizer boundary.
            if not tokens or tokens[-1] != self.marker_id:
                return self._encode(text)
            token_bytes = array("I", tokens).tobytes()
            size = sys.getsizeof(prefix) + sys.getsizeof(token_bytes)
            if size <= self.BUDGET_BYTES:
                with self.lock:
                    # An extension replaces its earlier prefix; unrelated
                    # concurrent conversations retain their own LRU entries.
                    self.entries.pop(key)
                    self.entries.put(prefix, token_bytes, size)
        return tokens + self._encode(text[boundary:])

    def split(self, text):
        """`text` cut just past its last MARKER where the tokenizer splits it
        there: the tokens before the cut, and the text after it, which
        encodes on its own. No tokens and all of `text` where it does not."""
        boundary = text.rfind(self.MARKER)
        if self.enabled and boundary >= 0:
            boundary += len(self.MARKER)
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
                "reused_tokens": self.reused_tokens,
            }

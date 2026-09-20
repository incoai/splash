"""Bounded reuse of exact rendered inputs for one immutable tokenizer."""

import hashlib
import threading
from array import array
from collections import OrderedDict
from concurrent.futures import Future, wait
from dataclasses import dataclass

if __package__:
    from .backend import remaining_request_time
else:
    from backend import remaining_request_time


@dataclass(frozen=True)
class EncodedInput:
    tokens: bytes
    offsets: bytes | None

    @classmethod
    def pack(cls, result):
        offsets = result.get("offset_mapping")
        return cls(
            array("I", result["input_ids"]).tobytes(),
            None
            if offsets is None
            else array("I", (value for pair in offsets for value in pair)).tobytes(),
        )

    @property
    def size(self):
        return len(self.tokens) + len(self.offsets or b"")

    def unpack(self):
        tokens = array("I")
        tokens.frombytes(self.tokens)
        result = {"input_ids": tokens.tolist()}
        if self.offsets is not None:
            offsets = array("I")
            offsets.frombytes(self.offsets)
            result["offset_mapping"] = list(zip(offsets[::2], offsets[1::2]))
        return result


class TokenizationCache:
    def __init__(self, tokenizer, budget_bytes=8 * 1024**2, max_entries=32):
        if budget_bytes < 0 or max_entries < 0:
            raise ValueError("tokenization cache limits must be nonnegative")
        self.tokenizer = tokenizer
        self.budget_bytes = budget_bytes
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._pending = {}
        self._lock = threading.Lock()
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    def encode(self, text, deadline, *, add_special_tokens=False, offsets=False):
        """Return caller-owned input IDs and, when requested, offsets."""
        # Key the rendered text, not messages: templates may depend on time or
        # other render-time values. The tokenizer belongs to this cache's model.
        digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).digest()
        key = (digest, add_special_tokens, offsets)
        remaining_request_time(deadline)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self._hits += 1
            pending = self._pending.get(key)
            owner = cached is None and pending is None
            if owner:
                pending = self._pending[key] = Future()
                self._misses += 1
        if cached is not None:
            return cached.unpack()
        if not owner:
            wait((pending,), timeout=remaining_request_time(deadline))
            remaining_request_time(deadline)
            encoded = pending.result()
            with self._lock:
                self._hits += 1
            return encoded.unpack()
        try:
            options = {"add_special_tokens": add_special_tokens}
            if offsets:
                options["return_offsets_mapping"] = True
            result = self.tokenizer(text, **options)
            encoded = EncodedInput.pack(result)
            with self._lock:
                if self.max_entries and encoded.size <= self.budget_bytes:
                    while self._entries and (
                        len(self._entries) >= self.max_entries
                        or self._bytes + encoded.size > self.budget_bytes
                    ):
                        _, old = self._entries.popitem(last=False)
                        self._bytes -= old.size
                    self._entries[key] = encoded
                    self._bytes += encoded.size
            # In-flight work is shared even when it cannot enter the LRU.
            # Each caller applies its own deadline after receiving the result.
            pending.set_result(encoded)
        except BaseException as error:
            pending.set_exception(error)
            raise
        finally:
            with self._lock:
                del self._pending[key]
        remaining_request_time(deadline)
        return encoded.unpack()

    def stats(self):
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "budget_bytes": self.budget_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "pending": len(self._pending),
            }

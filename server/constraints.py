"""Tokenizer contract and cached token-level output constraints."""

import threading
from collections import OrderedDict
from concurrent.futures import Future, wait

from llguidance import LLExecutor, LLMatcher, LLTokenizer
from llguidance.hf import from_tokenizer as guidance_tokenizer
from llguidance.numpy import (
    allocate_token_bitmask,
    fill_next_token_bitmask_par,
    fill_next_token_bitmask_par_with_draft_tokens,
)

if __package__:
    from . import runtime as engine_runtime
    from .errors import APIError, NativeError
    from .tool_schema import THINK_END, THINK_END_TOKEN_ID
else:
    from errors import APIError, NativeError
    from tool_schema import THINK_END, THINK_END_TOKEN_ID

    import runtime as engine_runtime


class TokenConstraint:
    VOCABULARY = 248320
    MAX_ROWS = 9
    EOS_TOKENS = (248044, 248046)

    def __init__(self, matcher, executor):
        self.matcher = matcher
        self.executor = executor
        self.bitmask = allocate_token_bitmask(self.MAX_ROWS, self.VOCABULARY)

    def masks(self, simulation_tokens):
        if len(simulation_tokens) >= self.MAX_ROWS:
            raise NativeError("constraint_error", "too many simulation tokens")
        in_range = next(
            (
                index
                for index, token in enumerate(simulation_tokens)
                if not 0 <= token < self.VOCABULARY
            ),
            len(simulation_tokens),
        )
        probe = self.matcher.deep_copy()
        valid_count = probe.validate_tokens(list(simulation_tokens[:in_range]))
        valid_tokens = simulation_tokens[:valid_count]
        if valid_tokens:
            fill_next_token_bitmask_par_with_draft_tokens(
                self.executor,
                [(self.matcher, 0, list(valid_tokens))],
                self.bitmask,
            )
        else:
            fill_next_token_bitmask_par(
                self.executor, [(self.matcher, 0)], self.bitmask
            )
        rows = len(simulation_tokens) + 1
        valid_rows = valid_count + 1
        if valid_rows < rows:
            self.bitmask[valid_rows:rows] = self.bitmask[valid_rows - 1]
        if not self.bitmask[:valid_rows].any(axis=1).all():
            raise NativeError("constraint_error", "output grammar has no valid token")
        return self.bitmask[:rows].tobytes()

    def consume(self, token_ids):
        if any(not 0 <= token < self.VOCABULARY for token in token_ids):
            raise NativeError("constraint_error", "generated token is out of range")
        # LLGuidance's bulk API rejects EOS after a NoExtension stop.
        stopped_eos = (
            len(token_ids) == 1
            and token_ids[0] in self.EOS_TOKENS
            and not self.matcher.is_error()
            and self.matcher.is_stopped()
            and self.matcher.is_accepting()
        )
        valid = (
            self.matcher.consume_token(token_ids[0])
            if stopped_eos
            else self.matcher.consume_tokens(token_ids)
        )
        if not valid:
            raise NativeError(
                "constraint_error", self.matcher.get_error() or "invalid token"
            )


def validate_tokenizer(tokenizer):
    vocabulary = tokenizer.get_vocab()
    if not vocabulary or any(
        type(token_id) is not int or not 0 <= token_id < TokenConstraint.VOCABULARY
        for token_id in vocabulary.values()
    ):
        raise engine_runtime.EngineUnhealthy(
            "tokenizer vocabulary does not fit the native model"
        )
    expected = {
        "<|endoftext|>": TokenConstraint.EOS_TOKENS[0],
        "<|im_end|>": TokenConstraint.EOS_TOKENS[1],
        THINK_END: THINK_END_TOKEN_ID,
    }
    for token, token_id in expected.items():
        if vocabulary.get(token) != token_id or tokenizer.encode(
            token, add_special_tokens=False
        ) != [token_id]:
            raise engine_runtime.EngineUnhealthy(
                f"tokenizer must encode {token!r} as native token {token_id}"
            )
    if tokenizer.eos_token_id not in TokenConstraint.EOS_TOKENS:
        raise engine_runtime.EngineUnhealthy(
            "tokenizer EOS token does not match the native model"
        )


class ConstraintFactory:
    DEFAULT_CACHE_SIZE = 32
    DEFAULT_CACHE_SOURCE_BYTES = 8 * 1024 * 1024

    def __init__(
        self,
        tokenizer,
        cache_size=DEFAULT_CACHE_SIZE,
        cache_source_bytes=DEFAULT_CACHE_SOURCE_BYTES,
    ):
        if (
            not isinstance(cache_size, int)
            or isinstance(cache_size, bool)
            or cache_size <= 0
        ):
            raise ValueError("constraint cache size must be positive")
        if (
            not isinstance(cache_source_bytes, int)
            or isinstance(cache_source_bytes, bool)
            or cache_source_bytes <= 0
        ):
            raise ValueError("constraint cache byte budget must be positive")
        self.tokenizer = guidance_tokenizer(
            tokenizer,
            n_vocab=TokenConstraint.VOCABULARY,
            eos_token=list(TokenConstraint.EOS_TOKENS),
            slices=LLTokenizer.json_slices(),
        )
        self.executor = LLExecutor()
        self.cache_size = cache_size
        self.cache_source_bytes = cache_source_bytes
        self.source_bytes = 0
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.pending = {}
        self.hits = 0
        self.misses = 0

    def create(self, grammar, *, timeout=None):
        matcher = self._matcher(grammar, timeout)
        return TokenConstraint(matcher.deep_copy(), self.executor)

    def _matcher(self, grammar, timeout):
        # Compilation uses the frontend's bounded preparation slots. Share
        # identical misses without blocking unrelated immutable templates.
        with self.lock:
            cached = self.cache.get(grammar)
            if cached is not None:
                self.cache.move_to_end(grammar)
                self.hits += 1
                return cached[0]
            pending = self.pending.get(grammar)
            owner = pending is None
            if owner:
                pending = self.pending[grammar] = Future()
        if not owner:
            if not wait((pending,), timeout=timeout).done:
                raise APIError(504, "request timed out", "request_timeout")
            matcher = pending.result()
            with self.lock:
                if grammar in self.cache:
                    self.cache.move_to_end(grammar)
                self.hits += 1
            return matcher
        try:
            error = LLMatcher.validate_grammar(grammar, self.tokenizer)
            if error:
                raise APIError(400, f"unsupported output schema: {error}")
            matcher = LLMatcher(self.tokenizer, grammar, log_level=0)
            if matcher.is_error():
                raise APIError(400, f"unsupported output schema: {matcher.get_error()}")
            size = len(grammar.encode())
            # Oversized grammars remain usable without displacing the cache.
            # This bounds source bytes; LLGuidance bounds compiler complexity.
            with self.lock:
                self.misses += 1
                if size <= self.cache_source_bytes:
                    self.cache[grammar] = (matcher, size)
                    self.source_bytes += size
                    while (
                        len(self.cache) > self.cache_size
                        or self.source_bytes > self.cache_source_bytes
                    ):
                        _, (_, evicted_size) = self.cache.popitem(last=False)
                        self.source_bytes -= evicted_size
            pending.set_result(matcher)
            return matcher
        except BaseException as error:
            pending.set_exception(error)
            raise
        finally:
            with self.lock:
                del self.pending[grammar]

    def stats(self):
        with self.lock:
            return {
                "entries": len(self.cache),
                "capacity": self.cache_size,
                "source_bytes": self.source_bytes,
                "source_budget_bytes": self.cache_source_bytes,
                "hits": self.hits,
                "misses": self.misses,
            }

"""Tokenizer contract and cached token-level output constraints."""

import threading
from concurrent.futures import Future, wait

from llguidance import LLExecutor, LLMatcher, LLTokenizer
from llguidance.hf import from_tokenizer as guidance_tokenizer
from llguidance.numpy import (
    allocate_token_bitmask,
    fill_next_token_bitmask_par,
    fill_next_token_bitmask_par_with_draft_tokens,
)

from .chat_format import THINK_END, THINK_START, TOOL_CALL_OPEN
from .errors import APIError, ConstraintError
from .lru import LRUCache


class TokenConstraint:
    def __init__(self, matcher, executor, ready):
        self.matcher = matcher
        self.executor = executor
        # From the engine's Ready: the target's vocabulary, which each mask
        # covers; its stop tokens, which end a grammar's output; and the rows
        # of one mask request, before the anchor and after each simulated
        # token.
        self.vocabulary = ready.vocabulary_size
        self.eos_tokens = ready.stop_token_ids
        self.rows = ready.mask_rows
        self.bitmask = allocate_token_bitmask(self.rows, self.vocabulary)
        # Generated batches the grammar has not consumed yet. The reader
        # thread commits them; the mask thread consumes them before the next
        # mask, so reading the native stream never waits for the grammar.
        self._lock = threading.Lock()
        self._committed: list[tuple[int, ...]] = []

    def commit(self, token_ids):
        if any(not 0 <= token < self.vocabulary for token in token_ids):
            raise ConstraintError("generated token is out of range")
        with self._lock:
            self._committed.append(tuple(token_ids))

    def finish(self):
        """Checks the tokens generated after the last mask."""
        self._consume_committed()

    def masks(self, simulation_tokens):
        self._consume_committed()
        if len(simulation_tokens) >= self.rows:
            raise ConstraintError("too many simulation tokens")
        in_range = next(
            (
                index
                for index, token in enumerate(simulation_tokens)
                if not 0 <= token < self.vocabulary
            ),
            len(simulation_tokens),
        )
        valid_count = self.matcher.validate_tokens(list(simulation_tokens[:in_range]))
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
            raise ConstraintError("output grammar has no valid token")
        return self.bitmask[:rows].tobytes()

    def _consume_committed(self):
        with self._lock:
            batches, self._committed = self._committed, []
        for token_ids in batches:
            # LLGuidance's bulk API rejects EOS after a NoExtension stop.
            stopped_eos = (
                len(token_ids) == 1
                and token_ids[0] in self.eos_tokens
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
                # The lines after the first dump the parser state, output
                # included.
                error = self.matcher.get_error()
                raise ConstraintError(
                    error.splitlines()[0] if error else "invalid token"
                )


class TokenizerContractError(ValueError):
    """The model's tokenizer does not encode the tokens its chat format or
    its engine serves by."""


def validate_tokenizer(tokenizer, chat_format):
    """Before the engine starts, refuse a tokenizer that does not encode the
    tags chat_format reads the model's output by as that format has them:
    the close of thinking, the model's own opening of it where the model
    opens it, and a call's opening must each be the format's token, and
    each stop text one token (check_capabilities checks which stop tokens,
    once the engine names them)."""
    vocabulary = tokenizer.get_vocab()
    expected = {
        THINK_END: chat_format.think_end_id,
        TOOL_CALL_OPEN: chat_format.call_open_id,
    }
    if chat_format.think_start_id is not None:
        expected[THINK_START] = chat_format.think_start_id
    for token, token_id in expected.items():
        if vocabulary.get(token) != token_id or tokenizer.encode(
            token, add_special_tokens=False
        ) != [token_id]:
            raise TokenizerContractError(
                f"tokenizer must encode {token!r} as native token {token_id}"
            )
    for stop in chat_format.stop_tokens:
        if len(tokenizer.encode(stop, add_special_tokens=False)) != 1:
            raise TokenizerContractError(
                f"tokenizer must encode the stop text {stop!r} as one token"
            )


def check_capabilities(tokenizer, chat_format, ready):
    """Once the engine is Ready, refuse a tokenizer that does not fit what it
    serves: its vocabulary must fit the engine's, its stop texts encode as
    exactly the engine's stop tokens, its EOS be one of them, and, where the
    engine serves images, the format's image placeholder be one token."""
    vocabulary = tokenizer.get_vocab()
    if not vocabulary or any(
        type(token_id) is not int or not 0 <= token_id < ready.vocabulary_size
        for token_id in vocabulary.values()
    ):
        raise TokenizerContractError(
            "tokenizer vocabulary does not fit the native model"
        )
    stops = tuple(
        tokenizer.encode(stop, add_special_tokens=False)[0]
        for stop in chat_format.stop_tokens
    )
    if sorted(stops) != sorted(ready.stop_token_ids):
        raise TokenizerContractError(
            f"the tokenizer encodes the stop texts {chat_format.stop_tokens} as "
            f"tokens {stops}, but the native model stops at {ready.stop_token_ids}"
        )
    if tokenizer.eos_token_id not in ready.stop_token_ids:
        raise TokenizerContractError(
            "tokenizer EOS token does not match the native model"
        )
    if ready.vision:
        pad = chat_format.image_pad
        if pad is None:
            raise TokenizerContractError(
                "the native model serves images, but its chat template has no "
                "image placeholder"
            )
        if len(tokenizer.encode(pad, add_special_tokens=False)) != 1:
            raise TokenizerContractError(
                f"tokenizer must encode the image placeholder {pad!r} as one token"
            )


def _grammar_error(error):
    # A compiler panic carries a backtrace rather than a reason, and the lines
    # after the first echo the grammar source with every schema it holds.
    if error.startswith("panic"):
        return APIError(400, "tool or output schema is too large to compile")
    return APIError(400, f"unsupported output schema: {error.splitlines()[0]}")


class ConstraintFactory:
    CACHE_SIZE = 32
    CACHE_SOURCE_BYTES = 8 * 1024 * 1024

    def __init__(self, tokenizer, ready):
        # The engine's Ready, whose vocabulary, stop tokens and mask rows its
        # constraints are sized by.
        self.ready = ready
        self.tokenizer = guidance_tokenizer(
            tokenizer,
            n_vocab=ready.vocabulary_size,
            eos_token=list(ready.stop_token_ids),
            slices=LLTokenizer.json_slices(),
        )
        self.executor = LLExecutor()
        # Compiled matchers by grammar, within a budget of grammar bytes.
        self.cache = LRUCache(self.CACHE_SOURCE_BYTES, self.CACHE_SIZE)
        self.lock = threading.Lock()
        self.pending = {}
        self.hits = 0
        self.misses = 0

    def create(self, grammar, *, timeout=None, prefixes=None):
        """`prefixes`, called when the grammar is compiled, returns pairs of
        tokens the output must be able to begin with and the error for a
        grammar that cannot. A grammar can compile and still exceed the
        parser's limits where generation reaches a construct, after the whole
        prompt has been processed."""
        matcher = self._matcher(grammar, timeout, prefixes)
        return TokenConstraint(matcher.deep_copy(), self.executor, self.ready)

    def _matcher(self, grammar, timeout, prefixes):
        # Compilation uses the frontend's bounded preparation slots. Share
        # identical misses without blocking unrelated immutable templates.
        with self.lock:
            cached = self.cache.get(grammar)
            if cached is not None:
                self.hits += 1
                return cached
            pending = self.pending.get(grammar)
            owner = pending is None
            if owner:
                pending = self.pending[grammar] = Future()
        if not owner:
            if not wait((pending,), timeout=timeout).done:
                raise APIError(504, "request timed out", "request_timeout")
            matcher = pending.result()
            with self.lock:
                # The shared matcher, if kept, is now the most recently used.
                self.cache.get(grammar)
                self.hits += 1
            return matcher
        try:
            error = LLMatcher.validate_grammar(grammar, self.tokenizer)
            if error:
                raise _grammar_error(error)
            matcher = LLMatcher(self.tokenizer, grammar, log_level=0)
            if matcher.is_error():
                raise _grammar_error(matcher.get_error())
            for tokens, message in prefixes() if prefixes else ():
                if not matcher.deep_copy().consume_tokens(tokens):
                    raise APIError(400, message)
            size = len(grammar.encode())
            # Oversized grammars remain usable without displacing the cache.
            # This bounds source bytes; LLGuidance bounds compiler complexity.
            with self.lock:
                self.misses += 1
                self.cache.put(grammar, matcher, size)
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
                "capacity": self.cache.capacity,
                "source_bytes": self.cache.bytes,
                "source_budget_bytes": self.cache.budget_bytes,
                "hits": self.hits,
                "misses": self.misses,
            }

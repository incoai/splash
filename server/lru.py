"""The server's caches: values kept within a byte budget, least recently
used first out."""

from collections import OrderedDict


class LRUCache:
    """Values by key, each of a size its owner gives, within `budget_bytes`
    and, if given, `capacity` entries: keeping one drops the least recently
    used until both hold, and a value larger than the whole budget is not
    kept. Its owner holds a lock around every use, as memoized holds the one
    it is given."""

    def __init__(self, budget_bytes, capacity=None):
        self.budget_bytes = budget_bytes
        self.capacity = capacity
        self.bytes = 0
        # The entries dropped to make room.
        self.evictions = 0
        self._entries = OrderedDict()

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        """The keys, the least recently used first."""
        return iter(self._entries)

    def get(self, key):
        """The value kept for `key`, now the most recently used; None when
        none is kept."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key, value, size):
        """Keep `value`, of `size` bytes, for `key` in place of any value
        kept for it; False, keeping nothing, when it alone outgrows the
        budget."""
        if size > self.budget_bytes:
            return False
        self.pop(key)
        self._entries[key] = value, size
        self.bytes += size
        while self.bytes > self.budget_bytes or (
            self.capacity is not None and len(self._entries) > self.capacity
        ):
            _, (_, evicted) = self._entries.popitem(last=False)
            self.bytes -= evicted
            self.evictions += 1
        return True

    def pop(self, key):
        """Drop the value kept for `key`: it, or None when none is kept."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        self.bytes -= entry[1]
        return entry[0]


def memoized(cache, lock, key, compute):
    """The value `cache` keeps for `key`, else the one compute() returns with
    its size, which `cache` then keeps. `lock` guards `cache`, and compute
    runs outside it: misses of one key in several threads each compute it,
    and one that finds a value kept meanwhile returns that value."""
    with lock:
        value = cache.get(key)
    if value is not None:
        return value
    value, size = compute()
    with lock:
        kept = cache.get(key)
        if kept is not None:
            return kept
        cache.put(key, value, size)
    return value

# Paged model-state cache

Target KV uses 32-token pages. Models declare additional cache groups whose
blocks share the target prefix identity:

- A checkpoint group stores an exact state at a token boundary.
- A sliding-window group stores immutable 32-token pages.

Qwen's recurrent checkpoint and DFlash's window are independent groups.
A full-attention target can declare no checkpoint group. The cache does not
interpret model payloads or depend on a recurrent architecture.

## Ownership and recency

`StateGroupCache` owns one group's entries, transfer state, accounting and LRU.
`StateCache` coordinates groups and their dependencies on the target prefix
chain. A restore lease pins the selected blocks and target ancestry. Acquiring,
moving or releasing a lease never changes recency.

The request completion path explicitly touches its target prefix and associated
state records. Checkpoint groups are touched before window dependencies. General memory pressure retains the existing preference for disposable rolling
checkpoints. Allocation retries select the failed group’s least recently used
record regardless of checkpoint status. Pinned entries and active IO cannot
be reclaimed.

Retiring the last checkpoint that needs a window page removes that orphan page.
Shared ancestry, other checkpoint groups and active leases are respected.
Window-only models retain their pages independently.

## Window planning

`planWindow` is the common rule for matching, restore validation and prefill
capture. Its inputs are restore boundary B, prompt end P, window size W,
optional materialization boundaries and the contiguous available old suffix
[A, B).

A boundary X needs the old rows `[max(0, X-W), B)` when X-B is less than W.
New rows come from prefill. Only P is required: an optional checkpoint or branch
boundary is omitted if its old rows are unavailable. Missing old pages therefore
do not prevent a long-suffix hit.

The earliest retained boundary determines which old pages to load. The planner
also produces the capture spans, including where context must be reset. Restore
loads only pages intersecting that old range and records its logical length.
Decode begins only after prefill has produced a complete active window.
A partial restore is never republished as a complete snapshot.

## Allocation and reclaim

The model reports the cache group whose allocation failed. The engine does not
choose a group based on the model architecture.

An optional snapshot first tries RAM. On refusal it reclaims one unpinned entry
from the failed group and retries only if that eviction returned reusable backing
to that group's pool. Shared backing that remains in use does not count. If RAM
still cannot be obtained, the cache reuses an existing disk copy, then attempts a
direct staged disk write, then skips and counts the publication.

Required execution allocations, including draft copy-on-write, first reclaim
within their group. If that cannot satisfy the request, the engine releases
physical target KV backing before its normal wait, yield or failure decision.
The same order applies while host memory pressure pauses growth.

## Draft page storage

`DraftPageTable` holds the active sliding window's page references.
Snapshots retain those references without copying the window. Before writing,
the model obtains private backing for shared pages; exclusive pages stay in
place. Query and verification scratch are not cached.

Each page contains the same 32-token range in every draft layer. Its layer views
hold adjacent K and V tensors. Production attention reads the page address table;
there is no alternate dense-storage path. Kernel compute tiles are independent
of this storage contract.

Idle retention is measured in pages. The allocation scope returns unfinished
allocation attempts without presenting those allocations as reclaim progress.

## Temporary disk offload

Temporary offload is enabled by the existing disk quota. Checkpoint payloads,
draft pages and target KV use fixed-size slot files sharing that quota. Files
are unlinked temporary storage and do not survive server restart.

A single `StateWrite<Payload>` contract describes writes of either one group
block or a complete restore boundary. Transfer destruction drains outstanding
IO. Disk copies can remain beside RAM copies and be reused during later pressure;
shared immutable draft pages are not written again while their disk copy exists.

The same group index also supports [persistent prefixes](../DEVELOPMENT.md#persistent-prefixes).
Persistence owns manifest admission and durable slot references; matching,
restoration, group eviction and temporary offload remain cache responsibilities.
The server protocol remains version 7.

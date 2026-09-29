# Cache groups

Target KV is indexed by the existing token-prefix tree. A model declares the
additional data needed to resume that target through `CacheGroupSpec`:

- A checkpoint group needs one payload at the resume boundary.
- A sliding-window group needs continuous coverage of its declared window.
- An empty declaration needs only target KV.

Qwen currently declares one recurrent checkpoint group and one DFlash window
group. Neither group owns the other. A future full-attention target can omit the
checkpoint group; its draft can still declare a window independently. Supporting
a new target also requires its execution implementation—this contract alone does
not add model support.

## Ownership and lookup

`StatePayload` is an immutable model-owned block. Its group, logical range and
opaque payload form a `CachedStateBlock`. Groups own disjoint physical resources;
within one group, several logical blocks may share backing. Resource accounting
counts that backing once. The model owns allocation geometry, serialization and
restoration; the engine does not interpret GDN or draft buffer layouts.

Each `StateGroupCache` controls its own block residency, transfers and eviction
eligibility. All groups and target KV use the same recency clock. A block is
eligible for eviction only without active restore pins. Disposable progress
checkpoints remain lower priority than ordinary cached conversation states.
Evicting a group block does not evict its sibling group at the same target
endpoint. Explicit retirement of a rolling checkpoint removes only exact-state
checkpoint groups: its window blocks may still serve newer boundaries. Those
blocks remain independently evictable. Logical removal of target KV removes
every state dependent on it.

Request completion refreshes complete restore points and every window fragment
they need. Shared fragments inherit the access of all dependent points, rather
than becoming artificially older because they occur earlier in the token chain.
Incomplete points do not refresh their remaining payloads. This applies with
offloading disabled as well as with either disk tier enabled. No model IDs or
payload-size thresholds participate in this policy. Snapshot admission retains
global LRU across groups and bounds live-cache displacement by model-reported
new allocation bytes, excluding shared pages
(rounded to the last victim). An occupied staging buffer leaves an uncopied old
state in RAM instead of discarding it to attempt a new snapshot. When a
new resident snapshot cannot fit, an existing disk copy is reused or the current
boundary is captured directly to disk before recycling live cached states.
This keeps the old restore points and avoids occupying staging just to free RAM
for a second copy. If the disk path is unavailable, bounded RAM recycling remains
the fallback.
Opportunistic hints still cannot displace cached work, and disposable checkpoints
only reclaim disposable entries.

Group stores expose exact block acquisition and own residency, transfers and
recency. They do not search for restore boundaries, publish whole snapshots or
count request hits; those responsibilities belong to the coordinating cache.

`CacheGroupCoordinator` searches only the matched target ancestry and returns the
deepest boundary satisfying every declared group. A missing window fragment
reduces the candidate boundary; an incomplete combination is never a cache hit.
`RestoreLease` pins every selected group block and the target endpoint before
execution-memory admission. Maintenance leases preserve access recency. During
speculative memory shrink, the cache protects one complete resident restore
bundle; demand-driven admission can reclaim it.

DFlash's circular execution table uses BF16 pages. A non-page-aligned window can
have two logical fragments in the same circular slot. Those fragments may come
from different snapshots. Qwen shares unchanged resident pages, reads disk pages
into private destinations, and merges only differing boundary fragments. COW
protects cached pages from subsequent execution writes. A failed or cancelled
restore cannot publish a promoted cache copy. Restore scratch allocations use the
same execution admission path as KV growth: the complete restore remains pinned
while reclaim retries. Transfers start only after restore allocations succeed.
If no further reclaim, pending transfer or other resident request can make the
restore fit, admission retries cold rather than failing the engine or repeatedly
attempting the same oversized restore. Host pressure and pending IO retain their ordinary wait rules.

## Disk storage and persistence

Temporary offloading and persistent caching retain independent switches and
quotas. Both use the same payload representation and transfer machinery. A
restored resident payload keeps its disk record, avoiding another write on the
next eviction. A narrower resident publication preserves an existing wider disk
range, so older restore points and pinned manifests retain complete coverage.
Model staging is bounded across groups.

Persistent manifests describe the complete target prefix and each required
group block: group ID, logical range, model metadata and opaque disk records.
There is no distinguished mandatory recurrent record. The model reopens its
records, including same-sized records belonging to different backing files.
The manifest is committed only after all payloads finish writing. Runtime RAM
eviction remains per group block; durable quota eviction removes a complete
manifest, with shared records released at their final owner.

The format namespace is `splash-prefix-cache-v3-groups-crc32c`. Files written with
the earlier composite manifest are safely reset and warm again; there is no
in-place conversion of those development cache records. The file path, capacity
options and four supported switch combinations are unchanged.

## Validation

`cache-groups` covers KV-only, window-only and multiple-group models, incomplete
windows, branch isolation, independent eviction, complete-window protection and
randomized boundary search against exhaustive coverage, shared-window checkpoint
retirement and dependency-aware recency with different group declaration orders. `persistent-cache`
reopens those model declarations across store lifetimes and checks payload bytes.
`qwen-state-storage` checks RAM/disk boundary merges, COW, cancellation and allocation
failures. Real-model oracle and serving tests cover numerical/output equivalence,
restart, corruption, quota pressure, concurrent requests and agent traces.

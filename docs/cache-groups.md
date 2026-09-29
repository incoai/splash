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
Evicting a group block preserves unrelated groups. Removing an exact-state
checkpoint frees window blocks that no compatible descendant can use with all
required checkpoint groups present. The prefix tree supplies that liveness
check; no second ownership index is needed. Window-only models retain their
independent block lifetime. Shared windows remain independently evictable and are retained across rolling
checkpoint replacement and branch changes. Logical removal of target KV removes
every state dependent on it.

A real restore refreshes the selected checkpoint and its window dependencies.
Request completion also refreshes complete restore points along that request's
path. This deliberately retains historical recovery points for prefix branching
and revisits; it is not a claim that every checkpoint was read during execution.
Shared blocks inherit the access of complete dependent points; incomplete
points do not refresh orphaned payloads. This applies with offloading disabled as
well as with either disk tier enabled. No model IDs or payload-size thresholds
participate in this policy. The shared recency clock operates with these retention
and reclaim priorities, not as an unconditional ordering of all allocations.

Snapshot admission bounds live-cache displacement by model-reported new
allocation bytes, excluding shared pages (rounded to the last victim). An
occupied staging buffer leaves an uncopied old state in RAM instead of discarding
it to attempt a new snapshot. When a new resident snapshot cannot fit, an
existing disk copy is reused or the current boundary is considered for direct
disk capture before recycling live cached states. If the disk path is unavailable
or declines the candidate, bounded RAM recycling remains the fallback.
Opportunistic hints cannot displace cached work, and disposable checkpoints
only reclaim disposable entries.

The model identifies which group can supply reusable execution backing.
Required execution first reclaims that group, including under host pressure.
When an allocation instead needs physical budget, target KV reclaim releases
backing rather than just returning page IDs to its pool.

Group stores expose exact block acquisition and own residency, transfers and
recency. They do not search for restore boundaries, publish whole snapshots or
count request hits; those responsibilities belong to the coordinating cache.

`CacheGroupCoordinator` searches only the matched target ancestry and returns the
deepest boundary satisfying every declared group. A missing window block
reduces the candidate boundary; an incomplete combination is never a cache hit.
`RestoreLease` pins every selected group block and the target endpoint before
execution-memory admission. Request leases release checkpoints before window
dependencies so recency does not depend on container destruction order.
Maintenance leases preserve recency. During
speculative memory shrink, the cache protects one complete resident restore
bundle; demand-driven admission can reclaim it.

DFlash's circular execution table uses 32-token BF16 pages, matching target KV
boundaries. Each cached draft record covers one aligned page. RAM restore shares
complete pages; disk restore overwrites exclusive destinations or allocates
private pages for shared destinations. COW protects cached pages from subsequent execution writes. A failed or cancelled
restore cannot publish a promoted cache copy. Restore scratch allocations use the
execution admission path: the complete restore remains pinned
while reclaim retries. Transfers start only after restore allocations succeed.
If no further reclaim, pending transfer or other resident request can make the
restore fit, admission retries cold rather than failing the engine or repeatedly
attempting the same oversized restore. Host pressure and pending IO retain
their ordinary wait rules.

## Disk storage and persistence

Temporary offloading and persistent caching retain independent switches and
quotas. Both use the same payload representation and transfer machinery. A
restored resident payload keeps its disk record, avoiding another write on the
next eviction. A narrower resident publication preserves an existing wider disk
range, so older restore points and pinned manifests retain complete coverage.
Model staging is bounded across groups.

Payloads use uncached positional IO. SQLite indexes committed durable extents;
temporary writes neither journal payloads nor commit per-slot metadata. Payload
synchronization precedes manifest commit. On restart, the store reconstructs
free extents, truncates the unused tail and punches free holes on APFS, without
relocating live payloads.

Direct capture has two synchronous phases. The model prepares a
`SnapshotWritePlan` describing the source backing and a writer; preparation
allocates no payload records and starts no IO. The source may borrow committed
lane buffers, so the plan must be consumed before the lane executes again and
must never enter a background queue. The writer stages borrowed bytes before
returning a transfer ticket.

`PersistentCache` applies the same size, reuse and write-credit admission to a
RAM snapshot or a direct-write plan, before either path may evict durable data.
Replay endpoints and observed junctions record demand before attempting a
snapshot. A bounded history of 4096 prefix fingerprints remembers whether a
boundary appeared in distinct request lifetimes, even if its payload and KV
blocks were rejected or evicted. It holds neither backing nor pins, counts no
retry twice, and forgets the least recently observed fingerprint when full.
The scheduler supplies its existing submission sequence, which remains stable
across resource suspension and distinct when a client request ID is reused.
Predicted hints and disposable progress checkpoints do not populate this
history: repeatedly predicting a useful state is not observed reuse. Fingerprints
affect admission only; full-key matching and complete restore coverage remain
mandatory. History is process-local and resets on restart; durable payloads
and their persisted recency retain the existing restart behavior.

Under quota pressure, repeated demand can admit a candidate without requiring
a prior successful cache hit. Actual hits and existing branch points remain
eligible. One-use tails still cannot displace durable prefixes, and repeated
demand does not bypass size, minimum length, write credits or dependency pins.
The demand history does not promote payload LRU positions.

An accepted direct capture enters the same bounded publication job and is not
admitted or charged a second time. A rejected candidate may use temporary space
when offloading is enabled, but that fallback cannot evict a durable manifest.
Ordinary snapshot RAM recycling applies the same restriction to any offload it starts, so it cannot
bypass admission indirectly after a declined direct capture.
Optional checkpoints use the same durable-preserving rule. Required execution
allocations and pressure-driven offloads may reclaim durable quota. Publisher
pacing includes accepted direct capture's source writes and missing target pages. The default allowance is 256 GiB/hour with
an initial burst of one cache capacity; it accommodates bursty agent reuse while
bounding sustained optional payload writes. It is not a cap on temporary offload
or all process/SSD writes.

Persistent manifests describe the complete target prefix and each required
group block: group ID, logical range, model metadata and opaque disk records.
There is no distinguished mandatory recurrent record. The model reopens its
records, including same-sized records belonging to different backing files.
The manifest is committed only after all payloads finish writing. Runtime RAM
eviction remains per group block; durable quota eviction removes a complete
manifest, with shared records released at their final owner.

Manifest compatibility uses `splash-prefix-cache-v3-groups-crc32c`; storage v2
pairs the SQLite index with an uncached `.data` file. The earlier SQLite-blob
storage is not reopened or converted. Explicit incompatible files are left in
place and serving falls back to RAM and configured temporary offload. The four
switch combinations remain independent.

## Validation

`cache-groups` covers KV-only, window-only and multiple-group models, incomplete
windows, branch isolation, independent eviction, complete-window protection and
randomized boundary search against exhaustive coverage, shared-window checkpoint
retirement, quota callbacks that retire the write source, and dependency-aware
recency with different group declaration orders. `persistent-cache` reopens those model declarations across store lifetimes and checks payload bytes,
as well as rejected direct candidates under reuse, size and write-credit pressure
with temporary offloading disabled, available and full.
`qwen-state-storage` checks aligned RAM/disk restoration, COW, cancellation and allocation
failures. Real-model oracle and serving tests cover numerical/output equivalence,
restart, corruption, quota pressure, concurrent requests and agent traces.

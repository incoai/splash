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
endpoint. Logical removal of target KV removes every state dependent on it.

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
restore cannot publish a promoted cache copy.

## Disk storage and persistence

Temporary offloading and persistent caching retain independent switches and
quotas. Both use the same payload representation and transfer machinery. A
restored resident payload keeps its disk record, avoiding another write on the
next eviction. Model staging is bounded across groups.

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
randomized boundary search against exhaustive coverage. `persistent-cache`
reopens those model declarations across store lifetimes and checks payload bytes.
`qwen-state-storage` checks RAM/disk boundary merges, COW, cancellation and allocation
failures. Real-model oracle and serving tests cover numerical/output equivalence,
restart, corruption, quota pressure, concurrent requests and agent traces.

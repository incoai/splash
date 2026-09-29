#include "engine/StateCache.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace splash::engine {

RestoreLease::RestoreLease(KvCache &kv, uint64_t block, CacheAccess access,
                           std::shared_ptr<const RestoreState> state,
                           std::vector<StateBlockLease> leases)
    : kv_(&kv), block_(block), access_(access), state_(std::move(state)),
      leases_(std::move(leases)) {
  kv_->retainActive(block_, access_);
}
RestoreLease::RestoreLease(RestoreLease &&other) noexcept
    : kv_(std::exchange(other.kv_, nullptr)), block_(other.block_),
      access_(other.access_), state_(std::move(other.state_)),
      leases_(std::move(other.leases_)) {}
RestoreLease &RestoreLease::operator=(RestoreLease &&other) noexcept {
  if (this != &other) {
    reset();
    kv_ = std::exchange(other.kv_, nullptr);
    block_ = other.block_;
    access_ = other.access_;
    state_ = std::move(other.state_);
    leases_ = std::move(other.leases_);
  }
  return *this;
}
RestoreLease::~RestoreLease() noexcept { reset(); }
void RestoreLease::reset() noexcept {
  // A request accesses checkpoint groups before their window dependencies.
  // Release explicitly in that order; vector destruction order is not policy.
  for (auto &lease : leases_)
    lease.reset();
  leases_.clear();
  state_.reset();
  if (kv_)
    kv_->releaseActive(block_, access_);
  kv_ = nullptr;
  block_ = 0;
}

StateCache::StateCache(KvCache &kv, CacheRecency &recency)
    : kv_(kv), recency_(recency), coordinator_({{0}}) {
  configure({{0}});
}
void StateCache::configure(std::vector<CacheGroupSpec> specs) {
  CacheGroupCoordinator coordinator(specs);
  if (std::equal(coordinator.groups().begin(), coordinator.groups().end(),
                 coordinator_.groups().begin(), coordinator_.groups().end()) &&
      groups_.size() == specs.size())
    return;
  for (const auto &[_, store] : groups_)
    if (store->snapshot().entries || store->writing())
      throw std::logic_error("cache groups cannot change after publication");
  std::map<CacheGroupId, std::unique_ptr<StateGroupCache>> groups;
  for (const auto &spec : specs) {
    auto store = std::make_unique<StateGroupCache>(kv_, recency_, spec.id);
    store->setOffloadEnabled(offloadEnabled_);
    if (spec.kind == CacheGroupKind::Checkpoint)
      store->setRemovalHandler(
          [this](uint64_t leaf) { return pruneWindowDependencies(leaf); });
    groups.emplace(spec.id, std::move(store));
  }
  groups_ = std::move(groups);
  coordinator_ = std::move(coordinator);
}
StateGroupCache &StateCache::group(CacheGroupId id) { return *groups_.at(id); }
const StateGroupCache &StateCache::group(CacheGroupId id) const {
  return *groups_.at(id);
}
uint64_t StateCache::endpoint(uint64_t leaf, uint32_t boundary) const {
  if (boundary % KvCache::pageTokens)
    throw std::invalid_argument(
        "state block is not aligned to target prefix identity");
  return kv_.ancestor(leaf, boundary / KvCache::pageTokens);
}
std::optional<RestoreState> StateCache::match(std::span<const uint64_t> chain,
                                              bool residentOnly) const {
  if (chain.empty())
    return std::nullopt;
  // Some callers provide just a candidate endpoint; matching always checks its
  // actual ancestry, never unrelated blocks with equal token positions.
  const auto full = kv_.chain(chain.back());
  return coordinator_.match(
      full.blocks.size() * KvCache::pageTokens, KvCache::pageTokens,
      [&](CacheGroupId id, uint32_t end) {
        if (!end || end % KvCache::pageTokens ||
            end / KvCache::pageTokens > full.blocks.size())
          return std::optional<CachedStateBlock>{};
        auto part = group(id).peek(full.blocks[end / KvCache::pageTokens - 1]);
        if (residentOnly && part &&
            part->payload->residentBytes() != part->payload->bytes())
          return std::optional<CachedStateBlock>{};
        return part;
      });
}
uint32_t StateCache::matchedBoundary(std::span<const uint64_t> chain) const {
  const auto state = match(chain);
  return state ? state->boundary : 0;
}
std::optional<RestoreLease>
StateCache::acquireDeepest(std::span<const uint64_t> chain,
                           CacheAccess access) {
  auto matched = match(chain);
  if (!matched)
    return std::nullopt;
  const uint64_t leaf = endpoint(chain.back(), matched->boundary);
  auto state = std::make_shared<RestoreState>(std::move(*matched));
  std::vector<StateBlockLease> leases;
  leases.reserve(state->blocks.size());
  for (auto &block : state->blocks) {
    const uint64_t id = endpoint(leaf, block.end);
    auto lease = group(block.group).acquireBlock(id, access);
    if (!lease || lease->kvBlock() != id)
      throw std::logic_error("cache group changed during restore acquisition");
    block.payload = lease->state();
    leases.push_back(std::move(*lease));
  }
  return RestoreLease(kv_, leaf, access, std::move(state), std::move(leases));
}
std::optional<RestoreLease> StateCache::acquireResumePoint() {
  std::optional<CacheEvictionCandidate> selected;
  bool selectedCheckpoint = false;
  const bool hasCheckpoint = std::any_of(
      coordinator_.groups().begin(), coordinator_.groups().end(),
      [](const auto &spec) { return spec.kind == CacheGroupKind::Checkpoint; });
  for (const auto &spec : coordinator_.groups()) {
    if (hasCheckpoint && spec.kind != CacheGroupKind::Checkpoint)
      continue;
    const auto &store = groups_.at(spec.id);
    const auto block = store->resumePoint();
    if (!block)
      continue;
    const auto state = match(std::span(&block, 1), true);
    if (!state || !state->resident())
      continue;
    const auto leaf = endpoint(block, state->boundary);
    const auto candidate = store->candidate(leaf);
    if (!candidate)
      continue;
    const bool checkpoint = bool(store->checkpoint(leaf));
    if (!selected || (selectedCheckpoint && !checkpoint) ||
        (selectedCheckpoint == checkpoint &&
         candidate->lastUsed > selected->lastUsed)) {
      selected = candidate;
      selectedCheckpoint = checkpoint;
    }
  }
  if (!selected)
    return std::nullopt;
  // One lease protects the entire common restore boundary, including every
  // window block and its target ancestry, for this speculative shrink.
  return acquireDeepest(std::span(&selected->id, 1), CacheAccess::Maintenance);
}
bool StateCache::isCheckpoint(
    const CacheEvictionCandidate &candidate) const noexcept {
  return candidate.group &&
         bool(group(*candidate.group).checkpoint(candidate.id));
}
void StateCache::recordLookup(bool hit, bool disk) noexcept {
  hit ? ++hits_ : ++misses_;
  if (disk)
    ++diskHits_;
}
void StateCache::setOffloadEnabled(bool enabled) noexcept {
  offloadEnabled_ = enabled;
  for (auto &[_, store] : groups_)
    store->setOffloadEnabled(enabled);
}
bool StateCache::contains(uint64_t block) const noexcept {
  return std::any_of(groups_.begin(), groups_.end(), [block](const auto &g) {
    return g.second->contains(block);
  });
}
bool StateCache::resident(uint64_t block) const noexcept {
  return std::any_of(groups_.begin(), groups_.end(), [block](const auto &g) {
    return g.second->resident(block);
  });
}
void StateCache::touch(std::span<const uint64_t> chain) noexcept {
  const auto find = [&](CacheGroupId id,
                        uint32_t end) -> std::optional<CachedStateBlock> {
    if (!end || end % KvCache::pageTokens ||
        end / KvCache::pageTokens > chain.size())
      return {};
    const auto block = chain[end / KvCache::pageTokens - 1];
    return kv_.contains(block) ? group(id).peek(block) : std::nullopt;
  };
  const auto visit = [&](const CachedStateBlock &part) {
    group(part.group).touch(chain[part.end / KvCache::pageTokens - 1]);
  };
  // Reuse follows complete restore points, not physical token order. A shared
  // window page inherits every dependent point's access; incomplete points do
  // not keep otherwise unusable payloads hot. Reference wrappers avoid heap
  // allocation while finishing a request, including failure cleanup.
  const CacheGroupCoordinator::Find lookup = std::cref(find);
  const CacheGroupCoordinator::Visit refresh = std::cref(visit);
  for (size_t i = 0; i < chain.size(); ++i)
    coordinator_.visitComplete((i + 1) * KvCache::pageTokens, lookup, refresh);
}
bool StateCache::touchIfResident(uint64_t block, bool checkpoint) {
  const auto state = match(std::span(&block, 1));
  if (!state ||
      state->boundary != kv_.chainLength(block) * KvCache::pageTokens ||
      !state->resident())
    return false;
  return touchIfStored(block, checkpoint);
}
bool StateCache::touchIfStored(uint64_t block, bool checkpoint) {
  const auto state = match(std::span(&block, 1));
  if (!state || state->boundary != kv_.chainLength(block) * KvCache::pageTokens)
    return false;
  for (const auto &part : state->blocks)
    static_cast<void>(
        group(part.group).touchIfStored(endpoint(block, part.end), checkpoint));
  return true;
}
void StateCache::validate(uint64_t leaf, const RestoreState &state) const {
  if (!kv_.contains(leaf) ||
      state.boundary != kv_.chainLength(leaf) * KvCache::pageTokens)
    throw std::invalid_argument("restore boundary has no target prefix");
  std::set<std::pair<CacheGroupId, uint32_t>> endpoints;
  for (const auto &part : state.blocks) {
    const auto specs = coordinator_.groups();
    const auto spec =
        std::find_if(specs.begin(), specs.end(),
                     [&](const auto &g) { return g.id == part.group; });
    if (spec == specs.end() || !part.payload || !part.payload->bytes() ||
        !part.end || part.end > state.boundary || part.begin > part.end ||
        part.begin % KvCache::pageTokens || part.end % KvCache::pageTokens ||
        (spec->kind == CacheGroupKind::Checkpoint ? part.begin != part.end
                                                  : part.begin == part.end) ||
        !endpoints.emplace(part.group, part.end).second)
      throw std::invalid_argument("invalid cache group block");
  }
}
void StateCache::publish(uint64_t leaf,
                         std::shared_ptr<const RestoreState> state,
                         bool checkpoint) {
  if (!state || !kv_.contains(leaf) ||
      state->boundary != kv_.chainLength(leaf) * KvCache::pageTokens ||
      !state->resident())
    throw std::invalid_argument("invalid resident restore state");
  validate(leaf, *state);
  if (const auto existing = match(std::span(&leaf, 1));
      !groups_.empty() && existing && existing->boundary == state->boundary &&
      existing->resident())
    throw std::logic_error("duplicate resident restore state");
  for (const auto &part : state->blocks) {
    if (!part.payload || part.begin > part.end || part.end > state->boundary)
      throw std::invalid_argument("invalid cache group block");
    auto &store = group(part.group);
    const uint64_t block = endpoint(leaf, part.end);
    const auto existing = store.peek(block);
    // One descriptor serves both tiers. A narrower RAM slice cannot replace
    // a wider disk range: earlier restore points (including pinned manifests)
    // still depend on that coverage. Equal ranges may acquire a RAM copy.
    if (existing && existing->begin <= part.begin &&
        (store.resident(block) || existing->begin < part.begin)) {
      static_cast<void>(store.touchIfStored(block, checkpoint));
      continue;
    }
    if (existing && existing->begin > part.begin && !store.evict(block).evicted)
      continue;
    store.publish(block, part.payload, checkpoint, part.begin);
  }
}
void StateCache::importDisk(uint64_t leaf,
                            std::shared_ptr<const RestoreState> state) {
  if (!state || state->boundary != kv_.chainLength(leaf) * KvCache::pageTokens)
    throw std::invalid_argument("invalid persistent restore state");
  validate(leaf, *state);
  for (const auto &part : state->blocks)
    group(part.group)
        .importDisk(endpoint(leaf, part.end), part.payload, part.begin);
}
bool StateCache::publishToDisk(uint64_t leaf, const StateWriter &write,
                               const std::function<void()> &completion,
                               const std::function<bool()> &makeRoom,
                               bool checkpoint) {
  if (touchIfStored(leaf, checkpoint))
    return true;
  if (writing())
    return false;
  auto transfer = write(completion);
  while (!transfer && makeRoom && makeRoom())
    transfer = write(completion);
  if (!transfer)
    return false;
  const auto state = transfer->state();
  if (!state)
    throw std::invalid_argument("empty disk snapshot");
  validate(leaf, *state);
  PendingSnapshot pending{leaf, std::move(transfer), {}};
  pending.blocks.reserve(state->blocks.size());
  try {
    for (const auto &part : state->blocks) {
      auto &store = group(part.group);
      const auto block = endpoint(leaf, part.end);
      if (store.importDisk(block, part.payload, part.begin, checkpoint)) {
        pending.blocks.push_back({part.group, block, part.payload});
        store.setExternalWrite(block, part.payload.get(), true);
      }
    }
    pending_ = std::move(pending);
    ++directOffloads_;
    ++directPublications_;
  } catch (...) {
    for (const auto &[id, block, payload] : pending.blocks) {
      group(id).setExternalWrite(block, payload.get(), false);
      group(id).invalidate(block, payload.get());
    }
    throw;
  }
  return true;
}

std::shared_ptr<const RestoreState> StateCache::diskCopy(uint64_t leaf) const {
  const auto chain = kv_.chain(leaf);
  const uint32_t boundary = chain.blocks.size() * KvCache::pageTokens;
  const auto find = [&](CacheGroupId id,
                        uint32_t end) -> std::optional<CachedStateBlock> {
    if (!end || end % KvCache::pageTokens || end > boundary)
      return {};
    const auto block = chain.blocks[end / KvCache::pageTokens - 1];
    auto part = group(id).peek(block);
    if (!part)
      return {};
    part->payload = group(id).diskCopy(block);
    return part->payload ? part : std::nullopt;
  };
  if (!coordinator_.complete(boundary, find))
    return {};
  return std::make_shared<RestoreState>(
      *coordinator_.match(boundary, KvCache::pageTokens, find));
}
bool StateCache::copyToDisk(uint64_t leaf,
                            const std::function<void()> &completion,
                            const std::function<bool()> &makeRoom) {
  if (diskCopy(leaf))
    return true;
  if (writing())
    return false;
  const auto state = match(std::span(&leaf, 1));
  if (!state || state->boundary != kv_.chainLength(leaf) * KvCache::pageTokens)
    return false;
  for (const auto &part : state->blocks) {
    auto &store = group(part.group);
    const auto block = endpoint(leaf, part.end);
    if (!store.diskCopy(block))
      return store.copyToDisk(block, completion, makeRoom);
  }
  return true;
}
uint64_t StateCache::pruneWindowDependencies(uint64_t leaf) noexcept {
  const auto hasCheckpoint = [&](uint64_t block) {
    for (const auto &spec : coordinator_.groups())
      if (spec.kind == CacheGroupKind::Checkpoint &&
          !group(spec.id).contains(block))
        return false;
    return true;
  };
  const std::function<bool(uint64_t)> needed = std::cref(hasCheckpoint);
  const uint32_t depth = kv_.chainLength(leaf);
  uint64_t reclaimed = 0;
  for (const auto &spec : coordinator_.groups()) {
    if (spec.kind != CacheGroupKind::SlidingWindow)
      continue;
    auto &store = group(spec.id);
    const uint32_t span = (spec.windowTokens - 1) / KvCache::pageTokens;
    // A block ending at d can serve only descendant checkpoints whose
    // window starts before d. Other branches and overlapping windows remain
    // discoverable in the existing prefix tree; no second ownership graph.
    uint64_t block = leaf;
    for (uint32_t d = depth; d && depth - d <= span; --d) {
      if (store.contains(block) &&
          !kv_.anyDescendant(block, d + span, needed)) {
        const auto eviction = store.evict(block);
        reclaimed += eviction.reclaimedBytes;
        // A failed restore can still pin the window while its checkpoint
        // lease is released. Existing readers keep their payload; defer
        // removal until the last lease leaves, as for other invalid states.
        if (!eviction.evicted)
          if (const auto unused = store.peek(block))
            store.invalidate(block, unused->payload.get());
      }
      block = d > 1 ? kv_.ancestor(block, d - 1) : 0;
    }
  }
  return reclaimed;
}

StateCheckpoint StateCache::checkpoint(uint64_t block) const {
  StateCheckpoint result;
  // Removing the exact checkpoint also releases window blocks that no other
  // checkpoint uses. Publication identity protects upgraded/replaced entries.
  for (const auto &spec : coordinator_.groups())
    if (spec.kind == CacheGroupKind::Checkpoint)
      if (auto point = group(spec.id).checkpoint(block)) {
        result.kvBlock = block;
        result.groups.emplace_back(spec.id, point);
      }
  return result;
}
bool StateCache::retireCheckpoint(const StateCheckpoint &point) noexcept {
  bool retired = true;
  for (const auto &[id, checkpoint] : point.groups)
    retired = group(id).retireCheckpoint(checkpoint) && retired;
  return retired;
}
std::optional<CacheEvictionCandidate> StateCache::evictionCandidate(
    bool checkpoints, std::optional<CacheGroupId> selected) const noexcept {
  std::optional<CacheEvictionCandidate> result;
  bool resultCheckpoint = false;
  for (const auto &[id, store] : groups_) {
    if (selected && *selected != id)
      continue;
    const auto candidate = store->evictionCandidate(checkpoints);
    if (!candidate)
      continue;
    const bool checkpoint = bool(store->checkpoint(candidate->id));
    if (!result || (checkpoint && !resultCheckpoint) ||
        (checkpoint == resultCheckpoint &&
         candidate->lastUsed < result->lastUsed)) {
      result = candidate;
      resultCheckpoint = checkpoint;
    }
  }
  return result;
}
std::optional<CacheGroupId>
StateCache::oldestGroup(uint64_t block, bool disk,
                        bool duplicate) const noexcept {
  std::optional<CacheEvictionCandidate> oldest;
  std::optional<CacheGroupId> result;
  for (const auto &[id, store] : groups_)
    if (const auto candidate = store->candidate(block, disk, duplicate);
        candidate && (!oldest || candidate->lastUsed < oldest->lastUsed)) {
      oldest = candidate;
      result = id;
    }
  return result;
}
std::optional<CacheEvictionCandidate>
StateCache::diskCandidate(bool duplicate) const noexcept {
  std::optional<CacheEvictionCandidate> result;
  for (const auto &[_, store] : groups_)
    if (const auto candidate = store->diskCandidate(duplicate);
        candidate && (!result || candidate->lastUsed < result->lastUsed))
      result = candidate;
  return result;
}
StateEviction StateCache::reclaim(uint64_t block,
                                  std::function<void()> completion,
                                  const std::function<bool()> &makeRoom,
                                  std::optional<CacheGroupId> selected) {
  const auto id = selected ? selected : oldestGroup(block);
  if (!id)
    return {};
  // Shared model staging is bounded across groups, not separately per group.
  const auto payload = group(*id).peek(block);
  if (offloadEnabled_ && writing() && payload &&
      payload->payload->canOffload() && !group(*id).diskCopy(block) &&
      !group(*id).writing(block)) {
    return {false, 0, true};
  }
  return group(*id).reclaim(block, std::move(completion), makeRoom);
}
StateEviction StateCache::evict(uint64_t block) noexcept {
  StateEviction result;
  for (auto &[_, store] : groups_) {
    const auto removed = store->evict(block);
    result.evicted |= removed.evicted;
    result.reclaimedBytes += removed.reclaimedBytes;
  }
  return result;
}
void StateCache::dropDisk(uint64_t block,
                          std::optional<CacheGroupId> selected) {
  if (const auto id = selected ? selected : oldestGroup(block, true, true))
    group(*id).dropDisk(block);
}
void StateCache::evictDiskOnly(uint64_t block,
                               std::optional<CacheGroupId> selected) {
  if (const auto id = selected ? selected : oldestGroup(block, true, false))
    static_cast<void>(group(*id).evict(block));
}
void StateCache::invalidate(uint64_t leaf, const RestoreState *state) noexcept {
  if (!state)
    return;
  for (const auto &part : state->blocks)
    group(part.group).invalidate(endpoint(leaf, part.end), part.payload.get());
}
void StateCache::invalidate(uint64_t block) noexcept {
  for (auto &[_, store] : groups_)
    store->invalidate(block);
}
bool StateCache::promotable(uint64_t leaf,
                            const RestoreState *state) const noexcept {
  return state && std::any_of(state->blocks.begin(), state->blocks.end(),
                              [&](const auto &part) {
                                return group(part.group)
                                    .promotable(endpoint(leaf, part.end),
                                                part.payload.get());
                              });
}
void StateCache::promote(uint64_t leaf, const RestoreState *source,
                         std::shared_ptr<const RestoreState> state) {
  if (!source || !state || source->boundary != state->boundary)
    throw std::invalid_argument("incompatible restore promotion");
  for (const auto &part : state->blocks) {
    const auto old = std::find_if(
        source->blocks.begin(), source->blocks.end(), [&](const auto &value) {
          return value.group == part.group && value.begin == part.begin &&
                 value.end == part.end;
        });
    if (old != source->blocks.end())
      group(part.group)
          .promote(endpoint(leaf, part.end), old->payload.get(), part.payload);
  }
}
bool StateCache::writing(uint64_t leaf) const noexcept {
  if (pending_ && pending_->leaf == leaf)
    return true;
  // A manifest can depend on an ancestor block in any group.
  while (leaf) {
    for (const auto &[_, store] : groups_)
      if (store->writing(leaf))
        return true;
    const auto depth = kv_.chainLength(leaf);
    leaf = depth > 1 ? kv_.ancestor(leaf, depth - 1) : 0;
  }
  return false;
}
bool StateCache::writing() const noexcept {
  return pending_ ||
         std::any_of(groups_.begin(), groups_.end(),
                     [](const auto &g) { return g.second->writing(); });
}
bool StateCache::pollOffload() {
  bool progress = false;
  if (pending_ && pending_->transfer->ready()) {
    auto done = std::move(*pending_);
    pending_.reset();
    const bool success = done.transfer->finish();
    if (!success)
      ++directFailures_;
    for (const auto &[id, block, payload] : done.blocks) {
      group(id).setExternalWrite(block, payload.get(), false);
      if (!success)
        group(id).invalidate(block, payload.get());
    }
    progress = true;
  }
  for (auto &[_, store] : groups_)
    progress = store->pollOffload() || progress;
  return progress;
}
StateCacheSnapshot StateCache::snapshot() const noexcept {
  StateCacheSnapshot result;
  for (const auto &[_, store] : groups_) {
    const auto s = store->snapshot();
    result.entries += s.entries;
    result.pinned += s.pinned;
    result.bytes += s.bytes;
    result.diskBytes += s.diskBytes;
    result.offloads += s.offloads;
    result.offloadFailures += s.offloadFailures;
    result.invalidations += s.invalidations;
    result.promotions += s.promotions;
    result.publications += s.publications;
    result.deduplicatedPublications += s.deduplicatedPublications;
    result.evictions += s.evictions;
    result.checkpointEntries += s.checkpointEntries;
    result.checkpointBytes += s.checkpointBytes;
    result.checkpointRetirements += s.checkpointRetirements;
    result.checkpointEvictions += s.checkpointEvictions;
  }
  result.offloads += directOffloads_;
  result.offloadFailures += directFailures_;
  result.publications += directPublications_;
  result.hits = hits_;
  result.misses = misses_;
  result.diskHits = diskHits_;
  result.promotionsSkipped = promotionsSkipped_;
  return result;
}

} // namespace splash::engine

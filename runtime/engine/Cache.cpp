#include "engine/Cache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {

Cache::Cache(KvPool &pool, CacheNamespace cacheNamespace)
    : pool_(pool), kv_(pool, cacheNamespace, recency_), states_(kv_, recency_) {
}

void Cache::beginRequest(uint64_t requestId) {
  if (!requestId)
    throw std::invalid_argument("invalid request id");
  auto [_, inserted] = requests_.emplace(requestId, Request{});
  if (!inserted)
    throw std::invalid_argument("duplicate request id");
}

void Cache::endRequest(uint64_t requestId) {
  auto found = requests_.find(requestId);
  if (found == requests_.end())
    return;
  Request &active = found->second;
  for (uint32_t page : active.pages)
    pool_.releasePage(page, false);
  if (!active.cachedBlocks.empty())
    kv_.releaseActive(active.cachedBlocks.back());
  // Refresh used states within their class. Ordinary states remain newer
  // than the finished KV tail; checkpoints retain their lower priority.
  for (uint64_t block : active.cachedBlocks)
    states_.touch(block);
  requests_.erase(found);
}

std::vector<uint64_t>
Cache::matchedBlocks(std::span<const uint32_t> prompt,
                     std::span<const ImageSpan> images) const {
  std::vector<uint64_t> blocks;
  if (prompt.empty())
    return blocks;
  // Leave one real input token to regenerate request-specific anchor logits.
  const size_t maximumBlocks = (prompt.size() - 1) / KvCache::pageTokens;
  blocks.reserve(maximumBlocks);
  uint64_t parent = 0;
  for (size_t index = 0; index < maximumBlocks; ++index) {
    const size_t begin = index * KvCache::pageTokens;
    auto match =
        kv_.find(parent, prompt.subspan(begin, KvCache::pageTokens),
                 blockImageIdentity(begin, KvCache::pageTokens, images));
    if (!match)
      break;
    parent = match->id;
    blocks.push_back(parent);
  }
  return blocks;
}

uint32_t Cache::cachedTokens(std::span<const uint32_t> prompt,
                             std::span<const ImageSpan> images) const {
  const auto blocks = matchedBlocks(prompt, images);
  for (size_t i = blocks.size(); i > 0; --i)
    if (states_.contains(blocks[i - 1]))
      return static_cast<uint32_t>(i * KvCache::pageTokens);
  return 0;
}

CacheProbe Cache::probe(std::span<const uint32_t> prompt,
                        std::span<const ImageSpan> images) const {
  CacheProbe result;
  result.owner_ = this;
  result.blocks_ = matchedBlocks(prompt, images);
  result.kvGeneration_ = kv_.generation();
  result.promptSize_ = prompt.size();
  result.images_.assign(images.begin(), images.end());
  // Include the first missed page: it may hold different tokens at lookup.
  const size_t maximumBlocks =
      prompt.empty() ? 0 : (prompt.size() - 1) / KvCache::pageTokens;
  const size_t checkedBlocks =
      std::min(maximumBlocks, result.blocks_.size() + 1);
  if (checkedBlocks)
    result.checkedTokens_.assign(
        prompt.begin(), prompt.begin() + checkedBlocks * KvCache::pageTokens);
  for (size_t i = result.blocks_.size(); i > 0; --i) {
    if (states_.contains(result.blocks_[i - 1])) {
      result.cachedTokens_ = static_cast<uint32_t>(i * KvCache::pageTokens);
      break;
    }
  }
  return result;
}

CacheLookup Cache::lookup(std::span<const uint32_t> prompt,
                          std::span<const ImageSpan> images,
                          const CacheProbe *probe) {
  CacheLookup result;
  const auto validProbe = [&] {
    if (!probe || probe->owner_ != this ||
        probe->kvGeneration_ != kv_.generation() ||
        probe->promptSize_ != prompt.size() ||
        !std::equal(probe->checkedTokens_.begin(),
                    probe->checkedTokens_.end(), prompt.begin()) ||
        !std::equal(probe->images_.begin(), probe->images_.end(),
                    images.begin(), images.end()))
      return false;
    for (uint64_t block : probe->blocks_)
      if (!kv_.contains(block))
        return false;
    return true;
  };
  std::vector<uint64_t> fallback;
  std::span<const uint64_t> blocks;
  if (validProbe()) {
    blocks = probe->blocks_;
  } else {
    fallback = matchedBlocks(prompt, images);
    blocks = fallback;
  }
  if (!blocks.empty()) {
    kv_.touch(blocks.back());
    result.kvBoundary = static_cast<uint32_t>(blocks.size() * KvCache::pageTokens);
    result.state = states_.acquireDeepest(blocks);
  }
  return result;
}

void Cache::recordLookup(const CacheLookup &result) {
  ++lookup_.lookups;
  lookup_.kvHitTokens += result.kvBoundary;
  lookup_.stateHitTokens += result.resumeBoundary();
  states_.recordLookup(result.state.has_value());
  if (result.junctionBoundary())
    ++lookup_.lazyJunctions;
}

void Cache::restoreRequest(uint64_t requestId, const CacheLookup &lookup) {
  Request &active = request(requestId);
  if (!active.pages.empty() || !active.cachedBlocks.empty()) {
    throw std::logic_error("restore target already owns KV pages");
  }
  if (!lookup.state)
    return;
  KvCache::Chain chain = kv_.chain(lookup.state->kvBlock());
  std::vector<uint32_t> retained;
  retained.reserve(chain.pages.size());
  kv_.retainActive(chain.blocks.back());
  try {
    for (uint32_t page : chain.pages) {
      pool_.retainPage(page, false);
      retained.push_back(page);
    }
  } catch (...) {
    for (uint32_t page : retained)
      pool_.releasePage(page, false);
    kv_.releaseActive(chain.blocks.back());
    throw;
  }
  active.pages = std::move(chain.pages);
  active.cachedBlocks = std::move(chain.blocks);
  ++active.pageTableRevision;
}

TokenAdmission Cache::ensureTokens(uint64_t requestId, uint64_t tokenCount) {
  Request &active = request(requestId);
  const uint64_t needed64 =
      (tokenCount + KvCache::pageTokens - 1) / KvCache::pageTokens;
  if (needed64 > std::numeric_limits<uint32_t>::max()) {
    return {KvPageAcquireFailure::LogicalCapacity,
            std::numeric_limits<uint32_t>::max(), pool_.freePageCount()};
  }
  const uint32_t needed = static_cast<uint32_t>(needed64);
  if (needed <= active.pages.size())
    return {};
  const uint32_t additional = needed - active.pages.size();
  if (!makeLogicalPages(additional)) {
    return {KvPageAcquireFailure::LogicalCapacity, additional,
            pool_.freePageCount()};
  }
  KvPageAcquisition acquired = pool_.acquirePages(additional, false);
  if (!acquired.granted()) {
    return {acquired.failure, additional, pool_.freePageCount(),
            acquired.allocationFailure};
  }
  try {
    active.pages.insert(active.pages.end(), acquired.pages.begin(),
                        acquired.pages.end());
  } catch (...) {
    for (uint32_t page : acquired.pages)
      pool_.releasePage(page, false);
    throw;
  }
  ++active.pageTableRevision;
  return {};
}

PageTableView Cache::pageTable(uint64_t requestId) const {
  const Request &active = request(requestId);
  return {active.pages, active.pageTableRevision};
}

uint64_t Cache::publishCommittedBlocks(uint64_t requestId,
                                       std::span<const uint32_t> exactTokens,
                                       uint32_t committedTokens,
                                       std::span<const ImageSpan> images) {
  Request &active = request(requestId);
  if (committedTokens > exactTokens.size()) {
    throw std::invalid_argument("committed KV exceeds exact token history");
  }
  const uint32_t fullBlocks = committedTokens / KvCache::pageTokens;
  const uint64_t requiredPages =
      (uint64_t{committedTokens} + KvCache::pageTokens - 1) / KvCache::pageTokens;
  if (requiredPages > active.pages.size()) {
    throw std::logic_error("committed KV has no physical request page");
  }
  if (fullBlocks < active.cachedBlocks.size())
    throw std::logic_error("committed KV history moved backwards");
  while (active.cachedBlocks.size() < fullBlocks) {
    const uint32_t logical = static_cast<uint32_t>(active.cachedBlocks.size());
    const uint64_t parent = logical ? active.cachedBlocks.back() : 0;
    const uint32_t begin = logical * KvCache::pageTokens;
    auto inserted =
        kv_.insert(parent, exactTokens.subspan(begin, KvCache::pageTokens),
                   active.pages[logical],
                   blockImageIdentity(begin, KvCache::pageTokens, images));
    const uint32_t writerPage = active.pages[logical];
    const bool replacePage = inserted.physicalPage != writerPage;
    if (replacePage)
      pool_.retainPage(inserted.physicalPage, false);
    try {
      kv_.retainActive(inserted.id);
      try {
        active.cachedBlocks.push_back(inserted.id);
      } catch (...) {
        kv_.releaseActive(inserted.id);
        throw;
      }
    } catch (...) {
      if (replacePage)
        pool_.releasePage(inserted.physicalPage, false);
      throw;
    }
    if (replacePage) {
      active.pages[logical] = inserted.physicalPage;
      ++active.pageTableRevision;
      pool_.releasePage(writerPage, false);
    }
    if (parent)
      kv_.releaseActive(parent);
  }
  return active.cachedBlocks.empty() ? 0 : active.cachedBlocks.back();
}

uint64_t Cache::blockAt(uint64_t requestId, uint32_t boundary) const {
  if (!boundary || boundary % KvCache::pageTokens) {
    throw std::invalid_argument("state boundary is not a complete KV block");
  }
  const Request &active = request(requestId);
  const size_t index = boundary / KvCache::pageTokens - 1;
  if (index >= active.cachedBlocks.size()) {
    throw std::out_of_range("state boundary KV block is not published");
  }
  return active.cachedBlocks[index];
}

bool Cache::reuseCompositeState(uint64_t kvBlock, bool checkpoint) {
  return states_.touchIfResident(kvBlock, checkpoint);
}

void Cache::publishCompositeState(uint64_t kvBlock,
                                  std::shared_ptr<const CompositeState> state,
                                  bool checkpoint) {
  states_.publish(kvBlock, std::move(state), checkpoint);
}

StateCheckpoint Cache::checkpointState(uint64_t kvBlock) const noexcept {
  return states_.checkpoint(kvBlock);
}

bool Cache::retireCheckpointState(StateCheckpoint checkpoint) noexcept {
  return states_.retireCheckpoint(checkpoint);
}

uint64_t Cache::reclaimCache(uint64_t targetBytes, bool evictAll,
                             bool keepResumePoint) {
  // Empty backing that is waiting behind an in-flight release will satisfy
  // part of the target by itself; evicting more cache now would only
  // discard reusable prefixes without returning memory any sooner.
  if (releaseDeferred())
    return 0;
  uint64_t released = reclaimEmptyExtents();
  auto needsMore = [&] { return evictAll || released < targetBytes; };
  while (needsMore() && !releaseDeferred()) {
    const CacheReclaimResult result =
        reclaimOne(CacheReclaimMode::ReleaseBacking, keepResumePoint);
    if (!result.madeProgress)
      break;
    released += result.reclaimedBytes;
  }
  return released;
}

CacheReclaimResult Cache::reclaimOne(CacheReclaimMode mode,
                                     bool keepResumePoint) {
  if (mode == CacheReclaimMode::ReleaseBacking) {
    if (releaseDeferred())
      return {};
    if (const uint64_t bytes = reclaimEmptyExtents())
      return {true, bytes};
  }

  const std::optional<CacheEvictionCandidate> state =
      states_.evictionCandidate(keepResumePoint);
  const bool checkpoint = state && states_.checkpoint(state->id);
  const std::optional<CacheEvictionCandidate> kv =
      checkpoint ? std::nullopt : oldestStateFreeKvBlock();
  if (!state && !kv)
    return {};
  if (state && (!kv || state->lastUsed <= kv->lastUsed)) {
    const StateEviction eviction = states_.evict(state->id);
    if (!eviction.evicted)
      throw std::logic_error("state eviction candidate became pinned");
    return {true, eviction.reclaimedBytes};
  }
  kv_.erase(kv->id);
  return {true, mode == CacheReclaimMode::ReleaseBacking
                    ? reclaimEmptyExtents()
                    : 0};
}

bool Cache::reclaimOneState(bool checkpointsOnly) {
  const std::optional<CacheEvictionCandidate> state =
      states_.evictionCandidate();
  return state && (!checkpointsOnly || states_.checkpoint(state->id)) &&
         states_.evict(state->id).evicted;
}

bool Cache::releaseDeferred() const noexcept {
  return pool_.reclaimableExtentCount() > 0 && releasePending();
}

bool Cache::releasePending() const noexcept {
  return !pool_.releaseReady();
}

uint64_t Cache::releaseGeneration() const noexcept {
  return pool_.releaseGeneration();
}

void Cache::releaseUnusedKvBacking() {
  while (pool_.reclaimEmptyExtents(true))
    pool_.awaitRelease();
}

CacheSnapshot Cache::snapshot() const {
  return {pool_.snapshot(),
          kv_.snapshot(),
          states_.snapshot(),
          lookup_,
          static_cast<uint32_t>(requests_.size())};
}

Cache::Request &Cache::request(uint64_t requestId) {
  auto found = requests_.find(requestId);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

const Cache::Request &Cache::request(uint64_t requestId) const {
  auto found = requests_.find(requestId);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

bool Cache::makeLogicalPages(uint32_t count) {
  while (pool_.freePageCount() < count) {
    if (!evictOneKvBlock())
      return false;
  }
  return true;
}

bool Cache::evictOneKvBlock() {
  uint64_t previous = 0;
  while (auto candidate = kv_.evictionCandidate(previous)) {
    previous = candidate->id;
    if (states_.contains(candidate->id)) {
      if (!states_.evict(candidate->id).evicted)
        continue;
    }
    kv_.erase(candidate->id);
    return true;
  }
  return false;
}

std::optional<CacheEvictionCandidate> Cache::oldestStateFreeKvBlock() const {
  uint64_t previous = 0;
  while (auto candidate = kv_.evictionCandidate(previous)) {
    previous = candidate->id;
    if (!states_.contains(candidate->id))
      return candidate;
  }
  return std::nullopt;
}

uint64_t Cache::reclaimEmptyExtents() {
  const uint64_t before = pool_.residentBackingBytes();
  static_cast<void>(pool_.reclaimEmptyExtents(false));
  const uint64_t after = pool_.residentBackingBytes();
  return before >= after ? before - after : 0;
}

} // namespace splash::engine

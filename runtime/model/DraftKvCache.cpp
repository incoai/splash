#include "model/DraftKvCache.hpp"

#include <stdexcept>

namespace splash::model {
DraftKvCache::Page::~Page() {
  tracker->bytes.fetch_sub(bytes, std::memory_order_relaxed);
}
DraftKvCache::Block::~Block() {
  if (pool && pool->open && page)
    pool->free.push_back(std::move(page));
}
DraftKvCache::DraftKvCache(metal::MetalBackend &backend,
                           metal::AllocationAdmission admission,
                           std::shared_ptr<StateAllocationTracker> tracker,
                           DraftStateLayout layout)
    : backend_(backend), admission_(std::move(admission)),
      tracker_(std::move(tracker)), layout_(layout) {
  if (!admission_ || !tracker_)
    throw std::invalid_argument(
        "draft KV cache needs admission and accounting");
  if (!layout.valid() || layout.tokens % layout.blockTokens)
    throw std::invalid_argument("invalid draft KV cache layout");
}
DraftKvCache::~DraftKvCache() {
  pool_->open = false;
  releaseIdle();
}
void DraftKvCache::releaseIdle(uint32_t keep) noexcept {
  while (pool_->free.size() > keep) {
    pool_->free.pop_back();
    --pool_->allocatedPages;
  }
}
std::shared_ptr<DraftKvCache::Block>
DraftKvCache::acquire(metal::AllocationFailure *failure) {
  auto block = std::make_shared<Block>();
  block->pool = pool_;
  if (!pool_->free.empty()) {
    block->page = std::move(pool_->free.back());
    pool_->free.pop_back();
  } else {
    // Returning a page from a destructor must never allocate.
    pool_->free.reserve(pool_->allocatedPages + 1);
    auto page = std::make_shared<Page>();
    page->tracker = tracker_;
    const auto admission = admission_(layout_.blockBytes(), [&] {
      const uint64_t before = backend_.memoryStats().allocatedBytes;
      page->buffer = backend_.allocateBuffer(
          layout_.blockBytes(), metal::BufferStorage::Shared, "draft-kv-page");
      page->bytes =
          metal::allocationDelta(before, backend_.memoryStats().allocatedBytes);
      tracker_->bytes.fetch_add(page->bytes, std::memory_order_relaxed);
    });
    if (!admission) {
      if (failure)
        *failure = admission.failure;
      return {};
    }
    ++pool_->allocatedPages;
    block->page = std::move(page);
  }
  return block;
}
std::vector<DraftKvPageViews> DraftKvCache::views(const Window &pages) const {
  std::vector<DraftKvPageViews> result(layout_.layers);
  const uint64_t tensorBytes = uint64_t{layout_.kvHeads} * layout_.blockTokens *
                               layout_.headDimension * 2;
  for (uint32_t layer = 0; layer < layout_.layers; ++layer) {
    auto &view = result[layer];
    for (const auto &block : pages) {
      view.keyPages.push_back(backend_.view(
          block->page->buffer, layer * 2 * tensorBytes, tensorBytes));
      view.valuePages.push_back(backend_.view(
          block->page->buffer, (layer * 2 + 1) * tensorBytes, tensorBytes));
    }
  }
  return result;
}
} // namespace splash::model

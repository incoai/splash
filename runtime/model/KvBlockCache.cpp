#include "model/KvBlockCache.hpp"

#include "AwakeClock.hpp"
#include "metal/abi/PagedAttention.h"
#include "model/KvCopyRead.hpp"
#include "model/SlotRecycler.hpp"
#include "ops/PageStorage.hpp"

#include <algorithm>
#include <array>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

namespace splash::model {
namespace {

constexpr uint32_t kPlaceBits = SPLASH_ATTENTION_LOCATION_BLOCK_BITS;
constexpr uint32_t kPlaceMask = (1u << kPlaceBits) - 1;
constexpr uint32_t kLanes = ExecutionLimits::maximumBatchWidth;

uint64_t nanosecondsSince(AwakeClock::time_point start) noexcept {
  return static_cast<uint64_t>((AwakeClock::now() - start).count());
}

uint64_t nanoseconds(double seconds) noexcept { return static_cast<uint64_t>(seconds * 1e9); }

const KvBlockCacheOptions &checked(const kv::Layout &layout, uint32_t extentPages,
                                   const KvBlockCacheOptions &options) {
  if (!layout.spillable()) throw std::invalid_argument("the hot block cache needs a layout that can spill");
  if (options.format.layout() != layout)
    throw std::invalid_argument("the hot block cache's slot format is another layout's");
  if (!options.readers || !options.laneSlots || !extentPages)
    throw std::invalid_argument("the hot block cache needs readers, lane slots and extents");
  return options;
}

// The cache's pages: the pool's K/V, at the pool's places in an extent.
kv::Layout withoutIndex(kv::Layout layout) {
  layout.indexDimension = 0;
  return layout;
}

uint32_t extentsFor(uint64_t slots, uint32_t extentSlots) noexcept {
  return static_cast<uint32_t>((slots + extentSlots - 1) / extentSlots);
}

} // namespace

struct KvBlockCache::Impl {
  // A list entry whose block no slot held when the resolve looked: its key,
  // source and place in its page, the lane whose list holds it, and its
  // location.
  struct Miss final {
    uint64_t key = 0;
    const KvPageSource *source = nullptr;
    uint32_t place = 0;
    uint32_t lane = 0;
    uint32_t *location = nullptr;
  };
  // A block a resolve reads into the slot it claimed, for the lanes (a bit
  // each) that list it.
  struct Read final {
    const KvPageSource *source = nullptr;
    uint32_t place = 0;
    uint32_t slot = 0;
    uint32_t lanes = 0;
  };
  // One read of a page's blocks a resolve misses in its layer, the reads
  // [first, end) of its blocks, in place order: from the first block's
  // record to the last's, the records between too.
  struct Span final {
    uint32_t first = 0;
    uint32_t end = 0;
  };
  // What a resolve's host step reads and writes: the lists from `list` on,
  // their counts and locations (Shared), and each lane's sources, which it
  // holds.
  struct Resolve final {
    metal::MetalBuffer lists, counts, locations;
    uint32_t stride = 0;
    uint32_t list = 0;
    std::vector<std::shared_ptr<const KvPageSources>> sources;
  };

  Impl(metal::MetalBackend &backend, metal::AllocationAdmission admission, const kv::Layout &layout,
       uint32_t extentPages, const KvBlockCacheOptions &options);

  [[nodiscard]] uint32_t floorLanes() const noexcept {
    return static_cast<uint32_t>(std::ranges::count(floors, true));
  }
  // required(n), in whole extents.
  [[nodiscard]] uint32_t floorExtents(uint32_t lanes) const noexcept {
    return lanes ? extentsFor(uint64_t{lanes + 1} * laneSlots, extentSlots) : 0;
  }
  // Allocates extents up to `held` extents, or none of them.
  [[nodiscard]] ModelMemoryAdmission grow(uint32_t held);
  // The directory and the location pages of `held` extents.
  void place(uint32_t held);
  // Extents and the directory change between commands only: no slot moves
  // while a command's GPU work reads it.
  void requireNoCommand() const {
    if (backend.commandInFlight()) throw std::logic_error("the hot block cache resizes while a command is in flight");
  }
  void resolve(uint32_t layer, const Resolve &resolve, KvReadStats &stats);
  // The bytes a span's read brings.
  [[nodiscard]] kv::SlotRange spanRange(uint32_t layer, const Span &span) const;
  // A reader task: throws when the span cannot be read.
  void readSpan(uint32_t layer, const Span &span) const;

  metal::MetalBackend &backend;
  const kv::SlotFormat format;
  const std::shared_ptr<SsdReaders> readers;
  const uint32_t laneSlots;
  const uint32_t extentSlots;
  const uint32_t targetExtents;
  kv::PageStorage pages;
  // The page table of the cache pages, which locations index (Shared).
  const metal::MetalBuffer locationPages;
  // A block's record, and one of zeros, which a block that cannot be read
  // leaves.
  const uint64_t recordBytes;
  const std::vector<std::byte> zeros;

  // The engine thread's: the state lanes that hold a floor, the extents
  // allocated, and the counts of every completed command.
  std::array<bool, kLanes> floors{};
  uint32_t extents = 0;
  KvReadStats totals;

  // The host steps of the command in flight and the engine thread between
  // commands hand the directory over under `mutex`: each layer's slots, the
  // resolves so far, a resolve's misses and reads (each slot's read), and
  // the blocks no read could bring.
  std::mutex mutex;
  std::vector<SlotRecycler> directory;
  uint64_t resolves = 0;
  std::vector<Miss> misses;
  std::vector<Read> reads;
  std::vector<Span> spans;
  std::vector<uint32_t> readOf;
  std::vector<KvReadFailure> failures;

  LatencyHistogram resolveTimes, readTimes, pickup, work, resume;
};

KvBlockCache::Impl::Impl(metal::MetalBackend &backend, metal::AllocationAdmission admission,
                         const kv::Layout &layout, uint32_t extentPages, const KvBlockCacheOptions &options)
    : backend(backend), format(checked(layout, extentPages, options).format), readers(options.readers),
      laneSlots(options.laneSlots), extentSlots(extentPages << kPlaceBits),
      targetExtents(static_cast<uint32_t>(options.targetBytes /
                                          (uint64_t{extentPages} * layout.kvBytesPerModelPage()))),
      pages(backend, std::move(admission), withoutIndex(layout),
            std::max(targetExtents, floorExtents(kLanes)) * extentPages, extentPages),
      locationPages(backend.allocateBuffer(uint64_t{pages.pageCount()} * sizeof(SplashKvPage),
                                           metal::BufferStorage::Shared, "kv-block-cache/location-pages")),
      recordBytes(format.recordBytes()), zeros(recordBytes),
      directory(layout.attentionLayers, SlotRecycler(0)) {}

ModelMemoryAdmission KvBlockCache::Impl::grow(uint32_t held) {
  requireNoCommand();
  for (uint32_t extent = extents; extent < held; ++extent)
    if (const metal::AllocationResult result = pages.allocateExtent(extent); !result) {
      for (uint32_t allocated = extents; allocated < extent; ++allocated) pages.releaseExtent(allocated);
      return {.ok = false, .allocationFailure = result.failure};
    }
  place(held);
  return {.ok = true};
}

void KvBlockCache::Impl::place(uint32_t held) {
  const std::lock_guard lock(mutex);
  const uint32_t slots = held * extentSlots;
  for (SlotRecycler &layer : directory) layer.resize(slots);
  readOf.resize(slots);
  std::vector<uint32_t> cachePages(uint64_t{held} * pages.extentPages());
  std::iota(cachePages.begin(), cachePages.end(), 0u);
  pages.writeEntries(cachePages, 0, locationPages);
  extents = held;
}

kv::SlotRange KvBlockCache::Impl::spanRange(uint32_t layer, const Span &span) const {
  const kv::SlotRange first = format.recordRange(layer, reads[span.first].place);
  const kv::SlotRange last = format.recordRange(layer, reads[span.end - 1].place);
  return {first.offset, last.offset + last.bytes - first.offset};
}

void KvBlockCache::Impl::readSpan(uint32_t layer, const Span &span) const {
  const kv::SlotRange range = spanRange(layer, span);
  const std::optional<std::span<const std::byte>> bytes = readKvCopy(*reads[span.first].source->copy, range);
  // Blocks that cannot be read leave zeros, which their lanes' attention
  // reads before they fail.
  for (uint32_t index = span.first; index < span.end; ++index) {
    const Read &read = reads[index];
    const std::span<const std::byte> record =
        bytes ? bytes->subspan(format.recordRange(layer, read.place).offset - range.offset, recordBytes)
              : std::span<const std::byte>(zeros);
    format.scatterRecord(record, pages.layerSpans(read.slot >> kPlaceBits, layer), read.slot & kPlaceMask);
  }
  if (!bytes) throw std::runtime_error("a spilled K/V block could not be read");
}

// Looks every Index entry up first, so that a block the resolve holds stays
// whatever it misses; then claims a slot for each block it misses, once for
// every lane that lists it, and reads them all, a page's in one read: reads
// of up to 16 KiB cost about the same, and the records of a layer's blocks
// lie in place order, so one read across a page's misses, the records
// between them too, costs less than a read for each.
void KvBlockCache::Impl::resolve(uint32_t layer, const Resolve &resolve, KvReadStats &stats) {
  const AwakeClock::time_point start = AwakeClock::now();
  const std::lock_guard lock(mutex);
  SlotRecycler &slots = directory[layer];
  const uint64_t load = ++resolves;
  const auto *lists = static_cast<const uint32_t *>(resolve.lists.contents());
  const auto *counts = static_cast<const uint32_t *>(resolve.counts.contents());
  auto *locations = static_cast<uint32_t *>(resolve.locations.contents());
  ++stats.resolves;
  misses.clear();
  uint32_t pinned = 0;
  for (uint32_t lane = 0; lane < resolve.sources.size(); ++lane) {
    const uint64_t list = uint64_t{resolve.list} + lane;
    const uint32_t entries = std::min(counts[list], resolve.stride);
    const uint32_t *entry = lists + list * resolve.stride;
    uint32_t *location = locations + list * resolve.stride;
    const KvPageSources *sources = resolve.sources[lane].get();
    stats.entries += entries;
    for (uint32_t index = 0; index < entries; ++index) {
      const uint32_t block = entry[index] >> SPLASH_ATTENTION_BLOCK_ROW_BITS;
      const KvPageSource *source = sources ? &(*sources)[block >> kPlaceBits] : nullptr;
      if (!source || !source->indexed()) {
        location[index] = SPLASH_ATTENTION_PAGED;
        ++stats.paged;
        continue;
      }
      if (!source->copy) throw std::logic_error("a page that holds only its index has no tier copy");
      const uint64_t key = source->copy->id() << kPlaceBits | (block & kPlaceMask);
      if (const std::optional<uint32_t> slot = slots.find(key)) {
        pinned += !slots.pinned(*slot, load);
        slots.touch(*slot, load, false);
        location[index] = *slot;
        ++stats.hits;
      } else {
        misses.push_back({key, source, block & kPlaceMask, lane, &location[index]});
      }
    }
  }
  // Each lane holding a floor selects at most laneSlots blocks, and the
  // floors leave laneSlots more: a resolve past them is a bug.
  if (misses.size() > slots.size() - pinned)
    throw std::logic_error("a resolve misses more blocks than the hot block cache has slots it does not use");
  reads.clear();
  const uint64_t ghostHits = slots.ghostHits();
  for (const Miss &miss : misses) {
    // A block an earlier lane of the resolve missed is read once.
    if (const std::optional<uint32_t> slot = slots.find(miss.key)) {
      reads[readOf[*slot]].lanes |= 1u << miss.lane;
      *miss.location = *slot;
      ++stats.hits;
      continue;
    }
    const uint32_t slot = slots.claim(miss.key, load, false);
    readOf[slot] = static_cast<uint32_t>(reads.size());
    reads.push_back({miss.source, miss.place, slot, 1u << miss.lane});
    *miss.location = slot;
  }
  stats.misses += reads.size();
  stats.ghostHits += slots.ghostHits() - ghostHits;
  if (!reads.empty()) {
    // A page's blocks, in place order, are one span.
    std::ranges::sort(reads, {}, [](const Read &read) { return std::pair(read.source->copy->id(), read.place); });
    spans.clear();
    for (uint32_t index = 0; index < reads.size(); ++index)
      if (!index || reads[index].source->copy->id() != reads[index - 1].source->copy->id())
        spans.push_back({index, index + 1});
      else
        spans.back().end = index + 1;
    // The reads run now, ahead of the background, as a batch that keeps its
    // failures (SsdReaders::runBatch): blocks the readers read twice in vain
    // fail the lanes that list them, not the command.
    const AwakeClock::time_point reading = AwakeClock::now();
    const std::shared_ptr<SsdReaders::Batch> batch =
        readers->runBatch(spans.size(), [this, layer](size_t index) { readSpan(layer, spans[index]); });
    readTimes.add(nanosecondsSince(reading));
    for (size_t index = 0; index < spans.size(); ++index) {
      const Span &span = spans[index];
      if (!batch->failed(index)) {
        stats.readBytes += spanRange(layer, span).bytes;
        continue;
      }
      stats.readFailures += span.end - span.first;
      for (uint32_t at = span.first; at < span.end; ++at) {
        const Read &read = reads[at];
        slots.free(read.slot);
        for (uint32_t lane = 0; lane < kLanes; ++lane)
          if (read.lanes >> lane & 1) failures.push_back({lane, read.source->block});
      }
    }
  }
  resolveTimes.add(nanosecondsSince(start));
}

KvBlockCache::KvBlockCache(metal::MetalBackend &backend, metal::AllocationAdmission admission,
                           const kv::Layout &layout, uint32_t extentPages, KvBlockCacheOptions options)
    : impl_(std::make_shared<Impl>(backend, std::move(admission), layout, extentPages, options)) {}

KvBlockCache::~KvBlockCache() = default;

ModelMemoryAdmission KvBlockCache::reserveLane(uint32_t stateLane) {
  Impl &impl = *impl_;
  if (stateLane >= kLanes) throw std::out_of_range("the hot block cache has no such lane");
  if (impl.floors[stateLane]) return {.ok = true};
  if (const uint32_t held = impl.floorExtents(impl.floorLanes() + 1); held > impl.extents)
    if (const ModelMemoryAdmission admission = impl.grow(held); !admission.granted()) return admission;
  impl.floors[stateLane] = true;
  return {.ok = true};
}

void KvBlockCache::releaseLane(uint32_t stateLane) noexcept {
  if (stateLane < kLanes) impl_->floors[stateLane] = false;
}

ModelMemoryAdmission KvBlockCache::growTowardTarget() {
  Impl &impl = *impl_;
  if (!impl.floorLanes() || impl.extents >= impl.targetExtents) return {.ok = true};
  return impl.grow(impl.extents + 1);
}

uint64_t KvBlockCache::releaseAboveFloor() {
  Impl &impl = *impl_;
  const uint32_t held = impl.extents;
  if (held <= impl.floorExtents(impl.floorLanes())) return 0;
  impl.requireNoCommand();
  // The slots of the extent that goes leave the directory first.
  impl.place(held - 1);
  impl.pages.releaseExtent(held - 1);
  return impl.pages.extentBytes();
}

ops::AttentionBlocks KvBlockCache::addResolve(metal::CommandGraph &graph, uint32_t layer,
                                              std::span<const std::shared_ptr<const KvPageSources>> lanes,
                                              const ops::AttentionBlocks &blocks,
                                              const metal::MetalBuffer &locations,
                                              std::shared_ptr<KvReadStats> stats) {
  const Impl &impl = *impl_;
  const uint64_t entries = (uint64_t{blocks.list} + lanes.size()) * blocks.stride;
  if (layer >= impl.format.layout().attentionLayers || lanes.empty() || lanes.size() > kLanes || !blocks.stride ||
      !stats)
    throw std::invalid_argument("invalid hot block cache resolve");
  if (!blocks.lists.contents() || !blocks.counts.contents() || !locations.contents() ||
      blocks.lists.sizeBytes() < entries * sizeof(uint32_t) || locations.sizeBytes() < entries * sizeof(uint32_t) ||
      blocks.counts.sizeBytes() < (uint64_t{blocks.list} + lanes.size()) * sizeof(uint32_t))
    throw std::invalid_argument("a resolve's lists, counts and locations are not CPU-visible or too small");
  Impl::Resolve resolve{blocks.lists, blocks.counts, locations, blocks.stride, blocks.list,
                        {lanes.begin(), lanes.end()}};
  // A decode step's resolves come every layer and most end within a
  // millisecond, about as long as an idle GPU takes to come back to speed.
  graph.host([impl = impl_, layer, resolve = std::move(resolve),
              stats = std::move(stats)] { impl->resolve(layer, resolve, *stats); },
             [impl = impl_](const metal::HostStepTiming &timing) {
               impl->pickup.add(nanoseconds(timing.pickupSeconds));
               impl->work.add(nanoseconds(timing.workSeconds));
               impl->resume.add(nanoseconds(timing.resumeSeconds));
             },
             metal::HostStepGpu::Awake);
  return {blocks.lists, blocks.counts, blocks.stride, blocks.list, locations, impl.locationPages};
}

std::vector<KvReadFailure> KvBlockCache::takeFailures() {
  std::vector<KvReadFailure> failures;
  {
    const std::lock_guard lock(impl_->mutex);
    failures = std::exchange(impl_->failures, {});
  }
  // A page's blocks fail together, in every layer: a lane names it once.
  std::ranges::sort(failures, {},
                    [](const KvReadFailure &failure) { return std::pair(failure.lane, failure.block); });
  failures.erase(std::unique(failures.begin(), failures.end()), failures.end());
  return failures;
}

void KvBlockCache::mergeStats(const KvReadStats &stats) {
  // Under the directory's lock, which the host steps that counted held.
  const std::lock_guard lock(impl_->mutex);
  impl_->totals += stats;
}

KvBlockCacheStats KvBlockCache::stats() const {
  const Impl &impl = *impl_;
  const uint64_t extentBytes = impl.pages.extentBytes();
  return {.bytes = impl.extents * extentBytes,
          .targetBytes = impl.targetExtents * extentBytes,
          .floorBytes = impl.floorExtents(impl.floorLanes()) * extentBytes,
          .slotsPerLayer = impl.extents * impl.extentSlots,
          .lanes = impl.floorLanes(),
          .reads = impl.totals,
          .resolve = impl.resolveTimes.snapshot(),
          .read = impl.readTimes.snapshot(),
          .pickup = impl.pickup.snapshot(),
          .work = impl.work.snapshot(),
          .resume = impl.resume.snapshot()};
}

} // namespace splash::model

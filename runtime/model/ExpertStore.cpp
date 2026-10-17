#include "model/ExpertStore.hpp"

#include "metal/abi/MoE.h"
#include "model/GgufImageLayout.hpp"
#include "model/GgufPreparation.hpp"
#include "model/Model.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"

#include <pthread/qos.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

constexpr uint32_t kNoSlot = UINT32_MAX;
// The most bytes one read of a read-ahead reads (Impl::Piece), and the
// readers that read ahead: what the load that takes a read-ahead over waits
// for is the pieces its readers took. An M5 Max's SSD reads 460 KiB pieces
// at 7.25 GB/s from four threads and at 7.41 GB/s from eighteen, but each
// then takes 276 us instead of 1223 us.
constexpr uint64_t kAheadPieceBytes = 512 << 10;
constexpr unsigned kAheadReaders = 4;
// Where each projection's staged rows start in a range of a pool: a multiple
// of the 2 bytes the repack's 16-bit loads of BF16 and F16 rows need
// (moe_expert_load), and of Apple silicon's 128-byte cache line
// (hw.cachelinesize), so that the threads that read two projections' rows in
// never write one line.
constexpr uint64_t kStagingAlignment = 256;
// The blocks of experts the slots a prefill load borrows place: one it
// stages, and one the read-ahead of the next block's reads meanwhile.
constexpr uint64_t kBorrowedBlocks = 2;
// A decode cycle's most rows: every lane's verify rows.
constexpr uint32_t kCycleRows = ExecutionLimits::maximumBatchWidth * ExecutionLimits::targetVerifyRows;
// A prefill chunk is long when its rows route at least this many times as
// many experts as a block has: routed evenly, that many routes leave about
// e^-3, 5%, of the experts unrouted, so it routes nearly every expert.
constexpr uint64_t kLongChunkRoutes = 3;

// ExpertStoreStats::uses and Impl::stepTimes hold a use per ops::MoePhase,
// at its value.
static_assert(static_cast<size_t>(ops::MoePhase::Prefill) == 0 && static_cast<size_t>(ops::MoePhase::Decode) == 1);

constexpr uint64_t alignUp(uint64_t value, uint64_t alignment) noexcept {
  return (value + alignment - 1) / alignment * alignment;
}

// One projection of an expert of some size: rows x columns in format, its
// native rows rowBytes each, and the bytes of its planes.
struct ProjectionLayout final {
  uint32_t format = 0;
  uint64_t rows = 0;
  uint64_t columns = 0;
  uint64_t rowBytes = 0;
  std::array<uint64_t, 3> planes{}; // plane0, plane1, meta
  bool operator==(const ProjectionLayout &) const = default;
};
using ExpertLayout = std::array<ProjectionLayout, 3>; // gate, up, down

ExpertLayout expertLayout(const gguf::StreamedExperts &experts, ops::MoeShape shape) {
  const uint64_t intermediate = shape.expertIntermediateSize, hidden = shape.hiddenSize;
  const std::array<uint64_t, 3> rows{intermediate, intermediate, hidden}, columns{hidden, hidden, intermediate};
  ExpertLayout layout;
  for (size_t z = 0; z < layout.size(); ++z) {
    const gguf::Repack &repack = experts.projections[z];
    if (repack.format >= GGUF_FMT_COUNT || repack.rows != shape.experts * rows[z] || repack.columns != columns[z])
      throw std::invalid_argument("streamed experts do not match the MoE shape");
    const QuantFormat &format = kQuantFormats[repack.format];
    const GgufPlaneBytes planes = ggufPlaneBytes(format, rows[z], columns[z]);
    layout[z] = {repack.format, rows[z], columns[z], ggufRowBytes(format, columns[z]),
                 {planes.plane0, planes.plane1, planes.meta}};
  }
  return layout;
}

// The bytes of an expert's native rows of each projection.
std::array<uint64_t, 3> nativeBytes(const ExpertLayout &layout) noexcept {
  return {layout[0].rows * layout[0].rowBytes, layout[1].rows * layout[1].rowBytes,
          layout[2].rows * layout[2].rowBytes};
}

// Places in `ranges` (ascending byte ranges of one buffer) for `count`
// experts of width[z] bytes of each projection: projection after
// projection, each from kStagingAlignment on, expert after expert, an
// expert's projection whole in one range, so that consecutive experts'
// follow one another within a range (one read); none when they do not all
// fit.
std::optional<std::vector<std::array<uint64_t, 3>>> place(std::span<const std::pair<uint64_t, uint64_t>> ranges,
                                                          const std::array<uint64_t, 3> &width, uint64_t count) {
  std::vector<std::array<uint64_t, 3>> places(count);
  if (!count) return places;
  if (ranges.empty()) return std::nullopt;
  size_t range = 0;
  uint64_t cursor = ranges.front().first;
  for (size_t z = 0; z < width.size(); ++z) {
    cursor = alignUp(cursor, kStagingAlignment);
    for (uint64_t expert = 0; expert < count; ++expert) {
      while (cursor + width[z] > ranges[range].second) {
        if (++range == ranges.size()) return std::nullopt;
        cursor = alignUp(ranges[range].first, kStagingAlignment);
      }
      places[expert][z] = cursor;
      cursor += width[z];
    }
  }
  return places;
}

// The arena bytes of `slots` experts of a layout: each projection's planes
// in a region of their own.
uint64_t arenaBytes(const ExpertLayout &layout, uint64_t slots) noexcept {
  uint64_t bytes = 0;
  for (const ProjectionLayout &projection : layout)
    for (const uint64_t plane : projection.planes) bytes += alignWeightOffset(slots * plane);
  return bytes;
}

// What a store's blocks alone decide (the store, ExpertStore::fixedBytes and
// borrowedSlots): each block's size, the sizes' layouts and blocks; the
// streamed size of the most blocks (lender), whose arena lends prefill loads
// borrowedSlots slots; a place's bytes of each projection (width), the most
// of a streamed size's expert; and the store's staging, stagingPlaces places
// in stagingBytes.
struct Geometry final {
  std::vector<uint32_t> sizeOf;
  std::vector<ExpertLayout> layouts;
  std::vector<uint32_t> blocks;
  uint32_t lender = kNoSlot;
  std::array<uint64_t, 3> width{};
  uint64_t stagingPlaces = 0, stagingBytes = 0;
  uint32_t borrowedSlots = 0;

  // A size of several blocks streams; one of one block holds every expert.
  [[nodiscard]] bool streams(uint32_t size) const noexcept { return blocks[size] > 1; }
};

Geometry geometryOf(const StreamedBlocks &streamed, ops::MoeShape shape) {
  if (!shape.valid() || streamed.blocks.empty()) throw std::invalid_argument("invalid expert store");
  Geometry g;
  for (const gguf::StreamedExperts &experts : streamed.blocks) {
    const ExpertLayout layout = expertLayout(experts, shape);
    const auto found = std::ranges::find(g.layouts, layout);
    const uint32_t size = static_cast<uint32_t>(found - g.layouts.begin());
    if (found == g.layouts.end()) {
      g.layouts.push_back(layout);
      g.blocks.push_back(0);
    }
    ++g.blocks[size];
    g.sizeOf.push_back(size);
  }
  for (uint32_t size = 0; size < g.layouts.size(); ++size)
    if (g.streams(size) && (g.lender == kNoSlot || g.blocks[size] > g.blocks[g.lender])) g.lender = size;
  // A place holds any streamed size's expert; without one, any size's (the
  // staging then only reads the sizes of one block, Impl::fill).
  for (uint32_t size = 0; size < g.layouts.size(); ++size)
    if (g.streams(size) || g.lender == kNoSlot) {
      const std::array<uint64_t, 3> bytes = nativeBytes(g.layouts[size]);
      for (size_t z = 0; z < 3; ++z) g.width[z] = std::max(g.width[z], bytes[z]);
    }
  // The staging places every expert a decode cycle's rows may route, so a
  // decode load is one step.
  g.stagingPlaces = std::min<uint64_t>(shape.experts, uint64_t{kCycleRows} * shape.expertsPerToken);
  const std::pair<uint64_t, uint64_t> unbounded{0, UINT64_MAX};
  const std::array<uint64_t, 3> last = place({&unbounded, 1}, g.width, g.stagingPlaces)->back();
  g.stagingBytes = alignUp(last[2] + g.width[2], kStagingAlignment);
  // When the staging places fewer than a block, which a prefill load may
  // route: the fewest of the lender's slots whose bytes in its plane regions
  // place kBorrowedBlocks blocks, each region's less what aligning its start
  // and the first places of two projections in it may cost (the store places
  // them where they are).
  if (g.lender != kNoSlot && g.stagingPlaces < shape.experts) {
    const ExpertLayout &lender = g.layouts[g.lender];
    const uint64_t places = kBorrowedBlocks * shape.experts, lost = 3 * kStagingAlignment;
    uint64_t slotBytes = 0;
    for (const ProjectionLayout &projection : lender)
      for (const uint64_t plane : projection.planes) slotBytes += plane;
    const auto fits = [&](uint64_t slots) {
      std::vector<std::pair<uint64_t, uint64_t>> ranges;
      uint64_t at = 0;
      for (const ProjectionLayout &projection : lender)
        for (const uint64_t plane : projection.planes)
          if (slots * plane > lost) {
            ranges.emplace_back(at, at + slots * plane - lost);
            at = alignUp(ranges.back().second, kStagingAlignment);
          }
      return place(ranges, g.width, places).has_value();
    };
    uint64_t slots = (places * (g.width[0] + g.width[1] + g.width[2]) + slotBytes - 1) / slotBytes;
    while (!fits(slots)) ++slots;
    g.borrowedSlots = static_cast<uint32_t>(slots);
  }
  return g;
}

uint64_t fixedBytesOf(const Geometry &g, uint32_t blocks, ops::MoeShape shape) noexcept {
  const uint64_t prediction =
      uint64_t{kCycleRows} * (shape.routerWidth() + 2ULL * shape.routesPerToken()) * sizeof(float);
  return g.stagingBytes + uint64_t{blocks} * shape.experts * sizeof(uint32_t) +
         uint64_t{shape.experts} * (sizeof(uint32_t) + 3 * sizeof(uint64_t)) + 4 * sizeof(uint32_t) + prediction;
}

} // namespace

struct ExpertStore::Impl {
  // Places a load stages its experts' native rows in for their repack
  // (stage), and a read-ahead the next block's (readAhead): place p's of
  // projection z from places[p][z] of `buffer` on, Geometry::width bytes,
  // and whether the load being staged took it.
  struct Pool final {
    const metal::MetalBuffer *buffer = nullptr;
    std::vector<std::array<uint64_t, 3>> places;
    std::vector<uint8_t> taken;
  };
  enum PoolKind : size_t { Staging, Borrowed };
  // The experts of one size: their slots in an arena; a streamed size's
  // recycled least recently used first, a long prefill chunk's loads before
  // the others.
  struct Size final {
    ExpertLayout layout;
    uint32_t blocks = 0;
    uint32_t slots = 0;
    // Projection z's plane i from regions[z][i] of the arena on.
    std::array<std::array<uint64_t, 3>, 3> regions{};
    metal::MetalBuffer arena;
    // The repack of its staged rows into slots.
    MoeExpertLoadParams load{};
    uint32_t loadGroups = 0;
    // Two recency orders, each from the least recently used slot (head) to
    // the most: Chunk, the experts a long prefill chunk loaded that no other
    // step routed since, which claim() recycles first, and the free slots;
    // Recent, the others. A slot is in one, order[slot].
    enum Order : uint8_t { Chunk, Recent };
    std::vector<uint32_t> previous, next;
    std::vector<uint8_t> order;
    std::array<uint32_t, 2> head{kNoSlot, kNoSlot}, tail{kNoSlot, kNoSlot};
    std::vector<uint32_t> ownerBlock, ownerExpert; // kNoSlot when free
    std::vector<uint64_t> lastLoad;                // the load that last routed it

    void unlink(uint32_t slot) {
      const uint8_t o = order[slot];
      (previous[slot] == kNoSlot ? head[o] : next[previous[slot]]) = next[slot];
      (next[slot] == kNoSlot ? tail[o] : previous[next[slot]]) = previous[slot];
    }
    void append(uint32_t slot, Order o) {
      order[slot] = o;
      previous[slot] = tail[o];
      next[slot] = kNoSlot;
      (tail[o] == kNoSlot ? head[o] : next[tail[o]]) = slot;
      tail[o] = slot;
    }
    void prepend(uint32_t slot, Order o) {
      order[slot] = o;
      next[slot] = head[o];
      previous[slot] = kNoSlot;
      (head[o] == kNoSlot ? tail[o] : previous[head[o]]) = slot;
      head[o] = slot;
    }
    void touch(uint32_t slot, uint64_t load, Order o) {
      lastLoad[slot] = load;
      unlink(slot);
      append(slot, o);
    }
    // Every slot free, in Chunk.
    void reset() {
      head = tail = {kNoSlot, kNoSlot};
      for (uint32_t slot = 0; slot < slots; ++slot) append(slot, Chunk);
    }
  };
  struct Block final {
    uint32_t size = 0;
    gguf::StreamedExperts experts;
    std::vector<uint32_t> slot; // each expert's, kNoSlot when not resident
    // Its router and shared-expert gate (bind), which predict its routes.
    ops::QuantizedSegment router, scalarGate;
  };
  // One read of a load: `count` consecutive experts from `expert` on of
  // projection z, from job `job` on, whose places follow one another.
  struct Read final {
    uint32_t z = 0, job = 0, expert = 0, count = 0;
  };
  // One read of a read-ahead: `count` rows from row `first` on of job
  // `job`'s expert's projection z.
  struct Piece final {
    uint32_t job = 0, z = 0;
    uint64_t first = 0, count = 0;
  };
  // A step's tiles (ops::MoeLoadPart): the grouping's descriptors and count.
  struct PartTiles final {
    metal::MetalBuffer descriptors, count;
  };

  Impl(metal::MetalBackend &backend, ops::MoeShape shape, Geometry geometry)
      : backend(backend), shape(shape), geometry(std::move(geometry)) {}
  // The readers of a read-ahead read into this store's memory: they end the
  // pieces they took before it goes, and then their thread.
  ~Impl() {
    takeAhead();
    waitAhead();
    {
      const std::lock_guard lock(aheadMutex);
      aheadEnding = true;
    }
    aheadChanged.notify_all();
    aheadThread.join();
  }

  // The host step of block `block`'s load (addLoad) of `rows` rows in the
  // tiles of `parts`; `predicted` when the step predicted the next block's
  // routes (predictionRoutes and predictionWeights).
  void load(uint32_t block, ops::MoePhase phase, uint32_t rows, bool predicted, std::span<const PartTiles> parts);
  void route(uint32_t block, std::span<const std::span<const MoeTileDescriptor>> parts, ops::MoePhase phase,
             uint32_t rows);
  // Whether a load of `rows` rows of `phase` is a long prefill chunk's,
  // which routes nearly every expert of its block.
  [[nodiscard]] bool longChunk(ops::MoePhase phase, uint32_t rows) const noexcept {
    return phase == ops::MoePhase::Prefill &&
           uint64_t{rows} * shape.expertsPerToken >= kLongChunkRoutes * shape.experts;
  }
  // The experts a load of `rows` rows may route.
  [[nodiscard]] uint64_t routable(uint32_t rows) const noexcept {
    return std::min<uint64_t>(shape.experts, uint64_t{rows} * shape.expertsPerToken);
  }
  // A load's pool: the borrowed slots for a prefill load that may route more
  // experts than the staging places, else the staging.
  [[nodiscard]] PoolKind poolOf(ops::MoePhase phase, uint32_t rows) const noexcept {
    return phase == ops::MoePhase::Prefill && routable(rows) > pools[Staging].places.size() ? Borrowed : Staging;
  }
  [[nodiscard]] bool streams(uint32_t block) const noexcept { return geometry.streams(blocks[block].size); }
  void stage(uint32_t block, Pool &pool, ExpertStoreStats::Use &use);
  std::span<const uint32_t> predictedExperts(uint32_t rows);
  void readAhead(uint32_t block, std::span<const uint32_t> experts, PoolKind kind, ExpertStoreStats::Use &use);
  void runAhead();
  void readAheadPieces() noexcept;
  uint64_t readPiece(size_t piece);
  void takeAhead();
  bool waitAhead();
  void joinAhead();
  void evict(Size &size, uint32_t slot);
  uint32_t claim(Size &size);
  void borrow();
  void giveBack();
  void abandon();
  void fill();
  void publish();

  metal::MetalBackend &backend;
  const ops::MoeShape shape;
  const Geometry geometry;
  std::vector<std::function<void()>> checks;
  std::vector<Size> sizes;
  std::vector<Block> blocks;
  // Slot tables (blocks x experts), the staging, a load's jobs' slots and
  // their projections' places (MoE::addExpertLoad's sources), and its grid
  // (MTLDispatchThreadgroupsIndirectArguments), all Shared.
  metal::MetalBuffer tables, staging, jobs, sources, grid;
  // The staging's places, and those of the lender's last borrowedSlots
  // slots; the slots are out of its recency orders while `borrowing`.
  std::array<Pool, 2> pools;
  bool borrowing = false;
  // A decode step's prediction of the next layer's routes (addLoad): its
  // router's fp32 scores of this block's input, then each row's experts and
  // routing weights as the routing selects them (ops::MoE::addRoutes), for
  // up to a cycle's rows; each expert's summed weight over the rows (negative
  // when none) orders the experts it names (predictedExperts).
  metal::MetalBuffer predictionScores, predictionRoutes, predictionWeights;
  std::vector<float> predictionSum;
  std::vector<uint32_t> everyExpert, predictionOrder;
  // A read-ahead (readAhead): the native rows of block aheadBlock's experts
  // aheadExperts (most likely first), job j's in place aheadPlace[j] of pool
  // aheadPool, in aheadJobPieces pieces each (aheadList, expert after
  // expert), which kAheadReaders readers take in order (aheadNext) until the
  // next load takes the rest (takeAhead): the readers took the first
  // aheadTaken. It runs (aheadRunning) from its start until a wait for its
  // readers (waitAhead). aheadJob[e] is expert e's job there, aheadBytes the
  // bytes the readers read, which count in aheadUse, and aheadFailure the
  // first of their failures (aheadFailed).
  uint32_t aheadBlock = kNoSlot;
  PoolKind aheadPool = Staging;
  std::vector<uint32_t> aheadExperts, aheadPlace, aheadJob;
  std::vector<Piece> aheadList;
  uint32_t aheadJobPieces = 0;
  std::atomic<size_t> aheadNext = 0;
  size_t aheadTaken = 0;
  std::atomic<uint64_t> aheadBytes = 0;
  ExpertStoreStats::Use *aheadUse = nullptr;
  bool aheadRunning = false;
  std::atomic<bool> aheadFailed = false;
  std::exception_ptr aheadFailure;
  // The readers' thread (aheadThread), which a read-ahead wakes (aheadQueued,
  // under aheadMutex, until its readers end) and which the store's end ends
  // (aheadEnding).
  std::mutex aheadMutex;
  std::condition_variable aheadChanged;
  bool aheadQueued = false;
  bool aheadEnding = false;
  uint64_t arenaTotal = 0;

  // A load and release() or restore(), which never overlap (no command is in
  // flight while the store releases or restores its memory), hand the state
  // below between their threads under `mutex`. stats() reads `published`, a
  // copy of the counters each of them leaves under statsMutex, and released()
  // the engine thread's own flag: neither waits for a load's reads.
  mutable std::mutex mutex;
  uint64_t serial = 0;
  // The routed experts of block pendingBlock's load that were not resident,
  // with the slots they claimed, and the places of the pool it stages them
  // in.
  std::vector<std::pair<uint32_t, uint32_t>> pending;
  std::vector<uint32_t> jobPlace;
  std::vector<Read> reads;
  // The load's experts the readers of a read-ahead began: their jobs, and
  // the pieces of theirs the readers did not take, which the load reads.
  std::vector<uint32_t> aheadJobs;
  std::vector<size_t> aheadRest;
  uint32_t pendingBlock = kNoSlot;
  // The tiles of the load being routed, one span per step it serves.
  std::vector<std::span<const MoeTileDescriptor>> partTiles;
  // The load that last routed each expert of the block being loaded.
  std::vector<uint64_t> routedIn;
  std::string failure;
  ExpertStoreStats counters;
  mutable std::mutex statsMutex;
  ExpertStoreStats published;
  bool released = false;
  // Per ops::MoePhase, the round trips of the loads' host steps, which their
  // commands' completions add from Metal's thread.
  struct StepTimes final {
    LatencyHistogram pickup, work, resume;
  };
  std::array<StepTimes, 2> stepTimes;
  // Last, so that it starts once every other member is constructed.
  std::thread aheadThread{[this] { runAhead(); }};
};

// The expert in a slot, if any, is no longer resident.
void ExpertStore::Impl::evict(Size &size, uint32_t slot) {
  const uint32_t owner = size.ownerBlock[slot];
  if (owner == kNoSlot) return;
  const uint32_t expert = size.ownerExpert[slot];
  blocks[owner].slot[expert] = kNoSlot;
  // A table entry of no resident expert names slot 0, which is inside the
  // arena: the kernels read only routed experts' entries, which a load set,
  // so it is never used.
  static_cast<uint32_t *>(tables.contents())[uint64_t{owner} * shape.experts + expert] = 0;
  size.ownerBlock[slot] = size.ownerExpert[slot] = kNoSlot;
  --counters.resident;
}

uint32_t ExpertStore::Impl::claim(Size &size) {
  // The slots a load routes are at their order's most recent end: an order
  // whose head the load routed holds only the load's.
  const auto takes = [&](uint32_t slot) { return slot != kNoSlot && size.lastLoad[slot] != serial; };
  const uint32_t slot = takes(size.head[Size::Chunk]) ? size.head[Size::Chunk] : size.head[Size::Recent];
  if (!takes(slot)) throw std::runtime_error("the expert arena holds fewer slots than one load's experts");
  evict(size, slot);
  return slot;
}

// The slots prefill loads stage in leave the lender's recency orders, their
// experts evicted, before the first such load routes; the GPU, which runs
// one command at a time, reads none of them since the last decode step.
void ExpertStore::Impl::borrow() {
  Size &size = sizes[geometry.lender];
  for (uint32_t slot = size.slots - geometry.borrowedSlots; slot < size.slots; ++slot) {
    evict(size, slot);
    size.unlink(slot);
  }
  borrowing = true;
}

// Before a decode step's load routes, the borrowed slots come back free,
// recycled first, and a read-ahead into them is forgotten; the GPU read the
// last prefill load's staging before this step's.
void ExpertStore::Impl::giveBack() {
  if (aheadPool == Borrowed) aheadBlock = kNoSlot;
  Size &size = sizes[geometry.lender];
  for (uint32_t slot = size.slots - geometry.borrowedSlots; slot < size.slots; ++slot) {
    size.lastLoad[slot] = 0;
    size.prepend(slot, Size::Chunk);
  }
  borrowing = false;
}

// A long prefill chunk routes more experts than the arena holds (nearly
// every expert of every block): the experts it loads join Chunk, which the
// next blocks' loads recycle first, and those it routes keep their order. So
// the experts decode steps routed stay, and each chunk of a prompt reads
// what the last left out, where a single order would have each chunk evict
// the blocks the next one reads first, and the decode steps after the prompt
// read their experts again (with a 16 GiB arena, Qwen3.8-Flash-Next's Q2_0
// GGUF reads 38% fewer experts in prefill than with one order, and 6% fewer
// in decode). A short chunk's experts, which its decode steps route again,
// are recent as a decode step's.
void ExpertStore::Impl::route(uint32_t index, std::span<const std::span<const MoeTileDescriptor>> parts,
                              ops::MoePhase phase, uint32_t rows) {
  Block &block = blocks[index];
  Size &size = sizes[block.size];
  const bool chunk = longChunk(phase, rows);
  ++serial;
  pendingBlock = index;
  for (std::span<const MoeTileDescriptor> tiles : parts)
    for (const MoeTileDescriptor &tile : tiles) {
      // Past the routed experts' tiles come the shared expert's (id experts).
      const uint32_t expert = tile.expert;
      if (expert >= shape.experts || routedIn[expert] == serial) continue;
      routedIn[expert] = serial;
      ++counters.uses[static_cast<size_t>(phase)].routed;
      if (const uint32_t slot = block.slot[expert]; slot != kNoSlot) {
        size.touch(slot, serial, chunk ? Size::Order(size.order[slot]) : Size::Recent);
        continue;
      }
      pending.push_back({expert, kNoSlot});
    }
  // Consecutive experts read together (stage).
  std::ranges::sort(pending);
  uint32_t *table = static_cast<uint32_t *>(tables.contents()) + uint64_t{index} * shape.experts;
  for (auto &[expert, slot] : pending) {
    slot = claim(size);
    size.ownerBlock[slot] = index;
    size.ownerExpert[slot] = expert;
    size.touch(slot, serial, chunk ? Size::Chunk : Size::Recent);
    block.slot[expert] = slot;
    table[expert] = slot;
    ++counters.resident;
  }
}

// Stages the misses of block `index`'s load in `pool`: an expert the readers
// of a read-ahead into the pool began where they read it, the rest of it read
// now, and the others in the lowest places left, each projection's runs of
// consecutive experts whose places follow one another in reads of at most
// kLoadStepBytes (but one expert): a prefill chunk's are long, a decode
// step's mostly one expert. The pool places every expert the load may route.
void ExpertStore::Impl::stage(uint32_t index, Pool &pool, ExpertStoreStats::Use &use) {
  const Block &block = blocks[index];
  const Size &size = sizes[block.size];
  const auto aheadOf = [&](uint32_t expert) {
    return aheadBlock == index && aheadJob[expert] != kNoSlot && aheadJob[expert] * aheadJobPieces < aheadTaken;
  };
  std::ranges::fill(pool.taken, uint8_t{0});
  jobPlace.assign(pending.size(), kNoSlot);
  aheadJobs.clear();
  for (uint32_t job = 0; job < pending.size(); ++job)
    if (const uint32_t expert = pending[job].first; aheadOf(expert)) {
      jobPlace[job] = aheadPlace[aheadJob[expert]];
      pool.taken[jobPlace[job]] = 1;
      aheadJobs.push_back(job);
    }
  uint32_t free = 0;
  for (uint32_t &place : jobPlace) {
    if (place != kNoSlot) continue;
    while (free < pool.taken.size() && pool.taken[free]) ++free;
    if (free == pool.taken.size()) throw std::logic_error("an expert load misses more experts than its pool places");
    place = free;
    pool.taken[free] = 1;
  }
  reads.clear();
  for (uint32_t z = 0; z < 3; ++z) {
    const ProjectionLayout &projection = size.layout[z];
    const uint64_t bytes = projection.rows * projection.rowBytes;
    const uint64_t perRead = std::max<uint64_t>(1, kLoadStepBytes / bytes);
    for (uint32_t job = 0; job < pending.size();) {
      const uint32_t expert = pending[job].first;
      if (aheadOf(expert)) {
        ++job;
        continue;
      }
      uint32_t end = job + 1;
      while (end < pending.size() && end - job < perRead && pending[end].first == pending[end - 1].first + 1 &&
             pool.places[jobPlace[end]][z] == pool.places[jobPlace[end - 1]][z] + bytes &&
             !aheadOf(pending[end].first))
        ++end;
      reads.push_back({z, job, expert, end - job});
      job = end;
    }
  }
  aheadRest.clear();
  for (const uint32_t job : aheadJobs) {
    const size_t at = aheadJob[pending[job].first];
    for (size_t piece = std::max(at * aheadJobPieces, aheadTaken); piece < (at + 1) * aheadJobPieces; ++piece)
      aheadRest.push_back(piece);
  }
  auto *rows = static_cast<uint8_t *>(pool.buffer->contents());
  std::atomic<uint64_t> restBytes = 0;
  parallelFor(reads.size() + aheadRest.size(), [&](size_t task, unsigned) {
    if (task >= reads.size()) {
      restBytes += readPiece(aheadRest[task - reads.size()]);
      return;
    }
    const Read &read = reads[task];
    const ProjectionLayout &projection = size.layout[read.z];
    readRepackRows(block.experts.projections[read.z], uint64_t{read.expert} * projection.rows,
                   uint64_t{read.count} * projection.rows, rows + pool.places[jobPlace[read.job]][read.z]);
  });
  // What the load read is the sources' as they were opened.
  if (!pending.empty())
    for (const std::function<void()> &check : checks) check();
  auto *slots = static_cast<uint32_t *>(jobs.contents());
  auto *places = static_cast<uint64_t *>(sources.contents());
  for (size_t job = 0; job < pending.size(); ++job) {
    slots[job] = pending[job].second;
    std::ranges::copy(pool.places[jobPlace[job]], places + 3 * job);
  }
  use.read += pending.size();
  for (const uint32_t job : aheadJobs)
    use.prefetched += (aheadJob[pending[job].first] + 1) * aheadJobPieces <= aheadTaken;
  use.readBytes += restBytes;
  for (const Read &read : reads) use.readBytes += read.count * size.layout[read.z].rows * size.layout[read.z].rowBytes;
  auto *threadgroups = static_cast<uint32_t *>(grid.contents());
  threadgroups[0] = size.loadGroups;
  threadgroups[1] = static_cast<uint32_t>(pending.size());
  threadgroups[2] = 3;
}

// The next block's experts a decode step's prediction names for its `rows`
// rows, by their summed routing weights, the largest first.
std::span<const uint32_t> ExpertStore::Impl::predictedExperts(uint32_t rows) {
  const auto *selected = static_cast<const uint32_t *>(predictionRoutes.contents());
  const auto *weights = static_cast<const float *>(predictionWeights.contents());
  const uint32_t routes = shape.routesPerToken();
  predictionOrder.clear();
  for (uint64_t row = 0; row < rows; ++row)
    for (uint32_t route = 0; route < shape.expertsPerToken; ++route) {
      const uint64_t at = row * routes + route;
      const uint32_t expert = selected[at];
      if (expert >= shape.experts) continue;
      float &sum = predictionSum[expert];
      if (sum < 0) {
        sum = 0;
        predictionOrder.push_back(expert);
      }
      sum += weights[at];
    }
  std::ranges::stable_sort(predictionOrder, std::ranges::greater{},
                           [&](uint32_t expert) { return predictionSum[expert]; });
  for (const uint32_t expert : predictionOrder) predictionSum[expert] = -1;
  return predictionOrder;
}

// Reads ahead, in the background, those of block `index`'s experts
// `experts` (most likely first) that are not resident, in that order, into
// the places of pool `kind` the load just staged left free, as many as
// there are.
void ExpertStore::Impl::readAhead(uint32_t index, std::span<const uint32_t> experts, PoolKind kind,
                                  ExpertStoreStats::Use &use) {
  const Block &block = blocks[index];
  const Size &size = sizes[block.size];
  const Pool &pool = pools[kind];
  aheadExperts.clear();
  aheadPlace.clear();
  uint32_t free = 0;
  for (const uint32_t expert : experts) {
    if (block.slot[expert] != kNoSlot) continue;
    while (free < pool.taken.size() && pool.taken[free]) ++free;
    if (free == pool.taken.size()) break;
    aheadExperts.push_back(expert);
    aheadPlace.push_back(free++);
  }
  std::ranges::fill(aheadJob, kNoSlot);
  if (aheadExperts.empty()) return;
  aheadList.clear();
  for (uint32_t job = 0; job < aheadExperts.size(); ++job) {
    aheadJob[aheadExperts[job]] = job;
    for (uint32_t z = 0; z < 3; ++z) {
      // Even pieces of at most kAheadPieceBytes (but one row).
      const ProjectionLayout &projection = size.layout[z];
      const uint64_t pieces = (projection.rows * projection.rowBytes + kAheadPieceBytes - 1) / kAheadPieceBytes;
      const uint64_t perPiece = (projection.rows + pieces - 1) / pieces;
      for (uint64_t first = 0; first < projection.rows; first += perPiece)
        aheadList.push_back({job, z, first, std::min(perPiece, projection.rows - first)});
    }
  }
  aheadJobPieces = static_cast<uint32_t>(aheadList.size() / aheadExperts.size());
  aheadBlock = index;
  aheadPool = kind;
  aheadUse = &use;
  aheadBytes = 0;
  aheadNext = 0;
  aheadTaken = 0;
  aheadFailed = false;
  aheadRunning = true;
  {
    const std::lock_guard lock(aheadMutex);
    aheadQueued = true;
  }
  aheadChanged.notify_all();
}

// The readers' thread: it reads each read-ahead's pieces with kAheadReaders
// readers (parallelFor), itself one of them, so that reading starts at once.
// It runs at the default quality of service, below the host steps': its other
// readers take libdispatch threads of that class, never one the load that
// waits for them reads with.
void ExpertStore::Impl::runAhead() {
  pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, 0);
  std::unique_lock lock(aheadMutex);
  for (;;) {
    aheadChanged.wait(lock, [this] { return aheadQueued || aheadEnding; });
    if (aheadEnding) return;
    lock.unlock();
    parallelFor(kAheadReaders, [this](size_t, unsigned) { readAheadPieces(); }, kAheadReaders);
    lock.lock();
    aheadQueued = false;
    aheadChanged.notify_all();
  }
}

// A reader of a read-ahead: it reads the pieces it takes, in order, until
// none is left. The first failure of the readers is the next load's
// (joinAhead).
void ExpertStore::Impl::readAheadPieces() noexcept {
  try {
    for (size_t piece; (piece = aheadNext++) < aheadList.size();) aheadBytes += readPiece(piece);
  } catch (...) {
    if (!aheadFailed.exchange(true)) aheadFailure = std::current_exception();
  }
}

// Reads piece `index` of the read-ahead into its expert's place; returns
// its bytes.
uint64_t ExpertStore::Impl::readPiece(size_t index) {
  const Piece &piece = aheadList[index];
  const Block &block = blocks[aheadBlock];
  const Size &size = sizes[block.size];
  const Pool &pool = pools[aheadPool];
  const ProjectionLayout &projection = size.layout[piece.z];
  readRepackRows(block.experts.projections[piece.z], uint64_t{aheadExperts[piece.job]} * projection.rows + piece.first,
                 piece.count,
                 static_cast<uint8_t *>(pool.buffer->contents()) + pool.places[aheadPlace[piece.job]][piece.z] +
                     piece.first * projection.rowBytes);
  return piece.count * projection.rowBytes;
}

// The load after a read-ahead takes the pieces its readers have not: they
// end with the pieces they took.
void ExpertStore::Impl::takeAhead() {
  if (aheadRunning) aheadTaken = std::min(aheadNext.exchange(aheadList.size()), aheadList.size());
}

// Waits for the readers of a read-ahead to end the pieces they took; false
// when none was running.
bool ExpertStore::Impl::waitAhead() {
  if (!aheadRunning) return false;
  std::unique_lock lock(aheadMutex);
  aheadChanged.wait(lock, [this] { return !aheadQueued; });
  aheadRunning = false;
  return true;
}

// Waits for the readers of a read-ahead the load took over, and counts the
// experts they read whole and the bytes they read; one that failed is
// forgotten and fails the load.
void ExpertStore::Impl::joinAhead() {
  if (!waitAhead()) return;
  if (aheadFailure) {
    aheadBlock = kNoSlot;
    std::rethrow_exception(std::exchange(aheadFailure, nullptr));
  }
  aheadUse->readAhead += aheadTaken / aheadJobPieces;
  aheadUse->readBytes += aheadBytes;
}

// Frees the slots a failed load claimed: their experts were not written.
void ExpertStore::Impl::abandon() {
  if (pendingBlock == kNoSlot) return;
  Block &block = blocks[pendingBlock];
  Size &size = sizes[block.size];
  uint32_t *table = static_cast<uint32_t *>(tables.contents()) + uint64_t{pendingBlock} * shape.experts;
  for (const auto &[expert, slot] : pending) {
    if (slot == kNoSlot || block.slot[expert] != slot) continue;
    block.slot[expert] = kNoSlot;
    table[expert] = 0;
    size.ownerBlock[slot] = size.ownerExpert[slot] = kNoSlot;
    size.lastLoad[slot] = 0;
    // A free slot is the first to be claimed.
    size.unlink(slot);
    size.prepend(slot, Size::Chunk);
    --counters.resident;
  }
  pending.clear();
  pendingBlock = kNoSlot;
}

// Reads every expert of each size of one block into its slot, expert e into
// slot e, through the staging, as many a command as it places.
void ExpertStore::Impl::fill() {
  for (uint32_t index = 0; index < blocks.size(); ++index) {
    if (streams(index)) continue;
    Block &block = blocks[index];
    Size &size = sizes[block.size];
    const std::pair<uint64_t, uint64_t> range{0, staging.sizeBytes()};
    const std::array<uint64_t, 3> width = nativeBytes(size.layout);
    uint64_t count = shape.experts;
    while (count && !place({&range, 1}, width, count)) --count;
    if (!count) throw std::invalid_argument("the expert staging holds less than one expert");
    const std::vector<std::array<uint64_t, 3>> placed = *place({&range, 1}, width, count);
    uint32_t *table = static_cast<uint32_t *>(tables.contents()) + uint64_t{index} * shape.experts;
    for (uint32_t expert = 0; expert < shape.experts; expert += static_cast<uint32_t>(count)) {
      const uint32_t batch = static_cast<uint32_t>(std::min<uint64_t>(count, shape.experts - expert));
      parallelFor(3, [&](size_t z, unsigned) {
        readRepackRows(block.experts.projections[z], uint64_t{expert} * size.layout[z].rows,
                       uint64_t{batch} * size.layout[z].rows,
                       static_cast<uint8_t *>(staging.contents()) + placed[0][z]);
      });
      for (const std::function<void()> &check : checks) check();
      auto *slots = static_cast<uint32_t *>(jobs.contents());
      auto *places = static_cast<uint64_t *>(sources.contents());
      for (uint32_t job = 0; job < batch; ++job) {
        const uint32_t slot = expert + job;
        slots[job] = slot;
        std::ranges::copy(placed[job], places + 3 * job);
        size.ownerBlock[slot] = index;
        size.ownerExpert[slot] = slot;
        block.slot[slot] = slot;
        table[slot] = slot;
      }
      auto *threadgroups = static_cast<uint32_t *>(grid.contents());
      threadgroups[0] = size.loadGroups;
      threadgroups[1] = batch;
      threadgroups[2] = 3;
      metal::CommandGraph graph;
      ops::MoE::addExpertLoad(graph, staging, size.arena, jobs, sources, size.load, batch,
                              metal::IndirectGrid{grid, 0});
      static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    }
    counters.resident += shape.experts;
  }
}

// Leaves a copy of the counters and the failure for stats().
void ExpertStore::Impl::publish() {
  const std::lock_guard lock(statsMutex);
  published = counters;
  published.failure = failure;
}

void ExpertStore::Impl::load(uint32_t index, ops::MoePhase phase, uint32_t rows, bool predicted,
                             std::span<const PartTiles> parts) {
  std::lock_guard lock(mutex);
  // No job runs until the load's are staged.
  static_cast<uint32_t *>(grid.contents())[1] = 0;
  if (!failure.empty()) throw std::runtime_error("expert store failed earlier: " + failure);
  // A load that fails before it routes frees none of the last load's slots
  // (abandon).
  pending.clear();
  pendingBlock = kNoSlot;
  ExpertStoreStats::Use &use = counters.uses[static_cast<size_t>(phase)];
  try {
    // The read-ahead the load takes over ends first: its readers finish the
    // pieces they took, a few hundred KiB each.
    takeAhead();
    joinAhead();
    const PoolKind kind = poolOf(phase, rows);
    // A prefill load staging in borrowed slots takes them out of the
    // recency orders before it routes; a decode load gives them back.
    if (phase == ops::MoePhase::Decode && borrowing) giveBack();
    if (kind == Borrowed && !borrowing) borrow();
    // A read-ahead serves the load after it alone, from the same pool.
    if (aheadPool != kind) aheadBlock = kNoSlot;
    partTiles.clear();
    for (const PartTiles &part : parts) {
      const auto *counted = static_cast<const MoeTileCount *>(part.count.contents());
      const auto *descriptors = static_cast<const MoeTileDescriptor *>(part.descriptors.contents());
      const uint64_t bound = part.descriptors.sizeBytes() / sizeof(MoeTileDescriptor);
      if (!counted || !descriptors || counted->tiles > bound)
        throw std::runtime_error("a MoE block's tiles are out of bounds");
      partTiles.push_back({descriptors, counted->tiles});
    }
    route(index, partTiles, phase, rows);
    ++use.loads;
    stage(index, pools[kind], use);
    // Once the block's misses are staged, the next block's likely experts
    // are read ahead while the GPU runs this one.
    aheadBlock = kNoSlot;
    if (longChunk(phase, rows) && index + 1 < blocks.size() && streams(index + 1))
      readAhead(index + 1, everyExpert, kind, use);
    else if (predicted)
      readAhead(index + 1, predictedExperts(rows), kind, use);
  } catch (const std::exception &error) {
    abandon();
    failure = error.what();
    publish();
    throw;
  }
  publish();
}

ExpertStore::ExpertStore(metal::MetalBackend &backend, StreamedBlocks streamed, ops::MoeShape shape,
                         ExpertStoreOptions options)
    : impl_(std::make_shared<Impl>(backend, shape, geometryOf(streamed, shape))) {
  Impl &impl = *impl_;
  const Geometry &geometry = impl.geometry;
  if (!options.memoryBytes)
    throw std::invalid_argument("invalid expert store");
  impl.checks = std::move(streamed.checks);
  for (uint32_t size = 0; size < geometry.layouts.size(); ++size)
    impl.sizes.push_back({.layout = geometry.layouts[size], .blocks = geometry.blocks[size]});
  for (uint32_t block = 0; block < streamed.blocks.size(); ++block)
    impl.blocks.push_back({.size = geometry.sizeOf[block],
                           .experts = std::move(streamed.blocks[block]),
                           .slot = std::vector<uint32_t>(shape.experts, kNoSlot)});
  const uint64_t fixed = fixedBytesOf(geometry, static_cast<uint32_t>(impl.blocks.size()), shape);
  const uint64_t arenaBudget = options.memoryBytes > fixed ? options.memoryBytes - fixed : 0;
  // Every streamed size holds the same share of its experts, but at least a
  // block's (a prefill chunk may route them all) and at most all of them,
  // and the lender the slots it lends besides; a size of one block all of
  // its: the largest share the arena holds.
  const auto slotsAt = [&](uint32_t size, double share) {
    const uint64_t all = uint64_t{geometry.blocks[size]} * shape.experts;
    if (!geometry.streams(size)) return all;
    return std::clamp<uint64_t>(static_cast<uint64_t>(share * double(all)), shape.experts, all) +
           (size == geometry.lender ? geometry.borrowedSlots : 0);
  };
  const auto bytesAt = [&](double share) {
    uint64_t bytes = 0;
    for (uint32_t size = 0; size < impl.sizes.size(); ++size)
      bytes += arenaBytes(impl.sizes[size].layout, slotsAt(size, share));
    return bytes;
  };
  if (bytesAt(0.0) > arenaBudget)
    throw std::invalid_argument("expert memory of " + std::to_string(options.memoryBytes >> 20) +
                                " MiB holds fewer slots (" + std::to_string(bytesAt(0.0) >> 20) +
                                " MiB) than a block's experts of each size, every expert of a size of one block "
                                "and the slots prefill loads borrow, beside the staging and tables (" +
                                std::to_string(fixed >> 20) + " MiB)");
  double low = 0.0, high = 1.0;
  if (bytesAt(high) <= arenaBudget) low = high;
  for (int step = 0; step < 64 && low < high; ++step) {
    const double middle = (low + high) / 2;
    (bytesAt(middle) <= arenaBudget ? low : high) = middle;
  }
  for (uint32_t index = 0; index < impl.sizes.size(); ++index) {
    Impl::Size &size = impl.sizes[index];
    size.slots = static_cast<uint32_t>(slotsAt(index, low));
    uint64_t region = 0;
    for (size_t z = 0; z < 3; ++z) {
      const ProjectionLayout &projection = size.layout[z];
      for (size_t plane = 0; plane < 3; ++plane) {
        size.regions[z][plane] = region;
        region += alignWeightOffset(uint64_t{size.slots} * projection.planes[plane]);
      }
      size.load.projections[z] = {projection.format,
                                  static_cast<uint32_t>(projection.rows),
                                  static_cast<uint32_t>(projection.columns),
                                  static_cast<uint32_t>(projection.rowBytes),
                                  {size.regions[z][0], size.regions[z][1], size.regions[z][2]},
                                  {projection.planes[0], projection.planes[1], projection.planes[2]}};
    }
    size.loadGroups = ops::MoE::expertLoadGroups(size.load);
    size.arena = backend.allocateBuffer(region, metal::BufferStorage::Shared,
                                        "experts/arena-" + std::to_string(index));
    impl.arenaTotal += region;
    size.previous.resize(size.slots);
    size.next.resize(size.slots);
    size.order.resize(size.slots);
    size.ownerBlock.assign(size.slots, kNoSlot);
    size.ownerExpert.assign(size.slots, kNoSlot);
    size.lastLoad.assign(size.slots, 0);
    size.reset();
  }
  impl.staging = backend.allocateBuffer(geometry.stagingBytes, metal::BufferStorage::Shared, "experts/staging");
  const std::pair<uint64_t, uint64_t> whole{0, geometry.stagingBytes};
  impl.pools[Impl::Staging] = {&impl.staging, *place({&whole, 1}, geometry.width, geometry.stagingPlaces), {}};
  // The borrowed slots' bytes in each plane region of the lender, where they
  // are.
  if (geometry.borrowedSlots) {
    const Impl::Size &lender = impl.sizes[geometry.lender];
    const uint64_t first = lender.slots - geometry.borrowedSlots;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (size_t z = 0; z < 3; ++z)
      for (size_t plane = 0; plane < 3; ++plane)
        if (const uint64_t bytes = lender.layout[z].planes[plane])
          ranges.emplace_back(lender.regions[z][plane] + first * bytes,
                              lender.regions[z][plane] + lender.slots * bytes);
    const auto places = place(ranges, geometry.width, kBorrowedBlocks * shape.experts);
    if (!places) throw std::logic_error("the borrowed slots do not place their experts");
    impl.pools[Impl::Borrowed] = {&lender.arena, *places, {}};
  }
  for (Impl::Pool &pool : impl.pools) pool.taken.assign(pool.places.size(), 0);
  const uint64_t tableBytes = uint64_t{shape.experts} * impl.blocks.size() * sizeof(uint32_t);
  impl.tables = backend.allocateBuffer(tableBytes, metal::BufferStorage::Shared, "experts/slot-tables");
  std::memset(impl.tables.contents(), 0, tableBytes);
  impl.aheadJob.assign(shape.experts, kNoSlot);
  const auto predictionBuffer = [&](uint64_t perRow, const char *label) {
    return backend.allocateBuffer(uint64_t{kCycleRows} * perRow * sizeof(float),
                                  metal::BufferStorage::Shared, label);
  };
  impl.predictionScores = predictionBuffer(shape.routerWidth(), "experts/prediction-scores");
  impl.predictionRoutes = predictionBuffer(shape.routesPerToken(), "experts/prediction-routes");
  impl.predictionWeights = predictionBuffer(shape.routesPerToken(), "experts/prediction-weights");
  impl.predictionSum.assign(shape.experts, -1);
  impl.everyExpert.resize(shape.experts);
  for (uint32_t expert = 0; expert < shape.experts; ++expert) impl.everyExpert[expert] = expert;
  impl.jobs = backend.allocateBuffer(uint64_t{shape.experts} * sizeof(uint32_t), metal::BufferStorage::Shared,
                                     "experts/jobs");
  impl.sources = backend.allocateBuffer(uint64_t{shape.experts} * 3 * sizeof(uint64_t), metal::BufferStorage::Shared,
                                        "experts/sources");
  impl.grid = backend.allocateBuffer(4 * sizeof(uint32_t), metal::BufferStorage::Shared, "experts/grid");
  std::memset(impl.grid.contents(), 0, 4 * sizeof(uint32_t));
  impl.routedIn.assign(shape.experts, 0);
  impl.counters.arenaBytes = impl.arenaTotal;
  impl.counters.borrowed = geometry.borrowedSlots;
  for (const Impl::Size &size : impl.sizes) {
    impl.counters.slots += size.slots;
    impl.counters.experts += uint64_t{size.blocks} * shape.experts;
  }
  impl.fill();
  impl.publish();
}

ExpertStore::~ExpertStore() = default;

void ExpertStore::bind(uint32_t block, ops::BlockMoeWeights &weights) {
  Impl &impl = *impl_;
  Impl::Block &streamed = impl.blocks.at(block);
  streamed.router = weights.router;
  streamed.scalarGate = weights.sharedScalarGate;
  const Impl::Size &size = impl.sizes[streamed.size];
  ops::BlockExpertProjection *const projections[] = {&weights.gate, &weights.up, &weights.down};
  for (size_t z = 0; z < 3; ++z) {
    const ProjectionLayout &projection = size.layout[z];
    const auto plane = [&](size_t index) {
      const uint64_t bytes = uint64_t{size.slots} * projection.planes[index];
      return bytes ? impl.backend.view(size.arena, size.regions[z][index], bytes) : metal::MetalBuffer{};
    };
    projections[z]->routed =
        ops::QuantizedSegment::planes(projection.format, static_cast<uint32_t>(size.slots * projection.rows),
                                      static_cast<uint32_t>(projection.columns), plane(0), plane(1), plane(2));
  }
  weights.slotTable = impl.backend.view(impl.tables, uint64_t{block} * impl.shape.experts * sizeof(uint32_t),
                                        uint64_t{impl.shape.experts} * sizeof(uint32_t));
  weights.slots = size.slots;
  // The experts of a size of one block stay in their slots: nothing loads
  // them.
  weights.stream = impl.streams(block) ? this : nullptr;
  weights.streamBlock = block;
}

void ExpertStore::addLoad(metal::CommandGraph &graph, std::span<const ops::MoeLoadPart> parts,
                          uint32_t block) const {
  const Impl &impl = *impl_;
  const Impl::Size &size = impl.sizes[impl.blocks.at(block).size];
  const ops::MoeShape shape = impl.shape;
  if (parts.empty()) throw std::invalid_argument("an expert load serves no step");
  if (!impl.streams(block)) throw std::invalid_argument("a block of a size of one block does not load");
  const ops::MoePhase phase = parts.front().plan->phase();
  uint32_t rows = 0;
  std::vector<Impl::PartTiles> tiles;
  tiles.reserve(parts.size());
  for (const ops::MoeLoadPart &part : parts) {
    if (part.plan->phase() != phase) throw std::invalid_argument("an expert load's steps differ in phase");
    rows += part.plan->rows();
    tiles.push_back({part.buffers->scratch.tileDescriptors, part.buffers->scratch.tileCount});
  }
  const Impl::Pool &pool = impl.pools[impl.poolOf(phase, rows)];
  const uint64_t routable = impl.routable(rows);
  if (routable > pool.places.size()) throw std::logic_error("an expert load may route more experts than it stages");
  // A decode step's layer predicts the next layer's routes, when its experts
  // stream: that layer's routing of this block's input.
  const bool predicted =
      phase == ops::MoePhase::Decode && block + 1 < impl.blocks.size() && impl.streams(block + 1);
  if (predicted) {
    if (parts.size() != 1) throw std::invalid_argument("a decode load serves one step");
    const Impl::Block &next = impl.blocks[block + 1];
    ops::MoE::addRoutes(graph, parts.front().buffers->input, next.router, next.scalarGate, impl.predictionScores,
                        impl.predictionRoutes, impl.predictionWeights, rows, shape,
                        parts.front().plan->configuration().ggufRouterTile);
  }
  // A decode step's loads keep the GPU awake: they come about every
  // millisecond and most end within one, about as long as an idle GPU takes
  // to come back to full speed; a prefill chunk's read for tens of ms.
  const bool awake = phase == ops::MoePhase::Decode;
  graph.host([impl = impl_, block, phase, rows, predicted, tiles] { impl->load(block, phase, rows, predicted, tiles); },
             [impl = impl_, phase](const metal::HostStepTiming &timing) {
               Impl::StepTimes &times = impl->stepTimes[static_cast<size_t>(phase)];
               const auto nanoseconds = [](double seconds) { return uint64_t(seconds * 1e9); };
               times.pickup.add(nanoseconds(timing.pickupSeconds));
               times.work.add(nanoseconds(timing.workSeconds));
               times.resume.add(nanoseconds(timing.resumeSeconds));
             },
             awake);
  ops::MoE::addExpertLoad(graph, *pool.buffer, size.arena, impl.jobs, impl.sources, size.load,
                          static_cast<uint32_t>(routable), metal::IndirectGrid{impl.grid, 0});
}

ExpertStoreStats ExpertStore::stats() const {
  ExpertStoreStats result;
  {
    const std::lock_guard lock(impl_->statsMutex);
    result = impl_->published;
  }
  for (size_t use = 0; use < impl_->stepTimes.size(); ++use) {
    const Impl::StepTimes &times = impl_->stepTimes[use];
    result.uses[use].pickup = times.pickup.snapshot();
    result.uses[use].work = times.work.snapshot();
    result.uses[use].resume = times.resume.snapshot();
  }
  return result;
}

bool ExpertStore::released() const noexcept { return impl_->released; }

void ExpertStore::release() {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  if (impl.released) return;
  // A read-ahead in flight writes into the memory released below: its
  // readers end the pieces they took, and it is forgotten.
  impl.takeAhead();
  impl.waitAhead();
  impl.aheadFailure = nullptr;
  impl.aheadBlock = kNoSlot;
  impl.borrowing = false;
  for (Impl::Block &block : impl.blocks) std::ranges::fill(block.slot, kNoSlot);
  std::memset(impl.tables.contents(), 0, impl.tables.sizeBytes());
  for (Impl::Size &size : impl.sizes) {
    impl.backend.releaseMemory(size.arena);
    std::ranges::fill(size.ownerBlock, kNoSlot);
    std::ranges::fill(size.ownerExpert, kNoSlot);
    std::ranges::fill(size.lastLoad, 0);
    size.reset();
  }
  impl.backend.releaseMemory(impl.staging);
  impl.counters.resident = 0;
  impl.released = true;
  impl.publish();
}

bool ExpertStore::restore(const metal::AllocationAdmission &admit) {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  if (!impl.released) throw std::logic_error("expert memory is not released");
  uint64_t bytes = impl.staging.sizeBytes();
  for (const Impl::Size &size : impl.sizes) bytes += size.arena.sizeBytes();
  std::vector<const metal::MetalBuffer *> restored;
  const metal::AllocationResult admitted = admit(bytes, [&] {
    try {
      for (const Impl::Size &size : impl.sizes) {
        impl.backend.restoreMemory(size.arena);
        restored.push_back(&size.arena);
      }
      impl.backend.restoreMemory(impl.staging);
    } catch (const metal::MetalAllocationError &) {
      // A buffer the driver refused leaves them all released.
      for (const metal::MetalBuffer *buffer : restored) impl.backend.releaseMemory(*buffer);
      throw;
    }
  });
  if (!admitted)
    throw metal::MetalAllocationError(std::string("unable to restore the expert arena: ") +
                                          metal::allocationFailureName(admitted.failure),
                                      admitted.failure);
  impl.released = false;
  impl.fill();
  impl.publish();
  return true;
}

uint64_t ExpertStore::fixedBytes(const StreamedBlocks &blocks, ops::MoeShape shape) {
  return fixedBytesOf(geometryOf(blocks, shape), static_cast<uint32_t>(blocks.blocks.size()), shape);
}

uint32_t ExpertStore::borrowedSlots(const StreamedBlocks &blocks, ops::MoeShape shape) {
  return geometryOf(blocks, shape).borrowedSlots;
}

metal::MetalBuffer residentExpertSlots(metal::MetalBackend &backend, uint32_t experts) {
  metal::MetalBuffer table =
      backend.allocateBuffer(uint64_t{experts} * sizeof(uint32_t), metal::BufferStorage::Shared,
                             "experts/resident-slots");
  auto *slot = static_cast<uint32_t *>(table.contents());
  for (uint32_t expert = 0; expert < experts; ++expert) slot[expert] = expert;
  return table;
}

} // namespace splash::model

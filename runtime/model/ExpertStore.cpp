#include "model/ExpertStore.hpp"

#include "metal/abi/MoE.h"
#include "model/ExpertSlots.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/GgufPreparation.hpp"
#include "model/Model.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightSource.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>
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
// readers that read ahead: what the round that stops a read-ahead waits for
// is the pieces its readers took. An M5 Max's SSD reads 460 KiB pieces
// at 7.25 GB/s from four threads and at 7.41 GB/s from eighteen, but each
// then takes 276 us instead of 1223 us.
constexpr uint64_t kAheadPieceBytes = 512 << 10;
constexpr unsigned kAheadReaders = 4;
// Apple silicon's pages: a read the kernel does straight into memory
// (F_NOCACHE without the buffer cache) starts and ends on them in the file
// and in memory. With its ends elsewhere it goes through the buffer cache:
// on an M5 Pro, four threads read 460 KiB pieces at 8.1 GB/s that way and
// at 9.3 GB/s (of the pieces' bytes) widened to whole pages, with 0.04
// instead of 0.09 CPU seconds per GB.
constexpr uint64_t kPageBytes = 16 << 10;
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
  [[nodiscard]] uint64_t nativeBytes() const noexcept { return rows * rowBytes; }
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

// The arena bytes of `slots` experts of a layout: each projection's planes
// in a region of their own.
uint64_t arenaBytes(const ExpertLayout &layout, uint64_t slots) noexcept {
  uint64_t bytes = 0;
  for (const ProjectionLayout &projection : layout)
    for (const uint64_t plane : projection.planes) bytes += alignWeightOffset(slots * plane);
  return bytes;
}

// Places of experts in the staging: place p's rows of projection z from
// at(p, z) on, in width[z] bytes, projection after projection, place after
// place. A place is a page wider than its expert's rows widened to whole
// pages, so that the rows of a run of consecutive experts in consecutive
// places, read widened to whole pages from the first place on, fit in them
// (Impl::Run), and every place starts on a page.
struct Places final {
  std::array<uint64_t, 3> base{}, width{};
  uint32_t count = 0;

  Places() = default;
  Places(const std::array<uint64_t, 3> &nativeBytes, uint32_t places) : count(places) {
    uint64_t at = 0;
    for (size_t z = 0; z < 3; ++z) {
      base[z] = at;
      width[z] = alignUp(nativeBytes[z], kPageBytes) + kPageBytes;
      at += uint64_t{places} * width[z];
    }
  }
  [[nodiscard]] uint64_t at(uint32_t place, uint32_t z) const noexcept { return base[z] + place * width[z]; }
  [[nodiscard]] uint64_t bytes() const noexcept { return base[2] + uint64_t{count} * width[2]; }
};

// What a store's blocks alone decide (the store and ExpertStore::
// fixedBytes): each block's layout, the layouts and their blocks, and the
// native bytes of each projection of a place, the most of a streamed
// layout's expert (of any layout's when none streams: the staging then only
// reads the sizes of one block, Impl::fill).
struct Geometry final {
  std::vector<uint32_t> layoutOf;
  std::vector<ExpertLayout> layouts;
  std::vector<uint32_t> blocks;
  std::array<uint64_t, 3> nativeBytes{};

  // A layout of several blocks streams; one of one block holds every expert.
  [[nodiscard]] bool streams(uint32_t layout) const noexcept { return blocks[layout] > 1; }
};

Geometry geometryOf(const StreamedBlocks &streamed, ops::MoeShape shape) {
  if (!shape.valid() || streamed.blocks.empty()) throw std::invalid_argument("invalid expert store");
  Geometry g;
  for (const gguf::StreamedExperts &experts : streamed.blocks) {
    const ExpertLayout layout = expertLayout(experts, shape);
    const auto found = std::ranges::find(g.layouts, layout);
    const uint32_t index = static_cast<uint32_t>(found - g.layouts.begin());
    if (found == g.layouts.end()) {
      g.layouts.push_back(layout);
      g.blocks.push_back(0);
    }
    ++g.blocks[index];
    g.layoutOf.push_back(index);
  }
  const bool streaming = std::ranges::any_of(g.blocks, [](uint32_t blocks) { return blocks > 1; });
  for (uint32_t layout = 0; layout < g.layouts.size(); ++layout)
    if (g.streams(layout) || !streaming)
      for (size_t z = 0; z < 3; ++z)
        g.nativeBytes[z] = std::max(g.nativeBytes[z], g.layouts[layout][z].nativeBytes());
  return g;
}

// The staging: the experts of a block a decode cycle's rows may route, and
// at least two places, of which a round of a load that is not a decode
// step's stages one half.
Places stagingOf(const Geometry &g, ops::MoeShape shape) {
  return {g.nativeBytes,
          static_cast<uint32_t>(std::max<uint64_t>(
              2, std::min<uint64_t>(shape.experts, uint64_t{kCycleRows} * shape.expertsPerToken)))};
}

uint64_t fixedBytesOf(const Geometry &g, uint32_t blocks, ops::MoeShape shape) {
  const uint64_t prediction =
      uint64_t{kCycleRows} * (shape.routerWidth() + 2ULL * shape.routesPerToken()) * sizeof(float);
  return stagingOf(g, shape).bytes() + uint64_t{blocks} * shape.experts * sizeof(uint32_t) +
         uint64_t{shape.experts} * (sizeof(uint32_t) + 3 * sizeof(uint64_t)) + 4 * sizeof(uint32_t) + prediction;
}

} // namespace

struct ExpertStore::Impl {
  // A run of the rows of projection z of consecutive experts in consecutive
  // places, from staging byte `at` on: read straight from `file` that stores
  // them, `bytes` of it from `offset` on (the rows widened to whole pages),
  // the first expert's rows `inPage` bytes in; or, without a file, built by
  // readRepackRows into `bytes` bytes.
  struct Run final {
    const WeightSource *file = nullptr;
    uint64_t at = 0, offset = 0, bytes = 0, inPage = 0;
  };
  // `count` consecutive experts of block `block` from `expert` on, in
  // consecutive places from `place` on, and each projection's run of them.
  struct Group final {
    uint32_t block = 0, expert = 0, count = 0, place = 0;
    std::array<Run, 3> runs;
  };
  // A read of bytes [from, to) of group `group`'s run of projection z.
  struct Piece final {
    uint32_t group = 0, z = 0;
    uint64_t from = 0, to = 0;
  };
  struct Reads final {
    std::vector<Group> groups;
    std::vector<Piece> pieces;
  };
  // The experts of one layout in one arena (of some of its blocks, when the
  // layout's would exceed the largest buffer): their slots, a streamed
  // size's recycled in its order (ExpertSlots).
  struct Size final {
    ExpertLayout layout;
    uint32_t blocks = 0;
    // Whether its layout is of several blocks (Geometry::streams).
    bool streams = false;
    uint32_t slots = 0;
    // Projection z's plane i from regions[z][i] of the arena on.
    std::array<std::array<uint64_t, 3>, 3> regions{};
    metal::MetalBuffer arena;
    // The repack of its staged rows into slots.
    MoeExpertLoadParams load{};
    uint32_t loadGroups = 0;
    // Which slot a load's missing expert takes.
    ExpertSlots recycling{0};
    std::vector<uint32_t> ownerBlock, ownerExpert; // kNoSlot when free
  };
  struct Block final {
    uint32_t size = 0;
    gguf::StreamedExperts experts;
    std::vector<uint32_t> slot; // each expert's, kNoSlot when not resident
    // Its router and shared-expert gate (bind), which predict its routes.
    ops::QuantizedSegment router, scalarGate;
  };
  // A step's tiles (ops::MoeLoadPart): the grouping's descriptors and count.
  struct PartTiles final {
    metal::MetalBuffer descriptors, count;
  };

  Impl(metal::MetalBackend &backend, ops::MoeShape shape, Geometry geometry, std::shared_ptr<WeightReaders> readers)
      : backend(backend), shape(shape), geometry(std::move(geometry)), readers(std::move(readers)) {}
  // The readers of a read-ahead read into this store's memory: they end the
  // pieces they took before it goes.
  ~Impl() {
    if (ahead.batch) readers->stop(*ahead.batch);
  }

  // Round `round` of the `rounds` of block `block`'s load (addLoad) of `rows`
  // rows in the tiles of `parts`; `predicted` when the step predicted the
  // next block's routes (predictionRoutes and predictionWeights).
  void load(uint32_t block, ops::MoePhase phase, uint32_t rows, bool predicted, std::span<const PartTiles> parts,
            uint32_t round, uint32_t rounds);
  void route(uint32_t block, std::span<const std::span<const MoeTileDescriptor>> parts, ops::MoePhase phase,
             uint32_t rows, ExpertStoreStats::Use &use);
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
  // The places a round of a load of `phase` that may route `routable`
  // experts stages: every place for a decode step's that they hold, else
  // half of them.
  [[nodiscard]] uint32_t roundPlaces(ops::MoePhase phase, uint64_t routable) const noexcept {
    return phase == ops::MoePhase::Decode && routable <= places.count ? places.count : places.count / 2;
  }
  [[nodiscard]] bool streams(uint32_t block) const noexcept { return sizes[blocks[block].size].streams; }
  // Adds to `reads` a group of the experts from `expert` on at the places
  // from `place` on of `staged`, `count` of them, read in pieces of at most
  // pieceBytes (whole pages, or whole rows for a built run), and returns it.
  const Group &addGroup(Reads &reads, uint32_t block, uint32_t expert, uint32_t count, uint32_t place,
                        const Places &staged, uint64_t pieceBytes) const;
  // Where expert k of `group` starts in the staging, its projection z.
  [[nodiscard]] uint64_t expertAt(const Group &group, uint32_t k, uint32_t z) const noexcept {
    const Run &run = group.runs[z];
    return run.at + run.inPage + k * sizes[blocks[group.block].size].layout[z].nativeBytes();
  }
  void readPiece(const Group &group, const Piece &piece) const;
  void stage(uint32_t block, uint32_t capacity, ExpertStoreStats::Use &use);
  std::span<const uint32_t> predictedExperts(uint32_t rows);
  void planAhead(uint32_t block, std::span<const uint32_t> experts, ExpertStoreStats::Use &use);
  void startAhead();
  // The host step after a long chunk's load: block `block`'s experts that
  // are not resident read ahead into the whole staging.
  void readBlockAhead(uint32_t block, ops::MoePhase phase);
  void countAhead(size_t group, size_t from, size_t to);
  void retireAhead();
  void evict(Size &size, uint32_t slot);
  void abandon();
  void fill();
  void publish();

  metal::MetalBackend &backend;
  const ops::MoeShape shape;
  const Geometry geometry;
  const std::shared_ptr<WeightReaders> readers;
  // What holds the sources open, and their checks (StreamedBlocks).
  std::shared_ptr<const void> files;
  std::vector<std::function<void()>> checks;
  std::vector<Size> sizes;
  std::vector<Block> blocks;
  // The staging's places, and which of them the round being staged holds.
  Places places;
  std::vector<uint8_t> taken;
  // Slot tables (blocks x experts), the staging, a round's jobs' slots and
  // their projections' sources (MoE::addExpertLoad's), and its grid
  // (MTLDispatchThreadgroupsIndirectArguments), all Shared.
  metal::MetalBuffer tables, staging, jobs, sources, grid;
  // A decode step's prediction of the next block's routes (addLoad): its
  // router's fp32 scores of this block's input, then each row's experts and
  // routing weights as the routing selects them (ops::MoE::addRoutes), for
  // up to a cycle's rows; each expert's summed weight over the rows (negative
  // when none) orders the experts it names (predictedExperts).
  metal::MetalBuffer predictionScores, predictionRoutes, predictionWeights;
  std::vector<float> predictionSum;
  std::vector<uint32_t> everyExpert, predictionOrder;
  // A read-ahead (startAhead) of block `block`'s experts, one a group, on the
  // readers (`batch`), until the next round halts it (stage takes what it
  // read) and the next read-ahead retires it (retireAhead); its use counts
  // each group once (`counted`). And the read-ahead a step plans
  // (planAhead), which its `then` starts: block `block`'s experts, most
  // likely first, counted in `use`.
  struct Ahead final {
    uint32_t block = kNoSlot;
    Reads reads;
    std::shared_ptr<WeightReaders::Batch> batch;
    ExpertStoreStats::Use *use = nullptr;
    std::vector<uint8_t> counted;
  } ahead;
  struct Plan final {
    uint32_t block = kNoSlot;
    std::vector<uint32_t> experts;
    ExpertStoreStats::Use *use = nullptr;
  } plan;
  uint64_t arenaTotal = 0;

  // A load and release() or restore(), which never overlap (no command is in
  // flight while the store releases or restores its memory), hand the state
  // below between their threads under `mutex`. stats() reads `published`, a
  // copy of the counters each of them leaves under statsMutex, and released()
  // the engine thread's own flag: neither waits for a load's reads.
  mutable std::mutex mutex;
  uint64_t serial = 0;
  // The routed experts of block pendingBlock's load that were not resident
  // and are not staged yet (`waiting`), in expert order, with the slots they
  // claimed, and those the round being staged stages: their slots hold no
  // expert until its repack ran.
  std::vector<std::pair<uint32_t, uint32_t>> pending, staged;
  std::vector<uint8_t> waiting;
  uint32_t pendingBlock = kNoSlot;
  // A round's reads.
  Reads now;
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

// Routes block `index`'s load: the experts its tiles name, each once; those
// resident keep their slots, and the others claim one each, in expert order
// (stage reads consecutive experts together).
void ExpertStore::Impl::route(uint32_t index, std::span<const std::span<const MoeTileDescriptor>> parts,
                              ops::MoePhase phase, uint32_t rows, ExpertStoreStats::Use &use) {
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
      ++use.routed;
      if (const uint32_t slot = block.slot[expert]; slot != kNoSlot) {
        size.recycling.touch(slot, serial, chunk);
        continue;
      }
      pending.push_back({expert, kNoSlot});
    }
  // Consecutive experts read together (stage).
  std::ranges::sort(pending);
  uint32_t *table = static_cast<uint32_t *>(tables.contents()) + uint64_t{index} * shape.experts;
  for (auto &[expert, slot] : pending) {
    slot = size.recycling.claim(uint64_t{index} << 32 | expert, serial, chunk);
    evict(size, slot);
    size.ownerBlock[slot] = index;
    size.ownerExpert[slot] = expert;
    block.slot[expert] = slot;
    table[expert] = slot;
    waiting[expert] = 1;
    ++counters.resident;
  }
  use.read += pending.size();
}

const ExpertStore::Impl::Group &ExpertStore::Impl::addGroup(Reads &reads, uint32_t index, uint32_t expert,
                                                            uint32_t count, uint32_t place, const Places &staged,
                                                            uint64_t pieceBytes) const {
  const Block &block = blocks[index];
  Group &group = reads.groups.emplace_back(Group{index, expert, count, place, {}});
  for (uint32_t z = 0; z < 3; ++z) {
    const ProjectionLayout &projection = sizes[block.size].layout[z];
    const uint64_t bytes = count * projection.nativeBytes();
    Run &run = group.runs[z];
    run.at = staged.at(place, z);
    run.bytes = bytes;
    uint64_t piece = pieceBytes / projection.rowBytes * projection.rowBytes;
    if (const auto stored = storedRepackRows(block.experts.projections[z], uint64_t{expert} * projection.rows,
                                             uint64_t{count} * projection.rows)) {
      const uint64_t start = stored->offset / kPageBytes * kPageBytes;
      run.file = stored->file;
      run.offset = start;
      run.inPage = stored->offset - start;
      run.bytes = std::min(alignUp(stored->offset + bytes, kPageBytes), stored->file->bytes()) - start;
      piece = pieceBytes;
    }
    piece = std::max(piece, projection.rowBytes);
    for (uint64_t from = 0; from < run.bytes; from += piece)
      reads.pieces.push_back(
          {static_cast<uint32_t>(reads.groups.size() - 1), z, from, std::min(from + piece, run.bytes)});
  }
  return group;
}

void ExpertStore::Impl::readPiece(const Group &group, const Piece &piece) const {
  const Run &run = group.runs[piece.z];
  const std::span<uint8_t> to{static_cast<uint8_t *>(staging.contents()) + run.at + piece.from, piece.to - piece.from};
  if (run.file) return run.file->readFile(run.offset + piece.from, to);
  const ProjectionLayout &projection = sizes[blocks[group.block].size].layout[piece.z];
  readRepackRows(blocks[group.block].experts.projections[piece.z],
                 uint64_t{group.expert} * projection.rows + piece.from / projection.rowBytes,
                 to.size() / projection.rowBytes, to.data());
}

// Stages a round of block `index`'s load: first the experts it routes that
// the read-ahead began, where they are, reading what the readers had not or
// failed to; then the others waiting, in expert order, in the lowest places
// left, up to `capacity` in all, each projection's runs of consecutive
// experts in consecutive places in reads of at most kLoadStepBytes. It
// writes their repack's jobs and grid.
void ExpertStore::Impl::stage(uint32_t index, uint32_t capacity, ExpertStoreStats::Use &use) {
  std::ranges::fill(taken, uint8_t{0});
  now.groups.clear();
  now.pieces.clear();
  auto *jobSlots = static_cast<uint32_t *>(jobs.contents());
  auto *jobSources = static_cast<uint64_t *>(sources.contents());
  const auto addJob = [&](const Group &group, uint32_t k) {
    const uint32_t expert = group.expert + k;
    const size_t job = staged.size();
    jobSlots[job] = blocks[index].slot[expert];
    for (uint32_t z = 0; z < 3; ++z) jobSources[3 * job + z] = expertAt(group, k, z);
    staged.push_back({expert, jobSlots[job]});
    waiting[expert] = 0;
  };
  // A read-ahead's group is one expert, and its pieces follow one another.
  // Of the pieces its readers still read (it is halted), the round waits for
  // those of an expert it stages, a few hundred us at most, and keeps the
  // places of the others.
  const std::vector<Piece> &aheadPieces = ahead.reads.pieces;
  const WeightReaders::Batch *batch = ahead.batch.get();
  const auto reading = [&](size_t piece) { return piece < batch->taken() && !batch->ended(piece); };
  for (size_t piece = 0, end = 0; batch && piece < aheadPieces.size(); piece = end) {
    const uint32_t at = aheadPieces[piece].group;
    const Group &group = ahead.reads.groups[at];
    while (end < aheadPieces.size() && aheadPieces[end].group == at) ++end;
    const bool routed = ahead.block == index && waiting[group.expert];
    bool ended = true;
    for (size_t rest = piece; rest < end; ++rest) {
      while (routed && reading(rest)) std::this_thread::yield();
      ended = ended && !reading(rest);
    }
    if (!ended) {
      taken[group.place] = 1;
      continue;
    }
    countAhead(at, piece, end);
    if (!routed) continue;
    taken[group.place] = 1;
    now.groups.push_back(group);
    bool whole = true;
    for (size_t rest = piece; rest < end; ++rest)
      if (rest >= batch->taken() || batch->failed(rest)) {
        const Piece &unread = aheadPieces[rest];
        now.pieces.push_back({static_cast<uint32_t>(now.groups.size() - 1), unread.z, unread.from, unread.to});
        whole = false;
      }
    use.prefetched += whole;
    addJob(now.groups.back(), 0);
  }
  const uint64_t left = std::ranges::count_if(pending, [&](const auto &entry) { return waiting[entry.first]; });
  const uint64_t due = std::max<uint64_t>(staged.size(), std::min<uint64_t>(capacity, staged.size() + left));
  uint32_t place = 0;
  for (size_t at = 0; at < pending.size() && staged.size() < due;) {
    if (!waiting[pending[at].first]) {
      ++at;
      continue;
    }
    while (place < places.count && taken[place]) ++place;
    if (place == places.count) throw std::logic_error("an expert load's round has fewer places than experts due");
    // Consecutive experts waiting, in consecutive free places, read together.
    uint32_t count = 1;
    while (at + count < pending.size() && staged.size() + count < due &&
           pending[at + count].first == pending[at].first + count && waiting[pending[at + count].first] &&
           place + count < places.count && !taken[place + count])
      ++count;
    const Group &group = addGroup(now, index, pending[at].first, count, place, places, kLoadStepBytes);
    for (uint32_t k = 0; k < count; ++k) {
      taken[place + k] = 1;
      addJob(group, k);
    }
    at += count;
  }
  std::atomic<uint64_t> bytes = 0;
  readers->run(now.pieces.size(), [&](size_t task) {
    const Piece &piece = now.pieces[task];
    readPiece(now.groups[piece.group], piece);
    bytes += piece.to - piece.from;
  });
  use.readBytes += bytes;
  // What the load read is the sources' as they were opened.
  if (!staged.empty())
    for (const std::function<void()> &check : checks) check();
  std::erase_if(pending, [&](const auto &entry) { return !waiting[entry.first]; });
  auto *threadgroups = static_cast<uint32_t *>(grid.contents());
  threadgroups[0] = sizes[blocks[index].size].loadGroups;
  threadgroups[1] = static_cast<uint32_t>(staged.size());
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

// Plans a read-ahead of block `index`'s experts `experts` (most likely
// first), which the step's `then` starts (startAhead).
void ExpertStore::Impl::planAhead(uint32_t index, std::span<const uint32_t> experts, ExpertStoreStats::Use &use) {
  plan.block = index;
  plan.experts.assign(experts.begin(), experts.end());
  plan.use = &use;
}

// Once the GPU went on (readers that start reading as the step ends hold its
// release back: 35B decode on an M6, 30-60 us a step with four of them),
// retires the last read-ahead and reads the planned experts ahead, in that
// order, into the places the round just staged left free, as many as there
// are. A release forgets the plan. It runs after the step let the GPU go on,
// where a throw would outlive the command, so it does not throw: an error
// becomes the store's failure, which the next load reports.
void ExpertStore::Impl::startAhead() {
  std::lock_guard lock(mutex);
  if (plan.block == kNoSlot) return;
  try {
    retireAhead();
    uint32_t place = 0;
    for (const uint32_t expert : plan.experts) {
      while (place < places.count && taken[place]) ++place;
      if (place == places.count) break;
      // Expert after expert, so that the experts a round finds read whole
      // are the likeliest; a piece starts on a page and so ends on one.
      addGroup(ahead.reads, plan.block, expert, 1, place++, places, kAheadPieceBytes);
    }
    const uint32_t block = std::exchange(plan.block, kNoSlot);
    if (ahead.reads.groups.empty()) return;
    ahead.block = block;
    ahead.use = plan.use;
    ahead.counted.assign(ahead.reads.groups.size(), 0);
    ahead.batch = readers->start(
        ahead.reads.pieces.size(),
        [this](size_t task) {
          const Piece &piece = ahead.reads.pieces[task];
          readPiece(ahead.reads.groups[piece.group], piece);
        },
        kAheadReaders);
  } catch (const std::exception &error) {
    plan.block = kNoSlot;
    if (failure.empty()) failure = std::string("reading experts ahead failed: ") + error.what();
    publish();
  }
}

// Counts group `group` of the read-ahead, its pieces [from, to), which
// ended or were never taken, once: the bytes its readers read, and its
// expert if they read all of it.
void ExpertStore::Impl::countAhead(size_t group, size_t from, size_t to) {
  if (ahead.counted[group]) return;
  ahead.counted[group] = 1;
  const WeightReaders::Batch &batch = *ahead.batch;
  bool whole = true;
  for (size_t p = from; p < to; ++p) {
    const Piece &piece = ahead.reads.pieces[p];
    if (p < batch.taken() && !batch.failed(p)) ahead.use->readBytes += piece.to - piece.from;
    else whole = false;
  }
  ahead.use->readAhead += whole;
}

// Ends the read-ahead once its readers ended the pieces they took; what it
// read that no round counted counts in its use.
void ExpertStore::Impl::retireAhead() {
  if (ahead.batch) {
    readers->stop(*ahead.batch);
    const std::vector<Piece> &pieces = ahead.reads.pieces;
    for (size_t piece = 0, end = 0; piece < pieces.size(); piece = end) {
      while (end < pieces.size() && pieces[end].group == pieces[piece].group) ++end;
      countAhead(pieces[piece].group, piece, end);
    }
    ahead.batch.reset();
  }
  ahead.reads.groups.clear();
  ahead.reads.pieces.clear();
  ahead.block = kNoSlot;
}

// Frees the slots of the experts a failed load had not repacked: those
// waiting and those its last round staged.
void ExpertStore::Impl::abandon() {
  if (pendingBlock == kNoSlot) return;
  Block &block = blocks[pendingBlock];
  Size &size = sizes[block.size];
  uint32_t *table = static_cast<uint32_t *>(tables.contents()) + uint64_t{pendingBlock} * shape.experts;
  for (const std::vector<std::pair<uint32_t, uint32_t>> *list : {&pending, &staged})
    for (const auto &[expert, slot] : *list) {
      waiting[expert] = 0;
      if (slot == kNoSlot || block.slot[expert] != slot) continue;
      block.slot[expert] = kNoSlot;
      table[expert] = 0;
      size.ownerBlock[slot] = size.ownerExpert[slot] = kNoSlot;
      size.recycling.free(slot);
      --counters.resident;
    }
  pending.clear();
  staged.clear();
  pendingBlock = kNoSlot;
}

// Reads every expert of each size of one block into its slot, expert e into
// slot e, through the staging, as many a command as it places.
void ExpertStore::Impl::fill() {
  for (uint32_t index = 0; index < blocks.size(); ++index) {
    if (streams(index)) continue;
    Block &block = blocks[index];
    Size &size = sizes[block.size];
    std::array<uint64_t, 3> nativeBytes;
    for (size_t z = 0; z < 3; ++z) nativeBytes[z] = size.layout[z].nativeBytes();
    const uint64_t count = std::min<uint64_t>(shape.experts, staging.sizeBytes() / Places(nativeBytes, 1).bytes());
    if (!count) throw std::invalid_argument("the expert staging holds less than one expert");
    const Places batch(nativeBytes, static_cast<uint32_t>(count));
    uint32_t *table = static_cast<uint32_t *>(tables.contents()) + uint64_t{index} * shape.experts;
    for (uint32_t expert = 0; expert < shape.experts; expert += batch.count) {
      const uint32_t experts = std::min(batch.count, shape.experts - expert);
      now.groups.clear();
      now.pieces.clear();
      const Group &group = addGroup(now, index, expert, experts, 0, batch, kLoadStepBytes);
      readers->run(now.pieces.size(), [&](size_t task) { readPiece(group, now.pieces[task]); });
      for (const std::function<void()> &check : checks) check();
      auto *jobSlots = static_cast<uint32_t *>(jobs.contents());
      auto *jobSources = static_cast<uint64_t *>(sources.contents());
      for (uint32_t k = 0; k < experts; ++k) {
        const uint32_t slot = expert + k;
        jobSlots[k] = slot;
        for (uint32_t z = 0; z < 3; ++z) jobSources[3 * k + z] = expertAt(group, k, z);
        size.ownerBlock[slot] = index;
        size.ownerExpert[slot] = slot;
        block.slot[slot] = slot;
        table[slot] = slot;
      }
      auto *threadgroups = static_cast<uint32_t *>(grid.contents());
      threadgroups[0] = size.loadGroups;
      threadgroups[1] = experts;
      threadgroups[2] = 3;
      metal::CommandGraph graph;
      ops::MoE::addExpertLoad(graph, staging, size.arena, jobs, sources, size.load, experts,
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
                             std::span<const PartTiles> parts, uint32_t round, uint32_t rounds) {
  std::lock_guard lock(mutex);
  // No job runs until the round's are staged; the last round's ran.
  static_cast<uint32_t *>(grid.contents())[1] = 0;
  staged.clear();
  if (!failure.empty()) throw std::runtime_error("expert store failed earlier: " + failure);
  ExpertStoreStats::Use &use = counters.uses[static_cast<size_t>(phase)];
  try {
    if (round == 0) {
      // A load whose later rounds did not run (its command failed) frees the
      // slots of the experts it had not staged; one that fails before it
      // routes frees none of the last load's (abandon).
      abandon();
      partTiles.clear();
      for (const PartTiles &part : parts) {
        const auto *counted = static_cast<const MoeTileCount *>(part.count.contents());
        const auto *descriptors = static_cast<const MoeTileDescriptor *>(part.descriptors.contents());
        const uint64_t bound = part.descriptors.sizeBytes() / sizeof(MoeTileDescriptor);
        if (!counted || !descriptors || counted->tiles > bound)
          throw std::runtime_error("a MoE block's tiles are out of bounds");
        partTiles.push_back({descriptors, counted->tiles});
      }
      route(index, partTiles, phase, rows, use);
      ++use.loads;
    } else if (pendingBlock != index) {
      throw std::logic_error("an expert load's round follows another block's load");
    } else if (pending.empty()) {
      // An earlier round staged every expert, and the read-ahead of the next
      // block runs on.
      return;
    }
    // The read-ahead halts: its readers take no more pieces.
    if (ahead.batch) readers->halt(*ahead.batch);
    stage(index, roundPlaces(phase, routable(rows)), use);
    // Once the round is staged, what comes next is read ahead while the GPU
    // repacks it: the load's experts still waiting, else the next block's
    // likely experts.
    ahead.block = kNoSlot;
    if (!pending.empty()) {
      if (round + 1 == rounds) throw std::logic_error("an expert load routes more experts than its rounds stage");
      std::vector<uint32_t> waitingExperts;
      for (const auto &entry : pending) waitingExperts.push_back(entry.first);
      planAhead(index, waitingExperts, use);
      return;
    }
    // A long chunk's next block is read ahead by the host step after its
    // last repack (readBlockAhead).
    if (predicted) {
      std::vector<uint32_t> experts;
      for (const uint32_t expert : predictedExperts(rows))
        if (blocks[index + 1].slot[expert] == kNoSlot) experts.push_back(expert);
      planAhead(index + 1, experts, use);
    }
  } catch (const std::exception &error) {
    abandon();
    failure = error.what();
    publish();
    throw;
  }
  publish();
}

void ExpertStore::Impl::readBlockAhead(uint32_t index, ops::MoePhase phase) {
  std::lock_guard lock(mutex);
  if (!failure.empty()) throw std::runtime_error("expert store failed earlier: " + failure);
  // The GPU repacked the load's last round: no place is taken.
  std::ranges::fill(taken, uint8_t{0});
  std::vector<uint32_t> experts;
  for (const uint32_t expert : everyExpert)
    if (blocks[index].slot[expert] == kNoSlot) experts.push_back(expert);
  planAhead(index, experts, counters.uses[static_cast<size_t>(phase)]);
}

ExpertStore::ExpertStore(metal::MetalBackend &backend, StreamedBlocks streamed, ops::MoeShape shape,
                         ExpertStoreOptions options)
    : impl_(std::make_shared<Impl>(backend, shape, geometryOf(streamed, shape), std::move(options.readers))) {
  Impl &impl = *impl_;
  const Geometry &geometry = impl.geometry;
  if (!options.memoryBytes || !impl.readers) throw std::invalid_argument("invalid expert store");
  impl.files = std::move(streamed.sources);
  impl.checks = std::move(streamed.checks);
  for (uint32_t block = 0; block < streamed.blocks.size(); ++block)
    impl.blocks.push_back(
        {.experts = std::move(streamed.blocks[block]), .slot = std::vector<uint32_t>(shape.experts, kNoSlot)});
  const uint64_t fixed = fixedBytesOf(geometry, static_cast<uint32_t>(impl.blocks.size()), shape);
  const uint64_t arenaBudget = options.memoryBytes > fixed ? options.memoryBytes - fixed : 0;
  const uint64_t largest = options.bufferBytes && options.bufferBytes < backend.capabilities().maxBufferLengthBytes
                               ? options.bufferBytes
                               : backend.capabilities().maxBufferLengthBytes;
  // Every streamed size holds the same share of its experts, but at least a
  // block's (a prefill chunk may route them all) and at most all of them; a
  // size of one block all of its: the largest share the arena holds.
  const auto slotsAt = [&](uint32_t size, double share) {
    const uint64_t all = uint64_t{impl.sizes[size].blocks} * shape.experts;
    if (!impl.sizes[size].streams) return all;
    return std::clamp<uint64_t>(static_cast<uint64_t>(share * double(all)), shape.experts, all);
  };
  const auto bytesAt = [&](double share) {
    uint64_t bytes = 0;
    for (uint32_t size = 0; size < impl.sizes.size(); ++size)
      bytes += arenaBytes(impl.sizes[size].layout, slotsAt(size, share));
    return bytes;
  };
  // Each layout's blocks in parts[layout] sizes of consecutive blocks: one,
  // and one more while a size's arena takes more than the largest buffer.
  std::vector<uint32_t> parts(geometry.layouts.size(), 1), sizeLayout;
  double low = 0.0;
  for (;;) {
    impl.sizes.clear();
    sizeLayout.clear();
    for (uint32_t layout = 0; layout < geometry.layouts.size(); ++layout) {
      std::vector<uint32_t> members;
      for (uint32_t block = 0; block < impl.blocks.size(); ++block)
        if (geometry.layoutOf[block] == layout) members.push_back(block);
      for (uint32_t part = 0; part < parts[layout]; ++part) {
        const size_t from = members.size() * part / parts[layout], to = members.size() * (part + 1) / parts[layout];
        for (size_t member = from; member < to; ++member)
          impl.blocks[members[member]].size = static_cast<uint32_t>(impl.sizes.size());
        impl.sizes.push_back({.layout = geometry.layouts[layout],
                              .blocks = static_cast<uint32_t>(to - from),
                              .streams = geometry.streams(layout)});
        sizeLayout.push_back(layout);
      }
    }
    if (bytesAt(0.0) > arenaBudget)
      throw std::invalid_argument("expert memory of " + std::to_string(options.memoryBytes >> 20) +
                                  " MiB holds fewer slots (" + std::to_string(bytesAt(0.0) >> 20) +
                                  " MiB) than a block's experts of each size and arena and every expert of a size "
                                  "of one block, beside the staging and tables (" +
                                  std::to_string(fixed >> 20) + " MiB)");
    double high = 1.0;
    low = 0.0;
    if (bytesAt(high) <= arenaBudget) low = high;
    for (int step = 0; step < 64 && low < high; ++step) {
      const double middle = (low + high) / 2;
      (bytesAt(middle) <= arenaBudget ? low : high) = middle;
    }
    uint32_t over = 0;
    while (over < impl.sizes.size() && arenaBytes(impl.sizes[over].layout, slotsAt(over, low)) <= largest) ++over;
    if (over == impl.sizes.size()) break;
    if (impl.sizes[over].blocks == 1)
      throw std::invalid_argument("a block's experts take more than the device's largest buffer");
    ++parts[sizeLayout[over]];
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
    size.recycling = ExpertSlots(size.slots);
    size.ownerBlock.assign(size.slots, kNoSlot);
    size.ownerExpert.assign(size.slots, kNoSlot);
  }
  impl.places = stagingOf(geometry, shape);
  impl.taken.assign(impl.places.count, 0);
  impl.staging = backend.allocateBuffer(impl.places.bytes(), metal::BufferStorage::Shared, "experts/staging");
  const uint64_t tableBytes = uint64_t{shape.experts} * impl.blocks.size() * sizeof(uint32_t);
  impl.tables = backend.allocateBuffer(tableBytes, metal::BufferStorage::Shared, "experts/slot-tables");
  std::memset(impl.tables.contents(), 0, tableBytes);
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
  impl.waiting.assign(shape.experts, 0);
  impl.counters.arenaBytes = impl.arenaTotal;
  impl.counters.stagingBytes = impl.staging.sizeBytes();
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
  const uint64_t routable = impl.routable(rows);
  const uint64_t perRound = impl.roundPlaces(phase, routable);
  const uint32_t rounds = static_cast<uint32_t>((routable + perRound - 1) / perRound);
  // A decode step's load predicts the next block's routes, when its experts
  // stream: that block's routing of this block's input.
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
  const metal::HostStepGpu gpu = phase == ops::MoePhase::Decode ? metal::HostStepGpu::Awake : metal::HostStepGpu::Idle;
  const auto timed = [impl = impl_, phase](const metal::HostStepTiming &timing) {
    Impl::StepTimes &times = impl->stepTimes[static_cast<size_t>(phase)];
    const auto nanoseconds = [](double seconds) { return uint64_t(seconds * 1e9); };
    times.pickup.add(nanoseconds(timing.pickupSeconds));
    times.work.add(nanoseconds(timing.workSeconds));
    times.resume.add(nanoseconds(timing.resumeSeconds));
  };
  // Each step's read-ahead starts once the GPU went on (startAhead).
  const auto startAhead = [impl = impl_] { impl->startAhead(); };
  for (uint32_t round = 0; round < rounds; ++round) {
    graph.host(
        [impl = impl_, block, phase, rows, predicted, tiles, round, rounds] {
          impl->load(block, phase, rows, predicted, tiles, round, rounds);
        },
        timed, gpu, startAhead);
    ops::MoE::addExpertLoad(graph, impl.staging, size.arena, impl.jobs, impl.sources, size.load,
                            static_cast<uint32_t>(std::min<uint64_t>(routable, impl.places.count)),
                            metal::IndirectGrid{impl.grid, 0});
  }
  // A long chunk routes nearly every expert of the next block too: once the
  // GPU has repacked the last round, the next block's experts are read ahead
  // into the whole staging, while the GPU runs this block's experts and the
  // next block's mixing, as many as the next load would otherwise read
  // with the GPU waiting.
  if (impl.longChunk(phase, rows) && block + 1 < impl.blocks.size() && impl.streams(block + 1))
    graph.host([impl = impl_, block, phase] { impl->readBlockAhead(block + 1, phase); }, timed,
               metal::HostStepGpu::Idle, startAhead);
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
  impl.retireAhead();
  impl.ahead.block = impl.plan.block = kNoSlot;
  for (Impl::Block &block : impl.blocks) std::ranges::fill(block.slot, kNoSlot);
  std::memset(impl.tables.contents(), 0, impl.tables.sizeBytes());
  for (Impl::Size &size : impl.sizes) {
    impl.backend.releaseMemory(size.arena);
    std::ranges::fill(size.ownerBlock, kNoSlot);
    std::ranges::fill(size.ownerExpert, kNoSlot);
    size.recycling.reset();
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

metal::MetalBuffer residentExpertSlots(metal::MetalBackend &backend, uint32_t experts) {
  metal::MetalBuffer table =
      backend.allocateBuffer(uint64_t{experts} * sizeof(uint32_t), metal::BufferStorage::Shared,
                             "experts/resident-slots");
  auto *slot = static_cast<uint32_t *>(table.contents());
  for (uint32_t expert = 0; expert < experts; ++expert) slot[expert] = expert;
  return table;
}

} // namespace splash::model

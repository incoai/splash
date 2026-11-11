#include "model/ExpertStore.hpp"
#include "AwakeClock.hpp"
#include "Checked.hpp"
#include "StderrLine.hpp"

#include "metal/abi/MoE.h"
#include "model/ExpertChunkLayout.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/GgufPreparation.hpp"
#include "model/Model.hpp"
#include "model/SlotRecycler.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightSource.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
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
// is the pieces its readers took. Four readers of such pieces reach nearly
// the SSD's full bandwidth, and a piece takes about 280 us, a quarter of
// what it takes on eighteen.
constexpr uint64_t kAheadPieceBytes = 512 << 10;
constexpr unsigned kAheadReaders = 4;
// A decode cycle's most rows: every lane's verify rows.
constexpr uint32_t kCycleRows = ExecutionLimits::maximumBatchWidth * ExecutionLimits::targetVerifyRows;
// A prefill chunk is long when its rows route at least this many times as
// many experts as a block has: routed evenly, that many routes leave about
// e^-3, 5%, of the experts unrouted, so it routes nearly every expert.
constexpr uint64_t kLongChunkRoutes = 3;

// ExpertStoreStats::uses and Impl::stepTimes hold a use per ops::MoePhase,
// at its value.
static_assert(static_cast<size_t>(ops::MoePhase::Prefill) == 0 && static_cast<size_t>(ops::MoePhase::Decode) == 1);

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

// The bytes of a chunk of `slots` experts of a layout: each projection's
// planes in a region of their own. A size's chunks take the bytes of one
// chunk of all its slots (ExpertChunkLayout).
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
      width[z] = alignUp(nativeBytes[z]) + kHostPageBytes;
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
  return stagingOf(g, shape).bytes() + 2 * uint64_t{blocks} * shape.experts * sizeof(uint32_t) +
         uint64_t{shape.experts} * (sizeof(uint32_t) + 3 * sizeof(uint64_t)) + 4 * sizeof(uint32_t) + prediction;
}

// The smallest store (ExpertStore::minimumBytes): its fixed bytes and the
// chunks of each layout of a block's experts, which for a layout of one block
// are all of its experts.
uint64_t minimumBytesOf(const Geometry &g, uint32_t blocks, ops::MoeShape shape) {
  uint64_t bytes = fixedBytesOf(g, blocks, shape);
  for (const ExpertLayout &layout : g.layouts)
    bytes += arenaBytes(layout, shape.experts);
  return bytes;
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
  // The experts of one layout: their slots, a streamed size's recycled in
  // its order (SlotRecycler), in its floor chunk and elastic chunks
  // (streamedExpertChunkLayout), or a size of one block's in its floor chunk
  // alone (wholeExpertChunkLayout), each a buffer of the layout's regions
  // over its own slots. The first chunks.liveChunks() of them are live and
  // the recycler holds their slots; the others are released.
  struct Size final {
    ExpertLayout layout;
    uint32_t blocks = 0;
    // Whether its layout is of several blocks (Geometry::streams).
    bool streams = false;
    // Its slots at the cap, which its blocks' segments, owners and slot
    // tables count, and where its chunks hold them.
    uint32_t slots = 0;
    ExpertChunkLayout chunkLayout;
    // Its blocks' routed segments, of its slots' format and size and no
    // planes (bind), the chunks' buffers, and their regions as the expert
    // kernels reach them, checked for those segments, which its blocks share.
    std::array<ops::QuantizedSegment, 3> routed;
    std::vector<metal::MetalBuffer> buffers;
    ops::ExpertChunks chunks;
    // The repack of its staged rows into slots.
    MoeExpertLoadParams load{};
    uint32_t loadGroups = 0;
    // Which slot a load's missing expert takes.
    SlotRecycler recycling{0};
    std::vector<uint32_t> ownerBlock, ownerExpert; // kNoSlot when free

    // Its slots above its floor, as many as its live chunks hold, or `live`.
    [[nodiscard]] uint32_t aboveFloor() const noexcept { return aboveFloor(chunks.liveChunks()); }
    [[nodiscard]] uint32_t aboveFloor(uint32_t live) const noexcept {
      return chunkLayout.first(live) - chunkLayout.first(chunkLayout.floorChunks);
    }
  };
  struct Block final {
    uint32_t size = 0;
    gguf::StreamedExperts experts;
    std::vector<uint32_t> slot; // each expert's, kNoSlot when not resident
    uint32_t resident = 0;      // the experts that have a slot
    // Its router and shared-expert gate (bind), which predict its routes.
    ops::QuantizedSegment router, scalarGate;
  };
  // A step's tiles (ops::MoeLoadPart): the grouping's descriptors and count.
  struct PartTiles final {
    metal::MetalBuffer descriptors, count;
  };

  Impl(metal::MetalBackend &backend, ops::MoeShape shape, Geometry geometry, std::shared_ptr<SsdReaders> readers)
      : backend(backend), shape(shape), geometry(std::move(geometry)), readers(std::move(readers)) {}
  // The readers of a read-ahead read into this store's memory: they end the
  // pieces they took before it goes.
  ~Impl() {
    if (ahead.batch) readers->stop(*ahead.batch);
  }

  // Round `round` of the `rounds` of block `block`'s load (addLoad) of `rows`
  // rows in the tiles of `parts`; `predicted` when the step predicted the
  // next block's routes (predictionRoutes and predictionWeights), from the
  // rows `verifyRows` names when the step routes its verify rows alone
  // (ops::MoeBuffers::verifyRows).
  void load(uint32_t block, ops::MoePhase phase, uint32_t rows, bool predicted, std::span<const PartTiles> parts,
            uint32_t round, uint32_t rounds, const metal::MetalBuffer &verifyRows);
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
  std::span<const uint32_t> predictedExperts(uint32_t rows, const metal::MetalBuffer &verifyRows);
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
  // The streamed size whose last live elastic chunk goes next, the one with
  // the largest share of its slots above its floor, and the size whose next
  // elastic chunk comes back next, the one with the smallest; kNoSlot when
  // none has such a chunk; the first of them also at `live` chunks of each
  // size (publish: the chunks in the order they go).
  [[nodiscard]] uint32_t shrinkingSize() const;
  [[nodiscard]] uint32_t shrinkingSize(std::span<const uint32_t> live) const noexcept;
  // Whether the store may give chunks back now: it holds its memory, no
  // command is in flight, no load failed, and no routed expert waits to be
  // staged. A command that failed after a load's last round staged leaves
  // nothing the store knows of, but a failed command ends the engine before
  // any shrink (ExpertStore.hpp): these guards are a defence.
  [[nodiscard]] bool mayShrink() const {
    return !released && failure.empty() && pending.empty() && !backend.commandInFlight();
  }
  [[nodiscard]] uint32_t growingSize() const noexcept;
  // Gives size `index`'s live chunks from `live` on back (releaseAboveFloor,
  // releaseToFloor); returns their bytes.
  uint64_t shrink(uint32_t index, uint32_t live);
  // Takes size `index`'s next chunk back under `admit`, its slots free.
  [[nodiscard]] metal::AllocationResult grow(uint32_t index, const metal::AllocationAdmission &admit);
  // The expert in slot `from` of `size` moves to its free slot `to`: its
  // planes, its owner, and its block's slot and table entry.
  void move(Size &size, uint32_t from, uint32_t to);
  // Before size `index`'s slots from `boundary` on are given back: no block
  // of it names them, every table entry of its blocks is the block's slot of
  // the expert (MOE_SLOT_ABSENT for one not resident), no slot from it on
  // has an owner, and its recycler holds the slots below it alone. Throws
  // std::logic_error otherwise.
  void requireBelow(uint32_t index, uint32_t boundary) const;
  // No repack names a slot: a load's host step that does not run (one
  // before it threw, or the command before it failed) leaves the repack
  // after it no job, rather than the last round's, whose slots a shrink may
  // have moved or given back; no slot is staged, as no load is under way;
  // and every load table entry names slot 0, inside every size's floor.
  void forgetRepack();
  // The live chunks' count, bytes and slots in the counters, none while the
  // store is released.
  void countLive();
  void publish();

  metal::MetalBackend &backend;
  const ops::MoeShape shape;
  const Geometry geometry;
  const std::shared_ptr<SsdReaders> readers;
  // What holds the sources open, and their checks (StreamedBlocks).
  std::shared_ptr<const void> files;
  std::vector<std::function<void()>> checks;
  std::vector<Size> sizes;
  std::vector<Block> blocks;
  // The staging's places, and which of them the round being staged holds.
  Places places;
  std::vector<uint8_t> taken;
  // Slot tables (blocks x experts), their copies for the passes after a
  // load (ops::BlockMoeWeights::loadSlotTable), the staging, a round's jobs'
  // slots and their projections' sources (MoE::addExpertLoad's), and its
  // grid (MTLDispatchThreadgroupsIndirectArguments), all Shared.
  metal::MetalBuffer tables, loadTables, staging, jobs, sources, grid;
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
    std::shared_ptr<SsdReaders::Batch> batch;
    ExpertStoreStats::Use *use = nullptr;
    std::vector<uint8_t> counted;
  } ahead;
  struct Plan final {
    uint32_t block = kNoSlot;
    std::vector<uint32_t> experts;
    ExpertStoreStats::Use *use = nullptr;
  } plan;

  // A load and release(), restore(), releaseAboveFloor() or growTowardCap(),
  // which never overlap (no command is in flight while the store gives its
  // memory back or takes it again), hand the state below and the sizes'
  // slots between their threads under `mutex`. stats() reads `published`, a
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
  // The staging and tables (fixedBytes), which counters' bytes count.
  uint64_t fixedBytes = 0;
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
  --blocks[owner].resident;
  // A table entry of no resident expert is MOE_SLOT_ABSENT, which only the
  // grouping reads: it puts the expert's tiles past the resident ones, whose
  // passes run after the load that gives it a slot and read the load table.
  static_cast<uint32_t *>(tables.contents())[uint64_t{owner} * shape.experts + expert] = MOE_SLOT_ABSENT;
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
  // Every routed expert's slot, for the passes after the load.
  uint32_t *loadTable = static_cast<uint32_t *>(loadTables.contents()) + uint64_t{index} * shape.experts;
  for (std::span<const MoeTileDescriptor> tiles : parts)
    for (const MoeTileDescriptor &tile : tiles) {
      // The shared expert's tiles (ids experts on) route nothing.
      const uint32_t expert = tile.expert;
      if (expert >= shape.experts || routedIn[expert] == serial) continue;
      routedIn[expert] = serial;
      ++use.routed;
      if (const uint32_t slot = block.slot[expert]; slot != kNoSlot) {
        size.recycling.touch(slot, serial, chunk);
        loadTable[expert] = slot;
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
    table[expert] = loadTable[expert] = slot;
    waiting[expert] = 1;
    ++block.resident;
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
      // Widened to whole host pages: an uncached (F_NOCACHE) read skips the
      // buffer cache only when it starts and ends on them, in the file and in
      // memory. Through the cache, reads take about twice the CPU per GB.
      const uint64_t start = stored->offset / kHostPageBytes * kHostPageBytes;
      run.file = stored->file;
      run.offset = start;
      run.inPage = stored->offset - start;
      run.bytes = std::min(alignUp(stored->offset + bytes), stored->file->bytes()) - start;
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
  const SsdReaders::Batch *batch = ahead.batch.get();
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
// rows, or those of them below their lanes' verify rows when the step routes
// them alone, by their summed routing weights, the largest first.
std::span<const uint32_t> ExpertStore::Impl::predictedExperts(uint32_t rows, const metal::MetalBuffer &verifyRows) {
  const auto *selected = static_cast<const uint32_t *>(predictionRoutes.contents());
  const auto *weights = static_cast<const float *>(predictionWeights.contents());
  // The selector wrote them before the step's GPU work ran up to this load.
  const auto *verified = static_cast<const uint32_t *>(verifyRows.contents());
  const uint32_t routes = shape.routesPerToken();
  predictionOrder.clear();
  for (uint64_t row = 0; row < rows; ++row) {
    if (verified && row % SPLASH_TARGET_VERIFY_ROWS >= verified[row / SPLASH_TARGET_VERIFY_ROWS]) continue;
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
// release back, 30-60 us a decode step), retires the last read-ahead and
// reads the planned experts ahead, in that order, into the places the round
// just staged left free, as many as there are. A release forgets the plan.
// It runs after the step let the GPU go on, where a throw would outlive the
// command, so it does not throw: an error becomes the store's failure, which
// the next load reports.
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
  const SsdReaders::Batch &batch = *ahead.batch;
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
      table[expert] = MOE_SLOT_ABSENT;
      size.ownerBlock[slot] = size.ownerExpert[slot] = kNoSlot;
      size.recycling.free(slot);
      --block.resident;
      --counters.resident;
    }
  pending.clear();
  staged.clear();
  pendingBlock = kNoSlot;
}

// Reads every expert of each size of one block into its slot, expert e into
// slot e, through the staging, as many a command as it places; its last
// repack's jobs are forgotten once it ran.
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
      ops::MoE::addExpertLoad(graph, staging, size.chunks, jobs, sources, size.load, experts,
                              metal::IndirectGrid{grid, 0});
      static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    }
    block.resident = shape.experts;
    counters.resident += shape.experts;
  }
  forgetRepack();
}

uint32_t ExpertStore::Impl::shrinkingSize() const {
  std::vector<uint32_t> live;
  for (const Size &size : sizes) live.push_back(size.chunks.liveChunks());
  return shrinkingSize(live);
}

uint32_t ExpertStore::Impl::shrinkingSize(std::span<const uint32_t> live) const noexcept {
  uint32_t chosen = kNoSlot;
  for (uint32_t index = 0; index < sizes.size(); ++index) {
    const Size &size = sizes[index];
    if (live[index] <= size.chunkLayout.floorChunks) continue;
    if (chosen == kNoSlot || uint64_t{size.aboveFloor(live[index])} * sizes[chosen].slots >
                                 uint64_t{sizes[chosen].aboveFloor(live[chosen])} * size.slots)
      chosen = index;
  }
  return chosen;
}

uint32_t ExpertStore::Impl::growingSize() const noexcept {
  uint32_t chosen = kNoSlot;
  for (uint32_t index = 0; index < sizes.size(); ++index) {
    const Size &size = sizes[index];
    if (size.chunks.liveChunks() == size.chunks.chunks()) continue;
    if (chosen == kNoSlot ||
        uint64_t{size.aboveFloor()} * sizes[chosen].slots < uint64_t{sizes[chosen].aboveFloor()} * size.slots)
      chosen = index;
  }
  return chosen;
}

// The experts of the chunks that go that fit move below them, after the
// recycler evicted those its order recycles first until they fit
// (SlotRecycler::compact), in one compaction, so that each moves once; once
// the slot tables are checked, no address table names those chunks from the
// next command on, and their memory goes, the last first. The slots claimed
// by a load in flight could move under it, so no command is in flight, and
// every command before succeeded (ExpertStore.hpp): no load is unfinished.
// Logs a line with what moved and what the store holds after.
uint64_t ExpertStore::Impl::shrink(uint32_t index, uint32_t live) {
  const AwakeClock::time_point started = AwakeClock::now();
  Size &size = sizes[index];
  const uint32_t chunks = size.chunks.liveChunks();
  const uint32_t boundary = size.chunkLayout.first(live);
  uint32_t evicted = 0, moved = 0;
  size.recycling.compact(
      boundary,
      [&](uint32_t slot) {
        evict(size, slot);
        ++evicted;
      },
      [&](uint32_t from, uint32_t to) {
        move(size, from, to);
        ++moved;
      });
  requireBelow(index, boundary);
  forgetRepack();
  size.chunks.keep(live);
  uint64_t bytes = 0;
  for (uint32_t chunk = chunks; chunk-- > live;) {
    backend.releaseMemory(size.buffers[chunk]);
    bytes += size.buffers[chunk].sizeBytes();
  }
  counters.shrinks += chunks - live;
  countLive();
  const double took = millisecondsSince(started);
  publish();
  logLine("Expert store gave back ", chunks - live == 1 ? "a chunk" : std::to_string(chunks - live) + " chunks",
          " of ", bytes >> 20, " MiB while memory is short, in ", std::fixed, std::setprecision(1), took,
          " ms: ", moved, " experts moved below ", chunks - live == 1 ? "it" : "them", " and ", evicted,
          " evicted; ", counters.chunks, " of ", counters.capChunks, " chunks hold ", counters.slots, " slots.");
  return bytes;
}

metal::AllocationResult ExpertStore::Impl::grow(uint32_t index, const metal::AllocationAdmission &admit) {
  Size &size = sizes[index];
  const uint32_t chunk = size.chunks.liveChunks();
  const metal::MetalBuffer &buffer = size.buffers[chunk];
  if (const metal::AllocationResult admitted = admit(buffer.sizeBytes(), [&] { backend.restoreMemory(buffer); });
      !admitted)
    return admitted;
  size.recycling.resize(size.chunkLayout.first(chunk + 1));
  size.chunks.keep(chunk + 1);
  return {};
}

void ExpertStore::Impl::move(Size &size, uint32_t from, uint32_t to) {
  // A slot's planes are one run in each of its chunk's nine regions (plane1's
  // region is meta's for a format without one, which has no run of its own).
  const std::vector<metal::MetalBuffer> &regions = *size.chunks.planes();
  const uint32_t fromChunk = size.chunkLayout.chunkOf(from), toChunk = size.chunkLayout.chunkOf(to);
  const uint64_t fromLocal = from - size.chunkLayout.first(fromChunk), toLocal = to - size.chunkLayout.first(toChunk);
  for (uint32_t z = 0; z < 3; ++z)
    for (uint32_t plane = 0; plane < 3; ++plane) {
      const uint64_t bytes = size.layout[z].planes[plane];
      if (!bytes) continue;
      const uint64_t region = z * 3 + plane;
      std::memcpy(
          static_cast<uint8_t *>(regions[toChunk * MOE_EXPERT_CHUNK_ADDRESSES + region].contents()) + toLocal * bytes,
          static_cast<const uint8_t *>(regions[fromChunk * MOE_EXPERT_CHUNK_ADDRESSES + region].contents()) +
              fromLocal * bytes,
          bytes);
    }
  const uint32_t owner = std::exchange(size.ownerBlock[from], kNoSlot);
  const uint32_t expert = std::exchange(size.ownerExpert[from], kNoSlot);
  size.ownerBlock[to] = owner;
  size.ownerExpert[to] = expert;
  if (owner == kNoSlot) return;
  blocks[owner].slot[expert] = to;
  static_cast<uint32_t *>(tables.contents())[uint64_t{owner} * shape.experts + expert] = to;
}

void ExpertStore::Impl::requireBelow(uint32_t index, uint32_t boundary) const {
  const Size &size = sizes[index];
  const auto *table = static_cast<const uint32_t *>(tables.contents());
  for (uint32_t block = 0; block < blocks.size(); ++block) {
    if (blocks[block].size != index) continue;
    for (uint32_t expert = 0; expert < shape.experts; ++expert) {
      const uint32_t slot = blocks[block].slot[expert];
      if ((slot != kNoSlot && slot >= boundary) ||
          table[uint64_t{block} * shape.experts + expert] != (slot == kNoSlot ? MOE_SLOT_ABSENT : slot))
        throw std::logic_error("block " + std::to_string(block) + "'s expert " + std::to_string(expert) +
                               " is in slot " + std::to_string(slot) + " past the expert slots kept, " +
                               std::to_string(boundary) + ", or its slot table names another");
    }
  }
  for (uint32_t slot = boundary; slot < size.slots; ++slot)
    if (size.ownerBlock[slot] != kNoSlot)
      throw std::logic_error("expert slot " + std::to_string(slot) + " past the slots kept, " +
                             std::to_string(boundary) + ", holds an expert");
  if (size.recycling.size() != boundary)
    throw std::logic_error("an expert size's recycler holds " + std::to_string(size.recycling.size()) +
                           " slots, not the " + std::to_string(boundary) + " kept");
}

void ExpertStore::Impl::forgetRepack() {
  static_cast<uint32_t *>(grid.contents())[1] = 0;
  staged.clear();
  std::memset(loadTables.contents(), 0, loadTables.sizeBytes());
}

void ExpertStore::Impl::countLive() {
  counters.chunks = counters.arenaBytes = counters.slots = counters.liveBytes = 0;
  if (released) return;
  for (const Size &size : sizes) {
    const uint32_t live = size.chunks.liveChunks();
    counters.chunks += live;
    for (uint32_t chunk = 0; chunk < live; ++chunk) counters.arenaBytes += size.buffers[chunk].sizeBytes();
    counters.slots += size.chunkLayout.first(live);
  }
  counters.liveBytes = fixedBytes + counters.arenaBytes;
}

// Leaves a copy of the counters and the failure for stats(), with what the
// next shrink and growth take.
void ExpertStore::Impl::publish() {
  const bool resizes = !released && failure.empty();
  const uint32_t shrinking = resizes ? shrinkingSize() : kNoSlot, growing = resizes ? growingSize() : kNoSlot;
  counters.releaseUnitBytes =
      shrinking == kNoSlot ? 0 : sizes[shrinking].buffers[sizes[shrinking].chunks.liveChunks() - 1].sizeBytes();
  // Every chunk above the floors, in the order releaseAboveFloor() gives
  // them back: their sizes differ (the last of a size is the rest of it).
  counters.releaseUnits.clear();
  if (shrinking != kNoSlot) {
    std::vector<uint32_t> live;
    for (const Size &size : sizes) live.push_back(size.chunks.liveChunks());
    for (uint32_t index = shrinking; index != kNoSlot; index = shrinkingSize(live))
      counters.releaseUnits.push_back(sizes[index].buffers[--live[index]].sizeBytes());
  }
  counters.growUnitBytes =
      growing == kNoSlot ? 0 : sizes[growing].buffers[sizes[growing].chunks.liveChunks()].sizeBytes();
  const std::lock_guard lock(statsMutex);
  published = counters;
  published.failure = failure;
}

void ExpertStore::Impl::load(uint32_t index, ops::MoePhase phase, uint32_t rows, bool predicted,
                             std::span<const PartTiles> parts, uint32_t round, uint32_t rounds,
                             const metal::MetalBuffer &verifyRows) {
  std::lock_guard lock(mutex);
  // No job runs until the round's are staged; the last round's ran.
  static_cast<uint32_t *>(grid.contents())[1] = 0;
  staged.clear();
  if (!failure.empty()) throw std::runtime_error("expert store failed earlier: " + failure);
  ExpertStoreStats::Use &use = counters.uses[static_cast<size_t>(phase)];
  try {
    if (round == 0) {
      // A load whose later rounds did not run (its command failed, which ends
      // the engine) left routed experts in slots the slot table names but no
      // repack wrote, which the grouping before this load took as resident.
      if (!pending.empty()) throw std::logic_error("an expert load follows one whose later rounds did not run");
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
      for (const uint32_t expert : predictedExperts(rows, verifyRows))
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
  const uint32_t blocks = static_cast<uint32_t>(impl.blocks.size());
  const uint64_t minimum = minimumBytesOf(geometry, blocks, shape);
  if (options.memoryBytes < minimum)
    throw std::invalid_argument("expert memory of " + std::to_string(options.memoryBytes >> 20) +
                                " MiB holds fewer slots than the smallest store (" + std::to_string(minimum >> 20) +
                                " MiB): a block's experts of each size and every expert of a size of one block, "
                                "beside the staging and tables");
  if (options.floorBytes && (options.floorBytes < minimum || options.floorBytes > options.memoryBytes))
    throw std::invalid_argument("an expert floor of " + std::to_string(options.floorBytes >> 20) +
                                " MiB lies outside the smallest store and the store's " +
                                std::to_string(options.memoryBytes >> 20) + " MiB");
  const uint64_t fixed = fixedBytesOf(geometry, blocks, shape);
  const uint64_t largest = options.bufferBytes && options.bufferBytes < backend.capabilities().maxBufferLengthBytes
                               ? options.bufferBytes
                               : backend.capabilities().maxBufferLengthBytes;
  for (uint32_t layout = 0; layout < geometry.layouts.size(); ++layout)
    impl.sizes.push_back(
        {.layout = geometry.layouts[layout], .blocks = geometry.blocks[layout], .streams = geometry.streams(layout)});
  for (uint32_t block = 0; block < blocks; ++block) impl.blocks[block].size = geometry.layoutOf[block];
  // Every streamed size holds the same share of its experts, but at least a
  // block's (a prefill chunk may route them all) and at most all of them; a
  // size of one block all of its: the largest share the chunks hold at the
  // cap, and as its floor the largest share the floor's bytes hold (or the
  // floor share of its slots). The minimum above holds a block's experts of
  // each size.
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
  // The largest share whose slots the memory beside the fixed bytes holds.
  const auto shareOf = [&](uint64_t memoryBytes) {
    const uint64_t budget = memoryBytes > fixed ? memoryBytes - fixed : 0;
    double low = 0.0, high = 1.0;
    if (bytesAt(high) <= budget) low = high;
    for (int step = 0; step < 64 && low < high; ++step) {
      const double middle = (low + high) / 2;
      (bytesAt(middle) <= budget ? low : high) = middle;
    }
    return low;
  };
  const double capShare = shareOf(options.memoryBytes);
  const double floorShare = options.floorBytes ? shareOf(options.floorBytes) : 0.0;
  for (uint32_t index = 0; index < impl.sizes.size(); ++index) {
    Impl::Size &size = impl.sizes[index];
    size.slots = static_cast<uint32_t>(slotsAt(index, capShare));
    const uint32_t floorSlots = options.floorBytes ? static_cast<uint32_t>(slotsAt(index, floorShare)) : 0;
    for (size_t z = 0; z < 3; ++z) {
      const ProjectionLayout &projection = size.layout[z];
      size.load.projections[z] = {projection.format,
                                  static_cast<uint32_t>(projection.rows),
                                  static_cast<uint32_t>(projection.columns),
                                  static_cast<uint32_t>(projection.rowBytes),
                                  {projection.planes[0], projection.planes[1], projection.planes[2]}};
      size.routed[z] =
          ops::QuantizedSegment::planes(projection.format, static_cast<uint32_t>(size.slots * projection.rows),
                                        static_cast<uint32_t>(projection.columns), {}, {}, {});
    }
    size.loadGroups = ops::MoE::expertLoadGroups(size.load);
    // Each chunk's regions over its own slots, in the expert kernels' order
    // (ops::ExpertChunks): plane1's is meta's for a format without one. The
    // floor's chunks take their memory now, the elastic chunks as
    // growTowardCap takes them.
    const ExpertSlotPlanes slotPlanes{size.layout[0].planes, size.layout[1].planes, size.layout[2].planes};
    size.chunkLayout = size.streams ? streamedExpertChunkLayout(slotPlanes, size.slots, floorSlots, shape.experts,
                                                                largest, options.elasticChunkBytes, options.floorShare)
                                    : wholeExpertChunkLayout(slotPlanes, size.slots, largest);
    const ExpertChunkLayout &chunks = size.chunkLayout;
    std::vector<metal::MetalBuffer> planes;
    for (uint32_t chunk = 0; chunk < chunks.chunks; ++chunk) {
      const uint64_t held = chunks.held(chunk);
      std::array<std::array<uint64_t, 3>, 3> regions{};
      uint64_t bytes = 0;
      for (size_t z = 0; z < 3; ++z)
        for (size_t plane = 0; plane < 3; ++plane) {
          regions[z][plane] = bytes;
          bytes += alignWeightOffset(held * size.layout[z].planes[plane]);
        }
      const std::string label = "experts/arena-" + std::to_string(index) + "-" + std::to_string(chunk);
      const metal::MetalBuffer &buffer = size.buffers.emplace_back(
          chunk < chunks.floorChunks ? backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label)
                                     : backend.reserveBuffer(bytes, metal::BufferStorage::Shared, label));
      for (size_t z = 0; z < 3; ++z) {
        const std::array<uint64_t, 3> &regionBytes = size.layout[z].planes;
        const metal::MetalBuffer meta = backend.view(buffer, regions[z][2], held * regionBytes[2]);
        planes.push_back(backend.view(buffer, regions[z][0], held * regionBytes[0]));
        planes.push_back(regionBytes[1] ? backend.view(buffer, regions[z][1], held * regionBytes[1]) : meta);
        planes.push_back(meta);
      }
      impl.counters.capBytes += bytes;
      if (chunk < chunks.floorChunks) impl.counters.floorBytes += bytes;
    }
    size.chunks = ops::ExpertChunks::of(std::move(planes), chunks.floorSlots, chunks.chunkSlots, size.slots,
                                        size.routed[0], size.routed[1], size.routed[2]);
    size.chunks.keep(chunks.floorChunks);
    size.recycling = SlotRecycler(chunks.first(chunks.floorChunks));
    size.ownerBlock.assign(size.slots, kNoSlot);
    size.ownerExpert.assign(size.slots, kNoSlot);
  }
  impl.places = stagingOf(geometry, shape);
  impl.taken.assign(impl.places.count, 0);
  impl.staging = backend.allocateBuffer(impl.places.bytes(), metal::BufferStorage::Shared, "experts/staging");
  const uint64_t tableBytes = uint64_t{shape.experts} * impl.blocks.size() * sizeof(uint32_t);
  impl.tables = backend.allocateBuffer(tableBytes, metal::BufferStorage::Shared, "experts/slot-tables");
  std::fill_n(static_cast<uint32_t *>(impl.tables.contents()), tableBytes / sizeof(uint32_t), MOE_SLOT_ABSENT);
  impl.loadTables = backend.allocateBuffer(tableBytes, metal::BufferStorage::Shared, "experts/load-slot-tables");
  std::memset(impl.loadTables.contents(), 0, tableBytes);
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
  impl.fixedBytes = fixed;
  impl.counters.capBytes += fixed;
  impl.counters.floorBytes += fixed;
  impl.counters.stagingBytes = impl.staging.sizeBytes();
  for (const Impl::Size &size : impl.sizes) {
    impl.counters.capChunks += size.chunkLayout.chunks;
    impl.counters.experts += uint64_t{size.blocks} * shape.experts;
  }
  impl.countLive();
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
  // The routed segments give the slots' format and size; their planes are
  // the chunks'.
  weights.gate.routed = size.routed[0];
  weights.up.routed = size.routed[1];
  weights.down.routed = size.routed[2];
  weights.chunks = size.chunks;
  weights.slotTable = impl.backend.view(impl.tables, uint64_t{block} * impl.shape.experts * sizeof(uint32_t),
                                        uint64_t{impl.shape.experts} * sizeof(uint32_t));
  weights.loadSlotTable = impl.backend.view(impl.loadTables,
                                            uint64_t{block} * impl.shape.experts * sizeof(uint32_t),
                                            uint64_t{impl.shape.experts} * sizeof(uint32_t));
  weights.slots = size.slots;
  // The experts of a size of one block stay in their slots: nothing loads
  // them.
  weights.stream = impl.streams(block) ? this : nullptr;
  weights.streamBlock = block;
}

void ExpertStore::addLoad(metal::CommandGraph &graph, std::span<const ops::MoeLoadPart> parts, uint32_t block,
                          const std::function<void()> &beside) const {
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
  const metal::MetalBuffer verifyRows = predicted ? parts.front().buffers->verifyRows : metal::MetalBuffer{};
  // The first round's step runs beside the passes over the experts resident
  // before it, whose slots route() leaves as they are (it touches every routed
  // resident slot before it claims any): they read the grouping's copy of the
  // slot table, no line of what a load's step writes.
  for (uint32_t round = 0; round < rounds; ++round) {
    graph.host(
        [impl = impl_, block, phase, rows, predicted, tiles, round, rounds, verifyRows] {
          impl->load(block, phase, rows, predicted, tiles, round, rounds, verifyRows);
        },
        timed, gpu, startAhead, round ? std::function<void()>{} : beside,
        {impl.tables, impl.loadTables, impl.staging, impl.jobs, impl.sources, impl.grid});
    ops::MoE::addExpertLoad(graph, impl.staging, size.chunks, impl.jobs, impl.sources, size.load,
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

uint64_t ExpertStore::readBytes() const {
  const std::lock_guard lock(impl_->statsMutex);
  return impl_->published.uses[0].readBytes + impl_->published.uses[1].readBytes;
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
  for (Impl::Block &block : impl.blocks) {
    std::ranges::fill(block.slot, kNoSlot);
    block.resident = 0;
  }
  std::fill_n(static_cast<uint32_t *>(impl.tables.contents()), impl.tables.sizeBytes() / sizeof(uint32_t),
              MOE_SLOT_ABSENT);
  // The chunks past the live ones were given back already.
  for (Impl::Size &size : impl.sizes) {
    for (uint32_t chunk = 0; chunk < size.chunks.liveChunks(); ++chunk) impl.backend.releaseMemory(size.buffers[chunk]);
    std::ranges::fill(size.ownerBlock, kNoSlot);
    std::ranges::fill(size.ownerExpert, kNoSlot);
    size.recycling.reset();
  }
  impl.backend.releaseMemory(impl.staging);
  impl.forgetRepack();
  impl.counters.resident = 0;
  impl.released = true;
  impl.countLive();
  impl.publish();
}

bool ExpertStore::restore(const metal::AllocationAdmission &admit) {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  if (!impl.released) throw std::logic_error("expert memory is not released");
  // The floor chunks and the staging come back in one part, or none of them.
  std::vector<const metal::MetalBuffer *> floor{&impl.staging};
  for (const Impl::Size &size : impl.sizes)
    for (uint32_t chunk = 0; chunk < size.chunkLayout.floorChunks; ++chunk) floor.push_back(&size.buffers[chunk]);
  uint64_t bytes = 0;
  for (const metal::MetalBuffer *buffer : floor) bytes += buffer->sizeBytes();
  std::vector<const metal::MetalBuffer *> restored;
  const metal::AllocationResult admitted = admit(bytes, [&] {
    try {
      for (const metal::MetalBuffer *buffer : floor) {
        impl.backend.restoreMemory(*buffer);
        restored.push_back(buffer);
      }
    } catch (const metal::MetalAllocationError &) {
      // A buffer the driver refused leaves them all released.
      for (const metal::MetalBuffer *buffer : restored) impl.backend.releaseMemory(*buffer);
      throw;
    }
  });
  if (!admitted)
    throw metal::MetalAllocationError(std::string("unable to restore the expert chunks: ") +
                                          metal::allocationFailureName(admitted.failure),
                                      admitted.failure);
  // The elastic chunks stay released: growTowardCap() takes them back, their
  // slots free, as memory allows.
  for (Impl::Size &size : impl.sizes) {
    size.chunks.keep(size.chunkLayout.floorChunks);
    size.recycling.resize(size.chunkLayout.first(size.chunkLayout.floorChunks));
    // Every slot free, the lowest claimed first.
    size.recycling.reset();
  }
  impl.released = false;
  impl.countLive();
  impl.fill();
  impl.publish();
  return true;
}

uint64_t ExpertStore::releaseAboveFloor() {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  if (!impl.mayShrink()) return 0;
  const uint32_t index = impl.shrinkingSize();
  return index == kNoSlot ? 0 : impl.shrink(index, impl.sizes[index].chunks.liveChunks() - 1);
}

uint64_t ExpertStore::releaseToFloor() {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  if (!impl.mayShrink()) return 0;
  uint64_t bytes = 0;
  for (uint32_t index = 0; index < impl.sizes.size(); ++index)
    if (const uint32_t floor = impl.sizes[index].chunkLayout.floorChunks;
        impl.sizes[index].chunks.liveChunks() > floor)
      bytes += impl.shrink(index, floor);
  return bytes;
}

metal::AllocationResult ExpertStore::growTowardCap(const metal::AllocationAdmission &admit) {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  if (impl.released || !impl.failure.empty() || impl.backend.commandInFlight()) return {};
  const uint32_t index = impl.growingSize();
  if (index == kNoSlot) return {};
  if (const metal::AllocationResult admitted = impl.grow(index, admit); !admitted) return admitted;
  ++impl.counters.regrows;
  impl.countLive();
  impl.publish();
  const Impl::Size &size = impl.sizes[index];
  logLine("Expert store took back a chunk of ", size.buffers[size.chunks.liveChunks() - 1].sizeBytes() >> 20,
          " MiB, its slots empty; ", impl.counters.chunks, " of ", impl.counters.capChunks, " chunks hold ",
          impl.counters.slots, " slots.");
  return {};
}

uint64_t ExpertStore::fixedBytes(const StreamedBlocks &blocks, ops::MoeShape shape) {
  return fixedBytesOf(geometryOf(blocks, shape), static_cast<uint32_t>(blocks.blocks.size()), shape);
}

uint64_t ExpertStore::minimumBytes(const StreamedBlocks &blocks, ops::MoeShape shape) {
  return minimumBytesOf(geometryOf(blocks, shape), static_cast<uint32_t>(blocks.blocks.size()), shape);
}

metal::MetalBuffer residentExpertSlots(metal::MetalBackend &backend, uint32_t experts) {
  metal::MetalBuffer table =
      backend.allocateBuffer(residentExpertSlotsBytes(experts), metal::BufferStorage::Shared,
                             "experts/resident-slots");
  auto *slot = static_cast<uint32_t *>(table.contents());
  for (uint32_t expert = 0; expert < experts; ++expert) slot[expert] = expert;
  return table;
}

} // namespace splash::model

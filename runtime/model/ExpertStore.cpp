#include "model/ExpertStore.hpp"
#include "AwakeClock.hpp"
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

// One projection of an expert of some layout: rows x columns in format, its
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

// Where an expert keeps its planes in its slot: projection z's plane i from
// offsets[z][i] on, the projections' planes in turn, each from a multiple of
// kSlotPlaneAlignment (more than any kernel's loads take); and the bytes of
// the slot that holds them.
constexpr uint64_t kSlotPlaneAlignment = 256;
struct SlotLayout final {
  std::array<std::array<uint64_t, 3>, 3> offsets{};
  uint64_t bytes = 0;
};
SlotLayout slotLayout(const ExpertLayout &layout) noexcept {
  SlotLayout slot;
  for (size_t z = 0; z < layout.size(); ++z)
    for (size_t plane = 0; plane < 3; ++plane) {
      slot.offsets[z][plane] = slot.bytes;
      slot.bytes += alignUp(layout[z].planes[plane], kSlotPlaneAlignment);
    }
  return slot;
}

// The bytes of a chunk of `slots` slots of `slotBytes` each, in turn. The
// store's chunks take the bytes of one chunk of all its slots
// (ExpertChunkLayout).
uint64_t chunkBytes(uint64_t slotBytes, uint64_t slots) noexcept { return alignWeightOffset(slots * slotBytes); }

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
// fixedBytes): each block's layout, the layouts, the native bytes of each
// projection of a place, the most of any layout's expert, and the bytes of a
// slot, the widest expert's in whole kWeightFileAlignment units: a chunk of
// any number of slots fills its region whole, so the store's floor and
// elastic chunks part its slots at any slot (ExpertChunkLayout).
struct Geometry final {
  std::vector<uint32_t> layoutOf;
  std::vector<ExpertLayout> layouts;
  std::array<uint64_t, 3> nativeBytes{};
  uint64_t slotBytes = 0;
};

Geometry geometryOf(const StreamedBlocks &streamed, ops::MoeShape shape) {
  if (!shape.valid() || streamed.blocks.empty()) throw std::invalid_argument("invalid expert store");
  Geometry g;
  for (const gguf::StreamedExperts &experts : streamed.blocks) {
    const ExpertLayout layout = expertLayout(experts, shape);
    const auto found = std::ranges::find(g.layouts, layout);
    g.layoutOf.push_back(static_cast<uint32_t>(found - g.layouts.begin()));
    if (found == g.layouts.end()) g.layouts.push_back(layout);
  }
  for (const ExpertLayout &layout : g.layouts) {
    for (size_t z = 0; z < 3; ++z) g.nativeBytes[z] = std::max(g.nativeBytes[z], layout[z].nativeBytes());
    g.slotBytes = std::max(g.slotBytes, alignWeightOffset(slotLayout(layout).bytes));
  }
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
  // A split decode load's two passes' tiles.
  const uint64_t split = 2 * uint64_t{ops::moeSplitTileCapacity(kCycleRows, shape)} * sizeof(MoeTileDescriptor);
  return stagingOf(g, shape).bytes() + uint64_t{blocks} * shape.experts * sizeof(uint32_t) +
         uint64_t{shape.experts} * (sizeof(uint32_t) + 3 * sizeof(uint64_t)) + 4 * sizeof(uint32_t) + prediction +
         split;
}

// The smallest store (ExpertStore::minimumBytes): its fixed bytes and a
// chunk of a block's experts in slots of the widest expert's size, which for
// a store of one block are all of its experts.
uint64_t minimumBytesOf(const Geometry &g, uint32_t blocks, ops::MoeShape shape) {
  return fixedBytesOf(g, blocks, shape) + chunkBytes(g.slotBytes, shape.experts);
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
  // The experts of every block, in slots as wide as the widest of them
  // (Geometry::slotBytes): their slots, a streamed store's recycled in its
  // order (SlotRecycler), in its floor chunk and elastic chunks
  // (streamedExpertChunkLayout), or a store of one block's in its floor chunk
  // alone (wholeExpertChunkLayout), each a buffer of its own slots in turn.
  // The first liveChunks() of them are live and the recycler holds their
  // slots; the others are released.
  struct Size final {
    uint32_t blocks = 0;
    // Whether it holds the experts of several blocks.
    bool streams = false;
    // Its slots at the cap, which its blocks' segments, owners and slot
    // tables count, and where its chunks hold them.
    uint32_t slots = 0;
    ExpertChunkLayout chunkLayout;
    // The chunks' buffers; and per layout (Geometry::layouts) its routed
    // segments, of the slots in its formats and no planes (bind), and the
    // chunks' regions as the expert kernels reach them, views of each chunk
    // from each of its planes' places in a slot on, checked for those
    // segments, which the layout's blocks share.
    std::vector<metal::MetalBuffer> buffers;
    std::vector<std::array<ops::QuantizedSegment, 3>> routed;
    std::vector<ops::ExpertChunks> chunks;
    // Which slot a load's missing expert takes.
    SlotRecycler recycling{0};
    std::vector<uint32_t> ownerBlock, ownerExpert; // kNoSlot when free

    // The live chunks, which every layout's views name alone.
    [[nodiscard]] uint32_t liveChunks() const noexcept { return chunks.front().liveChunks(); }
    void keep(uint32_t live) {
      for (ops::ExpertChunks &views : chunks) views.keep(live);
    }
  };
  struct Block final {
    // Its experts' layout (Geometry::layouts), and the repack of its staged
    // rows into the slots: projection z's plane i from its place in a slot on
    // (slotLayout), in its layout's views of the chunks.
    uint32_t layout = 0;
    MoeExpertLoadParams load{};
    uint32_t loadGroups = 0;
    gguf::StreamedExperts experts;
    std::vector<uint32_t> slot; // each expert's, kNoSlot when not resident
    // Its router, shared-expert gate and a SqrtSoftplusNormalized router's
    // selection bias (bind), which predict its routes.
    ops::QuantizedSegment router, scalarGate;
    metal::MetalBuffer selectionBias;
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
    if (demand) readers->stop(*demand);
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
  // Whether the blocks' experts stream: those of a store of several blocks.
  [[nodiscard]] bool streams() const noexcept { return size.streams; }
  // What a decode load's host step reports its round trip to.
  static std::function<void(const metal::HostStepTiming &)> timedDecode(std::shared_ptr<Impl> impl) {
    return [impl = std::move(impl)](const metal::HostStepTiming &timing) {
      StepTimes &times = impl->stepTimes[static_cast<size_t>(ops::MoePhase::Decode)];
      const auto nanoseconds = [](double seconds) { return uint64_t(seconds * 1e9); };
      times.pickup.add(nanoseconds(timing.pickupSeconds));
      times.work.add(nanoseconds(timing.workSeconds));
      times.resume.add(nanoseconds(timing.resumeSeconds));
    };
  }
  [[nodiscard]] const ExpertLayout &layoutOf(uint32_t block) const noexcept {
    return geometry.layouts[blocks[block].layout];
  }
  // Adds to `reads` a group of the experts from `expert` on at the places
  // from `place` on of `staged`, `count` of them, read in pieces of at most
  // pieceBytes (whole pages, or whole rows for a built run), and returns it.
  const Group &addGroup(Reads &reads, uint32_t block, uint32_t expert, uint32_t count, uint32_t place,
                        const Places &staged, uint64_t pieceBytes) const;
  // Where expert k of `group` starts in the staging, its projection z.
  [[nodiscard]] uint64_t expertAt(const Group &group, uint32_t k, uint32_t z) const noexcept {
    const Run &run = group.runs[z];
    return run.at + run.inPage + k * layoutOf(group.block)[z].nativeBytes();
  }
  void readPiece(const Group &group, const Piece &piece) const;
  void stage(uint32_t block, uint32_t capacity, ExpertStoreStats::Use &use);
  // A split decode load of block `block` (ExpertStore::addSplitLoadStart),
  // of `rows` rows in the tiles of `part`: the first host step routes it,
  // writes the two passes' tiles and starts its reads (stageStart); the
  // second waits for them (stageFinish) and plans the next block's
  // read-ahead, as load() does.
  void splitStart(uint32_t block, uint32_t rows, const PartTiles &part);
  void splitFinish(uint32_t block, uint32_t rows, bool predicted, const metal::MetalBuffer &verifyRows);
  void stageStart(uint32_t block);
  void stageFinish(uint32_t block, ExpertStoreStats::Use &use);
  std::span<const uint32_t> predictedExperts(uint32_t rows, const metal::MetalBuffer &verifyRows);
  void planAhead(uint32_t block, std::span<const uint32_t> experts, ExpertStoreStats::Use &use);
  void startAhead();
  // The host step after a long chunk's load: block `block`'s experts that
  // are not resident read ahead into the whole staging.
  void readBlockAhead(uint32_t block, ops::MoePhase phase);
  void countAhead(size_t group, size_t from, size_t to);
  void retireAhead();
  void evict(uint32_t slot);
  void abandon();
  void fill();
  // Whether the store may give chunks back now: it holds its memory, no
  // command is in flight, no load failed, and no routed expert waits to be
  // staged. A command that failed after a load's last round staged leaves
  // nothing the store knows of, but a failed command ends the engine before
  // any shrink (ExpertStore.hpp): these guards are a defence.
  [[nodiscard]] bool mayShrink() const {
    return !released && failure.empty() && pending.empty() && !backend.commandInFlight();
  }
  // Gives the live chunks from `live` on back (releaseAboveFloor,
  // releaseToFloor); returns their bytes.
  uint64_t shrink(uint32_t live);
  // Takes the next chunk back under `admit`, its slots free.
  [[nodiscard]] metal::AllocationResult grow(const metal::AllocationAdmission &admit);
  // The expert in slot `from` moves to the free slot `to`: its planes, its
  // owner, and its block's slot and table entry.
  void move(uint32_t from, uint32_t to);
  // Before the slots from `boundary` on are given back: no block names them,
  // every table entry is the block's slot of the expert (0 for one not
  // resident), no slot from it on has an owner, and the recycler holds the
  // slots below it alone. Throws std::logic_error otherwise.
  void requireBelow(uint32_t boundary) const;
  // No repack names a slot: a load's host step that does not run (one
  // before it threw, or the command before it failed) leaves the repack
  // after it no job, rather than the last round's, whose slots a shrink may
  // have moved or given back; and no slot is staged, as no load is under way.
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
  Size size;
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
  // memory back or takes it again), hand the state below and the size's
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
  // A split decode load's: the two passes' tiles (ops::MoeSplitTiles) of
  // up to splitCapacity tiles; its reads, which run beside the resident
  // experts' passes, and the bytes they read; the read-ahead's groups it took
  // over (their group, pieces [from, to)) and of their pieces those its
  // readers were still reading; and the experts it reads.
  std::array<metal::MetalBuffer, 2> splitTiles;
  uint32_t splitCapacity = 0;
  std::shared_ptr<SsdReaders::Batch> demand;
  std::atomic<uint64_t> demandBytes = 0;
  struct TakenOver final {
    size_t group = 0, from = 0, to = 0;
  };
  std::vector<TakenOver> takenOver;
  std::vector<size_t> awaited;
  std::vector<uint8_t> loading;
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
void ExpertStore::Impl::evict(uint32_t slot) {
  const uint32_t owner = size.ownerBlock[slot];
  if (owner == kNoSlot) return;
  const uint32_t expert = size.ownerExpert[slot];
  blocks[owner].slot[expert] = kNoSlot;
  // A table entry of no resident expert names slot 0, which is inside chunk
  // 0, the floor chunk: the kernels read only routed experts' entries, which
  // a load set, so it is never used.
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
    evict(slot);
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
    const ProjectionLayout &projection = layoutOf(index)[z];
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
  const ProjectionLayout &projection = layoutOf(group.block)[piece.z];
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
  threadgroups[0] = blocks[index].loadGroups;
  threadgroups[1] = static_cast<uint32_t>(staged.size());
  threadgroups[2] = 3;
}

// A split decode load's round (one stages every expert it routes), as
// stage() stages it but without waiting for a read: of the read-ahead's
// groups (it is halted), it takes over those of the experts it routes,
// awaiting their pieces still being read and reading those never taken or
// failed; it keeps the places of the others still being read; it places the
// other experts it reads and starts the reads on the readers.
void ExpertStore::Impl::stageStart(uint32_t index) {
  std::ranges::fill(taken, uint8_t{0});
  now.groups.clear();
  now.pieces.clear();
  takenOver.clear();
  awaited.clear();
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
  const std::vector<Piece> &aheadPieces = ahead.reads.pieces;
  const SsdReaders::Batch *batch = ahead.batch.get();
  const auto reading = [&](size_t piece) { return piece < batch->taken() && !batch->ended(piece); };
  for (size_t piece = 0, end = 0; batch && piece < aheadPieces.size(); piece = end) {
    const uint32_t at = aheadPieces[piece].group;
    const Group &group = ahead.reads.groups[at];
    while (end < aheadPieces.size() && aheadPieces[end].group == at) ++end;
    const bool routed = ahead.block == index && waiting[group.expert];
    if (!routed) {
      bool ended = true;
      for (size_t rest = piece; rest < end; ++rest) ended = ended && !reading(rest);
      if (ended) countAhead(at, piece, end);
      else taken[group.place] = 1;
      continue;
    }
    taken[group.place] = 1;
    now.groups.push_back(group);
    const auto g = static_cast<uint32_t>(now.groups.size() - 1);
    for (size_t rest = piece; rest < end; ++rest) {
      const Piece &unread = aheadPieces[rest];
      if (rest >= batch->taken()) now.pieces.push_back({g, unread.z, unread.from, unread.to});
      else if (!batch->ended(rest)) awaited.push_back(rest);
      else if (batch->failed(rest)) now.pieces.push_back({g, unread.z, unread.from, unread.to});
    }
    takenOver.push_back({at, piece, end});
    addJob(now.groups.back(), 0);
  }
  uint32_t place = 0;
  for (size_t at = 0; at < pending.size();) {
    if (!waiting[pending[at].first]) {
      ++at;
      continue;
    }
    while (place < places.count && taken[place]) ++place;
    if (place == places.count) throw std::logic_error("a split expert load has fewer places than experts");
    uint32_t count = 1;
    while (at + count < pending.size() && pending[at + count].first == pending[at].first + count &&
           waiting[pending[at + count].first] && place + count < places.count && !taken[place + count])
      ++count;
    const Group &group = addGroup(now, index, pending[at].first, count, place, places, kLoadStepBytes);
    for (uint32_t k = 0; k < count; ++k) {
      taken[place + k] = 1;
      addJob(group, k);
    }
    at += count;
  }
  demandBytes = 0;
  if (!now.pieces.empty())
    demand = readers->start(
        now.pieces.size(),
        [this](size_t task) {
          const Piece &piece = now.pieces[task];
          readPiece(now.groups[piece.group], piece);
          demandBytes += piece.to - piece.from;
        },
        kSsdReaders);
}

// A split decode load's round once the resident experts' passes ran: its
// reads ended (the read-ahead's pieces it awaited too, read again where they
// failed), what it read counted and checked, and the repack's grid.
void ExpertStore::Impl::stageFinish(uint32_t index, ExpertStoreStats::Use &use) {
  const SsdReaders::Batch *batch = ahead.batch.get();
  for (const size_t piece : awaited) {
    while (!batch->ended(piece)) std::this_thread::yield();
    if (batch->failed(piece)) {
      const Piece &unread = ahead.reads.pieces[piece];
      readPiece(ahead.reads.groups[unread.group], unread);
      use.readBytes += unread.to - unread.from;
    }
  }
  if (demand) {
    readers->finish(*demand);
    const std::shared_ptr<SsdReaders::Batch> ended = std::exchange(demand, nullptr);
    use.readBytes += demandBytes;
    if (const std::exception_ptr error = ended->failure()) std::rethrow_exception(error);
  }
  for (const TakenOver &group : takenOver) {
    bool whole = true;
    for (size_t piece = group.from; piece < group.to; ++piece)
      whole = whole && piece < batch->taken() && !batch->failed(piece);
    use.prefetched += whole;
    countAhead(group.group, group.from, group.to);
  }
  // What the load read is the sources' as they were opened.
  if (!staged.empty())
    for (const std::function<void()> &check : checks) check();
  std::erase_if(pending, [&](const auto &entry) { return !waiting[entry.first]; });
  if (!pending.empty()) throw std::logic_error("a split expert load left experts unstaged");
  auto *threadgroups = static_cast<uint32_t *>(grid.contents());
  threadgroups[0] = blocks[index].loadGroups;
  threadgroups[1] = static_cast<uint32_t>(staged.size());
  threadgroups[2] = 3;
}

void ExpertStore::Impl::splitStart(uint32_t index, uint32_t rows, const PartTiles &part) {
  std::lock_guard lock(mutex);
  // No job runs until the finish stages them.
  static_cast<uint32_t *>(grid.contents())[1] = 0;
  staged.clear();
  if (!failure.empty()) throw std::runtime_error("expert store failed earlier: " + failure);
  ExpertStoreStats::Use &use = counters.uses[static_cast<size_t>(ops::MoePhase::Decode)];
  try {
    abandon();
    const auto *counted = static_cast<const MoeTileCount *>(part.count.contents());
    const auto *descriptors = static_cast<const MoeTileDescriptor *>(part.descriptors.contents());
    const uint64_t bound = part.descriptors.sizeBytes() / sizeof(MoeTileDescriptor);
    if (!counted || !descriptors || counted->tiles > bound || counted->tiles > splitCapacity)
      throw std::runtime_error("a MoE block's tiles are out of bounds");
    partTiles.assign(1, {descriptors, counted->tiles});
    route(index, partTiles, ops::MoePhase::Decode, rows, use);
    ++use.loads;
    // Each live tile in the pass of its expert: the loaded one for an expert
    // this load reads, the resident one for the others and the shared
    // expert; in the other, no rows.
    for (const auto &entry : pending) loading[entry.first] = 1;
    auto *resident = static_cast<MoeTileDescriptor *>(splitTiles[0].contents());
    auto *loaded = static_cast<MoeTileDescriptor *>(splitTiles[1].contents());
    for (uint32_t tile = 0; tile < counted->tiles; ++tile) {
      const MoeTileDescriptor descriptor = descriptors[tile];
      const bool reads = descriptor.expert < shape.experts && loading[descriptor.expert];
      resident[tile] = {descriptor.expert, reads ? 0u : descriptor.rows};
      loaded[tile] = {descriptor.expert, reads ? descriptor.rows : 0u};
    }
    for (const auto &entry : pending) loading[entry.first] = 0;
    // The read-ahead halts: its readers take no more pieces.
    if (ahead.batch) readers->halt(*ahead.batch);
    stageStart(index);
    ahead.block = kNoSlot;
  } catch (const std::exception &error) {
    abandon();
    failure = error.what();
    publish();
    throw;
  }
  publish();
}

void ExpertStore::Impl::splitFinish(uint32_t index, uint32_t rows, bool predicted,
                                    const metal::MetalBuffer &verifyRows) {
  std::lock_guard lock(mutex);
  if (!failure.empty()) throw std::runtime_error("expert store failed earlier: " + failure);
  ExpertStoreStats::Use &use = counters.uses[static_cast<size_t>(ops::MoePhase::Decode)];
  try {
    if (pendingBlock != index) throw std::logic_error("a split expert load's finish follows another block's start");
    stageFinish(index, use);
    // The next block's likely experts are read ahead while the GPU repacks
    // and runs this block's loaded experts (startAhead).
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
  // A split load whose finish did not run (its command failed) stops its
  // reads before its places and slots go.
  if (demand) readers->stop(*std::exchange(demand, nullptr));
  if (pendingBlock == kNoSlot) return;
  Block &block = blocks[pendingBlock];
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

// Reads every expert of a store of one block into its slot, expert e into
// slot e, through the staging, as many a command as it places; its last
// repack's jobs are forgotten once it ran.
void ExpertStore::Impl::fill() {
  for (uint32_t index = 0; index < blocks.size(); ++index) {
    if (streams()) continue;
    Block &block = blocks[index];
    std::array<uint64_t, 3> nativeBytes;
    for (size_t z = 0; z < 3; ++z) nativeBytes[z] = layoutOf(index)[z].nativeBytes();
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
      threadgroups[0] = block.loadGroups;
      threadgroups[1] = experts;
      threadgroups[2] = 3;
      metal::CommandGraph graph;
      ops::MoE::addExpertLoad(graph, staging, size.chunks[block.layout], jobs, sources, block.load, experts,
                              metal::IndirectGrid{grid, 0});
      static_cast<void>(backend.submitCommandAsync(graph.command()).wait());
    }
    counters.resident += shape.experts;
  }
  forgetRepack();
}

// The experts of the chunks that go that fit move below them, after the
// recycler evicted those its order recycles first until they fit
// (SlotRecycler::compact), in one compaction, so that each moves once; once
// the slot tables are checked, no address table names those chunks from the
// next command on, and their memory goes, the last first. The slots claimed
// by a load in flight could move under it, so no command is in flight, and
// every command before succeeded (ExpertStore.hpp): no load is unfinished.
// Logs a line with what moved and what the store holds after.
uint64_t ExpertStore::Impl::shrink(uint32_t live) {
  const AwakeClock::time_point started = AwakeClock::now();
  const uint32_t chunks = size.liveChunks();
  const uint32_t boundary = size.chunkLayout.first(live);
  uint32_t evicted = 0, moved = 0;
  size.recycling.compact(
      boundary,
      [&](uint32_t slot) {
        evict(slot);
        ++evicted;
      },
      [&](uint32_t from, uint32_t to) {
        move(from, to);
        ++moved;
      });
  requireBelow(boundary);
  forgetRepack();
  size.keep(live);
  uint64_t bytes = 0;
  for (uint32_t chunk = chunks; chunk-- > live;) {
    backend.releaseMemory(size.buffers[chunk]);
    bytes += size.buffers[chunk].sizeBytes();
  }
  counters.shrinks += chunks - live;
  countLive();
  publish();
  const double took = millisecondsSince(started);
  logLine("Expert store gave back ", chunks - live == 1 ? "a chunk" : std::to_string(chunks - live) + " chunks",
          " of ", bytes >> 20, " MiB while memory is short, in ", std::fixed, std::setprecision(1), took,
          " ms: ", moved, " experts moved below ", chunks - live == 1 ? "it" : "them", " and ", evicted,
          " evicted; ", counters.chunks, " of ", counters.capChunks, " chunks hold ", counters.slots, " slots.");
  return bytes;
}

metal::AllocationResult ExpertStore::Impl::grow(const metal::AllocationAdmission &admit) {
  const uint32_t chunk = size.liveChunks();
  const metal::MetalBuffer &buffer = size.buffers[chunk];
  if (const metal::AllocationResult admitted = admit(buffer.sizeBytes(), [&] { backend.restoreMemory(buffer); });
      !admitted)
    return admitted;
  size.recycling.resize(size.chunkLayout.first(chunk + 1));
  size.keep(chunk + 1);
  return {};
}

void ExpertStore::Impl::move(uint32_t from, uint32_t to) {
  // A slot's planes are one run of its chunk, the slot's bytes.
  const uint64_t bytes = geometry.slotBytes;
  const uint32_t fromChunk = size.chunkLayout.chunkOf(from), toChunk = size.chunkLayout.chunkOf(to);
  const uint64_t fromLocal = from - size.chunkLayout.first(fromChunk), toLocal = to - size.chunkLayout.first(toChunk);
  std::memcpy(static_cast<uint8_t *>(size.buffers[toChunk].contents()) + toLocal * bytes,
              static_cast<const uint8_t *>(size.buffers[fromChunk].contents()) + fromLocal * bytes, bytes);
  const uint32_t owner = std::exchange(size.ownerBlock[from], kNoSlot);
  const uint32_t expert = std::exchange(size.ownerExpert[from], kNoSlot);
  size.ownerBlock[to] = owner;
  size.ownerExpert[to] = expert;
  if (owner == kNoSlot) return;
  blocks[owner].slot[expert] = to;
  static_cast<uint32_t *>(tables.contents())[uint64_t{owner} * shape.experts + expert] = to;
}

void ExpertStore::Impl::requireBelow(uint32_t boundary) const {
  const auto *table = static_cast<const uint32_t *>(tables.contents());
  for (uint32_t block = 0; block < blocks.size(); ++block) {
    for (uint32_t expert = 0; expert < shape.experts; ++expert) {
      const uint32_t slot = blocks[block].slot[expert];
      if ((slot != kNoSlot && slot >= boundary) ||
          table[uint64_t{block} * shape.experts + expert] != (slot == kNoSlot ? 0 : slot))
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
    throw std::logic_error("the expert recycler holds " + std::to_string(size.recycling.size()) +
                           " slots, not the " + std::to_string(boundary) + " kept");
}

void ExpertStore::Impl::forgetRepack() {
  static_cast<uint32_t *>(grid.contents())[1] = 0;
  staged.clear();
}

void ExpertStore::Impl::countLive() {
  counters.chunks = counters.arenaBytes = counters.slots = 0;
  if (released) return;
  const uint32_t live = size.liveChunks();
  counters.chunks = live;
  for (uint32_t chunk = 0; chunk < live; ++chunk) counters.arenaBytes += size.buffers[chunk].sizeBytes();
  counters.slots = size.chunkLayout.first(live);
}

// Leaves a copy of the counters and the failure for stats().
void ExpertStore::Impl::publish() {
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
    impl.blocks.push_back({.layout = geometry.layoutOf[block],
                           .experts = std::move(streamed.blocks[block]),
                           .slot = std::vector<uint32_t>(shape.experts, kNoSlot)});
  const uint32_t blocks = static_cast<uint32_t>(impl.blocks.size());
  const uint64_t minimum = minimumBytesOf(geometry, blocks, shape);
  if (options.memoryBytes < minimum)
    throw std::invalid_argument("expert memory of " + std::to_string(options.memoryBytes >> 20) +
                                " MiB holds fewer slots than the smallest store (" + std::to_string(minimum >> 20) +
                                " MiB): a block's experts in slots of the widest expert's size, beside the staging "
                                "and tables");
  if (options.floorBytes && (options.floorBytes < minimum || options.floorBytes > options.memoryBytes))
    throw std::invalid_argument("an expert floor of " + std::to_string(options.floorBytes >> 20) +
                                " MiB lies outside the smallest store and the store's " +
                                std::to_string(options.memoryBytes >> 20) + " MiB");
  if (geometry.slotBytes > UINT32_MAX) throw std::invalid_argument("an expert takes more than 4 GiB");
  const uint64_t fixed = fixedBytesOf(geometry, blocks, shape);
  const uint64_t largest = options.bufferBytes && options.bufferBytes < backend.capabilities().maxBufferLengthBytes
                               ? options.bufferBytes
                               : backend.capabilities().maxBufferLengthBytes;
  Impl::Size &size = impl.size;
  size.blocks = blocks;
  size.streams = blocks > 1;
  // A streamed store holds a share of its blocks' experts, but at least a
  // block's (a prefill chunk may route them all) and at most all of them; a
  // store of one block all of its: the largest share its chunks hold at the
  // cap, and as its floor the largest share the floor's bytes hold (or
  // kExpertFloorShare of its slots). The minimum above holds a block's
  // experts.
  const auto slotsAt = [&](double share) {
    const uint64_t all = uint64_t{blocks} * shape.experts;
    if (!size.streams) return all;
    return std::clamp<uint64_t>(static_cast<uint64_t>(share * double(all)), shape.experts, all);
  };
  const auto bytesAt = [&](double share) { return chunkBytes(geometry.slotBytes, slotsAt(share)); };
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
  size.slots = static_cast<uint32_t>(slotsAt(capShare));
  const uint32_t floorSlots = static_cast<uint32_t>(options.floorBytes ? slotsAt(floorShare)
                                                                       : kExpertFloorShare * double(size.slots));
  // Each chunk holds its slots in turn, a slot every plane of an expert as
  // its layout places them (slotLayout).
  ExpertSlotPlanes slotPlanes{};
  slotPlanes[0][0] = geometry.slotBytes;
  size.chunkLayout = size.streams ? streamedExpertChunkLayout(slotPlanes, size.slots, floorSlots, shape.experts,
                                                              largest, options.elasticChunkBytes)
                                  : wholeExpertChunkLayout(slotPlanes, size.slots, largest);
  const ExpertChunkLayout &chunks = size.chunkLayout;
  for (uint32_t chunk = 0; chunk < chunks.chunks; ++chunk) {
    const uint64_t bytes = chunkBytes(geometry.slotBytes, chunks.held(chunk));
    size.buffers.push_back(
        backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "experts/arena-" + std::to_string(chunk)));
    impl.counters.capBytes += bytes;
    if (chunk < chunks.floorChunks) impl.counters.floorBytes += bytes;
  }
  // Each layout's routed segments and its views of the chunks, in the expert
  // kernels' order (ops::ExpertChunks), each from its plane's place in a slot
  // to the chunk's end: plane1's is meta's for a format without one.
  for (const ExpertLayout &layout : geometry.layouts) {
    const SlotLayout slot = slotLayout(layout);
    std::array<ops::QuantizedSegment, 3> &routed = size.routed.emplace_back();
    std::vector<metal::MetalBuffer> planes;
    for (size_t z = 0; z < 3; ++z)
      routed[z] = ops::QuantizedSegment::planes(layout[z].format, static_cast<uint32_t>(size.slots * layout[z].rows),
                                                static_cast<uint32_t>(layout[z].columns), {}, {}, {});
    for (const metal::MetalBuffer &buffer : size.buffers)
      for (size_t z = 0; z < 3; ++z) {
        const auto view = [&](size_t plane) {
          return backend.view(buffer, slot.offsets[z][plane], buffer.sizeBytes() - slot.offsets[z][plane]);
        };
        const metal::MetalBuffer meta = view(2);
        planes.push_back(view(0));
        planes.push_back(layout[z].planes[1] ? view(1) : meta);
        planes.push_back(meta);
      }
    size.chunks.push_back(ops::ExpertChunks::of(std::move(planes), chunks.floorSlots, chunks.chunkSlots, size.slots,
                                                routed[0], routed[1], routed[2],
                                                static_cast<uint32_t>(geometry.slotBytes)));
  }
  // Each block's repack of its staged rows into the slots, its planes where
  // its layout keeps them in a slot, slots a slot's bytes apart.
  for (Impl::Block &block : impl.blocks) {
    for (size_t z = 0; z < 3; ++z) {
      const ProjectionLayout &projection = geometry.layouts[block.layout][z];
      block.load.projections[z] = {projection.format,
                                   static_cast<uint32_t>(projection.rows),
                                   static_cast<uint32_t>(projection.columns),
                                   static_cast<uint32_t>(projection.rowBytes),
                                   {geometry.slotBytes, geometry.slotBytes, geometry.slotBytes}};
    }
    block.loadGroups = ops::MoE::expertLoadGroups(block.load);
  }
  size.recycling = SlotRecycler(size.slots);
  size.ownerBlock.assign(size.slots, kNoSlot);
  size.ownerExpert.assign(size.slots, kNoSlot);
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
  // A split decode load's passes' tiles, of a decode step of every lane's
  // verify rows.
  impl.splitCapacity = ops::moeSplitTileCapacity(kCycleRows, shape);
  for (metal::MetalBuffer &tiles : impl.splitTiles)
    tiles = backend.allocateBuffer(uint64_t{impl.splitCapacity} * sizeof(MoeTileDescriptor),
                                   metal::BufferStorage::Shared, "experts/split-tiles");
  impl.loading.assign(shape.experts, 0);
  for (uint32_t expert = 0; expert < shape.experts; ++expert) impl.everyExpert[expert] = expert;
  impl.jobs = backend.allocateBuffer(uint64_t{shape.experts} * sizeof(uint32_t), metal::BufferStorage::Shared,
                                     "experts/jobs");
  impl.sources = backend.allocateBuffer(uint64_t{shape.experts} * 3 * sizeof(uint64_t), metal::BufferStorage::Shared,
                                        "experts/sources");
  impl.grid = backend.allocateBuffer(4 * sizeof(uint32_t), metal::BufferStorage::Shared, "experts/grid");
  std::memset(impl.grid.contents(), 0, 4 * sizeof(uint32_t));
  impl.routedIn.assign(shape.experts, 0);
  impl.waiting.assign(shape.experts, 0);
  impl.counters.capBytes += fixed;
  impl.counters.floorBytes += fixed;
  impl.counters.stagingBytes = impl.staging.sizeBytes();
  impl.counters.capChunks = size.chunkLayout.chunks;
  impl.counters.experts = uint64_t{blocks} * shape.experts;
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
  streamed.selectionBias = weights.selectionBias;
  const Impl::Size &size = impl.size;
  // The routed segments give the slots' format and size in the block's
  // layout; their planes are its layout's views of the chunks.
  const std::array<ops::QuantizedSegment, 3> &routed = size.routed[streamed.layout];
  weights.gate.routed = routed[0];
  weights.up.routed = routed[1];
  weights.down.routed = routed[2];
  weights.chunks = size.chunks[streamed.layout];
  weights.slotTable = impl.backend.view(impl.tables, uint64_t{block} * impl.shape.experts * sizeof(uint32_t),
                                        uint64_t{impl.shape.experts} * sizeof(uint32_t));
  weights.slots = size.slots;
  // The experts of a store of one block stay in their slots: nothing loads
  // them.
  weights.stream = impl.streams() ? this : nullptr;
  weights.streamBlock = block;
}

void ExpertStore::addLoad(metal::CommandGraph &graph, std::span<const ops::MoeLoadPart> parts,
                          uint32_t block) const {
  const Impl &impl = *impl_;
  const Impl::Block &loaded = impl.blocks.at(block);
  const ops::MoeShape shape = impl.shape;
  if (parts.empty()) throw std::invalid_argument("an expert load serves no step");
  if (!impl.streams()) throw std::invalid_argument("a block of a store of one block does not load");
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
      phase == ops::MoePhase::Decode && block + 1 < impl.blocks.size() && impl.streams();
  if (predicted) {
    if (parts.size() != 1) throw std::invalid_argument("a decode load serves one step");
    const Impl::Block &next = impl.blocks[block + 1];
    ops::MoE::addRoutes(graph, parts.front().buffers->input, next.router, next.scalarGate, impl.predictionScores,
                        impl.predictionRoutes, impl.predictionWeights, rows, shape,
                        parts.front().plan->configuration().ggufRouterTile, next.selectionBias);
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
  for (uint32_t round = 0; round < rounds; ++round) {
    graph.host(
        [impl = impl_, block, phase, rows, predicted, tiles, round, rounds, verifyRows] {
          impl->load(block, phase, rows, predicted, tiles, round, rounds, verifyRows);
        },
        timed, gpu, startAhead);
    ops::MoE::addExpertLoad(graph, impl.staging, impl.size.chunks[loaded.layout], impl.jobs, impl.sources, loaded.load,
                            static_cast<uint32_t>(std::min<uint64_t>(routable, impl.places.count)),
                            metal::IndirectGrid{impl.grid, 0});
  }
  // A long chunk routes nearly every expert of the next block too: once the
  // GPU has repacked the last round, the next block's experts are read ahead
  // into the whole staging, while the GPU runs this block's experts and the
  // next block's mixing, as many as the next load would otherwise read
  // with the GPU waiting.
  if (impl.longChunk(phase, rows) && block + 1 < impl.blocks.size() && impl.streams())
    graph.host([impl = impl_, block, phase] { impl->readBlockAhead(block + 1, phase); }, timed,
               metal::HostStepGpu::Idle, startAhead);
}

bool ExpertStore::addSplitLoadStart(metal::CommandGraph &graph, const ops::MoeLoadPart &part, uint32_t block,
                                    ops::MoeSplitTiles &tiles) const {
  const Impl &impl = *impl_;
  if (part.plan->phase() != ops::MoePhase::Decode || !impl.streams() ||
      impl.routable(part.plan->rows()) > impl.places.count || part.plan->maximumTiles() > impl.splitCapacity)
    return false;
  const ops::MoeShape shape = impl.shape;
  const uint32_t rows = part.plan->rows();
  // The next block's routes of this block's input, as addLoad predicts them.
  const bool predicted = block + 1 < impl.blocks.size() && impl.streams();
  if (predicted) {
    const Impl::Block &next = impl.blocks[block + 1];
    ops::MoE::addRoutes(graph, part.buffers->input, next.router, next.scalarGate, impl.predictionScores,
                        impl.predictionRoutes, impl.predictionWeights, rows, shape,
                        part.plan->configuration().ggufRouterTile, next.selectionBias);
  }
  tiles = {impl.splitTiles[0], impl.splitTiles[1]};
  const Impl::PartTiles partTiles{part.buffers->scratch.tileDescriptors, part.buffers->scratch.tileCount};
  graph.host(
      [impl = impl_, block, rows, partTiles] { impl->splitStart(block, rows, partTiles); },
      Impl::timedDecode(impl_), metal::HostStepGpu::Awake);
  return true;
}

void ExpertStore::addSplitLoadFinish(metal::CommandGraph &graph, const ops::MoeLoadPart &part,
                                     uint32_t block) const {
  const Impl &impl = *impl_;
  const Impl::Block &loaded = impl.blocks.at(block);
  const uint32_t rows = part.plan->rows();
  const bool predicted = block + 1 < impl.blocks.size() && impl.streams();
  graph.host(
      [impl = impl_, block, rows, predicted, verifyRows = part.buffers->verifyRows] {
        impl->splitFinish(block, rows, predicted, verifyRows);
      },
      Impl::timedDecode(impl_), metal::HostStepGpu::Awake, [impl = impl_] { impl->startAhead(); });
  ops::MoE::addExpertLoad(graph, impl.staging, impl.size.chunks[loaded.layout], impl.jobs, impl.sources, loaded.load,
                          static_cast<uint32_t>(std::min<uint64_t>(impl.routable(rows), impl.places.count)),
                          metal::IndirectGrid{impl.grid, 0});
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
  // readers end the pieces they took, and it is forgotten; so do the reads of
  // a split load whose finish did not run (its command failed).
  impl.retireAhead();
  if (impl.demand) impl.readers->stop(*std::exchange(impl.demand, nullptr));
  impl.ahead.block = impl.plan.block = kNoSlot;
  for (Impl::Block &block : impl.blocks) std::ranges::fill(block.slot, kNoSlot);
  std::memset(impl.tables.contents(), 0, impl.tables.sizeBytes());
  // The chunks past the live ones were given back already.
  Impl::Size &size = impl.size;
  for (uint32_t chunk = 0; chunk < size.liveChunks(); ++chunk) impl.backend.releaseMemory(size.buffers[chunk]);
  std::ranges::fill(size.ownerBlock, kNoSlot);
  std::ranges::fill(size.ownerExpert, kNoSlot);
  size.recycling.reset();
  impl.backend.releaseMemory(impl.staging);
  impl.forgetRepack();
  impl.counters.resident = 0;
  impl.released = true;
  impl.countLive();
  impl.publish();
}

bool ExpertStore::restore(const metal::AllocationAdmission &admit, const metal::AllocationAdmission &optional) {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  if (!impl.released) throw std::logic_error("expert memory is not released");
  // The floor chunks and the staging come back in one part, or none of them.
  Impl::Size &size = impl.size;
  std::vector<const metal::MetalBuffer *> floor{&impl.staging};
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
  size.keep(size.chunkLayout.floorChunks);
  size.recycling.resize(size.chunkLayout.first(size.chunkLayout.floorChunks));
  // Each elastic chunk comes back in a part of its own while `optional`
  // admits it, as a regrow takes them; the first refused stays released with
  // those after it, which a regrow takes later, so that neither the room the
  // cache may have taken while the store was smaller nor a host too short to
  // keep them holds the restore back.
  while (size.liveChunks() < size.chunkLayout.chunks)
    if (!impl.grow(optional)) break;
  // Every slot free, the lowest claimed first.
  size.recycling.reset();
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
  const uint32_t live = impl.size.liveChunks();
  return live > impl.size.chunkLayout.floorChunks ? impl.shrink(live - 1) : 0;
}

uint64_t ExpertStore::releaseToFloor() {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  if (!impl.mayShrink()) return 0;
  const uint32_t floor = impl.size.chunkLayout.floorChunks;
  return impl.size.liveChunks() > floor ? impl.shrink(floor) : 0;
}

uint64_t ExpertStore::growthBytes() const {
  std::lock_guard lock(impl_->mutex);
  const Impl &impl = *impl_;
  if (impl.released || !impl.failure.empty()) return 0;
  const Impl::Size &size = impl.size;
  const uint32_t live = size.liveChunks();
  return live < size.chunkLayout.chunks ? size.buffers[live].sizeBytes() : 0;
}

metal::AllocationResult ExpertStore::growTowardCap(const metal::AllocationAdmission &admit) {
  std::lock_guard lock(impl_->mutex);
  Impl &impl = *impl_;
  const Impl::Size &size = impl.size;
  if (impl.released || !impl.failure.empty() || impl.backend.commandInFlight() ||
      size.liveChunks() == size.chunkLayout.chunks)
    return {};
  if (const metal::AllocationResult admitted = impl.grow(admit); !admitted) return admitted;
  ++impl.counters.regrows;
  impl.countLive();
  impl.publish();
  logLine("Expert store took back a chunk of ", size.buffers[size.liveChunks() - 1].sizeBytes() >> 20,
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

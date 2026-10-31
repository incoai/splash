#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/MoE.h"
#include "metal/abi/QuantFormat.h"
#include "ops/Linear.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace splash::ops {

// How a MoE block's router weights the top-k experts it selects, the
// highest scores first and the lower expert id among equal scores: Softmax,
// by the softmax of their scores; SigmoidNormalized, by each one's
// sigmoid(score) over the sum of theirs plus 1e-20, in fp32. A sigmoid
// router selects by score as it would by sigmoid(score), which is monotonic.
enum class MoeRouter : uint8_t { Softmax, SigmoidNormalized };

// The routings the MoE kernels compile (kernels/shared/moe.metal,
// kernels/shared/moe_gguf.metal, kernels/decode/linear_gguf_sgmatrix.metal):
// a router with a shared expert of sharedSlices slices behind the block's
// scalar gate or none (MoeShape), and their kernels' name suffixes: the
// selection's, which reads all three, and slicesSuffix of the grouping and
// the expert passes, which read the slices alone.
struct CompiledMoeRouting final {
  MoeRouter router;
  uint32_t sharedSlices;
  bool sharedGate;
  std::string_view selectSuffix;
  std::string_view slicesSuffix;
};
inline constexpr std::array kCompiledMoeRoutings{
    CompiledMoeRouting{MoeRouter::Softmax, 1, true, "", ""},
    // Flint's: a sigmoid router, and an ungated shared expert twice the
    // routed experts' width.
    CompiledMoeRouting{MoeRouter::SigmoidNormalized, 2, false, "_sigmoid_s2", "_s2"},
};

// The shared expert is sharedSlices experts every row visits after its
// routed ones, ids experts to experts + sharedSlices - 1: each has the
// routed experts' intermediate width and runs through the same grouped
// tiles, and a shared expert sharedSlices times that wide is their sum
// (BlockExpertProjection). With sharedGate each slice's routing weight is the
// sigmoid of the row's product with the block's scalar gate, else 1.
struct MoeShape final {
  uint32_t hiddenSize = 0;
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  // The format of most routed expert weights, which picks the expert tile on
  // Apple9 GPUs of more than 10 cores (moeGgufTile).
  uint32_t expertFormat = GGUF_FMT_COUNT;
  // Whether the register tile takes every routed and shared expert weight
  // (quant_register_tensor).
  bool registerFormats = true;
  MoeRouter router = MoeRouter::Softmax;
  uint32_t sharedSlices = 1;
  bool sharedGate = true;
  // The format of the router's rows (QuantizedSegment::rows), which picks its
  // float tile (Linear::ggufFloatTile).
  uint32_t routerFormat = QuantizedSegment::kFloat32;

  [[nodiscard]] constexpr bool valid() const noexcept {
    // Experts the routing kernels' slots hold, of an intermediate width
    // that fills 64-column tiles, routed as a kernel compiles.
    return hiddenSize && hiddenSize % 256 == 0 && experts && experts <= SPLASH_MOE_WIDE_EXPERT_SLOTS &&
           expertsPerToken && expertsPerToken <= experts &&
           expertIntermediateSize && expertIntermediateSize % 64 == 0 && compiledRouting();
  }
  // The compiled routing of the router and the shared slices, or none.
  [[nodiscard]] constexpr const CompiledMoeRouting *compiledRouting() const noexcept {
    for (const CompiledMoeRouting &compiled : kCompiledMoeRoutings)
      if (compiled.router == router && compiled.sharedSlices == sharedSlices && compiled.sharedGate == sharedGate)
        return &compiled;
    return nullptr;
  }
  // The router's scores per row: the expert slots of the select and group
  // kernels that route the block (metal/abi/MoE.h), one thread each.
  [[nodiscard]] constexpr uint32_t routerWidth() const noexcept {
    return experts > SPLASH_MOE_EXPERT_SLOTS ? SPLASH_MOE_WIDE_EXPERT_SLOTS : SPLASH_MOE_EXPERT_SLOTS;
  }
  // Routed experts followed by the shared slices.
  [[nodiscard]] constexpr uint32_t routesPerToken() const noexcept {
    return expertsPerToken + sharedSlices;
  }
  bool operator==(const MoeShape &) const = default;
};

// A GGUF expert projection: the routed experts' format and size, slots * N
// rows of K inputs, whose planes the block's chunks hold (ExpertChunks; a
// resident block's routed planes are its one chunk's), and the shared
// expert's segment, whose format may differ, of its sharedSlices slices' N
// rows each in their order (the rows of slice s of a shared gate or up, rows
// s * N on; the column window s of a shared down, which a planner lays out
// so: gguf::TensorRows::windows).
struct BlockExpertProjection final {
  QuantizedSegment routed;
  QuantizedSegment shared;
};

// Where a block's routed experts are: the floor chunk, chunk 0, of
// floorSlots slots, and elastic chunks of chunkSlots slots each after it (the
// last may hold fewer), MOE_EXPERT_CHUNK_ADDRESSES views per chunk in the
// order of the expert kernels' address table (metal/abi/MoE.h), which the
// expert passes and loads reach through their addresses
// (metal::AddressTable). Slot s < floorSlots is local slot s of chunk 0; any
// other is local slot (s - floorSlots) % chunkSlots of chunk
// 1 + (s - floorSlots) / chunkSlots. A local slot l's planes start at row
// l * N of its chunk's regions. The blocks of a store size share the chunks
// and how many of them are live: a store gives its last chunks back while
// memory is short and takes them again, between commands (keep), and every
// copy's list names the live chunks alone, one list per count of them, so
// that the dispatches of a command share one address table, which never
// names memory given back. The chunks are checked view by view once, where
// `of` makes them, for the slots, floor and chunk slots and routed formats
// and sizes it records, which stay those of every chunk; a block's expert
// passes then only require its own slots and segments to be those
// (checkedFor). A default value is empty.
class ExpertChunks final {
public:
  ExpertChunks() = default;
  // The chunks `planes`, a floor chunk of floorSlots slots and elastic
  // chunks of chunkSlots slots each, which hold `slots` slots of the routed
  // segments gate, up and down: each planes of a GGUF_FMT_* format, slots * N
  // rows of K inputs. Throws unless there are 1 to MOE_EXPERT_MAX_CHUNKS
  // chunks, together `slots` slots or more, and each view of a chunk holds
  // its slots' planes (the floor chunk min(floorSlots, slots) slots, each
  // elastic chunk chunkSlots of them but the last that holds slots, which
  // holds the rest) as requireSegmentPlanes counts a segment of their rows. A
  // format without plane1 has its meta view in plane1's entry.
  [[nodiscard]] static ExpertChunks of(std::vector<metal::MetalBuffer> planes, uint32_t floorSlots,
                                       uint32_t chunkSlots, uint32_t slots, const QuantizedSegment &gate,
                                       const QuantizedSegment &up, const QuantizedSegment &down);

  // The live chunks' views, MOE_EXPERT_CHUNK_ADDRESSES per chunk in the
  // expert kernels' order; null for a default value.
  [[nodiscard]] const std::shared_ptr<const std::vector<metal::MetalBuffer>> &planes() const noexcept {
    static const std::shared_ptr<const std::vector<metal::MetalBuffer>> none;
    return live_ ? live_->planes[live_->count.load(std::memory_order_acquire) - 1] : none;
  }
  [[nodiscard]] uint32_t floorSlots() const noexcept { return floorSlots_; }
  [[nodiscard]] uint32_t chunkSlots() const noexcept { return chunkSlots_; }
  [[nodiscard]] bool empty() const noexcept { return !live_; }
  // Every chunk, and the first of them that are live, all of them until
  // keep() keeps fewer.
  [[nodiscard]] uint32_t chunks() const noexcept { return live_ ? static_cast<uint32_t>(live_->planes.size()) : 0; }
  [[nodiscard]] uint32_t liveChunks() const noexcept {
    return live_ ? live_->count.load(std::memory_order_acquire) : 0;
  }
  // Every copy's planes() names the first `chunks` chunks alone (1 to
  // chunks()) for the commands encoded from now on: a store keeps fewer
  // between commands, once no slot of its blocks lies past them, before it
  // gives the memory of the others back, and more once it took theirs again.
  // Throws std::invalid_argument for another count.
  void keep(uint32_t chunks);
  // Whether `of` checked them, at their floor and chunk slots, for `slots`
  // slots of routed segments of gate's, up's and down's formats and sizes.
  [[nodiscard]] bool checkedFor(uint32_t slots, const QuantizedSegment &gate, const QuantizedSegment &up,
                                const QuantizedSegment &down) const noexcept;

private:
  // A routed segment's format and size, which a check of the views reads.
  struct Routed final {
    uint32_t formatId = 0;
    uint32_t outputSize = 0;
    uint32_t inputSize = 0;
    bool operator==(const Routed &) const = default;
  };
  [[nodiscard]] static Routed routedOf(const QuantizedSegment &segment) noexcept {
    return {segment.formatId, segment.outputSize, segment.inputSize};
  }

  // The views of the first k chunks (planes[k - 1]), and k of the live
  // chunks, which every copy shares.
  struct Live final {
    std::vector<std::shared_ptr<const std::vector<metal::MetalBuffer>>> planes;
    std::atomic<uint32_t> count = 0;
  };
  std::shared_ptr<Live> live_;
  uint32_t floorSlots_ = 0;
  uint32_t chunkSlots_ = 0;
  uint32_t slots_ = 0;
  std::array<Routed, 3> routed_{}; // gate, up, down
};

struct MoeScratch;
struct MoeBuffers;
class MoePlan;
class MoeExpertStream;
enum class MoePhase : uint8_t { Prefill, Decode };

// The weights of one sparse MoE block, which the loaded model owns. The
// router is a float segment in the format its checkpoint holds it (a GGUF's
// F32 or BF16, which llama.cpp keeps unquantized, or a quantized tensor's
// native rows) and the shared expert's scalar gate F32 values, and both run
// in fp32; a block without the gate (!MoeShape::sharedGate) leaves it empty.
// The chunks hold `slots` experts, routed expert e in slot slotTable[e] (one
// uint32 per expert): every expert, in one chunk of the routed segments'
// planes, and the identity table when they are all resident
// (residentExpertChunks), or an expert store's slots and the block's table
// there when they stream (`stream`, the block's index in it `streamBlock`).
struct BlockMoeWeights final {
  QuantizedSegment router;           // [experts][hidden]
  QuantizedSegment sharedScalarGate; // [1][hidden]
  BlockExpertProjection gate;
  BlockExpertProjection up;
  BlockExpertProjection down;
  metal::MetalBuffer slotTable;
  uint32_t slots = 0;
  const MoeExpertStream *stream = nullptr;
  uint32_t streamBlock = 0;
  ExpertChunks chunks = {};
};

// The one chunk of a block whose routed segments hold all its `slots`
// experts, slot e expert e: its floor chunk, of floorSlots = chunkSlots =
// slots, gate's, up's and down's plane0, plane1Slot() and meta, which
// ExpertChunks::of checks.
[[nodiscard]] ExpertChunks residentExpertChunks(const BlockMoeWeights &weights);

// One MoE step of a block a load serves: its plan, and its buffers once the
// grouping wrote their tiles (MoeScratch::tileDescriptors and tileCount).
struct MoeLoadPart final {
  const MoeBuffers *buffers = nullptr;
  const MoePlan *plan = nullptr;
};

// Routed experts that stream into the slots of a store (model::ExpertStore),
// which owns the blocks' slot tables. addLoad adds, after the grouping of
// block `block`'s steps `parts` (all of one phase) and before their expert
// passes, what makes the experts their tiles name resident and the block's
// table name their slots; it may read a step's input (MoeBuffers::input).
// Several steps' parts load the union of their experts at once.
class MoeExpertStream {
public:
  virtual ~MoeExpertStream() = default;
  virtual void addLoad(metal::CommandGraph &graph, std::span<const MoeLoadPart> parts, uint32_t block) const = 0;
};

// Grouped-row scratch. Routes are sorted by expert into tiles of tileRows
// rows; every routed expert may leave one partially filled tile and no tile
// is empty, and each shared slice fills one tile per tileRows rows. The gate
// pass parks its output in expertOutput before the down pass overwrites it,
// so that field spans the wider of the two widths.
[[nodiscard]] constexpr uint32_t moeMaximumTiles(uint32_t rows, MoeShape shape,
                                                 uint32_t tileRows) noexcept {
  const uint32_t routed = rows * shape.expertsPerToken;
  return std::min(routed / tileRows + shape.experts, routed) +
         shape.sharedSlices * ((rows + tileRows - 1) / tileRows);
}

struct MoeWorkspace final {
  uint64_t selectedExpertsBytes = 0;
  uint64_t routingWeightsBytes = 0;
  uint64_t tileDescriptorsBytes = 0;
  uint64_t tileCountBytes = 0;
  uint64_t groupedRoutesBytes = 0;
  uint64_t routeRowsBytes = 0;
  uint64_t groupedInputBytes = 0;
  uint64_t expertIntermediateBytes = 0;
  uint64_t expertOutputBytes = 0;
  // The row sums of the Table16 tiles the GGUF register expert tile reads.
  uint64_t groupedSumsBytes = 0;
  bool operator==(const MoeWorkspace &) const = default;
};

// The scratch buffers of one MoE dispatch, sized by a plan's MoeWorkspace.
struct MoeScratch final {
  // rows * routesPerToken() routes: expert ids and fp32 routing weights.
  metal::MetalBuffer selectedExperts;
  metal::MetalBuffer routingWeights;
  // Sized by moeMaximumTiles(): tile descriptors, the tile count with the
  // expert passes' grids (MoeTileCount), the route at each grouped row,
  // each route's grouped row, and the grouped rows' inputs, intermediates and
  // outputs. The router parks its fp32 scores in groupedInput until the
  // gather claims it.
  metal::MetalBuffer tileDescriptors;
  metal::MetalBuffer tileCount;
  metal::MetalBuffer groupedRoutes;
  metal::MetalBuffer routeRows;
  metal::MetalBuffer groupedInput;
  metal::MetalBuffer expertIntermediate;
  metal::MetalBuffer expertOutput;
  // GGUF register plans: the row sums of the Table16 tiles in groupedInput.
  metal::MetalBuffer groupedSums;
};

// Each scratch buffer with the workspace field that sizes it, in field order.
struct MoeScratchField final {
  metal::MetalBuffer MoeScratch::*buffer;
  uint64_t MoeWorkspace::*bytes;
  const char *name;
};
inline constexpr std::array<MoeScratchField, 10> kMoeScratchFields{{
    {&MoeScratch::selectedExperts, &MoeWorkspace::selectedExpertsBytes, "MoE selected experts"},
    {&MoeScratch::routingWeights, &MoeWorkspace::routingWeightsBytes, "MoE routing weights"},
    {&MoeScratch::tileDescriptors, &MoeWorkspace::tileDescriptorsBytes, "MoE tile descriptors"},
    {&MoeScratch::tileCount, &MoeWorkspace::tileCountBytes, "MoE tile count"},
    {&MoeScratch::groupedRoutes, &MoeWorkspace::groupedRoutesBytes, "MoE grouped routes"},
    {&MoeScratch::routeRows, &MoeWorkspace::routeRowsBytes, "MoE route rows"},
    {&MoeScratch::groupedInput, &MoeWorkspace::groupedInputBytes, "MoE grouped input"},
    {&MoeScratch::expertIntermediate, &MoeWorkspace::expertIntermediateBytes, "MoE expert intermediate"},
    {&MoeScratch::expertOutput, &MoeWorkspace::expertOutputBytes, "MoE expert output"},
    {&MoeScratch::groupedSums, &MoeWorkspace::groupedSumsBytes, "MoE grouped sums"},
}};
static_assert(sizeof(MoeWorkspace) == kMoeScratchFields.size() * sizeof(uint64_t) &&
              sizeof(MoeScratch) == kMoeScratchFields.size() * sizeof(metal::MetalBuffer),
              "every scratch buffer is in kMoeScratchFields");
// The workspace fields of kMoeScratchFields, in its order.
inline constexpr auto kMoeWorkspaceFields = [] {
  std::array<uint64_t MoeWorkspace::*, kMoeScratchFields.size()> fields{};
  for (size_t i = 0; i < fields.size(); ++i) fields[i] = kMoeScratchFields[i].bytes;
  return fields;
}();

// The tile applies to grouping, gather and the expert passes together;
// changing it never changes the physical rows in a command. Plans run their
// passes on M8 tiles, or on M32 tiles to prefill (moeGgufPrefillTile).
enum class MoeExpertTile : uint8_t { M8 = 8, M32 = 32 };

// The expert tile of GGUF plans, which run grouped passes (gate, up with
// silu(gate), down; the staged tile runs gate and up in one pass) over the
// GGUF image: the half-staged tiles of
// kernels/shared/moe_gguf.metal, or Register, the exact register tile of
// kernels/decode/linear_gguf_sgmatrix.metal over Table16 tiles of the
// grouped rows (8-row tiles only). Apple9 GPUs of more than 10 cores run
// Register in both phases but for experts mostly in a format they stage. In
// decode, as Apple9's dense GGUF projections do (LinearGguf.cpp): its matrix
// operations share the FP32 pipe, where the register tile beats staging. In
// prefill it keeps the decode numerics at a cost: the staged 32-row tiles run
// long chunks faster.
enum class MoeGgufTile : uint8_t { Staged, Register };

// Apple9 stages experts mostly in a format apple9StagesFormat names: one
// 35B-shaped layer on a 40-core M3 Max decodes UD-Q2_K_XL's IQ2_XS and
// IQ3_XXS experts, and the IQ2, IQ3_XXS and IQ1 formats alone, 4-21% faster
// staged at B1-B4, where Q4_K/Q5_K, Q2_K and IQ4_XS experts take 3-34% longer.
// Apple9 GPUs of 10 cores or fewer (the base M3 and M4) stage experts in every
// format, in both phases: on a 10-core M4 such a layer in each of the 25
// formats the register tile takes, and one of Flash-Next's shape in 4-bit
// formats, take 12-45% less time staged at B1-B4 with rows routed at random,
// up to 17% less with each lane's rows routed among few experts (IQ4_NL at B1:
// 1% more) and 7-28% less on a 2048-row chunk (at most 4% more on 128- to
// 512-row chunks). The M3 Max's numbers predate the staged tile's one gate/up
// pass and its column tile for an expert's lone tile of one or two rows: at
// their code the M4 too ran Q4_K/Q5_K, Q2_K and IQ4_XS experts 2-15% faster on
// the register tile, so larger GPUs, unmeasured since, keep it.
[[nodiscard]] inline MoeGgufTile moeGgufTile(GpuFamilyClass family, uint32_t cores, MoeShape shape) noexcept {
  return family == GpuFamilyClass::Apple9 && cores > 10 && shape.registerFormats &&
                 !apple9StagesFormat(shape.expertFormat)
             ? MoeGgufTile::Register
             : MoeGgufTile::Staged;
}

// The rows of a GGUF prefill plan's tiles on the device's `tile`: 8 on the
// register tile. Staged: 8-row tiles while the chunk's routes average at most
// one row per expert (rows * topK <= experts), where the two tiles take the
// same time, and 32-row tiles beyond, which stream an expert's weights once
// for up to 32 of its rows (its tiles run 8-, 16- or 32-row matmuls by their
// live rows, moe_live_rows).
[[nodiscard]] constexpr MoeExpertTile moeGgufPrefillTile(MoeShape shape, uint32_t rows,
                                                         MoeGgufTile tile) noexcept {
  return tile == MoeGgufTile::Register ||
                 uint64_t{rows} * shape.expertsPerToken <= shape.experts
             ? MoeExpertTile::M8
             : MoeExpertTile::M32;
}

// The device's policy for a MoE plan, which the execution plans derive.
struct MoeConfig final {
  // M8 for decode plans; a prefill plan takes moeGgufPrefillTile's.
  MoeExpertTile expertTile = MoeExpertTile::M8;
  // The execution plans derive it from the GPU family, its planned core count
  // and the expert formats (moeGgufTile).
  MoeGgufTile ggufTile = MoeGgufTile::Staged;
  // The router's float tile; the execution plans derive it from the device,
  // the plan's rows and the router's format (Linear::ggufFloatTile).
  FloatTile ggufRouterTile = FloatTile::Simdgroup;
  bool operator==(const MoeConfig &) const = default;
};

// Constructed by the operator so workspace sizing and encoding use the same plan.
class MoePlan final {
public:
  [[nodiscard]] MoeShape shape() const noexcept { return shape_; }
  [[nodiscard]] uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] MoePhase phase() const noexcept { return phase_; }
  [[nodiscard]] MoeConfig configuration() const noexcept { return config_; }
  [[nodiscard]] uint32_t tileRows() const noexcept {
    return static_cast<uint32_t>(config_.expertTile);
  }
  [[nodiscard]] uint32_t maximumTiles() const noexcept { return maximumTiles_; }
  [[nodiscard]] const MoeWorkspace &workspace() const noexcept {
    return workspace_;
  }
  // Whether the expert passes run on the tiles' live rows
  // (kernels/decode/moe_rows.metal): decode plans on the register tile.
  [[nodiscard]] bool liveRows() const noexcept {
    return phase_ == MoePhase::Decode && config_.ggufTile == MoeGgufTile::Register;
  }

private:
  friend struct MoE;
  MoePlan(MoeShape shape, uint32_t rows, MoeConfig config, MoePhase phase);

  MoeShape shape_;
  uint32_t rows_;
  MoePhase phase_;
  MoeConfig config_;
  uint32_t maximumTiles_;
  MoeWorkspace workspace_;
};

struct MoeBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer residual;
  metal::MetalBuffer output;
  MoeScratch scratch;
  // A verify step's lanes' verify rows, a uint32 per lane (the draft
  // selector's, ops::DraftSelector::addVerifyRows): its routing routes each lane's verify
  // rows alone (moe_group_routes_verified*), and a streamed load predicts the
  // next block's experts from them. Empty, every row routes.
  metal::MetalBuffer verifyRows;
};

// Routes and executes grouped experts from immutable weight views.
struct MoE final {
  [[nodiscard]] static MoePlan prefillPlan(MoeShape shape, uint32_t rows, MoeConfig config);
  [[nodiscard]] static MoePlan decodePlan(MoeShape shape, uint32_t lanes, MoeConfig config);
  // A step's routing (addRouting: the router, the selection and the
  // grouping into tiles), the load of its streamed experts, and its routed
  // work (addRouted: the gather, the expert passes and the combine). Steps
  // that share a load run their routings, one load of the union, then their
  // routed work, each in its own scratch.
  static void add(metal::CommandGraph &graph, const MoeBuffers &buffers,
                  const BlockMoeWeights &weights, const MoePlan &plan);
  static void addRouting(metal::CommandGraph &graph, const MoeBuffers &buffers,
                         const BlockMoeWeights &weights, const MoePlan &plan);
  static void addRouted(metal::CommandGraph &graph, const MoeBuffers &buffers,
                        const BlockMoeWeights &weights, const MoePlan &plan);
  // The routing of `rows` rows of `input` by a block's `router` and
  // `sharedScalarGate` (unread without shape.sharedGate): the router's fp32
  // scores in `scores` (rows of shape.routerWidth()), then each row's top-k
  // experts and the shared slices with their routing weights in `selected`
  // and `routingWeights` (shape.routesPerToken() per row).
  static void addRoutes(metal::CommandGraph &graph, const metal::MetalBuffer &input, const QuantizedSegment &router,
                        const QuantizedSegment &sharedScalarGate, const metal::MetalBuffer &scores,
                        const metal::MetalBuffer &selected, const metal::MetalBuffer &routingWeights, uint32_t rows,
                        MoeShape shape, FloatTile routerTile);
  // A streamed experts' load (MoeExpertStream::addLoad): the gate, up and
  // down projections of up to `jobs` experts, job j's projection z's native
  // rows in `staging` from sources[3j + z] (bytes) on, repacked into the
  // planes of slot slots[j] of `chunks` as `params` lays them out (its
  // floor_slots and chunk_slots the chunks'). The host step before it writes
  // the sources, the slots and the grid (expertLoadGroups threadgroups, the
  // jobs it staged, 3) into `grid`. Any slot may be written, so the dispatch
  // reaches every chunk.
  static void addExpertLoad(metal::CommandGraph &graph, const metal::MetalBuffer &staging,
                            const ExpertChunks &chunks, const metal::MetalBuffer &slots,
                            const metal::MetalBuffer &sources, const MoeExpertLoadParams &params, uint32_t jobs,
                            const metal::IndirectGrid &grid);
  // The threadgroups that repack one job's largest projection.
  [[nodiscard]] static uint32_t expertLoadGroups(const MoeExpertLoadParams &params) noexcept;
};

} // namespace splash::ops

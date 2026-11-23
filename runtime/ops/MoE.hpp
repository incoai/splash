#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/MoE.h"
#include "ops/LaneRows.hpp"
#include "ops/Linear.hpp"
#include "ops/ShortConvolution.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
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
// kernels/shared/moe_gguf.metal):
// a router with a shared expert of sharedSlices slices behind the block's
// scalar gate or none (MoeShape), the most experts its selection and
// grouping are compiled for (MoeShape::routerWidth), and their kernels' name
// suffixes: the selection's, which reads all three, and slicesSuffix of the
// grouping and the expert passes, which read the slices alone. A biased
// selection (MoeShape::selectionBias) and a softcapped activation
// (MoeShape::gateSoftcap, upSoftcap; 0 for SiLU) are a routing's too, the
// activation's suffix the gate/up pass's.
struct CompiledMoeRouting final {
  MoeRouter router;
  uint32_t sharedSlices;
  bool sharedGate;
  uint32_t maximumExperts;
  std::string_view selectSuffix;
  std::string_view slicesSuffix;
  bool selectionBias = false;
  uint32_t gateSoftcap = 0;
  uint32_t upSoftcap = 0;
  std::string_view activationSuffix = "";
};
inline constexpr std::array kCompiledMoeRoutings{
    CompiledMoeRouting{MoeRouter::Softmax, 1, true, SPLASH_MOE_WIDE_EXPERT_SLOTS, "", ""},
    // Flint's: a sigmoid router, and an ungated shared expert twice the
    // routed experts' width.
    CompiledMoeRouting{MoeRouter::SigmoidNormalized, 2, false, SPLASH_MOE_EXPERT_SLOTS, "_sigmoid_s2", "_s2"},
    // A sigmoid router selecting by score plus a per-expert bias, no shared
    // expert, and SwiGLU softcapped at 4 (gate) and 25 (up).
    CompiledMoeRouting{MoeRouter::SigmoidNormalized, 0, false, SPLASH_MOE_EXPERT_SLOTS, "_sigmoid_biased_s0", "_s0",
                       true, 4, 25, "_softcap4_25"},
};

// A biased selection is the sigmoid router's, of a block without a shared
// expert (moe_route_select_f32_sigmoid_biased_s0).
static_assert(std::ranges::all_of(kCompiledMoeRoutings,
                                  [](const CompiledMoeRouting &compiled) {
                                    return !compiled.selectionBias ||
                                           (compiled.router == MoeRouter::SigmoidNormalized &&
                                            !compiled.sharedSlices && !compiled.sharedGate);
                                  }),
              "a biased selection is a sigmoid router's without a shared expert");

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
  MoeRouter router = MoeRouter::Softmax;
  uint32_t sharedSlices = 1;
  bool sharedGate = true;
  // Whether every block's experts are resident and in a routing and formats
  // the int8 expert passes take (moeInt8Experts and RuntimeGeometry::from):
  // its prefill plans then run the int8 passes and size their scratch
  // (MoeConfig::int8Passes).
  bool int8Experts = false;
  // A biased selection ranks the experts by sigmoid(score) plus each one's
  // bias (BlockMoeWeights::selectionBias) and scales its weights by
  // routedScale, which is 1 for every other selection.
  bool selectionBias = false;
  float routedScale = 1.0f;
  // The softcaps of a SwiGLU's gate factor and up value
  // (kernels/common/softcapped_swiglu.h); 0 for silu(gate) * up.
  uint32_t gateSoftcap = 0;
  uint32_t upSoftcap = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    // Experts of an intermediate width that fills 64-column tiles, and a
    // routing compiled for the router, shared slices, gate and experts.
    return hiddenSize && hiddenSize % 256 == 0 && experts && expertsPerToken && expertsPerToken <= experts &&
           expertIntermediateSize && expertIntermediateSize % 64 == 0 && compiledRouting() &&
           (selectionBias || routedScale == 1.0f);
  }
  // The compiled routing of the router, the shared slices and the experts,
  // or none.
  [[nodiscard]] constexpr const CompiledMoeRouting *compiledRouting() const noexcept {
    for (const CompiledMoeRouting &compiled : kCompiledMoeRoutings)
      if (compiled.router == router && compiled.sharedSlices == sharedSlices && compiled.sharedGate == sharedGate &&
          compiled.selectionBias == selectionBias && compiled.gateSoftcap == gateSoftcap &&
          compiled.upSoftcap == upSoftcap && experts <= compiled.maximumExperts)
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

// The weights of one sparse MoE block, which the loaded model owns. The
// router is a float segment in the format its checkpoint holds it (a GGUF's
// F32 or BF16, which llama.cpp keeps unquantized, or a quantized tensor's
// native rows) and the shared expert's scalar gate F32 values, and both run
// in fp32; a block without the gate (!MoeShape::sharedGate) leaves it empty.
// The chunks hold `slots` experts, routed expert e in slot slotTable[e] (one
// uint32 per expert): every expert, in one chunk of the routed segments'
// planes, and the identity table when they are all resident
// (residentExpertChunks), or an expert store's slots and the block's table
// there when they stream (`stream`, the block's index in it `streamBlock`),
// MOE_SLOT_ABSENT for an expert no slot holds. Of a streamed block, the
// grouping alone reads the slot table and copies it into the scratch the
// passes beside the load read (MoeScratch::expertSlots), and the passes after
// the load read loadSlotTable, the slot of each expert it routed, which its
// host step writes: no GPU work reads a line of either table while the step
// runs (metal::EventStep::beside). expertScales, F32 [3][experts], holds each
// routed expert's scale of its gate, up and down products, which the expert
// passes multiply their fp32 sums by before rounding them (the gate's and up's
// before the activation): a GGUF's per-expert .scale beside an experts tensor
// of any format, 1 where the source holds none.
struct BlockMoeWeights final {
  QuantizedSegment router;           // [experts][hidden]
  QuantizedSegment sharedScalarGate; // [1][hidden]
  BlockExpertProjection gate;
  BlockExpertProjection up;
  BlockExpertProjection down;
  metal::MetalBuffer expertScales;
  metal::MetalBuffer slotTable;
  uint32_t slots = 0;
  const MoeExpertStream *stream = nullptr;
  uint32_t streamBlock = 0;
  metal::MetalBuffer loadSlotTable = {};
  ExpertChunks chunks = {};
  // A biased selection's bias (MoeShape::selectionBias), F32 [experts].
  metal::MetalBuffer selectionBias = {};
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
// block `block`'s steps `parts` and before the expert passes that read the
// load slot table, what makes the experts their tiles name resident and both
// tables name their slots; it may read a step's input and the rows it routes
// (MoeBuffers::input, laneRows). Several steps' parts load the union of
// their experts at once. `beside` adds the dispatches that run while the load
// reads (metal::EventStep::beside): they may read the slots of the experts
// the grouping found resident, which the load leaves as they are, through its
// copy of the slot table (MoeScratch::expertSlots).
class MoeExpertStream {
public:
  virtual ~MoeExpertStream() = default;
  virtual void addLoad(metal::CommandGraph &graph, std::span<const MoeLoadPart> parts, uint32_t block,
                       const std::function<void()> &beside = {}) const = 0;
};

// The tiles of a step's routes, sorted by expert into tiles of at most
// tileRows routes: every routed expert may leave one partially filled tile
// and no tile is empty, and each shared slice fills one tile per tileRows
// rows.
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
  uint64_t expertSlotsBytes = 0;
  uint64_t groupedInputBytes = 0;
  uint64_t expertIntermediateBytes = 0;
  uint64_t expertOutputBytes = 0;
  uint64_t inputCodesBytes = 0;
  uint64_t inputScalesBytes = 0;
  uint64_t groupedCodesBytes = 0;
  uint64_t groupedScalesBytes = 0;
  bool operator==(const MoeWorkspace &) const = default;
};

// The scratch buffers of one MoE dispatch, sized by a plan's MoeWorkspace.
struct MoeScratch final {
  // rows * routesPerToken() routes: expert ids and fp32 routing weights.
  metal::MetalBuffer selectedExperts;
  metal::MetalBuffer routingWeights;
  // The tile descriptors, at most moeMaximumTiles(), and the tile count with
  // the expert passes' grids (MoeTileCount). Then the grouped rows, one per
  // route, each tile's following the tile before it's (MoeTileDescriptor):
  // the route at each grouped row, each route's grouped row, and the grouped
  // rows' inputs, intermediates and outputs, whatever the tile; the inputs
  // and intermediates hold tileRows - 1 rows more, which a last tile's matmul
  // may read past its live rows (moe_expert_slab.h). The router parks its
  // fp32 scores in groupedInput until the gather claims it.
  metal::MetalBuffer tileDescriptors;
  metal::MetalBuffer tileCount;
  metal::MetalBuffer groupedRoutes;
  metal::MetalBuffer routeRows;
  // A uint32 per expert: a streamed grouping's copy of the block's slot
  // table, which the GPU writes, for the passes beside its load.
  metal::MetalBuffer expertSlots;
  metal::MetalBuffer groupedInput;
  metal::MetalBuffer expertIntermediate;
  metal::MetalBuffer expertOutput;
  // The int8 passes' (MoeConfig::int8Passes): the chunk's input rows as int8
  // codes and their scales per 64 inputs, and the grouped rows' (the input's,
  // then the intermediate's), a row per route. Empty for plans without them.
  metal::MetalBuffer inputCodes;
  metal::MetalBuffer inputScales;
  metal::MetalBuffer groupedCodes;
  metal::MetalBuffer groupedScales;
};

// Each scratch buffer with the workspace field that sizes it, in field order.
struct MoeScratchField final {
  metal::MetalBuffer MoeScratch::*buffer;
  uint64_t MoeWorkspace::*bytes;
  const char *name;
};
inline constexpr std::array<MoeScratchField, 14> kMoeScratchFields{{
    {&MoeScratch::selectedExperts, &MoeWorkspace::selectedExpertsBytes, "MoE selected experts"},
    {&MoeScratch::routingWeights, &MoeWorkspace::routingWeightsBytes, "MoE routing weights"},
    {&MoeScratch::tileDescriptors, &MoeWorkspace::tileDescriptorsBytes, "MoE tile descriptors"},
    {&MoeScratch::tileCount, &MoeWorkspace::tileCountBytes, "MoE tile count"},
    {&MoeScratch::groupedRoutes, &MoeWorkspace::groupedRoutesBytes, "MoE grouped routes"},
    {&MoeScratch::routeRows, &MoeWorkspace::routeRowsBytes, "MoE route rows"},
    {&MoeScratch::expertSlots, &MoeWorkspace::expertSlotsBytes, "MoE expert slots"},
    {&MoeScratch::groupedInput, &MoeWorkspace::groupedInputBytes, "MoE grouped input"},
    {&MoeScratch::expertIntermediate, &MoeWorkspace::expertIntermediateBytes, "MoE expert intermediate"},
    {&MoeScratch::expertOutput, &MoeWorkspace::expertOutputBytes, "MoE expert output"},
    {&MoeScratch::inputCodes, &MoeWorkspace::inputCodesBytes, "MoE input codes"},
    {&MoeScratch::inputScales, &MoeWorkspace::inputScalesBytes, "MoE input scales"},
    {&MoeScratch::groupedCodes, &MoeWorkspace::groupedCodesBytes, "MoE grouped codes"},
    {&MoeScratch::groupedScales, &MoeWorkspace::groupedScalesBytes, "MoE grouped scales"},
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
// passes on M8 tiles, on M32 tiles where their routes average more than a row
// per expert, and on M64 tiles where a prefill chunk's average the device's
// MoeCalibration::wideTileRoutes (selectMoe). A row's bits are the same on
// every tile: the tiles' 8-, 16-, 32- and 64-row matmuls sum its products in
// one order (moe_live_rows).
enum class MoeExpertTile : uint8_t { M8 = 8, M32 = 32, M64 = 64 };

// The device's policy for a MoE plan, which the execution plans derive.
struct MoeConfig final {
  // A plan's by its routes per expert (selectMoe).
  MoeExpertTile expertTile = MoeExpertTile::M8;
  // The router's float tile; the execution plans derive it from the device,
  // the plan's rows and the router's format (selectFloatTile).
  FloatTile ggufRouterTile = FloatTile::Simdgroup;
  // A resident block's prefill passes on int8 activations where its expert
  // projections are af4g64 or af5g64 (MoE::addRouted, moe_int8.metal).
  bool int8Passes = false;
  bool operator==(const MoeConfig &) const = default;
};

// Constructed by the operator so workspace sizing and encoding use the same plan.
class MoePlan final {
public:
  [[nodiscard]] MoeShape shape() const noexcept { return shape_; }
  [[nodiscard]] uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] MoeConfig configuration() const noexcept { return config_; }
  [[nodiscard]] uint32_t tileRows() const noexcept {
    return static_cast<uint32_t>(config_.expertTile);
  }
  [[nodiscard]] uint32_t maximumTiles() const noexcept { return maximumTiles_; }
  [[nodiscard]] const MoeWorkspace &workspace() const noexcept {
    return workspace_;
  }

private:
  friend struct MoE;
  MoePlan(MoeShape shape, uint32_t rows, MoeConfig config);

  MoeShape shape_;
  uint32_t rows_;
  MoeConfig config_;
  uint32_t maximumTiles_;
  MoeWorkspace workspace_;
};

struct MoeBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer residual;
  metal::MetalBuffer output;
  MoeScratch scratch;
  // A decode step's `lanes` lanes: each lane's rows of the step, a
  // SplashLaneRows a lane, as the draft selector lays them out on the device
  // (ops::DraftSelector::addVerifyRows). Its routing routes its lanes' rows
  // alone, no row of no lane (moe_group_routes_verified*), and a streamed
  // load predicts the next MoE block's experts from the rows it routes. A
  // prompt chunk has no lanes: every row routes.
  uint32_t lanes = 0;
  metal::MetalBuffer laneRows;
};

// What a MoE block's combine reads (kernels/shared/moe.metal moe_combine):
// the grouped rows' expert outputs, each route's grouped row and routing
// weight, and the residual its sums start from, of `params`.
struct MoeCombine final {
  metal::MetalBuffer expertOutput;
  metal::MetalBuffer routeRows;
  metal::MetalBuffer routingWeights;
  metal::MetalBuffer residual;
  MoeCombineParams params{};
};

// A verify step's short convolution of its combined rows, which its combine
// runs in its own dispatch (MoE::addCombineConvolution): of static taps
// along the lanes' trees (`layout`, `rows`, `tree`; each lane's state
// stateIn[l]), onto buffers.residual, fp32, rounded into buffers.output
// (ShortConvOutput::RoundedSum), each row's combined input into the tape at
// the state's columns. The bits the combine into MoeBuffers::output and that
// convolution of it write, which is then unwritten.
struct MoeCombineConvolution final {
  ShortConvBuffers buffers;
  ShortConvShape shape;
  std::span<const metal::MetalBuffer> stateIn;
  ShortConvState state;
  metal::MetalBuffer layout;
  LayoutRows rows;
  metal::MetalBuffer tree;
  metal::MetalBuffer tape;
};

// Routes and executes grouped experts from immutable weight views.
struct MoE final {
  // The plan of a step of 1 to SPLASH_PREFILL_TOKEN_BUDGET rows.
  [[nodiscard]] static MoePlan plan(MoeShape shape, uint32_t rows, MoeConfig config);
  // A step's routing (addRouting: the router, the selection and the
  // grouping into tiles), the load of its streamed experts, and its routed
  // work (the gather, the expert passes and the combine). A streamed block's
  // runs the gather and the passes over the tiles of experts resident before
  // the load beside it (addResident, as MoeExpertStream::addLoad's `beside`),
  // the rest after it (addRouted); a resident block's runs all of it after
  // the routing (addRouted). Steps that share a load (a prefill group's
  // chunks, QwenTarget::addPrefillLayers) run their routings, one load of
  // the union beside the first step's resident work, then each step's work
  // through its combine before the next step's: they share the scratch their
  // gathers and passes write, all but their routing. With `convolution` (a
  // verify step's) the combine runs it on its rows (MoeCombineConvolution).
  static void addRouting(metal::CommandGraph &graph, const MoeBuffers &buffers,
                         const BlockMoeWeights &weights, const MoePlan &plan);
  static void addResident(metal::CommandGraph &graph, const MoeBuffers &buffers,
                          const BlockMoeWeights &weights, const MoePlan &plan);
  static void addRouted(metal::CommandGraph &graph, const MoeBuffers &buffers,
                        const BlockMoeWeights &weights, const MoePlan &plan,
                        const MoeCombineConvolution *convolution = nullptr);
  // A combine (`combine`, its rows the step's) running a verify step's
  // convolution of its rows (`convolution`, ShortConvolution::verifyParams'),
  // in one dispatch (moe_combine_convolution): the bits the combine into
  // bf16 rows and then the convolution of them write into
  // convolution.buffers.output and the tape, a lane's rows (at most
  // MOE_COMBINE_CONVOLUTION_ROWS) a threadgroup's, which pass their combined
  // values to the rows that tap them in threadgroup memory;
  // convolution.buffers.input is unread, and no combined row is written but
  // the tape's. The channels are the combine's hidden size, a multiple of
  // MOE_COMBINE_CONVOLUTION_CHANNELS.
  static void addCombineConvolution(metal::CommandGraph &graph, const MoeCombine &combine,
                                    const MoeCombineConvolution &convolution);
  // The routing of `rows` rows of `input` by a block's `router`,
  // `sharedScalarGate` (unread without shape.sharedGate) and `selectionBias`
  // (unread without shape.selectionBias): the router's fp32
  // scores in `scores` (rows of shape.routerWidth()), then each row's top-k
  // experts and the shared slices with their routing weights in `selected`
  // and `routingWeights` (shape.routesPerToken() per row).
  static void addRoutes(metal::CommandGraph &graph, const metal::MetalBuffer &input, const QuantizedSegment &router,
                        const QuantizedSegment &sharedScalarGate, const metal::MetalBuffer &selectionBias,
                        const metal::MetalBuffer &scores,
                        const metal::MetalBuffer &selected, const metal::MetalBuffer &routingWeights, uint32_t rows,
                        MoeShape shape, FloatTile routerTile);
  // A streamed experts' load (MoeExpertStream::addLoad): the gate, up and
  // down projections of up to `jobs` experts, job j's projection z staged in
  // `staging` as native rows or as the planes its checkpoint stores, plane s
  // from sources[(3j + z) * params.source_planes + s] (bytes) on
  // (MoeExpertLoadParams), repacked into the planes of slot slots[j] of
  // `chunks` as `params` lays them out (its floor_slots and chunk_slots the
  // chunks'). The host step before it writes
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

#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/MoE.h"
#include "metal/abi/QuantFormat.h"
#include "ops/Linear.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

namespace splash::ops {

// The shared expert is an expert every row visits, so it has the routed
// experts' intermediate width and runs through the same grouped tiles.
struct MoeShape final {
  uint32_t hiddenSize = 0;
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  // The format of most routed expert weights, which picks the expert tile on
  // Apple9 (moeGgufTile).
  uint32_t expertFormat = GGUF_FMT_COUNT;
  // Whether the register tile takes every routed and shared expert weight
  // (quant_register_tensor).
  bool registerFormats = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    // Experts the routing kernels' slots hold, of an intermediate width
    // that fills 64-column tiles.
    return hiddenSize && hiddenSize % 256 == 0 && experts && experts <= SPLASH_MOE_WIDE_EXPERT_SLOTS &&
           expertsPerToken && expertsPerToken <= experts &&
           expertIntermediateSize && expertIntermediateSize % 64 == 0;
  }
  // The router's scores per row: the expert slots of the select and group
  // kernels that route the block (metal/abi/MoE.h), one thread each.
  [[nodiscard]] constexpr uint32_t routerWidth() const noexcept {
    return experts > SPLASH_MOE_EXPERT_SLOTS ? SPLASH_MOE_WIDE_EXPERT_SLOTS : SPLASH_MOE_EXPERT_SLOTS;
  }
  // Routed experts followed by the shared expert.
  [[nodiscard]] constexpr uint32_t routesPerToken() const noexcept {
    return expertsPerToken + 1;
  }
  bool operator==(const MoeShape &) const = default;
};

// A GGUF expert projection: the routed experts in one segment of slots * N
// rows, routed expert e's planes from tile slot(e) * N / QUANT_TILE_ROWS on
// (moe_gguf_segment in kernels/common/moe_expert_slab.h, slot(e) from the
// block's slot table), and the shared expert's segment, whose format may
// differ.
struct BlockExpertProjection final {
  QuantizedSegment routed;
  QuantizedSegment shared;
};

struct MoeScratch;
struct MoeBuffers;
class MoePlan;
class MoeExpertStream;
enum class MoePhase : uint8_t { Prefill, Decode };

// The weights of one sparse MoE block, which the loaded model owns. The
// router and the shared expert's scalar gate are float tensors (llama.cpp
// keeps a GGUF's unquantized; an MLX target's load as their F32 values), and
// they run in fp32. The routed segments hold `slots` experts, routed expert e
// in slot slotTable[e] (one uint32 per expert): every expert and the identity
// table when they are all resident, or an expert store's slots and the
// block's table there when they stream (`stream`, the block's index in it
// `streamBlock`).
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
};

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
// is empty, and the shared expert fills one tile per tileRows rows. The gate
// pass parks its output in expertOutput before the down pass overwrites it,
// so that field spans the wider of the two widths.
[[nodiscard]] constexpr uint32_t moeMaximumTiles(uint32_t rows, MoeShape shape,
                                                 uint32_t tileRows) noexcept {
  const uint32_t routed = rows * shape.expertsPerToken;
  return std::min(routed / tileRows + shape.experts, routed) +
         (rows + tileRows - 1) / tileRows;
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
// grouped rows (8-row tiles only). Apple9 runs Register in both phases but
// for experts mostly in a format it stages. In decode, as its dense GGUF
// projections do (LinearGguf.cpp): its matrix operations share the FP32 pipe,
// where the register tile beats staging. In prefill it keeps the decode
// numerics at a cost: the staged 32-row tiles run long chunks faster.
enum class MoeGgufTile : uint8_t { Staged, Register };

// Apple9 stages experts mostly in a format apple9StagesFormat names: one
// 35B-shaped layer on a 40-core M3 Max decodes UD-Q2_K_XL's IQ2_XS and
// IQ3_XXS experts, and the IQ2, IQ3_XXS and IQ1 formats alone, 4-21% faster
// staged at B1-B4, where Q4_K/Q5_K, Q2_K and IQ4_XS experts take 3-34% longer.
[[nodiscard]] inline MoeGgufTile moeGgufTile(GpuFamilyClass family, MoeShape shape) noexcept {
  return family == GpuFamilyClass::Apple9 && shape.registerFormats && !apple9StagesFormat(shape.expertFormat)
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
  // The execution plans derive it from the GPU family.
  MoeGgufTile ggufTile = MoeGgufTile::Staged;
  // The tile of the F32 router; the execution plans derive it from the
  // device and the plan's rows (Linear::ggufFloatTile).
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
  // With `firstRow`, a decode plan's lanes route their rows before it to no
  // expert, and with `lastRow` those from it on neither (MoeGroupParams::
  // first_row, last_row); those rows' outputs are not theirs.
  static void add(metal::CommandGraph &graph, const MoeBuffers &buffers,
                  const BlockMoeWeights &weights, const MoePlan &plan, uint32_t firstRow = 0,
                  uint32_t lastRow = 0);
  static void addRouting(metal::CommandGraph &graph, const MoeBuffers &buffers,
                         const BlockMoeWeights &weights, const MoePlan &plan, uint32_t firstRow = 0,
                         uint32_t lastRow = 0);
  static void addRouted(metal::CommandGraph &graph, const MoeBuffers &buffers,
                        const BlockMoeWeights &weights, const MoePlan &plan);
  // The routing of `rows` rows of `input` by a block's `router` and
  // `sharedScalarGate`: the router's fp32 scores in `scores` (rows of
  // shape.routerWidth()), then each row's top-k experts and the shared
  // expert with their routing weights in `selected` and `routingWeights`
  // (shape.routesPerToken() per row).
  static void addRoutes(metal::CommandGraph &graph, const metal::MetalBuffer &input, const QuantizedSegment &router,
                        const QuantizedSegment &sharedScalarGate, const metal::MetalBuffer &scores,
                        const metal::MetalBuffer &selected, const metal::MetalBuffer &routingWeights, uint32_t rows,
                        MoeShape shape, FloatTile routerTile);
  // A streamed experts' load (MoeExpertStream::addLoad): the gate, up and
  // down projections of up to `jobs` experts, job j's projection z's native
  // rows in `staging` from sources[3j + z] (bytes) on, repacked into the
  // planes of slot slots[j] of `arena` as `params` lays them out. The host
  // step before it writes the sources, the slots and the grid
  // (expertLoadGroups threadgroups, the jobs it staged, 3) into `grid`.
  static void addExpertLoad(metal::CommandGraph &graph, const metal::MetalBuffer &staging,
                            const metal::MetalBuffer &arena, const metal::MetalBuffer &slots,
                            const metal::MetalBuffer &sources, const MoeExpertLoadParams &params, uint32_t jobs,
                            const metal::IndirectGrid &grid);
  // The threadgroups that repack one job's largest projection.
  [[nodiscard]] static uint32_t expertLoadGroups(const MoeExpertLoadParams &params) noexcept;
};

} // namespace splash::ops

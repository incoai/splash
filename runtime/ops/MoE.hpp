#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/MoE.h"
#include "metal/abi/QuantFormat.h"
#include "ops/Linear.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

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

  [[nodiscard]] constexpr bool valid() const noexcept {
    return hiddenSize && hiddenSize % 256 == 0 && experts && experts <= SPLASH_MOE_EXPERT_SLOTS &&
           expertsPerToken && expertsPerToken <= experts &&
           expertIntermediateSize && expertIntermediateSize % 256 == 0;
  }
  // Routed experts followed by the shared expert.
  [[nodiscard]] constexpr uint32_t routesPerToken() const noexcept {
    return expertsPerToken + 1;
  }
  bool operator==(const MoeShape &) const = default;
};

// A GGUF expert projection: every routed expert in one segment of experts *
// N rows, expert e's planes from tile e * N / 256 on (moe_gguf_segment in
// kernels/common/moe_expert_slab.h), and the shared expert's segment, whose
// format may differ.
struct BlockExpertProjection final {
  QuantizedSegment routed;
  QuantizedSegment shared;
};

// The weights of one sparse MoE block, which the loaded model owns. The
// router and the shared expert's scalar gate are float tensors (llama.cpp
// keeps a GGUF's unquantized; an MLX target's load as their F32 values), and
// they run in fp32.
struct BlockMoeWeights final {
  QuantizedSegment router;           // [experts][hidden]
  QuantizedSegment sharedScalarGate; // [1][hidden]
  BlockExpertProjection gate;
  BlockExpertProjection up;
  BlockExpertProjection down;
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
// silu(gate), down; the staged tile runs gate and up in one pass where they
// share their formats) over the GGUF image: the half-staged tiles of
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
  return family == GpuFamilyClass::Apple9 && !apple9StagesFormat(shape.expertFormat) ? MoeGgufTile::Register
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

enum class MoePhase : uint8_t { Prefill, Decode };

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
  [[nodiscard]] MoeConfig configuration() const noexcept { return config_; }
  [[nodiscard]] uint32_t tileRows() const noexcept {
    return static_cast<uint32_t>(config_.expertTile);
  }
  [[nodiscard]] uint32_t maximumTiles() const noexcept { return maximumTiles_; }
  [[nodiscard]] const MoeWorkspace &workspace() const noexcept {
    return workspace_;
  }
  // Whether every expert's tile reads the plan's rows in place, so no gather
  // runs (MoeGroupParams::in_place): rows that fill one 8-row tile on the
  // staged tile, a one-lane decode step or an 8-row prefill chunk. An 8-row
  // tile runs the 8-row matmul whatever rows it holds, so only the gather's
  // copy and dispatch go. Plans of more rows keep grouped tiles, which run
  // each expert on its routed rows alone.
  [[nodiscard]] bool rowsInPlace() const noexcept {
    return config_.expertTile == MoeExpertTile::M8 && rows_ == tileRows() &&
           config_.ggufTile == MoeGgufTile::Staged;
  }

private:
  friend struct MoE;
  MoePlan(MoeShape shape, uint32_t rows, MoeConfig config, MoePhase phase);

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
};

// Routes and executes grouped experts from immutable weight views.
struct MoE final {
  [[nodiscard]] static MoePlan prefillPlan(MoeShape shape, uint32_t rows, MoeConfig config);
  [[nodiscard]] static MoePlan decodePlan(MoeShape shape, uint32_t lanes, MoeConfig config);
  static void add(metal::CommandGraph &graph, const MoeBuffers &buffers,
                  const BlockMoeWeights &weights, const MoePlan &plan);
};

} // namespace splash::ops

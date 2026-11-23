#include "ops/ExecutionPlans.hpp"

#include "metal/abi/Gguf.h"
#include "ops/RelativeAttention.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace splash::ops {

ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device, Int8Activations int8Activations)
    : linear_(device, int8Activations, StepKind::Prefill), decodeLinear_(device, int8Activations, StepKind::Decode),
      draftLinear_(device, Int8Activations::Off, StepKind::Prefill),
      draftDecodeLinear_(device, Int8Activations::Off, StepKind::Decode) {}

namespace {

// The kernels the relative attention of a prompt chunk of `rows` rows after
// no history runs over `layout`'s pages, of each layer variant
// (Inkling2LoopTarget::planRelativeAttention): each dispatch's direct
// kernel, or its split and reduce.
std::string relativeKernels(const ExecutionPlans &plans, kv::Layout layout, uint32_t rows) {
  const uint32_t pages = (rows + kv::kPageTokens - 1) / kv::kPageTokens;
  const std::array sequence{PagedAttention::chunkParams(0, rows, pages * kv::kPageTokens, pages)};
  std::string kernels;
  for (const RelativeAttentionVariant variant :
       {RelativeAttentionVariant{.windowTokens = 0, .positionScaled = true},
        RelativeAttentionVariant{.windowTokens = layout.windowTokens, .positionScaled = false}}) {
    const RelativeAttentionPlan plan =
        RelativeAttention::plan(sequence, layout, plans.calibration().attention, plans.linear().int8Activations(),
                                RelativeAttention::sequenceWorkspace(layout), variant);
    for (const RelativeAttentionPlan::Dispatch &dispatch : plan.dispatches())
      kernels += plan.direct(dispatch) ? plan.directPipeline() + " "
                                       : plan.splitPipeline() + " " + plan.reducePipeline() + " ";
  }
  return kernels;
}

} // namespace

// A chunk's rows pick its projections' tile, and a tile's height names its
// kernels (gguf_decode_<format>_m<rows>_*, the register tile's _l<tiles>), so
// each tile and height a chunk takes is a class, and the prefill tile one
// more; and a MoE target's expert passes (selectMoe: the 8-, 32- or 64-row
// staged passes, or the int8 passes), each a class. A relative-attention
// target's chunk picks its attention's kernels by its rows too: after no
// history a split and a reduce up to 128 rows, and from 129 the direct
// kernel, its one split reading the chunk's pages (RelativeAttention::plan),
// over INT8 queries from 225 rows (8 pages) where INT8 pages take them on
// Apple10 and Apple11; each set a class. Only a chunk after a short history
// runs the direct kernel over bf16 queries, which no warmup ran before its
// 129-row class: a fresh 159-token prompt on the 16-core M5 Pro created
// relative_attention_q8_direct and its _w1023 while it waited. Every other
// kernel a chunk's rows pick today changes once along them (selectFloatTile,
// a MoE router's among them, the staged norm, the hyper-connection norm's
// quarters, and the Neural Engine split, which takes no class's first chunk:
// AneFfn::kMinimumRows), so a chunk of one row and one of the most take both
// sides. A class no warmup ran created its kernels while its request waited:
// 19 s for the 27B's cold 17-row chunk on a 16-core Apple9 plan.
std::vector<uint32_t> ExecutionPlans::prefillRowClasses(std::optional<std::pair<MoeShape, uint32_t>> moe,
                                                        std::optional<kv::Layout> relativeAttention) const {
  std::vector<uint32_t> first;
  std::tuple<LinearTile, uint32_t, MoeExpertTile, bool, std::string> last{};
  for (uint32_t rows = 1; rows <= SPLASH_PREFILL_TOKEN_BUDGET; ++rows) {
    const LinearPlan plan = linear_.plan({{GGUF_TILE_COLUMNS, GGUF_TILE_COLUMNS}, rows});
    const LinearTile tile = plan.configuration().tile;
    // Only the staged and register tiles' kernels are named by their height.
    const bool named = tile == LinearTile::GgufStaged || tile == LinearTile::GgufRegister;
    MoeConfig config{};
    if (moe)
      config = selectMoe(calibration(), linear_.int8Activations(), StepKind::Prefill, moe->first, moe->second, rows);
    std::tuple key{tile, named ? plan.storageRows() : 0, config.expertTile, config.int8Passes,
                   relativeAttention ? relativeKernels(*this, *relativeAttention, rows) : std::string()};
    if (first.empty() || key != last) first.push_back(rows);
    last = std::move(key);
  }
  // Warmup's chunk of the most rows (ServingRuntime::requireWarmup) warms the
  // class those rows start: the M5 Pro's 64-row expert passes of the 35B and
  // the 122B, whose second 2,048-row prefill took 0.6 s of the 35B's start.
  if (first.back() == SPLASH_PREFILL_TOKEN_BUDGET) first.pop_back();
  return first;
}

// A step's rows pick its projections' tile and height as a chunk's do
// (prefillRowClasses); a float projection's tile, which a step's rows move
// only at the accelerator tile's fewest rows, as they fill one 64-row tile
// (selectFloatTile); and a MoE target's expert tile (32-row tiles once its
// rows' routes outnumber its experts). Its router's tile changes with them
// on no device.
std::vector<uint32_t> ExecutionPlans::decodeRowClasses(uint32_t mostRows,
                                                       std::optional<std::pair<MoeShape, uint32_t>> moe) const {
  if (mostRows > SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("a decode step holds at most a verify block of each of the most lanes");
  const FloatTileCalibration &floats = calibration().floatTile;
  std::vector<uint32_t> first;
  std::tuple<LinearTile, uint32_t, bool, uint32_t> last{};
  for (uint32_t rows = 1; rows <= mostRows; ++rows) {
    const LinearPlan plan = decodeLinear_.plan({{GGUF_TILE_COLUMNS, GGUF_TILE_COLUMNS}, rows});
    const LinearTile tile = plan.configuration().tile;
    const bool named = tile == LinearTile::GgufStaged || tile == LinearTile::GgufRegister;
    MoeConfig config{};
    if (moe)
      config = selectMoe(calibration(), linear_.int8Activations(), StepKind::Decode, moe->first, moe->second, rows);
    const std::tuple key{tile, named ? plan.storageRows() : 0, floats.neuralAccelerator && rows >= floats.minimumRows,
                         uint32_t(config.expertTile)};
    if (first.empty() || key != last) first.push_back(rows);
    last = key;
  }
  return first;
}

AttentionPlan ExecutionPlans::attention(std::span<const kv::ChunkedPrefillParams> sequences, uint32_t queryHeads,
                                        kv::Layout layout, AttentionWorkspace capacity,
                                        AttentionVariant variant) const {
  return PagedAttention::plan(sequences, queryHeads, layout, calibration().attention, linear_.int8Activations(),
                              capacity, variant);
}

DraftAttentionPlan ExecutionPlans::draftAttention(DraftAttentionShape shape,
                                                 uint32_t lanes) const {
  return DraftAttention::plan(shape, lanes);
}

MoePlan ExecutionPlans::moe(MoeShape shape, uint32_t routerFormat, uint32_t rows, StepKind step) const {
  return MoE::plan(shape, rows, selectMoe(calibration(), linear_.int8Activations(), step, shape, routerFormat, rows));
}

AttentionWorkspace ExecutionPlans::sequenceAttentionWorkspace(uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::workspace(kAttentionSequenceSplits, queryHeads, layout);
}

AttentionWorkspace ExecutionPlans::laneAttentionWorkspace(uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::workspace(kv::kAttentionMaximumSplits, queryHeads, layout);
}

// The draft workspace is linear in the lanes: one lane's is every width's
// share.
DraftAttentionWorkspace ExecutionPlans::draftAttentionWorkspacePerLane(
    DraftAttentionShape shape) const {
  return DraftAttention::plan(shape, 1).workspace();
}

MoeWorkspace ExecutionPlans::moeWorkspace(MoeShape shape, uint32_t routerFormat, uint32_t maximumRows,
                                          StepKind step) const {
  // Validate the bound before iterating; every row is included even if a
  // future grouped layout's largest field is not monotone in row count.
  MoeWorkspace bound = moe(shape, routerFormat, maximumRows, step).workspace();
  for (uint32_t rows = 1; rows <= maximumRows; ++rows) {
    const MoeWorkspace required = moe(shape, routerFormat, rows, step).workspace();
    for (const auto field : kMoeWorkspaceFields) bound.*field = std::max(bound.*field, required.*field);
  }
  return bound;
}

// The gate scratch of the plan Linear::addSwiGlu runs over each rows of a
// decode step: the pair's up to one row group's rows, past them the
// up-with-gate pass's on the row groups, whose gate rows a pass of their own
// writes.
uint64_t ExecutionPlans::gateUpWorkspace(ProjectionShape shape, uint32_t maximumRows) const {
  uint64_t bound = 0;
  for (uint32_t rows = 1; rows <= maximumRows; ++rows) {
    const LinearEpilogue epilogue = rows <= kMaximumDecodeTileRows ? LinearEpilogue::GateUp : LinearEpilogue::UpWithGate;
    bound = std::max(bound,
                     decodeLinear_.plan({{shape.outputSize, shape.inputSize}, rows, epilogue}).gateScratchBytes());
  }
  return bound;
}

} // namespace splash::ops

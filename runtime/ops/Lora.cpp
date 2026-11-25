#include "ops/Lora.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/Lora.h"
#include "ops/BufferExtent.hpp"
#include "ops/Linear.hpp"

#include <stdexcept>
#include <utility>

namespace splash::ops::lora {
namespace {

metal::DispatchSize tiles(uint32_t columns, uint32_t rows) {
  return {columns / GGUF_STAGED_COLUMNS, (rows + kTileRows - 1) / kTileRows, 1};
}

// A kernel's B rows of a segment's `columns` columns.
void requireB(const metal::MetalBuffer &b, uint32_t columns, uint32_t rank) {
  if (columns % GGUF_STAGED_COLUMNS || !rank || rank % GGUF_STAGED_STEP)
    throw std::invalid_argument("a LoRA update takes whole column tiles and whole steps of rank");
  requireBytes(b, uint64_t{columns} * rank * sizeof(uint16_t), "LoRA B");
}

} // namespace

// The neural accelerator's tile takes at least 16 rows and inputs in
// multiples of 32.
FloatTile floatTile(const Linear &linear, uint32_t rows, uint32_t outputSize, uint32_t inputSize) {
  const FloatTile tile = linear.ggufFloatTile(SPLASH_PREFILL_TOKEN_BUDGET, outputSize, GGUF_FMT_BF16);
  return tile == FloatTile::NeuralAccelerator && (rows < 16 || inputSize % 32) ? FloatTile::Simdgroup : tile;
}

LoraRows addDown(metal::CommandGraph &graph, const Linear &linear, metal::MetalBuffer x, uint32_t rows,
                 const ProjectionLora *first, const ProjectionLora *second, metal::MetalBuffer t, bool decode) {
  if (first && !*first) first = nullptr;
  if (second && !*second) second = nullptr;
  if (!first) first = std::exchange(second, nullptr);
  if (!first) throw std::invalid_argument("no LoRA update to project");
  if (!rows || (second && second->a.inputSize != first->a.inputSize) || (decode && rows > kMaximumStagedRows))
    throw std::invalid_argument("invalid LoRA down projection");
  LoraRows result{std::move(t), first->values() + (second ? second->values() : 0), {0, first->values()}};
  const uint32_t inputSize = first->a.inputSize;
  requireBytes(x, uint64_t{rows} * inputSize * sizeof(uint16_t), "LoRA down input");
  requireBytes(result.t, rowsBytes(rows, result.stride), "LoRA t");
  // A decode step's few rows run one fragment of rows a threadgroup, K in
  // GGUF_FLOAT_FEW_SPLITS parts (gguf_float_few_bf16_bf16).
  if (decode) {
    const GgufFloatParams params{rows, inputSize, {first->values(), second ? second->values() : 0},
                                 {result.stride, result.stride}, {result.offsets[0], result.offsets[1]}};
    graph.add("gguf_float_few_bf16_bf16",
              {std::move(x), first->a.plane0, result.t, second ? second->a.plane0 : first->a.plane0, result.t},
              params, {(params.output_size[0] + params.output_size[1]) / 8, (rows + 7) / 8, 1},
              {32 * GGUF_FLOAT_FEW_SPLITS, 1, 1});
    return result;
  }
  FloatProjection projections[2];
  uint32_t count = 0;
  for (const ProjectionLora *update : {first, second})
    if (update) {
      projections[count] = {&update->a, result.t, result.stride, result.offsets[count],
                            floatTile(linear, rows, update->values(), inputSize)};
      ++count;
    }
  addGgufFloat(graph, std::move(x), {projections, count}, rows, FloatOutput::BFloat16);
  return result;
}

void addTile(metal::CommandGraph &graph, const Projection &p, const LoraRows &down, metal::MetalBuffer sums,
             metal::MetalBuffer residual, metal::MetalBuffer y, uint32_t rows) {
  const ProjectionLora &update = p.lora;
  const std::vector<QuantizedSegment> &segments = p.blocks().segments;
  if (!update || update.segments.size() != segments.size() || !rows)
    throw std::invalid_argument("a LoRA update holds one segment per segment of its projection");
  const uint64_t bytes = uint64_t{rows} * p.outputSize * sizeof(uint16_t);
  requireBytes(sums, bytes, "LoRA sums");
  requireBytes(y, bytes, "LoRA output");
  requireBytes(down.t, rowsBytes(rows, down.stride), "LoRA t");
  const bool adds = static_cast<bool>(residual);
  if (adds) requireBytes(residual, bytes, "LoRA residual");
  for (size_t index = 0; index < segments.size(); ++index) {
    const QuantizedSegment &s = segments[index];
    const ProjectionLora::Segment &segment = update.segments[index];
    // Rows that take no residual are their plan's as they are without an update.
    if (!segment.b && !adds) continue;
    if (segment.b) requireB(segment.b, s.outputSize, update.rank);
    const GgufLora tile{update.rank, down.stride, segment.b ? 1u : 0u,
                        {down.offsets[0] + segment.slot * update.rank, 0, 0}};
    graph.add("lora_tile", {sums, down.t, segment.b ? segment.b : down.t, adds ? residual : sums, y},
              LoraTileParams{rows, p.outputSize, s.columnOffset, adds ? LORA_TILE_RESIDUAL : LORA_TILE_DITHER,
                             update.seed + static_cast<uint32_t>(index), tile},
              tiles(s.outputSize, rows), {32, 1, 1});
  }
}

void addSwiGluTile(metal::CommandGraph &graph, const Projection &gate, const Projection &up, const LoraRows &down,
                   metal::MetalBuffer gateRows, metal::MetalBuffer upRows, metal::MetalBuffer h, uint32_t rows) {
  const uint32_t width = up.outputSize;
  if (gate.blocks().segments.size() != 1 || up.blocks().segments.size() != 1 || gate.outputSize != width ||
      gate.blocks().segments.front().outputSize != width || up.blocks().segments.front().outputSize != width ||
      (!gate.lora && !up.lora) || (gate.lora && up.lora && gate.lora.rank != up.lora.rank) || !rows)
    throw std::invalid_argument("a LoRA gate/up pair is one tensor each, of one rank");
  const uint64_t bytes = uint64_t{rows} * width * sizeof(uint16_t);
  requireBytes(gateRows, bytes, "LoRA gate rows");
  requireBytes(upRows, bytes, "LoRA up rows");
  requireBytes(h, bytes, "LoRA intermediate rows");
  requireBytes(down.t, rowsBytes(rows, down.stride), "LoRA t");
  // The rows t hold the gate's update first where it has one (addDown).
  const ProjectionLora::Segment *g = gate.lora ? &gate.lora.segments.front() : nullptr;
  const ProjectionLora::Segment *u = up.lora ? &up.lora.segments.front() : nullptr;
  const uint32_t rank = g ? gate.lora.rank : up.lora.rank;
  for (const ProjectionLora::Segment *segment : {g, u})
    if (segment) requireB(segment->b, width, rank);
  // A slot without a correction binds a B no tile reads.
  const metal::MetalBuffer &spare = g ? g->b : u->b;
  const GgufLora tile{rank, down.stride, (g ? 1u : 0u) | (u ? 2u : 0u),
                      {g ? down.offsets[0] + g->slot * rank : 0,
                       u ? down.offsets[g ? 1 : 0] + u->slot * rank : 0, 0}};
  graph.add("lora_swiglu_tile",
            {std::move(gateRows), std::move(upRows), down.t, g ? g->b : spare, u ? u->b : spare, std::move(h)},
            LoraTileParams{rows, width, 0, 0, 0, tile}, tiles(width, rows), {32, 1, 1});
}

} // namespace splash::ops::lora

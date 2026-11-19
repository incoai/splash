#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"
#include "ops/Weights.hpp"

#include <cstdint>

namespace splash::ops {

// The rows t = A x of one or two projections' LoRA updates
// (ProjectionLora) over one input: bf16 rows of `stride` values, each
// projection's from its offset in a row.
struct LoraRows final {
  metal::MetalBuffer t;
  uint32_t stride = 0;
  uint32_t offsets[2] = {0, 0};
};

// The kernels of a LoRA update, which Linear runs wherever it runs an updated
// projection (Linear::add, Linear::addSwiGlu); the Neural Engine FFN split
// runs them over its own rows.
namespace lora {

// The rows of a tile of the kernels that add B t (lora_tile,
// lora_swiglu_tile): they read the rows t of whole tiles.
inline constexpr uint32_t kTileRows = 32;
// The bytes of the rows t of `rows` rows of `values` values.
[[nodiscard]] constexpr uint64_t rowsBytes(uint64_t rows, uint32_t values) noexcept {
  return (rows + kTileRows - 1) / kTileRows * kTileRows * values * sizeof(uint16_t);
}

// The float projection's tile of the rows t = A x, or of B t, of a chunk of
// `rows` rows, `outputSize` values of `inputSize` bf16 inputs each: the tile
// of a full chunk whatever the chunk's rows, so that a row's values do not
// depend on how many rows its chunk holds.
[[nodiscard]] FloatTile floatTile(const Linear &linear, uint32_t rows, uint32_t outputSize, uint32_t inputSize);
// t = A x of `rows` rows of x into `t`: of `first`'s update and, of a gate/up
// pair, `second`'s after it; either may be null or hold no update. `decode`:
// the rows of a plan whose tile adds B t (Linear::takesLora), at most
// kMaximumDecodeTileRows, on the kernel of a decode step's few rows; any
// other plan's on the float projection.
[[nodiscard]] LoraRows addDown(metal::CommandGraph &graph, const Linear &linear, metal::MetalBuffer x, uint32_t rows,
                               const ProjectionLora *first, const ProjectionLora *second, metal::MetalBuffer t,
                               bool decode);
// y = bf16(sums + B t) over the segments of `p`, `down` holding the rows t of
// its update (lora_tile): with `residual`, every segment's rows take it;
// without, the updated segments' rows are rounded with a dither, in place
// where `sums` is `y`. Rows of p.outputSize values.
void addTile(metal::CommandGraph &graph, const Projection &p, const LoraRows &down, metal::MetalBuffer sums,
             metal::MetalBuffer residual, metal::MetalBuffer y, uint32_t rows);
// h = bf16((up + B t) * silu(gate + B t)) of a pair of one tensor each, `down`
// holding the gate's rows t and the up's (lora_swiglu_tile); `upRows` may be
// `h`.
void addSwiGluTile(metal::CommandGraph &graph, const Projection &gate, const Projection &up, const LoraRows &down,
                   metal::MetalBuffer gateRows, metal::MetalBuffer upRows, metal::MetalBuffer h, uint32_t rows);

} // namespace lora
} // namespace splash::ops

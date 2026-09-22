#include "FlashDenseSmallRows.hpp"

#include "metal/abi/FlashDenseSmallRows.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {
uint64_t product(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::invalid_argument("Flash small-row dense byte extent overflows");
  return a * b;
}
void requireBuffer(const metal::MetalBuffer &b, uint64_t bytes, const char *name) {
  if (!b || !bytes || b.sizeBytes() < bytes || !b.contents())
    throw std::invalid_argument(std::string("Flash small-row dense invalid Shared ") + name);
}
bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  if (a.sameView(b)) return true;
  const uintptr_t first = reinterpret_cast<uintptr_t>(a.contents());
  const uintptr_t second = reinterpret_cast<uintptr_t>(b.contents());
  if (!first || !second || first > UINTPTR_MAX - a.sizeBytes() ||
      second > UINTPTR_MAX - b.sizeBytes())
    throw std::invalid_argument("Flash small-row dense invalid Shared address extent");
  return first < second + b.sizeBytes() && second < first + a.sizeBytes();
}
} // namespace

FlashDenseSmallRowsWorkspace::FlashDenseSmallRowsWorkspace(metal::MetalBackend &backend,
                                                           uint32_t maximumInputSize)
    : backend_(&backend), maximumInputSize_(maximumInputSize) {
  if (!maximumInputSize || maximumInputSize > 32768 || maximumInputSize % 32)
    throw std::invalid_argument("Flash small-row dense maximum K must be aligned32 and1..32768");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  const uint64_t bytes = product(product(16, maximumInputSize), 2);
  paddedInput_ = backend.allocateBuffer((bytes + 16383) & ~uint64_t{16383},
      metal::BufferStorage::Shared, "flash-small-row-dense-positive-zero-padding");
  const uint64_t after = backend.memoryStats().allocatedBytes;
  if (after < before) throw std::logic_error("Flash small-row workspace allocation ledger regressed");
  allocatedBytes_ = after - before;
}

void addDenseBF16SmallRows(metal::MetalBackend &backend, metal::CommandGraph &graph,
                           metal::MetalBuffer input, const FlashTensor &weight,
                           metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                           uint32_t rows, FlashDenseSmallRowsWorkspace &workspace,
                           FlashDenseSmallRowsTile tile) {
  if (&backend != workspace.backend_ || !rows || rows > 16 ||
      weight.dtype != FlashDType::BF16 || weight.shape.size() != 2 ||
      !weight.shape[0] || weight.shape[0] > UINT32_MAX || weight.shape[0] % 64 ||
      !weight.shape[1] || weight.shape[1] > workspace.maximumInputSize_ || weight.shape[1] % 32)
    throw std::invalid_argument("Flash small-row dense invalid matrix/rows/workspace");
  uint32_t m = 0, n = 0;
  switch (tile) {
  case FlashDenseSmallRowsTile::M8N64: m = 8; n = 64; break;
  case FlashDenseSmallRowsTile::M8N128: m = 8; n = 128; break;
  case FlashDenseSmallRowsTile::M16N64: m = 16; n = 64; break;
  case FlashDenseSmallRowsTile::M16N128: m = 16; n = 128; break;
  default: throw std::invalid_argument("Flash small-row dense invalid tile");
  }
  const uint32_t k = static_cast<uint32_t>(weight.shape[1]);
  const uint32_t outputs = static_cast<uint32_t>(weight.shape[0]);
  const uint32_t paddedRows = (rows + m - 1) / m * m;
  const uint64_t weightBytes = product(product(outputs, k), 2);
  if (weight.logicalBytes < weightBytes)
    throw std::invalid_argument("Flash small-row dense weight logical extent is short");
  requireBuffer(weight.buffer, weightBytes, "weights");
  requireBuffer(input, product(product(rows, k), 2), "input");
  requireBuffer(output, product(product(rows, outputs), 2), "output");
  requireBuffer(diagnostics, sizeof(uint32_t), "diagnostics");
  requireBuffer(workspace.paddedInput_, product(product(paddedRows, k), 2), "padding");
  for (const auto &b : {input, output, diagnostics, workspace.paddedInput_})
    if (overlaps(b, weight.buffer))
      throw std::invalid_argument("Flash small-row dense aliases immutable weights");
  if (overlaps(input, output) || overlaps(input, diagnostics) || overlaps(output, diagnostics) ||
      overlaps(workspace.paddedInput_, input) || overlaps(workspace.paddedInput_, output) ||
      overlaps(workspace.paddedInput_, diagnostics))
    throw std::invalid_argument("Flash small-row dense buffer overlap");
  FlashDenseSmallRowsParams params{rows, paddedRows, k, outputs, 0, outputs, m, n};
  graph.add("flash_dense_small_rows_pad", {input, workspace.paddedInput_, diagnostics}, params,
      {(product(paddedRows, k) - 1) / 256 + 1, 1, 1});
  const auto dispatch = [&](uint32_t begin, uint32_t count, uint32_t tileN) {
    if (!count) return;
    params.output_begin = begin; params.output_count = count; params.tile_outputs = tileN;
    graph.add("flash_dense_small_rows_m" + std::to_string(m) + "_n" + std::to_string(tileN),
        {workspace.paddedInput_, weight.buffer, output, diagnostics}, params,
        {count / tileN, paddedRows / m, 1}, {128, 1, 1});
  };
  const uint32_t fullColumns = outputs / n * n;
  dispatch(0, fullColumns, n);
  if (fullColumns < outputs) dispatch(fullColumns, outputs - fullColumns, 64);
}

} // namespace splash::flash

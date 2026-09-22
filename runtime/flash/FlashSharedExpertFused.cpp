#include "FlashSharedExpertFused.hpp"
#include "flash/FlashMoE.hpp"
#include "metal/abi/FlashSharedExpertFused.h"
#include <limits>
#include <string>

namespace splash::flash {
namespace {
void require(bool condition, const char *reason) {
  if (!condition) throw std::invalid_argument(std::string("Flash shared expert fusion: ") + reason);
}
void bytes(const metal::MetalBuffer &buffer, uint64_t expected) {
  require(buffer && buffer.contents() && buffer.sizeBytes() >= expected,
      "Shared buffer extent is insufficient");
}
bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  const auto x = reinterpret_cast<uintptr_t>(a.contents());
  const auto y = reinterpret_cast<uintptr_t>(b.contents());
  require(x && y && x <= std::numeric_limits<uintptr_t>::max() - a.sizeBytes() &&
      y <= std::numeric_limits<uintptr_t>::max() - b.sizeBytes(), "invalid Shared address extent");
  return x < y + b.sizeBytes() && y < x + a.sizeBytes();
}
void tensor(const FlashTensor &value) {
  require(value.dtype == FlashDType::BF16 && value.shape == std::vector<uint64_t>{640, 2560} &&
      value.logicalBytes == 640ULL * 2560 * 2, "requires checked cached shared expert BF16 operands");
  bytes(value.buffer, value.logicalBytes);
}
} // namespace

void addSharedExpertFusedPrefill(metal::MetalBackend &backend,
    metal::CommandGraph &graph, const FlashTensor &gate, const FlashTensor &up,
    metal::MetalBuffer input, metal::MetalBuffer activated,
    metal::MetalBuffer diagnostics, uint32_t rows,
    const FlashSharedExpertFusedTail &tail) {
  const auto plan = flashSharedExpertFusedRows(rows);
  tensor(gate); tensor(up);
  bytes(input, uint64_t{rows} * 2560 * 2);
  bytes(activated, uint64_t{rows} * 640 * 2); bytes(diagnostics, 4);
  require(!overlaps(gate.buffer, up.buffer), "gate/up operand planes overlap");
  for (const auto &weight : {gate.buffer, up.buffer})
    for (const auto &value : {input, activated, diagnostics})
      require(!overlaps(weight, value), "input/writer overlaps immutable operand plane");
  require(!overlaps(activated, input) && !overlaps(diagnostics, input) &&
      !overlaps(diagnostics, activated), "writer/input overlap");
  const uint32_t remaining = plan.tailWholeRows + plan.tailVectorRows;
  if (remaining) {
    bytes(tail.gate, uint64_t{remaining} * 640 * 2);
    bytes(tail.up, uint64_t{remaining} * 640 * 2);
    require(!overlaps(tail.gate, tail.up), "tail gate/up overlap");
    for (const auto &temporary : {tail.gate, tail.up})
      for (const auto &value : {input, activated, diagnostics, gate.buffer, up.buffer})
        require(!overlaps(temporary, value), "tail scratch overlap");
  }
  // Validate backend ownership before mutating the caller's command graph.
  const auto fullInput = backend.view(input, 0, uint64_t{plan.fullRows} * 2560 * 2);
  const auto fullOutput = backend.view(activated, 0, uint64_t{plan.fullRows} * 640 * 2);
  (void)backend.view(gate.buffer, 0, gate.logicalBytes);
  (void)backend.view(up.buffer, 0, up.logicalBytes);
  (void)backend.view(diagnostics, 0, sizeof(uint32_t));
  metal::MetalBuffer tailInput, tailOutput, tailGate, tailUp;
  if (remaining) {
    tailInput = backend.view(input, uint64_t{plan.fullRows} * 2560 * 2, uint64_t{remaining} * 2560 * 2);
    tailOutput = backend.view(activated, uint64_t{plan.fullRows} * 640 * 2, uint64_t{remaining} * 640 * 2);
    tailGate = backend.view(tail.gate, 0, uint64_t{remaining} * 640 * 2);
    tailUp = backend.view(tail.up, 0, uint64_t{remaining} * 640 * 2);
  }
  const FlashSharedExpertFusedParams p{plan.fullRows, 2560, 640, 0, 640, 32, 128, 0};
  graph.add("flash_shared_expert_fused_prefill_m32_n128",
      {fullInput, gate.buffer, up.buffer, fullOutput, diagnostics}, p,
      {5, plan.fullRows / 32, 1}, {128, 1, 1});
  if (remaining) {
    addDenseBF16WholeK(backend, graph, tailInput, gate, tailGate, diagnostics,
        remaining, FlashAffineMPPTile::M16N64);
    addDenseBF16WholeK(backend, graph, tailInput, up, tailUp, diagnostics,
        remaining, FlashAffineMPPTile::M16N64);
    addSiLUMultiply(graph, tailGate, tailUp, tailOutput, diagnostics, remaining, 640);
  }
}
} // namespace splash::flash

#pragma once

#include "ops/DraftAttention.hpp"
#include "ops/KernelSelection.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::ops {

// The device's plans of every operator a model runs, at the runtime's
// Int8Activations, which its Linear holds and its attention plans read too.
// One runtime owns this object and production models borrow it; it never
// changes after creation.
class ExecutionPlans final {
public:
  explicit ExecutionPlans(const DeviceCapabilities &device, Int8Activations int8Activations);
  [[nodiscard]] const Linear &linear() const noexcept { return linear_; }
  // A DFlash2 draft's plans: linear()'s in bf16 activations
  // (Int8Activations::Off), so no prefill chunk runs the int8 prefill tile.
  // The draft's prefill gains nothing from it, while its context projection,
  // the widest a prefill runs, would size the tile's scratch: 38 MB on
  // Qwen3.6-35B-A3B GGUF, whose own projections never take the tile.
  [[nodiscard]] const Linear &draftLinear() const noexcept { return draftLinear_; }
  [[nodiscard]] const KernelCalibration &calibration() const noexcept { return linear_.calibration(); }
  // The fewest rows of each class of prefill chunk whose projections run
  // kernels of their own: each staged tile height a chunk takes, and the
  // prefill tile (ExecutionPlans.cpp). Startup warmup runs a chunk of each and
  // one of the most rows; kernel-selection checks that they select every
  // projection, float and MoE tile a chunk of any rows selects, and its
  // prefill attention after each history it plans.
  [[nodiscard]] std::vector<uint32_t> prefillRowClasses() const;

  [[nodiscard]] PrefillAttentionPlan prefillAttention(
      uint32_t rows, uint32_t historyTokens, uint32_t queryHeads, kv::Layout layout,
      AttentionVariant variant = {}) const;
  [[nodiscard]] VerifyAttentionPlan verifyAttention(
      uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
      std::span<const uint32_t> historyTokens, AttentionVariant variant = {}) const;
  [[nodiscard]] DraftAttentionPlan draftAttention(
      DraftAttentionShape shape, uint32_t lanes) const;
  // The plans of a MoE target's blocks, whose routers are in `routerFormat`
  // (moeRouterFormat).
  [[nodiscard]] MoePlan moePrefill(MoeShape shape, uint32_t routerFormat, uint32_t rows) const;
  [[nodiscard]] MoePlan moeDecode(MoeShape shape, uint32_t routerFormat, uint32_t lanes) const;

  // Bounds cover every row count up to the requested maximum, not just that
  // one. Decode arenas use a per-lane stride of
  // max_B ceil(requiredBytes(B)/B), independently for each scratch field.
  [[nodiscard]] AttentionWorkspace prefillAttentionWorkspace(
      uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] AttentionWorkspace verifyAttentionWorkspacePerLane(
      uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] DraftAttentionWorkspace draftAttentionWorkspacePerLane(
      DraftAttentionShape shape) const;
  [[nodiscard]] MoeWorkspace moePrefillWorkspace(
      MoeShape shape, uint32_t routerFormat, uint32_t maximumRows) const;
  [[nodiscard]] MoeWorkspace moeDecodeWorkspacePerLane(MoeShape shape, uint32_t routerFormat) const;
  // This scratch is one whole-command buffer, not a per-lane arena field.
  [[nodiscard]] uint64_t gateUpWorkspace(ProjectionShape shape) const;

private:
  Linear linear_;
  Linear draftLinear_;
};

} // namespace splash::ops

#pragma once

#include "ops/DraftAttention.hpp"
#include "ops/KernelSelection.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <span>

namespace splash::ops {

// The device's plans of every operator a model runs. One runtime owns this
// object and production models borrow it; it never changes after creation.
class ExecutionPlans final {
public:
  explicit ExecutionPlans(const DeviceCapabilities &device);
  [[nodiscard]] const Linear &linear() const noexcept { return linear_; }
  [[nodiscard]] const KernelCalibration &calibration() const noexcept { return linear_.calibration(); }

  [[nodiscard]] PrefillAttentionPlan prefillAttention(
      uint32_t rows, uint32_t historyTokens, uint32_t queryHeads, kv::Layout layout,
      AttentionVariant variant = {}) const;
  [[nodiscard]] VerifyAttentionPlan verifyAttention(
      uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
      std::span<const uint32_t> historyTokens, AttentionVariant variant = {}) const;
  [[nodiscard]] DraftAttentionPlan draftAttention(
      DraftAttentionShape shape, uint32_t lanes) const;
  // The plans of a MoE target's blocks, whose experts are in `formats`.
  [[nodiscard]] MoePlan moePrefill(MoeShape shape, const MoeFormats &formats, uint32_t rows) const;
  [[nodiscard]] MoePlan moeDecode(MoeShape shape, const MoeFormats &formats, uint32_t lanes) const;

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
      MoeShape shape, const MoeFormats &formats, uint32_t maximumRows) const;
  [[nodiscard]] MoeWorkspace moeDecodeWorkspacePerLane(MoeShape shape, const MoeFormats &formats) const;
  // This scratch is one whole-command buffer, not a per-lane arena field.
  [[nodiscard]] uint64_t gateUpWorkspace(ProjectionShape shape) const;

private:
  Linear linear_;
};

} // namespace splash::ops

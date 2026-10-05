#pragma once

#include "ops/DraftAttention.hpp"
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

  [[nodiscard]] PrefillAttentionPlan prefillAttention(
      uint32_t rows, uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] VerifyAttentionPlan verifyAttention(
      uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
      std::span<const uint32_t> historyTokens) const;
  [[nodiscard]] DraftAttentionPlan draftAttention(
      DraftAttentionShape shape, uint32_t lanes) const;
  [[nodiscard]] MoePlan moePrefill(MoeShape shape, uint32_t rows) const;
  [[nodiscard]] MoePlan moeDecode(MoeShape shape, uint32_t lanes) const;

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
      MoeShape shape, uint32_t maximumRows) const;
  [[nodiscard]] MoeWorkspace moeDecodeWorkspacePerLane(MoeShape shape) const;
  // This scratch is one whole-command buffer, not a per-lane arena field.
  [[nodiscard]] uint64_t gateUpWorkspace(ProjectionShape shape) const;

private:
  // The device's configuration of a MoE plan of `rows` rows in `phase`.
  [[nodiscard]] MoeConfig moeConfig(MoeShape shape, uint32_t rows, MoePhase phase) const;

  Linear linear_;
  uint32_t moeRouteWideRows_;
  GpuFamilyClass family_;
};

} // namespace splash::ops

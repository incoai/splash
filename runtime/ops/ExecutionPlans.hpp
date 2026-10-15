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
  // The draft attention of `lanes` lanes of `rows` query rows each.
  [[nodiscard]] DraftAttentionPlan draftAttention(
      DraftAttentionShape shape, uint32_t lanes, uint32_t rows) const;
  [[nodiscard]] MoePlan moePrefill(MoeShape shape, uint32_t rows) const;
  // The MoE of a decode cycle's rows, `blocks` row blocks of them: the
  // experts route each row alone, whichever lane it verifies.
  [[nodiscard]] MoePlan moeDecode(MoeShape shape, uint32_t blocks) const;

  // Bounds cover every row count up to the requested maximum, not just that
  // one. Decode arenas use a stride per row block of
  // max_B ceil(requiredBytes(B)/B) over B blocks, independently for each
  // scratch field.
  [[nodiscard]] AttentionWorkspace prefillAttentionWorkspace(
      uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] AttentionWorkspace verifyAttentionWorkspacePerRowBlock(
      uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] DraftAttentionWorkspace draftAttentionWorkspacePerRowBlock(
      DraftAttentionShape shape) const;
  [[nodiscard]] MoeWorkspace moePrefillWorkspace(
      MoeShape shape, uint32_t maximumRows) const;
  [[nodiscard]] MoeWorkspace moeDecodeWorkspacePerRowBlock(MoeShape shape) const;
  // This scratch is one whole-command buffer, not a per-lane arena field.
  [[nodiscard]] uint64_t gateUpWorkspace(ProjectionShape shape) const;

private:
  // The device's configuration of a MoE plan of `rows` rows in `phase`.
  [[nodiscard]] MoeConfig moeConfig(MoeShape shape, uint32_t rows, MoePhase phase) const;

  Linear linear_;
  GpuFamilyClass family_;
};

} // namespace splash::ops

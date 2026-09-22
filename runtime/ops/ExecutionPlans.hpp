#pragma once

#include "ops/DraftAttention.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <array>
#include <compare>
#include <span>
#include <vector>

namespace splash::ops {

struct AttentionShape final {
  uint32_t queryHeads = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  kv::Format format = kv::Format::Int8;
  auto operator<=>(const AttentionShape &) const = default;
};

// A startup-selected precompiled configuration applies only to this exact row
// count. Histories are calibrated empirically, not encoded into serving keys.
struct PrefillAttentionPolicy final {
  AttentionShape shape;
  uint32_t rows = SPLASH_PREFILL_TOKEN_BUDGET;
  auto operator<=>(const PrefillAttentionPolicy &) const = default;
};

struct VerifyAttentionPolicy final {
  AttentionShape shape;
  uint32_t lanes = 0;
  auto operator<=>(const VerifyAttentionPolicy &) const = default;
};

struct DraftAttentionWorkload final {
  DraftAttentionShape shape;
  uint32_t lanes = 0;
  std::strong_ordering operator<=>(const DraftAttentionWorkload &) const noexcept;
  bool operator==(const DraftAttentionWorkload &) const noexcept;
};

enum class MoePhase : uint8_t { Prefill, Decode };

struct MoeWorkload final {
  MoeShape shape;
  // Physical rows in both phases: decode uses 8, 16, 24 or 32.
  uint32_t rows = 0;
  MoePhase phase = MoePhase::Decode;
  std::strong_ordering operator<=>(const MoeWorkload &) const noexcept;
  bool operator==(const MoeWorkload &) const noexcept;
};

struct PrefillAttentionChoice final {
  PrefillAttentionPolicy workload;
  PrefillAttentionConfig configuration;
};
struct VerifyAttentionChoice final {
  VerifyAttentionPolicy workload;
  VerifyAttentionConfig configuration;
};
struct DraftAttentionChoice final {
  DraftAttentionWorkload workload;
  DraftAttentionConfiguration configuration;
};
struct MoeChoice final {
  MoeWorkload workload;
  MoeConfig configuration;
};

struct OperatorChoices final {
  std::vector<LinearChoice> linear;
  std::vector<PrefillAttentionChoice> prefillAttention;
  std::vector<VerifyAttentionChoice> verifyAttention;
  std::vector<DraftAttentionChoice> draftAttention;
  std::vector<MoeChoice> moe;

  [[nodiscard]] bool empty() const noexcept {
    return linear.empty() && prefillAttention.empty() && verifyAttention.empty() &&
           draftAttention.empty() && moe.empty();
  }
};

// One runtime owns this object; production models borrow it. Installation is
// a startup-only operation, before concurrent encoding. Arena sizing precedes
// confirmation; those trials may only toggle the preallocated baseline/selected
// pair. After Ready the owner and all borrowed plans remain immutable.
class ExecutionPlans final {
public:
  explicit ExecutionPlans(const DeviceCapabilities &device);
  [[nodiscard]] const Q4Linear &linear() const noexcept { return linear_; }
  // Validate every table before replacing any installed choice. Missing keys
  // always use the operator's shipped baseline; an empty install resets all.
  void install(const OperatorChoices &choices);

  [[nodiscard]] PrefillAttentionPlan prefillAttention(
      uint32_t rows, uint32_t queryHeads, kv::Layout layout,
      uint32_t historyTokens) const;
  [[nodiscard]] VerifyAttentionPlan verifyAttention(
      uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
      std::span<const uint32_t> historyTokens) const;
  [[nodiscard]] DraftAttentionPlan draftAttention(
      DraftAttentionShape shape, uint32_t lanes) const;
  [[nodiscard]] MoePlan moePrefill(MoeShape shape, uint32_t rows) const;
  [[nodiscard]] MoePlan moeDecode(MoeShape shape, uint32_t lanes) const;
  // Shipped baseline first, independent of installed choices. Every candidate
  // uses the same device router and expert-tile policy as production lookups
  // and encoding.
  [[nodiscard]] std::array<MoePlan, 2> moeCandidates(const MoeWorkload &workload) const;

  // Bounds include baseline and every matching installed key, not just the
  // currently requested row count. Packed decode arenas use a per-lane stride
  // of max_B ceil(requiredBytes(B)/B), independently for each scratch field.
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
  [[nodiscard]] uint64_t gateUpWorkspace(LinearMatrix matrix) const;

private:
  Q4Linear linear_;
  Q4Linear baselineLinear_;
  uint32_t moeRouteWideRows_ = kMoeRouteWideRows;
  MoeExpertSimdgroups moeDecodeSimdgroups_ = MoeExpertSimdgroups::Eight;
  OperatorChoices choices_;
};

} // namespace splash::ops

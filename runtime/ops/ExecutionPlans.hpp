#pragma once

#include "ops/DraftAttention.hpp"
#include "ops/KernelSelection.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace splash::ops {

// The device's plans of every operator a model runs, at the runtime's
// Int8Activations, which its Linear holds and its attention plans read too.
// One runtime owns this object and production models borrow it; it never
// changes after creation.
class ExecutionPlans final {
public:
  explicit ExecutionPlans(const DeviceCapabilities &device, Int8Activations int8Activations);
  // A prefill chunk's plans, and a decode step's, which take no int8 prefill
  // path whatever its rows (StepKind).
  [[nodiscard]] const Linear &linear() const noexcept { return linear_; }
  [[nodiscard]] const Linear &decodeLinear() const noexcept { return decodeLinear_; }
  // A DFlash2 draft's plans: linear()'s in bf16 activations
  // (Int8Activations::Off), so no prefill chunk runs the int8 prefill tile.
  // The draft's prefill gains nothing from it, while its context projection,
  // the widest a prefill runs, would size the tile's scratch: 38 MB on
  // Qwen3.6-35B-A3B GGUF, whose own projections never take the tile.
  [[nodiscard]] const Linear &draftLinear() const noexcept { return draftLinear_; }
  // The draft's plans of a decode step's rows (its context projections over the
  // step's rows, DFlashDraft::addContextCommit): draftLinear's in a decode step.
  [[nodiscard]] const Linear &draftDecodeLinear() const noexcept { return draftDecodeLinear_; }
  [[nodiscard]] const KernelCalibration &calibration() const noexcept { return linear_.calibration(); }
  // The fewest rows of each class of prefill chunk whose projections, MoE
  // experts or relative attention run kernels of their own: each staged tile
  // height a chunk takes, the prefill tile, each set of expert passes where
  // `moe` names a MoE target's blocks' shape and router format, and each set
  // of kernels the relative attention of a chunk after no history runs where
  // `relativeAttention` names the KV layout of a target whose layers attend
  // through it (ExecutionPlans.cpp); but not a class the most rows start,
  // whose chunk is warmup's of the most rows. Startup warmup runs a chunk of
  // each and one of the most rows; kernel-selection checks that they select
  // every projection and float tile, MoE expert passes and router tile a
  // chunk of any rows selects, and the kernels its prefill attention runs
  // after each history it plans.
  [[nodiscard]] std::vector<uint32_t> prefillRowClasses(std::optional<std::pair<MoeShape, uint32_t>> moe,
                                                        std::optional<kv::Layout> relativeAttention = {}) const;
  // The fewest rows of each class of decode step of up to `mostRows` rows,
  // at most a verify block a lane, whose kernels it picks by its rows: its
  // projections' tile and staged height, as prefillRowClasses walks them, a
  // float projection's tile (selectFloatTile), and a MoE target's expert
  // tile (selectMoe) where `moe` names its blocks' shape and router format.
  // A decode step's attention and GDN kernels follow its lane cap, not its
  // rows; its norms and heads follow its projections' plans. The choice of a
  // cycle's rows prices a class by a step of its last rows
  // (model::StepCosts), and the start creates that step's kernels alone, so
  // a class holds no kernel change; kernel-selection checks the tiles and
  // heights that would.
  [[nodiscard]] std::vector<uint32_t> decodeRowClasses(uint32_t mostRows,
                                                       std::optional<std::pair<MoeShape, uint32_t>> moe) const;

  // The attention of a command's sequences (PagedAttention::plan) whose
  // dispatches fit `capacity`, the workspace bound to it.
  [[nodiscard]] AttentionPlan attention(std::span<const kv::ChunkedPrefillParams> sequences, uint32_t queryHeads,
                                        kv::Layout layout, AttentionWorkspace capacity,
                                        AttentionVariant variant = {}) const;
  [[nodiscard]] DraftAttentionPlan draftAttention(
      DraftAttentionShape shape, uint32_t lanes) const;
  // The plan of `rows` rows of `step`'s work through a MoE target's blocks,
  // whose routers are in `routerFormat` (moeRouterFormat).
  [[nodiscard]] MoePlan moe(MoeShape shape, uint32_t routerFormat, uint32_t rows,
                            StepKind step) const;

  // The attention workspace that holds any one sequence's partials, a
  // prompt chunk's; and a decode lane's, the most split slots a lane of up to
  // a verify block's rows takes: one tile at the most splits, which a lane of
  // two tiles' 32 splits each stays within (PagedAttention.cpp,
  // attentionSplits), so that a step's lanes, their workspaces added, attend
  // in one dispatch.
  [[nodiscard]] AttentionWorkspace sequenceAttentionWorkspace(uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] AttentionWorkspace laneAttentionWorkspace(uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] DraftAttentionWorkspace draftAttentionWorkspacePerLane(
      DraftAttentionShape shape) const;
  // The workspace of every MoE plan of 1 to `maximumRows` rows of `step`'s
  // work, not just the most: a prompt chunk's, or a decode step's of its
  // lanes' packed rows (moe).
  [[nodiscard]] MoeWorkspace moeWorkspace(MoeShape shape, uint32_t routerFormat, uint32_t maximumRows,
                                          StepKind step) const;
  // The gate scratch of every gate/up plan of a decode step of 1 to
  // `maximumRows` rows of a pair of `shape` (decodeLinear's): one
  // whole-command buffer, not a per-lane arena field.
  [[nodiscard]] uint64_t gateUpWorkspace(ProjectionShape shape, uint32_t maximumRows) const;

private:
  Linear linear_;
  Linear decodeLinear_;
  Linear draftLinear_;
  Linear draftDecodeLinear_;
};

} // namespace splash::ops

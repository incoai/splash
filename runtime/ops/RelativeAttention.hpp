#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/RelativeAttention.h"
#include "ops/KernelCalibration.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace splash::ops {

// Relative attention (metal/abi/RelativeAttention.h, kernels/shared/
// relative_attention.metal): Inkling-2-Mini's attention, the paged attention
// of PagedAttention over the same KV pages, page tables, rows, tiles and
// splits, whose scores also take a bias of each key's backward distance,
// mixed by the row's own relative states, and whose softmax takes a sink of
// the row's own for each query head; its queries and keys are clipped heads
// without RoPE, its values normalized heads, its output gated. Its one
// compiled geometry is the model's: kQueryHeads query heads over the KV heads
// of the layouts it stores (stores), kWindowTokens keys before each row for a
// sliding layer. A prompt chunk's layer runs a prepare, a split and a reduce
// (addProjection, addAttention); a decode step's runs them in one dispatch
// (addDecode), each lane's block one tile.

// How one layer attends: over its whole history, or a window of the
// windowTokens keys before each row (kWindowTokens, a sliding layer's); and
// whether its queries and relative states are positionScaled by their rows'
// log scale (the full layers').
struct RelativeAttentionVariant final {
  uint32_t windowTokens = 0;
  bool positionScaled = false;
  bool operator==(const RelativeAttentionVariant &) const = default;
};

// A layer's rows, a row each of the command's packed rows: `projection`,
// bf16 rows of every query head's queries, then their output gates, their
// relative states (SPLASH_RELATIVE_ATTENTION_STATES a head) and their sinks
// (one a head), each a block, kProjectionWidth columns of rows
// kProjectionStride apart; `keys` and `values`, its convolved K and V rows,
// bf16, KV heads x head dimension each.
struct RelativeAttentionRows final {
  metal::MetalBuffer projection;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
};

// A layer's weights, F32 as its GGUF holds them: the learned scales of its
// queries ([query head][dimension]) and relative states ([query head]
// [state]), and its bias table (a row of states for each backward distance
// below SPLASH_RELATIVE_ATTENTION_EXTENT: [distance][state]).
struct RelativeAttentionWeights final {
  metal::MetalBuffer queryScale;
  metal::MetalBuffer relativeScale;
  metal::MetalBuffer bias;
};

// What the prepare stages for the attention, each in the paged attention's
// order of queries ([KV head][row][query head in group]), each sequence's
// after the one before it, a KV head's rows its chunk_stride apart: the
// queries (bf16), each fused row's relative states and its sink (F32).
struct RelativeAttentionStaging final {
  metal::MetalBuffer queries;
  metal::MetalBuffer states;
  metal::MetalBuffer sinks;
};
struct RelativeAttentionStagingBytes final {
  uint64_t queries = 0;
  uint64_t states = 0;
  uint64_t sinks = 0;
};

// What a decode step's single pass (addDecode) works in: its plan's partials
// and statistics (RelativeAttentionPlan::workspace) and its tiles' counters
// (RelativeAttention::kCountersBytes), one a KV head of each of a command's
// sequences, which every dispatch leaves zero, so the runtime allocates them
// zero.
struct RelativeAttentionWorkspace final {
  metal::MetalBuffer partials;
  metal::MetalBuffer statistics;
  metal::MetalBuffer counters;
};

// The attention of a command's sequences for one layer variant, planned once
// a step, as AttentionPlan: each sequence's history splits and the tiles of
// it that run, its kernels, and the dispatches its sequences share. A
// prompt chunk's plan (StepKind::Prefill) runs a split and a reduce of its
// sequences' tiles; a decode step's (StepKind::Decode) its single pass, each
// lane's block of rows one tile of up to a verify block's rows, whose splits
// each take two slots: their partials' and their staging's.
class RelativeAttentionPlan final {
public:
  using Sequence = AttentionPlan::Sequence;
  using Dispatch = AttentionPlan::Dispatch;

  const kv::Layout layout;
  const StepKind step;
  // The tile splits the workspace holds, which each dispatch's fit.
  const uint32_t workspaceSlots;

  [[nodiscard]] std::span<const Sequence> sequences() const noexcept { return {sequences_.data(), sequenceCount_}; }
  [[nodiscard]] std::span<const Dispatch> dispatches() const noexcept { return {dispatches_.data(), dispatchCount_}; }
  [[nodiscard]] AttentionWorkspace workspace() const noexcept;
  // A prompt chunk's split and reduce kernels, or a decode step's single
  // pass; the others are empty.
  [[nodiscard]] const std::string &splitPipeline() const noexcept { return splitPipeline_; }
  [[nodiscard]] const std::string &reducePipeline() const noexcept { return reducePipeline_; }
  [[nodiscard]] const std::string &decodePipeline() const noexcept { return decodePipeline_; }
  // The split's grid, or the single pass's: KV heads, the dispatch's
  // sequences and their most splits.
  [[nodiscard]] metal::DispatchSize splitGroups(const Dispatch &dispatch) const noexcept;
  [[nodiscard]] metal::DispatchSize reduceGroups(const Dispatch &dispatch) const noexcept;

private:
  friend class RelativeAttention;
  RelativeAttentionPlan(kv::Layout layout, StepKind step, uint32_t workspaceSlots)
      : layout(layout), step(step), workspaceSlots(workspaceSlots) {}
  std::array<Sequence, SPLASH_MAXIMUM_BATCH_WIDTH> sequences_{};
  uint32_t sequenceCount_ = 0;
  std::array<Dispatch, SPLASH_MAXIMUM_BATCH_WIDTH> dispatches_{};
  uint32_t dispatchCount_ = 0;
  std::string splitPipeline_;
  std::string reducePipeline_;
  std::string decodePipeline_;
};

class RelativeAttention final {
public:
  static constexpr uint32_t kQueryHeads = 24;
  static constexpr uint32_t kKvHeads = 8;
  static constexpr uint32_t kHeadDimension = 128;
  // A sliding layer's 1024 keys: the 1023 before a row and its own
  // (kv::Layout::windowTokens).
  static constexpr uint32_t kWindowTokens = 1023;
  static constexpr uint64_t kProjectionWidth =
      uint64_t{kQueryHeads} * (2 * kHeadDimension + SPLASH_RELATIVE_ATTENTION_STATES + 1);
  // The projection's rows are whole tiles of GGUF_TILE_COLUMNS outputs
  // (ops::Linear): its columns past kProjectionWidth are padding, which no
  // kernel reads (kernels/common/relative_attention_tile.h
  // RelativeAttentionRow::Stride).
  static constexpr uint64_t kProjectionStride =
      (kProjectionWidth + GGUF_TILE_COLUMNS - 1) / GGUF_TILE_COLUMNS * GGUF_TILE_COLUMNS;
  // The single pass's counters (RelativeAttentionWorkspace).
  static constexpr uint64_t kCountersBytes = uint64_t{SPLASH_MAXIMUM_BATCH_WIDTH} * kKvHeads * sizeof(uint32_t);
  // A decode lane's tile takes a history split per kDecodePagesPerSplit pages
  // it reads, at most kDecodeTileSplits shared out over the tiles of
  // kv::kAttentionTileRows rows its block spans (decodeSplits).
  static constexpr uint32_t kDecodePagesPerSplit = 4;
  static constexpr uint32_t kDecodeTileSplits = 12;

  // Whether a layout's pages are those the kernels read and store: kKvHeads
  // heads of kHeadDimension, every page read, values dimension-major.
  [[nodiscard]] static bool stores(kv::Layout layout) noexcept;

  // The staging of a command's sequences (their chunks, in row order).
  [[nodiscard]] static RelativeAttentionStagingBytes stagingBytes(
      std::span<const kv::ChunkedPrefillParams> sequences, kv::Layout layout);
  // The workspace of `slots` tile splits, as PagedAttention::workspace; and
  // those the runtime allocates, as ExecutionPlans::sequenceAttentionWorkspace
  // and laneAttentionWorkspace size the paged attention's: a prompt
  // sequence's (kAttentionSequenceSplits, any one sequence's splits) and a
  // decode lane's (kv::kAttentionMaximumSplits; a step's lanes add theirs).
  [[nodiscard]] static AttentionWorkspace workspace(uint32_t slots, kv::Layout layout);
  [[nodiscard]] static AttentionWorkspace sequenceWorkspace(kv::Layout layout) {
    return workspace(kAttentionSequenceSplits, layout);
  }
  [[nodiscard]] static AttentionWorkspace laneWorkspace(kv::Layout layout) {
    return workspace(kv::kAttentionMaximumSplits, layout);
  }

  // The plan of a command's sequences, whose dispatches each fit `capacity`:
  // a prompt chunk's with the history splits and INT8-query choice of
  // PagedAttention::plan; a decode step's (`step`), whose lanes share their
  // bound, the step's lane cap, with a lane's splits of decodeSplits and the
  // same INT8-query choice.
  [[nodiscard]] static RelativeAttentionPlan plan(std::span<const kv::ChunkedPrefillParams> sequences,
                                                  kv::Layout layout, const AttentionCalibration &calibration,
                                                  Int8Activations int8Activations, AttentionWorkspace capacity,
                                                  RelativeAttentionVariant variant = {},
                                                  StepKind step = StepKind::Prefill);
  // A prompt chunk's `plan`'s sequences from the tile that holds each one's
  // last row.
  [[nodiscard]] static RelativeAttentionPlan lastTiles(const RelativeAttentionPlan &plan);
  // The history splits of a decode lane's tile of `laneCap` rows that reads
  // `pages` pages: one per kDecodePagesPerSplit pages, at most
  // kDecodeTileSplits for a narrow cap's tile and half as many for a verify
  // block's, whose combine reads twice the rows. Measured on a 20-core M5 Pro
  // (a lane over 24 layers of distinct K/V, its splits forced, fused time a
  // layer): a page costs a split about 4 us, and each split more its
  // combine's reads, so a sliding layer's 33 pages ran fastest in 9 splits at
  // 1 and 8 rows (34.9 and 41.9 us, the three-dispatch path's 45.0 and 45.9),
  // a full layer's in 12 from 2,048 to 32,768 tokens (55.9 us at 2,048 and 1
  // row, 62.9 at 8, against 64.6 and 75.5), and a verify block's in 6 (a
  // sliding layer's 64.6 us against 78.1). It depends on the lane alone, so
  // batching never changes a lane's partition.
  [[nodiscard]] static uint32_t decodeSplits(uint64_t pages, uint32_t laneCap);

  // The prepare of a layer's rows, in one dispatch before the attention:
  // each query head's queries, relative states and sink staged in `staging`,
  // and each K/V head's clipped key and normalized value written into the
  // pages of `layer`, each sequence's through its page table. A row's
  // position (a positionScaled variant's log scale) is its sequence's
  // committed tokens plus its row, or with rows' tree its anchor's plus its
  // depth.
  static void addProjection(metal::CommandGraph &graph, const RelativeAttentionRows &layerRows,
                            const RelativeAttentionWeights &weights, const RelativeAttentionStaging &staging,
                            SplashKvLayer layer, std::span<const metal::MetalBuffer> pageTables,
                            std::span<const kv::ChunkedPrefillParams> sequences, const AttentionRows &rows,
                            kv::Layout layout, RelativeAttentionVariant variant = {});
  // The plan's dispatches over the staging into the out-projection's input
  // rows (`output`, as PagedAttention::addAttention's: output.packed the
  // layer's projection rows, whose gates multiply them), each sequence
  // reading its history where `histories` say (dense: no block lists),
  // adding the layer's bias (weights.bias) and its rows' sinks. A dispatch
  // whose sequences read their layer in two placements runs as a dispatch per
  // run of them that reads one. With rows' tree each sequence is a decode
  // lane's block of at most 32 rows whose rows attend their paths.
  static PreparedInput addAttention(metal::CommandGraph &graph, const RelativeAttentionPlan &plan,
                                    const RelativeAttentionStaging &staging, const RelativeAttentionWeights &weights,
                                    const metal::MetalBuffer &partials, const metal::MetalBuffer &statistics,
                                    std::span<const kv::ChunkedPrefillParams> sequences,
                                    std::span<const AttentionHistory> histories, const AttentionRows &rows,
                                    const AttentionOutputRows &output);
  // A decode step's layer, a dispatch for each of its plan's (a decode plan,
  // whose variant its kernel compiles): what addProjection and addAttention
  // compute, each lane's rows' K/V written into the pages of `layer` through
  // its page table, from which it also reads its history, and its rows
  // attending as addAttention's (with rows' tree, their paths) into
  // `output`, as addAttention writes them, gated by layerRows' projection
  // rows (output.packed is not read).
  static PreparedInput addDecode(metal::CommandGraph &graph, const RelativeAttentionPlan &plan,
                                 const RelativeAttentionRows &layerRows, const RelativeAttentionWeights &weights,
                                 SplashKvLayer layer, std::span<const metal::MetalBuffer> pageTables,
                                 std::span<const kv::ChunkedPrefillParams> sequences, const AttentionRows &rows,
                                 const RelativeAttentionWorkspace &workspace, const AttentionOutputRows &output);

  // The commit of each lane's accepted path after a decode step, as
  // PagedAttention::addPathCommit's (SplashKvPathCommitParams), of the pages
  // relative attention stores.
  static void addPathCommit(metal::CommandGraph &graph, kv::Layout layout, uint32_t extentPages,
                            std::span<const metal::MetalBuffer> pageTables, std::span<const uint32_t> committed,
                            uint32_t laneRows, const metal::MetalBuffer &paths);

private:
  static void group(RelativeAttentionPlan &plan);
};

} // namespace splash::ops

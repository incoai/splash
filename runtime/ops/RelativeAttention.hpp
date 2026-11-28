#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/RelativeAttention.h"
#include "ops/AttentionPlanning.hpp"
#include "ops/KernelCalibration.hpp"
#include "ops/LaneRows.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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
// (addProjection, addAttention), or where each of a dispatch's sequences
// reads its history in one split, a prepare and the direct kernel, whose
// tiles of the GPU family's rows (AttentionCalibration::directRows) write
// their rows themselves, the same bits; a decode step's runs them in one
// dispatch (addDecode), each lane's block one tile.

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
// sequences' tiles, or the direct kernel of a dispatch whose sequences each
// take one split (direct); a decode step's (StepKind::Decode) its single
// pass, each lane's block of rows one tile of up to a verify block's rows,
// whose splits each take two slots: their partials' and their staging's.
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
  // A prompt chunk's split and reduce kernels and its direct kernel (a
  // dispatch whose sequences each take one split: direct), or a decode
  // step's single pass; the others are empty.
  [[nodiscard]] const std::string &splitPipeline() const noexcept { return splitPipeline_; }
  [[nodiscard]] const std::string &reducePipeline() const noexcept { return reducePipeline_; }
  [[nodiscard]] const std::string &directPipeline() const noexcept { return directPipeline_; }
  [[nodiscard]] const std::string &decodePipeline() const noexcept { return decodePipeline_; }
  // A decode step's most rows a lane its tile holds (RelativeAttention::plan), each lane's bound in its single
  // pass.
  [[nodiscard]] uint32_t decodeLaneRows() const noexcept { return decodeLaneRows_; }
  // The split's grid, or the single pass's: KV heads, the dispatch's
  // sequences and their most splits.
  [[nodiscard]] metal::DispatchSize splitGroups(const Dispatch &dispatch) const noexcept;
  [[nodiscard]] metal::DispatchSize reduceGroups(const Dispatch &dispatch) const noexcept;
  // Whether a prompt chunk's dispatch runs the direct kernel instead of the
  // split and reduce: each of its sequences reads its history in one split.
  [[nodiscard]] bool direct(const Dispatch &dispatch) const noexcept {
    return step == StepKind::Prefill && dispatch.splits == 1;
  }
  // The direct kernel's grid: KV heads and the dispatch's tiles of
  // directRows rows (directTiles).
  [[nodiscard]] metal::DispatchSize directGroups(const Dispatch &dispatch) const noexcept;
  // The rows of the direct kernel's tiles, the family's
  // (AttentionCalibration::directRows), and a sequence's tiles of them from
  // its first tile's first row.
  [[nodiscard]] uint32_t directRows() const noexcept { return directRows_; }
  [[nodiscard]] uint32_t directTiles(const Sequence &sequence) const noexcept;

private:
  friend class RelativeAttention;
  RelativeAttentionPlan(kv::Layout layout, StepKind step, uint32_t workspaceSlots, uint32_t directRows)
      : layout(layout), step(step), workspaceSlots(workspaceSlots), directRows_(directRows) {}
  std::array<Sequence, SPLASH_MAXIMUM_BATCH_WIDTH> sequences_{};
  uint32_t sequenceCount_ = 0;
  std::array<Dispatch, SPLASH_MAXIMUM_BATCH_WIDTH> dispatches_{};
  uint32_t dispatchCount_ = 0;
  std::string splitPipeline_;
  std::string reducePipeline_;
  std::string directPipeline_;
  std::string decodePipeline_;
  uint32_t decodeLaneRows_ = 0;
  uint32_t directRows_;
};

class RelativeAttention final {
public:
  static constexpr uint32_t kQueryHeads = 24;
  static constexpr uint32_t kKvHeads = 8;
  static constexpr uint32_t kHeadDimension = 128;
  // A sliding layer's 1024 keys: the 1023 before a row and its own
  // (kv::Layout::windowTokens).
  static constexpr uint32_t kWindowTokens = 1023;
  // The single pass's tiles (relative_attention.metal): of 2 rows (6 fused
  // rows multiplied as 8), 5 (15 as 16), a paged tile's 8 (24) and a verify
  // block's 16 (48, two paged tiles' slots). A decode step's lanes take the
  // smallest that holds its most rows a lane (decodeTileRows): a 2-row
  // lane's of 8 fused rows took a short layer's window 9-11% less time than
  // the paged tile's 24 (one lane, 1K and 8K tokens, 20-core M5 Pro), and a
  // lane of 1 or 2 rows 21-33% less over a full layer's 512 to 8K tokens (one
  // and four lanes, 10-core M4).
  static constexpr std::array<uint32_t, 4> kDecodeTileRows{2, 5, kv::kAttentionTileRows, SPLASH_TARGET_VERIFY_ROWS};
  [[nodiscard]] static constexpr uint32_t decodeTileRows(uint32_t laneRows) {
    for (const uint32_t rows : kDecodeTileRows)
      if (laneRows <= rows) return rows;
    throw std::invalid_argument("a decode step's lane rows exceed a verify block's");
  }
  // A decode step's most rows a lane: its lane cap, or fewer where its other
  // lanes' anchors leave a lane fewer of its rows (each lane holds its
  // anchor row).
  [[nodiscard]] static constexpr uint32_t decodeLaneRows(const StepRows &step) noexcept {
    return std::min(step.laneCap, step.rows - std::min(step.rows, step.lanes - 1));
  }
  // The rows of the tile a decode step's lanes run (decodeTileRows of its
  // decodeLaneRows), which its rows move within its lane cap: the decode
  // step classes of a target that runs relative attention split where it
  // changes at the wide cap (ops::ExecutionPlans::decodeRowClasses), each
  // width's, whose classes then hold one tile at the narrow cap too.
  [[nodiscard]] static constexpr uint32_t decodeTileRows(const StepRows &step) {
    return decodeTileRows(decodeLaneRows(step));
  }
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
  // A decode lane's tile takes a history split per kDecodeCorePages / cores
  // pages it reads (4 on a 20-core GPU), but one a page up to (cores + 8) /
  // 4, and at most its share of the threadgroups the GPU keeps resident
  // (decodeSplits).
  static constexpr uint32_t kDecodeCorePages = 80;

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
  // bound, the step's lane cap, with each lane's splits of decodeSplits over
  // the step's lanes and the same INT8-query choice, of the variants its
  // single pass is compiled for alone, a full layer's (the whole history,
  // positionScaled) and a sliding layer's (kWindowTokens, unscaled): another
  // throws. Its lanes hold at most `laneRows` rows each (the step's
  // decodeLaneRows; 0: the cap): under the narrow cap they run the smallest
  // tile that holds them (decodeTileRows) with the cap's splits and rescale,
  // so that it writes the paged tile's bits; a verify block's lanes run its
  // tile.
  [[nodiscard]] static RelativeAttentionPlan plan(std::span<const kv::ChunkedPrefillParams> sequences,
                                                  kv::Layout layout, const AttentionCalibration &calibration,
                                                  Int8Activations int8Activations, AttentionWorkspace capacity,
                                                  RelativeAttentionVariant variant = {},
                                                  StepKind step = StepKind::Prefill, uint32_t laneRows = 0);
  // The history splits of each lane's tile of `laneCap` rows that reads
  // `pages` pages in a decode step of `lanes` lanes on the calibration's
  // GPU: one per kDecodeCorePages / cores pages, but at least one a page up
  // to (cores + 8) / 4, and at most decodeMostSplits, which the step's lanes
  // share; then as many as the runs of pages that many make
  // (relative_attention_pages), so that each reads some. A page costs a split
  // about 4 us, each split more its combine's reads, and past the threadgroups
  // the GPU keeps resident a split waits for a second wave. Measured over a
  // stream of layers of distinct K/V (us a layer, forced splits): four lanes'
  // window of 34 pages ran fastest in 2 splits each on a 16-core M5 Pro (103.4
  // against 124.5 in 3, where each lane alone took 7) and 4 on a 32-core M5
  // Max (68.2), where the 5 their share gives took 85.9, their 7-page splits
  // scoring bf16 queries, though over 8,192 tokens 5 ran 5% faster than 4 (280
  // against 294); two lanes' in 4 and 7 (61.4, 39.3). A lane's splits so depend
  // on the step's lanes, and with them its rows' sums' order; one lane's are
  // those it took alone.
  [[nodiscard]] static uint32_t decodeSplits(uint64_t pages, uint32_t laneCap, uint32_t lanes,
                                             const AttentionCalibration &calibration);
  // The most history splits of each lane's tile of `laneCap` rows in a
  // decode step of `lanes` lanes: the lanes' share of the threadgroups the
  // calibration's GPU keeps resident (decodeThreadgroupsPerFourCores every
  // four cores, of a verify block's tiles decodeWideThreadgroupsPerFourCores),
  // a KV head's of each split, to the nearest whole split; but no more than
  // one lane repays alone, the edge of its combine and of the threadgroups the
  // GPU keeps resident, (cores + 16) / 3 of a narrow cap's tiles and
  // (cores + 8) / 4 of a verify block's (13 and 8 took a full layer's 2,048
  // tokens 20% and 16% more than 12 and 7 on a 20-core M5 Pro, 17 its 8,192
  // tokens 9% more than 16 on a 32-core M5 Max); and at most
  // SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS. Apple10's and Apple11's 19 and 14
  // every four cores (4.75 and 3.5 a core) are fitted, so that the rounding
  // gives the fastest four-lane splits on 16, 20 and 32 cores; they are not
  // measured residency, which held 64 to 79 of a verify block's tiles on both
  // M5 Pros. One lane takes 10 and 6 on a 16-core M5 Pro, 12 and 7 on a
  // 20-core, 16 and 10 on a 32-core M5 Max; four lanes there 2, 3 and 5 each
  // of 8 rows, 2, 2 and 4 of 16. A 64-core GPU's lane would take 26 and 18, an
  // 80-core GPU's 32 and 22; four lanes there 10 and 7, and 12 and 9 each.
  [[nodiscard]] static uint32_t decodeMostSplits(uint32_t laneCap, uint32_t lanes,
                                                 const AttentionCalibration &calibration);

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
  // layer's projection rows, whose gates multiply them), every row of them
  // (an output of each sequence's last row alone throws), each sequence
  // reading its history where `histories` say (dense: no block lists),
  // adding the layer's bias (weights.bias) and its rows' sinks. A dispatch
  // whose sequences read their layer in two placements runs as a dispatch per
  // run of them that reads one, and each run whose sequences each read their
  // history in one split runs the direct kernel in place of the split and
  // the reduce (RelativeAttentionPlan::direct), the same bits, into plain
  // rows alone: such a chunk, of more than 128 rows, has an out-projection
  // that reads no table, and a table input throws. With rows' tree each
  // sequence is a decode lane's block of at most 32 rows whose rows attend
  // their paths, which tests alone run: a decode step runs the single pass
  // (addDecode), whose reference they are.
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
  // `output`, every row as addAttention writes them, gated by layerRows'
  // projection rows (output.packed is not read).
  static PreparedInput addDecode(metal::CommandGraph &graph, const RelativeAttentionPlan &plan,
                                 const RelativeAttentionRows &layerRows, const RelativeAttentionWeights &weights,
                                 SplashKvLayer layer, std::span<const metal::MetalBuffer> pageTables,
                                 std::span<const kv::ChunkedPrefillParams> sequences, const AttentionRows &rows,
                                 const RelativeAttentionWorkspace &workspace, const AttentionOutputRows &output);

  // The commit of each lane's accepted path after a decode step, as
  // PagedAttention::addPathCommit's (SplashKvPathCommitParams), of the pages
  // relative attention stores.
  static void addPathCommit(metal::CommandGraph &graph, kv::Layout layout, uint32_t regions, uint32_t extentPages,
                            std::span<const metal::MetalBuffer> pageTables, std::span<const uint32_t> committed,
                            uint32_t laneRows, const metal::MetalBuffer &paths);

private:
  static void group(RelativeAttentionPlan &plan);
};

} // namespace splash::ops

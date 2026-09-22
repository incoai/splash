#pragma once
#include "FlashGDNFused.hpp"

#include <memory>
#include <array>
#include <span>
#include <vector>

// Synchronous exact rollback GPU-graph helper. No model projections are replayed.
// The caller completes each begin graph before commit and each commit graph
// before another begin. An aborted trial's live state is terminal/discarded.
namespace splash::flash {
inline constexpr const char *kFlashGDNLazyRollbackRoute =
    ";gdn-lazy-initial-f32-snapshot-prepared-bf16-kv-exact-prefix-recurrence-replay-v1";
[[nodiscard]] bool flashGDNLazyRollbackEnabled();

// CPU graph-construction telemetry, accumulated since this record's creation.
// Counts describe GDN layer graphs, not request cycles or GPU completions.
// Byte fields are logical records/operand footprints, never measured traffic
// or bandwidth. Prepared records replace the ordinary scratch writes.
struct FlashGDNLazyRollbackCounters final {
  std::array<uint64_t, 64> layer_trial_graphs_by_rows_and_lanes{};
  uint64_t layer_trial_graphs_built = 0;
  uint64_t trial_lane_rows_planned = 0;
  uint64_t r1_bypass_trial_graphs = 0;
  uint64_t commit_calls = 0;
  uint64_t no_replay_commit_fastpaths = 0;
  uint64_t full_accept_commit_fastpaths = 0;
  uint64_t full_accepted_lanes = 0;
  uint64_t partial_replay_graphs_built = 0;
  uint64_t partial_replay_layer_lanes_planned = 0;
  uint64_t partial_replay_rows_planned = 0;
  uint64_t terminal_lanes_discarded = 0;
  uint64_t all_terminal_commit_calls = 0;
  uint64_t aborted_trial_calls = 0;
  uint64_t aborted_trial_lanes = 0;
  uint64_t logical_eager_prefix_record_bytes_reference = 0;
  uint64_t logical_initial_state_snapshot_bytes = 0;
  uint64_t logical_initial_history_snapshot_bytes = 0;
  uint64_t logical_raw_qkv_copy_bytes = 0;
  uint64_t logical_saved_prework_footprint_bytes = 0;
  uint64_t logical_incremental_verify_record_write_bytes = 0;
  int64_t logical_verify_record_write_bytes_avoided = 0;
  uint64_t logical_replay_initial_state_read_bytes = 0;
  uint64_t logical_replay_state_write_bytes = 0;
  uint64_t logical_replay_history_write_bytes = 0;
  uint64_t logical_replay_operand_footprint_bytes = 0;

  void recordTrial(uint32_t rows, uint32_t lanes);
  void recordCommit(uint32_t rows, std::span<const uint32_t> retained);
  void recordAbort(uint32_t lanes);
  void add(const FlashGDNLazyRollbackCounters &other) noexcept;
};

class FlashGDNLazyRollback final {
public:
  FlashGDNLazyRollback(splash::metal::MetalBackend &backend, uint32_t maximumRows,
                  uint32_t maximumLanes);
  ~FlashGDNLazyRollback();
  FlashGDNLazyRollback(const FlashGDNLazyRollback &) = delete;
  FlashGDNLazyRollback &operator=(const FlashGDNLazyRollback &) = delete;
  FlashGDNLazyRollback(FlashGDNLazyRollback &&) noexcept;
  FlashGDNLazyRollback &operator=(FlashGDNLazyRollback &&) noexcept;

  [[nodiscard]] uint64_t begin(splash::metal::CommandGraph &graph,
      const splash::flash::FlashGDNWeights &weights,
      const splash::flash::FlashGDNBuffers &buffers,
      const splash::flash::FlashGDNState &state, uint32_t rows, uint32_t lanes,
      float epsilon = 1e-6f);
  void commit(splash::metal::CommandGraph &graph, uint64_t ticket,
              std::span<const uint32_t> retained);
  void abort(uint64_t ticket);
  [[nodiscard]] bool pending() const noexcept;
  [[nodiscard]] FlashGDNLazyRollbackCounters counters() const noexcept;
  [[nodiscard]] uint64_t allocationBytes() const noexcept;
  [[nodiscard]] static uint64_t plannedBytes(uint32_t maximumRows, uint32_t maximumLanes);
  [[nodiscard]] uint32_t maximumRows() const noexcept;
  [[nodiscard]] uint32_t maximumLanes() const noexcept;
  [[nodiscard]] bool canariesIntact() const noexcept;
  [[nodiscard]] std::vector<splash::metal::MetalBuffer> arenaBuffers() const;
  [[nodiscard]] splash::flash::FlashGDNBuffers savedBuffers() const;
  [[nodiscard]] splash::metal::MetalBuffer initialRecurrent() const;
  [[nodiscard]] splash::metal::MetalBuffer initialConvolution() const;
  [[nodiscard]] constexpr uint64_t initialRecurrentLaneStrideBytes() const noexcept {
    return splash::flash::flashGDNRecurrentLaneBytes();
  }
  [[nodiscard]] constexpr uint64_t initialConvolutionLaneStrideBytes() const noexcept {
    return splash::flash::flashGDNConvolutionLaneBytes();
  }
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash

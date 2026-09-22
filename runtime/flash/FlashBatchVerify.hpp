#pragma once

#include "flash/FlashForward.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace splash::flash {

inline constexpr const char *kFlashBatchVerifySemantics =
    "native-flash-batched-provisional-target-prefix-v1";

struct FlashBatchVerifyResult final {
  metal::CommandTiming timing;
  // Borrowed lane-major BF16[lanes,rows,vocabulary] and [lanes,rows,10240].
  // Valid through commitBatch(), overwritten by the next verifyBatch().
  metal::MetalBuffer logitsBF16;
  metal::MetalBuffer hiddenBF16;
  std::vector<uint64_t> logicalLengths;
  uint32_t lanes = 0;
  uint32_t rows = 0;
  uint32_t capacity = 0;
  metal::MetalBuffer greedyResultsU32{};
  uint32_t greedyRows = 0;
};

// Independent speculative target executor, sharing the original request-owned
// GDN/PLE/QSA state. Uniform real incoming rows1..4 per lane; no padding or
// invented tokens. Projections/HC/router/MoE/head operate on <=16 flattened
// rows in ONE ordered 48-layer graph. Each lane retains its own causal begin.
// Use from the single GPU/control owner: calls and request-state moves or
// destruction must be serialized. Backend, weights and trunk outlive executor.
class FlashBatchVerify final {
public:
  FlashBatchVerify(metal::MetalBackend &backend, const FlashWeights &weights,
                   FlashForward &trunk, uint32_t capacity,
                   uint32_t maximumLanes = 4, uint32_t maximumRows = 4,
                   bool sharedDenseRoutes = false);
  ~FlashBatchVerify();
  FlashBatchVerify(const FlashBatchVerify &) = delete;
  FlashBatchVerify &operator=(const FlashBatchVerify &) = delete;
  FlashBatchVerify(FlashBatchVerify &&) noexcept;
  FlashBatchVerify &operator=(FlashBatchVerify &&) noexcept;

  [[nodiscard]] FlashBatchVerifyResult
  verifyBatch(std::span<FlashRequestState *const> states,
              std::span<const uint32_t> laneMajorTokens, uint32_t rows = 4);
  // Exact same live state implementations and lane order as the pending trial.
  // retained is 1..rows (anchor + committed drafts), independently per lane.
  // retained0 terminally aborts that lane; its pointer may be null, including
  // after destruction. Surviving lanes preserve original cohort order.
  // One GPU restore graph; all survivors retaining full needs no command. Logical QSA
  // lengths roll back while later appends replace stale provisional cache rows.
  [[nodiscard]] metal::CommandTiming
  commitBatch(std::span<FlashRequestState *const> states,
              std::span<const uint32_t> retained);
  // Terminal cancellation only. Every still-live pending lane is poisoned and
  // cannot reenter AR/batch decode. Safe after state wrapper moves/destruction.
  void abortBatch() noexcept;
  [[nodiscard]] bool pending() const noexcept;
  [[nodiscard]] static uint64_t
  workspacePlannedBytes(uint32_t capacity, uint32_t maximumLanes = 4,
                        uint32_t maximumRows = 4);
  [[nodiscard]] uint64_t workspaceBytes() const noexcept;
  [[nodiscard]] uint64_t pleSSDStagingBytes() const noexcept;
  // Read-only actual storage accounting for eager tapes or per-layer lazy
  // initial snapshots and prepared operands; excludes shared work scratch.
  [[nodiscard]] uint64_t verificationGDNStorageBytes() const noexcept;
  [[nodiscard]] bool lazyGDNRollbackEnabled() const noexcept;
  [[nodiscard]] FlashGDNLazyRollbackCounters lazyGDNRollbackCounters() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash

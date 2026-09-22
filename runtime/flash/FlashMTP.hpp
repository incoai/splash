#pragma once

#include "flash/FlashWeights.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace splash::metal { class CommandGraph; }

namespace splash::flash {

inline constexpr const char *kFlashMTPSemantics =
    "qwen4-lightning-head-global-hidden-rms-bf16-fuse-qsa-hc-moe-v1";
inline constexpr const char *kFlashMTPTeacherCacheSemantics =
    "mtp-teacher-cache-only-original-preparation-pooling-no-attention-or-mlp-v1";

enum class FlashMTPLogits : uint8_t { None, Last, All };

class FlashMTPForward;
class FlashBatchMTPForward;
class FlashDenseCache;
struct FlashQSAState;
struct FlashQSAWorkspace;
struct FlashQSAFastWorkspace;
struct FlashQSAFastInputs;

// The trained head has one QSA cache and no GDN/PLE state. Its offset counts
// folded (previous hidden, next token) pairs rather than backbone tokens.
class FlashMTPState final {
public:
  FlashMTPState();
  ~FlashMTPState();
  FlashMTPState(const FlashMTPState &) = delete;
  FlashMTPState &operator=(const FlashMTPState &) = delete;
  FlashMTPState(FlashMTPState &&) noexcept;
  FlashMTPState &operator=(FlashMTPState &&) noexcept;
  [[nodiscard]] uint64_t logicalLength() const noexcept;
  [[nodiscard]] uint32_t capacity() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class FlashMTPForward;
  friend class FlashBatchMTPForward;
  friend struct FlashMTPTeacherPrimeOracleAccess;
};

struct FlashMTPResult final {
  metal::CommandTiming timing;
  // Borrowed BF16[logitRows,vocabularySize], absent for None. Both result
  // buffers belong to the shared head arena until the next forward() call.
  metal::MetalBuffer logitsBF16;
  uint32_t logitRows = 0;
  // All four streams BEFORE the head's final HC mixer: BF16[rows,10240].
  // Feeding the final row into this same head's next call is supported.
  metal::MetalBuffer hiddenBF16;
  uint32_t hiddenRows = 0;
  uint64_t logicalLength = 0;
  // Optional borrowed FlashGreedyGPURowResult[greedyRows], exact BF16 greedy
  // IDs from the same command as the head. Absent when disabled or >16 rows.
  metal::MetalBuffer greedyResultsU32{};
  uint32_t greedyRows = 0;
};

// Original checkpoint's embedded one-layer Lightning head. This class only
// folds head pairs; the caller owns target verification and token acceptance.
class FlashMTPForward final {
public:
  FlashMTPForward(metal::MetalBackend &backend, const FlashWeights &weights,
                  uint32_t capacity, uint32_t maximumRows = 128);
  ~FlashMTPForward();
  FlashMTPForward(const FlashMTPForward &) = delete;
  FlashMTPForward &operator=(const FlashMTPForward &) = delete;
  FlashMTPForward(FlashMTPForward &&) noexcept;
  FlashMTPForward &operator=(FlashMTPForward &&) noexcept;

  [[nodiscard]] FlashMTPState createState();
  [[nodiscard]] FlashMTPResult
  forward(FlashMTPState &state, metal::MetalBuffer previousHiddenBF16,
          std::span<const uint32_t> nextTokens,
          FlashMTPLogits logits = FlashMTPLogits::Last);
  // Teacher pairs provide independent target features. Only the QSA cache
  // survives priming; this operation returns no borrowed head features.
  [[nodiscard]] metal::CommandTiming
  primeTeacherCache(FlashMTPState &state, metal::MetalBuffer previousHiddenBF16,
                    std::span<const uint32_t> nextTokens);
  // QSA rollback restores the logical offset. A future append replaces its
  // stale token rows and every newly completed compression block.
  void truncate(FlashMTPState &state, uint64_t retainedLength);
  [[nodiscard]] static uint64_t requestStateBytes(uint32_t capacity);
  [[nodiscard]] uint64_t workspaceBytes() const noexcept;
  // Attention policy is shared by sequential and joint head execution.
  [[nodiscard]] const char *attentionRouteSemantics() const noexcept;
  // Proposal projection choices only; target verification remains external.
  // attentionRouteSemantics also appends an enabled proposal tag for status.
  [[nodiscard]] const char *projectionRouteSemantics() const noexcept;
  [[nodiscard]] static uint64_t attentionWorkspacePlannedBytes(uint32_t maximumRows);
  [[nodiscard]] static uint64_t denseCachePlannedBytes(const FlashWeights &weights);
  [[nodiscard]] std::vector<metal::MetalBuffer> cachedOperandsOnly() const;
  [[nodiscard]] static uint64_t workspacePlannedBytes(uint32_t capacity,
                                                      uint32_t maximumRows = 128);

private:
  friend class FlashBatchMTPForward;
  [[nodiscard]] FlashMTPResult
  forwardImpl(FlashMTPState &state, metal::MetalBuffer previousHiddenBF16,
              std::span<const uint32_t> nextTokens, FlashMTPLogits logits,
              bool teacherCacheOnly);
  [[nodiscard]] metal::MetalBackend &batchBackend() const;
  [[nodiscard]] const FlashWeights &batchWeights() const;
  [[nodiscard]] uint32_t batchCapacity() const;
  [[nodiscard]] std::mutex &batchMutex() const;
  [[nodiscard]] const FlashDenseCache *batchDenseCache() const;
  [[nodiscard]] bool batchFuseHC() const;
  [[nodiscard]] bool batchQSAF32() const;
  [[nodiscard]] bool batchQSAMPP() const;
  void batchAddQSA(metal::CommandGraph &graph, const FlashQSAFastInputs &inputs,
      FlashQSAState &state, FlashQSAWorkspace &workspace,
      FlashQSAFastWorkspace &fastWorkspace, uint32_t begin, uint32_t rows);
  [[nodiscard]] bool ownsState(const FlashMTPState &state) const noexcept;
  void batchProject(metal::CommandGraph &graph, const std::string &prefix,
      const metal::MetalBuffer &input, const metal::MetalBuffer &output,
      const metal::MetalBuffer &diagnostics, uint32_t rows,
      uint32_t logicalRows = 0, bool allowQmvProposals = true);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash

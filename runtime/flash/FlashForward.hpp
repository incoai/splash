#pragma once

#include "flash/FlashWeights.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/CommandGraph.hpp"

#include <cstdint>
#include <array>
#include <memory>
#include <mutex>
#include <span>
#include <string>

namespace splash::flash {
// Graph construction counters only. They do not imply command submission,
// GPU completion, emitted tokens, or numerical acceptance. Buckets0..16 use
// the exact supplied row count; bucket17 includes every larger row count.
struct FlashHCUpEncodedCounters final {
  uint64_t graphBuildAttempts = 0, graphBuildRealRows = 0;
  uint64_t geometryEligibleAttempts = 0, geometryEligibleRealRows = 0;
  uint64_t cachedEncodedCalls = 0, cachedEncodedRealRows = 0;
  uint64_t skippedDependenciesOff = 0, skippedMissingOperand = 0,
      skippedUnsupportedGeometry = 0;
  std::array<uint64_t, 18> graphBuildCallsByRows{};
  std::array<uint64_t, 18> cachedEncodedCallsByRows{};
  void recordAttempt(uint32_t rows) noexcept {
    ++graphBuildAttempts; graphBuildRealRows += rows;
    ++graphBuildCallsByRows[rows <= 16 ? rows : 17];
  }
  void recordEligible(uint32_t rows) noexcept {
    ++geometryEligibleAttempts; geometryEligibleRealRows += rows;
  }
  void recordCachedEncoded(uint32_t rows) noexcept {
    ++cachedEncodedCalls; cachedEncodedRealRows += rows;
    ++cachedEncodedCallsByRows[rows <= 16 ? rows : 17];
  }
};

inline constexpr const char *kFlashForwardSemantics =
    "native-flash-f32affine-mlx-bf16-reductions-fast-silu-precise-sigmoid-moe-idtie-qsa-highid-v3";

class FlashForward;
class FlashBatchForward;
class FlashBatchVerify;
class FlashBatchPrefill;
class FlashExpertDenseCache;
class FlashInt8ExpertStore;
struct FlashGDNLazyRollbackCounters;
struct FlashQSABulkCounters;
class FlashPLEFused;

class FlashRequestState final {
public:
  FlashRequestState();
  ~FlashRequestState();
  FlashRequestState(const FlashRequestState &) = delete;
  FlashRequestState &operator=(const FlashRequestState &) = delete;
  FlashRequestState(FlashRequestState &&) noexcept;
  FlashRequestState &operator=(FlashRequestState &&) noexcept;
  [[nodiscard]] uint64_t logicalLength() const noexcept;
  [[nodiscard]] uint32_t capacity() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class FlashForward;
  friend class FlashBatchForward;
  friend class FlashBatchVerify;
  friend class FlashBatchPrefill;
  friend class FlashDeepPrefixOracle;
};

struct FlashForwardResult final {
  metal::CommandTiming timing;
  // Borrowed contiguous BF16[logitRows,vocabularySize]. Consume or copy before
  // the next forward() call reuses the shared workspace, even for another state.
  metal::MetalBuffer logitsBF16;
  uint32_t logitRows = 0;
  uint64_t logicalLength = 0;
  uint32_t capacity = 0;
  // Optional borrowed BF16[consumedRows,10240] target residual streams before
  // the final HC mixer. Valid until the next trunk forward/verify call.
  metal::MetalBuffer hiddenBF16;
  // Optional borrowed exact GPU argmax records for the actual logit rows.
  metal::MetalBuffer greedyResultsU32{};
  uint32_t greedyRows = 0;
};

struct FlashPersistedOperandStatus final {
  uint64_t bf16Tensors = 0, f32Tensors = 0;
  uint64_t bf16PayloadBytes = 0, f32PayloadBytes = 0;
  std::string storeManifestSha256;
};

// Correctness-first synchronous text trunk. Every call submits one graph with
// all 48 layers, mandatory PLE, true QSA and original mixed-affine coefficients.
// One shared scratch arena is reused by ordered layers; request-owned GDN/QSA/
// PLE caches are separate. No DFlash or MTP substitutions occur in this route.
class FlashForward final {
public:
  FlashForward(metal::MetalBackend &backend, const FlashWeights &weights,
               uint32_t capacity, uint32_t maximumRows = 128,
               uint32_t maximumVerifyRows = 0);
  ~FlashForward();
  FlashForward(const FlashForward &) = delete;
  FlashForward &operator=(const FlashForward &) = delete;
  FlashForward(FlashForward &&) noexcept;
  FlashForward &operator=(FlashForward &&) noexcept;

  [[nodiscard]] FlashRequestState createState();
  [[nodiscard]] bool ownsState(const FlashRequestState &state) const noexcept;
  [[nodiscard]] FlashForwardResult
  // Trunk windows may contain up to maximumRows (1..2048). All-logits output
  // is bounded to min(maximumRows,128); oversized requests fail before mutation.
  forward(FlashRequestState &state, std::span<const uint32_t> tokens,
          bool returnAllLogits = false, bool captureHidden = false);
  // Incoming window is [anchor,draft1..draft7], 1..8 tokens. Always returns
  // all logits and pre-mixer hidden rows. State is provisional until resolved;
  // another trunk call is rejected while the shared verification tape is live.
  [[nodiscard]] FlashForwardResult
  verify(FlashRequestState &state, std::span<const uint32_t> tokens);
  // retained includes the anchor: 1 + accepted drafts, limited by actual EOS/
  // output budget. Correction/bonus output is not a consumed verify input.
  // Restores exact GDN/PLE prefix state and QSA logical length, without replay.
  // Keeping the full window needs no new command and returns zero timing.
  [[nodiscard]] metal::CommandTiming
  commitVerify(FlashRequestState &state, uint32_t retained);
  // Cancelling a provisional trial is terminal for that state; it never
  // promotes partially modified caches back into a healthy request.
  void abortVerify(FlashRequestState &state) noexcept;
  // Admission estimate, not a reservation: the worker must apply its real
  // memory policy before createState(). Each physical buffer is rounded16K.
  [[nodiscard]] static uint64_t requestStateBytes(uint32_t capacity);
  // Extra shared prefix tape provisioned by constructor maximumVerifyRows.
  // 0 leaves the autoregressive allocation footprint unchanged; max is16.
  [[nodiscard]] static uint64_t verificationWorkspaceBytes(uint32_t maximumVerifyRows);
  // Conservative fixed-workspace admission estimate. Includes optional verify
  // tapes and its possible fused-GDN prehistory buffer; no request state.
  [[nodiscard]] static uint64_t
  workspacePlannedBytes(uint32_t capacity, uint32_t maximumRows = 128,
                        uint32_t maximumVerifyRows = 0);
  // Optional plan-dependent hot-expert allocations; caller reserves these
  // before construction in addition to the fixed workspace estimate.
  [[nodiscard]] static uint64_t expertCachePlannedBytes(const FlashWeights &weights);
  [[nodiscard]] static uint64_t floatDenseCachePlannedBytes(const FlashWeights &weights);
  [[nodiscard]] static uint64_t int8HeadPlannedBytes(const FlashWeights &weights);
  [[nodiscard]] uint64_t workspaceBytes() const noexcept;
  [[nodiscard]] uint64_t pleSSDStagingBytes() const noexcept;
  [[nodiscard]] uint64_t verificationGDNStorageBytes() const noexcept;
  [[nodiscard]] bool lazyGDNRollbackEnabled() const noexcept;
  [[nodiscard]] FlashGDNLazyRollbackCounters lazyGDNRollbackCounters() const noexcept;
  [[nodiscard]] std::string kernelRoutes() const;
  [[nodiscard]] FlashHCUpEncodedCounters hcUpEncodedCounters() const;
  // Completed graph construction only; these do not imply GPU submission.
  [[nodiscard]] uint64_t qsaOutF32N32EncodedCalls() const;
  [[nodiscard]] uint64_t qsaOutF32N32EncodedRealRows() const;
  // Successful target commands only, after GPU completion and diagnostics.
  [[nodiscard]] FlashQSABulkCounters qsaBulkPrefillCounters() const;
  // Immutable CPU metadata only; reading never maps files or submits work.
  [[nodiscard]] FlashPersistedOperandStatus persistedOperandStatus() const;
  // Verified saved BF16/F32 operands plus derived selected-INT8 payload/rank
  // backing only. Original checkpoint, PLE, embeddings and original Q8 head
  // code views are deliberately absent. Metadata enumeration submits no work.
  [[nodiscard]] std::vector<metal::MetalBuffer> cachedOperandsOnly() const;
  // Immutable derived operand; callers may retain its tensor view to share
  // the allocation without duplicating weights.
  [[nodiscard]] const FlashTensor *cachedVocabulary() const;
  [[nodiscard]] const FlashTensor *cachedFloatVocabulary() const;
  // Immutable selected-expert sidecar metadata and shared prefill operands.
  [[nodiscard]] const FlashInt8ExpertStore *batchInt8ExpertStore() const noexcept;
  // Borrowed I64[layer,maximumRows,10] diagnostic capture. Only the first
  // capturedRows entries of each layer are valid, until the next trunk call.
  [[nodiscard]] metal::MetalBuffer capturedExpertIDs(uint32_t &capturedRows,
                                                    uint32_t &rowStride) const;

private:
  friend class FlashBatchForward;
  friend class FlashBatchVerify;
  friend class FlashBatchPrefill;
  [[nodiscard]] metal::MetalBackend &batchBackend() const;
  [[nodiscard]] const FlashWeights &batchWeights() const;
  [[nodiscard]] uint32_t batchCapacity() const;
  [[nodiscard]] std::mutex &batchMutex() const;
  [[nodiscard]] bool batchDenseSmallRowsEnabled() const noexcept;
  // Shared prefill bridge uses the source's frozen opt-in and immutable BF16
  // operands. Existing caller gate/up planes provide tail scratch only.
  [[nodiscard]] bool batchSharedExpertFused(metal::CommandGraph &graph,
      const std::string &prefix, const metal::MetalBuffer &input,
      const metal::MetalBuffer &activated, const metal::MetalBuffer &diagnostics,
      uint32_t rows, const metal::MetalBuffer &tailGate,
      const metal::MetalBuffer &tailUp);
  [[nodiscard]] bool batchInt8HeadEnabled() const noexcept;
  [[nodiscard]] bool batchFloatDenseEnabled() const noexcept;
  // Same model-owned F32 cache and padding arena as the trunk. Returns false
  // for disabled dependencies, missing operands, or unqualified HC rows/roles.
  [[nodiscard]] bool batchHCFusedUpMixF32(metal::CommandGraph &graph,
      const std::string &upPrefix, const metal::MetalBuffer &normalized,
      const metal::MetalBuffer &activatedDown, const metal::MetalBuffer &mixed,
      const metal::MetalBuffer &diagnostics, uint32_t rows);
  [[nodiscard]] const FlashExpertDenseCache *batchExpertCache(uint32_t layer) const;
  [[nodiscard]] const FlashPLEFused *batchPLELookup() const noexcept;
  // CPU metadata preflight, under batchMutex(): caller-owned Shared output
  // must not overwrite source scratch or any immutable model operand.
  void batchValidateExternalDestination(const metal::MetalBuffer &destination) const;
  void batchProject(metal::CommandGraph &graph, const std::string &prefix,
                    const metal::MetalBuffer &input, const metal::MetalBuffer &output,
                    const metal::MetalBuffer &diagnostics, uint32_t rows);
  [[nodiscard]] FlashForwardResult
  forwardImpl(FlashRequestState &state, std::span<const uint32_t> tokens,
              bool returnAllLogits, bool captureHidden, bool verification);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash

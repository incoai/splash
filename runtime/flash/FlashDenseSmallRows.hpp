#pragma once

#include "FlashDenseCache.hpp"

namespace splash::flash {

enum class FlashDenseSmallRowsTile : uint32_t {
  M8N64 = 0, M8N128 = 1, M16N64 = 2, M16N128 = 3,
};

inline constexpr const char *kFlashDenseSmallRowsSemantics =
    "cached-bf16-exact-input-zero-padded-m8-m16-mpp-f32accum-bf16-v1";

// One reusable Shared BF16 activation arena; no weights are expanded here.
// Only paddedRows*actualK words are written, with actualK row stride. The
// untouched remainder is available for oracle canaries. Ordered commands may
// reuse it, but concurrent execution with the same workspace is unsupported.
class FlashDenseSmallRowsWorkspace final {
public:
  explicit FlashDenseSmallRowsWorkspace(metal::MetalBackend &backend,
                                         uint32_t maximumInputSize = 32768);
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }
  [[nodiscard]] uint32_t maximumInputSize() const noexcept { return maximumInputSize_; }
  [[nodiscard]] metal::MetalBuffer paddedInput() const { return paddedInput_; }

private:
  metal::MetalBackend *backend_;
  uint32_t maximumInputSize_;
  uint64_t allocatedBytes_ = 0;
  metal::MetalBuffer paddedInput_;
  friend void addDenseBF16SmallRows(metal::MetalBackend &, metal::CommandGraph &,
      metal::MetalBuffer, const FlashTensor &, metal::MetalBuffer,
      metal::MetalBuffer, uint32_t, FlashDenseSmallRowsWorkspace &,
      FlashDenseSmallRowsTile);
};

// Separate precision policy: weights are already cached BF16 operands, F32
// accumulation, final BF16. Input/output are unpadded Shared [rows,K]/[rows,N].
// rows1..16; N%64==0, K%32==0. Copies real BF16 input bits exactly, pads positive
// zero to ceil(rows/M)*M≤16, and guards real output rows. N128 column tails use
// a separate N64 dispatch. No source tensor/model/cache/default is changed.
void addDenseBF16SmallRows(metal::MetalBackend &backend, metal::CommandGraph &graph,
                           metal::MetalBuffer input, const FlashTensor &weights,
                           metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                           uint32_t rows, FlashDenseSmallRowsWorkspace &workspace,
                           FlashDenseSmallRowsTile tile);

} // namespace splash::flash

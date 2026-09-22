#pragma once

#include "FlashWeights.hpp"
#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::flash {

inline constexpr uint32_t kFlashGDNKeyHeads = 16;
inline constexpr uint32_t kFlashGDNValueHeads = 48;
inline constexpr uint32_t kFlashGDNHeadDimension = 128;
inline constexpr uint32_t kFlashGDNConvolutionWidth = 10240;
inline constexpr uint32_t kFlashGDNOutputWidth = 6144;
inline constexpr uint32_t kFlashGDNMaximumRows = 2048;
inline constexpr const char *kFlashGDNSemantics =
    "qwen4-bf16-l2-sigmoid-f32-delta-v1";

// The history holds the three most recent *projected*, unconvolved qkv rows,
// oldest first. Recurrent state is F32[lanes,48,128(value),128(key)]. These
// mutable buffers must belong to one layer and one ordered request stream;
// initially both must be zero. A successful dispatch consumes every supplied
// row and leaves state valid for the next dispatch, including rows=1 or 2.
struct FlashGDNState final {
  metal::MetalBuffer convolution;
  metal::MetalBuffer recurrent;
  // Zero selects the tight lane extent. Nonzero strides must be aligned to
  // the element size and at least the corresponding tight extent.
  uint64_t convolutionLaneStrideBytes = 0;
  uint64_t recurrentLaneStrideBytes = 0;
};

struct FlashGDNWeights final {
  // BF16[10240,4,1] or BF16[10240,4], channel-major causal taps.
  const FlashTensor *convolution = nullptr;
  const FlashTensor *aLog = nullptr;     // BF16[48], logarithm retained.
  const FlashTensor *timeBias = nullptr; // BF16[48].
  const FlashTensor *norm = nullptr;     // BF16[128], direct gamma.
};

// All row buffers are contiguous BF16 and lane-major [lanes,rows,width].
// a/b are separate projected gates, not an interleaved Splash projection.
struct FlashGDNBuffers final {
  metal::MetalBuffer qkv; // width=10240; input, preserved.
  metal::MetalBuffer z;   // width=6144; input, preserved.
  metal::MetalBuffer a;   // width=48; input, preserved.
  metal::MetalBuffer b;   // width=48; input, preserved.
  metal::MetalBuffer mixed;         // BF16 width=10240; scratch.
  metal::MetalBuffer decay;         // F32 width=48; scratch.
  metal::MetalBuffer beta;          // BF16 width=48; scratch.
  metal::MetalBuffer recurrentRows; // BF16 width=6144; scratch.
  metal::MetalBuffer output;        // BF16 width=6144.
  // Caller-cleared U32 sticky flags: 2 invalid parameters, 4 nonfinite math.
  // The caller must inspect this buffer after command completion.
  metal::MetalBuffer diagnostics;
};

[[nodiscard]] constexpr uint64_t flashGDNConvolutionLaneBytes() noexcept {
  return uint64_t{3} * kFlashGDNConvolutionWidth * sizeof(uint16_t);
}
[[nodiscard]] constexpr uint64_t flashGDNRecurrentLaneBytes() noexcept {
  return uint64_t{kFlashGDNValueHeads} * kFlashGDNHeadDimension *
         kFlashGDNHeadDimension * sizeof(float);
}

// Adds the convolution, Qwen4 L2 normalization/gates, exact sequential delta
// recurrence, sigmoid gated direct-gamma RMSNorm, and history carry. Every
// token is processed in order within its lane. No weights or activations are
// quantized here, and no existing Splash GDN kernel or state ABI is reused.
void addGDN(metal::CommandGraph &graph, const FlashGDNWeights &weights,
            const FlashGDNBuffers &buffers, const FlashGDNState &state,
            uint32_t rows, uint32_t lanes = 1, float normEpsilon = 1e-6f);

} // namespace splash::flash

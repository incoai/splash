// The post-norm block's add/norm chain against a CPU reference: the
// attention branch's output norm adds the block's input; the FFN's norm
// reads the sum; the FFN output's norm adds back to the sum. The FFN's
// projections run as an identity (the FFN output is its normed input), which
// isolates the chain the target's composition encodes from the projection
// numerics the quantized kernels' own tests hold.
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Normalization.hpp"
#include "ops/ValueMix.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::metal::CommandGraph;
using splash::ops::NormWeights;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;

constexpr uint32_t kRows = 8;
constexpr uint32_t kHidden = 512;

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }

private:
  uint64_t state_;
};

MetalBuffer randomBfloat(MetalBackend &backend, uint64_t count, Random &random, const char *label) {
  MetalBuffer buffer =
      backend.allocateBuffer(count * sizeof(uint16_t), BufferStorage::Shared, label);
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    values[index] = floatToBf16(random.unit());
  return buffer;
}

MetalBuffer normWeights(MetalBackend &backend, uint64_t count, Random &random, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(count * sizeof(float), BufferStorage::Shared, label);
  auto *values = static_cast<float *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    values[index] = 0.5f + random.unit();
  return buffer;
}

std::vector<float> rmsNorm(const std::vector<float> &x, const std::vector<float> &weight) {
  std::vector<float> out(x.size());
  for (uint32_t row = 0; row < kRows; ++row) {
    double sum = 0.0;
    for (uint32_t c = 0; c < kHidden; ++c)
      sum += double(x[uint64_t{row} * kHidden + c]) * x[uint64_t{row} * kHidden + c];
    const float inverse = 1.0f / std::sqrt(float(sum / kHidden) + 1e-6f);
    for (uint32_t c = 0; c < kHidden; ++c)
      out[uint64_t{row} * kHidden + c] =
          x[uint64_t{row} * kHidden + c] * inverse * weight[c];
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: ripple-chain-test METALLIB");
    splash::metal::MetalBackend backend(argv[1]);
    Random random(0xCA11A5EEDULL);

    // The inputs: the attention's raw output rows and the block's input.
    MetalBuffer attentionRaw = randomBfloat(backend, uint64_t{kRows} * kHidden, random, "attn raw");
    MetalBuffer blockInput = randomBfloat(backend, uint64_t{kRows} * kHidden, random, "block input");
    MetalBuffer normalized = backend.allocateBuffer(
        uint64_t{kRows} * kHidden * 2, BufferStorage::Shared, "normalized");
    MetalBuffer blockOut = backend.allocateBuffer(
        uint64_t{kRows} * kHidden * 2, BufferStorage::Shared, "block out");

    // The norms: F32 multipliers the norm kernels read directly.
    MetalBuffer postAttnW = normWeights(backend, kHidden, random, "post-attn norm");
    MetalBuffer ffnNormW = normWeights(backend, kHidden, random, "ffn norm");
    MetalBuffer postFfnW = normWeights(backend, kHidden, random, "post-ffn norm");
    const NormWeights postAttnNorm{postAttnW, true};
    const NormWeights ffnNorm{ffnNormW, true};
    const NormWeights postFfnNorm{postFfnW, true};

    // The attention's raw values, saved before the block consumes them.
    std::vector<float> attnOriginal(kRows * kHidden), blockOriginal(kRows * kHidden);
    {
      const auto *a = static_cast<const uint16_t *>(attentionRaw.contents());
      const auto *b = static_cast<const uint16_t *>(blockInput.contents());
      for (uint32_t i = 0; i < kRows * kHidden; ++i) {
        attnOriginal[i] = bf16ToFloat(a[i]);
        blockOriginal[i] = bf16ToFloat(b[i]);
      }
    }

    // The chain: the attention output's norm adds the block's input; the
    // FFN's norm reads the sum; the FFN is an identity (its output is its
    // normed input); the output's norm adds back to the sum.
    CommandGraph chain;
    splash::ops::Normalization::addRms(chain, attentionRaw, postAttnNorm, normalized, kHidden, kRows);
    splash::ops::ValueMix::sum(chain, normalized, blockInput, attentionRaw, kRows, kHidden);
    splash::ops::Normalization::addRms(chain, attentionRaw, ffnNorm, normalized, kHidden, kRows);
    splash::ops::Normalization::addRms(chain, normalized, postFfnNorm, normalized, kHidden, kRows);
    splash::ops::ValueMix::sum(chain, normalized, attentionRaw, blockOut, kRows, kHidden);
    static_cast<void>(backend.submitCommandAsync(chain.dispatches()).wait());

    // The CPU reference of the chain.
    const auto attnPost = rmsNorm(attnOriginal, [&] {
      const auto *values = static_cast<const float *>(postAttnW.contents());
      return std::vector<float>(values, values + kHidden);
    }());
    std::vector<float> ffnIn(kRows * kHidden);
    for (uint32_t i = 0; i < kRows * kHidden; ++i)
      ffnIn[i] = attnPost[i] + blockOriginal[i];
    const auto ffnNormed = rmsNorm(ffnIn, [&] {
      const auto *values = static_cast<const float *>(ffnNormW.contents());
      return std::vector<float>(values, values + kHidden);
    }());
    const auto ffnPost = rmsNorm(ffnNormed, [&] {
      const auto *values = static_cast<const float *>(postFfnW.contents());
      return std::vector<float>(values, values + kHidden);
    }());
    std::vector<float> blockExpected(kRows * kHidden);
    for (uint32_t i = 0; i < kRows * kHidden; ++i)
      blockExpected[i] = ffnPost[i] + ffnIn[i];

    const auto *actual = static_cast<const uint16_t *>(blockOut.contents());
    for (uint32_t i = 0; i < kRows * kHidden; ++i) {
      const float expected = blockExpected[i];
      const float value = bf16ToFloat(actual[i]);
      if (!std::isfinite(value) ||
          std::fabs(value - expected) > 0.02f + 0.02f * std::fabs(expected)) {
        std::cerr << "chain " << i << ": " << value << " vs " << expected << '\n';
        throw std::runtime_error("the add/norm chain diverged from its reference");
      }
    }
    std::cout << "ripple-chain-test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ripple-chain-test: FAIL: " << error.what() << '\n';
    return 1;
  }
}

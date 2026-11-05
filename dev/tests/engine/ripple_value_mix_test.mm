// The value residual's mix and placed value projection against a direct CPU
// reference: the mix lerps the normed input to the scaled embeddings by the
// sigmoid of the block's lambda, and a value projection placed at the packed
// row's value columns writes them there, into a destination the fused
// projection's rows share. These hold the two primitives RippleTarget's
// mixer runs to the reference the block graph states.
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"
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
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using namespace splash::ops;

constexpr uint32_t kRows = 8;
constexpr uint32_t kHidden = 512;
constexpr uint32_t kValueRows = 256;   // the value rows: 2 KV heads of 128
constexpr uint32_t kPacked = 1536;     // queries(1024) + keys(256) + values(256)
constexpr uint32_t kFusedRows = 1280;  // queries and keys, the fused projection's rows
constexpr uint32_t kValueColumns = kPacked - kFusedRows;
constexpr float kLambda = 0.75F;

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

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: ripple-value-mix METALLIB");
    MetalBackend backend(argv[1]);
    Random random(0x1a4da7ULL);

    // The mix: v_in = sigmoid(lambda) * a + (1 - sigmoid(lambda)) * emb.
    MetalBuffer a = randomBfloat(backend, uint64_t{kRows} * kHidden, random, "normed");
    MetalBuffer emb = randomBfloat(backend, uint64_t{kRows} * kHidden, random, "embeddings");
    MetalBuffer lambda = backend.allocateBuffer(4, BufferStorage::Shared, "lambda");
    *static_cast<float *>(lambda.contents()) = kLambda;
    MetalBuffer vIn = backend.allocateBuffer(uint64_t{kRows} * kHidden * 2, BufferStorage::Shared,
                                             "value input");
    CommandGraph mix;
    splash::ops::ValueMix::add(mix, a, emb, lambda, vIn, kRows, kHidden);
    static_cast<void>(backend.submitCommandAsync(mix.dispatches()).wait());

    const float sigmoid = 1.0f / (1.0f + std::exp(-kLambda));
    const auto *aValues = static_cast<const uint16_t *>(a.contents());
    const auto *embValues = static_cast<const uint16_t *>(emb.contents());
    const auto *mixed = static_cast<const uint16_t *>(vIn.contents());
    for (uint32_t index = 0; index < kRows * kHidden; ++index) {
      const float expected =
          sigmoid * bf16ToFloat(aValues[index]) + (1.0f - sigmoid) * bf16ToFloat(embValues[index]);
      const float actual = bf16ToFloat(mixed[index]);
      if (std::fabs(actual - expected) > 0.02f + 0.02f * std::fabs(expected))
        throw std::runtime_error("the value mix diverged from its reference at " +
                                 std::to_string(index));
    }

    // The placed projection: the value rows write at the packed row's value
    // columns, beside a fused projection's rows the packed buffer already
    // holds.
    MetalBuffer packed = backend.allocateBuffer(uint64_t{kRows} * kPacked * 2, BufferStorage::Shared,
                                                "packed rows");
    auto *packedValues = static_cast<uint16_t *>(packed.contents());
    for (uint32_t row = 0; row < kRows; ++row)
      for (uint32_t column = 0; column < kFusedRows; ++column)
        packedValues[uint64_t{row} * kPacked + column] = floatToBf16(random.unit());

    // The value weights: F32 rows, whose projection the float path runs.
    std::vector<float> weights(uint64_t{kValueRows} * kHidden);
    for (float &value : weights) value = random.unit();
    MetalBuffer weightBuffer = backend.allocateBuffer(
        weights.size() * sizeof(float), BufferStorage::Shared, "value weights");
    std::memcpy(weightBuffer.contents(), weights.data(), weights.size() * sizeof(float));
    Projection value(kValueRows, kHidden,
                     BlockWeights{std::vector{QuantizedSegment::floats(kValueRows, kHidden, weightBuffer)}});
    const Projection placed = Projection::placed(kPacked, kFusedRows, std::move(value));

    const uint32_t lanes = 1;
    CommandGraph project;
    const Linear linear(backend.capabilities());
    const LinearPlan plan = linear.decodePlan(placed, lanes);
    const LinearScratchSize size = plan.scratchSize();
    const auto zeros = [&](uint64_t bytes, const char *label) {
      MetalBuffer buffer = backend.allocateBuffer(std::max<uint64_t>(bytes, 16), BufferStorage::Shared, label);
      std::memset(buffer.contents(), 0, std::max<uint64_t>(bytes, 16));
      return buffer;
    };
    linear.add(project,
               {.input = vIn, .output = packed,
                .scratch = LinearScratch{zeros(size.input, "scratch-input"),
                                         zeros(size.sums, "scratch-sums"),
                                         zeros(size.partials, "scratch-partials"),
                                         zeros(size.counters, "scratch-counters")}},
               placed, plan);
    static_cast<void>(backend.submitCommandAsync(project.dispatches()).wait());

    // The reference: the mixed rows projected by the value weights, at the
    // packed row's value columns.
    for (uint32_t row = 0; row < kRows; ++row) {
      for (uint32_t column = 0; column < kValueColumns; ++column) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < kHidden; ++k)
          sum += bf16ToFloat(mixed[uint64_t{row} * kHidden + k]) * weights[uint64_t{column} * kHidden + k];
        const float expected = bf16ToFloat(packedValues[uint64_t{row} * kPacked + kFusedRows + column]);
        if (std::fabs(expected - sum) > 0.02f + 0.02f * std::fabs(sum))
          throw std::runtime_error("the placed value projection diverged at " +
                                   std::to_string(row) + "," + std::to_string(column));
      }
    }
    std::cout << "ripple_value_mix_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ripple_value_mix_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}

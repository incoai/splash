// The video text encoder's attention kernels (kernels/video/text_encoder.metal)
// against a double-precision CPU reference: the Q/K RMS norm and rotary
// embedding of the fused QKV rows, then causal GQA attention, for prompts of
// one row, a partial simdgroup's worth and more rows than a threadgroup grid
// side, with Qwen3-VL-32B's 64 query and 8 KV heads and a 2:1 GQA case.
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "video/TextEncoderABI.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t kHead = SPLASH_TEXT_ENCODER_HEAD_DIMENSION;

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

float fromBf16(uint16_t bits) { return std::bit_cast<float>(uint32_t{bits} << 16); }
uint16_t toBf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
float roundBf16(float value) { return fromBf16(toBf16(value)); }

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

MetalBuffer upload(MetalBackend &backend, const void *data, uint64_t bytes) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, "encoder test");
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

std::vector<float> download(const MetalBuffer &buffer, size_t count) {
  std::vector<float> result(count);
  const auto *bits = static_cast<const uint16_t *>(buffer.contents());
  for (size_t i = 0; i < count; ++i) result[i] = fromBf16(bits[i]);
  return result;
}

// Largest |got - want| over max(|want|, floor), the bf16 output's own scale.
double relativeError(const std::vector<float> &got, const std::vector<double> &want, double floor) {
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i)
    worst = std::max(worst, std::abs(double(got[i]) - want[i]) / std::max(std::abs(want[i]), floor));
  return worst;
}

void runCase(MetalBackend &backend, uint32_t rows, uint32_t queryHeads, uint32_t kvHeads) {
  const std::string label = std::to_string(rows) + " rows, " + std::to_string(queryHeads) + "/" +
                            std::to_string(kvHeads) + " heads";
  Random random(rows * 131 + queryHeads);
  const uint32_t heads = queryHeads + 2 * kvHeads;
  const uint32_t width = heads * kHead;
  std::vector<uint16_t> packed(size_t{rows} * width), queryNorm(kHead), keyNorm(kHead);
  for (uint16_t &value : packed) value = toBf16(4.0F * random.unit());
  for (uint32_t d = 0; d < kHead; ++d) {
    queryNorm[d] = toBf16(1.0F + 0.5F * random.unit());
    keyNorm[d] = toBf16(1.0F + 0.5F * random.unit());
  }
  std::vector<float> frequencies(kHead / 2);
  for (uint32_t i = 0; i < kHead / 2; ++i)
    frequencies[i] = 1.0F / std::pow(5000000.0F, float(2 * i) / float(kHead));

  MetalBuffer packedBuffer = upload(backend, packed.data(), packed.size() * 2);
  MetalBuffer queryNormBuffer = upload(backend, queryNorm.data(), kHead * 2);
  MetalBuffer keyNormBuffer = upload(backend, keyNorm.data(), kHead * 2);
  MetalBuffer frequencyBuffer = upload(backend, frequencies.data(), frequencies.size() * 4);
  const uint64_t queryBytes = uint64_t{queryHeads} * rows * kHead * 2;
  const uint64_t kvBytes = uint64_t{kvHeads} * rows * kHead * 2;
  MetalBuffer queries = backend.allocateBuffer(queryBytes, BufferStorage::Shared, "queries");
  MetalBuffer keys = backend.allocateBuffer(kvBytes, BufferStorage::Shared, "keys");
  MetalBuffer values = backend.allocateBuffer(kvBytes, BufferStorage::Shared, "values");
  MetalBuffer output = backend.allocateBuffer(uint64_t{rows} * queryHeads * kHead * 2, BufferStorage::Shared,
                                              "attention");
  CommandGraph graph;
  graph.add("text_encoder_qk_norm_rope",
            {packedBuffer, queryNormBuffer, keyNormBuffer, frequencyBuffer, queries, keys, values},
            TextEncoderHeadParams{rows, queryHeads, kvHeads, 1e-6F}, {rows, heads, 1}, {32, 1, 1});
  graph.add("text_encoder_attention", {queries, keys, values, output},
            TextEncoderHeadParams{rows, queryHeads, kvHeads, 1.0F / std::sqrt(float(kHead))},
            {rows, queryHeads, 1}, {32, 1, 1});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  // Reference: the norm rounded to bf16 as the kernel stores it, the
  // rotation in double from those values.
  std::vector<double> wantQ(size_t{queryHeads} * rows * kHead), wantK(size_t{kvHeads} * rows * kHead);
  std::vector<float> wantV(size_t{kvHeads} * rows * kHead);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t head = 0; head < heads; ++head) {
      const uint16_t *x = &packed[size_t{row} * width + size_t{head} * kHead];
      if (head >= queryHeads + kvHeads) {
        const uint32_t v = head - queryHeads - kvHeads;
        for (uint32_t d = 0; d < kHead; ++d) wantV[(size_t{v} * rows + row) * kHead + d] = fromBf16(x[d]);
        continue;
      }
      const bool query = head < queryHeads;
      double squares = 0.0;
      for (uint32_t d = 0; d < kHead; ++d) squares += double(fromBf16(x[d])) * fromBf16(x[d]);
      const double inverse = 1.0 / std::sqrt(squares / kHead + 1e-6);
      std::vector<double> normalized(kHead);
      for (uint32_t d = 0; d < kHead; ++d)
        normalized[d] = roundBf16(float(fromBf16(x[d]) * inverse *
                                        fromBf16(query ? queryNorm[d] : keyNorm[d])));
      double *out = query ? &wantQ[(size_t{head} * rows + row) * kHead]
                          : &wantK[(size_t{head - queryHeads} * rows + row) * kHead];
      for (uint32_t d = 0; d < kHead / 2; ++d) {
        const double angle = double(row) * frequencies[d];
        out[d] = normalized[d] * std::cos(angle) - normalized[d + kHead / 2] * std::sin(angle);
        out[d + kHead / 2] = normalized[d + kHead / 2] * std::cos(angle) + normalized[d] * std::sin(angle);
      }
    }
  const std::vector<float> gotQ = download(queries, wantQ.size());
  const std::vector<float> gotK = download(keys, wantK.size());
  require(relativeError(gotQ, wantQ, 0.25) < 1.6e-2, label + ": queries exceed one bf16 step");
  require(relativeError(gotK, wantK, 0.25) < 1.6e-2, label + ": keys exceed one bf16 step");
  require(download(values, wantV.size()) == wantV, label + ": values are not copied exactly");

  // Attention in double from the kernel's own bf16 queries and keys.
  std::vector<double> wantOut(size_t{rows} * queryHeads * kHead);
  const uint32_t group = queryHeads / kvHeads;
  for (uint32_t head = 0; head < queryHeads; ++head)
    for (uint32_t row = 0; row < rows; ++row) {
      const float *q = &gotQ[(size_t{head} * rows + row) * kHead];
      const uint32_t kv = head / group;
      std::vector<double> scores(row + 1);
      double maximum = -INFINITY;
      for (uint32_t j = 0; j <= row; ++j) {
        const float *k = &gotK[(size_t{kv} * rows + j) * kHead];
        double dot = 0.0;
        for (uint32_t d = 0; d < kHead; ++d) dot += double(q[d]) * k[d];
        scores[j] = dot / std::sqrt(double(kHead));
        maximum = std::max(maximum, scores[j]);
      }
      double total = 0.0;
      for (double &score : scores) total += (score = std::exp(score - maximum));
      for (uint32_t d = 0; d < kHead; ++d) {
        double sum = 0.0;
        for (uint32_t j = 0; j <= row; ++j) sum += scores[j] * wantV[(size_t{kv} * rows + j) * kHead + d];
        wantOut[(size_t{row} * queryHeads + head) * kHead + d] = sum / total;
      }
    }
  require(relativeError(download(output, wantOut.size()), wantOut, 0.25) < 1.6e-2,
          label + ": attention exceeds one bf16 step");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 2)
        throw std::invalid_argument("usage: text-encoder METALLIB");
      MetalBackend backend(argv[1]);
      for (const uint32_t rows : {1U, 19U, 70U}) runCase(backend, rows, 64, 8);
      runCase(backend, 33, 4, 2);
      std::cout << "text_encoder_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "text_encoder_metal_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}

// The video text encoder's attention kernels (kernels/video/text_encoder.metal)
// against a double-precision CPU reference: the Q/K RMS norm and rotary
// embedding of the fused QKV rows, then causal GQA attention, for prompts of
// one row, a partial simdgroup's worth and more rows than a threadgroup grid
// side, with Qwen3-VL-32B's 64 query and 8 KV heads and a 2:1 GQA case.
#include "LinearNumerics.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/video/TextEncoderABI.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;
using splash::test::require;
using splash::test::sharedBuffer;

constexpr uint32_t kHead = SPLASH_TEXT_ENCODER_HEAD_DIMENSION;

MetalBuffer upload(MetalBackend &backend, const void *data, uint64_t bytes) {
  MetalBuffer buffer = sharedBuffer(backend, bytes);
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

std::vector<float> download(const MetalBuffer &buffer, size_t count) {
  const auto *bits = static_cast<const uint16_t *>(buffer.contents());
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) values[i] = bf16ToFloat(bits[i]);
  return values;
}

// Within one bf16 step of want, relative to max(|want|, 0.25).
bool nearBf16(const std::vector<float> &got, const std::vector<double> &want) {
  for (size_t i = 0; i < want.size(); ++i)
    if (std::abs(got[i] - want[i]) >= 1.6e-2 * std::max(std::abs(want[i]), 0.25)) return false;
  return true;
}

void runCase(MetalBackend &backend, uint32_t rows, uint32_t queryHeads, uint32_t kvHeads) {
  const std::string label = std::to_string(rows) + " rows, " + std::to_string(kvHeads) + " KV heads: ";
  std::mt19937 random(rows * 131 + queryHeads);
  std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
  const uint32_t heads = queryHeads + 2 * kvHeads;
  const size_t queryCount = size_t{queryHeads} * rows * kHead, kvCount = size_t{kvHeads} * rows * kHead;
  std::vector<uint16_t> packed(size_t{rows} * heads * kHead), norms(2 * kHead);
  for (uint16_t &value : packed) value = floatToBf16(4.0F * unit(random));
  for (uint16_t &value : norms) value = floatToBf16(1.0F + 0.5F * unit(random));
  std::vector<float> frequencies(kHead / 2);
  for (uint32_t i = 0; i < kHead / 2; ++i)
    frequencies[i] = 1.0F / std::pow(5000000.0F, float(2 * i) / float(kHead));

  MetalBuffer queries = sharedBuffer(backend, queryCount * 2), output = sharedBuffer(backend, queryCount * 2);
  MetalBuffer keys = sharedBuffer(backend, kvCount * 2), values = sharedBuffer(backend, kvCount * 2);
  CommandGraph graph;
  graph.add("text_encoder_qk_norm_rope",
            {upload(backend, packed.data(), packed.size() * 2), upload(backend, norms.data(), kHead * 2),
             upload(backend, norms.data() + kHead, kHead * 2),
             upload(backend, frequencies.data(), frequencies.size() * 4), queries, keys, values},
            TextEncoderHeadParams{rows, queryHeads, kvHeads, 1e-6F}, {rows, heads, 1}, {32, 1, 1});
  graph.add("text_encoder_attention", {queries, keys, values, output},
            TextEncoderHeadParams{rows, queryHeads, kvHeads, 1.0F / std::sqrt(float(kHead))},
            {rows, queryHeads, 1}, {32, 1, 1});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  // The norm rounded to bf16 as the kernel stores it, the rotation in double
  // from those values; the values copied.
  std::vector<double> wantQ(queryCount), wantK(kvCount);
  std::vector<float> wantV(kvCount);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t head = 0; head < heads; ++head) {
      const uint16_t *x = &packed[(size_t{row} * heads + head) * kHead];
      const bool query = head < queryHeads, key = !query && head < queryHeads + kvHeads;
      const size_t at = (size_t{query ? head : (head - queryHeads) % kvHeads} * rows + row) * kHead;
      if (!query && !key) {
        for (uint32_t d = 0; d < kHead; ++d) wantV[at + d] = bf16ToFloat(x[d]);
        continue;
      }
      double squares = 0.0, normalized[kHead];
      for (uint32_t d = 0; d < kHead; ++d) squares += double(bf16ToFloat(x[d])) * bf16ToFloat(x[d]);
      const double inverse = 1.0 / std::sqrt(squares / kHead + 1e-6);
      for (uint32_t d = 0; d < kHead; ++d)
        normalized[d] = bf16ToFloat(
            floatToBf16(float(bf16ToFloat(x[d]) * inverse * bf16ToFloat(norms[(query ? 0 : kHead) + d]))));
      double *out = &(query ? wantQ : wantK)[at];
      for (uint32_t d = 0, h = kHead / 2; d < h; ++d) {
        const double angle = double(row) * frequencies[d];
        out[d] = normalized[d] * std::cos(angle) - normalized[d + h] * std::sin(angle);
        out[d + h] = normalized[d + h] * std::cos(angle) + normalized[d] * std::sin(angle);
      }
    }
  const std::vector<float> gotQ = download(queries, queryCount), gotK = download(keys, kvCount);
  require(nearBf16(gotQ, wantQ) && nearBf16(gotK, wantK), label + "queries or keys exceed one bf16 step");
  require(download(values, kvCount) == wantV, label + "values are not copied exactly");

  // Attention in double from the kernel's own bf16 queries and keys.
  std::vector<double> wantOut(queryCount);
  for (uint32_t head = 0; head < queryHeads; ++head)
    for (uint32_t row = 0; row < rows; ++row) {
      const float *q = &gotQ[(size_t{head} * rows + row) * kHead];
      const size_t kv = size_t{head / (queryHeads / kvHeads)} * rows;
      std::vector<double> scores(row + 1);
      for (uint32_t j = 0; j <= row; ++j) {
        const float *k = &gotK[(kv + j) * kHead];
        for (uint32_t d = 0; d < kHead; ++d) scores[j] += double(q[d]) * k[d];
        scores[j] /= std::sqrt(double(kHead));
      }
      const double maximum = *std::max_element(scores.begin(), scores.end());
      double total = 0.0;
      for (double &score : scores) total += (score = std::exp(score - maximum));
      for (uint32_t d = 0; d < kHead; ++d) {
        double sum = 0.0;
        for (uint32_t j = 0; j <= row; ++j) sum += scores[j] * wantV[(kv + j) * kHead + d];
        wantOut[(size_t{row} * queryHeads + head) * kHead + d] = sum / total;
      }
    }
  require(nearBf16(download(output, queryCount), wantOut), label + "attention exceeds one bf16 step");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "usage: text-encoder METALLIB");
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

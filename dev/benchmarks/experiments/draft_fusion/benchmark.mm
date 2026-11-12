#include "metal/CommandGraph.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/RoPE.h"
#include "ops/DraftAttention.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using namespace splash;
using namespace splash::metal;
constexpr uint32_t R = SPLASH_DRAFT_QUERY_ROWS, H = SPLASH_DRAFT_KV_HEADS,
                   D = SPLASH_DRAFT_HEAD_DIMENSION,
                   A = SPLASH_DRAFT_ATTENTION_WIDTH, P = SPLASH_DRAFT_QKV_WIDTH,
                   W = SPLASH_DRAFT_SLIDING_WINDOW;
struct Fixture {
  MetalBackend &backend;
  uint32_t lanes;
  CommandGraph graph;
  std::vector<ComputeDispatch> repeated;
  MetalBuffer qkv, queries, qkeys, qvalues, output, qn, kn, cosine, sine;
  std::vector<MetalBuffer> keys, values;
  std::vector<uint16_t> original, preparedQueries;
  Fixture(MetalBackend &b, uint32_t n, uint32_t length, uint32_t mode)
      : backend(b), lanes(n) {
    auto alloc = [&](uint64_t size, const char *label) {
      return b.allocateBuffer(size, BufferStorage::Shared, label);
    };
    const ops::DraftAttentionShape shape{
        2048, draft_dynamic_width(2048), P, A, 32, H, D};
    const auto plan = ops::DraftAttention::plan(shape, n);
    const auto workspace = plan.workspace();
    qkv = alloc(uint64_t{n} * R * P * 2, "input");
    queries = alloc(workspace.groupedQueriesBytes, "queries");
    qkeys = alloc(workspace.queryKeysBytes, "qkeys");
    qvalues = alloc(workspace.queryValuesBytes, "qvalues");
    output = alloc(uint64_t{n} * R * A * 2 + 256, "output");
    qn = alloc(D * 2, "qnorm");
    kn = alloc(D * 2, "knorm");
    cosine = alloc(uint64_t{n} * R * SPLASH_DRAFT_ROPE_PAIRS * 4, "cos");
    sine = alloc(cosine.sizeBytes(), "sin");
    std::mt19937 random(817);
    auto fill = [&](const MetalBuffer &buffer) {
      auto *p = static_cast<uint16_t *>(buffer.contents());
      for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i) {
        const float x = (int(random() % 1025) - 512) / 1024.f;
        uint32_t u;
        std::memcpy(&u, &x, 4);
        p[i] = u >> 16;
      }
    };
    fill(qkv);
    fill(queries);
    fill(qkeys);
    fill(qvalues);
    std::fill_n(static_cast<uint16_t *>(output.contents()),
                output.sizeBytes() / 2, uint16_t{0x7fc0});
    std::fill_n(static_cast<uint16_t *>(qn.contents()), D, uint16_t{0x3f80});
    std::fill_n(static_cast<uint16_t *>(kn.contents()), D, uint16_t{0x3f80});
    for (uint64_t i = 0; i < cosine.sizeBytes() / 4; ++i) {
      static_cast<float *>(cosine.contents())[i] =
          std::cos(float(i % 79) * .01f);
      static_cast<float *>(sine.contents())[i] = std::sin(float(i % 79) * .01f);
    }
    for (uint32_t lane = 0; lane < n; ++lane) {
      keys.push_back(alloc((uint64_t{H} * W * D + 128 * D) * 2, "ring keys"));
      values.push_back(alloc(keys.back().sizeBytes(), "ring values"));
      fill(keys.back());
      fill(values.back());
      for (auto buffer : {keys.back(), values.back()})
        std::fill_n(static_cast<uint16_t *>(buffer.contents()) +
                        uint64_t{H} * W * D,
                    128 * D, uint16_t{0x7fc0});
    }
    while (keys.size() < 4) {
      keys.push_back(keys.back());
      values.push_back(values.back());
    }
    original.assign(static_cast<uint16_t *>(qkv.contents()),
                    static_cast<uint16_t *>(qkv.contents()) +
                        qkv.sizeBytes() / 2);
    std::array<uint32_t, 4> lengths{};
    for (uint32_t i = 0; i < n; ++i)
      lengths[i] = length;
    if (mode != 2)
      ops::DraftAttention::addPrepare(
          graph, {qkv, queries, qn, kn, cosine, sine, qkeys, qvalues}, plan);
    if (mode == 0) {
      // The production reduction overwrites grouped queries. Save prepared Q
      // first.
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      auto *q = static_cast<const uint16_t *>(queries.contents());
      preparedQueries.assign(q, q + uint64_t{lanes} * R * A);
      ops::DraftAttention::addDecode(graph,
                                     {queries, keys, values, qkeys, qvalues},
                                     std::span(lengths).first(n), plan);
      ops::DraftAttention::addReorder(graph, queries, output, plan);
    } else {
      std::vector<MetalBuffer> bindings{queries};
      for (uint32_t i = 0; i < 4; ++i)
        bindings.push_back(keys[std::min(i, n - 1)]);
      for (uint32_t i = 0; i < 4; ++i)
        bindings.push_back(values[std::min(i, n - 1)]);
      bindings.insert(bindings.end(), {qkeys, qvalues, output});
      if (mode == 2)
        bindings.insert(bindings.end(), {qkv, qn, kn, cosine, sine});
      const DraftAttentionBatchParams params{
          W, n, {lengths[0], lengths[1], lengths[2], lengths[3]}};
      graph.add(mode == 2 ? "muse_attention_all" : "muse_attention_persistent",
                std::move(bindings), params, {H, n, 1});
    }
    for (uint32_t repeat = 0; repeat < 64; ++repeat)
      for (const auto &d : graph.dispatches())
        repeated.push_back(d);
  }
  double run() {
    return backend.submitCommandAsync(repeated).wait().gpuSeconds * 1e6 / 64;
  }
  void check(const Fixture &baseline) const {
    if (std::memcmp(output.contents(), baseline.output.contents(),
                    uint64_t{lanes} * R * A * 2))
      throw std::runtime_error("fused attention BF16 mismatch");
    if (std::memcmp(qkv.contents(), original.data(), qkv.sizeBytes()))
      throw std::runtime_error("fused attention wrote QKV inputs");
    for (uint64_t i = uint64_t{lanes} * R * A; i < output.sizeBytes() / 2; ++i)
      if (static_cast<uint16_t *>(output.contents())[i] != 0x7fc0)
        throw std::runtime_error("output tail overwritten");
    if (std::memcmp(queries.contents(), baseline.preparedQueries.data(),
                    uint64_t{lanes} * R * A * 2))
      throw std::runtime_error("prepared Q mismatch");
    for (auto pair : std::array<std::pair<MetalBuffer, MetalBuffer>, 2>{
             {{qkeys, baseline.qkeys}, {qvalues, baseline.qvalues}}})
      if (std::memcmp(pair.first.contents(), pair.second.contents(),
                      pair.first.sizeBytes()))
        throw std::runtime_error("prepared K/V mismatch");
    for (uint32_t lane = 0; lane < lanes; ++lane)
      for (const auto &buffer : {keys[lane], values[lane]})
        for (uint64_t i = uint64_t{H} * W * D; i < buffer.sizeBytes() / 2; ++i)
          if (static_cast<const uint16_t *>(buffer.contents())[i] != 0x7fc0)
            throw std::runtime_error("KV guard overwritten");
  }
};
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 2)
        return 2;
      MetalBackend backend(argv[1]);
      std::printf(
          "{\"scope\":\"synthetic complete draft attention;64 "
          "repetitions/command,8 warmup+16 recorded ABBA rounds; unchanged "
          "prepared math and compiled split order\",\"cases\":[");
      bool first = true;
      for (uint32_t n = 1; n <= 4; ++n)
        for (uint32_t length : {0u, 127u, 2047u, 2048u, 6000u}) {
          Fixture base(backend, n, length, 0),
              persistent(backend, n, length, 1), full(backend, n, length, 2);
          base.run();
          persistent.run();
          full.run();
          persistent.check(base);
          full.check(base);
          if (!first)
            std::printf(",");
          first = false;
          std::printf("{\"lanes\":%u,\"cache_length\":%u,\"exact_bf16_and_"
                      "input_parity\":true,\"variants\":[",
                      n, length);
          for (uint32_t mode = 1; mode <= 2; ++mode) {
            Fixture &current = mode == 1 ? persistent : full;
            if (mode == 2)
              std::printf(",");
            std::printf("{\"mode\":\"%s\",\"rounds_abba_us\":[",
                        mode == 1 ? "persistent_split_reduce"
                                  : "prepare_attention_reduce");
            for (uint32_t round = 0; round < 24; ++round) {
              const double a1 = base.run(), b1 = current.run(),
                           b2 = current.run(), a2 = base.run();
              current.check(base);
              if (round >= 8)
                std::printf("%s[%.6f,%.6f,%.6f,%.6f]", round == 8 ? "" : ",",
                            a1, b1, b2, a2);
            }
            std::printf("]}");
          }
          std::printf("]}");
          std::fflush(stdout);
        }
      std::puts("]}");
      return 0;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
}

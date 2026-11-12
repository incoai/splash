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
#include <string>
#include <vector>
using namespace splash;
using namespace splash::metal;
constexpr uint32_t R = SPLASH_DRAFT_QUERY_ROWS, H = SPLASH_DRAFT_KV_HEADS,
                   D = SPLASH_DRAFT_HEAD_DIMENSION,
                   A = SPLASH_DRAFT_ATTENTION_WIDTH, P = SPLASH_DRAFT_QKV_WIDTH,
                   W = SPLASH_DRAFT_SLIDING_WINDOW;
uint32_t inputSeed = 817;
bool nonuniformWeights = false;
struct Fixture {
  MetalBackend &backend;
  uint32_t lanes;
  CommandGraph graph;
  std::vector<ComputeDispatch> repeated;
  MetalBuffer qkv, queries, qkeys, qvalues, output, qn, kn, cosine, sine,
      finished;
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
    finished = alloc(uint64_t{n} * H * 4, "completion counters");
    std::memset(finished.contents(), 0, finished.sizeBytes());
    std::mt19937 random(inputSeed);
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
    if (nonuniformWeights) {
      fill(qn);
      fill(kn);
    }
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
    if (mode != 2) {
      if (mode == 4 || mode == 6 || mode == 7 || (mode >= 8 && mode <= 10)) {
        const uint32_t pack = (mode >= 8 && mode <= 10) ? 1u << (mode - 7) : 1u;
        const char *kernel = mode == 4    ? "muse_prepare_serial"
                             : mode == 8  ? "muse_prepare_pack2"
                             : mode == 9  ? "muse_prepare_pack4"
                             : mode == 10 ? "muse_prepare_pack8"
                                          : "muse_prepare_compact";
        graph.add(
            kernel, {qkv, queries, qn, kn, cosine, sine, qkeys, qvalues},
            {mode == 4 ? H : R * (SPLASH_DRAFT_QUERY_HEADS + H) / pack, n, 1},
            {mode == 4 ? 256u : 128u * pack, 1, 1});
      } else
        ops::DraftAttention::addPrepare(
            graph, {qkv, queries, qn, kn, cosine, sine, qkeys, qvalues}, plan);
    }
    const DraftAttentionBatchParams params{
        W, n, {lengths[0], lengths[1], lengths[2], lengths[3]}};
    if (mode == 0) {
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      auto *q = static_cast<const uint16_t *>(queries.contents());
      preparedQueries.assign(q, q + uint64_t{lanes} * R * A);
      ops::DraftAttention::addDecode(graph,
                                     {queries, keys, values, qkeys, qvalues},
                                     std::span(lengths).first(n), plan);
      ops::DraftAttention::addReorder(graph, queries, output, plan);
    } else if (mode == 3 || mode == 4 || mode == 6 || mode == 11 ||
               (mode >= 8 && mode <= 10)) {
      std::vector<MetalBuffer> bindings{queries};
      bindings.insert(bindings.end(), keys.begin(), keys.end());
      bindings.insert(bindings.end(), values.begin(), values.end());
      bindings.insert(bindings.end(), {qkeys, qvalues});
      if (mode == 11)
        bindings.push_back(output);
      graph.add(mode == 11 ? "muse_split_serial" : "draft_attention_bf16_split",
                std::move(bindings), params,
                {H, n, mode == 11 ? 1 : SPLASH_DRAFT_ATTENTION_SPLITS});
      graph.add("muse_reduce_parallel", {queries, output}, params, {H, n, 1});
    } else {
      std::vector<MetalBuffer> bindings{queries};
      for (uint32_t i = 0; i < 4; ++i)
        bindings.push_back(keys[i]);
      for (uint32_t i = 0; i < 4; ++i)
        bindings.push_back(values[i]);
      bindings.insert(bindings.end(), {qkeys, qvalues, output});
      if (mode == 2)
        bindings.insert(bindings.end(), {qkv, qn, kn, cosine, sine});
      if (mode == 5 || mode == 7)
        bindings.push_back(finished);
      graph.add(
          mode == 2                ? "muse_attention_all"
          : mode == 5 || mode == 7 ? "muse_attention_last_arriver"
                                   : "muse_attention_persistent",
          std::move(bindings), params,
          {H, n, mode == 5 || mode == 7 ? SPLASH_DRAFT_ATTENTION_SPLITS : 1});
    }
    for (uint32_t repeat = 0; repeat < 64; ++repeat)
      for (const auto &d : graph.dispatches())
        repeated.push_back(d);
  }
  double run() {
    return backend.submitCommandAsync(repeated).wait().gpuSeconds * 1e6 / 64;
  }
  void check(const Fixture &baseline) const {
    for (uint32_t i = 0; i < lanes * H; ++i)
      if (static_cast<const uint32_t *>(finished.contents())[i])
        throw std::runtime_error("completion counter not reset");
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
      if (argc < 2)
        return 2;
      bool validate = false, screen = false, holdout = false, confirm = false;
      for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--validate-only")
          validate = true;
        else if (std::string(argv[i]) == "--screen")
          screen = true;
        else if (std::string(argv[i]) == "--holdout")
          holdout = true;
        else if (std::string(argv[i]) == "--confirm")
          confirm = true;
        else if (std::string(argv[i]) == "--nonuniform-weights")
          nonuniformWeights = true;
        else if (std::string(argv[i]) == "--seed" && i + 1 < argc)
          inputSeed = static_cast<uint32_t>(std::stoul(argv[++i]));
        else
          return 2;
      }
      const char *names[] = {"production",
                             "persistent_split_reduce",
                             "prepare_attention_reduce",
                             "parallel_row_major_control",
                             "serial_prepare_parallel_attention",
                             "parallel_last_arriver",
                             "compact_prepare_parallel_attention",
                             "compact_prepare_last_arriver",
                             "pack2_prepare_parallel_attention",
                             "pack4_prepare_parallel_attention",
                             "pack8_prepare_parallel_attention",
                             "serial_split_separate_reduce"};
      MetalBackend backend(argv[1]);
      std::printf(
          "{\"scope\":\"complete attention;64 repetitions/command; baseline "
          "field identifies treatment control;8 warmup+16 recorded ABBA "
          "blocks\",\"validation_only\":%s,\"screen\":%s,\"seed\":%u,"
          "\"holdout\":%s,\"confirm_pack4\":%s,\"nonuniform_norm_weights\":%s,"
          "\"cases\":[",
          validate ? "true" : "false", screen ? "true" : "false", inputSeed,
          holdout ? "true" : "false", confirm ? "true" : "false",
          nonuniformWeights ? "true" : "false");
      bool first = true;
      const std::vector<uint32_t> history =
          holdout ? std::vector<uint32_t>{1, 128, 129, 511, 2055, 8192}
                  : std::vector<uint32_t>{0, 127, 2047, 2048, 6000};
      for (uint32_t n = 1; n <= 4; ++n)
        for (uint32_t length : history) {
          if (screen && ((n != 1 && n != 4) ||
                         (length != 0 && length != 127 && length != 2048)))
            continue;
          Fixture base(backend, n, length, 0), control(backend, n, length, 3);
          base.run();
          control.run();
          control.check(base);
          if (!first)
            std::printf(",");
          first = false;
          std::printf("{\"lanes\":%u,\"cache_length\":%u,\"variants\":[", n,
                      length);
          bool firstVariant = true;
          for (uint32_t mode : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u}) {
            if (confirm && mode != 9)
              continue;
            Fixture candidate(backend, n, length, mode);
            candidate.run();
            candidate.check(base);
            Fixture &reference = mode <= 3 ? base : control;
            if (!firstVariant)
              std::printf(",");
            firstVariant = false;
            std::printf("{\"mode\":\"%s\",\"baseline\":\"%s\",\"exact_bf16_and_"
                        "input_parity\":true,\"rounds_abba_us\":[",
                        names[mode], mode <= 3 ? names[0] : names[3]);
            if (!validate)
              for (uint32_t round = 0; round < 24; ++round) {
                const double a = reference.run(), b = candidate.run(),
                             c = candidate.run(), d = reference.run();
                candidate.check(base);
                if (round >= 8)
                  std::printf("%s[%.6f,%.6f,%.6f,%.6f]", round == 8 ? "" : ",",
                              a, b, c, d);
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

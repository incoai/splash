#include "../../../tests/engine/GgufFormatReference.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/Gguf.h"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#import <Foundation/Foundation.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using namespace splash;
using namespace splash::metal;
using namespace splash::ops;
using namespace gguf_reference;
struct FfnParams {
  uint32_t hidden, intermediate, splits, workers, phase, dynamic, spread, pad;
};
struct Weights {
  MetalBuffer norm, gw, gm, uw, um, dw, dm;
};
struct Values {
  MetalBuffer x, norm, mid, part, out, counter, state;
};
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 11)
        return 2;
      const uint32_t H = std::stoul(argv[2]), I = std::stoul(argv[3]),
                     R = std::stoul(argv[4]), workers = std::stoul(argv[5]),
                     rounds = std::stoul(argv[6]), ds = std::stoul(argv[7]),
                     seed = std::stoul(argv[9]);
      const bool validate = std::stoul(argv[8]);
      if (!H || H % 64 || !I || I % 64 || (R != 8 && R != 16 && R != 32) ||
          !workers || workers > 320 || !ds || ds > 8 || (I / 32) % ds)
        throw std::invalid_argument("unsupported dimensions");
      const bool registerInputs = std::stoul(argv[10]);
      const std::string prefix =
          registerInputs ? "muse_ffn_register_" : "muse_ffn_";
      MetalBackend backend(argv[1]);
      std::mt19937 rng(seed);
      const Fmt f = fmtNamed("af4g64");
      auto g = repack(f, makeNative(f, I, H, rng), I, H, nullptr),
           u = repack(f, makeNative(f, I, H, rng), I, H, nullptr),
           d = repack(f, makeNative(f, H, I, rng), H, I, nullptr);
      const uint64_t weightBytes = g.w0.size() + g.meta.size() + u.w0.size() +
                                   u.meta.size() + d.w0.size() + d.meta.size() +
                                   H * 2;
      const uint32_t copies =
          validate ? 1
                   : std::max<uint64_t>(3, ((384ull << 20) + weightBytes - 1) /
                                               weightBytes);
      auto alloc = [&](uint64_t bytes) {
        auto b = backend.allocateBuffer(std::max<uint64_t>(bytes, 16),
                                        BufferStorage::Shared, "pending FFN");
        std::memset(b.contents(), 0, b.sizeBytes());
        return b;
      };
      auto up = [&](const std::vector<uint8_t> &bytes) {
        auto b = alloc(bytes.size());
        std::memcpy(b.contents(), bytes.data(), bytes.size());
        return b;
      };
      std::uniform_real_distribution<float> unit(-1, 1);
      std::vector<__bf16> weight(H);
      for (auto &v : weight)
        v = __bf16(.5f + std::abs(unit(rng)));
      std::vector<__bf16> input(32ull * H, 0);
      for (uint64_t j = 0; j < uint64_t{R} * H; ++j)
        input[j] = __bf16(unit(rng));
      std::vector<Weights> weights;
      for (uint32_t c = 0; c < copies; ++c) {
        auto n = alloc(H * 2);
        std::memcpy(n.contents(), weight.data(), H * 2);
        weights.push_back({n, up(g.w0), up(g.meta), up(u.w0), up(u.meta),
                           up(d.w0), up(d.meta)});
      }
      const Linear linear(backend.capabilities(), Int8Activations::On);
      auto segment = [&](uint32_t n, uint32_t k, MetalBuffer w, MetalBuffer m) {
        return QuantizedSegment::planes(f, n, k, w, {}, m);
      };
      Projection gp(I, H, {{segment(I, H, weights[0].gw, weights[0].gm)}}),
          uproj(I, H, {{segment(I, H, weights[0].uw, weights[0].um)}}),
          dp(H, I, {{segment(H, I, weights[0].dw, weights[0].dm)}});
      auto gplan = linear.decodePlan(uproj, R / 8, LinearEpilogue::GateUp, &gp),
           dplan = linear.decodePlan(dp, R / 8, LinearEpilogue::Residual);
      const auto gc = gplan.configuration(), dc = dplan.configuration();
      const bool native = gc.tile == LinearTile::GgufStaged &&
                          gc.oneGateUpPass && gc.splits == 1 &&
                          dc.tile == LinearTile::GgufStaged && dc.splits == ds;
      const uint32_t gr = gc.tile == LinearTile::GgufStaged
                              ? gplan.storageRows()
                              : R,
                     dr = dc.tile == LinearTile::GgufStaged
                              ? dplan.storageRows()
                              : R;
      std::array<std::vector<Values>, 4> values;
      std::array<CommandGraph, 4> graphs;
      for (uint32_t mode = 0; mode < 4; ++mode)
        for (uint32_t c = 0; c < copies; ++c) {
          const auto &w = weights[c];
          auto guardAlloc = [&](uint64_t n) {
            auto b = alloc(n + 256);
            std::memset(static_cast<uint8_t *>(b.contents()) + n, 0xA5, 256);
            return b;
          };
          Values b{
              guardAlloc(32ull * H * 2), guardAlloc(32ull * H * 2),
              guardAlloc(32ull * I * 2), guardAlloc(8ull * 32 * H * 4),
              guardAlloc(32ull * H * 2), guardAlloc((H / 32 + I / 32 + 32) * 4),
              guardAlloc(16 * 4)};
          std::memcpy(b.x.contents(), input.data(), input.size() * 2);
          FfnParams p{H, I, ds, workers, 0, mode == 3 ? 1u : 0u, 1, 0};
          if (mode == 0) {
            static_cast<void>(Normalization::addRms(
                graphs[mode], b.x, {w.norm, false}, b.norm, H, R));
            graphs[mode].add(
                "gguf_decode_af4g64_m" + std::to_string(gr) + "_gate_up",
                {b.norm, w.gw, w.gw, w.gm, w.uw, w.uw, w.um, b.mid, b.part,
                 b.counter},
                GgufDecodeParams{H, 1, I, 0, 1}, {I / 32, 1, 1}, {32, 1, 1});
            graphs[mode].add(
                "gguf_decode_af4g64_m" + std::to_string(dr) + "_r",
                {b.mid, w.dw, w.dw, w.dm, b.out, b.part, b.counter, b.x},
                GgufDecodeParams{I, ds, H, 0, 1}, {H / 64, ds, 1}, {64, 1, 1});
          } else {
            auto bindings = std::vector<MetalBuffer>{
                b.x,  w.norm, w.gw,  w.gm,   w.uw,  w.um,   w.dw,
                w.dm, b.norm, b.mid, b.part, b.out, b.state};
            if (mode == 1)
              for (p.phase = 0; p.phase < 4; ++p.phase) {
                const uint32_t jobs = p.phase == 0   ? R
                                      : p.phase == 1 ? I / 32
                                      : p.phase == 2 ? H / 32 * ds
                                                     : H / 32;
                graphs[mode].add(prefix + "static_m" + std::to_string(R),
                                 bindings, p, {jobs, 1, 1}, {32, 1, 1});
              }
            else
              graphs[mode].add(prefix + "persistent_m" + std::to_string(R),
                               bindings, p, {workers, 1, 1}, {32, 1, 1});
          }
          values[mode].push_back(b);
        }
      auto run = [&](uint32_t mode) {
        for (auto &b : values[mode])
          std::memset(b.state.contents(), 0, 64);
        const auto t =
            backend.submitCommandAsync(graphs[mode].dispatches()).wait();
        for (const auto &b : values[mode]) {
          const auto *s = static_cast<const uint32_t *>(b.state.contents());
          if (mode >= 2 && (s[0] || s[2] || s[3] != 1 || s[4]))
            throw std::runtime_error("persistent forward progress failed");
        }
        return t.gpuSeconds * 1e6 / copies;
      };
      auto check = [&](uint32_t mode) {
        for (uint32_t c = 0; c < copies; ++c) {
          const auto &a = values[0][c];
          const auto &b = values[mode][c];
          for (auto pair : {std::pair{a.norm, b.norm}, std::pair{a.out, b.out},
                            std::pair{a.x, b.x}})
            if (std::memcmp(pair.first.contents(), pair.second.contents(),
                            uint64_t{R} * H * 2))
              throw std::runtime_error(
                  "norm/output/input parity failure mode=" +
                  std::to_string(mode));
          if (std::memcmp(a.mid.contents(), b.mid.contents(),
                          uint64_t{R} * I * 2))
            throw std::runtime_error("intermediate parity failure mode=" +
                                     std::to_string(mode));
          for (uint32_t s = 0; ds > 1 && s < ds; ++s)
            if (std::memcmp(static_cast<const float *>(a.part.contents()) +
                                uint64_t{s} * dr * H,
                            static_cast<const float *>(b.part.contents()) +
                                uint64_t{s} * R * H,
                            uint64_t{R} * H * 4))
              throw std::runtime_error("partial parity failure mode=" +
                                       std::to_string(mode));
          for (uint32_t j = 0; j < (H / 32 + I / 32 + 32); ++j)
            if (static_cast<const uint32_t *>(a.counter.contents())[j])
              throw std::runtime_error("native counter not reset");
          for (const auto *v :
               {&b.x, &b.norm, &b.mid, &b.part, &b.out, &b.counter, &b.state})
            for (uint64_t j = v->sizeBytes() - 256; j < v->sizeBytes(); ++j)
              if (static_cast<const uint8_t *>(v->contents())[j] != 0xA5)
                throw std::runtime_error("buffer guard overwritten");
        }
      };
      run(0);
      for (uint32_t mode = 1; mode < 4; ++mode) {
        run(mode);
        check(mode);
      }
      const bool shader = std::getenv("MTL_SHADER_VALIDATION");
      std::printf("{\"register_inputs\":%s,\"result\":",
                  registerInputs ? "true" : "false");
      if (validate)
        for (uint32_t repeat = 0; repeat < 4; ++repeat)
          for (uint32_t mode = 0; mode < 4; ++mode) {
            run(mode);
            if (mode)
              check(mode);
          }
      std::printf("{\"hidden\":%u,\"intermediate\":%u,\"rows\":%u,\"workers\":%"
                  "u,\"splits\":%u,\"copies\":%u,\"seed\":%u,\"native_"
                  "selection_matched\":%s,\"native_gate_rows\":%u,\"native_"
                  "down_rows\":%u,\"shader\":%s,\"comparisons\":[",
                  H, I, R, workers, ds, copies, seed, native ? "true" : "false",
                  gr, dr, shader ? "true" : "false");
      bool first = true;
      for (uint32_t mode = 1; mode < 4; ++mode)
        for (uint32_t control : std::vector<uint32_t>(
                 mode == 1 ? std::initializer_list<uint32_t>{0}
                           : std::initializer_list<uint32_t>{0, 1})) {
          if (!first)
            printf(",");
          first = false;
          printf(
              "{\"control\":%u,\"candidate\":%u,\"exact\":true,\"abba_us\":[",
              control, mode);
          if (!validate)
            for (uint32_t r = 0; r < rounds + 4; ++r) {
              double a = run(control), b = run(mode), c = run(mode),
                     d = run(control);
              if (r >= 4)
                printf("%s[%.6f,%.6f,%.6f,%.6f]", r == 4 ? "" : ",", a, b, c,
                       d);
            }
          run(mode);
          check(mode);
          printf("]}");
          fflush(stdout);
        }
      printf("]}}\n");
      return 0;
    } catch (const std::exception &e) {
      fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
}

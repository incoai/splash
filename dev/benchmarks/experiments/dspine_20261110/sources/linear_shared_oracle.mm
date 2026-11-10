#include "GgufFormatReference.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/Gguf.h"
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace splash;
using namespace gguf_reference;

struct Guarded {
  metal::MetalBuffer base, view;
  uint64_t bytes;
  Guarded(metal::MetalBackend &backend, uint64_t size, uint8_t fill=0):bytes(size) {
    base=backend.allocateBuffer(size+256,metal::BufferStorage::Shared,"gate/up oracle guarded");
    std::memset(base.contents(),fill,size);std::memset(static_cast<uint8_t *>(base.contents())+size,0xa5,256);
    view=backend.view(base,0,size);
  }
  void check()const {
    const auto *p=static_cast<const uint8_t *>(base.contents())+bytes;
    if(!std::all_of(p,p+256,[](uint8_t b){return b==0xa5;}))throw std::runtime_error("guard overwritten");
  }
};
struct Weight {
  metal::MetalBuffer code, other, meta;
  std::vector<float> reference;
};
Weight weight(metal::MetalBackend &backend,uint32_t n,uint32_t k,std::mt19937 &rng,bool reference) {
  const auto native=makeNative(Fmt(GGUF_FMT_AF4G64),n,k,rng);
  Weight w;
  const auto packed=repack(Fmt(GGUF_FMT_AF4G64),native,n,k,reference?&w.reference:nullptr);
  const auto upload=[&](const auto &v){auto b=backend.allocateBuffer(v.size(),metal::BufferStorage::Shared,"oracle planes");std::memcpy(b.contents(),v.data(),v.size());return b;};
  w.code=upload(packed.w0);w.other=upload(packed.w1);w.meta=upload(packed.meta);return w;
}

float rounded(double value) { return test::bf16ToFloat(test::floatToBf16(float(value))); }

void runLinear(metal::MetalBackend &backend, uint32_t rows, uint32_t live, uint32_t n, uint32_t k,
               bool benchmark, bool head = false) {
  std::mt19937 rng(9127 + rows + k);
  auto weights = weight(backend, n, k, rng, !benchmark);
  const uint32_t offset = benchmark ? 0 : 64, stride = n + 2 * offset;
  Guarded input(backend, uint64_t(rows) * k * 2), output(backend, uint64_t(rows) * stride * 4),
      aux(backend, uint64_t(rows) * stride * 2), partials(backend, uint64_t(16) * rows * stride * 4, 0xff),
      counters(backend, stride / 64 * 4);
  auto *x = static_cast<uint16_t *>(input.view.contents());
  auto *auxiliary = static_cast<uint16_t *>(aux.view.contents());
  std::vector<float> xf(uint64_t(rows) * k);
  std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < k; ++col) {
      const auto at = uint64_t(row) * k + col;
      x[at] = row < live ? test::floatToBf16(dist(rng)) : 0x7fc0;
      xf[at] = test::bf16ToFloat(x[at]);
    }
    for (uint32_t col = 0; col < stride; ++col)
      auxiliary[uint64_t(row) * stride + col] = test::floatToBf16(dist(rng));
  }
  uint64_t comparisons = 0, cpuChecks = 0;
  const auto splits = benchmark ? std::vector<uint32_t>{1, 2} : k == 512 ?
      std::vector<uint32_t>{1, 2, 4, 8, 16} : std::vector<uint32_t>{1, 2, 4};
  const auto epilogues = head ? std::vector<std::string>{"a_f32"} :
      benchmark ? std::vector<std::string>{"a", "r"} : std::vector<std::string>{"a", "r", "g", "a_f32"};
  for (const auto &ep : epilogues) for (bool spread : {false, true}) for (uint32_t split : splits) {
    std::vector<uint32_t> parts = rows == 64 ? std::vector<uint32_t>{0, 16, 32} : std::vector<uint32_t>{0, 16};
    std::vector<uint8_t> original;
    const uint32_t bytes = ep == "a_f32" ? 4 : 2;
    for (uint32_t part : parts) {
      std::memset(output.view.contents(), 0xa6, uint64_t(rows) * stride * 4);
      const auto name = "gguf_decode_af4g64_m" + std::to_string(rows) + "_" + ep +
                        (part ? "_rows" + std::to_string(part) : "");
      metal::CommandGraph graph;
      const uint32_t repeats = benchmark ? 32 : 3;
      for (uint32_t repeat = 0; repeat < repeats; ++repeat)
        graph.add(name, {input.view, weights.code, weights.other, weights.meta, output.view,
                        partials.view, counters.view, aux.view},
                  GgufDecodeParams{k, split, stride, offset, spread}, {n / 64, split, 1},
                  {part ? rows / part * 32 : 64, 1, 1});
      const auto start = std::chrono::steady_clock::now();
      backend.preparePipelines(graph.dispatches());
      const double prepare = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      const double seconds = backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
      input.check(); output.check(); aux.check(); partials.check(); counters.check();
      const auto *count = static_cast<const uint32_t *>(counters.view.contents());
      if (!std::all_of(count, count + stride / 64, [](uint32_t v) { return v == 0; }))
        throw std::runtime_error("shared rows did not release split counter");
      const auto *raw = static_cast<const uint8_t *>(output.view.contents());
      if (bytes == 2 && !std::all_of(raw + uint64_t(rows) * stride * 2,
          raw + uint64_t(rows) * stride * 4, [](uint8_t b) { return b == 0xa6; }))
        throw std::runtime_error("shared rows overwrote the BF16 output tail");
      if (!part) original.assign(raw, raw + uint64_t(rows) * stride * bytes);
      const auto value = [&](uint32_t row, uint32_t col) {
        const auto at = uint64_t(row) * stride + offset + col;
        return bytes == 4 ? double(static_cast<const float *>(output.view.contents())[at]) :
                           double(test::bf16ToFloat(static_cast<const uint16_t *>(output.view.contents())[at]));
      };
      for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < stride; ++col) {
          const auto at = (uint64_t(row) * stride + col) * bytes;
          if (col < offset || col >= offset + n) {
            if (!std::all_of(raw + at, raw + at + bytes, [](uint8_t b) { return b == 0xa6; }))
              throw std::runtime_error("shared rows overwrote output padding");
          } else if (row < live && part) {
            ++comparisons;
            if (std::memcmp(raw + at, original.data() + at, bytes))
              throw std::runtime_error("shared rows differ bitwise from original projection");
          } else if (row >= live && !std::isnan(value(row, col - offset))) {
            throw std::runtime_error("shared rows mixed NaN padding into valid rows");
          }
        }
      }
      if (!benchmark) for (uint32_t row : {0u, live / 2, live - 1}) for (uint32_t col : {0u, n / 2, n - 1}) {
        const auto d = dot(xf.data() + uint64_t(row) * k, weights.reference.data() + uint64_t(col) * k, k);
        const double bound = projectionBound(d, true);
        double lo = d.value - bound, hi = d.value + bound;
        const double a = test::bf16ToFloat(auxiliary[uint64_t(row) * stride + offset + col]);
        if (ep == "r") {
          const double add = (std::fabs(d.value) + bound + std::fabs(a)) * 0x1p-24;
          lo += a - add; hi += a + add;
        } else if (ep == "g") {
          const double s = a / (1 + std::exp(-a)), t = std::fabs(a) * 1.4426950408889634;
          const double rel = (2 * (3 + std::floor(2 * t)) + 2 * std::log(2.0) * t + 6) * 0x1p-24;
          const double e = std::fabs(s) * rel;
          const double low = rounded(lo), high = rounded(hi);
          lo = INFINITY; hi = -INFINITY;
          for (double u : {low, high}) for (double g : {s - e, s + e}) {
            lo = std::min(lo, u * g); hi = std::max(hi, u * g);
          }
        }
        if (bytes == 2) { lo = rounded(lo); hi = rounded(hi); }
        const double actual = value(row, col); ++cpuChecks;
        if (!std::isfinite(actual) || actual < lo || actual > hi)
          throw std::runtime_error("shared rows outside independent FP64 projection interval");
      }
      std::cout << "{\"rows\":" << rows << ",\"live_rows\":" << live << ",\"outputs\":" << n
                << ",\"inputs\":" << k << ",\"epilogue\":\"" << ep << "\",\"splits\":" << split
                << ",\"spread\":" << (spread ? "true" : "false") << ",\"part_rows\":" << part
                << ",\"repeats\":" << repeats << ",\"pipeline_prepare_wall_ms\":" << prepare
                << ",\"gpu_ms_per_projection\":" << seconds * 1000 / repeats << "}\n";
    }
  }
  std::cout << "{\"case_passed\":true,\"rows\":" << rows << ",\"live_rows\":" << live
            << ",\"bitwise_comparisons\":" << comparisons << ",\"independent_fp64_checks\":" << cpuChecks << "}\n";
}

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 3) throw std::invalid_argument("usage: linear-shared-oracle METALLIB [--benchmark]");
    const bool benchmark = argc == 3 && std::string(argv[2]) == "--benchmark";
    if (argc == 3 && !benchmark) throw std::invalid_argument("unknown argument");
    metal::MetalBackend backend(argv[1]);
    if (benchmark) {
      for (uint32_t rows : {32u, 64u}) {
        runLinear(backend, rows, rows, 17408, 5120, true);
        runLinear(backend, rows, rows, 5120, 17408, true);
        runLinear(backend, rows, rows, 5120, 5120, true);
        runLinear(backend, rows, rows, 248320, 5120, true, true);
      }
    } else {
      for (uint32_t rows : {32u, 64u}) for (uint32_t k : {512u, 1152u})
        for (uint32_t live : {rows, rows * 3 / 4}) runLinear(backend, rows, live, 128, k, false);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

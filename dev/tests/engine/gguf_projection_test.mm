// Production K-quant correctness oracle for Apple9/Apple10, with optional timing modes.
// Decode rows 8/16/24/32; full mode covers every gate/up format pair.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <map>
#include <vector>
#include "GgufFormatReference.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/Linear.h"
static id<MTLDevice> dev; static id<MTLCommandQueue> queue; static std::mt19937 rng(42);
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x8000) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static std::map<std::string, id<MTLComputePipelineState>> psoCache;
static id<MTLComputePipelineState> pso(id<MTLLibrary> lib, const std::string &name) {
  auto it = psoCache.find(name); if (it != psoCache.end()) return it->second;
  NSError *e = nil; id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!fn) { std::cerr << "no function " << name << "\n"; psoCache[name] = nil; return nil; }
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e];
  if (!p) { std::cerr << "pso failed " << name << ": " << e.localizedDescription.UTF8String << "\n"; psoCache[name] = nil; return nil; }
  psoCache[name] = p; return p;
}
struct Dispatch { id<MTLComputePipelineState> p; std::vector<id<MTLBuffer>> bufs; std::vector<uint8_t> params; int paramIndex; MTLSize grid, tg; };
static double runOnce(const std::vector<Dispatch> &ds, int n) {
  id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  for (int it = 0; it < n; ++it) for (auto &d : ds) {
    [enc setComputePipelineState:d.p];
    for (size_t i = 0; i < d.bufs.size(); ++i) if (d.bufs[i]) [enc setBuffer:d.bufs[i] offset:0 atIndex:i];
    [enc setBytes:d.params.data() length:d.params.size() atIndex:d.paramIndex];
    [enc dispatchThreadgroups:d.grid threadsPerThreadgroup:d.tg];
  }
  [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  if (cb.error) { std::cerr << "GPU error: " << cb.error.localizedDescription.UTF8String << "\n"; exit(1); }
  return (cb.GPUEndTime - cb.GPUStartTime) / n;
}
template <class T> static std::vector<uint8_t> bytes(const T &v) { return std::vector<uint8_t>((const uint8_t *)&v, (const uint8_t *)&v + sizeof v); }
static id<MTLBuffer> mkbuf(uint64_t n) { return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; }
// ================= formats: native GGUF blocks, reference values and planes from GgufFormatReference.hpp
using namespace gguf_reference;
// Optional external oracle: compile unmodified llama.cpp ggml-base and provide its
// dylib via SPLASH_GGML_ORACLE. The normal test remains self-contained/offline.
static void *ggmlOracle = nullptr;
static int oracleFailures = 0;
static void verifyNativeReference(Fmt f, const std::vector<uint8_t> &native, const std::vector<float> &values) {
  if (!ggmlOracle) return;
  std::vector<float> official; std::string error;
  if (!ggmlDequantize(ggmlOracle, f, native, official, error)) { printf("GGML oracle: %s FAIL\n", error.c_str()); ++oracleFailures; return; }
  if (memcmp(official.data(), values.data(), values.size() * sizeof(float))) { printf("CPU reference differs from upstream GGML: %s FAIL\n", fmtName(f)); ++oracleFailures; }
}
static uint64_t streamBytes(Fmt f, uint32_t N, uint32_t K) { const QuantFormat &i = kQuantFormats[f]; return uint64_t(N) * (K / 32) * (i.plane0_bytes + i.plane1_bytes) + uint64_t(N) * (K / 32 / i.meta_groups) * i.meta_bytes; }
// Scales in realistic per-format ranges for the GEMM checks.
static std::vector<uint8_t> makeNative(Fmt f, uint32_t N, uint32_t K) {
  std::uniform_real_distribution<float> dk(0.0005f, 0.004f), dx(0.00002f, 0.00015f), d6(0.00002f, 0.0001f), d3s(0.0001f, 0.0005f);
  std::uniform_real_distribution<float> &d = f == IQ4XS ? dx : f == Q6K ? d6 : f == IQ3S ? d3s : dk;
  return makeNative(f, N, K, rng, [&] { return f2h(d(rng)); });
}
static id<MTLBuffer> upload(const std::vector<uint8_t> &v) { id<MTLBuffer> b = mkbuf(v.size()); memcpy(b.contents, v.data(), v.size()); return b; }
// harness_prod: validate the production kernels (gguf_linear.metal ABI) from a compiled .metallib against the C++ reference.
//   harness_prod <splash.metallib>
struct Seg { Fmt fmt; uint32_t N; uint32_t colOffset; id<MTLBuffer> w0, w1, meta; std::vector<float> Wf, Ws; };
static Seg makeSegK(Fmt f, uint32_t N, uint32_t K, uint32_t colOffset);
static Seg makeSeg(Fmt f, uint32_t N, uint32_t K, uint32_t colOffset) {
  Seg s; s.fmt = f; s.N = N; s.colOffset = colOffset;
  std::vector<uint8_t> native = makeNative(f, N, K); Packed pk = repack(f, native, N, K, &s.Wf, &s.Ws);
  verifyNativeReference(f, native, s.Wf);
  s.w0 = upload(pk.w0); s.w1 = upload(pk.w1); s.meta = upload(pk.meta); if (!kQuantFormats[f].plane1_bytes) s.w1 = s.meta; return s;
}
static Seg makeSegK(Fmt f, uint32_t N, uint32_t K, uint32_t colOffset) {   // no float reference (large shapes)
  Seg s; s.fmt = f; s.N = N; s.colOffset = colOffset;
  std::vector<uint8_t> native = makeNative(f, N, K); Packed pk = repack(f, native, N, K, nullptr);
  s.w0 = upload(pk.w0); s.w1 = upload(pk.w1); s.meta = upload(pk.meta); if (!kQuantFormats[f].plane1_bytes) s.w1 = s.meta; return s;
}
// Round to nearest even, as the GPU converts to bf16.
static uint16_t bf16Rne(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
// Apple9 register kernels (decode/linear_gguf_sgmatrix.metal): decode_linear_gguf_prepare writes the Table16 table
// of L eight-row lanes, then the segment kernel writes columns [N, 2N) of a 2N-wide destination. Every format,
// L = 1..4, splits 1..8 and every epilogue, at input scale 1 and with sparse inputs of 1e5. The kernels are exact up
// to fp32 accumulation: an output must be the bf16 rounding of a value within 2^-16 sum|x| max|w| of the fp64 result
// over GGML's fp32 weights, and at most 1% of the outputs may differ from the rounded fp64 result (staging the
// weights in half changes 7-11%). Lane r of an L-lane dispatch equals the one-lane dispatch on its rows bitwise,
// under partials poisoned with NaN or a large finite value; the other destination rows and columns stay untouched
// and every counter returns to zero.
static int registerKernels(id<MTLLibrary> lib) {
  constexpr uint32_t N = 512, K = 2048, kMaxLanes = 4, kRows = 8 * kMaxLanes, stride = 2 * N, kSentinel = 0xFFFF;
  int failures = 0;
  id<MTLComputePipelineState> prepare = pso(lib, "decode_linear_gguf_prepare");
  if (!prepare) return 1;
  id<MTLBuffer> table = mkbuf(uint64_t(kRows) * K * 2), sums = mkbuf(uint64_t(kMaxLanes) * (K * 3 / 4) * 4);
  id<MTLBuffer> partials = mkbuf(uint64_t(8) * kRows * stride * 4), counters = mkbuf(stride / 64 * 4);
  memset(counters.contents, 0, counters.length);
  for (const float scale : {1.f, 1e5f})
    for (int fi = 0; fi < FMT_COUNT; ++fi) {
      const Seg s = makeSeg(Fmt(fi), N, K, 0);
      // Dense activations in [-1, 1], or one +-scale entry per 256 inputs.
      std::uniform_real_distribution<float> random(-1.f, 1.f);
      std::vector<uint16_t> x(size_t(kRows) * K, 0), aux(size_t(kRows) * stride);
      for (uint32_t r = 0; r < kRows; ++r)
        for (uint32_t k = 0; k < K; ++k)
          if (scale == 1.f) x[size_t(r) * K + k] = f2bf(random(rng));
          else if (k % 256 == (r * 31 + k / 256 * 97) % 256) x[size_t(r) * K + k] = f2bf((r + k / 256) % 2 ? -scale : scale);
      for (uint16_t &v : aux) v = f2bf(random(rng));
      std::vector<double> dot(size_t(kRows) * N), bound(size_t(kRows) * N);
      for (uint32_t n = 0; n < N; ++n) {
        const float *w = s.Wf.data() + size_t(n) * K;
        double wmax = 0;
        for (uint32_t k = 0; k < K; ++k) wmax = std::max(wmax, double(std::fabs(w[k])));
        for (uint32_t r = 0; r < kRows; ++r) {
          double acc = 0, mag = 0;
          for (uint32_t k = 0; k < K; ++k) { const double v = bf2f(x[size_t(r) * K + k]); acc += v * w[k]; mag += std::fabs(v); }
          dot[size_t(r) * N + n] = acc;
          bound[size_t(r) * N + n] = std::ldexp(mag * wmax, -16);
        }
      }
      const auto rowsOf = [&](const std::vector<uint16_t> &v, uint32_t width, uint32_t first, uint32_t rows) {
        id<MTLBuffer> b = mkbuf(uint64_t(rows) * width * 2); memcpy(b.contents, v.data() + size_t(first) * width, b.length); return b; };
      id<MTLBuffer> X = rowsOf(x, K, 0, kRows), A = rowsOf(aux, stride, 0, kRows), Y = mkbuf(uint64_t(kRows) * stride * 2);
      id<MTLBuffer> Yfused = mkbuf(Y.length);
      id<MTLBuffer> laneX[kMaxLanes], laneA[kMaxLanes], laneY[kMaxLanes];
      for (uint32_t r = 0; r < kMaxLanes; ++r) { laneX[r] = rowsOf(x, K, 8 * r, 8); laneA[r] = rowsOf(aux, stride, 8 * r, 8); laneY[r] = mkbuf(uint64_t(8) * stride * 2); }
      int formatFailures = 0; double worstFlips = 0;
      for (const char ep : {'a', 'r', 'g'})
        for (const uint32_t splits : {1u, 2u, 4u, 8u}) {
          char where[96];
          const auto run = [&](uint32_t lanes, id<MTLBuffer> in, id<MTLBuffer> gate, id<MTLBuffer> out, uint32_t poison) {
            char name[64]; snprintf(name, sizeof name, "decode_linear_gguf_sg_%s_l%u_%c", fmtName(fi), lanes, ep);
            id<MTLComputePipelineState> ps = pso(lib, name);
            if (!ps) { ++formatFailures; return; }
            std::fill_n((uint32_t *)partials.contents, partials.length / 4, poison);
            std::fill_n((uint16_t *)out.contents, out.length / 2, uint16_t(kSentinel));
            const Dispatch table16{prepare, {in, table, sums}, bytes(K), 3, MTLSizeMake(K / 32, lanes, 1), MTLSizeMake(128, 1, 1)};
            const Dispatch segment{ps, {table, sums, s.w0, s.w1, s.meta, out, partials, counters, gate},
                                   bytes(GgufDecodeParams{K, splits, stride, N}), 9, MTLSizeMake(N / 64, splits, 1), MTLSizeMake(128, 1, 1)};
            runOnce({table16, segment}, 1);
            uint32_t *cnt = (uint32_t *)counters.contents; uint32_t live = 0;
            for (uint32_t t = 0; t < stride / 64; ++t) live += cnt[t] != 0;
            if (live) { printf("  %s L=%u: %u counters not reset FAIL\n", where, lanes, live); ++formatFailures; memset(cnt, 0, counters.length); }
            const uint16_t *y = (const uint16_t *)out.contents; size_t touched = 0;
            for (size_t i = 0; i < out.length / 2; ++i) touched += (i / stride >= 8 * lanes || i % stride < N) && y[i] != kSentinel;
            if (touched) { printf("  %s L=%u: %zu writes outside the segment's rows and columns FAIL\n", where, lanes, touched); ++formatFailures; }
          };
          snprintf(where, sizeof where, "%s scale %.0e %c S=%u", fmtName(fi), scale, ep, splits);
          for (uint32_t r = 0; r < kMaxLanes; ++r) run(1, laneX[r], laneA[r], laneY[r], 0x7FC00000u);
          for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes) {
            run(lanes, X, A, Y, 0x7E800000u);
            for (uint32_t r = 0; r < lanes; ++r)
              if (memcmp((const uint16_t *)Y.contents + size_t(8 * r) * stride, laneY[r].contents, laneY[r].length)) {
                printf("  %s L=%u: lane %u differs from its one-lane dispatch FAIL\n", where, lanes, r); ++formatFailures; }
            if (ep != 'a') continue;
            // The fused kernel on the same table with one segment: the same bits.
            id<MTLComputePipelineState> fused = pso(lib, "decode_linear_gguf_sg_fused_l" + std::to_string(lanes));
            if (!fused) { ++formatFailures; continue; }
            std::fill_n((uint32_t *)partials.contents, partials.length / 4, 0x7FC00000u);
            std::fill_n((uint16_t *)Yfused.contents, Yfused.length / 2, uint16_t(kSentinel));
            const GgufDecodeFusedParams fp{K, splits, stride, {N, 0, 0}, {uint32_t(fi), 0, 0}, {N, 0, 0}};
            runOnce({Dispatch{fused, {table, sums, s.w0, s.w1, s.meta, s.w0, s.w1, s.meta, s.w0, s.w1, s.meta, Yfused, partials, counters},
                              bytes(fp), 14, MTLSizeMake(N / 64, splits, 1), MTLSizeMake(128, 1, 1)}}, 1);
            if (memcmp(Y.contents, Yfused.contents, Y.length)) { printf("  %s L=%u: fused kernel differs FAIL\n", where, lanes); ++formatFailures; }
          }
          // fp64 over all four lanes: the interval of bf16 values within the bound, and the flip count.
          const uint16_t *y = (const uint16_t *)Y.contents; size_t flips = 0, outside = 0;
          for (uint32_t r = 0; r < kRows; ++r)
            for (uint32_t n = 0; n < N; ++n) {
              const size_t i = size_t(r) * N + n, at = size_t(r) * stride + N + n;
              const double d = dot[i], e = bound[i], a = bf2f(aux[at]), got = bf2f(y[at]);
              double lo = d - e, hi = d + e, exact = d;
              if (ep == 'r') { exact = d + a; lo = exact - e - std::ldexp(std::fabs(exact), -22); hi = exact + e + std::ldexp(std::fabs(exact), -22); }
              if (ep == 'g') {
                const double sg = a / (1.0 + std::exp(-a)), u0 = bf2f(bf16Rne(float(lo))), u1 = bf2f(bf16Rne(float(hi)));
                const double p0 = u0 * sg, p1 = u1 * sg, slack = std::ldexp(std::max(std::fabs(p0), std::fabs(p1)), -18);
                lo = std::min(p0, p1) - slack; hi = std::max(p0, p1) + slack; exact = bf2f(bf16Rne(float(d))) * sg;
              }
              if (!(got >= bf2f(bf16Rne(float(lo))) && got <= bf2f(bf16Rne(float(hi))))) {
                if (outside++ < 3) printf("  %s row %u col %u: %.9g outside [%.9g, %.9g] (fp64 %.9g)\n", where, r, n, got, lo, hi, exact); }
              flips += y[at] != bf16Rne(float(exact));
            }
          worstFlips = std::max(worstFlips, double(flips) / (kRows * N));
          if (outside || flips * 100 > size_t(kRows) * N) {
            printf("  %s: %zu outputs outside the bound, %zu of %u differ from bf16(fp64) FAIL\n", where, outside, flips, kRows * N); ++formatFailures; }
        }
      printf("%-6s register scale %.0e: L 1-4, S 1-8, a/r/g: at most %.2f%% of outputs differ from bf16(fp64) %s\n",
             fmtName(fi), scale, 100 * worstFlips, formatFailures ? "FAIL" : "ok");
      failures += formatFailures;
    }
  // One fused dispatch over three segments of different formats writes the bits of the three single-tensor
  // dispatches: columns [0, 512) in format f, [512, 768) in f + 3 and [768, 1024) in f + 5.
  for (int fi = 0; fi < FMT_COUNT; ++fi) {
    const Fmt formats[3] = {Fmt(fi), Fmt((fi + 3) % FMT_COUNT), Fmt((fi + 5) % FMT_COUNT)};
    const uint32_t cols[3] = {512, 256, 256}, offsets[3] = {0, 512, 768};
    Seg segs[3];
    for (int i = 0; i < 3; ++i) segs[i] = makeSegK(formats[i], cols[i], K, offsets[i]);
    id<MTLBuffer> X = mkbuf(uint64_t(kRows) * K * 2), Y = mkbuf(uint64_t(kRows) * stride * 2), Yf = mkbuf(Y.length);
    { std::uniform_real_distribution<float> d(-1.f, 1.f); uint16_t *xx = (uint16_t *)X.contents; for (uint64_t i = 0; i < uint64_t(kRows) * K; ++i) xx[i] = f2bf(d(rng)); }
    int fusedFailures = 0;
    for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes)
      for (const uint32_t splits : {1u, 4u, 8u}) {
        id<MTLComputePipelineState> fused = pso(lib, "decode_linear_gguf_sg_fused_l" + std::to_string(lanes));
        if (!fused) { ++fusedFailures; continue; }
        const Dispatch table16{prepare, {X, table, sums}, bytes(K), 3, MTLSizeMake(K / 32, lanes, 1), MTLSizeMake(128, 1, 1)};
        std::vector<Dispatch> apart{table16};
        for (int i = 0; i < 3; ++i) {
          char name[64]; snprintf(name, sizeof name, "decode_linear_gguf_sg_%s_l%u_a", fmtName(formats[i]), lanes);
          id<MTLComputePipelineState> tensor = pso(lib, name);
          if (!tensor) { ++fusedFailures; break; }
          apart.push_back({tensor, {table, sums, segs[i].w0, segs[i].w1, segs[i].meta, Y, partials, counters, Y},
                           bytes(GgufDecodeParams{K, splits, stride, offsets[i]}), 9, MTLSizeMake(cols[i] / 64, splits, 1), MTLSizeMake(128, 1, 1)});
        }
        if (apart.size() != 4) continue;
        const GgufDecodeFusedParams all{K, splits, stride, {cols[0], cols[1], cols[2]},
                                    {uint32_t(formats[0]), uint32_t(formats[1]), uint32_t(formats[2])}, {offsets[0], offsets[1], offsets[2]}};
        const Dispatch one{fused, {table, sums, segs[0].w0, segs[0].w1, segs[0].meta, segs[1].w0, segs[1].w1, segs[1].meta,
                                   segs[2].w0, segs[2].w1, segs[2].meta, Yf, partials, counters}, bytes(all), 14,
                           MTLSizeMake(stride / 64, splits, 1), MTLSizeMake(128, 1, 1)};
        std::fill_n((uint16_t *)Y.contents, Y.length / 2, uint16_t(kSentinel));
        std::fill_n((uint16_t *)Yf.contents, Yf.length / 2, uint16_t(kSentinel));
        std::fill_n((uint32_t *)partials.contents, partials.length / 4, 0x7FC00000u);
        runOnce(apart, 1);
        runOnce({table16, one}, 1);
        uint32_t live = 0;
        for (uint32_t t = 0; t < stride / 64; ++t) live += ((uint32_t *)counters.contents)[t] != 0;
        if (live || memcmp(Y.contents, Yf.contents, Y.length)) {
          printf("  fused [%s|%s|%s] L=%u S=%u: differs from the single-tensor dispatches or leaves %u counters FAIL\n",
                 fmtName(formats[0]), fmtName(formats[1]), fmtName(formats[2]), lanes, splits, live);
          ++fusedFailures; memset(counters.contents, 0, counters.length);
        }
      }
    printf("register fused [%s|%s|%s]: L 1-4, S 1/4/8 equal to one dispatch per tensor %s\n",
           fmtName(formats[0]), fmtName(formats[1]), fmtName(formats[2]), fusedFailures ? "FAIL" : "ok");
    failures += fusedFailures;
  }
  return failures;
}
// The staged decode tile of a batch (gguf_linear.metal): 8, 16 or 32 rows, as LinearPlan::storageRows rounds them.
static uint32_t stagedTileRows(uint32_t rows) { return rows <= 8 ? 8 : rows <= 16 ? 16 : 32; }
// One staged decode dispatch of `rows` rows (its tile's kernel over storage of the tile's rows): single tensors by
// epilogue a/r/g, "fused" three segments, "gateup" a gate pass into G and an up pass; splits over grid.y.
struct StagedCase { std::string kind; uint32_t splits; };
static std::vector<Dispatch> stagedDispatch(id<MTLLibrary> lib, const StagedCase &c, uint32_t rows, uint32_t N, uint32_t K,
                                            const Seg &s, const Seg &g, const Seg (&parts)[3], id<MTLBuffer> x, id<MTLBuffer> aux,
                                            id<MTLBuffer> y, id<MTLBuffer> gate, id<MTLBuffer> partials, id<MTLBuffer> counters) {
  const std::string m = "_m" + std::to_string(stagedTileRows(rows));
  const MTLSize grid = MTLSizeMake(N / 64, c.splits, 1), tg = MTLSizeMake(64, 1, 1);
  const auto params = bytes(GgufDecodeParams{K, c.splits, N, 0});
  const auto tensor = [&](const Seg &w, char ep, id<MTLBuffer> out, id<MTLBuffer> a) {
    return Dispatch{pso(lib, std::string("gguf_decode_") + fmtName(w.fmt) + m + "_" + ep), {x, w.w0, w.w1, w.meta, out, partials, counters, a},
                    params, 8, grid, tg}; };
  if (c.kind == "fused") {
    const GgufDecodeFusedParams fp{K, c.splits, N, {256, 256, 256}, {uint32_t(parts[0].fmt), uint32_t(parts[1].fmt), uint32_t(parts[2].fmt)}, {0, 256, 512}};
    return {Dispatch{pso(lib, "gguf_decode_fused" + m), {x, parts[0].w0, parts[0].w1, parts[0].meta, parts[1].w0, parts[1].w1, parts[1].meta,
                                                         parts[2].w0, parts[2].w1, parts[2].meta, y, partials, counters}, bytes(fp), 13, grid, tg}};
  }
  if (c.kind == "gateup") return {tensor(g, 'a', gate, gate), tensor(s, 'g', y, gate)};
  return {tensor(s, c.kind[0], y, aux)};
}
// Staged decode tiles put every request lane in one matmul2d destination, so their lane invariance is the MPP property
// that a row's result does not depend on the destination height, nor on the other rows' inputs. Lane r of every 16- and
// 32-row dispatch, and of a 3-lane dispatch on the 32-row tile whose padding rows hold NaN inputs, must equal the 8-row
// kernel on its rows bitwise: plain, residual and up-with-gate tiles, split-K at 2, 4 and 8 partitions (partials
// poisoned with NaN), fused segments and gate/up, with and without splits.
static int stagedLanes(id<MTLLibrary> lib) {
  constexpr uint32_t N = 768, K = 2048, kMaxLanes = 4, kRows = 8 * kMaxLanes;
  int failures = 0;
  std::uniform_real_distribution<float> random(-1.f, 1.f);
  const auto filled = [&](uint64_t elements) {
    id<MTLBuffer> b = mkbuf(elements * 2); uint16_t *v = (uint16_t *)b.contents;
    for (uint64_t i = 0; i < elements; ++i) v[i] = f2bf(random(rng));
    return b; };
  id<MTLBuffer> X = filled(uint64_t(kRows) * K), A = filled(uint64_t(kRows) * N);
  id<MTLBuffer> laneX[kMaxLanes], laneA[kMaxLanes];
  for (uint32_t r = 0; r < kMaxLanes; ++r) {
    laneX[r] = mkbuf(uint64_t(8) * K * 2); memcpy(laneX[r].contents, (const uint16_t *)X.contents + size_t(8 * r) * K, laneX[r].length);
    laneA[r] = mkbuf(uint64_t(8) * N * 2); memcpy(laneA[r].contents, (const uint16_t *)A.contents + size_t(8 * r) * N, laneA[r].length);
  }
  // The 3-lane step's storage: the fourth lane's rows hold NaN inputs.
  id<MTLBuffer> X3 = mkbuf(X.length), A3 = mkbuf(A.length);
  memcpy(X3.contents, X.contents, X.length); memcpy(A3.contents, A.contents, A.length);
  std::fill_n((uint16_t *)X3.contents + size_t(24) * K, size_t(8) * K, uint16_t(0x7FC0));
  std::fill_n((uint16_t *)A3.contents + size_t(24) * N, size_t(8) * N, uint16_t(0x7FC0));
  id<MTLBuffer> partials = mkbuf(uint64_t(8) * kRows * N * 4), counters = mkbuf(N / 64 * 4);
  memset(counters.contents, 0, counters.length);
  for (int fi = 0; fi < FMT_COUNT; ++fi) {
    const Fmt f = Fmt(fi), f2 = Fmt((fi + 3) % FMT_COUNT), f3 = Fmt((fi + 5) % FMT_COUNT);
    const Seg s = makeSegK(f, N, K, 0), g = makeSegK(f2, N, K, 0);
    const Seg parts[3] = {makeSegK(f, 256, K, 0), makeSegK(f2, 256, K, 256), makeSegK(f3, 256, K, 512)};
    int formatFailures = 0;
    for (const StagedCase &c : std::vector<StagedCase>{{"a", 1}, {"r", 1}, {"g", 1}, {"a", 2}, {"r", 4}, {"g", 8},
                                                       {"fused", 1}, {"fused", 4}, {"gateup", 1}, {"gateup", 2}}) {
      const auto run = [&](uint32_t rows, id<MTLBuffer> x, id<MTLBuffer> aux, id<MTLBuffer> y) {
        id<MTLBuffer> gate = mkbuf(uint64_t(stagedTileRows(rows)) * N * 2);
        const std::vector<Dispatch> ds = stagedDispatch(lib, c, rows, N, K, s, g, parts, x, aux, y, gate, partials, counters);
        for (const Dispatch &d : ds) if (!d.p) return false;
        std::fill_n((uint32_t *)partials.contents, partials.length / 4, 0x7FC00000u);
        runOnce(ds, 1);
        return true; };
      id<MTLBuffer> laneY[kMaxLanes];
      bool ok = true;
      for (uint32_t r = 0; r < kMaxLanes && ok; ++r) ok = run(8, laneX[r], laneA[r], laneY[r] = mkbuf(uint64_t(8) * N * 2));
      for (uint32_t lanes = 2; lanes <= kMaxLanes && ok; ++lanes) {
        id<MTLBuffer> Y = mkbuf(uint64_t(stagedTileRows(8 * lanes)) * N * 2);
        ok = run(8 * lanes, lanes == 3 ? X3 : X, lanes == 3 ? A3 : A, Y);
        for (uint32_t r = 0; r < lanes && ok; ++r)
          if (memcmp((const uint16_t *)Y.contents + size_t(8 * r) * N, laneY[r].contents, laneY[r].length)) {
            printf("  staged %s %s S=%u: lane %u of %u lanes differs from the 8-row kernel FAIL\n", fmtName(f), c.kind.c_str(), c.splits, r, lanes);
            ++formatFailures;
          }
      }
      if (!ok) ++formatFailures;
      for (uint32_t t = 0; t < N / 64; ++t)
        if (((uint32_t *)counters.contents)[t]) { printf("  staged %s %s: counters not reset FAIL\n", fmtName(f), c.kind.c_str()); ++formatFailures; break; }
    }
    printf("%-6s staged lanes: 2-4 lanes (3 on the 32-row tile over NaN padding) plain, residual, gate, split 2/4/8, fused, gate/up equal to 8-row kernels %s\n",
           fmtName(f), formatFailures ? "FAIL" : "ok");
    failures += formatFailures;
  }
  return failures;
}
// time-sg <rounds> <per> <fmt[+fmt..]> <N[+N..]> <K> <a|r|g> [S+S..]: the Apple9 register kernels against the
// staged kernels on one projection of up to three column segments, L = 1..4 lanes and every valid K split, with the
// weights DRAM-cold (a ring of copies of at least 384 MiB) and the cases in ABBA order; prints medians. The register
// path runs one dispatch per segment, at one split count for all or at the per-segment list, and the fused kernel over
// all segments; the staged path runs its single-tensor kernel (one segment) or its fused kernel, on the tile of the
// lanes' rows (8, 16 or 32). g is gate/up: a gate pass plus an up pass on both paths, each streaming its own weights.
// The prepare dispatch is timed alone.
static int timeRegister(id<MTLLibrary> lib, int argc, char **argv) {
  if (argc < 9) { std::cerr << "usage: harness_prod <metallib> time-sg <rounds> <per> <fmt[+fmt]> <N[+N]> <K> <a|r|g> [S+S]\n"; return 2; }
  const auto list = [](const std::string &text) { std::vector<std::string> out; size_t at = 0;
    for (size_t plus; (plus = text.find('+', at)) != std::string::npos; at = plus + 1) out.push_back(text.substr(at, plus - at));
    out.push_back(text.substr(at)); return out; };
  const uint32_t rounds = std::stoul(argv[3]), per = std::stoul(argv[4]), K = std::stoul(argv[7]);
  const char ep = argv[8][0];
  struct Segment { int fmt; uint32_t N, offset, splits; Seg image; std::vector<Seg> ring; };
  std::vector<Segment> segs;
  const auto formats = list(argv[5]), widths = list(argv[6]), splitList = argc > 9 ? list(argv[9]) : std::vector<std::string>{};
  uint32_t N = 0;
  for (size_t i = 0; i < widths.size(); ++i) {
    Segment g{-1, uint32_t(std::stoul(widths[i])), N, splitList.size() == widths.size() ? uint32_t(std::stoul(splitList[i])) : 0u, {}, {}};
    for (int f = 0; f < FMT_COUNT; ++f) if (formats[std::min(i, formats.size() - 1)] == fmtName(f)) g.fmt = f;
    if (g.fmt < 0 || !g.N || g.N % 256) { std::cerr << "bad segment\n"; return 2; }
    N += g.N; segs.push_back(std::move(g));
  }
  if (!K || K % 256 || !strchr("arg", ep) || segs.size() > 3 || (segs.size() > 1 && ep != 'a')) { std::cerr << "bad case\n"; return 2; }
  // Up to 16 K splits, one more than production takes, to see whether its cap of 8 binds on large GPUs.
  constexpr uint32_t kMaxLanes = 4, kRows = 8 * kMaxLanes, kMaxSplits = 16;
  uint64_t imageBytes = 0;
  for (Segment &g : segs) { g.image = makeSegK(Fmt(g.fmt), g.N, K, g.offset); imageBytes += streamBytes(Fmt(g.fmt), g.N, K); }
  const uint32_t copies = (uint32_t)std::min<uint64_t>(64, std::max<uint64_t>(3, ((384ull << 20) + imageBytes - 1) / imageBytes));
  const auto clone = [](id<MTLBuffer> b) { id<MTLBuffer> c = mkbuf(b.length); memcpy(c.contents, b.contents, b.length); return c; };
  for (Segment &g : segs)
    for (uint32_t i = 0; i < copies; ++i) {
      Seg c = g.image; c.w0 = clone(g.image.w0); c.meta = clone(g.image.meta);
      c.w1 = kQuantFormats[g.fmt].plane1_bytes ? clone(g.image.w1) : c.meta;
      g.ring.push_back(c);
    }
  id<MTLBuffer> X = mkbuf(uint64_t(kRows) * K * 2), table = mkbuf(uint64_t(kRows) * K * 2), sums = mkbuf(uint64_t(kMaxLanes) * (K * 3 / 4) * 4);
  id<MTLBuffer> Y = mkbuf(uint64_t(kRows) * N * 2), A = mkbuf(uint64_t(kRows) * N * 2), G = mkbuf(uint64_t(kRows) * N * 2);
  id<MTLBuffer> partials = mkbuf(uint64_t(kMaxSplits) * kRows * N * 4), counters = mkbuf(N / 64 * 4);
  memset(counters.contents, 0, counters.length);
  { std::uniform_real_distribution<float> d(-1.f, 1.f); uint16_t *x = (uint16_t *)X.contents, *a = (uint16_t *)A.contents;
    for (uint64_t i = 0; i < uint64_t(kRows) * K; ++i) x[i] = f2bf(d(rng));
    for (uint64_t i = 0; i < uint64_t(kRows) * N; ++i) a[i] = f2bf(d(rng)); }
  const auto encode = [](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, std::initializer_list<id<MTLBuffer>> bufs,
                         const std::vector<uint8_t> &params, MTLSize grid, MTLSize tg) {
    [enc setComputePipelineState:p]; NSUInteger i = 0;
    for (id<MTLBuffer> b : bufs) [enc setBuffer:b offset:0 atIndex:i++];
    [enc setBytes:params.data() length:params.size() atIndex:i];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  };
  id<MTLComputePipelineState> prepare = pso(lib, "decode_linear_gguf_prepare");
  if (!prepare) return 1;
  // A case encodes one projection with weight copy c of every segment (gate/up: the up weights are copy c + 1).
  struct Case { std::string label; uint32_t lanes, weights; std::function<void(id<MTLComputeCommandEncoder>, uint32_t)> run; std::vector<double> t; };
  std::vector<Case> cases;
  const Segment &one = segs.front();
  for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes) {
    const uint32_t rows = 8 * lanes;
    cases.push_back({"prepare", lanes, 0, [=](id<MTLComputeCommandEncoder> enc, uint32_t) {
      encode(enc, prepare, {X, table, sums}, bytes(K), MTLSizeMake(K / 32, lanes, 1), MTLSizeMake(128, 1, 1)); }, {}});
    // staged: the production kernel of the tile of these rows, at every split count
    const std::string m = "_m" + std::to_string(stagedTileRows(rows));
    for (const uint32_t splits : {1u, 2u, 4u, 8u, 16u}) {
      if ((K / 32) % splits) continue;
      const MTLSize grid = MTLSizeMake(N / 64, splits, 1), tg = MTLSizeMake(64, 1, 1);
      const auto params = bytes(GgufDecodeParams{K, splits, N, 0});
      if (segs.size() > 1) {
        id<MTLComputePipelineState> p = pso(lib, "gguf_decode_fused" + m);
        if (!p) return 1;
        GgufDecodeFusedParams fp{K, splits, N, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (size_t i = 0; i < segs.size(); ++i) { fp.cols[i] = segs[i].N; fp.fmt[i] = segs[i].fmt; fp.offset[i] = segs[i].offset; }
        const auto fused = bytes(fp);
        cases.push_back({"staged fused S" + std::to_string(splits), lanes, 1, [&, p, fused, grid, tg](id<MTLComputeCommandEncoder> enc, uint32_t c) {
          const Seg &a = segs[0].ring[c], &b = segs[std::min<size_t>(1, segs.size() - 1)].ring[c], &d = segs.back().ring[c];
          encode(enc, p, {X, a.w0, a.w1, a.meta, b.w0, b.w1, b.meta, d.w0, d.w1, d.meta, Y, partials, counters}, fused, grid, tg); }, {}});
      } else if (ep == 'g') {
        id<MTLComputePipelineState> pg = pso(lib, std::string("gguf_decode_") + fmtName(one.fmt) + m + "_a");
        id<MTLComputePipelineState> pu = pso(lib, std::string("gguf_decode_") + fmtName(one.fmt) + m + "_g");
        if (!pg || !pu) return 1;
        cases.push_back({"staged S" + std::to_string(splits), lanes, 2, [&, pg, pu, params, grid, tg](id<MTLComputeCommandEncoder> enc, uint32_t c) {
          const Seg &g = one.ring[c], &u = one.ring[(c + 1) % copies];
          encode(enc, pg, {X, g.w0, g.w1, g.meta, G, partials, counters, G}, params, grid, tg);
          encode(enc, pu, {X, u.w0, u.w1, u.meta, Y, partials, counters, G}, params, grid, tg); }, {}});
      } else {
        id<MTLComputePipelineState> p = pso(lib, std::string("gguf_decode_") + fmtName(one.fmt) + m + "_" + ep);
        if (!p) return 1;
        cases.push_back({"staged S" + std::to_string(splits), lanes, 1, [&, p, params, grid, tg](id<MTLComputeCommandEncoder> enc, uint32_t c) {
          const Seg &w = one.ring[c];
          encode(enc, p, {X, w.w0, w.w1, w.meta, Y, partials, counters, A}, params, grid, tg); }, {}});
      }
    }
    // register: one dispatch per segment, splits on 256-input units; gate/up is a gate pass and an up pass
    std::vector<uint32_t> variants{1u, 2u, 4u, 8u, 16u};
    if (segs.front().splits) variants.push_back(0);   // the per-segment list
    for (const uint32_t uniform : variants) {
      bool valid = true;
      std::vector<std::pair<id<MTLComputePipelineState>, id<MTLComputePipelineState>>> kernels;
      std::vector<std::vector<uint8_t>> params;
      std::string label = "register S";
      for (const Segment &g : segs) {
        const uint32_t splits = uniform ? uniform : g.splits;
        valid &= K / 256 >= splits;
        label += (label.back() == 'S' ? "" : "+") + std::to_string(splits);
        char name[80];
        snprintf(name, sizeof name, "decode_linear_gguf_sg_%s_l%u_%c", fmtName(g.fmt), lanes, ep);
        kernels.push_back({pso(lib, name), pso(lib, std::string(name).replace(strlen(name) - 1, 1, "a"))});
        if (!kernels.back().first || !kernels.back().second) return 1;
        params.push_back(bytes(GgufDecodeParams{K, splits, N, g.offset}));
      }
      if (!valid) continue;
      cases.push_back({label, lanes, ep == 'g' ? 2u : 1u, [&, kernels, params, uniform](id<MTLComputeCommandEncoder> enc, uint32_t c) {
        for (size_t i = 0; i < segs.size(); ++i) {
          const Segment &g = segs[i];
          const uint32_t splits = uniform ? uniform : g.splits;
          const MTLSize grid = MTLSizeMake(g.N / 64, splits, 1);
          const Seg &w = g.ring[c], &u = g.ring[(c + 1) % copies];
          if (ep == 'g') {
            encode(enc, kernels[i].second, {table, sums, w.w0, w.w1, w.meta, G, partials, counters, G}, params[i], grid, MTLSizeMake(128, 1, 1));
            encode(enc, kernels[i].first, {table, sums, u.w0, u.w1, u.meta, Y, partials, counters, G}, params[i], grid, MTLSizeMake(128, 1, 1));
          } else {
            encode(enc, kernels[i].first, {table, sums, w.w0, w.w1, w.meta, Y, partials, counters, A}, params[i], grid, MTLSizeMake(128, 1, 1));
          }
        } }, {}});
    }
    // the fused kernel: every segment in one dispatch
    for (const uint32_t splits : {1u, 2u, 4u, 8u, 16u}) {
      if (K / 256 < splits || ep != 'a') continue;
      id<MTLComputePipelineState> p = pso(lib, "decode_linear_gguf_sg_fused_l" + std::to_string(lanes));
      if (!p) return 1;
      GgufDecodeFusedParams fp{K, splits, N, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      for (size_t i = 0; i < segs.size(); ++i) { fp.cols[i] = segs[i].N; fp.fmt[i] = segs[i].fmt; fp.offset[i] = segs[i].offset; }
      const auto params = bytes(fp);
      cases.push_back({"fused S" + std::to_string(splits), lanes, 1, [&, p, params, splits](id<MTLComputeCommandEncoder> enc, uint32_t c) {
        const Seg &a = segs[0].ring[c], &b = segs[std::min<size_t>(1, segs.size() - 1)].ring[c], &d = segs.back().ring[c];
        encode(enc, p, {table, sums, a.w0, a.w1, a.meta, b.w0, b.w1, b.meta, d.w0, d.w1, d.meta, Y, partials, counters}, params,
               MTLSizeMake(N / 64, splits, 1), MTLSizeMake(128, 1, 1)); }, {}});
    }
  }
  // Every register case reads the four-lane table.
  runOnce({Dispatch{prepare, {X, table, sums}, bytes(K), 3, MTLSizeMake(K / 32, kMaxLanes, 1), MTLSizeMake(128, 1, 1)}}, 1);
  uint32_t next = 0;
  const auto time = [&](Case &c) {
    id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (uint32_t i = 0; i < per; ++i) { c.run(enc, next % copies); next += c.weights; }
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.error) { std::cerr << "GPU error: " << cb.error.localizedDescription.UTF8String << "\n"; exit(1); }
    return (cb.GPUEndTime - cb.GPUStartTime) / per;
  };
  for (Case &c : cases) { double spent = 0; while (spent < 0.1) spent += time(c) * per; }
  for (uint32_t r = 0; r < rounds; ++r)
    for (size_t i = 0; i < cases.size(); ++i) { Case &c = cases[(r & 1) ? cases.size() - 1 - i : i]; c.t.push_back(time(c) * 1e3); }
  for (uint32_t t = 0; t < N / 64; ++t) if (((const uint32_t *)counters.contents)[t]) { std::cerr << "counters not reset\n"; return 1; }
  printf("%s %sx%u %c, %u weight copies of %.1f MB, median ms of %u rounds x %u:\n", argv[5], argv[6], K, ep, copies, imageBytes / 1e6, rounds, per);
  for (uint32_t lanes = 1; lanes <= kMaxLanes; ++lanes) {
    printf("  L%u", lanes);
    for (Case &c : cases) if (c.lanes == lanes) { std::sort(c.t.begin(), c.t.end()); printf(" | %s %.4f", c.label.c_str(), c.t[c.t.size() / 2]); }
    printf("\n");
  }
  return 0;
}
int main(int argc, char **argv) { @autoreleasepool {
  if (argc < 2) { std::cerr << "usage: harness_prod <metallib> [full|dequant|time-sg ...] [input-scale] [seed]\n"; return 2; }
  dev = MTLCreateSystemDefaultDevice(); queue = [dev newCommandQueue];
  NSError *err = nil; id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err];
  if (!lib) { std::cerr << "library load failed: " << err.localizedDescription.UTF8String << "\n"; return 1; }
  if (const char *oracle = std::getenv("SPLASH_GGML_ORACLE")) {
    ggmlOracle = dlopen(oracle, RTLD_NOW | RTLD_LOCAL);
    if (!ggmlOracle) { std::cerr << dlerror() << "\n"; return 2; }
    printf("Checking CPU dequantization against upstream GGML\n");
  }
  if (argc > 2 && std::string(argv[2]) == "dequant") {
    int failures = 0;
    for (int fi = 0; fi < FMT_COUNT; ++fi) {
      const uint32_t N = 256, K = 1024;
      Seg s = makeSeg(Fmt(fi), N, K, 0);
      id<MTLBuffer> Y = mkbuf(uint64_t(N) * K * 2);
      const GgufDecodeParams p{K, 1, N, 0};
      const std::string name = std::string("gguf_test_dequant_") + fmtName(fi);
      Dispatch d{pso(lib, name), {s.w0, s.w1, s.meta, Y}, bytes(p), 4,
                 MTLSizeMake(N * (K / 32) / 32, 1, 1), MTLSizeMake(32, 1, 1)};
      runOnce({d}, 1);
      const uint16_t *got = (const uint16_t *)Y.contents;
      size_t mismatch = 0;
      for (size_t i = 0; i < s.Wf.size(); ++i) mismatch += got[i] != f2h(s.Wf[i]);
      printf("%s: %zu weights, %zu different from FP16(GGML FP32 dequantization)\n", fmtName(fi), s.Wf.size(), mismatch);
      failures += mismatch != 0;
    }
    failures += oracleFailures;
    return failures ? 1 : 0;
  }
  if (argc > 2 && std::string(argv[2]) == "time-sg") return timeRegister(lib, argc, argv);
  const bool full = argc > 2 && std::string(argv[2]) == "full";
  const float inputScale = argc > 3 ? std::stof(argv[3]) : 1.f;
  if (argc > 4) rng.seed(std::stoul(argv[4]));
  printf("input scale %.0f, %s CPU oracle\n", inputScale, inputScale == 1.f ? "native GGUF + staged" : "sparse staged");
  const uint32_t K = 1024; int failures = 0;
  auto inputs = [&](uint32_t rows) {
    id<MTLBuffer> X = mkbuf(uint64_t(rows) * K * 2);
    auto *values = (uint16_t *)X.contents;
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    for (uint64_t i = 0; i < uint64_t(rows) * K; ++i) {
      // Sparse large values isolate BF16 input range from cancellation error.
      float value = inputScale == 1.f ? random(rng) :
          (i % K == (i / K * 31) % K ? (i / K % 2 ? -inputScale : inputScale) : 0.f);
      values[i] = f2bf(value);
    }
    return std::make_pair(X, X); // The CPU oracle reads the exact BF16 input bits.
  };
  auto refGemm = [&](id<MTLBuffer> Xref, const Seg &s, uint32_t rows, uint32_t stride, std::vector<double> &out, bool staged = false) { const auto &weights = (staged || inputScale != 1.f) ? s.Ws : s.Wf; const uint16_t *x = (const uint16_t *)Xref.contents;
    for (uint32_t r = 0; r < rows; ++r) for (uint32_t n = 0; n < s.N; ++n) { double acc = 0; for (uint32_t k = 0; k < K; ++k) acc += (double)bf2f(x[(size_t)r * K + k]) * weights[(size_t)n * K + k]; out[(size_t)r * stride + s.colOffset + n] = acc; } };
  auto compare = [&](const char *name, id<MTLBuffer> Y, const std::vector<double> &ref, uint32_t rows, uint32_t stride) { const uint16_t *y = (const uint16_t *)Y.contents; double maxerr = 0, maxOutsideRounding = 0, sumrel = 0; size_t cnt = 0;
    for (uint32_t r = 0; r < rows; ++r) for (uint32_t n = 0; n < stride; ++n) { double got = bf2f(y[(size_t)r * stride + n]), want = ref[(size_t)r * stride + n]; maxerr = std::max(maxerr, std::fabs(got - want));
      // Account separately for the final BF16 rounding cell; the pre-output
      // absolute error budget and the native-GGUF relative-error gate stay fixed.
      const uint16_t bits = y[(size_t)r * stride + n];
      const double halfUlp = 0.5 * std::max(std::fabs(bf2f(uint16_t(bits + 1)) - got), std::fabs(bf2f(uint16_t(bits - 1)) - got));
      maxOutsideRounding = std::max(maxOutsideRounding, std::fabs(got - want) - halfUlp); sumrel += std::fabs(got - want) / (std::fabs(want) + 1e-3); ++cnt; }
    const bool soft = strstr(name, "vs fp64") != nullptr; const bool ok = sumrel / cnt < 0.02 && (soft || maxOutsideRounding < 0.5 * inputScale); if (!ok) ++failures; printf("%-46s rows=%-3u maxabs=%.2e meanrel=%.1e %s\n", name, rows, maxerr, sumrel / cnt, ok ? "ok" : "FAIL"); };
  const uint32_t rowsList[4] = {8, 16, 24, 32};
  // Decode inputs hold the rows of the staged tile (a 3-lane batch runs the 32-row tile over four lanes of storage):
  // the padding rows are NaN, which no active row may see.
  auto decodeInputs = [&](uint32_t rows) {
    const uint32_t tile = stagedTileRows(rows);
    id<MTLBuffer> X = inputs(tile).first;
    std::fill_n((uint16_t *)X.contents + size_t(rows) * K, size_t(tile - rows) * K, uint16_t(0x7FC0));
    return X;
  };
  id<MTLBuffer> splitPartials = mkbuf(uint64_t(8) * 32 * 2048 * 4), splitCounters = mkbuf(4096);
  memset(splitCounters.contents, 0, splitCounters.length);
  const auto countersReset = [&](const char *name) {
    const uint32_t *cnt = (const uint32_t *)splitCounters.contents;
    for (uint32_t t = 0; t < 1024; ++t) if (cnt[t]) { printf("  %s: counter %u not reset (%u) FAIL\n", name, t, cnt[t]); ++failures; memset(splitCounters.contents, 0, splitCounters.length); return; } };
  const auto decodeName = [](Fmt f, uint32_t rows, char ep) { return std::string("gguf_decode_") + fmtName(f) + "_m" + std::to_string(stagedTileRows(rows)) + "_" + ep; };
  // 1) fused three-segment dispatch, one format triple per row count, one and four K partitions
  const Fmt triples[4][3] = {{Q4K, IQ4XS, Q80}, {Q5K, Q4K, Q6K}, {IQ4NL, Q3K, IQ3S}, {IQ4XS, Q5K, Q4K}};
  for (int ti = 0; ti < 4; ++ti) { const uint32_t rows = rowsList[ti], tile = stagedTileRows(rows); Seg s0 = makeSeg(triples[ti][0], 1024, K, 0), s1 = makeSeg(triples[ti][1], 512, K, 1024), s2 = makeSeg(triples[ti][2], 256, K, 1536);
    const uint32_t N = 1792; id<MTLBuffer> Xbf = decodeInputs(rows), Y = mkbuf(uint64_t(tile) * N * 2); std::vector<double> ref((size_t)rows * N);
    for (auto *s : {&s0, &s1, &s2}) refGemm(Xbf, *s, rows, N, ref);
    for (const uint32_t splits : {1u, 4u}) {
      const GgufDecodeFusedParams fp{K, splits, N, {s0.N, s1.N, s2.N}, {s0.fmt, s1.fmt, s2.fmt}, {s0.colOffset, s1.colOffset, s2.colOffset}};
      const std::string name = "gguf_decode_fused_m" + std::to_string(tile); id<MTLComputePipelineState> ps = pso(lib, name); if (!ps) { ++failures; continue; }
      Dispatch d{ps, {Xbf, s0.w0, s0.w1, s0.meta, s1.w0, s1.w1, s1.meta, s2.w0, s2.w1, s2.meta, Y, splitPartials, splitCounters}, bytes(fp), 13, MTLSizeMake(N / 64, splits, 1), MTLSizeMake(64, 1, 1)};
      runOnce({d}, 1); char label[96]; snprintf(label, sizeof label, "%s S%u [%s|%s|%s]", name.c_str(), splits, fmtName(s0.fmt), fmtName(s1.fmt), fmtName(s2.fmt)); compare(label, Y, ref, rows, N); countersReset(label); } }
  // 2) gate/up: a gate pass (a) into G and an up pass (g) that applies silu(bf16 gate) to the bf16 up value; full mode
  //    covers every format pair, at one, two or four K partitions by row count
  const Fmt gu[4][2] = {{IQ4XS, Q4K}, {Q5K, Q5K}, {Q4K, IQ4XS}, {Q3K, Q6K}};
  for (int ti = 0; ti < (full ? 4 * FMT_COUNT * FMT_COUNT : 4); ++ti) {
    const uint32_t rows = rowsList[full ? ti / (FMT_COUNT * FMT_COUNT) : ti], tile = stagedTileRows(rows), N = full ? 256 : 1024, splits = rows == 8 ? 1 : rows == 16 ? 2 : 4;
    const Fmt gf = full ? Fmt((ti / FMT_COUNT) % FMT_COUNT) : gu[ti][0];
    const Fmt uf = full ? Fmt(ti % FMT_COUNT) : gu[ti][1];
    Seg g = makeSeg(gf, N, K, 0), u = makeSeg(uf, N, K, 0);
    id<MTLBuffer> Xbf = decodeInputs(rows), Y = mkbuf(uint64_t(tile) * N * 2), G = mkbuf(uint64_t(tile) * N * 2);
    std::vector<double> rg((size_t)rows * N), ru((size_t)rows * N), ref((size_t)rows * N);
    refGemm(Xbf, g, rows, N, rg); refGemm(Xbf, u, rows, N, ru);
    for (size_t i = 0; i < ref.size(); ++i) { double gg = bf2f(f2bf((float)rg[i])), uu = bf2f(f2bf((float)ru[i])); ref[i] = gg / (1.0 + std::exp(-gg)) * uu; }
    const GgufDecodeParams pq{K, splits, N, 0}; const std::string n1 = decodeName(g.fmt, rows, 'a'), n2 = decodeName(u.fmt, rows, 'g');
    id<MTLComputePipelineState> p1 = pso(lib, n1), p2 = pso(lib, n2); if (!p1 || !p2) { ++failures; continue; }
    Dispatch d1{p1, {Xbf, g.w0, g.w1, g.meta, G, splitPartials, splitCounters, G}, bytes(pq), 8, MTLSizeMake(N / 64, splits, 1), MTLSizeMake(64, 1, 1)};
    Dispatch d2{p2, {Xbf, u.w0, u.w1, u.meta, Y, splitPartials, splitCounters, G}, bytes(pq), 8, MTLSizeMake(N / 64, splits, 1), MTLSizeMake(64, 1, 1)};
    runOnce({d1, d2}, 1); char label[96]; snprintf(label, sizeof label, "gate/up %s+%s S%u vs fp64", n1.c_str(), fmtName(u.fmt), splits); compare(label, Y, ref, rows, N); countersReset(label);
    // bf16 rounding of the gate before silu makes the fp64 reference flip by one bf16 ulp on rare elements: every
    // difference beyond 2% must be such a flip of gate or up over the staged (half) weights.
    refGemm(Xbf, g, rows, N, rg, true); refGemm(Xbf, u, rows, N, ru, true);
    for (size_t i = 0; i < ref.size(); ++i) { double gg = bf2f(f2bf((float)rg[i])), uu = bf2f(f2bf((float)ru[i])); ref[i] = gg / (1.0 + std::exp(-gg)) * uu; }
    const uint16_t *a = (const uint16_t *)Y.contents; size_t unexplained = 0; bool finite = true;
    for (size_t i = 0; i < (size_t)rows * N; ++i) { double got = bf2f(a[i]); finite &= std::isfinite(got); if (std::fabs(got - ref[i]) <= 0.02 * std::fabs(ref[i]) + 0.02) continue;
      bool expl = false; for (int dg = -1; dg <= 1 && !expl; ++dg) for (int du = -1; du <= 1 && !expl; ++du) {
        uint16_t gb = f2bf((float)rg[i]); uint16_t ub = f2bf((float)ru[i]); gb = (uint16_t)(gb + dg); ub = (uint16_t)(ub + du);
        double gg = bf2f(gb), uu = bf2f(ub), alt = gg / (1.0 + std::exp(-gg)) * uu; if (std::fabs(got - alt) <= 0.02 * std::fabs(alt) + 0.02) expl = true; }
      if (!expl) ++unexplained; }
    const bool ok = finite && unexplained == 0; if (!ok) ++failures;
    printf("%-46s rows=%-3u staged fp64 mismatches not explained by a 1-ulp bf16 flip of gate/up: %zu %s\n", label, rows, unexplained, ok ? "ok" : "FAIL"); }
  // 3) split-K with last-arriver reduction + residual epilogue, every format, splits 4, strided output (out_stride 2N, offset N)
  for (int fi = 0; fi < FMT_COUNT; ++fi) { const uint32_t rows = rowsList[fi % 4], tile = stagedTileRows(rows), N = 512, splits = 4; Seg s = makeSeg((Fmt)fi, N, K, 0);
    id<MTLBuffer> Xbf = decodeInputs(rows), Y = mkbuf(uint64_t(tile) * 2 * N * 2), R = mkbuf(uint64_t(tile) * 2 * N * 2);
    { uint16_t *rr = (uint16_t *)R.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(tile) * 2 * N; ++i) rr[i] = f2bf(d(rng)); }
    std::vector<double> ref((size_t)rows * 2 * N, 0.0), part((size_t)rows * N); refGemm(Xbf, s, rows, N, part);
    for (uint32_t r = 0; r < rows; ++r) for (uint32_t n = 0; n < N; ++n) ref[(size_t)r * 2 * N + N + n] = part[(size_t)r * N + n] + bf2f(((uint16_t *)R.contents)[(size_t)r * 2 * N + N + n]);
    for (uint32_t r = 0; r < tile; ++r) for (uint32_t n = 0; n < N; ++n) ((uint16_t *)Y.contents)[(size_t)r * 2 * N + n] = 0;
    const std::string name = decodeName((Fmt)fi, rows, 'r'); id<MTLComputePipelineState> ps = pso(lib, name); if (!ps) { ++failures; continue; }
    Dispatch d{ps, {Xbf, s.w0, s.w1, s.meta, Y, splitPartials, splitCounters, R}, bytes(GgufDecodeParams{K, splits, 2 * N, N}), 8, MTLSizeMake(N / 64, splits, 1), MTLSizeMake(64, 1, 1)};
    runOnce({d}, 3); char label[96]; snprintf(label, sizeof label, "%s S%u strided", name.c_str(), splits); compare(label, Y, ref, rows, 2 * N); countersReset(label); }
  // 4) single-segment a / r / g decode tiles (one partition) and prefill pfa/pfr/pfg
  for (int fi = 0; fi < FMT_COUNT; ++fi) { const uint32_t N = 1024; Seg s = makeSeg((Fmt)fi, N, K, 0);
    for (const char ep : {'a', 'r', 'g'}) { const uint32_t rows = rowsList[(fi + (ep == 'r') + 2 * (ep == 'g')) % 4], tile = stagedTileRows(rows);
      id<MTLBuffer> Xbf = decodeInputs(rows), Y = mkbuf(uint64_t(tile) * N * 2), A = mkbuf(uint64_t(tile) * N * 2); { uint16_t *aa = (uint16_t *)A.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(tile) * N; ++i) aa[i] = f2bf(d(rng)); }
      std::vector<double> ref((size_t)rows * N); refGemm(Xbf, s, rows, N, ref);
      if (ep == 'r') for (size_t i = 0; i < ref.size(); ++i) ref[i] += bf2f(((uint16_t *)A.contents)[i]);
      if (ep == 'g') for (size_t i = 0; i < ref.size(); ++i) { double gg = bf2f(((uint16_t *)A.contents)[i]); ref[i] *= gg / (1.0 + std::exp(-gg)); }
      const std::string name = decodeName((Fmt)fi, rows, ep); id<MTLComputePipelineState> ps = pso(lib, name); if (!ps) { ++failures; continue; }
      Dispatch d{ps, {Xbf, s.w0, s.w1, s.meta, Y, Y, Y, A}, bytes(GgufDecodeParams{K, 1, N, 0}), 8, MTLSizeMake(N / 64, 1, 1), MTLSizeMake(64, 1, 1)}; runOnce({d}, 1); compare(name.c_str(), Y, ref, rows, N); }
    // Prefill tiles over a 168-row chunk: the second 128-row tile holds 40 rows, so its last two simdgroups skip their
    // matmuls and leave rows 192..255 untouched. Chunks of up to 32 rows run the decode tiles (LinearGguf.cpp): rows
    // 0..m-1 through the decode tile of m rows must equal the prefill tile's bit for bit.
    for (const char ep : {'a', 'r', 'g'}) { const uint32_t rows = 168, storage = 256;
      auto [Xbf, Xref] = inputs(storage); id<MTLBuffer> Y = mkbuf(uint64_t(storage) * N * 2), A = mkbuf(uint64_t(storage) * N * 2);
      { uint16_t *aa = (uint16_t *)A.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(storage) * N; ++i) aa[i] = f2bf(d(rng)); }
      std::vector<double> ref((size_t)rows * N); refGemm(Xref, s, rows, N, ref);
      if (ep == 'r') for (size_t i = 0; i < ref.size(); ++i) ref[i] += bf2f(((uint16_t *)A.contents)[i]);
      if (ep == 'g') for (size_t i = 0; i < ref.size(); ++i) { double gg = bf2f(((uint16_t *)A.contents)[i]); ref[i] *= gg / (1.0 + std::exp(-gg)); }
      char name[80]; snprintf(name, sizeof name, "pf%c_%s_r32_sg4_n64_k64_p1", ep, fmtName(fi)); id<MTLComputePipelineState> ps = pso(lib, name); if (!ps) { ++failures; continue; }
      std::vector<id<MTLBuffer>> bufs{Xbf, s.w0, s.w1, s.meta, Y}; if (ep != 'a') bufs.push_back(A);
      std::fill_n((uint16_t *)Y.contents, size_t(storage) * N, uint16_t(0xFFFF));
      runOnce({Dispatch{ps, bufs, bytes(GgufPrefillParams{N, K, rows, 0, 0}), (int)bufs.size(), MTLSizeMake(storage / 128, N / 64, 1), MTLSizeMake(128, 1, 1)}}, 1);
      compare(name, Y, ref, rows, N);
      const uint16_t *y = (const uint16_t *)Y.contents; size_t touched = 0;
      for (size_t i = size_t(192) * N; i < size_t(storage) * N; ++i) touched += y[i] != 0xFFFF;
      if (touched) { printf("  %s: %zu writes by simdgroups past the chunk FAIL\n", name, touched); ++failures; }
      for (const uint32_t m : {8u, 16u, 24u, 32u}) {
        const std::string dname = decodeName((Fmt)fi, m, ep); id<MTLComputePipelineState> dp = pso(lib, dname); if (!dp) { ++failures; continue; }
        id<MTLBuffer> Ym = mkbuf(uint64_t(stagedTileRows(m)) * N * 2);
        runOnce({Dispatch{dp, {Xbf, s.w0, s.w1, s.meta, Ym, Ym, Ym, ep == 'a' ? Ym : A}, bytes(GgufDecodeParams{K, 1, N, 0}), 8, MTLSizeMake(N / 64, 1, 1), MTLSizeMake(64, 1, 1)}}, 1);
        if (memcmp(Ym.contents, Y.contents, size_t(m) * N * 2)) { printf("  %s rows 0..%u differ from %s FAIL\n", dname.c_str(), m - 1, name); ++failures; } } } }
  // 5) split-K visibility at production K: two projections back to back share the partials and counters, as every
  //    GGUF split projection of a decode step does, at split counts 2, 4 and 8 (stagedSplits takes each for
  //    such shapes at some core count; 8 is its cap). The partials are poisoned before
  //    each dispatch, with a large finite value and in another run with NaN, so a partial read before its writer
  //    published it changes the output. Both outputs must pass the fp64 check the unsplit kernels pass, not depend
  //    on the poison or the run, and leave every counter at zero.
  struct SplitCase { Fmt fmt; uint32_t N, K, splits, epilogue; };
  for (uint32_t rows : {8u, 32u}) {
    const SplitCase pairs[2][2] = {{{Q4K, 5120, 6144, 4, GGUF_EPILOGUE_RESIDUAL}, {Q6K, 5120, 17408, 8, GGUF_EPILOGUE_RESIDUAL}},
                                   {{IQ4XS, 12288, 5120, rows <= 16 ? 4u : 2u, GGUF_EPILOGUE_NONE}, {Q80, 1024, 5120, 8, GGUF_EPILOGUE_RESIDUAL}}};
    for (const auto &pair : pairs) {
      struct Operand { id<MTLBuffer> X, R, Y, w0, w1, meta; std::vector<double> ref; };
      std::vector<Operand> ops; uint64_t partialBytes = 0, counterBytes = 0;
      for (const SplitCase &c : pair) {
        Operand o; std::vector<uint8_t> native = makeNative(c.fmt, c.N, c.K); Packed pk = repack(c.fmt, native, c.N, c.K, nullptr);
        o.w0 = upload(pk.w0); o.w1 = upload(pk.w1); o.meta = upload(pk.meta); if (!kQuantFormats[c.fmt].plane1_bytes) o.w1 = o.meta;
        o.X = mkbuf(uint64_t(rows) * c.K * 2); o.R = mkbuf(uint64_t(rows) * c.N * 2); o.Y = mkbuf(uint64_t(rows) * c.N * 2);
        std::uniform_real_distribution<float> d(-1.f, 1.f); uint16_t *x = (uint16_t *)o.X.contents, *r = (uint16_t *)o.R.contents;
        for (uint64_t i = 0; i < uint64_t(rows) * c.K; ++i) x[i] = f2bf(d(rng));
        for (uint64_t i = 0; i < uint64_t(rows) * c.N; ++i) r[i] = f2bf(d(rng));
        // fp64 over the GGML fp32 values, one weight row at a time (no N x K float copy at these shapes)
        o.ref.assign(size_t(rows) * c.N, 0.0); std::vector<float> w(c.K);
        for (uint32_t n = 0; n < c.N; ++n) { const uint8_t *row = native.data() + size_t(n) * rowBytes(c.fmt, c.K);
          for (uint32_t g = 0; g < c.K / 32; ++g) { uint8_t p0[32], p1[8]; groupPack(c.fmt, row, g, w.data() + g * 32, p0, p1); }
          for (uint32_t m = 0; m < rows; ++m) { double acc = 0; for (uint32_t k = 0; k < c.K; ++k) acc += (double)bf2f(x[size_t(m) * c.K + k]) * w[k];
            o.ref[size_t(m) * c.N + n] = acc + (c.epilogue == GGUF_EPILOGUE_RESIDUAL ? bf2f(r[size_t(m) * c.N + n]) : 0.0); } }
        partialBytes = std::max(partialBytes, uint64_t(c.splits) * rows * c.N * 4); counterBytes = std::max(counterBytes, uint64_t(c.N / 64) * 4);
        ops.push_back(std::move(o));
      }
      id<MTLBuffer> partials = mkbuf(partialBytes), poison = mkbuf(partialBytes), counters = mkbuf(counterBytes); memset(counters.contents, 0, counterBytes);
      id<MTLComputePipelineState> copy = pso(lib, "gguf_copy"); if (!copy) { ++failures; continue; }
      std::vector<Dispatch> unsplit, split;
      for (size_t j = 0; j < 2; ++j) { const SplitCase &c = pair[j]; const Operand &o = ops[j]; const bool residual = c.epilogue == GGUF_EPILOGUE_RESIDUAL;
        id<MTLComputePipelineState> ps = pso(lib, decodeName(c.fmt, rows, residual ? 'r' : 'a'));
        unsplit.push_back({ps, {o.X, o.w0, o.w1, o.meta, o.Y, o.Y, o.Y, residual ? o.R : o.Y}, bytes(GgufDecodeParams{c.K, 1, c.N, 0}), 8, MTLSizeMake(c.N / 64, 1, 1), MTLSizeMake(64, 1, 1)});
        // gguf_copy, the production byte copy, poisons the partials in dispatch order.
        split.push_back({copy, {poison, partials}, bytes(GgufCopyParams{0, 0, uint32_t(partialBytes)}), 2, MTLSizeMake((partialBytes + 4095) / 4096, 1, 1), MTLSizeMake(256, 1, 1)});
        split.push_back({ps, {o.X, o.w0, o.w1, o.meta, o.Y, partials, counters, residual ? o.R : o.Y},
                         bytes(GgufDecodeParams{c.K, c.splits, c.N, 0}), 8, MTLSizeMake(c.N / 64, c.splits, 1), MTLSizeMake(64, 1, 1)}); }
      if (std::any_of(split.begin(), split.end(), [](const Dispatch &d) { return !d.p; }) ||
          std::any_of(unsplit.begin(), unsplit.end(), [](const Dispatch &d) { return !d.p; })) { ++failures; continue; }
      runOnce(unsplit, 1);
      for (size_t j = 0; j < 2; ++j) { char label[96]; snprintf(label, sizeof label, "unsplit %s %ux%u", fmtName(pair[j].fmt), pair[j].N, pair[j].K); compare(label, ops[j].Y, ops[j].ref, rows, pair[j].N); }
      std::vector<std::vector<uint8_t>> first;
      for (uint32_t bits : {0x7E800000u, 0x7FC00000u, 0x7E800000u}) {
        std::fill_n((uint32_t *)poison.contents, partialBytes / 4, bits); runOnce(split, 1);
        const uint32_t *cnt = (const uint32_t *)counters.contents; for (uint64_t t = 0; t < counterBytes / 4; ++t) if (cnt[t]) { printf("  counter %llu not reset (%u)\n", (unsigned long long)t, cnt[t]); ++failures; }
        for (size_t j = 0; j < 2; ++j) { const uint8_t *y = (const uint8_t *)ops[j].Y.contents; std::vector<uint8_t> out(y, y + ops[j].Y.length);
          if (first.size() < 2) first.push_back(out);
          else if (out != first[j]) { printf("  %s %ux%u S=%u: output depends on the poison or the run FAIL\n", fmtName(pair[j].fmt), pair[j].N, pair[j].K, pair[j].splits); ++failures; } } }
      for (size_t j = 0; j < 2; ++j) { char label[96]; snprintf(label, sizeof label, "shared split %s %ux%u S=%u", fmtName(pair[j].fmt), pair[j].N, pair[j].K, pair[j].splits); compare(label, ops[j].Y, ops[j].ref, rows, pair[j].N); } } }
  // 6) Apple9 register kernels
  failures += registerKernels(lib);
  // 7) staged lane invariance (Apple10 decode)
  failures += stagedLanes(lib);
  failures += oracleFailures;
  printf("%s (%d failures)\n", failures ? "VALIDATION FAILED" : "all production kernels validated", failures);
  return failures ? 1 : 0; } }

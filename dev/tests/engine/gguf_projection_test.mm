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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
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
static double timeIt(const std::vector<Dispatch> &ds, int iters) { runOnce(ds, 2); double best = 1e9; for (int r = 0; r < 5; ++r) best = std::min(best, runOnce(ds, iters)); return best; }
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
int main(int argc, char **argv) { @autoreleasepool {
  if (argc < 2) { std::cerr << "usage: harness_prod <metallib> [full|dequant|time-gu] [input-scale] [seed]\n"; return 2; }
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
      GgufParams p{N, K, 0, 0, 0};
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
  // 1) fused three-segment dispatch, one format triple per row count
  const Fmt triples[4][3] = {{Q4K, IQ4XS, Q80}, {Q5K, Q4K, Q6K}, {IQ4NL, Q3K, IQ3S}, {IQ4XS, Q5K, Q4K}};
  for (int ti = 0; ti < 4; ++ti) { const uint32_t rows = rowsList[ti]; Seg s0 = makeSeg(triples[ti][0], 1024, K, 0), s1 = makeSeg(triples[ti][1], 512, K, 1024), s2 = makeSeg(triples[ti][2], 256, K, 1536);
    const uint32_t N = 1792; auto [Xbf, Xref] = inputs(rows); id<MTLBuffer> Y = mkbuf(uint64_t(rows) * N * 2); std::vector<double> ref((size_t)rows * N);
    for (auto *s : {&s0, &s1, &s2}) refGemm(Xref, *s, rows, N, ref);
    GgufFusedParams fp{K, N, 3, 0, {s0.N, s1.N, s2.N}, {s0.fmt, s1.fmt, s2.fmt}, {s0.colOffset, s1.colOffset, s2.colOffset}};
    char name[64]; snprintf(name, sizeof name, "gguf_fused_m%u", rows); id<MTLComputePipelineState> ps = pso(lib, name); if (!ps) { ++failures; continue; }
    Dispatch d{ps, {Xbf, s0.w0, s0.w1, s0.meta, s1.w0, s1.w1, s1.meta, s2.w0, s2.w1, s2.meta, Y}, bytes(fp), 11, MTLSizeMake(N / 64, 1, 1), MTLSizeMake(64, 1, 1)};
    runOnce({d}, 1); char label[96]; snprintf(label, sizeof label, "%s [%s|%s|%s]", name, fmtName(s0.fmt), fmtName(s1.fmt), fmtName(s2.fmt)); compare(label, Y, ref, rows, N); }
  // 2) gate + up fused
  const Fmt gu[4][2] = {{IQ4XS, Q4K}, {Q5K, Q5K}, {Q4K, IQ4XS}, {Q3K, Q6K}};
  for (int ti = 0; ti < (full ? 4 * FMT_COUNT * FMT_COUNT : 4); ++ti) {
    const uint32_t rows = rowsList[full ? ti / (FMT_COUNT * FMT_COUNT) : ti], N = full ? 256 : 1024;
    const Fmt gf = full ? Fmt((ti / FMT_COUNT) % FMT_COUNT) : gu[ti][0];
    const Fmt uf = full ? Fmt(ti % FMT_COUNT) : gu[ti][1];
    Seg g = makeSeg(gf, N, K, 0), u = makeSeg(uf, N, K, 0);
    auto [Xbf, Xref] = inputs(rows); id<MTLBuffer> Y = mkbuf(uint64_t(rows) * N * 2); std::vector<double> rg((size_t)rows * N), ru((size_t)rows * N), ref((size_t)rows * N);
    refGemm(Xref, g, rows, N, rg); refGemm(Xref, u, rows, N, ru);
    for (size_t i = 0; i < ref.size(); ++i) { double gg = bf2f(f2bf((float)rg[i])), uu = bf2f(f2bf((float)ru[i])); ref[i] = gg / (1.0 + std::exp(-gg)) * uu; }
    GgufGateUpParams gp{K, N, N, g.fmt, u.fmt}; char name[64]; snprintf(name, sizeof name, "gguf_gateup_m%u", rows); id<MTLComputePipelineState> ps = pso(lib, name); if (!ps) { ++failures; continue; }
    Dispatch d{ps, {Xbf, g.w0, g.w1, g.meta, u.w0, u.w1, u.meta, Y}, bytes(gp), 8, MTLSizeMake(N / 64, 1, 1), MTLSizeMake(64, 1, 1)};
    runOnce({d}, 1); char label[96]; snprintf(label, sizeof label, "%s [%s|%s] vs fp64", name, fmtName(g.fmt), fmtName(u.fmt)); compare(label, Y, ref, rows, N);
    refGemm(Xref, g, rows, N, rg, true); refGemm(Xref, u, rows, N, ru, true);
    for (size_t i = 0; i < ref.size(); ++i) { double gg = bf2f(f2bf((float)rg[i])), uu = bf2f(f2bf((float)ru[i])); ref[i] = gg / (1.0 + std::exp(-gg)) * uu; }
    // bf16 rounding of the gate before silu makes the fp64 reference flip by one bf16 ulp on rare elements; the two-kernel GPU path
    // (validated sga gate -> bf16 scratch, sgg up with silu epilogue) has the same accumulation and must match near-exactly.
    id<MTLBuffer> G = mkbuf(uint64_t(rows) * N * 2), Y2 = mkbuf(uint64_t(rows) * N * 2); GgufParams pq{N, K, N / 64, 0, 0}; char n1[80], n2[80];
    snprintf(n1, sizeof n1, "sga_%s_m%u_c32_sg2_k32_b2_p1", fmtName(g.fmt), rows); snprintf(n2, sizeof n2, "sgg_%s_m%u_c32_sg2_k32_b2_p1", fmtName(u.fmt), rows);
    id<MTLComputePipelineState> p1 = pso(lib, n1), p2 = pso(lib, n2);
    if (p1 && p2) { Dispatch d1{p1, {Xbf, g.w0, g.w1, g.meta, G}, bytes(pq), 5, MTLSizeMake(N / 64, 1, 1), MTLSizeMake(64, 1, 1)}, d2{p2, {Xbf, u.w0, u.w1, u.meta, Y2, G}, bytes(pq), 6, MTLSizeMake(N / 64, 1, 1), MTLSizeMake(64, 1, 1)};
      runOnce({d1, d2}, 1); const uint16_t *a = (const uint16_t *)Y.contents, *b2 = (const uint16_t *)Y2.contents; double maxd = 0; size_t diff = 0; bool finite = true;
      for (size_t i = 0; i < (size_t)rows * N; ++i) { double dd = std::fabs(bf2f(a[i]) - bf2f(b2[i])); finite &= std::isfinite(dd); if (!std::isfinite(dd) || dd > 0) ++diff; maxd = std::max(maxd, dd); }
      // remaining differences must be bf16 rounding-boundary flips of gate or up (fp32 accumulation order differs from fp64)
      size_t unexplained = 0; const uint16_t *xx = (const uint16_t *)Xref.contents; (void)xx;
      for (size_t i = 0; i < (size_t)rows * N; ++i) { double got = bf2f(a[i]); if (std::fabs(got - ref[i]) <= 0.02 * std::fabs(ref[i]) + 0.02) continue;
        bool expl = false; for (int dg = -1; dg <= 1 && !expl; ++dg) for (int du = -1; du <= 1 && !expl; ++du) {
          uint16_t gb = f2bf((float)rg[i]); uint16_t ub = f2bf((float)ru[i]); gb = (uint16_t)(gb + dg); ub = (uint16_t)(ub + du);
          double gg = bf2f(gb), uu = bf2f(ub), alt = gg / (1.0 + std::exp(-gg)) * uu; if (std::fabs(got - alt) <= 0.02 * std::fabs(alt) + 0.02) expl = true; }
        if (!expl) ++unexplained; }
      const bool ok = finite && maxd < 1e-2 && unexplained == 0; if (!ok) ++failures;
      printf("%-46s rows=%-3u staged fp64 mismatches not explained by a 1-ulp bf16 flip of gate/up: %zu\n", name, rows, unexplained);
      printf("%-46s rows=%-3u vs two-kernel path: %zu differing elements, max diff %.2e %s\n", name, rows, diff, maxd, ok ? "ok" : "FAIL"); } }
  // 3) split-K with last-arriver reduction + residual epilogue, every format, splits 4, strided output (out_stride 2N, offset N)
  for (int fi = 0; fi < FMT_COUNT; ++fi) { const uint32_t rows = rowsList[fi % 4], N = 512, splits = 4; Seg s = makeSeg((Fmt)fi, N, K, 0);
    auto [Xbf, Xref] = inputs(rows); id<MTLBuffer> Y = mkbuf(uint64_t(rows) * 2 * N * 2), R = mkbuf(uint64_t(rows) * 2 * N * 2), partials = mkbuf(uint64_t(splits) * rows * N * 4), counters = mkbuf(4096);
    memset(counters.contents, 0, 4096); { uint16_t *rr = (uint16_t *)R.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(rows) * 2 * N; ++i) rr[i] = f2bf(d(rng)); }
    std::vector<double> ref((size_t)rows * 2 * N, 0.0), part((size_t)rows * N); refGemm(Xref, s, rows, N, part);
    for (uint32_t r = 0; r < rows; ++r) for (uint32_t n = 0; n < N; ++n) ref[(size_t)r * 2 * N + N + n] = part[(size_t)r * N + n] + bf2f(((uint16_t *)R.contents)[(size_t)r * 2 * N + N + n]);
    for (uint32_t r = 0; r < rows; ++r) for (uint32_t n = 0; n < N; ++n) { ((uint16_t *)Y.contents)[(size_t)r * 2 * N + n] = 0; ref[(size_t)r * 2 * N + n] = 0; }
    GgufSplitParams sp{N, K, splits, 2 * N, N, 1}; char name[64]; snprintf(name, sizeof name, "gguf_splitk_%s_m%u", fmtName(fi), rows); id<MTLComputePipelineState> ps = pso(lib, name); if (!ps) { ++failures; continue; }
    Dispatch d{ps, {Xbf, s.w0, s.w1, s.meta, partials, counters, Y, R}, bytes(sp), 8, MTLSizeMake(N / 64, splits, 1), MTLSizeMake(64, 1, 1)};
    runOnce({d}, 3); compare(name, Y, ref, rows, 2 * N); const uint32_t *cnt = (const uint32_t *)counters.contents; for (uint32_t t = 0; t < N / 64; ++t) if (cnt[t]) { printf("  counter %u not reset (%u)\n", t, cnt[t]); ++failures; } }
  // 4) single-segment sga / sgr / sgg (production ABI) and prefill pfa/pfr/pfg
  for (int fi = 0; fi < FMT_COUNT; ++fi) { const uint32_t N = 1024; Seg s = makeSeg((Fmt)fi, N, K, 0);
    for (const char *fam : {"sga", "sgr", "sgg"}) { const uint32_t rows = rowsList[(fi + (fam[2] == 'r') + 2 * (fam[2] == 'g')) % 4];
      auto [Xbf, Xref] = inputs(rows); id<MTLBuffer> Y = mkbuf(uint64_t(rows) * N * 2), A = mkbuf(uint64_t(rows) * N * 2); { uint16_t *aa = (uint16_t *)A.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(rows) * N; ++i) aa[i] = f2bf(d(rng)); }
      std::vector<double> ref((size_t)rows * N); refGemm(Xref, s, rows, N, ref);
      if (fam[2] == 'r') for (size_t i = 0; i < ref.size(); ++i) ref[i] += bf2f(((uint16_t *)A.contents)[i]);
      if (fam[2] == 'g') for (size_t i = 0; i < ref.size(); ++i) { double gg = bf2f(((uint16_t *)A.contents)[i]); ref[i] *= gg / (1.0 + std::exp(-gg)); }
      GgufParams pq{N, K, N / 64, 0, 0}; char name[80]; snprintf(name, sizeof name, "%s_%s_m%u_c32_sg2_k32_b2_p1", fam, fmtName(fi), rows); id<MTLComputePipelineState> ps = pso(lib, name); if (!ps) { ++failures; continue; }
      std::vector<id<MTLBuffer>> bufs{Xbf, s.w0, s.w1, s.meta, Y}; if (fam[2] != 'a') bufs.push_back(A);
      Dispatch d{ps, bufs, bytes(pq), (int)bufs.size(), MTLSizeMake(N / 64, 1, 1), MTLSizeMake(64, 1, 1)}; runOnce({d}, 1); compare(name, Y, ref, rows, N); }
    { const uint32_t rows = 128; auto [Xbf, Xref] = inputs(rows); id<MTLBuffer> Y = mkbuf(uint64_t(rows) * N * 2); std::vector<double> ref((size_t)rows * N); refGemm(Xref, s, rows, N, ref);
      GgufParams pq{N, K, 0, 0, 0}; char name[80]; snprintf(name, sizeof name, "pfa_%s_r32_sg4_n64_k64_p1", fmtName(fi)); id<MTLComputePipelineState> ps = pso(lib, name); if (!ps) { ++failures; continue; }
      Dispatch d{ps, {Xbf, s.w0, s.w1, s.meta, Y}, bytes(pq), 5, MTLSizeMake(rows / 128, N / 64, 1), MTLSizeMake(128, 1, 1)}; runOnce({d}, 1); compare(name, Y, ref, rows, N); } }
  // 5) split-K visibility at production K: two projections back to back share the partials and counters, as every
  //    GGUF split projection of a decode step does, at the splits addGguf picks (ggufSplits: n <= 1024 -> 8,
  //    n <= 6144 -> 4 for K <= 6144 else 8, n <= 12288 -> 4 for <= 16 rows else 2). The partials are poisoned before
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
        char name[80]; snprintf(name, sizeof name, "%s_%s_m%u_c32_sg2_k32_b2_p1", residual ? "sgr" : "sga", fmtName(c.fmt), rows);
        std::vector<id<MTLBuffer>> bufs{o.X, o.w0, o.w1, o.meta, o.Y}; if (residual) bufs.push_back(o.R);
        unsplit.push_back({pso(lib, name), bufs, bytes(GgufParams{c.N, c.K, c.N / 64, 0, 0}), (int)bufs.size(), MTLSizeMake(c.N / 64, 1, 1), MTLSizeMake(64, 1, 1)});
        snprintf(name, sizeof name, "gguf_splitk_%s_m%u", fmtName(c.fmt), rows);
        // gguf_copy, the production byte copy, poisons the partials in dispatch order.
        split.push_back({copy, {poison, partials}, bytes(GgufCopyParams{0, 0, uint32_t(partialBytes)}), 2, MTLSizeMake((partialBytes + 4095) / 4096, 1, 1), MTLSizeMake(256, 1, 1)});
        split.push_back({pso(lib, name), {o.X, o.w0, o.w1, o.meta, partials, counters, o.Y, residual ? o.R : o.Y},
                         bytes(GgufSplitParams{c.N, c.K, c.splits, c.N, 0, c.epilogue}), 8, MTLSizeMake(c.N / 64, c.splits, 1), MTLSizeMake(64, 1, 1)}); }
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
  failures += oracleFailures;
  printf("%s (%d failures)\n", failures ? "VALIDATION FAILED" : "all production kernels validated", failures);
  if (argc > 2 && std::string(argv[2]) == "time-gu") {
    const uint32_t NN = 17408, KK = 5120;
    for (auto pair : gu) {
      Seg g = makeSegK(pair[0], NN, KK, 0), u = makeSegK(pair[1], NN, KK, 0);
      for (uint32_t rows : rowsList) {
        id<MTLBuffer> X = mkbuf(uint64_t(rows) * KK * 2), Y = mkbuf(uint64_t(rows) * NN * 2), G = mkbuf(uint64_t(rows) * NN * 2);
        std::fill_n((uint16_t *)X.contents, size_t(rows) * KK, f2bf(0.25f));
        GgufGateUpParams gp{KK, NN, NN, g.fmt, u.fmt};
        GgufParams pq{NN, KK, NN / 64, 0, 0};
        char fused[40], gate[80], up[80];
        snprintf(fused, sizeof fused, "gguf_gateup_m%u", rows);
        snprintf(gate, sizeof gate, "sga_%s_m%u_c32_sg2_k32_b2_p1", fmtName(g.fmt), rows);
        snprintf(up, sizeof up, "sgg_%s_m%u_c32_sg2_k32_b2_p1", fmtName(u.fmt), rows);
        Dispatch df{pso(lib, fused), {X, g.w0, g.w1, g.meta, u.w0, u.w1, u.meta, Y}, bytes(gp), 8, MTLSizeMake(NN / 64, 1, 1), MTLSizeMake(64, 1, 1)};
        Dispatch dg{pso(lib, gate), {X, g.w0, g.w1, g.meta, G}, bytes(pq), 5, MTLSizeMake(NN / 64, 1, 1), MTLSizeMake(64, 1, 1)};
        Dispatch du{pso(lib, up), {X, u.w0, u.w1, u.meta, Y, G}, bytes(pq), 6, MTLSizeMake(NN / 64, 1, 1), MTLSizeMake(64, 1, 1)};
        const double tf = timeIt({df}, 10), ts = timeIt({dg, du}, 10);
        printf("BENCH gate=%s up=%s rows=%u fused_ms=%.6f separate_ms=%.6f\n", fmtName(g.fmt), fmtName(u.fmt), rows, tf * 1e3, ts * 1e3);
      }
    }
  }
  if (argc > 2 && std::string(argv[2]) == "time") {   // runtime-format-switch cost: gguf_fused (1 segment) vs sga, serialized by a dependent touch kernel
    const uint32_t KK = 5120, rows = 8; id<MTLComputePipelineState> touch = pso(lib, "gguf_touch");
    for (uint32_t NN : {16640u, 17408u}) for (int fi : {0, 1, 3}) { Seg s = makeSegK((Fmt)fi, NN, KK, 0);
      id<MTLBuffer> Xbf = mkbuf(uint64_t(rows) * KK * 2), Y = mkbuf(uint64_t(rows) * NN * 2); { uint16_t *a = (uint16_t *)Xbf.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(rows) * KK; ++i) a[i] = f2bf(d(rng)); }
      GgufFusedParams fp{KK, NN, 1, 0, {NN, 0, 0}, {s.fmt, 0, 0}, {0, 0, 0}}; GgufParams pq{NN, KK, NN / 64, 0, 0};
      Dispatch df{pso(lib, "gguf_fused_m8"), {Xbf, s.w0, s.w1, s.meta, s.w0, s.w1, s.meta, s.w0, s.w1, s.meta, Y}, bytes(fp), 11, MTLSizeMake(NN / 64, 1, 1), MTLSizeMake(64, 1, 1)};
      char nm[80]; snprintf(nm, sizeof nm, "sga_%s_m8_c32_sg2_k32_b2_p1", fmtName(fi)); Dispatch ds{pso(lib, nm), {Xbf, s.w0, s.w1, s.meta, Y}, bytes(pq), 5, MTLSizeMake(NN / 64, 1, 1), MTLSizeMake(64, 1, 1)};
      Dispatch dt{touch ? touch : pso(lib, nm), {Y}, {}, -1, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1)};
      auto timeS = [&](Dispatch d) { auto run = [&](int n) { id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          for (int it = 0; it < n; ++it) for (auto *x : {&d, &dt}) { [enc setComputePipelineState:x->p]; for (size_t i = 0; i < x->bufs.size(); ++i) [enc setBuffer:x->bufs[i] offset:0 atIndex:i]; if (x->paramIndex >= 0) [enc setBytes:x->params.data() length:x->params.size() atIndex:x->paramIndex]; [enc dispatchThreadgroups:x->grid threadsPerThreadgroup:x->tg]; }
          [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; return (cb.GPUEndTime - cb.GPUStartTime) / n; }; run(2); double best = 1e9; for (int r = 0; r < 5; ++r) best = std::min(best, run(20)); return best; };
      const double bytes = double(streamBytes((Fmt)fi, NN, KK)); const double tf = timeS(df), ts = timeS(ds);
      printf("N=%u %-6s serialized: gguf_fused(switch) %.3f ms (%.0f GB/s)  sga(compile-time) %.3f ms (%.0f GB/s)\n", NN, fmtName(fi), tf * 1e3, bytes / tf / 1e9, ts * 1e3, bytes / ts / 1e9); } }
  if (argc > 2 && std::string(argv[2]) == "time2") {   // narrow projections at M=8, serialized: split-K counts vs splash's residual_paired kernel
    id<MTLComputePipelineState> touch = pso(lib, "gguf_touch");
    auto timeS = [&](Dispatch d, id<MTLBuffer> y) { Dispatch dt{touch, {y}, {}, -1, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1)};
      auto run = [&](int n) { id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (int it = 0; it < n; ++it) for (auto *x : {&d, &dt}) { [enc setComputePipelineState:x->p]; for (size_t i = 0; i < x->bufs.size(); ++i) [enc setBuffer:x->bufs[i] offset:0 atIndex:i]; if (x->paramIndex >= 0) [enc setBytes:x->params.data() length:x->params.size() atIndex:x->paramIndex]; [enc dispatchThreadgroups:x->grid threadsPerThreadgroup:x->tg]; }
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; return (cb.GPUEndTime - cb.GPUStartTime) / n; }; run(2); double best = 1e9; for (int r = 0; r < 5; ++r) best = std::min(best, run(20)); return best; };
    const uint32_t rows = 8;
    for (auto shape : std::vector<std::pair<uint32_t, uint32_t>>{{5120, 6144}, {5120, 17408}, {6144, 5120}, {1024, 5120}}) { const uint32_t N = shape.first, KK = shape.second;
      id<MTLBuffer> Xbf = mkbuf(uint64_t(rows) * KK * 2), Y = mkbuf(uint64_t(rows) * N * 2), R = mkbuf(uint64_t(rows) * N * 2), partials = mkbuf(uint64_t(8) * rows * N * 4), counters = mkbuf(4096); memset(counters.contents, 0, 4096);
      { uint16_t *a = (uint16_t *)Xbf.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(rows) * KK; ++i) a[i] = f2bf(d(rng)); }
      // splash reference: residual paired N128 kernel, groups = tiles (<= 4 * cores) as its Apple10 policy does for these widths
      const uint64_t wb = uint64_t(N) * KK / 2, sb = uint64_t(N) * (KK / 64) * 2; id<MTLBuffer> Ws = mkbuf(wb), Ss = mkbuf(sb), Bs = mkbuf(sb);
      { uint8_t *w = (uint8_t *)Ws.contents; for (uint64_t i = 0; i < wb; ++i) w[i] = (uint8_t)rng(); uint16_t *sc = (uint16_t *)Ss.contents, *bi = (uint16_t *)Bs.contents; for (uint64_t i = 0; i < sb / 2; ++i) { sc[i] = f2bf(0.01f); bi[i] = f2bf(0.0f); } }
      const uint32_t t128 = N / 128, sgroups = t128 <= 64 ? t128 : 64; Q4Params qp{N, KK, sgroups};
      Dispatch dsp{pso(lib, "decode_linear_q4_n128_residual_paired"), {Xbf, Ws, Ss, Bs, R, Y}, bytes(qp), 6, MTLSizeMake(sgroups, 1, 1), MTLSizeMake(256, 1, 1)};
      const double tsp = timeS(dsp, Y); printf("\n== %ux%u  splash residual_paired: %.3f ms (%.0f GB/s)\n", N, KK, tsp * 1e3, (wb + 2 * sb) / tsp / 1e9);
      for (int fi : {0, 1, 3, 4}) { Seg s = makeSegK((Fmt)fi, N, KK, 0); const double gb = double(streamBytes((Fmt)fi, N, KK));
        char nm[80]; snprintf(nm, sizeof nm, "sgr_%s_m8_c32_sg2_k32_b2_p1", fmtName(fi)); GgufParams pq{N, KK, N / 64, 0, 0};
        Dispatch d1{pso(lib, nm), {Xbf, s.w0, s.w1, s.meta, Y, R}, bytes(pq), 6, MTLSizeMake(N / 64, 1, 1), MTLSizeMake(64, 1, 1)}; double t1 = timeS(d1, Y);
        printf("  %-6s no-split %.3f ms (%.0f GB/s, ratio %.2f)", fmtName(fi), t1 * 1e3, gb / t1 / 1e9, tsp / t1);
        for (uint32_t splits : {2u, 4u, 8u}) { if ((KK / 32) % splits) continue; snprintf(nm, sizeof nm, "gguf_splitk_%s_m8", fmtName(fi)); GgufSplitParams sp{N, KK, splits, N, 0, 1};
          Dispatch d2{pso(lib, nm), {Xbf, s.w0, s.w1, s.meta, partials, counters, Y, R}, bytes(sp), 8, MTLSizeMake(N / 64, splits, 1), MTLSizeMake(64, 1, 1)}; double t2 = timeS(d2, Y);
          printf(" | splits %u: %.3f ms (%.0f GB/s, %.2f)", splits, t2 * 1e3, gb / t2 / 1e9, tsp / t2); }
        printf("\n"); } }
  }
  if (argc > 2 && std::string(argv[2]) == "mmap") {   // lm_head Q6_K from the real package file (mmap + no-copy buffers) vs Metal-allocated copies, serialized
    const char *path = argc > 3 ? argv[3] : "/Users/liang2kl/dev/q4k-m5/pkg-gguf/target/head.bin";
    int fd = open(path, O_RDONLY); struct stat st{}; fstat(fd, &st); void *map = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    const uint32_t N = 248320, KK = 5120, rows = 8; const uint64_t page = 16384;
    // sections: header 16 B | final-norm 10240 B @16384 | desc 64 B @32768 | plane0 @49152 | plane1 | meta (each 16 KiB aligned)
    const uint64_t p0 = uint64_t(N) * (KK / 32) * 16, p1 = uint64_t(N) * (KK / 32) * 8, mb = uint64_t(N) * (KK / 256) * 20;
    const uint64_t off0 = 49152, off1 = (off0 + p0 + page - 1) / page * page, offm = (off1 + p1 + page - 1) / page * page;
    auto wrap = [&](uint64_t off, uint64_t len) { return [dev newBufferWithBytesNoCopy:(uint8_t *)map + off length:(len + page - 1) / page * page options:MTLResourceStorageModeShared deallocator:nil]; };
    id<MTLBuffer> W0 = wrap(off0, p0), W1 = wrap(off1, p1), Mt = wrap(offm, mb);
    id<MTLBuffer> C0 = mkbuf(p0), C1 = mkbuf(p1), Cm = mkbuf(mb); memcpy(C0.contents, (uint8_t *)map + off0, p0); memcpy(C1.contents, (uint8_t *)map + off1, p1); memcpy(Cm.contents, (uint8_t *)map + offm, mb);
    id<MTLBuffer> Xbf = mkbuf(uint64_t(rows) * KK * 2), Y = mkbuf(uint64_t(rows) * N * 2), Y2 = mkbuf(uint64_t(rows) * N * 2); { uint16_t *a = (uint16_t *)Xbf.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(rows) * KK; ++i) a[i] = f2bf(d(rng)); }
    id<MTLComputePipelineState> touch = pso(lib, "gguf_touch"), ps = pso(lib, "sga_q6k_m8_c32_sg2_k32_b2_p1"); GgufParams pq{N, KK, N / 64, 0, 0};
    Dispatch dm{ps, {Xbf, W0, W1, Mt, Y}, bytes(pq), 5, MTLSizeMake(N / 64, 1, 1), MTLSizeMake(64, 1, 1)}, dc{ps, {Xbf, C0, C1, Cm, Y2}, bytes(pq), 5, MTLSizeMake(N / 64, 1, 1), MTLSizeMake(64, 1, 1)};
    auto timeS = [&](Dispatch d, id<MTLBuffer> y, int reps) { Dispatch dt{touch, {y}, {}, -1, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1)}; std::vector<double> ts;
      for (int r = 0; r < reps; ++r) { id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (auto *x : {&d, &dt}) { [enc setComputePipelineState:x->p]; for (size_t i = 0; i < x->bufs.size(); ++i) [enc setBuffer:x->bufs[i] offset:0 atIndex:i]; if (x->paramIndex >= 0) [enc setBytes:x->params.data() length:x->params.size() atIndex:x->paramIndex]; [enc dispatchThreadgroups:x->grid threadsPerThreadgroup:x->tg]; }
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; ts.push_back(cb.GPUEndTime - cb.GPUStartTime); } return ts; };
    const double gb = double(p0 + p1 + mb) / 1e9;
    auto report = [&](const char *label, std::vector<double> ts) { printf("%-34s", label); for (double t : ts) printf(" %.2f", t * 1e3); printf("  ms  (last: %.0f GB/s)\n", gb / ts.back() / 1e9 * 1e9); };
    report("mmap no-copy (cold then warm):", timeS(dm, Y, 6));
    report("Metal-allocated copy:", timeS(dc, Y2, 6));
    report("mmap no-copy again:", timeS(dm, Y, 4));
    const uint16_t *a = (const uint16_t *)Y.contents, *b2 = (const uint16_t *)Y2.contents; size_t diff = 0; for (size_t i = 0; i < (size_t)rows * N; ++i) diff += a[i] != b2[i]; printf("outputs identical: %s\n", diff ? "NO" : "yes");
  }
  return failures ? 1 : 0; } }

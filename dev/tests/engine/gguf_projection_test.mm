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
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <random>
#include <string>
#include <map>
#include <vector>
#include "metal/abi/Gguf.h"
#include "metal/abi/Linear.h"
static id<MTLDevice> dev; static id<MTLCommandQueue> queue; static std::mt19937 rng(42);
static uint16_t f2h(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float h2f(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (float)h; }
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
// ================= formats: native GGUF blocks -> llama.cpp-faithful reference values + tile repack (planes + meta)
enum Fmt { Q4K = GGUF_FMT_Q4K, IQ4XS = GGUF_FMT_IQ4XS, IQ4NL = GGUF_FMT_IQ4NL, Q5K = GGUF_FMT_Q5K, Q6K = GGUF_FMT_Q6K,
           Q3K = GGUF_FMT_Q3K, Q80 = GGUF_FMT_Q80, IQ3S = GGUF_FMT_IQ3S, FMT_COUNT = GGUF_FMT_COUNT };
static const char *fmtName(uint32_t f) { return kQuantFormats[f].name; }
// Optional external oracle: compile unmodified llama.cpp ggml-base and provide its
// dylib via SPLASH_GGML_ORACLE. The normal test remains self-contained/offline.
static void *ggmlOracle = nullptr;
static void verifyNativeReference(Fmt f, const std::vector<uint8_t> &native, std::vector<float> &values) {
  if (!ggmlOracle) return;
  static const char *symbols[FMT_COUNT] = {
    "dequantize_row_q4_K", "dequantize_row_iq4_xs", "dequantize_row_iq4_nl", "dequantize_row_q5_K",
    "dequantize_row_q6_K", "dequantize_row_q3_K", "dequantize_row_q8_0", "dequantize_row_iq3_s"};
  using Dequantize = void (*)(const void *, float *, int64_t);
  auto decode = reinterpret_cast<Dequantize>(dlsym(ggmlOracle, symbols[f]));
  if (!decode) throw std::runtime_error(dlerror());
  std::vector<float> official(values.size());
  decode(native.data(), official.data(), official.size());
  if (memcmp(official.data(), values.data(), values.size() * sizeof(float)))
    throw std::runtime_error(std::string("CPU reference differs from upstream GGML: ") + fmtName(f));
  values = std::move(official);
}

static const float kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const uint32_t iq3s_grid[512] = {
#include "iq3s_grid.inc"
};
static uint32_t rowBytes(Fmt f, uint32_t K) { const QuantFormat &i = kQuantFormats[f]; return K / i.block_elements * i.block_bytes; }
static uint64_t streamBytes(Fmt f, uint32_t N, uint32_t K) { const QuantFormat &i = kQuantFormats[f]; return uint64_t(N) * (K / 32) * (i.plane0_bytes + i.plane1_bytes) + uint64_t(N) * (K / 32 / i.meta_groups) * i.meta_bytes; }
static std::vector<uint8_t> makeNative(Fmt f, uint32_t N, uint32_t K) {
  const QuantFormat &fi = kQuantFormats[f]; const uint32_t rb = rowBytes(f, K); std::vector<uint8_t> v((size_t)N * rb); for (auto &b : v) b = (uint8_t)rng();
  std::uniform_real_distribution<float> dk(0.0005f, 0.004f), dx(0.00002f, 0.00015f), d6(0.00002f, 0.0001f), d3s(0.0001f, 0.0005f);
  for (uint32_t n = 0; n < N; ++n) { uint8_t *row = v.data() + (size_t)n * rb;
    for (uint32_t b = 0; b < K / fi.block_elements; ++b) { uint8_t *blk = row + b * fi.block_bytes; uint16_t d = 0, m = 0; uint32_t off = 0;
      switch (f) { case Q4K: case Q5K: d = f2h(dk(rng)); m = f2h(dk(rng)); memcpy(blk + 2, &m, 2); break; case IQ4XS: d = f2h(dx(rng)); break; case IQ4NL: case Q80: d = f2h(dk(rng)); break;
        case Q6K: d = f2h(d6(rng)); off = 208; break; case Q3K: d = f2h(dk(rng)); off = 108; break; default: d = f2h(d3s(rng)); break; }
      memcpy(blk + off, &d, 2); } }
  return v;
}
static void scale_min_k4(const uint8_t *sc, int j, uint8_t &s, uint8_t &m) { if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); } }
// Planes in chunk order (metal/abi/QuantFormat.h): 4-bit linear codes pair-interleaved in chunk words, every
// other field a little-endian bit string over the slots.
static void packBits(const uint8_t *slots, uint32_t bits, uint8_t *dst) {
  for (uint32_t w = 0, per = 32 / bits; w < bits; ++w) { uint32_t v = 0; for (uint32_t i = 0; i < per; ++i) v |= uint32_t(slots[w * per + i]) << (bits * i); memcpy(dst + 4 * w, &v, 4); } }
static void packPairs(const uint8_t *slots, uint8_t *dst) {
  for (uint32_t c = 0; c < 4; ++c) { uint32_t v = 0; for (uint32_t i = 0; i < 8; ++i) v |= uint32_t(slots[8 * c + i]) << ((i & 1) * 16 + 4 * (i >> 1)); memcpy(dst + 4 * c, &v, 4); } }
// reference values (llama.cpp dequantize_row_* semantics) + plane bytes for group g of one row
static void groupPack(Fmt f, const uint8_t *row, uint32_t g, float vals[32], uint8_t p0[32], uint8_t p1[8]) {
  const QuantFormat &fi = kQuantFormats[f]; const uint8_t *blk = row + (g * 32 / fi.block_elements) * fi.block_bytes; const uint32_t j = (g * 32 % fi.block_elements) / 32;
  uint8_t lo[32], hi[32];   // per slot: the (low) code and its high bits
  switch (f) {
    case Q4K: { const uint8_t *q = blk + 16 + (j / 2) * 32; int sh = (j % 2) * 4; uint16_t d16, m16; memcpy(&d16, blk, 2); memcpy(&m16, blk + 2, 2); uint8_t sc, mn; scale_min_k4(blk + 4, j, sc, mn);
      for (int k = 0; k < 32; ++k) { const uint8_t code = (q[k] >> sh) & 15; lo[quant_slot(k)] = code; vals[k] = h2f(d16) * sc * code - h2f(m16) * mn; }
      packPairs(lo, p0); return; }
    case IQ4XS: { const uint8_t *qs = blk + 8 + j * 16; uint16_t d16, shh; memcpy(&d16, blk, 2); memcpy(&shh, blk + 2, 2); const uint8_t *sl = blk + 4;
      int ls = ((sl[j / 2] >> 4 * (j % 2)) & 0xf) | (((shh >> 2 * j) & 3) << 4); float s = h2f(d16) * (ls - 32);
      for (int k = 0; k < 32; ++k) { const uint8_t code = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4); lo[quant_slot(k)] = code; vals[k] = s * kv_iq4nl[code]; }
      packBits(lo, 4, p0); return; }
    case IQ4NL: { const uint8_t *qs = blk + 2; uint16_t d16; memcpy(&d16, blk, 2); float s = h2f(d16);
      for (int k = 0; k < 32; ++k) { const uint8_t code = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4); lo[quant_slot(k)] = code; vals[k] = s * kv_iq4nl[code]; }
      packBits(lo, 4, p0); return; }
    case Q5K: { const uint8_t *qh = blk + 16, *ql = blk + 48 + (j / 2) * 32; int sh = (j % 2) * 4; uint16_t d16, m16; memcpy(&d16, blk, 2); memcpy(&m16, blk + 2, 2); uint8_t sc, mn; scale_min_k4(blk + 4, j, sc, mn);
      for (int k = 0; k < 32; ++k) { uint8_t l4 = (ql[k] >> sh) & 15, h1 = (qh[k] >> j) & 1; lo[quant_slot(k)] = l4; hi[quant_slot(k)] = h1; vals[k] = h2f(d16) * sc * (l4 + 16 * h1) - h2f(m16) * mn; }
      packPairs(lo, p0); packBits(hi, 1, p1); return; }
    case Q6K: { const uint32_t n = j / 4, r = j % 4; const uint8_t *ql = blk + 64 * n, *qh = blk + 128 + 32 * n; const int8_t *sc = (const int8_t *)(blk + 192) + 8 * n; uint16_t d16; memcpy(&d16, blk + 208, 2); float d = h2f(d16);
      for (int k = 0; k < 32; ++k) { uint8_t l4 = (ql[k + 32 * (r & 1)] >> (4 * (r >> 1))) & 15, h2 = (qh[k] >> (2 * r)) & 3; int q6 = (l4 | (h2 << 4)) - 32; lo[quant_slot(k)] = l4; hi[quant_slot(k)] = h2; vals[k] = d * sc[2 * r + k / 16] * q6; }
      packPairs(lo, p0); packBits(hi, 2, p1); return; }
    case Q3K: { const uint32_t n = j / 4, jj = j % 4; const uint8_t *hm = blk, *q = blk + 32 + 32 * n; uint32_t aux[4]; memcpy(aux, blk + 96, 12); uint32_t tmp = aux[2];
      const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
      aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4); aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
      aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4); aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
      const int8_t *scales = (const int8_t *)aux; uint16_t d16; memcpy(&d16, blk + 108, 2); float d = h2f(d16);
      for (int k = 0; k < 32; ++k) { int c2 = (q[k] >> (2 * jj)) & 3, hb1 = (hm[k] >> j) & 1; lo[quant_slot(k)] = uint8_t(c2); hi[quant_slot(k)] = uint8_t(hb1); vals[k] = d * (scales[2 * j + k / 16] - 32) * (c2 - (hb1 ? 0 : 4)); }
      packBits(lo, 2, p0); packBits(hi, 1, p1); return; }
    case Q80: { uint16_t d16; memcpy(&d16, blk, 2); const int8_t *qs = (const int8_t *)(blk + 2); for (int k = 0; k < 32; ++k) { vals[k] = h2f(d16) * qs[k]; lo[quant_slot(k)] = uint8_t(qs[k]); }
      packBits(lo, 8, p0); return; }
    default: { const uint8_t *qs = blk + 2 + 8 * j, *qh = blk + 66, *signs = blk + 74 + 4 * j, *scales = blk + 106; uint16_t d16; memcpy(&d16, blk, 2);
      const uint32_t sc = (scales[j / 2] >> (4 * (j % 2))) & 0xf; const float db = h2f(d16) * (1 + 2 * sc);
      for (int l = 0; l < 4; ++l) { const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[2 * l] | ((qh[j] << (8 - 2 * l)) & 256))), *g2 = (const uint8_t *)(iq3s_grid + (qs[2 * l + 1] | ((qh[j] << (7 - 2 * l)) & 256)));
        for (int k = 0; k < 4; ++k) { vals[8 * l + k] = db * g1[k] * ((signs[l] & (1 << k)) ? -1.f : 1.f); vals[8 * l + 4 + k] = db * g2[k] * ((signs[l] & (1 << (4 + k))) ? -1.f : 1.f); } }
      // word c: grid indices of elements 4c.. and 16+4c.. (qs[c], qs[4 + c]), chunk c's signs, their ninth bits, the scale
      for (int k = 0; k < 32; ++k) hi[quant_slot(k)] = (signs[k / 8] >> (k % 8)) & 1;
      uint8_t sign[4]; packBits(hi, 1, sign);
      for (int c = 0; c < 4; ++c) { const uint32_t v = qs[c] | uint32_t(qs[4 + c]) << 8 | uint32_t(sign[c]) << 16 | ((qh[j] >> c) & 1u) << 24 | ((qh[j] >> (4 + c)) & 1u) << 25 | sc << 26; memcpy(p0 + 4 * c, &v, 4); }
      return; }
  }
}
static void metaPack(Fmt f, const uint8_t *row, uint32_t unit, uint8_t *dst) {
  const QuantFormat &fi = kQuantFormats[f]; const uint8_t *blk = row + (f == IQ4NL || f == Q80 ? unit * fi.block_bytes : unit * fi.block_bytes);
  switch (f) { case Q4K: case Q5K: memcpy(dst, blk, 16); break; case IQ4XS: memcpy(dst, blk, 8); break; case IQ4NL: case Q80: case IQ3S: memcpy(dst, blk, 2); break;
    case Q6K: memcpy(dst, blk + 192, 16); memcpy(dst + 16, blk + 208, 2); dst[18] = dst[19] = 0; break;
    case Q3K: memcpy(dst, blk + 108, 2); dst[2] = dst[3] = 0; memcpy(dst + 4, blk + 96, 12); break; default: break; }
}
struct Packed { std::vector<uint8_t> w0, w1, meta; };
static Packed repack(Fmt f, const std::vector<uint8_t> &native, uint32_t N, uint32_t K, std::vector<float> *Wf, std::vector<float> *Ws = nullptr) {
  const QuantFormat &fi = kQuantFormats[f]; const uint32_t G = K / 32, rb = rowBytes(f, K), units = G / fi.meta_groups;
  Packed p; p.w0.assign((size_t)N * G * fi.plane0_bytes, 0); p.w1.assign(fi.plane1_bytes ? (size_t)N * G * fi.plane1_bytes : 16, 0); p.meta.assign((size_t)N * units * fi.meta_bytes, 0); if (Wf) Wf->assign((size_t)N * K, 0.f); if (Ws) Ws->assign((size_t)N * K, 0.f);
  for (uint32_t n = 0; n < N; ++n) { const uint8_t *row = native.data() + (size_t)n * rb;
    for (uint32_t g = 0; g < G; ++g) { float vals[32]; uint8_t p0[32], p1[8]; groupPack(f, row, g, vals, p0, p1);
      memcpy(p.w0.data() + quant_tile_index(n, g, G) * fi.plane0_bytes, p0, fi.plane0_bytes); if (fi.plane1_bytes) memcpy(p.w1.data() + quant_tile_index(n, g, G) * fi.plane1_bytes, p1, fi.plane1_bytes);
      if (Wf) for (int k = 0; k < 32; ++k) (*Wf)[(size_t)n * K + g * 32 + k] = vals[k];
      if (Ws) for (int k = 0; k < 32; ++k) (*Ws)[(size_t)n * K + g * 32 + k] = h2f(f2h(vals[k])); }
    for (uint32_t u = 0; u < units; ++u) metaPack(f, row, u, p.meta.data() + quant_tile_index(n, u, units) * fi.meta_bytes); }
  return p;
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

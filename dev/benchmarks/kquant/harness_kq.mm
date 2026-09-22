// M5 harness: splash native kernels under splash's Apple10 dispatch policy vs Q4_K kernels (staged fp16 + direct uint4b),
// all real Qwen3.8-27B projection shapes, decode M=8/16/24/32, prefill rows 17..2048. `validate` mode checks vs fp64.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <map>
#include <vector>
struct Q4KParams { uint32_t output_size, input_size, persistent_groups; };
struct KQParams { uint32_t output_size, input_size, persistent_groups, out_stride, out_offset; };
struct ReduceParams { uint32_t splits, rows, cols, out_stride, out_offset, epilogue; };
struct Q4KHeader { uint16_t d, dmin; uint8_t scales[12]; };
struct Q4PrefillParams { uint32_t output_size, input_size; };
static id<MTLDevice> dev; static id<MTLCommandQueue> queue; static std::mt19937 rng(42); static uint32_t CORES = 16;
static uint16_t f2h(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float h2f(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (float)h; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x8000) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static void scale_min(const Q4KHeader &h, int j, float &s, float &m) {
  const uint8_t *q = h.scales; uint8_t sc, mn;
  if (j < 4) { sc = q[j] & 63; mn = q[j + 4] & 63; } else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
  s = h2f(h.d) * sc; m = -h2f(h.dmin) * mn;
}
static id<MTLLibrary> compile(const char *path) {
  std::ifstream f(path); std::stringstream ss; ss << f.rdbuf();
  MTLCompileOptions *opt = [MTLCompileOptions new]; opt.languageVersion = (MTLLanguageVersion)((4 << 16) + 0); opt.mathMode = MTLMathModeFast;
  NSError *err = nil; id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:ss.str().c_str()] options:opt error:&err];
  if (!lib) { std::cerr << "COMPILE FAILED " << path << ": " << err.localizedDescription.UTF8String << "\n"; exit(1); }
  return lib;
}
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
// ---- splash Apple10 decode policy (ops/Linear.cpp)
struct Policy { uint32_t full, wave, many; };
static uint32_t maxCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) { uint32_t worst = 0; for (uint32_t c = 0; c < cores; ++c) { uint32_t load = 0; for (uint32_t g = c; g < groups; g += cores) load += (tiles - g + groups - 1) / groups; worst = std::max(worst, load); } return worst; }
static uint32_t decodeGroups(uint32_t tiles, uint32_t cores, Policy p) {
  const uint32_t wave = p.wave * cores; if (tiles <= p.full * cores || tiles >= p.many * cores) return tiles;
  const uint32_t twoTile = (tiles + 1) / 2; if (twoTile > wave) return wave;
  const uint32_t balanced = (tiles + cores - 1) / cores; uint32_t groups = std::max(twoTile, p.full * cores * 3 / 4);
  while (maxCoreTiles(tiles, groups, cores) != balanced) ++groups; return groups; }
struct SplashCfg { std::string pipe; uint32_t tileN, groups, threads; };
static SplashCfg splashDecode(uint32_t n, uint32_t k, uint32_t lanes) {
  const uint32_t t128 = n / 128, t256 = n / 256;
  if (lanes == 1) {
    if (t256 >= 8 * CORES) return {"decode_linear_q4_n256_paired_sg4", 256, std::min(t256, 4 * CORES), 128};
    return {"decode_linear_q4_n128_paired", 128, decodeGroups(t128, CORES, {4, 4, 12}), 256};
  }
  if (lanes == 3) {
    if (t128 <= CORES && k >= 4096) return {"decode_linear_q4_n128_m24", 128, t128, 256};
    return {"decode_linear_q4_n128_m24_sg4", 128, decodeGroups(t128, CORES, {8, 8, 24}), 128};
  }
  if (lanes >= 3 && t256 >= 2 * CORES) return {"decode_linear_q4_n256_m32", 256, decodeGroups(t256, CORES, {3, 3, 8}), 256};
  if (lanes == 2) return {"decode_linear_q4_n128_m16", 128, decodeGroups(t128, CORES, {5, 4, 12}), 256};
  return {"decode_linear_q4_n128_m32", 128, decodeGroups(t128, CORES, {4, 4, 12}), 256};
}

// ================= formats: native GGUF blocks -> llama.cpp-faithful reference values + tile repack (planes + meta)
enum Fmt { Q4K = 0, IQ4XS, IQ4NL, Q5K, Q6K, Q3K, Q80, IQ3S, FMT_COUNT };
static const char *fmtName[FMT_COUNT] = {"q4k", "iq4xs", "iq4nl", "q5k", "q6k", "q3k", "q80", "iq3s"};
struct FmtInfo { uint32_t blockK, blockBytes, p0, p1, metaBytes, metaGroups; };
static FmtInfo finfo(Fmt f) {
  switch (f) { case Q4K: return {256, 144, 16, 0, 16, 8}; case IQ4XS: return {256, 136, 16, 0, 8, 8}; case IQ4NL: return {32, 18, 16, 0, 2, 1};
    case Q5K: return {256, 176, 16, 4, 16, 8}; case Q6K: return {256, 210, 16, 8, 20, 8}; case Q3K: return {256, 110, 8, 4, 16, 8};
    case Q80: return {32, 34, 32, 0, 2, 1}; default: return {256, 110, 16, 0, 2, 8}; } }
static const float kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const uint32_t iq3s_grid[512] = {
#include "iq3s_grid.inc"
};
static uint32_t rowBytes(Fmt f, uint32_t K) { const FmtInfo i = finfo(f); return K / i.blockK * i.blockBytes; }
static uint64_t streamBytes(Fmt f, uint32_t N, uint32_t K) { const FmtInfo i = finfo(f); return uint64_t(N) * (K / 32) * (i.p0 + i.p1) + uint64_t(N) * (K / 32 / i.metaGroups) * i.metaBytes; }
static std::vector<uint8_t> makeNative(Fmt f, uint32_t N, uint32_t K) {
  const FmtInfo fi = finfo(f); const uint32_t rb = rowBytes(f, K); std::vector<uint8_t> v((size_t)N * rb); for (auto &b : v) b = (uint8_t)rng();
  std::uniform_real_distribution<float> dk(0.0005f, 0.004f), dx(0.00002f, 0.00015f), d6(0.00002f, 0.0001f), d3s(0.0001f, 0.0005f);
  for (uint32_t n = 0; n < N; ++n) { uint8_t *row = v.data() + (size_t)n * rb;
    for (uint32_t b = 0; b < K / fi.blockK; ++b) { uint8_t *blk = row + b * fi.blockBytes; uint16_t d = 0, m = 0; uint32_t off = 0;
      switch (f) { case Q4K: case Q5K: d = f2h(dk(rng)); m = f2h(dk(rng)); memcpy(blk + 2, &m, 2); break; case IQ4XS: d = f2h(dx(rng)); break; case IQ4NL: case Q80: d = f2h(dk(rng)); break;
        case Q6K: d = f2h(d6(rng)); off = 208; break; case Q3K: d = f2h(dk(rng)); off = 108; break; default: d = f2h(d3s(rng)); break; }
      memcpy(blk + off, &d, 2); } }
  return v;
}
static void scale_min_k4(const uint8_t *sc, int j, uint8_t &s, uint8_t &m) { if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); } }
static uint32_t interleave8(const uint8_t *codes) { uint32_t v = 0; for (int k = 0; k < 8; ++k) { uint32_t lane = (k & 1) ? 16 + 4 * (k / 2) : 4 * (k / 2); v |= uint32_t(codes[k] & 15) << lane; } return v; }
static uint32_t natural8(const uint8_t *codes) { uint32_t v = 0; for (int k = 0; k < 8; ++k) v |= uint32_t(codes[k] & 15) << (4 * k); return v; }
// reference values (llama.cpp dequantize_row_* semantics) + plane bytes for group g of one row
static void groupPack(Fmt f, bool interleave, const uint8_t *row, uint32_t g, float vals[32], uint8_t p0[32], uint8_t p1[8]) {
  const FmtInfo fi = finfo(f); const uint8_t *blk = row + (g * 32 / fi.blockK) * fi.blockBytes; const uint32_t j = (g * 32 % fi.blockK) / 32;
  uint8_t codes[32]; uint32_t w[4] = {0, 0, 0, 0}; uint32_t h[2] = {0, 0};
  switch (f) {
    case Q4K: { const uint8_t *q = blk + 16 + (j / 2) * 32; int sh = (j % 2) * 4; uint16_t d16, m16; memcpy(&d16, blk, 2); memcpy(&m16, blk + 2, 2); uint8_t sc, mn; scale_min_k4(blk + 4, j, sc, mn);
      for (int k = 0; k < 32; ++k) { codes[k] = (q[k] >> sh) & 15; vals[k] = h2f(d16) * sc * codes[k] - h2f(m16) * mn; }
      for (int k = 0; k < 4; ++k) w[k] = interleave8(codes + 8 * k); memcpy(p0, w, 16); return; }
    case IQ4XS: { const uint8_t *qs = blk + 8 + j * 16; uint16_t d16, shh; memcpy(&d16, blk, 2); memcpy(&shh, blk + 2, 2); const uint8_t *sl = blk + 4;
      int ls = ((sl[j / 2] >> 4 * (j % 2)) & 0xf) | (((shh >> 2 * j) & 3) << 4); float s = h2f(d16) * (ls - 32);
      for (int k = 0; k < 32; ++k) { codes[k] = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4); vals[k] = s * kv_iq4nl[codes[k]]; }
      for (int k = 0; k < 4; ++k) w[k] = interleave ? interleave8(codes + 8 * k) : natural8(codes + 8 * k); memcpy(p0, w, 16); return; }
    case IQ4NL: { const uint8_t *qs = blk + 2; uint16_t d16; memcpy(&d16, blk, 2); float s = h2f(d16);
      for (int k = 0; k < 32; ++k) { codes[k] = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4); vals[k] = s * kv_iq4nl[codes[k]]; }
      for (int k = 0; k < 4; ++k) w[k] = interleave ? interleave8(codes + 8 * k) : natural8(codes + 8 * k); memcpy(p0, w, 16); return; }
    case Q5K: { const uint8_t *qh = blk + 16, *ql = blk + 48 + (j / 2) * 32; int sh = (j % 2) * 4; uint16_t d16, m16; memcpy(&d16, blk, 2); memcpy(&m16, blk + 2, 2); uint8_t sc, mn; scale_min_k4(blk + 4, j, sc, mn);
      uint32_t hb = 0; for (int k = 0; k < 32; ++k) { uint8_t lo = (ql[k] >> sh) & 15, hi = (qh[k] >> j) & 1; codes[k] = lo; vals[k] = h2f(d16) * sc * (lo + 16 * hi) - h2f(m16) * mn;
        int ww = k / 8, p = (k % 8) / 2; if (k & 1) hb |= uint32_t(hi) << (16 + 4 * ww + p); else hb |= uint32_t(hi) << (4 * ww + p); }
      for (int k = 0; k < 4; ++k) w[k] = interleave8(codes + 8 * k); memcpy(p0, w, 16); memcpy(p1, &hb, 4); return; }
    case Q6K: { const uint32_t n = j / 4, r = j % 4; const uint8_t *ql = blk + 64 * n, *qh = blk + 128 + 32 * n; const int8_t *sc = (const int8_t *)(blk + 192) + 8 * n; uint16_t d16; memcpy(&d16, blk + 208, 2); float d = h2f(d16);
      for (int k = 0; k < 32; ++k) { uint8_t lo = (ql[k + 32 * (r & 1)] >> (4 * (r >> 1))) & 15, hi = (qh[k] >> (2 * r)) & 3; int q6 = (lo | (hi << 4)) - 32; codes[k] = lo; vals[k] = d * sc[2 * r + k / 16] * q6;
        int hh = k / 16, ip = (k % 16) / 2; if (k & 1) h[hh] |= uint32_t(hi) << (16 + 2 * ip); else h[hh] |= uint32_t(hi) << (2 * ip); }
      for (int k = 0; k < 4; ++k) w[k] = interleave8(codes + 8 * k); memcpy(p0, w, 16); memcpy(p1, h, 8); return; }
    case Q3K: { const uint32_t n = j / 4, jj = j % 4; const uint8_t *hm = blk, *q = blk + 32 + 32 * n; uint32_t aux[4]; memcpy(aux, blk + 96, 12); uint32_t tmp = aux[2];
      const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
      aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4); aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
      aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4); aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
      const int8_t *scales = (const int8_t *)aux; uint16_t d16; memcpy(&d16, blk + 108, 2); float d = h2f(d16); uint32_t hb = 0;
      for (int k = 0; k < 32; ++k) { int c2 = (q[k] >> (2 * jj)) & 3, hb1 = (hm[k] >> j) & 1; vals[k] = d * (scales[2 * j + k / 16] - 32) * (c2 - (hb1 ? 0 : 4));
        int hh = k / 16, ip = (k % 16) / 2, i = k / 2; if (k & 1) { h[hh] |= uint32_t(c2) << (16 + 2 * ip); hb |= uint32_t(hb1) << (16 + i); } else { h[hh] |= uint32_t(c2) << (2 * ip); hb |= uint32_t(hb1) << i; } }
      memcpy(p0, h, 8); memcpy(p1, &hb, 4); return; }
    case Q80: { uint16_t d16; memcpy(&d16, blk, 2); const int8_t *qs = (const int8_t *)(blk + 2); for (int k = 0; k < 32; ++k) vals[k] = h2f(d16) * qs[k]; memcpy(p0, qs, 32); return; }
    default: { const uint8_t *qs = blk + 2 + 8 * j, *qh = blk + 66, *signs = blk + 74 + 4 * j, *scales = blk + 106; uint16_t d16; memcpy(&d16, blk, 2);
      const uint32_t sc = (scales[j / 2] >> (4 * (j % 2))) & 0xf; const float db = h2f(d16) * (1 + 2 * sc);
      for (int l = 0; l < 4; ++l) { const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[2 * l] | ((qh[j] << (8 - 2 * l)) & 256))), *g2 = (const uint8_t *)(iq3s_grid + (qs[2 * l + 1] | ((qh[j] << (7 - 2 * l)) & 256)));
        for (int k = 0; k < 4; ++k) { vals[8 * l + k] = db * g1[k] * ((signs[l] & (1 << k)) ? -1.f : 1.f); vals[8 * l + 4 + k] = db * g2[k] * ((signs[l] & (1 << (4 + k))) ? -1.f : 1.f); } }
      memcpy(p0, qs, 8); memcpy(p0 + 8, signs, 4); p0[12] = qh[j]; p0[13] = (uint8_t)sc; p0[14] = p0[15] = 0; return; }
  }
}
static void metaPack(Fmt f, const uint8_t *row, uint32_t unit, uint8_t *dst) {
  const FmtInfo fi = finfo(f); const uint8_t *blk = row + (f == IQ4NL || f == Q80 ? unit * fi.blockBytes : unit * fi.blockBytes);
  switch (f) { case Q4K: case Q5K: memcpy(dst, blk, 16); break; case IQ4XS: memcpy(dst, blk, 8); break; case IQ4NL: case Q80: case IQ3S: memcpy(dst, blk, 2); break;
    case Q6K: memcpy(dst, blk + 192, 16); memcpy(dst + 16, blk + 208, 2); dst[18] = dst[19] = 0; break;
    case Q3K: memcpy(dst, blk + 108, 2); dst[2] = dst[3] = 0; memcpy(dst + 4, blk + 96, 12); break; default: break; }
}
struct Packed { std::vector<uint8_t> w0, w1, meta; };
static Packed repack(Fmt f, bool interleave, const std::vector<uint8_t> &native, uint32_t N, uint32_t K, std::vector<float> *Wf) {
  const FmtInfo fi = finfo(f); const uint32_t G = K / 32, rb = rowBytes(f, K), units = G / fi.metaGroups;
  Packed p; p.w0.assign((size_t)N * G * fi.p0, 0); p.w1.assign(fi.p1 ? (size_t)N * G * fi.p1 : 16, 0); p.meta.assign((size_t)N * units * fi.metaBytes, 0); if (Wf) Wf->assign((size_t)N * K, 0.f);
  for (uint32_t n = 0; n < N; ++n) { const uint8_t *row = native.data() + (size_t)n * rb; const uint32_t tile = n / 256, c = n % 256;
    for (uint32_t g = 0; g < G; ++g) { float vals[32]; uint8_t p0[32], p1[8]; groupPack(f, interleave, row, g, vals, p0, p1);
      memcpy(p.w0.data() + (((size_t)tile * G + g) * 256 + c) * fi.p0, p0, fi.p0); if (fi.p1) memcpy(p.w1.data() + (((size_t)tile * G + g) * 256 + c) * fi.p1, p1, fi.p1);
      if (Wf) for (int k = 0; k < 32; ++k) (*Wf)[(size_t)n * K + g * 32 + k] = vals[k]; }
    for (uint32_t u = 0; u < units; ++u) metaPack(f, row, u, p.meta.data() + (((size_t)tile * units + u) * 256 + c) * fi.metaBytes); }
  return p;
}
static id<MTLBuffer> upload(const std::vector<uint8_t> &v) { id<MTLBuffer> b = mkbuf(v.size()); memcpy(b.contents, v.data(), v.size()); return b; }
static std::vector<uint8_t> readFile(const char *path) { std::ifstream f(path, std::ios::binary); std::vector<uint8_t> v((std::istreambuf_iterator<char>(f)), {}); return v; }
struct Variant { const char *name; Fmt fmt; bool interleave; };
static const std::vector<Variant> kVariants = {{"q4k", Q4K, true}, {"iq4xs", IQ4XS, true}, {"iq4xsB", IQ4XS, false}, {"iq4xsT", IQ4XS, false}, {"iq4xsP", IQ4XS, true}, {"iq4nl", IQ4NL, true}, {"iq4nlB", IQ4NL, false},
                                               {"q5k", Q5K, true}, {"q6k", Q6K, true}, {"q3k", Q3K, true}, {"q80", Q80, true}, {"iq3s", IQ3S, true}};
struct SGc { int C, S, KS, B, P; };
static const std::vector<SGc> kSg = {{32,4,32,2,1},{32,2,32,2,1},{32,1,32,2,1},{32,4,32,1,1},{32,2,32,2,2},{16,8,64,1,1},{64,1,32,2,1},{32,2,32,1,1},{32,1,32,1,1},{32,2,32,1,2},{32,1,32,1,2},{32,2,64,1,1},{32,2,64,1,2},{32,1,64,1,2},{32,1,32,2,2},{16,4,32,1,2},{16,2,64,1,2},{32,1,32,2,3}};
static const std::vector<SGc> kSk = {{32,4,32,2,1},{32,2,32,2,1},{16,8,64,1,1}};
static const std::vector<SGc> sgSerial = {{32,1,32,1,1},{32,2,32,1,1},{32,2,32,2,1},{32,1,32,2,1},{32,4,32,1,1}};
struct PFc { int R, S, N, KS; };
static const std::vector<PFc> kPf = {{32,4,64,64},{32,4,64,32},{16,8,64,64},{32,8,64,64},{16,4,128,32},{16,4,64,64}};

int main(int argc, char **argv) { @autoreleasepool {
  dev = MTLCreateSystemDefaultDevice(); queue = [dev newCommandQueue];
  const std::string mode = argc > 3 ? argv[3] : "full";
  id<MTLLibrary> libK = compile(argv[2]);
  if (mode == "dump") {   // harness_kq x x dump <variant> <raw> <N> <K> <out_prefix>
    const std::string vn = argv[4]; const Variant *var = nullptr; for (auto &v : kVariants) if (vn == v.name) var = &v; if (!var) return 1;
    const uint32_t N = atoi(argv[6]), K = atoi(argv[7]); std::vector<uint8_t> raw = readFile(argv[5]); Packed pk = repack(var->fmt, var->interleave, raw, N, K, nullptr);
    for (auto &kv : std::vector<std::pair<const char *, std::vector<uint8_t> *>>{{"plane0", &pk.w0}, {"plane1", &pk.w1}, {"meta", &pk.meta}}) { std::ofstream o(std::string(argv[8]) + "." + kv.first, std::ios::binary); o.write((const char *)kv.second->data(), kv.second->size()); }
    printf("dumped %zu %zu %zu\n", pk.w0.size(), pk.w1.size(), pk.meta.size()); return 0;
  }
  if (mode == "real") {   // harness_kq <splash> <kq> real <variant> <raw> <ref.f32> <N> <K>
    const std::string vn = argv[4]; const Variant *var = nullptr; for (auto &v : kVariants) if (vn == v.name) var = &v; if (!var) { fprintf(stderr, "unknown variant\n"); return 1; }
    const uint32_t N = atoi(argv[7]), K = atoi(argv[8]); std::vector<uint8_t> raw = readFile(argv[5]), refb = readFile(argv[6]);
    if (raw.size() != (size_t)N * rowBytes(var->fmt, K) || refb.size() != (size_t)N * K * 4) { fprintf(stderr, "size mismatch raw=%zu want %zu ref=%zu want %zu\n", raw.size(), (size_t)N * rowBytes(var->fmt, K), refb.size(), (size_t)N * K * 4); return 1; }
    const float *ref = (const float *)refb.data(); std::vector<float> Wf; Packed pk = repack(var->fmt, var->interleave, raw, N, K, &Wf);
    double maxd = 0, sumd = 0, sumabs = 0; size_t am = 0; for (size_t i = 0; i < Wf.size(); ++i) { double d = std::fabs((double)Wf[i] - ref[i]); if (d > maxd) { maxd = d; am = i; } sumd += d; sumabs += std::fabs(ref[i]); }
    printf("# %s %ux%u: repack/decode vs gguf-py: max|diff|=%.3e at [%zu][%zu] (ref %.6f mine %.6f) mean|diff|=%.3e mean|ref|=%.4e\n", vn.c_str(), N, K, maxd, am / K, am % K, ref[am], Wf[am], sumd / Wf.size(), sumabs / Wf.size());
    for (size_t i = 0; i < Wf.size(); ++i) Wf[i] = ref[i];
    id<MTLBuffer> W0 = upload(pk.w0), W1 = upload(pk.w1), Mt = upload(pk.meta);
    auto inputs = [&](uint32_t rows) { id<MTLBuffer> Xbf = mkbuf(uint64_t(rows) * K * 2), X16 = mkbuf(uint64_t(rows) * K * 2); uint16_t *a = (uint16_t *)Xbf.contents, *b = (uint16_t *)X16.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(rows) * K; ++i) { float v = bf2f(f2bf(d(rng))); a[i] = f2bf(v); b[i] = f2h(v); } return std::make_pair(Xbf, X16); };
    auto check = [&](id<MTLBuffer> X16, id<MTLBuffer> Y, uint32_t rows) { const uint16_t *x = (const uint16_t *)X16.contents, *y = (const uint16_t *)Y.contents; double maxerr = 0, sumrel = 0, sumref = 0; size_t cnt = 0;
      for (uint32_t r = 0; r < rows; r += std::max(1u, rows / 8)) for (uint32_t n = 0; n < N; n += 3) { double acc = 0; for (uint32_t k = 0; k < K; ++k) acc += (double)h2f(x[(size_t)r * K + k]) * Wf[(size_t)n * K + k];
        double got = bf2f(y[(size_t)r * N + n]); maxerr = std::max(maxerr, std::fabs(got - acc)); sumrel += std::fabs(got - acc) / (std::fabs(acc) + 1e-3); sumref += std::fabs(acc); ++cnt; }
      printf("maxabs=%.2e meanrel=%.1e mean|ref|=%.2e\n", maxerr, sumrel / cnt, sumref / cnt); };
    struct KC { std::string name; uint32_t rows, tileN, tileM, threads; bool prefill, bf; };
    for (auto &k : std::vector<KC>{{"sg_" + vn + "_m8_c32_sg4_k32_b2_p1", 8, 128, 0, 128, false, false}, {"sga_" + vn + "_m8_c32_sg4_k32_b2_p1", 8, 128, 0, 128, false, true}, {"sga_" + vn + "_m32_c32_sg2_k32_b2_p1", 32, 64, 0, 64, false, true},
                                    {"pf_" + vn + "_r32_sg4_n64_k64_p1", 128, 64, 128, 128, true, false}, {"pfa_" + vn + "_r32_sg4_n64_k32_p1", 128, 64, 128, 128, true, true}}) {
      id<MTLComputePipelineState> ps = pso(libK, k.name); if (!ps) continue; auto [Xbf, X16] = inputs(k.rows); id<MTLBuffer> Y = mkbuf(uint64_t(k.rows) * N * 2); KQParams pq{N, K, N / k.tileN, 0, 0};
      Dispatch d{ps, {k.bf ? Xbf : X16, W0, Mt, Y, {}, W1}, bytes(pq), 4, k.prefill ? MTLSizeMake(k.rows / k.tileM, N / k.tileN, 1) : MTLSizeMake(N / k.tileN, 1, 1), MTLSizeMake(k.threads, 1, 1)};
      d.bufs[4] = nil; runOnce({d}, 1); printf("%-44s rows=%-4u ", k.name.c_str(), k.rows); check(X16, Y, k.rows); }
    return 0;
  }
  id<MTLLibrary> libS = compile(argv[1]);
  const bool serial = mode == "serial";
  const bool quick = mode == "quick" || mode == "extra" || serial, validate = mode == "validate";
  id<MTLComputePipelineState> touch = pso(libK, "kq_touch");
  auto timeSerial = [&](Dispatch d, id<MTLBuffer> Y, int iters) {   // each GEMM followed by a dependent touch of Y
    Dispatch t{touch, {Y}, {}, -1, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1)};
    auto run = [&](int n) { id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      for (int it = 0; it < n; ++it) for (auto *x : {&d, &t}) { [enc setComputePipelineState:x->p]; for (size_t i = 0; i < x->bufs.size(); ++i) if (x->bufs[i]) [enc setBuffer:x->bufs[i] offset:0 atIndex:i]; if (x->paramIndex >= 0) [enc setBytes:x->params.data() length:x->params.size() atIndex:x->paramIndex]; [enc dispatchThreadgroups:x->grid threadsPerThreadgroup:x->tg]; }
      [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; return (cb.GPUEndTime - cb.GPUStartTime) / n; };
    run(2); double best = 1e9; for (int r = 0; r < 5; ++r) best = std::min(best, run(iters)); return best; };
  printf("# device %s apple10=%d mode=%s\n", dev.name.UTF8String, (int)[dev supportsFamily:(MTLGPUFamily)1010], mode.c_str());
  struct Shape { const char *label; uint32_t out, in; bool prefill; std::vector<Fmt> fmts; };
  const std::vector<Fmt> all = {Q4K, IQ4XS, IQ4NL, Q5K, Q6K, Q3K, Q80, IQ3S};
  std::vector<Shape> shapes = {{"gate/up 17408x5120", 17408, 5120, true, {Q4K, IQ4XS, IQ4NL, Q5K, Q3K, IQ3S}}, {"down 5120x17408", 5120, 17408, true, {Q4K, IQ4XS, IQ4NL, Q5K, Q6K, Q3K, IQ3S}},
                               {"attn_qkv 10240x5120", 10240, 5120, true, {Q4K, IQ4XS, IQ4NL, Q5K}}, {"attn_kv 1024x5120", 1024, 5120, true, {Q4K, Q5K, Q6K, Q80}}, {"out 5120x6144", 5120, 6144, true, {IQ4XS, Q5K, Q6K}},
                               {"ab 256x5120", 256, 5120, true, {Q80}}, {"attn_gate 6144x5120", 6144, 5120, true, {Q4K, IQ4XS, Q5K}}, {"attn_q 12288x5120", 12288, 5120, true, {Q4K, IQ4XS, Q5K}}, {"lm_head 248320x5120", 248320, 5120, false, {Q6K}}};
  if (quick) shapes.resize(6);
  if (serial) shapes = {{"gate/up 17408x5120", 17408, 5120, false, {Q4K, IQ4XS}}, {"lm_head 248320x5120", 248320, 5120, false, {Q6K, Q4K}}};
  if (mode == "extra") shapes = {{"attn_gate 6144x5120", 6144, 5120, true, {Q4K, IQ4XS, Q5K}}, {"attn_q 12288x5120", 12288, 5120, true, {Q4K, IQ4XS, Q5K}},
                                 {"gate/up 17408x5120", 17408, 5120, true, {Q6K}}, {"out 5120x6144", 5120, 6144, true, {Q4K}}, {"lm_head 248320x5120", 248320, 5120, false, {Q6K}}};
  if (validate) shapes = {{"val 1024x1024", 1024, 1024, true, all}, {"val 768x5120", 768, 5120, true, all}};
  printf("| phase | rows | shape | fmt | kernel | groups | ms | GB/s | TFLOPS | ratio vs splash | check |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
  for (const Shape &s : shapes) {
    const uint64_t wbytes = uint64_t(s.out) * s.in / 2, sbytes = uint64_t(s.out) * (s.in / 64) * 2;
    id<MTLBuffer> Ws = mkbuf(wbytes), Ss = mkbuf(sbytes), Bs = mkbuf(sbytes);
    { uint8_t *w = (uint8_t *)Ws.contents; for (uint64_t i = 0; i < wbytes; ++i) w[i] = (uint8_t)rng(); uint16_t *sc = (uint16_t *)Ss.contents, *bi = (uint16_t *)Bs.contents; std::uniform_real_distribution<float> pr(-0.02f, 0.02f); for (uint64_t i = 0; i < sbytes / 2; ++i) { sc[i] = f2bf(pr(rng)); bi[i] = f2bf(pr(rng)); } }
    const bool doCheck = uint64_t(s.out) * s.in <= 200000000ull;
    auto inputs = [&](uint32_t rows, uint32_t fill) { id<MTLBuffer> Xbf = mkbuf(uint64_t(rows) * s.in * 2), X16 = mkbuf(uint64_t(rows) * s.in * 2); uint16_t *a = (uint16_t *)Xbf.contents, *b = (uint16_t *)X16.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(fill) * s.in; ++i) { float v = bf2f(f2bf(d(rng))); a[i] = f2bf(v); b[i] = f2h(v); } return std::make_pair(Xbf, X16); };
    auto row = [&](const char *phase, uint32_t rows, const char *fmt, const std::string &kname, uint32_t groups, double t, double ts, const std::string &chk, uint64_t kbytes) {
      printf("| %s | %u | %s | %s | %s | %u | %.3f | %.1f | %.2f | %s | %s |\n", phase, rows, s.label, fmt, kname.c_str(), groups, t * 1e3, kbytes / t / 1e9, 2.0 * rows * s.out * s.in / t / 1e12, ts == t ? "1.00" : ([&]{ char b[16]; snprintf(b, sizeof b, "%.2f", ts / t); return std::string(b); })().c_str(), chk.c_str()); fflush(stdout); };
    const int itD = s.out > 100000 ? 5 : 20;
    struct Dec { uint32_t M, storage; id<MTLBuffer> Xbf, X16, Y; double ts; };
    std::vector<Dec> dec;
    for (uint32_t lanes = 1; lanes <= (serial ? 1u : 4u); ++lanes) {
      const uint32_t M = 8 * lanes, storage = lanes == 3 ? 32 : M;
      auto [Xbf, X16] = inputs(storage, M); id<MTLBuffer> Y = mkbuf(uint64_t(storage) * s.out * 2);
      SplashCfg sc = splashDecode(s.out, s.in, lanes); Q4KParams ps{s.out, s.in, sc.groups};
      Dispatch ds{pso(libS, sc.pipe), {Xbf, Ws, Ss, Bs, Y}, bytes(ps), 5, MTLSizeMake(sc.groups, 1, 1), MTLSizeMake(sc.threads, 1, 1)};
      double ts = serial ? timeSerial(ds, Y, itD) : timeIt({ds}, itD); row("decode", M, "splash", std::string(sc.pipe) + (serial ? " [serial]" : ""), sc.groups, ts, ts, "", wbytes + 2 * sbytes);
      dec.push_back({M, storage, Xbf, X16, Y, ts});
    }
    std::vector<uint32_t> rowsList = validate ? std::vector<uint32_t>{17, 64, 128, 256} : quick ? std::vector<uint32_t>{17, 128, 512, 2048} : std::vector<uint32_t>{17, 32, 64, 128, 256, 512, 1024, 2048};
    struct Pre { uint32_t rows, storage; id<MTLBuffer> Xbf, X16, Y; double ts; };
    std::vector<Pre> pre;
    if (s.prefill) for (uint32_t rows : rowsList) {
      const uint32_t storage = rows <= 32 ? 32 : (rows + 127) / 128 * 128, sstorage = (rows + 31) / 32 * 32;
      auto [Xbf, X16] = inputs(storage, storage); id<MTLBuffer> Y = mkbuf(uint64_t(storage) * s.out * 2), sums = mkbuf(uint64_t(sstorage) * (s.in / 64) * 4);
      Q4PrefillParams pp{s.out, s.in}; const int itP = validate ? 2 : (storage >= 1024 ? 4 : 10);
      Dispatch sd{pso(libS, "prefill_linear_q4_sums32"), {Xbf, sums}, bytes(pp), 2, MTLSizeMake(sstorage / 32, 1, 1), MTLSizeMake(256, 1, 1)};
      Dispatch ps{pso(libS, "prefill_linear_q4_n128_sg4"), {Xbf, Ws, Ss, Bs, Y, sums}, bytes(pp), 6, MTLSizeMake(sstorage / 32, s.out / 128, 1), MTLSizeMake(128, 1, 1)};
      double ts = timeIt({sd, ps}, itP); row("prefill", rows, "splash", "prefill_linear_q4_n128_sg4(+sums32)", sstorage / 32 * (s.out / 128), ts, ts, "", wbytes + 2 * sbytes);
      pre.push_back({rows, storage, Xbf, X16, Y, ts});
    }
    id<MTLBuffer> partials = mkbuf(uint64_t(8) * 32 * s.out * 4);
    for (const Variant &var : kVariants) {
      if (std::find(s.fmts.begin(), s.fmts.end(), var.fmt) == s.fmts.end()) continue;
      const Fmt f = var.fmt; const char *vn = var.name;
      std::vector<float> Wf; std::vector<uint8_t> native = makeNative(f, s.out, s.in); Packed pk = repack(f, var.interleave, native, s.out, s.in, doCheck ? &Wf : nullptr);
      native.clear(); native.shrink_to_fit();
      id<MTLBuffer> W0 = upload(pk.w0), W1 = upload(pk.w1), Mt = upload(pk.meta); pk = Packed();
      const uint64_t kbytes = streamBytes(f, s.out, s.in);
      auto check = [&](id<MTLBuffer> X16, id<MTLBuffer> Y, uint32_t rows) -> std::string {
        if (!doCheck) return ""; const uint16_t *x = (const uint16_t *)X16.contents, *y = (const uint16_t *)Y.contents; double maxerr = 0, sumrel = 0; size_t cnt = 0;
        for (uint32_t r = 0; r < rows; r += std::max(1u, rows / 8)) for (uint32_t n = 0; n < s.out; n += 7) {
          double acc = 0; for (uint32_t k = 0; k < s.in; ++k) acc += (double)h2f(x[(size_t)r * s.in + k]) * Wf[(size_t)n * s.in + k];
          double got = bf2f(y[(size_t)r * s.out + n]); maxerr = std::max(maxerr, std::fabs(got - acc)); sumrel += std::fabs(got - acc) / (std::fabs(acc) + 1e-3); ++cnt; }
        char b[96]; snprintf(b, sizeof b, "maxabs=%.2e meanrel=%.1e", maxerr, sumrel / cnt); return b; };
      // activation variants: bf16 (production path) always; fp16 in validate mode only
      std::vector<std::pair<const char *, bool>> acts = validate ? std::vector<std::pair<const char *, bool>>{{"a", true}, {"", false}} : std::vector<std::pair<const char *, bool>>{{"a", true}};
      char b[160];
      for (auto &act : acts) { const char *as = act.first; const bool bf = act.second;
        for (Dec &d : dec) { const uint32_t R = d.storage; id<MTLBuffer> X = bf ? d.Xbf : d.X16;
          for (auto &c : (serial ? sgSerial : kSg)) { const uint32_t tileN = c.C * c.S; if (s.out % tileN) continue; snprintf(b, sizeof b, "sg%s_%s_m%u_c%d_sg%d_k%d_b%d_p%d", as, vn, R, c.C, c.S, c.KS, c.B, c.P);
            id<MTLComputePipelineState> ps = pso(libK, b); if (!ps) continue; KQParams pq{s.out, s.in, s.out / tileN, 0, 0};
            Dispatch dk{ps, {X, W0, Mt, d.Y, {}, W1}, bytes(pq), 4, MTLSizeMake(s.out / tileN, 1, 1), MTLSizeMake(c.S * 32, 1, 1)}; dk.bufs[4] = nil;
            double t = serial ? timeSerial(dk, d.Y, itD) : timeIt({dk}, validate ? 2 : itD); row("decode", d.M, vn, std::string(b) + (R != d.M ? " [rows32]" : ""), s.out / tileN, t, d.ts, serial ? "" : check(d.X16, d.Y, d.M), kbytes);
            if (serial && c.C == 32 && c.KS == 32 && c.P == 1 && (c.S <= 2) && s.out / tileN > 128) for (uint32_t g : {64u, 128u, 256u}) { KQParams pg{s.out, s.in, g, 0, 0};
              Dispatch dg{ps, {X, W0, Mt, d.Y, {}, W1}, bytes(pg), 4, MTLSizeMake(g, 1, 1), MTLSizeMake(c.S * 32, 1, 1)}; dg.bufs[4] = nil;
              double tg = timeSerial(dg, d.Y, itD); row("decode", d.M, vn, std::string(b) + " groups=" + std::to_string(g), g, tg, d.ts, "", kbytes); } }
          if (!serial && s.out / 128 <= 96) for (auto &c : kSk) for (uint32_t splits : {2u, 4u, 8u}) { const uint32_t tileN = c.C * c.S, steps = s.in / c.KS; if (s.out % tileN || steps % splits) continue;
            snprintf(b, sizeof b, "sgk%s_%s_m%u_c%d_sg%d_k%d_b%d_p%d", as, vn, R, c.C, c.S, c.KS, c.B, c.P); id<MTLComputePipelineState> ps = pso(libK, b); if (!ps) continue;
            KQParams pq{s.out, s.in, splits, 0, 0}; const uint32_t count = R * s.out; id<MTLComputePipelineState> rp = pso(libK, "splitk_reduce"); ReduceParams rpar{splits, R, s.out, 0, 0, 0};
            auto timeSK = [&](int iters) { double best = 1e9; for (int r = 0; r < 6; ++r) { id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (int it = 0; it < iters; ++it) { [enc setComputePipelineState:ps]; [enc setBuffer:X offset:0 atIndex:0]; [enc setBuffer:W0 offset:0 atIndex:1]; [enc setBuffer:Mt offset:0 atIndex:2]; [enc setBuffer:partials offset:0 atIndex:3]; [enc setBytes:&pq length:sizeof pq atIndex:4]; [enc setBuffer:W1 offset:0 atIndex:5];
                  [enc dispatchThreadgroups:MTLSizeMake(s.out / tileN, splits, 1) threadsPerThreadgroup:MTLSizeMake(c.S * 32, 1, 1)];
                  [enc setComputePipelineState:rp]; [enc setBuffer:partials offset:0 atIndex:0]; [enc setBuffer:d.Y offset:0 atIndex:1]; [enc setBytes:&rpar length:sizeof rpar atIndex:2]; [enc setBuffer:d.Y offset:0 atIndex:3]; [enc dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; }
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; if (r) best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / iters); } return best; };
            double t = timeSK(validate ? 2 : itD); row("decode", d.M, vn, std::string(b) + " splits=" + std::to_string(splits) + "(+reduce)" + (R != d.M ? " [rows32]" : ""), s.out / tileN * splits, t, d.ts, check(d.X16, d.Y, d.M), kbytes); }
        }
        for (Pre &p : pre) { const int itP = validate ? 2 : (p.storage >= 1024 ? 4 : 10); id<MTLBuffer> X = bf ? p.Xbf : p.X16;
          if (p.storage == 32) { for (auto &c : kSg) { const uint32_t tileN = c.C * c.S; if (s.out % tileN) continue; snprintf(b, sizeof b, "sg%s_%s_m32_c%d_sg%d_k%d_b%d_p%d", as, vn, c.C, c.S, c.KS, c.B, c.P);
              id<MTLComputePipelineState> ps = pso(libK, b); if (!ps) continue; KQParams pq{s.out, s.in, s.out / tileN, 0, 0};
              Dispatch dk{ps, {X, W0, Mt, p.Y, {}, W1}, bytes(pq), 4, MTLSizeMake(s.out / tileN, 1, 1), MTLSizeMake(c.S * 32, 1, 1)}; dk.bufs[4] = nil;
              double t = timeIt({dk}, itP); row("prefill", p.rows, vn, std::string(b) + " [rows32]", s.out / tileN, t, p.ts, check(p.X16, p.Y, p.rows), kbytes); } continue; }
          for (auto &c : kPf) { const uint32_t tileM = c.R * c.S; if (p.storage % tileM || s.out % c.N) continue;
            snprintf(b, sizeof b, "pf%s_%s_r%d_sg%d_n%d_k%d_p1", as, vn, c.R, c.S, c.N, c.KS); id<MTLComputePipelineState> ps = pso(libK, b); if (!ps) continue;
            KQParams pq{s.out, s.in, 0, 0, 0}; Dispatch dk{ps, {X, W0, Mt, p.Y, {}, W1}, bytes(pq), 4, MTLSizeMake(p.storage / tileM, s.out / c.N, 1), MTLSizeMake(c.S * 32, 1, 1)}; dk.bufs[4] = nil;
            double t = timeIt({dk}, itP); row("prefill", p.rows, vn, std::string(b) + (p.storage != (p.rows + 31) / 32 * 32 ? " [pad" + std::to_string(p.storage) + "]" : ""), p.storage / tileM * (s.out / c.N), t, p.ts, check(p.X16, p.Y, p.rows), kbytes); }
        }
      }
    }
  }
  return 0; } }

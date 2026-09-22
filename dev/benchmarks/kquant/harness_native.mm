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


struct NParams { uint32_t output_size, input_size, row_bytes, pad; };
// harness_native <splash.metal> <kq.metal> <kqn.metal> [validate]
int main(int argc, char **argv) { @autoreleasepool {
  dev = MTLCreateSystemDefaultDevice(); queue = [dev newCommandQueue];
  id<MTLLibrary> libS = compile(argv[1]), libK = compile(argv[2]), libN = compile(argv[3]);
  const bool validate = argc > 4 && std::string(argv[4]) == "validate";
  id<MTLComputePipelineState> touch = pso(libN, "kqn_touch");
  auto timeSerial = [&](Dispatch d, id<MTLBuffer> Y, int iters) { Dispatch t{touch, {Y}, {}, -1, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1)};
    auto run = [&](int n) { id<MTLCommandBuffer> cb = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      for (int it = 0; it < n; ++it) for (auto *x : {&d, &t}) { [enc setComputePipelineState:x->p]; for (size_t i = 0; i < x->bufs.size(); ++i) if (x->bufs[i]) [enc setBuffer:x->bufs[i] offset:0 atIndex:i]; if (x->paramIndex >= 0) [enc setBytes:x->params.data() length:x->params.size() atIndex:x->paramIndex]; [enc dispatchThreadgroups:x->grid threadsPerThreadgroup:x->tg]; }
      [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; return (cb.GPUEndTime - cb.GPUStartTime) / n; };
    run(2); double best = 1e9; for (int r = 0; r < 5; ++r) best = std::min(best, run(iters)); return best; };
  struct Shape { const char *label; uint32_t out, in; std::vector<Fmt> fmts; };
  std::vector<Shape> shapes = validate ? std::vector<Shape>{{"val 768x1024", 768, 1024, {Q4K, Q5K, IQ4XS, Q6K}}}
      : std::vector<Shape>{{"gate/up 17408x5120", 17408, 5120, {Q4K, IQ4XS, Q5K, Q6K}}, {"down 5120x17408", 5120, 17408, {Q4K, IQ4XS, Q5K}}, {"lm_head 248320x5120", 248320, 5120, {Q6K, Q4K}}};
  printf("| shape | fmt | rows | kernel | ms | GB/s | vs splash | check |\n|---|---|---|---|---|---|---|---|\n");
  for (const Shape &s : shapes) {
    const uint64_t wbytes = uint64_t(s.out) * s.in / 2, sbytes = uint64_t(s.out) * (s.in / 64) * 2;
    id<MTLBuffer> Ws = mkbuf(wbytes), Ss = mkbuf(sbytes), Bs = mkbuf(sbytes);
    { uint8_t *w = (uint8_t *)Ws.contents; for (uint64_t i = 0; i < wbytes; ++i) w[i] = (uint8_t)rng(); uint16_t *sc = (uint16_t *)Ss.contents, *bi = (uint16_t *)Bs.contents; for (uint64_t i = 0; i < sbytes / 2; ++i) { sc[i] = f2bf(0.01f); bi[i] = f2bf(0.f); } }
    const bool doCheck = validate;
    for (uint32_t M : {8u, 32u}) {
      id<MTLBuffer> Xbf = mkbuf(uint64_t(M) * s.in * 2), X16 = mkbuf(uint64_t(M) * s.in * 2), Y = mkbuf(uint64_t(M) * s.out * 2);
      { uint16_t *a = (uint16_t *)Xbf.contents, *b = (uint16_t *)X16.contents; std::uniform_real_distribution<float> d(-1.f, 1.f); for (uint64_t i = 0; i < uint64_t(M) * s.in; ++i) { float v = bf2f(f2bf(d(rng))); a[i] = f2bf(v); b[i] = f2h(v); } }
      SplashCfg sc = splashDecode(s.out, s.in, M / 8); Q4KParams ps{s.out, s.in, sc.groups};
      Dispatch ds{pso(libS, sc.pipe), {Xbf, Ws, Ss, Bs, Y}, bytes(ps), 5, MTLSizeMake(sc.groups, 1, 1), MTLSizeMake(sc.threads, 1, 1)};
      const int it = s.out > 100000 ? 5 : 20; const double ts = validate ? 1 : timeSerial(ds, Y, it);
      if (!validate) printf("| %s | splash | %u | %s | %.3f | %.0f | 1.00 | |\n", s.label, M, sc.pipe.c_str(), ts * 1e3, (wbytes + 2 * sbytes) / ts / 1e9);
      for (Fmt f : s.fmts) {
        std::vector<float> Wf; std::vector<uint8_t> native = makeNative(f, s.out, s.in);
        Packed pk = repack(f, f == IQ4XS ? false : true, native, s.out, s.in, doCheck ? &Wf : nullptr);
        id<MTLBuffer> Wn = upload(native), W0 = upload(pk.w0), W1 = upload(pk.w1), Mt = upload(pk.meta); pk = Packed();
        const uint64_t kbytes = streamBytes(f, s.out, s.in), rowb = rowBytes(f, s.in);
        auto check = [&](uint32_t rows) -> std::string { if (!doCheck) return ""; const uint16_t *x = (const uint16_t *)X16.contents, *y = (const uint16_t *)Y.contents; double maxerr = 0, sumrel = 0; size_t cnt = 0;
          for (uint32_t r = 0; r < rows; r += std::max(1u, rows / 8)) for (uint32_t n = 0; n < s.out; n += 5) { double acc = 0; for (uint32_t k = 0; k < s.in; ++k) acc += (double)h2f(x[(size_t)r * s.in + k]) * Wf[(size_t)n * s.in + k];
            double got = bf2f(y[(size_t)r * s.out + n]); maxerr = std::max(maxerr, std::fabs(got - acc)); sumrel += std::fabs(got - acc) / (std::fabs(acc) + 1e-3); ++cnt; }
          char b[96]; snprintf(b, sizeof b, "maxabs=%.2e meanrel=%.1e", maxerr, sumrel / cnt); return b; };
        char nm[96]; snprintf(nm, sizeof nm, "nat_%s_m%u", fmtName[f], M); id<MTLComputePipelineState> pn = pso(libN, nm);
        if (pn) { NParams np{s.out, s.in, (uint32_t)rowb, 0}; Dispatch dn{pn, {Xbf, Wn, Y}, bytes(np), 3, MTLSizeMake(s.out / 64, 1, 1), MTLSizeMake(64, 1, 1)};
          if (validate) { runOnce({dn}, 1); printf("| %s | %s | %u | %s (native layout) | | | | %s |\n", s.label, fmtName[f], M, nm, check(M).c_str()); }
          else { double t = timeSerial(dn, Y, it); printf("| %s | %s | %u | %s (native layout) | %.3f | %.0f | %.2f | |\n", s.label, fmtName[f], M, nm, t * 1e3, kbytes / t / 1e9, ts / t); } }
        const char *rv = f == IQ4XS ? "iq4xsT" : fmtName[f]; snprintf(nm, sizeof nm, "sga_%s_m%u_c32_sg2_k32_b2_p1", rv, M); id<MTLComputePipelineState> pr = pso(libK, nm);
        if (pr) { KQParams pq{s.out, s.in, s.out / 64, 0, 0}; Dispatch dr{pr, {Xbf, W0, Mt, Y, {}, W1}, bytes(pq), 4, MTLSizeMake(s.out / 64, 1, 1), MTLSizeMake(64, 1, 1)}; dr.bufs[4] = nil;
          if (validate) { runOnce({dr}, 1); printf("| %s | %s | %u | %s (repacked) | | | | %s |\n", s.label, fmtName[f], M, nm, check(M).c_str()); }
          else { double t = timeSerial(dr, Y, it); printf("| %s | %s | %u | %s (repacked) | %.3f | %.0f | %.2f | |\n", s.label, fmtName[f], M, nm, t * 1e3, kbytes / t / 1e9, ts / t); } }
        fflush(stdout);
      }
    }
  }
  return 0; } }

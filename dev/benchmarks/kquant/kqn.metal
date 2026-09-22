// Native-layout staged decode kernels: read llama.cpp GGUF blocks in place (row-major [N][K/blockK][blockBytes]), no repack.
// One lane = one output column (a row of W); each step dequantizes two 32-groups of that row into a per-simdgroup fp16 stage
// and runs matmul2d. Layout facts: Q4_K 144 B (16-aligned), Q5_K 176 B (16-aligned), IQ4_XS 136 B (8-aligned), Q6_K 210 B (2-aligned).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;
struct NParams { uint output_size; uint input_size; uint row_bytes; uint pad; };
constant half kIQ4NL[16] = {-127.0h, -104.0h, -83.0h, -65.0h, -49.0h, -35.0h, -22.0h, -10.0h, 1.0h, 13.0h, 25.0h, 38.0h, 53.0h, 69.0h, 89.0h, 113.0h};

inline void k4_scale_min(uint4 hdr, ushort j, thread half2 &s2, thread half2 &m2) {
  const uchar4 q0 = as_type<uchar4>(hdr.y), q1 = as_type<uchar4>(hdr.z), q2 = as_type<uchar4>(hdr.w);
  const uchar q[12] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w, q2.x, q2.y, q2.z, q2.w};
  uchar sc, m;
  if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; } else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
  const half d = as_type<half>(ushort(hdr.x & 0xFFFF)), dmin = as_type<half>(ushort(hdr.x >> 16));
  s2 = half2(half(float(d) * float(sc))); m2 = half2(half(-float(dmin) * float(m)));
}
// natural byte order: word v = bytes k..k+3; low nibbles = sub-block A weights k..k+3, high nibbles = sub-block B weights k..k+3
inline void store_nat_affine(uint v, half2 sA, half2 mA, half2 sB, half2 mB, threadgroup half *dA, threadgroup half *dB) {
  const half2 k = half2(1024.0h);
  const half2 a02 = fma(as_type<half2>((v & 0x000F000Fu) | 0x64006400u) - k, sA, mA), a13 = fma(as_type<half2>(((v >> 8) & 0x000F000Fu) | 0x64006400u) - k, sA, mA);
  const half2 b02 = fma(as_type<half2>(((v >> 4) & 0x000F000Fu) | 0x64006400u) - k, sB, mB), b13 = fma(as_type<half2>(((v >> 12) & 0x000F000Fu) | 0x64006400u) - k, sB, mB);
  *((threadgroup half4 *)dA) = half4(a02.x, a13.x, a02.y, a13.y);
  *((threadgroup half4 *)dB) = half4(b02.x, b13.x, b02.y, b13.y);
}
// 5-bit: hb words hold the 5th bits already shifted to bit 4 of each byte
inline void store_nat_affine5(uint v, uint hbA, uint hbB, half2 sA, half2 mA, half2 sB, half2 mB, threadgroup half *dA, threadgroup half *dB) {
  const half2 k = half2(1024.0h);
  const uint a = (v & 0x0F0F0F0Fu) | hbA, b = ((v >> 4) & 0x0F0F0F0Fu) | hbB;
  const half2 a02 = fma(as_type<half2>((a & 0x001F001Fu) | 0x64006400u) - k, sA, mA), a13 = fma(as_type<half2>(((a >> 8) & 0x001F001Fu) | 0x64006400u) - k, sA, mA);
  const half2 b02 = fma(as_type<half2>((b & 0x001F001Fu) | 0x64006400u) - k, sB, mB), b13 = fma(as_type<half2>(((b >> 8) & 0x001F001Fu) | 0x64006400u) - k, sB, mB);
  *((threadgroup half4 *)dA) = half4(a02.x, a13.x, a02.y, a13.y);
  *((threadgroup half4 *)dB) = half4(b02.x, b13.x, b02.y, b13.y);
}

// ---- format policies over one block of one row. Each step = 64 weights = two 32-groups (A, B) with their K positions.
struct NQ4K {   // block_q4_K: half d, half dmin, uchar scales[12], uchar qs[128]; pair p: qs[32p..32p+32): low = sub-block 2p, high = 2p+1
  enum : uint { BlockBytes = 144, StepsPerBlock = 4 };
  struct Hdr { uint4 h; }; struct Pay { uint4 lo, hi; };
  static Hdr header(device uchar *blk) { return {*((device uint4 *)blk)}; }
  static Pay load(device uchar *blk, ushort p) { device uint4 *q = (device uint4 *)(blk + 16 + p * 32); return {q[0], q[1]}; }
  static void kpos(ushort p, thread uint &kA, thread uint &kB) { kA = p * 64; kB = p * 64 + 32; }
  static void dequant(Pay w, Hdr h, ushort p, threadgroup half *dA, threadgroup half *dB) {
    half2 sA, mA, sB, mB; k4_scale_min(h.h, 2 * p, sA, mA); k4_scale_min(h.h, 2 * p + 1, sB, mB);
    const uint words[8] = {w.lo.x, w.lo.y, w.lo.z, w.lo.w, w.hi.x, w.hi.y, w.hi.z, w.hi.w};
#pragma unroll
    for (ushort i = 0; i < 8; ++i) store_nat_affine(words[i], sA, mA, sB, mB, dA + 4 * i, dB + 4 * i);
  }
};
struct NQ5K {   // block_q5_K: dm 4, scales 12, qh[32] @16, qs[128] @48
  enum : uint { BlockBytes = 176, StepsPerBlock = 4 };
  struct Hdr { uint4 h; uint4 qh0, qh1; }; struct Pay { uint4 lo, hi; };
  static Hdr header(device uchar *blk) { device uint4 *q = (device uint4 *)blk; return {q[0], q[1], q[2]}; }
  static Pay load(device uchar *blk, ushort p) { device uint4 *q = (device uint4 *)(blk + 48 + p * 32); return {q[0], q[1]}; }
  static void kpos(ushort p, thread uint &kA, thread uint &kB) { kA = p * 64; kB = p * 64 + 32; }
  static void dequant(Pay w, Hdr h, ushort p, threadgroup half *dA, threadgroup half *dB) {
    half2 sA, mA, sB, mB; k4_scale_min(h.h, 2 * p, sA, mA); k4_scale_min(h.h, 2 * p + 1, sB, mB);
    const uint words[8] = {w.lo.x, w.lo.y, w.lo.z, w.lo.w, w.hi.x, w.hi.y, w.hi.z, w.hi.w};
    const uint qh[8] = {h.qh0.x, h.qh0.y, h.qh0.z, h.qh0.w, h.qh1.x, h.qh1.y, h.qh1.z, h.qh1.w};
#pragma unroll
    for (ushort i = 0; i < 8; ++i) {
      const uint hA = ((qh[i] >> (2 * p)) & 0x01010101u) << 4, hB = ((qh[i] >> (2 * p + 1)) & 0x01010101u) << 4;
      store_nat_affine5(words[i], hA, hB, sA, mA, sB, mB, dA + 4 * i, dB + 4 * i);
    }
  }
};
struct NIQ4XS {   // block_iq4_xs: half d, ushort scales_h, uchar scales_l[4], uchar qs[128]; sub-block j: qs[16j+b]: low = weight b, high = weight b+16
  enum : uint { BlockBytes = 136, StepsPerBlock = 4 };
  struct Hdr { uint2 h; }; struct Pay { uint2 a, b, c, d; };   // 8-byte aligned loads
  static Hdr header(device uchar *blk) { return {*((device uint2 *)blk)}; }
  static Pay load(device uchar *blk, ushort p) { device uint2 *q = (device uint2 *)(blk + 8 + p * 32); return {q[0], q[1], q[2], q[3]}; }
  static void kpos(ushort p, thread uint &kA, thread uint &kB) { kA = p * 64; kB = p * 64 + 32; }
  static half scale(Hdr h, ushort j) { const half d = as_type<half>(ushort(h.h.x & 0xFFFF)); const uint sh = h.h.x >> 16;
    const int ls = int((h.h.y >> (4 * j)) & 0xF) | int(((sh >> (2 * j)) & 3) << 4); return half(float(d) * float(ls - 32)); }
  static void sub(uint4 q, half2 s2, threadgroup half2 *tl, threadgroup half *dst) {   // 16 bytes of one sub-block -> 32 weights
    const uint words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (ushort i = 0; i < 4; ++i) { const uint v = words[i];
      const half2 p0 = tl[v & 255] * s2, p1 = tl[(v >> 8) & 255] * s2, p2 = tl[(v >> 16) & 255] * s2, p3 = tl[v >> 24] * s2;   // (weight b, weight b+16)
      *((threadgroup half4 *)(dst + 4 * i)) = half4(p0.x, p1.x, p2.x, p3.x);
      *((threadgroup half4 *)(dst + 16 + 4 * i)) = half4(p0.y, p1.y, p2.y, p3.y); }
  }
  static void dequant(Pay w, Hdr h, ushort p, threadgroup half2 *tl, threadgroup half *dA, threadgroup half *dB) {
    sub(uint4(w.a.x, w.a.y, w.b.x, w.b.y), half2(scale(h, 2 * p)), tl, dA);
    sub(uint4(w.c.x, w.c.y, w.d.x, w.d.y), half2(scale(h, 2 * p + 1)), tl, dB);
  }
};
struct NQ6K {   // block_q6_K: ql[128], qh[64], int8 scales[16], half d; only 2-byte aligned -> ushort loads
  // 128-half n: sb0 = ql[l]&0xF | (qh[l]&3)<<4, sb1 = ql[l+32]&0xF | (qh[l]>>2&3)<<4, sb2 = ql[l]>>4 | (qh[l]>>4&3)<<4, sb3 = ql[l+32]>>4 | (qh[l]>>6&3)<<4
  // step p (0..3): n = p>>1, pair = p&1 -> sub-blocks (2*pair, 2*pair+2) of half n, i.e. ql[64n + 32*pair ..+32) with qh[32n..32n+32)
  enum : uint { BlockBytes = 210, StepsPerBlock = 4 };
  struct Hdr { uint sc[4]; half d; }; struct Pay { uint ql[8], qh[8]; };
  static ushort ld16(device uchar *p) { return *((device ushort *)p); }
  static uint ld32(device uchar *p) { return uint(ld16(p)) | (uint(ld16(p + 2)) << 16); }
  static Hdr header(device uchar *blk) { Hdr h; for (ushort i = 0; i < 4; ++i) h.sc[i] = ld32(blk + 192 + 4 * i); h.d = as_type<half>(ld16(blk + 208)); return h; }
  static Pay load(device uchar *blk, ushort p) { Pay w; const uint n = p >> 1, pair = p & 1;
#pragma unroll
    for (ushort i = 0; i < 8; ++i) { w.ql[i] = ld32(blk + 64 * n + 32 * pair + 4 * i); w.qh[i] = ld32(blk + 128 + 32 * n + 4 * i); } return w; }
  static void kpos(ushort p, thread uint &kA, thread uint &kB) { const uint n = p >> 1, pair = p & 1; kA = 128 * n + 32 * pair; kB = kA + 64; }
  static void dequant(Pay w, Hdr h, ushort p, threadgroup half *dA, threadgroup half *dB) {
    const uint n = p >> 1, pair = p & 1; const float d = float(h.d);
    // scales: sub-block index within the block: sA = 8n + 2*pair (+ l/16), sB = 8n + 2*pair + 4 (+ l/16)
    const uint iA = 8 * n + 2 * pair, iB = iA + 4;
    const char4 cA = as_type<char4>(h.sc[iA >> 2]), cB = as_type<char4>(h.sc[iB >> 2]);
    const half2 sA0 = half2(half(d * float(cA[iA & 3]))), sA1 = half2(half(d * float(cA[(iA & 3) + 1])));
    const half2 sB0 = half2(half(d * float(cB[iB & 3]))), sB1 = half2(half(d * float(cB[(iB & 3) + 1])));
    const half2 k = half2(1056.0h);   // 1024 + 32
#pragma unroll
    for (ushort i = 0; i < 8; ++i) {
      const uint v = w.ql[i], q = w.qh[i];
      const uint a = (v & 0x0F0F0F0Fu) | (((q >> (2 * pair)) & 0x03030303u) << 4);          // sub-block 2*pair
      const uint b = ((v >> 4) & 0x0F0F0F0Fu) | (((q >> (2 * pair + 4)) & 0x03030303u) << 4); // sub-block 2*pair + 2
      const half2 sA = i < 4 ? sA0 : sA1, sB = i < 4 ? sB0 : sB1;
      const half2 a02 = (as_type<half2>((a & 0x003F003Fu) | 0x64006400u) - k) * sA, a13 = (as_type<half2>(((a >> 8) & 0x003F003Fu) | 0x64006400u) - k) * sA;
      const half2 b02 = (as_type<half2>((b & 0x003F003Fu) | 0x64006400u) - k) * sB, b13 = (as_type<half2>(((b >> 8) & 0x003F003Fu) | 0x64006400u) - k) * sB;
      *((threadgroup half4 *)(dA + 4 * i)) = half4(a02.x, a13.x, a02.y, a13.y);
      *((threadgroup half4 *)(dB + 4 * i)) = half4(b02.x, b13.x, b02.y, b13.y);
    }
  }
};

// ---- the tile: Rows x 32 columns per simdgroup, one lane per column, two K=32 matmuls per step (A and B groups may be non-adjacent)
template <class F, ushort Rows, bool Lut>
inline void nat_tile(device bfloat *input, device uchar *w, device bfloat *output, uint row_bytes, uint output_size, uint input_size,
                     uint origin, threadgroup half *stage /*32 x 64 halfs*/, threadgroup half2 *tl, uint simd_lane) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a0 = a.template slice<32, Rows>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bA(stage, dextents<int, 2>{32, 32}, array<int, 2>{1, 64});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bB(stage + 32, dextents<int, 2>{32, 32}, array<int, 2>{1, 64});
  auto sA = bA.slice<32, 32>(0, 0), sB = bB.slice<32, 32>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(sA), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  device uchar *row = w + ulong(origin + simd_lane) * row_bytes;
  const uint blocks = input_size / 256, steps = blocks * F::StepsPerBlock;
  typename F::Hdr hdr = F::header(row);
  typename F::Pay pay = F::load(row, 0);
  threadgroup half *my = stage + simd_lane * 64;
  for (uint step = 0; step < steps; ++step) {
    const uint blk = step / F::StepsPerBlock; const ushort p = step % F::StepsPerBlock;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (Lut) F::dequant(pay, hdr, p, tl, my, my + 32); else F::dequant(pay, hdr, p, my, my + 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (step + 1 < steps) {
      const uint nblk = (step + 1) / F::StepsPerBlock; const ushort np = (step + 1) % F::StepsPerBlock;
      if (nblk != blk) hdr = F::header(row + ulong(nblk) * F::BlockBytes);
      pay = F::load(row + ulong(nblk) * F::BlockBytes, np);
    }
    uint kA, kB; F::kpos(p, kA, kB);
    auto aA = a.template slice<32, Rows>(blk * 256 + kA, 0), aB = a.template slice<32, Rows>(blk * 256 + kB, 0);
    operation.run(aA, sA, acc); operation.run(aB, sB, acc);
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    output[ulong(index[1]) * output_size + origin + index[0]] = bfloat(acc[i]);
  }
}
#define NAT_K(F, f, R, LUT)                                                                                 \
  kernel void nat_##f##_m##R(device bfloat *input [[buffer(0)]], device uchar *w [[buffer(1)]], device bfloat *output [[buffer(2)]], \
                             constant NParams &p [[buffer(3)]], uint group [[threadgroup_position_in_grid]],   \
                             uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    threadgroup half stage[2 * 32 * 64]; threadgroup half2 tl[LUT ? 256 : 1];                              \
    if (LUT) { for (uint i = simd_group * 32 + simd_lane; i < 256; i += 64) tl[i] = half2(kIQ4NL[i & 15], kIQ4NL[i >> 4]); threadgroup_barrier(mem_flags::mem_threadgroup); } \
    nat_tile<F, R, LUT>(input, w, output, p.row_bytes, p.output_size, p.input_size, group * 64 + simd_group * 32, stage + simd_group * (32 * 64), tl, simd_lane); }
NAT_K(NQ4K, q4k, 8, false) NAT_K(NQ4K, q4k, 32, false)
NAT_K(NQ5K, q5k, 8, false) NAT_K(NQ5K, q5k, 32, false)
NAT_K(NIQ4XS, iq4xs, 8, true) NAT_K(NIQ4XS, iq4xs, 32, true)
NAT_K(NQ6K, q6k, 8, false) NAT_K(NQ6K, q6k, 32, false)
kernel void kqn_touch(device bfloat *y [[buffer(0)]], uint tid [[thread_position_in_grid]]) { if (tid == 0) y[0] = bfloat(float(y[0]) + 0.0f); }

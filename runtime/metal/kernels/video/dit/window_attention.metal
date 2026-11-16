// DiT window-softmax attention: int8 QK^T (SageAttention-style), fp16 P x fp16 V, driven by the key-range table.
//   dit_attention_key_mean_partial / dit_attention_key_mean   per-head key mean (exact for softmax)
//   dit_attention_quantize   fp16 head-major Q, K -> Hadamard(128) -> int8 Q (per row) and K - mean(K) (per 32-key
//                            block); row scales carry log2(e) / sqrt(128) and the 1/128 of the rotation
//   dit_window_attention     one threadgroup = 4 simdgroups x 16 query rows of a 64-row work item; each simdgroup
//                            reads the table record of its 16-row block (<= 3 union key ranges, <= 2 row segments)
//                            and walks the 32-key tiles of the union once in ascending order. Tiles fully inside one
//                            range of a non-mixed block run unmasked; other tiles get an element mask (the row's own
//                            ranges, the item's key clip): masked scores leave the max and their P is exactly 0.
//                            Items with a key clip write a partial (normalized fp16 O + (m, l)).
//   dit_window_combine       merges the key-chunk partials of the dense query groups
#include "metal/abi/video/AttentionABI.h"
#include "metal/kernels/video/common/quant.h"
#include "metal/kernels/video/common/window.h"

using namespace h3;

// ---------------------------------------------------------------------------------------------------- preparation
// grid (NCH, H), 128 threads: part[h][chunk][c] = sum over the chunk's rows < S of K[h][r][c]
kernel void dit_attention_key_mean_partial(device const half *K [[buffer(0)]], device float *part [[buffer(1)]],
                                           constant AttentionPrepParams &p [[buffer(2)]],
                                           uint2 tg [[threadgroup_position_in_grid]],
                                           ushort c [[thread_index_in_threadgroup]]) {
  const int h = tg.y, ch = tg.x;
  const int per = (p.S + p.nch - 1) / p.nch;
  const int r1 = min(p.S, (ch + 1) * per);
  device const half *k = K + ulong(h) * p.Spad * 128 + c;
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int r = ch * per;
  for (; r + 4 <= r1; r += 4) {
    s0 += float(k[ulong(r) * 128]);
    s1 += float(k[ulong(r + 1) * 128]);
    s2 += float(k[ulong(r + 2) * 128]);
    s3 += float(k[ulong(r + 3) * 128]);
  }
  for (; r < r1; ++r) s0 += float(k[ulong(r) * 128]);
  part[(ulong(h) * p.nch + ch) * 128 + c] = (s0 + s1) + (s2 + s3);
}

// grid (H), 128 threads: kmean[h][c] = sum_chunk part / S
kernel void dit_attention_key_mean(device const float *part [[buffer(0)]], device float *kmean [[buffer(1)]],
                                   constant AttentionPrepParams &p [[buffer(2)]], uint h [[threadgroup_position_in_grid]],
                                   ushort c [[thread_index_in_threadgroup]]) {
  float s = 0;
  for (int i = 0; i < p.nch; ++i) s += part[(ulong(h) * p.nch + i) * 128 + c];
  kmean[h * 128 + c] = s / float(p.S);
}

// grid (Spad / KB, H, 2), 256 threads (8 simdgroups); z = 0: Q rows, z = 1: K block. Lane holds 4 channels.
// sq[h][row] = amax_row / 127 * qk_scale_log2 / 128; sk[h][blk] = amax_blk / 127. Rows >= S are zeros.
kernel void dit_attention_quantize(device const half *Qh [[buffer(0)]], device const half *Kh [[buffer(1)]],
                                   device char *Qi [[buffer(2)]], device char *Ki [[buffer(3)]],
                                   device float *sq [[buffer(4)]], device float *sk [[buffer(5)]],
                                   device const float *kmean [[buffer(6)]], constant AttentionPrepParams &p [[buffer(7)]],
                                   uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  const int h = tg.y, blk = tg.x, isk = tg.z;
  const bool had = p.flags & 1;
  const int rows_per_sg = p.KB / 8;
  const ulong base = ulong(h) * p.Spad;
  float4 mu = isk ? *(device const float4 *)(kmean + h * 128 + 4 * lane) : float4(0);
  float4 v[8];
  float am = 0.0f;
  for (int j = 0; j < rows_per_sg; ++j) {
    const int row = blk * p.KB + sg * rows_per_sg + j;
    float4 x = 0.0f;
    if (row < p.S) {
      device const half *src = (isk ? Kh : Qh) + (base + row) * 128 + 4 * lane;
      x = float4(*(device const half4 *)src) - mu;
      if (had) x = wht128_lane4(x, lane);
    }
    v[j] = x;
    float4 a = fabs(x);
    float ra = max(max(a.x, a.y), max(a.z, a.w));
    if (!isk) {  // per-row Q scale
      ra = simd_max(ra);
      const float inv = ra > 0.0f ? 127.0f / ra : 0.0f;
      *(device char4 *)(Qi + (base + row) * 128 + 4 * lane) = char4(rint(clamp(x * inv, -127.0f, 127.0f)));
      if (lane == 0) sq[base + row] = ra / 127.0f * p.qk_scale_log2 * (had ? (1.0f / 128.0f) : 1.0f);
    }
    am = max(am, ra);
  }
  if (!isk) return;
  am = simd_max(am);
  if (lane == 0) red[sg] = am;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  am = red[0];
  for (int i = 1; i < 8; ++i) am = max(am, red[i]);
  const float inv = am > 0.0f ? 127.0f / am : 0.0f;
  for (int j = 0; j < rows_per_sg; ++j) {
    const int row = blk * p.KB + sg * rows_per_sg + j;
    *(device char4 *)(Ki + (base + row) * 128 + 4 * lane) = char4(rint(clamp(v[j] * inv, -127.0f, 127.0f)));
  }
  if (sg == 0 && lane == 0) sk[ulong(h) * (p.Spad / p.KB) + blk] = am / 127.0f;
}

// ---------------------------------------------------------------------------------------------------- window kernel
constant constexpr int kSentinel = -(1 << 30);

// Online softmax of one 16 x 32 int32 score tile with an element mask (valid bit i = slot i is an allowed key;
// full = all valid, no masking work). P = exp2 in fp16 of the log2-domain score (clamped at -24).
template <class ST, class PT, class OT>
inline void window_softmax(thread ST &sT, thread PT &pL, thread OT &oT, float aA, float aB, thread float &mA,
                           thread float &mB, thread float &lA, thread float &lB, uint valid, bool full) {
  if (!full) {
#pragma unroll
    for (ushort i = 0; i < kScoreSlots; ++i)
      if (!((valid >> i) & 1u)) sT[i] = kSentinel;
  }
  int xA = sT[0], xB = sT[4];
#pragma unroll
  for (ushort i = 1; i < kScoreSlots; ++i) {
    if (((i >> 2) & 1) == 0) xA = max(xA, int(sT[i]));
    else xB = max(xB, int(sT[i]));
  }
  xA = max(xA, simd_shuffle_xor(xA, ushort(1)));
  xA = max(xA, simd_shuffle_xor(xA, ushort(8)));
  xB = max(xB, simd_shuffle_xor(xB, ushort(1)));
  xB = max(xB, simd_shuffle_xor(xB, ushort(8)));
  const float tA = float(xA) * aA, tB = float(xB) * aB;
  if (tA > mA || tB > mB) {
    const float nA = max(mA, tA), nB = max(mB, tB);
    const float cA = exp2(mA - nA), cB = exp2(mB - nB);
    mA = nA;
    mB = nB;
    lA *= cA;
    lB *= cB;
#pragma unroll
    for (ushort i = 0; i < kOutSlots; ++i) oT[i] *= ((i >> 2) & 1) ? cB : cA;
  }
  const float bA = -mA, bB = -mB;
  half4 hA = 0.0h, hB = 0.0h;
#pragma unroll
  for (ushort i = 0; i < kScoreSlots; i += 4) {
    const bool b = (i >> 2) & 1;
    float4 s4 = float4(float(sT[i]), float(sT[i + 1]), float(sT[i + 2]), float(sT[i + 3]));
    half4 ph = exp2(half4(max(fma(s4, float4(b ? aB : aA), float4(b ? bB : bA)), -24.0f)));
    if (!full) {
      ph.x = ((valid >> i) & 1u) ? ph.x : 0.0h;
      ph.y = ((valid >> (i + 1)) & 1u) ? ph.y : 0.0h;
      ph.z = ((valid >> (i + 2)) & 1u) ? ph.z : 0.0h;
      ph.w = ((valid >> (i + 3)) & 1u) ? ph.w : 0.0h;
    }
    pL[i] = ph.x;
    pL[i + 1] = ph.y;
    pL[i + 2] = ph.z;
    pL[i + 3] = ph.w;
    if (b) hB += ph;
    else hA += ph;
  }
  lA += float(hA.x) + float(hA.y) + float(hA.z) + float(hA.w);
  lB += float(hB.x) + float(hB.y) + float(hB.z) + float(hB.w);
}

// grid: nitems * hb threadgroups (head-major: all items of head0, then head0 + 1, ...), 128 threads.
kernel void dit_window_attention(device uchar *Qb [[buffer(0)]], device uchar *Kb [[buffer(1)]],
                                 device half *Vb [[buffer(2)]], device bfloat *O [[buffer(3)]],
                                 device const float *sq [[buffer(4)]], device const float *sk [[buffer(5)]],
                                 constant WindowParams &p [[buffer(6)]], device const WindowItem *items [[buffer(7)]],
                                 device const int *tab [[buffer(8)]], device half *Op [[buffer(9)]],
                                 device float2 *ML [[buffer(10)]], uint tgx [[threadgroup_position_in_grid]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int D = kHeadDim, BQ = kRowsPerTile, BK = kKeysPerTile;
  const int lin = int(tgx), sgl = sg;
  const int ii = lin % p.nitems, hl = lin / p.nitems, h = p.head0 + hl;
  const WindowItem it = items[ii];
  const int q0 = it.qg * 64 + sgl * BQ;
  threadgroup_barrier(mem_flags::mem_threadgroup);  // simdgroups start the key loop together
  const int b16 = q0 / BQ;
  if (b16 >= p.nqb) return;
  device const int *rec = tab + (1 + b16) * H3_TABLE_RECORD;
  const int nU = rec[0];
  const bool mixed = rec[1] != 0;
  const ushort r0 = lane_r0(lane), cb = lane_cb(lane);
  const int rowA = q0 + r0, rowB = rowA + 8;
  const int sA = rowA < rec[2] ? 0 : 1, sB = rowB < rec[2] ? 0 : 1;
  const int nA = rec[3 + sA], nB = rec[3 + sB];
  device const int2 *RA = (device const int2 *)(rec + 14 + 6 * sA);
  device const int2 *RB = (device const int2 *)(rec + 14 + 6 * sB);
  const int2 a0 = RA[0], a1 = RA[1], a2 = RA[2], c0 = RB[0], c1 = RB[1], c2 = RB[2];
  const ulong hs = ulong(p.Spad) * D;
  auto Q = mk<int8_t>((device int8_t *)Qb + hs * h, D, p.Spad);
  auto K = mk<int8_t>((device int8_t *)Kb + hs * h, D, p.Spad);
  auto V = mk<half>(Vb + hs * h, D, p.Spad);
  auto mQ0 = Q.template slice<D, BQ>(0, q0);
  constexpr auto dqk = matmul2d_descriptor(BQ, BK, D, false, true, false);
  matmul2d<dqk, execution_simdgroup> opqk;
  constexpr auto dpv = matmul2d_descriptor(BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroup> oppv;
  auto qL0 = opqk.template get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  qL0.load(mQ0);
  auto pL0 = oppv.template get_left_input_cooperative_tensor<half, half, float>();
  auto V0 = V.template slice<D, BK>(0, 0);
  auto oT0 = oppv.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(pL0)>, decltype(V0), float>();
#pragma unroll
  for (ushort i = 0; i < kOutSlots; ++i) oT0[i] = 0.0f;
  const ulong sqo = ulong(h) * p.Spad + q0 + r0;
  const float qA0 = sq[sqo], qB0 = sq[sqo + 8];
  float mA0 = -INFINITY, mB0 = -INFINITY, lA0 = 0.0f, lB0 = 0.0f;
  int prev = -1;
  for (int u = 0; u < nU; ++u) {
    const int a = max(rec[8 + 2 * u], it.kbeg), e = min(rec[9 + 2 * u], it.kend);
    if (a >= e) continue;
    const int t0 = max(a / BK, prev + 1), t1 = (e - 1) / BK;
    for (int kt = t0; kt <= t1; ++kt) {
      const int k0 = kt * BK;
      auto mK = K.template slice<D, BK>(0, k0);
      auto mV = V.template slice<D, BK>(0, k0);
      auto s0 = opqk.template get_destination_cooperative_tensor<decltype(mQ0), decltype(mK), int>();
      const bool full = !mixed && a <= k0 && k0 + BK <= e;
      uint valid = 0xFFFFu;
      if (!full) {
        valid = 0;
#pragma unroll
        for (ushort i = 0; i < kScoreSlots; ++i) {
          const int col = k0 + 16 * (i >> 3) + cb + (i & 3);
          const bool inb = col >= it.kbeg && col < it.kend;
          const bool v = ((i >> 2) & 1) ? in3(col, nB, c0, c1, c2) : in3(col, nA, a0, a1, a2);
          valid |= uint(inb && v) << i;
        }
      }
      const float ks = sk[ulong(h) * p.nkb + kt];
      opqk.run(qL0, mK, s0);
      window_softmax(s0, pL0, oT0, qA0 * ks, qB0 * ks, mA0, mB0, lA0, lB0, valid, full);
      oppv.run(pL0, mV, oT0);
    }
    prev = max(prev, t1);
  }
  window_finish(oT0, mA0, mB0, lA0, lB0, h, hl, q0, it.slot, sgl * BQ, r0, cb, p, O, Op, ML);
}

// grid (ngroups * 8, hb), 256 threads (one row per simdgroup, lane = 4 columns). groups[g] = (qg, slot0, nch, 0).
kernel void dit_window_combine(device const half *Op [[buffer(0)]], device const float2 *ML [[buffer(1)]],
                               device bfloat *O [[buffer(2)]], constant WindowParams &p [[buffer(3)]],
                               device const int4 *groups [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  const int g = tg.x / 8, lr = (tg.x % 8) * 8 + sg, hl = tg.y, h = p.head0 + hl;
  const int4 G = groups[g];
  const int row = G.x * 64 + lr;
  if (row >= p.S) return;
  float M = -INFINITY;
  for (int c = 0; c < G.z; ++c) M = max(M, ML[(ulong(hl) * p.nslot + G.y + c) * 64 + lr].x);
  float4 acc = 0.0f;
  float L = 0.0f;
  for (int c = 0; c < G.z; ++c) {
    const ulong b = (ulong(hl) * p.nslot + G.y + c) * 64 + lr;
    const float2 ml = ML[b];
    if (!(ml.y > 0.0f)) continue;
    const float w = exp2(ml.x - M) * ml.y;
    L += w;
    acc += w * float4(*(device const half4 *)(Op + b * 128 + 4 * lane));
  }
  acc /= L;
  *(device vec<bfloat, 4> *)(O + ulong(row) * p.ldo + h * 128 + 4 * lane) = vec<bfloat, 4>(acc);
}

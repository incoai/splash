// Video Sparse Attention of the FastH3 variants (FastVideo VSA-H3, 64-token tiles; abi/video/VsaABI.h). Rows are in tile
// order inside the blocks (video/ops/VsaTable): video tiles [0, V) then the dense prefix tiles [V, T) (text, audio).
//   dit_vsa_permute          row gather dst[i] = src[index[i]] (packed order <-> tile order, bf16 residual rows)
//   dit_vsa_pool             per-tile fp32 means of Q, K, V (the post-RoPE fp16 tensors; QKFIX blocks add their fp32
//                            split dimensions back)
//   dit_vsa_scores           scores s = q_pool . k_pool / sqrt(128) of every (query tile, key tile) pair, fp32
//   dit_vsa_route            per (query tile, head): for video query tiles the top-k video key tiles (radix select on
//                            the fp32 scores, ties to the lower index) in ascending tile order; the score row is
//                            replaced by its softmax over all tiles
//   dit_vsa_compress         compressed output softmax(s) . v_pool for every query tile
//   dit_vsa_attention        fine stage, int8 QK^T (K - mean, per-32-key scales) + fp16 PV as dit_window_attention:
//                            one threadgroup = 4 simdgroups x 16 rows of one query tile; prefix tiles attend to every
//                            key, video tiles to the prefix rows plus their selected tiles (adjacent selected tiles
//                            merged into one key range); keys outside a range are masked per element
//   dit_vsa_attention_qkfix  the same with fp16 Q / K and the exact fp32 split-dimension term (QKFIX blocks)
//   dit_vsa_attention_dense, dit_vsa_attention_qkfix_dense
//                            the two for dense attention (flags bit1, every 64-row chunk over the keys [0, M)): the
//                            same arithmetic, operand shapes and key-block order, with the key loop a constant
//                            [0, M) range (no range walk; only the last 32-key block masked). Bit-identical output.
//   dit_producer_vsa         out-projection input: O + out_c[tile(row)] (.) gate -> Hadamard + int8 (K = 7168)
#include "metal/abi/video/LinearABI.h"
#include "metal/abi/video/QkFixABI.h"
#include "metal/abi/video/VsaABI.h"
#include "metal/kernels/video/common/quant.h"
#include "metal/kernels/video/common/window.h"

using namespace h3;

// ---------------------------------------------------------------------------------------------------- permute
// grid (rows), 224 threads; D multiple of 8
kernel void dit_vsa_permute(device const uint4 *src [[buffer(0)]], device uint4 *dst [[buffer(1)]],
                            device const int *index [[buffer(2)]], constant VsaPermuteParams &p [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                            ushort nt [[threads_per_threadgroup]]) {
  if (int(row) >= p.rows) return;
  const int n = p.D / 8;
  device const uint4 *s = src + ulong(index[row]) * n;
  device uint4 *d = dst + ulong(row) * n;
  for (int i = tid; i < n; i += nt) d[i] = s[i];
}

// ---------------------------------------------------------------------------------------------------- pool
// grid (T, H), 128 threads (one per dimension). PQ / PK / PV fp32 [H][Tp][128] (pad rows stay zero).
kernel void dit_vsa_pool(device const half *Q [[buffer(0)]], device const half *K [[buffer(1)]],
                         device const half *V [[buffer(2)]], device const float *QS [[buffer(3)]],
                         device const float *KS [[buffer(4)]], device const int *sdims [[buffer(5)]],
                         device const int2 *tiles [[buffer(6)]], device float *PQ [[buffer(7)]],
                         device float *PK [[buffer(8)]], device float *PV [[buffer(9)]],
                         constant VsaParams &p [[buffer(10)]], uint2 tg [[threadgroup_position_in_grid]],
                         ushort d [[thread_index_in_threadgroup]]) {
  const int j = tg.x, h = tg.y;
  const int2 t = tiles[j];
  const ulong r0 = ulong(h) * p.Spad + t.x;
  device const half *q = Q + r0 * 128 + d, *k = K + r0 * 128 + d, *v = V + r0 * 128 + d;
  float sq = 0.0f, sk = 0.0f, sv = 0.0f;
  for (int r = 0; r < t.y; ++r) {
    sq += float(q[r * 128]);
    sk += float(k[r * 128]);
    sv += float(v[r * 128]);
  }
  if (p.flags & 1) {  // QKFIX: the split dimensions are zero in the fp16 Q / K, exact in QS / KS
    const int4 sd = *(device const int4 *)sdims;
    for (int i = 0; i < H3_QKFIX_SPLIT; ++i) {
      if (sd[i] != int(d)) continue;
      float eq = 0.0f, ek = 0.0f;
      for (int r = 0; r < t.y; ++r) {
        eq += QS[(r0 + r) * H3_QKFIX_SPLIT + i];
        ek += KS[(r0 + r) * H3_QKFIX_SPLIT + i];
      }
      sq = eq;
      sk = ek;
    }
  }
  const float n = float(t.y);
  const ulong o = (ulong(h) * p.Tp + j) * 128 + d;
  PQ[o] = sq / n;
  PK[o] = sk / n;
  PV[o] = sv / n;
}

// ---------------------------------------------------------------------------------------------------- route
// Coarse stage in three kernels over the tile means (fp32, tiles padded to Tp = 32 n with zero rows):
//   dit_vsa_scores    S[h] = Q_pool[h] K_pool[h]^T / sqrt(128)  [Tp][Tp] (simdgroup 8 x 8 fp32 matrices)
//   dit_vsa_route     per (query tile, head): top-k of the video columns of its score row (video query tiles), then
//                     the row replaced by softmax over the T real tiles (zeros beyond T)
//   dit_vsa_compress  OC[h] = softmax(S[h]) V_pool[h]  [Tp][128]

// grid (Tp / 32, Tp / 32, H), 128 threads: simdgroup s computes rows [16 (s / 2), +16) x columns [16 (s % 2), +16) of
// the 32 x 32 block
kernel void dit_vsa_scores(device const float *PQ [[buffer(0)]], device const float *PK [[buffer(1)]],
                           device float *S [[buffer(2)]], constant VsaParams &p [[buffer(3)]],
                           uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
  const int Tp = p.Tp, h = tg.z;
  const int r0 = tg.y * 32 + (sg / 2) * 16, c0 = tg.x * 32 + (sg % 2) * 16;
  device const float *q = PQ + ulong(h) * Tp * 128, *k = PK + ulong(h) * Tp * 128;
  simdgroup_float8x8 acc[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int d = 0; d < 128; d += 8) {
    simdgroup_float8x8 a[2], b[2];
    for (int i = 0; i < 2; ++i) simdgroup_load(a[i], q + ulong(r0 + 8 * i) * 128 + d, 128);
    for (int j = 0; j < 2; ++j) simdgroup_load(b[j], k + ulong(c0 + 8 * j) * 128 + d, 128, ulong2(0, 0), true);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
  device float *s = S + ulong(h) * Tp * Tp;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      simdgroup_float8x8 o = acc[i][j];
      // scale on the way out (the reference divides the fp32 product by sqrt(128))
      thread auto &e = o.thread_elements();
      e[0] *= p.score_scale;
      e[1] *= p.score_scale;
      simdgroup_store(o, s + ulong(r0 + 8 * i) * Tp + c0 + 8 * j, Tp);
    }
}

inline uint vsa_key(float s) {
  const uint u = as_type<uint>(s);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// exclusive prefix sum of one uint per thread over the 256-thread group; total returned through *total
inline uint vsa_scan(uint v, threadgroup uint *part, ushort sg, ushort lane, thread uint &total) {
  const uint inc = simd_prefix_inclusive_sum(v);
  if (lane == 31) part[sg] = inc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint before = 0, all = 0;
  for (ushort i = 0; i < H3_VSA_ROUTE_THREADS / 32; ++i) {
    const uint x = part[i];
    before += i < sg ? x : 0u;
    all += x;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = all;
  return before + inc - v;
}

// grid (T, H), 256 threads. S fp32 [H][Tp][Tp] in (scores), out (softmax rows); SEL int32 [H][V][topk].
kernel void dit_vsa_route(device float *S [[buffer(0)]], device int *SEL [[buffer(1)]],
                          constant VsaParams &p [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float sc[H3_VSA_MAX_TILES];
  threadgroup float red[H3_VSA_ROUTE_THREADS / 32];
  threadgroup atomic_uint hist[256];
  threadgroup uint part[H3_VSA_ROUTE_THREADS / 32];
  threadgroup uint pick[2];
  const int qt = tg.x, h = tg.y, T = p.T;
  device float *row = S + (ulong(h) * p.Tp + qt) * p.Tp;
  float m = -INFINITY;
  for (int j = tid; j < T; j += H3_VSA_ROUTE_THREADS) {
    const float s = row[j];
    sc[j] = s;
    m = max(m, s);
  }
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red[0];
  for (ushort i = 1; i < H3_VSA_ROUTE_THREADS / 32; ++i) m = max(m, red[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (qt < p.V) {  // top-k of the video key tiles [0, V)
    const int V = p.V;
    device int *out = SEL + (ulong(h) * V + qt) * p.topk;
    if (p.topk >= V) {
      for (int j = tid; j < V; j += H3_VSA_ROUTE_THREADS) out[j] = j;
    } else {
      // MSB-first radix select of the k-th largest key, 8 bits per pass
      uint prefix = 0, mask = 0, rem = uint(p.topk);
      for (int shift = 24; shift >= 0; shift -= 8) {
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = tid; j < V; j += H3_VSA_ROUTE_THREADS) {
          const uint k = vsa_key(sc[j]);
          if ((k & mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // c(b) = candidates with digit >= b; the k-th largest has the digit b with c(b) >= rem > c(b + 1)
        const uint b = 255u - uint(tid);
        uint tot;
        const uint hb = atomic_load_explicit(&hist[b], memory_order_relaxed);
        const uint before = vsa_scan(hb, part, sg, lane, tot);  // digits > b
        if (before + hb >= rem && before < rem) {
          pick[0] = b;
          pick[1] = rem - before;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= pick[0] << shift;
        mask |= 255u << shift;
        rem = pick[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      // compaction in ascending tile order: keys > threshold, then the first `rem` keys equal to it
      const int cs = (V + H3_VSA_ROUTE_THREADS - 1) / H3_VSA_ROUTE_THREADS;
      const int j0 = min(V, int(tid) * cs), j1 = min(V, j0 + cs);
      uint gt = 0, eq = 0;
      for (int j = j0; j < j1; ++j) {
        const uint k = vsa_key(sc[j]);
        gt += k > prefix ? 1u : 0u;
        eq += k == prefix ? 1u : 0u;
      }
      uint tot;
      const uint eqBefore = vsa_scan(eq, part, sg, lane, tot);
      const uint take = eqBefore >= rem ? 0u : min(eq, rem - eqBefore);
      uint off = vsa_scan(gt + take, part, sg, lane, tot);
      uint seen = 0;
      for (int j = j0; j < j1; ++j) {
        const uint k = vsa_key(sc[j]);
        if (k > prefix || (k == prefix && seen++ < take)) out[off++] = j;
      }
    }
  }
  // every thread is done reading the scores of the top-k before the softmax overwrites them
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // softmax over the T real tiles, written back over the scores (pad columns zero)
  float l = 0.0f;
  for (int j = tid; j < T; j += H3_VSA_ROUTE_THREADS) {
    const float e = exp(sc[j] - m);
    sc[j] = e;
    l += e;
  }
  l = simd_sum(l);
  if (lane == 0) red[sg] = l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  l = 0.0f;
  for (ushort i = 0; i < H3_VSA_ROUTE_THREADS / 32; ++i) l += red[i];
  const float il = 1.0f / l;
  for (int j = tid; j < p.Tp; j += H3_VSA_ROUTE_THREADS) row[j] = j < T ? sc[j] * il : 0.0f;
}

// grid (Tp / 32, 128 / 32, H), 128 threads: OC[h][rows][cols] = P[h][rows] V_pool[h][:, cols], 32 x 32 per group
kernel void dit_vsa_compress(device const float *S [[buffer(0)]], device const float *PV [[buffer(1)]],
                             device float *OC [[buffer(2)]], constant VsaParams &p [[buffer(3)]],
                             uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]]) {
  const int Tp = p.Tp, h = tg.z;
  const int r0 = tg.x * 32 + (sg / 2) * 16, c0 = tg.y * 32 + (sg % 2) * 16;
  device const float *P = S + ulong(h) * Tp * Tp, *v = PV + ulong(h) * Tp * 128;
  simdgroup_float8x8 acc[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (int k = 0; k < Tp; k += 8) {
    simdgroup_float8x8 a[2], b[2];
    for (int i = 0; i < 2; ++i) simdgroup_load(a[i], P + ulong(r0 + 8 * i) * Tp + k, Tp);
    for (int j = 0; j < 2; ++j) simdgroup_load(b[j], v + ulong(k) * 128 + c0 + 8 * j, 128);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
  device float *o = OC + ulong(h) * Tp * 128;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) simdgroup_store(acc[i][j], o + ulong(r0 + 8 * i) * 128 + c0 + 8 * j, 128);
}

// ---------------------------------------------------------------------------------------------------- fine stage
constant constexpr int kVsaSentinel = -(1 << 30);

// dit_window_attention's online softmax of one 16 x 32 int32 score tile (valid bit i = slot i is an allowed key)
template <class ST, class PT, class OT>
inline void vsa_softmax(thread ST &sT, thread PT &pL, thread OT &oT, float aA, float aB, thread float &mA,
                        thread float &mB, thread float &lA, thread float &lB, uint valid, bool full) {
  if (!full) {
#pragma unroll
    for (ushort i = 0; i < kScoreSlots; ++i)
      if (!((valid >> i) & 1u)) sT[i] = kVsaSentinel;
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

// element mask of the 32-key block at k0 for the key range [a, e) (row independent)
inline uint vsa_mask(int k0, int a, int e, ushort cb) {
  uint valid = 0;
#pragma unroll
  for (ushort i = 0; i < kScoreSlots; ++i) {
    const int col = k0 + 16 * (i >> 3) + cb + (i & 3);
    valid |= uint(col >= a && col < e) << i;
  }
  return valid;
}

// Walks the key ranges of one item: prefix tiles -> [0, M); video tiles -> [Nv, M), then the selected tiles in
// ascending order, adjacent tiles merged. body(a, e) processes one range.
template <class Body>
inline void vsa_ranges(constant VsaParams &p, device const int2 *tiles, device const int *SEL, int h, int qt,
                       thread Body &body) {
  if (qt >= p.V || (p.flags & 2)) {  // prefix tiles, or every chunk of dense attention
    body(0, p.M);
    return;
  }
  body(p.Nv, p.M);
  device const int *sl = SEL + (ulong(h) * p.V + qt) * p.topk;
  int ra = -1, re = -1;
  for (int u = 0; u < p.topk; ++u) {
    const int2 t = tiles[sl[u]];
    if (t.x == re) {
      re += t.y;
      continue;
    }
    if (ra >= 0) body(ra, re);
    ra = t.x;
    re = t.x + t.y;
  }
  if (ra >= 0) body(ra, re);
}

// normalized bf16 store of the rows < rowEnd of one 16-row block
template <class OT>
inline void vsa_store(thread OT &oT, float lA, float lB, int h, int q0, int rowEnd, ushort r0, ushort cb,
                      device bfloat *O) {
  lA += simd_shuffle_xor(lA, ushort(1));
  lA += simd_shuffle_xor(lA, ushort(8));
  lB += simd_shuffle_xor(lB, ushort(1));
  lB += simd_shuffle_xor(lB, ushort(8));
  const float iA = lA > 0.0f ? 1.0f / lA : 0.0f, iB = lB > 0.0f ? 1.0f / lB : 0.0f;
  device bfloat *Oh = O + h * kHeadDim;
#pragma unroll
  for (ushort i = 0; i < kOutSlots; i += 4) {
    const bool b = (i >> 2) & 1;
    const int row = q0 + r0 + (b ? 8 : 0);
    const int col = 16 * (i >> 3) + cb;
    if (row >= rowEnd) continue;
    float4 v = float4(oT[i], oT[i + 1], oT[i + 2], oT[i + 3]) * (b ? iB : iA);
    *(device vec<bfloat, 4> *)(Oh + ulong(row) * H3_DIT_INNER + col) = vec<bfloat, 4>(v);
  }
}

// grid: nitems * hb threadgroups (all items of head0 first), 128 threads.
kernel void dit_vsa_attention(device uchar *Qb [[buffer(0)]], device uchar *Kb [[buffer(1)]],
                              device half *Vb [[buffer(2)]], device bfloat *O [[buffer(3)]],
                              device const float *sq [[buffer(4)]], device const float *sk [[buffer(5)]],
                              constant VsaParams &p [[buffer(6)]], device const int *items [[buffer(7)]],
                              device const int2 *tiles [[buffer(8)]], device const int *SEL [[buffer(9)]],
                              uint tgx [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int D = kHeadDim, BQ = kRowsPerTile, BK = kKeysPerTile;
  const int ii = int(tgx) % p.nitems, hl = int(tgx) / p.nitems, h = p.head0 + hl;
  const int qt = items[ii];
  const int2 tq = tiles[qt];
  const int q0 = tq.x + sg * BQ;
  threadgroup_barrier(mem_flags::mem_threadgroup);  // simdgroups start the key loop together
  if (sg * BQ >= tq.y) return;
  const ushort r0 = lane_r0(lane), cb = lane_cb(lane);
  const ulong hs = ulong(p.Spad) * D;
  auto Q = mk<int8_t>((device int8_t *)Qb + hs * h, D, p.Spad);
  auto K = mk<int8_t>((device int8_t *)Kb + hs * h, D, p.Spad);
  auto Vt = mk<half>(Vb + hs * h, D, p.Spad);
  auto mQ0 = Q.template slice<D, BQ>(0, q0);
  constexpr auto dqk = matmul2d_descriptor(BQ, BK, D, false, true, false);
  matmul2d<dqk, execution_simdgroup> opqk;
  constexpr auto dpv = matmul2d_descriptor(BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroup> oppv;
  auto qL0 = opqk.template get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  qL0.load(mQ0);
  auto pL0 = oppv.template get_left_input_cooperative_tensor<half, half, float>();
  auto V0 = Vt.template slice<D, BK>(0, 0);
  auto oT0 = oppv.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(pL0)>, decltype(V0), float>();
#pragma unroll
  for (ushort i = 0; i < kOutSlots; ++i) oT0[i] = 0.0f;
  const ulong sqo = ulong(h) * p.Spad + q0 + r0;
  const float qA0 = sq[sqo], qB0 = sq[sqo + 8];
  float mA0 = -INFINITY, mB0 = -INFINITY, lA0 = 0.0f, lB0 = 0.0f;
  auto body = [&](int a, int e) {
    for (int kt = a / BK; kt <= (e - 1) / BK; ++kt) {
      const int k0 = kt * BK;
      auto mK = K.template slice<D, BK>(0, k0);
      auto mV = Vt.template slice<D, BK>(0, k0);
      auto s0 = opqk.template get_destination_cooperative_tensor<decltype(mQ0), decltype(mK), int>();
      const bool full = a <= k0 && k0 + BK <= e;
      const uint valid = full ? 0xFFFFu : vsa_mask(k0, a, e, cb);
      const float ks = sk[ulong(h) * p.nkb + kt];
      opqk.run(qL0, mK, s0);
      vsa_softmax(s0, pL0, oT0, qA0 * ks, qB0 * ks, mA0, mB0, lA0, lB0, valid, full);
      oppv.run(pL0, mV, oT0);
    }
  };
  vsa_ranges(p, tiles, SEL, h, qt, body);
  vsa_store(oT0, lA0, lB0, h, q0, tq.x + tq.y, r0, cb, O);
}

// QKFIX blocks: fp16 Q / K [H][Spad][128] (split dims zeroed) + QS / KS fp32 [H][Spad][4] (dit_qkfix_window_attention)
kernel void dit_vsa_attention_qkfix(device half *Qb [[buffer(0)]], device half *Kb [[buffer(1)]],
                                    device half *Vb [[buffer(2)]], device bfloat *O [[buffer(3)]],
                                    device const float *QS [[buffer(4)]], device const float *KS [[buffer(5)]],
                                    constant VsaParams &p [[buffer(6)]], device const int *items [[buffer(7)]],
                                    device const int2 *tiles [[buffer(8)]], device const int *SEL [[buffer(9)]],
                                    uint tgx [[threadgroup_position_in_grid]],
                                    ushort sg [[simdgroup_index_in_threadgroup]],
                                    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int D = kHeadDim, BQ = kRowsPerTile, BK = kKeysPerTile, NS = H3_QKFIX_SPLIT;
  const int ii = int(tgx) % p.nitems, hl = int(tgx) / p.nitems, h = p.head0 + hl;
  const int qt = items[ii];
  const int2 tq = tiles[qt];
  const int q0 = tq.x + sg * BQ;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg * BQ >= tq.y) return;
  const ushort r0 = lane_r0(lane), cb = lane_cb(lane);
  const int rowA = q0 + r0, rowB = rowA + 8;
  const ulong hs = ulong(p.Spad) * D;
  auto Q = mk<half>(Qb + hs * h, D, p.Spad);
  auto K = mk<half>(Kb + hs * h, D, p.Spad);
  auto Vt = mk<half>(Vb + hs * h, D, p.Spad);
  auto mQ0 = Q.template slice<D, BQ>(0, q0);
  constexpr auto dqk = matmul2d_descriptor(BQ, BK, D, false, true, false);
  matmul2d<dqk, execution_simdgroup> opqk;
  constexpr auto dpv = matmul2d_descriptor(BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroup> oppv;
  auto qL0 = opqk.template get_left_input_cooperative_tensor<half, half, float>();
  qL0.load(mQ0);
  auto pL0 = oppv.template get_left_input_cooperative_tensor<half, half, float>();
  auto V0 = Vt.template slice<D, BK>(0, 0);
  auto oT0 = oppv.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(pL0)>, decltype(V0), float>();
#pragma unroll
  for (ushort i = 0; i < kOutSlots; ++i) oT0[i] = 0.0f;
  const float a = p.qk_scale;
  const bool split = (p.flags & 1) != 0;
  const ulong sbase = ulong(h) * p.Spad;
  const float4 qsA = split ? *(device const float4 *)(QS + (sbase + rowA) * NS) * a : float4(0.0f);
  const float4 qsB = split ? *(device const float4 *)(QS + (sbase + rowB) * NS) * a : float4(0.0f);
  float mA0 = -INFINITY, mB0 = -INFINITY, lA0 = 0.0f, lB0 = 0.0f;
  auto body = [&](int ra, int re) {
    for (int kt = ra / BK; kt <= (re - 1) / BK; ++kt) {
      const int k0 = kt * BK;
      auto mK = K.template slice<D, BK>(0, k0);
      auto mV = Vt.template slice<D, BK>(0, k0);
      auto s0 = opqk.template get_destination_cooperative_tensor<decltype(mQ0), decltype(mK), float>();
      const bool full = ra <= k0 && k0 + BK <= re;
      const uint valid = full ? 0xFFFFu : vsa_mask(k0, ra, re, cb);
      opqk.run(qL0, mK, s0);
      if (split) {
        const float4 ksl = *(device const float4 *)(KS + (sbase + k0 + lane) * NS);
#pragma unroll
        for (ushort u8 = 0; u8 < 8; ++u8) {
          const float4 ks = simd_shuffle(ksl, ushort(16 * (u8 >> 2) + cb + (u8 & 3)));
          const ushort iA = (u8 >> 2) * 8 + (u8 & 3);
          s0[iA] = fma(float(s0[iA]), a, dot(qsA, ks));
          s0[iA + 4] = fma(float(s0[iA + 4]), a, dot(qsB, ks));
        }
      } else {
#pragma unroll
        for (ushort i = 0; i < kScoreSlots; ++i) s0[i] = float(s0[i]) * a;
      }
      if (!full) {
#pragma unroll
        for (ushort i = 0; i < kScoreSlots; ++i)
          if (!((valid >> i) & 1u)) s0[i] = -INFINITY;
      }
      float xA = s0[0], xB = s0[4];
#pragma unroll
      for (ushort i = 1; i < kScoreSlots; ++i) {
        if (((i >> 2) & 1) == 0) xA = max(xA, float(s0[i]));
        else xB = max(xB, float(s0[i]));
      }
      xA = max(xA, simd_shuffle_xor(xA, ushort(1)));
      xA = max(xA, simd_shuffle_xor(xA, ushort(8)));
      xB = max(xB, simd_shuffle_xor(xB, ushort(1)));
      xB = max(xB, simd_shuffle_xor(xB, ushort(8)));
      if (xA > mA0 || xB > mB0) {
        const float nA_ = max(mA0, xA), nB_ = max(mB0, xB);
        const float cA = exp2(mA0 - nA_), cB = exp2(mB0 - nB_);
        mA0 = nA_;
        mB0 = nB_;
        lA0 *= cA;
        lB0 *= cB;
#pragma unroll
        for (ushort i = 0; i < kOutSlots; ++i) oT0[i] *= ((i >> 2) & 1) ? cB : cA;
      }
      half4 hA = 0.0h, hB = 0.0h;
#pragma unroll
      for (ushort i = 0; i < kScoreSlots; i += 4) {
        const bool b = (i >> 2) & 1;
        const float m = b ? mB0 : mA0;
        float4 t4 = float4(s0[i], s0[i + 1], s0[i + 2], s0[i + 3]);
        half4 ph = exp2(half4(max(t4 - m, -24.0f)));
        ph.x = ((valid >> i) & 1u) ? ph.x : 0.0h;
        ph.y = ((valid >> (i + 1)) & 1u) ? ph.y : 0.0h;
        ph.z = ((valid >> (i + 2)) & 1u) ? ph.z : 0.0h;
        ph.w = ((valid >> (i + 3)) & 1u) ? ph.w : 0.0h;
        pL0[i] = ph.x;
        pL0[i + 1] = ph.y;
        pL0[i + 2] = ph.z;
        pL0[i + 3] = ph.w;
        if (b) hB += ph;
        else hA += ph;
      }
      lA0 += float(hA.x) + float(hA.y) + float(hA.z) + float(hA.w);
      lB0 += float(hB.x) + float(hB.y) + float(hB.z) + float(hB.w);
      oppv.run(pL0, mV, oT0);
    }
  };
  vsa_ranges(p, tiles, SEL, h, qt, body);
  vsa_store(oT0, lA0, lB0, h, q0, tq.x + tq.y, r0, cb, O);
}

// ---------------------------------------------------------------------------------------------------- dense
// Dense attention (VsaTable::denseItems, flags bit1): dit_vsa_attention / dit_vsa_attention_qkfix with the key range a
// constant [0, M). Grid and bindings as theirs (buffer 9, the selection, unread).
kernel void dit_vsa_attention_dense(device uchar *Qb [[buffer(0)]], device uchar *Kb [[buffer(1)]],
                                    device half *Vb [[buffer(2)]], device bfloat *O [[buffer(3)]],
                                    device const float *sq [[buffer(4)]], device const float *sk [[buffer(5)]],
                                    constant VsaParams &p [[buffer(6)]], device const int *items [[buffer(7)]],
                                    device const int2 *tiles [[buffer(8)]],
                                    uint tgx [[threadgroup_position_in_grid]],
                                    ushort sg [[simdgroup_index_in_threadgroup]],
                                    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int D = kHeadDim, BQ = kRowsPerTile, BK = kKeysPerTile;
  const int ii = int(tgx) % p.nitems, hl = int(tgx) / p.nitems, h = p.head0 + hl;
  const int2 tq = tiles[items[ii]];
  const int q0 = tq.x + sg * BQ;
  threadgroup_barrier(mem_flags::mem_threadgroup);  // simdgroups start the key loop together
  if (sg * BQ >= tq.y) return;
  const ushort r0 = lane_r0(lane), cb = lane_cb(lane);
  const ulong hs = ulong(p.Spad) * D;
  auto Q = mk<int8_t>((device int8_t *)Qb + hs * h, D, p.Spad);
  auto K = mk<int8_t>((device int8_t *)Kb + hs * h, D, p.Spad);
  auto Vt = mk<half>(Vb + hs * h, D, p.Spad);
  auto mQ0 = Q.template slice<D, BQ>(0, q0);
  constexpr auto dqk = matmul2d_descriptor(BQ, BK, D, false, true, false);
  matmul2d<dqk, execution_simdgroup> opqk;
  constexpr auto dpv = matmul2d_descriptor(BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroup> oppv;
  auto qL0 = opqk.template get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  qL0.load(mQ0);
  auto pL0 = oppv.template get_left_input_cooperative_tensor<half, half, float>();
  auto V0 = Vt.template slice<D, BK>(0, 0);
  auto oT0 = oppv.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(pL0)>, decltype(V0), float>();
#pragma unroll
  for (ushort i = 0; i < kOutSlots; ++i) oT0[i] = 0.0f;
  const ulong sqo = ulong(h) * p.Spad + q0 + r0;
  const float qA0 = sq[sqo], qB0 = sq[sqo + 8];
  float mA0 = -INFINITY, mB0 = -INFINITY, lA0 = 0.0f, lB0 = 0.0f;
  const int e = p.M, nk = (e - 1) / BK + 1;
  device const float *skh = sk + ulong(h) * p.nkb;
  for (int kt = 0; kt < nk; ++kt) {
    const int k0 = kt * BK;
    const bool full = k0 + BK <= e;
    const uint valid = full ? 0xFFFFu : vsa_mask(k0, 0, e, cb);
    const float ks = skh[kt];
    auto mK = K.template slice<D, BK>(0, k0);
    auto mV = Vt.template slice<D, BK>(0, k0);
    auto s0 = opqk.template get_destination_cooperative_tensor<decltype(mQ0), decltype(mK), int>();
    opqk.run(qL0, mK, s0);
    vsa_softmax(s0, pL0, oT0, qA0 * ks, qB0 * ks, mA0, mB0, lA0, lB0, valid, full);
    oppv.run(pL0, mV, oT0);
  }
  vsa_store(oT0, lA0, lB0, h, q0, tq.x + tq.y, r0, cb, O);
}

kernel void dit_vsa_attention_qkfix_dense(device half *Qb [[buffer(0)]], device half *Kb [[buffer(1)]],
                                          device half *Vb [[buffer(2)]], device bfloat *O [[buffer(3)]],
                                          device const float *QS [[buffer(4)]], device const float *KS [[buffer(5)]],
                                          constant VsaParams &p [[buffer(6)]], device const int *items [[buffer(7)]],
                                          device const int2 *tiles [[buffer(8)]],
                                          uint tgx [[threadgroup_position_in_grid]],
                                          ushort sg [[simdgroup_index_in_threadgroup]],
                                          ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int D = kHeadDim, BQ = kRowsPerTile, BK = kKeysPerTile, NS = H3_QKFIX_SPLIT;
  const int ii = int(tgx) % p.nitems, hl = int(tgx) / p.nitems, h = p.head0 + hl;
  const int2 tq = tiles[items[ii]];
  const int q0 = tq.x + sg * BQ;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg * BQ >= tq.y) return;
  const ushort r0 = lane_r0(lane), cb = lane_cb(lane);
  const int rowA = q0 + r0, rowB = rowA + 8;
  const ulong hs = ulong(p.Spad) * D;
  auto Q = mk<half>(Qb + hs * h, D, p.Spad);
  auto K = mk<half>(Kb + hs * h, D, p.Spad);
  auto Vt = mk<half>(Vb + hs * h, D, p.Spad);
  auto mQ0 = Q.template slice<D, BQ>(0, q0);
  constexpr auto dqk = matmul2d_descriptor(BQ, BK, D, false, true, false);
  matmul2d<dqk, execution_simdgroup> opqk;
  constexpr auto dpv = matmul2d_descriptor(BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroup> oppv;
  auto qL0 = opqk.template get_left_input_cooperative_tensor<half, half, float>();
  qL0.load(mQ0);
  auto pL0 = oppv.template get_left_input_cooperative_tensor<half, half, float>();
  auto V0 = Vt.template slice<D, BK>(0, 0);
  auto oT0 = oppv.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(pL0)>, decltype(V0), float>();
#pragma unroll
  for (ushort i = 0; i < kOutSlots; ++i) oT0[i] = 0.0f;
  const float a = p.qk_scale;
  const bool split = (p.flags & 1) != 0;
  const ulong sbase = ulong(h) * p.Spad;
  const float4 qsA = split ? *(device const float4 *)(QS + (sbase + rowA) * NS) * a : float4(0.0f);
  const float4 qsB = split ? *(device const float4 *)(QS + (sbase + rowB) * NS) * a : float4(0.0f);
  float mA0 = -INFINITY, mB0 = -INFINITY, lA0 = 0.0f, lB0 = 0.0f;
  const int re = p.M, nk = (re - 1) / BK + 1;
  for (int kt = 0; kt < nk; ++kt) {
    const int k0 = kt * BK;
    auto mK = K.template slice<D, BK>(0, k0);
    auto mV = Vt.template slice<D, BK>(0, k0);
    auto s0 = opqk.template get_destination_cooperative_tensor<decltype(mQ0), decltype(mK), float>();
    const bool full = k0 + BK <= re;
    const uint valid = full ? 0xFFFFu : vsa_mask(k0, 0, re, cb);
    opqk.run(qL0, mK, s0);
    if (split) {
      const float4 ksl = *(device const float4 *)(KS + (sbase + k0 + lane) * NS);
#pragma unroll
      for (ushort u8 = 0; u8 < 8; ++u8) {
        const float4 ks = simd_shuffle(ksl, ushort(16 * (u8 >> 2) + cb + (u8 & 3)));
        const ushort iA = (u8 >> 2) * 8 + (u8 & 3);
        s0[iA] = fma(float(s0[iA]), a, dot(qsA, ks));
        s0[iA + 4] = fma(float(s0[iA + 4]), a, dot(qsB, ks));
      }
    } else {
#pragma unroll
      for (ushort i = 0; i < kScoreSlots; ++i) s0[i] = float(s0[i]) * a;
    }
    if (!full) {
#pragma unroll
      for (ushort i = 0; i < kScoreSlots; ++i)
        if (!((valid >> i) & 1u)) s0[i] = -INFINITY;
    }
    float xA = s0[0], xB = s0[4];
#pragma unroll
    for (ushort i = 1; i < kScoreSlots; ++i) {
      if (((i >> 2) & 1) == 0) xA = max(xA, float(s0[i]));
      else xB = max(xB, float(s0[i]));
    }
    xA = max(xA, simd_shuffle_xor(xA, ushort(1)));
    xA = max(xA, simd_shuffle_xor(xA, ushort(8)));
    xB = max(xB, simd_shuffle_xor(xB, ushort(1)));
    xB = max(xB, simd_shuffle_xor(xB, ushort(8)));
    if (xA > mA0 || xB > mB0) {
      const float nA_ = max(mA0, xA), nB_ = max(mB0, xB);
      const float cA = exp2(mA0 - nA_), cB = exp2(mB0 - nB_);
      mA0 = nA_;
      mB0 = nB_;
      lA0 *= cA;
      lB0 *= cB;
#pragma unroll
      for (ushort i = 0; i < kOutSlots; ++i) oT0[i] *= ((i >> 2) & 1) ? cB : cA;
    }
    half4 hA = 0.0h, hB = 0.0h;
#pragma unroll
    for (ushort i = 0; i < kScoreSlots; i += 4) {
      const bool b = (i >> 2) & 1;
      const float m = b ? mB0 : mA0;
      float4 t4 = float4(s0[i], s0[i + 1], s0[i + 2], s0[i + 3]);
      half4 ph = exp2(half4(max(t4 - m, -24.0f)));
      ph.x = ((valid >> i) & 1u) ? ph.x : 0.0h;
      ph.y = ((valid >> (i + 1)) & 1u) ? ph.y : 0.0h;
      ph.z = ((valid >> (i + 2)) & 1u) ? ph.z : 0.0h;
      ph.w = ((valid >> (i + 3)) & 1u) ? ph.w : 0.0h;
      pL0[i] = ph.x;
      pL0[i + 1] = ph.y;
      pL0[i + 2] = ph.z;
      pL0[i + 3] = ph.w;
      if (b) hB += ph;
      else hA += ph;
    }
    lA0 += float(hA.x) + float(hA.y) + float(hA.z) + float(hA.w);
    lB0 += float(hB.x) + float(hB.y) + float(hB.z) + float(hB.w);
    oppv.run(pL0, mV, oT0);
  }
  vsa_store(oT0, lA0, lB0, h, q0, tq.x + tq.y, r0, cb, O);
}

// ---------------------------------------------------------------------------------------------------- producer
// One threadgroup per row (dit_producer_inner layout, K = 7168): x = O + out_c[head][tile(row)] (.) gate, then the
// out-projection Hadamard (signs 0..7167 of the 14336 sequence) and per-token int8.
kernel void dit_producer_vsa(device const bfloat *attn [[buffer(0)]], device const half *gate [[buffer(1)]],
                             device const float *oc [[buffer(2)]], device const int *rowTile [[buffer(3)]],
                             device char *Q [[buffer(4)]], device float *S [[buffer(5)]],
                             device const char *signs [[buffer(6)]], constant VsaParams &p [[buffer(7)]],
                             uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                             ushort nt [[threads_per_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int NGP = H3_PRODUCER_GROUPS_INNER, KC = H3_DIT_INNER;
  threadgroup float red[32];
  if (int(row) >= p.M) return;
  const ushort halves = nt / 16, hh = tid / 16, j = tid & 15, nsg = nt / 32;
  const int tile = rowTile[row];
  float4 va[NGP], vb[NGP];
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    const int g = hh + i * halves;
    const int c = g * 128 + 8 * j;
    if (g < H3_DIT_HEADS && (p.flags & 2)) {  // dense attention: O alone
      device const bfloat *x = attn + ulong(row) * KC + c;
      va[i] = ld4(x);
      vb[i] = ld4(x + 4);
    } else if (g < H3_DIT_HEADS) {
      device const bfloat *x = attn + ulong(row) * KC + c;
      device const half *gt = gate + ulong(row) * KC + c;
      device const float *o = oc + (ulong(g) * p.Tp + tile) * 128 + 8 * j;
      va[i] = ld4(x) + ld4(o) * ld4(gt);
      vb[i] = ld4(x + 4) + ld4(o + 4) * ld4(gt + 4);
    } else {
      va[i] = 0.0f;
      vb[i] = 0.0f;
    }
    va[i] *= ld4s(signs + c);
    vb[i] *= ld4s(signs + c + 4);
    wht128(va[i], vb[i], lane);
  }
  float am = 0.0f;
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    float4 m4 = max(fabs(va[i]), fabs(vb[i]));
    am = max(am, max(max(m4.x, m4.y), max(m4.z, m4.w)));
  }
  am = simd_max(am);
  if (lane == 0) red[sg] = am;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  am = red[0];
  for (ushort k = 1; k < nsg; ++k) am = max(am, red[k]);
  const float inv = q8_inv(am);
  device char *q = Q + ulong(row) * KC;
#pragma unroll
  for (int i = 0; i < NGP; ++i) {
    const int c = (hh + i * halves) * 128 + 8 * j;
    *(device uint2 *)(q + c) = uint2(q4pack(va[i], inv), q4pack(vb[i], inv));
  }
  if (tid == 0) S[row] = am / kQMax * kRs128;
}

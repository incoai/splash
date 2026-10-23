// QKFIX: high-precision Q / K path for the blocks whose QK-norm gains make W8A8 + int8 QK inaccurate (45, 48, 49;
// block 49 gains up to 33.75 / 37.0 vs mean 1.08 in block 0).
//   dit_qkfix_producer          RMSNorm + adaLN modulate -> fp16 x [M, 5376] (no Hadamard; fp32 math as dit_producer)
//   dit_qkfix_qkv               q / k / v projections fp16 x fp16 -> fp32 (MPP matmul2d), fp32 epilogue: per-head RMS
//                               QK-norm, gain, 3D RoPE, no bf16 rounding; fp16 Q / K head-major with the split dims
//                               zeroed, the split dims in fp32 to QS / KS [H][Spad][4], raw q / k (as dit_qkv), fp16 V
//   dit_qkfix_window_attention  dit_window_attention with fp16 QK^T (fp32 accumulation) plus the exact fp32 dot of
//                               the split dims: s = a (Q16.K16) + (a QS).KS; partials merged by dit_window_combine
#include "metal/abi/video/QkFixABI.h"
#include "metal/kernels/video/common/quant.h"
#include "metal/kernels/video/common/window.h"

using namespace h3;

static inline float comp(float4 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w)); }

// ---------------------------------------------------------------------------------------------------- producer
// grid M threadgroups x 256 threads; K <= 256 * 4 * 6
kernel void dit_qkfix_producer(device const bfloat *X [[buffer(0)]], device half *Y [[buffer(1)]],
                               device const uchar *mod [[buffer(2)]], device const bfloat *normw [[buffer(3)]],
                               device const bfloat *ada [[buffer(4)]], constant QkFixProducerParams &p [[buffer(5)]],
                               uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[8];
  if (int(row) >= p.M) return;
  device const bfloat *x = X + ulong(row) * p.K;
  constexpr int NV = 6;
  float4 v[NV];
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    const int c = (tid + i * 256) * 4;
    v[i] = c < p.K ? ld4(x + c) : float4(0.0f);
    ss += dot(v[i], v[i]);
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (ushort k = 0; k < 8; ++k) tot += red[k];
  const float rinv = rsqrt(tot / float(p.K) + p.eps);
  device const bfloat *ad = ada + int(mod[row]) * p.ada_stride;
  device const bfloat *sh = ad + p.shift_off, *sc = ad + p.scale_off;
  device half *y = Y + ulong(row) * p.K;
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    const int c = (tid + i * 256) * 4;
    if (c >= p.K) break;
    float4 m = v[i] * rinv * ld4(normw + c) * (1.0f + ld4(sc + c)) + ld4(sh + c);
    *(device half4 *)(y + c) = half4(clamp(m, -65504.0f, 65504.0f));
  }
}

// ---------------------------------------------------------------------------------------------------- fp16 QKV
template <int TM, int TN, class Op, class TA, class TB>
inline void qkfix_tile(thread Op &op, thread TA &a, thread TB &b, int m0, int n0, device const bfloat *rope,
                       device const bfloat *qnw, device const bfloat *knw, device half *Qh, device half *Kh,
                       device half *Rq, device half *Rk, device float *QS, device float *KS, device const int *sdims,
                       device half *Vh, constant QkFixQkvParams &p, threadgroup float *tile, ushort sg, ushort lane,
                       ushort nsg) {
  constexpr int NS = H3_QKFIX_SPLIT;
  auto c = op.template get_destination_cooperative_tensor<TA, TB, float>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0.0f;
  op.run(a, b, c);
  const ushort cap = c.get_capacity();
#pragma unroll
  for (ushort i = 0; i < cap; ++i) {
    if (!c.is_valid_element(i)) continue;
    auto idx = c.get_multidimensional_index(i);
    tile[idx[1] * 128 + idx[0]] = (m0 + idx[1] < p.M) ? float(c[i]) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int head = n0 / 128, part = head / p.H, hh = head - part * p.H;
  if (part == 2) {  // v: plain fp16 store
    for (int r = sg; r < TM; r += nsg) {
      const int row = m0 + r;
      if (row >= p.M) break;
      const float4 y = *(threadgroup float4 *)(tile + r * 128 + 4 * lane);
      *(device vec<half, 4> *)(Vh + (ulong(hh) * p.Spad + p.row0 + row) * 128 + 4 * lane) =
          vec<half, 4>(clamp(y, -65504.0f, 65504.0f));
    }
    return;
  }
  device half *dstb = part == 0 ? Qh : Kh;
  device half *rawb = part == 0 ? Rq : Rk;
  device float *sb = part == 0 ? QS : KS;
  device const bfloat *nw = part == 0 ? qnw : knw;
  const float4 w4 = ld4(nw + 4 * lane);
  const int4 sd = *(device const int4 *)sdims;
  for (int r = sg; r < TM; r += nsg) {
    const int row = m0 + r;
    if (row >= p.M) break;
    float4 y = *(threadgroup float4 *)(tile + r * 128 + 4 * lane);
    const int grow = p.row0 + row;
    if (p.flags & 1) {
      int rr = grow;
      bool keep = true;
      if (p.flags & 2) {
        keep = grow < p.rtext || (grow >= p.rv0 && grow < p.rv1);
        rr = grow < p.rtext ? grow : p.rtext + (grow - p.rv0);
      }
      if (keep)
        *(device vec<half, 4> *)(rawb + (ulong(hh) * p.rspad + rr) * 128 + 4 * lane) =
            vec<half, 4>(clamp(y, -65504.0f, 65504.0f));
    }
    const float ss = simd_sum(dot(y, y));
    y = y * rsqrt(ss / 128.0f + p.eps) * w4;  // fp32, no bf16 rounding
    const float4 up = simd_shuffle_down(y, ushort(12)), dn = simd_shuffle_up(y, ushort(12));
    if (lane < 24) {
      const int jj = lane < 12 ? 4 * lane : 4 * lane - 48;
      device const bfloat *rp = rope + ulong(row) * 96;
      const float4 cs = ld4(rp + jj), sn = ld4(rp + 48 + jj);
      y = lane < 12 ? y * cs - up * sn : y * cs + dn * sn;
    }
    const ulong o = ulong(hh) * p.Spad + grow;
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      const int d = sd[j];
      if (d >= 0 && (d >> 2) == int(lane)) {
        sb[o * NS + j] = comp(y, d & 3);
        y = select(y, float4(0.0f), int4(0, 1, 2, 3) == int4(d & 3));
      }
    }
    *(device vec<half, 4> *)(dstb + o * 128 + 4 * lane) = vec<half, 4>(y);
  }
}

template <int TM, int TN, int NSG>
kernel void dit_qkfix_qkv(device half *A [[buffer(0)]], device half *W [[buffer(1)]], device const bfloat *rope [[buffer(2)]],
                          device const bfloat *qnw [[buffer(3)]], device const bfloat *knw [[buffer(4)]],
                          device half *Qh [[buffer(5)]], device half *Kh [[buffer(6)]], device half *Rq [[buffer(7)]],
                          device half *Rk [[buffer(8)]], device float *QS [[buffer(9)]], device float *KS [[buffer(10)]],
                          device const int *sdims [[buffer(11)]], constant QkFixQkvParams &p [[buffer(12)]],
                          device half *Vh [[buffer(13)]], uint2 tg0 [[threadgroup_position_in_grid]],
                          uint2 ng [[threadgroups_per_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  const uint2 tg = swz(tg0, ng, p.swizzle);
  threadgroup float tile[TM * 128];
  auto tA = mk<half>(A, p.K, p.M);
  auto tW = mk<half>(W, p.K, p.N);
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  const int m0 = tg.y * TM, n0 = tg.x * TN;
  if (m0 + TM <= p.M) {
    auto a = tA.template slice<dynamic_extent, TM>(0, m0);
    auto b = tW.template slice<dynamic_extent, TN>(0, n0);
    qkfix_tile<TM, TN>(op, a, b, m0, n0, rope, qnw, knw, Qh, Kh, Rq, Rk, QS, KS, sdims, Vh, p, tile, sg, lane, NSG);
  } else {
    auto a = tA.slice(0, m0);
    auto b = tW.slice(0, n0);
    qkfix_tile<TM, TN>(op, a, b, m0, n0, rope, qnw, knw, Qh, Kh, Rq, Rk, QS, KS, sdims, Vh, p, tile, sg, lane, NSG);
  }
}
template [[host_name("dit_qkfix_qkv")]] [[kernel]] void dit_qkfix_qkv<H3_QKFIX_TM, 128, 4>(
    device half *, device half *, device const bfloat *, device const bfloat *, device const bfloat *, device half *,
    device half *, device half *, device half *, device float *, device float *, device const int *,
    constant QkFixQkvParams &, device half *, uint2, uint2, ushort, ushort);

// ---------------------------------------------------------------------------------------------------- attention
// grid: nitems * hb threadgroups (head-major), 128 threads; Q / K fp16 [H][Spad][128], QS / KS fp32 [H][Spad][4].
// The split term: lane j loads KS of key k0 + j; the lane's 8 key columns are fetched with simd_shuffle and key u8
// feeds slots iA and iA + 4.
kernel void dit_qkfix_window_attention(device half *Qb [[buffer(0)]], device half *Kb [[buffer(1)]],
                                       device half *Vb [[buffer(2)]], device bfloat *O [[buffer(3)]],
                                       device const float *QS [[buffer(4)]], device const float *KS [[buffer(5)]],
                                       constant WindowParams &p [[buffer(6)]],
                                       device const WindowItem *items [[buffer(7)]], device const int *tab [[buffer(8)]],
                                       device half *Op [[buffer(9)]], device float2 *ML [[buffer(10)]],
                                       uint tgx [[threadgroup_position_in_grid]],
                                       ushort sg [[simdgroup_index_in_threadgroup]],
                                       ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int D = kHeadDim, BQ = kRowsPerTile, BK = kKeysPerTile, NS = H3_QKFIX_SPLIT;
  const int ii = int(tgx) % p.nitems, hl = int(tgx) / p.nitems, h = p.head0 + hl;
  const WindowItem it = items[ii];
  const int q0 = it.qg * 64 + sg * BQ;
  threadgroup_barrier(mem_flags::mem_threadgroup);
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
  auto Q = mk<half>(Qb + hs * h, D, p.Spad);
  auto K = mk<half>(Kb + hs * h, D, p.Spad);
  auto V = mk<half>(Vb + hs * h, D, p.Spad);
  auto mQ0 = Q.template slice<D, BQ>(0, q0);
  constexpr auto dqk = matmul2d_descriptor(BQ, BK, D, false, true, false);
  matmul2d<dqk, execution_simdgroup> opqk;
  constexpr auto dpv = matmul2d_descriptor(BQ, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroup> oppv;
  auto qL0 = opqk.template get_left_input_cooperative_tensor<half, half, float>();
  qL0.load(mQ0);
  auto pL0 = oppv.template get_left_input_cooperative_tensor<half, half, float>();
  auto V0 = V.template slice<D, BK>(0, 0);
  auto oT0 = oppv.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(pL0)>, decltype(V0), float>();
#pragma unroll
  for (ushort i = 0; i < kOutSlots; ++i) oT0[i] = 0.0f;
  const float a = p.scale;
  const bool split = (p.flags & 1) == 0;
  const ulong sbase = ulong(h) * p.Spad;
  const float4 qsA = split ? *(device const float4 *)(QS + (sbase + rowA) * NS) * a : float4(0.0f);
  const float4 qsB = split ? *(device const float4 *)(QS + (sbase + rowB) * NS) * a : float4(0.0f);
  float mA0 = -INFINITY, mB0 = -INFINITY, lA0 = 0.0f, lB0 = 0.0f;
  int prev = -1;
  for (int u = 0; u < nU; ++u) {
    const int ua = max(rec[8 + 2 * u], it.kbeg), ue = min(rec[9 + 2 * u], it.kend);
    if (ua >= ue) continue;
    const int t0 = max(ua / BK, prev + 1), t1 = (ue - 1) / BK;
    for (int kt = t0; kt <= t1; ++kt) {
      const int k0 = kt * BK;
      auto mK = K.template slice<D, BK>(0, k0);
      auto mV = V.template slice<D, BK>(0, k0);
      auto s0 = opqk.template get_destination_cooperative_tensor<decltype(mQ0), decltype(mK), float>();
      const bool full = !mixed && ua <= k0 && k0 + BK <= ue;
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
      opqk.run(qL0, mK, s0);
      // log2-domain scores in place: s = a * (Q16.K16) + (a * QS).KS; masked -> -inf
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
    prev = max(prev, t1);
  }
  window_finish(oT0, mA0, mB0, lA0, lB0, h, hl, q0, it.slot, sg * BQ, r0, cb, p, O, Op, ML);
}

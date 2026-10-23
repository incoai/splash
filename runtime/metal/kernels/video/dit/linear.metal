// DiT W8A8 GEMMs (MPP matmul2d i8 x i8 -> i32, one 64 x 128 tile per threadgroup) with fused epilogues. The
// destination stays a cooperative tensor; lanes hold runs of 4 consecutive columns of one row (checked at run time),
// which are dequantized and stored as one 8-byte write.
//   dit_qkv            dequant -> bf16 -> per-head RMS QK-norm -> 3D RoPE (96 of 128 dims) -> fp16 head-major
//                      Q, K, V [H][Spad][128] (a 128-wide tile is one head of q, k or v); also the pre-norm q / k
//                      (bf16-rounded linear output, stored fp16) of text and inner video rows for the linear branch
//   dit_gated_residual X[row] += gate[mod[row]] * dequant(acc), bf16 in place (out, to_out_linear, fc2)
//   dit_fc1_swiglu     two matmuls per tile (gate rows [n0, n0 + 128), value rows [F + n0, ...)), h = silu(g) * v,
//                      bf16 h; barrier before the K loop and tile swizzle 4
#include "metal/abi/video/LinearABI.h"
#include "metal/kernels/video/common/quant.h"

using namespace h3;

// ---------------------------------------------------------------------------------------------------- QKV
template <int TM, int TN, class Op, class TA, class TB>
inline void qkv_tile(thread Op &op, thread TA &a, thread TB &b, int m0, int n0, device const float *sa,
                     device const float *sw, device const bfloat *qnw, device const bfloat *knw,
                     device const bfloat *rope, device half *Qh, device half *Kh, device half *Vh, device half *Rq,
                     device half *Rk, constant QkvParams &p, threadgroup bfloat *tile, ushort sg, ushort lane,
                     ushort nsg) {
  auto c = op.template get_destination_cooperative_tensor<TA, TB, int>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0;
  op.run(a, b, c);
  const ushort cap = c.get_capacity();
  // Pass 1: dequant -> bf16 (the linear's output dtype) into threadgroup memory [TM][128].
  const int head = n0 / 128, part = head / p.H, hh = head - part * p.H;
#pragma unroll
  for (ushort i0 = 0; i0 < cap; i0 += 4) {
    bool run4 = (cap % 4 == 0) && c.is_valid_element(i0) && c.is_valid_element(i0 + 3);
    int r = 0, cc = 0;
    if (run4) {
      auto a0 = c.get_multidimensional_index(i0);
      auto a3 = c.get_multidimensional_index(i0 + 3);
      r = a0[1];
      cc = a0[0];
      run4 = a3[1] == a0[1] && a3[0] == a0[0] + 3 && (cc & 3) == 0;
    }
    if (run4) {
      int row = m0 + r;
      float4 v = float4(float(c[i0]), float(c[i0 + 1]), float(c[i0 + 2]), float(c[i0 + 3]));
      if (row < p.M) v *= sa[row] * *(device const float4 *)(sw + n0 + cc);
      *(threadgroup vec<bfloat, 4> *)(tile + r * 128 + cc) = vec<bfloat, 4>(v);
    } else {
      for (ushort i = i0; i < i0 + 4 && i < cap; ++i) {
        if (!c.is_valid_element(i)) continue;
        auto idx = c.get_multidimensional_index(i);
        int row = m0 + idx[1];
        tile[idx[1] * 128 + idx[0]] = bfloat(row < p.M ? float(c[i]) * sa[row] * sw[n0 + idx[0]] : 0.0f);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Pass 2: one simdgroup per row, lane holds dims 4 lane .. 4 lane + 3:
  //   q/k: y = bf16(x * rsqrt(mean(x^2) + eps) * w)   (norm rounded before RoPE)
  //        d < 48      : y[d] cos[d]    - y[d+48] sin[d]
  //        48 <= d < 96: y[d] cos[d-48] + y[d-48] sin[d-48]   (neox rotate_half on 96 dims)
  //        d >= 96     : pass through
  //   v  : copy.
  device half *dstb = part == 0 ? Qh : (part == 1 ? Kh : Vh);
  device half *rawb = part == 0 ? Rq : Rk;
  device const bfloat *nw = part == 0 ? qnw : knw;
  const float4 w4 = part < 2 ? ld4(nw + 4 * lane) : float4(1.0f);
  const bool raw = (p.flags & 1) && part < 2;
  for (int r = sg; r < TM; r += nsg) {
    int row = m0 + r;
    if (row >= p.M) break;
    float4 y = float4(*(threadgroup vec<bfloat, 4> *)(tile + r * 128 + 4 * lane));
    const int grow = p.row0 + row;
    device half *dst = dstb + (ulong(hh) * p.Spad + grow) * 128 + 4 * lane;
    if (raw) {
      int rr = grow;
      bool keep = true;
      if (p.flags & 2) {
        keep = grow < p.rtext || (grow >= p.rv0 && grow < p.rv1);
        rr = grow < p.rtext ? grow : p.rtext + (grow - p.rv0);
      }
      if (keep) {
        *(device vec<half, 4> *)(rawb + (ulong(hh) * p.rspad + rr) * 128 + 4 * lane) =
            vec<half, 4>(clamp(y, -65504.0f, 65504.0f));
      }
    }
    if (part < 2) {
      float ss = simd_sum(dot(y, y));
      y = float4(vec<bfloat, 4>(y * rsqrt(ss / 128.0f + p.eps) * w4));
      float4 up = simd_shuffle_down(y, ushort(12)), dn = simd_shuffle_up(y, ushort(12));
      if (lane < 24) {
        int jj = lane < 12 ? 4 * lane : 4 * lane - 48;
        device const bfloat *rr = rope + ulong(row) * 96;
        float4 cs = ld4(rr + jj), sn = ld4(rr + 48 + jj);
        y = lane < 12 ? y * cs - up * sn : y * cs + dn * sn;
      }
    }
    *(device vec<half, 4> *)dst = vec<half, 4>(y);
  }
}

template <int TM, int TN, int NSG>
kernel void dit_qkv(device int8_t *A [[buffer(0)]], device int8_t *W [[buffer(1)]], device const float *sa [[buffer(2)]],
                    device const float *sw [[buffer(3)]], device const bfloat *rope [[buffer(4)]],
                    device const bfloat *qnw [[buffer(5)]], device const bfloat *knw [[buffer(6)]],
                    device half *Qh [[buffer(7)]], device half *Kh [[buffer(8)]], device half *Vh [[buffer(9)]],
                    device half *Rq [[buffer(10)]], device half *Rk [[buffer(11)]],
                    constant QkvParams &p [[buffer(12)]], uint2 tg0 [[threadgroup_position_in_grid]],
                    uint2 ng [[threadgroups_per_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
  const uint2 tg = swz(tg0, ng, p.swizzle);
  threadgroup bfloat tile[TM * 128];
  auto tA = mk<int8_t>(A, p.K, p.M);
  auto tW = mk<int8_t>(W, p.K, p.N);
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  int m0 = tg.y * TM, n0 = int(tg.x) * TN;
  if (m0 + TM <= p.M && n0 + TN <= p.N) {
    auto a = tA.template slice<dynamic_extent, TM>(0, m0);
    auto b = tW.template slice<dynamic_extent, TN>(0, n0);
    qkv_tile<TM, TN>(op, a, b, m0, n0, sa, sw, qnw, knw, rope, Qh, Kh, Vh, Rq, Rk, p, tile, sg, lane, NSG);
  } else {
    auto a = tA.slice(0, m0);
    auto b = tW.slice(0, n0);
    qkv_tile<TM, TN>(op, a, b, m0, n0, sa, sw, qnw, knw, rope, Qh, Kh, Vh, Rq, Rk, p, tile, sg, lane, NSG);
  }
}
template [[host_name("dit_qkv")]] [[kernel]] void dit_qkv<H3_LINEAR_TM, H3_LINEAR_TN, H3_LINEAR_NSG>(
    device int8_t *, device int8_t *, device const float *, device const float *, device const bfloat *,
    device const bfloat *, device const bfloat *, device half *, device half *, device half *, device half *,
    device half *, constant QkvParams &, uint2, uint2, ushort, ushort);

// ---------------------------------------------------------------------------------------------------- gated residual
template <int TM, int TN, class Op, class TA, class TB>
inline void gated_residual_tile(thread Op &op, thread TA &a, thread TB &b, int m0, int n0, device const float *sa,
                                device const float *sw, device bfloat *Y, device const uchar *mod,
                                device const bfloat *ada, constant GatedResidualParams &p) {
  auto c = op.template get_destination_cooperative_tensor<TA, TB, int>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0;
  op.run(a, b, c);
  const ushort cap = c.get_capacity();
  auto deq = [&](ushort i, int row, int col) {
    float v = float(c[i]);
    v *= sa[row] * sw[col];
    return v;
  };
  // out = residual + gate[mod[row]][col] * dequant(acc); residual and out are the same buffer
#pragma unroll
  for (ushort i0 = 0; i0 < cap; i0 += 4) {
    bool run4 = (cap % 4 == 0) && c.is_valid_element(i0) && c.is_valid_element(i0 + 3);
    int row = 0, col = 0;
    if (run4) {
      auto a0 = c.get_multidimensional_index(i0);
      auto a3 = c.get_multidimensional_index(i0 + 3);
      row = m0 + a0[1];
      col = n0 + a0[0];
      run4 = a3[1] == a0[1] && a3[0] == a0[0] + 3 && row < p.M && col + 3 < p.N && (col & 3) == 0 && (p.ldo & 3) == 0;
    }
    if (run4) {
      float4 v = float4(float(c[i0]), float(c[i0 + 1]), float(c[i0 + 2]), float(c[i0 + 3]));
      v *= sa[row] * *(device const float4 *)(sw + col);
      device const bfloat *g = ada + int(mod[row]) * p.ada_stride + p.gate_off;
      device bfloat *y = Y + ulong(row) * p.ldo + col;
      float4 r = ld4(y) + ld4(g + col) * v;
      *(device vec<bfloat, 4> *)y = vec<bfloat, 4>(r);
    } else {
      for (ushort i = i0; i < i0 + 4 && i < cap; ++i) {
        if (!c.is_valid_element(i)) continue;
        auto idx = c.get_multidimensional_index(i);
        int r = m0 + idx[1], cc = n0 + idx[0];
        if (r >= p.M || cc >= p.N) continue;
        float v = deq(i, r, cc);
        device const bfloat *g = ada + int(mod[r]) * p.ada_stride + p.gate_off;
        device bfloat *y = Y + ulong(r) * p.ldo + cc;
        *y = bfloat(float(*y) + float(g[cc]) * v);
      }
    }
  }
}

template <int TM, int TN, int NSG>
kernel void dit_gated_residual(device int8_t *A [[buffer(0)]], device int8_t *W [[buffer(1)]],
                               device const float *sa [[buffer(2)]], device const float *sw [[buffer(3)]],
                               device bfloat *Y [[buffer(4)]], device const uchar *mod [[buffer(5)]],
                               device const bfloat *ada [[buffer(6)]], constant GatedResidualParams &p [[buffer(7)]],
                               uint2 tg0 [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]]) {
  const uint2 tg = swz(tg0, ng, p.swizzle);
  auto tA = mk<int8_t>(A, p.K, p.M);
  auto tW = mk<int8_t>(W, p.K, p.N);
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  int m0 = tg.y * TM, n0 = tg.x * TN;
  if (m0 + TM <= p.M && n0 + TN <= p.N) {
    auto a = tA.template slice<dynamic_extent, TM>(0, m0);
    auto b = tW.template slice<dynamic_extent, TN>(0, n0);
    gated_residual_tile<TM, TN>(op, a, b, m0, n0, sa, sw, Y, mod, ada, p);
  } else {
    auto a = tA.slice(0, m0);
    auto b = tW.slice(0, n0);
    gated_residual_tile<TM, TN>(op, a, b, m0, n0, sa, sw, Y, mod, ada, p);
  }
}
template [[host_name("dit_gated_residual")]] [[kernel]] void
dit_gated_residual<H3_LINEAR_TM, H3_LINEAR_TN, H3_LINEAR_NSG>(device int8_t *, device int8_t *, device const float *,
                                                              device const float *, device bfloat *,
                                                              device const uchar *, device const bfloat *,
                                                              constant GatedResidualParams &, uint2, uint2);

// ---------------------------------------------------------------------------------------------------- fc1 + SwiGLU
// A threadgroup owns TM rows x TNH hidden channels: two matmuls with the same descriptor, element i of both
// destinations has the same (row, col), so silu(g) * v is thread-local.
template <int TM, class Op, class TA, class TB>
inline void swiglu_tile(thread Op &op, thread TA &a, thread TB &bg, thread TB &bv, int m0, int n0,
                        device const float *sa, device const float *sw, device bfloat *Hb, constant SwiGluParams &p) {
  auto cg = op.template get_destination_cooperative_tensor<TA, TB, int>();
  auto cv = op.template get_destination_cooperative_tensor<TA, TB, int>();
#pragma unroll
  for (ushort i = 0; i < cg.get_capacity(); ++i)
    if (cg.is_valid_element(i)) { cg[i] = 0; cv[i] = 0; }
  op.run(a, bg, cg);
  op.run(a, bv, cv);
  auto scalar = [&](ushort i) {
    if (!cg.is_valid_element(i)) return;
    auto idx = cg.get_multidimensional_index(i);
    int r = idx[1];
    int row = m0 + r, col = n0 + idx[0];
    float h = 0.0f;
    if (row < p.M && col < p.F) {
      float g = float(cg[i]), v = float(cv[i]);
      float s = sa[row];
      g *= s * sw[col];
      v *= s * sw[p.F + col];
      h = silu_mul(g, v);
      bfloat hb = bfloat(h);
      Hb[ulong(row) * p.F + col] = hb;
    }
  };
  const ushort cap = cg.get_capacity();
#pragma unroll
  for (ushort i0 = 0; i0 < cap; i0 += 4) {
    bool run4 = (cap % 4 == 0) && cg.is_valid_element(i0) && cg.is_valid_element(i0 + 3);
    int row = 0, col = 0;
    if (run4) {
      auto a0 = cg.get_multidimensional_index(i0);
      auto a3 = cg.get_multidimensional_index(i0 + 3);
      row = m0 + a0[1];
      col = n0 + a0[0];
      run4 = a3[1] == a0[1] && a3[0] == a0[0] + 3 && row < p.M && col + 3 < p.F && (col & 3) == 0 && (p.F & 3) == 0;
    }
    if (!run4) {
      for (ushort i = i0; i < i0 + 4 && i < cap; ++i) scalar(i);
      continue;
    }
    float4 g = float4(float(cg[i0]), float(cg[i0 + 1]), float(cg[i0 + 2]), float(cg[i0 + 3]));
    float4 v = float4(float(cv[i0]), float(cv[i0 + 1]), float(cv[i0 + 2]), float(cv[i0 + 3]));
    float s = sa[row];
    g *= s * *(device const float4 *)(sw + col);
    v *= s * *(device const float4 *)(sw + p.F + col);
    float4 h = g * v / (1.0f + precise::exp(-g));
    vec<bfloat, 4> hb = vec<bfloat, 4>(h);
    *(device vec<bfloat, 4> *)(Hb + ulong(row) * p.F + col) = hb;
  }
}

template <int TM, int TNH, int NSG, int GM>
kernel void dit_fc1_swiglu(device int8_t *A [[buffer(0)]], device int8_t *W [[buffer(1)]],
                           device const float *sa [[buffer(2)]], device const float *sw [[buffer(3)]],
                           device bfloat *Hb [[buffer(4)]], constant SwiGluParams &p [[buffer(5)]],
                           uint2 tg0 [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]]) {
  threadgroup_barrier(mem_flags::mem_threadgroup);  // simdgroups start the K loop together
  const uint2 tg = swz(tg0, ng, GM);
  auto tA = mk<int8_t>(A, p.K, p.M);
  auto tW = mk<int8_t>(W, p.K, 2 * p.F);
  constexpr auto desc = matmul2d_descriptor(TM, TNH, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  int m0 = tg.y * TM, n0 = tg.x * TNH;
  auto bg = tW.template slice<dynamic_extent, TNH>(0, n0);
  auto bv = tW.template slice<dynamic_extent, TNH>(0, p.F + n0);
  if (m0 + TM <= p.M) {
    auto a = tA.template slice<dynamic_extent, TM>(0, m0);
    swiglu_tile<TM, decltype(op), decltype(a), decltype(bg)>(op, a, bg, bv, m0, n0, sa, sw, Hb, p);
  } else {
    auto a = tA.slice(0, m0);
    swiglu_tile<TM, decltype(op), decltype(a), decltype(bg)>(op, a, bg, bv, m0, n0, sa, sw, Hb, p);
  }
}
template [[host_name("dit_fc1_swiglu")]] [[kernel]] void
dit_fc1_swiglu<H3_LINEAR_TM, H3_LINEAR_TN, H3_LINEAR_NSG, 4>(device int8_t *, device int8_t *, device const float *,
                                                            device const float *, device bfloat *,
                                                            constant SwiGluParams &, uint2, uint2);

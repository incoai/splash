// Video VAE linears: A [M, K] x W [N, K] (K contiguous), f16 x f16 -> f32 or i8 x i8 -> i32 (dequant acc * sa[row] *
// sw[col]) on MPP matmul2d, with fused epilogues:
//   vae_qkv_{i8,f16}       +bias -> per-head RMS q / k norm (non-affine, fp32) -> RoPE (48 of 64 dims) ->
//                          fp16 head-major Q, K, V [tile][32][Spad][64]  (weight rows reordered q | k | v)
//   vae_residual_{i8,f16}  x += scale[c] * (acc + bias[c])  (fp32 residual, in place)
//   vae_proj_f16           y = acc + bias (fp32; proj_out)
//   vae_glu_{i8,f16}       fc1 with fused bias + silu(gate) * up -> fp16 h [M, 8192]
#include "metal/abi/video/VaeABI.h"
#include "metal/kernels/video/common/quant.h"

using namespace h3;

enum : int { EPI_RES = 0, EPI_OUT = 1, EPI_QKV = 2 };

template <bool INT, int EPI, int TM, int TN, class Op, class TA, class TB, class ACC>
inline void vae_tile(thread Op &op, thread TA &a, thread TB &b, int m0, int n0, device const float *sa,
                     device const float *sw, device const float *bias, device float *Y, device const float *aux,
                     constant VaeParams &p, device half *Qh, device half *Kh, device half *Vh,
                     device const float *aux2, threadgroup float *tile, ushort sg, ushort lane, ushort nsg) {
  auto c = op.template get_destination_cooperative_tensor<TA, TB, ACC>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = ACC(0);
  op.run(a, b, c);
  const ushort cap = c.get_capacity();
  auto deq = [&](ushort i, int row, int col) {
    float v = float(c[i]);
    if (INT) v *= sa[row] * sw[col];
    return v + bias[col];
  };
  if (EPI == EPI_RES || EPI == EPI_OUT) {
#pragma unroll
    for (ushort i0 = 0; i0 < cap; i0 += 4) {
      bool run4 = (cap % 4 == 0) && c.is_valid_element(i0) && c.is_valid_element(i0 + 3);
      int row = 0, col = 0;
      if (run4) {
        auto a0 = c.get_multidimensional_index(i0);
        auto a3 = c.get_multidimensional_index(i0 + 3);
        row = m0 + a0[1];
        col = n0 + a0[0];
        run4 = a3[1] == a0[1] && a3[0] == a0[0] + 3 && row < p.M && col + 3 < p.N && (col & 3) == 0 &&
               (p.ldo & 3) == 0;
      }
      if (run4) {
        float4 v = float4(float(c[i0]), float(c[i0 + 1]), float(c[i0 + 2]), float(c[i0 + 3]));
        if (INT) v *= sa[row] * ld4(sw + col);
        v += ld4(bias + col);
        device float *y = Y + ulong(row) * p.ldo + col;
        if (EPI == EPI_RES) *(device float4 *)y = ld4(y) + v * ld4(aux + col);
        else *(device float4 *)y = v;
      } else {
        for (ushort i = i0; i < i0 + 4 && i < cap; ++i) {
          if (!c.is_valid_element(i)) continue;
          auto idx = c.get_multidimensional_index(i);
          int r = m0 + idx[1], cc = n0 + idx[0];
          if (r >= p.M || cc >= p.N) continue;
          float v = deq(i, r, cc);
          device float *y = Y + ulong(r) * p.ldo + cc;
          if (EPI == EPI_RES) *y = *y + v * aux[cc];
          else *y = v;
        }
      }
    }
    return;
  }
  // EPI_QKV: N = 3 * 2048, columns [q heads 0..31 | k | v] x 64; a TN = 128 tile is two heads of one part.
  // Pass 1: acc + bias -> fp32 threadgroup tile [TM][128]. Pass 2: one simdgroup per row, lane l holds dims
  // 4 (l % 16)..+3 of head (l / 16):
  //   q/k: y = x * rsqrt(mean_64(x^2) + eps)   (non-affine, fp32)
  //        d < 24       : y[d] cos[d] - y[d+24] sin[d]
  //        24 <= d < 48 : y[d] cos[d] + y[d-24] sin[d]       (rotate_half on 48 dims; cos = aux, sin = aux2 [S][48])
  //        d >= 48      : pass through
  //   v  : copy.  -> fp16 [tile][32][Spad][64]
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
      if (INT && row < p.M) v *= sa[row] * ld4(sw + n0 + cc);
      v += ld4(bias + n0 + cc);
      *(threadgroup float4 *)(tile + r * 128 + cc) = v;
    } else {
      for (ushort i = i0; i < i0 + 4 && i < cap; ++i) {
        if (!c.is_valid_element(i)) continue;
        auto idx = c.get_multidimensional_index(i);
        int row = m0 + idx[1];
        tile[idx[1] * 128 + idx[0]] = row < p.M ? deq(i, row, n0 + idx[0]) : 0.0f;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int part = n0 / 2048, head = (n0 % 2048) / 64 + lane / 16, dl = lane & 15, d = 4 * dl;
  device half *dstb = part == 0 ? Qh : (part == 1 ? Kh : Vh);
  device const float *cosT = aux, *sinT = aux2;
  for (int r = sg; r < TM; r += nsg) {
    int row = m0 + r;
    if (row >= p.M) break;
    const int tb = row / p.S, rr = row - tb * p.S;
    float4 y = *(threadgroup float4 *)(tile + r * 128 + 4 * lane);
    if (part < 2) {
      float ss = dot(y, y);
      ss += simd_shuffle_xor(ss, 1);
      ss += simd_shuffle_xor(ss, 2);
      ss += simd_shuffle_xor(ss, 4);
      ss += simd_shuffle_xor(ss, 8);
      y = y * rsqrt(ss / 64.0f + p.eps);
      float4 up = simd_shuffle_down(y, ushort(6)), dn = simd_shuffle_up(y, ushort(6));
      if (dl < 12) {
        float4 cs = ld4(cosT + rr * 48 + d), sn = ld4(sinT + rr * 48 + d);
        y = dl < 6 ? y * cs - up * sn : y * cs + dn * sn;
      }
    }
    *(device half4 *)(dstb + ((ulong(tb) * p.H + head) * p.Spad + rr) * 64 + d) = half4(y);
  }
}

template <class TE, class ACC, bool INT, int TM, int TN, int NSG, int EPI>
kernel void vae_gemm(device TE *A [[buffer(0)]], device TE *W [[buffer(1)]], device const float *sa [[buffer(2)]],
                     device const float *sw [[buffer(3)]], device const float *bias [[buffer(4)]],
                     device float *Y [[buffer(5)]], device const float *aux [[buffer(6)]],
                     constant VaeParams &p [[buffer(7)]], device half *Qh [[buffer(8)]], device half *Kh [[buffer(9)]],
                     device half *Vh [[buffer(10)]], device const float *aux2 [[buffer(11)]],
                     uint2 tg0 [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  const uint2 tg = swz(tg0, ng, p.swizzle);
  static_assert(EPI != EPI_QKV || TN == 128, "qkv epilogue needs two heads per tile");
  threadgroup float tile[EPI == EPI_QKV ? TM * 128 : 4];
  auto tA = mk<TE>(A, p.K, p.M);
  auto tW = mk<TE>(W, p.K, p.N);
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  int m0 = tg.y * TM, n0 = tg.x * TN;
  if (m0 + TM <= p.M && n0 + TN <= p.N) {
    auto a = tA.template slice<dynamic_extent, TM>(0, m0);
    auto b = tW.template slice<dynamic_extent, TN>(0, n0);
    vae_tile<INT, EPI, TM, TN, decltype(op), decltype(a), decltype(b), ACC>(op, a, b, m0, n0, sa, sw, bias, Y, aux, p,
                                                                          Qh, Kh, Vh, aux2, tile, sg, lane, NSG);
  } else {
    auto a = tA.slice(0, m0);
    auto b = tW.slice(0, n0);
    vae_tile<INT, EPI, TM, TN, decltype(op), decltype(a), decltype(b), ACC>(op, a, b, m0, n0, sa, sw, bias, Y, aux, p,
                                                                          Qh, Kh, Vh, aux2, tile, sg, lane, NSG);
  }
}
#define GEMM_ARGS(TE)                                                                                            \
  device TE *, device TE *, device const float *, device const float *, device const float *, device float *, \
      device const float *, constant VaeParams &, device half *, device half *, device half *, device const float *, \
      uint2, uint2, ushort, ushort
#define GEMM_I8(TM, EPI, NAME) \
  template [[host_name(NAME)]] [[kernel]] void vae_gemm<int8_t, int, true, TM, H3_VAE_TN, H3_VAE_NSG, EPI>(GEMM_ARGS(int8_t));
#define GEMM_F16(TM, EPI, NAME) \
  template [[host_name(NAME)]] [[kernel]] void vae_gemm<half, float, false, TM, H3_VAE_TN, H3_VAE_NSG, EPI>(GEMM_ARGS(half));
GEMM_I8(H3_VAE_QKV_TM, EPI_QKV, "vae_qkv_i8")
GEMM_F16(H3_VAE_QKV_TM, EPI_QKV, "vae_qkv_f16")
GEMM_I8(H3_VAE_TM, EPI_RES, "vae_residual_i8")
GEMM_F16(H3_VAE_TM, EPI_RES, "vae_residual_f16")
GEMM_F16(H3_VAE_TM, EPI_OUT, "vae_proj_f16")

// fc1 + SwiGLU (two matmuls with one descriptor: gate rows [n0, n0 + TNH) and up rows [F + n0, ...)).
template <class TE, class ACC, bool INT, int TM, int TNH, int NSG>
kernel void vae_glu(device TE *A [[buffer(0)]], device TE *W [[buffer(1)]], device const float *sa [[buffer(2)]],
                    device const float *sw [[buffer(3)]], device const float *bias [[buffer(4)]],
                    device half *Hd [[buffer(5)]], constant VaeParams &p [[buffer(6)]],
                    uint2 tg0 [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]]) {
  const uint2 tg = swz(tg0, ng, p.swizzle);
  auto tA = mk<TE>(A, p.K, p.M);
  auto tW = mk<TE>(W, p.K, 2 * p.F);
  constexpr auto desc = matmul2d_descriptor(TM, TNH, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  const int m0 = tg.y * TM, n0 = tg.x * TNH;
  auto bg = tW.template slice<dynamic_extent, TNH>(0, n0);
  auto bu = tW.template slice<dynamic_extent, TNH>(0, p.F + n0);
  auto body = [&](auto a) {
    auto cg = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg), ACC>();
    auto cu = op.template get_destination_cooperative_tensor<decltype(a), decltype(bg), ACC>();
#pragma unroll
    for (ushort i = 0; i < cg.get_capacity(); ++i)
      if (cg.is_valid_element(i)) { cg[i] = ACC(0); cu[i] = ACC(0); }
    op.run(a, bg, cg);
    op.run(a, bu, cu);
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
        run4 = a3[1] == a0[1] && a3[0] == a0[0] + 3 && row < p.M && col + 3 < p.F && (col & 3) == 0;
      }
      if (run4) {
        float4 g = float4(float(cg[i0]), float(cg[i0 + 1]), float(cg[i0 + 2]), float(cg[i0 + 3]));
        float4 u = float4(float(cu[i0]), float(cu[i0 + 1]), float(cu[i0 + 2]), float(cu[i0 + 3]));
        if (INT) {
          float s = sa[row];
          g *= s * ld4(sw + col);
          u *= s * ld4(sw + p.F + col);
        }
        g += ld4(bias + col);
        u += ld4(bias + p.F + col);
        float4 h = g * u / (1.0f + precise::exp(-g));
        *(device half4 *)(Hd + ulong(row) * p.F + col) = half4(h);
      } else {
        for (ushort i = i0; i < i0 + 4 && i < cap; ++i) {
          if (!cg.is_valid_element(i)) continue;
          auto idx = cg.get_multidimensional_index(i);
          int r = m0 + idx[1], cc = n0 + idx[0];
          if (r >= p.M || cc >= p.F) continue;
          float g = float(cg[i]), u = float(cu[i]);
          if (INT) {
            g *= sa[r] * sw[cc];
            u *= sa[r] * sw[p.F + cc];
          }
          g += bias[cc];
          u += bias[p.F + cc];
          Hd[ulong(r) * p.F + cc] = half(g * u / (1.0f + precise::exp(-g)));
        }
      }
    }
  };
  if (m0 + TM <= p.M) body(tA.template slice<dynamic_extent, TM>(0, m0));
  else body(tA.slice(0, m0));
}
#define GLU_ARGS(TE)                                                                                       \
  device TE *, device TE *, device const float *, device const float *, device const float *, device half *, \
      constant VaeParams &, uint2, uint2
template [[host_name("vae_glu_i8")]] [[kernel]] void
vae_glu<int8_t, int, true, H3_VAE_TM, H3_VAE_TN, H3_VAE_NSG>(GLU_ARGS(int8_t));
template [[host_name("vae_glu_f16")]] [[kernel]] void
vae_glu<half, float, false, H3_VAE_TM, H3_VAE_TN, H3_VAE_NSG>(GLU_ARGS(half));

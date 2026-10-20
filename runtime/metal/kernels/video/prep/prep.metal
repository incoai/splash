// Native prep kernels. Their arithmetic reproduces the MLX 0.32.1 primitives the former Python prep ran (and so the
// golden prepared inputs) bit for bit: this file is compiled with -fno-fast-math like MLX's metallib, every MLX
// primitive keeps its own rounding step, and the matrix products issue the same matmul2d(16, 32, 16) fragment
// operations in the same K order with the same register layout as MLX's NAX kernels (steel/gemm/nax.h). The
// independent re-implementation is checked op by op against MLX dumps (dev/tools/prep_ref.py, build/dev/prep_check).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#include "metal/abi/video/PrepABI.h"

using namespace metal;

namespace {

// Lane -> (column, row) of a lane's first element in a 16 x 16 fragment: elements (row + 8 i, column + j), i < 2,
// j < 4, element index 4 i + j.
inline short2 frag_coord(ushort lane) {
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  return short2(fn, fm);
}

// C[16 x 32] += A[16 x 16] B: one matmul2d(16, 32, 16) on register fragments (B given as two 16 x 16 fragments: rows
// of B^T when TB, else column blocks of B).
template <typename CT, typename AT, typename BT, bool TB>
inline void mma(thread CT *c0, thread CT *c1, thread const AT *a, thread const BT *b0, thread const BT *b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, TB, true,
                                                             mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
  auto cb = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
  auto cc = op.template get_destination_cooperative_tensor<remove_addrspace_t<decltype(ca)>,
                                                            remove_addrspace_t<decltype(cb)>, CT>();
  for (short i = 0; i < 8; ++i) ca[i] = a[i];
  for (short i = 0; i < 8; ++i) {
    cb[i] = b0[i];
    cb[8 + i] = b1[i];
  }
  for (short i = 0; i < 8; ++i) {
    cc[i] = c0[i];
    cc[8 + i] = c1[i];
  }
  op.run(ca, cb, cc);
  for (short i = 0; i < 8; ++i) {
    c0[i] = cc[i];
    c1[i] = cc[8 + i];
  }
}

// bfloat math as MLX's bf16_math.h: evaluate in float, round to bfloat.
inline bfloat babs(bfloat x) { return static_cast<bfloat>(metal::fabs(static_cast<float>(x))); }
inline bfloat bexp(bfloat x) { return static_cast<bfloat>(metal::fast::exp(static_cast<float>(x))); }

// MLX Sigmoid (unary_ops.h) as it runs inside nn.silu, an mx.compile'd function: MLX builds compiled kernels from
// source at run time (MTLMathModeSafe), where metal::exp evaluates as metal::fast::exp (measured: the precompiled
// Sigmoid kernel and metal::precise::exp differ from nn.silu in 63 of 80640 timestep-MLP values, fast::exp in none).
template <typename T>
inline T sigmoid(T x) {
  auto y = 1 / (1 + metal::fast::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
template <>
inline bfloat sigmoid<bfloat>(bfloat x) {
  auto y = 1 / (1 + bexp(babs(x)));
  return (x < 0) ? y : 1 - y;
}

}  // namespace

// ---- matrix products ----------------------------------------------------------------------------------------------

// One simdgroup per 16 x 32 output block (grid x: N / 32, y: ceil(M / 16)).
template <typename T, bool BIAS>
kernel void prep_gemm(device const T *A [[buffer(0)]], device const T *B [[buffer(1)]], device const T *bias [[buffer(2)]],
                      device T *D [[buffer(3)]], constant PrepGemmParams &p [[buffer(4)]],
                      uint2 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  const int n0 = tg.x * 32, m0 = tg.y * 16;
  const short2 sc = frag_coord(lane);
  float c0[8], c1[8];
  for (short i = 0; i < 8; ++i) c0[i] = c1[i] = 0.0f;
  for (int k = 0; k < p.K; k += 16) {
    T a[8], b0[8], b1[8];
    for (short i = 0; i < 2; ++i) {
      const int r = m0 + sc.y + 8 * i;
      device const T *ar = A + long(r) * p.lda + k + sc.x;
      device const T *br0 = B + long(n0 + sc.y + 8 * i) * p.ldb + k + sc.x;
      device const T *br1 = B + long(n0 + 16 + sc.y + 8 * i) * p.ldb + k + sc.x;
      for (short j = 0; j < 4; ++j) {
        a[4 * i + j] = r < p.M ? ar[j] : T(0);
        b0[4 * i + j] = br0[j];
        b1[4 * i + j] = br1[j];
      }
    }
    mma<float, T, T, true>(c0, c1, a, b0, b1);
  }
  for (short i = 0; i < 2; ++i) {
    const int r = m0 + sc.y + 8 * i;
    if (r >= p.M) continue;
    for (short j = 0; j < 4; ++j) {
      const int col = n0 + sc.x + j;
      float v0 = c0[4 * i + j], v1 = c1[4 * i + j];
      if (BIAS) {
        v0 += static_cast<float>(bias[col]);
        v1 += static_cast<float>(bias[col + 16]);
      }
      D[long(r) * p.ldd + col] = static_cast<T>(v0);
      D[long(r) * p.ldd + col + 16] = static_cast<T>(v1);
    }
  }
}

#define GEMM(T, BIAS, NAME)                                                                                       \
  template [[host_name(NAME)]] [[kernel]] void prep_gemm<T, BIAS>(device const T *, device const T *,             \
                                                                   device const T *, device T *,                   \
                                                                   constant PrepGemmParams &, uint2, ushort);
GEMM(float, true, "prep_gemm_f32_bias")
GEMM(bfloat, true, "prep_gemm_bf16_bias")
GEMM(bfloat, false, "prep_gemm_bf16")

// MLX GemvWide<bfloat, NV, KL> (gemv.h), with the axpby bias epilogue when BIAS: k_lanes lanes reduce K for one
// output row, 32 / KL rows per simdgroup, KL / 8 simdgroups per threadgroup, vectors in chunks of NV.
template <int NV, int KL, bool BIAS>
kernel void prep_gemv_wide(device const bfloat *mat [[buffer(0)]], device const bfloat *in_vec [[buffer(1)]],
                           device const bfloat *bias [[buffer(2)]], device bfloat *out_vec [[buffer(3)]],
                           constant PrepGemvParams &p [[buffer(4)]], uint3 tid [[threadgroup_position_in_grid]],
                           uint3 tgpg [[threadgroups_per_grid]], uint simd_gid [[simdgroup_index_in_threadgroup]],
                           uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int unroll = 8;
  constexpr int num_simdgroups = KL / 8;
  constexpr int rows_per_simdgroup = 32 / KL;
  constexpr int rows_per_tg = rows_per_simdgroup * num_simdgroups;
  const int in_vec_size = p.K, out_vec_size = p.N, M = p.M;
  const short k_lane = simd_lid % KL;
  const short sg_row = simd_lid / KL;
  const int out_row = tid.y * rows_per_tg + rows_per_simdgroup * simd_gid + sg_row;
  const int row = min(out_row, out_vec_size - 1);
  const device bfloat *wrow = mat + int64_t(row) * p.ld;
  const device vec<bfloat, 4> *w4 = (const device vec<bfloat, 4> *)wrow;
  const int n_v4 = in_vec_size / 4;
  const int n_main = n_v4 - n_v4 % (KL * unroll);
  const device vec<bfloat, 4> *x4 = (const device vec<bfloat, 4> *)in_vec;
  const int x_ld4 = p.ldx / 4;
  const int n_chunks = (M + NV - 1) / NV;
  for (int chunk = tid.x; chunk < n_chunks; chunk += tgpg.x) {
    const int vec0 = chunk * NV;
    int x_off4[NV];
    for (int v = 0; v < NV; v++) x_off4[v] = min(vec0 + v, M - 1) * x_ld4;
    float result[NV] = {0};
    for (int base = 0; base < n_main; base += KL * unroll) {
      vec<float, 4> wf[unroll];
      for (int i = 0; i < unroll; i++) wf[i] = vec<float, 4>(w4[base + i * KL + k_lane]);
      for (int v = 0; v < NV; v++) {
        float acc = 0;
        for (int i = 0; i < unroll; i++) acc += dot(wf[i], vec<float, 4>(x4[x_off4[v] + base + i * KL + k_lane]));
        result[v] += acc;
      }
    }
    for (int idx = n_main + k_lane; idx < n_v4; idx += KL) {
      const vec<float, 4> wf = vec<float, 4>(w4[idx]);
      for (int v = 0; v < NV; v++) result[v] += dot(wf, vec<float, 4>(x4[x_off4[v] + idx]));
    }
    for (int v = 0; v < NV; v++)
      for (ushort off = KL / 2; off >= 1; off >>= 1) result[v] += simd_shuffle_down(result[v], off);
    if (k_lane == 0 && out_row < out_vec_size) {
      for (int v = 0; v < NV; v++) {
        if (vec0 + v < M) {
          const int64_t out_idx = int64_t(vec0 + v) * out_vec_size + out_row;
          if (BIAS) {
            const float bias_val = static_cast<float>(bias[out_row]);
            out_vec[out_idx] = static_cast<bfloat>(p.alpha * result[v] + p.beta * bias_val);
          } else {
            out_vec[out_idx] = static_cast<bfloat>(result[v]);
          }
        }
      }
    }
  }
}

#define GEMV1(NV, KL, BIAS, SUFFIX)                                                                                \
  template [[host_name("prep_gemv_wide_" #NV "_" #KL SUFFIX)]] [[kernel]] void prep_gemv_wide<NV, KL, BIAS>(        \
      device const bfloat *, device const bfloat *, device const bfloat *, device bfloat *,                         \
      constant PrepGemvParams &, uint3, uint3, uint, uint);
#define GEMV(NV, KL) GEMV1(NV, KL, true, "_bias") GEMV1(NV, KL, false, "")
GEMV(2, 32)
GEMV(3, 32)
GEMV(4, 32)
GEMV(5, 32)
GEMV(3, 16)
GEMV(4, 16)
GEMV(5, 16)

// ---- normalisation and attention ----------------------------------------------------------------------------------

// MLX rms_looped (axis > 4096; one threadgroup per row, N_READS 4) and rms_single_row (axis <= 4096).
kernel void prep_rms_looped(device const bfloat *x [[buffer(0)]], device const bfloat *w [[buffer(1)]],
                            device bfloat *out [[buffer(2)]], constant PrepRmsParams &p [[buffer(3)]],
                            uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                            uint lsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4, SIMD_SIZE = 32;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];
  const uint axis_size = p.axis;
  float acc = 0;
  x += gid * size_t(p.in_ld) + lid * N_READS;
  w += lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          acc += xi * xi;
        }
      }
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  out += gid * size_t(p.out_ld) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++)
        out[r + i] = w[i + r] * static_cast<bfloat>(x[r + i] * local_inv_mean[0]);
    } else {
      for (int i = 0; i < N_READS; i++)
        if ((r + lid * N_READS + i) < axis_size) out[r + i] = w[i + r] * static_cast<bfloat>(x[r + i] * local_inv_mean[0]);
    }
  }
}

kernel void prep_rms_row(device const bfloat *x [[buffer(0)]], device const bfloat *w [[buffer(1)]],
                         device bfloat *out [[buffer(2)]], constant PrepRmsParams &p [[buffer(3)]],
                         uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                         uint simd_lane_id [[thread_index_in_simdgroup]],
                         uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4, SIMD_SIZE = 32;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[SIMD_SIZE];
  const uint axis_size = p.axis;
  float acc = 0;
  float thread_x[N_READS];
  x += gid * size_t(p.in_ld) + lid * N_READS;
  w += lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      acc += thread_x[i] * thread_x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = (lid * N_READS + i < axis_size) ? (float)x[i] : 0;
      acc += thread_x[i] * thread_x[i];
    }
  }
  acc = simd_sum(acc);
  if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + p.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  out += gid * size_t(p.out_ld) + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) out[i] = w[i] * static_cast<bfloat>(thread_x[i] * local_inv_mean[0]);
  } else {
    for (int i = 0; i < N_READS; i++)
      if ((lid * N_READS + i) < axis_size) out[i] = w[i] * static_cast<bfloat>(thread_x[i] * local_inv_mean[0]);
  }
}

// MLX steel_attention_nax<bfloat, bq 64, bk 32, bd 128, wm 4, wn 1>, no mask, not causal: grid (NQ, heads), 4
// simdgroups of 16 query rows each.
kernel void prep_attention(device const bfloat *Q [[buffer(0)]], device const bfloat *K [[buffer(1)]],
                           device const bfloat *V [[buffer(2)]], device bfloat *O [[buffer(3)]],
                           constant PrepAttnParams &p [[buffer(4)]], uint simd_group_id [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr int BQ = 64, BK = 32, kU = 16, TD = 8, TK = 2;
  Q += long(tid.y) * p.q_h + long(tid.x) * BQ * p.q_l;
  K += long(tid.y) * p.k_h;
  V += long(tid.y) * p.v_h;
  O += long(tid.y) * p.o_h + long(tid.x) * BQ * p.o_l;
  const float scale2 = p.scale * 1.44269504089f;
  const short tm = kU * simd_group_id;
  Q += tm * p.q_l;
  const short2 sc = frag_coord(lane);
  const short sn = sc.x;
  float Ot[TD][8];
  for (short d = 0; d < TD; ++d)
    for (short e = 0; e < 8; ++e) Ot[d][e] = 0.0f;
  float max_score[2], sum_score[2] = {0, 0};
  for (short i = 0; i < 2; ++i) max_score[i] = -metal::numeric_limits<float>::max();
  const bool is_last_q = int(tid.x) == p.NQ_aligned;
  const short lim_rows_q = p.qL_rem - tm;
  const short lim_rows_k = p.kL_rem;
  // fragment load of 16 rows x 16 columns at src (row stride ld), rows >= lim (if limited) read as zero
  auto load = [&](thread auto *dst, device const bfloat *src, int ld, bool limited, int lim) {
    for (short i = 0; i < 2; ++i) {
      const int r = sc.y + 8 * i;
      for (short j = 0; j < 4; ++j)
        dst[4 * i + j] = (!limited || r < lim) ? static_cast<bfloat>(src[long(r) * ld + sc.x + j]) : bfloat(0);
    }
  };
  for (int kb = 0; kb < p.NK; kb++) {
    const bool is_last_k = kb == p.NK_aligned;
    float S[TK][8];
    for (short t = 0; t < TK; ++t)
      for (short e = 0; e < 8; ++e) S[t][e] = 0.0f;
    for (short id = 0; id < TD; id++) {
      bfloat qf[8], k0[8], k1[8];
      load(qf, Q + id * kU, p.q_l, !p.align_q && is_last_q, lim_rows_q);
      const bool lk = !p.align_k && is_last_k;
      load(k0, K + id * kU, p.k_l, lk, lim_rows_k);
      load(k1, K + kU * long(p.k_l) + id * kU, p.k_l, lk, lim_rows_k - kU);
      mma<float, bfloat, bfloat, true>(S[0], S[1], qf, k0, k1);
    }
    for (short t = 0; t < TK; ++t)
      for (short e = 0; e < 8; ++e) S[t][e] *= scale2;
    if (!p.align_k && is_last_k) {
      const float neg_inf = -metal::numeric_limits<float>::max();
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        for (short ii = 0; ii < 2; ii++)
          for (short jj = 0; jj < 4; jj++) {
            const short loc = ii * 4 + jj;
            S[ik][loc] = ((col_pos + jj) < p.kL_rem) ? S[ik][loc] : neg_inf;
          }
      }
    }
    float new_max[2], factor[2];
    for (short i = 0; i < 2; ++i) new_max[i] = max_score[i];
    for (short t = 0; t < TK; ++t)  // row max
      for (short i = 0; i < 2; i++) {
        float thr = metal::max(metal::max(S[t][4 * i + 0], S[t][4 * i + 1]), metal::max(S[t][4 * i + 2], S[t][4 * i + 3]));
        float qgr = simd_shuffle_xor(thr, ushort(1));
        qgr = metal::max(thr, qgr);
        float sgr = simd_shuffle_xor(qgr, ushort(8));
        sgr = metal::max(qgr, sgr);
        new_max[i] = metal::max(new_max[i], sgr);
      }
    for (short t = 0; t < TK; ++t)
      for (short i = 0; i < 2; i++)
        for (short j = 0; j < 4; j++) S[t][4 * i + j] = fast::exp2(S[t][4 * i + j] - new_max[i]);
    for (short i = 0; i < 2; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    for (short i = 0; i < 2; ++i) sum_score[i] = sum_score[i] * factor[i];
    for (short t = 0; t < TK; ++t)  // row sum
      for (short i = 0; i < 2; i++) {
        float thr = (S[t][4 * i + 0] + S[t][4 * i + 1]) + (S[t][4 * i + 2] + S[t][4 * i + 3]);
        float qgr = simd_shuffle_xor(thr, ushort(1));
        qgr = thr + qgr;
        float sgr = simd_shuffle_xor(qgr, ushort(8));
        sgr = qgr + sgr;
        sum_score[i] = sum_score[i] + sgr;
      }
    for (short d = 0; d < TD; ++d)
      for (short i = 0; i < 2; i++)
        for (short j = 0; j < 4; j++) Ot[d][4 * i + j] = Ot[d][4 * i + j] * factor[i];
    simdgroup_barrier(mem_flags::mem_none);
    for (short id = 0; id < TD; id += 2) {
      if (id == 4) threadgroup_barrier(mem_flags::mem_none);
      for (short ik = 0; ik < TK; ik++) {
        bfloat v0[8], v1[8];
        const bool lk = !p.align_k && is_last_k;
        device const bfloat *vb = V + long(ik * kU) * p.v_l + id * kU;
        load(v0, vb, p.v_l, lk, lim_rows_k - ik * kU);
        load(v1, vb + kU, p.v_l, lk, lim_rows_k - ik * kU);
        mma<float, float, bfloat, false>(Ot[id], Ot[id + 1], S[ik], v0, v1);
      }
    }
    K += BK * long(p.k_l);
    V += BK * long(p.v_l);
  }
  threadgroup_barrier(mem_flags::mem_none);
  float rcp[2];
  for (short i = 0; i < 2; ++i) rcp[i] = 1.f / sum_score[i];
  for (short d = 0; d < TD; ++d)
    for (short i = 0; i < 2; i++)
      for (short j = 0; j < 4; j++) Ot[d][4 * i + j] = Ot[d][4 * i + j] * rcp[i];
  O += tm * long(p.o_l);
  if (!p.align_q && is_last_q && lim_rows_q <= 0) return;
  for (short d = 0; d < TD; ++d)
    for (short i = 0; i < 2; i++) {
      const int r = sc.y + 8 * i;
      if (!p.align_q && is_last_q && r >= lim_rows_q) continue;
      for (short j = 0; j < 4; j++) O[long(r) * p.o_l + d * kU + sc.x + j] = static_cast<bfloat>(Ot[d][4 * i + j]);
    }
}

// ---- elementwise --------------------------------------------------------------------------------------------------

// nn.silu = x * sigmoid(x) (one compiled MLX kernel), fp32
kernel void prep_silu_f32(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                          constant PrepElemParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (int(i) >= p.n) return;
  const float v = x[i];
  const float s = sigmoid(v);
  y[i] = v * s;
}

// silu(fused[..., :ffn]) * fused[..., ffn:] (compiled silu in bfloat, then an MLX bfloat multiply)
kernel void prep_swiglu_bf16(device const bfloat *f [[buffer(0)]], device bfloat *y [[buffer(1)]],
                             constant PrepElemParams &p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) >= p.ffn) return;
  device const bfloat *row = f + long(gid.y) * 2 * p.ffn;
  const bfloat g = row[gid.x];
  const bfloat s = g * sigmoid(g);
  y[long(gid.y) * p.ffn + gid.x] = s * row[p.ffn + gid.x];
}

kernel void prep_add_bf16(device const bfloat *a [[buffer(0)]], device const bfloat *b [[buffer(1)]],
                          device bfloat *y [[buffer(2)]], constant PrepElemParams &p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (int(i) >= p.n) return;
  y[i] = a[i] + b[i];
}

kernel void prep_cast_bf16(device const float *x [[buffer(0)]], device bfloat *y [[buffer(1)]],
                           constant PrepElemParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (int(i) >= p.n) return;
  y[i] = static_cast<bfloat>(x[i]);
}

// RotaryPosEmbed3D: inv_freq = 1 / 10000^(arange(0, 32, 2) / 32) (16 values), then cos / sin of pos * inv_freq.
kernel void prep_rope_inv_freq(device float *inv [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i >= 16) return;
  const float e = static_cast<float>(2 * i) / 32.0f;
  const float pw = metal::pow(10000.0f, e);
  inv[i] = 1.0f / pw;
}

kernel void prep_rope_trig(device const float *pos [[buffer(0)]], device const float *inv [[buffer(1)]],
                           device float *cs [[buffer(2)]], device float *sn [[buffer(3)]],
                           constant PrepRopeParams &p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x, r = gid.y;
  if (c >= 48 || r >= p.rows) return;
  const float f = pos[long(r) * 3 + c / 16] * inv[c % 16];
  cs[long(r) * 48 + c] = metal::precise::cos(f);
  sn[long(r) * 48 + c] = metal::precise::sin(f);
}

// timestep_embedding(t, 256): exponent = -ln(1e4) * arange(128) / 128, emb = exp(exponent), [cos(t emb) | sin(t emb)]
kernel void prep_temb_sinusoid(device const float *t [[buffer(0)]], device float *y [[buffer(1)]],
                               constant PrepTembParams &p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
  const int i = gid.x, r = gid.y;
  if (i >= 128 || r >= p.T) return;
  const float ln = -9.210340371976184f;
  float e = ln * static_cast<float>(i);
  e = e / 128.0f;
  e = metal::precise::exp(e);
  const float a = t[r] * e;
  y[long(r) * 256 + i] = metal::precise::cos(a);
  y[long(r) * 256 + 128 + i] = metal::precise::sin(a);
}

// MLX random.normal on threefry bits: uniform(nextafter(-1, 0), 1) = range * min(bits / 2^32, 1 - 2^-24) + lo,
// erfinv (erf.h), x sqrt(2).
kernel void prep_normal(device const uint32_t *bits [[buffer(0)]], device float *y [[buffer(1)]],
                        constant PrepNormalParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (int(i) >= p.n) return;
  const float maxval = 4294967296.0f;
  const float upper = 0x1.fffffep-1f;
  float u = static_cast<float>(bits[i]);
  u = u / maxval;
  u = metal::isnan(u) ? u : (u < upper ? u : upper);
  float s = p.range * u;
  s = s + p.lo;
  // erfinv
  const float a = s;
  auto t = metal::fma(a, 0.0f - a, 1.0f);
  t = metal::log(t);
  float q;
  if (metal::abs(t) > 6.125f) {
    q = 3.03697567e-10f;
    q = metal::fma(q, t, 2.93243101e-8f);
    q = metal::fma(q, t, 1.22150334e-6f);
    q = metal::fma(q, t, 2.84108955e-5f);
    q = metal::fma(q, t, 3.93552968e-4f);
    q = metal::fma(q, t, 3.02698812e-3f);
    q = metal::fma(q, t, 4.83185798e-3f);
    q = metal::fma(q, t, -2.64646143e-1f);
    q = metal::fma(q, t, 8.40016484e-1f);
  } else {
    q = 5.43877832e-9f;
    q = metal::fma(q, t, 1.43285448e-7f);
    q = metal::fma(q, t, 1.22774793e-6f);
    q = metal::fma(q, t, 1.12963626e-7f);
    q = metal::fma(q, t, -5.61530760e-5f);
    q = metal::fma(q, t, -1.47697632e-4f);
    q = metal::fma(q, t, 2.31468678e-3f);
    q = metal::fma(q, t, 1.15392581e-2f);
    q = metal::fma(q, t, -2.32015476e-1f);
    q = metal::fma(q, t, 8.86226892e-1f);
  }
  const float e = a * q;
  const float sqrt2 = 1.41421356237309504880f;
  y[i] = sqrt2 * e;
}

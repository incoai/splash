#pragma once

// Native prep kernels (kernels/prep): host / kernel structs. These kernels reproduce MLX 0.32.1's arithmetic for the
// request-constant inputs (refined text, adaLN tables, RoPE, noise) bit for bit; see runtime/model/Prep.hpp.
#include "metal/abi/video/Layout.h"

// prep_gemm_*: D[M][N] = A[M][K] B[N][K]^T (+ bias[N]): MLX steel_gemm_fused_nax arithmetic (per 16 x 32 output block,
// K in steps of 16 through matmul2d(16, 32, 16, relaxed precision), bias added in fp32, one rounding to T).
struct PrepGemmParams {
  int32_t M, N, K, lda, ldb, ldd;
};

// prep_gemv_wide_*: out[M][N] = alpha x[M][K] mat[N][K]^T + beta bias[N] for M <= 15 (MLX gemv_wide).
struct PrepGemvParams {
  int32_t K, N, M, ld, ldx;
  float alpha, beta;
};

// prep_rms_*: rows of `axis` bf16 values, weight bf16 [axis]; in / out row strides in elements.
struct PrepRmsParams {
  float eps;
  uint32_t axis;
  int32_t in_ld, out_ld;
};

// prep_attention: one head per grid y, 64-row query blocks per grid x (MLX steel_attention_nax, bq 64, bk 32, bd 128).
struct PrepAttnParams {
  int32_t qL, kL, NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem;
  float scale;
  int32_t q_h, q_l, k_h, k_l, v_h, v_l, o_h, o_l;  // strides (elements) per head and per row
  int32_t align_q, align_k;
};

// elementwise kernels (prep_silu_*, prep_add_bf16, prep_cast_bf16, prep_swiglu_bf16): n elements; swiglu reads gate
// and value halves of rows of 2 * ffn.
struct PrepElemParams {
  int32_t n, ffn, ld, pad;
};

// prep_rope_trig: cos / sin of pos[row][axis] * inv_freq[f] for 48 columns (axis = c / 16, f = c % 16).
struct PrepRopeParams {
  int32_t rows, pad0, pad1, pad2;
};

// prep_temb_sinusoid: [T][256] = [cos(t e) | sin(t e)], e[i] = exp(-ln(1e4) i / 128).
struct PrepTembParams {
  int32_t T, pad0, pad1, pad2;
};

// prep_normal: uint32 bits -> MLX random.normal (uniform in (-1, 1), erfinv, x sqrt 2), fp32.
struct PrepNormalParams {
  int32_t n, pad0;
  float lo, range;
};

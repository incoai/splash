#pragma once

// DiT W8A8 linears: Hadamard int8 producers, the QKV GEMM with its QK-norm / RoPE epilogue, the gated-residual GEMM
// and fc1 + SwiGLU. Parameter blocks shared by ops/Linear and kernels/video/dit/{producer,linear}.metal.
#include "metal/abi/video/Layout.h"

// One MPP matmul2d tile shape for every DiT GEMM (TM x TN, execution_simdgroups<NSG>).
#define H3_LINEAR_TM 64
#define H3_LINEAR_TN 128
#define H3_LINEAR_NSG 4
// Producers: a thread holds 8 channels of G 128-groups (threads per row = 16 * K / 128 / G).
#define H3_PRODUCER_GROUPS_HIDDEN 3  // K = 5376 -> 224 threads
#define H3_PRODUCER_GROUPS_FFN 8     // K = 14336 -> 224 threads
#define H3_PRODUCER_GROUPS_INNER 4   // K = 7168 -> 224 threads

// dit_producer_norm / dit_producer_plain. The adaLN table is bf16 [modality][6][hidden]; ada_stride is the distance
// between modality rows, shift_off / scale_off pick the two rows of the table.
struct ProducerParams {
  int32_t M, K, ada_stride, shift_off;
  int32_t scale_off;
  float eps;
};

// dit_producer_gated (softmax gate x attention output, K = 7168) and dit_producer_branch (linear-branch readout of
// the inner video rows; signs 7168..14335 of the 14336-wide sequence: sign_off = 7168).
struct GatedProducerParams {
  int32_t M, K, sign_off, ldl;
  int32_t lg;  // readout groups present (all 56)
};

// dit_qkv: QKV GEMM with dequant -> bf16 -> per-head RMS QK-norm -> RoPE -> fp16 head-major Q / K / V, plus the
// pre-norm q / k ("raw", fp16) for the linear branch, stored for text rows and inner video rows only:
// rows [0, rtext) keep their index, rows [rv0, rv1) go to rtext + (row - rv0).
// FastH3 (VSA) blocks extend N with a fourth part, the compression gate to_gate_compress: its bf16-rounded output is
// stored fp16 row-major to G [M][ldg]. part0 is the part of the first 128-wide output tile (3: gate-only launch).
struct QkvParams {
  int32_t M, N, K, H;
  int32_t Spad, row0, flags, rspad;  // flags bit0: store raw q / k; bit1: compact raw rows (always both)
  int32_t swizzle, rtext, rv0, rv1;  // swizzle: M-blocks per tile group
  float eps;
  int32_t part0, ldg;
};

// dit_gated_residual: X[row] += gate[modality(row)] * dequant(A W^T), bf16 in place.
struct GatedResidualParams {
  int32_t M, N, K, ldo;
  int32_t ada_stride, gate_off, swizzle;
};

// dit_fc1_swiglu: h = silu(gate) * value from one [2F, K] weight (gate rows first), bf16 out.
struct SwiGluParams {
  int32_t M, K, F;
};

// Runtime LoRA of an installed adapter: y = W x + scale B (A x) of an adapted linear.
//   dit_lora_down / dit_lora_down_f16  T = scale (A x) fp16 [Mpad][ldt] of the GEMM's input rows: int8 rows times
//                                      their per-row scale (A rotated as the int8 weights), or fp16 rows (QKFIX); A bf16
//                                      [R][K], R = segments x rank
//   *_lora GEMMs                       the int8 GEMM's fp32 output plus T[:, seg rank ..] B^T (B bf16 [N][rank],
//                                      fp32 accumulation), before the epilogue; seg = output column / seg_n
#define H3_LORA_RANK 128
struct LoraDownParams {
  int32_t M, K, R, ldt;
  float scale;
  int32_t pad0, pad1, pad2;
};
struct LoraUpParams {
  int32_t ldt, mpad, seg_n, nsegs;  // T's row stride and padded rows; output columns per segment and segments
};

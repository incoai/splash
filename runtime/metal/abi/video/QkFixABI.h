#pragma once

// QKFIX: fp16 Q / K path of the blocks whose QK-norm gains make int8 QK inaccurate (45, 48, 49). The attention
// kernel uses WindowParams / WindowItem (AttentionABI.h).
#include "metal/abi/video/AttentionABI.h"

#define H3_QKFIX_TM 32       // fp16 QKV GEMM tile rows (x 128 columns, 4 simdgroups)
#define H3_QKFIX_SPLIT 4     // post-RoPE dimensions per head kept in fp32 (int32 [4], -1 = unused)

// dit_qkfix_producer: RMSNorm + adaLN modulate -> fp16 x [M][5376]
struct QkFixProducerParams {
  int32_t M, K, ada_stride, shift_off;
  int32_t scale_off;
  float eps;
};

// dit_qkfix_qkv: fp16 q / k / v GEMM with an fp32 QK-norm / RoPE epilogue (raw q / k rows as QkvParams)
struct QkFixQkvParams {
  int32_t M, N, K, H;
  int32_t Spad, row0, flags, rspad;
  int32_t rtext, rv0, rv1, swizzle;
  float eps;
};

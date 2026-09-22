#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct FlashMoEFusedParams {
  uint32_t rows;
  uint32_t selections;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t experts;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint64_t gate_weight_row_stride_bytes;
  uint64_t gate_weight_expert_stride_bytes;
  uint64_t gate_parameter_row_stride_bytes;
  uint64_t gate_parameter_expert_stride_bytes;
  uint64_t up_weight_row_stride_bytes;
  uint64_t up_weight_expert_stride_bytes;
  uint64_t up_parameter_row_stride_bytes;
  uint64_t up_parameter_expert_stride_bytes;
};

static_assert(sizeof(FlashMoEFusedParams) == 96,
              "Flash fused MoE parameters are 96 bytes on host and Metal");

struct FlashMoEDownFusedParams {
  uint32_t rows;
  uint32_t selections;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t experts;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint64_t weight_row_stride_bytes;
  uint64_t weight_expert_stride_bytes;
  uint64_t parameter_row_stride_bytes;
  uint64_t parameter_expert_stride_bytes;
};

static_assert(sizeof(FlashMoEDownFusedParams) == 64,
              "Flash fused expert-down parameters are 64 bytes");

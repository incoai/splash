#pragma once

// Experimental graph parameter layouts shared by host code and Metal.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct ExperimentalINT8Params {
  uint32_t n;
  uint32_t k;
  uint32_t rows;
  uint32_t epilogue; // 0 None, 1 Residual, 2 GateUp, 3 UpWithGate.
};

struct ExperimentalINT8QuantParams {
  uint32_t k;
  uint32_t rows;
};

static_assert(sizeof(ExperimentalINT8Params) == 16,
              "Experimental INT8 projection parameters are four uint32 values");
static_assert(sizeof(ExperimentalINT8QuantParams) == 8,
              "Experimental INT8 quantization parameters are two uint32 values");

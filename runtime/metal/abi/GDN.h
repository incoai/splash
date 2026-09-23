#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct GDNPrefillParams {
  uint32_t tokens;
};

static_assert(sizeof(GDNPrefillParams) == 4,
              "GDN prefill parameters are 4 bytes on both sides");

struct GDNPreparePrefillParams {
  uint32_t tokens;
  uint32_t packed_width;
};

static_assert(sizeof(GDNPreparePrefillParams) == 8,
              "GDN prefill prepare parameters are 8 bytes on both sides");

// tiled_heads (0 or 1) selects the value-head order of the GDN output, the
// out_proj input columns: 0 keeps a key head's value heads adjacent (head h
// at h); 1 is llama.cpp's tiled GGUF order, head h at
// (h % heads per key) * key heads + h / heads per key.
struct GDNGatePrefillParams {
  uint32_t tokens;
  uint32_t packed_width;
  uint32_t tiled_heads;
};

static_assert(sizeof(GDNGatePrefillParams) == 12,
              "GDN prefill gate parameters are 12 bytes on both sides");

struct GDNDecodeBatchParams {
  uint32_t tiled_heads; // As in GDNGatePrefillParams.
  uint32_t packed_width;
  uint32_t lanes;
  uint32_t layer;
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
};

static_assert(sizeof(GDNDecodeBatchParams) == 40,
              "GDN decode parameters are 40 bytes on both sides");

// Explicit padding aligns the uint64_t state strides and keeps transmitted
// bytes initialized.
struct GDNBatchCommitParams {
  uint32_t groups;
  uint32_t packed_width;
  uint32_t lanes;
  uint32_t packed_stride;
  uint32_t mixed_stride;
  uint32_t decay_stride;
  uint32_t beta_stride;
  uint32_t reserved0;
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
};

static_assert(sizeof(GDNBatchCommitParams) == 56,
              "GDN commit parameters are 56 bytes on both sides");

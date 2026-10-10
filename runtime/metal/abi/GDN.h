#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The taps of the GDN short convolution: a channel's input at a token and at
// the SPLASH_GDN_CONVOLUTION_TAPS - 1 tokens before it, which a state cell
// carries as that many rows of every channel.
#define SPLASH_GDN_CONVOLUTION_TAPS 4u

// The packed width, the value heads and the packed, mixed and gate row
// strides of the GDN kernels are constants of their compiled variant and the
// grids give the tasks; the state cells' strides come from the host.

// The prefill prepare and scan kernels.
struct GDNPrefillParams {
  uint32_t tokens;
};

static_assert(sizeof(GDNPrefillParams) == 4,
              "GDN prefill parameters are 4 bytes on both sides");

// tiled_heads bit 0 selects the value-head order of the GDN output, the
// out_proj input columns: 0 keeps a key head's value heads adjacent (head h
// at h); 1 is llama.cpp's tiled GGUF order, head h at
// (h % heads per key) * key heads + h / heads per key. Bit 1 gates the
// normalized output by sigmoid(z) (Qwen3.8-Flash-Next) instead of silu(z).
#define GDN_SIGMOID_GATE 2u
struct GDNGatePrefillParams {
  uint32_t tiled_heads;
};

static_assert(sizeof(GDNGatePrefillParams) == 4,
              "GDN prefill gate parameters are 4 bytes on both sides");

struct GDNDecodeBatchParams {
  uint32_t tiled_heads; // As in GDNGatePrefillParams.
  uint32_t layer;
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
};

static_assert(sizeof(GDNDecodeBatchParams) == 32,
              "GDN decode parameters are 32 bytes on both sides");

struct GDNBatchCommitParams {
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
};

static_assert(sizeof(GDNBatchCommitParams) == 24,
              "GDN commit parameters are 24 bytes on both sides");

#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/ExecutionGeometry.h"

// The taps of the GDN short convolution: a channel's input at a token and at
// the SPLASH_GDN_CONVOLUTION_TAPS - 1 tokens before it, which a state cell
// carries as that many rows of every channel.
#define SPLASH_GDN_CONVOLUTION_TAPS 4u

// The packed width, the value heads and the packed row stride of the GDN
// kernels are constants of their compiled variant and the grids give the
// tasks; the state cells' strides and a decode step's tape offsets come from
// the host.

// The prefill prepare and scan kernels.
struct GDNPrefillParams {
  uint32_t tokens;
};

static_assert(sizeof(GDNPrefillParams) == 4,
              "GDN prefill parameters are 4 bytes on both sides");

// tiled_heads (0 or 1) selects the value-head order of the GDN output, the
// out_proj input columns: 0 keeps a key head's value heads adjacent (head h
// at h); 1 is llama.cpp's tiled GGUF order, head h at
// (h % heads per key) * key heads + h / heads per key.
struct GDNGatePrefillParams {
  uint32_t tiled_heads;
};

static_assert(sizeof(GDNGatePrefillParams) == 4,
              "GDN prefill gate parameters are 4 bytes on both sides");

// A decode step leaves its rows on the lane's tape: one layer's holds, each
// field for every row in turn, the rows' convolution inputs ([conv_dim]
// bf16), their k then v rows ([key_width + value_width] bf16), decay
// ([value_heads] fp32) and beta ([value_heads] bf16). The lane's next step
// folds the rows it retained into the state before its own (GDN::addDecode).
inline constexpr uint64_t gdn_tape_kv_offset(uint32_t conv_dim) {
  return uint64_t(SPLASH_TARGET_VERIFY_ROWS) * conv_dim * 2;
}
inline constexpr uint64_t gdn_tape_decay_offset(uint32_t conv_dim,
                                                uint32_t kv_width) {
  return gdn_tape_kv_offset(conv_dim) +
         uint64_t(SPLASH_TARGET_VERIFY_ROWS) * kv_width * 2;
}
inline constexpr uint64_t gdn_tape_beta_offset(uint32_t conv_dim,
                                               uint32_t kv_width,
                                               uint32_t value_heads) {
  return gdn_tape_decay_offset(conv_dim, kv_width) +
         uint64_t(SPLASH_TARGET_VERIFY_ROWS) * value_heads * 4;
}
// One layer's tape, a multiple of 16 bytes.
inline constexpr uint64_t gdn_tape_layer_bytes(uint32_t conv_dim,
                                               uint32_t kv_width,
                                               uint32_t value_heads) {
  return (gdn_tape_beta_offset(conv_dim, kv_width, value_heads) +
          uint64_t(SPLASH_TARGET_VERIFY_ROWS) * value_heads * 2 + 15) /
         16 * 16;
}

struct GDNDecodeBatchParams {
  uint32_t tiled_heads; // As in GDNGatePrefillParams.
  uint32_t layer;
  uint64_t conv_layer_bytes;
  uint64_t recurrent_layer_bytes;
  uint64_t convolution_state_bytes;
  // Per lane, byte offsets in the tape buffer of this layer's tape of the
  // lane's previous step (the pending rows) and of this step, and how many
  // rows of the previous step the lane retained and its state does not hold
  // yet.
  uint64_t pending_tape[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint64_t step_tape[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t pending_rows[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(GDNDecodeBatchParams) == 112,
              "GDN decode parameters are 112 bytes on both sides");

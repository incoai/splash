#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A cached state stores a GDN layer's recurrent state as int8 codes, each
// value row's values in blocks of this many that share one bf16 scale, the
// block's largest magnitude over 127 (ops::StateCodec). Restoring 41 cached
// states this way changed the 27B's next-token distributions by a mean KL of
// 1.1e-4, below the 1.5e-4 a cold run of the same prompts differs by; a
// scale per value row of 128 had heavier tails on the 35B (KL max 0.15 nats
// against 0.07).
#define SPLASH_STATE_CODEC_BLOCK 32u
// The values one thread of the codec kernels holds: half a block, as one
// lane of the prefill scan holds sixteen key columns of its row, so a
// block's two threads are neighbours in their simdgroup.
#define SPLASH_STATE_CODEC_THREAD_VALUES 16u

// A cached layer of `rows` value rows of `columns` values: every value's
// code, row by row as the running state holds them, then each block's
// scale, in the same order.
inline constexpr uint64_t state_codec_scales_offset(uint32_t rows, uint32_t columns) {
  return uint64_t(rows) * columns;
}
inline constexpr uint64_t state_codec_layer_bytes(uint32_t rows, uint32_t columns) {
  return state_codec_scales_offset(rows, columns) + uint64_t(rows) * (columns / SPLASH_STATE_CODEC_BLOCK) * 2;
}

// One dispatch codes `layers` layers, the grid's y, each running layer
// state_layer_bytes and each cached one codes_layer_bytes after the last;
// threads_per_layer threads per layer, each holding
// SPLASH_STATE_CODEC_THREAD_VALUES consecutive values (ops::StateCodec).
struct StateCodecParams {
  uint64_t state_layer_bytes;
  uint64_t codes_layer_bytes;
  uint64_t scales_offset;
  uint32_t threads_per_layer;
  uint32_t padding;
};

static_assert(sizeof(StateCodecParams) == 32,
              "State codec parameters are 32 bytes on both sides");

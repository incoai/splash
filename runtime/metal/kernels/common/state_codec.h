#pragma once

#include "metal/abi/KernelABI.h"

// The cached form's arithmetic (ops::StateCodec), which the codec's kernels
// and the decode fold that stores a state's codes share, so that both make
// the same codes of a state. A block's scale is its largest magnitude over
// 127, rounded to bf16 as stored; a value's code is the value over the scale
// rounded to nearest even, the division exact, so the codes are what IEEE
// arithmetic makes of them on any device and on the CPU. A block of zeros has
// a zero scale and zero codes.

inline bfloat state_codec_scale(float amax) {
  return bfloat(amax * (1.0f / 127.0f));
}

inline char state_code(float value, float scale) {
  return scale > 0.0f ? char(clamp(precise::rint(precise::divide(value, scale)), -127.0f, 127.0f)) : char(0);
}

// The codes of the four values a lane holds of a value row, whose block
// eight neighbouring lanes hold, four each, as the decode scan's lanes hold
// a row's key columns: each value as a `State` cell stores it, the block's
// scale from the eight lanes' largest magnitude, which the block's first
// lane stores. Every lane of the simdgroup takes part.
template <class State>
inline void state_code_quad(thread const float (&values)[4], device char *codes, device bfloat *scale,
                            uint lane) {
  static_assert(SPLASH_STATE_CODEC_BLOCK == 8 * 4, "eight lanes of four values make a block");
  float stored[4];
  float amax = 0.0f;
  for (uint i = 0; i < 4; ++i) {
    stored[i] = float(State(values[i]));
    amax = max(amax, abs(stored[i]));
  }
  for (ushort distance = 1; distance < 8; distance *= 2)
    amax = max(amax, simd_shuffle_xor(amax, distance));
  const bfloat block = state_codec_scale(amax);
  const float wide = float(block);
  *reinterpret_cast<device char4 *>(codes) =
      char4(state_code(stored[0], wide), state_code(stored[1], wide), state_code(stored[2], wide),
            state_code(stored[3], wide));
  if (lane % 8 == 0)
    *scale = block;
}

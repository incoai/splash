#pragma once

#include "metal/abi/KernelABI.h"

// The cached form's arithmetic (ops::StateCodec), which the codec's kernels
// and the decode fold that stores a state's codes share, so that both make
// the same codes of a state. A block's scale is its largest magnitude over
// 127, rounded to bf16 as stored; a value's code is the value over the scale
// rounded to nearest even, the division exact, so the codes are what IEEE
// arithmetic makes of them on any device and on the CPU. A block of zeros has
// a zero scale and zero codes. A block that holds NaN or an infinity has a
// NaN scale, so every value of it restores as NaN (its code times the scale)
// rather than as a finite value the state never held, which the largest
// finite magnitude made of it: max() dropped the NaN.

// A value's magnitude as its bits, which order magnitudes as their values
// do and put infinity, then NaN, above every finite one, so that the largest
// of a block's says whether it is finite; the kernels build with fast math,
// under which a float comparison need not.
inline uint state_codec_magnitude(float value) {
  return as_type<uint>(value) & 0x7fffffffu;
}

inline bfloat state_codec_scale(uint amax) {
  return amax < 0x7f800000u ? bfloat(as_type<float>(amax) * (1.0f / 127.0f)) : as_type<bfloat>(ushort(0x7fc0));
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
  uint amax = 0;
  for (uint i = 0; i < 4; ++i) {
    stored[i] = float(State(values[i]));
    amax = max(amax, state_codec_magnitude(stored[i]));
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

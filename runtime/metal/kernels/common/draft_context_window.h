#pragma once

#include "metal/abi/KernelABI.h"

// One simdgroup per row and group of the context window: lane l holds the
// group's values 2l and 2l + 1, which share one byte of codes. The code of x
// is round((x - minimum) / scale) with the scale and minimum as stored, so
// loading reads back minimum + code * scale.
struct DraftContextGroup {
  uint row;
  uint group;
  uint groups;
  uint slot;
};

inline bool draft_context_group(constant DraftContextWindowParams &params,
                                uint threadgroup_index, uint simd_group,
                                thread DraftContextGroup &result) {
  result.groups = params.width / SPLASH_DRAFT_CONTEXT_GROUP;
  const uint task = threadgroup_index * SPLASH_DRAFT_CONTEXT_SIMDGROUPS + simd_group;
  result.row = task / result.groups;
  result.group = task % result.groups;
  result.slot =
      (params.start_position + result.row) % SPLASH_DRAFT_SLIDING_WINDOW;
  return result.row < params.rows;
}

// Stores one row and group of the context rows in the slot of its position.
inline void draft_context_store(device const bfloat *context, device uchar *window,
                                constant DraftContextWindowParams &params,
                                DraftContextGroup at, uint lane) {
  device const bfloat *values = context + ulong(at.row) * params.width +
                                at.group * SPLASH_DRAFT_CONTEXT_GROUP + 2 * lane;
  const float2 x = float2(float(values[0]), float(values[1]));
  const float low = simd_min(min(x.x, x.y));
  const float high = simd_max(max(x.x, x.y));
  const half scale = half(precise::divide(high - low, 15.0f));
  const half minimum = half(low);
  uint2 code = uint2(0);
  if (scale > half(0)) {
    const float2 steps = float2(precise::divide(x.x - float(minimum), float(scale)),
                                precise::divide(x.y - float(minimum), float(scale)));
    code = uint2(clamp(rint(steps), 0.0f, 15.0f));
  }
  device uchar *codes = window + ulong(at.slot) * (params.width / 2);
  codes[at.group * (SPLASH_DRAFT_CONTEXT_GROUP / 2) + lane] =
      uchar(code.x | (code.y << 4));
  if (lane == 0) {
    device half2 *groups = reinterpret_cast<device half2 *>(
        window + draft_context_codes_bytes(params.width));
    groups[ulong(at.slot) * at.groups + at.group] = half2(scale, minimum);
  }
}

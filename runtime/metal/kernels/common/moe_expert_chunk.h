#pragma once

#include "metal/abi/MoE.h"
#include <metal_stdlib>

using namespace metal;

// The expert passes and the expert load reach a slot's planes the same way
// (metal/abi/MoE.h): through the table of the chunks' GPU addresses, the
// regions of projection z (0 gate, 1 up, 2 down) in the chunk that holds the
// slot, and the slot's place in that chunk. The floor chunk, chunk 0, holds
// the first floor_slots slots, and each elastic chunk after it chunk_slots
// slots. w1 is the meta region for formats without a second plane. A slot is
// the same for every thread of a threadgroup, so the branch is uniform.
struct MoeExpertChunk {
  device uchar *w0;
  device uchar *w1;
  device uchar *meta;
  uint local;
};

inline MoeExpertChunk moe_expert_chunk(constant ulong *chunks, uint floor_slots, uint chunk_slots, uint slot,
                                       uint z) {
  uint chunk = 0, local = slot;
  if (slot >= floor_slots) {
    const uint elastic = slot - floor_slots;
    chunk = 1 + elastic / chunk_slots;
    local = elastic - (chunk - 1) * chunk_slots;
  }
  constant ulong *regions = chunks + chunk * MOE_EXPERT_CHUNK_ADDRESSES + z * 3;
  return {reinterpret_cast<device uchar *>(regions[0]), reinterpret_cast<device uchar *>(regions[1]),
          reinterpret_cast<device uchar *>(regions[2]), local};
}

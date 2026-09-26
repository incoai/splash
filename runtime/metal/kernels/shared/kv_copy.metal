#include "metal/abi/KvCopy.h"
#include <metal_stdlib>

using namespace metal;

// Moves whole KV pages of one attention layer between the four page buffers
// and host-visible staging, 16 bytes per thread per step. Idle slots and
// out-of-range pages copy nothing.
kernel void kv_copy_pages(device uchar *keyData [[buffer(0)]],
                          device uchar *keyScales [[buffer(1)]],
                          device uchar *valueData [[buffer(2)]],
                          device uchar *valueScales [[buffer(3)]],
                          device uchar *staging [[buffer(4)]],
                          device const SplashKvCopySlot *table [[buffer(5)]],
                          constant SplashKvCopyParams &params [[buffer(6)]],
                          uint slot [[threadgroup_position_in_grid]],
                          uint lane [[thread_position_in_threadgroup]],
                          uint lanes [[threads_per_threadgroup]]) {
  if (slot >= params.staging_slots) return;
  const SplashKvCopySlot entry = table[slot];
  if (entry.direction == SPLASH_KV_COPY_NONE || entry.page >= params.physical_pages) return;
  const bool toStaging = entry.direction == SPLASH_KV_COPY_TO_STAGING;
  device uchar *stage =
      staging + ulong(slot) * params.slot_bytes + params.layer_offset;
  device uchar *ranges[4] = {keyData + ulong(entry.page) * params.data_bytes,
                             keyScales + ulong(entry.page) * params.scale_bytes,
                             valueData + ulong(entry.page) * params.data_bytes,
                             valueScales + ulong(entry.page) * params.scale_bytes};
  const uint sizes[4] = {params.data_bytes, params.scale_bytes,
                         params.data_bytes, params.scale_bytes};
  for (uint range = 0; range < 4; ++range) {
    if (!sizes[range]) continue;
    device uint4 *pageWords = reinterpret_cast<device uint4 *>(ranges[range]);
    device uint4 *stageWords = reinterpret_cast<device uint4 *>(stage);
    const uint words = sizes[range] / 16;
    for (uint word = lane; word < words; word += lanes) {
      if (toStaging) {
        stageWords[word] = pageWords[word];
      } else {
        pageWords[word] = stageWords[word];
      }
    }
    stage += sizes[range];
  }
}

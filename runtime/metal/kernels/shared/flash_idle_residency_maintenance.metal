#include <metal_stdlib>
using namespace metal;

struct ImmutableOwners1134 { array<device const uint *, 1134> p [[id(0)]]; };
struct ImmutableNonPLEOwners1141 { array<device const uint *, 1141> p [[id(0)]]; };

// One four-byte read per complete original/derived immutable native owner.
// Output belongs only to this diagnostic command. No model math/state changes.
kernel void flash_idle_immutable_touch_v1(
    constant ImmutableOwners1134 &owners [[buffer(0)]],
    device uint *output [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint checksums[256];
  uint checksum = 0;
  for (uint index = lane; index < 1134; index += 256) {
    const uint mixed = owners.p[index][0] ^ (0x9e3779b9u * (index + 1u));
    output[16 + index] = mixed;
    checksum ^= mixed;
  }
  checksums[lane] = checksum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride; stride /= 2) {
    if (lane < stride) checksums[lane] ^= checksums[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!lane) { output[1] = checksums[0]; output[2] = 1134; }
}

// Exact SSD-mode reflected pointer count. These are only original non-PLE
// windows and verified derived weights: no SSD table, row cache or staging.
kernel void flash_idle_immutable_touch_ple_ssd_v1(
    constant ImmutableNonPLEOwners1141 &owners [[buffer(0)]],
    device uint *output [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint checksums[256];
  uint checksum = 0;
  for (uint index = lane; index < 1141; index += 256) {
    const uint mixed = owners.p[index][0] ^ (0x9e3779b9u * (index + 1u));
    output[16 + index] = mixed;
    checksum ^= mixed;
  }
  checksums[lane] = checksum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride; stride /= 2) {
    if (lane < stride) checksums[lane] ^= checksums[lane + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!lane) { output[1] = checksums[0]; output[2] = 1141; }
}

#include "metal/abi/VerifyBodyPack.h"
#include <metal_stdlib>
using namespace metal;
// Caller binds this path only after clamping every selected prefix to <=4.
kernel void verify_body_pack_gather(device const ushort *input [[buffer(0)]],
                                    device const ushort *residual [[buffer(1)]],
                                    device const uint *rows [[buffer(2)]],
                                    device ushort *packed [[buffer(3)]],
                                    device ushort *packed_residual [[buffer(4)]],
                                    constant VerifyBodyPackParams &p [[buffer(5)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  if (gid.y >= 32)
    return;
  const uint lane = gid.y / 4, row = gid.y % 4;
  const bool active = lane < p.lanes && row < min(rows[lane], 4u);
  const ulong original = ulong(lane) * 8 + row;
  if (gid.x < p.input_size)
    packed[ulong(gid.y) * p.input_size + gid.x] = active ? input[original * p.input_size + gid.x] : ushort(0);
  if (p.residual && gid.x < p.columns)
    packed_residual[ulong(gid.y) * p.output_size + gid.x] =
        active ? residual[original * p.output_size + gid.x] : ushort(0);
}
kernel void verify_body_pack_scatter(device const ushort *packed [[buffer(0)]],
                                     device const uint *rows [[buffer(1)]],
                                     device ushort *output [[buffer(2)]],
                                     constant VerifyBodyPackParams &p [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.columns || gid.y >= 64)
    return;
  const uint lane = gid.y / 8, row = gid.y % 8;
  const bool active = lane < p.lanes && row < min(rows[lane], 4u);
  output[ulong(gid.y) * p.output_size + gid.x] =
      active ? packed[ulong(lane * 4 + row) * p.output_size + gid.x] : ushort(0);
}

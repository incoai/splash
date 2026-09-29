#include <metal_stdlib>

using namespace metal;

kernel void test_copy_u32(device const uint *source [[buffer(0)]],
                         device uint *destination [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        destination[gid] = source[gid];
    }
}

kernel void test_add_u32(device uint *values [[buffer(0)]],
                       constant uint &count [[buffer(1)]],
                       constant uint &increment [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        values[gid] += increment;
    }
}

kernel void sparse_fill_copy_u32(device uint *sparseValues [[buffer(0)]],
                                 device uint *readback [[buffer(1)]],
                                 constant uint &count [[buffer(2)]],
                                 constant uint &value [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        sparseValues[gid] = value + gid;
        readback[gid] = sparseValues[gid];
    }
}

// The source has no ordinary buffer binding: only its GPU address is passed.
kernel void test_indirect_copy_u32(constant ulong &address [[buffer(0)]],
                                   device uint *destination [[buffer(1)]]) {
    destination[0] = reinterpret_cast<device const uint *>(address)[0];
}

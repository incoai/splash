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

// Adds function constant 0 to the first word: a pipeline per value.
constant uint kTestIncrement [[function_constant(0)]];
kernel void test_add_constant_u32(device uint *values [[buffer(0)]]) {
    values[0] += kTestIncrement;
}

// Row r of the grid writes word x of the buffer at table[r], which no
// dispatch binds; addressed_check_u32 reads it back the same way.
kernel void addressed_write_u32(device const ulong *table [[buffer(0)]],
                                constant uint &words [[buffer(1)]],
                                constant uint &seed [[buffer(2)]],
                                uint2 id [[thread_position_in_grid]]) {
    if (id.x < words) {
        reinterpret_cast<device uint *>(table[id.y])[id.x] =
            seed ^ (id.y * 131071u + id.x);
    }
}

kernel void addressed_check_u32(device const ulong *table [[buffer(0)]],
                                constant uint &words [[buffer(1)]],
                                constant uint &seed [[buffer(2)]],
                                device atomic_uint *mismatches [[buffer(3)]],
                                uint2 id [[thread_position_in_grid]]) {
    if (id.x < words &&
        reinterpret_cast<device const uint *>(table[id.y])[id.x] !=
            (seed ^ (id.y * 131071u + id.x))) {
        atomic_fetch_add_explicit(mismatches, 1u, memory_order_relaxed);
    }
}

// addressed_write_u32 and addressed_check_u32 through the table of addresses
// the backend writes for a dispatch (metal::AddressTable), in the constant
// address space after the parameters: the words of a row and the seed.
kernel void address_table_write_u32(constant uint2 &p [[buffer(0)]],
                                    constant ulong *table [[buffer(1)]],
                                    uint2 id [[thread_position_in_grid]]) {
    if (id.x < p.x) {
        reinterpret_cast<device uint *>(table[id.y])[id.x] = p.y ^ (id.y * 131071u + id.x);
    }
}

kernel void address_table_check_u32(device atomic_uint *mismatches [[buffer(0)]],
                                    constant uint2 &p [[buffer(1)]],
                                    constant ulong *table [[buffer(2)]],
                                    uint2 id [[thread_position_in_grid]]) {
    if (id.x < p.x &&
        reinterpret_cast<device const uint *>(table[id.y])[id.x] != (p.y ^ (id.y * 131071u + id.x))) {
        atomic_fetch_add_explicit(mismatches, 1u, memory_order_relaxed);
    }
}

// A host_step_wait that ends at once, without seeing its release: a test
// of the backend's spin check names this library host-step.metallib.
kernel void host_step_wait() {}

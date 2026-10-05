#include <metal_stdlib>
#include "metal/abi/ExecutionGeometry.h"
using namespace metal;

// Native Metal oracle for the persistent Q8 page format. Production writes
// pages directly from its prefill and verify kernels and does not link these
// conversion or gather symbols.
constant uint SplashQ8PageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
constant uint SplashQ8KVHeads = 4;
constant uint SplashQ8HeadDimension = 256;
constant uint SplashQ8ElementsPerLayerPage = 32768;
constant uint SplashQ8ScalesPerLayerPage = 128;

struct SplashQ8KVPageParams {
    uint source_page;
    uint destination_page;
    uint valid_tokens;
    uint reserved;
};

inline ulong splash_q8_key_data_index(
    uint page, uint head, uint token, uint dimension)
{
    return ulong(page) * SplashQ8ElementsPerLayerPage +
        (ulong(head) * SplashQ8PageTokens + token) *
            SplashQ8HeadDimension + dimension;
}

inline ulong splash_q8_key_scale_index(
    uint page, uint head, uint token)
{
    return ulong(page) * SplashQ8ScalesPerLayerPage +
        ulong(head) * SplashQ8PageTokens + token;
}

inline ulong splash_q8_value_data_index(
    uint page, uint head, uint token, uint dimension)
{
    return ulong(page) * SplashQ8ElementsPerLayerPage +
        (ulong(head) * SplashQ8HeadDimension + dimension) *
            SplashQ8PageTokens + token;
}

inline ulong splash_q8_value_scale_index(
    uint page, uint head, uint token)
{
    return ulong(page) * SplashQ8ScalesPerLayerPage +
        ulong(head) * SplashQ8PageTokens + token;
}

inline float splash_q8_load_key(
    device const char *keys,
    device const float *scales,
    uint page, uint head, uint token, uint dimension)
{
    float scale = scales[splash_q8_key_scale_index(page, head, token)];
    return float(keys[splash_q8_key_data_index(
        page, head, token, dimension)]) * scale;
}

inline float splash_q8_load_value(
    device const char *values,
    device const float *scales,
    uint page, uint head, uint token, uint dimension)
{
    float scale = scales[splash_q8_value_scale_index(page, head, token)];
    return float(values[splash_q8_value_data_index(
        page, head, token, dimension)]) * scale;
}

// Converts one BF16 cache page for one attention layer into per-token/head
// symmetric Q8. Dispatch 256 threadgroups of 256 threads: 128 K rows followed
// by 128 V rows. Each row has exactly one FP32 scale.
kernel void splash_q8_quantize_kv_page(
    device const bfloat *source_keys [[buffer(0)]],
    device const bfloat *source_values [[buffer(1)]],
    device char *destination_keys [[buffer(2)]],
    device float *destination_key_scales [[buffer(3)]],
    device char *destination_values [[buffer(4)]],
    device float *destination_value_scales [[buffer(5)]],
    constant SplashQ8KVPageParams &params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    if (group >= 2 * SplashQ8ScalesPerLayerPage ||
        thread_index >= SplashQ8HeadDimension) {
        return;
    }
    bool value_group = group >= SplashQ8ScalesPerLayerPage;
    uint local_group = value_group
        ? group - SplashQ8ScalesPerLayerPage : group;
    uint token = local_group % SplashQ8PageTokens;
    uint head = local_group / SplashQ8PageTokens;
    uint dimension = thread_index;
    bool valid = token < min(params.valid_tokens, SplashQ8PageTokens);
    ulong source_index = value_group
        ? splash_q8_value_data_index(
              params.source_page, head, token, dimension)
        : splash_q8_key_data_index(
              params.source_page, head, token, dimension);
    ulong destination_index = value_group
        ? splash_q8_value_data_index(
              params.destination_page, head, token, dimension)
        : splash_q8_key_data_index(
              params.destination_page, head, token, dimension);
    ulong scale_index = value_group
        ? splash_q8_value_scale_index(params.destination_page, head, token)
        : splash_q8_key_scale_index(params.destination_page, head, token);

    float value = valid ? float(
        value_group ? source_values[source_index] : source_keys[source_index])
        : 0.0f;
    float local_maximum = simd_max(abs(value));
    threadgroup float maxima[8];
    if (simd_lane == 0) maxima[simd_group] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        for (uint offset = 4; offset != 0; offset >>= 1) {
            if (simd_lane < offset) {
                maxima[simd_lane] = max(
                    maxima[simd_lane], maxima[simd_lane + offset]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = maxima[0] == 0.0f ? 0.0f : maxima[0] / 127.0f;
    int quantized = valid && maxima[0] != 0.0f
        ? int(rint(value * 127.0f / maxima[0])) : 0;
    quantized = clamp(quantized, -127, 127);
    if (value_group) {
        destination_values[destination_index] = char(quantized);
        if (thread_index == 0) destination_value_scales[scale_index] = scale;
    } else {
        destination_keys[destination_index] = char(quantized);
        if (thread_index == 0) destination_key_scales[scale_index] = scale;
    }
}

// Native-oracle conversion back into BF16 physical layouts. This symbol is
// absent from the production metallib.
kernel void splash_q8_dequantize_kv_page(
    device const char *source_keys [[buffer(0)]],
    device const float *source_key_scales [[buffer(1)]],
    device const char *source_values [[buffer(2)]],
    device const float *source_value_scales [[buffer(3)]],
    device bfloat *destination_keys [[buffer(4)]],
    device bfloat *destination_values [[buffer(5)]],
    constant SplashQ8KVPageParams &params [[buffer(6)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= 2 * SplashQ8ElementsPerLayerPage) return;
    bool value = index >= SplashQ8ElementsPerLayerPage;
    uint local = value ? index - SplashQ8ElementsPerLayerPage : index;
    uint head;
    uint token;
    uint dimension;
    if (!value) {
        dimension = local % SplashQ8HeadDimension;
        uint row = local / SplashQ8HeadDimension;
        token = row % SplashQ8PageTokens;
        head = row / SplashQ8PageTokens;
        ulong destination = splash_q8_key_data_index(
            params.destination_page, head, token, dimension);
        destination_keys[destination] = token < params.valid_tokens
            ? bfloat(splash_q8_load_key(
                  source_keys, source_key_scales, params.source_page,
                  head, token, dimension)) : bfloat(0.0f);
    } else {
        token = local % SplashQ8PageTokens;
        uint column = local / SplashQ8PageTokens;
        dimension = column % SplashQ8HeadDimension;
        head = column / SplashQ8HeadDimension;
        ulong destination = splash_q8_value_data_index(
            params.destination_page, head, token, dimension);
        destination_values[destination] = token < params.valid_tokens
            ? bfloat(splash_q8_load_value(
                  source_values, source_value_scales, params.source_page,
                  head, token, dimension)) : bfloat(0.0f);
    }
}

// Native-oracle logical gather. This symbol is absent from production.
kernel void splash_q8_gather_logical_kv_page(
    device const char *source_keys [[buffer(0)]],
    device const float *source_key_scales [[buffer(1)]],
    device const char *source_values [[buffer(2)]],
    device const float *source_value_scales [[buffer(3)]],
    device bfloat *logical_keys [[buffer(4)]],
    device bfloat *logical_values [[buffer(5)]],
    constant SplashQ8KVPageParams &params [[buffer(6)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= SplashQ8ElementsPerLayerPage) return;
    uint dimension = index % SplashQ8HeadDimension;
    uint row = index / SplashQ8HeadDimension;
    uint head = row % SplashQ8KVHeads;
    uint token = row / SplashQ8KVHeads;
    bool valid = token < min(params.valid_tokens, SplashQ8PageTokens);
    logical_keys[index] = valid ? bfloat(splash_q8_load_key(
        source_keys, source_key_scales, params.source_page,
        head, token, dimension)) : bfloat(0.0f);
    logical_values[index] = valid ? bfloat(splash_q8_load_value(
        source_values, source_value_scales, params.source_page,
        head, token, dimension)) : bfloat(0.0f);
}

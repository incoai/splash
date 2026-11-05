// Prism ML's input rotation (metal/abi/Gguf.h): the normalized Walsh-Hadamard transform of every block of
// GGUF_ROTATION_BLOCK values, in fp32 and rounded once to bf16. Thread tid holds values tid + 256 i of a block;
// butterfly stages of strides below 32 pair values of one simdgroup, the others pass through threadgroup memory.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_embedding_formats.h"
#include <metal_stdlib>

using namespace metal;

static_assert(GGUF_ROTATION_BLOCK == 4 * GGUF_ROTATION_THREADS, "a rotation thread holds four values of its block");
// 1 / sqrt(GGUF_ROTATION_BLOCK), which normalizes the transform.
constant constexpr float kRotationScale = 1.0f / 32.0f;
static_assert(GGUF_ROTATION_BLOCK == 1024, "kRotationScale is 1 / sqrt(GGUF_ROTATION_BLOCK)");

// The unnormalized transform of a block, v[i] being value tid + 256 i: each stage maps the pair (x, y) of values
// `stride` apart to (x + y, x - y). The transformed block is left in `values`.
inline void rotation_butterflies(float4 v, threadgroup float *values, uint tid) {
  for (ushort stride = 1; stride < 32; stride *= 2) {
    const float4 peer = simd_shuffle_xor(v, stride);
    v = (tid & stride) ? peer - v : v + peer;
  }
  for (uint i = 0; i < 4; ++i) values[tid + i * GGUF_ROTATION_THREADS] = v[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 32; stride < GGUF_ROTATION_BLOCK; stride *= 2) {
    for (uint pair = tid; pair < GGUF_ROTATION_BLOCK / 2; pair += GGUF_ROTATION_THREADS) {
      const uint a = (pair / stride) * (2 * stride) + pair % stride;
      const float x = values[a], y = values[a + stride];
      values[a] = x + y;
      values[a + stride] = x - y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The rotated input of a projection whose weights were stored for it: output = H (D input) per block of each row.
// Grid (width / GGUF_ROTATION_BLOCK, rows).
kernel void gguf_rotate(device const bfloat *input [[buffer(0)]], device const char *signs [[buffer(1)]],
                        device bfloat *output [[buffer(2)]], constant GgufRotationParams &p [[buffer(3)]],
                        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float values[GGUF_ROTATION_BLOCK];
  const uint column = group.x * GGUF_ROTATION_BLOCK;
  const ulong row = ulong(group.y) * p.width + column;
  float4 v;
  for (uint i = 0; i < 4; ++i) {
    const uint j = tid + i * GGUF_ROTATION_THREADS;
    v[i] = float(input[row + j]) * float(signs[column + j]);
  }
  rotation_butterflies(v, values, tid);
  for (uint i = 0; i < 4; ++i) {
    const uint j = tid + i * GGUF_ROTATION_THREADS;
    output[row + j] = bfloat(values[j] * kRotationScale);
  }
}

// The token rows of a PQ2_0 table stored rotated (GgufEmbedPQ20), as D (H r): the gathered values enter the
// transform in fp32, so each output rounds once. Grid (hidden / GGUF_ROTATION_BLOCK, rows).
kernel void gguf_embed_rotated_pq20(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]],
                                    device const char *signs [[buffer(2)]], device bfloat *output [[buffer(3)]],
                                    constant GgufEmbedParams &p [[buffer(4)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  using F = GgufEmbedPQ20;
  threadgroup float values[GGUF_ROTATION_BLOCK];
  // The runtime validates every token; the guard keeps a direct call inside the table.
  const uint token = tokens[group.y] < p.vocabulary ? tokens[group.y] : 0;
  const uint column = group.x * GGUF_ROTATION_BLOCK;
  device const uchar *row = table + ulong(token) * (p.hidden / F::Weights) * F::Bytes;
  float4 v;
  for (uint i = 0; i < 4; ++i) {
    const uint dim = column + tid + i * GGUF_ROTATION_THREADS;
    v[i] = F::decode(row + (dim / F::Weights) * F::Bytes, dim);
  }
  rotation_butterflies(v, values, tid);
  for (uint i = 0; i < 4; ++i) {
    const uint j = tid + i * GGUF_ROTATION_THREADS;
    output[ulong(group.y) * p.hidden + column + j] = bfloat(values[j] * kRotationScale * float(signs[column + j]));
  }
}

// The int8 rows of the int8-activation PQ2_0 decode tile (metal/abi/Gguf.h GgufA8QuantParams; splash serve
// --decode-int8): gguf_rotate's bf16 values H (D x), quantized per rotation block of a row with s = max |x| / 127
// (fp32) and q = clamp(rint(x / s), -127, 127), written as int8 in the tile's stage order (gguf_pq2w8_position),
// the scales after the p.rows int8 rows. No bf16 output. Grid (width / GGUF_ROTATION_BLOCK, rows).
kernel void gguf_rotate_a8(device const bfloat *input [[buffer(0)]], device const char *signs [[buffer(1)]],
                           device uchar *a8 [[buffer(2)]], constant GgufA8QuantParams &p [[buffer(3)]],
                           uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float values[GGUF_ROTATION_BLOCK];
  threadgroup float part[8];
  const uint K = p.input_size, column = group.x * GGUF_ROTATION_BLOCK;
  const ulong row = ulong(group.y) * K + column;
  float4 v;
  for (uint i = 0; i < 4; ++i) {
    const uint j = tid + i * GGUF_ROTATION_THREADS;
    v[i] = float(input[row + j]) * float(signs[column + j]);
  }
  rotation_butterflies(v, values, tid);
  float4 x;
  for (uint i = 0; i < 4; ++i) x[i] = float(bfloat(values[4 * tid + i] * kRotationScale));
  float m = max(max(fabs(x.x), fabs(x.y)), max(fabs(x.z), fabs(x.w)));
  m = simd_max(m);
  if (lane == 0) part[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = part[0];
  for (uint i = 1; i < 8; ++i) m = max(m, part[i]);
  const float s = m / 127.0f;
  device uchar *x8 = a8 + p.a8_offset;
  for (uint i = 0; i < 4; ++i) {
    const uint k = column + 4 * tid + i;
    const float q = s > 0.0f ? clamp(rint(x[i] / s), -127.0f, 127.0f) : 0.0f;
    x8[ulong(group.y) * K + (k & ~31u) + gguf_pq2w8_position(k & 31u)] = uchar(char(int(q)));
  }
  if (tid == 0) ((device float *)(x8 + ulong(p.rows) * K))[group.y * (K / GGUF_ROTATION_BLOCK) + group.x] = s;
}

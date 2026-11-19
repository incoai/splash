// The speech tokenizer decoder (ops::omni::CodecDecoder): fp32 activations
// in [time][channel] rows. The convolutions, transposed convolutions and
// linears of the decoder run an implicit-GEMM tile kernel whose A operand is
// gathered from the input rows each tap reaches (causal left padding reads
// zero) and whose epilogue carries the bias, the activation, a per-channel
// scale and the residual add: omni_codec_conv_half over half operands on the
// MPP matmul for every convolution of whole groups of 8 output channels (all
// of the model's), and omni_codec_conv over fp32 weights on simdgroup
// matrices, with the next step's operands fetched into registers while the
// current step multiplies, for the others (and as the tests' exact path).
// The small pieces (RVQ lookup, the depthwise ConvNeXt convolution,
// SnakeBeta, the SiLU gate, RoPE, the windowed causal attention and the
// output convolution) have their own kernels.

#include "metal/abi/OmniCodec.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

inline float omni_codec_snake(float x, float alpha, float inverse_beta) {
  const float s = precise::sin(x * alpha);
  return x + inverse_beta * s * s;
}

// erf to 1.5e-7 (Abramowitz and Stegun 7.1.26), the fp32 precision of
// the exact GELU.
inline float omni_codec_erf(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  x = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * x);
  const float polynomial =
      t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - polynomial * precise::exp(-x * x));
}

inline float omni_codec_gelu(float x) { return 0.5f * x * (1.0f + omni_codec_erf(x * 0.7071067811865476f)); }

// --------------------------------------------------------- implicit GEMM

constant constexpr uint kConvRows = OMNI_CODEC_CONV_TILE_ROWS;
constant constexpr uint kConvColumns = OMNI_CODEC_CONV_TILE_COLUMNS;
constant constexpr uint kConvDepth = OMNI_CODEC_CONV_TILE_DEPTH;
constant constexpr uint kConvThreads = OMNI_CODEC_CONV_THREADS;
constant constexpr uint kAStride = kConvDepth + 4;   // As [64][36]
constant constexpr uint kBStride = kConvColumns + 4; // Bs [32][68]
constant constexpr uint kCStride = kConvColumns + 4; // Cs [64][68], over As and Bs
constant constexpr uint kSharedFloats = kConvRows * kAStride + kConvDepth * kBStride;
static_assert(kConvRows * kCStride <= kSharedFloats, "the output tile reuses the operand tiles' memory");
// Per thread and step: 4 float4 of the A tile (64 x 32) and 4 of the B tile
// (32 x 64).
constant constexpr uint kLoadsPerThread = kConvRows * kConvDepth / 4 / kConvThreads;
static_assert(kLoadsPerThread == kConvDepth * kConvColumns / 4 / kConvThreads, "A and B tiles load alike");

// The operand loads of one step, kept in registers while the previous
// step's multiplies run, so their latency overlaps the arithmetic.
struct OmniCodecConvStage {
  float4 a[kLoadsPerThread];
  float4 b[kLoadsPerThread];
};

inline OmniCodecConvStage omni_codec_conv_fetch(device const float *input, device const float *history,
                                                device const float *weight, constant OmniCodecConvParams &p,
                                                uint step, uint steps, uint row0, uint column0, uint tid) {
  const uint tap = step / steps;
  const uint k0 = (step % steps) * kConvDepth;
  const int base = int(tap * p.dilation) - p.pad;
  device const float *w = weight + ulong(tap) * p.in_channels * p.out_channels;
  OmniCodecConvStage stage;
#pragma clang loop unroll(full)
  for (uint l = 0; l < kLoadsPerThread; ++l) {
    const uint i = tid + l * kConvThreads;
    const uint m = i / (kConvDepth / 4);
    const uint quad = (i % (kConvDepth / 4)) * 4;
    const int r = int(row0 + m) + base;
    stage.a[l] = float4(0.0f);
    if (row0 + m < p.rows) {
      if (r >= 0 && r < int(p.input_rows))
        stage.a[l] = *reinterpret_cast<device const float4 *>(input + ulong(r) * p.input_stride + k0 + quad);
      else if (r < 0 && r + int(p.history_rows) >= 0)
        stage.a[l] = *reinterpret_cast<device const float4 *>(history + ulong(r + int(p.history_rows)) * p.in_channels +
                                                              k0 + quad);
    }
    const uint k = i / (kConvColumns / 4);
    const uint n = column0 + (i % (kConvColumns / 4)) * 4;
    stage.b[l] = float4(0.0f);
    if (n < p.out_channels) stage.b[l] = *reinterpret_cast<device const float4 *>(w + ulong(k0 + k) * p.out_channels + n);
  }
  return stage;
}

inline void omni_codec_conv_stage(const thread OmniCodecConvStage &stage, threadgroup float4 *As,
                                  threadgroup float4 *Bs, uint tid) {
#pragma clang loop unroll(full)
  for (uint l = 0; l < kLoadsPerThread; ++l) {
    const uint i = tid + l * kConvThreads;
    const uint m = i / (kConvDepth / 4);
    const uint quad = i % (kConvDepth / 4);
    As[m * (kAStride / 4) + quad] = stage.a[l];
    const uint k = i / (kConvColumns / 4);
    const uint nq = i % (kConvColumns / 4);
    Bs[k * (kBStride / 4) + nq] = stage.b[l];
  }
}

kernel void omni_codec_conv(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
                            device const float *bias [[buffer(2)]], device const float *residual [[buffer(3)]],
                            device const float *scale [[buffer(4)]], device float *output [[buffer(5)]],
                            device const float *history [[buffer(6)]], constant OmniCodecConvParams &p [[buffer(7)]],
                            uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float4 shared[kSharedFloats / 4];
  threadgroup float4 *As4 = shared;
  threadgroup float4 *Bs4 = shared + kConvRows * kAStride / 4;
  threadgroup float *As = reinterpret_cast<threadgroup float *>(As4);
  threadgroup float *Bs = reinterpret_cast<threadgroup float *>(Bs4);
  threadgroup float *Cs = reinterpret_cast<threadgroup float *>(shared);
  const uint row0 = group.x * kConvRows;
  const uint column0 = group.y * kConvColumns;
  const uint sgRow = (sg / 2) * 32;
  const uint sgColumn = (sg % 2) * 32;

  // The fragment loops must unroll fully, or the accumulators leave the
  // registers and the kernel runs ten times slower.
  simdgroup_float8x8 acc[4][4];
#pragma clang loop unroll(full)
  for (uint i = 0; i < 4; ++i)
#pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);

  const uint steps = p.in_channels / kConvDepth;
  const uint total = p.taps * steps;
  OmniCodecConvStage stage = omni_codec_conv_fetch(input, history, weight, p, 0, steps, row0, column0, tid);
  for (uint step = 0; step < total; ++step) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    omni_codec_conv_stage(stage, As4, Bs4, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (step + 1 < total)
      stage = omni_codec_conv_fetch(input, history, weight, p, step + 1, steps, row0, column0, tid);
#pragma clang loop unroll(full)
    for (uint kk = 0; kk < kConvDepth; kk += 8) {
      simdgroup_float8x8 a[4], b[4];
#pragma clang loop unroll(full)
      for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sgRow + i * 8) * kAStride + kk, kAStride);
#pragma clang loop unroll(full)
      for (uint j = 0; j < 4; ++j) simdgroup_load(b[j], Bs + kk * kBStride + sgColumn + j * 8, kBStride);
#pragma clang loop unroll(full)
      for (uint i = 0; i < 4; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
  for (uint i = 0; i < 4; ++i)
#pragma clang loop unroll(full)
    for (uint j = 0; j < 4; ++j)
      simdgroup_store(acc[i][j], Cs + (sgRow + i * 8) * kCStride + sgColumn + j * 8, kCStride);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < kConvRows * kConvColumns / 4; i += kConvThreads) {
    const uint m = i / (kConvColumns / 4);
    const uint quad = (i % (kConvColumns / 4)) * 4;
    const uint t = row0 + m;
    const uint n = column0 + quad;
    if (t >= p.rows || n >= p.out_channels) continue;
    float4 v = *reinterpret_cast<threadgroup const float4 *>(Cs + m * kCStride + quad);
    if (p.flags & OMNI_CODEC_CONV_FLAG_BIAS) v += *reinterpret_cast<device const float4 *>(bias + n);
    if (p.flags & OMNI_CODEC_CONV_FLAG_GELU) {
      v.x = omni_codec_gelu(v.x);
      v.y = omni_codec_gelu(v.y);
      v.z = omni_codec_gelu(v.z);
      v.w = omni_codec_gelu(v.w);
    }
    if (p.flags & OMNI_CODEC_CONV_FLAG_SCALE) v *= *reinterpret_cast<device const float4 *>(scale + n);
    const ulong at = ulong(t) * p.output_stride + n;
    if (p.flags & OMNI_CODEC_CONV_FLAG_RESIDUAL) v += *reinterpret_cast<device const float4 *>(residual + at);
    *reinterpret_cast<device float4 *>(output + at) = v;
  }
}

// The same convolution over half-precision operands with fp32 accumulation
// on the MPP matmul (omni_codec_conv_half): the A tile is gathered from
// the fp32 rows as the fp32 kernel gathers it and rounded to half into
// threadgroup memory, the B tile read from the fp16 repack of the weights
// (CodecConv::weightHalf); the epilogue is the fp32 kernel's. Output
// channels are a multiple of 8. The product's error is that of rounding
// each operand to half (2^-11 relative), far below the codec's audible
// noise. The tile is Rows x Columns x Depth over Simdgroups; wider tiles
// (up to 128 x 128, depth 64) were within noise or slower on every decoder
// stage (docs/omni-performance.md, round 3), so one instance runs them all.
// Per thread and step: float4 loads of the A tile and uint4 (8 halves) of B.
template <uint Rows, uint Depth, uint Threads> constant constexpr uint kHalfALoads = Rows * Depth / 4 / Threads;
template <uint Columns, uint Depth, uint Threads> constant constexpr uint kHalfBLoads = Depth * Columns / 8 / Threads;
template <uint Rows, uint Columns, uint Depth, uint Threads>
struct OmniCodecHalfTile {
  static_assert(Rows * Depth % (4 * Threads) == 0 && Depth * Columns % (8 * Threads) == 0, "tile loads split evenly");
  float4 a[Rows * Depth / 4 / Threads];
  uint4 b[Depth * Columns / 8 / Threads];
};

template <uint Rows, uint Columns, uint Depth, uint Threads>
inline OmniCodecHalfTile<Rows, Columns, Depth, Threads>
omni_codec_half_fetch(device const float *input, device const float *history, device const half *weight,
                      constant OmniCodecConvParams &p, uint step, uint steps, uint row0, uint column0, uint tid) {
  using Tile = OmniCodecHalfTile<Rows, Columns, Depth, Threads>;
  const uint tap = step / steps;
  const uint k0 = (step % steps) * Depth;
  const int base = int(tap * p.dilation) - p.pad;
  device const half *w = weight + ulong(tap) * p.in_channels * p.out_channels;
  Tile stage;
#pragma clang loop unroll(full)
  for (uint l = 0; l < kHalfALoads<Rows, Depth, Threads>; ++l) {
    const uint i = tid + l * Threads;
    const uint m = i / (Depth / 4);
    const uint quad = (i % (Depth / 4)) * 4;
    const int r = int(row0 + m) + base;
    stage.a[l] = float4(0.0f);
    if (row0 + m < p.rows) {
      if (r >= 0 && r < int(p.input_rows))
        stage.a[l] = *reinterpret_cast<device const float4 *>(input + ulong(r) * p.input_stride + k0 + quad);
      else if (r < 0 && r + int(p.history_rows) >= 0)
        stage.a[l] = *reinterpret_cast<device const float4 *>(history + ulong(r + int(p.history_rows)) * p.in_channels +
                                                              k0 + quad);
    }
  }
#pragma clang loop unroll(full)
  for (uint l = 0; l < kHalfBLoads<Columns, Depth, Threads>; ++l) {
    const uint i = tid + l * Threads;
    const uint k = i / (Columns / 8);
    const uint n = column0 + (i % (Columns / 8)) * 8;
    stage.b[l] = uint4(0u);
    if (n < p.out_channels) stage.b[l] = *reinterpret_cast<device const uint4 *>(w + ulong(k0 + k) * p.out_channels + n);
  }
  return stage;
}

template <uint Rows, uint Columns, uint Depth, uint Simdgroups>
inline void omni_codec_conv_half_tile(device const float *input, device const half *weight, device const float *bias,
                                      device const float *residual, device const float *scale, device float *output,
                                      device const float *history, constant OmniCodecConvParams &p,
                                      threadgroup half *As, threadgroup half *Bs, uint2 group, uint tid) {
  constexpr uint Threads = Simdgroups * 32;
  using Tile = OmniCodecHalfTile<Rows, Columns, Depth, Threads>;
  const uint row0 = group.x * Rows;
  const uint column0 = group.y * Columns;
  auto a = tensor(As, dextents<int, 2>{int(Depth), int(Rows)}, array<int, 2>{1, int(Depth)});
  auto b = tensor(Bs, dextents<int, 2>{int(Columns), int(Depth)}, array<int, 2>{1, int(Columns)});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, Columns, Depth, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto a_slice = a.template slice<Depth, Rows>(0, 0);
  auto b_slice = b.template slice<Columns, Depth>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a_slice), decltype(b_slice), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;

  const uint steps = p.in_channels / Depth;
  const uint total = p.taps * steps;
  Tile stage = omni_codec_half_fetch<Rows, Columns, Depth, Threads>(input, history, weight, p, 0, steps, row0, column0, tid);
  for (uint step = 0; step < total; ++step) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint l = 0; l < kHalfALoads<Rows, Depth, Threads>; ++l) {
      const uint i = tid + l * Threads;
      const uint m = i / (Depth / 4);
      const uint quad = (i % (Depth / 4)) * 4;
      *reinterpret_cast<threadgroup half4 *>(As + m * Depth + quad) = half4(stage.a[l]);
    }
#pragma clang loop unroll(full)
    for (uint l = 0; l < kHalfBLoads<Columns, Depth, Threads>; ++l) {
      const uint i = tid + l * Threads;
      const uint k = i / (Columns / 8);
      const uint n = (i % (Columns / 8)) * 8;
      *reinterpret_cast<threadgroup uint4 *>(Bs + k * Columns + n) = stage.b[l];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (step + 1 < total)
      stage = omni_codec_half_fetch<Rows, Columns, Depth, Threads>(input, history, weight, p, step + 1, steps, row0,
                                                                   column0, tid);
    operation.run(a_slice, b_slice, accumulated);
  }
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    if (!accumulated.is_valid_element(i)) continue;
    auto index = accumulated.get_multidimensional_index(i);
    const uint n = column0 + index[0];
    const uint t = row0 + index[1];
    if (t >= p.rows || n >= p.out_channels) continue;
    float v = accumulated[i];
    if (p.flags & OMNI_CODEC_CONV_FLAG_BIAS) v += bias[n];
    if (p.flags & OMNI_CODEC_CONV_FLAG_GELU) v = omni_codec_gelu(v);
    if (p.flags & OMNI_CODEC_CONV_FLAG_SCALE) v *= scale[n];
    const ulong at = ulong(t) * p.output_stride + n;
    if (p.flags & OMNI_CODEC_CONV_FLAG_RESIDUAL) v += residual[at];
    output[at] = v;
  }
}

#define OMNI_CODEC_HALF_KERNEL(Name, Rows, Columns, Depth, Simdgroups)                                             \
  kernel void Name(device const float *input [[buffer(0)]], device const half *weight [[buffer(1)]],              \
                   device const float *bias [[buffer(2)]], device const float *residual [[buffer(3)]],            \
                   device const float *scale [[buffer(4)]], device float *output [[buffer(5)]],                   \
                   device const float *history [[buffer(6)]], constant OmniCodecConvParams &p [[buffer(7)]],      \
                   uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {       \
    threadgroup half As[Rows * Depth];                                                                            \
    threadgroup half Bs[Depth * Columns];                                                                         \
    omni_codec_conv_half_tile<Rows, Columns, Depth, Simdgroups>(input, weight, bias, residual, scale, output,     \
                                                                history, p, As, Bs, group, tid);                  \
  }
// The 64 x 64 x 32 tile of four simdgroups (OMNI_CODEC_CONV_TILE_*).
OMNI_CODEC_HALF_KERNEL(omni_codec_conv_half, 64, 64, 32, 4)
#undef OMNI_CODEC_HALF_KERNEL

// ---------------------------------------------------------- small kernels

kernel void omni_codec_depthwise(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
                                 device const float *bias [[buffer(2)]], device float *output [[buffer(3)]],
                                 device const float *history [[buffer(4)]],
                                 constant OmniCodecDepthwiseParams &p [[buffer(5)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= p.rows * p.channels) return;
  const uint t = index / p.channels;
  const uint c = index % p.channels;
  float sum = bias[c];
  for (uint k = 0; k < p.taps; ++k) {
    const int r = int(t) - int(p.taps - 1) + int(k);
    if (r >= 0) sum += weight[k * p.channels + c] * input[ulong(r) * p.channels + c];
    else if (r + int(p.history_rows) >= 0)
      sum += weight[k * p.channels + c] * history[ulong(r + int(p.history_rows)) * p.channels + c];
  }
  output[index] = sum;
}

kernel void omni_codec_history(device const float *input [[buffer(0)]], device const float *old [[buffer(1)]],
                               device float *updated [[buffer(2)]], constant OmniCodecHistoryParams &p [[buffer(3)]],
                               uint index [[thread_position_in_grid]]) {
  if (index >= p.history_rows * p.width) return;
  const uint l = index / p.width;
  const uint c = index % p.width;
  const uint source = p.input_rows + l; // in [old history; input]
  updated[index] = source < p.history_rows ? old[ulong(source) * p.width + c]
                                           : input[ulong(source - p.history_rows) * p.width + c];
}

kernel void omni_codec_elementwise(device const float *input [[buffer(0)]], device const float *alpha [[buffer(1)]],
                                   device const float *beta [[buffer(2)]], device float *output [[buffer(3)]],
                                   constant OmniCodecElementwiseParams &p [[buffer(4)]],
                                   uint index [[thread_position_in_grid]]) {
  if (index >= p.rows * p.channels) return;
  const uint c = index % p.channels;
  if (p.kind == 0u) {
    output[index] = omni_codec_snake(input[index], alpha[c], beta[c]);
  } else {
    const uint t = index / p.channels;
    const float gate = input[ulong(t) * 2 * p.channels + c];
    const float up = input[ulong(t) * 2 * p.channels + p.channels + c];
    output[index] = gate / (1.0f + precise::exp(-gate)) * up;
  }
}

kernel void omni_codec_lookup(device const uint *codes [[buffer(0)]], device const float *codebooks [[buffer(1)]],
                              device float *output [[buffer(2)]], constant OmniCodecLookupParams &p [[buffer(3)]],
                              uint index [[thread_position_in_grid]]) {
  const uint width = 2 * p.half_width;
  if (index >= p.frames * width) return;
  const uint f = index / width;
  const uint d = index % width;
  device const uint *frame = codes + ulong(f) * p.quantizers;
  float value = 0.0f;
  if (d < p.half_width) {
    const uint code = min(frame[0], p.codebook_size - 1);
    value = codebooks[ulong(code) * p.half_width + d];
  } else {
    for (uint q = 1; q < p.quantizers; ++q) {
      const uint code = min(frame[q], p.codebook_size - 1);
      value += codebooks[(ulong(q) * p.codebook_size + code) * p.half_width + d - p.half_width];
    }
  }
  output[index] = value;
}

kernel void omni_codec_rope(device float *qkv [[buffer(0)]], constant OmniCodecRopeParams &p [[buffer(1)]],
                            uint index [[thread_position_in_grid]]) {
  constexpr uint pairs = OMNI_CODEC_HEAD_DIM / 2;
  const uint perRow = 2 * p.heads * pairs; // queries and keys, 32 pairs per head
  if (index >= p.rows * perRow) return;
  const uint row = index / perRow;
  const uint rest = index % perRow;
  const uint which = rest / (p.heads * pairs); // 0 query, 1 key
  const uint head = (rest % (p.heads * pairs)) / pairs;
  const uint d = rest % pairs;
  const float inverse = 1.0f / precise::pow(p.theta, float(2 * d) / float(OMNI_CODEC_HEAD_DIM));
  const float angle = float(p.position + row) * inverse;
  const float c = precise::cos(angle);
  const float s = precise::sin(angle);
  device float *v = qkv + ulong(row) * 3 * p.heads * OMNI_CODEC_HEAD_DIM + (which * p.heads + head) * OMNI_CODEC_HEAD_DIM;
  const float x0 = v[d];
  const float x1 = v[d + pairs];
  v[d] = x0 * c - x1 * s;
  v[d + pairs] = x1 * c + x0 * s;
}

kernel void omni_codec_kv_append(device const float *qkv [[buffer(0)]], device float *keys [[buffer(1)]],
                                 device float *values [[buffer(2)]], constant OmniCodecAppendParams &p [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
  const uint width = p.heads * OMNI_CODEC_HEAD_DIM;
  if (index >= p.rows * width) return;
  const uint row = index / width;
  const uint c = index % width;
  device const float *source = qkv + ulong(row) * 3 * width;
  const ulong at = ulong((p.position + row) % p.capacity) * width + c;
  keys[at] = source[width + c];
  values[at] = source[2 * width + c];
}

kernel void omni_codec_attention(device const float *qkv [[buffer(0)]], device const float *keys [[buffer(1)]],
                                 device const float *values [[buffer(2)]], device float *output [[buffer(3)]],
                                 constant OmniCodecAttentionParams &p [[buffer(4)]],
                                 uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scores[OMNI_CODEC_ATTENTION_GROUPS][OMNI_CODEC_MAXIMUM_WINDOW];
  const uint item = group * OMNI_CODEC_ATTENTION_GROUPS + sg;
  if (item >= p.rows * p.heads) return;
  const uint row = item / p.heads;
  const uint head = item % p.heads;
  const uint width = p.heads * OMNI_CODEC_HEAD_DIM;
  device const float *q = qkv + ulong(row) * 3 * width + head * OMNI_CODEC_HEAD_DIM;
  const float q0 = q[lane];
  const float q1 = q[lane + 32];
  const uint position = p.position + row;
  const uint begin = position + 1 > p.window ? position + 1 - p.window : 0;
  float maximum = -INFINITY;
  for (uint j = begin; j <= position; ++j) {
    device const float *k = keys + ulong(j % p.capacity) * width + head * OMNI_CODEC_HEAD_DIM;
    const float score = simd_sum(q0 * k[lane] + q1 * k[lane + 32]) * p.scale;
    maximum = max(maximum, score);
    if (lane == 0) scores[sg][j - begin] = score;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0.0f;
  float o0 = 0.0f, o1 = 0.0f;
  for (uint j = begin; j <= position; ++j) {
    const float weight = precise::exp(scores[sg][j - begin] - maximum);
    device const float *v = values + ulong(j % p.capacity) * width + head * OMNI_CODEC_HEAD_DIM;
    sum += weight;
    o0 += weight * v[lane];
    o1 += weight * v[lane + 32];
  }
  device float *out = output + ulong(row) * p.heads * OMNI_CODEC_HEAD_DIM + head * OMNI_CODEC_HEAD_DIM;
  out[lane] = o0 / sum;
  out[lane + 32] = o1 / sum;
}

kernel void omni_codec_output(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
                              device const float *bias [[buffer(2)]], device float *output [[buffer(3)]],
                              device const float *history [[buffer(4)]], constant OmniCodecOutputParams &p [[buffer(5)]],
                              uint t [[thread_position_in_grid]]) {
  if (t >= p.rows) return;
  float sum = bias[0];
  for (uint k = 0; k < p.taps; ++k) {
    const int r = int(t) - int(p.taps - 1) + int(k);
    if (r + int(p.history_rows) < 0) continue;
    device const float *x = r >= 0 ? input + ulong(r) * p.channels : history + ulong(r + int(p.history_rows)) * p.channels;
    device const float *w = weight + k * p.channels;
    for (uint c = 0; c < p.channels; ++c) sum += w[c] * x[c];
  }
  output[t] = clamp(sum, -1.0f, 1.0f);
}

kernel void omni_codec_to_surface(device const float *rows [[buffer(0)]], device half *surface [[buffer(1)]],
                                  constant OmniCodecSurfaceParams &p [[buffer(2)]],
                                  uint index [[thread_position_in_grid]]) {
  if (index >= p.columns * p.channels) return;
  const uint c = index / p.columns;
  const uint t = index % p.columns;
  surface[ulong(c) * p.stride + t] = t < p.rows ? half(rows[ulong(t) * p.channels + c]) : half(0.0h);
}

kernel void omni_codec_from_surface(device const half *surface [[buffer(0)]], device float *rows [[buffer(1)]],
                                    constant OmniCodecSurfaceParams &p [[buffer(2)]],
                                    uint index [[thread_position_in_grid]]) {
  if (index >= p.rows * p.channels) return;
  const uint t = index / p.channels;
  const uint c = index % p.channels;
  rows[index] = float(surface[ulong(c) * p.stride + t]);
}

// Dense bf16 projections (ops::DenseLinear): output[m][n] = epilogue(sum_k
// input[m][k] * weight[n][k]) for weights stored as the checkpoints store
// them, [N][K] row-major bf16. Prefill rows run MPP matmul tiles with fp32
// accumulation. A decode step of one row runs one simdgroup per output
// (omni_gemv_*), each lane streaming eight weights at a time; steps of two
// to sixteen rows run 8x8 simdgroup MMAs on registers (omni_mma_*), so
// every weight is read once per step whatever the row count. Both read a
// packed copy of the weights too (omni_gemv_packed_*, omni_mma_packed_*,
// metal/abi/Omni.h), the same bits in about 0.78 of the bytes, with the
// same arithmetic.
//
// The epilogue is run-time (OmniGemmParams.flags, activation): bias, a
// gate/up pair (silu(gate) * up, Qwen3's SwiGLU, or with biases the
// talker's ResizeMLP is fc2(silu(fc1))), GELU (erf) or SiLU, a residual add,
// into bf16 or fp32 rows.

#include "metal/abi/Omni.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/sgmatrix.h"
#include "metal/kernels/common/split_reduce.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// Abramowitz & Stegun 7.1.26: the error is far below bf16 rounding.
inline float omni_erf(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  x = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * x);
  const float polynomial =
      t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - polynomial * precise::exp(-x * x));
}

inline float omni_gelu(float x) { return 0.5f * x * (1.0f + omni_erf(x * 0.7071067811865476f)); }

inline float omni_silu(float x) { return x / (1.0f + precise::exp(-x)); }

inline float omni_activate(float value, uint activation) {
  if (activation == 1u) return omni_gelu(value);
  if (activation == 2u) return omni_silu(value);
  return value;
}

// The epilogue of one output element from its fp32 sum (and the gate's).
template <typename Output>
inline void omni_gemm_store(float value, float gate, uint row, uint column, device const float *bias,
                            device const float *gate_bias, device const Output *residual, device Output *output,
                            constant OmniGemmParams &p) {
  if (p.flags & OMNI_GEMM_FLAG_BIAS) value += bias[column];
  if (p.flags & OMNI_GEMM_FLAG_GATE_UP) {
    if (p.flags & OMNI_GEMM_FLAG_BIAS) gate += gate_bias[column];
    value = omni_silu(gate) * value;
  } else {
    value = omni_activate(value, p.activation);
  }
  const ulong index = ulong(row) * p.output_stride + column;
  if (p.flags & OMNI_GEMM_FLAG_RESIDUAL) value += float(residual[index]);
  output[index] = Output(value);
}

// A TileM x TileN tile of `Simdgroups` simdgroups: one matmul call over the
// whole K (MPP walks K itself; calls of 64 inputs with a barrier after each
// ran the audio tower's 151-row GEMMs at 9-13 TFLOPS, the whole K at 15-17,
// omni-dense-bench --tower), and for gate/up a second call into the gate's
// sums. The sums of an output are the same bits at every tile and K split
// tried (dev/benchmarks/omni_dense_variants.metal).
template <ushort TileM, ushort TileN, ushort Simdgroups, bool GateUp, typename Output>
inline void omni_gemm_tile(device bfloat *input, device bfloat *weights, device bfloat *gate_weights,
                           device const float *bias, device const float *gate_bias, device const Output *residual,
                           device Output *output, constant OmniGemmParams &p, uint2 group) {
  const uint row_origin = group.x * TileM;
  const uint output_origin = group.y * TileN;
  const int K = int(p.input_size);
  auto a = tensor(input + ulong(row_origin) * p.input_stride, dextents<int, 2>{K, TileM},
                  array<int, 2>{1, int(p.input_stride)});
  auto b = tensor(weights + ulong(output_origin) * p.input_size, dextents<int, 2>{K, TileN}, array<int, 2>{1, K});
  constexpr auto descriptor = matmul2d_descriptor(TileM, TileN, static_cast<int>(dynamic_extent), false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;
  auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
  operation.run(a, b, accumulated);
  if constexpr (GateUp) {
    auto g = tensor(gate_weights + ulong(output_origin) * p.input_size, dextents<int, 2>{K, TileN},
                    array<int, 2>{1, K});
    auto gated = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
#pragma unroll
    for (ushort i = 0; i < gated.get_capacity(); ++i) gated[i] = 0.0f;
    operation.run(a, g, gated);
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      if (!accumulated.is_valid_element(i)) continue;
      auto index = accumulated.get_multidimensional_index(i);
      const uint row = row_origin + index[1];
      if (row >= p.rows) continue;
      omni_gemm_store<Output>(accumulated[i], gated[i], row, output_origin + index[0], bias, gate_bias, residual,
                              output, p);
    }
  } else {
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      if (!accumulated.is_valid_element(i)) continue;
      auto index = accumulated.get_multidimensional_index(i);
      const uint row = row_origin + index[1];
      if (row >= p.rows) continue;
      omni_gemm_store<Output>(accumulated[i], 0.0f, row, output_origin + index[0], bias, gate_bias, residual, output,
                              p);
    }
  }
}

#define OMNI_GEMM_KERNEL(Name, TileM, TileN, Simdgroups, GateUp, Output)                                         \
  kernel void Name(device bfloat *input [[buffer(0)]], device bfloat *weights [[buffer(1)]],                   \
                   device bfloat *gate_weights [[buffer(2)]], device const float *bias [[buffer(3)]],           \
                   device const float *gate_bias [[buffer(4)]], device const Output *residual [[buffer(5)]],    \
                   device Output *output [[buffer(6)]], constant OmniGemmParams &params [[buffer(7)]],          \
                   uint2 group [[threadgroup_position_in_grid]]) {                                             \
    omni_gemm_tile<TileM, TileN, Simdgroups, GateUp, Output>(input, weights, gate_weights, bias, gate_bias,     \
                                                             residual, output, params, group);                 \
  }
// omni_gemm_bf16_m{rows}n{columns}s{simdgroups}, _gu for gate/up, _f32 for
// fp32 rows.
#define OMNI_GEMM_TILE(TileM, TileN, Simdgroups)                                                                  \
  OMNI_GEMM_KERNEL(omni_gemm_bf16_m##TileM##n##TileN##s##Simdgroups, TileM, TileN, Simdgroups, false, bfloat)     \
  OMNI_GEMM_KERNEL(omni_gemm_bf16_m##TileM##n##TileN##s##Simdgroups##_f32, TileM, TileN, Simdgroups, false, float) \
  OMNI_GEMM_KERNEL(omni_gemm_bf16_m##TileM##n##TileN##s##Simdgroups##_gu, TileM, TileN, Simdgroups, true, bfloat)  \
  OMNI_GEMM_KERNEL(omni_gemm_bf16_m##TileM##n##TileN##s##Simdgroups##_gu_f32, TileM, TileN, Simdgroups, true, float)
OMNI_GEMM_TILE(32, 64, 8)
OMNI_GEMM_TILE(32, 128, 4)
OMNI_GEMM_TILE(64, 64, 4)
OMNI_GEMM_TILE(64, 64, 8)
OMNI_GEMM_TILE(64, 128, 4)
OMNI_GEMM_TILE(64, 128, 8)
#undef OMNI_GEMM_TILE
#undef OMNI_GEMM_KERNEL

// Eight bf16 values of a 16-byte vector as floats.
inline void omni_unpack8(uint4 v, thread float *out) {
  out[0] = as_type<float>(v.x << 16);
  out[1] = as_type<float>(v.x & 0xffff0000u);
  out[2] = as_type<float>(v.y << 16);
  out[3] = as_type<float>(v.y & 0xffff0000u);
  out[4] = as_type<float>(v.z << 16);
  out[5] = as_type<float>(v.z & 0xffff0000u);
  out[6] = as_type<float>(v.w << 16);
  out[7] = as_type<float>(v.w & 0xffff0000u);
}

// A column's bf16 weights as the checkpoint stores them.
struct OmniDenseColumn {
  device const uint4 *weights;
  void init(device const uchar *base, uint column, constant OmniGemmParams &p) {
    weights = reinterpret_cast<device const uint4 *>(base + ulong(column) * p.input_size * 2u);
  }
  void unpack8(uint k, thread float *out) const { omni_unpack8(weights[k / 8u], out); }
};

// The bf16 words (weights 0 1, 2 3, 4 5, 6 7) of packed group `group`,
// from its sign/mantissa bytes, code word and block header (and, for an
// escape block, its record).
inline uint4 omni_packed_words(uint2 sm, uint code, uint header, device const uint2 *escapes, uint group) {
  uint2 exponents = uint2(code);
  const uint base = header & 255u;
  const bool escape = base == OMNI_PACKED_ESCAPE;
  if (escape) exponents = escapes[(header >> 8) * (OMNI_PACKED_ESCAPE_BYTES / 8u) + (group & 7u)];
  const uint mask = escape ? 0x7f807f80u : 0x07800780u;
  const uint offset = escape ? 0u : base * 0x00800080u;
  const uint e0 = (rotate(exponents.x, 7u) & mask) + offset;
  const uint e1 = (rotate(exponents.x, escape ? 31u : 3u) & mask) + offset;
  const uint e2 = (rotate(exponents.y, escape ? 7u : 31u) & mask) + offset;
  const uint e3 = (rotate(exponents.y, escape ? 31u : 27u) & mask) + offset;
  const uint v0 = (sm.x & 0x007f007fu) | ((sm.x << 8) & 0x80008000u) | e0;
  const uint v1 = ((sm.x >> 8) & 0x007f007fu) | (sm.x & 0x80008000u) | e1;
  const uint v2 = (sm.y & 0x007f007fu) | ((sm.y << 8) & 0x80008000u) | e2;
  const uint v3 = ((sm.y >> 8) & 0x007f007fu) | (sm.y & 0x80008000u) | e3;
  return uint4(v0, v1, v2, v3);
}

// omni_packed_words of a block that is not an escape.
inline uint4 omni_packed_code_words(uint2 sm, uint code, uint header) {
  const uint offset = (header & 255u) * 0x00800080u;
  const uint e0 = (rotate(code, 7u) & 0x07800780u) + offset;
  const uint e1 = (rotate(code, 3u) & 0x07800780u) + offset;
  const uint e2 = (rotate(code, 31u) & 0x07800780u) + offset;
  const uint e3 = (rotate(code, 27u) & 0x07800780u) + offset;
  const uint v0 = (sm.x & 0x007f007fu) | ((sm.x << 8) & 0x80008000u) | e0;
  const uint v1 = ((sm.x >> 8) & 0x007f007fu) | (sm.x & 0x80008000u) | e1;
  const uint v2 = (sm.y & 0x007f007fu) | ((sm.y << 8) & 0x80008000u) | e2;
  const uint v3 = ((sm.y >> 8) & 0x007f007fu) | (sm.y & 0x80008000u) | e3;
  return uint4(v0, v1, v2, v3);
}

// A column of a packed weight (metal/abi/Omni.h): the same bf16 bits,
// rebuilt two weights a 32-bit word. Each weight's exponent is shifted to
// bits 14:7 of its half: a code nibble or an escape byte rotated into place
// and masked (the rotations' wrapped bits fall outside the masks).
struct OmniPackedColumn {
  device const uint2 *sign_mantissa;
  device const uint *codes;
  device const uint *headers;
  device const uint2 *escapes;
  void init(device const uchar *base, uint column, constant OmniGemmParams &p) {
    const ulong K = p.input_size, weights = ulong(p.output_size) * K;
    sign_mantissa = reinterpret_cast<device const uint2 *>(base + column * K);
    codes = reinterpret_cast<device const uint *>(base + weights + column * K / 2u);
    headers = reinterpret_cast<device const uint *>(base + weights * 3u / 2u + column * K / 16u);
    escapes = reinterpret_cast<device const uint2 *>(base + weights * 3u / 2u + weights / 16u);
  }
  void unpack8(uint k, thread float *out) const {
    const uint group = k / 8u;
    omni_unpack8(omni_packed_words(sign_mantissa[group], codes[group], headers[k / 64u], escapes, group), out);
  }
};

// C output columns per simdgroup over at most R rows (R >= the step's
// rows): lane l streams the weights l*8.. of every 256-wide block of K for
// each column and the matching inputs of each row, so one input load feeds
// C columns; each column's partial sums reduce across the simdgroup in lane
// order, the same order whatever C and R, and whichever Column stores the
// weights.
template <typename Column, typename Output, uint C, uint R>
inline void omni_gemv(device const bfloat *input, device const uchar *weights, device const uchar *gate_weights,
                      device const float *bias, device const float *gate_bias, device const Output *residual,
                      device Output *output, constant OmniGemmParams &p, uint group, uint simd_group,
                      uint lane) {
  const uint first = (group * 8u + simd_group) * C;
  if (first >= p.output_size) return;
  const uint K = p.input_size;
  const bool gate_up = p.flags & OMNI_GEMM_FLAG_GATE_UP;
  Column w[C], g[C];
#pragma unroll
  for (uint c = 0; c < C; ++c) {
    const uint column = min(first + c, p.output_size - 1);
    w[c].init(weights, column, p);
    g[c].init(gate_weights, column, p);
  }
  float sums[R][C], gates[R][C];
#pragma unroll
  for (uint row = 0; row < R; ++row)
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      sums[row][c] = 0.0f;
      gates[row][c] = 0.0f;
    }
  const uint rows = p.rows;
  for (uint k = lane * 8u; k < K; k += 256u) {
    float wv[C][8], gv[C][8];
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      w[c].unpack8(k, wv[c]);
      if (gate_up) g[c].unpack8(k, gv[c]);
    }
#pragma unroll
    for (uint row = 0; row < R; ++row) {
      if (row >= rows) break;
      float xv[8];
      omni_unpack8(*reinterpret_cast<device const uint4 *>(input + ulong(row) * p.input_stride + k), xv);
#pragma unroll
      for (uint c = 0; c < C; ++c) {
        float s = 0.0f, t = 0.0f;
#pragma unroll
        for (uint j = 0; j < 8; ++j) {
          s += xv[j] * wv[c][j];
          if (gate_up) t += xv[j] * gv[c][j];
        }
        sums[row][c] += s;
        gates[row][c] += t;
      }
    }
  }
#pragma unroll
  for (uint row = 0; row < R; ++row)
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      sums[row][c] = simd_sum(sums[row][c]);
      gates[row][c] = simd_sum(gates[row][c]);
    }
  if (lane != 0) return;
#pragma unroll
  for (uint c = 0; c < C; ++c) {
    if (first + c >= p.output_size) break;
#pragma unroll
    for (uint row = 0; row < R; ++row) {
      if (row >= rows) break;
      omni_gemm_store<Output>(sums[row][c], gates[row][c], row, first + c, bias, gate_bias, residual, output, p);
    }
  }
}

#define OMNI_GEMV_KERNEL(Name, Column, Weight, Output, C, R)                                                     \
  kernel void Name(device const bfloat *input [[buffer(0)]], device const Weight *weights [[buffer(1)]],        \
                   device const Weight *gate_weights [[buffer(2)]], device const float *bias [[buffer(3)]],      \
                   device const float *gate_bias [[buffer(4)]], device const Output *residual [[buffer(5)]],     \
                   device Output *output [[buffer(6)]], constant OmniGemmParams &params [[buffer(7)]],           \
                   uint group [[threadgroup_position_in_grid]], uint simd_group [[simdgroup_index_in_threadgroup]], \
                   uint lane [[thread_index_in_simdgroup]]) {                                                   \
    omni_gemv<Column, Output, C, R>(input, reinterpret_cast<device const uchar *>(weights),                     \
                                    reinterpret_cast<device const uchar *>(gate_weights), bias, gate_bias,     \
                                    residual, output, params, group, simd_group, lane);                         \
  }

// The instances ops/Omni.cpp denseTile names: two columns a simdgroup for
// one row of a plain projection, else one, for up to one, two or four rows.
#define OMNI_GEMV_VARIANTS(C, R)                                                                                  \
  OMNI_GEMV_KERNEL(omni_gemv_bf16_c##C##_r##R, OmniDenseColumn, bfloat, bfloat, C, R)                            \
  OMNI_GEMV_KERNEL(omni_gemv_bf16_c##C##_r##R##_f32, OmniDenseColumn, bfloat, float, C, R)
OMNI_GEMV_VARIANTS(1, 1)
OMNI_GEMV_VARIANTS(1, 2)
OMNI_GEMV_VARIANTS(1, 4)
OMNI_GEMV_VARIANTS(2, 1)
#undef OMNI_GEMV_VARIANTS
// Packed weights (metal/abi/Omni.h), one column per simdgroup: two or four
// columns streamed slower at every row count (ops/Omni.cpp denseTile).
#define OMNI_GEMV_PACKED_VARIANTS(R)                                                                              \
  OMNI_GEMV_KERNEL(omni_gemv_packed_c1_r##R, OmniPackedColumn, uchar, bfloat, 1, R)                              \
  OMNI_GEMV_KERNEL(omni_gemv_packed_c1_r##R##_f32, OmniPackedColumn, uchar, float, 1, R)
OMNI_GEMV_PACKED_VARIANTS(1)
OMNI_GEMV_PACKED_VARIANTS(2)
OMNI_GEMV_PACKED_VARIANTS(4)
#undef OMNI_GEMV_PACKED_VARIANTS
#undef OMNI_GEMV_KERNEL

// Decode steps of two to sixteen rows: 8x8 simdgroup MMAs on registers
// (common/sgmatrix.h). Each simdgroup computes W[8 columns][K] x X[8 R rows]^T
// for CF fragments of eight columns, four simdgroups a threadgroup, so every
// weight is read once a step and multiplies all its rows at once (the MPP
// tiles computed 32 rows whatever the step's rows, and their 32 or 64
// threadgroups left most of the GPU idle; ops/Omni.cpp denseTile has the
// timings). In each 64-wide chunk of K the lane of (fm, fn) holds 16
// consecutive weights of column fm from (fn / 2) * 16, so MMA j of the
// chunk takes their pair j as its k = fn, fn + 1 with no shuffle; its
// activation operand holds rows fn and fn + 1 at the same input, (fm / 2) *
// 16 + 2 j + (fm & 1), one half of each row's word. A column's sum adds the
// chunks in K order, each chunk's eight MMAs in order, so the arithmetic of
// a row depends on neither the step's rows, R, CF, the chunks a loop
// iteration loads (U) nor which source stores the weights.
//
// grid.y splits K into contiguous ranges of whole chunks (the host takes
// splits that divide them); each split stores its live rows' fp32 sums and
// the last to arrive adds them in split order (common/split_reduce.h)
// before the epilogue. The epilogue is omni_gemm_store's: gate and up keep
// separate fp32 sums until the SiLU product, a residual adds to the fp32
// projection, and only the final store rounds.
constant constexpr uint kOmniMmaSimdgroups = OMNI_MMA_THREADS / 32u;

// A lane's 16 weights of column n from (part * 16) in each 64-wide chunk, as
// eight bf16 pairs: fetch() loads a chunk, words() forms its pairs.
struct OmniMmaDense {
  device const uint4 *weights;
  struct Raw {
    uint4 low, high;
  };
  void init(device const uchar *base, uint column, uint part, constant OmniGemmParams &p) {
    weights = reinterpret_cast<device const uint4 *>(base + (ulong(column) * p.input_size + part * 16u) * 2u);
  }
  Raw fetch(uint chunk) const { return {weights[chunk * 8u], weights[chunk * 8u + 1u]}; }
  void words(Raw raw, thread uint4 (&out)[2]) const {
    out[0] = raw.low;
    out[1] = raw.high;
  }
};
// The same weights from the packed copy (metal/abi/Omni.h): the lane's two
// groups of eight are 16 sign/mantissa bytes, two code words and the
// chunk's block header, rebuilt to the bf16 words as the packed GEMV does.
struct OmniMmaPacked {
  device const uint4 *sign_mantissa;
  device const uint2 *codes;
  device const uint *headers;
  device const uint2 *escapes;
  uint part;
  struct Raw {
    uint4 sign_mantissa;
    uint2 codes;
    uint header;
  };
  void init(device const uchar *base, uint column, uint lane_part, constant OmniGemmParams &p) {
    const ulong K = p.input_size, weights = ulong(p.output_size) * K;
    part = lane_part;
    sign_mantissa = reinterpret_cast<device const uint4 *>(base + column * K + part * 16u);
    codes = reinterpret_cast<device const uint2 *>(base + weights + column * K / 2u + part * 8u);
    headers = reinterpret_cast<device const uint *>(base + weights * 3u / 2u + column * K / 16u);
    escapes = reinterpret_cast<device const uint2 *>(base + weights * 3u / 2u + weights / 16u);
  }
  Raw fetch(uint chunk) const { return {sign_mantissa[chunk * 4u], codes[chunk * 4u], headers[chunk]}; }
  void words(Raw raw, thread uint4 (&out)[2]) const {
    // Escape blocks are 0.3% of the checkpoints' blocks: a simdgroup whose
    // eight columns hold none takes the code path without the escape
    // selects (eight rows of gate/up 165.9 -> 158.1 us, the ASR head 1805
    // -> 1764).
    if (!simd_any((raw.header & 255u) == OMNI_PACKED_ESCAPE)) {
      out[0] = omni_packed_code_words(raw.sign_mantissa.xy, raw.codes.x, raw.header);
      out[1] = omni_packed_code_words(raw.sign_mantissa.zw, raw.codes.y, raw.header);
      return;
    }
    out[0] = omni_packed_words(raw.sign_mantissa.xy, raw.codes.x, raw.header, escapes, part * 2u);
    out[1] = omni_packed_words(raw.sign_mantissa.zw, raw.codes.y, raw.header, escapes, part * 2u + 1u);
  }
};

// The split partials: [split][stream][rows][N] fp32 of the live rows (stream
// 1 the gate's).
inline ulong omni_mma_partial(uint split, uint stream, uint row, uint column, constant OmniGemmParams &p) {
  return ((ulong(split) * 2u + stream) * p.rows + row) * p.output_size + column;
}

// R row fragments (rows 8 R) and CF column fragments a simdgroup, U chunks
// loaded a loop iteration. Gate/up is compile-time: its second weight's
// registers would otherwise spill the others (180 against 241 us for
// eight rows of gate/up when it was a run-time flag).
template <typename Source, uint R, uint CF, uint U, bool GateUp, typename Output>
inline void omni_mma(device const bfloat *input, device const uchar *weights, device const uchar *gate_weights,
                     device const float *bias, device const float *gate_bias, device const Output *residual,
                     device Output *output, constant OmniGemmParams &p, device coherent(device) float *partials,
                     device atomic_uint *counters, uint2 group, uint2 grid, uint thread_index, uint simd_group,
                     uint lane, threadgroup uint *arrival) {
  const uint rows = p.rows;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint fm = l.fm, fn = l.fn;
  const uint column_base = (group.x * kOmniMmaSimdgroups + simd_group) * CF * 8u;
  const uint splits = grid.y, chunks = p.input_size / 64u / splits, first = group.y * chunks;
  Source w[CF], g[CF];
#pragma unroll
  for (uint c = 0; c < CF; ++c) {
    w[c].init(weights, column_base + c * 8u + fm, fn / 2u, p);
    if (GateUp) g[c].init(gate_weights, column_base + c * 8u + fm, fn / 2u, p);
  }
  // Rows past the step's read as zeros (and their sums are not stored).
  device const uint4 *x[R][2];
  bool live[R][2];
#pragma unroll
  for (uint r = 0; r < R; ++r)
#pragma unroll
    for (uint h = 0; h < 2; ++h) {
      const uint row = r * 8u + fn + h;
      live[r][h] = row < rows;
      x[r][h] = reinterpret_cast<device const uint4 *>(input + ulong(min(row, rows - 1u)) * p.input_stride +
                                                       (fm / 2u) * 16u);
    }
  const uint shift = (fm & 1u) * 16u;
  float2 sums[CF][R], gates[CF][R];
#pragma unroll
  for (uint c = 0; c < CF; ++c)
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      sums[c][r] = float2(0.0f);
      gates[c][r] = float2(0.0f);
    }
  for (uint chunk = first; chunk < first + chunks; chunk += U) {
    typename Source::Raw raw[U][CF], gate_raw[U][CF];
#pragma unroll
    for (uint u = 0; u < U; ++u)
#pragma unroll
      for (uint c = 0; c < CF; ++c) {
        raw[u][c] = w[c].fetch(chunk + u);
        if (GateUp) gate_raw[u][c] = g[c].fetch(chunk + u);
      }
#pragma unroll
    for (uint u = 0; u < U; ++u) {
      uint4 xv[R][2][2], wv[CF][2], gv[CF][2];
#pragma unroll
      for (uint r = 0; r < R; ++r)
#pragma unroll
        for (uint h = 0; h < 2; ++h) {
          xv[r][h][0] = live[r][h] ? x[r][h][(chunk + u) * 8u] : uint4(0);
          xv[r][h][1] = live[r][h] ? x[r][h][(chunk + u) * 8u + 1u] : uint4(0);
        }
#pragma unroll
      for (uint c = 0; c < CF; ++c) {
        w[c].words(raw[u][c], wv[c]);
        if (GateUp) g[c].words(gate_raw[u][c], gv[c]);
      }
#pragma unroll
      for (uint j = 0; j < 8; ++j) {
        vec<bfloat, 2> b[R];
#pragma unroll
        for (uint r = 0; r < R; ++r)
          b[r] = as_type<vec<bfloat, 2>>(
              insert_bits(xv[r][0][j / 4][j % 4] >> shift, xv[r][1][j / 4][j % 4] >> shift, 16u, 16u));
#pragma unroll
        for (uint c = 0; c < CF; ++c) {
          const vec<bfloat, 2> a = as_type<vec<bfloat, 2>>(wv[c][j / 4][j % 4]);
#pragma unroll
          for (uint r = 0; r < R; ++r) sgmatrix::mma_acc<bfloat>(sums[c][r], a, b[r]);
          if (GateUp) {
            const vec<bfloat, 2> ag = as_type<vec<bfloat, 2>>(gv[c][j / 4][j % 4]);
#pragma unroll
            for (uint r = 0; r < R; ++r) sgmatrix::mma_acc<bfloat>(gates[c][r], ag, b[r]);
          }
        }
      }
    }
  }
  if (splits > 1) {
#pragma unroll
    for (uint c = 0; c < CF; ++c)
#pragma unroll
      for (uint r = 0; r < R; ++r)
#pragma unroll
        for (uint h = 0; h < 2; ++h) {
          if (!live[r][h]) continue;
          const uint column = column_base + c * 8u + fm, row = r * 8u + fn + h;
          partials[omni_mma_partial(group.y, 0, row, column, p)] = sums[c][r][h];
          if (GateUp) partials[omni_mma_partial(group.y, 1, row, column, p)] = gates[c][r][h];
        }
    if (!split_arrive_last(counters + group.x, splits, thread_index, arrival)) return;
#pragma unroll
    for (uint c = 0; c < CF; ++c)
#pragma unroll
      for (uint r = 0; r < R; ++r)
#pragma unroll
        for (uint h = 0; h < 2; ++h) {
          if (!live[r][h]) continue;
          const uint column = column_base + c * 8u + fm, row = r * 8u + fn + h;
          sums[c][r][h] = split_sum(float(sums[c][r][h]), group.y, splits,
                                    [&](uint s) { return partials[omni_mma_partial(s, 0, row, column, p)]; });
          if (GateUp)
            gates[c][r][h] = split_sum(float(gates[c][r][h]), group.y, splits,
                                       [&](uint s) { return partials[omni_mma_partial(s, 1, row, column, p)]; });
        }
    split_release(counters + group.x, thread_index);
  }
#pragma unroll
  for (uint c = 0; c < CF; ++c)
#pragma unroll
    for (uint r = 0; r < R; ++r)
#pragma unroll
      for (uint h = 0; h < 2; ++h) {
        if (!live[r][h]) continue;
        omni_gemm_store<Output>(sums[c][r][h], gates[c][r][h], r * 8u + fn + h, column_base + c * 8u + fm, bias,
                                gate_bias, residual, output, p);
      }
}

// The split partials and counters bind after the GEMM buffers, the
// parameters last; an unsplit dispatch reads neither.
#define OMNI_MMA_KERNEL(Name, Source, R, CF, U, GateUp, Output)                                                  \
  kernel void Name(device const bfloat *input [[buffer(0)]], device const uchar *weights [[buffer(1)]],         \
                   device const uchar *gate_weights [[buffer(2)]], device const float *bias [[buffer(3)]],       \
                   device const float *gate_bias [[buffer(4)]], device const Output *residual [[buffer(5)]],     \
                   device Output *output [[buffer(6)]], device coherent(device) float *partials [[buffer(7)]],   \
                   device atomic_uint *counters [[buffer(8)]], constant OmniGemmParams &params [[buffer(9)]],    \
                   uint2 group [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],           \
                   uint thread_index [[thread_index_in_threadgroup]],                                          \
                   uint simd_group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup uint arrival;                                                                                   \
    omni_mma<Source, R, CF, U, GateUp, Output>(input, weights, gate_weights, bias, gate_bias, residual, output,  \
                                               params, partials, counters, group, grid, thread_index,          \
                                               simd_group, lane, &arrival);                                     \
  }
#define OMNI_MMA_VARIANTS(Name, Source, R, CF, U)                                                                 \
  OMNI_MMA_KERNEL(Name, Source, R, CF, U, false, bfloat)                                                        \
  OMNI_MMA_KERNEL(Name##_f32, Source, R, CF, U, false, float)                                                   \
  OMNI_MMA_KERNEL(Name##_gu, Source, R, CF, U, true, bfloat)                                                    \
  OMNI_MMA_KERNEL(Name##_gu_f32, Source, R, CF, U, true, float)
// Up to eight rows one column fragment a simdgroup; nine to sixteen two,
// whose two row fragments' activation pairs then serve two weight
// fragments. Packed weights load two chunks an iteration (bf16 ones gain
// nothing from it), and from nine rows stay bf16: rebuilding the words
// costs more than the bytes it saves once the MMAs double.
OMNI_MMA_VARIANTS(omni_mma_bf16_r8, OmniMmaDense, 1, 1, 1)
OMNI_MMA_VARIANTS(omni_mma_bf16_r16, OmniMmaDense, 2, 2, 1)
OMNI_MMA_VARIANTS(omni_mma_packed_r8, OmniMmaPacked, 1, 1, 2)
#undef OMNI_MMA_VARIANTS
#undef OMNI_MMA_KERNEL

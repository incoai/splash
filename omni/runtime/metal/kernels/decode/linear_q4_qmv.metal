// Q4 decode projections of one live row, or two, LinearTile::Qmv
// (ops::Linear): a single ASR request's or TTS stream's step (or two's)
// through the affine storage tiles without the MPP tile, its barriers or a
// K split.
//
// The geometry follows MLX's qmv_fast (mlx/backend/metal/kernels/
// quantized.h, MIT license, Copyright 2023-2024 Apple Inc.; see
// THIRD_PARTY_NOTICES): each simdgroup owns Results adjacent outputs and all
// 32 lanes cooperate on every output, lane l taking 16 consecutive inputs of
// each 512-input step, so a step's activation chunk and its sum serve all of
// the simdgroup's outputs. Codes are masked in place, not shifted: the
// activations are prescaled by 1, 1/16, 1/256 and 1/4096 so each masked
// nibble's place value cancels. A chunk adds
// scale * dot(codes, x) + bias * sum(x) in fp32, and one simd_sum per output
// ends the projection; there are no barriers, partials or counters.
//
// The codes are read where ops::Projection keeps them, not in MLX's
// row-major order: output n's quant group g is parameter
// p = ((n / 256) * (K / 64) + g) * 256 + n % 256 of the [N / 256][K / 64][256]
// planes, its 64 codes the 32 bytes at 32 p (input 2i in the low nibble of
// byte i) and its scale and bias element p. Lane l reads 8 bytes at
// 32 p + 8 (l % 4); a simdgroup's adjacent outputs make each quant group's
// reads one contiguous run of code rows.
//
// Epilogues as q4_store_output (kernels/common/q4_mpp_tiles.h): one bf16
// rounding of the projection (none into an fp32 destination), the residual
// added after it, gate/up's SiLU product of the bf16 gate and up values.
// Norm projects bf16((x * inverse RMS) * weight), Normalization::addRms's
// bits: the threadgroup's 256 threads reduce the row's squares as
// rms_inverse does, so the inverse is the norm dispatch's to the bit and a
// folded projection equals the norm dispatch and the plain projection.
//
// Numerics: fp32 throughout, each lane's per-chunk terms in K order, then
// the lanes by simd_sum: the MPP tiles' per-group arithmetic
// (dot * scale + sum(x) * bias) in a different order, so outputs differ
// from theirs by fp32 association before the one bf16 rounding. Results
// does not change a sum.

#include "metal/abi/KernelABI.h"
#include "metal/abi/Linear.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/rms_inverse.h"

namespace q4_qmv {

constant constexpr uint kStorageRows = SPLASH_AFFINE_TILE_ROWS;
// A lane's inputs per step: a quarter of a quant group.
constant constexpr ushort kValues = 16;
constant constexpr uint kStride = 32 * kValues;
// The threadgroup: eight simdgroups, rms_inverse's 256 threads.
constant constexpr ushort kSimdgroups = 8;

enum class Mode : ushort { Plain, Residual, GateUp };

// The lane's kValues activations from `k` as floats, normalized by
// `inverse` and the norm's weights when Norm, and their sum; then each
// prescaled by the place value of its nibble within a 16-bit code word.
template <bool Norm>
inline float load_chunk(device const bfloat *input, device const bfloat *weight, uint k, float inverse,
                        thread float (&x)[kValues]) {
  device const uint4 *packed = reinterpret_cast<device const uint4 *>(input + k);
#pragma unroll
  for (ushort v = 0; v < kValues / 8; ++v) {
    const uint4 bits = packed[v];
    const uint words[4] = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
    for (ushort w = 0; w < 4; ++w) {
      x[8 * v + 2 * w] = as_type<float>(words[w] << 16);
      x[8 * v + 2 * w + 1] = as_type<float>(words[w] & 0xffff0000u);
    }
  }
  if (Norm) {
    device const uint4 *scaled = reinterpret_cast<device const uint4 *>(weight + k);
#pragma unroll
    for (ushort v = 0; v < kValues / 8; ++v) {
      const uint4 bits = scaled[v];
      const uint words[4] = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
      for (ushort w = 0; w < 4; ++w) {
        const float a = as_type<float>(words[w] << 16), b = as_type<float>(words[w] & 0xffff0000u);
        x[8 * v + 2 * w] = float(bfloat((x[8 * v + 2 * w] * inverse) * a));
        x[8 * v + 2 * w + 1] = float(bfloat((x[8 * v + 2 * w + 1] * inverse) * b));
      }
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (ushort i = 0; i < kValues; ++i)
    sum += x[i];
#pragma unroll
  for (ushort i = 0; i < kValues; ++i)
    x[i] *= 1.0f / float(1u << (4 * (i % 4)));
  return sum;
}

// dot(codes, x) of a lane's chunk: four 16-bit words of four codes each.
inline float chunk_dot(uint2 codes, const thread float (&x)[kValues]) {
  const ushort words[4] = {ushort(codes.x), ushort(codes.x >> 16), ushort(codes.y), ushort(codes.y >> 16)};
  float dot = 0.0f;
#pragma unroll
  for (ushort j = 0; j < 4; ++j) {
    const ushort w = words[j];
    dot += x[4 * j] * float(w & 0x000f) + x[4 * j + 1] * float(w & 0x00f0) + x[4 * j + 2] * float(w & 0x0f00) +
           x[4 * j + 3] * float(w & 0xf000);
  }
  return dot;
}

// The projection's planes; GateUp's up planes after the gate's.
struct Planes {
  device const uchar *weights;
  device const bfloat *scales;
  device const bfloat *biases;
  device const uchar *up_weights;
  device const bfloat *up_scales;
  device const bfloat *up_biases;
};

// The simdgroup's Results outputs from first_output.
template <ushort Results, Mode M, bool Norm, class Out>
inline void project(const thread Planes &planes, device const bfloat *input, device const bfloat *auxiliary,
                    device const bfloat *norm_weight, device Out *output, uint input_size, uint first_output,
                    uint lane, float inverse) {
  constexpr bool GateUp = M == Mode::GateUp;
  // Parameter p of the simdgroup's first output at quant group 0; output r
  // of the run at group g is p + g * 256 + r.
  const ulong first =
      ulong(first_output / kStorageRows) * (input_size / 64) * kStorageRows + first_output % kStorageRows;
  const uint code_offset = (lane % (64 / kValues)) * (kValues / 2);
  float acc[Results], up_acc[Results];
#pragma unroll
  for (ushort r = 0; r < Results; ++r) {
    acc[r] = 0.0f;
    up_acc[r] = 0.0f;
  }
  for (uint k = lane * kValues; k < input_size; k += kStride) {
    float x[kValues];
    const float sum = load_chunk<Norm>(input, norm_weight, k, inverse, x);
    const ulong group = first + ulong(k / 64) * kStorageRows;
#pragma unroll
    for (ushort r = 0; r < Results; ++r) {
      const ulong p = group + r;
      const uint2 codes = *reinterpret_cast<device const uint2 *>(planes.weights + p * 32 + code_offset);
      acc[r] += float(planes.scales[p]) * chunk_dot(codes, x) + float(planes.biases[p]) * sum;
      if (GateUp) {
        const uint2 up_codes = *reinterpret_cast<device const uint2 *>(planes.up_weights + p * 32 + code_offset);
        up_acc[r] += float(planes.up_scales[p]) * chunk_dot(up_codes, x) + float(planes.up_biases[p]) * sum;
      }
    }
  }
#pragma unroll
  for (ushort r = 0; r < Results; ++r) {
    const float total = simd_sum(acc[r]);
    const float up_total = GateUp ? simd_sum(up_acc[r]) : 0.0f;
    // Lane r stores output r, so the run's stores are one instruction.
    if (lane != r)
      continue;
    const uint n = first_output + r;
    float value;
    if (GateUp)
      value = splash_silu(float(bfloat(total))) * float(bfloat(up_total));
    else if (is_same_v<Out, float>)
      value = total;
    else
      value = float(bfloat(total));
    if (M == Mode::Residual)
      value += float(auxiliary[n]);
    output[n] = Out(value);
  }
}

// project() for two live rows that share each step's code loads: both
// rows' chunks are loaded first, then each output's codes are masked into
// floats once and both rows' dot products read them, each row's terms in
// project()'s order, so a row's sum is the one-row kernel's to the bit.
// More rows do not pay: at two the kernel's arithmetic already takes about
// the time of its loads, and a third row costs nearly a second dispatch's.
constant constexpr ushort kSharedRows = 2;
template <ushort Results, Mode M, class Out>
inline void project_shared(const thread Planes &planes, device const bfloat *input, device const bfloat *auxiliary,
                           device Out *output, uint input_size, uint output_size, uint first_output, uint lane) {
  constexpr bool GateUp = M == Mode::GateUp;
  const ulong first =
      ulong(first_output / kStorageRows) * (input_size / 64) * kStorageRows + first_output % kStorageRows;
  const uint code_offset = (lane % (64 / kValues)) * (kValues / 2);
  float acc[Results][kSharedRows], up_acc[Results][kSharedRows];
#pragma unroll
  for (ushort r = 0; r < Results; ++r)
#pragma unroll
    for (ushort row = 0; row < kSharedRows; ++row) {
      acc[r][row] = 0.0f;
      up_acc[r][row] = 0.0f;
    }
  // chunk_dot's sum over codes masked once into c.
  const auto dot = [](const thread float (&c)[kValues], const thread float (&x)[kValues]) {
    float total = 0.0f;
#pragma unroll
    for (ushort j = 0; j < 4; ++j)
      total += x[4 * j] * c[4 * j] + x[4 * j + 1] * c[4 * j + 1] + x[4 * j + 2] * c[4 * j + 2] +
               x[4 * j + 3] * c[4 * j + 3];
    return total;
  };
  const auto masked = [](uint2 codes, thread float (&c)[kValues]) {
    const ushort words[4] = {ushort(codes.x), ushort(codes.x >> 16), ushort(codes.y), ushort(codes.y >> 16)};
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      c[4 * j] = float(words[j] & 0x000f);
      c[4 * j + 1] = float(words[j] & 0x00f0);
      c[4 * j + 2] = float(words[j] & 0x0f00);
      c[4 * j + 3] = float(words[j] & 0xf000);
    }
  };
  for (uint k = lane * kValues; k < input_size; k += kStride) {
    float x[kSharedRows][kValues], sum[kSharedRows];
#pragma unroll
    for (ushort row = 0; row < kSharedRows; ++row)
      sum[row] = load_chunk<false>(input + ulong(row) * input_size, input, k, 1.0f, x[row]);
    const ulong group = first + ulong(k / 64) * kStorageRows;
#pragma unroll
    for (ushort r = 0; r < Results; ++r) {
      const ulong p = group + r;
      float c[kValues];
      masked(*reinterpret_cast<device const uint2 *>(planes.weights + p * 32 + code_offset), c);
      const float scale = float(planes.scales[p]), bias = float(planes.biases[p]);
#pragma unroll
      for (ushort row = 0; row < kSharedRows; ++row)
        acc[r][row] += scale * dot(c, x[row]) + bias * sum[row];
      if (GateUp) {
        masked(*reinterpret_cast<device const uint2 *>(planes.up_weights + p * 32 + code_offset), c);
        const float up_scale = float(planes.up_scales[p]), up_bias = float(planes.up_biases[p]);
#pragma unroll
        for (ushort row = 0; row < kSharedRows; ++row)
          up_acc[r][row] += up_scale * dot(c, x[row]) + up_bias * sum[row];
      }
    }
  }
#pragma unroll
  for (ushort r = 0; r < Results; ++r)
#pragma unroll
    for (ushort row = 0; row < kSharedRows; ++row) {
      const float total = simd_sum(acc[r][row]);
      const float up_total = GateUp ? simd_sum(up_acc[r][row]) : 0.0f;
      if (lane != r)
        continue;
      const ulong at = ulong(row) * output_size + first_output + r;
      float value;
      if (GateUp)
        value = splash_silu(float(bfloat(total))) * float(bfloat(up_total));
      else if (is_same_v<Out, float>)
        value = total;
      else
        value = float(bfloat(total));
      if (M == Mode::Residual)
        value += float(auxiliary[at]);
      output[at] = Out(value);
    }
}

} // namespace q4_qmv

// Threadgroup (0, g) holds the simdgroups' runs of Results outputs from
// g * kSimdgroups * Results.
#define Q4_QMV_THREADS                                                         \
  uint2 group [[threadgroup_position_in_grid]],                                \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simd_group [[simdgroup_index_in_threadgroup]]
// The row's inverse RMS: the norm forms reduce it as rms_inverse does, the
// others take one.
#define Q4_QMV_INVERSE_false 1.0f
#define Q4_QMV_INVERSE_true                                                    \
  rms_inverse(input, params.input_size, reductions, simd_group * 32 + lane,    \
              lane, simd_group)
#define Q4_QMV_REDUCTIONS_false
#define Q4_QMV_REDUCTIONS_true threadgroup float reductions[q4_qmv::kSimdgroups];
#define Q4_QMV_CALL(Results, M, Norm, PlaneList, Auxiliary, NormWeight)        \
  Q4_QMV_REDUCTIONS_##Norm const float inverse = Q4_QMV_INVERSE_##Norm;        \
  const q4_qmv::Planes planes = PlaneList;                                     \
  q4_qmv::project<Results, q4_qmv::Mode::M, Norm>(                             \
      planes, input, Auxiliary, NormWeight, output, params.input_size,         \
      (group.y * q4_qmv::kSimdgroups + simd_group) * Results, lane, inverse);

// The buffers are the sequential decode kernels' (ops::Linear::add): plain
// (input, weights, scales, biases, output), residual (..., residual,
// output), gate/up (input, the gate's planes, output, the up planes); the
// norm forms then the norm's weights; then Q4Params.
#define Q4_QMV_PLAIN(Name, Results, Norm, Out)                                 \
  kernel void Name(device const bfloat *input [[buffer(0)]],                   \
                   device const uchar *weights [[buffer(1)]],                  \
                   device const bfloat *scales [[buffer(2)]],                  \
                   device const bfloat *biases [[buffer(3)]],                  \
                   device Out *output [[buffer(4)]],                           \
                   Q4_QMV_NORM_BUFFER_##Norm(5)                                \
                   constant Q4Params &params [[buffer(Q4_QMV_AFTER_##Norm(5))]], \
                   Q4_QMV_THREADS) {                                           \
    Q4_QMV_CALL(Results, Plain, Norm,                                          \
                (q4_qmv::Planes{weights, scales, biases, weights, scales,      \
                                biases}),                                      \
                input, Q4_QMV_NORM_WEIGHT_##Norm)                              \
  }
#define Q4_QMV_RESIDUAL(Name, Results)                                         \
  kernel void Name(device const bfloat *input [[buffer(0)]],                   \
                   device const uchar *weights [[buffer(1)]],                  \
                   device const bfloat *scales [[buffer(2)]],                  \
                   device const bfloat *biases [[buffer(3)]],                  \
                   device const bfloat *residual [[buffer(4)]],                \
                   device bfloat *output [[buffer(5)]],                        \
                   constant Q4Params &params [[buffer(6)]], Q4_QMV_THREADS) {  \
    Q4_QMV_CALL(Results, Residual, false,                                      \
                (q4_qmv::Planes{weights, scales, biases, weights, scales,      \
                                biases}),                                      \
                residual, input)                                               \
  }
#define Q4_QMV_GATE_UP(Name, Results, Norm)                                    \
  kernel void Name(device const bfloat *input [[buffer(0)]],                   \
                   device const uchar *weights [[buffer(1)]],                  \
                   device const bfloat *scales [[buffer(2)]],                  \
                   device const bfloat *biases [[buffer(3)]],                  \
                   device bfloat *output [[buffer(4)]],                        \
                   device const uchar *up_weights [[buffer(5)]],               \
                   device const bfloat *up_scales [[buffer(6)]],               \
                   device const bfloat *up_biases [[buffer(7)]],               \
                   Q4_QMV_NORM_BUFFER_##Norm(8)                                \
                   constant Q4Params &params [[buffer(Q4_QMV_AFTER_##Norm(8))]], \
                   Q4_QMV_THREADS) {                                           \
    Q4_QMV_CALL(Results, GateUp, Norm,                                         \
                (q4_qmv::Planes{weights, scales, biases, up_weights,           \
                                up_scales, up_biases}),                        \
                input, Q4_QMV_NORM_WEIGHT_##Norm)                              \
  }
#define Q4_QMV_NORM_BUFFER_false(Index)
#define Q4_QMV_NORM_BUFFER_true(Index)                                         \
  device const bfloat *norm_weight [[buffer(Index)]],
#define Q4_QMV_AFTER_false(Index) Index
#define Q4_QMV_AFTER_true(Index) Index + 1
#define Q4_QMV_NORM_WEIGHT_false input
#define Q4_QMV_NORM_WEIGHT_true norm_weight

// Every form at Results outputs per simdgroup (ops::LinearConfig::
// simdgroupOutputs; Suffix _o2 for two): plain into bf16 and fp32 (_f32,
// the logits), the same normalizing their input (_norm), residual and
// gate/up, plain and normalizing.
#define Q4_QMV_FORMS(Suffix, Results)                                          \
  Q4_QMV_PLAIN(decode_linear_q4_qmv##Suffix, Results, false, bfloat)           \
  Q4_QMV_PLAIN(decode_linear_q4_qmv##Suffix##_f32, Results, false, float)      \
  Q4_QMV_PLAIN(decode_linear_q4_qmv_norm##Suffix, Results, true, bfloat)       \
  Q4_QMV_PLAIN(decode_linear_q4_qmv_norm##Suffix##_f32, Results, true, float)  \
  Q4_QMV_RESIDUAL(decode_linear_q4_qmv_residual##Suffix, Results)              \
  Q4_QMV_GATE_UP(decode_linear_q4_qmv_gate_up##Suffix, Results, false)         \
  Q4_QMV_GATE_UP(decode_linear_q4_qmv_gate_up_norm##Suffix, Results, true)
Q4_QMV_FORMS(, 4)
Q4_QMV_FORMS(_o2, 2)
#undef Q4_QMV_FORMS

// The forms of two live rows sharing a threadgroup (project_shared), grid
// (1, column blocks), suffix _r2: plain into bf16 and fp32, residual and
// gate/up, after the norm's dispatch (folding it into every threadgroup
// costs more than the dispatch at two rows).
#define Q4_QMV_SHARED_CALL(Results, M, PlaneList, Auxiliary)                   \
  const q4_qmv::Planes planes = PlaneList;                                     \
  q4_qmv::project_shared<Results, q4_qmv::Mode::M>(                            \
      planes, input, Auxiliary, output, params.input_size, params.output_size, \
      (group.y * q4_qmv::kSimdgroups + simd_group) * Results, lane);
#define Q4_QMV_SHARED_PLAIN(Name, Out)                                         \
  kernel void Name(device const bfloat *input [[buffer(0)]],                   \
                   device const uchar *weights [[buffer(1)]],                  \
                   device const bfloat *scales [[buffer(2)]],                  \
                   device const bfloat *biases [[buffer(3)]],                  \
                   device Out *output [[buffer(4)]],                           \
                   constant Q4Params &params [[buffer(5)]], Q4_QMV_THREADS) {  \
    Q4_QMV_SHARED_CALL(4, Plain,                                               \
                       (q4_qmv::Planes{weights, scales, biases, weights,       \
                                       scales, biases}),                       \
                       input)                                                  \
  }
Q4_QMV_SHARED_PLAIN(decode_linear_q4_qmv_r2, bfloat)
Q4_QMV_SHARED_PLAIN(decode_linear_q4_qmv_r2_f32, float)
kernel void decode_linear_q4_qmv_residual_r2(device const bfloat *input [[buffer(0)]],
                                             device const uchar *weights [[buffer(1)]],
                                             device const bfloat *scales [[buffer(2)]],
                                             device const bfloat *biases [[buffer(3)]],
                                             device const bfloat *residual [[buffer(4)]],
                                             device bfloat *output [[buffer(5)]],
                                             constant Q4Params &params [[buffer(6)]], Q4_QMV_THREADS) {
  Q4_QMV_SHARED_CALL(4, Residual, (q4_qmv::Planes{weights, scales, biases, weights, scales, biases}), residual)
}
kernel void decode_linear_q4_qmv_gate_up_r2(device const bfloat *input [[buffer(0)]],
                                            device const uchar *weights [[buffer(1)]],
                                            device const bfloat *scales [[buffer(2)]],
                                            device const bfloat *biases [[buffer(3)]],
                                            device bfloat *output [[buffer(4)]],
                                            device const uchar *up_weights [[buffer(5)]],
                                            device const bfloat *up_scales [[buffer(6)]],
                                            device const bfloat *up_biases [[buffer(7)]],
                                            constant Q4Params &params [[buffer(8)]], Q4_QMV_THREADS) {
  Q4_QMV_SHARED_CALL(2, GateUp, (q4_qmv::Planes{weights, scales, biases, up_weights, up_scales, up_biases}), input)
}
#undef Q4_QMV_SHARED_PLAIN
#undef Q4_QMV_SHARED_CALL
#undef Q4_QMV_NORM_WEIGHT_true
#undef Q4_QMV_NORM_WEIGHT_false
#undef Q4_QMV_AFTER_true
#undef Q4_QMV_AFTER_false
#undef Q4_QMV_NORM_BUFFER_true
#undef Q4_QMV_NORM_BUFFER_false
#undef Q4_QMV_GATE_UP
#undef Q4_QMV_RESIDUAL
#undef Q4_QMV_PLAIN
#undef Q4_QMV_CALL
#undef Q4_QMV_REDUCTIONS_true
#undef Q4_QMV_REDUCTIONS_false
#undef Q4_QMV_INVERSE_true
#undef Q4_QMV_INVERSE_false
#undef Q4_QMV_THREADS

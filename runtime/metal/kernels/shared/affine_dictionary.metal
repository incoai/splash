// Exact raw-BF16 coefficient dictionary experiment. Original AF4 codes,
// FP32 affine expression, FP16 staging and split-order reduction are shared.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_decode.h"
struct AffineDictionaryMeta {
  static constexpr constant uint Bytes = 2;
  device const uint *pairs;
  uint operator()(device uchar *p) const thread { return pairs[*reinterpret_cast<device const ushort *>(p)]; }
};
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
kernel void affine_dictionary_decode(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],
                        device uchar *meta [[buffer(3)]], device Out *output [[buffer(4)]],
                        device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]],
                        device bfloat *aux [[buffer(7)]], device const uint *pairs [[buffer(8)]], constant GgufDecodeParams &p [[buffer(9)]],
                        uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half2 tl[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS);
  threadgroup half stage[kStagedStages];
  threadgroup uint arrival;
  gguf_decode_tile<F, Rows, Ep>(input, w0, w1, meta, output, partials, counters, aux, p, group, simd_lane, simd_group,
                                stage, tl, &arrival, AffineDictionaryMeta{pairs});
}

template <class F, ushort Rows>
kernel void affine_dictionary_gate_up(device bfloat *input [[buffer(0)]], device uchar *gw0 [[buffer(1)]],
                                device uchar *gw1 [[buffer(2)]], device uchar *gmeta [[buffer(3)]],
                                device uchar *uw0 [[buffer(4)]], device uchar *uw1 [[buffer(5)]],
                                device uchar *umeta [[buffer(6)]], device bfloat *output [[buffer(7)]],
                                device coherent(device) float *partials [[buffer(8)]],
                                device atomic_uint *counters [[buffer(9)]], device const uint *gate_pairs [[buffer(10)]], device const uint *up_pairs [[buffer(11)]], constant GgufDecodeParams &p [[buffer(12)]],
                                uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half2 tl[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, lane, GGUF_GATE_UP_THREADS);
  threadgroup half stage[2 * GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP];
  threadgroup uint arrival;
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits, sb = group.y * per, se = sb + per,
             origin = group.x * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  auto gate = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, stage);
  auto up = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, stage);
  gguf_zero(gate);
  gguf_zero(up);
  gguf_staged_pair_steps<F, Rows>(input, gw0, gw1, gmeta, uw0, uw1, umeta, p.input_size, origin, stage, tl, lane, sb,
                                  se, staged_first_step(group.x / 2, sb, se, p.spread), gate, up, AffineDictionaryMeta{gate_pairs}, AffineDictionaryMeta{up_pairs});
  const auto value = [](float g, float u) { return bfloat(float(bfloat(u)) * splash_silu(float(bfloat(g)))); };
  const auto out = [&](uint row, uint column) { return ulong(row) * p.out_stride + column0 + column; };
  if (p.splits == 1) {
#pragma unroll
    for (ushort i = 0; i < gate.get_capacity(); ++i) {
      if (!gate.is_valid_element(i)) continue;
      const auto index = gate.get_multidimensional_index(i);
      output[out(index[1], index[0])] = value(gate[i], up[i]);
    }
    return;
  }
  const auto at = [&](uint s, uint row, uint column) {
    return (ulong(s) * Rows + row) * (2 * p.out_stride) + column0 + column;
  };
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index = gate.get_multidimensional_index(i);
    partials[at(group.y, index[1], index[0])] = gate[i];
    partials[at(group.y, index[1], index[0]) + p.out_stride] = up[i];
  }
  device atomic_uint *counter = counters + p.out_offset / GGUF_STAGED_COLUMNS + group.x;
  if (!split_arrive_last(counter, p.splits, lane, &arrival)) return;
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index = gate.get_multidimensional_index(i);
    const uint row = index[1], column = index[0];
    const float g = split_sum(float(gate[i]), group.y, p.splits, [&](uint s) { return partials[at(s, row, column)]; });
    const float u = split_sum(float(up[i]), group.y, p.splits,
                              [&](uint s) { return partials[at(s, row, column) + p.out_stride]; });
    output[out(row, column)] = value(g, u);
  }
  split_release(counter, lane);
}

template <class Out>
using AffineDictionaryKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device Out *,
    device coherent(device) float *, device atomic_uint *, device bfloat *, device const uint *,
    constant GgufDecodeParams &, uint2, uint, uint);
#define DICT_DECODE(R, ep, Ep, Out) \
  template [[host_name("experiment_affine4_dictionary_m" #R "_" #ep)]] kernel AffineDictionaryKernel<Out> affine_dictionary_decode<FmtAF4G64, R, Ep, Out>;
#define DICT_ROWS(R) DICT_DECODE(R, a, EpNone, bfloat) DICT_DECODE(R, r, EpResidual, bfloat) DICT_DECODE(R, g, EpUpWithGate, bfloat) DICT_DECODE(R, a_f32, EpNone, float)
DICT_ROWS(8)
DICT_ROWS(16)
DICT_ROWS(32)
#undef DICT_ROWS
#undef DICT_DECODE
using AffineDictionaryPairKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device uchar *,
    device uchar *, device uchar *, device bfloat *, device coherent(device) float *, device atomic_uint *,
    device const uint *, device const uint *, constant GgufDecodeParams &, uint2, uint);
#define DICT_PAIR(R) \
  template [[host_name("experiment_affine4_dictionary_m" #R "_gate_up")]] kernel AffineDictionaryPairKernel affine_dictionary_gate_up<FmtAF4G64, R>;
DICT_PAIR(8)
DICT_PAIR(16)
DICT_PAIR(32)
#undef DICT_PAIR
kernel void experiment_affine_dictionary_roundtrip(device const ushort *indices [[buffer(0)]],
    device const uint *pairs [[buffer(1)]], device uint *output [[buffer(2)]],
    constant uint &units [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < units) output[i] = pairs[indices[i]];
}

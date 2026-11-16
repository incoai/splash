#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/state_codec.h"

// The cached form of a GDN state's recurrent layers (ops::StateCodec,
// common/state_codec.h's arithmetic): one thread per
// SPLASH_STATE_CODEC_THREAD_VALUES consecutive values of a value row, two
// threads per block of SPLASH_STATE_CODEC_BLOCK, which any even threadgroup
// width keeps neighbours in one simdgroup.

constant constexpr uint kValues = SPLASH_STATE_CODEC_THREAD_VALUES;
static_assert(SPLASH_STATE_CODEC_BLOCK == 2 * SPLASH_STATE_CODEC_THREAD_VALUES,
              "a block is the values of two neighbouring threads");

template <class State>
inline void state_quantize_thread(device const State *state, device char *codes, constant StateCodecParams &p,
                                  uint index, uint layer) {
  // Both threads of a block are in or out together: threads_per_layer is
  // even and the pairs start at even threads.
  if (index >= p.threads_per_layer)
    return;
  device const State *values = reinterpret_cast<device const State *>(
      reinterpret_cast<device const char *>(state) + ulong(layer) * p.state_layer_bytes) + ulong(index) * kValues;
  float x[kValues];
  float amax = 0.0f;
  for (uint i = 0; i < kValues; i += 4) {
    const float4 four = float4(*reinterpret_cast<device const vec<State, 4> *>(values + i));
    for (uint j = 0; j < 4; ++j) {
      x[i + j] = four[j];
      amax = max(amax, abs(four[j]));
    }
  }
  amax = max(amax, simd_shuffle_xor(amax, 1));
  const bfloat scale = state_codec_scale(amax);
  device char *layerCodes = codes + ulong(layer) * p.codes_layer_bytes;
  device char4 *out = reinterpret_cast<device char4 *>(layerCodes + ulong(index) * kValues);
  for (uint i = 0; i < kValues; i += 4)
    out[i / 4] = char4(state_code(x[i], float(scale)), state_code(x[i + 1], float(scale)),
                       state_code(x[i + 2], float(scale)), state_code(x[i + 3], float(scale)));
  if ((index & 1) == 0)
    reinterpret_cast<device bfloat *>(layerCodes + p.scales_offset)[index / 2] = scale;
}

template <class State>
inline void state_dequantize_thread(device const char *codes, device State *state, constant StateCodecParams &p,
                                    uint index, uint layer) {
  if (index >= p.threads_per_layer)
    return;
  device const char *layerCodes = codes + ulong(layer) * p.codes_layer_bytes;
  const float scale = float(reinterpret_cast<device const bfloat *>(layerCodes + p.scales_offset)[index / 2]);
  device const char4 *in = reinterpret_cast<device const char4 *>(layerCodes + ulong(index) * kValues);
  device State *values = reinterpret_cast<device State *>(reinterpret_cast<device char *>(state) +
                                                          ulong(layer) * p.state_layer_bytes) +
                         ulong(index) * kValues;
  for (uint i = 0; i < kValues; i += 4)
    *reinterpret_cast<device vec<State, 4> *>(values + i) = vec<State, 4>(float4(in[i / 4]) * scale);
}

#define STATE_CODEC_KERNELS(Suffix, State)                                                           \
  kernel void state_quantize##Suffix(device const State *state [[buffer(0)]],                        \
                                     device char *codes [[buffer(1)]],                               \
                                     constant StateCodecParams &p [[buffer(2)]],                     \
                                     uint2 position [[thread_position_in_grid]]) {                   \
    state_quantize_thread<State>(state, codes, p, position.x, position.y);                           \
  }                                                                                                  \
  kernel void state_dequantize##Suffix(device const char *codes [[buffer(0)]],                       \
                                       device State *state [[buffer(1)]],                            \
                                       constant StateCodecParams &p [[buffer(2)]],                   \
                                       uint2 position [[thread_position_in_grid]]) {                 \
    state_dequantize_thread<State>(codes, state, p, position.x, position.y);                         \
  }

// The running state in fp32 and, as Name_bf16_state, in bf16 (GdnStateFormat).
STATE_CODEC_KERNELS(, float)
STATE_CODEC_KERNELS(_bf16_state, bfloat)
#undef STATE_CODEC_KERNELS

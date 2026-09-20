#include "metal/abi/KernelABI.h"

template <uint Hidden>
inline void q4_embedding_impl(device const uint *tokens,
                              device const uchar *weights,
                              device const bfloat *scales,
                              device const bfloat *biases,
                              device bfloat *output,
                              constant Q4EmbeddingParams &params, uint index,
                              uint grid_size) {
  constexpr uint QuantGroups = Hidden / 64;
  uint elements = params.rows * Hidden;
  for (uint element = index; element < elements; element += grid_size) {
    uint row = element / Hidden;
    uint dim = element % Hidden;
    uint raw_token = tokens[row];
    // Runtime validates every token. Keep the bounds guard local to the
    // storage table so a malformed direct operator call cannot read past it.
    uint token = raw_token < params.vocabulary_size ? raw_token : 0;
    uchar packed = weights[ulong(token) * (Hidden / 2) + dim / 2];
    float quantized = float((packed >> ((dim & 1) * 4)) & 15);
    ulong parameter = ulong(token) * QuantGroups + dim / 64;
    output[element] =
        bfloat(quantized * float(scales[parameter]) + float(biases[parameter]));
  }
}

#define Q4_EMBEDDING_ENTRY(Name, Hidden)                                      \
  kernel void Name(                                                          \
      device const uint *tokens [[buffer(0)]],                               \
      device const uchar *weights [[buffer(1)]],                             \
      device const bfloat *scales [[buffer(2)]],                             \
      device const bfloat *biases [[buffer(3)]],                             \
      device bfloat *output [[buffer(4)]],                                   \
      constant Q4EmbeddingParams &params [[buffer(5)]],                      \
      uint index [[thread_position_in_grid]],                                \
      uint grid_size [[threads_per_grid]]) {                                 \
    q4_embedding_impl<Hidden>(tokens, weights, scales, biases, output,       \
                              params, index, grid_size);                     \
  }

Q4_EMBEDDING_ENTRY(embedding_q4_h5120, 5120)
Q4_EMBEDDING_ENTRY(embedding_q4_h2048, 2048)
#undef Q4_EMBEDDING_ENTRY

template <uint Hidden>
inline void q8_embedding_impl(device const uint *tokens,
                              device const uchar *weights,
                              device const bfloat *scales,
                              device const bfloat *biases,
                              device bfloat *output,
                              constant Q4EmbeddingParams &params, uint index,
                              uint grid_size) {
  constexpr uint QuantGroups = Hidden / 64;
  uint elements = params.rows * Hidden;
  for (uint element = index; element < elements; element += grid_size) {
    uint row = element / Hidden;
    uint dim = element % Hidden;
    uint raw_token = tokens[row];
    uint token = raw_token < params.vocabulary_size ? raw_token : 0;
    uchar quantized = weights[ulong(token) * Hidden + dim];
    ulong parameter = ulong(token) * QuantGroups + dim / 64;
    output[element] =
        bfloat(float(quantized) * float(scales[parameter]) + float(biases[parameter]));
  }
}

#define Q8_EMBEDDING_ENTRY(Name, Hidden)                                      \
  kernel void Name(                                                          \
      device const uint *tokens [[buffer(0)]],                               \
      device const uchar *weights [[buffer(1)]],                             \
      device const bfloat *scales [[buffer(2)]],                             \
      device const bfloat *biases [[buffer(3)]],                             \
      device bfloat *output [[buffer(4)]],                                   \
      constant Q4EmbeddingParams &params [[buffer(5)]],                      \
      uint index [[thread_position_in_grid]],                                \
      uint grid_size [[threads_per_grid]]) {                                 \
    q8_embedding_impl<Hidden>(tokens, weights, scales, biases, output,       \
                              params, index, grid_size);                     \
  }

Q8_EMBEDDING_ENTRY(embedding_q8_h5120, 5120)
#undef Q8_EMBEDDING_ENTRY

#include "metal/abi/KernelABI.h"

// The value residual's mix (model/Ripple.hpp): the block's value input is the
// normed input lerped to the scaled embeddings by the sigmoid of the block's
// learned lambda, computed once per element in fp32 and rounded once to
// bf16, as the reference's fp32 mix feeding a bf16 projection is.
//
// v_in[e] = sigmoid(lambda) * a[e] + (1 - sigmoid(lambda)) * emb[e]
//
// a and emb are `elements` bf16 values (the block's normed input rows and the
// scaled token embedding rows, which every block mixes back in); lambda is
// the block's one F32 value.
kernel void ripple_value_mix(device const bfloat *a [[buffer(0)]],
                             device const bfloat *emb [[buffer(1)]],
                             device const float *lambda [[buffer(2)]],
                             device bfloat *v_in [[buffer(3)]],
                             constant uint &elements [[buffer(4)]],
                             uint index [[thread_position_in_grid]]) {
  if (index >= elements) return;
  const float s = 1.0f / (1.0f + exp(-lambda[0]));
  v_in[index] = bfloat(s * float(a[index]) + (1.0f - s) * float(emb[index]));
}

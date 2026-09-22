#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashSharedExpertFused.h"

#pragma METAL fp math_mode(safe)
#include "metal/kernels/common/flash_affine_mpp_common.h"
using namespace metal;
using namespace mpp::tensor_ops;

// Canonical compiled SwiGLU, distinct from the precise unary shared gate.
#pragma METAL fp math_mode(fast)
inline bfloat flash_shared_fused_compiled_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

template <ushort M, ushort N, bool Taps>
inline void flash_shared_expert_fused_tile(device bfloat *input,
    device bfloat *gateWeights, device bfloat *upWeights, device bfloat *output,
    device uint *diagnostics, device bfloat *gateTap, device bfloat *upTap,
    constant FlashSharedExpertFusedParams &p, uint3 group, uint3 threads, uint tid) {
  if (p.rows < 256 || p.rows > 8192 || p.rows % M || p.input_size != 2560 ||
      p.output_size != 640 || p.output_begin || p.output_count != 640 ||
      p.tile_rows != M || p.tile_outputs != N || p.reserved || group.z ||
      group.x >= 640 / N || group.y >= p.rows / M || threads.x != 128 ||
      threads.y != 1 || threads.z != 1) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u); return;
  }
  const uint row = group.y * M, column = group.x * N;
  const int k = int(p.input_size);
  auto a = tensor(input + ulong(row) * 2560,
      dextents<int, 2>{k, M}, array<int, 2>{1, k});
  auto gate = tensor(gateWeights + ulong(column) * 2560,
      dextents<int, 2>{k, N}, array<int, 2>{1, k});
  auto up = tensor(upWeights + ulong(column) * 2560,
      dextents<int, 2>{k, N}, array<int, 2>{1, k});
  constexpr auto descriptor = matmul2d_descriptor(M, N,
      static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto gateDot = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(gate), float>();
  auto upDot = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(up), float>();
  operation.run(a, gate, gateDot);
  operation.run(a, up, upDot);
#pragma unroll
  for (ushort i = 0; i < gateDot.get_capacity(); ++i) {
    if (!gateDot.is_valid_element(i)) continue;
    const auto index = gateDot.get_multidimensional_index(i);
    const ulong destination = ulong(row + index[1]) * 640 + column + index[0];
    const bfloat roundedGate = bfloat(gateDot[i]), roundedUp = bfloat(upDot[i]);
    if constexpr (Taps) { gateTap[destination] = roundedGate; upTap[destination] = roundedUp; }
    const bfloat sigmoid = flash_shared_fused_compiled_sigmoid(roundedGate);
    const bfloat silu = roundedGate * sigmoid;
    const bfloat activated = silu * roundedUp;
    if (!flash_mpp_finite(gateDot[i]) || !flash_mpp_finite(upDot[i]) ||
        !flash_mpp_finite(roundedGate) || !flash_mpp_finite(roundedUp) ||
        !flash_mpp_finite(activated)) {
      flash_mpp_error(diagnostics, 4u);
      output[destination] = bfloat(as_type<float>(0x7fc00000u));
    } else output[destination] = activated;
  }
}

#define FLASH_SHARED_FUSED_ENTRY(Name, M, N)                                \
kernel void Name(device bfloat *input [[buffer(0)]],                        \
    device bfloat *gate [[buffer(1)]], device bfloat *up [[buffer(2)]],      \
    device bfloat *output [[buffer(3)]], device uint *diag [[buffer(4)]],   \
    constant FlashSharedExpertFusedParams &p [[buffer(5)]],                       \
    uint3 group [[threadgroup_position_in_grid]],                         \
    uint3 threads [[threads_per_threadgroup]],                            \
    uint tid [[thread_index_in_threadgroup]]) {                           \
  flash_shared_expert_fused_tile<M, N, false>(input, gate, up, output, diag, \
      nullptr, nullptr, p, group, threads, tid);                           \
}

#define FLASH_SHARED_FUSED_TAPS(Name, M, N)                                 \
kernel void Name(device bfloat *input [[buffer(0)]],                        \
    device bfloat *gate [[buffer(1)]], device bfloat *up [[buffer(2)]],      \
    device bfloat *output [[buffer(3)]], device uint *diag [[buffer(4)]],   \
    device bfloat *gateTap [[buffer(5)]], device bfloat *upTap [[buffer(6)]], \
    constant FlashSharedExpertFusedParams &p [[buffer(7)]],                       \
    uint3 group [[threadgroup_position_in_grid]],                         \
    uint3 threads [[threads_per_threadgroup]],                            \
    uint tid [[thread_index_in_threadgroup]]) {                           \
  flash_shared_expert_fused_tile<M, N, true>(input, gate, up, output, diag, \
      gateTap, upTap, p, group, threads, tid);                             \
}




FLASH_SHARED_FUSED_ENTRY(flash_shared_expert_fused_prefill_m32_n128, 32, 128)




#undef FLASH_SHARED_FUSED_ENTRY
#undef FLASH_SHARED_FUSED_TAPS
#endif

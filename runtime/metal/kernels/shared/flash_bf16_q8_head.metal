// Original Q8 reconstructed and rounded to qualified cached BF16 coefficients.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashInt8Head.h"
#include "metal/kernels/common/flash_affine_mpp_common.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
inline bool bf16_q8_head_geometry(constant FlashInt8HeadParams &p) {
  return p.rows>=2&&p.rows<=4&&p.padded_rows==8&&p.input_size==2560&&
      p.output_size==248320&&p.bits==8&&p.group_size==64&&p.tile_rows==8&&
      p.tile_outputs==32&&p.weight_row_stride_bytes>=2560&&
      p.parameter_row_stride_bytes>=80&&!(p.parameter_row_stride_bytes%2);
}
// Register-owned cooperative BF16 right operand: no staging barriers.
template <ushort N, ushort BK, ushort SG>
inline void bf16_q8_head_register_tile(
    device bfloat *padded, device const uchar *codes,
    device const uchar *scaleBytes, device const uchar *biasBytes,
    device bfloat *output, device uint *diagnostics,
    constant FlashInt8HeadParams &p, uint3 group, uint3 threads, uint tid) {
  if (!bf16_q8_head_geometry(p) || p.tile_outputs != N || group.y || group.z ||
      group.x >= p.output_size / N || threads.x != SG * 32 ||
      threads.y != 1 || threads.z != 1) {
    if (!tid) flash_mpp_error(diagnostics, 2u); return;
  }
  constexpr auto descriptor = matmul2d_descriptor(8, N, BK, false, true,
      false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto a = tensor(padded, dextents<int, 2>{2560, 8}, array<int, 2>{1, 2560});
  auto at = a.template slice<BK, 8>(0, 0);
  auto bt = operation.template get_right_input_cooperative_tensor<bfloat,bfloat,float>();
  auto total = operation.template get_destination_cooperative_tensor<decltype(at),decltype(at),float>();
#pragma unroll
  for (ushort i=0;i<total.get_capacity();++i)
    if(total.is_valid_element(i))total[i]=0.0f;
  for(uint origin=0;origin<2560;origin+=BK) {
#pragma unroll
    for(ushort i=0;i<bt.get_capacity();++i) {
      if(!bt.is_valid_element(i))continue;
      const auto index=bt.get_multidimensional_index(i);
      const uint n=group.x*N+uint(index[1]),k=origin+uint(index[0]);
      const uint code=codes[ulong(n)*p.weight_row_stride_bytes+k];
      const ulong offset=ulong(n)*p.parameter_row_stride_bytes+(k/64)*2;
      const bfloat scale=*reinterpret_cast<device const bfloat *>(scaleBytes+offset);
      const bfloat bias=*reinterpret_cast<device const bfloat *>(biasBytes+offset);
      bt[i]=bfloat(flash_mpp_dequantize_f32(code,scale,bias));
    }
    at=a.template slice<BK,8>(origin,0);operation.run(at,bt,total);
  }
#pragma unroll
  for(ushort i=0;i<total.get_capacity();++i) {
    if(!total.is_valid_element(i))continue;
    const auto index=total.get_multidimensional_index(i);
    if(uint(index[1])>=p.rows)continue;
    const bfloat result=bfloat(total[i]);
    if(!flash_mpp_finite(total[i])||!flash_mpp_finite(result))flash_mpp_error(diagnostics,4u);
    output[ulong(index[1])*p.output_size+group.x*N+uint(index[0])]=result;
  }
}
kernel void flash_bf16_q8_head_register_m8_n32_k64_s1(
    device bfloat *input [[buffer(0)]],device const uchar *codes [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],device const uchar *biases [[buffer(3)]],
    device const float *unused [[buffer(4)]],device bfloat *output [[buffer(5)]],
    device uint *diagnostics [[buffer(6)]],constant FlashInt8HeadParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  (void)unused;bf16_q8_head_register_tile<32,64,1>(input,codes,scales,biases,
      output,diagnostics,p,group,threads,tid);
}
#endif

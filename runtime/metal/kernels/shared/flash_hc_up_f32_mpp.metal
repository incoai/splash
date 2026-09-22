#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashFloatDenseCache.h"
using namespace metal;using namespace mpp::tensor_ops;
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
inline void hf_error(device atomic_uint *d,uint f){atomic_fetch_or_explicit(d,f,memory_order_relaxed);}
inline bfloat hf_sigmoid(bfloat x){const bfloat e=bfloat(precise::exp(abs(float(x))));
  const bfloat denominator=bfloat(1.0f)+e;float reciprocal=1.0f/float(denominator);
  const uint bits=as_type<uint>(reciprocal);if(!(bits&0x7f800000u))reciprocal=as_type<float>(bits&0x80000000u);
  const bfloat tail=bfloat(reciprocal);return x<bfloat(0)?tail:bfloat(1.0f)-tail;}
template<ushort M,ushort N,ushort S,bool Debug>
inline void hf_up(device bfloat *padded,device float *weights,device const bfloat *normalized,
    device bfloat *mixed,device bfloat *rawDebug,device atomic_uint *diagnostics,
    constant FlashFloatDenseSmallRowsParams &p,uint3 group,uint3 threads,uint tid){
  if(!p.rows||p.rows>16||p.input_size!=320||p.output_size!=2560||p.output_begin||p.output_count!=2560||
      p.tile_rows!=M||p.tile_outputs!=N||p.padded_rows!=(p.rows+M-1)/M*M||p.padded_rows>16||
      group.x>=2560/N||group.y>=p.padded_rows/M||group.z||threads.x!=S*32||threads.y!=1||threads.z!=1){
    if(!tid)hf_error(diagnostics,2);return;}
  const uint rowBase=group.y*M,colBase=group.x*N;
  auto a=tensor(padded+ulong(rowBase)*320,dextents<int,2>{320,M},array<int,2>{1,320});
  auto firstB=tensor(weights+ulong(colBase)*320,dextents<int,2>{320,N},array<int,2>{1,320});
  constexpr auto descriptor=matmul2d_descriptor(M,N,static_cast<int>(dynamic_extent),false,true,false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<S>> operation;
  auto total=operation.template get_destination_cooperative_tensor<decltype(a),decltype(firstB),float>();
#pragma unroll
  for(ushort i=0;i<total.get_capacity();++i)if(total.is_valid_element(i))total[i]=0;
  for(uint stream=0;stream<4;++stream){
    auto b=tensor(weights+ulong(stream*2560+colBase)*320,dextents<int,2>{320,N},array<int,2>{1,320});
    auto dot=operation.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    operation.run(a,b,dot);
#pragma unroll
    for(ushort i=0;i<total.get_capacity();++i){if(!total.is_valid_element(i))continue;
      const auto index=total.get_multidimensional_index(i);const uint row=rowBase+index[1],col=colBase+index[0];
      if(row>=p.rows)continue;const bfloat raw=bfloat(dot[i]);
      if(!isfinite(dot[i])||!isfinite(float(raw)))hf_error(diagnostics,4);
      if(Debug)rawDebug[(ulong(row)*4+stream)*2560+col]=raw;
      const bfloat product=bfloat(float(hf_sigmoid(raw))*float(normalized[(ulong(row)*4+stream)*2560+col]));
      total[i]=float(bfloat(float(product)+total[i]));
    }
  }
#pragma unroll
  for(ushort i=0;i<total.get_capacity();++i){if(!total.is_valid_element(i))continue;
    const auto index=total.get_multidimensional_index(i);const uint row=rowBase+index[1];if(row>=p.rows)continue;
    const bfloat value=bfloat(total[i]/4.0f);mixed[ulong(row)*2560+colBase+index[0]]=value;
    if(!isfinite(float(value)))hf_error(diagnostics,4);
  }
}
#define HF_KERNEL(NAME,M,N,S,D) \
[[max_total_threads_per_threadgroup(S*32)]] kernel void NAME( \
    device bfloat *padded [[buffer(0)]],device float *weights [[buffer(1)]],device const bfloat *normalized [[buffer(2)]], \
    device bfloat *mixed [[buffer(3)]],device bfloat *raw [[buffer(4)]],device atomic_uint *diag [[buffer(5)]], \
    constant FlashFloatDenseSmallRowsParams &p [[buffer(6)]],uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]){ \
  hf_up<M,N,S,D>(padded,weights,normalized,mixed,raw,diag,p,group,threads,tid); \
}
#define HF_ENTRY(M,N,S) \
HF_KERNEL(flash_hc_up_f32_mpp_m##M##_n##N##_s##S,M,N,S,false) \
HF_KERNEL(flash_hc_up_f32_mpp_m##M##_n##N##_s##S##_debug,M,N,S,true)
HF_ENTRY(8,32,4)

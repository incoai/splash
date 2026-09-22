#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashDenseCache.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
#include "metal/kernels/common/flash_dense_traversal.h"
inline void dense64_error(device atomic_uint *diagnostics,uint flag) {
  atomic_fetch_or_explicit(diagnostics,flag,memory_order_relaxed);
}
template<ushort N, bool Traversal = false>
inline void dense64(device bfloat *input,device bfloat *weights,device bfloat *output,
    device atomic_uint *diagnostics,constant FlashDenseCacheParams &p,
    uint3 group,uint3 threads,uint tid) {
  if(!p.rows||p.rows>8192||p.rows%64||!p.input_size||p.input_size>32768||p.input_size%32||
      !p.output_size||!p.output_count||p.output_count%N||p.output_begin>p.output_size||
      p.output_count>p.output_size-p.output_begin||p.tile_rows!=64||p.tile_outputs!=N||
      (Traversal?p.reserved>4:p.reserved!=0)||group.z||
      (Traversal?!flash_dense_traversal_group_valid(group.xy,p.rows/64,
                            p.output_count/N,p.reserved)
                :group.x>=p.output_count/N||group.y>=p.rows/64)||
      threads.x!=256||threads.y!=1||threads.z!=1) {
    if(!tid)dense64_error(diagnostics,2u);return;
  }
  const uint2 tile=Traversal?flash_dense_traversal_tile(group.xy,p.reserved):group.xy;
  if(Traversal&&(tile.x>=p.output_count/N||tile.y>=p.rows/64))return;
  const uint row=tile.y*64,column=p.output_begin+tile.x*N;
  const int k=int(p.input_size);
  auto a=tensor(input+ulong(row)*p.input_size,dextents<int,2>{k,64},array<int,2>{1,k});
  auto b=tensor(weights+ulong(column)*p.input_size,dextents<int,2>{k,N},array<int,2>{1,k});
  constexpr auto descriptor=matmul2d_descriptor(64,N,static_cast<int>(dynamic_extent),
      false,true,false,matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<8>> operation;
  auto dot=operation.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
  operation.run(a,b,dot);
#pragma unroll
  for(ushort i=0;i<dot.get_capacity();++i) {
    if(!dot.is_valid_element(i))continue;
    const auto index=dot.get_multidimensional_index(i);const bfloat value=bfloat(dot[i]);
    if(!isfinite(dot[i])||!isfinite(float(value)))dense64_error(diagnostics,4u);
    output[ulong(row+index[1])*p.output_size+column+index[0]]=value;
  }
}
#define DENSE64_ENTRY(NAME,N,Traversal) \
[[max_total_threads_per_threadgroup(256)]] kernel void NAME( \
    device bfloat *input [[buffer(0)]],device bfloat *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]],device atomic_uint *diagnostics [[buffer(3)]], \
    constant FlashDenseCacheParams &p [[buffer(4)]],uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) { \
  dense64<N,Traversal>(input,weights,output,diagnostics,p,group,threads,tid); \
}
DENSE64_ENTRY(flash_dense_cache_m64_n64,64,false)
DENSE64_ENTRY(flash_dense_cache_m64_n128,128,false)
DENSE64_ENTRY(flash_dense_cache_m64_n64_traversal,64,true)
DENSE64_ENTRY(flash_dense_cache_m64_n128_traversal,128,true)

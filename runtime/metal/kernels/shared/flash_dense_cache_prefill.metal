// Opt-in whole-K BF16 MPP tiles for source-qualified2048-row cached projections.
// Geometry/traversal changes do not request physical die or memory placement.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashDenseCache.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
#include "metal/kernels/common/flash_dense_traversal.h"

inline void prefill_dense_error(device atomic_uint *diagnostics,uint flag) {
  atomic_fetch_or_explicit(diagnostics,flag,memory_order_relaxed);
}
template <ushort Groups>
inline void prefill_dense_tile(device bfloat *input,device bfloat *weights,
    device bfloat *output,device atomic_uint *diagnostics,
    constant FlashDenseCacheParams &p,uint3 group,uint3 threads,uint tid) {
  if (p.rows != 2048 || !p.input_size || p.input_size > 32768 || p.input_size % 32 ||
      !p.output_size || !p.output_count || p.output_count % 64 ||
      p.output_begin > p.output_size || p.output_count > p.output_size-p.output_begin ||
      p.tile_rows != 128 || p.tile_outputs != 64 || p.reserved > 4 || group.z ||
      !flash_dense_traversal_group_valid(group.xy,p.rows/128,p.output_count/64,p.reserved) ||
      threads.x != Groups*32 || threads.y != 1 || threads.z != 1) {
    if (!tid) prefill_dense_error(diagnostics,2u);
    return;
  }
  const uint2 tile = flash_dense_traversal_tile(group.xy,p.reserved);
  if (tile.x >= p.output_count/64 || tile.y >= p.rows/128) return;
  const uint row=tile.y*128,column=p.output_begin+tile.x*64;
  const int k=int(p.input_size);
  auto a=tensor(input+ulong(row)*p.input_size,dextents<int,2>{k,128},array<int,2>{1,k});
  auto b=tensor(weights+ulong(column)*p.input_size,dextents<int,2>{k,64},array<int,2>{1,k});
  constexpr auto descriptor=matmul2d_descriptor(128,64,static_cast<int>(dynamic_extent),
      false,true,false,matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor,execution_simdgroups<Groups>> operation;
  auto dot=operation.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
  operation.run(a,b,dot);
#pragma unroll
  for (ushort i=0;i<dot.get_capacity();++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index=dot.get_multidimensional_index(i);
    const bfloat value=bfloat(dot[i]);
    if (!isfinite(dot[i]) || !isfinite(float(value))) prefill_dense_error(diagnostics,4u);
    output[ulong(row+index[1])*p.output_size+column+index[0]]=value;
  }
}
#define PREFILL_DENSE_ENTRY(G) \
[[max_total_threads_per_threadgroup(G*32)]] kernel void flash_dense_cache_prefill_m128_n64_sg##G( \
    device bfloat *input [[buffer(0)]],device bfloat *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]],device atomic_uint *diagnostics [[buffer(3)]], \
    constant FlashDenseCacheParams &p [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  prefill_dense_tile<G>(input,weights,output,diagnostics,p,group,threads,tid); \
}
PREFILL_DENSE_ENTRY(4)
PREFILL_DENSE_ENTRY(8)
#undef PREFILL_DENSE_ENTRY

// R217 single isolated PQ2 candidate. Stock source/entries remain immutable.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/split_reduce.h"
template <ushort Rows, class Acc, class Store>
inline void gguf_store_sums(thread Acc &acc, uint splits, uint split, device coherent(device) float *partials,
                            device atomic_uint *counter, uint stride, uint column0, uint thread_index,
                            threadgroup uint *arrival, Store store) {
  if (splits == 1) { gguf_elements(acc, store); return; }
  const auto at = [&](uint s, uint row, uint column) { return (ulong(s) * Rows + row) * stride + column0 + column; };
  gguf_elements(acc, [&](uint row, uint column, float v) { partials[at(split, row, column)] = v; });
  if (!split_arrive_last(counter, splits, thread_index, arrival)) return;
  gguf_elements(acc, [&](uint row, uint column, float v) {
    store(row, column, split_sum(v, split, splits, [&](uint s) { return partials[at(s, row, column)]; }));
  });
  split_release(counter, thread_index);
}

// No weight stage, pair table, or stage pointer in this entry or its helpers.
constant constexpr auto lane_desc = matmul2d_descriptor(8,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
using LaneOp = matmul2d<lane_desc,execution_simdgroups<1>>;
inline uint2 lane_coord(uint l,uint i) {
 return uint2((i&3)+4*(l&1)+8*((l>>3)&1)+16*(i>>4),((l>>1)&3)+4*(l>>4)+8*((i>>2)&3));
}
template<class B> inline bool lane_mapping(thread B &b,uint lane) {
 if(b.get_capacity()!=32)return false;
 bool ok=true;
 for(ushort i=0;i<32;++i) {
  if(!b.is_valid_element(i)){ok=false;continue;}
  auto p=b.get_multidimensional_index(i);auto want=lane_coord(lane,i);
  ok=ok && uint(p[0])==want.x && uint(p[1])==want.y;
 }
 return simd_all(ok);
}
inline ushort4 stock_words(device uchar *w,uint origin,uint lane,uint step) {
 const uint c=((lane>>1)&3)+4*(lane>>4),part=(lane&1)+2*((lane>>3)&1);
 ushort4 words;
 #pragma unroll
 for(uint t=0;t<4;++t){uint n=origin+c+8*t;ulong slot=(ulong(n/256)*160+step)*256+n%256;
  words[t]=*((device ushort*)(w+slot*8+2*part));}
 return words;
}
inline ushort4 stock_scales(device uchar *m,uint origin,uint lane,uint group) {
 const uint c=((lane>>1)&3)+4*(lane>>4);ushort4 scales;
 #pragma unroll
 for(uint t=0;t<4;++t){uint n=origin+c+8*t;ulong slot=(ulong(n/256)*40+group)*256+n%256;
  scales[t]=((device ushort*)m)[slot];}
 return scales;
}
template<class B> inline void lane_decode(thread B &b,ushort4 words,ushort4 scales) {
 #pragma unroll
 for(ushort i=0;i<32;i+=2) {
  const uint t=(i>>2)&3,shift=2*((i&3)+4*(i>>4));
  const uint q0=(words[t]>>shift)&3u,q1=(words[t]>>(shift+2))&3u;
  const float d=float(as_type<half>(scales[t]));
  const half2 value=staged_linear<FmtPQ20>(q0|(q1<<16),d,0.0f);
  b[i]=value.x;b[i+1]=value.y;
 }
}

template<GgufEpilogue Ep>
kernel void lane_pq2(device bfloat *input [[buffer(0)]],device uchar *w0 [[buffer(1)]],
 device uchar *meta [[buffer(3)]],device bfloat *output [[buffer(4)]],
 device coherent(device) float *partials [[buffer(5)]],device atomic_uint *counters [[buffer(6)]],
 device bfloat *aux [[buffer(7)]],constant GgufDecodeParams &p [[buffer(8)]],
 uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup uint arrival;
 if(p.input_size!=5120 || p.splits!=1 || p.out_stride!=17408 || p.out_offset!=0 || group.x>=272 || group.y!=0 || sg>=2)return;
 const uint origin=group.x*64+sg*32;
 LaneOp op;
 auto a=tensor(input,dextents<int,2>{5120,8},array<int,2>{1,5120});
 auto a0=a.slice<32,8>(0,0);
 auto b=op.get_right_input_cooperative_tensor<bfloat,half,float>();
 auto acc=op.get_destination_cooperative_tensor<decltype(a0),decltype(b),float>();
 gguf_zero(acc); // all capacity slots exactly as stock; no uninitialized accumulator read
 if(!lane_mapping(b,lane))return; // candidate-context mismatch leaves host poison: fail before timing
 ushort4 next=stock_words(w0,origin,lane,0),cached;
 for(uint step=0;step<160;++step) {
  if((step&3)==0)cached=stock_scales(meta,origin,lane,step/4);
  const ushort4 current=next;
  if(step+1<160)next=stock_words(w0,origin,lane,step+1);
  lane_decode(b,current,cached);
  auto aa=a.slice<32,8>(step*32,0);op.run(aa,b,acc);
 }
 gguf_store_sums<8>(acc,p.splits,group.y,partials,counters+group.x,p.out_stride,origin,sg*32+lane,&arrival,[&](uint row,uint col,float v){
  const ulong o=ulong(row)*p.out_stride+origin+col;
  if constexpr(Ep==EpUpWithGate)v=float(bfloat(v))*gguf_silu(float(aux[o]));
  output[o]=bfloat(v);
 });
}
using LaneKernel=void(device bfloat*,device uchar*,device uchar*,device bfloat*,device coherent(device) float*,device atomic_uint*,device bfloat*,constant GgufDecodeParams&,uint2,uint,uint);
template [[host_name("r233_pq2_m8_a")]] kernel LaneKernel lane_pq2<EpNone>;
template [[host_name("r233_pq2_m8_g")]] kernel LaneKernel lane_pq2<EpUpWithGate>;

// Stock stage -> actual cooperative HALF versus lane-owned candidate HALF.

struct TinyParams{uint begin,end,origin,candidate;};
kernel void r233_values(device uchar *w [[buffer(0)]],device uchar *m [[buffer(1)]],
 device ushort *out [[buffer(2)]],constant TinyParams &p [[buffer(3)]],
 device uint4 *coordinates [[buffer(4)]],device uint *capacity [[buffer(5)]],uint lane [[thread_index_in_simdgroup]]) {
 LaneOp op;auto b=op.get_right_input_cooperative_tensor<bfloat,half,float>();
 capacity[lane]=b.get_capacity();
 if(!lane_mapping(b,lane))return;
 threadgroup half stock_stage[1024];threadgroup half2 stock_table[1];
 ushort4 cached;
 for(uint s=p.begin;s<p.end;++s){
  if(p.candidate){
   if(s==p.begin || (s&3)==0)cached=stock_scales(m,p.origin,lane,s/4);
   ushort4 packed=stock_words(w,p.origin,lane,s);lane_decode(b,packed,cached);
  }else{
   uint n=p.origin+lane;ulong wi=((n/256*160+s)*256+n%256)*8,mi=((n/256*40+s/4)*256+n%256)*2;
   auto packed=FmtPQ20::load(w+wi,w);auto scale=FmtPQ20::loadMeta(m+mi);
   dequant32<FmtPQ20>(packed,scale,ushort(s%4),stock_table,stock_stage+lane*32);
   simdgroup_barrier(mem_flags::mem_threadgroup);
   tensor<threadgroup half,dextents<int,2>,tensor_inline> bt(stock_stage,dextents<int,2>{32,32},array<int,2>{1,32});
   b.load(bt);
  }
  for(ushort i=0;i<32;++i){auto xy=b.get_multidimensional_index(i);uint k=uint(xy[0]),n=uint(xy[1]);
   out[(s-p.begin)*1024+n*32+k]=as_type<ushort>(b[i]);
   coordinates[(s-p.begin)*1024+lane*32+i]=uint4(k,n,i,uint(b.is_valid_element(i)));
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
 }
}

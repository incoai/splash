#include "metal/kernels/common/wide_tree_attention_tile.h"

using namespace metal;
kernel void research_wide_prepare_pages(device const SplashKvPage *table [[buffer(0)]],
 device const ResearchTreeWideState &tree [[buffer(1)]],device uchar *scratch [[buffer(2)]],
 constant SplashVerifyAttentionParams *params [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 if(group.y>=tree.leaf_count)return;
 uint committed=params[0].committed_tokens,page=committed/32+group.z;
 if(page>=(committed+8+31)/32)return;
 uint leaf=tree.leaves[group.y];
 const SplashKvAddressing<4,256,int8_t> addressing(params[0].kv,group.x);
 device uchar *block=scratch+((group.y*2+group.z)*4+group.x)*16640;
 device int8_t *keys=reinterpret_cast<device int8_t *>(block),*values=reinterpret_cast<device int8_t *>(block+8192);
 device float *ks=reinterpret_cast<device float *>(block+16384),*vs=reinterpret_cast<device float *>(block+16512);
 for(uint r=0;r<32;++r){uint position=page*32+r,source=position;bool valid=position<committed;
  if(!valid && position-committed<=tree.depth[leaf]){uint node=leaf;while(tree.depth[node]>position-committed)node=uint(tree.parents[node]);source=committed+node;valid=true;}
  int8_t key=0,value=0;float keyscale=1,valuescale=1;
  if(valid){auto from=addressing.page(table[source/32]);uint at=source%32;key=from.keys[at*256+tid];value=from.values[tid*32+at];keyscale=from.key_scales[at];valuescale=from.value_scales[at];}
  keys[r*256+tid]=key;values[tid*32+r]=value;if(tid==0){ks[r]=keyscale;vs[r]=valuescale;}
 }
}
kernel void verify_attention_research_wide_split(
 device bfloat *queries [[buffer(0)]],device float *partials [[buffer(1)]],device float *statistics [[buffer(2)]],
 device const SplashKvPage *table [[buffer(3)]],device const ResearchTreeWideState &tree [[buffer(4)]],
 constant SplashVerifyAttentionParams *params [[buffer(5)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 using namespace wide_tree16;
 constexpr uint M=48;threadgroup float scores[M*32];threadgroup bfloat probs[M*32];threadgroup float maxima[M],sums[M],previous[M];threadgroup atomic_uint rescale;
 constant SplashVerifyAttentionParams &p=params[0];if(group.x>=4||group.z>=2||group.y>=p.split_count)return;
 ulong offset=(group.x*32+group.z*8)*6*256,slot=(group.x*2+group.z)*p.slot_splits+group.y;
 wide_paged_attention_tile<4,6,8,256,0,int8_t>(queries+offset,table,p.kv,group.x,p.committed_tokens,8,p.split_count,group.y,partials,statistics,slot,scores,probs,maxima,sums,previous,&rescale,tid,tree);
}
kernel void verify_attention_research_wide_tail(
 device bfloat *queries [[buffer(0)]],device float *partials [[buffer(1)]],device float *statistics [[buffer(2)]],
 device const SplashKvPage *table [[buffer(3)]],device const ResearchTreeWideState &tree [[buffer(4)]],device uchar *scratch [[buffer(5)]],
 constant SplashVerifyAttentionParams *params [[buffer(6)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 using namespace wide_tree16;
 constexpr uint M=48;threadgroup float scores[M*32];threadgroup bfloat probs[M*32];threadgroup float maxima[M],sums[M],previous[M];threadgroup atomic_uint rescale;
 constant SplashVerifyAttentionParams &p=params[0];if(group.x>=4||group.z>=2||group.y>=p.split_count)return;
 ulong offset=(group.x*32+group.z*8)*6*256,slot=(group.x*2+group.z)*p.slot_splits+group.y;
 wide_canonical_attention_tail<4,6,8,256,0,int8_t>(queries+offset,table,p.kv,group.x,p.committed_tokens,8,p.split_count,group.y,partials,statistics,slot,scores,probs,maxima,sums,previous,&rescale,tid,tree,scratch,group.z*8);
}
kernel void verify_attention_research_wide_reduce(device const float *partials [[buffer(0)]],device const float *statistics [[buffer(1)]],device bfloat *output [[buffer(2)]],constant SplashVerifyAttentionParams *params [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 using namespace wide_tree16;threadgroup float weights[256],group_values[8];constant SplashVerifyAttentionParams &p=params[0];uint part=group.y/48,row=group.y%48;
 uint pages=wide_attention_pages(p.committed_tokens+8),per=wide_attention_pages_per_split(pages,p.split_count),written=(pages+per-1)/per;
 wide_attention_combine_splits<6,8,256>(partials,statistics,output+(group.x*32+part*8)*6*256,8,written,(group.x*2+part)*p.slot_splits,row,tid,weights,group_values);
}

#include "metal/abi/ResearchTree.h"
#include "metal/abi/ResearchTreeWide.h"
#include "metal/abi/GDN.h"
using namespace metal;
kernel void research_wide_select(device const uint *ids [[buffer(0)]],device const float *unary [[buffer(1)]],device const float *partials [[buffer(2)]],device uint *proposals [[buffer(3)]],device ResearchTreeWideState &tree [[buffer(4)]],uint tid [[thread_position_in_grid]]){
 if(tid)return;device const float *edges=partials+7*8*16;uint trunk[7];for(uint p=0;p<7;++p)trunk[p]=proposals[p];
 float probability[16][16]={};float mass[16]={};uint depth[16]={},candidate[16]={},children[16]={},used[16]={};int parent[16];parent[0]=-1;mass[0]=1;uint leaves=1,real=8;
 for(uint next=1;next<=7;++next){uint from=next-1,p=depth[from];float best=-INFINITY;for(uint i=0;i<16;++i)best=max(best,unary[p*16+i]+edges[(p*16+candidate[from])*16+i]);float sum=0;for(uint i=0;i<16;++i){probability[from][i]=exp(unary[p*16+i]+edges[(p*16+candidate[from])*16+i]-best);sum+=probability[from][i];}for(uint i=0;i<16;++i)probability[from][i]=mass[from]*probability[from][i]/sum;
  uint chosen=0;for(uint i=0;i<16;++i)if(ids[p*16+i]==trunk[p]){chosen=i;break;}children[from]=1;used[from]=1u<<chosen;parent[next]=int(from);depth[next]=p+1;candidate[next]=chosen;mass[next]=probability[from][chosen];
 }
 for(uint next=8;next<16;++next){float best=-1;uint from=0,chosen=0;
  for(uint r=0;r<next;++r){if(depth[r]>=7||(children[r]&&leaves>=4))continue;for(uint i=0;i<16;++i)if(!(used[r]&(1u<<i))){float value=probability[r][i];bool tie=value==best&&(depth[r]<depth[from]||(depth[r]==depth[from]&&(r<from||(r==from&&i<chosen))));if(value>best||tie){best=value;from=r;chosen=i;}}}
  if(best<0)break;leaves+=children[from]>0;children[from]++;used[from]|=1u<<chosen;parent[next]=int(from);depth[next]=depth[from]+1;candidate[next]=chosen;mass[next]=best;real=next+1;
  if(depth[next]<7){uint p=depth[next];float maximum=-INFINITY;for(uint i=0;i<16;++i)maximum=max(maximum,unary[p*16+i]+edges[(p*16+chosen)*16+i]);float sum=0;for(uint i=0;i<16;++i){probability[next][i]=exp(unary[p*16+i]+edges[(p*16+chosen)*16+i]-maximum);sum+=probability[next][i];}for(uint i=0;i<16;++i)probability[next][i]=mass[next]*probability[next][i]/sum;}
 }
 uint order[16],inverse[16],stack[16],size=1;stack[0]=0;for(uint out=0;out<real;++out){uint old=stack[--size];order[out]=old;inverse[old]=out;for(int n=int(real)-1;n>=1;--n)if(parent[n]==int(old))stack[size++]=uint(n);}
 tree.node_count=real;tree.leaf_count=0;for(uint out=0;out<16;++out){tree.snapshot_at[out]=255;tree.restore_at[out]=255;tree.row_leaf[out]=255;if(out<8)tree.path[out]=0;if(out<real){uint old=order[out];tree.parents[out]=out?int(inverse[parent[old]]):-1;tree.depth[out]=depth[old];if(out)proposals[out-1]=ids[(depth[old]-1)*16+candidate[old]];if(!children[old])tree.leaves[tree.leaf_count++]=out;}}
 uint pad=0;for(int node=int(real)-1;node>=0;node=tree.parents[node])if(children[order[node]]>1){pad=uint(node);break;}
 for(uint out=real;out<16;++out){tree.parents[out]=int(pad);tree.depth[out]=tree.depth[pad]+1;proposals[out-1]=ids[(tree.depth[out]-1)*16];}
 uint count[16]={},first[16];for(uint i=0;i<16;++i)first[i]=255;for(uint i=1;i<16;++i){uint p=uint(tree.parents[i]);if(!count[p])first[p]=i;++count[p];}uint forks=0;for(uint i=0;i<16;++i)if(count[i]>1)tree.snapshot_at[i]=forks++;for(uint i=1;i<16;++i)if(i!=first[tree.parents[i]])tree.restore_at[i]=tree.snapshot_at[tree.parents[i]];
 for(uint l=0;l<tree.leaf_count;++l)for(int n=int(tree.leaves[l]);n>=0;n=tree.parents[n])if(tree.row_leaf[n]==255)tree.row_leaf[n]=uchar(l);
}
kernel void research_wide_rope(device float *cosine [[buffer(0)]],device float *sine [[buffer(1)]],device const ResearchTreeWideState &tree [[buffer(2)]],uint tid [[thread_position_in_grid]]){
 if(tid>=32)return;float c[8],s[8];for(uint r=0;r<8;++r){c[r]=cosine[r*32+tid];s[r]=sine[r*32+tid];}for(uint r=0;r<16;++r){cosine[r*32+tid]=c[tree.depth[r]];sine[r*32+tid]=s[tree.depth[r]];}
}
kernel void research_wide_input(device const uint *anchor [[buffer(0)]],device const uint *proposals [[buffer(1)]],device uint *input [[buffer(2)]],device uint *verifyRows [[buffer(3)]],constant uint &vocabulary [[buffer(4)]],uint tid [[thread_position_in_grid]]){if(tid<16)input[tid]=min(tid?proposals[tid-1]:anchor[0],vocabulary-1);if(tid<2)verifyRows[tid]=8;}
kernel void research_wide_accept(device const uint *proposals [[buffer(0)]],device uint *output [[buffer(1)]],device uint &retained [[buffer(2)]],device uint &accepted_count [[buffer(3)]],device ResearchTreeWideState &tree [[buffer(4)]],constant ResearchTreeAccept &params [[buffer(5)]],uint tid [[thread_position_in_grid]]){
 if(tid)return;uint target[16],chosen[8];for(uint i=0;i<16;++i)target[i]=output[i];for(uint i=0;i<8;++i)tree.path[i]=0;uint at=0,accepted=0;
 while(accepted<7){uint token=target[at];chosen[accepted]=token;int child=-1;for(uint i=1;i<tree.node_count;++i)if(tree.parents[i]==int(at)&&proposals[i-1]==token){child=int(i);break;}if(child<0)break;at=uint(child);++accepted;tree.path[accepted]=at;}
 if(accepted==7)chosen[7]=target[at];accepted_count=accepted;retained=min(accepted+1,params.remaining);for(uint i=0;i<retained;++i){output[i]=chosen[i];bool stop=false;for(uint j=0;j<4;++j)stop|=chosen[i]!=SPLASH_NO_TOKEN&&chosen[i]==params.stop[j];if(stop){retained=i+1;break;}}for(uint i=retained;i<8;++i)tree.path[i]=0;
}
kernel void research_wide_rows_gather(device bfloat *rows [[buffer(0)]],device const ResearchTreeWideState &tree [[buffer(1)]],constant uint &width [[buffer(2)]],uint col [[thread_position_in_grid]]){if(col>=width)return;bfloat saved[16];for(uint i=0;i<16;++i)saved[i]=rows[i*width+col];for(uint i=0;i<8;++i)rows[i*width+col]=saved[tree.path[i]];}
kernel void research_wide_kv_gather(device bfloat *keys [[buffer(0)]],device bfloat *values [[buffer(1)]],device const ResearchTreeWideState &tree [[buffer(2)]],uint head [[threadgroup_position_in_grid]],uint dim [[thread_index_in_threadgroup]]){bfloat k[16],v[16];for(uint i=0;i<16;++i){k[i]=keys[(head*32+i)*256+dim];v[i]=values[(head*256+dim)*32+i];}for(uint i=0;i<8;++i){keys[(head*32+i)*256+dim]=k[tree.path[i]];values[(head*256+dim)*32+i]=v[tree.path[i]];}}
kernel void research_wide_tape_gather(device const uchar *source [[buffer(0)]],device uchar *destination [[buffer(1)]],device const ResearchTreeWideState &tree [[buffer(2)]],constant ResearchTreeTapeGather &p [[buffer(3)]],uint task [[thread_position_in_grid]]){
 uint width=p.conv+p.kv+p.heads,layer=task/width,col=task%width;if(layer>=p.layers)return;source+=2*p.layer_bytes*layer;destination+=p.slot+p.layer_bytes*layer;
 if(col<p.conv+p.kv){uint stride=p.conv;device const bfloat *src;device bfloat *dst;if(col<p.conv){src=reinterpret_cast<device const bfloat*>(source);dst=reinterpret_cast<device bfloat*>(destination);}else{col-=p.conv;stride=p.kv;src=reinterpret_cast<device const bfloat*>(source+16ul*p.conv*2);dst=reinterpret_cast<device bfloat*>(destination+gdn_tape_kv_offset(p.conv));}for(uint row=0;row<8;++row)dst[row*stride+col]=src[tree.path[row]*stride+col];}
 else{col-=p.conv+p.kv;auto sd=reinterpret_cast<device const float*>(source+16ul*(p.conv+p.kv)*2);auto sb=reinterpret_cast<device const bfloat*>(source+16ul*(p.conv+p.kv)*2+16ul*p.heads*4);auto dd=reinterpret_cast<device float*>(destination+gdn_tape_decay_offset(p.conv,p.kv));auto db=reinterpret_cast<device bfloat*>(destination+gdn_tape_beta_offset(p.conv,p.kv,p.heads));for(uint row=0;row<8;++row){dd[row*p.heads+col]=sd[tree.path[row]*p.heads+col];db[row*p.heads+col]=sb[tree.path[row]*p.heads+col];}}
}

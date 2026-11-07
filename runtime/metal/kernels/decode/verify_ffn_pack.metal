#include <metal_stdlib>
#include "metal/abi/VerifyFfnPack.h"
using namespace metal;
kernel void verify_ffn_pack_map(device const uint *rows [[buffer(0)]],
 device VerifyPackControl *control [[buffer(1)]],constant VerifyPackParams &p [[buffer(2)]],uint tid [[thread_position_in_grid]]) {
 if(tid!=0)return;
 device VerifyPackControl &c=*control;c.total=0;c.valid=p.lanes>=5&&p.lanes<=8;
 for(uint i=0;i<32;++i)c.map[i]=0xffffffffu;
 for(uint i=0;i<64;++i)c.inverse[i]=0xffffffffu;
 for(uint l=0;l<min(p.lanes,8u);++l){c.valid=c.valid&&rows[l]>=1&&rows[l]<=8;c.total+=min(rows[l],8u);}
 c.compact=c.valid&&c.total<=32;
 uint at=0;
 if(c.compact)for(uint l=0;l<p.lanes;++l)for(uint r=0;r<rows[l];++r){c.map[at]=l*8+r;c.inverse[l*8+r]=at;++at;}
 const uint xs[7]={(p.hidden+255)/256,p.intermediate/32,p.hidden/64,(p.hidden+255)/256,p.intermediate/64,p.intermediate/64,p.hidden/64};
 const uint ys[7]={32,p.up_splits,p.down_splits,64,p.up_splits,p.up_splits,p.down_splits};
 for(uint i=0;i<7;++i){const bool on=i<4?bool(c.compact):!bool(c.compact);c.grids[i][0]=on?xs[i]:0;c.grids[i][1]=on?ys[i]:0;c.grids[i][2]=on?1:0;}
}
kernel void verify_ffn_pack_gather(device const ushort *input [[buffer(0)]],device const ushort *residual [[buffer(1)]],
 device const VerifyPackControl *control [[buffer(2)]],device ushort *packed_input [[buffer(3)]],device ushort *packed_residual [[buffer(4)]],constant VerifyPackParams &p [[buffer(5)]],uint2 gid [[thread_position_in_grid]]) {
 if(gid.x>=p.hidden||gid.y>=32)return;
 const uint oldrow=control->map[gid.y];const ulong dst=ulong(gid.y)*p.hidden+gid.x;
 packed_input[dst]=oldrow<64?input[ulong(oldrow)*p.hidden+gid.x]:ushort(0);
 packed_residual[dst]=oldrow<64?residual[ulong(oldrow)*p.hidden+gid.x]:ushort(0);
}
kernel void verify_ffn_pack_scatter(device const ushort *packed [[buffer(0)]],device const ushort *residual [[buffer(1)]],
 device const VerifyPackControl *control [[buffer(2)]],device ushort *output [[buffer(3)]],constant VerifyPackParams &p [[buffer(4)]],uint2 gid [[thread_position_in_grid]]) {
 if(gid.x>=p.hidden||gid.y>=64)return;
 const uint src=control->inverse[gid.y];const ulong dst=ulong(gid.y)*p.hidden+gid.x;
 output[dst]=src<control->total?packed[ulong(src)*p.hidden+gid.x]:residual[dst];
}

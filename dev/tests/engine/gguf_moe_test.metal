// gguf-moe's two-pass reference (gguf_moe_test.mm): the gate and up passes that the production gate/up pass
// (moe_expert_gguf_m<rows>_gate_up[_mixed]) runs in one, apart, each on the expert tile of the production down pass
// (kernels/common/moe_gguf_expert_tile.h), built with the production flags, each sum times its expert's scale: the gate
// pass writes bf16(gate), and the up pass bf16(bf16(up) * silu(gate)) (gguf_epilogue<EpUpWithGate>) with the gate from
// `aux`. The production gate/up pass's intermediate is bitwise theirs. A softcapped routing's up pass (Activation,
// kernels/common/softcapped_swiglu.h) writes bf16(Activation::value(bf16(up), gate)) instead.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/moe_gguf_expert_tile.h"
#include "metal/kernels/common/softcapped_swiglu.h"

// The one format a reference tile decodes, its kernel's choice (moe_gguf_expert_tile's Formats).
template <class F> struct gguf_moe_test_format {
  template <class Body> void operator()(uint, Body body) const { body(F()); }
};

// A reference pass's tile in format F, out of line, as the production gate/up pass's tiles are
// (kernels/shared/moe_gguf.metal's moe_gate_up_staged): inlined, every format's tiles grew the 32-row passes until a
// 10-core M4 (Apple9) under shader validation computed them wrongly once MXFP8 joined the formats, the same passes
// right without validation and the production passes right with it.
template <class F, ushort Rows, uint Shared, bool Up, class Activation, ushort Partitions>
__attribute__((noinline)) void gguf_moe_test_tile(device bfloat *input, device const MoeTileDescriptor *tiles,
                                                  device const uint *tile_count, device uchar *sw0, device uchar *sw1,
                                                  device uchar *smeta, device bfloat *output, device const bfloat *aux,
                                                  device const uint *slots, device const float *expert_scales,
                                                  constant MoeGgufExpertParams &p, constant ulong *chunks, uint2 group,
                                                  uint simd_lane, uint simd_group, threadgroup half *stage,
                                                  threadgroup half2 *tl) {
  moe_gguf_expert_tile<Rows, Shared, Up ? 1 : 0, Partitions>(
      input, tiles, tile_count, sw0, sw1, smeta, slots, expert_scales, p, chunks, group, simd_lane, simd_group, stage, tl,
      [&](ulong o, float v) {
        if constexpr (Up && is_same<Activation, void>::value) output[o] = gguf_epilogue<EpUpWithGate>(v, aux, o);
        else if constexpr (Up) output[o] = bfloat(Activation::value(float(bfloat(v)), float(aux[o])));
        else output[o] = bfloat(v);
      },
      gguf_moe_test_format<F>());
}

// A reference pass: each tile in the format of its expert, routed or shared, of every format the list holds, over
// Partitions K partitions as the production pass (moe_gguf_expert_tile).
template <ushort Rows, uint Shared, bool Up, class Activation = void, ushort Partitions = 1>
kernel void gguf_moe_test_pass(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                               device const uint *tile_count [[buffer(2)]], device uchar *sw0 [[buffer(3)]],
                               device uchar *sw1 [[buffer(4)]], device uchar *smeta [[buffer(5)]],
                               device bfloat *output [[buffer(6)]], device const bfloat *aux [[buffer(7)]],
                               device const uint *slots [[buffer(8)]], device const float *expert_scales [[buffer(9)]],
                               constant MoeGgufExpertParams &p [[buffer(10)]], constant ulong *chunks [[buffer(11)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint simd_lane [[thread_index_in_simdgroup]],
                               uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[Partitions * kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  const uint t = moe_first_tile(tile_count, p.tiles) + group.y;
  if (t >= moe_end_tile(tile_count, p.tiles)) return;
  const uint format = moe_shared_slice(tiles[t].expert, p.experts) < Shared ? p.shared_format : p.routed_format;
  quant_format_switch(format, [&](auto f) {
    gguf_moe_test_tile<decltype(f), Rows, Shared, Up, Activation, Partitions>(input, tiles, tile_count, sw0, sw1, smeta, output,
                                                                  aux, slots, expert_scales, p, chunks, group,
                                                                  simd_lane, simd_group, stage, tl);
  });
}
using GgufMoeTestPass = void(device bfloat *, device const MoeTileDescriptor *, device const uint *, device uchar *,
                             device uchar *, device uchar *, device bfloat *, device const bfloat *,
                             device const uint *, device const float *, constant MoeGgufExpertParams &,
                             constant ulong *, uint2, uint, uint);
template [[host_name("gguf_moe_test_m8_gate")]] kernel GgufMoeTestPass gguf_moe_test_pass<8, 1, false>;
template [[host_name("gguf_moe_test_m8_up")]] kernel GgufMoeTestPass gguf_moe_test_pass<8, 1, true>;
template [[host_name("gguf_moe_test_m16_gate")]] kernel GgufMoeTestPass gguf_moe_test_pass<16, 1, false>;
template [[host_name("gguf_moe_test_m16_up")]] kernel GgufMoeTestPass gguf_moe_test_pass<16, 1, true>;
template [[host_name("gguf_moe_test_m16_gate_s2")]] kernel GgufMoeTestPass gguf_moe_test_pass<16, 2, false>;
template [[host_name("gguf_moe_test_m16_up_s2")]] kernel GgufMoeTestPass gguf_moe_test_pass<16, 2, true>;
template [[host_name("gguf_moe_test_m16_gate_s0_softcap4_25")]] kernel GgufMoeTestPass gguf_moe_test_pass<16, 0, false>;
template [[host_name("gguf_moe_test_m16_up_s0_softcap4_25")]] kernel GgufMoeTestPass
    gguf_moe_test_pass<16, 0, true, SoftcappedSwiglu<4, 25>>;
template [[host_name("gguf_moe_test_m32_gate")]] kernel GgufMoeTestPass gguf_moe_test_pass<32, 1, false>;
template [[host_name("gguf_moe_test_m32_up")]] kernel GgufMoeTestPass gguf_moe_test_pass<32, 1, true>;
template [[host_name("gguf_moe_test_m8_gate_s2")]] kernel GgufMoeTestPass gguf_moe_test_pass<8, 2, false>;
template [[host_name("gguf_moe_test_m8_up_s2")]] kernel GgufMoeTestPass gguf_moe_test_pass<8, 2, true>;
template [[host_name("gguf_moe_test_m32_gate_s2")]] kernel GgufMoeTestPass gguf_moe_test_pass<32, 2, false>;
template [[host_name("gguf_moe_test_m32_up_s2")]] kernel GgufMoeTestPass gguf_moe_test_pass<32, 2, true>;
template [[host_name("gguf_moe_test_m8_gate_s0_softcap4_25")]] kernel GgufMoeTestPass gguf_moe_test_pass<8, 0, false>;
template [[host_name("gguf_moe_test_m8_up_s0_softcap4_25")]] kernel GgufMoeTestPass
    gguf_moe_test_pass<8, 0, true, SoftcappedSwiglu<4, 25>>;
template [[host_name("gguf_moe_test_m32_gate_s0_softcap4_25")]] kernel GgufMoeTestPass gguf_moe_test_pass<32, 0, false>;
template [[host_name("gguf_moe_test_m32_up_s0_softcap4_25")]] kernel GgufMoeTestPass
    gguf_moe_test_pass<32, 0, true, SoftcappedSwiglu<4, 25>>;
template [[host_name("gguf_moe_test_m64_gate")]] kernel GgufMoeTestPass gguf_moe_test_pass<64, 1, false>;
template [[host_name("gguf_moe_test_m64_up")]] kernel GgufMoeTestPass gguf_moe_test_pass<64, 1, true>;
template [[host_name("gguf_moe_test_m64_gate_s0_softcap4_25")]] kernel GgufMoeTestPass gguf_moe_test_pass<64, 0, false>;
template [[host_name("gguf_moe_test_m64_up_s0_softcap4_25")]] kernel GgufMoeTestPass
    gguf_moe_test_pass<64, 0, true, SoftcappedSwiglu<4, 25>>;
// The two-K-partition decode passes' references.
template [[host_name("gguf_moe_test_m8k2_gate")]] kernel GgufMoeTestPass gguf_moe_test_pass<8, 1, false, void, 2>;
template [[host_name("gguf_moe_test_m8k2_up")]] kernel GgufMoeTestPass gguf_moe_test_pass<8, 1, true, void, 2>;
template [[host_name("gguf_moe_test_m8k2_gate_s2")]] kernel GgufMoeTestPass gguf_moe_test_pass<8, 2, false, void, 2>;
template [[host_name("gguf_moe_test_m8k2_up_s2")]] kernel GgufMoeTestPass gguf_moe_test_pass<8, 2, true, void, 2>;
template [[host_name("gguf_moe_test_m8k2_gate_s0_softcap4_25")]] kernel GgufMoeTestPass
    gguf_moe_test_pass<8, 0, false, void, 2>;
template [[host_name("gguf_moe_test_m8k2_up_s0_softcap4_25")]] kernel GgufMoeTestPass
    gguf_moe_test_pass<8, 0, true, SoftcappedSwiglu<4, 25>, 2>;

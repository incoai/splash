// GGUF MoE experts on the staged decode tile (kernels/common/gguf_staged_tile.h); the Apple9 register form is in
// kernels/decode/linear_gguf_sgmatrix.metal.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

// MoE experts (ops/MoE.cpp; kernels/shared/moe.metal groups the rows): threadgroup (x, y) computes
// 64 columns of grouped tile y with the weights of the tile's expert (moe_gguf_segment), in the format the tile picks
// at run time: on a 16-core M5 Pro one run-time-format dispatch over two segments is within -11..+8% of a dispatch
// per format (23040x2048 Q4_K and 92160x512 Q5_K, one to four lanes). aux is the gate of the up pass.
// Two simdgroups each stream their own 32 columns through the decode tile, grid (N / 64, tiles), on 8-row tiles
// (decode steps, short prefill chunks) or 32-row tiles (longer chunks, ops::moeGgufPrefillTile), where a tile runs the
// 8-, 16- or 32-row matmul that holds its live rows (moe_live_rows): an expert's last tile is mostly partial. On the
// 35B's real prefill routes the 32-row tiles take 17-30% less time than 64-row tiles sharing one 64-column stage over
// four 16-row simdgroups (512 and 2048 rows, 16-core M5 Pro).
template <ushort Rows, GgufEpilogue Ep>
inline void moe_gguf_expert_tile(device bfloat *input, device const MoeTileDescriptor *tiles, device const uint *tile_count,
                                 device uchar *w0, device uchar *w1, device uchar *meta, device uchar *sw0, device uchar *sw1,
                                 device uchar *smeta, device bfloat *output, device bfloat *aux,
                                 constant MoeGgufExpertParams &p, uint2 group, uint simd_lane, uint simd_group,
                                 threadgroup half *stage, threadgroup half2 *tl) {
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s = moe_gguf_segment(tile.expert, p, w0, w1, meta, sw0, sw1, smeta);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, p.input_size, my);
    gguf_zero(acc);
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, my, tl,
                                            simd_group * 32 + simd_lane, simd_lane, 0, p.input_size / GGUF_STAGED_STEP, 0, acc);
    gguf_elements(acc, [&](uint row, uint column, float v) {
      const ulong o = out + ulong(row) * p.output_size + origin + column;
      output[o] = gguf_epilogue<Ep>(v, aux, o);
    });
  };
  moe_live_rows<Rows>(tile.rows, run);
}
template <ushort Rows, GgufEpilogue Ep>
kernel void moe_expert_gguf(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                            device const uint *tile_count [[buffer(2)]], device uchar *w0 [[buffer(3)]],
                            device uchar *w1 [[buffer(4)]], device uchar *meta [[buffer(5)]], device uchar *sw0 [[buffer(6)]],
                            device uchar *sw1 [[buffer(7)]], device uchar *smeta [[buffer(8)]],
                            device bfloat *output [[buffer(9)]], device bfloat *aux [[buffer(10)]],
                            constant MoeGgufExpertParams &p [[buffer(11)]], uint2 group [[threadgroup_position_in_grid]],
                            uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  moe_gguf_expert_tile<Rows, Ep>(input, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta, output, aux, p, group,
                                 simd_lane, simd_group, stage, tl);
}
using MoeExpertGgufKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *, device uchar *,
                                 device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                 device bfloat *, device bfloat *, constant MoeGgufExpertParams &, uint2, uint, uint);
template [[host_name("moe_expert_gguf_m8_a")]] kernel MoeExpertGgufKernel moe_expert_gguf<8, EpNone>;
template [[host_name("moe_expert_gguf_m8_g")]] kernel MoeExpertGgufKernel moe_expert_gguf<8, EpUpWithGate>;
template [[host_name("moe_expert_gguf_m32_a")]] kernel MoeExpertGgufKernel moe_expert_gguf<32, EpNone>;
template [[host_name("moe_expert_gguf_m32_g")]] kernel MoeExpertGgufKernel moe_expert_gguf<32, EpUpWithGate>;

// The gate and up passes in one, where gate and up share their routed format and their shared format (ops/MoE.cpp),
// so the pair table is the threadgroup's: threadgroup (x, y) runs a gate simdgroup and an up simdgroup on the same 32
// columns of grouped tile y, grid (N / 32, tiles), each through its own stage as in moe_gguf_expert_tile. The gate
// simdgroup parks bf16(gate) in its stage, which its loop no longer reads, and the up simdgroup writes
// bf16(up) * silu(gate) as the up pass does (gguf_epilogue<EpUpWithGate>), so the intermediate is bitwise the two
// passes'. A pass of few tiles leaves cores idle and every pass ends in a tail; one pass runs gate and up together and
// has one tail. Against the two passes, both on live grids, on the 35B's MLX 4-bit and mxfp4 and GGUF UD-Q4_K_M and
// UD-Q2_K_XL checkpoints: a 40-core M5 Max decodes 1.1-4.3% faster at two to four lanes and prefills chunks of 16 to
// 512 rows 1.1-5.1% faster, within noise (-1.4..+0.6%) at one lane and on 8-row chunks; a 12-core M6 runs 0.5-2.7%
// and a 40-core M3 Max 0.9-5.5% faster.
template <ushort Rows>
kernel void moe_expert_gguf_gate_up(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                    device const uint *tile_count [[buffer(2)]], device uchar *gw0 [[buffer(3)]],
                                    device uchar *gw1 [[buffer(4)]], device uchar *gmeta [[buffer(5)]],
                                    device uchar *gsw0 [[buffer(6)]], device uchar *gsw1 [[buffer(7)]],
                                    device uchar *gsmeta [[buffer(8)]], device uchar *uw0 [[buffer(9)]],
                                    device uchar *uw1 [[buffer(10)]], device uchar *umeta [[buffer(11)]],
                                    device uchar *usw0 [[buffer(12)]], device uchar *usw1 [[buffer(13)]],
                                    device uchar *usmeta [[buffer(14)]], device bfloat *output [[buffer(15)]],
                                    constant MoeGgufExpertParams &p [[buffer(16)]],
                                    uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const bool up = simd_group == 1;
  const MoeGgufSegment s = up ? moe_gguf_segment(tile.expert, p, uw0, uw1, umeta, usw0, usw1, usmeta)
                              : moe_gguf_segment(tile.expert, p, gw0, gw1, gmeta, gsw0, gsw1, gsmeta);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * GGUF_STAGED_COLUMNS;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  threadgroup bfloat *gate = reinterpret_cast<threadgroup bfloat *>(stage);
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    static_assert(R * GGUF_STAGED_COLUMNS * sizeof(bfloat) <= kStagedSimdgroupStage * sizeof(half), "the gate fits a stage");
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, p.input_size, my);
    gguf_zero(acc);
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, my, tl,
                                            simd_group * 32 + simd_lane, simd_lane, 0, p.input_size / GGUF_STAGED_STEP, 0, acc);
    if (!up) gguf_elements(acc, [&](uint row, uint column, float v) { gate[row * GGUF_STAGED_COLUMNS + column] = bfloat(v); });
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up)
      gguf_elements(acc, [&](uint row, uint column, float v) {
        output[out + ulong(row) * p.output_size + origin + column] =
            bfloat(float(bfloat(v)) * splash_silu(float(gate[row * GGUF_STAGED_COLUMNS + column])));
      });
  };
  moe_live_rows<Rows>(tile.rows, run);
}
using MoeExpertGgufGateUpKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                       device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                       device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                       device uchar *, device uchar *, device bfloat *, constant MoeGgufExpertParams &,
                                       uint2, uint, uint);
template [[host_name("moe_expert_gguf_m8_gate_up")]] kernel MoeExpertGgufGateUpKernel moe_expert_gguf_gate_up<8>;
template [[host_name("moe_expert_gguf_m32_gate_up")]] kernel MoeExpertGgufGateUpKernel moe_expert_gguf_gate_up<32>;

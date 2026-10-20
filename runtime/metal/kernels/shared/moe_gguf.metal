// GGUF MoE experts on the staged decode tile (kernels/common/gguf_staged_tile.h); the Apple9 register form is in
// kernels/decode/linear_gguf_sgmatrix.metal.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_live_rows.h"
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

// MoE experts (ops/MoE.cpp; kernels/shared/moe.metal groups the rows): threadgroup (x, y) computes
// 64 columns of tile y with the weights of the tile's expert (moe_gguf_segment), in the format the tile picks
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
  device bfloat *x = p.in_place ? input : input + ulong(group.y) * Rows * p.input_size;
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
// columns of tile y, grid (N / 32, tiles), each through its own stage as in moe_gguf_expert_tile. The gate
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
  device bfloat *x = p.in_place ? input : input + ulong(group.y) * Rows * p.input_size;
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

// The Fast numerics tier (ops::MoeNumerics::Fast, ops::MoePlan::smallTilesInPlace): a one-lane decode step's 8-row
// in-place tiles run the 8-row matmul whatever rows route to their expert, and most experts take one or two of the 8.
// The Fast kernels run a routed tile of at most p.gemv_rows live rows (the grouping counts and lists them,
// MoeGroupParams::in_place 2) through the register GEMV of kernels/common/gguf_live_rows.h on those rows alone,
// reading them in place, and any other tile, the shared expert's always, through the tile the default kernels run, so
// those tiles' outputs are bitwise the default kernels'. The branch is uniform in a threadgroup. A row's result then
// depends on how many of the step's rows share its expert: deterministic for a step's routes, within the fp64 bound,
// not batch-invariant. The GEMV tile's threadgroup memory is the default kernels' (the stage holds a grid format's
// grid, the pair table a codebook format's table, either the partial sums), and the default kernels stay as they are.
// Two GEMV forms, one sum order (gguf_live_rows.h): the one-pass gate/up tile's 32 columns, lane l of simdgroup s
// owning column l of gate and up over part s of K; the other passes' 64 columns, lane l of simdgroup s owning column
// 32 s + l over all of K.

// The GEMV's live rows: run(integral_constant<ushort, R>{}) for R = rows, 1 <= rows <= Max.
template <ushort Max, class Run> inline void moe_fast_rows(uint rows, Run run) {
  if constexpr (Max > 1) {
    if (rows < Max) return moe_fast_rows<Max - 1>(rows, run);
  }
  run(integral_constant<ushort, Max>{});
}

// body(F()) for the run-time format of a GEMV tile, one of moe_fast_gemv_format's (uniform in a threadgroup; any other
// id decodes as the first, which the host never passes, as quant_format_switch does).
#define MOE_FAST_GEMV_FORMATS(X) X(FmtIQ2S) X(FmtIQ2XS) X(FmtIQ2XXS) X(FmtIQ3S) X(FmtIQ3XXS) X(FmtIQ4XS) X(FmtQ4K) X(FmtQ5K) X(FmtQ6K) X(FmtQ80)
#define MOE_FAST_GEMV_CASE(F) case F::Id: body(F()); break;
template <class Body> inline void moe_fast_format_switch(uint format, Body body) {
  switch (format) {
  default:
    MOE_FAST_GEMV_FORMATS(MOE_FAST_GEMV_CASE)
  }
}
#undef MOE_FAST_GEMV_CASE
#define MOE_FAST_GEMV_ONE(F) +1
#define MOE_FAST_GEMV_LISTED(F) +moe_fast_gemv_format(F::Id)
#define QUANT_FAST_ONE(F, f) +moe_fast_gemv_format(F::Id)
static_assert(0 MOE_FAST_GEMV_FORMATS(MOE_FAST_GEMV_LISTED) == 0 MOE_FAST_GEMV_FORMATS(MOE_FAST_GEMV_ONE) &&
                  0 QUANT_FORMATS(QUANT_FAST_ONE) == 0 MOE_FAST_GEMV_FORMATS(MOE_FAST_GEMV_ONE),
              "MOE_FAST_GEMV_FORMATS lists exactly moe_fast_gemv_format's formats");
#undef QUANT_FAST_ONE
#undef MOE_FAST_GEMV_LISTED
#undef MOE_FAST_GEMV_ONE

// The threadgroup memory of a GEMV tile of format F that keeps Scratch floats of partial sums: the grid of a grid
// format at the start of the stage when it fits beside them, the sums in the pair table unless a codebook format's
// table holds it or they do not fit, else in the stage after the grid.
template <class F, uint Scratch> struct MoeFastMemory {
  static constexpr constant uint StageWords = kStagedStages / 2, TableWords = kQuantPairTableEntries;
  static constexpr constant bool ScratchInTable = F::Kind != QuantCodebook && Scratch <= TableWords;
  static constexpr constant bool TgGrid =
      LiveRowsGrid<F>::Words && LiveRowsGrid<F>::Words + (ScratchInTable ? 0 : Scratch) <= StageWords;
  static_assert(ScratchInTable || (TgGrid ? LiveRowsGrid<F>::Words : 0) + Scratch <= StageWords, "the sums fit");
  // The pair table of a codebook format and the grid copy, filled by every thread; then the sums' memory.
  static threadgroup float *begin(threadgroup half *stage, threadgroup half2 *tl, uint thread_index) {
    quant_pair_table<F>(tl, thread_index, GGUF_STAGED_THREADS);
    if constexpr (TgGrid) {
      LiveRowsGrid<F>::fill((threadgroup uint *)stage, thread_index, GGUF_STAGED_THREADS);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return ScratchInTable ? (threadgroup float *)tl : (threadgroup float *)stage + (TgGrid ? LiveRowsGrid<F>::Words : 0);
  }
};

// The one-pass gate/up GEMV of tile `tile_index` (its R live rows at live[8 tile_index ..]) over columns
// [origin, origin + 32): part simd_group of K for gate and up column origin + simd_lane, part 1's sums through
// threadgroup memory, and part 0 writes bf16(up) * silu(bf16(gate)) as the default gate/up tile does.
template <class F, ushort R>
inline void moe_fast_gate_up_gemv(device bfloat *input, device const uint *live, uint tile_index, MoeGgufSegment g,
                                  MoeGgufSegment u, constant MoeGgufExpertParams &p, uint origin, uint simd_lane,
                                  uint simd_group, device bfloat *output, threadgroup half *stage,
                                  threadgroup half2 *tl) {
  typedef MoeFastMemory<F, 2 * R * GGUF_STAGED_COLUMNS> Memory;
  threadgroup float *partial = Memory::begin(stage, tl, simd_group * 32 + simd_lane);
  threadgroup const uint *grid = (threadgroup const uint *)stage;
  uint row[R];
  ulong rows[R];
#pragma unroll
  for (ushort r = 0; r < R; ++r) {
    row[r] = live[tile_index * 8 + r];
    rows[r] = ulong(row[r]) * p.input_size;
  }
  LiveRowsColumn<F> columns[2];
  columns[0].begin(g.w0, g.w1, g.meta, p.input_size, origin + simd_lane);
  columns[1].begin(u.w0, u.w1, u.meta, p.input_size, origin + simd_lane);
  float sum[1][2][R];
  live_rows_sums<F, kLiveRowsParts, 1, 2, R, Memory::TgGrid>(columns, p.input_size / 32, simd_group, input, rows, tl, grid, sum);
  if (simd_group == 1)
#pragma unroll
    for (ushort s = 0; s < 2; ++s)
#pragma unroll
      for (ushort r = 0; r < R; ++r) partial[(s * R + r) * GGUF_STAGED_COLUMNS + simd_lane] = sum[0][s][r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 1) return;
#pragma unroll
  for (ushort r = 0; r < R; ++r) {
    const float gate = sum[0][0][r] + partial[r * GGUF_STAGED_COLUMNS + simd_lane];
    const float up = sum[0][1][r] + partial[(R + r) * GGUF_STAGED_COLUMNS + simd_lane];
    output[(ulong(tile_index) * 8 + row[r]) * p.output_size + origin + simd_lane] =
        bfloat(float(bfloat(up)) * splash_silu(float(bfloat(gate))));
  }
}

// The GEMV of one pass of tile `tile_index` for column `column` over all of K, epilogue Ep: the gate/up passes read
// the plan's rows in place, the down pass the tile's rows of its input.
template <class F, ushort R, GgufEpilogue Ep>
inline void moe_fast_gemv(device bfloat *input, device const uint *live, uint tile_index, MoeGgufSegment s,
                          constant MoeGgufExpertParams &p, uint column, uint thread_index, device bfloat *output,
                          device bfloat *aux, threadgroup half *stage, threadgroup half2 *tl) {
  typedef MoeFastMemory<F, 0> Memory;
  static_cast<void>(Memory::begin(stage, tl, thread_index));
  device bfloat *x = p.in_place ? input : input + ulong(tile_index) * 8 * p.input_size;
  uint row[R];
  ulong rows[R];
#pragma unroll
  for (ushort r = 0; r < R; ++r) {
    row[r] = live[tile_index * 8 + r];
    rows[r] = ulong(row[r]) * p.input_size;
  }
  LiveRowsColumn<F> columns[1];
  columns[0].begin(s.w0, s.w1, s.meta, p.input_size, column);
  float sum[kLiveRowsParts][1][R];
  live_rows_sums<F, kLiveRowsParts, kLiveRowsParts, 1, R, Memory::TgGrid>(columns, p.input_size / 32, 0, x, rows, tl,
                                                          (threadgroup const uint *)stage, sum);
  static_assert(kLiveRowsParts == 2, "the parts add in part order");
#pragma unroll
  for (ushort r = 0; r < R; ++r) {
    const ulong o = (ulong(tile_index) * 8 + row[r]) * p.output_size + column;
    output[o] = gguf_epilogue<Ep>(sum[0][0][r] + sum[1][0][r], aux, o);
  }
}

// Whether tile `tile` of a Fast pass runs the GEMV.
inline bool moe_fast_gemv_tile(MoeTileDescriptor tile, constant MoeGgufExpertParams &p) {
  return tile.expert != p.experts && tile.rows <= p.gemv_rows;
}

// The gate, up and down passes of the Fast tier: moe_expert_gguf<8, Ep>'s bindings with the live rows at buffer(11).
template <GgufEpilogue Ep>
kernel void moe_expert_gguf_fast(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                 device const uint *tile_count [[buffer(2)]], device uchar *w0 [[buffer(3)]],
                                 device uchar *w1 [[buffer(4)]], device uchar *meta [[buffer(5)]],
                                 device uchar *sw0 [[buffer(6)]], device uchar *sw1 [[buffer(7)]],
                                 device uchar *smeta [[buffer(8)]], device bfloat *output [[buffer(9)]],
                                 device bfloat *aux [[buffer(10)]], device const uint *live [[buffer(11)]],
                                 constant MoeGgufExpertParams &p [[buffer(12)]],
                                 uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  if (moe_fast_gemv_tile(tile, p)) {
    const MoeGgufSegment s = moe_gguf_segment(tile.expert, p, w0, w1, meta, sw0, sw1, smeta);
    const uint column = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS + simd_lane;
    moe_fast_rows<SPLASH_MOE_FAST_MAX_ROWS>(tile.rows, [&](auto rows) {
      moe_fast_format_switch(s.format, [&](auto format) {
        moe_fast_gemv<decltype(format), decltype(rows)::value, Ep>(input, live, group.y, s, p, column,
                                                                   simd_group * 32 + simd_lane, output, aux, stage, tl);
      });
    });
    return;
  }
  moe_gguf_expert_tile<8, Ep>(input, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta, output, aux, p, group,
                              simd_lane, simd_group, stage, tl);
}
using MoeExpertGgufFastKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                     device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                     device uchar *, device bfloat *, device bfloat *, device const uint *,
                                     constant MoeGgufExpertParams &, uint2, uint, uint);
template [[host_name("moe_expert_gguf_fast_m8_a")]] kernel MoeExpertGgufFastKernel moe_expert_gguf_fast<EpNone>;
template [[host_name("moe_expert_gguf_fast_m8_g")]] kernel MoeExpertGgufFastKernel moe_expert_gguf_fast<EpUpWithGate>;

// The one-pass gate/up of the Fast tier: moe_expert_gguf_gate_up<8>'s bindings with the live rows at buffer(16). Its
// other tiles run moe_expert_gguf_gate_up<8>'s tile.
kernel void moe_expert_gguf_fast_m8_gate_up(
    device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
    device const uint *tile_count [[buffer(2)]], device uchar *gw0 [[buffer(3)]], device uchar *gw1 [[buffer(4)]],
    device uchar *gmeta [[buffer(5)]], device uchar *gsw0 [[buffer(6)]], device uchar *gsw1 [[buffer(7)]],
    device uchar *gsmeta [[buffer(8)]], device uchar *uw0 [[buffer(9)]], device uchar *uw1 [[buffer(10)]],
    device uchar *umeta [[buffer(11)]], device uchar *usw0 [[buffer(12)]], device uchar *usw1 [[buffer(13)]],
    device uchar *usmeta [[buffer(14)]], device bfloat *output [[buffer(15)]], device const uint *live [[buffer(16)]],
    constant MoeGgufExpertParams &p [[buffer(17)]], uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort Rows = 8;
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  if (moe_fast_gemv_tile(tile, p)) {
    const MoeGgufSegment g = moe_gguf_segment(tile.expert, p, gw0, gw1, gmeta, gsw0, gsw1, gsmeta);
    const MoeGgufSegment u = moe_gguf_segment(tile.expert, p, uw0, uw1, umeta, usw0, usw1, usmeta);
    moe_fast_rows<SPLASH_MOE_FAST_MAX_ROWS>(tile.rows, [&](auto rows) {
      moe_fast_format_switch(g.format, [&](auto format) {
        moe_fast_gate_up_gemv<decltype(format), decltype(rows)::value>(input, live, group.y, g, u, p,
                                                                       group.x * GGUF_STAGED_COLUMNS, simd_lane,
                                                                       simd_group, output, stage, tl);
      });
    });
    return;
  }
  // moe_expert_gguf_gate_up<8>'s tile.
  const bool up = simd_group == 1;
  const MoeGgufSegment s = up ? moe_gguf_segment(tile.expert, p, uw0, uw1, umeta, usw0, usw1, usmeta)
                              : moe_gguf_segment(tile.expert, p, gw0, gw1, gmeta, gsw0, gsw1, gsmeta);
  device bfloat *x = p.in_place ? input : input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * GGUF_STAGED_COLUMNS;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  threadgroup bfloat *gate = reinterpret_cast<threadgroup bfloat *>(stage);
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
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

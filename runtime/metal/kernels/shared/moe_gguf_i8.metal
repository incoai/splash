// Int8 activations for the GGUF MoE decode passes (ops::MoeInt8 in ops/MoE.hpp, metal/abi/MoE.h MoeInt8Params;
// splash serve --moe-int8, off by default, so no default plan dispatches these kernels).
// moe_gather_rows_a8: moe_gather_rows plus each grouped row as int8 and one fp32 scale (the A8 pre-pass, fused into the
// gather). moe_a8_fake: the bf16 passes' reference, the same rounding written back as bf16(s q). The W-A8 expert passes
// mirror moe_expert_gguf_gate_up and moe_expert_gguf_m8_a (kernels/shared/moe_gguf.metal) on 8-row tiles: a tile
// whose segment's format is in MoeInt8Params::formats runs the int8 walk (kernels/common/gguf_staged_i8.h), every
// other tile the half walk on the bf16 rows; the up epilogue also writes the down input's int8 rows with a scale per
// row and 32 columns (one gate/up threadgroup's columns).
#pragma clang fp reassociate(off)
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/gguf_column_tile.h"
#include "metal/kernels/common/gguf_staged_i8.h"
#include "metal/kernels/common/moe_expert_slab.h"

inline uint moe_a8_pack(float2 v, float s) {
  return uint(uchar(char(moe_a8_code(v.x, s)))) | uint(uchar(char(moe_a8_code(v.y, s)))) << 8;
}

// Grid (tiles, input_size / 256), 256 threads: moe_gather_rows' copy, and per row its whole input's max |x| (each
// simdgroup one row, every part reading the row again, from cache), its scale and the part's 256 int8 values.
kernel void moe_gather_rows_a8(device const bfloat *input [[buffer(0)]], device const uint *grouped_routes [[buffer(1)]],
                               device const MoeTileDescriptor *tiles [[buffer(2)]],
                               device const uint *tile_count [[buffer(3)]], device bfloat *grouped_input [[buffer(4)]],
                               device uchar *a8 [[buffer(5)]], constant MoeGatherA8Params &p [[buffer(6)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint thread_index [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  if (group.x >= *tile_count) return;
  const uint K = p.input_size, column = group.y * 256 + lane * 8;
  const uint rows = moe_staged_rows(tiles, group.x, p.tile_rows);
  device int8_t *x8 = reinterpret_cast<device int8_t *>(a8 + p.x8);
  device float *xs = reinterpret_cast<device float *>(a8 + p.x_scales);
  for (uint local = thread_index / 32; local < rows; local += 8) {
    const uint row = group.x * p.tile_rows + local;
    const uint route = grouped_routes[row];
    uint4 v = uint4(0);
    float m = 0.0f;
    if (route != ~0u) {
      device const bfloat *src = input + ulong(route / p.routes_per_row) * K;
      v = *reinterpret_cast<device const uint4 *>(src + column);
      for (uint j = lane * 8; j < K; j += 256) {
        const uint4 u = *reinterpret_cast<device const uint4 *>(src + j);
#pragma unroll
        for (ushort i = 0; i < 4; ++i) {
          const float2 f = float2(as_type<bfloat2>(u[i]));
          m = max(m, max(fabs(f.x), fabs(f.y)));
        }
      }
    }
    m = simd_max(m);
    *reinterpret_cast<device uint4 *>(grouped_input + ulong(row) * K + column) = v;
    const float s = moe_a8_scale(m, p.mode);
    uint2 q;
    q.x = moe_a8_pack(float2(as_type<bfloat2>(v.x)), s) | moe_a8_pack(float2(as_type<bfloat2>(v.y)), s) << 16;
    q.y = moe_a8_pack(float2(as_type<bfloat2>(v.z)), s) | moe_a8_pack(float2(as_type<bfloat2>(v.w)), s) << 16;
    *reinterpret_cast<device uint2 *>(x8 + ulong(row) * K + column) = q;
    if (group.y == 0 && lane == 0) xs[row] = s;
  }
}

// Grid (tiles, tile_rows), 32 threads: grouped row (tile, y) of `width`, in blocks of `block` (the width, or 32), as
// bf16(s q) with the scale of each block, over the rows the staged passes read.
kernel void moe_a8_fake(device bfloat *rows [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                        device const uint *tile_count [[buffer(2)]], constant MoeA8FakeParams &p [[buffer(3)]],
                        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (group.x >= *tile_count || group.y >= moe_staged_rows(tiles, group.x, p.tile_rows)) return;
  device bfloat *x = rows + (ulong(group.x) * p.tile_rows + group.y) * p.width;
  if (p.block == p.width) {
    float m = 0.0f;
    for (uint j = lane; j < p.width; j += 32) m = max(m, fabs(float(x[j])));
    m = simd_max(m);
    const float s = moe_a8_scale(m, p.mode);
    for (uint j = lane; j < p.width; j += 32) x[j] = bfloat(float(moe_a8_code(float(x[j]), s)) * s);
    return;
  }
  for (uint b = lane; b < p.width / 32; b += 32) {
    float m = 0.0f;
    for (uint j = 0; j < 32; ++j) m = max(m, fabs(float(x[b * 32 + j])));
    const float s = moe_a8_scale(m, p.mode);
    for (uint j = 0; j < 32; ++j) x[b * 32 + j] = bfloat(float(moe_a8_code(float(x[b * 32 + j]), s)) * s);
  }
}

inline bool moe_i8_takes(constant MoeInt8Params &q, uint format) { return format < 32 && ((q.formats >> format) & 1u); }

// A gate/up column tile's sums (moe_gate_up_column's); out of line as moe_gate_up_column.
__attribute__((noinline)) void moe_i8_gate_up_column(device bfloat *x, uint rows, uint K, MoeGgufSegment s,
                                                     uint column, threadgroup half2 *table,
                                                     thread float (&sums)[MOE_GGUF_COLUMN_ROWS]) {
  quant_format_switch(s.format, [&](auto f) {
    column_dot<decltype(f), MOE_GGUF_COLUMN_ROWS>(x, rows, K, s.w0, s.w1, s.meta, column, 0, K / 32, table, sums);
  });
}

// moe_expert_gguf_gate_up on 8-row tiles with int8 walks: each simdgroup's values go to its stage as bf16 (the gate
// simdgroup's are the gate the up pass reads, the up's its bf16(up)), then the up simdgroup, lane l on column l, writes
// bf16(bf16(up) * silu(gate)) as the up pass does and, with MOE_A8_DOWN, the down input's int8 values and the scale
// of the row's 32 columns.
template <ushort Tables, uint Shared>
kernel void moe_expert_gguf_i8_gate_up(device bfloat *input [[buffer(0)]],
                                       device const MoeTileDescriptor *tiles [[buffer(1)]],
                                       device const uint *tile_count [[buffer(2)]], device uchar *gsw0 [[buffer(3)]],
                                       device uchar *gsw1 [[buffer(4)]], device uchar *gsmeta [[buffer(5)]],
                                       device uchar *usw0 [[buffer(6)]], device uchar *usw1 [[buffer(7)]],
                                       device uchar *usmeta [[buffer(8)]], device bfloat *output [[buffer(9)]],
                                       device const uint *slots [[buffer(10)]], device uchar *a8 [[buffer(11)]],
                                       constant MoeGgufI8GateUpParams &pp [[buffer(12)]],
                                       constant ulong *chunks [[buffer(13)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint simd_lane [[thread_index_in_simdgroup]],
                                       uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort Rows = 8;
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[Tables * kQuantPairTableEntries];
  constant MoeGgufGateUpParams &p = pp.g;
  constant MoeInt8Params &q = pp.a;
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const bool up = simd_group == 1;
  const MoeGgufSegment gs = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size,
                                                     p.gate_routed_format, p.gate_shared_format, slots, chunks,
                                                     p.floor_slots, p.chunk_slots, 0, gsw0, gsw1, gsmeta);
  const MoeGgufSegment us = moe_gguf_segment<Shared>(tile.expert, p.experts, p.input_size, p.output_size,
                                                     p.up_routed_format, p.up_shared_format, slots, chunks,
                                                     p.floor_slots, p.chunk_slots, 1, usw0, usw1, usmeta);
  const uint thread_index = simd_group * 32 + simd_lane;
  bool tables = false;
  quant_format_switch(gs.format, [&](auto f) {
    tables |= quant_pair_entries<decltype(f)>(tl, thread_index, GGUF_STAGED_THREADS);
  });
  if constexpr (Tables == 2)
    quant_format_switch(us.format, [&](auto f) {
      tables |= quant_pair_entries<decltype(f)>(tl + kQuantPairTableEntries, thread_index, GGUF_STAGED_THREADS);
    });
  if (tables) threadgroup_barrier(mem_flags::mem_threadgroup);
  const MoeGgufSegment s = up ? us : gs;
  threadgroup half2 *table = tl + (Tables == 2 && up ? kQuantPairTableEntries : 0);
  const uint K = p.input_size, N = p.output_size;
  const uint first = group.y * Rows;
  device bfloat *x = input + ulong(first) * K;
  device int8_t *x8 = reinterpret_cast<device int8_t *>(a8 + q.x8) + ulong(first) * K;
  device const float *xs = reinterpret_cast<device const float *>(a8 + q.x_scales) + first;
  const ulong out = ulong(first) * N;
  const uint origin = group.x * GGUF_STAGED_COLUMNS;
  const bool i8 = moe_i8_takes(q, s.format);
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  // This simdgroup's values, bf16, row-major 8 x 32 at the start of its stage (the gate's where the up reads it).
  threadgroup bfloat *values = reinterpret_cast<threadgroup bfloat *>(my);
  uint rows = Rows;
  if (moe_column_tile(tiles, group.y)) {
    rows = tile.rows;
    float sums[MOE_GGUF_COLUMN_ROWS];
    moe_i8_gate_up_column(x, rows, K, s, origin + simd_lane, table, sums);
    for (uint row = 0; row < rows; ++row) values[row * GGUF_STAGED_COLUMNS + simd_lane] = bfloat(sums[row]);
  } else {
    quant_format_switch(s.format, [&](auto f) {
      typedef decltype(f) F;
      if constexpr (moe_i8_format<F>()) {
        if (i8) {
          I8Elements<Rows> e;
          staged_i8_accumulate<F, Rows, false>(x8, nullptr, K, s.w0, s.w1, s.meta, origin,
                                               reinterpret_cast<threadgroup uchar *>(my), table, simd_lane, e);
#pragma unroll
          for (ushort i = 0; i < 2 * Rows; ++i)
            if (e.ev[i]) values[e.er[i] * GGUF_STAGED_COLUMNS + e.ec[i]] = bfloat(e.fa[i] * xs[e.er[i]]);
          return;
        }
      }
      auto acc = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, K, my);
      gguf_zero(acc);
      staged_accumulate<F, Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, s.w0, s.w1, s.meta, K, origin, my, table,
                                                                        simd_lane, 0, K / GGUF_STAGED_STEP, 0, acc);
      gguf_elements(acc, [&](uint row, uint column, float v) { values[row * GGUF_STAGED_COLUMNS + column] = bfloat(v); });
    });
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!up) return;
  threadgroup bfloat *gate = reinterpret_cast<threadgroup bfloat *>(stage);
  device int8_t *d8 = reinterpret_cast<device int8_t *>(a8 + q.d8) + ulong(first) * N;
  device float *ds = reinterpret_cast<device float *>(a8 + q.d_scales) + ulong(first) * (N / 32);
  for (uint row = 0; row < rows; ++row) {
    const uint at = row * GGUF_STAGED_COLUMNS + simd_lane;
    const bfloat o = bfloat(float(values[at]) * splash_silu(float(gate[at])));
    output[out + ulong(row) * N + origin + simd_lane] = o;
    if (q.mode & MOE_A8_DOWN) {
      const float m = simd_max(fabs(float(o)));
      const float sc = moe_a8_scale(m, q.mode);
      d8[ulong(row) * N + origin + simd_lane] = int8_t(moe_a8_code(float(o), sc));
      if (simd_lane == 0) ds[ulong(row) * (N / 32) + origin / 32] = sc;
    }
  }
}
using MoeExpertGgufI8GateUpKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                         device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                         device uchar *, device bfloat *, device const uint *, device uchar *,
                                         constant MoeGgufI8GateUpParams &, constant ulong *, uint2, uint, uint);
template [[host_name("moe_expert_gguf_i8_m8_gate_up")]] kernel MoeExpertGgufI8GateUpKernel
    moe_expert_gguf_i8_gate_up<1, 1>;
template [[host_name("moe_expert_gguf_i8_m8_gate_up_mixed")]] kernel MoeExpertGgufI8GateUpKernel
    moe_expert_gguf_i8_gate_up<2, 1>;
template [[host_name("moe_expert_gguf_i8_m8_gate_up_s2")]] kernel MoeExpertGgufI8GateUpKernel
    moe_expert_gguf_i8_gate_up<1, 2>;
template [[host_name("moe_expert_gguf_i8_m8_gate_up_mixed_s2")]] kernel MoeExpertGgufI8GateUpKernel
    moe_expert_gguf_i8_gate_up<2, 2>;

// moe_expert_gguf_m8_a for the down pass with int8 walks on the down input's int8 rows and per-32 scales.
template <uint Shared>
kernel void moe_expert_gguf_i8_down(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                    device const uint *tile_count [[buffer(2)]], device uchar *sw0 [[buffer(3)]],
                                    device uchar *sw1 [[buffer(4)]], device uchar *smeta [[buffer(5)]],
                                    device bfloat *output [[buffer(6)]], device bfloat *aux [[buffer(7)]],
                                    device const uint *slots [[buffer(8)]], device uchar *a8 [[buffer(9)]],
                                    constant MoeGgufI8ExpertParams &pp [[buffer(10)]],
                                    constant ulong *chunks [[buffer(11)]], uint2 group [[threadgroup_position_in_grid]],
                                    uint simd_lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort Rows = 8;
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  constant MoeGgufExpertParams &p = pp.e;
  constant MoeInt8Params &q = pp.a;
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s = moe_gguf_segment<Shared>(tile.expert, p, slots, chunks, sw0, sw1, smeta);
  const uint K = p.input_size, N = p.output_size, first = group.y * Rows;
  device bfloat *x = input + ulong(first) * K;
  device int8_t *x8 = reinterpret_cast<device int8_t *>(a8 + q.d8) + ulong(first) * K;
  device const float *xs = reinterpret_cast<device const float *>(a8 + q.d_scales) + ulong(first) * (K / 32);
  const ulong out = ulong(first) * N;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  const uint thread_index = simd_group * 32 + simd_lane;
  const bool i8 = moe_i8_takes(q, s.format);
  if (moe_column_tile(tiles, group.y)) {
    float sums[MOE_GGUF_COLUMN_ROWS];
    column_dot_any<MOE_GGUF_COLUMN_ROWS>(s.format, x, tile.rows, K, s.w0, s.w1, s.meta, origin + simd_lane, 0, K / 32,
                                         tl, thread_index, GGUF_STAGED_THREADS, sums);
    for (uint row = 0; row < tile.rows; ++row) {
      const ulong o = out + ulong(row) * N + origin + simd_lane;
      output[o] = gguf_epilogue<EpNone>(sums[row], aux, o);
    }
    return;
  }
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  quant_format_switch(s.format, [&](auto f) {
    typedef decltype(f) F;
    quant_pair_table<F>(tl, thread_index, GGUF_STAGED_THREADS);
    if constexpr (moe_i8_format<F>()) {
      if (i8) {
        I8Elements<Rows> e;
        staged_i8_accumulate<F, Rows, true>(x8, xs, K, s.w0, s.w1, s.meta, origin,
                                            reinterpret_cast<threadgroup uchar *>(my), tl, simd_lane, e);
#pragma unroll
        for (ushort i = 0; i < 2 * Rows; ++i)
          if (e.ev[i]) {
            const ulong o = out + ulong(e.er[i]) * N + origin + e.ec[i];
            output[o] = gguf_epilogue<EpNone>(e.fa[i], aux, o);
          }
        return;
      }
    }
    auto acc = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, K, my);
    gguf_zero(acc);
    staged_accumulate<F, Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, s.w0, s.w1, s.meta, K, origin, my, tl,
                                                                      simd_lane, 0, K / GGUF_STAGED_STEP, 0, acc);
    gguf_elements(acc, [&](uint row, uint column, float v) {
      const ulong o = out + ulong(row) * N + origin + column;
      output[o] = gguf_epilogue<EpNone>(v, aux, o);
    });
  });
}
using MoeExpertGgufI8DownKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                       device uchar *, device uchar *, device uchar *, device bfloat *, device bfloat *,
                                       device const uint *, device uchar *, constant MoeGgufI8ExpertParams &,
                                       constant ulong *, uint2, uint, uint);
template [[host_name("moe_expert_gguf_i8_m8_a")]] kernel MoeExpertGgufI8DownKernel moe_expert_gguf_i8_down<1>;
template [[host_name("moe_expert_gguf_i8_m8_a_s2")]] kernel MoeExpertGgufI8DownKernel moe_expert_gguf_i8_down<2>;

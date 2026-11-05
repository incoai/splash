// ULMoE W2 routed experts (ops/MoE.cpp addUlmoeRouted, metal/abi/MoE.h MoeUlmoeParams): a ULMoE
// block's grouped passes. The shared expert's tiles (GGUF planes) run in the same dispatches on the GGUF column GEMV
// (kernels/common/gguf_column_tile.h) over their bf16 rows. Per step: moe_gather_rows_ulmoe[_a8] (routed rows in the
// block's salience order, int8 with a fp32 row scale on the int8 path), the gate/up pass of 2I columns
// (moe_ulmoe_w8 / moe_ulmoe_bf16, gate then up columns, bf16 into expertOutput), moe_ulmoe_swiglu[_a8]
// (bf16(bf16(up) silu(bf16(gate))), int8 with a row scale on the int8 path, into expertIntermediate), and the down pass
// (same kernels, H columns). Weights: metal/abi/MoE.h (codes restaged byte-major, meta, anchors).
#pragma clang fp reassociate(off)
#include "metal/abi/MoE.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/gguf_column_tile.h"
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

// ---------------------------------------------------------------- gather
// Threadgroup (tile, row) of 256 threads, each input_size / 256 inputs of the row. Routed tiles: bf16 (Int8 false)
// or int8 rows (row r at tile byte r * K, the 8 rows' scales after 8 K bytes); the scale of a row is max|x| / 127
// (1 for a zero row), q = clamp(rint(x / scale), -127, 127). Shared tiles and padding rows: bf16 rows / zeros.
template <bool Int8>
inline void moe_gather_ulmoe(device const bfloat *input, device const uint *grouped_routes,
                             device const MoeTileDescriptor *tiles, device const uint *tile_count, device uchar *grouped,
                             device const uint *order, constant MoeUlmoeGatherParams &p, uint2 group, uint thread_index,
                             uint simd_lane, uint simd_group, threadgroup float *partial) {
  if (group.x >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.x];
  const uint K = p.input_size, row = group.y, per = K / 256, first = thread_index * per;
  const uint route = grouped_routes[group.x * p.tile_rows + row];
  device const bfloat *x = input + ulong(route / p.routes_per_row) * K;
  const ulong tileBytes = ulong(p.tile_rows) * K * 2;
  if (tile.expert >= p.experts || !Int8) {
    const bool routed = tile.expert < p.experts;
    device bfloat *out = (device bfloat *)(grouped + group.x * tileBytes) + ulong(row) * K;
    for (uint i = first; i < first + per; ++i)
      out[i] = route == ~0u ? bfloat(0.0f) : x[routed ? order[i] : i];
    return;
  }
  float v[16];
  float mx = 0.0f;
  for (uint i = 0; i < per; ++i) {
    v[i] = route == ~0u ? 0.0f : float(x[order[first + i]]);
    mx = max(mx, fabs(v[i]));
  }
  mx = simd_max(mx);
  if (simd_lane == 0) partial[simd_group] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mx = 0.0f;
  for (uint s = 0; s < 8; ++s) mx = max(mx, partial[s]);
  const float scale = mx > 0.0f ? mx / 127.0f : 1.0f;
  device int8_t *q = (device int8_t *)(grouped + group.x * tileBytes) + ulong(row) * K;
  for (uint i = 0; i < per; ++i) q[first + i] = int8_t(clamp(rint(v[i] / scale), -127.0f, 127.0f));
  if (thread_index == 0) ((device float *)(grouped + group.x * tileBytes + ulong(p.tile_rows) * K))[row] =
      route == ~0u ? 0.0f : scale;
}
#define MOE_GATHER_ULMOE(name, int8)                                                                                   \
  kernel void name(device const bfloat *input [[buffer(0)]], device const uint *grouped_routes [[buffer(1)]],          \
                   device const MoeTileDescriptor *tiles [[buffer(2)]], device const uint *tile_count [[buffer(3)]],   \
                   device uchar *grouped [[buffer(4)]], device const uint *order [[buffer(5)]],                        \
                   constant MoeUlmoeGatherParams &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],      \
                   uint thread_index [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],    \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {                                               \
    threadgroup float partial[8];                                                                                      \
    moe_gather_ulmoe<int8>(input, grouped_routes, tiles, tile_count, grouped, order, p, group, thread_index,           \
                           simd_lane, simd_group, partial);                                                            \
  }
MOE_GATHER_ULMOE(moe_gather_rows_ulmoe, false)
MOE_GATHER_ULMOE(moe_gather_rows_ulmoe_a8, true)

// ---------------------------------------------------------------- swiglu between the passes
// Threadgroup (tile, row) of 256 threads: h = bf16(bf16(up) silu(bf16(gate))) of the gate/up pass's 2I bf16 columns
// (gate c at c, up c at I + c), as the GGUF gate/up epilogue writes it; bf16 rows of I, or for routed tiles on the
// int8 path int8 rows with a row scale (moe_gather_ulmoe's rule), in the tile's block of the intermediate; padding
// rows zero.
template <bool Int8>
inline void moe_swiglu_ulmoe(device const bfloat *raw, device const MoeTileDescriptor *tiles,
                             device const uint *tile_count, device uchar *intermediate, constant MoeUlmoeParams &p,
                             uint2 group, uint thread_index, uint simd_lane, uint simd_group, threadgroup float *partial) {
  if (group.x >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.x];
  const uint I = p.output_size, row = group.y, per = I / 256, first = thread_index * per;
  const bool live = row < tile.rows;
  device const bfloat *gu = raw + (ulong(group.x) * 8 + row) * 2 * I;
  float v[8];
  float mx = 0.0f;
  for (uint i = 0; i < per; ++i) {
    const uint c = first + i;
    v[i] = live ? float(bfloat(float(gu[I + c]) * splash_silu(float(gu[c])))) : 0.0f;
    mx = max(mx, fabs(v[i]));
  }
  device uchar *block = intermediate + ulong(group.x) * p.input_tile_bytes;
  if (!Int8 || tile.expert >= p.experts) {
    for (uint i = 0; i < per; ++i) ((device bfloat *)block)[ulong(row) * I + first + i] = bfloat(v[i]);
    return;
  }
  mx = simd_max(mx);
  if (simd_lane == 0) partial[simd_group] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mx = 0.0f;
  for (uint s = 0; s < 8; ++s) mx = max(mx, partial[s]);
  const float scale = mx > 0.0f ? mx / 127.0f : 1.0f;
  device int8_t *q = (device int8_t *)block + ulong(row) * I;
  for (uint i = 0; i < per; ++i) q[first + i] = int8_t(clamp(rint(v[i] / scale), -127.0f, 127.0f));
  if (thread_index == 0) ((device float *)(block + 8 * I))[row] = live ? scale : 0.0f;
}
#define MOE_SWIGLU_ULMOE(name, int8)                                                                                   \
  kernel void name(device const bfloat *raw [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],        \
                   device const uint *tile_count [[buffer(2)]], device uchar *intermediate [[buffer(3)]],              \
                   constant MoeUlmoeParams &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],             \
                   uint thread_index [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],    \
                   uint simd_group [[simdgroup_index_in_threadgroup]]) {                                               \
    threadgroup float partial[8];                                                                                      \
    moe_swiglu_ulmoe<int8>(raw, tiles, tile_count, intermediate, p, group, thread_index, simd_lane, simd_group,         \
                           partial);                                                                                   \
  }
MOE_SWIGLU_ULMOE(moe_ulmoe_swiglu, false)
MOE_SWIGLU_ULMOE(moe_ulmoe_swiglu_a8, true)

// ---------------------------------------------------------------- shared expert tiles
// Threadgroup memory of the ULMoE passes: the routed paths' stages alias the shared tiles' staged stages and pair
// table, which the GGUF expert passes use (kernels/shared/moe_gguf.metal).
constant constexpr uint kUlmoeStageHalves = kStagedStages + 2 * kQuantPairTableEntries;

// The shared expert's tile of a pass as the GGUF expert passes run it (moe_gguf_expert_tile): each simdgroup 32
// columns, the planes of set 0 below shared_split and of set 1 (at column - shared_split) from it on (a threadgroup's
// 64 columns lie in one set), over the tile's bf16 rows; tiles of <= MOE_GGUF_COLUMN_ROWS live rows on the column
// GEMV, the others on the staged tile's 8-row matmuls.
inline void ulmoe_shared_tile(device uchar *input, MoeTileDescriptor tile, uint t, device uchar *w00,
                              device uchar *w10, device uchar *meta0, device uchar *w01, device uchar *w11,
                              device uchar *meta1, device bfloat *output, constant MoeUlmoeParams &p, uint x,
                              uint simd_lane, uint simd_group, threadgroup half *stage) {
  const uint K = p.input_size, N = p.output_size, first = x * 64;
  const bool second = first >= p.shared_split;
  const uint format = second ? p.shared_format1 : p.shared_format0, base = second ? p.shared_split : 0;
  device uchar *w0 = second ? w01 : w00, *w1 = second ? w11 : w10, *meta = second ? meta1 : meta0;
  device bfloat *rows = (device bfloat *)(input + ulong(t) * p.input_tile_bytes);
  threadgroup half2 *table = (threadgroup half2 *)(stage + kStagedStages);
  const uint origin = first - base + simd_group * GGUF_STAGED_COLUMNS;
  const ulong out = ulong(t) * 8 * N + base;
  if (tile.rows <= MOE_GGUF_COLUMN_ROWS) {
    float sums[MOE_GGUF_COLUMN_ROWS];
    column_dot_any<MOE_GGUF_COLUMN_ROWS>(format, rows, tile.rows, K, w0, w1, meta, origin + simd_lane, 0, K / 32,
                                         table, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS, sums);
    for (uint r = 0; r < tile.rows; ++r) output[out + ulong(r) * N + origin + simd_lane] = bfloat(sums[r]);
    return;
  }
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  const auto run = [&](auto live) {
    constexpr ushort R = decltype(live)::value;
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(rows, K, my);
    gguf_zero(acc);
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(format, rows, w0, w1, meta, K, origin, my, table,
                                                                    simd_group * 32 + simd_lane, simd_lane, 0,
                                                                    K / GGUF_STAGED_STEP, 0, acc);
    gguf_elements(acc, [&](uint row, uint column, float v) {
      if (row < tile.rows) output[out + ulong(row) * N + origin + column] = bfloat(v);
    });
  };
  moe_live_rows<8>(tile.rows, run);
}

// ---------------------------------------------------------------- int8 path
// micro group j of meta mt: (s_j, byte bias (128 - z_j s_j) x 0x01010101)
inline uint2 ulw_mg(uint mt, uint j) {
  const uint s = 2u + 2u * (mt & 1u) + ((mt >> (16 + 2 * j)) & 3u), z = ((mt >> (8 + j)) & 1u) + 1u;
  return uint2(s, (128u - z * s) * 0x01010101u);
}
// bytes (q - z_j) s_j of inputs 4c .. 4c + 3 of a restaged word (no carries: bytes 114..149 before the xor)
inline uint ulw_bytes(uint word, uint c, uint2 k) { return (((word >> (2 * c)) & 0x03030303u) * k.x + k.y) ^ 0x80808080u; }
// one code word (16 inputs = micro groups j0, j0 + 1) as 16 int8 at dst
inline void ulw_stage_word(uint word, uint mt, uint j0, threadgroup uint4 *dst) {
  const uint2 k0 = ulw_mg(mt, j0), k1 = ulw_mg(mt, j0 + 1);
  *dst = uint4(ulw_bytes(word, 0, k0), ulw_bytes(word, 1, k0), ulw_bytes(word, 2, k1), ulw_bytes(word, 3, k1));
}

// NAX: per macro group two K=32 int8 matmul2d accumulate into one int32 accumulator (|acc| <= 64 x 14 x 127), then
// tot += acc x m (int32, |tot| < 2^31 for G <= 34 macro groups... G x 113,792 x 127); store
// float(tot) x anchor x a_scale[row]. Lane = column of the simdgroup's 32.
template <class Store>
inline void ulw_nax(device int8_t *x8, device const float *a_scale, device const uint4 *q, device const uint *mm,
                    device const float *anch, uint K, uint G, uint N, threadgroup uint *stg, uint lane, uint rows,
                    Store store) {
  constexpr ushort Rows = 8, Cols = 32, KS = 32, CAP = 16;
  threadgroup int8_t *bst = (threadgroup int8_t *)stg;
  threadgroup uint *mtg = stg + KS * Cols / 4;
  auto bt = tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline>(bst, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto at = tensor<device int8_t, dextents<int, 2>, tensor_inline>(x8, dextents<int, 2>{int(K), Rows}, array<int, 2>{1, int(K)});
  constexpr auto dk = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dk, execution_simdgroups<1>> op;
  auto b0 = bt.template slice<KS, Cols>(0, 0);
  auto a0 = at.template slice<KS, Rows>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int>();
  int tot[CAP]; ushort er[CAP], ec[CAP]; bool ev[CAP];
#pragma unroll
  for (ushort i = 0; i < CAP; ++i) {
    tot[i] = 0; er[i] = 0; ec[i] = 0; ev[i] = false;
    if (i < acc.get_capacity() && acc.is_valid_element(i)) { const auto ix = acc.get_multidimensional_index(i); er[i] = ix[1]; ec[i] = ix[0]; ev[i] = true; }
    if (i < acc.get_capacity()) acc[i] = 0;
  }
  const uint Gb = K / 64;
  uint4 w = q[0]; uint mt = mm[0];
  threadgroup uint4 *dst = (threadgroup uint4 *)(bst + lane * KS);
  for (uint g = 0; g < G; ++g) {
    const uint kb = (g < Gb ? g : g - Gb) * 64;
#pragma unroll
    for (ushort h = 0; h < 2; ++h) {
#pragma unroll
      for (ushort t = 0; t < 2; ++t) ulw_stage_word(w[2 * h + t], mt, 2 * (2 * h + t), dst + t);
      if (h == 0) mtg[lane] = (mt >> 1) & 0x7Fu;
      simdgroup_barrier(mem_flags::mem_threadgroup);
      if (h == 1 && g + 1 < G) { w = q[ulong(g + 1) * N]; mt = mm[ulong(g + 1) * N]; }
      auto as = at.template slice<KS, Rows>(int(kb + KS * h), 0);
      op.run(as, b0, acc);
      if (h == 1) {
#pragma unroll
        for (ushort i = 0; i < CAP; ++i) if (i < acc.get_capacity()) {
          if (ev[i]) tot[i] += acc[i] * int(mtg[ec[i]]);
          acc[i] = 0;
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
#pragma unroll
  for (ushort i = 0; i < CAP; ++i)
    if (ev[i] && er[i] < rows) store(uint(er[i]), uint(ec[i]), float(tot[i]) * anch[ec[i]] * a_scale[er[i]]);
}

// int8 register GEMV for tiles of R <= 2 live rows: lane = column; per macro group the lanes convert the R rows' 64
// int8 inputs to half in threadgroup memory (input 4c + b at 4c + {0, 2, 1, 3}[b]); (q - z) as exact half from
// 1024 + q; per micro group half2 FMAs (|partial| <= 1016, exact), x s_j in fp32 (exact), per macro group
// tot += int(sum) x m. The same integer tot as the NAX tile, so bit-identical outputs.
template <ushort R, class Store>
inline void ulw_gemv(device int8_t *x8, device const float *a_scale, device const uint4 *q, device const uint *mm,
                     device const float *anch, uint K, uint G, uint N, threadgroup half *ah, uint lane, Store store) {
  const uint Gb = K / 64;
  int tot[R];
#pragma unroll
  for (ushort r = 0; r < R; ++r) tot[r] = 0;
  const uint i0 = 2 * lane, p0 = (i0 & ~3u) | ((i0 >> 1) & 1u), p1 = p0 + 2;   // inputs 2l, 2l + 1
  threadgroup const half4 *ah4 = (threadgroup const half4 *)ah;
  uint4 w = q[0]; uint mt = mm[0];
  for (uint g = 0; g < G; ++g) {
    const uint kb = (g < Gb ? g : g - Gb) * 64;
#pragma unroll
    for (ushort r = 0; r < R; ++r) {
      const char2 a2 = *(device const char2 *)(x8 + r * K + kb + i0);
      ah[r * 64 + p0] = half(a2.x); ah[r * 64 + p1] = half(a2.y);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float ms[R];
#pragma unroll
    for (ushort r = 0; r < R; ++r) ms[r] = 0.0f;
#pragma unroll
    for (ushort t = 0; t < 4; ++t) {
      const uint word = w[t];
#pragma unroll
      for (ushort jj = 0; jj < 2; ++jj) {
        const uint j = 2 * t + jj;
        const uint s = 2u + 2u * (mt & 1u) + ((mt >> (16 + 2 * j)) & 3u);
        const half2 zz = as_type<half2>(0x64016401u + ((mt >> (8 + j)) & 1u) * 0x00010001u);
        half2 a2[R];
#pragma unroll
        for (ushort r = 0; r < R; ++r) a2[r] = half2(0.0h);
#pragma unroll
        for (ushort cc = 0; cc < 2; ++cc) {
          const uint c = 2 * jj + cc, x = word >> (2 * c);
          const half2 qa = as_type<half2>((x & 0x00030003u) | 0x64006400u) - zz;          // inputs 4c, 4c + 2
          const half2 qb = as_type<half2>(((x >> 8) & 0x00030003u) | 0x64006400u) - zz;   // inputs 4c + 1, 4c + 3
#pragma unroll
          for (ushort r = 0; r < R; ++r) {
            const half4 av = ah4[r * 16 + 4 * t + c];
            a2[r] = fma(qa, av.xy, a2[r]); a2[r] = fma(qb, av.zw, a2[r]);
          }
        }
#pragma unroll
        for (ushort r = 0; r < R; ++r) ms[r] = fma(float(a2[r].x) + float(a2[r].y), float(s), ms[r]);
      }
    }
    const int m = int((mt >> 1) & 0x7Fu);
#pragma unroll
    for (ushort r = 0; r < R; ++r) tot[r] += int(ms[r]) * m;
    if (g + 1 < G) { w = q[ulong(g + 1) * N]; mt = mm[ulong(g + 1) * N]; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort r = 0; r < R; ++r) store(uint(r), lane, float(tot[r]) * anch[lane] * a_scale[r]);
}

// Threadgroup (x, tile) of two simdgroups, 32 columns each: tiles of <= 2 live rows on the GEMV, others on NAX.
kernel void moe_ulmoe_w8(device uchar *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                         device const uint *tile_count [[buffer(2)]], device const uint4 *codes [[buffer(3)]],
                         device const uint *meta [[buffer(4)]], device const float *anchors [[buffer(5)]],
                         device bfloat *output [[buffer(6)]], device uchar *sw00 [[buffer(7)]],
                         device uchar *sw10 [[buffer(8)]], device uchar *smeta0 [[buffer(9)]],
                         device uchar *sw01 [[buffer(10)]], device uchar *sw11 [[buffer(11)]],
                         device uchar *smeta1 [[buffer(12)]], device uchar *shared_input [[buffer(13)]],
                         constant MoeUlmoeParams &p [[buffer(14)]],
                         uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                         uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint SG = 1024 / 4 + 32;   // uints per simdgroup: int8 stage + 32 macro scales
  static_assert(2 * SG * 4 <= kUlmoeStageHalves * 2, "the routed stages fit the shared tile's");
  threadgroup half shared_stage[kUlmoeStageHalves];
  threadgroup uint *stage = (threadgroup uint *)shared_stage;
  if (group.y >= *tile_count) return;
  // Tiles in reverse order: the shared expert's (the last ones), the longest per threadgroup, launch first.
  const uint t = *tile_count - 1 - group.y;
  const MoeTileDescriptor tile = tiles[t];
  if (tile.expert >= p.experts) {
    ulmoe_shared_tile(shared_input, tile, t, sw00, sw10, smeta0, sw01, sw11, smeta1, output, p, group.x, simd_lane,
                      simd_group, shared_stage);
    return;
  }
  const uint K = p.input_size, N = p.output_size, G = K / 64 + p.residual_size / 64;
  device int8_t *x8 = (device int8_t *)(input + ulong(t) * p.input_tile_bytes);
  device const float *a_scale = (device const float *)(x8 + 8 * K);
  const ulong out = ulong(t) * 8 * N;
  const uint origin = group.x * 64 + simd_group * 32;
  const ulong eo = ulong(tile.expert) * G * N + origin;
  const auto st = [&](uint row, uint column, float v) { output[out + ulong(row) * N + origin + column] = bfloat(v); };
  threadgroup uint *my = stage + simd_group * SG;
  if (tile.rows <= 1) {
    ulw_gemv<1>(x8, a_scale, codes + eo + simd_lane, meta + eo + simd_lane, anchors + origin, K, G, N,
                (threadgroup half *)my, simd_lane, st);
    return;
  }
  if (tile.rows <= 2) {
    ulw_gemv<2>(x8, a_scale, codes + eo + simd_lane, meta + eo + simd_lane, anchors + origin, K, G, N,
                (threadgroup half *)my, simd_lane, st);
    return;
  }
  ulw_nax(x8, a_scale, codes + eo + simd_lane, meta + eo + simd_lane, anchors + origin, K, G, N, my, simd_lane,
          tile.rows, st);
}

// ---------------------------------------------------------------- bf16 path (reference)
// Register GEMV over the tile's live rows (up to 8), lane = column, bf16 inputs widened to fp32 in threadgroup
// memory per macro group: per micro group j, sum_i (q_i - z_j) s_j a_i in fp32, per macro group tot += sum x m, then
// tot x anchor. Bounded, not bit-reproducible against the int8 path.
kernel void moe_ulmoe_bf16(device uchar *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                           device const uint *tile_count [[buffer(2)]], device const uint4 *codes [[buffer(3)]],
                           device const uint *meta [[buffer(4)]], device const float *anchors [[buffer(5)]],
                           device bfloat *output [[buffer(6)]], device uchar *sw00 [[buffer(7)]],
                           device uchar *sw10 [[buffer(8)]], device uchar *smeta0 [[buffer(9)]],
                           device uchar *sw01 [[buffer(10)]], device uchar *sw11 [[buffer(11)]],
                           device uchar *smeta1 [[buffer(12)]], device uchar *shared_input [[buffer(13)]],
                           constant MoeUlmoeParams &p [[buffer(14)]],
                           uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]]) {
  static_assert(2 * 8 * 64 * 4 <= kUlmoeStageHalves * 2, "the routed stages fit the shared tile's");
  threadgroup half shared_stage[kUlmoeStageHalves];
  threadgroup float (*stage)[8 * 64] = (threadgroup float (*)[8 * 64])shared_stage;
  if (group.y >= *tile_count) return;
  // Tiles in reverse order: the shared expert's (the last ones), the longest per threadgroup, launch first.
  const uint t = *tile_count - 1 - group.y;
  const MoeTileDescriptor tile = tiles[t];
  if (tile.expert >= p.experts) {
    ulmoe_shared_tile(shared_input, tile, t, sw00, sw10, smeta0, sw01, sw11, smeta1, output, p, group.x, simd_lane,
                      simd_group, shared_stage);
    return;
  }
  const uint K = p.input_size, N = p.output_size, G = K / 64 + p.residual_size / 64, Gb = K / 64;
  const uint rows = min(tile.rows, 8u);
  device const bfloat *x = (device const bfloat *)(input + ulong(t) * p.input_tile_bytes);
  const uint column = group.x * 64 + simd_group * 32 + simd_lane;
  device const uint4 *q = codes + ulong(tile.expert) * G * N + column;
  device const uint *mm = meta + ulong(tile.expert) * G * N + column;
  threadgroup float *a = stage[simd_group];
  float tot[8];
  for (uint r = 0; r < 8; ++r) tot[r] = 0.0f;
  for (uint g = 0; g < G; ++g) {
    const uint kb = (g < Gb ? g : g - Gb) * 64;
    for (uint r = 0; r < rows; ++r) {
      a[r * 64 + 2 * simd_lane] = float(x[ulong(r) * K + kb + 2 * simd_lane]);
      a[r * 64 + 2 * simd_lane + 1] = float(x[ulong(r) * K + kb + 2 * simd_lane + 1]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint4 w = q[ulong(g) * N];
    const uint mt = mm[ulong(g) * N];
    float sum[8];
    for (uint r = 0; r < 8; ++r) sum[r] = 0.0f;
    for (uint t = 0; t < 4; ++t) {
      for (uint jj = 0; jj < 2; ++jj) {
        const uint j = 2 * t + jj;
        const float s = float(2u + 2u * (mt & 1u) + ((mt >> (16 + 2 * j)) & 3u));
        const int z = int((mt >> (8 + j)) & 1u) + 1;
        float wv[8];
        for (uint i = 0; i < 8; ++i) {
          const uint k = 8 * jj + i;                       // input 16 t + k of the macro group
          const uint c = k >> 2, b = k & 3u;              // restaged: input 4c + b at bit 8b + 2c
          wv[i] = float(int((w[t] >> (8 * b + 2 * c)) & 3u) - z);
        }
        for (uint r = 0; r < rows; ++r) {
          float part = 0.0f;
          for (uint i = 0; i < 8; ++i) part = fma(wv[i], a[r * 64 + 16 * t + 8 * jj + i], part);
          sum[r] = fma(part, s, sum[r]);
        }
      }
    }
    const float m = float((mt >> 1) & 0x7Fu);
    for (uint r = 0; r < rows; ++r) tot[r] = fma(sum[r], m, tot[r]);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  const ulong out = ulong(t) * 8 * N + column;
  for (uint r = 0; r < rows; ++r) output[out + ulong(r) * N] = bfloat(tot[r] * anchors[column]);
}

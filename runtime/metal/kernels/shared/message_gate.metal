#pragma clang fp reassociate(off)
#include "metal/abi/KernelABI.h"
#include "metal/abi/MessageGate.h"
#include "metal/kernels/common/rms_inverse.h"
#include "metal/kernels/common/sgmatrix.h"
#include "metal/kernels/common/split_reduce.h"

#include <metal_stdlib>

using namespace metal;

// The fused message gates (metal/abi/MessageGate.h) over fp32 rows: a decode
// step's of MESSAGE_GATE_THREADS threads a threadgroup, a prompt chunk's of
// MESSAGE_GATE_TILE_THREADS (message_gate_rows below). Their reference path,
// which rounds where they do, runs in the tests alone
// (dev/tests/engine/message_gate_reference.h).
constant constexpr uint kGateThreads = MESSAGE_GATE_THREADS;
constant constexpr uint kGateSimdgroups = kGateThreads / 32;

// ---- The fused gates (metal/abi/MessageGate.h): a decode step's rows --------
//
// Each kernel runs in phases, functions out of line, which hand over through
// threadgroup memory alone: every sum is reduced by simd_sum into its own slot
// a simdgroup and then by one thread over the simdgroups in order; loops over
// a group's rows run to MESSAGE_GATE_GROUP_ROWS, the rows past the group's
// contributing zeros; no thread keeps an array across a barrier (as
// block_residual_mix.metal: an array indexed by a runtime bound and alive
// across a barrier lost one simdgroup's values now and then under shader
// validation). The sums are those of message_gate_test_prepare, the float
// projections' and message_gate_test_norm's, in their order within a thread and
// over the simdgroups. A dispatch runs MessageGateFusedParams::gates gates
// (grid z), gate g reading its weights, norm and output at slot g of their
// kernel's MESSAGE_GATE_MAXIMUM_GATES bindings and its rows of the scratch
// after gate g - 1's: rank values [gates][rows][rank], gated rows
// [gates][rows][hidden], partial sums [gates][groups][slices][rows of a group]
// and counters [gates][groups]. Weights: the gates' float rows as stored, F32
// or BF16 (the _f32 and _bf16 instances), each value widened to fp32.
constant constexpr uint kGroupRows = MESSAGE_GATE_GROUP_ROWS;
constant constexpr uint kDownOutputs = MESSAGE_GATE_DOWN_OUTPUTS;

// Quad i of a row of weights, in fp32.
inline float4 message_gate_quad(device const float *row, uint i) {
  return reinterpret_cast<device const float4 *>(row)[i];
}
inline float4 message_gate_quad(device const bfloat *row, uint i) {
  return float4(reinterpret_cast<device const bfloat4 *>(row)[i]);
}

// Gate g's binding of three.
template <typename T> inline T message_gate_pick(uint g, T first, T second, T third) {
  return g == 0 ? first : g == 1 ? second : third;
}

// What message_gate_down's phases share: a partial a simdgroup of each sum
// (each row's and each message's squares, then each output's rank value of
// each row), and each row's and message's inverse RMS (zero past the group's
// rows).
struct MessageGateDownShared {
  float partials[(2 + kDownOutputs) * kGroupRows * kGateSimdgroups];
  float scales[2 * kGroupRows];
};

// A group of `count` rows from row `first`: gate g's row r's x, or its
// message, the group's last row standing in for the rows past it (their
// scales are zero).
inline device const float *message_gate_row(device const float *x, device const float *message,
                                            constant MessageGateFusedParams &p, uint g, uint first, uint count,
                                            uint r, bool own) {
  const ulong row = first + min(r, count - 1);
  return own ? x + row * p.x_stride + p.x_columns[g] : message + row * p.message_stride + p.message_columns[g];
}

// Phase one: each row's and each message's inverse RMS, from quads: a
// thread's quads tid + MESSAGE_GATE_THREADS i in order, each quad's squares
// added in order (a sum of the same terms as message_gate_test_prepare's in
// another order: it reads a value a thread at a time).
__attribute__((noinline)) void message_gate_down_scales(device const float *x, device const float *message,
                                                        constant MessageGateFusedParams &p, uint g, uint first,
                                                        uint count, uint tid, uint lane, uint sg,
                                                        threadgroup MessageGateDownShared &shared) {
  const uint quads = p.hidden / 4;
  for (uint i = 0; i < 2 * kGroupRows; ++i) {
    device const float4 *row =
        reinterpret_cast<device const float4 *>(message_gate_row(x, message, p, g, first, count, i / 2, i % 2 == 0));
    float squares = 0.0f;
    if (i / 2 < count)
      for (uint d = tid; d < quads; d += kGateThreads) {
        const float4 v = row[d];
        squares += v.x * v.x;
        squares += v.y * v.y;
        squares += v.z * v.z;
        squares += v.w * v.w;
      }
    squares = simd_sum(squares);
    if (lane == 0) shared.partials[i * kGateSimdgroups + sg] = squares;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 2 * kGroupRows) {
    float total = 0.0f;
    for (uint s = 0; s < kGateSimdgroups; ++s) total += shared.partials[tid * kGateSimdgroups + s];
    shared.scales[tid] = tid / 2 < count ? rsqrt(total / p.hidden + kRmsEpsilon) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Phase two: the group's rank values k0 .. k0 + kDownOutputs - 1, each the
// dot of the weights' row (`down`, [rank][2 hidden]) with the normalized
// concatenation, rounded to bf16 as message_gate_test_prepare rounds it, the
// threads striding its quads; rounded to bf16 into `low` [rows][rank], the
// gate's.
template <typename Weight>
__attribute__((noinline)) void message_gate_down_values(device const float *x, device const float *message,
                                                        device const Weight *down, device bfloat *low,
                                                        constant MessageGateFusedParams &p, uint g, uint first,
                                                        uint count, uint k0, uint tid, uint lane, uint sg,
                                                        threadgroup MessageGateDownShared &shared) {
  const uint quarter = p.hidden / 4;
  float sums[kDownOutputs * kGroupRows];
  for (uint j = 0; j < kDownOutputs * kGroupRows; ++j) sums[j] = 0.0f;
  for (uint i = tid; i < 2 * quarter; i += kGateThreads) {
    float4 w[kDownOutputs];
    for (uint o = 0; o < kDownOutputs; ++o)
      w[o] = k0 + o < p.rank ? message_gate_quad(down + ulong(k0 + o) * 2 * p.hidden, i) : float4(0.0f);
    const bool own = i < quarter;
    for (uint r = 0; r < kGroupRows; ++r) {
      device const float4 *row =
          reinterpret_cast<device const float4 *>(message_gate_row(x, message, p, g, first, count, r, own));
      const float4 normalized = float4(bfloat4(row[own ? i : i - quarter] * shared.scales[2 * r + (own ? 0 : 1)]));
      for (uint o = 0; o < kDownOutputs; ++o) sums[o * kGroupRows + r] += dot(w[o], normalized);
    }
  }
  for (uint j = 0; j < kDownOutputs * kGroupRows; ++j) {
    const float sum = simd_sum(sums[j]);
    if (lane == 0) shared.partials[(2 * kGroupRows + j) * kGateSimdgroups + sg] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < kDownOutputs * kGroupRows) {
    float total = 0.0f;
    for (uint s = 0; s < kGateSimdgroups; ++s)
      total += shared.partials[(2 * kGroupRows + tid) * kGateSimdgroups + s];
    const uint o = tid / kGroupRows, r = tid % kGroupRows;
    if (r < count && k0 + o < p.rank) low[ulong(first + r) * p.rank + k0 + o] = bfloat(total);
  }
}

// The rank values of a group of MESSAGE_GATE_GROUP_ROWS rows of each gate, of
// MESSAGE_GATE_DOWN_OUTPUTS of them a threadgroup, each output's weights' row
// read once for every row of the group. Grid: (rank /
// MESSAGE_GATE_DOWN_OUTPUTS, groups, gates).
template <typename Weight>
kernel void message_gate_down(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                              device const Weight *down0 [[buffer(2)]], device const Weight *down1 [[buffer(3)]],
                              device const Weight *down2 [[buffer(4)]], device bfloat *low [[buffer(5)]],
                              constant MessageGateFusedParams &p [[buffer(6)]],
                              uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MessageGateDownShared shared;
  const uint first = group.y * kGroupRows, g = group.z;
  if (first >= p.rows || g >= p.gates) return;
  const uint count = min(kGroupRows, p.rows - first);
  message_gate_down_scales(x, message, p, g, first, count, tid, lane, sg, shared);
  message_gate_down_values(x, message, message_gate_pick(g, down0, down1, down2), low + ulong(g) * p.rows * p.rank,
                           p, g, first, count, group.x * kDownOutputs, tid, lane, sg, shared);
}

// What message_gate_apply's phases share: the group's rank values (zero past
// its rows), a partial a simdgroup of each row's squares, the arrival of
// split_arrive_last, and the last slice's inverse RMS of each row.
struct MessageGateApplyShared {
  float4 lows[kGroupRows * MESSAGE_GATE_MAXIMUM_RANK / 4];
  float partials[kGroupRows * kGateSimdgroups];
  uint arrival;
  float inverses[kGroupRows];
};

// Phase one: value d of each row of the group, a thread each: forget and gain
// from the rank values and the weights' rows ([hidden][rank]), the gated
// value in fp32 (message_gated's) to `values` [rows][hidden], and the slice's
// squares of each row, by thread 0 to its slot of `partials`
// [slice][MESSAGE_GATE_GROUP_ROWS] (the group's).
template <typename Weight>
__attribute__((noinline)) void message_gate_apply_values(device const float *x, device const float *message,
                                                         device const Weight *forget, device const Weight *gain,
                                                         device coherent(device) float *values,
                                                         device coherent(device) float *partials,
                                                         constant MessageGateFusedParams &p, uint g, uint first,
                                                         uint count, uint slice, uint tid, uint lane, uint sg,
                                                         threadgroup MessageGateApplyShared &shared) {
  const uint d = slice * kGateThreads + tid, quarters = p.rank / 4;
  const bool active = d < p.hidden;
  float f[kGroupRows], e[kGroupRows];
  for (uint r = 0; r < kGroupRows; ++r) f[r] = e[r] = 0.0f;
  if (active) {
    device const Weight *f4 = forget + ulong(d) * p.rank, *e4 = gain + ulong(d) * p.rank;
    for (uint j = 0; j < quarters; ++j) {
      const float4 a = message_gate_quad(f4, j), b = message_gate_quad(e4, j);
      for (uint r = 0; r < kGroupRows; ++r) {
        const float4 l = shared.lows[r * quarters + j];
        f[r] += dot(a, l);
        e[r] += dot(b, l);
      }
    }
  }
  for (uint r = 0; r < kGroupRows; ++r) {
    float squares = 0.0f;
    if (active && r < count) {
      const float xv = x[ulong(first + r) * p.x_stride + p.x_columns[g] + d];
      const float mv = message[ulong(first + r) * p.message_stride + p.message_columns[g] + d];
      const float v = (1.0f - precise::tanh(f[r])) * xv + precise::tanh(e[r]) * mv;
      values[ulong(first + r) * p.hidden + d] = v;
      squares = v * v;
    }
    squares = simd_sum(squares);
    if (lane == 0) shared.partials[r * kGateSimdgroups + sg] = squares;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0)
    for (uint r = 0; r < count; ++r) {
      float total = 0.0f;
      for (uint s = 0; s < kGateSimdgroups; ++s) total += shared.partials[r * kGateSimdgroups + s];
      partials[slice * kGroupRows + r] = total;
    }
}

// Phase two, by the slice that arrives last: each row's squares over the
// slices in slice order, by a thread a row, then the group's rows through the
// weighted RMS norm into bf16, as message_gate_test_norm writes each value, the
// threads striding the rows' quads.
__attribute__((noinline)) void message_gate_apply_norm(device coherent(device) float *values,
                                                       device coherent(device) float *partials,
                                                       device const float *norm, device bfloat *output,
                                                       constant MessageGateFusedParams &p, uint first, uint count,
                                                       uint slices, uint tid,
                                                       threadgroup MessageGateApplyShared &shared) {
  if (tid < count) {
    float total = 0.0f;
    for (uint s = 0; s < slices; ++s) total += partials[s * kGroupRows + tid];
    shared.inverses[tid] = rsqrt(total / p.hidden + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint quads = p.hidden / 4;
  device coherent(device) float4 *rows =
      reinterpret_cast<device coherent(device) float4 *>(values + ulong(first) * p.hidden);
  device bfloat4 *out = reinterpret_cast<device bfloat4 *>(output + ulong(first) * p.hidden);
  for (uint q = tid; q < count * quads; q += kGateThreads) {
    const uint r = q / quads, e = 4 * (q % quads);
    const float4 weights = float4(norm[e], norm[e + 1], norm[e + 2], norm[e + 3]);
    out[q] = bfloat4((rows[q] * shared.inverses[r]) * weights);
  }
}

// The gated rows of a group of each gate from its rank values (`low`): a
// slice of MESSAGE_GATE_THREADS values a threadgroup, its forget and gain rows
// read once for the group's rows; the slice that arrives last
// (kernels/common/split_reduce.h, the group's counter, zero before and after)
// writes the group's normalized rows. Grid: (slices, groups, gates).
template <typename Weight>
kernel void message_gate_apply(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                               device const Weight *forget0 [[buffer(2)]], device const Weight *forget1 [[buffer(3)]],
                               device const Weight *forget2 [[buffer(4)]], device const Weight *gain0 [[buffer(5)]],
                               device const Weight *gain1 [[buffer(6)]], device const Weight *gain2 [[buffer(7)]],
                               device const bfloat *low [[buffer(8)]], device const float *norm0 [[buffer(9)]],
                               device const float *norm1 [[buffer(10)]], device const float *norm2 [[buffer(11)]],
                               device coherent(device) float *values [[buffer(12)]],
                               device coherent(device) float *partials [[buffer(13)]],
                               device atomic_uint *counters [[buffer(14)]], device bfloat *output0 [[buffer(15)]],
                               device bfloat *output1 [[buffer(16)]], device bfloat *output2 [[buffer(17)]],
                               constant MessageGateFusedParams &p [[buffer(18)]],
                               uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MessageGateApplyShared shared;
  const uint first = group.y * kGroupRows, g = group.z;
  if (first >= p.rows || g >= p.gates) return;
  const uint count = min(kGroupRows, p.rows - first), slices = (p.hidden + kGateThreads - 1) / kGateThreads;
  const uint groups = (p.rows + kGroupRows - 1) / kGroupRows;
  device const bfloat *gateLow = low + ulong(g) * p.rows * p.rank;
  threadgroup float *lows = reinterpret_cast<threadgroup float *>(shared.lows);
  for (uint i = tid; i < kGroupRows * p.rank; i += kGateThreads)
    lows[i] = i < count * p.rank ? float(gateLow[ulong(first) * p.rank + i]) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device coherent(device) float *gateValues = values + ulong(g) * p.rows * p.hidden;
  device coherent(device) float *groupPartials = partials + (ulong(g) * groups + group.y) * slices * kGroupRows;
  device atomic_uint *counter = counters + g * groups + group.y;
  message_gate_apply_values(x, message, message_gate_pick(g, forget0, forget1, forget2),
                            message_gate_pick(g, gain0, gain1, gain2), gateValues, groupPartials, p, g, first, count,
                            group.x, tid, lane, sg, shared);
  if (!split_arrive_last(counter, slices, tid, &shared.arrival)) return;
  message_gate_apply_norm(gateValues, groupPartials, message_gate_pick(g, norm0, norm1, norm2),
                          message_gate_pick(g, output0, output1, output2), p, first, count, slices, tid, shared);
  split_release(counter, tid);
}

// ---- The fused gates of a prompt chunk's rows (metal/abi/MessageGate.h) ----
//
// message_gate_rows runs a gate of MESSAGE_GATE_TILE_ROWS rows a threadgroup
// from its rows to its output, in four phases that hand over through
// threadgroup memory and, for the gated rows, the threadgroup's own rows of
// device scratch: each row's and message's inverse RMS; the rank values, the
// down projection of the normalized concatenation, each normalized value
// rounded to bf16 as message_gate_test_prepare rounds it, each rank value rounded
// to bf16 as the float projection writes it; forget, gain and the gated row in
// fp32, a 64-value block of the rows after another, into the gated rows'
// scratch, with each row's squares; and the gated rows through the weighted
// RMS norm into bf16, as message_gate_test_norm writes each value. The products
// run on simdgroup float 8x8 MMA with exact operands (bf16 values widened, the
// weights as stored, BF16 ones widened, as shared/gguf_float.metal's), so a
// sum differs from the float projections' only by its fp32 order. Simdgroup s
// takes columns [8 s, 8 s + 8) of a 64-column block of either projection for
// the tile's 32 rows (four 8-row fragments), so the tile reads each weight
// once; the down projection's normalized inputs of a 64-input step, which
// every simdgroup multiplies, are staged in threadgroup memory, two steps'
// stages alternating. MMA j of a 64-input step multiplies inputs 8 i + j over
// its index i (gguf_float's order): lane (fm, fn) reads a row's inputs
// [8 fn, 8 fn + 16) and a weights' row's [8 fm, 8 fm + 8) of the step, whole
// runs of 8. Rows past the chunk read its last row with scales of zero and
// store nothing.
constant constexpr uint kTileRows = MESSAGE_GATE_TILE_ROWS;
constant constexpr uint kTileSimdgroups = MESSAGE_GATE_TILE_THREADS / 32;
constant constexpr uint kTileStep = 64;
constant constexpr uint kTileFragments = kTileRows / 8;
// A staged row of a step's 64 inputs, or of the rank values, bf16 values
// padded by 8 so that the fragments' rows start in other banks.
constant constexpr uint kTileStride = kTileStep + 8;
static_assert(kTileSimdgroups * 8 == kTileStep, "a tile's simdgroups take a block's 64 columns, 8 each");
static_assert(MESSAGE_GATE_MAXIMUM_RANK == kTileStep, "a rank's values are one step of the forget and gain sums");
static_assert(kTileRows * kTileStep / MESSAGE_GATE_TILE_THREADS == 8, "a thread stages a run of 8 inputs a step");
static_assert(kTileFragments == 4, "a lane's partial squares are a float4, a fragment each");

// Eight consecutive weights of a row from input k, widened to fp32: two
// quads, zero where `inside` is false.
struct MessageGateEight {
  float4 low, high;
  float operator[](uint i) const { return i < 4 ? low[i] : high[i - 4]; }
};
inline MessageGateEight message_gate_eight(device const float *row, uint k, bool inside) {
  device const float4 *quads = reinterpret_cast<device const float4 *>(row + k);
  return inside ? MessageGateEight{quads[0], quads[1]} : MessageGateEight{float4(0.0f), float4(0.0f)};
}
inline MessageGateEight message_gate_eight(device const bfloat *row, uint k, bool inside) {
  device const bfloat4 *quads = reinterpret_cast<device const bfloat4 *>(row + k);
  return inside ? MessageGateEight{float4(quads[0]), float4(quads[1])} : MessageGateEight{float4(0.0f), float4(0.0f)};
}

// What message_gate_rows' phases share: each row's and message's inverse RMS
// (zero past the chunk's rows); the down projection's two stages of a step's
// normalized inputs [row][kTileStride], bf16 as rounded, whose memory the rank
// values, bf16 as rounded [row][kTileStride] (zero past the rank), and a
// partial of each fragment's rows' squares a lane then take; and the rows'
// inverse RMS of the gated rows.
struct MessageGateRowsShared {
  float scales[2 * kTileRows];
  union {
    bfloat stages[2][kTileRows * kTileStride];
    struct {
      bfloat lows[kTileRows * kTileStride];
      float4 partials[kTileSimdgroups * 32];
    } gated;
  };
  float inverses[kTileRows];
};

// Phase one: vector i of the tile, row i / 2's x (i even) or message, a
// simdgroup a vector: a lane's quads lane + 32 j in order, each quad's squares
// added in order, then simd_sum.
__attribute__((noinline)) void message_gate_rows_scales(device const float *x, device const float *message,
                                                        constant MessageGateFusedParams &p, uint g, uint first,
                                                        uint count, uint lane, uint sg,
                                                        threadgroup MessageGateRowsShared &shared) {
  const uint quads = p.hidden / 4;
  for (uint i = sg; i < 2 * kTileRows; i += kTileSimdgroups) {
    device const float4 *row =
        reinterpret_cast<device const float4 *>(message_gate_row(x, message, p, g, first, count, i / 2, i % 2 == 0));
    float squares = 0.0f;
    if (i / 2 < count)
      for (uint d = lane; d < quads; d += 32) {
        const float4 v = row[d];
        squares += v.x * v.x;
        squares += v.y * v.y;
        squares += v.z * v.z;
        squares += v.w * v.w;
      }
    squares = simd_sum(squares);
    if (lane == 0) shared.scales[i] = i / 2 < count ? rsqrt(squares / p.hidden + kRmsEpsilon) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A thread's run of 8 of the concatenation's step `step` (x's half and then
// the message's): its row tid / 8's inputs from 8 (tid % 8), as read.
struct MessageGateRun {
  float4 low, high;
};
inline MessageGateRun message_gate_rows_load(device const float *x, device const float *message,
                                             constant MessageGateFusedParams &p, uint g, uint first, uint count,
                                             uint step, uint steps, uint tid) {
  const bool own = step < steps;
  device const float4 *run = reinterpret_cast<device const float4 *>(
      message_gate_row(x, message, p, g, first, count, tid / 8, own) + (own ? step : step - steps) * kTileStep +
      8 * (tid % 8));
  return {run[0], run[1]};
}
// The run normalized by its row's scale and rounded to bf16, into stage
// `stage`.
inline void message_gate_rows_stage(MessageGateRun run, bool own, uint tid, threadgroup bfloat *stage,
                                    threadgroup const float *scales) {
  const uint r = tid / 8;
  const float scale = scales[2 * r + (own ? 0 : 1)];
  threadgroup bfloat4 *out = reinterpret_cast<threadgroup bfloat4 *>(stage + r * kTileStride + 8 * (tid % 8));
  out[0] = bfloat4(run.low * scale);
  out[1] = bfloat4(run.high * scale);
}
// A lane's 16 staged values of a fragment's row from 8 fn, widened: the first
// 8 and the next 8.
inline void message_gate_rows_read(threadgroup const bfloat *row, thread MessageGateEight &low,
                                   thread MessageGateEight &high) {
  threadgroup const bfloat4 *a = reinterpret_cast<threadgroup const bfloat4 *>(row);
  low = {float4(a[0]), float4(a[1])};
  high = {float4(a[2]), float4(a[3])};
}

// A half's chains of a lane's four fragments: each fragment's even and odd
// MMAs apart, in fields, so that no array lives across phase two's barriers.
struct MessageGateRowsChains {
  float2 even0, odd0, even1, odd1, even2, odd2, even3, odd3;
};
// A fragment's MMAs of a step, from its row of the stage: MMA j, j = 0 to 7 in
// order, into `even` for even j and into `odd` for odd j.
inline void message_gate_rows_mma(threadgroup const bfloat *row, MessageGateEight b0, MessageGateEight b1,
                                  thread float2 &even, thread float2 &odd) {
  MessageGateEight low, high;
  message_gate_rows_read(row, low, high);
  for (uint j = 0; j < 8; j += 2) {
    sgmatrix::mma_acc<float>(even, float2(low[j], high[j]), float2(b0[j], b1[j]));
    sgmatrix::mma_acc<float>(odd, float2(low[j + 1], high[j + 1]), float2(b0[j + 1], b1[j + 1]));
  }
}

// Phase two: the tile's rank values, the down weights' rows ([rank][2
// hidden]) against the normalized concatenation, x's half and then the
// message's, a step of 64 inputs at a time, simdgroup s rank values
// [8 s, 8 s + 8), into `shared.gated.lows`. Each half's even and odd MMAs
// accumulate apart (four chains of 1,152 products at Inkling-2-Mini's 4,608
// inputs, as gguf_float splits its own), added at the end in order. The
// chains live across every step's barrier, so they are fields, not an array
// indexed at run time (which the decode kernels above keep none of across a
// barrier, and which the AIR held in memory here even with the loops over
// them marked to unroll): each half's steps run in a loop of their own,
// which takes that half's chains.
template <typename Weight>
__attribute__((noinline)) void message_gate_rows_down(device const float *x, device const float *message,
                                                      device const Weight *down, constant MessageGateFusedParams &p,
                                                      uint g, uint first, uint count, uint tid, uint lane, uint sg,
                                                      threadgroup MessageGateRowsShared &shared) {
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint steps = p.hidden / kTileStep, column = 8 * sg + l.fn;
  device const Weight *w0 = down + ulong(min(column, p.rank - 1)) * 2 * p.hidden,
                      *w1 = down + ulong(min(column + 1, p.rank - 1)) * 2 * p.hidden;
  // x's half's chains (own) and the message's (other).
  MessageGateRowsChains own{}, other{};
  message_gate_rows_stage(message_gate_rows_load(x, message, p, g, first, count, 0, steps, tid), true, tid,
                          shared.stages[0], shared.scales);
  // The weights load a step ahead.
  MessageGateEight next0 = message_gate_eight(w0, 8 * l.fm, column < p.rank),
                   next1 = message_gate_eight(w1, 8 * l.fm, column + 1 < p.rank);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Step `step` into its half's chains, the four fragments' in order; the next
  // step's inputs staged and its weights loaded.
  const auto run = [&](uint step, thread MessageGateRowsChains &chains) __attribute__((always_inline)) {
    const bool next = step + 1 < 2 * steps;
    const MessageGateRun ahead =
        next ? message_gate_rows_load(x, message, p, g, first, count, step + 1, steps, tid) : MessageGateRun{};
    const MessageGateEight b0 = next0, b1 = next1;
    if (next) {
      next0 = message_gate_eight(w0, (step + 1) * kTileStep + 8 * l.fm, column < p.rank);
      next1 = message_gate_eight(w1, (step + 1) * kTileStep + 8 * l.fm, column + 1 < p.rank);
    }
    threadgroup const bfloat *row = shared.stages[step % 2] + l.fm * kTileStride + 8 * l.fn;
    message_gate_rows_mma(row, b0, b1, chains.even0, chains.odd0);
    message_gate_rows_mma(row + 8 * kTileStride, b0, b1, chains.even1, chains.odd1);
    message_gate_rows_mma(row + 16 * kTileStride, b0, b1, chains.even2, chains.odd2);
    message_gate_rows_mma(row + 24 * kTileStride, b0, b1, chains.even3, chains.odd3);
    if (next) message_gate_rows_stage(ahead, step + 1 < steps, tid, shared.stages[(step + 1) % 2], shared.scales);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  };
  for (uint step = 0; step < steps; ++step) run(step, own);
  for (uint step = steps; step < 2 * steps; ++step) run(step, other);
  // Fragment f's rank values: (x's even + odd) + (the message's even + odd).
  const auto store = [&](uint f, float2 sum) {
    const uint r = 8 * f + l.fm;
    shared.gated.lows[r * kTileStride + column] = bfloat(sum.x);
    shared.gated.lows[r * kTileStride + column + 1] = bfloat(sum.y);
  };
  store(0, (own.even0 + own.odd0) + (other.even0 + other.odd0));
  store(1, (own.even1 + own.odd1) + (other.even1 + other.odd1));
  store(2, (own.even2 + own.odd2) + (other.even2 + other.odd2));
  store(3, (own.even3 + own.odd3) + (other.even3 + other.odd3));
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Phase three: forget and gain of a block of 64 values of the tile's rows
// after another, simdgroup s values [8 s, 8 s + 8) of the block, the weights'
// rows ([hidden][rank]) against the rank values, the gated value
// (message_gated's) into `gated` [rows][hidden], and each row's squares, a
// lane's in block order, into `shared.gated.partials`.
template <typename Weight>
__attribute__((noinline)) void message_gate_rows_gated(device const float *x, device const float *message,
                                                       device const Weight *forget, device const Weight *gain,
                                                       device float *gated, constant MessageGateFusedParams &p,
                                                       uint g, uint first, uint count, uint lane, uint sg,
                                                       threadgroup MessageGateRowsShared &shared) {
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint blocks = p.hidden / kTileStep;
  const bool inside = 8 * l.fm < p.rank;
  const uint k = inside ? 8 * l.fm : 0;
  float squares[kTileFragments];
  for (uint f = 0; f < kTileFragments; ++f) squares[f] = 0.0f;
  // The weights load a block ahead.
  const auto load = [&](uint block, thread MessageGateEight(&w)[4]) {
    const uint column = block * kTileStep + 8 * sg + l.fn;
    w[0] = message_gate_eight(forget + ulong(column) * p.rank, k, inside);
    w[1] = message_gate_eight(forget + ulong(column + 1) * p.rank, k, inside);
    w[2] = message_gate_eight(gain + ulong(column) * p.rank, k, inside);
    w[3] = message_gate_eight(gain + ulong(column + 1) * p.rank, k, inside);
  };
  MessageGateEight next[4];
  load(0, next);
  for (uint block = 0; block < blocks; ++block) {
    const uint column = block * kTileStep + 8 * sg + l.fn;
    const MessageGateEight f0 = next[0], f1 = next[1], e0 = next[2], e1 = next[3];
    if (block + 1 < blocks) load(block + 1, next);
    // The block's x and message values of the lane's rows, read before the
    // products (a row past the chunk's reads its last).
    float2 xs[kTileFragments], ms[kTileFragments];
    for (uint f = 0; f < kTileFragments; ++f) {
      const ulong row = first + min(8 * f + l.fm, count - 1);
      xs[f] = *reinterpret_cast<device const float2 *>(x + row * p.x_stride + p.x_columns[g] + column);
      ms[f] = *reinterpret_cast<device const float2 *>(message + row * p.message_stride + p.message_columns[g] + column);
    }
    // Every fragment's MMAs run in every lane, the rows past the chunk's too:
    // a lane's row decides only its stores.
    float2 forgets[kTileFragments], gains[kTileFragments];
    for (uint f = 0; f < kTileFragments; ++f) {
      MessageGateEight low, high;
      message_gate_rows_read(shared.gated.lows + (8 * f + l.fm) * kTileStride + 8 * l.fn, low, high);
      forgets[f] = gains[f] = float2(0.0f);
      for (uint j = 0; j < 8; ++j) {
        sgmatrix::mma_acc<float>(forgets[f], float2(low[j], high[j]), float2(f0[j], f1[j]));
        sgmatrix::mma_acc<float>(gains[f], float2(low[j], high[j]), float2(e0[j], e1[j]));
      }
    }
    for (uint f = 0; f < kTileFragments; ++f) {
      const uint r = 8 * f + l.fm;
      if (r < count) {
        const ulong row = first + r;
        const float2 xv = xs[f], mv = ms[f];
        float2 v;
        v.x = (1.0f - precise::tanh(forgets[f].x)) * xv.x + precise::tanh(gains[f].x) * mv.x;
        v.y = (1.0f - precise::tanh(forgets[f].y)) * xv.y + precise::tanh(gains[f].y) * mv.y;
        *reinterpret_cast<device float2 *>(gated + row * p.hidden + column) = v;
        squares[f] += v.x * v.x;
        squares[f] += v.y * v.y;
      }
    }
  }
  shared.gated.partials[sg * 32 + lane] = float4(squares[0], squares[1], squares[2], squares[3]);
}

// Phase four: each row's squares, by a thread a row over the simdgroups and
// their lanes of its row in order, then the tile's gated rows through the
// weighted RMS norm into bf16, the threads striding the rows' quads.
__attribute__((noinline)) void message_gate_rows_norm(device const float *gated, device const float *norm,
                                                      device bfloat *output, constant MessageGateFusedParams &p,
                                                      uint first, uint count, uint tid,
                                                      threadgroup MessageGateRowsShared &shared) {
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid < count) {
    const uint f = tid / 8, fm = tid % 8;
    float total = 0.0f;
    for (uint s = 0; s < kTileSimdgroups; ++s)
      for (uint lane = 0; lane < 32; ++lane)
        if (sgmatrix::lane_map(lane).fm == fm) total += shared.gated.partials[s * 32 + lane][f];
    shared.inverses[tid] = rsqrt(total / p.hidden + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint quads = p.hidden / 4;
  device const float4 *rows = reinterpret_cast<device const float4 *>(gated + ulong(first) * p.hidden);
  device bfloat4 *out = reinterpret_cast<device bfloat4 *>(output + ulong(first) * p.hidden);
  for (uint q = tid; q < count * quads; q += MESSAGE_GATE_TILE_THREADS) {
    const uint r = q / quads, e = 4 * (q % quads);
    const float4 weights = float4(norm[e], norm[e + 1], norm[e + 2], norm[e + 3]);
    out[q] = bfloat4((rows[q] * shared.inverses[r]) * weights);
  }
}

// A gate's gated rows of a tile of MESSAGE_GATE_TILE_ROWS rows a threadgroup,
// gate g's gated rows' scratch `gated` g. Grid: (tiles, 1, gates).
template <typename Weight>
kernel void message_gate_rows(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                              device const Weight *down0 [[buffer(2)]], device const Weight *down1 [[buffer(3)]],
                              device const Weight *down2 [[buffer(4)]], device const Weight *forget0 [[buffer(5)]],
                              device const Weight *forget1 [[buffer(6)]], device const Weight *forget2 [[buffer(7)]],
                              device const Weight *gain0 [[buffer(8)]], device const Weight *gain1 [[buffer(9)]],
                              device const Weight *gain2 [[buffer(10)]], device const float *norm0 [[buffer(11)]],
                              device const float *norm1 [[buffer(12)]], device const float *norm2 [[buffer(13)]],
                              device float *gated0 [[buffer(14)]], device float *gated1 [[buffer(15)]],
                              device float *gated2 [[buffer(16)]], device bfloat *output0 [[buffer(17)]],
                              device bfloat *output1 [[buffer(18)]], device bfloat *output2 [[buffer(19)]],
                              constant MessageGateFusedParams &p [[buffer(20)]],
                              uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MessageGateRowsShared shared;
  const uint first = group.x * kTileRows, g = group.z;
  if (first >= p.rows || g >= p.gates) return;
  const uint count = min(kTileRows, p.rows - first);
  device float *gated = message_gate_pick(g, gated0, gated1, gated2);
  message_gate_rows_scales(x, message, p, g, first, count, lane, sg, shared);
  message_gate_rows_down(x, message, message_gate_pick(g, down0, down1, down2), p, g, first, count, tid, lane, sg,
                         shared);
  message_gate_rows_gated(x, message, message_gate_pick(g, forget0, forget1, forget2),
                          message_gate_pick(g, gain0, gain1, gain2), gated, p, g, first, count, lane, sg, shared);
  message_gate_rows_norm(gated, message_gate_pick(g, norm0, norm1, norm2), message_gate_pick(g, output0, output1, output2),
                         p, first, count, tid, shared);
}

// The instances: F32 weights (_f32) and BF16 (_bf16), as a GGUF stores the
// gates' projections.
#define MESSAGE_GATE_INSTANCES(Weight, suffix)                                                                    \
  template [[host_name("message_gate_down" suffix)]] kernel void message_gate_down<Weight>(                       \
      device const float *, device const float *, device const Weight *, device const Weight *,                   \
      device const Weight *, device bfloat *, constant MessageGateFusedParams &, uint3, uint, uint, uint);        \
  template [[host_name("message_gate_apply" suffix)]] kernel void message_gate_apply<Weight>(                     \
      device const float *, device const float *, device const Weight *, device const Weight *,                   \
      device const Weight *, device const Weight *, device const Weight *, device const Weight *,                 \
      device const bfloat *, device const float *, device const float *, device const float *,                    \
      device coherent(device) float *, device coherent(device) float *, device atomic_uint *, device bfloat *,    \
      device bfloat *, device bfloat *, constant MessageGateFusedParams &, uint3, uint, uint, uint);              \
  template [[host_name("message_gate_rows" suffix)]] kernel void message_gate_rows<Weight>(                       \
      device const float *, device const float *, device const Weight *, device const Weight *,                   \
      device const Weight *, device const Weight *, device const Weight *, device const Weight *,                 \
      device const Weight *, device const Weight *, device const Weight *, device const float *,                  \
      device const float *, device const float *, device float *, device float *, device float *,                 \
      device bfloat *, device bfloat *, device bfloat *, constant MessageGateFusedParams &, uint3, uint, uint,     \
      uint);
MESSAGE_GATE_INSTANCES(float, "_f32")
MESSAGE_GATE_INSTANCES(bfloat, "_bf16")
#undef MESSAGE_GATE_INSTANCES

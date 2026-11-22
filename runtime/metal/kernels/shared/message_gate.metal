#pragma clang fp reassociate(off)
#include "metal/abi/KernelABI.h"
#include "metal/abi/MessageGate.h"
#include "metal/kernels/common/rms_inverse.h"
#include "metal/kernels/common/row_sums.h"
#include "metal/kernels/common/split_reduce.h"

#include <metal_stdlib>

using namespace metal;

// Message gates (metal/abi/MessageGate.h) over fp32 rows: a threadgroup of
// kGateThreads threads a row, each thread the columns tid + kGateThreads i.
constant constexpr uint kGateThreads = MESSAGE_GATE_THREADS;
constant constexpr uint kGateSimdgroups = kGateThreads / 32;

// A message gate's input to its down projection: the row and its message,
// each RMS-normalized without weights, side by side in a bf16 row of
// 2 hidden values (the source's down projection is a bf16 Linear).
kernel void message_gate_prepare(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                                 device bfloat *concat [[buffer(2)]], constant MessageGateParams &p [[buffer(3)]],
                                 uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kGateSimdgroups * 2];
  if (row >= p.rows) return;
  device const float *xr = x + ulong(row) * p.x_stride + p.x_column;
  device const float *mr = message + ulong(row) * p.message_stride + p.message_column;
  float squares[2] = {0.0f, 0.0f};
  for (uint d = tid; d < p.hidden; d += kGateThreads) {
    squares[0] += xr[d] * xr[d];
    squares[1] += mr[d] * mr[d];
  }
  threadgroup_row_sums<kGateSimdgroups>(squares, 2, scratch, lane, sg);
  const float xs = rsqrt(squares[0] / p.hidden + kRmsEpsilon), ms = rsqrt(squares[1] / p.hidden + kRmsEpsilon);
  device bfloat *out = concat + ulong(row) * 2 * p.hidden;
  for (uint d = tid; d < p.hidden; d += kGateThreads) {
    out[d] = bfloat(xr[d] * xs);
    out[p.hidden + d] = bfloat(mr[d] * ms);
  }
}

// The gated row, (1 - tanh(forget)) x + tanh(gain) message, in fp32 from the
// forget and gain projections' fp32 rows [rows][hidden], through the
// weighted RMS norm after it into bf16 rows.
inline float message_gated(device const float *xr, device const float *mr, device const float *forget,
                           device const float *gain, uint d) {
  return (1.0f - precise::tanh(forget[d])) * xr[d] + precise::tanh(gain[d]) * mr[d];
}
kernel void message_gate_norm(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                              device const float *forget [[buffer(2)]], device const float *gain [[buffer(3)]],
                              device const float *norm [[buffer(4)]], device bfloat *output [[buffer(5)]],
                              constant MessageGateParams &p [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kGateSimdgroups];
  if (row >= p.rows) return;
  device const float *xr = x + ulong(row) * p.x_stride + p.x_column;
  device const float *mr = message + ulong(row) * p.message_stride + p.message_column;
  device const float *f = forget + ulong(row) * p.hidden, *g = gain + ulong(row) * p.hidden;
  float squares[1] = {0.0f};
  for (uint d = tid; d < p.hidden; d += kGateThreads) {
    const float v = message_gated(xr, mr, f, g, d);
    squares[0] += v * v;
  }
  threadgroup_row_sums<kGateSimdgroups>(squares, 1, scratch, lane, sg);
  const float inverse = rsqrt(squares[0] / p.hidden + kRmsEpsilon);
  for (uint d = tid; d < p.hidden; d += kGateThreads)
    output[ulong(row) * p.hidden + d] = bfloat((message_gated(xr, mr, f, g, d) * inverse) * norm[d]);
}

// ---- The fused gate (metal/abi/MessageGate.h): a decode step's rows ---------
//
// Each kernel runs in phases, functions out of line, which hand over through
// threadgroup memory alone: every sum is reduced by simd_sum into its own slot
// a simdgroup and then by one thread over the simdgroups in order; loops over
// a group's rows run to MESSAGE_GATE_GROUP_ROWS, the rows past the group's
// contributing zeros; no thread keeps an array across a barrier (as
// block_residual_mix.metal: an array indexed by a runtime bound and alive
// across a barrier lost one simdgroup's values now and then under shader
// validation). The sums are those of message_gate_prepare, the float
// projections' and message_gate_norm's, in their order within a thread and
// over the simdgroups.
constant constexpr uint kGroupRows = MESSAGE_GATE_GROUP_ROWS;
constant constexpr uint kDownOutputs = MESSAGE_GATE_DOWN_OUTPUTS;

// What message_gate_down's phases share: a partial a simdgroup of each sum
// (each row's and each message's squares, then each output's rank value of
// each row), and each row's and message's inverse RMS (zero past the group's
// rows).
struct MessageGateDownShared {
  float partials[(2 + kDownOutputs) * kGroupRows * kGateSimdgroups];
  float scales[2 * kGroupRows];
};

// A group of `count` rows from row `first`: row r's x, or its message, the
// group's last row standing in for the rows past it (their scales are zero).
inline device const float *message_gate_row(device const float *x, device const float *message,
                                            constant MessageGateParams &p, uint first, uint count, uint r,
                                            bool own) {
  const ulong row = first + min(r, count - 1);
  return own ? x + row * p.x_stride + p.x_column : message + row * p.message_stride + p.message_column;
}

// Phase one: each row's and each message's inverse RMS, as
// message_gate_prepare's (the same sums, the same order).
__attribute__((noinline)) void message_gate_down_scales(device const float *x, device const float *message,
                                                        constant MessageGateParams &p, uint first, uint count,
                                                        uint tid, uint lane, uint sg,
                                                        threadgroup MessageGateDownShared &shared) {
  for (uint i = 0; i < 2 * kGroupRows; ++i) {
    device const float *row = message_gate_row(x, message, p, first, count, i / 2, i % 2 == 0);
    float squares = 0.0f;
    if (i / 2 < count)
      for (uint d = tid; d < p.hidden; d += kGateThreads) squares += row[d] * row[d];
    squares = simd_sum(squares);
    if (lane == 0) shared.partials[i * kGateSimdgroups + sg] = squares;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 2 * kGroupRows) {
    float total = 0.0f;
    for (uint g = 0; g < kGateSimdgroups; ++g) total += shared.partials[tid * kGateSimdgroups + g];
    shared.scales[tid] = tid / 2 < count ? rsqrt(total / p.hidden + kRmsEpsilon) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Phase two: the group's rank values k0 .. k0 + kDownOutputs - 1, each the
// dot of the weights' row (`down`, F32 [rank][2 hidden]) with the normalized
// concatenation, rounded to bf16 as message_gate_prepare rounds it, the
// threads striding its float4s; rounded to bf16 into `low` [rows][rank].
__attribute__((noinline)) void message_gate_down_values(device const float *x, device const float *message,
                                                        device const float *down, device bfloat *low,
                                                        constant MessageGateParams &p, uint first, uint count,
                                                        uint k0, uint tid, uint lane, uint sg,
                                                        threadgroup MessageGateDownShared &shared) {
  const uint quarter = p.hidden / 4;
  float sums[kDownOutputs * kGroupRows];
  for (uint j = 0; j < kDownOutputs * kGroupRows; ++j) sums[j] = 0.0f;
  for (uint i = tid; i < 2 * quarter; i += kGateThreads) {
    float4 w[kDownOutputs];
    for (uint o = 0; o < kDownOutputs; ++o)
      w[o] = k0 + o < p.rank ? reinterpret_cast<device const float4 *>(down + ulong(k0 + o) * 2 * p.hidden)[i]
                             : float4(0.0f);
    const bool own = i < quarter;
    for (uint r = 0; r < kGroupRows; ++r) {
      device const float4 *row =
          reinterpret_cast<device const float4 *>(message_gate_row(x, message, p, first, count, r, own));
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
    for (uint g = 0; g < kGateSimdgroups; ++g)
      total += shared.partials[(2 * kGroupRows + tid) * kGateSimdgroups + g];
    const uint o = tid / kGroupRows, r = tid % kGroupRows;
    if (r < count && k0 + o < p.rank) low[ulong(first + r) * p.rank + k0 + o] = bfloat(total);
  }
}

// The rank values of a group of MESSAGE_GATE_GROUP_ROWS rows, of
// MESSAGE_GATE_DOWN_OUTPUTS of them a threadgroup, each output's weights' row
// read once for every row of the group. Grid: (rank /
// MESSAGE_GATE_DOWN_OUTPUTS, groups); x, the message and their strides and
// columns in whole float4s.
kernel void message_gate_down(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                              device const float *down [[buffer(2)]], device bfloat *low [[buffer(3)]],
                              constant MessageGateParams &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MessageGateDownShared shared;
  const uint first = group.y * kGroupRows;
  if (first >= p.rows) return;
  const uint count = min(kGroupRows, p.rows - first);
  message_gate_down_scales(x, message, p, first, count, tid, lane, sg, shared);
  message_gate_down_values(x, message, down, low, p, first, count, group.x * kDownOutputs, tid, lane, sg, shared);
}

// What message_gate_apply's phases share: the group's rank values (zero past
// its rows), a partial a simdgroup of each row's squares, and the arrival of
// split_arrive_last.
struct MessageGateApplyShared {
  float4 lows[kGroupRows * MESSAGE_GATE_MAXIMUM_RANK / 4];
  float partials[kGroupRows * kGateSimdgroups];
  uint arrival;
};

// Phase one: value d of each row of the group, a thread each: forget and gain
// from the rank values and the weights' rows (F32 [hidden][rank]), the gated
// value in fp32 (message_gated's) to `values` [rows][hidden], and the slice's
// squares of each row, by thread 0 to its slot of `partials`
// [group][slice][MESSAGE_GATE_GROUP_ROWS].
__attribute__((noinline)) void message_gate_apply_values(device const float *x, device const float *message,
                                                         device const float *forget, device const float *gain,
                                                         device coherent(device) float *values,
                                                         device coherent(device) float *partials,
                                                         constant MessageGateParams &p, uint first, uint count,
                                                         uint slice, uint tid, uint lane, uint sg,
                                                         threadgroup MessageGateApplyShared &shared) {
  const uint d = slice * kGateThreads + tid, quarters = p.rank / 4;
  const bool active = d < p.hidden;
  float f[kGroupRows], g[kGroupRows];
  for (uint r = 0; r < kGroupRows; ++r) f[r] = g[r] = 0.0f;
  if (active) {
    device const float4 *f4 = reinterpret_cast<device const float4 *>(forget + ulong(d) * p.rank);
    device const float4 *g4 = reinterpret_cast<device const float4 *>(gain + ulong(d) * p.rank);
    for (uint j = 0; j < quarters; ++j) {
      const float4 a = f4[j], b = g4[j];
      for (uint r = 0; r < kGroupRows; ++r) {
        const float4 l = shared.lows[r * quarters + j];
        f[r] += dot(a, l);
        g[r] += dot(b, l);
      }
    }
  }
  for (uint r = 0; r < kGroupRows; ++r) {
    float squares = 0.0f;
    if (active && r < count) {
      const float xv = x[ulong(first + r) * p.x_stride + p.x_column + d];
      const float mv = message[ulong(first + r) * p.message_stride + p.message_column + d];
      const float v = (1.0f - precise::tanh(f[r])) * xv + precise::tanh(g[r]) * mv;
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
// slices in slice order, then the row through the weighted RMS norm into
// bf16, as message_gate_norm writes it.
__attribute__((noinline)) void message_gate_apply_norm(device coherent(device) float *values,
                                                       device coherent(device) float *partials,
                                                       device const float *norm, device bfloat *output,
                                                       constant MessageGateParams &p, uint first, uint count,
                                                       uint slices, uint tid) {
  for (uint r = 0; r < kGroupRows; ++r) {
    if (r >= count) break;
    float total = 0.0f;
    for (uint s = 0; s < slices; ++s) total += partials[s * kGroupRows + r];
    const float inverse = rsqrt(total / p.hidden + kRmsEpsilon);
    for (uint e = tid; e < p.hidden; e += kGateThreads)
      output[ulong(first + r) * p.hidden + e] = bfloat((values[ulong(first + r) * p.hidden + e] * inverse) * norm[e]);
  }
}

// The gated rows of a group from its rank values (`low`): a slice of
// MESSAGE_GATE_THREADS values a threadgroup, its forget and gain rows read
// once for the group's rows; the slice that arrives last
// (kernels/common/split_reduce.h, the group's counter, zero before and after)
// writes the group's normalized rows. Grid: (slices, groups).
kernel void message_gate_apply(device const float *x [[buffer(0)]], device const float *message [[buffer(1)]],
                               device const float *forget [[buffer(2)]], device const float *gain [[buffer(3)]],
                               device const bfloat *low [[buffer(4)]], device const float *norm [[buffer(5)]],
                               device coherent(device) float *values [[buffer(6)]],
                               device coherent(device) float *partials [[buffer(7)]],
                               device atomic_uint *counters [[buffer(8)]], device bfloat *output [[buffer(9)]],
                               constant MessageGateParams &p [[buffer(10)]],
                               uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup MessageGateApplyShared shared;
  const uint first = group.y * kGroupRows;
  if (first >= p.rows) return;
  const uint count = min(kGroupRows, p.rows - first), slices = (p.hidden + kGateThreads - 1) / kGateThreads;
  threadgroup float *lows = reinterpret_cast<threadgroup float *>(shared.lows);
  for (uint i = tid; i < kGroupRows * p.rank; i += kGateThreads)
    lows[i] = i < count * p.rank ? float(low[ulong(first) * p.rank + i]) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device coherent(device) float *groupPartials = partials + ulong(group.y) * slices * kGroupRows;
  message_gate_apply_values(x, message, forget, gain, values, groupPartials, p, first, count, group.x, tid, lane, sg,
                            shared);
  if (!split_arrive_last(counters + group.y, slices, tid, &shared.arrival)) return;
  message_gate_apply_norm(values, groupPartials, norm, output, p, first, count, slices, tid);
  split_release(counters + group.y, tid);
}

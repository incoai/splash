#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/rms_inverse.h"

// Every norm reads its weights in their stored type W: bfloat in the affine
// formats, float for a GGUF's F32 norms (the _f32 entry points). Both widen to
// fp32 exactly, so W changes only the loads.
//
// Wide decode norms hold their row in registers, kNormChunk columns at a time:
// 256 threads of kNormColumns columns each, one chunk for every hidden size
// in use. The first chunk's input and weight loads are all issued before the
// reduction instead of per output iteration, so the (cold) weights arrive in
// one round trip, which makes a chain of eight-row norms of 5120 1.8x (40-core
// M3 Max) to 3.3x (16-core M5 Pro) faster with the same bits.
constant constexpr uint kNormColumns = 32;
constant constexpr uint kNormThreads = 256;
constant constexpr uint kNormChunk = kNormThreads * kNormColumns;
// Rows of at most eight values per thread stream instead: a chunk would leave
// most of the 256 threads idle (a 2048-wide row fills 64), which made the
// 35B's MoE-input norms ~25% slower on a 40-core M3 Max. Both paths give the
// same bits.
constant constexpr uint kNormStreamingWidth = kNormThreads * 8;

// Columns begin + tid + 256 i of a row, zero past its end.
template <class T>
inline void load_norm_chunk(device const T *row, uint width, uint begin,
                            uint tid, thread T (&x)[kNormColumns]) {
  for (uint i = 0; i < kNormColumns; ++i) {
    const uint column = begin + tid + kNormThreads * i;
    x[i] = column < width ? row[column] : T(0.0f);
  }
}

// The thread's squares in column order, as rms_inverse adds them (zeros past
// the row's end add nothing).
inline float add_squares(thread const bfloat (&x)[kNormColumns], float sum) {
#pragma clang fp reassociate(off)
  for (uint i = 0; i < kNormColumns; ++i) {
    const float value = float(x[i]);
    sum += value * value;
  }
  return sum;
}

// The row's inverse RMS from its first chunk (in registers) and the rest.
inline float rms_inverse_of(thread const bfloat (&first)[kNormColumns],
                            device const bfloat *row, uint width,
                            threadgroup float *reductions, uint tid, uint lane,
                            uint sg) {
  float sum = add_squares(first, 0.0f);
  for (uint begin = kNormChunk; begin < width; begin += kNormChunk) {
    bfloat x[kNormColumns];
    load_norm_chunk(row, width, begin, tid, x);
    sum = add_squares(x, sum);
  }
  return rms_inverse_of_sums(sum, width, reductions, tid, lane, sg);
}

template <class W>
inline void norm_rms_row(device const bfloat *input, device const W *weight,
                         device bfloat *output, uint width, uint row,
                         uint thread_index, uint lane, uint simd_group,
                         threadgroup float *reductions) {
#pragma clang fp reassociate(off)
  if (width <= kNormStreamingWidth) {
    const float inverse = rms_inverse(input + row * width, width, reductions,
                                      thread_index, lane, simd_group);
    for (uint column = thread_index; column < width; column += kNormThreads)
      output[row * width + column] =
          bfloat((float(input[row * width + column]) * inverse) * float(weight[column]));
    return;
  }
  device const bfloat *row_input = input + row * width;
  bfloat x[kNormColumns];
  W w[kNormColumns];
  load_norm_chunk(row_input, width, 0, thread_index, x);
  load_norm_chunk(weight, width, 0, thread_index, w);
  const float inverse_rms = rms_inverse_of(x, row_input, width, reductions,
                                           thread_index, lane, simd_group);
  for (uint begin = 0; begin < width; begin += kNormChunk) {
    if (begin) {
      load_norm_chunk(row_input, width, begin, thread_index, x);
      load_norm_chunk(weight, width, begin, thread_index, w);
    }
    for (uint i = 0; i < kNormColumns; ++i) {
      const uint column = begin + thread_index + kNormThreads * i;
      if (column < width)
        output[row * width + column] =
            bfloat((float(x[i]) * inverse_rms) * float(w[i]));
    }
  }
}
#define NORM_RMS(Name, W) \
  kernel void Name(device const bfloat *input [[buffer(0)]], \
      device const W *weight [[buffer(1)]], device bfloat *output [[buffer(2)]], \
      constant uint &width [[buffer(3)]], uint row [[threadgroup_position_in_grid]], \
      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
      uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float reductions[8]; \
    norm_rms_row(input, weight, output, width, row, tid, lane, sg, reductions); \
  }
NORM_RMS(norm_rms, bfloat)
NORM_RMS(norm_rms_f32, float)
#undef NORM_RMS

// Plain norms of few rows of at most 2048 columns (ops/Normalization.cpp)
// load each row once into threadgroup memory, a vector per thread of a
// 1024-thread group, and scale it from there instead of reading it from
// device memory twice as norm_rms does. The reduction is rms_inverse's over
// the first 256 threads, so the bits match every other norm. On 2048-column
// rows, up to 64 rows (SPLASH_STAGED_NORM_ROWS), that is 1.3-2.6x faster than
// norm_rms in chains of dependent norms with DRAM-cold weights (40-core M3
// Max, 20-core M5 Pro); from 128 rows on, norm_rms's 256-thread groups overlap
// and match or beat it, and on 5120-column rows it does not beat the register
// path. WV is the weights' vector type: bfloat4, or packed_float4 for a GGUF's
// F32 norms, which assume no more than scalar alignment.
static_assert(SPLASH_STAGED_NORM_WIDTH % 4 == 0, "the staged row is a whole number of vectors");
static_assert(SPLASH_STAGED_NORM_THREADS % 256 == 0, "rms_inverse's 256 threads are whole simdgroups");
template <class WV>
inline void norm_rms_staged_row(device const bfloat4 *input, device const WV *weight,
                                device bfloat4 *output, uint width, uint row, uint tid,
                                uint lane, uint sg, threadgroup bfloat4 *stage,
                                threadgroup float *reductions) {
#pragma clang fp reassociate(off)
  const uint vectors = width / 4;
  for (uint column = tid; column < vectors; column += SPLASH_STAGED_NORM_THREADS)
    stage[column] = input[row * vectors + column];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The first 256 threads reduce the row as rms_inverse does; the other
  // simdgroups' partials are zero and unread.
  const float sum = tid < 256 ? row_squares((threadgroup const bfloat *)stage, width, tid) : 0.0f;
  const float inverse = rms_inverse_of_sums(sum, width, reductions, tid, lane, sg);
  for (uint column = tid; column < vectors; column += SPLASH_STAGED_NORM_THREADS)
    output[row * vectors + column] = bfloat4((float4(stage[column]) * inverse) * float4(weight[column]));
}
#define NORM_RMS_STAGED(Name, WV) \
  kernel void Name(device const bfloat4 *input [[buffer(0)]], \
      device const WV *weight [[buffer(1)]], device bfloat4 *output [[buffer(2)]], \
      constant uint &width [[buffer(3)]], uint row [[threadgroup_position_in_grid]], \
      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
      uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup bfloat4 stage[SPLASH_STAGED_NORM_WIDTH / 4]; \
    threadgroup float reductions[SPLASH_STAGED_NORM_THREADS / 32]; \
    norm_rms_staged_row(input, weight, output, width, row, tid, lane, sg, stage, reductions); \
  }
NORM_RMS_STAGED(norm_rms_staged, bfloat4)
NORM_RMS_STAGED(norm_rms_staged_f32, packed_float4)
#undef NORM_RMS_STAGED

// Keep the ordinary output for non-matrix consumers, and emit the consumer's
// matrix operand table (Table: q4sg::Table64 affine, gguf_sg::Table16 GGUF) from
// the same rounded bfloat values. No additional dispatch is needed. The table
// takes a simdgroup per 64-column span, so the first chunk's span pairs are
// loaded (L2-hot input, weights) alongside the reduction's columns.
template <class Table, class W>
inline void norm_rms_table(device const bfloat *input, device const W *weight,
                           device bfloat *output, device bfloat *table, device float *sums,
                           uint width, uint row, uint tid, uint lane, uint sg,
                           threadgroup float *reductions) {
#pragma clang fp reassociate(off)
  if (width <= kNormStreamingWidth) {
    device const bfloat *row_input = input + row * width;
    const float inverse = rms_inverse(row_input, width, reductions, tid, lane, sg);
    for (uint g = sg; g < width / 64; g += 8) {
      const uint k = g * 64 + lane * 2;
      const bfloat a = bfloat((float(row_input[k]) * inverse) * float(weight[k]));
      const bfloat b = bfloat((float(row_input[k + 1]) * inverse) * float(weight[k + 1]));
      output[row * width + k] = a;
      output[row * width + k + 1] = b;
      Table::write(table + ulong(row / 8) * width * 8,
                   sums + ulong(row / 8) * Table::sums_per_tile(width),
                   width, g, row % 8, lane, a, b);
    }
    return;
  }
  constexpr uint Spans = kNormChunk / 64 / 8;  // per simdgroup and chunk
  device const bfloat *row_input = input + row * width;
  bfloat x[kNormColumns];
  load_norm_chunk(row_input, width, 0, tid, x);
  bfloat2 in[Spans];
  vec<W, 2> w[Spans];
  const auto load_spans = [&](uint begin) {
    for (uint j = 0; j < Spans; ++j) {
      const uint k = begin + (sg + 8 * j) * 64 + lane * 2;
      in[j] = k < width ? bfloat2(row_input[k], row_input[k + 1]) : bfloat2(bfloat(0.0f));
      w[j] = k < width ? vec<W, 2>(weight[k], weight[k + 1]) : vec<W, 2>(W(0.0f));
    }
  };
  load_spans(0);
  const float inverse = rms_inverse_of(x, row_input, width, reductions, tid, lane, sg);
  for (uint begin = 0; begin < width; begin += kNormChunk) {
    if (begin)
      load_spans(begin);
    for (uint j = 0; j < Spans; ++j) {
      const uint k = begin + (sg + 8 * j) * 64 + lane * 2;
      if (k >= width)
        break;
      const bfloat a = bfloat((float(in[j].x) * inverse) * float(w[j].x));
      const bfloat b = bfloat((float(in[j].y) * inverse) * float(w[j].y));
      output[row * width + k] = a;
      output[row * width + k + 1] = b;
      Table::write(table + ulong(row / 8) * width * 8,
                   sums + ulong(row / 8) * Table::sums_per_tile(width), width, k / 64, row % 8, lane,
                   a, b);
    }
  }
}
#define NORM_RMS_TABLE(Name, Table, W) \
  kernel void Name(device const bfloat *input [[buffer(0)]], \
      device const W *weight [[buffer(1)]], device bfloat *output [[buffer(2)]], \
      device bfloat *table [[buffer(3)]], device float *sums [[buffer(4)]], \
      constant uint &width [[buffer(5)]], uint row [[threadgroup_position_in_grid]], \
      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
      uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float reductions[8]; \
    norm_rms_table<Table>(input, weight, output, table, sums, width, row, tid, lane, sg, reductions); \
  }
// The reachable pairs: Table64 feeds affine projections, from the affine targets' and the draft's bf16 norms.
// Table16 feeds a GGUF target's register-tile projections, from its F32 norms, and also the target's vocabulary
// head from the draft's bf16 final norm (DFlashDraft::addDecode). No F32 norm feeds an affine projection.
NORM_RMS_TABLE(norm_rms_table64_decode, q4sg::Table64, bfloat)
NORM_RMS_TABLE(norm_rms_table16_decode, gguf_sg::Table16, bfloat)
NORM_RMS_TABLE(norm_rms_table16_decode_f32, gguf_sg::Table16, float)
#undef NORM_RMS_TABLE

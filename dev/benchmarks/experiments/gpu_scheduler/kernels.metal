#include <metal_stdlib>
using namespace metal;
struct Params {
  uint width, stages, columns, stage;
};
inline void execute(device coherent(device) const float *input,
                    device coherent(device) float *output, uint job,
                    constant Params &p, threadgroup float *sums, uint tid,
                    uint lane, uint sg) {
  const uint stage = job / p.width, row = job % p.width;
  device coherent(device) const float *source =
      stage ? output + (job - p.width) * p.columns : input + row * p.columns;
  device coherent(device) const float *other =
      stage ? output + ((stage - 1) * p.width + (row + 1) % p.width) * p.columns
            : source;
  if (stage % 3 == 2) {
    float sum = 0;
    for (uint col = tid; col < p.columns; col += 256) {
      const float x = source[col];
      sum += x * x;
    }
    sum = simd_sum(sum);
    if (lane == 0)
      sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float total = 0;
      for (uint i = 0; i < 8; ++i)
        total += sums[i];
      sums[0] = rsqrt(total / p.columns + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint col = tid; col < p.columns; col += 256)
      output[job * p.columns + col] = source[col] * sums[0];
  } else
    for (uint col = tid; col < p.columns; col += 256) {
      const float x = source[col];
      output[job * p.columns + col] = stage % 3 == 0
                                          ? x * .875f + float(col % 31) * .001f
                                          : x / (1 + fast::exp(-other[col]));
    }
}
kernel void static_stage(device coherent(device) const float *input
                         [[buffer(0)]],
                         device coherent(device) float *output [[buffer(1)]],
                         constant Params &p [[buffer(2)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[8];
  execute(input, output, p.stage * p.width + row, p, sums, tid, lane, sg);
}
kernel void dynamic_dag(device coherent(device) const float *input
                        [[buffer(0)]],
                        device coherent(device) float *output [[buffer(1)]],
                        device atomic_uint *status [[buffer(2)]],
                        constant Params &p [[buffer(3)]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint jobs = p.width * p.stages;
  threadgroup uint selected;
  threadgroup float sums[8];
  for (uint attempt = 0; attempt < 4096; ++attempt) {
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope_device);
      selected = jobs + 1;
      if (atomic_load_explicit(status + jobs, memory_order_relaxed) == jobs)
        selected = jobs;
      else
        for (uint job = 0; job < jobs; ++job) {
          if (atomic_load_explicit(status + job, memory_order_relaxed) != 0)
            continue;
          const uint stage = job / p.width, row = job % p.width;
          if (stage && (atomic_load_explicit(status + job - p.width,
                                             memory_order_relaxed) != 2 ||
                        (stage % 3 == 1 &&
                         atomic_load_explicit(status + (stage - 1) * p.width +
                                                  (row + 1) % p.width,
                                              memory_order_relaxed) != 2)))
            continue;
          uint expected = 0;
          if (atomic_compare_exchange_weak_explicit(status + job, &expected, 1u,
                                                    memory_order_relaxed,
                                                    memory_order_relaxed)) {
            selected = job;
            break;
          }
        }
    }
    if (tid == 0)
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    const uint job = selected;
    if (job == jobs)
      return;
    if (job == jobs + 1)
      continue;
    execute(input, output, job, p, sums, tid, lane, sg);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope_device);
      atomic_store_explicit(status + job, 2u, memory_order_relaxed);
      atomic_fetch_add_explicit(status + jobs, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0 &&
      atomic_load_explicit(status + jobs, memory_order_relaxed) != jobs)
    atomic_store_explicit(status + jobs + 1, 1u, memory_order_relaxed);
}

// One macro task owns one row through affine/SiLU/RMS (or the final
// affine/SiLU). Neighbor affine values are recomputed, avoiding a dispatch
// between producers and their consumers while retaining the baseline's stored
// FP32 intermediates.
inline void execute_coarse(device coherent(device) const float *input,
                           device coherent(device) float *output, uint job,
                           constant Params &p, threadgroup float *sums,
                           uint tid, uint lane, uint sg) {
  const uint macro = job / p.width, row = job % p.width, start = macro * 3;
  device coherent(device) const float *source =
      macro ? output + (start - 1) * p.width * p.columns : input;
  device coherent(device) float *affine =
      output + (start * p.width + row) * p.columns;
  device coherent(device) float *silu = affine + p.width * p.columns;
  for (uint col = tid; col < p.columns; col += 256) {
    // Preserve the baseline's contraction and its FP32 affine boundary.
    const float a =
        fma(float(col % 31), .001f, source[row * p.columns + col] * .875f);
    const float b =
        fma(float(col % 31), .001f,
            source[((row + 1) % p.width) * p.columns + col] * .875f);
    affine[col] = a;
    silu[col] = a / (1 + fast::exp(-b));
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (macro < 2)
    execute(input, output, (start + 2) * p.width + row, p, sums, tid, lane, sg);
}
kernel void static_coarse(device coherent(device) const float *input
                          [[buffer(0)]],
                          device coherent(device) float *output [[buffer(1)]],
                          constant Params &p [[buffer(2)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[8];
  execute_coarse(input, output, p.stage * p.width + row, p, sums, tid, lane,
                 sg);
}

template <bool Coarse, bool Noop>
inline void run_queue(device coherent(device) const float *input,
                      device coherent(device) float *output,
                      device atomic_uint *status, constant Params &p,
                      threadgroup uint &selected, threadgroup float *sums,
                      uint tid, uint lane, uint sg) {
  const uint stages = Coarse ? 3 : p.stages;
  const uint jobs = p.width * stages;
  device atomic_uint *head = status + jobs + 2, *tail = status + jobs + 3,
                     *deps = status + jobs + 4;
  device atomic_uint *queue = status + 2 * jobs + 4,
                     *ready = status + 3 * jobs + 4;
  for (uint attempt = 0; attempt < 4096; ++attempt) {
    if (tid == 0) {
      selected = jobs + 1;
      if (atomic_load_explicit(status + jobs, memory_order_relaxed) == jobs)
        selected = jobs;
      else {
        uint pos = atomic_load_explicit(head, memory_order_relaxed);
        if (pos < atomic_load_explicit(tail, memory_order_relaxed) &&
            atomic_load_explicit(ready + pos, memory_order_relaxed)) {
          uint expected = pos;
          if (atomic_compare_exchange_weak_explicit(head, &expected, pos + 1,
                                                    memory_order_relaxed,
                                                    memory_order_relaxed)) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                                thread_scope_device);
            selected = atomic_load_explicit(queue + pos, memory_order_relaxed);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    const uint job = selected;
    if (job == jobs)
      return;
    if (job == jobs + 1)
      continue;
    if constexpr (!Noop) {
      if constexpr (Coarse)
        execute_coarse(input, output, job, p, sums, tid, lane, sg);
      else
        execute(input, output, job, p, sums, tid, lane, sg);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope_device);
      atomic_store_explicit(status + job, 2u, memory_order_relaxed);
      const uint stage = job / p.width, row = job % p.width;
      if (stage + 1 < stages) {
        const uint children = Coarse ? 2 : ((stage + 1) % 3 == 1 ? 2 : 1);
        for (uint i = 0; i < children; ++i) {
          const uint child =
              (stage + 1) * p.width + (i ? (row + p.width - 1) % p.width : row);
          const uint previous =
              atomic_fetch_sub_explicit(deps + child, 1u, memory_order_relaxed);
          atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                              thread_scope_device);
          if (previous == 1) {
            const uint slot =
                atomic_fetch_add_explicit(tail, 1u, memory_order_relaxed);
            if (slot >= jobs)
              atomic_store_explicit(status + jobs + 1, 1u,
                                    memory_order_relaxed);
            else {
              atomic_store_explicit(queue + slot, child, memory_order_relaxed);
              atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                                  thread_scope_device);
              atomic_store_explicit(ready + slot, 1u, memory_order_relaxed);
            }
          }
        }
      }
      atomic_fetch_add_explicit(status + jobs, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0 &&
      atomic_load_explicit(status + jobs, memory_order_relaxed) != jobs)
    atomic_store_explicit(status + jobs + 1, 1u, memory_order_relaxed);
}

kernel void dynamic_queue(device coherent(device) const float *input
                          [[buffer(0)]],
                          device coherent(device) float *output [[buffer(1)]],
                          device atomic_uint *status [[buffer(2)]],
                          constant Params &p [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint selected = 0;
  threadgroup float sums[8];
  run_queue<false, false>(input, output, status, p, selected, sums, tid, lane,
                          sg);
}

kernel void queue_noop(device coherent(device) const float *input [[buffer(0)]],
                       device coherent(device) float *output [[buffer(1)]],
                       device atomic_uint *status [[buffer(2)]],
                       constant Params &p [[buffer(3)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint selected = 0;
  threadgroup float sums[8];
  run_queue<false, true>(input, output, status, p, selected, sums, tid, lane,
                         sg);
}

kernel void queue_coarse(device coherent(device) const float *input
                         [[buffer(0)]],
                         device coherent(device) float *output [[buffer(1)]],
                         device atomic_uint *status [[buffer(2)]],
                         constant Params &p [[buffer(3)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint selected = 0;
  threadgroup float sums[8];
  run_queue<true, false>(input, output, status, p, selected, sums, tid, lane,
                         sg);
}

inline float affine_baseline(float x, uint col) {
  return fma(float(col % 31), .001f, x * .875f);
}
inline void normalize_private(device float *row, uint columns,
                              threadgroup float *sums, uint tid, uint lane,
                              uint sg) {
  float sum = 0;
  for (uint col = tid; col < columns; col += 256) {
    const float x = row[col];
    sum += x * x;
  }
  sum = simd_sum(sum);
  if (lane == 0)
    sums[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0;
    for (uint i = 0; i < 8; ++i)
      total += sums[i];
    sums[0] = rsqrt(total / columns + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint col = tid; col < columns; col += 256)
    row[col] = row[col] * sums[0];
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}
// The fixed DAG's three SiLU stages need a bounded neighboring-row halo.
// Each group owns one output row and recomputes that halo without a global
// queue.
kernel void static_halo(device const float *input [[buffer(0)]],
                        device float *output [[buffer(1)]],
                        device float *work [[buffer(2)]],
                        constant Params &p [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[8];
  device float *scratch = work + row * 3 * p.columns;
  for (uint halo = 0; halo < 3; ++halo) {
    const uint self = (row + halo) % p.width, neighbor = (self + 1) % p.width;
    device float *current = scratch + halo * p.columns;
    for (uint col = tid; col < p.columns; col += 256) {
      const float a = affine_baseline(input[self * p.columns + col], col);
      const float b = affine_baseline(input[neighbor * p.columns + col], col);
      const float y = a / (1 + fast::exp(-b));
      current[col] = y;
      if (halo == 0) {
        output[row * p.columns + col] = a;
        output[(p.width + row) * p.columns + col] = y;
      }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    normalize_private(current, p.columns, sums, tid, lane, sg);
    if (halo == 0)
      for (uint col = tid; col < p.columns; col += 256)
        output[(2 * p.width + row) * p.columns + col] = current[col];
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
  for (uint halo = 0; halo < 2; ++halo) {
    device float *current = scratch + halo * p.columns;
    device const float *neighbor = current + p.columns;
    for (uint col = tid; col < p.columns; col += 256) {
      const float a = affine_baseline(current[col], col),
                  b = affine_baseline(neighbor[col], col);
      const float y = a / (1 + fast::exp(-b));
      current[col] = y;
      if (halo == 0) {
        output[(3 * p.width + row) * p.columns + col] = a;
        output[(4 * p.width + row) * p.columns + col] = y;
      }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    normalize_private(current, p.columns, sums, tid, lane, sg);
    if (halo == 0)
      for (uint col = tid; col < p.columns; col += 256)
        output[(5 * p.width + row) * p.columns + col] = current[col];
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
  for (uint col = tid; col < p.columns; col += 256) {
    const float a = affine_baseline(scratch[col], col),
                b = affine_baseline(scratch[p.columns + col], col);
    output[(6 * p.width + row) * p.columns + col] = a;
    output[(7 * p.width + row) * p.columns + col] = a / (1 + fast::exp(-b));
  }
}

inline void execute_plain(device const float *input, device float *output,
                          uint job, constant Params &p, threadgroup float *sums,
                          uint tid, uint lane, uint sg) {
  const uint stage = job / p.width, row = job % p.width;
  device const float *source =
      stage ? output + (job - p.width) * p.columns : input + row * p.columns;
  device const float *other =
      stage ? output + ((stage - 1) * p.width + (row + 1) % p.width) * p.columns
            : source;
  if (stage % 3 == 2) {
    float sum = 0;
    for (uint col = tid; col < p.columns; col += 256) {
      const float x = source[col];
      sum += x * x;
    }
    sum = simd_sum(sum);
    if (lane == 0)
      sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float total = 0;
      for (uint i = 0; i < 8; ++i)
        total += sums[i];
      sums[0] = rsqrt(total / p.columns + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint col = tid; col < p.columns; col += 256)
      output[job * p.columns + col] = source[col] * sums[0];
  } else
    for (uint col = tid; col < p.columns; col += 256) {
      const float x = source[col];
      output[job * p.columns + col] =
          stage % 3 == 0 ? fma(float(col % 31), .001f, x * .875f)
                         : x / (1 + fast::exp(-other[col]));
    }
}
inline void execute_coarse_plain(device const float *input,
                                 device float *output, uint job,
                                 constant Params &p, threadgroup float *sums,
                                 uint tid, uint lane, uint sg) {
  const uint macro = job / p.width, row = job % p.width, start = macro * 3;
  device const float *source =
      macro ? output + (start - 1) * p.width * p.columns : input;
  device float *affine = output + (start * p.width + row) * p.columns;
  device float *silu = affine + p.width * p.columns;
  for (uint col = tid; col < p.columns; col += 256) {
    // Preserve the baseline's contraction and its FP32 affine boundary.
    const float a =
        fma(float(col % 31), .001f, source[row * p.columns + col] * .875f);
    const float b =
        fma(float(col % 31), .001f,
            source[((row + 1) % p.width) * p.columns + col] * .875f);
    affine[col] = a;
    silu[col] = a / (1 + fast::exp(-b));
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (macro < 2)
    execute_plain(input, output, (start + 2) * p.width + row, p, sums, tid,
                  lane, sg);
}

kernel void static_stage_plain(device const float *input [[buffer(0)]],
                               device float *output [[buffer(1)]],
                               constant Params &p [[buffer(2)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[8];
  execute_plain(input, output, p.stage * p.width + row, p, sums, tid, lane, sg);
}

kernel void static_coarse_plain(device const float *input [[buffer(0)]],
                                device float *output [[buffer(1)]],
                                constant Params &p [[buffer(2)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[8];
  execute_coarse_plain(input, output, p.stage * p.width + row, p, sums, tid,
                       lane, sg);
}

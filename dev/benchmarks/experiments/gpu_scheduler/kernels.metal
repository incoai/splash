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

kernel void dynamic_queue(device coherent(device) const float *input
                          [[buffer(0)]],
                          device coherent(device) float *output [[buffer(1)]],
                          device atomic_uint *status [[buffer(2)]],
                          constant Params &p [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint jobs = p.width * p.stages;
  device atomic_uint *head = status + jobs + 2, *tail = status + jobs + 3,
                     *deps = status + jobs + 4;
  device atomic_uint *queue = status + 2 * jobs + 4,
                     *ready = status + 3 * jobs + 4;
  threadgroup uint selected;
  threadgroup float sums[8];
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
    execute(input, output, job, p, sums, tid, lane, sg);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                          thread_scope_device);
      atomic_store_explicit(status + job, 2u, memory_order_relaxed);
      const uint stage = job / p.width, row = job % p.width;
      if (stage + 1 < p.stages) {
        const uint children = (stage + 1) % 3 == 1 ? 2 : 1;
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

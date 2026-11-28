#pragma once
#include <metal_stdlib>
using namespace metal;

// Cross-threadgroup reduction of a K split, shared by the split decode
// kernels. Each of the `splits` threadgroups of one output tile stores its
// fp32 partials, in its kernel's own layout, through a device
// coherent(device) pointer and then arrives at the tile's counter. The last
// one to arrive adds the splits in split order, its own from registers, and
// returns the counter to zero for the next dispatch. The result does not
// depend on scheduling, and no threadgroup waits for another.

// True in every thread of the threadgroup that arrives last. A store reaches
// another threadgroup's load only between two device-scope fences, the
// storing thread's after it and the loading thread's before it: a fence
// orders the accesses of the thread that runs it, and a barrier's memory
// flags those of its own threadgroup alone. So every thread fences its
// partials, the first barrier holds thread zero's arrival until all have,
// and every thread of the threadgroup whose arrival counted all the others
// fences again before it reads theirs. With no fence, half of 8e10 reads of
// another split's partials were of an earlier dispatch's (32-core M5 Max,
// tiles of 32 to 128 threads in 2 and 8 splits); with these, none, as with
// thread zero's alone, which Metal does not promise the other threads.
inline bool split_arrive_last(device atomic_uint *counter, uint splits,
                              uint thread_index, threadgroup uint *arrival) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  threadgroup_barrier(mem_flags::mem_none);
  if (thread_index == 0)
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (*arrival != splits - 1)
    return false;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  return true;
}

// split_arrive_last for a threadgroup of which one simdgroup alone stores the partials and arrives, the others having
// returned (the column tile, kernels/decode/linear_gguf_column.metal): the same fences around the arrival, the
// simdgroup's barrier for the threadgroup's, and lane 0's arrival broadcast to the simdgroup.
inline bool split_arrive_last_simdgroup(device atomic_uint *counter, uint splits, uint lane) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  simdgroup_barrier(mem_flags::mem_none);
  uint arrival = 0;
  if (lane == 0)
    arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
  if (simd_broadcast(arrival, 0) != splits - 1)
    return false;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                      thread_scope_device);
  return true;
}

// The splits' partials added in split order: `own` for split `self`,
// `partial(s)` for every other split s.
template <typename T, typename Partial>
inline T split_sum(T own, uint self, uint splits, Partial partial) {
#pragma clang fp reassociate(off)
  T total = T(0);
  for (uint s = 0; s < splits; ++s)
    total += s == self ? own : partial(s);
  return total;
}

// Returns the tile's counter to zero; called by the reducing threadgroup.
inline void split_release(device atomic_uint *counter, uint thread_index) {
  if (thread_index == 0)
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

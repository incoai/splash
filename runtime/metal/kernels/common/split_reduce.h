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

// True in every thread of the threadgroup that arrives last.
inline bool split_arrive_last(device atomic_uint *counter, uint splits,
                              uint thread_index, threadgroup uint *arrival) {
  // Every writer publishes its partials before thread zero signals arrival.
  // The device-scope fences pair through the counter.
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  return *arrival == splits - 1;
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

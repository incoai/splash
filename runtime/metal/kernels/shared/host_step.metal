// The system memory scope, which Metal's public headers leave out: the only
// fence under which a running kernel sees a store of the CPU. Device-scope atomics, volatile and coherent(device)
// loads never do (M5 Max, M5 Pro, M3 Max).
#pragma METAL internals : enable
#include <metal_stdlib>

using namespace metal;

constexpr constant thread_scope kSystemScope = static_cast<thread_scope>(3);

// Keeps the GPU awake through a host step (metal/MetalBackend.mm,
// encodeHostStep): one thread spins until the runner's release counter
// reaches release.x, at most release.y loads, and sees the runner's store
// within 2 us; it then writes release.x to `seen`, from which the backend
// learns that the spin ends at its release (checkHostStepSpin). The buffer
// after it still waits for the step's event, which the runner raises before
// the counter: if the spin ends at its bound, the GPU only idles as it would
// without it.
kernel void host_step_wait(volatile coherent(system) device const uint *released [[buffer(0)]],
                           constant uint2 &release [[buffer(1)]], device uint *seen [[buffer(2)]]) {
  for (uint n = 0; n < release.y; ++n) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, kSystemScope);
    if (as_type<int>(released[0] - release.x) >= 0) {
      seen[0] = release.x;
      return;
    }
  }
}

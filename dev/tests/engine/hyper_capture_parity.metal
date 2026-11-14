// Test library only: compile the actual production reference kernels unchanged.
#include "metal/kernels/shared/hyper_connection.metal"

template <bool WrongOrder>
inline void capture_candidate(device float *streams, device const bfloat *branch,
                              device const float *weights, device bfloat *capture,
                              constant HyperStreamMeanParams &p, uint2 group, uint tid) {
  const uint row = group.x, d = group.y * HYPER_CONNECTION_ROW_THREADS + tid;
  if (row >= p.rows || d >= p.hidden) return;
  device float *x = streams + ulong(row) * p.streams * p.hidden + d;
  const float b = float(branch[ulong(row) * p.hidden + d]);
  float updated[HYPER_CONNECTION_MAXIMUM_STREAMS];
  float sum = 0.0f;
  for (uint s = 0; s < p.streams; ++s) {
    const float v = x[s * p.hidden] + b * weights[row * p.streams + s];
    x[s * p.hidden] = v; // Preserve FP32 state, not just its BF16 capture.
    updated[s] = v;
    sum += v;
  }
  // Deliberately incorrect reduction: the fixture must detect this alternative.
  if constexpr (WrongOrder) if (p.streams == 4) sum = (updated[0] + updated[2]) + (updated[1] + updated[3]);
  capture[ulong(p.row + row) * p.stride + p.column + d] = bfloat(sum / float(p.streams));
}
#define CAPTURE_KERNEL(Name, Wrong) \
  kernel void Name(device float *streams [[buffer(0)]], device const bfloat *branch [[buffer(1)]], \
                   device const float *weights [[buffer(2)]], device bfloat *capture [[buffer(3)]], \
                   constant HyperStreamMeanParams &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], \
                   uint tid [[thread_index_in_threadgroup]]) { capture_candidate<Wrong>(streams, branch, weights, capture, p, group, tid); }
CAPTURE_KERNEL(test_hyper_combine_capture, false)
CAPTURE_KERNEL(test_hyper_capture_wrong_order, true)

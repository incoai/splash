#include "metal/abi/KernelABI.h"
#include "metal/abi/LogitFinish.h"

#include <metal_stdlib>

using namespace metal;

// The finish of a head's logits (metal/abi/LogitFinish.h), in place, a
// thread a logit: threadgroup (column group, row). tanh is the precise one,
// as the softcap's fp32 reference computes it.
kernel void logit_finish(device float *logits [[buffer(0)]], constant LogitFinishParams &p [[buffer(1)]],
                         uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint token = group.x * LOGIT_FINISH_THREADS + tid, row = group.y;
  if (token >= p.vocabulary || row >= p.rows) return;
  device float &logit = logits[ulong(row) * p.vocabulary + token];
  if (token >= p.tokens)
    logit = -INFINITY;
  else if (p.softcap != 0.0f)
    logit = p.softcap * precise::tanh(logit / p.softcap);
}

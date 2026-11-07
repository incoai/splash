#include <metal_stdlib>
#include "metal/abi/ExecutionGeometry.h"

// Metadata marker: the backend looks up this name and never dispatches it.
#if SPLASH_MAXIMUM_BATCH_WIDTH == 8u
kernel void splash_batch_geometry_8(device uint *target [[buffer(0)]]) { target[0] = 8; }
#else
kernel void splash_batch_geometry_4(device uint *target [[buffer(0)]]) { target[0] = 4; }
#endif

using namespace metal;

// Metal applies a residency set's endResidency at its next GPU operation.
// When the keep-alive lapses, the backend dispatches this kernel as
// that operation (metal/Residency.hpp): its pipeline is built with the
// library, so ending residency compiles nothing.
kernel void residency_kick(device uint *target [[buffer(0)]]) {
  target[0] = 0;
}

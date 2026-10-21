#include <metal_stdlib>

using namespace metal;

// Metal applies a residency set's endResidency at its next GPU operation.
// When the keep-alive lapses, the backend dispatches this kernel as
// that operation (metal/Residency.hpp): its pipeline is built with the
// library, so ending residency compiles nothing.
kernel void residency_kick(device uint *target [[buffer(0)]]) {
  target[0] = 0;
}

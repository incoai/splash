#pragma once

// Shader prelude of the affine Q4 kernels. The ABI headers below also
// compile as host C++; their shared parameter layouts use fixed-width
// scalars and explicit padding.

#include "metal/abi/Embedding.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Linear.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

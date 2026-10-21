#pragma once

// Shader prelude. The ABI headers below also compile as host C++; their
// shared parameter layouts use fixed-width scalars and explicit padding.

#include "metal/abi/AneFfn.h"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/Embedding.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "metal/abi/MoE.h"
#include "metal/abi/PagedAttention.h"
#include "metal/abi/RoPE.h"
#include "metal/abi/RowCopy.h"
#include "metal/abi/Sampling.h"
#include "metal/abi/Vision.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

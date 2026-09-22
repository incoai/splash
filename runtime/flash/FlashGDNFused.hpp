#pragma once

#include "FlashGDN.hpp"

namespace splash::flash {

enum class FlashGDNFusion : uint8_t {
  // Parallel convolution/L2/gates, then the qualified recurrence/RMS/carry.
  Prepare,
  // Head-owned persistent recurrence and RMS; final history carry remains
  // separate to protect shared q/k history from cross-threadgroup races.
  PersistentHead,
  PersistentHead512,
  PersistentHead1024,
};

// Experimental routes with exactly the qualified GDN data and state ABI.
// The default addGDN implementation remains unchanged. Every scratch output
// is retained so the independent staged oracle can compare the routes.
void addGDNFused(metal::CommandGraph &graph, const FlashGDNWeights &weights,
                 const FlashGDNBuffers &buffers, const FlashGDNState &state,
                 uint32_t rows, uint32_t lanes, FlashGDNFusion fusion,
                 float normEpsilon = 1e-6f);

struct FlashGDNCapture final {
  // F32[lane,row,48,value128,key128], snapshots after each consumed row.
  metal::MetalBuffer recurrentStates;
  // Zero chooses tight row bytes; zero lane stride chooses rows*rowStride.
  uint64_t rowStrideBytes = 0;
  uint64_t laneStrideBytes = 0;
  // Default captures all rows. Explicit rows-1 omits the full final prefix,
  // which remains in live state. Explicit0 permits an empty tape buffer.
  uint32_t capturedRows = 0xffffffffu;
  uint32_t threadsPerHead = 256;
};

// Persistent-head route with exact recurrent prefixes, bounded rows1..16.
// Convolution history still commits only the full prefix; shorter histories
// can be reconstructed from the incoming3rows and qkv projected inputs.
void addGDNFusedCaptured(metal::CommandGraph &graph,
                         const FlashGDNWeights &weights,
                         const FlashGDNBuffers &buffers,
                         const FlashGDNState &state,
                         const FlashGDNCapture &capture, uint32_t rows,
                         uint32_t lanes = 1, float normEpsilon = 1e-6f);

} // namespace splash::flash

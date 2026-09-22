#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::flash {

// Exact integer/bit-copy preprocessing for the checkpoint's E512, H2560,
// top-10 MoE. Counts and offsets describe valid IDs only; stable routeMap is
// sorted by expert, then ascending original flat route (row*selections+slot).
// Invalid tail routeMap entries are UINT32_MAX and packed input rows are zero.
// Duplicate valid IDs remain in the map, but set diagnostic bit 1.
// All scratch allocations are retained by MetalBuffer ownership and graphs.
// Inputs, diagnostics, and every scratch plane must be disjoint. Host guards
// detect all Shared view overlaps and exact Private aliases; differently
// offset Private aliases cannot be inspected through the current backend API.
struct FlashMoEBucketScratch {
  metal::MetalBuffer counts;       // U32[512]
  metal::MetalBuffer offsets;      // U32[513], valid route total at [512]
  metal::MetalBuffer routeMap;     // U32[rowCapacity*selectionCapacity]
  metal::MetalBuffer canonicalToPacked; // U32[routes], UINT32_MAX for invalid IDs
  metal::MetalBuffer packedInputs; // BF16[routes,2560]
  metal::MetalBuffer jobOffsets;   // U32[513], job total at [512]
  metal::MetalBuffer jobCount;     // U32[1]
  metal::MetalBuffer tileJobs;     // FlashMoEBucketJob[jobCapacity]
  uint32_t rowCapacity = 0;
  uint32_t selectionCapacity = 0;
  uint32_t routeCapacity = 0;
  uint32_t jobCapacity = 0;
};

// Safe fixed launch bound for GPU-generated jobs. No counts are read on host.
// Throws for unsupported geometry (rows 1..8192, selections 1..10, M8/16/32/64).
[[nodiscard]] uint32_t moEBucketJobCapacity(uint32_t rows,
                                          uint32_t selections,
                                          uint32_t tileRows);

[[nodiscard]] FlashMoEBucketScratch allocateMoEBucketScratch(
    metal::MetalBackend &backend, uint32_t rows, uint32_t selections = 10,
    metal::BufferStorage storage = metal::BufferStorage::Shared);

// Adds histogram, exclusive prefix, stable map, and BF16 gather dispatches to
// ONE graph. Input BF16[rows,2560], IDs I64[rows,selections]. Diagnostics is a
// caller-cleared sticky U32 word: 1 invalid/duplicate ID, 2 malformed shape,
// 4 nonfinite BF16 hidden input. Neither IDs nor input are modified.
void addMoEBucketPack(metal::CommandGraph &graph, metal::MetalBuffer input,
                      metal::MetalBuffer expertIDs,
                      const FlashMoEBucketScratch &scratch,
                      metal::MetalBuffer diagnostics, uint32_t rows,
                      uint32_t selections = 10);

// Adds GPU job prefix and job emission dispatches. Jobs cover each nonempty
// expert bucket in tileRows chunks and never cross an expert boundary. Launch
// matrix work with moEBucketJobCapacity(rows,selections,tileRows) groups and
// guard each group against jobCount. row_begin is an absolute packed row;
// valid rows in a job are min(tileRows, offsets[expert+1]-row_begin).
void addMoEBucketJobs(metal::CommandGraph &graph,
                      const FlashMoEBucketScratch &scratch,
                      metal::MetalBuffer diagnostics, uint32_t rows,
                      uint32_t tileRows, uint32_t selections = 10);

} // namespace splash::flash

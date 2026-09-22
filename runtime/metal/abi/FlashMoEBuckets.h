#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Physical rows may combine four independent 2,048-token prefill lanes.
// Keep the host and every bucket/blocked staging shader on the same bound.
enum : uint32_t {
  kFlashMoEBucketMaximumRows = 8192,
  kFlashMoEBucketMaximumSelections = 10,
};

// All integer extents are host-checked, then checked again by every kernel.
// Pack dispatches set tile_rows and job_capacity to zero. Job dispatches use
// tile_rows 8, 16, 32, or 64; job_capacity is ceil(routes/tile_rows) + 511.
struct FlashMoEBucketParams {
  uint32_t rows;
  uint32_t selections;
  uint32_t width;
  uint32_t experts;
  uint32_t routes;
  uint32_t tile_rows;
  uint32_t job_capacity;
  uint32_t reserved;
};

// One matrix tile job: expert ID and absolute first row in packedInputs.
// Consumers must check jobIndex < jobCount before dereferencing any fields.
// Unused jobs contain {UINT32_MAX, 0} for additional fail-closed protection.
struct FlashMoEBucketJob {
  uint32_t expert;
  uint32_t row_begin;
};

static_assert(sizeof(FlashMoEBucketParams) == 32,
              "Flash MoE bucket parameters are eight uint32 values");
static_assert(sizeof(FlashMoEBucketJob) == 8,
              "Flash MoE matrix job is an expert and packed row pair");

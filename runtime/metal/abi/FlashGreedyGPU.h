#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

enum : uint32_t {
  kFlashGreedyGPUMaximumRows = 16,
  kFlashGreedyGPUMaximumLanes = 4,
  kFlashGreedyGPUValuesPerPartition = 2048,
  kFlashGreedyGPUErrorNonfinite = 1,
  kFlashGreedyGPUErrorInputToken = 2,
  kFlashGreedyGPUErrorBudget = 4,
  kFlashGreedyGPUFinishNone = 0,
  kFlashGreedyGPUFinishStop = 1,
  kFlashGreedyGPUFinishLength = 2,
};

struct FlashGreedyGPUParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t row_stride;
  uint32_t partitions;
  uint32_t lanes;
  uint32_t rows_per_lane;
  uint32_t active_lane_mask;
  uint32_t reserved;
};

// Integer ordering, rather than floating-point reduction, preserves every
// finite BF16 value, including subnormals and the equivalence of signed zeros.
struct FlashGreedyGPURowResult {
  uint32_t token;
  uint32_t rank;
  uint32_t errors;
  uint32_t reserved;
};

// Only this fixed-size record needs host visibility after target verification.
// matched_drafts precedes EOS/quota truncation, matching the CPU controller.
// retained_rows is also the exact number of target input rows to commit.
struct FlashGreedyGPUPrefixResult {
  uint32_t output[kFlashGreedyGPUMaximumRows];
  uint32_t predictions[kFlashGreedyGPUMaximumRows];
  uint32_t matched_drafts;
  uint32_t retained_rows;
  uint32_t finish;
  uint32_t errors;
};

static_assert(sizeof(FlashGreedyGPUParams) == 32);
static_assert(sizeof(FlashGreedyGPURowResult) == 16);
static_assert(sizeof(FlashGreedyGPUPrefixResult) == 144);

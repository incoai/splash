#pragma once
#include "FlashMoEBlocked.h"

// Signed coefficient codes from a separately versioned offline store.
struct FlashInt8ExpertStoreParams {
  uint32_t rows, selections, route_capacity, job_capacity;
  uint32_t tile_rows, stored_experts, scale_group_size, reserved;
};
struct FlashInt8ExpertStoreGateParams {
  FlashMoEBlockedGateParams blocked;
  uint32_t stored_experts, flags, reserved0, reserved1;
};
struct FlashInt8ExpertStoreDownParams {
  FlashMoEBlockedDownParams blocked;
  uint32_t stored_experts, flags, reserved0, reserved1;
};
struct FlashInt8ExpertStoreSanitizeParams {
  uint32_t route_capacity, width, reserved0, reserved1;
};
static_assert(sizeof(FlashInt8ExpertStoreParams) == 32);
static_assert(sizeof(FlashInt8ExpertStoreGateParams) == 128);
static_assert(sizeof(FlashInt8ExpertStoreDownParams) == 96);
static_assert(sizeof(FlashInt8ExpertStoreSanitizeParams) == 16);

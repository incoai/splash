#pragma once
#include "FlashMoEBlocked.h"
#include "FlashAffine.h"

struct FlashExpertCacheConvertParams {
  FlashAffineParams affine;
  uint32_t cached_experts;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};
struct FlashExpertCacheGateParams {
  FlashMoEBlockedGateParams blocked;
  uint32_t cached_experts;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};
struct FlashExpertCacheDownParams {
  FlashMoEBlockedDownParams blocked;
  uint32_t cached_experts;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};
static_assert(sizeof(FlashExpertCacheConvertParams) == 80);
static_assert(sizeof(FlashExpertCacheGateParams) == 128);
static_assert(sizeof(FlashExpertCacheDownParams) == 96);

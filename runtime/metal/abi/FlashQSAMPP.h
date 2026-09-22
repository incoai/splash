#pragma once
#include "FlashQSA.h"

// Attention-only MPP route. Prepared BF16 queries/caches and the authoritative
// FP32-selected chronological block IDs are consumed without modification.
struct FlashQSAMPPParams {
  FlashQSAParams common;
  uint32_t score_tile;
  uint32_t reserved[3];
};
static_assert(sizeof(FlashQSAMPPParams) == 64,
              "Flash QSA MPP parameters have a stable 64-byte ABI");

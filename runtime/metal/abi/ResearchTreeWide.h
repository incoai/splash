#pragma once
#include "metal/abi/Sampling.h"
// Prefix is the already-oracled16current/8pending GDN fork descriptor.
struct ResearchTreeWideState {
 int32_t parents[16];
 uint32_t snapshot_at[16],restore_at[16],path[8];
 uint32_t depth[16],node_count,leaf_count,leaves[4];
 uint8_t row_leaf[16];
};
static_assert(sizeof(ResearchTreeWideState)==328,"wide research ABI");

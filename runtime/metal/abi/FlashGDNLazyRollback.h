#pragma once

#include "metal/abi/FlashGDN.h"

// Private experiments only. Verification stores one initial F32 snapshot and
// writes prepared operands directly to the caller's lane-major tape. The
// snapshot can have independently padded lane strides; convolution snapshots
// are tightly packed 3 * 10240 BF16 values (61440 bytes per lane).
struct FlashGDNLazyVerifyParams {
  FlashGDNParams gdn;
  uint64_t snapshot_lane_stride_bytes;
};

// gdn.rows remains the original verification window, including during replay.
// retained == gdn.rows masks a fully accepted lane before all state/tape reads.
// Otherwise replay restores the retained prefix from the initial snapshot.
struct FlashGDNLazyReplayParams {
  FlashGDNParams gdn;
  uint32_t retained[4];
  uint64_t snapshot_lane_stride_bytes;
};

struct FlashGDNLazyCopyParams {
  uint64_t bytes;
};

static_assert(sizeof(FlashGDNLazyVerifyParams) == 56,
              "Private lazy GDN verify ABI must match Metal");
static_assert(sizeof(FlashGDNLazyReplayParams) == 72,
              "Private lazy GDN replay ABI must match Metal");
static_assert(sizeof(FlashGDNLazyCopyParams) == 8,
              "Private lazy GDN copy ABI must match Metal");

// private_gdn_lazy_verify_sg16: original persistent buffers 0..15;
// initial F32 snapshot 16; FlashGDNLazyVerifyParams 17. Grid (48, lanes, 1),
// threads (512, 1, 1), SIMD width 32.
// private_gdn_lazy_replay_sg16: mixed 0, decay 1, beta 2, initial snapshot 3,
// live recurrent 4, diagnostics 5, FlashGDNLazyReplayParams 6. Same grid/threads.
// private_gdn_lazy_restore_convolution: initial tight history 0,
// saved lane-major raw QKV 1, live padded history 2, diagnostics 3,
// FlashGDNLazyReplayParams 4. Grid (40, lanes, 1), threads (256, 1, 1).
// private_gdn_lazy_copy: byte input 0, byte output 1, FlashGDNLazyCopyParams 2.
// Grid (ceil(bytes / 256), 1, 1), threads (256, 1, 1).

#pragma once

// The engine's execution constants on an M5 (Apple10 GPU), in one place; they are not request options. Tile shapes the
// kernels are compiled with live in the ABI headers (metal/abi/video/*.h) and are re-exported here. They were tuned on
// 768p clips (1,008 rows per latent frame); none is a shape: every dispatch takes its counts from the request's rows,
// tiles and frames, so a 480p canvas (390 or 405 rows per latent frame) runs on them as they are. The ones a 480p
// tuning pass should revisit are marked "768p-tuned".
#include "metal/abi/video/LinearABI.h"
#include "metal/abi/video/QkFixABI.h"
#include "metal/abi/video/VaeABI.h"

#include <cstddef>
#include <cstdint>

namespace splash::video::device {

struct Tile {
  int32_t tm, tn, nsg;
};

// ---- DiT --------------------------------------------------------------------------------------------------------
// Rows per GEMM / producer dispatch: larger M is split so every in-kernel element offset stays below 2^31.
inline constexpr int32_t kMaxRowsPerDispatch = 32768;
inline constexpr Tile kLinearTile{H3_LINEAR_TM, H3_LINEAR_TN, H3_LINEAR_NSG};  // qkv, out, to_out_linear, fc1, fc2
inline constexpr int32_t kQkvSwizzle = 8;       // QKV GEMM tile-order swizzle (M-blocks per group)
inline constexpr int32_t kResidualSwizzle = 0;  // out / to_out_linear / fc2
inline constexpr int32_t kQkFixTileRows = H3_QKFIX_TM;  // fp16 QKV GEMM of the QKFIX blocks
inline constexpr float kNormEps = 1e-5f;             // RMSNorm / QK-norm
inline constexpr float kBranchEps = 1e-6f;           // linear-branch readout RMSNorm
// Attention preparation: key-mean partial chunks, keys per int8 key-scale block
inline constexpr int32_t kKeyMeanChunks = 64;
inline constexpr int32_t kKeyBlock = H3_ATTENTION_KEYS;
// Window attention: 64-row query groups whose 16-row blocks all see >= kDenseKeyMin keys are split into chunks of
// kKeyChunk keys (partials + combine), all 56 heads in one dispatch, dense items after the window items.
// 768p-tuned: a request of fewer than kDenseKeyMin rows (768p 3 s; 480p up to 15 s, about 44,600 at 864 x 480) splits
// none and runs its dense groups as one item each over every key.
inline constexpr int32_t kKeyChunk = 8192;
inline constexpr int32_t kDenseKeyMin = 65536;
inline constexpr int32_t kHeadsPerWindowDispatch = H3_DIT_HEADS;
inline constexpr bool kDenseItemsLast = true;

// ---- SSD weight streaming ---------------------------------------------------------------------------------------
// The DiT's block ring (a 15 s clip fits in about 15 GB instead of 32 GB resident, at the same speed) and the fp16
// QKFIX weights' ring. 768p-tuned: a 480p block computes over 0.39x the rows of a 768p one while its weights read
// the same bytes, so the reads may no longer hide behind the compute.
inline constexpr int32_t kStreamSlots = 8;
inline constexpr int32_t kStreamQkFixSlots = 2;

// ---- Video VAE --------------------------------------------------------------------------------------------------
inline constexpr int32_t kVaeBatch = 7;  // tile-clips per layer dispatch (768p-tuned: 28 tiles = 4 x 7; 480p 12 / 15)
inline constexpr Tile kVaeQkvTile{H3_VAE_QKV_TM, H3_VAE_TN, H3_VAE_NSG};
inline constexpr Tile kVaeTile{H3_VAE_TM, H3_VAE_TN, H3_VAE_NSG};  // out, fc1 (GLU), fc2, proj_out
inline constexpr int32_t kVaeSwizzle = 0;
inline constexpr int32_t kVaeAttentionSimdgroups = H3_VAE_ATTENTION_NSG;
// Producer groups per thread: K = 2048 -> 2, K = 8192 -> 4
inline constexpr int32_t kVaeProducerGroups = 2;
inline constexpr int32_t kVaeFfnProducerGroups = 4;

// ---- Video VAE encoder (FL2VA keyframes) ---------------------------------------------------------------------
inline constexpr int32_t kVaeEncodeBatch = 7;           // 256 px tiles per command buffer (768p-tuned, as kVaeBatch)
inline constexpr Tile kVaeEncodeTileWide{64, 128, 4};   // convolutions with >= kVaeEncodeWideRows GEMM rows
inline constexpr Tile kVaeEncodeTileNarrow{32, 128, 4};
inline constexpr Tile kVaeEncodeTileOut{32, 64, 2};     // conv_out (48 channels padded to 64)
inline constexpr int32_t kVaeEncodeWideRows = 8192;
inline constexpr int32_t kVaeEncodeVideoTiles = 1;      // 17-frame tile-clips per command buffer (Ref2VA videos)

// ---- TAEH3 preview ----------------------------------------------------------------------------------------------
inline constexpr int32_t kPreviewChunk = 5;  // latent frames per command
inline constexpr Tile kPreviewTile256{32, 128, 4};  // conv tiles per channel level
inline constexpr Tile kPreviewTile128{64, 128, 4};
inline constexpr Tile kPreviewTile64{32, 64, 2};
inline constexpr Tile kPreviewTileOut{32, 16, 1};

} // namespace splash::video::device

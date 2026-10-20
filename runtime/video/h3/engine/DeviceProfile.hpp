#pragma once

// M5 Pro (48 GB, macOS 27) execution constants: every former tuning environment variable, hard-set in one place.
// Values were chosen by the sweep / tune-1 / kpush / diet / stream / taeh3 measurements and are bit-identical
// alternatives of each other where marked; they are not request options. Tile shapes the kernels are compiled with
// live in the ABI headers (metal/abi/*.h) and are re-exported here.
#include "metal/abi/video/LinearABI.h"
#include "metal/abi/video/QkFixABI.h"
#include "metal/abi/video/VaeABI.h"

#include <cstddef>
#include <cstdint>

namespace h3::device {

struct Tile {
  int32_t tm, tn, nsg;
};

// ---- DiT --------------------------------------------------------------------------------------------------------
// Rows per GEMM / producer dispatch: larger M is split so every in-kernel element offset stays below 2^31.
inline constexpr int32_t kMaxRowsPerDispatch = 32768;
inline constexpr Tile kLinearTile{H3_LINEAR_TM, H3_LINEAR_TN, H3_LINEAR_NSG};  // qkv, out, to_out_linear, fc1, fc2
inline constexpr int32_t kQkvSwizzle = 8;       // QKV GEMM tile-order swizzle (M-blocks per group); was LSWZ_QKV
inline constexpr int32_t kResidualSwizzle = 0;  // out / to_out_linear / fc2 (LSWZ_OUT, LSWZ_FC2 unset)
// fc1 + SwiGLU: bf16 h without a row-absmax side output, barrier before the K loop, swizzle 4 (FC1_MODE=plainxs4;
// compiled into dit_fc1_swiglu)
inline constexpr int32_t kQkFixTileRows = H3_QKFIX_TM;  // fp16 QKV GEMM of the QKFIX blocks (QKFIX_TM)
inline constexpr float kNormEps = 1e-5f;             // RMSNorm / QK-norm
inline constexpr float kBranchEps = 1e-6f;           // linear-branch readout RMSNorm
// Attention preparation: key-mean partial chunks, keys per int8 key-scale block
inline constexpr int32_t kKeyMeanChunks = 64;
inline constexpr int32_t kKeyBlock = H3_ATTENTION_KEYS;
// Window attention: 64-row query groups whose 16-row blocks all see >= kDenseKeyMin keys are split into chunks of
// kKeyChunk keys (partials + combine), all 56 heads in one dispatch, dense items after the window items
// (VDN_KC, VDN_KC_MIN, VDN_HB=56, VDN_DENSE_LAST=1).
inline constexpr int32_t kKeyChunk = 8192;
inline constexpr int32_t kDenseKeyMin = 65536;
inline constexpr int32_t kHeadsPerWindowDispatch = H3_DIT_HEADS;
inline constexpr bool kDenseItemsLast = true;
// QKFIX: post-RoPE dims kept in fp32 (manifest split, -1 padded) and fp16 V from the fp16 GEMM (QKFIX_V=1),
// register-held Q (QKFIX_QREG=1), exact split term on (QKFIX_SPLIT=1)
inline constexpr int32_t kQkFixSplitDims = H3_QKFIX_SPLIT;
// Command queue depth (one command buffer per block step, ~1,400 per forward)
inline constexpr int32_t kDitQueueDepth = 4096;

// ---- SSD weight streaming ---------------------------------------------------------------------------------------
inline constexpr int32_t kStreamQkFixSlots = 2;     // fp16 QKFIX weights ring (H3_STREAM_QKF_SLOTS)
inline constexpr int32_t kStreamReadThreads = 4;    // parallel preads per tensor (H3_STREAM_THREADS)
inline constexpr size_t kStreamChunkBytes = 8u << 20;  // pread size (H3_STREAM_CHUNK_MB)
inline constexpr bool kStreamNoCache = true;         // F_NOCACHE on the image (H3_STREAM_NOCACHE)
inline constexpr size_t kStreamVerifyBytes = 16384;  // image vs block files: first / last bytes of every streamed
                                                     // tensor compared at load (H3_STREAM_VERIFY=1)

// ---- Video VAE --------------------------------------------------------------------------------------------------
inline constexpr int32_t kVaeBatch = 7;  // tile-clips per command buffer (H3VAE_BATCH)
inline constexpr Tile kVaeQkvTile{H3_VAE_QKV_TM, H3_VAE_TN, H3_VAE_NSG};
inline constexpr Tile kVaeTile{H3_VAE_TM, H3_VAE_TN, H3_VAE_NSG};  // out, fc1 (GLU), fc2, proj_out
inline constexpr int32_t kVaeSwizzle = 0;
inline constexpr int32_t kVaeAttentionSimdgroups = H3_VAE_ATTENTION_NSG;
inline constexpr int32_t kVaeQueueDepth = 1024;
// Producer groups per thread: K = 2048 -> 2, K = 8192 -> 4
inline constexpr int32_t kVaeProducerGroups = 2;
inline constexpr int32_t kVaeFfnProducerGroups = 4;

// ---- TAEH3 preview ----------------------------------------------------------------------------------------------
inline constexpr int32_t kPreviewChunk = 5;  // latent frames per command buffer (TAEH3_CHUNK)
inline constexpr Tile kPreviewTile256{32, 128, 4};  // conv tiles per channel level (TAE_T256 / T128 / T64 / TFIN)
inline constexpr Tile kPreviewTile128{64, 128, 4};
inline constexpr Tile kPreviewTile64{32, 64, 2};
inline constexpr Tile kPreviewTileOut{32, 16, 1};

} // namespace h3::device

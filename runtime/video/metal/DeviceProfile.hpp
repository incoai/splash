#pragma once

// The video engine's execution constants per GPU family, in one place; they are not request options. A Profile row
// holds every per-chip host constant the operators and stages read at run time (dispatch limits, swizzles, tiles,
// batch and chunk sizes, row-count thresholds) and the kernel each tunable operator dispatches (host names in
// splash.metallib). VideoRuntime selects one row at start (selectProfile) and passes it down; nothing reads a row
// through a global. Tile shapes the kernels are compiled with live in the ABI headers (metal/abi/video/*.h) and are
// re-exported here. The values were tuned on 768p clips (1,008 rows per latent frame); none is a shape: every dispatch
// takes its counts from the request's rows, tiles and frames, so a 480p canvas (390 or 405 rows per latent frame)
// runs on them as they are. The ones a 480p tuning pass should revisit are marked "768p-tuned".
//
// Selection at start: the row of the device's Apple GPU family, or the highest one at or below it. A family above
// every row takes the highest and is logged untuned; a family below every row throws.
//
// Not in the row (compile-time below): the constants that size a buffer in the activation plan or the weight stream
// or that are kernel formats, and the numerical epsilons,
// which are not chip tuning.
#include "metal/DeviceCapabilities.hpp"
#include "metal/abi/video/LinearABI.h"
#include "metal/abi/video/QkFixABI.h"
#include "metal/abi/video/VaeABI.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace splash::video::device {

struct Tile {
  int32_t tm, tn, nsg;
};

struct Profile {
  const char *name;         // "m5"
  uint32_t appleGpuFamily;  // MTLGPUFamilyAppleN of the row

  // ---- DiT --------------------------------------------------------------------------------------------------------
  // Rows per GEMM / producer dispatch: larger M is split so every in-kernel element offset stays below 2^31.
  int32_t maxRowsPerDispatch;
  Tile linearTile;          // the int8 GEMMs' (qkv, out, to_out_linear, fc1, fc2) and the LoRA down-projections'
  int32_t qkvSwizzle;       // QKV GEMM tile-order swizzle (M-blocks per group), int8 and QKFIX
  int32_t residualSwizzle;  // out / to_out_linear / fc2
  int32_t qkFixTileRows;    // fp16 QKV GEMM of the QKFIX blocks
  // Window attention: 64-row query groups whose 16-row blocks all see >= denseKeyMin keys are split into chunks of
  // keyChunk keys (partials + combine), dense items after the window items. 768p-tuned: a request of fewer than
  // denseKeyMin rows (768p 3 s; 480p up to 15 s, about 44,600 at 864 x 480) splits none and runs its dense groups as
  // one item each over every key.
  int32_t keyChunk;
  int32_t denseKeyMin;
  // The kernels (host names) each DiT operator dispatches: dispatched with linearTile ...
  const char *qkv, *qkvLora;            // ops::qkv, ops::qkvGate (int8 QKV GEMM; with a LoRA)
  const char *residual, *residualLora;  // ops::gatedResidual: out, to_out_linear (no LoRA) and fc2
  const char *fc1, *fc1Lora;            // ops::fc1SwiGlu
  const char *loraDown, *loraDownHalf;  // ops::loraDown (int8 rows), ops::loraDownHalf (fp16 rows)
  // ... with qkFixTileRows
  const char *qkfixQkv, *qkfixQkvLora;  // ops::qkfixQkv
  // ... and the attention kernels
  const char *windowAttention, *windowAttentionQkFix;  // ops::window, ops::windowQkFix
  const char *vsaAttention, *vsaAttentionDense, *vsaAttentionQkFix, *vsaAttentionQkFixDense;  // ops::vsaAttention

  // ---- Video VAE --------------------------------------------------------------------------------------------------
  int32_t vaeBatch;  // tile-clips per layer dispatch (768p-tuned: 28 tiles = 4 x 7; 480p 12 / 15)
  Tile vaeQkvTile;
  Tile vaeTile;  // out, fc1 (GLU), fc2, proj_out
  int32_t vaeSwizzle;
  int32_t vaeAttentionSimdgroups;
  // Producer groups per thread: K = 2048 -> 2, K = 8192 -> 4
  int32_t vaeProducerGroups;
  int32_t vaeFfnProducerGroups;
  const char *vaeQkv;        // with vaeQkvTile
  const char *vaeGemm;       // out and fc2 (residual epilogue), with vaeTile
  const char *vaeGlu;        // fc1 + SwiGLU, with vaeTile
  const char *vaeProj;       // proj_out, with vaeTile
  const char *vaeAttention;  // with vaeAttentionSimdgroups

  // ---- Video VAE encoder (FL2VA keyframes) ---------------------------------------------------------------------
  // The convolutions' kernels are named by their tile (vae_enc_<kind><tm>x<tn>x<nsg>).
  int32_t vaeEncodeBatch;    // 256 px tiles per command buffer (768p-tuned, as vaeBatch)
  Tile vaeEncodeTileWide;    // convolutions with >= vaeEncodeWideRows GEMM rows
  Tile vaeEncodeTileNarrow;
  Tile vaeEncodeTileOut;     // conv_out (48 channels padded to 64)
  int32_t vaeEncodeWideRows;
  int32_t vaeEncodeVideoTiles;  // 17-frame tile-clips per command buffer (Ref2VA videos)

  // ---- TAEH3 preview ----------------------------------------------------------------------------------------------
  // The convolutions' kernels are named by their tile (preview_<kind><tm>x<tn>x<nsg>).
  int32_t previewChunk;  // latent frames per command
  Tile previewTile256;   // conv tiles per channel level
  Tile previewTile128;
  Tile previewTile64;
  Tile previewTileOut;
};

// M5 (Apple10): the values and kernels the video engine was tuned with. A kernel name and its Tile must change
// together, because tile sizes are compiled into each kernel instance.
inline constexpr Profile kM5{
    .name = "m5",
    .appleGpuFamily = 10,
    .maxRowsPerDispatch = 32768,
    .linearTile = {H3_LINEAR_TM, H3_LINEAR_TN, H3_LINEAR_NSG},
    .qkvSwizzle = 8,
    .residualSwizzle = 0,
    .qkFixTileRows = H3_QKFIX_TM,
    .keyChunk = 8192,
    .denseKeyMin = 65536,
    .qkv = "dit_qkv",
    .qkvLora = "dit_qkv_lora_post",
    .residual = "dit_gated_residual",
    .residualLora = "dit_gated_residual_lora_post",
    .fc1 = "dit_fc1_swiglu",
    .fc1Lora = "dit_fc1_swiglu_lora_post",
    .loraDown = "dit_lora_down",
    .loraDownHalf = "dit_lora_down_f16",
    .qkfixQkv = "dit_qkfix_qkv",
    .qkfixQkvLora = "dit_qkfix_qkv_lora",
    .windowAttention = "dit_window_attention",
    .windowAttentionQkFix = "dit_qkfix_window_attention",
    .vsaAttention = "dit_vsa_attention",
    .vsaAttentionDense = "dit_vsa_attention_dense",
    .vsaAttentionQkFix = "dit_vsa_attention_qkfix",
    .vsaAttentionQkFixDense = "dit_vsa_attention_qkfix_dense",
    .vaeBatch = 7,
    .vaeQkvTile = {H3_VAE_QKV_TM, H3_VAE_TN, H3_VAE_NSG},
    .vaeTile = {H3_VAE_TM, H3_VAE_TN, H3_VAE_NSG},
    .vaeSwizzle = 0,
    .vaeAttentionSimdgroups = H3_VAE_ATTENTION_NSG,
    .vaeProducerGroups = 2,
    .vaeFfnProducerGroups = 4,
    .vaeQkv = "vae_qkv_i8",
    .vaeGemm = "vae_residual_i8",
    .vaeGlu = "vae_glu_i8",
    .vaeProj = "vae_proj_f16",
    .vaeAttention = "vae_attention",
    .vaeEncodeBatch = 7,
    .vaeEncodeTileWide = {64, 128, 4},
    .vaeEncodeTileNarrow = {32, 128, 4},
    .vaeEncodeTileOut = {32, 64, 2},
    .vaeEncodeWideRows = 8192,
    .vaeEncodeVideoTiles = 1,
    .previewChunk = 5,
    .previewTile256 = {32, 128, 4},
    .previewTile128 = {64, 128, 4},
    .previewTile64 = {32, 64, 2},
    .previewTileOut = {32, 16, 1},
};
// Every row, by family ascending.
inline constexpr const Profile *kProfiles[] = {&kM5};

// The row of a device: the highest at or below its Apple GPU family. Throws std::invalid_argument for a family below
// every row's.
[[nodiscard]] const Profile &selectProfile(const DeviceCapabilities &caps);
// Whether a device's family is above every row's (it runs the highest one untuned).
[[nodiscard]] bool untunedFamily(const DeviceCapabilities &caps);
// "H3 video profile <name> device <device name> family <N> cores <C>": the line the engine logs at start
// (VideoRuntime::stages).
[[nodiscard]] std::string profileLine(const Profile &profile, const DeviceCapabilities &caps);
// "untuned GPU family N: video profile <name>", logged before profileLine for a family above every row.
[[nodiscard]] std::string untunedFamilyLine(const Profile &profile, const DeviceCapabilities &caps);

// ---- Compile-time: not in the row ----------------------------------------------------------------------------------
// Numerical constants of the model, not chip tuning.
inline constexpr float kNormEps = 1e-5f;    // RMSNorm / QK-norm
inline constexpr float kBranchEps = 1e-6f;  // linear-branch readout RMSNorm
// Attention preparation: key-mean partial chunks (they size MemoryPlan's key-mean partials), keys per int8 key-scale
// block (the key-scale buffer's layout and the kernels' H3_ATTENTION_KEYS).
inline constexpr int32_t kKeyMeanChunks = 64;
inline constexpr int32_t kKeyBlock = H3_ATTENTION_KEYS;
// Window attention: all 56 heads in one dispatch (they size MemoryPlan's key-chunk partials).
inline constexpr int32_t kHeadsPerWindowDispatch = H3_DIT_HEADS;
// ---- SSD weight streaming ---------------------------------------------------------------------------------------
// The DiT's block ring (a 15 s clip fits in about 15 GB instead of 32 GB resident, at the same speed) and the fp16
// QKFIX weights' ring, which size WeightStream's slot rings. 768p-tuned: a 480p block computes
// over 0.39x the rows of a 768p one while its weights read the same bytes, so the reads may no longer hide behind the
// compute.
inline constexpr int32_t kStreamSlots = 8;
inline constexpr int32_t kStreamQkFixSlots = 2;

} // namespace splash::video::device

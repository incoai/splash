#pragma once

// MiniMax-H3 shapes shared by host code and kernels. The engine runs this one model on M5 Pro; nothing here is a
// tuning knob (those live in engine/DeviceProfile.hpp).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// DiT (VDN-H3 hybrid attention, 50 blocks)
#define H3_DIT_BLOCKS 50
#define H3_DIT_HIDDEN 5376
#define H3_DIT_FFN 14336
#define H3_DIT_HEADS 56
#define H3_DIT_HEAD_DIM 128
#define H3_DIT_INNER (H3_DIT_HEADS * H3_DIT_HEAD_DIM)  // 7168
#define H3_DIT_QKV (3 * H3_DIT_INNER)                   // 21504
#define H3_DIT_GATES 256        // [beta 56 | softmax gate 56 | zero 16 | output-gate down 128]
#define H3_DIT_ROPE 96          // cos 48 | sin 48 per row (bf16)
#define H3_DIT_VIDEO_CHANNELS 96  // patchified video latent (1 x 2 x 2 x 24)
#define H3_DIT_AUDIO_CHANNELS 32
#define H3_ADA_ROWS 6           // shift / scale / gate of attention and MLP, per modality
#define H3_MODALITIES 3

// Linear-branch state (per head: 128 x 128 fp32 matrices)
#define H3_BRANCH_STATE (H3_DIT_HEAD_DIM * H3_DIT_HEAD_DIM)

// Key-range table of the window softmax (dit/WindowTable): int32 records of 32 per 16-row query block, header first.
#define H3_TABLE_MAGIC 0x544B4456
#define H3_TABLE_RECORD 32
#define H3_TABLE_ROWS 16
#define H3_ATTENTION_KEYS 32    // keys per score tile (int8 key-scale block)
#define H3_ATTENTION_GROUP 64   // query rows per window-attention work item

// Video VAE decoder (ViT, 36 layers)
#define H3_VAE_LAYERS 36
#define H3_VAE_HIDDEN 2048
#define H3_VAE_FFN 8192
#define H3_VAE_HEADS 32
#define H3_VAE_HEAD_DIM 64
#define H3_VAE_PATCH_OUT 3072   // proj_out: 3 colours x 4 frames x 16 x 16 pixels
#define H3_VAE_LATENT 24
#define H3_VAE_TILE 16          // latent tile side (256 px)
#define H3_VAE_TILE_FRAMES 7    // latent frames per tile-clip
#define H3_VAE_TILE_ROWS (H3_VAE_TILE_FRAMES * H3_VAE_TILE * H3_VAE_TILE + 5)  // + 4 register tokens + 1 zero token
#define H3_VAE_MAX_TILES 16     // per axis

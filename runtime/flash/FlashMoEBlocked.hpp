#pragma once

#include "flash/FlashMoEBuckets.hpp"
#include "flash/FlashWeights.hpp"

namespace splash::flash {

// Explicit alternate numerical route: original F32 affine reconstruction
// rounds to BF16 matrix operands, MPP multiply-accumulate traverses K64 chunks,
// then dot outputs round to BF16 before the qualified compiled SwiGLU.
// This is not the canonical affine vector lane reduction. The model/cache
// identity and qualification must record the complete producer policy.
inline constexpr const char *kFlashMoEBlockedSemantics =
    "stable-gpu-expert-buckets-f32coef-bf16-mpp-mac-k64-col8-v1";
inline constexpr const char *kFlashMoEBlockedQ4x8Semantics =
    "stable-gpu-expert-buckets-f32coef-bf16-mpp-mac-k64-col8-aligned-q4x8-staging-v1";
inline constexpr const char *kFlashMoEBlockedQ4x8M64Semantics =
    "stable-gpu-expert-buckets-f32coef-bf16-mpp-mac-k64-col8-aligned-q4x8-staging-m64n64-sg8-rows4096plus-nohot-v2";
inline constexpr const char *kFlashMoEBlockedDirectASemantics =
    "stable-gpu-expert-buckets-f32coef-bf16-mpp-mac-k64-col8-q4x8-device-a-sanitized-globalpad63-v1";
inline constexpr const char *kFlashMoEBlockedDirectAM64Semantics =
    "stable-gpu-expert-buckets-f32coef-bf16-mpp-mac-k64-col8-q4x8-device-a-sanitized-globalpad63-m64n64-sg8-rows4096plus-nohot-v1";

// SPLASH_FLASH_MOE_Q4X8 defaults off, accepts only0/1, and freezes on first
// use. Its aligned Q4/G64 staging preserves the original MPP/BF16 arithmetic;
// a source view without provable U32 alignment retains the control route.
[[nodiscard]] const char *flashMoEBlockedRouteSemantics();

// Frozen opt-in, accepting only0/1 and requiring Q4X8=1. Direct device A
// keeps coefficient reconstruction, K64 MAC traversal and all BF16 boundaries.
[[nodiscard]] bool flashMoEDirectAEnabled();

// Exact subtotal for every allocation in allocateMoEBlockedScratch, rounded
// independently to the caller's allocator alignment. No GPU contents read.
[[nodiscard]] uint64_t flashMoEBlockedWorkspacePlannedBytes(
    uint32_t rows, uint32_t selections = 10, uint64_t alignment = 16384);
[[nodiscard]] uint64_t flashMoEDirectAWorkspaceExtraBytes(
    uint32_t routeCapacity, uint64_t alignment = 16384);

enum class FlashMoEBlockedTile : uint32_t {
  M8N64 = 8,
  M16N64 = 16,
  M32N64 = 32,
  M64N64 = 64,
};

// Frozen opt-in SPLASH_FLASH_MOE_M64 accepts only0/1 and requires Q4X8=1.
// A hot-expert cache retains its existing M16/M32 producers. M64 is selected
// only for at least4096 physical rows; smaller routes preserve prior geometry.
[[nodiscard]] FlashMoEBlockedTile flashMoEBlockedTile(uint32_t rows,
                                                    bool hasHotExpertCache);

struct FlashMoEBlockedScratch {
  FlashMoEBucketScratch buckets;
  metal::MetalBuffer packedActivated; // BF16[routes(+63 direct A),640]
  metal::MetalBuffer scatteredDown;   // BF16[rows,selections,2560]
};

[[nodiscard]] FlashMoEBlockedScratch allocateMoEBlockedScratch(
    metal::MetalBackend &backend, uint32_t rows, uint32_t selections = 10,
    metal::BufferStorage storage = metal::BufferStorage::Shared);

// Pack original BF16[rows,2560] by expert and create GPU tile jobs. No host
// counts read or per-token CPU expert loop occurs. The same M is used by gate,
// up, and down; source weights remain unchanged. IDs are I64[rows,selections].
void addMoEBlockedPack(metal::CommandGraph &graph, metal::MetalBuffer input,
                       metal::MetalBuffer expertIDs,
                       const FlashMoEBlockedScratch &scratch,
                       metal::MetalBuffer diagnostics, uint32_t rows,
                       FlashMoEBlockedTile tile, uint32_t selections = 10);

// GPU jobs select nonempty experts. Gate/up share each staged A tile and run
// two BF16 MPP accumulators; output is packed BF16[validRoutes,640]. Static
// matrix tiles mask each expert bucket's final rows and never cross buckets.
void addMoEBlockedGateUp(metal::CommandGraph &graph,
                         const FlashAffineProjection &gate,
                         const FlashAffineProjection &up,
                         const FlashMoEBlockedScratch &scratch,
                         metal::MetalBuffer diagnostics, uint32_t rows,
                         FlashMoEBlockedTile tile, uint32_t selections = 10);

// MPP expert-down consumes packed activation rows and scatters BF16 outputs
// into the original [row,selection,2560] order, using the stable route map.
// The caller then uses the unchanged canonical BF16 col8 addCombine.
void addMoEBlockedDownScatter(metal::CommandGraph &graph,
                              const FlashAffineProjection &down,
                              const FlashMoEBlockedScratch &scratch,
                              metal::MetalBuffer diagnostics, uint32_t rows,
                              FlashMoEBlockedTile tile,
                              uint32_t selections = 10);

} // namespace splash::flash

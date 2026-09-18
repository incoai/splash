#pragma once

// Parameters of the KV page copy shared by the host and the kernel.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// One dispatch moves whole pages of one attention layer between the layer's
// four page buffers and host-visible staging. Threadgroup s serves staging
// slot s and reads its table entry: the page to move and the direction, none
// for an idle slot. A layer's bytes sit at layer_offset inside the slot as key
// data, key scales, value data, value scales; every range is a multiple of 16
// bytes.
#define SPLASH_KV_COPY_NONE 0u
#define SPLASH_KV_COPY_TO_STAGING 1u
#define SPLASH_KV_COPY_TO_PAGE 2u

struct SplashKvCopySlot {
  uint32_t page;
  uint32_t direction;
};

struct SplashKvCopyParams {
  uint32_t data_bytes;
  uint32_t scale_bytes;
  uint32_t slot_bytes;
  uint32_t layer_offset;
  uint32_t physical_pages;
  uint32_t staging_slots;
};

static_assert(sizeof(SplashKvCopySlot) == 8, "KV copy slots are 8 bytes on both sides");
static_assert(sizeof(SplashKvCopyParams) == 24,
              "KV copy parameters are 24 bytes on both sides");

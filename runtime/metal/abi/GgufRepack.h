#pragma once

// Editing this file re-prepares every GGUF model.

// Weight preparation's repack of native GGUF rows into the MDGG0001 planes
// (kernels/shared/gguf_repack.metal), shared by the host executor
// (model/GgufPreparation.cpp) and the kernel. The host stages a chunk of
// rows in image order; one thread per (row, 32-wide K group) writes its
// planes into the output buffer: plane0 at 0, then plane1 and meta.
#include "metal/abi/QuantFormat.h"

struct GgufRepackParams {
  uint32_t rows;          // rows of the chunk (a multiple of QUANT_TILE_ROWS)
  uint32_t input_size;    // K of the chunk (a multiple of 256)
  uint32_t fmt;           // GGUF_FMT_*
  uint32_t src_row_bytes; // bytes per staged row
  uint32_t dst_plane1;    // byte offsets in the output buffer
  uint32_t dst_meta;
};
static_assert(sizeof(GgufRepackParams) == 24, "GGUF repack parameters are 24 bytes on both sides");

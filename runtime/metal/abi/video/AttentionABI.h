#pragma once

// Window-softmax attention of the DiT (int8 QK, fp16 PV): key mean / int8 preparation, the window kernel driven by
// the key-range table, and the merge of key-chunk partials. The QKFIX kernel (QkFixABI.h) shares WindowParams.
#include "metal/abi/video/Layout.h"

// dit_attention_key_mean_partial / dit_attention_key_mean / dit_attention_quantize
struct AttentionPrepParams {
  int32_t S, Spad, H, KB;  // KB: keys per int8 key-scale block (32)
  int32_t nch, flags;      // nch: key-mean partial chunks; flags bit0: Hadamard(128) on Q and K (always set)
  float qk_scale_log2;     // log2(e) / sqrt(128), folded into the Q row scales
};

// dit_window_attention / dit_qkfix_window_attention / dit_window_combine. Grid: nitems x hb threadgroups (items of
// head0 first). nkb: key blocks per head of the key scales; nslot: partial slots per head (key-chunked groups).
struct WindowParams {
  int32_t S, Spad, H, ldo;
  int32_t nkb, nitems, head0, hb;
  int32_t nqb, nslot, flags;  // flags bit0 (QKFIX only): no exact fp32 split-dimension term
  float scale;                // QKFIX only: log2(e) / sqrt(128)
};

// One work item: a 64-row query group, its key clip [kbeg, kend) and the partial slot (-1 = final output).
struct WindowItem {
  int32_t qg, kbeg, kend, slot;
};

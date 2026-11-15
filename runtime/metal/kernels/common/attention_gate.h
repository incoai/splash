#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/attention_sequence.h"

// The output of a full-attention layer at element `element` of a row of
// every query head (QHeads x HeadDim elements a row): row `sequence_row` of
// a dispatch's sequence, its head's attention row, staged as the sequence's
// queries are ([KV head][row][query head in group][dimension]), times the
// sigmoid of the head's gate in its packed row `packed` where the layer is
// Gated, and as it is where it is not, which reads no packed row
// (`packed_qkv` may be null).
template <uint QHeads, uint KHeads, uint HeadDim, bool Gated>
inline bfloat full_attention_output_value(device const bfloat *packed_qkv,
                                          device const bfloat *attention,
                                          constant SplashAttentionSequence &sequence,
                                          uint sequence_row, ulong packed, uint element) {
  constexpr uint QStride = 2 * HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint HeadsPerKV = QHeads / KHeads;
  uint query_head = element / HeadDim;
  uint dim = element % HeadDim;
  float gate_scale = 1.0f;
  if constexpr (Gated) {
    float gate = float(packed_qkv[packed * PackedStride + query_head * QStride + HeadDim + dim]);
    gate_scale = splash_sigmoid(gate);
  }
  uint kv_head = query_head / HeadsPerKV;
  uint local_head = query_head % HeadsPerKV;
  ulong attention_index =
      (((ulong(sequence.stage_row) + ulong(kv_head) * sequence.chunk_stride + sequence_row) * HeadsPerKV +
        local_head) *
           HeadDim +
       dim);
  if constexpr (!Gated)
    return attention[attention_index];
  return bfloat(float(attention[attention_index]) * gate_scale);
}

// The row a gate thread of group (element group, row, sequence) writes of a
// dispatch's output rows, and the packed row it reads (attention_sequence.h):
// row `row` of the sequence, written as its packed row, or with last_rows
// (SplashAttentionParams) as row s; none (`live` false) past the sequence's
// rows.
struct SplashGateRow {
  ulong packed;
  ulong output;
  bool live;
};

inline SplashGateRow splash_gate_row(device const SplashLaneRows *layout, constant SplashAttentionParams &params,
                                     uint row, uint s) {
  const SplashAttentionRows rows = splash_attention_rows(layout, params, s);
  const ulong packed = ulong(rows.first) + row;
  return {packed, params.last_rows ? ulong(s) : packed, row < rows.rows};
}

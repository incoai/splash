#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"

// The gated output of a full-attention layer at element `element` of a
// lane's rows of every query head: the head's attention row, staged `stride`
// rows per KV head ([KV head][row][query head in group][dimension]), times
// the sigmoid of the head's gate in the packed row. Lane `lane`'s rows start
// at packed row lane x lane_rows and its staging at KV head lane x KHeads; a
// prefill chunk is lane 0.
template <uint QHeads, uint KHeads>
inline bfloat full_attention_gate_value(device const bfloat *packed_qkv,
                                        device const bfloat *attention,
                                        uint lane, uint lane_rows, uint stride,
                                        uint element) {
  constexpr uint HeadDim = 256, QStride = 2 * HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint HeadsPerKV = QHeads / KHeads;
  uint row = element / (QHeads * HeadDim);
  uint remainder = element % (QHeads * HeadDim);
  uint query_head = remainder / HeadDim;
  uint dim = remainder % HeadDim;
  float gate = float(packed_qkv[(ulong(lane) * lane_rows + row) * PackedStride +
                                query_head * QStride + HeadDim + dim]);
  float gate_scale = splash_sigmoid(gate);
  uint kv_head = query_head / HeadsPerKV;
  uint local_head = query_head % HeadsPerKV;
  ulong attention_index =
      (((ulong(lane) * KHeads + kv_head) * stride + row) * HeadsPerKV +
       local_head) *
          HeadDim +
      dim;
  return bfloat(float(attention[attention_index]) * gate_scale);
}

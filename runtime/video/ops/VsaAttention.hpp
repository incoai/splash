#pragma once

// FastH3's Video Sparse Attention (kernels/video/dit/vsa.metal): the fused QKV + compression-gate GEMM, the
// tile-order permutes, the coarse stage (tile means, routing scores, top-k selection, compressed output), the
// block-sparse fine stage (int8 QK or the QKFIX fp16 path) and the out-projection producer that adds the gated
// compressed output. Rows are in the tile order of a VsaTable inside the blocks.
#include "video/metal/Encoder.hpp"
#include "video/ops/Linear.hpp"
#include "video/ops/VsaTable.hpp"
#include "video/ops/WindowAttention.hpp"

namespace splash::video::ops {

struct VsaBuffers {
  BufferView pooledQ, pooledK, pooledV;  // fp32 [56][Tp][128]
  BufferView outC;                       // fp32 [56][Tp][128]
  BufferView scores;                     // fp32 [56][Tp][Tp]: routing scores, then their softmax
  BufferView selected;                   // int32 [56][V][k]
};

// int8 GEMM over the rows [q | k | v | compression gate] ([28672, 5376], no raw q / k): Q / K / V as ops::qkv, the
// gate rows' bf16-rounded output fp16 [M][7168] to `gate`; GateOnly: the gate rows alone (QKFIX blocks, whose q / k /
// v come from the fp16 GEMM); QkvOnly: the q / k / v rows alone (dense attention: no gate).
// With a LoRA (q / k / v only; not with GateOnly), its down-projection first and dit_qkv_lora_post.
enum class QkvRows : int32_t { All, GateOnly, QkvOnly };
void qkvGate(Encoder &e, ProducerOut in, int32_t rows, BufferView weight, BufferView weightScale, BufferView qNorm,
             BufferView kNorm, BufferView rope, const QkvTargets &out, BufferView gate, QkvRows parts,
             const LoraBinding &lora = {});
// dst[i] = src[index[i]] for bf16 rows of 5376 (index: the table's order or inverse buffer).
void vsaPermute(Encoder &e, BufferView src, BufferView dst, const metal::MetalBuffer &index, int32_t rows);
// Tile means of Q / K / V (qExact / kExact / splitDims: the QKFIX split dims, empty for int8 blocks).
void vsaPool(Encoder &e, const VsaTable &t, const AttentionBuffers &a, BufferView qExact, BufferView kExact,
             BufferView splitDims, const VsaBuffers &v);
// Routing scores, top-k video tiles, compressed output.
void vsaRoute(Encoder &e, const VsaTable &t, const AttentionBuffers &a, const VsaBuffers &v);
// Fine stage -> bf16 O [M][7168]: int8 QK (after ops::quantize) or, with qExact, the fp16 QK of the QKFIX kernel
// with the QKFIX blocks' exact fp32 split term. With a dense table (VsaSpec::dense) every 64-row chunk attends to all
// rows: no selection is read.
void vsaAttention(Encoder &e, const VsaTable &t, const AttentionBuffers &a, BufferView qExact, BufferView kExact,
                  const VsaBuffers &v, BufferView out);
// O + out_c[tile(row)] (.) gate -> Hadamard -> int8, K = 7168 (the out-projection input); dense: O alone.
void vsaQuantize(Encoder &e, const VsaTable &t, BufferView attention, BufferView gate, const VsaBuffers &v,
                 int32_t rows, int32_t paddedRows, BufferView signs, ProducerOut out);

} // namespace splash::video::ops

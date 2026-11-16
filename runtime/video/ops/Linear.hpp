#pragma once

// DiT W8A8 linears: Hadamard int8 producers and the GEMMs with fused epilogues (kernels/video/dit/{producer,linear}.metal).
// GEMMs over more than kMaxRowsPerDispatch rows are split into row chunks (buffer offsets) so every in-kernel element
// offset stays below 2^31.

#include "video/metal/Encoder.hpp"

#include <string>
#include <vector>

namespace splash::video::ops {

// adaLN table rows (per modality: shift / scale / gate of attention, then of the MLP)
enum class AdaRow : int32_t { Shift1 = 0, Scale1 = 1, Gate1 = 2, Shift2 = 3, Scale2 = 4, Gate2 = 5 };

struct ProducerOut {
  BufferView q;      // int8 [rows][K]
  BufferView scale;  // fp32 [rows]
};

// The runtime LoRA of one GEMM (an installed adapter's, kernels/video/dit/lora.metal): A bf16 [segments x 128][K] (rotated for
// the int8 GEMMs), B bf16 [N][128], the scratch T fp16 [rows rounded up to 64, + 64][ldt] that the down-projection
// (loraDown) writes and the GEMM's _lora_post kernel reads, and the adapter's scale. Empty: no LoRA.
struct LoraBinding {
  BufferView a, b, t;
  int32_t segments = 1, ldt = 0;
  float scale = 1.0f;
  [[nodiscard]] explicit operator bool() const { return static_cast<bool>(a); }
};
// T = scale (A x) of the int8 rows `in` (K inputs, their per-row scales).
void loraDown(Encoder &e, ProducerOut in, int32_t rows, int32_t K, const LoraBinding &lora);
// T = scale (A x) of fp16 rows x [rows][K] (the QKFIX blocks' unrotated input).
void loraDownHalf(Encoder &e, BufferView x, int32_t rows, int32_t K, const LoraBinding &lora);

// RMSNorm(X) * w * (1 + scale[mod]) + shift[mod] -> Hadamard -> int8, K = 5376.
void normQuantize(Encoder &e, BufferView x, int32_t rows, BufferView normWeight, BufferView ada, AdaRow shift,
                  AdaRow scale, BufferView tags, BufferView signs, ProducerOut out);
// Hadamard -> int8 of the SwiGLU output h, K = 14336.
void hiddenQuantize(Encoder &e, BufferView h, int32_t rows, BufferView signs, ProducerOut out);
// softmax gate (.) attention output -> Hadamard -> int8, K = 7168 (out-projection input).
void gatedQuantize(Encoder &e, BufferView attention, BufferView softmaxGate, int32_t rows, BufferView signs,
                   ProducerOut out);
// linear-branch readout of the inner video rows -> Hadamard (signs 7168..14335) -> int8 (to_out_linear input).
void branchQuantize(Encoder &e, BufferView readout, int32_t rows, BufferView signs, ProducerOut out);

struct QkvTargets {
  BufferView q, k, v;  // fp16 [56][Spad][128]
  BufferView rawQ, rawK;  // fp16 [56][rawRows][128]
  int32_t paddedRows, rawRows;
  int32_t text, innerRow0, innerRowEnd;  // raw rows kept: [0, text) and [innerRow0, innerRowEnd)
};
// int8 QKV GEMM [M, 21504] with QK-norm + RoPE and the raw q / k for the linear branch.
void qkv(Encoder &e, ProducerOut in, int32_t rows, BufferView weight, BufferView weightScale, BufferView qNorm,
         BufferView kNorm, BufferView rope, const QkvTargets &out, const LoraBinding &lora = {});
// X[row] += gate[mod[row]] * dequant(A W^T), N = 5376 (out: K = 7168, fc2: K = 14336, to_out_linear: K = 7168).
// With a LoRA (down-projection included): X[row] += gate[mod[row]] * (dequant(A W^T) + scale B (A_lora x)).
void gatedResidual(Encoder &e, ProducerOut in, int32_t rows, int32_t K, BufferView weight, BufferView weightScale,
                   BufferView x, BufferView tags, BufferView ada, AdaRow gate, const LoraBinding &lora = {});
// fc1 [2 x 14336, 5376] + SwiGLU -> bf16 h [M][14336].
void fc1SwiGlu(Encoder &e, ProducerOut in, int32_t rows, BufferView weight, BufferView weightScale, BufferView hidden,
               const LoraBinding &lora = {});

} // namespace splash::video::ops

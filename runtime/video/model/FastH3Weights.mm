#include "video/model/DitWeights.hpp"

#include "metal/abi/video/Layout.h"

#include <string>

namespace splash::video::model {

// A converted FastH3 block: q | k | v and the VSA compression gate (the fused GEMM of ops::qkvGate), out,
// fc1, fc2 and the norms; no linear branch. The int8 matrices are streamed to the slot tensors of qkv + gates (the
// gate after the q / k / v rows), out, fc1 and fc2.
void DitWeights::loadFastH3Block(metal::MetalBackend &backend, int b) {
  constexpr int64_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER, QKV = H3_DIT_QKV, HD = H3_DIT_HEAD_DIM;
  DitBlockWeights &w = blocks_[b];
  const std::string p = "blocks." + std::to_string(b) + ".";
  struct Matrix {
    const char *name;
    int64_t rows, cols;
    int slot;
    uint64_t slotOffset;
  };
  const Matrix matrices[] = {
      {"attn.qkv.qweight", QKV, D, 0, 0},
      {"attn.gate.qweight", C, D, 0, uint64_t(QKV * D)},
      {"attn.out.qweight", D, C, 1, 0},
      {"mlp.fc1.qweight", 2 * F, D, 2, 0},
      {"mlp.fc2.qweight", D, F, 3, 0},
  };
  for (const Matrix &m : matrices)
    streamed_.push_back({b, m.slot, m.slotOffset, &tensor(*checkpoint_, p + m.name, "I8", {m.rows, m.cols}), m.name});
  w.qkvgScale = allocate(backend, (QKV + C) * 4, "qkv + gate scale");
  residentBytes_ += w.qkvgScale.sizeBytes();
  read(tensor(*checkpoint_, p + "attn.qkv.wscale", "F32", {QKV}), w.qkvgScale.contents());
  read(tensor(*checkpoint_, p + "attn.gate.wscale", "F32", {C}),
       static_cast<uint8_t *>(w.qkvgScale.contents()) + QKV * 4);
  w.outScale = resident(backend, p + "attn.out.wscale", "F32", {D});
  w.fc1Scale = resident(backend, p + "mlp.fc1.wscale", "F32", {2 * F});
  w.fc2Scale = resident(backend, p + "mlp.fc2.wscale", "F32", {D});
  w.norm1 = resident(backend, p + "norm1", "BF16", {D});
  w.norm2 = resident(backend, p + "norm2", "BF16", {D});
  w.qNorm = resident(backend, p + "attn.q_norm", "BF16", {HD});
  w.kNorm = resident(backend, p + "attn.k_norm", "BF16", {HD});
}

} // namespace splash::video::model

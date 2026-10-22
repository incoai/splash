#include "video/h3/engine/MemoryPlan.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/Layout.h"

#include <algorithm>
#include <cstring>

namespace h3 {

MemoryPlan::MemoryPlan(metal::MetalBackend &backend, const model::SequenceLayout &lay, int32_t partialSlots,
                       bool qkfix) {
  MemoryPlan &plan = *this;
  const size_t M = lay.rows(), Spad = lay.paddedRows(), L = std::max(lay.text, 1), Na = std::max(lay.audio, 1);
  const size_t Fp = lay.innerFrames(), Nin = lay.innerRows(), Nv = static_cast<size_t>(lay.frames) * lay.tokensPerFrame;
  const size_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, H = H3_DIT_HEADS, HD = H3_DIT_HEAD_DIM, C = H3_DIT_INNER;
  const size_t heads = H * Spad * HD;       // one [H][Spad][128] tensor (elements)
  const size_t state = Fp * H * H3_BRANCH_STATE * 4;  // one fp32 [Fp][H][128][128] bank
  const size_t features = H * Nin * HD * 2;  // one fp16 [H][Nin][128] feature tensor
  const size_t raw = H * static_cast<size_t>(lay.rawRows()) * HD * 2;
  const size_t slots = static_cast<size_t>(device::kHeadsPerWindowDispatch) * partialSlots * H3_ATTENTION_GROUP;

  auto set = [&](Tensor t, const char *name, Slab s, size_t offset, size_t bytes, const char *lifetime) {
    plan.entries_[static_cast<size_t>(t)] = {name, s, offset, bytes, lifetime};
  };
  // ---- residual stream and request inputs (whole request)
  set(Tensor::Residual, "residual X bf16 [M][5376]", Slab::Residual, 0, M * D * 2, "request");
  set(Tensor::Tags, "modality tags u8 [M]", Slab::Tags, 0, M, "request");
  set(Tensor::Rope, "rope bf16 [M][96]", Slab::Rope, 0, M * H3_DIT_ROPE * 2, "request");
  set(Tensor::Text, "refined text bf16 [L][5376]", Slab::Text, 0, L * D * 2, "request");
  set(Tensor::AdaBlocks, "adaLN tables bf16 [50][3 | 4][6][5376]", Slab::AdaBlocks, 0,
      size_t(H3_DIT_BLOCKS) * lay.adaModalities() * H3_ADA_ROWS * D * 2, "step");
  set(Tensor::AdaFinal, "final adaLN table bf16 [3][2][5376]", Slab::AdaFinal, 0, size_t(H3_MODALITIES) * 2 * D * 2,
      "step");
  set(Tensor::SignsHidden, "Hadamard signs [5376]", Slab::SignsHidden, 0, D, "request");
  set(Tensor::SignsFfn, "Hadamard signs [14336]", Slab::SignsFfn, 0, F, "request");
  set(Tensor::VideoIn, "video latent in fp32 [C + Nv][96]", Slab::RawQ, 0,
      (size_t(lay.cond) + Nv) * H3_DIT_VIDEO_CHANNELS * 4,
      "embed (before block 0 writes raw q)");
  set(Tensor::VideoOut, "video velocity fp32 [Nv][96]", Slab::RawQ, 0, Nv * H3_DIT_VIDEO_CHANNELS * 4,
      "final layer (after block 49 read raw q)");
  set(Tensor::AudioIn, "audio latent in fp32 [Na][32]", Slab::AudioIn, 0, Na * H3_DIT_AUDIO_CHANNELS * 4, "step");
  set(Tensor::AudioOut, "audio velocity fp32 [Na][32]", Slab::AudioOut, 0, Na * H3_DIT_AUDIO_CHANNELS * 4, "step");
  // ---- linears: int8 producer output and its row scales; fc1 output
  set(Tensor::LinearIn, "int8 linear input [M][<=14336]", Slab::LinearIn, 0, M * F,
      "producer -> its GEMM (norm1 -> qkv / gates / frame mean, norm2 -> fc1, h -> fc2)");
  set(Tensor::RowScale, "linear input row scales fp32 [M]", Slab::RowScale, 0, M * 4, "producer -> GEMM");
  set(Tensor::Hidden, "SwiGLU h bf16 [M][14336]", Slab::QkHidden, 0, M * F * 2, "fc1 -> h producer");
  // ---- attention
  set(Tensor::QueryHalf, "Q fp16 [56][Spad][128]", Slab::QkHidden, 0, heads * 2, "qkv -> attention quantize");
  set(Tensor::KeyHalf, "K fp16 [56][Spad][128]", Slab::QkHidden, heads * 2, heads * 2, "qkv -> attention quantize");
  set(Tensor::Value, "V fp16 [56][Spad][128] (pad rows zero)", Slab::Value, 0, heads * 2,
      "qkv -> window attention, text features, value conv");
  set(Tensor::QueryInt8, "Q int8 [56][Spad][128]", Slab::LinearIn, 0, heads,
      "attention quantize -> window attention (LinearIn dead after gates / frame mean)");
  set(Tensor::KeyInt8, "K - mean(K) int8 [56][Spad][128]", Slab::LinearIn, heads, heads,
      "attention quantize -> window attention");
  set(Tensor::QueryScale, "Q row scales fp32 [56][Spad]", Slab::QueryScale, 0, H * Spad * 4, "quantize -> attention");
  set(Tensor::KeyScale, "K block scales fp32 [56][Spad / 32]", Slab::KeyScale, 0, H * Spad * 4,
      "quantize -> attention");
  set(Tensor::KeyMean, "key mean fp32 [56][128]", Slab::KeyMean, 0, H * HD * 4, "attention preparation");
  set(Tensor::KeyMeanPartial, "key mean partials fp32 [56][64][128]", Slab::KeyMeanPartial, 0,
      H * device::kKeyMeanChunks * HD * 4, "attention preparation");
  set(Tensor::AttentionOut, "attention output bf16 [M][7168]", Slab::RawK, 0, M * C * 2,
      "window attention -> gated producer (raw k consumed by the key conv before)");
  set(Tensor::AttentionOutQkFix, "attention output bf16 [M][7168] (QKFIX blocks)", Slab::LinearIn, 0, M * C * 2,
      "QKFIX attention -> gated producer (fp16 x consumed by the QKFIX GEMM)");
  set(Tensor::OutProjIn, "out-projection int8 input [M][7168]", Slab::LinearIn, 0, M * C, "gated producer -> out");
  set(Tensor::OutProjInQkFix, "out-projection int8 input [M][7168] (QKFIX blocks)", Slab::QkHidden, 0, M * C,
      "gated producer -> out (fp16 Q / K consumed by attention)");
  set(Tensor::WindowPartials, "key-chunk partials fp16 [56][slots][64][128]", Slab::WindowPartials, 0, slots * HD * 2,
      "window attention -> combine");
  set(Tensor::WindowStats, "key-chunk (m, l) fp32 [56][slots][64][2]", Slab::WindowStats, 0, slots * 8,
      "window attention -> combine");
  // ---- QKFIX
  set(Tensor::QkFixInput, "fp16 normalised x [M][5376]", Slab::LinearIn, 0, M * D * 2,
      "QKFIX producer -> QKFIX GEMM (after gates / frame mean read LinearIn)");
  set(Tensor::QkFixQueryExact, "QKFIX split dims of Q fp32 [56][Spad][4]", Slab::QkFixQueryExact, 0,
      qkfix ? H * Spad * H3_QKFIX_SPLIT * 4 : 0, "QKFIX GEMM -> attention");
  set(Tensor::QkFixKeyExact, "QKFIX split dims of K fp32 [56][Spad][4]", Slab::QkFixKeyExact, 0,
      qkfix ? H * Spad * H3_QKFIX_SPLIT * 4 : 0, "QKFIX GEMM -> attention");
  // ---- linear branch
  set(Tensor::RawQ, "raw q fp16 [56][L + Nin][128]", Slab::RawQ, 0, raw, "qkv -> readout");
  set(Tensor::RawK, "raw k fp16 [56][L + Nin][128]", Slab::RawK, 0, raw, "qkv -> text features, key conv");
  set(Tensor::Beta, "beta fp32 [M][56]", Slab::Beta, 0, M * H * 4, "gates -> features");
  set(Tensor::SoftmaxGate, "softmax gate fp32 [M][56]", Slab::SoftmaxGate, 0, M * H * 4, "gates -> gated producer");
  set(Tensor::OutGateHidden, "output-gate hidden fp16 [M][128]", Slab::OutGateHidden, 0, M * HD * 2,
      "gates -> readout");
  set(Tensor::FrameMean, "frame means fp32 [Fp][5376]", Slab::FrameMean, 0, Fp * D * 4, "frame mean -> alpha");
  set(Tensor::Alpha, "decays fp32 [Fp][7168]", Slab::Alpha, 0, Fp * C * 4, "alpha -> scan, gather");
  set(Tensor::TextKeys, "text key features fp16 [56][L][128]", Slab::TextKeys, 0, H * L * HD * 2,
      "text features -> text stats");
  set(Tensor::TextValues, "text value features fp16 [56][L][128]", Slab::TextValues, 0, H * L * HD * 2,
      "text features -> text stats");
  set(Tensor::KeyFeatures, "key features fp16 [56][Nin][128]", Slab::QkHidden, 0, features,
      "key conv -> stats (Q / K fp16 consumed by attention quantize)");
  set(Tensor::ValueFeatures, "value features fp16 [56][Nin][128]", Slab::QkHidden, features, features,
      "value conv -> stats");
  set(Tensor::StatsA, "A fp32 [Fp][56][128][128]", Slab::RawK, 0, state,
      "stats -> inverse (attention output consumed by the gated producer)");
  set(Tensor::StatsB, "B fp32 [Fp][56][128][128]", Slab::RawK, state, state, "stats -> scan");
  set(Tensor::Inverse, "(I + A)^-1 fp32 [Fp][56][128][128]", Slab::RawK, 2 * state, state, "inverse -> scan");
  set(Tensor::TextA, "text A fp32 [56][128][128]", Slab::TextA, 0, H * H3_BRANCH_STATE * 4, "text stats -> inverse");
  set(Tensor::TextB, "text B fp32 [56][128][128]", Slab::TextB, 0, H * H3_BRANCH_STATE * 4, "text stats -> scan");
  set(Tensor::TextInverse, "text inverse fp32 [56][128][128]", Slab::TextInverse, 0, H * H3_BRANCH_STATE * 4,
      "inverse -> scan");
  set(Tensor::Pivots, "min pivots fp32 [Fp][56]", Slab::Pivots, 0, Fp * H * 4, "inverse (diagnostic)");
  set(Tensor::TextPivots, "text min pivots fp32 [56]", Slab::TextPivots, 0, H * 4, "inverse (diagnostic)");
  set(Tensor::ScanPre, "forward scan bank fp32 [Fp][56][128][128]", Slab::LinearIn, 0, state,
      "scan -> gather (out-projection input consumed by out)");
  set(Tensor::ScanSuf, "reverse scan bank fp32 [Fp][56][128][128]", Slab::LinearIn, state, state, "scan -> gather");
  set(Tensor::ScanStart, "scan start states fp32 [2][56][128][128]", Slab::ScanStart, 0, 2 * H * H3_BRANCH_STATE * 4,
      "scan -> gather");
  set(Tensor::ScanScratch, "scan scratch fp32 [2][56][128][128]", Slab::ScanScratch, 0, 2 * H * H3_BRANCH_STATE * 4,
      "scan");
  set(Tensor::LinearState, "windowed state fp16 [Fp][56][128][128]", Slab::QkHidden, 0, Fp * H * H3_BRANCH_STATE * 2,
      "gather -> readout (features consumed by stats)");
  set(Tensor::StateScale, "state scales fp32 [Fp][56]", Slab::StateScale, 0, Fp * H * 4, "gather -> readout");
  set(Tensor::Bounds, "rebased window bounds int32 [Fp][2]", Slab::Bounds, 0, Fp * 8, "request");
  set(Tensor::Readout, "branch readout bf16 [Nin][7168]", Slab::Value, 0, Nin * C * 2,
      "readout -> branch producer (V consumed by attention and the value conv)");
  set(Tensor::BranchIn, "to_out_linear int8 input [Nin][7168]", Slab::LinearIn, 0, Nin * C,
      "branch producer -> to_out_linear (scan banks consumed by gather)");
  set(Tensor::BranchRowScale, "to_out_linear row scales fp32 [Nin]", Slab::BranchRowScale, 0, Nin * 4,
      "branch producer -> to_out_linear");

  for (const PlanEntry &e : plan.entries_) {
    size_t &slab = plan.slabBytes_[static_cast<size_t>(e.slab)];
    slab = std::max(slab, e.offset + e.bytes);
  }
  for (size_t s = 0; s < static_cast<size_t>(Slab::Count); ++s)
    if (plan.slabBytes_[s]) plan.slabs_[s] = allocate(backend, plan.slabBytes_[s], "activation");
  // pad rows of V must be finite (attention multiplies them by P = 0); QKFIX split dims start at 0
  for (Slab s : {Slab::Value, Slab::QkFixQueryExact, Slab::QkFixKeyExact})
    if (const metal::MetalBuffer &b = plan.slabs_[static_cast<size_t>(s)]) std::memset(b.contents(), 0, b.sizeBytes());
}

BufferView MemoryPlan::view(Tensor t) const {
  const PlanEntry &e = entries_[static_cast<size_t>(t)];
  return {slabs_[static_cast<size_t>(e.slab)], e.offset};
}

size_t MemoryPlan::totalBytes() const {
  size_t total = 0;
  for (size_t b : slabBytes_) total += b;
  return total;
}

} // namespace h3

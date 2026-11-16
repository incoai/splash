#include "video/stages/DenoisingStage.hpp"

#include "metal/abi/video/Layout.h"
#include "video/ops/Linear.hpp"
#include "video/ops/QkFix.hpp"

namespace splash::video {
namespace {

constexpr int32_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER;

using model::Tensor;
using ops::AdaRow;

} // namespace

ops::VsaBuffers Dit::vsaBuffers() const {
  ops::VsaBuffers v;
  v.pooledQ = plan_.view(Tensor::PooledQ);
  v.pooledK = plan_.view(Tensor::PooledK);
  v.pooledV = plan_.view(Tensor::PooledV);
  v.outC = plan_.view(Tensor::OutC);
  v.selected = plan_.view(Tensor::Selected);
  v.scores = plan_.view(Tensor::VsaScores);
  return v;
}

// A FastH3 block, its rows in tile order: norm1 -> QKV + compression gate -> VSA (tile means, routing, fine stage) ->
// O + gate (.) compressed output -> out; the MLP as VDN-H3's blocks. QKFIX blocks take q / k / v from the fp16 GEMM
// and the gate rows from the int8 GEMM, read before the fp16 x takes the LinearIn slab. Dense attention
// (VsaSpec::dense): QKV without gate rows, no coarse stage or selection, every 64-row chunk attends to all rows.
void Dit::encodeVsaBlock(int block, const BlockBinding &bind, Encoder &e) const {
  const model::DitBlockWeights &w = *bind.w;
  const bool qkfix = weights_->qkfixBlocks().count(block) != 0;
  const ops::VsaTable &t = *vsa_;
  const int32_t M = layout_.rows();
  const BufferView x = plan_.view(Tensor::Residual), tags = plan_.view(Tensor::Tags);
  const BufferView rope = plan_.view(Tensor::Rope);
  const size_t adaBlockBytes = size_t(layout_.adaModalities()) * H3_ADA_ROWS * D * 2;
  const BufferView ada = plan_.view(Tensor::AdaBlocks) + size_t(block) * adaBlockBytes;
  const BufferView signsHidden = plan_.view(Tensor::SignsHidden), signsFfn = plan_.view(Tensor::SignsFfn);
  const BufferView gate = plan_.view(Tensor::Gate);
  const ops::ProducerOut linearIn{plan_.view(Tensor::LinearIn), plan_.view(Tensor::RowScale)};
  const ops::ProducerOut outProjIn{plan_.view(qkfix ? Tensor::OutProjInQkFix : Tensor::OutProjIn),
                                   plan_.view(Tensor::RowScale)};
  const BufferView attentionOut = plan_.view(qkfix ? Tensor::AttentionOutQkFix : Tensor::AttentionOut);
  const ops::AttentionBuffers attention = attentionBuffers();
  ops::QkvTargets qkv = qkvTargets();
  qkv.rawRows = 0;  // no linear branch: no raw q / k (their slab holds the gate)
  const ops::QkFixTargets qkfixQkv{qkv, plan_.view(Tensor::QkFixQueryExact), plan_.view(Tensor::QkFixKeyExact)};
  const ops::VsaBuffers v = vsaBuffers();
  const model::AdapterBlock *ab = adapterBlock(block);
  const ops::LoraBinding none;

  // ---- attention half (the LoRA of q / k / v: the int8 GEMM's, or the QKFIX blocks' fp16 GEMM's). Dense attention
  // (pika-6's base): no gate rows, coarse stage or selection; the QKFIX blocks skip the int8 GEMM altogether.
  const bool dense = t.spec().dense;
  if (!(dense && qkfix))  // a dense QKFIX block reads no int8 x: its q / k / v come from the fp16 GEMM, no gate rows
    ops::normQuantize(e, x, M, w.norm1, ada, AdaRow::Shift1, AdaRow::Scale1, tags, signsHidden, linearIn);
  if (!(dense && qkfix))
    ops::qkvGate(e, linearIn, M, bind.qkvg, w.qkvgScale, w.qNorm, w.kNorm, rope, qkv, dense ? BufferView() : gate,
                 qkfix ? ops::QkvRows::GateOnly : (dense ? ops::QkvRows::QkvOnly : ops::QkvRows::All),
                 ab && !qkfix ? lora(ab->qkv) : none);
  if (qkfix) {
    ops::qkfixNormalize(e, x, M, w.norm1, ada, tags, plan_.view(Tensor::QkFixInput));
    ops::qkfixQkv(e, plan_.view(Tensor::QkFixInput), M, bind.qkfixWeight, bind.qkfixSplit, w.qNorm, w.kNorm, rope,
                  qkfixQkv, ab ? lora(ab->qkv, &ab->qkvPlain) : none);
  } else {
    ops::quantize(e, attention);
  }
  if (!dense) {
    ops::vsaPool(e, t, attention, qkfixQkv.qExact, qkfixQkv.kExact, qkfix ? bind.qkfixSplit : BufferView(), v);
    ops::vsaRoute(e, t, attention, v);
  }
  ops::vsaAttention(e, t, attention, qkfix ? qkfixQkv.qExact : BufferView(), qkfix ? qkfixQkv.kExact : BufferView(), v,
                    attentionOut);
  const BufferView hidden = plan_.view(Tensor::Hidden);
  ops::vsaQuantize(e, t, attentionOut, gate, v, M, layout_.paddedRows(), signsFfn, outProjIn);
  ops::gatedResidual(e, outProjIn, M, C, bind.out, w.outScale, x, tags, ada, AdaRow::Gate1, ab ? lora(ab->out) : none);

  // ---- MLP half: norm2 -> fc1 + SwiGLU -> fc2
  ops::normQuantize(e, x, M, w.norm2, ada, AdaRow::Shift2, AdaRow::Scale2, tags, signsHidden, linearIn);
  ops::fc1SwiGlu(e, linearIn, M, bind.fc1, w.fc1Scale, hidden, ab ? lora(ab->fc1) : none);
  ops::hiddenQuantize(e, hidden, M, signsFfn, linearIn);
  ops::gatedResidual(e, linearIn, M, F, bind.fc2, w.fc2Scale, x, tags, ada, AdaRow::Gate2, ab ? lora(ab->fc2) : none);
}

} // namespace splash::video

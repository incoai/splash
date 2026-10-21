#include "video/h3/engine/Scheduler.hpp"

#include "AwakeClock.hpp"
#include "video/h3/engine/DeviceProfile.hpp"
#include "video/h3/ops/DitIo.hpp"
#include "video/h3/ops/Linear.hpp"
#include "video/h3/ops/QkFix.hpp"

#include <algorithm>
#include <cstring>

namespace h3 {
namespace {

constexpr int32_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER;
constexpr size_t kAdaBlockBytes = size_t(H3_MODALITIES) * H3_ADA_ROWS * D * 2;

using ops::AdaRow;

} // namespace

Scheduler::Scheduler(metal::MetalBackend &backend, const model::SequenceLayout &layout, const MemoryPlan &plan,
                     const WindowTable &table, const model::DitWeights *weights, WeightStream *stream,
                     std::set<int> qkfixBlocks)
    : backend_(backend), layout_(layout), plan_(plan), table_(table), weights_(weights), stream_(stream),
      qkfixBlocks_(std::move(qkfixBlocks)), branch_(layout, branchBuffers()) {}

ops::AttentionBuffers Scheduler::attentionBuffers() const {
  ops::AttentionBuffers a;
  a.qHalf = plan_.view(Tensor::QueryHalf);
  a.kHalf = plan_.view(Tensor::KeyHalf);
  a.v = plan_.view(Tensor::Value);
  a.qInt8 = plan_.view(Tensor::QueryInt8);
  a.kInt8 = plan_.view(Tensor::KeyInt8);
  a.qScale = plan_.view(Tensor::QueryScale);
  a.kScale = plan_.view(Tensor::KeyScale);
  a.keyMean = plan_.view(Tensor::KeyMean);
  a.keyMeanPartial = plan_.view(Tensor::KeyMeanPartial);
  a.partials = plan_.view(Tensor::WindowPartials);
  a.partialStats = plan_.view(Tensor::WindowStats);
  a.rows = layout_.rows();
  a.paddedRows = layout_.paddedRows();
  return a;
}

ops::BranchBuffers Scheduler::branchBuffers() const {
  ops::BranchBuffers x;
  x.rawQ = plan_.view(Tensor::RawQ);
  x.rawK = plan_.view(Tensor::RawK);
  x.v = plan_.view(Tensor::Value);
  x.beta = plan_.view(Tensor::Beta);
  x.softmaxGate = plan_.view(Tensor::SoftmaxGate);
  x.outGateHidden = plan_.view(Tensor::OutGateHidden);
  x.frameMean = plan_.view(Tensor::FrameMean);
  x.alpha = plan_.view(Tensor::Alpha);
  x.textKeys = plan_.view(Tensor::TextKeys);
  x.textValues = plan_.view(Tensor::TextValues);
  x.keyFeatures = plan_.view(Tensor::KeyFeatures);
  x.valueFeatures = plan_.view(Tensor::ValueFeatures);
  x.statsA = plan_.view(Tensor::StatsA);
  x.statsB = plan_.view(Tensor::StatsB);
  x.inverse = plan_.view(Tensor::Inverse);
  x.textA = plan_.view(Tensor::TextA);
  x.textB = plan_.view(Tensor::TextB);
  x.textInverse = plan_.view(Tensor::TextInverse);
  x.pivots = plan_.view(Tensor::Pivots);
  x.textPivots = plan_.view(Tensor::TextPivots);
  x.scanPre = plan_.view(Tensor::ScanPre);
  x.scanSuf = plan_.view(Tensor::ScanSuf);
  x.scanStart = plan_.view(Tensor::ScanStart);
  x.scanScratch = plan_.view(Tensor::ScanScratch);
  x.state = plan_.view(Tensor::LinearState);
  x.stateScale = plan_.view(Tensor::StateScale);
  x.bounds = plan_.view(Tensor::Bounds);
  x.readout = plan_.view(Tensor::Readout);
  x.paddedRows = layout_.paddedRows();
  x.rawRows = layout_.rawRows();
  return x;
}

ops::QkvTargets Scheduler::qkvTargets() const {
  ops::QkvTargets t;
  t.q = plan_.view(Tensor::QueryHalf);
  t.k = plan_.view(Tensor::KeyHalf);
  t.v = plan_.view(Tensor::Value);
  t.rawQ = plan_.view(Tensor::RawQ);
  t.rawK = plan_.view(Tensor::RawK);
  t.paddedRows = layout_.paddedRows();
  t.rawRows = layout_.rawRows();
  t.text = layout_.text;
  t.innerRow0 = layout_.innerRow0();
  t.innerRowEnd = layout_.innerRow0() + layout_.innerRows();
  return t;
}

BlockBinding Scheduler::residentBinding(int block) const {
  const model::DitBlockWeights &w = weights_->block(block);
  BlockBinding binding{&w, w.qkvg, w.out, w.fc1, w.fc2, w.outLinear, {}, {}};
  if (qkfixBlocks_.count(block)) {
    binding.qkfixWeight = weights_->qkfix(block).wqkv;
    binding.qkfixSplit = weights_->qkfix(block).splitDims;
  }
  return binding;
}

// Kernel order and the MemoryPlan aliases depend on each other: a kernel may only overwrite a slab view whose last
// reader has been encoded before it (see the lifetimes in MemoryPlan.mm).
void Scheduler::encodeBlock(int block, const BlockBinding &bind, Encoder &e) const {
  const model::DitBlockWeights &w = *bind.w;
  const bool qkfix = qkfixBlocks_.count(block) != 0;
  const int32_t M = layout_.rows(), Nin = layout_.innerRows();
  const BufferView x = plan_.view(Tensor::Residual), tags = plan_.view(Tensor::Tags);
  const BufferView rope = plan_.view(Tensor::Rope);
  const BufferView ada = plan_.view(Tensor::AdaBlocks) + size_t(block) * kAdaBlockBytes;
  const BufferView signsHidden = plan_.view(Tensor::SignsHidden), signsFfn = plan_.view(Tensor::SignsFfn);
  const ops::ProducerOut linearIn{plan_.view(Tensor::LinearIn), plan_.view(Tensor::RowScale)};
  const ops::ProducerOut outProjIn{plan_.view(qkfix ? Tensor::OutProjInQkFix : Tensor::OutProjIn),
                                   plan_.view(Tensor::RowScale)};
  const ops::ProducerOut branchIn{plan_.view(Tensor::BranchIn), plan_.view(Tensor::BranchRowScale)};
  const BufferView attentionOut = plan_.view(qkfix ? Tensor::AttentionOutQkFix : Tensor::AttentionOut);
  const BufferView qkfixIn = plan_.view(Tensor::QkFixInput);
  const ops::AttentionBuffers attention = attentionBuffers();
  const ops::QkvTargets qkv = qkvTargets();
  const ops::QkFixTargets qkfixQkv{qkv, plan_.view(Tensor::QkFixQueryExact), plan_.view(Tensor::QkFixKeyExact)};
  const size_t row0 = static_cast<size_t>(layout_.innerRow0());

  // ---- attention half: norm1 -> QKV (+ raw q / k) -> branch gates and decays -> window attention -> out
  ops::normQuantize(e, x, M, w.norm1, ada, AdaRow::Shift1, AdaRow::Scale1, tags, signsHidden, linearIn);
  if (!qkfix) ops::qkv(e, linearIn, M, bind.qkvg, w.qkvgScale, w.qNorm, w.kNorm, rope, qkv);
  branch_.gates(e, linearIn, bind.qkvg, w.qkvgScale, w.sgBias);
  branch_.frameMean(e, linearIn);
  branch_.alpha(e, w);
  if (qkfix) {  // fp16 q / k / v from the unrotated weights (the int8 QKV GEMM is skipped)
    ops::qkfixNormalize(e, x, M, w.norm1, ada, tags, qkfixIn);
    ops::qkfixQkv(e, qkfixIn, M, bind.qkfixWeight, bind.qkfixSplit, w.qNorm, w.kNorm, rope, qkfixQkv);
  } else {
    ops::quantize(e, attention);
  }
  branch_.textFeatures(e);
  // raw k is read before the attention output takes its slab (QKFIX blocks keep O elsewhere and read it later)
  if (!qkfix) branch_.keyFeatures(e, w);
  if (qkfix) ops::windowQkFix(e, attention, qkfixQkv.qExact, qkfixQkv.kExact, table_, attentionOut);
  else ops::window(e, attention, table_, attentionOut);
  ops::gatedQuantize(e, attentionOut, plan_.view(Tensor::SoftmaxGate), M, signsFfn, outProjIn);
  ops::gatedResidual(e, outProjIn, M, C, bind.out, w.outScale, x, tags, ada, AdaRow::Gate1);

  // ---- linear branch of the inner video rows -> to_out_linear into the same residual (attention gate)
  if (qkfix) branch_.keyFeatures(e, w);
  branch_.valueFeatures(e, w);
  branch_.stats(e);
  branch_.textStats(e);
  branch_.inverse(e);
  branch_.textInverse(e);
  branch_.scan(e);
  branch_.gather(e);
  branch_.readout(e, w);
  ops::branchQuantize(e, plan_.view(Tensor::Readout), Nin, signsFfn, branchIn);
  ops::gatedResidual(e, branchIn, Nin, C, bind.outLinear, w.outLinearScale, x + row0 * D * 2, tags + row0, ada,
                     AdaRow::Gate1);

  // ---- MLP half: norm2 -> fc1 + SwiGLU -> fc2
  ops::normQuantize(e, x, M, w.norm2, ada, AdaRow::Shift2, AdaRow::Scale2, tags, signsHidden, linearIn);
  ops::fc1SwiGlu(e, linearIn, M, bind.fc1, w.fc1Scale, plan_.view(Tensor::Hidden));
  ops::hiddenQuantize(e, plan_.view(Tensor::Hidden), M, signsFfn, linearIn);
  ops::gatedResidual(e, linearIn, M, F, bind.fc2, w.fc2Scale, x, tags, ada, AdaRow::Gate2);
}

void Scheduler::forward(const uint16_t *adaBlocks, const uint16_t *adaFinal, const float *videoIn,
                        const float *audioIn, float *videoOut, float *audioOut, ForwardStats &stats) {
  const auto t0 = splash::AwakeClock::now();
  stats = ForwardStats{};
  const size_t videoBytes = static_cast<size_t>(layout_.frames) * layout_.tokensPerFrame * H3_DIT_VIDEO_CHANNELS * 4;
  const size_t audioBytes = static_cast<size_t>(layout_.audio) * H3_DIT_AUDIO_CHANNELS * 4;
  std::memcpy(plan_.view(Tensor::AdaBlocks).contents(), adaBlocks, size_t(H3_DIT_BLOCKS) * kAdaBlockBytes);
  std::memcpy(plan_.view(Tensor::AdaFinal).contents(), adaFinal, size_t(H3_MODALITIES) * 2 * D * 2);
  std::memcpy(plan_.view(Tensor::VideoIn).contents(), videoIn, videoBytes);
  if (audioBytes) std::memcpy(plan_.view(Tensor::AudioIn).contents(), audioIn, audioBytes);
  const ops::IoBuffers io{plan_.view(Tensor::Residual), plan_.view(Tensor::Tags),     plan_.view(Tensor::Text),
                          plan_.view(Tensor::AdaFinal), plan_.view(Tensor::VideoIn),  plan_.view(Tensor::AudioIn),
                          plan_.view(Tensor::VideoOut), plan_.view(Tensor::AudioOut)};
  {
    Encoder e(backend_);
    ops::embed(e, layout_, weights_->io(), io);
    stats.embedMs = run(backend_, e) * 1e3;
  }

  if (stream_) stream_->resetStats();
  const auto blocksStart = splash::AwakeClock::now();
  metal::CommandTicket running;
  WeightStream::Slot *runningSlot = nullptr, *runningQkFix = nullptr;
  for (int block = 0; block < blocks_; ++block) {
    BlockBinding binding = residentBinding(block);
    WeightStream::Slot *slot = nullptr, *qkfixSlot = nullptr;
    if (stream_) {
      slot = &stream_->next(&qkfixSlot);
      binding.qkvg = slot->tensor[WeightStream::Qkvg];
      binding.out = slot->tensor[WeightStream::Out];
      binding.fc1 = slot->tensor[WeightStream::Fc1];
      binding.fc2 = slot->tensor[WeightStream::Fc2];
      binding.outLinear = slot->tensor[WeightStream::OutLinear];
      if (qkfixSlot) binding.qkfixWeight = qkfixSlot->tensor[0];
    }
    Encoder e(backend_);
    encodeBlock(block, binding, e);
    if (block > 0) {
      stats.blocksGpuMs += wait(running) * 1e3;
      if (runningSlot) WeightStream::release(*runningSlot, runningQkFix);
    }
    if (slot) stream_->waitRead(*slot);  // the reader has filled the block's slots
    running = submit(backend_, e);
    runningSlot = slot;
    runningQkFix = qkfixSlot;
    ++stats.commandBuffers;
  }
  stats.blocksGpuMs += wait(running) * 1e3;
  if (runningSlot) WeightStream::release(*runningSlot, runningQkFix);
  stats.blocksSpanMs = splash::millisecondsSince(blocksStart);
  stats.gpuBusyMs = stats.blocksGpuMs;
  stats.gpuIdleMs = std::max(0.0, stats.blocksSpanMs - stats.blocksGpuMs);

  {
    Encoder e(backend_);
    ops::finalLayer(e, layout_, weights_->io(), io);
    stats.finalMs = run(backend_, e) * 1e3;
  }
  std::memcpy(videoOut, plan_.view(Tensor::VideoOut).contents(), videoBytes);
  if (audioBytes) std::memcpy(audioOut, plan_.view(Tensor::AudioOut).contents(), audioBytes);
  if (stream_) {
    const StreamStats s = stream_->stats();
    stats.streamWaitMs = s.waitMs;
    stats.streamWaitMaxMs = s.waitMaxMs;
    stats.streamWaits = s.waits;
    stats.streamReadGb = s.readGb;
    stats.streamReadS = s.readS;
    stats.streamReadMaxMs = s.readMaxMs;
    stats.streamBlocks = s.blocksRead;
  }
  stats.deviceMb = deviceMb(backend_);
  stats.wallMs = splash::millisecondsSince(t0);
}

} // namespace h3

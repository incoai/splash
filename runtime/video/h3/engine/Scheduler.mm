#include "video/h3/engine/Scheduler.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "video/h3/ops/DitIo.hpp"
#include "video/h3/ops/Linear.hpp"
#include "video/h3/ops/QkFix.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>

namespace h3 {
namespace {

constexpr int32_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER;
constexpr size_t kAdaBlockBytes = size_t(H3_MODALITIES) * H3_ADA_ROWS * D * 2;

using ops::AdaRow;

struct Committed {
  id<MTLCommandBuffer> cb;
  const char *name;
  StepClass cls;
};

// Production sink: one command buffer per step, committed at once. With streaming, `pending` counts the block's
// uncompleted command buffers plus a guard held while encoding; the last one to finish frees the slots.
class CommitSink final : public StepSink {
public:
  CommitSink(id<MTLCommandQueue> queue, std::vector<Committed> &out, std::shared_ptr<std::atomic<int>> pending,
             dispatch_semaphore_t release0, dispatch_semaphore_t release1)
      : queue_(queue), out_(out), pending_(std::move(pending)), release0_(release0), release1_(release1) {}

  void step(const char *name, StepClass cls, const Encode &encode) override {
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    encode(encoder);
    [encoder endEncoding];
    if (pending_) {
      ++*pending_;
      std::shared_ptr<std::atomic<int>> pending = pending_;
      dispatch_semaphore_t r0 = release0_, r1 = release1_;
      [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        if (--*pending == 0) {
          dispatch_semaphore_signal(r0);
          if (r1) dispatch_semaphore_signal(r1);
        }
      }];
    }
    [cb commit];
    out_.push_back({cb, name, cls});
  }

  // Drops the encoding guard.
  void finish() {
    if (pending_ && --*pending_ == 0) {
      dispatch_semaphore_signal(release0_);
      if (release1_) dispatch_semaphore_signal(release1_);
    }
  }

private:
  id<MTLCommandQueue> queue_;
  std::vector<Committed> &out_;
  std::shared_ptr<std::atomic<int>> pending_;
  dispatch_semaphore_t release0_, release1_;
};

double msSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

Scheduler::Scheduler(metal::MetalBackend &backend, const model::SequenceLayout &layout, const MemoryPlan &plan,
                     const WindowTable &table, const model::DitWeights *weights, WeightStream *stream,
                     std::set<int> qkfixBlocks)
    : backend_(backend), layout_(layout), plan_(plan), table_(table), weights_(weights), stream_(stream),
      qkfixBlocks_(std::move(qkfixBlocks)), branch_(backend, layout, branchBuffers()) {}

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
  BlockBinding binding{&w, w.qkvg, w.out, w.fc1, w.fc2, w.outLinear, nil, nil};
  if (qkfixBlocks_.count(block)) {
    binding.qkfixWeight = weights_->qkfix(block).wqkv;
    binding.qkfixSplit = weights_->qkfix(block).splitDims;
  }
  return binding;
}

// Step order and the MemoryPlan aliases depend on each other: a step may only overwrite a slab view whose last
// reader has been encoded before it (see the lifetimes in MemoryPlan.mm).
void Scheduler::encodeBlock(int block, const BlockBinding &bind, StepSink &s) const {
  const model::DitBlockWeights &w = *bind.w;
  const bool qkfix = qkfixBlocks_.count(block) != 0;
  const metal::MetalBackend &b = backend_;
  const int32_t M = layout_.rows(), Nin = layout_.innerRows();
  const metal::BufferView x = plan_.view(Tensor::Residual), tags = plan_.view(Tensor::Tags);
  const metal::BufferView rope = plan_.view(Tensor::Rope);
  const metal::BufferView ada = plan_.view(Tensor::AdaBlocks) + size_t(block) * kAdaBlockBytes;
  const metal::BufferView signsHidden = plan_.view(Tensor::SignsHidden), signsFfn = plan_.view(Tensor::SignsFfn);
  const ops::ProducerOut linearIn{plan_.view(Tensor::LinearIn), plan_.view(Tensor::RowScale)};
  const ops::ProducerOut outProjIn{plan_.view(qkfix ? Tensor::OutProjInQkFix : Tensor::OutProjIn),
                                   plan_.view(Tensor::RowScale)};
  const ops::ProducerOut branchIn{plan_.view(Tensor::BranchIn), plan_.view(Tensor::BranchRowScale)};
  const metal::BufferView attentionOut = plan_.view(qkfix ? Tensor::AttentionOutQkFix : Tensor::AttentionOut);
  const metal::BufferView qkfixIn = plan_.view(Tensor::QkFixInput);
  const ops::AttentionBuffers attention = attentionBuffers();
  const ops::QkvTargets qkv = qkvTargets();
  const ops::QkFixTargets qkfixQkv{qkv, plan_.view(Tensor::QkFixQueryExact), plan_.view(Tensor::QkFixKeyExact)};
  const size_t row0 = static_cast<size_t>(layout_.innerRow0());

  // ---- attention half: norm1 -> QKV (+ raw q / k) -> branch gates and decays -> window attention -> out
  s.step("norm1_quantize", StepClass::Other, [&](id<MTLComputeCommandEncoder> e) {
    ops::normQuantize(e, b, x, M, w.norm1, ada, AdaRow::Shift1, AdaRow::Scale1, tags, signsHidden, linearIn);
  });
  if (!qkfix)
    s.step("qkv", StepClass::Linear, [&](id<MTLComputeCommandEncoder> e) {
      ops::qkv(e, b, linearIn, M, bind.qkvg, w.qkvgScale, w.qNorm, w.kNorm, rope, qkv);
    });
  s.step("branch_gates", StepClass::Branch,
         [&](id<MTLComputeCommandEncoder> e) { branch_.gates(e, linearIn, bind.qkvg, w.qkvgScale, w.sgBias); });
  s.step("branch_frame_mean", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) { branch_.frameMean(e, linearIn); });
  s.step("branch_alpha", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) { branch_.alpha(e, w); });
  if (qkfix) {  // fp16 q / k / v from the unrotated weights (the int8 QKV GEMM is skipped)
    s.step("qkfix_normalize", StepClass::Other, [&](id<MTLComputeCommandEncoder> e) {
      ops::qkfixNormalize(e, b, x, M, w.norm1, ada, tags, qkfixIn);
    });
    s.step("qkfix_qkv", StepClass::Linear, [&](id<MTLComputeCommandEncoder> e) {
      ops::qkfixQkv(e, b, qkfixIn, M, bind.qkfixWeight, bind.qkfixSplit, w.qNorm, w.kNorm, rope, qkfixQkv);
    });
  } else {
    s.step("attention_quantize", StepClass::Attention,
           [&](id<MTLComputeCommandEncoder> e) { ops::quantize(e, b, attention); });
  }
  s.step("branch_text_features", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) { branch_.textFeatures(e); });
  // raw k is read before the attention output takes its slab (QKFIX blocks keep O elsewhere and read it later)
  if (!qkfix)
    s.step("branch_key_features", StepClass::Branch,
           [&](id<MTLComputeCommandEncoder> e) { branch_.keyFeatures(e, w); });
  s.step("window_attention", StepClass::Attention, [&](id<MTLComputeCommandEncoder> e) {
    if (qkfix) ops::windowQkFix(e, b, attention, qkfixQkv.qExact, qkfixQkv.kExact, table_, attentionOut);
    else ops::window(e, b, attention, table_, attentionOut);
  });
  s.step("out_quantize", StepClass::Other, [&](id<MTLComputeCommandEncoder> e) {
    ops::gatedQuantize(e, b, attentionOut, plan_.view(Tensor::SoftmaxGate), M, signsFfn, outProjIn);
  });
  s.step("out", StepClass::Linear, [&](id<MTLComputeCommandEncoder> e) {
    ops::gatedResidual(e, b, outProjIn, M, C, bind.out, w.outScale, x, tags, ada, AdaRow::Gate1);
  });

  // ---- linear branch of the inner video rows -> to_out_linear into the same residual (attention gate)
  if (qkfix)
    s.step("branch_key_features", StepClass::Branch,
           [&](id<MTLComputeCommandEncoder> e) { branch_.keyFeatures(e, w); });
  s.step("branch_value_features", StepClass::Branch,
         [&](id<MTLComputeCommandEncoder> e) { branch_.valueFeatures(e, w); });
  s.step("branch_stats", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) { branch_.stats(e); });
  s.step("branch_text_stats", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) { branch_.textStats(e); });
  s.step("branch_inverse", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) { branch_.inverse(e); });
  s.step("branch_text_inverse", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) { branch_.textInverse(e); });
  s.step("branch_scan", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) { branch_.scan(e); });
  s.step("branch_gather", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) { branch_.gather(e); });
  s.step("branch_readout", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) { branch_.readout(e, w); });
  s.step("branch_quantize", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) {
    ops::branchQuantize(e, b, plan_.view(Tensor::Readout), Nin, signsFfn, branchIn);
  });
  s.step("to_out_linear", StepClass::Branch, [&](id<MTLComputeCommandEncoder> e) {
    ops::gatedResidual(e, b, branchIn, Nin, C, bind.outLinear, w.outLinearScale, x + row0 * D * 2, tags + row0, ada,
                       AdaRow::Gate1);
  });

  // ---- MLP half: norm2 -> fc1 + SwiGLU -> fc2
  s.step("norm2_quantize", StepClass::Other, [&](id<MTLComputeCommandEncoder> e) {
    ops::normQuantize(e, b, x, M, w.norm2, ada, AdaRow::Shift2, AdaRow::Scale2, tags, signsHidden, linearIn);
  });
  s.step("fc1", StepClass::Linear, [&](id<MTLComputeCommandEncoder> e) {
    ops::fc1SwiGlu(e, b, linearIn, M, bind.fc1, w.fc1Scale, plan_.view(Tensor::Hidden));
  });
  s.step("fc2_quantize", StepClass::Other, [&](id<MTLComputeCommandEncoder> e) {
    ops::hiddenQuantize(e, b, plan_.view(Tensor::Hidden), M, signsFfn, linearIn);
  });
  s.step("fc2", StepClass::Linear, [&](id<MTLComputeCommandEncoder> e) {
    ops::gatedResidual(e, b, linearIn, M, F, bind.fc2, w.fc2Scale, x, tags, ada, AdaRow::Gate2);
  });
}

Status Scheduler::forward(const uint16_t *adaBlocks, const uint16_t *adaFinal, const float *videoIn,
                          const float *audioIn, float *videoOut, float *audioOut, ForwardStats &stats) {
  const auto t0 = std::chrono::steady_clock::now();
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
  H3_TRY(backend_.runSync([&](id<MTLComputeCommandEncoder> e) { ops::embed(e, backend_, layout_, weights_->io(), io); },
                          &stats.embedMs));

  if (stream_) stream_->resetStats();
  std::vector<Committed> cbs;
  cbs.reserve(H3_DIT_BLOCKS * 32);
  for (int block = 0; block < blocks_; ++block) {
    BlockBinding binding = residentBinding(block);
    std::shared_ptr<std::atomic<int>> pending;
    dispatch_semaphore_t release0 = nullptr, release1 = nullptr;
    if (stream_) {  // waits until the reader has filled the block's slot
      WeightStream::Slot *qkfixSlot = nullptr;
      WeightStream::Slot &slot = stream_->acquire(&qkfixSlot);
      binding.qkvg = slot.tensor[WeightStream::Qkvg];
      binding.out = slot.tensor[WeightStream::Out];
      binding.fc1 = slot.tensor[WeightStream::Fc1];
      binding.fc2 = slot.tensor[WeightStream::Fc2];
      binding.outLinear = slot.tensor[WeightStream::OutLinear];
      if (qkfixSlot) binding.qkfixWeight = qkfixSlot->tensor[0];
      pending = std::make_shared<std::atomic<int>>(1);
      release0 = slot.free;
      release1 = qkfixSlot ? qkfixSlot->free : nullptr;
    }
    CommitSink sink(backend_.queue(), cbs, pending, release0, release1);
    encodeBlock(block, binding, sink);
    sink.finish();
  }
  double byClass[4] = {0, 0, 0, 0};
  for (size_t j = 0; j < cbs.size(); ++j) {
    [cbs[j].cb waitUntilCompleted];
    if (cbs[j].cb.error)
      return error(Code::Gpu, "block step %s: %s", cbs[j].name, cbs[j].cb.error.localizedDescription.UTF8String);
    const double ms = (cbs[j].cb.GPUEndTime - cbs[j].cb.GPUStartTime) * 1e3;
    byClass[static_cast<int>(cbs[j].cls)] += ms;
    stats.blocksGpuMs += ms;
  }
  stats.blocksSpanMs = (cbs.back().cb.GPUEndTime - cbs.front().cb.GPUStartTime) * 1e3;
  {  // GPU busy = union of the command buffers' [start, end]; idle gaps inside the span
    std::vector<std::pair<double, double>> intervals;
    intervals.reserve(cbs.size());
    for (const Committed &c : cbs) intervals.push_back({c.cb.GPUStartTime, c.cb.GPUEndTime});
    std::sort(intervals.begin(), intervals.end());
    double busy = 0, gapMax = 0, start = intervals[0].first, end = intervals[0].second;
    for (size_t j = 1; j < intervals.size(); ++j) {
      if (intervals[j].first > end) {
        busy += end - start;
        const double gap = (intervals[j].first - end) * 1e3;
        gapMax = std::max(gapMax, gap);
        if (gap > 1.0) ++stats.gpuGaps;
        start = intervals[j].first;
      }
      end = std::max(end, intervals[j].second);
    }
    busy += end - start;
    stats.gpuBusyMs = busy * 1e3;
    stats.gpuIdleMs = stats.blocksSpanMs - stats.gpuBusyMs;
    stats.gpuGapMaxMs = gapMax;
  }
  stats.linearMs = byClass[0];
  stats.attentionMs = byClass[1];
  stats.otherMs = byClass[2];
  stats.branchMs = byClass[3];
  stats.commandBuffers = static_cast<int32_t>(cbs.size());

  H3_TRY(backend_.runSync(
      [&](id<MTLComputeCommandEncoder> e) { ops::finalLayer(e, backend_, layout_, weights_->io(), io); },
      &stats.finalMs));
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
  stats.deviceMb = backend_.allocatedMb();
  stats.wallMs = msSince(t0);
  return {};
}

} // namespace h3

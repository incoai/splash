#include "video/stages/DenoisingStage.hpp"

#include "AwakeClock.hpp"
#include "StderrLine.hpp"
#include "metal/abi/video/Layout.h"
#include "video/metal/DeviceProfile.hpp"
#include "video/model/Hadamard.hpp"
#include "video/ops/DitIo.hpp"
#include "video/ops/Linear.hpp"
#include "video/ops/QkFix.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

namespace splash::video {
namespace {

constexpr int32_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER;
constexpr size_t kAdaStepFinal = size_t(H3_MODALITIES) * 2 * H3_DIT_HIDDEN;

using model::Tensor;
using ops::AdaRow;

// The packed layout of the inputs, which must be whole: their tags those of [text | conditioning | audio | video]
// rows, text rows text (1) or video (0) for the rows of a keyframe's vision block.
ops::SequenceLayout layoutOf(const ops::VideoGeometry &g, const DitInputs &inputs) {
  if (inputs.textRows < 1 || !inputs.tags || !inputs.rope || !inputs.text)
    throw std::invalid_argument("DiT inputs incomplete");
  if (inputs.condRows < 0 || inputs.condRows % g.tokensPerFrame || (inputs.condRows && !inputs.cond))
    throw std::invalid_argument(
        std::format("conditioning rows: whole keyframes of {} rows and their latents", g.tokensPerFrame));
  ops::SequenceLayout lay;
  lay.text = inputs.textRows;
  lay.cond = inputs.condRows;
  lay.audio = g.audioRows;
  lay.frames = g.latentFrames;
  lay.tokensPerFrame = g.tokensPerFrame;
  lay.gridH = g.gridH;
  lay.gridW = g.gridW;
  for (int32_t r = 0; r < lay.rows(); ++r) {
    const uint8_t want = r < lay.text ? H3_TAG_TEXT
                                      : (r < lay.audioStart() ? H3_TAG_COND
                                                              : (r < lay.videoStart() ? H3_TAG_AUDIO : H3_TAG_VIDEO));
    if (inputs.tags[r] != want && !(r < lay.text && inputs.tags[r] == H3_TAG_VIDEO))
      throw std::invalid_argument(
          std::format("layout tags differ from [text | conditioning | audio | video] at row {}", r));
  }
  return lay;
}

ops::WindowTable uploadedTable(metal::MetalBackend &backend, const ops::SequenceLayout &layout,
                               const ClipOptions &options) {
  if (options.windowRadius < 0) throw std::invalid_argument("window radius must be >= 0");
  if (options.windowChunk < 0) throw std::invalid_argument("window chunk must be >= 0");
  ops::WindowTable table(layout, options.windowChunk, options.windowRadius);
  table.upload(backend);
  return table;
}

std::unique_ptr<model::WeightStream> startedStream(metal::MetalBackend &backend, const ClipOptions &options,
                                                   const model::DitWeights &weights, const std::set<int> &qkfix) {
  if (!options.stream) return nullptr;
  if (options.streamSlots < 1) throw std::invalid_argument("stream slots must be >= 1");
  auto stream = std::make_unique<model::WeightStream>(backend, weights.streamed(), options.streamSlots);
  stream->start(backend, H3_DIT_BLOCKS, qkfix);
  return stream;
}

// Rectified-flow Euler step (eta 0), float32 op by op as the numpy reference:
//   x0 = x + (1 - t) v;  r = sigma_next / sigma;  x' = r x + (1 - r) x0
void euler(std::vector<float> &x, const std::vector<float> &v, float t, float sigma, float sigmaNext) {
  const float oneMinusT = 1.0f - t;
  const float r = sigmaNext / sigma;
  const float oneMinusR = 1.0f - r;
  for (size_t i = 0; i < x.size(); ++i) {
    const float x0 = x[i] + oneMinusT * v[i];
    x[i] = r * x[i] + oneMinusR * x0;
  }
}

bool allFinite(const std::vector<float> &v) {
  for (float f : v)
    if (!std::isfinite(f)) return false;
  return true;
}

} // namespace

std::set<int> qkfixBlocks(const ClipOptions &options) {
  if (!options.qkfix) return {};
  if (options.qkfixBlocks.empty()) throw std::invalid_argument("QKFIX on without blocks");
  for (int32_t b : options.qkfixBlocks)
    if (b < 0 || b >= H3_DIT_BLOCKS) throw std::invalid_argument(std::format("QKFIX block {} out of range", b));
  return {options.qkfixBlocks.begin(), options.qkfixBlocks.end()};
}

Dit::Dit(metal::MetalBackend &backend, const ClipOptions &options, const ops::VideoGeometry &geometry,
         const DitInputs &inputs, std::unique_ptr<model::DitWeights> weights)
    : backend_(backend), layout_(layoutOf(geometry, inputs)), qkfixBlocks_(qkfixBlocks(options)),
      weights_(std::move(weights)), table_(uploadedTable(backend, layout_, options)),
      stream_(startedStream(backend, options, *weights_, qkfixBlocks_)),
      plan_(backend, layout_, table_.partialSlots(), !qkfixBlocks_.empty()), branch_(layout_, branchBuffers()) {
  const int32_t M = layout_.rows();
  std::memcpy(plan_.view(Tensor::Tags).contents(), inputs.tags, M);
  std::memcpy(plan_.view(Tensor::Rope).contents(), inputs.rope, size_t(M) * H3_DIT_ROPE * 2);
  std::memcpy(plan_.view(Tensor::Text).contents(), inputs.text, size_t(layout_.text) * H3_DIT_HIDDEN * 2);
  const std::vector<int8_t> signsHidden = model::hadamardSigns(H3_DIT_HIDDEN);
  const std::vector<int8_t> signsFfn = model::hadamardSigns(H3_DIT_FFN);
  std::memcpy(plan_.view(Tensor::SignsHidden).contents(), signsHidden.data(), signsHidden.size());
  std::memcpy(plan_.view(Tensor::SignsFfn).contents(), signsFfn.data(), signsFfn.size());
  const auto bounds = table_.branchBounds();
  std::memcpy(plan_.view(Tensor::Bounds).contents(), bounds.data(), bounds.size() * sizeof(bounds[0]));
  // Keyframe conditioning rows fp32 [C][96], fixed for the request.
  cond_.assign(inputs.cond, inputs.cond + size_t(layout_.cond) * H3_DIT_VIDEO_CHANNELS);
  const double weightsMb = (weights_->residentBytes() + (stream_ ? stream_->slotBytes() : 0)) / 1048576.0;
  splash::logLine(std::format("H3 DiT M={} L={} C={} Na={} F={} Spad={} window chunk={} radius={} density={:.4f} "
                              "items={} chunked_groups={} qkfix={} stream_slots={} weights_MB={:.0f} "
                              "activations_MB={:.0f} device_MB={:.0f}",
                              M, layout_.text, layout_.cond, layout_.audio, layout_.frames, layout_.paddedRows(),
                              options.windowChunk, options.windowRadius, table_.density(), table_.items().size(),
                              table_.chunkedGroups(), qkfixBlocks_.size(), stream_ ? stream_->slots() : 0, weightsMb,
                              plan_.totalBytes() / 1048576.0, deviceMb(backend_)));
}

ops::AttentionBuffers Dit::attentionBuffers() const {
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

ops::BranchBuffers Dit::branchBuffers() const {
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

ops::QkvTargets Dit::qkvTargets() const {
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

Dit::BlockBinding Dit::residentBinding(int block) const {
  const model::DitBlockWeights &w = weights_->block(block);
  BlockBinding binding{&w, w.qkvg, w.out, w.fc1, w.fc2, w.outLinear, {}, {}};
  if (qkfixBlocks_.count(block)) {
    binding.qkfixWeight = weights_->qkfix(block).wqkv;
    binding.qkfixSplit = weights_->qkfix(block).splitDims;
  }
  return binding;
}

// Kernel order and the MemoryPlan aliases depend on each other: a kernel may only overwrite a slab view whose last
// reader has been encoded before it (see the lifetimes in model/MemoryPlan.mm).
void Dit::encodeBlock(int block, const BlockBinding &bind, Encoder &e) const {
  const model::DitBlockWeights &w = *bind.w;
  const bool qkfix = qkfixBlocks_.count(block) != 0;
  const int32_t M = layout_.rows(), Nin = layout_.innerRows();
  const BufferView x = plan_.view(Tensor::Residual), tags = plan_.view(Tensor::Tags);
  const BufferView rope = plan_.view(Tensor::Rope);
  const size_t adaBlockBytes = size_t(layout_.adaModalities()) * H3_ADA_ROWS * D * 2;
  const BufferView ada = plan_.view(Tensor::AdaBlocks) + size_t(block) * adaBlockBytes;
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
  branch_.alpha(e, w.alphaDown, w.alphaUp, w.dtBias, w.aLog);
  if (qkfix) {  // fp16 q / k / v from the unrotated weights (the int8 QKV GEMM is skipped)
    ops::qkfixNormalize(e, x, M, w.norm1, ada, tags, qkfixIn);
    ops::qkfixQkv(e, qkfixIn, M, bind.qkfixWeight, bind.qkfixSplit, w.qNorm, w.kNorm, rope, qkfixQkv);
  } else {
    ops::quantize(e, attention);
  }
  branch_.textFeatures(e);
  // raw k is read before the attention output takes its slab (QKFIX blocks keep O elsewhere and read it later)
  if (!qkfix) branch_.keyFeatures(e, w.kSpatial, w.kTemporal);
  if (qkfix) ops::windowQkFix(e, attention, qkfixQkv.qExact, qkfixQkv.kExact, table_, attentionOut);
  else ops::window(e, attention, table_, attentionOut);
  ops::gatedQuantize(e, attentionOut, plan_.view(Tensor::SoftmaxGate), M, signsFfn, outProjIn);
  ops::gatedResidual(e, outProjIn, M, C, bind.out, w.outScale, x, tags, ada, AdaRow::Gate1);

  // ---- linear branch of the inner video rows -> to_out_linear into the same residual (attention gate)
  if (qkfix) branch_.keyFeatures(e, w.kSpatial, w.kTemporal);
  branch_.valueFeatures(e, w.vSpatial, w.vTemporal);
  branch_.stats(e);
  branch_.textStats(e);
  branch_.inverse(e);
  branch_.textInverse(e);
  branch_.scan(e);
  branch_.gather(e);
  branch_.readout(e, w.branchNorm, w.ogUp, w.ogBias);
  ops::branchQuantize(e, plan_.view(Tensor::Readout), Nin, signsFfn, branchIn);
  ops::gatedResidual(e, branchIn, Nin, C, bind.outLinear, w.outLinearScale, x + row0 * D * 2, tags + row0, ada,
                     AdaRow::Gate1);

  // ---- MLP half: norm2 -> fc1 + SwiGLU -> fc2
  ops::normQuantize(e, x, M, w.norm2, ada, AdaRow::Shift2, AdaRow::Scale2, tags, signsHidden, linearIn);
  ops::fc1SwiGlu(e, linearIn, M, bind.fc1, w.fc1Scale, plan_.view(Tensor::Hidden));
  ops::hiddenQuantize(e, plan_.view(Tensor::Hidden), M, signsFfn, linearIn);
  ops::gatedResidual(e, linearIn, M, F, bind.fc2, w.fc2Scale, x, tags, ada, AdaRow::Gate2);
}

void Dit::forward(const uint16_t *adaBlocks, const uint16_t *adaFinal, const float *videoIn, const float *audioIn,
                  float *videoOut, float *audioOut, ForwardStats &stats) {
  const auto t0 = splash::AwakeClock::now();
  stats = ForwardStats{};
  const size_t videoBytes = static_cast<size_t>(layout_.frames) * layout_.tokensPerFrame * H3_DIT_VIDEO_CHANNELS * 4;
  const size_t audioBytes = static_cast<size_t>(layout_.audio) * H3_DIT_AUDIO_CHANNELS * 4;
  std::memcpy(plan_.view(Tensor::AdaBlocks).contents(), adaBlocks,
              size_t(H3_DIT_BLOCKS) * layout_.adaModalities() * H3_ADA_ROWS * D * 2);
  std::memcpy(plan_.view(Tensor::AdaFinal).contents(), adaFinal, size_t(H3_MODALITIES) * 2 * D * 2);
  // conditioning rows first (VideoIn shares its slab with raw q, so they are copied every forward), then the target
  const size_t condBytes = cond_.size() * 4;
  if (condBytes) std::memcpy(plan_.view(Tensor::VideoIn).contents(), cond_.data(), condBytes);
  std::memcpy(static_cast<uint8_t *>(plan_.view(Tensor::VideoIn).contents()) + condBytes, videoIn, videoBytes);
  if (audioBytes) std::memcpy(plan_.view(Tensor::AudioIn).contents(), audioIn, audioBytes);
  const ops::IoBuffers io{plan_.view(Tensor::Residual), plan_.view(Tensor::Tags),     plan_.view(Tensor::Text),
                          plan_.view(Tensor::AdaFinal), plan_.view(Tensor::VideoIn),  plan_.view(Tensor::AudioIn),
                          plan_.view(Tensor::VideoOut), plan_.view(Tensor::AudioOut)};
  const model::DitIoWeights &w = weights_->io();
  {
    Encoder e(backend_);
    ops::embed(e, layout_, w.videoProj, w.videoProjBias, w.audioProj, w.audioProjBias, io);
    stats.embedMs = run(backend_, e) * 1e3;
  }

  if (stream_) stream_->resetStats();
  const auto blocksStart = splash::AwakeClock::now();
  metal::CommandTicket running;
  model::WeightStream::Slot *runningSlot = nullptr, *runningQkFix = nullptr;
  for (int block = 0; block < blocks_; ++block) {
    BlockBinding binding = residentBinding(block);
    model::WeightStream::Slot *slot = nullptr, *qkfixSlot = nullptr;
    if (stream_) {
      slot = &stream_->next(&qkfixSlot);
      binding.qkvg = slot->tensor[model::WeightStream::Qkvg];
      binding.out = slot->tensor[model::WeightStream::Out];
      binding.fc1 = slot->tensor[model::WeightStream::Fc1];
      binding.fc2 = slot->tensor[model::WeightStream::Fc2];
      binding.outLinear = slot->tensor[model::WeightStream::OutLinear];
      if (qkfixSlot) binding.qkfixWeight = qkfixSlot->tensor[0];
    }
    Encoder e(backend_);
    encodeBlock(block, binding, e);
    if (block > 0) {
      stats.blocksGpuMs += wait(running) * 1e3;
      if (runningSlot) model::WeightStream::release(*runningSlot, runningQkFix);
    }
    if (slot) stream_->waitRead(*slot);  // the reader has filled the block's slots
    running = submit(backend_, e);
    runningSlot = slot;
    runningQkFix = qkfixSlot;
  }
  stats.blocksGpuMs += wait(running) * 1e3;
  if (runningSlot) model::WeightStream::release(*runningSlot, runningQkFix);
  stats.blocksSpanMs = splash::millisecondsSince(blocksStart);

  {
    Encoder e(backend_);
    ops::finalLayer(e, layout_, w.finalNorm, w.videoOut, w.videoOutBias, w.audioOut, w.audioOutBias, io);
    stats.finalMs = run(backend_, e) * 1e3;
  }
  std::memcpy(videoOut, plan_.view(Tensor::VideoOut).contents(), videoBytes);
  if (audioBytes) std::memcpy(audioOut, plan_.view(Tensor::AudioOut).contents(), audioBytes);
  if (stream_) stats.stream = stream_->stats();
  stats.deviceMb = deviceMb(backend_);
  stats.wallMs = splash::millisecondsSince(t0);
}

void DenoisingStage::forward(VideoBatch &batch, StageEvents &events) {
  const model::Prepared &p = batch.prepared;
  const ClipOptions &options = batch.options;
  const ops::VideoGeometry &g = batch.geometry;
  const char *variant = options.variant == model::Variant::Turbo ? "dit-turbo" : "dit-default";
  {
    Dit dit(backend_, options, g, {p.textRows, p.tags.data(), p.rope.data(), p.text.data(), p.condRows, p.cond.data()},
            model::DitWeights::load(backend_, (weights_ / variant).string(), qkfixBlocks(options), options.stream));
    const size_t videoCount = size_t(g.videoRows()) * H3_DIT_VIDEO_CHANNELS;
    const size_t audioCount = size_t(g.audioRows) * H3_DIT_AUDIO_CHANNELS;
    std::vector<float> video(p.videoNoise.begin(), p.videoNoise.begin() + videoCount);
    std::vector<float> audio(p.audioNoise.begin(), p.audioNoise.begin() + audioCount);
    std::vector<float> videoV(videoCount), audioV(audioCount);
    const size_t adaStepBlocks = size_t(H3_DIT_BLOCKS) * dit.layout().adaModalities() * H3_ADA_ROWS * H3_DIT_HIDDEN;
    for (int32_t step = 0; step < p.steps; ++step) {
      ForwardStats stats;
      dit.forward(p.adaBlocks.data() + size_t(step) * adaStepBlocks, p.adaFinal.data() + size_t(step) * kAdaStepFinal,
                  video.data(), audio.data(), videoV.data(), audioV.data(), stats);
      if (!allFinite(videoV) || !allFinite(audioV)) splash::logWarning("H3 non-finite velocity at step ", step);
      euler(video, videoV, p.vT[step], p.vSigmas[step], p.vSigmas[step + 1]);
      euler(audio, audioV, p.aT[step], p.aSigmas[step], p.aSigmas[step + 1]);
      splash::logLine(std::format("H3 step {}/{} wall {:.1f}s blocks {:.1f}s (GPU {:.1f}s) stream wait {:.2f}s "
                                  "({} waits) read {:.1f} GB in {:.1f}s device_MB={:.0f}",
                                  step + 1, p.steps, stats.wallMs / 1e3, stats.blocksSpanMs / 1e3,
                                  stats.blocksGpuMs / 1e3, stats.stream.waitMs / 1e3, stats.stream.waits,
                                  stats.stream.readGb, stats.stream.readS, stats.deviceMb));
      events.progress(static_cast<uint32_t>(step + 1));
      if (events.cancelled()) throw Cancelled{};
    }
    batch.videoLatents = std::move(video);
    batch.audioLatents = std::move(audio);
  }
  batch.prepared = {};
}

} // namespace splash::video

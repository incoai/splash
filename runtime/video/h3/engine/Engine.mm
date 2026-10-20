#include "video/h3/engine/Engine.hpp"

#include "video/h3/engine/DeviceProfile.hpp"
#include "metal/abi/video/Layout.h"
#include "video/h3/model/AudioWeights.hpp"
#include "video/h3/model/Hadamard.hpp"
#include "video/h3/model/Synthetic.hpp"
#include "video/h3/model/PreviewWeights.hpp"
#include "video/h3/model/VaeLatents.hpp"
#include "video/h3/model/VaeWeights.hpp"
#include "video/h3/ops/DitIo.hpp"
#include "video/h3/ops/LinearBranch.hpp"
#include "video/h3/ops/QkFix.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace h3 {
namespace {

constexpr size_t kAdaStepBlocks = size_t(H3_DIT_BLOCKS) * H3_MODALITIES * H3_ADA_ROWS * H3_DIT_HIDDEN;
constexpr size_t kAdaStepFinal = size_t(H3_MODALITIES) * 2 * H3_DIT_HIDDEN;

Status validate(const Request &r) {
  if (r.windowRadius < 0) return error(Code::Argument, "window radius must be >= 0");
  if (r.windowChunk < 0) return error(Code::Argument, "window chunk must be >= 0");
  if (r.qkfix && r.qkfixBlocks.empty()) return error(Code::Argument, "QKFIX on without blocks");
  for (int32_t b : r.qkfixBlocks)
    if (b < 0 || b >= H3_DIT_BLOCKS) return error(Code::Argument, "QKFIX block %d out of range", b);
  if (r.stream && r.streamSlots < 1) return error(Code::Argument, "stream slots must be >= 1");
  return {};
}

model::SequenceLayout layoutOf(const model::VideoGeometry &g, int32_t textRows) {
  model::SequenceLayout lay;
  lay.text = textRows;
  lay.audio = g.audioRows;
  lay.frames = g.latentFrames;
  lay.tokensPerFrame = g.tokensPerFrame;
  lay.gridH = g.gridH;
  lay.gridW = g.gridW;
  return lay;
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

Status Engine::create(const EngineConfig &config, const Request &request, std::unique_ptr<Engine> &out) {
  auto engine = std::unique_ptr<Engine>(new Engine());
  engine->config_ = config;
  engine->request_ = request;
  H3_TRY(validate(request));
  H3_TRY(model::VideoGeometry::of(request.frames, request.width, request.height, engine->geometry_));
  H3_TRY(model::planVaeDecode(engine->geometry_, engine->vaePlan_));
  H3_TRY(metal::MetalBackend::create(config.metallib, device::kDitQueueDepth, engine->backend_));
  out = std::move(engine);
  return {};
}

Engine::~Engine() = default;

std::string Engine::ditDirectory() const {
  return config_.weights + (request_.variant == Variant::Turbo ? "/dit-turbo" : "/dit-default");
}

Status Engine::windowTable(const Request &request, int32_t textRows, WindowTable &out) {
  model::VideoGeometry g;
  H3_TRY(model::VideoGeometry::of(request.frames, request.width, request.height, g));
  return WindowTable::build(layoutOf(g, textRows), request.windowChunk, request.windowRadius, out);
}

Status Engine::prep(const model::PrepInputs &inputs, model::Prepared &out, const std::string &dump) {
  H3_TRY(model::Prep::run(*backend_, config_.weights, request_.variant, geometry_, inputs, out, dump));
  log_.info("H3 prep M=%d L=%d steps=%d distinct_t=%zu total %.1fs (layout %.2fs, text+temb+noise %.2fs, adaLN %.1fs "
            "reading %.1f GB) device_MB=%.0f",
            out.rows, out.textRows, out.steps, out.table.size(), out.totalS, out.layoutS, out.textS, out.adalnS,
            out.adalnReadGb, backend_->allocatedMb());
  return {};
}

Status Engine::loadDit(const DitInputs &inputs) { return loadDitWith(inputs, nullptr); }

Status Engine::loadDitSynthetic(int32_t textRows, int32_t blocks) {
  if (request_.stream) return error(Code::Argument, "synthetic weights are resident (no --stream)");
  model::Prepared p;
  model::Prep::layout(geometry_, textRows, p);
  H3_TRY(model::Prep::rope(*backend_, p));
  std::vector<uint16_t> text(size_t(textRows) * H3_DIT_HIDDEN);
  model::syntheticBf16(model::syntheticSeed(model::kSyntheticInputGroup, 0), text.size(), 1.0f, 0.0f, text.data());
  if (blocks < 1 || blocks > H3_DIT_BLOCKS) return error(Code::Argument, "synthetic: blocks must be 1..50");
  std::set<int> qkfix, used;
  for (int b = 0; b < blocks; ++b) used.insert(b);
  if (request_.qkfix)
    for (int b : request_.qkfixBlocks)
      if (b < blocks) qkfix.insert(b);
  std::unique_ptr<model::DitWeights> weights;
  H3_TRY(model::DitWeights::synthetic(*backend_, qkfix, used, weights));
  H3_TRY(loadDitWith({textRows, p.tags.data(), p.rope.data(), text.data()}, std::move(weights)));
  dit_->scheduler->setBlocks(blocks);
  return {};
}

Status Engine::loadDitWith(const DitInputs &inputs, std::unique_ptr<model::DitWeights> synthetic) {
  if (dit_) return error(Code::State, "DiT already loaded");
  if (inputs.textRows < 1 || !inputs.tags || !inputs.rope || !inputs.text)
    return error(Code::Argument, "DiT inputs incomplete");
  metal::MetalBackend &b = *backend_;
  auto dit = std::make_unique<DitState>();
  dit->layout = layoutOf(geometry_, inputs.textRows);
  const model::SequenceLayout &lay = dit->layout;
  const int32_t M = lay.rows();
  std::set<int> qkfix;
  if (request_.qkfix) qkfix.insert(request_.qkfixBlocks.begin(), request_.qkfixBlocks.end());
  // modality tags must match the packed layout [text | audio | video]
  for (int32_t r = 0; r < M; ++r) {
    const uint8_t want = r < lay.text ? 1 : (r < lay.videoStart() ? 2 : 0);
    if (inputs.tags[r] != want) return error(Code::Argument, "layout tags differ from [text | audio | video] at row %d", r);
  }
  H3_TRY(WindowTable::build(lay, request_.windowChunk, request_.windowRadius, dit->table));
  H3_TRY(dit->table.upload(b));
  if (synthetic) dit->weights = std::move(synthetic);
  else H3_TRY(model::DitWeights::load(b, ditDirectory(), qkfix, request_.stream, dit->weights));
  if (request_.stream) {
    const std::string image = request_.streamImage.empty() ? ditDirectory() + "/stream.bin" : request_.streamImage;
    H3_TRY(WeightStream::open(b, image, request_.streamSlots, dit->stream));
    WeightStream &s = *dit->stream;
    const uint64_t D = H3_DIT_HIDDEN, F = H3_DIT_FFN, C = H3_DIT_INNER;
    const uint64_t expected[WeightStream::kTensors] = {(H3_DIT_QKV + H3_DIT_GATES) * D, D * C, 2 * F * D, D * F, D * C};
    for (int t = 0; t < WeightStream::kTensors; ++t)
      if (s.tensorBytes(static_cast<WeightStream::Tensor>(t)) != expected[t])
        return error(Code::Io, "stream image %s: tensor %d has an unexpected size", image.c_str(), t);
    for (const model::StreamedTensor &t : dit->weights->streamed()) {
      uint64_t offset = 0;
      if (t.slotTensor >= 0) offset = s.tensorOffset(t.block, static_cast<WeightStream::Tensor>(t.slotTensor)) + t.slotOffset;
      else H3_TRY(s.qkfixOffset(t.block, offset));
      H3_TRY(s.verify(*t.file, t.info.offset, t.info.bytes, offset, t.name.c_str(), t.block));
    }
    H3_TRY(s.start(b, H3_DIT_BLOCKS, qkfix));
  }
  H3_TRY(MemoryPlan::create(b, lay, dit->table.partialSlots(), !qkfix.empty(), dit->plan));
  const MemoryPlan &plan = dit->plan;
  std::memcpy(plan.view(Tensor::Tags).contents(), inputs.tags, M);
  std::memcpy(plan.view(Tensor::Rope).contents(), inputs.rope, size_t(M) * H3_DIT_ROPE * 2);
  std::memcpy(plan.view(Tensor::Text).contents(), inputs.text, size_t(lay.text) * H3_DIT_HIDDEN * 2);
  const std::vector<int8_t> signsHidden = model::hadamardSigns(H3_DIT_HIDDEN), signsFfn = model::hadamardSigns(H3_DIT_FFN);
  std::memcpy(plan.view(Tensor::SignsHidden).contents(), signsHidden.data(), signsHidden.size());
  std::memcpy(plan.view(Tensor::SignsFfn).contents(), signsFfn.data(), signsFfn.size());
  const auto bounds = dit->table.branchBounds();
  std::memcpy(plan.view(Tensor::Bounds).contents(), bounds.data(), bounds.size() * sizeof(bounds[0]));
  std::vector<std::string> kernels = ops::linearKernels();
  for (const auto &list : {ops::attentionKernels(), ops::qkfixKernels(), ops::branchKernels(), ops::ioKernels()})
    kernels.insert(kernels.end(), list.begin(), list.end());
  H3_TRY(b.prepare(kernels));
  dit->scheduler = std::make_unique<Scheduler>(b, lay, dit->plan, dit->table, dit->weights.get(), dit->stream.get(), qkfix);
  dit_ = std::move(dit);
  DitInfo info;
  H3_TRY(ditInfo(info));
  log_.info("H3 DiT M=%d L=%d Na=%d F=%d Spad=%d window chunk=%d radius=%d density=%.4f items=%d chunked_groups=%d "
            "qkfix=%zu stream_slots=%d weights_MB=%.0f activations_MB=%.0f device_MB=%.0f",
            M, lay.text, lay.audio, lay.frames, lay.paddedRows(), request_.windowChunk, request_.windowRadius,
            info.density, info.windowItems, info.chunkedGroups, qkfix.size(), info.streamSlots, info.weightsMb,
            info.activationsMb, info.deviceMb);
  dit_->plan.log(log_);
  return {};
}

Status Engine::ditInfo(DitInfo &out) const {
  if (!dit_) return error(Code::State, "DiT not loaded");
  out.rows = dit_->layout.rows();
  out.paddedRows = dit_->layout.paddedRows();
  out.windowItems = static_cast<int32_t>(dit_->table.items().size());
  out.chunkedGroups = dit_->table.chunkedGroups();
  out.partialSlots = dit_->table.partialSlots();
  out.density = dit_->table.density();
  out.streamSlots = dit_->stream ? dit_->stream->slots() : 0;
  out.weightsMb = (dit_->weights->residentBytes() + (dit_->stream ? dit_->stream->slotBytes() : 0)) / 1048576.0;
  out.activationsMb = dit_->plan.totalBytes() / 1048576.0;
  out.deviceMb = backend_->allocatedMb();
  return {};
}

Status Engine::forward(const uint16_t *adaBlocks, const uint16_t *adaFinal, const float *videoIn, const float *audioIn,
                       float *videoOut, float *audioOut, ForwardStats &stats) {
  if (!dit_) return error(Code::State, "DiT not loaded");
  return dit_->scheduler->forward(adaBlocks, adaFinal, videoIn, audioIn, videoOut, audioOut, stats);
}

Status Engine::denoise(const Schedule &s, const float *videoNoise, const float *audioNoise, float *videoOut,
                       float *audioOut, const StepCallback &onStep) {
  if (!dit_) return error(Code::State, "DiT not loaded");
  if (s.steps < 1 || !s.adaBlocks || !s.adaFinal || !s.videoT || !s.audioT || !s.videoSigmas || !s.audioSigmas)
    return error(Code::Argument, "schedule incomplete");
  const size_t videoCount = size_t(geometry_.videoRows()) * H3_DIT_VIDEO_CHANNELS;
  const size_t audioCount = size_t(geometry_.audioRows) * H3_DIT_AUDIO_CHANNELS;
  std::vector<float> video(videoNoise, videoNoise + videoCount), audio(audioNoise, audioNoise + audioCount);
  std::vector<float> videoV(videoCount), audioV(audioCount);
  for (int32_t step = 0; step < s.steps; ++step) {
    ForwardStats stats;
    H3_TRY(forward(s.adaBlocks + size_t(step) * kAdaStepBlocks, s.adaFinal + size_t(step) * kAdaStepFinal,
                   video.data(), audio.data(), videoV.data(), audioV.data(), stats));
    if (!allFinite(videoV) || !allFinite(audioV)) log_.info("H3 non-finite velocity at step %d", step);
    euler(video, videoV, s.videoT[step], s.videoSigmas[step], s.videoSigmas[step + 1]);
    euler(audio, audioV, s.audioT[step], s.audioSigmas[step], s.audioSigmas[step + 1]);
    if (onStep) onStep(step, stats, video.data(), audio.data());
  }
  std::memcpy(videoOut, video.data(), videoCount * 4);
  std::memcpy(audioOut, audio.data(), audioCount * 4);
  return {};
}

void Engine::releaseDit() { dit_.reset(); }

Status Engine::loadVae() {
  if (vae_) return {};
  std::unique_ptr<model::VaeWeights> weights;
  const char *tag = request_.vaePrecision == VaePrecision::W8 ? "w8" : "f16";
  H3_TRY(model::VaeWeights::load(*backend_, config_.weights + "/vae", tag, weights));
  return ops::VaeDecoder::create(*backend_, std::move(weights), vae_);
}

void Engine::vaeLatents(const float *videoRows, std::vector<float> &z) const {
  const model::VideoGeometry &g = geometry_;
  const size_t frameFloats = size_t(g.latentHeight) * g.latentWidth * H3_VAE_LATENT;
  z.assign(frameFloats * (g.latentFrames + vaePlan_.padTokens), 0.0f);
  float mean[24], std[24];
  for (int c = 0; c < 24; ++c) {
    mean[c] = static_cast<float>(model::kVaeLatentMean[c]);
    std[c] = static_cast<float>(model::kVaeLatentStd[c]);
  }
  model::unpatchifyVideo(videoRows, g, std, mean, z.data());
  for (int32_t k = 0; k < vaePlan_.padTokens; ++k)
    std::memcpy(z.data() + (g.latentFrames + k) * frameFloats, z.data() + (g.latentFrames - 1) * frameFloats,
                frameFloats * 4);
}

Status Engine::decode(const float *videoRows, uint8_t *frames, DecodeStats &stats) {
  H3_TRY(loadVae());
  std::vector<float> z;
  vaeLatents(videoRows, z);
  const int32_t latentFrames = geometry_.latentFrames + vaePlan_.padTokens;
  H3_TRY(vae_->decode(z.data(), latentFrames, geometry_, vaePlan_, frames, stats));
  log_.info("H3 decode %d frames wall %.1fs gpu %.1fs nonfinite %lld device_MB %.0f", vaePlan_.outputFrames(),
            stats.wallS, stats.gpuS, static_cast<long long>(stats.nonfinite), stats.deviceMb);
  return {};
}

Status Engine::profileDecode(const float *videoRows, int32_t clips, DecodeStats &stats,
                             std::vector<std::pair<std::string, double>> &stageMs) {
  H3_TRY(loadVae());
  std::vector<float> z;
  vaeLatents(videoRows, z);
  model::VaePlan plan = vaePlan_;
  plan.chunks = std::max(1, std::min(clips, vaePlan_.chunks));
  plan.padFrames = 0;
  std::vector<uint8_t> frames(size_t(plan.outputFrames()) * geometry_.height * geometry_.width * 3);
  vae_->setProfile(true);
  const Status s = vae_->decode(z.data(), geometry_.latentFrames + vaePlan_.padTokens, geometry_, plan, frames.data(),
                                stats);
  for (int st = 0; st < ops::VaeDecoder::kStages; ++st)
    stageMs.push_back({ops::VaeDecoder::stageName(st), vae_->stageMs()[st]});
  vae_->setProfile(false);
  return s;
}

Status Engine::preview(const float *videoRows, uint8_t *frames, uint8_t *yuv, PreviewStats &stats) {
  const model::VideoGeometry &g = geometry_;
  if (!preview_) {
    std::unique_ptr<model::PreviewWeights> weights;
    H3_TRY(model::PreviewWeights::load(*backend_, config_.weights + "/preview", weights));
    H3_TRY(ops::PreviewDecoder::create(*backend_, std::move(weights), g.latentHeight, g.latentWidth, preview_));
  }
  std::vector<float> z(size_t(g.latentFrames) * g.latentHeight * g.latentWidth * H3_VAE_LATENT);
  model::unpatchifyVideo(videoRows, g, nullptr, nullptr, z.data());
  H3_TRY(preview_->decode(z.data(), g.latentFrames, frames, yuv, stats));
  log_.info("H3 preview %d frames wall %.2fs gpu %.2fs nonfinite %lld", stats.frames, stats.wallS, stats.gpuS,
            static_cast<long long>(stats.nonfinite));
  return {};
}

Status Engine::decodeAudio(const float *audioRows, float *wav, AudioStats &stats) {
  if (!audioRows || !wav) return error(Code::Argument, "audio decode: null rows or waveform");
  if (!audio_) {
    std::unique_ptr<model::AudioWeights> weights;
    H3_TRY(model::AudioWeights::load(*backend_, config_.weights + "/audio", weights));
    H3_TRY(ops::AudioDecoder::create(*backend_, std::move(weights), audio_));
  }
  H3_TRY(audio_->decode(audioRows, geometry_.audioRows / 2, wav, stats));
  log_.info("H3 audio %lld samples x 2 wall %.2fs gpu %.2fs nonfinite %lld device_MB %.0f",
            static_cast<long long>(stats.samples), stats.wallS, stats.gpuS, static_cast<long long>(stats.nonfinite),
            stats.deviceMb);
  return {};
}

} // namespace h3

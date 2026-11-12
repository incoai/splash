// The video engine of the golden-clip tests, which run it on a real model directory:
// the model's runtime (runtime/video/VideoRuntime.hpp), its
// stages changed by two options:
//
//   video-golden serve-native MODEL_DIRECTORY --metallib LIB [--hidden-states NPZ] [--keep DIRECTORY]
//                [--adapter PATH[:SCALE]]... [--no-adapter] [--renoise NPZ] [--forwards N]
//
//   --hidden-states NPZ  every request is conditioned on the f32 [L][5120] `hidden` of NPZ instead of its encoded
//                        prompt: a stage in place of the text encoding;
//   --keep DIRECTORY     each request writes its prompt's hidden states (hidden.npz), initial noise (noise.npz: video
//                        [rows][96] patchified as the DiT's rows, [(f, h / 2, w / 2)][c x 4 + 2 (h % 2) + w % 2] of
//                        the [24][F][H / 16][W / 16] latent noise; audio [rows][32]), final latents (final.npz) and
//                        video frames (frames.rgb, uint8 RGB) into DIRECTORY/REQUEST_ID, after the stages that make
//                        them: the noise to start another engine's clip of the same canvas from; and the first
//                        forward's velocity (step0_velocity.npz, video [rows][96], audio [rows][32]) after Denoise;
//   --adapter PATH[:SCALE] the runtime adapters (video/model/Adapter.hpp) in place of the DiT manifest's (repeatable; the
//                        DiT runs one); --no-adapter: none, whatever the manifest says;
//   --renoise NPZ        a renoise sampler's fresh noise (pika-6) from NPZ in place of the engine's own draws:
//                        video [steps - 1][rows][96] and audio [steps - 1][rows][32], in the noise's row layout;
//   --forwards N         diagnostics: the denoising stops after N forwards (step 0 alone: 1), the clip decoded from
//                        the latents it reached.
#include "Numpy.hpp"
#include "StderrLine.hpp"
#include "video/VideoRuntime.hpp"
#include "video/model/TextEncoder.hpp"

#import <Foundation/Foundation.h>

#include <sysexits.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace splash::video;
using splash::dev::NpyArray;
using Keep = std::function<void(const VideoBatch &)>;

constexpr int64_t kHiddenSize = model::TextEncoder::kHiddenSize;

std::vector<float> readHiddenStates(const std::filesystem::path &path) {
  std::map<std::string, NpyArray> npz;
  try {
    npz = splash::dev::readNpz(path.string());
  } catch (const std::runtime_error &) {
  }
  const auto found = npz.find("hidden");
  if (found == npz.end() || found->second.dtype != "<f4" || found->second.shape.size() != 2 ||
      found->second.shape[1] != kHiddenSize)
    throw std::invalid_argument(path.string() + " holds no f32 [L][5120] hidden");
  const auto *values = found->second.as<float>();
  return {values, values + found->second.count()};
}

// The text encoding's place, conditioning every request on the same hidden states.
class HiddenStatesStage final : public PipelineStage {
public:
  explicit HiddenStatesStage(std::vector<float> hidden) : hidden_(std::move(hidden)) {}
  [[nodiscard]] splash::protocol::VideoPhase phase() const override { return splash::protocol::VideoPhase::Encode; }
  void forward(VideoBatch &batch, StageEvents &) override { batch.hidden = hidden_; }

private:
  std::vector<float> hidden_;
};

// A stage, then what it left in the batch kept.
class KeptStage final : public PipelineStage {
public:
  KeptStage(std::unique_ptr<PipelineStage> stage, Keep keep) : stage_(std::move(stage)), keep_(std::move(keep)) {}
  [[nodiscard]] splash::protocol::VideoPhase phase() const override { return stage_->phase(); }
  [[nodiscard]] bool runs(const VideoBatch &batch) const override { return stage_->runs(batch); }
  [[nodiscard]] uint32_t steps(const VideoBatch &batch) const override { return stage_->steps(batch); }
  void forward(VideoBatch &batch, StageEvents &events) override {
    stage_->forward(batch, events);
    keep_(batch);
  }

private:
  std::unique_ptr<PipelineStage> stage_;
  Keep keep_;
};

void writeNpz(const std::filesystem::path &path, const std::vector<std::pair<std::string, NpyArray>> &arrays) {
  std::filesystem::create_directories(path.parent_path());
  std::vector<std::pair<std::string, const NpyArray *>> members;
  for (const auto &[name, array] : arrays) members.emplace_back(name, &array);
  splash::dev::writeNpz(path.string(), members);
}

// What the stage of `phase` leaves in a batch, kept into keep/REQUEST_ID; nothing for the other stages.
std::optional<Keep> keeping(splash::protocol::VideoPhase phase, const std::filesystem::path &keep) {
  const auto directory = [keep](const VideoBatch &batch) { return keep / std::to_string(batch.requestId); };
  switch (phase) {
  case splash::protocol::VideoPhase::Encode:
    return [directory](const VideoBatch &batch) {
      const auto rows = static_cast<int64_t>(batch.hidden.size()) / kHiddenSize;
      writeNpz(directory(batch) / "hidden.npz",
               {{"hidden", NpyArray::of("<f4", {rows, kHiddenSize}, batch.hidden.data(), batch.hidden.size() * 4)}});
    };
  case splash::protocol::VideoPhase::Prepare:
    return [directory](const VideoBatch &batch) {
      const std::vector<float> &video = batch.prepared.videoNoise, &audio = batch.prepared.audioNoise;
      writeNpz(directory(batch) / "noise.npz",
               {{"video", NpyArray::of("<f4", {batch.geometry.videoRows(), 96}, video.data(), video.size() * 4)},
                {"audio", NpyArray::of("<f4", {batch.geometry.audioRows, 32}, audio.data(), audio.size() * 4)}});
    };
  case splash::protocol::VideoPhase::Denoise:
    return [directory](const VideoBatch &batch) {
      const std::vector<float> &video = batch.videoLatents, &audio = batch.audioLatents;
      writeNpz(directory(batch) / "final.npz",
               {{"video", NpyArray::of("<f4", {batch.geometry.videoRows(), 96}, video.data(), video.size() * 4)},
                {"audio", NpyArray::of("<f4", {batch.geometry.audioRows, 32}, audio.data(), audio.size() * 4)}});
      const std::vector<float> &v0 = batch.step0Video, &a0 = batch.step0Audio;
      if (!v0.empty())
        writeNpz(directory(batch) / "step0_velocity.npz",
                 {{"video", NpyArray::of("<f4", {batch.geometry.videoRows(), 96}, v0.data(), v0.size() * 4)},
                  {"audio", NpyArray::of("<f4", {batch.geometry.audioRows, 32}, a0.data(), a0.size() * 4)}});
    };
  case splash::protocol::VideoPhase::Decode:
    return [directory](const VideoBatch &batch) {
      std::ofstream file(directory(batch) / "frames.rgb", std::ios::binary);
      file.write(reinterpret_cast<const char *>(batch.frames.data()), static_cast<std::streamsize>(batch.frames.size()));
      if (!file.flush()) throw std::runtime_error("cannot write the frames of request " + std::to_string(batch.requestId));
    };
  default:
    return std::nullopt;
  }
}

// The renoise sampler's fresh noise of an npz, in place of what the prep drew.
class RenoiseStage final : public PipelineStage {
public:
  RenoiseStage(std::unique_ptr<PipelineStage> prep, std::filesystem::path npz)
      : prep_(std::move(prep)), npz_(std::move(npz)) {}
  [[nodiscard]] splash::protocol::VideoPhase phase() const override { return prep_->phase(); }
  [[nodiscard]] bool runs(const VideoBatch &batch) const override { return prep_->runs(batch); }
  [[nodiscard]] uint32_t steps(const VideoBatch &batch) const override { return prep_->steps(batch); }
  void forward(VideoBatch &batch, StageEvents &events) override {
    prep_->forward(batch, events);
    model::Prepared &p = batch.prepared;
    if (!p.renoise) throw std::invalid_argument("--renoise: the variant's sampler is not renoise");
    const std::map<std::string, NpyArray> npz = splash::dev::readNpz(npz_.string());
    for (auto [name, target] : {std::pair<const char *, std::vector<float> *>{"video", &p.renoiseVideo},
                                std::pair<const char *, std::vector<float> *>{"audio", &p.renoiseAudio}}) {
      const auto found = npz.find(name);
      if (found == npz.end() || found->second.dtype != "<f4" || found->second.count() != target->size())
        throw std::invalid_argument(npz_.string() + ": " + name + " must be f32 of " + std::to_string(target->size()) +
                                    " values");
      const auto *values = found->second.as<float>();
      target->assign(values, values + found->second.count());
    }
    splash::logLine("H3 renoise noise from " + npz_.string());
  }

private:
  std::unique_ptr<PipelineStage> prep_;
  std::filesystem::path npz_;
};

// A stage that first limits the request's denoising forwards (--forwards).
class ForwardsStage final : public PipelineStage {
public:
  ForwardsStage(std::unique_ptr<PipelineStage> stage, int forwards) : stage_(std::move(stage)), forwards_(forwards) {}
  [[nodiscard]] splash::protocol::VideoPhase phase() const override { return stage_->phase(); }
  [[nodiscard]] bool runs(const VideoBatch &batch) const override { return stage_->runs(batch); }
  [[nodiscard]] uint32_t steps(const VideoBatch &batch) const override { return stage_->steps(batch); }
  void forward(VideoBatch &batch, StageEvents &events) override {
    batch.maxForwards = forwards_;
    stage_->forward(batch, events);
  }

private:
  std::unique_ptr<PipelineStage> stage_;
  int forwards_;
};

int run(int argc, char **argv) {
  if (argc < 3 || std::string_view(argv[1]) != "serve-native")
    throw std::invalid_argument(
        "usage: video-golden serve-native MODEL_DIRECTORY --metallib LIB [--hidden-states NPZ] [--keep DIRECTORY] "
        "[--adapter PATH[:SCALE]]... [--no-adapter] [--renoise NPZ] [--forwards N]");
  const std::filesystem::path modelRoot = argv[2];
  std::filesystem::path metallib, keep, renoise;
  int forwards = 0;
  std::optional<std::vector<float>> hidden;
  std::optional<std::vector<model::AdapterSpec>> adapters;
  for (int index = 3; index < argc; index += 2) {
    const std::string_view option(argv[index]);
    if (option == "--no-adapter") {
      adapters.emplace();
      --index;
      continue;
    }
    if (index + 1 == argc) throw std::invalid_argument(std::string(option) + " takes a value");
    if (option == "--adapter") {
      if (!adapters) adapters.emplace();
      adapters->push_back(model::parseAdapterSpec(argv[index + 1]));
    } else if (option == "--renoise")
      renoise = argv[index + 1];
    else if (option == "--forwards")
      forwards = std::stoi(argv[index + 1]);
    else if (option == "--metallib")
      metallib = argv[index + 1];
    else if (option == "--hidden-states")
      hidden = readHiddenStates(argv[index + 1]);
    else if (option == "--keep")
      keep = argv[index + 1];
    else
      throw std::invalid_argument("unknown option " + std::string(option));
  }
  if (metallib.empty()) throw std::invalid_argument("video-golden takes --metallib LIB");
  return serve(metallib, modelRoot, [&](splash::metal::MetalBackend &backend) {
    Stages stages = VideoRuntime::stages(backend, modelRoot, adapters);
    for (std::unique_ptr<PipelineStage> &stage : stages) {
      if (hidden && stage->phase() == splash::protocol::VideoPhase::Encode)
        stage = std::make_unique<HiddenStatesStage>(*hidden);
      if (!renoise.empty() && stage->phase() == splash::protocol::VideoPhase::Prepare)
        stage = std::make_unique<RenoiseStage>(std::move(stage), renoise);
      if (forwards > 0 && stage->phase() == splash::protocol::VideoPhase::Prepare)
        stage = std::make_unique<ForwardsStage>(std::move(stage), forwards);
      if (!keep.empty())
        if (std::optional<Keep> kept = keeping(stage->phase(), keep))
          stage = std::make_unique<KeptStage>(std::move(stage), std::move(*kept));
    }
    return stages;
  });
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      return run(argc, argv);
    } catch (const std::exception &error) {
      splash::writeStderrLine(std::string("error: ") + error.what());
      return EX_SOFTWARE;
    }
  }
}

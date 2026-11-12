#include "video/model/References.hpp"

#include "AwakeClock.hpp"
#include "video/model/AudioWeights.hpp"
#include "video/model/ReferenceLatents.hpp"
#include "video/model/VaeWeights.hpp"
#include "video/ops/AudioEncoder.hpp"
#include "video/ops/VaeEncoder.hpp"

#include <optional>
#include <stdexcept>

namespace splash::video::model {

namespace {
double since(AwakeClock::time_point t) { return millisecondsSince(t) / 1e3; }
} // namespace

int32_t referenceVideoFrames(int32_t frames) {
  constexpr int32_t clip = ops::VaeEncoder::kClip, tail = ops::VaeEncoder::kTailFrames;
  if (frames < clip + tail)
    throw std::invalid_argument("a reference video of " + std::to_string(frames) + " frames: at least 22 at 24 fps");
  return (frames - tail) / clip * clip + tail;
}

void encodeReferences(metal::MetalBackend &backend, const std::string &weights,
                      const std::vector<RequestReference> &references, EncodedReferences &out, uint64_t seed) {
  using Kind = ops::ReferenceShape::Kind;
  const auto t0 = AwakeClock::now();
  const size_t n = references.size();
  out.video.assign(n, {});
  out.audio.assign(n, {});
  out.inputs.assign(n, {});
  out.encodedFrames.assign(n, 0);
  for (size_t i = 0; i < n; ++i) {
    const RequestReference &r = references[i];
    if (r.kind != Kind::Video && r.kind != Kind::Image && r.kind != Kind::Audio)
      throw std::invalid_argument("reference " + std::to_string(i) + ": only video, image and audio references are encoded");
    if (r.kind == Kind::Audio) {
      if (!r.wav || r.samples <= 0)
        throw std::invalid_argument("reference " + std::to_string(i) + ": an audio reference needs its samples");
      continue;
    }
    if (!r.rgb || r.width % 16 || r.height % 16 || r.width <= 0 || r.height <= 0)
      throw std::invalid_argument("reference " + std::to_string(i) + ": pixels of a multiple of 16 per side");
    if (r.kind == Kind::Image && r.frames != 1)
      throw std::invalid_argument("reference " + std::to_string(i) + ": an image is one frame");
  }
  // the video VAE encoder: videos and images, its weights released after the last
  {
    std::optional<ops::VaeEncoder> vae;
    for (size_t i = 0; i < n; ++i) {
      const RequestReference &r = references[i];
      if (r.kind == Kind::Audio) {  // the audio VAE's alone
        out.inputs[i].shape.kind = Kind::Audio;
        out.inputs[i].shape.frames = 0;
        continue;
      }
      if (!vae) {
        const auto tl = AwakeClock::now();
        vae.emplace(backend, loadVaeEncoder(backend, weights + "/vae"));
        out.loadS += since(tl);
      }
      const auto te = AwakeClock::now();
      std::vector<float> moments;
      ops::VaeEncodeStats st;
      int32_t F = 1;
      if (r.kind == Kind::Video) {
        const int32_t frames = referenceVideoFrames(r.frames);
        out.encodedFrames[i] = frames;
        vae->encodeVideoRgb(r.rgb, frames, r.height, r.width, moments, st);
        F = (frames - ops::VaeEncoder::kTailFrames) / ops::VaeEncoder::kClip * ops::VaeEncoder::kClipLatents + 2;
      } else {
        out.encodedFrames[i] = 1;
        vae->encodeRgb(r.rgb, 1, r.height, r.width, moments, st);
      }
      out.nonfinite += st.nonfinite;
      const int32_t h = r.height / 16, w = r.width / 16;
      if (moments.size() != size_t(F) * h * w * 48)
        throw std::runtime_error("reference " + std::to_string(i) + ": " + std::to_string(moments.size()) +
                                 " moments, want " + std::to_string(size_t(F) * h * w * 48));
      referenceLatents(moments, F, h, w, seed, out.video[i]);
      ReferenceInput &in = out.inputs[i];
      in.shape.kind = r.kind;
      in.shape.frames = F;
      in.shape.latentHeight = h;
      in.shape.latentWidth = w;
      (r.kind == Kind::Video ? out.videoS : out.imageS) += since(te);
    }
  }
  // the audio VAE encoder: soundtracks
  {
    std::optional<ops::AudioEncoder> enc;
    for (size_t i = 0; i < n; ++i) {
      const RequestReference &r = references[i];
      if (!r.wav || r.samples <= 0) continue;
      if (!enc) {
        const auto tl = AwakeClock::now();
        enc.emplace(backend, loadAudioEncoder(backend, weights + "/audio"));
        out.loadS += since(tl);
      }
      const auto te = AwakeClock::now();
      ops::AudioEncodeStats st;
      enc->encode(r.wav, r.samples, out.audio[i], st);
      out.nonfinite += st.nonfinite;
      out.inputs[i].shape.audioLatents = int32_t(out.audio[i].size() / 64);
      out.audioS += since(te);
    }
  }
  for (size_t i = 0; i < n; ++i) {
    out.inputs[i].video = out.video[i].empty() ? nullptr : out.video[i].data();
    out.inputs[i].audio = out.audio[i].empty() ? nullptr : out.audio[i].data();
  }
  if (out.nonfinite)
    throw metal::MetalBackendError("references: " + std::to_string(out.nonfinite) + " non-finite encoder outputs");
  out.totalS = since(t0);
}

std::vector<TextEncoder::Image> referenceVisionImages(const std::vector<RequestReference> &references) {
  std::vector<TextEncoder::Image> images;
  for (const RequestReference &r : references) {
    const size_t frameBytes = size_t(r.width) * r.height * 3;
    if (r.kind == ops::ReferenceShape::Kind::Image) {
      images.push_back({r.rgb, uint32_t(r.width), uint32_t(r.height), nullptr});
    } else if (r.kind == ops::ReferenceShape::Kind::Video) {
      for (const auto &b : TextEncoder::videoBlockFrames(uint32_t(r.frames)))
        images.push_back({r.rgb + b.first * frameBytes, uint32_t(r.width), uint32_t(r.height),
                          r.rgb + b.second * frameBytes});
    }
  }
  return images;
}

} // namespace splash::video::model

#include "video/h3/model/AudioWeights.hpp"

#include "model/WeightStore.hpp"
#include "video/h3/model/WeightFiles.hpp"

namespace h3::model {

std::unique_ptr<AudioWeights> AudioWeights::load(metal::MetalBackend &backend, const std::string &dir) {
  using splash::model::WeightStoreError;
  auto w = std::unique_ptr<AudioWeights>(new AudioWeights());
  NSDictionary *manifest = readManifest(dir + "/manifest.json");
  const WeightSource source(dir + "/audio_dec_f16.bin");
  w->blob_ = allocate(backend, source.bytes(), "audio weights");
  read(source, 0, source.bytes(), w->blob_.contents());
  // entry -> view, with its shape (checked against `shape` where given; -1 = any)
  auto at = [&](const std::string &key, BufferView &view, std::vector<int64_t> shape) {
    NSDictionary *entry = manifest[@(key.c_str())];
    const std::optional<int64_t> offset = [entry isKindOfClass:[NSDictionary class]] ? integer(entry[@"off"]) : std::nullopt;
    NSArray *dims = [entry isKindOfClass:[NSDictionary class]] ? entry[@"shape"] : nil;
    if (!offset || ![dims isKindOfClass:[NSArray class]] || *offset < 0 || uint64_t(*offset) >= source.bytes())
      throw WeightStoreError(dir + "/manifest.json: bad entry " + key);
    if (!shape.empty()) {
      if (dims.count != shape.size()) throw WeightStoreError(dir + "/manifest.json: " + key + " rank");
      for (size_t i = 0; i < shape.size(); ++i)
        if (shape[i] >= 0 && integer(dims[i]) != shape[i])
          throw WeightStoreError(dir + "/manifest.json: " + key + " has an unexpected shape");
    }
    view = {w->blob_, static_cast<uint64_t>(*offset)};
  };
  auto conv = [&](const std::string &key, int32_t taps, int32_t cout, int32_t cin, bool bias, AudioConv &c) {
    at(key + ".w", c.w, {taps, cout, cin});
    at(key + ".wl", c.wl, {taps, cout, cin});
    if (bias) at(key + ".b", c.b, {cout});
    c.taps = taps;
    c.cout = cout;
    c.cin = cin;
  };
  conv("dec_in", 1, kLatentDim, 32, true, w->decIn);
  conv("conv_pre", 7, kDecoderDim, kLatentDim, true, w->convPre);
  static constexpr int kUpKernels[kStages] = {9, 9, 4, 4, 4, 4, 4};
  for (int s = 0; s < kStages; ++s) {
    const int32_t cin = kDecoderDim >> s, c = kDecoderDim >> (s + 1);
    AudioStageWeights &g = w->stages[s];
    conv("ups" + std::to_string(s), kUpKernels[s], c, cin, true, g.up);
    for (int j = 0; j < 3; ++j) {
      const std::string rb = "rb" + std::to_string(s * 3 + j);
      for (int d = 0; d < 3; ++d) {
        conv(rb + ".c1." + std::to_string(d), kResKernels[j], c, c, true, g.convs1[j][d]);
        conv(rb + ".c2." + std::to_string(d), kResKernels[j], c, c, true, g.convs2[j][d]);
      }
      for (int a = 0; a < 6; ++a) at(rb + ".act" + std::to_string(a), g.acts[j][a], {2, c});
    }
  }
  at("post.act", w->postAct, {2, kDecoderDim >> kStages});
  conv("conv_post", 7, 1, kDecoderDim >> kStages, false, w->convPost);
  at("filter", w->filter, {12});
  at("latents_mean", w->mean, {32});
  at("latents_std", w->std, {32});
  return w;
}

} // namespace h3::model

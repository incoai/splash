#include "video/model/AudioWeights.hpp"

#include "model/WeightStore.hpp"
#include "video/model/WeightFiles.hpp"

namespace splash::video::model {

ops::AudioDecoderWeights loadAudioDecoder(metal::MetalBackend &backend, const std::string &dir) {
  using splash::model::WeightStoreError;
  using ops::AudioDecoder;
  ops::AudioDecoderWeights w;
  NSDictionary *manifest = readManifest(dir + "/manifest.json");
  const WeightSource source(dir + "/audio_dec_f16.bin");
  const metal::MetalBuffer blob = allocate(backend, source.bytes(), "audio weights");
  read(source, 0, source.bytes(), blob.contents());
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
    view = {blob, static_cast<uint64_t>(*offset)};
  };
  auto conv = [&](const std::string &key, int32_t taps, int32_t cout, int32_t cin, bool bias, ops::AudioConv &c) {
    at(key + ".w", c.w, {taps, cout, cin});
    at(key + ".wl", c.wl, {taps, cout, cin});
    if (bias) at(key + ".b", c.b, {cout});
    c.taps = taps;
    c.cout = cout;
    c.cin = cin;
  };
  conv("dec_in", 1, AudioDecoder::kLatentDim, 32, true, w.decIn);
  conv("conv_pre", 7, AudioDecoder::kDecoderDim, AudioDecoder::kLatentDim, true, w.convPre);
  static constexpr int kUpKernels[AudioDecoder::kStages] = {9, 9, 4, 4, 4, 4, 4};
  for (int s = 0; s < AudioDecoder::kStages; ++s) {
    const int32_t cin = AudioDecoder::kDecoderDim >> s, c = AudioDecoder::kDecoderDim >> (s + 1);
    ops::AudioStageWeights &g = w.stages[s];
    conv("ups" + std::to_string(s), kUpKernels[s], c, cin, true, g.up);
    for (int j = 0; j < 3; ++j) {
      const std::string rb = "rb" + std::to_string(s * 3 + j);
      for (int d = 0; d < 3; ++d) {
        conv(rb + ".c1." + std::to_string(d), AudioDecoder::kResKernels[j], c, c, true, g.convs1[j][d]);
        conv(rb + ".c2." + std::to_string(d), AudioDecoder::kResKernels[j], c, c, true, g.convs2[j][d]);
      }
      for (int a = 0; a < 6; ++a) at(rb + ".act" + std::to_string(a), g.acts[j][a], {2, c});
    }
  }
  at("post.act", w.postAct, {2, AudioDecoder::kDecoderDim >> AudioDecoder::kStages});
  conv("conv_post", 7, 1, AudioDecoder::kDecoderDim >> AudioDecoder::kStages, false, w.convPost);
  at("filter", w.filter, {12});
  at("latents_mean", w.mean, {32});
  at("latents_std", w.std, {32});
  return w;
}

} // namespace splash::video::model

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

ops::AudioEncoderWeights loadAudioEncoder(metal::MetalBackend &backend, const std::string &audioDir) {
  using splash::model::WeightStoreError;
  using ops::AudioEncoder;
  ops::AudioEncoderWeights w;
  const std::string manifestPath = audioDir + "/manifest_enc.json";
  NSDictionary *manifest = readManifest(manifestPath);
  const WeightSource source(audioDir + "/audio_enc_f16.bin");
  const metal::MetalBuffer blob = allocate(backend, source.bytes(), "audio encoder weights");
  read(source, 0, source.bytes(), blob.contents());
  auto at = [&](const std::string &key, std::vector<int64_t> shape) -> BufferView {
    NSDictionary *entry = manifest[@(key.c_str())];
    const std::optional<int64_t> offset = [entry isKindOfClass:[NSDictionary class]] ? integer(entry[@"off"]) : std::nullopt;
    NSArray *dims = [entry isKindOfClass:[NSDictionary class]] ? entry[@"shape"] : nil;
    if (!offset || ![dims isKindOfClass:[NSArray class]] || *offset < 0 || uint64_t(*offset) >= source.bytes())
      throw WeightStoreError(manifestPath + ": bad entry " + key);
    if (dims.count != shape.size()) throw WeightStoreError(manifestPath + ": " + key + " rank");
    uint64_t elems = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (integer(dims[i]) != shape[i]) throw WeightStoreError(manifestPath + ": " + key + " has an unexpected shape");
      elems *= uint64_t(shape[i]);
    }
    const uint64_t item = [entry[@"dtype"] isEqual:@"float32"] ? 4 : 2;
    if (uint64_t(*offset) + elems * item > source.bytes()) throw WeightStoreError(manifestPath + ": " + key + " overruns");
    return {blob, static_cast<uint64_t>(*offset)};
  };
  auto host = [&](const std::string &key, std::vector<int64_t> shape) {
    return static_cast<const float *>(at(key, std::move(shape)).contents());
  };
  auto conv = [&](const std::string &key, int32_t taps, int32_t cout, int32_t cin, ops::AudioConv &c) {
    c.w = at(key + ".w", {taps, cout, cin});
    c.wl = at(key + ".wl", {taps, cout, cin});
    c.b = at(key + ".b", {cout});
    c.taps = taps;
    c.cout = cout;
    c.cin = cin;
  };
  auto linear = [&](const std::string &key, int32_t in, int32_t out) {
    return ops::AudioEncoderWeights::Linear{host(key + ".wt", {in, out}), host(key + ".b", {out})};
  };
  conv("enc_in", 7, AudioEncoder::kEncoderDim, 1, w.convIn);
  int32_t C = AudioEncoder::kEncoderDim;
  for (int i = 0; i < AudioEncoder::kBlocks; ++i) {
    const std::string blk = "blk" + std::to_string(i);
    ops::AudioEncoderWeights::Block &b = w.blocks[i];
    for (int u = 0; u < 3; ++u) {
      const std::string un = blk + ".u" + std::to_string(u);
      b.units[u].act1 = at(un + ".act1", {2, C});
      conv(un + ".c7", 7, C, C, b.units[u].conv7);
      b.units[u].act2 = at(un + ".act2", {2, C});
      conv(un + ".c1", 1, C, C, b.units[u].conv1);
    }
    b.act = at(blk + ".act", {2, C});
    conv(blk + ".down", 2, 2 * C, AudioEncoder::kRates[i] * C, b.down);
    C *= 2;
  }
  constexpr int32_t D = AudioEncoder::kLatentDim, L = AudioEncoder::kLatent;
  if (C != D) throw WeightStoreError(manifestPath + ": encoder width");
  w.finalAct = at("final.act", {2, D});
  conv("enc_out", 3, D, D, w.convOut);
  w.norm1W = host("pre.norm1.w", {D});
  w.norm1B = host("pre.norm1.b", {D});
  w.norm3W = host("pre.norm3.w", {D});
  w.norm3B = host("pre.norm3.b", {D});
  w.norm2W = host("pre.norm2.w", {L});
  w.norm2B = host("pre.norm2.b", {L});
  w.mlpNormW = host("pre.mlp.norm.w", {L});
  w.mlpNormB = host("pre.mlp.norm.b", {L});
  w.qkvT = host("pre.attn.qkv.wt", {D, 3 * D});
  w.qBias = host("pre.attn.q_bias", {D});
  w.vBias = host("pre.attn.v_bias", {D});
  w.attnProj = linear("pre.attn.proj", L, L);
  w.proj = linear("pre.proj", D, L);
  w.w0 = linear("pre.mlp.w0", L, 2 * L);
  w.w1 = linear("pre.mlp.w1", L, 2 * L);
  w.w2 = linear("pre.mlp.w2", 2 * L, L);
  w.meanProj = linear("mean_proj", L, L);
  w.mean = host("latents_mean", {L});
  w.std = host("latents_std", {L});
  return w;
}

} // namespace splash::video::model

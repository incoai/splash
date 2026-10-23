#include "video/model/VaeWeights.hpp"

#include "metal/abi/QuantFormat.h"
#include "model/WeightStore.hpp"
#include "video/model/WeightFiles.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <utility>
#include <vector>

namespace splash::video::model {

ops::VaeDecoderWeights loadVaeDecoder(metal::MetalBackend &backend, const std::string &dir, const std::string &tag) {
  using splash::model::WeightStoreError;
  ops::VaeDecoderWeights w;
  std::vector<metal::MetalBuffer> regions;
  std::vector<std::pair<int64_t, int64_t>> ranges;
  const std::string manifestPath = dir + "/manifest_" + tag + ".json";
  NSDictionary *manifest = readManifest(manifestPath);
  NSDictionary *globals = manifest[@"globals"];
  NSArray *layers = manifest[@"layers"];
  if (![globals isKindOfClass:[NSDictionary class]] || ![layers isKindOfClass:[NSArray class]] ||
      layers.count != H3_VAE_LAYERS)
    throw WeightStoreError(manifestPath + ": expected globals and " + std::to_string(H3_VAE_LAYERS) + " layers");
  for (NSDictionary *layer in layers)
    if (![layer isKindOfClass:[NSDictionary class]]) throw WeightStoreError(manifestPath + ": a layer is no object");
  // int8 layers are of the weights' format, QUANT_VIDEO_W8: rotated by the Hadamard signs of its seed.
  NSArray *int8Layers = manifest[@"i8"];
  if ([int8Layers isKindOfClass:[NSArray class]] && int8Layers.count &&
      integer(manifest[@"had_seed"]) != int64_t(QUANT_VIDEO_W8_SEED))
    throw WeightStoreError(manifestPath + ": its int8 layers are not of format " QUANT_VIDEO_W8 " (Hadamard seed " +
                           std::to_string(QUANT_VIDEO_W8_SEED) + ")");
  const std::string blob = dir + "/vae_dec_" + tag + ".bin";
  const WeightSource source(blob);
  // regions: globals, then one per layer (from its smallest offset to the next layer's)
  std::vector<int64_t> starts{0};
  for (NSDictionary *layer in layers) {
    int64_t first = std::numeric_limits<int64_t>::max();
    for (id key in layer) first = std::min(first, integer(layer[key]).value_or(first));
    starts.push_back(first);
  }
  starts.push_back(static_cast<int64_t>(source.bytes()));
  for (size_t i = 0; i + 1 < starts.size(); ++i) {
    const auto bytes = static_cast<uint64_t>(starts[i + 1] - starts[i]);
    regions.push_back(allocate(backend, bytes, "VAE weights"));
    read(source, static_cast<uint64_t>(starts[i]), bytes, regions.back().contents());
    ranges.push_back({starts[i], starts[i + 1]});
  }
  auto ref = [&](NSDictionary *record, NSString *key, bool required, BufferView &view) {
    const std::optional<int64_t> offset = integer(record[key]);
    if (!offset) {
      if (required) throw WeightStoreError(blob + ": missing " + key.UTF8String);
      view = {};
      return;
    }
    for (size_t i = 0; i < ranges.size(); ++i)
      if (*offset >= ranges[i].first && *offset < ranges[i].second) {
        view = {regions[i], static_cast<uint64_t>(*offset - ranges[i].first)};
        return;
      }
    throw WeightStoreError(blob + ": offset of " + key.UTF8String + " outside the file");
  };
  ops::VaeDecoderWeights &g = w;
  ref(globals, @"we", true, g.embed);
  ref(globals, @"be", true, g.embedBias);
  ref(globals, @"reg", true, g.registers);
  ref(globals, @"lnw", true, g.lnWeight);
  ref(globals, @"lnb", true, g.lnBias);
  ref(globals, @"wp", true, g.proj);
  ref(globals, @"bp", true, g.projBias);
  ref(globals, @"cos", true, g.cos);
  ref(globals, @"sin", true, g.sin);
  ref(globals, @"sg2048", true, g.signs2048);
  ref(globals, @"sg8192", true, g.signs8192);
  for (int l = 0; l < H3_VAE_LAYERS; ++l) {
    NSDictionary *r = layers[static_cast<NSUInteger>(l)];
    ops::VaeLayerWeights &L = w.layers[l];
    ref(r, @"wqkv", true, L.qkv);
    ref(r, @"qqkv", false, L.qkvScale);
    ref(r, @"bqkv", true, L.qkvBias);
    ref(r, @"wo", true, L.out);
    ref(r, @"qo", false, L.outScale);
    ref(r, @"bo", true, L.outBias);
    ref(r, @"w1", true, L.fc1);
    ref(r, @"q1", false, L.fc1Scale);
    ref(r, @"b1", true, L.fc1Bias);
    ref(r, @"w2", true, L.fc2);
    ref(r, @"q2", false, L.fc2Scale);
    ref(r, @"b2", true, L.fc2Bias);
    ref(r, @"s1", true, L.scale1);
    ref(r, @"s2", true, L.scale2);
    ref(r, @"n1", true, L.norm1);
    ref(r, @"n2", true, L.norm2);
  }
  return w;
}

ops::VaeEncoderWeights loadVaeEncoder(metal::MetalBackend &backend, const std::string &dir) {
  ops::VaeEncoderWeights w;
  using splash::model::WeightStoreError;
  const std::string manifestPath = dir + "/manifest_enc.json", blob = dir + "/vae_enc_f16.bin";
  if (!std::filesystem::is_regular_file(manifestPath) || !std::filesystem::is_regular_file(blob))
    throw WeightStoreError(dir + " holds no video VAE encoder (convert_vae.py --encoder)");
  NSDictionary *manifest = readManifest(manifestPath);
  NSDictionary *convs = manifest[@"convs"], *norms = manifest[@"norms"], *quant = manifest[@"quant_conv"];
  if (![convs isKindOfClass:[NSDictionary class]] || ![norms isKindOfClass:[NSDictionary class]] ||
      ![quant isKindOfClass:[NSDictionary class]])
    throw WeightStoreError(manifestPath + ": expected convs, norms and quant_conv");
  const WeightSource source(blob);
  w.blob = allocate(backend, source.bytes(), "VAE encoder weights");
  read(source, 0, source.bytes(), w.blob.contents());
  auto field = [&](NSDictionary *record, NSString *key) -> int64_t {
    const std::optional<int64_t> value =
        [record isKindOfClass:[NSDictionary class]] ? integer(record[key]) : std::nullopt;
    if (!value) throw WeightStoreError(manifestPath + ": missing " + key.UTF8String);
    return *value;
  };
  auto view = [&](NSDictionary *record, NSString *key) -> BufferView {
    const int64_t offset = field(record, key);
    if (offset < 0 || uint64_t(offset) >= source.bytes())
      throw WeightStoreError(blob + ": bad offset " + key.UTF8String);
    return {w.blob, static_cast<uint64_t>(offset)};
  };
  for (NSString *name in convs) {
    NSDictionary *record = convs[name];
    ops::VaeEncoderConv c;
    c.w = view(record, @"w");
    c.b = view(record, @"b");
    c.cin = static_cast<int32_t>(field(record, @"cin"));
    c.cinPad = static_cast<int32_t>(field(record, @"cin_pad"));
    c.cout = static_cast<int32_t>(field(record, @"cout"));
    c.coutPad = static_cast<int32_t>(field(record, @"cout_pad"));
    c.kd = static_cast<int32_t>(field(record, @"kd"));
    c.k = static_cast<int32_t>(field(record, @"k"));
    w.convs[name.UTF8String] = c;
  }
  for (NSString *name in norms) {
    NSDictionary *record = norms[name];
    w.norms[name.UTF8String] = ops::VaeEncoderNorm{view(record, @"w"), view(record, @"b"), static_cast<int32_t>(field(record, @"c"))};
  }
  w.quantW = view(quant, @"w");
  w.quantB = view(quant, @"b");
  for (int i = 0; i < ops::VaeEncoder::kLevels; ++i)
    for (int j = 0; j < 2; ++j) {
      const std::string p = "down." + std::to_string(i) + ".block." + std::to_string(j) + ".";
      for (const char *k : {"conv1", "conv2"})
        if (!w.convs.count(p + k)) throw WeightStoreError(blob + ": missing " + p + k);
    }
  for (const char *k : {"conv_in", "conv_out"})
    if (!w.convs.count(k)) throw WeightStoreError(blob + ": missing " + k);
  return w;
}

} // namespace splash::video::model

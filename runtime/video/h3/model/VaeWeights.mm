#include "video/h3/model/VaeWeights.hpp"

#include "model/WeightStore.hpp"
#include "video/h3/model/WeightFiles.hpp"

#include <algorithm>
#include <limits>

namespace h3::model {

std::unique_ptr<VaeWeights> VaeWeights::load(metal::MetalBackend &backend, const std::string &dir,
                                             const std::string &tag) {
  using splash::model::WeightStoreError;
  auto w = std::unique_ptr<VaeWeights>(new VaeWeights());
  const std::string manifestPath = dir + "/manifest_" + tag + ".json";
  NSDictionary *manifest = readManifest(manifestPath);
  NSDictionary *globals = manifest[@"globals"];
  NSArray *layers = manifest[@"layers"];
  if (![globals isKindOfClass:[NSDictionary class]] || ![layers isKindOfClass:[NSArray class]] ||
      layers.count != H3_VAE_LAYERS)
    throw WeightStoreError(manifestPath + ": expected globals and " + std::to_string(H3_VAE_LAYERS) + " layers");
  for (NSDictionary *layer in layers)
    if (![layer isKindOfClass:[NSDictionary class]]) throw WeightStoreError(manifestPath + ": a layer is no object");
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
    w->regions_.push_back(allocate(backend, bytes, "VAE weights"));
    read(source, static_cast<uint64_t>(starts[i]), bytes, w->regions_.back().contents());
    w->ranges_.push_back({starts[i], starts[i + 1]});
  }
  auto ref = [&](NSDictionary *record, NSString *key, bool required, BufferView &view) {
    const std::optional<int64_t> offset = integer(record[key]);
    if (!offset) {
      if (required) throw WeightStoreError(blob + ": missing " + key.UTF8String);
      view = {};
      return;
    }
    for (size_t i = 0; i < w->ranges_.size(); ++i)
      if (*offset >= w->ranges_[i].first && *offset < w->ranges_[i].second) {
        view = {w->regions_[i], static_cast<uint64_t>(*offset - w->ranges_[i].first)};
        return;
      }
    throw WeightStoreError(blob + ": offset of " + key.UTF8String + " outside the file");
  };
  VaeGlobalWeights &g = w->globals_;
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
    VaeLayerWeights &L = w->layers_[l];
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
    w->anyInt8_ |= L.int8Qkv() || L.int8Out() || L.int8Fc1() || L.int8Fc2();
  }
  return w;
}

} // namespace h3::model

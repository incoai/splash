#include "video/h3/model/VaeWeights.hpp"

#include "video/h3/model/Json.hpp"
#include "video/h3/model/Safetensors.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <limits>

namespace h3::model {

Status VaeWeights::load(metal::MetalBackend &backend, const std::string &dir, const std::string &tag,
                        std::unique_ptr<VaeWeights> &out) {
  auto w = std::unique_ptr<VaeWeights>(new VaeWeights());
  Json manifest;
  H3_TRY(Json::parseFile(dir + "/manifest_" + tag + ".json", manifest));
  const Json *globals = manifest.get("globals"), *layers = manifest.get("layers");
  if (!globals || !layers || layers->size() != H3_VAE_LAYERS)
    return error(Code::Io, "%s/manifest_%s.json: expected globals and %d layers", dir.c_str(), tag.c_str(),
                 H3_VAE_LAYERS);
  const std::string blob = dir + "/vae_dec_" + tag + ".bin";
  struct stat st {};
  if (stat(blob.c_str(), &st) != 0) return error(Code::Io, "cannot open %s", blob.c_str());
  // regions: globals, then one per layer (from its smallest offset to the next layer's)
  std::vector<int64_t> starts{0};
  for (const Json &layer : layers->array()) {
    int64_t first = std::numeric_limits<int64_t>::max();
    for (const auto &[key, value] : layer.object()) first = std::min(first, value.integer());
    starts.push_back(first);
  }
  starts.push_back(st.st_size);
  const int fd = ::open(blob.c_str(), O_RDONLY);
  if (fd < 0) return error(Code::Io, "cannot open %s", blob.c_str());
  Status status;
  for (size_t i = 0; i + 1 < starts.size() && status.ok(); ++i) {
    id<MTLBuffer> buffer = nil;
    status = backend.allocate(static_cast<size_t>(starts[i + 1] - starts[i]), buffer, "VAE weights");
    if (status.ok()) status = readFully(fd, starts[i], static_cast<size_t>(starts[i + 1] - starts[i]), buffer.contents, blob);
    w->regions_.push_back(buffer);
    w->ranges_.push_back({starts[i], starts[i + 1]});
  }
  ::close(fd);
  H3_TRY(status);
  auto ref = [&](const Json &record, const char *key, bool required, metal::BufferView &view) -> Status {
    const Json *offset = record.get(key);
    if (!offset) {
      if (required) return error(Code::Io, "%s: missing %s", blob.c_str(), key);
      view = {};
      return {};
    }
    const int64_t o = offset->integer();
    for (size_t i = 0; i < w->ranges_.size(); ++i)
      if (o >= w->ranges_[i].first && o < w->ranges_[i].second) {
        view = {w->regions_[i], static_cast<size_t>(o - w->ranges_[i].first)};
        return {};
      }
    return error(Code::Io, "%s: offset of %s outside the file", blob.c_str(), key);
  };
  VaeGlobalWeights &g = w->globals_;
  H3_TRY(ref(*globals, "we", true, g.embed));
  H3_TRY(ref(*globals, "be", true, g.embedBias));
  H3_TRY(ref(*globals, "reg", true, g.registers));
  H3_TRY(ref(*globals, "lnw", true, g.lnWeight));
  H3_TRY(ref(*globals, "lnb", true, g.lnBias));
  H3_TRY(ref(*globals, "wp", true, g.proj));
  H3_TRY(ref(*globals, "bp", true, g.projBias));
  H3_TRY(ref(*globals, "cos", true, g.cos));
  H3_TRY(ref(*globals, "sin", true, g.sin));
  H3_TRY(ref(*globals, "sg2048", true, g.signs2048));
  H3_TRY(ref(*globals, "sg8192", true, g.signs8192));
  for (int l = 0; l < H3_VAE_LAYERS; ++l) {
    const Json &r = layers->array()[l];
    VaeLayerWeights &L = w->layers_[l];
    H3_TRY(ref(r, "wqkv", true, L.qkv));
    H3_TRY(ref(r, "qqkv", false, L.qkvScale));
    H3_TRY(ref(r, "bqkv", true, L.qkvBias));
    H3_TRY(ref(r, "wo", true, L.out));
    H3_TRY(ref(r, "qo", false, L.outScale));
    H3_TRY(ref(r, "bo", true, L.outBias));
    H3_TRY(ref(r, "w1", true, L.fc1));
    H3_TRY(ref(r, "q1", false, L.fc1Scale));
    H3_TRY(ref(r, "b1", true, L.fc1Bias));
    H3_TRY(ref(r, "w2", true, L.fc2));
    H3_TRY(ref(r, "q2", false, L.fc2Scale));
    H3_TRY(ref(r, "b2", true, L.fc2Bias));
    H3_TRY(ref(r, "s1", true, L.scale1));
    H3_TRY(ref(r, "s2", true, L.scale2));
    H3_TRY(ref(r, "n1", true, L.norm1));
    H3_TRY(ref(r, "n2", true, L.norm2));
    w->anyInt8_ |= L.int8Qkv() || L.int8Out() || L.int8Fc1() || L.int8Fc2();
  }
  out = std::move(w);
  return {};
}

} // namespace h3::model

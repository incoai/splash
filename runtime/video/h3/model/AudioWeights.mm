#include "video/h3/model/AudioWeights.hpp"

#include "video/h3/model/Json.hpp"
#include "video/h3/model/Safetensors.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace h3::model {

Status AudioWeights::load(metal::MetalBackend &backend, const std::string &dir, std::unique_ptr<AudioWeights> &out) {
  auto w = std::unique_ptr<AudioWeights>(new AudioWeights());
  Json manifest;
  H3_TRY(Json::parseFile(dir + "/manifest.json", manifest));
  const std::string blob = dir + "/audio_dec_f16.bin";
  struct stat st {};
  if (stat(blob.c_str(), &st) != 0) return error(Code::Io, "cannot open %s", blob.c_str());
  H3_TRY(backend.allocate(static_cast<size_t>(st.st_size), w->blob_, "audio weights"));
  const int fd = ::open(blob.c_str(), O_RDONLY);
  if (fd < 0) return error(Code::Io, "cannot open %s", blob.c_str());
  Status read = readFully(fd, 0, static_cast<size_t>(st.st_size), w->blob_.contents, blob);
  ::close(fd);
  H3_TRY(read);
  // entry -> view, with its shape (checked against `shape` where given; -1 = any)
  auto at = [&](const std::string &key, metal::BufferView &view, std::vector<int64_t> shape) -> Status {
    const Json *entry = manifest.get(key);
    const Json *offset = entry ? entry->get("off") : nullptr;
    const Json *dims = entry ? entry->get("shape") : nullptr;
    if (!offset || !dims || offset->integer() < 0 || offset->integer() >= st.st_size)
      return error(Code::Io, "%s/manifest.json: bad entry %s", dir.c_str(), key.c_str());
    if (!shape.empty()) {
      if (dims->size() != shape.size()) return error(Code::Io, "%s/manifest.json: %s rank", dir.c_str(), key.c_str());
      for (size_t i = 0; i < shape.size(); ++i)
        if (shape[i] >= 0 && dims->array()[i].integer() != shape[i])
          return error(Code::Io, "%s/manifest.json: %s has an unexpected shape", dir.c_str(), key.c_str());
    }
    view = {w->blob_, static_cast<size_t>(offset->integer())};
    return {};
  };
  auto conv = [&](const std::string &key, int32_t taps, int32_t cout, int32_t cin, bool bias, AudioConv &c) -> Status {
    H3_TRY(at(key + ".w", c.w, {taps, cout, cin}));
    H3_TRY(at(key + ".wl", c.wl, {taps, cout, cin}));
    if (bias) H3_TRY(at(key + ".b", c.b, {cout}));
    c.taps = taps;
    c.cout = cout;
    c.cin = cin;
    return {};
  };
  H3_TRY(conv("dec_in", 1, kLatentDim, 32, true, w->decIn));
  H3_TRY(conv("conv_pre", 7, kDecoderDim, kLatentDim, true, w->convPre));
  static constexpr int kUpKernels[kStages] = {9, 9, 4, 4, 4, 4, 4};
  for (int s = 0; s < kStages; ++s) {
    const int32_t cin = kDecoderDim >> s, c = kDecoderDim >> (s + 1);
    AudioStageWeights &g = w->stages[s];
    H3_TRY(conv("ups" + std::to_string(s), kUpKernels[s], c, cin, true, g.up));
    for (int j = 0; j < 3; ++j) {
      const std::string rb = "rb" + std::to_string(s * 3 + j);
      for (int d = 0; d < 3; ++d) {
        H3_TRY(conv(rb + ".c1." + std::to_string(d), kResKernels[j], c, c, true, g.convs1[j][d]));
        H3_TRY(conv(rb + ".c2." + std::to_string(d), kResKernels[j], c, c, true, g.convs2[j][d]));
      }
      for (int a = 0; a < 6; ++a) H3_TRY(at(rb + ".act" + std::to_string(a), g.acts[j][a], {2, c}));
    }
  }
  H3_TRY(at("post.act", w->postAct, {2, kDecoderDim >> kStages}));
  H3_TRY(conv("conv_post", 7, 1, kDecoderDim >> kStages, false, w->convPost));
  H3_TRY(at("filter", w->filter, {12}));
  H3_TRY(at("latents_mean", w->mean, {32}));
  H3_TRY(at("latents_std", w->std, {32}));
  out = std::move(w);
  return {};
}

} // namespace h3::model

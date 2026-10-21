#include "video/h3/model/PreviewWeights.hpp"

#include "model/WeightStore.hpp"
#include "video/h3/model/WeightFiles.hpp"

namespace h3::model {

std::unique_ptr<PreviewWeights> PreviewWeights::load(metal::MetalBackend &backend, const std::string &dir) {
  auto w = std::unique_ptr<PreviewWeights>(new PreviewWeights());
  NSDictionary *manifest = readManifest(dir + "/manifest.json");
  const WeightSource source(dir + "/taeh3_dec_f16.bin");
  w->blob_ = allocate(backend, source.bytes(), "preview weights");
  read(source, 0, source.bytes(), w->blob_.contents());
  auto at = [&](const std::string &key, BufferView &view) {
    NSDictionary *entry = manifest[@(key.c_str())];
    const std::optional<int64_t> offset = [entry isKindOfClass:[NSDictionary class]] ? integer(entry[@"off"]) : std::nullopt;
    if (!offset || *offset < 0 || uint64_t(*offset) >= source.bytes())
      throw splash::model::WeightStoreError(dir + "/manifest.json: bad entry " + key);
    view = {w->blob_, static_cast<uint64_t>(*offset)};
  };
  at("d1.w", w->inputConv);
  at("d1.b", w->inputBias);
  const int memBlocks[9] = {3, 4, 5, 9, 10, 11, 15, 16, 17};
  for (int i = 0; i < 9; ++i) {
    const std::string m = "m" + std::to_string(memBlocks[i]);
    PreviewMemBlock &b = w->memBlocks[i];
    at(m + ".0.w", b.conv0);
    at(m + ".0.b", b.bias0);
    at(m + ".2.w", b.conv2);
    at(m + ".2.b", b.bias2);
    at(m + ".4.w", b.conv4);
    at(m + ".4.b", b.bias4);
  }
  const int tgrow[3] = {7, 13, 19}, up[3] = {8, 14, 20};
  for (int i = 0; i < 3; ++i) {
    at("g" + std::to_string(tgrow[i]) + ".w", w->tgrow[i]);
    at("u" + std::to_string(up[i]) + ".w", w->upsample[i]);
  }
  at("d22.w", w->outputConv);
  at("d22.b", w->outputBias);
  return w;
}

} // namespace h3::model

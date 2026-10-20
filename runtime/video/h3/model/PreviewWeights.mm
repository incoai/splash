#include "video/h3/model/PreviewWeights.hpp"

#include "video/h3/model/Json.hpp"
#include "video/h3/model/Safetensors.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace h3::model {

Status PreviewWeights::load(metal::MetalBackend &backend, const std::string &dir, std::unique_ptr<PreviewWeights> &out) {
  auto w = std::unique_ptr<PreviewWeights>(new PreviewWeights());
  Json manifest;
  H3_TRY(Json::parseFile(dir + "/manifest.json", manifest));
  const std::string blob = dir + "/taeh3_dec_f16.bin";
  struct stat st {};
  if (stat(blob.c_str(), &st) != 0) return error(Code::Io, "cannot open %s", blob.c_str());
  H3_TRY(backend.allocate(static_cast<size_t>(st.st_size), w->blob_, "preview weights"));
  const int fd = ::open(blob.c_str(), O_RDONLY);
  if (fd < 0) return error(Code::Io, "cannot open %s", blob.c_str());
  Status read = readFully(fd, 0, static_cast<size_t>(st.st_size), w->blob_.contents, blob);
  ::close(fd);
  H3_TRY(read);
  auto at = [&](const std::string &key, metal::BufferView &view) -> Status {
    const Json *entry = manifest.get(key);
    const Json *offset = entry ? entry->get("off") : nullptr;
    if (!offset || offset->integer() < 0 || offset->integer() >= st.st_size)
      return error(Code::Io, "%s/manifest.json: bad entry %s", dir.c_str(), key.c_str());
    view = {w->blob_, static_cast<size_t>(offset->integer())};
    return {};
  };
  H3_TRY(at("d1.w", w->inputConv));
  H3_TRY(at("d1.b", w->inputBias));
  const int memBlocks[9] = {3, 4, 5, 9, 10, 11, 15, 16, 17};
  for (int i = 0; i < 9; ++i) {
    const std::string m = "m" + std::to_string(memBlocks[i]);
    PreviewMemBlock &b = w->memBlocks[i];
    H3_TRY(at(m + ".0.w", b.conv0));
    H3_TRY(at(m + ".0.b", b.bias0));
    H3_TRY(at(m + ".2.w", b.conv2));
    H3_TRY(at(m + ".2.b", b.bias2));
    H3_TRY(at(m + ".4.w", b.conv4));
    H3_TRY(at(m + ".4.b", b.bias4));
  }
  const int tgrow[3] = {7, 13, 19}, up[3] = {8, 14, 20};
  for (int i = 0; i < 3; ++i) {
    H3_TRY(at("g" + std::to_string(tgrow[i]) + ".w", w->tgrow[i]));
    H3_TRY(at("u" + std::to_string(up[i]) + ".w", w->upsample[i]));
  }
  H3_TRY(at("d22.w", w->outputConv));
  H3_TRY(at("d22.b", w->outputBias));
  out = std::move(w);
  return {};
}

} // namespace h3::model

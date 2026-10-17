#include "model/GgufTarget.hpp"
#include "model/GgufPreparation.hpp"

#include <algorithm>
#include <regex>
#include <string>

namespace splash::model {
std::vector<std::filesystem::path> findTargetGgufs(const std::filesystem::path &directory) {
  std::vector<std::filesystem::path> found;
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(directory, error))
    if (entry.path().extension() == ".gguf") found.push_back(entry.path());
  if (error) throw GgufError("cannot list target directory: " + directory.string());
  if (found.empty()) throw GgufError("target directory holds no GGUF: " + directory.string());
  if (found.size() == 1) return found;
  // A split GGUF's parts end in -0000K-of-0000N; any other second file is an error.
  std::sort(found.begin(), found.end());
  const std::regex part(R"(.*-(\d{5})-of-(\d{5})\.gguf)");
  for (size_t index = 0; index < found.size(); ++index) {
    std::smatch match;
    const std::string name = found[index].filename().string();
    if (!std::regex_match(name, match, part) || std::stoul(match[1]) != index + 1 ||
        std::stoul(match[2]) != found.size())
      throw GgufError("target directory holds more than one GGUF, and not the parts of one split GGUF: " +
                      directory.string());
  }
  return found;
}

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                   const std::vector<std::filesystem::path> &paths,
                                   const QwenTargetDimensions &geometry)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(paths)) {
  rotation_ = planned_->source.file().rotation();
  planned_->images = gguf::planImages(planned_->source.file(), geometry);
}

WeightFile GgufTargetLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  return images_.load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                         const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, buffer, planned->images[index]);
                         planned->source.checkUnchanged();
                       }});
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 2) throw GgufError("target layer is out of range");
  return open(index);
}

WeightFile GgufTargetLoader::head() { return open(planned_->images.size() - 2); }

WeightFile GgufTargetLoader::embedding() { return open(planned_->images.size() - 1); }

} // namespace splash::model

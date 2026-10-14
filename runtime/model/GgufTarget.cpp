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

GgufTargetLoader::Planned::Planned(const std::vector<std::filesystem::path> &paths) {
  for (const std::filesystem::path &path : paths) files.emplace_back(path);
}

void GgufTargetLoader::Planned::checkUnchanged() const {
  for (const WeightSource &file : files) file.checkUnchanged();
}

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                   const std::vector<std::filesystem::path> &paths,
                                   const QwenTargetDimensions &geometry)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(paths)) {
  std::vector<WeightSource *> parsed;
  for (WeightSource &source : planned_->files) parsed.push_back(&source);
  const GgufFile file(parsed);
  rotation_ = file.rotation();
  planned_->images = gguf::planImages(file, geometry);
  planned_->layers = geometry.layers;
  planned_->pleHash = gguf::readPleHash(file, geometry);
}

WeightFile GgufTargetLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  return images_.load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                         const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, buffer, planned->images[index]);
                         planned->checkUnchanged();
                       }});
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  if (index >= planned_->layers) throw GgufError("target layer is out of range");
  return open(index);
}

WeightFile GgufTargetLoader::head() { return open(planned_->layers); }

WeightFile GgufTargetLoader::embedding() { return open(planned_->layers + 1); }

WeightFile GgufTargetLoader::ple() {
  if (planned_->images.size() != planned_->layers + 3) throw GgufError("the target has no PLE table");
  return open(planned_->layers + 2);
}

GgufMtpLoader::GgufMtpLoader(metal::MetalBackend &backend, WeightImages &images,
                             const std::filesystem::path &path, const QwenTargetDimensions &geometry)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(path)) {
  const GgufFile file(planned_->source);
  planned_->image = gguf::planMtpImage(file, geometry);
}

WeightFile GgufMtpLoader::open() {
  const gguf::Image &plan = planned_->image;
  return images_.load({"mtp/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_](std::span<uint8_t>,
                                                                 const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, buffer, planned->image);
                         planned->source.checkUnchanged();
                       }});
}

} // namespace splash::model

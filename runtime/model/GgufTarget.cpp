#include "model/GgufTarget.hpp"
#include "model/GgufImageLayout.hpp"
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

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, const std::vector<std::filesystem::path> &paths,
                                   const gguf::TargetGeometry &geometry, PreparationCheck admitConversion)
    : backend_(backend),
      files_([&backend] { backend.checkOperation(); }, std::move(admitConversion),
             [this] { checkUnchanged(); }) {
  for (const std::filesystem::path &path : paths) {
    sourceFiles_.emplace_back(path, [&backend] { backend.checkOperation(); });
    sources_.push_back(&sourceFiles_.back());
  }
  std::vector<WeightSource *> parsed;
  for (WeightSource &source : sourceFiles_) parsed.push_back(&source);
  const GgufFile file(parsed);
  checkUnchanged();
  // Validates the whole source before its tensor data is hashed.
  images_ = gguf::planImages(file, geometry);
  layers_ = geometry.layers;
  pleHash_ = gguf::readPleHash(file, geometry);
  rotation_ = file.rotation();
  for (const gguf::Image &image : images_) {
    backend.checkOperation();
    weights_.push_back(ggufImageWeight(sources_, image));
  }
}

void GgufTargetLoader::checkUnchanged() const {
  for (const WeightSource *source : sources_) source->checkUnchanged();
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  if (index >= layers_) throw GgufError("target layer is out of range");
  return open(index);
}

WeightFile GgufTargetLoader::head() { return open(layers_); }

WeightFile GgufTargetLoader::embedding() { return open(layers_ + 1); }

WeightFile GgufTargetLoader::ple() {
  if (images_.size() != layers_ + 3) throw GgufError("the target has no PLE table");
  return open(layers_ + 2);
}

void GgufTargetLoader::prepare() {
  for (size_t index = 0; index < images_.size(); ++index) static_cast<void>(files_.prepare(weights_[index], writer(index)));
}

WeightWriter GgufTargetLoader::writer(size_t index) {
  return [this, index](int destination, const PreparationCheck &admit) {
    writeGgufImage(backend_, sources_, destination, images_[index], admit);
  };
}

WeightFile GgufTargetLoader::open(size_t index) {
  const gguf::Image &image = images_[index];
  return files_.open(backend_, weights_[index], writer(index), kGgufImageMagic, image.layer, image.type);
}

GgufMtpLoader::GgufMtpLoader(metal::MetalBackend &backend, const std::filesystem::path &path,
                             const gguf::TargetGeometry &geometry, PreparationCheck admitConversion)
    : backend_(backend), source_(path, [&backend] { backend.checkOperation(); }),
      files_([&backend] { backend.checkOperation(); }, std::move(admitConversion),
             [this] { source_.checkUnchanged(); }) {
  const GgufFile file(source_);
  source_.checkUnchanged();
  image_ = gguf::planMtpImage(file, geometry);
  const WeightSource *sources[] = {&source_};
  weight_ = ggufImageWeight(sources, image_);
}

void GgufMtpLoader::prepare() {
  static_cast<void>(files_.prepare(weight_, [this](int destination, const PreparationCheck &admit) {
    const WeightSource *sources[] = {&source_};
    writeGgufImage(backend_, sources, destination, image_, admit);
  }));
}

WeightFile GgufMtpLoader::open() {
  return files_.open(backend_, weight_, [this](int destination, const PreparationCheck &admit) {
    const WeightSource *sources[] = {&source_};
    writeGgufImage(backend_, sources, destination, image_, admit);
  }, kGgufImageMagic, image_.layer, image_.type);
}

} // namespace splash::model

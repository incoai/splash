#include "model/GgufTarget.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/GgufPreparation.hpp"

namespace splash::model {
std::filesystem::path findTargetGguf(const std::filesystem::path &directory) {
  std::filesystem::path found;
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
    if (entry.path().extension() != ".gguf") continue;
    if (!found.empty()) throw GgufError("target directory holds more than one GGUF: " + directory.string());
    found = entry.path();
  }
  if (error) throw GgufError("cannot list target directory: " + directory.string());
  if (found.empty()) throw GgufError("target directory holds no GGUF: " + directory.string());
  return found;
}

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &path,
                                   const gguf::TargetGeometry &geometry, PreparationCheck admitConversion)
    : backend_(backend), source_(path, [&backend] { backend.checkOperation(); }),
      files_([&backend] { backend.checkOperation(); }, std::move(admitConversion),
             [this] { source_.checkUnchanged(); }) {
  const GgufFile file(source_);
  source_.checkUnchanged();
  // Validates the whole source before its tensor data is hashed.
  images_ = gguf::planImages(file, geometry);
  for (const gguf::Image &image : images_) {
    backend.checkOperation();
    weights_.push_back(ggufImageWeight(source_, image));
  }
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  if (index >= images_.size() - 2) throw GgufError("target layer is out of range");
  return open(index);
}

WeightFile GgufTargetLoader::head() { return open(images_.size() - 2); }

WeightFile GgufTargetLoader::embedding() { return open(images_.size() - 1); }

void GgufTargetLoader::prepare() {
  for (size_t index = 0; index < images_.size(); ++index) static_cast<void>(files_.prepare(weights_[index], writer(index)));
}

WeightWriter GgufTargetLoader::writer(size_t index) {
  return [this, index](int destination, const PreparationCheck &admit) {
    writeGgufImage(backend_, source_, destination, images_[index], admit);
  };
}

WeightFile GgufTargetLoader::open(size_t index) {
  const gguf::Image &image = images_[index];
  return files_.open(backend_, weights_[index], writer(index), kGgufImageMagic, image.layer, image.type);
}

} // namespace splash::model

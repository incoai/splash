#include "model/GgufTarget.hpp"
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
      images_(PreparedFiles([&backend] { backend.checkOperation(); }, std::move(admitConversion),
                            [this] { source_.checkUnchanged(); }),
              [this](const gguf::Image &image) {
                return WeightWriter([this, &image](int destination, const PreparationCheck &admit) {
                  writeGgufImage(backend_, source_, destination, image, admit);
                });
              }) {
  const GgufFile file(source_);
  source_.checkUnchanged();
  rotation_ = file.rotation();
  // Validates the whole source before its tensor data is hashed.
  for (gguf::Image &image : gguf::planImages(file, geometry)) {
    backend.checkOperation();
    PreparedWeight weight = ggufImageWeight(source_, image);
    images_.add(std::move(image), std::move(weight));
  }
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  if (index >= images_.size() - 2) throw GgufError("target layer is out of range");
  return images_.open(backend_, index);
}

WeightFile GgufTargetLoader::head() { return images_.open(backend_, images_.size() - 2); }

WeightFile GgufTargetLoader::embedding() { return images_.open(backend_, images_.size() - 1); }

void GgufTargetLoader::prepare() { images_.prepare(); }

} // namespace splash::model

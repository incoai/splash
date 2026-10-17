#include "model/GgufTarget.hpp"
#include "model/GgufPreparation.hpp"

#include <algorithm>
#include <regex>
#include <span>
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
                                   const std::vector<std::filesystem::path> &paths, std::string_view architecture,
                                   const QwenTargetDimensions &geometry, SparseRowStoreOptions table,
                                   ExpertStoreOptions experts)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(paths)) {
  planned_->images =
      gguf::planImages(planned_->source.file(), architecture, geometry,
                       {.streamExperts = experts.memoryBytes != 0, .streamTable = table.cacheBytes != 0});
  planned_->layers = geometry.layers;
  planned_->table = table;
  planned_->experts = experts;
}

StreamedBlocks GgufTargetLoader::streamedExperts() const {
  return {gguf::streamedExperts(std::span(planned_->images).first(planned_->layers)),
          {[planned = planned_] { planned->source.checkUnchanged(); }}};
}

const gguf::Image *GgufTargetLoader::tablePlan() const noexcept {
  const size_t index = planned_->layers + 2;
  return index < planned_->images.size() ? &planned_->images[index] : nullptr;
}

WeightFile GgufTargetLoader::table() {
  const gguf::Image *plan = tablePlan();
  if (!plan) throw GgufError("the target has no table");
  if (plan->streamedRows) throw GgufError("the target's table streams");
  return open(planned_->layers + 2);
}

std::optional<GgufTargetLoader::StreamedTable> GgufTargetLoader::streamTable() const {
  const gguf::Image *plan = tablePlan();
  if (!plan || !plan->streamedRows) return std::nullopt;
  const gguf::StreamedRows &rows = *plan->streamedRows;
  const WeightSource &source = file().source(rows.file);
  // The file the metadata was parsed from, unwritten since.
  source.checkUnchanged();
  return StreamedTable{
      std::make_shared<SparseRowStore>(SparseRowTable{rows.tensor, rows.rows, rows.rowBytes, rows.offset},
                                       openRowReader(source), planned_->table),
      rows.format,
      // As WeightImages::load records the image (WeightFile).
      {"target/" + plan->name, plan->magic, plan->layer, plan->type, plan->bytes, images_.contentIdentity()}};
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
  if (index >= planned_->layers) throw GgufError("target layer is out of range");
  return open(index);
}

WeightFile GgufTargetLoader::head() { return open(planned_->layers); }

WeightFile GgufTargetLoader::embedding() { return open(planned_->layers + 1); }

} // namespace splash::model

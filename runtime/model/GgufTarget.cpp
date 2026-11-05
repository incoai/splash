#include "model/GgufTarget.hpp"
#include "model/GgufPreparation.hpp"

#include <algorithm>
#include <regex>
#include <set>
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
  // A split GGUF is the parts of one name, NAME-0000K-of-0000N.gguf, K = 1..N,
  // as the installer holds it (upstream.split_order); any other second file is
  // an error. Of one name, name order is split order.
  std::sort(found.begin(), found.end());
  const std::regex part(R"((.*)-(\d{5})-of-(\d{5})\.gguf)");
  const std::string notSplit =
      "target directory holds more than one GGUF, and not the parts of one split GGUF: " + directory.string();
  struct Part {
    std::string name;
    unsigned long number, count;
  };
  std::vector<Part> parts;
  std::set<std::string> names;
  for (const std::filesystem::path &path : found) {
    std::smatch match;
    const std::string file = path.filename().string();
    if (!std::regex_match(file, match, part)) throw GgufError(notSplit);
    parts.push_back({match[1], std::stoul(match[2]), std::stoul(match[3])});
    names.insert(parts.back().name);
  }
  if (names.size() > 1) {
    std::string listed;
    for (const std::string &name : names) listed += (listed.empty() ? "" : ", ") + name;
    throw GgufError("target directory mixes the parts of several split GGUFs (" + listed +
                    "); a split GGUF is the parts of one name, NAME-0000K-of-0000N.gguf: " + directory.string());
  }
  for (size_t index = 0; index < parts.size(); ++index)
    if (parts[index].number != index + 1 || parts[index].count != parts.size()) throw GgufError(notSplit);
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
  planned_->firstMoeLayer = geometry.leadingDenseLayers;
  planned_->table = table;
  planned_->experts = experts;
}

StreamedBlocks GgufTargetLoader::streamedExperts() const {
  const std::span layers = std::span(planned_->images).first(planned_->layers);
  return {gguf::streamedExperts(layers.subspan(planned_->firstMoeLayer)), planned_,
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
                                       openRowReader(std::shared_ptr<const WeightSource>(planned_, &source)),
                                       planned_->table),
      rows.format, imageRecord(images_, "target/", *plan)};
}

WeightFile GgufTargetLoader::open(size_t index) {
  return loadImage(backend_, images_, "target/", {planned_, &planned_->images[index]},
                   [planned = planned_] { planned->source.checkUnchanged(); });
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  if (index >= planned_->layers) throw GgufError("target layer is out of range");
  return open(index);
}

WeightFile GgufTargetLoader::head() { return open(planned_->layers); }

WeightFile GgufTargetLoader::embedding() { return open(planned_->layers + 1); }

} // namespace splash::model

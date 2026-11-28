#include "model/GgufTarget.hpp"
#include "model/GgufPreparation.hpp"

#include <algorithm>
#include <regex>
#include <set>
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
                                   const std::vector<std::filesystem::path> &paths, const GgufTargetPlan &plan,
                                   SparseRowStoreOptions table, ExpertStoreOptions experts,
                                   gguf::Float16Weights weights)
    : GgufTargetLoader(backend, images, std::make_shared<Planned>(paths), plan, std::move(table), experts, weights) {}

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                   std::shared_ptr<Planned> planned, const GgufTargetPlan &plan,
                                   SparseRowStoreOptions table, ExpertStoreOptions experts,
                                   gguf::Float16Weights weights)
    : TargetImages(backend, images, std::shared_ptr<std::vector<gguf::Image>>(planned, &planned->images),
                   [planned] { planned->source.checkUnchanged(); }, plan.layerImages, plan.firstMoeLayer,
                   std::move(table), experts, weights),
      planned_(std::move(planned)) {
  const gguf::PlanOptions options = readingValues(planOptions());
  planned_->images = plan.images(planned_->source.file(), options);
}

std::optional<TargetImages::StreamedTable> GgufTargetLoader::streamTable() const {
  const gguf::Image *plan = tablePlan();
  if (!plan || !plan->streamedRows) return std::nullopt;
  const gguf::StreamedRows &rows = *plan->streamedRows;
  const WeightSource &source = file().source(rows.file);
  // The file the metadata was parsed from, unwritten since.
  source.checkUnchanged();
  return StreamedTable{
      std::make_shared<SparseRowStore>(SparseRowTable{rows.tensor, rows.rows, rows.rowBytes, rows.offset},
                                       openRowReader(std::shared_ptr<const WeightSource>(planned_, &source)),
                                       tableOptions()),
      rows.format, tableRecord(*plan)};
}

} // namespace splash::model

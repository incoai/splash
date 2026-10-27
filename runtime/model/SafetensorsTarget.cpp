#include "model/SafetensorsTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"

#include <algorithm>

namespace splash::model {
namespace {

// A virtual file of native rows. readRepackRows builds just the requested
// rows from their safetensors codes, scales and biases, across shard files.
class TableRowReader final : public RowReader {
public:
  TableRowReader(gguf::StreamedRows rows, std::shared_ptr<const void> owner)
      : rows_(std::move(rows)), owner_(std::move(owner)) {
    repack_.format = rows_.format;
    repack_.columns = rows_.rowBytes / kQuantFormats[rows_.format].block_bytes *
                       kQuantFormats[rows_.format].block_elements;
    repack_.sources = rows_.sources;
    for (const auto &source : rows_.sources) {
      addSource(source.file);
      for (const auto *tensor : {source.safetensors.scales, source.safetensors.biases})
        if (tensor) addSource(tensor->file);
    }
  }
  void read(uint64_t offset, std::span<uint8_t> bytes) override {
    if (offset % rows_.rowBytes || bytes.size() % rows_.rowBytes || offset > size() || bytes.size() > size() - offset)
      throw SparseRowError("PLE table read is outside its native rows");
    // BF16 embeddings are raw values, including widths shorter than the
    // 256-value blocks BF16 projections use. Repack's block row sizing is
    // therefore not the row sizing of this table.
    if (rows_.format == GGUF_FMT_BF16) {
      uint64_t first = offset / rows_.rowBytes;
      for (const auto &source : rows_.sources) {
        if (first >= source.rows) {
          first -= source.rows;
          continue;
        }
        const uint64_t count = std::min<uint64_t>(bytes.size(), (source.rows - first) * rows_.rowBytes);
        source.file->readData(source.offset + (source.first + first) * rows_.rowBytes, bytes.first(count));
        bytes = bytes.subspan(count);
        if (bytes.empty()) return;
        first = 0;
      }
      throw SparseRowError("PLE table read is outside its BF16 sources");
    }
    readRepackRows(repack_, offset / rows_.rowBytes, bytes.size() / rows_.rowBytes, bytes.data());
  }
  void checkUnchanged() override {
    for (const auto *source : sources_) source->checkUnchanged();
  }
  [[nodiscard]] uint64_t size() const override { return rows_.rows * rows_.rowBytes; }
private:
  void addSource(const WeightSource *source) {
    if (std::find(sources_.begin(), sources_.end(), source) == sources_.end()) sources_.push_back(source);
  }
  gguf::StreamedRows rows_;
  gguf::Repack repack_;
  std::shared_ptr<const void> owner_;
  std::vector<const WeightSource *> sources_;
};

} // namespace

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const QwenTargetDimensions &geometry, ExpertStoreOptions experts,
                                                 SparseRowStoreOptions table)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)), experts_(experts) {
  planned_->images =
      safetensors::planImages(planned_->checkpoint, geometry, {.streamExperts = experts.memoryBytes != 0,
                                                               .streamTable = table.cacheBytes != 0});
  planned_->layers = geometry.layers;
  planned_->table = std::move(table);
  float32Norms_ = geometry.hyperConnected() || safetensors::float32Norms(safetensors::moduleNames(planned_->checkpoint));
}

StreamedBlocks SafetensorsTargetLoader::streamedExperts() const {
  return {gguf::streamedExperts(std::span(planned_->images).first(planned_->layers)), planned_,
          {[planned = planned_] { planned->checkpoint.checkUnchanged(); }}};
}

WeightFile SafetensorsTargetLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  return images_.load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                         const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, buffer, planned->images[index]);
                         planned->checkpoint.checkUnchanged();
                       },
                       plan.residentBytes});
}

WeightFile SafetensorsTargetLoader::layer(uint32_t index) {
  if (index >= planned_->layers) throw WeightStoreError("target layer is out of range");
  return open(index);
}

WeightFile SafetensorsTargetLoader::head() { return open(planned_->layers); }

WeightFile SafetensorsTargetLoader::embedding() { return open(planned_->layers + 1); }

WeightFile SafetensorsTargetLoader::table() {
  if (planned_->images.size() <= planned_->layers + 2) throw WeightStoreError("target has no PLE table");
  if (planned_->images.back().streamedRows) throw WeightStoreError("target's PLE table streams");
  return open(planned_->layers + 2);
}

uint64_t SafetensorsTargetLoader::tableRows() const {
  if (planned_->images.size() <= planned_->layers + 2) throw WeightStoreError("target has no PLE table");
  const auto &plan = planned_->images.back();
  if (plan.streamedRows) return plan.streamedRows->rows;
  uint64_t rows = 0;
  for (const auto &copy : plan.copies) rows += copy.source.rows;
  return rows;
}

std::optional<StreamedTable> SafetensorsTargetLoader::streamTable() const {
  if (planned_->images.size() <= planned_->layers + 2 || !planned_->images.back().streamedRows) return std::nullopt;
  const auto &plan = planned_->images.back();
  const auto &rows = *plan.streamedRows;
  planned_->checkpoint.checkUnchanged();
  return StreamedTable{
      std::make_shared<SparseRowStore>(SparseRowTable{rows.tensor, rows.rows, rows.rowBytes, 0},
                                       std::make_unique<TableRowReader>(rows, planned_), planned_->table),
      rows.format,
      {"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes, images_.contentIdentity()}};
}

} // namespace splash::model

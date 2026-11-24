#include "model/SafetensorsTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"
#include "metal/abi/QuantFormat.h"


namespace splash::model {
namespace {

// The native rows of a table that streams from a checkpoint, its shards'
// rows in order (the copies of its image), as one file of rows back to back
// that SparseRowStore reads: each read builds whole rows from their codes,
// scales and biases (readRepackRows).
class TableReader final : public RowReader {
public:
  TableReader(gguf::Repack table, std::shared_ptr<const void> planned, const SafetensorsCheckpoint &checkpoint)
      : table_(std::move(table)), planned_(std::move(planned)), checkpoint_(checkpoint),
        rowBytes_(ggufRowBytes(kQuantFormats[table_.format], table_.columns)) {}
  void read(uint64_t offset, std::span<uint8_t> bytes) override {
    if (offset % rowBytes_ || bytes.size() % rowBytes_)
      throw SparseRowError("a streamed table is read by whole rows");
    readRepackRows(table_, offset / rowBytes_, bytes.size() / rowBytes_, bytes.data());
  }
  void checkUnchanged() override { checkpoint_.checkUnchanged(); }
  [[nodiscard]] uint64_t size() const override { return table_.rows * rowBytes_; }

private:
  gguf::Repack table_;
  std::shared_ptr<const void> planned_; // holds the checkpoint
  const SafetensorsCheckpoint &checkpoint_;
  uint64_t rowBytes_;
};

} // namespace

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const SafetensorsTargetPlan &plan, SparseRowStoreOptions table,
                                                 ExpertStoreOptions experts)
    : SafetensorsTargetLoader(backend, images, std::make_shared<Planned>(directory), plan, std::move(table),
                              experts) {}

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 std::shared_ptr<Planned> planned, const SafetensorsTargetPlan &plan,
                                                 SparseRowStoreOptions table, ExpertStoreOptions experts)
    : TargetImages(backend, images, std::shared_ptr<std::vector<gguf::Image>>(planned, &planned->images),
                   [planned] { planned->checkpoint.checkUnchanged(); }, plan.layerImages, plan.firstMoeLayer,
                   std::move(table), experts),
      planned_(std::move(planned)) {
  gguf::PlanOptions options = planOptions();
  options.exactlyMxfp8 = exactlyMxfp8;
  SafetensorsTargetPlan::Images plannedImages = plan.images(planned_->checkpoint, options);
  planned_->images = std::move(plannedImages.images);
  float32Norms_ = plannedImages.float32Norms;
}

std::optional<TargetImages::StreamedTable> SafetensorsTargetLoader::streamTable() const {
  const gguf::Image *plan = tablePlan();
  if (!plan || !plan->streamedRows) return std::nullopt;
  const gguf::StreamedRows &rows = *plan->streamedRows;
  const QuantFormat &format = kQuantFormats[rows.format];
  gguf::Repack table;
  table.format = rows.format;
  table.rows = rows.rows;
  table.columns = rows.rowBytes / format.block_bytes * format.block_elements;
  for (const gguf::Copy &copy : plan->copies) table.sources.push_back(copy.source);
  planned_->checkpoint.checkUnchanged();
  return StreamedTable{
      std::make_shared<SparseRowStore>(SparseRowTable{rows.tensor, rows.rows, rows.rowBytes, 0},
                                       std::make_unique<TableReader>(std::move(table), planned_, planned_->checkpoint),
                                       tableOptions()),
      rows.format, tableRecord(*plan)};
}

} // namespace splash::model

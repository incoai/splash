#include "model/ModelSizing.hpp"

#include "model/DSpark.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/ExpertStore.hpp"
#include "model/GgufTarget.hpp"
#include "model/ModelFactory.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SparseRowStore.hpp"
#include "model/VisionLoader.hpp"
#include "model/WeightSource.hpp"

#include <span>
#include <type_traits>
#include <vector>

namespace splash::model {
namespace {

uint64_t loadedBytes(std::span<const gguf::Image> images) {
  uint64_t bytes = 0;
  for (const gguf::Image &image : images)
    bytes += gguf::loadedBytes(image);
  return bytes;
}

// What a target of `layout` whose images `plan` plans for what streams
// takes, while their sources are open: what loads while every part `streams`
// lets stream does, and what each such part adds where it loads instead,
// the routed experts' slot table with them.
template <class Layout, class Plan>
StreamableWeights targetWeights(const Layout &layout, const Plan &plan, gguf::PlanOptions streams) {
  streams.streamExperts = streams.streamExperts && layout.ffnKind == QwenFfnKind::SparseMoe;
  StreamableWeights result;
  std::vector<gguf::Image> streamed;
  try {
    streamed = plan(streams);
  } catch (const GgufError &error) {
    if (!streams.streamTable)
      throw;
    // A table that cannot stream loads with the weights; whatever else
    // failed, the plan without it fails again.
    result.tableRefusal = error.what();
    streams.streamTable = false;
    streamed = plan(streams);
  }
  result.loadedBytes = loadedBytes(streamed);
  for (const gguf::Image &image : streamed)
    if (image.streamedRows) {
      result.tableBytes = loadedBytes(plan({.streamExperts = streams.streamExperts})) - result.loadedBytes;
      const SparseRowStoreOptions rows = streamedTableRows(layout);
      result.tableRowBytes = image.streamedRows->rowBytes;
      result.tableMinimumBytes = SparseRowStore::workingBytes(image.streamedRows->rowBytes, rows.prefetchRows,
                                                            rows.stageRows) +
                               SparseRowStore::cacheBytesFor(1, image.streamedRows->rowBytes);
    }
  if (streams.streamExperts) {
    result.expertBytes = loadedBytes(plan({.streamTable = result.tableBytes != 0})) - result.loadedBytes +
                         residentExpertSlotsBytes(layout.experts);
    StreamedBlocks blocks;
    blocks.blocks = gguf::streamedExperts(
        std::span(streamed).subspan(layout.leadingDenseLayers, layout.layers - layout.leadingDenseLayers));
    result.expertMinimumBytes = ExpertStore::minimumBytes(blocks, layout.moeBlockShape());
  }
  return result;
}

// DeepSeek-V4.1's Engram tables, the last part of its split `target`
// (deepSeek41EngramTables): a table that only streams, through a row cache,
// or, where `streams` leaves it to load, more than a Mac holds. The hash's
// buffers, from the metadata, load with the weights.
void addEngramTables(StreamableWeights &weights, const GgufFile &target, const DeepSeek41Layout &layout,
                     bool streams) {
  weights.tableStreamsOnly = true;
  weights.loadedBytes += deepSeek41EngramHashBytes(layout);
  const DeepSeek41EngramTables tables = deepSeek41EngramTables(target, layout);
  const uint64_t bytes = tables.table.rows * tables.table.rowBytes;
  if (!streams) {
    weights.loadedBytes += bytes;
    return;
  }
  const SparseRowStoreOptions rows = streamedTableRows(layout);
  weights.tableBytes = bytes;
  weights.tableRowBytes = tables.table.rowBytes;
  weights.tableMinimumBytes = SparseRowStore::workingBytes(tables.table.rowBytes, rows.prefetchRows, rows.stageRows) +
                              SparseRowStore::cacheBytesFor(1, tables.table.rowBytes);
}

// The bytes of the images of DSpark's GGUF in `directory`, its experts
// with its layers.
uint64_t dsparkImageBytes(const std::filesystem::path &directory, const ModelDescriptor &descriptor) {
  const auto *target = std::get_if<DeepSeek41Layout>(&descriptor.target);
  const std::optional<std::filesystem::path> path = findDSparkGguf(directory);
  if (!target || !path) throw std::invalid_argument("a DSpark draft serves DeepSeek-V4.1 from its GGUF in draft/");
  const OpenGguf file(std::vector<std::filesystem::path>{*path});
  const DSparkLayout layout = readDSparkLayout(file.file(), *target);
  uint64_t bytes = 0;
  for (const gguf::Image &image :
       gguf::planDSparkImages(file.file(), layout.layers, layout.targetLayerCount, layout.markovRank))
    bytes += image.bytes;
  return bytes;
}

} // namespace

StreamableWeights streamableWeights(const std::filesystem::path &root, const ModelDescriptor &descriptor,
                                    gguf::PlanOptions streams) {
  StreamableWeights result = std::visit(
      [&](const auto &layout) {
        const std::filesystem::path directory = root / "target";
        if (descriptor.targetSource == TargetSource::Gguf) {
          const OpenGguf target(findTargetGgufs(directory));
          StreamableWeights weights = targetWeights(
              layout,
              [&](gguf::PlanOptions options) {
                return gguf::planImages(target.file(), layout.ggufArchitecture, layout, options);
              },
              streams);
          if constexpr (std::is_same_v<std::decay_t<decltype(layout)>, DeepSeek41Layout>)
            addEngramTables(weights, target.file(), layout, streams.streamTable);
          return weights;
        }
        const SafetensorsCheckpoint checkpoint(directory);
        return targetWeights(
            layout, [&](gguf::PlanOptions options) { return safetensors::planImages(checkpoint, layout, options); },
            streams);
      },
      descriptor.target);
  if (descriptor.draft.dflash()) result.loadedBytes += draftImageBytes(root / "draft", descriptor.draft);
  if (descriptor.draft.dspark()) result.loadedBytes += dsparkImageBytes(root / "draft", descriptor);
  if (descriptor.hasVision()) result.loadedBytes += visionImageBytes(descriptor.vision);
  return result;
}

} // namespace splash::model

#include "model/ModelSizing.hpp"

#include "model/DraftCheckpoint.hpp"
#include "model/ExpertStore.hpp"
#include "model/GgufTarget.hpp"
#include "model/ModelFactory.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SparseRowStore.hpp"
#include "model/VisionLoader.hpp"

#include <span>
#include <vector>

namespace splash::model {
namespace {

// The prompt whose table rows an automatic row cache holds, every id its
// stages read: a cache hits only rows read before, and a 128K-token prompt
// reads about 1.1M distinct rows of its 2M ids. A larger cache hits no more
// often (Qwen3.8-Flash-Next, 256 MiB to 2 GiB), while each GiB more in the
// expert store makes a 24 GB Mac decode 11-23% faster: the rest goes to the
// experts.
constexpr uint64_t kTableShareTokens = 128 * 1024;

uint64_t loadedBytes(std::span<const gguf::Image> images) {
  uint64_t bytes = 0;
  for (const gguf::Image &image : images)
    bytes += gguf::loadedBytes(image);
  return bytes;
}

// What a target of `layout` whose images `plan` plans for what streams
// takes, while their sources are open: what loads while every part `streams`
// lets stream does, and what each such part adds where it loads instead,
// the routed experts' slot table with them. Its images begin with
// `layerImages` layer images, the first MoE layer's `firstMoeLayer`.
template <class Layout, class Plan>
StreamableWeights targetWeights(const Layout &layout, const Plan &plan, gguf::PlanOptions streams,
                                uint32_t layerImages, uint32_t firstMoeLayer) {
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
      const uint64_t working =
          SparseRowStore::workingBytes(image.streamedRows->rowBytes, rows.prefetchRows, rows.stageRows);
      result.tableMinimumBytes = working + SparseRowStore::cacheBytesFor(1, image.streamedRows->rowBytes);
      // A stage takes a prefill chunk's ids (SparseRowStoreOptions::stageRows).
      result.tableShareBytes =
          working + SparseRowStore::cacheBytesFor(
                        kTableShareTokens / ExecutionLimits::prefillTokenBudget * rows.stageRows,
                        image.streamedRows->rowBytes);
    }
  if (streams.streamExperts) {
    result.expertBytes = loadedBytes(plan({.streamTable = result.tableBytes != 0})) - result.loadedBytes +
                         residentExpertSlotsBytes(layout.experts);
    StreamedBlocks blocks;
    blocks.blocks = gguf::streamedExperts(
        std::span(streamed).subspan(firstMoeLayer, layerImages - firstMoeLayer));
    result.expertMinimumBytes = ExpertStore::minimumBytes(blocks, layout.moeBlockShape());
  }
  return result;
}

} // namespace

StreamableWeights streamableWeights(const std::filesystem::path &root, const ModelDescriptor &descriptor,
                                    gguf::PlanOptions streams) {
  StreamableWeights result = std::visit(
      [&](const auto &layout) {
        const std::filesystem::path directory = root / "target";
        if (descriptor.targetSource == TargetSource::Gguf) {
          const OpenGguf target(findTargetGgufs(directory));
          const GgufTargetPlan plan = ggufTargetPlan(layout);
          return targetWeights(
              layout, [&](gguf::PlanOptions options) { return plan.images(target.file(), options); }, streams,
              plan.layerImages, plan.firstMoeLayer);
        }
        const SafetensorsCheckpoint checkpoint(directory);
        const safetensors::NormStorage norms = safetensors::normStorage(checkpoint, layout);
        return targetWeights(
            layout,
            [&](gguf::PlanOptions options) { return safetensors::planImages(checkpoint, layout, norms, options); },
            streams, layout.layers, layout.leadingDenseLayers);
      },
      descriptor.target);
  result.loadedBytes += draftImageBytes(root / "draft", descriptor.draft);
  if (descriptor.hasVision()) result.loadedBytes += visionImageBytes(descriptor.vision);
  return result;
}

} // namespace splash::model

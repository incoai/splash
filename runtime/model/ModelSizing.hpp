#pragma once

// What a model's weights take, sized from its files before it loads, which
// the memory planner sizes the stores of the parts that stream from the SSD
// by (engine::planStreamedMemory): the images that load whatever streams,
// and each part that may stream, whole and at its smallest store.

#include "model/GgufImage.hpp"
#include "model/ModelDescriptor.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace splash::model {

struct StreamableWeights final {
  // The images the model loads while every part below streams: the
  // target's without its routed experts or table, the draft's and the
  // vision tower's.
  uint64_t loadedBytes = 0;
  // A MoE target's routed experts: what the model takes more where they
  // load with their layers, their planes in the images and their slot table
  // (residentExpertSlots), and the smallest store that streams them
  // (ExpertStore::minimumBytes); zero for a dense FFN.
  uint64_t expertBytes = 0;
  uint64_t expertFloorBytes = 0;
  // The table a GGUF target holds (Qwen3.8-Flash-Next's PLE n-gram table):
  // what its image takes where it loads, the smallest row cache that
  // streams it, which holds one row beside its working memory
  // (SparseRowStore), and the bytes of a row, which a step stages
  // (StreamedParts::tableRowBytes); zero without one.
  uint64_t tableBytes = 0;
  uint64_t tableFloorBytes = 0;
  uint64_t tableRowBytes = 0;
  // Why the table cannot stream (gguf::tableRows), which then loads with
  // the weights: among loadedBytes, with no part of its own.
  std::string tableRefusal;
};

// The model's weights in `root`, planned from its files' metadata as its
// loaders plan them, without loading them, with the parts `streams` lets
// stream as parts of their own: one it does not is among the images that
// load.
[[nodiscard]] StreamableWeights streamableWeights(const std::filesystem::path &root,
                                                  const ModelDescriptor &descriptor, gguf::PlanOptions streams);

} // namespace splash::model

#pragma once

// Flint's target (model/Flint.hpp): its layers over the blocks every Qwen
// target shares (model/QwenTarget.hpp). Layer l reads its input rows from
// buffers.hidden[l & 1]; its attention and its FFN each read the RMS norm of
// the rows before them and add their output to them, the FFN into the other
// buffer, as the Qwen3.5 architecture's layers do, but every layer attends
// and none keeps GDN state. A layer attends as its variant does (sliding or
// not, rotated or not: QwenTargetGeometry::attentionVariant) with its sinks,
// which the shared attention mixer takes from the layout and the layer's
// weights; its FFN is dense on the leading layers and a MoE block, routed as
// the target's MoE shape says, on the others. The layers the draft reads copy
// their output rows into the captured rows, and the head reads the final
// norm of the last layer's.

#include "model/QwenTarget.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

struct FlintLayout;
struct FlintLayerWeights;

class FlintTarget final : public QwenTarget {
public:
  FlintTarget(const QwenTargetWeights<FlintLayout, FlintLayerWeights> &weights, const QwenTargetGeometry &geometry,
              metal::MetalBackend &backend, const ops::ExecutionPlans &operators);

  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) override;
  void addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers, const kv::PageStorage &kv,
                 SpilledKvReads &spilled, std::span<const kv::ChunkedPrefillParams> chunks,
                 uint32_t lanes) const override;
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t lanes, ops::LinearScratch scratch) const override;

private:
  // The shared prefill's layer loop runs the family's (QwenTarget::addPrefillLayers).
  friend class QwenTarget;

  void addPrefillBegin(PrefillStep &) const {}
  [[nodiscard]] bool attends(uint32_t) const { return true; }
  void addPrefillLayer(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] const ops::BlockMoeWeights *moe(uint32_t index) const;
  void addPrefillExperts(PrefillStep &step, uint32_t index) const;
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step) const;

  const QwenTargetWeights<FlintLayout, FlintLayerWeights> &weights_;
};

} // namespace splash::model

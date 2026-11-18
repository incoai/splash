#include "model/NemotronHTarget.hpp"

#include "model/ExpertStore.hpp"
#include "ops/Mamba2.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace splash::model {

namespace {

constexpr uint32_t kVerifyRows = ExecutionLimits::targetVerifyRows;
// A full batch's verify rows, which the decode scratch holds (a step's storage
// lanes are at most a batch's, QwenTarget::decodeStorageLanes).
constexpr uint64_t kScratchRows = uint64_t{ExecutionLimits::maximumBatchWidth} * kVerifyRows;
// A prefill chunk's rows and the two slices a sequence's Mamba-2 rows run
// past its last row: the last slice's and the fold's (addPrefillMamba).
constexpr uint64_t kPrefillScratchRows = uint64_t{ExecutionLimits::prefillTokenBudget} + 2 * kVerifyRows;
constexpr uint64_t kAlignment = 256;

uint64_t alignUp(uint64_t bytes) { return (bytes + kAlignment - 1) / kAlignment * kAlignment; }

// The bytes of each tensor of rows the decode and prefill scratch hold, in
// order: packed, y, gated, latent, MoE latent, shared mid.
std::array<uint64_t, 6> rowTensorBytes(uint64_t rows) {
  const NemotronHLayout layout;
  return {rows * layout.mambaInProjectionRows * 2, rows * layout.mambaInnerSize * 4,
          rows * layout.mambaInnerSize * 2,        rows * layout.moeLatentSize * 2,
          rows * layout.moeLatentSize * 2,         rows * layout.sharedIntermediateSize * 2};
}

std::vector<uint64_t> decodeTensorBytes(const QwenTargetGeometry &) {
  const auto rows = rowTensorBytes(kScratchRows);
  return {rows.begin(), rows.end()};
}

// The prefill scratch's: its rows', then two one-layer state cells
// (convolution, recurrent each) and a one-layer tape of two slots.
std::vector<uint64_t> prefillTensorBytes(const QwenTargetGeometry &g) {
  const auto rows = rowTensorBytes(kPrefillScratchRows);
  std::vector<uint64_t> bytes(rows.begin(), rows.end());
  const ops::GdnStateStrides state = g.stateLayout.strides();
  for (int cell = 0; cell < 2; ++cell) bytes.insert(bytes.end(), {state.convolutionLayerBytes, state.recurrentLayerBytes});
  bytes.push_back(2 * mamba2TapeLayerBytes(g));
  return bytes;
}

uint64_t totalBytes(const std::vector<uint64_t> &sizes) {
  uint64_t total = 0;
  for (const uint64_t bytes : sizes) total += alignUp(bytes);
  return total;
}

std::vector<metal::MetalBuffer> place(metal::MetalBackend &backend, const metal::MetalBuffer &buffer,
                                      const std::vector<uint64_t> &sizes) {
  std::vector<metal::MetalBuffer> views;
  uint64_t cursor = 0;
  for (const uint64_t bytes : sizes) {
    views.push_back(backend.view(buffer, cursor, bytes));
    cursor += alignUp(bytes);
  }
  return views;
}

} // namespace

NemotronHTarget::NemotronHTarget(const NemotronHWeights &weights, const QwenTargetGeometry &geometry,
                                 metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights) {
  // ops::Mamba2 keeps the state in fp32, and its cached states are that
  // state (RuntimeResources' exactCachedStates).
  if (geometry_.stateLayout.format != ops::GdnStateFormat::Float32 || geometry_.stateLayout.cachedCodes)
    throw std::invalid_argument("Nemotron-H keeps its Mamba-2 state, cached states too, in fp32");
  const std::vector<uint64_t> decodeSizes = decodeTensorBytes(geometry_), prefillSizes = prefillTensorBytes(geometry_);
  decode_ = backend_.allocateBuffer(totalBytes(decodeSizes), metal::BufferStorage::Shared, "nemotron-h-decode-scratch");
  prefill_ =
      backend_.allocateBuffer(totalBytes(prefillSizes), metal::BufferStorage::Shared, "nemotron-h-prefill-scratch");
  const std::vector<metal::MetalBuffer> d = place(backend_, decode_, decodeSizes);
  decodeRows_ = {d[0], d[1], d[2], d[3], d[4], d[5]};
  const std::vector<metal::MetalBuffer> p = place(backend_, prefill_, prefillSizes);
  prefillRows_ = {p[0], p[1], p[2], p[3], p[4], p[5]};
  prefillStates_ = {{p[6], p[8]}, {p[7], p[9]}, p[10]};
}

QwenTargetScratchBytes NemotronHTarget::scratchBytes(const QwenTargetGeometry &geometry) {
  return {totalBytes(prefillTensorBytes(geometry)), totalBytes(decodeTensorBytes(geometry))};
}

QwenTargetScratchBytes NemotronHTarget::allocatedScratchBytes() const noexcept {
  return {prefill_.sizeBytes(), decode_.sizeBytes()};
}

std::vector<metal::MetalBuffer> NemotronHTarget::addPrefill(metal::CommandGraph &graph,
                                                            std::span<const QwenTargetPrefillChunk> chunks,
                                                            const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                            QwenPrefillLayers layers, ops::AneFfn *aneFfn) {
  if (spilled.active()) throw std::logic_error("Nemotron-H attends no spilled K/V");
  std::vector<PrefillStep> steps = prefillSteps(graph, chunks, kv, spilled, layers, aneFfn);
  return addPrefillLayers(std::span(steps), *this, layers);
}

bool NemotronHTarget::attends(uint32_t index) const {
  return std::holds_alternative<QwenAttentionWeights>(weights_.layers[index].block);
}

const ops::BlockMoeWeights *NemotronHTarget::moe(uint32_t index) const {
  const auto *block = std::get_if<NemotronMoeWeights>(&weights_.layers[index].block);
  return block ? &block->experts : nullptr;
}

// A layer's block reads the RMS norm of its input rows and adds its output to
// them into the other buffer. The step's next Mamba-2 and attention layer are
// those of the layers before it: the shared loop counts every layer that does
// not attend a GDN layer, which an FFN-alone layer is not.
void NemotronHTarget::addPrefillLayer(PrefillStep &step, uint32_t index) const {
  const auto before = [&]<class Block>(std::type_identity<Block>) {
    return static_cast<uint32_t>(std::count_if(weights_.layers.begin(), weights_.layers.begin() + index,
                                               [](const NemotronHLayerWeights &layer) {
                                                 return std::holds_alternative<Block>(layer.block);
                                               }));
  };
  step.gdnLayer = before(std::type_identity<NemotronMambaWeights>{});
  step.attentionLayer = before(std::type_identity<QwenAttentionWeights>{});
  const NemotronHLayerWeights &layer = weights_.layers[index];
  const QwenTargetPrefillBuffers &b = step.buffers;
  const metal::MetalBuffer input = b.hidden[index & 1];
  ops::Normalization::addRms(step.graph, input, layer.norm, b.normalized, geometry_.hiddenSize, step.rows);
  std::visit([&](const auto &block) { addBlock(step, block, input, b.hidden[(index & 1) ^ 1]); }, layer.block);
}

// A MoE block's routed experts (its routing ran in addPrefillLayer), then the
// draft's captures of the layer's output: a decode step's of every row.
void NemotronHTarget::addPrefillExperts(PrefillStep &step, uint32_t index) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const metal::MetalBuffer output = b.hidden[(index & 1) ^ 1];
  if (const auto *block = std::get_if<NemotronMoeWeights>(&weights_.layers[index].block)) {
    ops::MoE::addRouted(step.graph, step.ffn, block->experts, *step.moe);
    addProjection(step, block->latentUp, rows(step).moeLatent, output, b.gdnOutput);
  }
  if (step.decode) {
    if (const auto slot = geometry_.captureSlot(index))
      addCapture(step.graph, *slot, output, 0, step.decode->capturedTargetHidden, 0, step.rows);
    return;
  }
  forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t count) {
    addCapture(step.graph, slot, output, source, b.captured, destination, count);
  });
}

metal::MetalBuffer NemotronHTarget::addPrefillEnd(PrefillStep &step) const {
  const metal::MetalBuffer hidden = step.buffers.hidden[geometry_.layers & 1];
  if (const QwenTargetDecodeBuffers *d = step.decode)
    addNormalizedHead(step.graph, weights_.finalNorm, hidden, d->finalHidden, d->logits, step.rows,
                      step.buffers.linearScratch);
  return hidden;
}

void NemotronHTarget::addProjection(PrefillStep &step, const ops::Projection &projection, metal::MetalBuffer input,
                                    metal::MetalBuffer output, metal::MetalBuffer residual) const {
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan plan =
      linear.plan(projection, step.rows, residual ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None);
  linear.add(step.graph,
             {.input = std::move(input), .output = std::move(output), .residual = std::move(residual),
              .scratch = step.buffers.linearScratch},
             projection, plan);
}

void NemotronHTarget::addBlock(PrefillStep &step, const NemotronMambaWeights &block, metal::MetalBuffer input,
                               metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Rows &r = rows(step);
  const uint32_t layer = step.gdnLayer++;
  addProjection(step, block.in, b.normalized, r.packed);
  if (const QwenTargetDecodeBuffers *d = step.decode)
    ops::Mamba2::addDecode(step.graph, r.packed, block.mixer, d->currentGdnStates, d->nextGdnStates, d->gdnTape,
                           std::span(d->gdnTapeLanes).first(d->step.lanes), b.laneRows, step.rows, r.y,
                           mamba2Shape(geometry_), layer, geometry_.stateLayout.strides());
  else
    for (const QwenTargetPrefillSequence &sequence : step.sequences) addPrefillMamba(step, block, layer, sequence);
  ops::Mamba2::addGroupNorm(step.graph, r.y, block.mixer.norm, r.gated, weights_.layout.rmsEpsilon,
                            mamba2Shape(geometry_), step.rows);
  addProjection(step, block.out, r.gated, std::move(output), std::move(input));
}

// A sequence's rows of layer `layer`, in slices of up to a verify block's
// rows that end at each state it captures: slice k writes the state before
// its rows (after slice k - 1's, which it folds from the tape) into the next
// cell, a capture's views when its first row is the capture's, else the
// one-layer cell it does not read; a last dispatch folds the last slice into
// the sequence's state out, and again into the codes of a state cached at
// the rows' end (exact: the state itself). A slice runs a verify block's rows
// from its first, so the rows past it (the next slice's, which it writes
// again, or the scratch's past the chunk) are computed and dropped.
// ponytail: a verify block's rows a dispatch and a 2K-row scratch (~230 MB);
// a chunked scan kernel when prefill speed or memory matters.
void NemotronHTarget::addPrefillMamba(PrefillStep &step, const NemotronMambaWeights &block, uint32_t layer,
                                      const QwenTargetPrefillSequence &sequence) const {
  const Rows &r = prefillRows_;
  const PrefillStates &s = prefillStates_;
  const NemotronHLayout &layout = weights_.layout;
  std::vector<uint32_t> boundaries{0};
  for (const QwenTargetStateCapture &capture : sequence.stateCaptures) {
    if (capture.row <= boundaries.back() || capture.row >= sequence.rows)
      throw std::invalid_argument("Nemotron-H prefill state captures out of order");
    boundaries.push_back(capture.row);
  }
  boundaries.push_back(sequence.rows);
  struct Slice final {
    uint32_t begin, rows;
    const QwenTargetStateCapture *capture;  // whose state the slice writes
  };
  std::vector<Slice> slices;
  for (size_t b = 0; b + 1 < boundaries.size(); ++b)
    for (uint32_t row = boundaries[b]; row < boundaries[b + 1]; row += kVerifyRows)
      slices.push_back({row, std::min(kVerifyRows, boundaries[b + 1] - row),
                        row == boundaries[b] && b ? &sequence.stateCaptures[b - 1] : nullptr});
  const uint64_t tapeBytes = mamba2TapeLayerBytes(geometry_);
  metal::MetalBuffer convolution = sequence.convolutionIn[layer], recurrent = sequence.recurrentIn[layer];
  // The one-layer cell the state is in, 2 for none of them.
  uint32_t cell = 2;
  const auto run = [&](size_t k, uint32_t row, metal::MetalBuffer convolutionOut, metal::MetalBuffer recurrentOut) {
    ops::Mamba2::addPrefillStep(step.graph, rowsOf<uint16_t>(r.packed, row, kVerifyRows, layout.mambaInProjectionRows),
                                block.mixer, {convolution, recurrent, convolutionOut, recurrentOut}, s.tape,
                                (k + 1) % 2 * tapeBytes, k % 2 * tapeBytes, k ? slices[k - 1].rows : 0,
                                rowsOf<float>(r.y, row, kVerifyRows, layout.mambaInnerSize), mamba2Shape(geometry_));
  };
  for (size_t k = 0; k <= slices.size(); ++k) {
    const bool fold = k == slices.size();
    // The fold runs past the sequence's rows, whose outputs it drops.
    const uint32_t row = sequence.rowBegin + (fold ? sequence.rows : slices[k].begin);
    metal::MetalBuffer convolutionOut, recurrentOut;
    if (fold) {
      convolutionOut = sequence.convolutionOut[layer], recurrentOut = sequence.recurrentOut[layer];
      // The cached state's, its convolution rows the sequence's (above), into
      // a cell the fold does not read.
      if (!sequence.endCodes.empty()) run(k, row, s.convolution[cell == 0 ? 1 : 0], sequence.endCodes[layer]);
    } else if (const QwenTargetStateCapture *capture = slices[k].capture) {
      convolutionOut = capture->convolution[layer], recurrentOut = capture->recurrent[layer];
    } else {
      // The cell this slice does not read.
      cell = cell == 0 ? 1 : 0;
      convolutionOut = s.convolution[cell], recurrentOut = s.recurrent[cell];
    }
    run(k, row, convolutionOut, recurrentOut);
    if (fold || slices[k].capture) cell = 2;
    convolution = convolutionOut, recurrent = recurrentOut;
  }
}

void NemotronHTarget::addBlock(PrefillStep &step, const QwenAttentionWeights &block, metal::MetalBuffer input,
                               metal::MetalBuffer output) const {
  addPrefillMixer(step, block, {}, {std::move(output), std::move(input)});
}

// The router reads the normalized rows, the routed experts run in the latent
// of the block's down projection (ops::MoE, addPrefillExperts), and the shared
// expert reads the normalized rows: output = input + shared + latent up of the
// routed experts' sum. A decode step's lanes route their rows alone.
void NemotronHTarget::addBlock(PrefillStep &step, const NemotronMoeWeights &block, metal::MetalBuffer input,
                               metal::MetalBuffer) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Rows &r = rows(step);
  addProjection(step, block.latentDown, b.normalized, r.latent);
  step.moe = operators_.moe(block.shape, geometry_.moeRouterFormat, step.rows);
  step.ffn = {.input = r.latent, .output = r.moeLatent, .scratch = b.moe, .routerInput = b.normalized};
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    step.ffn.lanes = d->step.lanes;
    step.ffn.laneRows = b.laneRows;
  }
  ops::MoE::addRouting(step.graph, step.ffn, block.experts, *step.moe);
  addProjection(step, block.sharedUp, b.normalized, r.sharedMid);
  ops::MoE::addReluSquared(step.graph, r.sharedMid, uint64_t{step.rows} * block.sharedUp.outputSize);
  addProjection(step, block.sharedDown, r.sharedMid, b.gdnOutput, std::move(input));
}

void NemotronHTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                                   metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t rows,
                                   ops::LinearScratch scratch) const {
  addNormalizedHead(graph, weights_.finalNorm, std::move(hidden), std::move(finalHidden), std::move(logits), rows,
                    scratch);
}

QwenTargetScratchBytes qwenTargetScratchBytes(const NemotronHWeights &, const QwenTargetGeometry &geometry, uint32_t,
                                              uint64_t) {
  return NemotronHTarget::scratchBytes(geometry);
}

} // namespace splash::model

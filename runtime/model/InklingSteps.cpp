#include "model/InklingSteps.hpp"

#include "Checked.hpp"
#include "metal/abi/ShortConvolution.h"
#include "ops/ShortConvolution.hpp"

#include <cstring>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace splash::model {
namespace {

constexpr uint32_t kChunkRows = ExecutionLimits::prefillTokenBudget;
// A decode step's rows as its plans store them: every lane's verify block,
// packed, for the runtime's lanes (decodeStorageRows).
uint32_t stepRows(const QwenTargetGeometry &g) noexcept { return decodeStorageRows(g.lanes); }
constexpr uint32_t kBlockLayers = InklingLayout::residualBlockLayers;

// The stream of each of a loop pass's two blocks (its first core block's,
// which the second's first layer pushes, and its output's), the passes
// taking the two pairs of states in turn.
constexpr uint32_t passBlockState(uint32_t pass) noexcept { return 2 + 2 * (pass % 2); }
constexpr uint32_t passOutputState(uint32_t pass) noexcept { return 3 + 2 * (pass % 2); }

// The state element of a history and the tape (kShortConvolutionStateValueBytes).
constexpr ops::FloatElement kStateElement =
    kShortConvolutionStateValueBytes == 2 ? ops::FloatElement::BFloat16 : ops::FloatElement::Float32;
static_assert(kShortConvolutionStateValueBytes == ops::elementBytes(kStateElement));

// A scratch tensor (InklingScratch): its views, each of `rowBytes` a row of
// a step's rows plus `fixedBytes`, and whether each chunk of a prefill
// group holds its own (the residual's states, which a layer's MoE work reads
// after the group's load of its experts, QwenTarget::addPrefillLayers, and
// which a prefill's passes carry from command to command).
struct Tensor final {
  std::span<metal::MetalBuffer> (*views)(InklingScratch &);
  uint64_t rowBytes = 0;
  uint64_t fixedBytes = 0;
  bool carried = false;
};

template <class T> constexpr uint64_t bytesFor(uint64_t elements) noexcept { return elements * sizeof(T); }

// The views of scratch member `Field`: one buffer, or an array of them.
template <auto Field> std::span<metal::MetalBuffer> viewsOf(InklingScratch &scratch) {
  auto &field = scratch.*Field;
  if constexpr (std::is_same_v<std::remove_cvref_t<decltype(field)>, metal::MetalBuffer>)
    return {&field, 1};
  else
    return field;
}

// A prefill chunk's or a decode step's tensors, in their order in its buffer.
std::vector<Tensor> tensors(const QwenTargetGeometry &g, bool decode) {
  const InklingSides sides = InklingSides::of(g);
  const uint32_t hidden = g.hiddenSize, kvWidth = g.attentionKvHeads * g.attentionHeadDimension;
  std::vector<Tensor> result{
      {&viewsOf<&InklingScratch::states>, bytesFor<float>(hidden), 0, true},
      {&viewsOf<&InklingScratch::inputs>, bytesFor<float>(hidden)},
      {&viewsOf<&InklingScratch::messages>, bytesFor<float>(hidden)},
      {&viewsOf<&InklingScratch::low>, bytesFor<float>(InklingLayout::loopGateRank)},
      {&viewsOf<&InklingScratch::keyInput>, bytesFor<uint16_t>(hidden)},
      {&viewsOf<&InklingScratch::valueInput>, bytesFor<uint16_t>(hidden)},
      {&viewsOf<&InklingScratch::querySide>, bytesFor<float>(sides.query)},
      {&viewsOf<&InklingScratch::keySide>, bytesFor<float>(sides.globalKey)},
      {&viewsOf<&InklingScratch::valueSide>, bytesFor<float>(sides.globalKey)},
      {&viewsOf<&InklingScratch::keys>, bytesFor<float>(kvWidth)},
      {&viewsOf<&InklingScratch::values>, bytesFor<float>(kvWidth)},
      {&viewsOf<&InklingScratch::relative>,
       bytesFor<float>(uint64_t{g.attentionQueryHeads} * g.relativeDimension)},
      {&viewsOf<&InklingScratch::zeros>, bytesFor<uint16_t>(hidden)},
  };
  if (decode)
    result.push_back({&viewsOf<&InklingScratch::tape>, 0,
                      uint64_t{g.layers} * stepRows(g) * g.shortConvolutionChannels() *
                          kShortConvolutionStateValueBytes});
  return result;
}

uint64_t viewBytes(const Tensor &tensor, uint32_t rows) noexcept { return tensor.rowBytes * rows + tensor.fixedBytes; }

// The bytes of a buffer of `rows` rows whose carried tensors it holds
// `chunks` times; and its views of chunk `chunk`.
uint64_t bufferBytes(const QwenTargetGeometry &g, bool decode, uint32_t rows, uint32_t chunks) {
  InklingScratch counted;
  uint64_t total = 0;
  for (const Tensor &tensor : tensors(g, decode))
    total += alignUp(viewBytes(tensor, rows)) * tensor.views(counted).size() * (tensor.carried ? chunks : 1);
  return total;
}

} // namespace

InklingLayerStates inklingLayerStates(const QwenTargetDimensions &g, uint32_t layer) {
  static_assert(kInklingLoop.loopStart == kBlockLayers && kInklingLoop.loopEnd == 3 * kBlockLayers &&
                    kInklingLoop.loopPasses == 3 && kInklingLoop.weightLayers() == 4 * kBlockLayers,
                "Inkling's states are those of a prelude block, two loop blocks of three passes and a coda block");
  if (layer >= g.layers || g.loopStart != kInklingLoop.loopStart || g.loopEnd != kInklingLoop.loopEnd ||
      g.loopPasses != kInklingLoop.loopPasses || g.layers != kInklingLoop.layers)
    throw std::invalid_argument("not a layer of Inkling's loop");
  const uint32_t weight = g.weightLayer(layer), block = weight / kBlockLayers;
  // The loop's pass of a core layer; the coda's states are those the last
  // pass left.
  const uint32_t pass = weight < g.loopEnd ? g.loopPass(layer) : g.loopPasses - 1;
  // The stream of each block, which its first layer pushes to the blocks
  // as the next block's first layer.
  const std::array<uint32_t, 4> streams{kInklingPreludeState, passBlockState(pass), passOutputState(pass),
                                        passBlockState(g.loopPasses)};
  InklingLayerStates states;
  states.blocks = {kInklingEmbeddingState, kInklingPreludeState, passBlockState(pass), passOutputState(pass)};
  states.output = streams[block];
  states.input = weight % kBlockLayers ? states.output : block ? streams[block - 1] : kInklingEmbeddingState;
  states.ffnBlocks = block + 1;
  states.attentionBlocks = weight % kBlockLayers ? states.ffnBlocks : block;
  if (g.loopPass(layer)) {
    states.messages = true;
    states.bank = {passBlockState(pass - 1), passOutputState(pass - 1)};
  }
  return states;
}

InklingSteps::InklingSteps(const QwenTargetGeometry &geometry, metal::MetalBackend &backend, uint32_t prefillChunks)
    : geometry_(geometry), backend_(backend), prefillChunks_(prefillChunks) {
  if (!prefillChunks_ || !geometry_.shortConvolutionTaps ||
      geometry_.shortConvolutionTaps != SHORT_CONVOLUTION_TAPS || geometry_.layers != kInklingLoop.layers)
    throw std::invalid_argument("invalid Inkling step geometry");
  prefill_ = backend_.allocateBuffer(bufferBytes(geometry_, false, kChunkRows, prefillChunks_),
                                     metal::BufferStorage::Shared, "inkling-prefill-scratch");
  decode_ = backend_.allocateBuffer(bufferBytes(geometry_, true, stepRows(geometry_), 1), metal::BufferStorage::Shared,
                                    "inkling-decode-scratch");
  // The rows the MoE adds a layer's output to.
  for (const metal::MetalBuffer &zeros : {prefillScratch(0).zeros, decodeScratch().zeros})
    std::memset(zeros.contents(), 0, zeros.sizeBytes());
}

QwenTargetScratchBytes InklingSteps::scratchBytes(const QwenTargetGeometry &geometry, uint32_t prefillChunks) {
  return {bufferBytes(geometry, false, kChunkRows, prefillChunks), bufferBytes(geometry, true, stepRows(geometry), 1)};
}

QwenTargetScratchBytes InklingSteps::allocatedBytes() const noexcept {
  return {prefill_.sizeBytes(), decode_.sizeBytes()};
}

InklingScratch InklingSteps::prefillScratch(uint32_t index) const {
  if (index >= prefillChunks_) throw std::out_of_range("Inkling prefill chunk");
  InklingScratch scratch{.rows = kChunkRows};
  uint64_t cursor = 0;
  for (const Tensor &tensor : tensors(geometry_, false)) {
    const uint64_t bytes = viewBytes(tensor, kChunkRows);
    const std::span<metal::MetalBuffer> views = tensor.views(scratch);
    if (tensor.carried) cursor += alignUp(bytes) * views.size() * index;
    for (metal::MetalBuffer &view : views) {
      view = backend_.view(prefill_, cursor, bytes);
      cursor += alignUp(bytes);
    }
    if (tensor.carried) cursor += alignUp(bytes) * views.size() * (prefillChunks_ - 1 - index);
  }
  return scratch;
}

InklingScratch InklingSteps::decodeScratch() const {
  InklingScratch scratch{.rows = stepRows(geometry_)};
  uint64_t cursor = 0;
  for (const Tensor &tensor : tensors(geometry_, true)) {
    const uint64_t bytes = viewBytes(tensor, scratch.rows);
    for (metal::MetalBuffer &view : tensor.views(scratch)) {
      view = backend_.view(decode_, cursor, bytes);
      cursor += alignUp(bytes);
    }
  }
  return scratch;
}

uint32_t InklingSteps::channels(InklingConvolution convolution) const noexcept {
  const uint32_t kvWidth = geometry_.attentionKvHeads * geometry_.attentionHeadDimension;
  return convolution == InklingConvolution::Key || convolution == InklingConvolution::Value ? kvWidth
                                                                                           : geometry_.hiddenSize;
}

uint32_t InklingSteps::channelOffset(InklingConvolution convolution) const noexcept {
  const uint32_t kvWidth = geometry_.attentionKvHeads * geometry_.attentionHeadDimension;
  switch (convolution) {
  case InklingConvolution::Key: return 0;
  case InklingConvolution::Value: return kvWidth;
  case InklingConvolution::Attention: return 2 * kvWidth;
  case InklingConvolution::Ffn: return 2 * kvWidth + geometry_.hiddenSize;
  }
  return 0;
}

// A cell's auxiliary state holds the convolution histories after any PLE
// history and QSA pending keys (GdnStateLayout::auxiliaryBytes, none of which
// Inkling has): each layer's HISTORY rows of its channels, layer after layer.
metal::MetalBuffer InklingSteps::history(const metal::MetalBuffer &auxiliary, uint32_t layer,
                                         InklingConvolution convolution) const {
  if (layer >= geometry_.layers) throw std::out_of_range("Inkling convolution history layer");
  const uint64_t base = geometry_.pleHistoryBytes() + geometry_.indexerPendingBytes();
  const uint64_t layerElements = uint64_t{SHORT_CONVOLUTION_HISTORY} * historyStride();
  return backend_.view(auxiliary,
                       base + (layer * layerElements + channelOffset(convolution)) * kShortConvolutionStateValueBytes,
                       ops::shortConvolutionHistoryBytes(kStateElement, channels(convolution), historyStride()));
}

metal::MetalBuffer InklingSteps::histories(const metal::MetalBuffer &auxiliary) const {
  return backend_.view(auxiliary, geometry_.pleHistoryBytes() + geometry_.indexerPendingBytes(),
                       geometry_.shortConvolutionStateBytes());
}

ops::StridedRows InklingSteps::tape(const InklingScratch &scratch, uint32_t layer,
                                    InklingConvolution convolution) const {
  if (!scratch.tape || layer >= geometry_.layers) throw std::logic_error("an Inkling tape is a decode step's");
  const uint64_t layerElements = uint64_t{scratch.rows} * historyStride();
  const uint64_t offset = (layer * layerElements + channelOffset(convolution)) * kShortConvolutionStateValueBytes;
  return {backend_.view(scratch.tape, offset, scratch.tape.sizeBytes() - offset), historyStride()};
}

} // namespace splash::model

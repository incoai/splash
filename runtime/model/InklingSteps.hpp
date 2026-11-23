#pragma once

// What Inkling 2 Mini (model/Inkling.hpp) needs of a step beyond its weights
// and the buffers every Qwen target binds, which its target keeps
// (model/InklingTarget.hpp): the fp32 states of its residual (the
// embedding's rows and the streams of its blocks, which the depth attention
// mixes), its layers' depth mixes, loop messages and gates, the bf16 inputs
// of its key and value sides, its attention's sides, convolved keys and
// values and relative states, the zero rows its MoE adds its output to, and a
// decode step's convolution tape; where each of its 48 layers reads and
// writes its residual's states (inklingLayerStates); and where a state
// cell keeps each layer's convolution histories.

#include "model/Inkling.hpp"
#include "model/QwenTarget.hpp"
#include "ops/FloatRows.hpp"

#include <array>
#include <cstdint>

namespace splash::model {

// The residual's states a step keeps (InklingScratch::states), each a row
// of hiddenSize fp32 values per step row: the embedding's rows x; the
// prelude's stream, whose output every pass of the loop starts from; and two
// pairs of a pass's streams, its first core block's and its second's, which
// the passes take in turn, so that a pass's messages read the pass before's
// while it writes its own. The coda's stream takes the pair the last pass
// left. A layer's stream holds its input until its attention's output
// replaces or adds to it, and then its output.
inline constexpr uint32_t kInklingEmbeddingState = 0;
inline constexpr uint32_t kInklingPreludeState = 1;
inline constexpr uint32_t kInklingStates = 6;

// A layer's states (inklingLayerStates): its input's, and its output's,
// which a layer that starts a block of the depth attention writes anew
// (its input joins the blocks) and any other adds to in place; the states of
// the completed blocks, in order, of which its attention's depth mix reads
// the first attentionBlocks and its FFN's the first ffnBlocks (a
// block-starting layer's input among them), each beside the stream; and on a
// pass of the loop after its first, the states of the pass before that its
// messages mix (its first core block's and its output).
struct InklingLayerStates final {
  uint32_t input = 0;
  uint32_t output = 0;
  std::array<uint32_t, 4> blocks{};
  uint32_t attentionBlocks = 0;
  uint32_t ffnBlocks = 0;
  std::array<uint32_t, 2> bank{};
  bool messages = false;
  [[nodiscard]] constexpr bool startsBlock() const noexcept { return input != output; }
  bool operator==(const InklingLayerStates &) const = default;
};

// The states of layer `layer` (0-47) of Inkling's run order: blocks of
// InklingLayout::residualBlockLayers weight layers, the prelude's one, two of
// the loop's per pass, and the coda's; a block's first layer pushes its input
// to the blocks, as the model's blocks list grows.
[[nodiscard]] InklingLayerStates inklingLayerStates(const QwenTargetDimensions &dimensions, uint32_t layer);
// The states the head's depth mix reads: every block's and the coda's
// stream, which holds the last layer's output.
inline constexpr std::array<uint32_t, 5> kInklingFinalStates{0, 1, 2, 3, 4};

// The convolutions of a layer, in the order of their channels in a layer's
// history: its keys, its values, its attention's output and its MoE's
// output.
enum class InklingConvolution : uint8_t { Key, Value, Attention, Ffn };

// A step's scratch: views of its rows, `rows` rows of each, a prefill
// chunk's or a decode step's most.
//  - states: the residual's states (kInklingStates), fp32 [rows][hidden];
//  - inputs: a layer's attention inputs' depth mixes (query, key and value
//    on a global layer; one of a sliding layer) and its FFN input's, fp32
//    [rows][hidden];
//  - messages: a pass's messages to the query, key, value and FFN inputs,
//    fp32 [rows][hidden], over which a sliding layer's gates write its
//    query, key and value inputs; low: each gate's rank values, fp32
//    [rows][loopGateRank];
//  - keyInput, valueInput: the bf16 [rows][hidden] key and value inputs
//    after their norms, where they differ from the query's (the chunk's
//    normalized rows);
//  - querySide, keySide, valueSide: the sides' fp32 rows (InklingSides,
//    the key and value sides globalKey wide); keys and values: the
//    convolved keys and values, fp32 [rows][kvHeads * 128]; relative: the
//    relative states, fp32 [rows][heads][relativeDimension];
//  - zeros: bf16 [rows][hidden] zeros, the residual the MoE adds its output
//    to;
//  - tape: a decode step's convolution inputs of every layer, as a state's
//    history keeps them, layer l's rows from l * rows rows on, each
//    shortConvolutionChannels() wide, its convolutions' inputs side by side
//    as a history keeps them; none in a prefill's.
struct InklingScratch final {
  uint32_t rows = 0;
  std::array<metal::MetalBuffer, kInklingStates> states;
  std::array<metal::MetalBuffer, 3> inputs;
  std::array<metal::MetalBuffer, InklingLayout::loopMessages> messages;
  std::array<metal::MetalBuffer, InklingLayout::loopMessages> low;
  metal::MetalBuffer keyInput;
  metal::MetalBuffer valueInput;
  metal::MetalBuffer querySide;
  metal::MetalBuffer keySide;
  metal::MetalBuffer valueSide;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer relative;
  metal::MetalBuffer zeros;
  metal::MetalBuffer tape;
};

class InklingSteps final {
public:
  // Scratch for prefill commands of up to `prefillChunks` chunks.
  InklingSteps(const QwenTargetGeometry &geometry, metal::MetalBackend &backend, uint32_t prefillChunks);
  [[nodiscard]] static QwenTargetScratchBytes scratchBytes(const QwenTargetGeometry &geometry,
                                                           uint32_t prefillChunks);
  // What the constructor allocated, scratchBytes' of its geometry.
  [[nodiscard]] QwenTargetScratchBytes allocatedBytes() const noexcept;

  // Chunk `index`'s scratch of a prefill command: its own residual states,
  // which its layers carry from one to the next, and the scratch a layer's
  // work shares with the other chunks'.
  [[nodiscard]] InklingScratch prefillScratch(uint32_t index) const;
  // A decode step's scratch, of its most rows.
  [[nodiscard]] InklingScratch decodeScratch() const;

  // A convolution's history of layer `layer` in a state cell's auxiliary
  // state (GdnStateLayout::auxiliaryBytes): its HISTORY rows of its channels,
  // historyStride() elements apart; and every layer's histories, for a
  // commit's units.
  [[nodiscard]] metal::MetalBuffer history(const metal::MetalBuffer &auxiliary, uint32_t layer,
                                           InklingConvolution convolution) const;
  [[nodiscard]] metal::MetalBuffer histories(const metal::MetalBuffer &auxiliary) const;
  // A decode step's rows of a layer's convolution on its tape, from row 0.
  [[nodiscard]] ops::StridedRows tape(const InklingScratch &scratch, uint32_t layer,
                                      InklingConvolution convolution) const;
  // The elements from one history row (or tape row) to the next: a layer's
  // channels of all its convolutions.
  [[nodiscard]] uint32_t historyStride() const noexcept { return geometry_.shortConvolutionChannels(); }
  // The channels of a convolution and its first in a history row.
  [[nodiscard]] uint32_t channels(InklingConvolution convolution) const noexcept;
  [[nodiscard]] uint32_t channelOffset(InklingConvolution convolution) const noexcept;

private:
  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  uint32_t prefillChunks_ = 0;
  metal::MetalBuffer prefill_;
  metal::MetalBuffer decode_;
};

} // namespace splash::model

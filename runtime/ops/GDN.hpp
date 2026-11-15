#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/LaneRows.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>
#include <string_view>

namespace splash::ops {

// Tensor geometry mapped to a compiled Metal variant during graph construction.
struct GdnShape final {
  uint32_t keyHeads = 0;
  uint32_t valueHeads = 0;
  uint32_t headDimension = 0;
  uint32_t convolutionDimension = 0;
  uint32_t packedWidth = 0;
  // Gates the normalized output by sigmoid(z) (Qwen3.8-Flash-Next), not silu(z).
  bool sigmoidGate = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return keyHeads && valueHeads && valueHeads % keyHeads == 0 &&
           headDimension && convolutionDimension && packedWidth &&
           convolutionDimension ==
               (uint64_t{2} * keyHeads + valueHeads) * headDimension &&
           packedWidth >= convolutionDimension +
                              uint64_t{valueHeads} * headDimension +
                              uint64_t{2} * valueHeads;
  }

  bool operator==(const GdnShape &) const = default;
};

// The value-head order of the GDN output, the out_proj input columns. Grouped
// keeps a key head's value heads adjacent; Tiled is llama.cpp's GGUF order,
// value head h at (h % heads per key) * key heads + h / heads per key.
enum class GdnHeadOrder : uint8_t { Grouped, Tiled };

// The recurrent state's storage, selected once for a runtime and every state
// it holds: fp32, or bf16 at half the bytes. The kernels compute in fp32 with
// either; they widen a bf16 state where they load it and round it to nearest
// even where they store it.
enum class GdnStateFormat : uint32_t { Float32 = 1, BFloat16 = 2 };

[[nodiscard]] constexpr bool validFormat(GdnStateFormat format) noexcept {
  return format == GdnStateFormat::Float32 || format == GdnStateFormat::BFloat16;
}

[[nodiscard]] constexpr std::string_view formatName(GdnStateFormat format) noexcept {
  switch (format) {
  case GdnStateFormat::Float32: return "fp32";
  case GdnStateFormat::BFloat16: return "bf16";
  }
  return "invalid";
}

[[nodiscard]] constexpr uint32_t elementBytes(GdnStateFormat format) noexcept {
  return format == GdnStateFormat::BFloat16 ? 2 : 4;
}

struct GdnStateStrides final {
  uint64_t convolutionLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convolutionStateBytes = 0;
  GdnStateFormat format{};

  [[nodiscard]] constexpr bool valid() const noexcept {
    return convolutionLayerBytes && recurrentLayerBytes &&
           convolutionStateBytes && validFormat(format);
  }
};

struct GdnPrefillBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer convolutionIn;
  metal::MetalBuffer convolutionOut;
  metal::MetalBuffer queries;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer decayWeights;
  metal::MetalBuffer timeBias;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  metal::MetalBuffer recurrentIn;
  metal::MetalBuffer recurrentOut;
  metal::MetalBuffer recurrentRows;
  NormWeights mixerNorm;
  metal::MetalBuffer hidden;
};

// A state a prefill captures inside its rows: the convolution carry and the
// recurrent state after its first `tokens` rows, which it stores in these
// buffers as a prefill ending there would.
struct GdnPrefillCapture final {
  uint32_t tokens = 0;
  metal::MetalBuffer convolution;
  metal::MetalBuffer recurrent;
};

// A lane's decode tape (metal/abi/GDN.h): each step leaves its rows'
// convolution inputs, k, v and gates there, and the lane's next step folds
// the rows it retained into the state before its own, so the state cell a
// step writes holds the state before the step's rows. A slot holds one step
// of every layer, gdnTapeLayerBytes(shape) apart; the offsets are in
// GdnDecodeBuffers::tape.
struct GdnTapeLane final {
  uint64_t pendingSlot = 0; // the lane's previous step's
  uint64_t stepSlot = 0;    // this step's
  // The previous step's rows the lane retained, which the state does not
  // hold yet: 0 after a prefill or a restore.
  uint32_t pendingRows = 0;
};

[[nodiscard]] uint64_t gdnTapeLayerBytes(const GdnShape &shape) noexcept;

// A lane's block of a decode step's rows (its LaneRows: `rows` rows, 1 to
// SPLASH_TARGET_VERIFY_ROWS, from row `first` of the step's packed, hidden
// and table rows on), and the lane's tape, whose first rows they are.
struct GdnDecodeBlock final : LaneRows {
  GdnTapeLane tape;
};

struct GdnDecodeBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer convolutionWeights;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
  metal::MetalBuffer tape;
  // Each lane's block, ascending in the step's rows, which do not overlap.
  std::span<const GdnDecodeBlock> blocks;
  metal::MetalBuffer decayWeights;
  metal::MetalBuffer timeBias;
  NormWeights mixerNorm;
  metal::MetalBuffer hidden;
  // Each lane's verify rows (ops::DraftSelector::addVerifyRows): its block's
  // rows past them get zero recurrent output.
  metal::MetalBuffer verifyRows;
  LinearScratch linearScratch{};
};

// A lane's decode step to fold rows of alone (GDN::addFold): the cell the
// step wrote, which holds the state before its rows, the tape its rows are
// on (every layer's, from stepSlot), the count of them the lane retained,
// which the step's acceptance wrote, and the cell the fold writes, of the
// same strides.
struct GdnFoldBuffers final {
  metal::MetalBuffer state;
  metal::MetalBuffer tape;
  uint64_t stepSlot = 0;
  metal::MetalBuffer retained;
  metal::MetalBuffer fold;
};

class GDN final {
public:
  // Each capture, ascending inside the rows, ends one run of the prepare and
  // scan dispatches, and the next run starts from the state it stored; the
  // gate then runs once over every row. The scan is sequential, so with an
  // fp32 state the runs compute what one run does, and a bf16 state is
  // rounded at each capture as a prefill ending there rounds it.
  static void addPrefill(metal::CommandGraph &graph, const metal::MetalBackend &backend,
                         GdnPrefillBuffers buffers, GdnShape shape, uint32_t tokens,
                         GdnStateFormat state, GdnHeadOrder order,
                         std::span<const GdnPrefillCapture> captures = {});
  // The step of buffers.blocks' lanes, 1 to SPLASH_MAXIMUM_BATCH_WIDTH.
  // Also writes the out-projection's `input` table into
  // buffers.linearScratch when it is not Plain, the table rows of the blocks'
  // rows, and throws when the scratch cannot hold it.
  static PreparedInput addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                                 GdnShape shape, uint32_t layer,
                                 GdnStateStrides state,
                                 GdnHeadOrder order,
                                 LinearInput input);
  // The state after the first `rows` of a decode step's rows, in each of
  // `layers` layers: those rows folded into the state before them, and the
  // convolution rows they leave, in buffers.fold, bit for bit the state the
  // lane's next step holds when the lane retained exactly those rows. A lane
  // that retained fewer leaves the fold as it was.
  static void addFold(metal::CommandGraph &graph, GdnFoldBuffers buffers, GdnShape shape,
                      uint32_t layers, uint32_t rows, GdnStateStrides state);
};

} // namespace splash::ops

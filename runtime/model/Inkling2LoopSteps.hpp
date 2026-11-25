#pragma once

// What Inkling-2-Mini (model/Inkling2Loop.hpp) needs of a step beyond its
// weights and the buffers every Qwen target binds, which its target keeps
// (model/Inkling2LoopTarget.hpp): the order of its layer applications and
// where each reads and writes its rows (Inkling2LoopSchedule, pure), and the
// scratch of its block states, the pass before's states, its residual inside
// a layer, its mixes, messages and gates, its attention's k and v sides and,
// in decode, every convolution's input rows, which the commit reads
// (Inkling2LoopScratch).

#include "model/Inkling2Loop.hpp"
#include "model/QwenTarget.hpp"

#include <cstdint>
#include <vector>

namespace splash::model {

// The scratch of a step's rows (a prefill chunk's or a decode step's), bf16
// rows of hiddenSize unless said: the block stack's planes and the pass
// before's (planes of a step's most rows each); the residual inside a layer
// (fp32), which the FFN's routed work reads after the layer's load; the
// mixes' outputs, up to three queries a row (fp32, [rows][3][hiddenSize]),
// and the messages, four a row (fp32, [rows][4][hiddenSize]); the k and v
// inputs (the q input is buffers.normalized); the key and value projections
// and their convolved rows (kvWidth each), a decode step's of one row side by
// side (2 kvWidth, keys then values: model/Inkling2Loop.hpp
// keyValueProjection), and a full layer's dynamic taps of each (KV heads x
// taps a row); a prompt chunk's relative states and row
// sinks the attention's projection stages for its split (fp32, in the
// queries' staged rows); a message gate's rank values (loopGateRank, three
// gates' in a row, up to ops::MessageGate::kFusedRows rows), forget and gain
// rows (fp32; a decode step's forget rows three fused gates' gated rows and
// its gain rows their partial sums; a prompt chunk's forget, gain and
// concatenation rows its three tiled gates' gated rows, fp32 of hiddenSize
// each, with gateConcat empty in decode and sharing bytes with gateLow in
// prefill), and the fused gates' counters (ops::MessageGate::kCountersBytes,
// zero between calls); zero rows, which the MoE adds its weighted sum to
// (ops::MoeBuffers::residual); and a decode step's convolution inputs of every
// application, rows of stateColumns() values, which its commit reads, and its
// relative attention's counters (ops::RelativeAttention::kCountersBytes, zero
// between layers). Non-overlapping mix/gate and attention tensors share one
// page-aligned transient region past the persistent tensors.
struct Inkling2LoopScratch final {
  metal::MetalBuffer stack;
  metal::MetalBuffer previous;
  metal::MetalBuffer residual;
  metal::MetalBuffer mixed;
  metal::MetalBuffer messages;
  metal::MetalBuffer keyInput;
  metal::MetalBuffer valueInput;
  metal::MetalBuffer keys;
  metal::MetalBuffer values;
  metal::MetalBuffer keyValues;
  metal::MetalBuffer keyTaps;
  metal::MetalBuffer valueTaps;
  metal::MetalBuffer convolvedKeys;
  metal::MetalBuffer convolvedValues;
  metal::MetalBuffer relativeStates;
  metal::MetalBuffer rowSinks;
  metal::MetalBuffer gateConcat;
  metal::MetalBuffer gateLow;
  metal::MetalBuffer gateForget;
  metal::MetalBuffer gateGain;
  metal::MetalBuffer gateCounters;
  metal::MetalBuffer zeros;
  metal::MetalBuffer tape;
  metal::MetalBuffer attentionCounters;
  // The rows each plane of `stack` and `previous` holds.
  uint32_t planeRows = 0;
};

class Inkling2LoopSteps final {
public:
  // Scratch for prefill commands of up to `prefillChunks` chunks of a
  // target of `geometry` whose loop is `shape`.
  Inkling2LoopSteps(const QwenTargetGeometry &geometry, const Inkling2LoopShape &shape, metal::MetalBackend &backend,
                    uint32_t prefillChunks);
  [[nodiscard]] static QwenTargetScratchBytes scratchBytes(const QwenTargetGeometry &geometry,
                                                           const Inkling2LoopShape &shape, uint32_t prefillChunks);
  [[nodiscard]] QwenTargetScratchBytes allocatedBytes() const noexcept;

  // Chunk `index`'s scratch of a prefill command.
  [[nodiscard]] Inkling2LoopScratch prefillScratch(uint32_t index) const;
  // The scratch of a decode step, at its most rows.
  [[nodiscard]] Inkling2LoopScratch verifyScratch() const;

  // A tensor placed in its buffer.
  struct Placed final {
    metal::MetalBuffer Inkling2LoopScratch::*field;
    uint64_t rowBytes;
    uint64_t fixedBytes;
    bool carried;
    uint64_t offset;
  };

private:
  QwenTargetGeometry geometry_;
  Inkling2LoopShape shape_;
  metal::MetalBackend &backend_;
  uint32_t prefillChunks_ = 0;
  metal::MetalBuffer prefill_;
  metal::MetalBuffer decode_;
  std::vector<Placed> prefillTensors_;
  std::vector<Placed> decodeTensors_;
};

} // namespace splash::model

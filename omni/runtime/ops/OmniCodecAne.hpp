#pragma once

// The speech tokenizer decoder's convolution stack on the Neural Engine: the
// initial convolution, the four SnakeBeta / transposed-convolution /
// residual-unit blocks, the output SnakeBeta and the output convolution
// (about 95% of the decoder's FLOPs), as one MIL program of fp16 constants
// with a function per chunk size, which runs beside the GPU's decode steps.
// The GPU runs the decoder's front (RVQ lookup through the ConvNeXt upsample
// stages, CodecDecoder::addFrontStream) and writes its rows into a lane's
// input surface (addInput); an evaluation waits for the GPU's signal on a
// shared event, reads the lane's histories of every causal convolution and
// writes the samples and the updated histories, in ping-pong halves the
// lane's chunks alternate between, so a stream of chunks decodes as the
// GPU's stream does, up to fp16.
//
// Surfaces hold fp16 channel-major tensors, [channels][columns] with each
// channel's columns contiguous (the MIL [1, C, 1, T] layout): the input
// [latent][rows], the samples [1][samples] and each history [channels]
// [length].

#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/OmniCodec.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace splash::ops::omni {

class CodecAneStack final {
public:
  // A program of one function per chunk size of `functionFrames` (ascending,
  // at most the decoder's maximumFrames), with the surfaces of `lanes`
  // streams. Compiles (or loads the service's compilation of) the program
  // under `limits`; throws if the Neural Engine is unavailable or refuses it.
  CodecAneStack(metal::MetalBackend &backend, const CodecDecoderWeights &weights, std::vector<uint32_t> functionFrames,
                uint32_t lanes, ane::Program::Limits limits);
  ~CodecAneStack();
  CodecAneStack(const CodecAneStack &) = delete;
  CodecAneStack &operator=(const CodecAneStack &) = delete;

  [[nodiscard]] std::span<const uint32_t> functionFrames() const noexcept { return functionFrames_; }
  // The smallest function's frames that hold `frames`; 0 if none does.
  [[nodiscard]] uint32_t functionFor(uint32_t frames) const noexcept;
  [[nodiscard]] uint32_t lanes() const noexcept { return static_cast<uint32_t>(lanes_.size()); }
  // The surfaces' bytes.
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }
  // The program's MIL source (for tools and tests).
  [[nodiscard]] const std::string &source() const noexcept { return source_; }

  // Clears a lane's histories for a new stream; no evaluation of the lane
  // may be in flight.
  void reset(uint32_t lane);
  // GPU: `rows` (fp32 [frames * 4][latent], CodecDecoder::stackInput) into
  // the lane's input surface, the columns up to `function` frames' zero.
  void addInput(metal::CommandGraph &graph, uint32_t lane, const CodecRows &rows, uint32_t function);
  // Queues the lane's evaluation of the function of `function` frames
  // (Program::enqueue): it starts once `event` reaches `wait`, raises it to
  // `signal` when done, and reports through `done`. Throws if it cannot be
  // queued. The lane's input must hold the chunk once the event reaches
  // `wait`; its samples are valid once `done(true)` ran, until its next
  // evaluation is queued.
  void enqueue(uint32_t lane, uint32_t function, const metal::SharedEvent &event, uint64_t wait, uint64_t signal,
               std::function<void(bool)> done);
  // After an evaluation of the lane succeeded: its next chunk reads the
  // histories that evaluation wrote.
  void advance(uint32_t lane);
  // The leading out.size() samples of the lane's last evaluation, of the
  // function of `function` frames.
  void samples(uint32_t lane, uint32_t function, std::span<float> out) const;

  // GPU: the lane's current histories into the stream's stack histories at
  // `phase` (fp32 rows), so the GPU continues the stream (CodecDecoder::
  // addStackStream); and the lane's input columns back into `rows`.
  void addHistoriesToStream(metal::CommandGraph &graph, uint32_t lane, const CodecStream &stream, uint32_t phase) const;
  void addInputToRows(metal::CommandGraph &graph, uint32_t lane, const CodecRows &rows) const;
  // GPU: the stream's stack histories at `phase` into the lane's current
  // histories, so the Neural Engine continues a stream the GPU began.
  void addHistoriesFromStream(metal::CommandGraph &graph, uint32_t lane, const CodecStream &stream,
                              uint32_t phase) const;

private:
  struct Layers;
  struct Lane;
  // How a function's late blocks run in pieces: the blocks from `split` (the
  // first whose output is wider than kPieceColumns columns) on, the output
  // SnakeBeta and convolution run per piece of the split block's input
  // columns, `pieces` equal parts of pieceSamples samples each, which are the
  // samples surface's rows. The Neural Engine's compiler refuses a function
  // with tensors past 32768 columns (18 frames' samples), and pieces of about
  // 4096 samples evaluate fastest (25 frames: 21.2 ms whole to 32768 columns,
  // 16.6 ms in pieces of at most 4096, 18.6 ms at 8192, 20.0 ms at 2048).
  struct Plan final {
    uint32_t split = 0, pieces = 1, samplesPerColumn = 1, pieceSamples = 0;
  };
  static constexpr uint32_t kPieceColumns = 4096;
  [[nodiscard]] Plan plan(uint32_t frames) const;
  [[nodiscard]] std::string function(uint32_t frames) const;

  metal::MetalBackend &backend_;
  std::vector<uint32_t> functionFrames_;
  std::unique_ptr<Layers> layers_;
  std::vector<std::unique_ptr<Lane>> lanes_;
  std::string source_;
  std::unique_ptr<ane::Program> program_;
  uint64_t bytes_ = 0;
};

} // namespace splash::ops::omni

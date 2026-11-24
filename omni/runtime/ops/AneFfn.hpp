#pragma once

#include "ane/Handoff.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Linear.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace splash::ops {

// The SwiGLU FFN of a prefill chunk split by intermediate channel between the
// GPU and the Neural Engine (ported from Splash for the Omni decoders' affine
// Q4 layers: hidden 2048, intermediate 6144). The GPU runs the leading
// channels with its prefill kernels over views of the projections' planes:
// gate and up over their leading rows, down over the leading inputs of its
// rows. The ANE runs the rest as one W8A8 program over the chunk's rows:
// Hadamard-rotated int8 activations and per-row int8 weights, which the GPU
// rotates and quantizes from the Q4 planes once, when the split is built,
// into surfaces of every layer's own (resident; about 0.5 GB at half the
// channels of 28 layers). The GPU adds the ANE's partial down projection to
// its own. A shared event orders each layer's evaluation between the GPU's
// packing of its inputs and that join, inside the one prefill command
// (ane::Handoff).
//
// Any failure of the ANE's work stops the split for the life of the process:
// finish() then reports the command's results unusable, and the caller runs
// the chunk again on the GPU alone, which every later chunk runs too.
class AneFfn final {
public:
  // The intermediate channels a split moves in: whole blocks of the ANE's
  // intermediate rotation, whole 256-row tiles of the weight planes for the
  // GPU.
  static constexpr uint32_t kChannelUnit = 512;
  // The most rows of a chunk, which every surface holds.
  static constexpr uint32_t kMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;

  // The ANE takes `aneUnits` of the units() of intermediate channels, in a
  // program with a function of each of `functionRows` rows (ascending,
  // multiples of 32 up to kMaximumRows): a chunk runs the smallest that holds
  // it, from functionRows.front() rows. The program is compiled and loaded
  // here; `interrupted` returning true ends the wait for the ANE's service
  // with ane::Interrupted. Throws if the layers are not ones the split takes
  // (unsupported()) or the program does not come up.
  AneFfn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, uint32_t aneUnits,
         std::span<const uint32_t> functionRows, std::function<bool()> interrupted = {});
  AneFfn(const AneFfn &) = delete;
  AneFfn &operator=(const AneFfn &) = delete;

  // Why the split does not take `layers`, or null when it does: affine Q4
  // projections of one shape whose hidden size ane_ffn_rotate takes.
  [[nodiscard]] static const char *unsupported(std::span<const SwiGluProjections> layers);
  // The units of intermediate channels of `layers`: the ANE takes from one
  // to all but one of them.
  [[nodiscard]] static uint32_t units(std::span<const SwiGluProjections> layers);

  // The fraction of intermediate channels the ANE takes, and how many.
  [[nodiscard]] double share() const noexcept;
  [[nodiscard]] uint32_t aneChannels() const noexcept { return shape_.ane; }
  // What Metal allocated for the split: the resident int8 weights and the
  // chunk's surfaces.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }
  // The least rows of a chunk the split takes, up to maximumRows()
  // (functionRows.front() until set): a chunk runs the smallest function
  // that holds it, padded to its rows.
  [[nodiscard]] uint32_t minimumRows() const noexcept { return minimumRows_; }
  void setMinimumRows(uint32_t rows);
  // The rows of its largest function, the most a chunk it takes may have.
  [[nodiscard]] uint32_t maximumRows() const noexcept { return evaluations_.back().rows; }

  // Whether the split takes a chunk of `rows` rows: from minimumRows() to
  // maximumRows(), until it stops.
  [[nodiscard]] bool splits(uint32_t rows) const;
  // Whether the split stopped, and why.
  [[nodiscard]] bool retired() const { return handoff_.retired(); }
  [[nodiscard]] std::string reason() const { return handoff_.reason(); }

  // Starts encoding a command; false once the split stopped.
  bool begin();
  // Layer `layer`'s FFN of a chunk of `rows` rows, which splits() takes,
  // encoded in layer order from layer 0 after begin(): output = residual +
  // FFN of ffn.normalized, whose Q4 sums the norm wrote.
  void add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
           metal::MetalBuffer output, uint32_t rows);
  // Starts the ANE's evaluations of the command encoded since begin(), then
  // submits it (MetalBackend::submitCommandAsync). A submission that throws
  // stops the split before the exception goes on.
  [[nodiscard]] metal::CommandTicket commit(const metal::CommandGraph &graph, metal::CommandCompletion completion = {});
  // After the command committed last has completed: whether its outputs hold
  // the split's FFN. False when its ANE work failed or wrote values that are
  // not finite, which stops the split and logs why, once.
  [[nodiscard]] bool finish();
  // The Neural Engine's milliseconds over the evaluations of the command
  // finish() found usable last (ane::Handoff::finish).
  [[nodiscard]] double lastAneMilliseconds() const noexcept { return lastAneMilliseconds_; }

  // Checks the split against the GPU alone on `layers`, those it was built
  // from, with chunks in `ffn` and the residual and output of alternate
  // layers in `hidden`, all of which it overwrites: each function at its own
  // rows over the first two layers, with the ANE's output filled with NaN
  // before so that rows it leaves unwritten show, against the leading rows of
  // a full chunk on the GPU alone, which computes every row alike whatever
  // the chunk's rows. Returns the largest error of what the ANE adds to the
  // GPU's part (AneFfn.cpp). Throws if an output is not finite, that error
  // exceeds its bound or the ANE's work fails, which stops the split.
  [[nodiscard]] double verify(std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                              const std::array<metal::MetalBuffer, 2> &hidden);

private:
  // The split's channels: hidden and intermediate, the layers', the GPU's
  // and the ANE's.
  struct Shape final {
    uint32_t hidden = 0, intermediate = 0, layers = 0, gpu = 0, ane = 0;
  };
  // A layer's int8 weights on the ANE: gate's and up's rows of the ANE's
  // channels over the hidden inputs, down's rows over the ANE's channels,
  // and each one's per-row scales.
  struct Weights final {
    ane::Surface gate, up, down, gateScale, upScale, downScale;
  };
  // The GPU's share of a layer: gate and up over views of their leading
  // rows, down over the leading inputs of its rows; and the whole
  // projections, from which the ANE's weights were staged.
  struct Layer final {
    Projection gate, up, down;
  };
  // An input of the ANE program's functions: its name, the surface of each
  // layer bound to it (one for the chunk's inputs, which every layer
  // shares), and the rows and width the functions read of it, a width of 0
  // reading the chunk's rows.
  struct Input final {
    std::string name;
    std::vector<const ane::Surface *> surfaces;
    uint32_t rows = 0, width = 0;
  };
  // What encode() adds of a split layer: the GPU's work alone (its channels
  // and the packing and join of the ANE's), or with the ANE's evaluation,
  // which the shared event orders between the packing and the join.
  enum class Parts : uint8_t { Gpu, Both };
  // The program's function of `rows` rows over each layer's surfaces.
  struct Evaluation final {
    uint32_t rows = 0;
    std::vector<ane::Program::Binding> bindings;
  };
  // An evaluation of layer `layer`: it starts once the event reaches
  // `ready`, and raises it to `done`.
  struct Job final {
    uint32_t evaluation, layer;
    uint64_t ready, done;
  };

  [[nodiscard]] static Shape shapeOf(std::span<const SwiGluProjections> layers, uint32_t aneUnits);
  [[nodiscard]] std::vector<Input> programInputs() const;
  [[nodiscard]] std::string program(std::span<const Input> inputs, std::span<const uint32_t> rows) const;
  [[nodiscard]] std::string function(std::span<const Input> inputs, uint32_t rows) const;
  // Rotates and quantizes every layer's ANE weights from `layers`' planes.
  void stageWeights(std::span<const SwiGluProjections> layers);
  void encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
              metal::MetalBuffer output, uint32_t rows, Parts parts);
  // finish() but for its log line: if every evaluation of the command
  // committed last succeeded and its joins read finite values, the Neural
  // Engine's time over them; none otherwise, which stops the split.
  [[nodiscard]] std::optional<AwakeClock::duration> completed();
  // Logs, once, that the split stopped.
  void warnStopped();
  metal::MetalBuffer allocate(uint64_t bytes, const char *label);
  ane::Surface surface(uint32_t rows, uint32_t width, ane::Surface::Element element);

  metal::MetalBackend &backend_;
  const Linear linear_;
  Shape shape_;
  uint32_t minimumRows_ = 0;
  std::vector<Layer> layers_;
  // The chunk's rotated rows, their per-token scales and the ANE's partial
  // down projection, each of kMaximumRows rows whose row stride each
  // function declares; the Hadamard signs; and the status word a join sets
  // when it reads a value that is not finite (ane_ffn_join).
  metal::MetalBuffer signs_, rotated_, status_;
  ane::Surface inputs_, tokenScale_, partial_;
  std::vector<Weights> weights_;
  std::unique_ptr<ane::Program> program_;
  // By rows, ascending.
  std::vector<Evaluation> evaluations_;
  // The jobs of the command being encoded, which commit() starts, and the
  // layer it adds next.
  std::vector<Job> jobs_;
  uint32_t nextLayer_ = 0;
  uint32_t committedEvaluations_ = 0;
  double lastAneMilliseconds_ = 0.0;
  // Whether a command committed jobs that finish() has not yet judged.
  bool unfinished_ = false;
  bool warned_ = false;
  uint64_t allocatedBytes_ = 0;
  // Destroyed first: it waits for the evaluations the ANE has not reported.
  ane::Handoff handoff_;
};

} // namespace splash::ops

#pragma once

#include "ane/Handoff.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/AneFfnCalibration.hpp"
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

namespace ane_ffn {
class Measurement;
} // namespace ane_ffn

// The dense FFN of a prefill command split by intermediate channel between the
// GPU and the Neural Engine. The GPU runs the leading channels with its
// prefill kernels over views of the projections' planes;
// the ANE runs the rest as one W8A8 program over a step's rows, with int8
// weights the GPU requantizes from those planes one layer ahead into
// double-buffered surfaces, and the GPU adds the ANE's partial down projection
// to its own. A command of enough rows runs as two steps, its rows' halves
// (firstStepRows), whose layers interleave: the ANE runs a step's FFN while
// the GPU runs the other step's mixer, which needs only the earlier rows'
// K/V and state (QwenTarget::addPrefillLayers places each join). Each kind of
// command, one step or two, takes its own share of the channels (Shares),
// both in the one program over the one set of surfaces: the surfaces hold the
// ANE's channels in descending units, the last first, so that every share's
// channels are their leading rows (and down's leading inputs). A shared
// event orders each step's ANE evaluation between the GPU's packing of its
// inputs and that join, within the one prefill command (ane::Handoff). Any
// failure of the ANE's work stops the split for the life of the process:
// finish() then reports the command's results unusable, and the caller runs
// the command again on the GPU alone, which every later command runs too. A
// split that loses to the GPU alone pauses after a command whose results are
// usable, and stops the same way once a probe after the pause loses too
// (setBreaker, ane_ffn::Pause).
class AneFfn final {
public:
  // A step runs on the smallest of the program's functions that holds it,
  // one every kProgramStep rows from kMinimumRows to the prefill budget,
  // since an evaluation costs the ANE its function's rows. The least is
  // 1.7.0's.
  static constexpr uint32_t kProgramStep = 128, kMinimumRows = 512, kMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
  static_assert(kMaximumRows >= kMinimumRows && (kMaximumRows - kMinimumRows) % kProgramStep == 0,
                "the functions cover the chunks the split takes");
  // The least rows of a command a split runs in two steps: twice one step's
  // least, so each half takes a function of at least as many. Two steps beat
  // one from 1,024 rows on every Mac and model measured (an M6, two M5 Pros
  // and an M5 Max, Qwen3.8-27B in MLX 4-bit, UD-Q4_K_M and UD-IQ3_XXS), and
  // from 672 on the M6, so this gives up 2-4% there; below 896 rows on the
  // M5 Max the halves' evaluations stalled the GPU at random. A program of
  // both kinds holds one step's functions to this (5) and the halves' (5):
  // Apple9's ANE compiler takes 15, where it refuses 22 (ANECCompile() FAILED
  // on an M4, each time).
  static constexpr uint32_t kTwoStepRows = 2 * kMinimumRows;
  static_assert(kTwoStepRows <= kMaximumRows && (kTwoStepRows - kMinimumRows) / kProgramStep +
                                                        (kMaximumRows / 2 - kMinimumRows) / kProgramStep + 2 <=
                                                    15,
                "a program of both kinds holds at most the 15 functions Apple9 compiles");
  // The GPU part's down projection is a view of leading inputs, which only
  // the prefill residual tiles (GgufPrefill, GgufPrefillInt8) run: no chunk
  // the split takes stages.
  static_assert(kMinimumRows > kMaximumStagedRows, "the split's chunks run the prefill residual tiles");

  // The kinds of command: one step, or two pipelined steps.
  enum class Steps : uint8_t { One, Two };
  // The most evaluations the ANE's client holds in flight for a program (its
  // queue depth, macOS 26.5 and 27.0), which a command queues all of before
  // the GPU starts (commit()): one a layer of each step, but a command's
  // last layer, whose rows are read, runs on the GPU alone. A command of
  // more would wait 10 s on its client and fail. Qwen3.8-27B's 64 layers
  // queue 126 in two steps.
  static constexpr uint32_t kMostInFlight = 127;
  // Whether a command of `steps` over `layers` FFN layers queues no more
  // evaluations than kMostInFlight.
  [[nodiscard]] static constexpr bool fits(uint32_t layers, Steps steps) noexcept {
    return layers && uint64_t{layers - 1} * (steps == Steps::Two ? 2 : 1) <= kMostInFlight;
  }
  // The ANE's units of the units() of intermediate channels in a command of
  // each kind; none where the split takes no command of that kind.
  struct Shares final {
    uint32_t oneStep = 0, twoSteps = 0;
    [[nodiscard]] uint32_t most() const noexcept { return oneStep > twoSteps ? oneStep : twoSteps; }
  };
  // The least rows of a command the split takes in one step, and of one it
  // takes in two, from which every larger command runs in two; none of
  // either kind it takes no command of.
  struct Rows final {
    uint32_t oneStep = 0, twoSteps = 0;
  };

  // The ANE takes `shares` of the units() of intermediate channels, and the
  // GPU the rest through `linear`'s prefill plans, in commands of `rows`:
  // its program holds the functions programRows(shares) gives. The program
  // is compiled and loaded here, and the wait for the ANE's service ends with
  // ane::Interrupted once `interrupted` returns true. Its gate and up run in
  // blocks where `blocks` says so, else as the backend's chip runs them best
  // (AneFfn.cpp). Throws std::invalid_argument for a kind of command without
  // its share or rows, or rows no function holds.
  AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
         Shares shares, Rows rows, std::function<bool()> interrupted, std::optional<bool> blocks = std::nullopt);
  AneFfn(const AneFfn &) = delete;
  AneFfn &operator=(const AneFfn &) = delete;

  // Why the split does not take `layers`, or null when it does: unrotated
  // quantized projections of one shape (AneFfn.cpp).
  [[nodiscard]] static const char *unsupported(std::span<const SwiGluProjections> layers);
  // The units the split moves intermediate channels in, of `layers`, which
  // it takes: the ANE takes from one to all but one of them.
  [[nodiscard]] static uint32_t units(std::span<const SwiGluProjections> layers);
  // The Metal memory of the split of `layers` with the ANE taking `shares`.
  [[nodiscard]] static uint64_t plannedBytes(std::span<const SwiGluProjections> layers, Shares shares);
  // The rows of the functions of a split of `shares`, ascending, by kind of
  // command: of one share, one step's every kProgramStep rows from
  // kMinimumRows to kMaximumRows (13), 1.7.0's; of both, one step's to
  // kTwoStepRows (5), whose last holds the chunks below it, and the halves'
  // from kMinimumRows to half the most rows (5). The rows a split takes only
  // choose among them (setRows), so every split of the same shares runs the
  // one program, which the ANE's service compiles once: calibration times
  // the program that serves.
  [[nodiscard]] static std::array<std::vector<uint32_t>, 2> programRows(Shares shares);
  // What tells this Mac's calibrations of a split of `layers` apart for one
  // macOS version, engine build and most units its plan holds: the layers'
  // shapes and formats, and the Int8Activations of `linear`, whose plans run
  // the GPU's part and so move the units the split takes (Qwen3.8-27B MLX
  // 4-bit's takes 10 of 34 with them on, 15 off, on a 20-core M5 Pro).
  [[nodiscard]] static std::string calibrationKey(std::span<const SwiGluProjections> layers, const Linear &linear);

  // The fraction of intermediate channels the ANE takes in a command of
  // `steps`; 0 for a kind the split takes none of.
  [[nodiscard]] double share(Steps steps) const noexcept;
  // What Metal allocated for the split, which plannedBytes() bounds.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }
  // The least rows of a command the split takes, of either kind.
  [[nodiscard]] uint32_t minimumRows() const noexcept;
  [[nodiscard]] const Rows &rows() const noexcept { return rows_; }
  // Takes commands of `rows`, which its shares and functions hold. Throws
  // std::invalid_argument otherwise.
  void setRows(Rows rows);
  // Pauses the split for `pause` once it loses to the GPU alone, and stops
  // it once a probe loses too (ane_ffn::Pause): `breaker` judges each command
  // finish() finds split, and `lost` runs once when it stops. Until this is
  // called, none does.
  void setBreaker(ane_ffn::Breaker breaker, std::function<void()> lost,
                  AwakeClock::duration pause = ane_ffn::Pause::kLength) noexcept {
    breaker_ = std::move(breaker);
    lost_ = std::move(lost);
    pause_ = ane_ffn::Pause(pause);
  }

  // The commands finish() found split since the split was built, their
  // evaluations, and the Neural Engine's milliseconds over them
  // (ane::Handoff::finish).
  struct Served final {
    uint64_t commands = 0, evaluations = 0;
    double milliseconds = 0.0;
  };
  [[nodiscard]] const Served &served() const noexcept { return served_; }

  // Checks the split against the GPU alone on `layers`, those it was built
  // from, with chunks in `ffn` and the residual and output of alternate
  // layers in `hidden`, all of which it overwrites: each function at its own
  // rows over the first two layers, which stage the ANE's weights through
  // both sets, with the ANE's output filled with NaN before so that rows it
  // leaves unwritten show; each through begin(), the layers encoded as add()
  // encodes them, commit() and its completion check; each against the
  // leading rows of a full chunk on the GPU, which computes every row alike
  // whatever the chunk's rows. Returns the largest error of what the ANE
  // adds to the GPU's part (AneFfn.cpp). Throws ane::Refused if an output is
  // not finite, that error exceeds its bound or the ANE's work fails, which
  // stops the split.
  [[nodiscard]] double verify(std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                              const std::array<metal::MetalBuffer, 2> &hidden);

  // Whether the split takes a command of `rows` rows: from minimumRows() to
  // kMaximumRows, but not while it pauses (setBreaker) and not once it stops.
  [[nodiscard]] bool splits(uint32_t rows) const;
  // The kind of a command of `rows` rows the split takes: two steps from
  // rows().twoSteps, one below.
  [[nodiscard]] Steps steps(uint32_t rows) const noexcept;
  // The rows of the first step of a command of `rows` rows: the first half
  // of one the split takes in two steps; all of them otherwise.
  [[nodiscard]] uint32_t firstStepRows(uint32_t rows) const;
  // Whether the split stopped, and why.
  [[nodiscard]] bool retired() const { return handoff_.retired(); }
  [[nodiscard]] std::string reason() const { return handoff_.reason(); }

  // Starts encoding a command of `steps`, whose share its layers take; false
  // once the split stopped. A command of a kind the split takes none of adds
  // no layer (std::logic_error).
  bool begin(Steps steps);
  // Layer `layer`'s FFN of a step of `rows` rows, from kMinimumRows to
  // kMaximumRows, after begin(): output = residual + FFN of ffn.normalized,
  // whose ANE part join() adds. Each step adds its layers in order from layer
  // 0, the steps of a layer one after the other, and join() comes between
  // adds: the first step's join once the second step's mixer is encoded, the
  // second's once the first step's next mixer is.
  void add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
           metal::MetalBuffer output, uint32_t rows);
  // Encodes the join of the step added last, if not joined: the wait for its
  // ANE evaluation and the addition of its part to the output.
  void join(metal::CommandGraph &graph);
  // Starts the ANE's evaluations of the command encoded since begin(), then
  // submits it (MetalBackend::submitCommandAsync). A submission that throws
  // stops the split before the exception goes on. Throws std::logic_error
  // for a step not joined.
  [[nodiscard]] metal::CommandTicket commit(const metal::CommandGraph &graph, metal::CommandCompletion completion);
  // After the command committed last has completed, in `timing`: whether its
  // outputs hold the split's FFN. False when its ANE work failed, which stops
  // the split and logs why, once; true otherwise, also when the breaker stops
  // the split after it. Logs a buffer of it that stalled (ane_ffn::late).
  [[nodiscard]] bool finish(const metal::CommandTiming &timing = {});

  // The engine's idle release (engine::ReleasableMemory). Between commands:
  // unloads the program, whose memory the ANE service gives back, and keeps
  // the split's own (allocatedBytes()), which the backend's residency
  // keep-alive unwires: the surfaces and scratch every command writes again,
  // and the row scales and signs computed once. True once it unloaded the
  // program; false, doing nothing, when it is unloaded, or the split stopped
  // while evaluations it gave up on may still use it. A stopped split unloads
  // it once they have all reported, and keeps it unloaded. A program that does
  // not unload stops the split. Throws std::logic_error while a command it
  // encoded is not committed or not finished.
  [[nodiscard]] bool release();
  // Loads the program release() unloaded again, unless the split stopped;
  // nothing otherwise. Never throws: a program that does not load stops the
  // split, and the GPU runs every later chunk alone.
  void restore() noexcept;

private:
  // Times the split's layers and their GPU part against the GPU's alone.
  friend class ane_ffn::Measurement;

  // The split's channels: hidden and intermediate, the layers', the GPU's and
  // the ANE's, and the segments of the hidden inputs and of down's inputs the
  // ANE takes.
  struct Shape final {
    uint32_t hidden = 0, intermediate = 0, layers = 0, gpu = 0, ane = 0;
    std::vector<uint32_t> inputSegments, downSegments;
  };
  // A staging set of the ANE's int8 weights: gate's and up's rows over each
  // segment of the hidden inputs, down's rows over each segment of its ANE
  // inputs, and each one's per-row scales.
  struct Weights final {
    std::vector<ane::Surface> gate, up, down;
    ane::Surface gateScale, upScale, downScale;
  };
  // Everything the split allocates (allocate() in AneFfn.cpp). The chunk's
  // rotated input rows in int8 segments, their per-token scales and the ANE's
  // partial down projection, which every function reads or writes, hold
  // kMaximumRows rows, whose row stride each function declares. A join that
  // reads a partial value or token scale that is not finite sets the status
  // word (ane_ffn_join).
  struct Memory final {
    metal::MetalBuffer signs, rotated, status;
    std::vector<ane::Surface> inputs;
    ane::Surface tokenScale, partial;
    std::array<Weights, 2> sets;
  };
  // A projection's weight planes as the ane_ffn kernels bind them: an image
  // tensor's plane0, plane1 and meta, in groups of 32 inputs, of their
  // `groups` per row, and its `format`.
  struct Planes final {
    // Throws unless the planes hold all of the projection's weights.
    explicit Planes(const Projection &projection);
    std::array<metal::MetalBuffer, 3> buffers;
    uint32_t groups = 0, format = 0;
  };
  // A matrix of a layer: its index in Layer::planes, and the part of the
  // layer's per-row int8 scales rowScales() views.
  enum class Matrix : uint8_t { Gate, Up, Down };
  // A layer's GPU part in a share: gate and up over views of their leading
  // rows, down over the leading inputs of its rows.
  struct GpuPart final {
    Projection gate, up, down;
  };
  // An input of the ANE program's functions: its name, the surface each
  // weight set binds to it, and the rows and width the functions read of it,
  // a width of 0 reading the chunk's rows.
  struct Input final {
    std::string name;
    std::array<const ane::Surface *, 2> surfaces;
    uint32_t rows = 0, width = 0;
  };
  // What encode() adds of a split layer: the GPU's work alone (its channels,
  // the staging of the next layer's int8 weights, and the packing and join
  // of the ANE's channels), or with the ANE's evaluation, which the shared
  // event orders between the packing and the join.
  enum class Parts : uint8_t { Gpu, Both };
  // The program's function of `rows` rows over the surfaces of each weight
  // set, by set.
  struct Evaluation final {
    uint32_t rows = 0;
    std::vector<ane::Program::Binding> bindings;
  };
  // An evaluation of weight set `set`: it starts once the event reaches
  // `ready`, and raises it to `done`.
  struct Job final {
    uint32_t evaluation, set;
    uint64_t ready, done;
  };
  // A step's join that join() encodes: into `output`, of `rows` rows, after
  // the event reaches `done` (none for the GPU's part alone).
  struct Join final {
    metal::MetalBuffer output;
    uint32_t rows = 0;
    uint64_t done = 0;
  };

  // A kind of command's share: its channels (a Shape of its own), its layers'
  // GPU parts, every layer's per-row int8 scales of the ANE's weights over
  // its channels, and its program's functions, by rows, ascending.
  struct Share final {
    Shape shape;
    std::vector<GpuPart> layers;
    metal::MetalBuffer rowScales;
    std::vector<Evaluation> evaluations;
  };

  // The split with a function of each of `functionRows` rows, ascending, by
  // kind of command, and no rows set: every command of a function's rows.
  AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
         Shares shares, std::function<bool()> interrupted, const std::array<std::vector<uint32_t>, 2> &functionRows,
         std::optional<bool> blocks = std::nullopt);
  [[nodiscard]] static Shape shapeOf(std::span<const SwiGluProjections> layers, uint32_t aneUnits);
  // Every allocation of the split, which `buffer` and `surface` make:
  // plannedBytes() counts them and the constructor allocates them: the
  // surfaces of the most units, and each share's row scales.
  template <class MakeBuffer, class MakeSurface>
  [[nodiscard]] static Memory allocate(const Shape &shape, MakeBuffer &&buffer, MakeSurface &&surface);
  [[nodiscard]] static uint64_t rowScaleBytes(const Shape &shape);
  // A share's program inputs, the leading rows and columns of the surfaces
  // its channels take, and its function of `rows` rows over them.
  [[nodiscard]] static std::vector<Input> programInputs(const Shape &shape, const Memory &memory);
  [[nodiscard]] static std::string function(const Shape &shape, std::span<const Input> inputs,
                                            const ane::Surface &output, uint32_t rows, bool blocks);
  // The share of the command being encoded.
  [[nodiscard]] const Share &share() const;
  // Fills the CPU-visible bf16 rows `normalized` with the same values of a
  // normalized row's magnitude each time.
  static void fillNormalized(const metal::MetalBuffer &normalized);
  void encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
              metal::MetalBuffer output, uint32_t rows, Parts parts);
  void addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const;
  [[nodiscard]] metal::MetalBuffer rowScales(const Share &share, uint32_t layer, Matrix matrix) const;
  // Starts `job` on the ANE through the handoff.
  void queue(const Job &job);
  // finish() but for its log line and judgment: if every evaluation of the
  // command committed last succeeded and its joins read finite values, the
  // Neural Engine's time over them (ane::Handoff::finish); none otherwise,
  // which stops the split.
  [[nodiscard]] std::optional<ane::Handoff::Ran> completed();
  // Logs, once, that the split stopped.
  void warnStopped();

  metal::MetalBackend &backend_;
  const Linear linear_;
  // The surfaces' channels: the most units of the shares.
  Shape shape_;
  Rows rows_;
  // Each layer's planes, from which the GPU stages the ANE's int8 weights.
  std::vector<std::array<Planes, 3>> planes_;
  Memory memory_;
  std::unique_ptr<ane::Program> program_;
  // By kind of command; none for a kind the split takes none of.
  std::array<std::optional<Share>, 2> shares_;
  // The kind of the command being encoded.
  Steps steps_ = Steps::One;
  // The jobs of the command being encoded, which commit() starts; the
  // layers it has added, and the join of the step added last until join()
  // encodes it.
  std::vector<Job> jobs_;
  uint32_t layersAdded_ = 0;
  std::optional<Join> pending_;
  // The layers and evaluations of the command committed last, and its
  // evaluations' function rows over a layer.
  uint32_t committedLayers_ = 0, committedEvaluations_ = 0, committedRows_ = 0;
  ane_ffn::Breaker breaker_;
  std::function<void()> lost_;
  ane_ffn::Pause pause_;
  Served served_;
  // Whether a command committed jobs that finish() has not yet judged.
  bool unfinished_ = false;
  // From a release() that unloaded the program until restore().
  bool released_ = false;
  bool warned_ = false;
  uint64_t allocatedBytes_ = 0;
  // Destroyed first: it waits for the evaluations the ANE has not reported.
  ane::Handoff handoff_;
};

} // namespace splash::ops

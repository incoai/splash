#pragma once

#include "ane/Handoff.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/AneFfn.h"
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
#include <utility>
#include <vector>

namespace splash::ops {

namespace ane_ffn {
class Measurement;

// Output columns of a projection, {begin, length} each.
using Columns = std::vector<std::pair<uint32_t, uint32_t>>;
// A kind of the mixers' projections as the model lays it out: its layers'
// projections, all of one shape; the units the split moves its columns in,
// of `unitColumns` each, the GPU taking the leading columns of all but the
// ANE's units; and the ANE's columns with the ANE taking `aneUnits`: those
// ahead of the mixer, which a layer joins before the mixer reads them, and
// those it defers, which the mixer reads only at its gate and the ANE computes
// while the GPU runs the scan or the attention, none for an output
// projection. Ahead and deferred each list a first range and then at most one
// more, or equal blocks at a fixed stride (ane_proj_join).
struct ProjectionLayout final {
  ProjectionKind kind = ProjectionKind::GdnInput;
  std::vector<const Projection *> layers;
  uint32_t units = 0, unitColumns = 0;
  std::function<Columns(uint32_t aneUnits)> ahead, deferred;
};
} // namespace ane_ffn

// The dense FFN of a prefill chunk split by intermediate channel between the
// GPU and the Neural Engine. The GPU runs the leading channels with its
// prefill kernels over views of the projections' planes;
// the ANE runs the rest as one W8A8 program over the chunk's rows, with int8
// weights the GPU requantizes from those planes one layer ahead into
// double-buffered surfaces, and the GPU adds the ANE's partial down projection
// to its own. A shared event orders each layer's ANE evaluation between the
// GPU's packing of its inputs and that join, within the one prefill command
// (ane::Handoff). Any failure of the ANE's work stops the split for the life
// of the process: finish() then reports the command's results unusable, and
// the caller runs the chunk again on the GPU alone, which every later chunk
// runs too. An evaluation that only ran late stops it until the ANE has
// reported it, a few times at most, and the next command starts over. A split
// that loses to the GPU alone stops for good, after a command whose results
// are usable (setBreaker).
class AneFfn final {
public:
  // A chunk runs on the smallest of the program's functions that holds it,
  // one every kProgramStep rows from kMinimumRows to the prefill budget,
  // since an evaluation costs the ANE its function's rows.
  static constexpr uint32_t kMinimumRows = 512, kProgramStep = 128, kMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
  static_assert(kMaximumRows >= kMinimumRows && (kMaximumRows - kMinimumRows) % kProgramStep == 0,
                "the functions cover the chunks the split takes");

  // The ANE takes `aneUnits` of the units() of intermediate channels. The
  // program is compiled and loaded here, and the wait for the ANE's service
  // ends with ane::Interrupted once `interrupted` returns true. The split
  // takes every chunk its functions hold until setMinimumRows().
  AneFfn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, uint32_t aneUnits,
         std::function<bool()> interrupted);
  AneFfn(const AneFfn &) = delete;
  AneFfn &operator=(const AneFfn &) = delete;

  // Why the split does not take `layers`, or null when it does: unrotated
  // quantized projections of one shape (AneFfn.cpp).
  [[nodiscard]] static const char *unsupported(std::span<const SwiGluProjections> layers);
  // The units the split moves intermediate channels in, of `layers`, which
  // it takes: the ANE takes from one to all but one of them.
  [[nodiscard]] static uint32_t units(std::span<const SwiGluProjections> layers);
  // The Metal memory of the split of `layers` with the ANE taking `aneUnits`.
  [[nodiscard]] static uint64_t plannedBytes(std::span<const SwiGluProjections> layers, uint32_t aneUnits);

  // The fraction of intermediate channels the ANE takes.
  [[nodiscard]] double share() const noexcept;
  // What Metal allocated for the split, which plannedBytes() bounds.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }
  // The least rows of a chunk the split takes (ane_ffn::minimumRows), from
  // kMinimumRows to kMaximumRows.
  [[nodiscard]] uint32_t minimumRows() const noexcept { return minimumRows_; }
  void setMinimumRows(uint32_t rows);
  // Stops the split once it loses to the GPU alone: `breaker` judges each
  // command finish() finds split, and `lost` runs once when it trips. Until
  // this is called, none does.
  void setBreaker(ane_ffn::Breaker breaker, std::function<void()> lost) noexcept {
    breaker_ = std::move(breaker);
    lost_ = std::move(lost);
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
  // adds to the GPU's part (AneFfn.cpp). Throws if an output is not finite,
  // that error exceeds its bound or the ANE's work fails, which stops the
  // split.
  [[nodiscard]] double verify(std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                              const std::array<metal::MetalBuffer, 2> &hidden);

  // Whether the split takes a chunk of `rows` rows: from minimumRows() to
  // kMaximumRows, until it stops.
  [[nodiscard]] bool splits(uint32_t rows) const;
  // Whether the split stopped, and why.
  [[nodiscard]] bool retired() const { return handoff_.retired(); }
  [[nodiscard]] std::string reason() const { return handoff_.reason(); }

  // Starts encoding a command; false once the split stopped.
  bool begin();
  // Layer `layer`'s FFN of a chunk of `rows` rows, from kMinimumRows to
  // kMaximumRows, encoded in layer order from layer 0 after begin(): output =
  // residual + FFN of ffn.normalized.
  void add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
           metal::MetalBuffer output, uint32_t rows);
  // Starts the ANE's evaluations of the command encoded since begin(), then
  // submits it (MetalBackend::submitCommandAsync). A submission that throws
  // stops the split before the exception goes on.
  [[nodiscard]] metal::CommandTicket commit(const metal::CommandGraph &graph, metal::CommandCompletion completion);
  // After the command committed last has completed: whether its outputs hold
  // the split's FFN. False when its ANE work failed, which stops the split
  // and logs why, once; true otherwise, also when the breaker stops the
  // split after it.
  [[nodiscard]] bool finish();

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

  // The mixers' projections, split by output channel over this split's
  // handoff (AneFfn.cpp): the GPU computes a projection's leading
  // columns with its prefill kernels over views of the leading rows of the
  // projection's planes, the ANE the rest as a W8A8 program of its own over
  // the chunk's rows, with int8 rows the GPU stages a layer ahead, and a join
  // writes the ANE's columns where the layout puts them. A projection's ANE
  // work fails, stops and resumes with the FFN's: it is the same split. Each
  // kind judges itself against the GPU alone (setProjectionBreaker), and one
  // that loses goes back to the GPU alone without stopping the rest.
  //
  // Why the split does not take `layout`, or null when it does.
  [[nodiscard]] static const char *unsupportedProjections(const ane_ffn::ProjectionLayout &layout);
  // The Metal memory addProjections() allocates for `layout` with the ANE
  // taking `aneUnits`.
  [[nodiscard]] static uint64_t plannedProjectionBytes(const ane_ffn::ProjectionLayout &layout, uint32_t aneUnits);
  // Splits `layout`'s projections with the ANE taking `aneUnits` of its
  // units, from one to all but one, over chunks this split takes: allocates
  // their memory, compiles and loads their program, a function of each of
  // this split's functions' rows, and computes each row's int8 scale. Throws
  // if the split cannot take them or already takes the kind; the split goes
  // on without them.
  void addProjections(const ane_ffn::ProjectionLayout &layout, uint32_t aneUnits);
  // Whether a chunk of `rows` rows splits projections of `kind`: the split
  // takes the chunk (splits()) and the kind, whose breaker has not tripped.
  [[nodiscard]] bool projects(ane_ffn::ProjectionKind kind, uint32_t rows) const;
  // The ANE's share of `kind`'s units, 0 for a kind the split does not take.
  [[nodiscard]] double projectionShare(ane_ffn::ProjectionKind kind) const noexcept;
  // Stops splitting `kind` once it loses to the GPU alone, as setBreaker()
  // the FFN: `breaker` judges each command finish() finds split, and the
  // owner's `lost` runs once when any breaker trips.
  void setProjectionBreaker(ane_ffn::ProjectionKind kind, ane_ffn::Breaker breaker);
  // Layer `layer`'s projection of `kind`, of the layers of that kind, from 0
  // in order after begin(), for a chunk that projects(): output = input x the
  // projection's rows, ahead of the mixer, plus `residual` for an output
  // projection, which takes one. An input projection leaves its deferred
  // columns to addDeferredJoin().
  void addProjection(ane_ffn::ProjectionKind kind, metal::CommandGraph &graph, uint32_t layer,
                     metal::MetalBuffer input, metal::MetalBuffer output, uint32_t rows, LinearScratch scratch,
                     metal::MetalBuffer residual = {});
  // Waits for the deferred columns of the input projection addProjection()
  // added last of `kind` and joins them into `output`: before the mixer reads
  // them, and before anything the GPU signals the handoff with next.
  void addDeferredJoin(ane_ffn::ProjectionKind kind, metal::CommandGraph &graph, metal::MetalBuffer output,
                       uint32_t rows);
  // Checks `kind`'s split against the GPU alone as verify() checks the FFN's,
  // on its first two layers, reading `input` and writing `output`, after
  // `residual` for an output projection, all of which it overwrites. Returns
  // the largest error of what the ANE adds. Throws as verify() does, and the
  // split then goes on without the kind; only a failure of the ANE's work
  // stops the split.
  [[nodiscard]] double verifyProjections(ane_ffn::ProjectionKind kind, metal::MetalBuffer input,
                                         metal::MetalBuffer output, LinearScratch scratch,
                                         metal::MetalBuffer residual = {});

private:
  // Times the split's layers and their GPU part against the GPU's alone.
  friend class ane_ffn::Measurement;

  // The split's channels: hidden and intermediate, the layers', the GPU's and
  // the ANE's, and the segments of down's inputs the ANE takes.
  struct Shape final {
    uint32_t hidden = 0, intermediate = 0, layers = 0, gpu = 0, ane = 0;
    std::vector<uint32_t> downSegments;
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
    metal::MetalBuffer signs, rowScales, rotated, status;
    std::vector<ane::Surface> inputs;
    ane::Surface tokenScale, partial;
    std::array<Weights, 2> sets;
  };
  // A projection's weight planes as the ane_ffn kernels bind them: an image
  // tensor's plane0, plane1 and meta, in groups of 32 inputs, of their
  // `groups` per row, and its `format`; `first` and `rows` are the
  // projection's output columns the tensor holds.
  struct Planes final {
    // Throws unless the planes hold all of the projection's weights.
    explicit Planes(const Projection &projection);
    Planes(const QuantizedSegment &segment, uint32_t inputs);
    std::array<metal::MetalBuffer, 3> buffers;
    uint32_t groups = 0, format = 0, first = 0, rows = 0;
  };
  // A matrix of a layer: its index in Layer::planes, and the part of the
  // layer's per-row int8 scales rowScales() views.
  enum class Matrix : uint8_t { Gate, Up, Down };
  // A layer, held by value: the planes of its gate, up and down, from which
  // the GPU stages the ANE's int8 weights, and the GPU's share of it: gate
  // and up over views of their leading rows, down over the leading inputs of
  // its rows.
  struct Layer final {
    std::array<Planes, 3> planes;
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
    // The FFN's, kFfn, or a projection kind's (its index), and for a
    // projection whether it evaluates the deferred columns.
    uint8_t owner = kFfn;
    bool deferred = false;
  };
  static constexpr uint8_t kFfn = ane_ffn::kProjectionKinds;
  // The ANE's columns of a projection kind, ahead of the mixer or deferred:
  // the layout's columns and how many they hold; their int8 rows over each
  // segment of the inputs and their per-row scales, staged in two sets, and
  // each layer's per-row int8 scales, computed once; the partial output, its
  // rows channel-major before each token's input scale; and the program's
  // function of each of the split's rows, by rows, ascending.
  struct ProjectionColumns final {
    ane_ffn::Columns columns;
    uint32_t count = 0;
    std::array<std::vector<ane::Surface>, 2> weights;
    std::array<ane::Surface, 2> scales;
    metal::MetalBuffer rowScales;
    ane::Surface partial;
    std::vector<Evaluation> evaluations;
  };
  // A projection kind the split takes (addProjections()): its shape, the
  // GPU's views of each layer's leading rows and the rows its part writes,
  // the layers' planes the ANE's rows are staged from, the rotated inputs and
  // their int8 segments of a projection whose inputs are not the hidden rows
  // the FFN's surfaces hold, the ANE's columns, its program, and its breaker.
  struct ProjectionSplit final {
    static constexpr uint32_t kNotStaged = ~0u;
    ane_ffn::ProjectionKind kind = ane_ffn::ProjectionKind::GdnInput;
    uint32_t units = 0, aneUnits = 0, width = 0, inputs = 0, gpu = 0;
    std::vector<uint32_t> segments;
    std::vector<const Projection *> layers;
    // Per layer, the planes of each tensor the projection stacks.
    std::vector<std::vector<Planes>> planes;
    std::vector<Projection> gpuRows;
    metal::MetalBuffer gpuPart, rotated;
    std::vector<ane::Surface> x;
    ProjectionColumns ahead, deferred;
    std::unique_ptr<ane::Program> program;
    ane_ffn::Breaker breaker;
    // What Metal allocated for it.
    uint64_t allocatedBytes = 0;
    // Once its breaker tripped.
    bool lost = false;
    // The layer addProjection() takes next, the one whose rows the weight
    // sets hold the last staged of, and while its deferred evaluation runs,
    // the event value that raises once done.
    uint32_t next = 0, staged = kNotStaged;
    uint64_t deferredDone = 0;
  };
  // The kind's split, null for one the split does not take.
  [[nodiscard]] ProjectionSplit *projection(ane_ffn::ProjectionKind kind) const noexcept {
    return projections_[static_cast<size_t>(kind)].get();
  }
  // The projection's shape and memory with the ANE taking `aneUnits` of
  // `layout`, which `buffer` and `surface` make: plannedProjectionBytes()
  // counts them and addProjections() allocates them.
  template <class MakeBuffer, class MakeSurface>
  [[nodiscard]] static std::unique_ptr<ProjectionSplit> allocateProjection(const ane_ffn::ProjectionLayout &layout,
                                                                           uint32_t aneUnits, MakeBuffer &&buffer,
                                                                           MakeSurface &&surface);
  // The program of a projection's ahead and deferred columns, a function of
  // each of `rows` rows of each, and their names.
  [[nodiscard]] static std::string projectionProgram(const ProjectionSplit &split, const std::vector<ane::Surface> &x,
                                                     std::span<const uint32_t> rows);
  [[nodiscard]] static std::string projectionFunction(bool deferred, uint32_t rows);
  // Layer `layer`'s int8 rows and their scales of `split`'s ANE columns into
  // weight set `set`.
  void addProjectionWeights(const ProjectionSplit &split, metal::CommandGraph &graph, uint32_t layer,
                            uint32_t set) const;
  // Stages each projection kind's next rows into the set its next layer
  // reads, once, at a point where the GPU waits for the ANE anyway: the
  // FFN's join (encode()), whose ANE part outlasts the GPU's on most Macs.
  void addNextProjectionWeights(metal::CommandGraph &graph);
  // The join of `columns` of `split` over `rows` rows.
  [[nodiscard]] AneProjJoinParams joinParams(const ProjectionSplit &split, const ProjectionColumns &columns,
                                             uint32_t rows) const;
  // The ANE's input surfaces of `split`: the FFN's for the hidden rows.
  [[nodiscard]] const std::vector<ane::Surface> &inputsOf(const ProjectionSplit &split) const noexcept {
    return split.x.empty() ? memory_.inputs : split.x;
  }
  // The rows of the split's functions, ascending.
  [[nodiscard]] std::vector<uint32_t> functionRowsOf() const;
  // addProjections() with a function of each of `functionRows` rows, and its
  // undoing, which calibration's timings take (ane_ffn::Measurement).
  void addProjections(const ane_ffn::ProjectionLayout &layout, uint32_t aneUnits,
                      std::span<const uint32_t> functionRows);
  void removeProjections(ane_ffn::ProjectionKind kind);
  // verifyProjections() but for its undoing of a kind that fails.
  [[nodiscard]] double checkProjections(ane_ffn::ProjectionKind kind, metal::MetalBuffer input,
                                        metal::MetalBuffer output, LinearScratch scratch, metal::MetalBuffer residual);
  // addProjection(), or with Parts::Gpu the GPU's work alone, whose deferred
  // join encodeDeferredJoin() adds without waiting.
  void encodeProjection(ane_ffn::ProjectionKind kind, metal::CommandGraph &graph, uint32_t layer,
                        metal::MetalBuffer input, metal::MetalBuffer output, uint32_t rows, LinearScratch scratch,
                        metal::MetalBuffer residual, Parts parts);
  void encodeDeferredJoin(ane_ffn::ProjectionKind kind, metal::CommandGraph &graph, metal::MetalBuffer output,
                          uint32_t rows, Parts parts);

  // The split with a function of each of `functionRows` rows, ascending.
  AneFfn(metal::MetalBackend &backend, std::span<const SwiGluProjections> layers, uint32_t aneUnits,
         std::function<bool()> interrupted, std::span<const uint32_t> functionRows);
  [[nodiscard]] static Shape shapeOf(std::span<const SwiGluProjections> layers, uint32_t aneUnits);
  // Every allocation of the split, which `buffer` and `surface` make:
  // plannedBytes() counts them and the constructor allocates them.
  template <class MakeBuffer, class MakeSurface>
  [[nodiscard]] static Memory allocate(const Shape &shape, MakeBuffer &&buffer, MakeSurface &&surface);
  // The program's inputs; the program over them, with a function of each of
  // `rows` rows; and its function of `rows` rows.
  [[nodiscard]] static std::vector<Input> programInputs(const Shape &shape, const Memory &memory);
  [[nodiscard]] static std::string program(const Shape &shape, std::span<const Input> inputs,
                                           const ane::Surface &output, std::span<const uint32_t> rows);
  [[nodiscard]] static std::string function(const Shape &shape, std::span<const Input> inputs,
                                            const ane::Surface &output, uint32_t rows);
  // Fills the CPU-visible bf16 rows `normalized` with the same values of a
  // normalized row's magnitude each time.
  static void fillNormalized(const metal::MetalBuffer &normalized);
  void encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
              metal::MetalBuffer output, uint32_t rows, Parts parts);
  void addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const;
  [[nodiscard]] metal::MetalBuffer rowScales(uint32_t layer, Matrix matrix) const;
  // Starts `job` on the ANE through the handoff.
  void queue(const Job &job);
  // finish() but for its log line and judgment: if every evaluation of the
  // command committed last succeeded and its joins read finite values, the
  // Neural Engine's time over each, in the order of the command's jobs
  // (ane::Handoff::finishEach); none otherwise, which stops the split.
  [[nodiscard]] std::optional<std::vector<AwakeClock::duration>> completed();
  // Logs, once, that the split stopped.
  void warnStopped();

  metal::MetalBackend &backend_;
  const Linear linear_;
  // Ends the wait for the ANE's service of each program the split compiles.
  const std::function<bool()> interrupted_;
  Shape shape_;
  uint32_t minimumRows_ = kMinimumRows;
  std::vector<Layer> layers_;
  Memory memory_;
  std::unique_ptr<ane::Program> program_;
  // By rows, ascending.
  std::vector<Evaluation> evaluations_;
  // The jobs of the command being encoded, which commit() starts, and the
  // layer it adds next.
  std::vector<Job> jobs_;
  uint32_t nextLayer_ = 0;
  // The jobs of the command committed last, and its function's rows.
  std::vector<Job> committed_;
  uint32_t committedRows_ = 0;
  // The function rows of the command being encoded.
  uint32_t encodingRows_ = 0;
  // The projection kinds the split takes, by kind.
  std::array<std::unique_ptr<ProjectionSplit>, ane_ffn::kProjectionKinds> projections_;
  ane_ffn::Breaker breaker_;
  std::function<void()> lost_;
  Served served_;
  // Whether a command committed jobs that finish() has not yet judged.
  bool unfinished_ = false;
  // From a release() that unloaded the program until restore().
  bool released_ = false;
  bool warned_ = false;
  // Restarts after a late evaluation so far (begin()).
  uint32_t restarts_ = 0;
  [[nodiscard]] bool restartable() const;
  uint64_t allocatedBytes_ = 0;
  // Destroyed first: it waits for the evaluations the ANE has not reported.
  ane::Handoff handoff_;
};

} // namespace splash::ops

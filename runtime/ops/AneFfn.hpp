#pragma once

#include "ane/Handoff.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/AneFfnCalibration.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

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
// runs too. A split that loses to the GPU alone stops the same way, after a
// command whose results are usable (setBreaker).
class AneFfn final {
public:
  // The program has one function, of kEvaluationRows rows, and a chunk runs
  // as its pieces of that many rows, each an evaluation of the function into
  // surfaces of its own, the last padded: an evaluation costs the ANE its
  // function's rows. The service compiles a function in time about
  // proportional to its rows and channels, so one function of 512 rows
  // compiles in a tenth of the time of the thirteen, one every 128 rows from
  // 512 to 2,048, that each chunk ran on whole before. A pipelined command
  // cuts its chunks at whole pieces (Runtime's prefillChunkRows), so that only
  // its last chunk pads.
  static constexpr uint32_t kMinimumRows = 512, kEvaluationRows = 512, kMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
  static constexpr uint32_t kPieces = kMaximumRows / kEvaluationRows;
  // The least rows of a split of slots that takes no chunk alone, only those
  // a command pipelines (splits()).
  static constexpr uint32_t kPipelinedOnly = kMaximumRows + 1;
  static_assert(kMaximumRows >= kMinimumRows && kMaximumRows % kEvaluationRows == 0,
                "a chunk the split takes is whole pieces, but for its last one's padding");
  // The GPU part's down projection is a view of leading inputs, which only
  // the prefill residual tiles (GgufPrefill, GgufPrefillInt8) run: no chunk
  // the split takes stages.
  static_assert(kMinimumRows > kMaximumDecodeTileRows, "the split's chunks run the prefill residual tiles");

  // The ANE takes `aneUnits` of the units() of intermediate channels, and
  // the GPU the rest through `linear`'s prefill plans. The program is
  // compiled and loaded here, and the wait for the ANE's service ends with
  // ane::Interrupted once `interrupted` returns true. The split takes every
  // chunk from kMinimumRows rows until setMinimumRows(). A command may hold up to
  // `slots` chunks whose FFNs are in flight at once (start() and join()),
  // each slot with its own inputs, output and shared event.
  AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
         uint32_t aneUnits, std::function<bool()> interrupted, uint32_t slots = 1);
  ~AneFfn();
  AneFfn(const AneFfn &) = delete;
  AneFfn &operator=(const AneFfn &) = delete;

  // Why the split does not take `layers`, or null when it does: unrotated
  // quantized projections of one shape (AneFfn.cpp).
  [[nodiscard]] static const char *unsupported(std::span<const SwiGluProjections> layers);
  // The units the split moves intermediate channels in, of `layers`, which
  // it takes: the ANE takes from one to all but one of them.
  [[nodiscard]] static uint32_t units(std::span<const SwiGluProjections> layers);
  // The Metal memory of the split of `layers` with the ANE taking `aneUnits`,
  // over `slots` slots.
  [[nodiscard]] static uint64_t plannedBytes(std::span<const SwiGluProjections> layers, uint32_t aneUnits,
                                             uint32_t slots = 1);
  // What tells this Mac's calibrations of a split of `layers` apart for one
  // macOS version, engine build and most units its plan holds: the layers'
  // shapes and formats, and the Int8Activations of `linear`, whose plans run
  // the GPU's part and so move the units the split takes (Qwen3.8-27B MLX
  // 4-bit's takes 10 of 34 with them on, 15 off, on a 20-core M5 Pro).
  [[nodiscard]] static std::string calibrationKey(std::span<const SwiGluProjections> layers, const Linear &linear);

  // The fraction of intermediate channels the ANE takes.
  [[nodiscard]] double share() const noexcept;
  // What Metal allocated for the split, which plannedBytes() bounds.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }
  // The least rows of a chunk the split takes alone, as the only chunk of its
  // command (ane_ffn::minimumRows): from kMinimumRows to kMaximumRows, or
  // kPipelinedOnly for a split of slots.
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
  // layers in `hidden`, all of which it overwrites: chunks of one piece, of
  // pieces whose last pads and of the most rows (verifiedRows() in
  // AneFfn.cpp) over the first two layers, which stage the ANE's weights
  // through both sets, with the ANE's output filled with NaN before so that
  // rows it leaves unwritten show; each through begin(), the layers encoded
  // as add() encodes them, commit() and its completion check; each against
  // the leading rows of a full chunk on the GPU, which computes every row
  // alike whatever the chunk's rows. Returns the largest error of what the ANE
  // adds to the GPU's part (AneFfn.cpp). Throws ane::Refused if an output is
  // not finite, that error exceeds its bound or the ANE's work fails, which
  // stops the split.
  [[nodiscard]] double verify(std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                              const std::array<metal::MetalBuffer, 2> &hidden);

  // Whether the split takes a chunk of `rows` rows, until it stops: alone,
  // from minimumRows() to kMaximumRows; one of a command's `pipelined`
  // chunks, from kMinimumRows, as the ANE runs its FFN beside the GPU's work
  // on the others.
  [[nodiscard]] bool splits(uint32_t rows, bool pipelined = false) const;
  // Whether the split stopped, and why.
  [[nodiscard]] bool retired() const;
  [[nodiscard]] std::string reason() const;
  // The chunks a command may have in flight at once.
  [[nodiscard]] uint32_t slots() const noexcept { return static_cast<uint32_t>(handoffs_.size()); }

  // Starts encoding a command; false once the split stopped.
  bool begin();
  // Layer `layer`'s FFN of a chunk of `rows` rows, from kMinimumRows to
  // kMaximumRows, encoded in layer order from layer 0 after begin(): output =
  // residual + FFN of ffn.normalized. start(), stageNext() and join() of
  // slot 0.
  void add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
           metal::MetalBuffer output, uint32_t rows);
  // A pipelined command's parts of add(), for chunks in slots: start() packs
  // the chunk's normalized rows for the ANE, raises its slot's event and adds
  // the GPU's part into `output`; join() waits for the ANE's evaluation and
  // adds its part, before anything reads `output`. Each slot's layers start
  // in order from layer 0, one start() and join() each. The GPU runs other
  // chunks' work between them while the ANE runs this one's. Every chunk of a
  // layer starts on weights staged before its first: begin() stages none, the
  // first start() of a layer stages it unless stageNext() did, and
  // stageNext(layer) stages layer + 1 once every chunk of `layer` has
  // started, and every chunk has joined layer - 1, whose staging set it
  // overwrites.
  // With `norm`, which only a split whose ANE takes every channel takes
  // (fusesNorm()), start() reads `residual` and normalizes it by `norm` as it
  // packs it, and ffn.normalized stays unwritten: no GPU part reads it.
  void start(metal::CommandGraph &graph, uint32_t layer, uint32_t slot, const PrefillFfnBuffers &ffn,
             metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows,
             const NormWeights *norm = nullptr);
  [[nodiscard]] bool fusesNorm() const noexcept { return shape_.gpu == 0; }
  void join(metal::CommandGraph &graph, uint32_t slot);
  void stageNext(metal::CommandGraph &graph, uint32_t layer);
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
  // A piece's surfaces: its rows' rotated inputs in int8 segments, their
  // per-token scales and the ANE's partial down projection, which the
  // function reads or writes, of kEvaluationRows rows, whose row stride the
  // function declares.
  struct Piece final {
    std::vector<ane::Surface> inputs;
    ane::Surface tokenScale, partial;
  };
  // A slot's pieces, kPieces of them, the first rows of its chunk first.
  struct Slot final {
    std::vector<Piece> pieces;
  };
  // Everything the split allocates (allocate() in AneFfn.cpp): each slot's
  // surfaces, and the scratch and weight sets the slots share. A join that
  // reads a partial value or token scale that is not finite sets the status
  // word (ane_ffn_join).
  struct Memory final {
    metal::MetalBuffer signs, rowScales, rotated, status;
    std::vector<Slot> slots;
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
  // A layer, held by value: the planes of its gate, up and down, from which
  // the GPU stages the ANE's int8 weights, and the GPU's share of it: gate
  // and up over views of their leading rows, down over the leading inputs of
  // its rows; none when the ANE takes every channel.
  struct Layer final {
    std::array<Planes, 3> planes;
    std::optional<Projection> gate, up, down;
  };
  // An input of the ANE program's function: its name, the surface each
  // weight set binds to it, and the rows and width the function reads of it,
  // a width of 0 reading a piece's rows.
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
  // An evaluation of piece `piece` of slot `slot` on weight set `set`: it
  // starts once the slot's event reaches `ready`, the GPU's or the piece
  // before's, and raises it to `done`. A chunk's first piece holds its rows,
  // whole pieces; the others none.
  struct Job final {
    uint32_t set, slot, piece, rows;
    uint64_t ready, done;
  };
  // A slot's chunk between start() and join(): the value of its event the
  // join waits for (none for the GPU's part alone), and the output and rows
  // it joins into.
  struct Pending final {
    bool active = false, ane = false;
    uint64_t done = 0;
    metal::MetalBuffer output, residual;
    uint32_t rows = 0;
  };
  // What feeds a committed command's jobs to the ANE, at most kQueueDepth
  // outstanding (the service refuses its 128th outstanding request): a
  // thread that queues the next job whenever one reports (AneFfn.cpp).
  struct Feeder;

  [[nodiscard]] static Shape shapeOf(std::span<const SwiGluProjections> layers, uint32_t aneUnits);
  // Every allocation of the split over `slots` slots, which `buffer` and
  // `surface` make: plannedBytes() counts them and the constructor allocates
  // them.
  template <class MakeBuffer, class MakeSurface>
  [[nodiscard]] static Memory allocate(const Shape &shape, uint32_t slots, MakeBuffer &&buffer,
                                       MakeSurface &&surface);
  // The program's inputs for piece `piece` of slot `slot`, and the program
  // over them: its one function, of kEvaluationRows rows.
  [[nodiscard]] static std::vector<Input> programInputs(const Shape &shape, const Memory &memory, uint32_t slot,
                                                        uint32_t piece);
  [[nodiscard]] static std::string program(const Shape &shape, std::span<const Input> inputs,
                                           const ane::Surface &output);
  [[nodiscard]] static std::string function(const Shape &shape, std::span<const Input> inputs,
                                            const ane::Surface &output, uint32_t rows);
  // Fills the CPU-visible bf16 rows `normalized` with the same values of a
  // normalized row's magnitude each time.
  static void fillNormalized(const metal::MetalBuffer &normalized);
  void encode(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn, metal::MetalBuffer residual,
              metal::MetalBuffer output, uint32_t rows, Parts parts);
  void startIn(metal::CommandGraph &graph, uint32_t layer, uint32_t slot, const PrefillFfnBuffers &ffn,
               metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, Parts parts,
               const NormWeights *norm = nullptr);
  // A view of `count` hidden rows of `buffer` from row `first`, in bf16.
  [[nodiscard]] metal::MetalBuffer hiddenRows(const metal::MetalBuffer &buffer, uint32_t first, uint32_t count) const;
  // Stages layer `layer`'s weights unless this command did.
  void stage(metal::CommandGraph &graph, uint32_t layer);
  void addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const;
  [[nodiscard]] metal::MetalBuffer rowScales(uint32_t layer, Matrix matrix) const;
  // Starts `job` on the ANE through its slot's handoff; `reported` runs once
  // the ANE has reported it, or at once if it never starts.
  void queue(const Job &job, std::function<void()> reported);
  // Each slot's handoff, in turn; retires every slot's with `reason`.
  void retireAll(const std::string &reason);
  // Cancels the command committed last on every slot.
  void cancelAll(const std::string &reason);
  // finish() but for its log line and judgment: if every evaluation of the
  // command committed last succeeded and its joins read finite values, the
  // Neural Engine's time over them (ane::Handoff::finish); none otherwise,
  // which stops the split.
  [[nodiscard]] std::optional<AwakeClock::duration> completed();
  // Logs, once, that the split stopped.
  void warnStopped();

  metal::MetalBackend &backend_;
  const Linear linear_;
  Shape shape_;
  uint32_t minimumRows_ = kMinimumRows;
  std::vector<Layer> layers_;
  Memory memory_;
  std::unique_ptr<ane::Program> program_;
  // The function over each piece's surfaces with each weight set, by
  // (set * slots + slot) * kPieces + piece.
  std::vector<ane::Program::Binding> bindings_;
  // The jobs of the command being encoded, which commit() starts; the layer
  // each slot adds next and its chunk between start() and join(); and the
  // layers whose weights the command staged, the last one's.
  std::vector<Job> jobs_;
  std::vector<uint32_t> nextLayer_;
  std::vector<Pending> pending_;
  std::optional<uint32_t> staged_;
  // The chunks' evaluations of the command committed last, a layer's each,
  // and their rows (the most, whole pieces), and whether more than one slot
  // ran them.
  uint32_t committedEvaluations_ = 0, committedRows_ = 0;
  bool committedSlots_ = false;
  ane_ffn::Breaker breaker_;
  std::function<void()> lost_;
  Served served_;
  // Whether a command committed jobs that finish() has not yet judged.
  bool unfinished_ = false;
  // From a release() that unloaded the program until restore().
  bool released_ = false;
  bool warned_ = false;
  uint64_t allocatedBytes_ = 0;
  // Destroyed first: each waits for the evaluations the ANE has not reported
  // of its slot, once the destructor has stopped the feeder.
  std::shared_ptr<Feeder> feeder_;
  std::vector<std::unique_ptr<ane::Handoff>> handoffs_;
};

} // namespace splash::ops

#pragma once

#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace splash::ops {

// One layer's affine SwiGLU projections: down(silu(gate x) * up x).
struct SwiGluProjections final {
  const Projection *gate = nullptr;
  const Projection *up = nullptr;
  const Projection *down = nullptr;
};

// The dense FFN of a prefill chunk split by intermediate channel between the
// GPU and the Neural Engine. The GPU runs the leading channels with the affine
// Q4 prefill kernels; the ANE runs the rest as one W8A8 program over 2048
// rows, with int8 weights the GPU requantizes from the Q4 planes one layer
// ahead into double-buffered surfaces. The GPU adds the ANE's partial down
// projection to its own. A shared event orders each layer's ANE evaluation
// between the GPU's input packing and that join, within the one command.
class AneFfn final {
public:
  // The rows of every ANE evaluation, and the fewest a chunk needs for the
  // split to beat the GPU alone; smaller chunks keep the whole FFN on the GPU.
  static constexpr uint32_t kRows = 2048;
  static constexpr uint32_t kMinimumRows = 512;

  // `share` is the fraction of intermediate channels the ANE takes.
  AneFfn(metal::MetalBackend &backend, const Linear &linear, std::span<const SwiGluProjections> layers,
         double share);
  ~AneFfn();
  AneFfn(const AneFfn &) = delete;
  AneFfn &operator=(const AneFfn &) = delete;

  // The Metal memory of a split of `layers` layers of `hidden` x
  // `intermediate` projections at `share`.
  [[nodiscard]] static uint64_t plannedBytes(uint32_t layers, uint32_t hidden, uint32_t intermediate, double share);
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }

  // Starts encoding a command, discarding the jobs of one never submitted.
  void begin() noexcept { jobs_.clear(); }
  // Layer `layer`'s FFN of `rows` rows, which QwenTarget encodes in layer
  // order from layer 0 within each command: output = residual + FFN of
  // `normalized` (whose Q4 sums the norm wrote). The scratch buffers are the
  // prefill arena's dense FFN buffers.
  void add(metal::CommandGraph &graph, uint32_t layer, metal::MetalBuffer normalized, metal::MetalBuffer sums,
           metal::MetalBuffer gateScratch, metal::MetalBuffer intermediate, metal::MetalBuffer downSums,
           metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, LinearScratch scratch);

  // Queues the ANE evaluations of the command encoded since the last submit,
  // before the command is committed. If one cannot be queued, those queued
  // are released and drained, and the command must not be committed.
  void submit();
  // After the command has completed: waits for its evaluations to report,
  // and throws if one failed.
  void finish();

private:
  struct Weights final {
    // The ANE's rows of gate and up in two input segments, down in segments
    // of its inputs, and the shared per-row scale of each.
    std::vector<ane::Surface> gate, up, down;
    ane::Surface gateScale, upScale, downScale;
  };
  struct Layer final {
    SwiGluProjections source;
    // The GPU's share: gate and up views of the leading rows, and down's
    // leading inputs repacked.
    Projection gate, up, down;
  };
  struct Job final {
    uint32_t set;
    uint64_t wait, signal;
  };
  // Evaluations complete in the order they are queued.
  struct Evaluations final {
    std::mutex mutex;
    std::condition_variable changed;
    uint64_t completed = 0;
    bool failed = false;
  };

  void addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const;
  [[nodiscard]] metal::MetalBuffer rowScales(uint32_t layer, uint32_t part) const;
  // Waits for the queued evaluations, releasing each one's wait first when
  // `release` (no Metal command will signal it).
  [[nodiscard]] bool wait(bool release);

  metal::MetalBackend &backend_;
  const Linear &linear_;
  uint32_t hidden_ = 0, intermediate_ = 0, gpuChannels_ = 0, aneChannels_ = 0;
  std::vector<uint32_t> downSegments_;
  std::vector<Layer> layers_;
  metal::MetalBuffer signs_, rowScales_, rotated_;
  std::vector<ane::Surface> inputs_;
  ane::Surface tokenScale_, partial_;
  std::array<Weights, 2> sets_;
  std::unique_ptr<ane::Program> program_;
  std::array<std::vector<ane::Surface>, 2> bindings_;
  metal::SharedEvent event_;
  uint64_t value_ = 0;
  // Encoded but not queued, and queued (the queued-th evaluation is the
  // last of queued_).
  std::vector<Job> jobs_, queued_;
  uint64_t queuedCount_ = 0;
  std::shared_ptr<Evaluations> evaluations_ = std::make_shared<Evaluations>();
  uint64_t allocatedBytes_ = 0;
};

} // namespace splash::ops

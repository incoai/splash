#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "metal/CommandGraph.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace splash::ops {

// Immutable views of one packed Q4 projection.  StorageN is part of the
// package ABI; the operator may choose a different compute tile at runtime.
// One GGUF GGUF tensor occupying output columns [columnOffset,
// columnOffset + outputSize) of a projection. Layouts and kernels live in
// metal/abi/Gguf.h and kernels/shared/gguf_linear.metal.
struct GgufSegment final {
  metal::MetalBuffer plane0;
  metal::MetalBuffer plane1;
  metal::MetalBuffer meta;
  uint32_t type = 0;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  uint32_t p0 = 0;
  uint32_t p1 = 0;
  uint32_t metaBytes = 0;
  uint32_t metaGroups = 0;
  uint32_t columnOffset = 0;
  uint32_t formatId = 0;    // GGUF_FMT_* (metal/abi/QuantFormat.h)
  const char *format = "";
};

struct Q4Projection final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  // Non-empty: a GGUF GGUF projection; weights/scales/biases are unused.
  std::vector<GgufSegment> gguf{};
  metal::MetalBuffer kqPartials{};    // fp32 split-K partials scratch (shared)
  metal::MetalBuffer kqCounters{};    // split-K arrival counters (one uint per 64-column tile, zero at rest)
};

// Q8 affine projections use per-64-input quantization and StorageN=256 order.
// Used by the MoE router and shared-expert gate.
struct Q8Projection final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
};

// Expert-major Q4 slabs keep one complete StorageN-packed projection per
// expert. The operator selects expertStrideBytes directly; no per-expert
// MetalBuffer objects or weight copies are created at runtime.
struct ExpertQ4Projection final {
  metal::MetalBuffer packed;
  uint32_t experts = 0;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  uint64_t expertStrideBytes = 0;
};

struct LinearMatrix final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  auto operator<=>(const LinearMatrix &) const = default;
};

enum class LinearPhase : uint8_t { Prefill, Decode };
enum class LinearEpilogue : uint8_t { None, Residual, GateUp, UpWithGate };
// Compute tiles over the StorageN=256 packing. Paired tiles pipeline two
// quant groups of one lane. Split tiles keep one 8-row tile per threadgroup
// and split K into four partitions whose fp32 partial sums are reduced before
// the bf16 rounding; they take one lane, K % 1024 == 0 and one threadgroup
// per tile. Paired256 is the four-simdgroup N256 paired tile. Simdgroup
// uses bf16 8x8 matrix operations and an explicit activation/split workspace.
enum class LinearTile : uint8_t {
  N128, N256, Paired128, Split32, Split64, Paired256, Simdgroup
};
enum class LinearSimdgroups : uint8_t { Four = 4, Eight = 8 };

struct LinearWorkload final {
  LinearMatrix matrix;
  uint32_t rows = 0;
  LinearPhase phase = LinearPhase::Decode;
  LinearEpilogue epilogue = LinearEpilogue::None;
  auto operator<=>(const LinearWorkload &) const = default;
};

struct LinearConfig final {
  LinearTile tile = LinearTile::N128;
  // Decode grid size. Prefill uses its matrix grid and requires zero here.
  uint32_t groups = 0;
  // Simdgroups per threadgroup, independent of the persistent grid size: the
  // cooperative scope of one tile, or for split tiles the four partitions
  // together (Split32 is 4 x 1, Split64 is 4 x 2; Paired256 runs four).
  LinearSimdgroups simdgroups = LinearSimdgroups::Eight;
  // Cross-threadgroup K partitions for Simdgroup; all other tiles use one.
  uint32_t splits = 1;
  bool operator==(const LinearConfig &) const = default;
};

struct LinearChoice final {
  LinearWorkload workload;
  LinearConfig configuration;
};

// Reused serially within one decode command stream. Counters are zeroed at
// allocation and restored by each completed split dispatch. Never share this
// workspace between concurrent command streams. Within a batched dispatch,
// each eight-row tile owns disjoint input, sums, partials and counters.
struct LinearScratch final {
  metal::MetalBuffer input;
  metal::MetalBuffer sums;
  metal::MetalBuffer partials;
  metal::MetalBuffer counters;
};
struct LinearScratchSize final {
  uint64_t input = 0, sums = 0, partials = 0, counters = 0;
  [[nodiscard]] uint64_t bytes() const noexcept { return input + sums + partials + counters; }
};

// The activation layout a decode plan reads: the producer's bf16 rows, or an
// X^T table with fp32 row sums in LinearScratch that a producer can emit
// alongside its ordinary output.
enum class LinearInput : uint8_t {
  Plain,    // bf16 [rows][K]
  Table64,  // affine simdgroup table, one sum per 64 inputs (kernels/common/q4_sgmatrix.h)
};
// The scratch table currently holds `source` in `layout`. Plain means the
// scratch describes nothing. Producers return it, consumers accept it and
// return what the scratch describes after their dispatch.
struct PreparedInput final {
  metal::MetalBuffer source;
  LinearInput layout = LinearInput::Plain;
};

class LinearPlan final {
public:
  [[nodiscard]] LinearWorkload workload() const noexcept { return workload_; }
  [[nodiscard]] LinearConfig configuration() const noexcept { return config_; }
  [[nodiscard]] uint32_t storageRows() const noexcept;
  [[nodiscard]] uint32_t tileColumns() const noexcept;
  [[nodiscard]] uint32_t threadsPerThreadgroup() const noexcept;
  // fp32 partial sums the kernel reduces before the single bf16 rounding of
  // the projection: 1 for the sequential tiles, whose outputs are bitwise
  // identical for a workload; 4 for split tiles; 1-8 for Simdgroup. The latter
  // also reassociates within each quantization group, even with one split.
  [[nodiscard]] uint32_t partialSums() const noexcept;
  [[nodiscard]] bool usesSimdgroup() const noexcept;
  [[nodiscard]] LinearInput input() const noexcept;
  [[nodiscard]] LinearScratchSize scratchSize() const noexcept;
  [[nodiscard]] uint64_t sumsBytes() const noexcept;
  [[nodiscard]] uint64_t gateScratchBytes() const noexcept;
  [[nodiscard]] uint64_t downSumsBytes() const noexcept;
  [[nodiscard]] std::string_view pipeline() const noexcept { return pipeline_; }
  [[nodiscard]] std::string_view secondPipeline() const noexcept {
    return secondPipeline_;
  }

private:
  friend class Q4Linear;
  LinearPlan(LinearWorkload workload, LinearConfig config);
  LinearWorkload workload_;
  LinearConfig config_;
  std::string_view pipeline_;
  std::string_view secondPipeline_;
};

// The plan defines which fields are used and how much scratch they require.
struct LinearBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer output;
  metal::MetalBuffer sums;
  metal::MetalBuffer residual;
  metal::MetalBuffer gateScratch;
  metal::MetalBuffer downSums;
  LinearScratch scratch{};
  // What the scratch table holds (for example after fused RMSNorm). A plan
  // that reads a table prepares one unless this describes its input.
  PreparedInput prepared{};
};

struct Q4DispatchStats final {
  uint64_t fusedSourceOperations = 0;
  uint64_t m16Dispatches = 0;
  uint64_t m24Dispatches = 0;
  uint64_t m32Dispatches = 0;
};

// Owns Q4 pipeline selection and dispatch. Device policy uses GPU family,
// core count and workload tile counts.
class Q4Linear final {
public:
  explicit Q4Linear(const DeviceCapabilities &device) noexcept;

  // One lane: at most 3 tiles * 4 group counts, 2 split tiles, 2 paired
  // N256 grids, and 4 Apple9 simdgroup K splits (including its baseline):
  // 3 * 4 + 2 + 2 + 4 = 20. Other families have no simdgroup candidates
  // and at most one additional baseline (17). M24 replaces Paired128 with
  // N128/four-simdgroup candidates and adds up to four matrix K splits,
  // with no one-lane tiles (at most 17).
  static constexpr std::size_t kMaximumCandidates = 20;

  [[nodiscard]] LinearPlan plan(LinearWorkload workload) const;
  // The layout the decode plan of this projection reads, for its producer.
  [[nodiscard]] LinearInput decodeInput(const Q4Projection &projection, uint32_t lanes,
                                        LinearEpilogue epilogue = LinearEpilogue::None) const;
  [[nodiscard]] LinearScratchSize decodeScratchSize(LinearWorkload workload) const;
  [[nodiscard]] static LinearPlan plan(LinearWorkload workload, LinearConfig config);
  [[nodiscard]] std::vector<LinearPlan> candidates(LinearWorkload workload) const;
  // Installed only at startup; encoding does a read-only lookup, never tuning.
  void setChoices(std::span<const LinearChoice> choices);
  // Returns what the scratch table describes after the dispatch.
  PreparedInput add(metal::CommandGraph &graph, LinearBuffers buffers,
                    const Q4Projection &projection, const LinearPlan &plan,
                    const Q4Projection *gate = nullptr,
                    Q4DispatchStats *stats = nullptr) const;

  void addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input,
                      metal::MetalBuffer sums, LinearMatrix matrix,
                      uint32_t rows) const;
  void addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input,
                  const Q4Projection &projection, metal::MetalBuffer output,
                  metal::MetalBuffer sums, LinearMatrix matrix,
                  uint32_t rows) const;
  void addPrefillUpWithGate(
      metal::CommandGraph &graph, metal::MetalBuffer input,
      const Q4Projection &up, metal::MetalBuffer gateScratch,
      metal::MetalBuffer output, metal::MetalBuffer sums,
      metal::MetalBuffer downSums, LinearMatrix matrix,
      uint32_t rows) const;
  void addPrefillResidual(metal::CommandGraph &graph,
                          metal::MetalBuffer input,
                          const Q4Projection &projection,
                          metal::MetalBuffer residual,
                          metal::MetalBuffer output, metal::MetalBuffer sums,
                          LinearMatrix matrix, uint32_t rows) const;

  PreparedInput addDecode(metal::CommandGraph &graph,
                          metal::MetalBuffer input, const Q4Projection &projection,
                          metal::MetalBuffer output, LinearMatrix matrix,
                          LinearScratch scratch = {}) const;
  PreparedInput addDecodeBatch(metal::CommandGraph &graph,
                               metal::MetalBuffer input,
                               const Q4Projection &projection,
                               metal::MetalBuffer output, LinearMatrix matrix,
                               uint32_t lanes, Q4DispatchStats &stats,
                               LinearScratch scratch = {}, PreparedInput prepared = {}) const;
  PreparedInput addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                               const Q4Projection &gate, const Q4Projection &up,
                               metal::MetalBuffer gateScratch,
                               metal::MetalBuffer output, LinearMatrix matrix,
                               uint32_t lanes, Q4DispatchStats &stats,
                               LinearScratch scratch = {}, PreparedInput prepared = {}) const;
  PreparedInput addResidualBatch(metal::CommandGraph &graph,
                                 metal::MetalBuffer input,
                                 const Q4Projection &projection,
                                 metal::MetalBuffer residual,
                                 metal::MetalBuffer output, LinearMatrix matrix,
                                 uint32_t lanes, Q4DispatchStats &stats,
                                 LinearScratch scratch = {}, PreparedInput prepared = {}) const;

private:
  [[nodiscard]] LinearConfig baseline(LinearWorkload workload) const;
  uint32_t appleGpuFamily_ = 0;
  uint32_t gpuCores_ = 0;
  std::vector<LinearChoice> choices_;
};

} // namespace splash::ops

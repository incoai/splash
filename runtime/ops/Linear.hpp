#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/Weights.hpp"

#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::ops {

// The instance of kernel `name` that writes `destination`: a plain decode
// kernel's fp32 instance is "<name>_f32".
[[nodiscard]] inline std::string kernelInstance(std::string_view name, FloatOutput destination) {
  return std::string(name) + (destination == FloatOutput::Float32 ? "_f32" : "");
}
// The tile of a float projection (kernels/shared/gguf_float.metal): fp32
// simdgroup MMA on the weights as stored, or the neural accelerator's bf16
// matmul on each weight's three bf16 parts, which sum to it exactly. Both
// round only in fp32 accumulation; Linear::ggufFloatTile picks one.
enum class FloatTile : uint8_t { Simdgroup, NeuralAccelerator };
// out[r][outOffset + n] = sum_k input[r][k] W[n][k] for rows r < `rows` of a
// float segment, into a destination of `outStride` columns (LinearGguf.cpp).
void addGgufFloat(metal::CommandGraph &graph, metal::MetalBuffer input, const QuantizedSegment &weights,
                  metal::MetalBuffer output, uint32_t rows, uint32_t outStride, uint32_t outOffset,
                  FloatOutput type, FloatTile tile);

struct LinearMatrix final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  auto operator<=>(const LinearMatrix &) const = default;
};

// Throws unless `projection` is an affine projection of `matrix` whose planes
// hold all of its Q4 weights, scales and biases.
void requireAffineProjection(const Projection &projection, LinearMatrix matrix);

enum class LinearPhase : uint8_t { Prefill, Decode };
enum class LinearEpilogue : uint8_t { None, Residual, GateUp, UpWithGate };
// Compute tiles over the StorageN=256 packing. Paired tiles pipeline two
// quant groups of one lane. Split tiles keep one 8-row tile per threadgroup
// and split K into four partitions whose fp32 partial sums are reduced before
// the bf16 rounding; they take one lane, K % 1024 == 0 and one threadgroup
// per tile. Paired256 is the four-simdgroup N256 paired tile. Simdgroup
// uses bf16 8x8 matrix operations and an explicit activation/split workspace.
// GgufStaged dequantizes GGUF weights per simdgroup into threadgroup memory
// for matmul2d: 64 columns per decode threadgroup (two simdgroups) of 8, 16
// or 32 rows with optional K splits; prefill runs 128-row tiles, or the
// decode tiles for chunks of up to 32 rows. GgufRegister is the exact
// register tile on bf16 8x8 matrix operations (Apple9): 64 columns per
// threadgroup, every request lane in one threadgroup, optional K splits.
enum class LinearTile : uint8_t {
  N128, N256, Paired128, Split32, Split64, Paired256, Simdgroup, GgufStaged, GgufRegister
};
// The GGUF formats Apple9's staged tiles decode faster than its register
// tiles, dense and MoE: IQ3_XXS, the IQ2 formats and IQ1, whose operands the
// register tiles build from grid lookups beside their matrix operations
// (LinearGguf.cpp, MoE.hpp).
[[nodiscard]] bool apple9StagesFormat(uint32_t format) noexcept;
enum class LinearSimdgroups : uint8_t { Two = 2, Four = 4, Eight = 8 };

struct LinearWorkload final {
  LinearMatrix matrix;
  uint32_t rows = 0;
  LinearPhase phase = LinearPhase::Decode;
  LinearEpilogue epilogue = LinearEpilogue::None;
  WeightLayout weightLayout = WeightLayout::Affine64;
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
  // Cross-threadgroup K partitions for Simdgroup and the GGUF tiles, a power
  // of two up to kMaximumSplits; all other tiles use one.
  uint32_t splits = 1;
  static constexpr uint32_t kMaximumSplits = 8;
  [[nodiscard]] constexpr bool validSplits() const noexcept {
    return splits && splits <= kMaximumSplits && !(splits & (splits - 1));
  }
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
  // The bf16 input rows a rotated projection's quantized segments read
  // (ProjectionShape::rotated): rotatedBytes() of its plan.
  metal::MetalBuffer rotated{};
};
struct LinearScratchSize final {
  uint64_t input = 0, sums = 0, partials = 0, counters = 0, rotated = 0;
  [[nodiscard]] uint64_t bytes() const noexcept { return input + sums + partials + counters + rotated; }
  // Grows each field to hold `other`'s too.
  LinearScratchSize &include(const LinearScratchSize &other) noexcept {
    input = std::max(input, other.input);
    sums = std::max(sums, other.sums);
    partials = std::max(partials, other.partials);
    counters = std::max(counters, other.counters);
    rotated = std::max(rotated, other.rotated);
    return *this;
  }
};
// LinearScratch::rotated bytes of a rotated projection's plan of storageRows
// rows of `width` inputs.
[[nodiscard]] constexpr uint64_t rotatedBytes(uint32_t width, uint64_t storageRows) noexcept {
  return uint64_t{width} * storageRows * 2;
}

// The activation layout a decode plan reads: the producer's bf16 rows, or an
// X^T table with fp32 row sums in LinearScratch that a producer can emit
// alongside its ordinary output.
enum class LinearInput : uint8_t {
  Plain,    // bf16 [rows][K]
  Table64,  // affine simdgroup table, one sum per 64 inputs (kernels/common/q4_sgmatrix.h)
  Table16,  // GGUF simdgroup table, sums per 16 and 32 inputs (kernels/common/gguf_sgmatrix.h)
};
// Scratch bytes a producer writes for `rows` rows of `width` inputs.
[[nodiscard]] constexpr uint64_t tableBytes(uint32_t width, uint64_t rows) noexcept {
  return uint64_t{width} * rows * 2;
}
[[nodiscard]] uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept;
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
  [[nodiscard]] FloatOutput destination() const noexcept { return destination_; }
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
  // The affine tile's kernel; the plan runs its kernelInstance for destination().
  [[nodiscard]] std::string_view pipeline() const noexcept { return pipeline_; }
  [[nodiscard]] std::string_view secondPipeline() const noexcept {
    return secondPipeline_;
  }

private:
  friend class Linear;
  LinearPlan(LinearWorkload workload, LinearConfig config, FloatOutput destination = FloatOutput::BFloat16);
  // Block plans (LinearGguf.cpp).
  void requireBlockConfiguration() const;
  [[nodiscard]] uint32_t blockStorageRows() const noexcept;
  [[nodiscard]] LinearScratchSize blockScratchSize() const noexcept;
  LinearWorkload workload_;
  LinearConfig config_;
  FloatOutput destination_;
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

struct LinearDispatchStats final {
  uint64_t fusedSourceOperations = 0;
  uint64_t m16Dispatches = 0;
  uint64_t m24Dispatches = 0;
  uint64_t m32Dispatches = 0;
};

// Owns projection pipeline selection and dispatch for both weight layouts.
// Device policy uses GPU family, core count and workload tile counts.
class Linear final {
public:
  explicit Linear(const DeviceCapabilities &device) noexcept;

  // One lane: at most 3 tiles * 4 group counts, 2 split tiles, 2 paired
  // N256 grids, and 4 Apple9 simdgroup K splits (including its baseline):
  // 3 * 4 + 2 + 2 + 4 = 20. Other families have no simdgroup candidates
  // and at most one additional baseline (17). M24 replaces Paired128 with
  // N128/four-simdgroup candidates and adds up to four matrix K splits,
  // with no one-lane tiles (at most 17).
  static constexpr std::size_t kMaximumCandidates = 20;

  [[nodiscard]] LinearPlan plan(LinearWorkload workload) const;
  // The plan of `workload` in the projection's weight layout, into its destination type; a gate/up plan also runs
  // `gate`.
  [[nodiscard]] LinearPlan plan(LinearWorkload workload, const Projection &projection,
                                const Projection *gate = nullptr) const;
  // Rows of storage a decode step of `rows` rows binds for a projection of
  // `shape`: the storageRows of its decode plans, which every epilogue shares.
  [[nodiscard]] uint32_t decodeStorageRows(uint32_t rows, ProjectionShape shape) const;
  // The plans of this projection's matrix in its layout. A decode plan's
  // input() is the layout its producer writes.
  [[nodiscard]] LinearPlan prefillPlan(const Projection &projection, uint32_t rows,
                                       LinearEpilogue epilogue) const;
  [[nodiscard]] LinearPlan decodePlan(const Projection &projection, uint32_t lanes,
                                      LinearEpilogue epilogue = LinearEpilogue::None,
                                      const Projection *gate = nullptr) const;
  [[nodiscard]] LinearScratchSize decodeScratchSize(LinearWorkload workload) const;
  // The scratch of every prefill chunk and epilogue of a projection of
  // `shape`: the split partials and counters of the chunks that run the GGUF
  // decode tiles (LinearGguf.cpp).
  [[nodiscard]] LinearScratchSize prefillScratchSize(ProjectionShape shape) const;
  // The tile of a float projection of `rows` rows into `outputSize` columns
  // on this device (LinearGguf.cpp).
  [[nodiscard]] FloatTile ggufFloatTile(uint32_t rows, uint32_t outputSize) const noexcept;
  [[nodiscard]] static LinearPlan plan(LinearWorkload workload, LinearConfig config,
                                       FloatOutput destination = FloatOutput::BFloat16);
  [[nodiscard]] std::vector<LinearPlan> candidates(LinearWorkload workload) const;
  // Installed only at startup; encoding does a read-only lookup, never tuning.
  // Block projection plans are not tuned: their workloads take no choice.
  void setChoices(std::span<const LinearChoice> choices);
  // Returns what the scratch table describes after the dispatch.
  PreparedInput add(metal::CommandGraph &graph, LinearBuffers buffers,
                    const Projection &projection, const LinearPlan &plan,
                    const Projection *gate = nullptr,
                    LinearDispatchStats *stats = nullptr) const;

  // The Q4 input sums of `rows` rows an affine prefill projection reads.
  void addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer sums,
                      const Projection &consumer, uint32_t rows) const;
  // The projections of `rows` rows through their own matrix. `scratch` holds
  // the partials and counters of split plans (GGUF chunks of up to 32 rows);
  // reused serially within one command stream, as in decode.
  void addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                  metal::MetalBuffer output, metal::MetalBuffer sums, uint32_t rows,
                  LinearScratch scratch = {}) const;
  void addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &up,
                            metal::MetalBuffer gateScratch, metal::MetalBuffer output, metal::MetalBuffer sums,
                            metal::MetalBuffer downSums, uint32_t rows, LinearScratch scratch = {}) const;
  void addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                          metal::MetalBuffer residual, metal::MetalBuffer output, metal::MetalBuffer sums,
                          uint32_t rows, LinearScratch scratch = {}) const;

  PreparedInput addDecode(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                          metal::MetalBuffer output, LinearScratch scratch = {}) const;
  PreparedInput addDecodeBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                               const Projection &projection, metal::MetalBuffer output, uint32_t lanes,
                               LinearDispatchStats &stats, LinearScratch scratch = {},
                               PreparedInput prepared = {}) const;
  PreparedInput addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &gate,
                               const Projection &up, metal::MetalBuffer gateScratch, metal::MetalBuffer output,
                               uint32_t lanes, LinearDispatchStats &stats, LinearScratch scratch = {},
                               PreparedInput prepared = {}) const;
  PreparedInput addResidualBatch(metal::CommandGraph &graph, metal::MetalBuffer input,
                                 const Projection &projection, metal::MetalBuffer residual,
                                 metal::MetalBuffer output, uint32_t lanes, LinearDispatchStats &stats,
                                 LinearScratch scratch = {}, PreparedInput prepared = {}) const;

private:
  // The device's configuration of the workload; a block plan's tile may follow the formats of the projections it
  // runs.
  [[nodiscard]] LinearConfig baseline(LinearWorkload workload,
                                      std::span<const Projection *const> projections = {}) const;
  // Counts `dispatches` dispatches that each fuse `lanes` request lanes.
  static void account(LinearDispatchStats &stats, uint32_t lanes, uint32_t dispatches) noexcept;
  // GGUF policy and dispatch (LinearGguf.cpp). Block plans are not tuned.
  [[nodiscard]] LinearConfig ggufBaseline(LinearWorkload workload,
                                          std::span<const Projection *const> projections) const;
  // The scratch of every tile a block decode plan of the workload may take.
  [[nodiscard]] LinearScratchSize ggufDecodeScratchSize(LinearWorkload workload) const;
  void addGguf(metal::CommandGraph &graph, const LinearBuffers &buffers,
               const Projection &projection, const LinearPlan &plan,
               const Projection *gate, LinearDispatchStats *stats) const;
  void addGgufStaged(metal::CommandGraph &graph, const LinearBuffers &buffers,
                     const Projection &projection, const LinearPlan &plan,
                     const Projection *gate) const;
  void addGgufRegister(metal::CommandGraph &graph, const LinearBuffers &buffers,
                       const Projection &projection, const LinearPlan &plan,
                       const Projection *gate) const;
  void addGgufFloatSegments(metal::CommandGraph &graph, const LinearBuffers &buffers,
                            const Projection &projection, const LinearPlan &plan) const;
  uint32_t appleGpuFamily_ = 0;
  uint32_t gpuCores_ = 0;
  std::vector<LinearChoice> choices_;
};

} // namespace splash::ops

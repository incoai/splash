#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Weights.hpp"

#include <algorithm>
#include <array>
#include <compare>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

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
// Throws unless the planes of the quantized `segment` hold every tile of its
// outputSize x inputSize weights (metal/abi/QuantFormat.h), naming them
// "<what> plane0", "<what> plane1" and "<what> meta".
void requireSegmentPlanes(const QuantizedSegment &segment, std::string_view what);

enum class LinearPhase : uint8_t { Prefill, Decode };
enum class LinearEpilogue : uint8_t { None, Residual, GateUp, UpWithGate };
// Compute tiles over the affine storage tiles. Paired tiles pipeline two
// quant groups of one lane. Split128 is the N128 tile with K split across
// `splits` threadgroups, grid (column tiles, splits), every lane's rows in each
// tile; the last threadgroup of a tile to finish reduces the fp32 partial sums
// before the bf16 rounding. Paired256 is the four-simdgroup N256 paired tile.
// Q4Register is the register tile on bf16 8x8 matrix operations (Apple9),
// over an activation table and with optional K splits in an explicit
// workspace.
// The GGUF tiles run 64 columns per threadgroup. The staged tiles, GgufStaged
// and GgufPrefill, dequantize GGUF weights into threadgroup memory for
// matmul2d. GgufStaged: the two-simdgroup staged tile of 8, 16 or 32 rows
// (decode, and prefill chunks of up to 32 rows), each simdgroup staging its
// own columns, with optional K splits. GgufPrefill: the 128-row shared-stage
// prefill tile. GgufRegister is Q4Register's twin over GGUF weights, exact:
// every request lane in one threadgroup, optional K splits.
enum class LinearTile : uint8_t {
  N128, N256, Paired128, Split128, Paired256, Q4Register, GgufStaged, GgufPrefill, GgufRegister
};
// The decode tiles hold at most a full decode batch; GGUF prefill chunks of up
// to this many rows run the staged tile (Linear::ggufBaseline).
inline constexpr uint32_t kMaximumDecodeTileRows = SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS;
// The GGUF formats Apple9's staged tiles decode faster than its register
// tiles, dense and MoE: IQ3_XXS, the IQ2 formats and IQ1, whose operands the
// register tiles build from grid lookups beside their matrix operations
// (LinearGguf.cpp, MoE.hpp).
[[nodiscard]] bool apple9StagesFormat(uint32_t format) noexcept;
enum class LinearSimdgroups : uint8_t { Four = 4, Eight = 8 };

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
  // Persistent threadgroups of the N128, N256, Paired128 and Paired256 decode
  // tiles (1 to their column tiles); 0 for every other plan, whose grid
  // covers the matrix.
  uint32_t groups = 0;
  // Simdgroups of an affine Q4 tile's threadgroup, independent of the
  // persistent grid size: the cooperative scope of one tile (Paired256 runs
  // four). The GGUF tiles fix their threadgroups in their kernels
  // (GGUF_*_THREADS) and leave this at its default.
  LinearSimdgroups simdgroups = LinearSimdgroups::Eight;
  // Cross-threadgroup K partitions for Split128, Q4Register, GgufStaged and
  // GgufRegister, a power of two up to kMaximumSplits (Split128 takes at
  // least two); all other tiles use one.
  uint32_t splits = 1;
  static constexpr uint32_t kMaximumSplits = 8;
  [[nodiscard]] constexpr bool validSplits() const noexcept {
    return splits && splits <= kMaximumSplits && !(splits & (splits - 1));
  }
  bool operator==(const LinearConfig &) const = default;
};

// Reused serially within one decode command stream. Counters are zeroed at
// allocation and restored by each completed split dispatch. Never share this
// workspace between concurrent command streams. Within a batched dispatch of
// the Q4Register tile, each eight-row tile owns disjoint input, sums, partials
// and counters; Split128 holds every row of the step in each tile.
struct LinearScratch final {
  metal::MetalBuffer input;
  metal::MetalBuffer sums;
  metal::MetalBuffer partials;
  metal::MetalBuffer counters;
  // The bf16 input rows a rotated projection's quantized segments read
  // (ProjectionShape::rotated), sized by decode/prefillScratchSize(shape).rotated.
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

// The activation layout a decode plan reads: the producer's bf16 rows, or an
// X^T table with fp32 row sums in LinearScratch that a producer can emit
// alongside its ordinary output.
enum class LinearInput : uint8_t {
  Plain,    // bf16 [rows][K]
  Table64,  // Q4Register's table, one sum per 64 inputs (kernels/common/q4_sgmatrix.h)
  Table16,  // GgufRegister's table, sums per 16 and 32 inputs (kernels/common/gguf_sgmatrix.h)
};
// Scratch bytes a producer writes for `rows` rows of `width` inputs.
[[nodiscard]] constexpr uint64_t tableBytes(uint32_t width, uint64_t rows) noexcept {
  return uint64_t{width} * rows * 2;
}
[[nodiscard]] uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept;
// Throws unless `scratch` holds the `layout` table a producer writes for `rows`
// rows of `width` inputs: a table layout, whole lanes of rows and whole
// 64-input spans.
void requireTableScratch(const LinearScratch &scratch, LinearInput layout, uint32_t width, uint32_t rows);
// The kernel name suffix of a producer that writes the `layout` table:
// "_table16", "_table64", or none for Plain.
[[nodiscard]] const char *tableSuffix(LinearInput layout) noexcept;
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
  // Threadgroups over the column tiles: the configured groups of a
  // persistent decode tile, every column tile otherwise.
  [[nodiscard]] uint32_t groups() const noexcept;
  [[nodiscard]] uint32_t threadsPerThreadgroup() const noexcept;
  [[nodiscard]] bool usesQ4Register() const noexcept;
  // The layout the producer of this plan's input writes. A rotated
  // projection prepares its table from the rotated rows itself
  // (LinearGguf.cpp), so its producer writes plain rows.
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
  // The plan's projection multiplies the rotated input (InputRotation).
  bool rotated_ = false;
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

// Missing core metadata uses one intermediate estimate for all families.
// This is a fallback, not a calibrated optimum. Reported counts always win.
inline constexpr uint32_t kAssumedGpuCores = 32;
// The GPU core count kernel policy plans for: the reported one, or
// kAssumedGpuCores when the device does not report it.
[[nodiscard]] constexpr uint32_t plannedGpuCores(const DeviceCapabilities &device) noexcept {
  return device.gpuCoreCount ? device.gpuCoreCount : kAssumedGpuCores;
}
// The GPU family classes kernel policy tells apart: Apple9 (family 9: M3,
// M4), whose matrix operations share the FP32 pipe, and Apple10 (family 10
// and later: M5, M6), whose cores each hold a neural accelerator. Startup
// refuses families below 9.
enum class GpuFamilyClass : uint8_t { Apple9, Apple10 };
[[nodiscard]] constexpr GpuFamilyClass gpuFamilyClass(uint32_t appleGpuFamily) noexcept {
  return appleGpuFamily >= 10 ? GpuFamilyClass::Apple10 : GpuFamilyClass::Apple9;
}

// Owns projection pipeline selection and dispatch for both weight layouts.
// Device policy uses the GPU family class, core count and workload tile
// counts.
class Linear final {
public:
  explicit Linear(const DeviceCapabilities &device) noexcept;

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
  // The scratch of every decode plan of a projection of `shape`: each lane
  // count, the None, Residual and GateUp epilogues and every tile the device
  // may run them on, and the rotated rows of a full decode batch.
  [[nodiscard]] LinearScratchSize decodeScratchSize(ProjectionShape shape) const;
  // The scratch of every prefill chunk and epilogue of a projection of
  // `shape`: the split partials and counters of the chunks that run the GGUF
  // staged tile (LinearGguf.cpp), and the rotated rows of a full chunk.
  [[nodiscard]] LinearScratchSize prefillScratchSize(ProjectionShape shape) const;
  // The tile of a float projection of `rows` rows into `outputSize` columns
  // on this device (LinearGguf.cpp).
  [[nodiscard]] FloatTile ggufFloatTile(uint32_t rows, uint32_t outputSize) const noexcept;
  // The plan of `config` for `workload`, whether or not the device's policy
  // picks it.
  [[nodiscard]] static LinearPlan plan(LinearWorkload workload, LinearConfig config,
                                       FloatOutput destination);
  // Returns what the scratch table describes after the dispatch.
  PreparedInput add(metal::CommandGraph &graph, LinearBuffers buffers,
                    const Projection &projection, const LinearPlan &plan,
                    const Projection *gate = nullptr) const;

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
                            metal::MetalBuffer downSums, uint32_t rows, LinearScratch scratch) const;
  void addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                          metal::MetalBuffer residual, metal::MetalBuffer output, metal::MetalBuffer sums,
                          uint32_t rows, LinearScratch scratch) const;

private:
  // The device's configuration of the workload; a block plan's tile may follow the formats of the projections it
  // runs.
  [[nodiscard]] LinearConfig baseline(LinearWorkload workload,
                                      std::span<const Projection *const> projections = {}) const;
  // The partials and counters a dispatch of `splits` K partitions binds: the
  // scratch's, or for one partition, which reads neither, the output.
  [[nodiscard]] static std::array<metal::MetalBuffer, 2> splitScratch(const LinearBuffers &buffers,
                                                                      uint32_t splits);
  // GGUF policy and dispatch (LinearGguf.cpp). Block plans are not tuned.
  [[nodiscard]] LinearConfig ggufBaseline(LinearWorkload workload,
                                          std::span<const Projection *const> projections) const;
  // The scratch of every tile a block decode plan of the workload may take.
  [[nodiscard]] LinearScratchSize ggufDecodeScratchSize(LinearWorkload workload) const;
  void addGguf(metal::CommandGraph &graph, const LinearBuffers &buffers,
               const Projection &projection, const LinearPlan &plan,
               const Projection *gate) const;
  void addGgufStaged(metal::CommandGraph &graph, const LinearBuffers &buffers,
                     const Projection &projection, const LinearPlan &plan,
                     const Projection *gate) const;
  void addGgufPrefill(metal::CommandGraph &graph, const LinearBuffers &buffers,
                      const Projection &projection, const LinearPlan &plan) const;
  void addGgufRegister(metal::CommandGraph &graph, const LinearBuffers &buffers,
                       const Projection &projection, const LinearPlan &plan,
                       const Projection *gate) const;
  void addGgufFloatSegments(metal::CommandGraph &graph, const LinearBuffers &buffers,
                            const Projection &projection, const LinearPlan &plan) const;
  GpuFamilyClass family_;
  uint32_t gpuCores_ = 0;
};

} // namespace splash::ops

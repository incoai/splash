#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Weights.hpp"

#include <algorithm>
#include <array>
#include <compare>
#include <cstdint>
#include <string>
#include <string_view>

namespace splash::ops {

// The instance of kernel `name` that writes `destination`: a plain decode
// kernel's fp32 instance is "<name>_f32".
[[nodiscard]] inline std::string kernelInstance(std::string_view name, FloatOutput destination) {
  return std::string(name) + (destination == FloatOutput::Float32 ? "_f32" : "");
}
// The instance of prefill residual kernel `name` that reads a view of the
// leading inputs of wider weight rows (Projection::leadingInputs):
// "<name>_leading_inputs".
[[nodiscard]] inline std::string leadingInputsInstance(std::string_view name) {
  return std::string(name) + "_leading_inputs";
}

struct LinearMatrix final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  auto operator<=>(const LinearMatrix &) const = default;
};

// Throws unless `projection` is an affine projection of `matrix` whose planes
// hold all of its Q4 weights, scales and biases, in rows of its
// planeInputSize() inputs.
void requireAffineProjection(const Projection &projection, LinearMatrix matrix);

enum class LinearPhase : uint8_t { Prefill, Decode };
enum class LinearEpilogue : uint8_t { None, Residual, GateUp, UpWithGate };
// Compute tiles over the affine storage tiles. Paired tiles pipeline two
// quant groups of one lane. Split128 is the N128 tile with K split across
// `splits` threadgroups, grid (column tiles, splits), every lane's rows in each
// tile; the last threadgroup of a tile to finish reduces the fp32 partial sums
// before the bf16 rounding. Paired256 is the four-simdgroup N256 paired tile.
// LaneSplit128 is one lane's paired N128 tile (eight simdgroups) over 1 to
// kMaximumSplits K partitions, gate/up accumulating both projections in one
// dispatch. Q4Register is the register tile on bf16 8x8 matrix operations
// (Apple9), over an activation table and with optional K splits in an
// explicit workspace. Qmv is the scalar projection of one live row, or of
// two sharing its threadgroups (LinearConfig::sharedRows)
// (kernels/decode/linear_q4_qmv.metal): every simdgroup's 32 lanes split K
// for a run of adjacent outputs, eight simdgroups a threadgroup, no K split.
enum class LinearTile : uint8_t { N128, N256, Paired128, Split128, Paired256, Q4Register, LaneSplit128, Qmv };
// The decode tiles hold at most a full decode batch.
inline constexpr uint32_t kMaximumDecodeTileRows = SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS;
enum class LinearSimdgroups : uint8_t { Four = 4, Eight = 8 };

struct LinearWorkload final {
  LinearMatrix matrix;
  uint32_t rows = 0;
  LinearPhase phase = LinearPhase::Decode;
  LinearEpilogue epilogue = LinearEpilogue::None;
  WeightLayout weightLayout = WeightLayout::Affine64;
  // Decode: the leading rows of the step that are live, 0 for all of them.
  // The others carry whatever the buffers hold and are never read back, so a
  // plan of one or two live rows (Qmv) neither reads nor writes them.
  uint32_t liveRows = 0;
  auto operator<=>(const LinearWorkload &) const = default;
};

struct LinearConfig final {
  LinearTile tile = LinearTile::N128;
  // Persistent threadgroups of the N128, N256, Paired128 and Paired256 decode
  // tiles (1 to their column tiles); 0 for every other plan, whose grid
  // covers the matrix.
  uint32_t groups = 0;
  // Simdgroups of a tile's threadgroup, independent of the persistent grid
  // size: the cooperative scope of one tile (Paired256 runs four).
  LinearSimdgroups simdgroups = LinearSimdgroups::Eight;
  // Cross-threadgroup K partitions for Split128, LaneSplit128 and
  // Q4Register, a power of two up to kMaximumSplits (Split128 takes at least
  // two); all other tiles use one.
  uint32_t splits = 1;
  static constexpr uint32_t kMaximumSplits = 8;
  // Qmv: the adjacent outputs of each simdgroup, 2 or 4; 0 for every other
  // tile.
  uint32_t simdgroupOutputs = 0;
  // Qmv: two live rows share each threadgroup and its code loads, four
  // outputs a simdgroup (two for gate/up), after the norm's dispatch;
  // otherwise each live row runs its own threadgroups.
  bool sharedRows = false;
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
};
// One layer's SwiGLU projections: down(silu(gate x) * up x).
struct SwiGluProjections final {
  const Projection *gate = nullptr;
  const Projection *up = nullptr;
  const Projection *down = nullptr;
};
// A prefill chunk's buffers of its dense FFN: the normalized rows and their
// Q4 sums, gate's output, the intermediate rows and their Q4 sums, and the
// linear scratch.
struct PrefillFfnBuffers final {
  metal::MetalBuffer normalized, sums, gateScratch, intermediate, downSums;
  LinearScratch scratch;
};
struct LinearScratchSize final {
  uint64_t input = 0, sums = 0, partials = 0, counters = 0;
  [[nodiscard]] uint64_t bytes() const noexcept { return input + sums + partials + counters; }
  // Grows each field to hold `other`'s too.
  LinearScratchSize &include(const LinearScratchSize &other) noexcept {
    input = std::max(input, other.input);
    sums = std::max(sums, other.sums);
    partials = std::max(partials, other.partials);
    counters = std::max(counters, other.counters);
    return *this;
  }
};

// The activation layout a decode plan reads: the producer's bf16 rows, or an
// X^T table with fp32 row sums in LinearScratch that a producer can emit
// alongside its ordinary output.
enum class LinearInput : uint8_t {
  Plain,    // bf16 [rows][K]
  Table64,  // Q4Register's table, one sum per 64 inputs (kernels/common/q4_sgmatrix.h)
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
// "_table64", or none for Plain.
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
  // The layout the producer of this plan's input writes.
  [[nodiscard]] LinearInput input() const noexcept;
  [[nodiscard]] LinearScratchSize scratchSize() const noexcept;
  [[nodiscard]] uint64_t sumsBytes() const noexcept;
  [[nodiscard]] uint64_t gateScratchBytes() const noexcept;
  [[nodiscard]] uint64_t downSumsBytes() const noexcept;
  // The tile's kernel; the plan runs its kernelInstance for destination().
  [[nodiscard]] std::string_view pipeline() const noexcept { return pipeline_; }
  [[nodiscard]] std::string_view secondPipeline() const noexcept {
    return secondPipeline_;
  }
  // Whether the plan can normalize its input row itself
  // (LinearBuffers::normWeight): a one-row Qmv plain or gate/up projection
  // of few enough outputs.
  [[nodiscard]] bool canFoldNorm() const noexcept { return !normPipeline_.empty(); }
  // Whether a decode step of `liveRows` live rows should fold its norm into
  // this plan rather than run the norm dispatch: canFoldNorm() and its one
  // live row.
  [[nodiscard]] bool foldsNorm(uint32_t liveRows) const noexcept;

private:
  friend class Linear;
  LinearPlan(LinearWorkload workload, LinearConfig config, FloatOutput destination = FloatOutput::BFloat16);
  LinearWorkload workload_;
  LinearConfig config_;
  FloatOutput destination_;
  std::string_view pipeline_;
  std::string_view secondPipeline_;
  std::string_view normPipeline_;
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
  // A plan that canFoldNorm() reads `input` as its un-normalized live row
  // when given the RMS norm's bf16 weights here (normRows: that one row): it
  // projects bf16((x * inverse RMS) * weight), the bits
  // Normalization::addRms writes.
  metal::MetalBuffer normWeight{};
  uint32_t normRows = 0;
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

// Owns projection pipeline selection and dispatch. Device policy uses the
// GPU family class, core count and workload tile counts.
class Linear final {
public:
  explicit Linear(const DeviceCapabilities &device) noexcept;

  [[nodiscard]] LinearPlan plan(LinearWorkload workload) const;
  // The plan of `workload` for the projection, into its destination type; a
  // gate/up plan also runs `gate`.
  [[nodiscard]] LinearPlan plan(LinearWorkload workload, const Projection &projection,
                                const Projection *gate = nullptr) const;
  // Rows of storage a decode step of `rows` rows binds for a projection of
  // `shape`: the storageRows of its decode plans, which every epilogue shares.
  [[nodiscard]] uint32_t decodeStorageRows(uint32_t rows, ProjectionShape shape) const;
  // The plans of this projection's matrix. A decode plan's input() is the
  // layout its producer writes.
  [[nodiscard]] LinearPlan prefillPlan(const Projection &projection, uint32_t rows,
                                       LinearEpilogue epilogue) const;
  // `liveRows` leading rows of the lanes are live (0: all of them).
  [[nodiscard]] LinearPlan decodePlan(const Projection &projection, uint32_t lanes,
                                      LinearEpilogue epilogue = LinearEpilogue::None,
                                      const Projection *gate = nullptr, uint32_t liveRows = 0) const;
  // The scratch of every decode plan of a projection of `shape`: each lane
  // count, the None, Residual and GateUp epilogues and every tile the device
  // may run them on.
  [[nodiscard]] LinearScratchSize decodeScratchSize(ProjectionShape shape) const;
  // The scratch of every prefill chunk and epilogue of a projection of
  // `shape`.
  [[nodiscard]] LinearScratchSize prefillScratchSize(ProjectionShape shape) const;
  // The plan of `config` for `workload`, whether or not the device's policy
  // picks it.
  [[nodiscard]] static LinearPlan plan(LinearWorkload workload, LinearConfig config,
                                       FloatOutput destination);
  // Returns what the scratch table describes after the dispatch.
  PreparedInput add(metal::CommandGraph &graph, LinearBuffers buffers,
                    const Projection &projection, const LinearPlan &plan,
                    const Projection *gate = nullptr) const;

  // The Q4 input sums of `rows` rows a prefill projection reads.
  void addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer sums,
                      const Projection &consumer, uint32_t rows) const;
  // The projections of `rows` rows through their own matrix. `scratch` holds
  // the partials and counters of split plans; reused serially within one
  // command stream, as in decode.
  void addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                  metal::MetalBuffer output, metal::MetalBuffer sums, uint32_t rows,
                  LinearScratch scratch = {}) const;
  void addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &up,
                            metal::MetalBuffer gateScratch, metal::MetalBuffer output, metal::MetalBuffer sums,
                            metal::MetalBuffer downSums, uint32_t rows, LinearScratch scratch) const;
  void addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                          metal::MetalBuffer residual, metal::MetalBuffer output, metal::MetalBuffer sums,
                          uint32_t rows, LinearScratch scratch) const;
  // output = residual + down(silu(gate x) * up x) of `rows` normalized rows,
  // whose Q4 sums the norm wrote.
  void addPrefillSwiGlu(metal::CommandGraph &graph, const SwiGluProjections &ffn, const PrefillFfnBuffers &buffers,
                        metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows) const;

private:
  // The device's configuration of the workload.
  [[nodiscard]] LinearConfig baseline(LinearWorkload workload) const;
  // The partials and counters a dispatch of `splits` K partitions binds: the
  // scratch's, or for one partition, which reads neither, the output.
  [[nodiscard]] static std::array<metal::MetalBuffer, 2> splitScratch(const LinearBuffers &buffers,
                                                                      uint32_t splits);
  GpuFamilyClass family_;
  uint32_t gpuCores_ = 0;
};

} // namespace splash::ops

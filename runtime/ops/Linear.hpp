#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "ops/KernelCalibration.hpp"
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
// The instance of prefill residual kernel `name` that reads a view of the
// leading inputs of wider weight rows (Projection::leadingInputs):
// "<name>_leading_inputs".
[[nodiscard]] inline std::string leadingInputsInstance(std::string_view name) {
  return std::string(name) + "_leading_inputs";
}
// The tile of a float projection (kernels/shared/gguf_float.metal): fp32
// simdgroup MMA on the weights as stored, or the neural accelerator's bf16
// matmul on each weight's three bf16 parts, which sum to it exactly. Both
// round only in fp32 accumulation; selectFloatTile picks one
// (KernelSelection.cpp).
enum class FloatTile : uint8_t { Simdgroup, NeuralAccelerator };
// out[r][outOffset + n] = sum_k input[r][k] W[n][k] for rows r < `rows` of a
// float segment, into a destination of `outStride` columns (LinearGguf.cpp).
void addGgufFloat(metal::CommandGraph &graph, metal::MetalBuffer input, const QuantizedSegment &weights,
                  metal::MetalBuffer output, uint32_t rows, uint32_t outStride, uint32_t outOffset,
                  FloatOutput type, FloatTile tile);
// One of several float projections of an input, each into its own
// destination on its own tile, as the QSA indexer projects its query and key.
struct FloatProjection final {
  const QuantizedSegment *weights = nullptr;
  metal::MetalBuffer output;
  uint32_t outStride = 0;
  uint32_t outOffset = 0;
  FloatTile tile = FloatTile::Simdgroup;
};
// The projections of `rows` rows of one input, each dispatch running up to
// GGUF_FLOAT_PROJECTIONS (metal/abi/Gguf.h) consecutive ones of one value
// type and tile, so that each one's outputs are those of its own dispatch.
void addGgufFloat(metal::CommandGraph &graph, metal::MetalBuffer input, std::span<const FloatProjection> projections,
                  uint32_t rows, FloatOutput type);

struct LinearMatrix final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  auto operator<=>(const LinearMatrix &) const = default;
};

// Throws unless the planes of the quantized `segment` hold every tile of its
// outputSize x inputSize weights (metal/abi/QuantFormat.h), naming them
// "<what> plane0", "<what> plane1" and "<what> meta": in rows of
// `planeInputs` inputs, of which it is a view of the leading ones
// (Projection::leadingInputs), or of its inputSize when that is 0.
void requireSegmentPlanes(const QuantizedSegment &segment, std::string_view what, uint32_t planeInputs = 0);

// A gate/up pair runs as one plan, GateUp, on the decode tiles; past them, as
// a gate plan and an up plan that applies silu(gate), UpWithGate
// (Linear::addSwiGlu), which the column tile does not run.
enum class LinearEpilogue : uint8_t { None, Residual, GateUp, UpWithGate };
// The tiles run 64 columns per threadgroup; a one-pass gate/up
// (LinearConfig::oneGateUpPass) runs 32 columns of both tensors per
// threadgroup of one simdgroup. The staged tiles, GgufStaged and GgufPrefill,
// dequantize the weights into threadgroup memory for matmul2d.
// GgufStaged: the two-simdgroup staged tile of 8, 16, 32 or 64 rows, each
// simdgroup staging its own columns, with optional K splits. GgufPrefill: the
// 128-row shared-stage prefill tile.
// GgufRegister: the exact register tile on bf16 8x8 matrix operations
// (Apple9), over an activation table, all the plan's rows in one threadgroup,
// up to kMaximumRegisterRows, optional K splits.
// GgufPrefillInt8: plans past the decode tiles of MLX affine 4- and 5-bit g64
// weights (quant_prefill_int8_format) in int8 activations, quantized per row
// and 64-input group into the scratch table, against the weights' exact codes
// (kernels/prefill/gguf_prefill_int8.metal); not exact, unsplit, the prefill
// tile's storage rows.
// GgufColumn: a plan of at most GGUF_COLUMN_ROWS rows as a GEMV in registers,
// each lane one column, reading those rows alone
// (kernels/decode/linear_gguf_column.metal); unsplit across threadgroups, a
// gate/up pair of one format in one pass.
enum class LinearTile : uint8_t { GgufStaged, GgufPrefill, GgufRegister, GgufPrefillInt8, GgufColumn };
// The decode tiles (the staged tile's 8- to 64-row heights, the register and
// column tiles) hold a plan of up to 64 rows, a decode step of four lanes of
// 16; a plan of more rows runs the prefill tiles (selectProjection).
inline constexpr uint32_t kMaximumDecodeTileRows = 64;
// The register tile computes its rows in groups of an activation table's tile
// (Table16, metal/abi/Gguf.h): a plan of r rows stores ⌈r / 8⌉ of them, at
// most four (its kernels' lanes), so it holds plans of up to 32 rows.
[[nodiscard]] constexpr uint32_t tableTiles(uint32_t rows) noexcept {
  return (rows + GGUF_TABLE16_ROWS - 1) / GGUF_TABLE16_ROWS;
}
inline constexpr uint32_t kMaximumRegisterRows = SPLASH_MAXIMUM_BATCH_WIDTH * GGUF_TABLE16_ROWS;
// A one-pass gate/up pair's simdgroup holds both tensors' sums of every row:
// up to 32 rows (kernels/shared/gguf_linear.metal); a plan of more runs the
// pair in two passes.
inline constexpr uint32_t kMaximumPairRows = 32;
// The int8 decode tiles' simdgroups hold their int32 and fp32 sums of every
// row up to 32 rows; a plan of more keeps bf16 activations on the staged tile
// (kernels/shared/gguf_linear.metal).
inline constexpr uint32_t kMaximumInt8Rows = 32;
// The staged tile that holds `rows` rows: MPP computes 16-row fragments, so
// the tile holds 8, 16, 32 or 64 rows, the fewest that hold the plan's (24
// rows run the 32-row tile; a plan of at most 8 may run the 16-row tile,
// LinearConfig::sixteenRows).
[[nodiscard]] constexpr uint32_t stagedTileRows(uint32_t rows) noexcept {
  return rows <= 8 ? 8 : rows <= 16 ? 16 : rows <= 32 ? 32 : 64;
}
// The rows a staged plan of `rows` rows stores: its tile's, or 16 for a plan
// on the 16-row tile (LinearConfig::sixteenRows) or on the int8 tiles, whose
// shortest height is 16 (LinearConfig::decodeInt8).
[[nodiscard]] constexpr uint32_t stagedStorageRows(uint32_t rows, bool sixteenRows, bool decodeInt8 = false) noexcept {
  return sixteenRows || decodeInt8 ? std::max(16u, stagedTileRows(rows)) : stagedTileRows(rows);
}

// `rows` rows of a matrix through an epilogue. The rows pick the plan's tile
// (selectProjection), whatever computes them: a plan stores at least its rows
// (LinearPlan::storageRows), and the rows past them are padding that its tiles
// may compute and no reader reads.
struct LinearWorkload final {
  LinearMatrix matrix;
  uint32_t rows = 0;
  LinearEpilogue epilogue = LinearEpilogue::None;
  auto operator<=>(const LinearWorkload &) const = default;
};

struct LinearConfig final {
  LinearTile tile = LinearTile::GgufStaged;
  // Cross-threadgroup K partitions for GgufStaged and GgufRegister, a power
  // of two up to kMaximumSplits; GgufPrefill uses one.
  uint32_t splits = 1;
  static constexpr uint32_t kMaximumSplits = 16;
  // GgufStaged: each column tile starts its walk over its K partition at its
  // own step (kernels/common/gguf_staged_tile.h) instead of every tile at the
  // partition's first; no other tile spreads.
  bool spread = false;
  // GgufStaged gate/up on a decode tile: gate and up in one dispatch whose
  // simdgroups each stage both tensors' 32 columns
  // (gguf_decode_<format>_m<rows>_gate_up, gate and up in one format), instead
  // of a gate pass and an up pass; a GgufColumn gate/up always.
  bool oneGateUpPass = false;
  // GgufStaged plan of at most 8 rows: the 16-row tile (LinearPlan::
  // storageRows), whose padding rows it computes and discards, instead of the
  // 8-row tile.
  bool sixteenRows = false;
  // GgufStaged plan on a decode tile of rotated PQ2_0 segments in int8
  // activations: the rotated input as int8 rows, one scale per row and
  // rotation block (gguf_rotate_int8), into the int8 tiles
  // (gguf_decode_int8_pq20_m<rows>_*) at the plan's storage rows, splits,
  // spread and passes.
  bool decodeInt8 = false;
  [[nodiscard]] constexpr bool validSplits() const noexcept {
    return splits && splits <= kMaximumSplits && !(splits & (splits - 1));
  }
  bool operator==(const LinearConfig &) const = default;
};

// Reused serially within one command stream. Counters are zeroed at
// allocation and restored by each completed split dispatch. Never share this
// workspace between concurrent command streams.
struct LinearScratch final {
  // The register tile's Table16 table and its sums; the int8 prefill tile's
  // codes and scales.
  metal::MetalBuffer input;
  metal::MetalBuffer sums;
  metal::MetalBuffer partials;
  metal::MetalBuffer counters;
  // The bf16 input rows a rotated projection's quantized segments read
  // (ProjectionShape::rotated), sized by scratchSize(shape, rows).rotated.
  metal::MetalBuffer rotated{};
};
// One layer's SwiGLU projections: down(silu(gate x) * up x).
struct SwiGluProjections final {
  const Projection *gate = nullptr;
  const Projection *up = nullptr;
  const Projection *down = nullptr;
};
// A prefill chunk's buffers of its dense FFN: the normalized rows, gate's
// output, the intermediate rows and the linear scratch.
struct PrefillFfnBuffers final {
  metal::MetalBuffer normalized, gateScratch, intermediate;
  LinearScratch scratch;
};
struct LinearScratchSize final {
  uint64_t input = 0, sums = 0, partials = 0, counters = 0, rotated = 0;
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

// The activation layout a plan reads: the producer's bf16 rows, or an X^T
// table with fp32 row sums in LinearScratch that a producer can emit
// alongside its ordinary output; or the int8 prefill tile's codes, which only
// that tile writes, for a next plan over the same rows (a SwiGLU's up after
// its gate).
enum class LinearInput : uint8_t {
  Plain,        // bf16 [rows][K]
  Table16,      // the register tile's table, sums per 16 and 32 inputs (kernels/common/gguf_sgmatrix.h)
  PrefillInt8,  // the int8 prefill tile's codes and scales per 64 inputs (kernels/prefill/gguf_prefill_int8.metal)
};
// Scratch bytes a producer writes for `rows` rows of `width` inputs.
[[nodiscard]] constexpr uint64_t tableBytes(uint32_t width, uint64_t rows) noexcept {
  return uint64_t{width} * rows * 2;
}
[[nodiscard]] uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept;
// Throws unless `scratch` holds the `layout` table a producer writes for `rows`
// rows of `width` inputs: a table layout, whole 64-input spans, and the table
// tiles that hold the rows (tableTiles), whose padding rows no producer
// writes.
void requireTableScratch(const LinearScratch &scratch, LinearInput layout, uint32_t width, uint32_t rows);
// The kernel name suffix of a producer that writes the `layout` table:
// "_table16", or none for Plain.
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
  // The layout the producer of this plan's input writes. A rotated
  // projection prepares its table from the rotated rows itself
  // (LinearGguf.cpp), so its producer writes plain rows.
  [[nodiscard]] LinearInput input() const noexcept;
  [[nodiscard]] LinearScratchSize scratchSize() const noexcept;
  [[nodiscard]] uint64_t gateScratchBytes() const noexcept;

private:
  friend class Linear;
  LinearPlan(LinearWorkload workload, LinearConfig config, FloatOutput destination = FloatOutput::BFloat16);
  // Throws unless the tile takes the workload and the splits (LinearGguf.cpp).
  void requireConfiguration() const;
  LinearWorkload workload_;
  LinearConfig config_;
  FloatOutput destination_;
  // The plan's projection multiplies the rotated input (InputRotation).
  bool rotated_ = false;
};

// The plan defines which fields are used and how much scratch they require.
struct LinearBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer output;
  metal::MetalBuffer residual;
  metal::MetalBuffer gateScratch;
  LinearScratch scratch{};
  // What the scratch table holds (for example after fused RMSNorm). A plan
  // that reads a table prepares one unless this describes its input.
  PreparedInput prepared{};
};

// Whether plans may round activations to int8 where an int8-activation path
// takes them, a precision choice made once per runtime (ExecutionPlans) beside
// the device's measured KernelCalibration: the int8 prefill and decode tiles
// (selectProjection) and INT8 attention queries (PagedAttention). Off keeps
// every activation bf16; on, the GPU's int8 matmul, twice its bf16 rate, takes
// about a quarter off Qwen3.8-27B MLX 4-bit prefill.
enum class Int8Activations : uint8_t { Off, On };

// Owns projection pipeline selection and dispatch. Plans follow the device's
// KernelCalibration and the runtime's Int8Activations (KernelSelection.cpp),
// which ExecutionPlans also gives its attention plans.
class Linear final {
public:
  explicit Linear(const DeviceCapabilities &device, Int8Activations int8Activations) noexcept;
  [[nodiscard]] const KernelCalibration &calibration() const noexcept { return calibration_; }
  [[nodiscard]] Int8Activations int8Activations() const noexcept { return int8Activations_; }

  [[nodiscard]] LinearPlan plan(LinearWorkload workload) const;
  // The plan of `workload` for the projection, into its destination type; a gate/up plan also runs `gate`.
  [[nodiscard]] LinearPlan plan(LinearWorkload workload, const Projection &projection,
                                const Projection *gate = nullptr) const;
  // The projection's plan of `rows` rows (its input() the layout its
  // producer writes); a gate/up plan also runs `gate`. A plan of `rows` rows
  // of a chunk of `chunkRows` (0: `rows`) takes the chunk's configuration at
  // its own rows' height, which computes each row as the chunk's plan does.
  [[nodiscard]] LinearPlan plan(const Projection &projection, uint32_t rows,
                                LinearEpilogue epilogue = LinearEpilogue::None, const Projection *gate = nullptr,
                                uint32_t chunkRows = 0) const;
  // Rows of storage a step of `rows` rows binds for a projection of `shape`:
  // the most storageRows of its plans, a one-pass gate/up pair's among them.
  [[nodiscard]] uint32_t storageRows(uint32_t rows, ProjectionShape shape) const;
  // The scratch of every plan of up to `rows` rows of a projection of
  // `shape`, every epilogue and every tile the device may run them on: the
  // register tile's table and the split partials and counters of the decode
  // and staged tiles, the rotated rows of `rows` rows, and where the device
  // runs the int8 prefill tile on the shape's formats, its codes and scales of
  // `rows` rows.
  [[nodiscard]] LinearScratchSize scratchSize(ProjectionShape shape, uint32_t rows) const;
  // The tile of a float projection of `rows` rows into `outputSize` columns
  // of rows in `formatId` (QuantizedSegment::rows) on this device
  // (selectFloatTile).
  [[nodiscard]] FloatTile ggufFloatTile(uint32_t rows, uint32_t outputSize, uint32_t formatId) const noexcept;
  // The plan of `config` for `workload`, whether or not the device's policy
  // picks it.
  [[nodiscard]] static LinearPlan plan(LinearWorkload workload, LinearConfig config,
                                       FloatOutput destination);
  // Returns what the scratch table describes after the dispatch.
  PreparedInput add(metal::CommandGraph &graph, LinearBuffers buffers,
                    const Projection &projection, const LinearPlan &plan,
                    const Projection *gate = nullptr) const;

  // The projections of `rows` rows through their own matrix, on the plans of
  // a chunk of `chunkRows` rows (plan). `scratch` holds what the plans need
  // beside their buffers: a table they prepare, partials and counters of a
  // split; reused serially within one command stream.
  void addProjection(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                     metal::MetalBuffer output, uint32_t rows, LinearScratch scratch = {},
                     uint32_t chunkRows = 0) const;
  void addResidualProjection(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                             metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows,
                             LinearScratch scratch, uint32_t chunkRows = 0) const;
  // output = residual + down(silu(gate x) * up x) of `rows` normalized rows:
  // one gate/up plan where the chunk's rows run the decode tiles, otherwise a
  // gate plan and an up plan that takes what gate's plan leaves of x in the
  // scratch table; `normalized` describes what its producer wrote of the
  // first plan's input (swiGluInput).
  void addSwiGlu(metal::CommandGraph &graph, const SwiGluProjections &ffn, const PrefillFfnBuffers &buffers,
                 metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows, uint32_t chunkRows = 0,
                 PreparedInput normalized = {}) const;
  // The layout the producer of a SwiGLU's normalized rows writes for its first
  // plan (addSwiGlu).
  [[nodiscard]] LinearInput swiGluInput(const SwiGluProjections &ffn, uint32_t rows, uint32_t chunkRows = 0) const;

private:
  // The partials and counters a dispatch of `splits` K partitions binds: the
  // scratch's, or for one partition, which reads neither, the output.
  [[nodiscard]] static std::array<metal::MetalBuffer, 2> splitScratch(const LinearBuffers &buffers,
                                                                      uint32_t splits);
  // The scratch of every tile a decode plan of the workload may take, the
  // register tile's where it takes the projection's formats.
  [[nodiscard]] LinearScratchSize ggufDecodeScratchSize(LinearWorkload workload, bool registerFormats) const;
  void addGguf(metal::CommandGraph &graph, const LinearBuffers &buffers,
               const Projection &projection, const LinearPlan &plan,
               const Projection *gate) const;
  void addGgufStaged(metal::CommandGraph &graph, const LinearBuffers &buffers,
                     const Projection &projection, const LinearPlan &plan,
                     const Projection *gate) const;
  void addGgufPrefill(metal::CommandGraph &graph, const LinearBuffers &buffers,
                      const Projection &projection, const LinearPlan &plan) const;
  void addGgufPrefillInt8(metal::CommandGraph &graph, const LinearBuffers &buffers,
                          const Projection &projection, const LinearPlan &plan) const;
  void addGgufRegister(metal::CommandGraph &graph, const LinearBuffers &buffers,
                       const Projection &projection, const LinearPlan &plan,
                       const Projection *gate) const;
  void addGgufColumn(metal::CommandGraph &graph, const LinearBuffers &buffers, const Projection &projection,
                     const LinearPlan &plan, const Projection *gate) const;
  void addGgufFloatSegments(metal::CommandGraph &graph, const LinearBuffers &buffers,
                            const Projection &projection, const LinearPlan &plan) const;
  KernelCalibration calibration_;
  Int8Activations int8Activations_;
};

} // namespace splash::ops

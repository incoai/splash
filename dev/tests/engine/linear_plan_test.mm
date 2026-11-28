#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/GDN.hpp"
#include "ops/LaneRows.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/QuantFormat.h"

#include <array>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;

using splash::test::rejects;
using splash::test::require;
using splash::test::requireExtent;

// Whether one unsplit staged lane of a projection of `shape` runs the 16-row
// tile on Apple10 or Apple11 (`family` 10 or later) where its bits per weight
// decide it: below four and a half it does, and float segments alone or
// Apple11 from four and a half do not. Elsewhere Apple10's wave law decides
// (nullopt).
std::optional<bool> sixteenRowsByBits(const ProjectionShape &shape, uint32_t family) {
  if (!shape.weightBits16) return false;
  if (shape.weightBits16 < 72) return true;
  if (family >= 11) return false;
  return std::nullopt;
}

// Each streams 13.5 GB/s a core (the 12-core M6's measured plain read, 161.9
// GB/s), exactly, so that twice the cores stream twice the bytes.
DeviceCapabilities simulatedDevice(uint32_t family, uint32_t cores) {
  DeviceCapabilities device;
  device.appleGpuFamily = family;
  device.gpuCoreCount = cores;
  device.memoryBytesPerSecond = uint64_t{13'500'000'000} * device.plannedGpuCores();
  return device;
}
Linear gpu(uint32_t family, uint32_t cores, StepKind step = StepKind::Prefill) {
  return Linear(simulatedDevice(family, cores), Int8Activations::On, step);
}

// Throws naming the rule a plan broke and the plan.
[[noreturn]] void broke(const char *rule, uint32_t family, uint32_t cores, LinearWorkload w) {
  throw std::runtime_error(std::string(rule) + ": family " + std::to_string(family) + ", " +
                           std::to_string(cores) + " cores, N " + std::to_string(w.matrix.outputSize) +
                           ", K " + std::to_string(w.matrix.inputSize) + ", " + std::to_string(w.rows) +
                           " rows, epilogue " + std::to_string(unsigned(w.epilogue)));
}

bool sameScratch(LinearScratchSize a, LinearScratchSize b) {
  return a.input == b.input && a.sums == b.sums && a.partials == b.partials && a.counters == b.counters;
}

// A block projection of equal segments tiling its columns, one per format;
// planning reads only their geometry and formats.
Projection blockProjection(uint32_t n, uint32_t k, std::span<const uint32_t> formats) {
  BlockWeights weights;
  const uint32_t width = n / uint32_t(formats.size());
  for (const uint32_t format : formats) {
    QuantizedSegment s = QuantizedSegment::planes(format, width, k, {}, {}, {});
    s.columnOffset = uint32_t(weights.segments.size()) * width;
    weights.segments.push_back(s);
  }
  return Projection(n, k, std::move(weights));
}
// ... of `segments` segments in `format`, Q4_K unless it says otherwise.
Projection blockProjection(uint32_t n, uint32_t k, uint32_t segments, uint32_t format = GGUF_FMT_Q4K) {
  return blockProjection(n, k, std::vector<uint32_t>(segments, format));
}

// fp32 destinations (the logits): the plan of a projection with an fp32
// destination keeps the configuration, input table and scratch of its bf16
// plan on every family and core count, and only plain plans on the decode
// tiles take one. ggufProjectionMatrix checks the fp32 kernel it encodes.
void floatOutputPlans() {
  const auto fp32 = [](Projection p) {
    p.destination = FloatOutput::Float32;
    return p;
  };
  for (const uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores = 0; cores <= 128; ++cores) {
      const Linear linear = gpu(family, cores);
      for (const uint32_t n : {256U, 5120U, 16640U, 248320U})
        for (const uint32_t k : {2048U, 5120U}) {
          const Projection p = blockProjection(n, k, 1);
          for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
            const LinearWorkload w{{n, k}, lanes * 8, LinearEpilogue::None};
            const LinearPlan bf16 = linear.plan(w, p), head = linear.plan(w, fp32(p));
            if (head.configuration() != bf16.configuration() || head.destination() != FloatOutput::Float32 ||
                head.input() != bf16.input() || head.storageRows() != bf16.storageRows() ||
                !sameScratch(head.scratchSize(), bf16.scratchSize()))
              broke("an fp32 plan differs from its bf16 plan", family, cores, w);
          }
        }
    }
  std::vector<LinearWorkload> others{{{5120, 2048}, 8, LinearEpilogue::Residual},
                                     {{5120, 2048}, 8, LinearEpilogue::GateUp},
                                     {{5120, 2048}, 8, LinearEpilogue::UpWithGate}};
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
    others.push_back({{5120, 2048}, 2048, epilogue});
  const Linear linear = gpu(10, 20);
  const Projection p = blockProjection(5120, 2048, 1);
  for (const LinearWorkload &w : others) {
    (void)linear.plan(w, p);
    rejects([&] { (void)linear.plan(w, fp32(p)); }, "an fp32 destination takes a plain projection on the decode tiles",
            "an fp32 plan of other than a plain projection on the decode tiles was accepted");
  }
}

// The matrix tile is a family's whose calibration plans it (ProjectionCalibration::matrixSplits, Apple10's and
// Apple11's): Apple9's kernels of it write values outside their bounds (gguf-projection on a 10-core M4). No plan of
// a family without the law holds it, at any core count, rows, shape, epilogue or step, and the families with the law
// do plan it; ggufProjectionMatrix checks that a Linear without the law refuses one handed to it.
void matrixTileFamilies() {
  uint32_t planned = 0, plans = 0;
  for (const uint32_t family : {9U, 10U, 11U})
    for (const uint32_t cores : {0U, 8U, 10U, 12U, 16U, 20U, 32U, 40U, 64U, 80U, 128U})
      for (const StepKind step : {StepKind::Prefill, StepKind::Decode}) {
        const Linear linear = gpu(family, cores, step);
        const bool law = linear.calibration().projection.matrixSplits.has_value();
        if (family == 9 && law) broke("Apple9's calibration plans the matrix tile", family, cores, {});
        for (const auto &[n, k] : {std::pair{256U, 2048U}, {512U, 2048U}, {5120U, 2048U}, {5120U, 17408U}, {16640U, 5120U}}) {
          const Projection p = blockProjection(n, k, 1);
          for (uint32_t rows = 1; rows <= 2 * kMaximumStagedRows; ++rows)
            for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp,
                                        LinearEpilogue::UpWithGate}) {
              const LinearWorkload w{{n, k}, rows, epilogue};
              // A workload the planner takes: it refuses some of these rows and epilogues for every family.
              LinearTile tile;
              try {
                tile = linear.plan(w, p).configuration().tile;
              } catch (const std::invalid_argument &) {
                continue;
              }
              ++plans;
              if (tile != LinearTile::GgufMatrix) continue;
              if (!law) broke("a family whose calibration plans no matrix tile planned one", family, cores, w);
              ++planned;
            }
        }
      }
  require(planned > 0 && plans > 100'000, "the sweep checks nothing: " + std::to_string(planned) +
                                               " matrix plans of " + std::to_string(plans));
}

// Every Apple9 GPU runs plans of up to 32 rows on the register tile (which
// takes Q4_K) and of 33 to 64 rows, past its four table tiles, on the staged
// tile's 64-row height, the decode tiles' most (KernelCalibration.hpp: the
// 10-core M4 measured it); past 64 rows the 128-row prefill tile.
void apple9StagesPastTheRegisterTile() {
  const Projection down = blockProjection(5120, 17408, 1);
  for (const uint32_t cores : {1U, 10U, 11U, 14U, 40U})
    for (const uint32_t rows : {32U, 33U, 48U, 64U, 65U}) {
      const LinearPlan plan =
          gpu(9, cores).plan({{5120, 17408}, rows, LinearEpilogue::Residual}, down);
      const bool staged = rows > kMaximumRegisterRows && rows <= kMaximumDecodeTileRows;
      const LinearTile tile =
          rows <= kMaximumRegisterRows ? LinearTile::GgufRegister : staged ? LinearTile::GgufStaged : LinearTile::GgufPrefill;
      if (plan.configuration().tile != tile ||
          plan.storageRows() != (rows <= 32 ? 32U : staged ? stagedTileRows(rows) : 128U))
        broke("Apple9 chunk tile", 9, cores, plan.workload());
    }
}

// A decode step's plans of 65 to 128 rows run two row groups at its rows,
// Apple9's of the staged tile (stagedRowGroups) and Apple10's and Apple11's of
// the matrix tile (matrixRowGroups), unsplit or split by the device's law, and
// past 128 rows the prefill tile; a prefill chunk's plans of those rows keep
// 1.7.0's prefill tiles (StepKind).
void decodeStepsRunRowGroups() {
  const Projection down = blockProjection(5120, 17408, 1), gdn = blockProjection(16640, 5120, 1);
  for (const auto [family, cores] : {std::pair{9U, 10U}, {10U, 16U}, {10U, 32U}, {11U, 12U}})
    for (const Projection *p : {&down, &gdn})
      for (const uint32_t rows : {65U, 80U, 81U, 96U, 97U, 112U, 113U, 128U, 129U}) {
        const LinearWorkload w{{p->outputSize, p->inputSize}, rows, LinearEpilogue::Residual};
        const LinearPlan step = gpu(family, cores, StepKind::Decode).plan(w, *p),
                         chunk = gpu(family, cores).plan(w, *p);
        // Apple9's step runs the staged tile's row groups, Apple10's and Apple11's the matrix tile's.
        const bool groups = rows <= kMaximumStagedRows, matrix = family >= 10;
        const LinearTile tile =
            !groups ? LinearTile::GgufPrefill : matrix ? LinearTile::GgufMatrix : LinearTile::GgufStaged;
        if (step.configuration().tile != tile ||
            (groups && step.storageRows() != (matrix ? matrixStorageRows(rows) : stagedTileRows(rows))) ||
            chunk.configuration().tile != LinearTile::GgufPrefill)
          broke("decode step row groups", family, cores, w);
      }
  static_assert(stagedTileRows(65) == 96 && stagedTileRows(96) == 96 && stagedTileRows(97) == 128 &&
                    stagedTileRows(128) == 128,
                "row groups of 48 rows at 65-96, of 64 at 97-128");
  static_assert(matrixStorageRows(65) == 96 && matrixStorageRows(96) == 96 && matrixStorageRows(97) == 128 &&
                    matrixStorageRows(128) == 128,
                "the matrix tile's two row groups hold the fragments of half the rows each: 96 at 65-96, 128 at "
                "97-128, as the staged tile's");
}

// A plan of some of a chunk's rows (Linear::plan with chunkRows) takes the
// chunk's configuration at its own rows' height, in no more scratch than the
// chunk's; rows past the chunk's are refused, on a Q4_K projection and on an
// af4g64 one, whose chunks past the staged tile run the int8 prefill tile on
// Apple10 and Apple11. gguf-projection checks that such a plan writes the
// chunk's bits.
void chunkRowPlans() {
  const Projection q4k = blockProjection(5120, 17408, 1), af4 = blockProjection(5120, 17408, 1, GGUF_FMT_AF4G64);
  for (const uint32_t family : {9U, 10U, 11U})
    for (const uint32_t cores : {10U, 12U, 16U, 20U, 32U, 40U}) {
      const Linear linear = gpu(family, cores);
      for (const Projection *weights : {&q4k, &af4})
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
          for (const uint32_t chunk : {1U, 7U, 32U, 33U, 64U, 65U, 300U, 2048U})
            for (const uint32_t rows : {1U, 4U, chunk}) {
              if (rows > chunk) continue;
              const LinearPlan whole = linear.plan(*weights, chunk, epilogue),
                               part = linear.plan(*weights, rows, epilogue, nullptr, chunk);
              const LinearScratchSize own = part.scratchSize(), bound = whole.scratchSize();
              if (part.configuration() != whole.configuration() || part.workload().rows != rows ||
                  own.partials > bound.partials || own.counters > bound.counters || own.input > bound.input ||
                  own.sums > bound.sums)
                broke("a chunk's rows departed from its plan", family, cores, part.workload());
            }
      rejects([&] { (void)linear.plan(q4k, 9, LinearEpilogue::None, nullptr, 8); },
              "a plan of a chunk's rows takes rows of its chunk", "a plan of more rows than its chunk was accepted");
    }
}

// Projections side by side (Projection::sideBySide): two or three tensors of
// one shape and format as one projection, its plan at every row count of a
// decode step and a prompt chunk on every family and core count the
// configuration and storage rows of one tensor's own plan, its scratch within
// the arena bound of its shape (Linear::scratchSize), its step's storage rows
// one tensor's; tensors of two formats or shapes, or of several segments,
// stand side by side in no projection, and one tensor or four are refused.
// gguf-projection checks that each tensor's columns are its own plan's bits.
void sideBySidePlans() {
  for (const uint32_t format : {GGUF_FMT_Q80, GGUF_FMT_Q4K, GGUF_FMT_IQ4XS, GGUF_FMT_AF4G64}) {
    const Projection tensor = blockProjection(1024, 2304, 1, format);
    const std::array<const Projection *, 3> three{&tensor, &tensor, &tensor};
    for (const size_t count : {size_t{2}, size_t{3}}) {
      const std::optional<Projection> side = Projection::sideBySide(std::span(three).first(count));
      require(side && side->outputSize == count * 1024 && side->planOutputs() == 1024 &&
                  side->shape().planOutputs == 1024 && side->blocks().segments.size() == count &&
                  side->blocks().segments.back().columnOffset == (count - 1) * 1024,
              "tensors of one shape and format stand side by side");
      for (const uint32_t family : {9U, 10U, 11U})
        for (const uint32_t cores : {10U, 12U, 16U, 20U, 32U, 40U})
          for (const StepKind step : {StepKind::Prefill, StepKind::Decode}) {
            const Linear linear(simulatedDevice(family, cores), Int8Activations::On, step);
            for (const uint32_t rows : {1U, 2U, 3U, 8U, 9U, 16U, 24U, 32U, 33U, 48U, 64U, 65U, 96U, 128U, 129U, 2048U}) {
              const LinearPlan own = linear.plan(tensor, rows), plan = linear.plan(*side, rows);
              const LinearScratchSize needed = plan.scratchSize(), bound = linear.scratchSize(side->shape(), rows);
              if (plan.configuration() != own.configuration() || plan.storageRows() != own.storageRows() ||
                  plan.workload().matrix.outputSize != side->outputSize || needed.input > bound.input ||
                  needed.sums > bound.sums || needed.partials > bound.partials || needed.counters > bound.counters ||
                  linear.storageRows(rows, side->shape()) != linear.storageRows(rows, tensor.shape()))
                broke("a projection side by side departed from its tensors' plan", family, cores, plan.workload());
            }
          }
    }
  }
  const Projection q80 = blockProjection(1024, 2304, 1, GGUF_FMT_Q80), q4k = blockProjection(1024, 2304, 1),
                   wide = blockProjection(2048, 2304, 1, GGUF_FMT_Q80), fused = blockProjection(2048, 2304, 2);
  for (const auto &[a, b] : {std::pair{&q80, &q4k}, std::pair{&q80, &wide}, std::pair{&fused, &fused}})
    require(!Projection::sideBySide(std::array<const Projection *, 2>{a, b}),
            "tensors of two formats, two shapes or several segments stood side by side");
  rejects([&] { (void)Projection::sideBySide(std::array<const Projection *, 1>{&q80}); },
          "projections side by side are two or three", "one tensor stood side by side");
  rejects([&] { (void)Projection::sideBySide(std::array<const Projection *, 4>{&q80, &q80, &q80, &q80}); },
          "projections side by side are two or three", "four tensors stood side by side");
}

// The int8 prefill tile (ProjectionCalibration::prefillInt8): Apple10 and
// Apple11 run every plan past the staged tile of a unit of af4g64, af5g64 and
// Q8_0 segments on it, at the prefill tile's storage, Apple9 none; a unit with a
// segment of another format keeps the prefill tile, and so does every unit
// with int8 activations off, whose plans are then the bf16 ones and whose
// device row is the same; a workload of no formats plans the same either way.
// The arena bound of a shape holds the tile's codes and scales of a full chunk
// wherever the family runs it on the shape's formats, or on a shape that names
// none, beside what its decode tiles' plans of up to 32 rows hold.
void prefillInt8Rule() {
  const Projection af4 = blockProjection(5120, 17408, 1, GGUF_FMT_AF4G64);
  const Projection af5 = blockProjection(12288, 5120, 3, GGUF_FMT_AF5G64);
  const Projection q80 = blockProjection(8192, 2048, 2, GGUF_FMT_Q80);
  const std::array<uint32_t, 3> mixedFormats{GGUF_FMT_AF4G64, GGUF_FMT_AF5G64, GGUF_FMT_Q4K};
  const Projection mixed = blockProjection(12288, 5120, mixedFormats);
  for (const auto [family, cores] : {std::pair{9U, 10U}, {9U, 40U}, {10U, 16U}, {10U, 32U}, {11U, 12U}}) {
    const Linear linear = gpu(family, cores), bf16(simulatedDevice(family, cores), Int8Activations::Off, StepKind::Prefill);
    const ProjectionCalibration &law = linear.calibration().projection;
    const bool int8 = family >= 10;
    require(law.prefillInt8 == int8 && bf16.calibration().projection.prefillInt8 == int8,
            "the int8 prefill tile takes Apple10's chunks past the staged tile and no Apple9 chunk, in the device's "
            "row whatever the setting");
    const LinearWorkload noFormats{{5120, 17408}, 2048, LinearEpilogue::None};
    require(linear.plan(noFormats).configuration() == bf16.plan(noFormats).configuration(),
            "int8 activations changed a prefill plan of no formats");
    for (const uint32_t rows : {33U, 64U, 65U, 128U, 2048U})
      for (const auto &[p, epilogue] : {std::pair{&af4, LinearEpilogue::Residual}, {&af5, LinearEpilogue::None},
                                        {&af4, LinearEpilogue::UpWithGate}, {&q80, LinearEpilogue::None},
                                        {&q80, LinearEpilogue::Residual}, {&mixed, LinearEpilogue::None}}) {
        const LinearPlan plan = linear.plan(*p, rows, epilogue), off = bf16.plan(*p, rows, epilogue);
        // A chunk of the decode tiles' rows runs Apple10's and Apple11's matrix tile, or Apple9's staged tile.
        const LinearTile decodeTile = family >= 10 ? LinearTile::GgufMatrix : LinearTile::GgufStaged;
        const LinearTile bf16Tile = rows <= kMaximumDecodeTileRows ? decodeTile : LinearTile::GgufPrefill;
        const LinearTile tile =
            bf16Tile == LinearTile::GgufPrefill && int8 && p != &mixed ? LinearTile::GgufPrefillInt8 : bf16Tile;
        const uint32_t k = plan.workload().matrix.inputSize;
        if (plan.configuration().tile != tile ||
            (tile == LinearTile::GgufPrefillInt8 &&
             (plan.storageRows() != (rows + 127) / 128 * 128 || plan.input() != LinearInput::Plain ||
              !sameScratch(plan.scratchSize(), {gguf_prefill_int8_table_bytes(k, plan.storageRows()),
                                                gguf_prefill_int8_scale_bytes(k, plan.storageRows()), 0, 0}))))
          broke("int8 prefill tile rule", family, cores, plan.workload());
        const LinearConfig bf16Config =
            tile == LinearTile::GgufPrefillInt8 ? LinearConfig{.tile = LinearTile::GgufPrefill} : plan.configuration();
        if (off.configuration() != bf16Config)
          broke("int8 activations off keep the bf16 plans", family, cores, off.workload());
      }
    constexpr uint32_t kChunk = SPLASH_PREFILL_TOKEN_BUDGET;
    const LinearScratchSize bound = linear.scratchSize({5120, 17408}, kChunk),
                            bf16Bound = bf16.scratchSize({5120, 17408}, kChunk),
                            decodeTiles = linear.scratchSize({5120, 17408}, kMaximumDecodeTileRows);
    require(bound.input == std::max<uint64_t>(decodeTiles.input, int8 ? gguf_prefill_int8_table_bytes(17408, kChunk) : 0) &&
                bound.sums == std::max<uint64_t>(decodeTiles.sums, int8 ? gguf_prefill_int8_scale_bytes(17408, kChunk) : 0),
            "the chunk scratch bound holds the int8 tile's codes and scales where the family runs it");
    require(bf16Bound.input == decodeTiles.input && bf16Bound.sums == decodeTiles.sums &&
                bf16Bound.partials == bound.partials && bf16Bound.counters == bound.counters &&
                bf16Bound.rotated == bound.rotated,
            "the chunk scratch bound with int8 activations off holds no int8 codes or scales");
    const LinearScratchSize q4k = linear.scratchSize(blockProjection(5120, 17408, 1).shape(), kChunk),
                            mixedBound = linear.scratchSize(mixed.shape(), kChunk),
                            mixedDecode = linear.scratchSize(mixed.shape(), kMaximumDecodeTileRows);
    require(sameScratch(linear.scratchSize(af4.shape(), kChunk), bound) && sameScratch(q4k, bf16Bound) &&
                mixedBound.input == mixedDecode.input && mixedBound.sums == mixedDecode.sums,
            "the chunk scratch bound holds int8 codes and scales for a shape of formats the tile does not take");
  }
}

// A GGUF decode projection whose split count is pinned on one core count.
struct SplitAnchor final {
  const char *projection;
  uint32_t cores, n, k, rows;
  LinearEpilogue epilogue;
  uint32_t segments, splits;
};

LinearPlan anchorPlan(uint32_t family, const SplitAnchor &anchor) {
  return gpu(family, anchor.cores).plan({{anchor.n, anchor.k}, anchor.rows, anchor.epilogue},
                                        blockProjection(anchor.n, anchor.k, anchor.segments));
}

void requireAnchorSplits(const LinearPlan &plan, const SplitAnchor &anchor, const char *policy) {
  const uint32_t splits = plan.configuration().splits;
  if (splits != anchor.splits)
    throw std::runtime_error(std::string(policy) + ": " + anchor.projection + " on " + std::to_string(anchor.cores) +
                             " cores plans " + std::to_string(splits) + " K splits, not " +
                             std::to_string(anchor.splits));
}

constexpr auto kNone = LinearEpilogue::None, kResidual = LinearEpilogue::Residual, kGateUp = LinearEpilogue::GateUp;

// Apple10's matrix law (kMatrixLaw: 600 ns of the device's rate in flight,
// 13.5 GB/s a core here) on Q4_K, for every projection kind: the grids of 32
// columns a threadgroup that keep fewer bytes in flight (64 a sixteenth of a
// bit) than it asks split, the rest run unsplit.
constexpr std::array kMatrixSplitAnchors{
    SplitAnchor{"27B down", 16, 5120, 17408, 8, kResidual, 1, 1},
    SplitAnchor{"27B down", 10, 5120, 17408, 8, kResidual, 1, 1},
    SplitAnchor{"27B down", 40, 5120, 17408, 8, kResidual, 1, 1},
    SplitAnchor{"35B output projection", 16, 2048, 4096, 8, kResidual, 1, 1},
    SplitAnchor{"35B shared-expert gate/up", 16, 512, 2048, 32, kGateUp, 1, 2},
    SplitAnchor{"27B gate/up", 16, 17408, 5120, 8, kGateUp, 1, 1},
    SplitAnchor{"two-segment fused projection", 16, 4096, 5120, 8, kNone, 2, 1},
    SplitAnchor{"27B three-segment GDN input", 40, 16640, 5120, 8, kNone, 3, 1},
    SplitAnchor{"27B vocabulary head", 16, 248320, 5120, 8, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 256 tensor", 16, 1024, 256, 32, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 3072 tensor", 16, 1024, 3072, 32, kNone, 1, 1},
    SplitAnchor{"Inkling-2-Mini k", 12, 1024, 2304, 8, kNone, 1, 1},
    SplitAnchor{"Inkling-2-Mini k", 40, 1024, 2304, 8, kNone, 1, 4},
    SplitAnchor{"Flash-Next hyper-connection down", 12, 320, 10240, 8, kNone, 1, 4},
    SplitAnchor{"Flash-Next hyper-connection down", 40, 320, 10240, 8, kNone, 1, 8}};

// One wave (four threadgroups per core) down to one 256-input unit per
// partition, eight waves while partitions keep 1024 inputs, at most eight;
// sixteen where eight leave fewer than three threadgroups per core and
// partitions keep 512 inputs, which 512 x 5120's sixteen would not.
constexpr std::array kRegisterSplitAnchors{
    SplitAnchor{"27B down", 40, 5120, 17408, 8, kResidual, 1, 8},
    SplitAnchor{"27B out_proj, four lanes", 40, 5120, 6144, 32, kResidual, 1, 4},
    SplitAnchor{"27B three-segment GDN input, two lanes", 40, 16640, 5120, 16, kNone, 3, 4},
    SplitAnchor{"27B three-segment attention input", 40, 14336, 5120, 8, kNone, 3, 4},
    SplitAnchor{"27B gate/up, three lanes", 40, 17408, 5120, 24, kGateUp, 1, 4},
    SplitAnchor{"27B vocabulary head", 40, 248320, 5120, 8, kNone, 1, 1},
    SplitAnchor{"27B down", 10, 5120, 17408, 8, kResidual, 1, 4},
    SplitAnchor{"27B three-segment GDN input", 10, 16640, 5120, 8, kNone, 3, 2},
    SplitAnchor{"35B two-segment GDN input", 40, 12288, 2048, 8, kNone, 2, 2},
    SplitAnchor{"35B two-segment GDN input", 80, 12288, 2048, 8, kNone, 2, 2},
    SplitAnchor{"35B shared-expert gate/up", 40, 512, 2048, 8, kGateUp, 1, 8},
    SplitAnchor{"35B shared-expert down", 40, 2048, 512, 8, kResidual, 1, 2},
    SplitAnchor{"35B output projection", 40, 2048, 4096, 8, kResidual, 1, 8},
    SplitAnchor{"35B vocabulary head", 40, 248320, 2048, 8, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 5120 tensor", 40, 1024, 5120, 8, kNone, 1, 8},
    SplitAnchor{"narrow 512 x 5120 tensor", 40, 512, 5120, 8, kNone, 1, 8},
    SplitAnchor{"narrow 1024 x 1024 tensor", 40, 1024, 1024, 8, kNone, 1, 4},
    SplitAnchor{"Flash-Next hyper-connection down", 12, 320, 10240, 8, kNone, 1, 8},
    SplitAnchor{"Flash-Next hyper-connection down", 14, 320, 10240, 8, kNone, 1, 16},
    SplitAnchor{"Flash-Next hyper-connection down, four lanes", 40, 320, 10240, 32, kNone, 1, 16}};

// fp32 K-split partials over the plan's rows and columns and one completion
// counter per 64-column tile (LinearPlan::scratchSize).
uint64_t splitPartialsBytes(const LinearPlan &plan) {
  return uint64_t{plan.configuration().splits} * plan.storageRows() * plan.workload().matrix.outputSize *
         sizeof(float);
}
uint64_t splitCountersBytes(const LinearPlan &plan) {
  return uint64_t{plan.workload().matrix.outputSize / GGUF_TILE_COLUMNS} * sizeof(uint32_t);
}
// The bf16 gate projection a GGUF gate/up plan writes before its up pass.
uint64_t gateBytes(const LinearPlan &plan) {
  return uint64_t{plan.storageRows()} * plan.workload().matrix.outputSize * sizeof(uint16_t);
}

// GGUF projections plan with their segments: the matrix split policy for
// every projection kind (Apple10), Apple9's register and staged tiles, exact
// split scratch over the tile's rows, and prefill tiles of 128 rows.
void ggufPlans() {
  const Linear linear = gpu(10, 16);
  // A block projection without segments would reach the dispatch paths with
  // nothing to index or encode.
  rejects([] { (void)Projection(5120, 17408, BlockWeights{}); }, "block projection has no segments",
          "a block projection without segments was accepted");
  // Its segments tile the leading columns in order: a gap, an overlap, another
  // input width or a segment past the end is not a projection; padding past
  // the last segment is.
  const auto segment = [](uint32_t n, uint32_t k, uint32_t offset) {
    QuantizedSegment s = QuantizedSegment::planes(GGUF_FMT_Q4K, n, k, {}, {}, {});
    s.columnOffset = offset;
    return s;
  };
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(256, 256, 320)}}); },
          "block segments do not tile the projection", "segments with a gap between them were accepted");
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(256, 256, 128)}}); },
          "block segments do not tile the projection", "overlapping segments were accepted");
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 512, 0)}}); },
          "block segments do not tile the projection", "a segment of another input width was accepted");
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(768, 256, 256)}}); },
          "block segments do not tile the projection", "a segment past the projection's end was accepted");
  require(Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(256, 256, 256)}}).blocks().segments.size() == 2,
          "a projection with padding past its segments was rejected");
  // Apple10 runs every plan of 3 to 64 rows (a chunk's; a decode step's to 128)
  // on the matrix tile, its fragments' rows stored, K split by its law: the 27B
  // down's 160 threadgroups keep 737 KB in flight, past the 130 KB that 16 cores
  // of 13.5 GB/s deliver in 600 ns, and run unsplit.
  const LinearWorkload down{{5120, 17408}, 8, LinearEpilogue::Residual};
  const LinearPlan single = linear.plan(down, blockProjection(5120, 17408, 1));
  require(single.configuration() == LinearConfig{.tile = LinearTile::GgufMatrix, .splits = 1} &&
              single.input() == LinearInput::Plain && single.storageRows() == 16 &&
              sameScratch(single.scratchSize(), LinearScratchSize{}),
          "GGUF single-tensor decode plan");
  for (const SplitAnchor &anchor : kMatrixSplitAnchors) {
    const LinearPlan plan = anchorPlan(10, anchor);
    require(plan.configuration().tile == LinearTile::GgufMatrix, "Apple10 GGUF matrix plan");
    requireAnchorSplits(plan, anchor, "GGUF matrix split policy");
  }
  // A split matrix plan's fp32 partials over its rows and one counter per 32 columns; a gate/up pair's gate pass
  // writes its fragments' rows.
  const LinearPlan gateUpPlan = linear.plan({{512, 2048}, 32, LinearEpilogue::GateUp},
                                            blockProjection(512, 2048, 1));
  require(gateUpPlan.configuration().splits == 2 && gateUpPlan.gateScratchBytes() == gateBytes(gateUpPlan) &&
              gateUpPlan.scratchSize().partials == uint64_t{2} * 32 * 512 * sizeof(float) &&
              gateUpPlan.scratchSize().counters == 512 / GGUF_COLUMN_COLUMNS * sizeof(uint32_t),
          "GGUF matrix gate/up runs a gate pass into the gate scratch");
  // The matrix tile holds fragments of 16 rows: 24 rows run two.
  const LinearPlan three = linear.plan({{5120, 17408}, 24, LinearEpilogue::Residual},
                                       blockProjection(5120, 17408, 1));
  require(three.storageRows() == 32 && three.configuration() == single.configuration(),
          "GGUF matrix three-lane plans run two fragments");
  for (const auto [rows, storage] :
       {std::pair{1U, 16U}, {8U, 16U}, {9U, 16U}, {17U, 32U}, {25U, 32U}, {32U, 32U}, {33U, 48U}, {48U, 48U},
        {49U, 64U}, {64U, 64U}, {65U, 128U}, {100U, 128U}, {129U, 256U}, {2048U, 2048U}}) {
    const LinearPlan prefill = linear.plan({{5120, 17408}, rows, LinearEpilogue::UpWithGate},
                                           blockProjection(5120, 17408, 1));
    // Plans of up to 64 rows take the decode tiles, here the matrix tile (an
    // up-with-gate of one row too, which the column tile does not run), unsplit;
    // the 128-row prefill tile takes the rest.
    const bool matrix = rows <= kMaximumDecodeTileRows;
    require(prefill.storageRows() == storage && prefill.gateScratchBytes() == gateBytes(prefill) &&
                prefill.configuration().tile == (matrix ? LinearTile::GgufMatrix : LinearTile::GgufPrefill) &&
                prefill.configuration().splits == 1 && prefill.scratchSize().partials == 0,
            "GGUF tile rows and splits");
  }
  // The staged and matrix tiles run plans of at most 128 rows, two row groups.
  const LinearWorkload longPrefill{{5120, 17408}, 129, LinearEpilogue::None};
  rejects([&] { (void)Linear::plan(longPrefill, {.tile = LinearTile::GgufStaged}, FloatOutput::BFloat16); },
          "the staged block tile runs plans of up to 128 rows", "the staged tile took a plan past 128 rows");
  rejects([&] { (void)Linear::plan(longPrefill, {.tile = LinearTile::GgufMatrix}, FloatOutput::BFloat16); },
          "the matrix block tile takes up to two row groups' rows, a group per simdgroup, a gate/up in two passes",
          "the matrix tile took a plan past 128 rows");
  rejects([&] { (void)Linear::plan(down, {.tile = LinearTile::GgufStaged, .splits = 3}, FloatOutput::BFloat16); },
          "staged block splits take whole 32-input groups", "the staged tile split K three ways");
  // The arena bound, of the shape without its formats, covers every plan on
  // the decode tiles; here the 64-row tile's is the largest, which fused and
  // gate/up plans share.
  const ProjectionShape downShape{5120, 17408};
  const LinearPlan most = linear.plan({{5120, 17408}, kMaximumDecodeTileRows, LinearEpilogue::Residual},
                                      blockProjection(5120, 17408, 1));
  require(most.storageRows() == kMaximumDecodeTileRows &&
              linear.scratchSize(downShape, kMaximumDecodeTileRows).partials == most.scratchSize().partials,
          "GGUF decode scratch bound");
  // Float projections take the neural accelerator tile from 1.5 of its
  // 64 x 32 tiles per core for BF16 values, 2 for F32 values and 3 for a
  // quantized format's rows: on 16 cores a router of 256 experts from 129,
  // 193 and 321 rows, F32 alpha/beta (N 64) from 961; never on Apple9 or
  // below 16 rows.
  const Linear oneCore = gpu(10, 1);
  constexpr uint32_t kF32 = QuantizedSegment::kFloat32;
  for (const auto [format, first] : {std::pair{uint32_t{GGUF_FMT_BF16}, 129U}, std::pair{kF32, 193U},
                                     std::pair{uint32_t{GGUF_FMT_AF8G64}, 321U}, std::pair{uint32_t{GGUF_FMT_Q4K}, 321U}})
    require(linear.ggufFloatTile(32, 256, format) == FloatTile::Simdgroup &&
                linear.ggufFloatTile(first - 1, 256, format) == FloatTile::Simdgroup &&
                linear.ggufFloatTile(first, 256, format) == FloatTile::NeuralAccelerator &&
                linear.ggufFloatTile(2048, 256, format) == FloatTile::NeuralAccelerator &&
                oneCore.ggufFloatTile(15, 256, format) == FloatTile::Simdgroup &&
                oneCore.ggufFloatTile(16, 256, format) == FloatTile::NeuralAccelerator,
            std::string("GGUF float tile rule of ") + (format == kF32 ? "f32" : kQuantFormats[format].name) + " rows");
  require(linear.ggufFloatTile(960, 64, kF32) == FloatTile::Simdgroup &&
              linear.ggufFloatTile(961, 64, kF32) == FloatTile::NeuralAccelerator,
          "GGUF float tile rule of F32 alpha/beta");

  // Apple9 decodes every GGUF width with the exact register tile, all lanes
  // in one threadgroup, and K splits from the core count; prefill stages.
  for (const SplitAnchor &anchor : kRegisterSplitAnchors) {
    const LinearPlan plan = anchorPlan(9, anchor);
    require(plan.configuration().tile == LinearTile::GgufRegister && plan.input() == LinearInput::Table16,
            "Apple9 GGUF register plan");
    requireAnchorSplits(plan, anchor, "Apple9 GGUF register split policy");
  }
  const Linear m3 = gpu(9, 40);
  for (const uint32_t rows : {8U, 32U}) {
    const LinearPlan plan = m3.plan({{5120, 17408}, rows, LinearEpilogue::Residual},
                                    blockProjection(5120, 17408, 1));
    const LinearScratchSize size = plan.scratchSize();
    require(plan.configuration().splits == 8 && size.input == tableBytes(17408, rows) &&
                size.sums == tableSumsBytes(LinearInput::Table16, 17408, rows) &&
                size.partials == splitPartialsBytes(plan) && size.counters == splitCountersBytes(plan) &&
                plan.gateScratchBytes() == 0,
            "Apple9 GGUF register scratch");
  }
  const LinearPlan head = m3.plan({{248320, 5120}, 8, LinearEpilogue::None},
                                  blockProjection(248320, 5120, 1));
  require(head.scratchSize().partials == 0 && head.scratchSize().counters == 0,
          "Apple9 GGUF register scratch without splits holds partials or counters");
  const LinearPlan registerGateUp = m3.plan({{17408, 5120}, 16, LinearEpilogue::GateUp},
                                            blockProjection(17408, 5120, 1));
  require(registerGateUp.gateScratchBytes() == gateBytes(registerGateUp),
          "Apple9 GGUF gate/up runs a gate pass into the gate scratch");
  // Past the register tile's 32 rows the staged tile holds a plan up to 64, the prefill tile past them.
  for (const uint32_t rows : {33U, 64U, 100U})
    require(m3.plan({{5120, 17408}, rows, LinearEpilogue::Residual}, blockProjection(5120, 17408, 1))
                    .configuration()
                    .tile == (rows <= kMaximumDecodeTileRows ? LinearTile::GgufStaged : LinearTile::GgufPrefill),
            "Apple9 GGUF plans past the register tile stage up to 64 rows, take the prefill tile past them");
  require(m3.ggufFloatTile(2048, 256, QuantizedSegment::kFloat32) == FloatTile::Simdgroup,
          "Apple9 float projections take the simdgroup tile");
  // The bound covers the register tile's 32 rows and the staged tile's 64.
  const LinearScratchSize bound = m3.scratchSize(downShape, kMaximumDecodeTileRows);
  require(bound.partials ==
              std::max(m3.plan({down.matrix, kMaximumRegisterRows, LinearEpilogue::Residual},
                               blockProjection(5120, 17408, 1)).scratchSize().partials,
                       m3.plan({down.matrix, kMaximumDecodeTileRows, LinearEpilogue::Residual},
                               blockProjection(5120, 17408, 1)).scratchSize().partials),
          "Apple9 GGUF decode scratch bound");
  // Apple9 stages the IQ2, IQ3_XXS and IQ1 formats wherever the staged tile
  // holds the lanes' rows unpadded, and Q2_K from two lanes. A projection
  // stages only when every quantized segment's format does; its plan binds
  // the register tile's rows, and the decode scratch bound covers both tiles.
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const auto plan = [&](std::initializer_list<uint32_t> formats) {
      const uint32_t n = uint32_t(formats.size()) * 1024;
      return m3.plan({{n, 5120}, lanes * 8, LinearEpilogue::None},
                     blockProjection(n, 5120, formats));
    };
    const LinearTile iq = lanes == 3 ? LinearTile::GgufRegister : LinearTile::GgufStaged;
    const LinearTile q2k = lanes == 2 || lanes == 4 ? LinearTile::GgufStaged : LinearTile::GgufRegister;
    bool staged = true;
    for (const uint32_t format :
         {GGUF_FMT_IQ3XXS, GGUF_FMT_IQ2XXS, GGUF_FMT_IQ2XS, GGUF_FMT_IQ2S, GGUF_FMT_IQ1S, GGUF_FMT_IQ1M})
      staged = staged && plan({format}).configuration().tile == iq;
    require(staged && plan({GGUF_FMT_IQ3XXS, GGUF_FMT_IQ2S, GGUF_FMT_IQ1M}).configuration().tile == iq,
            "Apple9 stages the IQ2, IQ3_XXS and IQ1 formats at unpadded lanes");
    require(plan({GGUF_FMT_Q2K}).configuration().tile == q2k, "Apple9 stages Q2_K from two unpadded lanes");
    for (const uint32_t format : {GGUF_FMT_Q4K, GGUF_FMT_Q6K, GGUF_FMT_IQ4XS, GGUF_FMT_IQ3S, GGUF_FMT_Q80,
                                  GGUF_FMT_Q40, GGUF_FMT_Q50, GGUF_FMT_Q51, GGUF_FMT_MXFP4, GGUF_FMT_BF16})
      require(plan({format}).configuration().tile == LinearTile::GgufRegister, "Apple9 keeps other formats' registers");
    // F16, which the register tile does not take (quant_register_format), stages at every lane count, alone or
    // beside a register format.
    require(plan({GGUF_FMT_F16}).configuration().tile == LinearTile::GgufStaged &&
                plan({GGUF_FMT_Q4K, GGUF_FMT_F16}).configuration().tile == LinearTile::GgufStaged,
            "Apple9 stages a projection with an F16 segment");
    // Its decode storage holds the staged tile's rows: three lanes' 24 rows pad to 32.
    const ProjectionShape f16Shape = blockProjection(1024, 5120, std::array<uint32_t, 1>{GGUF_FMT_F16}).shape();
    require(!f16Shape.registerFormats &&
                m3.storageRows(lanes * 8, f16Shape) == plan({GGUF_FMT_F16}).storageRows() &&
                m3.storageRows(lanes * 8, ProjectionShape{1024, 5120}) == lanes * 8,
            "Apple9 decode storage holds an F16 projection's staged rows");
    require(plan({GGUF_FMT_IQ3XXS, GGUF_FMT_Q4K}).configuration().tile == LinearTile::GgufRegister &&
                plan({GGUF_FMT_IQ3XXS}).storageRows() == plan({GGUF_FMT_Q4K}).storageRows(),
            "Apple9 keeps a mixed projection's registers, and a staged plan binds the register rows");
    // A gate/up plan runs its gate on the same tile: it stages only when the gate's formats do too.
    const LinearWorkload gateUp{{1024, 5120}, lanes * 8, LinearEpilogue::GateUp};
    const Projection up = blockProjection(1024, 5120, 1, GGUF_FMT_IQ2XXS),
                     iqGate = blockProjection(1024, 5120, 1, GGUF_FMT_IQ2XS), q4kGate = blockProjection(1024, 5120, 1);
    require(m3.plan(gateUp, up, &iqGate).configuration().tile == iq &&
                m3.plan(gateUp, up, &q4kGate).configuration().tile == LinearTile::GgufRegister,
            "Apple9 staged a gate/up plan whose gate keeps the register tile");
    const LinearPlan stagedDown = m3.plan({down.matrix, lanes * 8, LinearEpilogue::Residual},
                                          blockProjection(5120, 17408, 1, GGUF_FMT_IQ2XS));
    const LinearPlan registerPlan = m3.plan({down.matrix, lanes * 8, LinearEpilogue::Residual},
                                            blockProjection(5120, 17408, 1));
    for (const LinearScratchSize size : {stagedDown.scratchSize(), registerPlan.scratchSize()})
      require(bound.input >= size.input && bound.sums >= size.sums && bound.partials >= size.partials &&
                  bound.counters >= size.counters,
              "Apple9 GGUF decode scratch bound covers both tiles");
  }
  // Apple9 registers projections of whole 64-input spans in a format of one group per meta unit, of fewer inputs
  // than 256 or no multiple of it (Qwen3.8-Flash-Next's Q8_0 hyper-connection projections take 320), and its decode
  // scratch bound covers them. A format of several groups per meta unit (Q2_0) stages such a projection, and its bound
  // holds no table.
  for (const uint32_t k : {64U, 192U, 320U, 640U}) {
    const Projection q80 = blockProjection(1024, k, 1, GGUF_FMT_Q80), q20 = blockProjection(1024, k, 1, GGUF_FMT_Q20);
    const LinearScratchSize narrow = m3.scratchSize(q80.shape(), kMaximumDecodeTileRows);
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const LinearWorkload w{{1024, k}, lanes * 8, kNone};
      const LinearPlan plan = m3.plan(w, q80);
      const LinearScratchSize need = plan.scratchSize();
      require(plan.configuration().tile == LinearTile::GgufRegister && narrow.input >= need.input &&
                  narrow.sums >= need.sums && narrow.partials >= need.partials && narrow.counters >= need.counters,
              "Apple9 registers Q8_0 projections of " + std::to_string(k) + " inputs within the scratch bound");
      require(m3.plan(w, q20).configuration().tile == LinearTile::GgufStaged &&
                  m3.scratchSize(q20.shape(), kMaximumDecodeTileRows).input == 0,
              "Apple9 registers a Q2_0 projection of " + std::to_string(k) + " inputs");
    }
  }
  // Apple10 runs every format on the matrix tile (IQ2_XXS's grid among them), on every grid: its register and staged
  // tiles take no plan but the int8 decode tiles' (decodeInt8). A gate/up pair runs two passes, its decode storage
  // holds the matrix tile's fragments of the shape's rows, and the decode scratch bound covers every plan.
  require(linear.plan({{5120, 17408}, 24, LinearEpilogue::Residual},
                      blockProjection(5120, 17408, 1, GGUF_FMT_IQ2XXS)).configuration().tile == LinearTile::GgufMatrix,
          "Apple10 runs every format on the matrix tile");
  {
    // Flash-Next's hyper-connection down projection, Q8_0 of ten threadgroups (87 KB in flight unsplit), splits K
    // by the law at every count of rows from 3, a decode step's or a prompt chunk's: in two on 20 cores of 13.5 GB/s
    // (162 KB in 600 ns), in four on 32 (259 KB).
    const Projection hyperDown = blockProjection(320, 10240, 1, GGUF_FMT_Q80);
    for (const auto [cores, splits] : {std::pair{20U, 2U}, {32U, 4U}})
      for (uint32_t rows = GGUF_COLUMN_ROWS + 1; rows <= kMaximumDecodeTileRows; ++rows)
        require(gpu(10, cores).plan({{320, 10240}, rows, kNone}, hyperDown).configuration() ==
                    LinearConfig{.tile = LinearTile::GgufMatrix, .splits = splits},
                "Apple10 splits the hyper-connection down projection by the matrix law");
    const Linear m5 = gpu(10, 32);
    for (const uint32_t format : {GGUF_FMT_Q4K, GGUF_FMT_IQ2XXS, GGUF_FMT_Q20})
      for (const auto [n, k] : {std::pair{1024U, 5120U}, {512U, 2560U}, {2560U, 6144U}, {320U, 10240U}}) {
        const Projection p = blockProjection(n, k, 1, format), gate = blockProjection(n, k, 1, format);
        const LinearScratchSize bound = m5.scratchSize(p.shape(), kMaximumDecodeTileRows);
        for (uint32_t rows = 8; rows <= kMaximumDecodeTileRows; rows += 8) {
          const LinearPlan plan = m5.plan({{n, k}, rows, kNone}, p);
          const LinearPlan pairPlan = m5.plan({{n, k}, rows, kGateUp}, p, &gate);
          const LinearScratchSize need = plan.scratchSize();
          require(plan.configuration().tile == LinearTile::GgufMatrix &&
                      pairPlan.configuration().tile == LinearTile::GgufMatrix &&
                      !pairPlan.configuration().oneGateUpPass &&
                      std::max(plan.storageRows(), pairPlan.storageRows()) == m5.storageRows(rows, p.shape()) &&
                      plan.storageRows() == matrixStorageRows(rows) && bound.input >= need.input &&
                      bound.sums >= need.sums && bound.partials >= need.partials && bound.counters >= need.counters,
                  "Apple10 decode storage and scratch bound follow the matrix tile");
        }
      }
  }
  // No unrotated plan runs the staged tile's 16-row lane on any family: Apple10 and Apple11 run the matrix tile,
  // whose fragments hold 16 rows, and Apple9 never took the lane. A projection of float segments alone takes the
  // float tiles, its quantized plan none.
  for (const uint32_t family : {9U, 10U, 11U})
    for (const uint32_t cores : {0U, 10U, 12U, 16U, 20U, 32U, 40U, 80U}) {
      const Linear device = gpu(family, cores);
      for (const auto [n, k] : {std::pair{17408U, 5120U}, {5120U, 17408U}, {248320U, 5120U}, {1024U, 5120U}})
        for (const uint32_t format : {GGUF_FMT_IQ3XXS, GGUF_FMT_IQ4XS, GGUF_FMT_Q4K, GGUF_FMT_AF4G64, GGUF_FMT_Q80}) {
          const Projection p = blockProjection(n, k, 1, format);
          const LinearPlan one = device.plan({{n, k}, 8, kNone}, p);
          require(!one.configuration().sixteenRows &&
                      one.storageRows() == (family >= 10 ? 16U : 8U) &&
                      one.storageRows() <= device.storageRows(8, p.shape()),
                  "one lane's plan stores its tile's rows: family " + std::to_string(family) + ", " +
                      std::to_string(cores) + " cores, " + std::to_string(n) + " x " + std::to_string(k));
        }
    }
  // Apple9's staged tile splits K by the register tile's tiers of eight
  // partitions: on 40 cores 17408 x 5120 in four, 5120 x 17408 and 320 x
  // 10240 (sixteen on the register tile) in eight.
  for (const auto [n, k, splits] : {std::tuple{17408U, 5120U, 4U}, {5120U, 17408U, 8U}, {320U, 10240U, 8U}}) {
    const LinearPlan decode = m3.plan({{n, k}, 8, LinearEpilogue::None},
                                      blockProjection(n, k, 1, GGUF_FMT_IQ2XXS));
    require(decode.configuration() == LinearConfig{.tile = LinearTile::GgufStaged, .splits = splits},
            "Apple9 staged split tiers");
  }
  for (const LinearConfig config :
       {LinearConfig{.tile = LinearTile::GgufRegister, .splits = 3},
        LinearConfig{.tile = LinearTile::GgufRegister, .splits = 2 * LinearConfig::kMaximumSplits}})
    rejects([&] { (void)Linear::plan(down, config, FloatOutput::BFloat16); },
            "the register block tile takes up to four table tiles' rows and a K unit per split",
            "the register tile took a K split that is not a power of two up to the maximum");
  // Split boundaries fall on coefficient units, of 64 inputs at the fewest, and the register tile holds no more
  // rows than its four table tiles, a plan of 33 rows on the decode tiles stages.
  const LinearWorkload chunk{{5120, 17408}, 128, LinearEpilogue::None};
  rejects(
      [&] {
        (void)Linear::plan({{5120, 192}, 8, LinearEpilogue::None},
                           {.tile = LinearTile::GgufRegister, .splits = 4}, FloatOutput::BFloat16);
      },
      "the register block tile takes up to four table tiles' rows and a K unit per split",
      "the register tile split three K units four ways");
  for (const uint32_t rows : {kMaximumRegisterRows + 1, 128U})
    rejects(
        [&] {
          (void)Linear::plan({chunk.matrix, rows, LinearEpilogue::None}, {.tile = LinearTile::GgufRegister},
                             FloatOutput::BFloat16);
        },
        "the register block tile takes up to four table tiles' rows and a K unit per split",
        "a plan of " + std::to_string(rows) + " rows took the register tile");
  // The prefill tile runs plans of any rows on their matrix grid, unsplit.
  rejects([&] { (void)Linear::plan(chunk, {.tile = LinearTile::GgufPrefill, .splits = 2}, FloatOutput::BFloat16); },
          "invalid block prefill configuration", "the prefill tile split K");
  // No plan spreads its walks: Apple9's staged tile walks in lockstep, and Apple10 and Apple11 run the matrix tile,
  // as every plan past the decode tiles does.
  const Projection iq = blockProjection(5120, 17408, 1, GGUF_FMT_IQ2XXS);
  for (const uint32_t family : {9U, 10U, 11U})
    for (const uint32_t cores : {8U, 12U, 15U, 16U, 20U, 40U}) {
      const Linear device = gpu(family, cores);
      for (const uint32_t rows : {8U, 33U, 64U}) {
        const LinearConfig decode =
            device.plan({down.matrix, rows, LinearEpilogue::Residual}, iq).configuration();
        require(decode.tile == (family >= 10 ? LinearTile::GgufMatrix : LinearTile::GgufStaged) && !decode.spread,
                "GGUF decode walk default");
      }
      for (const uint32_t rows : {65U, 128U})
        require(!device.plan({down.matrix, rows, LinearEpilogue::Residual}, iq)
                     .configuration().spread,
                "a GGUF plan past the decode tiles spreads its walks");
      // A gate/up pair runs two passes, of one format or two, its gate pass into the gate scratch. The decode scratch
      // bound, of a shape without formats, covers the partials of either: the matrix tile's over its rows and a
      // counter per 32 columns, Apple9's staged tile's over its tile's rows and one per 64.
      for (const uint32_t n : {1024U, 17408U}) {
        const LinearMatrix matrix{n, 5120};
        const Projection up = blockProjection(n, 5120, 1, GGUF_FMT_IQ2XXS),
                         same = blockProjection(n, 5120, 1, GGUF_FMT_IQ2XXS),
                         other = blockProjection(n, 5120, 1, GGUF_FMT_IQ2XS);
        for (const uint32_t rows : {8U, 32U}) {
          const LinearWorkload gateUp{matrix, rows, LinearEpilogue::GateUp};
          const LinearPlan one = device.plan(gateUp, up, &same), two = device.plan(gateUp, up, &other);
          require(!one.configuration().oneGateUpPass && !two.configuration().oneGateUpPass &&
                      one.gateScratchBytes() == gateBytes(one),
                  "GGUF gate/up pass default");
          const LinearScratchSize bound = device.scratchSize({n, 5120}, kMaximumDecodeTileRows);
          for (const LinearPlan *plan : {&one, &two}) {
            const LinearScratchSize need = plan->scratchSize();
            const uint32_t splits = plan->configuration().splits;
            const bool matrixTile = plan->configuration().tile == LinearTile::GgufMatrix;
            const uint64_t partialRows = matrixTile ? rows : plan->storageRows();
            const uint32_t counterColumns = matrixTile ? GGUF_COLUMN_COLUMNS : GGUF_TILE_COLUMNS;
            require(bound.partials >= need.partials && bound.counters >= need.counters &&
                        need.partials == (splits > 1 ? uint64_t{splits} * partialRows * n * 4 : 0) &&
                        need.counters == (splits > 1 ? n / counterColumns * 4 : 0),
                    "GGUF gate/up scratch");
          }
        }
      }
    }
  // The 27B's pair, 17408 x 5120 in MLX 4-bit, runs the matrix tile in two passes, unsplit, on 16 to 40 cores.
  for (const uint32_t cores : {16U, 20U, 32U, 40U}) {
    const Projection up = blockProjection(17408, 5120, 1, GGUF_FMT_AF4G64),
                     gate = blockProjection(17408, 5120, 1, GGUF_FMT_AF4G64);
    const LinearConfig c =
        gpu(10, cores).plan({{17408, 5120}, 8, LinearEpilogue::GateUp}, up, &gate).configuration();
    require(c == LinearConfig{.tile = LinearTile::GgufMatrix, .splits = 1},
            "the 27B gate/up pair's passes on " + std::to_string(cores) + " cores");
  }
  rejects(
      [&] {
        (void)Linear::plan(down, {.tile = LinearTile::GgufRegister, .splits = 8, .spread = true},
                           FloatOutput::BFloat16);
      },
      "only the staged block tile spreads its walks", "the register tile spread its walks");
  rejects(
      [&] {
        (void)Linear::plan(down, {.tile = LinearTile::GgufStaged, .oneGateUpPass = true}, FloatOutput::BFloat16);
      },
      "only a staged or column block gate/up of up to 32 rows runs gate and up in one pass",
      "a residual plan ran gate/up in one pass");
  rejects(
      [&] {
        (void)Linear::plan({{17408, 5120}, kMaximumPairRows + 1, LinearEpilogue::GateUp},
                           {.tile = LinearTile::GgufStaged, .oneGateUpPass = true}, FloatOutput::BFloat16);
      },
      "only a staged or column block gate/up of up to 32 rows runs gate and up in one pass",
      "a gate/up pair of 33 rows ran in one pass");
  // The 16-row tile holds a plan of at most 8 rows on the staged tile only.
  for (const auto &[w, config] :
       {std::pair{down, LinearConfig{.tile = LinearTile::GgufRegister, .sixteenRows = true}},
        {LinearWorkload{down.matrix, 16, LinearEpilogue::Residual},
         LinearConfig{.tile = LinearTile::GgufStaged, .sixteenRows = true}}})
    rejects([&] { (void)Linear::plan(w, config, FloatOutput::BFloat16); },
            "only a staged block plan of at most 8 rows runs the 16-row tile",
            "the 16-row tile took more rows than 8 or another tile");
  // Int8 activations run on a staged plan whose K splits keep whole PQ2_0 blocks, past the int8 tiles' 32 rows in
  // row groups of them (int8RowGroups): not on the register tile, past the staged tile's 128 rows, or 16 splits of
  // 5120 inputs (160 groups, no multiple of 16 blocks of 4).
  for (const auto &[w, config, expected] :
       {std::tuple{LinearWorkload{down.matrix, 16, LinearEpilogue::Residual},
                   LinearConfig{.tile = LinearTile::GgufRegister, .decodeInt8 = true}, "runs int8 activations"},
        {LinearWorkload{down.matrix, kMaximumStagedRows + 1, LinearEpilogue::Residual},
         LinearConfig{.tile = LinearTile::GgufStaged, .decodeInt8 = true}, "plans of up to 128 rows"},
        {LinearWorkload{{5120, 5120}, 16, LinearEpilogue::None},
         LinearConfig{.tile = LinearTile::GgufStaged, .splits = 16, .decodeInt8 = true}, "runs int8 activations"}})
    rejects([&] { (void)Linear::plan(w, config, FloatOutput::BFloat16); }, expected,
            "int8 activations took another tile, rows past the staged tile's or a split of a PQ2_0 block");
  // A plan of fewer rows on the int8 tiles stores their 16.
  require(Linear::plan(down, {.tile = LinearTile::GgufStaged, .decodeInt8 = true}, FloatOutput::BFloat16)
                  .storageRows() == 16,
          "an 8-row plan on the int8 tiles stores 16 rows");
  rejects([&] { (void)Linear::plan(chunk, {.tile = LinearTile::GgufPrefill, .spread = true}, FloatOutput::BFloat16); },
          "only the staged block tile spreads its walks", "the prefill tile spread its walks");
}

// A plan of one or two rows runs the column tile on every device and format at its own rows, a gate/up pair of one
// format in one pass and one of two formats on the tiles of more rows; a plan of more rows as before. It takes scratch
// only split, its partials of its rows (a pair's of both tensors), and a projection of 4096 columns of four bits per
// weight or more splits on no device here. The column tile takes no more rows, no partition of fewer groups than its
// simdgroups and no up-with-gate.
void columnPlans() {
  for (const auto [family, cores] : {std::pair{9U, 10U}, {10U, 32U}, {11U, 12U}})
    for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format)
      for (const uint32_t k : {2048U, 4096U}) {
        if (k % quant_column_unit(kQuantFormats[format].block_elements)) continue;
        const Linear linear = gpu(family, cores);
        const Projection up = blockProjection(4096, k, 1, format), gate = blockProjection(4096, k, 1, format);
        const uint32_t other = format == GGUF_FMT_Q4K ? GGUF_FMT_Q80 : GGUF_FMT_Q4K;
        const Projection mixed = blockProjection(4096, k, 1, other);
        for (const LinearEpilogue e : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
          for (uint32_t rows = 1; rows <= kMaximumDecodeTileRows; ++rows) {
            const bool gateUp = e == LinearEpilogue::GateUp;
            const LinearWorkload w{{4096, k}, rows, e};
            const LinearPlan plan = linear.plan(w, up, gateUp ? &gate : nullptr);
            const LinearConfig c = plan.configuration();
            if (rows > GGUF_COLUMN_ROWS) {
              require(c.tile != LinearTile::GgufColumn, "a plan of more than two rows ran the column tile");
              continue;
            }
            require(c.tile == LinearTile::GgufColumn && c.oneGateUpPass == gateUp,
                    "a plan of one or two rows runs the column tile");
            require(c.splits == 1 || !quant_four_bits(format), "a projection of four bits or more split");
            const LinearScratchSize split{0, 0, uint64_t{c.splits} * rows * (gateUp ? 2 : 1) * 4096 * sizeof(float),
                                          4096 / GGUF_COLUMN_COLUMNS * sizeof(uint32_t)};
            require(plan.storageRows() == rows &&
                        sameScratch(plan.scratchSize(), c.splits > 1 ? split : LinearScratchSize{}) &&
                        plan.gateScratchBytes() == 0 && plan.input() == LinearInput::Plain,
                    "the column tile stores its rows and takes scratch for its partials alone");
            if (gateUp)
              require(linear.plan(w, up, &mixed).configuration().tile != LinearTile::GgufColumn,
                      "a gate/up pair of two formats ran the column tile");
          }
      }
  for (const auto &[w, config, rule] :
       {std::tuple{LinearWorkload{{4096, 2048}, GGUF_COLUMN_ROWS + 1, LinearEpilogue::None},
                   LinearConfig{.tile = LinearTile::GgufColumn}, "more rows"},
        {LinearWorkload{{4096, 2048}, 1, LinearEpilogue::None},
         LinearConfig{.tile = LinearTile::GgufColumn, .splits = 16}, "a partition of fewer groups than simdgroups"},
        {LinearWorkload{{4096, 2048}, 1, LinearEpilogue::UpWithGate},
         LinearConfig{.tile = LinearTile::GgufColumn}, "an up-with-gate"}})
    rejects([&] { (void)Linear::plan(w, config, FloatOutput::BFloat16); },
            "the column block tile takes up to GGUF_COLUMN_ROWS rows, a group per simdgroup, no up-with-gate",
            std::string("the column tile ran ") + rule);
  std::cout << "column plans: ok\n";
}

// A plan of 3 to 64 rows runs the decode tiles, whoever runs it: the configuration and storage of the plan of its table
// tiles' rows (8 a tile), whose rows past its own are padding, on every device row and epilogue, a gate/up pair's
// among them. One or two rows run the column tile (columnPlans).
void decodeTileRows() {
  const Projection down = blockProjection(5120, 17408, 1), iq = blockProjection(17408, 5120, 1, GGUF_FMT_IQ2XXS),
                   af4 = blockProjection(17408, 5120, 1, GGUF_FMT_AF4G64),
                   af4Gate = blockProjection(17408, 5120, 1, GGUF_FMT_AF4G64);
  for (const auto [family, cores] : {std::pair{9U, 10U}, {9U, 40U}, {10U, 16U}, {10U, 32U}, {11U, 12U}}) {
    const Linear linear = gpu(family, cores);
    for (const auto &[p, epilogue, gate] :
         {std::tuple{&down, LinearEpilogue::Residual, static_cast<const Projection *>(nullptr)},
          {&iq, LinearEpilogue::None, nullptr}, {&af4, LinearEpilogue::GateUp, &af4Gate}})
      for (uint32_t rows = GGUF_COLUMN_ROWS + 1; rows <= kMaximumDecodeTileRows; ++rows) {
        const LinearPlan plan = linear.plan(*p, rows, epilogue, gate),
                         tiles = linear.plan(*p, tableTiles(rows) * 8, epilogue, gate);
        if (plan.configuration() != tiles.configuration() || plan.storageRows() != tiles.storageRows() ||
            plan.storageRows() < rows)
          broke("a plan on the decode tiles departed from its table tiles' plan", family, cores, plan.workload());
      }
  }
}

// A rotated projection's scratch holds the bf16 rotated rows of the decode
// tiles' most (64 rows) in decode and of the token budget (2048) in prefill;
// an unrotated one's holds none.
void scratchBoundsRotated() {
  const Linear linear = gpu(10, 16);
  for (const bool rotated : {false, true}) {
    const ProjectionShape shape{5120, 17408, rotated};
    require(linear.scratchSize(shape, kMaximumDecodeTileRows).rotated == (rotated ? uint64_t{17408} * 64 * 2 : 0) &&
                linear.scratchSize(shape, SPLASH_PREFILL_TOKEN_BUDGET).rotated == (rotated ? uint64_t{17408} * 2048 * 2 : 0),
            "rotated scratch bounds");
  }
}

// The GGUF decode split rules are per-core laws, checked at every core count
// (zero is the fallback), every family and a grid of widths, inputs, epilogues
// and segment counts, in Q4_K and IQ4_XS (four and a half and four and a
// quarter bits per weight, both four-bit formats), a gate/up pair beside a
// gate of its format, rather than at the measured machines:
// - every batch width runs one tile's plan: Apple10's and Apple11's matrix
//   tile over its 16-row fragments; Apple9's register tile, or its staged tile
//   over its tile's rows; the matrix and register tiles and Apple9's staged
//   tile take one split at every width, so a request's sums there do not
//   depend on the requests batched with it;
// - a split count is a power of two up to the maximum whose partitions keep
//   the kernel's floor (register: one 256-input unit, staged and matrix: 512
//   inputs in whole 32-input groups);
// - the staged tile never spreads its walks on Apple9, the register and
//   matrix tiles never, and no gate/up pair runs in one pass;
// - it depends on the grid per core only: doubling the width and the core
//   count (the cores' bytes a second with them) keeps it; more cores never
//   lower it and a wider grid never raises it;
// - the arena bound (of the shape without formats) covers every segment
//   count, format and pass mode.
void ggufCoreLaws() {
  constexpr std::array<uint32_t, 16> widths{256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120,
                                            6144, 8192, 12288, 16384, 24576, 65536, 248320};
  constexpr std::array<uint32_t, 11> inputs{256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 12288, 17408};
  for (const uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores = 0; cores <= 128; ++cores) {
      const Linear linear = gpu(family, cores), more = gpu(family, cores + 1), twice = gpu(family, 2 * cores);
      const uint32_t planned = simulatedDevice(family, cores).plannedGpuCores();
      const bool spreads = family >= 10 && planned >= 16,
                 twiceSpreads = family >= 10 && simulatedDevice(family, 2 * cores).plannedGpuCores() >= 16;
      for (const uint32_t format : {uint32_t{GGUF_FMT_Q4K}, uint32_t{GGUF_FMT_IQ4XS}})
        for (const uint32_t n : widths)
          for (const uint32_t k : inputs)
            for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
              for (uint32_t segments = 1; segments <= (epilogue == LinearEpilogue::None ? 3U : 1U); ++segments) {
                const bool gateUp = epilogue == LinearEpilogue::GateUp;
                const Projection p = blockProjection(n, k, segments, format), gate = blockProjection(n, k, 1, format);
                const auto plan = [&](const Linear &l, uint32_t width, uint32_t rows) {
                  const Projection up = blockProjection(width, k, segments, format),
                                   g = blockProjection(width, k, 1, format);
                  return l.plan({{width, k}, rows, epilogue}, up, gateUp ? &g : nullptr);
                };
                // The grid of a lane's column tiles: a one-format pair's of
                // both tensors where it may run in one pass.
                const uint32_t columns = gateUp && spreads ? 2 * n : n;
                const auto registers = [&](uint32_t lanes, uint32_t width) {
                  return family == 9 || 2 * uint64_t{width / 64} * lanes < uint64_t{3} * planned;
                };
                // One lane may run the 16-row tile unsplit where it stages on
                // Apple10 or Apple11: a pair where its tensors' own plans do, a
                // projection where its bits decide it (IQ4_XS always, Q4_K
                // never on Apple11), else as Apple10's wave law says.
                const LinearWorkload own{{n, k}, 8, LinearEpilogue::None};
                const std::optional<bool> sixteen =
                    family < 10 || registers(1, columns) ? std::optional(false)
                    : gateUp ? std::optional(linear.plan(own, gate).configuration().sixteenRows)
                             : sixteenRowsByBits(p.shape(), family);
                // Apple10 and Apple11 run the matrix tile at every batch width, over
                // its fragments' rows, at one configuration: its law reads the
                // grid and the rate, not the rows. Its partitions keep 512 inputs in
                // whole groups, a gate/up pair runs two passes, and its split
                // depends on the grid per core alone.
                if (family >= 10) {
                  std::optional<LinearConfig> first;
                  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
                    const uint32_t rows = lanes * 8;
                    const LinearPlan other = plan(linear, n, rows);
                    const LinearConfig c = other.configuration();
                    const uint32_t s = c.splits;
                    require(c.tile == LinearTile::GgufMatrix && (!first || c == *first) &&
                                other.input() == LinearInput::Plain && other.storageRows() == matrixStorageRows(rows) &&
                                !c.sixteenRows && !c.spread && !c.oneGateUpPass,
                            "GGUF matrix plan depends on the batch width");
                    require(s >= 1 && s <= 8 && (s & (s - 1)) == 0 && (s == 1 || (k / s >= 512 && (k / 32) % s == 0)),
                            "GGUF matrix partition below the kernel floor");
                    if (!first) first = c;
                  }
                  if (cores)
                    for (const uint32_t rows : {8U, 32U}) {
                      const LinearConfig at = plan(linear, n, rows).configuration(),
                                         doubled = plan(twice, 2 * n, rows).configuration(),
                                         withMore = plan(more, n, rows).configuration(),
                                         wider = plan(linear, 2 * n, rows).configuration();
                      require(doubled == at, "GGUF matrix plan depends on more than the grid per core");
                      require(withMore.splits >= at.splits, "GGUF matrix split count falls with more cores");
                      require(wider.splits <= at.splits, "GGUF matrix split count rises with the width");
                    }
                  const LinearScratchSize bound = linear.scratchSize({n, k}, kMaximumDecodeTileRows);
                  for (const uint32_t rows : {8U, 16U, 24U, 32U}) {
                    const LinearScratchSize need =
                        linear.plan({{n, k}, rows, epilogue}, p, gateUp ? &gate : nullptr).scratchSize();
                    require(bound.input >= need.input && bound.sums >= need.sums && bound.partials >= need.partials &&
                                bound.counters >= need.counters,
                            "GGUF decode arena bound below a matrix plan");
                  }
                  continue;
                }
                // Every batch width takes its tile and binds its tile's rows; the
                // tiers' tiles take one configuration at every width.
                std::optional<LinearConfig> registerConfig, stagedConfig, oneLane;
                for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
                  const uint32_t rows = lanes * 8;
                  const LinearPlan other = plan(linear, n, rows);
                  const LinearConfig c = other.configuration();
                  const bool staged = !registers(lanes, columns), tiers = !staged || family == 9;
                  std::optional<LinearConfig> &first = staged ? stagedConfig : registerConfig;
                  require(c.tile == (staged ? LinearTile::GgufStaged : LinearTile::GgufRegister) &&
                              (!first || (tiers ? c == *first : c.splits <= first->splits)) &&
                              other.input() == (staged ? LinearInput::Plain : LinearInput::Table16) &&
                              (lanes == 1 && c.splits == 1 ? !sixteen.has_value() || c.sixteenRows == *sixteen
                                                           : !c.sixteenRows) &&
                              other.storageRows() == (c.sixteenRows ? 16U : staged && rows == 24 ? 32U : rows),
                          "GGUF decode plan depends on the batch width");
                  if (!first || !tiers) first = c;
                  if (lanes == 1) oneLane = c;
                  // The unsplit 16-row lane runs the two-lane plan.
                  if (lanes == 2 && oneLane->sixteenRows) {
                    LinearConfig padded = *oneLane;
                    padded.sixteenRows = false;
                    require(padded == c, "GGUF one-lane 16-row plan differs from the two-lane plan");
                  }
                  const uint32_t s = c.splits;
                  require(s >= 1 && s <= LinearConfig::kMaximumSplits && (s & (s - 1)) == 0 &&
                              c.spread == (staged && planned >= 16) && (!c.oneGateUpPass || (gateUp && spreads)),
                          "GGUF decode plan split count, walk or passes");
                  require(s == 1 || (staged ? k / s >= 512 && (k / 32) % s == 0 : k / 256 / s >= 1),
                          "GGUF decode partition below the kernel floor");
                }
                if (cores) {
                  for (const uint32_t rows : {8U, 32U}) {
                    const LinearConfig at = plan(linear, n, rows).configuration();
                    // The tile, its split and passes; the staged walk follows
                    // the core count (kApple10SpreadFromCores), and with it whether
                    // a pair may run in one pass.
                    const LinearConfig doubled = plan(twice, 2 * n, rows).configuration();
                    require((gateUp && spreads != twiceSpreads) ||
                                (doubled.tile == at.tile && doubled.splits == at.splits &&
                                 doubled.oneGateUpPass == at.oneGateUpPass && doubled.sixteenRows == at.sixteenRows),
                            "GGUF decode plan depends on more than the grid per core");
                    if (at.tile == LinearTile::GgufStaged && family >= 10) continue;
                    const LinearConfig withMore = plan(more, n, rows).configuration(),
                                       wider = plan(linear, 2 * n, rows).configuration();
                    require(withMore.tile != at.tile || withMore.splits >= at.splits,
                            "GGUF split count falls with more cores");
                    require(wider.tile != at.tile || wider.splits <= at.splits,
                            "GGUF split count rises with the width");
                  }
                }
                const LinearScratchSize bound = linear.scratchSize({n, k}, kMaximumDecodeTileRows);
                for (const uint32_t rows : {8U, 16U, 24U, 32U}) {
                  const LinearScratchSize need =
                      linear.plan({{n, k}, rows, epilogue}, p, gateUp ? &gate : nullptr)
                          .scratchSize();
                  require(bound.input >= need.input && bound.sums >= need.sums &&
                              bound.partials >= need.partials && bound.counters >= need.counters,
                          "GGUF decode arena bound below a plan");
                }
              }
    }
  // The float tile follows the grid per core and a step's work: once a chunk
  // takes the neural accelerator, longer chunks, fewer cores and rows of less
  // work a step (BF16 below F32 below a quantized format's) keep it; never
  // below 16 rows or on Apple9.
  constexpr std::array<uint32_t, 3> kByWork{GGUF_FMT_BF16, QuantizedSegment::kFloat32, GGUF_FMT_AF8G64};
  for (uint32_t cores = 1; cores <= 128; ++cores)
    for (const uint32_t n : {16U, 64U, 256U, 1024U})
      for (size_t w = 0; w < kByWork.size(); ++w) {
        const uint32_t format = kByWork[w];
        for (const uint32_t family : {10U, 11U}) {
          const Linear linear = gpu(family, cores), more = gpu(family, cores + 1);
          bool accelerator = false;
          for (uint32_t rows = 1; rows <= 2048; ++rows) {
            const bool now = linear.ggufFloatTile(rows, n, format) == FloatTile::NeuralAccelerator;
            require((!accelerator || now) && (!now || rows >= 16) &&
                        (more.ggufFloatTile(rows, n, format) == FloatTile::Simdgroup || now) &&
                        (!now || !w || linear.ggufFloatTile(rows, n, kByWork[w - 1]) == FloatTile::NeuralAccelerator),
                    "GGUF float tile is not monotone in rows, cores and work");
            accelerator = now;
          }
        }
        require(gpu(9, cores).ggufFloatTile(2048, n, format) == FloatTile::Simdgroup, "Apple9 GGUF float tile");
      }
}

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  if (!bytes) return {};
  auto buffer = test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// An n x k segment in `format` at column `offset`, whose planes hold whole
// tiles of its rows, unwritten: encoding reads only their sizes.
QuantizedSegment segmentPlanes(metal::MetalBackend &backend, uint32_t format, uint32_t n, uint32_t k,
                               uint32_t offset = 0) {
  const QuantFormat &f = kQuantFormats[format];
  const uint64_t units = uint64_t{(n + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS} * (k / 32);
  QuantizedSegment s =
      QuantizedSegment::planes(format, n, k, allocate(backend, units * f.plane0_bytes),
                               allocate(backend, units * f.plane1_bytes),
                               allocate(backend, units / f.meta_groups * f.meta_bytes));
  s.columnOffset = offset;
  return s;
}

// A block projection dispatches only as the plan's matrix, and each segment
// fills whole 64-column tiles, with every buffer the plan needs: the matching
// projection encodes its dispatch, so only the projection can reject.
void ggufProjectionMatrix(metal::MetalBackend &backend) {
  const Linear linear = gpu(10, 16);
  const auto segment = [&](uint32_t n, uint32_t offset) {
    return segmentPlanes(backend, GGUF_FMT_Q4K, n, 17408, offset);
  };
  const Projection matching(5120, 17408, BlockWeights{{segment(5120, 0)}});
  const LinearPlan plan = linear.plan({{5120, 17408}, 8}, matching);
  const uint64_t rows = plan.storageRows();
  const LinearScratchSize scratch = plan.scratchSize();
  const LinearBuffers buffers{
      .input = allocate(backend, rows * 17408 * 2),
      .output = allocate(backend, rows * 5120 * 2),
      .gateScratch = allocate(backend, plan.gateScratchBytes()),
      .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                  allocate(backend, scratch.partials), allocate(backend, scratch.counters)}};
  {
    metal::CommandGraph graph;
    (void)linear.add(graph, buffers, matching, plan);
    require(graph.dispatches().size() == 1, "the matching block projection did not encode its dispatch");
  }
  // The plan is the matrix tile's, which a family whose calibration plans none refuses when handed it, with nothing
  // encoded (matrixTileFamilies: its own selection never makes one).
  {
    const Linear apple9 = gpu(9, 10);
    require(plan.configuration().tile == LinearTile::GgufMatrix && !apple9.calibration().projection.matrixSplits,
            "the refusal's plan is not a matrix plan handed to a family without the law");
    metal::CommandGraph graph;
    rejects([&] { (void)apple9.add(graph, buffers, matching, plan); },
            "the matrix block tile is not this GPU family's",
            "a family whose calibration plans no matrix tile ran one handed to it");
    require(graph.empty(), "a refused matrix plan encoded a dispatch");
  }
  // The padding past a projection's segments is part of it, not room for a
  // narrower plan; a segment of 32 columns leaves a partial tile.
  for (const auto &[projection, refusal] :
       {std::pair{Projection(5376, 17408, BlockWeights{{segment(5120, 0)}}), "block projection does not match plan"},
        std::pair{Projection(5120, 17408, BlockWeights{{segment(5088, 0), segment(32, 5088)}}),
                  "block segments do not fill whole column tiles"}}) {
    metal::CommandGraph graph;
    rejects([&] { (void)linear.add(graph, buffers, projection, plan); }, refusal,
            "a block projection that is not the plan's tiles was accepted");
    require(graph.empty(), "a block projection that is not the plan's tiles encoded a dispatch");
  }
  // A projection with an fp32 destination writes fp32 rows through the
  // kernel's fp32 instance (8 rows: the matrix tile's, which took the staged
  // tile's place); the fused kernels have none, so one of several
  // tensors runs a dispatch of each tensor's, into its own columns.
  Projection head = matching, fused(5120, 17408, BlockWeights{{segment(2560, 0), segment(2560, 2560)}});
  head.destination = fused.destination = FloatOutput::Float32;
  const LinearPlan fp32 = linear.plan({{5120, 17408}, 8}, head);
  LinearBuffers fp32Buffers = buffers;
  fp32Buffers.output = allocate(backend, rows * 5120 * sizeof(float));
  metal::CommandGraph graph;
  rejects([&] { (void)linear.add(graph, buffers, head, fp32); }, "projection output buffer holds",
          "fp32 rows were written into a bf16 output");
  require(graph.empty(), "an invalid fp32 block projection encoded a dispatch");
  (void)linear.add(graph, fp32Buffers, head, fp32);
  require(graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == "gguf_matrix_a_f32_m16",
          "the fp32 block projection did not encode its fp32 kernel (" +
              (graph.dispatches().empty() ? std::string("none") : std::string(graph.dispatches()[0].pipelineName)) + ")");
  for (const LinearConfig config : {LinearConfig{.tile = LinearTile::GgufStaged}, {.tile = LinearTile::GgufRegister}}) {
    const LinearPlan plan = Linear::plan({{5120, 17408}, 8}, config, FloatOutput::Float32);
    const LinearScratchSize size = plan.scratchSize();
    LinearBuffers tensors = fp32Buffers;
    tensors.scratch = {allocate(backend, size.input), allocate(backend, size.sums), allocate(backend, size.partials),
                       allocate(backend, size.counters)};
    metal::CommandGraph each;
    (void)linear.add(each, tensors, fused, plan);
    const std::string kernel = config.tile == LinearTile::GgufStaged ? "gguf_decode_q4k_m8_a_f32"
                                                                     : "gguf_decode_sg_q4k_l1_a_f32";
    std::vector<uint32_t> offsets;
    for (const metal::ComputeDispatch &dispatch : each.dispatches())
      if (dispatch.pipelineName == kernel)
        offsets.push_back(static_cast<const GgufDecodeParams *>(dispatch.bytes.front().data)->out_offset);
    require(offsets == std::vector<uint32_t>{0, 2560},
            "an fp32 projection of two tensors did not run " + kernel + " into each one's columns");
  }
}

// A producer writes the table its consumer's plan reads into scratch that
// holds it: a table layout without that scratch is refused before anything
// is encoded, and Plain runs the plain kernel. Every other buffer holds what
// its kernel reaches.
void producerTableContract(metal::MetalBackend &backend) {
  constexpr uint32_t kLanes = 1, kRows = 8, kHidden = 5120;
  const NormWeights norm{allocate(backend, kHidden * 2)};
  const auto plainKernel = [](const metal::CommandGraph &graph, std::string_view kernel) {
    return graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == kernel;
  };
  {
    const metal::MetalBuffer rows = allocate(backend, kRows * kHidden * 2);
    metal::CommandGraph graph;
    rejects([&] { (void)Normalization::addRms(graph, rows, norm, rows, kHidden, kRows, {}, LinearInput::Table16); },
            "linear table buffer holds", "the norm wrote a table without scratch");
    require(graph.empty() &&
                Normalization::addRms(graph, rows, norm, rows, kHidden, kRows).layout == LinearInput::Plain &&
                plainKernel(graph, "norm_rms"),
            "the norm's table contract");
  }
  {
    const GdnShape shape{16, 32, 128, 8192, 12544};
    const uint64_t carried = 3 * uint64_t{shape.convolutionDimension} * 2,
                   recurrent = uint64_t{shape.valueHeads} * shape.headDimension * shape.headDimension * 4;
    const GdnStateStrides strides{carried, recurrent, carried, GdnStateFormat::Float32};
    const metal::MetalBuffer state = allocate(backend, carried + recurrent);
    const std::array<metal::MetalBuffer, SPLASH_MAXIMUM_BATCH_WIDTH> states{state, state, state, state, state, state, state, state};
    GdnDecodeBuffers buffers;
    buffers.packed = allocate(backend, kRows * shape.packedWidth * 2);
    buffers.convolutionWeights = allocate(backend, shape.convolutionDimension * 4 * 2);
    buffers.currentStates = states;
    buffers.nextStates = states;
    const std::array<GdnTapeLane, kLanes> tapes{GdnTapeLane{0, gdnTapeLayerBytes(shape), 0}};
    buffers.tape = allocate(backend, 2 * gdnTapeLayerBytes(shape));
    buffers.tapes = tapes;
    buffers.laneRows = allocate(backend, kLanes * sizeof(SplashLaneRows));
    buffers.rows = buffers.laneCap = kRows;
    buffers.decayWeights = allocate(backend, shape.valueHeads * 4);
    buffers.timeBias = allocate(backend, shape.valueHeads * 2);
    buffers.mixerNorm = {allocate(backend, shape.headDimension * 2)};
    buffers.hidden = allocate(backend, kRows * shape.valueHeads * shape.headDimension * 2);
    buffers.tree = allocate(backend, kRows * sizeof(SplashTreeRow));
    metal::CommandGraph graph;
    rejects(
        [&] {
          (void)GDN::addDecode(graph, buffers, shape, 0, strides, GdnHeadOrder::Grouped, LinearInput::Table16);
        },
        "linear table buffer holds", "the GDN decode wrote a table without scratch");
    require(graph.empty() &&
                GDN::addDecode(graph, buffers, shape, 0, strides, GdnHeadOrder::Grouped, LinearInput::Plain)
                        .layout == LinearInput::Plain &&
                plainKernel(graph, "verify_gdn_fused_vh32"),
            "the GDN decode's table contract");
  }
  {
    // The 27B's attention: 24 query heads over 4 KV heads of 256, a lane's
    // 8 rows staged in 32 per KV head; its reduce writes the out-projection's
    // rows.
    const kv::Layout layout{1, 4, 256};
    const metal::MetalBuffer packed = allocate(backend, kRows * (24 * 2 + 4 * 2) * 256 * 2);
    const metal::MetalBuffer queries = allocate(backend, 4 * 32 * 6 * 256 * 2);
    const metal::MetalBuffer hidden = allocate(backend, kRows * 24 * 256 * 2);
    const std::array lane{PagedAttention::chunkParams(0, kRows, 32, 1)};
    const metal::MetalBuffer laneRows = allocate(backend, sizeof(SplashLaneRows));
    *static_cast<SplashLaneRows *>(laneRows.contents()) = {0, kRows};
    const AttentionRows rows{laneRows, kRows};
    const AttentionPlan plan = PagedAttention::plan(lane, 24, layout, kApple10Attention, Int8Activations::Off,
                                                    PagedAttention::workspace(kv::kAttentionMaximumSplits, 24, layout));
    const metal::MetalBuffer partials = allocate(backend, plan.workspace().partialsBytes),
                             statistics = allocate(backend, plan.workspace().statisticsBytes);
    const std::array histories{AttentionHistory{{1, 0}, allocate(backend, sizeof(SplashKvPage))}};
    const auto attend = [&](metal::CommandGraph &graph, LinearInput input) {
      return PagedAttention::addAttention(graph, plan, queries, partials, statistics, lane, histories, rows,
                                          {packed, hidden, AttentionOutput::Every, {}, input});
    };
    metal::CommandGraph graph;
    rejects([&] { (void)attend(graph, LinearInput::Table16); }, "linear table buffer holds",
            "the attention reduce wrote a table without scratch");
    require(graph.empty() && attend(graph, LinearInput::Plain).layout == LinearInput::Plain &&
                graph.dispatches().size() == 2 && graph.dispatches()[1].pipelineName == "attention_reduce",
            "the attention reduce's table contract");
  }
}

// A rotated projection prepares its register tile's table from the rotated
// rows (LinearGguf.cpp), so its plan asks its producer for plain rows: on
// Apple9 a rotated PQ2_0 projection keeps the register tile and its input
// norm runs the plain kernel, where an unrotated one reads Table16.
void rotatedRegisterInput(metal::MetalBackend &backend) {
  const Linear m3 = gpu(9, 40);
  Projection rotated = blockProjection(5120, 17408, 1, GGUF_FMT_PQ20);
  const Projection plain = rotated;
  rotated.rotation.signs = allocate(backend, 17408);
  const LinearScratch scratch{.input = allocate(backend, tableBytes(17408, 32)),
                              .sums = allocate(backend, tableSumsBytes(LinearInput::Table16, 17408, 32))};
  const NormWeights norm{allocate(backend, 17408 * 2)};
  const metal::MetalBuffer rows = allocate(backend, uint64_t{32} * 17408 * 2);
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const LinearPlan plan = m3.plan(rotated, lanes * SPLASH_DRAFT_QUERY_ROWS);
    require(plan.configuration().tile == LinearTile::GgufRegister && plan.input() == LinearInput::Plain &&
                m3.plan(plain, lanes * SPLASH_DRAFT_QUERY_ROWS).input() == LinearInput::Table16,
            "a rotated register plan's producer writes a table it discards");
    metal::CommandGraph graph;
    (void)Normalization::addRms(graph, rows, norm, rows, 17408, lanes * 8, scratch, plan.input());
    require(graph.dispatches().back().pipelineName.find("_table") == std::string::npos,
            "a rotated register plan's input norm wrote a table");
  }
}

// Int8 activations off keep a rotated PQ2_0 projection's decode plans off the
// int8 tiles and change nothing else of them; on, the plans of 16 rows or more
// take them (gguf-projection checks each device's choice). Bonsai's GDN input,
// 8192 x 5120, on a 16-core Apple10.
void decodeInt8Setting(metal::MetalBackend &backend) {
  constexpr uint32_t n = 8192, k = 5120;
  Projection rotated(n, k, BlockWeights{{segmentPlanes(backend, GGUF_FMT_PQ20, n, k)}});
  rotated.rotation.signs = allocate(backend, k);
  const Linear on = gpu(10, 16), off(simulatedDevice(10, 16), Int8Activations::Off, StepKind::Prefill);
  bool int8 = false;
  for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes) {
    LinearConfig config = on.plan(rotated, lanes * SPLASH_DRAFT_QUERY_ROWS).configuration();
    int8 |= config.decodeInt8;
    config.decodeInt8 = false;
    require(off.plan(rotated, lanes * SPLASH_DRAFT_QUERY_ROWS).configuration() == config,
            "int8 activations off moved a rotated PQ2_0 decode plan other than off the int8 tiles");
  }
  require(int8, "no decode plan of a rotated PQ2_0 projection takes int8 activations with them on");
}

// The start's precision line names a rotated PQ2_0 unit's int8 decode tiles
// by the plans that took them (ExecutionPlans::activationsDescription): none
// before any plan; "in decode" once a decode step's 16-row plan took them; a
// prefill chunk of 33 rows, on the staged tile in bf16, adds nothing; one of
// 16 rows, which takes them, adds the short chunks. Bonsai's GDN input on a
// 16-core Apple10.
void rotatedPrecisionClause(metal::MetalBackend &backend) {
  constexpr uint32_t n = 8192, k = 5120;
  Projection rotated(n, k, BlockWeights{{segmentPlanes(backend, GGUF_FMT_PQ20, n, k)}});
  rotated.rotation.signs = allocate(backend, k);
  const ExecutionPlans plans(simulatedDevice(10, 16), Int8Activations::On);
  const kv::Layout layout{1, 2, 256};
  const auto line = [&] { return plans.activationsDescription(layout, 8, 0, {}, {}); };
  require(line() == "auto, bf16 throughout for this model on this GPU", "a line named a path before a plan: " + line());
  require(plans.decodeLinear().plan(rotated, 16).configuration().decodeInt8 &&
              line() == "auto, int8 in rotated PQ2_0 projections in decode",
          "a decode plan's int8 tiles were not named in decode: " + line());
  require(!plans.linear().plan(rotated, 33).configuration().decodeInt8 &&
              line() == "auto, int8 in rotated PQ2_0 projections in decode",
          "a bf16 prefill chunk of 33 rows changed the line: " + line());
  require(plans.linear().plan(rotated, 16).configuration().decodeInt8 &&
              line() == "auto, int8 in rotated PQ2_0 projections in decode and prefill chunks of up to 32 rows",
          "a short prefill chunk's int8 tiles were not named: " + line());
}

// The int8 tiles quantize a rotated input (LinearGguf.cpp): a plan in int8
// activations over unrotated PQ2_0 weights is refused, not run in bf16.
void decodeInt8TakesRotation(metal::MetalBackend &backend) {
  constexpr uint32_t n = 256, k = 2048;
  const Projection plain(n, k, BlockWeights{{segmentPlanes(backend, GGUF_FMT_PQ20, n, k)}});
  const LinearPlan plan = Linear::plan({{n, k}, 8, LinearEpilogue::None},
                                       {.tile = LinearTile::GgufStaged, .sixteenRows = true, .decodeInt8 = true},
                                       FloatOutput::BFloat16);
  const LinearBuffers buffers{.input = allocate(backend, uint64_t{16} * k * 2),
                              .output = allocate(backend, uint64_t{16} * n * 2)};
  metal::CommandGraph graph;
  rejects([&] { (void)gpu(10, 16).add(graph, buffers, plain, plan); },
          "only a rotated PQ2_0 projection takes int8 activations", "an unrotated projection took int8 activations");
  require(graph.empty(), "a refused int8 plan encoded a dispatch");
}

// Each buffer only a block projection's dispatches reach, at its extent and
// one element short: a rotated projection's int8 sign of every input and its
// rotated rows of the plan's storage; a float projection's fp32 weights, its
// input rows and its output rows of 512 columns up to its last column, 192,
// and those of a second one in its dispatch; and the planes of a quantized
// segment.
void blockExtents(metal::MetalBackend &backend) {
  const Linear linear = gpu(10, 16);
  constexpr uint32_t n = 256, k = 2048;
  Projection rotated(n, k, BlockWeights{{segmentPlanes(backend, GGUF_FMT_PQ20, n, k)}});
  rotated.rotation.signs = allocate(backend, k);
  const LinearPlan plan = linear.plan(rotated, 8);
  const uint64_t rows = plan.storageRows();
  const LinearScratchSize scratch = plan.scratchSize();
  const LinearBuffers buffers{.input = allocate(backend, rows * k * 2),
                              .output = allocate(backend, rows * n * 2),
                              .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                                          allocate(backend, scratch.partials), allocate(backend, scratch.counters),
                                          allocate(backend, rows * k * 2)}};
  requireExtent(backend, rotated.rotation.signs, k, 1, "rotation sign",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  Projection changed = rotated;
                  changed.rotation.signs = view;
                  (void)linear.add(graph, buffers, changed, plan);
                });
  requireExtent(backend, buffers.scratch.rotated, rows * k * 2, 2, "projection rotated input",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  LinearBuffers changed = buffers;
                  changed.scratch.rotated = view;
                  (void)linear.add(graph, changed, rotated, plan);
                });

  constexpr uint32_t floatRows = 37, floatColumns = 64, outStride = 512, outOffset = 128;
  const QuantizedSegment weights =
      QuantizedSegment::floats(floatColumns, k, allocate(backend, uint64_t{floatColumns} * k * 4));
  const metal::MetalBuffer input = allocate(backend, uint64_t{floatRows} * k * 2),
                           output = allocate(backend, uint64_t{floatRows} * outStride * 4);
  const auto project = [&](metal::CommandGraph &graph, const metal::MetalBuffer &rowsIn,
                           const QuantizedSegment &segment, const metal::MetalBuffer &rowsOut) {
    addGgufFloat(graph, rowsIn, segment, rowsOut, floatRows, outStride, outOffset, FloatOutput::Float32,
                 FloatTile::Simdgroup);
  };
  requireExtent(backend, weights.plane0, uint64_t{floatColumns} * k * 4, 4, "float projection weight",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  project(graph, input, QuantizedSegment::floats(floatColumns, k, view), output);
                });
  requireExtent(backend, input, uint64_t{floatRows} * k * 2, 2, "float projection input",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  project(graph, view, weights, output);
                });
  requireExtent(backend, output, (uint64_t{floatRows - 1} * outStride + outOffset + floatColumns) * 4, 4,
                "float projection output", [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  project(graph, input, weights, view);
                });
  // The second projection of a dispatch of two, as the QSA indexer's key beside its query: its weights and its own
  // rows of floatColumns columns.
  const QuantizedSegment second =
      QuantizedSegment::floats(floatColumns, k, allocate(backend, uint64_t{floatColumns} * k * 4));
  const metal::MetalBuffer secondRows = allocate(backend, uint64_t{floatRows} * floatColumns * 4);
  const auto pair = [&](metal::CommandGraph &graph, const QuantizedSegment &segment,
                        const metal::MetalBuffer &rowsOut) {
    const std::array<FloatProjection, 2> projections{FloatProjection{&weights, output, outStride, outOffset},
                                                     FloatProjection{&segment, rowsOut, floatColumns, 0}};
    addGgufFloat(graph, input, projections, floatRows, FloatOutput::Float32);
  };
  requireExtent(backend, second.plane0, uint64_t{floatColumns} * k * 4, 4, "float projection weight",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  pair(graph, QuantizedSegment::floats(floatColumns, k, view), secondRows);
                });
  requireExtent(backend, secondRows, uint64_t{floatRows} * floatColumns * 4, 4, "float projection output",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) { pair(graph, second, view); });

  // A fused projection of a 320-row Q5_K segment and a 192-row Q4_K one.
  // Each plane is tiles of QUANT_TILE_ROWS rows by groups of 32 inputs, or
  // meta units, and ends at the last row's unit of the last group: row 319 of
  // the Q5_K segment and row 191 of the Q4_K one, in the tile that holds it.
  const std::array segments{segmentPlanes(backend, GGUF_FMT_Q5K, 320, k),
                            segmentPlanes(backend, GGUF_FMT_Q4K, 192, k, 320)};
  const Projection fused(512, k, BlockWeights{{segments[0], segments[1]}});
  const LinearPlan fusedPlan = linear.plan(fused, 8);
  const LinearScratchSize fusedScratch = fusedPlan.scratchSize();
  const LinearBuffers fusedBuffers{.input = allocate(backend, fusedPlan.storageRows() * k * 2),
                                   .output = allocate(backend, fusedPlan.storageRows() * 512 * 2),
                                   .scratch = {allocate(backend, fusedScratch.input),
                                               allocate(backend, fusedScratch.sums),
                                               allocate(backend, fusedScratch.partials),
                                               allocate(backend, fusedScratch.counters)}};
  for (size_t index = 0; index < segments.size(); ++index) {
    const QuantFormat &format = kQuantFormats[segments[index].formatId];
    const uint64_t groups = k / 32, units = groups / format.meta_groups;
    const auto lastUnit = [&](uint64_t blocks) {
      const uint64_t last = (index == 0 ? 320 : 192) - 1;
      return (last / QUANT_TILE_ROWS * blocks + blocks - 1) * QUANT_TILE_ROWS + last % QUANT_TILE_ROWS + 1;
    };
    for (const auto &[member, bytes, element, name] :
         std::initializer_list<std::tuple<metal::MetalBuffer QuantizedSegment::*, uint64_t, uint64_t, const char *>>{
             {&QuantizedSegment::plane0, lastUnit(groups) * format.plane0_bytes, format.plane0_bytes,
              "projection plane0"},
             {&QuantizedSegment::plane1, lastUnit(groups) * format.plane1_bytes, format.plane1_bytes,
              "projection plane1"},
             {&QuantizedSegment::meta, lastUnit(units) * format.meta_bytes, format.meta_bytes, "projection meta"}}) {
      if (!bytes) continue;
      requireExtent(backend, segments[index].*member, bytes, element, name,
                    [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                      std::array changed = segments;
                      changed[index].*member = view;
                      (void)linear.add(graph, fusedBuffers, Projection(512, k, BlockWeights{{changed[0], changed[1]}}),
                                       fusedPlan);
                    });
    }
  }
}

// A view of the leading 512 inputs of rows of 1024 (Projection::leadingInputs)
// over 512 outputs, in tiles of QUANT_TILE_ROWS rows: the prefill residual
// tile encodes its leading-input instance with the view's parameters, and each
// plane is read up to the last group the view reads of its last tile, after
// every group of the tiles before. A plane at that extent encodes, one element
// shorter or holding only a matrix of the view's own inputs is refused. Every
// other plan and a rotated view are refused before anything is encoded. The
// views' factories refuse inputs beyond the projection's or of part of a quant
// group or meta unit, rows of part of a tile or past the tensor's, float or
// rotated weights or a view for leading inputs, and a view of a view's inputs
// for rows; a view of rows holds its ranges of each plane, or of float rows, in
// order, with the source's rotation and destination.
void leadingInputViews(metal::MetalBackend &backend) {
  const Linear linear = gpu(10, 16);
  constexpr uint32_t n = 512, k = 512, wide = 1024, rows = 168;
  const auto buffersOf = [&](const LinearPlan &plan) {
    const uint64_t storage = plan.storageRows();
    const LinearScratchSize scratch = plan.scratchSize();
    return LinearBuffers{.input = allocate(backend, storage * k * 2),
                         .output = allocate(backend, storage * n * 2),
                         .residual = allocate(backend, storage * n * 2),
                         .gateScratch = allocate(backend, plan.gateScratchBytes()),
                         .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                                     allocate(backend, scratch.partials), allocate(backend, scratch.counters)}};
  };
  const auto residualPrefill = [&](LinearConfig config, LinearEpilogue epilogue) {
    return Linear::plan({{n, k}, rows, epilogue}, config, FloatOutput::BFloat16);
  };
  // The extent of a plane the view reads, in units of `unitBytes`: every tile
  // but the last, `rowGroups` groups (or meta units) of each of its
  // QUANT_TILE_ROWS rows, then the first `readGroups` of the last tile.
  const auto reach = [](uint64_t rowGroups, uint64_t readGroups, uint64_t unitBytes) {
    return ((n / QUANT_TILE_ROWS - 1) * rowGroups + readGroups) * QUANT_TILE_ROWS * unitBytes;
  };

  const QuantFormat &q5k = kQuantFormats[GGUF_FMT_Q5K];
  const QuantizedSegment wideSegment = segmentPlanes(backend, GGUF_FMT_Q5K, n, wide);
  const auto ggufView = [&](const QuantizedSegment &planes) {
    return Projection(n, wide, BlockWeights{{planes}}).leadingInputs(k);
  };
  const Projection gguf = ggufView(wideSegment);
  const LinearPlan ggufPlan = residualPrefill({.tile = LinearTile::GgufPrefill}, LinearEpilogue::Residual);
  {
    metal::CommandGraph graph;
    (void)linear.add(graph, buffersOf(ggufPlan), gguf, ggufPlan);
    const std::string kernel = leadingInputsInstance("gguf_prefill_q5k_r");
    const GgufPrefillLeadingParams params{{k, rows, n, 0}, wide};
    require(graph.dispatches().size() == 1, "a view of leading inputs did not encode one dispatch");
    const metal::ComputeDispatch &dispatch = graph.dispatches()[0];
    require(dispatch.pipelineName == kernel && dispatch.bytes.size() == 1 &&
                dispatch.bytes[0].sizeBytes == sizeof(params) &&
                std::memcmp(dispatch.bytes[0].data, &params, sizeof(params)) == 0,
            "a view of leading inputs did not encode " + kernel + " with its parameters");
  }
  const LinearBuffers ggufBuffers = buffersOf(ggufPlan);
  const QuantizedSegment narrow = segmentPlanes(backend, GGUF_FMT_Q5K, n, k);
  for (const auto &[member, rowGroups, readGroups, unitBytes, name] :
       std::initializer_list<
           std::tuple<metal::MetalBuffer QuantizedSegment::*, uint64_t, uint64_t, uint64_t, const char *>>{
           {&QuantizedSegment::plane0, wide / 32, k / 32, q5k.plane0_bytes, "projection plane0"},
           {&QuantizedSegment::plane1, wide / 32, k / 32, q5k.plane1_bytes, "projection plane1"},
           {&QuantizedSegment::meta, wide / 32 / q5k.meta_groups, k / 32 / q5k.meta_groups, q5k.meta_bytes,
            "projection meta"}}) {
    const auto add = [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
      QuantizedSegment planes = wideSegment;
      planes.*member = view;
      (void)linear.add(graph, ggufBuffers, ggufView(planes), ggufPlan);
    };
    requireExtent(backend, wideSegment.*member, reach(rowGroups, readGroups, unitBytes), unitBytes, name, add);
    metal::CommandGraph graph;
    rejects([&] { add(graph, narrow.*member); }, std::string(name) + " buffer holds",
            "a view of leading inputs over planes of its own inputs was accepted");
    require(graph.empty(), "a view of leading inputs over planes of its own inputs encoded a dispatch");
  }

  Projection rotated = gguf;
  rotated.rotation.signs = allocate(backend, k);
  const auto refuses = [&](const Projection &view, const LinearPlan &plan) {
    metal::CommandGraph graph;
    rejects([&] { (void)linear.add(graph, buffersOf(plan), view, plan); },
            "a view of leading inputs runs only the quantized prefill residual tiles",
            "a view of leading inputs ran on a plan without its instance");
    require(graph.empty(), "a refused view of leading inputs encoded a dispatch");
  };
  refuses(gguf, linear.plan(gguf, 8, LinearEpilogue::Residual));
  for (const LinearEpilogue epilogue : {LinearEpilogue::None, LinearEpilogue::UpWithGate})
    refuses(gguf, residualPrefill({.tile = LinearTile::GgufPrefill}, epilogue));
  refuses(gguf, Linear::plan({{n, k}, 32, LinearEpilogue::Residual},
                             {.tile = LinearTile::GgufStaged}, FloatOutput::BFloat16));
  refuses(rotated, ggufPlan);

  // The views' factories: inputs beyond the projection's or of part of a quant group or meta unit, and a projection
  // of float or rotated weights or of a view, are refused.
  const Projection wideGguf(n, wide, BlockWeights{{wideSegment}});
  constexpr std::string_view kInputs = "a view of leading inputs takes whole quant groups and meta units";
  constexpr std::string_view kSource = "a view of leading inputs takes one unrotated quantized tensor";
  rejects([&] { (void)wideGguf.leadingInputs(wide + 256); }, kInputs, "a view of more inputs than rows hold");
  rejects([&] { (void)wideGguf.leadingInputs(k + 16); }, kInputs, "a view of part of a quant group");
  rejects([&] { (void)wideGguf.leadingInputs(k + 64); }, kInputs, "a view of part of a Q5_K meta unit");
  const Projection floats(n, wide, BlockWeights{{QuantizedSegment::floats(n, wide, allocate(backend, n * wide * 4))}});
  Projection rotatedSource = wideGguf;
  rotatedSource.rotation.signs = allocate(backend, wide);
  rejects([&] { (void)floats.leadingInputs(k); }, kSource, "a view of float weights");
  rejects([&] { (void)rotatedSource.leadingInputs(k); }, kSource, "a view of rotated weights");
  rejects([&] { (void)gguf.leadingInputs(k / 2); }, kSource, "a view of a view");
  rejects([&] { (void)gguf.rows(backend, {{0, 256}}); }, "a view of rows takes a projection of one tensor",
          "a view of the rows of a view of leading inputs");

  // A view of rows, whole plane tiles, takes each plane's tiles of its ranges in order (the leading 256 rows and the
  // last 128, with an empty range between), and the source's rotation and fp32 destination.
  Projection source = rotatedSource;
  source.destination = FloatOutput::Float32;
  const Projection ggufRows = source.rows(backend, {{0, 256}, {256, 0}, {n - 128, 128}});
  require(ggufRows.outputSize == 384 && ggufRows.inputSize == wide && !ggufRows.planeInputs() &&
              ggufRows.rotation.signs.sameView(source.rotation.signs) &&
              ggufRows.destination == FloatOutput::Float32 && ggufRows.blocks().segments.size() == 2,
          "a view of two ranges of Q5_K rows is not a projection of their 384 rows");
  const auto bytes = [&](uint32_t rows) {
    return std::array<uint64_t, 3>{uint64_t{rows} * (wide / 32) * q5k.plane0_bytes,
                                   uint64_t{rows} * (wide / 32) * q5k.plane1_bytes,
                                   uint64_t{rows} * (wide / 32 / q5k.meta_groups) * q5k.meta_bytes};
  };
  for (const auto &[index, first, count] : {std::tuple{0U, 0U, 256U}, std::tuple{1U, n - 128, 128U}}) {
    const QuantizedSegment &segment = ggufRows.blocks().segments[index];
    const std::array<uint64_t, 3> offset = bytes(first), size = bytes(count);
    require(segment.outputSize == count && segment.inputSize == wide && segment.columnOffset == index * 256 &&
                segment.plane0.sameView(backend.view(wideSegment.plane0, offset[0], size[0])) &&
                segment.plane1.sameView(backend.view(wideSegment.plane1, offset[1], size[1])) &&
                segment.meta.sameView(backend.view(wideSegment.meta, offset[2], size[2])),
            "a view of " + std::to_string(count) + " Q5_K rows from " + std::to_string(first) +
                " does not hold their tiles");
  }
  // Float rows lie in row order.
  const Projection floatRows = floats.rows(backend, {{n - 64, 64}});
  require(floatRows.blocks().segments.front().isFloat() &&
              floatRows.blocks().segments.front().plane0.sameView(
                  backend.view(floats.blocks().segments.front().plane0, uint64_t{n - 64} * wide * 4, 64 * wide * 4)),
          "a view of 64 float rows does not hold them");
  for (const Projection::RowRange range :
       {Projection::RowRange{0, QUANT_TILE_ROWS / 2}, {QUANT_TILE_ROWS / 2, QUANT_TILE_ROWS}, {0, n + QUANT_TILE_ROWS},
        {n - QUANT_TILE_ROWS, 2 * QUANT_TILE_ROWS}})
    rejects([&] { (void)wideGguf.rows(backend, {range}); }, "whole QUANT_TILE_ROWS tiles of the tensor's rows",
            "a view of " + std::to_string(range.count) + " rows from " + std::to_string(range.first));
  rejects([&] { (void)wideGguf.rows(backend, {{0, 0}}); }, "block projection has no segments", "a view of no rows");
}

// Each scratch field of a GGUF register plan at its extent and one element
// short: the Apple9 decode tile that reads an activation table, on a plan of
// a 40-core Apple9 GPU that splits K. requireExtent only encodes, so this
// runs on any device.
void registerTileExtents(metal::MetalBackend &backend) {
  const Linear m3 = gpu(9, 40);
  constexpr uint32_t n = 512, k = 2048, lanes = 2;
  constexpr uint64_t rows = lanes * 8;
  const Projection block(n, k, BlockWeights{{segmentPlanes(backend, GGUF_FMT_Q4K, n, k)}});
  const LinearPlan plan = m3.plan(block, lanes * SPLASH_DRAFT_QUERY_ROWS);
  const LinearScratchSize scratch = plan.scratchSize();
  require(plan.configuration().tile == LinearTile::GgufRegister && scratch.input && scratch.sums &&
              scratch.partials && scratch.counters,
          "the GGUF Apple9 plan is not a register plan that splits K");
  const LinearBuffers buffers{.input = allocate(backend, rows * k * 2),
                              .output = allocate(backend, rows * n * 2),
                              .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                                          allocate(backend, scratch.partials), allocate(backend, scratch.counters)}};
  for (const auto &[member, bytes, element, name] :
       std::initializer_list<std::tuple<metal::MetalBuffer LinearScratch::*, uint64_t, uint64_t, const char *>>{
           {&LinearScratch::input, scratch.input, 2, "projection scratch table"},
           {&LinearScratch::sums, scratch.sums, 4, "projection scratch sums"},
           {&LinearScratch::partials, scratch.partials, 4, "projection partials"},
           {&LinearScratch::counters, scratch.counters, 4, "projection counters"}})
    requireExtent(backend, buffers.scratch.*member, bytes, element, name,
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    LinearBuffers changed = buffers;
                    changed.scratch.*member = view;
                    (void)m3.add(graph, changed, block, plan);
                  });
}

// The register tile splits K into whole coefficient units of each segment's format (quant_register_unit), where a
// plan, which knows no format, bounds the splits by the fewest inputs a unit holds, 64: 512 Q4_K inputs, two units,
// take two splits but not four, 512 Q8_0 inputs four, 192 two, and Q2_K's 256 inputs, two 128-input units, two but
// not four. A refused projection encodes nothing.
void registerSplitUnits(metal::MetalBackend &backend) {
  const Linear m3 = gpu(9, 40);
  constexpr uint32_t n = 512, rows = 8;
  for (const auto [format, k, splits, takes] :
       {std::tuple{GGUF_FMT_Q4K, 512U, 2U, true}, {GGUF_FMT_Q4K, 512U, 4U, false}, {GGUF_FMT_Q80, 512U, 4U, true},
        {GGUF_FMT_Q80, 192U, 2U, true}, {GGUF_FMT_Q2K, 256U, 2U, true}, {GGUF_FMT_Q2K, 256U, 4U, false}}) {
    const Projection block(n, k, BlockWeights{{segmentPlanes(backend, format, n, k)}});
    const LinearPlan plan = Linear::plan({{n, k}, rows, kNone},
                                         {.tile = LinearTile::GgufRegister, .splits = splits}, FloatOutput::BFloat16);
    const LinearScratchSize scratch = plan.scratchSize();
    const LinearBuffers buffers{.input = allocate(backend, uint64_t{rows} * k * 2),
                                .output = allocate(backend, uint64_t{rows} * n * 2),
                                .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                                            allocate(backend, scratch.partials), allocate(backend, scratch.counters)}};
    const std::string what = std::string(kQuantFormats[format].name) + " weights of " + std::to_string(k) +
                             " inputs in " + std::to_string(splits) + " splits";
    metal::CommandGraph graph;
    if (takes) {
      (void)m3.add(graph, buffers, block, plan);
      require(!graph.empty(), "the register tile encoded nothing for " + what);
    } else {
      rejects([&] { (void)m3.add(graph, buffers, block, plan); }, "the register tile takes a coefficient unit per split",
              "the register tile took " + what);
      require(graph.empty(), "the register tile encoded refused " + what);
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: linear-plan <production.metallib|--cpu>");
    ggufPlans();
    matrixTileFamilies();
    scratchBoundsRotated();
    columnPlans();
    decodeTileRows();
    ggufCoreLaws();
    floatOutputPlans();
    apple9StagesPastTheRegisterTile();
    decodeStepsRunRowGroups();
    chunkRowPlans();
    sideBySidePlans();
    prefillInt8Rule();
    if (std::string_view(argv[1]) == "--cpu") {
      std::cout << "Linear CPU plans: PASS\n";
      return 0;
    }
    metal::MetalBackend backend(argv[1]);
    ggufProjectionMatrix(backend);
    producerTableContract(backend);
    rotatedRegisterInput(backend);
    decodeInt8Setting(backend);
    rotatedPrecisionClause(backend);
    decodeInt8TakesRotation(backend);
    blockExtents(backend);
    leadingInputViews(backend);
    registerTileExtents(backend);
    registerSplitUnits(backend);
    std::cout << "Linear plans: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "Linear plans: FAIL: " << error.what() << '\n';
    return 1;
  }
}

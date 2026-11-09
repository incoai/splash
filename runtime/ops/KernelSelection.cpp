#include "ops/KernelSelection.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <array>
#include <variant>

namespace splash::ops {
namespace {

static_assert(kApple9Projection.stagedPrefillRows <= kMaximumStagedPrefillRows &&
                  kApple9SmallProjection.stagedPrefillRows <= kMaximumStagedPrefillRows &&
                  kApple10Projection.stagedPrefillRows <= kMaximumStagedPrefillRows &&
                  kApple11Projection.stagedPrefillRows <= kMaximumStagedPrefillRows,
              "the staged tile holds the prefill chunks it runs");

// The register tile runs a unit of any class only at a lane count and grid it
// runs a unit of FourBitsOrMore at, so that class's boundary bounds every
// unit's register plan (decodeConfigurations). A gate/up pair of a grid lookup
// (or Q2_K) tensor and a four-bit one is UnderFourBits, which registers
// wherever both its tensors' classes do, so their storage bounds the pair's
// (decodeStorageRows).
constexpr bool registerBoundsHold(const RegisterBoundary &boundary) noexcept {
  const auto &rows = boundary.belowQuarters;
  const auto &fourBits = rows[size_t(FormatClass::FourBitsOrMore)];
  const auto &underFourBits = rows[size_t(FormatClass::UnderFourBits)];
  for (size_t lane = 0; lane < fourBits.size(); ++lane) {
    for (const auto &row : rows)
      if (row[lane] > fourBits[lane]) return false;
    for (const FormatClass grid : {FormatClass::GridLookup, FormatClass::GridLookupOrQ2K})
      if (underFourBits[lane] < std::min(rows[size_t(grid)][lane], fourBits[lane])) return false;
  }
  return true;
}
static_assert(registerBoundsHold(kApple9Projection.decodeRegister) &&
                  registerBoundsHold(kApple9SmallProjection.decodeRegister) &&
                  registerBoundsHold(kApple10Projection.decodeRegister) &&
                  registerBoundsHold(kApple11Projection.decodeRegister),
              "a class registers a grid FourBitsOrMore stages, or a mixed pair stages where its tensors register");

// The most bits per weight a format stores, in sixteenths (BF16 and F16).
constexpr uint32_t kMostWeightBits16 = [] {
  uint32_t most = 0;
  for (const QuantFormat &f : kQuantFormats) most = std::max(most, f.block_bytes * 128 / f.block_elements);
  return most;
}();

// The split count doubles while some tier asks (TierLaw).
uint32_t tierSplits(uint32_t n, uint32_t k, uint32_t cores, std::span<const SplitTier> tiers) {
  const uint64_t grid = n / GGUF_TILE_COLUMNS;
  uint32_t splits = 1;
  const auto asks = [&](const SplitTier &t) {
    return splits < t.splits && grid * splits < uint64_t{t.threadgroupsPerCore} * cores &&
           k / (2 * splits) >= t.inputs;
  };
  while ((k / 32) % (2 * splits) == 0 && std::any_of(tiers.begin(), tiers.end(), asks)) splits *= 2;
  return splits;
}

// The waves of a dispatch of `groups` threadgroups of a staged tile on
// `cores` cores, `resident` on each core at once.
uint64_t waves(uint64_t groups, uint32_t cores, uint32_t resident) {
  return ((groups + cores - 1) / cores + resident - 1) / resident;
}

// The time of a dispatch of `groups` threadgroups of a staged tile, `steps`
// K steps each, on `cores` cores, in bits a core streams at its share of DRAM,
// times the cores: every wave but the last holds `resident` threadgroups on
// each core and the last the rest, each threadgroup streaming the weights of
// a step of 64 columns, of `bits16` sixteenths of a bit each, per step; the
// last wave takes at least `saturating` threadgroups' time when it is the
// grid's only one and `following`'s otherwise. A split grid's threadgroups
// each also store their fp32 partials of `rows` rows, which the last
// partition reads back.
uint64_t stagedTime(uint64_t groups, uint32_t steps, uint32_t splits, uint32_t rows, uint32_t bits16,
                    uint32_t cores, StagedOccupancy tile) {
  const uint64_t count = waves(groups, cores, tile.resident), full = (count - 1) * tile.resident * cores,
                 floor = uint64_t{count > 1 ? tile.following : tile.saturating} * cores;
  const uint64_t stepBits = uint64_t{GGUF_TILE_COLUMNS} * 32 * bits16 / 16,
                 partialBits = splits > 1 ? 2 * uint64_t{rows} * GGUF_TILE_COLUMNS * 32 : 0;
  return steps * stepBits * (full + std::max(groups - full, floor)) + groups * partialBits;
}

// The wave law's split of `rows` rows (a gate/up pair's two tensors when it
// may run in one pass), and passes, of least time; ties keep fewer
// partitions, then one pass. The law tells the pass modes apart only where a
// last wave falls below its floor (the 27B pair's 17 one-pass threadgroups per
// core on a 32-core M5 Max: the 17th runs a second wave alone, 333 us against
// 245 in two passes). Elsewhere it ties, and one pass takes up to 22% less
// time in 133 of the 152 tied cases measured. Below
// ProjectionCalibration::onePassFromCores a pair runs two passes whatever the
// law.
SplitChoice waveSplits(const WaveLaw &law, uint32_t n, uint32_t k, uint32_t rows, uint32_t bits16, uint32_t cores,
                       Passes passes) {
  const uint32_t tileRows = stagedTileRows(rows);
  SplitChoice best;
  uint64_t least = UINT64_MAX;
  for (uint32_t splits = 1; splits <= law.maximumSplits && (k / 32) % splits == 0 &&
                            (splits == 1 || k / splits >= law.partitionInputs);
       splits *= 2) {
    const uint32_t steps = k / 32 / splits;
    const uint64_t tiles = uint64_t{n / GGUF_TILE_COLUMNS} * splits;
    if (passes != Passes::Two) {
      const uint64_t time = stagedTime(2 * tiles, steps, splits, tileRows, bits16, cores, law.gateUpPair);
      if (time < least) least = time, best = {splits, true};
    }
    if (passes != Passes::One) {
      // A pair's two passes take twice one tensor's time.
      const uint64_t time =
          (passes == Passes::Either ? 2 : 1) * stagedTime(tiles, steps, splits, tileRows, bits16, cores, law.tile);
      if (time < least) least = time, best = {splits, false};
    }
  }
  return best;
}

// Whether the register tile runs a decode step of `lanes` lanes over
// `columnTiles` column tiles of a unit of `formatClass` (RegisterBoundary).
bool registers(const RegisterBoundary &boundary, FormatClass formatClass, uint32_t lanes, uint64_t columnTiles,
               uint32_t cores) {
  return 4 * columnTiles * lanes < uint64_t{boundary.belowQuarters[size_t(formatClass)][lanes - 1]} * cores;
}

// The staged tile's configuration of `rows` rows.
LinearConfig staged(const ProjectionCalibration &p, uint32_t n, uint32_t k, uint32_t rows, uint32_t bits16,
                    uint32_t cores, Passes passes) {
  const SplitChoice choice = splitsFor(p.stagedSplits, n, k, rows, bits16, cores, passes);
  return {.tile = LinearTile::GgufStaged, .splits = choice.splits, .oneGateUpPass = choice.onePass};
}

// Whether one unsplit lane of `tiles` column tiles of `bits16` sixteenths of
// a bit per weight runs the 16-row tile: never of 0 bits (float segments
// alone or unknown formats, unmeasured), below sixteenRowBelowBits16, and
// from there under a wave law where its grid runs in as many waves of the
// 16-row tile as of the 8-row tile, sixteenRowWaves at most.
bool sixteenRowLane(const ProjectionCalibration &p, uint64_t tiles, uint32_t bits16, uint32_t cores) {
  if (!bits16) return false;
  if (bits16 < p.sixteenRowBelowBits16) return true;
  const WaveLaw *law = std::get_if<WaveLaw>(&p.stagedSplits);
  if (!law) return false;
  const uint64_t waves16 = waves(tiles, cores, law->sixteenRowResident);
  return waves16 <= p.sixteenRowWaves && waves16 == waves(tiles, cores, law->tile.resident);
}

// A dense GGUF projection's configuration in bf16 activations. Prefill chunks
// of up to stagedPrefillRows rows run the staged tile at their rows' height
// with stagedSplits, longer ones the prefill tile unsplit. A decode step runs
// the register tile where it takes every quantized segment and the class's
// RegisterBoundary holds, a one-pass-eligible gate/up pair's column tiles
// counting both tensors', splitting by registerSplits; else the staged tile,
// splitting by stagedSplits (a wave law also picks one or two passes for an
// eligible pair), its walks spread from spreadFromCores, and one unsplit lane
// on the 16-row tile where sixteenRowLane takes it, a pair's only where both
// its tensors' own plans take it.
LinearConfig bf16Projection(const KernelCalibration &calibration, LinearWorkload w, const UnitFormats &unit,
                            std::span<const UnitFormats> pairTensors) {
  const auto [n, k] = w.matrix;
  const ProjectionCalibration &p = calibration.projection;
  const uint32_t cores = calibration.cores;
  if (w.phase == LinearPhase::Prefill) {
    if (w.rows <= p.stagedPrefillRows) return staged(p, n, k, w.rows, unit.weightBits16, cores, Passes::Two);
    return {.tile = LinearTile::GgufPrefill};
  }
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  const bool gateUp = w.epilogue == LinearEpilogue::GateUp,
             onePass = gateUp && unit.oneFormat && cores >= p.onePassFromCores;
  if (unit.registerTakes &&
      registers(p.decodeRegister, unit.formatClass, lanes, (onePass ? 2 : 1) * uint64_t{n / GGUF_TILE_COLUMNS},
                cores))
    return {.tile = LinearTile::GgufRegister,
            .splits = splitsFor(p.registerSplits, n, k, w.rows, unit.weightBits16, cores).splits};
  LinearConfig config = staged(p, n, k, w.rows, unit.weightBits16, cores, onePass ? Passes::Either : Passes::Two);
  config.spread = cores >= p.spreadFromCores;
  config.sixteenRows =
      lanes == 1 && config.splits == 1 &&
      (gateUp ? !pairTensors.empty() && std::all_of(pairTensors.begin(), pairTensors.end(),
                                                    [&](const UnitFormats &tensor) {
                                                      return bf16Projection(calibration, {w.matrix, w.rows}, tensor, {})
                                                          .sixteenRows;
                                                    })
              : sixteenRowLane(p, n / GGUF_TILE_COLUMNS, unit.weightBits16, cores));
  return config;
}

} // namespace

bool decodeInt8TakesSplits(uint32_t k, uint32_t splits) noexcept {
  return k % GGUF_ROTATION_BLOCK == 0 && splits && (k / 32) % (splits * kQuantFormats[GGUF_FMT_PQ20].meta_groups) == 0;
}

UnitFormats unitFormats(std::span<const Projection *const> projections) noexcept {
  UnitFormats unit;
  FormatClassifier classes;
  uint32_t bits = 0, count = 0;
  bool decodeInt8 = true, prefillInt8 = true, quantized = false;
  for (const Projection *p : projections) {
    if (!p) continue;
    bits += p->shape().weightBits16, ++count;
    for (const QuantizedSegment &s : p->blocks().segments) {
      if (s.isFloat()) continue;
      unit.registerTakes &= quant_register_tensor(s.formatId, s.inputSize);
      classes.add(s.formatId);
      decodeInt8 &= static_cast<bool>(p->rotation) && quant_decode_int8_format(s.formatId);
      prefillInt8 &= quant_prefill_int8_format(s.formatId);
      quantized = true;
    }
  }
  unit.decodeInt8Takes = quantized && decodeInt8;
  unit.prefillInt8Takes = quantized && prefillInt8;
  unit.formatClass = classes.formatClass();
  unit.weightBits16 = count ? bits / count : 0;
  if (projections.size() == 2 && projections[0] && projections[1]) {
    const std::vector<QuantizedSegment> &up = projections[0]->blocks().segments,
                                        &gate = projections[1]->blocks().segments;
    unit.oneFormat = up.size() == 1 && gate.size() == 1 && !up.front().isFloat() &&
                     up.front().formatId == gate.front().formatId;
  }
  return unit;
}

UnitFormats unitFormats(const ProjectionShape &shape) noexcept {
  return {.formatClass = shape.formatClass,
          .registerTakes = shape.registerFormats,
          .weightBits16 = shape.weightBits16,
          .prefillInt8Takes = shape.prefillInt8Formats};
}

SplitChoice splitsFor(const SplitLaw &law, uint32_t n, uint32_t k, uint32_t rows, uint32_t bits16, uint32_t cores,
                      Passes passes) noexcept {
  if (const TierLaw *tiers = std::get_if<TierLaw>(&law)) return {tierSplits(n, k, cores, tiers->tiers), false};
  return waveSplits(std::get<WaveLaw>(law), n, k, rows, bits16 ? bits16 : kMostWeightBits16, cores, passes);
}

// The configuration in bf16 activations (bf16Projection), then with int8
// activations on the plan's last choice, which leaves the rest of it as it is:
// a chunk on the prefill tile runs the int8 prefill tile where the row has
// prefillInt8 and the tile takes the unit, and a staged decode plan of at
// least decodeInt8FromRows rows (one lane's 16-row plan counts 16) whose K
// splits keep whole PQ2_0 blocks runs the int8 decode tiles where they take
// the unit, at the plan's rows, splits and passes.
LinearConfig selectProjection(const KernelCalibration &calibration, Int8Activations int8Activations,
                              LinearWorkload w, const UnitFormats &unit, std::span<const UnitFormats> pairTensors) {
  LinearConfig config = bf16Projection(calibration, w, unit, pairTensors);
  if (int8Activations == Int8Activations::Off) return config;
  const ProjectionCalibration &p = calibration.projection;
  if (config.tile == LinearTile::GgufPrefill && p.prefillInt8 && unit.prefillInt8Takes)
    config.tile = LinearTile::GgufPrefillInt8;
  config.decodeInt8 = w.phase == LinearPhase::Decode && config.tile == LinearTile::GgufStaged &&
                      unit.decodeInt8Takes && decodeInt8TakesSplits(w.matrix.inputSize, config.splits) &&
                      stagedStorageRows(w.rows, config.sixteenRows) >= p.decodeInt8FromRows;
  return config;
}

// The plan of the shape's flags; where the register tile may run (it runs a
// unit of FourBitsOrMore), a unit of other formats or one it does not take
// stages, of the most bits per weight, whose partials weigh least against the
// weights: the most K partitions of each pass mode, since the law's split
// never falls with the bits per weight and never rises with the rows (a
// split's partials grow with its rows alone, the same in either pass mode);
// and a gate/up pair of one format in one pass where it may.
std::vector<LinearConfig> decodeConfigurations(const KernelCalibration &calibration, LinearWorkload w,
                                               bool registerTakes) {
  const ProjectionCalibration &p = calibration.projection;
  const uint32_t cores = calibration.cores, n = w.matrix.outputSize, k = w.matrix.inputSize;
  std::vector<LinearConfig> configurations{bf16Projection(calibration, w, {.registerTakes = registerTakes}, {})};
  if (registers(p.decodeRegister, FormatClass::FourBitsOrMore, w.rows / SPLASH_TARGET_VERIFY_ROWS,
                n / GGUF_TILE_COLUMNS, cores))
    configurations.push_back(staged(p, n, k, w.rows, 0, cores, Passes::Two));
  if (w.epilogue == LinearEpilogue::GateUp && cores >= p.onePassFromCores)
    configurations.push_back(staged(p, n, k, w.rows, 0, cores, Passes::One));
  return configurations;
}

FloatTile selectFloatTile(const KernelCalibration &calibration, uint32_t rows, uint32_t outputSize,
                          uint32_t formatId) noexcept {
  const FloatTileCalibration &f = calibration.floatTile;
  const uint64_t tiles = uint64_t{(rows + 63) / 64} * ((outputSize + 31) / 32);
  // The grid in half tiles per core.
  const uint32_t halves = formatId == GGUF_FMT_BF16                 ? f.bf16Halves
                          : formatId == QuantizedSegment::kFloat32 ? f.f32Halves
                                                                   : f.quantizedRowsHalves;
  return f.neuralAccelerator && rows >= f.minimumRows && 2 * tiles >= uint64_t{halves} * calibration.cores
             ? FloatTile::NeuralAccelerator
             : FloatTile::Simdgroup;
}

uint32_t moeRouterFormat(std::span<const BlockMoeWeights *const> blocks) noexcept {
  // Routers per format, F32 counted one past the GGUF formats.
  std::array<uint32_t, GGUF_FMT_COUNT + 1> routers{};
  for (const BlockMoeWeights *block : blocks) ++routers[std::min<uint32_t>(block->router.formatId, GGUF_FMT_COUNT)];
  const auto router = std::max_element(routers.begin(), routers.end());
  return *router && router < routers.end() - 1 ? uint32_t(router - routers.begin()) : QuantizedSegment::kFloat32;
}

// Every GPU runs the experts on the staged tile: against the register tile, a
// 35B-shaped layer takes 16-42% less time on a 10-core M4 at one to four
// decode lanes of random routes; Apple9 GPUs of 11 to 40 cores are
// unmeasured. Plans run 8-row tiles while a prefill chunk's routes average at
// most one row per expert (rows * topK <= experts), where 8- and 32-row tiles
// take the same time, and 32-row tiles beyond, which stream an expert's
// weights once for up to 32 of its rows (its tiles run 8-, 16- or 32-row
// matmuls by their live rows, moe_live_rows). The router's float tile follows
// selectFloatTile by its rows' format.
MoeConfig selectMoe(const KernelCalibration &calibration, const MoeShape &shape, uint32_t routerFormat,
                    uint32_t rows, MoePhase phase) noexcept {
  MoeConfig config;
  config.expertTile = phase == MoePhase::Prefill && uint64_t{rows} * shape.expertsPerToken > shape.experts
                          ? MoeExpertTile::M32
                          : MoeExpertTile::M8;
  config.ggufRouterTile = selectFloatTile(calibration, rows, shape.experts, routerFormat);
  return config;
}

} // namespace splash::ops

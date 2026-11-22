#pragma once

// One selection rule per operator family over the device's KernelCalibration:
// dense GGUF projections, float projections and MoE blocks. Internal to ops:
// other code plans through Linear and ExecutionPlans.

#include "ops/KernelCalibration.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::ops {

// What selection reads of the quantized segments of a weight unit: one
// projection or a gate/up pair's two.
struct UnitFormats final {
  FormatClass formatClass = FormatClass::FourBitsOrMore;
  // The register tile takes every quantized segment (quant_register_tensor).
  bool registerTakes = true;
  // Their bits per weight in sixteenths (a pair's two projections' mean); 0
  // unknown, which the staged split law takes for the most a format stores.
  uint32_t weightBits16 = 0;
  // A gate/up pair of single tensors of one format, which the one-pass
  // kernels take.
  bool oneFormat = false;
  // The int8 decode tiles take every quantized segment: a rotated
  // projection's in a format they take (quant_decode_int8_format), over whose
  // rotated input one int8 scale per rotation block holds each row's values
  // evenly; of a shape, a rotated one's of such formats
  // (ProjectionShape::decodeInt8Formats).
  bool decodeInt8Takes = false;
  // The int8 prefill tile takes every quantized segment, of which there is
  // one at least (quant_prefill_int8_format).
  bool prefillInt8Takes = false;
  bool operator==(const UnitFormats &) const = default;
};
// The formats of the projections of a plan (a gate/up plan's up and gate).
[[nodiscard]] UnitFormats unitFormats(std::span<const Projection *const> projections) noexcept;
// Formats unknown (arena bounds): the shape's flags.
[[nodiscard]] UnitFormats unitFormats(const ProjectionShape &shape) noexcept;

// How a staged decode plan of a gate/up pair may run: each tensor in a pass of
// its own (also a single tensor's one pass), both in one, or either.
enum class Passes : uint8_t { Two, One, Either };
struct SplitChoice final {
  uint32_t splits = 1;
  bool onePass = false;
};
// The K splits of `law` for `rows` rows of an n x k matrix of `bits16`
// sixteenths of a bit per weight on `cores` cores; a wave law also chooses the
// passes of a gate/up pair among `passes`.
[[nodiscard]] SplitChoice splitsFor(const SplitLaw &law, uint32_t n, uint32_t k, uint32_t rows, uint32_t bits16,
                                    uint32_t cores, Passes passes = Passes::Two) noexcept;

// Whether the int8 decode tiles take `splits` K partitions of K = k: whole
// rotation blocks, and whole 128-input PQ2_0 blocks per partition.
[[nodiscard]] bool decodeInt8TakesSplits(uint32_t k, uint32_t splits) noexcept;
// A dense GGUF projection's configuration: the tile, its splits, spread, one
// pass and sixteen rows, then, with `int8Activations` on, whether it runs in
// int8 activations, as `step`'s work takes them (int8PrefillPaths). A gate/up
// decode's `pairTensors` are its up's and gate's own formats, whose plans
// decide its 16-row tile; none, it runs none.
[[nodiscard]] LinearConfig selectProjection(const KernelCalibration &calibration, Int8Activations int8Activations,
                                            StepKind step, LinearWorkload workload, const UnitFormats &unit,
                                            std::span<const UnitFormats> pairTensors = {});
// Every configuration a decode plan of the workload may take on the device
// when only a shape's formats are known (arena bounds): whether the register
// tile takes them, and whether the int8 decode tiles do, which run a staged
// configuration where `step`'s plans take them, at its storage rows and its
// splits or the most fewer that keep whole PQ2_0 blocks, on row groups of
// their own (int8RowGroups), each with its own counters.
[[nodiscard]] std::vector<LinearConfig> decodeConfigurations(const KernelCalibration &calibration,
                                                             Int8Activations int8Activations, StepKind step,
                                                             LinearWorkload workload, const UnitFormats &unit);

// The tile of a float projection of `rows` rows into `outputSize` columns of
// rows in `formatId` (QuantizedSegment::rows).
[[nodiscard]] FloatTile selectFloatTile(const KernelCalibration &calibration, uint32_t rows, uint32_t outputSize,
                                        uint32_t formatId) noexcept;

// The router format a MoE target's plans follow: one plan per step serves
// every block (QwenTarget.cpp), at the format of most routers' rows, F32 for
// none.
[[nodiscard]] uint32_t moeRouterFormat(std::span<const BlockMoeWeights *const> blocks) noexcept;
// Whether each projection's routed and shared experts are, in every block, in
// one format the int8 prefill tile takes: what the weights' files say, before
// and after they load. Whether the experts stream is the start's choice
// (RuntimeGeometry::from takes it); MoeShape::int8Experts is both.
[[nodiscard]] bool moeInt8Experts(std::span<const BlockMoeWeights *const> blocks) noexcept;
// The routes an expert takes on average from which a chunk's experts run the
// int8 passes (selectMoe).
inline constexpr uint32_t kInt8PassRoutesPerExpert = 6;
// A MoE plan's configuration of `rows` rows of `step`'s work, its routers in
// `routerFormat`, with int8 activations `int8Activations`.
[[nodiscard]] MoeConfig selectMoe(const KernelCalibration &calibration, Int8Activations int8Activations,
                                  StepKind step, const MoeShape &shape, uint32_t routerFormat,
                                  uint32_t rows) noexcept;

} // namespace splash::ops

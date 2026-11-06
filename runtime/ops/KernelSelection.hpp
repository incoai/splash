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
// projection, a gate/up pair's two, or a MoE target's routed experts (their
// shared segments enter registerTakes only).
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

// A dense GGUF projection's configuration: the tile, its splits, spread, one
// pass and sixteen rows. A gate/up decode's `pairTensors` are its up's and
// gate's own formats, whose plans decide its 16-row tile; none, it runs none.
[[nodiscard]] LinearConfig selectProjection(const KernelCalibration &calibration, LinearWorkload workload,
                                            const UnitFormats &unit, std::span<const UnitFormats> pairTensors = {});
// Every configuration a decode plan of the workload may take on the device
// when only whether the register tile takes its formats is known (arena
// bounds).
[[nodiscard]] std::vector<LinearConfig> decodeConfigurations(const KernelCalibration &calibration,
                                                             LinearWorkload workload, bool registerTakes);

// The tile of a float projection of `rows` rows into `outputSize` columns of
// rows in `formatId` (QuantizedSegment::rows).
[[nodiscard]] FloatTile selectFloatTile(const KernelCalibration &calibration, uint32_t rows, uint32_t outputSize,
                                        uint32_t formatId) noexcept;

// The formats a MoE target's plans follow. Its routed experts are one weight
// unit: one plan per step serves every block (QwenTarget.cpp). Their class is
// the class of the format holding most routed elements; registerTakes holds
// over every block's routed and shared segments; the router is the format of
// most routers' rows, F32 for none.
struct MoeFormats final {
  UnitFormats experts;
  uint32_t router = QuantizedSegment::kFloat32;
  bool operator==(const MoeFormats &) const = default;
};
[[nodiscard]] MoeFormats moeFormats(std::span<const BlockMoeWeights *const> blocks) noexcept;
// A MoE plan's configuration of `rows` rows in `phase`.
[[nodiscard]] MoeConfig selectMoe(const KernelCalibration &calibration, const MoeShape &shape,
                                  const MoeFormats &formats, uint32_t rows, MoePhase phase) noexcept;

} // namespace splash::ops

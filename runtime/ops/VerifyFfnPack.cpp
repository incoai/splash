#include "ops/VerifyFfnPack.hpp"
#include "metal/abi/Gguf.h"
#include "ops/BufferExtent.hpp"
#include <cstddef>
namespace splash::ops {
namespace {
bool supported(const Projection *p) {
  return p && !p->rotation && !p->planeInputs() && p->destination == FloatOutput::BFloat16 &&
         p->blocks().segments.size() == 1 && !p->blocks().segments.front().isFloat() &&
         p->blocks().segments.front().formatId == GGUF_FMT_AF4G64;
}
} // namespace
bool addVerifyFfnPack(metal::CommandGraph &graph, const SwiGluProjections &ffn, metal::MetalBuffer normalized,
                      metal::MetalBuffer residual, metal::MetalBuffer output, metal::MetalBuffer gateScratch,
                      metal::MetalBuffer wideIntermediate, LinearScratch scratch,
                      metal::MetalBuffer verifyRows, const VerifyFfnPackBuffers &b, const LinearPlan &upPlan,
                      const LinearPlan &downPlan, uint32_t lanes) {
  if (!b.control || lanes < 5 || lanes > 8 || !supported(ffn.gate) || !supported(ffn.up) ||
      !supported(ffn.down) || upPlan.configuration().tile != LinearTile::GgufStaged ||
      downPlan.configuration().tile != LinearTile::GgufStaged || upPlan.storageRows() != 64 ||
      downPlan.storageRows() != 64 || upPlan.input() != LinearInput::Plain ||
      downPlan.input() != LinearInput::Plain || upPlan.configuration().oneGateUpPass ||
      upPlan.workload().epilogue != LinearEpilogue::GateUp ||
      downPlan.workload().epilogue != LinearEpilogue::Residual)
    return false;
  const uint32_t H = ffn.up->inputSize, I = ffn.up->outputSize;
  if (!H || !I || H % 64 || I % 64 || ffn.gate->inputSize != H || ffn.gate->outputSize != I ||
      ffn.down->inputSize != I || ffn.down->outputSize != H)
    return false;
  const auto u = upPlan.configuration(), d = downPlan.configuration();
  auto smallU = u;
  smallU.oneGateUpPass = true;
  smallU.sixteenRows = false;
  auto smallD = d;
  smallD.sixteenRows = false;
  const auto compactUp =
      Linear::plan({{I, H}, 32, LinearPhase::Decode, LinearEpilogue::GateUp}, smallU, FloatOutput::BFloat16);
  const auto compactDown = Linear::plan({{H, I}, 32, LinearPhase::Decode, LinearEpilogue::Residual}, smallD,
                                        FloatOutput::BFloat16);
  LinearScratchSize needed = upPlan.scratchSize();
  needed.include(downPlan.scratchSize());
  needed.include(compactUp.scratchSize());
  needed.include(compactDown.scratchSize());
  requireBytes(scratch.partials, needed.partials, "packed FFN partials");
  requireBytes(scratch.counters, needed.counters, "packed FFN counters");
  requireBytes(normalized, uint64_t{64} * H * 2, "wide FFN input");
  requireBytes(residual, uint64_t{64} * H * 2, "wide FFN residual");
  requireBytes(output, uint64_t{64} * H * 2, "wide FFN output");
  requireBytes(gateScratch, uint64_t{64} * I * 2, "wide FFN gate");
  requireBytes(wideIntermediate, uint64_t{64} * I * 2, "wide FFN intermediate");
  requireBytes(verifyRows, uint64_t{lanes} * 4, "packed FFN verify rows");
  requireBytes(b.control, sizeof(VerifyPackControl), "packed FFN control");
  for (const auto *v : {&b.input, &b.residual, &b.output})
    requireBytes(*v, uint64_t{32} * H * 2, "compact FFN rows");
  requireBytes(b.intermediate, uint64_t{32} * I * 2, "compact FFN intermediate");
  for (const Projection *p : {ffn.gate, ffn.up, ffn.down})
    requireSegmentPlanes(p->blocks().segments.front(), "packed FFN weight");
  // Unsplit kernels never touch these bindings; bind an existing output if no split plan needs scratch.
  const auto partial = scratch.partials ? scratch.partials : output;
  const auto counters = scratch.counters ? scratch.counters : output;
  const auto &g = ffn.gate->blocks().segments.front(), &w = ffn.up->blocks().segments.front(),
             &z = ffn.down->blocks().segments.front();
  const VerifyPackParams p{lanes, H, I, u.splits, d.splits};
  const GgufDecodeParams pu{H, u.splits, I, 0, uint32_t(u.spread)}, pd{I, d.splits, H, 0, uint32_t(d.spread)};
  const auto grid = [&](uint32_t n) {
    return metal::IndirectGrid{b.control, offsetof(VerifyPackControl, grids) + n * 12};
  };
  graph.add("verify_ffn_pack_map", {verifyRows, b.control}, p, {1, 1, 1}, {1, 1, 1});
  graph.add("verify_ffn_pack_gather", {normalized, residual, b.control, b.input, b.residual}, p,
            {(H + 255) / 256, 32, 1}, grid(0), {256, 1, 1});
  graph.add("gguf_decode_af4g64_m32_gate_up",
            {b.input, g.plane0, g.plane1Slot(), g.meta, w.plane0, w.plane1Slot(), w.meta, b.intermediate,
             partial, counters},
            pu, {I / 32, u.splits, 1}, grid(1), {32, 1, 1});
  graph.add("gguf_decode_af4g64_m32_r",
            {b.intermediate, z.plane0, z.plane1Slot(), z.meta, b.output, partial, counters, b.residual}, pd,
            {H / 64, d.splits, 1}, grid(2), {64, 1, 1});
  graph.add("verify_ffn_pack_scatter", {b.output, residual, b.control, output}, p, {(H + 255) / 256, 64, 1},
            grid(3), {256, 1, 1});
  graph.add("gguf_decode_af4g64_m64_a",
            {normalized, g.plane0, g.plane1Slot(), g.meta, gateScratch, partial, counters, normalized}, pu,
            {I / 64, u.splits, 1}, grid(4), {64, 1, 1});
  graph.add("gguf_decode_af4g64_m64_g",
            {normalized, w.plane0, w.plane1Slot(), w.meta, wideIntermediate, partial, counters, gateScratch},
            pu, {I / 64, u.splits, 1}, grid(5), {64, 1, 1});
  graph.add("gguf_decode_af4g64_m64_r",
            {wideIntermediate, z.plane0, z.plane1Slot(), z.meta, output, partial, counters, residual}, pd,
            {H / 64, d.splits, 1}, grid(6), {64, 1, 1});
  return true;
}
} // namespace splash::ops

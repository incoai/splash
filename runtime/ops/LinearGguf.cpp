// GGUF projections: plan policy and dispatch (kernels/shared/gguf_linear.metal).
#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::ops {
namespace {

// Decode tiles: 64 output columns per threadgroup, two simdgroups of 32
// columns each. Prefill tiles: 64 columns, four simdgroups of 32 or 8 rows.
constexpr uint32_t kDecodeTileColumns = 64;
constexpr uint32_t kDecodeThreads = 64;
constexpr uint32_t kPrefillTileColumns = 64;
constexpr uint32_t kPrefillThreads = 128;
constexpr uint32_t kPrefillRows = 128;
constexpr uint32_t kSmallPrefillRows = 32;

std::string decodeKernel(const char *family, const char *format, uint32_t rows) {
  return std::string(family) + "_" + format + "_m" + std::to_string(rows) + "_c32_sg2_k32_b2_p1";
}
std::string prefillKernel(const char *family, const char *format, uint32_t tileRows) {
  return std::string(family) + "_" + format + (tileRows == kSmallPrefillRows ? "_r8" : "_r32") +
         "_sg4_n64_k64_p1";
}

// Measured with serialized M=8 sweeps on a 16-core Apple10 GPU: four splits
// for out_proj (K 6144), eight for down (K 17408). Not derived from the core
// count yet.
uint32_t stagedSplits(uint32_t n, uint32_t k, uint32_t rows) {
  if (n <= 1024) return 8;
  if (n <= 6144) return k <= 6144 ? 4 : 8;
  if (n <= 12288) return rows <= 16 ? 4 : 2;
  return 1;
}

void requireSegments(const Q4Projection &p, LinearMatrix matrix) {
  uint32_t covered = 0;
  for (const GgufSegment &s : p.gguf) {
    if (s.inputSize != matrix.inputSize || s.columnOffset != covered ||
        s.outputSize % kDecodeTileColumns)
      throw std::invalid_argument("GGUF segments do not tile the projection");
    covered += s.outputSize;
  }
  if (covered != matrix.outputSize)
    throw std::invalid_argument("GGUF segments do not cover the projection");
}

} // namespace

LinearConfig Q4Linear::ggufBaseline(LinearWorkload w, uint32_t segments) const {
  if (w.phase == LinearPhase::Prefill) return {LinearTile::GgufStaged, 0, LinearSimdgroups::Four};
  const auto [n, k] = w.matrix;
  // The fused multi-tensor and gate/up kernels take no K splits.
  const uint32_t splits =
      segments > 1 || w.epilogue == LinearEpilogue::GateUp ? 1 : stagedSplits(n, k, w.rows);
  return {LinearTile::GgufStaged, n / kDecodeTileColumns, LinearSimdgroups::Two, splits};
}

void Q4Linear::addGguf(metal::CommandGraph &graph, const LinearBuffers &b,
                       const Q4Projection &p, const LinearPlan &plan,
                       const Q4Projection *gate, Q4DispatchStats *stats) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  if (config.tile != LinearTile::GgufStaged || w.quant != QuantFamily::Gguf)
    throw std::invalid_argument("GGUF projection requires a GGUF plan");
  const auto [n, k] = w.matrix;
  requireSegments(p, w.matrix);
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (!gate || gate->gguf.empty()) throw std::invalid_argument("GGUF gate projection is missing");
    requireSegments(*gate, w.matrix);
  } else if (gate) {
    throw std::invalid_argument("unexpected GGUF gate projection");
  }
  const uint32_t rows = plan.storageRows();
  const auto need = [&](const metal::MetalBuffer &buffer, uint64_t bytes, const char *what) {
    if (buffer.sizeBytes() < bytes)
      throw std::invalid_argument(std::string("GGUF ") + what + " buffer holds " +
                                  std::to_string(buffer.sizeBytes()) + " bytes, needs " +
                                  std::to_string(bytes) + " (rows " + std::to_string(rows) +
                                  ", n " + std::to_string(n) + ")");
  };
  need(b.input, uint64_t{rows} * k * 2, "input");
  need(b.output, uint64_t{rows} * n * 2, "output");
  if (w.epilogue == LinearEpilogue::Residual) need(b.residual, uint64_t{rows} * n * 2, "residual");
  need(b.gateScratch, plan.gateScratchBytes(), "gate scratch");
  const auto epilogueId = [](LinearEpilogue e) {
    return e == LinearEpilogue::Residual     ? GGUF_EPILOGUE_RESIDUAL
           : e == LinearEpilogue::UpWithGate ? GGUF_EPILOGUE_UP_WITH_GATE
                                             : GGUF_EPILOGUE_NONE;
  };
  const auto plane1 = [](const GgufSegment &s) { return s.plane1 ? s.plane1 : s.meta; };
  if (w.phase == LinearPhase::Prefill) {
    // One dispatch per segment over 128-row tiles (32-row tiles for short
    // chunks); rows past w.rows stay inside the budget-sized prefill buffers.
    const uint32_t tileRows = rows == kSmallPrefillRows ? kSmallPrefillRows : kPrefillRows;
    const metal::MetalBuffer aux = w.epilogue == LinearEpilogue::Residual ? b.residual
                                 : w.epilogue == LinearEpilogue::UpWithGate ? b.gateScratch
                                                                             : b.output;
    for (const GgufSegment &s : p.gguf) {
      const GgufParams params{s.outputSize, k, 0, n, s.columnOffset};
      const metal::DispatchSize grid{rows / tileRows, s.outputSize / kPrefillTileColumns, 1};
      if (w.epilogue == LinearEpilogue::None)
        graph.add(prefillKernel("pfa", s.format, tileRows), {b.input, s.plane0, plane1(s), s.meta, b.output},
                  params, grid, {kPrefillThreads, 1, 1});
      else
        graph.add(prefillKernel(w.epilogue == LinearEpilogue::Residual ? "pfr" : "pfg", s.format, tileRows),
                  {b.input, s.plane0, plane1(s), s.meta, b.output, aux}, params, grid, {kPrefillThreads, 1, 1});
    }
    return;
  }
  // Decode: one dispatch per projection. Fused multi-format segments
  // (qkv|z|ab, q|k|v) and gate+up run one kernel each; single tensors run
  // plain tiles or split-K with the reduction in the last arriving group.
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (gate->gguf.size() != 1 || p.gguf.size() != 1 || config.splits != 1)
      throw std::invalid_argument("GGUF gate/up requires single tensors and no K splits");
    const GgufSegment &g = gate->gguf.front(), &u = p.gguf.front();
    graph.add("gguf_gateup_m" + std::to_string(rows),
              {b.input, g.plane0, plane1(g), g.meta, u.plane0, plane1(u), u.meta, b.output},
              GgufGateUpParams{k, n, n, g.formatId, u.formatId}, {n / kDecodeTileColumns, 1, 1},
              {kDecodeThreads, 1, 1});
  } else if (p.gguf.size() > 1) {
    if (p.gguf.size() > 3 || w.epilogue != LinearEpilogue::None || config.splits != 1)
      throw std::invalid_argument("GGUF fused projection needs <= 3 segments, no epilogue and no K splits");
    // Dispatch order is tile order: segments with the most bytes per tile
    // first, so their threadgroups do not form the tail (alpha/beta are Q8_0).
    std::vector<const GgufSegment *> order;
    for (const GgufSegment &s : p.gguf) order.push_back(&s);
    const auto bitsPerWeight = [](const GgufSegment &s) {
      return (s.p0 + s.p1) * 8.0 / 32.0 + s.metaBytes * 8.0 / (32.0 * s.metaGroups);
    };
    std::stable_sort(order.begin(), order.end(), [&](const GgufSegment *a, const GgufSegment *c) {
      return bitsPerWeight(*a) > bitsPerWeight(*c);
    });
    GgufFusedParams fp{k, n, static_cast<uint32_t>(p.gguf.size()), 0, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<metal::MetalBuffer> bindings{b.input};
    for (size_t i = 0; i < 3; ++i) {
      const GgufSegment &s = *order[std::min(i, order.size() - 1)];
      if (i < order.size()) {
        fp.cols[i] = s.outputSize;
        fp.fmt[i] = s.formatId;
        fp.offset[i] = s.columnOffset;
      }
      bindings.insert(bindings.end(), {s.plane0, plane1(s), s.meta});
    }
    bindings.push_back(b.output);
    graph.add("gguf_fused_m" + std::to_string(rows), std::move(bindings), fp,
              {n / kDecodeTileColumns, 1, 1}, {kDecodeThreads, 1, 1});
  } else {
    const GgufSegment &s = p.gguf.front();
    const uint32_t tiles = n / kDecodeTileColumns, splits = config.splits;
    const metal::MetalBuffer aux = w.epilogue == LinearEpilogue::Residual ? b.residual : b.output;
    if (splits > 1) {
      const LinearScratchSize scratch = plan.scratchSize();
      need(b.scratch.partials, scratch.partials, "partials");
      need(b.scratch.counters, scratch.counters, "counters");
      graph.add(std::string("gguf_splitk_") + s.format + "_m" + std::to_string(rows),
                {b.input, s.plane0, plane1(s), s.meta, b.scratch.partials, b.scratch.counters, b.output, aux},
                GgufSplitParams{n, k, splits, n, 0, epilogueId(w.epilogue)}, {tiles, splits, 1},
                {kDecodeThreads, 1, 1});
    } else if (w.epilogue == LinearEpilogue::None) {
      graph.add(decodeKernel("sga", s.format, rows), {b.input, s.plane0, plane1(s), s.meta, b.output},
                GgufParams{n, k, tiles, n, 0}, {tiles, 1, 1}, {kDecodeThreads, 1, 1});
    } else {
      graph.add(decodeKernel("sgr", s.format, rows),
                {b.input, s.plane0, plane1(s), s.meta, b.output, aux}, GgufParams{n, k, tiles, n, 0},
                {tiles, 1, 1}, {kDecodeThreads, 1, 1});
    }
  }
  if (stats && rows > SPLASH_TARGET_VERIFY_ROWS) {
    const uint32_t lanes = rows / SPLASH_TARGET_VERIFY_ROWS;
    stats->fusedSourceOperations += lanes;
    if (lanes == 2) ++stats->m16Dispatches;
    else if (lanes == 3) ++stats->m24Dispatches;
    else ++stats->m32Dispatches;
  }
}

} // namespace splash::ops

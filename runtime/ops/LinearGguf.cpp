// GGUF projections: plan policy and dispatch (kernels/shared/gguf_linear.metal,
// kernels/decode/linear_gguf_sgmatrix.metal).
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
// columns each. Prefill tiles: 64 columns, four simdgroups of 32 rows.
constexpr uint32_t kDecodeTileColumns = 64;
constexpr uint32_t kDecodeThreads = 64;
constexpr uint32_t kPrefillTileColumns = 64;
constexpr uint32_t kPrefillThreads = 128;
constexpr uint32_t kPrefillRows = 128;

std::string decodeKernel(const std::string &family, const char *format, uint32_t rows) {
  return family + "_" + format + "_m" + std::to_string(rows) + "_c32_sg2_k32_b2_p1";
}
std::string prefillKernel(const std::string &family, const char *format) {
  return family + "_" + format + "_r32_sg4_n64_k64_p1";
}

// Staged tile: split K until the grid holds 32 threadgroups per core,
// keeping at least 1024 inputs (whole 32-input groups) per partition. Like
// the register rule it ignores the batch width, so a request's sums do not
// depend on the requests it is batched with. On a 16-core M5 Pro over the 27B
// shapes at one to four lanes (DRAM-cold medians): 2.2% over the fastest
// split in total, 1.2% on the single tensors the 27B splits.
uint32_t stagedSplits(uint32_t n, uint32_t k, uint32_t cores) {
  const uint32_t grid = n / kDecodeTileColumns;
  uint32_t splits = 1;
  while (splits < 8 && uint64_t(grid) * splits < 32ULL * cores && k / (2 * splits) >= 1024) splits *= 2;
  return splits;
}

// Apple9 register tile: split K until the grid holds sixteen threadgroups
// per core, keeping at least two 256-input coefficient units per partition.
// On a 40-core M3 over the 27B shapes at one to four lanes, within 3.3% of
// the fastest split everywhere and 0.4% in total.
uint32_t simdgroupSplits(uint32_t n, uint32_t k, uint32_t cores) {
  const uint32_t grid = n / kDecodeTileColumns, units = k / 256;
  uint32_t splits = 1;
  while (splits < 8 && uint64_t(grid) * splits < 16ULL * cores && units / (2 * splits) >= 2)
    splits *= 2;
  return splits;
}

const metal::MetalBuffer &plane1(const GgufSegment &s) { return s.plane1 ? s.plane1 : s.meta; }

// The request lanes one decode dispatch fuses, whatever its tile height.
void recordLanes(Q4DispatchStats *stats, uint32_t rows) {
  if (!stats || rows <= SPLASH_TARGET_VERIFY_ROWS) return;
  const uint32_t lanes = rows / SPLASH_TARGET_VERIFY_ROWS;
  stats->fusedSourceOperations += lanes;
  if (lanes == 2) ++stats->m16Dispatches;
  else if (lanes == 3) ++stats->m24Dispatches;
  else ++stats->m32Dispatches;
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
  // Prefill: 128-row tiles. A chunk of up to 32 rows runs the decode tile
  // of its rows rounded up to eight-row lanes (two simdgroups): the same
  // half stage and matmul rows, so its outputs equal the prefill tile's bit
  // for bit (gguf-projection full), and each simdgroup streams its own 32
  // columns instead of four 8-row simdgroups sharing a stage. On a 17408 x
  // 5120 Q4_K projection that is 1.8-2.9x faster on a 16-core M5 Pro (its
  // neural accelerator pads 8 rows to 16) and 1.1-2.8x on a 40-core M3 Max.
  if (w.phase == LinearPhase::Prefill)
    return {LinearTile::GgufStaged, 0,
            w.rows <= SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS ? LinearSimdgroups::Two
                                                                              : LinearSimdgroups::Four};
  const auto [n, k] = w.matrix;
  // Apple9 runs matrix operations on the FP32 pipe, so the exact register
  // kernel beats staging (scratchpad DESIGN); every segment and gate/up split.
  if (appleGpuFamily_ == 9)
    return {LinearTile::GgufSimdgroup, n / kDecodeTileColumns, LinearSimdgroups::Four,
            simdgroupSplits(n, k, gpuCores_)};
  // The fused multi-tensor and gate/up kernels take no K splits.
  const uint32_t splits =
      segments > 1 || w.epilogue == LinearEpilogue::GateUp ? 1 : stagedSplits(n, k, gpuCores_);
  return {LinearTile::GgufStaged, n / kDecodeTileColumns, LinearSimdgroups::Two, splits};
}

void Q4Linear::addGguf(metal::CommandGraph &graph, const LinearBuffers &b,
                       const Q4Projection &p, const LinearPlan &plan,
                       const Q4Projection *gate, Q4DispatchStats *stats) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  if ((config.tile != LinearTile::GgufStaged && config.tile != LinearTile::GgufSimdgroup) ||
      w.quant != QuantFamily::Gguf)
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
  if (config.tile == LinearTile::GgufSimdgroup) {
    addGgufSimdgroup(graph, b, p, plan, gate);
    recordLanes(stats, w.rows);
    return;
  }
  if (w.phase == LinearPhase::Prefill) {
    // One dispatch per segment, over the decode tiles (ggufBaseline) or
    // 128-row tiles; rows past w.rows stay inside the budget-sized prefill
    // buffers, and the simdgroups of a 128-row tile that only hold them skip
    // their matmuls.
    const metal::MetalBuffer aux = w.epilogue == LinearEpilogue::Residual ? b.residual
                                 : w.epilogue == LinearEpilogue::UpWithGate ? b.gateScratch
                                                                             : b.output;
    const char epilogue = w.epilogue == LinearEpilogue::None       ? 'a'
                        : w.epilogue == LinearEpilogue::Residual   ? 'r'
                                                                   : 'g';
    for (const GgufSegment &s : p.gguf) {
      std::vector<metal::MetalBuffer> bindings{b.input, s.plane0, plane1(s), s.meta, b.output};
      if (w.epilogue != LinearEpilogue::None) bindings.push_back(aux);
      if (config.simdgroups == LinearSimdgroups::Two) {
        const uint32_t tiles = s.outputSize / kDecodeTileColumns;
        graph.add(decodeKernel(std::string("sg") + epilogue, s.format, rows), std::move(bindings),
                  GgufParams{s.outputSize, k, tiles, n, s.columnOffset}, {tiles, 1, 1}, {kDecodeThreads, 1, 1});
      } else {
        graph.add(prefillKernel(std::string("pf") + epilogue, s.format), std::move(bindings),
                  GgufPrefillParams{s.outputSize, k, w.rows, n, s.columnOffset},
                  {rows / kPrefillRows, s.outputSize / kPrefillTileColumns, 1}, {kPrefillThreads, 1, 1});
      }
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
  recordLanes(stats, w.rows);
}

// All lanes in each threadgroup. Single tensors run their format's kernel;
// fused projections (qkv|z|ab, q|k|v) run every segment in one dispatch.
// Gate/up runs as a gate pass into the gate scratch and an up pass whose
// epilogue applies silu(gate) to the bf16 up value, as the fused kernels do.
void Q4Linear::addGgufSimdgroup(metal::CommandGraph &graph, const LinearBuffers &b,
                                const Q4Projection &p, const LinearPlan &plan,
                                const Q4Projection *gate) const {
  const LinearWorkload w = plan.workload();
  const LinearConfig config = plan.configuration();
  const auto [n, k] = w.matrix;
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  const LinearScratchSize size = plan.scratchSize();
  const auto need = [](const metal::MetalBuffer &buffer, uint64_t bytes, const char *what) {
    if (buffer.sizeBytes() < bytes)
      throw std::invalid_argument(std::string("GGUF simdgroup ") + what + " scratch is below the plan");
  };
  need(b.scratch.input, size.input, "table");
  need(b.scratch.sums, size.sums, "sums");
  need(b.scratch.partials, size.partials, "partials");
  need(b.scratch.counters, size.counters, "counters");
  if (b.prepared.layout != LinearInput::Table16 || !b.prepared.source.sameView(b.input))
    graph.add("decode_linear_gguf_prepare", {b.input, b.scratch.input, b.scratch.sums}, k,
              {k / 32, lanes, 1}, {128, 1, 1});
  const metal::DispatchSize grid{n / kDecodeTileColumns, config.splits, 1};
  const std::string suffix = "_l" + std::to_string(lanes);
  if (p.gguf.size() > 1) {
    if (p.gguf.size() > 3 || w.epilogue != LinearEpilogue::None)
      throw std::invalid_argument("GGUF fused projection needs <= 3 segments and no epilogue");
    GgufSgFusedParams params{k, config.splits, n, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, b.scratch.sums};
    for (size_t i = 0; i < 3; ++i) {
      const GgufSegment &s = p.gguf[std::min(i, p.gguf.size() - 1)];
      if (i < p.gguf.size()) {
        params.cols[i] = s.outputSize;
        params.fmt[i] = s.formatId;
        params.offset[i] = s.columnOffset;
      }
      bindings.insert(bindings.end(), {s.plane0, plane1(s), s.meta});
    }
    bindings.insert(bindings.end(), {b.output, b.scratch.partials, b.scratch.counters});
    graph.add("decode_linear_gguf_sg_fused" + suffix, std::move(bindings), params, grid, {128, 1, 1});
    return;
  }
  const auto tensor = [&](const GgufSegment &s, char epilogue, const metal::MetalBuffer &output,
                          const metal::MetalBuffer &aux) {
    graph.add(std::string("decode_linear_gguf_sg_") + s.format + suffix + "_" + epilogue,
              {b.scratch.input, b.scratch.sums, s.plane0, plane1(s), s.meta, output, b.scratch.partials,
               b.scratch.counters, aux},
              GgufSgParams{k, config.splits, n, s.columnOffset}, grid, {128, 1, 1});
  };
  switch (w.epilogue) {
  case LinearEpilogue::None: tensor(p.gguf.front(), 'a', b.output, b.output); break;
  case LinearEpilogue::Residual: tensor(p.gguf.front(), 'r', b.output, b.residual); break;
  case LinearEpilogue::GateUp:
    if (gate->gguf.size() != 1) throw std::invalid_argument("GGUF gate/up requires single tensors");
    tensor(gate->gguf.front(), 'a', b.gateScratch, b.gateScratch);
    tensor(p.gguf.front(), 'g', b.output, b.gateScratch);
    break;
  case LinearEpilogue::UpWithGate: throw std::invalid_argument("GGUF decode has no up-with-gate projection");
  }
}

} // namespace splash::ops

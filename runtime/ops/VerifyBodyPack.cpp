#include "ops/VerifyBodyPack.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/VerifyBodyPack.h"
#include "ops/BufferExtent.hpp"
namespace splash::ops {
bool supportsVerifyBodyPack(const Projection &projection, const LinearPlan &widePlan, bool residual,
                            uint32_t lanes) {
  const auto w = widePlan.workload();
  const auto config = widePlan.configuration();
  if (lanes < 5 || lanes > 8 || projection.rotation || projection.planeInputs() ||
      projection.destination != FloatOutput::BFloat16 || config.tile != LinearTile::GgufStaged ||
      widePlan.storageRows() != 64 || widePlan.input() != LinearInput::Plain || config.oneGateUpPass ||
      (w.epilogue != LinearEpilogue::None && w.epilogue != LinearEpilogue::Residual) ||
      (w.epilogue == LinearEpilogue::Residual && projection.blocks().segments.size() != 1) ||
      w.phase != LinearPhase::Decode || w.matrix.inputSize != projection.inputSize ||
      w.matrix.outputSize != projection.outputSize)
    return false;
  const uint32_t K = projection.inputSize, N = projection.outputSize;
  if (!K || !N || K % 64 || N % 8 || (bool(residual) != (w.epilogue == LinearEpilogue::Residual)))
    return false;
  uint32_t end = 0;
  for (const auto &s : projection.blocks().segments) {
    if (s.inputSize != K || s.columnOffset != end || !s.outputSize || end > N || s.outputSize > N - end)
      return false;
    if (s.isFloat()) {
      if (w.epilogue != LinearEpilogue::None ||
          (s.formatId != QuantizedSegment::kFloat32 && s.formatId != GGUF_FMT_BF16) || s.outputSize % 8)
        return false;
    } else if (s.formatId != GGUF_FMT_AF4G64 || s.outputSize % 64 || s.columnOffset % 64)
      return false;
    end += s.outputSize;
  }
  if (!end || end > N)
    return false;
  return true;
}
bool addVerifyBodyPack(const Linear &linear, metal::CommandGraph &graph, const Projection &projection,
                       const LinearPlan &widePlan, metal::MetalBuffer input, metal::MetalBuffer residual,
                       metal::MetalBuffer output, LinearScratch scratch, metal::MetalBuffer verifyRows,
                       const VerifyBodyPackBuffers &b, uint32_t lanes) {
  if (!b.input || !supportsVerifyBodyPack(projection, widePlan, bool(residual), lanes))
    return false;
  const auto w = widePlan.workload();
  const auto config = widePlan.configuration();
  const uint32_t K = projection.inputSize, N = projection.outputSize;
  const auto &last = projection.blocks().segments.back();
  const uint32_t end = last.columnOffset + last.outputSize;
  auto smallConfig = config;
  smallConfig.sixteenRows = false;
  const auto small =
      Linear::plan({w.matrix, 32, LinearPhase::Decode, w.epilogue}, smallConfig, FloatOutput::BFloat16);
  auto needed = widePlan.scratchSize();
  needed.include(small.scratchSize());
  requireBytes(scratch.partials, needed.partials, "packed body partials");
  requireBytes(scratch.counters, needed.counters, "packed body counters");
  requireBytes(input, uint64_t{64} * K * 2, "wide body input");
  requireBytes(output, uint64_t{64} * N * 2, "wide body output");
  if (residual)
    requireBytes(residual, uint64_t{64} * N * 2, "wide body residual");
  requireBytes(verifyRows, uint64_t{lanes} * 4, "packed body verify rows");
  requireBytes(b.input, uint64_t{32} * K * 2, "compact body input");
  requireBytes(b.output, uint64_t{32} * N * 2, "compact body output");
  if (residual)
    requireBytes(b.residual, uint64_t{32} * N * 2, "compact body residual");
  for (const auto &s : projection.blocks().segments) {
    if (s.isFloat())
      requireBytes(s.plane0, uint64_t{s.outputSize} * QuantizedSegment::rowBytes(s.formatId, K),
                   "packed body float weight");
    else
      requireSegmentPlanes(s, "packed body weight");
  }
  const VerifyBodyPackParams p{lanes, N, K, uint32_t(bool(residual)), end};
  graph.add("verify_body_pack_gather",
            {input, residual ? residual : input, verifyRows, b.input, b.residual ? b.residual : b.input}, p,
            {(std::max(K, residual ? N : 0u) + 255) / 256, 32, 1}, {256, 1, 1});
  linear.add(graph,
             {.input = b.input,
              .output = b.output,
              .residual = residual ? b.residual : metal::MetalBuffer{},
              .scratch = scratch},
             projection, small);
  graph.add("verify_body_pack_scatter", {b.output, verifyRows, output}, p, {(N + 255) / 256, 64, 1},
            {256, 1, 1});
  return true;
}
} // namespace splash::ops

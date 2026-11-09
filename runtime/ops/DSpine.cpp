#include "ops/DSpine.hpp"
#include "ops/BufferExtent.hpp"
#include "ops/Embedding.hpp"

#include <cmath>
#include <stdexcept>

namespace splash::ops {
namespace {
void requireShape(DSpineParams p) {
  if (!p.hidden || p.hidden > 5120 || !p.code || p.code > 1024 || !p.rank || p.rank > 512 ||
      !p.rows || p.rows % SPLASH_DRAFT_QUERY_ROWS || p.rows > SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_DRAFT_QUERY_ROWS)
    throw std::invalid_argument("unsupported DSpine bridge shape");
}
uint64_t hiddenBytes(DSpineParams p) { return uint64_t{p.rows} * p.hidden * 2; }
void requireStash(const DSpineBuffers &b, DSpineParams p) {
  requireShape(p);
  requireBytes(b.raw, hiddenBytes(p), "DSpine raw hidden");
  requireBytes(b.rms, uint64_t{p.rows} * 4, "DSpine RMS");
}
void gatherHead(metal::CommandGraph &graph, const Projection &head, metal::MetalBuffer ids,
                metal::MetalBuffer output, uint32_t rows) {
  if (head.rotation || head.blocks().segments.size() != 1)
    throw std::invalid_argument("experimental DSpine requires a single unrotated vocabulary projection");
  const QuantizedSegment &s = head.blocks().segments.front();
  requireBytes(output, uint64_t{rows} * head.inputSize * 2, "DSpine candidate head");
  if (s.isFloat()) {
    if (s.formatId == QuantizedSegment::kFloat32) {
      graph.add("dspine_gather_f32", {ids, s.plane0, output}, DSpineGatherParams{rows, head.inputSize},
                {(uint64_t{rows} * head.inputSize + 255) / 256, 1, 1});
    } else {
      Embedding::add(graph, ids, {head.outputSize, head.inputSize, StoredRows{s.plane0, s.formatId}}, output, rows);
    }
    return;
  }
  requireSegmentPlanes(s, "DSpine vocabulary head");
  const uint64_t tasks = uint64_t{rows} * (head.inputSize / 32);
  graph.add(std::string("dspine_gather_") + s.name(), {ids, s.plane0, s.plane1Slot(), s.meta, output},
            DSpineGatherParams{rows, head.inputSize}, {(tasks + 31) / 32, 1, 1}, {32, 1, 1});
}
} // namespace

void DSpine::addResidual(metal::CommandGraph &g, metal::MetalBuffer input, metal::MetalBuffer residual,
                         metal::MetalBuffer output, uint32_t width, uint32_t rows) {
  if (!width || !rows) throw std::invalid_argument("empty DSpine residual");
  const uint64_t elements = uint64_t{width} * rows;
  for (const auto *b : {&input, &residual, &output}) requireBytes(*b, elements * 2, "DSpine residual");
  g.add("dspine_residual", {input, residual, output}, DSpineParams{width, 0, 0, rows}, {(elements + 255) / 256, 1, 1});
}

void DSpine::addReadInput(metal::CommandGraph &g, metal::MetalBuffer input, metal::MetalBuffer normalized,
                          const DSpineBuffers &b, DSpineParams p) {
  requireStash(b, p);
  requireBytes(input, hiddenBytes(p), "DSpine read input");
  requireBytes(normalized, hiddenBytes(p), "DSpine read normalized");
  g.add("dspine_read_input", {input, normalized, b.raw, b.rms}, p, {p.rows, 1, 1});
}

void DSpine::addCodes(metal::CommandGraph &g, metal::MetalBuffer tokens, const DSpineWeights &w,
                      const DSpineBuffers &b, DSpineParams p) {
  requireShape(p);
  requireBytes(tokens, uint64_t{p.rows} * 4, "DSpine input token");
  requireBytes(b.codes, uint64_t{p.rows} * p.code * 2, "DSpine codes");
  g.add("dspine_codes", {b.codes, w.codebook, tokens}, p, {p.rows, 1, 1});
}

void DSpine::addMessage(metal::CommandGraph &g, const DSpineWeights &w, const DSpineBuffers &b, DSpineParams p) {
  requireShape(p);
  requireBytes(b.packed, uint64_t{p.rows} * 3 * p.rank * 2, "DSpine packed features");
  requireBytes(b.message, uint64_t{p.rows} * p.rank * 2, "DSpine message");
  requireBytes(w.gateBias, uint64_t{p.rank} * 2, "DSpine gate bias");
  g.add("dspine_message", {b.packed, w.gateBias, b.message}, p, {(uint64_t{p.rows} * p.rank + 255) / 256, 1, 1});
}

void DSpine::addWrite(metal::CommandGraph &g, metal::MetalBuffer delta, metal::MetalBuffer output,
                      const DSpineBuffers &b, DSpineParams p) {
  requireStash(b, p);
  requireBytes(delta, hiddenBytes(p), "DSpine delta");
  requireBytes(output, hiddenBytes(p), "DSpine output");
  g.add("dspine_write", {b.raw, b.rms, delta, output}, p, {(uint64_t{p.rows} * p.hidden + 255) / 256, 1, 1});
}

void DSpine::addRefinement(metal::CommandGraph &g, const DSpineWeights &w, const NormWeights &norm,
                           const Projection &head, const DSpineBuffers &b, const DraftSelectorBuffers &s,
                           DSpineParams p, std::span<const uint32_t> anchors, std::span<const SamplingPolicy> policies) {
  requireStash(b, p);
  const uint32_t lanes = p.rows / SPLASH_DRAFT_QUERY_ROWS;
  if (anchors.size() != lanes || policies.size() != lanes || w.write.empty() || norm.float32 ||
      head.inputSize != p.hidden || head.outputSize < SPLASH_DRAFT_CANDIDATES)
    throw std::invalid_argument("invalid DSpine refinement batch");
  const auto &write = w.write.back();
  if (write.inputSize != p.rank || write.outputSize != p.hidden || write.rotation ||
      write.blocks().segments.size() != 1 || !write.blocks().segments.front().isFloat() ||
      write.blocks().segments.front().formatId != GGUF_FMT_BF16)
    throw std::invalid_argument("DSpine refinement requires BF16 write weights");
  requireBytes(write.blocks().segments.front().plane0, uint64_t{p.hidden} * p.rank * 2, "DSpine final write");
  requireBytes(w.gateBias, uint64_t{p.rank} * 2, "DSpine gate bias");
  DSpineRefineParams params{p, {}, 0};
  params.selection.lanes = lanes;
  params.selection.vocabulary = head.outputSize;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (anchors[lane] >= head.outputSize) throw std::invalid_argument("DSpine anchor outside vocabulary");
    params.selection.anchor[lane] = anchors[lane];
    params.selection.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples()) params.selection.sampling_mask |= 1u << lane;
  }
  const auto workspace = DraftSelector::workspace(lanes * SPLASH_DRAFT_PROPOSAL_TOKENS);
  requireBytes(s.logits, uint64_t{p.rows} * head.outputSize * 4, "DSpine logits");
  requireBytes(s.partialIds, workspace.partialIdsBytes, "DSpine top ids");
  requireBytes(s.partialValues, workspace.partialValuesBytes, "DSpine top values");
  requireBytes(s.candidates, workspace.candidatesBytes, "DSpine candidates");
  requireBytes(s.proposedTokens, uint64_t{lanes} * SPLASH_DRAFT_PROPOSAL_TOKENS * 4, "DSpine proposals");
  requireBytes(s.proposalProbabilities, workspace.proposalProbabilitiesBytes, "DSpine probabilities");
  requireBytes(s.confidence, uint64_t{lanes} * SPLASH_DRAFT_PROPOSAL_TOKENS * 4, "DSpine confidence");
  requireBytes(s.uniforms, uint64_t{lanes} * SPLASH_SAMPLING_UNIFORMS * 4, "DSpine uniforms");
  requireBytes(b.packed, uint64_t{p.rows} * 3 * p.rank * 2, "DSpine refinement query");
  requireBytes(w.codebook, uint64_t{head.outputSize} * p.code * 2, "DSpine codebook");
  requireBytes(w.predecessorGate, uint64_t{p.rank} * p.code * 2, "DSpine predecessor gate");
  requireBytes(w.previous, uint64_t{p.rank} * p.code * 2, "DSpine predecessor value");
  requireBytes(norm.buffer, uint64_t{p.hidden} * 2, "DSpine final norm");
  requireBytes(b.refineMessage, uint64_t{lanes} * p.rank * 4, "DSpine refinement message");
  requireBytes(b.refineState, uint64_t{lanes} * p.hidden * 4, "DSpine refinement state");
  g.add("draft_select_top16_sharded", {s.logits, s.partialIds, s.partialValues}, head.outputSize,
        {uint64_t{lanes} * SPLASH_DRAFT_PROPOSAL_TOKENS * SPLASH_DRAFT_SAMPLING_SHARDS, 1, 1});
  g.add("dspine_select_candidates", {s.partialIds, s.partialValues, s.candidates},
        {uint64_t{lanes} * SPLASH_DRAFT_PROPOSAL_TOKENS, 1, 1}, {32, 1, 1});
  gatherHead(g, head, s.candidates, b.candidateHead, lanes * SPLASH_DRAFT_PROPOSAL_TOKENS * SPLASH_DRAFT_CANDIDATES);
  // Positions remain conditional on the selected predecessor. Each matrix
  // vector product spans the GPU; dispatch order carries the dependency
  // without another command submission or CPU readback.
  for (uint32_t position = 0; position < SPLASH_DRAFT_PROPOSAL_TOKENS; ++position) {
    params.position = position;
    g.add("dspine_refine_message", {b.packed, w.codebook, w.predecessorGate, w.previous,
          w.gateBias, s.proposedTokens, b.refineMessage}, params, {(p.rank + 3) / 4, lanes, 1}, {128, 1, 1});
    g.add("dspine_refine_state", {b.raw, b.rms, b.refineMessage, write.blocks().segments.front().plane0,
          b.refineState}, params, {(p.hidden + 3) / 4, lanes, 1}, {128, 1, 1});
    g.add("dspine_refine_score", {b.refineState, norm.buffer, b.candidateHead, s.candidates,
          s.uniforms, s.proposedTokens, s.proposalProbabilities, s.confidence}, params, {lanes, 1, 1});
  }
}

} // namespace splash::ops

#include "omni/Qwen3Decoder.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "ops/BufferExtent.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace splash::omni {
namespace {

constexpr uint32_t kQkvWidth = OMNI_DECODER_QKV_WIDTH;
constexpr uint32_t kAttentionWidth = OMNI_DECODER_ATTENTION_WIDTH;
constexpr uint32_t kVerifyRows = SPLASH_TARGET_VERIFY_ROWS; // the Q4 decode tiles' rows per lane

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept { return (value + multiple - 1) / multiple * multiple; }

bool isAffine(const Qwen3Projection &p) noexcept { return std::holds_alternative<ops::Projection>(p); }

ops::ProjectionShape shapeOf(const Qwen3Projection &p) {
  if (isAffine(p)) return std::get<ops::Projection>(p).shape();
  const ops::omni::DenseWeights &d = std::get<ops::omni::DenseWeights>(p);
  return {d.outputSize, d.inputSize, ops::WeightLayout::Affine64};
}

// The Q4 prefill sums of `rows` rows of `width` inputs, as
// Normalization::addRmsWithQ4Sums and Linear::addPrefillSums lay them out:
// [32-row tile][64-input group][row].
uint64_t prefillSumsBytes(uint32_t rows, uint32_t width) {
  const uint64_t groups = width / 64;
  return (uint64_t{roundUp(rows, 32)} / 32) * groups * 32 * sizeof(float);
}

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes, uint64_t &total, const char *label) {
  if (!bytes) return {};
  total += bytes;
  metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

} // namespace

Qwen3Arena::Qwen3Arena(metal::MetalBackend &backend, const ops::Linear &linear, const Qwen3Weights &weights,
                       uint32_t maxRows, uint32_t maxSequences, uint32_t maxLogitRows, bool decode, bool verify)
    : maxRows_(maxRows), maxSequences_(maxSequences), maxLogitRows_(maxLogitRows), decode_(decode || verify),
      verify_(verify) {
  decode = decode_; // a verify arena runs the decode projections
  const Qwen3Dims &d = weights.dims;
  if (!maxRows || !maxSequences || maxRows > SPLASH_PREFILL_TOKEN_BUDGET)
    throw std::invalid_argument("a decoder arena takes 1..2048 rows and at least one sequence");
  if (decode && maxRows > OMNI_MAXIMUM_LANES) throw std::invalid_argument("a decode arena takes at most 32 rows");
  if (maxLogitRows > OMNI_MAXIMUM_LANES) throw std::invalid_argument("at most 32 rows take logits per forward");
  rowCapacity = roundUp(maxRows, 64);
  logitCapacity = roundUp(std::max<uint32_t>(maxLogitRows, 1), kVerifyRows);
  tileCapacity = (maxRows + OMNI_PREFILL_TILE_ROWS - 1) / OMNI_PREFILL_TILE_ROWS + maxSequences;
  uint64_t &total = bytes_;
  hidden = allocate(backend, uint64_t{rowCapacity} * d.hidden * 2, total, "omni hidden");
  residual = allocate(backend, uint64_t{rowCapacity} * d.hidden * 2, total, "omni residual");
  normalized = allocate(backend, uint64_t{rowCapacity} * d.hidden * 2, total, "omni normalized");
  qkv = allocate(backend, uint64_t{rowCapacity} * kQkvWidth * 2, total, "omni qkv");
  attention = allocate(backend, uint64_t{rowCapacity} * kAttentionWidth * 2, total, "omni attention");
  intermediate = allocate(backend, uint64_t{rowCapacity} * d.intermediate * 2, total, "omni intermediate");
  for (TableSlot &s : slots) {
    s.rowSequence = allocate(backend, uint64_t{rowCapacity} * 4, total, "omni row sequence");
    s.sequences = allocate(backend, uint64_t{maxSequences} * sizeof(OmniSequenceParams), total, "omni sequences");
    s.tiles = allocate(backend, uint64_t{tileCapacity} * sizeof(OmniPrefillTile), total, "omni tiles");
  }
  // A visibility table has a one-row sequence and a tile per row.
  visibilitySlot.rowSequence = allocate(backend, uint64_t{rowCapacity} * 4, total, "omni visibility rows");
  visibilitySlot.sequences =
      allocate(backend, uint64_t{rowCapacity} * sizeof(OmniSequenceParams), total, "omni visibility sequences");
  visibilitySlot.tiles = allocate(backend, uint64_t{rowCapacity} * sizeof(OmniPrefillTile), total, "omni visibility tiles");
  logitRows = allocate(backend, uint64_t{logitCapacity} * 4, total, "omni logit rows");
  // The selected rows are a head's input: whole 32-row tiles for a dense
  // head of five or more rows (ops::omni::densePaddedRows).
  const uint64_t selectedRows = roundUp(logitCapacity, 32);
  selectedRaw = allocate(backend, selectedRows * d.hidden * 2, total, "omni selected raw");
  selected = allocate(backend, selectedRows * d.hidden * 2, total, "omni selected");
  if (weights.head) logits = allocate(backend, uint64_t{logitCapacity} * d.headRows * 4, total, "omni logits");
  // Attention workspace: the larger of the prefill and decode shapes the
  // arena can run.
  const ops::omni::AttentionWorkspace attentionSpace =
      decode && !verify ? ops::omni::Attention::decodeWorkspace(maxRows)
                        : ops::omni::Attention::prefillWorkspace(roundUp(maxRows, OMNI_PREFILL_TILE_ROWS), tileCapacity);
  // A prefill tile loads a whole tile of query rows whatever rows it has, so
  // a one-row tile at the last row (visibility tables) reads up to a tile
  // past the rows: one tile of the last KV head's rows more.
  queries = allocate(backend,
                     attentionSpace.queriesBytes +
                         uint64_t{OMNI_PREFILL_TILE_ROWS} * (OMNI_DECODER_QUERY_HEADS / 8) * OMNI_DECODER_HEAD_DIM * 2,
                     total, "omni queries");
  // A decode forward's first layer stores its rows' rotations for the rest.
  if (decode) rotations = allocate(backend, ops::omni::Attention::rotationBytes(rowCapacity), total, "omni rotations");
  partials = allocate(backend, attentionSpace.partialsBytes, total, "omni attention partials");
  statistics = allocate(backend, attentionSpace.statisticsBytes, total, "omni attention statistics");
  attentionCounters = allocate(backend, attentionSpace.countersBytes, total, "omni attention counters");
  // Q4 scratch: the sums of every prefill projection's input and the decode
  // plans' tables, partials and counters, over every projection shape.
  bool affine = false;
  ops::LinearScratchSize scratchSize;
  uint64_t gateScratchBytes = 0, downSumsBytes = 0;
  const auto include = [&](const Qwen3Projection &p, bool gateUp) {
    if (!isAffine(p)) return;
    affine = true;
    const ops::Projection &projection = std::get<ops::Projection>(p);
    scratchSize.include(linear.decodeScratchSize(projection.shape()));
    scratchSize.include(linear.prefillScratchSize(projection.shape()));
    for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes) {
      const ops::LinearPlan plan = linear.decodePlan(projection, lanes, gateUp ? ops::LinearEpilogue::GateUp : ops::LinearEpilogue::None, gateUp ? &projection : nullptr);
      gateScratchBytes = std::max(gateScratchBytes, plan.gateScratchBytes());
      downSumsBytes = std::max(downSumsBytes, plan.downSumsBytes());
    }
    if (!decode && gateUp) {
      const ops::LinearPlan plan = linear.prefillPlan(projection, roundUp(maxRows, 32), ops::LinearEpilogue::UpWithGate);
      gateScratchBytes = std::max(gateScratchBytes, plan.gateScratchBytes());
      downSumsBytes = std::max(downSumsBytes, plan.downSumsBytes());
    }
  };
  for (const Qwen3LayerWeights &layer : weights.layers) {
    include(layer.qkv, false);
    include(layer.output, false);
    include(layer.gate, false);
    include(layer.up, true);
    include(layer.down, false);
  }
  if (weights.head) include(*weights.head, false);
  if (affine) {
    scratch.input = allocate(backend, scratchSize.input, total, "omni linear table");
    scratch.sums = allocate(backend, scratchSize.sums, total, "omni linear table sums");
    scratch.partials = allocate(backend, scratchSize.partials, total, "omni linear partials");
    scratch.counters = allocate(backend, scratchSize.counters, total, "omni linear counters");
    sums = allocate(backend, prefillSumsBytes(rowCapacity, d.hidden), total, "omni norm sums");
    attentionSums = allocate(backend, prefillSumsBytes(rowCapacity, kAttentionWidth), total, "omni attention sums");
    // Prefill gate scratch and down sums are sized by the prefill plans; the
    // decode ones by the decode plans.
    if (!decode) {
      gateScratchBytes = std::max(gateScratchBytes, uint64_t{rowCapacity} * d.intermediate * 2);
      downSumsBytes = std::max(downSumsBytes, prefillSumsBytes(rowCapacity, d.intermediate));
    }
    gateScratch = allocate(backend, gateScratchBytes, total, "omni gate scratch");
    downSums = allocate(backend, downSumsBytes, total, "omni down sums");
  }
  // The dense projections' split scratch over every shape and row count.
  ops::omni::DenseSplitScratch dense;
  const auto includeDense = [&](const Qwen3Projection &p, uint32_t rows) {
    if (isAffine(p)) return;
    const ops::omni::DenseWeights &w = std::get<ops::omni::DenseWeights>(p);
    dense.include(ops::omni::denseSplitScratch(w.outputSize, w.inputSize, rows));
  };
  for (const Qwen3LayerWeights &layer : weights.layers) {
    includeDense(layer.qkv, maxRows);
    includeDense(layer.output, maxRows);
    includeDense(layer.up, maxRows);
    includeDense(layer.down, maxRows);
  }
  if (weights.head) includeDense(*weights.head, std::max<uint32_t>(maxLogitRows, 1));
  denseSplitPartials = allocate(backend, dense.partialsBytes, total, "omni dense split partials");
  denseSplitCounters = allocate(backend, dense.countersBytes, total, "omni dense split counters");
}

Qwen3Decoder::Qwen3Decoder(metal::MetalBackend &backend, Qwen3Weights weights)
    : weights_(std::move(weights)), linear_(backend.capabilities()) {
  const Qwen3Dims &d = weights_.dims;
  if (!d.layers || !d.hidden || !d.intermediate || weights_.layers.size() != d.layers)
    throw std::invalid_argument("decoder dimensions do not match its layers");
  if (weights_.head && (!d.headRows || !d.vocabulary || d.vocabulary > d.headRows))
    throw std::invalid_argument("decoder head rows must cover its vocabulary");
  for (const Qwen3LayerWeights &layer : weights_.layers) {
    const ops::ProjectionShape qkv = shapeOf(layer.qkv), output = shapeOf(layer.output), gate = shapeOf(layer.gate),
                               up = shapeOf(layer.up), down = shapeOf(layer.down);
    if (qkv.outputSize != kQkvWidth || qkv.inputSize != d.hidden || output.outputSize != d.hidden ||
        output.inputSize != kAttentionWidth || gate.outputSize != d.intermediate || gate.inputSize != d.hidden ||
        up.outputSize != d.intermediate || up.inputSize != d.hidden || down.outputSize != d.hidden ||
        down.inputSize != d.intermediate)
      throw std::invalid_argument("decoder layer projection shapes do not match its dimensions");
    if (isAffine(layer.gate) != isAffine(layer.up)) throw std::invalid_argument("gate and up differ in layout");
  }
  std::vector<float> inverse(OMNI_DECODER_HEAD_DIM / 2);
  for (uint32_t i = 0; i < inverse.size(); ++i)
    inverse[i] = static_cast<float>(std::pow(d.ropeTheta, -2.0 * i / OMNI_DECODER_HEAD_DIM));
  inverseFrequencies_ = backend.allocateBuffer(inverse.size() * 4, metal::BufferStorage::Shared, "omni rope");
  std::memcpy(inverseFrequencies_.contents(), inverse.data(), inverse.size() * 4);
}

bool Qwen3Decoder::affine() const noexcept {
  return !weights_.layers.empty() && isAffine(weights_.layers.front().qkv);
}

void Qwen3Decoder::uploadTables(Qwen3Arena &arena, const ops::omni::AttentionTables &tables) {
  if (tables.rows > arena.maxRows() || tables.sequences.size() > arena.maxSequences() ||
      tables.tiles.size() > arena.tileCapacity)
    throw std::invalid_argument("attention tables exceed the arena");
  arena.slot = (arena.slot + 1) % Qwen3Arena::kTableSlots;
  Qwen3Arena::TableSlot &slot = arena.slots[arena.slot];
  std::memcpy(slot.sequences.contents(), tables.sequences.data(), tables.sequences.size() * sizeof(OmniSequenceParams));
  std::memcpy(slot.tiles.contents(), tables.tiles.data(), tables.tiles.size() * sizeof(OmniPrefillTile));
  auto *rowSequence = static_cast<uint32_t *>(slot.rowSequence.contents());
  for (uint32_t s = 0; s < tables.sequences.size(); ++s) {
    const OmniSequenceParams &sequence = tables.sequences[s];
    for (uint32_t r = 0; r < sequence.rows; ++r) rowSequence[sequence.row_offset + r] = s;
  }
}

void Qwen3Decoder::uploadVisibility(Qwen3Arena &arena, const ops::omni::AttentionTables &tables) {
  if (tables.rows > arena.maxRows() || tables.sequences.size() > arena.rowCapacity ||
      tables.tiles.size() > arena.rowCapacity || tables.tiles.size() > arena.tileCapacity)
    throw std::invalid_argument("visibility tables exceed the arena");
  Qwen3Arena::TableSlot &slot = arena.visibilitySlot;
  std::memcpy(slot.sequences.contents(), tables.sequences.data(), tables.sequences.size() * sizeof(OmniSequenceParams));
  std::memcpy(slot.tiles.contents(), tables.tiles.data(), tables.tiles.size() * sizeof(OmniPrefillTile));
  auto *rowSequence = static_cast<uint32_t *>(slot.rowSequence.contents());
  for (uint32_t s = 0; s < tables.sequences.size(); ++s) {
    const OmniSequenceParams &sequence = tables.sequences[s];
    for (uint32_t r = 0; r < sequence.rows; ++r) rowSequence[sequence.row_offset + r] = s;
  }
}

void Qwen3Decoder::addProjection(metal::CommandGraph &graph, Qwen3Arena &arena, const Qwen3Projection &projection,
                                 metal::MetalBuffer input, metal::MetalBuffer inputSums, metal::MetalBuffer output,
                                 metal::MetalBuffer residual, uint32_t rows, ops::PreparedInput prepared,
                                 ops::FloatOutput destination) const {
  if (isAffine(projection)) {
    const ops::Projection &p = std::get<ops::Projection>(projection);
    if (arena.decode()) {
      const uint32_t lanes = roundUp(rows, kVerifyRows) / kVerifyRows;
      ops::Projection typed = p;
      typed.destination = destination;
      const ops::LinearPlan plan = linear_.decodePlan(
          typed, lanes, residual ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None, nullptr, rows);
      linear_.add(graph, {.input = input, .output = output, .residual = residual, .scratch = arena.scratch, .prepared = prepared},
                  typed, plan);
      return;
    }
    if (destination == ops::FloatOutput::Float32) throw std::logic_error("prefill projections write bf16");
    if (residual)
      linear_.addPrefillResidual(graph, input, p, residual, output, inputSums, rows, arena.scratch);
    else
      linear_.addPrefill(graph, input, p, output, inputSums, rows, arena.scratch);
    return;
  }
  const ops::omni::DenseWeights &d = std::get<ops::omni::DenseWeights>(projection);
  ops::omni::DenseLinear::add(graph, d,
                              {.input = input, .output = output, .residual = residual,
                               .partials = arena.denseSplitPartials, .counters = arena.denseSplitCounters},
                              rows, {.activation = ops::omni::Activation::None, .residual = static_cast<bool>(residual),
                                     .destination = destination});
}

ops::omni::AttentionBuffers Qwen3Decoder::attentionBuffers(const Qwen3Arena &arena, const KvBinding &kv,
                                                           uint32_t layer) const {
  const Qwen3LayerWeights &w = weights_.layers[layer];
  ops::omni::AttentionBuffers attention;
  attention.qkv = arena.qkv;
  attention.queryNorm = w.queryNorm;
  attention.keyNorm = w.keyNorm;
  attention.inverseFrequencies = inverseFrequencies_;
  attention.rowSequence = arena.currentSlot().rowSequence;
  attention.sequences = arena.currentSlot().sequences;
  attention.tiles = arena.currentSlot().tiles;
  attention.pageTable = kv.pageTable;
  attention.pages = kv.pages;
  attention.queries = arena.queries;
  attention.partials = arena.partials;
  attention.statistics = arena.statistics;
  attention.output = arena.attention;
  attention.rotations = arena.rotations;
  attention.counters = arena.attentionCounters;
  return attention;
}

// A decode step's attention through the fused decode kernel, or the long
// kernel for a lone row over a long history, and a verify forward's through
// the fused verify kernel when its sequences fit it, else prepare and the
// prefill kernels, whose sequences take any rows.
void Qwen3Decoder::addAttention(metal::CommandGraph &graph, const Qwen3Arena &arena,
                                const ops::omni::AttentionBuffers &attention, const ops::omni::AttentionTables &tables,
                                uint32_t layer, const KvBinding &kv, const ops::omni::AttentionTables *visibility) const {
  const float epsilon = weights_.dims.rmsEpsilon;
  if (visibility) {
    // The rows' keys and values stored and their queries rotated at their
    // positions, then attention over what each row sees.
    ops::omni::Attention::addPrepare(graph, attention, tables, layer, epsilon, kv.pageTableEntries, kv.pageCount);
    ops::omni::AttentionBuffers seen = attention;
    seen.rowSequence = arena.visibilitySlot.rowSequence;
    seen.sequences = arena.visibilitySlot.sequences;
    seen.tiles = arena.visibilitySlot.tiles;
    ops::omni::Attention::addPrefill(graph, seen, *visibility, layer, kv.pageTableEntries, kv.pageCount);
    return;
  }
  const bool decode = arena.decode() && !arena.verify();
  // Layer 0 stores the forward's rotation table, the later layers load it.
  const ops::omni::Rotations rotations = layer ? ops::omni::Rotations::Load : ops::omni::Rotations::Store;
  if (decode && ops::omni::Attention::fusable(tables)) {
    ops::omni::Attention::addDecodeFused(graph, attention, tables, layer, epsilon, kv.pageTableEntries, kv.pageCount,
                                         rotations);
    return;
  }
  if (decode) {
    // Not fusable: a lone row past the fused kernel's pages (addDecodeLong
    // refuses anything else).
    ops::omni::Attention::addDecodeLong(graph, attention, tables, layer, epsilon, kv.pageTableEntries, kv.pageCount,
                                        rotations);
    return;
  }
  if (arena.verify() && ops::omni::Attention::verifyFusable(tables)) {
    ops::omni::Attention::addVerifyFused(graph, attention, tables, layer, epsilon, kv.pageTableEntries, kv.pageCount,
                                         rotations);
    return;
  }
  ops::omni::Attention::addPrepare(graph, attention, tables, layer, epsilon, kv.pageTableEntries, kv.pageCount);
  ops::omni::Attention::addPrefill(graph, attention, tables, layer, kv.pageTableEntries, kv.pageCount);
}

void Qwen3Decoder::addLayer(metal::CommandGraph &graph, Qwen3Arena &arena, const KvBinding &kv,
                            const ops::omni::AttentionTables &tables, uint32_t layer, uint32_t rows,
                            ops::AneFfn *split, bool qkvReady, const ops::omni::AttentionTables *visibility) const {
  const Qwen3LayerWeights &w = weights_.layers[layer];
  const Qwen3Dims &d = weights_.dims;
  const bool affineLayer = isAffine(w.qkv);
  const bool decode = arena.decode();
  // Decode rows run the Q4 tiles in whole lanes of 8 rows; the padded rows
  // carry whatever the buffers hold and are never read back.
  const uint32_t tileRows = decode && affineLayer ? roundUp(rows, kVerifyRows) : rows;
  const uint32_t lanes = roundUp(rows, kVerifyRows) / kVerifyRows;

  // A decode step of few live rows folds each layer's two norms into the
  // one-lane projection that reads it, which normalizes the live rows itself
  // with the norm dispatch's bits (ops::LinearPlan::foldsNorm). The final
  // norm stays a dispatch unless the caller's head folds it
  // (ForwardStep::rawSelected).
  const auto folds = [&](const ops::LinearPlan &plan) { return plan.foldsNorm(rows); };

  // 1. Input norm and the fused QKV projection, unless the caller gathered
  // the rows' (ForwardStep::firstQkvReady).
  ops::PreparedInput prepared;
  bool projected = qkvReady;
  if (qkvReady) {
  } else if (affineLayer && !decode) {
    ops::Normalization::addRmsWithQ4Sums(graph, arena.hidden, w.inputNorm, arena.normalized, arena.sums, d.hidden, rows);
  } else if (affineLayer) {
    const ops::Projection &qkv = std::get<ops::Projection>(w.qkv);
    const ops::LinearPlan plan = linear_.decodePlan(qkv, lanes, ops::LinearEpilogue::None, nullptr, rows);
    if (folds(plan)) {
      linear_.add(graph,
                  {.input = arena.hidden, .output = arena.qkv, .scratch = arena.scratch,
                   .normWeight = w.inputNorm.buffer, .normRows = rows},
                  qkv, plan);
      projected = true;
    } else {
      prepared = ops::Normalization::addRms(graph, arena.hidden, w.inputNorm, arena.normalized, d.hidden, tileRows,
                                            arena.scratch, plan.input());
    }
  } else {
    ops::Normalization::addRms(graph, arena.hidden, w.inputNorm, arena.normalized, d.hidden, rows);
  }
  if (!projected) addProjection(graph, arena, w.qkv, arena.normalized, arena.sums, arena.qkv, {}, rows, prepared);

  // 2. Attention over the lanes' pages.
  addAttention(graph, arena, attentionBuffers(arena, kv, layer), tables, layer, kv, visibility);

  // 3. Output projection with the residual: residual = hidden + O(attention).
  if (affineLayer && !decode)
    linear_.addPrefillSums(graph, arena.attention, arena.attentionSums, std::get<ops::Projection>(w.output), rows);
  addProjection(graph, arena, w.output, arena.attention, arena.attentionSums, arena.residual, arena.hidden, rows, {});

  // 4. Post-attention norm and the SwiGLU FFN: hidden = residual + down(silu(gate n) * up n).
  if (affineLayer && !decode) {
    ops::Normalization::addRmsWithQ4Sums(graph, arena.residual, w.postAttentionNorm, arena.normalized, arena.sums,
                                         d.hidden, rows);
    const ops::SwiGluProjections ffn{&std::get<ops::Projection>(w.gate), &std::get<ops::Projection>(w.up),
                                     &std::get<ops::Projection>(w.down)};
    const ops::PrefillFfnBuffers buffers{arena.normalized,   arena.sums,     arena.gateScratch,
                                         arena.intermediate, arena.downSums, arena.scratch};
    if (split)
      split->add(graph, layer, buffers, arena.residual, arena.hidden, rows);
    else
      linear_.addPrefillSwiGlu(graph, ffn, buffers, arena.residual, arena.hidden, rows);
  } else if (affineLayer) {
    const ops::Projection &gate = std::get<ops::Projection>(w.gate), &up = std::get<ops::Projection>(w.up),
                          &down = std::get<ops::Projection>(w.down);
    const ops::LinearPlan gateUpPlan = linear_.decodePlan(up, lanes, ops::LinearEpilogue::GateUp, &gate, rows);
    if (folds(gateUpPlan)) {
      linear_.add(graph,
                  {.input = arena.residual, .output = arena.intermediate, .gateScratch = arena.gateScratch,
                   .scratch = arena.scratch, .normWeight = w.postAttentionNorm.buffer, .normRows = rows},
                  up, gateUpPlan, &gate);
    } else {
      const ops::PreparedInput normalized = ops::Normalization::addRms(
          graph, arena.residual, w.postAttentionNorm, arena.normalized, d.hidden, tileRows, arena.scratch,
          gateUpPlan.input());
      linear_.add(graph,
                  {.input = arena.normalized, .output = arena.intermediate, .gateScratch = arena.gateScratch,
                   .scratch = arena.scratch, .prepared = normalized},
                  up, gateUpPlan, &gate);
    }
    linear_.add(graph,
                {.input = arena.intermediate, .output = arena.hidden, .residual = arena.residual, .scratch = arena.scratch},
                down, linear_.decodePlan(down, lanes, ops::LinearEpilogue::Residual, nullptr, rows));
  } else {
    ops::Normalization::addRms(graph, arena.residual, w.postAttentionNorm, arena.normalized, d.hidden, rows);
    ops::omni::DenseLinear::addGateUp(graph, std::get<ops::omni::DenseWeights>(w.gate),
                                      std::get<ops::omni::DenseWeights>(w.up),
                                      {.input = arena.normalized, .output = arena.intermediate,
                                       .partials = arena.denseSplitPartials, .counters = arena.denseSplitCounters},
                                      rows);
    ops::omni::DenseLinear::add(graph, std::get<ops::omni::DenseWeights>(w.down),
                                {.input = arena.intermediate, .output = arena.hidden, .residual = arena.residual,
                                 .partials = arena.denseSplitPartials, .counters = arena.denseSplitCounters},
                                rows, {.residual = true});
  }
}

double Qwen3Decoder::enableAneFfn(metal::MetalBackend &backend, const AneFfnPolicy &policy, Qwen3Arena &prefill) {
  if (!affine()) throw std::invalid_argument("the Neural Engine split takes affine Q4 decoders");
  if (prefill.decode() || prefill.maxRows() < ops::AneFfn::kMaximumRows)
    throw std::invalid_argument("the Neural Engine split checks itself with a full prefill arena");
  std::vector<ops::SwiGluProjections> layers;
  for (const Qwen3LayerWeights &layer : weights_.layers)
    layers.push_back({&std::get<ops::Projection>(layer.gate), &std::get<ops::Projection>(layer.up),
                      &std::get<ops::Projection>(layer.down)});
  if (const char *reason = ops::AneFfn::unsupported(layers)) throw std::invalid_argument(reason);
  const ops::PrefillFfnBuffers buffers{prefill.normalized,   prefill.sums,     prefill.gateScratch,
                                       prefill.intermediate, prefill.downSums, prefill.scratch};
  aneFfn_.reset();
  auto split = std::make_unique<ops::AneFfn>(backend, layers, policy.units, policy.functionRows);
  const double error = split->verify(layers, buffers, {prefill.hidden, prefill.residual});
  aneFfn_ = std::move(split);
  return error;
}

ops::AneFfn *Qwen3Decoder::aneFfnFor(uint32_t rows) const noexcept {
  return aneFfn_ && aneFfn_->splits(rows) ? aneFfn_.get() : nullptr;
}

void Qwen3Decoder::addHead(metal::CommandGraph &graph, Qwen3Arena &arena, uint32_t rows, uint32_t logitRows,
                           bool rawSelected) const {
  const Qwen3Dims &d = weights_.dims;
  // The selected rows, then their final norm and the head as a decode step
  // (at most 32 rows): fp32 logits. A decode step whose logit rows are
  // 0..rows-1 reads its hidden rows directly.
  const auto *selectedRows = static_cast<const uint32_t *>(arena.logitRows.contents());
  bool identity = arena.decode() && logitRows == rows;
  for (uint32_t r = 0; identity && r < logitRows; ++r) identity = selectedRows[r] == r;
  const metal::MetalBuffer source = identity ? arena.hidden : arena.selectedRaw;
  if (!identity)
    ops::omni::Embedding::addGatherRows(graph, arena.logitRows, arena.hidden, arena.selectedRaw, logitRows, rows, d.hidden);
  const uint32_t lanes = roundUp(logitRows, kVerifyRows) / kVerifyRows;
  const uint32_t tileRows = lanes * kVerifyRows;
  if (!weights_.head) {
    // The caller's head folds the final norm (TtsModel::addPredictorHead),
    // or reads the rows the norm writes here.
    if (rawSelected) return;
    ops::Normalization::addRms(graph, source, weights_.finalNorm, arena.selected, d.hidden, logitRows);
    return;
  }
  const Qwen3Projection &head = *weights_.head;
  if (isAffine(head)) {
    ops::Projection typed = std::get<ops::Projection>(head);
    typed.destination = ops::FloatOutput::Float32;
    const ops::LinearPlan plan = linear_.decodePlan(typed, lanes, ops::LinearEpilogue::None, nullptr, logitRows);
    const ops::PreparedInput normalized = ops::Normalization::addRms(
        graph, source, weights_.finalNorm, arena.selected, d.hidden, tileRows, arena.scratch, plan.input());
    linear_.add(graph, {.input = arena.selected, .output = arena.logits, .scratch = arena.scratch, .prepared = normalized},
                typed, plan);
    return;
  }
  ops::Normalization::addRms(graph, source, weights_.finalNorm, arena.selected, d.hidden, logitRows);
  ops::omni::DenseLinear::add(graph, std::get<ops::omni::DenseWeights>(head),
                              {.input = arena.selected, .output = arena.logits, .partials = arena.denseSplitPartials,
                               .counters = arena.denseSplitCounters},
                              logitRows, {.destination = ops::FloatOutput::Float32});
}

void Qwen3Decoder::addForward(metal::CommandGraph &graph, Qwen3Arena &arena, const KvBinding &kv,
                              const ForwardStep &step) const {
  if (!step.tables) throw std::invalid_argument("a forward takes attention tables");
  const ops::omni::AttentionTables &tables = *step.tables;
  const uint32_t rows = tables.rows;
  if (!rows || rows > arena.maxRows() || tables.sequences.size() > arena.maxSequences())
    throw std::invalid_argument("forward rows exceed the arena");
  if (arena.decode() && !arena.verify()) {
    // A decode step is one row per sequence, or two for every sequence
    // (the fused attention's short form; the split kernels take one).
    const uint32_t perSequence = tables.sequences.front().rows;
    if (perSequence < 1 || perSequence > 2 || rows != tables.sequences.size() * perSequence)
      throw std::invalid_argument("a decode step is one or two rows per sequence");
    if (perSequence == 2 && !ops::omni::Attention::fusable(tables))
      throw std::invalid_argument("a two-row decode step takes the fused attention's page limit");
  }
  if (step.logitRows > arena.maxLogitRows()) throw std::invalid_argument("forward selects too many logit rows");
  if (step.aneFfn && (arena.decode() || !affine() || !step.aneFfn->splits(rows)))
    throw std::invalid_argument("the Neural Engine split takes affine prefill chunks of its rows");
  if (step.firstQkvReady && (!arena.decode() || !sameFirstQkv(rows, rows)))
    throw std::invalid_argument("gathered QKV rows take a decode step of the standard layers");
  if (step.rawSelected && weights_.head) throw std::invalid_argument("raw selected rows are for a decoder without a head");
  if (step.visibility && (arena.decode() || step.visibility->rows != rows || step.visibility->paddedRows != tables.paddedRows))
    throw std::invalid_argument("visibility tables take a prefill-path forward of the same rows");
  const HiddenCapture *capture = step.capture;
  if (capture) {
    const uint64_t width = weights_.dims.hidden, slots = capture->layers.size();
    if (rows > capture->rowCapacity || capture->slots.size() != slots)
      throw std::invalid_argument("the hidden capture does not hold the forward's rows");
    for (const metal::MetalBuffer &slot : capture->slots)
      if (slot.sizeBytes() < (uint64_t{rows - 1} * slots + 1) * width * 2)
        throw std::invalid_argument("a hidden capture slot does not hold the forward's rows");
  }
  for (uint32_t layer = 0; layer < weights_.dims.layers; ++layer) {
    addLayer(graph, arena, kv, tables, layer, rows, step.aneFfn, layer == 0 && step.firstQkvReady, step.visibility);
    if (!capture) continue;
    for (size_t j = 0; j < capture->layers.size(); ++j) {
      if (capture->layers[j] != layer) continue;
      const uint64_t width = weights_.dims.hidden, slots = capture->layers.size();
      ops::omni::Embedding::addGatherRows(graph, capture->rowIndex, arena.hidden, capture->slots[j], rows, rows,
                                          static_cast<uint32_t>(width), static_cast<uint32_t>(slots));
    }
  }
  if (step.logitRows) addHead(graph, arena, rows, step.logitRows, step.rawSelected);
}

void Qwen3Decoder::addKeysValues(metal::CommandGraph &graph, Qwen3Arena &arena, const KvBinding &kv,
                                 const ops::omni::AttentionTables &tables) const {
  const uint32_t rows = tables.rows;
  if (arena.decode() || !rows || rows > arena.maxRows() || tables.sequences.size() > arena.maxSequences())
    throw std::invalid_argument("context keys and values take a prefill-path arena of their rows");
  for (uint32_t layer = 0; layer < weights_.dims.layers; ++layer) {
    const Qwen3LayerWeights &w = weights_.layers[layer];
    // The fused QKV of the rows as they are; the queries are discarded.
    if (isAffine(w.qkv)) linear_.addPrefillSums(graph, arena.hidden, arena.sums, std::get<ops::Projection>(w.qkv), rows);
    addProjection(graph, arena, w.qkv, arena.hidden, arena.sums, arena.qkv, {}, rows, {});
    ops::omni::Attention::addPrepare(graph, attentionBuffers(arena, kv, layer), tables, layer, weights_.dims.rmsEpsilon,
                                     kv.pageTableEntries, kv.pageCount);
  }
}

void Qwen3Decoder::addFirstQkv(metal::CommandGraph &graph, Qwen3Arena &arena, metal::MetalBuffer input,
                               metal::MetalBuffer qkv, uint32_t rows) const {
  if (!arena.decode() || !rows || rows > arena.maxRows() || !sameFirstQkv(rows, rows))
    throw std::invalid_argument("a first QKV takes a decode arena of the standard layers");
  const Qwen3LayerWeights &w = weights_.layers.front();
  const Qwen3Dims &d = weights_.dims;
  // addLayer's step 1 with the norm as its own dispatch (a folded norm
  // gives the same bits).
  ops::PreparedInput prepared;
  if (isAffine(w.qkv)) {
    const uint32_t lanes = roundUp(rows, kVerifyRows) / kVerifyRows;
    const ops::LinearPlan plan =
        linear_.decodePlan(std::get<ops::Projection>(w.qkv), lanes, ops::LinearEpilogue::None, nullptr, rows);
    prepared = ops::Normalization::addRms(graph, input, w.inputNorm, arena.normalized, d.hidden, lanes * kVerifyRows,
                                          arena.scratch, plan.input());
  } else {
    ops::Normalization::addRms(graph, input, w.inputNorm, arena.normalized, d.hidden, rows);
  }
  addProjection(graph, arena, w.qkv, arena.normalized, arena.sums, std::move(qkv), {}, rows, prepared);
}

bool Qwen3Decoder::firstQkvRowwise(uint32_t rows) const {
  const Qwen3Projection &qkv = weights_.layers.front().qkv;
  if (!isAffine(qkv)) return rows == 1;
  return linear_.decodePlan(std::get<ops::Projection>(qkv), 1, ops::LinearEpilogue::None, nullptr, rows)
             .configuration()
             .tile == ops::LinearTile::Qmv;
}

bool Qwen3Decoder::sameFirstQkv(uint32_t rows, uint32_t otherRows) const {
  const Qwen3Projection &qkv = weights_.layers.front().qkv;
  if (!isAffine(qkv))
    return ops::omni::DenseLinear::sameRowArithmetic(std::get<ops::omni::DenseWeights>(qkv), rows, otherRows);
  // A one-lane plan computes each row alike at every row count it runs: the
  // tiles at three to eight rows, Qmv at one or two.
  const auto lanes = [](uint32_t n) { return roundUp(n, kVerifyRows) / kVerifyRows; };
  const auto config = [&](uint32_t n) {
    return linear_.decodePlan(std::get<ops::Projection>(qkv), lanes(n), ops::LinearEpilogue::None, nullptr, n)
        .configuration();
  };
  return lanes(rows) == lanes(otherRows) && config(rows) == config(otherRows);
}

} // namespace splash::omni

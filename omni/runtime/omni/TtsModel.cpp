#include "omni/TtsModel.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <string_view>
#include <stdexcept>

namespace splash::omni {
namespace {

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept { return (value + multiple - 1) / multiple * multiple; }

metal::MetalBuffer zeroed(metal::MetalBackend &backend, uint64_t bytes, std::string_view label) {
  metal::MetalBuffer buffer = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

} // namespace

BiasedProjection::BiasedProjection(metal::MetalBackend &backend, const ops::Linear &linear, Qwen3Projection projection,
                                   metal::MetalBuffer bias, uint32_t maxRows, std::string_view label)
    : linear_(&linear), projection_(std::move(projection)), bias_(std::move(bias)), maxRows_(maxRows) {
  if (!maxRows || maxRows > SPLASH_PREFILL_TOKEN_BUDGET) throw std::invalid_argument("a biased projection takes 1..2048 rows");
  const uint32_t capacity = roundUp(maxRows, 64);
  if (std::holds_alternative<ops::Projection>(projection_)) {
    const ops::Projection &p = std::get<ops::Projection>(projection_);
    outputSize_ = p.outputSize;
    inputSize_ = p.inputSize;
    sums_ = zeroed(backend, (uint64_t{capacity} / 32) * (inputSize_ / 64) * 32 * 4, label);
    raw_ = zeroed(backend, uint64_t{capacity} * outputSize_ * 2, label);
    // The bias as 32 bf16 rows, the residual of the decode plans that run
    // short inputs in one dispatch.
    biasRows_ = zeroed(backend, uint64_t{32} * outputSize_ * 2, label);
    ops::LinearScratchSize size = linear.prefillScratchSize(p.shape());
    size.include(linear.decodeScratchSize(p.shape()));
    scratch_.input = zeroed(backend, size.input ? size.input : 16, label);
    scratch_.sums = zeroed(backend, size.sums ? size.sums : 16, label);
    scratch_.partials = zeroed(backend, size.partials ? size.partials : 16, label);
    scratch_.counters = zeroed(backend, size.counters ? size.counters : 16, label);
  } else {
    const ops::omni::DenseWeights &d = std::get<ops::omni::DenseWeights>(projection_);
    outputSize_ = d.outputSize;
    inputSize_ = d.inputSize;
  }
  ops::requireBytes(bias_, uint64_t{outputSize_} * 4, "projection bias");
  if (biasRows_) {
    const auto *values = static_cast<const float *>(bias_.contents());
    auto *rows = static_cast<uint16_t *>(biasRows_.contents());
    for (uint32_t r = 0; r < 32; ++r)
      for (uint32_t n = 0; n < outputSize_; ++n) {
        const uint32_t bits = std::bit_cast<uint32_t>(values[n]);
        // Round to nearest even bf16.
        const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
        rows[r * outputSize_ + n] = static_cast<uint16_t>(rounded >> 16);
      }
  }
}

void BiasedProjection::add(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer output,
                           uint32_t rows, ops::omni::Activation activation) const {
  if (!rows || rows > maxRows_) throw std::invalid_argument("biased projection rows exceed its capacity");
  if (std::holds_alternative<ops::Projection>(projection_)) {
    const ops::Projection &p = std::get<ops::Projection>(projection_);
    if (rows <= 32 && activation == ops::omni::Activation::None) {
      // One decode-tile dispatch: the bias rows are its residual.
      const uint32_t lanes = roundUp(rows, 8) / 8;
      const ops::LinearPlan plan = linear_->decodePlan(p, lanes, ops::LinearEpilogue::Residual, nullptr, rows);
      linear_->add(graph, {.input = std::move(input), .output = std::move(output), .residual = biasRows_, .scratch = scratch_},
                   p, plan);
      return;
    }
    linear_->addPrefillSums(graph, input, sums_, p, rows);
    linear_->addPrefill(graph, input, p, raw_, sums_, rows, scratch_);
    ops::omni::Norms::addBiasActivate(graph, raw_, bias_, std::move(output), rows, outputSize_, activation);
    return;
  }
  ops::omni::DenseWeights d = std::get<ops::omni::DenseWeights>(projection_);
  d.bias = bias_;
  ops::omni::DenseLinear::add(graph, d, {.input = std::move(input), .output = std::move(output)}, rows,
                              {.activation = activation});
}

TtsModel::TtsModel(metal::MetalBackend &backend, TtsWeights weights, const TtsConfig &config, uint32_t maxTextRows,
                   uint32_t maxLanes)
    : backend_(&backend), config_(config), weights_(std::move(weights)), maxTextRows_(maxTextRows),
      maxLanes_(maxLanes) {
  if (!maxLanes || maxLanes > OMNI_MAXIMUM_LANES / 2) throw std::invalid_argument("TTS takes 1..16 lanes");
  talker_ = std::make_unique<Qwen3Decoder>(backend, weights_.talker);
  predictor_ = std::make_unique<Qwen3Decoder>(backend, weights_.predictor);
  tables_ = {weights_.codecEmbedding, weights_.predictorEmbeddings, config.codecVocabulary,
             config.predictorVocabulary, config.talker.hidden};
  fc1_ = std::make_unique<BiasedProjection>(backend, talker_->linear(), weights_.textFc1, weights_.textFc1Bias,
                                            maxTextRows, "tts fc1");
  fc2_ = std::make_unique<BiasedProjection>(backend, talker_->linear(), weights_.textFc2, weights_.textFc2Bias,
                                            maxTextRows, "tts fc2");
  if (config.predictorProjection())
    smallToMtp_ = std::make_unique<BiasedProjection>(backend, predictor_->linear(), weights_.smallToMtp,
                                                     weights_.smallToMtpBias, 2 * maxLanes, "tts small_to_mtp");
  const uint32_t capacity = roundUp(maxTextRows, 64);
  textEmbedded_ = zeroed(backend, uint64_t{maxLanes} * capacity * config.textHidden * 2, "tts text embedded");
  fc1Out_ = zeroed(backend, uint64_t{capacity} * config.textHidden * 2, "tts fc1 out");
  // Sub-steps 1..14 feed the predictor small_to_mtp(table[step - 1][code]):
  // a fixed function of the code, so project every table row once here,
  // through the same decode dispatch a step of up to eight rows runs (its
  // rows sum independently), leaving the steps one gather each.
  const uint32_t vocabulary = config.predictorVocabulary, width = config.talker.hidden;
  const uint32_t outputWidth = config.predictor.hidden, tables = config.codeGroups - 2;
  // A power of two, so whole chunks tile the vocabulary at any lane count
  // (three lanes would otherwise give six-row chunks).
  const uint32_t chunk = std::bit_floor(std::min(8u, 2 * maxLanes));
  if (vocabulary % chunk) throw std::invalid_argument("the predictor vocabulary is not whole projection chunks");
  const uint64_t inputRow = uint64_t{width} * 2, outputRow = uint64_t{outputWidth} * 2;
  if (smallToMtp_) {
    // Eight rows of slack: a decode dispatch writes whole eight-row tiles.
    projectedTables_ = zeroed(backend, (uint64_t{tables} * vocabulary + 8) * outputWidth * 2, "tts projected tables");
    metal::CommandGraph graph;
    for (uint64_t row = 0; row < uint64_t{tables} * vocabulary; row += chunk)
      smallToMtp_->add(graph,
                       backend.view(weights_.predictorEmbeddings, row * inputRow,
                                    weights_.predictorEmbeddings.sizeBytes() - row * inputRow),
                       backend.view(projectedTables_, row * outputRow, projectedTables_.sizeBytes() - row * outputRow),
                       chunk, ops::omni::Activation::None);
    (void)backend.submitCommandAsync(graph.command()).wait();
  } else {
    // Without the projection the tables are the predictor's input as
    // stored: 15 of them, so the 14 read keep a table of slack.
    projectedTables_ = weights_.predictorEmbeddings;
  }
  // Sub-steps 1..14 then start the predictor at layer 0's attention: its
  // input norm and QKV projection are a fixed function of the code too.
  // The rows go through the dispatches a decode step of qkvTableRows_
  // rows runs, which compute each row alike for every row count that
  // sameFirstQkv matches: three to eight rows on the affine layers, two to
  // sixteen on the dense MMA tiles. A step of one or two rows on the Qmv
  // tile, or of one on the dense GEMV, takes the tables as well.
  {
    const uint32_t qkvWidth = OMNI_DECODER_QKV_WIDTH;
    qkvTableRows_ = predictor_->affine() || maxLanes > 4 ? 8 : 4;
    if (vocabulary % qkvTableRows_) throw std::invalid_argument("the predictor vocabulary is not whole QKV chunks");
    qkvTables_ = zeroed(backend, uint64_t{tables} * vocabulary * qkvWidth * 2, "tts predictor qkv tables");
    Qwen3Arena arena(backend, predictor_->linear(), predictor_->weights(), 8, 1, 1, true);
    metal::CommandGraph qkvGraph;
    const uint64_t qkvRow = uint64_t{qkvWidth} * 2;
    for (uint64_t row = 0; row < uint64_t{tables} * vocabulary; row += qkvTableRows_)
      predictor_->addFirstQkv(qkvGraph, arena,
                              backend.view(projectedTables_, row * outputRow, projectedTables_.sizeBytes() - row * outputRow),
                              backend.view(qkvTables_, row * qkvRow, qkvTables_.sizeBytes() - row * qkvRow), qkvTableRows_);
    (void)backend.submitCommandAsync(qkvGraph.command()).wait();
  }
  // The predictor heads run as decode plans over the arena's selected rows.
  if (std::holds_alternative<ops::Projection>(weights_.predictorHeads.front())) {
    ops::LinearScratchSize size;
    for (const Qwen3Projection &head : weights_.predictorHeads)
      size.include(predictor_->linear().decodeScratchSize(std::get<ops::Projection>(head).shape()));
    headScratch_.input = zeroed(backend, size.input ? size.input : 16, "tts head table");
    headScratch_.sums = zeroed(backend, size.sums ? size.sums : 16, "tts head sums");
    headScratch_.partials = zeroed(backend, size.partials ? size.partials : 16, "tts head partials");
    headScratch_.counters = zeroed(backend, size.counters ? size.counters : 16, "tts head counters");
  }
}

void TtsModel::addTextProjection(metal::CommandGraph &graph, uint32_t lane, std::span<const uint32_t> tokens,
                                 metal::MetalBuffer output) const {
  const auto rows = static_cast<uint32_t>(tokens.size());
  if (!rows || rows > maxTextRows_) throw std::invalid_argument("text projection rows exceed the model's capacity");
  if (lane >= maxLanes_) throw std::invalid_argument("text projection lane out of range");
  const uint64_t region = uint64_t{roundUp(maxTextRows_, 64)} * config_.textHidden * 2;
  weights_.textEmbedding.copyRows(tokens, static_cast<uint8_t *>(textEmbedded_.contents()) + lane * region);
  const metal::MetalBuffer embedded = backend_->view(textEmbedded_, lane * region, region);
  fc1_->add(graph, embedded, fc1Out_, rows, ops::omni::Activation::Silu);
  fc2_->add(graph, fc1Out_, std::move(output), rows, ops::omni::Activation::None);
}

void TtsModel::addPredictorInput(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer output,
                                 uint32_t rows) const {
  if (smallToMtp_) smallToMtp_->add(graph, std::move(input), std::move(output), rows, ops::omni::Activation::None);
  else
    ops::omni::Norms::addActivate(graph, std::move(input), std::move(output), uint64_t{rows} * config_.predictor.hidden,
                                  ops::omni::Activation::None, false);
}

bool TtsModel::addPredictorCodeInput(metal::CommandGraph &graph, metal::MetalBuffer codes, uint32_t step,
                                     metal::MetalBuffer output, metal::MetalBuffer qkv, uint32_t rows) const {
  if (!step || step + 1 >= config_.codeGroups) throw std::invalid_argument("predictor sub-step out of range");
  const uint32_t vocabulary = config_.predictorVocabulary;
  const uint64_t bytes = uint64_t{vocabulary} * config_.predictor.hidden * 2;
  ops::omni::Embedding::addGather(graph, codes, backend_->view(projectedTables_, (step - 1) * bytes, bytes),
                                  std::move(output), rows, vocabulary, config_.predictor.hidden, 1, OMNI_CODE_GROUPS);
  // A step of one or two rows projects on the Qmv tile, whose QKV differs
  // from the tables' only by the association of the fp32 sums; it takes the
  // tables too (one lane's step 7.33 -> 7.18 ms).
  if (!(predictor_->sameFirstQkv(rows, qkvTableRows_) || predictor_->firstQkvRowwise(rows))) return false;
  const uint64_t qkvBytes = uint64_t{vocabulary} * OMNI_DECODER_QKV_WIDTH * 2;
  ops::omni::Embedding::addGather(graph, std::move(codes), backend_->view(qkvTables_, (step - 1) * qkvBytes, qkvBytes),
                                  std::move(qkv), rows, vocabulary, OMNI_DECODER_QKV_WIDTH, 1, OMNI_CODE_GROUPS);
  return true;
}

bool TtsModel::predictorHeadFoldsNorm(const Qwen3Arena &arena, uint32_t rows) const {
  // The one-lane plans of the heads (alike in shape) at few live rows.
  const Qwen3Projection &head = weights_.predictorHeads.front();
  if (!std::holds_alternative<ops::Projection>(head) || !arena.decode()) return false;
  ops::Projection typed = std::get<ops::Projection>(head);
  typed.destination = ops::FloatOutput::Float32;
  return predictor_->linear()
      .decodePlan(typed, roundUp(rows, SPLASH_TARGET_VERIFY_ROWS) / SPLASH_TARGET_VERIFY_ROWS,
                  ops::LinearEpilogue::None, nullptr, rows)
      .foldsNorm(rows);
}

void TtsModel::addPredictorHead(metal::CommandGraph &graph, Qwen3Arena &arena, uint32_t group, uint32_t rows,
                                metal::MetalBuffer logits, metal::MetalBuffer raw) const {
  if (group + 1 >= config_.codeGroups) throw std::invalid_argument("predictor head group out of range");
  if (!rows || rows > OMNI_MAXIMUM_LANES) throw std::invalid_argument("predictor head rows exceed a decode batch");
  const Qwen3Projection &head = weights_.predictorHeads[group];
  const uint32_t lanes = roundUp(rows, SPLASH_TARGET_VERIFY_ROWS) / SPLASH_TARGET_VERIFY_ROWS;
  if (std::holds_alternative<ops::Projection>(head)) {
    ops::Projection typed = std::get<ops::Projection>(head);
    typed.destination = ops::FloatOutput::Float32;
    const ops::LinearPlan plan = predictor_->linear().decodePlan(typed, lanes, ops::LinearEpilogue::None, nullptr, rows);
    if (raw) {
      // The final norm folded into the head (the bits of the norm dispatch).
      if (!predictorHeadFoldsNorm(arena, rows)) throw std::invalid_argument("the predictor head does not fold its norm");
      predictor_->linear().add(graph,
                               {.input = std::move(raw), .output = std::move(logits), .scratch = headScratch_,
                                .normWeight = weights_.predictor.finalNorm.buffer, .normRows = rows},
                               typed, plan);
      return;
    }
    predictor_->linear().add(graph, {.input = arena.selected, .output = std::move(logits), .scratch = headScratch_},
                             typed, plan);
    return;
  }
  ops::omni::DenseLinear::add(graph, std::get<ops::omni::DenseWeights>(head),
                              {.input = arena.selected, .output = std::move(logits)}, rows,
                              {.destination = ops::FloatOutput::Float32});
}

} // namespace splash::omni

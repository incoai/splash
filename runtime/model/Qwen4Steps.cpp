#include "model/Qwen4Steps.hpp"

#include "Checked.hpp"
#include "metal/abi/Qwen4.h"
#include "model/Qwen4Exp.hpp"
#include "model/SparseRowStore.hpp"
#include "ops/Qwen4.hpp"

#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>

namespace splash::model {
namespace {

constexpr uint32_t kLanes = ExecutionLimits::maximumBatchWidth;
constexpr uint32_t kChunkRows = ExecutionLimits::prefillTokenBudget;
constexpr uint32_t kVerifyRows = ExecutionLimits::targetVerifyRows;
// A decode step's most rows: every lane's verify rows, packed.
constexpr uint32_t kStepRows = kLanes * kVerifyRows;
// The QSA selection scores a prefill chunk's rows in batches of this many,
// which bounds its score scratch (a row scores every block of the context);
// a verify scores every lane's rows at once.
constexpr uint32_t kPrefillScoreRows = 256;
constexpr uint32_t kIndexWidth = QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION;

template <class T> constexpr uint64_t bytesFor(uint64_t elements) noexcept { return elements * sizeof(T); }

// The QSA blocks whose last row is one of `rows` rows from `position`.
constexpr uint32_t blockCount(uint64_t position, uint32_t rows) noexcept {
  return static_cast<uint32_t>((position + rows) / QWEN4_QSA_BLOCK - position / QWEN4_QSA_BLOCK);
}
// A prefill chunk's blocks, each sequence's after the one before it: a
// sequence completes at most one block more than its rows hold whole, and a
// chunk holds at most a sequence per lane. A verify lane's.
constexpr uint64_t kChunkBlockBytes = (kChunkRows / QWEN4_QSA_BLOCK + kLanes) * sizeof(Qwen4QsaBlock);
constexpr uint64_t kLaneBlockBytes =
    (kVerifyRows + QWEN4_QSA_BLOCK - 1) / QWEN4_QSA_BLOCK * sizeof(Qwen4QsaBlock);

// The words of a row's QSA bitmap: a bit per block of the context, the
// partial block at its end and the current one included.
uint32_t maskWords(const QwenTargetGeometry &g) { return (g.maximumContextTokens / QWEN4_QSA_BLOCK + 2 + 31) / 32; }
// A row's scores: one per block its bitmap covers.
uint32_t scoreStride(const QwenTargetGeometry &g) { return maskWords(g) * 32; }

// The attention's tiles of rows, whose block masks become a block list each
// (ops::PagedAttention::addBlockLists): a verify lane's rows, or a prefill
// tile's. A row attends to at most its top blocks and the block of its tail,
// so a tile's list holds at most this many entries.
constexpr uint32_t kTileRows = SPLASH_TARGET_VERIFY_ROWS;
static_assert(kTileRows == SPLASH_PREFILL_ATTENTION_TILE_ROWS, "verify and prefill tiles hold as many rows");
uint32_t listStride(const QwenTargetGeometry &g) { return kTileRows * (g.indexerTopBlocks + 1); }

// The tokens before a lane's verify rows that its rows' n-grams reach.
uint32_t priorCount(const QwenTargetGeometry &g) { return g.pleNgram - 1; }

// A scratch tensor (Qwen4Scratch): its bytes per row of a step and its fixed
// bytes, and whether a prefill group's chunks each hold their own: the
// residual streams and injection weights, which a layer's combine reads after
// the group's load of the layer's experts (QwenTarget::addPrefill), and the
// inputs the CPU writes before the command runs.
struct Tensor final {
  metal::MetalBuffer Qwen4Scratch::*field;
  uint64_t rowBytes;
  uint64_t fixedBytes;
  bool carried;
};

std::vector<Tensor> tensors(const QwenTargetGeometry &g, uint32_t scoreRows) {
  std::vector<Tensor> result{
      {&Qwen4Scratch::streams, bytesFor<float>(g.streamWidth()), 0, true},
      {&Qwen4Scratch::normalized, bytesFor<uint16_t>(g.streamWidth()), 0, false},
      {&Qwen4Scratch::low, bytesFor<uint16_t>(g.hyperRank), 0, false},
      {&Qwen4Scratch::gate, bytesFor<uint16_t>(g.streamWidth()), 0, false},
      {&Qwen4Scratch::weights, bytesFor<float>(g.hyperConnections), 0, true},
      {&Qwen4Scratch::branch, bytesFor<uint16_t>(g.hiddenSize), 0, false},
      {&Qwen4Scratch::zeros, bytesFor<uint16_t>(g.hiddenSize), 0, false},
      {&Qwen4Scratch::pleIndices, bytesFor<uint32_t>(g.pleHeads()), 0, true},
      {&Qwen4Scratch::pleRows, bytesFor<uint16_t>(g.pleWidth()), 0, false},
      {&Qwen4Scratch::pleValue, bytesFor<uint16_t>(g.hiddenSize), 0, false},
      {&Qwen4Scratch::pleGated, bytesFor<float>(g.streamWidth()), 0, false},
      {&Qwen4Scratch::pleConvolution, bytesFor<uint16_t>(g.streamWidth()), 0, false},
      {&Qwen4Scratch::indexerQuery, bytesFor<float>(kIndexWidth), 0, false},
      {&Qwen4Scratch::indexerKey, bytesFor<float>(QWEN4_QSA_DIMENSION), 0, false},
      {&Qwen4Scratch::qsaQueries, bytesFor<uint16_t>(kIndexWidth), 0, false},
      {&Qwen4Scratch::qsaScores, 0, bytesFor<float>(uint64_t{scoreRows} * scoreStride(g)), false},
      {&Qwen4Scratch::qsaMask, bytesFor<uint32_t>(maskWords(g)), 0, false},
      // A list of listStride entries per tile of rows, and a word per row
      // for the lists' counts: a step's rows fill whole tiles but the last of
      // each of its sequences (at most one a lane), which may hold fewer.
      {&Qwen4Scratch::qsaLists, bytesFor<uint32_t>(listStride(g) / kTileRows),
       bytesFor<uint32_t>(uint64_t{kLanes} * listStride(g)), false},
      {&Qwen4Scratch::qsaListCounts, bytesFor<uint32_t>(1), bytesFor<uint32_t>(kLanes), false},
  };
  // A streamed table's step stages up to every row id's row, each distinct
  // one once, and each id's slot.
  if (g.pleStreamRowBytes) {
    result.push_back({&Qwen4Scratch::pleSlots, bytesFor<uint32_t>(g.pleHeads()), 0, true});
    result.push_back({&Qwen4Scratch::pleStaging, uint64_t{g.pleHeads()} * g.pleStreamRowBytes, 0, true});
  }
  return result;
}

uint64_t bytesOf(const Tensor &tensor, uint64_t rows) noexcept { return tensor.rowBytes * rows + tensor.fixedBytes; }

// Where the scratch's regions start and the bytes of each of its buffers.
struct Layout final {
  uint64_t prefillBlocks = 0;
  uint64_t prefillBytes = 0;
  uint64_t decodeBlocks = 0;
  uint64_t decodePrior = 0;
  uint64_t decodeBytes = 0;
};

Layout layout(const QwenTargetGeometry &g, uint32_t chunks) {
  Layout result;
  for (const Tensor &tensor : tensors(g, kPrefillScoreRows))
    result.prefillBlocks += alignUp(bytesOf(tensor, kChunkRows)) * (tensor.carried ? chunks : 1);
  result.prefillBytes = result.prefillBlocks + alignUp(kChunkBlockBytes) * chunks;
  for (const Tensor &tensor : tensors(g, kStepRows)) result.decodeBlocks += alignUp(bytesOf(tensor, kStepRows));
  result.decodePrior = result.decodeBlocks + alignUp(kLanes * kLaneBlockBytes);
  result.decodeBytes = result.decodePrior + alignUp(bytesFor<uint32_t>(kLanes * priorCount(g)));
  return result;
}

template <class T> T *contents(const metal::MetalBuffer &buffer) {
  if (!buffer.contents()) throw std::logic_error("Qwen3.8-Flash-Next's step scratch is not CPU-visible");
  return static_cast<T *>(buffer.contents());
}

// The records of the blocks whose last row is one of `rows` rows from
// `position`, each its first token's position and RoPE position.
void writeBlocks(uint64_t position, uint32_t rows, const QwenTargetRequest &request, Qwen4QsaBlock *out) {
  for (uint64_t last = position; last < position + rows; ++last) {
    if (last % QWEN4_QSA_BLOCK != QWEN4_QSA_BLOCK - 1) continue;
    const uint64_t first = last + 1 - QWEN4_QSA_BLOCK;
    const std::array<uint32_t, 3> rope = request.rotary(first);
    *out++ = {static_cast<uint32_t>(first), {rope[0], rope[1], rope[2]}};
  }
}

} // namespace

Qwen4Steps::Qwen4Steps(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry,
                       metal::MetalBackend &backend, uint32_t prefillChunks)
    : weights_(weights), geometry_(geometry), backend_(backend), prefillChunks_(prefillChunks) {
  // The hash keeps at most QWEN4_PLE_MAXIMUM_NGRAM tokens of a row's n-grams.
  if (!prefillChunks_ || static_cast<bool>(weights_.pleStore) != static_cast<bool>(geometry_.pleStreamRowBytes) ||
      geometry_.pleNgram < 2 || geometry_.pleNgram > QWEN4_PLE_MAXIMUM_NGRAM)
    throw std::invalid_argument("invalid Qwen3.8-Flash-Next step geometry");
  const Layout bytes = layout(geometry_, prefillChunks_);
  prefill_ = backend_.allocateBuffer(bytes.prefillBytes, metal::BufferStorage::Shared, "qwen4-prefill-scratch");
  decode_ = backend_.allocateBuffer(bytes.decodeBytes, metal::BufferStorage::Shared, "qwen4-decode-scratch");
  prefillBlocksOffset_ = bytes.prefillBlocks;
  decodeBlocksOffset_ = bytes.decodeBlocks;
  decodePriorOffset_ = bytes.decodePrior;
  uint64_t cursor = 0;
  for (const Tensor &tensor : tensors(geometry_, kPrefillScoreRows)) {
    prefillTensors_.push_back({tensor.field, tensor.rowBytes, tensor.fixedBytes, tensor.carried, cursor});
    cursor += alignUp(bytesOf(tensor, kChunkRows)) * (tensor.carried ? prefillChunks_ : 1);
  }
  cursor = 0;
  for (const Tensor &tensor : tensors(geometry_, kStepRows)) {
    decodeTensors_.push_back({tensor.field, tensor.rowBytes, tensor.fixedBytes, tensor.carried, cursor});
    cursor += alignUp(bytesOf(tensor, kStepRows));
  }
  // The rows the MoE adds a block's output to.
  for (const metal::MetalBuffer &zeros : {prefillScratch(0).zeros, verifyScratch(kStepRows).zeros})
    std::memset(zeros.contents(), 0, zeros.sizeBytes());
}

QwenTargetScratchBytes Qwen4Steps::scratchBytes(const QwenTargetGeometry &geometry, uint32_t prefillChunks) {
  const Layout bytes = layout(geometry, prefillChunks);
  return {bytes.prefillBytes, bytes.decodeBytes};
}

QwenTargetScratchBytes Qwen4Steps::allocatedBytes() const noexcept {
  return {prefill_.sizeBytes(), decode_.sizeBytes()};
}

Qwen4Scratch Qwen4Steps::prefillScratch(uint32_t index) const {
  if (index >= prefillChunks_) throw std::out_of_range("Qwen3.8-Flash-Next prefill chunk");
  Qwen4Scratch scratch{.scoreRows = kPrefillScoreRows,
                       .scoreStride = scoreStride(geometry_),
                       .maskWords = maskWords(geometry_),
                       .listStride = listStride(geometry_)};
  for (const Placed &tensor : prefillTensors_) {
    const uint64_t bytes = tensor.rowBytes * kChunkRows + tensor.fixedBytes;
    scratch.*tensor.field =
        backend_.view(prefill_, tensor.offset + (tensor.carried ? index * alignUp(bytes) : 0), bytes);
  }
  return scratch;
}

std::vector<Qwen4Blocks> Qwen4Steps::preparePrefill(uint32_t index, const QwenTargetPrefillChunk &chunk) {
  const Qwen4Scratch scratch = prefillScratch(index);
  const metal::MetalBuffer records =
      backend_.view(prefill_, prefillBlocksOffset_ + index * alignUp(kChunkBlockBytes), kChunkBlockBytes);
  auto *indices = contents<uint32_t>(scratch.pleIndices);
  std::vector<Qwen4Blocks> blocks;
  uint64_t written = 0;
  for (const QwenTargetPrefillSequence &sequence : chunk.sequences) {
    const uint64_t position = sequence.chunk.committed_tokens;
    const uint32_t count = blockCount(position, sequence.rows);
    const uint64_t bytes = count * sizeof(Qwen4QsaBlock);
    if (sequence.tokens.size() != sequence.rows || uint64_t{sequence.rowBegin} + sequence.rows > chunk.rows ||
        written + bytes > kChunkBlockBytes)
      throw std::invalid_argument("invalid Qwen3.8-Flash-Next prefill sequence");
    qwen4PleRows(weights_, position, sequence.tokens, sequence.request.tokens,
                 indices + uint64_t{sequence.rowBegin} * geometry_.pleHeads());
    if (!count) {
      blocks.push_back({});
      continue;
    }
    const metal::MetalBuffer view = backend_.view(records, written, bytes);
    writeBlocks(position, sequence.rows, sequence.request, contents<Qwen4QsaBlock>(view));
    blocks.push_back({view, count});
    written += bytes;
  }
  // A chunk's rows at once, so that its lanes share their rows.
  if (weights_.pleStore)
    stage(*weights_.pleStore, geometry_.pleStreamRowBytes, scratch, chunk.rows * geometry_.pleHeads(), true);
  return blocks;
}

void Qwen4Steps::readAhead(std::span<const QwenTargetPrefillChunk> chunks) {
  if (!weights_.pleStore) return;
  // Each lane's last sequence of the command, which ends its rows there.
  std::array<const QwenTargetPrefillSequence *, kLanes> last{};
  for (const QwenTargetPrefillChunk &chunk : chunks)
    for (const QwenTargetPrefillSequence &sequence : chunk.sequences) {
      if (sequence.lane >= kLanes) throw std::invalid_argument("invalid Qwen3.8-Flash-Next prefill lane");
      last[sequence.lane] = &sequence;
    }
  const uint32_t heads = geometry_.pleHeads();
  ahead_.clear();
  uint64_t budget = kChunkRows;
  for (const QwenTargetPrefillSequence *sequence : last) {
    if (!sequence) continue;
    const std::span<const uint32_t> prompt = sequence->request.tokens;
    const uint64_t from = sequence->chunk.committed_tokens + sequence->rows;
    const uint64_t end = std::min<uint64_t>(prompt.size(), from + budget);
    if (from >= end) continue;
    const size_t base = ahead_.size();
    ahead_.resize(base + (end - from) * heads);
    qwen4PleRows(weights_, from, prompt.subspan(from, end - from), prompt, ahead_.data() + base);
    budget -= end - from;
    if (!budget) break;
  }
  if (!ahead_.empty()) weights_.pleStore->prefetch(ahead_);
}

Qwen4Scratch Qwen4Steps::verifyScratch(uint32_t rows) const {
  if (!rows || rows > kStepRows) throw std::out_of_range("Qwen3.8-Flash-Next verify rows");
  Qwen4Scratch scratch{.scoreRows = kStepRows,
                       .scoreStride = scoreStride(geometry_),
                       .maskWords = maskWords(geometry_),
                       .listStride = listStride(geometry_)};
  for (const Placed &tensor : decodeTensors_)
    scratch.*tensor.field = backend_.view(decode_, tensor.offset, tensor.rowBytes * rows + tensor.fixedBytes);
  return scratch;
}

void Qwen4Steps::prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request) {
  if (lane >= kLanes) throw std::invalid_argument("invalid Qwen3.8-Flash-Next verify lane");
  const uint32_t prior = priorCount(geometry_);
  uint32_t *before = contents<uint32_t>(priorTokens()) + lane * prior;
  for (uint32_t back = 1; back <= prior; ++back)
    before[back - 1] = position >= back && position - back < request.tokens.size()
                           ? request.tokens[position - back]
                           : QWEN4_PLE_NONE;
  preparedPositions_[lane] = position;
  if (blockCount(position, kVerifyRows))
    writeBlocks(position, kVerifyRows, request,
                contents<Qwen4QsaBlock>(backend_.view(decode_, decodeBlocksOffset_ + lane * kLaneBlockBytes,
                                                      kLaneBlockBytes)));
}

Qwen4LaneBlocks Qwen4Steps::verifyBlocks(std::span<const uint32_t> positions) const {
  if (positions.empty() || positions.size() > kLanes) throw std::invalid_argument("invalid Qwen3.8-Flash-Next lanes");
  Qwen4LaneBlocks blocks{.records = backend_.view(decode_, decodeBlocksOffset_, kLanes * kLaneBlockBytes),
                         .counts = {},
                         .laneRecords = kLaneBlockBytes / sizeof(Qwen4QsaBlock)};
  for (uint32_t lane = 0; lane < positions.size(); ++lane) {
    if (preparedPositions_[lane] != positions[lane])
      throw std::logic_error("a Qwen3.8-Flash-Next verify lane's inputs were prepared for another position");
    blocks.counts[lane] = blockCount(positions[lane], kVerifyRows);
  }
  return blocks;
}

metal::MetalBuffer Qwen4Steps::priorTokens() const {
  return backend_.view(decode_, decodePriorOffset_, bytesFor<uint32_t>(kLanes * priorCount(geometry_)));
}

void Qwen4Steps::addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer tokens, uint32_t lanes) const {
  if (!lanes || lanes > kLanes) throw std::invalid_argument("invalid PLE hash");
  const Qwen4ExpLayout &layout = weights_.layout;
  const Qwen4PleHash &hash = weights_.pleHash;
  const uint32_t heads = layout.pleHeads();
  if (heads > QWEN4_PLE_MAXIMUM_HEADS || hash.headVocabularies.size() != heads || hash.headOffsets.size() != heads)
    throw std::logic_error("PLE hash geometry mismatch");
  Qwen4PleHashParams params{};
  params.rows = lanes * kVerifyRows;
  params.ngram = layout.pleNgram;
  params.heads_per_ngram = layout.pleHeadsPerNgram;
  params.eos = layout.pleEosToken;
  std::copy(hash.multipliers.begin(), hash.multipliers.end(), params.multipliers);
  std::copy(hash.headVocabularies.begin(), hash.headVocabularies.end(), params.vocabularies);
  std::copy(hash.headOffsets.begin(), hash.headOffsets.end(), params.offsets);
  const Qwen4Scratch scratch = verifyScratch(params.rows);
  ops::Qwen4::addPleHash(graph, std::move(tokens), priorTokens(), scratch.pleIndices, params, heads);
  // A verify step stages a few rows per lane, most of them already in the
  // cache, so its host step is short and keeps the GPU awake
  // (metal::EventStep::awake).
  if (weights_.pleStore)
    graph.host([store = weights_.pleStore, rowBytes = geometry_.pleStreamRowBytes, scratch,
                count = params.rows * heads] { stage(*store, rowBytes, scratch, count, false); },
               {}, true);
}

std::vector<uint32_t> Qwen4Steps::verifyPleRows(uint32_t lane) const {
  if (lane >= kLanes) throw std::out_of_range("Qwen3.8-Flash-Next verify lane");
  const uint32_t heads = geometry_.pleHeads();
  const auto *indices = contents<const uint32_t>(verifyScratch(kStepRows).pleIndices);
  return {indices + uint64_t{lane} * kVerifyRows * heads, indices + uint64_t{lane + 1} * kVerifyRows * heads};
}

// The products stay below 2^63 (the multipliers are bounded by the
// vocabulary), so the unsigned hash equals the reference's signed one.
void qwen4PleRows(const Qwen4ExpWeights &weights, uint64_t position, std::span<const uint32_t> tokens,
                  std::span<const uint32_t> history, uint32_t *out) {
  const Qwen4ExpLayout &layout = weights.layout;
  const Qwen4PleHash &hash = weights.pleHash;
  const uint32_t eos = layout.pleEosToken;
  const auto token = [&](uint64_t at) -> std::optional<uint32_t> {
    if (at >= position) return tokens[at - position];
    if (at < history.size()) return history[at];
    return std::nullopt;
  };
  for (uint32_t row = 0; row < tokens.size(); ++row) {
    std::array<uint64_t, QWEN4_PLE_MAXIMUM_NGRAM> context{tokens[row]};
    bool cut = false;
    for (uint32_t back = 1; back < layout.pleNgram; ++back) {
      const std::optional<uint32_t> previous = position + row >= back ? token(position + row - back) : std::nullopt;
      cut = cut || !previous || *previous == eos;
      context[back] = cut ? eos : *previous;
    }
    uint32_t *rowHeads = out + uint64_t{row} * layout.pleHeads();
    for (uint32_t gram = 2; gram <= layout.pleNgram; ++gram) {
      uint64_t mixed = context[0] * hash.multipliers[0];
      for (uint32_t j = 1; j < gram; ++j) mixed ^= context[j] * hash.multipliers[j];
      for (uint32_t g = 0; g < layout.pleHeadsPerNgram; ++g) {
        const uint32_t head = (gram - 2) * layout.pleHeadsPerNgram + g;
        rowHeads[head] = static_cast<uint32_t>(mixed % hash.headVocabularies[head] + hash.headOffsets[head]);
      }
    }
  }
}

// The command that reads the staged rows waits for the stage: a prefill's is
// submitted after it returns, a verify's continues past its host step. No
// other command reading the scratch is in flight: the engine runs one command
// at a time. A row that cannot be read throws, before a prefill's command is
// submitted or as its host step's failure, which stops the engine. A
// prefill's stage takes the rows its command's read-ahead brought in.
void Qwen4Steps::stage(SparseRowStore &store, uint64_t rowBytes, const Qwen4Scratch &scratch, uint32_t count,
                       bool prefill) {
  if (scratch.pleIndices.sizeBytes() < bytesFor<uint32_t>(count) ||
      scratch.pleSlots.sizeBytes() < bytesFor<uint32_t>(count) || scratch.pleStaging.sizeBytes() < count * rowBytes)
    throw std::logic_error("PLE staging is smaller than its step");
  store.stage({contents<const uint32_t>(scratch.pleIndices), count},
              {contents<uint8_t>(scratch.pleStaging), count * rowBytes}, {contents<uint32_t>(scratch.pleSlots), count},
              prefill ? SparseRowUse::Prefill : SparseRowUse::Decode, prefill);
}

} // namespace splash::model

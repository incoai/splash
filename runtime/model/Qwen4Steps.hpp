#pragma once

// What Qwen3.8-Flash-Next (model/Qwen4Exp.hpp) needs of a step beyond its
// weights and the buffers every Qwen target binds, which its target keeps
// (model/Qwen4ExpTarget.hpp): the scratch of its hyper-connected streams, its
// PLE layer and its QSA attention, and the inputs it makes of a step's tokens
// on the CPU. Those are the PLE rows of prompt rows (the n-gram hashes of
// their tokens), the tokens before a decode lane's rows (from which the GPU
// hashes the rows' PLE rows, since only it has their proposals), the QSA
// blocks a step's rows complete, and, when the PLE table streams from its
// file (--ple-memory), each step's PLE rows staged from the table: a
// prefill's before its command is submitted, a verify's in a host step of its
// command, between the hash and the gather.

#include "model/QwenTarget.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

struct Qwen4ExpWeights;
class SparseRowStore;

// The scratch of a step's rows (a prefill chunk's or a decode step's): the
// fp32 residual streams, a mix's normalized streams, low-rank and gate rows
// and injection weights, a block's branch and the zero rows the MoE adds its
// output to, the PLE layer's row ids, gathered rows, projections, gated
// values and convolution inputs, and the QSA indexer's projections, prepared
// queries, raw keys (a decode step's keys of every attention layer,
// keyLayerRows rows apart, which its commit reads), scores of scoreRows rows
// of scoreStride each, block bitmaps of maskWords words per row, and the
// block lists of the attention's tiles of rows (listStride entries each,
// ops::AttentionBlocks) with their counts. A streamed PLE
// table's step also stages each distinct row it gathers once (pleStaging)
// and each row id's row there (pleSlots).
struct Qwen4Scratch final {
  metal::MetalBuffer streams;
  metal::MetalBuffer normalized;
  metal::MetalBuffer low;
  metal::MetalBuffer gate;
  metal::MetalBuffer weights;
  metal::MetalBuffer branch;
  metal::MetalBuffer zeros;
  metal::MetalBuffer pleIndices;
  metal::MetalBuffer pleSlots;
  metal::MetalBuffer pleStaging;
  metal::MetalBuffer pleRows;
  metal::MetalBuffer pleValue;
  metal::MetalBuffer pleGated;
  metal::MetalBuffer pleConvolution;
  metal::MetalBuffer indexerQuery;
  metal::MetalBuffer indexerKey;
  metal::MetalBuffer qsaQueries;
  metal::MetalBuffer qsaKeys;
  metal::MetalBuffer qsaScores;
  metal::MetalBuffer qsaMask;
  metal::MetalBuffer qsaLists;
  metal::MetalBuffer qsaListCounts;
  uint32_t scoreRows = 0;
  uint32_t scoreStride = 0;
  uint32_t maskWords = 0;
  uint32_t listStride = 0;
  uint32_t keyLayerRows = 0;
};

// The PLE rows of `tokens` from `position` on into `out`, each row's heads'
// in order: the 2- to
// n-gram hashes of a row's token and the ones before it, which come from
// `history` before `position`, an EOS or the sequence start cutting the
// window there, each n-gram's heads taking its hash modulo their row counts.
void qwen4PleRows(const Qwen4ExpWeights &weights, uint64_t position, std::span<const uint32_t> tokens,
                  std::span<const uint32_t> history, uint32_t *out);

// The QSA blocks a prefill sequence's rows complete, which its attention
// layers pool: `count` BlockIndexerBlock records (none without a block).
struct Qwen4Blocks final {
  metal::MetalBuffer records;
  uint32_t count = 0;
};

// The QSA blocks a verify step's lanes complete: lane l's counts[l] records
// from record l * laneRecords of `records`.
struct Qwen4LaneBlocks final {
  metal::MetalBuffer records;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> counts{};
  uint32_t laneRecords = 0;
};

class Qwen4Steps final {
public:
  // Scratch for prefill commands of up to `prefillChunks` chunks.
  Qwen4Steps(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
             uint32_t prefillChunks);
  // The scratch of a target whose streamed table's rows are `tableRowBytes`
  // each (zero: the table loads).
  [[nodiscard]] static QwenTargetScratchBytes scratchBytes(const QwenTargetGeometry &geometry, uint64_t tableRowBytes,
                                                           uint32_t prefillChunks);
  // What the constructor allocated, scratchBytes' of its geometry.
  [[nodiscard]] QwenTargetScratchBytes allocatedBytes() const noexcept;

  // Chunk `index`'s scratch of a prefill command.
  [[nodiscard]] Qwen4Scratch prefillScratch(uint32_t index) const;
  // Writes the PLE rows and QSA blocks of chunk `index`'s sequences and,
  // when the table streams, stages the chunk's PLE rows. Returns each
  // sequence's blocks.
  [[nodiscard]] std::vector<Qwen4Blocks> preparePrefill(uint32_t index, const QwenTargetPrefillChunk &chunk);
  // Each sequence's blocks preparePrefill wrote, which the chunk's later
  // passes over the layers read on (QwenPrefillLayers).
  [[nodiscard]] std::vector<Qwen4Blocks> prefillBlocks(uint32_t index, const QwenTargetPrefillChunk &chunk) const;
  // When the table streams, starts reading the PLE rows of the prompt rows
  // that follow each lane's rows of a prefill command, at most a command's
  // chunks' worth and its lanes' in order: the readers bring them into the
  // table's cache while the command runs, where the next command's first
  // stage finishes them for all of its chunks.
  void readAhead(std::span<const QwenTargetPrefillChunk> chunks);

  // The scratch of a decode step's first `rows` rows (the target's storage
  // rows of the step, QwenTarget::decodeStorageLanes).
  [[nodiscard]] Qwen4Scratch verifyScratch(uint32_t rows) const;
  // Writes batch lane `lane`'s inputs of its verify rows from `position` on:
  // the tokens before them and the QSA blocks they complete.
  void prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request);
  // The blocks prepareVerifyLane wrote for each lane of a step, whose rows
  // start at positions[lane].
  [[nodiscard]] Qwen4LaneBlocks verifyBlocks(std::span<const uint32_t> positions) const;
  // The positions prepareVerifyLane prepared the first `lanes` lanes for.
  [[nodiscard]] std::array<uint32_t, ExecutionLimits::maximumBatchWidth> verifyPositions(uint32_t lanes) const;
  // The QSA pending keys of `layers` attention layers from `layer` on in a
  // state cell's auxiliary state.
  [[nodiscard]] metal::MetalBuffer qsaPending(const metal::MetalBuffer &auxiliary, uint32_t layer,
                                              uint32_t layers = 1) const;
  // The GPU's PLE rows of `lanes` lanes' verify rows, from their tokens and
  // the tokens before each lane's, and, when the table streams, the host step
  // that stages them.
  void addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer tokens, uint32_t lanes) const;
  // The PLE rows of batch lane `lane`'s verify rows of the last step, each
  // row's heads' in order.
  [[nodiscard]] std::vector<uint32_t> verifyPleRows(uint32_t lane) const;

private:
  // A scratch tensor placed in its buffer (Qwen4Steps.cpp tensors()).
  struct Placed final {
    metal::MetalBuffer Qwen4Scratch::*field;
    uint64_t rowBytes;
    uint64_t fixedBytes;
    bool carried;
    uint64_t offset;
  };

  // Stages from `store` the rows, of rowBytes each, of the first `count` row
  // ids of `scratch`. A static function, so that a host step that stages
  // holds what it uses rather than this object.
  static void stage(SparseRowStore &store, uint64_t rowBytes, const Qwen4Scratch &scratch, uint32_t count,
                    bool prefill);
  // Each lane's tokens before its verify rows, which the hash reads.
  [[nodiscard]] metal::MetalBuffer priorTokens() const;

  const Qwen4ExpWeights &weights_;
  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  uint32_t prefillChunks_ = 0;
  // The bytes of a row of a streamed PLE table, which a step stages; zero
  // for a table in memory.
  uint64_t pleRowBytes_ = 0;
  // The prefill scratch: its tensors, a carried one's per chunk, then each
  // chunk's QSA blocks from byte prefillBlocksOffset_. The decode scratch:
  // its tensors for a step's most rows, then each lane's QSA blocks from byte
  // decodeBlocksOffset_ and the tokens before its rows from byte
  // decodePriorOffset_.
  metal::MetalBuffer prefill_;
  metal::MetalBuffer decode_;
  std::vector<Placed> prefillTensors_;
  std::vector<Placed> decodeTensors_;
  uint64_t prefillBlocksOffset_ = 0;
  uint64_t decodeBlocksOffset_ = 0;
  uint64_t decodePriorOffset_ = 0;
  // The position each lane's verify inputs were prepared for.
  std::array<uint64_t, ExecutionLimits::maximumBatchWidth> preparedPositions_{};
  // Row ids of a read-ahead, kept between commands for their allocation.
  std::vector<uint32_t> ahead_;
};

} // namespace splash::model

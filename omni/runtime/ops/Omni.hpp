#pragma once

// The operators of the Omni (TTS/ASR) models over the kernels of
// runtime/metal/kernels/omni: dense bf16 projections, the Qwen3 decoder's
// paged attention, encoder attention, norms, token gathers and sampling.
// Each add* checks the extents its kernels reach before encoding anything.

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Omni.h"
#include "ops/Weights.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::ops::omni {

enum class Activation : uint32_t { None = 0, Gelu = 1, Silu = 2 };

// A dense projection as the checkpoint stores it: bf16 [outputSize][inputSize]
// and, when present, an fp32 bias of outputSize. `packed`, when present,
// holds the same bits packed (metal/abi/Omni.h), which decode steps of up
// to eight rows stream instead, about 0.78 of the bytes; the other kernels
// read `weight`.
struct DenseWeights final {
  metal::MetalBuffer weight;
  metal::MetalBuffer bias;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  metal::MetalBuffer packed{};
};

// Packing a bf16 [outputSize][inputSize] weight (metal/abi/Omni.h) in row
// ranges, so a loader can spread it over threads: count each range's escape
// blocks, then pack each range with the escape records before it.
[[nodiscard]] uint32_t packedEscapeBlocks(std::span<const uint16_t> rows, uint32_t inputSize);
// The packed bytes of a weight with `escapeBlocks` escape blocks.
[[nodiscard]] uint64_t packedDenseBytes(uint32_t outputSize, uint32_t inputSize, uint64_t escapeBlocks);
// Packs rows [first, first + count) of `weights` (every row) into `packed`
// (packedDenseBytes of the whole weight), their escape records numbered
// from `firstEscape`.
void packDenseRows(std::span<const uint16_t> weights, uint32_t outputSize, uint32_t inputSize, uint32_t first,
                   uint32_t count, uint32_t firstEscape, std::span<uint8_t> packed);
// The bf16 bits of a packed weight.
void unpackDense(std::span<const uint8_t> packed, uint32_t outputSize, uint32_t inputSize,
                 std::span<uint16_t> weights);

struct GemmBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer output;
  metal::MetalBuffer residual;
  uint32_t inputStride = 0;  // 0: inputSize
  uint32_t outputStride = 0; // 0: outputSize
  // A decode step's K-split scratch (denseSplitScratch): fp32 partials and
  // the split counters, zeroed at allocation and returned to zero by every
  // split dispatch. Reused serially within one command stream, never by
  // concurrent ones. Without them the projections run unsplit.
  metal::MetalBuffer partials{};
  metal::MetalBuffer counters{};
};

// The split scratch the decode projections of up to `maxRows` rows of a
// weight of this shape bind (bytes; zero when no step of them splits).
struct DenseSplitScratch final {
  uint64_t partialsBytes = 0;
  uint64_t countersBytes = 0;
  DenseSplitScratch &include(const DenseSplitScratch &other) noexcept;
};
[[nodiscard]] DenseSplitScratch denseSplitScratch(uint32_t outputSize, uint32_t inputSize, uint32_t maxRows);

struct GemmOptions final {
  Activation activation = Activation::None;
  bool residual = false;
  FloatOutput destination = FloatOutput::BFloat16;
  // Steps of two to sixteen rows on the MMA tiles. False keeps the GEMV to
  // four rows and the MPP tiles past that, whose arithmetic of a row does
  // not depend on the rows beside it from five rows on (the audio tower, a
  // clip's rows encoding to the same bits alone or batched with others).
  bool stepTiles = true;
};

// Rows the input and output buffers of a `rows`-row dense projection must
// hold: whole tiles of the kernel the rows run.
[[nodiscard]] uint32_t densePaddedRows(uint32_t rows) noexcept;

class DenseLinear final {
public:
  // output = act(weight input + bias) (+ residual) over `rows` rows.
  static void add(metal::CommandGraph &graph, const DenseWeights &weights, const GemmBuffers &buffers,
                  uint32_t rows, GemmOptions options = {});
  // output = silu(gate input + gate bias) * (up input + up bias) (+ residual).
  static void addGateUp(metal::CommandGraph &graph, const DenseWeights &gate, const DenseWeights &up,
                        const GemmBuffers &buffers, uint32_t rows, GemmOptions options = {});
  // Whether add() of `weights` over `rows` and over `otherRows` rows (with
  // split scratch) computes each row with the same arithmetic: the GEMV
  // kernels share each lane's summation order, the MMA tiles theirs at one
  // split count, the MPP tiles theirs.
  [[nodiscard]] static bool sameRowArithmetic(const DenseWeights &weights, uint32_t rows, uint32_t otherRows);
};

// A sequence of an attention dispatch (metal/abi/Omni.h OmniSequenceParams
// without the split count, which the operator chooses).
struct AttentionSequence final {
  uint32_t pageTableOffset = 0;
  uint32_t pageTableStride = 0;
  uint32_t committedTokens = 0;
  uint32_t rows = 0;
  uint32_t rowOffset = 0;
  uint32_t positionOffset = 0;
};

// The host-side tables of one attention dispatch, as the kernels read
// them: written into `sequences` and `tiles` buffers by the caller.
struct AttentionTables final {
  std::vector<OmniSequenceParams> sequences;
  std::vector<OmniPrefillTile> tiles;
  uint32_t rows = 0;
  uint32_t paddedRows = 0;
};
// The tables of `sequences` whose rows are contiguous: the prefill tiles of
// each sequence and the split counts of its history.
[[nodiscard]] AttentionTables attentionTables(std::span<const AttentionSequence> sequences, bool decode);

struct AttentionBuffers final {
  metal::MetalBuffer qkv;                // bf16 [rows][OMNI_DECODER_QKV_WIDTH]
  metal::MetalBuffer queryNorm, keyNorm; // bf16 [128]
  metal::MetalBuffer inverseFrequencies; // fp32 [64]
  metal::MetalBuffer rowSequence;        // u32 [rows]
  metal::MetalBuffer sequences;          // OmniSequenceParams [sequences]
  metal::MetalBuffer tiles;              // OmniPrefillTile [tiles]
  metal::MetalBuffer pageTable;          // u32
  metal::MetalBuffer pages;              // OMNI_PAGE_BYTES per page
  metal::MetalBuffer queries;            // bf16 [8][paddedRows][2][128]
  metal::MetalBuffer partials;           // fp32
  metal::MetalBuffer statistics;         // fp32
  metal::MetalBuffer output;             // bf16 [rows][2048]
  // fp32 [rows][64][cos, sin]: the fused decode attention's rotation table
  // (Attention::addDecodeFused).
  metal::MetalBuffer rotations;
  // u32 [sequences][16]: the long decode attention's split arrivals per
  // query head, zeros between dispatches (Attention::addDecodeLong).
  metal::MetalBuffer counters;
};

// Whether a fused decode attention computes its rotations, also stores them
// in AttentionBuffers::rotations (a forward's first layer) or loads them
// from there (the later layers of the same rows and positions), which saves
// the later layers the sines and cosines and gives the same values.
enum class Rotations : uint32_t {
  Compute = OMNI_ROTATIONS_COMPUTE,
  Store = OMNI_ROTATIONS_STORE,
  Load = OMNI_ROTATIONS_LOAD,
};

struct AttentionWorkspace final {
  uint64_t queriesBytes = 0;
  uint64_t partialsBytes = 0;
  uint64_t statisticsBytes = 0;
  uint64_t countersBytes = 0;
};

class Attention final {
public:
  [[nodiscard]] static AttentionWorkspace prefillWorkspace(uint32_t paddedRows, uint32_t tiles) noexcept;
  [[nodiscard]] static AttentionWorkspace decodeWorkspace(uint32_t lanes) noexcept;
  // Normalizes and rotates the rows' queries and keys into `queries` and the
  // pages of `layer`, and stores their values.
  static void addPrepare(metal::CommandGraph &graph, const AttentionBuffers &buffers, const AttentionTables &tables,
                         uint32_t layer, float rmsEpsilon, uint32_t pageTableEntries, uint32_t pageCount);
  static void addPrefill(metal::CommandGraph &graph, const AttentionBuffers &buffers, const AttentionTables &tables,
                         uint32_t layer, uint32_t pageTableEntries, uint32_t pageCount);
  // The split attention of one row per sequence over the queries addPrepare
  // wrote, and its reduction. No forward runs it (a decode step takes the
  // fused or the long kernel); it is the reference the long kernel's bits
  // are checked against (omni_kernels_test).
  static void addDecode(metal::CommandGraph &graph, const AttentionBuffers &buffers, const AttentionTables &tables,
                        uint32_t layer, uint32_t pageTableEntries, uint32_t pageCount);
  // Whether a decode step runs the fused kernel: one or two rows per
  // sequence, at any history unless the long kernel takes it (Omni.cpp).
  [[nodiscard]] static bool fusable(const AttentionTables &tables) noexcept;
  // Whether every sequence's history and rows fit one page.
  [[nodiscard]] static bool fitsOnePage(const AttentionTables &tables) noexcept;
  // Prepare, store, attention and output in one dispatch (fusable tables).
  // Tables that fit one page take the one-page kernel, and steps of a few
  // one-row sequences a threadgroup per query head (both the same bits as
  // a threadgroup per KV head; Omni.cpp), unless `kvHeads` is true (the
  // tests' reference).
  static void addDecodeFused(metal::CommandGraph &graph, const AttentionBuffers &buffers, const AttentionTables &tables,
                             uint32_t layer, float rmsEpsilon, uint32_t pageTableEntries, uint32_t pageCount,
                             Rotations rotations = Rotations::Compute, bool kvHeads = false);
  // Whether a decode step runs the long kernel: one sequence of one row
  // over more than OMNI_FUSED_MAXIMUM_PAGES pages.
  [[nodiscard]] static bool longFusable(const AttentionTables &tables) noexcept;
  // Prepare, store, the split attention and its reduction in one dispatch
  // (longFusable tables): a threadgroup per query head and split, the last
  // split of each to finish reducing. The same bits as addPrepare and
  // addDecode over the same splits.
  static void addDecodeLong(metal::CommandGraph &graph, const AttentionBuffers &buffers, const AttentionTables &tables,
                            uint32_t layer, float rmsEpsilon, uint32_t pageTableEntries, uint32_t pageCount,
                            Rotations rotations = Rotations::Compute);
  // Whether a verify forward (ragged rows, each seeing the tokens through
  // its own) fits the fused verify kernel.
  [[nodiscard]] static bool verifyFusable(const AttentionTables &tables) noexcept;
  // Prepare, store, attention and output of a verify forward in one
  // dispatch, a threadgroup per KV head and row (verifyFusable tables), or
  // per query head and row for a few rows (the same bits) unless `kvHeads`
  // is true (the tests' reference).
  static void addVerifyFused(metal::CommandGraph &graph, const AttentionBuffers &buffers, const AttentionTables &tables,
                             uint32_t layer, float rmsEpsilon, uint32_t pageTableEntries, uint32_t pageCount,
                             Rotations rotations = Rotations::Compute, bool kvHeads = false);
  // Bytes of the rotation table of `rows` rows.
  [[nodiscard]] static constexpr uint64_t rotationBytes(uint32_t rows) noexcept {
    return uint64_t{rows} * OMNI_DECODER_HEAD_DIM / 2 * 8;
  }
};

struct DenseAttentionBuffers final {
  metal::MetalBuffer queries, keys, values; // bf16 rows of `stride` elements, head h at h * 64
  metal::MetalBuffer output;                // bf16 [rows][1024]
  metal::MetalBuffer rows;                  // OmniDenseRow [rows]
};
enum class DenseMask : uint32_t { None = 0, CausalWindow = 1 };
class DenseAttention final {
public:
  // Keys and values must hold whole blocks of 32 rows past every row's
  // key_end (keyRows: the rows the buffers hold).
  static void add(metal::CommandGraph &graph, const DenseAttentionBuffers &buffers, uint32_t rows, uint32_t keyRows,
                  uint32_t stride, DenseMask mask, uint32_t window, float scale, uint32_t heads = OMNI_ENCODER_HEADS);
};

class Norms final {
public:
  static void addRms(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer weight,
                     metal::MetalBuffer output, uint32_t rows, uint32_t width, float epsilon, bool float32);
  // fp32 rows; bf16 weights and output when !float32Weights, else fp32.
  static void addLayer(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer weight,
                       metal::MetalBuffer bias, metal::MetalBuffer output, uint32_t rows, uint32_t width,
                       float epsilon, bool float32Weights);
  static void addActivate(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer output,
                          uint64_t count, Activation activation, bool float32);
  static void addConvert(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer output,
                         uint64_t count, bool toFloat32);
  // output = act(input + bias) over bf16 rows of `width` and an fp32 bias.
  static void addBiasActivate(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer bias,
                              metal::MetalBuffer output, uint32_t rows, uint32_t width, Activation activation);
};

class Embedding final {
public:
  // Output row i goes to row i * outputRowStride (1: contiguous).
  static void addGather(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer table,
                        metal::MetalBuffer output, uint32_t rows, uint32_t vocabulary, uint32_t width,
                        uint32_t outputRowStride = 1, uint32_t tokenStride = 1);
  static void addGatherRows(metal::CommandGraph &graph, metal::MetalBuffer rowIndex, metal::MetalBuffer source,
                            metal::MetalBuffer output, uint32_t rows, uint32_t sourceRows, uint32_t width,
                            uint32_t outputRowStride = 1);
  struct TtsTables final {
    metal::MetalBuffer codec;      // bf16 [codecVocabulary][width]
    metal::MetalBuffer predictor;  // bf16 [15][predictorVocabulary][width]
    uint32_t codecVocabulary = 0;
    uint32_t predictorVocabulary = 0;
    uint32_t width = 0;
  };
  static void addTtsStepInput(metal::CommandGraph &graph, metal::MetalBuffer codes, const TtsTables &tables,
                              metal::MetalBuffer trailingRows, metal::MetalBuffer trailingPool,
                              uint32_t trailingPoolRows, metal::MetalBuffer output, uint32_t rows);
  static void addPromptAssemble(metal::CommandGraph &graph, metal::MetalBuffer positions, metal::MetalBuffer textRows,
                                uint32_t textRowCount, const TtsTables &tables, metal::MetalBuffer speaker,
                                metal::MetalBuffer frames, uint32_t frameCount, metal::MetalBuffer output,
                                uint32_t rows);
};

class Sampling final {
public:
  static void addSampleCodes(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer history,
                             metal::MetalBuffer historyCounts, metal::MetalBuffer uniforms,
                             metal::MetalBuffer output, const OmniSampleParams &params);
  // Row r's greedy token to output[r * outputStride].
  static void addArgmax(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer output,
                        uint32_t rows, uint32_t vocabulary, uint32_t stride, uint32_t outputStride = 1);
  // The same, each row's token through `map` (u32 [vocabulary]): a draft's
  // reduced vocabulary to its target's tokens.
  static void addArgmaxMapped(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer output,
                              metal::MetalBuffer map, uint32_t rows, uint32_t vocabulary, uint32_t stride,
                              uint32_t outputStride = 1);
};

} // namespace splash::ops::omni

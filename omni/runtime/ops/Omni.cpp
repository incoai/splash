#include "ops/Omni.hpp"

#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops::omni {
namespace {

constexpr uint32_t kQkvWidth = OMNI_DECODER_QKV_WIDTH;
constexpr uint32_t kAttentionWidth = OMNI_DECODER_ATTENTION_WIDTH;
constexpr uint32_t kKvHeads = OMNI_DECODER_KV_HEADS;
constexpr uint32_t kGroup = OMNI_DECODER_QUERY_HEADS / OMNI_DECODER_KV_HEADS;
constexpr uint32_t kHeadDim = OMNI_DECODER_HEAD_DIM;
constexpr uint32_t kTileRows = OMNI_PREFILL_TILE_ROWS;

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept { return (value + multiple - 1) / multiple * multiple; }

// The dense GEMM tile of `rows` rows into `outputSize` columns of
// `inputSize` inputs. One row, and gate/up pairs up to three rows, run the
// GEMV compiled for the next power of two of rows, two columns per
// simdgroup for one row of a plain projection, else one; two to sixteen
// rows the MMA tiles (gemm_bf16.metal omni_mma_*), which read each weight
// once a step at close to one row's GEMV rate; more rows the MPP tiles
// (mppTile).
// Packed weights (DenseWeights::packed) stream through the packed GEMV and,
// up to eight rows, the packed MMA tile; from nine rows the MMAs double and
// rebuilding the bf16 words costs more than the bytes it saves, so those
// steps read the bf16 weights.
//
// Microseconds with the Qwen3-ASR-1.7B-bf16 weights from DRAM
// (omni-dense-bench --real --policy; round 13's tile / MMA bf16 / MMA
// packed): eight rows' QKV 4096x2048 81.8 / 60.0 / 51.6, O 2048x2048 51.2 /
// 31.0 / 28.1, gate/up 2x6144x2048 242.7 / 180.9 / 165.4, down 2048x6144
// 162.3 / 93.5 / 82.4, the 151936x2048 head 2741 / 2127 / 1753 (the 32x64 MPP
// tile computed 32 rows in 32 to 96 threadgroups for most projections);
// two rows' 55.7 / 59.5 / 50.5, 30.7 / 30.9 / 27.1, 82.5 / 92.6 / 78.5 and
// 1844 / 2103 / 1690 (round 13's packed GEMV); the TTS code predictor's
// eight rows of O 1024x2048 35.4 / 17.7 / 17.2 and down 1024x3072 55.0 /
// 25.5 / 23.6. Sixteen rows (MPP / MMA bf16): QKV 84.4 / 70.9, O 51.8 / 40.4,
// gate/up 245.7 / 193.6, down 164.0 / 108.1, the head 2779 / 2285. Gate/up
// at two and three rows: the packed GEMV 144.7 and 158.1, the packed MMA
// tile 164. Smaller MPP tiles (8 or 16 rows, 32 or 64 columns, four
// simdgroups) with K split up to eight ways reached at most 229 GB/s.
//
// The MMA tiles split K in two, with split scratch, when they would run
// fewer than 64 threadgroups or K reaches 4096 (the decoders' down
// projections). A tile's sums of a row, in K order, do not depend on the
// step's rows nor on which copy holds the weights, so the packed copy still
// projects the bf16 bits.
struct DenseTile final {
  enum class Kind : uint8_t { Gemv, Mma, Mpp };
  Kind kind = Kind::Gemv;
  // Mpp: rows and columns a threadgroup; Gemv and Mma: columns a threadgroup.
  uint32_t rows = 0, columns = 0;
  std::string name;
  bool packed = false;
  uint32_t splits = 1;
  // Mpp: simdgroups a threadgroup.
  uint32_t simdgroups = 0;
};

// The MPP tile of `rows` rows (every tile sums an output to the same bits):
// 32x64 of eight simdgroups to 32 rows or while that makes at most 64
// threadgroups; 64x128 of four for K of 2048 and more from 48 of those;
// 64x64 of eight simdgroups up to 48 of those; else 64x64 of four.
// Microseconds (omni-dense-bench --tower, weights from DRAM; round 17's
// tiles, 32x64 to 64 rows else 64x128, eight simdgroups, in calls of 64
// inputs with a barrier after each, in brackets): the audio tower at 151
// rows QKV 3072x1024 65.8 (93.6), out 1024x1024 22.7 (36.1), fc1 4096x1024
// 87.2 (111.0), fc2 1024x4096 81.3 (119.1), conv_out 1024x7680 143.8
// (215.0), conv2 512x4352 over 6400 rows 1175.9 (1403.8); the bf16
// decoder's prefill of 166 rows QKV 4096x2048 166.2, O 2048x2048 79.7,
// gate/up 2x6144x2048 593.1, down 2048x6144 220.1. The best tile of each
// shape and row count from 24 to 2000 rows saved at most 3% over these.
DenseTile mppTile(uint32_t rows, uint32_t outputSize, uint32_t inputSize) {
  const auto tile = [](uint32_t m, uint32_t n, uint32_t simdgroups) {
    return DenseTile{DenseTile::Kind::Mpp, m, n,
                     "omni_gemm_bf16_m" + std::to_string(m) + "n" + std::to_string(n) + "s" + std::to_string(simdgroups),
                     false, 1, simdgroups};
  };
  const uint32_t rowTiles = (rows + 63) / 64;
  if (rows <= 32 || (rows + 31) / 32 * (outputSize / 64) <= 64) return tile(32, 64, 8);
  if (inputSize >= 2048 && outputSize % 128 == 0 && rowTiles * (outputSize / 128) >= 48) return tile(64, 128, 4);
  if (rowTiles * (outputSize / 64) <= 48) return tile(64, 64, 8);
  return tile(64, 64, 4);
}
constexpr uint32_t kMmaRows = 16;
constexpr uint32_t kMmaSplitThreadgroups = 64;
constexpr uint32_t kMmaSplitInputs = 4096;
DenseTile denseTile(uint32_t rows, uint32_t outputSize, uint32_t inputSize, bool gateUp, bool packed,
                    bool splittable, bool stepTiles = true) {
  if (rows == 1 || (gateUp && rows <= 3) || (!stepTiles && rows <= 4)) {
    const uint32_t padded = rows <= 1 ? 1 : rows <= 2 ? 2 : 4;
    if (packed) return {DenseTile::Kind::Gemv, 0, 8, "omni_gemv_packed_c1_r" + std::to_string(padded), true};
    const uint32_t columns = rows == 1 && !gateUp ? 2 : 1;
    return {DenseTile::Kind::Gemv, 0, 8 * columns,
            "omni_gemv_bf16_c" + std::to_string(columns) + "_r" + std::to_string(padded)};
  }
  if (stepTiles && rows <= kMmaRows) {
    const uint32_t chunks = inputSize / 64;
    // The packed tile loads two chunks an iteration.
    const bool packedTile = packed && rows <= 8 && chunks % 2 == 0;
    const uint32_t columns = rows <= 8 ? 32 : 64;
    DenseTile tile{DenseTile::Kind::Mma, 0, columns,
                   std::string(packedTile ? "omni_mma_packed" : "omni_mma_bf16") + (rows <= 8 ? "_r8" : "_r16"),
                   packedTile};
    // Splits take whole pairs of chunks whether or not the weights are
    // packed, so both copies split alike.
    const bool wanted = outputSize / columns < kMmaSplitThreadgroups || inputSize >= kMmaSplitInputs;
    if (splittable && wanted && chunks % 4 == 0) tile.splits = 2;
    return tile;
  }
  return mppTile(rows, outputSize, inputSize);
}

// The rows a tile reads and writes: the MPP tiles whole tiles of rows.
uint32_t tileRows(const DenseTile &tile, uint32_t rows) noexcept {
  return tile.kind == DenseTile::Kind::Mpp ? roundUp(rows, tile.rows) : rows;
}

uint64_t splitPartialsBytes(const DenseTile &tile, uint32_t rows, uint32_t outputSize) noexcept {
  // [split][stream][row][column] fp32, the gate's stream allocated always.
  return tile.splits > 1 ? uint64_t{tile.splits} * 2 * rows * outputSize * 4 : 0;
}
uint64_t splitCountersBytes(const DenseTile &tile, uint32_t outputSize) noexcept {
  return tile.splits > 1 ? uint64_t{outputSize / tile.columns} * 4 : 0;
}

void requireDense(const DenseWeights &w, const char *what) {
  if (!w.outputSize || !w.inputSize || w.inputSize % 64 || w.outputSize % 64)
    throw std::invalid_argument(std::string(what) + ": dense projections take whole 64-wide inputs and outputs");
  requireBytes(w.weight, uint64_t{w.outputSize} * w.inputSize * 2, "dense weight");
  if (w.packed) requireBytes(w.packed, packedDenseBytes(w.outputSize, w.inputSize, 0), "packed dense weight");
  if (w.bias) requireBytes(w.bias, uint64_t{w.outputSize} * 4, "dense bias");
}

void addDense(metal::CommandGraph &graph, const DenseWeights &weights, const DenseWeights *gate,
              const GemmBuffers &buffers, uint32_t rows, GemmOptions options) {
  requireDense(weights, "projection");
  if (gate) {
    requireDense(*gate, "gate");
    if (gate->outputSize != weights.outputSize || gate->inputSize != weights.inputSize)
      throw std::invalid_argument("gate and up projections differ in shape");
    if (static_cast<bool>(gate->bias) != static_cast<bool>(weights.bias))
      throw std::invalid_argument("gate and up projections differ in bias");
  }
  if (!rows) throw std::invalid_argument("a dense projection takes at least one row");
  const uint32_t n = weights.outputSize, k = weights.inputSize;
  const uint32_t inputStride = buffers.inputStride ? buffers.inputStride : k;
  const uint32_t outputStride = buffers.outputStride ? buffers.outputStride : n;
  if (inputStride < k || outputStride < n) throw std::invalid_argument("dense row strides are narrower than the rows");
  const DenseTile tile = denseTile(rows, n, k, gate != nullptr, weights.packed && (!gate || gate->packed),
                                   buffers.partials && buffers.counters, options.stepTiles);
  const uint32_t paddedRows = tileRows(tile, rows);
  const uint64_t elementBytes = ops::elementBytes(options.destination);
  requireBytes(buffers.input, rowBytes(paddedRows, inputStride, k, 2), "dense input");
  requireBytes(buffers.output, rowBytes(rows, outputStride, n, elementBytes), "dense output");
  if (options.residual) requireBytes(buffers.residual, rowBytes(rows, outputStride, n, elementBytes), "dense residual");
  OmniGemmParams params{};
  params.rows = rows;
  params.output_size = n;
  params.input_size = k;
  params.flags = (weights.bias ? OMNI_GEMM_FLAG_BIAS : 0u) | (options.residual ? OMNI_GEMM_FLAG_RESIDUAL : 0u) |
                 (gate ? OMNI_GEMM_FLAG_GATE_UP : 0u);
  params.activation = static_cast<uint32_t>(options.activation);
  params.input_stride = inputStride;
  params.output_stride = outputStride;
  const metal::MetalBuffer bias = weights.bias ? weights.bias : weights.weight;
  const metal::MetalBuffer weight = tile.packed ? weights.packed : weights.weight;
  const metal::MetalBuffer gateWeight = gate ? (tile.packed ? gate->packed : gate->weight) : weight;
  const metal::MetalBuffer gateBias = gate && gate->bias ? gate->bias : bias;
  const metal::MetalBuffer residual = options.residual ? buffers.residual : buffers.output;
  const bool f32 = options.destination == FloatOutput::Float32;
  std::vector<metal::MetalBuffer> bindings{buffers.input, weight, gateWeight, bias, gateBias, residual, buffers.output};
  switch (tile.kind) {
  case DenseTile::Kind::Gemv:
    graph.add(tile.name + (f32 ? "_f32" : ""), std::move(bindings), params, {(n + tile.columns - 1) / tile.columns, 1, 1});
    break;
  case DenseTile::Kind::Mma: {
    if (n % tile.columns) throw std::logic_error("dense tile columns do not divide the output");
    // Unsplit, the kernel reads neither partials nor counters: the output
    // stands in for both.
    metal::MetalBuffer partials = buffers.output, counters = buffers.output;
    if (tile.splits > 1) {
      requireBytes(buffers.partials, splitPartialsBytes(tile, rows, n), "dense split partials");
      requireBytes(buffers.counters, splitCountersBytes(tile, n), "dense split counters");
      partials = buffers.partials;
      counters = buffers.counters;
    }
    bindings.insert(bindings.end(), {partials, counters});
    graph.add(tile.name + (gate ? "_gu" : "") + (f32 ? "_f32" : ""), std::move(bindings), params,
              {n / tile.columns, tile.splits, 1}, {OMNI_MMA_THREADS, 1, 1});
    break;
  }
  case DenseTile::Kind::Mpp:
    if (n % tile.columns) throw std::logic_error("dense tile columns do not divide the output");
    graph.add(tile.name + (gate ? "_gu" : "") + (f32 ? "_f32" : ""), std::move(bindings), params,
              {paddedRows / tile.rows, n / tile.columns, 1}, {tile.simdgroups * 32, 1, 1});
    break;
  }
}

uint32_t historySplits(uint32_t visibleTokens, uint32_t maximumSplits) noexcept {
  const uint32_t pages = (visibleTokens + OMNI_PAGE_TOKENS - 1) / OMNI_PAGE_TOKENS;
  // Eight pages per split, one per simdgroup of the split kernels, so short
  // histories do not fan out and long ones keep every simdgroup busy.
  return std::clamp<uint32_t>((pages + 7) / 8, 1, maximumSplits);
}

// The split grid's depth: the most splits any sequence takes (the kernels'
// partials keep the maximum's stride; deeper threadgroups would only exit).
uint32_t activeSplits(const AttentionTables &t) noexcept {
  uint32_t splits = 1;
  for (const OmniSequenceParams &s : t.sequences) splits = std::max(splits, s.splits);
  return splits;
}

void requireAttention(const AttentionBuffers &b, const AttentionTables &t, uint32_t pageTableEntries,
                      uint32_t pageCount) {
  if (t.sequences.empty() || !t.rows) throw std::invalid_argument("an attention dispatch takes rows");
  if (t.sequences.size() > OMNI_MAXIMUM_LANES * 64) throw std::invalid_argument("too many attention sequences");
  requireBytes(b.sequences, t.sequences.size() * sizeof(OmniSequenceParams), "attention sequence");
  requireBytes(b.pageTable, uint64_t{pageTableEntries} * 4, "page table");
  requireBytes(b.pages, uint64_t{pageCount} * OMNI_PAGE_BYTES, "KV page");
  requireBytes(b.queries, uint64_t{kKvHeads} * t.paddedRows * kGroup * kHeadDim * 2, "attention query");
  for (const OmniSequenceParams &s : t.sequences) {
    const uint32_t blocks = (s.committed_tokens + s.rows + OMNI_PAGE_TOKENS - 1) / OMNI_PAGE_TOKENS;
    if (!s.rows || s.row_offset + s.rows > t.rows) throw std::invalid_argument("attention sequence rows are outside the dispatch");
    if (blocks > s.page_table_stride) throw std::invalid_argument("attention sequence exceeds its page table");
    if (uint64_t{s.page_table_offset} + blocks > pageTableEntries)
      throw std::invalid_argument("attention sequence page table is outside the table");
  }
}

// A fused decode step (one row per sequence) or verify forward of at most
// kQueryHeadRows rows runs a threadgroup per query head instead of per KV
// head (the same bits): few rows leave most of the GPU idle, and each
// simdgroup's page loop then scores one vector instead of two. Every
// layer's attention at a 200-token history (omni_asr_verify_bench, M5 Pro,
// GPU ms over 28 layers): decode at 1 / 2 / 4 lanes 0.30 -> 0.17, 0.34 ->
// 0.26, 0.47 -> 0.47; verify of one lane's 1 / 2 / 3 / 4 rows 0.30 -> 0.17,
// 0.30 -> 0.19, 0.31 -> 0.23, 0.31 -> 0.28, but 5 and 6 rows 0.33 -> 0.38
// and 0.34 -> 0.47.
constexpr uint32_t kQueryHeadRows = 4;
bool queryHeadAttention(uint32_t rows) noexcept { return rows <= kQueryHeadRows; }

// A decode step's attention past OMNI_FUSED_MAXIMUM_PAGES pages. Chained
// 4-bit ASR forwards (omni_asr_verify_bench --decode-histories, M5 Pro, GPU
// ms a step; prepare + split + reduce / the fused kernel / the long kernel):
// one lane at 512 tokens 5.06 / 4.94 / 4.94, 1000 5.20 / 5.13 / 5.15, 4000
// 6.79 / 6.52 / 6.42; two lanes at 530 5.26 / 5.19, four 5.98 / 5.69, eight
// 7.00 / 6.84, and the fused kernel stays ahead at several lanes to 4000
// tokens and when one lane is long and the others short (one lane at 530
// and three at 150: 5.70 / 5.44). So a lone row takes the long kernel (one
// dispatch where prepare, split and reduce take three, a threadgroup per
// query head and split) and several rows the fused kernel at any history.
// Prepare, split and reduce (Attention::addDecode) no longer run in a
// forward; omni_kernels_test checks the long kernel's bits against them.

uint32_t pagesOf(uint32_t visibleTokens) noexcept { return (visibleTokens + OMNI_PAGE_TOKENS - 1) / OMNI_PAGE_TOKENS; }

// Whether a decode step of `sequences` sequences of `rows` rows each, the
// longest over `pages` pages, runs the long kernel.
bool longDecode(size_t sequences, uint32_t rows, uint32_t pages) noexcept {
  return sequences == 1 && rows == 1 && pages > OMNI_FUSED_MAXIMUM_PAGES;
}

// The long kernel's threadgroups each run a simdgroup page loop: a round of
// one page per simdgroup is fastest up to three splits (24 pages; 512 tokens
// 4.92 ms against 4.96 in two splits), and past that two splits of longer
// loops beat more splits of one round (1000 tokens 5.11 against 5.14 in
// four, 1500 5.31 against 5.54 in six, 4000 6.39 against 6.82 in sixteen).
// Up to 24 pages a split's pages, and so the arithmetic, are those of
// prepare, split and reduce.
constexpr uint32_t kLongRoundPages = 24;

// A stored group's position p holds weight kPackedOrder[p] of its eight.
constexpr uint32_t kPackedOrder[8] = {0, 2, 1, 3, 4, 6, 5, 7};
constexpr uint32_t kPackedBlock = 64;
// Escape record indices fill a header's bits 31:8.
constexpr uint64_t kMaximumEscapes = uint64_t{1} << 24;

void requirePackable(uint64_t elements, uint32_t outputSize, uint32_t inputSize) {
  if (!outputSize || !inputSize || inputSize % kPackedBlock || outputSize % kPackedBlock)
    throw std::invalid_argument("packed weights take whole 64-wide inputs and outputs");
  if (elements != uint64_t{outputSize} * inputSize) throw std::invalid_argument("packed weight extent mismatch");
}

uint32_t exponentOf(uint16_t bits) noexcept { return (bits >> 7) & 255u; }

// The base exponent of a block of 64 weights, or OMNI_PACKED_ESCAPE.
uint32_t blockBase(const uint16_t *block) noexcept {
  uint32_t least = 255, most = 0;
  for (uint32_t i = 0; i < kPackedBlock; ++i) {
    least = std::min(least, exponentOf(block[i]));
    most = std::max(most, exponentOf(block[i]));
  }
  return most == 255 || most - least > 15 ? OMNI_PACKED_ESCAPE : least;
}

struct PackedPlanes final {
  uint64_t codes, headers, escapes;
};
PackedPlanes packedPlanes(uint32_t outputSize, uint32_t inputSize) noexcept {
  const uint64_t weights = uint64_t{outputSize} * inputSize;
  return {weights, weights * 3 / 2, weights * 3 / 2 + weights / 16};
}

} // namespace

uint32_t packedEscapeBlocks(std::span<const uint16_t> rows, uint32_t inputSize) {
  if (!inputSize || inputSize % kPackedBlock || rows.size() % inputSize)
    throw std::invalid_argument("packed weights take whole 64-wide rows");
  uint64_t escapes = 0;
  for (uint64_t at = 0; at < rows.size(); at += kPackedBlock)
    escapes += blockBase(rows.data() + at) == OMNI_PACKED_ESCAPE;
  if (escapes >= kMaximumEscapes) throw std::invalid_argument("a packed weight holds too many escape blocks");
  return static_cast<uint32_t>(escapes);
}

uint64_t packedDenseBytes(uint32_t outputSize, uint32_t inputSize, uint64_t escapeBlocks) {
  if (escapeBlocks >= kMaximumEscapes) throw std::invalid_argument("a packed weight holds too many escape blocks");
  return packedPlanes(outputSize, inputSize).escapes + escapeBlocks * OMNI_PACKED_ESCAPE_BYTES;
}

void packDenseRows(std::span<const uint16_t> weights, uint32_t outputSize, uint32_t inputSize, uint32_t first,
                   uint32_t count, uint32_t firstEscape, std::span<uint8_t> packed) {
  requirePackable(weights.size(), outputSize, inputSize);
  if (uint64_t{first} + count > outputSize) throw std::invalid_argument("packed rows are outside the weight");
  const PackedPlanes planes = packedPlanes(outputSize, inputSize);
  if (packed.size() < planes.escapes) throw std::invalid_argument("packed weight buffer is too small");
  uint64_t escape = firstEscape;
  const uint64_t end = (uint64_t{first} + count) * inputSize / kPackedBlock;
  for (uint64_t b = uint64_t{first} * inputSize / kPackedBlock; b < end; ++b) {
    const uint16_t *block = weights.data() + b * kPackedBlock;
    const uint32_t base = blockBase(block);
    uint32_t header = base;
    uint8_t *record = nullptr;
    if (base == OMNI_PACKED_ESCAPE) {
      if (packedDenseBytes(outputSize, inputSize, escape + 1) > packed.size())
        throw std::invalid_argument("packed weight buffer is too small for its escape blocks");
      header |= static_cast<uint32_t>(escape) << 8;
      record = packed.data() + planes.escapes + escape * OMNI_PACKED_ESCAPE_BYTES;
      ++escape;
    }
    std::memcpy(packed.data() + planes.headers + b * 4, &header, 4);
    for (uint32_t group = 0; group < kPackedBlock / 8; ++group) {
      const uint16_t *w = block + group * 8;
      const uint64_t at = b * kPackedBlock + group * 8;
      for (uint32_t p = 0; p < 8; ++p) {
        const uint16_t bits = w[kPackedOrder[p]];
        packed[at + p] = static_cast<uint8_t>(((bits >> 8) & 0x80u) | (bits & 0x7fu));
        if (record) record[group * 8 + p] = static_cast<uint8_t>(exponentOf(bits));
      }
      // Weights 0 2 4 6 in the low 16 bits' nibbles, then 1 3 5 7.
      uint32_t code = 0;
      for (uint32_t i = 0; i < 8 && !record; ++i) code |= (exponentOf(w[i]) - base) << ((i % 2 ? 16 : 0) + 4 * (i / 2));
      std::memcpy(packed.data() + planes.codes + b * 32 + group * 4, &code, 4);
    }
  }
}

void unpackDense(std::span<const uint8_t> packed, uint32_t outputSize, uint32_t inputSize,
                 std::span<uint16_t> weights) {
  requirePackable(weights.size(), outputSize, inputSize);
  const PackedPlanes planes = packedPlanes(outputSize, inputSize);
  if (packed.size() < planes.escapes) throw std::invalid_argument("packed weight buffer is too small");
  for (uint64_t b = 0; b < weights.size() / kPackedBlock; ++b) {
    uint32_t header;
    std::memcpy(&header, packed.data() + planes.headers + b * 4, 4);
    const uint32_t base = header & 255u;
    const uint64_t record = planes.escapes + uint64_t{header >> 8} * OMNI_PACKED_ESCAPE_BYTES;
    if (base == OMNI_PACKED_ESCAPE && record + OMNI_PACKED_ESCAPE_BYTES > packed.size())
      throw std::invalid_argument("packed escape record is outside the buffer");
    for (uint32_t group = 0; group < kPackedBlock / 8; ++group) {
      uint32_t code;
      std::memcpy(&code, packed.data() + planes.codes + b * 32 + group * 4, 4);
      for (uint32_t p = 0; p < 8; ++p) {
        const uint32_t i = kPackedOrder[p];
        const uint32_t sm = packed[b * kPackedBlock + group * 8 + p];
        const uint32_t exponent = base == OMNI_PACKED_ESCAPE
                                      ? packed[record + group * 8 + p]
                                      : base + ((code >> ((i % 2 ? 16 : 0) + 4 * (i / 2))) & 15u);
        weights[b * kPackedBlock + group * 8 + i] = static_cast<uint16_t>(((sm & 0x80u) << 8) | (exponent << 7) | (sm & 0x7fu));
      }
    }
  }
}

uint32_t densePaddedRows(uint32_t rows) noexcept {
  if (rows <= kMmaRows) return rows;
  return roundUp(rows, 64);
}

DenseSplitScratch &DenseSplitScratch::include(const DenseSplitScratch &other) noexcept {
  partialsBytes = std::max(partialsBytes, other.partialsBytes);
  countersBytes = std::max(countersBytes, other.countersBytes);
  return *this;
}

DenseSplitScratch denseSplitScratch(uint32_t outputSize, uint32_t inputSize, uint32_t maxRows) {
  DenseSplitScratch size;
  for (uint32_t rows = 1; rows <= std::min(maxRows, kMmaRows); ++rows)
    for (const bool gateUp : {false, true})
      for (const bool packed : {false, true}) {
        const DenseTile tile = denseTile(rows, outputSize, inputSize, gateUp, packed, true);
        size.include({splitPartialsBytes(tile, rows, outputSize), splitCountersBytes(tile, outputSize)});
      }
  return size;
}

bool DenseLinear::sameRowArithmetic(const DenseWeights &weights, uint32_t rows, uint32_t otherRows) {
  const uint32_t n = weights.outputSize, k = weights.inputSize;
  const bool packed = static_cast<bool>(weights.packed);
  const DenseTile tile = denseTile(rows, n, k, false, packed, true);
  const DenseTile other = denseTile(otherRows, n, k, false, packed, true);
  if (tile.kind != other.kind) return false;
  if (tile.kind == DenseTile::Kind::Mma) return tile.splits == other.splits;
  return tile.kind == DenseTile::Kind::Gemv || tile.name == other.name;
}

void DenseLinear::add(metal::CommandGraph &graph, const DenseWeights &weights, const GemmBuffers &buffers,
                      uint32_t rows, GemmOptions options) {
  addDense(graph, weights, nullptr, buffers, rows, options);
}

void DenseLinear::addGateUp(metal::CommandGraph &graph, const DenseWeights &gate, const DenseWeights &up,
                            const GemmBuffers &buffers, uint32_t rows, GemmOptions options) {
  addDense(graph, up, &gate, buffers, rows, options);
}

AttentionTables attentionTables(std::span<const AttentionSequence> sequences, bool decode) {
  AttentionTables tables;
  uint32_t rows = 0;
  for (const AttentionSequence &s : sequences) {
    if (!s.rows || s.rowOffset != rows) throw std::invalid_argument("attention sequences must be contiguous rows");
    if (decode && (s.rows < 1 || s.rows > 2)) throw std::invalid_argument("a decode sequence is one or two rows");
    OmniSequenceParams params{};
    params.page_table_offset = s.pageTableOffset;
    params.page_table_stride = s.pageTableStride;
    params.committed_tokens = s.committedTokens;
    params.rows = s.rows;
    params.row_offset = s.rowOffset;
    params.position_offset = s.positionOffset;
    params.splits = historySplits(s.committedTokens + s.rows,
                                  decode ? OMNI_DECODE_MAXIMUM_SPLITS : OMNI_PREFILL_MAXIMUM_SPLITS);
    const uint32_t pages = pagesOf(s.committedTokens + s.rows);
    if (decode && longDecode(sequences.size(), s.rows, pages) && pages > kLongRoundPages) params.splits = 2;
    tables.sequences.push_back(params);
    const uint32_t sequence = static_cast<uint32_t>(tables.sequences.size() - 1);
    for (uint32_t first = 0; first < s.rows; first += kTileRows)
      tables.tiles.push_back({sequence, s.rowOffset + first, std::min(kTileRows, s.rows - first), 0});
    rows += s.rows;
  }
  tables.rows = rows;
  tables.paddedRows = roundUp(rows, kTileRows);
  return tables;
}

AttentionWorkspace Attention::prefillWorkspace(uint32_t paddedRows, uint32_t tiles) noexcept {
  const uint64_t slots = uint64_t{tiles} * kKvHeads * OMNI_PREFILL_MAXIMUM_SPLITS;
  const uint64_t fused = kTileRows * kGroup;
  return {uint64_t{kKvHeads} * paddedRows * kGroup * kHeadDim * 2, slots * fused * kHeadDim * 4, slots * fused * 2 * 4};
}

AttentionWorkspace Attention::decodeWorkspace(uint32_t lanes) noexcept {
  const uint64_t slots = uint64_t{lanes} * kKvHeads * OMNI_DECODE_MAXIMUM_SPLITS;
  return {uint64_t{kKvHeads} * roundUp(lanes, kTileRows) * kGroup * kHeadDim * 2, slots * kGroup * kHeadDim * 4,
          slots * kGroup * 2 * 4, uint64_t{lanes} * kKvHeads * kGroup * 4};
}

void Attention::addPrepare(metal::CommandGraph &graph, const AttentionBuffers &b, const AttentionTables &t,
                           uint32_t layer, float rmsEpsilon, uint32_t pageTableEntries, uint32_t pageCount) {
  requireAttention(b, t, pageTableEntries, pageCount);
  requireBytes(b.qkv, uint64_t{t.rows} * kQkvWidth * 2, "attention qkv");
  requireBytes(b.queryNorm, kHeadDim * 2, "query norm");
  requireBytes(b.keyNorm, kHeadDim * 2, "key norm");
  requireBytes(b.inverseFrequencies, kHeadDim / 2 * 4, "inverse frequency");
  requireBytes(b.rowSequence, uint64_t{t.rows} * 4, "row sequence");
  const OmniQkvParams params{t.rows, layer, static_cast<uint32_t>(t.sequences.size()), rmsEpsilon,
                             OMNI_ROTATIONS_COMPUTE};
  graph.add("omni_qkv_prepare",
            {b.qkv, b.queryNorm, b.keyNorm, b.inverseFrequencies, b.rowSequence, b.sequences, b.pageTable, b.pages,
             b.queries},
            params, {uint64_t{t.rows} * (OMNI_DECODER_QUERY_HEADS + kKvHeads), 1, 1}, {kHeadDim, 1, 1});
}

void Attention::addPrefill(metal::CommandGraph &graph, const AttentionBuffers &b, const AttentionTables &t,
                           uint32_t layer, uint32_t pageTableEntries, uint32_t pageCount) {
  requireAttention(b, t, pageTableEntries, pageCount);
  const auto tiles = static_cast<uint32_t>(t.tiles.size());
  const AttentionWorkspace workspace = prefillWorkspace(t.paddedRows, tiles);
  requireBytes(b.tiles, uint64_t{tiles} * sizeof(OmniPrefillTile), "attention tile");
  requireBytes(b.partials, workspace.partialsBytes, "attention partial");
  requireBytes(b.statistics, workspace.statisticsBytes, "attention statistic");
  requireBytes(b.output, uint64_t{t.rows} * kAttentionWidth * 2, "attention output");
  const OmniAttentionParams params{layer, tiles, static_cast<uint32_t>(t.sequences.size()), t.paddedRows};
  graph.add("omni_attention_prefill_split",
            {b.queries, b.pageTable, b.pages, b.sequences, b.tiles, b.partials, b.statistics}, params,
            {kKvHeads, tiles, activeSplits(t)});
  graph.add("omni_attention_prefill_reduce", {b.partials, b.statistics, b.output, b.sequences, b.tiles}, params,
            {kKvHeads, tiles, kTileRows * kGroup}, {kHeadDim, 1, 1});
}

void Attention::addDecode(metal::CommandGraph &graph, const AttentionBuffers &b, const AttentionTables &t,
                          uint32_t layer, uint32_t pageTableEntries, uint32_t pageCount) {
  requireAttention(b, t, pageTableEntries, pageCount);
  const auto lanes = static_cast<uint32_t>(t.sequences.size());
  if (lanes != t.rows) throw std::invalid_argument("decode attention takes one row per sequence");
  const AttentionWorkspace workspace = decodeWorkspace(lanes);
  requireBytes(b.partials, workspace.partialsBytes, "attention partial");
  requireBytes(b.statistics, workspace.statisticsBytes, "attention statistic");
  requireBytes(b.output, uint64_t{t.rows} * kAttentionWidth * 2, "attention output");
  const OmniAttentionParams params{layer, 0, lanes, t.paddedRows};
  graph.add("omni_attention_decode_split", {b.queries, b.pageTable, b.pages, b.sequences, b.partials, b.statistics},
            params, {kKvHeads, lanes, activeSplits(t)});
  graph.add("omni_attention_decode_reduce", {b.partials, b.statistics, b.output, b.sequences}, params,
            {kKvHeads, lanes, kGroup}, {kHeadDim, 1, 1});
}

bool Attention::fusable(const AttentionTables &t) noexcept {
  if (t.sequences.empty()) return false;
  const uint32_t rowsPerSequence = t.sequences.front().rows;
  if (rowsPerSequence < 1 || rowsPerSequence > 2 || t.rows != t.sequences.size() * rowsPerSequence) return false;
  for (const OmniSequenceParams &s : t.sequences) {
    if (s.rows != rowsPerSequence) return false;
    const uint32_t pages = pagesOf(s.committed_tokens + s.rows);
    if (longDecode(t.sequences.size(), s.rows, pages)) return false;
  }
  return true;
}

bool Attention::longFusable(const AttentionTables &t) noexcept {
  return t.sequences.size() == 1 && t.rows == 1 &&
         longDecode(1, 1, pagesOf(t.sequences.front().committed_tokens + 1));
}

void Attention::addDecodeLong(metal::CommandGraph &graph, const AttentionBuffers &b, const AttentionTables &t,
                              uint32_t layer, float rmsEpsilon, uint32_t pageTableEntries, uint32_t pageCount,
                              Rotations rotations) {
  requireAttention(b, t, pageTableEntries, pageCount);
  if (!longFusable(t)) throw std::invalid_argument("decode step does not fit the long decode attention");
  const AttentionWorkspace workspace = decodeWorkspace(1);
  requireBytes(b.qkv, uint64_t{t.rows} * kQkvWidth * 2, "attention qkv");
  requireBytes(b.queryNorm, kHeadDim * 2, "query norm");
  requireBytes(b.keyNorm, kHeadDim * 2, "key norm");
  requireBytes(b.inverseFrequencies, kHeadDim / 2 * 4, "inverse frequency");
  requireBytes(b.partials, workspace.partialsBytes, "attention partial");
  requireBytes(b.statistics, workspace.statisticsBytes, "attention statistic");
  requireBytes(b.counters, workspace.countersBytes, "attention counter");
  requireBytes(b.output, uint64_t{t.rows} * kAttentionWidth * 2, "attention output");
  if (rotations != Rotations::Compute) requireBytes(b.rotations, rotationBytes(t.rows), "attention rotation table");
  const OmniQkvParams params{1, layer, 1, rmsEpsilon, static_cast<uint32_t>(rotations)};
  graph.add("omni_attention_decode_long",
            {b.qkv, b.queryNorm, b.keyNorm, b.inverseFrequencies, b.sequences, b.pageTable, b.pages, b.output,
             rotations == Rotations::Compute ? b.output : b.rotations, b.partials, b.statistics, b.counters},
            params, {kKvHeads * kGroup, 1, activeSplits(t)});
}

bool Attention::fitsOnePage(const AttentionTables &t) noexcept {
  for (const OmniSequenceParams &s : t.sequences)
    if (s.committed_tokens + s.rows > OMNI_PAGE_TOKENS) return false;
  return !t.sequences.empty();
}

void Attention::addDecodeFused(metal::CommandGraph &graph, const AttentionBuffers &b, const AttentionTables &t,
                               uint32_t layer, float rmsEpsilon, uint32_t pageTableEntries, uint32_t pageCount,
                               Rotations rotations, bool kvHeads) {
  requireAttention(b, t, pageTableEntries, pageCount);
  if (!fusable(t)) throw std::invalid_argument("decode step is not fusable");
  const auto lanes = static_cast<uint32_t>(t.sequences.size());
  requireBytes(b.qkv, uint64_t{t.rows} * kQkvWidth * 2, "attention qkv");
  requireBytes(b.queryNorm, kHeadDim * 2, "query norm");
  requireBytes(b.keyNorm, kHeadDim * 2, "key norm");
  requireBytes(b.inverseFrequencies, kHeadDim / 2 * 4, "inverse frequency");
  requireBytes(b.output, uint64_t{t.rows} * kAttentionWidth * 2, "attention output");
  // Computing rotations reads no table; the output stands in for it.
  if (rotations != Rotations::Compute) requireBytes(b.rotations, rotationBytes(t.rows), "attention rotation table");
  const OmniQkvParams params{t.rows, layer, lanes, rmsEpsilon, static_cast<uint32_t>(rotations)};
  const bool pair = t.sequences.front().rows == 2;
  const char *kernel = pair ? "omni_attention_decode_fused_r2" : "omni_attention_decode_fused";
  // The one-page kernel runs four simdgroups per row.
  uint32_t threads = metal::CommandGraph::kDefaultThreads;
  uint32_t heads = kKvHeads;
  if (!kvHeads && fitsOnePage(t)) {
    kernel = pair ? "omni_attention_decode_page_r2" : "omni_attention_decode_page";
    threads = 128 * t.sequences.front().rows;
  } else if (!pair && !kvHeads && queryHeadAttention(t.rows)) {
    kernel = "omni_attention_decode_fused_head";
    heads = kKvHeads * kGroup;
  }
  graph.add(kernel,
            {b.qkv, b.queryNorm, b.keyNorm, b.inverseFrequencies, b.sequences, b.pageTable, b.pages, b.output,
             rotations == Rotations::Compute ? b.output : b.rotations},
            params, {heads, lanes, 1}, {threads, 1, 1});
}

// A verify row's threadgroup takes its history in rounds of 8 pages, one
// per simdgroup, about 7 us a round at one lane; the prefill kernels split
// the history over more threadgroups. The fused kernel wins while the rows
// times the rounds stay within OMNI_VERIFY_MAXIMUM_ROWS, and loses beyond
// 16 pages (dev/benchmarks/omni_asr_verify_bench.mm).
bool Attention::verifyFusable(const AttentionTables &t) noexcept {
  if (t.sequences.empty()) return false;
  uint32_t pages = 0;
  for (const OmniSequenceParams &s : t.sequences) {
    if (!s.rows) return false;
    pages = std::max(pages, (s.committed_tokens + s.rows + OMNI_PAGE_TOKENS - 1) / OMNI_PAGE_TOKENS);
  }
  return pages <= OMNI_VERIFY_MAXIMUM_PAGES && t.rows * ((pages + 7) / 8) <= OMNI_VERIFY_MAXIMUM_ROWS;
}

void Attention::addVerifyFused(metal::CommandGraph &graph, const AttentionBuffers &b, const AttentionTables &t,
                               uint32_t layer, float rmsEpsilon, uint32_t pageTableEntries, uint32_t pageCount,
                               Rotations rotations, bool kvHeads) {
  requireAttention(b, t, pageTableEntries, pageCount);
  if (!verifyFusable(t)) throw std::invalid_argument("verify forward is not fusable");
  const auto sequences = static_cast<uint32_t>(t.sequences.size());
  requireBytes(b.qkv, uint64_t{t.rows} * kQkvWidth * 2, "attention qkv");
  requireBytes(b.queryNorm, kHeadDim * 2, "query norm");
  requireBytes(b.keyNorm, kHeadDim * 2, "key norm");
  requireBytes(b.inverseFrequencies, kHeadDim / 2 * 4, "inverse frequency");
  requireBytes(b.output, uint64_t{t.rows} * kAttentionWidth * 2, "attention output");
  if (rotations != Rotations::Compute) requireBytes(b.rotations, rotationBytes(t.rows), "attention rotation table");
  uint32_t most = 0;
  for (const OmniSequenceParams &s : t.sequences) most = std::max(most, s.rows);
  const OmniQkvParams params{t.rows, layer, sequences, rmsEpsilon, static_cast<uint32_t>(rotations)};
  const bool byQueryHead = !kvHeads && queryHeadAttention(t.rows);
  graph.add(byQueryHead ? "omni_attention_verify_fused_head" : "omni_attention_verify_fused",
            {b.qkv, b.queryNorm, b.keyNorm, b.inverseFrequencies, b.sequences, b.pageTable, b.pages, b.output,
             rotations == Rotations::Compute ? b.output : b.rotations},
            params, {byQueryHead ? kKvHeads * kGroup : kKvHeads, sequences, most});
}

void DenseAttention::add(metal::CommandGraph &graph, const DenseAttentionBuffers &b, uint32_t rows, uint32_t keyRows,
                         uint32_t stride, DenseMask mask, uint32_t window, float scale, uint32_t heads) {
  const uint32_t width = heads * OMNI_ENCODER_HEAD_DIM;
  if (!rows || !heads || heads > OMNI_ENCODER_HEADS || stride < width || keyRows % OMNI_DENSE_KEY_BLOCK)
    throw std::invalid_argument("dense attention takes 1..16 heads, rows of at least heads * 64 and keys in whole "
                                "blocks of 32");
  const uint32_t paddedRows = roundUp(rows, OMNI_DENSE_TILE_ROWS);
  requireBytes(b.queries, rowBytes(paddedRows, stride, width, 2), "dense attention query");
  requireBytes(b.keys, rowBytes(keyRows, stride, width, 2), "dense attention key");
  requireBytes(b.values, rowBytes(keyRows, stride, width, 2), "dense attention value");
  requireBytes(b.output, uint64_t{rows} * width * 2, "dense attention output");
  requireBytes(b.rows, uint64_t{rows} * sizeof(OmniDenseRow), "dense attention row");
  OmniDenseParams params{};
  params.rows = rows;
  params.qkv_stride = stride;
  params.mask = static_cast<uint32_t>(mask);
  params.window = window;
  params.scale = scale;
  params.heads = heads;
  graph.add("omni_attention_dense", {b.queries, b.keys, b.values, b.output, b.rows}, params,
            {paddedRows / OMNI_DENSE_TILE_ROWS, heads, 1});
}

void Norms::addRms(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer weight,
                   metal::MetalBuffer output, uint32_t rows, uint32_t width, float epsilon, bool float32) {
  if (!rows || !width) throw std::invalid_argument("a norm takes rows and a width");
  const uint64_t element = float32 ? 4 : 2;
  requireBytes(input, uint64_t{rows} * width * element, "norm input");
  requireBytes(output, uint64_t{rows} * width * element, "norm output");
  requireBytes(weight, uint64_t{width} * element, "norm weight");
  const OmniNormParams params{rows, width, epsilon, 0};
  graph.add(float32 ? "omni_rms_norm_f32" : "omni_rms_norm_bf16", {std::move(input), std::move(weight), std::move(output)},
            params, {rows, 1, 1});
}

void Norms::addLayer(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer weight,
                     metal::MetalBuffer bias, metal::MetalBuffer output, uint32_t rows, uint32_t width,
                     float epsilon, bool float32Weights) {
  if (!rows || !width) throw std::invalid_argument("a norm takes rows and a width");
  const uint64_t element = float32Weights ? 4 : 2;
  requireBytes(input, uint64_t{rows} * width * 4, "norm input");
  requireBytes(output, uint64_t{rows} * width * element, "norm output");
  requireBytes(weight, uint64_t{width} * element, "norm weight");
  requireBytes(bias, uint64_t{width} * element, "norm bias");
  const OmniNormParams params{rows, width, epsilon, 0};
  graph.add(float32Weights ? "omni_layer_norm_f32" : "omni_layer_norm_f32_bf16",
            {std::move(input), std::move(weight), std::move(bias), std::move(output)}, params, {rows, 1, 1});
}

void Norms::addActivate(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer output,
                        uint64_t count, Activation activation, bool float32) {
  if (!count || count > UINT32_MAX) throw std::invalid_argument("an activation takes a count below 2^32");
  const uint64_t element = float32 ? 4 : 2;
  requireBytes(input, count * element, "activation input");
  requireBytes(output, count * element, "activation output");
  const OmniElementwiseParams params{static_cast<uint32_t>(count), static_cast<uint32_t>(activation)};
  graph.add(float32 ? "omni_activate_f32" : "omni_activate_bf16", {std::move(input), std::move(output)}, params,
            {(count + 255) / 256, 1, 1});
}

void Norms::addConvert(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer output,
                       uint64_t count, bool toFloat32) {
  if (!count || count > UINT32_MAX) throw std::invalid_argument("a conversion takes a count below 2^32");
  requireBytes(input, count * (toFloat32 ? 2 : 4), "conversion input");
  requireBytes(output, count * (toFloat32 ? 4 : 2), "conversion output");
  const OmniElementwiseParams params{static_cast<uint32_t>(count), 0};
  graph.add(toFloat32 ? "omni_convert_bf16_f32" : "omni_convert_f32_bf16", {std::move(input), std::move(output)},
            params, {(count + 255) / 256, 1, 1});
}

void Norms::addBiasActivate(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer bias,
                            metal::MetalBuffer output, uint32_t rows, uint32_t width, Activation activation) {
  if (!rows || !width) throw std::invalid_argument("a bias takes rows and a width");
  requireBytes(input, uint64_t{rows} * width * 2, "bias input");
  requireBytes(output, uint64_t{rows} * width * 2, "bias output");
  requireBytes(bias, uint64_t{width} * 4, "bias");
  const OmniBiasParams params{rows, width, static_cast<uint32_t>(activation), 0};
  graph.add("omni_bias_activate", {std::move(input), std::move(bias), std::move(output)}, params,
            {(uint64_t{rows} * width + 255) / 256, 1, 1});
}

void Embedding::addGather(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer table,
                          metal::MetalBuffer output, uint32_t rows, uint32_t vocabulary, uint32_t width,
                          uint32_t outputRowStride, uint32_t tokenStride) {
  if (!rows || !vocabulary || !width) throw std::invalid_argument("a gather takes rows, a vocabulary and a width");
  const uint32_t tokenStep = std::max(tokenStride, 1u);
  requireBytes(tokens, rowBytes(rows, tokenStep, 1, 4), "embedding token");
  requireBytes(table, uint64_t{vocabulary} * width * 2, "embedding table");
  const uint32_t stride = std::max(outputRowStride, 1u);
  requireBytes(output, rowBytes(rows, uint64_t{stride} * width, width, 2), "embedding output");
  const OmniEmbedParams params{rows, vocabulary, width, stride, tokenStep, {0, 0, 0}};
  const uint64_t elements = uint64_t{rows} * width;
  graph.add("omni_embed_bf16", {std::move(tokens), std::move(table), std::move(output)}, params,
            {(elements + 255) / 256, 1, 1});
}

void Embedding::addGatherRows(metal::CommandGraph &graph, metal::MetalBuffer rowIndex, metal::MetalBuffer source,
                              metal::MetalBuffer output, uint32_t rows, uint32_t sourceRows, uint32_t width,
                              uint32_t outputRowStride) {
  if (!rows || !sourceRows || !width) throw std::invalid_argument("a row gather takes rows and a width");
  requireBytes(rowIndex, uint64_t{rows} * 4, "row index");
  requireBytes(source, uint64_t{sourceRows} * width * 2, "row source");
  const uint32_t stride = std::max(outputRowStride, 1u);
  requireBytes(output, rowBytes(rows, uint64_t{stride} * width, width, 2), "row output");
  const OmniEmbedParams params{rows, sourceRows, width, stride, 1, {0, 0, 0}};
  const uint64_t elements = uint64_t{rows} * width;
  graph.add("omni_gather_rows_bf16", {std::move(rowIndex), std::move(source), std::move(output)}, params,
            {(elements + 255) / 256, 1, 1});
}

namespace {
void requireTtsTables(const Embedding::TtsTables &t) {
  if (!t.codecVocabulary || !t.predictorVocabulary || !t.width) throw std::invalid_argument("TTS tables need sizes");
  requireBytes(t.codec, uint64_t{t.codecVocabulary} * t.width * 2, "codec table");
  requireBytes(t.predictor, uint64_t{OMNI_CODE_GROUPS - 1} * t.predictorVocabulary * t.width * 2, "predictor table");
}
} // namespace

void Embedding::addTtsStepInput(metal::CommandGraph &graph, metal::MetalBuffer codes, const TtsTables &tables,
                                metal::MetalBuffer trailingRows, metal::MetalBuffer trailingPool,
                                uint32_t trailingPoolRows, metal::MetalBuffer output, uint32_t rows) {
  requireTtsTables(tables);
  if (!rows) throw std::invalid_argument("a step input takes rows");
  requireBytes(codes, uint64_t{rows} * OMNI_CODE_GROUPS * 4, "step code");
  requireBytes(trailingRows, uint64_t{rows} * 4, "trailing row");
  requireBytes(trailingPool, uint64_t{trailingPoolRows} * tables.width * 2, "trailing pool");
  requireBytes(output, uint64_t{rows} * tables.width * 2, "step output");
  const OmniTtsStepParams params{rows, tables.width, tables.codecVocabulary, tables.predictorVocabulary};
  const uint64_t elements = uint64_t{rows} * tables.width;
  graph.add("omni_tts_step_input",
            {std::move(codes), tables.codec, tables.predictor, std::move(trailingRows), std::move(trailingPool),
             std::move(output)},
            params, {(elements + 255) / 256, 1, 1});
}

void Embedding::addPromptAssemble(metal::CommandGraph &graph, metal::MetalBuffer positions, metal::MetalBuffer textRows,
                                  uint32_t textRowCount, const TtsTables &tables, metal::MetalBuffer speaker,
                                  metal::MetalBuffer frames, uint32_t frameCount, metal::MetalBuffer output,
                                  uint32_t rows) {
  requireTtsTables(tables);
  if (!rows) throw std::invalid_argument("a prompt takes rows");
  requireBytes(positions, uint64_t{rows} * sizeof(OmniPromptPosition), "prompt position");
  requireBytes(textRows, uint64_t{textRowCount} * tables.width * 2, "prompt text row");
  requireBytes(output, uint64_t{rows} * tables.width * 2, "prompt output");
  if (frameCount) requireBytes(frames, uint64_t{frameCount} * OMNI_CODE_GROUPS * 4, "prompt frame");
  if (speaker) requireBytes(speaker, uint64_t{tables.width} * 2, "speaker row");
  const metal::MetalBuffer speakerRow = speaker ? speaker : tables.codec;
  const metal::MetalBuffer frameCodes = frameCount ? frames : positions;
  const OmniTtsStepParams params{rows, tables.width, tables.codecVocabulary, tables.predictorVocabulary};
  const uint64_t elements = uint64_t{rows} * tables.width;
  graph.add("omni_prompt_assemble",
            {std::move(positions), std::move(textRows), tables.codec, tables.predictor, speakerRow, frameCodes,
             std::move(output)},
            params, {(elements + 255) / 256, 1, 1});
}

void Sampling::addSampleCodes(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer history,
                              metal::MetalBuffer historyCounts, metal::MetalBuffer uniforms,
                              metal::MetalBuffer output, const OmniSampleParams &params) {
  if (!params.rows || !params.vocabulary || params.vocabulary > OMNI_SAMPLE_MAXIMUM_VOCABULARY)
    throw std::invalid_argument("code sampling takes rows of at most 4096 ids");
  if (params.top_k > OMNI_SAMPLE_MAXIMUM_TOP_K) throw std::invalid_argument("code sampling keeps at most 256 ids");
  requireBytes(logits, uint64_t{params.rows} * params.vocabulary * 4, "sampling logits");
  if (params.repetition_penalty != 1.0F) {
    requireBytes(history, uint64_t{params.rows} * params.history_stride * 4, "sampling history");
    requireBytes(historyCounts, uint64_t{params.rows} * 4, "sampling history count");
  }
  if (params.temperature > 0.0F)
    requireBytes(uniforms, rowBytes(params.rows, std::max(params.uniform_stride, 1u), 1, 4), "sampling uniform");
  const uint32_t outputStride = std::max(params.output_stride, 1u);
  requireBytes(output, rowBytes(params.rows, outputStride, 1, 4), "sampling output");
  const metal::MetalBuffer past = history ? history : logits;
  const metal::MetalBuffer counts = historyCounts ? historyCounts : logits;
  const metal::MetalBuffer draws = uniforms ? uniforms : logits;
  graph.add("omni_sample_codes", {std::move(logits), past, counts, draws, std::move(output)}, params,
            {params.rows, 1, 1});
}

void Sampling::addArgmax(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer output,
                         uint32_t rows, uint32_t vocabulary, uint32_t stride, uint32_t outputStride) {
  if (!rows || !vocabulary || stride < vocabulary) throw std::invalid_argument("argmax takes rows of a vocabulary");
  const uint32_t step = std::max(outputStride, 1u);
  requireBytes(logits, rowBytes(rows, stride, vocabulary, 4), "argmax logits");
  requireBytes(output, rowBytes(rows, step, 1, 4), "argmax output");
  const OmniArgmaxParams params{rows, vocabulary, stride, step};
  graph.add("omni_argmax", {std::move(logits), std::move(output)}, params, {rows, 1, 1}, {1024, 1, 1});
}

void Sampling::addArgmaxMapped(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer output,
                               metal::MetalBuffer map, uint32_t rows, uint32_t vocabulary, uint32_t stride,
                               uint32_t outputStride) {
  if (!rows || !vocabulary || stride < vocabulary) throw std::invalid_argument("argmax takes rows of a vocabulary");
  const uint32_t step = std::max(outputStride, 1u);
  requireBytes(logits, rowBytes(rows, stride, vocabulary, 4), "argmax logits");
  requireBytes(output, rowBytes(rows, step, 1, 4), "argmax output");
  requireBytes(map, uint64_t{vocabulary} * 4, "argmax token map");
  const OmniArgmaxParams params{rows, vocabulary, stride, step};
  graph.add("omni_argmax_mapped", {std::move(logits), std::move(output), std::move(map)}, params, {rows, 1, 1},
            {1024, 1, 1});
}

} // namespace splash::ops::omni

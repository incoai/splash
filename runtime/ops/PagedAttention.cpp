#include "ops/PagedAttention.hpp"

#include "metal/abi/RoPE.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {

namespace {

enum class KernelLayout : uint8_t { Kv4Group6, Kv2Group8, Kv2Group12 };

// One thread per gate element, across rows x query heads x head dimension.
uint64_t gateGroups(uint64_t rows, uint32_t queryHeads, uint32_t headDimension) {
  const uint64_t elements = rows * queryHeads * headDimension;
  return (elements + metal::CommandGraph::kDefaultThreads - 1) /
         metal::CommandGraph::kDefaultThreads;
}

KernelLayout storageKernelLayout(kv::Layout layout) {
  if (!layout.valid() || layout.headDimension != 256) {
    throw std::invalid_argument("no KV store kernel for layout");
  }
  if (layout.kvHeads == 4)
    return KernelLayout::Kv4Group6;
  if (layout.kvHeads == 2)
    return KernelLayout::Kv2Group8;
  throw std::invalid_argument("no KV store kernel for layout");
}

KernelLayout attentionKernelLayout(uint32_t queryHeads, kv::Layout layout) {
  // Select a compiled GQA variant so kernels need no geometry branches. The
  // value order picks the family: the sparse kernels gather blocks of
  // token-major values (compiled for 24 query heads over two KV heads,
  // Qwen3.8-Flash-Next's), and the dense page loop reads dimension-major ones.
  const KernelLayout storage = storageKernelLayout(layout);
  if (layout.valuesTokenMajor) {
    if (storage != KernelLayout::Kv2Group8 || queryHeads != 24)
      throw std::invalid_argument("no sparse attention kernel for layout");
    return KernelLayout::Kv2Group12;
  }
  if (!((storage == KernelLayout::Kv4Group6 && queryHeads == 24) ||
        (storage == KernelLayout::Kv2Group8 && queryHeads == 16)))
    throw std::invalid_argument("no paged-attention kernel for layout");
  return storage;
}

// The KV store of a layout's format, KV heads and value order.
std::string_view storePipeline(kv::Layout layout, bool verify) {
  const KernelLayout kernel = storageKernelLayout(layout);
  const bool bf16 = layout.format == kv::Format::BFloat16;
  if (layout.valuesTokenMajor) {
    if (kernel != KernelLayout::Kv2Group8)
      throw std::invalid_argument("no KV store kernel for the layout's value order");
    return verify ? (bf16 ? "verify_attention_bf16_store_kv2_vt" : "verify_attention_q8_store_kv2_vt")
                  : (bf16 ? "prefill_attention_bf16_store_kv2_vt" : "prefill_attention_q8_store_kv2_vt");
  }
  if (kernel == KernelLayout::Kv4Group6)
    return verify ? (bf16 ? "verify_attention_bf16_store" : "verify_attention_q8_store")
                  : (bf16 ? "prefill_attention_bf16_store" : "prefill_attention_q8_store");
  return verify ? (bf16 ? "verify_attention_bf16_store_kv2_g8" : "verify_attention_q8_store_kv2_g8")
                : (bf16 ? "prefill_attention_bf16_store_kv2_g8" : "prefill_attention_q8_store_kv2_g8");
}

std::string_view pipeline(KernelLayout layout, std::string_view kv4Group6,
                          std::string_view kv2Group8, std::string_view kv2Group12) noexcept {
  return layout == KernelLayout::Kv4Group6   ? kv4Group6
         : layout == KernelLayout::Kv2Group8 ? kv2Group8
                                             : kv2Group12;
}

// One dispatch normalizes both the queries and the keys, so their norms must
// share a stored type.
std::string qkNormKernel(std::string_view name, const NormWeights &queryNorm,
                         const NormWeights &keyNorm, uint32_t headDimension) {
  std::string kernel = normKernel(name, queryNorm, headDimension);
  if (normKernel(name, keyNorm, headDimension) != kernel)
    throw std::invalid_argument("query and key norms differ in type");
  return kernel;
}

AttentionWorkspace attentionWorkspace(uint64_t rows, uint32_t headDimension) {
  return {rows * headDimension * sizeof(float), rows * 2 * sizeof(float)};
}

// Verify scratch is sized once for the maximum split count of every lane, so
// a lane's history-scaled partition never needs a reallocation.
AttentionWorkspace verifyWorkspaceBound(uint32_t lanes, uint32_t queryHeads,
                                        kv::Layout layout) {
  return attentionWorkspace(uint64_t{lanes} * kv::kVerifyRows *
                                kv::kVerifyMaximumSplits * queryHeads,
                            layout.headDimension);
}

// A staging of `rows` rows of each of `heads` heads, `stride` rows apart:
// queries and attention rows [KV head][row][query head in group][dimension],
// keys [KV head][row][dimension] (a group of one), values
// [KV head][dimension][row]. Verify stages each lane's KV heads in lane
// order, the lane's rows in kVerifyChunkStride rows of each.
struct Staging final {
  uint64_t heads;
  uint32_t stride;
  uint32_t rows;

  [[nodiscard]] uint64_t rowBytes(uint32_t group, uint32_t headDimension) const noexcept {
    const uint64_t row = uint64_t{group} * headDimension;
    return ops::rowBytes(heads, stride * row, rows * row, 2);
  }
  [[nodiscard]] uint64_t valueBytes(uint32_t headDimension) const noexcept {
    return ops::rowBytes(heads * headDimension, stride, rows, 2);
  }
};
Staging verifyStaging(uint32_t lanes, kv::Layout layout) {
  return {uint64_t{lanes} * layout.kvHeads, kv::kVerifyChunkStride, kv::kVerifyRows};
}

// A packed projection row: every query head's q and gate, then the K and V
// heads. The gates read each row's query heads only.
uint64_t packedWidth(uint32_t queryHeads, kv::Layout layout) {
  return (uint64_t{queryHeads} * 2 + uint64_t{layout.kvHeads} * 2) * layout.headDimension;
}
uint64_t gateBytes(uint64_t rows, uint32_t queryHeads, kv::Layout layout) {
  return rowBytes(rows, packedWidth(queryHeads, layout), uint64_t{queryHeads} * 2 * layout.headDimension, 2);
}

void requireRopeTables(const metal::MetalBuffer &cosine, const metal::MetalBuffer &sine, uint64_t rows) {
  const uint64_t bytes = rows * SPLASH_TARGET_ROPE_PAIRS * sizeof(float);
  requireBytes(cosine, bytes, "attention RoPE cosine");
  requireBytes(sine, bytes, "attention RoPE sine");
}

// A chunk's page table holds the pages of its visible tokens.
void requirePageTable(const metal::MetalBuffer &table, const kv::ChunkedPrefillParams &chunk) {
  requireBytes(table, uint64_t{kv::chunkedPrefillRequiredPages(chunk)} * sizeof(SplashKvPage), "page table");
}

// A dispatch's tiles read lists `blocks.list` to `blocks.list` + `tiles`, and
// only a sparse plan takes lists.
void requireBlocks(const AttentionBlocks &blocks, bool sparse, uint64_t tiles) {
  if (!blocks.stride) return;
  if (!sparse) throw std::invalid_argument("the attention kernels take no block lists");
  const uint64_t lists = uint64_t{blocks.list} + tiles;
  requireBytes(blocks.lists, lists * blocks.stride * sizeof(uint32_t), "attention block lists");
  requireBytes(blocks.counts, lists * sizeof(uint32_t), "attention block list counts");
}

// What a sparse kernel binds for its lists and their counts: theirs, or
// without lists another buffer of the dispatch, of which it reads nothing.
std::array<metal::MetalBuffer, 2> blockBindings(const AttentionBlocks &blocks, const metal::MetalBuffer &other) {
  return blocks.stride ? std::array{blocks.lists, blocks.counts} : std::array{other, other};
}

// Each query tile's history splits: the maximum shared out over the chunk's
// tiles, at least one.
uint32_t prefillSplits(uint32_t tiles) {
  constexpr uint32_t maximum = SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS;
  return std::clamp(maximum / tiles, 1U, maximum);
}

} // namespace

PrefillAttentionPlan PagedAttention::prefillPlan(
    uint32_t rows, uint32_t queryHeads, kv::Layout layout) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!rows || rows > kv::kChunkedPrefillMaximumRows)
    throw std::invalid_argument("invalid attention workspace rows");
  const uint32_t tiles = kv::prefillAttentionTiles(rows);
  const uint32_t splits = prefillSplits(tiles);
  const uint32_t fusedRows =
      kv::kPrefillAttentionTileRows * (queryHeads / layout.kvHeads);
  return {rows, splits,
          attentionWorkspace(uint64_t{tiles} * splits * layout.kvHeads * fusedRows,
                             layout.headDimension),
          layout.format == kv::Format::BFloat16
              ? pipeline(kernel, "prefill_attention_bf16_split",
                         "prefill_attention_bf16_split_kv2_g8",
                         "prefill_attention_bf16_sparse_kv2_g12")
              : pipeline(kernel, "prefill_attention_q8_split",
                         "prefill_attention_q8_split_kv2_g8",
                         "prefill_attention_q8_sparse_kv2_g12"),
          pipeline(kernel, "prefill_attention_reduce",
                   "prefill_attention_reduce_kv2_g8", "prefill_attention_sparse_reduce_kv2_g12"),
          {layout.kvHeads, tiles, splits}, {layout.kvHeads, fusedRows, tiles}, layout.valuesTokenMajor,
          layout, queryHeads};
}

VerifyAttentionPlan PagedAttention::verifyPlan(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid attention workspace batch width");
  if (historyTokens.size() != lanes)
    throw std::invalid_argument("invalid verify attention history vector");
  std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> laneSplits{};
  uint32_t splits = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (uint64_t{historyTokens[lane]} + kv::kVerifyRows >
        kv::kMaximumPhysicalTokens)
      throw std::invalid_argument(
          "verify attention history exceeds physical context");
    laneSplits[lane] = kv::verifyAttentionSplits(historyTokens[lane]);
    splits = std::max(splits, laneSplits[lane]);
  }
  return {lanes, laneSplits, splits,
          verifyWorkspaceBound(lanes, queryHeads, layout),
          layout.format == kv::Format::BFloat16
              ? pipeline(kernel, "verify_attention_bf16_split",
                         "verify_attention_bf16_split_kv2_g8",
                         "verify_attention_bf16_sparse_kv2_g12")
              : pipeline(kernel, "verify_attention_q8_split",
                         "verify_attention_q8_split_kv2_g8",
                         "verify_attention_q8_sparse_kv2_g12"),
          pipeline(kernel, "verify_attention_reduce",
                   "verify_attention_reduce_kv2_g8", "verify_attention_sparse_reduce_kv2_g12"),
          // A sparse lane's list fills only the first of the plan's splits.
          // Its split grid is one-dimensional, x the KV head and then the
          // split: an M5 Max spread the same order as a two-dimensional grid
          // (x the KV head, y the split) worse, running it up to 1.7x slower
          // on most grids of more than 32 splits.
          layout.valuesTokenMajor ? metal::DispatchSize{layout.kvHeads * splits, 1, lanes}
                                  : metal::DispatchSize{layout.kvHeads, splits, lanes},
          {layout.kvHeads,
           kv::kVerifyRows * (queryHeads / layout.kvHeads), lanes},
          layout.valuesTokenMajor,
          pipeline(kernel, {}, {},
                   layout.format == kv::Format::BFloat16 ? "verify_attention_bf16_sparse_located_kv2_g12"
                                                         : "verify_attention_q8_sparse_located_kv2_g12"),
          storePipeline(layout, true),
          {uint64_t{lanes} * 2 * kv::kVerifyRows * layout.kvHeads, 1, 1},
          {layout.headDimension, 1, 1}, layout, queryHeads};
}

AttentionWorkspace PagedAttention::prefillWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) {
  (void)attentionKernelLayout(queryHeads, layout);
  if (!maximumRows || maximumRows > kv::kChunkedPrefillMaximumRows)
    throw std::invalid_argument("invalid attention workspace rows");
  // Allocation-time bound for every shorter sequence. Encoding resolves its
  // own exact plan in constant time; no bound scan occurs on the hot path.
  uint64_t slots = 0;
  for (uint32_t tiles = 1; tiles <= kv::prefillAttentionTiles(maximumRows); ++tiles)
    slots = std::max(slots, uint64_t{tiles} * prefillSplits(tiles));
  return attentionWorkspace(slots * kv::kPrefillAttentionTileRows * queryHeads,
                             layout.headDimension);
}

AttentionWorkspace PagedAttention::verifyWorkspace(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout) {
  (void)attentionKernelLayout(queryHeads, layout);
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid attention workspace batch width");
  return verifyWorkspaceBound(lanes, queryHeads, layout);
}

void PagedAttention::addPrefillProjection(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    const NormWeights &queryNorm, const NormWeights &keyNorm,
    metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
    metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
    metal::MetalBuffer chunkValues, uint32_t tokens, uint32_t stride,
    uint32_t queryHeads, kv::Layout layout) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!tokens || !stride)
    throw std::invalid_argument("invalid paged prefill projection geometry");
  const Staging staging{layout.kvHeads, stride, tokens};
  requireBytes(packed, uint64_t{tokens} * packedWidth(queryHeads, layout) * 2, "attention q/k/v");
  requireRopeTables(ropeCos, ropeSin, tokens);
  requireBytes(queries, staging.rowBytes(queryHeads / layout.kvHeads, layout.headDimension), "attention query");
  requireBytes(chunkKeys, staging.rowBytes(1, layout.headDimension), "attention key");
  requireBytes(chunkValues, staging.valueBytes(layout.headDimension), "attention value");
  const FullPrefillParams params{tokens, stride};
  graph.add(qkNormKernel(pipeline(kernel, "prefill_attention_qkv",
                                  "prefill_attention_qkv_kv2_g8", "prefill_attention_qkv_kv2_g12"),
                         queryNorm, keyNorm, layout.headDimension),
            {std::move(packed), queryNorm.buffer, keyNorm.buffer,
             std::move(ropeCos), std::move(ropeSin), std::move(queries),
             std::move(chunkKeys), std::move(chunkValues)},
            params,
            {uint64_t{tokens} * (queryHeads + layout.kvHeads), 1, 1});
}

void PagedAttention::addPrefillGate(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer attention, metal::MetalBuffer hidden, uint32_t tokens,
    uint32_t stride, uint32_t queryHeads, kv::Layout layout) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!tokens || !stride)
    throw std::invalid_argument("invalid paged prefill gate geometry");
  requireBytes(packed, gateBytes(tokens, queryHeads, layout), "attention gate");
  requireBytes(attention,
               Staging{layout.kvHeads, stride, tokens}.rowBytes(queryHeads / layout.kvHeads, layout.headDimension),
               "attention row");
  requireBytes(hidden, uint64_t{tokens} * queryHeads * layout.headDimension * 2, "attention hidden");
  const FullPrefillParams params{tokens, stride};
  graph.add(std::string(pipeline(kernel, "prefill_attention_gate",
                                 "prefill_attention_gate_kv2_g8", "prefill_attention_gate_kv2_g12")),
            {std::move(packed), std::move(attention), std::move(hidden)}, params,
            {gateGroups(tokens, queryHeads, layout.headDimension), 1, 1});
}

void PagedAttention::addVerifyProjection(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    const NormWeights &queryNorm, const NormWeights &keyNorm,
    metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
    metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
    metal::MetalBuffer chunkValues, uint32_t queryHeads, kv::Layout layout,
    uint32_t lanes) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid paged verify projection geometry");
  const uint64_t rows = uint64_t{lanes} * kv::kVerifyRows;
  const Staging staging = verifyStaging(lanes, layout);
  requireBytes(packed, rows * packedWidth(queryHeads, layout) * 2, "attention q/k/v");
  requireRopeTables(ropeCos, ropeSin, rows);
  requireBytes(queries, staging.rowBytes(queryHeads / layout.kvHeads, layout.headDimension), "attention query");
  requireBytes(chunkKeys, staging.rowBytes(1, layout.headDimension), "attention key");
  requireBytes(chunkValues, staging.valueBytes(layout.headDimension), "attention value");
  graph.add(qkNormKernel(pipeline(kernel, "verify_attention_qkv",
                                  "verify_attention_qkv_kv2_g8", "verify_attention_qkv_kv2_g12"),
                         queryNorm, keyNorm, layout.headDimension),
            {std::move(packed), queryNorm.buffer, keyNorm.buffer,
             std::move(ropeCos), std::move(ropeSin), std::move(queries),
             std::move(chunkKeys), std::move(chunkValues)},
            {uint64_t{SPLASH_TARGET_VERIFY_ROWS} * (queryHeads + layout.kvHeads),
             lanes, 1});
}

PreparedInput PagedAttention::addVerifyGate(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer attention, metal::MetalBuffer hidden,
    uint32_t queryHeads, kv::Layout layout, uint32_t lanes, LinearScratch scratch,
    LinearInput input) {
  const KernelLayout kernel = attentionKernelLayout(queryHeads, layout);
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid paged verify gate geometry");
  const uint32_t rows = lanes * SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(packed, gateBytes(rows, queryHeads, layout), "attention gate");
  requireBytes(attention, verifyStaging(lanes, layout).rowBytes(queryHeads / layout.kvHeads, layout.headDimension),
               "attention row");
  requireBytes(hidden, uint64_t{rows} * queryHeads * layout.headDimension * 2, "attention hidden");
  if (input != LinearInput::Plain) {
    const uint32_t width = queryHeads * layout.headDimension;
    requireTableScratch(scratch, input, width, rows);
    std::string name = std::string("verify_attention_gate") + tableSuffix(input);
    name += pipeline(kernel, "", "_kv2_g8", "_kv2_g12");
    graph.add(std::move(name), {packed, attention, hidden, scratch.input, scratch.sums},
              {width / 64 * lanes, 1, 1}, {256, 1, 1});
    return {std::move(hidden), input};
  }
  graph.add(std::string(pipeline(kernel, "verify_attention_gate",
                                 "verify_attention_gate_kv2_g8", "verify_attention_gate_kv2_g12")),
            {std::move(packed), std::move(attention), hidden},
            FullDecodeBatchParams{lanes},
            {gateGroups(rows, queryHeads, layout.headDimension), 1, 1});
  return {};
}

kv::ChunkedPrefillParams PagedAttention::prefillParams(
    uint64_t logicalPosition, uint32_t chunkTokens, uint32_t chunkStride,
    uint32_t pageTableEntries) {
  if (logicalPosition > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("KV logical position exceeds kernel ABI");
  kv::ChunkedPrefillParams params{static_cast<uint32_t>(logicalPosition),
                                  chunkTokens, chunkStride, pageTableEntries,
                                  {}};
  const std::string_view error = kv::chunkedPrefillValidationError(params);
  if (!error.empty())
    throw std::invalid_argument(std::string(error));
  return params;
}

kv::ChunkedPrefillParams PagedAttention::verifyParams(uint64_t logicalPosition,
                                                      uint32_t pageTableEntries) {
  return prefillParams(logicalPosition, kv::kVerifyRows,
                       kv::kVerifyChunkStride, pageTableEntries);
}

void PagedAttention::addPrefillStore(
    metal::CommandGraph &graph, SplashKvLayer layer,
    metal::MetalBuffer chunkKeys, metal::MetalBuffer chunkValues,
    metal::MetalBuffer pageTable,
    const kv::ChunkedPrefillParams &params, kv::Layout layout) {
  const std::string_view store = storePipeline(layout, false);
  const Staging staging{layout.kvHeads, params.chunk_stride, params.chunk_tokens};
  requireBytes(chunkKeys, staging.rowBytes(1, layout.headDimension), "attention key");
  requireBytes(chunkValues, staging.valueBytes(layout.headDimension), "attention value");
  requirePageTable(pageTable, params);
  kv::ChunkedPrefillParams layerParams = params;
  layerParams.kv = layer;
  graph.add(std::string(store),
            {std::move(chunkKeys), std::move(chunkValues), std::move(pageTable)},
            layerParams, {uint64_t{2} * params.chunk_tokens * layout.kvHeads, 1, 1},
            {layout.headDimension, 1, 1});
}

void PagedAttention::addPrefill(
    metal::CommandGraph &graph, SplashKvLayer layer,
    metal::MetalBuffer queries, metal::MetalBuffer output,
    metal::MetalBuffer partials, metal::MetalBuffer statistics,
    metal::MetalBuffer pageTable, const kv::ChunkedPrefillParams &chunk,
    const PrefillAttentionPlan &plan, AttentionBlocks blocks) {
  if (chunk.chunk_tokens != plan.rows)
    throw std::invalid_argument("prefill attention rows do not match plan");
  const std::string_view error = kv::chunkedPrefillValidationError(chunk);
  if (!error.empty())
    throw std::invalid_argument(std::string(error));
  // The tiles read the queries and write the output in whole tiles of rows.
  const Staging staging{plan.layout.kvHeads, chunk.chunk_stride, plan.tiles * kv::kPrefillAttentionTileRows};
  const uint64_t rows = staging.rowBytes(plan.queryHeads / plan.layout.kvHeads, plan.layout.headDimension);
  requireBytes(queries, rows, "attention query");
  requireBytes(output, rows, "attention output");
  requireBytes(partials, plan.workspace.partialsBytes, "prefill attention partials");
  requireBytes(statistics, plan.workspace.statisticsBytes, "prefill attention statistics");
  requirePageTable(pageTable, chunk);
  requireBlocks(blocks, plan.sparse, plan.tiles);
  if (blocks.locations) throw std::invalid_argument("prefill attention takes no block locations");
  const kv::PrefillAttentionParams params{
      chunk.committed_tokens, chunk.chunk_tokens, chunk.chunk_stride,
      chunk.page_table_entries, layer, plan.splits, blocks.stride, blocks.list};
  std::vector<metal::MetalBuffer> split{std::move(queries), partials, statistics, std::move(pageTable)};
  std::vector<metal::MetalBuffer> reduce{partials, statistics, std::move(output)};
  if (plan.sparse) {
    const auto [lists, counts] = blockBindings(blocks, partials);
    split.insert(split.end(), {lists, counts});
    reduce.push_back(counts);
  }
  graph.add(std::string(plan.splitPipeline), std::move(split), params, plan.splitGroups);
  graph.add(std::string(plan.reducePipeline), std::move(reduce), params, plan.reduceGroups);
}

void PagedAttention::addVerify(metal::CommandGraph &graph, SplashKvLayer layer,
                               PagedVerifyBuffers buffers,
                               std::span<const kv::ChunkedPrefillParams> chunks,
                               const VerifyAttentionPlan &plan, AttentionBlocks blocks) {
  constexpr uint32_t maximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  if (chunks.size() != plan.lanes || buffers.pageTables.size() != maximumLanes) {
    throw std::invalid_argument("invalid paged verify batch");
  }
  const Staging staging = verifyStaging(plan.lanes, plan.layout);
  const uint32_t headDimension = plan.layout.headDimension;
  const uint64_t rows = staging.rowBytes(plan.queryHeads / plan.layout.kvHeads, headDimension);
  requireBytes(buffers.chunkKeys, staging.rowBytes(1, headDimension), "attention key");
  requireBytes(buffers.chunkValues, staging.valueBytes(headDimension), "attention value");
  requireBytes(buffers.queries, rows, "attention query");
  requireBytes(buffers.output, rows, "attention output");
  requireBytes(buffers.partials, plan.workspace.partialsBytes, "verify attention partials");
  requireBytes(buffers.statistics, plan.workspace.statisticsBytes, "verify attention statistics");
  for (uint32_t lane = 0; lane < plan.lanes; ++lane)
    requirePageTable(buffers.pageTables[lane], chunks[lane]);
  requireBlocks(blocks, plan.sparse, plan.lanes);
  // The located split reads a location for each entry of each lane's list.
  if (blocks.locations) {
    if (!blocks.stride || !blocks.locationPages)
      throw std::invalid_argument("attention block locations need lists and their pages");
    requireBytes(blocks.locations, (uint64_t{blocks.list} + plan.lanes) * blocks.stride * sizeof(uint32_t),
                 "attention block locations");
  }
  // verifyParams validated each chunk, and the plan scaled each lane's
  // split count from the same committed history; every lane's partials use
  // the plan-wide slot stride.
  std::array<kv::ChunkedPrefillParams, maximumLanes> stores{};
  std::array<kv::VerifyAttentionParams, maximumLanes> attention{};
  for (uint32_t lane = 0; lane < plan.lanes; ++lane) {
    const kv::ChunkedPrefillParams &chunk = chunks[lane];
    stores[lane] = chunk;
    stores[lane].kv = layer;
    attention[lane] = {chunk.committed_tokens, chunk.page_table_entries, layer,
                       plan.laneSplits[lane], plan.splits, blocks.stride,
                       blocks.list + lane};
  }
  const auto &tables = buffers.pageTables;
  graph.add(std::string(plan.storePipeline_),
            {buffers.chunkKeys, buffers.chunkValues, tables[0], tables[1], tables[2],
             tables[3]},
            stores, plan.storeGroups_, plan.storeThreads_);
  std::vector<metal::MetalBuffer> split{buffers.queries, buffers.partials, buffers.statistics,
                                        tables[0], tables[1], tables[2], tables[3]};
  std::vector<metal::MetalBuffer> reduce{buffers.partials, buffers.statistics, buffers.output};
  if (plan.sparse) {
    const auto [lists, counts] = blockBindings(blocks, buffers.partials);
    split.insert(split.end(), {lists, counts});
    if (blocks.locations) split.insert(split.end(), {blocks.locations, blocks.locationPages});
    reduce.push_back(counts);
  }
  graph.add(std::string(blocks.locations ? plan.locatedSplitPipeline_ : plan.splitPipeline), std::move(split),
            attention, plan.splitGroups);
  graph.add(std::string(plan.reducePipeline), std::move(reduce), attention, plan.reduceGroups);
}

void PagedAttention::addBlockLists(metal::CommandGraph &graph, const metal::MetalBuffer &mask, uint32_t maskWords,
                                   std::span<const BlockListRows> sequences, const AttentionBlocks &blocks) {
  // The sparse kernels read a list's rows as a verify lane's or a prefill
  // tile's, which are as many, each a bit of an entry.
  constexpr uint32_t tileRows = kv::kPrefillAttentionTileRows;
  static_assert(tileRows == kv::kVerifyRows && tileRows <= SPLASH_ATTENTION_BLOCK_ROW_BITS,
                "a block list's rows are a verify lane's and a prefill tile's");
  if (sequences.empty() || sequences.size() > SPLASH_MAXIMUM_BATCH_WIDTH || !maskWords || !blocks.stride)
    throw std::invalid_argument("invalid attention block lists");
  SplashAttentionBlockListParams params{
      .tile_rows = tileRows, .mask_words = maskWords, .stride = blocks.stride,
      .sequences = static_cast<uint32_t>(sequences.size()), .sequence = {}};
  uint64_t maskRows = 0, lists = 0;
  uint32_t tiles = 0;
  for (size_t index = 0; index < sequences.size(); ++index) {
    const BlockListRows &rows = sequences[index];
    if (!rows.rows) throw std::invalid_argument("invalid attention block lists");
    const uint64_t historyBlocks = (uint64_t{rows.firstPosition} + rows.rows + SPLASH_ATTENTION_BLOCK_TOKENS - 1) /
                                   SPLASH_ATTENTION_BLOCK_TOKENS;
    if (uint64_t{maskWords} * 32 < historyBlocks)
      throw std::invalid_argument("attention block masks narrower than their rows' history");
    const uint32_t sequenceTiles = (rows.rows + tileRows - 1) / tileRows;
    maskRows = std::max(maskRows, uint64_t{rows.maskRow} + rows.rows);
    lists = std::max(lists, uint64_t{blocks.list} + rows.list + sequenceTiles);
    tiles = std::max(tiles, sequenceTiles);
    params.sequence[index] = {.mask_row = rows.maskRow, .rows = rows.rows, .first_position = rows.firstPosition,
                              .list = blocks.list + rows.list};
  }
  requireBytes(mask, maskRows * maskWords * sizeof(uint32_t), "attention block mask");
  requireBytes(blocks.lists, lists * blocks.stride * sizeof(uint32_t), "attention block lists");
  requireBytes(blocks.counts, lists * sizeof(uint32_t), "attention block list counts");
  graph.add("attention_block_lists", {mask, blocks.lists, blocks.counts}, params,
            {tiles, sequences.size(), 1}, {SPLASH_ATTENTION_BLOCK_LIST_THREADS, 1, 1});
}

void PagedAttention::addStageLayer(metal::CommandGraph &graph, SplashKvLayer poolLayer, metal::MetalBuffer poolTable,
                                   SplashKvLayer stagedLayer, metal::MetalBuffer stagedTable, uint32_t firstPage,
                                   uint32_t pages, kv::Layout layout) {
  if (!layout.valid() || !pages) throw std::invalid_argument("invalid KV stage geometry");
  // Both tables hold the staged pages' entries.
  const uint64_t tableBytes = (uint64_t{firstPage} + pages) * sizeof(SplashKvPage);
  requireBytes(poolTable, tableBytes, "KV stage pool table");
  requireBytes(stagedTable, tableBytes, "KV stage staged table");
  const SplashKvStageParams params{poolLayer, stagedLayer, firstPage,
                                   static_cast<uint32_t>(layout.dataBytesPerLayerPage()),
                                   static_cast<uint32_t>(layout.scaleBytesPerLayerPage())};
  graph.add("paged_kv_stage_layer", {std::move(poolTable), std::move(stagedTable)}, params, {pages, 1, 1});
}

} // namespace splash::ops

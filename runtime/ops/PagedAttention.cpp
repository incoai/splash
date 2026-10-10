#include "ops/PagedAttention.hpp"

#include "metal/abi/RoPE.h"
#include "ops/BufferExtent.hpp"
#include "metal/abi/Qwen4.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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
  // Select a compiled GQA variant so kernels need no geometry branches.
  const KernelLayout result = storageKernelLayout(layout);
  if ((result == KernelLayout::Kv4Group6 && queryHeads == 24) ||
      (result == KernelLayout::Kv2Group8 && queryHeads == 16))
    return result;
  // Qwen3.8-Flash-Next: 24 query heads over two KV heads; its KV storage is
  // the two-head layout's.
  if (result == KernelLayout::Kv2Group8 && queryHeads == 24) return KernelLayout::Kv2Group12;
  throw std::invalid_argument("no paged-attention kernel for layout");
}

// The kernels of a group of twelve (kernels/*/attention_*.metal _kv2_g12),
// by their group-of-eight name; the KV stores depend on the KV heads alone.
constexpr std::pair<std::string_view, std::string_view> kGroup12Kernels[] = {
    {"prefill_attention_q8_split_kv2_g8", "prefill_attention_q8_split_kv2_g12"},
    {"prefill_attention_bf16_split_kv2_g8", "prefill_attention_bf16_split_kv2_g12"},
    {"prefill_attention_reduce_kv2_g8", "prefill_attention_reduce_kv2_g12"},
    {"prefill_attention_qkv_kv2_g8", "prefill_attention_qkv_kv2_g12"},
    {"prefill_attention_gate_kv2_g8", "prefill_attention_gate_kv2_g12"},
    {"verify_attention_q8_split_kv2_g8", "verify_attention_q8_split_kv2_g12"},
    {"verify_attention_bf16_split_kv2_g8", "verify_attention_bf16_split_kv2_g12"},
    {"verify_attention_reduce_kv2_g8", "verify_attention_reduce_kv2_g12"},
    {"verify_attention_qkv_kv2_g8", "verify_attention_qkv_kv2_g12"},
    {"verify_attention_gate_kv2_g8", "verify_attention_gate_kv2_g12"},
    // The table gates' suffix (verify_attention_gate_table16_kv2_g12, ...).
    {"_kv2_g8", "_kv2_g12"},
};

// The GQA-12 split kernels, which also bind the QSA bitmaps.
bool masked(std::string_view pipeline) noexcept {
  return pipeline.ends_with("_split_kv2_g12");
}

std::string_view pipeline(KernelLayout layout, std::string_view kv4Group6,
                          std::string_view kv2Group8) noexcept {
  if (layout == KernelLayout::Kv4Group6) return kv4Group6;
  if (layout == KernelLayout::Kv2Group12)
    for (const auto &[group8, group12] : kGroup12Kernels)
      if (group8 == kv2Group8) return group12;
  return kv2Group8;
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
                         "prefill_attention_bf16_split_kv2_g8")
              : pipeline(kernel, "prefill_attention_q8_split",
                         "prefill_attention_q8_split_kv2_g8"),
          pipeline(kernel, "prefill_attention_reduce",
                   "prefill_attention_reduce_kv2_g8"),
          {layout.kvHeads, tiles, splits}, {layout.kvHeads, fusedRows, tiles}, layout, queryHeads};
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
                         "verify_attention_bf16_split_kv2_g8")
              : pipeline(kernel, "verify_attention_q8_split",
                         "verify_attention_q8_split_kv2_g8"),
          pipeline(kernel, "verify_attention_reduce",
                   "verify_attention_reduce_kv2_g8"),
          {layout.kvHeads, splits, lanes},
          {layout.kvHeads,
           kv::kVerifyRows * (queryHeads / layout.kvHeads), lanes},
          layout.format == kv::Format::BFloat16
              ? pipeline(kernel, "verify_attention_bf16_store",
                         "verify_attention_bf16_store_kv2_g8")
              : pipeline(kernel, "verify_attention_q8_store",
                         "verify_attention_q8_store_kv2_g8"),
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
                                  "prefill_attention_qkv_kv2_g8"),
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
                                 "prefill_attention_gate_kv2_g8")),
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
                                  "verify_attention_qkv_kv2_g8"),
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
    name += pipeline(kernel, "", "_kv2_g8");
    graph.add(std::move(name), {packed, attention, hidden, scratch.input, scratch.sums},
              {width / 64 * lanes, 1, 1}, {256, 1, 1});
    return {std::move(hidden), input};
  }
  graph.add(std::string(pipeline(kernel, "verify_attention_gate",
                                 "verify_attention_gate_kv2_g8")),
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
  const KernelLayout kernel = storageKernelLayout(layout);
  const auto store = layout.format == kv::Format::BFloat16
      ? pipeline(kernel, "prefill_attention_bf16_store", "prefill_attention_bf16_store_kv2_g8")
      : pipeline(kernel, "prefill_attention_q8_store", "prefill_attention_q8_store_kv2_g8");
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
    const PrefillAttentionPlan &plan, QsaMask mask) {
  if (chunk.chunk_tokens != plan.rows)
    throw std::invalid_argument("prefill attention rows do not match plan");
  const std::string_view error = kv::chunkedPrefillValidationError(chunk);
  if (!error.empty())
    throw std::invalid_argument(std::string(error));
  // The tiles read the queries and write the output in whole tiles of rows.
  const Staging staging{plan.layout.kvHeads, chunk.chunk_stride,
                        kv::prefillAttentionTiles(plan.rows) * kv::kPrefillAttentionTileRows};
  const uint64_t rows = staging.rowBytes(plan.queryHeads / plan.layout.kvHeads, plan.layout.headDimension);
  requireBytes(queries, rows, "attention query");
  requireBytes(output, rows, "attention output");
  requireBytes(partials, plan.workspace.partialsBytes, "prefill attention partials");
  requireBytes(statistics, plan.workspace.statisticsBytes, "prefill attention statistics");
  requirePageTable(pageTable, chunk);
  const kv::PrefillAttentionParams params{
      chunk.committed_tokens, chunk.chunk_tokens, chunk.chunk_stride,
      chunk.page_table_entries, layer, plan.splits};
  if (masked(plan.splitPipeline))
    graph.add(std::string(plan.splitPipeline),
              {std::move(queries), partials, statistics, std::move(pageTable)}, params,
              {mask.words ? mask.bits : partials}, Qwen4QsaMaskParams{mask.words, mask.row0},
              plan.splitGroups);
  else
    graph.add(std::string(plan.splitPipeline),
              {std::move(queries), partials, statistics, std::move(pageTable)},
              params, plan.splitGroups);
  graph.add(std::string(plan.reducePipeline),
            {std::move(partials), std::move(statistics), std::move(output)},
            params, plan.reduceGroups);
}

void PagedAttention::addVerify(metal::CommandGraph &graph, SplashKvLayer layer,
                               PagedVerifyBuffers buffers,
                               std::span<const kv::ChunkedPrefillParams> chunks,
                               const VerifyAttentionPlan &plan, QsaMask mask) {
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
                       plan.laneSplits[lane], plan.splits};
  }
  const auto &tables = buffers.pageTables;
  graph.add(std::string(plan.storePipeline_),
            {buffers.chunkKeys, buffers.chunkValues, tables[0], tables[1], tables[2],
             tables[3]},
            stores, plan.storeGroups_, plan.storeThreads_);
  if (masked(plan.splitPipeline))
    graph.add(std::string(plan.splitPipeline),
              {buffers.queries, buffers.partials, buffers.statistics, tables[0],
               tables[1], tables[2], tables[3]},
              attention, {mask.words ? mask.bits : buffers.partials},
              Qwen4QsaMaskParams{mask.words, mask.row0}, plan.splitGroups);
  else
    graph.add(std::string(plan.splitPipeline),
              {buffers.queries, buffers.partials, buffers.statistics, tables[0],
               tables[1], tables[2], tables[3]},
              attention, plan.splitGroups);
  graph.add(std::string(plan.reducePipeline),
            {buffers.partials, buffers.statistics, buffers.output},
            attention, plan.reduceGroups);
}

} // namespace splash::ops

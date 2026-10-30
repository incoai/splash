#include "ops/PagedAttention.hpp"

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

// The parts a dense tile of a layout's GQA group runs in, each a threadgroup
// of the split (SPLASH_ATTENTION_TILE_PARTS); a sparse tile is one.
uint64_t tileParts(uint32_t queryHeads, kv::Layout layout) {
  return layout.valuesTokenMajor ? 1 : SPLASH_ATTENTION_TILE_PARTS(queryHeads / layout.kvHeads);
}

// One thread per gate element, across rows x query heads x head dimension.
uint64_t gateGroups(uint64_t rows, uint32_t queryHeads, uint32_t headDimension) {
  const uint64_t elements = rows * queryHeads * headDimension;
  return (elements + metal::CommandGraph::kDefaultThreads - 1) /
         metal::CommandGraph::kDefaultThreads;
}

// The compiled attention geometries (kernels/{prefill,decode}/attention_qkv.metal,
// paged_attention.metal and paged_attention_store.metal, and
// shared/sparse_attention.metal): query heads over KV heads of a head
// dimension in a value order, and their kernels' name suffix, the stores' by
// KV heads, head dimension and value order. The sparse kernels gather blocks
// of token-major values (Qwen3.8-Flash-Next's); the dense page loop reads
// dimension-major ones. Each geometry's kernels are compiled for:
// - gated: projection rows that hold each query head's output gate after its
//   queries, by whose sigmoid the output multiplies the attention rows
//   (Qwen3.5's); without, the output gathers them as they are;
// - sinks: a learned sink per query head, which every reduce adds;
// - rotaryPairs: the pairs its RoPE rotates, and unrotated: a projection
//   without RoPE (NoPE) as well, for an AttentionVariant that is not rotated;
// - windowTokens: a split and reduce over a window of that many keys before
//   each row as well as over the whole history, for an AttentionVariant of
//   that window; 0 without.
struct CompiledAttention final {
  uint32_t queryHeads;
  uint32_t kvHeads;
  uint32_t headDimension;
  bool valuesTokenMajor;
  bool gated;
  bool sinks;
  uint32_t rotaryPairs;
  bool unrotated;
  uint32_t windowTokens;
  std::string_view suffix;
  std::string_view storeSuffix;
};
constexpr std::array kCompiledAttention{
    CompiledAttention{24, 4, 256, false, true, false, 32, false, 0, "", ""},
    CompiledAttention{16, 2, 256, false, true, false, 32, false, 0, "_kv2_g8", "_kv2_g8"},
    CompiledAttention{32, 2, 256, false, true, false, 32, false, 0, "_kv2_g16", "_kv2_g8"},
    CompiledAttention{24, 2, 256, true, true, false, 32, false, 0, "_kv2_g12", "_kv2_vt"},
    CompiledAttention{32, 4, 128, false, false, true, 64, true, 512, "_kv4_g8_d128", "_kv4_d128"},
};

// Whether a compiled geometry's pages are a layout's: its KV heads, head
// dimension and value order.
bool storesLayout(const CompiledAttention &compiled, kv::Layout layout) {
  return compiled.kvHeads == layout.kvHeads && compiled.headDimension == layout.headDimension &&
         compiled.valuesTokenMajor == layout.valuesTokenMajor;
}

// The KV store kernels' suffix of a layout's KV heads, head dimension and
// value order.
std::string_view storeSuffix(kv::Layout layout) {
  if (layout.valid())
    for (const CompiledAttention &compiled : kCompiledAttention)
      if (storesLayout(compiled, layout)) return compiled.storeSuffix;
  throw std::invalid_argument("no KV store kernel for layout");
}

// The compiled geometry of `queryHeads` over a layout's KV heads, so the
// kernels need no geometry branches, which must compile a layer's variant.
const CompiledAttention &compiledAttention(uint32_t queryHeads, kv::Layout layout,
                                           AttentionVariant variant = {}) {
  static_cast<void>(storeSuffix(layout));
  for (const CompiledAttention &compiled : kCompiledAttention) {
    if (compiled.queryHeads != queryHeads || !storesLayout(compiled, layout)) continue;
    if ((!variant.rotated && !compiled.unrotated) ||
        (variant.windowTokens && variant.windowTokens != compiled.windowTokens))
      throw std::invalid_argument("no attention kernel for the layer's variant");
    return compiled;
  }
  throw std::invalid_argument(layout.valuesTokenMajor ? "no sparse attention kernel for layout"
                                                       : "no paged-attention kernel for layout");
}

// A phase's ("prefill" or "verify") attention kernel `name` with a suffix.
std::string kernelName(std::string_view phase, std::string_view name, std::string_view suffix) {
  return std::string(phase) + "_attention_" + std::string(name) + std::string(suffix);
}

// The split and reduce kernels' suffix of a geometry's variant: its window's.
std::string splitSuffix(const CompiledAttention &compiled, AttentionVariant variant) {
  return std::string(compiled.suffix) + (variant.windowTokens ? "_w" + std::to_string(variant.windowTokens) : "");
}

// The reduce kernel of a geometry's variant, which adds the geometry's sinks.
std::string reduceKernel(std::string_view phase, const CompiledAttention &compiled, AttentionVariant variant) {
  return kernelName(phase, compiled.valuesTokenMajor ? "sparse_reduce" : "reduce",
                    splitSuffix(compiled, variant) + (compiled.sinks ? "_sinks" : ""));
}

// The q/k norm and RoPE kernel of a geometry's variant, before its norms'
// type.
std::string projectionKernel(std::string_view phase, const CompiledAttention &compiled, AttentionVariant variant) {
  return kernelName(phase, "qkv", std::string(compiled.suffix) + (variant.rotated ? "" : "_nope"));
}

// A layout's KV format, as its kernels' names hold it.
std::string kernelFormat(kv::Layout layout) {
  return layout.format == kv::Format::ZipBFloat16 ? "zip" : layout.format == kv::Format::BFloat16 ? "bf16" : "q8";
}
// The format of a layout's prefill split: ZBF16 prefill attends the BF16
// scratch its history expands into (KvZipPrefill).
std::string prefillSplitFormat(kv::Layout layout) {
  return layout.format == kv::Format::ZipBFloat16 ? "bf16" : kernelFormat(layout);
}

// The most query tiles a ZBF16 prefill's split decodes its pages for, each
// tile decoding every page it reads; a longer chunk expands its history into
// the BF16 scratch once and attends that. At four tiles (32 rows) decoding
// is 9-24% faster than expanding from 8K tokens of history on both models
// (the 35B's 5% slower at 2K); at eight, up to 15% slower on the 35B at
// 2K-8K and 2% on the 27B at 64K (M5 Pro, captured KV).
constexpr uint32_t kZipDirectTiles = 4;

// The longest lane history the 35B's ZBF16 verify takes its split with own
// page scores at (kernels/common/paged_attention_zip_tile.h); past it the
// scores share the decoded tile. Both compute the same bits.
constexpr uint32_t kZipOwnScoresHistory = 16384;

// A verify split's kernel suffix past its geometry's and window's.
std::string_view verifySplitSuffix(const CompiledAttention &compiled, kv::Layout layout,
                                   std::span<const uint32_t> historyTokens) {
  if (layout.format != kv::Format::ZipBFloat16 || compiled.queryHeads != 8 * compiled.kvHeads)
    return "";
  return std::ranges::max(historyTokens) <= kZipOwnScoresHistory ? "_short" : "";
}

// ZBF16 kernels bind the codec before their parameters.
void requireCodec(kv::Layout layout, const metal::MetalBuffer &codec) {
  if (layout.format == kv::Format::ZipBFloat16 && !codec)
    throw std::invalid_argument("ZBF16 attention requires the KV codec buffer");
}

// ZBF16 prefill's codec, BF16 scratch and its table, whole.
void requireZipPrefill(kv::Layout layout, const KvZipPrefill &zip) {
  if (layout.format != kv::Format::ZipBFloat16)
    return;
  requireCodec(layout, zip.codec);
  const kv::Layout scratch = kv::zipScratchLayout(layout);
  requireBytes(zip.scratch, uint64_t{kv::zipScratchPages()} * scratch.bytesPerModelPage(),
               "ZBF16 prefill scratch");
  requireBytes(zip.scratchTable, uint64_t{kv::zipScratchPages()} * sizeof(SplashKvPage),
               "ZBF16 prefill scratch table");
}

// Where the scratch's only layer lies in its single extent.
SplashKvLayer zipScratchLayer(kv::Layout layout) {
  const kv::Layout scratch = kv::zipScratchLayout(layout);
  return splash_kv_layer(kv::zipScratchPages(), static_cast<uint32_t>(scratch.dataBytesPerLayerPage()), 0, 0);
}

// The pages a chunk's rows touch: a ZBF16 store encodes one page's rows of a
// KV head per threadgroup.
uint32_t chunkPages(const kv::ChunkedPrefillParams &params) {
  const uint32_t last = params.committed_tokens + params.chunk_tokens - 1;
  return last / kv::kPageTokens - params.committed_tokens / kv::kPageTokens + 1;
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

// A packed projection row: every query head's q and, gated, its gate, then
// the K and V heads. The gates read each row's query heads only.
uint64_t packedWidth(const CompiledAttention &compiled) {
  return (uint64_t{compiled.queryHeads} * (compiled.gated ? 2 : 1) + uint64_t{compiled.kvHeads} * 2) *
         compiled.headDimension;
}
uint64_t gateBytes(uint64_t rows, const CompiledAttention &compiled) {
  return rowBytes(rows, packedWidth(compiled), uint64_t{compiled.queryHeads} * 2 * compiled.headDimension, 2);
}

// The q/k projection's bindings: the packed rows and norms, a rotated
// variant's RoPE tables of `rows` rows, then the staging of the queries, keys
// and values.
std::vector<metal::MetalBuffer> projectionBindings(const CompiledAttention &compiled, AttentionVariant variant,
                                                   uint64_t rows, metal::MetalBuffer packed,
                                                   const NormWeights &queryNorm, const NormWeights &keyNorm,
                                                   metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                                                   metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
                                                   metal::MetalBuffer chunkValues) {
  std::vector<metal::MetalBuffer> bindings{std::move(packed), queryNorm.buffer, keyNorm.buffer};
  if (variant.rotated) {
    const uint64_t bytes = rows * compiled.rotaryPairs * sizeof(float);
    requireBytes(ropeCos, bytes, "attention RoPE cosine");
    requireBytes(ropeSin, bytes, "attention RoPE sine");
    bindings.insert(bindings.end(), {std::move(ropeCos), std::move(ropeSin)});
  }
  bindings.insert(bindings.end(), {std::move(queries), std::move(chunkKeys), std::move(chunkValues)});
  return bindings;
}

// A plan whose reduce adds the geometry's sinks takes the layer's, an F32 per
// query head; another takes none.
void requireSinks(const metal::MetalBuffer &sinks, bool takes, uint32_t queryHeads) {
  if (takes)
    requireBytes(sinks, uint64_t{queryHeads} * sizeof(float), "attention sinks");
  else if (sinks)
    throw std::invalid_argument("the attention kernels take no sinks");
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

// A reduce threadgroup combines one fused row's splits, a thread per
// dimension.
metal::DispatchSize reduceThreads(kv::Layout layout) { return {layout.headDimension, 1, 1}; }

// Each query tile's history splits: the maximum shared out over the chunk's
// tiles, at least one, and with a window of windowTokens keys before each
// row no more than the pages it reads (a split each).
uint32_t prefillSplits(uint32_t tiles, uint32_t windowTokens) {
  constexpr uint32_t maximum = SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS;
  const uint32_t splits = std::clamp(maximum / tiles, 1U, maximum);
  return windowTokens ? std::min(splits, kv::windowAttentionPages(windowTokens)) : splits;
}

} // namespace

PrefillAttentionPlan PagedAttention::prefillPlan(
    uint32_t rows, uint32_t queryHeads, kv::Layout layout, AttentionVariant variant) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout, variant);
  if (!rows || rows > kv::kChunkedPrefillMaximumRows)
    throw std::invalid_argument("invalid attention workspace rows");
  const uint32_t tiles = kv::prefillAttentionTiles(rows);
  const uint32_t splits = prefillSplits(tiles, variant.windowTokens);
  const uint32_t fusedRows =
      kv::kPrefillAttentionTileRows * (queryHeads / layout.kvHeads);
  const bool zipExpand = layout.format == kv::Format::ZipBFloat16 && tiles > kZipDirectTiles;
  return {rows, splits,
          attentionWorkspace(uint64_t{tiles} * splits * layout.kvHeads * fusedRows,
                             layout.headDimension),
          kernelName("prefill",
                     (zipExpand ? prefillSplitFormat(layout) : kernelFormat(layout)) +
                         (kernel.valuesTokenMajor ? "_sparse" : "_split"),
                     splitSuffix(kernel, variant)),
          reduceKernel("prefill", kernel, variant),
          {layout.kvHeads * tileParts(queryHeads, layout), tiles, splits},
          {layout.kvHeads, fusedRows, tiles}, layout.valuesTokenMajor, kernel.sinks, zipExpand,
          layout, queryHeads};
}

VerifyAttentionPlan PagedAttention::verifyPlan(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens, AttentionVariant variant) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout, variant);
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
    laneSplits[lane] = kv::verifyAttentionSplits(historyTokens[lane], variant.windowTokens);
    splits = std::max(splits, laneSplits[lane]);
  }
  return {lanes, laneSplits, splits,
          verifyWorkspaceBound(lanes, queryHeads, layout),
          kernelName("verify", kernelFormat(layout) + (kernel.valuesTokenMajor ? "_sparse" : "_split"),
                     splitSuffix(kernel, variant) + std::string(verifySplitSuffix(kernel, layout, historyTokens))),
          reduceKernel("verify", kernel, variant),
          // A sparse lane's list fills only the first of the plan's splits.
          // Its split grid is one-dimensional, x the KV head and then the
          // split: an M5 Max spread the same order as a two-dimensional grid
          // (x the KV head, y the split) worse, running it up to 1.7x slower
          // on most grids of more than 32 splits.
          layout.valuesTokenMajor ? metal::DispatchSize{layout.kvHeads * splits, 1, lanes}
                                  : metal::DispatchSize{layout.kvHeads * tileParts(queryHeads, layout), splits, lanes},
          {layout.kvHeads,
           kv::kVerifyRows * (queryHeads / layout.kvHeads), lanes},
          layout.valuesTokenMajor, kernel.sinks,
          kernel.valuesTokenMajor ? kernelName("verify", kernelFormat(layout) + "_sparse_located", kernel.suffix) : "",
          kernelName("verify", kernelFormat(layout) + "_store", storeSuffix(layout)),
          // ZBF16: per lane, a threadgroup per (tensor, KV head) of the two
          // pages eight rows can touch; otherwise one per row.
          {uint64_t{lanes} * 2 * (layout.format == kv::Format::ZipBFloat16 ? 2 : kv::kVerifyRows) * layout.kvHeads,
           1, 1},
          {layout.headDimension, 1, 1}, layout, queryHeads};
}

AttentionWorkspace PagedAttention::prefillWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) {
  (void)compiledAttention(queryHeads, layout);
  if (!maximumRows || maximumRows > kv::kChunkedPrefillMaximumRows)
    throw std::invalid_argument("invalid attention workspace rows");
  // Allocation-time bound for every shorter sequence and every variant, a
  // window taking no more splits than the whole history. Encoding resolves
  // its own exact plan in constant time; no bound scan occurs on the hot
  // path.
  uint64_t slots = 0;
  for (uint32_t tiles = 1; tiles <= kv::prefillAttentionTiles(maximumRows); ++tiles)
    slots = std::max(slots, uint64_t{tiles} * prefillSplits(tiles, 0));
  return attentionWorkspace(slots * kv::kPrefillAttentionTileRows * queryHeads,
                             layout.headDimension);
}

AttentionWorkspace PagedAttention::verifyWorkspace(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout) {
  (void)compiledAttention(queryHeads, layout);
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid attention workspace batch width");
  return verifyWorkspaceBound(lanes, queryHeads, layout);
}

bool PagedAttention::compiles(uint32_t queryHeads, kv::Layout layout, bool gated, uint32_t rotaryPairs) {
  const CompiledAttention &compiled = compiledAttention(queryHeads, layout);
  return compiled.gated == gated && compiled.rotaryPairs == rotaryPairs;
}

void PagedAttention::addPrefillProjection(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    const NormWeights &queryNorm, const NormWeights &keyNorm,
    metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
    metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
    metal::MetalBuffer chunkValues, uint32_t tokens, uint32_t stride,
    uint32_t queryHeads, kv::Layout layout, AttentionVariant variant) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout, variant);
  if (!tokens || !stride)
    throw std::invalid_argument("invalid paged prefill projection geometry");
  const Staging staging{layout.kvHeads, stride, tokens};
  requireBytes(packed, uint64_t{tokens} * packedWidth(kernel) * 2, "attention q/k/v");
  requireBytes(queries, staging.rowBytes(queryHeads / layout.kvHeads, layout.headDimension), "attention query");
  requireBytes(chunkKeys, staging.rowBytes(1, layout.headDimension), "attention key");
  requireBytes(chunkValues, staging.valueBytes(layout.headDimension), "attention value");
  const FullPrefillParams params{tokens, stride};
  graph.add(qkNormKernel(projectionKernel("prefill", kernel, variant), queryNorm, keyNorm, layout.headDimension),
            projectionBindings(kernel, variant, tokens, std::move(packed), queryNorm, keyNorm, std::move(ropeCos),
                               std::move(ropeSin), std::move(queries), std::move(chunkKeys),
                               std::move(chunkValues)),
            params, {uint64_t{tokens} * (queryHeads + layout.kvHeads), 1, 1}, {layout.headDimension, 1, 1});
}

void PagedAttention::addPrefillGate(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer attention, metal::MetalBuffer hidden, uint32_t tokens,
    uint32_t stride, uint32_t queryHeads, kv::Layout layout) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout);
  if (!tokens || !stride)
    throw std::invalid_argument("invalid paged prefill gate geometry");
  requireBytes(attention,
               Staging{layout.kvHeads, stride, tokens}.rowBytes(queryHeads / layout.kvHeads, layout.headDimension),
               "attention row");
  requireBytes(hidden, uint64_t{tokens} * queryHeads * layout.headDimension * 2, "attention hidden");
  const FullPrefillParams params{tokens, stride};
  const metal::DispatchSize groups{gateGroups(tokens, queryHeads, layout.headDimension), 1, 1};
  if (!kernel.gated) {
    graph.add(kernelName("prefill", "gather", kernel.suffix), {std::move(attention), std::move(hidden)}, params,
              groups);
    return;
  }
  requireBytes(packed, gateBytes(tokens, kernel), "attention gate");
  graph.add(kernelName("prefill", "gate", kernel.suffix),
            {std::move(packed), std::move(attention), std::move(hidden)}, params, groups);
}

void PagedAttention::addVerifyProjection(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    const NormWeights &queryNorm, const NormWeights &keyNorm,
    metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
    metal::MetalBuffer queries, metal::MetalBuffer chunkKeys,
    metal::MetalBuffer chunkValues, uint32_t queryHeads, kv::Layout layout,
    uint32_t lanes, AttentionVariant variant) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout, variant);
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid paged verify projection geometry");
  const uint64_t rows = uint64_t{lanes} * kv::kVerifyRows;
  const Staging staging = verifyStaging(lanes, layout);
  requireBytes(packed, rows * packedWidth(kernel) * 2, "attention q/k/v");
  requireBytes(queries, staging.rowBytes(queryHeads / layout.kvHeads, layout.headDimension), "attention query");
  requireBytes(chunkKeys, staging.rowBytes(1, layout.headDimension), "attention key");
  requireBytes(chunkValues, staging.valueBytes(layout.headDimension), "attention value");
  graph.add(qkNormKernel(projectionKernel("verify", kernel, variant), queryNorm, keyNorm, layout.headDimension),
            projectionBindings(kernel, variant, rows, std::move(packed), queryNorm, keyNorm, std::move(ropeCos),
                               std::move(ropeSin), std::move(queries), std::move(chunkKeys),
                               std::move(chunkValues)),
            {uint64_t{SPLASH_TARGET_VERIFY_ROWS} * (queryHeads + layout.kvHeads), lanes, 1},
            {layout.headDimension, 1, 1});
}

PreparedInput PagedAttention::addVerifyGate(
    metal::CommandGraph &graph, metal::MetalBuffer packed,
    metal::MetalBuffer attention, metal::MetalBuffer hidden,
    uint32_t queryHeads, kv::Layout layout, uint32_t lanes, LinearScratch scratch,
    LinearInput input) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout);
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid paged verify gate geometry");
  const uint32_t rows = lanes * SPLASH_TARGET_VERIFY_ROWS;
  if (kernel.gated) requireBytes(packed, gateBytes(rows, kernel), "attention gate");
  requireBytes(attention, verifyStaging(lanes, layout).rowBytes(queryHeads / layout.kvHeads, layout.headDimension),
               "attention row");
  requireBytes(hidden, uint64_t{rows} * queryHeads * layout.headDimension * 2, "attention hidden");
  // An ungated geometry's gather reads no packed row.
  const std::string name = kernel.gated ? "gate" : "gather";
  std::vector<metal::MetalBuffer> bindings{std::move(attention), hidden};
  if (kernel.gated) bindings.insert(bindings.begin(), std::move(packed));
  if (input != LinearInput::Plain) {
    const uint32_t width = queryHeads * layout.headDimension;
    requireTableScratch(scratch, input, width, rows);
    bindings.insert(bindings.end(), {scratch.input, scratch.sums});
    graph.add(kernelName("verify", name + tableSuffix(input), kernel.suffix), std::move(bindings),
              {uint64_t{width} / 64 * lanes, 1, 1}, {256, 1, 1});
    return {std::move(hidden), input};
  }
  graph.add(kernelName("verify", name, kernel.suffix), std::move(bindings), FullDecodeBatchParams{lanes},
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
    const kv::ChunkedPrefillParams &params, kv::Layout layout, const KvZipPrefill &zip) {
  requireZipPrefill(layout, zip);
  const std::string store = kernelName("prefill", kernelFormat(layout) + "_store", storeSuffix(layout));
  const Staging staging{layout.kvHeads, params.chunk_stride, params.chunk_tokens};
  requireBytes(chunkKeys, staging.rowBytes(1, layout.headDimension), "attention key");
  requireBytes(chunkValues, staging.valueBytes(layout.headDimension), "attention value");
  requirePageTable(pageTable, params);
  kv::ChunkedPrefillParams layerParams = params;
  layerParams.kv = layer;
  if (layout.format == kv::Format::ZipBFloat16) {
    // A threadgroup encodes a KV head's rows of one page.
    graph.add(std::string(store), {std::move(chunkKeys), std::move(chunkValues), std::move(pageTable), zip.codec},
              layerParams, {uint64_t{2} * chunkPages(params) * layout.kvHeads, 1, 1},
              {layout.headDimension, 1, 1});
    return;
  }
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
    const PrefillAttentionPlan &plan, AttentionBlocks blocks, metal::MetalBuffer sinks,
    const KvZipPrefill &zip) {
  requireZipPrefill(plan.layout, zip);
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
  requireSinks(sinks, plan.sinks, plan.queryHeads);
  // A long ZBF16 chunk attends through the BF16 split and reduce: the history
  // it reads here, the chunk's rows included, expands into the BF16 scratch
  // first.
  const bool expands = plan.zipExpand;
  if (expands) {
    kv::ChunkedPrefillParams expand = chunk;
    expand.kv = layer;
    const uint32_t visiblePages = kv::chunkedPrefillRequiredPages(chunk);
    graph.add(kernelName("prefill", "zip_expand", storeSuffix(plan.layout)),
              {pageTable, zip.scratchTable, zip.codec}, SplashKvZipExpandParams{expand, zipScratchLayer(plan.layout)},
              {uint64_t{2} * visiblePages * plan.layout.kvHeads, 1, 1}, {plan.layout.headDimension, 1, 1});
  }
  const kv::PrefillAttentionParams params{
      chunk.committed_tokens, chunk.chunk_tokens, chunk.chunk_stride,
      chunk.page_table_entries, expands ? zipScratchLayer(plan.layout) : layer, plan.splits, blocks.stride,
      blocks.list};
  std::vector<metal::MetalBuffer> split{std::move(queries), partials, statistics,
                                        expands ? zip.scratchTable : std::move(pageTable)};
  if (plan.layout.format == kv::Format::ZipBFloat16 && !expands) split.push_back(zip.codec);
  std::vector<metal::MetalBuffer> reduce{partials, statistics, std::move(output)};
  if (plan.sparse) {
    const auto [lists, counts] = blockBindings(blocks, partials);
    split.insert(split.end(), {lists, counts});
    reduce.push_back(counts);
  }
  if (plan.sinks) reduce.push_back(std::move(sinks));
  graph.add(std::string(plan.splitPipeline), std::move(split), params, plan.splitGroups);
  graph.add(std::string(plan.reducePipeline), std::move(reduce), params, plan.reduceGroups,
            reduceThreads(plan.layout));
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
  requireSinks(buffers.sinks, plan.sinks, plan.queryHeads);
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
  requireCodec(plan.layout, buffers.kvCodec);
  const bool zipped = plan.layout.format == kv::Format::ZipBFloat16;
  const auto &tables = buffers.pageTables;
  std::vector<metal::MetalBuffer> store{buffers.chunkKeys, buffers.chunkValues, tables[0], tables[1], tables[2],
                                        tables[3]};
  if (zipped) store.push_back(buffers.kvCodec);
  graph.add(std::string(plan.storePipeline_), std::move(store), stores, plan.storeGroups_, plan.storeThreads_);
  std::vector<metal::MetalBuffer> split{buffers.queries, buffers.partials, buffers.statistics,
                                        tables[0], tables[1], tables[2], tables[3]};
  if (zipped) split.push_back(buffers.kvCodec);
  std::vector<metal::MetalBuffer> reduce{buffers.partials, buffers.statistics, buffers.output};
  if (plan.sparse) {
    const auto [lists, counts] = blockBindings(blocks, buffers.partials);
    split.insert(split.end(), {lists, counts});
    if (blocks.locations) split.insert(split.end(), {blocks.locations, blocks.locationPages});
    reduce.push_back(counts);
  }
  if (plan.sinks) reduce.push_back(buffers.sinks);
  graph.add(std::string(blocks.locations ? plan.locatedSplitPipeline_ : plan.splitPipeline), std::move(split),
            attention, plan.splitGroups);
  graph.add(std::string(plan.reducePipeline), std::move(reduce), attention, plan.reduceGroups,
            reduceThreads(plan.layout));
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

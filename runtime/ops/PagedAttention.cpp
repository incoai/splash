#include "ops/PagedAttention.hpp"

#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {

namespace {

constexpr uint32_t kMaximumSequences = SPLASH_MAXIMUM_BATCH_WIDTH;

// Whether a layout's attention reads the blocks its index selects, through
// the sparse kernels and block lists.
bool selectsBlocks(kv::Layout layout) { return layout.reads == kv::Reads::SelectedBlocks; }

// The parts a dense tile of a layout's GQA group runs in, each a threadgroup
// of the split (SPLASH_ATTENTION_TILE_PARTS); a sparse tile is one.
uint32_t tileParts(uint32_t queryHeads, kv::Layout layout) {
  return selectsBlocks(layout) ? 1 : SPLASH_ATTENTION_TILE_PARTS(queryHeads / layout.kvHeads);
}

// One thread per gate element, across rows x query heads x head dimension.
uint64_t gateGroups(uint64_t rows, uint32_t queryHeads, uint32_t headDimension) {
  const uint64_t elements = rows * queryHeads * headDimension;
  return (elements + metal::CommandGraph::kDefaultThreads - 1) /
         metal::CommandGraph::kDefaultThreads;
}

// The compiled attention geometries (kernels/shared/attention_qkv.metal,
// paged_attention.metal, paged_attention_store.metal and
// sparse_attention.metal): query heads over KV heads of a head dimension and
// how they read the pages, and their kernels' name suffix, the stores' by KV
// heads, head dimension and the value order of the reader. The sparse
// kernels read selected blocks of token-major values (Qwen3.8-Flash-Next's);
// the dense page loop reads every page of dimension-major ones. Each
// geometry's kernels are compiled for:
// - gated: projection rows that hold each query head's output gate after its
//   queries, by whose sigmoid the output multiplies the attention rows
//   (Qwen3.5's); without, the output gathers them as they are;
// - sinks: a learned sink per query head, which every reduce adds;
// - rotaryPairs: the pairs its RoPE rotates, and unrotated: a projection
//   without RoPE (NoPE) as well, for an AttentionVariant that is not rotated;
// - windowTokens: a split and reduce over a window of that many keys before
//   each row as well as over the whole history, for an AttentionVariant of
//   that window; 0 without.
// Its q/k norm kernels take the norm types its families store (normKernel):
// F32 and bf16, or F32 alone for the geometries of families that load from
// GGUFs only (Qwen3.8-Flash-Next's, Flint's).
struct CompiledAttention final {
  uint32_t queryHeads;
  uint32_t kvHeads;
  uint32_t headDimension;
  kv::Reads reads;
  bool gated;
  bool sinks;
  uint32_t rotaryPairs;
  bool unrotated;
  uint32_t windowTokens;
  std::string_view suffix;
  std::string_view storeSuffix;
};
constexpr std::array kCompiledAttention{
    CompiledAttention{24, 4, 256, kv::Reads::EveryPage, true, false, 32, false, 0, "", ""},
    CompiledAttention{16, 2, 256, kv::Reads::EveryPage, true, false, 32, false, 0, "_kv2_g8", "_kv2_g8"},
    CompiledAttention{32, 2, 256, kv::Reads::EveryPage, true, false, 32, false, 0, "_kv2_g16", "_kv2_g8"},
    CompiledAttention{24, 2, 256, kv::Reads::SelectedBlocks, true, false, 32, false, 0, "_kv2_g12", "_kv2_vt"},
    CompiledAttention{32, 4, 128, kv::Reads::EveryPage, false, true, 64, true, 512, "_kv4_g8_d128", "_kv4_d128"},
};

// Whether a compiled geometry's pages are a layout's: its KV heads, head
// dimension and value order.
bool storesLayout(const CompiledAttention &compiled, kv::Layout layout) {
  return compiled.kvHeads == layout.kvHeads && compiled.headDimension == layout.headDimension &&
         kv::valuesTokenMajorFor(compiled.reads) == layout.valuesTokenMajor;
}

// The KV store kernels' suffix of a layout's KV heads, head dimension and
// value order.
std::string_view storeSuffix(kv::Layout layout) {
  if (layout.valid())
    for (const CompiledAttention &compiled : kCompiledAttention)
      if (storesLayout(compiled, layout)) return compiled.storeSuffix;
  throw std::invalid_argument("no KV store kernel for layout");
}

// The compiled geometry of `queryHeads` over a layout's KV heads that reads
// its pages as it does, so the kernels need no geometry branches, which must
// compile a layer's variant.
const CompiledAttention &compiledAttention(uint32_t queryHeads, kv::Layout layout,
                                           AttentionVariant variant = {}) {
  static_cast<void>(storeSuffix(layout));
  for (const CompiledAttention &compiled : kCompiledAttention) {
    if (compiled.queryHeads != queryHeads || compiled.reads != layout.reads || !storesLayout(compiled, layout))
      continue;
    if ((!variant.rotated && !compiled.unrotated) ||
        (variant.windowTokens && variant.windowTokens != compiled.windowTokens))
      throw std::invalid_argument("no attention kernel for the layer's variant");
    return compiled;
  }
  throw std::invalid_argument(selectsBlocks(layout) ? "no sparse attention kernel for layout"
                                                    : "no paged-attention kernel for layout");
}

// The attention kernel `name` with a suffix.
std::string kernelName(std::string_view name, std::string_view suffix) {
  return "attention_" + std::string(name) + std::string(suffix);
}

// The split and reduce kernels' suffix of a geometry's variant: its window's.
std::string splitSuffix(const CompiledAttention &compiled, AttentionVariant variant) {
  return std::string(compiled.suffix) + (variant.windowTokens ? "_w" + std::to_string(variant.windowTokens) : "");
}

// A layout's KV format, as its kernels' names hold it.
std::string kernelFormat(kv::Layout layout) { return layout.format == kv::Format::BFloat16 ? "bf16" : "q8"; }

// The pages a tile that sees visibleTokens reads: with a window of
// windowTokens keys before each row, no more than its window's.
uint64_t readPages(uint64_t visibleTokens, uint32_t windowTokens) {
  const uint64_t seen = (visibleTokens + kv::kPageTokens - 1) / kv::kPageTokens;
  return windowTokens ? std::min<uint64_t>(seen, kv::windowAttentionPages(windowTokens)) : seen;
}

// The history splits of each of a sequence's `tiles` query tiles, which read
// `pages` pages. A sequence of one tile (a decode lane's of up to 8 rows)
// takes a split per kTilePagesPerSplit pages, between kSharedSplits and
// kAttentionMaximumSplits. A longer one, a lane's of 9 to 16 rows too, takes
// a whole split per pagesPerSplit pages (AttentionCalibration), at least
// kSharedSplits shared out over its tiles (all a short history takes) and at
// most kSharedSplits, nor over all its tiles more than
// kAttentionSequenceSplits: taking a tile's splits ran a sequence of two to
// four tiles up to 9% slower (on an M6 at 128K tokens). With a window, no
// more than the pages it reads (a split each). It depends on the sequence
// alone, so batching never changes a sequence's partition.
constexpr uint32_t kSharedSplits = 32;
constexpr uint32_t kTilePagesPerSplit = 16;
uint32_t attentionSplits(uint32_t tiles, uint64_t pages, uint32_t windowTokens, uint32_t pagesPerSplit) {
  const uint32_t fewest = std::clamp(kSharedSplits / tiles, 1U, kSharedSplits);
  const uint32_t most = std::clamp(kAttentionSequenceSplits / tiles, fewest, kSharedSplits);
  const uint32_t splits = static_cast<uint32_t>(
      tiles == 1 ? std::clamp<uint64_t>((pages + kTilePagesPerSplit - 1) / kTilePagesPerSplit, kSharedSplits,
                                        kv::kAttentionMaximumSplits)
                 : std::clamp<uint64_t>(pages / pagesPerSplit, fewest, most));
  return windowTokens ? std::min(splits, kv::windowAttentionPages(windowTokens)) : splits;
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

// The fused rows of one tile split's partials, of every KV head and part.
uint64_t slotRows(uint32_t queryHeads) { return uint64_t{kv::kAttentionTileRows} * queryHeads; }

// The command's rows: each sequence's first packed row and first staged row,
// the rows and staged KV-head rows of the sequences before it; and the last
// sequence's staged rows, which end the staging its kernels reach.
struct CommandRows final {
  std::array<uint32_t, kMaximumSequences> rowBegin{};
  std::array<uint32_t, kMaximumSequences> stageRow{};
  uint32_t rows = 0;
  uint64_t staged = 0;
  kv::ChunkedPrefillParams last{};
  uint32_t kvHeads = 0;

  // The staging's bytes up to the last sequence's last KV head's first
  // `reach` rows, `width` elements each (queries, keys and attention rows),
  // or, transposed, of its values.
  [[nodiscard]] uint64_t rowBytes(uint64_t width, uint32_t reach) const noexcept {
    return (staged - kvHeads * uint64_t{last.chunk_stride} + (kvHeads - 1) * uint64_t{last.chunk_stride} + reach) *
           width * sizeof(uint16_t);
  }
  [[nodiscard]] uint64_t valueBytes(uint32_t headDimension) const noexcept {
    const uint64_t first = (staged - kvHeads * uint64_t{last.chunk_stride}) * headDimension;
    return (first + (uint64_t{kvHeads} * headDimension - 1) * last.chunk_stride + last.chunk_tokens) *
           sizeof(uint16_t);
  }
};
CommandRows commandRows(std::span<const kv::ChunkedPrefillParams> sequences, kv::Layout layout) {
  if (sequences.empty() || sequences.size() > kMaximumSequences)
    throw std::invalid_argument("invalid attention sequences");
  CommandRows result;
  for (size_t s = 0; s < sequences.size(); ++s) {
    const std::string_view error = kv::chunkedPrefillValidationError(sequences[s]);
    if (!error.empty()) throw std::invalid_argument(std::string(error));
    result.rowBegin[s] = result.rows;
    result.stageRow[s] = static_cast<uint32_t>(result.staged);
    result.rows += sequences[s].chunk_tokens;
    result.staged += uint64_t{layout.kvHeads} * sequences[s].chunk_stride;
  }
  result.last = sequences.back();
  result.kvHeads = layout.kvHeads;
  if (result.staged > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("attention staging exceeds the kernel ABI");
  return result;
}

// Sequence s's rows from `first` on, the dispatch's from firstRow on.
SplashAttentionSequence sequenceRows(const kv::ChunkedPrefillParams &chunk, const CommandRows &rows, size_t s,
                                     uint32_t first, uint32_t firstRow) {
  return {.committed_tokens = chunk.committed_tokens + first,
          .rows = chunk.chunk_tokens - first,
          .first_row = firstRow,
          .row_begin = rows.rowBegin[s] + first,
          .stage_row = rows.stageRow[s] + first,
          .chunk_stride = chunk.chunk_stride,
          .page_table_entries = chunk.page_table_entries,
          .split_count = 0,
          .first_tile = 0,
          .first_slot = 0,
          .block_stride = 0,
          .block_list = 0};
}

// The parameters of every row of the command's sequences, in layer `layer`.
kv::AttentionParams everyRow(std::span<const kv::ChunkedPrefillParams> sequences, const CommandRows &rows,
                             SplashKvLayer layer = {}) {
  kv::AttentionParams params{.sequences = static_cast<uint32_t>(sequences.size()),
                             .rows = rows.rows,
                             .tiles = 0,
                             .trees = 0,
                             .kv = layer,
                             .sequence = {}};
  for (size_t s = 0; s < sequences.size(); ++s) {
    params.sequence[s] = sequenceRows(sequences[s], rows, s, 0, rows.rowBegin[s]);
    params.tiles += kv::attentionTiles(sequences[s].chunk_tokens);
  }
  return params;
}

// The staging of a command's rows its kernels reach, of `reach` rows of the
// last sequence's: queries and attention rows hold a KV head's group of query
// heads a row, keys one head, and values a row per dimension.
void requireStaging(const AttentionStaging &staging, const CommandRows &rows, uint32_t group, kv::Layout layout,
                    uint32_t reach, bool keys, bool queries) {
  const uint64_t row = layout.headDimension;
  if (keys) {
    requireBytes(staging.keys, rows.rowBytes(row, reach), "attention key");
    requireBytes(staging.values, rows.valueBytes(layout.headDimension), "attention value");
  }
  if (queries) requireBytes(staging.queries, rows.rowBytes(group * row, reach), "attention query");
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
                                                   const AttentionStaging &staging) {
  std::vector<metal::MetalBuffer> bindings{std::move(packed), queryNorm.buffer, keyNorm.buffer};
  if (variant.rotated) {
    const uint64_t bytes = rows * compiled.rotaryPairs * sizeof(float);
    requireBytes(ropeCos, bytes, "attention RoPE cosine");
    requireBytes(ropeSin, bytes, "attention RoPE sine");
    bindings.insert(bindings.end(), {std::move(ropeCos), std::move(ropeSin)});
  }
  bindings.insert(bindings.end(), {staging.queries, staging.keys, staging.values});
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

// A sequence's tiles read lists `blocks.list` to `blocks.list` + `tiles`, and
// only a sparse plan takes lists.
void requireBlocks(const AttentionBlocks &blocks, bool sparse, uint64_t tiles) {
  if (!blocks.stride) return;
  if (!sparse) throw std::invalid_argument("the attention kernels take no block lists");
  const uint64_t lists = uint64_t{blocks.list} + tiles;
  requireBytes(blocks.lists, lists * blocks.stride * sizeof(uint32_t), "attention block lists");
  requireBytes(blocks.counts, lists * sizeof(uint32_t), "attention block list counts");
  if (!blocks.locations) return;
  if (!blocks.locationPages) throw std::invalid_argument("attention block locations need their pages");
  requireBytes(blocks.locations, lists * blocks.stride * sizeof(uint32_t), "attention block locations");
}

// Whether two sequences' blocks bind as one dispatch's: the same lists,
// counts and locations, or none.
bool sameLayer(SplashKvLayer a, SplashKvLayer b) { return a.extent_pages == b.extent_pages && a.offset == b.offset; }

bool sameBindings(const AttentionBlocks &a, const AttentionBlocks &b) {
  return a.lists.sameView(b.lists) && a.counts.sameView(b.counts) && a.locations.sameView(b.locations) &&
         a.locationPages.sameView(b.locationPages) && a.rowMasks.sameView(b.rowMasks);
}

// A reduce threadgroup combines one fused row's splits, a thread per
// dimension.
metal::DispatchSize reduceThreads(kv::Layout layout) { return {layout.headDimension, 1, 1}; }

} // namespace

AttentionWorkspace AttentionPlan::workspace() const noexcept {
  uint64_t slots = 0;
  for (const Dispatch &dispatch : dispatches()) slots = std::max<uint64_t>(slots, dispatch.slots);
  return attentionWorkspace(slots * slotRows(queryHeads), layout.headDimension);
}

// A sparse sequence's list fills only the first of its splits: the sparse
// split grid is one-dimensional, x the KV head and then the split
// (sparse_attention.metal).
metal::DispatchSize AttentionPlan::splitGroups(const Dispatch &dispatch) const noexcept {
  if (sparse) return {uint64_t{layout.kvHeads} * dispatch.splits, dispatch.tiles, 1};
  return {uint64_t{layout.kvHeads} * tileParts(queryHeads, layout), dispatch.tiles, dispatch.splits};
}

metal::DispatchSize AttentionPlan::reduceGroups(const Dispatch &dispatch) const noexcept {
  return {layout.kvHeads, kv::kAttentionTileRows * (queryHeads / layout.kvHeads), dispatch.tiles};
}

AttentionPlan PagedAttention::plan(std::span<const kv::ChunkedPrefillParams> sequences, uint32_t queryHeads,
                                   kv::Layout layout, const AttentionCalibration &calibration,
                                   Int8Activations int8Activations, AttentionWorkspace capacity,
                                   AttentionVariant variant) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout, variant);
  static_cast<void>(commandRows(sequences, layout));
  const AttentionWorkspace slot = attentionWorkspace(slotRows(queryHeads), layout.headDimension);
  const uint64_t slots = std::min(capacity.partialsBytes / slot.partialsBytes,
                                  capacity.statisticsBytes / slot.statisticsBytes);
  const bool sparse = selectsBlocks(layout);
  AttentionPlan plan(layout, queryHeads, sparse, kernel.sinks,
                     static_cast<uint32_t>(std::min<uint64_t>(slots, std::numeric_limits<uint32_t>::max())));
  uint64_t splitReads = 0;
  for (const kv::ChunkedPrefillParams &chunk : sequences) {
    const uint32_t tiles = kv::attentionTiles(chunk.chunk_tokens);
    // A sparse tile reads its list's blocks, a few thousand tokens whatever
    // its history: a sequence of more than one tile keeps the shared splits
    // (by its history it ran up to 47% slower: Qwen3.8-Flash-Next's 64-row
    // chunk after 64K tokens), and a decode lane's one tile splits by its
    // history, which caps how far a long list spreads.
    const uint64_t pages = sparse && tiles > 1
                               ? 0
                               : readPages(uint64_t{chunk.committed_tokens} + chunk.chunk_tokens,
                                           variant.windowTokens);
    const uint32_t splits = attentionSplits(tiles, pages, variant.windowTokens, calibration.prefillPagesPerSplit);
    splitReads = std::max(splitReads, (pages + splits - 1) / splits);
    plan.sequences_[plan.sequenceCount_++] = {chunk.chunk_tokens, chunk.committed_tokens, 0, splits};
  }
  const std::string suffix = splitSuffix(kernel, variant);
  const bool int8Queries = !sparse && int8Activations == Int8Activations::On && layout.format == kv::Format::Int8 &&
                           splitReads >= calibration.int8QueriesFromPages;
  plan.splitPipeline_ =
      kernelName((int8Queries ? "q8q" : kernelFormat(layout)) + (sparse ? "_sparse" : "_split"), suffix);
  if (sparse) plan.locatedSplitPipeline_ = kernelName(kernelFormat(layout) + "_sparse_located", kernel.suffix);
  plan.reducePipeline_ =
      kernelName(sparse ? "sparse_reduce" : "reduce", suffix + (kernel.sinks ? "_sinks" : ""));
  group(plan);
  return plan;
}

AttentionPlan PagedAttention::lastTiles(const AttentionPlan &plan) {
  AttentionPlan tails = plan;
  for (AttentionPlan::Sequence &sequence : std::span(tails.sequences_).first(tails.sequenceCount_))
    sequence.firstTile = kv::attentionTiles(sequence.rows) - 1;
  group(tails);
  return tails;
}

void PagedAttention::group(AttentionPlan &plan) {
  plan.dispatchCount_ = 0;
  for (uint32_t s = 0; s < plan.sequenceCount_; ++s) {
    const AttentionPlan::Sequence &sequence = plan.sequences_[s];
    const uint32_t tiles = kv::attentionTiles(sequence.rows) - sequence.firstTile;
    const uint32_t slots = tiles * sequence.splits;
    if (slots > plan.workspaceSlots)
      throw std::invalid_argument("the attention workspace holds no sequence's splits");
    // A sequence that does not fit the last dispatch starts the next.
    if (!plan.dispatchCount_ || plan.dispatches_[plan.dispatchCount_ - 1].slots + slots > plan.workspaceSlots)
      plan.dispatches_[plan.dispatchCount_++] = {};
    AttentionPlan::Dispatch *dispatch = &plan.dispatches_[plan.dispatchCount_ - 1];
    dispatch->sequence[dispatch->sequences++] = s;
    dispatch->tiles += tiles;
    dispatch->splits = std::max(dispatch->splits, sequence.splits);
    dispatch->slots += slots;
  }
}

AttentionWorkspace PagedAttention::workspace(uint32_t slots, uint32_t queryHeads, kv::Layout layout) {
  (void)compiledAttention(queryHeads, layout);
  if (!slots) throw std::invalid_argument("invalid attention workspace slots");
  return attentionWorkspace(slots * slotRows(queryHeads), layout.headDimension);
}

bool PagedAttention::compiles(uint32_t queryHeads, kv::Layout layout, bool gated, uint32_t rotaryPairs) {
  const CompiledAttention &compiled = compiledAttention(queryHeads, layout);
  return compiled.gated == gated && compiled.rotaryPairs == rotaryPairs;
}

kv::ChunkedPrefillParams PagedAttention::chunkParams(uint64_t logicalPosition, uint32_t chunkTokens,
                                                     uint32_t chunkStride, uint32_t pageTableEntries) {
  if (logicalPosition > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("KV logical position exceeds kernel ABI");
  const kv::ChunkedPrefillParams params{static_cast<uint32_t>(logicalPosition), chunkTokens, chunkStride,
                                        pageTableEntries};
  const std::string_view error = kv::chunkedPrefillValidationError(params);
  if (!error.empty())
    throw std::invalid_argument(std::string(error));
  return params;
}

void PagedAttention::addProjection(metal::CommandGraph &graph, metal::MetalBuffer packed,
                                   const NormWeights &queryNorm, const NormWeights &keyNorm,
                                   metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                                   const AttentionStaging &staging,
                                   std::span<const kv::ChunkedPrefillParams> sequences, uint32_t queryHeads,
                                   kv::Layout layout, AttentionVariant variant) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout, variant);
  const CommandRows rows = commandRows(sequences, layout);
  requireBytes(packed, uint64_t{rows.rows} * packedWidth(kernel) * 2, "attention q/k/v");
  requireStaging(staging, rows, queryHeads / layout.kvHeads, layout, rows.last.chunk_tokens, true, true);
  graph.add(qkNormKernel(kernelName("qkv", std::string(kernel.suffix) + (variant.rotated ? "" : "_nope")),
                         queryNorm, keyNorm, layout.headDimension),
            projectionBindings(kernel, variant, rows.rows, std::move(packed), queryNorm, keyNorm, std::move(ropeCos),
                               std::move(ropeSin), staging),
            everyRow(sequences, rows), {uint64_t{rows.rows} * (queryHeads + layout.kvHeads), 1, 1},
            {layout.headDimension, 1, 1});
}

void PagedAttention::addStore(metal::CommandGraph &graph, SplashKvLayer layer, const AttentionStaging &staging,
                              std::span<const metal::MetalBuffer> pageTables,
                              std::span<const kv::ChunkedPrefillParams> sequences, kv::Layout layout) {
  const std::string store = kernelName(kernelFormat(layout) + "_store", storeSuffix(layout));
  const CommandRows rows = commandRows(sequences, layout);
  if (pageTables.size() != sequences.size())
    throw std::invalid_argument("a KV store takes a page table per sequence");
  requireStaging(staging, rows, 1, layout, rows.last.chunk_tokens, true, false);
  for (size_t s = 0; s < sequences.size(); ++s) requirePageTable(pageTables[s], sequences[s]);
  const auto tables = laneBindings(pageTables);
  std::vector<metal::MetalBuffer> bindings{staging.keys, staging.values};
  bindings.insert(bindings.end(), tables.begin(), tables.end());
  graph.add(store, std::move(bindings), everyRow(sequences, rows, layer),
            {uint64_t{2} * rows.rows * layout.kvHeads, 1, 1}, {layout.headDimension, 1, 1});
}

void PagedAttention::addAttention(metal::CommandGraph &graph, const AttentionPlan &plan,
                                  const AttentionStaging &staging, const metal::MetalBuffer &partials,
                                  const metal::MetalBuffer &statistics,
                                  std::span<const kv::ChunkedPrefillParams> sequences,
                                  std::span<const AttentionHistory> histories, metal::MetalBuffer sinks,
                                  metal::MetalBuffer tree) {
  const CommandRows rows = commandRows(sequences, plan.layout);
  const std::span<const AttentionPlan::Sequence> planned = plan.sequences();
  if (planned.size() != sequences.size() || histories.size() != sequences.size())
    throw std::invalid_argument("attention sequences do not match the plan");
  for (size_t s = 0; s < sequences.size(); ++s) {
    if (sequences[s].chunk_tokens != planned[s].rows || sequences[s].committed_tokens != planned[s].historyTokens)
      throw std::invalid_argument("attention sequences do not match the plan");
    requirePageTable(histories[s].pageTable, sequences[s]);
    requireBlocks(histories[s].blocks, plan.sparse, kv::attentionTiles(planned[s].rows));
    // A tree's rows are a block's from its anchor, a bit of a row's
    // ancestors each, and through lists their row masks too.
    if (tree && (planned[s].firstTile || planned[s].rows > 32))
      throw std::invalid_argument("attention tree rows take whole blocks of at most 32 rows");
    if (tree && histories[s].blocks.stride)
      requireBytes(histories[s].blocks.rowMasks, uint64_t{rows.rows} * sizeof(uint32_t), "attention row masks");
  }
  if (tree) requireBytes(tree, uint64_t{rows.rows} * sizeof(SplashTreeRow), "attention tree rows");
  // The tiles read the queries and write the attention rows in whole tiles
  // of rows, which each sequence's stride holds.
  const uint32_t group = plan.queryHeads / plan.layout.kvHeads;
  const uint32_t tileRows = kv::attentionTiles(rows.last.chunk_tokens) * kv::kAttentionTileRows;
  requireStaging(staging, rows, group, plan.layout, tileRows, false, true);
  requireBytes(staging.rows, rows.rowBytes(uint64_t{group} * plan.layout.headDimension, tileRows),
               "attention output");
  const AttentionWorkspace workspace = plan.workspace();
  requireBytes(partials, workspace.partialsBytes, "attention partials");
  requireBytes(statistics, workspace.statisticsBytes, "attention statistics");
  requireSinks(sinks, plan.sinks, plan.queryHeads);
  // A dispatch's sequences read one layer, which SplashAttentionParams::kv
  // places in their pages' extents: where a planned dispatch's sequences read
  // two placements (a spilled history staged apart from the pool), each run of
  // them that reads one is a dispatch of its own, on the planned dispatch's
  // partial slots.
  const auto encode = [&](const AttentionPlan::Dispatch &dispatch, SplashKvLayer layer, uint32_t firstSlot) {
    kv::AttentionParams params{.sequences = dispatch.sequences,
                               .rows = 0,
                               .tiles = dispatch.tiles,
                               .trees = tree ? 1U : 0U,
                               .kv = layer,
                               .sequence = {}};
    std::array<metal::MetalBuffer, kMaximumSequences> tables;
    // The lists the dispatch binds: its listed sequences', which must be one
    // binding; a located dispatch's sequences all read lists.
    AttentionBlocks blocks{};
    for (uint32_t d = 0; d < dispatch.sequences; ++d) {
      const AttentionBlocks &listed = histories[dispatch.sequence[d]].blocks;
      if (!listed.stride) continue;
      if (blocks.stride && !sameBindings(listed, blocks))
        throw std::invalid_argument("an attention dispatch's sequences bind different block lists");
      blocks = listed;
    }
    uint32_t tile = 0, slot = firstSlot;
    for (uint32_t d = 0; d < dispatch.sequences; ++d) {
      const uint32_t s = dispatch.sequence[d];
      const AttentionPlan::Sequence &sequence = planned[s];
      const AttentionHistory &history = histories[s];
      if (blocks.locations && !history.blocks.stride)
        throw std::invalid_argument("a located attention dispatch's sequences all read block lists");
      SplashAttentionSequence &entry = params.sequence[d];
      entry = sequenceRows(sequences[s], rows, s, sequence.firstTile * kv::kAttentionTileRows, params.rows);
      entry.split_count = sequence.splits;
      entry.first_tile = tile;
      entry.first_slot = slot;
      entry.block_stride = history.blocks.stride;
      entry.block_list = history.blocks.stride ? history.blocks.list + sequence.firstTile : 0;
      const uint32_t tiles = kv::attentionTiles(sequence.rows) - sequence.firstTile;
      params.rows += entry.rows;
      tile += tiles;
      slot += tiles * plan.layout.kvHeads * sequence.splits;
      tables[d] = history.pageTable;
    }
    const auto bound = laneBindings(std::span(tables).first(dispatch.sequences));
    std::vector<metal::MetalBuffer> split{staging.queries, partials, statistics};
    split.insert(split.end(), bound.begin(), bound.end());
    // Without tree rows a split binds another buffer of the dispatch for
    // them, of which it reads nothing.
    split.push_back(tree ? tree : partials);
    std::vector<metal::MetalBuffer> reduce{partials, statistics, staging.rows};
    if (plan.sparse) {
      // Without lists, or without tree rows for the row masks, a sparse
      // kernel binds another buffer of the dispatch for them, of which it
      // reads nothing.
      const metal::MetalBuffer &lists = blocks.stride ? blocks.lists : partials;
      const metal::MetalBuffer &counts = blocks.stride ? blocks.counts : partials;
      const metal::MetalBuffer &rowMasks = blocks.stride && tree ? blocks.rowMasks : partials;
      split.insert(split.end(), {lists, counts, rowMasks});
      if (blocks.locations) split.insert(split.end(), {blocks.locations, blocks.locationPages});
      reduce.push_back(counts);
    }
    if (plan.sinks) reduce.push_back(sinks);
    graph.add(std::string(plan.splitPipeline(static_cast<bool>(blocks.locations))), std::move(split), params,
              plan.splitGroups(dispatch));
    graph.add(std::string(plan.reducePipeline()), std::move(reduce), params, plan.reduceGroups(dispatch),
              reduceThreads(plan.layout));
    return slot;
  };
  for (const AttentionPlan::Dispatch &dispatch : plan.dispatches()) {
    uint32_t slot = 0;
    for (uint32_t first = 0, end = 0; first < dispatch.sequences; first = end) {
      const SplashKvLayer layer = histories[dispatch.sequence[first]].kv;
      AttentionPlan::Dispatch run{};
      for (end = first; end < dispatch.sequences && sameLayer(histories[dispatch.sequence[end]].kv, layer); ++end) {
        const AttentionPlan::Sequence &sequence = planned[dispatch.sequence[end]];
        const uint32_t tiles = kv::attentionTiles(sequence.rows) - sequence.firstTile;
        run.sequence[run.sequences++] = dispatch.sequence[end];
        run.tiles += tiles;
        run.splits = std::max(run.splits, sequence.splits);
        run.slots += tiles * sequence.splits;
      }
      slot = encode(run, layer, slot);
    }
  }
}

PreparedInput PagedAttention::addGate(metal::CommandGraph &graph, metal::MetalBuffer packed,
                                      const AttentionStaging &staging, metal::MetalBuffer hidden,
                                      std::span<const kv::ChunkedPrefillParams> sequences, AttentionRows rows,
                                      uint32_t queryHeads, kv::Layout layout, LinearScratch scratch,
                                      LinearInput input) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout);
  const CommandRows command = commandRows(sequences, layout);
  kv::AttentionParams params = everyRow(sequences, command);
  // Each sequence's last row alone, as output row s.
  if (rows == AttentionRows::Last) {
    params.rows = params.tiles = static_cast<uint32_t>(sequences.size());
    for (size_t s = 0; s < sequences.size(); ++s)
      params.sequence[s] = sequenceRows(sequences[s], command, s, sequences[s].chunk_tokens - 1,
                                        static_cast<uint32_t>(s));
  }
  const uint32_t width = queryHeads * layout.headDimension;
  if (kernel.gated) requireBytes(packed, gateBytes(command.rows, kernel), "attention gate");
  requireBytes(staging.rows,
               command.rowBytes(uint64_t{queryHeads / layout.kvHeads} * layout.headDimension,
                                command.last.chunk_tokens),
               "attention row");
  requireBytes(hidden, uint64_t{params.rows} * width * 2, "attention hidden");
  // An ungated geometry's gather reads no packed row.
  const std::string name = kernel.gated ? "gate" : "gather";
  std::vector<metal::MetalBuffer> bindings{staging.rows, hidden};
  if (kernel.gated) bindings.insert(bindings.begin(), std::move(packed));
  if (input != LinearInput::Plain) {
    // A thread writes two elements of a row, each table tile's rows in turn.
    const uint64_t threads = uint64_t{params.rows} * width / 2;
    if (threads % 256) throw std::logic_error("an attention gate's table takes whole threadgroups");
    requireTableScratch(scratch, input, width, params.rows);
    bindings.insert(bindings.end(), {scratch.input, scratch.sums});
    graph.add(kernelName(name + tableSuffix(input), kernel.suffix), std::move(bindings), params,
              {threads / 256, 1, 1}, {256, 1, 1});
    return {std::move(hidden), input};
  }
  graph.add(kernelName(name, kernel.suffix), std::move(bindings), params,
            {gateGroups(params.rows, queryHeads, layout.headDimension), 1, 1});
  return {};
}

void PagedAttention::addBlockLists(metal::CommandGraph &graph, const metal::MetalBuffer &mask, uint32_t maskWords,
                                   std::span<const BlockListRows> sequences, const AttentionBlocks &blocks) {
  // The sparse kernels read a list's rows as an attention tile's, each a bit
  // of an entry.
  constexpr uint32_t tileRows = kv::kAttentionTileRows;
  static_assert(tileRows <= SPLASH_ATTENTION_BLOCK_ROW_BITS, "a block list entry holds a bit per tile row");
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
    const uint32_t sequenceTiles = kv::attentionTiles(rows.rows);
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

void PagedAttention::addPathCommit(metal::CommandGraph &graph, kv::Layout layout, uint32_t extentPages,
                                   std::span<const metal::MetalBuffer> pageTables,
                                   std::span<const uint32_t> committed, uint32_t laneRows,
                                   const metal::MetalBuffer &paths) {
  const uint32_t lanes = static_cast<uint32_t>(committed.size());
  if (!lanes || lanes > kMaximumSequences || pageTables.size() != lanes || !extentPages || !laneRows ||
      laneRows > SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("invalid KV path commit");
  SplashKvPathCommitParams params{.extent_pages = extentPages, .regions = layout.attentionLayers, .committed_tokens = {}};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint64_t pages = (uint64_t{committed[lane]} + laneRows + kv::kPageTokens - 1) / kv::kPageTokens;
    requireBytes(pageTables[lane], pages * sizeof(SplashKvPage), "page table");
    params.committed_tokens[lane] = committed[lane];
  }
  requireBytes(paths, uint64_t{lanes} * sizeof(uint32_t), "KV path commit paths");
  const auto tables = laneBindings(pageTables);
  std::vector<metal::MetalBuffer> bindings(tables.begin(), tables.end());
  bindings.push_back(paths);
  graph.add(kernelName(kernelFormat(layout) + "_commit", storeSuffix(layout)), std::move(bindings), params,
            {uint64_t{layout.attentionLayers} * layout.kvHeads, lanes, 1}, {layout.headDimension, 1, 1});
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

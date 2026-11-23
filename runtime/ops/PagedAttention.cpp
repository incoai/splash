#include "ops/PagedAttention.hpp"

#include "metal/abi/ShortConvolution.h"
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

// The compiled attention geometries (kernels/shared/attention_qkv.metal,
// paged_attention.metal, sparse_attention.metal and
// kv_path_commit.metal): query heads over KV heads of a head dimension and
// how they read the pages, and their kernels' name suffix, the path commits'
// by KV heads, head dimension and the value order of the reader. The sparse
// kernels read selected blocks of token-major values (Qwen3.8-Flash-Next's);
// the dense page loop reads every page of dimension-major ones. Each
// geometry's kernels are compiled for:
// - gated: projection rows that hold each query head's output gate after its
//   queries, by whose sigmoid the reduce multiplies the attention rows it
//   writes (Qwen3.5's); without, it writes them as they are;
// - sinks: a learned sink per query head, which every reduce adds;
// - rotaryPairs: the pairs its RoPE rotates, and unrotated: a projection
//   without RoPE (NoPE) as well, for an AttentionVariant that is not rotated;
// - windowTokens: a split and reduce over a window of that many keys before
//   each row as well as over the whole history, for an AttentionVariant of
//   that window; 0 without;
// - relative: a relative geometry's (metal/abi/PagedAttention.h,
//   RelativeProjection), without RoPE (no rotaryPairs) and gated: a prepare
//   of query side rows and convolved keys and values, unscaled and
//   positionScaled, in place of the q/k norms; splits that bias their scores
//   by distance; and a reduce that adds its rows' own sinks and gates them,
//   both from their query side rows (Inkling 2's).
// Its q/k norm kernels take the norm types its families store (normKernel):
// F32 and bf16, or F32 alone for the geometries of families whose norms load
// as F32 from GGUFs and MLX images alike (Qwen3.8-Flash-Next's, Flint's:
// model/SafetensorsImage.hpp float32Norms).
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
  bool relative;
  std::string_view suffix;
  std::string_view commitSuffix;
};
constexpr std::array kCompiledAttention{
    CompiledAttention{24, 4, 256, kv::Reads::EveryPage, true, false, 32, false, 0, false, "", ""},
    CompiledAttention{16, 2, 256, kv::Reads::EveryPage, true, false, 32, false, 0, false, "_kv2_g8", "_kv2_g8"},
    CompiledAttention{32, 2, 256, kv::Reads::EveryPage, true, false, 32, false, 0, false, "_kv2_g16", "_kv2_g8"},
    CompiledAttention{24, 2, 256, kv::Reads::SelectedBlocks, true, false, 32, false, 0, false, "_kv2_g12",
                      "_kv2_vt"},
    CompiledAttention{32, 4, 128, kv::Reads::EveryPage, false, true, 64, true, 512, false, "_kv4_g8_d128",
                      "_kv4_d128"},
    CompiledAttention{24, 8, 128, kv::Reads::EveryPage, true, false, 0, true, 1023, true, "_kv8_g3_d128",
                      "_kv8_d128"},
};

// Whether a compiled geometry's pages are a layout's: its KV heads, head
// dimension and value order.
bool storesLayout(const CompiledAttention &compiled, kv::Layout layout) {
  return compiled.kvHeads == layout.kvHeads && compiled.headDimension == layout.headDimension &&
         kv::valuesTokenMajorFor(compiled.reads) == layout.valuesTokenMajor;
}

// The KV path commit kernels' suffix of a layout's KV heads, head dimension
// and value order, whose pages a compiled geometry stores.
std::string_view commitSuffix(kv::Layout layout) {
  if (layout.valid())
    for (const CompiledAttention &compiled : kCompiledAttention)
      if (storesLayout(compiled, layout)) return compiled.commitSuffix;
  throw std::invalid_argument("no KV store kernel for layout");
}

// The compiled geometry of `queryHeads` over a layout's KV heads that reads
// its pages as it does, so the kernels need no geometry branches.
const CompiledAttention &compiledAttention(uint32_t queryHeads, kv::Layout layout) {
  static_cast<void>(commitSuffix(layout));
  for (const CompiledAttention &compiled : kCompiledAttention)
    if (compiled.queryHeads == queryHeads && compiled.reads == layout.reads && storesLayout(compiled, layout))
      return compiled;
  throw std::invalid_argument(selectsBlocks(layout) ? "no sparse attention kernel for layout"
                                                    : "no paged-attention kernel for layout");
}

// The compiled geometry, which must compile a layer's variant: its rotation
// (RoPE where the geometry has pairs), its window and its position scale (a
// relative geometry's).
const CompiledAttention &compiledAttention(uint32_t queryHeads, kv::Layout layout, AttentionVariant variant) {
  const CompiledAttention &compiled = compiledAttention(queryHeads, layout);
  if ((!variant.rotated && !compiled.unrotated) || (variant.rotated && !compiled.rotaryPairs) ||
      (variant.windowTokens && variant.windowTokens != compiled.windowTokens) ||
      (variant.positionScaled && !compiled.relative))
    throw std::invalid_argument("no attention kernel for the layer's variant");
  return compiled;
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

// The command's sequences' bounds and staging: each sequence's first staged
// row, the staged KV-head rows of the sequences before it; their bounds'
// sum and the largest bound, which the per-row kernels' grids span; and the
// last sequence's staged rows, which end the staging its kernels reach.
struct CommandRows final {
  std::array<uint32_t, kMaximumSequences> stageRow{};
  uint64_t bounds = 0;
  uint32_t mostRows = 0;
  uint64_t staged = 0;
  kv::ChunkedPrefillParams last{};
  uint32_t kvHeads = 0;

  // The staging's bytes up to the last sequence's last KV head's first
  // `reach` rows, `width` elements each (queries and attention rows).
  [[nodiscard]] uint64_t rowBytes(uint64_t width, uint32_t reach) const noexcept {
    return (staged - kvHeads * uint64_t{last.chunk_stride} + (kvHeads - 1) * uint64_t{last.chunk_stride} + reach) *
           width * sizeof(uint16_t);
  }
};
CommandRows commandRows(std::span<const kv::ChunkedPrefillParams> sequences, kv::Layout layout) {
  if (sequences.empty() || sequences.size() > kMaximumSequences)
    throw std::invalid_argument("invalid attention sequences");
  CommandRows result;
  for (size_t s = 0; s < sequences.size(); ++s) {
    const std::string_view error = kv::chunkedPrefillValidationError(sequences[s]);
    if (!error.empty()) throw std::invalid_argument(std::string(error));
    result.stageRow[s] = static_cast<uint32_t>(result.staged);
    result.bounds += sequences[s].chunk_tokens;
    result.mostRows = std::max(result.mostRows, sequences[s].chunk_tokens);
    result.staged += uint64_t{layout.kvHeads} * sequences[s].chunk_stride;
  }
  result.last = sequences.back();
  result.kvHeads = layout.kvHeads;
  if (result.staged > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("attention staging exceeds the kernel ABI");
  return result;
}

// Sequence s's rows from row `first` of its layout entry on.
SplashAttentionSequence sequenceRows(const kv::ChunkedPrefillParams &chunk, const CommandRows &rows, size_t s,
                                     uint32_t first) {
  return {.committed_tokens = chunk.committed_tokens + first,
          .row_bound = chunk.chunk_tokens - first,
          .layout_row = first,
          .stage_row = rows.stageRow[s] + first,
          .chunk_stride = chunk.chunk_stride,
          .page_table_entries = chunk.page_table_entries,
          .split_count = 0,
          .first_tile = 0,
          .first_slot = 0,
          .block_stride = 0,
          .block_list = 0};
}

// A command's rows (AttentionRows) of its sequences: at least a row, at most
// their bounds' sum, each sequence's layout entry, and a tree row a row.
void requireRows(const AttentionRows &rows, const CommandRows &command, size_t sequences) {
  if (!rows.rows || rows.rows > command.bounds)
    throw std::invalid_argument("attention rows exceed their sequences' bounds");
  requireBytes(rows.layout, sequences * sizeof(SplashLaneRows), "attention layout");
  if (rows.tree) requireBytes(rows.tree, uint64_t{rows.rows} * sizeof(SplashTreeRow), "attention tree rows");
}

// The parameters of every row of the command's sequences, in layer `layer`.
kv::AttentionParams everyRow(std::span<const kv::ChunkedPrefillParams> sequences, const CommandRows &command,
                             const AttentionRows &rows, SplashKvLayer layer = {}) {
  kv::AttentionParams params{.sequences = static_cast<uint32_t>(sequences.size()),
                             .first_sequence = 0,
                             .rows = rows.rows,
                             .tiles = 0,
                             .trees = rows.tree ? 1U : 0U,
                             .last_rows = 0,
                             .kv = layer,
                             .sequence = {}};
  for (size_t s = 0; s < sequences.size(); ++s) {
    params.sequence[s] = sequenceRows(sequences[s], command, s, 0);
    params.tiles += kv::attentionTiles(sequences[s].chunk_tokens);
  }
  return params;
}

// The grid of a per-row kernel: `groups` groups of each row of every
// sequence, up to the largest bound.
metal::DispatchSize rowGroups(uint64_t groups, const CommandRows &command, size_t sequences) {
  return {groups, command.mostRows, sequences};
}

// The staged queries of a command's rows its kernels reach, of `reach` rows
// of the last sequence's: a KV head's group of query heads a row.
void requireQueries(const metal::MetalBuffer &queries, const CommandRows &rows, uint32_t group, kv::Layout layout,
                    uint32_t reach) {
  requireBytes(queries, rows.rowBytes(uint64_t{group} * layout.headDimension, reach), "attention query");
}

// A packed projection row: every query head's q and, gated, its gate, then
// the K and V heads. The gates read each row's query heads only; a relative
// geometry's, every row's query side row up to its sinks.
uint64_t packedWidth(const CompiledAttention &compiled) {
  return (uint64_t{compiled.queryHeads} * (compiled.gated ? 2 : 1) + uint64_t{compiled.kvHeads} * 2) *
         compiled.headDimension;
}
uint64_t gateBytes(uint64_t rows, const CompiledAttention &compiled) {
  if (compiled.relative)
    return rowBytes(rows, relativeQuerySideWidth(compiled.queryHeads, compiled.headDimension),
                    SPLASH_RELATIVE_SINKS(compiled.queryHeads, compiled.headDimension) + compiled.queryHeads,
                    sizeof(float));
  return rowBytes(rows, packedWidth(compiled), uint64_t{compiled.queryHeads} * 2 * compiled.headDimension, 2);
}

// A relative geometry's rows' relative states, of every query head.
uint64_t relativeStateBytes(uint64_t rows, uint32_t queryHeads) {
  return rows * queryHeads * kRelativeWidth * sizeof(float);
}

// The q/k projection's bindings: the packed rows and norms, a rotated
// variant's RoPE tables of `rows` rows, then the staging of the queries and
// the sequences' page tables, which take the keys and values.
std::vector<metal::MetalBuffer> projectionBindings(const CompiledAttention &compiled, AttentionVariant variant,
                                                   uint64_t rows, metal::MetalBuffer packed,
                                                   const NormWeights &queryNorm, const NormWeights &keyNorm,
                                                   metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                                                   metal::MetalBuffer queries,
                                                   std::span<const metal::MetalBuffer> pageTables) {
  std::vector<metal::MetalBuffer> bindings{std::move(packed), queryNorm.buffer, keyNorm.buffer};
  if (variant.rotated) {
    const uint64_t bytes = rows * compiled.rotaryPairs * sizeof(float);
    requireBytes(ropeCos, bytes, "attention RoPE cosine");
    requireBytes(ropeSin, bytes, "attention RoPE sine");
    bindings.insert(bindings.end(), {std::move(ropeCos), std::move(ropeSin)});
  }
  bindings.push_back(std::move(queries));
  const auto tables = laneBindings(pageTables);
  bindings.insert(bindings.end(), tables.begin(), tables.end());
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
  AttentionPlan plan(layout, queryHeads, sparse, kernel.sinks, kernel.relative,
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
    const uint32_t splits = attentionSplits(tiles, pages, variant.windowTokens, calibration.pagesPerSplit);
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

bool PagedAttention::compiles(uint32_t queryHeads, kv::Layout layout, bool gated, uint32_t rotaryPairs,
                              bool relative) {
  const CompiledAttention &compiled = compiledAttention(queryHeads, layout);
  return compiled.gated == gated && compiled.rotaryPairs == rotaryPairs && compiled.relative == relative;
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
                                   metal::MetalBuffer queries, SplashKvLayer layer,
                                   std::span<const metal::MetalBuffer> pageTables,
                                   std::span<const kv::ChunkedPrefillParams> sequences, const AttentionRows &rows,
                                   uint32_t queryHeads, kv::Layout layout, AttentionVariant variant) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout, variant);
  if (kernel.relative) throw std::invalid_argument("a relative geometry's rows take its relative projection");
  const CommandRows command = commandRows(sequences, layout);
  if (pageTables.size() != sequences.size())
    throw std::invalid_argument("a q/k projection takes a page table per sequence");
  requireRows(rows, command, sequences.size());
  requireBytes(packed, uint64_t{rows.rows} * packedWidth(kernel) * 2, "attention q/k/v");
  requireQueries(queries, command, queryHeads / layout.kvHeads, layout, command.last.chunk_tokens);
  for (size_t s = 0; s < sequences.size(); ++s) requirePageTable(pageTables[s], sequences[s]);
  std::vector<metal::MetalBuffer> bindings = projectionBindings(
      kernel, variant, rows.rows, std::move(packed), queryNorm, keyNorm, std::move(ropeCos), std::move(ropeSin),
      std::move(queries), pageTables);
  bindings.push_back(rows.layout);
  graph.add(qkNormKernel(kernelName(kernelFormat(layout) + "_qkv",
                                    std::string(kernel.suffix) + (variant.rotated ? "" : "_nope")),
                         queryNorm, keyNorm, layout.headDimension),
            std::move(bindings), everyRow(sequences, command, rows, layer),
            rowGroups(queryHeads + layout.kvHeads, command, sequences.size()), {layout.headDimension, 1, 1});
}

void PagedAttention::addRelativeProjection(metal::CommandGraph &graph, const RelativeProjection &inputs,
                                           metal::MetalBuffer queries, metal::MetalBuffer states,
                                           SplashKvLayer layer, std::span<const metal::MetalBuffer> pageTables,
                                           std::span<const kv::ChunkedPrefillParams> sequences,
                                           const AttentionRows &rows, uint32_t queryHeads, kv::Layout layout,
                                           AttentionVariant variant, const RelativeConvolution *convolution) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout, variant);
  if (!kernel.relative || variant.rotated)
    throw std::invalid_argument("a relative projection takes an unrotated relative geometry");
  if (convolution) {
    // A verify step that convolves its keys and values itself: the sides,
    // taps, a history a lane and the tape in place of the keys and values,
    // and the convolution's parameters after the attention's.
    const RelativeConvolution &conv = *convolution;
    const CommandRows command = commandRows(sequences, layout);
    if (!rows.tree || sequences.size() > SPLASH_MAXIMUM_BATCH_WIDTH || pageTables.size() != sequences.size() ||
        conv.histories.size() != sequences.size())
      throw std::invalid_argument("a convolving prepare takes a verify step's lanes, a page table and a history each");
    requireRows(rows, command, sequences.size());
    const uint64_t keyWidth = uint64_t{layout.kvHeads} * layout.headDimension;
    requireBytes(inputs.querySide,
                 uint64_t{rows.rows} * relativeQuerySideWidth(queryHeads, layout.headDimension) * sizeof(float),
                 "attention query side");
    const uint64_t sideRow = uint64_t{conv.sideStride} * sizeof(float);
    const uint64_t taps = uint64_t{SHORT_CONVOLUTION_TAPS};
    if (conv.sideStride < keyWidth ||
        (conv.headChannels && conv.lambdaOffset + keyWidth / conv.headChannels * taps > conv.sideStride))
      throw std::invalid_argument("a convolving prepare's side rows do not hold its inputs and taps");
    requireBytes(conv.keySide, uint64_t{rows.rows - 1} * sideRow + keyWidth * sizeof(float), "attention key side");
    requireBytes(conv.valueSide, uint64_t{rows.rows - 1} * sideRow + keyWidth * sizeof(float),
                 "attention value side");
    requireBytes(conv.keyTaps, keyWidth * taps * sizeof(float), "key convolution taps");
    requireBytes(conv.valueTaps, keyWidth * taps * sizeof(float), "value convolution taps");
    const uint64_t historyBytes =
        (uint64_t{SHORT_CONVOLUTION_HISTORY - 1} * conv.historyStride + conv.valueOffset + keyWidth) * 2;
    for (const metal::MetalBuffer &history : conv.histories) requireBytes(history, historyBytes, "key and value history");
    requireBytes(conv.tape,
                 (uint64_t{rows.rows - 1} * conv.tapeStride + conv.valueOffset + keyWidth) * 2,
                 "key and value tape");
    requireBytes(inputs.queryScale, uint64_t{queryHeads} * layout.headDimension * sizeof(float),
                 "attention query scale");
    requireBytes(inputs.relativeScale, uint64_t{queryHeads} * kRelativeWidth * sizeof(float),
                 "attention relative scale");
    requireBytes(states, relativeStateBytes(rows.rows, queryHeads), "attention relative states");
    requireQueries(queries, command, queryHeads / layout.kvHeads, layout, command.last.chunk_tokens);
    for (size_t s = 0; s < sequences.size(); ++s) requirePageTable(pageTables[s], sequences[s]);
    std::vector<metal::MetalBuffer> bindings{inputs.querySide,    conv.keySide, conv.valueSide,
                                             inputs.queryScale,   inputs.relativeScale, std::move(queries),
                                             std::move(states)};
    // A lane past the step's binds lane 0's page table and history, which it never reads.
    for (size_t s = 0; s < SPLASH_MAXIMUM_BATCH_WIDTH; ++s)
      bindings.push_back(pageTables[s < pageTables.size() ? s : 0]);
    bindings.insert(bindings.end(), {rows.tree, rows.layout, conv.keyTaps, conv.valueTaps});
    for (size_t s = 0; s < SPLASH_MAXIMUM_BATCH_WIDTH; ++s)
      bindings.push_back(conv.histories[s < conv.histories.size() ? s : 0]);
    bindings.push_back(conv.tape);
    const SplashRelativeConvolutionParams params{
        everyRow(sequences, command, rows, layer),
        {conv.sideStride, conv.lambdaOffset, conv.headChannels, conv.historyStride,
         conv.tapeStride, conv.valueOffset, conv.scale, 0}};
    graph.add(kernelName(kernelFormat(layout) + "_qkv_conv", std::string(kernel.suffix) +
                                                                 (variant.positionScaled ? "_scaled" : "")),
              std::move(bindings), params,
              rowGroups(queryHeads + layout.kvHeads, command, sequences.size()), {layout.headDimension, 1, 1});
    return;
  }
  const CommandRows command = commandRows(sequences, layout);
  if (pageTables.size() != sequences.size())
    throw std::invalid_argument("a q/k projection takes a page table per sequence");
  requireRows(rows, command, sequences.size());
  const uint64_t keyWidth = uint64_t{layout.kvHeads} * layout.headDimension;
  requireBytes(inputs.querySide,
               uint64_t{rows.rows} * relativeQuerySideWidth(queryHeads, layout.headDimension) * sizeof(float),
               "attention query side");
  requireBytes(inputs.keys, uint64_t{rows.rows} * keyWidth * sizeof(float), "attention keys");
  requireBytes(inputs.values, uint64_t{rows.rows} * keyWidth * sizeof(float), "attention values");
  requireBytes(inputs.queryScale, uint64_t{queryHeads} * layout.headDimension * sizeof(float),
               "attention query scale");
  requireBytes(inputs.relativeScale, uint64_t{queryHeads} * kRelativeWidth * sizeof(float),
               "attention relative scale");
  requireBytes(states, relativeStateBytes(rows.rows, queryHeads), "attention relative states");
  requireQueries(queries, command, queryHeads / layout.kvHeads, layout, command.last.chunk_tokens);
  for (size_t s = 0; s < sequences.size(); ++s) requirePageTable(pageTables[s], sequences[s]);
  std::vector<metal::MetalBuffer> bindings{inputs.querySide,    inputs.keys,        inputs.values,    inputs.queryScale,
                                           inputs.relativeScale, std::move(queries), std::move(states)};
  const auto tables = laneBindings(pageTables);
  bindings.insert(bindings.end(), tables.begin(), tables.end());
  // Without tree rows the prepare binds another buffer of the dispatch for
  // them, of which it reads nothing.
  bindings.insert(bindings.end(), {rows.tree ? rows.tree : inputs.querySide, rows.layout});
  graph.add(kernelName(kernelFormat(layout) + "_qkv", std::string(kernel.suffix) +
                                                          (variant.positionScaled ? "_scaled" : "")),
            std::move(bindings), everyRow(sequences, command, rows, layer),
            rowGroups(queryHeads + layout.kvHeads, command, sequences.size()), {layout.headDimension, 1, 1});
}

PreparedInput PagedAttention::addAttention(metal::CommandGraph &graph, const AttentionPlan &plan,
                                           metal::MetalBuffer queries, const metal::MetalBuffer &partials,
                                           const metal::MetalBuffer &statistics,
                                           std::span<const kv::ChunkedPrefillParams> sequences,
                                           std::span<const AttentionHistory> histories, const AttentionRows &rows,
                                           const AttentionOutputRows &output, metal::MetalBuffer sinks,
                                           const RelativeBias &relative) {
  const CommandRows command = commandRows(sequences, plan.layout);
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
    if (rows.tree && (planned[s].firstTile || planned[s].rows > 32))
      throw std::invalid_argument("attention tree rows take whole blocks of at most 32 rows");
    if (rows.tree && histories[s].blocks.stride)
      requireBytes(histories[s].blocks.rowMasks, uint64_t{rows.rows} * sizeof(uint32_t), "attention row masks");
  }
  requireRows(rows, command, sequences.size());
  // The tiles read the queries in whole tiles of rows, which each sequence's
  // stride holds.
  const uint32_t group = plan.queryHeads / plan.layout.kvHeads;
  const uint32_t tileRows = kv::attentionTiles(command.last.chunk_tokens) * kv::kAttentionTileRows;
  requireQueries(queries, command, group, plan.layout, tileRows);
  // The reduce writes every row as its packed row, or each sequence's last
  // as row s of a plan of its last tiles, gating it by its packed row's gate
  // where the geometry has one, and the out-projection's table of the rows
  // where the input takes one; without, another buffer of the dispatch takes
  // the packed rows', the table's and the sums' bindings, of which it reads
  // nothing.
  const bool last = output.output == AttentionOutput::Last, table = output.input != LinearInput::Plain;
  for (const AttentionPlan::Sequence &sequence : planned)
    if (last && sequence.firstTile + 1 != kv::attentionTiles(sequence.rows))
      throw std::invalid_argument("an attention of each sequence's last row takes the plan of its last tiles");
  const CompiledAttention &kernel = compiledAttention(plan.queryHeads, plan.layout);
  const uint32_t width = plan.queryHeads * plan.layout.headDimension;
  const uint32_t outputRows = last ? static_cast<uint32_t>(sequences.size()) : rows.rows;
  if (kernel.gated) requireBytes(output.packed, gateBytes(rows.rows, kernel), "attention gate");
  requireBytes(output.hidden, uint64_t{outputRows} * width * 2, "attention hidden");
  if (table) requireTableScratch(output.scratch, output.input, width, outputRows);
  const metal::MetalBuffer &gates = kernel.gated ? output.packed : output.hidden;
  const metal::MetalBuffer &tableInput = table ? output.scratch.input : output.hidden;
  const metal::MetalBuffer &tableSums = table ? output.scratch.sums : output.hidden;
  const std::string reducePipeline = std::string(plan.reducePipeline()) + tableSuffix(output.input);
  const AttentionWorkspace workspace = plan.workspace();
  requireBytes(partials, workspace.partialsBytes, "attention partials");
  requireBytes(statistics, workspace.statisticsBytes, "attention statistics");
  requireSinks(sinks, plan.sinks, plan.queryHeads);
  if (plan.relative) {
    requireBytes(relative.states, relativeStateBytes(rows.rows, plan.queryHeads), "attention relative states");
    requireBytes(relative.table, uint64_t{kRelativeExtent} * kRelativeWidth * sizeof(float),
                 "attention relative table");
  } else if (relative.states || relative.table) {
    throw std::invalid_argument("the attention kernels take no relative bias");
  }
  // A dispatch's sequences read one layer, which SplashAttentionParams::kv
  // places in their pages' extents: where a planned dispatch's sequences read
  // two placements (a spilled history staged apart from the pool), each run of
  // them that reads one is a dispatch of its own, on the planned dispatch's
  // partial slots.
  // A dispatch's sequences are the plan's from its first on (group).
  const auto encode = [&](const AttentionPlan::Dispatch &dispatch, SplashKvLayer layer, uint32_t firstSlot) {
    kv::AttentionParams params{.sequences = dispatch.sequences,
                               .first_sequence = dispatch.sequence[0],
                               .rows = rows.rows,
                               .tiles = dispatch.tiles,
                               .trees = rows.tree ? 1U : 0U,
                               .last_rows = last ? 1U : 0U,
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
      if (s != dispatch.sequence[0] + d) throw std::logic_error("an attention dispatch's sequences are not in order");
      const AttentionPlan::Sequence &sequence = planned[s];
      const AttentionHistory &history = histories[s];
      if (blocks.locations && !history.blocks.stride)
        throw std::invalid_argument("a located attention dispatch's sequences all read block lists");
      SplashAttentionSequence &entry = params.sequence[d];
      entry = sequenceRows(sequences[s], command, s, sequence.firstTile * kv::kAttentionTileRows);
      entry.split_count = sequence.splits;
      entry.first_tile = tile;
      entry.first_slot = slot;
      entry.block_stride = history.blocks.stride;
      entry.block_list = history.blocks.stride ? history.blocks.list + sequence.firstTile : 0;
      const uint32_t tiles = kv::attentionTiles(sequence.rows) - sequence.firstTile;
      tile += tiles;
      slot += tiles * plan.layout.kvHeads * sequence.splits;
      tables[d] = history.pageTable;
    }
    const auto bound = laneBindings(std::span(tables).first(dispatch.sequences));
    std::vector<metal::MetalBuffer> split{queries, partials, statistics};
    split.insert(split.end(), bound.begin(), bound.end());
    // Without tree rows a split binds another buffer of the dispatch for
    // them, of which it reads nothing.
    split.insert(split.end(), {rows.tree ? rows.tree : partials, rows.layout});
    if (plan.relative) split.insert(split.end(), {relative.states, relative.table});
    std::vector<metal::MetalBuffer> reduce{partials, statistics, gates, output.hidden, tableInput, tableSums,
                                           rows.layout};
    if (plan.sparse) {
      // Without lists, or without tree rows for the row masks, a sparse
      // kernel binds another buffer of the dispatch for them, of which it
      // reads nothing.
      const metal::MetalBuffer &lists = blocks.stride ? blocks.lists : partials;
      const metal::MetalBuffer &counts = blocks.stride ? blocks.counts : partials;
      const metal::MetalBuffer &rowMasks = blocks.stride && rows.tree ? blocks.rowMasks : partials;
      split.insert(split.end(), {lists, counts, rowMasks});
      if (blocks.locations) split.insert(split.end(), {blocks.locations, blocks.locationPages});
      reduce.push_back(counts);
    }
    if (plan.sinks) reduce.push_back(sinks);
    graph.add(std::string(plan.splitPipeline(static_cast<bool>(blocks.locations))), std::move(split), params,
              plan.splitGroups(dispatch));
    graph.add(reducePipeline, std::move(reduce), params, plan.reduceGroups(dispatch), reduceThreads(plan.layout));
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
  return table ? PreparedInput{output.hidden, output.input} : PreparedInput{};
}

void PagedAttention::addBlockLists(metal::CommandGraph &graph, const metal::MetalBuffer &mask, uint32_t maskWords,
                                   std::span<const BlockListRows> sequences, const AttentionRows &rows,
                                   const AttentionBlocks &blocks) {
  // The sparse kernels read a list's rows as an attention tile's, each a bit
  // of an entry.
  constexpr uint32_t tileRows = kv::kAttentionTileRows;
  static_assert(tileRows <= SPLASH_ATTENTION_BLOCK_ROW_BITS, "a block list entry holds a bit per tile row");
  if (sequences.empty() || sequences.size() > SPLASH_MAXIMUM_BATCH_WIDTH || !maskWords || !blocks.stride ||
      !rows.rows)
    throw std::invalid_argument("invalid attention block lists");
  SplashAttentionBlockListParams params{
      .tile_rows = tileRows, .mask_words = maskWords, .stride = blocks.stride,
      .sequences = static_cast<uint32_t>(sequences.size()), .rows = rows.rows, .sequence = {}};
  uint64_t lists = 0;
  uint32_t tiles = 0, entries = 0;
  for (size_t index = 0; index < sequences.size(); ++index) {
    const BlockListRows &sequence = sequences[index];
    if (!sequence.rows) throw std::invalid_argument("invalid attention block lists");
    const uint64_t historyBlocks =
        (uint64_t{sequence.firstPosition} + sequence.rows + SPLASH_ATTENTION_BLOCK_TOKENS - 1) /
        SPLASH_ATTENTION_BLOCK_TOKENS;
    if (uint64_t{maskWords} * 32 < historyBlocks)
      throw std::invalid_argument("attention block masks narrower than their rows' history");
    const uint32_t sequenceTiles = kv::attentionTiles(sequence.rows);
    lists = std::max(lists, uint64_t{blocks.list} + sequence.list + sequenceTiles);
    tiles = std::max(tiles, sequenceTiles);
    entries = std::max(entries, sequence.sequence + 1);
    params.sequence[index] = {.sequence = sequence.sequence, .row_bound = sequence.rows,
                              .first_position = sequence.firstPosition, .list = blocks.list + sequence.list};
  }
  requireBytes(rows.layout, uint64_t{entries} * sizeof(SplashLaneRows), "attention layout");
  requireBytes(mask, uint64_t{rows.rows} * maskWords * sizeof(uint32_t), "attention block mask");
  requireBytes(blocks.lists, lists * blocks.stride * sizeof(uint32_t), "attention block lists");
  requireBytes(blocks.counts, lists * sizeof(uint32_t), "attention block list counts");
  graph.add("attention_block_lists", {mask, blocks.lists, blocks.counts, rows.layout}, params,
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
  graph.add(kernelName(kernelFormat(layout) + "_commit", commitSuffix(layout)), std::move(bindings), params,
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

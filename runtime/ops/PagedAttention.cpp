#include "ops/PagedAttention.hpp"

#include "ops/AttentionPlanning.hpp"
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
//   that window; 0 without.
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
  std::string_view suffix;
  std::string_view commitSuffix;
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

// The KV path commit kernels' suffix of a layout's KV heads, head dimension
// and value order, whose pages a compiled geometry stores.
std::string_view commitSuffix(kv::Layout layout) {
  if (layout.valid())
    for (const CompiledAttention &compiled : kCompiledAttention)
      if (storesLayout(compiled, layout)) return compiled.commitSuffix;
  throw std::invalid_argument("no KV store kernel for layout");
}

// The compiled geometry of `queryHeads` over a layout's KV heads that reads
// its pages as it does, so the kernels need no geometry branches, which must
// compile a layer's variant.
const CompiledAttention &compiledAttention(uint32_t queryHeads, kv::Layout layout,
                                           AttentionVariant variant = {}) {
  static_cast<void>(commitSuffix(layout));
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

// A layout's KV path commit kernel.
std::string commitKernel(kv::Layout layout) { return kernelName(kernelFormat(layout) + "_commit", commitSuffix(layout)); }

// One dispatch normalizes both the queries and the keys, so their norms must
// share a stored type.
std::string qkNormKernel(std::string_view name, const NormWeights &queryNorm,
                         const NormWeights &keyNorm, uint32_t headDimension) {
  std::string kernel = normKernel(name, queryNorm, headDimension);
  if (normKernel(name, keyNorm, headDimension) != kernel)
    throw std::invalid_argument("query and key norms differ in type");
  return kernel;
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
// the K and V heads. The gates read each row's query heads only.
uint64_t packedWidth(const CompiledAttention &compiled) {
  return (uint64_t{compiled.queryHeads} * (compiled.gated ? 2 : 1) + uint64_t{compiled.kvHeads} * 2) *
         compiled.headDimension;
}
uint64_t gateBytes(uint64_t rows, const CompiledAttention &compiled) {
  return rowBytes(rows, packedWidth(compiled), uint64_t{compiled.queryHeads} * 2 * compiled.headDimension, 2);
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
  plan.dispatchCount_ = groupDispatches(std::span(plan.sequences_).first(plan.sequenceCount_), plan.workspaceSlots,
                                        plan.dispatches_);
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
                                   metal::MetalBuffer queries, SplashKvLayer layer,
                                   std::span<const metal::MetalBuffer> pageTables,
                                   std::span<const kv::ChunkedPrefillParams> sequences, const AttentionRows &rows,
                                   uint32_t queryHeads, kv::Layout layout, AttentionVariant variant) {
  const CompiledAttention &kernel = compiledAttention(queryHeads, layout, variant);
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

PreparedInput PagedAttention::addAttention(metal::CommandGraph &graph, const AttentionPlan &plan,
                                           metal::MetalBuffer queries, const metal::MetalBuffer &partials,
                                           const metal::MetalBuffer &statistics,
                                           std::span<const kv::ChunkedPrefillParams> sequences,
                                           std::span<const AttentionHistory> histories, const AttentionRows &rows,
                                           const AttentionOutputRows &output, metal::MetalBuffer sinks) {
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
  addKvPathCommit(graph, commitKernel, layout, extentPages, pageTables, committed, laneRows, paths);
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

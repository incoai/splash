#include "ops/RelativeAttention.hpp"

#include "ops/AttentionPlanning.hpp"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {

namespace {

constexpr uint32_t kMaximumSequences = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kGroup = RelativeAttention::kQueryHeads / RelativeAttention::kKvHeads;
constexpr uint32_t kStates = SPLASH_RELATIVE_ATTENTION_STATES;
static_assert(SPLASH_ATTENTION_TILE_PARTS(kGroup) == 1, "a group of three runs whole tiles");
static_assert(SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS == 2 * kv::kAttentionTileRows &&
                  SPLASH_TARGET_KV_BLOCK_TOKENS % SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS == 0,
              "a direct tile is two of the paged attention's, whole within a staged stride");

void requireLayout(kv::Layout layout) {
  if (!RelativeAttention::stores(layout)) throw std::invalid_argument("no relative attention kernel for layout");
}

void requireVariant(RelativeAttentionVariant variant) {
  if (variant.windowTokens && variant.windowTokens != RelativeAttention::kWindowTokens)
    throw std::invalid_argument("no relative attention kernel for the layer's window");
}

// A layout's KV format, as the kernels' names hold it.
std::string kernelFormat(kv::Layout layout) { return layout.format == kv::Format::BFloat16 ? "bf16" : "q8"; }

// The workspace of `rows` fused rows' partials and statistics, and the fused
// rows of one tile split of every KV head (PagedAttention.cpp's
// attentionWorkspace and slotRows).
AttentionWorkspace workspaceOf(uint64_t rows) {
  return {rows * RelativeAttention::kHeadDimension * sizeof(float), rows * 2 * sizeof(float)};
}
uint64_t slotRows() { return uint64_t{kv::kAttentionTileRows} * RelativeAttention::kQueryHeads; }

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

void requireRows(const AttentionRows &rows, const CommandRows &command, size_t sequences) {
  if (!rows.rows || rows.rows > command.bounds)
    throw std::invalid_argument("attention rows exceed their sequences' bounds");
  requireBytes(rows.layout, sequences * sizeof(SplashLaneRows), "attention layout");
  if (rows.tree) requireBytes(rows.tree, uint64_t{rows.rows} * sizeof(SplashTreeRow), "attention tree rows");
}

void requirePageTable(const metal::MetalBuffer &table, const kv::ChunkedPrefillParams &chunk) {
  requireBytes(table, uint64_t{kv::chunkedPrefillRequiredPages(chunk)} * sizeof(SplashKvPage), "page table");
}

// The staging a command's kernels reach, of `reach` rows of the last
// sequence's: bf16 queries, and F32 states and sinks, whose bytes are those
// of twice as many bf16 elements (CommandRows::rowBytes counts bf16).
void requireStaging(const RelativeAttentionStaging &staging, const CommandRows &command, uint32_t reach) {
  constexpr uint64_t halves = sizeof(float) / sizeof(uint16_t);
  requireBytes(staging.queries, command.rowBytes(uint64_t{kGroup} * RelativeAttention::kHeadDimension, reach),
               "attention query");
  requireBytes(staging.states, command.rowBytes(uint64_t{kGroup} * kStates * halves, reach),
               "attention relative states");
  requireBytes(staging.sinks, command.rowBytes(kGroup * halves, reach), "attention sinks");
}

bool sameLayer(SplashKvLayer a, SplashKvLayer b) { return a.extent_pages == b.extent_pages && a.offset == b.offset; }

} // namespace

AttentionWorkspace RelativeAttentionPlan::workspace() const noexcept {
  uint64_t slots = 0;
  for (const Dispatch &dispatch : dispatches()) slots = std::max<uint64_t>(slots, dispatch.slots);
  return workspaceOf(slots * slotRows());
}

metal::DispatchSize RelativeAttentionPlan::splitGroups(const Dispatch &dispatch) const noexcept {
  return {layout.kvHeads, step == StepKind::Decode ? dispatch.sequences : dispatch.tiles, dispatch.splits};
}

metal::DispatchSize RelativeAttentionPlan::reduceGroups(const Dispatch &dispatch) const noexcept {
  return {layout.kvHeads, kv::kAttentionTileRows * kGroup, dispatch.tiles};
}

uint32_t RelativeAttentionPlan::directTiles(const Sequence &sequence) noexcept {
  return (sequence.rows - sequence.firstTile * kv::kAttentionTileRows + SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS - 1) /
         SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS;
}

metal::DispatchSize RelativeAttentionPlan::directGroups(const Dispatch &dispatch) const noexcept {
  uint64_t tiles = 0;
  for (uint32_t d = 0; d < dispatch.sequences; ++d) tiles += directTiles(sequences_[dispatch.sequence[d]]);
  return {layout.kvHeads, tiles, 1};
}

bool RelativeAttention::stores(kv::Layout layout) noexcept {
  return layout.valid() && layout.kvHeads == kKvHeads && layout.headDimension == kHeadDimension &&
         layout.reads == kv::Reads::EveryPage && !layout.valuesTokenMajor && !layout.indexDimension;
}

RelativeAttentionStagingBytes RelativeAttention::stagingBytes(std::span<const kv::ChunkedPrefillParams> sequences,
                                                              kv::Layout layout) {
  requireLayout(layout);
  const uint64_t fused = commandRows(sequences, layout).staged * kGroup;
  return {fused * kHeadDimension * 2, fused * kStates * sizeof(float), fused * sizeof(float)};
}

AttentionWorkspace RelativeAttention::workspace(uint32_t slots, kv::Layout layout) {
  requireLayout(layout);
  if (!slots) throw std::invalid_argument("invalid attention workspace slots");
  return workspaceOf(slots * slotRows());
}

RelativeAttentionPlan RelativeAttention::plan(std::span<const kv::ChunkedPrefillParams> sequences,
                                              kv::Layout layout, const AttentionCalibration &calibration,
                                              Int8Activations int8Activations, AttentionWorkspace capacity,
                                              RelativeAttentionVariant variant, StepKind step) {
  requireLayout(layout);
  requireVariant(variant);
  static_cast<void>(commandRows(sequences, layout));
  const bool decode = step == StepKind::Decode;
  // A decode step's lanes share the step's lane cap, at most a verify
  // block's rows: one tile of the single pass each.
  if (decode)
    for (const kv::ChunkedPrefillParams &chunk : sequences)
      if (chunk.chunk_tokens != sequences.front().chunk_tokens || chunk.chunk_tokens > SPLASH_TARGET_VERIFY_ROWS)
        throw std::invalid_argument("a decode step's lanes share a lane cap of at most a verify block's rows");
  if (decode && !calibration.cores)
    throw std::invalid_argument("a decode step's splits fill a device's cores: a calibration of none");
  const AttentionWorkspace slot = workspaceOf(slotRows());
  const uint64_t slots = std::min(capacity.partialsBytes / slot.partialsBytes,
                                  capacity.statisticsBytes / slot.statisticsBytes);
  RelativeAttentionPlan plan(layout, step,
                             static_cast<uint32_t>(std::min<uint64_t>(slots, std::numeric_limits<uint32_t>::max())));
  uint64_t splitReads = 0;
  for (const kv::ChunkedPrefillParams &chunk : sequences) {
    const uint32_t tiles = kv::attentionTiles(chunk.chunk_tokens);
    const uint64_t pages =
        readPages(uint64_t{chunk.committed_tokens} + chunk.chunk_tokens, variant.windowTokens);
    const uint32_t splits = decode ? decodeSplits(pages, chunk.chunk_tokens, calibration.cores)
                                   : attentionSplits(tiles, pages, variant.windowTokens, calibration.pagesPerSplit);
    splitReads = std::max(splitReads, (pages + splits - 1) / splits);
    plan.sequences_[plan.sequenceCount_++] = {chunk.chunk_tokens, chunk.committed_tokens, 0, splits};
  }
  const std::string window = variant.windowTokens ? "_w" + std::to_string(variant.windowTokens) : "";
  const bool int8Queries = int8Activations == Int8Activations::On && layout.format == kv::Format::Int8 &&
                           splitReads >= calibration.int8QueriesFromPages;
  const std::string format = int8Queries ? std::string("q8q") : kernelFormat(layout);
  if (decode) {
    plan.decodePipeline_ = "relative_attention_" + format + "_decode" +
                           (sequences.front().chunk_tokens > kv::kAttentionTileRows ? "_r16" : "") + window +
                           (variant.positionScaled ? "_scaled" : "");
  } else {
    plan.splitPipeline_ = "relative_attention_" + format + "_split" + window;
    plan.reducePipeline_ = "relative_attention_reduce" + window;
    plan.directPipeline_ = "relative_attention_" + format + "_direct" + window;
  }
  group(plan);
  return plan;
}

uint32_t RelativeAttention::decodeMostSplits(uint32_t laneCap, uint32_t cores) {
  const uint32_t most = laneCap > kv::kAttentionTileRows ? (cores + 8) / 4 : (cores + 16) / 3;
  return std::clamp<uint32_t>(most, 1, SPLASH_RELATIVE_ATTENTION_DECODE_SPLITS);
}

uint32_t RelativeAttention::decodeSplits(uint64_t pages, uint32_t laneCap, uint32_t cores) {
  const uint64_t most = decodeMostSplits(laneCap, cores);
  const uint64_t fewest = std::clamp<uint64_t>(pages, 1, decodeMostSplits(SPLASH_TARGET_VERIFY_ROWS, cores));
  const uint64_t splits = std::clamp<uint64_t>((pages * cores + kDecodeCorePages - 1) / kDecodeCorePages,
                                               std::min(fewest, most), most);
  const uint64_t perSplit = std::max<uint64_t>((pages + splits - 1) / splits, 1);
  return static_cast<uint32_t>(std::max<uint64_t>((pages + perSplit - 1) / perSplit, 1));
}

void RelativeAttention::group(RelativeAttentionPlan &plan) {
  plan.dispatchCount_ = groupDispatches(std::span(plan.sequences_).first(plan.sequenceCount_), plan.workspaceSlots,
                                        plan.dispatches_, plan.step == StepKind::Decode ? 2 : 1);
}

void RelativeAttention::addProjection(metal::CommandGraph &graph, const RelativeAttentionRows &layerRows,
                                      const RelativeAttentionWeights &weights, const RelativeAttentionStaging &staging,
                                      SplashKvLayer layer, std::span<const metal::MetalBuffer> pageTables,
                                      std::span<const kv::ChunkedPrefillParams> sequences, const AttentionRows &rows,
                                      kv::Layout layout, RelativeAttentionVariant variant) {
  requireLayout(layout);
  requireVariant(variant);
  const CommandRows command = commandRows(sequences, layout);
  if (pageTables.size() != sequences.size())
    throw std::invalid_argument("a q/k projection takes a page table per sequence");
  requireRows(rows, command, sequences.size());
  const uint64_t kvBytes = uint64_t{rows.rows} * kKvHeads * kHeadDimension * 2;
  requireBytes(layerRows.projection, rowBytes(rows.rows, kProjectionStride, kProjectionWidth, 2),
               "attention projection rows");
  requireBytes(layerRows.keys, kvBytes, "attention keys");
  requireBytes(layerRows.values, kvBytes, "attention values");
  requireBytes(weights.queryScale, uint64_t{kQueryHeads} * kHeadDimension * sizeof(float), "attention query scale");
  requireBytes(weights.relativeScale, uint64_t{kQueryHeads} * kStates * sizeof(float), "attention relative scale");
  requireStaging(staging, command, command.last.chunk_tokens);
  for (size_t s = 0; s < sequences.size(); ++s) requirePageTable(pageTables[s], sequences[s]);
  std::vector<metal::MetalBuffer> bindings{layerRows.projection, layerRows.keys,  layerRows.values,
                                           weights.queryScale,   weights.relativeScale, staging.queries,
                                           staging.states,       staging.sinks};
  const auto tables = laneBindings(pageTables);
  bindings.insert(bindings.end(), tables.begin(), tables.end());
  // Without tree rows the prepare binds another buffer of the dispatch for
  // them, of which it reads nothing.
  bindings.insert(bindings.end(), {rows.tree ? rows.tree : layerRows.projection, rows.layout});
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
  graph.add("relative_attention_" + kernelFormat(layout) + "_prepare" + (variant.positionScaled ? "_scaled" : ""),
            std::move(bindings), params, {kQueryHeads + kKvHeads, command.mostRows, sequences.size()},
            {kHeadDimension, 1, 1});
}

PreparedInput RelativeAttention::addAttention(metal::CommandGraph &graph, const RelativeAttentionPlan &plan,
                                              const RelativeAttentionStaging &staging,
                                              const RelativeAttentionWeights &weights,
                                              const metal::MetalBuffer &partials,
                                              const metal::MetalBuffer &statistics,
                                              std::span<const kv::ChunkedPrefillParams> sequences,
                                              std::span<const AttentionHistory> histories, const AttentionRows &rows,
                                              const AttentionOutputRows &output) {
  const CommandRows command = commandRows(sequences, plan.layout);
  const std::span<const RelativeAttentionPlan::Sequence> planned = plan.sequences();
  if (planned.size() != sequences.size() || histories.size() != sequences.size())
    throw std::invalid_argument("attention sequences do not match the plan");
  for (size_t s = 0; s < sequences.size(); ++s) {
    if (sequences[s].chunk_tokens != planned[s].rows || sequences[s].committed_tokens != planned[s].historyTokens)
      throw std::invalid_argument("attention sequences do not match the plan");
    requirePageTable(histories[s].pageTable, sequences[s]);
    if (histories[s].blocks.stride) throw std::invalid_argument("relative attention reads every page");
    if (rows.tree && (planned[s].firstTile || planned[s].rows > 32))
      throw std::invalid_argument("attention tree rows take whole blocks of at most 32 rows");
  }
  requireRows(rows, command, sequences.size());
  // The tiles read the staging in whole tiles of rows, the direct kernel's of
  // SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS, which each sequence's stride holds
  // (a multiple of a KV block's tokens).
  const RelativeAttentionPlan::Sequence &lastSequence = planned.back();
  requireStaging(staging, command,
                 lastSequence.splits == 1 ? lastSequence.firstTile * kv::kAttentionTileRows +
                                                RelativeAttentionPlan::directTiles(lastSequence) *
                                                    SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS
                                          : kv::attentionTiles(command.last.chunk_tokens) * kv::kAttentionTileRows);
  requireBytes(weights.bias, uint64_t{SPLASH_RELATIVE_ATTENTION_EXTENT} * kStates * sizeof(float),
               "attention relative bias");
  const bool last = output.output == AttentionOutput::Last, table = output.input != LinearInput::Plain;
  for (const RelativeAttentionPlan::Sequence &sequence : planned)
    if (last && sequence.firstTile + 1 != kv::attentionTiles(sequence.rows))
      throw std::invalid_argument("an attention of each sequence's last row takes the plan of its last tiles");
  const uint32_t width = kQueryHeads * kHeadDimension;
  const uint32_t outputRows = last ? static_cast<uint32_t>(sequences.size()) : rows.rows;
  // The reduce reads each row's gates, the block after its queries.
  requireBytes(output.packed, rowBytes(rows.rows, kProjectionStride, 2 * uint64_t{width}, 2), "attention gate");
  requireBytes(output.hidden, uint64_t{outputRows} * width * 2, "attention hidden");
  if (table) requireTableScratch(output.scratch, output.input, width, outputRows);
  const metal::MetalBuffer &tableInput = table ? output.scratch.input : output.hidden;
  const metal::MetalBuffer &tableSums = table ? output.scratch.sums : output.hidden;
  const std::string reducePipeline = plan.reducePipeline() + tableSuffix(output.input);
  const AttentionWorkspace workspace = plan.workspace();
  requireBytes(partials, workspace.partialsBytes, "attention partials");
  requireBytes(statistics, workspace.statisticsBytes, "attention statistics");
  // A dispatch of one split a sequence runs the direct kernel, whose tiles
  // of SPLASH_RELATIVE_ATTENTION_DIRECT_ROWS rows count first_tile and the
  // dispatch's tiles, and whose slots, a tile's INT8 query codes, are fewer
  // than the split's.
  const auto encode = [&](const RelativeAttentionPlan::Dispatch &dispatch, SplashKvLayer layer, uint32_t firstSlot) {
    const bool direct = plan.direct(dispatch);
    kv::AttentionParams params{.sequences = dispatch.sequences,
                               .first_sequence = dispatch.sequence[0],
                               .rows = rows.rows,
                               .tiles = direct ? static_cast<uint32_t>(plan.directGroups(dispatch).y)
                                               : dispatch.tiles,
                               .trees = rows.tree ? 1U : 0U,
                               .last_rows = last ? 1U : 0U,
                               .kv = layer,
                               .sequence = {}};
    std::array<metal::MetalBuffer, kMaximumSequences> tables;
    uint32_t tile = 0, slot = firstSlot;
    for (uint32_t d = 0; d < dispatch.sequences; ++d) {
      const uint32_t s = dispatch.sequence[d];
      if (s != dispatch.sequence[0] + d) throw std::logic_error("an attention dispatch's sequences are not in order");
      const RelativeAttentionPlan::Sequence &sequence = planned[s];
      SplashAttentionSequence &entry = params.sequence[d];
      entry = sequenceRows(sequences[s], command, s, sequence.firstTile * kv::kAttentionTileRows);
      entry.split_count = sequence.splits;
      entry.first_tile = tile;
      entry.first_slot = slot;
      const uint32_t tiles = direct ? RelativeAttentionPlan::directTiles(sequence)
                                    : kv::attentionTiles(sequence.rows) - sequence.firstTile;
      tile += tiles;
      slot += tiles * kKvHeads * sequence.splits;
      tables[d] = histories[s].pageTable;
    }
    const auto bound = laneBindings(std::span(tables).first(dispatch.sequences));
    std::vector<metal::MetalBuffer> split{staging.queries, partials};
    if (!direct) split.push_back(statistics);
    split.insert(split.end(), bound.begin(), bound.end());
    // Without tree rows a split binds another buffer of the dispatch for
    // them, of which it reads nothing.
    split.insert(split.end(), {rows.tree ? rows.tree : partials, rows.layout, staging.states, weights.bias});
    if (direct) {
      split.insert(split.end(), {output.packed, output.hidden, tableInput, tableSums, staging.sinks});
      graph.add(plan.directPipeline() + tableSuffix(output.input), std::move(split), params,
                plan.directGroups(dispatch));
      return slot;
    }
    graph.add(plan.splitPipeline(), std::move(split), params, plan.splitGroups(dispatch));
    graph.add(reducePipeline,
              {partials, statistics, output.packed, output.hidden, tableInput, tableSums, rows.layout, staging.sinks},
              params, plan.reduceGroups(dispatch), {kHeadDimension, 1, 1});
    return slot;
  };
  for (const RelativeAttentionPlan::Dispatch &dispatch : plan.dispatches()) {
    uint32_t slot = 0;
    for (uint32_t first = 0, end = 0; first < dispatch.sequences; first = end) {
      const SplashKvLayer layer = histories[dispatch.sequence[first]].kv;
      RelativeAttentionPlan::Dispatch run{};
      for (end = first; end < dispatch.sequences && sameLayer(histories[dispatch.sequence[end]].kv, layer); ++end) {
        const RelativeAttentionPlan::Sequence &sequence = planned[dispatch.sequence[end]];
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

PreparedInput RelativeAttention::addDecode(metal::CommandGraph &graph, const RelativeAttentionPlan &plan,
                                           const RelativeAttentionRows &layerRows,
                                           const RelativeAttentionWeights &weights, SplashKvLayer layer,
                                           std::span<const metal::MetalBuffer> pageTables,
                                           std::span<const kv::ChunkedPrefillParams> sequences,
                                           const AttentionRows &rows, const RelativeAttentionWorkspace &workspace,
                                           const AttentionOutputRows &output) {
  if (plan.step != StepKind::Decode) throw std::invalid_argument("a single pass takes a decode step's plan");
  const CommandRows command = commandRows(sequences, plan.layout);
  const std::span<const RelativeAttentionPlan::Sequence> planned = plan.sequences();
  if (planned.size() != sequences.size() || pageTables.size() != sequences.size())
    throw std::invalid_argument("attention sequences do not match the plan");
  for (size_t s = 0; s < sequences.size(); ++s) {
    if (sequences[s].chunk_tokens != planned[s].rows || sequences[s].committed_tokens != planned[s].historyTokens)
      throw std::invalid_argument("attention sequences do not match the plan");
    requirePageTable(pageTables[s], sequences[s]);
  }
  requireRows(rows, command, sequences.size());
  const uint64_t kvBytes = uint64_t{rows.rows} * kKvHeads * kHeadDimension * 2;
  requireBytes(layerRows.projection, rowBytes(rows.rows, kProjectionStride, kProjectionWidth, 2),
               "attention projection rows");
  requireBytes(layerRows.keys, kvBytes, "attention keys");
  requireBytes(layerRows.values, kvBytes, "attention values");
  requireBytes(weights.queryScale, uint64_t{kQueryHeads} * kHeadDimension * sizeof(float), "attention query scale");
  requireBytes(weights.relativeScale, uint64_t{kQueryHeads} * kStates * sizeof(float), "attention relative scale");
  requireBytes(weights.bias, uint64_t{SPLASH_RELATIVE_ATTENTION_EXTENT} * kStates * sizeof(float),
               "attention relative bias");
  const bool last = output.output == AttentionOutput::Last, table = output.input != LinearInput::Plain;
  const uint32_t width = kQueryHeads * kHeadDimension;
  const uint32_t outputRows = last ? static_cast<uint32_t>(sequences.size()) : rows.rows;
  requireBytes(output.hidden, uint64_t{outputRows} * width * 2, "attention hidden");
  if (table) requireTableScratch(output.scratch, output.input, width, outputRows);
  const AttentionWorkspace needed = plan.workspace();
  requireBytes(workspace.partials, needed.partialsBytes, "attention partials");
  requireBytes(workspace.statistics, needed.statisticsBytes, "attention statistics");
  requireBytes(workspace.counters, kCountersBytes, "attention counters");
  for (const RelativeAttentionPlan::Dispatch &dispatch : plan.dispatches()) {
    kv::AttentionParams params{.sequences = dispatch.sequences,
                               .first_sequence = dispatch.sequence[0],
                               .rows = rows.rows,
                               .tiles = dispatch.tiles,
                               .trees = rows.tree ? 1U : 0U,
                               .last_rows = last ? 1U : 0U,
                               .kv = layer,
                               .sequence = {}};
    std::array<metal::MetalBuffer, kMaximumSequences> tables;
    // Each sequence's splits take a slot of partials and one of staging.
    uint32_t slot = 0;
    for (uint32_t d = 0; d < dispatch.sequences; ++d) {
      const uint32_t s = dispatch.sequence[d];
      if (s != dispatch.sequence[0] + d) throw std::logic_error("an attention dispatch's sequences are not in order");
      SplashAttentionSequence &entry = params.sequence[d];
      entry = sequenceRows(sequences[s], command, s, 0);
      entry.split_count = planned[s].splits;
      entry.first_tile = d;
      entry.first_slot = slot;
      slot += 2 * kv::attentionTiles(planned[s].rows) * kKvHeads * planned[s].splits;
      tables[d] = pageTables[s];
    }
    const auto bound = laneBindings(std::span(tables).first(dispatch.sequences));
    std::vector<metal::MetalBuffer> bindings{layerRows.projection, layerRows.keys,       layerRows.values,
                                             weights.queryScale,   weights.relativeScale, weights.bias,
                                             workspace.partials,   workspace.statistics,  workspace.counters,
                                             output.hidden,        table ? output.scratch.input : output.hidden,
                                             table ? output.scratch.sums : output.hidden};
    bindings.insert(bindings.end(), bound.begin(), bound.end());
    // Without tree rows the single pass binds another buffer of the dispatch
    // for them, of which it reads nothing.
    bindings.insert(bindings.end(), {rows.tree ? rows.tree : layerRows.projection, rows.layout});
    graph.add(plan.decodePipeline() + tableSuffix(output.input), std::move(bindings), params,
              plan.splitGroups(dispatch));
  }
  return table ? PreparedInput{output.hidden, output.input} : PreparedInput{};
}

void RelativeAttention::addPathCommit(metal::CommandGraph &graph, kv::Layout layout, uint32_t extentPages,
                                      std::span<const metal::MetalBuffer> pageTables,
                                      std::span<const uint32_t> committed, uint32_t laneRows,
                                      const metal::MetalBuffer &paths) {
  requireLayout(layout);
  const uint32_t lanes = static_cast<uint32_t>(committed.size());
  if (!lanes || lanes > kMaximumSequences || pageTables.size() != lanes || !extentPages || !laneRows ||
      laneRows > SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("invalid KV path commit");
  SplashKvPathCommitParams params{
      .extent_pages = extentPages, .regions = layout.attentionLayers, .committed_tokens = {}};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint64_t pages = (uint64_t{committed[lane]} + laneRows + kv::kPageTokens - 1) / kv::kPageTokens;
    requireBytes(pageTables[lane], pages * sizeof(SplashKvPage), "page table");
    params.committed_tokens[lane] = committed[lane];
  }
  requireBytes(paths, uint64_t{lanes} * sizeof(uint32_t), "KV path commit paths");
  const auto tables = laneBindings(pageTables);
  std::vector<metal::MetalBuffer> bindings(tables.begin(), tables.end());
  bindings.push_back(paths);
  graph.add("relative_attention_" + kernelFormat(layout) + "_commit", std::move(bindings), params,
            {uint64_t{layout.attentionLayers} * layout.kvHeads, lanes, 1}, {layout.headDimension, 1, 1});
}

} // namespace splash::ops

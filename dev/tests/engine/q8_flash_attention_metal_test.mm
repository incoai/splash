#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "HostKvExtents.hpp"
#include "TestChecks.hpp"
#include "TestFamilies.hpp"
#include "ops/PagedAttention.hpp"
#include "Q8PageFormatReference.hpp"
#include "ops/KernelCalibration.hpp"
#include "ops/LaneRows.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash::kv;
using splash::test::HostKvExtents;

namespace {

// A lane's rows, a tile's at most, staged in one KV block per KV head.
constexpr uint32_t kStride = kPageTokens;
constexpr uint32_t kQueryStride = kStride;
constexpr uint32_t kRows = kAttentionTileRows;
static_assert(kRows == splash::ops::kNarrowLaneRows, "a lane of the narrow cap's rows is one tile");

// The production GQA geometries, every family's (TestFamilies.hpp). The
// group size selects the kernel specialization, which the plans name
// (ops::PagedAttention). A QSA family's group runs the sparse kernels, which
// attend through block lists, bound before their parameters, and read
// token-major values; a group of more than eight runs each tile in parts,
// each a threadgroup of the split with partial slots of its own
// (splash_attention_tile_part). A dense geometry also runs its INT8-query
// entries (q8q), and Apple8's entries of half queries and probabilities
// (_h), whose queries it converts to half.
struct Shape {
  uint32_t kvHeads;
  uint32_t queryHeadsPerKvHead;
  splash::kv::Reads reads;
  bool int8Queries = false;
  bool halfQueries = false;
  uint32_t queryHeads() const { return kvHeads * queryHeadsPerKvHead; }
  uint32_t fusedRows() const { return kRows * queryHeadsPerKvHead; }
  bool sparse() const { return reads == splash::kv::Reads::SelectedBlocks; }
  uint32_t parts() const { return sparse() ? 1 : SPLASH_ATTENTION_TILE_PARTS(queryHeadsPerKvHead); }
  // The partial of a fused row of the tile of KV head `head` from split
  // `split` of `slotSplits`: [KV head][part][split][part's fused row].
  uint64_t partialRow(uint32_t head, uint32_t slotSplits, uint32_t split, uint32_t row) const {
    const uint32_t partRows = fusedRows() / parts();
    return ((uint64_t{head} * parts() + row / partRows) * slotSplits + split) * partRows + row % partRows;
  }
  Layout layout() const {
    return {.attentionLayers = 1, .kvHeads = kvHeads, .headDimension = kHeadDimension,
            .valuesTokenMajor = splash::kv::valuesTokenMajorFor(reads), .reads = reads};
  }
  // The split and reduce entries, as the plans name them.
  std::string split() const { return plan().splitPipeline(); }
  std::string reduce() const { return plan().reducePipeline(); }

private:
  // A plan of a lane's rows, with INT8 queries in every split or none.
  splash::ops::AttentionPlan plan() const {
    const std::array lane{splash::ops::PagedAttention::chunkParams(0, kRows, kStride, 1)};
    return splash::ops::PagedAttention::plan(
        lane, queryHeads(), layout(), splash::test::attentionRow(int8Queries ? 1U : splash::ops::kNever),
        splash::ops::Int8Activations::On,
        splash::ops::PagedAttention::workspace(kAttentionMaximumSplits, queryHeads(), layout()), {}, halfQueries);
  }
};
const std::vector<Shape> kShapes = [] {
  std::vector<Shape> shapes;
  for (const splash::test::AttentionGeometry &geometry : splash::test::familyAttentionGeometries()) {
    shapes.push_back({geometry.kvHeads, geometry.queryHeads / geometry.kvHeads, geometry.reads});
    if (geometry.reads == splash::kv::Reads::EveryPage) {
      shapes.push_back({geometry.kvHeads, geometry.queryHeads / geometry.kvHeads, geometry.reads, true});
      shapes.push_back({geometry.kvHeads, geometry.queryHeads / geometry.kvHeads, geometry.reads, false, true});
    }
  }
  return shapes;
}();

using splash::test::require;

// A decode lane's tile: 32 splits until it reads more than 16 pages a split,
// one per 16 pages after, at most kAttentionMaximumSplits.
uint32_t laneSplits(uint32_t history) {
  const std::array lane{splash::ops::PagedAttention::chunkParams(history, kRows, kStride,
                                                                 (history + kRows + kPageTokens - 1) / kPageTokens)};
  const splash::kv::Layout layout{.attentionLayers = 1, .kvHeads = 4, .headDimension = kHeadDimension};
  return splash::ops::PagedAttention::plan(lane, 24, layout, splash::ops::kApple10Attention,
                                           splash::ops::Int8Activations::On,
                                           splash::ops::PagedAttention::workspace(kAttentionMaximumSplits, 24, layout))
      .sequences()[0]
      .splits;
}

void testContract() {
  ChunkedPrefillParams finalCycle{
      splash::kv::kMaximumLogicalTokens - 1, SPLASH_TARGET_VERIFY_ROWS, kStride,
      (splash::kv::kMaximumPhysicalTokens + kPageTokens - 1) / kPageTokens};
  require(chunkedPrefillValidationError(finalCycle).empty(),
          "the final cycle's verification rows exceeded physical KV scratch");
  require(laneSplits(0) == 32 && laneSplits(16 * 1024) == 33 && laneSplits(131072) == kAttentionMaximumSplits,
          "a lane's split scaling departed from one split per 16 visible pages");
  ++finalCycle.committed_tokens;
  require(chunkedPrefillValidationError(finalCycle) == "context_out_of_range",
          "physical KV scratch exceeded its fixed fifteen-row allowance");
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, uint64_t bytes) {
  id<MTLBuffer> result =
      [device newBufferWithLength:std::max<uint64_t>(bytes, 1)
                          options:MTLResourceStorageModeShared];
  if (!result)
    throw std::runtime_error("Metal buffer allocation failed");
  std::memset(result.contents, 0, result.length);
  return result;
}

// The packed projection rows of `rows` rows of a shape, from which its
// reduce reads each query head's gate (every geometry of the head dimension
// is gated): every gate +inf, whose sigmoid is 1, so that the reduce writes
// each attention row as it is.
id<MTLBuffer> identityGates(id<MTLDevice> device, Shape shape, uint64_t rows) {
  const uint64_t width = (uint64_t{shape.queryHeads()} * 2 + uint64_t{shape.kvHeads} * 2) * kHeadDimension;
  id<MTLBuffer> packed = makeBuffer(device, rows * width * sizeof(uint16_t));
  auto *elements = static_cast<uint16_t *>(packed.contents);
  for (uint64_t row = 0; row < rows; ++row)
    for (uint32_t head = 0; head < shape.queryHeads(); ++head)
      std::fill_n(elements + row * width + (uint64_t{head} * 2 + 1) * kHeadDimension, kHeadDimension,
                  uint16_t{0x7f80});
  return packed;
}

id<MTLComputePipelineState>
makePipeline(id<MTLDevice> device, id<MTLLibrary> library,
             const std::string &name) {
  id<MTLFunction> function =
      [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!function)
    throw std::runtime_error("missing Metal kernel: " + name);
  NSError *error = nil;
  id<MTLComputePipelineState> result =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!result)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  return result;
}

// Shader validation (MTL_SHADER_VALIDATION=1, how make test-engine runs this
// test) instruments every pipeline's threadgroup memory (2x-12 bytes for
// these tiles), so the printed lengths are the kernels' own only when it is
// off. The checks below compare pipelines with each other, which holds
// either way; an absolute bound would only ever be checked uninstrumented.
bool shaderValidationEnabled() {
  const char *value = std::getenv("MTL_SHADER_VALIDATION");
  return value && std::strcmp(value, "0") != 0;
}

void finish(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    std::string message = "Metal command failed";
    if (command.error) {
      message += ": ";
      message += command.error.localizedDescription.UTF8String;
    }
    throw std::runtime_error(message);
  }
}

float keyPattern(uint32_t token, uint32_t head, uint32_t dimension) {
  int32_t centered =
      int32_t((uint64_t{token} * 37 + head * 101 + dimension * 17 +
               uint64_t{token} * dimension * 3) %
              2003) -
      1001;
  return float(centered) / 2002.0f;
}

float valuePattern(uint32_t token, uint32_t head, uint32_t dimension) {
  int32_t centered =
      int32_t((uint64_t{token} * 53 + head * 79 + dimension * 29 +
               uint64_t{token} * dimension * 5) %
              2011) -
      1005;
  return float(centered) / 1005.0f;
}

float queryPattern(uint32_t row, uint32_t head, uint32_t dimension) {
  int32_t centered = int32_t((uint64_t{row} * 43 + head * 67 + dimension * 11 +
                              uint64_t{head} * dimension * 7) %
                             1019) -
                     509;
  return float(centered) / 2036.0f;
}

// The attention layer under test is the second of a pool's two, so its region
// starts past the first one's in every extent.
constexpr uint32_t kLayer = 1;

// One lane's rows after its history, of one tile, its sequence's parameters
// held here; a dispatch of several lanes stages each lane's after the one
// before it.
struct Case {
  Shape shape;
  SplashAttentionSequence params;
  // Page ids, and their entries for the kernels.
  std::vector<uint32_t> pageTable;
  id<MTLBuffer> pageTableBuffer;
  // The pool's extents, which a dispatch reaching them makes resident.
  std::vector<id<MTLBuffer>> extents;
  std::unique_ptr<HostKvExtents> pool;
  id<MTLBuffer> queries;
  // QSA block masks (group of twelve only): maskWords words per row of the
  // rows' attended blocks of SPLASH_ATTENTION_BLOCK_TOKENS tokens; none when
  // maskWords is 0.
  std::vector<uint32_t> mask;
  uint32_t maskWords = 0;
  // The rows' places as a tree (metal/abi/VerifyTree.h, SplashAttentionParams
  // ::trees): a row sees, of the rows, its ancestors and itself; empty for
  // rows that attend causally. Through QSA masks a tree's row sees only
  // those of them its row mask holds (ops::AttentionBlocks::rowMasks).
  std::vector<SplashTreeRow> tree;
  std::vector<uint32_t> rowMasks;
  // The token whose patterns each slot's K/V hold and the row whose query
  // each row holds: themselves, or for a chain of a tree's path its rows'.
  std::vector<uint32_t> tokens;
  std::vector<uint32_t> queryRows;

  uint32_t token(uint32_t slot) const { return tokens.empty() ? slot : tokens[slot]; }
  uint32_t queryRow(uint32_t row) const { return queryRows.empty() ? row : queryRows[row]; }
  bool attends(uint32_t row, uint32_t token) const {
    if (!maskWords) return true;
    const uint32_t block = token / SPLASH_ATTENTION_BLOCK_TOKENS;
    return (mask[uint64_t{row} * maskWords + block / 32] >> (block % 32)) & 1u;
  }
  bool onPath(uint32_t row, uint32_t token) const {
    const uint32_t visible = tree.empty() ? 0u : tree[row].ancestors & (maskWords ? rowMasks[row] : ~0u);
    return tree.empty() || token < params.committed_tokens || ((visible >> (token - params.committed_tokens)) & 1u);
  }

  // The attention's output of lane `lane`'s row of a dispatch of the case's
  // rows a lane, as its reduce writes the out-projection's input rows:
  // [lane][row][query head][dimension].
  uint64_t outputIndex(uint32_t lane, uint32_t head, uint32_t row, uint32_t dimension) const {
    return ((uint64_t{lane} * params.row_bound + row) * shape.queryHeads() + head) * kHeadDimension + dimension;
  }
  uint64_t queryIndex(uint32_t head, uint32_t row, uint32_t dimension) const {
    const uint32_t kvHead = head / shape.queryHeadsPerKvHead;
    const uint32_t localHead = head % shape.queryHeadsPerKvHead;
    return ((uint64_t{kvHead} * kQueryStride + row) * shape.queryHeadsPerKvHead +
            localHead) *
               kHeadDimension +
           dimension;
  }
  template <typename T> T *slab(uint32_t tensor, uint32_t logicalPage) const {
    return pool->slab<T>(kLayer, tensor, pageTable[logicalPage]);
  }
};

// The case's pages, mixed over three or more extents of its pool, or in one
// extent of the same pages.
Case makeCase(id<MTLDevice> device, Shape shape, uint32_t committed,
              uint32_t activeRows, uint32_t splits, bool oneExtent = false) {
  Case result;
  result.shape = shape;
  Layout layout = shape.layout();
  layout.attentionLayers = kLayer + 1;
  uint32_t pages = (committed + activeRows + kPageTokens - 1) / kPageTokens;
  const auto spread = HostKvExtents::spread(pages + 2);
  const HostKvExtents::Geometry geometry =
      oneExtent ? HostKvExtents::Geometry{spread.extentPages * spread.extents, 1} : spread;
  std::vector<HostKvExtents::Extent> extents;
  for (uint32_t extent = 0; extent < geometry.extents; ++extent) {
    result.extents.push_back(makeBuffer(
        device, HostKvExtents::extentBytes(layout, geometry.extentPages)));
    extents.push_back({static_cast<std::byte *>(result.extents.back().contents),
                       result.extents.back().gpuAddress});
  }
  result.pool = std::make_unique<HostKvExtents>(layout, geometry.extentPages,
                                                std::move(extents));
  result.pageTable = HostKvExtents::mixedPages(spread, pages, committed + activeRows);
  result.pageTableBuffer = makeBuffer(device, pages * sizeof(SplashKvPage));
  result.pool->writeTable(result.pageTable, result.pageTableBuffer.contents);
  result.params = {.committed_tokens = committed,
                   .row_bound = activeRows,
                   .layout_row = 0,
                   .stage_row = 0,
                   .chunk_stride = kStride,
                   .page_table_entries = pages,
                   .split_count = splits,
                   .first_tile = 0,
                   .first_slot = 0,
                   .block_stride = 0,
                   .block_list = 0};
  uint64_t queryElements =
      uint64_t{shape.queryHeads()} * kQueryStride * kHeadDimension;
  result.queries = makeBuffer(device, queryElements * sizeof(BFloat16Bits));
  return result;
}

// The same pages and queries in one extent.
Case oneExtentCopy(id<MTLDevice> device, const Case &data) {
  Case result = makeCase(device, data.shape, data.params.committed_tokens,
                         data.params.row_bound, data.params.split_count, true);
  const Layout layout = data.shape.layout();
  const uint64_t dataBytes = layout.dataBytesPerLayerPage();
  const uint64_t scaleBytes = layout.scaleBytesPerLayerPage();
  for (uint32_t page = 0; page < data.pageTable.size(); ++page)
    for (uint32_t tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES; ++tensor)
      std::memcpy(result.slab<std::byte>(tensor, page), data.slab<std::byte>(tensor, page),
                  tensor % 2 ? scaleBytes : dataBytes);
  std::memcpy(result.queries.contents, data.queries.contents, data.queries.length);
  result.mask = data.mask;
  result.maskWords = data.maskWords;
  result.tree = data.tree;
  result.rowMasks = data.rowMasks;
  return result;
}

// Blocks a QSA row attends past kQsaTopBlocks complete blocks; up to there
// every block up to its own, as block_indexer_select writes them.
constexpr uint32_t kQsaTopBlocks = 40;
// Block lists of every block for every row ahead of a dispatch's lists, and
// after them, which a tile reading another list than its own would attend.
constexpr uint32_t kQsaJunkLists = 3;

// The case's block list (metal/abi/PagedAttention.h): each block of its
// rows' history that one of them attends, with their bits.
std::vector<uint32_t> blockList(const Case &data) {
  constexpr uint32_t kBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;
  const uint32_t history = (data.params.committed_tokens + data.params.row_bound + kBlockTokens - 1) / kBlockTokens;
  std::vector<uint32_t> result;
  for (uint32_t block = 0; block < history; ++block) {
    uint32_t rows = 0;
    for (uint32_t row = 0; row < data.params.row_bound; ++row)
      rows |= uint32_t(data.attends(row, block * kBlockTokens)) << row;
    if (rows) result.push_back(block << SPLASH_ATTENTION_BLOCK_ROW_BITS | rows);
  }
  return result;
}

// A dispatch's block lists and their counts: kQsaJunkLists lists, then
// `copies` copies of the case's (one per verify lane), then kQsaJunkLists
// more; `stride` entries each.
struct BlockLists {
  id<MTLBuffer> lists;
  id<MTLBuffer> counts;
  uint32_t stride;
};

BlockLists qsaBlockLists(id<MTLDevice> device, const Case &data, uint32_t copies) {
  const std::vector<uint32_t> list = blockList(data);
  constexpr uint32_t kBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;
  const uint32_t history = (data.params.committed_tokens + data.params.row_bound + kBlockTokens - 1) / kBlockTokens;
  const uint32_t total = copies + 2 * kQsaJunkLists;
  BlockLists result{makeBuffer(device, uint64_t{total} * history * 4), makeBuffer(device, total * 4), history};
  auto *entries = static_cast<uint32_t *>(result.lists.contents);
  auto *counts = static_cast<uint32_t *>(result.counts.contents);
  for (uint32_t index = 0; index < total; ++index) {
    uint32_t *out = entries + uint64_t{index} * history;
    const bool junk = index < kQsaJunkLists || index >= kQsaJunkLists + copies;
    if (junk)
      for (uint32_t block = 0; block < history; ++block)
        out[block] = block << SPLASH_ATTENTION_BLOCK_ROW_BITS | ((1u << kRows) - 1u);
    else
      std::copy(list.begin(), list.end(), out);
    counts[index] = junk ? history : uint32_t(list.size());
  }
  return result;
}

// Masks whose blocks a tile's rows attend in every way its list must tell
// apart: whole pages for every row, one block per row of a page (no single
// row attends the whole page), half a page or one block that only one row
// attends, and pages no row attends; each row also attends its own block.
// Rows past kQsaTopBlocks complete blocks attend sparsely, earlier ones
// causally.
void setQsaMask(Case &data) {
  constexpr uint32_t kBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;
  constexpr uint32_t kPageBlocks = kPageTokens / kBlockTokens;
  const uint32_t visible = data.params.committed_tokens + data.params.row_bound;
  data.maskWords = visible / kBlockTokens / 32 + 2;
  data.mask.assign(uint64_t{data.params.row_bound} * data.maskWords, 0u);
  for (uint32_t row = 0; row < data.params.row_bound; ++row) {
    uint32_t *bits = data.mask.data() + uint64_t{row} * data.maskWords;
    const uint32_t complete = (data.params.committed_tokens + row + 1) / kBlockTokens;
    for (uint32_t block = 0; block <= complete; ++block) {
      const uint32_t page = block / kPageBlocks, slot = block % kPageBlocks;
      const bool take = complete <= kQsaTopBlocks || block == complete ||
                        page % 5 == 0 || (page % 5 == 2 && slot == row % 8) ||
                        (page % 11 == 1 + row % 8 && slot % 2 == 0) ||
                        (page % 13 == 2 + row % 8 && slot == (page + row) % 8);
      if (take) bits[block / 32] |= 1u << (block % 32);
    }
  }
}

// The page format is head-major inside a page, so a two-head page is the
// first two heads of the four-head oracle page: quantize through the shared
// oracle and store the layout's own head count, values in its order.
void fill(Case &data) {
  const Shape shape = data.shape;
  const Layout layout = shape.layout();
  std::vector<float> pageKeys;
  std::vector<float> pageValues;
  auto quantized = std::make_unique<Q8LayerPage>();
  uint32_t visibleTokens = data.params.committed_tokens + data.params.row_bound;
  uint32_t visiblePages = (visibleTokens + kPageTokens - 1) / kPageTokens;
  for (uint32_t logicalPage = 0; logicalPage < visiblePages; ++logicalPage) {
    uint32_t valid =
        std::min(kPageTokens, visibleTokens - logicalPage * kPageTokens);
    pageKeys.assign(uint64_t{valid} * kKvHeads * kHeadDimension, 0.0f);
    pageValues.assign(uint64_t{valid} * kKvHeads * kHeadDimension, 0.0f);
    for (uint32_t token = 0; token < valid; ++token) {
      uint32_t global = logicalPage * kPageTokens + token;
      for (uint32_t head = 0; head < shape.kvHeads; ++head) {
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          uint64_t index = logicalIndex(token, head, dimension);
          pageKeys[index] = bfloat16ToFloat(
              floatToBFloat16(keyPattern(data.token(global), head, dimension)));
          pageValues[index] = bfloat16ToFloat(
              floatToBFloat16(valuePattern(data.token(global), head, dimension)));
        }
      }
    }
    quantizeLayerPage(pageKeys, pageValues, valid, *quantized);
    const uint64_t elements = layout.elementsPerLayerPage();
    const uint64_t scales = layout.scalesPerTensorLayerPage();
    std::copy_n(quantized->keys.begin(), elements,
                data.slab<int8_t>(SPLASH_KV_KEYS, logicalPage));
    std::copy_n(quantized->keyScales.begin(), scales,
                data.slab<float>(SPLASH_KV_KEY_SCALES, logicalPage));
    int8_t *values = data.slab<int8_t>(SPLASH_KV_VALUES, logicalPage);
    for (uint32_t head = 0; head < shape.kvHeads; ++head)
      for (uint32_t token = 0; token < kPageTokens; ++token)
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
          values[data.pool->valueElement(head, token, dimension)] =
              quantized->values[splash_kv_value_element(head, token, dimension)];
    std::copy_n(quantized->valueScales.begin(), scales,
                data.slab<float>(SPLASH_KV_VALUE_SCALES, logicalPage));
  }

  auto *queries = static_cast<BFloat16Bits *>(data.queries.contents);
  for (uint32_t head = 0; head < shape.queryHeads(); ++head)
    for (uint32_t row = 0; row < data.params.row_bound; ++row)
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension)
        queries[data.queryIndex(head, row, dimension)] =
            floatToBFloat16(queryPattern(data.queryRow(row), head, dimension));
}

float loadKey(const Case &data, uint32_t token, uint32_t head,
              uint32_t dimension) {
  uint32_t logicalPage = token / kPageTokens;
  uint32_t pageToken = token % kPageTokens;
  float scale = data.slab<const float>(SPLASH_KV_KEY_SCALES, logicalPage)[
      splash_kv_scale_element(head, pageToken)];
  return float(data.slab<const int8_t>(SPLASH_KV_KEYS, logicalPage)[
             splash_kv_key_element(head, pageToken, dimension)]) *
         scale;
}

float loadValue(const Case &data, uint32_t token, uint32_t head,
                uint32_t dimension) {
  uint32_t logicalPage = token / kPageTokens;
  uint32_t pageToken = token % kPageTokens;
  float scale = data.slab<const float>(SPLASH_KV_VALUE_SCALES, logicalPage)[
      splash_kv_scale_element(head, pageToken)];
  return float(data.slab<const int8_t>(SPLASH_KV_VALUES, logicalPage)[
             data.pool->valueElement(head, pageToken, dimension)]) *
         scale;
}

// Double-precision softmax attention over stored Q8 pages or the unquantized
// BF16 pattern, with rounding to BF16 only at the final output.
std::vector<BFloat16Bits> cpuReference(const Case &data, bool quantized) {
  const Shape shape = data.shape;
  std::vector<BFloat16Bits> output(
      uint64_t{shape.queryHeads()} * kQueryStride * kHeadDimension,
      BFloat16Bits{0});
  const auto *queries =
      static_cast<const BFloat16Bits *>(data.queries.contents);
  for (uint32_t queryHead = 0; queryHead < shape.queryHeads(); ++queryHead) {
    uint32_t kvHead = queryHead / shape.queryHeadsPerKvHead;
    for (uint32_t row = 0; row < data.params.row_bound; ++row) {
      uint32_t visible = data.params.committed_tokens + row + 1;
      std::vector<double> weights(visible);
      double maximum = -std::numeric_limits<double>::infinity();
      for (uint32_t token = 0; token < visible; ++token) {
        double score = 0.0;
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const float query = bfloat16ToFloat(
              queries[data.queryIndex(queryHead, row, dimension)]);
          const float key =
              quantized
                  ? loadKey(data, token, kvHead, dimension)
                  : bfloat16ToFloat(floatToBFloat16(
                        keyPattern(data.token(token), kvHead, dimension)));
          score += double(query) * key;
        }
        weights[token] = data.attends(row, token) && data.onPath(row, token)
                             ? score * 0.0625
                             : -std::numeric_limits<double>::infinity();
        maximum = std::max(maximum, weights[token]);
      }
      double denominator = 0.0;
      for (double &weight : weights) {
        weight = std::exp(weight - maximum);
        denominator += weight;
      }
      for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        double value = 0.0;
        for (uint32_t token = 0; token < visible; ++token) {
          const float storedValue =
              quantized
                  ? loadValue(data, token, kvHead, dimension)
                  : bfloat16ToFloat(floatToBFloat16(
                        valuePattern(data.token(token), kvHead, dimension)));
          value += weights[token] * storedValue;
        }
        output[data.queryIndex(queryHead, row, dimension)] =
            floatToBFloat16(float(value / denominator));
      }
    }
  }
  return output;
}

// The split of one geometry and its reduction.
struct Pipelines {
  std::string splitName;
  id<MTLComputePipelineState> split;
  id<MTLComputePipelineState> reduce;
};

Pipelines makePipelines(id<MTLDevice> device, id<MTLLibrary> library,
                        Shape shape) {
  Pipelines result;
  result.splitName = shape.split();
  result.split = makePipeline(device, library, result.splitName);
  result.reduce = makePipeline(device, library, shape.reduce());
  std::cout << "pipeline=" << result.splitName << " threadgroup_bytes=" << result.split.staticThreadgroupMemoryLength
            << (shaderValidationEnabled() ? " (instrumented by shader validation)" : "")
            << '\n';
  return result;
}

// The split, reduce and every scratch buffer of one production-sized
// dispatch: partials and statistics are the workspace production allocates
// for its lanes, which holds [lane's tile][KV head][part][split] at every
// split count.
struct Dispatch {
  std::vector<uint8_t> output;
  std::vector<uint8_t> partials;
  std::vector<uint8_t> statistics;
};

std::vector<uint8_t> copyOf(id<MTLBuffer> buffer) {
  const auto *bytes = static_cast<const uint8_t *>(buffer.contents);
  return {bytes, bytes + buffer.length};
}

// A lane of a dispatch: its rows, up to its case's row bound, and their
// places as a tree (Case::tree), none for rows that attend causally.
struct DispatchLane {
  uint32_t rows;
  std::vector<SplashTreeRow> tree;
};

// A dispatch of lanes `lanes`, each of the case's pages and queries after its
// history, as a decode step packs its lanes: lane l's rows the dispatch's from
// the lanes' before it on, its tiles (of the bound's rows) the dispatch's
// after theirs, and its staging the KV heads after theirs. Every tree row of
// no lane's row holds 0xFF, whose ancestors read as a chain's. The reduce
// writes the lanes' rows as the out-projection's input rows, gated by gates
// of +inf (identityGates). Through QSA masks each lane is the case's rows, of
// one tile, with a copy of its list and row masks.
Dispatch dispatch(id<MTLDevice> device, id<MTLCommandQueue> queue,
                  id<MTLComputePipelineState> split,
                  id<MTLComputePipelineState> reduce, const Case &data,
                  std::span<const DispatchLane> lanes) {
  const Shape shape = data.shape;
  const auto width = uint32_t(lanes.size());
  const uint32_t splits = data.params.split_count, bound = data.params.row_bound;
  const uint32_t laneTiles = (bound + kRows - 1) / kRows;
  std::vector<uint32_t> first;
  uint64_t rows = 0;
  for (const DispatchLane &lane : lanes) {
    require(lane.rows <= bound, "a dispatch's lane holds more rows than its bound");
    require(!data.maskWords || (laneTiles == 1 && lane.rows == bound),
            "a dispatch's QSA lanes are its case's rows, a tile each");
    first.push_back(uint32_t(rows));
    rows += lane.rows;
  }
  const uint64_t laneBytes = data.queries.length;
  id<MTLBuffer> queries = makeBuffer(device, width * laneBytes);
  id<MTLBuffer> packed = identityGates(device, shape, rows);
  id<MTLBuffer> output = makeBuffer(device, rows * shape.queryHeads() * kHeadDimension * sizeof(BFloat16Bits));
  // The workspace production allocates for the lanes' tiles, which holds
  // every split count's slots.
  const splash::ops::AttentionWorkspace workspace = splash::ops::PagedAttention::workspace(
      width * laneTiles * kAttentionMaximumSplits, shape.queryHeads(), shape.layout());
  id<MTLBuffer> partials = makeBuffer(device, workspace.partialsBytes);
  id<MTLBuffer> statistics = makeBuffer(device, workspace.statisticsBytes);
  for (uint32_t lane = 0; lane < width; ++lane) {
    std::memcpy(static_cast<uint8_t *>(queries.contents) + lane * laneBytes,
                data.queries.contents, laneBytes);
  }
  // The _h entries read half queries: the case's bf16 values, each exact in
  // half's range, against the same references.
  if (shape.halfQueries) {
    auto *values = static_cast<BFloat16Bits *>(queries.contents);
    for (uint64_t i = 0; i < queries.length / sizeof(BFloat16Bits); ++i) {
      const _Float16 value = static_cast<_Float16>(bfloat16ToFloat(values[i]));
      std::memcpy(values + i, &value, sizeof(value));
    }
  }
  // With QSA masks every lane attends through a copy of the case's list,
  // lane l's list kQsaJunkLists + l.
  const BlockLists blocks = data.maskWords ? qsaBlockLists(device, data, width) : BlockLists{partials, partials, 0};
  // Each lane's tree and, through QSA masks, a copy of the case's row masks,
  // at its rows, in buffers of a bound's rows a lane.
  id<MTLBuffer> tree = partials, rowMasks = partials;
  const bool trees = !lanes.empty() && !lanes[0].tree.empty();
  if (trees) {
    tree = makeBuffer(device, uint64_t{width} * bound * sizeof(SplashTreeRow));
    std::memset(tree.contents, 0xFF, tree.length);
    for (uint32_t lane = 0; lane < width; ++lane) {
      require(lanes[lane].tree.size() == lanes[lane].rows, "a lane's tree is not its rows'");
      std::copy(lanes[lane].tree.begin(), lanes[lane].tree.end(),
                static_cast<SplashTreeRow *>(tree.contents) + first[lane]);
    }
  }
  if (!data.rowMasks.empty()) {
    rowMasks = makeBuffer(device, uint64_t{width} * bound * sizeof(uint32_t));
    for (uint32_t lane = 0; lane < width; ++lane)
      std::copy(data.rowMasks.begin(), data.rowMasks.end(), static_cast<uint32_t *>(rowMasks.contents) + first[lane]);
  }
  SplashAttentionParams params{.sequences = width, .first_sequence = 0, .rows = uint32_t(rows),
                               .tiles = width * laneTiles, .trees = trees ? 1U : 0U, .last_rows = 0,
                               .kv = data.pool->layer(kLayer), .sequence = {}};
  // Each lane's rows of the command, after the lanes before it.
  id<MTLBuffer> layout = makeBuffer(device, width * sizeof(SplashLaneRows));
  for (uint32_t lane = 0; lane < width; ++lane) {
    static_cast<SplashLaneRows *>(layout.contents)[lane] = {first[lane], lanes[lane].rows};
    SplashAttentionSequence &sequence = params.sequence[lane];
    sequence = data.params;
    sequence.stage_row = lane * shape.kvHeads * kStride;
    sequence.first_tile = lane * laneTiles;
    sequence.first_slot = lane * laneTiles * shape.kvHeads * splits;
    sequence.block_stride = blocks.stride;
    sequence.block_list = blocks.stride ? kQsaJunkLists + lane : 0;
  }
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:split];
  [encoder setBuffer:queries offset:0 atIndex:0];
  [encoder setBuffer:partials offset:0 atIndex:1];
  [encoder setBuffer:statistics offset:0 atIndex:2];
  for (uint32_t index = 3; index < 7; ++index)
    [encoder setBuffer:data.pageTableBuffer offset:0 atIndex:index];
  // A split binds the tree rows, or without them another buffer of the
  // dispatch, then the layout, and a sparse split the lists, their counts and
  // the row masks, or without lists another buffer of the dispatch, of which
  // it reads nothing.
  [encoder setBuffer:tree offset:0 atIndex:7];
  [encoder setBuffer:layout offset:0 atIndex:8];
  NSUInteger paramsIndex = 9;
  if (shape.sparse()) {
    [encoder setBuffer:blocks.lists offset:0 atIndex:paramsIndex++];
    [encoder setBuffer:blocks.counts offset:0 atIndex:paramsIndex++];
    [encoder setBuffer:rowMasks offset:0 atIndex:paramsIndex++];
  }
  [encoder setBytes:&params length:sizeof(params) atIndex:paramsIndex];
  // The kernel reaches the extents only through the page entries.
  for (id<MTLBuffer> extent : data.extents)
    [encoder useResource:extent usage:MTLResourceUsageRead];
  // The sparse split's grid is one-dimensional (AttentionPlan::splitGroups).
  [encoder dispatchThreadgroups:shape.sparse() ? MTLSizeMake(shape.kvHeads * splits, width * laneTiles, 1)
                                               : MTLSizeMake(shape.kvHeads * shape.parts(), width * laneTiles, splits)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  // The plain reduce writes no table: the output rows take the table's and
  // its sums' bindings.
  [encoder setComputePipelineState:reduce];
  [encoder setBuffer:partials offset:0 atIndex:0];
  [encoder setBuffer:statistics offset:0 atIndex:1];
  [encoder setBuffer:packed offset:0 atIndex:2];
  for (NSUInteger index = 3; index < 6; ++index)
    [encoder setBuffer:output offset:0 atIndex:index];
  [encoder setBuffer:layout offset:0 atIndex:6];
  paramsIndex = 7;
  if (shape.sparse())
    [encoder setBuffer:blocks.counts offset:0 atIndex:paramsIndex++];
  [encoder setBytes:&params length:sizeof(params) atIndex:paramsIndex];
  [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, shape.fusedRows(), width * laneTiles)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  [encoder endEncoding];
  finish(command);
  return {copyOf(output), copyOf(partials), copyOf(statistics)};
}

// A dispatch of `width` lanes, each the case's rows and tree.
Dispatch dispatch(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> split,
                  id<MTLComputePipelineState> reduce, const Case &data, uint32_t width) {
  const std::vector<DispatchLane> lanes(width, DispatchLane{data.params.row_bound, data.tree});
  return dispatch(device, queue, split, reduce, data, lanes);
}

// The BF16 quality gate measures Q8 quantization fidelity on the synthetic
// pattern and is calibrated for short histories; long-history cases exercise
// the split geometry against the Q8 reference only.
void checkOutput(const Case &data, uint32_t width,
                 const std::vector<BFloat16Bits> &expectedQ8,
                 const std::vector<BFloat16Bits> &expectedBf16,
                 const std::vector<uint8_t> &outputBytes, bool qualityGate,
                 const std::string &label) {
  const Shape shape = data.shape;
  const uint32_t activeRows = data.params.row_bound;
  const auto *actual =
      reinterpret_cast<const BFloat16Bits *>(outputBytes.data());
  require(outputBytes.size() ==
              uint64_t{width} * activeRows * shape.queryHeads() * kHeadDimension * sizeof(BFloat16Bits),
          "attention output size departed from its lanes");
  double dot = 0.0;
  double actualSquared = 0.0;
  double expectedSquared = 0.0;
  float maximumAbsolute = 0.0f;
  double qualityDot = 0.0;
  double qualityActualSquared = 0.0;
  double qualityBf16Squared = 0.0;
  float qualityMaximumAbsolute = 0.0f;
  uint32_t top1Matches = 0;
  uint32_t top1Rows = 0;
  // With QSA bitmaps each row attends its own blocks: a row that lost some
  // must not hide in the cosine pooled over every row.
  std::vector<std::array<double, 3>> rowSums(uint64_t{width} * activeRows, {0.0, 0.0, 0.0});
  for (uint32_t lane = 0; lane < width; ++lane) {
    for (uint32_t head = 0; head < shape.queryHeads(); ++head) {
      for (uint32_t row = 0; row < activeRows; ++row) {
        std::array<double, 3> &rowSum = rowSums[uint64_t{lane} * activeRows + row];
        uint32_t actualTop = 0;
        float actualTopValue = -std::numeric_limits<float>::infinity();
        float bf16TopValue = -std::numeric_limits<float>::infinity();
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const uint64_t local = data.queryIndex(head, row, dimension);
          const uint64_t index = data.outputIndex(lane, head, row, dimension);
          const float observed = bfloat16ToFloat(actual[index]);
          const float q8Reference = bfloat16ToFloat(expectedQ8[local]);
          const float bf16Reference = bfloat16ToFloat(expectedBf16[local]);
          maximumAbsolute =
              std::max(maximumAbsolute, std::abs(observed - q8Reference));
          dot += double(observed) * q8Reference;
          actualSquared += double(observed) * observed;
          expectedSquared += double(q8Reference) * q8Reference;
          rowSum[0] += double(observed) * q8Reference;
          rowSum[1] += double(observed) * observed;
          rowSum[2] += double(q8Reference) * q8Reference;
          qualityMaximumAbsolute = std::max(
              qualityMaximumAbsolute, std::abs(observed - bf16Reference));
          qualityDot += double(observed) * bf16Reference;
          qualityActualSquared += double(observed) * observed;
          qualityBf16Squared += double(bf16Reference) * bf16Reference;
          if (observed > actualTopValue) {
            actualTopValue = observed;
            actualTop = dimension;
          }
          bf16TopValue = std::max(bf16TopValue, bf16Reference);
        }
        // The reference can hold an exact bf16 tie between dimensions; any
        // dimension at its maximum is the row's top-1.
        top1Matches += bfloat16ToFloat(expectedBf16[data.queryIndex(
                           head, row, actualTop)]) == bf16TopValue;
        ++top1Rows;
      }
    }
  }
  double cosine = dot / std::sqrt(actualSquared * expectedSquared);
  const double qualityCosine =
      qualityDot / std::sqrt(qualityActualSquared * qualityBf16Squared);
  const double top1Agreement = double(top1Matches) / top1Rows;
  std::cout << label << " lanes=" << width
            << " active_rows=" << activeRows
            << " committed=" << data.params.committed_tokens
            << " splits=" << data.params.split_count << " q8_cosine=" << cosine
            << " bf16_cosine=" << qualityCosine
            << " bf16_max_absolute=" << qualityMaximumAbsolute
            << " bf16_top1=" << top1Agreement << '\n';
  require(cosine > 0.9995,
          "production batch attention differs from its Q8 reference");
  if (data.maskWords)
    for (const auto &[rowDot, rowActual, rowExpected] : rowSums)
      require(rowDot / std::sqrt(rowActual * rowExpected) > 0.999,
              "a QSA row's attention differs from its Q8 reference");
  require(maximumAbsolute < 0.02f,
          "production batch attention exceeds its Q8 error bound");
  require(!qualityGate ||
              (qualityCosine > 0.9999 && qualityMaximumAbsolute < 0.01f &&
               top1Agreement >= 0.98),
          "Q8 production attention failed its BF16 quality gate");
}

// Independent submissions of the same graph, and of the same pages in another
// extent layout, must preserve all outputs and scratch results. Report the
// first differing element before failing.
void requireIdentical(const std::string &pipeline, const Case &data,
                      uint32_t width, const Dispatch &first,
                      const Dispatch &repeat, const char *repeated) {
  struct Buffer {
    const char *name;
    const std::vector<uint8_t> Dispatch::*bytes;
    size_t elementBytes;
  };
  constexpr std::array<Buffer, 3> buffers{
      {{"output", &Dispatch::output, sizeof(BFloat16Bits)},
       {"partials", &Dispatch::partials, sizeof(float)},
       {"statistics", &Dispatch::statistics, sizeof(float)}}};
  for (const Buffer &buffer : buffers) {
    const std::vector<uint8_t> &left = first.*buffer.bytes;
    const std::vector<uint8_t> &right = repeat.*buffer.bytes;
    const auto [leftAt, rightAt] =
        std::mismatch(left.begin(), left.end(), right.begin(), right.end());
    if (leftAt == left.end() && rightAt == right.end())
      continue;
    const size_t element = size_t(leftAt - left.begin()) / buffer.elementBytes;
    const auto value = [&](const std::vector<uint8_t> &bytes) {
      const size_t at = element * buffer.elementBytes;
      if (at + buffer.elementBytes > bytes.size())
        return std::string("absent");
      if (buffer.elementBytes == sizeof(float)) {
        float decoded = 0.0f;
        std::memcpy(&decoded, bytes.data() + at, sizeof decoded);
        return std::to_string(decoded);
      }
      BFloat16Bits decoded{};
      std::memcpy(&decoded, bytes.data() + at, sizeof decoded);
      return std::to_string(bfloat16ToFloat(decoded));
    };
    size_t differing = 0;
    for (size_t i = 0; i < std::min(left.size(), right.size()); ++i)
      differing += left[i] != right[i];
    differing += std::max(left.size(), right.size()) - std::min(left.size(), right.size());
    std::cout << repeated << " differs from the first submission: pipeline="
              << pipeline << " kv_heads=" << data.shape.kvHeads
              << " group=" << data.shape.queryHeadsPerKvHead
              << " committed=" << data.params.committed_tokens
              << " active_rows=" << data.params.row_bound
              << " lanes=" << width
              << " splits=" << data.params.split_count << " buffer=" << buffer.name
              << " first_differing_element=" << element << " first=" << value(left)
              << " repeat=" << value(right) << " differing_bytes=" << differing
              << " of " << left.size() << '\n';
    throw std::runtime_error(std::string(repeated) +
                             " is not bit-identical to the first submission");
  }
}

// A tree's rows through QSA lists against its paths run as chains, each row
// selected as the block indexer selects it (ops::BlockIndexer::addTreeSelect):
// tree row i at depth d, on the path p_0..p_d, attends the position blocks of
// a selection of its own; its bitmap names that selection's blocks below the
// anchor's block, and from there on the blocks of its row mask's slots, the
// path's rows p_k at positions in blocks it selects. The chain of a leaf's
// path holds at row k tree row p_k's query, its K/V at the slot of its depth
// and its selection, so its row k attends the same keys and values at the
// same positions as tree row p_k. Each tree row's output is its chain row's
// within rounding, every value of a head's row within a bf16 step of the
// row's largest: the path's rows sit at other slots than the chain's, and the
// tree's list (its eight rows' blocks) is longer than a chain's, so its
// entries fall into other steps and splits and the fp32 sums run in another
// order: bit for bit at 162 tokens; at 4094, 59% of the values differ, by at
// most 0.78 of a bf16 step of their row's largest (an M6). Rows 4 and 6
// branch off rows 0 and 1, so their packed predecessors are not their
// parents, and a sibling of a row in a block it selects is never among its
// keys.
void treeAgainstChains(id<MTLDevice> device, id<MTLCommandQueue> queue, const Pipelines &pipelines, Shape shape,
                       uint32_t committed, uint32_t splits, bool qualityGate) {
  constexpr uint32_t kBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;
  constexpr std::array<int, kRows> kParents{-1, 0, 1, 2, 0, 4, 1, 6};
  std::vector<SplashTreeRow> tree;
  for (uint32_t row = 0; row < kRows; ++row) {
    const int parent = kParents[row];
    tree.push_back(parent < 0 ? SplashTreeRow{1u, SPLASH_TREE_ROOT, 0}
                              : SplashTreeRow{tree[parent].ancestors | 1u << row, static_cast<uint16_t>(parent),
                                              static_cast<uint16_t>(tree[parent].depth + 1)});
  }
  const uint32_t anchor = committed, history = anchor / kBlockTokens;
  // Tree row t's selection of position blocks: past kQsaTopBlocks complete
  // blocks some of the history's by page and some of the block's by row, and
  // its tail; up to there every block up to its own.
  const auto selects = [&](uint32_t t, uint32_t block) {
    const uint32_t complete = (anchor + tree[t].depth + 1) / kBlockTokens;
    if (complete <= kQsaTopBlocks || block >= complete) return block <= complete;
    const uint32_t page = block / 8;
    return block >= history ? (block + t) % 2 == 0 : page % 5 == 0 || (page % 3 == 1 && block % 8 == (t + page) % 8);
  };
  // A case's bitmaps and row masks, row r taking tree row of[r]'s selection.
  const auto setMasks = [&](Case &data, const std::vector<uint32_t> &of) {
    data.maskWords = (data.params.committed_tokens + data.params.row_bound) / kBlockTokens / 32 + 2;
    data.mask.assign(uint64_t{data.params.row_bound} * data.maskWords, 0u);
    data.rowMasks.assign(data.params.row_bound, 0u);
    for (uint32_t r = 0; r < data.params.row_bound; ++r) {
      uint32_t *bits = data.mask.data() + uint64_t{r} * data.maskWords;
      for (uint32_t block = 0; block < history; ++block)
        if (selects(of[r], block)) bits[block / 32] |= 1u << (block % 32);
      uint32_t depth = 0;
      for (uint32_t rest = data.tree[r].ancestors; rest; rest &= rest - 1, ++depth) {
        if (!selects(of[r], (anchor + depth) / kBlockTokens)) continue;
        const uint32_t slot = std::countr_zero(rest), block = (anchor + slot) / kBlockTokens;
        data.rowMasks[r] |= 1u << slot;
        bits[block / 32] |= 1u << (block % 32);
      }
    }
  };
  const std::string name = pipelines.splitName + "_tree_chains";
  Case whole = makeCase(device, shape, committed, kRows, splits);
  whole.tree = tree;
  std::vector<uint32_t> own(kRows);
  for (uint32_t row = 0; row < kRows; ++row) own[row] = row;
  setMasks(whole, own);
  fill(whole);
  const Dispatch treeRun = dispatch(device, queue, pipelines.split, pipelines.reduce, whole, 1);
  checkOutput(whole, 1, cpuReference(whole, true), cpuReference(whole, false), treeRun.output, qualityGate, name);
  const auto *treeOut = reinterpret_cast<const BFloat16Bits *>(treeRun.output.data());
  uint64_t compared = 0, differing = 0;
  float worst = 0.0f;
  for (uint32_t leaf = 0; leaf < kRows; ++leaf) {
    if (std::find(kParents.begin(), kParents.end(), int(leaf)) != kParents.end()) continue;
    std::vector<uint32_t> path;
    for (uint32_t rest = tree[leaf].ancestors; rest; rest &= rest - 1) path.push_back(std::countr_zero(rest));
    const uint32_t rows = static_cast<uint32_t>(path.size());
    Case chain = makeCase(device, shape, committed, rows, splits);
    for (uint32_t row = 0; row < rows; ++row)
      chain.tree.push_back({(2u << row) - 1u, static_cast<uint16_t>(row ? row - 1 : SPLASH_TREE_ROOT),
                            static_cast<uint16_t>(row)});
    for (uint32_t slot = 0; slot < committed + rows; ++slot)
      chain.tokens.push_back(slot < anchor ? slot : anchor + path[slot - anchor]);
    chain.queryRows = path;
    setMasks(chain, path);
    fill(chain);
    const Dispatch chainRun = dispatch(device, queue, pipelines.split, pipelines.reduce, chain, 1);
    checkOutput(chain, 1, cpuReference(chain, true), cpuReference(chain, false), chainRun.output, qualityGate,
                name + "_leaf" + std::to_string(leaf));
    const auto *chainOut = reinterpret_cast<const BFloat16Bits *>(chainRun.output.data());
    for (uint32_t head = 0; head < shape.queryHeads(); ++head)
      for (uint32_t row = 0; row < rows; ++row) {
        float largest = 0.0f, difference = 0.0f;
        for (uint32_t dimension = 0; dimension < kHeadDimension; ++dimension) {
          const float a = bfloat16ToFloat(treeOut[whole.outputIndex(0, head, path[row], dimension)]);
          const float b = bfloat16ToFloat(chainOut[chain.outputIndex(0, head, row, dimension)]);
          largest = std::max({largest, std::abs(a), std::abs(b)});
          difference = std::max(difference, std::abs(a - b));
          differing += a != b;
          ++compared;
        }
        worst = std::max(worst, difference / (largest / 128.0f));
      }
  }
  std::cout << name << " committed=" << committed << " values=" << compared << " differing=" << differing
            << " worst_in_bf16_steps_of_the_row=" << worst << '\n';
  require(worst <= 1.0f, "a tree row's attention departs from its path's chain by more than its rounding");
}

// Lanes of 16, 5, 9 and 12 rows of a verify block's bound, two tiles a lane,
// packed by their first rows (0, 16, 21 and 30) after the same history, each
// a tree that branches: each lane's rows are a dispatch of it alone, bit for
// bit, whose rows match the reference of the lane's tree. The tree rows of no
// lane's row hold 0xFF, whose ancestors read as a chain's (dispatch), so that
// a lane that took its tree rows at another row than its first (lanes 2 and 3
// at lane x 16) attends another lane's tree or a chain.
void raggedTrees(id<MTLDevice> device, id<MTLCommandQueue> queue, const Pipelines &pipelines, Shape shape,
                 uint32_t committed, uint32_t splits, bool qualityGate) {
  constexpr uint32_t kBound = SPLASH_TARGET_VERIFY_ROWS;
  // A chain with a branch off row 2 at row 8, two children a row, a chain
  // with branches off rows 1 and 4, and two children a row again.
  const std::array<std::vector<int>, 4> kParents{
      std::vector<int>{-1, 0, 1, 2, 3, 4, 5, 6, 2, 8, 9, 10, 11, 12, 13, 14}, std::vector<int>{-1, 0, 0, 1, 1},
      std::vector<int>{-1, 0, 1, 2, 1, 4, 5, 4, 7}, std::vector<int>{-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5}};
  std::array<DispatchLane, 4> lanes;
  for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
    std::vector<SplashTreeRow> &tree = lanes[lane].tree;
    for (uint32_t row = 0; row < kParents[lane].size(); ++row) {
      const int parent = kParents[lane][row];
      tree.push_back(parent < 0 ? SplashTreeRow{1u, SPLASH_TREE_ROOT, 0}
                                : SplashTreeRow{tree[parent].ancestors | 1u << row, static_cast<uint16_t>(parent),
                                                static_cast<uint16_t>(tree[parent].depth + 1)});
    }
    lanes[lane].rows = uint32_t(tree.size());
  }
  Case data = makeCase(device, shape, committed, kBound, splits);
  fill(data);
  const std::string name = pipelines.splitName + "_ragged_trees";
  const Dispatch packed = dispatch(device, queue, pipelines.split, pipelines.reduce, data, lanes);
  const uint64_t rowBytes = uint64_t{shape.queryHeads()} * kHeadDimension * sizeof(BFloat16Bits);
  uint64_t first = 0;
  for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
    const Dispatch alone =
        dispatch(device, queue, pipelines.split, pipelines.reduce, data, std::span(lanes).subspan(lane, 1));
    Case reference = makeCase(device, shape, committed, lanes[lane].rows, splits);
    reference.tree = lanes[lane].tree;
    fill(reference);
    checkOutput(reference, 1, cpuReference(reference, true), cpuReference(reference, false), alone.output,
                qualityGate, name + "_lane" + std::to_string(lane));
    require(!std::memcmp(packed.output.data() + first * rowBytes, alone.output.data(), lanes[lane].rows * rowBytes),
            name + ": lane " + std::to_string(lane) + " of a packed step attends otherwise than alone");
    first += lanes[lane].rows;
  }
}


// Keep the CPU reference gates and verify that a second independent
// submission, and the same pages in one extent, produce identical output and
// scratch, of a dispatch of `width` lanes of the case's rows.
void runCase(id<MTLDevice> device, id<MTLCommandQueue> queue,
             const Pipelines &pipelines, Shape shape,
             uint32_t committed, uint32_t activeRows, uint32_t width,
             bool qualityGate = true, uint32_t splits = 32,
             bool qsa = false, bool tree = false) {
  require(width >= 1 && width <= 4 && activeRows >= 1 && activeRows <= kRows &&
              splits <= kAttentionMaximumSplits,
          "invalid attention case");
  Case data = makeCase(device, shape, committed, activeRows, splits);
  fill(data);
  if (qsa) setQsaMask(data);
  if (tree) {
    // Siblings at every depth past the anchor's: row r follows kParents[r].
    constexpr std::array<uint32_t, kRows> kParents{0, 0, 0, 1, 2, 3, 1, 4};
    for (uint32_t row = 0; row < activeRows; ++row) {
      const SplashTreeRow parent = row ? data.tree[kParents[row]] : SplashTreeRow{0, SPLASH_TREE_ROOT, 0};
      data.tree.push_back({(row ? parent.ancestors : 0u) | 1u << row,
                           static_cast<uint16_t>(row ? kParents[row] : SPLASH_TREE_ROOT),
                           static_cast<uint16_t>(row ? parent.depth + 1 : 0)});
    }
    // Through QSA masks, row masks that leave out some of a path's rows: a
    // third of the rows its parent, a third itself.
    if (qsa)
      for (uint32_t row = 0; row < activeRows; ++row)
        data.rowMasks.push_back(row % 3 == 1 ? ~(1u << kParents[row]) : row % 3 == 2 ? ~(1u << row) : ~0u);
  }
  const Case oneExtent = oneExtentCopy(device, data);
  const std::vector<BFloat16Bits> expectedQ8 = cpuReference(data, true);
  const std::vector<BFloat16Bits> expectedBf16 = cpuReference(data, false);
  const std::string name = pipelines.splitName + (qsa ? "_qsa" : "") + (tree ? "_tree" : "");
  const auto run = [&](const Case &c) { return dispatch(device, queue, pipelines.split, pipelines.reduce, c, width); };
  const Dispatch first = run(data);
  checkOutput(data, width, expectedQ8, expectedBf16, first.output, qualityGate, name);
  const Dispatch repeat = run(data);
  checkOutput(data, width, expectedQ8, expectedBf16, repeat.output, qualityGate,
              name + "_repeat");
  requireIdentical(name, data, width, first, repeat, "attention repeat");
  requireIdentical(name, data, width, first, run(oneExtent), "attention over one extent");
}

// Isolate the merge from QK/PV: large differences in maxima, cancellation,
// ragged split counts and inactive rows exercise the reduction of a tile of
// eight active rows or fewer, which writes the active rows alone.
void checkReduce(id<MTLDevice> device, id<MTLCommandQueue> queue,
                 id<MTLLibrary> library, Shape shape, uint32_t splits,
                 uint32_t activeRows) {
  const uint32_t m = shape.fusedRows(), d = kHeadDimension;
  auto partials = makeBuffer(device, uint64_t{shape.kvHeads} * splits * m * d * 4);
  auto stats = makeBuffer(device, uint64_t{shape.kvHeads} * splits * m * 2 * 4);
  auto *p = static_cast<float *>(partials.contents);
  auto *t = static_cast<float *>(stats.contents);
  for (uint32_t head = 0; head < shape.kvHeads; ++head)
    for (uint32_t split = 0; split < splits; ++split)
      for (uint32_t row = 0; row < m; ++row) {
        const uint64_t index = shape.partialRow(head, splits, split, row);
        const bool active = row / shape.queryHeadsPerKvHead < activeRows;
        // Alternate underflow-scale differences and fractional exp weights.
        const float spacing = row % 2 ? 0.25f : 500.0f;
        t[index * 2] = active ? float(int(split % 5) - 2) * spacing : -INFINITY;
        t[index * 2 + 1] = active ? 1.0f + float(split % 7) : 0.0f;
        for (uint32_t dim = 0; dim < d; ++dim)
          p[index * d + dim] = float(int((split * 71 + dim * 37 + row * 13) % 257) - 128) * 0.125f;
      }
  // The merge of the partials into a poisoned output of a tile's rows, gated
  // by gates of +inf; a sparse reduce without lists binds another buffer of
  // the dispatch for their counts, of which it reads nothing.
  id<MTLBuffer> layout = makeBuffer(device, sizeof(SplashLaneRows));
  *static_cast<SplashLaneRows *>(layout.contents) = {0, activeRows};
  id<MTLBuffer> packed = identityGates(device, shape, kRows);
  const auto reduce = [&](const SplashAttentionParams &params) {
    id<MTLBuffer> output = makeBuffer(device, uint64_t{kRows} * shape.queryHeads() * d * 2);
    std::memset(output.contents, 0xa5, output.length);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:makePipeline(device, library, shape.reduce())];
    [encoder setBuffer:partials offset:0 atIndex:0];
    [encoder setBuffer:stats offset:0 atIndex:1];
    [encoder setBuffer:packed offset:0 atIndex:2];
    for (NSUInteger index = 3; index < 6; ++index)
      [encoder setBuffer:output offset:0 atIndex:index];
    [encoder setBuffer:layout offset:0 atIndex:6];
    NSUInteger paramsIndex = 7;
    if (shape.sparse())
      [encoder setBuffer:partials offset:0 atIndex:paramsIndex++];
    [encoder setBytes:&params length:sizeof(params) atIndex:paramsIndex];
    [encoder dispatchThreadgroups:MTLSizeMake(shape.kvHeads, m, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    finish(command);
    return output;
  };
  // The fp64 merge of each active row, and the poison kept in the inactive
  // rows: fused row `row` of KV head `head` is query head head x group +
  // row % group of output row row / group.
  const auto check = [&](id<MTLBuffer> output) {
    const auto *out = static_cast<const BFloat16Bits *>(output.contents);
    for (uint32_t head = 0; head < shape.kvHeads; ++head)
      for (uint32_t row = 0; row < m; ++row) {
        const uint64_t offset = (uint64_t{row / shape.queryHeadsPerKvHead} * shape.queryHeads() +
                                 head * shape.queryHeadsPerKvHead + row % shape.queryHeadsPerKvHead) *
                                d;
        const bool active = row / shape.queryHeadsPerKvHead < activeRows;
        if (!active) {
          for (uint32_t dim = 0; dim < d; ++dim)
            require(out[offset + dim] == 0xa5a5, "attention merge wrote an inactive row");
          continue;
        }
        double maximum = -INFINITY;
        for (uint32_t split = 0; split < splits; ++split)
          maximum = std::max(maximum, double(t[shape.partialRow(head, splits, split, row) * 2]));
        for (uint32_t dim = 0; dim < d; ++dim) {
          double numerator = 0, denominator = 0;
          for (uint32_t split = 0; split < splits; ++split) {
            const uint64_t index = shape.partialRow(head, splits, split, row);
            const double weight = std::exp(double(t[index * 2]) - maximum);
            numerator += weight * p[index * d + dim];
            denominator += weight * t[index * 2 + 1];
          }
          const double expected = numerator / denominator;
          const double actual = bfloat16ToFloat(out[offset + dim]);
          // Final bf16 rounding plus fp32 reduction; cancellation uses an
          // absolute floor so an exact zero reference remains a useful check.
          require(std::isfinite(actual) &&
                      std::abs(actual - expected) <= std::abs(expected) * 0.004 + 0.000002,
                  "attention merge differs from fp64 reference");
        }
      }
  };
  // A history every split takes a share of: a page each of the page loop's,
  // and of a sparse reduce without lists the least blocks a split takes.
  const uint32_t splitTokens = shape.sparse() ? SPLASH_SPARSE_SPLIT_BLOCKS * SPLASH_ATTENTION_BLOCK_TOKENS : 32;
  const uint32_t committed = splits * splitTokens - activeRows;
  const uint32_t pages = splits * splitTokens / 32;
  SplashAttentionParams params{.sequences = 1, .first_sequence = 0, .rows = activeRows, .tiles = 1, .trees = 0,
                               .last_rows = 0, .kv = {1, 0}, .sequence = {}};
  params.sequence[0] = {.committed_tokens = committed,
                        .row_bound = activeRows,
                        .layout_row = 0,
                        .stage_row = 0,
                        .chunk_stride = kStride,
                        .page_table_entries = pages,
                        .split_count = splits,
                        .first_tile = 0,
                        .first_slot = 0,
                        .block_stride = 0,
                        .block_list = 0};
  check(reduce(params));
}

void run(const char *libraryPath) {
  testContract();
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device)
    throw std::runtime_error("Metal device unavailable");
  NSError *error = nil;
  NSURL *url =
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:libraryPath]];
  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  if (!library)
    throw std::runtime_error(error.localizedDescription.UTF8String);
  id<MTLCommandQueue> queue = [device newCommandQueue];
  for (const Shape shape : kShapes) {
    // The INT8-query and half-query entries share their geometry's reduce.
    if (!shape.int8Queries && !shape.halfQueries)
      for (uint32_t splits : {1U, 3U, 7U, 32U, 65U, 128U}) {
        checkReduce(device, queue, library, shape, splits, kRows);
        checkReduce(device, queue, library, shape, splits, 1);
      }
    const Pipelines pipelines = makePipelines(device, library, shape);
    for (uint32_t width = 1; width <= 4; ++width)
      runCase(device, queue, pipelines, shape, 127, 8, width);
    runCase(device, queue, pipelines, shape, 0, 8, 4);
    for (uint32_t activeRows = 1; activeRows < kRows; ++activeRows)
      runCase(device, queue, pipelines, shape, 127, activeRows, 1);
    runCase(device, queue, pipelines, shape, 129, 8, 4);
    runCase(device, queue, pipelines, shape, 421, 3, 1);
    runCase(device, queue, pipelines, shape, 421, 3, 4);
    // Several pages per split, a partially filled last page, and a history
    // that leaves some splits empty.
    runCase(device, queue, pipelines, shape, 1'100, 8, 2, false);
    runCase(device, queue, pipelines, shape, 4'093, 5, 1, false);
    // History-scaled partitions: every split count the plan can produce must
    // agree with the reference through the same store/split/reduce path,
    // including counts that leave a ragged last split.
    runCase(device, queue, pipelines, shape, 4'093, 8, 1, false, 64);
    runCase(device, queue, pipelines, shape, 4'093, 8, 3, false, 65);
    runCase(device, queue, pipelines, shape, 4'093, 8, 2, false,
            kAttentionMaximumSplits);
    runCase(device, queue, pipelines, shape, 1'100, 8, 1, false, 1);
    runCase(device, queue, pipelines, shape, 8'192, 5, 1, false);
    runCase(device, queue, pipelines, shape, 32'768, 7, 1, false);
    runCase(device, queue, pipelines, shape, 32'768, 8, 2, false, 65);
    // Tree rows: each sees only its path's rows of the block, over a short
    // history and a long one, a lane alone and four.
    runCase(device, queue, pipelines, shape, 0, 8, 1, true, 32, false, true);
    runCase(device, queue, pipelines, shape, 127, 8, 4, true, 32, false, true);
    runCase(device, queue, pipelines, shape, 4'093, 7, 2, false, 65, false, true);
    // Tree rows of lanes of 16, 5, 9 and 12 rows packed by their first rows,
    // over no history, one past a page, and a long one over 65 splits of two
    // pages each, at which a 16-row lane's second tile reads a page more
    // than its first (129 and 128), so that the split and the reduce of each
    // tile partition its own pages (Flint's geometry alone had 16-row trees
    // over long histories, attention_variants_metal_test).
    raggedTrees(device, queue, pipelines, shape, 0, 32, true);
    raggedTrees(device, queue, pipelines, shape, 127, 32, true);
    raggedTrees(device, queue, pipelines, shape, 4'083, 65, false);
    if (shape.queryHeadsPerKvHead == 12) {
      // Tree rows through QSA lists and row masks, causal and sparse, and
      // against their paths as chains.
      runCase(device, queue, pipelines, shape, 160, 8, 2, true, 32, true, true);
      runCase(device, queue, pipelines, shape, 4'093, 8, 3, false, 65, true, true);
      treeAgainstChains(device, queue, pipelines, shape, 162, 32, true);
      treeAgainstChains(device, queue, pipelines, shape, 4'094, 65, false);
      // QSA masks: a tile attends only the blocks its list names. Causal
      // rows only; tiles crossing from causal to sparse rows (complete
      // blocks 40 to 42, and only the last row past 40); sparse rows over
      // several units per split, ragged and empty splits.
      runCase(device, queue, pipelines, shape, 127, 8, 1, true, 32, true);
      runCase(device, queue, pipelines, shape, 160, 8, 2, true, 32, true);
      runCase(device, queue, pipelines, shape, 158, 6, 1, true, 32, true);
      runCase(device, queue, pipelines, shape, 1'100, 8, 2, false, 32, true);
      runCase(device, queue, pipelines, shape, 1'100, 8, 1, false, 1, true);
      runCase(device, queue, pipelines, shape, 4'093, 5, 1, false, 32, true);
      runCase(device, queue, pipelines, shape, 4'093, 8, 3, false, 65, true);
      runCase(device, queue, pipelines, shape, 32'768, 7, 1, false, 32, true);
      runCase(device, queue, pipelines, shape, 32'768, 8, 2, false, 65, true);
      // At a long history the sparse kernels agree with the reference
      // through a list, whose count sets a split's share, and without one,
      // over every block as the page loop attends them (past 64K tokens of
      // full attention the bf16 probabilities' rounding alone moves the
      // output past the cosine bound).
      runCase(device, queue, pipelines, shape, 131'072, 8, 2, false, kAttentionMaximumSplits, true);
      runCase(device, queue, pipelines, shape, 65'536, 8, 1, false, kAttentionMaximumSplits);
    }
  }
  std::cout << "q8_flash_attention_metal_test: ok\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      if (argc != 2)
        throw std::runtime_error(
            "usage: q8_flash_attention_metal_test <metallib>");
      run(argv[1]);
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "q8_flash_attention_metal_test: " << error.what() << '\n';
      return 1;
    }
  }
}

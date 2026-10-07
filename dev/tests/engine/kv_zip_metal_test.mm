// ZBF16 KV pages (metal/abi/KvZip.h) against a host reference codec and the
// BF16 kernels:
//   - the prefill and verify stores write the bytes the host encoder writes,
//     and every stored row decodes to the bits it came from;
//   - a verify store over rejected speculative rows leaves the committed
//     rows intact and decodes the new ones;
//   - prefill and verify attention over ZBF16 pages give the BF16 kernels'
//     output over the same values, bit for bit;
//   - a slab whose overflow does not fit is counted, not written past.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "ops/PagedAttention.hpp"
#include "tuning/HostKvExtents.hpp"
#include "tuning/KvZipHost.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace splash;
using ops::tuning::HostKvExtents;
using ops::tuning::bf16ToFloat;
using ops::tuning::floatToBf16;
using test::require;
using namespace ops::tuning::kvzip;

constexpr uint32_t kLayer = 1;

// ---------------------------------------------------------------------------
// KV-like test values: per (head, dimension) a Gaussian of its own scale, a
// few outliers and exact zeros, so values escape their window. Harsh values
// have ten times the outliers.

struct Values final {
  // [tensor][head] token-major rows of the whole sequence.
  std::array<std::vector<std::vector<uint16_t>>, 2> rows;
  uint32_t tokens = 0;
};

// The scales are the "model's": the same for every seed, as one model's KV
// serves every request.
Values makeValues(uint32_t kvHeads, uint32_t tokens, uint32_t seed, bool harsh = false) {
  std::mt19937 random(seed), model(1);
  std::normal_distribution<float> normal;
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  const float outliers = harsh ? 0.008f : 0.0008f, zeros = harsh ? 0.002f : 0.0002f;
  Values values;
  values.tokens = tokens;
  for (uint32_t tensor = 0; tensor < 2; ++tensor) {
    values.rows[tensor].resize(kvHeads);
    for (uint32_t head = 0; head < kvHeads; ++head) {
      std::array<float, kDims> scale{};
      for (float &value : scale) value = std::exp2(-6.0f + 8.0f * uniform(model));
      auto &rows = values.rows[tensor][head];
      rows.resize(uint64_t{tokens} * kDims);
      for (uint32_t token = 0; token < tokens; ++token)
        for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
          float value = normal(random) * scale[dimension];
          const float pick = uniform(random);
          if (pick < outliers) value *= 4096.0f;
          else if (pick < outliers + zeros) value = 0.0f;
          rows[uint64_t{token} * kDims + dimension] = floatToBf16(value);
        }
    }
  }
  return values;
}

// Best 8- and 16-binade windows of each (tensor, head, dimension), as
// dev/tools/kvzip_calibrate.py fits them.
std::array<std::vector<HeadBases>, 2> calibrate(const Values &values) {
  std::array<std::vector<HeadBases>, 2> result;
  for (uint32_t tensor = 0; tensor < 2; ++tensor)
    for (const auto &rows : values.rows[tensor]) {
      std::vector<std::array<uint32_t, 256>> histograms(kDims);
      for (uint32_t token = 0; token < values.tokens; ++token)
        for (uint32_t dimension = 0; dimension < kDims; ++dimension)
          ++histograms[dimension][(rows[uint64_t{token} * kDims + dimension] >> 7) & 0xFF];
      result[tensor].push_back(windows(histograms));
    }
  return result;
}

// ---------------------------------------------------------------------------
// Pages of `layers` attention layers in BF16 or ZBF16 (the layer under test
// is kLayer), a sequence's page table and its staged rows.

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  auto buffer = test::sharedBuffer(backend, std::max<uint64_t>(bytes, 16));
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
  return buffer;
}

struct Pool final {
  kv::Layout layout;
  HostKvExtents extents;
  SplashKvLayer layer;
  metal::MetalBuffer codec;
  std::array<std::vector<HeadBases>, 2> bases;
  uint32_t tested = kLayer;

  ZipPage page(uint32_t id) const { return {extents, layout.kvHeads, id, tested}; }
};

// The layer under test is `tested`: kLayer, past layer 0's region, unless
// the case needs layer 0's own store.
Pool makePool(metal::MetalBackend &backend, kv::Format format, uint32_t kvHeads, uint32_t pages,
              const std::array<std::vector<HeadBases>, 2> &bases, uint32_t layers = 4,
              uint32_t tested = kLayer) {
  kv::Layout layout{layers, kvHeads, kDims, format};
  const auto spread = HostKvExtents::spread(pages + 2);
  Pool pool{layout, HostKvExtents(backend, layout, spread.extentPages, spread.extents), {}, {}, bases, tested};
  pool.layer = pool.extents.layer(tested);
  if (format == kv::Format::ZipBFloat16) {
    pool.codec = allocate(backend, splash_kvzip_codec_bytes(layout.attentionLayers, kvHeads));
    auto *bytes = static_cast<uint8_t *>(pool.codec.contents());
    SplashKvZipHeader header{};
    header.layers = layers;
    header.kv_heads = kvHeads;
    std::memcpy(bytes, &header, sizeof(header));
    // Other layers' bases differ from the tested layer's, so a kernel that
    // reads the wrong layer's decodes wrong bits.
    for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer)
      for (uint32_t tensor = 0; tensor < 2; ++tensor)
        for (uint32_t head = 0; head < kvHeads; ++head) {
          uint8_t *at = bytes + splash_kvzip_base_offset(kvHeads, layer, tensor, head);
          for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
            at[dimension] = layer == tested ? bases[tensor][head].base3[dimension] : 1;
            at[kDims + dimension] = layer == tested ? bases[tensor][head].base4[dimension] : 2;
          }
        }
  }
  return pool;
}

// Writes tokens [0, tokens) of a sequence into its pages, as committed
// history: BF16 as it is, ZBF16 through the host store. Stale bytes fill
// the rest of every page, as a reused
// page holds: any bytes for ZBF16, finite values for BF16 (a BF16 page's
// stale rows are earlier KV, and its kernels weigh them by exact zeros).
void writeHistory(Pool &pool, const std::vector<uint32_t> &pages, const Values &values, uint32_t tokens) {
  for (uint32_t page = 0; page < pages.size(); ++page)
    for (uint32_t layer = 0; layer < pool.layout.attentionLayers; ++layer)
      for (uint32_t tensor = 0; tensor < 4; ++tensor) {
        uint8_t *bytes = pool.extents.slab<uint8_t>(layer, tensor, pages[page]);
        const uint64_t count = tensor % 2 ? pool.layout.scaleBytesPerLayerPage() : pool.layout.dataBytesPerLayerPage();
        for (uint64_t byte = 0; byte < count; ++byte) bytes[byte] = uint8_t(byte * 131 + page * 7 + 0x5a);
        if (pool.layout.format == kv::Format::BFloat16)
          for (uint64_t byte = 1; byte < count; byte += 2) bytes[byte] = uint8_t(0x3C | (bytes[byte] & 0x80));
      }
  for (uint32_t tensor = 0; tensor < 2; ++tensor)
    for (uint32_t head = 0; head < pool.layout.kvHeads; ++head) {
      const auto &rows = values.rows[tensor][head];
      for (uint32_t page = 0; page * kRows < tokens; ++page) {
        const uint32_t count = std::min(kRows, tokens - page * kRows);
        if (pool.layout.format == kv::Format::BFloat16) {
          auto *slab = pool.extents.slab<uint16_t>(pool.tested, tensor ? SPLASH_KV_VALUES : SPLASH_KV_KEYS,
                                                   pages[page]);
          for (uint32_t row = 0; row < count; ++row)
            for (uint32_t dimension = 0; dimension < kDims; ++dimension)
              slab[tensor ? splash_kv_value_element(head, row, dimension)
                          : splash_kv_key_element(head, row, dimension)] =
                  rows[uint64_t{page * kRows + row} * kDims + dimension];
        } else {
          require(!storeRows(pool.page(pages[page]), tensor, head, rows.data() + uint64_t{page} * kRows * kDims,
                             0, count, pool.bases[tensor][head]),
                  "test history dropped escapes");
        }
      }
    }
}

// Stages rows [first, first + count) of the values as a chunk of `stride`
// rows: keys [head][row][dimension], values [head][dimension][row].
void stage(const Values &values, uint32_t kvHeads, uint32_t first, uint32_t count, uint32_t stride,
           uint16_t *keys, uint16_t *staged) {
  for (uint32_t head = 0; head < kvHeads; ++head)
    for (uint32_t row = 0; row < count; ++row)
      for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
        const uint64_t source = uint64_t{first + row} * kDims + dimension;
        keys[(uint64_t{head} * stride + row) * kDims + dimension] = values.rows[0][head][source];
        staged[(uint64_t{head} * kDims + dimension) * stride + row] = values.rows[1][head][source];
      }
}

// Every committed row of a ZBF16 sequence decodes to its value bits, and the
// store wrote what the host store writes: sign/mantissa bytes, codes, and
// the escapes in row order. Returns the escapes stored.
uint32_t requireDecodes(const Pool &pool, const std::vector<uint32_t> &pages, const Values &values,
                        uint32_t tokens, const std::string &what) {
  uint32_t escapes = 0;
  for (uint32_t tensor = 0; tensor < 2; ++tensor)
    for (uint32_t head = 0; head < pool.layout.kvHeads; ++head)
      for (uint32_t page = 0; page * kRows < tokens; ++page) {
        const uint32_t count = std::min(kRows, tokens - page * kRows);
        const ZipPage zip = pool.page(pages[page]);
        const uint16_t *expected = values.rows[tensor][head].data() + uint64_t{page} * kRows * kDims;
        const auto decoded = decodeSlab(zip, tensor, head, count, pool.bases[tensor][head]);
        const std::string where = what + " (tensor " + std::to_string(tensor) + ", head " +
                                  std::to_string(head) + ", page " + std::to_string(page) + ")";
        require(std::equal(decoded.begin(), decoded.end(), expected),
                where + ": a stored row does not decode to its bits");
        const uint8_t *actualData = zip.data(tensor, head);
        const uint32_t *actualEscapes = zip.escapes(tensor, head);
        uint32_t stored = 0;
        for (uint32_t row = 0; row < count; ++row) {
          const RowCode code = encodeRow(expected + row * kDims, row, pool.bases[tensor][head]);
          for (uint32_t pair = 0; pair < kDims / 2; ++pair)
            require(actualData[SPLASH_KVZIP_SM_BYTES + row * SPLASH_KVZIP_CODE_BYTES_PER_ROW + pair] ==
                        uint8_t(code.code[2 * pair] | (code.code[2 * pair + 1] << 4)),
                    where + ": codes differ from the host store's");
          for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
            const uint32_t bits = expected[row * kDims + dimension];
            require(actualData[row * kDims + dimension] == uint8_t(((bits >> 8) & 0x80) | (bits & 0x7F)),
                    where + ": sign/mantissa bytes differ from the host store's");
          }
          for (const uint32_t escape : code.escapes) {
            require(stored < SPLASH_KVZIP_ESCAPES && actualEscapes[stored] == escape,
                    where + ": escape " + std::to_string(stored) + " differs from the host store's");
            ++stored;
          }
        }
        require(zip.count(tensor, head) == stored, where + ": the escape count differs from the host store's");
        escapes += stored;
      }
  return escapes;
}

uint32_t overflowSlabs(const Pool &pool) {
  uint32_t count = 0;
  std::memcpy(&count, pool.codec.contents(), sizeof(count));
  return count;
}

// ZBF16 prefill's codec and BF16 scratch (ops::KvZipPrefill), the scratch
// shared by every case of one KV head count as the runtime's arena holds one.
ops::KvZipPrefill zipPrefill(metal::MetalBackend &backend, const Pool &pool) {
  static std::array<ops::KvZipPrefill, 5> scratches;
  ops::KvZipPrefill &shared = scratches[pool.layout.kvHeads];
  if (!shared.scratch) {
    const kv::Layout scratch = kv::zipScratchLayout(pool.layout);
    shared.scratch = test::sharedBuffer(backend, uint64_t{kv::zipScratchPages()} * scratch.bytesPerModelPage());
    shared.scratchTable = test::sharedBuffer(backend, uint64_t{kv::zipScratchPages()} * sizeof(SplashKvPage));
    auto *entries = static_cast<SplashKvPage *>(shared.scratchTable.contents());
    for (uint32_t page = 0; page < kv::zipScratchPages(); ++page)
      entries[page] = splash_kv_page_entry(shared.scratch.gpuAddress(), page);
  }
  // Stale bytes, as a scratch an earlier chunk filled holds.
  std::memset(shared.scratch.contents(), 0x5a, shared.scratch.sizeBytes());
  return {pool.codec, shared.scratch, shared.scratchTable};
}

std::vector<uint32_t> distinctPages(const Pool &pool, uint32_t count, uint32_t seed) {
  return HostKvExtents::mixedPages({pool.extents.extentPages(), pool.extents.extentCount()}, count, seed);
}

std::string compare(const std::vector<uint16_t> &left, const std::vector<uint16_t> &right) {
  uint32_t different = 0;
  float maximum = 0;
  for (size_t index = 0; index < left.size(); ++index)
    if (left[index] != right[index]) {
      ++different;
      maximum = std::max(maximum, std::abs(bf16ToFloat(left[index]) - bf16ToFloat(right[index])));
    }
  return different ? std::to_string(different) + " outputs differ (largest " + std::to_string(maximum) + ")"
                   : std::string{};
}

// ---------------------------------------------------------------------------

// Prefill store + attention of `rows` rows after `history` committed tokens,
// in BF16 and in ZBF16 over the same values. The two outputs must be equal.
void checkPrefill(metal::MetalBackend &backend, uint32_t queryHeads, uint32_t history, uint32_t rows,
                  bool harsh = false) {
  const uint32_t kvHeads = queryHeads == 24 ? 4 : 2;
  const uint32_t tokens = history + rows, pageCount = (tokens + 31) / 32;
  const Values values = makeValues(kvHeads, tokens, 7 + history + rows, harsh);
  const auto bases = calibrate(makeValues(kvHeads, 4096, 999, harsh));
  const uint32_t stride = (rows + 31) / 32 * 32;
  auto keys = allocate(backend, uint64_t{kvHeads} * stride * kDims * 2);
  auto staged = allocate(backend, keys.sizeBytes());
  stage(values, kvHeads, history, rows, stride, static_cast<uint16_t *>(keys.contents()),
        static_cast<uint16_t *>(staged.contents()));
  auto queries = allocate(backend, uint64_t{queryHeads} * stride * kDims * 2);
  std::mt19937 random(history * 31 + rows);
  std::normal_distribution<float> normal;
  for (uint64_t index = 0; index < uint64_t{queryHeads} * rows * kDims; ++index)
    static_cast<uint16_t *>(queries.contents())[index] = floatToBf16(normal(random));

  std::array<std::vector<uint16_t>, 2> outputs;
  uint32_t escapes = 0;
  for (const kv::Format format : {kv::Format::BFloat16, kv::Format::ZipBFloat16}) {
    Pool pool = makePool(backend, format, kvHeads, pageCount, bases);
    const auto pages = distinctPages(pool, pageCount, tokens);
    writeHistory(pool, pages, values, history);
    auto table = allocate(backend, uint64_t{pageCount} * sizeof(SplashKvPage));
    pool.extents.writeTable(pages, table.contents());
    const auto chunk = ops::PagedAttention::prefillParams(history, rows, stride, pageCount);
    const auto plan = ops::PagedAttention::prefillPlan(rows, queryHeads, pool.layout);
    auto output = allocate(backend, queries.sizeBytes());
    auto partials = allocate(backend, plan.workspace.partialsBytes);
    auto statistics = allocate(backend, plan.workspace.statisticsBytes);
    const ops::KvZipPrefill zip =
        format == kv::Format::ZipBFloat16 ? zipPrefill(backend, pool) : ops::KvZipPrefill{};
    metal::CommandGraph graph;
    ops::PagedAttention::addPrefillStore(graph, pool.layer, keys, staged, table, chunk, pool.layout, zip);
    require(format != kv::Format::ZipBFloat16 ||
                graph.dispatches().size() == (history ? 3U : 2U),
            "ZBF16 prefill store should encode store, expansion (with history) and the scratch's store");
    ops::PagedAttention::addPrefill(graph, pool.layer, queries, output, partials, statistics, table, chunk, plan,
                                    zip);
    require(std::string(plan.splitPipeline).find("bf16_split") != std::string::npos,
            "prefill should attend through the BF16 split");
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    if (format == kv::Format::ZipBFloat16) {
      escapes = requireDecodes(pool, pages, values, tokens, "prefill store");
      require(overflowSlabs(pool) == 0, "prefill store counted an overflow");
    }
    const auto *bits = static_cast<const uint16_t *>(output.contents());
    outputs[format == kv::Format::ZipBFloat16] = {bits, bits + uint64_t{queryHeads} * rows * kDims};
  }
  const std::string difference = compare(outputs[0], outputs[1]);
  require(difference.empty(), "ZBF16 prefill attention differs from BF16: " + difference);
  require(escapes, "no value escaped its window");
  std::cout << "zbf16 prefill: q=" << queryHeads << " history=" << history << " rows=" << rows
            << (harsh ? " harsh" : "") << " escapes=" << escapes << " PASS (bit-equal to bf16)\n";
}

// Verify of `lanes` lanes at the given histories, twice: the second command
// starts three rows into the first one's eight, as after a partial accept,
// over new rows. Attention must equal BF16's at both, and every committed
// row must decode.
void checkVerify(metal::MetalBackend &backend, uint32_t queryHeads, std::array<uint32_t, 4> histories,
                 uint32_t lanes) {
  const uint32_t kvHeads = queryHeads == 24 ? 4 : 2;
  constexpr uint32_t stride = kv::kVerifyChunkStride, rows = kv::kVerifyRows, accepted = 3;
  std::array<Values, 4> first, second;
  std::array<uint32_t, 4> pageCounts{};
  uint32_t allPages = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t tokens = histories[lane] + accepted + rows;
    first[lane] = makeValues(kvHeads, tokens, 101 + lane);
    second[lane] = first[lane];
    // The second command's rows differ from the first's past the accepted ones.
    const Values other = makeValues(kvHeads, tokens, 211 + lane);
    for (uint32_t tensor = 0; tensor < 2; ++tensor)
      for (uint32_t head = 0; head < kvHeads; ++head)
        std::copy(other.rows[tensor][head].begin() + uint64_t{histories[lane] + accepted} * kDims,
                  other.rows[tensor][head].end(),
                  second[lane].rows[tensor][head].begin() + uint64_t{histories[lane] + accepted} * kDims);
    pageCounts[lane] = (tokens + 31) / 32;
    allPages += pageCounts[lane];
  }
  // One calibration serves every lane, as one model's bases serve every
  // request.
  const auto bases = calibrate(makeValues(kvHeads, 4096, 999));
  std::array<std::array<std::vector<uint16_t>, 2>, 2> outputs;
  for (const kv::Format format : {kv::Format::BFloat16, kv::Format::ZipBFloat16}) {
    // The layer under test is layer 0.
    Pool pool = makePool(backend, format, kvHeads, allPages, bases, 4, 0);
    Pool &host = pool;
    const auto ids = distinctPages(pool, allPages, allPages + lanes);
    std::array<std::vector<uint32_t>, 4> pages;
    std::array<metal::MetalBuffer, 4> tables;
    uint32_t next = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      pages[lane].assign(ids.begin() + next, ids.begin() + next + pageCounts[lane]);
      next += pageCounts[lane];
      tables[lane] = allocate(backend, uint64_t{pageCounts[lane]} * sizeof(SplashKvPage));
      pool.extents.writeTable(pages[lane], tables[lane].contents());
      writeHistory(host, pages[lane], first[lane], histories[lane]);
    }
    for (uint32_t lane = lanes; lane < 4; ++lane) tables[lane] = tables[0];
    auto keys = allocate(backend, uint64_t{lanes} * kvHeads * stride * kDims * 2);
    auto staged = allocate(backend, keys.sizeBytes());
    auto queries = allocate(backend, uint64_t{lanes} * queryHeads * stride * kDims * 2);
    std::mt19937 random(lanes * 17 + queryHeads);
    std::normal_distribution<float> normal;
    for (uint64_t index = 0; index < queries.sizeBytes() / 2; ++index)
      static_cast<uint16_t *>(queries.contents())[index] = floatToBf16(normal(random));
    for (uint32_t step = 0; step < 2; ++step) {
      std::array<kv::ChunkedPrefillParams, 4> chunks{};
      std::array<uint32_t, 4> committed{};
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        committed[lane] = histories[lane] + (step ? accepted : 0);
        const uint64_t laneRows = uint64_t{lane} * kvHeads * stride * kDims;
        stage(step ? second[lane] : first[lane], kvHeads, committed[lane], rows, stride,
              static_cast<uint16_t *>(keys.contents()) + laneRows,
              static_cast<uint16_t *>(staged.contents()) + laneRows);
        chunks[lane] = ops::PagedAttention::verifyParams(committed[lane], pageCounts[lane]);
      }
      const auto plan = ops::PagedAttention::verifyPlan(lanes, queryHeads, pool.layout,
                                                        std::span(committed).first(lanes));
      auto output = allocate(backend, queries.sizeBytes());
      auto partials = allocate(backend, plan.workspace.partialsBytes);
      auto statistics = allocate(backend, plan.workspace.statisticsBytes);
      metal::CommandGraph graph;
      ops::PagedAttention::addVerify(graph, pool.layer,
                                     {keys, staged, queries, partials, statistics, output, tables, pool.codec},
                                     std::span(chunks).first(lanes), plan);
      require(graph.dispatches().size() == 3, "verify should encode store, split and reduce");
      (void)backend.submitCommandAsync(graph.dispatches()).wait();
      const auto *bits = static_cast<const uint16_t *>(output.contents());
      outputs[step][format == kv::Format::ZipBFloat16] = {bits, bits + output.sizeBytes() / 2};
      if (format == kv::Format::ZipBFloat16) {
        for (uint32_t lane = 0; lane < lanes; ++lane)
          (void)requireDecodes(pool, pages[lane], step ? second[lane] : first[lane], committed[lane] + rows,
                               "verify store, step " + std::to_string(step) + ", lane " + std::to_string(lane));
        require(overflowSlabs(pool) == 0, "verify store counted an overflow");
      }
    }
  }
  for (uint32_t step = 0; step < 2; ++step) {
    const std::string difference = compare(outputs[step][0], outputs[step][1]);
    require(difference.empty(), "ZBF16 verify attention differs from BF16 at step " + std::to_string(step) +
                                    ": " + difference);
  }
  std::cout << "zbf16 verify: q=" << queryHeads << " lanes=" << lanes << " histories=" << histories[0]
            << ".." << histories[lanes - 1] << " PASS (bit-equal to bf16, rejected rows rewritten)\n";
}

// Rows of uniformly random bits escape far more often than the table holds:
// every slab is counted, the elements whose escapes were dropped decode
// finite and the rest exact; no byte past the page changes.
void checkEscapeOverflow(metal::MetalBackend &backend) {
  constexpr uint32_t kvHeads = 4, rows = 32;
  Values values;
  values.tokens = rows;
  std::mt19937 random(5);
  for (uint32_t tensor = 0; tensor < 2; ++tensor)
    for (uint32_t head = 0; head < kvHeads; ++head) {
      values.rows[tensor].emplace_back(rows * kDims);
      for (auto &bits : values.rows[tensor].back()) bits = uint16_t(random());
    }
  std::array<std::vector<HeadBases>, 2> bases{std::vector<HeadBases>(kvHeads), std::vector<HeadBases>(kvHeads)};
  Pool pool = makePool(backend, kv::Format::ZipBFloat16, kvHeads, 3, bases, 4, 0);
  const auto pages = distinctPages(pool, 3, 9);
  auto table = allocate(backend, 3 * sizeof(SplashKvPage));
  pool.extents.writeTable(pages, table.contents());
  std::vector<std::vector<uint8_t>> before;
  for (uint32_t extent = 0; extent < pool.extents.extentCount(); ++extent) {
    const auto bytes = pool.extents.bytes(extent);
    before.emplace_back(reinterpret_cast<const uint8_t *>(bytes.data()),
                        reinterpret_cast<const uint8_t *>(bytes.data()) + bytes.size());
  }
  auto keys = allocate(backend, uint64_t{kvHeads} * rows * kDims * 2);
  auto staged = allocate(backend, keys.sizeBytes());
  stage(values, kvHeads, 0, rows, rows, static_cast<uint16_t *>(keys.contents()),
        static_cast<uint16_t *>(staged.contents()));
  metal::CommandGraph graph;
  ops::PagedAttention::addPrefillStore(graph, pool.layer, keys, staged, table,
                                      ops::PagedAttention::prefillParams(0, rows, rows, 1), pool.layout,
                                      zipPrefill(backend, pool));
  (void)backend.submitCommandAsync(graph.dispatches()).wait();
  require(overflowSlabs(pool) == 2 * kvHeads, "every slab that dropped escapes must be counted once, got " +
                                                  std::to_string(overflowSlabs(pool)));
  const ZipPage zip = pool.page(pages[0]);
  uint32_t exact = 0;
  for (uint32_t tensor = 0; tensor < 2; ++tensor)
    for (uint32_t head = 0; head < kvHeads; ++head) {
      require(zip.count(tensor, head) == SPLASH_KVZIP_ESCAPES, "a full escape table should hold every entry");
      const auto decoded = decodeSlab(zip, tensor, head, rows, bases[tensor][head]);
      // The host store on the same values says which elements kept their
      // escapes: the first table's worth, in row order.
      std::vector<uint32_t> kept;
      for (uint32_t row = 0; row < rows && kept.size() < SPLASH_KVZIP_ESCAPES; ++row)
        for (const uint32_t escape : encodeRow(values.rows[tensor][head].data() + row * kDims, row, bases[tensor][head]).escapes)
          if (kept.size() < SPLASH_KVZIP_ESCAPES) kept.push_back(escape & 0xFFFF);
      for (uint32_t position = 0; position < rows * kDims; ++position) {
        const uint16_t bits = decoded[position], source = values.rows[tensor][head][position];
        const uint32_t delta = (((source >> 7) & 0xFF) - bases[tensor][head].base4[position % kDims]) & 0xFF;
        const bool escaped = delta >= 16;
        const bool dropped = escaped && !std::binary_search(kept.begin(), kept.end(), position);
        require(((bits >> 7) & 0xFF) != 0xFF || !dropped, "a dropped escape decoded to an infinity or NaN");
        require(dropped || bits == source, "an element with its escape does not decode to its bits");
        exact += !dropped;
      }
    }
  for (uint32_t extent = 0; extent < pool.extents.extentCount(); ++extent) {
    const auto bytes = pool.extents.bytes(extent);
    const auto *after = reinterpret_cast<const uint8_t *>(bytes.data());
    for (uint32_t page = 0; page < pool.extents.extentPages(); ++page) {
      const uint32_t id = extent * pool.extents.extentPages() + page;
      if (id == pages[0]) continue;
      for (uint32_t layer = 0; layer < pool.layout.attentionLayers; ++layer)
        for (uint32_t tensor = 0; tensor < 4; ++tensor) {
          const uint64_t offset = pool.extents.slab<uint8_t>(layer, tensor, id) - after;
          const uint64_t count = tensor % 2 ? pool.layout.scaleBytesPerLayerPage() : pool.layout.dataBytesPerLayerPage();
          require(std::equal(after + offset, after + offset + count, before[extent].begin() + offset),
                  "an overflowing store wrote outside its page");
        }
    }
  }
  std::cout << "zbf16 escape overflow: " << overflowSlabs(pool) << " slabs counted, " << exact
            << " elements exact, dropped escapes finite, other pages intact PASS\n";
}

void checkLayout() {
  for (const uint32_t heads : {4U, 2U}) {
    const kv::Layout zip{16, heads, 256, kv::Format::ZipBFloat16};
    const kv::Layout bf16{16, heads, 256, kv::Format::BFloat16};
    require(zip.valid() && zip.dataBytesPerLayerPage() == heads * 12288ull &&
                zip.scaleBytesPerLayerPage() == heads * 1024ull,
            "ZBF16 page geometry changed");
    require(zip.bytesPerModelPage() * 1000 / bf16.bytesPerModelPage() == 812,
            "ZBF16 pages should be 0.8125 of BF16's");
    require(zip.extentAlignmentPages() == (heads == 4 ? 16U : 32U), "ZBF16 extent alignment changed");
  }
  require(!kv::Layout{16, 4, 128, kv::Format::ZipBFloat16}.valid(),
          "ZBF16 accepted a head dimension its codec does not cover");
  std::cout << "zbf16 layout: 0.8125x of bf16 pages PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    checkLayout();
    if (argc < 2) return 0;
    metal::MetalBackend backend(argv[1]);
    checkEscapeOverflow(backend);
    for (const uint32_t heads : {24U, 16U}) {
      for (const auto [history, rows] : std::array<std::array<uint32_t, 2>, 6>{
               {{0, 1}, {0, 64}, {13, 50}, {100, 77}, {1023, 257}, {4093, 1057}}})
        checkPrefill(backend, heads, history, rows);
      checkPrefill(backend, heads, 45, 300, true);
      checkVerify(backend, heads, {0, 29, 31, 1000}, 4);
      checkVerify(backend, heads, {4090, 0, 0, 0}, 1);
    }
    std::cout << "kv zip: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "kv zip: FAIL: " << error.what() << '\n';
    return 1;
  }
}

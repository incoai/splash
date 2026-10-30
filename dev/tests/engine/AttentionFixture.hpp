#pragma once

#include "HostKvExtents.hpp"
#include "KvZipHost.hpp"
#include "LinearNumerics.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

// The paged-attention fixture attention-sweep times: deterministic Page32
// history of every lane in the extents of a pool, one chunk of rows per lane
// with its queries, and the production store and attention graph over them;
// and the move of a block from a page to a block cache that a verify reads
// through its locations.
namespace splash::test {

// Real BF16 KV a fixture can hold instead of its synthetic history: the
// pages a persistent cache wrote for a model (dev/tools/kvzip_calibrate.py
// reads the same files), one page per slot of `layers` attention layers of
// `heads` KV heads of 256 dimensions, each the keys then the values of every
// KV head (keys token-major, values dimension-major). History token t of a
// lane takes row t % 32 of a sample page, the middle layer, KV head h % heads.
class KvSample final {
public:
  KvSample(const std::string &path, uint32_t layers, uint32_t heads) : heads_(heads) {
    const int file = ::open(path.c_str(), O_RDONLY);
    struct stat info {};
    if (file < 0 || ::fstat(file, &info) != 0)
      throw std::runtime_error("cannot open KV sample " + path);
    const uint64_t payload = uint64_t{layers} * 2 * heads * kv::kPageTokens * kDimensions * 2;
    slotBytes_ = (payload + 16383) / 16384 * 16384;
    bytes_ = uint64_t(info.st_size);
    pages_ = uint32_t(bytes_ / slotBytes_);
    void *mapped = pages_ ? ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, file, 0) : MAP_FAILED;
    ::close(file);
    if (mapped == MAP_FAILED)
      throw std::runtime_error("KV sample " + path + " holds no page of this shape");
    data_ = static_cast<const uint16_t *>(mapped);
    layer_ = layers / 2;
  }
  ~KvSample() { ::munmap(const_cast<uint16_t *>(data_), bytes_); }
  KvSample(const KvSample &) = delete;
  KvSample &operator=(const KvSample &) = delete;

  [[nodiscard]] uint32_t heads() const noexcept { return heads_; }
  // The BF16 bits of element d of KV head `head`'s token in tensor 0 (keys)
  // or 1 (values) of the lane's history.
  [[nodiscard]] uint16_t bits(uint32_t lane, uint32_t tensor, uint32_t head, uint32_t token,
                              uint32_t d) const noexcept {
    const uint32_t page = uint32_t((uint64_t{lane} * 7919 + token / kv::kPageTokens) % pages_);
    const uint32_t row = token % kv::kPageTokens;
    const uint64_t slab = uint64_t{kv::kPageTokens} * kDimensions;
    const uint64_t at = uint64_t{page} * (slotBytes_ / 2) +
                        ((uint64_t{layer_} * 2 + tensor) * heads_ + head % heads_) * slab +
                        (tensor ? uint64_t{d} * kv::kPageTokens + row : uint64_t{row} * kDimensions + d);
    return data_[at];
  }

private:
  static constexpr uint32_t kDimensions = SPLASH_KVZIP_DIMENSIONS;
  uint32_t heads_, layer_ = 0, pages_ = 0;
  uint64_t slotBytes_ = 0, bytes_ = 0;
  const uint16_t *data_ = nullptr;
};

// The target attention layer a fixture holds: its query and KV heads and
// their width, its cache format, its value order (kv::Layout::
// valuesTokenMajor) and its variant of the geometry (a window, RoPE).
struct AttentionShape final {
  uint32_t queryHeads = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  kv::Format format = kv::Format::Int8;
  bool valuesTokenMajor = false;
  ops::AttentionVariant variant{};
};

// Where a fixture's attention layer sits: layer `layer` of a pool of
// poolLayers attention layers, in extents of extentPages pages.
struct AttentionFixtureGeometry final {
  uint32_t poolLayers = 0;
  uint32_t layer = 0;
  uint32_t extentPages = 0;
};

// The sizes of one fixture: `lanes` lanes of `rows` rows after their
// histories, the pool that holds their pages, and the one 16 KiB-aligned
// shared allocation that holds the pool's extents and, after them, the
// tensors.
struct AttentionFixturePlan final {
  static constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  static constexpr uint64_t kAlignment = 16 * 1024;
  // Sinks: an F32 per query head, which a geometry with learned sinks adds.
  // Table0 is lane 0's page table; every other lane's follows it. The ZBF16
  // codec, prefill scratch and its table (ops::KvZipPrefill) are empty for
  // other formats.
  enum class Tensor : uint32_t {
    ChunkKeys, ChunkValues, Queries, Output, Partials, Statistics, Sinks,
    ZipCodec, ZipScratch, ZipScratchTable,
    Table0, Count = Table0 + kMaximumLanes
  };
  using Histories = std::array<uint32_t, kMaximumLanes>;

  AttentionShape shape;
  AttentionFixtureGeometry geometry;
  uint32_t lanes = 0;
  uint32_t rows = 0;
  // A lane's rows of chunk keys and values, queries and output: whole pages.
  uint32_t stride = 0;
  Histories histories{};
  // The pages of each lane's history and rows.
  Histories pages{};
  uint32_t poolPages = 0;
  uint32_t extents = 0;
  // The pool's extents, extentStride() apart, at the start of the allocation.
  uint64_t poolBytes = 0;
  std::array<uint64_t, static_cast<size_t>(Tensor::Count)> sizes{};
  uint64_t bytes = 0;

  // The pages of each of `lanes` lanes of `rows` rows after their histories;
  // lanes past `lanes` have none.
  [[nodiscard]] static Histories pagesOf(uint32_t lanes, uint32_t rows,
                                         const Histories &histories) {
    if (!lanes || lanes > kMaximumLanes || !rows)
      throw std::invalid_argument("attention fixture lanes or rows are invalid");
    Histories pages{};
    for (uint32_t lane = 0; lane < kMaximumLanes; ++lane) {
      if (lane >= lanes) {
        if (histories[lane])
          throw std::invalid_argument("inactive attention fixture lane has a history");
        continue;
      }
      const uint64_t tokens = uint64_t{histories[lane]} + rows;
      if (tokens > kv::kMaximumPhysicalTokens)
        throw std::invalid_argument("attention fixture history exceeds context");
      pages[lane] = uint32_t((tokens + kv::kPageTokens - 1) / kv::kPageTokens);
    }
    return pages;
  }
  // A pool for lanes of these pages: their pages and one or two spare ones,
  // since an odd pool permits the stride-two page permutation the fixture
  // fills.
  [[nodiscard]] static uint32_t poolPagesFor(const Histories &pages) {
    uint32_t total = 0;
    for (const uint32_t lanePages : pages) total += lanePages;
    return total + 1 + total % 2;
  }

  // Scratch is the caller's: the largest workspace of the attention plans
  // it encodes.
  [[nodiscard]] static AttentionFixturePlan make(AttentionShape shape, uint32_t lanes,
                                                 uint32_t rows, const Histories &histories,
                                                 ops::AttentionWorkspace scratch,
                                                 AttentionFixtureGeometry geometry) {
    AttentionFixturePlan plan;
    plan.shape = shape;
    plan.geometry = geometry;
    plan.lanes = lanes;
    plan.rows = rows;
    plan.histories = histories;
    plan.pages = pagesOf(lanes, rows, histories);
    plan.poolPages = poolPagesFor(plan.pages);
    const kv::Layout pool = plan.poolLayout();
    if (!pool.valid() ||
        geometry.layer >= geometry.poolLayers || !geometry.extentPages ||
        geometry.extentPages % pool.extentAlignmentPages() ||
        geometry.extentPages > SPLASH_KV_PAGE_INDEX_MASK)
      throw std::invalid_argument("attention fixture geometry is invalid");
    plan.stride = (rows + kv::kPageTokens - 1) / kv::kPageTokens * kv::kPageTokens;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      plan.size(table(lane), uint64_t{plan.pages[lane]} * sizeof(SplashKvPage));
    plan.extents = (plan.poolPages + geometry.extentPages - 1) / geometry.extentPages;
    plan.poolBytes =
        uint64_t{plan.extents} * HostKvExtents::extentStride(pool, geometry.extentPages);
    const uint64_t chunks =
        uint64_t{lanes} * shape.kvHeads * plan.stride * shape.headDimension * sizeof(uint16_t);
    const uint64_t queries =
        uint64_t{lanes} * shape.queryHeads * plan.stride * shape.headDimension * sizeof(uint16_t);
    plan.size(Tensor::ChunkKeys, chunks);
    plan.size(Tensor::ChunkValues, chunks);
    plan.size(Tensor::Queries, queries);
    plan.size(Tensor::Output, queries);
    plan.size(Tensor::Partials, scratch.partialsBytes);
    plan.size(Tensor::Statistics, scratch.statisticsBytes);
    plan.size(Tensor::Sinks, uint64_t{shape.queryHeads} * sizeof(float));
    if (shape.format == kv::Format::ZipBFloat16) {
      plan.size(Tensor::ZipCodec, splash_kvzip_codec_bytes(geometry.poolLayers, shape.kvHeads));
      plan.size(Tensor::ZipScratch, uint64_t{kv::zipScratchPages()} * kv::zipScratchLayout(pool).bytesPerModelPage());
      plan.size(Tensor::ZipScratchTable, uint64_t{kv::zipScratchPages()} * sizeof(SplashKvPage));
    }
    plan.bytes = aligned(plan.poolBytes);
    for (uint64_t size : plan.sizes) {
      const uint64_t allocation = aligned(size);
      if (allocation > std::numeric_limits<uint64_t>::max() - plan.bytes)
        throw std::invalid_argument("attention fixture size overflow");
      plan.bytes += allocation;
    }
    return plan;
  }

  [[nodiscard]] static Tensor table(uint32_t lane) noexcept {
    return static_cast<Tensor>(static_cast<uint32_t>(Tensor::Table0) + lane);
  }
  [[nodiscard]] static uint64_t aligned(uint64_t bytes) {
    if (bytes > std::numeric_limits<uint64_t>::max() - kAlignment + 1)
      throw std::invalid_argument("attention fixture size overflow");
    return (bytes + kAlignment - 1) & ~(kAlignment - 1);
  }
  // Attention plans are per layer.
  [[nodiscard]] kv::Layout layout() const noexcept { return layoutOf(1); }
  [[nodiscard]] kv::Layout poolLayout() const noexcept { return layoutOf(geometry.poolLayers); }
  // Queries and output are [lane][KV head][row][query head in group][dimension].
  [[nodiscard]] uint64_t queryIndex(uint32_t lane, uint32_t head, uint32_t row,
                                    uint32_t dimension) const noexcept {
    const uint32_t group = shape.queryHeads / shape.kvHeads;
    return (((uint64_t{lane} * shape.kvHeads + head / group) * stride + row) * group +
            head % group) * shape.headDimension + dimension;
  }

private:
  [[nodiscard]] kv::Layout layoutOf(uint32_t layers) const noexcept {
    return {.attentionLayers = layers,
            .kvHeads = shape.kvHeads,
            .headDimension = shape.headDimension,
            .format = shape.format,
            .valuesTokenMajor = shape.valuesTokenMajor};
  }
  void size(Tensor tensor, uint64_t value) noexcept {
    sizes[static_cast<size_t>(tensor)] = value;
  }
};

// A fixture's extents start its one allocation and its tensors are views of
// the rest; kernels reach the extents only through the lanes' page tables, so
// the allocation is addressed (metal::BufferReach), resident for every
// command buffer. Lanes past the plan's lanes bind lane 0's page table, as the
// runtime's padded lanes do.
class AttentionFixture final {
public:
  using Tensor = AttentionFixturePlan::Tensor;

  AttentionFixture(metal::MetalBackend &backend, AttentionFixturePlan plan,
                   std::string_view label)
      : plan_(std::move(plan)),
        base_(backend.allocateBuffer(plan_.bytes, metal::BufferStorage::Shared, label,
                                     metal::BufferReach::Addressed)),
        pages_(HostKvExtents::contiguous(plan_.poolLayout(), plan_.geometry.extentPages,
                                         plan_.extents,
                                         static_cast<std::byte *>(base_.contents()),
                                         base_.gpuAddress())),
        layer_(pages_.layer(plan_.geometry.layer)) {
    uint64_t offset = AttentionFixturePlan::aligned(plan_.poolBytes);
    for (size_t index = 0; index < plan_.sizes.size(); ++index) {
      if (plan_.sizes[index])
        buffers_[index] = backend.view(base_, offset, plan_.sizes[index]);
      offset += AttentionFixturePlan::aligned(plan_.sizes[index]);
    }
    uint32_t firstPage = 0;
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane) {
      for (uint32_t page = 0; page < plan_.pages[lane]; ++page)
        pageIds_[lane].push_back((2 * (firstPage + page) + 1) % plan_.poolPages);
      firstPage += plan_.pages[lane];
      tables_[lane] = buffer(AttentionFixturePlan::table(lane));
    }
    for (uint32_t lane = plan_.lanes; lane < AttentionFixturePlan::kMaximumLanes; ++lane)
      tables_[lane] = tables_[0];
  }

  // ZBF16's codec and prefill scratch; empty for other formats.
  [[nodiscard]] ops::KvZipPrefill zip() const {
    return {buffer(Tensor::ZipCodec), buffer(Tensor::ZipScratch), buffer(Tensor::ZipScratchTable)};
  }

  // Zeroes the allocation, then writes the page tables, every lane's
  // history, chunk and queries. Long histories consult `stop` every 256
  // tokens; false when it stopped the fill. BF16 and ZBF16 history comes
  // from `sample` when one is given.
  bool fill(const std::function<bool()> &stop = {}, const KvSample *sample = nullptr) {
    std::memset(base_.contents(), 0, plan_.bytes);
    const bool zipped = plan_.shape.format == kv::Format::ZipBFloat16;
    sample_ = plan_.shape.format == kv::Format::Int8 ? nullptr : sample;
    auto *chunkKeys = data<uint16_t>(Tensor::ChunkKeys);
    auto *chunkValues = data<uint16_t>(Tensor::ChunkValues);
    auto *queries = data<uint16_t>(Tensor::Queries);
    const uint32_t dimensions = plan_.shape.headDimension;
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane) {
      pages_.writeTable(pageIds_[lane], tables_[lane].contents());
      for (uint32_t token = 0; token < plan_.histories[lane]; ++token) {
        if (token % 256 == 0 && stop && stop()) return false;
        for (uint32_t head = 0; head < plan_.shape.kvHeads; ++head) {
          const auto key = [&](uint32_t d) { return historyKey(lane, head, token, d); };
          const auto value = [&](uint32_t d) { return historyValue(lane, head, token, d); };
          if (plan_.shape.format == kv::Format::Int8) {
            *scale(lane, SPLASH_KV_KEY_SCALES, head, token) = 0.006f;
            *scale(lane, SPLASH_KV_VALUE_SCALES, head, token) = 0.007f;
            auto *keys = keyRow<int8_t>(lane, head, token);
            for (uint32_t d = 0; d < dimensions; ++d) {
              keys[d] = key(d);
              *valueOf<int8_t>(lane, head, token, d) = value(d);
            }
          } else if (zipped) {
            break; // encodeZipHistory writes the pages
          } else {
            auto *keys = keyRow<uint16_t>(lane, head, token);
            for (uint32_t d = 0; d < dimensions; ++d) {
              keys[d] = historyBits(lane, 0, head, token, d);
              *valueOf<uint16_t>(lane, head, token, d) = historyBits(lane, 1, head, token, d);
            }
          }
        }
      }
      for (uint32_t row = 0; row < plan_.rows; ++row) {
        for (uint32_t head = 0; head < plan_.shape.kvHeads; ++head) {
          const uint64_t base =
              (uint64_t{lane} * plan_.shape.kvHeads + head) * plan_.stride * dimensions;
          for (uint32_t d = 0; d < dimensions; ++d) {
            chunkKeys[base + row * dimensions + d] = floatToBf16(
                float(int((row * 37 + head * 101 + d * 17 + lane * 7) % 255) - 127) * 0.006f);
            chunkValues[base + uint64_t{d} * plan_.stride + row] = floatToBf16(
                float(int((row * 53 + head * 79 + d * 29 + lane * 19) % 255) - 127) * 0.007f);
          }
        }
        for (uint32_t head = 0; head < plan_.shape.queryHeads; ++head)
          for (uint32_t d = 0; d < dimensions; ++d)
            queries[plan_.queryIndex(lane, head, row, d)] = floatToBf16(
                float(int((row * 43 + head * 67 + d * 11 + head * d * 7 + lane * 29) % 1019) -
                      509) /
                1018.0f);
      }
    }
    for (uint32_t head = 0; head < plan_.shape.queryHeads; ++head)
      data<float>(Tensor::Sinks)[head] = 0.5f * float(head % 7) - 1.0f;
    if (zipped) {
      auto *scratch = data<SplashKvPage>(Tensor::ZipScratchTable);
      for (uint32_t page = 0; page < kv::zipScratchPages(); ++page)
        scratch[page] = splash_kv_page_entry(buffer(Tensor::ZipScratch).gpuAddress(), page);
      encodeZipHistory();
    }
    return true;
  }

  // A lane's history token: INT8 keys and values, and BF16 ones scaled by
  // 0.006 and 0.007.
  [[nodiscard]] static int historyKey(uint32_t lane, uint32_t head, uint32_t token, uint32_t d) {
    return int((uint64_t{token} * 37 + head * 101 + d * 17 + uint64_t{token} * d * 3 + lane * 7) % 255) -
           127;
  }
  [[nodiscard]] static int historyValue(uint32_t lane, uint32_t head, uint32_t token, uint32_t d) {
    return int((uint64_t{token} * 53 + head * 79 + d * 29 + uint64_t{token} * d * 5 + lane * 19) % 255) -
           127;
  }
  // A lane's BF16 history element, keys (tensor 0) or values: the sample's,
  // or the INT8 history scaled.
  [[nodiscard]] uint16_t historyBits(uint32_t lane, uint32_t tensor, uint32_t head, uint32_t token,
                                     uint32_t d) const {
    if (sample_) return sample_->bits(lane, tensor, head, token, d);
    return tensor ? floatToBf16(historyValue(lane, head, token, d) * 0.007f)
                  : floatToBf16(historyKey(lane, head, token, d) * 0.006f);
  }

  // ZBF16 history: the BF16 history the other formats hold, encoded page by
  // page (KvZipHost.hpp) with bases fitted to all of it, as a model's
  // calibration fits its KV.
  void encodeZipHistory() {
    using namespace kvzip;
    const uint32_t heads = plan_.shape.kvHeads;
    std::vector<std::vector<std::array<uint32_t, 256>>> histograms(
        2 * heads, std::vector<std::array<uint32_t, 256>>(kDims));
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane)
      for (uint32_t token = 0; token < plan_.histories[lane]; ++token)
        for (uint32_t tensor = 0; tensor < 2; ++tensor)
          for (uint32_t head = 0; head < heads; ++head)
            for (uint32_t d = 0; d < kDims; ++d)
              ++histograms[tensor * heads + head][d][(historyBits(lane, tensor, head, token, d) >> 7) & 0xFF];
    std::vector<HeadBases> bases;
    auto *codec = data<uint8_t>(Tensor::ZipCodec);
    SplashKvZipHeader header{};
    header.layers = plan_.geometry.poolLayers;
    header.kv_heads = heads;
    std::memcpy(codec, &header, sizeof(header));
    for (uint32_t tensor = 0; tensor < 2; ++tensor)
      for (uint32_t head = 0; head < heads; ++head) {
        bases.push_back(windows(histograms[tensor * heads + head]));
        uint8_t *at = codec + splash_kvzip_base_offset(heads, plan_.geometry.layer, tensor, head);
        for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
          at[splash_kvzip_base_byte(dimension)] = bases.back().base3[dimension];
          at[kDims + splash_kvzip_base_byte(dimension)] = bases.back().base4[dimension];
        }
      }
    std::vector<uint16_t> rows(kRows * kDims);
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane)
      for (uint32_t page = 0; page * kRows < plan_.histories[lane]; ++page) {
        const ZipPage zip{pages_, heads, pageIds_[lane][page], plan_.geometry.layer};
        const uint32_t count = std::min(kRows, plan_.histories[lane] - page * kRows);
        for (uint32_t tensor = 0; tensor < 2; ++tensor)
          for (uint32_t head = 0; head < heads; ++head) {
            for (uint32_t row = 0; row < count; ++row)
              for (uint32_t d = 0; d < kDims; ++d)
                rows[row * kDims + d] = historyBits(lane, tensor, head, page * kRows + row, d);
            if (storeRows(zip, tensor, head, rows.data(), 0, count, bases[tensor * heads + head]))
              throw std::runtime_error("attention fixture ZBF16 history dropped escapes");
          }
      }
  }

  // One store and attention of the fixture's rows, encoded as the runtime
  // encodes them, through block lists (sparse plans) when `blocks` has them.
  void addGraph(metal::CommandGraph &graph, const ops::PrefillAttentionPlan &attention,
                const ops::AttentionBlocks &blocks = {}) const {
    const kv::ChunkedPrefillParams chunk = ops::PagedAttention::prefillParams(
        plan_.histories[0], plan_.rows, plan_.stride, plan_.pages[0]);
    ops::PagedAttention::addPrefillStore(graph, layer_, buffer(Tensor::ChunkKeys),
                                         buffer(Tensor::ChunkValues), tables_[0], chunk,
                                         plan_.layout(), zip());
    ops::PagedAttention::addPrefill(graph, layer_, buffer(Tensor::Queries),
                                    buffer(Tensor::Output), buffer(Tensor::Partials),
                                    buffer(Tensor::Statistics), tables_[0], chunk, attention,
                                    blocks, sinks(attention.sinks), zip());
  }
  // Each lane's rows are its verify rows, in the verify chunk stride.
  void addGraph(metal::CommandGraph &graph, const ops::VerifyAttentionPlan &attention,
                const ops::AttentionBlocks &blocks = {}) const {
    if (plan_.rows != kv::kVerifyRows || plan_.stride != kv::kVerifyChunkStride)
      throw std::logic_error(
          "a verify fixture stages its lanes' verify rows in the verify chunk stride");
    std::array<kv::ChunkedPrefillParams, AttentionFixturePlan::kMaximumLanes> chunks{};
    for (uint32_t lane = 0; lane < attention.lanes; ++lane)
      chunks[lane] =
          ops::PagedAttention::verifyParams(plan_.histories[lane], plan_.pages[lane]);
    ops::PagedAttention::addVerify(
        graph, layer_,
        {buffer(Tensor::ChunkKeys), buffer(Tensor::ChunkValues), buffer(Tensor::Queries),
         buffer(Tensor::Partials), buffer(Tensor::Statistics), buffer(Tensor::Output), tables_,
         sinks(attention.sinks), buffer(Tensor::ZipCodec)},
        std::span(chunks).first(attention.lanes), attention, blocks);
  }

  [[nodiscard]] const AttentionFixturePlan &plan() const noexcept { return plan_; }
  [[nodiscard]] metal::MetalBuffer buffer(Tensor tensor) const {
    return buffers_[static_cast<size_t>(tensor)];
  }

  // A lane's token in its page: one KV head's row of keys, one of its values
  // in the layout's value order, and its scale of keys or values
  // (SPLASH_KV_KEY_SCALES or SPLASH_KV_VALUE_SCALES), INT8 only.
  template <typename T>
  [[nodiscard]] T *keyRow(uint32_t lane, uint32_t head, uint32_t token) const {
    return slab<T>(lane, SPLASH_KV_KEYS, token) +
           splash_kv_key_element(head, token % kv::kPageTokens, 0, plan_.shape.headDimension);
  }
  template <typename T>
  [[nodiscard]] T *valueOf(uint32_t lane, uint32_t head, uint32_t token, uint32_t dimension) const {
    return slab<T>(lane, SPLASH_KV_VALUES, token) +
           pages_.valueElement(head, token % kv::kPageTokens, dimension);
  }
  [[nodiscard]] float *scale(uint32_t lane, uint32_t tensor, uint32_t head,
                             uint32_t token) const {
    return slab<float>(lane, tensor, token) +
           splash_kv_scale_element(head, token % kv::kPageTokens);
  }
  // The pool's extents, and the page of a lane's token in them.
  [[nodiscard]] const HostKvExtents &pool() const noexcept { return pages_; }
  [[nodiscard]] uint32_t page(uint32_t lane, uint32_t token) const {
    return pageIds_[lane][token / kv::kPageTokens];
  }

private:
  template <typename T> T *data(Tensor tensor) const {
    return static_cast<T *>(buffer(tensor).contents());
  }
  // The layer's sinks for a plan whose reduce adds them, none otherwise.
  [[nodiscard]] metal::MetalBuffer sinks(bool added) const {
    return added ? buffer(Tensor::Sinks) : metal::MetalBuffer{};
  }
  template <typename T> T *slab(uint32_t lane, uint32_t tensor, uint32_t token) const {
    return pages_.slab<T>(plan_.geometry.layer, tensor, page(lane, token));
  }

  AttentionFixturePlan plan_;
  const KvSample *sample_ = nullptr;
  metal::MetalBuffer base_;
  HostKvExtents pages_;
  SplashKvLayer layer_;
  std::array<metal::MetalBuffer, static_cast<size_t>(Tensor::Count)> buffers_{};
  std::array<std::vector<uint32_t>, AttentionFixturePlan::kMaximumLanes> pageIds_;
  std::array<metal::MetalBuffer, AttentionFixturePlan::kMaximumLanes> tables_{};
};

// Moves one block of SPLASH_ATTENTION_BLOCK_TOKENS tokens of layer `layer`,
// every KV head's keys, values and, for INT8, their scales, from place
// fromPlace of page fromPage of `from` to place toPlace of page toPage of
// `to`: extents of `layout`, whose values are token-major, so that a head's
// block is one run of each tensor. As a page whose blocks a spilled request
// reads from its block cache no longer holds them, the source is left with
// the complement of every byte it held: a kernel that still reads the block
// there reads other bits.
inline void moveBlock(kv::Layout layout, uint32_t layer, const HostKvExtents &from, uint32_t fromPage,
                      uint32_t fromPlace, const HostKvExtents &to, uint32_t toPage, uint32_t toPlace) {
  if (!layout.valuesTokenMajor)
    throw std::invalid_argument("a block of values in dimension order is not one run");
  constexpr uint32_t tokens = SPLASH_ATTENTION_BLOCK_TOKENS;
  const uint64_t elementBytes = layout.format == kv::Format::Int8 ? 1 : 2;
  const auto move = [&](uint32_t tensor, uint64_t source, uint64_t target, uint64_t bytes) {
    std::byte *moved = from.slab<std::byte>(layer, tensor, fromPage) + source;
    std::copy_n(moved, bytes, to.slab<std::byte>(layer, tensor, toPage) + target);
    std::for_each_n(moved, bytes, [](std::byte &byte) { byte = ~byte; });
  };
  for (uint32_t head = 0; head < layout.kvHeads; ++head) {
    // Token-major values sit where keys do (HostKvExtents::valueElement).
    for (const uint32_t tensor : {SPLASH_KV_KEYS, SPLASH_KV_VALUES})
      move(tensor, splash_kv_key_element(head, fromPlace * tokens, 0, layout.headDimension) * elementBytes,
           splash_kv_key_element(head, toPlace * tokens, 0, layout.headDimension) * elementBytes,
           uint64_t{tokens} * layout.headDimension * elementBytes);
    if (layout.format == kv::Format::Int8)
      for (const uint32_t tensor : {SPLASH_KV_KEY_SCALES, SPLASH_KV_VALUE_SCALES})
        move(tensor, splash_kv_scale_element(head, fromPlace * tokens) * sizeof(float),
             splash_kv_scale_element(head, toPlace * tokens) * sizeof(float), tokens * sizeof(float));
  }
}

} // namespace splash::test

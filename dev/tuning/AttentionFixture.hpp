#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/PagedAttention.hpp"
#include "tuning/HostKvExtents.hpp"
#include "tuning/KvZipHost.hpp"
#include "tuning/LinearNumerics.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <functional>
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
// with its queries, and the production store and attention graph over them.
namespace splash::ops::tuning {

// Real BF16 KV a fixture can hold instead of its synthetic history: the
// pages a persistent cache wrote for a model (dev/tools/kvzip_calibrate.py
// reads the same files), one page per slot of `layers` attention layers,
// each the keys then the values of every KV head (keys token-major, values
// dimension-major). History token t of a lane takes row t % 32 of a sample
// page, layer `layer`, KV head h % heads.
class KvSample final {
public:
  KvSample(const std::string &path, uint32_t layers, uint32_t heads)
      : heads_(heads) {
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
  static constexpr uint32_t kDimensions = SPLASH_KV_HEAD_DIMENSION;
  uint32_t heads_, layer_ = 0, pages_ = 0;
  uint64_t slotBytes_ = 0, bytes_ = 0;
  const uint16_t *data_ = nullptr;
};

// The target attention layer a fixture holds: its query and KV heads and its
// cache format.
struct AttentionShape final {
  uint32_t queryHeads = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  kv::Format format = kv::Format::Int8;
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
  static constexpr uint32_t kHeadDimension = SPLASH_KV_HEAD_DIMENSION;
  static constexpr uint64_t kAlignment = 16 * 1024;
  // Table0 is lane 0's page table; every other lane's follows it. The ZBF16
  // codec, prefill scratch and its table (KvZipPrefill) are empty for other
  // formats.
  enum class Tensor : uint32_t {
    ChunkKeys, ChunkValues, Queries, Output, Partials, Statistics,
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
                                                 AttentionWorkspace scratch,
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
    if (shape.headDimension != kHeadDimension || !pool.valid() ||
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
        uint64_t{lanes} * shape.kvHeads * plan.stride * kHeadDimension * sizeof(uint16_t);
    const uint64_t queries =
        uint64_t{lanes} * shape.queryHeads * plan.stride * kHeadDimension * sizeof(uint16_t);
    plan.size(Tensor::ChunkKeys, chunks);
    plan.size(Tensor::ChunkValues, chunks);
    plan.size(Tensor::Queries, queries);
    plan.size(Tensor::Output, queries);
    plan.size(Tensor::Partials, scratch.partialsBytes);
    plan.size(Tensor::Statistics, scratch.statisticsBytes);
    if (shape.format == kv::Format::ZipBFloat16) {
      plan.size(Tensor::ZipCodec, splash_kvzip_codec_bytes(geometry.poolLayers, shape.kvHeads));
      plan.size(Tensor::ZipScratch, uint64_t{kv::zipScratchPages()} *
                                        kv::zipScratchLayout(pool).bytesPerModelPage());
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
  [[nodiscard]] kv::Layout layout() const noexcept {
    return {1, shape.kvHeads, shape.headDimension, shape.format};
  }
  [[nodiscard]] kv::Layout poolLayout() const noexcept {
    return {geometry.poolLayers, shape.kvHeads, shape.headDimension, shape.format};
  }
  // Queries and output are [lane][KV head][row][query head in group][dimension].
  [[nodiscard]] uint64_t queryIndex(uint32_t lane, uint32_t head, uint32_t row,
                                    uint32_t dimension) const noexcept {
    const uint32_t group = shape.queryHeads / shape.kvHeads;
    return (((uint64_t{lane} * shape.kvHeads + head / group) * stride + row) * group +
            head % group) * kHeadDimension + dimension;
  }

private:
  void size(Tensor tensor, uint64_t value) noexcept {
    sizes[static_cast<size_t>(tensor)] = value;
  }
};

// A fixture's extents start its one allocation and its tensors are views of
// the rest, resident for every command like every backend buffer; kernels
// reach the extents only through the lanes' page tables. Lanes past the
// plan's lanes bind lane 0's page table, as the runtime's padded lanes do.
class AttentionFixture final {
public:
  using Tensor = AttentionFixturePlan::Tensor;

  AttentionFixture(metal::MetalBackend &backend, AttentionFixturePlan plan,
                   std::string_view label)
      : plan_(std::move(plan)),
        base_(backend.allocateBuffer(plan_.bytes, metal::BufferStorage::Shared, label)),
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
  [[nodiscard]] KvZipPrefill zip() const {
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
    constexpr uint32_t dimensions = AttentionFixturePlan::kHeadDimension;
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
            auto *values = valueColumn<int8_t>(lane, head, token);
            for (uint32_t d = 0; d < dimensions; ++d) {
              keys[d] = key(d);
              values[d * kv::kPageTokens] = value(d);
            }
          } else if (zipped) {
            break; // encodeZipHistory writes the pages
          } else {
            auto *keys = keyRow<uint16_t>(lane, head, token);
            auto *values = valueColumn<uint16_t>(lane, head, token);
            for (uint32_t d = 0; d < dimensions; ++d) {
              keys[d] = historyBits(lane, 0, head, token, d);
              values[d * kv::kPageTokens] = historyBits(lane, 1, head, token, d);
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
    return int((uint64_t{token} * 37 + head * 101 + d * 17 + uint64_t{token} * d * 3 + lane * 7) %
               255) - 127;
  }
  [[nodiscard]] static int historyValue(uint32_t lane, uint32_t head, uint32_t token, uint32_t d) {
    return int((uint64_t{token} * 53 + head * 79 + d * 29 + uint64_t{token} * d * 5 + lane * 19) %
               255) - 127;
  }

  // A lane's BF16 history element, keys (tensor 0) or values: the sample's,
  // or the INT8 history scaled.
  [[nodiscard]] uint16_t historyBits(uint32_t lane, uint32_t tensor, uint32_t head, uint32_t token,
                                     uint32_t d) const {
    if (sample_)
      return sample_->bits(lane, tensor, head, token, d);
    return tensor ? floatToBf16(historyValue(lane, head, token, d) * 0.007f)
                  : floatToBf16(historyKey(lane, head, token, d) * 0.006f);
  }

  // ZBF16 history: the BF16 history the other formats hold, encoded page by
  // page (tuning/KvZipHost.hpp) with bases fitted to all of it, as a model's
  // calibration fits its KV.
  void encodeZipHistory() {
    using namespace kvzip;
    const uint32_t heads = plan_.shape.kvHeads;
    const auto element = [&](uint32_t lane, uint32_t tensor, uint32_t head, uint32_t token,
                             uint32_t d) { return historyBits(lane, tensor, head, token, d); };
    std::vector<std::vector<std::array<uint32_t, 256>>> histograms(
        2 * heads, std::vector<std::array<uint32_t, 256>>(kDims));
    for (uint32_t lane = 0; lane < plan_.lanes; ++lane)
      for (uint32_t token = 0; token < plan_.histories[lane]; ++token)
        for (uint32_t tensor = 0; tensor < 2; ++tensor)
          for (uint32_t head = 0; head < heads; ++head)
            for (uint32_t d = 0; d < kDims; ++d)
              ++histograms[tensor * heads + head][d][(element(lane, tensor, head, token, d) >> 7) & 0xFF];
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
        std::memcpy(at, bases.back().base3.data(), kDims);
        std::memcpy(at + kDims, bases.back().base4.data(), kDims);
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
                rows[row * kDims + d] = element(lane, tensor, head, page * kRows + row, d);
            if (storeRows(zip, tensor, head, rows.data(), 0, count, bases[tensor * heads + head]))
              throw std::runtime_error("attention fixture ZBF16 history dropped escapes");
          }
      }
  }

  // One store and attention of the fixture's rows, encoded as the runtime
  // encodes them.
  void addGraph(metal::CommandGraph &graph, const PrefillAttentionPlan &attention) const {
    const kv::ChunkedPrefillParams chunk = PagedAttention::prefillParams(
        plan_.histories[0], plan_.rows, plan_.stride, plan_.pages[0]);
    PagedAttention::addPrefillStore(graph, layer_, buffer(Tensor::ChunkKeys),
                                    buffer(Tensor::ChunkValues), tables_[0], chunk,
                                    plan_.layout(), zip());
    PagedAttention::addPrefill(graph, layer_, buffer(Tensor::Queries), buffer(Tensor::Output),
                               buffer(Tensor::Partials), buffer(Tensor::Statistics), tables_[0],
                               chunk, attention, zip());
  }
  // Each lane's rows are its verify rows, in the verify chunk stride.
  void addGraph(metal::CommandGraph &graph, const VerifyAttentionPlan &attention) const {
    if (plan_.rows != kv::kVerifyRows || plan_.stride != kv::kVerifyChunkStride)
      throw std::logic_error(
          "a verify fixture stages its lanes' verify rows in the verify chunk stride");
    std::array<kv::ChunkedPrefillParams, AttentionFixturePlan::kMaximumLanes> chunks{};
    for (uint32_t lane = 0; lane < attention.lanes; ++lane)
      chunks[lane] = PagedAttention::verifyParams(plan_.histories[lane], plan_.pages[lane]);
    PagedAttention::addVerify(
        graph, layer_,
        {buffer(Tensor::ChunkKeys), buffer(Tensor::ChunkValues), buffer(Tensor::Queries),
         buffer(Tensor::Partials), buffer(Tensor::Statistics), buffer(Tensor::Output), tables_,
         buffer(Tensor::ZipCodec)},
        std::span(chunks).first(attention.lanes), attention);
  }

  [[nodiscard]] const AttentionFixturePlan &plan() const noexcept { return plan_; }
  [[nodiscard]] metal::MetalBuffer buffer(Tensor tensor) const {
    return buffers_[static_cast<size_t>(tensor)];
  }

  // A lane's token in its page: one KV head's row of keys, its column of
  // values (one element every page token), and its scale of keys or values
  // (SPLASH_KV_KEY_SCALES or SPLASH_KV_VALUE_SCALES), INT8 only.
  template <typename T>
  [[nodiscard]] T *keyRow(uint32_t lane, uint32_t head, uint32_t token) const {
    return slab<T>(lane, SPLASH_KV_KEYS, token) +
           splash_kv_key_element(head, token % kv::kPageTokens, 0);
  }
  template <typename T>
  [[nodiscard]] T *valueColumn(uint32_t lane, uint32_t head, uint32_t token) const {
    return slab<T>(lane, SPLASH_KV_VALUES, token) +
           splash_kv_value_element(head, token % kv::kPageTokens, 0);
  }
  [[nodiscard]] float *scale(uint32_t lane, uint32_t tensor, uint32_t head,
                             uint32_t token) const {
    return slab<float>(lane, tensor, token) +
           splash_kv_scale_element(head, token % kv::kPageTokens);
  }

private:
  template <typename T> T *data(Tensor tensor) const {
    return static_cast<T *>(buffer(tensor).contents());
  }
  template <typename T> T *slab(uint32_t lane, uint32_t tensor, uint32_t token) const {
    return pages_.slab<T>(plan_.geometry.layer, tensor,
                          pageIds_[lane][token / kv::kPageTokens]);
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

} // namespace splash::ops::tuning

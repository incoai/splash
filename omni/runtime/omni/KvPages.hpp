#pragma once

// The KV pages of the Omni decoders (metal/abi/Omni.h): one pool of
// OMNI_PAGE_BYTES pages, allocated a page per (sequence, layer, 32-token
// block), and the lanes' page tables, [lane][layer][block] -> page, in one
// shared buffer the attention kernels read.

#include "metal/MetalBackend.hpp"
#include "metal/abi/Omni.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

namespace splash::omni {

class KvPagePool final {
public:
  KvPagePool(metal::MetalBackend &backend, uint32_t pageCount, std::string_view label = "omni kv pages")
      : pageCount_(pageCount) {
    if (!pageCount) throw std::invalid_argument("a KV pool takes at least one page");
    pages_ = backend.allocateBuffer(uint64_t{pageCount} * OMNI_PAGE_BYTES, metal::BufferStorage::Shared, label);
    // Attention multiplies masked probabilities (zero) into every slot of a
    // page it visits, so slots never written must be finite: a page is
    // zeroed the first time it is handed out (zeroed_), not at startup.
    zeroed_.assign(pageCount, false);
    free_.reserve(pageCount);
    for (uint32_t page = pageCount; page-- > 0;) free_.push_back(page);
  }
  [[nodiscard]] const metal::MetalBuffer &pages() const noexcept { return pages_; }
  [[nodiscard]] uint32_t pageCount() const noexcept { return pageCount_; }
  [[nodiscard]] uint32_t freePages() const noexcept { return static_cast<uint32_t>(free_.size()); }
  [[nodiscard]] std::optional<uint32_t> allocate() {
    if (free_.empty()) return std::nullopt;
    const uint32_t page = free_.back();
    free_.pop_back();
    if (!zeroed_[page]) {
      std::memset(static_cast<uint8_t *>(pages_.contents()) + uint64_t{page} * OMNI_PAGE_BYTES, 0, OMNI_PAGE_BYTES);
      zeroed_[page] = true;
    }
    return page;
  }
  void release(uint32_t page) {
    if (page >= pageCount_) throw std::logic_error("released a page outside the pool");
    free_.push_back(page);
  }

private:
  metal::MetalBuffer pages_;
  uint32_t pageCount_;
  std::vector<uint32_t> free_;
  std::vector<bool> zeroed_;
};

// The page tables of `lanes` lanes of `layers` layers of up to `blocks`
// 32-token blocks, in one shared buffer: lane l's table starts at entry
// l * layers * blocks, with `blocks` entries per layer.
class LanePageTables final {
public:
  LanePageTables(metal::MetalBackend &backend, uint32_t lanes, uint32_t layers, uint32_t blocks,
                 std::string_view label = "omni page tables")
      : lanes_(lanes), layers_(layers), blocks_(blocks) {
    if (!lanes || !layers || !blocks) throw std::invalid_argument("page tables take lanes, layers and blocks");
    table_ = backend.allocateBuffer(uint64_t{entries()} * 4, metal::BufferStorage::Shared, label);
    std::memset(table_.contents(), 0, uint64_t{entries()} * 4);
  }
  [[nodiscard]] const metal::MetalBuffer &buffer() const noexcept { return table_; }
  [[nodiscard]] uint32_t entries() const noexcept { return lanes_ * layers_ * blocks_; }
  [[nodiscard]] uint32_t lanes() const noexcept { return lanes_; }
  [[nodiscard]] uint32_t layers() const noexcept { return layers_; }
  [[nodiscard]] uint32_t blocks() const noexcept { return blocks_; }
  [[nodiscard]] uint32_t offset(uint32_t lane) const noexcept { return lane * layers_ * blocks_; }
  [[nodiscard]] uint32_t stride() const noexcept { return blocks_; }
  [[nodiscard]] uint32_t maximumTokens() const noexcept { return blocks_ * OMNI_PAGE_TOKENS; }
  void set(uint32_t lane, uint32_t layer, uint32_t block, uint32_t page) {
    if (lane >= lanes_ || layer >= layers_ || block >= blocks_) throw std::logic_error("page table entry out of range");
    static_cast<uint32_t *>(table_.contents())[offset(lane) + layer * blocks_ + block] = page;
  }
  [[nodiscard]] uint32_t get(uint32_t lane, uint32_t layer, uint32_t block) const {
    return static_cast<const uint32_t *>(table_.contents())[offset(lane) + layer * blocks_ + block];
  }

private:
  metal::MetalBuffer table_;
  uint32_t lanes_, layers_, blocks_;
};

// One lane's KV allocation: the pages of every block it holds, per layer,
// grown as tokens are committed and returned to the pool when it ends.
class LaneKv final {
public:
  LaneKv(KvPagePool &pool, LanePageTables &tables, uint32_t lane) : pool_(&pool), tables_(&tables), lane_(lane) {}
  ~LaneKv() { release(); }
  LaneKv(const LaneKv &) = delete;
  LaneKv &operator=(const LaneKv &) = delete;
  [[nodiscard]] uint32_t lane() const noexcept { return lane_; }
  [[nodiscard]] uint32_t blocks() const noexcept { return blocks_; }
  // Ensures the lane holds pages for `tokens` tokens; false when the pool
  // has none left (the lane keeps what it had).
  [[nodiscard]] bool reserve(uint32_t tokens) {
    const uint32_t needed = (tokens + OMNI_PAGE_TOKENS - 1) / OMNI_PAGE_TOKENS;
    if (needed > tables_->blocks()) return false;
    while (blocks_ < needed) {
      std::vector<uint32_t> pages;
      for (uint32_t layer = 0; layer < tables_->layers(); ++layer) {
        const std::optional<uint32_t> page = pool_->allocate();
        if (!page) {
          for (uint32_t p : pages) pool_->release(p);
          return false;
        }
        pages.push_back(*page);
      }
      for (uint32_t layer = 0; layer < tables_->layers(); ++layer) tables_->set(lane_, layer, blocks_, pages[layer]);
      owned_.insert(owned_.end(), pages.begin(), pages.end());
      ++blocks_;
    }
    return true;
  }
  void release() {
    for (uint32_t page : owned_) pool_->release(page);
    owned_.clear();
    blocks_ = 0;
  }

private:
  KvPagePool *pool_;
  LanePageTables *tables_;
  uint32_t lane_;
  uint32_t blocks_ = 0;
  std::vector<uint32_t> owned_;
};

} // namespace splash::omni

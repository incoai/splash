#pragma once

#include "engine/CacheRecency.hpp"

#include <cstdint>
#include <optional>
#include <set>
#include <utility>

namespace splash::engine {

// Members in last-used order. A member allocates its tree node once, when it
// is published, and keeps it while unlinked, so linking and unlinking under
// memory pressure never allocate.
class RecencyOrder final {
public:
  class Node final {
  public:
    [[nodiscard]] bool linked() const noexcept { return owner_ != nullptr; }

  private:
    friend class RecencyOrder;
    std::set<std::pair<uint64_t, uint64_t>>::node_type handle_;
    std::pair<uint64_t, uint64_t> key_{};
    RecencyOrder *owner_ = nullptr;
  };

  // The only operation that allocates.
  [[nodiscard]] static Node allocate(uint64_t id) {
    std::set<std::pair<uint64_t, uint64_t>> scratch;
    Node node;
    node.handle_ = scratch.extract(scratch.emplace(0, id).first);
    return node;
  }

  void link(Node &node, uint64_t lastUsed, uint64_t id) noexcept {
    if (node.linked() || node.handle_.empty())
      std::terminate();
    node.handle_.value() = {lastUsed, id};
    if (!order_.insert(std::move(node.handle_)).inserted)
      std::terminate();
    node.key_ = {lastUsed, id};
    node.owner_ = this;
  }

  static void unlink(Node &node) noexcept {
    if (!node.linked())
      std::terminate();
    node.handle_ = node.owner_->order_.extract(node.key_);
    if (node.handle_.empty())
      std::terminate();
    node.owner_ = nullptr;
  }

  [[nodiscard]] std::optional<CacheEvictionCandidate> oldest() const noexcept {
    if (order_.empty())
      return std::nullopt;
    return CacheEvictionCandidate{order_.begin()->second, order_.begin()->first};
  }

  // The member after `after` in the order, for a scan that skips members.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  next(const CacheEvictionCandidate &after) const noexcept {
    auto position = order_.upper_bound({after.lastUsed, after.id});
    if (position == order_.end())
      return std::nullopt;
    return CacheEvictionCandidate{position->second, position->first};
  }

  [[nodiscard]] uint64_t newestId() const noexcept {
    return order_.empty() ? 0 : order_.rbegin()->second;
  }

private:
  std::set<std::pair<uint64_t, uint64_t>> order_;
};

} // namespace splash::engine

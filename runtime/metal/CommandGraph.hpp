#pragma once

#include "MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::metal {

// An ordered dispatch list for one command buffer. Buffers bind at indices
// 0..n-1; an optional parameter struct binds at index n and is copied into
// graph-owned storage until submission.
class CommandGraph final {
public:
  static constexpr uint32_t kDefaultThreads = 256;

  CommandGraph() = default;
  // Dispatches point into payloads_; a copy would keep pointing at the source.
  CommandGraph(const CommandGraph &) = delete;
  CommandGraph &operator=(const CommandGraph &) = delete;
  CommandGraph(CommandGraph &&) noexcept = default;
  CommandGraph &operator=(CommandGraph &&) noexcept = default;

  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           DispatchSize groups, DispatchSize threads = {kDefaultThreads, 1, 1}) {
    push(std::move(pipeline), std::move(buffers), groups, threads);
  }

  template <class Params>
  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           const Params &params, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params>,
                  "dispatch parameters must be plain data");
    payloads_.emplace_back(sizeof(Params));
    std::memcpy(payloads_.back().data(), &params, sizeof(Params));
    ComputeDispatch &dispatch =
        push(std::move(pipeline), std::move(buffers), groups, threads);
    dispatch.bytes.push_back({static_cast<uint32_t>(dispatch.buffers.size()),
                              payloads_.back().data(), sizeof(Params)});
  }

  // As above, then `after` from index n + 1 and `tail` after them: a
  // kernel whose parameter struct precedes further bindings.
  template <class Params, class Tail>
  void add(std::string pipeline, std::vector<MetalBuffer> buffers, const Params &params,
           std::vector<MetalBuffer> after, const Tail &tail, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params> && std::is_trivially_copyable_v<Tail>,
                  "dispatch parameters must be plain data");
    add(std::move(pipeline), std::move(buffers), params, groups, threads);
    ComputeDispatch &dispatch = dispatches_.back();
    uint32_t index = static_cast<uint32_t>(dispatch.buffers.size()) + 1;
    for (MetalBuffer &buffer : after) dispatch.buffers.push_back({index++, std::move(buffer)});
    payloads_.emplace_back(sizeof(Tail));
    std::memcpy(payloads_.back().data(), &tail, sizeof(Tail));
    dispatch.bytes.push_back({index, payloads_.back().data(), sizeof(Tail)});
  }

  [[nodiscard]] bool empty() const noexcept { return dispatches_.empty(); }
  [[nodiscard]] std::span<const ComputeDispatch> dispatches() const noexcept {
    return dispatches_;
  }

private:
  ComputeDispatch &push(std::string pipeline, std::vector<MetalBuffer> buffers,
                        DispatchSize groups, DispatchSize threads) {
    ComputeDispatch dispatch;
    dispatch.pipelineName = std::move(pipeline);
    dispatch.threadgroups = groups;
    dispatch.threadsPerThreadgroup = threads;
    dispatch.buffers.reserve(buffers.size());
    for (uint32_t index = 0; index < buffers.size(); ++index) {
      dispatch.buffers.push_back({index, std::move(buffers[index])});
    }
    dispatches_.push_back(std::move(dispatch));
    return dispatches_.back();
  }

  std::deque<std::vector<std::byte>> payloads_;
  std::vector<ComputeDispatch> dispatches_;
};

} // namespace splash::metal

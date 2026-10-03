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

  // Event steps (EventStep): signal after all earlier work, or hold later
  // work until the event reaches value.
  void signal(SharedEvent event, uint64_t value) { step(std::move(event), value, true); }
  void wait(SharedEvent event, uint64_t value) { step(std::move(event), value, false); }

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

  void step(SharedEvent event, uint64_t value, bool signal) {
    ComputeDispatch dispatch;
    dispatch.event = EventStep{std::move(event), value, signal};
    dispatches_.push_back(std::move(dispatch));
  }

  std::deque<std::vector<std::byte>> payloads_;
  std::vector<ComputeDispatch> dispatches_;
};

} // namespace splash::metal

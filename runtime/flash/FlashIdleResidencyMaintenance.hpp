#pragma once

#include "flash/FlashIdleResidencyPolicy.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace splash::flash::idle_maintenance {
inline constexpr const char *kPipeline = "flash_idle_immutable_touch_v1";
inline constexpr const char *kSSDPipeline = "flash_idle_immutable_touch_ple_ssd_v1";
inline const char *pipelineFor(StorageMode mode) {
  (void)ownerGeometry(mode);
  return mode == StorageMode::Raw ? kPipeline : kSSDPipeline;
}

// Owns references to existing immutable native allocations. It never copies
// weight payloads or references mutable request/model workspaces. Backend
// reflection validates and retains the complete readonly argument owner list.
class Maintenance final {
public:
  static uint64_t plannedBytes(metal::MetalBackend &backend,
      StorageMode mode = StorageMode::Raw) {
    if (!backend.supportsArgumentBuffersTier2())
      throw std::invalid_argument("idle residency maintenance requires Tier2 readonly arguments");
    const uint64_t arguments = backend.readOnlyArgumentBufferByteCount(pipelineFor(mode), 0);
    if (arguments > UINT64_MAX - 16383) throw std::overflow_error("idle residency argument allocation overflows");
    const uint64_t roundedArguments = (arguments + 16383) & ~uint64_t{16383};
    constexpr uint64_t roundedOutput = (uint64_t{kOutputWords} * sizeof(uint32_t) + 16383) & ~uint64_t{16383};
    if (roundedArguments > UINT64_MAX - roundedOutput) throw std::overflow_error("idle residency allocation sum overflows");
    return roundedArguments + roundedOutput;
  }
  Maintenance(metal::MetalBackend &backend, std::vector<metal::MetalBuffer> selected,
      StorageMode mode = StorageMode::Raw)
      : sources_(std::move(selected)) {
    const auto expected = ownerGeometry(mode);
    const char *pipeline = pipelineFor(mode);
    if (sources_.size() != expected.ownerCount) throw std::invalid_argument("idle residency maintenance owner count differs");
    std::vector<metal::BufferBinding> bindings;
    bindings.reserve(sources_.size()); expected_.reserve(sources_.size());
    uint64_t bytes = 0;
    for (uint32_t index = 0; index < sources_.size(); ++index) {
      const auto &source = sources_[index];
      if (!source || source.storage() != metal::BufferStorage::Shared ||
          !source.contents() || reinterpret_cast<uintptr_t>(source.contents()) % alignof(uint32_t) ||
          source.sizeBytes() < sizeof(uint32_t) || source.sizeBytes() % 16384)
        throw std::invalid_argument("idle residency maintenance owner is not an aligned immutable Shared base");
      for (uint32_t previous = 0; previous < index; ++previous)
        if (source.sameView(sources_[previous]) || overlaps(source, sources_[previous]))
          throw std::invalid_argument("idle residency maintenance repeats an immutable owner");
      if (source.sizeBytes() > UINT64_MAX - bytes) throw std::overflow_error("idle residency owner bytes overflow");
      bytes += source.sizeBytes();
      uint32_t first = 0; std::memcpy(&first, source.contents(), sizeof(first));
      const uint32_t mixed = first ^ (0x9e3779b9u * (index + 1u));
      expected_.push_back(mixed); checksum_ ^= mixed;
      bindings.push_back({index, source});
    }
    if (bytes != expected.ownerBytes) throw std::invalid_argument("idle residency immutable byte union differs");
    ownerBytes_ = bytes;
    const uint64_t before = backend.memoryStats().allocatedBytes;
    arguments_ = backend.makeReadOnlyArgumentBuffer(pipeline, 0, bindings,
        "idle residency readonly original and verified derived owners");
    output_ = backend.allocateBuffer(kOutputWords * sizeof(uint32_t), metal::BufferStorage::Shared,
        "idle residency maintenance bounded diagnostics");
    graph_.add(pipeline, {arguments_, output_}, {1, 1, 1}, {256, 1, 1});
    allocatedBytes_ = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
    if (allocatedBytes_ > plannedBytes(backend, mode)) throw std::runtime_error("idle residency allocation ledger exceeds plan");
    clear();
  }
  Maintenance(const Maintenance &) = delete;
  Maintenance &operator=(const Maintenance &) = delete;
  uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }
  uint64_t ownerCount() const noexcept { return sources_.size(); }
  uint64_t ownerBytes() const noexcept { return ownerBytes_; }
  metal::CommandTiming run(metal::MetalBackend &backend) {
    clear();
    const auto timing = backend.submitCommand(graph_.dispatches());
    const auto *output = static_cast<const uint32_t *>(output_.contents());
    if (output[kCountIndex] != sources_.size() || output[kChecksumIndex] != checksum_)
      throw std::runtime_error("idle residency maintenance count/checksum differs");
    for (uint32_t index = 0; index < kOutputWords; ++index) {
      const uint32_t expected = index == kCountIndex ? static_cast<uint32_t>(sources_.size()) :
          index == kChecksumIndex ? checksum_ :
          index >= kOutputValueBegin && index < kOutputValueBegin + expected_.size()
              ? expected_[index - kOutputValueBegin] : kGuard;
      if (output[index] != expected) throw std::runtime_error("idle residency maintenance output/guard differs");
    }
    return timing;
  }
private:
  static bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) noexcept {
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a.contents());
    const uintptr_t bb = reinterpret_cast<uintptr_t>(b.contents());
    return aa <= bb ? uint64_t{bb - aa} < a.sizeBytes() : uint64_t{aa - bb} < b.sizeBytes();
  }
  void clear() {
    auto *output = static_cast<uint32_t *>(output_.contents());
    std::fill_n(output, kOutputWords, kGuard);
    output[kChecksumIndex] = kUnwritten; output[kCountIndex] = kUnwritten;
    std::fill_n(output + kOutputValueBegin, expected_.size(), kUnwritten);
  }
  std::vector<metal::MetalBuffer> sources_;
  std::vector<uint32_t> expected_;
  uint32_t checksum_ = 0;
  uint64_t allocatedBytes_ = 0;
  uint64_t ownerBytes_ = 0;
  metal::MetalBuffer arguments_, output_;
  metal::CommandGraph graph_;
};
} // namespace splash::flash::idle_maintenance

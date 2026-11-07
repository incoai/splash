#pragma once

#include "TestEngine.hpp"
#include "TestMetalStatistics.hpp"
#include "engine/MemoryGovernor.hpp"
#include "metal/MetalBackend.hpp"

#include <deque>
#include <memory>
#include <string>

// The memory governor and the control pass read nothing from the backend but
// its memory statistics, so this stand-in lets a test set them without a GPU.
// It defines the backend's members: one translation unit of a test that links
// no Metal backend includes it.
namespace splash::metal {

struct MetalBackend::Impl {};
MetalBackend::MetalBackend(std::string, double, double)
    : impl_(std::make_unique<Impl>()) {}
MetalBackend::~MetalBackend() = default;
MetalMemoryStats MetalBackend::memoryStats() const noexcept {
  return test::metalStatistics();
}
MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept {
  return test::metalStatistics();
}

} // namespace splash::metal

namespace splash::test {

// The engines a test builds without a governor get one each
// (test::ampleGovernor): a limit and a host far beyond any test's memory,
// sampled once, which every class of claim fits.
inline engine::MemoryGovernor *newAmpleGovernor() {
  static metal::MetalBackend backend("");
  static std::deque<engine::MemoryGovernor> governors;
  constexpr uint64_t ample = 1ULL << 50;
  engine::MemoryGovernor &governor = governors.emplace_back(
      backend, ample, 1, 0, [] { return std::optional<engine::HostMemory>({ample, ample}); }, 0);
  static_cast<void>(governor.evaluate());
  return &governor;
}
inline const bool kAmpleGovernorSet = (ampleGovernor = &newAmpleGovernor, true);

} // namespace splash::test

// CPU graph construction, without a Metal device or model: 650 dispatches
// with eight empty buffer bindings and parameter sizes around the inline
// boundary. Build the same source against each revision, run in ABBA order,
// and compare each fixture separately. This is not a serving benchmark.
#include "metal/CommandGraph.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

namespace {

struct AllocationCounts {
  bool enabled = false;
  uint64_t calls = 0;
  uint64_t requestedBytes = 0;
};
thread_local AllocationCounts allocations;

void *allocate(size_t bytes) {
  void *memory = std::malloc(bytes ? bytes : 1);
  if (!memory) throw std::bad_alloc();
  if (allocations.enabled) {
    ++allocations.calls;
    allocations.requestedBytes += bytes;
  }
  return memory;
}

using splash::metal::CommandGraph;
using splash::metal::MetalBuffer;
constexpr uint32_t kDispatches = 650, kWarmups = 20, kSamples = 80;

template <size_t Words>
void append(CommandGraph &graph, const MetalBuffer &buffer, uint32_t seed) {
  std::array<uint32_t, Words> params;
  for (size_t index = 0; index < Words; ++index) params[index] = seed + index;
  graph.add("graph_benchmark_pipeline", {buffer, buffer, buffer, buffer, buffer, buffer, buffer, buffer},
            params, {32, 1, 1}, {64, 1, 1});
}

template <size_t Words>
void build(CommandGraph &graph, const MetalBuffer &buffer) {
  for (uint32_t index = 0; index < kDispatches; ++index) {
    if constexpr (Words != 0) {
      append<Words>(graph, buffer, index);
    } else {
      switch (index % 4) {
      case 0: append<3>(graph, buffer, index); break;
      case 1: append<8>(graph, buffer, index); break;
      case 2: append<32>(graph, buffer, index); break;
      case 3: append<55>(graph, buffer, index); break;
      }
    }
  }
}

uint64_t verify(const CommandGraph &graph) {
  uint64_t checksum = 0;
  const auto dispatches = graph.dispatches();
  if (dispatches.size() != kDispatches) throw std::runtime_error("missing dispatches");
  for (size_t index = 0; index < dispatches.size(); ++index) {
    const auto &dispatch = dispatches[index];
    if (dispatch.buffers.size() != 8 || dispatch.bytes.size() != 1 || dispatch.bytes[0].index != 8)
      throw std::runtime_error("incorrect bindings");
    const auto &bytes = dispatch.bytes[0];
    for (size_t word = 0; word < bytes.sizeBytes / sizeof(uint32_t); ++word) {
      uint32_t value;
      std::memcpy(&value, static_cast<const std::byte *>(bytes.data) + word * sizeof(value), sizeof(value));
      if (value != index + word) throw std::runtime_error("incorrect parameter bytes");
      checksum += value;
    }
  }
  return checksum;
}

template <size_t Words>
void measure(const char *fixture, const MetalBuffer &buffer) {
  std::array<double, kSamples> samples;
  uint64_t checksum = 0;
  for (uint32_t sample = 0; sample < kWarmups + kSamples; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    CommandGraph graph;
    build<Words>(graph, buffer);
    const auto stop = std::chrono::steady_clock::now();
    if (sample >= kWarmups)
      samples[sample - kWarmups] = std::chrono::duration<double, std::micro>(stop - start).count();
    checksum += verify(graph);
  }
  // Allocation instrumentation is disabled for the timing samples. Count
  // requested bytes, including transient allocations, not RSS or peak use.
  allocations = {true, 0, 0};
  {
    CommandGraph graph;
    build<Words>(graph, buffer);
    checksum += verify(graph);
  }
  allocations.enabled = false;
  std::sort(samples.begin(), samples.end());
  const double median = (samples[kSamples / 2 - 1] + samples[kSamples / 2]) / 2;
  std::printf("fixture=%s dispatches=%u median_build_us=%.3f allocations=%llu requested_bytes=%llu checksum=%llu\n",
              fixture, kDispatches, median, static_cast<unsigned long long>(allocations.calls),
              static_cast<unsigned long long>(allocations.requestedBytes), static_cast<unsigned long long>(checksum));
}

} // namespace

void *operator new(std::size_t bytes) { return allocate(bytes); }
void *operator new[](std::size_t bytes) { return allocate(bytes); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }

int main() {
  try {
    const MetalBuffer buffer;
    measure<3>("12-byte", buffer);
    measure<8>("32-byte", buffer);
    measure<32>("128-byte", buffer);
    measure<33>("132-byte", buffer);
    measure<129>("516-byte", buffer);
    measure<0>("mixed-12-32-128-220", buffer);
  } catch (const std::exception &error) {
    allocations.enabled = false;
    std::fprintf(stderr, "command-graph-benchmark: %s\n", error.what());
    return EXIT_FAILURE;
  }
}

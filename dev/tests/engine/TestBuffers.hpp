#pragma once

#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::test {

// A shared, unlabeled buffer of `bytes` bytes, the kind tests allocate.
inline metal::MetalBuffer sharedBuffer(metal::MetalBackend &backend, uint64_t bytes) {
  return backend.allocateBuffer(bytes, metal::BufferStorage::Shared, {});
}

// An operator's check of one buffer (ops/BufferExtent.hpp): encode(graph,
// view) must encode with the buffer cut to `bytes`, the extent its kernels
// reach, and refuse it one `element`-byte element shorter with the buffer's
// message, "<what> buffer holds", before encoding anything. `buffer` holds
// at least `bytes`.
template <class Encode>
void requireExtent(metal::MetalBackend &backend, const metal::MetalBuffer &buffer, uint64_t bytes,
                   uint64_t element, std::string_view what, Encode encode) {
  const std::string name(what);
  require(bytes >= element && bytes % element == 0 && buffer.sizeBytes() >= bytes,
          name + ": the extent is not whole elements of the buffer");
  metal::CommandGraph graph;
  encode(graph, backend.view(buffer, 0, bytes));
  require(!graph.empty(), name + ": a buffer at the extent encoded nothing");
  const metal::MetalBuffer shorter = bytes > element ? backend.view(buffer, 0, bytes - element) : metal::MetalBuffer{};
  metal::CommandGraph refused;
  rejects([&] { encode(refused, shorter); }, name + " buffer holds",
          "a " + name + " buffer one element short was accepted");
  require(refused.empty(), "a " + name + " buffer one element short encoded a dispatch");
}

// The extent of buffer `index` of one call: `bytes` of `element`-byte
// elements, which the operator's message names `what`.
struct BufferExtent final {
  size_t index;
  uint64_t bytes;
  uint64_t element;
  const char *what;
};

// requireExtent for each buffer of one call, every buffer allocated at its
// extent: encode(graph, buffers) reads them by index.
template <class Encode>
void requireExtents(metal::MetalBackend &backend, std::span<const BufferExtent> extents, Encode encode) {
  std::vector<metal::MetalBuffer> buffers;
  for (const BufferExtent &extent : extents) {
    if (buffers.size() <= extent.index) buffers.resize(extent.index + 1);
    buffers[extent.index] = sharedBuffer(backend, extent.bytes);
  }
  for (const BufferExtent &extent : extents)
    requireExtent(backend, buffers[extent.index], extent.bytes, extent.element, extent.what,
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    std::vector<metal::MetalBuffer> changed = buffers;
                    changed[extent.index] = view;
                    encode(graph, changed);
                  });
}

} // namespace splash::test

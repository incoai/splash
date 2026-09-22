// Native primitive qualification against an independent scalar reference.
// This executable never loads a checkpoint and only submits synthetic inputs.
#include "flash/FlashMoE.hpp"
#include "metal/abi/FlashMoE.h"
#include "../flash/FlashMoEReference.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace splash::flash;
namespace ref = splash::flash::reference;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid Flash MoE graph request accepted");
}

template <class T>
MetalBuffer copyBuffer(MetalBackend &backend, const std::vector<T> &source,
                        const char *label) {
  auto buffer = backend.allocateBuffer(source.size() * sizeof(T),
                                       BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), source.data(), source.size() * sizeof(T));
  return buffer;
}

MetalBuffer bf16Buffer(MetalBackend &backend, uint64_t elements,
                        const char *label) {
  auto buffer = backend.allocateBuffer(elements * sizeof(uint16_t),
                                       BufferStorage::Shared, label);
  std::fill_n(static_cast<uint16_t *>(buffer.contents()), elements, uint16_t{0x7fc1});
  return buffer;
}

struct Comparisons {
  uint64_t elements = 0;
  uint64_t bitExact = 0;
  uint64_t withinOneUlp = 0;
};

void compare(const MetalBuffer &got, std::span<const uint16_t> expected,
             const char *label, Comparisons &count) {
  const auto *actual = static_cast<const uint16_t *>(got.contents());
  for (uint64_t index = 0; index < expected.size(); ++index) {
    ++count.elements;
    if (actual[index] == expected[index]) {
      ++count.bitExact;
      continue;
    }
    const float observed = ref::fromBF16(actual[index]);
    const float wanted = ref::fromBF16(expected[index]);
    int exponent = 0;
    std::frexp(std::fabs(wanted), &exponent);
    const float ulp = std::max(std::ldexp(1.0f, exponent - 8),
                               std::ldexp(1.0f, -133));
    require(std::isfinite(observed) && std::fabs(observed - wanted) <= ulp,
            std::string(label) + " exceeds one BF16 ULP at " +
                std::to_string(index) + " got=" + std::to_string(observed) +
                " expected=" + std::to_string(wanted));
    ++count.withinOneUlp;
  }
}

void fixture(MetalBackend &backend, uint32_t rows, uint32_t width,
              uint32_t experts, uint32_t selections, bool equalLogits,
              bool normalizeTopK, Comparisons &count) {
  std::vector<uint16_t> logits(uint64_t{rows} * experts);
  std::vector<uint16_t> gate(uint64_t{rows} * selections * width);
  std::vector<uint16_t> up(gate.size()), down(gate.size());
  std::vector<uint16_t> shared(uint64_t{rows} * width), sharedGate(rows);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t expert = 0; expert < experts; ++expert) {
      const int value = int((uint64_t{expert} * 73 + row * 19) % 513) - 256;
      logits[uint64_t{row} * experts + expert] =
          ref::toBF16(equalLogits ? 0.0f : float(value) / 16.0f);
    }
    // Explicitly select the highest legal expert to exercise ID 511.
    if (!equalLogits) logits[uint64_t{row} * experts + experts - 1] = ref::toBF16(20.0f);
    sharedGate[row] = ref::toBF16(float(int(row % 257) - 128) / 8.0f);
  }
  for (uint64_t index = 0; index < gate.size(); ++index) {
    gate[index] = ref::toBF16(float(int(index % 257) - 128) / 8.0f);
    up[index] = ref::toBF16(float(int((index * 7) % 129) - 64) / 16.0f);
    down[index] = ref::toBF16(float(int((index * 37) % 257) - 128) / 32.0f);
  }
  for (uint64_t index = 0; index < shared.size(); ++index)
    shared[index] = ref::toBF16(float(int((index * 11) % 129) - 64) / 32.0f);

  const auto routed = ref::route(logits, rows, experts, selections, normalizeTopK);
  const auto activated = ref::swiglu(gate, up, rows, width, selections);
  const auto combined = ref::combine(down, routed.indices, routed.scores, shared,
                                     sharedGate, rows, width, experts, selections);
  auto logitBuffer = copyBuffer(backend, logits, "flash fixture logits");
  auto gateBuffer = copyBuffer(backend, gate, "flash fixture gate");
  auto upBuffer = copyBuffer(backend, up, "flash fixture up");
  auto downBuffer = copyBuffer(backend, down, "flash fixture down");
  auto sharedBuffer = copyBuffer(backend, shared, "flash fixture shared");
  auto sharedGateBuffer = copyBuffer(backend, sharedGate, "flash fixture shared gate");
  auto ids = backend.allocateBuffer(uint64_t{rows} * selections * sizeof(int64_t),
                                    BufferStorage::Shared, "flash fixture IDs");
  auto weights = bf16Buffer(backend, uint64_t{rows} * selections, "flash fixture weights");
  auto activation = bf16Buffer(backend, gate.size(), "flash fixture activation");
  auto output = bf16Buffer(backend, shared.size(), "flash fixture combined");
  auto diagnostic = copyBuffer(backend, std::vector<uint32_t>{0x80}, "flash fixture diagnostic");
  CommandGraph graph;
  addRoute(graph, logitBuffer, ids, weights, diagnostic, rows, experts, selections,
            normalizeTopK);
  addSiLUMultiply(graph, gateBuffer, upBuffer, activation, diagnostic, rows, width,
                   selections);
  addCombine(graph, downBuffer, ids, weights, sharedBuffer, sharedGateBuffer,
              output, diagnostic, rows, width, experts, selections);
  const auto timing = backend.submitCommand(graph.dispatches());
  require(*static_cast<uint32_t *>(diagnostic.contents()) == 0x80,
          "valid MoE command changed caller-owned sticky diagnostics");
  const auto *gotIDs = static_cast<const int64_t *>(ids.contents());
  for (uint64_t index = 0; index < routed.indices.size(); ++index)
    require(gotIDs[index] == int64_t(routed.indices[index]),
            "router selected incorrect expert at " + std::to_string(index));
  compare(weights, routed.scores, "route weights", count);
  compare(activation, activated, "activation", count);
  compare(output, combined, "combine", count);
  std::cout << "fixture rows=" << rows << " width=" << width
            << " experts=" << experts << " selections=" << selections
            << " equal-logits=" << equalLogits << " normalized=" << normalizeTopK
            << " gpu-ms=" << timing.gpuSeconds * 1000.0 << '\n';
}

void invalidFixtures(MetalBackend &backend) {
  auto logits = copyBuffer(backend, std::vector<uint16_t>(512, ref::toBF16(0.0f)), "bad logits");
  auto ids = copyBuffer(backend, std::vector<int64_t>(10, 0), "bad IDs");
  auto weights = bf16Buffer(backend, 10, "bad weights");
  auto diagnostics = copyBuffer(backend, std::vector<uint32_t>{0x80}, "sticky bad diagnostics");
  auto source = copyBuffer(backend, std::vector<uint16_t>(10 * 17, ref::toBF16(1.0f)), "bad expert data");
  auto shared = copyBuffer(backend, std::vector<uint16_t>(17, ref::toBF16(1.0f)), "bad shared data");
  auto gate = copyBuffer(backend, std::vector<uint16_t>{ref::toBF16(0.0f)}, "bad shared gate");
  auto out = bf16Buffer(backend, 17, "bad combined output");
  static_cast<uint16_t *>(logits.contents())[511] = 0x7fc0;
  CommandGraph badRoute;
  addRoute(badRoute, logits, ids, weights, diagnostics, 1, 512, 10);
  (void)backend.submitCommand(badRoute.dispatches());
  require(*static_cast<uint32_t *>(diagnostics.contents()) == 0x84,
          "nonfinite router did not OR numeric diagnostic");
  for (uint32_t index = 0; index < 10; ++index) {
    require(static_cast<int64_t *>(ids.contents())[index] == -1,
            "nonfinite route did not poison all IDs");
    require(std::isnan(ref::fromBF16(static_cast<uint16_t *>(weights.contents())[index])),
            "nonfinite route did not poison all scores");
  }
  static_cast<uint16_t *>(logits.contents())[511] = ref::toBF16(0.0f);
  CommandGraph goodAfterBad;
  addRoute(goodAfterBad, logits, ids, weights, diagnostics, 1, 512, 10);
  (void)backend.submitCommand(goodAfterBad.dispatches());
  require(*static_cast<uint32_t *>(diagnostics.contents()) == 0x84,
          "good route cleared prior sticky diagnostics");
  static_cast<int64_t *>(ids.contents())[9] = 512;
  CommandGraph badCombine;
  addCombine(badCombine, source, ids, weights, shared, gate, out, diagnostics,
              1, 17, 512, 10);
  (void)backend.submitCommand(badCombine.dispatches());
  require(*static_cast<uint32_t *>(diagnostics.contents()) == 0x85,
          "out-of-range ID did not OR ID diagnostic");
  for (uint32_t index = 0; index < 17; ++index)
    require(std::isnan(ref::fromBF16(static_cast<uint16_t *>(out.contents())[index])),
            "out-of-range ID did not poison combine row");
  CommandGraph malformed;
  malformed.add("flash_moe_route", {logits, ids, weights, diagnostics},
                FlashMoERouteParams{1, 513, 10, 1}, {1, 1, 1}, {256, 1, 1});
  (void)backend.submitCommand(malformed.dispatches());
  require(*static_cast<uint32_t *>(diagnostics.contents()) == 0x87,
          "malformed shader shape did not OR parameter diagnostic");
  auto activated = bf16Buffer(backend, 10 * 17, "bad activation output");
  *static_cast<uint32_t *>(diagnostics.contents()) = 0x80;
  static_cast<uint16_t *>(source.contents())[169] = 0x7fc0;
  CommandGraph badActivation;
  addSiLUMultiply(badActivation, source, source, activated, diagnostics, 1, 17, 10);
  (void)backend.submitCommand(badActivation.dispatches());
  require(*static_cast<uint32_t *>(diagnostics.contents()) == 0x84 &&
              std::isnan(ref::fromBF16(static_cast<uint16_t *>(activated.contents())[169])),
          "nonfinite activation did not poison output and OR numeric diagnostic");
  static_cast<uint16_t *>(source.contents())[169] = ref::toBF16(1.0f);
  for (uint32_t slot = 0; slot < 10; ++slot)
    static_cast<int64_t *>(ids.contents())[slot] = slot;
  *static_cast<uint32_t *>(diagnostics.contents()) = 0x80;
  static_cast<int64_t *>(ids.contents())[9] = 8;
  CommandGraph duplicateCombine;
  addCombine(duplicateCombine, source, ids, weights, shared, gate, out, diagnostics,
              1, 17, 512, 10);
  (void)backend.submitCommand(duplicateCombine.dispatches());
  require(*static_cast<uint32_t *>(diagnostics.contents()) == 0x81,
          "duplicate expert ID did not OR ID diagnostic");
  static_cast<int64_t *>(ids.contents())[9] = 9;
  *static_cast<uint32_t *>(diagnostics.contents()) = 0x80;
  static_cast<uint16_t *>(weights.contents())[9] = ref::toBF16(1.125f);
  CommandGraph invalidWeight;
  addCombine(invalidWeight, source, ids, weights, shared, gate, out, diagnostics,
              1, 17, 512, 10);
  (void)backend.submitCommand(invalidWeight.dispatches());
  require(*static_cast<uint32_t *>(diagnostics.contents()) == 0x84,
          "invalid route score did not OR numeric diagnostic");
  rejects([&] { CommandGraph g; addRoute(g, logits, ids, weights, diagnostics, 1, 513, 10); });
  rejects([&] { CommandGraph g; addRoute(g, logits, ids, weights, diagnostics, 1, 512, 11); });
  rejects([&] { CommandGraph g; addRoute(g, logits, ids, weights, diagnostics, 2049, 512, 10); });
  rejects([&] { CommandGraph g; addRoute(g, logits, ids, weights, diagnostics, 1, 512, 0); });
  rejects([&] { CommandGraph g; addRoute(g, logits, ids, logits, diagnostics, 1, 512, 10); });
  rejects([&] { CommandGraph g; addSiLUMultiply(g, source, source, out, diagnostics, 1, 2561, 10); });
  rejects([&] { CommandGraph g; addCombine(g, source, ids, weights, shared, gate, shared, diagnostics, 1, 17, 512, 10); });
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2, "usage: flash-moe-metal-test /absolute/private/splash.metallib");
      MetalBackend backend(argv[1]);
      Comparisons count;
      fixture(backend, 1, 640, 512, 10, false, true, count);
      fixture(backend, 4, 2560, 512, 10, false, true, count);
      fixture(backend, 8, 640, 512, 10, true, true, count);
      fixture(backend, 224, 640, 512, 10, false, true, count);
      fixture(backend, 2048, 17, 512, 10, false, true, count);
      fixture(backend, 1, 17, 1, 1, false, true, count);
      fixture(backend, 4, 17, 512, 10, false, false, count);
      invalidFixtures(backend);
      std::cout << "PASS Flash MoE primitives elements=" << count.elements
                << " bit-exact=" << count.bitExact
                << " within-one-ulp=" << count.withinOneUlp << '\n';
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "FAIL " << error.what() << '\n';
      return 1;
    }
  }
}
